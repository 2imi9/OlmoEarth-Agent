# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Optional Asta backend for litsearch: full-text ranked literature search.

`Asta <https://github.com/allenai/asta-plugins>`_ is Ai2's scientific-research
toolkit; its ``asta literature find`` command runs a retrieval pipeline over
the *full text* of open-access publications and returns relevance-ranked
papers with AI relevance judgements and supporting snippets — a real upgrade
over our metadata-only arXiv/OpenAlex search when grounding EO claims.

Integration follows the Shippy deterministic-CLI pattern: the agent never
talks to Ai2's hosted APIs itself; it invokes the ``asta`` CLI (argv, no
shell) which handles auth, retries, and — conveniently — writes its result to
a local JSON file (``-o``), which we keep under the confined workspace root.

Everything is detection-gated and optional: no CLI on PATH means the backend
reports itself unavailable with install guidance, and the arXiv/OpenAlex
sources keep working untouched. Auth is the CLI's own (``asta auth login`` /
``ASTA_TOKEN``); a failure surfaces the CLI's message plus the login hint.

The subprocess env is credential-scrubbed like ``tools/system.py`` (the
agent's Studio/LLM/cloud keys are dropped) but ``ASTA_*`` variables pass
through — the CLI needs its own token. Note the CLI makes its own network
calls, so the in-process egress guard does not cover it; the trust boundary
is "the operator chose to install and authenticate the Asta CLI".

Asta marks some papers' text as not legal to show (``legalToShow: false`` in
its ``Paper`` model; checked against asta-plugins 0.120.0). Such a record
keeps its metadata and scores; its abstract, snippets and relevance summary
are dropped, and ``content_withheld`` says why. The CLI writes that text to
its file whatever the flag says, so the file is rewritten without it before
its path is returned.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import uuid
from pathlib import Path
from typing import Any

from olmoearth_agent.security.child_env import scrubbed_env
from olmoearth_agent.security.paths import workspace_root

#: Env var overriding the CLI binary (absolute path or name on PATH).
ASTA_BIN_ENV = "OLMOEARTH_ASTA_BIN"
#: Env var overriding the per-search wall-clock cap, seconds.
ASTA_TIMEOUT_ENV = "OLMOEARTH_ASTA_TIMEOUT"
#: Default cap: the pipeline typically takes 30-60s; leave headroom.
DEFAULT_TIMEOUT_S = 180.0

#: Caps keeping one search's records context-friendly.
MAX_RECORDS = 25
_SNIPPETS_PER_PAPER = 3
_SNIPPET_CHARS = 400
_SUMMARY_CHARS = 500
_ABSTRACT_CHARS = 1200
_AUTHOR_CAP = 10
_STDERR_TAIL = 600

_SLUG_RE = re.compile(r"[^a-z0-9]+")

#: A record's ``content_withheld`` when Asta's ``legalToShow`` is false.
CONTENT_WITHHELD = "Asta marks this paper's text as not legal to show"
#: The flag's name as the CLI writes it, and the name Asta's model accepts.
_LEGAL_KEYS = ("legalToShow", "legal_to_show")
#: A ``Paper``'s fields that hold the paper's text, as the CLI writes them and
#: as Asta's model also accepts them. Each has a default in that model, so the
#: CLI's file stays valid without them.
_TEXT_KEYS = frozenset(
    {
        "abstract",
        "snippets",
        "relevanceJudgement",
        "relevance_judgement",
        "citationContexts",
        "citation_contexts",
    }
)
#: The search's warning when the CLI's file could not be rewritten without the
#: withheld text (the result then has no ``saved_to``).
ARTIFACT_NOT_REWRITTEN = (
    "The CLI's file could not be rewritten without the text Asta marks not "
    "legal to show, so its path is not returned."
)


class AstaUnavailableError(RuntimeError):
    """Raised when the Asta CLI is not installed / not on PATH."""


class AstaSearchError(RuntimeError):
    """Raised when the CLI runs but fails (auth, network, timeout)."""


def asta_bin() -> str | None:
    """Resolved path of the Asta CLI, or ``None`` when not installed."""
    configured = os.environ.get(ASTA_BIN_ENV, "").strip() or "asta"
    return shutil.which(configured)


def asta_available() -> bool:
    """True when the Asta CLI is on PATH (auth is checked at call time)."""
    return asta_bin() is not None


def _timeout_s() -> float:
    """Per-search wall-clock cap (env-overridable)."""
    try:
        return float(os.environ.get(ASTA_TIMEOUT_ENV, DEFAULT_TIMEOUT_S))
    except (TypeError, ValueError):
        return DEFAULT_TIMEOUT_S


def _scrubbed_env() -> dict[str, str]:
    """Child env minus the agent's own credentials; ``ASTA_*`` passes through.

    The same denylist as every child the agent starts
    (:mod:`olmoearth_agent.security.child_env`): the CLI authenticates with
    its own ``ASTA_*`` variables, which pass whatever they hold.
    """
    return scrubbed_env(keep_prefixes=("ASTA",))


async def _run_cli(argv: list[str]) -> tuple[int, str, str]:
    """Run the CLI (argv, no shell) under the timeout; returns (rc, out, err).

    Isolated so tests monkeypatch it instead of spawning processes.
    """
    proc = await asyncio.create_subprocess_exec(
        *argv,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=_scrubbed_env(),
    )
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=_timeout_s())
    except asyncio.TimeoutError:
        proc.kill()
        await proc.communicate()
        raise AstaSearchError(
            f"asta literature find timed out after {_timeout_s():.0f}s; "
            "retry with a narrower query or raise "
            f"{ASTA_TIMEOUT_ENV}."
        ) from None
    return (
        proc.returncode or 0,
        out.decode("utf-8", "replace"),
        err.decode("utf-8", "replace"),
    )


def _truncate(text: Any, limit: int) -> str | None:
    """One-line, length-capped rendering of a text field (None-safe)."""
    if not text or not isinstance(text, str):
        return None
    cleaned: str = " ".join(text.split())
    return cleaned if len(cleaned) <= limit else cleaned[:limit].rstrip() + "…"


def _legal_to_show(paper: dict[str, Any]) -> bool:
    """False when Asta marks the paper's text as not legal to show.

    The CLI writes ``legalToShow`` (Asta's default is true); ``legal_to_show``
    is honoured too. Only an explicit ``false`` withholds the text.
    """
    return not any(paper.get(key) is False for key in _LEGAL_KEYS)


def _without_text(paper: Any) -> Any:
    """``paper`` minus its text fields when Asta marks it not legal to show."""
    if isinstance(paper, dict) and not _legal_to_show(paper):
        return {key: value for key, value in paper.items() if key not in _TEXT_KEYS}
    return paper


def _withhold_in_artifact(path: Path, payload: dict[str, Any]) -> bool:
    """Rewrite the CLI's file without the text of papers Asta may not show.

    The CLI writes every ``Paper`` field whatever ``legalToShow`` says. Each
    paper whose flag is false loses its abstract, snippets, relevance
    judgement and citation contexts; every other field and paper is kept. The
    file is written beside itself and moved over it (``os.replace`` within
    one folder), so it is never left half-written, and a failed write leaves
    no file beside it. A file with no such text is not touched.

    Returns
    -------
    bool
        True when the file holds no withheld text. False when the rewrite
        failed: the file is then as the CLI wrote it.
    """
    results = payload.get("results")
    if not isinstance(results, list):
        return True
    kept = [_without_text(paper) for paper in results]
    if kept == results:
        return True
    staged = path.with_name(path.name + ".tmp")
    text = json.dumps({**payload, "results": kept}, indent=2)
    try:
        staged.write_text(text, encoding="utf-8")
        os.replace(staged, path)
    except OSError:
        staged.unlink(missing_ok=True)
        return False
    return True


def parse_asta_results(
    payload: dict[str, Any], *, max_results: int, include_abstract: bool
) -> list[dict[str, Any]]:
    """Map Asta's ``LiteratureSearchResult`` JSON to curated litsearch records.

    Keeps the shared record shape (id/title/authors/year/venue/url/...) so
    asta results read like the other sources, and adds the asta-only signals:
    ``relevance_score``, ``relevance_summary`` (the AI judgement), and capped
    full-text ``snippets``.

    A paper whose ``legalToShow`` (or ``legal_to_show``) is false keeps its
    title, authors, year, venue, url, id and scores; its abstract, snippets
    and relevance summary are dropped, and its record carries
    ``content_withheld`` (:data:`CONTENT_WITHHELD`). Missing or true keeps the
    text, and the record has no ``content_withheld``.
    """
    results = payload.get("results")
    if not isinstance(results, list):
        return []
    ranked = sorted(
        (p for p in results if isinstance(p, dict)),
        key=lambda p: -(p.get("relevanceScore") or 0.0),
    )
    records: list[dict[str, Any]] = []
    for paper in ranked[: max(1, min(int(max_results), MAX_RECORDS))]:
        shown = _legal_to_show(paper)
        judgement: dict[str, Any] = (
            (paper.get("relevanceJudgement") or {}) if shown else {}
        )
        snippets = (
            [
                _truncate(s.get("text"), _SNIPPET_CHARS)
                for s in (paper.get("snippets") or [])[:_SNIPPETS_PER_PAPER]
                if isinstance(s, dict) and s.get("text")
            ]
            if shown
            else []
        )
        corpus_id = paper.get("corpusId")
        record: dict[str, Any] = {
            "id": f"CorpusId:{corpus_id}" if corpus_id is not None else None,
            "title": _truncate(paper.get("title"), 300),
            "authors": [
                a.get("name")
                for a in (paper.get("authors") or [])[:_AUTHOR_CAP]
                if isinstance(a, dict) and a.get("name")
            ],
            "year": paper.get("year"),
            "venue": paper.get("venue") or None,
            "doi": None,
            "arxiv_id": None,
            "url": paper.get("url")
            or (
                f"https://www.semanticscholar.org/p/{corpus_id}"
                if corpus_id is not None
                else None
            ),
            "abstract": (
                _truncate(paper.get("abstract"), _ABSTRACT_CHARS)
                if include_abstract and shown
                else None
            ),
            "cited_by_count": paper.get("citationCount"),
            "pdf_url": None,
            "source": "asta",
            "relevance_score": paper.get("relevanceScore"),
            "relevance_summary": _truncate(
                judgement.get("relevanceSummary"), _SUMMARY_CHARS
            ),
            "snippets": snippets,
        }
        if not shown:
            record["content_withheld"] = CONTENT_WITHHELD
        records.append(record)
    return records


async def search_asta(
    query: str,
    *,
    max_results: int = 10,
    include_abstract: bool = False,
) -> dict[str, Any]:
    """Full-text ranked literature search via ``asta literature find``.

    Returns ``{"count", "papers", "sources": ["asta"], "warnings",
    "saved_to"}`` — the same envelope as :func:`search_literature`, plus the
    path of the CLI's full JSON artifact (kept under the workspace root, so
    follow-up analysis can reread it without re-searching). ``papers`` drops
    the text of a paper Asta marks not legal to show
    (:func:`parse_asta_results`), and so does the artifact: when it holds
    such text, it is rewritten without that paper's abstract, snippets,
    relevance judgement and citation contexts before its path is returned.
    If that rewrite fails, the result has no ``saved_to`` and ``warnings``
    holds :data:`ARTIFACT_NOT_REWRITTEN`.

    Raises
    ------
    AstaUnavailableError
        When the CLI is not installed (callers surface install guidance).
    AstaSearchError
        When the CLI fails or times out (message includes the auth hint).
    """
    query = (query or "").strip()
    if not query:
        raise ValueError("query is required and must be non-empty.")
    binary = asta_bin()
    if binary is None:
        raise AstaUnavailableError(
            "the Asta CLI is not installed (or not on PATH). Install "
            "asta-plugins (github.com/allenai/asta-plugins), run "
            "'asta auth login' once, or set "
            f"{ASTA_BIN_ENV} to the binary path."
        )

    out_dir = workspace_root() / "litsearch"
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = _SLUG_RE.sub("-", query.lower()).strip("-")[:40] or "query"
    out_path = out_dir / f"asta_{slug}_{uuid.uuid4().hex[:8]}.json"

    rc, _stdout, stderr = await _run_cli(
        [
            binary,
            "literature",
            "find",
            query,
            "-o",
            str(out_path),
            "--timeout",
            str(int(_timeout_s())),
        ]
    )
    if rc != 0:
        raise AstaSearchError(
            f"asta literature find exited with code {rc}: "
            f"{stderr.strip()[-_STDERR_TAIL:] or 'no error output'}. "
            "If this is an authentication error, run 'asta auth login' once."
        )
    try:
        payload = json.loads(Path(out_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AstaSearchError(
            f"asta ran but its output file could not be read ({exc})."
        ) from exc

    papers = parse_asta_results(
        payload, max_results=max_results, include_abstract=include_abstract
    )
    result: dict[str, Any] = {
        "count": len(papers),
        "papers": papers,
        "sources": ["asta"],
        "warnings": [],
    }
    if _withhold_in_artifact(out_path, payload):
        result["saved_to"] = str(out_path)
    else:
        result["warnings"].append(ARTIFACT_NOT_REWRITTEN)
    return result
