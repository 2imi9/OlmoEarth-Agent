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

# Same policy as tools/system.py: drop the agent's own credentials from the
# child env — but ASTA_* must pass through (the CLI authenticates with it).
_SECRET_FRAGMENTS = (
    "API_KEY",
    "APIKEY",
    "SECRET",
    "TOKEN",
    "PASSWORD",
    "PASSWD",
    "CREDENTIAL",
)
_SECRET_PREFIXES = (
    "OLMOEARTH_",
    "LLM_",
    "AWS_",
    "ANTHROPIC_",
    "OPENAI_",
    "GEMINI_",
    "HF_",
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
    """Child env minus the agent's own credentials; ``ASTA_*`` passes through."""
    env: dict[str, str] = {}
    for name, value in os.environ.items():
        upper = name.upper()
        if upper.startswith("ASTA"):
            env[name] = value
            continue
        if upper.startswith(_SECRET_PREFIXES):
            continue
        if any(fragment in upper for fragment in _SECRET_FRAGMENTS):
            continue
        env[name] = value
    return env


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


def parse_asta_results(
    payload: dict[str, Any], *, max_results: int, include_abstract: bool
) -> list[dict[str, Any]]:
    """Map Asta's ``LiteratureSearchResult`` JSON to curated litsearch records.

    Keeps the shared record shape (id/title/authors/year/venue/url/...) so
    asta results read like the other sources, and adds the asta-only signals:
    ``relevance_score``, ``relevance_summary`` (the AI judgement), and capped
    full-text ``snippets``.
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
        judgement = paper.get("relevanceJudgement") or {}
        snippets = [
            _truncate(s.get("text"), _SNIPPET_CHARS)
            for s in (paper.get("snippets") or [])[:_SNIPPETS_PER_PAPER]
            if isinstance(s, dict) and s.get("text")
        ]
        corpus_id = paper.get("corpusId")
        records.append(
            {
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
                    if include_abstract
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
        )
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
    follow-up analysis can reread it without re-searching).

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
    return {
        "count": len(papers),
        "papers": papers,
        "sources": ["asta"],
        "warnings": [],
        "saved_to": str(out_path),
    }
