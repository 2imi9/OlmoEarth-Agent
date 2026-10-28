# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Unit tests for the optional Asta litsearch backend (analysis/asta.py)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from olmoearth_agent.analysis import asta
from olmoearth_agent.security.paths import OUTPUT_ROOT_ENV

_PAYLOAD: dict[str, Any] = {
    "query": "cloud masking sentinel-2",
    "results": [
        {
            "corpusId": 111,
            "title": "Low-relevance paper",
            "year": 2020,
            "authors": [{"name": "A. Author", "id": "1"}],
            "venue": "RSE",
            "url": "https://www.semanticscholar.org/p/111",
            "citationCount": 5,
            "relevanceScore": 0.31,
            "relevanceJudgement": {"relevanceSummary": "tangential"},
            "snippets": [],
        },
        {
            "corpusId": 222,
            "title": "Highly relevant cloud-mask study",
            "abstract": "We evaluate cloud masking...",
            "year": 2024,
            "authors": [
                {"name": "B. Author", "id": "2"},
                {"name": "C. Author", "id": "3"},
            ],
            "venue": "TGRS",
            "url": None,
            "citationCount": 42,
            "relevanceScore": 0.97,
            "relevanceJudgement": {"relevanceSummary": "directly on topic"},
            "snippets": [
                {
                    "text": "Our cloud mask improves F1 by 0.12. " + "x" * 500,
                    "sectionTitle": "Results",
                },
                {"text": "second snippet", "sectionTitle": "Methods"},
                {"text": "third snippet", "sectionTitle": "Methods"},
                {"text": "fourth snippet is dropped", "sectionTitle": "Appendix"},
            ],
        },
    ],
}


def _cli_paper(**overrides: Any) -> dict[str, Any]:
    """One paper as ``asta literature find`` writes it (asta-plugins 0.120.0).

    The CLI dumps its ``Paper`` model by field name
    (``model_dump(mode="json")``), so every key is camelCase and
    ``legalToShow`` is always present (default true).
    """
    paper: dict[str, Any] = {
        "corpusId": 333,
        "title": "A paper Asta may not show",
        "abstract": "Full abstract text.",
        "year": 2025,
        "authors": [{"name": "D. Author", "id": "4"}],
        "venue": "RSE",
        "journal": None,
        "url": "https://www.semanticscholar.org/p/333",
        "publicationDate": "2025-01-02",
        "citationCount": 7,
        "categories": [],
        "relevanceScore": 0.88,
        "relevanceJudgement": {
            "relevance": 3,
            "relevanceSummary": "on topic",
            "relevanceCriteriaJudgements": [],
        },
        "snippets": [
            {
                "text": "a full-text passage",
                "sectionTitle": "Results",
                "sentenceIds": [],
            }
        ],
        "citationContexts": [],
        "legalToShow": True,
        "numOfOmittedCitationContextsDueLegal": 0,
    }
    paper.update(overrides)
    return paper


@pytest.fixture(autouse=True)
def _workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv(OUTPUT_ROOT_ENV, str(tmp_path))
    return tmp_path


def test_parse_ranks_caps_and_maps_records() -> None:
    records = asta.parse_asta_results(_PAYLOAD, max_results=10, include_abstract=True)
    assert [r["id"] for r in records] == ["CorpusId:222", "CorpusId:111"]  # ranked
    top = records[0]
    assert top["title"] == "Highly relevant cloud-mask study"
    assert top["authors"] == ["B. Author", "C. Author"]
    assert top["source"] == "asta"
    assert top["cited_by_count"] == 42
    assert top["relevance_summary"] == "directly on topic"
    # Null url falls back to a Semantic Scholar link; snippets capped + truncated.
    assert top["url"] == "https://www.semanticscholar.org/p/222"
    assert len(top["snippets"]) == 3
    assert all(len(s) <= 401 for s in top["snippets"])
    assert top["abstract"].startswith("We evaluate")
    # include_abstract=False strips abstracts.
    lean = asta.parse_asta_results(_PAYLOAD, max_results=1, include_abstract=False)
    assert len(lean) == 1 and lean[0]["abstract"] is None


@pytest.mark.parametrize("flag", ["legalToShow", "legal_to_show"])
def test_parse_withholds_text_asta_marks_not_legal_to_show(flag: str) -> None:
    """legalToShow false (or snake_case legal_to_show): metadata and scores only."""
    paper = _cli_paper()
    del paper["legalToShow"]
    paper[flag] = False
    (record,) = asta.parse_asta_results(
        {"query": "q", "results": [paper]}, max_results=5, include_abstract=True
    )
    assert record["abstract"] is None
    assert record["snippets"] == []
    assert record["relevance_summary"] is None
    assert record["content_withheld"] == asta.CONTENT_WITHHELD
    assert "not legal to show" in record["content_withheld"]
    # Title, authors, year, venue, url, id and scores are kept.
    assert record["id"] == "CorpusId:333"
    assert record["title"] == "A paper Asta may not show"
    assert record["authors"] == ["D. Author"]
    assert record["year"] == 2025
    assert record["venue"] == "RSE"
    assert record["url"] == "https://www.semanticscholar.org/p/333"
    assert record["relevance_score"] == 0.88
    assert record["cited_by_count"] == 7
    assert record["source"] == "asta"


@pytest.mark.parametrize(
    "legal", [{"legalToShow": True}, {}], ids=["legalToShow-true", "missing"]
)
def test_parse_keeps_text_when_legal_to_show_is_true_or_missing(
    legal: dict[str, bool],
) -> None:
    paper = _cli_paper()
    del paper["legalToShow"]
    paper.update(legal)
    (record,) = asta.parse_asta_results(
        {"query": "q", "results": [paper]}, max_results=5, include_abstract=True
    )
    assert record["abstract"] == "Full abstract text."
    assert record["snippets"] == ["a full-text passage"]
    assert record["relevance_summary"] == "on topic"
    assert "content_withheld" not in record


def test_parse_withholds_only_the_paper_marked_not_legal() -> None:
    """One withheld paper leaves the others' text and the ranking alone."""
    payload = {
        "query": "q",
        "results": [
            _cli_paper(corpusId=1, relevanceScore=0.5, legalToShow=False),
            _cli_paper(corpusId=2, relevanceScore=0.9),
        ],
    }
    records = asta.parse_asta_results(payload, max_results=5, include_abstract=True)
    assert [r["id"] for r in records] == ["CorpusId:2", "CorpusId:1"]
    shown, withheld = records
    assert "content_withheld" not in shown and shown["snippets"]
    assert withheld["content_withheld"] == asta.CONTENT_WITHHELD
    assert withheld["snippets"] == [] and withheld["abstract"] is None


def test_parse_tolerates_malformed_payload() -> None:
    assert asta.parse_asta_results({}, max_results=5, include_abstract=False) == []
    assert (
        asta.parse_asta_results(
            {"results": "nope"}, max_results=5, include_abstract=False
        )
        == []
    )


@pytest.mark.asyncio
async def test_search_unavailable_raises_with_guidance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(asta, "asta_bin", lambda: None)
    assert asta.asta_available() is False
    with pytest.raises(asta.AstaUnavailableError, match="asta auth login"):
        await asta.search_asta("cloud masking")


@pytest.mark.asyncio
async def test_search_invokes_cli_and_reads_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, Any] = {}

    async def fake_cli(argv: list[str]) -> tuple[int, str, str]:
        seen["argv"] = argv
        out_path = Path(argv[argv.index("-o") + 1])
        out_path.write_text(json.dumps(_PAYLOAD), encoding="utf-8")
        return 0, "", ""

    monkeypatch.setattr(asta, "asta_bin", lambda: "asta")
    monkeypatch.setattr(asta, "_run_cli", fake_cli)
    result = await asta.search_asta("cloud masking sentinel-2", max_results=1)

    assert seen["argv"][:3] == ["asta", "literature", "find"]
    assert seen["argv"][3] == "cloud masking sentinel-2"
    assert result["count"] == 1
    assert result["papers"][0]["id"] == "CorpusId:222"
    assert result["sources"] == ["asta"]
    # The CLI's full artifact is kept inside the confined workspace.
    saved = Path(result["saved_to"])
    assert saved.is_file()
    assert tmp_path.resolve() in saved.resolve().parents
    # Nothing is withheld, so the file is as the CLI wrote it.
    assert saved.read_text(encoding="utf-8") == json.dumps(_PAYLOAD)


@pytest.mark.asyncio
async def test_search_withholds_text_from_the_cli_artifact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The argv the 0.120.0 CLI takes, and its file read with legalToShow false."""
    seen: dict[str, Any] = {}
    payload = {
        "query": "cloud masking",
        "results": [_cli_paper(legalToShow=False)],
        "thread_id": None,
        "narrative": None,
    }

    async def fake_cli(argv: list[str]) -> tuple[int, str, str]:
        seen["argv"] = argv
        Path(argv[5]).write_text(json.dumps(payload), encoding="utf-8")
        return 0, "", "Results saved to: x"

    monkeypatch.delenv(asta.ASTA_TIMEOUT_ENV, raising=False)
    monkeypatch.setattr(asta, "asta_bin", lambda: "asta")
    monkeypatch.setattr(asta, "_run_cli", fake_cli)
    result = await asta.search_asta(
        "cloud masking", max_results=3, include_abstract=True
    )

    # asta literature find QUERY -o PATH --timeout SECONDS (asta-plugins 0.120.0).
    argv = seen["argv"]
    assert argv[:5] == ["asta", "literature", "find", "cloud masking", "-o"]
    assert argv[6:] == ["--timeout", str(int(asta.DEFAULT_TIMEOUT_S))]
    (paper,) = result["papers"]
    assert paper["content_withheld"] == asta.CONTENT_WITHHELD
    assert paper["abstract"] is None and paper["snippets"] == []
    assert paper["relevance_summary"] is None
    assert paper["title"] == "A paper Asta may not show"
    # The CLI's file, whose path the answer carries, holds none of that text.
    saved = Path(result["saved_to"])
    assert saved == Path(argv[5])
    text = saved.read_text(encoding="utf-8")
    for withheld in ("Full abstract text.", "a full-text passage", "on topic"):
        assert withheld not in text
    (kept,) = json.loads(text)["results"]
    assert kept["title"] == "A paper Asta may not show"
    assert kept["legalToShow"] is False


@pytest.mark.asyncio
async def test_search_rewrites_only_the_withheld_papers_text_in_the_cli_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The file keeps every other field and paper; only withheld text goes."""
    context = {"text": "a citing passage", "sourceCorpusId": 9}
    withheld = _cli_paper(corpusId=1, legalToShow=False, citationContexts=[context])
    shown = _cli_paper(corpusId=2, relevanceScore=0.5)
    payload = {
        "query": "q",
        "results": [withheld, shown],
        "thread_id": "t-1",
        "narrative": None,
    }

    async def fake_cli(argv: list[str]) -> tuple[int, str, str]:
        Path(argv[argv.index("-o") + 1]).write_text(json.dumps(payload), "utf-8")
        return 0, "", ""

    monkeypatch.setattr(asta, "asta_bin", lambda: "asta")
    monkeypatch.setattr(asta, "_run_cli", fake_cli)
    result = await asta.search_asta("q", max_results=5)

    saved = Path(result["saved_to"])
    written = json.loads(saved.read_text(encoding="utf-8"))
    text_keys = {"abstract", "snippets", "relevanceJudgement", "citationContexts"}
    assert written["results"] == [
        {k: v for k, v in withheld.items() if k not in text_keys},
        shown,
    ]
    assert {k: v for k, v in written.items() if k != "results"} == {
        "query": "q",
        "thread_id": "t-1",
        "narrative": None,
    }
    # Written beside itself and moved over it: no file is left beside it.
    assert [p.name for p in saved.parent.iterdir()] == [saved.name]
    assert result["warnings"] == []


@pytest.mark.asyncio
async def test_search_leaves_out_the_path_when_the_file_cannot_be_rewritten(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed rewrite: no saved_to, a warning, and no file left beside it."""
    payload = {"query": "q", "results": [_cli_paper(legalToShow=False)]}
    seen: dict[str, Path] = {}

    async def fake_cli(argv: list[str]) -> tuple[int, str, str]:
        seen["out"] = Path(argv[argv.index("-o") + 1])
        seen["out"].write_text(json.dumps(payload), encoding="utf-8")
        return 0, "", ""

    def refuse(_src: Any, _dst: Any) -> None:
        raise OSError("read-only folder")

    monkeypatch.setattr(asta, "asta_bin", lambda: "asta")
    monkeypatch.setattr(asta, "_run_cli", fake_cli)
    monkeypatch.setattr(asta.os, "replace", refuse)
    result = await asta.search_asta("q", max_results=5, include_abstract=True)

    assert "saved_to" not in result
    assert result["warnings"] == [asta.ARTIFACT_NOT_REWRITTEN]
    (paper,) = result["papers"]
    assert paper["content_withheld"] == asta.CONTENT_WITHHELD
    assert paper["abstract"] is None and paper["snippets"] == []
    # The CLI's file is as it wrote it, and no staged file is left beside it.
    out = seen["out"]
    assert json.loads(out.read_text(encoding="utf-8")) == payload
    assert [p.name for p in out.parent.iterdir()] == [out.name]


@pytest.mark.asyncio
async def test_search_cli_failure_surfaces_stderr_and_auth_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fail_cli(_argv: list[str]) -> tuple[int, str, str]:
        return 1, "", "401 unauthorized: no token"

    monkeypatch.setattr(asta, "asta_bin", lambda: "asta")
    monkeypatch.setattr(asta, "_run_cli", fail_cli)
    with pytest.raises(asta.AstaSearchError) as err:
        await asta.search_asta("anything")
    assert "401 unauthorized" in str(err.value)
    assert "asta auth login" in str(err.value)


@pytest.mark.asyncio
async def test_search_empty_query_rejected() -> None:
    with pytest.raises(ValueError, match="query is required"):
        await asta.search_asta("   ")


def test_scrubbed_env_keeps_asta_drops_agent_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ASTA_TOKEN", "asta-secret")
    monkeypatch.setenv("OLMOEARTH_API_KEY", "studio-secret")
    monkeypatch.setenv("LLM_API_KEY", "llm-secret")
    env = asta._scrubbed_env()
    assert env.get("ASTA_TOKEN") == "asta-secret"  # the CLI needs its own auth
    assert "OLMOEARTH_API_KEY" not in env
    assert "LLM_API_KEY" not in env
