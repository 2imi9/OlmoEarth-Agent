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
            "authors": [{"name": "B. Author", "id": "2"}, {"name": "C. Author", "id": "3"}],
            "venue": "TGRS",
            "url": None,
            "citationCount": 42,
            "relevanceScore": 0.97,
            "relevanceJudgement": {"relevanceSummary": "directly on topic"},
            "snippets": [
                {"text": "Our cloud mask improves F1 by 0.12. " + "x" * 500,
                 "sectionTitle": "Results"},
                {"text": "second snippet", "sectionTitle": "Methods"},
                {"text": "third snippet", "sectionTitle": "Methods"},
                {"text": "fourth snippet is dropped", "sectionTitle": "Appendix"},
            ],
        },
    ],
}


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


def test_parse_tolerates_malformed_payload() -> None:
    assert asta.parse_asta_results({}, max_results=5, include_abstract=False) == []
    assert (
        asta.parse_asta_results({"results": "nope"}, max_results=5, include_abstract=False)
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
