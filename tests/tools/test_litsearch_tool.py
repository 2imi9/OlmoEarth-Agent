# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Tests for the olmoearth_litsearch tool bundle."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from olmoearth_agent.harness.state import ThreadState
from olmoearth_agent.llm.types import ToolCall
from olmoearth_agent.security.paths import OUTPUT_ROOT_ENV
from olmoearth_agent.tools import litsearch as litsearch_tools
from olmoearth_agent.tools.litsearch import build_litsearch_tools
from olmoearth_agent.tools.registry import ToolContext, ToolRegistry


def _ctx() -> ToolContext:
    return ToolContext(studio=None, state=ThreadState())  # type: ignore[arg-type]


def test_bundle_is_one_tool_that_searches_or_resolves() -> None:
    (tool,) = build_litsearch_tools()
    assert tool.spec.name == "olmoearth_litsearch"
    assert tool.spec.parameters["type"] == "object"
    # The former olmoearth_litsearch_resolve is the identifier argument.
    assert {"query", "identifier"} <= set(tool.spec.parameters["properties"])
    assert tool.spec.parameters["required"] == []


@pytest.mark.asyncio
async def test_search_handler_wires_args(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    async def fake_search(**kwargs: Any) -> dict[str, Any]:
        seen.update(kwargs)
        return {"count": 0, "papers": [], "sources": ["arxiv"], "warnings": []}

    monkeypatch.setattr(litsearch_tools, "search_literature", fake_search)
    tool = next(
        t for t in build_litsearch_tools() if t.spec.name == "olmoearth_litsearch"
    )
    result = await tool.handler(
        {"query": "spatial cross validation", "source": "arxiv", "max_results": 3},
        _ctx(),
    )
    assert seen["query"] == "spatial cross validation"
    assert seen["source"] == "arxiv"
    assert seen["max_results"] == 3
    assert result["count"] == 0


@pytest.mark.asyncio
async def test_empty_query_surfaces_via_dispatch() -> None:
    registry = ToolRegistry()
    registry.register_all(build_litsearch_tools())
    result = await registry.dispatch(
        ToolCall(id="c1", name="olmoearth_litsearch", arguments={"query": "  "}),
        _ctx(),
    )
    assert result["ok"] is False
    assert "query" in result["error"]


@pytest.mark.asyncio
async def test_resolve_bad_identifier_surfaces_via_dispatch() -> None:
    registry = ToolRegistry()
    registry.register_all(build_litsearch_tools())
    result = await registry.dispatch(
        ToolCall(
            id="c2",
            name="olmoearth_litsearch",
            arguments={"identifier": "not-an-id"},
        ),
        _ctx(),
    )
    assert result["ok"] is False
    assert "could not parse" in result["error"]


@pytest.mark.asyncio
async def test_identifier_resolves_instead_of_searching(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, Any] = {}

    async def fake_resolve(**kwargs: Any) -> dict[str, Any]:
        seen.update(kwargs)
        return {"found": True, "paper": {"id": "doi:10.1/x"}, "warnings": []}

    async def no_search(**_kwargs: Any) -> dict[str, Any]:
        raise AssertionError("an identifier must not run a search")

    monkeypatch.setattr(litsearch_tools, "resolve_identifier", fake_resolve)
    monkeypatch.setattr(litsearch_tools, "search_literature", no_search)
    (tool,) = build_litsearch_tools()
    result = await tool.handler({"identifier": " 10.1/x "}, _ctx())
    assert seen == {"identifier": "10.1/x", "include_abstract": True}
    assert result["found"] is True


@pytest.mark.asyncio
async def test_neither_query_nor_identifier_names_both() -> None:
    registry = ToolRegistry()
    registry.register_all(build_litsearch_tools())
    result = await registry.dispatch(
        ToolCall(id="c3", name="olmoearth_litsearch", arguments={}), _ctx()
    )
    assert result["ok"] is False
    assert "query" in result["error"] and "identifier" in result["error"]


@pytest.mark.asyncio
async def test_asta_source_unavailable_returns_guidance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from olmoearth_agent.analysis import asta

    monkeypatch.setattr(asta, "asta_bin", lambda: None)
    tool = next(
        t for t in build_litsearch_tools() if t.spec.name == "olmoearth_litsearch"
    )
    result = await tool.handler({"query": "cloud masking", "source": "asta"}, _ctx())
    # Structured guidance with a working fallback, not an exception.
    assert result["available"] is False
    assert "source='both'" in result["hint"]
    assert "asta auth login" in result["hint"]


@pytest.mark.asyncio
async def test_asta_source_routes_to_asta_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from olmoearth_agent.analysis import asta as asta_mod

    seen: dict[str, Any] = {}

    async def fake_asta(**kwargs: Any) -> dict[str, Any]:
        seen.update(kwargs)
        return {
            "count": 1,
            "papers": [{"id": "CorpusId:1"}],
            "sources": ["asta"],
            "warnings": [],
            "saved_to": "x.json",
        }

    monkeypatch.setattr(asta_mod, "asta_available", lambda: True)
    monkeypatch.setattr(asta_mod, "search_asta", fake_asta)
    tool = next(
        t for t in build_litsearch_tools() if t.spec.name == "olmoearth_litsearch"
    )
    result = await tool.handler(
        {"query": "karst detection", "source": "asta", "max_results": 5}, _ctx()
    )
    assert seen["query"] == "karst detection"
    assert seen["max_results"] == 5
    assert result["sources"] == ["asta"]


@pytest.mark.asyncio
async def test_asta_source_withholds_text_asta_marks_not_legal_to_show(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No text of a paper Asta marks not legal to show, in the answer or its file."""
    from olmoearth_agent.analysis import asta as asta_mod

    paper = {
        "corpusId": 9,
        "title": "Withheld paper",
        "abstract": "secret abstract",
        "year": 2024,
        "authors": [{"name": "E. Author", "id": "5"}],
        "venue": "TGRS",
        "url": None,
        "citationCount": 3,
        "relevanceScore": 0.7,
        "relevanceJudgement": {"relevance": 2, "relevanceSummary": "secret summary"},
        "snippets": [{"text": "secret snippet", "sectionTitle": None}],
        "legalToShow": False,
    }

    async def fake_cli(argv: list[str]) -> tuple[int, str, str]:
        out = Path(argv[argv.index("-o") + 1])
        out.write_text(json.dumps({"query": "q", "results": [paper]}), "utf-8")
        return 0, "", ""

    monkeypatch.setenv(OUTPUT_ROOT_ENV, str(tmp_path))
    monkeypatch.setattr(asta_mod, "asta_bin", lambda: "asta")
    monkeypatch.setattr(asta_mod, "_run_cli", fake_cli)
    (tool,) = build_litsearch_tools()
    result = await tool.handler(
        {"query": "karst", "source": "asta", "include_abstract": True}, _ctx()
    )
    (record,) = result["papers"]
    assert record["content_withheld"] == asta_mod.CONTENT_WITHHELD
    assert record["title"] == "Withheld paper"
    assert record["url"] == "https://www.semanticscholar.org/p/9"
    assert "secret" not in json.dumps(result["papers"])
    # Nor does the CLI's file whose path the answer carries.
    saved = Path(result["saved_to"])
    assert tmp_path.resolve() in saved.resolve().parents
    assert "secret" not in saved.read_text(encoding="utf-8")
