# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Tests for the olmoearth_litsearch tool bundle."""

from __future__ import annotations

from typing import Any

import pytest

from olmoearth_agent.harness.state import ThreadState
from olmoearth_agent.llm.types import ToolCall
from olmoearth_agent.tools import litsearch as litsearch_tools
from olmoearth_agent.tools.litsearch import build_litsearch_tools
from olmoearth_agent.tools.registry import ToolContext, ToolRegistry


def _ctx() -> ToolContext:
    return ToolContext(studio=None, state=ThreadState())  # type: ignore[arg-type]


def test_bundle_exposes_two_tools_with_object_schemas() -> None:
    tools = build_litsearch_tools()
    names = {t.spec.name for t in tools}
    assert names == {"olmoearth_litsearch", "olmoearth_litsearch_resolve"}
    for t in tools:
        assert t.spec.parameters["type"] == "object"


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
            name="olmoearth_litsearch_resolve",
            arguments={"identifier": "not-an-id"},
        ),
        _ctx(),
    )
    assert result["ok"] is False
    assert "could not parse" in result["error"]


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
        return {"count": 1, "papers": [{"id": "CorpusId:1"}], "sources": ["asta"],
                "warnings": [], "saved_to": "x.json"}

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
