# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Tests for the olmoearth_similarity_search tool."""

from __future__ import annotations

import pytest

from olmoearth_agent.harness.state import ThreadState
from olmoearth_agent.llm.types import ToolCall
from olmoearth_agent.tools.registry import ToolContext, ToolRegistry
from olmoearth_agent.tools.similarity import build_similarity_tools

_CORPUS = [[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]]


def _ctx() -> ToolContext:
    return ToolContext(studio=None, state=ThreadState())  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_tool_returns_matches() -> None:
    tool = build_similarity_tools()[0]
    result = await tool.handler(
        {"query": [1.0, 0.0], "corpus": _CORPUS, "ids": ["a", "b", "c"], "k": 2},
        _ctx(),
    )
    assert result["matches"][0]["id"] == "a"
    assert "geographic_prior" not in result


@pytest.mark.asyncio
async def test_tool_adds_geographic_prior() -> None:
    tool = build_similarity_tools()[0]
    result = await tool.handler(
        {
            "query": [1.0, 0.0],
            "corpus": _CORPUS,
            "k": 3,
            "query_coord": [0.0, 0.0],
            "coords": [[0.1, 0.1], [0.0, 0.1], [-0.1, 0.0]],
        },
        _ctx(),
    )
    assert "geographic_prior" in result
    assert result["geographic_prior"]["warning"] is True


@pytest.mark.asyncio
async def test_tool_rejects_coords_length_mismatch() -> None:
    tool = build_similarity_tools()[0]
    with pytest.raises(ValueError, match="coords length"):
        await tool.handler(
            {
                "query": [1.0, 0.0],
                "corpus": _CORPUS,
                "query_coord": [0.0, 0.0],
                "coords": [[0.1, 0.1]],
            },
            _ctx(),
        )


@pytest.mark.asyncio
async def test_empty_corpus_surfaces_to_model_via_dispatch() -> None:
    registry = ToolRegistry()
    registry.register_all(build_similarity_tools())
    result = await registry.dispatch(
        ToolCall(
            id="c1",
            name="olmoearth_similarity_search",
            arguments={"query": [1.0, 0.0], "corpus": []},
        ),
        _ctx(),
    )
    assert result["ok"] is False
    assert "corpus" in result["error"]
