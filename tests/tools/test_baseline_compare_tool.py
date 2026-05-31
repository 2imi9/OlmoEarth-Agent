# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Tests for the olmoearth_baseline_compare tool."""

from __future__ import annotations

import pytest

from olmoearth_agent.harness.state import ThreadState
from olmoearth_agent.llm.types import ToolCall
from olmoearth_agent.tools.baseline_compare import build_baseline_compare_tools
from olmoearth_agent.tools.registry import ToolContext, ToolRegistry


def _ctx() -> ToolContext:
    return ToolContext(studio=None, state=ThreadState())  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_tool_metric_comparison_only() -> None:
    tool = build_baseline_compare_tools()[0]
    result = await tool.handler(
        {
            "y_true": [1, 1, 0, 0],
            "olmoearth_pred": [1, 1, 0, 0],
            "alphaearth_pred": [1, 0, 0, 1],
        },
        _ctx(),
    )
    assert result["overall_winner"] == "olmoearth"
    assert "difference_raster" not in result


@pytest.mark.asyncio
async def test_tool_adds_difference_raster() -> None:
    tool = build_baseline_compare_tools()[0]
    result = await tool.handler(
        {
            "y_true": [1, 0],
            "olmoearth_pred": [1, 0],
            "alphaearth_pred": [1, 1],
            "olmoearth_layer": [0.9, 0.8],
            "alphaearth_layer": [0.4, 0.5],
        },
        _ctx(),
    )
    assert "difference_raster" in result
    assert result["difference_raster"]["olmoearth_higher_fraction"] == 1.0


@pytest.mark.asyncio
async def test_length_error_surfaces_via_dispatch() -> None:
    registry = ToolRegistry()
    registry.register_all(build_baseline_compare_tools())
    result = await registry.dispatch(
        ToolCall(
            id="c1",
            name="olmoearth_baseline_compare",
            arguments={
                "y_true": [1, 1, 0, 0],
                "olmoearth_pred": [1, 1, 0],
                "alphaearth_pred": [1, 0, 0, 1],
            },
        ),
        _ctx(),
    )
    assert result["ok"] is False
    assert "same length" in result["error"]
