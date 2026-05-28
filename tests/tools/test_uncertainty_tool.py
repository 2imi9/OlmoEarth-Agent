# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Tests for the olmoearth_area_of_applicability tool."""

from __future__ import annotations

import pytest

from olmoearth_agent.harness.state import ThreadState
from olmoearth_agent.llm.types import ToolCall
from olmoearth_agent.tools.registry import ToolContext, ToolRegistry
from olmoearth_agent.tools.uncertainty import build_uncertainty_tools

_TRAIN = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]]


def _ctx() -> ToolContext:
    return ToolContext(studio=None, state=ThreadState())  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_tool_flags_in_distribution() -> None:
    tool = build_uncertainty_tools()[0]
    result = await tool.handler(
        {"train_features": _TRAIN, "new_features": [[0.5, 0.5]]}, _ctx()
    )
    assert result["inside_aoa"] == [True]
    assert result["verdict"] == "within-AOA"


@pytest.mark.asyncio
async def test_tool_flags_out_of_distribution() -> None:
    tool = build_uncertainty_tools()[0]
    result = await tool.handler(
        {"train_features": _TRAIN, "new_features": [[1000.0, 1000.0]]}, _ctx()
    )
    assert result["ood_fraction"] == 1.0
    assert result["verdict"] == "mostly-OOD"


@pytest.mark.asyncio
async def test_invalid_input_surfaces_to_model_via_dispatch() -> None:
    registry = ToolRegistry()
    registry.register_all(build_uncertainty_tools())
    result = await registry.dispatch(
        ToolCall(
            id="c1",
            name="olmoearth_area_of_applicability",
            arguments={"train_features": [[0.0, 0.0]], "new_features": [[1.0, 1.0]]},
        ),
        _ctx(),
    )
    assert result["ok"] is False
    assert "2" in result["error"]
