# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Tests for the olmoearth_change_detect tool."""

from __future__ import annotations

import pytest

from olmoearth_agent.analysis.change_detect import TooFewDatesError
from olmoearth_agent.harness.state import ThreadState
from olmoearth_agent.llm.types import ToolCall
from olmoearth_agent.tools.change_detect import build_change_detect_tools
from olmoearth_agent.tools.registry import ToolContext, ToolRegistry


def _ctx() -> ToolContext:
    return ToolContext(studio=None, state=ThreadState())  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_change_detect_tool_happy_path() -> None:
    tool = build_change_detect_tools()[0]
    result = await tool.handler(
        {
            "series": [
                {"date": "2024-01-01", "value": 0.1},
                {"date": "2024-02-01", "value": 0.5},
                {"date": "2024-03-01", "value": 0.2},
            ],
            "metric": "positive_fraction",
        },
        _ctx(),
    )
    assert result["n_dates"] == 3
    assert result["trend"] == "oscillating"
    assert result["reversals"] == 1
    assert result["metric"] == "positive_fraction"


@pytest.mark.asyncio
async def test_change_detect_tool_refuses_two_dates() -> None:
    tool = build_change_detect_tools()[0]
    with pytest.raises(TooFewDatesError):
        await tool.handler(
            {
                "series": [
                    {"date": "2024-01-01", "value": 0.1},
                    {"date": "2024-06-01", "value": 0.5},
                ]
            },
            _ctx(),
        )


@pytest.mark.asyncio
async def test_refusal_surfaces_to_model_via_dispatch() -> None:
    registry = ToolRegistry()
    registry.register_all(build_change_detect_tools())
    result = await registry.dispatch(
        ToolCall(
            id="c1",
            name="olmoearth_change_detect",
            arguments={
                "series": [
                    {"date": "2024-01-01", "value": 0.1},
                    {"date": "2024-06-01", "value": 0.5},
                ]
            },
        ),
        _ctx(),
    )
    assert result["ok"] is False
    assert "3" in result["error"]
