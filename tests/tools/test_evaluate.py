# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Tests for the olmoearth-evaluate tool bundle."""

from __future__ import annotations

import pytest

from olmoearth_agent.harness.state import ThreadState
from olmoearth_agent.tools.evaluate import build_evaluate_tools
from olmoearth_agent.tools.registry import RegisteredTool, ToolContext


def _tool(name: str) -> RegisteredTool:
    return next(t for t in build_evaluate_tools() if t.spec.name == name)


def _ctx() -> ToolContext:
    return ToolContext(studio=None, state=ThreadState())  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_classification_metrics_tool() -> None:
    result = await _tool("olmoearth_classification_metrics").handler(
        {"y_true": [1, 1, 0, 0], "y_pred": [1, 0, 0, 0]}, _ctx()
    )
    assert result["accuracy"] == 0.75
    assert result["per_class"]["1"]["recall"] == 0.5
