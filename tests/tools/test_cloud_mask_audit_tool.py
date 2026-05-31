# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Tests for the olmoearth_cloud_mask_audit tool."""

from __future__ import annotations

import pytest

from olmoearth_agent.harness.state import ThreadState
from olmoearth_agent.llm.types import ToolCall
from olmoearth_agent.tools.cloud_mask_audit import build_cloud_mask_audit_tools
from olmoearth_agent.tools.registry import ToolContext, ToolRegistry


def _ctx() -> ToolContext:
    return ToolContext(studio=None, state=ThreadState())  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_audit_ensemble_only() -> None:
    tool = build_cloud_mask_audit_tools()[0]
    result = await tool.handler(
        {"masks": {"cfmask": [1, 0, 1, 0], "s2cloudless": [1, 1, 0, 0]}},
        _ctx(),
    )
    assert result["ensemble"]["disagreement_rate"] == 0.5
    assert "attribution" not in result


@pytest.mark.asyncio
async def test_audit_with_verdict() -> None:
    tool = build_cloud_mask_audit_tools()[0]
    result = await tool.handler(
        {
            "masks": {"a": [0, 0, 0, 0], "b": [0, 0, 0, 0]},
            "model_error": [1, 1, 0, 0],
        },
        _ctx(),
    )
    assert result["attribution"]["verdict"] == "model-limited"


@pytest.mark.asyncio
async def test_invalid_masks_surface_to_model_via_dispatch() -> None:
    registry = ToolRegistry()
    registry.register_all(build_cloud_mask_audit_tools())
    result = await registry.dispatch(
        ToolCall(
            id="c1",
            name="olmoearth_cloud_mask_audit",
            arguments={"masks": {"only": [0, 1, 0]}},
        ),
        _ctx(),
    )
    assert result["ok"] is False
    assert "masks" in result["error"]
