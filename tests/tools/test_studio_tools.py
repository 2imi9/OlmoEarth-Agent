# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Tests for the foundational Studio tools (the request-AOI directive)."""

from __future__ import annotations

import pytest

from olmoearth_agent.llm.types import ToolCall
from olmoearth_agent.tools.registry import ToolContext, ToolRegistry
from olmoearth_agent.tools.studio import build_studio_tools


def test_request_aoi_is_registered() -> None:
    names = [t.spec.name for t in build_studio_tools()]
    assert "olmoearth_request_aoi" in names


@pytest.mark.asyncio
async def test_request_aoi_returns_needs_aoi_directive() -> None:
    registry = ToolRegistry()
    registry.register_all(build_studio_tools())
    result = await registry.dispatch(
        ToolCall(
            id="c1",
            name="olmoearth_request_aoi",
            arguments={"purpose": "change detection", "suggested_name": "My AOI"},
        ),
        # request_aoi makes no Studio call, so a None client is fine.
        ctx=ToolContext(studio=None, state=None),  # type: ignore[arg-type]
    )
    assert result["ok"] is True
    inner = result["result"]
    assert inner["needs_aoi"] is True
    assert inner["purpose"] == "change detection"
    assert inner["suggested_name"] == "My AOI"
