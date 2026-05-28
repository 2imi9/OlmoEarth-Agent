# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Tests for the olmoearth-case-narrative tool."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from olmoearth_agent.harness.state import ThreadState
from olmoearth_agent.tools.narrative import build_narrative_tools
from olmoearth_agent.tools.registry import ToolContext


@pytest.mark.asyncio
async def test_case_narrative_tool_uses_state_provenance() -> None:
    state = ThreadState()
    state.provenance.record_tool_call(
        "olmoearth_load_context", {}, {"ok": True, "result": {}}
    )
    ctx = ToolContext(studio=None, state=state)  # type: ignore[arg-type]
    tool = build_narrative_tools()[0]
    result = await tool.handler(
        {
            "title": "My Case",
            "results": [
                {
                    "result_id": "r1",
                    "tile_urls": ["/t"],
                    "property_names": ["s"],
                    "creation_time": datetime.now(timezone.utc).isoformat(),
                }
            ],
        },
        ctx,
    )
    md = result["markdown"]
    assert "# My Case" in md
    assert "olmoearth_load_context" in md  # provenance pulled from state
    assert result["result_count"] == 1
    assert result["gated"] is False  # creation_time is "now" -> fresh
