# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Tests for the olmoearth_automate tool bundle."""

from __future__ import annotations

from typing import Any

import pytest

from olmoearth_agent.harness.state import ThreadState
from olmoearth_agent.llm.types import ToolCall
from olmoearth_agent.tools import automate as automate_tools
from olmoearth_agent.tools.automate import build_automate_tools
from olmoearth_agent.tools.registry import ToolContext, ToolRegistry


def _ctx() -> ToolContext:
    return ToolContext(studio=None, state=ThreadState())  # type: ignore[arg-type]


def test_bundle_exposes_one_tool_with_object_schema() -> None:
    tools = build_automate_tools()
    assert [t.spec.name for t in tools] == ["olmoearth_automate"]
    assert tools[0].spec.parameters["type"] == "object"


@pytest.mark.asyncio
async def test_handler_wires_args(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    async def fake_automate(**kwargs: Any) -> dict[str, Any]:
        seen.update(kwargs)
        return {"decision": "embeddings", "config": {}, "ask_for": []}

    monkeypatch.setattr(automate_tools, "automate", fake_automate)
    tool = build_automate_tools()[0]
    out = await tool.handler(
        {"task": "200 samples, 9 classes, t4", "hf_dataset": "org/ds"}, _ctx()
    )
    assert seen["task"] == "200 samples, 9 classes, t4"
    assert seen["hf_dataset"] == "org/ds"
    assert out["decision"] == "embeddings"


@pytest.mark.asyncio
async def test_dispatch_with_no_inputs_returns_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # No hf_dataset -> no network; the default decision path runs in-process.
    registry = ToolRegistry()
    registry.register_all(build_automate_tools())
    result = await registry.dispatch(
        ToolCall(id="c1", name="olmoearth_automate", arguments={}), _ctx()
    )
    assert result["ok"] is True
    assert result["result"]["decision"] == "embeddings"
    assert result["result"]["ask_for"]  # missing inputs surfaced honestly
