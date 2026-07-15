# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Unit tests for the remember/forget tool bundle (tools/memory.py)."""

from __future__ import annotations

from pathlib import Path

import pytest

from olmoearth_agent.harness import memory
from olmoearth_agent.llm.types import ToolCall
from olmoearth_agent.security.paths import OUTPUT_ROOT_ENV
from olmoearth_agent.tools.memory import build_memory_tools
from olmoearth_agent.tools.registry import ToolContext, ToolRegistry


@pytest.fixture(autouse=True)
def _workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv(OUTPUT_ROOT_ENV, str(tmp_path))
    return tmp_path


def _registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register_all(build_memory_tools())
    return registry


_CTX = ToolContext(studio=None, state=None)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_remember_then_forget_roundtrip(tmp_path: Path) -> None:
    registry = _registry()
    result = await registry.dispatch(
        ToolCall(
            id="c1",
            name="olmoearth_remember",
            arguments={"key": "default_project", "value": "PA Karst (12ab)"},
        ),
        _CTX,
    )
    assert result["ok"] is True
    assert result["result"]["stored_keys"] == ["default_project"]
    assert memory.load_facts(tmp_path)["default_project"]["value"] == "PA Karst (12ab)"

    result = await registry.dispatch(
        ToolCall(
            id="c2", name="olmoearth_forget", arguments={"key": "default_project"}
        ),
        _CTX,
    )
    assert result["ok"] is True
    assert result["result"] == {"key": "default_project", "removed": True}
    assert memory.load_facts(tmp_path) == {}


@pytest.mark.asyncio
async def test_remember_bad_key_surfaces_guidance() -> None:
    result = await _registry().dispatch(
        ToolCall(
            id="c1",
            name="olmoearth_remember",
            arguments={"key": "Not A Slug", "value": "v"},
        ),
        _CTX,
    )
    assert result["ok"] is False
    assert "invalid memory key" in result["error"]


@pytest.mark.asyncio
async def test_remember_missing_value_rejected_by_schema() -> None:
    result = await _registry().dispatch(
        ToolCall(id="c1", name="olmoearth_remember", arguments={"key": "k"}),
        _CTX,
    )
    assert result["ok"] is False
    assert "missing required argument 'value'" in result["error"]
