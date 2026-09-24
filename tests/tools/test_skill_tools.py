# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Tests for the skill-loading tool (olmoearth_load_skill)."""

from __future__ import annotations

from pathlib import Path

import pytest

from olmoearth_agent.harness.state import ThreadState
from olmoearth_agent.skills.loader import SkillLoader
from olmoearth_agent.tools.registry import RegisteredTool, ToolContext
from olmoearth_agent.tools.skill_tools import build_skill_tools


def _make_skill(root: Path, name: str) -> None:
    folder = root / name
    folder.mkdir(parents=True)
    (folder / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: desc for {name}\n---\n# {name}\nsteps\n",
        encoding="utf-8",
    )


def _tool(tools: list[RegisteredTool], name: str) -> RegisteredTool:
    return next(t for t in tools if t.spec.name == name)


def test_there_is_no_listing_tool(tmp_path: Path) -> None:
    """The skill index is in the system prompt; load_skill's error lists names."""
    tools = build_skill_tools(SkillLoader(root=tmp_path))
    assert [t.spec.name for t in tools] == ["olmoearth_load_skill"]


@pytest.mark.asyncio
async def test_load_skill_tool_returns_body(tmp_path: Path) -> None:
    _make_skill(tmp_path, "alpha")
    tools = build_skill_tools(SkillLoader(root=tmp_path))
    ctx = ToolContext(studio=None, state=ThreadState())  # type: ignore[arg-type]
    result = await _tool(tools, "olmoearth_load_skill").handler({"name": "alpha"}, ctx)
    assert "# alpha" in result["instructions"]


@pytest.mark.asyncio
async def test_load_skill_tool_unknown_lists_available(tmp_path: Path) -> None:
    _make_skill(tmp_path, "alpha")
    tools = build_skill_tools(SkillLoader(root=tmp_path))
    ctx = ToolContext(studio=None, state=ThreadState())  # type: ignore[arg-type]
    # Unknown skill raises (dispatch -> ok=False); the error carries the
    # available names so the model can retry.
    with pytest.raises(ValueError, match="unknown skill 'ghost'.*alpha"):
        await _tool(tools, "olmoearth_load_skill").handler({"name": "ghost"}, ctx)
