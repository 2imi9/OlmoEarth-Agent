# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Tests for the skill-loading tool (olmoearth_load_skill) and deferred groups."""

from __future__ import annotations

from pathlib import Path

import pytest

from olmoearth_agent.harness.state import ThreadState
from olmoearth_agent.llm.types import ToolSpec
from olmoearth_agent.skills.loader import SkillLoader
from olmoearth_agent.tools.registry import RegisteredTool, ToolContext, ToolRegistry
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


def _registry_with_deferred_group() -> ToolRegistry:
    """A registry with one core tool and one tool deferred under ``beta``."""

    async def noop(_args: dict[str, object], _ctx: ToolContext) -> dict[str, object]:
        return {}

    def spec(name: str) -> ToolSpec:
        return ToolSpec(name=name, description=name, parameters={"type": "object"})

    registry = ToolRegistry()
    registry.register(RegisteredTool(spec=spec("core_tool"), handler=noop))
    registry.register(
        RegisteredTool(spec=spec("olmoearth_beta_tool"), handler=noop), group="beta"
    )
    return registry


def test_there_is_no_listing_tool(tmp_path: Path) -> None:
    """The skill index is in the system prompt; load_skill's error lists names."""
    tools = build_skill_tools(SkillLoader(root=tmp_path))
    assert [t.spec.name for t in tools] == ["olmoearth_load_skill"]


@pytest.mark.asyncio
async def test_load_skill_loads_a_deferred_group(tmp_path: Path) -> None:
    registry = _registry_with_deferred_group()
    tools = build_skill_tools(SkillLoader(root=tmp_path), registry=registry)
    # The description names the group and its tools, so the model can ask.
    assert "beta (beta_tool)" in tools[0].spec.description
    state = ThreadState()
    ctx = ToolContext(studio=None, state=state)  # type: ignore[arg-type]
    assert "olmoearth_beta_tool" not in {s.name for s in registry.active_specs()}
    result = await _tool(tools, "olmoearth_load_skill").handler({"name": "beta"}, ctx)
    assert result["tools_loaded"] == ["olmoearth_beta_tool"]
    assert "instructions" not in result  # a tool-only skill has no SKILL.md
    assert state.loaded_groups == {"beta"}
    names = {s.name for s in registry.active_specs(state.loaded_groups)}
    assert names == {"core_tool", "olmoearth_beta_tool"}


@pytest.mark.asyncio
async def test_load_skill_returns_steps_and_tools_together(tmp_path: Path) -> None:
    """An instruction skill whose tools are deferred (like olmoearth-rslearn)."""
    _make_skill(tmp_path, "beta")
    registry = _registry_with_deferred_group()
    tools = build_skill_tools(SkillLoader(root=tmp_path), registry=registry)
    state = ThreadState()
    ctx = ToolContext(studio=None, state=state)  # type: ignore[arg-type]
    result = await _tool(tools, "olmoearth_load_skill").handler({"name": "beta"}, ctx)
    assert "# beta" in result["instructions"]
    assert result["tools_loaded"] == ["olmoearth_beta_tool"]
    assert state.loaded_groups == {"beta"}


@pytest.mark.asyncio
async def test_unknown_skill_lists_instruction_skills_and_groups(
    tmp_path: Path,
) -> None:
    _make_skill(tmp_path, "alpha")
    registry = _registry_with_deferred_group()
    tools = build_skill_tools(SkillLoader(root=tmp_path), registry=registry)
    ctx = ToolContext(studio=None, state=ThreadState())  # type: ignore[arg-type]
    with pytest.raises(
        ValueError, match="unknown skill 'ghost'; available: alpha, beta"
    ):
        await _tool(tools, "olmoearth_load_skill").handler({"name": "ghost"}, ctx)


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
