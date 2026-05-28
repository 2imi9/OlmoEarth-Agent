# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Tools that expose the vendored ``SKILL.md`` packages to the agent.

``olmoearth_list_skills`` returns the progressive-disclosure index
(name + description) of the vendored skills (#1-#4). ``olmoearth_load_skill``
pulls one skill's full instructions into context when a task matches.
"""

from __future__ import annotations

from typing import Any

from olmoearth_agent.llm.types import ToolSpec
from olmoearth_agent.skills.loader import SkillLoader
from olmoearth_agent.tools.registry import RegisteredTool, ToolContext


def build_skill_tools(loader: SkillLoader | None = None) -> list[RegisteredTool]:
    """Return the skill-loading tool bundle (binds a :class:`SkillLoader`)."""
    skill_loader = loader or SkillLoader()

    async def _list_skills(
        _args: dict[str, Any], _ctx: ToolContext
    ) -> dict[str, Any]:
        return {
            "skills": [
                {"name": s.name, "description": s.description}
                for s in skill_loader.discover()
            ]
        }

    async def _load_skill(args: dict[str, Any], _ctx: ToolContext) -> dict[str, Any]:
        name = args["name"]
        try:
            body = skill_loader.load(name)
        except KeyError:
            available = [s.name for s in skill_loader.discover()]
            return {"error": f"unknown skill {name!r}", "available": available}
        return {"name": name, "instructions": body}

    return [
        RegisteredTool(
            spec=ToolSpec(
                name="olmoearth_list_skills",
                description=(
                    "List the available OlmoEarth instruction skills (data "
                    "prep, Studio job config, embeddings) with their "
                    "descriptions. Call this when a task involves preparing "
                    "labels, configuring a Studio job, or choosing "
                    "embeddings-vs-fine-tune, to see which skill to load."
                ),
                parameters={"type": "object", "properties": {}, "required": []},
            ),
            handler=_list_skills,
        ),
        RegisteredTool(
            spec=ToolSpec(
                name="olmoearth_load_skill",
                description=(
                    "Load the full step-by-step instructions for one "
                    "OlmoEarth skill by name (from olmoearth_list_skills). "
                    "Returns the SKILL.md body; follow it to complete the "
                    "task, citing the pitfall numbers and reference docs it "
                    "names."
                ),
                parameters={
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                },
            ),
            handler=_load_skill,
        ),
    ]
