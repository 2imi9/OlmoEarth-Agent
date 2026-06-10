# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Tools that expose the vendored ``SKILL.md`` packages to the agent.

``olmoearth_list_skills`` returns the progressive-disclosure index
(name + description) of the vendored skills (#1-#4). ``olmoearth_load_skill``
pulls one skill's full instructions into context when a task matches.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from olmoearth_agent.llm.types import ToolSpec
from olmoearth_agent.tools.registry import RegisteredTool, ToolContext

if TYPE_CHECKING:
    from olmoearth_agent.skills.loader import SkillLoader


def build_skill_tools(loader: "SkillLoader | None" = None) -> list[RegisteredTool]:
    """Return the skill-loading tool bundle (binds a :class:`SkillLoader`)."""
    # Imported lazily: a module-level import creates a cycle
    # (skills.loader -> skills/__init__ -> skills.registry -> this module)
    # that breaks whenever tools.skill_tools is imported before skills.registry.
    from olmoearth_agent.skills.loader import SkillLoader

    skill_loader = loader or SkillLoader()

    async def _list_skills(_args: dict[str, Any], _ctx: ToolContext) -> dict[str, Any]:
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
        except KeyError as exc:
            # Raise so dispatch reports ok=False with the recovery hint in the
            # error (a returned {"error": ...} dict is wrapped as ok=True, hiding
            # the failure). The available names let the model retry.
            available = [s.name for s in skill_loader.discover()]
            raise ValueError(
                f"unknown skill {name!r}; available: {', '.join(available)}"
            ) from exc
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
