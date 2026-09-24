# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""The skill-loading tool: ``olmoearth_load_skill``.

It pulls one vendored ``SKILL.md`` package's full instructions into context
when a task matches. The index of those skills (name + description) is
already in the system prompt, and an unknown name's error lists the
available ones, so there is no listing tool.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from olmoearth_agent.llm.types import ToolSpec
from olmoearth_agent.tools.registry import RegisteredTool, ToolContext

if TYPE_CHECKING:
    from olmoearth_agent.skills.loader import SkillLoader


def build_skill_tools(loader: "SkillLoader | None" = None) -> list[RegisteredTool]:
    """Return the skill-loading tool (binds a :class:`SkillLoader`)."""
    # Imported lazily: a module-level import creates a cycle
    # (skills.loader -> skills/__init__ -> skills.registry -> this module)
    # that breaks whenever tools.skill_tools is imported before skills.registry.
    from olmoearth_agent.skills.loader import SkillLoader

    skill_loader = loader or SkillLoader()

    async def _load_skill(args: dict[str, Any], _ctx: ToolContext) -> dict[str, Any]:
        """Return one skill's SKILL.md body (unknown names list the available)."""
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
                name="olmoearth_load_skill",
                description=(
                    "Load the full step-by-step instructions for one "
                    "OlmoEarth skill by name (listed in the system prompt). "
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
