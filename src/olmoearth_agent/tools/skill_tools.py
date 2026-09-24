# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""The skill-loading tool: ``olmoearth_load_skill``.

Two kinds of skill load through it. An instruction skill (a ``SKILL.md``
package, #1-#3 and #17, in ``skills/packages/``) returns its full steps; the
index of those skills is already in the system prompt, so there is no listing
tool.
A skill whose tools are deferred (registered with a ``group``; see
:mod:`olmoearth_agent.tools.registry`) has its group loaded for the rest of
the run, so those tools' specs are sent from the next turn on. A skill can be
both (``olmoearth-rslearn``: its SKILL.md and its four tools).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from olmoearth_agent.llm.types import ToolSpec
from olmoearth_agent.tools.registry import RegisteredTool, ToolContext

if TYPE_CHECKING:
    from olmoearth_agent.skills.loader import SkillLoader
    from olmoearth_agent.tools.registry import ToolRegistry


def _deferred_note(groups: dict[str, list[str]]) -> str:
    """One sentence naming the skills that bring deferred tools, and the tools."""
    if not groups:
        return ""
    parts = [
        f"{group} ({', '.join(n.removeprefix('olmoearth_') for n in names)})"
        for group, names in groups.items()
    ]
    return (
        " These tools are not sent until their skill is loaded: "
        + "; ".join(parts)
        + "."
    )


def build_skill_tools(
    loader: "SkillLoader | None" = None,
    *,
    registry: "ToolRegistry | None" = None,
) -> list[RegisteredTool]:
    """Return the skill-loading tool, bound to a loader and (optionally) a registry.

    With ``registry``, loading a skill that names a deferred group loads that
    group into the run's state, and the tool's description lists the groups
    registered so far (register the deferred bundles first).
    """
    # Imported lazily: a module-level import creates a cycle
    # (skills.loader -> skills/__init__ -> skills.registry -> this module)
    # that breaks whenever tools.skill_tools is imported before skills.registry.
    from olmoearth_agent.skills.loader import SkillLoader

    skill_loader = loader or SkillLoader()

    async def _load_skill(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
        """Return a skill's SKILL.md steps and/or load its deferred tools."""
        name = str(args["name"]).strip()
        try:
            body: str | None = skill_loader.load(name)
        except KeyError:
            body = None
        groups = registry.groups() if registry is not None else {}
        tools = groups.get(name, [])
        if body is None and not tools:
            # Raise so dispatch reports ok=False with the recovery hint in the
            # error (a returned {"error": ...} dict is wrapped as ok=True, hiding
            # the failure). The available names let the model retry.
            available = sorted({s.name for s in skill_loader.discover()} | set(groups))
            raise ValueError(
                f"unknown skill {name!r}; available: {', '.join(available)}"
            )
        out: dict[str, Any] = {"name": name}
        if body is not None:
            out["instructions"] = body
        if tools:
            loaded = getattr(ctx.state, "loaded_groups", None)
            if isinstance(loaded, set):
                loaded.add(name)
            out["tools_loaded"] = tools
            out["tools_note"] = (
                "these tools are available from your next call; call them "
                "directly by name"
            )
        return out

    groups = registry.groups() if registry is not None else {}
    return [
        RegisteredTool(
            spec=ToolSpec(
                name="olmoearth_load_skill",
                description=(
                    "Load one OlmoEarth skill by name. An instruction skill "
                    "(listed in the system prompt) returns its SKILL.md steps: "
                    "follow them, citing the pitfall numbers and reference docs "
                    "they name." + _deferred_note(groups)
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
