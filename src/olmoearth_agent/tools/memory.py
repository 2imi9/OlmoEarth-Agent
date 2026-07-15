# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Tools for the cross-thread preference memory (``harness/memory.py``).

``olmoearth_remember`` / ``olmoearth_forget`` let the model persist a standing
user preference the moment the user states one, so later conversations start
with it already applied (the harness injects the stored facts into the system
prompt). Core tools, always registered — not a skill.
"""

from __future__ import annotations

from typing import Any

from olmoearth_agent.harness import memory
from olmoearth_agent.llm.types import ToolSpec
from olmoearth_agent.tools.registry import RegisteredTool, ToolContext

_REMEMBER_SPEC = ToolSpec(
    name="olmoearth_remember",
    description=(
        "Save ONE durable user preference so future conversations apply it "
        "automatically (e.g. default project, usual area of interest, "
        "preferred imagery sources, reporting style). Use ONLY when the user "
        "explicitly states a standing preference ('always...', 'my default "
        "is...', 'from now on...'); never store one-off task details, tool "
        "output, or anything secret (keys, tokens). Overwrites the key if it "
        "already exists."
    ),
    parameters={
        "type": "object",
        "properties": {
            "key": {
                "type": "string",
                "description": (
                    "Short slug naming the preference, e.g. 'default_project' "
                    "or 'preferred_sources' (lowercase letters, digits, '-', "
                    "'_'; max 40 chars)."
                ),
            },
            "value": {
                "type": "string",
                "description": (
                    "The preference, one line, max 240 chars. Include ids "
                    "where known, e.g. 'PA Karst (project_id 12ab)'."
                ),
            },
        },
        "required": ["key", "value"],
    },
)

_FORGET_SPEC = ToolSpec(
    name="olmoearth_forget",
    description=(
        "Delete one saved user preference by key, when the user retracts it "
        "('stop assuming...', 'forget my default...')."
    ),
    parameters={
        "type": "object",
        "properties": {
            "key": {"type": "string", "description": "The stored key to remove."}
        },
        "required": ["key"],
    },
)


async def _remember(args: dict[str, Any], _ctx: ToolContext) -> dict[str, Any]:
    """Handler: persist one preference; the store enforces caps/sanitizing."""
    return memory.remember(str(args["key"]), str(args["value"]))


async def _forget(args: dict[str, Any], _ctx: ToolContext) -> dict[str, Any]:
    """Handler: drop one preference; says whether it existed."""
    key = str(args["key"])
    return {"key": key, "removed": memory.forget(key)}


def build_memory_tools() -> list[RegisteredTool]:
    """The cross-thread memory bundle for the default registry."""
    return [
        RegisteredTool(spec=_REMEMBER_SPEC, handler=_remember),
        RegisteredTool(spec=_FORGET_SPEC, handler=_forget),
    ]
