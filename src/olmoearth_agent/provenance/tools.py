# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""The ``olmoearth-provenance`` tool bundle (skill #14).

Exposes the run's provenance log to the agent so it can report what was
done and emit a replay skeleton at the end of a task.
"""

from __future__ import annotations

from typing import Any

from olmoearth_agent.llm.types import ToolSpec
from olmoearth_agent.tools.registry import RegisteredTool, ToolContext


async def _provenance_summary(
    _args: dict[str, Any], ctx: ToolContext
) -> dict[str, Any]:
    log = ctx.state.provenance
    return {
        "run_id": log.run_id,
        "entry_count": len(log.entries),
        "calls": [e.api_call for e in log.entries],
        "replay_script": log.replay_script(),
    }


def build_provenance_tools() -> list[RegisteredTool]:
    """Return the provenance tool bundle."""
    return [
        RegisteredTool(
            spec=ToolSpec(
                name="olmoearth_provenance_summary",
                description=(
                    "Summarize what this run did: the ordered list of tool "
                    "calls and a replay skeleton. Call this at the end of a "
                    "task to give the user an auditable record."
                ),
                parameters={"type": "object", "properties": {}, "required": []},
            ),
            handler=_provenance_summary,
        ),
    ]
