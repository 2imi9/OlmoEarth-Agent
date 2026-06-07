# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""The ``olmoearth-case-narrative`` tool bundle (skill #14).

Turns a run's results + provenance into a stakeholder Markdown report,
with a freshness gate that refuses to render stale tiles.
"""

from __future__ import annotations

from typing import Any

from olmoearth_agent.llm.types import ToolSpec
from olmoearth_agent.reporting.narrative import build_narrative
from olmoearth_agent.tools.registry import RegisteredTool, ToolContext


async def _case_narrative(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    return build_narrative(
        args["title"],
        args.get("results", []),
        provenance=ctx.state.provenance,
        window_hours=float(args.get("freshness_window_hours", 24)),
    )


def build_narrative_tools() -> list[RegisteredTool]:
    """Return the ``olmoearth-case-narrative`` tool bundle."""
    return [
        RegisteredTool(
            spec=ToolSpec(
                name="olmoearth_case_narrative",
                description=(
                    "Write a stakeholder Markdown report from prediction "
                    "results (tile URLs + properties) and the run's "
                    "provenance. Tiles older than the freshness window are "
                    "NOT rendered (struck through): important for "
                    "disaster-response briefs. Call at the end of a task to "
                    "produce the deliverable."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "results": {
                            "type": "array",
                            "items": {"type": "object"},
                            "description": (
                                "Each: result_id, tile_urls, property_names, "
                                "creation_time (ISO-8601)."
                            ),
                        },
                        "freshness_window_hours": {
                            "type": "number",
                            "default": 24,
                        },
                    },
                    "required": ["title", "results"],
                },
            ),
            handler=_case_narrative,
        ),
    ]
