# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 OlmoEarth Agent contributors
"""The ``olmoearth-predict`` tool bundle (skill #5) — the core run loop.

Search predictions (to discover reusable ``model_id``s), submit a new
prediction, and poll it. Polling reuses the foundational
``olmoearth_get_prediction`` tool. Result sub-tools (pixel-value,
features, files) are a follow-up within this skill.
"""

from __future__ import annotations

from typing import Any

from olmoearth_agent.llm.types import ToolSpec
from olmoearth_agent.tools.registry import RegisteredTool, ToolContext


async def _search_predictions(
    args: dict[str, Any], ctx: ToolContext
) -> dict[str, Any]:
    env = await ctx.studio.search_predictions(
        project_id=args.get("project_id"),
        limit=int(args.get("limit", 50)),
        offset=int(args.get("offset", 0)),
    )
    return {
        "total": env.total,
        "predictions": [
            {
                "id": r.get("id"),
                "name": r.get("name"),
                "status": r.get("status"),
                "model_id": r.get("model_id"),
            }
            for r in env.records
        ],
    }


async def _submit_prediction(
    args: dict[str, Any], ctx: ToolContext
) -> dict[str, Any]:
    record = await ctx.studio.submit_prediction(
        name=args["name"],
        project_id=args["project_id"],
        area_id=args["area_id"],
        model_id=args["model_id"],
        start_time=args["start_time"],
        end_time=args["end_time"],
    )
    prediction_id = record.get("id")
    if prediction_id:
        ctx.state.prediction_ids.append(prediction_id)
    return {"id": prediction_id, "status": record.get("status")}


def build_predict_tools() -> list[RegisteredTool]:
    """Return the ``olmoearth-predict`` tool bundle (search + submit).

    Poll with the foundational ``olmoearth_get_prediction`` tool.
    """
    return [
        RegisteredTool(
            spec=ToolSpec(
                name="olmoearth_search_predictions",
                description=(
                    "Search predictions, optionally scoped to a project. "
                    "Returns id, name, status, and model_id for each. Use "
                    "this to discover a reusable model_id before submitting "
                    "a new prediction. Read-only."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "project_id": {"type": "string"},
                        "limit": {"type": "integer", "default": 50},
                        "offset": {"type": "integer", "default": 0},
                    },
                    "required": [],
                },
            ),
            handler=_search_predictions,
        ),
        RegisteredTool(
            spec=ToolSpec(
                name="olmoearth_submit_prediction",
                description=(
                    "Submit a new prediction. Requires name, project_id, "
                    "area_id, model_id, and a start_time/end_time "
                    "(ISO-8601). Get a model_id by reusing one from "
                    "olmoearth_search_predictions. Returns the new "
                    "prediction id and status; poll it with "
                    "olmoearth_get_prediction."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "project_id": {"type": "string"},
                        "area_id": {"type": "string"},
                        "model_id": {"type": "string"},
                        "start_time": {"type": "string"},
                        "end_time": {"type": "string"},
                    },
                    "required": [
                        "name",
                        "project_id",
                        "area_id",
                        "model_id",
                        "start_time",
                        "end_time",
                    ],
                },
            ),
            handler=_submit_prediction,
        ),
    ]
