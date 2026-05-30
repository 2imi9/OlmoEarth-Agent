# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 OlmoEarth Agent contributors
"""The ``olmoearth-change-detect`` tool bundle (skill #6).

A multi-date trajectory diff over per-date layer summaries. Composes
skill #5 (``olmoearth-predict``): the agent fetches a prediction result
per date, reduces each to one ``value``, and passes the dated series
here. Refuses fewer than three dates — a two-date diff hides drift.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from olmoearth_agent.analysis.change_detect import diff_layers
from olmoearth_agent.llm.types import ToolSpec
from olmoearth_agent.tools.registry import RegisteredTool, ToolContext


def _is_iso(value: Any) -> bool:
    try:
        datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return True
    except (ValueError, AttributeError):
        return False


async def _change_detect(args: dict[str, Any], _ctx: ToolContext) -> dict[str, Any]:
    points = args["series"]
    # Dates are optional. If any point lacks a parseable ISO date — a brief that
    # gave bare ordered values, or a garbled model-emitted date — label the
    # points by input order (t1..tN). The trajectory math only needs order, not
    # calendar dates, so this avoids erroring and burning a retry turn.
    if all(_is_iso(p.get("date")) for p in points):
        series = [(str(p["date"]), float(p["value"])) for p in points]
    else:
        series = [(f"t{i + 1}", float(p["value"])) for i, p in enumerate(points)]
    result = diff_layers(series, tol=float(args.get("tol", 0.0)))
    metric = args.get("metric")
    if metric:
        result["metric"] = str(metric)
    return result


def build_change_detect_tools() -> list[RegisteredTool]:
    """Return the ``olmoearth-change-detect`` tool bundle."""
    return [
        RegisteredTool(
            spec=ToolSpec(
                name="olmoearth_change_detect",
                description=(
                    "Compute a multi-date change trajectory from >= 3 dated "
                    "layer summaries. Each series point is one prediction "
                    "date reduced to a single value (e.g. positive-class "
                    "fraction or mean score over the AOI, from a skill #5 "
                    "result). Returns per-step deltas, net change, the "
                    "largest-change interval, a reversal count, and a trend "
                    "label (increasing/decreasing/stable/oscillating). "
                    "REFUSES fewer than 3 distinct dates: a two-date diff "
                    "reports net change but cannot tell a steady trend from "
                    "a reversal (a flood that peaked then receded), so always "
                    "run >= 3 dates. tol treats tiny steps as no-change for "
                    "the trend label. Pure/read-only."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "series": {
                            "type": "array",
                            "minItems": 3,
                            "items": {
                                "type": "object",
                                "properties": {
                                    "date": {
                                        "type": "string",
                                        "description": (
                                            "OPTIONAL ISO-8601 date/datetime. "
                                            "Omit it if you only have ordered "
                                            "values — points are then taken in "
                                            "the order given. Do not invent dates."
                                        ),
                                    },
                                    "value": {"type": "number"},
                                },
                                "required": ["value"],
                            },
                        },
                        "metric": {
                            "type": "string",
                            "description": "Optional label for the summarized value.",
                        },
                        "tol": {"type": "number", "default": 0.0},
                    },
                    "required": ["series"],
                },
            ),
            handler=_change_detect,
        ),
    ]
