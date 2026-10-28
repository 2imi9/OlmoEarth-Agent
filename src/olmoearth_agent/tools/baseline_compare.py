# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""The ``olmoearth-baseline-compare`` tool bundle (skill #6, opt-in).

Side-by-side OlmoEarth-vs-AlphaEarth comparison on shared labels / AOI:
a per-metric table (always) plus an optional cell-by-cell difference
raster. Operates on predictions/scores the caller already has. The
AlphaEarth side is data the user exported from the public Google Earth
Engine "Satellite Embedding" dataset (no live GEE connection / MCP).

Its winner is a point metric on whatever labels the caller gives, with no
design, no interval and no test, so the tool is opt-in
(``OLMOEARTH_OPT_IN_TOOLS=1``), and every result forbids reading the winner
as the more accurate map: which of two maps is more accurate is the
which-map route (``rules.WHICH_MAP_ROUTE``).
"""

from __future__ import annotations

from typing import Any

from olmoearth_agent.analysis.baseline import compare_metrics, difference_raster
from olmoearth_agent.llm.types import ToolSpec
from olmoearth_agent.tools import statistical_rules as rules
from olmoearth_agent.tools.registry import Capability, RegisteredTool, ToolContext

#: What the table's winner is, and is not, stated with every result.
WINNER_NOTE = (
    "The winner is the higher point metric on the labels given, with no design, "
    "no interval and no test: it does not show which map is more accurate. "
    f"For that: {rules.WHICH_MAP_ROUTE}."
)

_LABELS = {
    "type": "array",
    "items": {"type": ["integer", "string"]},
    "description": "Class labels: integer codes or string names, one per pixel.",
}
_SCORES = {"type": "array", "items": {"type": "number"}}


async def _baseline_compare(args: dict[str, Any], _ctx: ToolContext) -> dict[str, Any]:
    """Handler for ``olmoearth_baseline_compare``: the table, its note and its claim."""
    label_a = str(args.get("olmoearth_label", "olmoearth"))
    label_b = str(args.get("alphaearth_label", "alphaearth"))
    result = compare_metrics(
        args["y_true"],
        args["olmoearth_pred"],
        args["alphaearth_pred"],
        label_a=label_a,
        label_b=label_b,
    )
    olmo_layer = args.get("olmoearth_layer")
    alpha_layer = args.get("alphaearth_layer")
    if olmo_layer is not None and alpha_layer is not None:
        result["difference_raster"] = difference_raster(
            [float(x) for x in olmo_layer],
            [float(x) for x in alpha_layer],
            label_a=label_a,
            label_b=label_b,
        )
    result["winner_note"] = WINNER_NOTE
    return rules.add_contract(
        result,
        forbidden_claims=[
            rules.more_accurate_without_its_test(
                f"this table's winner ({result.get('overall_winner')}) compares "
                "point metrics on the labels given, with no design, interval or "
                "test, so it names no map more accurate"
            )
        ],
    )


def build_baseline_compare_tools() -> list[RegisteredTool]:
    """Return the ``olmoearth-baseline-compare`` tool bundle."""
    return [
        RegisteredTool(
            spec=ToolSpec(
                name="olmoearth_baseline_compare",
                description=(
                    "Compare OlmoEarth vs a baseline foundation model "
                    "(e.g. AlphaEarth) head-to-head on the "
                    "same ground truth. Pass y_true plus each model's "
                    "predicted labels (olmoearth_pred, alphaearth_pred); "
                    "returns a per-metric table (accuracy / macro-F1 / "
                    "mean-IoU) with deltas and a winner, and an "
                    "overall_winner. Optionally pass aligned score layers "
                    "(olmoearth_layer + alphaearth_layer) for a cell-by-cell "
                    "difference raster. The winner is a point metric with no "
                    "design, interval or test, so it does not show which map is "
                    "more accurate: for that, olmoearth_plan_label_sample with "
                    "design='disagreement', then olmoearth_estimate_map_error. "
                    "The baseline side (e.g. "
                    "AlphaEarth) is data/predictions you supply (e.g. exported "
                    "from the GEE Satellite Embedding dataset). Read-only."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "y_true": _LABELS,
                        "olmoearth_pred": _LABELS,
                        "alphaearth_pred": _LABELS,
                        "olmoearth_layer": _SCORES,
                        "alphaearth_layer": _SCORES,
                        "olmoearth_label": {"type": "string"},
                        "alphaearth_label": {"type": "string"},
                    },
                    "required": ["y_true", "olmoearth_pred", "alphaearth_pred"],
                },
            ),
            handler=_baseline_compare,
            capability=Capability(
                does="OlmoEarth against a baseline model's predictions, on "
                "ground-truth labels the caller gives; its winner is a point "
                "metric, untested"
            ),
        ),
    ]
