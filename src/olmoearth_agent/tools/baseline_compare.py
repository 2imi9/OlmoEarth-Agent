# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""The ``olmoearth-baseline-compare`` tool bundle (skill #7).

Side-by-side OlmoEarth-vs-AlphaEarth comparison on shared labels / AOI:
a per-metric table (always) plus an optional cell-by-cell difference
raster. Operates on predictions/scores the caller already has. The
AlphaEarth side is data the user exported from the public Google Earth
Engine "Satellite Embedding" dataset (no live GEE connection / MCP).
"""

from __future__ import annotations

from typing import Any

from olmoearth_agent.analysis.baseline import compare_metrics, difference_raster
from olmoearth_agent.llm.types import ToolSpec
from olmoearth_agent.tools.registry import RegisteredTool, ToolContext

_LABELS = {
    "type": "array",
    "items": {"type": ["integer", "string"]},
    "description": "Class labels: integer codes or string names, one per pixel.",
}
_SCORES = {"type": "array", "items": {"type": "number"}}


async def _baseline_compare(args: dict[str, Any], _ctx: ToolContext) -> dict[str, Any]:
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
    return result


def build_baseline_compare_tools() -> list[RegisteredTool]:
    """Return the ``olmoearth-baseline-compare`` tool bundle."""
    return [
        RegisteredTool(
            spec=ToolSpec(
                name="olmoearth_baseline_compare",
                description=(
                    "Compare OlmoEarth vs AlphaEarth head-to-head on the "
                    "same ground truth. Pass y_true plus each model's "
                    "predicted labels (olmoearth_pred, alphaearth_pred); "
                    "returns a per-metric table (accuracy / macro-F1 / "
                    "mean-IoU) with deltas and a winner, and an "
                    "overall_winner. Optionally pass aligned score layers "
                    "(olmoearth_layer + alphaearth_layer) for a cell-by-cell "
                    "difference raster. Use this to substantiate an "
                    "'outperforms AlphaEarth' claim in a transfer region "
                    "(Ma et al. arXiv:2601.00857). The AlphaEarth side is "
                    "data you exported from the GEE Satellite Embedding "
                    "dataset. Read-only."
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
        ),
    ]
