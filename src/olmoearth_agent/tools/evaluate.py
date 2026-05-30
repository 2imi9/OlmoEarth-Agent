# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 OlmoEarth Agent contributors
"""The ``olmoearth-evaluate`` tool bundle (skill #8).

Honest map-accuracy tools: the random-vs-spatial CV inflation check and
classification metrics. NNDM-LOO (Milà et al. 2022) is the remaining
piece of this skill: a documented follow-up.
"""

from __future__ import annotations

from typing import Any

from olmoearth_agent.evaluation.metrics import classification_metrics
from olmoearth_agent.evaluation.spatial_cv import cv_inflation_diagnostic
from olmoearth_agent.llm.types import ToolSpec
from olmoearth_agent.tools.registry import RegisteredTool, ToolContext

_POINTS_SCHEMA = {
    "type": "array",
    "items": {
        "type": "array",
        "items": {"type": "number"},
        "minItems": 2,
        "maxItems": 2,
    },
    "description": "List of [lon, lat] coordinate pairs in degrees.",
}

_LABEL_LIST = {
    "type": "array",
    "items": {"type": ["integer", "string"]},
    "description": "Class labels: integer codes or string names, one per pixel.",
}


async def _cv_inflation_check(
    args: dict[str, Any], _ctx: ToolContext
) -> dict[str, Any]:
    points = [(float(p[0]), float(p[1])) for p in args["points"]]
    return cv_inflation_diagnostic(
        points,
        n_folds=int(args.get("n_folds", 5)),
        block_deg=float(args.get("block_deg", 0.5)),
    )


async def _classification_metrics(
    args: dict[str, Any], _ctx: ToolContext
) -> dict[str, Any]:
    return classification_metrics(args["y_true"], args["y_pred"])


def build_evaluate_tools() -> list[RegisteredTool]:
    """Return the ``olmoearth-evaluate`` tool bundle."""
    return [
        RegisteredTool(
            spec=ToolSpec(
                name="olmoearth_cv_inflation_check",
                description=(
                    "Check whether RANDOM cross-validation would overstate "
                    "map accuracy on these label points, vs spatial-block "
                    "CV. Returns the mean test-to-train distance under each "
                    "and their ratio (Ploton 2020 / Meyer-Pebesma effect). "
                    "Run this BEFORE reporting any accuracy number, and use "
                    "spatial-block CV when the ratio is >= 1.5."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "points": _POINTS_SCHEMA,
                        "n_folds": {"type": "integer", "default": 5},
                        "block_deg": {"type": "number", "default": 0.5},
                    },
                    "required": ["points"],
                },
            ),
            handler=_cv_inflation_check,
        ),
        RegisteredTool(
            spec=ToolSpec(
                name="olmoearth_classification_metrics",
                description=(
                    "Compute accuracy plus per-class precision / recall / "
                    "F1 / IoU, macro-F1, and mean IoU from aligned "
                    "ground-truth and predicted label lists."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "y_true": _LABEL_LIST,
                        "y_pred": _LABEL_LIST,
                    },
                    "required": ["y_true", "y_pred"],
                },
            ),
            handler=_classification_metrics,
        ),
    ]
