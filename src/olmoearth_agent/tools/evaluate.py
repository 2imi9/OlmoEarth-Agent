# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""The ``olmoearth-evaluate`` tool bundle (skill #7).

Honest map-accuracy tools: the random-vs-spatial CV inflation check,
classification metrics, and NNDM-LOO cross-validation (Mila et al. 2022, a
faithful port of CAST's ``nndm``) for an unbiased accuracy estimate over the
actual prediction area.
"""

from __future__ import annotations

import json
from typing import Any

from olmoearth_agent.evaluation.metrics import classification_metrics
from olmoearth_agent.evaluation.nndm import nndm_loo
from olmoearth_agent.evaluation.spatial_cv import cv_inflation_diagnostic
from olmoearth_agent.llm.types import ToolSpec
from olmoearth_agent.security.paths import safe_path
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


async def _nndm_cv(args: dict[str, Any], _ctx: ToolContext) -> dict[str, Any]:
    points = [(float(p[0]), float(p[1])) for p in args["points"]]
    pred_points = [(float(p[0]), float(p[1])) for p in args["pred_points"]]
    phi = args.get("phi")
    result = nndm_loo(
        points,
        pred_points,
        phi=float(phi) if phi is not None else None,
        min_train=float(args.get("min_train", 0.5)),
    )
    folds = result.pop("folds")
    result.pop("gjstar_km", None)  # internal diagnostic, not surfaced
    output_path = args.get("output_path")
    if output_path:
        # Folds are integer indices (not geometry); writing them to a file keeps
        # a large training set's per-fold lists out of the chat (rule 1).
        # output_path is model-controlled -> confine it to the workspace root so
        # a traversing/absolute path can't write outside it.
        dest = safe_path(output_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "phi_km": result["phi_km"],
            "min_train": result["min_train"],
            "folds": folds,
        }
        dest.write_text(json.dumps(payload), encoding="utf-8")
        result["folds_path"] = str(dest)
        result["folds_written"] = len(folds)
    else:
        result["folds"] = folds
    return result


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
        RegisteredTool(
            spec=ToolSpec(
                name="olmoearth_nndm_cv",
                description=(
                    "Build NNDM Leave-One-Out cross-validation folds (Nearest "
                    "Neighbour Distance Matching, Mila et al. 2022; a port of R "
                    "CAST's nndm) for an UNBIASED map-accuracy estimate. Give the "
                    "training label points AND a sample of points from the "
                    "prediction area; it matches the CV test-to-train distance "
                    "distribution to the prediction-to-train one by excluding "
                    "near neighbours per fold. Returns the optimistic-bias before "
                    "(LOO) vs after (NNDM) and the per-fold train/test/exclude "
                    "indices. Use when reporting accuracy for a map over an area "
                    "larger or sparser than the labels (stronger than the "
                    "random-vs-spatial inflation check). Pass output_path to "
                    "write the folds to a file for a large training set."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "points": _POINTS_SCHEMA,
                        "pred_points": {
                            **_POINTS_SCHEMA,
                            "description": (
                                "Sample of [lon, lat] points from the prediction "
                                "area (e.g. raster-cell centroids)."
                            ),
                        },
                        "phi": {
                            "type": "number",
                            "description": (
                                "Autocorrelation range in km to match within; "
                                "omit to use the maximum observed distance."
                            ),
                        },
                        "min_train": {"type": "number", "default": 0.5},
                        "output_path": {
                            "type": "string",
                            "description": (
                                "Optional file path to write the per-fold indices "
                                "to (JSON), instead of returning them inline."
                            ),
                        },
                    },
                    "required": ["points", "pred_points"],
                },
            ),
            handler=_nndm_cv,
        ),
    ]
