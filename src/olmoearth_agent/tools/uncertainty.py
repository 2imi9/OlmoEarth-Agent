# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""The ``olmoearth-uncertainty`` tool bundle (skill #10).

Exposes the Meyer-Pebesma Area-of-Applicability OOD flag. Operates on
feature vectors the caller already has (training-data features + the AOI
points to assess); returns summary statistics only (rule §3.1).
"""

from __future__ import annotations

from typing import Any

from olmoearth_agent.analysis.uncertainty import area_of_applicability
from olmoearth_agent.llm.types import ToolSpec
from olmoearth_agent.tools.registry import RegisteredTool, ToolContext

_VECTORS_SCHEMA = {
    "type": "array",
    "items": {"type": "array", "items": {"type": "number"}},
}


async def _area_of_applicability(
    args: dict[str, Any], _ctx: ToolContext
) -> dict[str, Any]:
    weights = args.get("weights")
    return area_of_applicability(
        [[float(x) for x in v] for v in args["train_features"]],
        [[float(x) for x in v] for v in args["new_features"]],
        weights=[float(w) for w in weights] if weights is not None else None,
    )


def build_uncertainty_tools() -> list[RegisteredTool]:
    """Return the ``olmoearth-uncertainty`` tool bundle."""
    return [
        RegisteredTool(
            spec=ToolSpec(
                name="olmoearth_area_of_applicability",
                description=(
                    "Flag which AOI points fall OUTSIDE a model's Area of "
                    "Applicability (Meyer-Pebesma 2021). Pass the training "
                    "data's feature vectors and the new points' feature "
                    "vectors (predictors or embeddings); returns each new "
                    "point's dissimilarity index, whether it is inside the "
                    "AOA, the OOD fraction, and a verdict. Use this for OOD "
                    "detection: softmax confidence is NOT OOD detection: a "
                    "model can be confidently wrong on data unlike its "
                    "training set. Optional per-feature importance weights. "
                    "Read-only; summary stats only."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "train_features": _VECTORS_SCHEMA,
                        "new_features": _VECTORS_SCHEMA,
                        "weights": {
                            "type": "array",
                            "items": {"type": "number"},
                            "description": (
                                "Optional per-feature importance weights "
                                "(length = feature dimension)."
                            ),
                        },
                    },
                    "required": ["train_features", "new_features"],
                },
            ),
            handler=_area_of_applicability,
        ),
    ]
