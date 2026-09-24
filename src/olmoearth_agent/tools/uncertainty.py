# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""The ``olmoearth-uncertainty`` tool bundle (skill #9): the OOD half.

``olmoearth_area_of_applicability`` is the Meyer-Pebesma Area-of-Applicability
OOD flag over caller-supplied feature vectors; it returns summary statistics
only (rule §3.1).

The skill's other half, ensemble disagreement across two or more distinct
Studio results, is ``olmoearth_compare_results`` with ``mode='ensemble'``
(:mod:`olmoearth_agent.tools.compare`), which shares the grid sampler and the
no-data rule with every other comparison.

Scope boundary with skill #18 (``olmoearth-review-set``). These signals
answer *is this prediction in a regime I should trust* -- self-consistency
across distinct results, and distance from the training distribution. They
do **not** rank which windows are wrong: upstream
(`2imi9/olmoearth_inferenceX <https://github.com/2imi9/olmoearth_inferenceX>`_)
measured ensemble disagreement and feature-space typicality against the
model's own top-1-minus-top-2 margin on seven expert-labelled testbeds and
neither ranked errors better than the margin. When per-class scores exist,
route "which windows are wrong" to ``olmoearth_review_set``; for a Studio
result whose band is a binary score in ``[0, 1]``, to
``olmoearth_review_set_from_result``.
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
    """Handler for ``olmoearth_area_of_applicability``."""
    weights = args.get("weights")
    return area_of_applicability(
        [[float(x) for x in v] for v in args["train_features"]],
        [[float(x) for x in v] for v in args["new_features"]],
        weights=[float(w) for w in weights] if weights is not None else None,
    )


def build_uncertainty_tools() -> list[RegisteredTool]:
    """Return the ``olmoearth-uncertainty`` tool bundle (the AOA flag)."""
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
                    "This is the OOD question, NOT the error-ranking "
                    "question: to order windows by how likely each one is "
                    "WRONG, use olmoearth_review_set (measured: distance to "
                    "the training features has no scale-free advantage over "
                    "the model's own margin, 13/27 scenes, sign p=1.00). Use "
                    "both together: trust a prediction most when it is inside "
                    "the AOA and high-margin. "
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
