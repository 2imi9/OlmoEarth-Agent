# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""The ``olmoearth-uncertainty`` tool bundle (skill #9).

Two tools:

- ``olmoearth_area_of_applicability`` -- the Meyer-Pebesma Area-of-Applicability
  OOD flag over caller-supplied feature vectors.
- ``olmoearth_ensemble_uncertainty`` -- epistemic (ensemble-disagreement)
  confidence: samples two or more prediction *results* on a shared-extent grid
  (Studio pixel-value) and treats the per-point values as ensemble members, so
  the variance is real (distinct results), never a fake re-read of one
  deterministic result.

Both return summary statistics only (rule §3.1).
"""

from __future__ import annotations

import asyncio
from typing import Any

from olmoearth_agent.analysis.aoi import geometry_bbox
from olmoearth_agent.analysis.raster_compare import grid_points, intersect_bbox
from olmoearth_agent.analysis.uncertainty import (
    area_of_applicability,
    prediction_confidence,
)
from olmoearth_agent.llm.types import ToolSpec
from olmoearth_agent.tools.registry import RegisteredTool, ToolContext

_VECTORS_SCHEMA = {
    "type": "array",
    "items": {"type": "array", "items": {"type": "number"}},
}

#: Concurrency for grid pixel-value sampling (bounds load on Studio + proxy).
_SAMPLE_CONCURRENCY = 8


async def _area_of_applicability(
    args: dict[str, Any], _ctx: ToolContext
) -> dict[str, Any]:
    weights = args.get("weights")
    return area_of_applicability(
        [[float(x) for x in v] for v in args["train_features"]],
        [[float(x) for x in v] for v in args["new_features"]],
        weights=[float(w) for w in weights] if weights is not None else None,
    )


def _result_bbox(record: dict[str, Any]) -> list[float] | None:
    """A result's ``[min_lon, min_lat, max_lon, max_lat]`` from its geometry."""
    geom = (record.get("result_metadata") or {}).get("geometry")
    if not geom:
        return None
    try:
        return geometry_bbox(geom)
    except ValueError:
        return None


def _band_val(record: dict[str, Any], prop: str | None) -> tuple[Any, bool]:
    """One pixel-value: ``(value, is_categorical)``; ``(None, False)`` if absent."""
    bands = record.get("bands") or []
    if not bands:
        return None, False
    band = (
        next((b for b in bands if b.get("property_name") == prop), bands[0])
        if prop
        else bands[0]
    )
    cls = band.get("classification")
    if cls is not None:
        return cls, True
    return band.get("raw_value"), False


async def _ensemble_uncertainty(
    args: dict[str, Any], ctx: ToolContext
) -> dict[str, Any]:
    # Distinct results only: a duplicate id would re-read ONE deterministic
    # result and fake zero disagreement -- the exact thing this tool promises
    # it never does. Dedupe preserving order, then require >= 2 remain.
    result_ids = list(dict.fromkeys(args["result_ids"]))
    if len(result_ids) < 2:
        return {
            "comparable": False,
            "reason": "need >= 2 DISTINCT result_ids; duplicate ids re-read one "
            "deterministic result and fake zero disagreement",
        }
    grid = max(2, min(10, int(args.get("grid", 4))))
    prop = args.get("property_name")

    records = await asyncio.gather(
        *[ctx.studio.get_prediction_result(r) for r in result_ids]
    )
    bboxes = [_result_bbox(rec) for rec in records]
    if any(b is None for b in bboxes):
        return {"comparable": False, "reason": "a result is missing geometry/extent"}
    shared = bboxes[0]
    for b in bboxes[1:]:
        shared = intersect_bbox(shared, b)
    if shared is None:
        return {"comparable": False, "reason": "results do not share a common extent"}

    points = grid_points(shared, grid)
    sem = asyncio.Semaphore(_SAMPLE_CONCURRENCY)

    async def sample(rid: str, lon: float, lat: float) -> Any:
        async with sem:
            try:
                return await ctx.studio.pixel_value(rid, lon, lat)
            except Exception:  # off-raster / nodata / transient -> drop the draw
                return None

    flat = await asyncio.gather(
        *[sample(rid, lon, lat) for (lon, lat) in points for rid in result_ids]
    )

    n_res = len(result_ids)
    samples: list[list[Any]] = []
    detected_categorical: bool | None = None
    n_dropped = 0
    for pi in range(len(points)):
        vals: list[Any] = []
        for ri in range(n_res):
            rec = flat[pi * n_res + ri]
            if rec is None:
                continue
            value, is_cat = _band_val(rec, prop)
            if value is None:
                continue
            if detected_categorical is None:
                detected_categorical = is_cat
            vals.append(value)
        # An ensemble needs >= 2 members at a point to measure disagreement.
        if len(vals) >= 2:
            samples.append(vals)
        else:
            n_dropped += 1

    if not samples:
        return {
            "comparable": False,
            "reason": "no grid point had >= 2 overlapping valid samples across "
            "the ensemble",
        }
    value_type = args.get("value_type") or (
        "categorical" if detected_categorical else "regression"
    )
    try:
        out = prediction_confidence(samples, value_type=value_type)
    except (ValueError, TypeError):
        # A regression value_type over non-numeric labels (results disagree on
        # band type) -> a clean answer, not an opaque float() traceback.
        return {
            "comparable": False,
            "reason": f"sampled values are not consistent with value_type "
            f"'{value_type}' (the results may disagree on band type); pass "
            "value_type explicitly to disambiguate",
        }
    out.update(
        {
            "comparable": True,
            "n_results": n_res,
            "grid": f"{grid}x{grid}",
            "property_name": prop,
            "shared_extent_bbox": [round(v, 5) for v in shared],
            "n_points_dropped": n_dropped,
            "method": "each result sampled pointwise (pixel-value) on a grid "
            "over the shared extent; the >=2 values at a point are treated as "
            "ensemble members, so the spread is real disagreement between "
            "distinct results -- epistemic uncertainty, not accuracy. Slow: "
            "pixel-value is ~tens of seconds per point per result.",
        }
    )
    return out


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
        RegisteredTool(
            spec=ToolSpec(
                name="olmoearth_ensemble_uncertainty",
                description=(
                    "Estimate epistemic (ensemble-disagreement) uncertainty "
                    "across TWO OR MORE prediction results over their shared "
                    "extent. Samples each result on a common grid (Studio "
                    "pixel-value) and treats the >=2 values at each point as "
                    "ensemble members, so the spread is REAL disagreement "
                    "between distinct results -- not a fake re-read of one "
                    "deterministic result. Returns per-point + aggregate "
                    "dispersion -> a confidence in [0,1]: regression reports "
                    "mean/std/coefficient-of-variation/range; categorical "
                    "reports majority class, vote fractions, normalized Shannon "
                    "entropy, and margin. This is model self-consistency, NOT a "
                    "calibrated probability of correctness -- pair it with "
                    "olmoearth_area_of_applicability for the OOD regime. "
                    "Read-only; summary stats only. Slow (pixel-value is ~tens "
                    "of seconds per point per result) -- keep grid small."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "result_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 2,
                            "description": "Two or more prediction-result ids to "
                            "treat as an ensemble; sampled over their shared "
                            "(intersected) extent.",
                        },
                        "property_name": {
                            "type": "string",
                            "description": "Band/property to sample; defaults to "
                            "the first band.",
                        },
                        "grid": {
                            "type": "integer",
                            "default": 4,
                            "description": "Grid size N (N*N sample points; "
                            "2-10). Small because pixel-value is slow.",
                        },
                        "value_type": {
                            "type": "string",
                            "enum": ["regression", "categorical"],
                            "description": "How to interpret sampled values; "
                            "auto-detected from the first band if omitted.",
                        },
                    },
                    "required": ["result_ids"],
                },
            ),
            handler=_ensemble_uncertainty,
        ),
    ]
