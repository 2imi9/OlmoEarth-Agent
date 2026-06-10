# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""The ``olmoearth-predict`` tool bundle (skill #4): the core run loop.

Search predictions (to discover reusable ``model_id``s), submit a new
prediction, and poll it. Polling reuses the foundational
``olmoearth_get_prediction`` tool. ``olmoearth_pixel_value`` reads the
model's output at a single point; feature-search remains a follow-up
within this skill.
"""

from __future__ import annotations

import asyncio
from typing import Any

from olmoearth_agent.analysis.aoi import geometry_bbox
from olmoearth_agent.analysis.raster_compare import (
    compare_categorical,
    compare_narration,
    compare_numeric,
    grid_points,
    intersect_bbox,
    normalize_kind,
)
from olmoearth_agent.llm.types import ToolSpec
from olmoearth_agent.tools.registry import RegisteredTool, ToolContext

#: Concurrency for grid pixel-value sampling (bounds load on Studio + proxy).
_SAMPLE_CONCURRENCY = 8


def _result_bbox(record: dict[str, Any]) -> list[float] | None:
    """A result's [min_lon,min_lat,max_lon,max_lat] from result_metadata.geometry."""
    geom = (record.get("result_metadata") or {}).get("geometry")
    if not geom:
        return None
    try:
        return geometry_bbox(geom)
    except ValueError:
        return None


def _select_band(
    record: dict[str, Any], property_name: str | None
) -> dict[str, Any] | None:
    """Pick the pixel-value band to read: the named property, else the first."""
    bands: list[dict[str, Any]] = record.get("bands") or []
    if not bands:
        return None
    if property_name:
        return next(
            (b for b in bands if b.get("property_name") == property_name), bands[0]
        )
    return bands[0]


def _band_value(record: dict[str, Any], property_name: str | None) -> Any:
    """Extract a pixel-value band's value: ``classification`` if set, else
    ``raw_value``. Picks the named property, or the first band."""
    band = _select_band(record, property_name)
    if band is None:
        return None
    cls = band.get("classification")
    return cls if cls is not None else band.get("raw_value")


def _is_categorical(record: dict[str, Any], property_name: str | None) -> bool:
    band = _select_band(record, property_name)
    return band is not None and band.get("classification") is not None


async def _compare_results(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    a_id = args["result_id_a"]
    b_id = args["result_id_b"]
    prop = args.get("property_name")
    grid = max(2, min(12, int(args.get("grid", 6))))
    tol = float(args.get("tolerance", 0.1))
    kind = normalize_kind(args.get("kind"))

    rec_a = await ctx.studio.get_prediction_result(a_id)
    rec_b = await ctx.studio.get_prediction_result(b_id)
    bbox = intersect_bbox(_result_bbox(rec_a), _result_bbox(rec_b))
    if bbox is None:
        return {
            "comparable": False,
            "reason": "the two results have no overlapping extent (or missing bounds)",
        }
    points = grid_points(bbox, grid)

    sem = asyncio.Semaphore(_SAMPLE_CONCURRENCY)

    async def sample(result_id: str, lon: float, lat: float) -> Any:
        async with sem:
            try:
                rec = await ctx.studio.pixel_value(result_id, lon, lat)
                return rec
            except Exception:  # off-raster / nodata / transient -> drop the point
                return None

    recs_a = await asyncio.gather(*[sample(a_id, lo, la) for lo, la in points])
    recs_b = await asyncio.gather(*[sample(b_id, lo, la) for lo, la in points])

    # Detect categorical vs regression from the first valid sample.
    first = next((r for r in recs_a if r), None)
    categorical = bool(first and _is_categorical(first, prop))
    vals_a = [(_band_value(r, prop) if r else None) for r in recs_a]
    vals_b = [(_band_value(r, prop) if r else None) for r in recs_b]
    pairs = list(zip(vals_a, vals_b))
    stats = (
        compare_categorical(pairs)
        if categorical
        else compare_numeric(pairs, tolerance=tol)
    )
    value_type = "classification" if categorical else "regression"
    narration = compare_narration(stats, kind=kind, value_type=value_type)
    return {
        "comparable": True,
        "result_id_a": a_id,
        "result_id_b": b_id,
        "property_name": prop or (first.get("bands", [{}])[0].get("property_name") if first else None),
        "kind": kind,
        "value_type": value_type,
        "narration": narration,
        "grid": f"{grid}x{grid}",
        "samples_requested": len(points),
        "shared_extent_bbox": [round(v, 5) for v in bbox],
        "stats": stats,
        "method": "pointwise pixel-value sampled on a grid over the shared "
        "extent (an estimate, not every pixel); no ground truth, so this is "
        + narration["framing"]
        + ".",
    }


async def _search_predictions(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    project_id = args.get("project_id")
    env = await ctx.studio.search_predictions(
        project_id=project_id,
        limit=int(args.get("limit", 50)),
        offset=int(args.get("offset", 0)),
    )
    # With a project_id filter the match list is built client-side, so
    # env.total (the unfiltered server total) would mislead; report the
    # actual returned count instead.
    return {
        "total": len(env.records) if project_id else env.total,
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


async def _submit_prediction(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
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


def _summarize_result(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "result_id": record.get("id"),
        "prediction_id": record.get("prediction_id"),
        "tile_urls": record.get("tile_urls"),
        "property_names": record.get("property_names"),
        "file_format": record.get("file_format"),
    }


async def _fetch_results(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    env = await ctx.studio.search_prediction_results(
        prediction_id=args["prediction_id"],
        limit=int(args.get("scan_limit", 200)),
    )
    return {
        "prediction_id": args["prediction_id"],
        "result_count": len(env.records),
        "results": [_summarize_result(r) for r in env.records],
    }


async def _get_prediction_result(
    args: dict[str, Any], ctx: ToolContext
) -> dict[str, Any]:
    record = await ctx.studio.get_prediction_result(args["result_id"])
    summary = _summarize_result(record)
    summary["result_metadata"] = record.get("result_metadata")
    return summary


async def _pixel_value(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    """Read one prediction result's model output at a single coordinate.

    The user supplies the point, so echoing it back with the value is rule
    §3.1-compliant (no agent-derived geometry is leaked). Off-raster / nodata
    / a transient failure resolve to ``available: False`` with a reason rather
    than an exception, so the model gets a clean "no value there" answer.
    """
    result_id = args["result_id"]
    lon = float(args["lon"])
    lat = float(args["lat"])
    prop = args.get("property_name")
    point = {"lon": round(lon, 6), "lat": round(lat, 6)}
    try:
        record = await ctx.studio.pixel_value(result_id, lon, lat)
    except Exception as exc:  # off-raster / nodata / transient -> clean answer
        return {
            "result_id": result_id,
            "queried_point": point,
            "available": False,
            "reason": "pixel-value request failed (off-raster, nodata, or "
            f"transient): {str(exc)[:200]}",
        }
    band = _select_band(record, prop)
    if band is None:
        return {
            "result_id": result_id,
            "queried_point": point,
            "available": False,
            "reason": "no value at this point (off-raster or nodata)",
        }
    value = _band_value(record, prop)
    categorical = _is_categorical(record, prop)
    out: dict[str, Any] = {
        "result_id": result_id,
        "queried_point": point,
        "available": value is not None,
        "property_name": band.get("property_name") or prop,
        "value_type": "classification" if categorical else "regression",
        "value": value,
        "bands": [
            {
                "property_name": b.get("property_name"),
                "value": (
                    b.get("classification")
                    if b.get("classification") is not None
                    else b.get("raw_value")
                ),
            }
            for b in (record.get("bands") or [])
        ],
    }
    if value is None:
        # Band exists but carries no value here -- report it like the other
        # unavailable paths instead of a bare available=false.
        out["reason"] = "band present but no value at this point (nodata)"
    return out


def build_predict_tools() -> list[RegisteredTool]:
    """Return the ``olmoearth-predict`` tool bundle (search + submit + results).

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
        RegisteredTool(
            spec=ToolSpec(
                name="olmoearth_fetch_results",
                description=(
                    "Fetch the output results for a prediction: tile URLs "
                    "(XYZ/MVT map layers), property names, and file format. "
                    "Scans recent prediction-results and filters to this "
                    "prediction (the API has no server-side prediction_id "
                    "filter). Increase scan_limit if results are older."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "prediction_id": {"type": "string"},
                        "scan_limit": {"type": "integer", "default": 200},
                    },
                    "required": ["prediction_id"],
                },
            ),
            handler=_fetch_results,
        ),
        RegisteredTool(
            spec=ToolSpec(
                name="olmoearth_get_prediction_result",
                description=(
                    "Fetch one prediction-result by its result id: tile "
                    "URLs, property names, result metadata, and file format."
                ),
                parameters={
                    "type": "object",
                    "properties": {"result_id": {"type": "string"}},
                    "required": ["result_id"],
                },
            ),
            handler=_get_prediction_result,
        ),
        RegisteredTool(
            spec=ToolSpec(
                name="olmoearth_pixel_value",
                description=(
                    "Read ONE prediction result's model output at a single "
                    "lon/lat point (the Studio /pixel-value endpoint). Returns "
                    "the value (raw_value for a regression layer, the class for "
                    "a categorical layer), the value_type, and every band's "
                    "value at that point. If the point is off-raster or nodata, "
                    "returns available=false with a reason. Use this to answer "
                    "'what does the model predict at this exact location?'; to "
                    "compare TWO results over an area use olmoearth_compare_results."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "result_id": {"type": "string"},
                        "lon": {"type": "number"},
                        "lat": {"type": "number"},
                        "property_name": {
                            "type": "string",
                            "description": "Band/property to read; defaults to "
                            "the first band.",
                        },
                    },
                    "required": ["result_id", "lon", "lat"],
                },
            ),
            handler=_pixel_value,
        ),
        RegisteredTool(
            spec=ToolSpec(
                name="olmoearth_compare_results",
                description=(
                    "Quantitatively compare TWO prediction results over their "
                    "shared area, with no ground truth. Samples both rasters on "
                    "a grid (pointwise pixel-value) and returns mean difference, "
                    "mean-absolute difference, RMSE, correlation, and an "
                    "agreement fraction (regression) or class agreement "
                    "(classification). Set kind='cross_model' (default) to "
                    "compare TWO different models over the same area (how much "
                    "they agree -- divergence, not accuracy); set kind='temporal' "
                    "to compare ONE model's output at an earlier (A) vs a later "
                    "(B) date (net change over time, later minus earlier). The "
                    "returned narration adapts to the kind. Use this for a "
                    "numeric comparison instead of only a visual / metadata one; "
                    "for accuracy against labels use "
                    "olmoearth_classification_metrics."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "result_id_a": {"type": "string"},
                        "result_id_b": {"type": "string"},
                        "kind": {
                            "type": "string",
                            "enum": ["cross_model", "temporal"],
                            "default": "cross_model",
                            "description": "cross_model = two models, same area "
                            "(agreement); temporal = one model, earlier (A) vs "
                            "later (B) date (change over time).",
                        },
                        "property_name": {
                            "type": "string",
                            "description": "Band/property to compare; defaults to "
                            "the first band.",
                        },
                        "grid": {
                            "type": "integer",
                            "default": 6,
                            "description": "Grid size N (N*N sample points; 2-12).",
                        },
                        "tolerance": {
                            "type": "number",
                            "default": 0.1,
                            "description": "Regression: |a-b| <= tolerance counts "
                            "as agreement.",
                        },
                    },
                    "required": ["result_id_a", "result_id_b"],
                },
            ),
            handler=_compare_results,
        ),
    ]
