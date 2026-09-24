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

from olmoearth_agent.analysis.raster_compare import (
    band_is_nodata,
    compare_categorical,
    compare_group_categorical,
    compare_group_narration,
    compare_group_numeric,
    compare_narration,
    compare_numeric,
    declared_fields,
    declared_range,
    grid_points,
    intersect_bbox,
    intersect_bboxes,
    normalize_kind,
    result_bbox,
)
from olmoearth_agent.llm.types import ToolSpec
from olmoearth_agent.tools.registry import RegisteredTool, ToolContext
from olmoearth_agent.tools.sampling import (
    FAILED,
    NODATA,
    OK,
    SAMPLE_CONCURRENCY,
    ResultNodata,
    band_value,
    default_property,
    is_categorical,
    model_summary,
    read_sample,
    result_nodata_context,
    sample_records,
    select_band,
)

#: Concurrency for grid pixel-value sampling (bounds load on Studio + proxy).
_SAMPLE_CONCURRENCY = SAMPLE_CONCURRENCY

#: Kept for importers of the pre-``sampling`` helper names.
_select_band = select_band
_band_value = band_value
_is_categorical = is_categorical

#: How a sampled value is judged no-data, stated in every sampling result.
NODATA_RULE = (
    "a sample is no-data, and never enters a statistic, when its band is "
    "missing or NaN, equals the model's nodata_value, lies outside the band's "
    "declared regression range, or carries no class on a classification band"
)


def _property_summary(name: str | None, nodata: ResultNodata) -> dict[str, Any]:
    """A property's name and what its result declares about it (no geometry)."""
    out: dict[str, Any] = {"property_name": name}
    out.update(nodata.declared.get(str(name), {}))
    return out


def _sampled_property(
    reads: list[tuple[Any, bool, str]], recs: list[Any], prop: str | None
) -> str | None:
    """The property name the sampled bands actually carried (first usable one)."""
    for rec, (_value, _cat, status) in zip(recs, reads):
        if rec is not None and status != FAILED:
            band = select_band(rec, prop)
            if band is not None and isinstance(band.get("property_name"), str):
                return str(band["property_name"])
    return None


def _different_properties(
    a_id: str, b_id: str, prop_a: dict[str, Any], prop_b: dict[str, Any]
) -> dict[str, Any]:
    """The refusal for two results that measure different properties."""
    return {
        "comparable": False,
        "result_id_a": a_id,
        "result_id_b": b_id,
        "property_a": prop_a,
        "property_b": prop_b,
        "reason": (
            f"the two results measure different properties "
            f"({prop_a.get('property_name')!r} and {prop_b.get('property_name')!r}); "
            "a difference, an RMSE or an agreement fraction between them "
            "mixes two quantities. Compare results of the same property, or "
            "pass allow_different_properties=true to read only whether they "
            "rise and fall together (the correlation)."
        ),
    }


async def _compare_results(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    """Handler for ``olmoearth_compare_results``."""
    a_id = args["result_id_a"]
    b_id = args["result_id_b"]
    prop = args.get("property_name")
    grid = max(2, min(12, int(args.get("grid", 6))))
    tol = float(args.get("tolerance", 0.1))
    kind = normalize_kind(args.get("kind"))
    allow_different = bool(args.get("allow_different_properties", False))

    rec_a = await ctx.studio.get_prediction_result(a_id)
    rec_b = await ctx.studio.get_prediction_result(b_id)
    bbox = intersect_bbox(result_bbox(rec_a), result_bbox(rec_b))
    if bbox is None:
        return {
            "comparable": False,
            "reason": "the two results have no overlapping extent (or missing bounds)",
        }
    cache: dict[str, dict[str, Any] | None] = {}
    nd_a = await result_nodata_context(ctx, rec_a, cache=cache)
    nd_b = await result_nodata_context(ctx, rec_b, cache=cache)

    # Different properties are refused BEFORE any slow sampling, from what the
    # two result records declare; the sampled bands re-check it below.
    name_a, name_b = default_property(rec_a, prop), default_property(rec_b, prop)
    if name_a and name_b and name_a != name_b and not allow_different:
        return _different_properties(
            a_id, b_id, _property_summary(name_a, nd_a), _property_summary(name_b, nd_b)
        )

    points = grid_points(bbox, grid)
    sem = asyncio.Semaphore(_SAMPLE_CONCURRENCY)
    recs_a = await sample_records(ctx, a_id, points, sem)
    recs_b = await sample_records(ctx, b_id, points, sem)
    reads_a = [read_sample(r, prop, nd_a) for r in recs_a]
    reads_b = [read_sample(r, prop, nd_b) for r in recs_b]

    name_a = _sampled_property(reads_a, recs_a, prop) or name_a
    name_b = _sampled_property(reads_b, recs_b, prop) or name_b
    different = bool(name_a and name_b and name_a != name_b)
    if different and not allow_different:
        return _different_properties(
            a_id, b_id, _property_summary(name_a, nd_a), _property_summary(name_b, nd_b)
        )

    pairs: list[tuple[Any, Any]] = []
    n_nodata = n_failed = 0
    for (va, _ca, sa), (vb, _cb, sb) in zip(reads_a, reads_b):
        if sa == OK and sb == OK:
            pairs.append((va, vb))
        elif NODATA in (sa, sb):
            n_nodata += 1
        else:
            n_failed += 1
    first = next((r for r in reads_a + reads_b if r[2] == OK), None)
    categorical = bool(first and first[1])
    stats = (
        compare_categorical(pairs)
        if categorical
        else compare_numeric(pairs, tolerance=tol)
    )
    value_type = "classification" if categorical else "regression"
    narration = compare_narration(stats, kind=kind, value_type=value_type)
    out: dict[str, Any] = {
        "comparable": True,
        "result_id_a": a_id,
        "result_id_b": b_id,
        "property_name": prop or name_a,
        "property_a": _property_summary(name_a, nd_a),
        "property_b": _property_summary(name_b, nd_b),
        "kind": kind,
        "value_type": value_type,
        "narration": narration,
        "grid": f"{grid}x{grid}",
        "samples_requested": len(points),
        "n_nodata_dropped": n_nodata,
        "n_failed_dropped": n_failed,
        "nodata_rule": NODATA_RULE,
        "shared_extent_bbox": [round(v, 5) for v in bbox],
        "stats": stats,
        "method": "pointwise pixel-value sampled on a grid over the shared "
        "extent (an estimate, not every pixel); points where either map is "
        "no-data are dropped before any statistic; no ground truth, so this is "
        + narration["framing"]
        + ".",
    }
    if different:
        out["warning"] = (
            f"the two results measure different properties ({name_a!r} and "
            f"{name_b!r}), so their values are different quantities: only the "
            "correlation (whether they rise and fall together) is meaningful. "
            "Do not read the mean difference, RMSE or agreement fraction as a "
            "gap between them, and do not say one reads higher than the other."
        )
        r = stats.get("correlation")
        narration["headline"] = (
            f"different properties: correlation {r} across "
            f"{stats.get('n_samples', 0)} cells; the other statistics compare "
            "different quantities"
        )
    return out


#: Group-compare bounds. Results are capped at 6 (15 pairs) and the default
#: grid is small because every sample is a live Studio pixel-value call
#: (~0.5-1 min each): N results x grid^2 points is the cost driver.
_GROUP_MAX_RESULTS = 6
_GROUP_DEFAULT_GRID = 3
_GROUP_MAX_GRID = 8


async def _compare_group(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    """Handler for ``olmoearth_compare_group``."""
    raw_ids = args.get("result_ids") or []
    ids = [str(r) for r in raw_ids if str(r).strip()]
    # Order-preserving dedup: a repeated id would just re-sample the same raster.
    ids = list(dict.fromkeys(ids))
    if len(ids) < 2:
        return {
            "comparable": False,
            "reason": "need at least 2 distinct result_ids (2-6).",
        }
    if len(ids) > _GROUP_MAX_RESULTS:
        return {
            "comparable": False,
            "reason": f"too many results ({len(ids)}); cap is "
            f"{_GROUP_MAX_RESULTS} (each adds grid^2 slow pixel-value calls).",
        }
    prop = args.get("property_name")
    grid = max(2, min(_GROUP_MAX_GRID, int(args.get("grid", _GROUP_DEFAULT_GRID))))
    tol = float(args.get("tolerance", 0.1))

    records = await asyncio.gather(*[ctx.studio.get_prediction_result(r) for r in ids])
    bbox = intersect_bboxes([result_bbox(rec) for rec in records])
    if bbox is None:
        return {
            "comparable": False,
            "reason": "the results have no extent shared by all of them "
            "(or one is missing bounds)",
        }
    points = grid_points(bbox, grid)
    cache: dict[str, dict[str, Any] | None] = {}
    nodata = [await result_nodata_context(ctx, rec, cache=cache) for rec in records]

    sem = asyncio.Semaphore(_SAMPLE_CONCURRENCY)
    sampled = [await sample_records(ctx, rid, points, sem) for rid in ids]
    reads = [
        [read_sample(r, prop, nd) for r in recs] for recs, nd in zip(sampled, nodata)
    ]
    nodata_by_result = {
        rid: sum(1 for _v, _c, st in row if st == NODATA)
        for rid, row in zip(ids, reads)
    }
    first_ok = next((r for row in reads for r in row if r[2] == OK), None)
    first = next((r for recs in sampled for r in recs if r), None)
    categorical = bool(first_ok and first_ok[1])
    series = [[v for v, _c, _st in row] for row in reads]
    group = (
        compare_group_categorical(series)
        if categorical
        else compare_group_numeric(series, tolerance=tol)
    )
    value_type = "classification" if categorical else "regression"

    # Index -> id / coordinate mapping for readability (the analysis layer
    # works on indices; ids and lon/lat only exist here).
    for entry in group["pairwise"]:
        entry["result_id_a"] = ids[entry.pop("a_index")]
        entry["result_id_b"] = ids[entry.pop("b_index")]
    divergent = group.pop("most_divergent_pair", None)
    if divergent is not None:
        divergent = {
            "result_id_a": ids[divergent["a_index"]],
            "result_id_b": ids[divergent["b_index"]],
        }
    for spot in group["ensemble"].get("top_disagreement_points", []):
        lon, lat = points[spot.pop("point_index")]
        spot["lon"], spot["lat"] = lon, lat

    narration = compare_group_narration(
        group, n_results=len(ids), value_type=value_type
    )
    return {
        "comparable": True,
        "result_ids": ids,
        "property_name": prop
        or (first.get("bands", [{}])[0].get("property_name") if first else None),
        "value_type": value_type,
        "narration": narration,
        "grid": f"{grid}x{grid}",
        "samples_requested": len(points) * len(ids),
        "n_nodata_dropped": sum(nodata_by_result.values()),
        "nodata_by_result": nodata_by_result,
        "nodata_rule": NODATA_RULE,
        "shared_extent_bbox": [round(v, 5) for v in bbox],
        "pairwise": group["pairwise"],
        "ensemble": group["ensemble"],
        "most_divergent_pair": divergent,
        "method": "pointwise pixel-value sampled on a grid over the extent "
        "shared by all results (an estimate, not every pixel); no-data "
        "samples are dropped before any statistic; no ground truth, so this "
        "is " + narration["framing"] + ".",
    }


#: Cap on distinct models resolved per olmoearth_search_predictions call.
_MAX_MODELS_RESOLVED = 20

#: What a model listing does not contain, stated with it every time.
_MODELS_SCOPE_NOTE = (
    "These are the models in this Studio account only. Ai2 also publishes "
    "fine-tuned OlmoEarth models (the olmoearth_projects configurations and "
    "their task cards) that are not in this listing; say so instead of "
    "presenting this list as every model the user can run. prediction_type "
    "says what a model outputs: a *_regression output is a value per pixel, "
    "not a class probability and not a confidence."
)


async def _search_predictions(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    """Handler for ``olmoearth_search_predictions``: predictions plus their models."""
    project_id = args.get("project_id")
    env = await ctx.studio.search_predictions(
        project_id=project_id,
        limit=int(args.get("limit", 50)),
        offset=int(args.get("offset", 0)),
    )
    # With a project_id filter the match list is built client-side, so
    # env.total (the unfiltered server total) would mislead; report the
    # actual returned count instead.
    model_ids = list(
        dict.fromkeys(str(r["model_id"]) for r in env.records if r.get("model_id"))
    )[:_MAX_MODELS_RESOLVED]
    cache: dict[str, dict[str, Any] | None] = {}
    summaries = await asyncio.gather(*[model_summary(ctx, m, cache) for m in model_ids])
    models = {
        mid: {k: v for k, v in summary.items() if k != "model_id"}
        for mid, summary in zip(model_ids, summaries)
        if summary is not None
    }
    out: dict[str, Any] = {
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
    if model_ids:
        out["models"] = models
        out["models_note"] = _MODELS_SCOPE_NOTE
    return out


async def _submit_prediction(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    """Handler for ``olmoearth_submit_prediction``."""
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


def _declared_outputs(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Each output property with what the result declares about it.

    A regression output carries its declared ``[min_value, max_value]``; a
    classification output its class count and labels (capped). No geometry.
    """
    return [
        {"property_name": name, **info}
        for name, info in declared_fields(record).items()
    ]


def _summarize_result(record: dict[str, Any]) -> dict[str, Any]:
    """A prediction-result record as the agent sees it (no geometry)."""
    meta = record.get("result_metadata") or {}
    out: dict[str, Any] = {
        "result_id": record.get("id"),
        "prediction_id": record.get("prediction_id"),
        "tile_urls": record.get("tile_urls"),
        "property_names": record.get("property_names"),
        "file_format": record.get("file_format"),
    }
    outputs = _declared_outputs(record)
    if outputs:
        out["outputs"] = outputs
    if meta.get("start_datetime") or meta.get("end_datetime"):
        out["period"] = [meta.get("start_datetime"), meta.get("end_datetime")]
    return out


async def _fetch_results(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    """Handler for ``olmoearth_fetch_results``."""
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
    """Handler for ``olmoearth_get_prediction_result``.

    ``result_metadata`` is returned without its ``geometry`` (rule §3.1): the
    extent is raw coordinates, and the tools that need it read it themselves.
    """
    record = await ctx.studio.get_prediction_result(args["result_id"])
    summary = _summarize_result(record)
    meta = record.get("result_metadata")
    if isinstance(meta, dict) and "geometry" in meta:
        meta = {k: v for k, v in meta.items() if k != "geometry"}
        meta["geometry"] = "omitted (rule 3.1)"
    summary["result_metadata"] = meta
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
    rng = declared_range(band)
    nodata = value is not None and band_is_nodata(band)
    out: dict[str, Any] = {
        "result_id": result_id,
        "queried_point": point,
        "available": value is not None and not nodata,
        "property_name": band.get("property_name") or prop,
        "value_type": "classification" if categorical else "regression",
        "value": None if nodata else value,
        "bands": [
            {
                "property_name": b.get("property_name"),
                "value": band_value({"bands": [b]}, None),
            }
            for b in (record.get("bands") or [])
        ],
    }
    if rng is not None:
        out["declared_range"] = list(rng)
    if nodata:
        # Studio returns no-data as a value (e.g. -1.0 on a [0, 1] band);
        # reporting it as the model's output would be a fabricated reading.
        out["nodata"] = True
        out["raw_value"] = band.get("raw_value")
        out["reason"] = (
            "no-data at this point: the raw value lies outside the band's "
            "declared range or is not a number (Studio's no-data sentinel), "
            "so it is not a model output"
        )
    elif value is None:
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
                    "Search predictions, optionally scoped to a project: id, "
                    "name, status and model_id for each, plus a 'models' map "
                    "with each model's name, model_type and prediction_type "
                    "(e.g. per_pixel_regression: a value per pixel, not a class "
                    "probability or a confidence). Use it to find a reusable "
                    "model_id and to say what each model predicts. It lists "
                    "this account's models only; Ai2 also publishes fine-tuned "
                    "OlmoEarth models (task cards), so do not present the list "
                    "as complete. Read-only."
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
                    "(XYZ/MVT map layers), property names, file format, and "
                    "'outputs' (each property's declared regression range or "
                    "classes). Scans recent prediction-results and filters to "
                    "this prediction (no server-side filter); increase "
                    "scan_limit if results are older."
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
                    "URLs, property names, declared outputs (regression range "
                    "or classes per property), result metadata (without its "
                    "geometry), and file format."
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
                    "value at that point. If the point is off-raster or nodata "
                    "(including Studio's no-data sentinel, a value outside the "
                    "band's declared range such as -1 on a [0, 1] score), "
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
                    "shared area, with no ground truth: samples both rasters on "
                    "a grid (pixel-value) and returns mean difference, "
                    "mean-absolute difference, RMSE, correlation, and an "
                    "agreement fraction (regression) or class agreement "
                    "(classification). kind='cross_model' (default): two "
                    "models, same area (divergence, not accuracy); "
                    "kind='temporal': one model, earlier (A) vs later (B) date. "
                    "Points where either map is no-data (e.g. -1 on a [0, 1] "
                    "score) are dropped and counted in n_nodata_dropped; report "
                    "it. Results of DIFFERENT properties are refused with both "
                    "declared ranges unless allow_different_properties=true, "
                    "and then only the correlation is meaningful. For accuracy "
                    "against labels use olmoearth_classification_metrics."
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
                        "allow_different_properties": {
                            "type": "boolean",
                            "default": False,
                            "description": "Compare two results whose properties "
                            "differ anyway; only their correlation is then "
                            "meaningful, and the result carries a warning.",
                        },
                    },
                    "required": ["result_id_a", "result_id_b"],
                },
            ),
            handler=_compare_results,
        ),
        RegisteredTool(
            spec=ToolSpec(
                name="olmoearth_compare_group",
                description=(
                    "Quantitatively compare a GROUP of 2-6 prediction results "
                    "(different models) over the extent shared by all of them, "
                    "with no ground truth. Samples every raster on the same "
                    "grid (pointwise pixel-value) and returns (a) a PAIRWISE "
                    "matrix -- agreement / difference stats for every pair -- "
                    "and (b) an ENSEMBLE consensus: the fraction of cells where "
                    "ALL models agree (within tolerance, or same class), the "
                    "most divergent pair, and the most-contested grid points "
                    "(lon/lat). Use when the user compares THREE OR MORE "
                    "results, or asks where an ensemble of models converges / "
                    "diverges. For exactly TWO results prefer "
                    "olmoearth_compare_results (richer two-way stats + temporal "
                    "mode); for ONE model across MULTIPLE DATES use "
                    "olmoearth_trace_shifts (trajectory over time). No-data "
                    "samples are dropped before any statistic and counted in "
                    "n_nodata_dropped. Each result adds grid^2 slow Studio "
                    "pixel-value calls, so keep the grid small (default 3x3)."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "result_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 2,
                            "maxItems": _GROUP_MAX_RESULTS,
                            "description": "2-6 prediction-result ids "
                            "(different models over a shared area).",
                        },
                        "property_name": {
                            "type": "string",
                            "description": "Band/property to compare; defaults to "
                            "the first band.",
                        },
                        "grid": {
                            "type": "integer",
                            "default": _GROUP_DEFAULT_GRID,
                            "description": "Grid size N (N*N sample points per "
                            f"result; 2-{_GROUP_MAX_GRID}).",
                        },
                        "tolerance": {
                            "type": "number",
                            "default": 0.1,
                            "description": "Regression: a cell is consensus when "
                            "max-min across models <= tolerance.",
                        },
                    },
                    "required": ["result_ids"],
                },
            ),
            handler=_compare_group,
        ),
    ]
