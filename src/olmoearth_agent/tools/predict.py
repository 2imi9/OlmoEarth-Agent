# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""The ``olmoearth-predict`` tool bundle (skill #4): the core run loop.

Search predictions (to discover reusable ``model_id``s), submit a new
prediction, and poll it. Polling reuses the foundational
``olmoearth_get_prediction`` tool. ``olmoearth_pixel_value`` reads the
model's output at a single point; comparing results over an area is
``olmoearth_compare_results`` (:mod:`olmoearth_agent.tools.compare`);
feature-search remains a follow-up within this skill.
"""

from __future__ import annotations

import asyncio
from typing import Any

from olmoearth_agent.analysis.raster_compare import (
    band_is_nodata,
    declared_fields,
    declared_range,
)
from olmoearth_agent.llm.types import ToolSpec
from olmoearth_agent.tools.registry import Capability, RegisteredTool, ToolContext
from olmoearth_agent.tools.sampling import (
    band_value,
    is_categorical,
    model_summary,
    select_band,
)

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
    band = select_band(record, prop)
    if band is None:
        return {
            "result_id": result_id,
            "queried_point": point,
            "available": False,
            "reason": "no value at this point (off-raster or nodata)",
        }
    value = band_value(record, prop)
    categorical = is_categorical(record, prop)
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
            capability=Capability(
                does="this account's predictions, and each model's name, type "
                "and prediction_type",
                cannot=(
                    "say what a model was trained on (label fields, training "
                    "data, metrics)",
                ),
            ),
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
            capability=Capability(
                does="run a Studio model over a Studio area and period; returns a "
                "prediction id to poll"
            ),
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
            capability=Capability(
                does="a prediction's results (tile URLs, property names, "
                "declared outputs)"
            ),
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
            capability=Capability(
                does="one result's tile URLs, properties and declared outputs, "
                "without its geometry"
            ),
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
                    "compare results over an area use olmoearth_compare_results."
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
            capability=Capability(does="one result's value at one lon/lat point"),
        ),
    ]
