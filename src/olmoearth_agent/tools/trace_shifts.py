# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""The ``olmoearth_trace_shifts`` tool: timeseries shifts of one model's estimates.

Fills the deliberately unclaimed seam between the compare tools and change
detection: ``olmoearth_compare_results`` handles two results (its
``kind="temporal"`` trusts the CALLER to order them), ``olmoearth_compare_group``
is N different *models* over one area, and ``olmoearth_change_detect`` needs a
ready-made ``(date, value)`` series. This tool takes N >= 3 result ids of ONE
model over the same area, discovers each result's date itself (via its
prediction's ``start_time``), orders the series chronologically, samples every
result on one shared grid, and traces how the estimate shifted: per-step pixel
correlation + per-point trajectories + a legend-calibrated shift size
(``value_range`` from the project's annotation form when known, observed range
otherwise).

Cost model matches the other sampling tools: every sample is a live Studio
pixel-value call (~0.5-1 min each), so results are capped and the default grid
is small — N results x grid^2 is the budget driver.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

from olmoearth_agent.analysis.change_detect import _try_parse
from olmoearth_agent.analysis.raster_compare import (
    grid_points,
    intersect_bboxes,
    result_bbox,
)
from olmoearth_agent.analysis.trace_shifts import (
    MIN_RESULTS,
    trace_categorical,
    trace_narration,
    trace_numeric,
)
from olmoearth_agent.llm.types import ToolSpec
from olmoearth_agent.tools.predict import _band_value, _is_categorical, _select_band
from olmoearth_agent.tools.registry import RegisteredTool, ToolContext

#: Concurrency for grid pixel-value sampling (bounds load on Studio + proxy).
_SAMPLE_CONCURRENCY = 8

#: Trace bounds. 8 results x 3x3 grid = 72 pixel-value calls, on par with one
#: default two-result compare (grid 6 -> 72); the grid caps lower than the
#: pair tool because N multiplies every extra point.
_TRACE_MAX_RESULTS = 8
_TRACE_DEFAULT_GRID = 3
_TRACE_MAX_GRID = 6


async def _result_meta(
    ctx: ToolContext, record: dict[str, Any]
) -> tuple[str | None, str | None]:
    """A result's ``(date, model_id)`` from its prediction record.

    Result records carry no date of their own; the prediction they came from
    does (``start_time``, fallback ``end_time``), and it also names the model
    — used to refuse a mixed-model series. Best-effort: any lookup failure
    yields ``(None, None)`` and the trace falls back to given-order with an
    explicit flag, rather than failing the run.
    """
    prediction_id = record.get("prediction_id")
    if not prediction_id:
        return None, None
    try:
        prediction = await ctx.studio.get_prediction(str(prediction_id))
    except Exception:  # date discovery is best-effort, never fatal
        return None, None
    date = prediction.get("start_time") or prediction.get("end_time")
    model_id = prediction.get("model_id")
    return (str(date) if date else None), (str(model_id) if model_id else None)


def _sort_instant(parsed: datetime) -> datetime:
    """A tz-comparable instant: naive datetimes are read as UTC.

    Studio timestamps mix ``Z``-suffixed (tz-aware) and bare (naive) forms;
    Python refuses to order aware vs naive, which would crash the
    chronological sort instead of tripping the loud given-order fallback.
    """
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)


async def _trace_shifts(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    raw_ids = args.get("result_ids") or []
    ids = list(dict.fromkeys(str(r).strip() for r in raw_ids if str(r).strip()))
    if len(ids) < MIN_RESULTS:
        return {
            "comparable": False,
            "reason": f"shift tracing needs >= {MIN_RESULTS} distinct result "
            f"ids (got {len(ids)}): a two-date diff hides gradual drift and "
            "cannot tell a steady trend from a reversal. For exactly two "
            "results use olmoearth_compare_results with kind='temporal'.",
        }
    if len(ids) > _TRACE_MAX_RESULTS:
        return {
            "comparable": False,
            "reason": f"too many results ({len(ids)} > {_TRACE_MAX_RESULTS}): "
            "each result adds grid^2 slow pixel-value calls. Trace the "
            f"{_TRACE_MAX_RESULTS} most relevant dates.",
        }
    prop = args.get("property_name")
    grid = max(2, min(_TRACE_MAX_GRID, int(args.get("grid", _TRACE_DEFAULT_GRID))))
    # Negative tolerance would make "increased" and "decreased" overlap and
    # push shifted_fraction past 1.0 — clamp to the sane floor.
    tol = max(0.0, float(args.get("tolerance", 0.1)))
    raw_range = args.get("value_range")
    value_range = _value_range(raw_range)
    range_invalid = raw_range is not None and value_range is None

    records = [await ctx.studio.get_prediction_result(rid) for rid in ids]
    bbox = intersect_bboxes([result_bbox(rec) for rec in records])
    if bbox is None:
        return {
            "comparable": False,
            "reason": "the results have no common overlapping extent (or one "
            "is missing bounds); trace needs every result over the same area",
        }

    # Date + model discovery from each result's prediction — before any slow
    # sampling, so a refusal here costs nothing.
    metas = [await _result_meta(ctx, rec) for rec in records]
    dates = [m[0] for m in metas]
    known_models = {m[1] for m in metas if m[1]}
    if len(known_models) > 1:
        return {
            "comparable": False,
            "reason": "the results span multiple models "
            f"({sorted(known_models)}); this tool traces ONE model's "
            "estimates over time — a mixed series would report model "
            "disagreement as temporal drift. For model-vs-model comparison "
            "use olmoearth_compare_group.",
        }
    model_check = (
        "single-model" if known_models and all(m[1] for m in metas) else "unverified"
    )

    # Chronological ordering from the prediction dates. If any date is
    # missing/unparseable, keep the given order and say so — a wrong order
    # silently flips every "shift" sign, so the fallback must be loud.
    parsed = [_try_parse(d) if d else None for d in dates]
    known = [p for p in parsed if p is not None]
    if len(known) == len(parsed):
        instants = [_sort_instant(p) for p in known]
        if len(set(instants)) < MIN_RESULTS:
            return {
                "comparable": False,
                "reason": f"the results share dates ({len(set(instants))} "
                f"distinct of {len(ids)}): a shift trace needs >= "
                f"{MIN_RESULTS} distinct dates to separate movement over "
                "time from re-estimation noise. For same-date reruns use "
                "olmoearth_ensemble_uncertainty.",
            }
        order = sorted(range(len(ids)), key=lambda i: instants[i])
        ordering = "chronological"
    else:
        order = list(range(len(ids)))
        ordering = "given-order"
    ids = [ids[i] for i in order]
    dates = [dates[i] for i in order]

    points = grid_points(bbox, grid)
    sem = asyncio.Semaphore(_SAMPLE_CONCURRENCY)

    async def sample(result_id: str, lon: float, lat: float) -> Any:
        async with sem:
            try:
                return await ctx.studio.pixel_value(result_id, lon, lat)
            except Exception:  # off-raster / nodata / transient -> drop the point
                return None

    sampled: list[list[Any]] = [  # sequential across results, concurrent within one
        await asyncio.gather(*[sample(rid, lo, la) for lo, la in points])
        for rid in ids
    ]

    # First USABLE sample (a record can be truthy yet carry empty/None bands,
    # e.g. a nodata pixel) — it decides categorical-vs-regression and names
    # the traced band, so it must actually have a value.
    first = next(
        (
            r
            for row in sampled
            for r in row
            if r and _band_value(r, prop) is not None
        ),
        None,
    )
    if first is None:
        return {
            "comparable": False,
            "reason": "no grid point returned a usable sample from any result",
        }
    categorical = _is_categorical(first, prop)
    series: list[list[Any]] = [
        [(_band_value(r, prop) if r else None) for r in row] for row in sampled
    ]
    if not categorical and any(
        v is not None and not isinstance(v, (int, float))
        for row in series
        for v in row
    ):
        # Results disagree on band type (some numeric raw_value, some class
        # labels): a clean refusal, not a float() traceback after the whole
        # sampling budget was already spent computing half a trace.
        return {
            "comparable": False,
            "reason": "sampled values mix numeric and categorical bands "
            "across the results; pass property_name to pin one field",
        }

    if categorical:
        trace = trace_categorical(series)
        value_type = "classification"
    else:
        trace = trace_numeric(series, tolerance=tol, value_range=value_range)
        value_type = "regression"
        if range_invalid:
            trace["calibration"]["note"] = (
                "supplied value_range was invalid (need [min, max] with "
                "max > min) and was ignored; " + trace["calibration"]["note"]
            )
    narration = trace_narration(trace, value_type=value_type, ordering=ordering)

    # Remap analysis indices to real ids/dates/coords (tool-layer convention).
    for step in trace["steps"]:
        a, b = step.pop("a_index"), step.pop("b_index")
        step["from"] = {"result_id": ids[a], "date": dates[a]}
        step["to"] = {"result_id": ids[b], "date": dates[b]}
    for pp in trace["trajectory"].get("top_shift_points", []):
        lon, lat = points[pp.pop("point_index")]
        pp["lon"], pp["lat"] = lon, lat
        if "largest_step_between" in pp:
            k_from, k_to = pp.pop("largest_step_between")
            pp["largest_step_from_date"] = dates[k_from]
            pp["largest_step_to_date"] = dates[k_to]

    sampled_names: set[str] = {
        name
        for row in sampled
        for r in row
        if r
        for name in [(_select_band(r, prop) or {}).get("property_name")]
        if isinstance(name, str)
    }
    envelope: dict[str, Any] = {
        "comparable": True,
        "result_ids": ids,
        "dates": dates,
        "ordering": ordering,
        "model_check": model_check,
        "property_name": prop
        or (_select_band(first, prop) or {}).get("property_name"),
        "value_type": value_type,
        "narration": narration,
        "grid": f"{grid}x{grid}",
        "samples_requested": len(points) * len(ids),
        "shared_extent_bbox": [round(v, 5) for v in bbox],
        "steps": trace["steps"],
        "trajectory": trace["trajectory"],
        "method": "each dated result sampled pointwise (pixel-value) on the "
        "same grid over the shared extent (an estimate, not every pixel); "
        + narration["framing"]
        + ".",
    }
    if not prop and len(sampled_names) > 1:
        # With no property_name, every result reads its own first band —
        # if the results disagree on what that band is, the traced field is
        # ambiguous and the caller should pin it.
        envelope["property_name_note"] = (
            f"results disagree on the default band ({sorted(sampled_names)}); "
            "pass property_name to pin the traced field"
        )
    if value_type == "classification":
        envelope["transitions"] = trace["transitions"]
    else:
        envelope["calibration"] = trace["calibration"]
    return envelope


def _value_range(raw: Any) -> tuple[float, float] | None:
    """Validate an optional ``[min, max]`` field range (None if unusable)."""
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        return None
    try:
        lo, hi = float(raw[0]), float(raw[1])
    except (TypeError, ValueError):
        return None
    return (lo, hi) if hi > lo else None


def build_trace_shift_tools() -> list[RegisteredTool]:
    """Return the shift-tracing bundle (skill #5, change-detection family)."""
    return [
        RegisteredTool(
            spec=ToolSpec(
                name="olmoearth_trace_shifts",
                description=(
                    "Trace how ONE model's estimates SHIFTED across 3+ dated "
                    "prediction results over the same area: orders the results "
                    "chronologically (by each prediction's start_time), samples "
                    "them all on one shared grid, and returns per-step stats "
                    "(Pearson correlation, mean delta, agreement), per-point "
                    "trajectories (net change, slope, largest step, top shifted "
                    "points), and shift sizes calibrated to the field's value "
                    "range. Use when the user asks how an estimate evolved / "
                    "drifted / shifted over time across multiple runs. Routing: "
                    "exactly TWO results -> olmoearth_compare_results "
                    "(kind='temporal'); N different MODELS -> "
                    "olmoearth_compare_group; an already-extracted (date,value) "
                    "series -> olmoearth_change_detect. Slow: every sample is a "
                    "live pixel-value call (~0.5-1 min); results x grid^2 is "
                    "the cost. Measures estimate movement, NOT verified ground "
                    "change and not accuracy."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "result_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 3,
                            "maxItems": _TRACE_MAX_RESULTS,
                            "description": "3-8 completed prediction-result ids "
                            "of ONE model over the same area (any order; the "
                            "tool sorts chronologically).",
                        },
                        "property_name": {
                            "type": "string",
                            "description": "Band/field to trace (default: the "
                            "result's first band).",
                        },
                        "grid": {
                            "type": "integer",
                            "default": _TRACE_DEFAULT_GRID,
                            "maximum": _TRACE_MAX_GRID,
                            "description": "Sample grid side (NxN per result); "
                            "keep small - every point is a slow live call.",
                        },
                        "tolerance": {
                            "type": "number",
                            "default": 0.1,
                            "description": "|delta| <= tolerance counts as no "
                            "shift (regression only).",
                        },
                        "value_range": {
                            "type": "array",
                            "items": {"type": "number"},
                            "minItems": 2,
                            "maxItems": 2,
                            "description": "Known [min, max] of this field from "
                            "the project's annotation form (Project Settings), "
                            "so shifts read as a fraction of the real range; "
                            "omit to fall back to the observed range.",
                        },
                    },
                    "required": ["result_ids"],
                },
            ),
            handler=_trace_shifts,
        )
    ]
