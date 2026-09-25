# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""The ``olmoearth_compare_results`` tool: how do these Studio results differ?

One tool for the one question that four tools used to answer
(``olmoearth_compare_results``, ``olmoearth_compare_group``,
``olmoearth_trace_shifts`` and ``olmoearth_ensemble_uncertainty``, merged
here). It takes 2-8 prediction-result ids, samples every result on one grid
of windows over their shared extent through the shared sampler
(:mod:`olmoearth_agent.tools.sampling`, so no-data never enters a statistic),
and reads the samples in one of four modes:

- ``pair``: two results; mean and mean-absolute difference, RMSE, correlation
  and an agreement fraction (regression) or class agreement. ``kind`` frames
  it: ``cross_model`` (two models) or ``temporal`` (one model, earlier vs
  later, ordered by the predictions' dates when they are known).
- ``group``: 2-6 results of different models; a pairwise matrix, the ensemble
  consensus and the most-contested windows.
- ``series``: 3-8 dated results of ONE model, ordered by date; per-step
  stats, per-window trajectories, and shift sizes calibrated to the field's
  value range.
- ``ensemble``: 2-8 results of one quantity as ensemble members; per-window
  dispersion folded into a confidence (self-consistency, not correctness).

``auto`` (the default) chooses from what the results' predictions say: two
results are a pair (``temporal`` when they are one model on two dates);
three or more are a series when they are one model on at least three
distinct dates, and a group otherwise.

Results of different properties are refused before any sampling (a pair or
a group may pass ``allow_different_properties``; then only correlations are
meaningful, and only they are returned: a difference, an RMSE, an agreement
fraction or an ensemble spread between two quantities is left out). Windows are addressed by grid ``(row, col)`` and row-major
``window_index``, row 0 the northernmost (rule §3.1): no coordinate and no
extent is returned. Every sample is a live pixel-value call, so the grid is
small by default and capped per mode.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from math import cos, radians
from typing import Any

from olmoearth_agent.analysis.change_detect import _try_parse
from olmoearth_agent.analysis.output_contract import add_forbidden, add_must_state
from olmoearth_agent.analysis.raster_compare import (
    COMPARISON_KINDS,
    compare_categorical,
    compare_group_categorical,
    compare_group_narration,
    compare_group_numeric,
    compare_narration,
    compare_numeric,
    declared_range,
    grid_windows,
    intersect_bboxes,
    result_bbox,
)
from olmoearth_agent.analysis.trace_shifts import (
    MIN_RESULTS,
    trace_categorical,
    trace_narration,
    trace_numeric,
)
from olmoearth_agent.analysis.uncertainty import prediction_confidence
from olmoearth_agent.llm.types import ToolSpec
from olmoearth_agent.tools import statistical_rules as rules
from olmoearth_agent.tools.registry import RegisteredTool, ToolContext
from olmoearth_agent.tools.sampling import (
    FAILED,
    NODATA,
    NODATA_RULE,
    OK,
    SAMPLE_CONCURRENCY,
    ResultNodata,
    default_property,
    model_summary,
    prediction_meta,
    read_sample,
    result_nodata_context,
    sample_records,
    select_band,
)

#: Most results one call compares (a series of 8 at 3x3 = 72 pixel-value
#: calls, as many as a pair at the default 6x6).
MAX_RESULTS = 8

#: The modes, ``auto`` first (the default).
MODES = ("auto", "pair", "group", "series", "ensemble")

#: Per mode: (default grid side, max grid side, min results, max results).
#: Every sample is a live pixel-value call (~0.5-1 min each through a proxy),
#: so results x grid^2 is the cost; a group caps at 6 results (15 pairs).
_LIMITS: dict[str, tuple[int, int, int, int]] = {
    "pair": (6, 12, 2, 2),
    "group": (3, 8, 2, 6),
    "series": (3, 6, MIN_RESULTS, MAX_RESULTS),
    "ensemble": (4, 10, 2, MAX_RESULTS),
}


@dataclass
class _Sampled:
    """One comparison's inputs and samples, in the order the mode reads them."""

    ids: list[str]
    records: list[dict[str, Any]]
    nodata: list[ResultNodata]
    sampled: list[list[dict[str, Any] | None]]
    reads: list[list[tuple[Any, bool, str]]]
    names: list[str | None]
    prop: str | None
    grid: int

    @property
    def n_points(self) -> int:
        """Grid windows sampled per result."""
        return self.grid * self.grid


def _refuse(reason: str, **extra: Any) -> dict[str, Any]:
    """A ``comparable: False`` answer with its reason."""
    out: dict[str, Any] = {"comparable": False, "reason": reason}
    out.update(extra)
    return out


def _cell(index: int, grid: int) -> dict[str, int]:
    """A window's address: row-major index and ``(row, col)``, row 0 north."""
    row, col = divmod(index, grid)
    return {"window_index": index, "row": row, "col": col}


def _address(entry: dict[str, Any], grid: int) -> None:
    """Replace an analysis-layer ``point_index`` by the window's address."""
    entry.update(_cell(int(entry.pop("point_index")), grid))


def _extent_km2(bbox: list[float]) -> float:
    """Approximate area of the shared extent in km² (no coordinates returned)."""
    minx, miny, maxx, maxy = bbox
    mid = radians((miny + maxy) / 2)
    return round((maxx - minx) * 111.32 * cos(mid) * (maxy - miny) * 110.57, 1)


def _sort_instant(parsed: datetime) -> datetime:
    """A tz-comparable instant: naive datetimes are read as UTC.

    Studio timestamps mix ``Z``-suffixed (tz-aware) and bare (naive) forms;
    Python refuses to order aware vs naive, which would crash the
    chronological sort instead of tripping the loud given-order fallback.
    """
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)


def _value_range(raw: Any) -> tuple[float, float] | None:
    """Validate an optional ``[min, max]`` field range (None if unusable)."""
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        return None
    try:
        lo, hi = float(raw[0]), float(raw[1])
    except (TypeError, ValueError):
        return None
    return (lo, hi) if hi > lo else None


def _is_number(value: Any) -> bool:
    """True for an int or float that is not a bool."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _property_summary(name: str | None, nodata: ResultNodata) -> dict[str, Any]:
    """A property's name and what its result declares about it (no geometry)."""
    out: dict[str, Any] = {"property_name": name}
    out.update(nodata.declared.get(str(name), {}))
    return out


def _sampled_property(
    reads: list[tuple[Any, bool, str]],
    recs: list[dict[str, Any] | None],
    prop: str | None,
) -> str | None:
    """The property name the sampled bands actually carried (first usable one)."""
    for rec, (_value, _cat, status) in zip(recs, reads):
        if rec is not None and status != FAILED:
            band = select_band(rec, prop)
            if band is not None and isinstance(band.get("property_name"), str):
                return str(band["property_name"])
    return None


def _count_refusal(mode: str, n: int) -> dict[str, Any] | None:
    """Refuse a result count the mode cannot read, before any Studio call."""
    if mode == "pair" and n != 2:
        return _refuse(
            f"mode='pair' compares exactly two results (got {n}); use "
            "mode='group' for several models, mode='series' for one model "
            "over dates.",
            mode=mode,
        )
    if mode == "series" and n < MIN_RESULTS:
        return _refuse(
            f"a series needs >= {MIN_RESULTS} distinct result ids (got {n}): a "
            "two-date diff hides gradual drift and cannot tell a steady trend "
            "from a reversal. For exactly two dated results use mode='pair' "
            "with kind='temporal'.",
            mode=mode,
        )
    cap = _LIMITS[mode][3]
    if n > cap:
        return _refuse(
            f"too many results for mode='{mode}' ({n} > {cap}): each result "
            "adds grid^2 slow pixel-value calls and a group adds every pair. "
            "Pass fewer ids, or mode='ensemble' for the spread across all "
            "of them.",
            mode=mode,
        )
    return None


def _mixed_types(reads: list[list[tuple[Any, bool, str]]]) -> bool:
    """True when the usable samples mix numbers with classes or labels."""
    kinds = {
        bool(categorical) or not _is_number(value)
        for row in reads
        for value, categorical, status in row
        if status == OK
    }
    return len(kinds) > 1


def _different_properties(
    a_id: str, b_id: str, prop_a: dict[str, Any], prop_b: dict[str, Any]
) -> dict[str, Any]:
    """The pair refusal for two results that measure different properties."""
    return {
        "comparable": False,
        "mode": "pair",
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


def _property_refusal(
    mode: str, ids: list[str], names: list[str | None], nodata: list[ResultNodata]
) -> dict[str, Any]:
    """The refusal for results of different properties, in the mode's shape.

    Both shapes carry the forbidden claims of a comparison across properties
    (a combined statistic, a winner), since a refusal is also read as one.
    """
    claims = rules.different_properties(names)
    if mode == "pair":
        return rules.add_contract(
            _different_properties(
                ids[0],
                ids[1],
                _property_summary(names[0], nodata[0]),
                _property_summary(names[1], nodata[1]),
            ),
            forbidden_claims=claims,
        )
    distinct = sorted({n for n in names if n})
    escape = (
        " or pass allow_different_properties=true to read only the correlations"
        if mode == "group"
        else ""
    )
    refusal = _refuse(
        f"the results measure different properties ({distinct}); a "
        f"{mode} of them would mix quantities, not measure how one quantity "
        f"differs. Pass results of one property (or pin property_name){escape}.",
        mode=mode,
        result_ids=ids,
        properties=[
            {"result_id": rid, **_property_summary(name, nd)}
            for rid, name, nd in zip(ids, names, nodata)
        ],
    )
    return rules.add_contract(refusal, forbidden_claims=claims)


#: What a comparison of two different properties keeps: each map's own mean
#: and whether the two rise and fall together. Every other statistic (a
#: difference, an RMSE, an agreement fraction) mixes two quantities, so it is
#: not returned (exp86 round 4: a warning not to read them was not enough; the
#: answers put them in a table).
_SINGLE_QUANTITY_STATS = ("n_samples", "mean_a", "mean_b", "correlation", "note")


def _without_mixed(stats: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """``stats`` less every statistic that mixes the two quantities, and the
    names of those left out."""
    kept = {k: v for k, v in stats.items() if k in _SINGLE_QUANTITY_STATS}
    return kept, sorted(k for k in stats if k not in _SINGLE_QUANTITY_STATS)


def _different_warning(names: list[str | None]) -> str:
    """The warning carried by an allowed comparison of different properties."""
    distinct = sorted({n for n in names if n})
    return (
        f"the results measure different properties ({distinct}), so their "
        "values are different quantities: only the correlation (whether they "
        "rise and fall together) is meaningful, and it is the only statistic "
        "returned between them. A mean difference, an RMSE, an agreement "
        "fraction or an ensemble spread would mix two quantities, so none is "
        "computed; do not work one out, and do not say one reads higher than "
        "the other."
    )


#: Stated with a comparison of two different properties.
DIFFERENT_PROPERTIES_MUST_STATE = (
    "The results measure different properties: only whether they rise and fall "
    "together (the correlation) is meaningful between them."
)

#: Forbidden with a comparison of two different properties.
COMBINED_STATISTIC_FORBIDDEN = {
    "id": "combined_statistic_across_properties",
    "why": "a difference, an RMSE, an agreement fraction, an ensemble spread, or "
    "'one reads higher than the other' between two properties mixes two "
    "quantities; none is computed, so none may be worked out or stated",
}


def _mark_different(out: dict[str, Any]) -> None:
    """The output contract of a comparison across properties: what to state, what
    not to claim; each once, however often a result is marked."""
    add_must_state(out, [DIFFERENT_PROPERTIES_MUST_STATE])
    add_forbidden(out, [COMBINED_STATISTIC_FORBIDDEN])


def _pair(
    s: _Sampled,
    *,
    kind: str,
    tolerance: float,
    different: bool,
) -> dict[str, Any]:
    """Pair mode: two-way divergence (or change) statistics."""
    pairs: list[tuple[Any, Any]] = []
    n_nodata = n_failed = 0
    for (va, _ca, sa), (vb, _cb, sb) in zip(s.reads[0], s.reads[1]):
        if sa == OK and sb == OK:
            pairs.append((va, vb))
        elif NODATA in (sa, sb):
            n_nodata += 1
        else:
            n_failed += 1
    first = next((r for r in s.reads[0] + s.reads[1] if r[2] == OK), None)
    categorical = bool(first and first[1])
    stats = (
        compare_categorical(pairs)
        if categorical
        else compare_numeric(pairs, tolerance=tolerance)
    )
    value_type = "classification" if categorical else "regression"
    narration = compare_narration(stats, kind=kind, value_type=value_type)
    out: dict[str, Any] = {
        "result_id_a": s.ids[0],
        "result_id_b": s.ids[1],
        "property_name": s.prop or s.names[0],
        "property_a": _property_summary(s.names[0], s.nodata[0]),
        "property_b": _property_summary(s.names[1], s.nodata[1]),
        "kind": kind,
        "value_type": value_type,
        "narration": narration,
        # Cells, not samples: a pair samples each cell twice (exp86 round 1
        # read the 72 samples of a 6x6 grid as 72 cells).
        "n_cells_compared": stats.get("n_samples", 0),
        "n_cells_dropped": s.n_points - stats.get("n_samples", 0),
        "n_nodata_dropped": n_nodata,
        "n_failed_dropped": n_failed,
        "stats": stats,
        "method": "pointwise pixel-value sampled on a grid over the shared "
        "extent (an estimate, not every pixel); windows where either map is "
        "no-data are dropped before any statistic; no ground truth, so this is "
        + narration["framing"]
        + ".",
    }
    if different:
        out["stats"], out["statistics_left_out"] = _without_mixed(stats)
        out["warning"] = _different_warning(s.names)
        _mark_different(out)
        narration["headline"] = (
            f"different properties: correlation {stats.get('correlation')} "
            f"across {stats.get('n_samples', 0)} cells; no other statistic "
            "between them is returned"
            if not categorical
            else f"different properties: two class sets across "
            f"{stats.get('n_samples', 0)} cells; no statistic between them is "
            "returned"
        )
    return out


def _group(s: _Sampled, *, tolerance: float, different: bool) -> dict[str, Any]:
    """Group mode: pairwise matrix, ensemble consensus, most-contested windows."""
    nodata_by_result = {
        rid: sum(1 for _v, _c, st in row if st == NODATA)
        for rid, row in zip(s.ids, s.reads)
    }
    first = next((r for row in s.reads for r in row if r[2] == OK), None)
    categorical = bool(first and first[1])
    series = [[v for v, _c, _st in row] for row in s.reads]
    group = (
        compare_group_categorical(series)
        if categorical
        else compare_group_numeric(series, tolerance=tolerance)
    )
    value_type = "classification" if categorical else "regression"
    # The analysis layer works on indices; ids and window addresses exist here.
    for entry in group["pairwise"]:
        entry["result_id_a"] = s.ids[entry.pop("a_index")]
        entry["result_id_b"] = s.ids[entry.pop("b_index")]
    divergent = group.pop("most_divergent_pair", None)
    if divergent is not None:
        divergent = {
            "result_id_a": s.ids[divergent["a_index"]],
            "result_id_b": s.ids[divergent["b_index"]],
        }
    for spot in group["ensemble"].get("top_disagreement_points", []):
        _address(spot, s.grid)
    narration = compare_group_narration(
        group, n_results=len(s.ids), value_type=value_type
    )
    out: dict[str, Any] = {
        "property_name": s.prop or next((n for n in s.names if n), None),
        "value_type": value_type,
        "narration": narration,
        "n_nodata_dropped": sum(nodata_by_result.values()),
        "nodata_by_result": nodata_by_result,
        "pairwise": group["pairwise"],
        "ensemble": group["ensemble"],
        "most_divergent_pair": divergent,
        "method": "pointwise pixel-value sampled on a grid over the extent "
        "shared by all results (an estimate, not every pixel); no-data "
        "samples are dropped before any statistic; no ground truth, so this "
        "is " + narration["framing"] + ".",
    }
    if different:
        # Pairs of one property keep their statistics; a pair of two
        # properties keeps each map's mean and the correlation. An ensemble
        # or a most-divergent pair across quantities means nothing.
        left_out: set[str] = set()
        name_of = dict(zip(s.ids, s.names))
        for entry in out["pairwise"]:
            if name_of[entry["result_id_a"]] != name_of[entry["result_id_b"]]:
                entry["stats"], dropped = _without_mixed(entry["stats"])
                left_out.update(dropped)
        out["ensemble"] = {
            "note": "not computed: the results measure different properties"
        }
        out["most_divergent_pair"] = None
        out["statistics_left_out"] = sorted(left_out | {"ensemble"})
        out["warning"] = _different_warning(s.names)
        _mark_different(out)
        narration["headline"] = (
            f"different properties across {len(s.ids)} results: pairwise "
            "correlations only between results of different properties"
        )
    return out


def _series(
    s: _Sampled,
    *,
    dates: list[str | None],
    ordering: str,
    model_check: str,
    tolerance: float,
    raw_range: Any,
) -> dict[str, Any]:
    """Series mode: how one model's estimates shifted across dated results."""
    n_nodata = sum(1 for row in s.reads for _v, _c, st in row if st == NODATA)
    first = next((r for row in s.reads for r in row if r[2] == OK), None)
    if first is None:
        return _refuse(
            "no grid window returned a usable sample from any result "
            f"({n_nodata} no-data samples dropped)",
            mode="series",
            result_ids=s.ids,
        )
    categorical = first[1]
    series: list[list[Any]] = [[v for v, _c, _st in row] for row in s.reads]
    value_range = _value_range(raw_range)
    if categorical:
        trace = trace_categorical(series)
        value_type = "classification"
    else:
        trace = trace_numeric(series, tolerance=tolerance, value_range=value_range)
        value_type = "regression"
        if raw_range is not None and value_range is None:
            trace["calibration"]["note"] = (
                "supplied value_range was invalid (need [min, max] with "
                "max > min) and was ignored; " + trace["calibration"]["note"]
            )
    narration = trace_narration(trace, value_type=value_type, ordering=ordering)
    for step in trace["steps"]:
        a, b = step.pop("a_index"), step.pop("b_index")
        step["from"] = {"result_id": s.ids[a], "date": dates[a]}
        step["to"] = {"result_id": s.ids[b], "date": dates[b]}
    for pp in trace["trajectory"].get("top_shift_points", []):
        _address(pp, s.grid)
        if "largest_step_between" in pp:
            k_from, k_to = pp.pop("largest_step_between")
            pp["largest_step_from_date"] = dates[k_from]
            pp["largest_step_to_date"] = dates[k_to]
    out: dict[str, Any] = {
        "dates": dates,
        "ordering": ordering,
        "model_check": model_check,
        "property_name": s.prop or next((n for n in s.names if n), None),
        "value_type": value_type,
        "narration": narration,
        "n_nodata_dropped": n_nodata,
        "steps": trace["steps"],
        "trajectory": trace["trajectory"],
        "method": "each dated result sampled pointwise (pixel-value) on the "
        "same grid over the shared extent (an estimate, not every pixel); "
        "no-data samples are dropped before any statistic; "
        + narration["framing"]
        + ".",
    }
    if value_type == "classification":
        out["transitions"] = trace["transitions"]
    else:
        out["calibration"] = trace["calibration"]
    return out


def _ensemble(s: _Sampled, *, value_type: str | None) -> dict[str, Any]:
    """Ensemble mode: per-window dispersion across results of one quantity."""
    samples: list[list[Any]] = []
    kept: list[int] = []
    detected: bool | None = None
    n_dropped = n_nodata = 0
    for pi in range(s.n_points):
        vals: list[Any] = []
        for row in s.reads:
            value, is_cat, status = row[pi]
            if status == NODATA:
                n_nodata += 1
            if status != OK:
                continue
            if detected is None:
                detected = is_cat
            vals.append(value)
        # An ensemble needs >= 2 members at a window to measure disagreement.
        if len(vals) >= 2:
            samples.append(vals)
            kept.append(pi)
        else:
            n_dropped += 1
    if not samples:
        return _refuse(
            "no grid window had >= 2 overlapping valid samples across the "
            f"ensemble ({n_nodata} no-data samples dropped)",
            mode="ensemble",
            result_ids=s.ids,
        )
    if value_type in ("classification", "categorical"):
        vt = "categorical"
    elif value_type == "regression":
        vt = "regression"
    else:
        vt = "categorical" if detected else "regression"
    try:
        out = prediction_confidence(samples, value_type=vt)
    except (ValueError, TypeError):
        # A regression value_type over labels (results disagree on band type).
        return _refuse(
            f"sampled values are not consistent with value_type "
            f"'{value_type or vt}' (the results may disagree on band type); "
            "pass value_type or property_name to disambiguate",
            mode="ensemble",
            result_ids=s.ids,
        )
    out["value_type"] = "classification" if vt == "categorical" else "regression"
    for entry, pi in zip(out["per_point"], kept):
        entry.update(_cell(pi, s.grid))
    summary = out.get("summary") or {}
    out.update(
        {
            "property_name": s.prop or next((n for n in s.names if n), None),
            "n_results": len(s.ids),
            "n_points_dropped": n_dropped,
            "n_nodata_dropped": n_nodata,
            "narration": {
                "headline": (
                    f"mean confidence {summary.get('mean_confidence')} across "
                    f"{out['n_points']} windows, {len(s.ids)} results as "
                    "ensemble members"
                ),
                "framing": "disagreement between distinct results (epistemic "
                "uncertainty, self-consistency), not accuracy",
            },
            "method": "each result sampled pointwise (pixel-value) on a grid "
            "over the shared extent; no-data draws (a missing value, the "
            "model's nodata_value, or a value outside the band's declared "
            "range) are dropped first; the >=2 values at a window are treated "
            "as ensemble members, so the spread is real disagreement between "
            "distinct results -- epistemic uncertainty, not accuracy.",
        }
    )
    return out


async def _compare_results(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    """Handler for ``olmoearth_compare_results`` (all four modes)."""
    raw_ids = args.get("result_ids") or []
    # Distinct results only: a repeated id would re-read ONE deterministic
    # result and fake zero disagreement.
    ids = list(dict.fromkeys(str(r).strip() for r in raw_ids if str(r).strip()))
    mode = str(args.get("mode") or "auto")
    mode = mode if mode in MODES else "auto"
    prop = args.get("property_name")
    if len(ids) < 2:
        return _refuse(
            f"need >= 2 DISTINCT result_ids (got {len(ids)}): a repeated id "
            "re-reads one deterministic result and fakes zero disagreement.",
            mode=mode,
        )
    if len(ids) > MAX_RESULTS:
        return _refuse(
            f"too many results ({len(ids)} > {MAX_RESULTS}): each result adds "
            f"grid^2 slow pixel-value calls. Compare the {MAX_RESULTS} most "
            "relevant.",
            mode=mode,
        )
    if mode != "auto" and (refusal := _count_refusal(mode, len(ids))):
        return refusal

    records = list(
        await asyncio.gather(*[ctx.studio.get_prediction_result(r) for r in ids])
    )
    bbox = intersect_bboxes([result_bbox(rec) for rec in records])
    if bbox is None:
        return _refuse(
            "the results share no common extent (their overlap is empty, or "
            "one is missing bounds); every result must cover the same area",
            mode=mode,
            result_ids=ids,
        )

    # Date and model of each result from its prediction, before any sampling:
    # they choose the mode, order a series, and give the model's nodata_value.
    metas = list(await asyncio.gather(*[prediction_meta(ctx, rec) for rec in records]))
    model_ids = [m for _d, m in metas]
    known_models = {m for m in model_ids if m}
    one_model = len(known_models) == 1 and all(model_ids)
    parsed = [_try_parse(d) if d else None for d, _m in metas]
    known_dates = [_sort_instant(p) for p in parsed if p is not None]
    instants = known_dates if len(known_dates) == len(parsed) else None
    n_dates = len(set(instants)) if instants is not None else 0

    if mode == "auto":
        if len(ids) == 2:
            mode = "pair"
        elif one_model and n_dates >= MIN_RESULTS:
            mode = "series"
        else:
            mode = "group"
        if refusal := _count_refusal(mode, len(ids)):
            return refusal

    order = list(range(len(ids)))
    ordering = "given-order"
    kind = "cross_model"
    model_check = "unverified"
    if mode == "series":
        if len(known_models) > 1:
            return _refuse(
                f"the results span multiple models ({sorted(known_models)}); a "
                "series traces ONE model's estimates over time -- a mixed series "
                "would report model disagreement as temporal drift. For "
                "model-vs-model comparison use mode='group'.",
                mode=mode,
                result_ids=ids,
            )
        model_check = "single-model" if one_model else "unverified"
        if instants is not None:
            if n_dates < MIN_RESULTS:
                return _refuse(
                    f"the results share dates ({n_dates} distinct of {len(ids)}): "
                    f"a shift trace needs >= {MIN_RESULTS} distinct dates to "
                    "separate movement over time from re-estimation noise. For "
                    "same-date reruns use mode='ensemble'.",
                    mode=mode,
                    result_ids=ids,
                )
            order = sorted(order, key=known_dates.__getitem__)
            ordering = "chronological"
    elif mode == "pair":
        explicit = args.get("kind")
        if explicit in COMPARISON_KINDS:
            kind = str(explicit)
        elif one_model and n_dates == 2:
            kind = "temporal"
        if kind == "temporal":
            if len(known_models) > 1:
                return _refuse(
                    f"the two results come from different models "
                    f"({sorted(known_models)}): a temporal comparison would "
                    "report model disagreement as change over time. Use "
                    "kind='cross_model'.",
                    mode=mode,
                    result_ids=ids,
                )
            if instants is not None and n_dates == 2:
                order = sorted(order, key=known_dates.__getitem__)
                ordering = "chronological"

    # Each model is looked up once, and only now that no refusal above applies.
    cache: dict[str, dict[str, Any] | None] = {}
    for model_id in dict.fromkeys(m for m in model_ids if m):
        await model_summary(ctx, model_id, cache)
    models = [cache.get(m) if m else None for m in model_ids]
    ids = [ids[i] for i in order]
    records = [records[i] for i in order]
    dates = [metas[i][0] for i in order]
    nodata = [
        await result_nodata_context(
            ctx, records[k], model=models[i], lookup_model=False
        )
        for k, i in enumerate(order)
    ]

    # Different properties are refused BEFORE any slow sampling, from what the
    # records declare; the sampled bands re-check it below.
    allow_different = mode in ("pair", "group") and bool(
        args.get("allow_different_properties", False)
    )
    names = [default_property(rec, prop) for rec in records]
    if len({n for n in names if n}) > 1 and not allow_different:
        return _property_refusal(mode, ids, names, nodata)

    grid_default, grid_max, _lo, _hi = _LIMITS[mode]
    grid = max(2, min(grid_max, int(args.get("grid", grid_default))))
    cells = grid_windows(bbox, grid)
    points = [(lon, lat) for _r, _c, lon, lat in cells]
    sem = asyncio.Semaphore(SAMPLE_CONCURRENCY)
    sampled = list(
        await asyncio.gather(*[sample_records(ctx, rid, points, sem) for rid in ids])
    )
    reads = [
        [read_sample(r, prop, nd) for r in row] for row, nd in zip(sampled, nodata)
    ]
    names = [
        _sampled_property(row_reads, row, prop) or name
        for row_reads, row, name in zip(reads, sampled, names)
    ]
    different = len({n for n in names if n}) > 1
    if different and not allow_different:
        return _property_refusal(mode, ids, names, nodata)
    value_type = args.get("value_type") if mode == "ensemble" else None
    if not value_type and _mixed_types(reads):
        # Results disagree on band type: a clean refusal, not a float()
        # traceback after the sampling budget is spent.
        return _refuse(
            "sampled values mix numeric and categorical bands across the "
            "results (they disagree on value_type); pass property_name to pin "
            "one field",
            mode=mode,
            result_ids=ids,
        )

    s = _Sampled(ids, records, nodata, sampled, reads, names, prop, grid)
    # Negative tolerance would make "agree" empty and, in a series, push
    # shifted_fraction past 1.0: clamp to the floor.
    tolerance = max(0.0, float(args.get("tolerance", 0.1)))
    if mode == "pair":
        out = _pair(s, kind=kind, tolerance=tolerance, different=different)
        if kind == "temporal":
            out["dates"] = dates
            out["ordering"] = ordering
    elif mode == "group":
        out = _group(s, tolerance=tolerance, different=different)
    elif mode == "series":
        out = _series(
            s,
            dates=dates,
            ordering=ordering,
            model_check=model_check,
            tolerance=tolerance,
            raw_range=args.get("value_range"),
        )
    else:
        out = _ensemble(s, value_type=value_type)
    if out.get("comparable") is False:
        return out
    claims: list[dict[str, str]] = []
    if unthresholded := _unthresholded_regression(s, out):
        claims.append(rules.unthresholded_regression(unthresholded))
    if different:
        claims += rules.different_properties(names)
    return rules.add_contract(
        {
            "comparable": True,
            "mode": mode,
            "result_ids": ids,
            **out,
            "grid": f"{grid}x{grid}",
            "n_cells": s.n_points,
            "samples_requested": s.n_points * len(ids),
            "sampling_note": _sampling_note(s, out, mode),
            "shared_extent_km2": _extent_km2(bbox),
            "nodata_rule": NODATA_RULE,
        },
        forbidden_claims=claims,
    )


def _unthresholded_regression(
    s: _Sampled, out: dict[str, Any]
) -> list[tuple[str | None, tuple[float, float] | None]]:
    """The compared regression bands that have no decision threshold.

    A band declared (or sampled) with the range ``[0, 1]`` is the score the
    estimation tools decide at 0.5, as they state; any other regression band
    has no threshold here, so no error rate or classification metric (exp86
    round 7, brief 3 on Studio: "estimate each map's error rate" for a band
    declared 0.2 to 1.2). Returns ``(property_name, declared range)`` per band.
    """
    if out.get("value_type") != "regression":
        return []
    found: dict[str | None, tuple[float, float] | None] = {}
    for name, nd, row in zip(s.names, s.nodata, s.sampled):
        band = next(
            (b for rec in row if rec and (b := select_band(rec, name)) is not None),
            None,
        )
        rng = declared_range(band, nd.declared.get(str(name)))
        if rng != (0.0, 1.0) and name not in found:
            found[name] = rng
    return list(found.items())


def _sampling_note(s: _Sampled, out: dict[str, Any], mode: str) -> str:
    """Cells against samples, stated, so neither count is read as the other."""
    n_results = len(s.ids)
    note = (
        f"the {s.grid}x{s.grid} grid has {s.n_points} cells; each of the "
        f"{n_results} results was sampled once per cell, so "
        f"{s.n_points * n_results} samples were requested"
    )
    if mode == "pair":
        return note + (
            f"; {out['n_cells_compared']} compared, {out['n_cells_dropped']} "
            "dropped (no-data or failed in either map; n_nodata_dropped and "
            "n_failed_dropped count those cells)"
        )
    note += "; n_nodata_dropped counts samples, not cells"
    if mode == "ensemble":
        note += (
            f"; {out.get('n_points_dropped', 0)} cells were dropped for holding "
            "fewer than two valid values"
        )
    return note


def build_compare_tools() -> list[RegisteredTool]:
    """Return the comparison bundle: ``olmoearth_compare_results``."""
    return [
        RegisteredTool(
            spec=ToolSpec(
                name="olmoearth_compare_results",
                description=(
                    "How do these Studio prediction results differ? One tool "
                    "for 2-8 results over their shared area, with no ground "
                    "truth: agreement or estimate movement, never accuracy "
                    "(with labels: olmoearth_classification_metrics). Samples "
                    "each result on one grid of windows; no-data (e.g. -1 on a "
                    "[0, 1] score) is dropped and counted; results of different "
                    "properties are refused. mode: pair (2: difference, RMSE, "
                    "correlation, agreement; kind cross_model or temporal), "
                    "group (2-6 models: pairwise, consensus, most-contested "
                    "windows), series (3-8 dated results of ONE model, "
                    "date-ordered: steps, trajectories; estimate movement, not "
                    "ground change), ensemble (2-8 results of one quantity: "
                    "per-window spread as a confidence, not correctness). auto "
                    "(default): 2 -> pair (temporal if one model on two dates); "
                    "3+ -> series if one model on 3+ dates, else group. Windows "
                    "are (row, col), row 0 north. Not for ranking windows to "
                    "review (olmoearth_review_set_from_result). Slow: results x "
                    "grid^2 live calls."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "result_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 2,
                            "maxItems": MAX_RESULTS,
                            "description": "Result ids over a shared area "
                            "(a pair: A then B).",
                        },
                        "mode": {
                            "type": "string",
                            "enum": list(MODES),
                            "default": "auto",
                        },
                        "kind": {
                            "type": "string",
                            "enum": list(COMPARISON_KINDS),
                            "description": "Pair only; inferred if omitted.",
                        },
                        "property_name": {
                            "type": "string",
                            "description": "Band to compare (default: first).",
                        },
                        "grid": {
                            "type": "integer",
                            "description": "N (N*N windows per result); "
                            "default/cap pair 6/12, group 3/8, series 3/6, "
                            "ensemble 4/10.",
                        },
                        "tolerance": {
                            "type": "number",
                            "default": 0.1,
                            "description": "|difference| <= tolerance "
                            "agrees (regression).",
                        },
                        "value_range": {
                            "type": "array",
                            "items": {"type": "number"},
                            "minItems": 2,
                            "maxItems": 2,
                            "description": "Series: the field's known [min, max].",
                        },
                        "value_type": {
                            "type": "string",
                            "enum": ["regression", "classification"],
                            "description": "Ensemble: override detection.",
                        },
                        "allow_different_properties": {
                            "type": "boolean",
                            "default": False,
                            "description": "Pair or group: compare different "
                            "properties anyway (correlations only; warned).",
                        },
                    },
                    "required": ["result_ids"],
                },
            ),
            handler=_compare_results,
        )
    ]
