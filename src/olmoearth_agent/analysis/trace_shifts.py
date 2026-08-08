# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Timeseries shift tracing over one model's dated prediction results.

The question this answers: *how did the model's ESTIMATE move across N dated
results over the same area?* — traced exactly the way the compare tools work,
by pixel correlation on a shared grid, then read against the field's value
range ("legend") so a shift has real units.

Index-based like the rest of this layer (``raster_compare``): the caller
supplies ``series[k][p]`` = chronological result ``k`` sampled at grid point
``p``; result ids, dates, and lon/lat exist only in the tool layer, which
remaps ``a_index``/``b_index``/``point_index`` on the way out.

Three views of the same series:

* **steps** — every consecutive pair ``(k, k+1)`` scored with the shared
  :func:`~olmoearth_agent.analysis.raster_compare.compare_numeric` /
  ``compare_categorical`` stats (Pearson correlation, mean delta, agreement),
  so each step of the timeline reads like a familiar two-result compare.
* **trajectory** — per-point paths folded into aggregates: net change
  (last valid minus first valid), ordinary-least-squares slope per step,
  the largest single step, and the fraction of points that shifted beyond
  the tolerance, plus the top shifted points for hotspot follow-up.
* **calibration** (numeric only) — shifts expressed as a fraction of the
  field's value range: the caller-supplied range when the project's
  annotation form is known, else the observed range of the sampled values
  (clearly labeled, since an observed range understates the true one).

Honest by construction: this measures how the *estimate* moved between
result dates — it is not verified ground change, and with no ground truth it
says nothing about accuracy. Pure Python (no numpy/GDAL), like the rest of
``analysis/``.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from olmoearth_agent.analysis.raster_compare import (
    compare_categorical,
    compare_numeric,
)

#: A trace needs at least three results: a two-date diff is just
#: ``olmoearth_compare_results`` kind="temporal" and cannot tell a steady
#: trend from a reversal (same rationale as change detection's MIN_DATES).
MIN_RESULTS = 3

#: Cap on the most-shifted points listed for follow-up.
_HOTSPOT_CAP = 5

_PLACES = 6


def ols_slope(points: list[tuple[float, float]]) -> float | None:
    """Ordinary-least-squares slope of ``y`` over ``x`` (None if degenerate)."""
    n = len(points)
    if n < 2:
        return None
    mean_x = sum(x for x, _ in points) / n
    mean_y = sum(y for _, y in points) / n
    denom = sum((x - mean_x) ** 2 for x, _ in points)
    if denom == 0:
        return None
    num = sum((x - mean_x) * (y - mean_y) for x, y in points)
    return num / denom


def _step_entries(
    series: list[list[Any]], *, tolerance: float | None
) -> list[dict[str, Any]]:
    """Consecutive-pair stats: one entry per timeline step ``k -> k+1``."""
    steps: list[dict[str, Any]] = []
    for k in range(len(series) - 1):
        pairs = list(zip(series[k], series[k + 1]))
        stats = (
            compare_numeric(pairs, tolerance=tolerance)
            if tolerance is not None
            else compare_categorical(pairs)
        )
        steps.append({"a_index": k, "b_index": k + 1, "stats": stats})
    return steps


def _calibration(
    values: list[float], value_range: tuple[float, float] | None
) -> dict[str, Any]:
    """The value range shifts are read against, with its provenance."""
    if value_range is not None and value_range[1] > value_range[0]:
        vmin, vmax = float(value_range[0]), float(value_range[1])
        return {
            "source": "supplied",
            "vmin": vmin,
            "vmax": vmax,
            "note": "shift fractions use the supplied field range "
            "(e.g. from the project's annotation form)",
        }
    vmin = min(values) if values else 0.0
    vmax = max(values) if values else 1.0
    return {
        "source": "observed",
        "vmin": round(vmin, _PLACES),
        "vmax": round(vmax, _PLACES),
        "note": "no field range supplied; fractions use the observed range "
        "of the sampled values, which can understate the true range",
    }


def trace_numeric(
    series: list[list[float | None]],
    *,
    tolerance: float,
    value_range: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """Trace a numeric (regression) estimate series across dated results.

    ``series[k][p]`` is chronological result ``k`` at grid point ``p``
    (``None`` = failed/off-raster sample). Returns ``steps`` (consecutive
    pairwise stats), ``trajectory`` (per-point paths folded into
    aggregates + hotspots), and ``calibration``.
    """
    n_results = len(series)
    n_points = len(series[0]) if series else 0
    steps = _step_entries(series, tolerance=tolerance)

    all_values = [v for row in series for v in row if v is not None]
    calibration = _calibration(all_values, value_range)
    span = calibration["vmax"] - calibration["vmin"]

    per_point: list[dict[str, Any]] = []
    for p in range(n_points):
        valid = [
            (float(k), float(v))
            for k, v in ((k, series[k][p]) for k in range(n_results))
            if v is not None
        ]
        if len(valid) < 2:
            continue
        total = valid[-1][1] - valid[0][1]
        largest = 0.0
        largest_between = (int(valid[0][0]), int(valid[0][0]))
        for (xa, va), (xb, vb) in zip(valid, valid[1:]):
            # Round before comparing so equal-size steps are true ties (the
            # later step then wins) instead of float noise picking a winner.
            step = round(vb - va, _PLACES)
            if abs(step) >= abs(largest):
                largest = step
                largest_between = (int(xa), int(xb))
        per_point.append(
            {
                "point_index": p,
                "n_observations": len(valid),
                "total_change": round(total, _PLACES),
                "slope_per_step": round(ols_slope(valid) or 0.0, _PLACES),
                "largest_step": round(largest, _PLACES),
                "largest_step_between": list(largest_between),
            }
        )

    used = len(per_point)
    if used == 0:
        trajectory: dict[str, Any] = {
            "n_points_used": 0,
            "note": "no grid point had two or more valid samples",
        }
    else:
        totals = [pp["total_change"] for pp in per_point]
        increased = sum(1 for t in totals if t > tolerance)
        decreased = sum(1 for t in totals if t < -tolerance)
        mean_total = sum(totals) / used
        trajectory = {
            "n_points_used": used,
            "tolerance": tolerance,
            "mean_total_change": round(mean_total, _PLACES),
            "mean_slope_per_step": round(
                sum(pp["slope_per_step"] for pp in per_point) / used, _PLACES
            ),
            "increased_fraction": round(increased / used, 4),
            "decreased_fraction": round(decreased / used, 4),
            "shifted_fraction": round((increased + decreased) / used, 4),
            "mean_total_change_fraction_of_range": (
                round(mean_total / span, 4) if span > 0 else None
            ),
            "top_shift_points": sorted(
                (pp for pp in per_point if abs(pp["total_change"]) > tolerance),
                key=lambda pp: -abs(pp["total_change"]),
            )[:_HOTSPOT_CAP],
        }
    return {
        "n_results": n_results,
        "n_points": n_points,
        "steps": steps,
        "trajectory": trajectory,
        "calibration": calibration,
    }


def trace_categorical(series: list[list[Any]]) -> dict[str, Any]:
    """Trace a categorical (classification) estimate series across results.

    Same shape as :func:`trace_numeric` minus calibration, plus
    ``transitions``: the class-to-class changes counted over every
    consecutive step (the categorical "legend" reading of a shift).
    """
    n_results = len(series)
    n_points = len(series[0]) if series else 0
    steps = _step_entries(series, tolerance=None)

    transitions: Counter[tuple[str, str]] = Counter()
    per_point: list[dict[str, Any]] = []
    for p in range(n_points):
        valid = [
            (k, series[k][p]) for k in range(n_results) if series[k][p] is not None
        ]
        if len(valid) < 2:
            continue
        changes = 0
        for (_, va), (_, vb) in zip(valid, valid[1:]):
            if va != vb:
                changes += 1
                transitions[(str(va), str(vb))] += 1
        per_point.append(
            {
                "point_index": p,
                "n_observations": len(valid),
                "n_changes": changes,
                "first_class": valid[0][1],
                "last_class": valid[-1][1],
                "net_changed": valid[0][1] != valid[-1][1],
            }
        )

    used = len(per_point)
    if used == 0:
        trajectory: dict[str, Any] = {
            "n_points_used": 0,
            "note": "no grid point had two or more valid samples",
        }
    else:
        net_changed = sum(1 for pp in per_point if pp["net_changed"])
        trajectory = {
            "n_points_used": used,
            "net_changed_fraction": round(net_changed / used, 4),
            "stable_fraction": round(
                sum(1 for pp in per_point if pp["n_changes"] == 0) / used, 4
            ),
            "mean_changes_per_point": round(
                sum(pp["n_changes"] for pp in per_point) / used, 4
            ),
            "top_shift_points": sorted(
                (pp for pp in per_point if pp["n_changes"] > 0),
                key=lambda pp: -pp["n_changes"],
            )[:_HOTSPOT_CAP],
        }
    return {
        "n_results": n_results,
        "n_points": n_points,
        "steps": steps,
        "transitions": [
            {"from": a, "to": b, "count": c}
            for (a, b), c in transitions.most_common(8)
        ],
        "trajectory": trajectory,
    }


def trace_narration(
    result: dict[str, Any], *, value_type: str, ordering: str
) -> dict[str, Any]:
    """Plain-language reading of a trace, honest about what it measures.

    Same contract family as ``compare_narration``: ``labels`` + ``headline``
    + ``framing`` (the webui and the tool's ``method`` string consume
    ``framing``).
    """
    n = result.get("n_results", 0)
    trajectory = result.get("trajectory") or {}
    labels = {
        "first": "earliest result",
        "last": "latest result",
        "series": "chronological estimate series",
    }
    framing = (
        "shifts of one model's ESTIMATES across dated results (pixel "
        "sampling on a shared grid): how the estimate moved between run "
        "dates, not verified ground change, and not accuracy"
    )
    if ordering == "given-order":
        framing += "; result dates were not all available, so the series is "
        framing += "in the order given, not verified chronological"

    if trajectory.get("n_points_used", 0) == 0:
        return {
            "labels": labels,
            "headline": "no grid point had enough valid samples to trace",
            "framing": framing,
        }

    if value_type == "classification":
        pct = round(100 * trajectory.get("net_changed_fraction", 0.0), 1)
        headline = (
            f"class changed at {pct}% of sampled points between the earliest "
            f"and latest of {n} dated results"
        )
        transitions = result.get("transitions") or []
        if transitions:
            top = transitions[0]
            headline += (
                f"; most common transition {top['from']} -> {top['to']} "
                f"({top['count']}x)"
            )
        return {"labels": labels, "headline": headline, "framing": framing}

    mean_total = trajectory.get("mean_total_change", 0.0)
    tol = trajectory.get("tolerance", 0.0)
    shifted_pct = round(100 * trajectory.get("shifted_fraction", 0.0), 1)
    if mean_total > tol:
        word = "net increase"
    elif mean_total < -tol:
        word = "net decrease"
    else:
        word = "no net change"
    headline = (
        f"{word} of {round(abs(mean_total), _PLACES)} in the mean estimate "
        f"across {n} dated results; {shifted_pct}% of sampled points shifted "
        "beyond tolerance"
    )
    fraction = trajectory.get("mean_total_change_fraction_of_range")
    if fraction is not None:
        headline += (
            f" (mean net shift = {round(100 * abs(fraction), 1)}% of the "
            f"{result.get('calibration', {}).get('source', 'value')} range)"
        )
    return {"labels": labels, "headline": headline, "framing": framing}
