# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Quantitative comparison of prediction-result rasters, no ground truth.

The agent can show two result rasters side by side, but "which differs and by
how much" needs numbers. With no ground-truth labels, an accuracy metric is
undefined; what *is* well-defined is the **agreement / divergence** between the
two model outputs. This module computes that from values sampled on a grid over
the shared extent (pointwise via Studio ``pixel-value``): mean difference,
mean-absolute difference, RMSE-between-models, Pearson correlation, and a
threshold agreement fraction (regression) or class-agreement (categorical).

Pure Python: no GDAL/numpy (the agent stays torch-free; full-raster pixel math
remains out-of-process). Sampling is a grid, not every pixel -- an estimate,
labeled as such.
"""

from __future__ import annotations

from math import sqrt
from typing import Any


def intersect_bbox(
    a: list[float] | None, b: list[float] | None
) -> list[float] | None:
    """Intersection of two ``[min_lon, min_lat, max_lon, max_lat]`` boxes.

    Returns ``None`` if either is missing or they do not overlap (so the
    caller can report "the two results do not overlap" instead of sampling an
    empty region).
    """
    if not a or not b or len(a) != 4 or len(b) != 4:
        return None
    minx, miny = max(a[0], b[0]), max(a[1], b[1])
    maxx, maxy = min(a[2], b[2]), min(a[3], b[3])
    if minx >= maxx or miny >= maxy:
        return None
    return [minx, miny, maxx, maxy]


def intersect_bboxes(boxes: list[list[float] | None]) -> list[float] | None:
    """Intersection of N bounding boxes (fold of :func:`intersect_bbox`).

    Returns ``None`` if the list is empty, any box is missing, or the common
    overlap is empty — the group comparison needs an extent shared by *every*
    result.
    """
    if not boxes:
        return None
    acc = boxes[0]
    for box in boxes[1:]:
        acc = intersect_bbox(acc, box)
        if acc is None:
            return None
    return acc


def grid_points(bbox: list[float], n: int) -> list[tuple[float, float]]:
    """An ``n`` x ``n`` grid of interior ``(lon, lat)`` points across ``bbox``.

    Cell centers (the ``(i + 0.5) / n`` fractions) keep samples off the edges,
    where a result raster often has nodata.
    """
    minx, miny, maxx, maxy = bbox
    pts: list[tuple[float, float]] = []
    for i in range(n):
        lon = minx + (maxx - minx) * (i + 0.5) / n
        for j in range(n):
            lat = miny + (maxy - miny) * (j + 0.5) / n
            pts.append((round(lon, 6), round(lat, 6)))
    return pts


def pearson(xs: list[float], ys: list[float]) -> float | None:
    """Pearson correlation of paired samples, or ``None`` if undefined."""
    n = len(xs)
    if n < 2:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 0 or syy <= 0:  # a constant series has no correlation
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return sxy / sqrt(sxx * syy)


def compare_numeric(
    pairs: list[tuple[float | None, float | None]], *, tolerance: float
) -> dict[str, Any]:
    """Divergence stats for paired regression values (``a`` = first result).

    Pairs where either side is ``None`` (nodata / off-raster / a failed
    sample) are dropped. ``tolerance`` defines "agree": ``|a - b| <=
    tolerance``. All numbers rounded for clean JSON.
    """
    paired = [(a, b) for a, b in pairs if a is not None and b is not None]
    n = len(paired)
    if n == 0:
        return {"n_samples": 0, "note": "no overlapping valid samples"}
    xs = [a for a, _ in paired]
    ys = [b for _, b in paired]
    diffs = [b - a for a, b in paired]
    abs_diffs = [abs(d) for d in diffs]
    agree = sum(1 for d in abs_diffs if d <= tolerance)
    r = pearson(xs, ys)
    return {
        "n_samples": n,
        "mean_a": round(sum(xs) / n, 6),
        "mean_b": round(sum(ys) / n, 6),
        "mean_diff_b_minus_a": round(sum(diffs) / n, 6),
        "mean_abs_diff": round(sum(abs_diffs) / n, 6),
        "max_abs_diff": round(max(abs_diffs), 6),
        "rmse_between_models": round(sqrt(sum(d * d for d in diffs) / n), 6),
        "correlation": None if r is None else round(r, 4),
        "agreement_fraction": round(agree / n, 4),
        "tolerance": tolerance,
    }


def compare_categorical(
    pairs: list[tuple[Any, Any]],
) -> dict[str, Any]:
    """Agreement stats for paired categorical (classification) values."""
    paired = [(a, b) for a, b in pairs if a is not None and b is not None]
    n = len(paired)
    if n == 0:
        return {"n_samples": 0, "note": "no overlapping valid samples"}
    agree = sum(1 for a, b in paired if a == b)
    return {
        "n_samples": n,
        "agreement_fraction": round(agree / n, 4),
        "n_disagree": n - agree,
    }


#: Top-K most-divergent grid points surfaced from a group comparison. A cap,
#: not a page size: the full per-point breakdown would bloat the tool result
#: (it accumulates in the model's context) without adding decision value.
_GROUP_HOTSPOT_CAP = 5


def _pair_entries(
    series: list[list[Any]],
    stat_fn: Any,
) -> list[dict[str, Any]]:
    """Per-pair stats for every (i, j) combination of N aligned series."""
    return [
        {
            "a_index": i,
            "b_index": j,
            "stats": stat_fn(list(zip(series[i], series[j]))),
        }
        for i in range(len(series))
        for j in range(i + 1, len(series))
    ]


def _most_divergent_pair(pairwise: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The pair with the lowest agreement fraction (ties: fewer samples last)."""
    scored = [
        p
        for p in pairwise
        if p["stats"].get("agreement_fraction") is not None
    ]
    if not scored:
        return None
    worst = min(
        scored,
        key=lambda p: (p["stats"]["agreement_fraction"], -p["stats"]["n_samples"]),
    )
    return {"a_index": worst["a_index"], "b_index": worst["b_index"]}


def compare_group_numeric(
    series: list[list[float | None]], *, tolerance: float
) -> dict[str, Any]:
    """Pairwise + ensemble divergence stats for N aligned regression series.

    ``series`` holds one value list per result, aligned on the same grid
    points (``None`` = nodata / failed sample). Pairwise reuses
    :func:`compare_numeric`. The ensemble half treats the >=2 valid values at
    each point as ensemble members: a point is *consensus* when its spread
    (max - min) is within ``tolerance``, and the most-divergent points are
    surfaced (by index; the caller maps indices to coordinates).
    """
    pairwise = _pair_entries(
        series, lambda pairs: compare_numeric(pairs, tolerance=tolerance)
    )
    n_points = len(series[0]) if series else 0
    used = 0
    consensus = 0
    spreads: list[float] = []
    stds: list[float] = []
    per_point: list[tuple[float, int, int]] = []  # (spread, point_index, n_valid)
    for p in range(n_points):
        vals: list[float] = [v for s in series if (v := s[p]) is not None]
        if len(vals) < 2:
            continue
        used += 1
        spread = max(vals) - min(vals)
        mean = sum(vals) / len(vals)
        stds.append(sqrt(sum((v - mean) ** 2 for v in vals) / len(vals)))
        spreads.append(spread)
        if spread <= tolerance:
            consensus += 1
        per_point.append((spread, p, len(vals)))
    per_point.sort(key=lambda t: -t[0])
    ensemble: dict[str, Any] = {
        "n_points_used": used,
        "tolerance": tolerance,
    }
    if used:
        ensemble.update(
            consensus_fraction=round(consensus / used, 4),
            mean_spread=round(sum(spreads) / used, 6),
            max_spread=round(max(spreads), 6),
            mean_ensemble_std=round(sum(stds) / used, 6),
            top_disagreement_points=[
                {"point_index": idx, "spread": round(spread, 6), "n_models": k}
                for spread, idx, k in per_point[:_GROUP_HOTSPOT_CAP]
                if spread > tolerance
            ],
        )
    else:
        ensemble["note"] = "no grid point has valid values from 2+ results"
    return {
        "pairwise": pairwise,
        "ensemble": ensemble,
        "most_divergent_pair": _most_divergent_pair(pairwise),
    }


def compare_group_categorical(series: list[list[Any]]) -> dict[str, Any]:
    """Pairwise + ensemble agreement stats for N aligned categorical series.

    A point is *unanimous* when every valid value agrees; otherwise its
    majority share (modal-class fraction) measures how split the models are.
    """
    pairwise = _pair_entries(series, compare_categorical)
    n_points = len(series[0]) if series else 0
    used = 0
    unanimous = 0
    majority_shares: list[float] = []
    per_point: list[tuple[float, int, int]] = []  # (majority_share, index, n_valid)
    for p in range(n_points):
        vals = [s[p] for s in series if s[p] is not None]
        if len(vals) < 2:
            continue
        used += 1
        counts: dict[Any, int] = {}
        for v in vals:
            counts[v] = counts.get(v, 0) + 1
        share = max(counts.values()) / len(vals)
        majority_shares.append(share)
        if len(counts) == 1:
            unanimous += 1
        per_point.append((share, p, len(vals)))
    per_point.sort(key=lambda t: t[0])  # most split first
    ensemble: dict[str, Any] = {"n_points_used": used}
    if used:
        ensemble.update(
            unanimous_fraction=round(unanimous / used, 4),
            mean_majority_share=round(sum(majority_shares) / used, 4),
            top_disagreement_points=[
                {
                    "point_index": idx,
                    "majority_share": round(share, 4),
                    "n_models": k,
                }
                for share, idx, k in per_point[:_GROUP_HOTSPOT_CAP]
                if share < 1.0
            ],
        )
    else:
        ensemble["note"] = "no grid point has valid values from 2+ results"
    return {
        "pairwise": pairwise,
        "ensemble": ensemble,
        "most_divergent_pair": _most_divergent_pair(pairwise),
    }


def compare_group_narration(
    group: dict[str, Any], *, n_results: int, value_type: str
) -> dict[str, Any]:
    """Human labels for a group (N-result, cross-model) comparison.

    Group compare has one meaning — N different models over the same area —
    so unlike :func:`compare_narration` there is no kind branch (a multi-DATE
    series belongs to the change-detection trajectory, not here).
    """
    ensemble = group.get("ensemble") or {}
    used = ensemble.get("n_points_used", 0)
    if value_type == "classification":
        share = ensemble.get("unanimous_fraction")
        headline = (
            f"all {n_results} models pick the same class at "
            f"{round(share * 100)}% of {used} cells"
            if share is not None
            else "no overlapping valid cells with 2+ model values"
        )
    else:
        share = ensemble.get("consensus_fraction")
        headline = (
            f"all {n_results} models agree within tolerance at "
            f"{round(share * 100)}% of {used} cells"
            if share is not None
            else "no overlapping valid cells with 2+ model values"
        )
    return {
        "headline": headline,
        "framing": "model-vs-model consensus across the group "
        "(no ground truth), not accuracy",
    }


#: The two comparison modes. ``cross_model`` = two different models over the
#: same area (how much do they agree?); ``temporal`` = one model's output at an
#: earlier (A) vs a later (B) date (how much did it change over time?).
COMPARISON_KINDS = ("cross_model", "temporal")


def normalize_kind(kind: str | None) -> str:
    """Coerce a kind to a known value, defaulting to ``cross_model``."""
    return kind if kind in COMPARISON_KINDS else "cross_model"


def compare_narration(
    stats: dict[str, Any], *, kind: str, value_type: str
) -> dict[str, Any]:
    """Kind-aware human labels for a comparison result.

    The numbers in ``stats`` are identical regardless of kind -- ``B - A`` is
    ``B - A`` either way. What differs is the *meaning*: ``cross_model`` asks
    "how much do two model outputs agree?", while ``temporal`` asks "how much
    did one model's output change between an earlier (A) and a later (B)
    date?". Returns the A / B / difference labels plus a one-line ``headline``
    and a ``framing`` caveat so the agent and the web UI tell the right story
    instead of always saying "model agreement".
    """
    kind = normalize_kind(kind)
    n = stats.get("n_samples", 0)
    agree = stats.get("agreement_fraction")
    agree_pct = None if agree is None else round(agree * 100)

    if kind == "temporal":
        labels = {"a": "earlier", "b": "later", "diff": "change (later - earlier)"}
        if value_type == "classification":
            headline = (
                f"class unchanged in {agree_pct}% of {n} cells"
                if agree_pct is not None
                else "no overlapping valid cells to compare over time"
            )
        else:
            net = stats.get("mean_diff_b_minus_a")
            if net is None:
                headline = "no overlapping valid cells to compare over time"
            elif net > 0:
                headline = f"net increase of {net} (later - earlier) across {n} cells"
            elif net < 0:
                headline = (
                    f"net decrease of {abs(net)} (later - earlier) across {n} cells"
                )
            else:
                headline = f"no net change (later - earlier) across {n} cells"
        framing = (
            "change over time for one model (later minus earlier), "
            "not model-vs-model agreement"
        )
    else:  # cross_model
        labels = {"a": "model A", "b": "model B", "diff": "difference (B - A)"}
        if value_type == "classification":
            headline = (
                f"the two models agree on {agree_pct}% of {n} cells"
                if agree_pct is not None
                else "no overlapping valid cells to compare"
            )
        else:
            headline = (
                f"the two models agree within tolerance on {agree_pct}% of {n} cells"
                if agree_pct is not None
                else "no overlapping valid cells to compare"
            )
        framing = (
            "model-vs-model agreement (no ground truth), not accuracy"
        )
    return {"kind": kind, "labels": labels, "headline": headline, "framing": framing}
