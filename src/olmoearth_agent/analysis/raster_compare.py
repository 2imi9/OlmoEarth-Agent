# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Quantitative comparison of two prediction-result rasters, no ground truth.

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
