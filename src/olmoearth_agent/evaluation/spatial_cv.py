# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Spatial cross-validation and the random-vs-spatial inflation diagnostic.

Pure Python. Coordinates are ``(lon, lat)`` in degrees; distances are
great-circle kilometres (haversine). Brute-force nearest-neighbour is
O(n^2): fine for the hundreds-to-thousands of label points typical of
an OlmoEarth case study.

References:
- Roberts et al. 2017 (Ecography): spatial block CV.
- Ploton et al. 2020 (Nat. Commun. 11:4540): random CV inflates accuracy
  on spatially autocorrelated data.
- Meyer & Pebesma 2021 (MEE 12:1620): formalization + area of applicability.
"""

from __future__ import annotations

import math
import random as _random

Point = tuple[float, float]  # (lon, lat) in degrees
_EARTH_RADIUS_KM = 6371.0088


def haversine_km(a: Point, b: Point) -> float:
    """Great-circle distance between two ``(lon, lat)`` points, in km."""
    lon1, lat1 = a
    lon2, lat2 = b
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    h = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    )
    return 2 * _EARTH_RADIUS_KM * math.asin(min(1.0, math.sqrt(h)))


def spatial_block_folds(
    points: list[Point], n_folds: int = 5, block_deg: float = 0.5
) -> list[int]:
    """Assign each point to a spatial-block CV fold.

    Points are bucketed into a ``block_deg`` × ``block_deg`` grid; each
    whole block is assigned to one fold (round-robin over blocks sorted
    by grid key). Spatially-close points share a block and therefore a
    fold, so train and test are spatially separated.

    Raises
    ------
    ValueError
        If ``n_folds < 2`` or ``block_deg <= 0``.
    """
    if n_folds < 2:
        raise ValueError("n_folds must be >= 2")
    if block_deg <= 0:
        raise ValueError("block_deg must be > 0")
    blocks: dict[tuple[int, int], list[int]] = {}
    for i, (lon, lat) in enumerate(points):
        key = (math.floor(lon / block_deg), math.floor(lat / block_deg))
        blocks.setdefault(key, []).append(i)
    fold_of = [0] * len(points)
    for fold_rank, key in enumerate(sorted(blocks)):
        fold = fold_rank % n_folds
        for i in blocks[key]:
            fold_of[i] = fold
    return fold_of


def random_folds(points: list[Point], n_folds: int = 5, seed: int = 3407) -> list[int]:
    """Assign each point to a random (non-spatial) CV fold (seeded)."""
    if n_folds < 2:
        raise ValueError("n_folds must be >= 2")
    order = list(range(len(points)))
    _random.Random(seed).shuffle(order)  # noqa: S311 - deterministic split
    fold_of = [0] * len(points)
    for rank, i in enumerate(order):
        fold_of[i] = rank % n_folds
    return fold_of


def _mean_test_to_train_nn(points: list[Point], fold_of: list[int]) -> float:
    """Mean distance from each point to its nearest point in another fold.

    In k-fold CV a point's "train set" is every point outside its own
    fold. This is the nearest-neighbour distance that drives the Ploton
    inflation effect: small under random folds (autocorrelated neighbours
    leak across the split), large under spatial folds.
    """
    distances: list[float] = []
    n = len(points)
    for i in range(n):
        fold_i = fold_of[i]
        best = math.inf
        for j in range(n):
            if j == i or fold_of[j] == fold_i:
                continue
            d = haversine_km(points[i], points[j])
            if d < best:
                best = d
        if best != math.inf:
            distances.append(best)
    return sum(distances) / len(distances) if distances else 0.0


def cv_inflation_diagnostic(
    points: list[Point],
    n_folds: int = 5,
    block_deg: float = 0.5,
    seed: int = 3407,
) -> dict[str, object]:
    """Quantify how optimistic random CV is vs spatial-block CV.

    Compares the mean test-point-to-nearest-train-point distance under
    random folds vs spatial-block folds. A large ratio means random CV
    is testing on points that sit right next to training points, its
    accuracy will be inflated (Ploton 2020). Report spatial-block CV
    metrics instead.

    Returns a JSON-able summary with both mean distances, their ratio,
    a risk band, and a recommendation.
    """
    if len(points) < n_folds:
        raise ValueError("need at least n_folds points")
    random_assign = random_folds(points, n_folds, seed)
    spatial_assign = spatial_block_folds(points, n_folds, block_deg)
    nn_random = _mean_test_to_train_nn(points, random_assign)
    nn_spatial = _mean_test_to_train_nn(points, spatial_assign)
    ratio = nn_spatial / nn_random if nn_random > 0 else None

    if ratio is None:
        risk = "unknown"
    elif ratio >= 3.0:
        risk = "high"
    elif ratio >= 1.5:
        risk = "moderate"
    else:
        risk = "low"

    recommendation = (
        "Report spatial-block CV metrics; random CV will overstate " "accuracy here."
        if risk in ("high", "moderate")
        else "Random and spatial CV are comparable; either is defensible."
    )
    return {
        "n_points": len(points),
        "n_folds": n_folds,
        "block_deg": block_deg,
        "mean_test_to_train_km_random": round(nn_random, 3),
        "mean_test_to_train_km_spatial": round(nn_spatial, 3),
        "spatial_to_random_ratio": round(ratio, 2) if ratio is not None else None,
        "inflation_risk": risk,
        "recommendation": recommendation,
    }
