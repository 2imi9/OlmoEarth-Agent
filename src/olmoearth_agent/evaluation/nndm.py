# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Nearest Neighbour Distance Matching (NNDM) Leave-One-Out cross-validation.

A faithful pure-Python port of the geographical-space ``nndm`` algorithm from
the R package **CAST** (`HannaMeyer/CAST`, `R/nndm.R`), introduced by

    Mila, C., Mateu, J., Pebesma, E., Meyer, H. (2022). "Nearest Neighbour
    Distance Matching Leave-One-Out Cross-Validation for map validation."
    Methods in Ecology and Evolution 13, 1304-1316.

Why it matters: plain random / LOO cross-validation reports an *optimistic*
map-accuracy number when training points are spatially clustered, because each
left-out point usually has a near training neighbour (Ploton 2020). NNDM fixes
the LOO folds so the distribution of *test-to-training* nearest-neighbour
distances during CV matches the distribution of *prediction-to-training*
nearest-neighbour distances in the real map -- by excluding, for each left-out
point, the training neighbours that are "too close". The result is an
honest accuracy estimate for the area you actually predict over.

This module reuses :func:`~olmoearth_agent.evaluation.spatial_cv.haversine_km`
(great-circle km on ``(lon, lat)`` degrees) so distances and ``phi`` are in km,
consistent with the rest of the evaluation stack. It is O(n^2) in the number of
training points -- fine for the hundreds-to-thousands of labels typical of an
OlmoEarth case study; for very large sets prefer CAST's k-fold ``knndm``.

The core (:func:`_nndm_core`) is distance-agnostic (operates on a precomputed
training distance matrix) so it can be unit-tested against a hand-traced
reference execution of the CAST algorithm, independent of any RNG.
"""

from __future__ import annotations

import math
from typing import Any

from olmoearth_agent.evaluation.spatial_cv import Point, haversine_km

# A distance matrix cell is a float, or None for "not available" (the diagonal,
# and edges the algorithm has excluded). Mirrors NA in the R source.
_Row = list[float | None]


def _row_min(row: _Row) -> float:
    """Smallest finite entry of a matrix row; ``inf`` if the row is empty."""
    vals = [x for x in row if x is not None]
    return min(vals) if vals else math.inf


def _ecdf_at(sample: list[float], r: float) -> float:
    """Empirical CDF of ``sample`` evaluated at ``r`` (fraction <= r)."""
    if not sample:
        return 0.0
    return sum(1 for v in sample if v <= r) / len(sample)


def cv_optimism_km(test_nn: list[float], pred_nn: list[float]) -> float:
    """One-sided ECDF area that quantifies a CV's optimistic bias (km).

    The integral over distance ``r`` of ``max(0, F_test(r) - F_pred(r))``, where
    ``F_test`` is the ECDF of the *test-to-training* nearest-neighbour distances
    during CV and ``F_pred`` is the ECDF of the *prediction-to-training* NN
    distances over the map. It is positive exactly where the CV tests on points
    that sit *closer* to training data than the real prediction locations do --
    the source of the over-optimistic accuracy estimate (Ploton 2020).

    This is the one-sided quantity NNDM minimises (cf. the matching condition in
    CAST ``R/nndm.R``); lower is better, and NNDM never increases it. A
    symmetric L1 distance would be the wrong metric -- pushing test distances up
    to remove optimism can overshoot in the other direction without adding bias.
    """
    xs = sorted(set(test_nn) | set(pred_nn))
    if len(xs) < 2:
        return 0.0
    area = 0.0
    for i in range(len(xs) - 1):
        width = xs[i + 1] - xs[i]
        diff = _ecdf_at(test_nn, xs[i]) - _ecdf_at(pred_nn, xs[i])
        area += max(0.0, diff) * width
    return area


def _nndm_core(
    tdist: list[_Row], gij: list[float], phi: float, min_train: float
) -> tuple[list[_Row], list[float]]:
    """The NNDM matching loop on a precomputed training distance matrix.

    Faithful port of the ``while(rmin <= phi)`` loop in CAST ``R/nndm.R``.

    Parameters
    ----------
    tdist
        ``n x n`` training distance matrix with ``None`` on the diagonal
        (and symmetric off-diagonal). Mutated on a copy, not in place.
    gij
        Prediction-to-training nearest-neighbour distances (the *target*
        distribution); length = number of prediction points.
    phi
        Match distances only up to ``phi``; beyond it all points train freely.
    min_train
        Each left-out point keeps strictly more than this fraction of the
        training set (guards against emptying a fold).

    Returns
    -------
    (td, gjstar)
        ``td`` is the exclusion matrix (``None`` where row i must NOT train on
        column k); ``gjstar`` is the final matched test-to-train NN distances.

    Notes
    -----
    Exclusions are **directional / row-wise** (``tdist[jmin, kmin] <- NA`` in
    R touches one row only): point i excluding a near neighbour does not stop
    that neighbour from training on i. Float comparisons are exact because
    ``rmin`` is always an actual stored matrix entry (never recomputed), just
    as in the reference.
    """
    n = len(tdist)
    m = len(gij)
    td: list[_Row] = [row[:] for row in tdist]
    gjstar = [_row_min(td[j]) for j in range(n)]

    rmin = min(gjstar)
    jmin = gjstar.index(rmin)
    kmin = [k for k in range(n) if td[jmin][k] == rmin]

    while rmin <= phi:
        gjstar_le = sum(1 for v in gjstar if v <= rmin)
        gij_le = sum(1 for v in gij if v <= rmin)
        finite_jmin = sum(1 for x in td[jmin] if x is not None)
        # Remove this nearest edge iff the LOO ECDF stays >= the target ECDF
        # after the removal, and jmin keeps > min_train of the training set.
        if (gjstar_le - 1) / n >= gij_le / m and finite_jmin / n > min_train:
            for k in kmin:
                td[jmin][k] = None
            gjstar = [_row_min(td[j]) for j in range(n)]
            ge = [v for v in gjstar if v >= rmin]
            if not ge:
                break
            rmin = min(ge)
            jmin = next(j for j in range(n) if gjstar[j] == rmin)
            kmin = [k for k in range(n) if td[jmin][k] == rmin]
        elif all(v <= rmin for v in gjstar):
            break
        else:
            rmin = min(v for v in gjstar if v > rmin)
            jmin = next(j for j in range(n) if gjstar[j] == rmin)
            kmin = [k for k in range(n) if td[jmin][k] == rmin]

    return td, gjstar


def _training_distance_matrix(points: list[Point]) -> list[_Row]:
    """Symmetric great-circle distance matrix (km), ``None`` on the diagonal."""
    n = len(points)
    tdist: list[_Row] = [[None] * n for _ in range(n)]
    for j in range(n):
        for k in range(j + 1, n):
            d = haversine_km(points[j], points[k])
            tdist[j][k] = d
            tdist[k][j] = d
    return tdist


def nndm_loo(
    points: list[Point],
    pred_points: list[Point],
    *,
    phi: float | None = None,
    min_train: float = 0.5,
) -> dict[str, Any]:
    """Build NNDM Leave-One-Out CV folds and a CV-representativeness diagnostic.

    Parameters
    ----------
    points
        Training label locations as ``(lon, lat)`` degrees.
    pred_points
        A sample of locations from the prediction area (e.g. raster-cell
        centroids), ``(lon, lat)`` degrees. These define the target
        nearest-neighbour-distance distribution NNDM matches to.
    phi
        Autocorrelation range in km; distance matching is limited to this
        range. ``None`` (default) uses the maximum observed distance (CAST's
        ``phi="max"``), i.e. match over the whole prediction area.
    min_train
        Minimum fraction of the training set each fold must keep (default 0.5).

    Returns
    -------
    dict
        ``folds`` (per left-out point: ``test``, ``train`` indices, ``exclude``
        indices), the matched ``gjstar`` distances, and a diagnostic comparing
        the LOO vs NNDM test-to-train NN-distance ECDFs against the
        prediction-to-train ECDF (``ecdf_mismatch_loo`` /
        ``ecdf_mismatch_nndm`` / ``mismatch_reduction``), plus a recommendation.
    """
    n = len(points)
    if n < 2:
        raise ValueError("need at least 2 training points")
    if not pred_points:
        raise ValueError("need at least 1 prediction point")
    if not 0.0 <= min_train <= 1.0:
        raise ValueError("min_train must be between 0 and 1")
    if phi is not None and phi <= 0:
        raise ValueError("phi must be positive (or None for the max distance)")

    tdist = _training_distance_matrix(points)
    gij = [min(haversine_km(p, t) for t in points) for p in pred_points]
    gj = [_row_min(tdist[j]) for j in range(n)]  # plain LOO NN distances

    if phi is None:
        all_t = [v for row in tdist for v in row if v is not None]
        phi = max(max(gij), max(all_t)) + 1e-9

    td, gjstar = _nndm_core(tdist, gij, phi, min_train)

    folds: list[dict[str, Any]] = []
    n_excluded = 0
    for i in range(n):
        train = [k for k in range(n) if td[i][k] is not None]
        exclude = [k for k in range(n) if td[i][k] is None and k != i]
        n_excluded += len(exclude)
        folds.append({"test": i, "train": train, "exclude": exclude})

    optimism_loo = cv_optimism_km(gj, gij)
    optimism_nndm = cv_optimism_km(gjstar, gij)
    reduction = (
        (optimism_loo - optimism_nndm) / optimism_loo if optimism_loo > 0 else 0.0
    )
    folds_with_excl = sum(1 for f in folds if f["exclude"])

    if optimism_loo <= 1e-9:
        recommendation = (
            "Training and prediction nearest-neighbour distances already align; "
            "plain LOO cross-validation is unbiased here."
        )
    elif reduction >= 0.15:
        recommendation = (
            "Plain LOO/random CV would overstate map accuracy here. Use the NNDM "
            "folds (train on each fold's 'train' indices, test on its 'test' "
            "index) for an unbiased estimate over the prediction area."
        )
    else:
        recommendation = (
            "Small optimistic bias corrected; the NNDM folds give a slightly more "
            "honest accuracy estimate than plain LOO."
        )

    return {
        "n_train": n,
        "n_pred": len(pred_points),
        "phi_km": round(phi, 4),
        "min_train": min_train,
        "n_excluded_total": n_excluded,
        "mean_exclude_per_fold": round(n_excluded / n, 3),
        "max_exclude_per_fold": max(len(f["exclude"]) for f in folds),
        "folds_with_exclusions": folds_with_excl,
        "cv_optimism_loo_km": round(optimism_loo, 4),
        "cv_optimism_nndm_km": round(optimism_nndm, 4),
        "optimism_reduction": round(reduction, 3),
        "recommendation": recommendation,
        "folds": folds,
        "gjstar_km": [round(v, 4) for v in gjstar],
    }
