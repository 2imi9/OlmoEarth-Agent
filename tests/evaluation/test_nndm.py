# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Tests for the NNDM-LOO port, verified against the CAST reference algorithm.

Two independent cross-checks of correctness:
1. ``_nndm_core`` reproduces a by-hand trace of CAST's ``R/nndm.R`` while-loop
   on a fixed 4-point distance matrix (RNG-independent, exact).
2. The ``predpoints == trainpoints`` invariant that every such case in CAST's
   own ``test-nndm.R`` relies on: the matching never fires, so the folds reduce
   to plain LOO (no exclusions, ``Gjstar == Gj``).
"""

from __future__ import annotations

import pytest

from olmoearth_agent.evaluation.nndm import _nndm_core, cv_optimism_km, nndm_loo
from olmoearth_agent.evaluation.spatial_cv import haversine_km


def _excludes(td: list[list[float | None]]) -> list[list[int]]:
    """Reconstruct per-row exclusion indices (None, off-diagonal)."""
    n = len(td)
    return [[k for k in range(n) if td[i][k] is None and k != i] for i in range(n)]


def test_core_matches_hand_traced_cast_reference() -> None:
    # 4 training points on a line (x = 0, 1, 2, 10); symmetric, None diagonal.
    tdist: list[list[float | None]] = [
        [None, 1.0, 2.0, 10.0],
        [1.0, None, 1.0, 9.0],
        [2.0, 1.0, None, 8.0],
        [10.0, 9.0, 8.0, None],
    ]
    gij = [5.0, 5.0, 5.0, 5.0]  # prediction-to-train NN distances (dispersed)
    phi = max(max(gij), 10.0) + 1e-9  # CAST phi="max"
    td, gjstar = _nndm_core(tdist, gij, phi, min_train=0.5)

    # Hand-traced expected result of CAST's matching loop (see module docstring).
    assert gjstar == [2.0, 9.0, 2.0, 8.0]
    assert _excludes(td) == [[1], [0, 2], [1], []]
    # The input matrix must not be mutated.
    assert tdist[0][1] == 1.0


def test_core_phi_zero_disables_matching() -> None:
    tdist: list[list[float | None]] = [
        [None, 1.0, 2.0],
        [1.0, None, 1.5],
        [2.0, 1.5, None],
    ]
    td, gjstar = _nndm_core(tdist, [9.0, 9.0], phi=0.0, min_train=0.5)
    assert gjstar == [1.0, 1.0, 1.5]  # == plain LOO NN distances
    assert _excludes(td) == [[], [], []]


def test_predpoints_equal_trainpoints_reduces_to_plain_loo() -> None:
    # CAST invariant: when predictions sit exactly on the training points, the
    # prediction-to-train NN distances are all 0, the matching condition can
    # never fire, and NNDM == plain LOO (no exclusions, Gjstar == Gj).
    points = [(0.0, 0.0), (0.1, 0.0), (0.25, 0.0), (1.0, 1.0), (1.2, 1.0)]
    res = nndm_loo(points, pred_points=points)
    assert res["n_excluded_total"] == 0
    assert res["folds_with_exclusions"] == 0
    expected_gj = [
        round(min(haversine_km(p, q) for q in points if q != p), 4) for p in points
    ]
    assert res["gjstar_km"] == expected_gj


def test_clustered_training_triggers_exclusions_and_reduces_mismatch() -> None:
    # Two tight training clusters (small NN distances) but predictions spread
    # over a wide area (large NN distances) -> big LOO/prediction mismatch that
    # NNDM should correct by excluding near neighbours.
    cluster_a = [(0.00 + 0.01 * i, 0.00) for i in range(5)]
    cluster_b = [(5.00 + 0.01 * i, 5.00) for i in range(5)]
    points = cluster_a + cluster_b
    pred_points = [(x, y) for x in (-2.0, 0.0, 2.0, 4.0, 6.0) for y in (-2.0, 2.0, 6.0)]

    res = nndm_loo(points, pred_points)
    assert res["n_excluded_total"] > 0
    assert res["folds_with_exclusions"] > 0
    # NNDM reduces the optimistic bias relative to plain LOO.
    assert res["cv_optimism_nndm_km"] < res["cv_optimism_loo_km"]
    assert res["optimism_reduction"] > 0.0


def test_nndm_never_increases_optimism() -> None:
    # General property: NNDM never makes the optimistic bias worse than LOO.
    configs = [
        ([(0.0, 0.0), (0.5, 0.0), (1.0, 0.0), (3.0, 3.0)], [(2.0, 2.0), (4.0, 4.0)]),
        (
            [(0.0, 0.0), (0.1, 0.1), (0.2, 0.0), (2.0, 2.0), (2.1, 2.0)],
            [(1.0, 1.0), (3.0, 1.0), (-1.0, 0.5)],
        ),
    ]
    for points, preds in configs:
        res = nndm_loo(points, preds)
        assert res["cv_optimism_nndm_km"] <= res["cv_optimism_loo_km"] + 1e-6


def test_fold_structure_invariants() -> None:
    points = [(0.0, 0.0), (0.2, 0.1), (0.4, 0.0), (3.0, 3.0), (3.1, 3.2), (3.0, 2.8)]
    pred_points = [(1.0, 1.0), (2.0, 2.0), (1.5, 0.5)]
    res = nndm_loo(points, pred_points)
    n = res["n_train"]
    for fold in res["folds"]:
        i = fold["test"]
        assert i not in fold["train"]  # a point never trains on itself
        assert i not in fold["exclude"]
        assert set(fold["train"]).isdisjoint(fold["exclude"])
        # train + exclude + {i} partition the index set.
        assert set(fold["train"]) | set(fold["exclude"]) | {i} == set(range(n))
        assert len(fold["train"]) >= 1  # a fold is never emptied


def test_cv_optimism_is_one_sided() -> None:
    assert cv_optimism_km([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0.0
    # Test distances SHORTER than prediction distances -> optimistic bias > 0.
    assert cv_optimism_km([0.0, 0.0], [10.0, 10.0]) > 0.0
    # Test distances LONGER than prediction distances -> no optimism (one-sided).
    assert cv_optimism_km([10.0, 10.0], [0.0, 0.0]) == 0.0


def test_validation_errors() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        nndm_loo([(0.0, 0.0)], [(1.0, 1.0)])
    with pytest.raises(ValueError, match="at least 1 prediction"):
        nndm_loo([(0.0, 0.0), (1.0, 1.0)], [])
    with pytest.raises(ValueError, match="min_train"):
        nndm_loo([(0.0, 0.0), (1.0, 1.0)], [(1.0, 1.0)], min_train=1.5)
    with pytest.raises(ValueError, match="phi"):
        nndm_loo([(0.0, 0.0), (1.0, 1.0)], [(1.0, 1.0)], phi=-1.0)
