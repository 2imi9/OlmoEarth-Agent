# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Tests for spatial cross-validation and the inflation diagnostic."""

from __future__ import annotations

import pytest

from olmoearth_agent.evaluation.spatial_cv import (
    cv_inflation_diagnostic,
    haversine_km,
    random_folds,
    spatial_block_folds,
)


def test_haversine_one_degree_latitude() -> None:
    # 1 degree of latitude is ~111 km anywhere.
    assert haversine_km((0.0, 0.0), (0.0, 1.0)) == pytest.approx(111.19, abs=0.5)


def test_haversine_zero() -> None:
    assert haversine_km((10.0, 20.0), (10.0, 20.0)) == pytest.approx(0.0)


def test_spatial_block_keeps_same_block_together() -> None:
    # Two clusters far apart; block_deg=0.5 puts each cluster in one block.
    points = [(0.0, 0.0), (0.1, 0.1), (10.0, 10.0), (10.1, 10.1)]
    folds = spatial_block_folds(points, n_folds=2, block_deg=0.5)
    assert folds[0] == folds[1]  # cluster A together
    assert folds[2] == folds[3]  # cluster B together
    assert folds[0] != folds[2]  # different clusters -> different folds


def test_random_folds_are_balanced() -> None:
    points = [(float(i), 0.0) for i in range(20)]
    folds = random_folds(points, n_folds=5, seed=1)
    counts = [folds.count(f) for f in range(5)]
    assert counts == [4, 4, 4, 4, 4]


def test_spatial_block_validates_args() -> None:
    with pytest.raises(ValueError):
        spatial_block_folds([(0.0, 0.0)], n_folds=1)
    with pytest.raises(ValueError):
        spatial_block_folds([(0.0, 0.0)], n_folds=2, block_deg=0)


def test_inflation_diagnostic_flags_clustered_data() -> None:
    # Three tight clusters ~10 degrees apart. Random CV leaks neighbours
    # across folds (tiny test-to-train distance); spatial CV holds out a
    # whole cluster (large distance) -> high inflation ratio.
    clusters = [(0.0, 0.0), (10.0, 0.0), (20.0, 0.0)]
    points = [
        (cx + 0.02 * k, cy + 0.02 * k) for cx, cy in clusters for k in range(5)
    ]
    result = cv_inflation_diagnostic(points, n_folds=3, block_deg=0.5)
    assert result["mean_test_to_train_km_spatial"] > result[
        "mean_test_to_train_km_random"
    ]
    assert result["spatial_to_random_ratio"] >= 1.5
    assert result["inflation_risk"] in ("moderate", "high")


def test_inflation_diagnostic_needs_enough_points() -> None:
    with pytest.raises(ValueError):
        cv_inflation_diagnostic([(0.0, 0.0)], n_folds=5)
