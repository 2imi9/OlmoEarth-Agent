# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Tests for the pure raster-comparison stats."""

from __future__ import annotations

from olmoearth_agent.analysis.raster_compare import (
    compare_categorical,
    compare_narration,
    compare_numeric,
    grid_points,
    intersect_bbox,
    normalize_kind,
    pearson,
)


def test_intersect_bbox_overlap() -> None:
    assert intersect_bbox([0, 0, 10, 10], [5, 5, 20, 20]) == [5, 5, 10, 10]


def test_intersect_bbox_disjoint_is_none() -> None:
    assert intersect_bbox([0, 0, 1, 1], [2, 2, 3, 3]) is None
    assert intersect_bbox(None, [0, 0, 1, 1]) is None


def test_grid_points_count_and_interior() -> None:
    pts = grid_points([0, 0, 10, 10], 4)
    assert len(pts) == 16
    # all strictly inside the bbox (cell centers, never on the edge)
    assert all(0 < lon < 10 and 0 < lat < 10 for lon, lat in pts)


def test_pearson_perfect_and_constant() -> None:
    assert pearson([1, 2, 3], [2, 4, 6]) == 1.0  # perfectly correlated
    assert pearson([1, 1, 1], [2, 3, 4]) is None  # a constant has no correlation


def test_compare_numeric_identical_is_full_agreement() -> None:
    pairs = [(0.1, 0.1), (0.5, 0.5), (0.9, 0.9)]
    s = compare_numeric(pairs, tolerance=0.0)
    assert s["n_samples"] == 3
    assert s["mean_abs_diff"] == 0.0
    assert s["agreement_fraction"] == 1.0
    assert s["correlation"] == 1.0


def test_compare_numeric_drops_none_and_scores_diff() -> None:
    pairs = [(0.2, 0.4), (None, 0.5), (0.6, 0.6), (0.8, None)]
    s = compare_numeric(pairs, tolerance=0.1)
    assert s["n_samples"] == 2  # the two None pairs dropped
    assert s["mean_diff_b_minus_a"] == 0.1  # (0.2 + 0.0) / 2
    assert s["agreement_fraction"] == 0.5  # only (0.6,0.6) within 0.1


def test_compare_numeric_empty() -> None:
    assert compare_numeric([(None, None)], tolerance=0.1)["n_samples"] == 0


def test_compare_categorical_agreement() -> None:
    s = compare_categorical([("a", "a"), ("a", "b"), ("c", "c"), (None, "c")])
    assert s["n_samples"] == 3
    assert s["n_disagree"] == 1
    assert s["agreement_fraction"] == round(2 / 3, 4)


def test_normalize_kind() -> None:
    assert normalize_kind("temporal") == "temporal"
    assert normalize_kind("cross_model") == "cross_model"
    assert normalize_kind(None) == "cross_model"  # default
    assert normalize_kind("nonsense") == "cross_model"  # unknown -> default


def test_narration_cross_model_regression_frames_agreement() -> None:
    stats = {"n_samples": 9, "agreement_fraction": 0.78, "mean_diff_b_minus_a": 0.05}
    nar = compare_narration(stats, kind="cross_model", value_type="regression")
    assert nar["kind"] == "cross_model"
    assert nar["labels"] == {"a": "model A", "b": "model B", "diff": "difference (B - A)"}
    assert "two models agree" in nar["headline"]
    assert "78% of 9 cells" in nar["headline"]
    assert "model-vs-model agreement" in nar["framing"]
    assert "accuracy" in nar["framing"]


def test_narration_cross_model_classification() -> None:
    nar = compare_narration(
        {"n_samples": 4, "agreement_fraction": 0.5}, kind="cross_model",
        value_type="classification",
    )
    assert nar["headline"] == "the two models agree on 50% of 4 cells"


def test_narration_temporal_regression_increase() -> None:
    stats = {"n_samples": 9, "agreement_fraction": 1.0, "mean_diff_b_minus_a": 0.05}
    nar = compare_narration(stats, kind="temporal", value_type="regression")
    assert nar["labels"] == {
        "a": "earlier", "b": "later", "diff": "change (later - earlier)"
    }
    assert nar["headline"] == "net increase of 0.05 (later - earlier) across 9 cells"
    assert "change over time" in nar["framing"]
    assert "not model-vs-model agreement" in nar["framing"]


def test_narration_temporal_regression_decrease_uses_magnitude() -> None:
    nar = compare_narration(
        {"n_samples": 9, "mean_diff_b_minus_a": -0.2}, kind="temporal",
        value_type="regression",
    )
    # sign is conveyed by the word, magnitude is positive
    assert nar["headline"] == "net decrease of 0.2 (later - earlier) across 9 cells"


def test_narration_temporal_regression_no_change() -> None:
    nar = compare_narration(
        {"n_samples": 9, "mean_diff_b_minus_a": 0.0}, kind="temporal",
        value_type="regression",
    )
    assert nar["headline"] == "no net change (later - earlier) across 9 cells"


def test_narration_temporal_classification_frames_change() -> None:
    nar = compare_narration(
        {"n_samples": 9, "agreement_fraction": 0.9}, kind="temporal",
        value_type="classification",
    )
    assert nar["headline"] == "class unchanged in 90% of 9 cells"


def test_narration_unknown_kind_falls_back_to_cross_model() -> None:
    nar = compare_narration(
        {"n_samples": 1, "agreement_fraction": 1.0}, kind="bogus",
        value_type="regression",
    )
    assert nar["kind"] == "cross_model"
