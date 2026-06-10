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


# ---------------------------------------------------------------------------
# group (N-result) comparison
# ---------------------------------------------------------------------------


def test_intersect_bboxes_folds_and_rejects_disjoint() -> None:
    from olmoearth_agent.analysis.raster_compare import intersect_bboxes

    assert intersect_bboxes(
        [[0, 0, 10, 10], [5, 5, 20, 20], [0, 0, 8, 8]]
    ) == [5, 5, 8, 8]
    assert intersect_bboxes([[0, 0, 1, 1], [2, 2, 3, 3], [0, 0, 9, 9]]) is None
    assert intersect_bboxes([[0, 0, 1, 1], None]) is None
    assert intersect_bboxes([]) is None


def test_compare_group_numeric_pairwise_and_consensus() -> None:
    from olmoearth_agent.analysis.raster_compare import compare_group_numeric

    # Three aligned series: B = A + 0.05 (within tolerance), C = A + 0.5 (out).
    a = [1.0, 2.0, 3.0, 4.0]
    b = [v + 0.05 for v in a]
    c = [v + 0.5 for v in a]
    out = compare_group_numeric([a, b, c], tolerance=0.1)

    assert len(out["pairwise"]) == 3  # (0,1), (0,2), (1,2)
    ab = next(p for p in out["pairwise"] if (p["a_index"], p["b_index"]) == (0, 1))
    assert ab["stats"]["agreement_fraction"] == 1.0
    ac = next(p for p in out["pairwise"] if (p["a_index"], p["b_index"]) == (0, 2))
    assert ac["stats"]["agreement_fraction"] == 0.0

    ens = out["ensemble"]
    assert ens["n_points_used"] == 4
    assert ens["consensus_fraction"] == 0.0  # spread 0.5 > tolerance everywhere
    assert ens["max_spread"] == 0.5
    # every point is a hotspot candidate; capped list carries index + spread
    spots = ens["top_disagreement_points"]
    assert spots and all(s["spread"] == 0.5 and s["n_models"] == 3 for s in spots)
    # the most divergent pair must involve C (index 2)
    assert 2 in (out["most_divergent_pair"]["a_index"], out["most_divergent_pair"]["b_index"])


def test_compare_group_numeric_full_consensus_and_nones() -> None:
    from olmoearth_agent.analysis.raster_compare import compare_group_numeric

    a = [1.0, None, 3.0]
    b = [1.01, 2.0, None]  # only point 0 has 2+ valid values
    out = compare_group_numeric([a, b], tolerance=0.1)
    ens = out["ensemble"]
    assert ens["n_points_used"] == 1
    assert ens["consensus_fraction"] == 1.0
    assert ens["top_disagreement_points"] == []  # within tolerance -> no hotspot


def test_compare_group_numeric_no_usable_points() -> None:
    from olmoearth_agent.analysis.raster_compare import compare_group_numeric

    out = compare_group_numeric([[None, 1.0], [2.0, None]], tolerance=0.1)
    assert out["ensemble"]["n_points_used"] == 0
    assert "note" in out["ensemble"]
    assert out["most_divergent_pair"] is None  # no pair has samples


def test_compare_group_categorical_unanimity_and_majority() -> None:
    from olmoearth_agent.analysis.raster_compare import compare_group_categorical

    a = ["water", "urban", "forest"]
    b = ["water", "urban", "crop"]
    c = ["water", "bare", "crop"]
    out = compare_group_categorical([a, b, c])

    assert len(out["pairwise"]) == 3
    ens = out["ensemble"]
    assert ens["n_points_used"] == 3
    assert ens["unanimous_fraction"] == round(1 / 3, 4)  # only point 0
    # point 1: majority 2/3 (urban); point 2: majority 2/3 (crop); point 0: 1.0
    assert ens["mean_majority_share"] == round((1.0 + 2 / 3 + 2 / 3) / 3, 4)
    spots = ens["top_disagreement_points"]
    assert spots and all(s["majority_share"] < 1.0 for s in spots)


def test_compare_group_narration_headlines() -> None:
    from olmoearth_agent.analysis.raster_compare import (
        compare_group_narration,
    )

    nar = compare_group_narration(
        {"ensemble": {"n_points_used": 9, "consensus_fraction": 0.5}},
        n_results=3,
        value_type="regression",
    )
    assert "all 3 models agree within tolerance at 50% of 9 cells" == nar["headline"]
    assert "not accuracy" in nar["framing"]

    nar_cls = compare_group_narration(
        {"ensemble": {"n_points_used": 4, "unanimous_fraction": 0.25}},
        n_results=4,
        value_type="classification",
    )
    assert "all 4 models pick the same class at 25% of 4 cells" == nar_cls["headline"]

    empty = compare_group_narration(
        {"ensemble": {"n_points_used": 0}}, n_results=3, value_type="regression"
    )
    assert "no overlapping valid cells" in empty["headline"]
