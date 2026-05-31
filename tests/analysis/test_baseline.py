# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Tests for the OlmoEarth-vs-AlphaEarth baseline comparison (skill #7)."""

from __future__ import annotations

import pytest

from olmoearth_agent.analysis.baseline import compare_metrics, difference_raster

_TRUTH = [1, 1, 0, 0]
_PERFECT = [1, 1, 0, 0]
_HALF = [1, 0, 0, 1]


def test_olmoearth_wins_all_metrics() -> None:
    out = compare_metrics(_TRUTH, _PERFECT, _HALF)
    assert out["overall_winner"] == "olmoearth"
    acc = out["comparison"][0]
    assert acc["metric"] == "accuracy"
    assert acc["olmoearth"] == 1.0
    assert acc["alphaearth"] == 0.5
    assert acc["delta"] == 0.5
    assert acc["winner"] == "olmoearth"


def test_tie_when_predictions_identical() -> None:
    out = compare_metrics(_TRUTH, _HALF, _HALF)
    assert out["overall_winner"] == "tie"
    assert all(row["winner"] == "tie" for row in out["comparison"])


def test_alphaearth_wins_when_better() -> None:
    out = compare_metrics(_TRUTH, _HALF, _PERFECT)
    assert out["overall_winner"] == "alphaearth"


def test_custom_labels() -> None:
    out = compare_metrics(_TRUTH, _PERFECT, _HALF, label_a="ours", label_b="theirs")
    assert out["overall_winner"] == "ours"
    assert out["comparison"][0]["winner"] == "ours"
    assert "ours" in out["comparison"][0]


def test_reuses_full_classification_metrics() -> None:
    out = compare_metrics(_TRUTH, _PERFECT, _HALF)
    assert out["model_a"]["metrics"]["accuracy"] == 1.0
    assert "per_class" in out["model_b"]["metrics"]


def test_compare_metrics_propagates_length_error() -> None:
    with pytest.raises(ValueError, match="same length"):
        compare_metrics(_TRUTH, [1, 1, 0], _HALF)


def test_difference_raster_basics() -> None:
    out = difference_raster([0.9, 0.8, 0.5], [0.5, 0.5, 0.5])
    assert out["n_cells"] == 3
    assert out["max_abs_difference"] == 0.4
    assert out["olmoearth_higher_fraction"] == pytest.approx(2 / 3)
    assert out["alphaearth_higher_fraction"] == 0.0
    assert out["tie_fraction"] == pytest.approx(1 / 3)
    assert out["differences"] == [0.4, 0.3, 0.0]


def test_difference_raster_tolerance_counts_small_gaps_as_tie() -> None:
    out = difference_raster([1.0, 1.0], [0.999, 0.5], tol=0.01)
    # first cell within tol -> tie; second clearly olmoearth-higher
    assert out["tie_fraction"] == 0.5
    assert out["olmoearth_higher_fraction"] == 0.5


def test_difference_raster_rejects_unequal_length() -> None:
    with pytest.raises(ValueError, match="same length"):
        difference_raster([0.1, 0.2], [0.1])


def test_difference_raster_rejects_empty() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        difference_raster([], [])
