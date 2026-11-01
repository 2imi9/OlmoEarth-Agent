# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Tests for the ensemble confidence map (skill #4, compare_results' ensemble mode)."""

from __future__ import annotations

import pytest

from olmoearth_agent.analysis.uncertainty import prediction_confidence


def test_confidence_regression_low_variance_is_high() -> None:
    out = prediction_confidence([[10.0, 10.1, 9.9, 10.0]], value_type="regression")
    pp = out["per_point"][0]
    assert pp["mean"] == pytest.approx(10.0, abs=0.05)
    assert pp["std"] < 0.1
    assert pp["coefficient_of_variation"] < 0.02
    assert pp["confidence"] > 0.97
    assert "epistemic" in out["caveat"].lower()
    assert "not a calibrated probability" in out["caveat"].lower()


def test_confidence_regression_high_variance_is_lower() -> None:
    out = prediction_confidence([[1.0, 50.0, 100.0, 5.0]])
    pp = out["per_point"][0]
    assert pp["std"] > 30
    assert pp["coefficient_of_variation"] > 0.8
    low = prediction_confidence([[10.0, 10.1, 9.9, 10.0]])["per_point"][0]["confidence"]
    assert pp["confidence"] < low  # more spread -> less confidence


def test_confidence_categorical_unanimous_is_entropy_zero() -> None:
    out = prediction_confidence(
        [["forest", "forest", "forest"]], value_type="categorical"
    )
    pp = out["per_point"][0]
    assert pp["majority_class"] == "forest"
    assert pp["entropy"] == pytest.approx(0.0)
    assert pp["margin"] == pytest.approx(1.0)
    assert pp["confidence"] == pytest.approx(1.0)
    assert out["summary"]["n_unanimous"] == 1


def test_confidence_categorical_even_split_is_max_entropy() -> None:
    out = prediction_confidence([["a", "b", "a", "b"]], value_type="categorical")
    pp = out["per_point"][0]
    assert pp["entropy"] == pytest.approx(1.0)
    assert pp["confidence"] == pytest.approx(0.0)
    assert pp["margin"] == pytest.approx(0.0)
    assert pp["vote_fractions"]["a"] == pytest.approx(0.5)


def test_confidence_single_sample_is_degenerate() -> None:
    out = prediction_confidence([[42.0]], value_type="regression")
    pp = out["per_point"][0]
    assert pp["std"] == pytest.approx(0.0)
    assert pp["coefficient_of_variation"] == pytest.approx(0.0)
    assert pp["confidence"] == pytest.approx(1.0)
    assert out["summary"]["n_degenerate"] == 1
    assert out["ensemble_size"]["min"] == 1


def test_confidence_zero_mean_cv_is_guarded_not_crash() -> None:
    out = prediction_confidence([[-1.0, 1.0, -1.0, 1.0]], value_type="regression")
    pp = out["per_point"][0]
    assert pp["mean"] == pytest.approx(0.0)
    assert pp["coefficient_of_variation"] is None  # undefined, not a crash
    assert out["summary"]["n_confidence_undefined"] == 1


def test_confidence_rejects_empty_and_bad_value_type() -> None:
    with pytest.raises(ValueError, match="at least one"):
        prediction_confidence([])
    with pytest.raises(ValueError, match=">= 1"):
        prediction_confidence([[]])
    with pytest.raises(ValueError, match="value_type"):
        prediction_confidence([[1.0]], value_type="bogus")
