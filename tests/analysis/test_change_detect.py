# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Tests for the change-detection trajectory diff (skill #6)."""

from __future__ import annotations

import pytest

from olmoearth_agent.analysis.change_detect import (
    TooFewDatesError,
    diff_layers,
    enforce_min_3_dates,
)


def test_enforce_passes_for_three_distinct_dates() -> None:
    enforce_min_3_dates(["2024-01-01", "2024-02-01", "2024-03-01"])  # no raise


def test_enforce_refuses_two_dates() -> None:
    with pytest.raises(TooFewDatesError):
        enforce_min_3_dates(["2024-01-01", "2024-02-01"])


def test_enforce_counts_distinct_not_length() -> None:
    # three entries but only two distinct dates is still a refusal
    with pytest.raises(TooFewDatesError):
        enforce_min_3_dates(["2024-01-01", "2024-01-01", "2024-02-01"])


def test_diff_layers_refuses_naive_two_date() -> None:
    with pytest.raises(TooFewDatesError):
        diff_layers([("2024-01-01", 0.1), ("2024-06-01", 0.5)])


def test_diff_layers_rejects_duplicate_dates() -> None:
    # 3 distinct dates clears enforce_min_3_dates; the 4th repeats one,
    # which is an ambiguous trajectory point and is rejected.
    with pytest.raises(ValueError, match="duplicate"):
        diff_layers(
            [
                ("2024-01-01", 0.1),
                ("2024-02-01", 0.2),
                ("2024-03-01", 0.3),
                ("2024-03-01", 0.4),
            ]
        )


def test_diff_layers_falls_back_to_order_for_non_iso_labels() -> None:
    # Ordinal labels (caller passed bare ordered values, no calendar dates) →
    # input order is the trajectory order; no error, since the trajectory math
    # only needs order. Real ISO dates still sort (see test_diff_layers_sorts).
    out = diff_layers([("t1", 0.1), ("t2", 0.2), ("t3", 0.35)])
    assert out["trend"] == "increasing"
    assert out["values"] == [0.1, 0.2, 0.35]
    assert out["dates"] == ["t1", "t2", "t3"]


def test_monotonic_increase() -> None:
    out = diff_layers([("2024-01-01", 0.1), ("2024-02-01", 0.2), ("2024-03-01", 0.35)])
    assert out["n_dates"] == 3
    assert out["trend"] == "increasing"
    assert out["monotonic"] is True
    assert out["reversals"] == 0
    assert out["net_change"] == pytest.approx(0.25)
    assert out["step_deltas"] == pytest.approx([0.1, 0.15])
    assert out["max_step"]["interval"] == ["2024-02-01", "2024-03-01"]
    assert out["max_step"]["delta"] == pytest.approx(0.15)


def test_monotonic_decrease() -> None:
    out = diff_layers([("2024-01-01", 0.9), ("2024-02-01", 0.5), ("2024-03-01", 0.2)])
    assert out["trend"] == "decreasing"
    assert out["monotonic"] is True
    assert out["net_change"] == pytest.approx(-0.7)


def test_stable_when_flat() -> None:
    out = diff_layers([("2024-01-01", 0.3), ("2024-02-01", 0.3), ("2024-03-01", 0.3)])
    assert out["trend"] == "stable"
    assert out["reversals"] == 0
    assert out["monotonic"] is True
    assert out["net_change"] == 0.0


def test_oscillating_counts_reversal_and_picks_largest_step() -> None:
    # a peak then recede: a two-date (Jan->Mar) diff would call this +0.1
    out = diff_layers([("2024-01-01", 0.1), ("2024-02-01", 0.5), ("2024-03-01", 0.2)])
    assert out["trend"] == "oscillating"
    assert out["monotonic"] is False
    assert out["reversals"] == 1
    assert out["net_change"] == pytest.approx(0.1)
    # largest-magnitude step is the Jan->Feb rise, not the net
    assert out["max_step"]["interval"] == ["2024-01-01", "2024-02-01"]
    assert out["max_step"]["delta"] == pytest.approx(0.4)


def test_series_is_sorted_by_date() -> None:
    out = diff_layers([("2024-03-01", 3.0), ("2024-01-01", 1.0), ("2024-02-01", 2.0)])
    assert out["dates"] == ["2024-01-01", "2024-02-01", "2024-03-01"]
    assert out["values"] == [1.0, 2.0, 3.0]


def test_tolerance_collapses_tiny_dip() -> None:
    series = [("2024-01-01", 0.0), ("2024-02-01", -0.0005), ("2024-03-01", 0.5)]
    # with no tolerance the tiny dip is a real reversal
    strict = diff_layers(series)
    assert strict["trend"] == "oscillating"
    assert strict["reversals"] == 1
    # with tolerance the dip is noise: a clean increase
    smoothed = diff_layers(series, tol=0.001)
    assert smoothed["trend"] == "increasing"
    assert smoothed["reversals"] == 0
    assert smoothed["monotonic"] is True


def test_handles_iso_datetimes_with_z() -> None:
    out = diff_layers(
        [
            ("2024-03-01T00:00:00Z", 0.3),
            ("2024-01-01T00:00:00Z", 0.1),
            ("2024-02-01T00:00:00Z", 0.2),
        ]
    )
    assert out["dates"][0] == "2024-01-01T00:00:00Z"
    assert out["trend"] == "increasing"
