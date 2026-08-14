# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Unit tests for the timeseries shift-trace math (analysis/trace_shifts.py)."""

from __future__ import annotations

import pytest

from olmoearth_agent.analysis.trace_shifts import (
    ols_slope,
    trace_categorical,
    trace_narration,
    trace_numeric,
)


def test_ols_slope_linear_and_degenerate() -> None:
    assert ols_slope([(0.0, 1.0), (1.0, 1.2), (2.0, 1.4)]) == pytest.approx(0.2)
    assert ols_slope([(0.0, 1.0)]) is None  # one point
    assert ols_slope([(1.0, 2.0), (1.0, 3.0)]) is None  # vertical


def test_trace_numeric_linear_series() -> None:
    # 3 results x 4 points, each step adds exactly 0.2 everywhere.
    base = [0.1, 0.2, 0.3, 0.4]
    series = [
        base,
        [v + 0.2 for v in base],
        [v + 0.4 for v in base],
    ]
    out = trace_numeric(series, tolerance=0.1, value_range=(0.0, 1.0))
    assert out["n_results"] == 3
    assert out["n_points"] == 4
    # Per-step stats reuse the compare shape: perfect correlation, delta 0.2.
    assert len(out["steps"]) == 2
    for step in out["steps"]:
        assert step["stats"]["mean_diff_b_minus_a"] == 0.2
        assert step["stats"]["correlation"] == 1.0
    trajectory = out["trajectory"]
    assert trajectory["n_points_used"] == 4
    assert trajectory["mean_total_change"] == 0.4
    assert trajectory["mean_slope_per_step"] == 0.2
    assert trajectory["increased_fraction"] == 1.0
    assert trajectory["shifted_fraction"] == 1.0
    # Legend calibration: supplied range 0-1 -> 0.4 = 40% of range.
    assert out["calibration"]["source"] == "supplied"
    assert trajectory["mean_total_change_fraction_of_range"] == 0.4
    top = trajectory["top_shift_points"][0]
    assert top["total_change"] == 0.4
    assert top["largest_step_between"] == [1, 2]  # last equal-size step wins >=


def test_trace_numeric_flat_series_and_observed_range() -> None:
    series = [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]]
    out = trace_numeric(series, tolerance=0.1, value_range=None)
    trajectory = out["trajectory"]
    assert trajectory["mean_total_change"] == 0.0
    assert trajectory["shifted_fraction"] == 0.0
    assert trajectory["top_shift_points"] == []
    calib = out["calibration"]
    assert calib["source"] == "observed"
    # Constant series -> zero-width observed range -> fraction is None.
    assert trajectory["mean_total_change_fraction_of_range"] is None
    narration = trace_narration(out, value_type="regression", ordering="chronological")
    assert "no net change" in narration["headline"]
    assert "not verified ground change" in narration["framing"]


def test_trace_numeric_skips_sparse_points_and_flags_jump() -> None:
    # Point 0: valid everywhere with one big jump at step 1->2.
    # Point 1: only one valid sample -> excluded from trajectories.
    series = [
        [0.1, None],
        [0.1, 0.9],
        [0.6, None],
    ]
    out = trace_numeric(series, tolerance=0.1, value_range=(0.0, 1.0))
    trajectory = out["trajectory"]
    assert trajectory["n_points_used"] == 1
    top = trajectory["top_shift_points"][0]
    assert top["total_change"] == 0.5
    assert top["largest_step"] == 0.5
    assert top["largest_step_between"] == [1, 2]


def test_trace_numeric_empty_when_no_point_has_two_samples() -> None:
    out = trace_numeric([[None, 1.0], [None, None], [None, None]], tolerance=0.1)
    assert out["trajectory"]["n_points_used"] == 0
    narration = trace_narration(out, value_type="regression", ordering="chronological")
    assert "no grid point" in narration["headline"]


def test_trace_categorical_transitions_and_stability() -> None:
    series = [
        ["forest", "water", "urban"],
        ["forest", "crop", "urban"],
        ["crop", "crop", "urban"],
    ]
    out = trace_categorical(series)
    assert len(out["steps"]) == 2
    trajectory = out["trajectory"]
    assert trajectory["n_points_used"] == 3
    assert trajectory["net_changed_fraction"] == round(2 / 3, 4)
    assert trajectory["stable_fraction"] == round(1 / 3, 4)
    transitions = {(t["from"], t["to"]): t["count"] for t in out["transitions"]}
    assert transitions[("water", "crop")] == 1
    assert transitions[("forest", "crop")] == 1
    narration = trace_narration(
        out, value_type="classification", ordering="chronological"
    )
    assert "class changed at 66.7% of sampled points" in narration["headline"]


def test_narration_flags_unverified_order() -> None:
    series = [[0.1], [0.2], [0.3]]
    out = trace_numeric(series, tolerance=0.05)
    narration = trace_narration(out, value_type="regression", ordering="given-order")
    assert "not verified chronological" in narration["framing"]
