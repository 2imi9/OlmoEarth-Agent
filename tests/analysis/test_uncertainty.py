# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Tests for the Area-of-Applicability OOD flag (skill #10)."""

from __future__ import annotations

import pytest

from olmoearth_agent.analysis.uncertainty import area_of_applicability, ood_flag

# Square + centre: both features vary, so neither standardizes to zero.
TRAIN = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]]


def test_point_on_training_data_is_inside() -> None:
    out = area_of_applicability(TRAIN, [[0.5, 0.5]])
    assert out["new_di"] == [0.0]
    assert out["inside_aoa"] == [True]
    assert out["ood_fraction"] == 0.0
    assert out["verdict"] == "within-AOA"


def test_in_distribution_point_is_inside() -> None:
    out = area_of_applicability(TRAIN, [[0.55, 0.45]])
    assert out["inside_aoa"] == [True]
    assert out["verdict"] == "within-AOA"


def test_far_point_is_out_of_distribution() -> None:
    out = area_of_applicability(TRAIN, [[1000.0, 1000.0]])
    assert out["inside_aoa"] == [False]
    assert out["ood_fraction"] == 1.0
    assert out["verdict"] == "mostly-OOD"
    assert out["new_di"][0] > out["aoa_threshold"]


def test_partially_ood_when_some_inside_some_outside() -> None:
    out = area_of_applicability(TRAIN, [[0.5, 0.5], [0.5, 0.5], [1000.0, 1000.0]])
    assert out["inside_aoa"] == [True, True, False]
    assert out["ood_fraction"] == pytest.approx(1 / 3)
    assert out["verdict"] == "partially-OOD"


def test_structure_and_counts() -> None:
    out = area_of_applicability(TRAIN, [[0.5, 0.5], [2.0, 2.0]])
    assert out["n_train"] == 5
    assert out["n_new"] == 2
    assert out["aoa_threshold"] >= 0.0
    assert len(out["new_di"]) == 2
    assert set(out["train_di_summary"]) == {"min", "median", "q75", "max"}


def test_weights_change_the_result() -> None:
    point = [[0.5, 3.0]]
    base = area_of_applicability(TRAIN, point)
    weighted = area_of_applicability(TRAIN, point, weights=[1.0, 10.0])
    assert base["new_di"] != weighted["new_di"]


def test_constant_feature_is_dropped() -> None:
    # feature 1 is constant -> standardizes away; AOA runs on feature 0
    train = [[0.0, 5.0], [1.0, 5.0], [2.0, 5.0], [0.5, 5.0]]
    assert area_of_applicability(train, [[1.0, 5.0]])["inside_aoa"] == [True]
    assert area_of_applicability(train, [[100.0, 5.0]])["inside_aoa"] == [False]


def test_empty_feature_vectors_rejected() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        area_of_applicability([[], []], [[]])


def test_requires_at_least_two_training_vectors() -> None:
    with pytest.raises(ValueError, match=">= 2"):
        area_of_applicability([[0.0, 0.0]], [[1.0, 1.0]])


def test_mismatched_dimensions_rejected() -> None:
    with pytest.raises(ValueError, match="same length"):
        area_of_applicability([[0.0, 0.0], [1.0, 1.0, 1.0]], [[0.5, 0.5]])


def test_empty_new_rejected() -> None:
    with pytest.raises(ValueError, match="at least one new"):
        area_of_applicability(TRAIN, [])


def test_weights_length_must_match() -> None:
    with pytest.raises(ValueError, match="weights length"):
        area_of_applicability(TRAIN, [[0.5, 0.5]], weights=[1.0])


def test_zero_variance_training_rejected() -> None:
    with pytest.raises(ValueError, match="zero variance"):
        area_of_applicability([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]], [[2.0, 2.0]])


def test_ood_flag_within() -> None:
    out = ood_flag([0.1, 0.2, 0.3], 1.0)
    assert out["flags"] == [False, False, False]
    assert out["ood_fraction"] == 0.0
    assert out["verdict"] == "within-AOA"


def test_ood_flag_mostly() -> None:
    out = ood_flag([2.0, 3.0], 1.0)
    assert out["verdict"] == "mostly-OOD"
    assert out["ood_fraction"] == 1.0


def test_ood_flag_partially() -> None:
    out = ood_flag([2.0, 0.1, 0.1], 1.0)
    assert out["verdict"] == "partially-OOD"
    assert out["ood_fraction"] == pytest.approx(1 / 3)


def test_ood_flag_empty() -> None:
    out = ood_flag([], 1.0)
    assert out["ood_fraction"] == 0.0
    assert out["verdict"] == "within-AOA"
