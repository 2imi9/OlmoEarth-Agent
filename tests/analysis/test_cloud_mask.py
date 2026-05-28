# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Tests for the cloud-mask ensemble audit (skill #11)."""

from __future__ import annotations

import pytest

from olmoearth_agent.analysis.cloud_mask import (
    MaskShapeError,
    ensemble_disagree,
    verdict_classifier,
)


def test_unanimous_clear() -> None:
    out = ensemble_disagree({"cfmask": [0, 0, 0, 0], "s2cloudless": [0, 0, 0, 0]})
    assert out["agreement_rate"] == 1.0
    assert out["disagreement_rate"] == 0.0
    assert out["unanimous_clear_fraction"] == 1.0
    assert out["per_algorithm_cloud_fraction"] == {"cfmask": 0.0, "s2cloudless": 0.0}


def test_unanimous_cloud() -> None:
    out = ensemble_disagree({"a": [1, 1, 1], "b": [1, 1, 1], "c": [1, 1, 1]})
    assert out["unanimous_cloud_fraction"] == 1.0
    assert out["agreement_rate"] == 1.0
    assert out["majority_cloud_fraction"] == 1.0


def test_partial_disagreement_metrics() -> None:
    # votes per pixel: 2, 1, 1, 0
    out = ensemble_disagree({"a": [1, 0, 1, 0], "b": [1, 1, 0, 0]})
    assert out["n_pixels"] == 4
    assert out["n_algorithms"] == 2
    assert out["agreement_rate"] == 0.5  # pixels 0 (both cloud) and 3 (both clear)
    assert out["disagreement_rate"] == 0.5
    assert out["per_algorithm_cloud_fraction"] == {"a": 0.5, "b": 0.5}
    assert out["pairwise_disagreement"] == {"a|b": 0.5}
    assert out["mean_pairwise_disagreement"] == 0.5
    assert out["vote_histogram"] == {"0": 0.25, "1": 0.5, "2": 0.25}
    assert out["majority_cloud_fraction"] == 0.25  # only the both-cloud pixel


def test_per_algorithm_fraction_flags_aggressive_algorithm() -> None:
    out = ensemble_disagree({"aggressive": [1, 1, 1, 1], "conservative": [0, 0, 0, 0]})
    assert out["per_algorithm_cloud_fraction"]["aggressive"] == 1.0
    assert out["per_algorithm_cloud_fraction"]["conservative"] == 0.0
    assert out["disagreement_rate"] == 1.0


def test_three_algorithm_vote_histogram() -> None:
    # votes per pixel: 3, 2, 1, 0
    out = ensemble_disagree({"a": [1, 1, 1, 0], "b": [1, 1, 0, 0], "c": [1, 0, 0, 0]})
    assert out["vote_histogram"] == {"0": 0.25, "1": 0.25, "2": 0.25, "3": 0.25}
    assert out["majority_cloud_fraction"] == 0.5  # votes 3 and 2 are a majority of 3


def test_requires_at_least_two_masks() -> None:
    with pytest.raises(MaskShapeError, match=">= 2 masks"):
        ensemble_disagree({"only": [0, 1, 0]})


def test_unequal_length_masks_rejected() -> None:
    with pytest.raises(MaskShapeError, match="same length"):
        ensemble_disagree({"a": [0, 1], "b": [0, 1, 0]})


def test_non_binary_mask_rejected() -> None:
    with pytest.raises(MaskShapeError, match="binary"):
        ensemble_disagree({"a": [0, 2], "b": [0, 1]})


def test_empty_masks_rejected() -> None:
    with pytest.raises(MaskShapeError, match="non-empty"):
        ensemble_disagree({"a": [], "b": []})


def test_verdict_model_limited() -> None:
    # errors land on pixels every algorithm calls clear -> no cloud excuse
    out = verdict_classifier({"a": [0, 0, 0, 0], "b": [0, 0, 0, 0]}, [1, 1, 0, 0])
    assert out["verdict"] == "model-limited"
    assert out["errors_in_agreed_clear_fraction"] == 1.0
    assert out["n_errors"] == 2


def test_verdict_cloud_mask_limited() -> None:
    # errors land where all algorithms agree it is cloud
    out = verdict_classifier({"a": [1, 1, 0, 0], "b": [1, 1, 0, 0]}, [1, 1, 0, 0])
    assert out["verdict"] == "cloud-mask-limited"
    assert out["cloud_related_error_fraction"] == 1.0


def test_verdict_inconclusive_when_split() -> None:
    # one error on an agreed-clear pixel, one on an agreed-cloud pixel
    out = verdict_classifier({"a": [0, 0, 1, 1], "b": [0, 0, 1, 1]}, [1, 0, 1, 0])
    assert out["verdict"] == "inconclusive"
    assert out["errors_in_agreed_clear_fraction"] == 0.5
    assert out["errors_in_agreed_cloud_fraction"] == 0.5


def test_verdict_threshold_is_tunable() -> None:
    # the 50/50 split tips to cloud-mask-limited at a 0.5 threshold
    out = verdict_classifier(
        {"a": [0, 0, 1, 1], "b": [0, 0, 1, 1]}, [1, 0, 1, 0], threshold=0.5
    )
    assert out["verdict"] == "cloud-mask-limited"


def test_verdict_no_errors() -> None:
    out = verdict_classifier({"a": [0, 1], "b": [1, 0]}, [0, 0])
    assert out["verdict"] == "no-errors"
    assert out["n_errors"] == 0


def test_verdict_rejects_mismatched_error_length() -> None:
    with pytest.raises(MaskShapeError, match="match the mask"):
        verdict_classifier({"a": [0, 1], "b": [1, 0]}, [1, 0, 1])


def test_verdict_rejects_non_binary_error() -> None:
    with pytest.raises(MaskShapeError, match="model_error must be binary"):
        verdict_classifier({"a": [0, 1], "b": [1, 0]}, [1, 9])
