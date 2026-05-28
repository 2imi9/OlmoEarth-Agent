# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Tests for classification metrics."""

from __future__ import annotations

import pytest

from olmoearth_agent.evaluation.metrics import classification_metrics


def test_perfect_prediction() -> None:
    m = classification_metrics([1, 0, 1, 0], [1, 0, 1, 0])
    assert m["accuracy"] == 1.0
    assert m["macro_f1"] == 1.0
    assert m["mean_iou"] == 1.0


def test_known_binary_confusion() -> None:
    # class 1: tp=1, fp=0, fn=1 ; class 0: tp=2, fp=1, fn=0
    m = classification_metrics([1, 1, 0, 0], [1, 0, 0, 0])
    assert m["accuracy"] == 0.75
    one = m["per_class"]["1"]
    assert one["precision"] == 1.0
    assert one["recall"] == 0.5
    assert one["f1"] == pytest.approx(0.6667, abs=1e-4)
    assert one["iou"] == 0.5
    zero = m["per_class"]["0"]
    assert zero["precision"] == pytest.approx(0.6667, abs=1e-4)
    assert zero["recall"] == 1.0
    assert zero["iou"] == pytest.approx(0.6667, abs=1e-4)


def test_support_counts() -> None:
    m = classification_metrics(["a", "a", "b"], ["a", "b", "b"])
    assert m["per_class"]["a"]["support"] == 2
    assert m["per_class"]["b"]["support"] == 1
    assert m["labels"] == ["a", "b"]


def test_validates_inputs() -> None:
    with pytest.raises(ValueError):
        classification_metrics([1, 2], [1])
    with pytest.raises(ValueError):
        classification_metrics([], [])
