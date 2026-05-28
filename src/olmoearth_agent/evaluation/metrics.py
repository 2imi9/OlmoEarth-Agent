# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Classification metrics for map accuracy (pure Python).

Per-class precision / recall / F1 / IoU plus overall accuracy, macro-F1,
and mean IoU. Works for binary and multi-class labels.
"""

from __future__ import annotations

from typing import Any


def classification_metrics(
    y_true: list[Any], y_pred: list[Any]
) -> dict[str, Any]:
    """Compute per-class and overall classification metrics.

    Parameters
    ----------
    y_true, y_pred
        Equal-length sequences of class labels (any hashable type).

    Returns
    -------
    dict
        ``accuracy``, ``n``, ``labels``, ``per_class`` (precision /
        recall / f1 / iou / support per class), ``macro_f1``, ``mean_iou``.

    Raises
    ------
    ValueError
        If the inputs differ in length or are empty.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if not y_true:
        raise ValueError("inputs must be non-empty")

    n = len(y_true)
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    accuracy = correct / n

    labels = sorted({str(v) for v in (*y_true, *y_pred)})
    true_s = [str(v) for v in y_true]
    pred_s = [str(v) for v in y_pred]

    per_class: dict[str, dict[str, float | int]] = {}
    for cls in labels:
        tp = sum(1 for t, p in zip(true_s, pred_s) if t == cls and p == cls)
        fp = sum(1 for t, p in zip(true_s, pred_s) if t != cls and p == cls)
        fn = sum(1 for t, p in zip(true_s, pred_s) if t == cls and p != cls)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )
        iou = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
        per_class[cls] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "iou": round(iou, 4),
            "support": tp + fn,
        }

    macro_f1 = sum(c["f1"] for c in per_class.values()) / len(per_class)
    mean_iou = sum(c["iou"] for c in per_class.values()) / len(per_class)
    return {
        "accuracy": round(accuracy, 4),
        "n": n,
        "labels": labels,
        "per_class": per_class,
        "macro_f1": round(macro_f1, 4),
        "mean_iou": round(mean_iou, 4),
    }
