# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""OlmoEarth-vs-AlphaEarth baseline comparison (skill #7, ``olmoearth-baseline-compare``).

Pure Python, no heavy deps. Compares two models side-by-side on the same
labels / AOI: :func:`compare_metrics` builds a per-metric table (reusing
the evaluate skill's :func:`classification_metrics`) plus deltas and a
winner, and :func:`difference_raster` gives the cell-by-cell gap between
two score layers.

The point (Ma et al. arXiv:2601.00857): AlphaEarth-based models are
documented to underperform under cross-region *transfer*, so a fine-tuned
OlmoEarth "substantially outperformed" claim only lands with a real
side-by-side run in a region where AlphaEarth actually fails. Both
functions are source-agnostic: the AlphaEarth side comes from data the
user exported from the public Google Earth Engine "Satellite Embedding"
dataset (no live GEE connection needed; there is no official Earth Engine
MCP and an external-API dependency is deliberately avoided).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from olmoearth_agent.evaluation.metrics import classification_metrics

#: The overall metrics compared head-to-head.
_HEADLINE_METRICS = ("accuracy", "macro_f1", "mean_iou")

_PLACES = 6


def compare_metrics(
    y_true: Sequence[Any],
    pred_a: Sequence[Any],
    pred_b: Sequence[Any],
    *,
    label_a: str = "olmoearth",
    label_b: str = "alphaearth",
    tol: float = 0.0,
) -> dict[str, Any]:
    """Compare two models' predictions against shared ground truth.

    Parameters
    ----------
    y_true
        Ground-truth labels.
    pred_a, pred_b
        Aligned predicted labels from the two models.
    label_a, label_b
        Names for the two models (default ``olmoearth`` / ``alphaearth``).
    tol
        A metric delta within ``tol`` counts as a tie.

    Returns
    -------
    dict
        ``model_a`` / ``model_b`` (label + full metrics), a ``comparison``
        table (accuracy / macro-F1 / mean-IoU with each model's value,
        ``delta`` = a - b, and ``winner``), and ``overall_winner`` (the
        model winning the most headline metrics).

    Raises
    ------
    ValueError
        Propagated from :func:`classification_metrics` if a prediction
        list is the wrong length or the inputs are empty.
    """
    metrics_a = classification_metrics(list(y_true), list(pred_a))
    metrics_b = classification_metrics(list(y_true), list(pred_b))

    comparison = []
    wins_a = wins_b = 0
    for metric in _HEADLINE_METRICS:
        a, b = metrics_a[metric], metrics_b[metric]
        delta = round(a - b, _PLACES)
        if delta > tol:
            winner = label_a
            wins_a += 1
        elif delta < -tol:
            winner = label_b
            wins_b += 1
        else:
            winner = "tie"
        comparison.append(
            {"metric": metric, label_a: a, label_b: b, "delta": delta, "winner": winner}
        )

    if wins_a > wins_b:
        overall = label_a
    elif wins_b > wins_a:
        overall = label_b
    else:
        overall = "tie"

    return {
        "model_a": {"label": label_a, "metrics": metrics_a},
        "model_b": {"label": label_b, "metrics": metrics_b},
        "comparison": comparison,
        "overall_winner": overall,
    }


def difference_raster(
    layer_a: Sequence[float],
    layer_b: Sequence[float],
    *,
    tol: float = 0.0,
    label_a: str = "olmoearth",
    label_b: str = "alphaearth",
) -> dict[str, Any]:
    """Cell-by-cell difference between two aligned score layers.

    Parameters
    ----------
    layer_a, layer_b
        Equal-length per-cell score arrays for the same AOI grid.
    tol
        A per-cell ``|a - b|`` within ``tol`` counts as a tie.
    label_a, label_b
        Names for the two layers.

    Returns
    -------
    dict
        ``n_cells``, ``mean_difference`` (a - b), ``mean_abs_difference``,
        ``max_abs_difference``, the fraction of cells where each layer is
        higher, ``tie_fraction``, and the per-cell ``differences``.

    Raises
    ------
    ValueError
        If the layers differ in length or are empty.
    """
    if len(layer_a) != len(layer_b):
        raise ValueError("layers must have the same length.")
    if not layer_a:
        raise ValueError("layers must be non-empty.")

    diffs = [float(a) - float(b) for a, b in zip(layer_a, layer_b)]
    n = len(diffs)
    a_higher = sum(1 for d in diffs if d > tol)
    b_higher = sum(1 for d in diffs if d < -tol)
    abs_diffs = [abs(d) for d in diffs]

    return {
        "n_cells": n,
        "mean_difference": round(sum(diffs) / n, _PLACES),
        "mean_abs_difference": round(sum(abs_diffs) / n, _PLACES),
        "max_abs_difference": round(max(abs_diffs), _PLACES),
        f"{label_a}_higher_fraction": round(a_higher / n, _PLACES),
        f"{label_b}_higher_fraction": round(b_higher / n, _PLACES),
        "tie_fraction": round((n - a_higher - b_higher) / n, _PLACES),
        "differences": [round(d, _PLACES) for d in diffs],
    }
