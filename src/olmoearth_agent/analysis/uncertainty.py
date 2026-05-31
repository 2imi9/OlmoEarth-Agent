# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Area-of-Applicability OOD flag (skill #10, ``olmoearth-uncertainty``).

Pure Python, no heavy deps. Implements the Meyer & Pebesma (2021, MEE
12:1620) Area of Applicability: a model's predictions are trustworthy
only where new data resembles the training data in (scaled,
importance-weighted) feature space.

The dissimilarity index (DI) of a point is its nearest-training-point
distance divided by the mean pairwise distance among training points; a
point is outside the area of applicability (OOD) when its DI exceeds a
threshold derived from the training data's own leave-one-out DI
distribution (the outlier-adjusted ``Q75 + 1.5·IQR``, as in the R
``CAST`` package).

The reason this skill exists: softmax confidence is **not** OOD
detection: a model can be confidently wrong on data unlike anything it
was trained on (AlphaEarth's documented transfer failure under domain
shift is exactly what AOA flags). :func:`area_of_applicability` is the
OOD half; the repeated-sampling *confidence map* (epistemic uncertainty
from repeated stochastic inference) is the documented follow-up.
"""

from __future__ import annotations

from collections.abc import Sequence
from statistics import mean, median
from typing import Any

#: Minimum training feature vectors needed to define a feature space.
MIN_TRAIN = 2

_PLACES = 6


def _validate(
    train: Sequence[Sequence[float]],
    new: Sequence[Sequence[float]],
    weights: Sequence[float] | None,
) -> int:
    """Validate inputs and return the common feature dimensionality."""
    if len(train) < MIN_TRAIN:
        raise ValueError(
            f"need >= {MIN_TRAIN} training feature vectors to compute AOA; "
            f"got {len(train)}."
        )
    if not new:
        raise ValueError("need at least one new feature vector to assess.")
    dim = len(train[0])
    if dim == 0:
        raise ValueError("feature vectors must be non-empty.")
    for vec in (*train, *new):
        if len(vec) != dim:
            raise ValueError("all feature vectors must have the same length.")
    if weights is not None and len(weights) != dim:
        raise ValueError("weights length must match the feature dimension.")
    return dim


def _scaler(
    train: Sequence[Sequence[float]], dim: int
) -> tuple[list[float], list[float]]:
    """Per-feature mean and population std over the training set."""
    means = [mean(vec[j] for vec in train) for j in range(dim)]
    stds = []
    for j in range(dim):
        m = means[j]
        var = sum((vec[j] - m) ** 2 for vec in train) / len(train)
        stds.append(var**0.5)
    return means, stds


def _scale(
    vec: Sequence[float],
    means: Sequence[float],
    stds: Sequence[float],
    weights: Sequence[float] | None,
    dim: int,
) -> list[float]:
    """Standardize one vector; constant features (std 0) drop out."""
    out = []
    for j in range(dim):
        if stds[j] == 0:
            out.append(0.0)
            continue
        value = (vec[j] - means[j]) / stds[j]
        if weights is not None:
            value *= weights[j]
        out.append(value)
    return out


def _distance(a: Sequence[float], b: Sequence[float]) -> float:
    """Euclidean distance."""
    return float(sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5)


def _quantile(ordered: Sequence[float], q: float) -> float:
    """Linear-interpolation quantile of an already-sorted sequence."""
    pos = q * (len(ordered) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    return float(ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo))


def ood_flag(di_values: Sequence[float], threshold: float) -> dict[str, Any]:
    """Flag which dissimilarity-index values fall outside the AOA.

    Parameters
    ----------
    di_values
        Dissimilarity indices, e.g. from :func:`area_of_applicability`.
    threshold
        The AOA threshold; values strictly above it are out-of-distribution.

    Returns
    -------
    dict
        ``flags`` (per-value OOD booleans), ``ood_fraction``, and a
        ``verdict`` (``within-AOA`` / ``partially-OOD`` / ``mostly-OOD``).
    """
    flags = [di > threshold for di in di_values]
    fraction = sum(flags) / len(flags) if flags else 0.0
    if fraction == 0:
        verdict = "within-AOA"
    elif fraction < 0.5:
        verdict = "partially-OOD"
    else:
        verdict = "mostly-OOD"
    return {
        "flags": flags,
        "ood_fraction": round(fraction, _PLACES),
        "verdict": verdict,
    }


def area_of_applicability(
    train_features: Sequence[Sequence[float]],
    new_features: Sequence[Sequence[float]],
    *,
    weights: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Flag which new points fall outside a model's Area of Applicability.

    Parameters
    ----------
    train_features
        Feature vectors for the training data (predictors or embeddings).
    new_features
        Feature vectors for the points to assess (the AOI / prediction
        locations).
    weights
        Optional per-feature importance weights (length = feature
        dimension); features are scaled then multiplied by these.

    Returns
    -------
    dict
        ``n_train``, ``n_new``, ``d_bar`` (mean pairwise training
        distance), ``aoa_threshold``, ``train_di_summary``
        (min/median/q75/max), ``new_di`` (per-point dissimilarity index),
        ``inside_aoa`` (per-point booleans), ``ood_fraction``, and a
        ``verdict``.

    Raises
    ------
    ValueError
        If there are fewer than :data:`MIN_TRAIN` training vectors, the
        vectors differ in length, ``weights`` is the wrong length, or the
        training features have no variance (cannot define a distance).
    """
    dim = _validate(train_features, new_features, weights)
    means, stds = _scaler(train_features, dim)
    if all(s == 0 for s in stds):
        raise ValueError("training features have zero variance; cannot compute AOA.")

    train = [_scale(v, means, stds, weights, dim) for v in train_features]
    new = [_scale(v, means, stds, weights, dim) for v in new_features]
    n = len(train)

    pairwise = [
        _distance(train[i], train[k]) for i in range(n) for k in range(i + 1, n)
    ]
    # d_bar > 0 here: all-identical training is already caught by the
    # zero-variance guard above.
    d_bar = mean(pairwise)

    di_train = [
        min(_distance(train[i], train[k]) for k in range(n) if k != i) / d_bar
        for i in range(n)
    ]
    ordered = sorted(di_train)
    q75 = _quantile(ordered, 0.75)
    iqr = q75 - _quantile(ordered, 0.25)
    threshold = q75 + 1.5 * iqr

    di_new = [min(_distance(v, t) for t in train) / d_bar for v in new]
    flagged = ood_flag(di_new, threshold)

    return {
        "n_train": n,
        "n_new": len(new),
        "d_bar": round(d_bar, _PLACES),
        "aoa_threshold": round(threshold, _PLACES),
        "train_di_summary": {
            "min": round(ordered[0], _PLACES),
            "median": round(median(di_train), _PLACES),
            "q75": round(q75, _PLACES),
            "max": round(ordered[-1], _PLACES),
        },
        "new_di": [round(d, _PLACES) for d in di_new],
        "inside_aoa": [not f for f in flagged["flags"]],
        "ood_fraction": flagged["ood_fraction"],
        "verdict": flagged["verdict"],
    }
