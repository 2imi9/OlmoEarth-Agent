# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Area-of-Applicability OOD flag (skill #9, ``olmoearth-uncertainty``).

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
OOD half; :func:`prediction_confidence` is the ensemble-disagreement
*confidence map* -- epistemic uncertainty from the spread across two or
more distinct prediction results, exposed as the
``olmoearth_ensemble_uncertainty`` tool.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from math import log
from statistics import mean, median
from typing import Any

#: Minimum training feature vectors needed to define a feature space.
MIN_TRAIN = 2

_PLACES = 6

#: Below this absolute mean, the coefficient of variation is undefined.
_CV_EPS = 1e-12

#: The honest framing for every confidence result: dispersion is epistemic
#: uncertainty, not a calibrated correctness probability.
CONFIDENCE_CAVEAT = (
    "Ensemble / repeated-sample dispersion is epistemic uncertainty: it "
    "measures how much the supplied predictions disagree across draws or "
    "ensemble members, NOT a calibrated probability that a prediction is "
    "correct. Low dispersion (high confidence) means the models are "
    "self-consistent -- necessary but not sufficient for correctness (an "
    "ensemble can agree and still be wrong under domain shift). Complements, "
    "does not replace, the Area-of-Applicability OOD flag; trust a prediction "
    "most when it is both inside the AOA and low-dispersion."
)


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


def _regression_point(xs: Sequence[float]) -> dict[str, Any]:
    """Dispersion stats for one point's repeated/ensemble regression draws.

    ``confidence`` is a bounded, scale-free transform of the coefficient of
    variation (``1 / (1 + CV)``: CV 0 -> 1.0, CV -> inf -> 0.0), a monotone
    dispersion->confidence map, NOT a probability. When the mean is ~0 the CV
    is undefined: ``confidence`` is 1.0 if the draws are identical (zero
    spread) and ``None`` otherwise (reported, not faked).
    """
    m = len(xs)
    mu = mean(xs)
    sigma = (sum((x - mu) ** 2 for x in xs) / m) ** 0.5  # population std
    if abs(mu) < _CV_EPS:
        cv: float | None = None
        conf: float | None = 1.0 if sigma == 0 else None
    else:
        cv = sigma / abs(mu)
        conf = 1.0 / (1.0 + cv)
    return {
        "n": m,
        "mean": round(mu, _PLACES),
        "std": round(sigma, _PLACES),
        "coefficient_of_variation": None if cv is None else round(cv, _PLACES),
        "range": round(max(xs) - min(xs), _PLACES),
        "confidence": None if conf is None else round(conf, _PLACES),
    }


def _categorical_point(labels: Sequence[Any]) -> dict[str, Any]:
    """Vote stats for one point's repeated/ensemble categorical draws.

    ``entropy`` is the Shannon entropy normalized by ``ln(K_eff)`` over the
    *observed* classes, so a unanimous vote is 0.0 and a uniform split over the
    observed classes is 1.0. ``confidence = 1 - entropy``. Because
    normalization uses observed (not global) classes, a 2-way 50/50 reads as
    entropy 1.0 -- maximal uncertainty given what was seen.
    """
    m = len(labels)
    counts = Counter(labels)
    k_eff = len(counts)
    top = max(counts.values())
    majority = sorted((lab for lab, c in counts.items() if c == top), key=str)[0]
    if k_eff <= 1:
        h_norm = 0.0
    else:
        h = -sum((c / m) * log(c / m) for c in counts.values())
        h_norm = h / log(k_eff)
    ordered = sorted((c / m for c in counts.values()), reverse=True)
    top2 = ordered[1] if len(ordered) > 1 else 0.0
    fracs = {str(lab): round(c / m, _PLACES) for lab, c in counts.items()}
    return {
        "n": m,
        "majority_class": majority,
        "vote_fractions": fracs,
        "entropy": round(h_norm, _PLACES),
        "margin": round(ordered[0] - top2, _PLACES),
        "confidence": round(1.0 - h_norm, _PLACES),
    }


def prediction_confidence(
    samples: Sequence[Sequence[Any]],
    *,
    value_type: str = "regression",
) -> dict[str, Any]:
    """Per-point + aggregate confidence from repeated/ensemble predictions.

    Parameters
    ----------
    samples
        Per-point lists of repeated or ensemble inference outputs: ``samples[i]``
        is the draws at point ``i`` (each >= 1). For ``regression`` the draws are
        numbers; for ``categorical`` they are class labels.
    value_type
        ``"regression"`` (numeric dispersion: mean / std / CV / range) or
        ``"categorical"`` (majority class / vote fractions / normalized Shannon
        entropy / margin).

    Returns
    -------
    dict
        ``value_type``, ``n_points``, ``ensemble_size`` (min/max draws),
        ``per_point`` (parallel to ``samples``), ``summary`` (aggregate), and
        the honest ``caveat`` (see :data:`CONFIDENCE_CAVEAT`).

    Raises
    ------
    ValueError
        If ``samples`` is empty, any point has zero draws, or ``value_type`` is
        unknown.

    Notes
    -----
    The dispersion is **epistemic uncertainty** (model self-consistency), not a
    calibrated correctness probability -- see :data:`CONFIDENCE_CAVEAT`. This is
    the repeated-sampling confidence map that complements
    :func:`area_of_applicability` (OOD).
    """
    if not samples:
        raise ValueError("need at least one point's samples.")
    if value_type not in ("regression", "categorical"):
        raise ValueError("value_type must be 'regression' or 'categorical'.")
    for pt in samples:
        if not pt:
            raise ValueError("every point needs >= 1 sample/draw.")
    sizes = [len(pt) for pt in samples]

    if value_type == "regression":
        per_point = [_regression_point([float(x) for x in pt]) for pt in samples]
        confs = [p["confidence"] for p in per_point if p["confidence"] is not None]
        cvs = [
            p["coefficient_of_variation"]
            for p in per_point
            if p["coefficient_of_variation"] is not None
        ]
        summary: dict[str, Any] = {
            "mean_confidence": round(mean(confs), _PLACES) if confs else None,
            "n_confidence_undefined": len(per_point) - len(confs),
            "mean_cv": round(mean(cvs), _PLACES) if cvs else None,
            "max_cv": round(max(cvs), _PLACES) if cvs else None,
            "n_degenerate": sum(1 for s in sizes if s == 1),
        }
    else:
        per_point = [_categorical_point(pt) for pt in samples]
        confs = [p["confidence"] for p in per_point]
        ents = [p["entropy"] for p in per_point]
        summary = {
            "mean_confidence": round(mean(confs), _PLACES),
            "mean_entropy": round(mean(ents), _PLACES),
            "max_entropy": round(max(ents), _PLACES),
            "mean_margin": round(mean(p["margin"] for p in per_point), _PLACES),
            "n_unanimous": sum(1 for p in per_point if p["entropy"] == 0.0),
        }

    return {
        "value_type": value_type,
        "n_points": len(samples),
        "ensemble_size": {"min": min(sizes), "max": max(sizes)},
        "per_point": per_point,
        "summary": summary,
        "caveat": CONFIDENCE_CAVEAT,
    }
