# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Ensemble-disagreement confidence (skill #4, compare_results' ensemble mode).

Pure Python, no heavy deps. :func:`prediction_confidence` is the
ensemble-disagreement *confidence map*: epistemic uncertainty from the spread
across two or more distinct prediction results, exposed as
``olmoearth_compare_results`` with ``mode='ensemble'``. Softmax confidence is
not out-of-distribution detection, and neither is this: a model, or an
ensemble, can be confidently wrong on data unlike anything it was trained on.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from math import log
from statistics import mean
from typing import Any

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
    "ensemble can agree and still be wrong under domain shift)."
)


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
    # population std; typeshed types float ** float as Any (a negative base
    # gives a complex), and this base is a mean of squares, never negative
    sigma: float = (sum((x - mu) ** 2 for x in xs) / m) ** 0.5
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
    the repeated-sampling confidence map.
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
