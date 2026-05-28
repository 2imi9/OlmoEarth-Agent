# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Cloud-mask ensemble audit (skill #11, ``olmoearth-cloud-mask-audit``).

Pure Python, no raster deps. Given several aligned cloud masks for the
same scene — one per algorithm (CFMask, s2cloudless, Sen2Cor, MAJA, or
any others) — :func:`ensemble_disagree` surfaces where the algorithms
*disagree* rather than asserting a single ground-truth mask, and
:func:`verdict_classifier` attributes a model's errors to the mask or to
the model itself.

The point (Skakun et al. CMIX, RSE 274:112990, 2022) is that cloud
algorithms diverge most on thin and semi-transparent cloud, so a single
mask is misleading; the ensemble disagreement is the honest signal. When
a prediction looks wrong, the verdict answers "bad cloud mask, or bad
model?" — if the errors sit where the algorithms disagree (or all call
cloud) it is mask-limited; if they sit on agreed-clear pixels the model
itself is the likely cause.

Producing the per-algorithm masks for a real scene (STAC search +
s2cloudless etc.) is the gated ``fetch_cloud_masks`` step (``SKILLS.md``
#11); this module is the algorithm-agnostic computation over masks the
caller already has, and returns summary statistics only (no per-pixel
geometry), so it satisfies operational rule §3.1.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import combinations
from typing import Any

#: An ensemble needs at least this many masks to have any disagreement.
MIN_MASKS = 2

#: Default share of errors that must land on one side to return a
#: non-``inconclusive`` verdict.
DEFAULT_VERDICT_THRESHOLD = 0.6

_PLACES = 6

_VERDICT_EXPLANATION = {
    "cloud-mask-limited": (
        "Most model errors fall where the cloud algorithms disagree or all "
        "flag cloud — the errors are plausibly mask/cloud artefacts, not the "
        "model."
    ),
    "model-limited": (
        "Most model errors fall where every cloud algorithm agrees the pixel "
        "is clear — no cloud excuse, so the model itself is the likely cause."
    ),
    "inconclusive": (
        "Model errors split between agreed-clear and cloud-affected pixels — "
        "the audit cannot attribute them to mask vs model at this threshold."
    ),
}


class MaskShapeError(ValueError):
    """Raised when masks are too few, unequal length, or not binary."""


def _validate(masks: Mapping[str, Sequence[int]]) -> int:
    """Validate the mask ensemble and return the common pixel count."""
    if len(masks) < MIN_MASKS:
        raise MaskShapeError(
            f"ensemble disagreement needs >= {MIN_MASKS} masks; got {len(masks)}."
        )
    lengths = {len(m) for m in masks.values()}
    if len(lengths) != 1:
        raise MaskShapeError(
            f"all masks must have the same length; got lengths {sorted(lengths)}."
        )
    n = lengths.pop()
    if n == 0:
        raise MaskShapeError("masks must be non-empty.")
    for name, mask in masks.items():
        if any(v not in (0, 1) for v in mask):
            raise MaskShapeError(
                f"mask {name!r} must be binary (0 = clear, 1 = cloud)."
            )
    return n


def ensemble_disagree(masks: Mapping[str, Sequence[int]]) -> dict[str, Any]:
    """Summarize where an ensemble of cloud masks agrees and disagrees.

    Parameters
    ----------
    masks
        Algorithm name -> aligned binary mask (``1`` = cloud, ``0`` =
        clear). All masks must cover the same pixels in the same order.

    Returns
    -------
    dict
        ``n_pixels``, ``n_algorithms``, ``algorithms``; ``agreement_rate``
        and ``disagreement_rate``; ``unanimous_cloud_fraction`` /
        ``unanimous_clear_fraction`` / ``majority_cloud_fraction``;
        ``per_algorithm_cloud_fraction`` (which algorithms are
        aggressive vs conservative); ``mean_pairwise_disagreement`` and
        the full ``pairwise_disagreement`` matrix; and ``vote_histogram``
        (fraction of pixels flagged by exactly k algorithms).

    Raises
    ------
    MaskShapeError
        If there are fewer than :data:`MIN_MASKS` masks, the masks differ
        in length, or any mask is not binary.
    """
    n = _validate(masks)
    names = list(masks)
    n_masks = len(names)

    votes = [sum(masks[name][i] for name in names) for i in range(n)]
    unanimous_cloud = sum(1 for v in votes if v == n_masks)
    unanimous_clear = sum(1 for v in votes if v == 0)
    agreement = unanimous_cloud + unanimous_clear
    majority_cloud = sum(1 for v in votes if v * 2 > n_masks)

    raw_pairs = {
        f"{a}|{b}": sum(1 for i in range(n) if masks[a][i] != masks[b][i]) / n
        for a, b in combinations(names, 2)
    }
    mean_pairwise = sum(raw_pairs.values()) / len(raw_pairs)

    return {
        "n_pixels": n,
        "n_algorithms": n_masks,
        "algorithms": names,
        "agreement_rate": round(agreement / n, _PLACES),
        "disagreement_rate": round((n - agreement) / n, _PLACES),
        "unanimous_cloud_fraction": round(unanimous_cloud / n, _PLACES),
        "unanimous_clear_fraction": round(unanimous_clear / n, _PLACES),
        "majority_cloud_fraction": round(majority_cloud / n, _PLACES),
        "per_algorithm_cloud_fraction": {
            name: round(sum(masks[name]) / n, _PLACES) for name in names
        },
        "mean_pairwise_disagreement": round(mean_pairwise, _PLACES),
        "pairwise_disagreement": {k: round(v, _PLACES) for k, v in raw_pairs.items()},
        "vote_histogram": {
            str(k): round(sum(1 for v in votes if v == k) / n, _PLACES)
            for k in range(n_masks + 1)
        },
    }


def verdict_classifier(
    masks: Mapping[str, Sequence[int]],
    model_error: Sequence[int],
    *,
    threshold: float = DEFAULT_VERDICT_THRESHOLD,
) -> dict[str, Any]:
    """Attribute a model's errors to the cloud mask or to the model.

    Parameters
    ----------
    masks
        The cloud-mask ensemble (see :func:`ensemble_disagree`).
    model_error
        Aligned binary mask, ``1`` where the model's prediction was
        wrong. Same length as the cloud masks.
    threshold
        Share of error pixels that must land on one side to return a
        decisive verdict; otherwise ``inconclusive`` (default
        :data:`DEFAULT_VERDICT_THRESHOLD`).

    Returns
    -------
    dict
        ``n_errors``; the fraction of errors in agreed-clear, contested,
        and agreed-cloud pixels; ``cloud_related_error_fraction``; and a
        ``verdict`` (``cloud-mask-limited`` / ``model-limited`` /
        ``inconclusive``) with a plain-language ``explanation``.

    Raises
    ------
    MaskShapeError
        If the masks are invalid (see :func:`ensemble_disagree`) or
        ``model_error`` is the wrong length or not binary.
    """
    n = _validate(masks)
    if len(model_error) != n:
        raise MaskShapeError(
            f"model_error length {len(model_error)} must match the mask " f"length {n}."
        )
    if any(v not in (0, 1) for v in model_error):
        raise MaskShapeError("model_error must be binary (1 = model wrong).")

    names = list(masks)
    n_masks = len(names)
    votes = [sum(masks[name][i] for name in names) for i in range(n)]
    error_idx = [i for i in range(n) if model_error[i] == 1]
    n_err = len(error_idx)

    if n_err == 0:
        return {
            "n_errors": 0,
            "verdict": "no-errors",
            "explanation": "No model-error pixels supplied; nothing to attribute.",
        }

    in_clear = sum(1 for i in error_idx if votes[i] == 0)
    in_cloud = sum(1 for i in error_idx if votes[i] == n_masks)
    in_contested = n_err - in_clear - in_cloud
    clear_frac = in_clear / n_err
    cloud_related_frac = (in_contested + in_cloud) / n_err

    if cloud_related_frac >= threshold:
        verdict = "cloud-mask-limited"
    elif clear_frac >= threshold:
        verdict = "model-limited"
    else:
        verdict = "inconclusive"

    return {
        "n_errors": n_err,
        "errors_in_agreed_clear_fraction": round(clear_frac, _PLACES),
        "errors_in_contested_fraction": round(in_contested / n_err, _PLACES),
        "errors_in_agreed_cloud_fraction": round(in_cloud / n_err, _PLACES),
        "cloud_related_error_fraction": round(cloud_related_frac, _PLACES),
        "verdict": verdict,
        "explanation": _VERDICT_EXPLANATION[verdict],
    }
