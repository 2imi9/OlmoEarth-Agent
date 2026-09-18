# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Label-free review-set ranking (skill #18, ``olmoearth-review-set``).

Pure Python, no heavy deps. Given the per-class scores a model emitted for
a grid of windows, this module answers the question a practitioner asks
the moment a prediction map lands: *which windows do I check first, and
how much of the error do I catch if I only check 5% of them?*

The signal is the model's own **top-1 minus top-2 margin**. That choice is
not an aesthetic one. The companion research repository
`2imi9/olmoearth_inferenceX <https://github.com/2imi9/olmoearth_inferenceX>`_
measured it against every alternative it could build -- ensemble
disagreement, cross-model disagreement, feature-space typicality /
distance-to-training-data, two-view disagreement, flip-and-rotate
consistency -- on seven expert-labelled testbeds, and none of them ranked
errors better than the margin. Those verdicts are carried here as
:data:`EVIDENCE` so a tool result can cite them instead of asserting them.

Scope, stated plainly because the neighbouring skill #9 owns the other
half: this module ranks *relative* suspicion so a finite review budget is
spent well. It is **not** a calibrated probability of correctness and it is
**not** out-of-distribution detection -- a model can be confidently wrong on
data unlike its training set, which is what
:func:`~olmoearth_agent.analysis.uncertainty.area_of_applicability` is for.

Why this is a bring-your-own-scores tool: the Studio API exposes only
``raw_value`` (regression) and ``classification`` (categorical) per band --
no probabilities, no logits, no per-class scores (see
:meth:`~olmoearth_agent.studio.client.StudioClient.pixel_value`). A margin
cannot be recovered from a hard class label, so the scores must come from
wherever inference actually ran (rslearn ``model predict``, an
embeddings+probe pass, or any exported softmax). This mirrors skill #8,
which likewise takes caller-supplied embeddings.

Coordinates are never accepted or returned: windows are addressed by index
and by caller-supplied id, so a review set cannot leak geometry into chat
(operational rule §3.1).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

#: Default review budgets (fraction of all windows sent to a human).
DEFAULT_BUDGETS: tuple[float, ...] = (0.01, 0.05, 0.10)

#: Default cap on how many review rows are returned inline.
DEFAULT_MAX_LISTED = 50

#: Largest grid this tool will accept (keeps a pure-Python sort honest).
MAX_WINDOWS = 200_000

#: Measured verdicts from 2imi9/olmoearth_inferenceX, carried so a tool
#: result can cite evidence rather than assert authority. Each entry names
#: the claim id in that repository's ledger (``docs/claims.yaml``).
EVIDENCE: dict[str, str] = {
    "suite-margin-wins-every-task": (
        "The headline result. On all 24 scored tasks of Ai2's own published "
        "embedding suite -- tasks Ai2 chose for their own paper, 14 distinct "
        "sources, 6,435,473 graded units, accuracy 0.333 to 0.979 -- the "
        "model's own margin beat the best no-model control on every one. "
        "Excess-AURC leads 0.0157 to 0.2278, median 0.1158; one-sided exact "
        "sign test 24/24, p = 6e-08. One of the two controls is embedding "
        "distance, i.e. a feature-space-typicality score."
    ),
    "shrug-signals-rejected": (
        "Ensemble mutual information (the disagreement component) and -NCDD "
        "lose to the model's own confidence on both testbeds under OlmoEarth "
        "v1.2, preregistered: pooled excess-AURC lead +0.0060 and +0.0022, "
        "per tile 263/61 and 352/76. The embedding and input signals are "
        "3-9x worse on all four arms."
    ),
    "feature-typicality-rejected": (
        "kNN distance, Mahalanobis distance, PCA residual and ViM against "
        "three reference sets: no score beats confidence on either testbed, "
        "and every one loses on at least 317 of 351 Bolivia tiles."
    ),
    "embedding-dissimilarity-rejected": (
        "Embedding dissimilarity to the training features has no scale-free "
        "advantage over confidence (13/27 scenes, sign p=1.00); every "
        "feature-typicality generalisation loses on both testbeds."
    ),
    "cross-model-disagreement-rejected": (
        "Disagreement with a second model of the same family beats "
        "confidence on only 10 of 27 scenes against WorldCover, and on "
        "84/265 Bolivia tiles against hand labels."
    ),
    "two-view-disagreement-rejected": (
        "A head on an outside representation errs where OlmoEarth errs, so "
        "two-view disagreement captures 0.242 of the errors at a 10% budget "
        "against 0.454 for confidence (excess AURC 0.067 vs 0.0215)."
    ),
    "backbone-version-disagreement-rejected": (
        "Disagreement between OlmoEarth v1 and v1.2 is not a useful error " "signal."
    ),
    "anysat-witness-rejected": (
        "An outside representation as the neighbourhood space adds nothing "
        "over the model's own (159/143, p=0.19)."
    ),
    "no-encoder-internal-signal-beats-confidence": (
        "No signal from the encoder's internals, its pretraining objective, "
        "a posterior over the probe head, feature-space typicality, a second "
        "model of the same family, or flip-and-rotate consistency ranks "
        "errors better than confidence on expert labels."
    ),
}

#: Honest limits on the evidence above, surfaced with every ranking.
EVIDENCE_LIMITS: tuple[str, ...] = (
    "Measured on the OlmoEarth family at window level against seven "
    "references (photointerpretation, another model's output, expert "
    "annotation of a served product, in-situ ground observation, farmers' "
    "declarations). Not a general theorem about every model or every task.",
    "Known exception: on one Sen1Floods11 flood event (Bolivia) a no-model "
    "NDWI control ranked errors as well as confidence, and better under "
    "v1.2 -- a cheap physical control is worth running alongside.",
    "This ranks suspicion; it does not calibrate it. Capture at a fixed "
    "budget is bounded by budget/error_rate, so it is not comparable across "
    "populations with different error rates (use AUROC for that).",
    "If a fitted readout produced these scores, report its generalisation "
    "gap: a probe that memorised its practice data reversed which "
    "confidence signal ranked best. The ordering degrades with memorisation "
    "and only severe memorisation reverses it.",
    "The margin does NOT beat ensemble PREDICTIVE ENTROPY -- it ties it. The "
    "rejection above is of the disagreement component (mutual information), "
    "not of entropy, and at tile level on the v1 test split ensemble mutual "
    "information was 0.003 better than confidence. If you already have an "
    "ensemble, its entropy is a fair alternative; its disagreement is not.",
    "These numbers order a review well and are badly miscalibrated as "
    "probabilities: good for deciding what to look at first, poor for "
    "deciding what to trust at a threshold.",
)


# --------------------------------------------------------------------------- ranking primitives


def _as_matrix(scores: Sequence[Sequence[float]]) -> list[list[float]]:
    """Validate and copy an ``n_windows x n_classes`` score matrix."""
    if not scores:
        raise ValueError("scores is empty; pass one score vector per window")
    if len(scores) > MAX_WINDOWS:
        raise ValueError(
            f"{len(scores)} windows exceeds MAX_WINDOWS={MAX_WINDOWS}; "
            "aggregate to coarser windows first"
        )
    out: list[list[float]] = []
    width: int | None = None
    for i, row in enumerate(scores):
        vec = [float(v) for v in row]
        if width is None:
            width = len(vec)
        elif len(vec) != width:
            raise ValueError(
                f"window {i} has {len(vec)} classes, window 0 has {width}; "
                "every score vector must have the same length"
            )
        if width < 2:
            raise ValueError(
                "a top-1-minus-top-2 margin needs >= 2 classes per window; "
                "got a single score, which carries no margin"
            )
        if any(math.isnan(v) or math.isinf(v) for v in vec):
            raise ValueError(f"window {i} has a non-finite score")
        out.append(vec)
    return out


def detect_score_type(matrix: Sequence[Sequence[float]]) -> str:
    """Return ``"probability"`` if rows look like a simplex, else ``"logit"``."""
    for row in matrix:
        if any(v < 0.0 or v > 1.0 for v in row):
            return "logit"
        if abs(sum(row) - 1.0) > 1e-3:
            return "logit"
    return "probability"


def margins(matrix: Sequence[Sequence[float]]) -> list[float]:
    """Top-1 minus top-2 score for every window (higher = more confident)."""
    out: list[float] = []
    for row in matrix:
        top2 = sorted(row, reverse=True)[:2]
        out.append(float(top2[0] - top2[1]))
    return out


def predicted_classes(matrix: Sequence[Sequence[float]]) -> list[int]:
    """Arg-max class index for every window (first max wins on a tie)."""
    return [max(range(len(row)), key=row.__getitem__) for row in matrix]


def boundary_counts(classes: Sequence[int], rows: int, cols: int) -> list[int]:
    """Count the 8-neighbours predicted as a *different* class, 0-8.

    The nine-level boundary indicator. Windows are row-major, so window
    ``r * cols + c`` sits at grid position ``(r, c)``.
    """
    if rows * cols != len(classes):
        raise ValueError(
            f"grid {rows}x{cols} = {rows * cols} cells but {len(classes)} "
            "windows were given"
        )
    out: list[int] = []
    for r in range(rows):
        for c in range(cols):
            mine = classes[r * cols + c]
            n = 0
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < rows and 0 <= cc < cols:
                        if classes[rr * cols + cc] != mine:
                            n += 1
            out.append(n)
    return out


# --------------------------------------------------------------------------- estimators
# Ports of oe_inferencex.metrics (pure numpy there, pure Python here). Ties
# are resolved by expectation under random tie-breaking so no result depends
# on the raster order of the input -- which matters because the boundary
# indicator has only nine levels and therefore ties heavily.


def _group_starts(sorted_scores: Sequence[float]) -> list[int]:
    """Index where each run of equal scores begins."""
    return [0] + [
        i
        for i in range(1, len(sorted_scores))
        if sorted_scores[i] != sorted_scores[i - 1]
    ]


def _sorted_by(
    uncertainty: Sequence[float], errors: Sequence[float], *, descending: bool
) -> tuple[list[float], list[float]]:
    """Stable co-sort of ``(uncertainty, errors)``."""
    idx = sorted(
        range(len(uncertainty)),
        key=lambda i: -uncertainty[i] if descending else uncertainty[i],
    )
    return (
        [float(uncertainty[i]) for i in idx],
        [1.0 if errors[i] else 0.0 for i in idx],
    )


def _prefix(values: Sequence[float]) -> list[float]:
    """Prefix sums with a leading zero."""
    out = [0.0]
    for v in values:
        out.append(out[-1] + v)
    return out


def aurc_expected(uncertainty: Sequence[float], errors: Sequence[float]) -> float:
    """Area under the risk-coverage curve, tie-aware. Lower ranks errors better."""
    n = len(errors)
    if n == 0:
        raise ValueError("aurc_expected needs at least one unit")
    s, e = _sorted_by(uncertainty, errors, descending=False)
    starts = _group_starts(s)
    sizes = [
        (starts[k + 1] if k + 1 < len(starts) else n) - starts[k]
        for k in range(len(starts))
    ]
    pre = _prefix(e)
    e_before = [pre[st] for st in starts]
    e_group = [pre[st + sz] - pre[st] for st, sz in zip(starts, sizes)]
    total, g = 0.0, 0
    for i in range(n):
        if g + 1 < len(starts) and i >= starts[g + 1]:
            g += 1
        pos = i - starts[g] + 1
        total += (e_before[g] + e_group[g] * pos / sizes[g]) / (i + 1)
    return total / n


def oracle_aurc(n: int, k: int) -> float:
    """AURC of the perfect ranker: ``n`` units, ``k`` errors, errors rejected first."""
    if n <= 0:
        raise ValueError("oracle_aurc needs n > 0")
    return sum(max(0, i - (n - k)) / i for i in range(1, n + 1)) / n


def excess_aurc(uncertainty: Sequence[float], errors: Sequence[float]) -> float:
    """E-AURC: AURC minus the oracle's, comparable across differing error rates."""
    k = int(sum(1 for x in errors if x))
    return aurc_expected(uncertainty, errors) - oracle_aurc(len(errors), k)


def capture_at_budget(
    uncertainty: Sequence[float],
    errors: Sequence[float],
    budgets: Sequence[float] = DEFAULT_BUDGETS,
) -> dict[float, float]:
    """Share of all errors inside the most suspect fraction ``b`` of windows.

    Tie-aware: a run of equal scores straddling the cut contributes its
    errors in proportion to the share of the run inside the budget.
    """
    n = len(errors)
    if n == 0:
        raise ValueError("capture_at_budget needs at least one unit")
    s, e = _sorted_by(uncertainty, errors, descending=True)
    starts = _group_starts(s)
    sizes = [
        (starts[k + 1] if k + 1 < len(starts) else n) - starts[k]
        for k in range(len(starts))
    ]
    pre = _prefix(e)
    e_group = [pre[st + sz] - pre[st] for st, sz in zip(starts, sizes)]
    total = max(sum(e), 1.0)
    out: dict[float, float] = {}
    for b in budgets:
        k = max(1, int(round(b * n)))
        captured = 0.0
        for gi, (st, sz) in enumerate(zip(starts, sizes)):
            if st + sz <= k:
                captured += e_group[gi]
            elif st < k:
                captured += e_group[gi] * (k - st) / sz
        out[b] = captured / total
    return out


def auroc(uncertainty: Sequence[float], errors: Sequence[float]) -> float | None:
    """AUROC of a suspicion score against the error indicator, ties counted half.

    ``None`` when every unit is an error or none is. Unlike capture at a
    budget this has no base-rate ceiling, so it is the statistic for
    comparing two populations with different error rates.
    """
    n = len(errors)
    pos = sum(1 for x in errors if x)
    if pos == 0 or pos == n:
        return None
    s, e = _sorted_by(uncertainty, errors, descending=False)
    starts = _group_starts(s)
    sizes = [
        (starts[k + 1] if k + 1 < len(starts) else n) - starts[k]
        for k in range(len(starts))
    ]
    total, g = 0.0, 0
    for i in range(n):
        if g + 1 < len(starts) and i >= starts[g + 1]:
            g += 1
        if e[i]:
            total += starts[g] + 0.5 * sizes[g]
    return (total - pos * pos / 2.0) / (pos * (n - pos))


def attainable_ceiling(budget: float, error_rate: float) -> float:
    """The most error any budget can catch: ``min(1, budget / error_rate)``.

    A capture of 0.45 at a 10% budget sounds mediocre until you notice that
    with a 20% error rate the ceiling is 0.50, so the ranker reached 90% of
    what was reachable. Reporting capture without this is how a good ranker
    gets talked down and a bad one talked up.
    """
    if not 0.0 < budget <= 1.0:
        raise ValueError(f"budget must be in (0, 1], got {budget}")
    if not 0.0 <= error_rate <= 1.0:
        raise ValueError(f"error_rate must be in [0, 1], got {error_rate}")
    if error_rate == 0.0:
        return 0.0
    return min(1.0, budget / error_rate)


def sign_test_p(wins: int, decisive: int) -> float:
    """One-sided exact sign test: P(X >= wins) under Binomial(decisive, 1/2)."""
    if decisive <= 0:
        return 1.0
    tail = sum(math.comb(decisive, i) for i in range(wins, decisive + 1))
    return tail / (2.0**decisive)


# --------------------------------------------------------------------------- the two public entry points


def review_set(
    scores: Sequence[Sequence[float]],
    *,
    budget: float = 0.05,
    ids: Sequence[str] | None = None,
    grid: tuple[int, int] | None = None,
    order: str = "confidence",
    score_type: str | None = None,
    error_rate: float | None = None,
    max_listed: int = DEFAULT_MAX_LISTED,
) -> dict[str, Any]:
    """Rank windows by suspicion and return the ones to review first.

    Parameters
    ----------
    scores
        ``n_windows x n_classes`` model scores (logits or probabilities).
        Needs >= 2 classes per window: a margin is a gap between two.
    budget
        Fraction of windows a reviewer can actually look at, in (0, 1].
    ids
        Optional caller-supplied window labels, parallel to ``scores``.
    grid
        Optional ``(rows, cols)``; when given, windows are treated as a
        row-major grid and the nine-level boundary indicator is computed.
        Required for ``order="boundary_first"``.
    order
        ``"confidence"`` -- ascending margin, most suspect first.
        ``"boundary_first"`` -- windows touching a predicted class boundary
        first, ascending margin within each block.
    score_type
        ``"logit"`` or ``"probability"``; auto-detected when omitted.
    error_rate
        Optional known or estimated error rate, used to report the
        attainable ceiling at this budget.
    max_listed
        Cap on inline review rows; the count is always reported in full.

    Returns
    -------
    dict
        ``review`` (the ordered rows), ``n_review``, the margin summary,
        the ceiling arithmetic when ``error_rate`` is given, plus the
        ``evidence`` and ``caveats`` blocks.

    Notes
    -----
    No coordinates are accepted or emitted (operational rule §3.1).
    """
    if order not in ("confidence", "boundary_first"):
        raise ValueError(
            f"order must be 'confidence' or 'boundary_first', got {order!r}"
        )
    if not 0.0 < budget <= 1.0:
        raise ValueError(f"budget must be in (0, 1], got {budget}")
    matrix = _as_matrix(scores)
    n = len(matrix)
    if ids is not None and len(ids) != n:
        raise ValueError(f"ids length {len(ids)} != {n} windows")
    detected = detect_score_type(matrix)
    used_type = score_type or detected

    marg = margins(matrix)
    classes = predicted_classes(matrix)
    bnd: list[int] | None = None
    if grid is not None:
        bnd = boundary_counts(classes, int(grid[0]), int(grid[1]))
    elif order == "boundary_first":
        raise ValueError(
            "order='boundary_first' needs grid=(rows, cols) to know which "
            "windows are adjacent"
        )

    if order == "boundary_first":
        # `bnd` is non-None here: the grid check above raises otherwise. Bound
        # to a local so the closure needs no assert (stripped under -O).
        touches = [b > 0 for b in (bnd or [])]

        def rank_key(i: int) -> tuple[float, ...]:
            """Boundary windows first, ascending margin within each block."""
            return (0.0 if touches[i] else 1.0, marg[i])

    else:

        def rank_key(i: int) -> tuple[float, ...]:
            """Ascending margin: the least confident window is opened first."""
            return (marg[i],)

    ranked = sorted(range(n), key=rank_key)

    k = max(1, int(round(budget * n)))
    rows: list[dict[str, Any]] = []
    for rank, i in enumerate(ranked[:k][:max_listed], start=1):
        row: dict[str, Any] = {
            "rank": rank,
            "window_index": i,
            "predicted_class": classes[i],
            "margin": round(marg[i], 6),
        }
        if ids is not None:
            row["id"] = ids[i]
        if bnd is not None:
            row["boundary_neighbours"] = bnd[i]
        rows.append(row)

    ordered_margins = sorted(marg)
    out: dict[str, Any] = {
        "n_windows": n,
        "n_classes": len(matrix[0]),
        "budget": budget,
        "n_review": k,
        "realised_budget": round(k / n, 6),
        "order": order,
        "signal": "top-1 minus top-2 margin (lower = more suspect)",
        "score_type": used_type,
        "score_type_detected": detected,
        "review": rows,
        "n_review_listed": len(rows),
        "margin_summary": {
            "min": round(ordered_margins[0], 6),
            "median": round(ordered_margins[n // 2], 6),
            "max": round(ordered_margins[-1], 6),
            "cut_at_budget": round(sorted(marg)[min(k, n) - 1], 6),
        },
        "evidence": EVIDENCE,
        "caveats": list(EVIDENCE_LIMITS),
    }
    if bnd is not None:
        out["n_boundary_windows"] = sum(1 for b in bnd if b > 0)
    if error_rate is not None:
        ceiling = attainable_ceiling(budget, error_rate)
        out["ceiling"] = {
            "error_rate": error_rate,
            "attainable_ceiling": round(ceiling, 6),
            "note": (
                "No ranker can catch more than min(1, budget/error_rate) of "
                "the errors at this budget. Judge a realised capture against "
                "this ceiling, not against 1.0."
            ),
        }
    if used_type == "probability" and len(matrix[0]) > 2:
        out["caveats"] = list(out["caveats"]) + [
            "These look like probabilities over >2 classes. The measured "
            "evidence is for the LOGIT margin; with more than two classes "
            "the softmax margin can order windows differently. Pass logits "
            "when you have them."
        ]
    return out


def grade_rule(
    signal: Sequence[float],
    errors: Sequence[float],
    *,
    baseline: Sequence[float] | None = None,
    control: Sequence[float] | None = None,
    groups: Sequence[str] | None = None,
    budgets: Sequence[float] = DEFAULT_BUDGETS,
    higher_is_suspect: bool = True,
) -> dict[str, Any]:
    """Grade a candidate audit rule against confidence and a no-model control.

    This is the honest-broker half of the skill: it answers "is my clever
    new suspicion signal actually better than the model's own margin?" with
    the same machinery that rejected ensemble disagreement, cross-model
    disagreement and feature-space typicality upstream. A signal that
    sounds principled and loses here should not ship.

    Parameters
    ----------
    signal
        The candidate suspicion score, one per graded unit.
    errors
        0/1 error indicator per unit (1 = the model was wrong here).
    baseline
        The incumbent, normally negated margin. Compared head to head.
    control
        A no-model control (e.g. a band index, or pixel variance). A
        candidate that cannot beat this is measuring the scene, not the
        model.
    groups
        Optional per-unit group id (scene, event, region). When given, the
        comparison is also run per group and summarised with a one-sided
        exact sign test -- the right unit of replication, because units
        inside a scene are not independent.
    budgets
        Review budgets to report capture at.
    higher_is_suspect
        Set ``False`` if a *low* value of ``signal`` means suspect; the
        signal is negated so every arm is compared in one direction.

    Returns
    -------
    dict
        Per-arm E-AURC / AUROC / capture, the head-to-head verdict, and
        the per-group sign test when ``groups`` is given.
    """
    n = len(errors)
    if n == 0:
        raise ValueError("grade_rule needs at least one graded unit")
    if len(signal) != n:
        raise ValueError(f"signal length {len(signal)} != errors length {n}")
    err = [1.0 if x else 0.0 for x in errors]
    n_err = int(sum(err))
    if n_err == 0 or n_err == n:
        return {
            "gradable": False,
            "reason": (
                f"{n_err} of {n} units are errors; a ranking comparison needs "
                "both errors and non-errors present"
            ),
            "n_units": n,
        }

    arms: dict[str, list[float]] = {}
    sign = 1.0 if higher_is_suspect else -1.0
    arms["candidate"] = [sign * float(v) for v in signal]
    if baseline is not None:
        if len(baseline) != n:
            raise ValueError(f"baseline length {len(baseline)} != {n}")
        arms["baseline"] = [float(v) for v in baseline]
    if control is not None:
        if len(control) != n:
            raise ValueError(f"control length {len(control)} != {n}")
        arms["control"] = [float(v) for v in control]

    scored: dict[str, dict[str, Any]] = {}
    for name, values in arms.items():
        cap = capture_at_budget(values, err, budgets)
        scored[name] = {
            "excess_aurc": round(excess_aurc(values, err), 6),
            "auroc": (lambda a: None if a is None else round(a, 6))(auroc(values, err)),
            "capture_at_budget": {str(b): round(cap[b], 6) for b in budgets},
        }

    out: dict[str, Any] = {
        "gradable": True,
        "n_units": n,
        "n_errors": n_err,
        "error_rate": round(n_err / n, 6),
        "arms": scored,
        "ceiling_at_budget": {
            str(b): round(attainable_ceiling(b, n_err / n), 6) for b in budgets
        },
        "note": (
            "E-AURC: lower is better, 0 is the oracle. Capture is bounded by "
            "ceiling_at_budget, so read it as a share of that ceiling. AUROC "
            "has no base-rate ceiling."
        ),
    }

    if "baseline" in scored:
        delta = scored["candidate"]["excess_aurc"] - scored["baseline"]["excess_aurc"]
        out["verdict"] = {
            "candidate_beats_baseline_pooled": bool(delta < 0),
            "excess_aurc_delta": round(delta, 6),
            "reading": (
                "negative delta = the candidate ranks errors better than the "
                "baseline on the pooled units"
            ),
        }
    if "control" in scored:
        d = scored["candidate"]["excess_aurc"] - scored["control"]["excess_aurc"]
        out.setdefault("verdict", {})["candidate_beats_control_pooled"] = bool(d < 0)

    if groups is not None:
        if len(groups) != n:
            raise ValueError(f"groups length {len(groups)} != {n}")
        if baseline is None:
            raise ValueError(
                "a per-group sign test needs a baseline to compare against"
            )
        buckets: dict[str, list[int]] = {}
        for i, g in enumerate(groups):
            buckets.setdefault(str(g), []).append(i)
        wins = losses = skipped = 0
        for members in buckets.values():
            ge = [err[i] for i in members]
            if sum(ge) in (0, len(ge)):
                skipped += 1
                continue
            c = excess_aurc([arms["candidate"][i] for i in members], ge)
            b = excess_aurc([arms["baseline"][i] for i in members], ge)
            if c < b:
                wins += 1
            elif c > b:
                losses += 1
        decisive = wins + losses
        out["per_group"] = {
            "n_groups": len(buckets),
            "n_gradable_groups": len(buckets) - skipped,
            "n_skipped_groups": skipped,
            "candidate_wins": wins,
            "baseline_wins": losses,
            # Not rounded: a p-value rounded to 6dp reports 1/1024 as
            # 0.000977 and anything below 5e-7 as exactly 0.
            "sign_test_p_one_sided": sign_test_p(wins, decisive),
            "note": (
                "The group (scene / event / region) is the unit of "
                "replication; units inside one group are not independent, so "
                "a pooled win on many correlated units is weaker evidence "
                "than a win on many groups."
            ),
        }
    return out


def compare_scores(
    scores_a: Sequence[Sequence[float]],
    scores_b: Sequence[Sequence[float]],
    grid: tuple[int, int] | None = None,
    max_listed: int = DEFAULT_MAX_LISTED,
) -> dict[str, Any]:
    """Compare two inferences of the same windows: how much they differ and where.

    Which side is right is not decided here, because it cannot be without labels:
    upstream measured that the more confident side is right on 51 to 70 percent
    of differing windows and that confidence does not order the set (exp58), so
    the honest answer to "which one do I believe" is to decline and say why.
    """
    a = _as_matrix(scores_a)
    b = _as_matrix(scores_b)
    if len(a) != len(b):
        msg = f"the two inferences cover different window counts: {len(a)} and {len(b)}"
        raise ValueError(msg)
    ca, cb = predicted_classes(a), predicted_classes(b)
    ma, mb = margins(a), margins(b)
    diff = [i for i in range(len(a)) if ca[i] != cb[i]]
    out: dict[str, Any] = {
        "n_windows": len(a),
        "n_differing": len(diff),
        "share_differing": round(len(diff) / len(a), 6) if a else 0.0,
        "mean_margin_a_on_differing": (
            round(sum(ma[i] for i in diff) / len(diff), 6) if diff else None
        ),
        "mean_margin_b_on_differing": (
            round(sum(mb[i] for i in diff) / len(diff), 6) if diff else None
        ),
        "which_side_is_right": "not resolvable without labels",
        "evidence": EVIDENCE,
        "caveats": list(EVIDENCE_LIMITS)
        + [
            "The more confident side is right on 51 to 70 percent of differing windows "
            "(exp58, upstream) and confidence does not order the set, so this comparison "
            "does not identify a winner; declining to pick a side is the correct answer."
        ],
    }
    if grid is not None:
        rows_, cols_ = grid
        if rows_ * cols_ != len(a):
            msg = f"grid {grid} does not match {len(a)} windows"
            raise ValueError(msg)
        bcount = boundary_counts(ca, rows_, cols_)
        on_boundary = [bcount[i] > 0 for i in range(len(a))]
        out["boundary_share_overall"] = round(sum(on_boundary) / len(a), 6)
        out["boundary_share_of_differing"] = (
            round(sum(on_boundary[i] for i in diff) / len(diff), 6) if diff else None
        )
        out["where"] = (
            "mostly on class boundaries"
            if diff
            and out["boundary_share_of_differing"] > out["boundary_share_overall"]
            else "spread across the scene"
        )
    listed = []
    for i in diff[:max_listed]:
        row: dict[str, Any] = {
            "window_index": i,
            "class_a": ca[i],
            "class_b": cb[i],
            "margin_a": round(ma[i], 6),
            "margin_b": round(mb[i], 6),
        }
        if grid is not None:
            row["row"], row["col"] = divmod(i, grid[1])
        listed.append(row)
    out["differing"] = listed
    out["n_differing_listed"] = len(listed)
    return out
