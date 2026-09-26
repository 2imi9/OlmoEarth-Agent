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
:data:`EVIDENCE`. A result states them as ONE scoped sentence,
``evidence_scope``: what was measured, on what, and whether that covers the
case at hand (:func:`ranking_evidence_scope`, :func:`comparison_evidence_scope`);
the full text stays in :func:`evidence_detail`, which the tools save to a file
and name by path. exp86 rounds 6 and 7 found answers widening the inline
blocks to cases no experiment grades (a regression band read as a
probability; "every no-model control" for a suite with two).

A result also carries the shared output contract the harness checks an
answer against: ``facts`` (one-sentence statements computed by code, each
with an ``id`` and its fields), ``must_state`` (what an answer reporting the
result must convey) and ``forbidden_claims`` (claims it must not make).

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
which likewise takes caller-supplied embeddings. The one Studio output that
does carry a margin is a regression band that is a binary score in
``[0, 1]``: :func:`regression_scores` reads it as ``[1 - s, s]``, with the
assumption stated in :data:`BINARY_SCORE_ASSUMPTION`.

Coordinates are never accepted or returned: windows are addressed by index
and by caller-supplied id, so a review set cannot leak geometry into chat
(operational rule §3.1).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from olmoearth_agent.analysis.output_contract import add_must_state

#: Default review budgets (fraction of all windows sent to a human).
DEFAULT_BUDGETS: tuple[float, ...] = (0.01, 0.05, 0.10)

#: Default cap on how many review rows are returned inline.
DEFAULT_MAX_LISTED = 50
#: Class pairs (A's class, B's class) listed by count in a comparison's class_changes.
CLASS_CHANGES_LISTED = 10
#: Bands per axis in a comparison's spatial breakdown (rows, and columns).
SPATIAL_BANDS = 4

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

#: The measured reason a comparison declines to pick a side (upstream exp58).
COMPARISON_EVIDENCE = (
    "The more confident side is right on 51 to 70 percent of differing windows "
    "(exp58, upstream) and confidence does not order the set, so this comparison "
    "does not identify a winner; declining to pick a side is the correct answer."
)

# --------------------------------------------------------------------------- evidence, scoped to the case

#: What the ranking evidence measured and on what: the opening of every
#: ranking's ``evidence_scope`` (claim ``suite-margin-wins-every-task``; the
#: suite's margin is of the probe's probabilities, as exp70 computes it).
SUITE_MEASURED = (
    "Ai2's 24-task embedding suite (exp70: window level, OlmoEarth family) found "
    "the top-1 minus top-2 probability margin ranked errors better than the best "
    "no-model control on all 24 tasks"
)

#: What the comparison evidence measured and on what (claim
#: ``tool-vs-diff-resolution``).
COMPARISON_MEASURED = (
    "On 15 pairs of Sen1Floods11 flood maps with expert labels (exp58: window "
    "level; crop offsets, backbones, sensors, fine-tuning and encoders) the more "
    "confident side was right on 51 to 70 percent of differing windows"
)

#: ``evidence_covers_this_case`` values; every value but ``"yes"`` puts a
#: short sentence stating the limit in ``must_state``.
COVERS = ("yes", "in part", "no", "not known")

#: The values for which the evidence does not, or may not, cover the case: a
#: field the answer reads then carries none of the experiment's numbers and
#: not its dataset's name, and ``evidence_scope`` only names the source and
#: says it does not cover the case. exp86 round 8: "the more confident side was
#: only right 51-70% of the time in comparable cases" (brief 3 on the cluster)
#: and "the two Sen1Floods11 flood maps" (brief 7 on files), for a pair the
#: comparison said its evidence does not cover. ``"in part"`` (an OlmoEarth
#: model's multi-class logits) keeps only exp76's figure about the logit margin
#: itself, the case's own evidence; the suite's 24-task finding, measured on
#: the probability margin, stays in the file (:data:`PARTLY_COVERED`).
NOT_COVERED = ("no", "not known")

#: The value for which the evidence covers the case's kind but not its form:
#: the suite measured the probability margin, and these rows are logits.
#: ``evidence_outside_its_scope`` is emitted, naming what the suite did not
#: measure (the exp76 review of fix-r8: the 24-task figure sat in
#: ``evidence_scope`` beside a ranking by the logit margin).
PARTLY_COVERED = "in part"

#: Where the ranking evidence was measured, named without its numbers or its
#: dataset (the suite and its figures stay in :func:`evidence_detail`).
RANKING_SOURCE = (
    "The ranking evidence (exp70 of 2imi9/olmoearth_inferenceX) was measured on "
    "OlmoEarth models' class-score margins"
)

#: Where the side-picking evidence was measured, named the same way.
COMPARISON_SOURCE = (
    "The side-picking evidence (exp58 of 2imi9/olmoearth_inferenceX) was measured "
    "on other pairs of maps with labels"
)

#: Forbidden with every result whose evidence does not, or may not, cover the
#: case (:data:`NOT_COVERED`); the contract's fixed id.
EVIDENCE_OUTSIDE_ITS_SCOPE = "evidence_outside_its_scope"

# The must_state sentence for each case the evidence does not cover: at most
# 25 words, a limit the answer must convey, never an instruction (the long
# scope sentence stays in ``evidence_scope``, the full text in the file).

#: A Studio regression band in [0, 1] read as [1 - s, s].
MUST_STATE_BINARY_SCORE = (
    "No recorded experiment grades a regression score read as a probability."
)
#: A regression band ranked by its distance from a threshold.
MUST_STATE_THRESHOLD_DISTANCE = (
    "No recorded experiment grades a regression band's distance from a decision "
    "threshold."
)
#: Scores that do not name the model they came from.
MUST_STATE_UNNAMED_MODEL = (
    "The ranking evidence is for OlmoEarth models; these scores do not name their "
    "model, so it may not apply."
)
#: Multi-class logits: the form this ranking uses is not the best measured one
#: (claims top1-beats-the-margin-on-multiclass and
#: no-whole-vector-score-beats-the-probability-forms: one minus the top
#: probability beat the probability margin on 14 of 16, and the probability
#: margin beat the logit margin on all 16, so it beat the logit margin on at
#: least those 14). Also the reading of the scores provider's multi-class
#: warning, whose own text ends in an argument no agent tool takes.
MUST_STATE_MULTICLASS_LOGIT = (
    "This ranking uses the logit margin; on Ai2's suite one minus the top "
    "probability ranked errors better on 14 of 16 multi-class tasks."
)
#: A comparison: no side can be picked.
MUST_STATE_NO_WINNER = (
    "No recorded experiment grades which of these two maps is right; without "
    "labels neither side can be picked."
)


def must_state_other_model(model: str) -> str:
    """The must_state sentence for scores from a model outside the OlmoEarth family."""
    name = model if len(model.split()) == 1 and len(model) <= 80 else "another model"
    return (
        f"The ranking evidence is for OlmoEarth models; these scores come from "
        f"{name}, so it may not apply."
    )


def is_olmoearth(model: str | None) -> bool:
    """Whether a model name (a repo such as ``allenai/OlmoEarth-v1-FT-AWF-Base``) is OlmoEarth's."""
    return bool(model) and "olmoearth" in str(model).lower().replace("-", "").replace(
        "_", ""
    )


def ranking_evidence_scope(
    *, score_kind: str, n_classes: int, model: str | None = None
) -> dict[str, str | None]:
    """ONE sentence: what the ranking evidence measured, on what, and whether it covers this case.

    Parameters
    ----------
    score_kind
        What the rows are: ``"logit"`` or ``"probability"`` (class scores),
        ``"window_confidence_logit"`` or ``"window_confidence_probability"``
        (a scores provider's pooled confidence), ``"binary_score"`` (a
        regression band in ``[0, 1]`` read as ``[1 - s, s]``) or
        ``"threshold_distance"`` (a regression band's distance from a
        threshold).
    n_classes
        Classes per row.
    model
        The model the scores came from, when the input names one.

    Returns
    -------
    dict
        ``sentence`` (the scope, however long it needs to be), ``source``
        (the experiment, ``"exp70"``), ``covers`` (one of :data:`COVERS`) and
        ``must_state``: the limit in at most 25 words when the evidence does
        not cover the case, else ``None``.
    """
    # Where the evidence does not, or may not, cover the case the sentence names
    # its source only: none of the suite's figures, and not the suite (exp86
    # round 8 carried a scope's numbers and dataset to cases it excluded).
    if score_kind == "binary_score":
        return {
            "sentence": f"{RANKING_SOURCE} and does not cover this case: no "
            "recorded experiment grades a regression score read as a probability.",
            "source": "exp70",
            "covers": "no",
            "must_state": MUST_STATE_BINARY_SCORE,
        }
    if score_kind == "threshold_distance":
        return {
            "sentence": f"{RANKING_SOURCE} and does not cover this case: no "
            "recorded experiment grades a regression band's distance from a "
            "decision threshold.",
            "source": "exp70",
            "covers": "no",
            "must_state": MUST_STATE_THRESHOLD_DISTANCE,
        }
    if not is_olmoearth(model):
        named = (
            f"these scores come from {model}, not an OlmoEarth model"
            if model
            else "these scores do not name the model they came from"
        )
        return {
            "sentence": f"{RANKING_SOURCE} and may not cover this case: {named}.",
            "source": "exp70",
            "covers": "not known",
            "must_state": (
                must_state_other_model(str(model))
                if model
                else MUST_STATE_UNNAMED_MODEL
            ),
        }
    if n_classes <= 2:
        return {
            "sentence": f"{SUITE_MEASURED}, which covers this case in kind (an "
            "OlmoEarth model's two-class margin) but not this map itself.",
            "source": "exp70",
            "covers": "yes",
            "must_state": None,
        }
    if score_kind == "window_confidence_probability":
        return {
            "sentence": f"{SUITE_MEASURED}, which covers this case in kind (an "
            f"OlmoEarth model's top-1 probability over {n_classes} classes, which "
            "ranked errors slightly better than that margin on 14 of the suite's 16 "
            "multi-class tasks, exp76) but not this map itself.",
            "source": "exp70",
            "covers": "yes",
            "must_state": None,
        }
    if score_kind == "probability":
        return {
            "sentence": f"{SUITE_MEASURED}, which covers this case in kind (an "
            f"OlmoEarth model's probability margin over {n_classes} classes) but not "
            "this map itself; one minus the top probability ranked errors slightly "
            "better on 14 of the suite's 16 multi-class tasks (exp76).",
            "source": "exp70",
            "covers": "yes",
            "must_state": None,
        }
    # Logits: the suite's finding is about the probability margin, so none of
    # its figures is stated; exp76's figure about the logit margin itself is.
    return {
        "sentence": "The ranking evidence (exp70 of 2imi9/olmoearth_inferenceX) "
        "measured OlmoEarth models' probability margin, not the logit margin this "
        f"ranking uses over {n_classes} classes, so it covers this case only in "
        "part; on 14 of 16 multi-class suite tasks the logit margin ranked errors "
        "worse than one minus the top probability (exp76).",
        "source": "exp70",
        "covers": PARTLY_COVERED,
        "must_state": MUST_STATE_MULTICLASS_LOGIT,
    }


def comparison_evidence_scope() -> dict[str, Any]:
    """ONE sentence: where the side-picking evidence was measured, and that it does not cover this pair.

    The source only: exp58's figures and its dataset stay in the file
    (:func:`evidence_detail`), since exp86 round 8 applied them to pairs this
    sentence excluded ("the more confident side was only right 51-70% of the
    time in comparable cases"; "the two Sen1Floods11 flood maps" for a pair of
    GEOID-Flood maps).
    """
    return {
        "sentence": f"{COMPARISON_SOURCE} and does not cover this pair: no recorded "
        "experiment grades which of these two maps is right, and without labels "
        "neither side can be picked.",
        "source": "exp58",
        "covers": "no",
        "must_state": MUST_STATE_NO_WINNER,
    }


def evidence_outside_scope(scope: dict[str, Any]) -> list[dict[str, str]]:
    """The ``evidence_outside_its_scope`` claim for a scope that does not cover the case.

    Empty when the evidence covers the case. Covered only in part (multi-class
    logits), the reason names what the suite did not measure, the logit
    margin, and the one figure that is about it (exp76). Otherwise it names
    the source, never its numbers or dataset.
    """
    if scope.get("covers") == PARTLY_COVERED:
        return [
            {
                "id": EVIDENCE_OUTSIDE_ITS_SCOPE,
                "why": "the ranking evidence (exp70) measured the probability "
                "margin, and this ranking uses the logit margin, which the suite "
                "did not measure as a ranker against no-model controls: its "
                "finding that the margin ranked errors better than every control "
                "does not carry over to this ranking, neither 'in comparable "
                "cases' nor as 'the evidence behind this ranking'; the only "
                "figure about this form is exp76's (the logit margin ranked "
                "errors worse than one minus the top probability on 14 of 16 "
                "multi-class suite tasks)",
            }
        ]
    if scope.get("covers") not in NOT_COVERED:
        return []
    source = str(scope.get("source") or "upstream")
    comparison = source == "exp58"
    finding = (
        "its finding about how often the more confident side was right"
        if comparison
        else "its finding that the margin ranks errors well"
    )
    does = "does not" if scope["covers"] == "no" else "may not"
    case = "pair" if comparison else "case"
    named = ", and the maps here are not named after its dataset" if comparison else ""
    return [
        {
            "id": EVIDENCE_OUTSIDE_ITS_SCOPE,
            "why": f"evidence_scope names upstream evidence ({source}) that the tool "
            f"says {does} cover this {case}: none of its numbers, its dataset or "
            f"{finding} applies here, neither 'in comparable cases' nor as "
            f"'upstream evidence shows'{named}; the file at evidence_detail_path "
            f"keeps that record, not a finding about this {case}",
        }
    ]


def evidence_detail() -> dict[str, Any]:
    """The full evidence text a result's ``evidence_scope`` stands for, for a file."""
    return {
        "format": "olmoearth-agent/review-evidence@1",
        "source": "2imi9/olmoearth_inferenceX, docs/claims.yaml (claim ids below)",
        "ranking": {
            "measured": SUITE_MEASURED + ".",
            "claims": dict(EVIDENCE),
            "limits": list(EVIDENCE_LIMITS),
            "multi_class": (
                "On the suite's 16 multi-class tasks one minus the top probability "
                "ranked errors better than the top-1 minus top-2 probability margin "
                "on 14, by a small amount (claim top1-beats-the-margin-on-multiclass), "
                "and the logit margin ranked errors worse than the probability "
                "margin on all 16 (claim "
                "no-whole-vector-score-beats-the-probability-forms); exp76."
            ),
        },
        "comparison": {
            "measured": COMPARISON_MEASURED + ".",
            "reading": COMPARISON_EVIDENCE,
        },
    }


def _scope_fields(scope: dict[str, Any]) -> dict[str, Any]:
    """``evidence_scope``, ``evidence_covers_this_case`` and the must_state it implies.

    ``must_state`` gets the scope's short sentence, not the scope itself: the
    scope sentence runs to 60 words and more, and the contract holds a
    must_state sentence to 25 (exp87 review).
    """
    out: dict[str, Any] = {
        "evidence_scope": scope["sentence"],
        "evidence_covers_this_case": scope["covers"],
        "must_state": [],
    }
    if scope.get("must_state"):
        add_must_state(out, [scope["must_state"]])
    return out


#: Stated with every ranking: the margin orders a review; it is not an error rate.
RANKING_FORBIDDEN: tuple[dict[str, str], ...] = (
    {
        "id": "error_rate_without_labels",
        "why": "the margin orders windows for review; it is not a probability of "
        "error, and the review set is chosen to hold errors, not sampled, so no "
        "error rate follows from it without labelling a designed sample "
        "(olmoearth_plan_label_sample)",
    },
)

#: Stated with every comparison: no side is right without labels.
COMPARISON_FORBIDDEN: tuple[dict[str, str], ...] = (
    {
        "id": "winner_without_labels",
        "why": "without labels neither map can be shown right where they differ; "
        "the more confident side is not the right one (see evidence_scope)",
    },
)

#: What ``boundary_neighbours`` counts, stated beside it (exp86 round 7 read it
#: as "likely true class-boundary pixels" and as "edge cells").
BOUNDARY_NEIGHBOURS_MEANS = (
    "boundary_neighbours counts a window's 8 neighbours whose PREDICTED class "
    "differs from its own (0 to 8): a boundary of the map, not a true class "
    "boundary, and not a measure of error."
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


# --------------------------------------------------------------------------- a Studio regression band as scores

#: The assumption that makes a binary score in [0, 1] rankable, returned with
#: every ranking built from one (upstream documents the same reading of a
#: Studio ``per_pixel_regression`` output of a two-class task).
BINARY_SCORE_ASSUMPTION = (
    "The band is a score in [0, 1] decided at 0.5 and is treated as "
    "P(positive) for ranking only: each window's scores are [1 - s, s], so "
    "its margin is |2s - 1| and the least decided windows (s nearest 0.5) are "
    "opened first; the most confident windows (s near 0 or 1) come last. "
    "Ranking by distance from 0.5 needs no calibration, but these numbers are "
    "not probabilities of error, and no recorded experiment grades this case."
)


def regression_scores(
    values: Sequence[float],
    *,
    min_value: float | None,
    max_value: float | None,
    threshold: float | None = None,
) -> tuple[list[list[float]], str, str]:
    """Two-class scores for a regression band, so the margin ranking applies.

    Returns ``(rows, score_kind, assumption)``. A band whose declared range
    is ``[0, 1]`` read at 0.5 (``threshold`` omitted or 0.5) is a binary
    score: rows are ``[1 - s, s]`` (``score_kind="binary_score"``). Any other
    band needs a decision threshold, because a margin is a distance from a
    decision: rows are ``[-d / 2, d / 2]`` with ``d = (v - t) / (max - min)``,
    so the margin is ``|v - t|`` scaled by the declared range
    (``score_kind="threshold_distance"``; unscaled when no range is declared).
    Class 1 is "above the threshold" in both cases.

    Raises
    ------
    ValueError
        For a non-binary band without ``threshold``, or a non-finite value.
    """
    vals = [float(v) for v in values]
    if any(math.isnan(v) or math.isinf(v) for v in vals):
        raise ValueError("a sampled value is not finite; drop no-data first")
    binary = min_value == 0.0 and max_value == 1.0
    if binary and (threshold is None or threshold == 0.5):
        return [[1.0 - v, v] for v in vals], "binary_score", BINARY_SCORE_ASSUMPTION
    if threshold is None:
        declared = (
            f"[{min_value}, {max_value}]"
            if min_value is not None and max_value is not None
            else "not declared"
        )
        raise ValueError(
            f"this regression band's range is {declared}, not [0, 1]: a margin "
            "is a distance from a decision, so it needs a decision threshold. "
            "Pass threshold (the value at which the map's decision flips)."
        )
    t = float(threshold)
    span = 1.0
    scaled = "unscaled (the band declares no range)"
    if min_value is not None and max_value is not None and max_value > min_value:
        span = float(max_value) - float(min_value)
        scaled = f"divided by the declared range {span:g}"
    rows = [[-(v - t) / span / 2.0, (v - t) / span / 2.0] for v in vals]
    assumption = (
        f"The band is read as a decision at the threshold {t:g}: a window's "
        f"margin is |value - {t:g}| {scaled}, and the windows nearest the "
        "threshold are opened first. This assumes the map's decision flips at "
        "that threshold; the distance is not a probability of error, and no "
        "recorded experiment grades this case."
    )
    return rows, "threshold_distance", assumption


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
    score_kind: str | None = None,
    model: str | None = None,
    keep_full: bool = False,
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
    score_kind
        What the rows are when they are not class scores (see
        :func:`ranking_evidence_scope`): ``"window_confidence"`` (a scores
        provider's rows; read with ``score_type``), ``"binary_score"`` or
        ``"threshold_distance"``. Only the evidence scope reads it.
    model
        The model the scores came from, when known; only the evidence scope
        reads it (the measurement is on the OlmoEarth family).
    keep_full
        Also return ``review_full``: every window of the review set at the
        budget as its own row, listed or not, for the caller to save.

    Returns
    -------
    dict
        ``review`` (the ordered rows), ``n_review``, the margin summary,
        the ceiling arithmetic when ``error_rate`` is given, the one-sentence
        ``evidence_scope`` (and ``evidence_covers_this_case``), and the output
        contract: ``facts`` (``margin_ratio``), ``must_state`` (a short
        sentence stating the limit when the evidence does not cover this
        case; the scope itself stays in ``evidence_scope``) and
        ``forbidden_claims``.

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
            """Boundary windows first, ascending margin within each block; ties by
            descending window index, as olmoearth-inferencex's ``review_order``."""
            return (0.0 if touches[i] else 1.0, marg[i], -i)

    else:

        def rank_key(i: int) -> tuple[float, ...]:
            """Ascending margin: the least confident window is opened first; ties by
            descending window index, as olmoearth-inferencex's ``review_order``, so the
            two produce the same review set on a tied map."""
            return (marg[i], -i)

    ranked = sorted(range(n), key=rank_key)

    k = max(1, int(round(budget * n)))

    def review_row(rank: int, i: int) -> dict[str, Any]:
        """One review row: its rank, window, class, margin and what else is known."""
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
        return row

    rows = [
        review_row(rank, i) for rank, i in enumerate(ranked[:k][:max_listed], start=1)
    ]

    summary = margin_summary(marg, ranked, k, len(rows))
    kind = score_kind or used_type
    if kind == "window_confidence":
        kind = f"window_confidence_{used_type}"
    elif kind not in ("binary_score", "threshold_distance"):
        kind = used_type
    scope = ranking_evidence_scope(
        score_kind=kind, n_classes=len(matrix[0]), model=model
    )
    ratio = margin_ratio_fact(
        summary, _margin_range([marg[i] for i in ranked[:MARGIN_RATIO_FIRST]])
    )
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
        "margin_summary": summary,
        **_scope_fields(scope),
        "facts": [ratio] if ratio else [],
        "forbidden_claims": [dict(c) for c in RANKING_FORBIDDEN]
        + evidence_outside_scope(scope),
    }
    if keep_full:
        # Every window of the review set, listed or not, as rows of their own
        # (a caller writes them to a file): exp86 round 8 (brief 8 on the
        # cluster) said "the full list is saved" where no file held it.
        out["review_full"] = [
            review_row(rank, i) for rank, i in enumerate(ranked[:k], start=1)
        ]
    if bnd is not None:
        out["n_boundary_windows"] = sum(1 for b in bnd if b > 0)
        out["boundary_neighbours_means"] = BOUNDARY_NEIGHBOURS_MEANS
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
    return out


def _num(value: float) -> str:
    """A number for a fact's sentence: at most 6 decimals, no trailing zeros."""
    return f"{value:,.6f}".rstrip("0").rstrip(".")


def _pct(share: float) -> str:
    """A share as a percent with one decimal, for a fact's sentence."""
    return f"{share * 100:.1f}%"


def _ratios(
    median: float, rng: list[float] | None
) -> tuple[float | None, float | None]:
    """``(median / highest, median / lowest)`` of a margin range, rounded; a 0 end gives ``None``."""
    if not rng:
        return None, None
    lo, hi = float(rng[0]), float(rng[1])
    return (
        round(median / hi, 2) if hi > 0 else None,
        round(median / lo, 2) if lo > 0 else None,
    )


def _times(low: float | None, high: float | None) -> str:
    """``13.00 to 41.00 times``, ``at least 13.00 times``, or ``no finite multiple of``."""
    if low is None:
        return "no finite multiple of"
    if high is None:
        return f"at least {low:,.2f} times"
    if low == high:
        return f"{low:,.2f} times"
    return f"{low:,.2f} to {high:,.2f} times"


def _range_text(rng: list[float]) -> str:
    """``0.1 to 0.4``, or ``0.1`` when both ends are one value."""
    lo, hi = float(rng[0]), float(rng[1])
    return _num(lo) if lo == hi else f"{_num(lo)} to {_num(hi)}"


#: The first listed windows an answer usually shows (a table of ten); the
#: ``margin_ratio`` fact states their own ratio beside the listed windows'.
#: exp86 round 8 (brief 8 on the cluster, run 2) showed ten windows and gave
#: them the 50 listed windows' "13 to 41 times"; the ten shown are 20.68 to
#: 41.31 times below the median.
MARGIN_RATIO_FIRST = 10


def margin_ratio_fact(
    summary: dict[str, Any], first_range: list[float] | None = None
) -> dict[str, Any] | None:
    """The ``margin_ratio`` fact: the median margin over the listed windows' margins,
    over the first listed ones', and over every margin in the review set.

    exp86 round 6 (brief 8) called the listed windows "3-4 orders of magnitude
    more uncertain" where the median margin was 13 to 41 times theirs. A
    ratio's low end is the median over the highest margin of the range and
    its high end the median over the lowest; an end whose margin is 0 (a tie
    between a window's top two classes) is ``None``. The review set's range
    covers all ``n_review`` windows at the budget, listed or not (exp87
    review: an answer about the review set read the listed range as its).
    ``listed_n`` says how many windows the listed ratio covers, and
    ``first_range``, the margin range of the first
    :data:`MARGIN_RATIO_FIRST` listed windows when more are listed, gives
    theirs (exp86 round 8 put the 50 listed windows' ratio on the 10 shown).

    Returns
    -------
    dict or None
        ``id``, ``listed_n``, ``listed_low``, ``listed_high``,
        ``review_set_low``, ``review_set_high``, ``versus`` (``"median"``),
        with ``first_range`` also ``first_n``, ``first_low`` and
        ``first_high``, and ``sentence``; ``None`` when the review set is
        empty.
    """
    review = summary.get("review_set", {}).get("margin_range")
    if not review:
        return None
    listed = summary["listed"]["margin_range"]
    median = float(summary["median_margin"])
    n, k = summary["n_windows"], summary["review_set"]["n"]
    n_listed = summary["listed"]["n"]
    listed_low, listed_high = _ratios(median, listed)
    review_low, review_high = _ratios(median, review)
    first_n = min(MARGIN_RATIO_FIRST, n_listed)
    first = first_range if listed and first_range and n_listed > first_n else None
    first_low, first_high = _ratios(median, first)
    shown = (
        f", {_times(first_low, first_high)} those of the first {first_n} listed "
        f"({_range_text(first)})"
        if first
        else ""
    )
    head = f"The median margin over all {n:,} windows ({_num(median)}) is"
    whole = (
        f"all {k:,} windows in the review set"
        if k > 1
        else "the 1 window in the review set"
    )
    if not listed:
        body = (
            f"{_times(review_low, review_high)} the margins of {whole} "
            f"({_range_text(review)}); none is listed"
        )
    elif n_listed >= k:
        body = (
            f"{_times(listed_low, listed_high)} the margins of {whole}, every one "
            f"listed ({_range_text(listed)}){shown}"
        )
    else:
        some = (
            "the 1 listed window"
            if n_listed == 1
            else f"the {n_listed:,} listed windows"
        )
        body = (
            f"{_times(listed_low, listed_high)} the margins of {some} "
            f"({_range_text(listed)}){shown} and {_times(review_low, review_high)} "
            f"those of {whole} ({_range_text(review)})"
        )
    # The listed windows are the first of the review set, so its lowest margin
    # is the lowest of both.
    tail = (
        "; a margin of 0 is a tie between a window's top two classes, so no ratio "
        "bounds it"
        if float(review[0]) == 0.0
        else ""
    )
    out: dict[str, Any] = {
        "id": "margin_ratio",
        "listed_n": n_listed,
        "listed_low": listed_low,
        "listed_high": listed_high,
        "review_set_low": review_low,
        "review_set_high": review_high,
        "versus": "median",
    }
    if first:
        out.update(first_n=first_n, first_low=first_low, first_high=first_high)
    out["sentence"] = f"{head} {body}{tail}."
    return out


def _margin_range(values: list[float]) -> list[float] | None:
    """``[lowest, highest]`` of some margins, rounded, or ``None`` for none."""
    return [round(min(values), 6), round(max(values), 6)] if values else None


def margin_summary(
    marg: Sequence[float], ranked: Sequence[int], k: int, n_listed: int
) -> dict[str, Any]:
    """The margins of all windows, of the listed ones and of the rest, labelled.

    exp86 round 2 (brief 2, run 3) read the summary's median as the lower end
    of the unlisted windows' range ("scores ranged from ~0.96 to ~0.99 ...
    (median margin 0.965)") where the lowest unlisted margin was 0.786. Every
    field now says what it is, and the unlisted windows get their own range.

    Parameters
    ----------
    marg
        Every window's margin.
    ranked
        Window indices in review order.
    k
        Windows in the review set at the budget.
    n_listed
        Review rows listed inline (the first ``n_listed`` of ``ranked``).

    Returns
    -------
    dict
        ``n_windows``, ``lowest_margin``, ``median_margin``,
        ``highest_margin``, ``margin_at_budget_cut``, ``listed``,
        ``not_listed`` and ``review_set`` (every window at the budget, listed
        or not; each ``{n, margin_range}``) and a ``reading``.
    """
    n = len(marg)
    ordered = sorted(marg)
    listed = [marg[i] for i in ranked[:n_listed]]
    rest = [marg[i] for i in ranked[n_listed:]]
    in_review = [marg[i] for i in ranked[: min(k, n)]]
    not_listed = _margin_range(rest)
    reading = (
        f"Over all {n} windows: lowest_margin is the smallest margin (the most "
        "suspect window) and highest_margin the largest; median_margin is the "
        "middle value (half the windows lie below it), not a lower end. "
        f"margin_at_budget_cut is the {min(k, n)}-th lowest margin (n_review at "
        f"the budget); review_set is the margin range of all {len(in_review)} "
        "windows in the review set, listed or not. "
    )
    if not_listed is None:
        reading += "Every window is listed, so not_listed has no range."
    else:
        reading += (
            f"The {len(rest)} windows not listed (not_listed) have margins from "
            f"{not_listed[0]:g} to {not_listed[1]:g}."
        )
    return {
        "n_windows": n,
        "lowest_margin": round(ordered[0], 6),
        "median_margin": round(ordered[n // 2], 6),
        "highest_margin": round(ordered[-1], 6),
        "margin_at_budget_cut": round(ordered[min(k, n) - 1], 6),
        "listed": {"n": len(listed), "margin_range": _margin_range(listed)},
        "not_listed": {"n": len(rest), "margin_range": not_listed},
        "review_set": {"n": len(in_review), "margin_range": _margin_range(in_review)},
        "reading": reading,
    }


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


def _class_label(k: Any, names: dict[str, str] | None) -> str:
    """``class 4 (grassland_barren)`` when the class has a name, else ``class 4``."""
    name = names.get(str(k)) if names else None
    return f"class {k} ({name})" if name else f"class {k}"


def _bands(size: int, n_bands: int) -> list[tuple[int, int]]:
    """Contiguous ``(first, last)`` index ranges splitting ``0 .. size - 1`` into
    ``n_bands`` near-equal bands (one per index when ``size`` is smaller)."""
    n = max(1, min(n_bands, size))
    edges = [size * k // n for k in range(n + 1)]
    return [(edges[k], edges[k + 1] - 1) for k in range(n)]


#: How a grid's rows run on the ground, when the scores are georeferenced:
#: row 0 in the north, or in the south. ``None`` is a grid with no
#: georeference, whose rows need not run north to south at all.
ROW_ORDERS = ("north_to_south", "south_to_north")

#: The words for row band 0 by row order: exp86 round 8 (brief 7 on files)
#: called row band 0 of a grid of 400 chips stacked in dataset order "the
#: northmost band" and rows 2,800-4,199 "mid-south", from the tool's own "the
#: north edge of a north-up map" for a file with no georeference.
_BAND_0 = {
    "north_to_south": "the north edge: the scores' window locations put row 0 north",
    "south_to_north": "the south edge: the scores' window locations put row 0 south",
}

#: Said of a grid with no georeference.
NO_GEOREFERENCE = (
    "the scores carry no georeference (no window locations or transform), so the "
    "rows need not run north to south and no band is a compass direction"
)


def spatial_breakdown(
    differing: Sequence[int],
    present: Sequence[int],
    grid: tuple[int, int],
    n_bands: int = SPATIAL_BANDS,
    *,
    row_order: str | None = None,
) -> dict[str, Any]:
    """Where the differing windows sit: their share in each row band and each column band.

    exp86 rounds 6 and 7 read place off the first listed windows ("a long
    strip along the north edge" for 1.3% of the differences; blocks invented
    from ten windows). This counts every differing window instead.

    Bands are numbered from 0: row band 0 holds row 0, the grid's first rows,
    and column band 0 holds column 0. Row band 0 is called north (or south)
    only with a ``row_order`` from georeferenced scores (:data:`ROW_ORDERS`);
    without one the reading says the rows need not run north to south.

    Parameters
    ----------
    differing
        Row-major grid index of every differing window.
    present
        Row-major grid index of every compared window (the whole grid, or
        the windows a scores file holds).
    grid
        ``(rows, cols)``.
    n_bands
        Bands per axis (fewer when the grid has fewer rows or columns).
    row_order
        ``"north_to_south"``, ``"south_to_north"`` or ``None`` (no
        georeference).

    Returns
    -------
    dict
        ``row_bands`` and ``col_bands`` (each band's index range, windows,
        differing windows, ``share_of_differing``, ``share_of_windows`` and
        ``share_of_band_differing``), ``top_band_share`` (row band 0's share
        of the differing windows), the ``max_band`` holding the most of them,
        with ``n_tied`` other bands holding as many, ``row_order`` and a
        ``reading``.
    """
    order = row_order if row_order in ROW_ORDERS else None
    band_0 = f" ({_BAND_0[order]})" if order else ""
    unreferenced = f"; {NO_GEOREFERENCE}" if order is None else ""
    rows, cols = int(grid[0]), int(grid[1])
    n_diff, n_present = len(differing), len(present)

    def axis(size: int, coord: Any, label: str) -> list[dict[str, Any]]:
        bands = _bands(size, n_bands)
        band_of = [0] * size
        for k, (first, last) in enumerate(bands):
            for i in range(first, last + 1):
                band_of[i] = k
        n_w, n_d = [0] * len(bands), [0] * len(bands)
        for w in present:
            n_w[band_of[coord(w)]] += 1
        for w in differing:
            n_d[band_of[coord(w)]] += 1
        return [
            {
                "band": k,
                label: [first, last],
                "of_grid": (
                    f"{round(100 * first / size, 1):g}-"
                    f"{round(100 * (last + 1) / size, 1):g}%"
                ),
                "n_windows": n_w[k],
                "n_differing": n_d[k],
                "share_of_differing": round(n_d[k] / n_diff, 6) if n_diff else None,
                "share_of_windows": round(n_w[k] / n_present, 6) if n_present else None,
                "share_of_band_differing": (
                    round(n_d[k] / n_w[k], 6) if n_w[k] else None
                ),
            }
            for k, (first, last) in enumerate(bands)
        ]

    row_bands = axis(rows, lambda w: int(w) // cols, "rows")
    col_bands = axis(cols, lambda w: int(w) % cols, "cols")
    out: dict[str, Any] = {
        "grid": [rows, cols],
        "n_bands": [len(row_bands), len(col_bands)],
        "row_bands": row_bands,
        "col_bands": col_bands,
        "top_band_share": row_bands[0]["share_of_differing"],
        "max_band": None,
        "row_order": order,
        "reading": (
            "Each band is a strip of whole rows (row_bands) or whole columns "
            "(col_bands) of the window grid, numbered from 0: row band 0 holds "
            f"row 0, the grid's first rows{band_0}, and column band 0 holds column "
            f"0{unreferenced}. share_of_differing is the band's share of ALL "
            "differing windows; share_of_windows is its share of all compared "
            "windows, the share an even spread would give it; "
            "share_of_band_differing is the share of the band's own windows that "
            "differ. top_band_share is row band 0's share_of_differing; max_band "
            "is the band, of either axis, holding the most differing windows."
        ),
    }
    if not n_diff:
        return out
    # The most differing windows, by count: rows before columns and the lower
    # band on a tie, and the tie counted, so an even spread is not read as a
    # concentration.
    candidates = [("rows", b) for b in row_bands] + [("cols", b) for b in col_bands]
    most = max(b["n_differing"] for _, b in candidates)
    axis_name, top = next((a, b) for a, b in candidates if b["n_differing"] == most)
    out["max_band"] = {
        "axis": axis_name,
        "band": top["band"],
        "of_grid": top["of_grid"],
        "share": top["share_of_differing"],
        axis_name: top[axis_name],
        "share_of_windows": top["share_of_windows"],
        "n_tied": sum(1 for _, b in candidates if b["n_differing"] == most) - 1,
    }
    return out


def _band_where(axis: str, first: int, last: int) -> str:
    """``rows 0-15``, ``column 3``."""
    word = "row" if axis == "rows" else "column"
    return f"{word}s {first}-{last}" if first != last else f"{word} {first}"


def concentration_fact(
    spatial: dict[str, Any], n_differing: int
) -> dict[str, Any] | None:
    """The ``concentration`` fact: row band 0's share of the differing windows,
    and the band of either axis holding the most of them.

    ``top_band_share`` is row band 0's share (the grid's first rows), whatever
    band holds the most: exp86 round 7 put the differences "along the north
    edge" where that band held 1.3% of them. Row band 0 is the northmost (or
    southmost) band only with the ``row_order`` of georeferenced scores; exp86
    round 8 (brief 7 on files) called it "the northmost band" for a grid of
    chips stacked in dataset order, so without one the sentence names it by
    its rows and says the rows need not run north to south.

    Returns
    -------
    dict or None
        ``id``, ``grid``, ``n_differing``, ``top_band_share``, ``max_band``
        (``axis`` ``"rows"`` or ``"cols"``, ``band`` from 0, ``of_grid``,
        ``share``), ``row_order`` and ``sentence``; ``None`` when nothing
        differs.
    """
    top = spatial.get("max_band")
    if not top or not n_differing:
        return None
    rows, cols = spatial["grid"]
    n_row_bands, n_col_bands = spatial["n_bands"]
    north = spatial["row_bands"][0]
    north_where = _band_where("rows", *north["rows"])
    order = spatial.get("row_order")
    if order == "north_to_south":
        band_0 = f"the northmost row band ({north_where}, band 0 of {n_row_bands})"
    elif order == "south_to_north":
        band_0 = (
            f"the southmost row band ({north_where}, band 0 of {n_row_bands}; row 0 "
            "is the south edge here)"
        )
    else:
        band_0 = (
            f"row band 0 ({north_where}, the grid's first rows, band 0 of "
            f"{n_row_bands})"
        )
    unreferenced = (
        " The scores carry no georeference, so the rows need not run north to " "south."
        if order not in ROW_ORDERS
        else ""
    )
    first, last = top[top["axis"]]
    where = _band_where(top["axis"], first, last)
    size, noun = (rows, "rows") if top["axis"] == "rows" else (cols, "columns")
    tied = (
        ""
        if not top["n_tied"]
        else f" (tied with {top['n_tied']} other band{'s' if top['n_tied'] > 1 else ''})"
    )
    head = (
        f"Of the {n_differing:,} differing windows, {_pct(north['share_of_differing'])} "
        f"lie in {band_0}, which holds {_pct(north['share_of_windows'])} of all "
        "compared windows"
    )
    if top["axis"] == "rows" and top["band"] == 0:
        body = (
            f"; that is the most of any of the {n_row_bands} row and {n_col_bands} "
            f"column bands{tied}."
        )
    else:
        body = (
            f"; the most in any band{tied}, {_pct(top['share'])}, lie in {where} "
            f"({top['of_grid']} of the grid's {size:,} {noun}, band {top['band']}), "
            f"which holds {_pct(top['share_of_windows'])} of all compared windows."
        )
    return {
        "id": "concentration",
        "grid": [rows, cols],
        "n_differing": n_differing,
        "top_band_share": spatial["top_band_share"],
        "max_band": {
            "axis": top["axis"],
            "band": top["band"],
            "of_grid": top["of_grid"],
            "share": top["share"],
        },
        "row_order": order if order in ROW_ORDERS else None,
        "sentence": head + body + unreferenced,
    }


def _class_pairs(
    pairs: dict[tuple[Any, Any], int], n_diff: int, names: dict[str, str] | None
) -> list[dict[str, Any]]:
    """Class changes as unordered pairs, each with both directions' counts, largest first."""
    merged: dict[tuple[Any, Any], dict[tuple[Any, Any], int]] = {}
    for (pa, pb), n in pairs.items():
        key = (pa, pb) if pa <= pb else (pb, pa)
        merged.setdefault(key, {})[(pa, pb)] = n
    out = []
    for (x, y), dirs in merged.items():
        total = sum(dirs.values())
        directions = sorted(
            (
                {"class_a": pa, "class_b": pb, "n": dirs.get((pa, pb), 0)}
                for pa, pb in ((x, y), (y, x))
            ),
            key=lambda d: -d["n"],
        )
        entry: dict[str, Any] = {
            "pair": f"{x}<->{y}",
            "classes": [x, y],
            "n": total,
            "share_of_differing": round(total / n_diff, 6),
            "directions": directions,
        }
        if names:
            entry["class_names"] = [names.get(str(x)), names.get(str(y))]
        out.append(entry)
    out.sort(key=lambda e: (-e["n"], str(e["classes"])))
    return out


def _change(pa: Any, pb: Any, names: dict[str, str] | None) -> str:
    """``class 2 (grass) in map A to class 1 (forest) in map B``."""
    return f"{_class_label(pa, names)} in map A to {_class_label(pb, names)} in map B"


def _ranked_changes(
    pairs: dict[tuple[Any, Any], int]
) -> list[tuple[tuple[Any, Any], int]]:
    """Directed class changes, largest count first, ties in a fixed order."""
    return sorted(pairs.items(), key=lambda kv: (-kv[1], str(kv[0])))


def dominant_change_fact(
    pairs: dict[tuple[Any, Any], int],
    n_diff: int,
    names: dict[str, str] | None,
) -> dict[str, Any] | None:
    """The ``dominant_change`` fact: the largest directed class change A -> B, with its reverse.

    Every direction whose count equals the top one is named in the sentence,
    the reverse included (``tied_with_reverse``), so a tie is never read as
    one direction winning. Counted over ``pairs``, every differing window's
    ``(class in A, class in B)``.

    Returns
    -------
    dict or None
        ``id``, ``from_class``, ``to_class``, ``n``, ``share``,
        ``reverse_n``, ``reverse_share``, ``tied_with_reverse`` and
        ``sentence``; ``None`` when nothing differs.
    """
    if not pairs or not n_diff:
        return None
    ranked = _ranked_changes(pairs)
    (pa, pb), n = ranked[0]
    tied = [key for key, count in ranked[1:] if count == n]
    rev = pairs.get((pb, pa), 0)
    tied_with_reverse = (pb, pa) in tied
    counted = f"{n:,} of the {n_diff:,} differing windows ({_pct(n / n_diff)})"
    reverse = (
        f"the reverse, {_change(pb, pa, names)}, is {rev:,} ({_pct(rev / n_diff)})"
    )
    if not tied:
        sentence = (
            f"The most common change is {_change(pa, pb, names)}: {counted}; {reverse}."
        )
    elif tied == [(pb, pa)]:
        sentence = (
            f"The most common change is a tie between a direction and its reverse, "
            f"{counted} each: {_change(pa, pb, names)}, and {_change(pb, pa, names)}."
        )
    else:
        named = [_change(a, b, names) for a, b in [(pa, pb), *tied]]
        listed = ", ".join(named[:-1]) + ", and " + named[-1]
        sentence = (
            f"The most common change is a tie between {len(named)} directions, "
            f"{counted} each: {listed}"
            + ("." if tied_with_reverse else f"; for the first, {reverse}.")
        )
    return {
        "id": "dominant_change",
        "from_class": pa,
        "to_class": pb,
        "n": n,
        "share": round(n / n_diff, 6),
        "reverse_n": rev,
        "reverse_share": round(rev / n_diff, 6),
        "tied_with_reverse": tied_with_reverse,
        "sentence": sentence,
    }


def more_confident_fact(a_more: int, b_more: int, n_diff: int) -> dict[str, Any] | None:
    """The ``more_confident_side`` fact, stated with why it does not pick a winner.

    The side is decided on counts (windows where A's margin is larger, where
    B's is), never on rounded shares, and is ``"neither"`` when they are
    equal. Returns ``None`` when nothing differs.
    """
    if not n_diff:
        return None
    equal = n_diff - a_more - b_more
    share_a, share_b = round(a_more / n_diff, 6), round(b_more / n_diff, 6)
    share_equal = round(equal / n_diff, 6)
    tie_text = f", equal on {_pct(equal / n_diff)}" if equal else ""
    # No upstream figure: the comparison's evidence does not cover this pair,
    # and exp86 round 8 carried exp58's "51 to 70 percent" from this sentence
    # to a pair it excludes ("in comparable cases", brief 3 on the cluster).
    upstream = (
        "a margin is not a label, and no recorded experiment grades which of "
        "these two maps is right"
    )
    if a_more == b_more:
        side = "neither"
        sentence = (
            f"Neither map is more often the more confident: maps A and B each have "
            f"the larger margin on {_pct(a_more / n_diff)} of the {n_diff:,} "
            f"differing windows{tie_text}; a larger margin would not make a side "
            f"right anyway: {upstream}."
        )
    else:
        side, other = ("A", "B") if a_more > b_more else ("B", "A")
        mine, theirs = max(a_more, b_more), min(a_more, b_more)
        sentence = (
            f"Map {side} has the larger margin on {_pct(mine / n_diff)} of the "
            f"{n_diff:,} differing windows (map {other} on {_pct(theirs / n_diff)}"
            f"{tie_text}); this does not make {side} right there: {upstream}."
        )
    return {
        "id": "more_confident_side",
        "side": side,
        "share_a": share_a,
        "share_b": share_b,
        "share_equal": share_equal,
        "sentence": sentence,
    }


#: What a comparison's boundary shares count, stated beside them (exp86 round 6
#: read a 78% boundary share as "boundary placement rather than whole-window
#: flips").
BOUNDARY_SHARE_MEANS = (
    "boundary_share_overall and boundary_share_of_differing count windows with at "
    "least one of their 8 neighbours of a different class in map A's PREDICTED "
    "classes: a boundary of map A, not a true class boundary. Every differing "
    "window is a change of that whole window's class, on a boundary or not."
)


def compare_scores(
    scores_a: Sequence[Sequence[float]],
    scores_b: Sequence[Sequence[float]],
    grid: tuple[int, int] | None = None,
    max_listed: int | None = DEFAULT_MAX_LISTED,
    *,
    windows: Sequence[int] | None = None,
    class_names: dict[str, str] | None = None,
    row_order: str | None = None,
) -> dict[str, Any]:
    """Compare two inferences of the same windows: how much they differ and where.

    Which side is right is not decided here, because it cannot be without labels:
    upstream measured that the more confident side is right on 51 to 70 percent
    of differing windows and that confidence does not order the set (exp58), so
    the honest answer to "which one do I believe" is to decline and say why.
    That measurement does not cover the pair at hand, so no field of the result
    carries its figures or its dataset (only :func:`evidence_detail` does).

    Parameters
    ----------
    scores_a, scores_b
        The two inferences' rows, one per window, in the same order.
    grid
        ``(rows, cols)`` of the window grid. With ``windows`` omitted the rows
        are the whole grid row-major, and the boundary reading is made.
    max_listed
        Differing windows listed, first in window order; ``None`` lists all.
    windows
        The row-major grid index of each row, when the rows leave windows out
        (a scores file's no-data windows). With ``grid``, the spatial breakdown
        and the listing place rows by it; the boundary reading, which needs
        every window, is not made.
    class_names
        Class id (as a string) to name, when the scores name their classes;
        used in ``class_changes``, ``class_pairs``, the listing and the facts.
    row_order
        How the grid's rows run on the ground when the scores are
        georeferenced (:data:`ROW_ORDERS`); ``None`` for scores with no
        georeference, whose row bands are never called north or south.

    Returns
    -------
    dict
        The counts (``n_differing``, ``share_differing``), the directed
        ``class_changes`` and undirected ``class_pairs``, each side's share of
        the differing windows on which it is the more confident, the
        ``spatial`` breakdown when a grid is known, the listing
        (``differing``), the one-sentence ``evidence_scope`` and the output
        contract (``facts``, ``must_state``, ``forbidden_claims``).
    """
    a = _as_matrix(scores_a)
    b = _as_matrix(scores_b)
    if len(a) != len(b):
        msg = f"the two inferences cover different window counts: {len(a)} and {len(b)}"
        raise ValueError(msg)
    n = len(a)
    if windows is not None:
        windows = [int(w) for w in windows]
        if len(windows) != n:
            msg = f"windows names {len(windows)} grid cells for {n} rows"
            raise ValueError(msg)
    if grid is not None:
        rows_, cols_ = int(grid[0]), int(grid[1])
        if windows is None and rows_ * cols_ != n:
            msg = f"grid {grid} does not match {n} windows"
            raise ValueError(msg)
        if windows is not None and any(not 0 <= w < rows_ * cols_ for w in windows):
            msg = f"a window index lies outside the {rows_}x{cols_} grid"
            raise ValueError(msg)
    names = class_names or None
    ca, cb = predicted_classes(a), predicted_classes(b)
    ma, mb = margins(a), margins(b)
    diff = [i for i in range(n) if ca[i] != cb[i]]
    scope = comparison_evidence_scope()
    out: dict[str, Any] = {
        "n_windows": n,
        "n_differing": len(diff),
        "share_differing": round(len(diff) / n, 6) if a else 0.0,
        "mean_margin_a_on_differing": (
            round(sum(ma[i] for i in diff) / len(diff), 6) if diff else None
        ),
        "mean_margin_b_on_differing": (
            round(sum(mb[i] for i in diff) / len(diff), 6) if diff else None
        ),
        "which_side_is_right": "not resolvable without labels",
        **_scope_fields(scope),
    }
    if grid is not None and windows is None:
        bcount = boundary_counts(ca, rows_, cols_)
        on_boundary = [bcount[i] > 0 for i in range(n)]
        out["boundary_share_overall"] = round(sum(on_boundary) / n, 6)
        out["boundary_share_of_differing"] = (
            round(sum(on_boundary[i] for i in diff) / len(diff), 6) if diff else None
        )
        out["where"] = (
            "mostly on class boundaries"
            if diff
            and out["boundary_share_of_differing"] > out["boundary_share_overall"]
            else "spread across the scene"
        )
        out["boundary_means"] = BOUNDARY_SHARE_MEANS
    # What the differences are, over all of them: the listing below is the first
    # few in window order (the top rows of the grid), and exp86 round 6's answers
    # read patterns off it that the full set contradicts ("most flip 1 -> 0" when
    # 1,247 of 1,570 flip 0 -> 1; "a strip along the north edge" for 1.3%).
    pairs: dict[tuple[Any, Any], int] = {}
    for i in diff:
        pairs[(ca[i], cb[i])] = pairs.get((ca[i], cb[i]), 0) + 1
    ranked = _ranked_changes(pairs)
    out["class_changes"] = [
        {
            "class_a": pa,
            "class_b": pb,
            "n": count,
            "share_of_differing": round(count / len(diff), 6),
        }
        for (pa, pb), count in ranked[:CLASS_CHANGES_LISTED]
    ]
    if names:
        for change in out["class_changes"]:
            change["class_name_a"] = names.get(str(change["class_a"]))
            change["class_name_b"] = names.get(str(change["class_b"]))
    out["n_class_changes"] = len(pairs)
    # The same changes with direction set aside ("A <-> B 37%" was read as both
    # directions, exp86 round 7): each pair carries both directions' counts.
    all_pairs = _class_pairs(pairs, len(diff), names)
    out["class_pairs"] = all_pairs[:CLASS_CHANGES_LISTED]
    out["n_class_pairs"] = len(all_pairs)
    out["class_pairs_reading"] = (
        "class_pairs counts, for each unordered pair of classes, the windows where "
        "one map has one class and the other map the other, both directions "
        "together; 'directions' splits that count by which map has which class, "
        "as class_changes does."
    )
    a_more = sum(ma[i] > mb[i] for i in diff)
    b_more = sum(mb[i] > ma[i] for i in diff)
    out["a_more_confident_share_of_differing"] = (
        round(a_more / len(diff), 6) if diff else None
    )
    out["b_more_confident_share_of_differing"] = (
        round(b_more / len(diff), 6) if diff else None
    )
    cols_of = int(grid[1]) if grid is not None else None
    listed = []
    for i in diff if max_listed is None else diff[:max_listed]:
        w = windows[i] if windows is not None else i
        row: dict[str, Any] = {
            "window_index": w,
            "class_a": ca[i],
            "class_b": cb[i],
            "margin_a": round(ma[i], 6),
            "margin_b": round(mb[i], 6),
        }
        if names:
            row["class_name_a"] = names.get(str(ca[i]))
            row["class_name_b"] = names.get(str(cb[i]))
        if cols_of:
            row["row"], row["col"] = divmod(w, cols_of)
        listed.append(row)
    out["differing"] = listed
    out["n_differing_listed"] = len(listed)
    out["listing_order"] = (
        "the listed windows are the first differing windows in window order (from "
        "the grid's first rows), not a sample of them: read which classes change "
        "and in which direction from class_changes and the shares, not from the list"
    )
    facts: list[dict[str, Any]] = []
    dominant = dominant_change_fact(pairs, len(diff), names)
    if dominant:
        facts.append(dominant)
    confident = more_confident_fact(a_more, b_more, len(diff))
    if confident:
        facts.append(confident)
    if grid is not None:
        present = windows if windows is not None else range(n)
        where_diff = [windows[i] for i in diff] if windows is not None else diff
        out["spatial"] = spatial_breakdown(
            where_diff, list(present), (rows_, cols_), row_order=row_order
        )
        concentration = concentration_fact(out["spatial"], len(diff))
        if concentration:
            facts.append(concentration)
    out["facts"] = facts
    out["forbidden_claims"] = [
        dict(c) for c in COMPARISON_FORBIDDEN
    ] + evidence_outside_scope(scope)
    return out
