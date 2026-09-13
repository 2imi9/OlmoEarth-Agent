# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""The ``olmoearth-review-set`` tool bundle (skill #18).

Three tools:

- ``olmoearth_review_set`` -- rank windows by the model's own top-1 minus
  top-2 margin and return the ones a reviewer should open first at a budget.
- ``olmoearth_grade_review_rule`` -- grade any candidate suspicion signal
  against the margin baseline and a no-model control, with a per-group sign
  test, so a signal that merely sounds principled cannot ship unmeasured.
- ``olmoearth_review_budget_ceiling`` -- the arithmetic that stops a good
  ranker being talked down: no rule can catch more than
  ``min(1, budget / error_rate)`` of the errors at a given budget.

Bring-your-own-scores, like skill #8's embeddings: Studio's pixel-value
returns only ``raw_value`` and ``classification``, never probabilities or
logits, so a margin cannot be recovered from a Studio result. Pass the
scores from wherever inference ran. When all you have is hard classes from
two or more Studio results, skill #9's ``olmoearth_ensemble_uncertainty``
is the tool that still works.

No coordinates in, no coordinates out (rule §3.1).
"""

from __future__ import annotations

from typing import Any

from olmoearth_agent.analysis.review_set import (
    DEFAULT_BUDGETS,
    DEFAULT_MAX_LISTED,
    attainable_ceiling,
    grade_rule,
    review_set,
)
from olmoearth_agent.llm.types import ToolSpec
from olmoearth_agent.tools.registry import RegisteredTool, ToolContext

_SCORES_SCHEMA = {
    "type": "array",
    "items": {"type": "array", "items": {"type": "number"}},
    "description": (
        "Per-window model scores, one inner array per window, one number per "
        "class (>= 2 classes). Logits preferred; probabilities accepted."
    ),
}

_SERIES = {"type": "array", "items": {"type": "number"}}


async def _review_set(args: dict[str, Any], _ctx: ToolContext) -> dict[str, Any]:
    """Handler for ``olmoearth_review_set``."""
    grid = args.get("grid")
    return review_set(
        args["scores"],
        budget=float(args.get("budget", 0.05)),
        ids=args.get("ids"),
        grid=(int(grid[0]), int(grid[1])) if grid else None,
        order=str(args.get("order", "confidence")),
        score_type=args.get("score_type"),
        error_rate=(
            float(args["error_rate"]) if args.get("error_rate") is not None else None
        ),
        max_listed=int(args.get("max_listed", DEFAULT_MAX_LISTED)),
    )


async def _grade_review_rule(args: dict[str, Any], _ctx: ToolContext) -> dict[str, Any]:
    """Handler for ``olmoearth_grade_review_rule``."""
    budgets = args.get("budgets") or list(DEFAULT_BUDGETS)
    return grade_rule(
        args["signal"],
        args["errors"],
        baseline=args.get("baseline"),
        control=args.get("control"),
        groups=args.get("groups"),
        budgets=[float(b) for b in budgets],
        higher_is_suspect=bool(args.get("higher_is_suspect", True)),
    )


async def _budget_ceiling(args: dict[str, Any], _ctx: ToolContext) -> dict[str, Any]:
    """Handler for ``olmoearth_review_budget_ceiling``."""
    budget = float(args["budget"])
    error_rate = float(args["error_rate"])
    ceiling = attainable_ceiling(budget, error_rate)
    out: dict[str, Any] = {
        "budget": budget,
        "error_rate": error_rate,
        "attainable_ceiling": round(ceiling, 6),
        "random_baseline": round(budget, 6),
        "formula": "min(1, budget / error_rate)",
        "note": (
            "A reviewer opening this fraction of windows cannot see more "
            "than this share of the errors, however good the ranking is. "
            "Reading a realised capture against 1.0 instead of against this "
            "ceiling is the most common way a working ranker gets dismissed."
        ),
    }
    observed = args.get("observed_capture")
    if observed is not None:
        obs = float(observed)
        out["observed_capture"] = obs
        out["share_of_ceiling"] = round(obs / ceiling, 6) if ceiling > 0 else None
        out["multiple_of_random"] = round(obs / budget, 6) if budget > 0 else None
        if obs > ceiling + 1e-9:
            out["warning"] = (
                "The observed capture exceeds the attainable ceiling, which is "
                "impossible: the budget, the error rate, or the capture was "
                "measured on a different set of units."
            )
    return out


def build_review_set_tools() -> list[RegisteredTool]:
    """Return the ``olmoearth-review-set`` tool bundle."""
    return [
        RegisteredTool(
            spec=ToolSpec(
                name="olmoearth_review_set",
                description=(
                    "Decide WHICH WINDOWS A HUMAN SHOULD CHECK FIRST in a "
                    "prediction map, without any labels. Ranks windows by the "
                    "model's own top-1-minus-top-2 margin (low margin = "
                    "suspect) and returns the review set at your budget, "
                    "optionally boundary-first. Use this whenever the user "
                    "asks 'where is this map likely wrong', 'what should I "
                    "check', 'give me a QA list', or wants to spend a fixed "
                    "review effort well. REQUIRES per-class scores (logits or "
                    "probabilities), which Studio does NOT return -- its "
                    "pixel-value gives only raw_value/classification -- so "
                    "pass scores from wherever inference ran (rslearn predict, "
                    "an embeddings+probe pass, any exported softmax). If you "
                    "only have hard classes from 2+ Studio results, use "
                    "olmoearth_ensemble_uncertainty instead. Evidence: on all "
                    "24 tasks of Ai2's own published embedding suite (14 "
                    "sources, 6.4M graded units) the margin beat the best "
                    "no-model control, 24/24, one-sided sign test p=6e-08; "
                    "and ensemble disagreement, cross-model disagreement, "
                    "two-view disagreement and feature-space typicality were "
                    "each measured against it on expert labels and none "
                    "ranked errors better. The returned 'evidence' block "
                    "names those verdicts and 'caveats' names the honest "
                    "exceptions. This ranks RELATIVE suspicion; it is not a "
                    "calibrated error probability and not OOD detection -- "
                    "pair it with olmoearth_area_of_applicability for the OOD "
                    "regime. Read-only; window indices and ids only, never "
                    "coordinates."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "scores": _SCORES_SCHEMA,
                        "budget": {
                            "type": "number",
                            "default": 0.05,
                            "description": (
                                "Fraction of windows a reviewer can open, in "
                                "(0, 1]. 0.05 = the most suspect 5%."
                            ),
                        },
                        "ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "Optional window labels parallel to 'scores'. "
                                "Do not put lat/lon here (rule §3.1)."
                            ),
                        },
                        "grid": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "minItems": 2,
                            "maxItems": 2,
                            "description": (
                                "Optional [rows, cols]; windows are read "
                                "row-major. Enables the nine-level boundary "
                                "indicator and boundary-first ordering."
                            ),
                        },
                        "order": {
                            "type": "string",
                            "enum": ["confidence", "boundary_first"],
                            "default": "confidence",
                            "description": (
                                "'confidence' = ascending margin. "
                                "'boundary_first' = windows touching a "
                                "predicted class boundary first (needs grid)."
                            ),
                        },
                        "score_type": {
                            "type": "string",
                            "enum": ["logit", "probability"],
                            "description": (
                                "Auto-detected when omitted. The measured "
                                "evidence is for the logit margin."
                            ),
                        },
                        "error_rate": {
                            "type": "number",
                            "description": (
                                "Optional known/estimated error rate; adds the "
                                "attainable-ceiling arithmetic for this budget."
                            ),
                        },
                        "max_listed": {
                            "type": "integer",
                            "default": DEFAULT_MAX_LISTED,
                            "description": "Cap on inline review rows.",
                        },
                    },
                    "required": ["scores"],
                },
            ),
            handler=_review_set,
        ),
        RegisteredTool(
            spec=ToolSpec(
                name="olmoearth_grade_review_rule",
                description=(
                    "Measure whether a CANDIDATE suspicion signal is actually "
                    "better than the model's own confidence at finding errors, "
                    "on units where you DO have labels. Give the candidate "
                    "signal, a 0/1 error indicator, optionally the baseline "
                    "(negated margin) and a no-model control, and optionally "
                    "group ids (scene / event / region). Returns E-AURC "
                    "(lower is better, 0 = oracle), AUROC, error capture at "
                    "each budget with the attainable ceiling, a head-to-head "
                    "verdict, and a one-sided exact sign test across groups -- "
                    "the group is the unit of replication because units inside "
                    "one scene are not independent. Use this before adopting "
                    "ANY new reliability heuristic: upstream this same "
                    "machinery rejected ensemble disagreement, cross-model "
                    "disagreement, two-view disagreement and feature-space "
                    "typicality, all of which sounded principled. A candidate "
                    "that cannot beat the no-model control is measuring the "
                    "scene, not the model. Read-only."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "signal": dict(
                            _SERIES,
                            description="Candidate suspicion score per graded unit.",
                        ),
                        "errors": dict(
                            _SERIES,
                            description=(
                                "0/1 per unit; 1 = the model was wrong there."
                            ),
                        ),
                        "baseline": dict(
                            _SERIES,
                            description=(
                                "The incumbent, normally the NEGATED margin so "
                                "higher means more suspect."
                            ),
                        ),
                        "control": dict(
                            _SERIES,
                            description=(
                                "A no-model control (band index, pixel "
                                "variance, class rarity)."
                            ),
                        ),
                        "groups": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "Optional group id per unit (scene / event / "
                                "region) for the sign test."
                            ),
                        },
                        "budgets": dict(
                            _SERIES,
                            description="Review budgets; default 0.01/0.05/0.10.",
                        ),
                        "higher_is_suspect": {
                            "type": "boolean",
                            "default": True,
                            "description": (
                                "False if a LOW candidate value means suspect."
                            ),
                        },
                    },
                    "required": ["signal", "errors"],
                },
            ),
            handler=_grade_review_rule,
        ),
        RegisteredTool(
            spec=ToolSpec(
                name="olmoearth_review_budget_ceiling",
                description=(
                    "The ceiling arithmetic for a review budget: no ranking "
                    "rule can catch more than min(1, budget/error_rate) of the "
                    "errors when a reviewer only opens 'budget' of the "
                    "windows. Call this whenever anyone quotes an error-capture "
                    "number, including one from another tool or paper: pass "
                    "observed_capture to get the share of the ceiling actually "
                    "reached and the multiple of random review. Judging a "
                    "capture of 0.45 against 1.0 rather than against a ceiling "
                    "of 0.50 is the standard way a working ranker gets "
                    "dismissed. Pure arithmetic; no data leaves the process."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "budget": {
                            "type": "number",
                            "description": "Fraction of windows reviewed, (0, 1].",
                        },
                        "error_rate": {
                            "type": "number",
                            "description": "Fraction of windows that are wrong, [0, 1].",
                        },
                        "observed_capture": {
                            "type": "number",
                            "description": (
                                "Optional realised share of errors captured, to "
                                "score against the ceiling."
                            ),
                        },
                    },
                    "required": ["budget", "error_rate"],
                },
            ),
            handler=_budget_ceiling,
        ),
    ]
