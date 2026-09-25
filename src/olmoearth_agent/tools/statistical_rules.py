# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""What a statistical tool's result lets an answer say, stated by code.

exp86's audits (rounds 6 and 7) found answers that broke the statistics the
tools had just applied: a looser alpha proposed after nothing certified, a
Bonferroni re-run after the prefix rule failed, an error rate offered for a
regression score with no threshold, a third dated map offered to settle which
of two is right, 69/300 worked out by hand. The tools' notes said so in
prose, and the prose was not followed. This module gives the tools the
shared output contract's three keys, so a harness can check an answer
against them:

- ``facts``: ``[{"id", "sentence", ...fields}]``, a fact the answer may
  repeat, with its sentence written by code (the answer never derives it);
- ``must_state``: short sentences the answer must convey whenever it reports
  the result (a scope limit);
- ``forbidden_claims``: ``[{"id", "why"}]``, claims the answer must not make
  about the result, each id at most once per result. Every id is one of the
  contract's fixed ids (:data:`FIXED_IDS`).

A result carries them as top-level keys; a refusal raised as an error
(:func:`refusal`) carries them on the registry's failed envelope.

The builders here are shared by the estimation tools
(:mod:`olmoearth_agent.tools.estimation`) and the comparisons
(:mod:`olmoearth_agent.tools.compare`, ``olmoearth_compare_review``).
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

# --------------------------------------------------------------------------- the contract's fixed ids

POST_HOC_ALPHA = "post_hoc_alpha"
RULE_SWITCH_AFTER_FAILURE = "rule_switch_after_failure"
CERTIFY_FROM_NONRANDOM_DESIGN = "certify_from_nonrandom_design"
ERROR_RATE_WITHOUT_LABELS = "error_rate_without_labels"
ERROR_RATE_FOR_UNTHRESHOLDED_REGRESSION = "error_rate_for_unthresholded_regression"
SUBSET_LABELLING_SUFFICIENT = "subset_labelling_sufficient"
WINNER_WITHOUT_LABELS = "winner_without_labels"
COMBINED_STATISTIC_ACROSS_PROPERTIES = "combined_statistic_across_properties"
ANOTHER_DATE_SETTLES_IT = "another_date_settles_it"
#: The simple-random interval put in place of a stratified design's own.
SIMPLE_RANDOM_INTERVAL = "simple_random_interval_for_stratified_design"

#: The contract's fixed ids: every forbidden claim a tool emits is one of them.
FIXED_IDS = (
    POST_HOC_ALPHA,
    RULE_SWITCH_AFTER_FAILURE,
    CERTIFY_FROM_NONRANDOM_DESIGN,
    ERROR_RATE_WITHOUT_LABELS,
    ERROR_RATE_FOR_UNTHRESHOLDED_REGRESSION,
    SUBSET_LABELLING_SUFFICIENT,
    WINNER_WITHOUT_LABELS,
    COMBINED_STATISTIC_ACROSS_PROPERTIES,
    ANOTHER_DATE_SETTLES_IT,
    SIMPLE_RANDOM_INTERVAL,
)


def forbidden(claim_id: str, why: str) -> dict[str, str]:
    """One ``forbidden_claims`` entry."""
    return {"id": claim_id, "why": why}


def fact(fact_id: str, sentence: str, **fields: Any) -> dict[str, Any]:
    """One ``facts`` entry: its id, its sentence, then its fields."""
    return {"id": fact_id, "sentence": sentence, **fields}


def add_contract(
    out: dict[str, Any],
    *,
    facts: Iterable[dict[str, Any]] = (),
    must_state: Iterable[str] = (),
    forbidden_claims: Iterable[dict[str, str]] = (),
) -> dict[str, Any]:
    """Append to ``out``'s contract keys, creating each only when it gets an entry.

    Entries already there are kept, so two parts of a tool can each add
    theirs; a forbidden claim whose id is already listed is not repeated,
    and neither is a sentence already in ``must_state``.
    """
    new_facts = list(facts)
    if new_facts:
        out.setdefault("facts", []).extend(new_facts)
    for sentence in must_state:
        stated = out.setdefault("must_state", [])
        if sentence not in stated:
            stated.append(sentence)
    for claim in forbidden_claims:
        listed = out.setdefault("forbidden_claims", [])
        if all(c.get("id") != claim["id"] for c in listed):
            listed.append(claim)
    return out


def refusal(
    message: str,
    *,
    facts: Iterable[dict[str, Any]] = (),
    must_state: Iterable[str] = (),
    forbidden_claims: Iterable[dict[str, str]] = (),
) -> ValueError:
    """A tool's refusal as a ``ValueError`` that also carries contract keys.

    Raised from a handler it stays an error (``ok`` False, the same
    ``error`` text, the same repeated-failure count), and the registry puts
    its ``facts``, ``must_state`` and ``forbidden_claims`` on the failed
    envelope beside the error (:meth:`ToolRegistry.dispatch`).
    """
    err = ValueError(message)
    err.contract = add_contract(  # type: ignore[attr-defined]
        {}, facts=facts, must_state=must_state, forbidden_claims=forbidden_claims
    )
    return err


def percent(value: float | None) -> str:
    """A fraction as a percent: one decimal from 1%, two significant figures below."""
    if value is None:
        return "n/a"
    pct = 100.0 * float(value)
    if pct == 0.0 or abs(pct) >= 1.0:
        return f"{pct:.1f}%"
    return f"{pct:.2g}%"


# --------------------------------------------------------------------------- claims about labels and designs


def error_rate_without_labels() -> dict[str, str]:
    """A plan holds no labels, so it has no error rate yet."""
    return forbidden(
        ERROR_RATE_WITHOUT_LABELS,
        "the plan holds no labels: no error rate, accuracy or interval exists "
        "until the reviewer has labelled its windows, so none may be stated or "
        "guessed from the scores (a confidence score is not a probability of "
        "error)",
    )


def subset_labelling_sufficient(design: str) -> dict[str, str]:
    """Only a stratified sheet's full set of rows is its design.

    The package draws a stratified sample stratum by stratum, the least
    confident (stratum 0) first, and the sheet keeps that order, so its first
    rows are the least confident windows only. Emitted only for a design with
    at least two strata: a random design's sheet is in random order, and a
    prefix of it is a smaller random sample (:func:`prefix_is_random_sample`).
    """
    return forbidden(
        SUBSET_LABELLING_SUFFICIENT,
        "label every window in the sheet; labelling only its first rows biases "
        f"the estimate: the sheet lists this {design} design's strata in turn, "
        "the least confident (stratum 0) first, each stratum's windows in random "
        "order, so its first rows are the least confident windows only, a rate "
        "over them describes those windows and not the map, and the strata "
        "after them get no labels; olmoearth_estimate_map_error refuses a design "
        "with any window unlabelled",
    )


def prefix_is_random_sample(n_rows: int, route: str | None) -> dict[str, Any]:
    """The ``prefix_is_random_sample`` fact: a random sheet's first rows are a random sample.

    The package draws a random design without replacement in a random order
    (``Generator.choice``, shuffled) and the sheet keeps that order, so its
    first ``k`` rows, ``k`` fixed before any label is seen, are a simple
    random sample of size ``k``: the estimate stays unbiased and its interval
    widens. ``route`` is the scores argument (``"scores"`` or
    ``"scores_path"``) that olmoearth_estimate_map_error takes with those rows
    as ``window_indices``; with ``design_path`` it needs every row labelled,
    and a Studio result sampled by the plan itself has no scores to pass.
    """
    sentence = (
        f"The sheet's {n_rows} rows are in the package's random draw order, so "
        "labelling only its first k rows, with k fixed before any label is seen, "
        "is a smaller random sample: the estimate stays unbiased and its "
        "interval widens."
    )
    if route:
        sentence += (
            " olmoearth_estimate_map_error takes those rows as window_indices "
            f"with the same {route}; with design_path it needs every row labelled."
        )
    else:
        sentence += (
            " olmoearth_estimate_map_error with design_path needs every row "
            "labelled, so these tools estimate only the full sheet."
        )
    return fact(
        "prefix_is_random_sample",
        sentence,
        n_rows=n_rows,
        estimate_prefix_with=f"window_indices and {route}" if route else None,
    )


def simple_random_interval(method: str | None) -> dict[str, str]:
    """A stratified design's interval is its own, never p +/- 1.96 sqrt(p(1-p)/n)."""
    return forbidden(
        SIMPLE_RANDOM_INTERVAL,
        "this design is stratified, so a simple-random-sample interval "
        "(p +/- 1.96 sqrt(p(1 - p)/n)) does not apply to it, nor any interval "
        "worked out by hand; the interval is [low, high] by the method returned"
        + (f" ({method})" if method else ""),
    )


def certify_from_nonrandom_design(design: str | None) -> dict[str, str]:
    """No zone is certified from a stratified design, by the tool or by hand."""
    return forbidden(
        CERTIFY_FROM_NONRANDOM_DESIGN,
        f"this design is {design!r}, not 'random': a certified zone's guarantee "
        "needs the labels inside each candidate zone to be a random sample of "
        "it, which a stratified draw is not, so no zone can be certified from "
        "these labels (olmoearth_certify_zone refuses them); certification "
        "needs a new plan with design='random'",
    )


def post_hoc_alpha(alpha: float | None) -> dict[str, str]:
    """alpha is fixed before the labels are seen; a later one certifies nothing."""
    this = f"alpha={alpha:g} was fixed for this call; " if alpha is not None else ""
    return forbidden(
        POST_HOC_ALPHA,
        f"{this}the guarantee holds only for an alpha fixed before the labels "
        "were seen. Proposing or evaluating another alpha after seeing these "
        "bounds (e.g. that a looser alpha would certify some share of the map) "
        "is not a certification: another alpha tests other levels, a level "
        "whose upper_bound is below an alpha is not thereby accepted, and "
        "trying alphas until one certifies is not covered",
    )


def rule_switch_after_failure(
    *,
    certified: bool | None,
    levels: list[dict[str, Any]] | None,
    delta: float | None,
    rule: str | None,
) -> dict[str, str]:
    """The rule is fixed with alpha; a second rule after the first is a second look.

    With the levels tested (the package lists them smallest zone first), the
    reason says how each rule reads their ``p_value``: prefix accepts levels
    from the smallest zone upward while ``p_value <= delta`` and stops at the
    first failure, so the smallest zone's p_value decides whether it accepts
    any; bonferroni accepts any level with ``p_value <= delta / J``, ``J`` the
    number of levels, so the smallest p_value decides. When neither accepts a
    level the reason says so, and the other rule's outcome is never guessed.
    """
    if certified:
        second = "re-running certification under another rule to get a larger zone"
    elif certified is False:
        second = (
            "re-running certification under another rule after this one "
            "certified nothing"
        )
    else:
        second = "choosing or changing the rule after the labels are seen"
    which = f" ({rule!r} here)" if rule else ""
    head = (
        f"the rule{which} is fixed with alpha before the labels are seen; "
        f"{second} is a second look that the guarantee does not cover"
    )
    pvals = [
        float(lv["p_value"])
        for lv in levels or []
        if isinstance(lv.get("p_value"), (int, float))
    ]
    if levels is not None and not levels:
        tail = "; no level was tested, so no rule can certify with these labels"
    elif pvals and delta is not None and len(pvals) == len(levels or []):
        n = len(pvals)
        first, smallest = pvals[0], min(pvals)
        tail = (
            "; prefix accepts levels from the smallest zone upward while "
            f"p_value <= delta = {delta:g} and stops at the first failure (the "
            f"smallest zone's p_value is {first:.3g}); bonferroni accepts any "
            f"level with p_value <= delta/{n} = {delta / n:.3g} (the smallest "
            f"p_value of the {n} levels tested is {smallest:.3g})"
        )
        if first > delta and smallest > delta / n:
            tail += ", so no level passes under either rule"
    else:
        tail = ""
    return forbidden(RULE_SWITCH_AFTER_FAILURE, head + tail)


# --------------------------------------------------------------------------- claims about comparisons


def unthresholded_regression(
    bands: list[tuple[str | None, tuple[float, float] | None]],
) -> dict[str, str]:
    """A regression score with no decision threshold has no error rate."""
    named = "; ".join(
        f"{name!r} (declared range "
        f"{f'[{rng[0]:g}, {rng[1]:g}]' if rng else 'none'})"
        for name, rng in bands
    )
    return forbidden(
        ERROR_RATE_FOR_UNTHRESHOLDED_REGRESSION,
        f"{named}: a per-pixel regression value with no decision threshold "
        "has no error rate, accuracy, confusion matrix or other "
        "classification metric; one exists only once a threshold is named "
        "(olmoearth_plan_label_sample takes 'threshold')",
    )


def different_properties(names: Iterable[str | None]) -> list[dict[str, str]]:
    """Two properties: no statistic combines them, and no side wins without labels."""
    distinct = sorted({n for n in names if n})
    return [
        forbidden(
            COMBINED_STATISTIC_ACROSS_PROPERTIES,
            f"the results measure different properties ({distinct}): a "
            "difference, ratio, average, RMSE, agreement share or any other "
            "statistic that combines the two quantities means nothing, and "
            "neither reads higher or lower than the other; only each map's own "
            "mean and the correlation are returned",
        ),
        winner_without_labels(
            "the correlation says only whether the two rise and fall together"
        ),
    ]


def winner_without_labels(detail: str | None = None) -> dict[str, str]:
    """No labels are used by a comparison, so no map is shown right or better.

    The comparisons take no labels at all (``olmoearth_compare_results`` and
    ``olmoearth_compare_review``), so this is emitted on every comparison
    they return, whatever the dates or properties; ``detail`` adds what the
    comparison does say.
    """
    return forbidden(
        WINNER_WITHOUT_LABELS,
        "no labels were used: nothing here says which map is right, more "
        "accurate or better" + (f"; {detail}" if detail else ""),
    )


#: What a review comparison's differing windows do not say.
MORE_CONFIDENT_IS_NOT_RIGHT = (
    "neither map is shown right where they differ, and the more confident side "
    "is not thereby the right one"
)


def across_dates(dates: dict[str, Any]) -> list[dict[str, str]]:
    """Maps of different or overlapping periods: another date settles nothing; no side wins."""
    a, b = dates.get("a"), dates.get("b")
    when = f" ({a} and {b})" if a and b else ""
    labels = dates.get("labels")
    return [
        forbidden(
            ANOTHER_DATE_SETTLES_IT,
            f"the maps describe different or overlapping periods{when}: a window "
            "where they differ either changed on the ground or is wrong in one "
            "map, and another unlabelled map, of any date, cannot tell which; "
            "only a reference dated to each map can",
        ),
        winner_without_labels(
            MORE_CONFIDENT_IS_NOT_RIGHT
            + (
                f"; labels dated {labels} would measure which map matches the "
                "ground at that date, counting the other wrong wherever the "
                "ground changed"
                if labels
                else ""
            )
        ),
    ]
