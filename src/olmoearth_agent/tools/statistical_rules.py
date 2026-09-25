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
  about the result. The ids below are the contract's fixed ids, except
  :data:`SIMPLE_RANDOM_INTERVAL`, which only the answer's reader displays.

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

#: Not one of the contract's fixed ids (none covers it): the simple-random
#: interval put in place of a stratified design's own.
SIMPLE_RANDOM_INTERVAL = "simple_random_interval_for_stratified_design"


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
    """Labelling only part of a plan's sheet is not the plan."""
    order = (
        "the sheet lists a stratified plan stratum by stratum, least confident "
        "first, so its first rows are one stratum"
        if design != "random"
        else "the design is every drawn window, not its first rows"
    )
    return forbidden(
        SUBSET_LABELLING_SUFFICIENT,
        "label every window in the sheet; labelling only the first rows biases "
        f"the estimate ({order}), and olmoearth_estimate_map_error refuses a "
        "design with any window unlabelled",
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

    With the levels tested, the reason gives their smallest ``p_value`` and
    what each rule needs of it (prefix: ``delta``; bonferroni: ``delta`` over
    the number of levels), so the other rule's outcome is not guessed: when
    the smallest p_value is above ``delta``, no level passes under either.
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
    elif pvals and delta is not None:
        n = len(levels or [])
        smallest = min(pvals)
        tail = (
            f"; the smallest p_value of the {n} levels tested is {smallest:.3g}, "
            f"and prefix accepts a level at p_value <= delta = {delta:g}, "
            f"bonferroni at <= delta/{n} = {delta / n:.3g}"
        )
        if smallest > delta:
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
        forbidden(
            WINNER_WITHOUT_LABELS,
            "no labels were used: nothing here says which map is right, more "
            "accurate or better; the correlation says only whether the two "
            "rise and fall together",
        ),
    ]


def across_dates(dates: dict[str, Any]) -> list[dict[str, str]]:
    """Two maps of different times: another date settles nothing; no side wins."""
    a, b = dates.get("a"), dates.get("b")
    when = f" ({a} and {b})" if a and b else ""
    labels = dates.get("labels")
    return [
        forbidden(
            ANOTHER_DATE_SETTLES_IT,
            f"the maps describe different times{when}: a window where they "
            "differ either changed on the ground or is wrong in one map, and "
            "another unlabelled map, of any date, cannot tell which; only a "
            "reference dated to each map can",
        ),
        forbidden(
            WINNER_WITHOUT_LABELS,
            "no labels were used: neither map is shown right where they "
            "differ, and the more confident side is not thereby the right one"
            + (
                f"; labels dated {labels} could grade only a map of that date"
                if labels
                else ""
            ),
        ),
    ]
