# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""What a statistical tool's result lets an answer say, stated by code.

exp86's audits (rounds 6 and 7) found answers that broke the statistics the
tools had just applied: a looser alpha proposed after nothing certified, a
Bonferroni re-run after the prefix rule failed, an error rate offered for a
regression score with no threshold, a third dated map offered to settle which
of two is right, 69/300 worked out by hand; round 8's audit found "a
guaranteed-certifiable region" promised from a random plan. The tools' notes said so in
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

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from olmoearth_agent.analysis.output_contract import add_forbidden, add_must_state
from olmoearth_agent.analysis.raster_compare import MIN_INTERVAL_N, correlation_interval

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
#: Where, or in what pattern, two maps agree or disagree ("anywhere",
#: "nowhere", "one is high where the other is low", "large parts agree"),
#: read from one pooled correlation, which has no location (exp86 round 8).
SPATIAL_PATTERN_FROM_ONE_CORRELATION = "spatial_pattern_from_one_correlation"
#: That two maps do, or do not, co-vary ("do not agree at all", "independent",
#: "unrelated", "strongly agree") when the correlation's 95% interval holds
#: both no relation and a moderate one (round 8: r = -0.02 on 25 cells, where
#: a 12 x 12 grid of the same pair gives 0.50).
AGREEMENT_FROM_UNCERTAIN_CORRELATION = "agreement_from_uncertain_correlation"
#: A review set, margins, "most ambiguous" or "least certain" windows, or a
#: per-class review offered for a regression band that has no threshold.
REVIEW_SET_FOR_UNTHRESHOLDED_REGRESSION = "review_set_for_unthresholded_regression"
#: Labels or a reference for one date, or one reference plus another model
#: run, offered to say which of two differently dated maps is right.
ONE_REFERENCE_SETTLES_TWO_DATES = "one_reference_settles_two_dates"
#: An experiment's result applied to a case the tool says it does not cover
#: ("in comparable cases", "upstream evidence shows" for this pair).
EVIDENCE_OUTSIDE_ITS_SCOPE = "evidence_outside_its_scope"
#: A design, sample size or plan promised to certify a zone: a random design
#: makes certification possible, never certain.
CERTIFICATION_GUARANTEED = "certification_guaranteed"
#: A review set's windows called likely wrong ("most likely mislabeled", "the
#: likeliest spots for a wrong call", "probably errors"): the margin orders a
#: review, it is no probability of error (exp86 round 9, B8/cluster and
#: B2/studio).
MARGIN_AS_ERROR_PROBABILITY = "margin_as_error_probability"

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
    SPATIAL_PATTERN_FROM_ONE_CORRELATION,
    AGREEMENT_FROM_UNCERTAIN_CORRELATION,
    REVIEW_SET_FOR_UNTHRESHOLDED_REGRESSION,
    ONE_REFERENCE_SETTLES_TWO_DATES,
    EVIDENCE_OUTSIDE_ITS_SCOPE,
    CERTIFICATION_GUARANTEED,
    MARGIN_AS_ERROR_PROBABILITY,
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
    # the contract's own helpers: must_state keeps its limits (at most 3
    # sentences of at most 25 words) and each forbidden id is listed once
    sentences = list(must_state)
    if sentences:
        add_must_state(out, sentences)
    claims = list(forbidden_claims)
    if claims:
        add_forbidden(out, claims)
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


def _names(names: Sequence[str]) -> str:
    """``A``, ``A and B``, ``A, B and C``."""
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + f" and {names[-1]}"


def _split_text(split: Any) -> str | None:
    """``train/val split 0.75/0.25`` (``/test`` when it has a share), or ``None``."""
    if not isinstance(split, Mapping):
        return None
    parts: list[tuple[str, float]] = []
    for name in ("train", "val", "test"):
        value = split.get(f"{name}_prop")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if value or name != "test":  # a test share of 0 is no part
                parts.append((name, float(value)))
    if len(parts) < 2:
        return None
    return (
        "/".join(n for n, _ in parts) + " split " + "/".join(f"{v:g}" for _, v in parts)
    )


def labels_in_studio_fact(
    models: Iterable[Mapping[str, Any] | None],
) -> dict[str, Any] | None:
    """The ``labels_in_studio`` fact: the Studio models fine-tuned on a label field.

    exp86 round 9 (B3/studio run 1) answered "no ground-truth labels exist"
    while the run's own Studio model records showed both models fine-tuned on
    a label field of the user's project, with a 0.75/0.25 train/val split:
    the tools said only that no labels were used, and the model filled the
    rest from general knowledge. A model summary
    (:func:`olmoearth_agent.tools.sampling.summarize_model`) whose record
    names a label field is stated here: labels for the project may exist in
    Studio, and no tool of this run looked them up for this area (no agent
    tool reads Studio labels). No person's id or name is read or stated;
    ``label_field_id`` is the id of the label field.

    Returns
    -------
    dict or None
        ``id``, ``sentence``, ``models`` (``name``, ``model_id``,
        ``label_field_id`` and ``split`` of each), or ``None`` when no model
        names a label field.
    """
    trained = [m for m in models if m and m.get("trained_on_labels")]
    if not trained:
        return None
    names = [str(m.get("name") or m.get("model_id")) for m in trained]
    splits = {_split_text(m.get("split")) for m in trained}
    split = splits.pop() if len(splits) == 1 else None
    one = len(trained) == 1
    sentence = (
        f"{_names(names)} {'was' if one else 'were each'} fine-tuned in Studio on "
        f"a label field of {'its' if one else 'their'} project"
        + (f" ({split})" if split else "")
        + ", so labels for this project may exist in Studio; no tool of this run "
        "looked them up for this area."
    )
    return fact(
        "labels_in_studio",
        sentence,
        models=[
            {
                "name": m.get("name"),
                "model_id": m.get("model_id"),
                "label_field_id": m.get("label_field_id"),
                "split": m.get("split"),
            }
            for m in trained
        ],
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
        "needs a new plan with design='random', which makes a certified zone "
        "possible, not certain",
    )


#: What every step that names a random plan for a certified zone adds: exp86
#: round 8 (B5/files/2) read "a certified zone needs a new plan with
#: design='random'" as "a guaranteed-certifiable region", a precondition as
#: an outcome.
POSSIBLE_NOT_CERTAIN = (
    "A random design makes a certified zone possible, not certain: "
    "olmoearth_certify_zone may certify nothing."
)


def certification_guaranteed(detail: str | None = None) -> dict[str, str]:
    """No design, budget or plan promises a certified zone.

    exp86 round 8 (B5/files/2) offered "a guaranteed-certifiable region"
    from a random plan. The package certifies a zone only where the labels
    drawn in it hold few enough errors for its exact test, so a random
    design can certify nothing: on that map's own 300-label random design
    (F3) every one of 18 levels failed. ``detail`` says what this result
    adds (the design, the budget, the outcome).
    """
    return forbidden(
        CERTIFICATION_GUARANTEED,
        "a random design makes a certified zone possible, never certain: "
        "olmoearth_certify_zone certifies a zone only where the labels drawn "
        "in it hold few enough errors for its exact test, so it may certify "
        "none, and no design, sample size, budget or plan can promise a "
        "certified (or 'certifiable') region, or a zone of any size"
        + (f"; {detail}" if detail else ""),
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


def _bands_named(bands: list[tuple[str | None, tuple[float, float] | None]]) -> str:
    """``'sample_number' (declared range [0.2, 1.2])``, for each band."""
    return "; ".join(
        f"{name!r} (declared range "
        f"{f'[{rng[0]:g}, {rng[1]:g}]' if rng else 'none'})"
        for name, rng in bands
    )


def unthresholded_regression(
    bands: list[tuple[str | None, tuple[float, float] | None]],
) -> dict[str, str]:
    """A regression score with no decision threshold has no error rate."""
    return forbidden(
        ERROR_RATE_FOR_UNTHRESHOLDED_REGRESSION,
        f"{_bands_named(bands)}: a per-pixel regression value with no decision threshold "
        "has no error rate, accuracy, confusion matrix or other "
        "classification metric; one exists only once a threshold is named "
        "(olmoearth_plan_label_sample takes 'threshold')",
    )


def review_set_for_unthresholded_regression(
    bands: list[tuple[str | None, tuple[float, float] | None]],
) -> dict[str, str]:
    """A regression band with no decision threshold has no margin to review by.

    exp86 round 8 (brief 3 on Studio) offered a "per-class review set" and
    "the most-ambiguous (lowest-margin) windows" of KarstNumber, a band
    declared 0.2 to 1.2 with no threshold, beside the error rate the
    comparison already forbade (:func:`unthresholded_regression`). The reason
    is scoped to a margin- or threshold-based review set of the named band:
    another band of the run, or a ranking that needs no threshold (a group's
    disagreement, an ensemble's spread), is not what it forbids.
    """
    return forbidden(
        REVIEW_SET_FOR_UNTHRESHOLDED_REGRESSION,
        f"{_bands_named(bands)}: a margin-based review set "
        "(olmoearth_review_set_from_result) ranks a band's windows by their "
        "distance from a decision threshold, and this band has none: it has no "
        "margins, no 'most ambiguous' or lowest-margin windows and no per-class "
        "review; a review set of this band needs a threshold for it "
        "(olmoearth_review_set_from_result takes 'threshold')",
    )


def labelling_low_confidence_only() -> dict[str, str]:
    """A sample of the low-confidence windows only is not a sample of the map.

    Emitted by the comparisons, whose answers point to labelling to settle
    which map is right: exp86 round 8 (brief 3 on Studio, run 3) offered "a
    targeted labeling sample from the lower-confidence windows of each map"
    and "defensible error rates per map" from it. The plan tool's designs
    all draw from every window (``confidence`` and ``proportional`` stratify
    all of them by margin; ``random`` draws from all of them alike).
    """
    return forbidden(
        SUBSET_LABELLING_SUFFICIENT,
        "labels drawn only from the low-confidence, least certain or most "
        "ambiguous windows (or from a review set) are not a sample of the map: "
        "an error rate over them describes those windows and overstates the "
        "map's, so it is no defensible error rate for either map; each of "
        "olmoearth_plan_label_sample's designs ('confidence', 'proportional', "
        "'random') draws from every window of the map, and its estimate needs "
        "every drawn window labelled",
    )


#: Stated with a comparison of maps of different or overlapping times (the
#: exp86 round 7 build's dates sentence, merged): exp86 round 8 offered labels
#: for "either date" (brief 3 on the cluster), or for one date "plus a
#: date-matched second inference" (brief 7 on files), to settle which is right.
DATED_MAPS_MUST_STATE = (
    "Maps of different times may differ by real change; labels for one date "
    "grade only that date's map; each map needs its own date's reference."
)


def one_reference_settles_two_dates(
    a: str | None = None,
    b: str | None = None,
    labels: str | None = None,
    *,
    partly: bool = False,
) -> dict[str, str]:
    """Labels for one date, or one reference and another run, settle nothing across dates.

    ``a`` and ``b`` are the maps' dates, ``labels`` the labels' date when
    given; ``partly`` is set when only one map's date is known. exp86 round 8:
    "if you have reference labels for either date ... I can grade which map is
    right" (brief 3 on the cluster), "or for at least one, plus a date-matched
    second inference" (brief 7 on files).
    """
    if partly:
        head = (
            "only one map's date is known, so labels or a reference for that "
            "date cannot grade the other map"
        )
    else:
        when = f" ({a} and {b})" if a and b else ""
        head = (
            f"the maps describe different times{when}: labels or a reference for "
            "one date grade only the map of that date, and would count the other "
            "wrong wherever the ground changed"
        )
    dated = f" (labels dated {labels} grade only a map of that date)" if labels else ""
    return forbidden(
        ONE_REFERENCE_SETTLES_TWO_DATES,
        f"{head}; one reference plus another model run adds an unlabelled map "
        "and settles nothing; to say which map is right where they differ, each "
        f"map needs a reference of its own date{dated}",
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


#: What a comparison's lack of labels is, and is not: none were given to it;
#: that is no finding that none exist.
NO_LABELS_GIVEN = (
    "no labels were given to this comparison, which takes none: nothing here "
    "says which map is right, more accurate or better, and nothing here says "
    "whether labels for these maps exist (none was looked up)"
)


def winner_without_labels(detail: str | None = None) -> dict[str, str]:
    """No labels are given to a comparison, so no map is shown right or better.

    The comparisons take no labels at all (``olmoearth_compare_results`` and
    ``olmoearth_compare_review``), so this is emitted on every comparison
    they return, whatever the dates or properties; ``detail`` adds what the
    comparison does say. The reason says the labels were not given to this
    comparison, never that none exist: exp86 round 9 (B3/studio run 1) read
    "no labels were used" as "no ground-truth labels exist", of two Studio
    models each fine-tuned on a label field of the user's project.
    """
    return forbidden(
        WINNER_WITHOUT_LABELS, NO_LABELS_GIVEN + (f"; {detail}" if detail else "")
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


# --------------------------------------------------------------------------- claims about a correlation

#: A correlation's 95% interval that holds 0 and reaches this far from it on
#: either side holds both no relation and a moderate one (exp86 round 8: r =
#: -0.0172 over 25 cells, interval -0.41 to 0.38).
MODERATE_CORRELATION = 0.3

#: Stated whenever a comparison returns a correlation. exp86 round 8 (brief 3
#: on Studio) read one pooled r as a place: "do not agree spatially at all",
#: "do not rise and fall together anywhere", "one is high where the other is
#: indifferent, and vice versa".
CORRELATION_MUST_STATE = (
    "One correlation says nothing about where two maps agree or differ."
)

#: The same, when a returned correlation's interval holds 0 (or, below four
#: cells, does not exist): the sample cannot say whether the maps co-vary.
UNCERTAIN_CORRELATION_MUST_STATE = (
    "One correlation says nothing about where two maps agree or differ, and one "
    "whose 95% interval spans zero cannot say whether they co-vary."
)


def correlation_fact(
    r: float, n: int, *, pair: tuple[str, str] | None = None
) -> dict[str, Any]:
    """The ``correlation`` fact: r, the cells it pools and its 95% interval, stated.

    The interval is Fisher's z (:func:`~olmoearth_agent.analysis.raster_compare.
    correlation_interval`); below four cells there is none. The sentence says
    that the sample cannot say whether the maps co-vary when the interval holds
    0, and the sign otherwise; it always says that a correlation has no
    location. exp86 round 8 read r = -0.0172 over 25 cells as "do not agree at
    all" where the interval was -0.41 to 0.38. ``pair`` names the two results
    when a comparison returns several correlations (a group, a series).

    Returns
    -------
    dict
        ``id`` "correlation", ``sentence``, ``r``, ``n``, ``ci_low`` and
        ``ci_high`` (2 decimals; ``None`` without an interval), ``level``,
        ``method``, ``co_varies`` ("unknown", "positive" or "negative"),
        ``holds_moderate`` (the interval holds 0 and reaches
        :data:`MODERATE_CORRELATION`, or there is none) and, with ``pair``,
        ``result_id_a`` and ``result_id_b``.
    """
    interval = correlation_interval(r, n)
    who = f"Between results {pair[0]} and {pair[1]}, the" if pair else "The"
    head = f"{who} correlation is {r:g} over {n:,} cells"
    where = "; a correlation says nothing about where the maps agree or differ."
    low = high = None
    if interval is None:
        co_varies, moderate = "unknown", True
        body = (
            f"; with fewer than {MIN_INTERVAL_N} cells it has no interval, so "
            "this sample cannot say whether the maps co-vary"
        )
    else:
        lo, hi = interval
        low, high = round(lo, 2), round(hi, 2)
        stated = f", 95% interval {low:.2f} to {high:.2f} (Fisher's z)"
        if lo <= 0.0 <= hi:
            co_varies = "unknown"
            moderate = max(abs(lo), abs(hi)) >= MODERATE_CORRELATION
            holds = (
                "both no relation and a moderate one" if moderate else "0 (no relation)"
            )
            body = (
                f"{stated}: the interval holds {holds}, so this sample cannot "
                "say whether the maps co-vary"
            )
        else:
            moderate = False
            co_varies = "positive" if lo > 0.0 else "negative"
            sign = (
                "the maps' values tend to rise and fall together across the "
                "sampled cells"
                if co_varies == "positive"
                else "the maps' values tend to move in opposite directions across "
                "the sampled cells"
            )
            body = f"{stated}: {sign}"
    out = fact(
        "correlation",
        head + body + where,
        r=r,
        n=n,
        ci_low=low,
        ci_high=high,
        level=0.95,
        method="Fisher's z",
        co_varies=co_varies,
        holds_moderate=moderate,
    )
    if pair:
        out["result_id_a"], out["result_id_b"] = pair
    return out


def spatial_pattern_from_one_correlation() -> dict[str, str]:
    """A pooled correlation has no location, so it places no agreement or difference."""
    return forbidden(
        SPATIAL_PATTERN_FROM_ONE_CORRELATION,
        "a correlation is one number pooled over all the compared cells and has "
        "no location: it cannot say where the maps agree or differ, that they "
        "agree or differ anywhere or nowhere, that large parts agree while the "
        "rest differs, or that one is high where the other is low; only a "
        "per-window value, which a correlation is not, can place them",
    )


#: Most correlations an ``agreement_from_uncertain_correlation`` reason names.
_UNCERTAIN_NAMED = 4


def agreement_from_uncertain_correlation(
    facts: Sequence[dict[str, Any]],
) -> dict[str, str]:
    """No agreement, or its absence, from correlations whose interval holds both.

    ``facts`` are the ``correlation`` facts whose ``holds_moderate`` is true: an
    interval that holds 0 and reaches :data:`MODERATE_CORRELATION` on a side,
    or none at all.
    """
    parts = []
    for item in facts[:_UNCERTAIN_NAMED]:
        who = (
            f"results {item['result_id_a']} and {item['result_id_b']}: "
            if item.get("result_id_a")
            else ""
        )
        if item.get("ci_low") is None:
            parts.append(
                f"{who}r = {item['r']:g} over {item['n']} cells, too few for an "
                "interval"
            )
        else:
            parts.append(
                f"{who}r = {item['r']:g} over {item['n']} cells, 95% interval "
                f"{item['ci_low']:.2f} to {item['ci_high']:.2f}"
            )
    more = len(facts) - _UNCERTAIN_NAMED
    listed = "; ".join(parts) + (f"; and {more} more" if more > 0 else "")
    return forbidden(
        AGREEMENT_FROM_UNCERTAIN_CORRELATION,
        f"{listed}: an interval that holds both no relation and a moderate one "
        "cannot say whether the maps co-vary, so neither 'they do not agree at "
        "all', 'independent', 'unrelated' or 'effectively zero' nor 'they "
        "agree' follows from it; more cells narrow the interval",
    )
