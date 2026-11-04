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
#: A map named more accurate (or better, or right) without the test or
#: interval of the which-map grading that names it and its number of labels,
#: any map where the grading names neither, or the map it does not name.
MORE_ACCURATE_WITHOUT_ITS_TEST = "more_accurate_without_its_test"
#: A certified, certifiable or trusted zone, or a certificate, offered or
#: stated for a class map without top-k scores: it has no confidence to rank
#: its windows by, so no zone of it can be certified, whatever the design,
#: budget or alpha.
ZONE_FOR_A_CLASS_MAP = "zone_for_a_class_map"
#: An error rate, accuracy or share wrong stated in the answer for a map whose
#: error rate the request did not estimate (olmoearth_estimate_map_error or
#: olmoearth_certify_zone on labels of the map); the stakeholder report
#: refuses a title or result typed with one.
ACCURACY_NOT_MEASURED = "accuracy_not_measured"
#: A disagreement design, plan, sample or budget offered for two maps of
#: different times (agent trial 12, B3/cluster run 5 and B7/files run 11):
#: labels where they differ cannot say which is more accurate across dates.
TWO_MAP_DESIGN_ACROSS_DATES = "two_map_design_across_dates"
#: The whole-map error rate set against alpha to explain or predict that no
#: zone certified, or to say every part of the map is above alpha (agent
#: trial 12, B6/files runs 8, 9 and 13): it describes the whole map, not its
#: most confident part.
WHOLE_MAP_RATE_FOR_A_ZONE = "whole_map_rate_for_a_zone"
#: Two maps' differences called mostly edges or boundaries while most
#: differing windows lie in large 8-connected patches (agent trial 12,
#: B3/cluster run 25): a boundary share counts neighbours, not regions. The
#: comparison emits it (``analysis.review_set.mostly_edges``).
MOSTLY_EDGES = "mostly_edges"

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
    MORE_ACCURATE_WITHOUT_ITS_TEST,
    ZONE_FOR_A_CLASS_MAP,
    ACCURACY_NOT_MEASURED,
    TWO_MAP_DESIGN_ACROSS_DATES,
    WHOLE_MAP_RATE_FOR_A_ZONE,
    MOSTLY_EDGES,
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


#: What a stakeholder report (skill #4) of a request that estimated no error
#: rate states. A request is one run of the agent: on the web UI, one message.
NOT_MEASURED_MUST_STATE = (
    "No error rate of this map has been estimated in this request; "
    "olmoearth_estimate_map_error estimates one from labels on a planned sample."
)
#: The same, for a request that graded only which of two maps is more
#: accurate: that grading says nothing of either map's own accuracy
#: (``WHICH_MAP_NOTE``).
NEITHER_MAP_MEASURED_MUST_STATE = (
    "Neither map's own accuracy has been measured in this request; the "
    "which-map grading says which is more accurate, not how accurate either is."
)


def accuracy_not_measured(*, which_map: bool = False) -> dict[str, str]:
    """No error rate, accuracy or interval for a map this request did not measure.

    Emitted by the stakeholder report (``olmoearth_case_narrative``) when no
    call of the request estimated the map's error rate from labels
    (``olmoearth_estimate_map_error``, ``olmoearth_certify_zone``): an
    accuracy figure in the answer has no tool behind it, and the report
    refuses a title or result typed with one. ``harness/checks.py`` reads it
    (:func:`~olmoearth_agent.harness.checks.states_accuracy_figure`), and not
    once the run holds a measured error rate (a ``whole_map_estimate`` fact).
    ``which_map`` is set when the request graded which of two maps is more
    accurate, which says neither map's own accuracy.
    """
    graded = (
        "; the which-map grading of this request says which map is more "
        "accurate, not how accurate either is"
        if which_map
        else ""
    )
    return forbidden(
        ACCURACY_NOT_MEASURED,
        "this request estimated no error rate: no call of it graded labels of "
        "the map (olmoearth_estimate_map_error, olmoearth_certify_zone), so no "
        "error rate, accuracy, share wrong or interval may be stated for the map "
        "in the answer; the report's accuracy section says none was estimated "
        "(on the web UI each message is a request of its own)" + graded,
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


def _model_split_text(model: Mapping[str, Any]) -> str | None:
    """A trained model's split in words, as far as its record says it.

    The record's proportions as the split (:func:`_split_text`), unless it
    also names a split field (``split_field_named``): then the proportions,
    a test share of 0 kept, are said to be recorded beside it, and which of
    the two training followed is not recorded. The kelp model's record names
    both, train/val/test 0.75/0.25/0, while its split field tags 557 of its
    1056 annotations test (live3, 31 October 2026): its proportions are no
    statement of the split training used.

    Examples
    --------
    >>> _model_split_text({"split": {"train_prop": 0.75, "val_prop": 0.25}})
    'train/val split 0.75/0.25'
    """
    split = model.get("split")
    if not model.get("split_field_named"):
        return _split_text(split)
    parts: list[tuple[str, float]] = []
    if isinstance(split, Mapping):
        for name in ("train", "val", "test"):
            value = split.get(f"{name}_prop")
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                parts.append((name, float(value)))
    if not parts:
        return "a split field recorded; whether training followed it is not recorded"
    names = "/".join(n for n, _ in parts)
    shares = "/".join(f"{v:g}" for _, v in parts)
    return (
        f"split proportions {names} {shares} recorded beside a split field; "
        "which of the two training followed is not recorded"
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
    Studio, ``olmoearth_studio_label_inventory`` counts those the model's
    training selector selects, and Studio records no random design over a
    map for them, so none can be taken as a random sample of a map (as the
    inventory says, ``studio_labels.NOT_A_SAMPLE``; whether the train/val
    split was drawn at random is not stated, either way). "A map" holds for
    every caller: one model's dated results in a comparison are several
    maps, and a search lists models, not maps. The sentence holds
    whether or not the run called the inventory (live3, 31 Oct 2026, called
    it in the run that stated this fact, whose sentence then said no tool of
    the run had looked the labels up). A record that names a split field
    beside its proportions has them stated as recorded beside it, not as the
    training split (:func:`_model_split_text`): the inventory counts the
    annotations by that field. No person's id or name is read or stated;
    ``label_field_id`` is the id of the label field.

    Returns
    -------
    dict or None
        ``id``, ``sentence``, ``models`` (``name``, ``model_id``,
        ``label_field_id``, ``split`` and ``split_field_named`` of each), or
        ``None`` when no model names a label field.
    """
    trained = [m for m in models if m and m.get("trained_on_labels")]
    if not trained:
        return None
    names = [str(m.get("name") or m.get("model_id")) for m in trained]
    splits = {_model_split_text(m) for m in trained}
    split = splits.pop() if len(splits) == 1 else None
    one = len(trained) == 1
    sentence = (
        f"{_names(names)} {'was' if one else 'were each'} fine-tuned in Studio on "
        f"a label field of {'its' if one else 'their'} project"
        + (f" ({split})" if split else "")
        + ", so labels for this project may exist in Studio; "
        f"olmoearth_studio_label_inventory counts those {'its' if one else 'their'} "
        f"training {'selector selects' if one else 'selectors select'}, and Studio "
        "records no random design over a map for them, so none can be taken as a "
        "random sample of a map."
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
                "split_field_named": m.get("split_field_named"),
            }
            for m in trained
        ],
    )


#: When the first rows of a random sheet stand in for it: a stop that does not
#: depend on the labels already read. A stop decided by what they show (the
#: errors look rare) is optional stopping, and the prefix is then no random
#: sample of a size fixed before the labels (the step 11 review).
STOP_NOT_ON_THE_LABELS = (
    "for a reason that does not depend on the labels already read (time, "
    "labour), never because of what they show"
)


def _design_file_routes(
    design: str, *, certificate: bool, class_map: bool
) -> tuple[str, str]:
    """The tools that read a plan's design file, and what its first rows lack.

    olmoearth_estimate_map_error reads every plan's design file;
    olmoearth_certify_zone only a random design's of a map with confidence
    (it refuses any other design and a class map); olmoearth_write_certificate
    only a plan drawn with ``seed_from_map`` (``certificate``; it refuses any
    other as ``NOT_FROM_MAP``). The step 11 review found every random-order
    plan, a plain random one and a single-stratum confidence one included,
    told the three tools needed its rows, which offered it routes the tools
    refuse.

    Returns
    -------
    tuple of str
        The tools with their verb (``"... needs"`` or ``"... need"``), and
        what a prefix of the sheet gives none of (``"certified zone or
        certificate"``, ``"certified zone"``, ``"certificate"`` or ``""``).
    """
    names = ["olmoearth_estimate_map_error"]
    lacks: list[str] = []
    if design == "random" and not class_map:
        names.append("olmoearth_certify_zone")
        lacks.append("certified zone")
    if certificate:
        names.append("olmoearth_write_certificate")
        lacks.append("certificate")
    verb = "needs" if len(names) == 1 else "need"
    return f"{_names(names)} {verb}", " or ".join(lacks)


def subset_labelling_sufficient(
    design: str,
    *,
    by_class: bool = False,
    random_order: bool = False,
    route: str | None = None,
    certificate: bool = False,
    class_map: bool = False,
) -> dict[str, str]:
    """Every row of a plan's sheet is its design: a part is not enough.

    Emitted with every plan, since every plan writes a design file and the
    routes that read one (``design_path``) need every row labelled. The
    package draws a stratified sample stratum by stratum, the least
    confident (stratum 0) first, and the sheet keeps that order, so its first
    rows are the least confident windows only. With ``by_class`` the strata
    are the map's classes (the class design), so the sheet runs class by
    class and its first rows are one class only.

    With ``random_order`` (a random design, or a stratified one with a single
    non-empty stratum) the sheet is in the package's random draw order: its
    first rows are a smaller random sample (:func:`prefix_is_random_sample`),
    but only as a fallback when labelling stops early for a reason that does
    not depend on the labels read (:data:`STOP_NOT_ON_THE_LABELS`), estimated
    through ``window_indices`` and ``route`` (the scores argument; ``None``
    for a Studio result the plan sampled itself), with none of what the
    full sheet's other routes give. Agent trial 12 (N2/cluster run 23) said
    "A partial labelling is fine" of a ``seed_from_map`` plan drawn for a
    certificate (``certificate``). The reason names only the routes this
    plan has (:func:`_design_file_routes`): the certified zone for a random
    design of a map with confidence (not a class map, ``class_map``), the
    certificate for a ``seed_from_map`` plan. ``design`` "disagreement" is
    the two-map design, graded only on its whole sheet.
    """
    if design == "disagreement":
        return forbidden(
            SUBSET_LABELLING_SUFFICIENT,
            "write a class, or ? where it cannot be judged, on every row of the "
            "sheet: olmoearth_estimate_map_error grades a design='disagreement' "
            "plan only on its whole sheet, and a part of it labelled is no plan "
            "of its own",
        )
    if random_order:
        tools, lacks = _design_file_routes(
            design, certificate=certificate, class_map=class_map
        )
        why = (
            f"label every row of the sheet: with design_path, {tools} every row "
            "labelled, so a partial labelling is not fine for this plan"
        )
        if certificate:
            why += (
                "; it was drawn with seed_from_map for a certificate, which needs "
                "every row"
            )
        if route:
            why += (
                "; only as a fallback when labelling stops early, "
                f"{STOP_NOT_ON_THE_LABELS}, are the first k rows labelled before "
                "the stop a smaller random sample, estimated with window_indices "
                f"and the same {route}, with a wider interval"
                + (f" and no {lacks}" if lacks else "")
            )
        else:
            why += (
                "; a Studio result sampled by the plan has no scores to pass as "
                "window_indices, so only the full sheet is estimated"
            )
        return forbidden(SUBSET_LABELLING_SUFFICIENT, why)
    if by_class:
        return forbidden(
            SUBSET_LABELLING_SUFFICIENT,
            "label every window in the sheet; labelling only its first rows "
            "biases the estimate: the sheet lists the class design's strata, "
            "the map's classes, in turn, each class's windows in random order, "
            "so its first rows are one class only, a rate over them describes "
            "that class and not the map, and the classes after them get no "
            "labels; olmoearth_estimate_map_error refuses a design with any "
            "window unlabelled",
        )
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


def prefix_is_random_sample(
    n_rows: int,
    route: str | None,
    *,
    design: str = "random",
    certificate: bool = False,
    class_map: bool = False,
) -> dict[str, Any]:
    """The ``prefix_is_random_sample`` fact: every row, a prefix only as a fallback.

    The routes that read the design file (``design_path``) need every row
    labelled, and the sentence names only the ones this plan has
    (:func:`_design_file_routes`): olmoearth_estimate_map_error always,
    olmoearth_certify_zone for a ``design`` "random" of a map with
    confidence (not ``class_map``), olmoearth_write_certificate for a plan
    drawn with ``seed_from_map`` (``certificate``). The sentence leads with
    that requirement; agent trial 12 (N2/cluster run 23) read the fallback,
    stated first, as "A partial labelling is fine" on a ``seed_from_map``
    plan drawn for a certificate (``certificate``, which the sentence then
    names). The package draws a random design without replacement in a
    random order and the sheet keeps that order, so only as a fallback when
    labelling stops early for a reason that does not depend on the labels
    read (:data:`STOP_NOT_ON_THE_LABELS`; a stop decided by what they show
    is optional stopping) are its first ``k`` rows a simple random sample of
    size ``k``: the estimate stays unbiased and its interval widens, and they
    give none of the certified zone and certificate the plan's other routes
    give. ``route`` is the scores argument (``"scores"`` or
    ``"scores_path"``) that olmoearth_estimate_map_error takes with those
    rows as ``window_indices``; a Studio result sampled by the plan itself
    has no scores to pass.
    """
    tools, lacks = _design_file_routes(
        design, certificate=certificate, class_map=class_map
    )
    sentence = (
        f"With design_path, {tools} every one of the sheet's {n_rows} rows labelled."
    )
    if certificate:
        sentence += (
            " This plan was drawn with seed_from_map for a certificate, which "
            "needs every row labelled."
        )
    if route:
        tail = f", and they give no {lacks}." if lacks else "."
        sentence += (
            " Only as a fallback if labelling stops early, "
            f"{STOP_NOT_ON_THE_LABELS}: the sheet is in the package's random draw "
            "order, so the first k rows labelled before the stop are a smaller "
            "random sample; olmoearth_estimate_map_error takes them as "
            f"window_indices with the same {route}, the estimate stays unbiased "
            "and its interval widens" + tail
        )
    else:
        sentence += (
            " Only the full sheet is estimated: a Studio result sampled by the "
            "plan has no scores to pass as window_indices."
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


def says_why(field: str | None) -> str:
    """The pointer " (FIELD says why)" to a result's field giving a reason, or ""."""
    return f" ({field} says why)" if field else ""


def whole_map_rate_for_a_zone(
    alpha: float,
    smallest: Mapping[str, Any] | None = None,
    why_field: str | None = "levels_tested",
) -> dict[str, str]:
    """The whole-map rate neither explains nor predicts that no zone certified.

    Emitted by ``olmoearth_certify_zone`` when nothing is certified. Agent
    trial 12 (B6/files runs 8, 9 and 13) explained the failure by the
    whole-map rate: "nowhere near the 5% bar anywhere on the map", "far
    above the 5% bar you want", "well above the 5% bar, so this outcome is
    expected". The zones tested are the map's most confident windows, and
    the smallest of them, which the prefix rule tests first, held 3 wrong of
    46 labels. ``smallest`` is the result's first level (``coverage``,
    ``n_labelled_inside``, ``n_wrong_inside``), as the package gives it, or
    ``None`` when no level was tested; ``why_field`` is then the result's
    field that says why (``levels_tested`` in olmoearth_certify_zone's,
    ``note`` in olmoearth_write_certificate's, which has no
    ``levels_tested``: the step 11 review), or ``None`` for none.
    """
    if smallest:
        evidence = (
            "; the evidence on the most confident part is the smallest zone "
            f"tested, coverage {float(smallest.get('coverage') or 0.0):g}: "
            f"{smallest.get('n_labelled_inside')} labels, "
            f"{smallest.get('n_wrong_inside')} wrong"
        )
    else:
        evidence = "; no zone was tested" + says_why(why_field)
    return forbidden(
        WHOLE_MAP_RATE_FOR_A_ZONE,
        "the whole-map error rate describes the whole map, not its most "
        f"confident part: set against alpha={alpha:g} it neither explains nor "
        "predicts that no zone certified, and it does not show that every part "
        "of the map is above alpha" + evidence,
    )


# --------------------------------------------------------------------------- a class map

#: What a class map without top-k scores gets, in one sentence, stated by
#: every tool that refuses one: the review-set tools,
#: ``olmoearth_certify_zone`` and a ranking design of
#: ``olmoearth_plan_label_sample``. A Studio classification band's
#: pixel-value carries the class only, with no probability: nothing ranks its
#: windows, so it has no review set and no certified zone, and a random or
#: class-design sample of it still measures its error rate and per-class
#: accuracy, as labels where it differs from another map of one time say
#: which of the two is more accurate. A class map with top-k scores
#: (:data:`TOPK_MAP_KIND`) is never refused this way: its rank-1 minus rank-2
#: score is its margin.
CLASS_MAP_NO_CONFIDENCE = (
    "A class map without top-k scores has no confidence to rank or certify, so "
    "it gets no review set and no certified zone; it can still be measured "
    "(olmoearth_plan_label_sample with design='random' or 'class', then "
    "olmoearth_estimate_map_error) or compared with another map of the same "
    "time (olmoearth_plan_label_sample with design='disagreement')."
)

#: The kind of map a Studio segmentation result is when its prediction was
#: submitted with ``probability_output``: each window's class, as a class map
#: has, and each rank's softmax score as bands of their own (``P_rank1_prob``
#: and ``P_rank2_prob`` beside the class property ``P``). Every text that
#: tells it from a class map without them names it so.
TOPK_MAP_KIND = "class map with top-k scores"

#: Where a class map's confidence comes from, stated after
#: :data:`CLASS_MAP_NO_CONFIDENCE` in every refusal of a class map without
#: top-k scores. Studio's ``PredictionWrite.probability_output`` asks for
#: the top-k softmax scores beside the classes ("segmentation only"); a
#: margin needs two ranks. Worded as a requirement that names the scores, so
#: an answer repeating it offers no zone (``zone_for_a_class_map``).
TOPK_FROM_PROBABILITY_OUTPUT = (
    "A review set or a certified zone of a Studio class map needs its top-k "
    "class scores: a new prediction of a segmentation model, submitted with "
    "probability_output (top_k of 2 or more), carries them."
)

#: A refusal of a class map without top-k scores: the class-map sentence,
#: then where top-k scores come from.
CLASS_MAP_REFUSAL = f"{CLASS_MAP_NO_CONFIDENCE} {TOPK_FROM_PROBABILITY_OUTPUT}"


def topk_scores_must_state(quantized: bool = True) -> str:
    """What every result read from a class map's top-k scores must state.

    Studio's top-k scores are a model's softmax, uncalibrated (the files
    zip's README: "These are uncalibrated softmax scores, not calibrated
    probabilities"), stored as uint8 levels of 1/254 (``quantized``, a score
    declared ``[0, 254]``): their margin orders windows for review and
    certification, and is no probability that a window's class is right.
    """
    form = ", quantized to 1/254" if quantized else ""
    return (
        f"The top-k scores are Studio's uncalibrated softmax scores{form}: their "
        "margin ranks windows and is not a probability the class is right."
    )


#: What ``olmoearth_scores_from_studio`` states of a Studio class map's top-k
#: scores, the per-pixel scores it downloaded: the same scores as
#: :func:`topk_scores_must_state`'s, but the scores file it writes ranks a
#: window by its pooled top-1 score, not the margin, so the sentence says they
#: rank windows.
STUDIO_TOPK_SCORES_MUST_STATE = (
    "These scores are Studio's uncalibrated softmax scores, quantized to "
    "1/254: they rank windows and are not probabilities that a class is right."
)

#: What every later result on such a scores file states
#: (``olmoearth_review_set``, ``olmoearth_plan_label_sample``,
#: ``olmoearth_estimate_map_error`` and ``olmoearth_certify_zone``): its
#: rows are window scores, each the mean of the window's valid pixels' top-1
#: scores (``scores_file``'s signal), which are not on the 1/254 grid the
#: pixels' scores are, so :data:`STUDIO_TOPK_SCORES_MUST_STATE` would call
#: them quantized. Until the final code step of 31 October 2026 these tools
#: carried the scores' assumption as a warning or ``assumption`` only.
STUDIO_WINDOW_SCORES_MUST_STATE = (
    "These window scores pool Studio's uncalibrated softmax scores, each "
    "quantized to 1/254: they rank windows and are not probabilities that a "
    "class is right."
)


def zone_for_a_class_map(detail: str | None = None) -> dict[str, str]:
    """No certified zone, and no certificate, for a class map without top-k scores.

    A certified zone is the most confident share of a map whose error rate an
    exact test bounds; a class map without top-k scores has no confidence to
    order its windows by, so ``olmoearth_certify_zone`` refuses its designs,
    random ones included. Emitted by every result about such a class map:
    its plan, its estimate, the refusals; never by a result about a
    :data:`TOPK_MAP_KIND`, whose random design can certify a zone.
    ``detail`` says what this result adds.
    """
    return forbidden(
        ZONE_FOR_A_CLASS_MAP,
        "a class map without top-k scores has no confidence to rank its windows "
        "by, so no zone of it can be certified (olmoearth_certify_zone refuses "
        "it) whatever the design, budget or alpha: no certified, certifiable or "
        "trusted zone and no certificate may be offered or stated for it; labels "
        "give it an error rate and per-class accuracy, each with its interval"
        + (f"; {detail}" if detail else ""),
    )


def valid_interval_must_state(
    estimate: float | None,
    low: float,
    high: float,
    design: str,
    n_labelled: int,
    *,
    class_map: bool,
) -> str:
    """The sentence an answer stating an error rate conveys: valid interval and design.

    Written from ``oe_inferencex.estimate.estimate_error_rate(valid=True)``'s
    ``valid_interval`` (its own point, and its ends), never re-derived: the
    interval valid at every sample size, the design it rests on and the
    number of labels; for a class map, that it has no certified zone. At
    most 25 words, as the contract's ``must_state`` is. ``estimate`` is
    ``None`` when the package gives the point as a range.
    """
    head = (
        f"Error rate {percent(estimate)}: " if estimate is not None else "Error rate: "
    )
    sentence = (
        f"{head}valid 95% interval {percent(low)} to {percent(high)}, {design} "
        f"design, {n_labelled} labels"
    )
    return sentence + ("; a class map has no certified zone." if class_map else ".")


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


# --------------------------------------------------------------------------- which of two maps is more accurate

#: The route that names a more accurate map: the package's grading of labels
#: on a random sample of the windows where two maps of one time differ
#: (``oe_inferencex.estimate.sample_disagreement`` and
#: ``compare_from_disagreement``).
WHICH_MAP_ROUTE = (
    "olmoearth_plan_label_sample with design='disagreement', then "
    "olmoearth_estimate_map_error on the class the reviewer saw in each "
    "planned window"
)

#: Stated with every two-map plan and grade: the package's note
#: (``WHICH_MAP_NOTE``) in one sentence of the contract's length.
NOT_EITHER_ACCURACY = (
    "This says which map is more accurate, not either map's accuracy; that "
    "needs a random sample of each map."
)

#: Stated with a two-map grade: a map is one training run and one
#: prediction, so its verdict ranks no model (Corley et al. 2026, arXiv
#: 2605.12678, R4: no gain claimed inside run-to-run variation; in
#: olmoearth-inferencex's record a head redraw moved 2 to 4% of windows,
#: exp57, and a library upgrade 14 of 259 points, exp111).
MAPS_NOT_MODELS = (
    "This grades these two maps on these windows; it does not rank the models "
    "that made them on other areas, dates or training runs."
)


def more_accurate_without_labels(n_disagree: int, n_population: int) -> dict[str, str]:
    """A two-map plan holds no labels, so it names no more accurate map.

    The plan is the label-free half of the which-map route: how often two
    maps differ, or which is more confident, says nothing about which is
    right. ``winner_without_labels``' reason for the comparisons says they
    take no labels; a plan's labels are still to come, so its reason says so.
    """
    return forbidden(
        WINNER_WITHOUT_LABELS,
        f"this plan holds no labels yet: that the two maps differ on {n_disagree} "
        f"of the {n_population} windows compared says nothing about which is more "
        "accurate, better or right, and neither does which is more confident; "
        "only olmoearth_estimate_map_error on the reviewer's classes for the "
        "planned windows names one, by the package's test or interval, and it "
        "may name neither",
    )


def more_accurate_without_its_test(detail: str) -> dict[str, str]:
    """A map is more accurate only as the package's which-map grading names it.

    The rule for an answer that names a more accurate map: it names the map
    the package's verdict names, and states the test or interval that verdict
    rests on (the exact sign test while the labels are at most a fifth of the
    differing windows, the interval past that) and the number of labels
    (``harness/checks.py`` reads a sentence that names a map without them
    against the run's ``which_map`` facts). ``detail`` says what this result
    names.
    """
    return forbidden(
        MORE_ACCURATE_WITHOUT_ITS_TEST,
        "a map is more accurate only as the package's grading of labels on the "
        f"windows where two maps differ names it ({WHICH_MAP_ROUTE}): the exact "
        "sign test while the labels are at most a fifth of the differing "
        "windows, the interval past that; an answer that names a more accurate "
        "map states that test or interval and the number of labels" + f"; {detail}",
    )


def which_map_must_state(result: Mapping[str, Any]) -> str:
    """The sentence an answer about which of two maps is more accurate conveys.

    Written from ``oe_inferencex.estimate.compare_from_disagreement``'s
    result, never re-derived: the map its ``verdict`` names (or neither), the
    rule that decided (``verdict_rule``: the exact sign test, with its
    p-value, or the interval), the number of labels, and the difference in
    accuracy, a minus b, as its interval over the windows compared, in
    points. At most 25 words, as the contract's ``must_state`` is.
    """
    v = result.get("verdict")
    n = int(result["n_labelled"])
    difference = result["difference"]
    lo, hi = 100.0 * float(difference["low"]), 100.0 * float(difference["high"])
    level = f"{100.0 * float(result.get('conf', 0.95)):g}%"
    sign = result.get("verdict_rule") == "sign"
    if v in ("a", "b"):
        if sign:
            p = float((result.get("sign_test") or {}).get("p_value", 1.0))
            return (
                f"Map {v} is more accurate: exact sign test on {n} labels, p = "
                f"{p:.3g}; a minus b: {lo:+.1f} to {hi:+.1f} points, {level} "
                "interval."
            )
        return (
            f"Map {v} is more accurate: the {level} interval on a minus b, from "
            f"{n} labels, is {lo:+.1f} to {hi:+.1f} points."
        )
    if sign:
        return (
            f"These {n} labels name neither map more accurate (exact sign test); "
            f"a minus b: {lo:+.1f} to {hi:+.1f} points, {level} interval."
        )
    return (
        f"These {n} labels name neither map more accurate: the {level} interval "
        f"on a minus b is {lo:+.1f} to {hi:+.1f} points."
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


#: What two maps of different or overlapping times do not get, stated by
#: ``olmoearth_compare_review`` as ``not_for_these_maps`` (agent trial 12:
#: B3/cluster run 5 offered "a disagreement-based labeling budget"; B7/files
#: runs 7 and 12 offered another unlabelled map of the same period and "a
#: same-date third inference to separate change from error").
NOT_FOR_MAPS_OF_DIFFERENT_TIMES = (
    "design='disagreement' (which map is more accurate) needs two maps of one "
    "time, so olmoearth_plan_label_sample refuses it for these two maps given "
    "these dates as date_a and date_b (do not call it without them); another "
    "unlabelled map or inference, of any date, cannot tell change from error."
)


def two_map_design_across_dates(
    a: str | None = None, b: str | None = None
) -> dict[str, str]:
    """No disagreement design, plan, sample or budget for maps of different times.

    Agent trial 12 offered "a disagreement-based labeling budget" (B3/cluster
    run 5) and "olmoearth_plan_label_sample (design="disagreement") ... on the
    reviewer's per-window judgments" (B7/files run 11) for two maps that
    olmoearth_compare_review had read as of different times, a design
    olmoearth_plan_label_sample refuses for them when given their dates (a
    scores file's come only from ``date_a`` and ``date_b``; without them the
    plan reads the dates as unstated and draws it). ``a`` and ``b`` are the
    maps' dates when known.
    """
    when = f" ({a} and {b})" if a and b else ""
    return forbidden(
        TWO_MAP_DESIGN_ACROSS_DATES,
        "design='disagreement' needs two maps of one time; for maps of different "
        f"times{when} no labels on the differing windows say which map is more "
        "accurate, since a window where they differ may have changed on the "
        "ground: olmoearth_plan_label_sample refuses the design for them when "
        "given their dates, so no disagreement design, plan, sample or labelling "
        "budget may be offered; "
        "each map gets a sample of its own, labelled at its own date",
    )


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
