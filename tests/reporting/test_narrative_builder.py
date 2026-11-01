# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Tests for the case-narrative builder, its freshness gate and its accuracy section."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from olmoearth_agent.provenance.log import ProvenanceLog
from olmoearth_agent.reporting.narrative import (
    ACCURACY_HEADING,
    CONFIDENT_END,
    HOW_TO_MEASURE,
    INTERVAL_NAMES,
    LABELS_CHANGED,
    MAP_NOT_GIVEN,
    MAP_NOT_IN_REQUEST,
    MEASURED_LEAD,
    NEITHER_MAP_MEASURED,
    NO_DESIGN_LIMIT,
    NOT_MEASURED,
    USER_VS_PRODUCER_LEAD,
    ZONE_MEANING,
    accuracy_section,
    build_narrative,
    freshness_check,
    typed_text,
)

NOW = datetime(2026, 5, 28, 12, 0, 0, tzinfo=timezone.utc)


def test_freshness_fresh() -> None:
    r = freshness_check("2026-05-28T06:00:00Z", window_hours=24, now=NOW)
    assert r["fresh"] is True
    assert r["age_hours"] == 6.0


def test_freshness_stale() -> None:
    r = freshness_check("2026-05-26T12:00:00Z", window_hours=24, now=NOW)
    assert r["fresh"] is False
    assert r["age_hours"] == 48.0


def test_freshness_unparseable() -> None:
    r = freshness_check("not-a-date", now=NOW)
    assert r["fresh"] is None
    assert "error" in r


def test_build_narrative_renders_fresh_gates_stale() -> None:
    results = [
        {
            "result_id": "fresh1",
            "tile_urls": ["/t/fresh"],
            "property_names": ["score"],
            "creation_time": "2026-05-28T11:00:00Z",
        },
        {
            "result_id": "stale1",
            "tile_urls": ["/t/stale"],
            "property_names": ["score"],
            "creation_time": "2026-05-25T12:00:00Z",
        },
    ]
    out = build_narrative("Klamath alfalfa", results, window_hours=24, now=NOW)
    md = out["markdown"]
    assert out["result_count"] == 2
    assert out["stale_count"] == 1
    assert out["gated"] is True
    assert "# Klamath alfalfa" in md
    assert "/t/fresh" in md  # fresh tile rendered
    assert "/t/stale" not in md  # stale tile withheld
    assert "~~stale1~~" in md  # stale result struck through
    assert "Freshness gate" in md


def test_build_narrative_with_provenance_log() -> None:
    log = ProvenanceLog(run_id="r1")
    log.record_tool_call(
        "olmoearth_load_context", {}, {"ok": True, "result": {"project_count": 5}}
    )
    out = build_narrative("Run", [], provenance=log, now=NOW)
    assert "## Provenance" in out["markdown"]
    assert "olmoearth_load_context" in out["markdown"]


def test_build_narrative_no_stale_not_gated() -> None:
    results = [
        {
            "result_id": "r1",
            "tile_urls": ["/t"],
            "property_names": [],
            "creation_time": "2026-05-28T11:30:00Z",
        }
    ]
    out = build_narrative("T", results, window_hours=24, now=NOW)
    assert out["gated"] is False
    assert "Freshness gate" not in out["markdown"]


# --------------------------------------------------------------------------- the accuracy section

_PLAN = "olmoearth_plan_label_sample"
_GRADE = "olmoearth_estimate_map_error"
_CERTIFY = "olmoearth_certify_zone"


def _call(tool: str, result: dict[str, Any], **arguments: Any) -> dict[str, Any]:
    """A recorded verification call that ran, as ``VerificationCall`` holds it."""
    return {
        "tool": tool,
        "arguments": arguments,
        "ok": True,
        "result": {"available": True, **result},
    }


#: A class-design estimate as olmoearth_estimate_map_error returns it (trimmed).
_CLASS_ESTIMATE = {
    "design_path": "/w/class.json",
    "plan_design": "class",
    "design": "class",
    "seed": 7,
    "n_labelled": 1204,
    "n_population": 16384,
    "nominal_coverage": 0.95,
    "valid_interval": {
        "estimate": 0.1234,
        "low": 0.0912,
        "high": 0.1601,
        "method": "valid m",
    },
    "intervals": [
        {"interval": "valid", "low": 0.0912, "high": 0.1601, "method": "valid m"},
        {"interval": "design", "low": 0.0961, "high": 0.1557, "method": "union m"},
    ],
    "valid_note": "The package's note on the valid interval.",
    "condition_note": "The package's note on the classes.",
    "bounds_note": "",
    "scope": (
        "This is the error rate of the whole map. On another map that part was "
        "74.1% wrong. Draw the sample with --condition."
    ),
    "per_class": {
        "valid_note": "Each class's interval holds on its own, not jointly.",
        "per_class": {
            "1": {"n_labelled_map_class": 400},
            "2": {"n_labelled_map_class": 12, "warning": "fewer than 20 labelled."},
        },
    },
    "must_state": ["Error rate 12.3%: valid 95% interval, class design."],
    "facts": [
        {"id": "weakest_classes", "sentence": "Class 2 is the lowest.", "lowest": [2]}
    ],
}
#: A certified zone, of a design no estimate of the run graded.
_ZONE = {
    "certified": True,
    "design": "random",
    "seed": 0,
    "n_labelled": 300,
    "n_zone": 1200,
    "coverage": 0.6,
    "alpha": 0.1,
    "delta": 0.1,
    "rule": "bonferroni",
    "verdict": "certified: the 1200 most confident windows (coverage 0.6 of the "
    "population) under rule 'bonferroni' at alpha=0.1, delta=0.1",
    "issued_note": "The package's note on what delta counts.",
    "note": "The package's note on the rule.",
    "scope": (
        "The zone's error rate is certified over all its windows together. To "
        "certify each part, draw the sample with --condition."
    ),
    "must_state": ["Only the 1200 most confident windows are certified."],
    "facts": [
        {
            "id": "whole_map_estimate",
            "sentence": "The rate from 300 labels.",
            "estimate": 0.06,
        }
    ],
}
#: A which-map grading (a design='disagreement' plan).
_GRADED = {
    "design_path": "/w/pair.json",
    "design": "disagreement",
    "seed": 3,
    "n_labelled": 13,
    "which_map": "a",
    "verdict_rule": "interval",
    "difference": {"estimate": 0.0833, "low": 0.0833, "high": 0.0833},
    "sign_test": {"p_value": 0.09},
    "map_a": {"result_id": "ra", "property_name": "landcover"},
    "map_b": {"scores_path": "/w/b.json"},
    "note": "The package's note on the difference.",
    "verdict_note": "The package's note on the verdict.",
    "must_state": ["Map a is more accurate: the interval, from 13 labels."],
    "facts": [{"id": "which_map", "sentence": "Map a is more accurate, 13 labels."}],
}
#: A plan that no grading of the run read: its labels are still to come.
_WAITING = {"design": "random", "seed": 9, "budget": 40, "design_path": "/w/q.json"}


def test_the_accuracy_section_names_each_interval_the_design_and_the_seed() -> None:
    """Every figure is the tool's, formatted as the tools format their own;
    every interval is named with its level; the map is named by its plan; the
    package's notes are stated (the per-class table's and each named class's
    included), and its scope notes up to another map's figures or the command
    line's option; failed and unavailable calls measure nothing; a plan no
    grading read awaits its labels."""
    plan = {"design": "class", "seed": 7, "design_path": "/w/class.json"}
    calls = [
        _call(_PLAN, plan, result_id="lc-1", property_name="landcover"),
        _call(_GRADE, _CLASS_ESTIMATE, design_path="/w/class.json"),
        _call(_CERTIFY, _ZONE, design_path="/w/zone.json", alpha=0.1),
        _call(_GRADE, _GRADED, design_path="/w/pair.json"),
        _call(_PLAN, _WAITING),
        {"tool": _GRADE, "arguments": {}, "ok": False, "result": None},
        {"tool": _GRADE, "arguments": {}, "ok": True, "result": {"available": False}},
    ]
    lines, accuracy = accuracy_section(calls)
    assert lines[:3] == [ACCURACY_HEADING, "", MEASURED_LEAD]
    error_rate = [
        "### Error rate (`olmoearth_estimate_map_error`)",
        "",
        "- Map: Studio result `lc-1` (landcover).",
        "- Error rate: 12.3%, estimated from 1,204 labels on a population of "
        "16,384 windows.",
        "- Interval valid at every sample size (95%): 9.1% to 16.0% (method: "
        "valid m).",
        "- Interval this design earns (95%): 9.6% to 15.6% (method: union m).",
        "- Labels, design and seed: 1,204 labels on the class design, seed 7; "
        "design file `/w/class.json`.",
        "- Per class: Class 2 is the lowest.",
        "- olmoearth-inferencex's note on class 2: fewer than 20 labelled.",
        "- Limit stated by the tool: Error rate 12.3%: valid 95% interval, class "
        "design.",
        "- olmoearth-inferencex's note: The package's note on the classes.",
        "- olmoearth-inferencex's note: The package's note on the valid interval.",
        "- olmoearth-inferencex's note: Each class's interval holds on its own, "
        "not jointly.",
        "- olmoearth-inferencex's note on scope (its first sentences, before "
        "another map's figures and the command line's options): This is the "
        "error rate of the whole map.",
        "",
    ]
    zone = [
        "### Certified zone (`olmoearth_certify_zone`)",
        "",
        f"- Map: {MAP_NOT_IN_REQUEST}.",
        "- Certified: the 1200 most confident windows (coverage 0.6 of the "
        "population) under rule 'bonferroni' at alpha=0.1, delta=0.1.",
        ZONE_MEANING,
        "- Labels, design and seed: 300 labels on the random design, seed 0; "
        "design file `/w/zone.json`.",
        "- Limit stated by the tool: Only the 1200 most confident windows are "
        "certified.",
        "- What delta counts (olmoearth-inferencex's note, verbatim): The "
        "package's note on what delta counts.",
        "- olmoearth-inferencex's note: The package's note on the rule.",
        "- olmoearth-inferencex's note on scope (its first sentences, before "
        "another map's figures and the command line's options): The zone's error "
        "rate is certified over all its windows together.",
        "- Error rate from the same labels: The rate from 300 labels.",
        "",
    ]
    which_map = [
        "### Which of two maps is more accurate (`olmoearth_estimate_map_error`)",
        "",
        "- Maps: map a is Studio result `ra` (landcover); map b is scores file "
        "`/w/b.json`.",
        "- Verdict: Map a is more accurate, 13 labels.",
        "- Labels, design and seed: 13 labels on the disagreement design, seed 3; "
        "design file `/w/pair.json`.",
        "- Limit stated by the tool: Map a is more accurate: the interval, from 13 "
        "labels.",
        "- olmoearth-inferencex's note: The package's note on the difference.",
        "- olmoearth-inferencex's note: The package's note on the verdict.",
        "",
    ]
    awaiting = [
        "### Planned, awaiting labels (`olmoearth_plan_label_sample`)",
        "",
        "- Labels, design and seed: 40 windows to label on the random design, "
        "seed 9; design file `/w/q.json`.",
        "",
    ]
    assert lines[4:] == error_rate + zone + which_map + awaiting
    text = "\n".join(lines)
    assert "74.1%" not in text and "--condition" not in text
    assert accuracy["measured"] is True and accuracy["which_map"] is True
    kinds = [s["kind"] for s in accuracy["statements"]]
    assert kinds == ["error_rate", "certified_zone", "which_map", "plan"]
    rate, certified = accuracy["statements"][:2]
    assert rate["estimate"] == 0.1234 and rate["n_labelled"] == 1204
    assert rate["map"] == "Studio result `lc-1` (landcover)"
    assert rate["labels_changed"] is False
    assert rate["intervals"] == [
        {"interval": "valid", "low": 0.0912, "high": 0.1601, "method": "valid m"},
        {"interval": "design", "low": 0.0961, "high": 0.1557, "method": "union m"},
    ]
    assert certified["stated_as_certified"] is True
    assert certified["changed_after_an_earlier_zone"] == []


def test_a_wilson_interval_is_named_nominal_never_a_plain_level() -> None:
    """The confidence and proportional designs' interval is the package's
    Wilson interval on the design's effective sample size, which it calls
    nominal: named so, while an exact or union-bound one carries its level."""
    estimate = {
        "design_path": "/w/c.json",
        "design": "confidence",
        "seed": 1,
        "n_labelled": 300,
        "n_population": 2000,
        "nominal_coverage": 0.95,
        "estimate": 0.1,
        "low": 0.07,
        "high": 0.14,
        "method": "stratified by confidence margin; Wilson interval on the "
        "design's effective sample size",
    }
    lines, _ = accuracy_section([_call(_GRADE, estimate)])
    assert (
        "- Interval this design earns (nominal 95%, approximate: not valid at every "
        "sample size): 7.0% to 14.0% (method: stratified by confidence margin; "
        "Wilson interval on the design's effective sample size)."
    ) in lines
    exact = {**estimate, "method": "exact hypergeometric interval"}
    lines, _ = accuracy_section([_call(_GRADE, exact)])
    assert (
        "- Interval this design earns (95%): 7.0% to 14.0% (method: exact "
        "hypergeometric interval)."
    ) in lines


def test_the_section_states_what_a_classs_users_accuracy_leaves_out() -> None:
    """Agent trial 12 (N3/files) called a class well mapped on its user's
    accuracy alone: the section states the tool's user_vs_producer fact right
    after the ranking, and its must_state as a limit; with no such fact it
    states nothing of the kind."""
    contrast = {
        "id": "user_vs_producer",
        "sentence": "Beside each class's user's accuracy: Class 4: what the map "
        "calls class 4 is right 93.3% (86.7% to 97.2%) of the time, but it finds "
        "only 69.0% (60.2% to 76.8%) of the ground that is class 4.",
    }
    stated = (
        "Class 4: the map finds only 60.2% to 76.8% of the class's ground "
        "(producer's accuracy); it misses the rest."
    )
    result = {
        **_CLASS_ESTIMATE,
        "facts": [*_CLASS_ESTIMATE["facts"], contrast],
        "must_state": [*_CLASS_ESTIMATE["must_state"], stated],
    }
    lines, _ = accuracy_section([_call(_GRADE, result, design_path="/w/class.json")])
    at = lines.index("- Per class: Class 2 is the lowest.")
    assert lines[at + 1] == f"- {USER_VS_PRODUCER_LEAD}: {contrast['sentence']}"
    assert f"- Limit stated by the tool: {stated}" in lines
    plain, _ = accuracy_section(
        [_call(_GRADE, _CLASS_ESTIMATE, design_path="/w/class.json")]
    )
    assert not [line for line in plain if USER_VS_PRODUCER_LEAD in line]


def test_a_zone_tried_again_on_the_same_labels_is_a_second_look() -> None:
    """A zone call after an earlier one on the same design file with another
    alpha, delta or rule, or after a call that graded other labels there, is
    never stated as certified; an exact repeat is stated as the first was."""
    whole = [{"id": "whole_map_estimate", "sentence": "s", "estimate": 0.06}]
    nothing = {
        **_ZONE,
        "certified": False,
        "alpha": 0.05,
        "rule": "prefix",
        "n_zone": None,
        "coverage": None,
        "verdict": "No zone is certified at alpha=0.05, delta=0.1 under rule "
        "'prefix': no share of the map is shown to be wrong at most alpha=0.05 of "
        "the time.",
        "must_state": [],
        "facts": whole,
    }
    certified = {**_ZONE, "facts": whole}
    labels = {"design_path": "/w/z.json", "wrong": [0, 1, 0]}
    calls = [
        _call(_CERTIFY, nothing, **labels, alpha=0.05),
        _call(_CERTIFY, certified, **labels, alpha=0.1, rule="bonferroni"),
        _call(_CERTIFY, {**certified, "delta": 0.2}, **labels, alpha=0.1),
        _call(_CERTIFY, certified, **labels, alpha=0.1, rule="bonferroni"),
    ]
    lines, accuracy = accuracy_section(calls)
    first, looser, delta, repeat = accuracy["statements"]
    assert first["changed_after_an_earlier_zone"] == []
    assert looser["changed_after_an_earlier_zone"] == ["alpha", "rule"]
    assert delta["changed_after_an_earlier_zone"] == ["alpha", "delta", "rule"]
    # A repeat of the second is still after calls with another alpha, delta
    # and rule (the first and the third).
    assert repeat["changed_after_an_earlier_zone"] == ["alpha", "delta", "rule"]
    assert [s["stated_as_certified"] for s in accuracy["statements"]] == [False] * 4
    assert (
        "- Not stated as a certified zone: this call (alpha=0.1, delta=0.1, rule "
        "'bonferroni') came after an earlier call of this request on the same "
        "design file and changed its alpha and rule once that result was seen, a "
        "second look that falls outside what the guarantee covers."
    ) in lines
    text = "\n".join(lines)
    assert "Certified: the 1200" not in text and ZONE_MEANING not in text
    assert "The package's note on what delta counts." not in text
    assert f"- {nothing['verdict']}" in lines
    # The same settings and labels, twice: the second is stated as the first.
    twice, accuracy = accuracy_section(calls[1:2] + calls[3:4])
    assert [s["stated_as_certified"] for s in accuracy["statements"]] == [True, True]
    assert twice.count(ZONE_MEANING) == 2
    # Other labels after a graded call on the same design file.
    other = {**certified, "facts": [{**whole[0], "estimate": 0.07}]}
    flipped = {"design_path": "/w/z.json", "wrong": [1, 1, 0]}
    calls = [
        _call(_GRADE, {"n_labelled": 3, "estimate": 0.06, "facts": whole}, **labels),
        _call(_CERTIFY, other, **flipped, alpha=0.1, rule="bonferroni"),
    ]
    lines, accuracy = accuracy_section(calls)
    rate, zone = accuracy["statements"]
    assert rate["labels_changed"] is False
    assert zone["changed_after_an_earlier_zone"] == ["labels"]
    assert any("changed its labels once that result was seen" in s for s in lines)


def test_labels_changed_on_a_design_file_are_said() -> None:
    """Estimates of one design file: other inline labels, or another whole-map
    figure, are labels changed after a result was seen; the same labels are
    not, and labels read from a file are compared by their figure only."""
    whole = {"id": "whole_map_estimate", "sentence": "s", "estimate": 0.06}
    est = {"design_path": "/w/d.json", "n_labelled": 3, "estimate": 0.06}
    same = {**est, "facts": [whole]}
    other = {**est, "estimate": 0.1, "facts": [{**whole, "estimate": 0.1}]}
    calls = [
        _call(_GRADE, same, wrong=[0, 1, 0]),
        _call(_GRADE, same, wrong=[0, 1, 0]),
        _call(_GRADE, same, wrong=[1, 0, 0]),
        _call(_GRADE, same, labels_path="/w/l.csv"),
        _call(_GRADE, other, labels_path="/w/l.csv"),
    ]
    lines, accuracy = accuracy_section(calls)
    changed = [s["labels_changed"] for s in accuracy["statements"]]
    assert changed == [False, False, True, False, True]
    assert lines.count(LABELS_CHANGED) == 2
    # A file of the same labels gives the same figure: no change is said.
    lines, accuracy = accuracy_section(calls[:2] + calls[3:4])
    assert [s["labels_changed"] for s in accuracy["statements"]] == [False] * 3


def test_the_section_says_when_nothing_was_measured() -> None:
    lines, accuracy = accuracy_section([])
    assert lines == [ACCURACY_HEADING, "", NOT_MEASURED, "", HOW_TO_MEASURE, ""]
    assert accuracy == {"measured": False, "which_map": False, "statements": []}
    # A which-map grading alone says which map is more accurate, not how
    # accurate either is.
    graded, _ = accuracy_section([_call(_GRADE, _GRADED)])
    assert graded[-2:] == [NEITHER_MAP_MEASURED, ""]
    assert NOT_MEASURED not in graded


def test_build_narrative_reads_the_logs_verification_calls() -> None:
    """The report takes the verification calls the way it takes provenance:
    from the run's log, never from the caller's arguments; a call of the
    classification-metrics tool, which went on 31 October 2026, in an older
    log measures nothing and adds nothing to a section that states no error
    rate."""
    log = ProvenanceLog(run_id="r9")
    log.record_tool_call(
        _GRADE,
        {"design_path": "/w/class.json"},
        {"ok": True, "result": {"available": True, **_CLASS_ESTIMATE}},
    )
    out = build_narrative("Run", [], provenance=log, now=NOW)
    assert out["accuracy"]["measured"] is True
    assert "- Error rate: 12.3%, estimated from 1,204 labels" in out["markdown"]
    assert out["provenance"] == [
        {"api_call": _GRADE, "args_sha": log.entries[0].request_hash[:12]}
    ]
    assert out["not_a_source"] == ["markdown"]
    metrics = ProvenanceLog(run_id="r10")
    metrics.record_tool_call(
        "olmoearth_classification_metrics",
        {"y_true": [0], "y_pred": [0]},
        {"ok": True, "result": {"accuracy": 1.0, "n": 1}},
    )
    out = build_narrative("Run", [], provenance=metrics, now=NOW)
    assert out["accuracy"] == {"measured": False, "which_map": False, "statements": []}
    section = [ACCURACY_HEADING, "", NOT_MEASURED, "", HOW_TO_MEASURE, ""]
    assert "\n".join(section) in out["markdown"]


def test_the_text_the_report_renders_from_its_caller() -> None:
    """The title and each result's id and property names: what the tool
    refuses an accuracy figure in."""
    results = [
        {"result_id": "r1", "property_names": ["flood", "depth"], "tile_urls": ["/t"]},
        {"property_names": None},
    ]
    assert typed_text("Brief", results) == ["Brief", "r1", "flood", "depth"]


def test_the_section_reads_a_range_a_refused_zone_and_unlabelled_designs() -> None:
    """Windows labelled with no design (no design file, so no seed), as the
    tool returns them; an estimate given as a range (unjudged windows); a
    zone refused for a class map; and a grading whose maps its result does
    not name."""
    no_design = {
        "design": "none (checked as a random sample)",
        "estimated_as": "random",
        "n_labelled": 40,
        "n_population": 400,
        "nominal_coverage": 0.95,
        "estimate": 0.125,
        "low": 0.05,
        "high": 0.2,
        "method": "exact hypergeometric interval (simple random sample of a "
        "finite map)",
        "review_set_check": {
            "looks_like_a_review_set": False,
            "looks_drawn_from_the_confident_end": True,
        },
    }
    ranged = {
        "design_path": "/w/r.json",
        "design": "random",
        "seed": 2,
        "n_labelled": 40,
        "n_population": 400,
        "estimate": None,
        "estimate_range": [0.1, 0.125],
        "low": 0.05,
        "high": 0.2,
        "method": "m",
    }
    refused = {
        "certified": False,
        "design": "random",
        "seed": 5,
        "class_map": True,
        "reason": "A class map has no confidence to rank or certify.",
    }
    calls = [
        _call(_GRADE, no_design, window_indices=[1, 2], scores=[[0.1, 0.9]] * 3),
        _call(_GRADE, ranged),
        _call(_CERTIFY, refused, design_path="/w/lc.json", alpha=0.1),
        _call(_GRADE, {**_GRADED, "map_a": None, "map_b": {}}),
    ]
    lines, accuracy = accuracy_section(calls)
    no_design_lines = [
        "- Map: scores given inline.",
        "- Error rate: 12.5%, estimated from 40 labels on a population of 400 "
        "windows.",
        f"- {INTERVAL_NAMES['none']} (95%): 5.0% to 20.0% (method: exact "
        "hypergeometric interval (simple random sample of a finite map)).",
        NO_DESIGN_LIMIT.format(n="40"),
        CONFIDENT_END,
    ]
    start = lines.index(no_design_lines[0])
    assert lines[start : start + len(no_design_lines)] == no_design_lines
    assert "random design" not in "\n".join(lines[: start + len(no_design_lines)])
    assert (
        "- Error rate: between 10.0% and 12.5%, estimated from 40 labels on a "
        "population of 400 windows."
    ) in lines
    # No nominal coverage returned: no level is stated.
    assert "- Interval this design earns: 5.0% to 20.0% (method: m)." in lines
    assert (
        "- No certified zone: A class map has no confidence to rank or certify."
        in lines
    )
    assert (
        "- Design and seed: the random design, seed 5; design file `/w/lc.json`."
        in lines
    )
    assert "- Maps: map a is not named; map b is not named." in lines
    assert accuracy["measured"] is True
    rate, by_range, zone, _graded = accuracy["statements"]
    assert rate["design"] == "none (checked as a random sample)"
    assert rate["design_path"] is None and rate["seed"] is None
    assert by_range["seed"] == 2 and by_range["estimate"] is None
    assert zone["certified"] is False and zone["n_labelled"] is None


# --------------------------------------------------------------------------- certificates

_WRITE = "olmoearth_write_certificate"
_VERIFY = "olmoearth_verify_certificate"
_MAP_SEED = 2**100 + 7
#: A certificate as olmoearth_write_certificate returns it (trimmed).
_WRITTEN = {
    "design_path": "/w/map.json",
    "design": "random",
    "seed": _MAP_SEED,
    "alpha": 0.2,
    "delta": 0.1,
    "rule": "prefix",
    "n_labelled": 60,
    "certified": True,
    "certificate_written": True,
    "certificate_path": "/w/c/certificate.json",
    "date": "2024-06-01",
    "n_zone": 128,
    "coverage": 1.0,
    "statement": "The package's statement of the zone.",
    "what_it_proves": "The tool's words on what a certificate proves.",
    "issued_note": "The package's note on what delta counts.",
    "issuers_word": "The package's note on the issuer's word.",
    "labels_sha256": "0123456789ab",
    "must_state": ["It certifies the zone's windows taken together."],
    "facts": [
        {
            "id": "whole_map_estimate",
            "sentence": "The rate from 60 labels.",
            "estimate": 0.0,
        },
        {"id": "certificate", "sentence": "The zone is packed."},
    ],
}
#: A write call that certified nothing, with the whole-map estimate.
_NOT_WRITTEN = {
    "design_path": "/w/none.json",
    "design": "random",
    "seed": 3,
    "n_labelled": 60,
    "certified": False,
    "certificate_written": False,
    "reason": "No zone is certified on these labels.",
    "facts": [
        {
            "id": "whole_map_estimate",
            "sentence": "The rate from these 60 labels.",
            "estimate": 0.5,
        }
    ],
}
#: A certificate checked against a copy of the map, verified.
_CHECKED = {
    "verified": True,
    "certificate_path": "/w/c/certificate.json",
    "map_read": "/w/copy.tif",
    "map_given": True,
    "failed_checks": [],
    "statement": "The statement certify gives again.",
    "notes": ["The package's note on the issuer's word."],
    "seed": _MAP_SEED,
    "what_it_proves": "The tool's words on what a certificate proves.",
    "issued_note": "The package's note on what delta counts.",
    "window_lookup": {"window": [2, 3], "inside": True, "where": "inside the zone"},
}
#: A tampered certificate, checked against the map it names.
_TAMPERED = {
    "verified": False,
    "certificate_path": "/w/t/certificate.json",
    "map_read": "/w/map.tif",
    "map_given": False,
    "checks": {"bundle": {"ok": False, "detail": "labels.csv changed"}},
    "failed_checks": ["bundle"],
    "statement": None,
    "issued_note": "The package's note on what delta counts.",
}


def test_a_certificate_is_stated_with_its_zones_note_and_its_check() -> None:
    """A certificate restates its zone: the package's statement, what it
    proves, ISSUED_NOTE and the issuer's word, each verbatim, with its seed;
    a check states verified or each failed check, and the window asked."""
    lines, accuracy = accuracy_section(
        [
            _call(_WRITE, _WRITTEN, design_path="/w/map.json", wrong=[0] * 60),
            _call(_VERIFY, _CHECKED, certificate_path="/w/c", window=[2, 3]),
            _call(_VERIFY, _TAMPERED, certificate_path="/w/t"),
            _call(_WRITE, _NOT_WRITTEN, design_path="/w/none.json"),
        ]
    )
    assert accuracy["measured"] is True and lines[2] == MEASURED_LEAD
    assert "### Certificate (`olmoearth_write_certificate`)" in lines
    for line in (
        "- Written: `/w/c/certificate.json`, for the map of 2024-06-01 as the "
        "issuer states, and verified by olmoearth-inferencex before it was "
        "written.",
        "- Statement (olmoearth-inferencex's, verbatim): The package's statement "
        "of the zone.",
        f"- Labels, design and seed: 60 labels on the random design, seed "
        f"{_MAP_SEED}; design file `/w/map.json`.",
        "- What a certificate proves (the tool's words): The tool's words on "
        "what a certificate proves.",
        "- What delta counts (olmoearth-inferencex's note, verbatim): The "
        "package's note on what delta counts.",
        "- Taken on the issuer's word (olmoearth-inferencex's note, verbatim): "
        "The package's note on the issuer's word.",
        "- Limit stated by the tool: It certifies the zone's windows taken "
        "together.",
        "### Certificate checked (`olmoearth_verify_certificate`)",
        "- Certificate: `/w/c/certificate.json`; map: `/w/copy.tif`.",
        "- Statement (olmoearth-inferencex's, verbatim): The statement certify "
        "gives again.",
        "- Window 2, 3: inside the zone.",
        "- olmoearth-inferencex's note: The package's note on the issuer's word.",
        "- Not verified, so its statement is not shown to hold for this map: "
        "bundle: labels.csv changed.",
        f"- Certificate: `/w/t/certificate.json`; map: `/w/map.tif`{MAP_NOT_GIVEN}.",
        "- No certificate: No zone is certified on these labels.",
        # each write call states the whole-map estimate from its labels
        "- Error rate from the same labels: The rate from 60 labels.",
        "- Error rate from the same labels: The rate from these 60 labels.",
    ):
        assert line in lines, line
    # The zone's note is stated with every certificate written or verified.
    note = "- What delta counts (olmoearth-inferencex's note, verbatim): The "
    assert sum(line.startswith(note) for line in lines) == 2
    kinds = [s["kind"] for s in accuracy["statements"]]
    assert kinds == ["certificate"] * 2 + ["certificate_check"] * 2
    written, none, checked, tampered = accuracy["statements"]
    assert written["written"] is True and written["seed"] == _MAP_SEED
    assert (written["n_zone"], written["alpha"], written["date"]) == (
        128,
        0.2,
        "2024-06-01",
    )
    assert none["written"] is False and none["certificate_path"] is None
    assert checked["verified"] is True and checked["window"] == [2, 3]
    assert tampered["failed_checks"] == ["bundle"] and not tampered["map_given"]


def test_a_certificate_call_with_no_estimate_measures_nothing() -> None:
    """A certificate call measures an error rate only when its result states
    the whole-map estimate from its labels, as olmoearth_write_certificate's
    does: one without it, which certified nothing, leaves the section saying
    no error rate was measured (and the narrative tool's guard on a typed
    figure in place)."""
    bare = {k: v for k, v in _NOT_WRITTEN.items() if k != "facts"}
    lines, accuracy = accuracy_section(
        [_call(_WRITE, bare, design_path="/w/none.json")]
    )
    assert accuracy["measured"] is False and lines[2] == NOT_MEASURED
    assert "- No certificate: No zone is certified on these labels." in lines
    lines, accuracy = accuracy_section(
        [_call(_WRITE, _NOT_WRITTEN, design_path="/w/none.json")]
    )
    assert accuracy["measured"] is True and lines[2] == MEASURED_LEAD
    stated = "- Error rate from the same labels: The rate from these 60 labels."
    assert stated in lines


def test_a_certificate_on_other_labels_than_its_zone_is_said() -> None:
    """certify_zone on one label file, then a certificate on another with the
    same whole-map figure (as many windows wrong): the labels' digests
    differ, so the certificate is stated with LABELS_CHANGED and its own
    figure. The same digest says no change."""
    figure = {
        "id": "whole_map_estimate",
        "sentence": "The rate from 60 labels.",
        "estimate": 0.05,
        "low": 0.01,
        "high": 0.14,
    }
    zone = {**_ZONE, "labels_sha256": "aaaaaaaaaaaa", "facts": [figure]}

    def written(digest: str, labels: str) -> dict[str, Any]:
        result = {**_WRITTEN, "labels_sha256": digest, "facts": [figure]}
        return _call(_WRITE, result, design_path="/w/map.json", labels_path=labels)

    first = _call(
        _CERTIFY, zone, design_path="/w/map.json", labels_path="/w/a.csv", alpha=0.1
    )
    stated = "- Error rate from the same labels: The rate from 60 labels."
    lines, accuracy = accuracy_section([first, written("bbbbbbbbbbbb", "/w/b.csv")])
    zone_said, certificate = accuracy["statements"]
    assert zone_said["kind"] == "certified_zone"
    assert certificate["kind"] == "certificate"
    assert certificate["labels_changed"] is True and LABELS_CHANGED in lines
    assert lines.count(stated) == 2
    lines, accuracy = accuracy_section([first, written("aaaaaaaaaaaa", "/w/a.csv")])
    assert accuracy["statements"][1]["labels_changed"] is False
    assert LABELS_CHANGED not in lines and lines.count(stated) == 1
