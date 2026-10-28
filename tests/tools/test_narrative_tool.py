# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Tests for the olmoearth-case-narrative tool.

The report's "How accurate is this map" section states what the request's
verification tools measured, from their recorded results: these runs call
the real olmoearth-inferencex tools (``uv sync --all-extras``), record each
call as the lead agent does (``ProvenanceLog.record_tool_call``), and hold
every figure of the report to the tool's own. A request that measured
nothing gets a report that says so; a title or result typed with an
accuracy figure is refused; a zone tried again on the same labels with
another alpha or rule is a second look, never stated as certified.
"""

from __future__ import annotations

import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from olmoearth_agent.harness.checks import (
    FORBIDDEN,
    NUMBERS,
    RunEvidence,
    ToolRecord,
    check_forbidden_claims,
    run_checks,
    unsupported_of,
)
from olmoearth_agent.harness.state import ThreadState
from olmoearth_agent.llm.types import ToolCall
from olmoearth_agent.reporting.narrative import (
    ACCURACY_HEADING,
    CONFIDENT_END,
    HOW_TO_MEASURE,
    INTERVAL_NAMES,
    LABELS_CHANGED,
    LABELS_METRICS_NOTE,
    MEASURED_LEAD,
    NOT_MEASURED,
    ZONE_MEANING,
)
from olmoearth_agent.tools import statistical_rules as rules
from olmoearth_agent.tools.estimation import build_estimation_tools
from olmoearth_agent.tools.evaluate import build_evaluate_tools
from olmoearth_agent.tools.narrative import build_narrative_tools
from olmoearth_agent.tools.registry import ToolContext, ToolRegistry
from olmoearth_agent.tools.statistical_rules import percent

PLAN = "olmoearth_plan_label_sample"
GRADE = "olmoearth_estimate_map_error"
CERTIFY = "olmoearth_certify_zone"
METRICS = "olmoearth_classification_metrics"
NARRATIVE = "olmoearth_case_narrative"


@pytest.mark.asyncio
async def test_case_narrative_tool_uses_state_provenance() -> None:
    state = ThreadState()
    state.provenance.record_tool_call(
        "olmoearth_load_context", {}, {"ok": True, "result": {}}
    )
    ctx = ToolContext(studio=None, state=state)  # type: ignore[arg-type]
    tool = build_narrative_tools()[0]
    result = await tool.handler(
        {
            "title": "My Case",
            "results": [
                {
                    "result_id": "r1",
                    "tile_urls": ["/t"],
                    "property_names": ["s"],
                    "creation_time": datetime.now(timezone.utc).isoformat(),
                }
            ],
        },
        ctx,
    )
    md = result["markdown"]
    assert "# My Case" in md
    assert "olmoearth_load_context" in md  # provenance pulled from state
    assert result["result_count"] == 1
    assert result["gated"] is False  # creation_time is "now" -> fresh


# --------------------------------------------------------------------------- a run, recorded as the agent records it


@pytest.fixture(autouse=True)
def _roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("OLMOEARTH_SCORES_ROOT", str(tmp_path))
    monkeypatch.setenv("OLMOEARTH_OUTPUT_ROOT", str(tmp_path))
    return tmp_path


class _Run:
    """One run: each call dispatched, then recorded as the lead agent records it
    (its ``ToolRecord`` for the answer checks, its provenance entry for the
    report)."""

    def __init__(self) -> None:
        self.state = ThreadState()
        self.records: list[ToolRecord] = []
        self.registry = ToolRegistry()
        self.registry.register_all(build_estimation_tools())
        self.registry.register_all(build_evaluate_tools())
        self.registry.register_all(build_narrative_tools())

    async def dispatch(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """The call's envelope, recorded whether it succeeded or not."""
        envelope: dict[str, Any] = await self.registry.dispatch(
            ToolCall(id=str(len(self.records)), name=name, arguments=args),
            ToolContext(studio=None, state=self.state),  # type: ignore[arg-type]
        )
        self.records.append(ToolRecord(name, args, envelope))
        self.state.provenance.record_tool_call(name, args, envelope)
        return envelope

    async def call(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        envelope = await self.dispatch(name, args)
        assert envelope["ok"] is True, envelope
        result: dict[str, Any] = envelope["result"]
        return result

    def evidence(self, *, without_last: bool = False) -> RunEvidence:
        """The run's evidence for the answer checks (less the last call)."""
        records = self.records[:-1] if without_last else self.records
        return RunEvidence(
            tools=list(records), user_messages=["Write the brief for this map."]
        )


def _map(n: int = 2000, seed: int = 11) -> tuple[list[list[float]], list[int]]:
    """Three-class logits with errors concentrated at low margin, and the truth
    (as tests/tools/test_estimation_tool.py draws them)."""
    rng = random.Random(seed)
    scores, wrong = [], []
    for _ in range(n):
        row = [rng.gauss(0.0, 1.5) for _ in range(3)]
        top = sorted(row, reverse=True)
        scores.append(row)
        wrong.append(1 if rng.random() < 0.6 * 2.718 ** -(top[0] - top[1]) else 0)
    return scores, wrong


def _design(plan: dict[str, Any]) -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads(Path(plan["design_path"]).read_text())
    return loaded


def _scores_file(path: Path, classes: list[int]) -> str:
    """A scores file of a 6 x 6 grid, one-hot over three classes."""
    rows = [[1.0 if k == c else 0.0 for k in range(3)] for c in classes]
    path.write_text(json.dumps({"grid": [6, 6], "scores": rows}))
    return str(path)


def _section(markdown: str) -> str:
    """The report's accuracy section, from its heading to the next section."""
    start = markdown.index(ACCURACY_HEADING)
    end = markdown.find("\n## ", start + len(ACCURACY_HEADING))
    return markdown[start:] if end < 0 else markdown[start:end]


async def _random_plan(run: _Run) -> tuple[dict[str, Any], list[int]]:
    """A random plan of 300 windows (seed 0) and its labels' ``wrong``."""
    scores, truth = _map()
    plan = await run.call(PLAN, {"scores": scores, "budget": 300, "design": "random"})
    return plan, [truth[i] for i in _design(plan)["sample"]["indices"]]


async def _random_estimate_and_zone(run: _Run) -> dict[str, Any]:
    """A random plan of 300 windows (seed 0), its error rate and its zone.

    The inputs of tests/tools/test_estimation_tool.py's certified zone: at
    alpha 0.1 under the bonferroni rule these labels certify one.
    """
    plan, wrong = await _random_plan(run)
    labels = {"design_path": plan["design_path"], "wrong": wrong}
    estimate = await run.call(GRADE, labels)
    zone = await run.call(CERTIFY, {**labels, "alpha": 0.1, "rule": "bonferroni"})
    return {"plan": plan, "estimate": estimate, "zone": zone}


async def _class_design_estimate(run: _Run) -> dict[str, Any]:
    """A class-design plan (seed 7) and its valid-first estimate, per class."""
    scores, _truth = _map()
    plan = await run.call(
        PLAN, {"scores": scores, "budget": 300, "design": "class", "seed": 7}
    )
    stored = _design(plan)
    idx, mc = stored["sample"]["indices"], stored["population"]["map_class"]
    reference = [mc[i] if k % 7 else (mc[i] + 1) % 3 for k, i in enumerate(idx)]
    wrong = [int(r != mc[i]) for r, i in zip(reference, idx)]
    labels = {
        "design_path": plan["design_path"],
        "wrong": wrong,
        "reference": reference,
        "n_classes": 3,
    }
    return await run.call(GRADE, labels)


async def _which_map(run: _Run, tmp_path: Path) -> dict[str, Any]:
    """Two scores files of one 6 x 6 grid, every differing window labelled (seed 3).

    The truth is window i's class i % 3; map a is wrong where i % 7 == 0 and
    map b where i % 4 == 0, so they differ on 13 of the 36 windows.
    """
    truth = [i % 3 for i in range(36)]
    a = [(t + 1) % 3 if i % 7 == 0 else t for i, t in enumerate(truth)]
    b = [(t + 2) % 3 if i % 4 == 0 else t for i, t in enumerate(truth)]
    plan = await run.call(
        PLAN,
        {
            "scores_path": _scores_file(tmp_path / "a.json", a),
            "scores_path_b": _scores_file(tmp_path / "b.json", b),
            "budget": 13,
            "seed": 3,
            "design": "disagreement",
            "date_a": "2024-06-01",
            "date_b": "2024-06-01",
        },
    )
    idx = _design(plan)["sample"]["indices"]
    reference = [str(truth[i]) for i in idx]
    return await run.call(
        GRADE, {"design_path": plan["design_path"], "reference": reference}
    )


def _brief(title: str = "Flood brief") -> dict[str, Any]:
    return {"title": title, "results": []}


# --------------------------------------------------------------------------- a run that measured


@pytest.mark.asyncio
async def test_the_report_states_each_measured_number_as_the_tool_returned_it(
    tmp_path: Path,
) -> None:
    """An error rate, a class-design estimate, a certified zone and a which-map
    grading, then the report: each number, the map it is of, every interval
    with what it is, the labels, design and seed, what the zone's alpha bounds
    and its ISSUED_NOTE word for word, the which-map verdict with its test or
    interval, and the notes that limit each (the package's scope notes up to
    another map's figures); the section's numbers are all in the tool results
    of the run."""
    pytest.importorskip("oe_inferencex.estimate")
    zone_module = pytest.importorskip("oe_inferencex.certify.zone")
    run = _Run()
    measured = await _random_estimate_and_zone(run)
    plan, est, zone = measured["plan"], measured["estimate"], measured["zone"]
    cls = await _class_design_estimate(run)
    graded = await _which_map(run, tmp_path)
    report = await run.call(NARRATIVE, _brief())
    md = report["markdown"]
    section = _section(md)
    assert section.startswith(f"{ACCURACY_HEADING}\n\n{MEASURED_LEAD}")
    assert NOT_MEASURED not in md

    # The random design's error rate, the map it is of, its intervals (the one
    # valid at every sample size first, then the design's, which under a
    # random sample is the same exact interval: the package's hypergeometric
    # interval) and its labels, design, seed.
    assert "- Map: scores given inline." in section
    est_valid = est["valid_interval"]
    assert (
        f"- Error rate: {percent(est_valid['estimate'])}, estimated from "
        f"{est['n_labelled']:,} labels on a population of {est['n_population']:,} "
        "windows."
    ) in section
    assert est["method"].startswith("exact hypergeometric")
    assert (est_valid["low"], est_valid["high"]) == (est["low"], est["high"])
    valid_line = section.index(
        f"- Interval valid at every sample size (95%): {percent(est_valid['low'])} "
        f"to {percent(est_valid['high'])} (method: {est_valid['method']})."
    )
    design_line = section.index(
        f"- Interval this design earns (95%): {percent(est['low'])} to "
        f"{percent(est['high'])} (method: {est['method']})."
    )
    assert valid_line < design_line
    assert (
        f"- Labels, design and seed: {est['n_labelled']:,} labels on the random "
        f"design, seed 0; design file `{plan['design_path']}`."
    ) in section
    assert est["seed"] == plan["seed"] == 0
    # The package's scope note, up to another map's figures (exp88's PASTIS
    # rates) and the command line's option.
    scope_head = est["scope"].split(". ")[0]
    assert f"{scope_head}." in section
    assert "PASTIS" not in section and "--condition" not in section

    # The class design: the interval valid at every sample size first, then the
    # design's, each named with its level; the per-class ranking and the
    # package's notes, the per-class table's (its intervals hold one by one,
    # not jointly) included.
    valid = cls["valid_interval"]
    design_interval = next(e for e in cls["intervals"] if e["interval"] == "design")
    first = section.index(
        f"- Interval valid at every sample size (95%): {percent(valid['low'])} to "
        f"{percent(valid['high'])} (method: {valid['method']})."
    )
    second = section.index(
        f"- Interval this design earns (95%): {percent(design_interval['low'])} to "
        f"{percent(design_interval['high'])} (method: {design_interval['method']})."
    )
    assert first < second
    assert f"- Error rate: {percent(valid['estimate'])}, estimated from" in section
    assert f"{cls['n_labelled']:,} labels on the class design, seed 7" in section
    assert f"design file `{cls['design_path']}`" in section
    weakest = next(f for f in cls["facts"] if f["id"] == "weakest_classes")
    assert f"- Per class: {weakest['sentence']}" in section
    assert f"- olmoearth-inferencex's note: {cls['valid_note']}" in section
    per_class_note = cls["per_class"]["valid_note"]
    assert f"- olmoearth-inferencex's note: {per_class_note}" in section
    for limit in cls["must_state"]:
        assert f"- Limit stated by the tool: {limit}" in section

    # The certified zone: its verdict, what alpha bounds, its limit, its scope
    # note and ISSUED_NOTE verbatim.
    assert zone["certified"] is True and zone["seed"] == 0
    assert zone["verdict"][1:] in section
    assert ZONE_MEANING in section
    assert section.count(zone_module.ISSUED_NOTE) == 1
    assert zone["issued_note"] == zone_module.ISSUED_NOTE
    for limit in zone["must_state"]:
        assert f"- Limit stated by the tool: {limit}" in section
    assert f"- olmoearth-inferencex's note: {zone['note']}" in section
    assert zone["scope"].split(". ")[0] in section

    # Which map is more accurate: the verdict with its test or interval, the
    # labels, design and seed, and the package's notes word for word.
    which = next(f for f in graded["facts"] if f["id"] == "which_map")
    assert f"- Verdict: {which['sentence']}" in section
    n = graded["n_labelled"]
    assert graded["seed"] == 3
    assert f"{n:,} labels on the disagreement design, seed 3" in section
    assert rules.which_map_must_state(graded) in section
    assert f"- olmoearth-inferencex's note: {graded['note']}" in section
    assert f"- olmoearth-inferencex's note: {graded['verdict_note']}" in section

    # The statements copy the tools' figures, equal to them.
    accuracy = report["accuracy"]
    assert accuracy["measured"] is True and accuracy["which_map"] is True
    rate, by_class, certified, which_map = accuracy["statements"]
    assert (rate["kind"], rate["design"], rate["seed"]) == ("error_rate", "random", 0)
    assert rate["map"] == "scores given inline" and rate["labels_changed"] is False
    assert rate["estimate"] == est_valid["estimate"]
    assert rate["intervals"] == [
        {
            "interval": "valid",
            "low": est_valid["low"],
            "high": est_valid["high"],
            "method": est_valid["method"],
        },
        {
            "interval": "design",
            "low": est["low"],
            "high": est["high"],
            "method": est["method"],
        },
    ]
    assert (by_class["design"], by_class["seed"]) == ("class", 7)
    assert by_class["estimate"] == valid["estimate"]
    assert [e["interval"] for e in by_class["intervals"]] == ["valid", "design"]
    assert by_class["intervals"][0]["low"] == valid["low"]
    assert by_class["intervals"][0]["high"] == valid["high"]
    assert certified["kind"] == "certified_zone" and certified["certified"] is True
    assert certified["stated_as_certified"] is True
    assert certified["changed_after_an_earlier_zone"] == []
    keys = ("n_zone", "coverage", "alpha", "delta")
    assert {k: certified[k] for k in keys} == {k: zone[k] for k in keys}
    assert which_map["which_map"] == graded["which_map"]
    assert which_map["difference"]["low"] == graded["difference"]["low"]
    assert which_map["difference"]["high"] == graded["difference"]["high"]
    assert which_map["n_labelled"] == n
    # A measured run's report forbids nothing of its own.
    assert "forbidden_claims" not in report and "must_state" not in report

    # Numbers come only from the tool results: the section, against the run's
    # other calls, holds no number they do not, and makes no claim they forbid.
    found = run_checks(section, run.evidence(without_last=True), [NUMBERS, FORBIDDEN])
    assert found == {}, json.dumps(found, indent=1)


@pytest.mark.asyncio
async def test_a_confidence_designs_interval_is_named_nominal() -> None:
    """The plan's default design: the interval valid at every sample size is
    named first, then the package's design interval, a Wilson interval on the
    design's effective sample size, which it calls nominal (its VALID_NOTE
    gives the coverage it fell to); the report names it so, never a plain
    95%."""
    pytest.importorskip("oe_inferencex.estimate")
    run = _Run()
    scores, truth = _map()
    plan = await run.call(PLAN, {"scores": scores, "budget": 300})
    assert plan["design"] == "confidence"
    wrong = [truth[i] for i in _design(plan)["sample"]["indices"]]
    est = await run.call(GRADE, {"design_path": plan["design_path"], "wrong": wrong})
    assert "Wilson" in est["method"]
    report = await run.call(NARRATIVE, _brief())
    section = _section(report["markdown"])
    valid = est["valid_interval"]
    first = section.index(
        f"- Interval valid at every sample size (95%): {percent(valid['low'])} to "
        f"{percent(valid['high'])} (method: {valid['method']})."
    )
    second = section.index(
        "- Interval this design earns (nominal 95%, approximate: not valid at "
        f"every sample size): {percent(est['low'])} to {percent(est['high'])} "
        f"(method: {est['method']})."
    )
    assert first < second
    assert f"- Error rate: {percent(valid['estimate'])}, estimated from" in section
    assert "- Interval this design earns (95%)" not in section
    (rate,) = report["accuracy"]["statements"]
    assert [e["interval"] for e in rate["intervals"]] == ["valid", "design"]
    found = run_checks(section, run.evidence(without_last=True), [NUMBERS, FORBIDDEN])
    assert found == {}, json.dumps(found, indent=1)


@pytest.mark.asyncio
async def test_the_packages_warning_is_of_the_design_interval() -> None:
    """No labelled window wrong under the confidence design: the package warns
    that the design's variance is zero and its interval the simple-random
    Wilson bound. Beside the interval valid at every sample size, which takes
    no variance, the report names the warning, word for word, as of the
    interval this design earns."""
    pytest.importorskip("oe_inferencex.estimate")
    run = _Run()
    scores, _truth = _map()
    plan = await run.call(PLAN, {"scores": scores, "budget": 300})
    assert plan["design"] == "confidence"
    wrong = [0] * len(_design(plan)["sample"]["indices"])
    est = await run.call(GRADE, {"design_path": plan["design_path"], "wrong": wrong})
    warning = est["warning"]
    assert warning.startswith("no labelled window was wrong")
    report = await run.call(NARRATIVE, _brief())
    section = _section(report["markdown"])
    note = f"- olmoearth-inferencex's note on the interval this design earns: {warning}"
    assert note in section
    assert f"- olmoearth-inferencex's note: {warning}" not in section
    assert section.count(warning) == 1
    valid_line = section.index("- Interval valid at every sample size (95%)")
    design_line = section.index("- Interval this design earns (nominal 95%")
    assert valid_line < design_line < section.index(note)
    found = run_checks(section, run.evidence(without_last=True), [NUMBERS, FORBIDDEN])
    assert found == {}, json.dumps(found, indent=1)


@pytest.mark.asyncio
async def test_windows_labelled_without_a_design_are_not_a_designs_sample() -> None:
    """window_indices: the package estimates them as a simple random sample
    after a check that does not make them one. The report names the interval
    for what it would be, states what the check does not establish, and
    names no design and no seed."""
    pytest.importorskip("oe_inferencex.estimate")
    run = _Run()
    scores, truth = _map()
    drawn = random.Random(3).sample(range(len(scores)), 100)
    est = await run.call(
        GRADE,
        {"window_indices": drawn, "wrong": [truth[i] for i in drawn], "scores": scores},
    )
    assert est["design"] == "none (checked as a random sample)"
    report = await run.call(NARRATIVE, _brief())
    section = _section(report["markdown"])
    assert (
        f"- {INTERVAL_NAMES['none']} (95%): {percent(est['low'])} to "
        f"{percent(est['high'])} (method: {est['method']})."
    ) in section
    assert "Interval this design earns" not in section
    labels = "- Labels and design: 100 labels on windows labelled without a design"
    assert labels in section
    assert "does not make them a random sample" in section
    assert "- Map: scores given inline." in section
    assert "random design" not in section
    confident_end = est["review_set_check"]["looks_drawn_from_the_confident_end"]
    assert (CONFIDENT_END in section) is confident_end
    (rate,) = report["accuracy"]["statements"]
    assert rate["design"] == "none (checked as a random sample)"
    assert rate["seed"] is None and rate["design_path"] is None
    assert report["accuracy"]["measured"] is True
    # The provenance summary names no random design and no seed for them.
    (call,) = run.state.provenance.verification_calls()[:1]
    assert call["design"] == "none (checked as a random sample)"
    assert "seed" not in call


# --------------------------------------------------------------------------- a second look


@pytest.mark.asyncio
async def test_a_zone_tried_again_with_another_alpha_is_no_certificate() -> None:
    """certify at alpha 0.05 (prefix) certifies nothing; at alpha 0.1 under
    bonferroni on the same labels it certifies. The second is a second look
    the guarantee does not cover (post_hoc_alpha, rule_switch_after_failure):
    the report states it as one, never as a certified zone, and writes no
    sentence the run's contract forbids."""
    pytest.importorskip("oe_inferencex.estimate")
    zone_module = pytest.importorskip("oe_inferencex.certify.zone")
    run = _Run()
    plan, wrong = await _random_plan(run)
    labels = {"design_path": plan["design_path"], "wrong": wrong}
    nothing = await run.call(CERTIFY, {**labels, "alpha": 0.05})
    assert nothing["certified"] is False
    second = await run.call(CERTIFY, {**labels, "alpha": 0.1, "rule": "bonferroni"})
    assert second["certified"] is True
    report = await run.call(NARRATIVE, _brief())
    section = _section(report["markdown"])
    looked = (
        "- Not stated as a certified zone: this call (alpha=0.1, delta=0.1, rule "
        "'bonferroni') came after an earlier call of this request on the same "
        "design file and changed its alpha and rule once that result was seen, a "
        "second look that falls outside what the guarantee covers."
    )
    assert looked in section
    assert nothing["verdict"] in section
    assert second["verdict"][1:] not in section
    assert zone_module.ISSUED_NOTE not in section and ZONE_MEANING not in section
    for limit in second["must_state"]:
        assert limit not in section
    first, again = report["accuracy"]["statements"]
    assert (first["certified"], first["changed_after_an_earlier_zone"]) == (False, [])
    assert again["certified"] is True and again["stated_as_certified"] is False
    assert again["changed_after_an_earlier_zone"] == ["alpha", "rule"]
    # The line is no claim the run forbids, and its numbers are the run's.
    evidence = run.evidence(without_last=True)
    assert check_forbidden_claims(looked, evidence) == []
    assert run_checks(looked, evidence, [NUMBERS]) == {}


@pytest.mark.asyncio
async def test_labels_changed_after_a_result_was_seen_are_said() -> None:
    """Two estimates of one design file with other labels: the second says its
    labels changed after a result was seen; a zone on those labels after an
    estimate of other labels is a second look, not stated as certified."""
    pytest.importorskip("oe_inferencex.estimate")
    run = _Run()
    plan, wrong = await _random_plan(run)
    path = plan["design_path"]
    flipped = [1 - w if k < 5 else w for k, w in enumerate(wrong)]
    first = await run.call(GRADE, {"design_path": path, "wrong": wrong})
    again = await run.call(GRADE, {"design_path": path, "wrong": wrong})
    changed = await run.call(GRADE, {"design_path": path, "wrong": flipped})
    assert changed["estimate"] != first["estimate"] == again["estimate"]
    await run.call(CERTIFY, {"design_path": path, "wrong": flipped, "alpha": 0.1})
    report = await run.call(NARRATIVE, _brief())
    section = _section(report["markdown"])
    statements = report["accuracy"]["statements"]
    assert [s["labels_changed"] for s in statements[:3]] == [False, False, True]
    zone = statements[3]
    assert zone["changed_after_an_earlier_zone"] == ["labels"]
    assert zone["stated_as_certified"] is False
    assert section.count(LABELS_CHANGED) == 1
    assert "changed its labels once that result was seen" in section
    # The summary shows the two label sets apart by their digests.
    calls = run.state.provenance.verification_calls()
    assert calls[1]["wrong"] == calls[2]["wrong"] != calls[3]["wrong"]


# --------------------------------------------------------------------------- a run that did not


@pytest.mark.asyncio
async def test_a_report_without_measurement_says_so_and_states_no_figure() -> None:
    """No verification call: the section says no error rate was estimated in
    the request and names the tools that estimate one; the result requires
    the answer to say so and forbids any accuracy figure."""
    run = _Run()
    report = await run.call(NARRATIVE, _brief())
    section = _section(report["markdown"])
    assert section == f"{ACCURACY_HEADING}\n\n{NOT_MEASURED}\n\n{HOW_TO_MEASURE}\n"
    assert "olmoearth_estimate_map_error" in section and "%" not in section
    # A request is one run: on the web UI an earlier message's estimate is
    # stated only when estimated again in this request.
    assert "each message is a request of its own" in HOW_TO_MEASURE
    assert report["accuracy"] == {
        "measured": False,
        "which_map": False,
        "labels_metrics": False,
        "statements": [],
    }
    assert report["must_state"] == [rules.NOT_MEASURED_MUST_STATE]
    ids = [c["id"] for c in report["forbidden_claims"]]
    assert ids == [rules.ACCURACY_NOT_MEASURED]
    assert report["not_a_source"] == ["markdown"]


@pytest.mark.asyncio
async def test_a_planned_sample_awaiting_labels_is_listed_not_measured() -> None:
    """A plan alone holds no labels: the report lists it, with its design and
    seed, and states no error rate."""
    pytest.importorskip("oe_inferencex.estimate")
    run = _Run()
    scores, _truth = _map()
    plan = await run.call(
        PLAN, {"scores": scores, "budget": 300, "design": "random", "seed": 5}
    )
    report = await run.call(NARRATIVE, _brief())
    section = _section(report["markdown"])
    assert NOT_MEASURED in section and HOW_TO_MEASURE in section
    assert (
        "- Labels, design and seed: 300 windows to label on the random design, "
        f"seed 5; design file `{plan['design_path']}`."
    ) in section
    assert "%" not in section
    assert report["accuracy"]["measured"] is False
    assert report["accuracy"]["statements"] == [
        {
            "kind": "plan",
            "tool": PLAN,
            "design": "random",
            "seed": 5,
            "design_path": plan["design_path"],
            "budget": 300,
        }
    ]
    ids = [c["id"] for c in report["forbidden_claims"]]
    assert ids == [rules.ACCURACY_NOT_MEASURED]
    # The section itself states no accuracy: no claim of the run flags it.
    assert check_forbidden_claims(section, run.evidence()) == []


@pytest.mark.asyncio
async def test_a_which_map_grading_alone_measures_neither_maps_accuracy(
    tmp_path: Path,
) -> None:
    """The grading's verdict is stated, and neither map's own accuracy; its
    difference in accuracy, a minus b, which its next steps tell the answer
    to report, is not read as an unmeasured accuracy."""
    pytest.importorskip("oe_inferencex.estimate")
    run = _Run()
    graded = await _which_map(run, tmp_path)
    report = await run.call(NARRATIVE, _brief("Two flood maps"))
    section = _section(report["markdown"])
    which = next(f for f in graded["facts"] if f["id"] == "which_map")
    assert f"- Verdict: {which['sentence']}" in section
    assert report["accuracy"]["measured"] is False
    assert report["accuracy"]["which_map"] is True
    assert report["must_state"] == [rules.NEITHER_MAP_MEASURED_MUST_STATE]
    assert "which-map grading" in report["forbidden_claims"][0]["why"]
    evidence = run.evidence()
    difference = "The difference in accuracy, a minus b, is 8.3 points."
    assert check_forbidden_claims(difference, evidence) == []
    own = "Map a's accuracy is 94%."
    assert [v["text"] for v in check_forbidden_claims(own, evidence)] == [own]


@pytest.mark.asyncio
async def test_accuracy_against_given_labels_is_not_an_estimated_error_rate() -> None:
    """olmoearth_classification_metrics (skill #7) computes accuracy against
    labels the caller gives: the report says it is not the map's error rate
    from a planned sample, and an answer may state that accuracy."""
    run = _Run()
    metrics = await run.call(METRICS, {"y_true": [0, 1, 1, 0], "y_pred": [0, 1, 0, 0]})
    assert metrics["accuracy"] == 0.75
    report = await run.call(NARRATIVE, _brief())
    section = _section(report["markdown"])
    assert section == (
        f"{ACCURACY_HEADING}\n\n{NOT_MEASURED}\n\n{LABELS_METRICS_NOTE}\n\n"
        f"{HOW_TO_MEASURE}\n"
    )
    assert report["accuracy"]["labels_metrics"] is True
    # Its must_state holds beside that accuracy: no error rate was estimated.
    assert report["must_state"] == [rules.NOT_MEASURED_MUST_STATE]
    stated = "Accuracy against the 4 labels you gave is 75.0%."
    assert check_forbidden_claims(stated, run.evidence()) == []


# --------------------------------------------------------------------------- a figure the model typed


@pytest.mark.asyncio
async def test_a_model_typed_accuracy_in_the_title_or_results_is_refused() -> None:
    """The title and results are the model's typing, rendered as given: an
    accuracy figure in them is refused, so the report never shows one no
    tool returned, whether or not the answer repeats it."""
    run = _Run()
    refused = await run.dispatch(NARRATIVE, _brief("Flood extent: 94% accurate"))
    assert refused["ok"] is False
    assert "in the report's title" in refused["error"]
    assert "accuracy section" in refused["error"]
    # The error names where, never the figure: a figure echoed in a result
    # would be a source for the answer's number check.
    assert "94" not in refused["error"]
    assert run.evidence().pool.unsupported("94%") == ["94%"]
    typed = {
        "title": "Flood brief",
        "results": [{"result_id": "r1", "property_names": ["flood (91% correct)"]}],
    }
    refused = await run.dispatch(NARRATIVE, typed)
    assert refused["ok"] is False and "in the report's results" in refused["error"]
    fine = await run.call(NARRATIVE, _brief("Flood extent, accuracy not measured"))
    assert fine["markdown"].startswith("# Flood extent, accuracy not measured")


@pytest.mark.asyncio
async def test_a_typed_accuracy_beside_a_measured_one_is_refused_too() -> None:
    """With an error rate measured, the accuracy section states the tool's
    figures; one typed into the title is refused all the same, and any other
    number typed there is in no tool result (the report's Markdown is no
    source of numbers)."""
    pytest.importorskip("oe_inferencex.estimate")
    run = _Run()
    plan, wrong = await _random_plan(run)
    await run.call(GRADE, {"design_path": plan["design_path"], "wrong": wrong})
    refused = await run.dispatch(NARRATIVE, _brief("Flood extent: 93.21% accurate"))
    assert refused["ok"] is False
    title = "Flood brief: 3,412.5 hectares under water"
    report = await run.call(NARRATIVE, _brief(title))
    assert report["accuracy"]["measured"] is True
    # The figure is in no other record of the run (else the test proves nothing).
    assert run.evidence(without_last=True).pool.unsupported("3,412.5") == ["3,412.5"]
    found = run_checks(report["markdown"], run.evidence(), [NUMBERS])
    assert unsupported_of(found[NUMBERS]) == ["3,412.5"]
