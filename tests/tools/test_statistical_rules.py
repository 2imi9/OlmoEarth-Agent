# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""The statistical rules the estimation tools state: next steps, facts, claims.

exp86's blind audit of rounds 6 and 7 found answers that proposed a looser
alpha or a Bonferroni re-run after nothing certified, offered to certify from
a stratified design, offered an error rate for a regression score with no
threshold, and worked out 69/300 and 300 - 173 by hand. These tests hold
``olmoearth_plan_label_sample``, ``olmoearth_estimate_map_error`` and
``olmoearth_certify_zone`` to the output contract that replaces those
guesses: ``next_steps`` by design and outcome, ``facts`` with code-written
sentences, ``must_state`` and ``forbidden_claims``. Every number stays the
package's. The comparisons' claims are in ``test_comparison_rules.py``.
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any

import pytest
from pytest_httpx import HTTPXMock

from olmoearth_agent.analysis.review_set import margins
from olmoearth_agent.harness.state import ThreadState
from olmoearth_agent.llm.types import ToolCall, ToolSpec
from olmoearth_agent.studio.client import StudioClient, StudioConfig
from olmoearth_agent.tools import statistical_rules as rules
from olmoearth_agent.tools.estimation import (
    _jsonable,
    _weakest_classes,
    build_estimation_tools,
)
from olmoearth_agent.tools.registry import RegisteredTool, ToolContext, ToolRegistry
from olmoearth_agent.tools.review_set import build_review_set_tools

estimate = pytest.importorskip("oe_inferencex.estimate")

BASE = "http://mock-studio/api/v1"


def _ctx() -> ToolContext:
    return ToolContext(studio=None, state=ThreadState())  # type: ignore[arg-type]


async def _call(name: str, args: dict[str, Any], ctx: ToolContext | None = None) -> Any:
    registry = ToolRegistry()
    registry.register_all(build_estimation_tools())
    registry.register_all(build_review_set_tools())
    return await registry.dispatch(
        ToolCall(id="1", name=name, arguments=args), ctx or _ctx()
    )


async def _result(
    name: str, args: dict[str, Any], ctx: ToolContext | None = None
) -> Any:
    out = await _call(name, args, ctx)
    assert out["ok"] is True, out
    return out["result"]


def _map(n: int = 1200, seed: int = 7) -> tuple[list[list[float]], list[int]]:
    """Three-class logits with errors concentrated at low margin, and the truth."""
    rng = random.Random(seed)
    scores, wrong = [], []
    for _ in range(n):
        row = [rng.gauss(0.0, 1.5) for _ in range(3)]
        top = sorted(row, reverse=True)
        scores.append(row)
        wrong.append(1 if rng.random() < 0.6 * 2.718 ** -(top[0] - top[1]) else 0)
    return scores, wrong


def _ids(out: dict[str, Any]) -> list[str]:
    return [c["id"] for c in out.get("forbidden_claims", [])]


def _why(out: dict[str, Any], claim_id: str) -> str:
    (why,) = [c["why"] for c in out["forbidden_claims"] if c["id"] == claim_id]
    return why


def _fact(out: dict[str, Any], fact_id: str) -> dict[str, Any]:
    (entry,) = [f for f in out["facts"] if f["id"] == fact_id]
    return entry


#: The output contract's fixed forbidden-claim ids, as the contract lists them.
CONTRACT_IDS = {
    "post_hoc_alpha",
    "rule_switch_after_failure",
    "certify_from_nonrandom_design",
    "error_rate_without_labels",
    "error_rate_for_unthresholded_regression",
    "subset_labelling_sufficient",
    "winner_without_labels",
    "combined_statistic_across_properties",
    "another_date_settles_it",
    "simple_random_interval_for_stratified_design",
    "spatial_pattern_from_one_correlation",
    "agreement_from_uncertain_correlation",
    "review_set_for_unthresholded_regression",
    "one_reference_settles_two_dates",
    "evidence_outside_its_scope",
    "certification_guaranteed",
}


def _holds_to_the_contract(out: dict[str, Any]) -> None:
    """At most 3 must_state sentences of at most 25 words; fixed, distinct ids."""
    stated = out.get("must_state", [])
    assert len(stated) <= 3, stated
    assert all(len(sentence.split()) <= 25 for sentence in stated), stated
    ids = _ids(out)
    assert len(ids) == len(set(ids)) and set(ids) <= CONTRACT_IDS, ids


@pytest.fixture(autouse=True)
def _scores_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("OLMOEARTH_SCORES_ROOT", str(tmp_path))
    return tmp_path


async def _plan(design: str, n: int = 2000, budget: int = 300) -> dict[str, Any]:
    scores, _truth = _map(n, seed=11)
    return await _result(
        "olmoearth_plan_label_sample",
        {"scores": scores, "budget": budget, "design": design},
    )


def _labels_of(plan: dict[str, Any], n: int = 2000) -> list[int]:
    _scores, truth = _map(n, seed=11)
    idx = json.loads(Path(plan["design_path"]).read_text())["sample"]["indices"]
    return [truth[i] for i in idx]


# --------------------------------------------------------------------------- the module


def test_add_contract_appends_and_repeats_no_claim() -> None:
    out: dict[str, Any] = {"facts": [{"id": "x", "sentence": "s"}]}
    rules.add_contract(
        out,
        facts=[rules.fact("y", "t", n=1)],
        must_state=["a", "a"],
        forbidden_claims=[rules.forbidden("p", "one"), rules.forbidden("p", "two")],
    )
    assert out["facts"] == [
        {"id": "x", "sentence": "s"},
        {"id": "y", "sentence": "t", "n": 1},
    ]
    assert out["must_state"] == ["a"]
    assert out["forbidden_claims"] == [{"id": "p", "why": "one"}]
    empty: dict[str, Any] = {}
    rules.add_contract(empty)
    assert empty == {}  # no key without an entry


def test_every_forbidden_id_is_one_of_the_contracts_fixed_ids() -> None:
    assert set(rules.FIXED_IDS) == CONTRACT_IDS
    assert len(rules.FIXED_IDS) == len(CONTRACT_IDS)
    assert (
        rules.SIMPLE_RANDOM_INTERVAL == "simple_random_interval_for_stratified_design"
    )


@pytest.mark.asyncio
async def test_a_refusal_keeps_its_error_and_carries_the_contract_keys() -> None:
    """A refusal raised as an error stays the same error (text, ok, the
    repeated-failure count) and states its facts on the failed envelope."""

    async def refuse(_args: dict[str, Any], _ctx: ToolContext) -> Any:
        raise rules.refusal(
            "budget 50 is more than the 40 valid windows",
            facts=[rules.fact("unused_labels", "s", n=10)],
        )

    async def plain(_args: dict[str, Any], _ctx: ToolContext) -> Any:
        err = ValueError("no")
        err.contract = {"ok": True, "facts": [{"id": "x"}]}  # type: ignore[attr-defined]
        raise err

    registry = ToolRegistry()
    for name, handler in (("refuse", refuse), ("plain", plain)):
        registry.register(
            RegisteredTool(
                ToolSpec(name=name, description="t", parameters={"type": "object"}),
                handler,
            )
        )
    ctx = _ctx()
    first = await registry.dispatch(ToolCall(id="1", name="refuse", arguments={}), ctx)
    assert first["ok"] is False
    assert first["error"] == "ValueError: budget 50 is more than the 40 valid windows"
    assert first["facts"] == [{"id": "unused_labels", "sentence": "s", "n": 10}]
    assert "must_state" not in first and "forbidden_claims" not in first
    again = await registry.dispatch(ToolCall(id="2", name="refuse", arguments={}), ctx)
    assert again["same_error_count"] == 2 and again["facts"] == first["facts"]
    # Only the contract's keys are taken: a refusal cannot turn itself into ok.
    other = await registry.dispatch(ToolCall(id="3", name="plain", arguments={}), ctx)
    assert other["ok"] is False and other["facts"] == [{"id": "x"}]


def test_percent_keeps_small_rates_readable() -> None:
    assert rules.percent(0.23) == "23.0%"
    assert rules.percent(0.19376942845246936) == "19.4%"
    assert rules.percent(0.0) == "0.0%"
    assert rules.percent(0.0042) == "0.42%"
    assert rules.percent(None) == "n/a"


def test_the_rule_switch_reason_gives_the_p_values_both_rules_need() -> None:
    """exp86 round 7 (B6/files/2) offered a Bonferroni re-run where every
    p_value was at least 0.554: the reason says no level passes either way."""
    failed = [{"p_value": 0.554}, {"p_value": 0.80}, {"p_value": 0.93}]
    why = rules.rule_switch_after_failure(
        certified=False, levels=failed, delta=0.1, rule="prefix"
    )["why"]
    assert "certified nothing" in why and "'prefix'" in why
    assert "smallest p_value of the 3 levels tested is 0.554" in why
    assert "delta = 0.1" in why and "delta/3 = 0.0333" in why
    assert why.endswith("so no level passes under either rule")
    # The prefix rule as the package applies it: from the smallest zone up,
    # while p_value <= delta, stopping at the first failure.
    assert (
        "prefix accepts levels from the smallest zone upward while p_value <= "
        "delta = 0.1 and stops at the first failure (the smallest zone's "
        "p_value is 0.554)" in why
    )
    assert "bonferroni accepts any level with p_value <= delta/3 = 0.0333" in why
    # A p_value under delta at a larger zone passes neither rule here: prefix
    # stopped at the smallest zone (0.3), and 0.05 is above delta/3.
    stopped = rules.rule_switch_after_failure(
        certified=False,
        levels=[{"p_value": 0.3}, {"p_value": 0.05}, {"p_value": 0.9}],
        delta=0.1,
        rule="prefix",
    )["why"]
    assert "(the smallest zone's p_value is 0.3)" in stopped
    assert stopped.endswith("so no level passes under either rule")
    # A p_value under delta: the other rule's outcome is not claimed.
    close = rules.rule_switch_after_failure(
        certified=False,
        levels=[{"p_value": 0.02}, {"p_value": 0.3}],
        delta=0.1,
        rule="bonferroni",
    )["why"]
    assert "0.02" in close and "either rule" not in close
    none = rules.rule_switch_after_failure(
        certified=False, levels=[], delta=0.1, rule="prefix"
    )["why"]
    assert "no level was tested" in none
    general = rules.rule_switch_after_failure(
        certified=None, levels=None, delta=None, rule=None
    )["why"]
    assert "after the labels are seen" in general and "p_value" not in general


# --------------------------------------------------------------------------- plan


@pytest.mark.asyncio
async def test_a_plan_states_its_next_steps_by_design() -> None:
    random_plan = await _plan("random")
    steps = " ".join(random_plan["next_steps"])
    assert "Label every one of the 300 windows" in steps
    assert random_plan["labels_csv_path"] in steps
    assert "olmoearth_estimate_map_error" in steps
    assert "olmoearth_certify_zone" in steps and "fixed now" in steps

    stratified = await _plan("confidence")
    steps = " ".join(stratified["next_steps"])
    assert "not a certification" in steps and "design='random'" in steps
    assert "fixed now" not in steps
    for plan in (random_plan, stratified):
        # The budget fits: no labels are left over, and inline scores have no
        # sampled-grid scope to state.
        assert "must_state" not in plan
        assert plan["budget"] == plan["budget_requested"] == 300
        _holds_to_the_contract(plan)
    assert _ids(stratified) == [
        rules.ERROR_RATE_WITHOUT_LABELS,
        rules.SUBSET_LABELLING_SUFFICIENT,
    ]
    assert "facts" not in stratified
    # A random sheet's first rows are a smaller random sample: no subset claim.
    assert _ids(random_plan) == [rules.ERROR_RATE_WITHOUT_LABELS]
    assert [f["id"] for f in random_plan["facts"]] == ["prefix_is_random_sample"]


def _sheet(plan: dict[str, Any]) -> list[dict[str, str]]:
    with open(plan["labels_csv_path"], encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


@pytest.mark.asyncio
async def test_a_stratified_sheet_lists_its_strata_least_confident_first() -> None:
    """The reason the subset claim gives, checked on the plan's own sheet: the
    rows follow the package's draw, stratum 0 (the least confident) first."""
    for design in ("confidence", "proportional"):
        plan = await _plan(design)
        rows = _sheet(plan)
        drawn = json.loads(Path(plan["design_path"]).read_text())
        assert [int(r["window_index"]) for r in rows] == drawn["sample"]["indices"]
        strata = [int(r["stratum"]) for r in rows]
        assert strata == sorted(strata) and strata[0] == 0
        first = plan["allocation"][0]
        assert set(strata[:first]) == {0}  # the first rows are one stratum
        margin = drawn["population"]["margin"]
        head = [margin[int(r["window_index"])] for r in rows[:first]]
        rest = [margin[int(r["window_index"])] for r in rows[first:]]
        assert max(head) <= min(rest)  # and the least confident windows
        why = _why(plan, rules.SUBSET_LABELLING_SUFFICIENT)
        assert why.startswith(
            "label every window in the sheet; labelling only its first rows "
            "biases the estimate"
        )
        assert f"this {design} design's strata in turn" in why
        assert "the least confident (stratum 0) first" in why
        assert "the strata after them get no labels" in why


@pytest.mark.asyncio
async def test_a_random_sheets_first_rows_are_a_smaller_random_sample() -> None:
    """exp87 build review: a random design's sheet is in the package's random
    draw order, so its first k rows are a simple random sample of size k; the
    plan states that fact instead of forbidding a prefix, and the route it
    names estimates one."""
    plan = await _plan("random")
    rows = _sheet(plan)
    drawn = json.loads(Path(plan["design_path"]).read_text())["sample"]["indices"]
    order = [int(r["window_index"]) for r in rows]
    assert order == drawn and order != sorted(order)
    prefix = _fact(plan, "prefix_is_random_sample")
    assert prefix["n_rows"] == 300
    assert prefix["estimate_prefix_with"] == "window_indices and scores"
    sentence = prefix["sentence"]
    assert "labelling only its first k rows" in sentence
    assert "with k fixed before any label is seen" in sentence
    assert "the estimate stays unbiased and its interval widens" in sentence
    assert "window_indices with the same scores" in sentence
    assert "with design_path it needs every row labelled" in sentence
    # The route works: the first 100 rows, estimated as a random sample.
    scores, truth = _map(2000, seed=11)
    first = order[:100]
    out = await _result(
        "olmoearth_estimate_map_error",
        {
            "window_indices": first,
            "wrong": [truth[i] for i in first],
            "scores": scores,
        },
    )
    assert out["n_labelled"] == 100
    assert out["estimate"] == pytest.approx(sum(truth[i] for i in first) / 100)
    full = await _result(
        "olmoearth_estimate_map_error",
        {"design_path": plan["design_path"], "wrong": _labels_of(plan)},
    )
    assert out["high"] - out["low"] > full["high"] - full["low"]  # it widens


@pytest.mark.asyncio
async def test_a_single_stratum_design_is_a_random_draw_in_effect() -> None:
    """A margin too flat to stratify puts every window in one stratum: the
    package says the draw is random in effect, so its sheet's prefix is too."""
    out = await _result(
        "olmoearth_plan_label_sample",
        {"scores": [[0.3, 0.7]] * 200, "budget": 40, "design": "confidence"},
    )
    assert "random sample in effect" in out["note"]
    assert _ids(out) == [rules.ERROR_RATE_WITHOUT_LABELS]
    assert _fact(out, "prefix_is_random_sample")["n_rows"] == 40


@pytest.mark.asyncio
async def test_a_plan_from_a_scores_file_names_the_file_route(tmp_path: Path) -> None:
    scores, _truth = _map(400, seed=3)
    path = tmp_path / "scores.json"
    path.write_text(json.dumps({"scores": scores}))
    plan = await _result(
        "olmoearth_plan_label_sample",
        {"scores_path": str(path), "budget": 50, "design": "random"},
    )
    prefix = _fact(plan, "prefix_is_random_sample")
    assert prefix["estimate_prefix_with"] == "window_indices and scores_path"
    assert "window_indices with the same scores_path" in prefix["sentence"]


@pytest.mark.asyncio
async def test_a_plan_of_a_regression_band_without_threshold_has_no_error_rate(
    httpx_mock: HTTPXMock,
) -> None:
    """exp86 round 7 (B3/studio): "estimate each map's error rate" for
    KarstNumber, a regression declared 0.2 to 1.2 with no threshold."""
    httpx_mock.add_response(
        url=f"{BASE}/prediction-results/kn",
        json={
            "records": [
                {
                    "id": "kn",
                    "property_names": ["sample_number"],
                    "result_metadata": {
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [[[0, 0], [6, 0], [6, 6], [0, 6], [0, 0]]],
                        },
                        "regression_fields": [
                            {
                                "property_name": "sample_number",
                                "min_value": 0.2,
                                "max_value": 1.2,
                            }
                        ],
                    },
                }
            ]
        },
    )
    async with StudioClient(StudioConfig(api_key="k", base_url=BASE)) as studio:
        ctx = ToolContext(studio=studio, state=ThreadState())
        out = await _result(
            "olmoearth_plan_label_sample", {"result_id": "kn", "budget": 50}, ctx
        )
    assert out["ranked"] is False and out["declared_range"] == [0.2, 1.2]
    assert _ids(out) == [rules.ERROR_RATE_FOR_UNTHRESHOLDED_REGRESSION]
    why = _why(out, rules.ERROR_RATE_FOR_UNTHRESHOLDED_REGRESSION)
    assert why.startswith("'sample_number' (declared range [0.2, 1.2])")
    assert "no error rate" in why and "threshold" in why
    assert "threshold" in out["next_steps"][0]


# --------------------------------------------------------------------------- estimate


@pytest.mark.asyncio
async def test_an_estimate_states_the_whole_map_estimate_and_its_design_rules() -> None:
    plan = await _plan("confidence")
    out = await _result(
        "olmoearth_estimate_map_error",
        {"design_path": plan["design_path"], "wrong": _labels_of(plan)},
    )
    whole = _fact(out, "whole_map_estimate")
    # The output contract's fields, and no others.
    assert set(whole) == {"id", "sentence", "estimate", "low", "high", "design"}
    for key in ("estimate", "low", "high"):
        assert whole[key] == out[key], key
    assert whole["design"] == "confidence"
    assert "warning" not in out and "warns" not in whole["sentence"]
    _holds_to_the_contract(out)
    assert (
        f"estimated at {rules.percent(out['estimate'])} (95% interval "
        f"{rules.percent(out['low'])} to {rules.percent(out['high'])}) from 300 "
        "labelled windows of a population of 2000" in whole["sentence"]
    )
    assert out["method"] in whole["sentence"]
    assert _ids(out) == [
        rules.SIMPLE_RANDOM_INTERVAL,
        rules.CERTIFY_FROM_NONRANDOM_DESIGN,
    ]
    assert "1.96" in _why(out, rules.SIMPLE_RANDOM_INTERVAL)
    assert "'confidence'" in _why(out, rules.CERTIFY_FROM_NONRANDOM_DESIGN)
    steps = " ".join(out["next_steps"])
    assert "No zone can be certified from this confidence design" in steps
    assert "must_state" not in out  # inline scores, not a sampled grid

    random_plan = await _plan("random")
    srs = await _result(
        "olmoearth_estimate_map_error",
        {"design_path": random_plan["design_path"], "wrong": _labels_of(random_plan)},
    )
    assert "forbidden_claims" not in srs
    assert "olmoearth_certify_zone" in " ".join(srs["next_steps"])
    assert "the map's use requires" in " ".join(srs["next_steps"])


@pytest.mark.asyncio
async def test_the_whole_map_estimate_carries_the_packages_warning() -> None:
    """No labelled window wrong: the package warns that the stratified
    interval falls back to the simple-random bound; the fact says so."""
    plan = await _plan("confidence")
    out = await _result(
        "olmoearth_estimate_map_error",
        {"design_path": plan["design_path"], "wrong": [0] * 300},
    )
    assert out["warning"].startswith("no labelled window was wrong")
    sentence = _fact(out, "whole_map_estimate")["sentence"]
    assert f"The package warns: {out['warning'].rstrip('.')}." in sentence
    assert sentence.index("method:") < sentence.index("The package warns")


@pytest.mark.asyncio
async def test_an_estimate_over_a_studio_grid_states_its_scope(tmp_path: Path) -> None:
    rows = [[1 - s, s] for s in (0.1, 0.9, 0.45, 0.8, 0.3, 0.6, 0.05, 0.95)]
    (tmp_path / "s.json").write_text(
        json.dumps(
            {
                "result_id": "kb",
                "grid": [3, 3],
                "windows": [0, 1, 2, 3, 5, 6, 7, 8],
                "scores": rows,
                "score_kind": "binary_score",
            }
        )
    )
    out = await _result(
        "olmoearth_estimate_map_error",
        {
            "window_indices": [0, 3, 6, 8],
            "wrong": [0, 1, 0, 0],
            "scores_path": str(tmp_path / "s.json"),
        },
    )
    assert out["must_state"] == [
        "The rate describes the map at the 8 sampled grid points of a Studio "
        "result (one pixel each), not every pixel of the map."
    ]
    assert _fact(out, "whole_map_estimate")["estimate"] == 0.25
    assert "plan one with olmoearth_plan_label_sample(design='random')" in " ".join(
        out["next_steps"]
    )


def _row(labelled: int, ua: tuple[float, float, float] | None, **kw: Any) -> Any:
    """One class row of the package's per-class table."""
    user = dict(zip(("estimate", "low", "high"), ua)) if ua else None
    return {"n_labelled_map_class": labelled, "user_accuracy": user, **kw}


#: exp86 round 8, B5/files/2: the package's per-class table on F2's
#: confidence design (rows of the confusion matrix are the map's class).
ROUND_8_PER_CLASS = {
    "design": "confidence",
    "nominal_coverage": 0.95,
    "confusion_counts": [
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 2, 0, 0, 7, 0, 0, 0, 0],
        [0, 1, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 110, 0, 0, 3, 0],
        [0, 5, 0, 0, 3, 0, 0, 2, 0],
        [0, 6, 0, 0, 13, 0, 20, 5, 0],
        [0, 3, 0, 0, 3, 0, 3, 111, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0],
    ],
    "per_class": {
        "0": _row(0, None, map_share=0.0006, n_labelled_reference_class=0),
        "1": _row(9, (0.22549, 0.06487, 0.54995), map_share=0.0273),
        "2": _row(2, (0.0, 0.0, 0.65762), map_share=0.0054),
        "3": _row(0, None, map_share=0.0, n_labelled_reference_class=0),
        "4": _row(114, (0.96376, 0.91262, 0.98544), map_share=0.3276),
        "5": _row(10, (0.0, 0.0, 0.27753), map_share=0.0298),
        "6": _row(44, (0.45117, 0.32309, 0.58607), map_share=0.1551),
        "7": _row(120, (0.92281, 0.86281, 0.95785), map_share=0.4507),
        "8": _row(1, (0.0, 0.0, 0.79345), map_share=0.0034),
    },
}


def test_the_weakest_classes_are_ranked_by_code_zero_correct_first() -> None:
    """exp86 round 8 (B5/files/2) named class 1 (22.5%) and class 6 (45.1%)
    the weakest; class 5 had none of its 10 map-labelled windows correct."""
    fact = _weakest_classes(ROUND_8_PER_CLASS)
    assert fact is not None and fact["id"] == "weakest_classes"
    assert fact["accuracy"] == "user's" and fact["min_labelled"] == 5
    assert fact["lowest"] == [5, 1, 6]
    assert [r["class"] for r in fact["ranked"]] == [5, 1, 6, 7, 4]
    assert [(r["correct"], r["labelled"]) for r in fact["ranked"][:3]] == [
        (0, 10),
        (2, 9),
        (20, 44),
    ]
    # Classes on fewer than 5 map-labelled windows are named, never ranked;
    # class 3, in neither the map nor the labels, is not in this map at all.
    assert fact["too_few_to_rank"] == [
        {"class": 0, "labelled": 0, "correct": 0},
        {"class": 2, "labelled": 2, "correct": 0},
        {"class": 8, "labelled": 1, "correct": 0},
    ]
    sentence = fact["sentence"]
    assert sentence.startswith(
        "Ranked by user's accuracy (of the windows the map puts in a class, the "
        "share the labels agree with) over the 5 classes with at least 5 "
        "map-labelled windows, the lowest are class 5 (0 of 10 correct; 0.0%, "
        "95% interval 0.0% to 27.8%), class 1 (2 of 9 correct; 22.5%, 95% "
        "interval 6.5% to 55.0%) and class 6 (20 of 44 correct; 45.1%, 95% "
        "interval 32.3% to 58.6%)."
    )
    assert (
        "Classes 0, 2 and 8 have fewer than 5 map-labelled windows (0, 2 and 1), "
        "too few to rank." in sentence
    )
    # A weighted estimate beside a count (22.5% beside 2 of 9) is explained.
    assert sentence.endswith("so it need not equal correct over labelled.")
    assert "class 3" not in sentence and "Class 3" not in sentence


def test_every_class_with_none_correct_is_named_before_the_next_lowest() -> None:
    """Four ranked classes with none correct are all named: the cap of three
    applies to the next lowest, so none of them is left out as class 5 was."""
    per_class = {
        "design": "random",
        "nominal_coverage": 0.95,
        "confusion_counts": [
            [0, 6, 0, 0, 0, 0],
            [5, 0, 0, 0, 0, 0],
            [0, 0, 0, 7, 0, 0],
            [0, 0, 8, 0, 0, 0],
            [0, 0, 0, 0, 3, 3],
            [0, 0, 0, 0, 0, 30],
        ],
        "per_class": {
            "0": _row(6, (0.0, 0.0, 0.39)),
            "1": _row(5, (0.0, 0.0, 0.45)),
            "2": _row(7, (0.0, 0.0, 0.35)),
            "3": _row(8, (0.0, 0.0, 0.31)),
            "4": _row(6, (0.5, 0.18, 0.82)),
            "5": _row(30, (1.0, 0.89, 1.0)),
        },
    }
    fact = _weakest_classes(per_class)
    assert fact is not None
    # none correct first, the most labelled (narrowest) of them first
    assert fact["lowest"] == [3, 2, 0, 1]
    assert fact["too_few_to_rank"] == []
    assert "Classes" not in fact["sentence"]  # no class too few to rank
    # a random design's user's accuracy is correct over labelled: no weighting
    assert "weighted" not in fact["sentence"]
    one = _weakest_classes(
        {
            "design": "random",
            "confusion_counts": [[3, 1], [0, 4]],
            "per_class": {
                "0": _row(4, (0.75, 0.3, 0.95)),
                "1": _row(4, (1.0, 0.5, 1.0)),
            },
        }
    )
    assert one is not None and one["lowest"] == [] and one["ranked"] == []
    assert one["sentence"].startswith(
        "No class has at least 5 map-labelled windows, so none is ranked by "
        "user's accuracy"
    )
    assert one["sentence"].endswith(
        "Classes 0 and 1 have fewer than 5 map-labelled windows (4 and 4), too "
        "few to rank."
    )
    assert _weakest_classes({"per_class": {}}) is None


@pytest.mark.asyncio
async def test_an_estimate_with_reference_classes_ranks_its_weakest() -> None:
    """Through the tool: a map class every label contradicts comes first, and
    a class only the labels hold is named as too few to rank."""
    plan = await _plan("confidence")
    design = json.loads(Path(plan["design_path"]).read_text())
    idx, mc = design["sample"]["indices"], design["population"]["map_class"]
    wrong = _labels_of(plan)
    # map class 2 is always wrong; three windows are a class the map never has
    reference = [
        (mc[i] + 1) % 3 if (w or mc[i] == 2) else mc[i] for i, w in zip(idx, wrong)
    ]
    reference[:3] = [3, 3, 3]
    wrong = [int(r != mc[i]) for i, r in zip(idx, reference)]
    out = await _result(
        "olmoearth_estimate_map_error",
        {
            "design_path": plan["design_path"],
            "wrong": wrong,
            "reference": reference,
            "n_classes": 4,
        },
    )
    fact = _fact(out, "weakest_classes")
    assert fact["lowest"][0] == 2 and fact["ranked"][0]["correct"] == 0
    confusion = out["per_class"]["confusion_counts"]
    for r in fact["ranked"]:
        assert r["correct"] == confusion[r["class"]][r["class"]]
        assert r["labelled"] == sum(confusion[r["class"]])
        ua = out["per_class"]["per_class"][str(r["class"])]["user_accuracy"]
        assert (r["estimate"], r["low"], r["high"]) == (
            ua["estimate"],
            ua["low"],
            ua["high"],
        )
    assert fact["too_few_to_rank"] == [{"class": 3, "labelled": 0, "correct": 0}]
    assert any("weakest_classes" in s for s in out["next_steps"])
    _holds_to_the_contract(out)
    # Without reference classes there is no table, and no ranking.
    plain = await _result(
        "olmoearth_estimate_map_error",
        {"design_path": plan["design_path"], "wrong": _labels_of(plan)},
    )
    assert [f["id"] for f in plain["facts"]] == ["whole_map_estimate"]
    assert not any("weakest_classes" in s for s in plain["next_steps"])


# --------------------------------------------------------------------------- certify


@pytest.mark.asyncio
async def test_nothing_certified_says_what_follows_and_what_does_not() -> None:
    """exp86 rounds 2 to 7 (brief 6): a looser alpha or another rule was
    offered after nothing certified, and "~23%" was worked out as 69/300."""
    plan = await _plan("random")
    wrong = _labels_of(plan)
    out = await _result(
        "olmoearth_certify_zone",
        {"design_path": plan["design_path"], "wrong": wrong, "alpha": 0.05},
    )
    assert out["certified"] is False
    steps = out["next_steps"]
    assert steps[0].startswith("Label more windows under a NEW random design")
    assert "design='random'" in steps[0] and "larger budget" in steps[0]
    assert steps[1].startswith("Or report the whole-map estimate and interval")
    assert steps[2].startswith("Not a looser alpha and not another rule")
    assert _ids(out) == [rules.POST_HOC_ALPHA, rules.RULE_SWITCH_AFTER_FAILURE]
    assert "alpha=0.05 was fixed for this call" in _why(out, rules.POST_HOC_ALPHA)
    switch = _why(out, rules.RULE_SWITCH_AFTER_FAILURE)
    levels = out["levels"]
    smallest = min(lv["p_value"] for lv in levels)
    assert f"is {smallest:.3g}" in switch
    assert f"delta/{len(levels)}" in switch
    neither = levels[0]["p_value"] > 0.1 and smallest > 0.1 / len(levels)
    assert ("no level passes under either rule" in switch) == neither
    _holds_to_the_contract(out)
    # The whole-map estimate is the package's, on the same labels.
    whole = _fact(out, "whole_map_estimate")
    direct = estimate.estimate_error_rate(
        json.loads(Path(plan["design_path"]).read_text())["sample"], wrong
    )
    assert whole["estimate"] == pytest.approx(direct["estimate"])
    assert whole["estimate"] == pytest.approx(sum(wrong) / len(wrong))  # a random draw
    assert (whole["low"], whole["high"]) == pytest.approx(
        (direct["low"], direct["high"])
    )
    assert "not a certification" in whole["sentence"]
    assert "must_state" not in out


@pytest.mark.asyncio
async def test_a_certified_zone_states_that_nothing_outside_it_is() -> None:
    plan = await _plan("random")
    out = await _result(
        "olmoearth_certify_zone",
        {
            "design_path": plan["design_path"],
            "wrong": _labels_of(plan),
            "alpha": 0.1,
            "rule": "bonferroni",
        },
    )
    assert out["certified"] is True
    assert out["must_state"] == [
        f"Only the {out['n_zone']} most confident windows (coverage "
        f"{out['coverage']:g}) are certified; nothing outside the zone is."
    ]
    assert out["next_steps"][0].startswith("Report the zone as returned (certified")
    assert "zone_path" in out["next_steps"][0]
    assert "Nothing outside the zone is certified" in out["next_steps"][1]
    assert _ids(out) == [rules.POST_HOC_ALPHA, rules.RULE_SWITCH_AFTER_FAILURE]
    assert "to get a larger zone" in _why(out, rules.RULE_SWITCH_AFTER_FAILURE)
    _holds_to_the_contract(out)


@pytest.mark.asyncio
async def test_a_certification_over_a_studio_grid_states_its_scope(
    tmp_path: Path,
) -> None:
    """A zone and a rate over a Studio result's grid points carry the scope a
    plan and an estimate of that grid carry, as the same sentence."""
    scores, truth = _map(256, seed=5)
    rows = [[1 - s, s] for s in (min(max(r[0] / 8 + 0.5, 0.0), 1.0) for r in scores)]
    (tmp_path / "grid.json").write_text(
        json.dumps(
            {
                "result_id": "kb",
                "grid": [16, 16],
                "scores": rows,
                "score_kind": "binary_score",
            }
        )
    )
    plan = await _result(
        "olmoearth_plan_label_sample",
        {"scores_path": str(tmp_path / "grid.json"), "budget": 120, "design": "random"},
    )
    idx = json.loads(Path(plan["design_path"]).read_text())["sample"]["indices"]
    scope = (
        "The rate describes the map at the 256 sampled grid points of a Studio "
        "result (one pixel each), not every pixel of the map."
    )
    assert plan["must_state"] == [scope]
    for alpha, rule in ((0.05, "prefix"), (0.5, "bonferroni")):
        out = await _result(
            "olmoearth_certify_zone",
            {
                "design_path": plan["design_path"],
                "wrong": [truth[i] for i in idx],
                "alpha": alpha,
                "rule": rule,
            },
        )
        assert out["must_state"][-1] == scope
        assert len(out["must_state"]) == (2 if out["certified"] else 1)
        _holds_to_the_contract(out)


@pytest.mark.asyncio
async def test_a_refused_certification_names_the_design_and_the_estimate() -> None:
    """exp86 round 6 (B5/files/1 and 3) offered to certify from a confidence
    design; the refusal says why, and what the labels can give instead."""
    plan = await _plan("confidence")
    out = await _result(
        "olmoearth_certify_zone",
        {"design_path": plan["design_path"], "wrong": [0] * 300, "alpha": 0.05},
    )
    assert out["certified"] is False and out["design"] == "confidence"
    assert "reason" in out  # the trial's parity check reads a refusal by it
    assert _ids(out) == [
        rules.CERTIFY_FROM_NONRANDOM_DESIGN,
        rules.POST_HOC_ALPHA,
        rules.RULE_SWITCH_AFTER_FAILURE,
    ]
    assert "'confidence'" in _why(out, rules.CERTIFY_FROM_NONRANDOM_DESIGN)
    assert "alpha=0.05" in _why(out, rules.POST_HOC_ALPHA)
    assert "olmoearth_estimate_map_error" in out["next_steps"][0]
    assert "design='random'" in out["next_steps"][1]


@pytest.mark.asyncio
async def test_the_rules_change_no_number_the_package_returns() -> None:
    """The contract keys are added beside the package's numbers, never in place."""
    scores, truth = _map(2000, seed=11)
    plan = await _plan("random")
    idx = json.loads(Path(plan["design_path"]).read_text())["sample"]["indices"]
    wrong = [truth[i] for i in idx]
    out = await _result(
        "olmoearth_certify_zone",
        {"design_path": plan["design_path"], "wrong": wrong, "alpha": 0.25},
    )
    direct = estimate.certify_zone(margins(scores), idx, wrong, 0.25)
    direct.pop("zone_indices_in_order", None)
    expected = _jsonable(direct)
    assert {key: out[key] for key in expected} == expected
    assert out["certified"] is (expected["coverage"] is not None)
