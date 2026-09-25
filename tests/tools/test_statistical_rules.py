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

import json
import random
from pathlib import Path
from typing import Any

import pytest
from pytest_httpx import HTTPXMock

from olmoearth_agent.analysis.review_set import margins
from olmoearth_agent.harness.state import ThreadState
from olmoearth_agent.llm.types import ToolCall
from olmoearth_agent.studio.client import StudioClient, StudioConfig
from olmoearth_agent.tools import statistical_rules as rules
from olmoearth_agent.tools.estimation import _jsonable, build_estimation_tools
from olmoearth_agent.tools.registry import ToolContext, ToolRegistry
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
        assert _ids(plan) == [
            rules.ERROR_RATE_WITHOUT_LABELS,
            rules.SUBSET_LABELLING_SUFFICIENT,
        ]
        why = _why(plan, rules.SUBSET_LABELLING_SUFFICIENT)
        assert why.startswith(
            "label every window in the sheet; labelling only the first rows "
            "biases the estimate"
        )
        # The budget fits: no labels are left over, and inline scores have no
        # sampled-grid scope to state.
        assert "facts" not in plan and "must_state" not in plan
        assert plan["budget"] == plan["budget_requested"] == 300
    assert "stratum by stratum" in _why(stratified, rules.SUBSET_LABELLING_SUFFICIENT)
    assert "stratum" not in _why(random_plan, rules.SUBSET_LABELLING_SUFFICIENT)


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
    for key in ("estimate", "low", "high", "method", "n_labelled", "n_population"):
        assert whole[key] == out[key], key
    assert whole["design"] == "confidence"
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
    assert ("no level passes under either rule" in switch) == (smallest > 0.1)
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
