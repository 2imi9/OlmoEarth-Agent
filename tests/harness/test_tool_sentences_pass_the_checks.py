# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""An answer made of the tools' own sentences passes every answer check.

The contract between the tools and the harness: a result's ``facts``,
``must_state``, notes (``which_side_is_right``, ``reading``,
``listing_note``, ``listing_order``, ``evidence_scope``) and
``next_steps`` are what an answer repeats, so no check may flag them while
the same result forbids its claims. The review of branch 2imi9/fix-r8 found
the tools' correlation fact flagged as ``agreement_from_uncertain_correlation``
and their dates must_state as ``one_reference_settles_two_dates``; an answer
that followed the tool would have been rewritten and marked.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import pytest

from olmoearth_agent.harness.checks import (
    FORBIDDEN,
    NUMBERS,
    RunEvidence,
    ToolRecord,
    run_checks,
)
from olmoearth_agent.harness.state import ThreadState
from olmoearth_agent.llm.types import ToolCall
from olmoearth_agent.tools import statistical_rules as rules
from olmoearth_agent.tools.compare import _correlation_contract
from olmoearth_agent.tools.estimation import build_estimation_tools
from olmoearth_agent.tools.registry import ToolContext, ToolRegistry
from olmoearth_agent.tools.review_set import build_review_set_tools

#: The keys whose sentences an answer repeats.
_NOTES = (
    "which_side_is_right",
    "reading",
    "listing_note",
    "listing_order",
    "evidence_scope",
    "caveats",
    "next_steps",
    "design_note",
    "design_requirement",
    "not_for_these_maps",
    "strata_meaning",
)


@pytest.fixture(autouse=True)
def _workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("OLMOEARTH_OUTPUT_ROOT", str(tmp_path))
    monkeypatch.setenv("OLMOEARTH_SCORES_ROOT", str(tmp_path))
    return tmp_path


def _notes(value: Any, depth: int = 0) -> list[str]:
    out: list[str] = []
    if isinstance(value, dict) and depth < 4:
        for key, item in value.items():
            if key in ("forbidden_claims", "facts", "must_state"):
                continue
            if key in _NOTES and isinstance(item, str):
                out.append(item)
            elif key in _NOTES and isinstance(item, list):
                out += [x for x in item if isinstance(x, str)]
            else:
                out += _notes(item, depth + 1)
    elif isinstance(value, list) and depth < 4:
        for item in value:
            out += _notes(item, depth + 1)
    return out


def _said(result: dict[str, Any]) -> list[str]:
    """Every sentence of a result an answer may repeat."""
    facts = [f["sentence"] for f in result.get("facts", []) if f.get("sentence")]
    return facts + list(result.get("must_state", [])) + _notes(result)


def _assert_passes(
    name: str, arguments: dict[str, Any], result: dict[str, Any]
) -> None:
    said = _said(result)
    assert said, result
    run = RunEvidence(
        tools=[ToolRecord(name, arguments, {"ok": True, "result": result})],
        user_messages=["Which map is right, and where do they differ?"],
    )
    found = run_checks("\n\n".join(said), run)
    assert found == {}, json.dumps(found, indent=1)


async def _call(name: str, args: dict[str, Any]) -> dict[str, Any]:
    registry = ToolRegistry()
    registry.register_all(build_review_set_tools())
    registry.register_all(build_estimation_tools())
    out = await registry.dispatch(
        ToolCall(id="1", name=name, arguments=args),
        ToolContext(studio=None, state=ThreadState()),  # type: ignore[arg-type]
    )
    assert out["ok"] is True, out
    return out["result"]


@pytest.mark.parametrize(
    "found",
    [
        # exp86 round 8, B3/studio: r = -0.0172 over 25 cells
        [(-0.0172, 25, None)],
        [(0.62, 144, None)],
        [(-0.55, 144, None)],
        [(0.3, 3, None)],
        # a group: one pair's sign known, another's not
        [(0.62, 144, ("a1", "b1")), (0.05, 144, ("a1", "c1")), (0.1, 25, ("b1", "c1"))],
    ],
)
def test_the_correlation_facts_and_must_state_pass(
    found: list[tuple[float, int, tuple[str, str] | None]],
) -> None:
    facts, must_state, claims, next_steps = _correlation_contract(
        found, mode="group" if found[0][2] else "pair", grid=6
    )
    result = {
        "facts": facts,
        "must_state": must_state,
        "next_steps": next_steps,
        "forbidden_claims": claims,
    }
    ids = {c["id"] for c in claims}
    assert rules.SPATIAL_PATTERN_FROM_ONE_CORRELATION in ids
    _assert_passes("olmoearth_compare_results", {}, result)


@pytest.mark.asyncio
@pytest.mark.parametrize("labels_date", [None, "2023-06-01"])
async def test_a_comparison_of_two_dated_maps_passes(labels_date: str | None) -> None:
    """exp86 round 8, B3/cluster and B7/files: the dates must_state and
    which_side_is_right, beside one_reference_settles_two_dates."""
    pytest.importorskip("oe_inferencex.compare")
    rng = random.Random(5)
    a = [[rng.random(), rng.random(), rng.random()] for _ in range(64)]
    b = [row if i % 5 else row[::-1] for i, row in enumerate(a)]
    args: dict[str, Any] = {
        "scores_a": a,
        "scores_b": b,
        "grid": [8, 8],
        "date_a": "2023-01-01/2023-12-31",
        "date_b": "2022-01-01/2022-12-31",
        "max_listed": 5,
    }
    if labels_date:
        args["labels_date"] = labels_date
    out = await _call("olmoearth_compare_review", args)
    ids = {c["id"] for c in out["forbidden_claims"]}
    assert {rules.ONE_REFERENCE_SETTLES_TWO_DATES, "evidence_outside_its_scope"} <= ids
    assert rules.DATED_MAPS_MUST_STATE in out["must_state"]
    _assert_passes("olmoearth_compare_review", args, out)


@pytest.mark.asyncio
async def test_a_review_set_of_multiclass_logits_listed_short_passes(
    tmp_path: Path,
) -> None:
    """exp86 round 8, B8/cluster: an OlmoEarth model's 9-class logits, covered
    by the ranking evidence only in part, listed short with its full list
    saved to a CSV."""
    rng = random.Random(8)
    rows = [[rng.gauss(0.0, 2.0) for _ in range(9)] for _ in range(400)]
    (tmp_path / "s.json").write_text(
        json.dumps(
            {
                "grid": [20, 20],
                "scores": rows,
                "score_type": "logit",
                "model": {"repo": "allenai/OlmoEarth-v1-FT-AWF-Base"},
            }
        )
    )
    args = {"scores_path": str(tmp_path / "s.json"), "budget": 0.1, "max_listed": 12}
    out = await _call("olmoearth_review_set", args)
    assert out["evidence_covers_this_case"] == "in part"
    assert "evidence_outside_its_scope" in {c["id"] for c in out["forbidden_claims"]}
    assert out["review_list_rows"] == 40
    _assert_passes("olmoearth_review_set", args, out)


#: AWF's classes, as a scores file from olmoearth_scores_from_file names them.
_AWF = {
    "0": "woodland_forest",
    "1": "open_water",
    "2": "shrubland_savanna",
    "3": "herbaceous_wetland",
    "4": "grassland_barren",
    "5": "agriculture_settlement",
    "6": "montane_forest",
    "7": "lava_forest",
    "8": "urban_dense_development",
}


@pytest.mark.asyncio
async def test_a_review_sets_class_counts_and_its_list_file_pass(
    tmp_path: Path,
) -> None:
    """exp86 round 9, B8/cluster: the classes of the whole review set
    (review_set_classes) and the sentence naming its CSV (list_file), beside
    margin_as_error_probability and the other claims of a ranking."""
    rng = random.Random(9)
    rows = [[rng.gauss(0.0, 2.0) for _ in range(9)] for _ in range(400)]
    (tmp_path / "awf.json").write_text(
        json.dumps(
            {
                "grid": [20, 20],
                "scores": rows,
                "map_class": [max(range(9), key=r.__getitem__) for r in rows],
                "classes": _AWF,
                "score_type": "logit",
                "model": {"repo": "allenai/OlmoEarth-v1-FT-AWF-Base"},
            }
        )
    )
    args = {"scores_path": str(tmp_path / "awf.json"), "budget": 0.1, "max_listed": 10}
    out = await _call("olmoearth_review_set", args)
    assert {"margin_ratio", "review_set_classes", "list_file"} <= {
        f["id"] for f in out["facts"]
    }
    assert rules.MARGIN_AS_ERROR_PROBABILITY in {
        c["id"] for c in out["forbidden_claims"]
    }
    _assert_passes("olmoearth_review_set", args, out)


@pytest.mark.asyncio
async def test_a_review_set_of_a_studio_top_k_scores_file_passes(
    tmp_path: Path,
) -> None:
    """live3 (31 October 2026): a scores file pooled from a Studio class map's
    own files, scoped as a Studio model's scores (no recorded experiment
    covers them) and stating Studio's uncalibrated scores in must_state."""
    rng = random.Random(10)
    confidence = [round(rng.uniform(0.5, 1.0), 6) for _ in range(64)]
    classes = [i % 2 for i in range(64)]
    rows = [[c, 0.0] if k == 0 else [0.0, c] for c, k in zip(confidence, classes)]
    assumption = (
        "The scores are Studio's uncalibrated softmax scores, quantized to 1/254 "
        "and decoded as (value - 1) / 254, the files README's formula: they rank "
        "windows and are not probabilities that a class is right."
    )
    (tmp_path / "kelp.json").write_text(
        json.dumps(
            {
                "grid": [8, 8],
                "score_kind": "window_confidence",
                "score_type": "probability",
                "scores": rows,
                "p1": confidence,
                "map_class": classes,
                "classes": {"0": "absent", "1": "present"},
                "model": {"repo": None, "id": "model-kelp-test"},
                "studio": {"kind": "top_k", "k": 2, "assumption": assumption},
                "assumption": assumption,
                "warnings": [assumption],
            }
        )
    )
    args = {"scores_path": str(tmp_path / "kelp.json"), "budget": 0.1}
    out = await _call("olmoearth_review_set", args)
    assert out["evidence_covers_this_case"] == "no"
    assert rules.STUDIO_TOPK_SCORES_MUST_STATE in out["must_state"]
    _assert_passes("olmoearth_review_set", args, out)


@pytest.mark.parametrize("n_models", [1, 2])
def test_the_labels_in_studio_fact_passes_beside_a_comparisons_claims(
    n_models: int,
) -> None:
    """exp86 round 9, B3/studio: the models' label fields, beside the claims a
    comparison of two Studio results forbids."""
    split = {"train_prop": 0.75, "val_prop": 0.25, "test_prop": 0.0}
    models = [
        {"name": n, "model_id": f"m{n}", "trained_on_labels": True, "split": split}
        for n in ("KarstBinary", "KarstNumber")[:n_models]
    ]
    band = [("sample_number", (0.2, 1.2))]
    result = rules.add_contract(
        {},
        facts=[rules.labels_in_studio_fact(models)],
        forbidden_claims=[
            rules.winner_without_labels(),
            rules.labelling_low_confidence_only(),
            rules.unthresholded_regression(band),
            rules.review_set_for_unthresholded_regression(band),
            rules.error_rate_without_labels(),
        ],
    )
    _assert_passes("olmoearth_compare_results", {}, result)


def _map(n: int = 2000, seed: int = 11) -> list[list[float]]:
    rng = random.Random(seed)
    return [[rng.gauss(0.0, 1.5) for _ in range(3)] for _ in range(n)]


@pytest.mark.asyncio
@pytest.mark.parametrize("design", ["random", "confidence"])
async def test_a_plan_its_estimate_and_its_certification_pass(design: str) -> None:
    """The estimation tools' next steps say a random design makes a certified
    zone possible, not certain (certification_guaranteed); the weakest
    classes are ranked in code."""
    plan_args = {"scores": _map(), "budget": 300, "design": design}
    plan = await _call("olmoearth_plan_label_sample", plan_args)
    _assert_passes("olmoearth_plan_label_sample", plan_args, plan)
    sample = json.loads(Path(plan["design_path"]).read_text())
    idx = sample["sample"]["indices"]
    mc = sample["population"]["map_class"]
    reference = [mc[i] if k % 7 else (mc[i] + 1) % 3 for k, i in enumerate(idx)]
    wrong = [int(r != mc[i]) for r, i in zip(reference, idx)]
    est_args = {
        "design_path": plan["design_path"],
        "wrong": wrong,
        "reference": reference,
        "n_classes": 3,
    }
    estimate = await _call("olmoearth_estimate_map_error", est_args)
    assert "weakest_classes" in {f["id"] for f in estimate["facts"]}
    _assert_passes("olmoearth_estimate_map_error", est_args, estimate)
    cert_args = {"design_path": plan["design_path"], "wrong": wrong, "alpha": 0.05}
    certified = await _call("olmoearth_certify_zone", cert_args)
    _assert_passes("olmoearth_certify_zone", cert_args, certified)


@pytest.mark.asyncio
async def test_agent_trial_12s_fixes_in_the_tools_own_words_pass() -> None:
    """The class plan's per-stratum note, the user_vs_producer fact and its
    must_state, and a zone call that certified nothing (its whole-map
    estimate naming the smallest zone tested, beside
    whole_map_rate_for_a_zone): an answer made of them is not flagged."""
    pytest.importorskip("oe_inferencex.estimate")
    # The map calls 1,200 windows class 0, 400 class 1 and 400 class 2; 300
    # of the windows it calls 2 are class 1 on the ground.
    map_class = [0] * 1200 + [1] * 400 + [2] * 400
    truth = [0] * 1200 + [1] * 400 + [2 if i % 4 == 0 else 1 for i in range(400)]
    rows = [[3.0 if k == c else 0.0 for k in range(3)] for c in map_class]
    plan_args = {"scores": rows, "budget": 300, "design": "class"}
    plan = await _call("olmoearth_plan_label_sample", plan_args)
    assert "per_stratum" in plan
    _assert_passes("olmoearth_plan_label_sample", plan_args, plan)
    idx = json.loads(Path(plan["design_path"]).read_text())["sample"]["indices"]
    reference = [truth[i] for i in idx]
    est_args = {
        "design_path": plan["design_path"],
        "wrong": [int(r != map_class[i]) for r, i in zip(reference, idx)],
        "reference": reference,
        "n_classes": 3,
    }
    estimate = await _call("olmoearth_estimate_map_error", est_args)
    assert "user_vs_producer" in {f["id"] for f in estimate["facts"]}
    _assert_passes("olmoearth_estimate_map_error", est_args, estimate)
    random_args = {"scores": _map(), "budget": 300, "design": "random"}
    random_plan = await _call("olmoearth_plan_label_sample", random_args)
    _assert_passes("olmoearth_plan_label_sample", random_args, random_plan)
    cert_args = {
        "design_path": random_plan["design_path"],
        "wrong": [int(k % 3 == 0) for k in range(300)],
        "alpha": 0.05,
    }
    nothing = await _call("olmoearth_certify_zone", cert_args)
    assert nothing["certified"] is False
    ids = {c["id"] for c in nothing["forbidden_claims"]}
    assert rules.WHOLE_MAP_RATE_FOR_A_ZONE in ids
    _assert_passes("olmoearth_certify_zone", cert_args, nothing)


@pytest.mark.asyncio
@pytest.mark.parametrize("design", ["random", "confidence"])
async def test_a_rate_measured_after_its_plan_is_no_false_alarm(design: str) -> None:
    """Every plan forbids error_rate_without_labels, and a run that planned and
    then graded the labels holds both: the estimate's own figure, stated
    plainly, passes the number check and the forbidden claims, while the
    plan alone still forbids it, and a rate the estimate did not give is the
    number check's to flag."""
    plan_args = {"scores": _map(), "budget": 300, "design": design}
    plan = await _call("olmoearth_plan_label_sample", plan_args)
    idx = json.loads(Path(plan["design_path"]).read_text())["sample"]["indices"]
    wrong = [int(i % 5 == 0) for i in idx]
    est_args = {"design_path": plan["design_path"], "wrong": wrong}
    estimate = await _call("olmoearth_estimate_map_error", est_args)
    valid = estimate["valid_interval"]
    own = (
        f"The map's error rate is {rules.percent(valid['estimate'])} "
        f"({rules.percent(valid['low'])} to {rules.percent(valid['high'])})."
    )
    planned = ToolRecord(
        "olmoearth_plan_label_sample", plan_args, {"ok": True, "result": plan}
    )
    graded = ToolRecord(
        "olmoearth_estimate_map_error", est_args, {"ok": True, "result": estimate}
    )
    asked = ["How wrong is this map? I can label 300 windows."]
    plan_only = RunEvidence(tools=[planned], user_messages=asked)
    assert set(run_checks(own, plan_only, [FORBIDDEN])) == {FORBIDDEN}
    measured = RunEvidence(tools=[planned, graded], user_messages=asked)
    assert run_checks(own, measured, [NUMBERS, FORBIDDEN]) == {}
    # Two decimals: no share of the run's results rounds to all three.
    made_up = "The map's error rate is 71.37% (68.21% to 74.43%)."
    found = run_checks(made_up, measured, [NUMBERS, FORBIDDEN])
    assert set(found) == {NUMBERS}
    assert all(v["check"] == NUMBERS for v in found[NUMBERS])


def _passes(
    name: str, arguments: dict[str, Any], result: dict[str, Any], said: list[str]
) -> None:
    """``said`` as the answer to a run of one call: no check flags it."""
    run = RunEvidence(
        tools=[ToolRecord(name, arguments, {"ok": True, "result": result})],
        user_messages=["Which part of this map can I trust, and how do you know?"],
    )
    found = run_checks("\n\n".join(said), run)
    assert found == {}, json.dumps(found, indent=1)


@pytest.mark.asyncio
async def test_a_certified_zones_note_on_what_delta_counts_passes() -> None:
    """olmoearth-inferencex 1.20.3's ISSUED_NOTE, which a certified zone's
    meaning ends with: an answer that repeats it beside the zone's must_state
    is not flagged (certification_guaranteed and post_hoc_alpha included)."""
    zone_module = pytest.importorskip("oe_inferencex.certify.zone")
    plan = await _call(
        "olmoearth_plan_label_sample",
        {"scores": _map(), "budget": 300, "design": "random"},
    )
    args = {"design_path": plan["design_path"], "wrong": [0] * 300, "alpha": 0.1}
    out = await _call("olmoearth_certify_zone", args)
    assert out["certified"] is True
    assert out["meaning"].endswith(zone_module.ISSUED_NOTE)
    _passes(
        "olmoearth_certify_zone",
        args,
        out,
        list(out.get("must_state", [])) + [zone_module.ISSUED_NOTE],
    )


@pytest.mark.asyncio
async def test_the_suites_evidence_scope_passes_where_it_covers_the_case(
    tmp_path: Path,
) -> None:
    """The scope sentence of a case the suite covers in kind (an OlmoEarth
    model's two-class probability margin) states the margin against a random
    order and the controls' near chance (the red team of 30 September 2026);
    an answer that repeats it with the ranking's facts is not flagged."""
    rng = random.Random(12)
    rows = [[p, 1.0 - p] for p in (rng.random() for _ in range(400))]
    (tmp_path / "two.json").write_text(
        json.dumps(
            {
                "grid": [20, 20],
                "scores": rows,
                "score_type": "probability",
                "model": {"repo": "allenai/OlmoEarth-v1-Base"},
            }
        )
    )
    args = {"scores_path": str(tmp_path / "two.json"), "budget": 0.1, "max_listed": 10}
    out = await _call("olmoearth_review_set", args)
    assert out["evidence_covers_this_case"] == "yes"
    scope = out["evidence_scope"]
    assert "random order" in scope and "near chance" in scope
    facts = [f["sentence"] for f in out.get("facts", []) if f.get("sentence")]
    said = facts + list(out.get("must_state", [])) + [scope]
    _passes("olmoearth_review_set", args, out, said)
