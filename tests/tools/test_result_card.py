# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""The result card: the few rows a panel shows above a result's figure.

Each tool writes its card from its own result (``statistical_rules.add_card``)
with the helpers its ``must_state`` uses, so a card's number is the
sentence's own text. These hold each tool's row names and their order, and
each value to the result's own fields, its ``must_state``, its verdict or its
fact. The MCP App leaves the card out of the copy the model reads
(``tests/mcp_app/test_mcp_server.py``). These run the real package
(``uv sync --all-extras``).
"""

from __future__ import annotations

import csv
import json
import math
import os
import random
import re
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import httpx
import pytest
from pytest_httpx import HTTPXMock

from olmoearth_agent.harness.state import ThreadState
from olmoearth_agent.llm.types import ToolCall
from olmoearth_agent.studio.client import StudioClient, StudioConfig
from olmoearth_agent.tools import statistical_rules as rules
from olmoearth_agent.tools.estimation import (
    NO_DESIGN,
    POINTS_TO_LABEL,
    SQUARES_TO_LABEL,
    _plan_card,
    _to_label_row_name,
    _two_map_plan_card,
    build_estimation_tools,
)
from olmoearth_agent.tools.registry import ToolContext, ToolRegistry
from olmoearth_agent.tools.review_set import (
    BOUNDARY_FIRST_RANKING,
    build_review_set_tools,
)

estimate = pytest.importorskip("oe_inferencex.estimate")

PLAN = "olmoearth_plan_label_sample"
GRADE = "olmoearth_estimate_map_error"
CERTIFY = "olmoearth_certify_zone"
BASE = "http://mock-studio/api/v1"


@pytest.fixture(autouse=True)
def _roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("OLMOEARTH_SCORES_ROOT", str(tmp_path))
    monkeypatch.setenv("OLMOEARTH_OUTPUT_ROOT", str(tmp_path))
    return tmp_path


def _registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register_all(build_estimation_tools())
    registry.register_all(build_review_set_tools())
    return registry


async def _ok(
    name: str, args: dict[str, Any], ctx: ToolContext | None = None
) -> dict[str, Any]:
    ctx = ctx or ToolContext(studio=None, state=ThreadState())  # type: ignore[arg-type]
    out = await _registry().dispatch(ToolCall(id="1", name=name, arguments=args), ctx)
    assert out["ok"] is True, out
    return dict(out["result"])


def _names(result: dict[str, Any]) -> list[str]:
    return [row[0] for row in result[rules.CARD_KEY]]


def _row(result: dict[str, Any], name: str) -> list[str]:
    (row,) = [r for r in result[rules.CARD_KEY] if r[0] == name]
    return list(row)


def _span(low: Any, high: Any) -> str:
    return f"{rules.percent(low)} to {rules.percent(high)}"


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


def _indices(plan: dict[str, Any]) -> list[int]:
    design = json.loads(Path(plan["design_path"]).read_text(encoding="utf-8"))
    return [int(i) for i in design["sample"]["indices"]]


def _fact(result: dict[str, Any], fact_id: str) -> str:
    (fact,) = [f for f in result["facts"] if f["id"] == fact_id]
    return str(fact["sentence"])


# --------------------------------------------------------------------------- one map


@pytest.mark.asyncio
async def test_a_plan_and_its_estimate_card_the_numbers_their_sentences_state() -> None:
    scores, truth = _map()
    plan = await _ok(PLAN, {"scores": scores, "budget": 300})
    # A scores file's windows are squares (r4h V7); "Points to label" is a
    # Studio result's sampled grid only.
    assert _names(plan) == ["Squares to label", "Design", "Population", "Seed"]
    assert _row(plan, "Squares to label") == ["Squares to label", "300"]
    assert _row(plan, "Design") == ["Design", "confidence"]
    assert _row(plan, "Population") == [
        "Population",
        f"{plan['n_population']} windows",
        plan["population"]["what_the_rate_will_be_of"],
    ]
    assert _row(plan, "Seed") == ["Seed", "0"]

    wrong = [truth[i] for i in _indices(plan)]
    out = await _ok(GRADE, {"design_path": plan["design_path"], "wrong": wrong})
    valid = out["valid_interval"]
    assert _names(out) == [
        "Error rate",
        "95% interval valid at every sample size",
        "Interval this design earns",
        "Labels",
        "Design",
    ]
    stated = out["must_state"][0]
    rate = _row(out, "Error rate")[1]
    assert rate == rules.percent(valid["estimate"]) and rate in stated
    first = _row(out, "95% interval valid at every sample size")
    assert first[1:] == [_span(valid["low"], valid["high"]), valid["method"]]
    assert first[1] in stated
    # The confidence design's own interval is nominal (Wilson's on the
    # effective sample size): named as the fact names it, with no level, and
    # "nominal 95%" only in its detail.
    second = _row(out, "Interval this design earns")
    assert second[1:] == [
        _span(out["low"], out["high"]),
        f"nominal 95%, approximate; {out['method']}",
    ]
    assert second[1] in _fact(out, "whole_map_estimate")
    assert "the interval this design earns is" in _fact(out, "whole_map_estimate")
    assert not any(row[0].startswith("95% interval this") for row in out["card"])
    assert _row(out, "Labels")[1] == f"300 of {out['n_population']} windows"
    assert _row(out, "Design")[1] == "confidence" and "confidence design" in stated


def test_a_plan_cards_first_row_names_points_only_for_a_sampled_grid() -> None:
    """r4h V7: "Points to label" when the plan carries a Studio result's
    sampling block (a dict), "Squares to label" otherwise, on the one-map and
    the which-map plan cards alike; never "Windows to label"."""
    assert POINTS_TO_LABEL == "Points to label"
    assert SQUARES_TO_LABEL == "Squares to label"
    grid = {"grid": "16x16", "n_valid": 171}
    assert _to_label_row_name({"sampling": grid}) == "Points to label"
    for out in ({}, {"sampling": None}, {"sampling": "16x16"}):
        assert _to_label_row_name(out) == "Squares to label"
    base: dict[str, Any] = {
        "budget": 40,
        "budget_requested": 50,
        "n_disagree": 12,
        "n_population": 63,
        "seed": 3,
    }
    two = _two_map_plan_card({**base, "sampling": grid})
    assert two[0] == ["Points to label", "40 (of 50 requested)"]
    assert _two_map_plan_card(base)[0] == ["Squares to label", "40 (of 50 requested)"]
    one = _plan_card({**base, "design": "random", "sampling": grid}, None)
    assert one[0] == ["Points to label", "40 (of 50 requested)"]
    assert _plan_card({**base, "design": "random"}, None)[0] == [
        "Squares to label",
        "40 (of 50 requested)",
    ]
    assert "Windows to label" not in json.dumps([two, one])


@pytest.mark.asyncio
async def test_an_estimate_with_a_window_not_judged_cards_the_count() -> None:
    scores, truth = _map(400)
    plan = await _ok(PLAN, {"scores": scores, "budget": 60, "design": "random"})
    indices = _indices(plan)
    sheet = Path(plan["labels_csv_path"])
    with sheet.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    for row in rows:
        i = int(row["window_index"])
        row["wrong"] = "?" if i == indices[0] else str(truth[i])
    with sheet.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    out = await _ok(
        GRADE, {"design_path": plan["design_path"], "labels_path": str(sheet)}
    )
    assert out["n_unjudged"] == 1
    valid = out["valid_interval"]
    rate = (
        rules.percent(valid["estimate"])
        if valid.get("estimate") is not None
        else _span(*valid["estimate_range"])
    )
    assert _row(out, "Error rate")[1] == rate
    assert (
        _row(out, "95% interval valid at every sample size")[1] in out["must_state"][0]
    )
    assert _row(out, "Labels")[1] == (
        f"60 of {out['n_population']} windows, 1 not judged (?)"
    )
    assert _row(out, "Design")[1] == "random"
    # A random design's own interval covers its level: 95%, not nominal.
    design_row = _row(out, "Interval this design earns")
    assert design_row[2] == f"95%; {out['method']}"


@pytest.mark.asyncio
@pytest.mark.parametrize("design", ["confidence", "proportional"])
async def test_a_nominal_designs_interval_with_a_window_not_judged_covers_95(
    design: str,
) -> None:
    """With a window not judged (?) the package gives the confidence and
    proportional designs' interval as the union bound over the strata: 95%,
    no longer nominal."""
    scores, truth = _map(600)
    plan = await _ok(PLAN, {"scores": scores, "budget": 90, "design": design})
    first = _indices(plan)[0]
    sheet = Path(plan["labels_csv_path"])
    with sheet.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    for row in rows:
        i = int(row["window_index"])
        row["wrong"] = "?" if i == first else str(truth[i])
    with sheet.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    out = await _ok(
        GRADE, {"design_path": plan["design_path"], "labels_path": str(sheet)}
    )
    assert out["n_unjudged"] == 1
    design_row = _row(out, "Interval this design earns")
    assert design_row[2] == f"95%; {out['method']}"
    assert "nominal" not in design_row[2]


@pytest.mark.asyncio
async def test_windows_labelled_without_a_design_card_design_none() -> None:
    scores, truth = _map()
    drawn = random.Random(3).sample(range(len(scores)), 100)
    out = await _ok(
        GRADE,
        {"window_indices": drawn, "wrong": [truth[i] for i in drawn], "scores": scores},
    )
    assert _names(out) == ["Error rate", "95% interval", "Labels", "Design"]
    whole = _fact(out, "whole_map_estimate")
    assert _row(out, "Error rate")[1] == rules.percent(out["estimate"])
    assert _row(out, "Error rate")[1] in whole
    interval = _row(out, "95% interval")
    assert interval[1:] == [_span(out["low"], out["high"]), out["method"]]
    assert interval[1] in whole
    assert _row(out, "Labels")[1] == "100 of 1200 windows"
    assert _row(out, "Design")[1] == NO_DESIGN


async def _random_zone_inputs() -> tuple[str, list[int]]:
    scores, truth = _map(2000, seed=11)
    plan = await _ok(PLAN, {"scores": scores, "budget": 300, "design": "random"})
    return plan["design_path"], [truth[i] for i in _indices(plan)]


@pytest.mark.asyncio
async def test_a_zone_cards_the_zone_and_its_test_but_no_whole_map_rate() -> None:
    design_path, wrong = await _random_zone_inputs()
    args = {"design_path": design_path, "wrong": wrong}
    zone = await _ok(CERTIFY, {**args, "alpha": 0.1, "rule": "bonferroni"})
    assert zone["certified"] is True
    assert _names(zone) == ["Certified zone", "Test", "Labels", "Design"]
    certified = _row(zone, "Certified zone")[1]
    assert certified == (
        f"the {zone['n_zone']} most confident windows (coverage {zone['coverage']:g})"
    )
    assert any(certified in sentence for sentence in zone["must_state"])
    assert _row(zone, "Test")[1] == "alpha=0.1, delta=0.1, rule 'bonferroni'"
    assert "alpha=0.1, delta=0.1" in zone["verdict"]
    assert "rule 'bonferroni'" in zone["verdict"]
    assert _row(zone, "Labels")[1] == f"300 of {zone['n_population']} windows"
    assert _row(zone, "Design")[1] == "random"

    none = await _ok(CERTIFY, {**args, "alpha": 0.05})
    assert none["certified"] is False
    assert _row(none, "Certified zone")[1] == "none"
    assert none["verdict"].startswith("No zone is certified")
    assert _row(none, "Test")[1] == "alpha=0.05, delta=0.1, rule 'prefix'"
    # The whole-map rate is left off: beside a zone it would pass for its rate.
    for result in (zone, none):
        assert not any("rate" in name.lower() for name in _names(result))


@pytest.mark.asyncio
async def test_a_zone_refused_on_a_stratified_design_has_no_card() -> None:
    scores, _truth = _map(2000, seed=11)
    plan = await _ok(PLAN, {"scores": scores, "budget": 300})
    refused = await _ok(
        CERTIFY, {"design_path": plan["design_path"], "wrong": [0] * 300, "alpha": 0.1}
    )
    assert refused["certified"] is False and refused["reason"]
    assert rules.CARD_KEY not in refused


# --------------------------------------------------------------------------- two maps


def _scores_file(path: Path, classes: list[int], grid: list[int]) -> str:
    rows = [[1.0 if k == c else 0.0 for k in range(3)] for c in classes]
    path.write_text(json.dumps({"grid": grid, "scores": rows}))
    return str(path)


@pytest.mark.asyncio
@pytest.mark.parametrize("budget", [4, 20, 200])
async def test_a_which_map_plan_and_grade_card_the_verdict_as_stated(
    tmp_path: Path, budget: int
) -> None:
    """The maps differ on 38 of 64 windows: four labels are graded by the exact
    sign test, twenty by the interval, and a budget above the differing
    windows plans each of them and says so."""
    a = [i % 3 for i in range(64)]
    b = [x if i % 5 in (0, 1) else (x + 1) % 3 for i, x in enumerate(a)]
    files = {
        "scores_path": _scores_file(tmp_path / "a.json", a, [8, 8]),
        "scores_path_b": _scores_file(tmp_path / "b.json", b, [8, 8]),
        "budget": budget,
        "seed": 3,
        "design": "disagreement",
    }
    plan = await _ok(PLAN, files)
    assert plan["n_disagree"] == 38
    assert plan["budget"] == min(budget, 38)
    assert _names(plan) == [
        "Squares to label",
        "Design",
        "Where the maps differ",
        "Seed",
    ]
    planned = plan["budget"]
    want = f"{planned}" + (f" (of {budget} requested)" if budget != planned else "")
    assert _row(plan, "Squares to label")[1] == want
    assert _row(plan, "Design")[1] == "disagreement"
    differ = _row(plan, "Where the maps differ")[1]
    assert (
        differ == f"{plan['n_disagree']} of the {plan['n_population']} windows compared"
    )
    assert differ in plan["design_note"]
    assert _row(plan, "Seed")[1] == "3"

    idx = _indices(plan)
    graded = await _ok(
        GRADE,
        {"design_path": plan["design_path"], "reference": [str(a[i]) for i in idx]},
    )
    assert _names(graded) == [
        "More accurate",
        "Decided by",
        "Difference in accuracy, a minus b",
        "Labels",
        "Design",
    ]
    stated = graded["must_state"][0]
    verdict, n = graded.get("verdict"), graded["n_labelled"]
    named = (
        f"Map {verdict}" if verdict in ("a", "b") else "Neither map, by these labels"
    )
    assert _row(graded, "More accurate")[1] == named
    decided = _row(graded, "Decided by")[1]
    assert graded["verdict_rule"] == ("sign" if budget == 4 else "interval")
    if graded["verdict_rule"] == "sign":
        p = float(graded["sign_test"]["p_value"])
        assert decided == (
            f"exact sign test on {n} labels, p = {p:.3g}"
            if verdict in ("a", "b")
            else "exact sign test"
        )
    else:
        assert decided == "the 95% interval"
    assert decided in stated
    diff = _row(graded, "Difference in accuracy, a minus b")
    lo, hi = (100.0 * float(graded["difference"][k]) for k in ("low", "high"))
    assert diff[1] == f"{lo:+.1f} to {hi:+.1f} points" and diff[1] in stated
    assert diff[2] == (
        f"95% interval over the {graded['n_population']} windows compared"
    )
    assert _row(graded, "Labels")[1] == (
        f"{n} of the {graded['n_disagree']} windows where the maps differ"
    )
    assert _row(graded, "Design")[1] == "disagreement"


# --------------------------------------------------------------------------- review sets


@pytest.mark.asyncio
async def test_a_review_set_cards_its_count_signal_and_list_file() -> None:
    scores, _truth = _map(200)
    out = await _ok(
        "olmoearth_review_set", {"scores": scores, "budget": 0.5, "max_listed": 5}
    )
    # r4h V7: the card's first row names squares, never "windows".
    assert _names(out) == ["Squares to look at first", "Ranked by", "Full list"]
    assert _row(out, "Squares to look at first")[1] == (
        f"{out['n_review']:,} of {out['n_windows']:,} "
        f"(budget {out['realised_budget']:g})"
    )
    assert _row(out, "Ranked by")[1] == out["signal"]
    assert _row(out, "Full list")[1] == os.path.basename(out["review_list_path"])
    # Every window listed: no list file, no row for it.
    small = await _ok("olmoearth_review_set", {"scores": scores[:16], "budget": 0.25})
    assert _names(small) == ["Squares to look at first", "Ranked by"]


@pytest.mark.asyncio
async def test_a_boundary_first_review_set_is_not_carded_as_ranked_by_margin() -> None:
    """Boundary windows come first, by margin within each block: the card names
    that order as the tool's schema does, with the margin as its detail."""
    scores, _truth = _map(64)
    out = await _ok(
        "olmoearth_review_set",
        {"scores": scores, "budget": 0.25, "grid": [8, 8], "order": "boundary_first"},
    )
    assert out["order"] == "boundary_first"
    assert _row(out, "Ranked by") == [
        "Ranked by",
        BOUNDARY_FIRST_RANKING,
        out["signal"],
    ]
    assert "class boundary first" in BOUNDARY_FIRST_RANKING


_KB_BBOX = [0.0, 0.0, 6.0, 6.0]


def _kb_value(lon: float, lat: float) -> float:
    """A deterministic [0, 1] score at a point, no-data (-1) on about a third."""
    frac = math.sin(lon * 12.9898 + lat * 78.233) * 43758.5453
    frac -= math.floor(frac)
    return -1.0 if frac < 0.33 else round(frac, 4)


def _mock_kb(httpx_mock: HTTPXMock) -> None:
    """Serve result 'kb', a [0, 1] score band, and its pixel values."""
    box = [[[0, 0], [6, 0], [6, 6], [0, 6], [0, 0]]]
    field = {"property_name": "sample_karst_score", "min_value": 0.0, "max_value": 1.0}
    httpx_mock.add_response(
        url=f"{BASE}/prediction-results/kb",
        json={
            "records": [
                {
                    "id": "kb",
                    "property_names": ["sample_karst_score"],
                    "result_metadata": {
                        "geometry": {"type": "Polygon", "coordinates": box},
                        "regression_fields": [field],
                    },
                }
            ]
        },
        is_reusable=True,
    )

    def pixel(request: httpx.Request) -> httpx.Response:
        q = parse_qs(urlparse(str(request.url)).query)
        band = {
            "property_name": "sample_karst_score",
            "raw_value": _kb_value(float(q["lon"][0]), float(q["lat"][0])),
            "classification": None,
            "regression": {"min_value": 0.0, "max_value": 1.0},
        }
        return httpx.Response(200, json={"records": [{"bands": [band]}]})

    httpx_mock.add_callback(
        pixel, url=re.compile(r".*/pixel-value\?.*"), is_reusable=True
    )


@pytest.mark.asyncio
async def test_a_studio_results_plan_and_review_set_card_the_sampled_grid(
    httpx_mock: HTTPXMock,
) -> None:
    _mock_kb(httpx_mock)
    async with StudioClient(StudioConfig(api_key="k", base_url=BASE)) as studio:
        ctx = ToolContext(studio=studio, state=ThreadState())
        plan = await _ok(
            PLAN, {"result_id": "kb", "grid": 6, "budget": 5, "design": "random"}, ctx
        )
        ranked = await _ok(
            "olmoearth_review_set_from_result",
            {"result_id": "kb", "grid": 6, "save_scores": False},
            ctx,
        )
    sampling = plan["sampling"]
    # A Studio result's plan labels its sampled grid's points (r4h V7).
    assert _names(plan) == [
        "Points to label",
        "Design",
        "Population",
        "Sampled grid",
        "Seed",
    ]
    planned, asked = plan["budget"], plan["budget_requested"]
    want = f"{planned}" + (f" (of {asked} requested)" if asked != planned else "")
    assert _row(plan, "Points to label")[1] == want
    assert _row(plan, "Sampled grid")[1] == f"6x6, {sampling['n_valid']} valid points"
    assert _row(plan, "Design") == ["Design", "random"]

    budgets = ranked["budgets"]
    assert len(budgets) == 3
    # The reader test's B/R3 and B/R4: the share checked, the first k windows,
    # "how decided" for the margin, and "(k of N; outlined in the picture)" on
    # the largest budget's row only, the one the figure outlines.
    shares = [
        f"If you check {100 * entry['budget']:g}% of the windows" for entry in budgets
    ]
    assert shares == [
        "If you check 1% of the windows",
        "If you check 5% of the windows",
        "If you check 10% of the windows",
    ]
    assert _names(ranked) == [*shares, "Sampled grid", "Property", "Ranked by"]
    n_valid = ranked["sampling"]["n_valid"]
    for at, (share, entry) in enumerate(zip(shares, budgets, strict=True)):
        k = entry["n_review"]
        outlined = (
            f" ({k:,} of {n_valid:,}; outlined in the picture)" if at == 2 else ""
        )
        plural = "" if k == 1 else "s"
        assert _row(ranked, share)[1] == (
            f"the first {k:,} window{plural}{outlined}; how decided: up to "
            f"{entry['margin_cut']:g}"
        )
    assert "margin up to" not in json.dumps(ranked[rules.CARD_KEY])
    grid = ranked["sampling"]
    assert _row(ranked, "Sampled grid")[1] == f"6x6, {grid['n_valid']} valid points"
    assert _row(ranked, "Property")[1] == "sample_karst_score"
    assert _row(ranked, "Ranked by") == [
        "Ranked by",
        "margin, lowest first",
        ranked["signal"],
    ]
    assert len(ranked[rules.CARD_KEY]) <= rules.CARD_MAX_ROWS
