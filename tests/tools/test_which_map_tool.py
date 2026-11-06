# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Which of two maps is more accurate: the two-map design and its grade.

``olmoearth_plan_label_sample`` with ``design='disagreement'`` reads each
window's class in two Studio results of one area (through
``olmoearth_compare_results``' own pair sampler) or in two scores files of one
window grid, refuses maps that ``oe_inferencex.compare.dates_reading`` does
not read as one time, and draws the windows to label where the maps differ
(``estimate.sample_disagreement``); ``olmoearth_estimate_map_error`` grades
the reviewer's classes with ``estimate.compare_from_disagreement``. These run
the real package (``uv sync --all-extras``) and hold every number to it: the
agent computes no verdict and no interval.
"""

from __future__ import annotations

import base64
import csv
import hashlib
import json
import struct
import xml.etree.ElementTree as ET
import zlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from olmoearth_agent.analysis.raster_compare import grid_windows
from olmoearth_agent.harness.checks import (
    RunEvidence,
    ToolRecord,
    check_forbidden_claims,
    check_must_state,
    run_checks,
)
from olmoearth_agent.harness.state import ThreadState
from olmoearth_agent.llm.types import ToolCall
from olmoearth_agent.reporting import figures as fg
from olmoearth_agent.tools import statistical_rules as rules
from olmoearth_agent.tools.estimation import (
    UNSTATED_DATES_MUST_STATE,
    _jsonable,
    build_estimation_tools,
)
from olmoearth_agent.tools.registry import ToolContext, ToolRegistry
from olmoearth_agent.tools.sampling import prediction_period

estimate = pytest.importorskip("oe_inferencex.estimate")
decide = pytest.importorskip("oe_inferencex.decide")

PLAN = "olmoearth_plan_label_sample"
GRADE = "olmoearth_estimate_map_error"
_BANNED_KEYS = {"lon", "lat", "bbox", "coordinates", "geometry", "centres_lon_lat"}

#: The two results' shared box and grid (6 x 6 = 36 windows).
_BOX = {"type": "Polygon", "coordinates": [[[0, 0], [4, 0], [4, 4], [0, 4], [0, 0]]]}
GRID = 6
_CODES = {"water": 0, "forest": 1, "crop": 2}
_LANDCOVER = {
    "classification_fields": [
        {
            "property_name": "landcover",
            "allowed_values": [{"value": v, "label": k} for k, v in _CODES.items()],
        }
    ]
}
JUNE_2024 = ("2024-06-01T00:00:00Z", "2024-06-30T00:00:00Z")


def _truth(row: int, col: int) -> str:
    """What is on the ground: water west of column 3, forest east of it."""
    return "water" if col < 3 else "forest"


def _map_a(row: int, col: int) -> str | None:
    """Map a: right but for 5 windows called crop (row + col a multiple of 7)."""
    return "crop" if (row + col) % 7 == 0 else _truth(row, col)


def _map_b(row: int, col: int) -> str | None:
    """Map b: columns 2 and 3 swapped, and no-data at (0, 5)."""
    if (row, col) == (0, 5):
        return None
    if col in (2, 3):
        return "forest" if col == 2 else "water"
    return _truth(row, col)


#: Where map a and map b differ (15 of the 35 windows both hold a class): map a
#: is right on 10 of them, map b on 3, neither on 2.
N_COMPARED, N_DIFFER = 35, 15


def _score_meta(lo: float, hi: float) -> dict[str, Any]:
    """A regression band of ``[lo, hi]``, as a result declares it."""
    field = {"property_name": "landcover", "min_value": lo, "max_value": hi}
    return {"regression_fields": [field]}


def _score(labels: Callable[[int, int], str | None]) -> Callable[[int, int], Any]:
    """A map as a [0, 1] score of water: 0.9 where it says water, else 0.1."""

    def value(row: int, col: int) -> float | None:
        label = labels(row, col)
        return None if label is None else (0.9 if label == "water" else 0.1)

    return value


class _Studio:
    """Two results over one box, as the Studio client returns them: class
    bands, or score bands when a map gives numbers."""

    def __init__(
        self,
        maps: dict[str, Callable[[int, int], Any]],
        periods: dict[str, tuple[str, str]],
        meta: dict[str, Any] | None = None,
    ) -> None:
        self.maps = maps
        self.periods = periods
        self.meta = meta or _LANDCOVER
        self.cells = grid_windows([0, 0, 4, 4], GRID)
        self.pixel_calls = 0

    async def get_prediction_result(self, result_id: str) -> dict[str, Any]:
        return {
            "id": result_id,
            "prediction_id": f"p-{result_id}",
            "property_names": ["landcover"],
            "result_metadata": {"geometry": _BOX, **self.meta},
        }

    async def get_prediction(self, prediction_id: str) -> dict[str, Any]:
        start, end = self.periods[prediction_id.removeprefix("p-")]
        return {
            "id": prediction_id,
            "model_id": f"m-{prediction_id}",
            "start_time": start,
            "end_time": end,
        }

    async def get_model(self, model_id: str) -> dict[str, Any]:
        return {"id": model_id, "name": f"model {model_id}", "wizard_answers": {}}

    async def pixel_value(self, result_id: str, lon: float, lat: float) -> Any:
        self.pixel_calls += 1
        row, col, _lo, _la = min(
            self.cells, key=lambda c: (c[2] - lon) ** 2 + (c[3] - lat) ** 2
        )
        value = self.maps[result_id](row, col)
        band: dict[str, Any] = {"property_name": "landcover"}
        if value is None:
            band.update(raw_value=None, classification=None)
        elif isinstance(value, str):
            band.update(
                raw_value=_CODES[value],
                classification={"label": value, "color": "#000000"},
            )
        else:
            regression = self.meta["regression_fields"][0]
            band.update(raw_value=value, classification=None, regression=regression)
        return {"bands": [band]}


def _studio(
    b: Callable[[int, int], str | None] = _map_b,
    period_b: tuple[str, str] = JUNE_2024,
) -> _Studio:
    return _Studio({"ra": _map_a, "rb": b}, {"ra": JUNE_2024, "rb": period_b})


@pytest.fixture(autouse=True)
def _scores_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("OLMOEARTH_SCORES_ROOT", str(tmp_path))
    monkeypatch.setenv("OLMOEARTH_OUTPUT_ROOT", str(tmp_path))
    return tmp_path


async def _call(name: str, args: dict[str, Any], studio: Any = None) -> Any:
    registry = ToolRegistry()
    registry.register_all(build_estimation_tools())
    ctx = ToolContext(studio=studio, state=ThreadState())
    return await registry.dispatch(ToolCall(id="1", name=name, arguments=args), ctx)


def _plan_args(budget: int, seed: int = 0, **extra: Any) -> dict[str, Any]:
    return {
        "result_id": "ra",
        "result_id_b": "rb",
        "budget": budget,
        "seed": seed,
        "grid": GRID,
        "design": "disagreement",
        **extra,
    }


async def _plan(budget: int, seed: int = 0) -> dict[str, Any]:
    out = await _call(PLAN, _plan_args(budget, seed), _studio())
    assert out["ok"] is True, out
    return out["result"]


def _design(plan: dict[str, Any]) -> dict[str, Any]:
    return json.loads(Path(plan["design_path"]).read_text())


def _keys(obj: Any) -> set[str]:
    if isinstance(obj, dict):
        return set(obj).union(*(_keys(v) for v in obj.values()))
    if isinstance(obj, list):
        return set().union(*(_keys(v) for v in obj))
    return set()


def _window(index: int) -> tuple[int, int]:
    return divmod(index, GRID)


def _codes(labels: list[str | None]) -> tuple[list[int], dict[str, int]]:
    """Classes as codes, -1 for none (the draw reads only where codes differ)."""
    table = {k: i for i, k in enumerate(sorted({v for v in labels if v}))}
    return [table[v] if v else -1 for v in labels], table


def _record(name: str, args: dict[str, Any], result: dict[str, Any]) -> ToolRecord:
    return ToolRecord(name, args, {"ok": True, "result": result})


# --------------------------------------------------------------------------- the rules


@pytest.mark.parametrize(
    ("verdict", "rule", "start"),
    [
        ("a", "sign", "Map a is more accurate: exact sign test on 120 labels, p ="),
        ("b", "interval", "Map b is more accurate: the 95% interval on a minus b,"),
        (None, "sign", "These 120 labels name neither map more accurate (exact"),
        (None, "interval", "These 120 labels name neither map more accurate: the"),
    ],
)
def test_the_required_statement_names_the_test_and_the_labels(
    verdict: str | None, rule: str, start: str
) -> None:
    """At most 25 words, as the contract's must_state is, in all four forms."""
    result = {
        "verdict": verdict,
        "verdict_rule": rule,
        "n_labelled": 120,
        "conf": 0.95,
        "difference": {"low": -0.2341, "high": 0.1236},
        "sign_test": {"p_value": 0.000123},
    }
    sentence = rules.which_map_must_state(result)
    assert sentence.startswith(start)
    assert len(sentence.split()) <= 25
    assert "120 labels" in sentence and "-23.4 to +12.4 points" in sentence
    assert ("sign test" if rule == "sign" else "interval") in sentence


def test_a_predictions_period_is_its_start_and_end() -> None:
    """The dates a Studio map describes, as dates_reading reads a period."""
    june = {"start_time": "2024-06-01T00:00:00Z", "end_time": "2024-06-30T00:00:00Z"}
    assert prediction_period(june) == "2024-06-01T00:00:00Z/2024-06-30T00:00:00Z"
    assert prediction_period({"start_time": "2024-06-01"}) == "2024-06-01"
    same = {"start_time": "2024-06-01", "end_time": "2024-06-01"}
    assert prediction_period(same) == "2024-06-01"
    assert prediction_period(None) is None and prediction_period({}) is None


# --------------------------------------------------------------------------- the plan


@pytest.mark.asyncio
async def test_the_plan_refuses_two_maps_of_different_times() -> None:
    """Across dates a window where the maps differ can be a change on the
    ground, so labels there cannot say which map is more accurate; the dates
    are read from the predictions before any pixel is sampled."""
    for period_b, status in (
        (("2023-06-01T00:00:00Z", "2023-06-30T00:00:00Z"), "different_time"),
        (("2024-06-15T00:00:00Z", "2024-07-15T00:00:00Z"), "overlapping_time"),
    ):
        studio = _studio(period_b=period_b)
        out = await _call(PLAN, _plan_args(10), studio)
        assert out["ok"] is False, out
        assert "the two maps are not of one time" in out["error"]
        assert f"(dates_reading: {status})" in out["error"]
        assert "cannot say which map is more accurate" in out["error"]
        assert studio.pixel_calls == 0
        ids = {c["id"] for c in out["forbidden_claims"]}
        assert {
            rules.ONE_REFERENCE_SETTLES_TWO_DATES,
            rules.ANOTHER_DATE_SETTLES_IT,
            rules.WINNER_WITHOUT_LABELS,
            # agent trial 12: a disagreement design offered across dates
            rules.TWO_MAP_DESIGN_ACROSS_DATES,
        } <= ids
        (why,) = [
            c["why"]
            for c in out["forbidden_claims"]
            if c["id"] == rules.TWO_MAP_DESIGN_ACROSS_DATES
        ]
        assert "no labels on the differing windows say which map" in why
        assert out["must_state"] == [rules.DATED_MAPS_MUST_STATE]


@pytest.mark.asyncio
async def test_the_plan_refuses_two_maps_that_never_differ() -> None:
    """The package's own refusal: neither is more accurate, nothing to label."""
    out = await _call(PLAN, _plan_args(10), _studio(b=_map_a))
    assert out["ok"] is False
    assert "give the same class in every window both predict" in out["error"]
    assert "(36 windows compared)" in out["error"]


@pytest.mark.asyncio
async def test_the_drawn_windows_are_sample_disagreements_for_the_seed() -> None:
    """Each window's class in both maps, read by the pair sampler, then the
    package's draw: the same windows in the same order for the same seed."""
    labels_a = [_map_a(r, c) for r in range(GRID) for c in range(GRID)]
    labels_b = [_map_b(r, c) for r in range(GRID) for c in range(GRID)]
    valid = [x is not None and y is not None for x, y in zip(labels_a, labels_b)]
    ca, _table = _codes(labels_a + labels_b)
    class_a, class_b = ca[: GRID * GRID], ca[GRID * GRID :]
    for seed in (0, 7):
        studio = _studio()
        out = await _call(PLAN, _plan_args(6, seed), studio)
        assert out["ok"] is True, out
        plan = out["result"]
        direct = estimate.sample_disagreement(
            class_a, class_b, 6, valid=valid, seed=seed
        )
        drawn = [int(i) for i in direct["indices"].tolist()]
        assert _design(plan)["sample"]["indices"] == drawn
        assert [w["window_index"] for w in plan["windows"]] == drawn
        assert plan["n_population"] == direct["n_population"] == N_COMPARED
        assert plan["n_disagree"] == direct["n_disagree"] == N_DIFFER
        assert plan["budget"] == 6 and plan["seed"] == seed
        # The pair sampler: one pixel-value call per window and result.
        assert studio.pixel_calls == 2 * GRID * GRID
        for w in plan["windows"]:
            row, col = _window(w["window_index"])
            assert (w["row"], w["col"]) == (row, col)
            assert (w["class_a"], w["class_b"]) == (_map_a(row, col), _map_b(row, col))
    # A seed draws its own windows.
    first = (await _plan(6, 0))["windows"]
    other = (await _plan(6, 7))["windows"]
    assert [w["window_index"] for w in first] != [w["window_index"] for w in other]


@pytest.mark.asyncio
async def test_the_plan_gives_the_packages_budget_advice_and_no_coordinates() -> None:
    plan_pkg = pytest.importorskip("oe_inferencex.plan")
    plan = await _plan(20)
    # A budget above the differing windows labels every one of them.
    assert plan["budget"] == N_DIFFER and plan["budget_requested"] == 20
    advice = _jsonable(plan_pkg.plan_which_map(N_COMPARED, N_DIFFER, 0.05))
    assert plan["budget_advice"] == advice
    fact = next(f for f in plan["facts"] if f["id"] == "which_map_budget")
    assert fact["labels"] == advice["labels"] and fact["difference"] == 0.05
    unused = next(f for f in plan["facts"] if f["id"] == "unused_labels")
    assert (unused["n"], unused["requested"], unused["planned"]) == (5, 20, 15)
    assert plan["dates"]["status"] == "same_time"
    assert plan["map_a"]["result_id"] == "ra" and plan["map_b"]["result_id"] == "rb"
    assert plan["sampling"]["grid"] == "6x6"
    assert plan["sampling"]["grid_capped"] is False
    assert plan["must_state"] == [
        rules.NOT_EITHER_ACCURACY,
        f"The comparison covers the {N_COMPARED} sampled grid points where both "
        "Studio results hold a class (one pixel each), not every pixel.",
        "The plan holds 15 windows, not the 20 requested.",
    ]
    ids = [c["id"] for c in plan["forbidden_claims"]]
    # Its grading reads the whole sheet, as every design file's does.
    assert ids == [
        rules.WINNER_WITHOUT_LABELS,
        rules.ERROR_RATE_WITHOUT_LABELS,
        rules.SUBSET_LABELLING_SUFFICIENT,
    ]
    subset = plan["forbidden_claims"][2]["why"]
    assert subset.startswith("write a class, or ? where it cannot be judged")
    winner = plan["forbidden_claims"][0]["why"]
    assert winner.startswith("this plan holds no labels yet")
    # Rule 1: the result holds no location; the reviewer's sheet does.
    assert not _BANNED_KEYS & _keys(plan)
    with open(plan["labels_csv_path"], encoding="utf-8", newline="") as fh:
        header = next(csv.reader(fh))
    assert header == [
        "order",
        "window_index",
        "row",
        "col",
        "class_a",
        "class_b",
        "lon",
        "lat",
        "reference_class",
    ]
    # A grid above the pair's cap is held to it and says so.
    capped = await _call(PLAN, _plan_args(5, grid=30), _studio())
    assert capped["ok"] is True, capped
    sampling = capped["result"]["sampling"]
    assert sampling["grid"] == "12x12" and sampling["grid_capped"] is True
    assert sampling["grid_requested"] == 30


@pytest.mark.asyncio
async def test_two_score_bands_are_read_as_classes_at_their_decision() -> None:
    """A [0, 1] score is cut at 0.5 (class 1 above it), as the estimation tools
    decide it, or at the threshold given; a band of another range needs one,
    refused before any pixel is sampled."""
    maps = {"ra": _score(_map_a), "rb": _score(_map_b)}
    periods = {"ra": JUNE_2024, "rb": JUNE_2024}
    unit = _Studio(maps, periods, meta=_score_meta(0.0, 1.0))
    out = await _call(PLAN, _plan_args(6), unit)
    assert out["ok"] is True, out
    plan = out["result"]
    assert "1 above 0.5, 0 at or below it" in plan["assumption"]
    for w in plan["windows"]:
        row, col = _window(w["window_index"])
        water = (int(_map_a(row, col) == "water"), int(_map_b(row, col) == "water"))
        assert (w["class_a"], w["class_b"]) == water
    # At 0.95 neither map has a class 1 window: they never differ.
    high = await _call(PLAN, _plan_args(6, threshold=0.95), unit)
    assert high["ok"] is False
    assert "give the same class in every window" in high["error"]
    wide = _Studio(maps, periods, meta=_score_meta(0.2, 1.2))
    undecided = await _call(PLAN, _plan_args(6), wide)
    assert undecided["ok"] is False and "pass threshold" in undecided["error"]
    assert "'landcover' (declared range [0.2, 1.2])" in undecided["error"]
    assert wide.pixel_calls == 0
    ids = {c["id"] for c in undecided["forbidden_claims"]}
    assert rules.ERROR_RATE_FOR_UNTHRESHOLDED_REGRESSION in ids


class _TwoProperties(_Studio):
    """Map b's result holds another property than map a's."""

    async def get_prediction_result(self, result_id: str) -> dict[str, Any]:
        record = await super().get_prediction_result(result_id)
        if result_id == "rb":
            record["property_names"] = ["burn_scar"]
        return record


@pytest.mark.asyncio
async def test_the_plan_refuses_two_properties_in_its_own_words() -> None:
    """The pair sampler refuses two properties before any pixel is sampled;
    its escape (allow_different_properties) is pair mode's, not the plan's."""
    studio = _TwoProperties(
        {"ra": _map_a, "rb": _map_b}, {"ra": JUNE_2024, "rb": JUNE_2024}
    )
    out = await _call(PLAN, _plan_args(10), studio)
    assert out["ok"] is False, out
    assert "('landcover' and 'burn_scar')" in out["error"]
    assert "pass results of one property, or pin property_name" in out["error"]
    assert "allow_different_properties" not in out["error"]
    assert studio.pixel_calls == 0
    assert rules.WINNER_WITHOUT_LABELS in {c["id"] for c in out["forbidden_claims"]}


@pytest.mark.asyncio
async def test_the_plan_and_its_grade_say_only_what_the_checks_allow() -> None:
    """An answer made of the tools' own sentences passes every answer check."""
    plan_args = _plan_args(20)
    plan = (await _call(PLAN, plan_args, _studio()))["result"]
    design = _design(plan)
    refs = [_truth(*_window(i)) for i in design["sample"]["indices"]]
    grade_args = {"design_path": plan["design_path"], "reference": refs}
    graded = (await _call(GRADE, grade_args))["result"]
    for name, args, result in ((PLAN, plan_args, plan), (GRADE, grade_args, graded)):
        said = [f["sentence"] for f in result["facts"]] + list(result["must_state"])
        said += list(result["next_steps"])
        said += [s for s in (result.get("design_note"),) if s]
        said += [result["dates"]["reading"]] if "dates" in result else []
        run = RunEvidence(
            tools=[_record(name, args, result)],
            user_messages=["Which of the two maps is more accurate?"],
        )
        found = run_checks("\n\n".join(said), run)
        assert found == {}, json.dumps(found, indent=1)
    # The plan holds no labels: a map named by the package's own lower-case
    # letter is a winner without them.
    plan_run = RunEvidence(tools=[_record(PLAN, plan_args, plan)])
    for claim in ("Map a is more accurate.", "map b is the better map."):
        violations = check_forbidden_claims(claim, plan_run)
        assert [v["text"] for v in violations] == [claim]
        assert violations[0]["detail"].startswith(rules.WINNER_WITHOUT_LABELS)


# --------------------------------------------------------------------------- the grade


@pytest.mark.asyncio
async def test_the_grade_is_the_packages_with_one_unjudged_window() -> None:
    """Every number is the package's on the design's sample and the classes
    the plan recorded; a ? goes in as unjudged."""
    plan = await _plan(20)  # every differing window: a census of them
    design = _design(plan)
    idx = design["sample"]["indices"]
    assert [design["classes"][c] for c in design["class_a"]] == [
        _map_a(*_window(i)) for i in idx
    ]
    assert [design["classes"][c] for c in design["class_b"]] == [
        _map_b(*_window(i)) for i in idx
    ]
    refs = [_truth(*_window(i)) for i in idx]
    refs[0] = "?"
    out = await _call(GRADE, {"design_path": plan["design_path"], "reference": refs})
    assert out["ok"] is True, out
    graded = out["result"]
    code_of = {v: k for k, v in enumerate(design["classes"])}
    expected = _jsonable(
        estimate.compare_from_disagreement(
            design["sample"],
            design["class_a"],
            design["class_b"],
            [-1 if r == "?" else code_of[r] for r in refs],
            unjudged=[r == "?" for r in refs],
        )
    )
    assert {k: graded[k] for k in expected} == expected
    assert graded["n_unjudged"] == 1 and graded["n_labelled"] == N_DIFFER
    assert graded["n_a_right"] + graded["n_b_right"] + graded["n_neither"] == 14
    # A census of the differing windows: the interval decides, and names map a
    # (10 or 9 right in a, 3 or 2 in b, 1 not judged).
    assert graded["verdict_rule"] == "interval" and graded["verdict"] == "a"
    assert graded["which_map"] == "a"
    assert graded["more_accurate_map"]["result_id"] == "ra"
    assert graded["note"] == estimate.WHICH_MAP_NOTE
    assert graded["verdict_note"] == estimate.SIGN_TEST_NOTE
    fact = next(f for f in graded["facts"] if f["id"] == "which_map")
    assert fact["more_accurate"] == "a" and fact["n_labelled"] == N_DIFFER
    assert (fact["low"], fact["high"]) == (
        expected["difference"]["low"],
        expected["difference"]["high"],
    )
    # The same labels through the sheet the plan wrote.
    sheet = Path(plan["labels_csv_path"])
    with open(sheet, encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    for row, ref in zip(rows, refs):
        row["reference_class"] = ref
    with open(sheet, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    by_sheet = await _call(
        GRADE, {"design_path": plan["design_path"], "labels_path": str(sheet)}
    )
    assert {k: by_sheet["result"][k] for k in expected} == expected


@pytest.mark.asyncio
async def test_the_answer_states_the_test_and_the_number_of_labels() -> None:
    """The rule for an answer naming a more accurate map: the test or interval
    it rests on and the number of labels, required and checked."""
    for budget, rule in ((3, "sign"), (20, "interval")):
        plan = await _plan(budget)
        idx = _design(plan)["sample"]["indices"]
        args = {
            "design_path": plan["design_path"],
            "reference": [_truth(*_window(i)) for i in idx],
        }
        graded = (await _call(GRADE, args))["result"]
        n = graded["n_labelled"]
        assert graded["verdict_rule"] == rule
        # The package's own sentence for a sign-test verdict, none for the interval.
        assert graded.get("sign_test_statement") == decide.sign_test_head(graded)
        assert (rule == "sign") == ("sign_test_statement" in graded)
        stated = graded["must_state"][0]
        assert stated == rules.which_map_must_state(graded)
        assert f"{n} labels" in stated
        assert ("exact sign test" if rule == "sign" else "interval") in stated
        assert graded["must_state"][1] == rules.NOT_EITHER_ACCURACY
        ids = [c["id"] for c in graded["forbidden_claims"]]
        assert ids == [rules.MORE_ACCURATE_WITHOUT_ITS_TEST]
        run = RunEvidence(tools=[_record(GRADE, args, graded)])
        # Named without its test and labels, or the other map: flagged.
        for claim in ("Map a is more accurate.", "Map b is more accurate."):
            assert [v["text"] for v in check_forbidden_claims(claim, run)] == [claim]
        assert check_forbidden_claims(stated, run) == []
        # The required statement is appended when the answer leaves it out.
        lacking = check_must_state(f"{GRADE} says map a is more accurate.", run)
        assert stated in [v["text"] for v in lacking]
        if rule == "sign":
            # Three labels are too few for the exact sign test to name a map.
            assert graded["which_map"] == "undetermined" and graded["verdict"] is None
            # More labels read after these are a second test, in the package's words.
            assert decide.comparison_second_look() in graded["next_steps"]
        else:
            assert graded["which_map"] == "a" and graded["verdict"] == "a"


@pytest.mark.asyncio
async def test_the_grade_says_it_grades_these_maps_not_their_models(
    tmp_path: Path,
) -> None:
    """A map is one training run and one prediction (Corley et al. 2026, R4):
    a grade of two dated scores files states it grades these two maps on these
    windows, and an answer that keeps its key terms passes the must_state
    check. What the figures cover comes first in the contract's three, so a
    Studio grade keeps its grid points instead."""
    studio = await _plan(20)
    idx = _design(studio)["sample"]["indices"]
    studio_args = {
        "design_path": studio["design_path"],
        "reference": [_truth(*_window(i)) for i in idx],
    }
    out = await _call(GRADE, studio_args)
    assert rules.MAPS_NOT_MODELS not in out["result"]["must_state"]
    assert out["result"]["must_state"][2].startswith("The comparison covers the ")
    a = [0, 0, 1, 1, 2, 2, 0, 1, 2, 0, 1, 2, 0, 0, 1, 1]
    b = [0, 1, 1, 2, 2, 0, 0, 1, 1, 0, 1, 0, 0, 2, 1, 1]
    files = {
        "scores_path": _scores_file(tmp_path / "a.json", a, [4, 4]),
        "scores_path_b": _scores_file(tmp_path / "b.json", b, [4, 4]),
        "budget": 4,
        "seed": 3,
        "design": "disagreement",
        "date_a": "2024-06-01",
        "date_b": "2024-06-01",
    }
    planned = await _call(PLAN, files)
    assert planned["ok"] is True, planned
    plan = planned["result"]
    assert rules.MAPS_NOT_MODELS not in plan["must_state"]
    idx = _design(plan)["sample"]["indices"]
    args = {
        "design_path": plan["design_path"],
        "reference": [str(a[i]) for i in idx],
    }
    graded = (await _call(GRADE, args))["result"]
    assert graded["must_state"] == [
        rules.which_map_must_state(graded),
        rules.NOT_EITHER_ACCURACY,
        rules.MAPS_NOT_MODELS,
    ]
    run = RunEvidence(tools=[_record(GRADE, args, graded)])
    lacking = check_must_state(f"{GRADE} says map a is more accurate.", run)
    assert rules.MAPS_NOT_MODELS in [v["text"] for v in lacking]
    said = (
        f"{GRADE} grades these two maps on these windows only; it does not rank "
        "the models behind them on other areas, dates or training runs."
    )
    assert rules.MAPS_NOT_MODELS not in [v["text"] for v in check_must_state(said, run)]


@pytest.mark.asyncio
async def test_a_two_map_grade_takes_classes_not_wrong() -> None:
    plan = await _plan(4)
    path = plan["design_path"]
    wrong = await _call(GRADE, {"design_path": path, "wrong": [0, 1, 0, 1]})
    assert wrong["ok"] is False and "not from wrong" in wrong["error"]
    short = await _call(GRADE, {"design_path": path, "reference": ["water"] * 3})
    assert short["ok"] is False
    assert "3 references for the 4 windows of the design" in short["error"]
    # A class neither map writes is right in neither, and is listed.
    odd = await _call(
        GRADE, {"design_path": path, "reference": ["Water", "cloud", "?", "forest"]}
    )
    assert odd["ok"] is True, odd
    assert odd["result"]["references_outside_the_maps_classes"] == ["cloud"]
    fact = next(
        f
        for f in odd["result"]["facts"]
        if f["id"] == "references_outside_the_maps_classes"
    )
    assert (fact["n"], fact["n_judged"], fact["classes"]) == (1, 3, ["cloud"])
    assert "1 of 3 (cloud)" in fact["sentence"]
    # References that all lie outside the maps' classes (codes for names) are
    # not graded: every window would be right in neither map.
    codes = await _call(GRADE, {"design_path": path, "reference": ["0", "1", "?", "2"]})
    assert codes["ok"] is False
    assert "none of the 3 judged references (0, 1, 2)" in codes["error"]
    assert all(name in codes["error"] for name in _CODES)
    assert "as class_a and class_b write classes" in codes["error"]
    # A certified zone needs a random sample of one map, not this design.
    zone = await _call(
        "olmoearth_certify_zone",
        {"design_path": path, "wrong": [0] * 4, "alpha": 0.1},
    )
    assert zone["ok"] is True and zone["result"]["certified"] is False
    assert "where two maps differ" in zone["result"]["reason"]


# --------------------------------------------------------------------------- two scores files


def _scores_file(
    path: Path,
    classes: list[int],
    grid: list[int],
    centres: list[list[float]] | None = None,
) -> str:
    rows = [[1.0 if k == c else 0.0 for k in range(3)] for c in classes]
    data: dict[str, Any] = {"grid": grid, "scores": rows}
    if centres is not None:
        data["centres_lon_lat"] = centres
    path.write_text(json.dumps(data))
    return str(path)


@pytest.mark.asyncio
async def test_two_scores_files_of_one_grid_and_their_dates(tmp_path: Path) -> None:
    a = [0, 0, 1, 1, 2, 2, 0, 1, 2, 0, 1, 2, 0, 0, 1, 1]
    b = [0, 1, 1, 2, 2, 0, 0, 1, 1, 0, 1, 0, 0, 2, 1, 1]
    files = {
        "scores_path": _scores_file(tmp_path / "a.json", a, [4, 4]),
        "scores_path_b": _scores_file(tmp_path / "b.json", b, [4, 4]),
        "budget": 4,
        "seed": 3,
        "design": "disagreement",
    }
    out = await _call(PLAN, files)
    assert out["ok"] is True, out
    plan = out["result"]
    direct = estimate.sample_disagreement(a, b, 4, seed=3)
    assert _design(plan)["sample"]["indices"] == direct["indices"].tolist()
    assert plan["n_disagree"] == sum(x != y for x, y in zip(a, b))
    assert plan["map_b"]["scores_path"] == files["scores_path_b"]
    # No date given: read as unstated, and said.
    assert plan["dates"]["status"] == "unstated"
    assert UNSTATED_DATES_MUST_STATE in plan["must_state"]
    # Graded, the dates sentence keeps its place in the contract's three, so
    # rules.MAPS_NOT_MODELS is the one left out.
    idx = _design(plan)["sample"]["indices"]
    graded = await _call(
        GRADE,
        {"design_path": plan["design_path"], "reference": [str(a[i]) for i in idx]},
    )
    assert graded["ok"] is True, graded
    assert graded["result"]["must_state"] == [
        rules.which_map_must_state(graded["result"]),
        rules.NOT_EITHER_ACCURACY,
        UNSTATED_DATES_MUST_STATE,
    ]
    june = {"date_a": "2024-06-01", "date_b": "2024-06-01"}
    same = await _call(PLAN, {**files, **june})
    assert same["result"]["dates"]["status"] == "same_time"
    assert UNSTATED_DATES_MUST_STATE not in same["result"]["must_state"]
    months = {"date_a": "2024-03-01", "date_b": "2024-09-01"}
    apart = await _call(PLAN, {**files, **months})
    assert apart["ok"] is False
    assert "(dates_reading: different_time)" in apart["error"]
    half = await _call(PLAN, {**files, "date_a": "2024-03-01"})
    assert half["ok"] is False and "needs both maps' dates" in half["error"]
    year = await _call(PLAN, {**files, "date_a": "2024", "date_b": "2024"})
    assert year["ok"] is False and "'2024-01-01/2024-12-31'" in year["error"]
    other = _scores_file(tmp_path / "c.json", a, [2, 8])
    grids = await _call(PLAN, {**files, "scores_path_b": other})
    assert grids["ok"] is False and "not of one window grid" in grids["error"]


@pytest.mark.asyncio
async def test_dated_and_undated_plans_keep_their_own_designs(tmp_path: Path) -> None:
    """The design file's name follows what the grade reads from it.

    The same two maps planned at one budget and seed without dates, with
    dates, and with one map under another file name get three design files;
    each grade states its own plan's dates and names its own files, and the
    same arguments give the same file again.
    """
    a = [0, 0, 1, 1, 2, 2, 0, 1, 2, 0, 1, 2, 0, 0, 1, 1]
    b = [0, 1, 1, 2, 2, 0, 0, 1, 1, 0, 1, 0, 0, 2, 1, 1]
    files = {
        "scores_path": _scores_file(tmp_path / "a.json", a, [4, 4]),
        "scores_path_b": _scores_file(tmp_path / "b.json", b, [4, 4]),
        "budget": 4,
        "seed": 3,
        "design": "disagreement",
    }
    copy = _scores_file(tmp_path / "a_copy.json", a, [4, 4])

    async def plan(args: dict[str, Any]) -> dict[str, Any]:
        out = await _call(PLAN, args)
        assert out["ok"] is True, out
        return dict(out["result"])

    undated = await plan(files)
    dated = await plan({**files, "date_a": "2024-06-01", "date_b": "2024-06-01"})
    renamed = await plan({**files, "scores_path": copy})
    designs = [p["design_path"] for p in (undated, dated, renamed)]
    assert len(set(designs)) == 3, designs
    idx = _design(undated)["sample"]["indices"]
    assert _design(dated)["sample"]["indices"] == idx
    reference = [str(a[i]) for i in idx]

    async def grade(planned: dict[str, Any]) -> dict[str, Any]:
        args = {"design_path": planned["design_path"], "reference": reference}
        out = await _call(GRADE, args)
        assert out["ok"] is True, out
        return dict(out["result"])

    first, second, third = [await grade(p) for p in (undated, dated, renamed)]
    assert first["must_state"][2] == UNSTATED_DATES_MUST_STATE
    assert second["must_state"][2] == rules.MAPS_NOT_MODELS
    assert first["map_a"]["scores_path"] == files["scores_path"]
    assert third["map_a"]["scores_path"] == copy
    again = await plan(files)
    assert again["design_path"] == undated["design_path"]


@pytest.mark.asyncio
async def test_two_scores_files_of_one_shape_on_other_ground(tmp_path: Path) -> None:
    """One window grid is one shape and, where both files locate a window, one
    centre: two files of other areas are not compared window by window."""
    a, b = [0, 1] * 8, [1, 0] * 8
    here = [[10.0 + 0.01 * (i % 4), 50.0 - 0.01 * (i // 4)] for i in range(16)]
    there = [[lon + 0.5, lat] for lon, lat in here]
    plan = {
        "scores_path": _scores_file(tmp_path / "a.json", a, [4, 4], here),
        "budget": 4,
        "design": "disagreement",
    }
    on_the_grid = _scores_file(tmp_path / "b.json", b, [4, 4], here)
    same = await _call(PLAN, {**plan, "scores_path_b": on_the_grid})
    assert same["ok"] is True, same
    with open(same["result"]["labels_csv_path"], encoding="utf-8", newline="") as fh:
        assert {"lon", "lat"} <= set(next(csv.reader(fh)))
    elsewhere = _scores_file(tmp_path / "c.json", b, [4, 4], there)
    moved = await _call(PLAN, {**plan, "scores_path_b": elsewhere})
    assert moved["ok"] is False
    assert "not of one window grid" in moved["error"]
    assert "window 0 has another centre in each file" in moved["error"]
    assert "10.5" not in moved["error"]  # rule 1: no location in the reply


@pytest.mark.asyncio
async def test_map_b_needs_the_two_map_design(tmp_path: Path) -> None:
    path = _scores_file(tmp_path / "a.json", [0, 1] * 8, [4, 4])
    out = await _call(
        PLAN,
        {"scores_path": path, "scores_path_b": path, "budget": 4, "design": "random"},
    )
    assert out["ok"] is False and "pass design='disagreement'" in out["error"]
    alone = {"scores_path": path, "budget": 4, "design": "disagreement"}
    one = await _call(PLAN, alone)
    assert one["ok"] is False and "compares two maps of one time" in one["error"]


# --------------------------------------------------------------------------- the figures

_SVG = "{http://www.w3.org/2000/svg}"


def _figure_svg(result: dict[str, Any], stem: str, folder: Path) -> ET.Element:
    """The figure a result names, written as ``<stem>_figure.svg`` in ``folder``.

    Parsing it also checks it is a well-formed SVG document.
    """
    path = Path(result["figure_path"])
    assert path == folder.resolve() / f"{stem}_figure.svg"
    text = path.read_text(encoding="utf-8")
    svg = ET.fromstring(text)  # noqa: S314 -- the figure the tool just wrote
    assert svg.tag == f"{_SVG}svg"
    return svg


def _marked(svg: ET.Element, colour: str, cols: int) -> list[int]:
    """The windows a figure outlines or fills in ``colour``, by row-major index.

    A window's mark is drawn with a non-scaling stroke or is one window in
    size; the legend's swatches and the charts' bars are neither.
    """
    image = next(svg.iter(f"{_SVG}image"))
    scale = float(image.get("width", "0")) / cols
    marked = []
    for rect in svg.iter(f"{_SVG}rect"):
        if colour not in (rect.get("fill"), rect.get("stroke")):
            continue
        outlined = rect.get("vector-effect") == "non-scaling-stroke"
        filled = rect.get("width") == rect.get("height") == f"{scale:g}"
        if outlined or filled:
            row = int(float(rect.get("y", "0")) // scale)
            marked.append(row * cols + int(float(rect.get("x", "0")) // scale))
    return sorted(marked)


def _base(svg: ET.Element, window: int, cols: int) -> tuple[int, int, int]:
    """The grid's colour at ``window`` under the marks: that window's centre
    pixel in the PNG the figure embeds (8-bit RGB, filter 0 on every row, as
    the figures write it)."""
    image = next(svg.iter(f"{_SVG}image"))
    data = base64.b64decode(image.get("href", "").split(",", 1)[1])
    (width,) = struct.unpack(">I", data[16:20])
    idat, at = b"", 8
    while at < len(data):
        (n,) = struct.unpack(">I", data[at : at + 4])
        if data[at + 4 : at + 8] == b"IDAT":
            idat += data[at + 8 : at + 8 + n]
        at += 12 + n
    pixels = zlib.decompress(idat)
    scale = width // cols
    row, col = divmod(window, cols)
    y, x = row * scale + scale // 2, col * scale + scale // 2
    at = y * (1 + 3 * width) + 1 + 3 * x
    return pixels[at], pixels[at + 1], pixels[at + 2]


@pytest.mark.asyncio
async def test_the_plan_draws_where_the_maps_differ(tmp_path: Path) -> None:
    """The plan's figure: amber where the two maps' classes differ, the drawn
    windows outlined, on the grid both were sampled on. The design file keeps
    where they differ and where either map holds no class, so its grade
    draws the same grid."""
    plan = await _plan(6)
    design = _design(plan)
    pairs = [(_map_a(r, c), _map_b(r, c)) for r in range(GRID) for c in range(GRID)]
    differ = [i for i, (x, y) in enumerate(pairs) if x and y and x != y]
    assert design["population"]["differ"] == differ and len(differ) == N_DIFFER
    assert design["population"]["not_compared"] == [5]  # map b's no-data, (0, 5)
    svg = _figure_svg(plan, Path(plan["design_path"]).stem, tmp_path)
    assert _marked(svg, fg.OUTLINE, cols=GRID) == sorted(design["sample"]["indices"])
    assert _base(svg, differ[0], GRID) == fg.DIFFER
    assert _base(svg, 1, GRID) == fg.AGREE  # both maps say water there
    assert _base(svg, 5, GRID) == fg.NO_DATA
    caption = plan["figure_caption"]
    assert (
        f"(a) The {N_COMPARED} windows compared on its {GRID} x {GRID} window grid: "
        f"amber where the two maps' classes differ ({N_DIFFER})" in caption
    )
    assert "the 6 outlined windows are drawn at random" in caption


@pytest.mark.asyncio
async def test_the_grade_draws_the_side_each_label_took(tmp_path: Path) -> None:
    """The grade's figure: each drawn window filled with the side its label
    took (map a's class, map b's, neither) or outlined when not judged; the
    caption's counts are the package's and its verdict is the required
    sentence. A design file written before the plan kept where the maps
    differ is drawn with only its drawn windows as differing, every other
    window unknown."""
    plan = await _plan(20)  # every differing window: a census of them
    idx = _design(plan)["sample"]["indices"]
    refs = [_truth(*_window(i)) for i in idx]
    refs[0] = "?"
    args = {"design_path": plan["design_path"], "reference": refs}
    graded = (await _call(GRADE, args))["result"]
    digest = hashlib.sha256(json.dumps(refs).encode("utf-8")).hexdigest()[:8]
    stem = Path(plan["design_path"]).stem
    svg = _figure_svg(graded, f"{stem}_grade_{digest}", tmp_path)
    sides: dict[str, list[int]] = {"a": [], "b": [], "neither": []}
    for i, ref in zip(idx[1:], refs[1:]):
        a, b = _map_a(*_window(i)), _map_b(*_window(i))
        sides["a" if ref == a else "b" if ref == b else "neither"].append(i)
    assert [len(sides[s]) for s in ("a", "b", "neither")] == [
        graded["n_a_right"],
        graded["n_b_right"],
        graded["n_neither"],
    ]
    for side, colour in (("a", fg.MAP_A), ("b", fg.MAP_B), ("neither", fg.NEITHER)):
        assert _marked(svg, colour, cols=GRID) == sorted(sides[side]), side
    assert _marked(svg, fg.UNJUDGED, cols=GRID) == [idx[0]]
    caption = graded["figure_caption"]
    assert (
        f"blue where the label is map a's class ({graded['n_a_right']}), orange map "
        f"b's ({graded['n_b_right']}), grey neither ({graded['n_neither']}), "
        "outlined not judged (1)" in caption
    )
    assert graded["must_state"][0] == rules.which_map_must_state(graded)
    assert graded["must_state"][0] in caption

    # A design file written before the plan kept where the maps differ.
    old = _design(plan)
    del old["population"]["differ"], old["population"]["not_compared"]
    (tmp_path / "design_before.json").write_text(json.dumps(old))
    args_old = {**args, "design_path": str(tmp_path / "design_before.json")}
    before = (await _call(GRADE, args_old))["result"]
    old_svg = _figure_svg(before, f"design_before_grade_{digest}", tmp_path)
    assert before["figure_caption"] == caption  # the same labels, counts and verdict
    differing = fg.fade(fg.DIFFER, 0.45)
    assert _base(svg, idx[1], GRID) == _base(old_svg, idx[1], GRID) == differing
    assert _base(svg, 1, GRID) == fg.AGREE  # both maps say water there
    assert _base(old_svg, 1, GRID) == fg.NO_DATA  # not drawn, so unknown
    assert _base(svg, 5, GRID) == _base(old_svg, 5, GRID) == fg.NO_DATA
