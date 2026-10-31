# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""A Studio class map: measured from labels, never ranked or certified.

A classification result gives each window's class and no probability, so
nothing ranks its windows: it has no review set and no certified zone. A
sample drawn without its confidence still measures it.
``olmoearth_plan_label_sample`` reads it with the comparison's own sampler
and draws ``design='random'`` or ``design='class'`` (the package's condition
design by map class, ``class_floor`` labels per class at least), with
``plan.plan_error_rate``'s and ``plan.plan_class``'s label budgets; the
ranking designs are refused for it. ``olmoearth_estimate_map_error`` reports
``estimate_error_rate(valid=True)``'s interval valid at every sample size
first, then the design's, and ``estimate_per_class(valid=True)`` the same way
per class. ``olmoearth_certify_zone`` and the review set refuse it in one
sentence. These run the real package (``uv sync --all-extras``) and hold
every number to it.
"""

from __future__ import annotations

import csv
import json
import math
import random
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
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
from olmoearth_agent.tools import statistical_rules as rules
from olmoearth_agent.tools.estimation import (
    DEFAULT_MIN_PER_CLASS,
    _jsonable,
    build_estimation_tools,
)
from olmoearth_agent.tools.registry import ToolContext, ToolRegistry
from olmoearth_agent.tools.review_set import CLASS_READS, build_review_set_tools

estimate = pytest.importorskip("oe_inferencex.estimate")
plan_pkg = pytest.importorskip("oe_inferencex.plan")
wording = pytest.importorskip("oe_inferencex.common.wording")

PLAN = "olmoearth_plan_label_sample"
GRADE = "olmoearth_estimate_map_error"
CERTIFY = "olmoearth_certify_zone"
_BANNED_KEYS = {"lon", "lat", "bbox", "coordinates", "geometry", "centres_lon_lat"}

#: The result's box and grid (8 x 8 = 64 windows, one of them no-data).
_BOX = {"type": "Polygon", "coordinates": [[[0, 0], [4, 0], [4, 4], [0, 4], [0, 0]]]}
GRID = 8
_CODES = {"water": 1, "forest": 2, "crop": 3}
_LANDCOVER = {
    "classification_fields": [
        {
            "property_name": "landcover",
            "allowed_values": [{"value": v, "label": k} for k, v in _CODES.items()],
        }
    ]
}
#: The classes as the agent codes them: by name (_code_classes).
CLASSES = ["crop", "forest", "water"]
N_VALID = 63


def _map_class(row: int, col: int) -> str | None:
    """The map: water west of column 3, forest to column 6, crop in column 7.

    The window at (0, 7) is no-data. Water holds 24 windows, forest 32 and
    crop 7.
    """
    if (row, col) == (0, 7):
        return None
    if col == 7:
        return "crop"
    return "water" if col < 3 else "forest"


def _truth(row: int, col: int) -> str:
    """What is on the ground: the map's class, but forest is crop where row +
    col is a multiple of 4 (8 windows), and crop is forest in rows 1 to 3."""
    label = _map_class(row, col) or "water"
    if label == "forest" and (row + col) % 4 == 0:
        return "crop"
    if label == "crop" and row <= 3:
        return "forest"
    return label


class _Studio:
    """One classification result over the box, as the Studio client returns it.

    With ``declared`` off, the result declares no band type, so the band is
    known to be a classification only from the pixels sampled.
    """

    def __init__(
        self,
        labels: Callable[[int, int], str | None] = _map_class,
        *,
        declared: bool = True,
    ) -> None:
        self.labels = labels
        self.declared = declared
        self.cells = grid_windows([0, 0, 4, 4], GRID)
        self.pixel_calls = 0

    async def get_prediction_result(self, result_id: str) -> dict[str, Any]:
        meta: dict[str, Any] = {"geometry": _BOX}
        if self.declared:
            meta.update(_LANDCOVER)
        return {
            "id": result_id,
            "prediction_id": f"p-{result_id}",
            "property_names": ["landcover"],
            "result_metadata": meta,
        }

    async def get_prediction(self, prediction_id: str) -> dict[str, Any]:
        return {
            "id": prediction_id,
            "model_id": "m-1",
            "start_time": "2024-06-01T00:00:00Z",
            "end_time": "2024-06-30T00:00:00Z",
        }

    async def get_model(self, model_id: str) -> dict[str, Any]:
        return {"id": model_id, "name": "LandCover", "wizard_answers": {}}

    async def pixel_value(self, result_id: str, lon: float, lat: float) -> Any:
        self.pixel_calls += 1
        row, col, _lo, _la = min(
            self.cells, key=lambda c: (c[2] - lon) ** 2 + (c[3] - lat) ** 2
        )
        label = self.labels(row, col)
        band: dict[str, Any] = {"property_name": "landcover"}
        if label is None:
            band.update(raw_value=None, classification=None)
        else:
            band.update(
                raw_value=_CODES[label],
                classification={"label": label, "color": "#000000"},
            )
        return {"bands": [band]}


class _EdgeStudio(_Studio):
    """The same map, its west column outside the result's files (Studio answers
    404 there, as probe 5 found on 31 October 2026) and one call failing."""

    async def pixel_value(self, result_id: str, lon: float, lat: float) -> Any:
        row, col, _lo, _la = min(
            self.cells, key=lambda c: (c[2] - lon) ** 2 + (c[3] - lat) ** 2
        )
        if col == 0:
            request = httpx.Request("GET", "http://mock-studio/api/v1/pixel-value")
            response = httpx.Response(404, request=request)
            raise httpx.HTTPStatusError("HTTP 404", request=request, response=response)
        if (row, col) == (5, 5):
            raise RuntimeError("timeout")
        return await super().pixel_value(result_id, lon, lat)


@pytest.fixture(autouse=True)
def _scores_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("OLMOEARTH_SCORES_ROOT", str(tmp_path))
    monkeypatch.setenv("OLMOEARTH_OUTPUT_ROOT", str(tmp_path))
    return tmp_path


async def _call(name: str, args: dict[str, Any], studio: Any = None) -> Any:
    registry = ToolRegistry()
    registry.register_all(build_estimation_tools())
    registry.register_all(build_review_set_tools())
    ctx = ToolContext(studio=studio, state=ThreadState())
    return await registry.dispatch(ToolCall(id="1", name=name, arguments=args), ctx)


async def _plan(design: str, budget: int = 30, seed: int = 3, **extra: Any) -> Any:
    args = {
        "result_id": "lc",
        "grid": GRID,
        "budget": budget,
        "design": design,
        "seed": seed,
        **extra,
    }
    out = await _call(PLAN, args, _Studio())
    assert out["ok"] is True, out
    return out["result"]


def _design(plan: dict[str, Any]) -> dict[str, Any]:
    return json.loads(Path(plan["design_path"]).read_text())


def _population() -> tuple[list[float], list[int]]:
    """The class map's margin (0 where a window holds a class) and its codes."""
    labels = [_map_class(r, c) for r in range(GRID) for c in range(GRID)]
    margin = [0.0 if v else math.nan for v in labels]
    codes = [CLASSES.index(v) if v else -1 for v in labels]
    return margin, codes


def _seen(indices: list[int]) -> list[str]:
    """The class a reviewer sees in each window, in the design's order."""
    return [_truth(*divmod(i, GRID)) for i in indices]


def _keys(obj: Any) -> set[str]:
    if isinstance(obj, dict):
        return set(obj).union(*(_keys(v) for v in obj.values()))
    if isinstance(obj, list):
        return set().union(*(_keys(v) for v in obj))
    return set()


def _ids(result: dict[str, Any]) -> list[str]:
    return [c["id"] for c in result.get("forbidden_claims", [])]


def _fact(result: dict[str, Any], fact_id: str) -> dict[str, Any]:
    (entry,) = [f for f in result["facts"] if f["id"] == fact_id]
    return entry


# --------------------------------------------------------------------------- the plan


@pytest.mark.asyncio
async def test_a_class_map_is_planned_under_the_random_design() -> None:
    """Each window's class, read by the pair sampler; the package's random draw."""
    plan = await _plan("random")
    design = _design(plan)
    margin, codes = _population()
    direct = estimate.sample_for_estimation(margin, 30, design="random", seed=3)
    assert design["sample"]["indices"] == direct["indices"].tolist()
    assert [w["window_index"] for w in plan["windows"]] == direct["indices"].tolist()
    assert plan["n_population"] == N_VALID
    population = design["population"]
    assert population["score_kind"] == "class_map"
    assert population["classes"] == CLASSES
    assert population["map_class"] == codes
    # Each window shows its class as the map writes it; no coordinates.
    for window in plan["windows"]:
        row, col = divmod(window["window_index"], GRID)
        assert (window["row"], window["col"]) == (row, col)
        assert window["map_class"] == _map_class(row, col)
    assert not _BANNED_KEYS & _keys(plan)
    sampling = plan["sampling"]
    assert (sampling["n_windows_sampled"], sampling["n_valid"]) == (64, N_VALID)
    assert sampling["n_nodata_dropped"] == 1
    assert plan["map_classes"] == {"0": "crop", "1": "forest", "2": "water"}
    assert plan["class_map_note"] == rules.CLASS_MAP_NO_CONFIDENCE
    assert rules.CLASS_MAP_NO_CONFIDENCE in plan["next_steps"]
    ids = _ids(plan)
    assert rules.ZONE_FOR_A_CLASS_MAP in ids
    assert rules.CERTIFICATION_GUARANTEED not in ids
    assert "prefix_is_random_sample" in {f["id"] for f in plan["facts"]}
    header = next(csv.reader(Path(plan["labels_csv_path"]).open()))
    assert {"map_class", "lon", "lat", "reference_class"} <= set(header)
    # A random sheet is in random order: hiding the map's class is enough.
    assert "To label blind, hide map_class." in plan["how_to_label"]
    assert "shuffled" not in plan["how_to_label"]


@pytest.mark.asyncio
async def test_a_class_maps_points_outside_its_files_are_counted_apart() -> None:
    """Probe 5 (31 October 2026): the kelp grid's "failed" points were HTTP
    404s where no file of the result lies. The sample counts them apart; any
    other failure stays a failed call, and no-data stays no-data."""
    args = {"result_id": "lc", "grid": GRID, "budget": 20, "design": "random"}
    out = await _call(PLAN, args, _EdgeStudio())
    assert out["ok"] is True, out
    sampling = out["result"]["sampling"]
    # 64 points: column 0 outside (8), (5, 5) failed (1), (0, 7) no-data (1).
    assert sampling["n_outside_dropped"] == 8
    assert sampling["n_failed_dropped"] == 1
    assert sampling["n_nodata_dropped"] == 1
    assert sampling["n_valid"] == 54
    note = sampling["outside_note"]
    assert "outside the result's files (Studio answered 404)" in note


@pytest.mark.asyncio
async def test_a_class_map_is_planned_under_the_class_design() -> None:
    """The package's condition design by map class, marked as its CLI marks it."""
    plan = await _plan("class", min_per_class=5)
    design = _design(plan)
    margin, codes = _population()
    direct = estimate.sample_for_estimation(
        margin,
        30,
        design="condition",
        seed=3,
        condition=codes,
        condition_names=dict(enumerate(CLASSES)),
        class_floor=5,
    )
    sample = design["sample"]
    assert sample["indices"] == direct["indices"].tolist()
    assert sample["design"] == "condition"
    assert sample["condition"]["by"] == "map class"
    assert sample["condition"]["min_per_class"] == 5
    assert sample["condition"]["names"] == CLASSES
    # Each class at least 5 labels (crop all 7), the rest by size, each
    # stratum named by its class as the map writes it, with the agent's code
    # (agent trial 12 read stratum h as class h).
    assert direct["allocation"] == [7, 12, 11]
    assert plan["per_stratum"] == [
        {"stratum": 0, "class": "crop", "code": 0, "windows": 7, "labels": 7},
        {"stratum": 1, "class": "forest", "code": 1, "windows": 32, "labels": 12},
        {"stratum": 2, "class": "water", "code": 2, "windows": 24, "labels": 11},
    ]
    assert "strata_sizes" not in plan and "allocation" not in plan
    assert plan["min_per_class"] == 5
    assert plan["class_blind_note"] == estimate.CLASS_BLIND_NOTE
    assert "at least 5 labels" in plan["design_note"]
    # The whole map's valid interval first, then the union bound: the
    # design_note does not call the union bound the whole-map interval.
    note = plan["design_note"]
    assert note.index("valid at every sample size") < note.index("union bound")
    # The sheet runs class by class, so labelling blind needs a shuffled
    # display, as the package's CLASS_BLIND_NOTE says; the agent's sheet has
    # map_class and stratum, and no condition column.
    how = plan["how_to_label"]
    assert "hide map_class and stratum" in how and "shuffled order" in how
    assert "no condition column" in how
    header = next(csv.reader(Path(plan["labels_csv_path"]).open()))
    assert "condition" not in header and {"map_class", "stratum"} <= set(header)
    ids = _ids(plan)
    assert rules.ZONE_FOR_A_CLASS_MAP in ids
    # The sheet runs class by class, not least confident first.
    why = _why(plan, rules.SUBSET_LABELLING_SUFFICIENT)
    assert "the map's classes, in turn" in why
    assert "least confident" not in why


def _why(result: dict[str, Any], claim_id: str) -> str:
    (why,) = [c["why"] for c in result["forbidden_claims"] if c["id"] == claim_id]
    return str(why)


def _numeric(row: int, col: int) -> int | None:
    """The map with numeric classes, one id missing: water 1, forest 2 (3 in
    rows 0 and 1), crop 5; 24, 24, 8 and 7 windows."""
    label = _map_class(row, col)
    if label is None:
        return None
    if label == "forest" and row < 2:
        return 3
    return {"water": 1, "forest": 2, "crop": 5}[label]


class _NumericStudio(_Studio):
    """The same result, its classes numeric ids with no label (Studio's
    classification object then carries the value only)."""

    async def get_prediction_result(self, result_id: str) -> dict[str, Any]:
        out = await super().get_prediction_result(result_id)
        values = [{"value": v} for v in (1, 2, 3, 5)]
        out["result_metadata"]["classification_fields"] = [
            {"property_name": "landcover", "allowed_values": values}
        ]
        return out

    async def pixel_value(self, result_id: str, lon: float, lat: float) -> Any:
        self.pixel_calls += 1
        row, col, _lo, _la = min(
            self.cells, key=lambda c: (c[2] - lon) ** 2 + (c[3] - lat) ** 2
        )
        value = _numeric(row, col)
        band: dict[str, Any] = {"property_name": "landcover"}
        if value is None:
            band.update(raw_value=None, classification=None)
        else:
            band.update(
                raw_value=value, classification={"value": value, "color": "#000000"}
            )
        return {"bands": [band]}


@pytest.mark.asyncio
async def test_a_numeric_class_maps_strata_name_the_classes_it_writes() -> None:
    """The step 11 review: a class map writing ids [1, 2, 3, 5] is coded 0 to
    3 over its classes, so stratum 3 is code 3, which is class 5. per_stratum
    names each stratum's class as the map writes it, the code beside it."""
    args = {"result_id": "lc", "grid": GRID, "budget": 30, "design": "class"}
    out = await _call(PLAN, {**args, "seed": 3, "min_per_class": 5}, _NumericStudio())
    assert out["ok"] is True, out
    plan = out["result"]
    sample = _design(plan)["sample"]
    strata = plan["per_stratum"]
    assert [s["class"] for s in strata] == [1, 2, 3, 5]
    assert [s["code"] for s in strata] == [0, 1, 2, 3]
    assert [s["windows"] for s in strata] == [24, 24, 8, 7] == sample["sizes"]
    assert [s["labels"] for s in strata] == sample["allocation"]
    assert plan["map_classes"] == {"0": 1, "1": 2, "2": 3, "3": 5}
    # Each listed window shows its class as the map writes it.
    for window in plan["windows"]:
        assert window["map_class"] == _numeric(*divmod(window["window_index"], GRID))


@pytest.mark.asyncio
async def test_the_class_design_floor_is_the_packages() -> None:
    """The schema's default is the package's CLASS_FLOOR, and a budget below
    the floors is refused in the package's words."""
    assert DEFAULT_MIN_PER_CLASS == estimate.CLASS_FLOOR
    out = await _call(
        PLAN,
        {"result_id": "lc", "grid": GRID, "budget": 30, "design": "class"},
        _Studio(),
    )
    assert out["ok"] is False
    assert "cannot give each of the 3 map classes" in out["error"]
    assert "min_per_class" in out["error"]


@pytest.mark.asyncio
@pytest.mark.parametrize("design", ["confidence", "proportional"])
async def test_a_class_map_is_refused_under_a_ranking_design(design: str) -> None:
    """No confidence to stratify by: refused before any pixel is sampled."""
    studio = _Studio()
    out = await _call(PLAN, {"result_id": "lc", "budget": 20, "design": design}, studio)
    assert out["ok"] is True
    result = out["result"]
    assert result["ranked"] is False and result["class_map"] is True
    # The class-map sentence, then where top-k scores come from.
    assert result["reason"].endswith(rules.CLASS_MAP_REFUSAL)
    assert "confidence margin" in result["reason"]
    assert result["designs_for_a_class_map"] == ["random", "class"]
    # The class design's floor, as the package allocates it: a smaller class
    # gets all its windows.
    assert "or all the windows of a smaller class" in result["next_steps"][0]
    assert _ids(result) == [
        rules.ERROR_RATE_WITHOUT_LABELS,
        rules.ZONE_FOR_A_CLASS_MAP,
    ]
    assert studio.pixel_calls == 0


@pytest.mark.asyncio
async def test_an_undeclared_class_band_is_sampled_once() -> None:
    """A result that declares no band type is known to be a class map only
    once its grid is sampled: the plan reads its classes from that sampling
    (one pixel-value call per grid point, not two), and draws what the same
    map declared draws. A ranking design is refused only after that sampling."""
    declared_studio = _Studio()
    args = {"result_id": "lc", "grid": GRID, "budget": 30, "design": "random"}
    declared = await _call(PLAN, {**args, "seed": 3}, declared_studio)
    assert declared["ok"] is True, declared
    assert declared_studio.pixel_calls == GRID * GRID
    before = _design(declared["result"])
    studio = _Studio(declared=False)
    out = await _call(PLAN, {**args, "seed": 3}, studio)
    assert out["ok"] is True, out
    assert studio.pixel_calls == GRID * GRID
    plan = out["result"]
    assert plan["windows"] == declared["result"]["windows"]
    assert plan["sampling"] == declared["result"]["sampling"]
    after = _design(plan)
    assert after["population"] == before["population"]
    assert after["sample"] == before["sample"]
    assert CLASS_READS not in _keys(out)
    ranked = _Studio(declared=False)
    refused = await _call(PLAN, {**args, "design": "confidence"}, ranked)
    assert refused["ok"] is True and refused["result"]["class_map"] is True
    assert ranked.pixel_calls == GRID * GRID
    assert CLASS_READS not in _keys(refused)


@pytest.mark.asyncio
async def test_a_class_map_is_refused_under_the_sequential_design() -> None:
    """The sequential design is not offered: refused before any Studio call."""
    studio = _Studio()
    out = await _call(
        PLAN, {"result_id": "lc", "budget": 20, "design": "sequential"}, studio
    )
    assert out["ok"] is False
    assert "must be one of" in out["error"] and "sequential" in out["error"]
    assert studio.pixel_calls == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("design", ["random", "class"])
async def test_the_plan_numbers_are_the_packages(design: str) -> None:
    """plan.plan_error_rate at the stated width, and plan.plan_class for a
    stated class share, as the package returns them."""
    plan = await _plan(design, min_per_class=5, width=0.2, class_share=0.1)
    rate = _jsonable(plan_pkg.plan_error_rate(N_VALID, 0.2, None))
    advice = plan["label_budget"]
    assert advice["planned_by"] == "oe_inferencex.plan.plan_error_rate"
    for key, value in advice.items():
        if key != "planned_by":
            assert value == rate[key], key
    assert advice["labels"] == rate["labels"] is not None
    fact = _fact(plan, "label_budget")
    assert (fact["labels"], fact["width"]) == (rate["labels"], 0.2)
    sentence = fact["sentence"]
    # No rate stated: the package's 50%, named as an assumption.
    assert rate["error_rate_stated"] is False
    assert "assumed wrong 50.0% of the time" in sentence
    # The class design: the budget plans a random sample's interval, and the
    # union bound's width against it is the package's conditional statement.
    assert ("union bound" in sentence) is (design == "class")
    assert ("is for a random sample's interval" in sentence) is (design == "class")
    if design == "class":
        assert "wider than a random sample's interval when the classes'" in sentence
        assert "can be narrower when they are far apart" in sentence
    by_class = _jsonable(plan_pkg.plan_class(N_VALID, 0.1, 3))
    one = plan["class_budget"]
    assert one["planned_by"] == "oe_inferencex.plan.plan_class"
    for key, value in one.items():
        if key != "planned_by":
            assert value == by_class[key], key
    assert _fact(plan, "class_budget")["labels"] == by_class["labels"]


@pytest.mark.asyncio
async def test_the_default_width_is_ten_points() -> None:
    plan = await _plan("random")
    direct = _jsonable(plan_pkg.plan_error_rate(N_VALID, 0.10, None))
    assert plan["label_budget"]["width"] == 0.1
    assert plan["label_budget"]["labels"] == direct["labels"]
    assert "class_budget" not in plan


@pytest.mark.asyncio
async def test_a_stated_error_rate_is_not_called_an_assumption() -> None:
    """The label budget planned at the caller's expected error rate states it
    as the map's rate; only the package's default is named an assumption."""
    plan = await _plan("random", width=0.2, expected_error_rate=0.2)
    direct = _jsonable(plan_pkg.plan_error_rate(N_VALID, 0.2, 0.2))
    assert direct["error_rate_stated"] is True and direct["labels"] is not None
    assert plan["label_budget"]["labels"] == direct["labels"]
    sentence = _fact(plan, "label_budget")["sentence"]
    assert "of a map wrong 20.0% of the time" in sentence
    assert "assumed" not in sentence


# --------------------------------------------------------------------------- the estimate


async def _graded(design: str) -> tuple[dict[str, Any], Any]:
    """A plan of ``design``, labelled with the class seen, and its estimate."""
    plan = await _plan(design, budget=40, seed=5, min_per_class=5)
    sample = _design(plan)["sample"]
    out = await _call(
        GRADE,
        {"design_path": plan["design_path"], "reference": _seen(sample["indices"])},
    )
    assert out["ok"] is True, out
    return plan, out["result"]


@pytest.mark.asyncio
@pytest.mark.parametrize("design", ["random", "class"])
async def test_the_estimate_reports_the_valid_interval_first(design: str) -> None:
    """estimate_error_rate(valid=True) and estimate_per_class(valid=True) on
    the same sample, in the package's words; the valid interval first."""
    plan, result = await _graded(design)
    stored = _design(plan)
    sample, population = stored["sample"], stored["population"]
    indices = sample["indices"]
    codes = [CLASSES.index(c) for c in _seen(indices)]
    wrong = [int(c != population["map_class"][i]) for c, i in zip(codes, indices)]
    words = wording.class_wording(sample)
    direct = words(_jsonable(estimate.estimate_error_rate(sample, wrong, valid=True)))
    for key, value in direct.items():
        assert result[key] == value, key
    keys = list(result)
    assert keys.index("valid_interval") < keys.index("estimate")
    assert keys.index("valid_interval") < keys.index("low")
    first, second = result["intervals"]
    assert first["interval"] == "valid" and second["interval"] == "design"
    assert (first["low"], first["high"]) == (
        direct["valid_interval"]["low"],
        direct["valid_interval"]["high"],
    )
    assert (second["low"], second["high"], second["method"]) == (
        direct["low"],
        direct["high"],
        direct["method"],
    )
    assert result["valid_note"] == estimate.VALID_NOTE
    assert result["plan_design"] == design
    # Neither design's interval is approximate: under a random sample the two
    # are the same exact interval (VALID_NOTE), under the class design the
    # second is the union bound over the classes.
    report = result["report"]
    assert "approximate" not in report
    assert "not every pixel" in report
    if design == "random":
        assert (first["low"], first["high"]) == (second["low"], second["high"])
        assert "the same exact interval" in report
    if design == "class":
        assert "union bound over the classes" in report
        assert result["design"] == "class"
        assert result["condition_by"] == "map class"
        assert "by map class" in result["valid_interval"]["method"]
    # Per class: the package's table, each class's valid intervals first.
    per_class = words(
        _jsonable(
            estimate.estimate_per_class(
                sample, codes, population["map_class"], n_classes=3, valid=True
            )
        )
    )
    table = result["per_class"]
    assert list(table)[0] == "overall_accuracy_valid"
    assert table["overall_accuracy_valid"] == per_class["overall_accuracy_valid"]
    assert table["valid_note"] == estimate.PER_CLASS_VALID_NOTE
    for key, row in per_class["per_class"].items():
        got = table["per_class"][key]
        assert list(got)[:2] == ["class", "valid"]
        assert got["class"] == CLASSES[int(key)]
        assert {k: v for k, v in got.items() if k != "class"} == row
    # The contract: the valid interval and the design; no zone for a class map.
    valid = direct["valid_interval"]
    assert result["must_state"][0] == rules.valid_interval_must_state(
        valid["estimate"],
        valid["low"],
        valid["high"],
        design,
        len(indices),
        class_map=True,
    )
    assert any("sampled grid points" in s for s in result["must_state"])
    whole = _fact(result, "whole_map_estimate")
    assert (whole["low"], whole["high"]) == (valid["low"], valid["high"])
    assert (whole["design_low"], whole["design_high"]) == (
        direct["low"],
        direct["high"],
    )
    sentence = whole["sentence"]
    assert sentence.index("valid at every sample") < sentence.index("this design earns")
    ids = _ids(result)
    assert rules.ZONE_FOR_A_CLASS_MAP in ids
    assert rules.CERTIFICATION_GUARANTEED not in ids
    assert rules.CLASS_MAP_NO_CONFIDENCE in result["next_steps"]


@pytest.mark.asyncio
async def test_the_labelling_sheet_gives_the_same_estimate() -> None:
    """The sheet's reference_class, as the map writes classes, is the route the
    reviewer takes; wrong is read from it."""
    plan, by_list = await _graded("random")
    sheet = Path(plan["labels_csv_path"])
    rows = list(csv.DictReader(sheet.open()))
    for row in rows:
        row["reference_class"] = _truth(int(row["row"]), int(row["col"]))
    with sheet.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    out = await _call(
        GRADE, {"design_path": plan["design_path"], "labels_path": str(sheet)}
    )
    assert out["ok"] is True, out
    assert out["result"]["valid_interval"] == by_list["valid_interval"]
    assert out["result"]["per_class"] == by_list["per_class"]


def _fill_sheet(
    plan: dict[str, Any], fill: Callable[[int, dict[str, str]], None]
) -> str:
    """The plan's sheet with each row passed to ``fill(order, row)``; its path."""
    sheet = Path(plan["labels_csv_path"])
    with sheet.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    for row in rows:
        fill(int(row["order"]), row)
    with sheet.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return str(sheet)


def _seen_in(row: dict[str, str]) -> str:
    """The class a reviewer sees in a sheet's window."""
    return _truth(int(row["row"]), int(row["col"]))


def _wrong_in(row: dict[str, str]) -> str:
    """The sheet's wrong for a window: 1 where the class seen is not map_class."""
    return str(int(_seen_in(row) != row["map_class"]))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case, refusal, at",
    [
        ("one class missing", "1 of 40 design windows have no reference_class", 7),
        ("classes, or wrong", "20 of 40 design windows have no reference_class", 1),
        ("wrong flipped", "wrong disagrees with the class seen on 1 of 40", 3),
        ("wrong partly filled", "wrong is filled on 39 of 40 design windows", 9),
    ],
)
async def test_a_sheet_whose_columns_do_not_hold_together_is_refused(
    case: str, refusal: str, at: int
) -> None:
    """The sheet is held to what the arguments are: a reference_class filled
    on some windows only is refused, naming the first window without one
    (whatever wrong holds there); a wrong column filled beside a full
    reference_class must be full and agree with it window by window."""
    plan = await _plan("random", budget=40, seed=5)
    indices = _design(plan)["sample"]["indices"]

    def fill(order: int, row: dict[str, str]) -> None:
        if case == "one class missing":
            row["reference_class"] = "" if order == at else _seen_in(row)
        elif case == "classes, or wrong":
            # Even rows the class seen only, odd rows wrong only.
            if order % 2:
                row["wrong"] = _wrong_in(row)
            else:
                row["reference_class"] = _seen_in(row)
        elif case == "wrong flipped":
            row["reference_class"] = _seen_in(row)
            wrong = int(_wrong_in(row))
            row["wrong"] = str(1 - wrong if order == at else wrong)
        else:
            row["reference_class"] = _seen_in(row)
            row["wrong"] = "" if order == at else _wrong_in(row)

    path = _fill_sheet(plan, fill)
    out = await _call(GRADE, {"design_path": plan["design_path"], "labels_path": path})
    assert out["ok"] is False, out
    assert refusal in out["error"]
    assert f"(first: window {indices[at]})" in out["error"]


@pytest.mark.asyncio
async def test_a_sheet_with_both_columns_or_wrong_alone_is_graded() -> None:
    """A wrong column that agrees with a full reference_class gives the
    estimate the classes give; wrong alone, with reference_class blank on
    every window, gives the same rate and no per-class table."""
    plan, by_list = await _graded("random")

    def both(_order: int, row: dict[str, str]) -> None:
        row["reference_class"] = _seen_in(row)
        row["wrong"] = _wrong_in(row)

    path = _fill_sheet(plan, both)
    out = await _call(GRADE, {"design_path": plan["design_path"], "labels_path": path})
    assert out["ok"] is True, out
    assert out["result"]["valid_interval"] == by_list["valid_interval"]
    assert out["result"]["per_class"] == by_list["per_class"]

    def wrong_only(_order: int, row: dict[str, str]) -> None:
        row["reference_class"] = ""
        row["wrong"] = _wrong_in(row)

    path = _fill_sheet(plan, wrong_only)
    alone = await _call(
        GRADE, {"design_path": plan["design_path"], "labels_path": path}
    )
    assert alone["ok"] is True, alone
    assert alone["result"]["valid_interval"] == by_list["valid_interval"]
    assert "per_class" not in alone["result"]


@pytest.mark.asyncio
async def test_wrong_alone_gives_the_rate_and_must_agree_with_the_classes() -> None:
    plan = await _plan("random", budget=40, seed=5)
    stored = _design(plan)
    indices = stored["sample"]["indices"]
    seen = _seen(indices)
    wrong = [int(s != _map_class(*divmod(i, GRID))) for s, i in zip(seen, indices)]
    alone = await _call(GRADE, {"design_path": plan["design_path"], "wrong": wrong})
    assert alone["ok"] is True
    direct = _jsonable(
        estimate.estimate_error_rate(stored["sample"], wrong, valid=True)
    )
    assert alone["result"]["valid_interval"] == direct["valid_interval"]
    assert "per_class" not in alone["result"]
    flipped = [1 - wrong[0], *wrong[1:]]
    clash = await _call(
        GRADE,
        {"design_path": plan["design_path"], "wrong": flipped, "reference": seen},
    )
    assert clash["ok"] is False
    assert "wrong disagrees with the class seen on 1 of 40 windows" in clash["error"]
    unjudged = await _call(
        GRADE, {"design_path": plan["design_path"], "reference": ["?", *seen[1:]]}
    )
    assert unjudged["ok"] is False and "'?'" in unjudged["error"]


@pytest.mark.asyncio
async def test_a_class_the_map_never_writes_is_its_own_row() -> None:
    """A class seen that the map never draws gets a code after the map's (the
    package's never-predicted class); codes in place of names are refused."""
    plan = await _plan("random", budget=40, seed=5)
    stored = _design(plan)
    indices = stored["sample"]["indices"]
    seen = _seen(indices)
    seen[0] = "wetland"
    out = await _call(GRADE, {"design_path": plan["design_path"], "reference": seen})
    assert out["ok"] is True, out
    result = out["result"]
    table = [*CLASSES, "wetland"]
    assert result["classes"] == {str(c): v for c, v in enumerate(table)}
    assert result["references_outside_the_maps_classes"] == ["wetland"]
    codes = [(CLASSES + ["wetland"]).index(s) for s in seen]
    direct = _jsonable(
        estimate.estimate_per_class(
            stored["sample"],
            codes,
            stored["population"]["map_class"],
            n_classes=4,
            valid=True,
        )
    )
    row = result["per_class"]["per_class"]["3"]
    assert row["class"] == "wetland"
    assert row["producer_accuracy"] == direct["per_class"]["3"]["producer_accuracy"]
    assert row["producer_accuracy"]["estimate"] == 0.0
    numbered = await _call(
        GRADE,
        {"design_path": plan["design_path"], "reference": [str(c) for c in codes]},
    )
    assert numbered["ok"] is False
    assert "is a class the map writes" in numbered["error"]


# --------------------------------------------------------------------------- refusals


@pytest.mark.asyncio
@pytest.mark.parametrize("design", ["random", "class"])
async def test_certify_refuses_a_class_map_in_one_sentence(design: str) -> None:
    plan = await _plan(design, budget=30, min_per_class=5)
    out = await _call(
        CERTIFY, {"design_path": plan["design_path"], "wrong": [0] * 30, "alpha": 0.1}
    )
    assert out["ok"] is True
    result = out["result"]
    assert result["certified"] is False and result["class_map"] is True
    # The class-map sentence, then where top-k scores come from.
    assert result["reason"] == rules.CLASS_MAP_REFUSAL
    assert _ids(result) == [rules.ZONE_FOR_A_CLASS_MAP]
    # A random design does not help a class map: no step says it would.
    assert "design_requirement" not in result
    assert not any("design='random'" in s for s in result["next_steps"])


@pytest.mark.asyncio
async def test_the_review_set_refuses_a_class_map_in_one_sentence() -> None:
    studio = _Studio()
    out = await _call("olmoearth_review_set_from_result", {"result_id": "lc"}, studio)
    assert out["ok"] is True
    result = out["result"]
    assert result["ranked"] is False
    assert result["reason"] == rules.CLASS_MAP_REFUSAL
    assert _ids(result) == [rules.ZONE_FOR_A_CLASS_MAP]
    assert studio.pixel_calls == 0


# --------------------------------------------------------------------------- the answer checks

#: The keys whose sentences an answer repeats (as
#: tests/harness/test_tool_sentences_pass_the_checks.py reads them).
_NOTES = ("next_steps", "design_note", "design_requirement", "caveats")


def _said(result: dict[str, Any]) -> list[str]:
    """Every sentence of a result an answer may repeat."""
    facts = [f["sentence"] for f in result.get("facts", []) if f.get("sentence")]
    notes: list[str] = []
    for key in _NOTES:
        value = result.get(key)
        items = value if isinstance(value, list) else [value]
        notes += [s for s in items if isinstance(s, str)]
    return facts + list(result.get("must_state", [])) + notes


def _run(name: str, result: dict[str, Any]) -> RunEvidence:
    return RunEvidence(
        tools=[ToolRecord(name, {}, {"ok": True, "result": result})],
        user_messages=["How wrong is this land-cover map? I can label 40 windows."],
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("design", ["random", "class"])
async def test_a_class_maps_own_sentences_pass_the_checks(design: str) -> None:
    """An answer made of the tools' sentences is not flagged, the new
    zone_for_a_class_map claim and the valid-interval statement included."""
    plan = await _plan(design, budget=40, seed=5, min_per_class=5, class_share=0.1)
    _plan_run, result = await _graded(design)
    certify = await _call(
        CERTIFY, {"design_path": plan["design_path"], "wrong": [0] * 40, "alpha": 0.1}
    )
    for name, out in (
        (PLAN, plan),
        (GRADE, result),
        (CERTIFY, certify["result"]),
    ):
        said = _said(out)
        assert said, out
        found = run_checks("\n\n".join(said), _run(name, out))
        assert found == {}, json.dumps(found, indent=1)


@pytest.mark.asyncio
async def test_an_accuracy_without_its_interval_or_a_zone_is_flagged() -> None:
    """The contract an answer about a class map's accuracy is held to: it
    states the valid interval and the design, and offers no certified zone."""
    _plan_run, result = await _graded("random")
    run = _run(GRADE, result)
    bare = f"The map's error rate is {100 * result['estimate']:.2f}%."
    assert check_must_state(bare, run), bare
    stated = " ".join(result["must_state"])
    assert check_must_state(stated, run) == []
    offer = "With 200 more random labels I can certify the top 30% of this map."
    flagged = check_forbidden_claims(offer, run)
    assert [v["text"] for v in flagged] == [offer]
    assert flagged[0]["detail"].startswith(rules.ZONE_FOR_A_CLASS_MAP)


# --------------------------------------------------------------------------- the class design of a scores map


def _logits(n: int = 400, seed: int = 4) -> list[list[float]]:
    rng = random.Random(seed)
    return [[rng.gauss(0.0, 1.5) for _ in range(3)] for _ in range(n)]


@pytest.mark.asyncio
async def test_a_scores_maps_class_design_cannot_certify() -> None:
    """The class design reads a scores map's arg-max class; its estimate is
    valid-first; a certified zone needs a random design, as for any stratified
    design."""
    scores = _logits()
    out = await _call(
        PLAN,
        {"scores": scores, "budget": 120, "design": "class", "min_per_class": 20},
    )
    assert out["ok"] is True, out
    plan = out["result"]
    stored = _design(plan)
    classes = stored["population"]["map_class"]
    assert classes == [max(range(3), key=row.__getitem__) for row in scores]
    direct = estimate.sample_for_estimation(
        stored["population"]["margin"],
        120,
        design="condition",
        seed=0,
        condition=classes,
        condition_names={c: f"class {c}" for c in sorted(set(classes))},
        class_floor=20,
    )
    assert stored["sample"]["indices"] == direct["indices"].tolist()
    assert stored["sample"]["condition"]["by"] == "map class"
    assert "label_budget" in plan and "map_classes" not in plan
    indices = stored["sample"]["indices"]
    reference = [
        classes[i] if k % 5 else (classes[i] + 1) % 3 for k, i in enumerate(indices)
    ]
    wrong = [int(r != classes[i]) for r, i in zip(reference, indices)]
    graded = await _call(
        GRADE,
        {
            "design_path": plan["design_path"],
            "wrong": wrong,
            "reference": reference,
            "n_classes": 3,
        },
    )
    assert graded["ok"] is True, graded
    result = graded["result"]
    words = wording.class_wording(stored["sample"])
    direct = estimate.estimate_error_rate(stored["sample"], wrong, valid=True)
    assert result["valid_interval"] == words(_jsonable(direct))["valid_interval"]
    assert list(result).index("valid_interval") < list(result).index("estimate")
    ids = _ids(result)
    assert rules.CERTIFY_FROM_NONRANDOM_DESIGN in ids
    assert rules.ZONE_FOR_A_CLASS_MAP not in ids
    certify = await _call(
        CERTIFY,
        {"design_path": plan["design_path"], "wrong": wrong, "alpha": 0.1},
    )
    assert certify["result"]["certified"] is False
    assert "simple random sample" in certify["result"]["reason"]
