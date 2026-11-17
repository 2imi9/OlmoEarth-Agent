# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Windows a reviewer could not judge: ``?`` in a one-map sheet's ``wrong`` column,
or in a class map's class seen (``reference``, the sheet's ``reference_class``).

``olmoearth_estimate_map_error`` reads a ``?`` as the package does
(``estimate_error_rate(unjudged=, valid=True)``): right at each interval's
lower end, wrong at its upper. With any ``?`` it gives no per-class table and
no figure, states the count in ``must_state`` after the valid interval, and
refuses it on a ``seed_from_map`` plan before the plan's labels are fixed.
The zone and the certificate need 1 or 0 on every window and refuse a ``?``
by name. These run the real package (``uv sync --all-extras``); the
certificate plan runs its command line as a subprocess, as
``test_certificate_tool.py`` does. A class map's ``?`` graded end to end, on
the random and the class design, is held in ``test_class_map_tool.py``, whose
Studio stub draws the class map.
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any

import pytest

from olmoearth_agent.harness.state import ThreadState
from olmoearth_agent.llm.types import ToolCall
from olmoearth_agent.tools import statistical_rules as rules
from olmoearth_agent.tools.certificate import build_certificate_tools
from olmoearth_agent.tools.estimation import (
    UNJUDGED_HOW,
    UNJUDGED_STRATIFIED_HOW,
    VALID_FIRST_HOW,
    _class_map_labels,
    _class_map_labels_unjudged,
    _jsonable,
    _labels,
    _labels_unjudged,
    _wrong_cells,
    build_estimation_tools,
    unjudged_must_state,
)
from olmoearth_agent.tools.registry import ToolContext, ToolRegistry
from olmoearth_agent.tools.scores_file import build_scores_file_tools

estimate = pytest.importorskip("oe_inferencex.estimate")

PLAN = "olmoearth_plan_label_sample"
GRADE = "olmoearth_estimate_map_error"
CERTIFY = "olmoearth_certify_zone"
WRITE = "olmoearth_write_certificate"
#: The plan arguments of each one-map design over a 400-window map.
DESIGNS: dict[str, dict[str, Any]] = {
    "random": {"budget": 60, "design": "random"},
    "confidence": {"budget": 60},
    "class": {"budget": 120, "design": "class", "min_per_class": 20},
}
#: A class map's population, as a design file keeps it: windows 0 to 5.
CLASS_POPULATION = {
    "classes": ["crop", "forest", "water"],
    "map_class": [0, 1, 2, 0, 1, 2],
}
X0, Y0 = 256_330.0, 9_718_100.0  # UTM 37S, as test_certificate_tool.py's run
BUDGET = 60
ALPHA = 0.2


def _ctx() -> ToolContext:
    return ToolContext(studio=None, state=ThreadState())  # type: ignore[arg-type]


def _registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register_all(build_scores_file_tools())
    registry.register_all(build_estimation_tools())
    registry.register_all(build_certificate_tools())
    return registry


async def _call(name: str, args: dict[str, Any]) -> dict[str, Any]:
    return await _registry().dispatch(
        ToolCall(id="1", name=name, arguments=args), _ctx()
    )


@pytest.fixture(autouse=True)
def _scores_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("OLMOEARTH_SCORES_ROOT", str(tmp_path))
    monkeypatch.setenv("OLMOEARTH_OUTPUT_ROOT", str(tmp_path))
    return tmp_path


def _map(n: int = 400, seed: int = 7) -> tuple[list[list[float]], list[int]]:
    """Three-class logits with errors concentrated at low margin, and the truth."""
    rng = random.Random(seed)
    scores, wrong = [], []
    for _ in range(n):
        row = [rng.gauss(0.0, 1.5) for _ in range(3)]
        top = sorted(row, reverse=True)
        scores.append(row)
        wrong.append(1 if rng.random() < 0.6 * 2.718 ** -(top[0] - top[1]) else 0)
    return scores, wrong


async def _plan(args: dict[str, Any]) -> dict[str, Any]:
    out = await _call(PLAN, args)
    assert out["ok"] is True, out
    return dict(out["result"])


def _design(plan: dict[str, Any]) -> dict[str, Any]:
    return dict(json.loads(Path(plan["design_path"]).read_text(encoding="utf-8")))


def _write(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _sheet(
    plan: dict[str, Any],
    wrong: dict[int, str],
    path: Path,
    reference: dict[int, str] | None = None,
) -> str:
    """The plan's sheet with ``wrong`` (and ``reference_class``) filled per
    window, written to ``path``; its path."""
    with Path(plan["labels_csv_path"]).open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    for row in rows:
        i = int(row["window_index"])
        row["wrong"] = wrong[i]
        if reference is not None:
            row["reference_class"] = reference[i]
    _write(path, rows)
    return str(path)


async def _grade(plan: dict[str, Any], labels_path: str) -> dict[str, Any]:
    out = await _call(
        GRADE, {"design_path": plan["design_path"], "labels_path": labels_path}
    )
    assert out["ok"] is True, out
    return dict(out["result"])


def _ends(interval: dict[str, Any]) -> list[float]:
    """An interval's point as a range: ``[estimate, estimate]`` when it is one value."""
    if interval.get("estimate") is None:
        return list(interval["estimate_range"])
    return [interval["estimate"], interval["estimate"]]


# --------------------------------------------------------------------------- the cells


def test_wrong_cells_reads_zero_one_and_cannot_tell() -> None:
    cells: list[Any] = ["0", "1", "0.0", "1.0", 0, 1, 0.0, 1.0, "?", " ? ", " 1 "]
    indices = list(range(10, 10 + len(cells)))
    values, unjudged = _wrong_cells(cells, indices)
    assert values == [0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1]
    assert all(type(v) is int for v in values)
    assert unjudged == [False] * 8 + [True, True, False]


@pytest.mark.parametrize(
    "cell", ["0.5", "2", "", "yes", "??", True, False, None, 0.5, 2, -1]
)
def test_wrong_cells_refuses_anything_else_naming_the_window(cell: Any) -> None:
    with pytest.raises(ValueError, match=r"^window 42: wrong is ") as info:
        _wrong_cells(["1", cell, "0"], [7, 42, 9])
    message = str(info.value)
    assert f"wrong is {cell!r};" in message
    assert "or ? where it cannot be told" in message


def test_wrong_cells_refuses_a_length_mismatch() -> None:
    with pytest.raises(ValueError, match="2 labels for the 3 windows of the design"):
        _wrong_cells(["0", "?"], [1, 2, 3])


# --------------------------------------------------------------------------- one map's labels


def test_labels_refuses_cannot_tell_by_name() -> None:
    indices = [4, 9, 2]
    refusal = r"^window 9 is \? \(cannot tell\): this tool needs 1 or 0 on every"
    with pytest.raises(ValueError, match=refusal):
        _labels({"wrong": [0, "?", 1]}, indices)
    judged = _labels({"wrong": [0, 1, 1], "reference": [0, "2", 1.0]}, indices)
    assert judged == ([0, 1, 1], [0, 2, 1])


def test_labels_unjudged_flags_cannot_tell_and_drops_the_classes() -> None:
    indices = [4, 9, 2]
    unsure = _labels_unjudged({"wrong": [0, "?", 1], "reference": [0, 1, 2]}, indices)
    assert unsure == ([0, 0, 1], [False, True, False], None)
    judged = _labels_unjudged(
        {"wrong": ["0", "1", "1"], "reference": ["0", "1.0", 2]}, indices
    )
    assert judged == ([0, 1, 1], [False, False, False], [0, 1, 2])


def test_a_sheets_cannot_tell_is_read_and_refused_by_name(tmp_path: Path) -> None:
    sheet = tmp_path / "sheet.csv"
    _write(
        sheet,
        [
            {"window_index": "4", "wrong": "0", "reference_class": "0"},
            {"window_index": "9", "wrong": " ? ", "reference_class": "1"},
            {"window_index": "2", "wrong": "1", "reference_class": "2"},
        ],
    )
    args = {"labels_path": str(sheet)}
    assert _labels_unjudged(args, [4, 9, 2]) == ([0, 0, 1], [False, True, False], None)
    with pytest.raises(ValueError, match=r"window 9 is \? \(cannot tell\)"):
        _labels(args, [4, 9, 2])


# --------------------------------------------------------------------------- a class map's labels


def test_a_class_maps_cannot_tell_in_wrong_alone_is_flagged() -> None:
    indices = [3, 1, 5]
    out = _class_map_labels_unjudged(
        {"wrong": ["0", "?", "1"]}, indices, CLASS_POPULATION
    )
    classes = CLASS_POPULATION["classes"]
    assert out == ([0, 0, 1], [False, True, False], None, classes, [])
    with pytest.raises(ValueError, match=r"^window 1 is \? \(cannot tell\): this tool"):
        _class_map_labels({"wrong": ["0", "?", "1"]}, indices, CLASS_POPULATION)


#: r4h S4: the refusal when ? sits in only one of wrong and the class seen.
APART = (
    "window 1 is ? in one of wrong and the class seen but not in the other: write "
    "? in both, or a label in both"
)


def test_a_class_maps_cannot_tell_in_one_column_only_is_refused() -> None:
    """With both columns given, a ? must sit on the same windows in both."""
    indices = [3, 1, 5]
    seen = ["crop", "forest", "crop"]
    judged = _class_map_labels_unjudged(
        {"wrong": ["0", "0", "1"], "reference": seen}, indices, CLASS_POPULATION
    )
    assert judged[:3] == ([0, 0, 1], [False, False, False], [0, 1, 0])
    for args in (
        {"wrong": ["0", "?", "1"], "reference": seen},
        {"wrong": ["0", "0", "1"], "reference": ["crop", "?", "crop"]},
        {"wrong": ["0", "1", "1"], "reference": ["crop", "?", "crop"]},
    ):
        with pytest.raises(ValueError) as info:
            _class_map_labels_unjudged(args, indices, CLASS_POPULATION)
        assert str(info.value) == APART


def test_a_class_maps_cannot_tell_in_the_class_seen_is_flagged() -> None:
    """r4h S4: a ? in the references is read as a ? in wrong is: 0 with its
    flag set (the package bounds it both ways); it enters neither the class
    table nor the outsiders, and with any ? no reference codes are returned
    (no per-class table). wrong given beside it holds ? on the same window."""
    indices = [3, 1, 5]  # the map writes crop, forest, water there
    classes = CLASS_POPULATION["classes"]
    flagged = ([1, 0, 0], [False, True, False], None, classes, [])
    alone = {"reference": ["forest", "?", "water"]}
    assert _class_map_labels_unjudged(alone, indices, CLASS_POPULATION) == flagged
    both = {**alone, "wrong": ["1", "?", "0"]}
    assert _class_map_labels_unjudged(both, indices, CLASS_POPULATION) == flagged
    spaced = {"reference": [" forest ", " ? ", "WATER"]}
    assert _class_map_labels_unjudged(spaced, indices, CLASS_POPULATION) == flagged


def test_every_window_cannot_tell_is_accepted() -> None:
    indices = [3, 1, 5]
    classes = CLASS_POPULATION["classes"]
    out = _class_map_labels_unjudged(
        {"reference": ["?", "?", "?"]}, indices, CLASS_POPULATION
    )
    assert out == ([0, 0, 0], [True, True, True], None, classes, [])


def test_a_blank_class_seen_is_still_refused() -> None:
    """r4h S4: a blank (or missing) reference is no answer; the refusal names
    ? as the way to say a window cannot be told."""
    indices = [3, 1, 5]
    for reference in (["crop", "", "water"], ["crop", None, "water"]):
        with pytest.raises(ValueError) as info:
            _class_map_labels_unjudged(
                {"reference": reference}, indices, CLASS_POPULATION
            )
        assert str(info.value) == (
            "1 of 3 design windows have no class (first: window 1); write the class "
            "seen, or ? where it cannot be told, in every drawn window"
        )


def test_the_outside_refusal_counts_judged_references_only() -> None:
    """References all outside the map's classes are refused, ? aside; one
    judged reference inside them is enough, and ? is never an outsider."""
    indices = [3, 1, 5]
    with pytest.raises(ValueError) as info:
        _class_map_labels_unjudged(
            {"reference": ["wetland", "?", "?"]}, indices, CLASS_POPULATION
        )
    message = str(info.value)
    assert message.startswith("none of the 1 references (wetland) is a class")
    assert "?" not in message.split(":")[0]
    out = _class_map_labels_unjudged(
        {"reference": ["wetland", "?", "water"]}, indices, CLASS_POPULATION
    )
    table = [*CLASS_POPULATION["classes"], "wetland"]
    assert out == ([1, 0, 0], [False, True, False], None, table, ["wetland"])


def test_the_certified_zone_route_refuses_a_cannot_tell_in_the_class_seen() -> None:
    """_class_map_labels (the zone's and the certificate's route) needs 1 or 0
    on every window: a ? in the references is refused by name, as in wrong."""
    indices = [3, 1, 5]
    refusal = r"^window 1 is \? \(cannot tell\): this tool needs 1 or 0 on every"
    with pytest.raises(ValueError, match=refusal):
        _class_map_labels(
            {"reference": ["forest", "?", "water"]}, indices, CLASS_POPULATION
        )
    judged = _class_map_labels(
        {"reference": ["forest", "forest", "water"]}, indices, CLASS_POPULATION
    )
    assert judged == ([1, 0, 0], [1, 1, 2], CLASS_POPULATION["classes"], [])


def test_a_class_maps_sheet_takes_cannot_tell_in_either_column(
    tmp_path: Path,
) -> None:
    """The sheet: ? in wrong with reference_class blank on every window; ? in
    reference_class with wrong blank on every window (the class face's Cannot
    tell); ? in both on the same window. A ? in wrong beside a class seen is
    refused."""
    indices = [3, 1, 5]
    sheet = tmp_path / "class_sheet.csv"
    args = {"labels_path": str(sheet)}
    rows = [
        {"window_index": str(i), "wrong": w, "reference_class": ""}
        for i, w in zip(indices, ["0", "?", "1"])
    ]
    _write(sheet, rows)
    out = _class_map_labels_unjudged(args, indices, CLASS_POPULATION)
    assert out[:3] == ([0, 0, 1], [False, True, False], None)
    flagged = ([1, 0, 0], [False, True, False], None)
    seen = ["forest", "?", "water"]
    _write(sheet, [dict(r, wrong="", reference_class=s) for r, s in zip(rows, seen)])
    assert _class_map_labels_unjudged(args, indices, CLASS_POPULATION)[:3] == flagged
    wrong = ["1", "?", "0"]
    _write(
        sheet,
        [dict(r, wrong=w, reference_class=s) for r, w, s in zip(rows, wrong, seen)],
    )
    assert _class_map_labels_unjudged(args, indices, CLASS_POPULATION)[:3] == flagged
    seen = ["crop", "forest", "crop"]
    _write(sheet, [dict(r, reference_class=s) for r, s in zip(rows, seen)])
    with pytest.raises(ValueError) as info:
        _class_map_labels_unjudged(args, indices, CLASS_POPULATION)
    assert str(info.value) == APART


# --------------------------------------------------------------------------- the estimate


@pytest.mark.asyncio
@pytest.mark.parametrize("design", ["random", "confidence", "class"])
async def test_a_window_not_judged_is_bounded_both_ways(
    design: str, tmp_path: Path
) -> None:
    """One window '?' on the plan's own sheet: the valid interval's lower end
    is the sheet's with it read as 0, its upper end the sheet's with it read
    as 1; the count is stated after the valid interval; no per-class table and
    no figure, while the same sheet judged throughout draws one."""
    scores, truth = _map(400)
    plan = await _plan({"scores": scores, "grid": 20, **DESIGNS[design]})
    assert plan["design"] == design
    indices = _design(plan)["sample"]["indices"]
    first = indices[0]
    labels = {i: str(truth[i]) for i in indices}
    unsure = _sheet(plan, {**labels, first: "?"}, Path(plan["labels_csv_path"]))
    as_right = _sheet(plan, {**labels, first: "0"}, tmp_path / "as_right.csv")
    as_wrong = _sheet(plan, {**labels, first: "1"}, tmp_path / "as_wrong.csv")
    result = await _grade(plan, unsure)
    right = await _grade(plan, as_right)
    wrong = await _grade(plan, as_wrong)

    assert result["n_unjudged"] == 1
    assert right.get("n_unjudged", 0) == 0 and wrong.get("n_unjudged", 0) == 0
    valid = result["valid_interval"]
    assert valid["low"] == right["valid_interval"]["low"]
    assert valid["high"] == wrong["valid_interval"]["high"]
    assert valid["low"] < valid["high"]
    point = [_ends(right["valid_interval"])[0], _ends(wrong["valid_interval"])[1]]
    assert _ends(valid) == point
    first_entry, _second = result["intervals"]
    assert (first_entry["low"], first_entry["high"]) == (valid["low"], valid["high"])
    if design == "confidence":
        # The union bound over the strata, in place of the design's Wilson.
        assert "could not be judged" in result["method"]
    else:
        assert (result["low"], result["high"]) == (right["low"], wrong["high"])
    assert "bounds_note" in result

    stated = rules.valid_interval_must_state(
        valid.get("estimate"),
        valid["low"],
        valid["high"],
        design,
        len(indices),
        class_map=False,
    )
    assert result["must_state"][0] == stated
    assert result["must_state"][1].startswith("1 of ")
    assert result["must_state"][1] == unjudged_must_state(1, len(indices))
    (whole,) = [f for f in result["facts"] if f["id"] == "whole_map_estimate"]
    assert (whole["low"], whole["high"]) == (valid["low"], valid["high"])
    assert "1 of the labelled windows could not be judged (?)" in whole["sentence"]

    assert "per_class" not in result
    assert "figure_path" not in result and "figure_caption" not in result
    assert Path(right["figure_path"]).is_file()

    report = result["report"]
    assert UNJUDGED_HOW in report
    if design == "confidence":
        assert report.startswith(UNJUDGED_STRATIFIED_HOW)
        assert "union bound over the strata" in report
    else:
        assert report.startswith(VALID_FIRST_HOW[design])
    assert UNJUDGED_HOW not in right["report"]


@pytest.mark.asyncio
async def test_a_sheet_with_no_unjudged_window_grades_as_before(
    tmp_path: Path,
) -> None:
    """Judged throughout, with reference classes: the package's numbers field
    for field, the per-class table, the figure (the design keeps a margin),
    and no word of windows not judged. The same sheet with one '?' drops the
    reference classes and the figure."""
    scores, truth = _map(400)
    plan = await _plan({"scores": scores, "grid": 20, **DESIGNS["random"]})
    stored = _design(plan)
    sample = stored["sample"]
    indices = sample["indices"]
    mc = stored["population"]["map_class"]
    wrong = [truth[i] for i in indices]
    reference = [mc[i] if w == 0 else (mc[i] + 1) % 3 for i, w in zip(indices, wrong)]
    labels = {i: str(w) for i, w in zip(indices, wrong)}
    classes = {i: str(r) for i, r in zip(indices, reference)}
    path = _sheet(plan, labels, Path(plan["labels_csv_path"]), classes)
    result = await _grade(plan, path)

    direct = _jsonable(estimate.estimate_error_rate(sample, wrong, valid=True))
    for key, value in direct.items():
        assert result[key] == value, key
    assert result.get("n_unjudged", 0) == 0
    assert "per_class" in result
    assert Path(result["figure_path"]).is_file()
    assert not any("could not be judged" in s for s in result["must_state"])
    assert result["report"].startswith(VALID_FIRST_HOW["random"])
    assert UNJUDGED_HOW not in result["report"]
    by_list = await _call(
        GRADE,
        {"design_path": plan["design_path"], "wrong": wrong, "reference": reference},
    )
    assert by_list["ok"] is True, by_list
    assert by_list["result"]["valid_interval"] == result["valid_interval"]
    assert by_list["result"]["per_class"] == result["per_class"]

    unsure = _sheet(plan, {**labels, indices[0]: "?"}, tmp_path / "unsure.csv", classes)
    bounded = await _grade(plan, unsure)
    assert bounded["n_unjudged"] == 1
    assert "per_class" not in bounded and "figure_path" not in bounded


# --------------------------------------------------------------------------- zone, certificate


@pytest.mark.asyncio
async def test_certify_zone_refuses_a_window_not_judged() -> None:
    scores, truth = _map(400)
    plan = await _plan({"scores": scores, **DESIGNS["random"]})
    indices = _design(plan)["sample"]["indices"]
    labels = {i: str(truth[i]) for i in indices}
    path = _sheet(plan, {**labels, indices[3]: "?"}, Path(plan["labels_csv_path"]))
    args = {"design_path": plan["design_path"], "labels_path": path}
    zone = await _call(CERTIFY, {**args, "alpha": 0.3})
    assert zone["ok"] is False, zone
    assert f"window {indices[3]} is ? (cannot tell)" in zone["error"]
    assert "Only olmoearth_estimate_map_error with the design_path" in zone["error"]
    graded = await _call(GRADE, args)
    assert graded["ok"] is True, graded
    assert graded["result"]["n_unjudged"] == 1


async def _direct_run(root: Path) -> dict[str, Any]:
    """A direct run of two 32 x 32 windows, read by from-olmoearth and the tool,
    as ``test_certificate_tool.py`` writes it (128 windows of 4 px)."""
    np = pytest.importorskip("numpy")
    rasterio = pytest.importorskip("rasterio")
    transform = pytest.importorskip("rasterio.transform")
    oe_reader = pytest.importorskip("oe_inferencex.readers.olmoearth")
    for k in range(2):
        layer = root / "ds" / "windows" / "default" / f"w{k}" / "layers" / "output"
        (layer / "probs").mkdir(parents=True)
        z = np.random.default_rng(30 + k).normal(0.0, 1.5, (3, 32, 32))
        e = np.exp(z - z.max(0, keepdims=True))
        profile = {
            "driver": "GTiff",
            "height": 32,
            "width": 32,
            "count": 3,
            "dtype": "float32",
            "crs": "EPSG:32737",
            "transform": transform.Affine(10.0, 0.0, X0 + 320.0 * k, 0.0, -10.0, Y0),
        }
        with rasterio.open(layer / "probs" / "geotiff.tif", "w", **profile) as dst:
            dst.write((e / e.sum(0, keepdims=True)).astype(np.float32))
        (layer / "completed").write_text("", encoding="utf-8")
    oe_reader.read_output(str(root / "ds"), str(root / "run"))
    out = await _call("olmoearth_scores_from_file", {"run_dir": "run"})
    assert out["ok"] is True, out
    return dict(out["result"])


@pytest.mark.asyncio
async def test_a_certificate_plan_refuses_a_window_not_judged(tmp_path: Path) -> None:
    """On a seed_from_map plan a '?' is refused by the estimate, the zone and
    the certificate, before any label is fixed in the package's sheet: the
    plan then grades labels that judge that window."""
    run = await _direct_run(tmp_path)
    plan = await _plan(
        {
            "scores_path": run["scores_path"],
            "budget": BUDGET,
            "design": "random",
            "seed_from_map": True,
            "alpha": ALPHA,
        }
    )
    package_sheet = Path(plan["seed_from_map"]["package_sheet"])
    held = package_sheet.read_bytes()
    indices = _design(plan)["sample"]["indices"]
    own = Path(plan["labels_csv_path"])
    unsure = _sheet(plan, {i: "?" if i == indices[0] else "0" for i in indices}, own)
    args = {"design_path": plan["design_path"], "labels_path": unsure}
    for tool, extra, needs in (
        (GRADE, {}, "a certificate plan (seed_from_map) needs 1 or 0"),
        (CERTIFY, {"alpha": ALPHA}, "this tool needs 1 or 0"),
        (WRITE, {"date": "2024-06-01"}, "this tool needs 1 or 0"),
    ):
        out = await _call(tool, {**args, **extra})
        assert out["ok"] is False, (tool, out)
        assert f"window {indices[0]} is ? (cannot tell): {needs}" in out["error"], tool
    assert package_sheet.read_bytes() == held
    # Read as 0 and fixed, the '?' would refuse this 1 as other labels.
    judged = _sheet(plan, {i: "1" if i == indices[0] else "0" for i in indices}, own)
    graded = await _call(
        GRADE, {"design_path": plan["design_path"], "labels_path": judged}
    )
    assert graded["ok"] is True, graded
    assert graded["result"].get("n_unjudged", 0) == 0
