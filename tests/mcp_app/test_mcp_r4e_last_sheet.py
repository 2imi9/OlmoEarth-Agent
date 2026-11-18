# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""The label sheet in the last fix round before the overlay (r4h, 13 Nov 2026).

From reader test 2 and the pass-3 review, builder B2's part:

- S4 (REAL): a class map's sheet takes Cannot tell (``?`` in
  ``reference_class``): the class face offers it, a save takes it, the
  counts and the progress line count it, the estimate grades it as a range
  (with ``tools.estimation``'s cannot-tell handling, B4's part), and a sheet
  that is not a class map's still refuses it in ``reference_class``.
- Q1: a save that empties a graded sheet keeps its mark, marked changed and
  blanked, so answers given again after the result was seen grade as
  changed; a fresh blank sheet beside an old mark still loses it.
- Q2: whether the answers changed is told by a digest of the label cells,
  not the file's bytes.
- ``class_counts`` and ``result_holds_answers`` (contract B2 to B1), and the
  page copy's ``period_text``.
- The sheet's words of the string table (section 1) as ``views/sheet.js``
  builds them, read back from the file (no test here runs the script): a JS
  concatenation of single-quoted strings and names is read into its text
  (:func:`_concat`), so a missing space between two joined pieces fails here
  as it would on the face.
"""

from __future__ import annotations

import csv
import io
import json
import math
import re
from pathlib import Path
from typing import Any

import pytest

from olmoearth_agent.mcp_app import imagery, sheet, views
from olmoearth_agent.tools.review_set import slug

ONE_MAP = [
    "order",
    "window_index",
    "row",
    "col",
    "map_class",
    "lon",
    "lat",
    "stratum",
    "wrong",
    "reference_class",
]

#: A class map's classes as it writes them, and its class of 10 windows:
#: window 7 water, window 3 trees, window 5 crops.
CLASSES = ["water", "trees", "crops"]
CLASS_MAP = [0, 0, 0, 1, 0, 2, 0, 0, 0, 0]

#: The map's class of each of 20 windows on the seen fixture's design: present
#: (1) at windows 4 and 12, absent (0) elsewhere.
SEEN_MAP = [1 if w in (4, 12) else 0 for w in range(20)]

#: A period as Studio gives it, and as the class face prints it.
WINDOW = {"start": "2024-04-01", "end": "2024-06-30T23:59:59.999000Z"}
PERIOD = "1 Apr 2024 to 30 Jun 2024"


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    out = tmp_path / "outputs"
    out.mkdir()
    monkeypatch.setenv("OLMOEARTH_OUTPUT_ROOT", str(out))
    monkeypatch.delenv("OLMOEARTH_SCORES_ROOT", raising=False)
    return out


def _write(path: Path, header: list[str], rows: list[list[str]]) -> Path:
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)
    return path


def _rows(path: Path) -> list[list[str]]:
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.reader(fh))


def _sheet_rows(windows: tuple[int, ...], on_map: list[Any]) -> list[list[str]]:
    return [
        [str(order), str(w), "0", str(w), str(on_map[w]), "10.5", "50.2", "0", "", ""]
        for order, w in enumerate(windows, start=1)
    ]


def _class_map(
    root: Path,
    *,
    name: str = "land",
    source: dict[str, Any] | None = None,
    certificate: bool = False,
) -> Path:
    """A class map's plan, labelled with the class seen: windows 7 (water), 3
    (trees) and 5 (crops); its design names its source when given."""
    population: dict[str, Any] = {
        "score_kind": "class_map",
        "classes": CLASSES,
        "map_class": CLASS_MAP,
    }
    if source is not None:
        population["source"] = source
    design: dict[str, Any] = {"design": "random", "population": population}
    if certificate:
        design["certificate"] = {"seed_rule": "from_map", "alpha": 0.05}
    (root / f"{name}.json").write_text(json.dumps(design), encoding="utf-8")
    on_map = [CLASSES[c] for c in CLASS_MAP]
    rows = _sheet_rows((7, 3, 5), on_map)
    return _write(root / f"{name}_to_label.csv", ONE_MAP, rows)


def _seen(root: Path, windows: tuple[int, ...] = (4, 9, 12, 3)) -> Path:
    """A named scores map's plan (the kelp trial's): Present and Absent."""
    design = {
        "design": "random",
        "population": {
            "score_kind": "window_confidence",
            "class_names": {"0": "absent", "1": "present"},
            "map_class": SEEN_MAP,
        },
    }
    (root / "kelp.json").write_text(json.dumps(design), encoding="utf-8")
    return _write(root / "kelp_to_label.csv", ONE_MAP, _sheet_rows(windows, SEEN_MAP))


def _scores_map(root: Path) -> Path:
    """A scores map's plan with no class names: labelled right or wrong."""
    design = {"design": "random", "population": {"score_kind": "window_confidence"}}
    (root / "conf.json").write_text(json.dumps(design), encoding="utf-8")
    return _write(root / "conf_to_label.csv", ONE_MAP, _sheet_rows((1, 2), [0] * 20))


def _by_window(path: Path) -> dict[int, str]:
    """Each window's handle by its index: the server's own map, never sent."""
    _, handles = sheet.window_handles(str(path))
    return {index: handle for handle, index in handles.items()}


def _answer(path: Path, face: str, answers: dict[int, str], **extra: Any) -> Any:
    """A panel's save of these answers by window, on ``face``, at the version
    the panel read."""
    read = sheet.read_sheet(str(path), face)
    at = _by_window(path)
    return sheet.save_answers(
        str(path),
        sheet_id=read["sheet_id"],
        face=face,
        rule=read["label_rule"],
        answers=[{"handle": at[w], "answer": a} for w, a in answers.items()],
        version=read["version"],
        **extra,
    )


def _set_cell(path: Path, window: str, column: str, value: str) -> None:
    """Edit one cell of the sheet, as a spreadsheet would."""
    rows = _rows(path)
    at, at_window = rows[0].index(column), rows[0].index("window_index")
    for row in rows[1:]:
        if row[at_window] == window:
            row[at] = value
    _write(path, rows[0], rows[1:])


def _graded_place(path: Path) -> Path:
    return path.with_name("." + path.name + sheet.GRADED_SUFFIX)


def _mark(path: Path) -> dict[str, Any]:
    return json.loads(_graded_place(path).read_text(encoding="utf-8"))


def _class_row(
    name: str, seen: int, agreed: int, *, outside: bool = False
) -> dict[str, Any]:
    """A class map's class as the counts list it: its code is its name."""
    return {
        "code": name,
        "name": name,
        "seen": seen,
        "agreed": agreed,
        "outside": outside,
    }


# ------------------------------------------------- S4: Cannot tell on a class map


def test_a_class_maps_face_offers_cannot_tell_after_its_classes(root: Path) -> None:
    """The class face's answers end with Cannot tell (key ? alone, no digit);
    a certificate plan's class sheet offers none."""
    read = sheet.read_sheet(str(_class_map(root)))
    assert read["face"] == "class" and read["takes_unjudged"] is True
    assert read["answers"] == [
        {"value": "water", "label": "water", "keys": ["1"]},
        {"value": "trees", "label": "trees", "keys": ["2"]},
        {"value": "crops", "label": "crops", "keys": ["3"]},
        {"value": "?", "label": sheet.CANNOT_TELL, "keys": ["?"]},
    ]
    cert = sheet.read_sheet(str(_class_map(root, name="cert", certificate=True)))
    assert [a["value"] for a in cert["answers"]] == CLASSES


def test_a_class_maps_face_with_many_classes_keeps_cannot_tell_last(
    root: Path,
) -> None:
    """Past nine classes, the digits stop at nine and Cannot tell is still
    listed, with its own key (the panel draws it as a button after the first
    nine, never only in the list of more)."""
    many = [f"class {i}" for i in range(12)]
    design = {
        "design": "random",
        "population": {"score_kind": "class_map", "classes": many},
    }
    (root / "many.json").write_text(json.dumps(design), encoding="utf-8")
    path = _write(
        root / "many_to_label.csv",
        ONE_MAP,
        [["1", "4", "0", "4", "class 0", "10.5", "50.2", "0", "", ""]],
    )
    answers = sheet.read_sheet(str(path))["answers"]
    assert [a["value"] for a in answers] == [*many, "?"]
    assert [a["keys"] for a in answers[:9]] == [[str(i)] for i in range(1, 10)]
    assert answers[9]["keys"] == [] and answers[-1]["keys"] == ["?"]


def test_a_class_sheets_cannot_tell_is_saved_and_counted(root: Path) -> None:
    """A panel's ? on the class face goes into reference_class; it counts as
    an answer (the counts, the plan card's line, the page copy's answer), so
    the sheet completes with it; its squares are graded with it as a Cannot
    tell cell after the judged ones (the owner's O2 of 14 Nov 2026), and
    class_counts counts it apart."""
    path = _class_map(root)
    out = _answer(path, "class", {7: "?"})
    assert (out["cells_changed"], out["n_labelled"]) == (1, 1)
    cells = {r[1]: (r[-2], r[-1]) for r in _rows(path)[1:]}
    assert cells["7"] == ("", "?")
    progress = sheet.sheet_progress(str(path))
    assert (progress["n_labelled"], progress["n_unjudged"]) == (1, 1)
    assert progress["progress"] == "1 of 3 labelled, 1 cannot tell"
    _answer(path, "class", {3: "trees", 5: "shrubs"})
    read = sheet.read_sheet(str(path))
    assert (read["n_labelled"], read["n_rows"], read["n_unjudged"]) == (3, 3, 1)
    assert sorted(w["answer"] for w in read["windows"]) == ["?", "shrubs", "trees"]
    assert "mix_problem" not in read
    version = sheet.sheet_version(str(path))
    classes = [
        _class_row("water", 0, 0),
        _class_row("trees", 1, 1),
        _class_row("crops", 0, 0),
        _class_row("shrubs", 1, 0, outside=True),
    ]
    assert sheet.graded_squares(str(path), version) == {
        "n": 3,
        "n_unjudged": 1,
        "face": "class",
        "cells": [
            {"seen": "trees", "wrong": False},
            {"seen": "shrubs", "wrong": True},
            {"seen": None, "wrong": False, "unjudged": True},
        ],
        "classes": classes,
    }
    counts = sheet.class_counts(str(path), version)
    assert counts == {"n": 3, "n_unjudged": 1, "classes": classes}


def test_cannot_tell_in_reference_class_is_refused_off_a_class_map(
    root: Path,
) -> None:
    """A scores map's one-map sheet (right or wrong) and a certificate plan's
    class sheet refuse ? in reference_class (the file route), with the file
    as it was; the panel's save on a certificate class sheet refuses it as no
    answer."""
    for path in (_scores_map(root), _class_map(root, name="cert", certificate=True)):
        before = path.read_bytes()
        window = int(_rows(path)[1][1])
        with pytest.raises(ValueError) as refused:
            sheet.save_labels(
                str(path), [{"window_index": window, "reference_class": "?"}]
            )
        assert str(refused.value) == sheet.UNJUDGED_NOT_HERE
        assert path.read_bytes() == before
    cert = root / "cert_to_label.csv"
    with pytest.raises(ValueError, match=r"\? is not one of this sheet's answers"):
        _answer(cert, "class", {7: "?"})
    assert sheet.UNJUDGED_NOT_HERE == (
        "'?' (cannot tell) is not taken in reference_class on this sheet: only a "
        "two-map sheet and a class map's sheet that takes Cannot tell take it there"
    )


def test_cannot_tell_in_both_columns_on_one_window_is_no_mix(root: Path) -> None:
    """Both label columns full (a sheet edited by hand): ? agrees only with ?
    on the same window, as the estimate takes it (B4's contract)."""
    path = _class_map(root)
    edits = (("7", "?", "?"), ("3", "trees", "0"), ("5", "crops", "0"))
    for window, ref, wrong in edits:
        _set_cell(path, window, "reference_class", ref)
        _set_cell(path, window, "wrong", wrong)
    assert "mix_problem" not in sheet.read_sheet(str(path))
    counts = sheet.class_counts(str(path), sheet.sheet_version(str(path)))
    assert counts is not None and (counts["n"], counts["n_unjudged"]) == (3, 1)
    _set_cell(path, "7", "reference_class", "water")
    assert sheet.read_sheet(str(path))["mix_problem"] == (
        "its wrong column disagrees with the classes seen"
    )
    # ? in one column only: no counts.
    assert sheet.class_counts(str(path), sheet.sheet_version(str(path))) is None


# ----------------------------------------------- S4: graded through the estimate

#: The fake Studio's grid (8 x 8 = 64 windows) and its class map's codes.
GRID = 8
_CODES = {"water": 1, "forest": 2, "crop": 3}
_BOX = {"type": "Polygon", "coordinates": [[[0, 0], [4, 0], [4, 4], [0, 4], [0, 0]]]}


def _land(row: int, col: int) -> str:
    """The map: water west of column 3, crop in column 7, forest between."""
    if col == 7:
        return "crop"
    return "water" if col < 3 else "forest"


class _Studio:
    """One classification result over the box, as the Studio client returns it
    (tests/tools/test_class_map_tool.py's, shortened)."""

    def __init__(self) -> None:
        from olmoearth_agent.analysis.raster_compare import grid_windows

        self.cells = grid_windows([0, 0, 4, 4], GRID)

    async def get_prediction_result(self, result_id: str) -> dict[str, Any]:
        fields = [
            {
                "property_name": "landcover",
                "allowed_values": [{"value": v, "label": k} for k, v in _CODES.items()],
            }
        ]
        return {
            "id": result_id,
            "prediction_id": f"p-{result_id}",
            "property_names": ["landcover"],
            "result_metadata": {"geometry": _BOX, "classification_fields": fields},
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
        row, col, _lo, _la = min(
            self.cells, key=lambda c: (c[2] - lon) ** 2 + (c[3] - lat) ** 2
        )
        label = _land(row, col)
        band = {
            "property_name": "landcover",
            "raw_value": _CODES[label],
            "classification": {"label": label, "color": "#000000"},
        }
        return {"bands": [band]}


async def _tool(name: str, args: dict[str, Any], studio: Any = None) -> Any:
    from olmoearth_agent.harness.state import ThreadState
    from olmoearth_agent.llm.types import ToolCall
    from olmoearth_agent.tools.estimation import build_estimation_tools
    from olmoearth_agent.tools.registry import ToolContext, ToolRegistry

    registry = ToolRegistry()
    registry.register_all(build_estimation_tools())
    ctx = ToolContext(studio=studio, state=ThreadState())
    return await registry.dispatch(ToolCall(id="1", name=name, arguments=args), ctx)


@pytest.mark.asyncio
async def test_a_class_sheet_with_cannot_tell_is_graded_as_a_range(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A class map planned by result_id (its sheet of points), every point
    answered on the class face as the map says, one Cannot tell: the estimate
    grades it (B4's cannot-tell handling in tools/estimation.py) with the
    error rate a range, n_unjudged 1 and no per-class table, as a seen
    sheet's Cannot tell is graded; its squares are graded with the Cannot
    tell point last (the owner's O2 of 14 Nov 2026)."""
    pytest.importorskip("oe_inferencex.estimate")
    monkeypatch.setenv("OLMOEARTH_SCORES_ROOT", str(tmp_path))
    monkeypatch.setenv("OLMOEARTH_OUTPUT_ROOT", str(tmp_path))
    args = {"result_id": "lc", "grid": GRID, "budget": 20, "design": "random"}
    plan = await _tool("olmoearth_plan_label_sample", {**args, "seed": 3}, _Studio())
    assert plan["ok"] is True, plan
    path = Path(plan["result"]["labels_csv_path"])
    read = sheet.read_sheet(str(path))
    assert (read["face"], read["unit"]) == ("class", "point")
    assert read["answers"][-1]["value"] == "?"
    design = json.loads(Path(read["design_path"]).read_text(encoding="utf-8"))
    population = design["population"]
    at = _by_window(path)
    answers = {
        w: str(population["classes"][population["map_class"][w]]) for w in sorted(at)
    }
    answers[sorted(at)[0]] = "?"
    _answer(path, "class", answers)
    assert sheet.read_sheet(str(path))["n_unjudged"] == 1
    out = await _tool(
        "olmoearth_estimate_map_error",
        {"design_path": read["design_path"], "labels_path": str(path)},
    )
    assert out["ok"] is True, out
    result = out["result"]
    assert result["n_unjudged"] == 1
    assert "per_class" not in result
    span = result["valid_interval"].get("estimate_range")
    assert isinstance(span, (list, tuple)) and len(span) == 2
    assert all(isinstance(v, (int, float)) and math.isfinite(v) for v in span)
    assert span[0] < span[1]
    version = sheet.sheet_version(str(path))
    squares = sheet.graded_squares(str(path), version)
    assert squares is not None and (squares["n"], squares["n_unjudged"]) == (20, 1)
    assert squares["cells"][-1] == {"seen": None, "wrong": False, "unjudged": True}
    assert all("unjudged" not in cell for cell in squares["cells"][:-1])
    assert sum(row["seen"] for row in squares["classes"]) == 19
    counts = sheet.class_counts(str(path), version)
    assert counts is not None and (counts["n"], counts["n_unjudged"]) == (20, 1)


# ----------------------------------------------------------- Q1: the blanked sheet


def test_a_cleared_class_sheet_answered_again_grades_as_changed(root: Path) -> None:
    """The lead's ruling Q1: a class map's sheet graded on the class face, the
    classes seen cleared by the panel's link (clear_column), the squares
    answered on the right-or-wrong face: the mark is kept beside the blank
    sheet (changed, blanked at the version written), the page copy says the
    sheet was graded, the first answer after the clear keeps the mark, and the
    next grading is changed."""
    path = _class_map(root)
    _answer(path, "class", {7: "water", 3: "trees", 5: "crops"})
    first = sheet.record_grading(str(path), sheet.sheet_version(str(path)))
    assert (first["changed"], first["gradings"]) == (False, 1)
    cleared = _answer(path, "class", {}, clear_column="reference_class")
    assert cleared["n_labelled"] == 0
    mark = _mark(path)
    assert mark["changed"] is True and mark["blanked"] == cleared["version"]
    blank = sheet.read_sheet(str(path))
    assert blank["n_labelled"] == 0 and blank["faces"] == ["class", "wrong"]
    assert blank["graded"] == {
        "first_at": first["first_at"],
        "changed_since_first": True,
    }
    # The cleared sheet holds no answer: not counted as checked (r4i F4).
    assert sheet.graded_state(str(path)) == {
        "first_at": first["first_at"],
        "changed_since_first": True,
        "changed": True,
        "answered": False,
    }
    _answer(path, "wrong", {7: "0", 3: "0", 5: "0"})
    assert _graded_place(path).is_file()
    again = sheet.record_grading(str(path), sheet.sheet_version(str(path)))
    assert (again["changed"], again["gradings"], again["first_at"]) == (
        True,
        2,
        first["first_at"],
    )


def test_answers_cleared_one_by_one_keep_the_mark_too(root: Path) -> None:
    """Q1 by the other way: each answer cleared by its own save (the file
    route), the last save empties the sheet and keeps the mark; answered
    again, it grades as changed."""
    path = _class_map(root)
    sheet.save_labels(
        str(path),
        [
            {"window_index": 7, "reference_class": "water"},
            {"window_index": 3, "reference_class": "trees"},
            {"window_index": 5, "reference_class": "crops"},
        ],
    )
    sheet.record_grading(str(path), sheet.sheet_version(str(path)))
    for window in (7, 3, 5):
        sheet.save_labels(str(path), [{"window_index": window, "reference_class": ""}])
    assert _mark(path)["blanked"] == sheet.sheet_version(str(path))
    sheet.save_labels(str(path), [{"window_index": 7, "reference_class": "water"}])
    assert _graded_place(path).is_file() and _mark(path)["changed"] is True
    sheet.save_labels(
        str(path),
        [
            {"window_index": 3, "reference_class": "trees"},
            {"window_index": 5, "reference_class": "crops"},
        ],
    )
    out = sheet.record_grading(str(path), sheet.sheet_version(str(path)))
    assert (out["changed"], out["gradings"]) == (True, 2)


def test_a_blank_sheet_at_another_version_than_the_blanked_one_loses_the_mark(
    root: Path,
) -> None:
    """A sheet blank at a version the mark does not name (a fresh sheet written
    in its place) is no graded sheet: no mark, and the first save removes it."""
    path = _class_map(root)
    _answer(path, "class", {7: "water", 3: "trees", 5: "crops"})
    sheet.record_grading(str(path), sheet.sheet_version(str(path)))
    _answer(path, "class", {}, clear_column="reference_class")
    mark = _mark(path)
    mark["blanked"] = "v" + "a" * 16
    _graded_place(path).write_text(json.dumps(mark), encoding="utf-8")
    assert sheet.graded_state(str(path)) is None
    assert sheet.read_sheet(str(path))["graded"] is None
    _answer(path, "class", {7: "water"})
    assert not _graded_place(path).exists()


# ---------------------------------------------- Q2: the answers, not the bytes


def _resaved(path: Path) -> None:
    """The sheet saved again by a spreadsheet with no cell changed: a UTF-8
    byte-order mark, LF line ends and every cell quoted."""
    text = io.StringIO(newline="")
    writer = csv.writer(text, quoting=csv.QUOTE_ALL, lineterminator="\n")
    writer.writerows(_rows(path))
    path.write_bytes(b"\xef\xbb\xbf" + text.getvalue().encode("utf-8"))


def test_a_resave_with_the_same_answers_is_no_change(root: Path) -> None:
    """The lead's ruling Q2: the mark keeps a digest of the answers graded; a
    re-save with other bytes and the same cells is no change (graded state,
    page copy, next grading); one answer changed is."""
    path = _seen(root)
    sheet.save_labels(
        str(path),
        [
            {"window_index": w, "reference_class": c}
            for w, c in ((4, "1"), (9, "1"), (12, "0"), (3, "0"))
        ],
    )
    first_version = sheet.sheet_version(str(path))
    first = sheet.record_grading(str(path), first_version)
    assert re.fullmatch(r"[0-9a-f]{64}", _mark(path)["first"]["answers"])
    _resaved(path)
    assert sheet.sheet_version(str(path)) != first_version
    unchanged = {"first_at": first["first_at"], "changed_since_first": False}
    assert sheet.graded_state(str(path)) == {
        **unchanged,
        "changed": False,
        "answered": True,
    }
    assert sheet.read_sheet(str(path))["graded"] == unchanged
    assert sheet.sheet_progress(str(path))["graded"] == unchanged
    again = sheet.record_grading(str(path), sheet.sheet_version(str(path)))
    assert (again["changed"], again["gradings"]) == (False, 2)
    sheet.save_labels(str(path), [{"window_index": 3, "reference_class": "1"}])
    assert sheet.graded_state(str(path))["changed_since_first"] is True
    changed = sheet.record_grading(str(path), sheet.sheet_version(str(path)))
    assert (changed["changed"], changed["gradings"]) == (True, 3)


def test_a_mark_without_the_answers_digest_compares_the_version(root: Path) -> None:
    """A mark written before the digest was kept falls back to the version:
    a re-save with other bytes then reads as changed (the safe side)."""
    path = _seen(root)
    sheet.save_labels(
        str(path),
        [{"window_index": w, "reference_class": "0"} for w in (4, 9, 12, 3)],
    )
    sheet.record_grading(str(path), sheet.sheet_version(str(path)))
    mark = _mark(path)
    del mark["first"]["answers"]
    _graded_place(path).write_text(json.dumps(mark), encoding="utf-8")
    assert sheet.graded_state(str(path))["changed_since_first"] is False
    _resaved(path)
    assert sheet.graded_state(str(path))["changed_since_first"] is True


# ------------------------------------------------------- class_counts (B2 to B1)


def test_class_counts_of_a_seen_sheet_count_cannot_tell_apart(root: Path) -> None:
    """Every class the sheet answers with, seen or not, over the judged
    squares; Cannot tell counted apart; none outside the map's classes."""
    path = _seen(root)
    sheet.save_labels(
        str(path),
        [
            {"window_index": w, "reference_class": c}
            for w, c in ((4, "1"), (9, "1"), (12, "?"), (3, "0"))
        ],
    )
    version = sheet.sheet_version(str(path))
    assert sheet.class_counts(str(path), version) == {
        "n": 4,
        "n_unjudged": 1,
        "classes": [
            {"code": "1", "name": "Present", "seen": 2, "agreed": 1, "outside": False},
            {"code": "0", "name": "Absent", "seen": 1, "agreed": 1, "outside": False},
        ],
    }
    # Not the version graded, an answer missing, a right-or-wrong sheet: none.
    assert sheet.class_counts(str(path), "v" + "a" * 16) is None
    sheet.save_labels(str(path), [{"window_index": 3, "reference_class": ""}])
    assert sheet.class_counts(str(path), sheet.sheet_version(str(path))) is None
    scores = _scores_map(root)
    sheet.save_labels(
        str(scores),
        [{"window_index": 1, "wrong": "0"}, {"window_index": 2, "wrong": "1"}],
    )
    assert sheet.class_counts(str(scores), sheet.sheet_version(str(scores))) is None
    # A path that is no plan's sheet: none, never raised.
    assert sheet.class_counts(str(root / "nothing.csv"), version) is None


# ------------------------------------------------ result_holds_answers (B2 to B1)


def _result_sheet(root: Path, result_id: str, tag: str, labelled: bool) -> Path:
    """A plan's sheet of a Studio result, named as the plan tool names it
    (design_<slug of the result, 12>_<digest>_<design>_<budget>_s<seed>)."""
    stem = f"design_{slug(result_id, 12)}_{tag}_random_20_s0"
    design = {
        "design": "random",
        "population": {
            "score_kind": "window_confidence",
            "source": {"scores_path": "zone.json", "result_id": result_id},
        },
    }
    (root / f"{stem}.json").write_text(json.dumps(design), encoding="utf-8")
    rows = _sheet_rows((1, 2), [0] * 20)
    if labelled:
        rows[0][ONE_MAP.index("wrong")] = "?"
    return _write(root / f"{stem}_to_label.csv", ONE_MAP, rows)


def test_a_result_holds_answers_once_any_of_its_sheets_holds_one(root: Path) -> None:
    """True once a sheet of the result holds a label cell (Cannot tell
    included); never for another result, a blank sheet, a slug shared with
    another result whose design names that one, or an id that is no text."""
    rid = "00000000-0000-4000-8000-00000000aaaa"
    twin = "00000000-0000-4000-8000-00000000bbbb"
    assert slug(rid, 12) == slug(twin, 12)
    _result_sheet(root, rid, "1111aaaa", labelled=False)
    assert sheet.result_holds_answers(rid) is False
    _result_sheet(root, rid, "2222bbbb", labelled=True)
    assert sheet.result_holds_answers(rid) is True
    assert sheet.result_holds_answers(f"  {rid} ") is True
    assert sheet.result_holds_answers(twin) is False
    assert sheet.result_holds_answers("another-result") is False
    for odd in (None, "", "   ", 7):
        assert sheet.result_holds_answers(odd) is False


# --------------------------------------------------------- the page copy's period


def test_the_page_copy_names_the_maps_period(root: Path) -> None:
    """E/sheet-3: period_text from the design's source date_window, else from
    the top level of the scores file it names (inside the read roots), else
    None."""
    named = _class_map(root, name="named", source={"date_window": WINDOW})
    assert sheet.read_sheet(str(named))["period_text"] == PERIOD
    scores = root / "zone_scores.json"
    scores.write_text(json.dumps({"date_window": WINDOW, "scores": []}), "utf-8")
    filed = _class_map(root, name="filed", source={"scores_path": str(scores)})
    assert sheet.read_sheet(str(filed))["period_text"] == PERIOD
    bare = _class_map(root, name="bare", source={"result_id": "r1"})
    assert sheet.read_sheet(str(bare))["period_text"] is None
    odd = {"start": "soon", "end": "2024-06-30"}
    odd_one = _class_map(root, name="odd", source={"date_window": odd})
    assert sheet.read_sheet(str(odd_one))["period_text"] is None
    missing = _class_map(
        root, name="missing", source={"scores_path": str(root / "gone.json")}
    )
    assert sheet.read_sheet(str(missing))["period_text"] is None


def test_a_scores_file_outside_the_roots_is_not_read_for_its_period(
    root: Path, tmp_path: Path
) -> None:
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    scores = outside / "zone_scores.json"
    scores.write_text(json.dumps({"date_window": WINDOW}), encoding="utf-8")
    path = _class_map(root, name="far", source={"scores_path": str(scores)})
    assert sheet.read_sheet(str(path))["period_text"] is None


# ------------------------------------------------------------- the words (Python)


def test_the_rule_scope_and_the_pictures_refusal_are_the_string_tables() -> None:
    """C5 and A/F21 (section 1), each with its WORDING comment."""
    assert imagery.RULE_SCOPE == (
        "This rule came with the app (set 11 Nov 2026 for kelp canopy maps)."
    )
    assert imagery.NO_FOOTPRINT.endswith(" Street map still works.")
    assert "Open map" not in imagery.NO_FOOTPRINT
    source = Path(imagery.__file__).read_text(encoding="utf-8")
    assert "# WORDING: r4h C5\nRULE_SCOPE = " in source
    assert "# WORDING: r4h A/F21\nNO_FOOTPRINT = (" in source
    own = Path(sheet.__file__).read_text(encoding="utf-8")
    assert "# WORDING: r4h S4 (model-facing)\nUNJUDGED_NOT_HERE = (" in own


# ------------------------------------------------------------- the words (sheet.js)

#: One token of a JS concatenation: a single-quoted string, a plus, or a name.
_TOKEN = re.compile(
    r"\s*(?:'((?:[^'\\]|\\.)*)'|(\+)|([A-Za-z_][A-Za-z0-9_.]*(?:\([^()]*\))?))"
)


def _concat(expr: str, names: dict[str, str]) -> str:
    """The text a JS concatenation of single-quoted strings and names makes,
    each name given its value in names; anything else fails."""
    out: list[str] = []
    text = expr.strip()
    at = 0
    while at < len(text):
        match = _TOKEN.match(text, at)
        assert match is not None, text[at:]
        literal, _plus, name = match.groups()
        if literal is not None:
            out.append(re.sub(r"\\(.)", r"\1", literal))
        elif name is not None:
            out.append(names[name])
        at = match.end()
    return "".join(out)


def _script() -> str:
    return (views.VIEW_DIR / "sheet.js").read_text(encoding="utf-8")


def _between(text: str, start: str, end: str) -> str:
    at = text.index(start)
    return text[at : text.index(end, at)]


def _after(text: str, start: str, end: str) -> str:
    at = text.index(start) + len(start)
    return text[at : text.index(end, at)]


def _entry(block: str, start: str) -> str:
    """One entry's value in a JS object literal: from start to the comma that
    ends its last line."""
    at = block.index(start) + len(start)
    end = block.find(",\n", at)
    return (block[at:] if end < 0 else block[at:end]).strip().rstrip(",")


def _words() -> str:
    return _between(_script(), "OE.sheetWords = (unit) => {", "\n};")


def _said(key: str, unit: str, **names: str) -> str:
    """An entry of OE.sheetWords as the face says it, for a unit word."""
    words = _words()
    one = "point" if unit == "point" else "square"
    tail = _after(words, "const cannotTellRest = ", ";\n")
    rest = _concat(tail, {"many": one + "s"})
    value = re.sub(r"^\([a-z]+\) => ", "", _entry(words, f"    {key}: "))
    given = {"one": one, "many": one + "s", "cannotTellRest": rest, **names}
    return _concat(value, given)


def test_the_sheets_face_lines_are_the_string_tables() -> None:
    """S1, S2, S3 and A/F21 (section 1), as the face says them, for squares
    and for points; the Cannot tell line as r4i's F5 says it, cut by r5-e H1
    (what a guess costs is in the fold)."""
    assert _said("cannotTellLine", "square") == (
        "Cannot judge a square? Press Cannot tell; do not guess."
    )
    # r5-b J13 made noResult the lead's third line; r5-e H1 removed it.
    assert "\n    noResult: " not in _words()
    # r5-b J29: the finish line's second sentence has its object; r5-e H5: it
    # is that sentence alone.
    assert _said("everyDone", "square") == "After you press, do not change an answer."
    # r4-m M6, in one clause since r5-e H1: "..., without looking at the map."
    assert _said("whatPlaced", "point") == (
        "This sheet has no pictures. See where each point is, then say what was "
        "there from your own imagery or records, without looking at the map."
    )
    assert _said("streetMap", "square") == "Street map"
    assert _said("openMapOf", "point", k="4") == "Street map: point 4 on OpenStreetMap"
    # A sheet with no pictures and no places (r5-b J11): it cannot show where
    # a point is, so it says what to answer from (r5-f F12: the way out is a
    # question of its own).
    assert _said("whatBare", "point") == (
        "This sheet has no pictures and no places, so it cannot show where a point "
        "is. Answer only from your own records of each point, without looking at the "
        "map. No records? Stop and tell Claude."
    )


def test_the_sheets_fold_lines_are_the_string_tables() -> None:
    """S1 (A/F04, B/sheet-blind-line, C/sheet-3) and A/F21: what goes into the
    one closed fold; what Cannot tell costs as r4i's F5 and O2 say it (the
    grid of squares stays, only the per-class lines go)."""
    assert _said("cannotTell", "square") == (
        "Cannot tell is for a square you cannot judge from the dates shown. Do not "
        "guess: a guess can make the result wrong. A Cannot tell does not: you "
        "still get a result, with a wider range, but without the per-class lines."
    )
    bare = _said("cannotTellBare", "point")
    assert bare == (
        "Cannot tell is for a point you cannot judge. Do not guess: a guess can "
        "make the result wrong. A Cannot tell does not: you still get a result, "
        "with a wider range, but without the per-class lines."
    )
    assert "dates" not in bare
    assert _said("blindSeen", "square") == (
        "Blind: this panel is not sent what the map says for a square, or where "
        "the square is, so you answer without seeing the map's answer."
    )
    assert _said("codeLine", "square") == (
        "While you label, this panel is not sent a square's number, place or order."
    )
    assert _said("openMap", "square") == (
        "Street map fetches one square's place only when you press it; it never "
        "shows the map's answer."
    )


def test_the_class_faces_instruction_names_the_period() -> None:
    """E/sheet-3, both forms: with the map's period (r5-e H1: it starts with
    what to do, no longer as the lead starts), and without it (unchanged)."""
    block = _after(
        _words(),
        "classWhat: (period) => (period",
        "),\n    // WORDING: r5b J26\n    noRule",
    )
    known, unknown = block.split("\n      : ")
    names = {"one": "point", "period": PERIOD}
    assert _concat(known.strip().removeprefix("? "), names) == (
        "Pick what was there during 1 Apr 2024 to 30 Jun 2024, using the map's class "
        "names."
    )
    assert _concat(unknown, names) == (
        "For each point, pick what was there in the period the map covers, using "
        "the map's class names; ask whoever made the map for that period."
    )
    editor = _between(_script(), "OE.sheetEditor = (app, data,", "\n};")
    told = _between(editor, "  function faceInstructionText() {", "\n  }\n")
    # r5-d R08: on the face only with the period; without it, in the fold
    # (tests/mcp_app/test_mcp_r5d_sheet.py).
    assert "if (face === 'class') return period ? W.classWhat(period) : '';" in told
    assert "on the ground there, as the map writes classes" not in _script()
    assert (
        "const period = typeof data.period_text === 'string' && data.period_text"
        in editor
    )


def test_every_new_sheet_word_carries_its_wording_comment() -> None:
    """Each changed or added entry of OE.sheetWords has a WORDING line of its
    round right above it: r4h's, and r4i's for the Cannot tell words it
    changed (F5, O2)."""
    lines = _words().split("\n")
    rounds = {
        "whatPlaced": "r4h",
        "cannotTellLine": "r4i F5",
        "cannotTell": "r4i F5 O2",
        "cannotTellBare": "r4i F5 O2",
        "blindSeen": "r4h",
        "openMap": "r4h",
        "streetMap": "r4h",
        "classWhat": "r4h",
        "openMapOf": "r4h",
        "everyDone": "r4h",
        "codeLine": "r4h",
    }
    for key, tag in rounds.items():
        (at,) = [i for i, line in enumerate(lines) if line.startswith(f"    {key}: ")]
        assert lines[at - 1].strip().startswith(f"// WORDING: {tag}"), key
    assert "// WORDING: r4i F5 O2\n  const cannotTellRest = " in _words()


def test_the_face_holds_three_lines_and_one_fold() -> None:
    """S1: the file name, the keys, the blind and code lines, what Cannot
    tell does and the clear-date note sit in ONE closed fold under the
    answers; the face keeps the lead, the rule with its scope, one Cannot
    tell line and the S2 line; no "sampling group (stratum)" anywhere, no
    "Open map". r5-b J13, J14: the Cannot tell line and the S2 line are the
    lead's second and third lines, and the rule stands right above the
    answers (with no rule, the line that none was set: J26). r5-e H1, H2: the
    lead is two lines (no S2 line), and the rule's scope is in the fold, the
    kelp-only line after it."""
    script = _script()
    editor = _between(script, "OE.sheetEditor = (app, data,", "\n};")
    assert "sampling group" not in script and "Open map" not in script
    assert "window_index" not in script
    layout = _between(editor, "  wrap.append(", "  // One listener for the keys")
    order = [
        "OE.el('h1', {}, 'Label sheet'), partLine)",
        "gradedNote,",
        "leadLines,",
        "faceInstruction,",
        "ruleNote,",
        "switchRow,",
        "viewToggle, filterBox)",
        "notice,",
        "ruleBlock,",
        "noRuleLine,",
        "area,",
        "details),",
        "progress, OE.el('div', { class: 'row' }, done, saveButton), finishType)",
    ]
    places = [layout.index(text) for text in order]
    assert places == sorted(places)
    assert "data.name" not in layout
    fold = _between(editor, "  const foldLabel = 'Details';", "\n\n")
    for text in (
        "OE.el('div', { class: 'mono', style: 'overflow-wrap:anywhere' }, data.name)",
        "instruction, hint, ...foldLines, clearNote,",
        "OE.fold(foldLabel, ...foldParts)",
    ):
        assert text in fold, text
    assert "// WORDING: r4h S1\n  const foldLabel = 'Details';" in editor
    # A class map's class face has the seen sheet's blind line (the owner's O1).
    assert (
        "const sentLine = seen || face === 'class' ? W.blindSeen : wrongFace ? "
        "W.wrongLine : blind ? W.codeLine : '';" in editor
    )
    assert "[sentLine, data.has_place ? W.openMap : '']" in editor
    assert (
        "const clearNote = pageRule ? (picturesPossible ? OE.el('div', { class: "
        "'small faint' }, W.clear) : null) : null;" in editor
    )
    # r5-b J13: the Cannot tell line is the lead's second line; a graded sheet
    # has none (r5-b J06); r5-e H1: no sheet has the S2 line as a third.
    assert "next: gradedNote ? null : cannotLineText || null," in editor
    assert "cannot: null," in editor
    # The rule block carries the rule and, where the face has one, the cloud
    # line; r5-e H2: its scope is in the Details fold (scopeNote) and, after
    # review, the kelp-only line right after it (limitNote).
    rule = _between(editor, "  const ruleBlock = pageRule ?", "\n  const clearNote")
    assert "W.clear" not in rule and "data.rule_scope" not in rule
    assert "classNoPeriod, scopeNote, limitNote, gradedWhy," in fold
    # One Cannot tell line where the sheet takes it; a certificate's own line.
    line = _between(editor, "  const cannotLineText = (() => {", "\n  })();")
    assert "if (takesQ) return W.cannotTellLine;" in line
    assert "'A certificate needs ' + orList(named) + ' on every ' + W.one + '.'" in line
    assert "'A certificate needs right or wrong on every ' + W.one + '.'" in line
    # The fold's instruction keeps what Cannot tell does; the face's has none.
    told = _between(editor, "  function instructionText() {", "\n  }\n")
    assert "(picturesPossible ? W.cannotTell : W.cannotTellBare)" in told
    face = _between(editor, "  function faceInstructionText() {", "\n  }\n")
    assert "cannotTell" not in face and "worked out when each label" not in script
    # A sheet with places says what to judge from (E/sheet-1), squares or
    # points (r5-b J11).
    assert (
        "const whatBare = data.has_place ? W.whatPlaced : (twoMap ? "
        "W.whatBarePair : W.whatBare);" in editor
    )
    assert "what: wrongFace ? null : picturesPossible ? W.what : whatBare," in editor


def test_the_footer_shows_the_finish_button_from_the_first_open() -> None:
    """Cheap 4 (B/sheet-finish-button): "Saved automatically" as plain text
    beside the counter, Save labels only while a change waits, a save runs or
    one failed, and the finish button from the first open, greyed until every
    answer is saved, with the A/F17 line beside it then."""
    editor = _between(_script(), "OE.sheetEditor = (app, data,", "\n};")
    assert (
        "const savedNote = OE.el('span', { class: 'small muted' }, W.saved);" in editor
    )
    assert (
        "counterNum, OE.el('span', {}, 'labelled'), counterNote, savedNote);" in editor
    )
    progress = _between(editor, "  function updateProgress() {", "\n  }\n")
    for text in (
        "const showSave = Boolean(left) || state.saving || state.saveFailed;",
        "saveButton.hidden = !showSave;",
        "savedNote.hidden = showSave;",
        "saveButton.textContent = state.saving ? 'Saving...' : 'Save labels';",
        "updateFinish();",
    ):
        assert text in progress, text
    assert "W.saved" not in progress
    assert "const ready = () => complete() && !pending() && !state.stale;" in editor
    assert "if (finishing || !ready()) return;" in editor
    control = _between(editor, "  function finishControl() {", "\n  }\n")
    assert "'aria-disabled': 'true'," in control
    finish = _between(editor, "  function updateFinish() {", "\n  }\n")
    for text in (
        "button.setAttribute('aria-disabled', String(!ok));",
        "if (ok && !done.contains(everyDoneLine)) done.prepend(everyDoneLine);",
        "else if (!ok && done.contains(everyDoneLine)) everyDoneLine.remove();",
    ):
        assert text in finish, text
    # r5-d R06: a sheet already graded says that it gave its result instead.
    assert (
        "const everyDoneLine = OE.el('span', { class: 'small' },\n"
        "    data.graded && typeof data.graded === 'object' ? W.everyDoneGraded"
        " : W.everyDone);" in editor
    )
    assert "function finished(" not in editor
    # A save that failed keeps Save labels in view until one goes through.
    save = _between(editor, "  async function save(", "\n  }\n")
    assert "if (!err.stale) state.saveFailed = true;" in save
    assert "state.saveFailed = false;" in save


def test_a_class_face_offers_cannot_tell_in_the_panel() -> None:
    """S4 in sheet.js: the class face's list, keys and keys line include ?
    where the face takes it, and Cannot tell is always a button after the
    first nine classes."""
    editor = _between(_script(), "OE.sheetEditor = (app, data,", "\n};")
    options = _between(editor, "  function classOptions(h) {", "\n  }\n")
    assert "if (!twoMap) return takesQ ? [...listed, OE.UNJUDGED] : listed;" in options
    keys = _between(editor, "  function answerKeys(w) {", "\n  }\n")
    assert (
        "return takesQ ? [...classes, { value: OE.UNJUDGED, label: 'Cannot tell', "
        "keys: [OE.UNJUDGED] }] : classes;" in keys
    )
    hint = _between(editor, "  function listHintText() {", "\n  }\n")
    assert "if (takesQ) parts.push('? cannot tell');" in hint
    assert "twoMap && takesQ" not in editor
    answers = _between(editor, "  function faceAnswers(w) {", "\n  }\n")
    assert (
        "[...named.slice(0, 9), ...cannotTell].map((a) => OE.el('button', {" in answers
    )
    assert "const rest = named.slice(9);" in answers


def test_the_pictures_colours_are_a_joined_toggle_and_each_date_a_block() -> None:
    """A/F23 and A/F24: the colour switch is OE.toggle with each button's
    data-focus kept; each date an outlined block, its label on top."""
    dates = _between(_script(), "OE.datesBlock = (", "\n};")
    assert (
        "const colours = OE.toggle(kinds, colour, (kind) => { if (choose) "
        "choose(kind); },\n    { label: 'Colours of the pictures', small: true });"
        in dates
    )
    assert (
        "colours.querySelectorAll('button').forEach((b, i) => "
        "b.setAttribute('data-focus', 'colour:' + kinds[i][0]));" in dates
    )
    assert "font-weight:600' : ''" not in dates
    assert (
        "{ class: 'stack', style: 'gap:4px;border:1px solid var(--rule);"
        "border-radius:4px;padding:8px' }," in dates
    )


def test_what_claude_hears_names_no_window_class_or_answer() -> None:
    """The context line stays counts and words only (test_mcp_server holds the
    same)."""
    told = _between(_script(), "  function contextText() {", "\n  }\n")
    for left_out in ("map_class", "stratum", "handle", "answer"):
        assert left_out not in told, left_out
