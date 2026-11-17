# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""A scores map's plan, labelled through the MCP App's panel, graded by the estimate.

A scores map (the confidence design on per-class scores) is graded from
``wrong`` on every window; the panel's label rule says so, and the labels
it writes the way ``views/sheet.js`` sends them are graded. A scores map
whose file names its classes is labelled with the class seen instead, and
the server works out ``wrong`` from the design's map class. The class map's
and the two-map design's round trips are beside their own tools' tests
(``tests/tools/test_class_map_tool.py``, ``tests/tools/test_which_map_tool.py``).
A plan from a review set's saved points is of points (finding R3), and the
graded mark survives the plan tool writing a lost design file again, while a
fresh sheet under the same name starts clean (R6).
"""

from __future__ import annotations

import csv
import json
import os
import random
from pathlib import Path
from typing import Any

import pytest

from olmoearth_agent.harness.state import ThreadState
from olmoearth_agent.llm.types import ToolCall
from olmoearth_agent.mcp_app import sheet as panel
from olmoearth_agent.tools import estimation
from olmoearth_agent.tools.estimation import build_estimation_tools
from olmoearth_agent.tools.registry import ToolContext, ToolRegistry

pytest.importorskip("oe_inferencex.estimate")


@pytest.fixture(autouse=True)
def _roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("OLMOEARTH_SCORES_ROOT", str(tmp_path))
    monkeypatch.setenv("OLMOEARTH_OUTPUT_ROOT", str(tmp_path))
    return tmp_path


async def _call(name: str, args: dict[str, Any]) -> Any:
    registry = ToolRegistry()
    registry.register_all(build_estimation_tools())
    ctx = ToolContext(studio=None, state=ThreadState())  # type: ignore[arg-type]
    return await registry.dispatch(ToolCall(id="1", name=name, arguments=args), ctx)


def _scores(n: int = 600, seed: int = 11) -> tuple[list[list[float]], list[int]]:
    """Three-class logits and, per window, whether the map is wrong there."""
    rng = random.Random(seed)
    scores, wrong = [], []
    for _ in range(n):
        row = [rng.gauss(0.0, 1.5) for _ in range(3)]
        top = sorted(row, reverse=True)
        scores.append(row)
        wrong.append(1 if rng.random() < 0.6 * 2.718 ** -(top[0] - top[1]) else 0)
    return scores, wrong


def test_the_panel_label_kinds_are_the_estimates() -> None:
    assert panel.CLASS_LABELLED == estimation.CLASS_LABELLED
    assert panel.TWO_MAP_DESIGN == estimation.TWO_MAP_DESIGN


@pytest.mark.asyncio
async def test_a_scores_map_is_labelled_right_or_wrong_and_graded() -> None:
    scores, truth = _scores()
    plan = await _call("olmoearth_plan_label_sample", {"scores": scores, "budget": 60})
    assert plan["ok"] is True, plan
    data = panel.read_sheet(plan["result"]["labels_csv_path"])
    assert data["label_rule"] == "wrong" and data["kind"] == "one_map"
    assert data["design_path"]
    # The finish button's call is the one made below, with these two paths.
    assert (data["job"], data["finish_label"]) == (
        "estimate",
        "Estimate the error rate",
    )
    assert data["finish_call"] == (
        f"olmoearth_estimate_map_error with design_path {data['design_path']} "
        f"and labels_path {data['path']}"
    )
    # What the panel sends on the right-or-wrong face: each window's answer by
    # its handle (the server's own map from handle to window, never sent).
    assert data["face"] == "wrong"
    _, handles = panel.window_handles(data["path"])
    answers = [
        {"handle": w["handle"], "answer": str(truth[handles[w["handle"]]])}
        for w in data["windows"]
    ]
    # The version the panel read goes with the save, as the panel sends it.
    saved = panel.save_answers(
        data["path"],
        sheet_id=data["sheet_id"],
        face=data["face"],
        rule=data["label_rule"],
        answers=answers,
        version=data["version"],
    )
    assert saved["n_labelled"] == data["n_rows"]
    assert saved["version"] == panel.sheet_progress(data["path"])["version"]
    out = await _call(
        "olmoearth_estimate_map_error",
        {"design_path": data["design_path"], "labels_path": data["path"]},
    )
    assert out["ok"] is True, out


@pytest.mark.asyncio
async def test_classes_alone_on_a_scores_map_are_refused() -> None:
    """What the panel wrote before the label rule (review wf_c3894049-4e7, P1):
    the class seen on a scores map; the estimate needs wrong there."""
    scores, _truth = _scores()
    plan = await _call("olmoearth_plan_label_sample", {"scores": scores, "budget": 40})
    data = panel.read_sheet(plan["result"]["labels_csv_path"])
    _, handles = panel.window_handles(data["path"])
    panel.save_labels(
        data["path"],
        [{"window_index": w, "reference_class": "1"} for w in handles.values()],
    )
    out = await _call(
        "olmoearth_estimate_map_error",
        {"design_path": data["design_path"], "labels_path": data["path"]},
    )
    assert out["ok"] is False and "'wrong'" in out["error"]


def _two_class(
    n: int = 600, seed: int = 5
) -> tuple[list[list[float]], list[int], list[int]]:
    """Two-class logits, the map's class (the arg-max) and whether the map is
    wrong, per window."""
    rng = random.Random(seed)
    scores, on_map, wrong = [], [], []
    for _ in range(n):
        row = [rng.gauss(0.0, 1.5) for _ in range(2)]
        scores.append(row)
        on_map.append(0 if row[0] > row[1] else 1)
        wrong.append(1 if rng.random() < 0.6 * 2.718 ** -abs(row[0] - row[1]) else 0)
    return scores, on_map, wrong


@pytest.mark.asyncio
async def test_a_named_scores_map_is_labelled_with_the_class_seen_and_graded(
    _roots: Path,
) -> None:
    """A scores file that names its classes, planned through scores_path: the
    panel's copy of the sheet asks for the class seen and holds no map class,
    the server works out wrong from the design's map class, and the estimate
    grades the sheet, per class too. One window the reviewer cannot tell
    bounds the rate and leaves the per-class table out."""
    scores, on_map, truth = _two_class()
    path = _roots / "kelp_scores.json"
    path.write_text(
        json.dumps(
            {
                "grid": [20, 30],
                "scores": scores,
                "classes": {"0": "absent", "1": "present"},
            }
        ),
        encoding="utf-8",
    )
    plan = await _call(
        "olmoearth_plan_label_sample", {"scores_path": str(path), "budget": 60}
    )
    assert plan["ok"] is True, plan
    data = panel.read_sheet(plan["result"]["labels_csv_path"])
    assert data["label_rule"] == "seen" and data["face"] == "seen"
    # A scores file's windows: squares, which pictures may show.
    assert (data["unit"], data["pictures_possible"]) == ("square", True)
    assert [a["value"] for a in data["answers"]] == ["1", "0", "?"]
    assert all(set(w) == {"handle", "answer"} for w in data["windows"])
    assert '"map_class"' not in json.dumps(data)
    design = json.loads(Path(data["design_path"]).read_text(encoding="utf-8"))
    assert design["population"]["map_class"] == on_map
    # What the panel sends: the code seen alone, by handle (the other class
    # where the map is wrong).
    _, handles = panel.window_handles(data["path"])
    windows = [handles[w["handle"]] for w in data["windows"]]
    answers = [
        {"handle": w["handle"], "answer": str(on_map[i] ^ truth[i])}
        for w, i in zip(data["windows"], windows, strict=True)
    ]
    saved = panel.save_answers(
        data["path"],
        sheet_id=data["sheet_id"],
        face="seen",
        rule="seen",
        answers=answers,
    )
    assert saved["n_labelled"] == data["n_rows"] == len(windows)
    with open(data["path"], encoding="utf-8", newline="") as fh:
        written = {int(r["window_index"]): r["wrong"] for r in csv.DictReader(fh)}
    assert written == {w: str(truth[w]) for w in windows}
    args = {"design_path": data["design_path"], "labels_path": data["path"]}
    out = await _call("olmoearth_estimate_map_error", args)
    assert out["ok"] is True, out
    assert out["result"]["per_class"]
    panel.save_answers(
        data["path"],
        sheet_id=data["sheet_id"],
        face="seen",
        rule="seen",
        answers=[{"handle": data["windows"][0]["handle"], "answer": "?"}],
    )
    again = await _call("olmoearth_estimate_map_error", args)
    assert again["ok"] is True, again
    assert again["result"]["n_unjudged"] == 1
    assert "per_class" not in again["result"]


@pytest.mark.asyncio
async def test_a_plan_from_a_review_sets_saved_points_is_of_points() -> None:
    """Round 4, finding R3: olmoearth_review_set_from_result saves a Studio
    result's sampled points (its own writer, SampledScores.to_file: one pixel
    per point, no raster); the plan tool, drawing from that file, copies its
    result into the design's source and no provenance, so the sheet is of
    points and offers no pictures."""
    from olmoearth_agent.analysis.review_set import regression_scores
    from olmoearth_agent.tools.review_set import SampledScores, write_json_file

    rng = random.Random(3)
    values = [rng.random() for _ in range(64)]
    rows, kind, assumption = regression_scores(values, min_value=0.0, max_value=1.0)
    sampled = SampledScores(
        result_id="00000000-0000-4000-8000-000000000001",
        property_name="kelp",
        score_kind=kind,
        assumption=assumption,
        threshold=0.5,
        declared_range=(0.0, 1.0),
        grid=8,
        windows=list(range(64)),
        values=values,
        scores=rows,
        centres=[(18.3 + i % 8 / 100, -33.9 - i // 8 / 100) for i in range(64)],
        n_sampled=64,
        n_nodata=0,
        n_failed=0,
    )
    path = write_json_file("scores_00000000_kelp_g8.json", sampled.to_file())
    plan = await _call(
        "olmoearth_plan_label_sample",
        {"scores_path": path, "design": "random", "budget": 20},
    )
    assert plan["ok"] is True, plan
    data = panel.read_sheet(plan["result"]["labels_csv_path"])
    assert (data["unit"], data["pictures_possible"]) == ("point", False)
    design = json.loads(Path(data["design_path"]).read_text(encoding="utf-8"))
    source = design["population"]["source"]
    assert source["result_id"] == sampled.result_id
    assert "provenance" not in source and "studio" not in source


def _graded_place(path: str) -> Path:
    return Path(path).with_name("." + Path(path).name + panel.GRADED_SUFFIX)


def _answer_all(data: dict[str, Any], truth: list[int]) -> None:
    """The panel's save of every window's answer, right or wrong, by handle."""
    _, handles = panel.window_handles(data["path"])
    panel.save_answers(
        data["path"],
        sheet_id=data["sheet_id"],
        face=data["face"],
        rule=data["label_rule"],
        answers=[
            {"handle": w["handle"], "answer": str(truth[handles[w["handle"]]])}
            for w in data["windows"]
        ],
        version=data["version"],
    )


@pytest.mark.asyncio
async def test_the_graded_mark_survives_a_design_file_written_again() -> None:
    """Round 4, R6 (the lead's task J), through the plan tool itself: graded,
    the design file deleted, the plan run again with the same arguments (it
    keeps the labelled sheet and writes the design file again, the same
    content with another stamp), an answer changed, graded again: the mark
    says the answers changed. Then the labelled sheet moved aside and the
    plan run again: a fresh blank sheet under the same name has no notice,
    and its first grading is clean."""
    scores, truth = _scores()
    request = {"scores": scores, "budget": 60}
    plan = await _call("olmoearth_plan_label_sample", request)
    assert plan["ok"] is True, plan
    data = panel.read_sheet(plan["result"]["labels_csv_path"])
    _answer_all(data, truth)
    args = {"design_path": data["design_path"], "labels_path": data["path"]}
    assert (await _call("olmoearth_estimate_map_error", args))["ok"] is True
    first = panel.record_grading(
        data["path"],
        panel.sheet_version(data["path"]),
        design_path=data["design_path"],
    )
    assert (first["changed"], first["gradings"]) == (False, 1)
    # The design file lost, the plan run again: the sheet kept, the design
    # file written again.
    Path(data["design_path"]).unlink()
    kept = await _call("olmoearth_plan_label_sample", request)
    assert kept["ok"] is True, kept
    assert os.path.realpath(kept["result"]["labels_csv_path"]) == data["path"]
    assert Path(data["design_path"]).is_file()
    copy = panel.read_sheet(data["path"])
    assert copy["graded"] == {
        "first_at": first["first_at"],
        "changed_since_first": False,
    }
    # One answer changed, then graded again: changed.
    window = copy["windows"][0]
    flipped = "0" if window["answer"] == "1" else "1"
    panel.save_answers(
        data["path"],
        sheet_id=copy["sheet_id"],
        face=copy["face"],
        rule=copy["label_rule"],
        answers=[{"handle": window["handle"], "answer": flipped}],
        version=copy["version"],
    )
    assert (await _call("olmoearth_estimate_map_error", args))["ok"] is True
    again = panel.record_grading(data["path"], panel.sheet_version(data["path"]))
    assert (again["changed"], again["gradings"], again["first_at"]) == (
        True,
        2,
        first["first_at"],
    )
    assert panel.graded_state(data["path"])["changed"] is True
    # Moved aside, planned again: a fresh blank sheet under the same name.
    Path(data["path"]).rename(Path(data["path"]).with_name("first_reviewer.csv"))
    fresh = await _call("olmoearth_plan_label_sample", request)
    assert fresh["ok"] is True, fresh
    blank = panel.read_sheet(fresh["result"]["labels_csv_path"])
    assert blank["path"] == data["path"] and blank["n_labelled"] == 0
    assert _graded_place(blank["path"]).is_file()
    assert blank["graded"] is None and panel.graded_state(blank["path"]) is None
    _answer_all(blank, truth)
    assert not _graded_place(blank["path"]).exists()
    clean = panel.record_grading(blank["path"], panel.sheet_version(blank["path"]))
    assert (clean["changed"], clean["gradings"]) == (False, 1)
