# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Tests for scripts/verify_demo_tile.py, which writes docs/VERIFICATION_DEMO.md.

The script drives the agent's verification tools on the Dynamic World tile
that olmoearth-inferencex ships. Here its own functions run on a small
synthetic tile (24 x 24 windows of three classes, the truth fixed), every
tool for real, as on the cluster: the page states each figure as the tool
returned it, the truth check is computed from the tile's truth (another
truth for the same map flips it while the tools' figures stay), and the
envelopes file holds every call and no path of the run. The real tile is
read only to check that it is the package's, read and credited as the
package's demo reads and credits it.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from olmoearth_agent.provenance.log import VERIFICATION_TOOLS
from olmoearth_agent.reporting.narrative import ACCURACY_HEADING, INTERVAL_NAMES
from olmoearth_agent.security.paths import OUTPUT_ROOT_ENV
from olmoearth_agent.skills.registry import OPT_IN_ENV, OPT_IN_TOOLS
from olmoearth_agent.tools.review_set import SCORES_ROOT_ENV
from olmoearth_agent.tools.statistical_rules import percent

try:
    import numpy as np
    import rasterio  # noqa: F401 - the script writes the tile's GeoTIFF with it
    from oe_inferencex import demo as pkg_demo

    HAVE_EXTRA = True
except ImportError:  # pragma: no cover - the core install has neither
    HAVE_EXTRA = False

pytestmark = pytest.mark.skipif(
    not HAVE_EXTRA, reason="needs the inferencex extra (numpy, rasterio)"
)

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "verify_demo_tile.py"

#: The tools in the order the script calls them, when a zone is certified.
ORDER = [
    "olmoearth_scores_from_file",
    "olmoearth_review_set",
    "olmoearth_plan_label_sample",
    "olmoearth_estimate_map_error",
    "olmoearth_certify_zone",
    "olmoearth_write_certificate",
    "olmoearth_verify_certificate",
    "olmoearth_verify_certificate",
    "olmoearth_plan_label_sample",
    "olmoearth_estimate_map_error",
    "olmoearth_case_narrative",
    "olmoearth_provenance_summary",
]

#: The synthetic run's review budget: above the 50 windows a review set lists
#: inline, so its windows are read from its list file, as on the real tile.
REVIEW_BUDGET = 0.1

#: The synthetic tile's side, in windows.
SIDE = 24


@pytest.fixture(scope="module")
def script() -> ModuleType:
    """The script, imported from its file."""
    spec = importlib.util.spec_from_file_location("verify_demo_tile", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # its dataclasses look their module up
    spec.loader.exec_module(module)
    return module


def _synthetic(script: ModuleType) -> Any:
    """24 x 24 windows of three classes in bands of 8 rows, about a tenth wrong.

    The map is wrong only at its least confident windows (top-1 probability
    0.40 to 0.50; every right window 0.55 to 0.95), so a random sample of 300
    certifies a zone at alpha 0.1. The expert marked nothing in 20 windows
    (-1), and one annotated window has no finite probability: 555 windows
    are kept.
    """
    rng = np.random.default_rng(7)
    truth = np.repeat((np.arange(SIDE) // 8)[:, None], SIDE, axis=1)
    wrong = rng.random((SIDE, SIDE)) < 0.1
    mapped = np.where(wrong, (truth + 1) % 3, truth)
    unsure = rng.uniform(0.40, 0.50, (SIDE, SIDE))
    sure = rng.uniform(0.55, 0.95, (SIDE, SIDE))
    top = np.where(wrong, unsure, sure)
    rows, cols = np.indices((SIDE, SIDE))
    probs = np.zeros((3, SIDE, SIDE))
    probs[mapped, rows, cols] = top
    probs[(mapped + 1) % 3, rows, cols] = 0.6 * (1.0 - top)
    probs[(mapped + 2) % 3, rows, cols] = 0.4 * (1.0 - top)
    probs = probs.astype(np.float32)
    probs[:, SIDE - 1, SIDE - 1] = np.nan
    expert = truth.astype(np.int16)
    expert[:2, :10] = -1
    meta = {
        "classes": ["water", "crops", "bare"],
        "tile": "label_dw_synthetic-20240601.tif",
        "lon": -71.8,
        "lat": -16.4,
        "window_m": 40,
        "zenodo_record": 4766508,
        "licence": "CC BY 4.0",
    }
    return script.Tile(probs=probs, expert=expert, meta=meta)


@pytest.fixture(scope="module")
def tile(script: ModuleType) -> Any:
    """The synthetic tile."""
    return _synthetic(script)


@pytest.fixture(scope="module")
def generated(
    script: ModuleType, tile: Any, tmp_path_factory: pytest.TempPathFactory
) -> Any:
    """The script's run on the synthetic tile: its calls, truth check and files.

    A call that fails puts its whole envelope in the test log, since the
    tests run where nobody watches them.
    """
    work = tmp_path_factory.mktemp("work").resolve()
    docs = tmp_path_factory.mktemp("docs")
    with pytest.MonkeyPatch.context() as mp:
        # run_demo points both roots at the work folder; restored afterwards.
        mp.setenv(SCORES_ROOT_ENV, str(work))
        mp.setenv(OUTPUT_ROOT_ENV, str(work))
        mp.delenv(OPT_IN_ENV, raising=False)
        try:
            return script.generate(tile, work, docs, review_budget=REVIEW_BUDGET)
        except script.CallFailed as exc:
            envelope = json.dumps(exc.call.envelope, indent=1, default=str)
            pytest.fail(f"{exc}\n{envelope}")


def _demo(script: ModuleType, work: Path, **results: tuple[str, Any]) -> Any:
    """A finished run of the given steps, each ``step=(tool, envelope)``."""
    run = script.Run()
    for step, (tool, envelope) in results.items():
        run.steps[step] = len(run.calls)
        run.calls.append(
            script.Call(step=step, tool=tool, arguments={}, envelope=envelope)
        )
    return script.Demo(run=run, work=work, version=None, tamper=None)


def _ok(result: dict[str, Any]) -> dict[str, Any]:
    """A succeeded call's envelope."""
    return {"ok": True, "result": {"available": True, **result}}


def _truth(tile: Any) -> tuple[Any, Any]:
    """The tile's kept windows and those the map gets wrong, flat, from scratch."""
    kept = (tile.expert >= 0) & np.isfinite(tile.probs).all(axis=0)
    finite = np.where(np.isfinite(tile.probs), tile.probs, -np.inf)
    wrong = kept & (finite.argmax(axis=0) != tile.expert)
    return kept.ravel(), wrong.ravel()


def _row(page: str, prefix: str) -> str:
    """The one line of the page that starts with ``prefix``."""
    (line,) = [x for x in page.splitlines() if x.startswith(prefix)]
    return line


def test_the_tools_run_in_the_agents_order(
    script: ModuleType, generated: Any, tile: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every call succeeds, in the agent's order, default tools only, recorded
    as the lead agent records them; the labels are the expert's, written into
    each plan's own sheet."""
    demo = generated.demo
    run = demo.run
    assert [c.tool for c in run.calls] == ORDER
    assert all(c.envelope["ok"] is True for c in run.calls)
    assert not set(ORDER) & set(OPT_IN_TOOLS)
    assert [e.api_call for e in run.state.provenance.entries] == ORDER
    recorded = [v.tool for v in run.state.provenance.verification]
    assert recorded == [t for t in ORDER if t in VERIFICATION_TOOLS]
    plan = demo.result("plan")
    assert plan["design"] == "random" and plan["budget"] == 300
    assert plan["seed_from_map"]["alpha"] == 0.1
    class_plan = demo.result("class_plan")
    assert (class_plan["design"], class_plan["seed"]) == ("class", 0)
    # Three classes of about 185 windows: the floor is 150, so 300 stands,
    # fixed from the scores file the plan reads.
    scores_path = demo.result("scores")["scores_path"]
    monkeypatch.setenv(SCORES_ROOT_ENV, str(demo.work))
    assert demo.class_budget == script.class_design_budget(scores_path)
    assert demo.class_budget["floor"] == 150
    assert class_plan["budget"] == demo.class_budget["budget"] == 300
    # Every envelope is on disk in the work folder too, as the run went.
    log = json.loads((demo.work / script.ENVELOPES).read_text(encoding="utf-8"))
    assert [c["tool"] for c in log["calls"]] == ORDER
    for made in (plan, class_plan):
        with open(made["labels_csv_path"], encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == made["budget"]
        for row in rows:
            r, c = divmod(int(row["window_index"]), SIDE)
            seen = int(tile.expert[r, c])
            assert int(row["reference_class"]) == seen
            assert int(row["wrong"]) == int(seen != int(row["map_class"]))
    assert demo.result("certify")["certified"] is True, "the synthetic map certifies"


def test_the_page_states_each_figure_as_the_tool_returned_it(
    script: ModuleType, generated: Any
) -> None:
    """Each row of the calls table holds the tool's own figures, formatted as
    the tools format them; the report's accuracy section and the verification
    calls are on the page verbatim; no path of the run is left on it."""
    page, demo = generated.page, generated.demo
    assert page.splitlines()[0] == script.FIRST_LINE
    assert "`make verify-demo`" in script.FIRST_LINE
    assert "scripts/verify_demo_tile.py" in script.FIRST_LINE
    assert generated.page_path.read_text(encoding="utf-8") == page

    def row(step: str) -> str:
        n = demo.number(step)
        return _row(page, f"| {n} | `{demo.run.calls[n - 1].tool}` |")

    scores = demo.result("scores")
    shown = f"{scores['n_valid']:,} valid windows of a {SIDE} x {SIDE} grid"
    assert shown in row("scores")
    review = demo.result("review")
    shown = f"{review['n_review']:,} windows to check first, of {review['n_windows']:,}"
    assert shown in row("review")
    # Every design's estimate gives the interval valid at every sample size
    # first, then the design's (under the random design the same exact one).
    for step in ("estimate", "class_estimate"):
        est, line = demo.result(step), row(step)
        point = est["valid_interval"]
        shown = f"error rate {percent(point['estimate'])} from {est['n_labelled']:,}"
        assert shown in line
        intervals = est["intervals"]
        assert [e["interval"] for e in intervals] == ["valid", "design"]
        for entry in intervals:
            # A table cell escapes a "|" in the package's words (script.cell).
            ends = f"{percent(entry['low'])} to {percent(entry['high'])}"
            assert script.cell(f"{ends} (method: {entry['method']})") in line
    random_valid, random_design = demo.result("estimate")["intervals"]
    assert (random_valid["low"], random_valid["high"]) == (
        random_design["low"],
        random_design["high"],
    )
    zone = demo.result("certify")
    assert row("certify").endswith(f"| {script.cell(zone['verdict'])} |")
    written = demo.result("certificate")
    assert written["certificate_written"] is True
    assert written["n_zone"] == zone["n_zone"]
    shown = f"{written['n_zone']:,} windows, coverage {written['coverage']:g}"
    assert shown in row("certificate")
    assert demo.result("verify")["verified"] is True
    assert "verified: yes" in row("verify")
    assert demo.result("tamper")["verified"] is False
    assert "verified: no" in row("tamper")
    assert demo.tamper is None
    # The tampered copy was checked, so the lead describes that check.
    assert f"{script.CALLS_LEAD} {script.TAMPER_LEAD}" in page
    # The class budget was not raised, so its row gives no reason for one.
    assert "planned" not in row("class_plan")
    # What the rate is of: the 40 m windows, not Dynamic World's 10 m pixels.
    heading = "## Does each interval cover the true error rate of the tile's"
    assert f"{heading} 40 m windows" in page
    assert "not Dynamic World's 10 m accuracy" in page
    assert script.ONE_DRAW in page and "delta" in script.ONE_DRAW

    section = script.report_section(demo.result("narrative")["markdown"])
    assert section.startswith(ACCURACY_HEADING)
    assert script.placeholder(section, demo.work) in page
    calls = demo.result("provenance")["verification_calls"]
    dumped = json.dumps(calls, indent=1, default=str)
    assert script.placeholder(dumped, demo.work) in page
    assert str(demo.work) not in page
    assert os.path.realpath(demo.work) not in page
    assert script.WORK in page


def test_the_truth_check_is_computed_from_the_truth(
    script: ModuleType, generated: Any, tile: Any
) -> None:
    """The population's error rate, each interval's coverage, the review set's
    share of the errors and the zone's error rate come from the tile's truth;
    another truth for the same map gives other flags from the same figures."""
    demo, truth = generated.demo, generated.truth
    kept, wrong = _truth(tile)
    n, n_wrong = int(kept.sum()), int(wrong.sum())
    rate = n_wrong / n
    assert n == 555
    assert truth["population"] == {
        "grid": [SIDE, SIDE],
        "n_windows": n,
        "n_wrong": n_wrong,
        "error_rate": rate,
    }

    # The intervals are the tools' own, each one the tool returns, named as
    # the tool names it: the random design's valid interval and its own, then
    # the class design's valid interval and its own.
    estimate, by_class = demo.result("estimate"), demo.result("class_estimate")
    expected = [*estimate["intervals"], *by_class["intervals"]]
    got = [(e["low"], e["high"]) for e in truth["intervals"]]
    assert got == [(e["low"], e["high"]) for e in expected]
    assert [e["interval"] for e in truth["intervals"]] == [
        "valid",
        "design",
        "valid",
        "design",
    ]
    assert [e["design"] for e in truth["intervals"]] == [
        "random",
        "random",
        "class",
        "class",
    ]
    for entry in truth["intervals"]:
        assert entry["covers"] is (entry["low"] <= rate <= entry["high"])
        name = INTERVAL_NAMES[entry["interval"]]
        check = f"| {name}, {entry['design']} design (call {entry['call']})"
        holds = "| covers |" if entry["covers"] else "| misses |"
        assert _row(generated.page, check).endswith(holds)

    # The review set's windows, from its own list file.
    review = demo.result("review")
    assert review["n_review"] > 50 and review["review_list_path"]
    with open(review["review_list_path"], encoding="utf-8", newline="") as fh:
        listed = [int(row["window_index"]) for row in csv.DictReader(fh)]
    inside = int(wrong[listed].sum())
    assert truth["review_set"]["n_review"] == len(listed) == review["n_review"]
    assert truth["review_set"]["errors_inside"] == inside
    assert truth["review_set"]["share_of_errors"] == inside / n_wrong
    assert truth["review_set"]["ceiling"] == min(1.0, (len(listed) / n) / rate)

    # The certified zone's windows, from the zone file the tool wrote.
    zone = demo.result("certify")
    stored = json.loads(Path(zone["zone_path"]).read_text(encoding="utf-8"))
    in_zone = [int(i) for i in stored["window_indices"]]
    zone_wrong = int(wrong[in_zone].sum())
    assert truth["zone"]["n_zone"] == len(in_zone) == zone["n_zone"]
    assert truth["zone"]["n_wrong"] == zone_wrong
    assert truth["zone"]["at_most_alpha"] is (zone_wrong / len(in_zone) <= 0.1)
    # The zone's row names delta, read from the same zone file: its statement
    # may fail on up to delta of draws.
    assert truth["zone"]["delta"] == stored["delta"]
    check = f"| Certified zone (call {truth['zone']['call']})"
    assert f"delta={stored['delta']:g} of draws" in _row(generated.page, check)

    # Another truth for the same map, every annotated window wrong: the tools'
    # figures are the same, and every flag is computed again from it.
    mapped = np.where(np.isfinite(tile.probs), tile.probs, -np.inf).argmax(axis=0)
    other = script.Tile(
        probs=tile.probs,
        expert=np.where(tile.expert >= 0, (mapped + 1) % 3, -1).astype(np.int16),
        meta=tile.meta,
    )
    again = script.truth_check(script.truth_of(other), demo)
    assert again["population"]["error_rate"] == 1.0
    assert [(e["low"], e["high"]) for e in again["intervals"]] == got
    assert [e["covers"] for e in again["intervals"]] == [False] * len(got)
    assert again["review_set"]["share_of_errors"] == len(listed) / n
    assert again["zone"]["at_most_alpha"] is False


def test_the_envelopes_file_holds_every_call_and_no_path_of_the_run(
    script: ModuleType, generated: Any
) -> None:
    """docs/verification_demo.json: every envelope in call order with the work
    folder written <work>, sorted keys, and the truth check."""
    text = generated.document_path.read_text(encoding="utf-8")
    assert text == generated.document
    work = generated.demo.work
    assert str(work) not in text and os.path.realpath(work) not in text
    doc = json.loads(text)
    assert text == json.dumps(doc, indent=1, sort_keys=True) + "\n"
    assert doc["work_folder"] == script.WORK
    calls = generated.demo.run.calls
    assert [c["tool"] for c in doc["calls"]] == [c.tool for c in calls]
    for kept, call in zip(doc["calls"], calls):
        raw = json.dumps(call.envelope, sort_keys=True, default=str)
        stored = json.dumps(kept["envelope"], sort_keys=True)
        assert stored == script.placeholder(raw, work)
    assert doc["truth_check"] == json.loads(json.dumps(generated.truth))


def test_the_tile_is_the_one_the_package_ships(script: ModuleType) -> None:
    """The tile is found in the installed package, read as its demo reads it,
    and credited word for word as its demo credits it."""
    assert os.path.samefile(script.sample_path(), pkg_demo.SAMPLE)
    tile = script.load_tile()
    real = pkg_demo._real()
    assert script.credit(tile.meta) == real["credit"]
    assert np.array_equal(tile.probs, real["scores"], equal_nan=True)
    assert np.array_equal(tile.expert, real["truth"])
    assert tile.classes == list(real["classes"])
    assert script.tile_date(tile.meta) == "2019-06-09"
    kept = tile.kept()
    assert not kept[tile.expert < 0].any()
    assert 0 < int(kept.sum()) < kept.size


@pytest.mark.asyncio
async def test_a_call_that_does_not_succeed_stops_the_run(
    script: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed call raises with its envelope, recorded all the same; an
    opt-in tool is never called."""
    monkeypatch.setenv(SCORES_ROOT_ENV, str(tmp_path))
    monkeypatch.setenv(OUTPUT_ROOT_ENV, str(tmp_path))
    run = script.Run(log=tmp_path / script.ENVELOPES)
    with pytest.raises(script.CallFailed) as caught:
        await run.call("estimate", "olmoearth_estimate_map_error", {})
    envelope = caught.value.call.envelope
    assert envelope["ok"] is False
    # The message says why, so a log that shows only the exception does too.
    tool = "olmoearth_estimate_map_error"
    assert str(caught.value) == f"{tool} did not succeed: {envelope['error']}"
    # The failed call is on disk all the same.
    log = json.loads((tmp_path / script.ENVELOPES).read_text(encoding="utf-8"))
    assert [c["envelope"]["ok"] for c in log["calls"]] == [False]
    assert run.calls == [caught.value.call]
    assert len(run.state.provenance.entries) == 1
    with pytest.raises(script.DemoError, match="default tools"):
        await run.call("grade", "olmoearth_grade_review_rule", {})
    assert len(run.calls) == 1


def test_the_command_prints_a_failed_calls_envelope(
    script: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The command ends with status 1 and the failed call's envelope, and
    writes no page; a work folder already used is refused with status 2."""
    envelope = {
        "ok": False,
        "error": "ValueError: a budget of 300 cannot give each map class its 50",
        "tool": "olmoearth_plan_label_sample",
    }
    failed = script.Call(
        step="class_plan",
        tool="olmoearth_plan_label_sample",
        arguments={"budget": 300},
        envelope=envelope,
    )

    def stop(*_args: Any, **_kwargs: Any) -> Any:
        raise script.CallFailed(failed)

    monkeypatch.setattr(script, "generate", stop)
    monkeypatch.setattr(script, "load_tile", lambda: None)
    docs = tmp_path / "docs"
    assert script.main(["--work", str(tmp_path / "work"), "--docs", str(docs)]) == 1
    head, body = capsys.readouterr().err.split("\n", 1)
    assert head == "olmoearth_plan_label_sample did not succeed; its envelope:"
    assert json.loads(body) == envelope
    assert not docs.exists()

    used = tmp_path / "used"
    used.mkdir()
    (used / "notes.txt").write_text("mine", encoding="utf-8")
    assert script.main(["--work", str(used), "--docs", str(docs)]) == 2
    assert "not a new or empty folder" in capsys.readouterr().err


@pytest.mark.asyncio
async def test_no_certified_zone_skips_both_checks_and_says_so(
    script: ModuleType,
) -> None:
    """With no certificate written, neither check runs; the page says so, and
    the truth table's zone row has nothing to check."""
    note = await script.check_certificate(
        None, {"certificate_written": False}, Path("map.tif"), Path("work")
    )
    assert note.startswith("No zone was certified")
    row = script._zone_row({"zone": {"call": 5, "certified": False}})
    assert row == "| Certified zone (call 5) | no zone certified | - | - |"
    # The calls table's lead does not describe a check that did not run.
    demo = _demo(script, Path("work"))
    demo.tamper = note
    assert script.calls_lead(demo) == script.CALLS_LEAD


def test_the_class_budget_is_fixed_from_the_map_alone(
    script: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The class design plans 300, or the package's floor when the map needs
    more, counted from the scores file's map classes before any label; the
    page says why when it is raised, and no option changes it."""
    monkeypatch.setenv(SCORES_ROOT_ENV, str(tmp_path))

    def scores_file(name: str, sizes: list[int]) -> str:
        classes = [c for c, n in enumerate(sizes) for _ in range(n)]
        payload = {"scores": [[0.5, 0.0]] * len(classes), "map_class": classes}
        path = tmp_path / name
        path.write_text(json.dumps(payload), encoding="utf-8")
        return str(path)

    assert script.inferencex.load("estimate").CLASS_FLOOR == 50
    few = script.class_design_budget(scores_file("few.json", [10, 60, 400]))
    assert few == {"planned": 300, "per_class": 50, "floor": 110, "budget": 300}
    # The real tile's map classes over all its 16,380 windows (the record's
    # exp119, aicr job 1554014, where a class sample of 300 was refused).
    sizes = [10, 440, 86, 0, 5256, 512, 2694, 7328, 54]
    many = script.class_design_budget(scores_file("many.json", sizes))
    assert many == {"planned": 300, "per_class": 50, "floor": 360, "budget": 360}
    with pytest.raises(script.DemoError, match="map_class"):
        script.class_design_budget(scores_file("none.json", []))

    demo = _demo(script, tmp_path)
    demo.class_budget = many
    call = script.Call(
        step="class_plan",
        tool=script.PLAN,
        arguments={"design": "class", "budget": 360, "seed": 0},
        envelope={},
    )
    text = script._args_text(demo, call)
    assert text.startswith("design 'class', budget 360, seed 0 (300 planned, raised")
    assert "min(windows, 50) summed over the map's classes: 360" in text
    with pytest.raises(SystemExit):
        script.main(["--class-budget", "400"])


def test_the_population_and_the_sheets_are_held_to_the_tile(
    script: ModuleType, tile: Any, tmp_path: Path
) -> None:
    """A scores file of another population, a sheet whose map class or expert
    class is not the tile's (the sheet left as it was), or a review set that
    lists fewer windows than it counts stops the run."""
    script.check_population(tile, {"grid": [SIDE, SIDE], "n_valid": 555})
    with pytest.raises(script.DemoError, match="555"):
        script.check_population(tile, {"grid": [SIDE, SIDE], "n_valid": 554})
    with pytest.raises(script.DemoError):
        script.check_population(tile, {"grid": [SIDE, SIDE + 1], "n_valid": 555})

    mapped = tile.map_class()
    cases = {
        "other_class.csv": (10 * SIDE + 10, (int(mapped[10, 10]) + 1) % 3),
        "unannotated.csv": (0, int(mapped[0, 0])),  # the expert marked nothing
    }
    header = ["order", "window_index", "map_class", "reference_class", "wrong"]
    for name, (window, shown) in cases.items():
        sheet = tmp_path / name
        with open(sheet, "w", encoding="utf-8", newline="") as fh:
            csv.writer(fh).writerows([header, [0, window, shown, "", ""]])
        before = sheet.read_bytes()
        plan = {"labels_csv_path": str(sheet), "design_path": str(tmp_path / "d.json")}
        with pytest.raises(script.DemoError, match="not the tile's"):
            script.label_plan(tile, plan)
        assert sheet.read_bytes() == before

    listed = script.review_windows({"review": [{"window_index": 1}], "n_review": 1})
    assert listed == [1]
    with pytest.raises(script.DemoError, match="lists 1 windows of 2"):
        script.review_windows({"review": [{"window_index": 1}], "n_review": 2})


def test_a_zone_outside_the_population_stops_the_run(
    script: ModuleType, tile: Any, tmp_path: Path
) -> None:
    """The zone's windows are held to the population, and its delta is read
    from the zone file."""
    truth = script.truth_of(tile)

    def zone_of(windows: list[int]) -> Any:
        path = tmp_path / f"zone_{len(windows)}.json"
        stored = {"alpha": 0.1, "delta": 0.05, "window_indices": windows}
        path.write_text(json.dumps(stored), encoding="utf-8")
        result = _ok({"certified": True, "zone_path": str(path)})
        return _demo(script, tmp_path, certify=(script.CERTIFY, result))

    kept = 10 * SIDE + 10
    zone = script.zone_truth(truth, zone_of([kept]))
    assert (zone["n_zone"], zone["alpha"], zone["delta"]) == (1, 0.1, 0.05)
    with pytest.raises(script.DemoError, match="outside the population"):
        script.zone_truth(truth, zone_of([kept, 0]))  # window 0: not annotated


class _Checks:
    """A stand-in for the run in check_certificate: each check's answer."""

    def __init__(
        self, script: ModuleType, verified: bool, tamper: dict[str, Any]
    ) -> None:
        self.script, self.verified, self.tamper = script, verified, tamper
        self.steps: list[str] = []

    async def call(self, step: str, tool: str, args: dict[str, Any]) -> Any:
        """The check against the copy of the map, which succeeded."""
        self.steps.append(step)
        failed = [] if self.verified else ["map_sha256"]
        return {"available": True, "verified": self.verified, "failed_checks": failed}

    async def dispatch(self, step: str, tool: str, args: dict[str, Any]) -> Any:
        """The check of the tampered copy, answered with ``tamper``."""
        self.steps.append(step)
        return self.script.Call(
            step=step, tool=tool, arguments=args, envelope=self.tamper
        )


def _bundle(tmp_path: Path) -> tuple[dict[str, Any], Path, Path]:
    """A written certificate's result, a map and a new work folder."""
    bundle = tmp_path / "bundle"
    bundle.mkdir(parents=True)
    (bundle / "certificate.json").write_text("{}", encoding="utf-8")
    (bundle / "labels.csv").write_text("window_index,wrong\n5,0\n", encoding="utf-8")
    raster = tmp_path / "scores.tif"
    raster.write_bytes(b"map")
    work = tmp_path / "work"
    work.mkdir()
    certificate = str(bundle / "certificate.json")
    return {"certificate_written": True, "certificate_path": certificate}, raster, work


@pytest.mark.asyncio
async def test_each_certificate_check_must_go_the_right_way(
    script: ModuleType, tmp_path: Path
) -> None:
    """The certificate must verify against the copy of the map and its
    tampered copy must not; a refusal of the tampered copy is the page's
    note, and a check that goes the wrong way stops the run."""
    not_verified = _ok({"verified": False})
    cases = {
        "honest": (False, not_verified),
        "refused": (True, {"ok": False, "error": "ValueError: no map"}),
        "tampered": (True, _ok({"verified": True})),
        "right": (True, not_verified),
    }
    outcome: dict[str, Any] = {}
    for name, (verified, tamper) in cases.items():
        written, raster, work = _bundle(tmp_path / name)
        run = _Checks(script, verified, tamper)
        try:
            outcome[name] = await script.check_certificate(run, written, raster, work)
        except script.DemoError as exc:
            outcome[name] = exc
        if name != "honest":
            # The copy's first label is flipped; nothing else changes.
            labels = work / script.TAMPERED / "labels.csv"
            flipped = labels.read_text(encoding="utf-8")
            assert flipped.splitlines() == ["window_index,wrong", "5,1"]
            assert run.steps == ["verify", "tamper"]
        else:
            assert run.steps == ["verify"]
    assert "did not verify against a copy of the map" in str(outcome["honest"])
    assert "map_sha256" in str(outcome["honest"])
    assert outcome["refused"].startswith(
        "olmoearth_verify_certificate refused the tampered copy"
    )
    assert "ValueError: no map" in outcome["refused"]
    assert "one label flipped verified" in str(outcome["tampered"])
    assert outcome["right"] is None

    # The calls table's lead describes the tampered copy's check only when
    # it ran and was answered.
    demo = _demo(script, tmp_path, tamper=(script.VERIFY, not_verified))
    assert script.calls_lead(demo) == f"{script.CALLS_LEAD} {script.TAMPER_LEAD}"
    refused = _demo(script, tmp_path, tamper=(script.VERIFY, cases["refused"][1]))
    refused.tamper = outcome["refused"]
    assert script.calls_lead(refused) == script.CALLS_LEAD

    bare = tmp_path / "bare"
    bare.mkdir()
    (bare / "labels.csv").write_text("window_index\n5\n", encoding="utf-8")
    with pytest.raises(script.DemoError, match="no 'wrong' column"):
        script.flip_one_label(bare, tmp_path / "bare_copy")


def test_a_tile_or_report_not_as_expected_is_refused(
    script: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A tile name with no date, a report with no accuracy section and a
    package without the tile are refused by name."""
    with pytest.raises(script.DemoError, match="no date"):
        script.tile_date({"tile": "x.tif"})
    with pytest.raises(script.DemoError, match="no date"):
        script.tile_date({})
    with pytest.raises(script.DemoError, match="accuracy section"):
        script.report_section("# A report\n\nNo accuracy here.\n")
    monkeypatch.setattr(script.importlib.resources, "files", lambda _name: tmp_path)
    with pytest.raises(script.DemoError, match="inferencex extra"):
        script.sample_path()
    with pytest.raises(script.DemoError, match="no centre"):
        script.tile_grid({"window_m": 40}, 8, 8)


def test_the_geotiff_is_placed_on_the_tiles_centre(script: ModuleType) -> None:
    """The run's raster is georeferenced, as a direct run's is (the package
    certifies one that is not from 1.20.3, but no direct run writes it): the
    UTM zone of the recorded centre, north up, windows of window_m metres,
    the centre in the middle of the grid."""
    from rasterio.warp import transform as reproject

    meta = {"lon": -71.8352855451905, "lat": -16.409944165598645, "window_m": 40}
    crs, transform = script.tile_grid(meta, 128, 96)
    assert crs == "EPSG:32719"
    assert (transform.a, transform.e, transform.b, transform.d) == (40, -40, 0, 0)
    xs, ys = reproject("EPSG:4326", crs, [meta["lon"]], [meta["lat"]])
    centre_x, centre_y = transform * (96 / 2, 128 / 2)
    assert centre_x == pytest.approx(xs[0]) and centre_y == pytest.approx(ys[0])
    north = script.tile_grid({"lon": 5.0, "lat": 52.0, "window_m": 10}, 4, 4)
    assert north[0] == "EPSG:32631"


def test_no_spelling_of_the_work_folder_survives_json_escaping(
    script: ModuleType, tmp_path: Path
) -> None:
    """The work folder is written <work> in the data before it is
    serialised, so a non-ASCII folder (escaped \\uXXXX) or a Windows one
    (backslashes doubled) leaves no spelling of itself behind."""
    work = tmp_path / "wörk"
    sheet = str(work / "sheet.csv")
    verification = [{"labels_path": sheet}]
    demo = _demo(
        script,
        work,
        plan=(script.PLAN, _ok({"labels_csv_path": sheet})),
        provenance=(script.PROVENANCE, _ok({"verification_calls": verification})),
    )
    document = script.envelopes_document(demo, {"population": {}})
    block = script.json_text(verification, work, indent=1)
    for text in (document, block):
        assert str(work) not in text
        assert json.dumps(str(work))[1:-1] not in text  # the escaped spelling
        assert f"{script.WORK}/sheet.csv" in text
    windows = Path("C:\\Users\\x\\work")
    text = script.json_text({"path": "C:\\Users\\x\\work\\sheet.csv"}, windows)
    assert "Users" not in text
    assert json.loads(text) == {"path": f"{script.WORK}\\sheet.csv"}


def test_without_the_extra_the_command_says_what_to_install(
    script: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """With the package missing, or older than the agent needs, the command
    prints the line that installs it and exits 2, before any folder or page
    is made."""
    real = importlib.util.find_spec

    def find_spec(name: str, package: str | None = None) -> Any:
        return None if name == "oe_inferencex" else real(name, package)

    monkeypatch.setattr(importlib.util, "find_spec", find_spec)
    work, docs = tmp_path / "work", tmp_path / "docs"
    assert script.main(["--work", str(work), "--docs", str(docs)]) == 2
    err = capsys.readouterr().err
    assert "oe_inferencex not installed" in err
    assert f"olmoearth-inferencex >= {script.inferencex.MIN_VERSION}" in err
    assert "uv sync --all-extras" in err and "3.11 to 3.13" in err
    assert not work.exists() and not docs.exists()

    monkeypatch.setattr(importlib.util, "find_spec", real)
    monkeypatch.setattr(script.inferencex, "older_version", lambda: "1.19.0")
    assert script.main(["--work", str(work), "--docs", str(docs)]) == 2
    assert "olmoearth-inferencex 1.19.0 installed" in capsys.readouterr().err
    assert not work.exists() and not docs.exists()


def test_the_command_stops_cleanly_whatever_stops_the_run(
    script: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A docs path that is a file is refused with status 2 before the run; a
    check that goes the wrong way, or a page that cannot be built, ends with
    status 1, a message and where the calls' envelopes are."""
    called: list[Any] = []
    monkeypatch.setattr(script, "generate", lambda *a, **k: called.append(a))
    monkeypatch.setattr(script, "load_tile", lambda: None)
    page = tmp_path / "page.md"
    page.write_text("mine", encoding="utf-8")
    first = tmp_path / "first"
    assert script.main(["--work", str(first), "--docs", str(page)]) == 2
    assert "is not a folder" in capsys.readouterr().err
    assert not called and not first.exists()

    def stop(_tile: Any, work: Path, _docs: Path, **_options: Any) -> Any:
        (work / script.ENVELOPES).write_text('{"calls": []}\n', encoding="utf-8")
        raise script.DemoError("a copy of the certificate with one label flipped")

    monkeypatch.setattr(script, "generate", stop)
    work = tmp_path / "work"
    docs = tmp_path / "docs"
    assert script.main(["--work", str(work), "--docs", str(docs)]) == 1
    err = capsys.readouterr().err
    assert err.startswith("verify_demo_tile: a copy of the certificate")
    assert f"envelopes are in {work.resolve() / script.ENVELOPES}" in err
    assert not docs.exists()

    def broken(*_args: Any, **_kwargs: Any) -> Any:
        raise OSError("disk full")

    monkeypatch.setattr(script, "generate", broken)
    again = tmp_path / "again"
    assert script.main(["--work", str(again), "--docs", str(docs)]) == 1
    err = capsys.readouterr().err
    assert err.startswith("verify_demo_tile: the page could not be built")
    assert "OSError('disk full')" in err
    assert not docs.exists()
