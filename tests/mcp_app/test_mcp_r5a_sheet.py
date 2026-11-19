# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Round r5-a (17 Nov 2026), track SHEET, as specs/r5-a-round.md gives it:
pictures that fail or are missing, and the finish message. The views are read
as text, as views.view_html() serves them (no test here runs the scripts),
and the server's words through its own functions. No test touches the
network: the search and the reads are replaced.

- L1: a picture call waits 200 s (the server can take 160 s); a call that
  throws shows one fixed sentence, never the thrown error's text, with "Try
  again" in the pane, which repeats the call for the same square; the
  server's own sentences another try can get past (a slow fetch, the
  catalogue, the image files) carry ``again`` and get the button too; any
  other refusal carries none.
- L2: when no month of the map's period has a scene with data on the square,
  the answer carries ``no_image`` and the pane says one fixed line in place of
  a list of empty months, with " Press Cannot tell." on a face that takes
  Cannot tell. A search cut short is told by the record alone.
- L3: the finish message starts in plain words ("I have answered every square
  of this sheet. Estimate the error rate."; every point on a sheet of points;
  a two-map sheet and a certificate plan's keep their own verb), then the
  lines Claude needs, as they were.
- L4: the Check card says, under the press, that a second press opens the
  same squares; the line is hidden while the card shows a part as picked, or
  as pressed with no reply.
"""

from __future__ import annotations

import csv
import inspect
import json
import re
from pathlib import Path
from typing import Any

import httpx
import pytest

from olmoearth_agent.mcp_app import imagery, review, s2_fetch, sheet, views

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
#: A sheet with no place: no ``lon`` and no ``lat`` column.
NO_PLACE = [
    "order",
    "window_index",
    "row",
    "col",
    "map_class",
    "wrong",
    "reference_class",
]
TWO_MAP = [
    "order",
    "window_index",
    "row",
    "col",
    "class_a",
    "class_b",
    "reference_class",
]
P = 11
GRID = (6, 10)
#: The other imagery tests' grid (tests/mcp_app/test_mcp_imagery.py): 10 m
#: pixels on a UTM zone; a square's place is worked out from it, never written.
MAP_EPSG = 32611
MAP_ORIGIN = (500000, 3700000)

#: L1: the pane's fixed sentence and its button (sheet.js).
NOT_ARRIVED = "The pictures did not arrive. Press Try again."
TRY_AGAIN = "Try again"
#: L1: a picture call in a view's script, up to the end of its statement.
PICTURE_CALL = re.compile(
    r"OE\.call\(app, 'olmoearth_app_(?:window|review)_image',[^;]*;"
)
#: L2: what a face with Cannot tell adds to the pane's one line.
NO_IMAGE_PRESS = " Press Cannot tell."
#: L4: the Check card's line under the press.
PICKED_BEFORE = "Picked before? Press again: the same squares open."


@pytest.fixture(autouse=True)
def _no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*_a: Any, **_k: Any) -> Any:
        raise AssertionError("a test reached for the network")

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", refuse)
    s2_fetch.clear_cache()
    imagery._maps.clear()
    imagery._test_cache.clear()


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    out = tmp_path / "outputs"
    out.mkdir()
    monkeypatch.setenv("OLMOEARTH_OUTPUT_ROOT", str(out))
    monkeypatch.delenv("OLMOEARTH_SCORES_ROOT", raising=False)
    return out


def _read(name: str) -> str:
    """A file of views/, as it is on disk."""
    return (views.VIEW_DIR / name).read_text(encoding="utf-8")


def _between(text: str, start: str, end: str) -> str:
    """The part of text from start up to the first end after it."""
    at = text.index(start)
    return text[at : text.index(end, at)]


def _write(path: Path, header: list[str], rows: list[list[str]]) -> Path:
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)
    return path


def _centres(indexes: list[int]) -> list[tuple[float, float]]:
    """Each window's centre on the test grid, as lon and lat, worked out."""
    affine = pytest.importorskip("affine")
    warp = pytest.importorskip("rasterio.warp")
    transform = affine.Affine(10, 0, MAP_ORIGIN[0], 0, -10, MAP_ORIGIN[1])
    xs, ys = [], []
    for index in indexes:
        r, c = divmod(index, GRID[1])
        x, y = transform * (c * P + P / 2, r * P + P / 2)
        xs.append(x)
        ys.append(y)
    lons, lats = warp.transform(f"EPSG:{MAP_EPSG}", "EPSG:4326", xs, ys)
    return list(zip(lons, lats))


def _kelp_sheet(root: Path, *, end: str = "2024-06-30T23:59:59.999000Z") -> Path:
    """A plan's sheet of two squares (windows 14 and 25) on a scores map of
    presence and absence, with its design, its scores file and its map raster;
    the map's period runs from 1 Apr 2024 to ``end``."""
    rasterio = pytest.importorskip("rasterio")
    np = pytest.importorskip("numpy")
    affine = pytest.importorskip("affine")
    folder = root / "run"
    folder.mkdir(exist_ok=True)
    raster = folder / "scores.tif"
    h, w = GRID[0] * P, GRID[1] * P
    with rasterio.open(
        raster,
        "w",
        driver="GTiff",
        width=w,
        height=h,
        count=2,
        dtype="float32",
        crs=f"EPSG:{MAP_EPSG}",
        transform=affine.Affine(10, 0, MAP_ORIGIN[0], 0, -10, MAP_ORIGIN[1]),
    ) as dst:
        dst.write(np.zeros((2, h, w), dtype=np.float32))
    scores = root / "scores_run_p11.json"
    scores.write_text(
        json.dumps(
            {
                "scores": [[0.2, 0.8]] * (GRID[0] * GRID[1]),
                "grid": list(GRID),
                "patch_px": P,
                "raster": {
                    "file": raster.name,
                    "path": str(raster),
                    "shape": [2, h, w],
                    "sha256": "abc",
                },
                "date_window": {"start": "2024-04-01T00:00:00Z", "end": end},
            }
        ),
        encoding="utf-8",
    )
    population = {
        "score_kind": "window_confidence",
        "class_names": {"0": "absent", "1": "present"},
        "source": {"scores_path": str(scores)},
    }
    (root / "kelp.json").write_text(
        json.dumps({"design": "random", "population": population}), encoding="utf-8"
    )
    rows = []
    for order, (index, centre) in enumerate(zip((14, 25), _centres([14, 25])), 1):
        r, c = divmod(index, GRID[1])
        lon, lat = (f"{value:.7f}" for value in centre)
        rows.append(
            [str(order), str(index), str(r), str(c), "0", lon, lat, "0", "", ""]
        )
    return _write(root / "kelp_to_label.csv", ONE_MAP, rows)


def _pictures(path: Path, index: int) -> dict[str, Any]:
    """window_image of the window ``index``, asked by its handle as the page asks."""
    sheet_id, handles = sheet.window_handles(str(path))
    handle = next(h for h, i in handles.items() if i == index)
    return imagery.window_image(str(path), handle, sheet_id=sheet_id, on=True)


def _record(path: Path) -> list[dict[str, Any]]:
    """The lines of the pictures' record beside a sheet."""
    text = imagery.record_path(path).read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines()]


def _review_key(root: Path) -> str:
    """A review set ranked from the kelp map's scores file (one centre per
    row, worked out), with window 14 listed first: its key."""
    _kelp_sheet(root)
    scores = root / "scores_run_p11.json"
    payload = json.loads(scores.read_text(encoding="utf-8"))
    every = list(range(GRID[0] * GRID[1]))
    payload["centres_lon_lat"] = [list(centre) for centre in _centres(every)]
    scores.write_text(json.dumps(payload), encoding="utf-8")
    listed = {
        "rank": 1,
        "window_index": 14,
        "row": 1,
        "col": 4,
        "margin": 0.1,
        "predicted_class": 1,
        "class_name": "present",
    }
    block = review.review_view(
        "olmoearth_review_set",
        {"n_review": 1, "review": [listed]},
        {"scores_path": str(scores)},
    )
    assert block is not None and block["pictures"] is True
    return str(block["key"])


def _fails(monkeypatch: pytest.MonkeyPatch, kind: str) -> None:
    """The search fails with a fetch error of this kind."""

    def fail(*_a: Any, **_k: Any) -> Any:
        raise s2_fetch.FetchError(kind)

    monkeypatch.setattr(imagery, "_scenes", fail)


def _no_scenes(monkeypatch: pytest.MonkeyPatch) -> None:
    """The search finds no scene in any month and the reads return none, so a
    picture call runs to its end (and its record) without a fetch."""
    monkeypatch.setattr(
        imagery,
        "_scenes",
        lambda _real, _extent, months, _client, _budget: ([[] for _ in months], []),
    )
    monkeypatch.setattr(
        s2_fetch,
        "dated",
        lambda _fetch, scenes, _box, _square: [(None, None, 0) for _ in scenes],
    )


class _Busy:
    """A turn, or a sheet's search, that never comes free."""

    def acquire(self, timeout: float | None = None) -> bool:
        return False

    def release(self) -> None:
        return None


# ------------------------------------------------- L1: the server says again


def test_l1_only_a_slow_fetch_the_catalogue_and_the_image_files_say_again() -> None:
    assert imagery.RETRY_SENTENCES == {
        imagery.TOO_SLOW,
        imagery.CATALOGUE_FAILED,
        imagery.FETCH_FAILED,
        imagery.REVIEW_FETCH_FAILED,
    }
    for said in imagery.RETRY_SENTENCES:
        assert imagery._failed(said) == {"ok": False, "error": said, "again": True}
    # Every sentence a fetch's failure can pick is one another try can get past.
    for table in (imagery._FETCH_SENTENCE, imagery._REVIEW_FETCH_SENTENCE):
        assert set(table.values()) <= imagery.RETRY_SENTENCES
    # A refusal that would be the same on a second try carries no again key.
    for said in (
        imagery.LONG_PERIOD,
        imagery.NO_PERIOD,
        imagery.NO_FOOTPRINT,
        imagery.MISMATCH,
        imagery.CHANGED,
        imagery.NO_RECORD,
        imagery.UNEXPECTED,
    ):
        assert imagery._failed(said) == {"ok": False, "error": said}


@pytest.mark.parametrize(
    ("kind", "sentence"),
    [
        ("catalogue", "CATALOGUE_FAILED"),
        ("fetch", "FETCH_FAILED"),
        ("budget", "FETCH_FAILED"),
        ("deadline", "TOO_SLOW"),
    ],
)
def test_l1_a_fetch_that_fails_on_a_sheets_square_says_again(
    root: Path, monkeypatch: pytest.MonkeyPatch, kind: str, sentence: str
) -> None:
    path = _kelp_sheet(root)
    _fails(monkeypatch, kind)
    assert _pictures(path, 14) == {
        "ok": False,
        "error": getattr(imagery, sentence),
        "again": True,
    }


def test_l1_a_call_that_waits_too_long_for_its_turn_says_again(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two picture calls run at a time, and one search per sheet: a call
    whose turn, or whose sheet's search, does not come says the slow-fetch
    sentence, which another try can get past."""
    path = _kelp_sheet(root)
    slow = {"ok": False, "error": imagery.TOO_SLOW, "again": True}
    with monkeypatch.context() as patch:
        patch.setattr(imagery, "_calls", _Busy())
        assert _pictures(path, 14) == slow
    with monkeypatch.context() as patch:
        patch.setattr(imagery, "_lock_for", lambda _real: _Busy())
        assert _pictures(path, 14) == slow


def test_l1_a_refusal_a_second_try_cannot_change_has_no_again(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _kelp_sheet(root)
    sheet_id, _ = sheet.window_handles(str(path))
    other = "w" + "a" * 16
    assert imagery.window_image(str(path), other, sheet_id=sheet_id, on=True) == {
        "ok": False,
        "error": imagery.NOT_ON_SHEET,
    }
    # A period of more than 12 months: refused before any fetch.
    long = _kelp_sheet(root, end="2025-09-30T23:59:59.999000Z")
    imagery._maps.clear()
    assert _pictures(long, 14) == {"ok": False, "error": imagery.LONG_PERIOD}

    # An error nobody worded: the fixed sentence (logged), no again.
    def broke(*_a: Any, **_k: Any) -> Any:
        raise RuntimeError("an internal detail")

    monkeypatch.setattr(imagery, "_window_image", broke)
    assert _pictures(long, 14) == {"ok": False, "error": imagery.UNEXPECTED}


def test_l1_a_review_set_and_the_test_scene_say_again(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    key = _review_key(root)
    _fails(monkeypatch, "budget")
    assert imagery.review_image(key, 1, on=True) == {
        "ok": False,
        "error": imagery.REVIEW_FETCH_FAILED,
        "again": True,
    }
    _fails(monkeypatch, "deadline")
    for asked in (True, False):
        imagery._test_cache.clear()
        assert imagery.public_test_scene(review=asked) == {
            "ok": False,
            "error": imagery.TOO_SLOW,
            "again": True,
        }
    # A rank that is not listed would not be listed on a second try either.
    assert imagery.review_image(key, 7, on=True) == {
        "ok": False,
        "error": review.NOT_LISTED,
    }


# --------------------------------------------------- L1: the pane, as text


def test_l1_the_picture_calls_wait_as_long_as_the_server_can_take() -> None:
    script = _read("sheet.js")
    found = re.search(r"^OE\.PICTURES_WAIT_MS = (\d+);$", script, re.MULTILINE)
    assert found is not None
    wait_ms = int(found.group(1))
    assert wait_ms == 200_000
    # The server: a wait for its turn, one for the sheet's search, then its
    # own search and its reads, each with a budget of its own.
    budget = inspect.signature(s2_fetch.Budget).parameters["seconds"].default
    worst = 2 * imagery.WAIT_SECONDS + 2 * budget
    assert worst == 160 and wait_ms >= worst * 1000
    # OE.call hands its options to the client as they are.
    call = _between(_read("common.js"), "OE.call = async (", "\n};")
    assert "args = {}, options = undefined) => {" in call
    assert "await app.callServerTool({ name, arguments: args }, options);" in call
    # The sheet's pane and its test scene, and a review set's pane and its
    # test scene: two picture calls each, and each carries the wait.
    editor = _between(script, "OE.sheetEditor = (app, data,", "\n};")
    for text in (editor, _read("result.html")):
        calls = PICTURE_CALL.findall(text)
        assert len(calls) == 2, calls
        for asked in calls:
            assert asked.endswith("{ timeout: OE.PICTURES_WAIT_MS });"), asked


def test_l1_a_thrown_error_is_never_printed_in_the_pane() -> None:
    script = _read("sheet.js")
    for line in (
        f"// WORDING: r5a L1\nOE.PICTURES_NOT_ARRIVED = '{NOT_ARRIVED}';",
        f"// WORDING: r5a L1\nOE.TRY_AGAIN = '{TRY_AGAIN}';",
        "OE.picturesFailed = () => "
        "({ ok: false, error: OE.PICTURES_NOT_ARRIVED, again: true });",
    ):
        assert line in script, line
    editor = _between(script, "OE.sheetEditor = (app, data,", "\n};")
    page = _read("result.html")
    caught = "} catch (_) {\n      answer = OE.picturesFailed();\n    }"
    functions = (
        # r5-f F01: reveal, whether a press of Pictures asked for it.
        (editor, "  async function loadPictures(h, reveal = false) {"),
        (editor, "  async function loadTest() {"),
        (page, "  async function load(i) {"),
        (page, "  async function loadTest() {"),
    )
    for text, start in functions:
        body = _between(text, start, "\n  }\n")
        assert caught in body, start
        # The thrown error has no name there, so its text cannot be shown.
        assert "err" not in re.findall(r"[A-Za-z_]+", body), start
    # The old line, which printed the thrown error's own text, is gone.
    for text in (script, page):
        assert "error: err.message" not in text


def test_l1_try_again_repeats_the_call_for_the_same_square() -> None:
    editor = _between(_read("sheet.js"), "OE.sheetEditor = (app, data,", "\n};")
    # The sheet's pane: the button only where another try can get past the
    # failure (the answer's again), and it asks for the same window.
    pane = _between(editor, "  function renderPreview(shown) {", "\n  }\n")
    failed = _between(pane, "    if (!answer.ok) {", "\n    }\n")
    assert "answer.error || 'No pictures.'" in failed
    assert "answer.again ? OE.el('div', { class: 'row' }," in failed
    assert "onclick: () => retryPictures(h) },\n            OE.TRY_AGAIN)" in failed
    retry = _between(editor, "  async function retryPictures(h) {", "\n  }\n")
    assert "const asked = loadPictures(h);" in retry
    # A failure is never kept as an answer: the next ask is a new call. It
    # stays in the pane only while its window is shown.
    load = _between(
        editor, "  async function loadPictures(h, reveal = false) {", "\n  }\n"
    )
    assert "state.images.delete(h);" in load
    assert "state.failed = { h, answer: answer || {} };" in load
    # The pane reads the kept failure when it is drawn again in place (else
    # it would say "Loading the pictures..." for ever).
    drop = "if (state.failed && state.failed.h !== h) state.failed = null;"
    read = (
        "state.images.get(h) || (state.failed && state.failed.answer) "
        "|| { loading: true };"
    )
    assert drop in pane and read in pane
    # Another square's failure is dropped before the pane reads it.
    assert pane.index(drop) < pane.index(read)
    # The sheet's test scene: the same button, the test scene's call again.
    test = _between(editor, "  function renderTest() {", "\n  }\n")
    assert "answer.again ? OE.el('div', { class: 'row' }," in test
    assert "onclick: () => loadTest() },\n            OE.TRY_AGAIN)" in test
    # A review set's pane: the button only where another try can get past the
    # failure, for the square and for its test scene.
    page = _read("result.html")
    assert (
        "if (answer.again) parts.push(OE.el('div', { class: 'row' }, "
        "button('again', REVIEW_WORDS.again, () => load(i))));"
    ) in page
    assert (
        "if (t.again) parts.push(OE.el('div', { class: 'row' }, "
        "button('again', REVIEW_WORDS.again, loadTest)));"
    ) in page
    assert f"    again: '{TRY_AGAIN}',\n" in page


# --------------------------------------------------- L2: no image was found


def test_l2_no_scene_in_any_month_is_said_once(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _kelp_sheet(root)
    _no_scenes(monkeypatch)
    out = _pictures(path, 14)
    assert out["ok"] is True and out["no_image"] is True, out
    # The months still come, empty, for a panel drawn before this round.
    assert [d["month"] for d in out["dates"]] == ["Apr 2024", "May 2024", "Jun 2024"]
    for date in out["dates"]:
        assert date["pictures"] == []
        assert date["label"] == f"{date['month']}: no scene with data here"
    # The record says what was served: no scene in any month.
    served = [line["served"] for line in _record(path) if "served" in line]
    assert len(served) == 1
    assert [m["scene"] for m in served[0]["months"]] == [None, None, None]


def test_l2_a_month_with_a_scene_has_no_flag(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert imagery._no_image([(None, None, 3), (None, None, 0)]) is True
    assert imagery._no_image([(None, None, 3), (object(), object(), 1)]) is False
    path = _kelp_sheet(root)
    _no_scenes(monkeypatch)

    def one_scene(_fetch: Any, scenes: Any, _box: Any, _square: Any) -> Any:
        return [(object(), object(), 1)] + [(None, None, 0) for _ in scenes[1:]]

    def plain_dates(months: Any, _results: Any, _square: Any) -> Any:
        dates = [{"month": m[0], "label": m[0], "pictures": []} for m in months]
        return dates, [{"month": m[0]} for m in months]

    monkeypatch.setattr(s2_fetch, "dated", one_scene)
    monkeypatch.setattr(imagery, "_dates", plain_dates)
    out = _pictures(path, 14)
    assert out["ok"] is True and "no_image" not in out, out


def test_l2_a_search_cut_short_stays_in_the_record_only(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The catalogue held more scenes than one search takes: the record's
    search line keeps saying so (complete: false), and the pane's answer
    says only that no image was found."""
    path = _kelp_sheet(root)
    asked: list[Any] = []

    def cut(_client: Any, _budget: Any, _bbox: Any, lo: Any, _hi: Any) -> Any:
        asked.append(lo)
        query = {"body": {}, "pages": 5, "returned": 500, "kept": 0, "matched": 900}
        return [], {**query, "complete": False}

    monkeypatch.setattr(s2_fetch, "search_month", cut)
    monkeypatch.setattr(
        s2_fetch,
        "dated",
        lambda _fetch, scenes, _box, _square: [(None, None, 0) for _ in scenes],
    )
    out = _pictures(path, 14)
    assert out["ok"] is True and out["no_image"] is True, out
    assert set(out) == {
        "ok",
        "handle",
        "synthetic",
        "reading_rule",
        "rule_scope",
        "rule_source",
        # No rule_limit since r5-l L1.
        "label_is",
        "caption",
        "dates",
        "pictures",
        "no_image",
    }
    assert "complete" not in json.dumps(out)
    searches = [line["search"] for line in _record(path) if "search" in line]
    assert len(searches) == 1 and len(searches[0]["months"]) == 3
    assert [m["query"]["complete"] for m in searches[0]["months"]] == [False] * 3
    # The stored search is used again for the sheet's other square.
    assert _pictures(path, 25)["no_image"] is True
    assert len(asked) == 3


def test_l2_a_review_sets_square_is_flagged_too(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    key = _review_key(root)
    _no_scenes(monkeypatch)
    out = imagery.review_image(key, 1, on=True)
    assert out["ok"] is True and out["no_image"] is True, out
    assert out["reading_rule"] is None and out["label_is"] is None
    # The test scene is of no map: it never carries the flag, also with no
    # scene in any of its three months, beside a sheet or a review set.
    for beside in (False, True):
        imagery._test_cache.clear()
        scene = imagery.public_test_scene(review=beside)
        assert scene["ok"] is True and scene["test_scene"] is True, scene
        assert [d["pictures"] for d in scene["dates"]] == [[], [], []]
        assert "no_image" not in scene


def test_l2_the_pane_says_one_line_in_place_of_the_empty_months() -> None:
    script = _read("sheet.js")
    assert (
        "// WORDING: r5a L2\n"
        "OE.NO_IMAGE = (unit) => 'No satellite image was found for this ' "
        "+ (unit === 'point' ? 'point' : 'square')\n"
        "  + ' in the map\\'s period.';"
    ) in script
    assert f"// WORDING: r5a L2\nOE.NO_IMAGE_PRESS = '{NO_IMAGE_PRESS}';" in script
    words = _between(script, "OE.sheetWords = (unit) => {", "\n};")
    assert "    // WORDING: r5a L2\n    noImage: OE.NO_IMAGE(one),\n" in words
    # The sheet's pane: the line before any date is drawn, with what to
    # press only on a face that takes Cannot tell; no list of months there.
    editor = _between(script, "OE.sheetEditor = (app, data,", "\n};")
    pane = _between(editor, "  function renderPreview(shown) {", "\n  }\n")
    branch = _between(pane, "    if (answer.no_image) {", "\n    }\n")
    assert "W.noImage + (takesQ ? OE.NO_IMAGE_PRESS : '')" in branch
    assert "datesBlock" not in branch and "return;" in branch
    assert "'Turn off pictures'" in branch
    assert pane.index("if (answer.no_image) {") < pane.index("datesBlock(answer.dates")
    # A review set's pane: the line alone (nothing is labelled there).
    page = _read("result.html")
    assert "    // WORDING: r5a L2\n    noImage: OE.NO_IMAGE(one),\n" in page
    assert (
        "if (answer.no_image) parts.push(OE.el('div', { class: 'small oe-no-image' }, "
        "REVIEW_WORDS.noImage));"
    ) in page
    assert "NO_IMAGE_PRESS" not in page


# --------------------------------------------------- L3: the finish message


def test_l3_the_finish_message_starts_in_plain_words() -> None:
    assert set(sheet.JOB_LABELS) == set(sheet._JOB_ASKS)
    said = sheet.finish_message
    assert said("estimate", unit="square", call="c", labels_path="p") == (
        "I have answered every square of this sheet. Estimate the error rate. "
        "Estimate the map's error rate from it: c."
    )
    assert said("which_map", unit="square", call="c", labels_path="p") == (
        "I have answered every square of this sheet. Say which map is more "
        "accurate. Say which map is more accurate from it: c."
    )
    assert said("certify", unit="square", call="c", labels_path="p") == (
        "I have answered every square of this sheet. Certify the zone. Certify "
        "the zone from it: c. If a zone is certified, offer to write its "
        "certificate (olmoearth_write_certificate with the same design_path and "
        "labels_path)."
    )
    # No design file beside the sheet: the plain start, then the older line.
    assert said(None, unit="square", call="c", labels_path="p") == (
        "I have answered every square of this sheet. Estimate the map error from "
        "it: labels_path p."
    )
    # A sheet of points says points.
    points = said("estimate", unit="point", call="c", labels_path="p")
    assert points.startswith(
        "I have answered every point of this sheet. Estimate the error rate. "
    )
    for job in ("estimate", "which_map", "certify", None):
        message = said(job, unit="square", call="c", labels_path="p")
        assert "window" not in message and "finished labelling" not in message


def _plan(
    root: Path, name: str, design: dict[str, Any], map_class: str = "water"
) -> Path:
    """A plan's sheet of one square or point, with no place, and its design."""
    (root / f"{name}.json").write_text(json.dumps(design), encoding="utf-8")
    row = ["1", "4", "0", "4", map_class, "", ""]
    return _write(root / f"{name}_to_label.csv", NO_PLACE, [row])


def test_l3_a_sheet_says_its_own_unit_and_keeps_the_lines_claude_needs(
    root: Path,
) -> None:
    classes = {"score_kind": "class_map", "classes": ["water", "trees"]}
    by_result = {**classes, "source": {"result_id": "r1"}}
    squares = _plan(root, "squares", {"design": "random", "population": classes})
    points = _plan(root, "points", {"design": "random", "population": by_result})
    for path, unit in ((squares, "square"), (points, "point")):
        page = sheet.read_sheet(str(path))
        assert page["unit"] == unit and page["job"] == "estimate"
        design = (root / path.name.replace("_to_label.csv", ".json")).resolve()
        call = (
            f"olmoearth_estimate_map_error with design_path {design} and "
            f"labels_path {path.resolve()}"
        )
        assert page["finish_call"] == call
        assert page["finish_message"] == (
            f"I have answered every {unit} of this sheet. Estimate the error rate. "
            f"Estimate the map's error rate from it: {call}."
        )
        # No file name and no tool name before the line Claude needs.
        plain = page["finish_message"].split(" Estimate the map's")[0]
        assert ".csv" not in plain and "olmoearth" not in plain
        # The plan card's read gives the same message.
        progress = sheet.sheet_progress(str(path))
        assert progress["finish_message"] == page["finish_message"]
    # A two-map sheet keeps its own verb, in the same plain shape.
    (root / "pair.json").write_text(
        json.dumps({"design": "disagreement", "classes": ["crop", "water"]}),
        encoding="utf-8",
    )
    row = ["1", "4", "0", "4", "water", "crop", ""]
    pair = sheet.read_sheet(str(_write(root / "pair_to_label.csv", TWO_MAP, [row])))
    assert pair["finish_message"].startswith(
        "I have answered every square of this sheet. Say which map is more "
        "accurate. Say which map is more accurate from it: "
        "olmoearth_estimate_map_error with design_path "
    )
    # A certificate plan's too, and its message still never names the estimate.
    plan = {
        "design": "random",
        "population": {"score_kind": "window_confidence"},
        "certificate": {"seed_rule": "from_map", "alpha": 0.05},
    }
    cert = sheet.read_sheet(str(_plan(root, "cert", plan, map_class="1")))
    assert cert["finish_message"].startswith(
        "I have answered every square of this sheet. Certify the zone. "
        "Certify the zone from it: olmoearth_certify_zone with design_path "
    )
    assert "estimate_map_error" not in cert["finish_message"]


def test_l3_the_wording_is_marked_and_the_old_start_is_gone() -> None:
    source = inspect.getsource(sheet.finish_message)
    marked = (
        "    # WORDING: r5a L3\n"
        '    start = f"I have answered every {unit} of this sheet. "\n'
    )
    assert marked in source
    assert "finished labelling" not in inspect.getsource(sheet)


# ------------------------------------------------------- L4: coming back


def test_l4_the_check_card_says_what_a_second_press_does() -> None:
    page = _read("check.html")
    assert f"// WORDING: r5a L4\nconst PICKED_BEFORE = '{PICKED_BEFORE}';" in page
    # Round r5-e K4 moved the line from under the press into the card's
    # Details, after how the pick is fixed (tests/mcp_app/test_mcp_r5e_check.py):
    # no line is left under the press but the one that says why it is off.
    # It keeps L4's rule there: shown only while the card has a press and
    # shows no part as picked or as pressed with no reply.
    assert "againLine" not in page and "oe-check-again" not in page
    start = "  const pressArea = OE.el('div', { class: 'stack' },"
    lines = _between(page, start, ";\n").split("\n")
    at = lines.index("    OE.el('div', { class: 'row' }, press),")
    assert lines[at + 1] == "    limitLine)"
    fold = _between(page, "  function fillFold() {", "\n  }\n")
    shown = (
        "    const again = drawable().length > 0 && !models.some((m) => "
        "m.kind === 'drawn' || m.kind === 'unknown');\n"
        "    nodes.push(OE.el('p', {}, FIXED));\n"
        "    if (again) nodes.push(OE.el('p', {}, PICKED_BEFORE));\n"
    )
    assert shown in fold
    assert page.count("PICKED_BEFORE") == 2
