# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Round r5-r (20 Nov 2026), track RESULT, as specs/r5-r-round.md gives it.

- R1: the result card's headline (``div.oe-rate``'s ``p.big``) sits in an
  outlined box: a 1.5 px border in the card's text colour, 6 px corners,
  12 px 14 px inside, the paper background; the range figure stays right
  under it, outside the box. The out-of-date rate of a withdrawn result
  keeps its warning words and the box. No other card has it.
- R2: once a labelled sheet's estimate is shown, its card offers "Export the
  labels". The press calls the views' tool ``olmoearth_app_export_labels``
  (``labels_export.export_labels``), which writes next to the sheet
  ``<stem>_labels_export.csv`` (plain CSV, one row per square in the order
  the sheet showed them) and ``<stem>_labels_export.geojson`` (one Polygon
  per square: its window's square on the map grid, in EPSG:4326). Refused
  before every square is answered, before the sheet has its result, once
  its answers changed after that result, and for
  a path that is not the sheet's own inside the output root. The model
  reads that the files were written and their paths, never a place.

The views are read as text (no test here runs the scripts). The browser
harness (scripts/mcp_views_e2e.py) draws the card in
17zr_estimate_export_labels, 17zs_estimate_export_download and
17zt_two_map_export. The places of every fixture are made up from the
window's index (or worked out from the fixture's own made-up grid), never
written here.
"""

from __future__ import annotations

import csv
import json
import os
import stat
from pathlib import Path
from typing import Any

import pytest

mcp = pytest.importorskip("mcp")
from mcp import types  # noqa: E402

from olmoearth_agent.mcp_app import (  # noqa: E402
    auth,
    imagery,
    labels_export,
    server,
    sheet,
    views,
)

BASE = "http://mock-studio/api/v1"

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
TWO_MAP = [
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
#: The fixture map's grid (rows, columns) and its window's side in pixels.
GRID = (6, 10)
P = 11
N = GRID[0] * GRID[1]

#: The two-map fixture's windows: index, map a's class, map b's, the class seen.
PAIRS = [
    (3, "water", "trees", "water"),
    (8, "trees", "water", "Trees"),
    (12, "water", "trees", "?"),
]

#: The box's rule in result.html, whole.
BOX_CSS = (
    "  .oe-rate p.big, .oe-result p.oe-out-of-date {"
    " border: 1.5px solid var(--text, #0a3235); border-radius: 6px;\n"
    "    padding: 12px 14px; background: var(--paper, #ffffff); }\n"
)


@pytest.fixture(autouse=True)
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The output root, the read roots' own, and no key."""
    out = tmp_path / "outputs"
    out.mkdir()
    monkeypatch.setenv("OLMOEARTH_OUTPUT_ROOT", str(out))
    monkeypatch.delenv("OLMOEARTH_SCORES_ROOT", raising=False)
    monkeypatch.delenv(auth.KEY_ENV, raising=False)
    monkeypatch.setenv(auth.KEY_FILE_ENV, str(tmp_path / "cfg" / "studio.env"))
    imagery._maps.clear()
    return out.resolve()


def _read(name: str) -> str:
    """A file of views/, as it is on disk."""
    return (views.VIEW_DIR / name).read_text(encoding="utf-8")


def _between(text: str, start: str, end: str) -> str:
    """The part of text from start up to the first end after it."""
    at = text.index(start)
    return text[at : text.index(end, at)]


def _in_order(text: str, *parts: str) -> bool:
    """Whether the parts stand in text in this order."""
    at = 0
    for part in parts:
        found = text.find(part, at)
        if found < 0:
            return False
        at = found + len(part)
    return True


def _made_up(index: int) -> tuple[str, str]:
    """A window's made-up centre cells, from its index alone."""
    return f"{7 + index / 1000:.3f}", f"{3 + index / 500:.3f}"


def _write(path: Path, header: list[str], rows: list[list[str]]) -> Path:
    """A sheet's file, as a plan writes it."""
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)
    return path


def _one_map_design(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """A named scores map's design: present where the window's index is even."""
    population: dict[str, Any] = {
        "score_kind": "window_confidence",
        "class_names": {"0": "absent", "1": "present"},
        "map_class": [1 - index % 2 for index in range(N)],
    }
    population.update(extra or {})
    return {"design": "random", "population": population}


def _one_map(
    root: Path,
    answers: list[tuple[int, str, str]] | None = None,
    *,
    design: dict[str, Any] | None = None,
    places: dict[int, tuple[str, str]] | None = None,
) -> Path:
    """A named scores map's sheet (labelled with the class seen) and its
    design: by default window 14 seen present (the map agrees), 25 seen
    present (the map says absent) and 36 Cannot tell."""
    if answers is None:
        answers = [(14, "1", "0"), (25, "1", "1"), (36, "?", "?")]
    (root / "kelp.json").write_text(
        json.dumps(design if design is not None else _one_map_design()),
        encoding="utf-8",
    )
    rows = []
    for order, (index, ref, wrong) in enumerate(answers, start=1):
        r, c = divmod(index, GRID[1])
        lon, lat = (places or {}).get(index) or _made_up(index)
        on_map = str(1 - index % 2)
        cells = [str(order), str(index), str(r), str(c), on_map, lon, lat, "0"]
        rows.append([*cells, wrong, ref])
    return _write(root / "kelp_to_label.csv", ONE_MAP, rows)


def _two_map(root: Path) -> Path:
    """A two-map sheet and its design: window 3 seen water (map a's class),
    8 seen trees in other letters (map a's class), 12 Cannot tell."""
    (root / "pair.json").write_text(
        json.dumps({"design": "disagreement", "classes": ["water", "trees"]}),
        encoding="utf-8",
    )
    rows = []
    for order, (index, a, b, ref) in enumerate(PAIRS, start=1):
        r, c = divmod(index, GRID[1])
        lon, lat = _made_up(index)
        rows.append([str(order), str(index), str(r), str(c), a, b, lon, lat, ref])
    return _write(root / "pair_to_label.csv", TWO_MAP, rows)


def _grade(path: Path) -> None:
    """Mark the sheet graded, as an ok estimate does."""
    design = path.with_name(path.name[: -len(sheet.SHEET_SUFFIX)] + ".json")
    version = sheet.sheet_version(str(path))
    sheet.record_grading(str(path), version, design_path=str(design))
    assert sheet.graded_state(str(path)) is not None


def _design_of(path: Path) -> str:
    """The design file beside a sheet."""
    return str(path.with_name(path.name[: -len(sheet.SHEET_SUFFIX)] + ".json"))


def _shown_order(path: Path) -> list[int]:
    """The windows in the order the sheet shows them: by their handles."""
    _, handles = sheet.window_handles(str(path))
    return [handles[handle] for handle in sorted(handles)]


def _csv_rows(path: Path) -> list[list[str]]:
    """A written CSV's rows, header first."""
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.reader(fh))


def _export(path: Path, **kwargs: Any) -> dict[str, Any]:
    """The export of a sheet with its own design file."""
    return labels_export.export_labels(str(path), _design_of(path), **kwargs)


def _app() -> server.AppServer:
    """An app with the agent's own registry and no Studio key."""
    return server.AppServer(None, keys=auth.KeyStore(), base_url=BASE)


async def _call(srv: Any, name: str, arguments: dict[str, Any]) -> types.CallToolResult:
    """One tool call as a client makes it."""
    request = types.CallToolRequest(
        method="tools/call",
        params=types.CallToolRequestParams(name=name, arguments=arguments),
    )
    result = await srv.request_handlers[types.CallToolRequest](request)
    return result.root


def _texts(result: types.CallToolResult) -> list[str]:
    """What the model reads of a result."""
    return [c.text for c in result.content if isinstance(c, types.TextContent)]


# ---------------------------------------------------------------- R2, the files


def test_r2_one_map_csv_columns_and_rows(root: Path) -> None:
    """R2: a one-map sheet's CSV: plain (the header row first, no comment
    line), one row per square in the order the sheet showed them, with the
    map's answer and what the reviewer chose in the sheet's words."""
    path = _one_map(root)
    _grade(path)
    answer = _export(path)
    csv_path, geojson_path = labels_export.export_paths(path)
    assert csv_path == root / "kelp_labels_export.csv"
    assert geojson_path == root / "kelp_labels_export.geojson"
    assert csv_path.read_text(encoding="utf-8").startswith("square,window_index,")
    rows = _csv_rows(csv_path)
    assert rows[0] == list(labels_export.ONE_MAP_COLUMNS)
    assert rows[0] == [
        "square",
        "window_index",
        "row",
        "col",
        "lon",
        "lat",
        "map_answer",
        "seen",
        "wrong",
    ]
    said = {
        14: ["Present", "Present", "0"],
        25: ["Absent", "Present", "1"],
        36: ["Present", "Cannot tell", "?"],
    }
    order = _shown_order(path)
    assert sorted(order) == [14, 25, 36] and len(rows) == 4
    for at, (index, row) in enumerate(zip(order, rows[1:], strict=True), start=1):
        r, c = divmod(index, GRID[1])
        lon, lat = _made_up(index)
        want = [str(at), str(index), str(r), str(c)]
        want += [sheet.degrees(float(lon)), sheet.degrees(float(lat))]
        assert row == want + said[index], index
    # The answer names the files and the folder, never a text or a place.
    assert answer["folder"] == str(root) and answer["n"] == 3
    assert [f["kind"] for f in answer["files"]] == ["csv", "geojson"]
    assert [f["name"] for f in answer["files"]] == [csv_path.name, geojson_path.name]
    assert [f["path"] for f in answer["files"]] == [str(csv_path), str(geojson_path)]
    assert all("text" not in f for f in answer["files"])
    for written in (csv_path, geojson_path):
        assert stat.S_IMODE(os.stat(written).st_mode) == 0o600


def test_r2_two_map_csv_has_the_second_maps_answer(root: Path) -> None:
    """R2: a two-map sheet's CSV adds the second map's answer, and whether
    each map is wrong there; a class seen is matched as the estimate
    matches classes (its letters' case aside)."""
    path = _two_map(root)
    _grade(path)
    _export(path)
    rows = _csv_rows(labels_export.export_paths(path)[0])
    assert rows[0] == list(labels_export.TWO_MAP_COLUMNS)
    assert rows[0][6:] == ["map_answer", "map_b_answer", "seen", "wrong", "wrong_b"]
    said = {
        3: ["water", "trees", "water", "0", "1"],
        8: ["trees", "water", "Trees", "0", "1"],
        12: ["water", "trees", "Cannot tell", "?", "?"],
    }
    order = _shown_order(path)
    assert sorted(order) == [3, 8, 12]
    for at, (index, row) in enumerate(zip(order, rows[1:], strict=True), start=1):
        assert row[:2] == [str(at), str(index)]
        assert row[6:] == said[index], index


def test_r2_a_right_or_wrong_sheet_says_right_or_wrong(root: Path) -> None:
    """R2: on a sheet labelled right or wrong the reviewer chose Right,
    Wrong or Cannot tell; the map's answer is the sheet's cell."""
    design = {"design": "random", "population": {"score_kind": "window_confidence"}}
    path = _one_map(root, [(14, "", "0"), (25, "", "1"), (36, "", "?")], design=design)
    _grade(path)
    _export(path)
    rows = _csv_rows(labels_export.export_paths(path)[0])
    got = {row[1]: row[6:] for row in rows[1:]}
    assert got == {
        "14": ["1", "Right", "0"],
        "25": ["0", "Wrong", "1"],
        "36": ["1", "Cannot tell", "?"],
    }


def test_r2_a_text_cell_is_never_a_formula() -> None:
    """R2: a class typed by hand cannot start a spreadsheet formula."""
    assert labels_export._cell("=1+1") == "'=1+1"
    assert labels_export._cell("water") == "water"
    assert labels_export._cell("a\nb") == "ab"


def test_r2_geojson_polygons_are_the_squares_on_the_map_grid(root: Path) -> None:
    """R2: the GeoJSON holds one Polygon per square: its window's square on
    the map's grid, as the pictures outline it, in EPSG:4326: a closed ring
    of five positions, longitude first, counterclockwise; its properties
    are the CSV row's, without the place."""
    rasterio = pytest.importorskip("rasterio")
    np = pytest.importorskip("numpy")
    from affine import Affine
    from rasterio.warp import transform as warp_transform

    epsg = 32611
    to_map = Affine(10, 0, 500000, 0, -10, 3700000)
    folder = root / "run"
    folder.mkdir()
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
        crs=f"EPSG:{epsg}",
        transform=to_map,
    ) as dst:
        dst.write(np.zeros((2, h, w), dtype=np.float32))
    scores = root / "scores_run_p11.json"
    scores.write_text(
        json.dumps(
            {
                "scores": [[0.2, 0.8]] * N,
                "grid": list(GRID),
                "patch_px": P,
                "raster": {
                    "file": "scores.tif",
                    "path": str(raster),
                    "shape": [2, h, w],
                    "sha256": "abc",
                },
                "date_window": {
                    "start": "2024-04-01T00:00:00Z",
                    "end": "2024-06-30T23:59:59.999000Z",
                },
            }
        ),
        encoding="utf-8",
    )

    def corners(index: int) -> list[tuple[float, float]]:
        """The square's corners in EPSG:4326: top left, bottom left, bottom
        right, top right."""
        r, c = divmod(index, GRID[1])
        cells = [(c, r), (c, r + 1), (c + 1, r + 1), (c + 1, r)]
        points = [to_map * (x * P, y * P) for x, y in cells]
        lons, lats = warp_transform(
            f"EPSG:{epsg}", "EPSG:4326", [p[0] for p in points], [p[1] for p in points]
        )
        return list(zip(lons, lats, strict=True))

    def centre(index: int) -> tuple[str, str]:
        """The window's centre cells, worked out from the fixture's grid."""
        r, c = divmod(index, GRID[1])
        x, y = to_map * (c * P + P / 2, r * P + P / 2)
        (lon,), (lat,) = warp_transform(f"EPSG:{epsg}", "EPSG:4326", [x], [y])
        return f"{lon:.7f}", f"{lat:.7f}"

    design = _one_map_design({"source": {"scores_path": str(scores)}})
    places = {index: centre(index) for index in (14, 25, 36)}
    path = _one_map(root, design=design, places=places)
    _grade(path)
    answer = _export(path)
    assert answer["outline"] == labels_export.OUTLINE_SQUARE
    data = json.loads(labels_export.export_paths(path)[1].read_text(encoding="utf-8"))
    assert data["type"] == "FeatureCollection" and "crs" not in data
    order = _shown_order(path)
    assert len(data["features"]) == 3
    rows = _csv_rows(labels_export.export_paths(path)[0])
    for at, (index, feature) in enumerate(
        zip(order, data["features"], strict=True), start=1
    ):
        assert feature["type"] == "Feature"
        geometry = feature["geometry"]
        assert geometry["type"] == "Polygon" and len(geometry["coordinates"]) == 1
        ring = geometry["coordinates"][0]
        assert len(ring) == 5 and ring[0] == ring[-1]
        for got, want in zip(ring[:4], corners(index), strict=True):
            assert got == pytest.approx(want, abs=1e-6)
        assert all(-180 <= x <= 180 and -90 <= y <= 90 for x, y in ring)
        # Counterclockwise: the ring's signed area is positive.
        area = sum(
            x0 * y1 - x1 * y0
            for (x0, y0), (x1, y1) in zip(ring[:-1], ring[1:], strict=True)
        )
        assert area > 0
        # The window's own centre lies inside its square.
        lon, lat = (float(v) for v in places[index])
        xs, ys = [p[0] for p in ring], [p[1] for p in ring]
        assert min(xs) < lon < max(xs) and min(ys) < lat < max(ys)
        r, c = divmod(index, GRID[1])
        row = dict(zip(rows[0], rows[at], strict=True))
        assert feature["properties"] == {
            "square": at,
            "window_index": index,
            "row": r,
            "col": c,
            "map_answer": row["map_answer"],
            "seen": row["seen"],
            "wrong": row["wrong"],
        }


def test_r2_without_a_grid_the_geojson_holds_the_centres(root: Path) -> None:
    """R2: where the map's grid cannot be read here (the design names no
    scores file), each feature is the centre its row gives, as a Point; a
    row without a place gets no geometry."""
    blank = {36: ("", "")}
    path = _one_map(root, places=blank)
    _grade(path)
    answer = _export(path)
    assert answer["outline"] == labels_export.OUTLINE_POINT
    data = json.loads(labels_export.export_paths(path)[1].read_text(encoding="utf-8"))
    for index, feature in zip(_shown_order(path), data["features"], strict=True):
        assert "lon" not in feature["properties"]
        assert "lat" not in feature["properties"]
        if index == 36:
            assert feature["geometry"] is None
            continue
        lon, lat = (float(v) for v in _made_up(index))
        assert feature["geometry"] == {"type": "Point", "coordinates": [lon, lat]}


def _scores_file(root: Path, *, period: bool) -> Path:
    """A scores file that names a small raster on the fixture's made-up grid
    (inside the output root), with its period or without one."""
    rasterio = pytest.importorskip("rasterio")
    np = pytest.importorskip("numpy")
    from affine import Affine

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
        crs="EPSG:32611",
        transform=Affine(10, 0, 500000, 0, -10, 3700000),
    ) as dst:
        dst.write(np.zeros((2, h, w), dtype=np.float32))
    meta: dict[str, Any] = {
        "scores": [[0.2, 0.8]] * N,
        "grid": list(GRID),
        "patch_px": P,
        "raster": {
            "file": "scores.tif",
            "path": str(raster),
            "shape": [2, h, w],
            "sha256": "abc",
        },
    }
    if period:
        meta["date_window"] = {
            "start": "2024-04-01T00:00:00Z",
            "end": "2024-06-30T23:59:59.999000Z",
        }
    scores = root / "scores_run_p11.json"
    scores.write_text(json.dumps(meta), encoding="utf-8")
    return scores


def test_r2_a_map_with_no_period_is_outlined_too(root: Path) -> None:
    """R2 (RESULT-T2): the outline needs the map's grid and its raster's
    header, not its period: a scores file that records no period still
    gives Polygons (``imagery._load_grid``), while the pictures still
    refuse it (``imagery._load_map``); with its period both read it."""
    scores = _scores_file(root, period=False)
    with pytest.raises(imagery._Refused):
        imagery._load_map(str(scores))
    grid = imagery._load_grid(str(scores))
    assert (grid.grid, grid.patch) == (GRID, P)
    design = _one_map_design({"source": {"scores_path": str(scores)}})
    path = _one_map(root, design=design)
    _grade(path)
    answer = _export(path)
    assert answer["outline"] == labels_export.OUTLINE_SQUARE
    data = json.loads(labels_export.export_paths(path)[1].read_text(encoding="utf-8"))
    assert len(data["features"]) == 3
    for feature in data["features"]:
        assert feature["geometry"]["type"] == "Polygon"
        ring = feature["geometry"]["coordinates"][0]
        assert len(ring) == 5 and ring[0] == ring[-1]
    imagery._maps.clear()
    with_period = imagery._load_map(str(_scores_file(root, period=True)))
    assert (with_period.grid, with_period.patch) == (GRID, P)
    assert with_period.transform == grid.transform and with_period.crs == grid.crs


def test_r2_the_formula_guard_is_the_csvs_alone(root: Path) -> None:
    """R2 (RESULT-T3), through the export itself: a map's class written
    ``-1`` is ``'-1`` in the CSV (never read as a formula) and ``-1`` in
    the GeoJSON's properties, the map's answer as it is; a negative
    coordinate cell keeps its sign bare in the CSV."""
    design = {"design": "random", "population": {"score_kind": "window_confidence"}}
    (root / "kelp.json").write_text(json.dumps(design), encoding="utf-8")
    lon, lat = _made_up(14)
    cells = ["1", "14", "1", "4", "-1", "-" + lon, lat, "0", "1", ""]
    path = _write(root / "kelp_to_label.csv", ONE_MAP, [cells])
    _grade(path)
    _export(path)
    csv_path, geojson_path = labels_export.export_paths(path)
    header, row = _csv_rows(csv_path)
    written = dict(zip(header, row, strict=True))
    assert written["map_answer"] == "'-1"
    assert written["lon"] == sheet.degrees(-float(lon))
    assert written["lon"].startswith("-")
    assert (written["seen"], written["wrong"]) == ("Wrong", "1")
    data = json.loads(geojson_path.read_text(encoding="utf-8"))
    assert data["features"][0]["properties"]["map_answer"] == "-1"
    assert labels_export._guard("-1") == "'-1"
    assert labels_export._clean("-1") == "-1"


def _class_map(root: Path, rows: list[tuple[int, str, str, str]]) -> Path:
    """A class map's sheet and its design (water where the window's index is
    even, trees where it is odd, and a code out of the classes' range at
    window 58): each row the window's index, the sheet's own ``map_class``
    cell, ``wrong`` and the class seen."""
    on_map = [index % 2 for index in range(N)]
    on_map[58] = 9
    population = {
        "score_kind": "class_map",
        "classes": ["water", "trees"],
        "map_class": on_map,
    }
    (root / "land.json").write_text(
        json.dumps({"design": "class", "population": population}), encoding="utf-8"
    )
    lines = []
    for order, (index, cell, wrong, ref) in enumerate(rows, start=1):
        r, c = divmod(index, GRID[1])
        lon, lat = _made_up(index)
        cells = [str(order), str(index), str(r), str(c), cell, lon, lat, "0"]
        lines.append([*cells, wrong, ref])
    return _write(root / "land_to_label.csv", ONE_MAP, lines)


def test_r2_a_class_maps_sheet_is_graded_against_the_designs_class(
    root: Path,
) -> None:
    """R2 (RESULT-T4): on a class map's sheet the map's answer is the
    design's class at the design's code (not the sheet's own cell, which
    here differs or is blank), and ``wrong`` is worked out from the class
    seen as the estimate matches classes; a design code out of range falls
    back to the sheet's cell."""
    rows = [
        (14, "trees", "", "water"),
        (25, "", "", "water"),
        (36, "water", "", "Water"),
        (47, "trees", "", "?"),
        (58, "trees", "", "trees"),
    ]
    path = _class_map(root, rows)
    _grade(path)
    _export(path)
    written = _csv_rows(labels_export.export_paths(path)[0])
    got = {row[1]: row[6:] for row in written[1:]}
    assert got == {
        "14": ["water", "water", "0"],
        "25": ["trees", "water", "1"],
        "36": ["water", "Water", "0"],
        "47": ["trees", "Cannot tell", "?"],
        "58": ["trees", "trees", "0"],
    }


def test_r2_a_class_maps_sheet_labelled_right_or_wrong(root: Path) -> None:
    """R2 (RESULT-T4): a class map's sheet holding ``wrong`` alone says
    Right or Wrong, with the design's class as the map's answer."""
    path = _class_map(root, [(14, "", "0", ""), (25, "", "1", ""), (36, "", "?", "")])
    _grade(path)
    _export(path)
    written = _csv_rows(labels_export.export_paths(path)[0])
    got = {row[1]: row[6:] for row in written[1:]}
    assert got == {
        "14": ["water", "Right", "0"],
        "25": ["trees", "Wrong", "1"],
        "36": ["water", "Cannot tell", "?"],
    }


def test_r2_a_sheet_of_points_says_points_and_has_no_outline(root: Path) -> None:
    """R2 (RESULT-T4): a sheet of a Studio result's points is refused in
    its own word ("point"); once answered and graded its GeoJSON holds each
    point's centre (``outline`` point), no geometry for a row without a
    place, and ``outline`` none when no row has one."""
    population = {
        "score_kind": "window_confidence",
        "source": {"result_id": "made-up-result"},
    }
    design = {"design": "random", "population": population}
    assert sheet.window_unit(design) == sheet.UNIT_POINT == "point"
    answers = [(14, "", "0"), (25, "", ""), (36, "", "?")]
    path = _one_map(root, answers, design=design)
    with pytest.raises(ValueError) as refused:
        _export(path)
    line = labels_export.NOT_EVERY.format(unit="point", left=1, n=3)
    assert str(refused.value) == line
    answers[1] = (25, "", "1")
    path = _one_map(root, answers, design=design, places={36: ("", "")})
    _grade(path)
    answer = _export(path)
    assert answer["outline"] == labels_export.OUTLINE_POINT
    data = json.loads(labels_export.export_paths(path)[1].read_text(encoding="utf-8"))
    kinds = {
        feature["properties"]["window_index"]: (feature["geometry"] or {}).get("type")
        for feature in data["features"]
    }
    assert kinds == {14: "Point", 25: "Point", 36: None}


def test_r2_a_sheet_with_no_place_has_no_geometry(root: Path) -> None:
    """R2 (RESULT-T4): when no row gives a place every feature's geometry
    is null and the answer's ``outline`` says none."""
    blank = {index: ("", "") for index in (14, 25, 36)}
    path = _one_map(root, places=blank)
    _grade(path)
    answer = _export(path)
    assert answer["outline"] == labels_export.OUTLINE_NONE
    data = json.loads(labels_export.export_paths(path)[1].read_text(encoding="utf-8"))
    assert [feature["geometry"] for feature in data["features"]] == [None] * 3


def test_r2_with_text_each_file_carries_its_text(root: Path) -> None:
    """R2: for a host that saves files the answer carries each file's text,
    the same text as the file written."""
    path = _one_map(root)
    _grade(path)
    answer = _export(path, text=True)
    csv_path, geojson_path = labels_export.export_paths(path)
    with open(csv_path, encoding="utf-8", newline="") as fh:
        assert answer["files"][0]["text"] == fh.read()
    assert answer["files"][1]["text"] == geojson_path.read_text(encoding="utf-8")
    assert [f["mime"] for f in answer["files"]] == ["text/csv", "application/geo+json"]


# ------------------------------------------------------------- R2, the refusals


def test_r2_refused_before_every_square_is_answered(root: Path) -> None:
    """R2: nothing is exported while a square has no answer; the refusal is
    one plain line, and no file is written."""
    path = _one_map(root, [(14, "1", "0"), (25, "", ""), (36, "?", "?")])
    with pytest.raises(ValueError) as refused:
        _export(path)
    line = "Answer every square on the sheet first: 1 of 3 have no answer yet."
    assert str(refused.value) == line
    assert line == labels_export.NOT_EVERY.format(unit="square", left=1, n=3)
    assert not any(p.exists() for p in labels_export.export_paths(path))
    assert labels_export.offer(str(path)) is None


def test_r2_refused_before_the_sheet_has_its_result(root: Path) -> None:
    """R2: every square answered but no estimate yet (no graded mark):
    refused, nothing written; once graded, the export is offered."""
    path = _one_map(root)
    with pytest.raises(ValueError) as refused:
        _export(path)
    assert str(refused.value) == labels_export.NO_RESULT
    assert not any(p.exists() for p in labels_export.export_paths(path))
    assert labels_export.offer(str(path)) is None
    _grade(path)
    assert labels_export.offer(str(path)) == {
        "labels_csv_path": str(path),
        "design_path": _design_of(path),
    }


def test_r2_paths_outside_the_output_root_are_refused(
    root: Path, tmp_path: Path
) -> None:
    """R2: the sheet is checked as every sheet tool checks it (inside the
    output root), and design_path must be the design file beside it."""
    path = _one_map(root)
    _grade(path)
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    stray = _write(outside / "kelp_to_label.csv", ONE_MAP, [])
    (outside / "kelp.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="under the agent's output folder"):
        labels_export.export_labels(str(stray), str(outside / "kelp.json"))
    with pytest.raises(ValueError, match="under the agent's output folder"):
        labels_export.export_labels(str(root / "kelp.json"), _design_of(path))
    (root / "other.json").write_text("{}", encoding="utf-8")
    for design in (
        str(outside / "kelp.json"),
        str(root / "other.json"),
        str(path),
        "",
        None,
    ):
        with pytest.raises(ValueError) as refused:
            labels_export.export_labels(str(path), design)
        assert str(refused.value) == labels_export.NOT_ITS_DESIGN, design
    # A link at the design's name that leads out of the root is not its design.
    (root / "kelp.json").unlink()
    (root / "kelp.json").symlink_to(outside / "kelp.json")
    with pytest.raises(ValueError) as refused:
        labels_export.export_labels(str(path), str(root / "kelp.json"))
    assert str(refused.value) == labels_export.NOT_ITS_DESIGN
    assert not any(p.exists() for p in labels_export.export_paths(path))


def test_r2_the_refusals_are_marked_as_wording() -> None:
    """R2: each sentence a user can read from the export is marked."""
    source = Path(labels_export.__file__).read_text(encoding="utf-8")
    names = (
        "NOT_EVERY",
        "NO_RESULT",
        "CHANGED",
        "NOT_ITS_DESIGN",
        "NOT_WRITTEN",
        "MODEL_NOTE",
    )
    for name in names:
        assert f"# WORDING: r5r\n{name} = " in source, name


def test_r2_a_refusal_stands_alone_on_the_card() -> None:
    """R2 (D3): the card shows the server's refusal as it is, so each is a
    whole sentence that names no argument and says no word a face never
    says (design rule R7)."""
    for line in (
        labels_export.NOT_EVERY.format(unit="square", left=1, n=3),
        labels_export.NO_RESULT,
        labels_export.CHANGED,
        labels_export.NOT_ITS_DESIGN,
        labels_export.NOT_WRITTEN,
    ):
        assert line[0].isupper() and line.endswith("."), line
        assert "_path" not in line and "design" not in line.lower(), line
    with pytest.raises(ValueError) as refused:
        labels_export.export_labels(None, None)
    assert str(refused.value) == labels_export.NOT_ITS_DESIGN


def test_r2_refused_once_the_answers_changed_after_the_result(root: Path) -> None:
    """R2 (RESULT-T1): the result on the card came from the answers first
    graded. Once an answer changed (every square still answered) the export
    is refused in one plain sentence, nothing is written and nothing is
    offered; so too after the changed answers were graded again (the mark's
    own ``changed``, the flag a result is withdrawn on), whatever the sheet
    holds then."""
    path = _one_map(root)
    _grade(path)
    assert labels_export.offer(str(path)) is not None
    # One answer changed after the first result; not graded again.
    _one_map(root, [(14, "0", "1"), (25, "1", "1"), (36, "?", "?")])
    state = sheet.graded_state(str(path))
    assert state is not None and state["changed_since_first"] is True
    with pytest.raises(ValueError) as refused:
        _export(path)
    assert str(refused.value) == labels_export.CHANGED
    assert not any(p.exists() for p in labels_export.export_paths(path))
    assert labels_export.offer(str(path)) is None
    # Graded again on the changed answers, then the first answers put back.
    _grade(path)
    _one_map(root)
    state = sheet.graded_state(str(path))
    assert state is not None and state["changed"] is True
    with pytest.raises(ValueError) as refused:
        _export(path)
    assert str(refused.value) == labels_export.CHANGED
    assert not any(p.exists() for p in labels_export.export_paths(path))


# ----------------------------------------------------------- R2, the views' tool


def test_r2_the_tool_is_one_of_the_views_tools() -> None:
    """R2: olmoearth_app_export_labels is app-only (APP_TOOLS), has its
    handler, takes the sheet's two paths, and is not one of the quiet tools
    (the model reads its refusal)."""
    assert server.EXPORT_TOOL == "olmoearth_app_export_labels"
    assert server.EXPORT_TOOL in server.APP_TOOLS
    assert server.EXPORT_TOOL not in server.QUIET_APP_TOOLS
    assert callable(getattr(server.AppServer, "_app_export_labels"))
    schema = server.APP_TOOLS[server.EXPORT_TOOL][1]
    assert schema["required"] == ["labels_csv_path", "design_path"]
    assert set(schema["properties"]) == {"labels_csv_path", "design_path", "text"}


@pytest.mark.asyncio
async def test_r2_the_models_text_names_the_files_and_no_place(root: Path) -> None:
    """R2: the model reads that the files were written and their paths: no
    contents, and no place anywhere in the result without text; with text
    (a host that saves files) the place is in the files' texts alone."""
    path = _one_map(root)
    _grade(path)
    srv = server.build_server(_app())
    arguments = {"labels_csv_path": str(path), "design_path": _design_of(path)}
    result = await _call(srv, server.EXPORT_TOOL, arguments)
    assert result.isError is not True
    csv_path, geojson_path = labels_export.export_paths(path)
    assert [json.loads(text) for text in _texts(result)] == [
        {
            "ok": True,
            "note": labels_export.MODEL_NOTE,
            "files": [str(csv_path), str(geojson_path)],
        }
    ]
    places = [cell for index in (14, 25, 36) for cell in _made_up(index)]
    whole = result.model_dump_json(by_alias=True)
    assert not any(cell in whole for cell in places)
    assert places[0] in csv_path.read_text(encoding="utf-8")
    with_text = await _call(srv, server.EXPORT_TOOL, {**arguments, "text": True})
    assert not any(cell in "".join(_texts(with_text)) for cell in places)
    files = (with_text.structuredContent or {})["files"]
    assert files[0]["text"].startswith("square,window_index,")
    # A refusal is the model's to read too: one plain line, no place.
    refused = await _call(
        srv, server.EXPORT_TOOL, {**arguments, "design_path": str(path)}
    )
    assert refused.isError is True
    assert labels_export.NOT_ITS_DESIGN in "".join(_texts(refused))
    bad = await _call(srv, server.EXPORT_TOOL, {**arguments, "text": "yes"})
    assert bad.isError is True and "text must be true or false" in _texts(bad)[0]


def test_r2_an_estimates_view_offers_the_export(root: Path) -> None:
    """R2: the estimate's view carries the export's two paths only for the
    estimate tool, on the sheet it graded, with an ok result that is not
    withdrawn, and only while the sheet's answers are those first graded
    (``server._export_offer``, on real sheets); the tool's call puts what it
    gives into the view (read as text)."""
    path = _one_map(root)
    _grade(path)
    name, ok = server.ESTIMATE_TOOL, {"ok": True}
    want = {"labels_csv_path": str(path), "design_path": _design_of(path)}
    assert server._export_offer(name, str(path), ok, {}) == want
    calm = {"result": {"withdrawn": None}}
    assert server._export_offer(name, str(path), ok, calm) == want
    gone = {"result": {"withdrawn": {"title": "This result is out of date"}}}
    assert server._export_offer(name, str(path), ok, gone) is None
    assert server._export_offer(name, str(path), {"ok": False}, {}) is None
    assert server._export_offer(name, None, ok, {}) is None
    other = "olmoearth_plan_label_sample"
    assert server._export_offer(other, str(path), ok, {}) is None
    # An answer changed after the first result: nothing is offered.
    _one_map(root, [(14, "0", "1"), (25, "1", "1"), (36, "?", "?")])
    assert server._export_offer(name, str(path), ok, {}) is None
    source = Path(server.__file__).read_text(encoding="utf-8")
    assert _in_order(
        source,
        "        export = await asyncio.to_thread(\n"
        "            _export_offer, name, graded_path, envelope, view\n"
        "        )\n",
        "        if export is not None:\n",
        '            view["export"] = export\n',
    )
    assert "content=_text(_app_said(name, result)), structuredContent=result" in source


# ------------------------------------------------------------- R2, the card


def test_r2_the_card_offers_the_export_over_its_details() -> None:
    """R2: "Export the labels" is a plain button (not the dark kind), the
    result card's last block over its Details fold, and on an estimate
    without that card (a two-map comparison) over the fold too."""
    page = _read("result.html")
    card = _between(page, "function resultCard(app, data) {", "\n}\n")
    assert _in_order(
        card,
        "const offer = exportOf(data);",
        "if (offer) card.append(exportBlock(app, offer));",
        "if (folded.length) card.append(OE.fold(RESULT_WORDS.details, ...folded));",
    )
    show = _between(page, "function show(app, params) {", "\n}\n")
    assert _in_order(
        show,
        "if (plan) root.append(sheetButton(app, data, figureSlot));",
        "const offer = exportOf(data);",
        "if (offer) root.append(exportBlock(app, offer));",
        "if (inFold.length) root.append(OE.fold(FOLD_LABEL, ...inFold));",
    )
    # A withdrawn result's card has no export.
    gone = _between(page, "function withdrawnCard(data) {", "\n}\n")
    assert "exportBlock" not in gone
    block = _between(page, "function exportBlock(app, offer) {", "\n}\n")
    assert "class: 'primary'" not in block
    assert "type: 'button', 'data-focus': 'export'," in block
    assert "}, EXPORT_WORDS.button);" in block


def test_r2_the_cards_words() -> None:
    """R2: the card's words, marked; it says once that the files are saved
    on this computer, in the app's folder."""
    page = _read("result.html")
    words = _between(page, "// WORDING: r5r\nconst EXPORT_WORDS = {", "\n};\n")
    for entry in (
        "  button: 'Export the labels',\n",
        "  saved: 'Saved on this computer, in the app\\'s folder.',\n",
        "  folder: 'Folder: ',\n",
        "  centres: 'The GeoJSON file holds each square\\'s centre point,"
        " not its outline.',\n",
        "  noPlaces: 'The GeoJSON file holds no places: this sheet has none.',\n",
        "  copy: 'Copy path',\n",
        "  download: 'Download',\n",
        "  notExported: (why) => 'The labels were not exported: ' + why + '.',\n",
        "  notMade: 'the files could not be written',\n",
    ):
        assert entry in words, entry
    assert page.count("EXPORT_WORDS.saved") == 1


def test_r2_the_saved_line_is_announced_and_the_folder_said_once() -> None:
    """R2 (D1, D2, D4): the saved sentence has a status line of its own,
    made before its words and emptied at a press and at a failure; the
    folder is said once (each file's whole path is hidden until Copy path
    has to select it, which OE.copyButton unhides); each Copy path is
    described by its file's name; a GeoJSON without outlines is said in one
    small line (RESULT-T2)."""
    page = _read("result.html")
    block = _between(page, "function exportBlock(app, offer) {", "\n}\n")
    assert _in_order(
        block,
        "const saved = OE.el('div', { class: 'small muted oe-export-saved',"
        " role: 'status', 'aria-live': 'polite' });",
        "function draw(list, folder, canSave, outline) {",
        "const nameId = 'oe-export-name-' + block + '-' + file.kind;",
        "const path = OE.el('div', { class: 'small mono csv-path oe-export-path',"
        " hidden: true }, file.path);",
        "const copy = OE.copyButton(file.path, path, EXPORT_WORDS.copy);",
        "copy.setAttribute('aria-describedby', nameId);",
        "OE.el('div', { class: 'oe-export-name', id: nameId }, file.name),"
        " path, buttons);",
        "const held = outline === 'point' ? EXPORT_WORDS.centres"
        " : (outline === 'none' ? EXPORT_WORDS.noPlaces : '');",
        "saved.textContent = EXPORT_WORDS.saved;",
        "held ? OE.el('div', { class: 'small muted oe-export-outline' }, held) : null,",
        "OE.el('div', { class: 'row' }, button), said, saved, files);",
    )
    assert block.count("saved.textContent = '';") == 2
    assert block.count("EXPORT_WORDS.folder") == 1
    style = _between(page, "<style>", "</style>")
    assert "  .oe-export-path[hidden] { display: none; }\n" in style
    assert ".oe-export.stack > .oe-export-saved:empty" in style
    copy = _between(_read("common.js"), "OE.copyButton = (text, shown, label", "\n};")
    assert _in_order(
        copy,
        "if (!copied && shown) {",
        "shown.hidden = false;",
        "range.selectNodeContents(shown);",
    )


def test_r2_the_press_sends_the_two_paths_and_shows_names_and_paths() -> None:
    """R2: the press calls the views' tool with the sheet's two paths, and
    with text only where the host saves files; each file gets a Copy path
    button (OE.copyButton, which shows and selects the file's path where the
    panel may not write the clipboard) and, only where the host saves
    files, a Download button that hands the host the file's text. What
    Claude is told names the folder (when the answer names one,
    RESULT-T5) and the files, nothing else. A refusal of the server's is
    said as it is; any other failure gets the one fixed sentence, never the
    host's words (D3)."""
    page = _read("result.html")
    block = _between(page, "function exportBlock(app, offer) {", "\n}\n")
    for text in (
        "const canSave = typeof OE.canDownload === 'function' && OE.canDownload(app);",
        "const got = await OE.call(app, 'olmoearth_app_export_labels',\n"
        "          { labels_csv_path: offer.labels_csv_path,"
        " design_path: offer.design_path, text: canSave });",
        "OE.copyButton(file.path, path, EXPORT_WORDS.copy)",
        "if (canSave && typeof file.text === 'string' && file.text)"
        " buttons.append(downloadButton(file));",
        "await app.downloadFile({ contents: [{ type: 'resource', resource:"
        " { uri: 'file:///' + exportName(file), mimeType, text: file.text } }] });",
        "const where = folder ? ' to the folder ' + folder : '';",
        "OE.tellClaude(app, 'Result card: the labels were exported' + where + ' as '\n"
        "      + list.map((file) => file.name).join(' and ') + '.', 'export');",
        "if (list.length !== 2) throw new Error(EXPORT_WORDS.notMade);",
        "const refusal = err && err.fromServer === true"
        " && typeof err.message === 'string' ? err.message.trim() : '';",
        "say(refusal || EXPORT_WORDS.notExported(EXPORT_WORDS.notMade), true);",
    ):
        assert text in block, text
    # The host's own words never reach the line: no other use of the error.
    assert block.count("err.message") == 2 and "bareText" not in block
    # A file's text is never put into the page, a toast or Claude's context.
    assert "OE.toast(" not in block
    assert block.count("file.text") == 3
    offer = _between(page, "function exportOf(data) {", "\n}\n")
    assert (
        "return { labels_csv_path: e.labels_csv_path, design_path: e.design_path };"
        in offer
    )


# ------------------------------------------------------------- R1, the box


def test_r1_the_headline_is_boxed() -> None:
    """R1: the box's CSS, whole, on the headline of the rate block and on a
    withdrawn result's out-of-date rate; the headline's markup is the one
    sentence in p.big, its range figure the next node, outside the box."""
    page = _read("result.html")
    style = _between(page, "<style>", "</style>")
    assert BOX_CSS in style
    assert style.count("1.5px solid") == 1 and style.count("border-radius: 6px") == 1
    # The rule the box adds to stays, word for word.
    assert (
        "  .oe-rate p.big { font-size: 20px; line-height: 28px; font-weight: 600; }\n"
        in style
    )
    rate = _between(page, "function rateBlock(r) {", "\n}\n")
    assert _in_order(
        rate,
        "return OE.el('div', { class: 'oe-rate' },\n",
        "    OE.el('p', { class: 'big' }, r.headline),\n",
        "    line,\n",
        "    text(r.hedge) ? OE.el('p', { class: 'oe-hedge' }, r.hedge) : null,\n",
    )


def test_r1_the_out_of_date_rate_keeps_its_words_and_the_box() -> None:
    """R1: a withdrawn result's rate keeps its warning words (r5-k K1) and
    is the one other line the box's rule names."""
    page = _read("result.html")
    gone = _between(page, "function withdrawnCard(data) {", "\n}\n")
    assert "OE.el('p', { class: 'small oe-out-of-date' }, old)" in gone
    words = _between(page, "const RESULT_WORDS = {", "\n};\n")
    assert (
        "outOfDateRate: (headline, at) => 'Out of date, do not use (computed '"
        " + at + ' from the changed answers): ' + headline," in words
    )
    assert page.count("p.oe-out-of-date") == 1


def test_r1_no_other_view_has_the_box() -> None:
    """R1: not on other cards: no other view's page names the box's rule."""
    for name in ("check.html", "start.html", "sheet.html", "map.html"):
        assert "1.5px solid var(--text, #0a3235)" not in _read(name), name
