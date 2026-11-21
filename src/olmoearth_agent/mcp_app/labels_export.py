# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""A labelled sheet's answers as two files, once its result is out (r5-r R2).

The result card of a labelled sheet offers "Export the labels". Its press
calls the views' tool ``olmoearth_app_export_labels``
(:func:`export_labels`), which writes, next to the sheet:

- ``<stem>_labels_export.csv``: a plain CSV (a header row, then one row per
  square, no comment line), in the order the sheet showed its squares (the
  order of their handles, so "Square 29" on the sheet is ``square`` 29
  here): :data:`ONE_MAP_COLUMNS`, or :data:`TWO_MAP_COLUMNS` for a two-map
  sheet.
- ``<stem>_labels_export.geojson``: a FeatureCollection, one feature per
  square, its properties the CSV row without the place. Its geometry is the
  square of the window on the map's grid, as the pictures outline it,
  reprojected to EPSG:4326 (a closed ring of five positions, longitude
  first); where the map's grid cannot be read here (a sheet of points, no
  scores file that names a raster or one outside the read folders, a grid
  that does not match the raster, a row that disagrees with the grid, no
  ``inferencex`` extra), the centre the sheet's row gives as a Point, or
  no geometry when the row has none. The answer's ``outline`` says which,
  and the card says so when it is not the square.

Both files are written whole, mode 0600, and replace an earlier export.
Nothing is exported before every square is answered (:data:`NOT_EVERY`),
before the sheet has its result (the graded mark the estimate writes:
:data:`NO_RESULT`), or once its answers changed after that result
(:data:`CHANGED`: the result on the card is not theirs, round r5-k K1).
Blind labelling is over by then, so the map's answer is written.

The files hold each square's place. The tool's answer does not: it names
the files and their folder (and, for a host that saves files, their text,
which the page passes straight on to the host), and the model reads only
that the files were written and where (:func:`model_text`).
"""

from __future__ import annotations

import csv
import importlib
import io
import json
import logging
import math
import os
import tempfile
from pathlib import Path
from typing import Any

from olmoearth_agent.mcp_app import imagery, sheet

logger = logging.getLogger(__name__)

#: The suffixes of the two files, after the sheet's stem (its name without
#: ``sheet.SHEET_SUFFIX``).
CSV_SUFFIX = "_labels_export.csv"
GEOJSON_SUFFIX = "_labels_export.geojson"

#: The media types the page hands the host with a file's text.
CSV_MIME = "text/csv"
GEOJSON_MIME = "application/geo+json"

#: The CSV's columns for a one-map sheet, in order.
ONE_MAP_COLUMNS = (
    "square",
    "window_index",
    "row",
    "col",
    "lon",
    "lat",
    "map_answer",
    "seen",
    "wrong",
)

#: The CSV's columns for a two-map sheet: the second map's answer after the
#: first's, and whether the second map is wrong after whether the first is.
TWO_MAP_COLUMNS = (
    "square",
    "window_index",
    "row",
    "col",
    "lon",
    "lat",
    "map_answer",
    "map_b_answer",
    "seen",
    "wrong",
    "wrong_b",
)

#: The columns left out of a feature's properties: its geometry says the place.
_PLACE_COLUMNS = ("lon", "lat")

#: The columns written as whole numbers in a feature's properties.
_WHOLE_COLUMNS = ("square", "window_index", "row", "col")

#: The words of a right-or-wrong sheet's answers, as its face says them.
RIGHT = "Right"
WRONG = "Wrong"

#: Refused while a square has no answer.
# WORDING: r5r
NOT_EVERY = "Answer every {unit} on the sheet first: {left} of {n} have no answer yet."

#: Refused before the sheet has its result.
# WORDING: r5r
NO_RESULT = (
    "The labels are exported once the sheet has its result. Press the sheet's "
    "finish button first."
)

#: Refused once the sheet's answers are not the ones its result was worked
#: out from: changed since the first result, or graded again after a change.
# WORDING: r5r
CHANGED = (
    "The answers on this sheet changed after its result, so the labels are "
    "not exported."
)

#: Refused when the two paths are not a sheet's own: ``labels_csv_path`` no
#: text, or ``design_path`` not the design file beside the sheet. The card
#: shows a refusal as it is, so it names no argument.
# WORDING: r5r
NOT_ITS_DESIGN = (
    "These labels cannot be exported from this card. Ask Claude for the "
    "result again."
)

#: Said when a file could not be written. The card shows it as it is, so it
#: says itself that nothing was exported.
# WORDING: r5r
NOT_WRITTEN = (
    "The labels were not exported: the files could not be written in the "
    "sheet's folder."
)

#: What the model reads once the files are written (:func:`model_text`).
# WORDING: r5r
MODEL_NOTE = (
    "The labels were exported: two files were written beside the label sheet, "
    "at the paths in files. Tell the user where they are. The files hold each "
    "square's place: do not open them or print their contents in the chat."
)

#: How a feature's geometry was made (the answer's ``outline``).
OUTLINE_SQUARE = "square"
OUTLINE_POINT = "point"
OUTLINE_NONE = "none"

#: A square's corners on the map's grid as (columns, rows) from its top left
#: corner: top left, bottom left, bottom right, top right, so the ring runs
#: counterclockwise on a north-up map (RFC 7946).
_CORNERS = ((0, 0), (0, 1), (1, 1), (1, 0))

#: What a spreadsheet would take as a formula at a cell's start.
_FORMULA_START = ("=", "+", "-", "@", "\t", "\r")


#: The CSV's answer columns: the ones a hand-typed class can reach, so the
#: ones guarded against a formula there (never ``lon`` or ``lat``: a
#: negative one keeps its sign bare).
_ANSWER_COLUMNS = ("map_answer", "map_b_answer", "seen", "wrong", "wrong_b")


def _clean(text: Any) -> str:
    """An answer as both files hold it: printable characters only, at most
    ``sheet.MAX_LABEL_CHARS``."""
    clean = "".join(ch for ch in str(text) if ch.isprintable())
    return clean[: sheet.MAX_LABEL_CHARS]


def _guard(clean: str) -> str:
    """A CSV cell never read as a formula (a leading ``'`` then), as the
    look-first CSV writes its cells. The CSV's alone: the GeoJSON holds the
    answer as it is (a class written ``-1`` stays ``-1`` there)."""
    return "'" + clean if clean.startswith(_FORMULA_START) else clean


def _cell(text: Any) -> str:
    """A text cell of the CSV: cleaned (:func:`_clean`), then guarded
    (:func:`_guard`)."""
    return _guard(_clean(text))


def _named(names: dict[str, str], value: str) -> str:
    """A class as the sheet words it: a code's name with a capital first
    letter ("Present"), as a seen sheet's answers say it; any other value
    as written."""
    name = names.get(value)
    return name[:1].upper() + name[1:] if name else value


def export_paths(real: Path) -> tuple[Path, Path]:
    """Where a resolved sheet's two export files are: next to it,
    ``<stem>_labels_export.csv`` and ``<stem>_labels_export.geojson``."""
    stem = real.name[: -len(sheet.SHEET_SUFFIX)]
    return real.with_name(stem + CSV_SUFFIX), real.with_name(stem + GEOJSON_SUFFIX)


def _own_design(real: Path, design_path: Any) -> None:
    """Hold ``design_path`` to the sheet's own design file: the regular file
    ``<stem>.json`` beside the resolved sheet (so inside the output root),
    named by its real path. ``ValueError`` (:data:`NOT_ITS_DESIGN`) for any
    other path, a link among them."""
    if not isinstance(design_path, str) or not design_path or "\x00" in design_path:
        raise ValueError(NOT_ITS_DESIGN)
    own = real.with_name(real.name[: -len(sheet.SHEET_SUFFIX)] + ".json")
    if os.path.realpath(design_path) != str(own) or not own.is_file():
        raise ValueError(NOT_ITS_DESIGN)


def _map_class(
    design: dict[str, Any] | None, rule: str | None, index: int, cell: str
) -> str:
    """The map's class of a window as written (a code, or a class map's
    name), before any naming.

    A seen sheet's is the design's (``population.map_class``, the one its
    ``wrong`` was worked out from); a class map's the design's class at the
    design's code there, as the estimate grades it; any other sheet's, and
    either one when the design cannot tell, the sheet's ``map_class`` cell.
    """
    population = (design or {}).get("population")
    population = population if isinstance(population, dict) else {}
    if rule == "seen":
        try:
            return str(sheet._map_code(population, index))
        except ValueError:
            return cell
    classes, on_map = population.get("classes"), population.get("map_class")
    if rule == "class" and isinstance(classes, list) and isinstance(on_map, list):
        code = on_map[index] if 0 <= index < len(on_map) else None
        if (
            isinstance(code, int)
            and not isinstance(code, bool)
            and 0 <= code < len(classes)
        ):
            return str(classes[code])
    return cell


def _differs(seen: str, on_map: str) -> str:
    """``wrong`` for a class seen against a map's class: ``"1"`` where they
    are not one class, matched as the estimate matches classes (case, outer
    spaces and ``3.0`` against ``3`` aside), else ``"0"``."""
    from olmoearth_agent.tools.estimation import _class_key

    return "0" if _class_key(seen) == _class_key(on_map) else "1"


def _one_map_answers(
    row: dict[str, str],
    *,
    design: dict[str, Any] | None,
    rule: str | None,
    names: dict[str, str],
) -> dict[str, str]:
    """A one-map row's ``map_answer``, ``seen`` and ``wrong``.

    ``map_answer``: the map's class there (:func:`_map_class`), by its name
    where the design names it. ``seen``: what the reviewer chose, as the
    sheet's face says it: the class seen (a seen sheet's code by its name),
    Right or Wrong on a sheet labelled right or wrong, or Cannot tell.
    ``wrong``: the sheet's cell (1, 0 or ?), or, on a class map's sheet
    with the class seen alone, worked out from it as the estimate does.
    """
    index = int(str(row.get("window_index") or "").strip())
    ref = (row.get("reference_class") or "").strip()
    said = (row.get("wrong") or "").strip()
    on_map = _map_class(design, rule, index, (row.get("map_class") or "").strip())
    if sheet.UNJUDGED in (ref, said):
        seen, wrong = sheet.CANNOT_TELL, sheet.UNJUDGED
    elif ref:
        seen = _named(names, ref)
        wrong = said if said in ("0", "1") else _differs(ref, on_map)
    else:
        seen = {"0": RIGHT, "1": WRONG}.get(said, "")
        wrong = said if said in ("0", "1") else ""
    return {"map_answer": _named(names, on_map), "seen": seen, "wrong": wrong}


def _two_map_answers(row: dict[str, str]) -> dict[str, str]:
    """A two-map row's answers: the two maps' classes (``map_answer``,
    ``map_b_answer``), the class seen or Cannot tell (``seen``), and
    whether each map is wrong there (``wrong``, ``wrong_b``: 1, 0, or ?
    with Cannot tell)."""
    ref = (row.get("reference_class") or "").strip()
    a = (row.get("class_a") or "").strip()
    b = (row.get("class_b") or "").strip()
    unsure = ref == sheet.UNJUDGED
    return {
        "map_answer": a,
        "map_b_answer": b,
        "seen": sheet.CANNOT_TELL if unsure else ref,
        "wrong": sheet.UNJUDGED if unsure else _differs(ref, a),
        "wrong_b": sheet.UNJUDGED if unsure else _differs(ref, b),
    }


def _place(row: dict[str, str]) -> tuple[float, float] | None:
    """A row's centre as ``(lon, lat)``, or ``None`` when the sheet gives none."""
    lon = sheet._coordinate(row.get("lon"), 180.0)
    lat = sheet._coordinate(row.get("lat"), 90.0)
    return None if lon is None or lat is None else (lon, lat)


def _records(
    rows: list[dict[str, str]],
    *,
    kind: str,
    rule: str | None,
    design: dict[str, Any] | None,
) -> list[dict[str, str]]:
    """The export's rows, one per sheet row in the order given: every column
    of :data:`ONE_MAP_COLUMNS` (or :data:`TWO_MAP_COLUMNS`) as text, the
    answers cleaned (:func:`_clean`) and otherwise as they are: the guard
    against a spreadsheet formula is the CSV's (:func:`_csv_text`)."""
    names = sheet._class_names(design)
    out: list[dict[str, str]] = []
    for at, row in enumerate(rows, start=1):
        place = _place(row)
        record = {
            "square": str(at),
            "window_index": (row.get("window_index") or "").strip(),
            "row": (row.get("row") or "").strip(),
            "col": (row.get("col") or "").strip(),
            "lon": sheet.degrees(place[0]) if place else "",
            "lat": sheet.degrees(place[1]) if place else "",
        }
        if kind == "two_map":
            answers = _two_map_answers(row)
        else:
            answers = _one_map_answers(row, design=design, rule=rule, names=names)
        record.update({key: _clean(value) for key, value in answers.items()})
        out.append(record)
    return out


def _csv_text(columns: tuple[str, ...], records: list[dict[str, str]]) -> str:
    """The records as plain CSV (RFC 4180, as the label sheets are): the
    header row, then one row each; no comment line. An answer cell is never
    read as a formula (:func:`_guard`, on :data:`_ANSWER_COLUMNS` only)."""
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer)
    writer.writerow(columns)
    writer.writerows(
        [
            _guard(record[column]) if column in _ANSWER_COLUMNS else record[column]
            for column in columns
        ]
        for record in records
    )
    return buffer.getvalue()


def _agrees(row: dict[str, str], r: int, c: int) -> bool:
    """Whether the row's own ``row`` and ``col`` cells, where it has them,
    are the window's on the map's grid."""
    for cell, value in ((row.get("row"), r), (row.get("col"), c)):
        text = str(cell or "").strip()
        if text and text != str(value):
            return False
    return True


def _outlines(
    design: dict[str, Any] | None, rows: list[dict[str, str]]
) -> list[list[list[float]]] | None:
    """Each row's square as a closed ring of ``[lon, lat]`` positions
    (EPSG:4326), in the rows' order, or ``None`` when the map's grid cannot
    be read here.

    The square is the window's on the map's grid, as the pictures outline
    it: ``patch_px`` map pixels a side from column ``col * patch_px`` and
    row ``row * patch_px`` of the raster the design's scores file names
    (``sheet.picture_sources``, the first one placing the window), its four
    corners reprojected and the first repeated. The grid is read without
    the map's period (``imagery._load_grid``): a map that records none is
    outlined too. ``None`` for a design that names no such file (a Studio
    result's points, inline scores), a raster that cannot be read or lies
    outside the read folders, a grid that does not match the raster, a
    window outside the grid or one whose row's own ``row`` and ``col``
    disagree, or without the ``inferencex`` extra; never raises.
    """
    paths = sheet.picture_sources(design)
    if not paths or not rows:
        return None
    try:
        found = imagery._load_grid(paths[0])
        affine = importlib.import_module("affine")
        warp = importlib.import_module("rasterio.warp")
        to_map = affine.Affine(*found.transform)
        patch, (grid_rows, grid_cols) = found.patch, found.grid
        xs: list[float] = []
        ys: list[float] = []
        for row in rows:
            index = int(str(row.get("window_index") or "").strip())
            r, c = divmod(index, grid_cols)
            if index < 0 or r >= grid_rows or not _agrees(row, r, c):
                return None
            for across, down in _CORNERS:
                x, y = to_map * ((c + across) * patch, (r + down) * patch)
                xs.append(float(x))
                ys.append(float(y))
        lons, lats = warp.transform(found.crs, "EPSG:4326", xs, ys)
    except Exception as exc:  # noqa: BLE001 - no grid here: the centres instead
        # The kind of failure only: never a path or a place.
        logger.info("an export's squares were not outlined (%s)", type(exc).__name__)
        return None
    if not all(math.isfinite(v) for v in (*lons, *lats)):
        return None
    rings: list[list[list[float]]] = []
    for at in range(len(rows)):
        corners = [
            [round(float(lons[i]), 7), round(float(lats[i]), 7)]
            for i in range(4 * at, 4 * at + 4)
        ]
        rings.append([*corners, list(corners[0])])
    return rings


def _property(column: str, value: str) -> Any:
    """A CSV cell as a feature's property: a whole number for the counting
    columns where the cell is one, else the text."""
    if column in _WHOLE_COLUMNS:
        try:
            return int(value)
        except ValueError:
            return value
    return value


def _geojson(
    columns: tuple[str, ...],
    records: list[dict[str, str]],
    rows: list[dict[str, str]],
    rings: list[list[list[float]]] | None,
) -> tuple[str, str]:
    """The records as a GeoJSON FeatureCollection's text, and how its
    geometry was made (:data:`OUTLINE_SQUARE`, :data:`OUTLINE_POINT` or
    :data:`OUTLINE_NONE`).

    One feature per record, in order: its properties the record's columns
    without ``lon`` and ``lat``; its geometry the square's Polygon
    (``rings``), else the row's centre as a Point, else ``null``. Longitude
    first, EPSG:4326, as RFC 7946 has it (no ``crs`` member).
    """
    features: list[dict[str, Any]] = []
    pointed = False
    for at, (record, row) in enumerate(zip(records, rows, strict=True)):
        geometry: dict[str, Any] | None = None
        if rings is not None:
            geometry = {"type": "Polygon", "coordinates": [rings[at]]}
        else:
            place = _place(row)
            if place is not None:
                pointed = True
                geometry = {"type": "Point", "coordinates": [place[0], place[1]]}
        features.append(
            {
                "type": "Feature",
                "geometry": geometry,
                "properties": {
                    column: _property(column, record[column])
                    for column in columns
                    if column not in _PLACE_COLUMNS
                },
            }
        )
    if rings is not None:
        outline = OUTLINE_SQUARE
    else:
        outline = OUTLINE_POINT if pointed else OUTLINE_NONE
    text = json.dumps(
        {"type": "FeatureCollection", "features": features}, ensure_ascii=False
    )
    return text + "\n", outline


def _write_text(path: Path, text: str) -> None:
    """Replace a file whole: a new 0600 file beside it, then moved over it,
    written as given (``newline=""``: the CSV keeps its own line ends). A
    link at its place is refused (``ValueError``), never followed."""
    if path.is_symlink():
        raise ValueError(f"{path.name} is a link; it is not written")
    fd, staged = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as fh:
            fh.write(text)
        os.chmod(staged, 0o600)
        os.replace(staged, path)
    except BaseException:
        Path(staged).unlink(missing_ok=True)
        raise


def _graded_again(real: Path) -> bool:
    """Whether the sheet's graded mark says a later grading graded other
    answers (the mark's own ``changed``, the flag the server withdraws a
    result on). Called under ``sheet._sheet_lock``; false for no mark or one
    that cannot be read."""
    try:
        mark = sheet._read_mark(real)
    except (OSError, ValueError):
        return False
    return bool(mark and mark.get("changed"))


def export_labels(
    labels_csv_path: Any, design_path: Any, *, text: bool = False
) -> dict[str, Any]:
    """Write a labelled sheet's two export files next to it (r5-r R2).

    ``labels_csv_path`` is checked as every sheet tool checks it
    (``sheet.resolve_sheet``: a plan's ``*_to_label.csv`` inside the output
    root) and ``design_path`` must be the design file beside it
    (:func:`_own_design`). Refused, with nothing written, while a square has
    no answer (:data:`NOT_EVERY`, counted as the panels count,
    ``sheet._counted_as_shown``), before the sheet has its result
    (:data:`NO_RESULT`: no graded mark) and once its answers changed after
    that result (:data:`CHANGED`: they are not the ones first graded, or a
    later grading graded other answers). The mark is read under the same
    lock as the rows written, so the rows exported are the ones checked.

    The answer: ``folder`` (the sheet's folder), ``n`` (the squares
    written), ``outline`` (how the GeoJSON's geometry was made) and
    ``files``, the CSV then the GeoJSON, each ``{"kind", "name", "path",
    "mime"}`` and, with ``text`` (a host that saves files), its ``text``.
    Never a place outside those texts. ``ValueError`` with a plain sentence
    for every refusal, :data:`NOT_WRITTEN` when a file cannot be written.
    """
    if not isinstance(labels_csv_path, str):
        raise ValueError(NOT_ITS_DESIGN)
    real = sheet.resolve_sheet(labels_csv_path)
    _own_design(real, design_path)
    design = sheet.design_of(real)[1]
    with sheet._sheet_lock(real):
        secret = sheet._sheet_secret(real)
        header, body, version = sheet._read_stamped(real, secret)
        _, handles = sheet._windows(real, header, body, secret)
        # The mark against the very rows read, under the one lock.
        graded = sheet._graded_field(
            real, version, header=header, body=body, design=design
        )
        again = _graded_again(real)
    kind, rule, counts = sheet._counted_as_shown(header, body, design)
    n, left = counts["n_rows"], counts["n_rows"] - counts["n_labelled"]
    if n < 1 or left > 0:
        raise ValueError(
            NOT_EVERY.format(unit=sheet.window_unit(design), left=left, n=n)
        )
    if graded is None:
        raise ValueError(NO_RESULT)
    if graded["changed_since_first"] or again:
        raise ValueError(CHANGED)
    at_index = header.index("window_index")
    by_index = {int(row[at_index]): row for row in body}
    # The order the sheet showed its squares in: the order of their handles.
    rows = [dict(zip(header, by_index[handles[h]])) for h in sorted(handles)]
    columns = TWO_MAP_COLUMNS if kind == "two_map" else ONE_MAP_COLUMNS
    records = _records(rows, kind=kind, rule=rule, design=design)
    csv_text = _csv_text(columns, records)
    geojson_text, outline = _geojson(columns, records, rows, _outlines(design, rows))
    csv_path, geojson_path = export_paths(real)
    try:
        _write_text(csv_path, csv_text)
        _write_text(geojson_path, geojson_text)
    except Exception as exc:  # noqa: BLE001 - a fixed sentence for the panel
        # Names only in the log: never the path or a place.
        logger.warning("a labels export was not written (%s)", type(exc).__name__)
        raise ValueError(NOT_WRITTEN) from exc
    files: list[dict[str, str]] = []
    for what, path, mime, whole in (
        ("csv", csv_path, CSV_MIME, csv_text),
        ("geojson", geojson_path, GEOJSON_MIME, geojson_text),
    ):
        entry = {"kind": what, "name": path.name, "path": str(path), "mime": mime}
        if text:
            entry["text"] = whole
        files.append(entry)
    return {"folder": str(real.parent), "n": n, "outline": outline, "files": files}


def model_text(result: dict[str, Any]) -> dict[str, Any]:
    """What the model reads of an export's answer: that the files were
    written and their paths (:data:`MODEL_NOTE`), never their contents."""
    files = result.get("files")
    paths = [
        entry["path"]
        for entry in (files if isinstance(files, list) else [])
        if isinstance(entry, dict) and isinstance(entry.get("path"), str)
    ]
    return {"ok": True, "note": MODEL_NOTE, "files": paths}


def offer(path: Any) -> dict[str, str] | None:
    """What a result card needs to offer the export of the sheet it graded:
    ``{"labels_csv_path", "design_path"}``, the two arguments of
    :func:`export_labels`; ``None`` for no sheet, one with no design file
    beside it, one with a square unanswered, one with no graded mark or one
    whose answers changed since its first result (:func:`export_labels`
    would refuse it). Reads only; never raises."""
    if not isinstance(path, str) or not path:
        return None
    try:
        progress = sheet.sheet_progress(path)
    except Exception:  # noqa: BLE001 - unknown: no export offered
        return None
    design_path = progress.get("design_path")
    n, done = progress.get("n_rows"), progress.get("n_labelled")
    if not isinstance(design_path, str) or not design_path:
        return None
    if not isinstance(n, int) or not isinstance(done, int) or n < 1 or done < n:
        return None
    graded = progress.get("graded")
    if not isinstance(graded, dict) or graded.get("changed_since_first"):
        return None
    return {"labels_csv_path": str(progress.get("path")), "design_path": design_path}
