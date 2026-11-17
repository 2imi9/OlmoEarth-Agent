# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""A review set's window cards for the result panel, and the record behind them.

A ranked review set (``olmoearth_review_set``, ``olmoearth_review_set_from_result``)
lists its windows in rank order. Where panels show, the result view gets one
card per listed window (:func:`review_view`); when the scores file the
windows were ranked from records their map (:func:`where_of`: the map
raster's path, the window size, the period and each window's centre, as
``olmoearth_scores_from_file`` and ``olmoearth_scores_from_studio`` write
them), each card opens the window's pictures, the way the label sheet shows
them (``imagery.review_image``). Nothing else is pictured here: inline scores
(a call that passes ``scores``, whatever else it passes, is ranked from
them), a file with no recorded map (the grid of single pixels
``olmoearth_review_set_from_result`` samples), or a record that cannot be
written keep their rows, and the view says why there are no pictures.

The page never gets a path or a place: it names a window by the review's key
(:data:`KEY_PATTERN`, 24 hex characters of a SHA-256 of the tool, the scores
file's real path, its size and mtime, and the listed windows) and the
window's rank. The record (``review_sets/<key>.json`` under
``review_set.write_root()``: the folder 0700, every file 0600, each written
whole) keeps the scores file's real path, size and mtime, so a press on a
file rewritten since (``olmoearth_review_set_from_result`` writes a fixed
name) is refused (:func:`review_row`), and the file must still lie inside
``review_set.read_roots()``.

Round r4-k (K7, K8): a Studio result's sampled points get a record too
(``where`` ``"points"``, the same key recipe; still no pictures), when their
scores file records each point's centre. Two presses then give places, for
the user alone: a square's or point's street map link
(:func:`review_link`, opened in the user's browser) and the look-first list
with its places as a CSV file (:func:`review_csv`: the text for the host's
download, or a file written beside the records). A place never goes into a
page's text, a toast, a log line or the model's text.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import math
import os
import re
import tempfile
import threading
from pathlib import Path
from typing import Any

from olmoearth_agent.mcp_app import sheet
from olmoearth_agent.security.paths import safe_path
from olmoearth_agent.tools import review_set

logger = logging.getLogger(__name__)

#: The format of a review's record (``<key>.json``).
FORMAT = "olmoearth-agent/review-pictures@1"

#: The folder under the write root that holds the review records.
REVIEW_DIR = "review_sets"

#: A review's key: 24 lowercase hexadecimal characters.
KEY_PATTERN = re.compile(r"[0-9a-f]{24}")

#: The longest window caption (a Studio class name is anyone's text).
CAPTION_CHARS = 64

#: The largest record (bytes) read back.
MAX_RECORD_BYTES = 1024 * 1024

#: The block's ``unit`` (:func:`review_view`): the panel's word for one of
#: its windows, as ``sheet.UNIT_SQUARE`` and ``sheet.UNIT_POINT`` say it (a
#: point for a Studio result's sampled pixels, else a square).
UNIT_SQUARE = "square"
UNIT_POINT = "point"

# The sentences the panel shows in place of pictures, and the refusals of a
# press: a closed set, with no path, place or host.

#: A review set ranked from scores passed in the call, not from a file. This
#: sentence and the press's refusals below say "squares": a review set with
#: pictures, or one of inline scores, ranks a scores map's squares (a Studio
#: result's points get :data:`STUDIO_POINTS`); no "window" on a face (the
#: owner's ruling 6 of 13 Nov 2026).
# WORDING: squares
NO_SCORES_FILE = (
    "Pictures need the scores file these squares were ranked from; these scores "
    "were passed inline."
)
# WORDING: squares
#: A scores file that does not record its map (raster, window size and each
#: window's centre).
NO_PLACE_RECORDED = (
    "This scores file does not record the map its squares come from, so no "
    "pictures can be shown."
)
# WORDING: r4p2
#: A scores file that records its map but no period (an ``oe-inferencex
#: from-olmoearth`` run's, a Studio zone whose result declares none).
NO_PERIOD_RECORDED = (
    "This scores file records no period, so the dates to show are unknown."
)
# WORDING: r4p2
#: A review set of a Studio result's sampled points
#: (``olmoearth_review_set_from_result``): its file records no map.
STUDIO_POINTS = "Pictures of a Studio result's sampled points are not built here."
# WORDING: r4p2
#: A scores file outside the read roots, or one that cannot be read now.
NOT_READ_HERE = "This scores file cannot be read here, so no pictures are shown."
# WORDING: held for the reader test
#: The review's record could not be written beside the other records.
NO_RECORD = (
    "The record these pictures need could not be written in the output folder, "
    "so no pictures can be shown."
)
#: A key that names no review here.
# WORDING: squares
REVIEW_UNKNOWN = (
    "This review set is not known here; ask Claude to rank the squares again."
)
#: A rank the review does not list.
# WORDING: squares
NOT_LISTED = "That square is not one this review set lists."
#: A scores file changed (or gone) since the windows were ranked.
# WORDING: squares
REVIEW_CHANGED = (
    "The scores file changed since these squares were ranked; ask Claude to rank "
    "them again."
)

# Round r4-k: the street map (K7) and the CSV of places (K8). Refusals in
# the same closed set: no path, place or host, and no "window" on a face.

#: A listed square or point whose scores file records no usable centre.
# WORDING: r4k K7
NO_CENTRE = (
    "Its scores file does not record where this lies, so no street map can be "
    "opened."
)
#: A list whose scores file does not record every listed place.
# WORDING: r4k K8
NO_CENTRES = (
    "Its scores file does not record where each of these lies, so the list "
    "cannot be saved."
)
#: The CSV could not be written beside the records (save mode).
# WORDING: r4k K8
CSV_NOT_SAVED = "The list could not be saved in the output folder."

#: The CSV's file name: what it is, in the name itself (a reader that opens
#: the file sees it first), then the review's key.
# WORDING: r4k K8
CSV_NAME = "look_first_not_a_sample_{key}.csv"

#: The CSV's note column, on every row: a look-first list is not a sample,
#: so no rate comes from it (r4-i's statistics limits). Never a rate column.
# WORDING: r4k K8
CSV_NOTE = "look-first list, not a random sample; no error rate comes from it"

#: The CSV's header row, first in the file; ``{unit}`` is the block's unit
#: (:data:`UNIT_SQUARE` or :data:`UNIT_POINT`), the column of each one's
#: index on its map's grid.
# WORDING: r4k K8
CSV_HEADER = ("rank", "{unit}", "map_says", "lon", "lat", "note")

#: The tool that ranks a Studio result's sampled points.
STUDIO_TOOL = "olmoearth_review_set_from_result"

#: A record's ``where``: a scores file that records its map (pictures, the
#: street map and the CSV) or a Studio result's points (the street map and
#: the CSV only).
WHERE_MAP = "map"
WHERE_POINTS = "points"

#: Each scores file's windows and centres, by its identity (a few files).
_centres: dict[tuple[str, int, int], tuple[list[int] | None, list[Any]]] = {}
_centres_lock = threading.Lock()


def review_dir() -> Path:
    """The folder of the review records, made (0700) when missing.

    ``OSError`` when it cannot be made, and a ``PathTraversalError`` when it
    resolves outside the write root.
    """
    folder = safe_path(REVIEW_DIR, root=review_set.write_root())
    folder.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(folder, 0o700)
    return folder


def _write_text(path: Path, text: str) -> None:
    """Replace a file whole: a new 0600 file beside it, then moved over it.
    Written as given (``newline=""``: a CSV keeps its own line ends)."""
    fd, staged = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as fh:
            fh.write(text)
        os.chmod(staged, 0o600)
        os.replace(staged, path)
    except BaseException:
        Path(staged).unlink(missing_ok=True)
        raise


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Replace a record whole: a new 0600 file beside it, then moved over it."""
    _write_text(path, json.dumps(payload, sort_keys=True))


def _read_json(path: Path) -> dict[str, Any] | None:
    """A record's contents; ``None`` when it is missing, not ours or unreadable.
    A link is not followed."""
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError:
        return None
    try:
        with os.fdopen(fd, "rb") as fh:
            raw = fh.read(MAX_RECORD_BYTES + 1)
        if len(raw) > MAX_RECORD_BYTES:
            return None
        data = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def inside_roots(real: str) -> bool:
    """Whether a real path lies inside ``review_set.read_roots()``."""
    for root in review_set.read_roots():
        try:
            if os.path.commonpath([root, real]) == root:
                return True
        except ValueError:  # another drive (Windows)
            continue
    return False


def where_of(meta: dict[str, Any], n_rows: int) -> tuple[str, str | None]:
    """``("map", None)`` when a scores file records its map: the map raster's
    path, the window size (``patch_px``), the period (``date_window`` with a
    start and an end) and one centre per row (``centres_lon_lat``); else
    ``("none", reason)``: :data:`NO_PLACE_RECORDED` for a file missing the
    raster, the window size or the centres, :data:`NO_PERIOD_RECORDED` for
    one that has them and no usable period.

    Decided from what the file holds, never from which tool ranked it: the
    file ``olmoearth_review_set_from_result`` writes (a grid of single
    pixels) records no raster, window size or period.
    """
    raster = meta.get("raster")
    patch = meta.get("patch_px")
    window = meta.get("date_window")
    centres = meta.get("centres_lon_lat")
    placed = (
        isinstance(raster, dict)
        and isinstance(raster.get("path"), str)
        and bool(raster["path"])
        and isinstance(patch, int)
        and not isinstance(patch, bool)
        and patch >= 1
        and isinstance(centres, list)
        and len(centres) == n_rows
    )
    if not placed:
        return "none", NO_PLACE_RECORDED
    dated = (
        isinstance(window, dict)
        and bool(window.get("start"))
        and bool(window.get("end"))
    )
    if not dated:
        return "none", NO_PERIOD_RECORDED
    return "map", None


def review_key(
    tool: str, real: str, size: int, mtime_ns: int, indices: list[list[int]]
) -> str:
    """The key of one ranking of one scores file: 24 hex characters of a SHA-256
    of the tool, the file's real path, its size and mtime, and the listed
    windows (``[rank, window_index]`` each)."""
    text = json.dumps([tool, real, size, mtime_ns, indices], separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]


def _int(value: Any) -> int | None:
    """``value`` as an int when it is one (never a bool), else ``None``."""
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _number(value: Any) -> float | None:
    """``value`` as a number when it is one (never a bool), else ``None``."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def caption(row: dict[str, Any]) -> str:
    """A window's caption: what the map says there, never a bare class name.

    ``"Map says: <class>"`` from the row's class name (a Studio or file name:
    printable characters only, at most :data:`CAPTION_CHARS`, shown as text),
    else from its class code; a value read with no class, ``"Map value:
    0.512"`` (three places); else ``""``.
    """
    # WORDING: held for the reader test
    name = row.get("class_name")
    if isinstance(name, str):
        clean = "".join(ch for ch in name if ch.isprintable()).strip()
        if clean:
            return "Map says: " + clean[:CAPTION_CHARS]
    score = _number(row.get("score"))
    if score is not None:
        return f"Map value: {score:.3f}"
    code = _int(row.get("predicted_class"))
    if code is not None:
        return f"Map says: class {code}"
    return ""


def _numbers(row: dict[str, Any]) -> dict[str, Any]:
    """A window's numbers for the panel's closed table: its index, row, column,
    margin and (a Studio result's) score; never a place."""
    out: dict[str, Any] = {
        "window_index": _int(row.get("window_index")),
        "row": _int(row.get("row")),
        "col": _int(row.get("col")),
        "margin": _number(row.get("margin")),
    }
    score = _number(row.get("score"))
    if score is not None:
        out["score"] = score
    return out


def n_review(inner: dict[str, Any]) -> int | None:
    """How many windows the review set holds: ``n_review``, or a Studio
    result's largest budget's."""
    n = _int(inner.get("n_review"))
    if n is not None:
        return n
    budgets = inner.get("budgets")
    if isinstance(budgets, list) and budgets and isinstance(budgets[-1], dict):
        return _int(budgets[-1].get("n_review"))
    return None


def review_view(
    tool: str, inner: dict[str, Any], arguments: dict[str, Any]
) -> dict[str, Any] | None:
    """The view's ``review`` block of a ranked review set, and its record.

    ``{key, where, pictures, why_none, n_review, n_listed, unit, windows}``,
    each window ``{rank, caption, numbers}`` in rank order; ``unit`` what one
    window is, as the panel words it: :data:`UNIT_POINT` for a Studio
    result's (its windows are single pixels at the points of a grid), else
    :data:`UNIT_SQUARE` (r4-d-1 finding R8); a Studio result's
    (``olmoearth_review_set_from_result``) also ``check_request``, its
    ``{result_id, prediction_id}``, for the card's button that asks for the
    Check card of the same map. ``key`` and ``pictures`` only when the scores
    file records its map (``where`` ``"map"``) and the record is written;
    a Studio result's points whose scores file records each point's centre
    get a record and a ``key`` too (``where`` ``"points"``, r4-k K7), for
    the street map and the CSV, but ``pictures`` stays false and
    ``why_none`` :data:`STUDIO_POINTS`. Otherwise ``key`` is ``None`` and
    ``why_none`` says why, by cause: inline scores (:data:`NO_SCORES_FILE`);
    a Studio result's sampled points (:data:`STUDIO_POINTS`, whatever else
    is missing); a file outside the read roots or that cannot be read
    (:data:`NOT_READ_HERE`); a file with no recorded map
    (:data:`NO_PLACE_RECORDED`) or no period (:data:`NO_PERIOD_RECORDED`,
    :func:`where_of`); no record (:data:`NO_RECORD`). The scores file is the
    call's ``scores_path`` (a Studio result's: the one its result names), its
    real path checked inside ``review_set.read_roots()``. The record keeps
    each listed window's caption (``captions``, by rank) for the CSV. No
    path, centre or coordinate is in the block. ``None`` for a result with no
    listed window; never raises.
    """
    rows = [r for r in inner.get("review") or [] if isinstance(r, dict)]
    windows: list[dict[str, Any]] = [
        {"rank": _int(r.get("rank")), "caption": caption(r), "numbers": _numbers(r)}
        for r in rows
        if _int(r.get("rank")) is not None
    ]
    if not windows:
        return None
    studio = tool == STUDIO_TOOL
    block: dict[str, Any] = {
        "key": None,
        "where": "none",
        "pictures": False,
        "why_none": STUDIO_POINTS if studio else NO_PLACE_RECORDED,
        "n_review": n_review(inner),
        "n_listed": len(windows),
        "unit": UNIT_POINT if studio else UNIT_SQUARE,
        "windows": windows,
    }
    result_id = inner.get("result_id")
    if studio and isinstance(result_id, str) and result_id.strip():
        prediction_id = inner.get("prediction_id")
        block["check_request"] = {
            "result_id": result_id.strip(),
            "prediction_id": (
                prediction_id.strip()
                if isinstance(prediction_id, str) and prediction_id.strip()
                else None
            ),
        }
    if arguments.get("scores") is not None:
        # Ranked from the inline rows, whatever scores_path also says.
        block["why_none"] = NO_SCORES_FILE
        return block
    given = inner.get("scores_path") if studio else arguments.get("scores_path")
    if not isinstance(given, str) or not given.strip():
        block["why_none"] = STUDIO_POINTS if studio else NO_SCORES_FILE
        return block
    # Past here a Studio result's points keep STUDIO_POINTS as the reason.
    unread = STUDIO_POINTS if studio else NOT_READ_HERE
    try:
        real = os.path.realpath(given)
        if not inside_roots(real):
            block["why_none"] = unread
            return block
        st = os.stat(real)
        loaded = review_set.load_scores_file(real)
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        logger.info("a review set's scores file could not be read for its cards")
        block["why_none"] = unread
        return block
    where, reason = where_of(loaded.meta, len(loaded.scores))
    # A Studio result's points: no map to picture, but a place per point
    # when the file records each one's centre (r4-k K7).
    points = studio and where != WHERE_MAP and _centred(loaded.meta, len(loaded.scores))
    if where != WHERE_MAP and not points:
        block["why_none"] = STUDIO_POINTS if studio else reason
        return block
    indices = [
        [int(w["rank"]), int(w["numbers"]["window_index"])]
        for w in windows
        if w["numbers"]["window_index"] is not None
    ]
    if len(indices) != len(windows):
        block["why_none"] = STUDIO_POINTS if studio else NO_PLACE_RECORDED
        return block
    key = review_key(tool, real, st.st_size, st.st_mtime_ns, indices)
    record = {
        "format": FORMAT,
        "key": key,
        "tool": tool,
        "scores_path": real,
        "size": st.st_size,
        "mtime_ns": st.st_mtime_ns,
        "result_id": inner.get("result_id"),
        "where": WHERE_POINTS if points else WHERE_MAP,
        "windows": [{"rank": r, "window_index": i} for r, i in indices],
        # What the map says at each, for the CSV's map_says (r4-k K8).
        "captions": {str(w["rank"]): w["caption"] for w in windows},
        # The point path (a Studio result's period) is not built: D1 and D3.
        "period": None,
    }
    try:
        _write_json(review_dir() / f"{key}.json", record)
    except Exception:  # noqa: BLE001 - no record: no pictures, the rows stay
        logger.warning("a review set's record was not written", exc_info=True)
        block["why_none"] = STUDIO_POINTS if points else NO_RECORD
        return block
    if points:
        # The street map and the CSV only: no pictures, the reason stays.
        block.update(key=key, where=WHERE_POINTS)
        return block
    block.update(key=key, where=WHERE_MAP, pictures=True, why_none=None)
    return block


def _centred(meta: dict[str, Any], n_rows: int) -> bool:
    """Whether a scores file records one centre per row (``centres_lon_lat``,
    as ``olmoearth_review_set_from_result`` writes it beside its points)."""
    centres = meta.get("centres_lon_lat")
    return isinstance(centres, list) and len(centres) == n_rows and n_rows > 0


def window_centres(scores_real: str) -> tuple[list[int] | None, list[Any]]:
    """A scores file's windows (``None`` when its rows are the whole grid) and
    its centres (``centres_lon_lat``, one per row), cached by the file's
    identity (path, mtime, size: a few files).

    ``ValueError`` (:data:`NO_CENTRE`) when the file records no centre per
    row; ``OSError`` or ``ValueError`` when it cannot be read.
    """
    st = os.stat(scores_real)
    ident = (scores_real, st.st_mtime_ns, st.st_size)
    with _centres_lock:
        got = _centres.get(ident)
    if got is None:
        loaded = review_set.load_scores_file(scores_real)
        centres = loaded.meta.get("centres_lon_lat")
        if not isinstance(centres, list) or len(centres) != len(loaded.scores):
            raise ValueError(NO_CENTRE)
        got = (loaded.windows, centres)
        with _centres_lock:
            if len(_centres) >= 4:
                _centres.clear()
            _centres[ident] = got
    return got


def window_centre(scores_real: str, index: int) -> tuple[float, float]:
    """One listed window's centre ``(lon, lat)``: the one its scores file
    records for it (its place in ``windows``, or its index when the file has
    none). The pictures (``imagery``) and the street map and CSV here read
    it alike.

    ``ValueError`` (:data:`NO_CENTRE`) when the file records none for it, or
    one that is not a place on Earth (not finite, or beyond 180 and 90
    degrees).
    """
    try:
        windows, centres = window_centres(scores_real)
        place = windows.index(index) if windows is not None else index
        if not 0 <= place < len(centres):
            raise ValueError(NO_CENTRE)
        lon, lat = (float(v) for v in centres[place][:2])
    except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
        raise ValueError(NO_CENTRE) from exc
    if not (
        math.isfinite(lon) and math.isfinite(lat) and abs(lon) <= 180 and abs(lat) <= 90
    ):
        raise ValueError(NO_CENTRE)
    return lon, lat


def _record(key: Any) -> tuple[Path, dict[str, Any]]:
    """A review's record by its key, and its path: ``ValueError``
    (:data:`REVIEW_UNKNOWN`) for a key that is not one, or has no record of
    ours."""
    if not isinstance(key, str) or KEY_PATTERN.fullmatch(key) is None:
        raise ValueError(REVIEW_UNKNOWN)
    try:
        path = review_dir() / f"{key}.json"
    except Exception as exc:  # noqa: BLE001 - no folder: no review here
        raise ValueError(REVIEW_UNKNOWN) from exc
    record = _read_json(path)
    if record is None or record.get("format") != FORMAT or record.get("key") != key:
        raise ValueError(REVIEW_UNKNOWN)
    return path, record


def _scores_unchanged(record: dict[str, Any]) -> str:
    """The record's scores file's real path, checked: ``ValueError`` with
    :data:`REVIEW_UNKNOWN` when the record names none, :data:`REVIEW_CHANGED`
    when the file is gone, outside the read roots, or not the size and mtime
    the key was made from."""
    scores = record.get("scores_path")
    if not isinstance(scores, str):
        raise ValueError(REVIEW_UNKNOWN)
    try:
        real = os.path.realpath(scores)
        st = os.stat(real)
    except (OSError, ValueError) as exc:
        raise ValueError(REVIEW_CHANGED) from exc
    if (
        real != scores
        or not inside_roots(real)
        or st.st_size != record.get("size")
        or st.st_mtime_ns != record.get("mtime_ns")
    ):
        raise ValueError(REVIEW_CHANGED)
    return real


def _placed(record: dict[str, Any], *, points: bool) -> bool:
    """Whether a record's windows can be placed for the caller: a recorded
    map always; a Studio result's points only for the street map and the CSV
    (``points``), never for pictures."""
    where = record.get("where")
    return bool(where == WHERE_MAP or (points and where == WHERE_POINTS))


def review_row(key: Any, rank: Any, *, points: bool = False) -> tuple[Path, str, int]:
    """One listed window of a review, checked: the record's path, the scores
    file's real path and the window's index.

    ``ValueError`` with one fixed sentence: :data:`REVIEW_UNKNOWN` for a key
    that is not one (or has no record of ours), :data:`NOT_LISTED` for a
    rank the review does not list, :data:`NO_PLACE_RECORDED` for a review
    with no recorded map, and :data:`REVIEW_CHANGED` when the scores file is
    gone, outside the read roots, or not the size and mtime the key was made
    from. ``points`` (the street map, :func:`review_link`, r4-k K7) also
    takes a Studio result's points (``where`` ``"points"``); the pictures
    (``imagery.review_image``) never pass it, so points get none.
    """
    if not isinstance(key, str) or KEY_PATTERN.fullmatch(key) is None:
        raise ValueError(REVIEW_UNKNOWN)
    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
        raise ValueError(NOT_LISTED)
    path, record = _record(key)
    listed = [
        w
        for w in record.get("windows") or []
        if isinstance(w, dict) and _int(w.get("rank")) == rank
    ]
    if not listed or _int(listed[0].get("window_index")) is None:
        raise ValueError(NOT_LISTED)
    if not _placed(record, points=points):
        raise ValueError(NO_PLACE_RECORDED)
    real = _scores_unchanged(record)
    return path, real, int(listed[0]["window_index"])


def review_link(key: Any, rank: Any) -> dict[str, str]:
    """One listed square's or point's OpenStreetMap link, at the user's press
    on Street map (r4-k K7): ``{"url"}``.

    Checked as :func:`review_row` checks it (with ``points``: a Studio
    result's points too), its centre the one its scores file records
    (:func:`window_centre`), the link as the sheet's (``sheet.osm_url``).
    ``ValueError`` with :func:`review_row`'s sentences, or :data:`NO_CENTRE`.
    The link goes to the user's browser only.
    """
    _, real, index = review_row(key, rank, points=True)
    lon, lat = window_centre(real, index)
    return {"url": sheet.osm_url(lon, lat)}


#: What a spreadsheet would take as a formula at a cell's start.
_FORMULA_START = ("=", "+", "-", "@", "\t", "\r")


def _cell(text: Any) -> str:
    """A text cell of the CSV: printable characters only, at most 80, and
    never read as a formula (a leading ``'`` then)."""
    clean = "".join(ch for ch in str(text) if ch.isprintable())[:80]
    return "'" + clean if clean.startswith(_FORMULA_START) else clean


def review_places(key: Any) -> list[list[str]]:
    """The look-first list with its places, for the CSV (r4-k K8): the header
    row first (:data:`CSV_HEADER`, its second column named by the unit: a
    Studio result's ``point``, else ``square``), then one row per listed
    square or point in rank order: its rank, its index on its map's grid,
    what the map says there (the card's caption), its centre's lon and lat
    (at most 6 decimals, as the street map link) and :data:`CSV_NOTE`. Never
    a rate column.

    ``ValueError``: :data:`REVIEW_UNKNOWN` for a key of no review here,
    :data:`NO_PLACE_RECORDED` for one with no place recorded,
    :data:`REVIEW_CHANGED` for a scores file changed since the ranking,
    :data:`NO_CENTRES` when a listed centre is missing.
    """
    _, record = _record(key)
    if not _placed(record, points=True):
        raise ValueError(NO_PLACE_RECORDED)
    real = _scores_unchanged(record)
    listed: list[tuple[int, int]] = []
    for entry in record.get("windows") or []:
        if not isinstance(entry, dict):
            continue
        rank, index = _int(entry.get("rank")), _int(entry.get("window_index"))
        if rank is not None and index is not None:
            listed.append((rank, index))
    if not listed:
        raise ValueError(NOT_LISTED)
    listed.sort()
    unit = UNIT_POINT if record.get("tool") == STUDIO_TOOL else UNIT_SQUARE
    captions = record.get("captions")
    said: dict[str, Any] = captions if isinstance(captions, dict) else {}
    rows = [[column.format(unit=unit) for column in CSV_HEADER]]
    try:
        for at, window in listed:
            lon, lat = window_centre(real, window)
            rows.append(
                [
                    str(at),
                    str(window),
                    _cell(said.get(str(at)) or ""),
                    sheet.degrees(lon),
                    sheet.degrees(lat),
                    CSV_NOTE,
                ]
            )
    except ValueError as exc:
        raise ValueError(NO_CENTRES) from exc
    return rows


def review_csv(key: Any, *, save: bool = False) -> dict[str, str]:
    """The look-first list with its places as one CSV file (r4-k K8).

    The rows of :func:`review_places`, written as CSV (RFC 4180, as the
    label sheets are), named :data:`CSV_NAME`. ``save`` false: ``{"name",
    "text"}``, for the host's download (the page passes the text straight
    on; it never enters the page's text or the model's). ``save`` true (a
    host without downloads): the same file written whole under
    ``review_sets/`` (0600, the folder 0700), and ``{"name", "file"}``,
    ``file`` the file's name only (never a path). ``ValueError`` with
    :func:`review_places`'s sentences, or :data:`CSV_NOT_SAVED`.
    """
    rows = review_places(key)
    name = CSV_NAME.format(key=key)
    buffer = io.StringIO(newline="")
    csv.writer(buffer).writerows(rows)
    text = buffer.getvalue()
    if not save:
        return {"name": name, "text": text}
    try:
        _write_text(review_dir() / name, text)
    except Exception as exc:  # noqa: BLE001 - a fixed sentence for the panel
        # Names only in the log: never the path or a place.
        logger.warning("a look-first list was not saved (%s)", type(exc).__name__)
        raise ValueError(CSV_NOT_SAVED) from exc
    return {"name": name, "file": name}
