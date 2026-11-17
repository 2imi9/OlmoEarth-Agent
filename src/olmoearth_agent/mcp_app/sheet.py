# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""The reviewer's label sheet, read and written for the label view.

``olmoearth_plan_label_sample`` writes a sheet beside its design:
``<stem>_to_label.csv``, one row per window to label, in the order to label
them. A one-map sheet has ``map_class``, ``stratum`` and the label columns
``wrong`` and ``reference_class``; a two-map sheet (``design='disagreement'``)
has ``class_a``, ``class_b`` and ``reference_class``. The view shows the rows
and writes the labels the reviewer gives into those columns; every other cell
and the order of the rows and columns stay as they were, so the sheet still
belongs to its design.

A scores map whose design names its classes (``population.class_names``)
and holds the map's class of every window (``population.map_class``) is
labelled with the class seen, blind (label rule ``"seen"``): the view picks
one of the design's codes; the server writes the code into
``reference_class`` and works out ``wrong`` from the design's
``map_class``, never from the sheet's cell, on every row that holds an
answer at each save. Such a row counts as labelled, and the view sees its
answer, only once its ``wrong`` is worked out, as the estimate reads it.
Names that are all a scores file's placeholders (``class_0``,
``class_1``) name no class: such a sheet stays right or wrong.

The view never gets a row (:func:`read_sheet`, the page copy). Each window
reaches it as a handle, ``w`` and 16 base32 characters, keyed by a secret
kept in a hidden file beside the sheet (:func:`_sheet_secret`), with its
answer in the column the view writes (the face: ``seen``, ``class``,
``wrong`` or ``pair``, :func:`_faces`), in the order of the handles: no
window index, place, row, column, order or stratum, and the map's class
only on the face that labels right or wrong. The sheet's id
(:func:`_sheet_id`) names the windows drawn: a sheet drawn again with
other windows refuses an older page (:data:`CHANGED_UNDER_PAGE`). What the
handle hides is the window from the page; it hides nothing from the model,
which reads the plan. A two-map sheet's view may ask for the two maps'
classes at the reviewer's press (``unblind``); a class map's sheet has no
such switch, and no page is ever sent a stratum (the owner's O1 and the
last round's F11, 14 Nov 2026); a window's map link is given at the press
alone (:func:`window_link`). The pictures and the saves name the window by
its handle (:func:`window_row`, :func:`save_answers`); :func:`save_labels`
stays, by window index, for the file route.

One live copy: the page copy and the counts carry the sheet's version, a
keyed digest of the file's bytes under the sheet's secret
(:func:`_version`). A panel's save names the version it read; once the
sheet was saved elsewhere since (another panel, a spreadsheet), the save is
refused (:class:`StaleSheet`) and the file left as it is, so the panel
reads the sheet again and sends its own changes anew: the first save wins.

How far a sheet is labelled (:func:`sheet_progress`) is counts only: the
sheet tool's result and the plan card carry it, never a row, so the view
reads the sheet from the file each time it opens. The projects card lists
the sheets part labelled at the top of the two roots, by counts only, each
with the request that opens it again (:func:`sheets_in_progress`).

Two more hidden files beside a sheet (the owner's extras of 13 Nov 2026),
each mode 0600, written whole, never through a link: ``.<sheet>.rule``, the
labelling rule the user typed on the Check card before its squares were
picked (:func:`set_typed_rule`, refused once any answer is saved): for a map
the owner's rule does not cover, and, by the owner's option A, for a map of
presence and absence that is not of kelp, whose typed rule replaces the
owner's (the page copy and the pictures print it, :func:`labelling_rule`);
and ``.<sheet>.graded``, the version first and last graded
(:func:`record_grading`), bound to the design's content (a digest of its
``to_label``) and read as no mark while the sheet holds no answer, so a fresh
sheet planned again under the same name does not inherit it, so a reopened
sheet says its answers should not change and the estimate says when they did
(:func:`graded_state`). A save that empties a graded sheet (a column cleared
from the panel, or every answer cleared one by one) keeps the mark, marked
changed and ``blanked`` at the version written, so answers given again after
the result was seen grade as changed (the lead's ruling Q1 of 13 Nov 2026);
whether the answers changed is told by a digest of the label cells kept at the
first grading, not by the file's bytes (Q2), so a spreadsheet's re-save with
no answer changed is no change. Once graded, :func:`graded_squares` gives the
result card its counts of the classes seen and of where the map agreed, its
Cannot tell squares after them (the owner's O2 of 14 Nov 2026), and
:func:`class_counts` the same counts with Cannot tell counted apart.
:func:`result_holds_answers` says whether any sheet of a Studio result already
holds an answer (the Check card's settled rule).

Only a file named ``*_to_label.csv`` under the tools' write root or the
workspace root is read or written, so a path a tool result or the model
passes cannot reach any other file.
"""

from __future__ import annotations

import base64
import contextlib
import csv
import hashlib
import hmac
import io
import json
import math
import os
import re
import secrets
import stat
import tempfile
import threading
import unicodedata
from collections.abc import Iterator
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

try:  # POSIX: a lock another process (a second server) also honours
    import fcntl
except ImportError:  # pragma: no cover - Windows has no fcntl
    fcntl = None  # type: ignore[assignment]

from olmoearth_agent.reporting.figure_file import read_figure_svg
from olmoearth_agent.security.paths import workspace_root
from olmoearth_agent.tools.review_set import read_roots, slug, write_root

#: The suffix of every sheet a plan writes.
SHEET_SUFFIX = "_to_label.csv"

#: The largest sheet (bytes) the view reads; a plan's sheet is far smaller.
MAX_SHEET_BYTES = 5 * 1024 * 1024

#: The largest design file (bytes) read for the sheet's class list.
MAX_DESIGN_BYTES = 20 * 1024 * 1024

#: The longest class name accepted as a label.
MAX_LABEL_CHARS = 200

#: The label of a window the reviewer could not judge ("cannot tell"): in
#: ``reference_class`` on a two-map sheet and on a class map's sheet that
#: takes it (the last round's S4), in ``wrong`` on a one-map sheet labelled
#: right or wrong (the estimate bounds it both ways).
UNJUDGED = "?"

#: Why ``?`` is refused in ``reference_class`` (:func:`_label_value`): only a
#: two-map sheet and a class map's sheet that takes Cannot tell take it there.
#: Said to Claude (the file route), not on a face.
# WORDING: r4h S4 (model-facing)
UNJUDGED_NOT_HERE = (
    "'?' (cannot tell) is not taken in reference_class on this sheet: only a "
    "two-map sheet and a class map's sheet that takes Cannot tell take it there"
)

#: The population score kinds whose labels are the classes seen, as the map
#: writes them (``tools.estimation.CLASS_LABELLED``, which a test holds this
#: to): a class map, with top-k scores or without. Any other one-map design
#: (a scores map: the confidence or random design on scores) is graded from
#: ``wrong`` on every window.
CLASS_LABELLED = ("class_map", "topk_score")

#: Class names that make a map one of presence and absence: a seen sheet's
#: answers put them first and second, and the pictures' labelling rule
#: (``imagery.reading_rule``) is for such a map.
PRESENT_NAMES = frozenset({"present", "presence"})
ABSENT_NAMES = frozenset({"absent", "absence"})

#: A class name a scores file gives a class it was not told the name of
#: (``class_0``, ``class_1``: ``tools.scores_file``): a design whose names are
#: all of this form names no class, so its sheet stays right or wrong.
PLACEHOLDER_NAME = re.compile(r"class_\d+", re.IGNORECASE)

#: The words of the answer ``?`` on a seen sheet.
CANNOT_TELL = "Cannot tell"

#: The suffix of the hidden file beside a sheet that holds its handles'
#: secret (``.<sheet name>.handles``): neither ``.csv`` nor ``.json``, so no
#: tool that reads a path the model chooses can take it.
HANDLES_SUFFIX = ".handles"

#: The faces a sheet is labelled on: the column the view writes
#: (:func:`_faces`). ``seen``: the class seen on a named scores map, as a
#: code; ``class``: the class seen on a class map; ``wrong``: right or
#: wrong; ``pair``: the class seen on a two-map sheet.
FACES = ("seen", "class", "wrong", "pair")

#: The column each face writes.
FACE_COLUMN = {
    "seen": "reference_class",
    "class": "reference_class",
    "wrong": "wrong",
    "pair": "reference_class",
}

#: Why a handle is refused: it names no window of the sheet as it is now. One
#: fixed sentence for a sheet of squares or of points (``imagery.NOT_ON_SHEET``
#: is it too): no "window" on a face (the owner's ruling 6 of 13 Nov 2026).
# WORDING: squares
UNKNOWN_HANDLE = "That square or point is not on this sheet."

#: Why a save, a picture or a link from a panel of an older server is refused
#: (it names a window by its index).
OLDER_PANEL = "This panel is older than the server: open the sheet again."

#: Why the right-or-wrong face is refused on a class map holding a class seen.
FACE_WRONG_REFUSED = (
    "Right or wrong shows the map's class, so it is offered only before any "
    "class seen is saved; clear the classes seen first."
)

#: Why the class-seen face is refused on a class map holding right or wrong.
FACE_CLASS_REFUSED = (
    "This sheet holds right or wrong labels; clear them before labelling the "
    "class seen."
)

#: Why a value is refused on a face that takes listed answers only.
NOT_AN_ANSWER = "{value} is not one of this sheet's answers: {labels}."

#: Why a window's map link is refused: its row has no place to show. Only a
#: sheet of squares can lack one (inline scores, or a scores file without
#: centres); a Studio result's points always carry theirs. No "window" on a
#: face (the owner's ruling 6 of 13 Nov 2026).
# WORDING: squares
NO_PLACE = "This square has no place on the sheet to show on a map."

#: Why a column is refused for clearing.
NOT_CLEARED = "That column is not cleared from this panel."

#: Said with a named scores map's sheet already labelled right or wrong.
RULE_NOTE = (
    "This sheet was started as right or wrong, so it stays that way; the map's "
    "class is shown."
)

#: Why a save from a panel is refused when the sheet's label rule is no
#: longer the one the panel drew.
CHANGED_UNDER_PAGE = "This sheet changed since the panel opened it; open it again."

#: Why a panel's save is refused when the sheet was saved since the panel
#: read it (:class:`StaleSheet`): the panel reads it again and keeps going.
STALE_SHEET = (
    "This sheet was saved elsewhere (another panel or a spreadsheet) since this "
    "panel read it."
)

#: Why a version that is not text is refused.
BAD_VERSION = "version is the sheet's version as the panel read it"

#: A sheet's version as :func:`_version` writes it: ``v`` and 16 base32
#: characters (lower case).
_VERSION_RE = re.compile(r"v[a-z2-7]{16}")

#: The hidden file beside a sheet that marks it graded (``.<sheet>.graded``,
#: :func:`record_grading`; the owner's ruling of 13 Nov 2026, extra 2), and
#: its format tag.
GRADED_SUFFIX = ".graded"
GRADED_FORMAT = "olmoearth-agent/sheet-graded@1"

#: The hidden file beside a sheet that holds the labelling rule the user typed
#: on the Check card before its squares were picked (``.<sheet>.rule``,
#: :func:`set_typed_rule`; the owner's ruling of 13 Nov 2026, extra 1), and its
#: format tag.
RULE_SUFFIX = ".rule"
RULE_FORMAT = "olmoearth-agent/sheet-rule@1"

#: The bounds of a typed labelling rule, in characters (:func:`clean_rule`).
RULE_MIN_CHARS = 10
RULE_MAX_CHARS = 300

#: The largest hidden file beside a sheet that is read (a mark or a rule).
HIDDEN_MAX_BYTES = 64 * 1024

#: Why a typed labelling rule is refused (:func:`clean_rule`).
# WORDING: r4p2
RULE_TEXT = (
    f"The labelling rule is one line of {RULE_MIN_CHARS} to {RULE_MAX_CHARS} "
    "characters."
)

#: Why a typed rule is refused once the sheet holds an answer: it is fixed
#: before any square is checked (:func:`set_typed_rule`).
# WORDING: squares
RULE_AFTER_ANSWER = (
    "This sheet already holds answers: its labelling rule is fixed before any "
    "square is answered."
)

#: Said with a rule the user typed before the draw (the owner's rule says
#: ``imagery.RULE_SCOPE``): ``{date}`` is the day it was typed.
# WORDING: squares
TYPED_RULE_SCOPE = "Typed by you before the squares were picked, {date}."

#: Why a ``typed_at`` is refused: when the rule was typed, as text.
BAD_MOMENT = "typed_at is when the rule was typed, as ISO 8601 text"


class StaleSheet(ValueError):
    """A panel's save that names a version of the sheet older than the file.

    The sheet was saved elsewhere (another panel, a spreadsheet) since the
    panel read it, so the file is left as it is (:data:`STALE_SHEET`). Unlike
    :data:`CHANGED_UNDER_PAGE`, a read of the sheet and a new save can
    succeed: the panel takes what was saved there and sends its own changes
    again.
    """


#: Why a ``wrong`` cell is refused on a seen sheet.
SEEN_NO_WRONG = (
    "On this sheet right or wrong is worked out from the class seen; send "
    "reference_class only."
)

#: Held while a sheet is read and written, so two saves in this process cannot
#: interleave; :func:`_sheet_lock` adds a file lock for other processes.
_LOCK = threading.Lock()


@contextlib.contextmanager
def _sheet_lock(real: Path) -> Iterator[None]:
    """Hold the sheet for a read and write: this process's lock and a file lock.

    The file lock is on a hidden file beside the sheet (``.<sheet>.lock``),
    not on the sheet itself, which a save replaces with a new file.
    """
    with _LOCK:
        if fcntl is None:  # pragma: no cover - Windows
            yield
            return
        lock_path = real.with_name("." + real.name + ".lock")
        with open(lock_path, "a", encoding="utf-8") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)


def resolve_sheet(path: str) -> Path:
    """The sheet's real path; ``ValueError`` unless it is a plan's sheet under a root.

    A plan's sheet is a regular file named ``*_to_label.csv`` whose real path
    lies under :func:`~olmoearth_agent.tools.review_set.write_root` or the
    workspace root, of at most :data:`MAX_SHEET_BYTES`.
    """
    if not isinstance(path, str) or not path or "\x00" in path:
        raise ValueError("labels_csv_path must be the plan's _to_label.csv path")
    real = Path(os.path.realpath(path))
    roots = (write_root(), workspace_root())
    inside = any(real.is_relative_to(os.path.realpath(root)) for root in roots)
    if not inside or not real.name.endswith(SHEET_SUFFIX):
        raise ValueError(
            "Only a plan's *_to_label.csv under the agent's output folder can be "
            "opened; pass labels_csv_path from olmoearth_plan_label_sample"
        )
    if not real.is_file():
        raise ValueError(f"{real.name} does not exist; plan the sample again")
    if real.stat().st_size > MAX_SHEET_BYTES:
        raise ValueError(f"{real.name} is larger than a plan's sheet can be")
    return real


def _read(real: Path) -> tuple[list[str], list[list[str]]]:
    """The sheet's header and rows (each padded to the header's width)."""
    return _parse(real, real.read_bytes())


def _read_stamped(real: Path, secret: bytes) -> tuple[list[str], list[list[str]], str]:
    """The sheet's header and rows, and its version, from one read of its bytes.

    The version is :func:`_version` of the bytes read, so it names exactly
    what was parsed. Called under :func:`_sheet_lock`, with the sheet's
    secret (:func:`_sheet_secret`).
    """
    raw = real.read_bytes()
    header, body = _parse(real, raw)
    return header, body, _version(secret, raw)


def _parse(real: Path, raw: bytes) -> tuple[list[str], list[list[str]]]:
    """The header and rows of a sheet's bytes, as a text read parsed them.

    UTF-8, with ``\\r\\n`` and ``\\r`` turned into ``\\n`` (the universal
    newlines of a text read), then read as CSV; each row padded to the
    header's width.
    """
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(
            f"{real.name} is not UTF-8; a spreadsheet saved it in another "
            "encoding. Save it again as CSV UTF-8, comma-separated"
        ) from exc
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    rows = [row for row in csv.reader(io.StringIO(text, newline="")) if row]
    if not rows:
        raise ValueError(f"{real.name} is empty")
    header = rows[0]
    if "window_index" not in header or "reference_class" not in header:
        raise ValueError(
            f"{real.name} has no window_index and reference_class columns; a "
            "spreadsheet may have saved it with ';' or a tab. Save it again as "
            "CSV UTF-8, comma-separated"
        )
    body = [row + [""] * (len(header) - len(row)) for row in rows[1:]]
    return header, body


def _design_path(real: Path) -> Path:
    """Where the design file beside a sheet is (``<stem>.json``), read or not."""
    return real.with_name(real.name[: -len(SHEET_SUFFIX)] + ".json")


def _design_of(real: Path) -> tuple[Path | None, dict[str, Any] | None]:
    """The design file beside the sheet (``<stem>.json``) and its contents."""
    design = _design_path(real)
    try:
        if not design.is_file() or design.stat().st_size > MAX_DESIGN_BYTES:
            return None, None
        data = json.loads(design.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None, None
    return (design, data) if isinstance(data, dict) else (None, None)


def design_of(real: Path) -> tuple[Path | None, dict[str, Any] | None]:
    """The design file beside a resolved sheet and its contents (the window
    pictures find the map from it)."""
    return _design_of(real)


#: The word the sheet's face says for one of its windows (the page copy's
#: ``unit``, :func:`window_unit`; the lead's ruling of 13 Nov 2026, round 4
#: pass 2, contract C1): ``"square"`` for a patch of a scores file's map,
#: ``"point"`` for a single pixel a Studio result was sampled at.
# WORDING: squares
UNIT_SQUARE = "square"
# WORDING: squares
UNIT_POINT = "point"


def _population_source(design: dict[str, Any] | None) -> dict[str, Any]:
    """The design's ``population.source``, or ``{}`` when it has none."""
    population = (design or {}).get("population")
    source = population.get("source") if isinstance(population, dict) else None
    return source if isinstance(source, dict) else {}


#: The format tag of a scores file the tools write
#: (``tools.review_set.SampledScores.to_file``, ``tools.scores_file``).
SCORES_FORMAT = "olmoearth-agent/scores@1"

#: The largest scores file read to tell whether it holds a Studio result's
#: sampled points (:func:`_points_file`): such a file holds one row per point
#: of a grid of at most 16 x 16, far smaller; a larger file is not one, and is
#: not read for it.
POINTS_FILE_MAX_BYTES = 1024 * 1024

#: The keys only a scores file read off a map raster carries
#: (``tools.scores_file``: the window size, the raster, how it was made); a
#: Studio result's sampled points file carries none of them.
_RASTER_KEYS = ("patch_px", "raster", "provenance")

#: What :func:`_points_result` found for each scores file read, by the file's
#: identity (its real path, write stamp and size): the result it names, or
#: ``None``. A few files; cleared when full.
_points_found: dict[tuple[str, int, int], str | None] = {}
_points_lock = threading.Lock()
_POINTS_KEPT = 32


def _points_result(real: str) -> str | None:
    """The Studio result a scores file holds the sampled points of, read off
    the file's own top level (never a raster), or ``None`` when it is not such
    a file (:func:`_points_file`).

    Such a file (``olmoearth_review_set_from_result``'s saved file,
    ``tools.review_set.SampledScores.to_file``) is an object in the tools'
    scores format (:data:`SCORES_FORMAT`) that names its result
    (``result_id``), gives each point's centre (``centres_lon_lat``) and a
    square grid (``[n, n]``), and carries no window size, raster or
    provenance (:data:`_RASTER_KEYS`), which every file read off a map raster
    has. A file larger than :data:`POINTS_FILE_MAX_BYTES`, or one that is not
    UTF-8 JSON, is not one.
    """
    try:
        with open(real, "rb") as fh:
            raw = fh.read(POINTS_FILE_MAX_BYTES + 1)
        data = json.loads(raw.decode("utf-8"))
    except (OSError, ValueError, RecursionError):
        return None
    if len(raw) > POINTS_FILE_MAX_BYTES or not isinstance(data, dict):
        return None
    if data.get("format") != SCORES_FORMAT:
        return None
    if any(key in data for key in _RASTER_KEYS):
        return None
    grid = data.get("grid")
    if not (
        isinstance(grid, list)
        and len(grid) == 2
        and all(isinstance(g, int) and not isinstance(g, bool) for g in grid)
        and grid[0] == grid[1]
        and grid[0] > 0
    ):
        return None
    if not isinstance(data.get("centres_lon_lat"), list):
        return None
    result_id = data.get("result_id")
    if not isinstance(result_id, str) or not result_id.strip():
        return None
    return result_id.strip()


def _points_file(part: Any) -> bool:
    """Whether a source's scores file holds a Studio result's sampled points
    (``olmoearth_review_set_from_result``'s saved file): single pixels at the
    points of a grid over the result, with no raster to picture them from
    (the lead's ruling of 13 Nov 2026, round 4, finding R3).

    The source first, from the design alone: it names a scores file and a
    Studio result (``result_id``, which the plan copies from such a file) and
    no provenance or Studio-files block (``provenance``, ``studio``), which
    the plan copies from every file read off a map raster, a Check card's
    zone among them: such a source is a square's, and no file is opened.
    Then the file's own top level (:func:`_points_result`), only for a
    ``.json`` file inside the read roots of at most
    :data:`POINTS_FILE_MAX_BYTES`, naming the same result. A file that cannot
    be read is not one: the sheet is then a square's, as before. Read once
    per file identity (:data:`_points_found`).
    """
    if not isinstance(part, dict):
        return False
    path, result_id = part.get("scores_path"), part.get("result_id")
    if not isinstance(path, str) or not path or not isinstance(result_id, str):
        return False
    if not result_id.strip() or "provenance" in part or "studio" in part:
        return False
    try:
        real = os.path.realpath(path)
        inside = any(os.path.commonpath([root, real]) == root for root in read_roots())
        st = os.stat(real)
    except (OSError, ValueError):
        return False
    if not inside or not real.endswith(".json") or not stat.S_ISREG(st.st_mode):
        return False
    if st.st_size > POINTS_FILE_MAX_BYTES:
        return False
    key = (real, int(st.st_mtime_ns), int(st.st_size))
    with _points_lock:
        known = key in _points_found
        named = _points_found.get(key)
    if not known:
        named = _points_result(real)
        with _points_lock:
            if len(_points_found) >= _POINTS_KEPT:
                _points_found.clear()
            _points_found[key] = named
    return named is not None and named == result_id.strip()


def picture_sources(design: dict[str, Any] | None) -> list[Any]:
    """The scores files the window pictures place a sheet's windows from,
    read off the design's structure (no raster is opened).

    ``[scores_path]`` for a design drawn from a scores file
    (``population.source.scores_path``); ``[map_a's, map_b's]`` for a
    two-map design whose two maps each name one; else ``[]``: a plan drawn
    from a Studio result (``result_id``), from a scores file of a Studio
    result's sampled points (:func:`_points_file`, which alone reads the
    small file's top level), from inline scores, or with no design file.
    ``imagery._map_of`` reads the map from these, the first one placing the
    window, so :func:`pictures_possible` and the pictures agree.
    """
    source = _population_source(design)
    if source.get("scores_path"):
        return [] if _points_file(source) else [source["scores_path"]]
    a, b = source.get("map_a"), source.get("map_b")
    if (
        isinstance(a, dict)
        and isinstance(b, dict)
        and a.get("scores_path")
        and b.get("scores_path")
        and not _points_file(a)
        and not _points_file(b)
    ):
        return [a["scores_path"], b["scores_path"]]
    return []


def pictures_possible(design: dict[str, Any] | None) -> bool:
    """Whether a sheet's windows can have pictures at all (the page copy's
    ``pictures_possible``): its design names the scores file(s) that place
    them (:func:`picture_sources`), none of them a Studio result's sampled
    points (:func:`_points_file`). Read off the structure (and that small
    file's top level), never the raster: a sheet for which this is true can
    still be refused by the pictures themselves (no period, a raster that
    cannot be read)."""
    return bool(picture_sources(design))


def window_unit(design: dict[str, Any] | None) -> str:
    """What one of the sheet's windows is, as its face says it: :data:`UNIT_POINT`
    or :data:`UNIT_SQUARE` (the page copy's ``unit``).

    A point when the design's population source names a Studio result
    (``result_id``) and no scores file, or a scores file of that result's
    sampled points (:func:`_points_file`: ``olmoearth_review_set_from_result``'s
    saved file), for one map, or for either map of a two-map design (whose
    two maps are both Studio results or both scores files): such a plan's
    windows are single pixels at the points of a grid over the result, of
    which nothing else was read. A square otherwise: a patch of a scores
    file's map, as the pictures outline it, a Check card's zone among them.
    """
    source = _population_source(design)

    def by_result(part: Any) -> bool:
        """A source of a Studio result's points: by ``result_id`` with no
        scores file, or with a scores file of its sampled points."""
        if not isinstance(part, dict) or not part.get("result_id"):
            return False
        return not part.get("scores_path") or _points_file(part)

    if by_result(source):
        return UNIT_POINT
    if by_result(source.get("map_a")) or by_result(source.get("map_b")):
        return UNIT_POINT
    return UNIT_SQUARE


#: The months as a period is written ("1 Apr 2024"): fixed words, never the
#: locale's (the server's result card writes the same).
_MONTHS = (
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
)

#: The largest scores file read for the map's period (:func:`_period_text`).
PERIOD_FILE_MAX_BYTES = MAX_DESIGN_BYTES

#: What :func:`_scores_period` found for each scores file read, by the file's
#: identity (its real path, write stamp and size). A few files; cleared when
#: full.
_periods_found: dict[tuple[str, int, int], str | None] = {}
_periods_lock = threading.Lock()
_PERIODS_KEPT = 32


def _day_text(value: Any) -> str | None:
    """A date or time as given (``2024-04-01...``) as ``"1 Apr 2024"``, or
    ``None``."""
    if not isinstance(value, str) or len(value) < 10:
        return None
    try:
        day = date.fromisoformat(value[:10])
    except ValueError:
        return None
    return f"{day.day} {_MONTHS[day.month - 1]} {day.year}"


def _window_text(window: Any) -> str | None:
    """A ``date_window`` (its start and end, as given) as ``"1 Apr 2024 to 30
    Jun 2024"``, or ``None`` when either end cannot be read."""
    if not isinstance(window, dict):
        return None
    start, end = _day_text(window.get("start")), _day_text(window.get("end"))
    if start is None or end is None:
        return None
    return f"{start} to {end}"


def _scores_period(path: str) -> str | None:
    """The period a scores file records (its top level's ``date_window``,
    :func:`_window_text`), or ``None``: only a ``.json`` file inside the read
    roots, a regular file of at most :data:`PERIOD_FILE_MAX_BYTES`, read once
    per file identity (:data:`_periods_found`). A file that cannot be read
    records none."""
    try:
        real = os.path.realpath(path)
        inside = any(os.path.commonpath([root, real]) == root for root in read_roots())
        st = os.stat(real)
    except (OSError, ValueError):
        return None
    if not inside or not real.endswith(".json") or not stat.S_ISREG(st.st_mode):
        return None
    if st.st_size > PERIOD_FILE_MAX_BYTES:
        return None
    key = (real, int(st.st_mtime_ns), int(st.st_size))
    with _periods_lock:
        known = key in _periods_found
        found = _periods_found.get(key)
    if known:
        return found
    try:
        with open(real, "rb") as fh:
            data = json.loads(fh.read(PERIOD_FILE_MAX_BYTES + 1).decode("utf-8"))
    except (OSError, ValueError, RecursionError):
        data = None
    found = _window_text(data.get("date_window")) if isinstance(data, dict) else None
    with _periods_lock:
        if len(_periods_found) >= _PERIODS_KEPT:
            _periods_found.clear()
        _periods_found[key] = found
    return found


def _period_text(design: dict[str, Any] | None) -> str | None:
    """The period the sheet's map covers, as the class face names it (the
    page copy's ``period_text``, reader test 2's E/sheet-3): ``"1 Apr 2024 to
    30 Jun 2024"`` from the design's ``population.source.date_window``, else
    from the top level of the scores file that source names
    (:func:`_scores_period`), else ``None`` (the face then says to ask whoever
    made the map). Never a window's place or class."""
    source = _population_source(design)
    said = _window_text(source.get("date_window"))
    if said is not None:
        return said
    path = source.get("scores_path")
    if not isinstance(path, str) or not path:
        return None
    return _scores_period(path)


def _str_list(value: Any) -> list[str]:
    """``value`` as a list of strings, or ``[]`` when it is not a list."""
    return [str(v) for v in value] if isinstance(value, list) else []


def _classes(
    design: dict[str, Any] | None, rows: list[dict[str, str]], kind: str
) -> list[str]:
    """The classes the design names, in its order, then any the sheet shows besides.

    A disagreement design keeps its classes at its top level (``classes``);
    a class map's are its population's (``population.classes``). The classes
    the rows show besides are sorted by name, never in the rows' order, so
    the list tells which classes the sample holds, not where.
    """
    design = design or {}
    population = design.get("population")
    population = population if isinstance(population, dict) else {}
    listed = _str_list(design.get("classes")) if kind == "two_map" else []
    listed = listed or _str_list(population.get("classes"))
    columns = ("class_a", "class_b") if kind == "two_map" else ("map_class",)
    seen = list(dict.fromkeys(listed))
    extra = {
        value
        for row in rows
        for column in columns
        if (value := (row.get(column) or "").strip()) and value not in seen
    }
    return seen + sorted(extra, key=lambda c: (c.lower(), c))


def _class_names(design: dict[str, Any] | None) -> dict[str, str]:
    """A scores map's class names by code (``population.class_names``, from
    its scores file): the view shows a name where the sheet writes ``0`` or
    ``1``. An entry that is no decimal code and non-empty name is left out."""
    population = (design or {}).get("population")
    raw = population.get("class_names") if isinstance(population, dict) else None
    pairs: list[tuple[Any, Any]] = []
    if isinstance(raw, dict):
        pairs = list(raw.items())
    elif isinstance(raw, list):
        pairs = list(enumerate(raw))
    return {
        str(code).strip(): name.strip()[:MAX_LABEL_CHARS]
        for code, name in pairs
        if str(code).strip().isdecimal() and isinstance(name, str) and name.strip()
    }


#: Why a certificate plan's sheet takes no Cannot tell.
_NO_UNJUDGED = (
    "this plan is a certificate plan (seed_from_map): the certificate and the "
    "certified zone need 1 or 0 on every window, so Cannot tell (?) is not taken"
)


def takes_unjudged(design: dict[str, Any] | None) -> bool:
    """Whether the sheet takes Cannot tell (``?``) in ``wrong``: every plan's
    but a certificate plan's (``certificate.seed_rule == "from_map"``), whose
    labels the certificate fixes with a 1 or a 0 per window."""
    block = (design or {}).get("certificate")
    return not (isinstance(block, dict) and block.get("seed_rule") == "from_map")


def label_rule(design: dict[str, Any] | None, kind: str) -> str | None:
    """How the estimate reads this sheet: ``"class"``, ``"wrong"``, or ``None``.

    ``"class"``: the reviewer writes the class seen (``reference_class``),
    as for a two-map sheet and a class map's (:data:`CLASS_LABELLED`), whose
    ``wrong`` may be filled instead, on every window, for the error rate
    alone. ``"wrong"``: a scores map's, graded from ``wrong`` on every
    window. ``None``: no design file beside the sheet to tell.
    """
    if kind == "two_map":
        return "class"
    if design is None:
        return None
    population = design.get("population")
    score_kind = population.get("score_kind") if isinstance(population, dict) else None
    return "class" if score_kind in CLASS_LABELLED else "wrong"


def _kind(header: list[str]) -> str:
    """``two_map`` for a sheet with ``class_a`` and ``class_b``, else ``one_map``."""
    return "two_map" if {"class_a", "class_b"} <= set(header) else "one_map"


def _seen_codes(design: dict[str, Any] | None) -> list[str] | None:
    """The codes a reviewer answers with on a named scores map, or ``None``.

    A scores map (a ``population.score_kind`` not in :data:`CLASS_LABELLED`)
    whose design names at least two classes by code (``class_names``, each
    code written as ``int`` writes it) and holds the map's class of every
    window (``population.map_class``, a list of whole numbers, ``-1`` where a
    window has none), every class on the map named. In order: the name like
    present first, the name like absent second, then the rest by code.
    Names that are all a scores file's placeholders (:data:`PLACEHOLDER_NAME`:
    ``class_0``, ``class_1``) name no class: ``None``, so the sheet stays
    right or wrong.
    """
    population = (design or {}).get("population")
    if not isinstance(population, dict):
        return None
    if population.get("score_kind") in CLASS_LABELLED:
        return None
    names = {
        code: name
        for code, name in _class_names(design).items()
        if code == str(int(code))
    }
    on_map = population.get("map_class")
    if len(names) < 2 or not isinstance(on_map, list):
        return None
    if all(PLACEHOLDER_NAME.fullmatch(name) for name in names.values()):
        return None
    if any(not isinstance(c, int) or isinstance(c, bool) for c in on_map):
        return None
    if not {str(c) for c in on_map if c >= 0} <= set(names):
        return None

    def place(code: str) -> tuple[int, int]:
        """Present-like first, absent-like second, then by code."""
        name = names[code].lower()
        first = 0 if name in PRESENT_NAMES else 1 if name in ABSENT_NAMES else 2
        return first, int(code)

    return sorted(names, key=place)


def _started_right_or_wrong(header: list[str], body: list[list[str]]) -> bool:
    """Whether a row has ``wrong`` filled and ``reference_class`` blank."""
    at_wrong = header.index("wrong")
    at_ref = header.index("reference_class")
    return any(row[at_wrong].strip() and not row[at_ref].strip() for row in body)


def _rule(
    design: dict[str, Any] | None,
    kind: str,
    header: list[str],
    body: list[list[str]],
) -> str | None:
    """How the sheet is labelled now: :func:`label_rule`'s rule, or ``"seen"``.

    ``"seen"``: a named scores map (:func:`_seen_codes`), whose sheet has a
    ``wrong`` column and no row labelled right or wrong alone (``wrong``
    filled, ``reference_class`` blank). A sheet already started that way
    stays ``"wrong"``: nothing is read anew.
    """
    rule = label_rule(design, kind)
    if (
        rule == "wrong"
        and _seen_codes(design) is not None
        and "wrong" in header
        and not _started_right_or_wrong(header, body)
    ):
        return "seen"
    return rule


def sheet_rule(path: str) -> str | None:
    """How a plan's sheet is labelled now (:func:`_rule`), read from the files."""
    real = resolve_sheet(path)
    with _sheet_lock(real):
        header, body = _read(real)
    return _rule(_design_of(real)[1], _kind(header), header, body)


def _secret_path(real: Path) -> Path:
    """The hidden file beside the sheet that holds its handles' secret."""
    return real.with_name("." + real.name + HANDLES_SUFFIX)


def _read_secret(place: Path) -> str | None:
    """The secret file's text, never through a link; ``None`` when it is missing.

    ``ValueError`` when a link stands at its place.
    """
    try:
        fd = os.open(place, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except FileNotFoundError:
        return None
    except OSError as exc:  # ELOOP: a link at the secret's place
        if place.is_symlink():
            raise ValueError(
                f"{place.name} is a link; the sheet's handles cannot be kept"
            ) from exc
        raise
    with os.fdopen(fd, "r", encoding="ascii", errors="replace") as fh:
        return fh.read(256)


def _sheet_secret(real: Path) -> bytes:
    """The sheet's handles' secret, made the first time; called under :func:`_sheet_lock`.

    64 hex characters in ``.<sheet name>.handles`` beside the sheet, mode
    0600, made with ``O_EXCL`` and never written through a link (a link at
    its place is refused). A malformed file is replaced (a new file, then
    moved over it): pages opened before then fail safely on their sheet id.
    A deleted file gives a new secret, which refuses those pages the same way.
    """
    place = _secret_path(real)
    text = _read_secret(place)
    if text is None:
        made = secrets.token_hex(32)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        try:
            fd = os.open(place, flags, 0o600)
        except FileExistsError:
            text = _read_secret(place)
        else:
            with os.fdopen(fd, "w", encoding="ascii") as fh:
                fh.write(made)
            return bytes.fromhex(made)
    value = (text or "").strip()
    if re.fullmatch(r"[0-9a-f]{64}", value):
        return bytes.fromhex(value)
    made = secrets.token_hex(32)
    fd, staged = tempfile.mkstemp(prefix=place.name + ".", dir=real.parent)
    try:
        with os.fdopen(fd, "w", encoding="ascii") as fh:
            fh.write(made)
        os.chmod(staged, 0o600)
        os.replace(staged, place)
    except BaseException:
        Path(staged).unlink(missing_ok=True)
        raise
    return bytes.fromhex(made)


def _code(secret: bytes, domain: bytes, message: bytes, prefix: str) -> str:
    """``prefix`` and 16 base32 characters (lower case) of a keyed digest."""
    digest = hmac.new(secret, domain + message, "sha256").digest()[:10]
    return prefix + base64.b32encode(digest).decode("ascii").lower()


def _draw(real: Path, indices: list[int]) -> bytes:
    """The draw: a digest of the sheet's name and its windows, sorted.

    A plan drawn again with the same windows keeps its handles; other
    windows give other handles and another sheet id.
    """
    text = real.name + "\0" + ",".join(str(i) for i in sorted(indices))
    return hashlib.sha256(text.encode("utf-8")).digest()


def _handle(secret: bytes, draw: bytes, index: int) -> str:
    """A window's handle: ``w`` and 16 base32 characters, 17 in all."""
    return _code(secret, b"olmoearth-window\0", draw + b"\0" + str(index).encode(), "w")


def _sheet_id(secret: bytes, draw: bytes) -> str:
    """The sheet's id for its draw: ``s`` and 16 base32 characters."""
    return _code(secret, b"olmoearth-sheet\0", draw, "s")


def _version(secret: bytes, raw: bytes) -> str:
    """The sheet's version: ``v`` and 16 base32 characters of a keyed digest of
    its bytes, under the sheet's own secret (a domain of its own).

    Keyed, not a plain hash of the file: a page holding a seen sheet's copy
    could otherwise test guesses of the map's classes against it. The secret
    is the sheet's, kept beside it, so a server restart keeps every version
    and an open panel's first save after it goes through.
    """
    return _code(secret, b"olmoearth-version\0", raw, "v")


def _windows(
    real: Path, header: list[str], body: list[list[str]], secret: bytes
) -> tuple[str, dict[str, int]]:
    """The sheet's id and its windows by handle, from the file and the secret.

    Worked out at every call (one keyed digest per window), never cached.
    ``ValueError`` for a window index that is not a whole number, one listed
    twice, or two handles alike.
    """
    at = header.index("window_index")
    indices: list[int] = []
    for row in body:
        try:
            indices.append(int(row[at]))
        except ValueError as exc:
            raise ValueError(
                f"{real.name} has a window_index that is not a number"
            ) from exc
    if len(set(indices)) != len(indices):
        raise ValueError(f"{real.name} lists a window twice")
    draw = _draw(real, indices)
    handles = {_handle(secret, draw, i): i for i in indices}
    if len(handles) != len(indices):  # pragma: no cover - 80 bits apart
        raise ValueError(f"{real.name} gives two windows one handle")
    return _sheet_id(secret, draw), handles


def _checked(
    real: Path,
    header: list[str],
    body: list[list[str]],
    sheet_id: Any,
    secret: bytes | None = None,
) -> dict[str, int]:
    """The windows by handle, once ``sheet_id`` is the sheet's; called under
    :func:`_sheet_lock`. ``ValueError`` (:data:`CHANGED_UNDER_PAGE`) when the
    sheet was drawn again, or its secret made anew, since the page read it.
    ``secret``: the sheet's, when the caller read it already."""
    if secret is None:
        secret = _sheet_secret(real)
    own, handles = _windows(real, header, body, secret)
    given = sheet_id if isinstance(sheet_id, str) else ""
    if not hmac.compare_digest(own.encode(), given.encode()):
        raise ValueError(CHANGED_UNDER_PAGE)
    return handles


def window_handles(path: str) -> tuple[str, dict[str, int]]:
    """The sheet's id and its windows by handle, for the server and tests.

    Never sent to a page: the page has the handles alone (:func:`read_sheet`).
    """
    real = resolve_sheet(path)
    with _sheet_lock(real):
        header, body = _read(real)
        return _windows(real, header, body, _sheet_secret(real))


def _seen_answer_values(design: dict[str, Any] | None) -> frozenset[str]:
    """The answers a seen sheet's ``reference_class`` may hold: the codes
    (:func:`_seen_codes`), and Cannot tell (``?``) where the sheet takes it."""
    values = set(_seen_codes(design) or [])
    if takes_unjudged(design):
        values.add(UNJUDGED)
    return frozenset(values)


def _seen_labelled(ref: str, wrong: str, answers: frozenset[str]) -> bool:
    """Whether a seen sheet's row is labelled: an answer in ``reference_class``
    and ``wrong`` worked out from it. A code typed into ``reference_class``
    with ``wrong`` blank (a spreadsheet) is not yet, until a save works
    ``wrong`` out (:func:`_rederive_seen`)."""
    return bool(wrong.strip()) and ref.strip() in answers


def _face_choices(rule: str | None, kind: str, header: list[str]) -> list[str]:
    """The faces a sheet's rule can be labelled on, whether or not offered now.

    A class map's sheet with a ``wrong`` column: the class seen or right or
    wrong; any other sheet one face (:func:`_faces`).
    """
    if kind == "two_map":
        return ["pair"]
    if rule == "seen":
        return ["seen"]
    if rule == "class":
        return ["class", "wrong"] if "wrong" in header else ["class"]
    return ["wrong"]


def _faces(
    rule: str | None, kind: str, header: list[str], body: list[list[str]]
) -> list[str]:
    """The faces offered now, the default first.

    A two-map sheet: ``pair``. A seen sheet: ``seen``. A sheet labelled
    right or wrong (a scores map's, or one without a design file):
    ``wrong``. A class map's: ``class`` while no row holds ``wrong`` alone,
    ``wrong`` while no ``reference_class`` is filled, so right or wrong,
    which shows the map's class, is never offered once a class seen is
    saved; with both (a sheet edited by hand) ``class``, and its mix is said
    (:func:`_mix_problem`).
    """
    choices = _face_choices(rule, kind, header)
    if choices != ["class", "wrong"]:
        return choices
    at_ref = header.index("reference_class")
    offered: list[str] = []
    if not _started_right_or_wrong(header, body):
        offered.append("class")
    if not any(row[at_ref].strip() for row in body):
        offered.append("wrong")
    return offered or ["class"]


def _counted(value: str, field: str, takes: bool) -> bool:
    """Whether a label cell holds a label the panel counts: anything but
    blank, and Cannot tell (``?``) in ``wrong`` only where the sheet takes it."""
    value = value.strip()
    return bool(value) and not (field == "wrong" and value == UNJUDGED and not takes)


def _mix_problem(
    face: str,
    rule: str | None,
    design: dict[str, Any] | None,
    header: list[str],
    body: list[list[str]],
) -> tuple[str, str] | None:
    """What the estimate would refuse in a sheet's two label columns, and the
    column to clear, or ``None``; the page's own check, from the files.

    For a one-map sheet labelled on ``class`` or ``wrong`` whose rule is the
    class seen or unknown (no design file). Both columns full: ``wrong``
    must agree with the class seen on every window, against the sheet's
    ``map_class`` (trimmed, any case); Cannot tell (``?``) agrees only with
    Cannot tell in the other column, as the estimate takes it on the same
    windows in both (the last round's S4). Else any label in the column not
    being labelled is a mix.
    """
    if face not in ("class", "wrong") or rule not in ("class", None):
        return None
    if "wrong" not in header:
        return None
    takes = takes_unjudged(design)
    at_ref, at_wrong = header.index("reference_class"), header.index("wrong")
    at_map = header.index("map_class") if "map_class" in header else None
    refs = sum(_counted(row[at_ref], "reference_class", takes) for row in body)
    wrongs = sum(_counted(row[at_wrong], "wrong", takes) for row in body)
    other = "wrong" if face == "class" else "reference_class"
    if body and refs == len(body) and wrongs == len(body):

        def agrees(row: list[str]) -> bool:
            """Whether ``wrong`` says what the class seen says of the map."""
            said, seen_class = row[at_wrong].strip(), row[at_ref].strip()
            if UNJUDGED in (said, seen_class):
                return said == seen_class
            on_map = row[at_map] if at_map is not None else ""
            same = seen_class.lower() == on_map.strip().lower()
            return (said == "1") == (not same)

        if all(agrees(row) for row in body):
            return None
        return "its wrong column disagrees with the classes seen", other
    count = refs if face == "wrong" else wrongs
    if not count:
        return None
    plural = "" if count == 1 else "s"
    return f"it also has {count} label{plural} in its {other} column", other


def _keys_for(
    values: list[tuple[str, str]], *, letters: bool, digits: bool
) -> list[list[str]]:
    """Each answer's keys, for ``(value, label)`` pairs in order.

    ``?`` for Cannot tell; with ``letters``, an answer's first letter (upper
    case) where no other answer but Cannot tell starts with it; with
    ``digits``, its place, 1 to 9 (Cannot tell takes one only beside
    letters, as on a seen sheet).
    """
    firsts = [label.strip()[:1].upper() for _, label in values]
    named = [f for f, (v, _) in zip(firsts, values, strict=True) if v != UNJUDGED]
    keys: list[list[str]] = []
    for i, ((value, _), first) in enumerate(zip(values, firsts, strict=True)):
        own: list[str] = []
        if value == UNJUDGED:
            own.append(UNJUDGED)
        elif letters and first.isalpha() and named.count(first) == 1:
            own.append(first)
        if digits and i < 9 and (value != UNJUDGED or letters):
            own.append(str(i + 1))
        keys.append(own)
    return keys


def _answers(
    face: str, design: dict[str, Any] | None, kind: str, rows: list[dict[str, str]]
) -> tuple[list[dict[str, Any]], bool]:
    """The answers a face takes, each ``{"value", "label", "keys"}``, and
    whether a class not listed may be typed (``open``).

    ``seen``: the design's codes with their names, then Cannot tell (``?``)
    where the sheet takes it; keys: a first letter no other answer shares,
    and 1 to 9. ``wrong``: Right (0, R), Wrong (1, W), and Cannot tell (?)
    where the sheet takes it; no digits (0 and 1 are the cells' own
    values). ``class``: the design's classes, then those the sample shows,
    sorted by name, then Cannot tell where the sheet takes it (the last
    round's S4: the estimate bounds it both ways, as on a seen sheet); digits;
    open. ``pair``: the design's classes (those the two maps show, sorted,
    besides), then Cannot tell; open. Cannot tell's key is ``?`` alone beside
    digits. A certificate plan's sheet never takes ``?``.
    """
    takes = takes_unjudged(design)
    if face == "seen":
        names = _class_names(design)
        values = [
            (code, names[code][:1].upper() + names[code][1:])
            for code in _seen_codes(design) or []
        ]
        letters, digits, open_ = True, True, False
    elif face == "wrong":
        values = [("0", "Right"), ("1", "Wrong")]
        letters, digits, open_ = True, False, False
    else:
        values = [(c, c) for c in _classes(design, rows, kind) if c != UNJUDGED]
        letters, digits, open_ = False, True, True
    if takes and face in ("seen", "wrong", "class", "pair"):
        values.append((UNJUDGED, CANNOT_TELL))
    keys = _keys_for(values, letters=letters, digits=digits)
    answers = [
        {"value": value, "label": label, "keys": own}
        for (value, label), own in zip(values, keys, strict=True)
    ]
    return answers, open_


def _face_refused(face: str, choices: list[str]) -> str:
    """Why a face the sheet does not offer now is refused: a class map's
    other face by why it waits; any face its rule cannot have, as a page
    drawn for another sheet (:data:`CHANGED_UNDER_PAGE`)."""
    if face not in choices:
        return CHANGED_UNDER_PAGE
    if face == "wrong":
        return FACE_WRONG_REFUSED
    if face == "class":
        return FACE_CLASS_REFUSED
    return CHANGED_UNDER_PAGE


def _window_answer(face: str, row: dict[str, str], seen_values: frozenset[str]) -> str:
    """A window's answer on a face: its column's cell; on a seen sheet only
    where the row is labelled (:func:`_seen_labelled`), so the panel never
    shows a sheet as done that the estimate would refuse."""
    ref = (row.get("reference_class") or "").strip()
    if face == "seen":
        return ref if _seen_labelled(ref, row.get("wrong") or "", seen_values) else ""
    if face == "wrong":
        return (row.get("wrong") or "").strip()
    return ref


def _page_windows(
    face: str,
    rows: list[dict[str, str]],
    handles: dict[str, int],
    *,
    design: dict[str, Any] | None,
    unblind: bool,
) -> list[dict[str, Any]]:
    """The windows as the page gets them, in the order of their handles.

    Each ``{"handle", "answer"}``; on the ``wrong`` face also ``map_class``
    (its name where the design names it), and on no other face. ``unblind``
    (the reviewer's press on a two-map sheet's blind switch, the ``pair`` face
    only): the two maps' classes (``pair``). Never the ``stratum``: the owner's
    O1 of 14 Nov 2026 took the class sheet's switch away, and no face shows a
    stratum (the last round's F11).
    """
    by_index = {int(row["window_index"]): row for row in rows}
    names = _class_names(design)
    seen_values = _seen_answer_values(design)
    out: list[dict[str, Any]] = []
    for handle in sorted(handles):
        row = by_index[handles[handle]]
        shown: dict[str, Any] = {
            "handle": handle,
            "answer": _window_answer(face, row, seen_values),
        }
        if face == "wrong":
            on_map = (row.get("map_class") or "").strip()
            shown["map_class"] = names.get(on_map, on_map)
        if unblind and face == "pair":
            shown["pair"] = [
                (row.get("class_a") or "").strip(),
                (row.get("class_b") or "").strip(),
            ]
        out.append(shown)
    return out


#: The two-map design's name (``tools.estimation.TWO_MAP_DESIGN``, which a
#: test holds this to).
TWO_MAP_DESIGN = "disagreement"

#: The finish button's words for each job a sheet's labels serve
#: (:func:`sheet_job`).
JOB_LABELS = {
    "estimate": "Estimate the error rate",
    "certify": "Certify the zone",
    "which_map": "Say which map is more accurate",
}

#: The finish button's words when no design file beside the sheet tells its job.
NO_JOB_LABEL = "Ask Claude to grade it"

#: What the finish message asks Claude for, by job.
_JOB_ASKS = {
    "estimate": "Estimate the map's error rate from it",
    "certify": "Certify the zone from it",
    "which_map": "Say which map is more accurate from it",
}


def sheet_job(design: dict[str, Any] | None, kind: str) -> str | None:
    """What the sheet's labels are for: the job its finish button names.

    ``"which_map"`` for a two-map sheet (``design='disagreement'``);
    ``None`` when no design file beside a one-map sheet tells;
    ``"certify"`` for a certificate plan (``certificate.seed_rule ==
    "from_map"``), whose labels go to its certified zone and certificate;
    ``"estimate"`` for every other plan.
    """
    if kind == "two_map" or (design or {}).get("design") == TWO_MAP_DESIGN:
        return "which_map"
    if design is None:
        return None
    return "estimate" if takes_unjudged(design) else "certify"


def finish_call(
    job: str | None,
    *,
    labels_path: str,
    design_path: str | None,
    design: dict[str, Any] | None,
) -> str:
    """The tool call that grades the labelled sheet, in words for Claude.

    A certificate plan's is ``olmoearth_certify_zone`` at the alpha the plan
    fixed (``certificate.alpha``), the only alpha certify_zone takes on its
    labels; every other sheet's is ``olmoearth_estimate_map_error``.
    """
    paths = [f"design_path {design_path}"] if design_path else []
    paths.append(f"labels_path {labels_path}")
    if job != "certify":
        return "olmoearth_estimate_map_error with " + " and ".join(paths)
    block = (design or {}).get("certificate")
    alpha = block.get("alpha") if isinstance(block, dict) else None
    if isinstance(alpha, (int, float)) and not isinstance(alpha, bool):
        at = f"alpha {alpha:g} (fixed by the plan)"
    else:
        at = "the alpha the plan fixed (certificate.alpha in its design file)"
    return "olmoearth_certify_zone with " + ", ".join(paths) + " and " + at


def finish_message(job: str | None, *, name: str, call: str, labels_path: str) -> str:
    """The message the finish button sends to the chat, naming ``call``.

    A certificate plan's also asks Claude to offer the certificate once a zone
    is certified. Without a design file (``job`` None) it is the panel's
    older message, which names the sheet only. It names no unit, so it is
    true of a sheet of squares and of one of points, and no "window" (the
    owner's ruling 6 of 13 Nov 2026).
    """
    # WORDING: squares
    start = f"I have finished labelling the sheet {name}. "
    ask = _JOB_ASKS.get(job) if job else None
    if ask is None:
        return start + f"Estimate the map error from it: labels_path {labels_path}."
    text = start + f"{ask}: {call}."
    if job == "certify":
        text += (
            " If a zone is certified, offer to write its certificate "
            "(olmoearth_write_certificate with the same design_path and labels_path)."
        )
    return text


def _finish(
    real: Path, kind: str, design_path: Path | None, design: dict[str, Any] | None
) -> dict[str, Any]:
    """The sheet's job and its finish button, for the view and for Claude:
    ``job``, ``finish_label``, ``finish_call`` and ``finish_message``."""
    job = sheet_job(design, kind)
    call = finish_call(
        job,
        labels_path=str(real),
        design_path=str(design_path) if design_path else None,
        design=design,
    )
    return {
        "job": job,
        "finish_label": JOB_LABELS[job] if job else NO_JOB_LABEL,
        "finish_call": call,
        "finish_message": finish_message(
            job, name=real.name, call=call, labels_path=str(real)
        ),
    }


#: The plan card's button, by how far its sheet is labelled (:func:`sheet_progress`).
#: Before any label it names no unit (a sheet's windows are squares or
#: points, :func:`window_unit`): the lead's ruling of 13 Nov 2026 (round 4,
#: pass 2, finding 13; contract C2), and the Check card, the plan card's lead
#: and the model's notes say the same words.
# WORDING: r4p2
OPEN_FIRST = "Open the sheet here"
OPEN_AGAIN = "Continue labelling"
OPEN_DONE = "Open the sheet"

#: The suffix of a plan's figure, written beside its sheet under the same stem.
FIGURE_SUFFIX = "_figure.svg"

#: Said in place of a plan's figure while it is held (:func:`figure_shown`),
#: and the refusal of :func:`plan_figure` then; the plan card
#: (``views/result.html``'s ``FIGURE_LATER``) says the same words. No unit and
#: no "window" (the owner's ruling 6 of 13 Nov 2026): a plan's sheet is of
#: squares or of points.
# WORDING: squares
FIGURE_LATER = (
    "The plan's figure shades the map by its confidence, so it shows here once "
    "the whole sheet is labelled."
)

#: :func:`plan_figure`'s refusal when the counts allow the figure but no
#: figure a tool wrote is beside the sheet.
FIGURE_GONE = "The plan's figure file is missing or cannot be read."


def _counts(
    header: list[str],
    body: list[list[str]],
    answers: frozenset[str] | None = None,
) -> dict[str, int]:
    """How far the sheet is labelled: ``n_rows``, ``n_labelled`` (a row with
    ``reference_class`` or ``wrong`` filled) and ``n_unjudged`` (a labelled
    row that says Cannot tell, ``?``, in either).

    ``answers``: a seen sheet's (:func:`_seen_answer_values`), whose row is
    labelled only once its ``wrong`` is worked out from an answer
    (:func:`_seen_labelled`), as the estimate reads it.
    """
    at_ref = header.index("reference_class")
    at_wrong = header.index("wrong") if "wrong" in header else None
    cells = [
        (row[at_ref].strip(), row[at_wrong].strip() if at_wrong is not None else "")
        for row in body
    ]
    if answers is None:
        labelled = [(ref, wrong) for ref, wrong in cells if ref or wrong]
    else:
        labelled = [
            (ref, wrong) for ref, wrong in cells if _seen_labelled(ref, wrong, answers)
        ]
    return {
        "n_rows": len(body),
        "n_labelled": len(labelled),
        "n_unjudged": sum(1 for pair in labelled if UNJUDGED in pair),
    }


def _counted_as_shown(
    header: list[str], body: list[list[str]], design: dict[str, Any] | None
) -> tuple[str, str | None, dict[str, int]]:
    """The sheet's kind, its label rule now (:func:`_rule`) and its counts
    (:func:`_counts`), as the panels show them.

    A seen sheet's row counts once its ``wrong`` is worked out from an
    answer the design names (:func:`_seen_answer_values`), as the estimate
    reads it; any other sheet's once a label cell is filled.
    :func:`sheet_progress` and :func:`sheets_in_progress` count so.
    """
    kind = _kind(header)
    rule = _rule(design, kind, header, body)
    answers = _seen_answer_values(design) if rule == "seen" else None
    return kind, rule, _counts(header, body, answers)


def _opens_blind(rule: str | None, header: list[str], body: list[list[str]]) -> bool:
    """Whether the panel opens the sheet blind, as ``views/sheet.js`` draws it.

    A seen sheet, and a sheet labelled with the class seen (a class map's, a
    two-map sheet): the map's class is hidden, or never sent. Not a sheet
    labelled right or wrong (a scores map's, a class map's holding ``wrong``
    alone, one without a design file), which shows the map's class. The
    server sends the plan card no figure of the map's confidence beside a
    blind sheet until every window is labelled (:func:`figure_shown`).
    """
    if rule == "seen":
        return True
    if rule != "class":
        return False
    at_ref = header.index("reference_class")
    at_wrong = header.index("wrong") if "wrong" in header else None
    wrongs = at_wrong is not None and any(row[at_wrong].strip() for row in body)
    refs = any(row[at_ref].strip() for row in body)
    return not (wrongs and not refs)


def progress_words(n_labelled: int, n_rows: int, n_unjudged: int) -> tuple[str, str]:
    """The plan card's line and button words for a sheet labelled this far.

    The line is ``"k of n labelled"``, with ``", u cannot tell"`` when any
    window says Cannot tell; the button is :data:`OPEN_FIRST` before any
    label, :data:`OPEN_AGAIN` part way, :data:`OPEN_DONE` once every window
    is labelled.
    """
    line = f"{n_labelled} of {n_rows} labelled"
    if n_unjudged:
        line += f", {n_unjudged} cannot tell"
    if n_labelled <= 0:
        return line, OPEN_FIRST
    return line, OPEN_AGAIN if n_labelled < n_rows else OPEN_DONE


def sheet_progress(path: str) -> dict[str, Any]:
    """How far a plan's sheet is labelled: counts only, for the plan card and
    the sheet tool's result.

    ``path``, ``name``, ``kind``, ``design_path``, ``label_rule``
    (:func:`_rule`), ``blind`` (:func:`_opens_blind`: the server then holds
    the plan's figure from its card until every window is labelled,
    :func:`figure_shown`),
    ``takes_unjudged``, the counts (:func:`_counts`), ``progress`` and
    ``open_label`` (:func:`progress_words`), and the job and its finish
    button (``job``, ``finish_label``, ``finish_call``, ``finish_message``);
    ``version`` (:func:`_version`, read with the rows), which a panel
    compares with the version it holds to find a sheet saved elsewhere;
    ``reading_rule``, ``rule_scope`` and ``rule_source``, the labelling rule
    (:func:`labelling_rule`; ``None`` each without one), and ``rule_limit``
    (``imagery.RULE_KELP_ONLY`` for the owner's rule, else ``None``:
    :func:`_rule_fields`), and ``graded``
    (``{first_at, changed_since_first}`` once the sheet was graded, else
    ``None``: :func:`record_grading`).
    Never a row, a place, a class or a label: the view reads the rows from
    the file when it opens (:func:`read_sheet`).
    """
    real = resolve_sheet(path)
    with _sheet_lock(real):
        header, body, version = _read_stamped(real, _sheet_secret(real))
    design_path, design = _design_of(real)
    kind, rule, counts = _counted_as_shown(header, body, design)
    line, label = progress_words(
        counts["n_labelled"], counts["n_rows"], counts["n_unjudged"]
    )
    rows = [dict(zip(header, row)) for row in body]
    return {
        "path": str(real),
        "name": real.name,
        "kind": kind,
        "design_path": str(design_path) if design_path else None,
        "label_rule": rule,
        "blind": _opens_blind(rule, header, body),
        "takes_unjudged": takes_unjudged(design),
        **counts,
        "progress": line,
        "open_label": label,
        **_finish(real, kind, design_path, design),
        "version": version,
        **_rule_fields(real, design, rows, kind),
        "graded": _graded_field(real, version, header=header, body=body, design=design),
    }


def figure_shown(progress: dict[str, Any] | None) -> bool:
    """Whether a plan's figure may reach its card, from its sheet's counts.

    ``progress`` is :func:`sheet_progress`'s answer, or ``None`` when the
    counts cannot be read. The figure shades each window by the map's
    confidence, which on a confidence design is its stratum, so it is held
    while the sheet opens blind (``blind``, :func:`_opens_blind`) and is not
    complete, and while the counts cannot be read. A sheet labelled right or
    wrong shows the map's class anyway: its figure shows at once. A two-map
    sheet is blind too, so its figure (where the maps differ, no confidence)
    waits as well.
    """
    if not isinstance(progress, dict):
        return False
    if progress.get("blind") is False:
        return True
    rows, labelled = progress.get("n_rows"), progress.get("n_labelled")
    if not isinstance(rows, int) or not isinstance(labelled, int):
        return False
    return 0 < rows <= labelled


def plan_figure(path: str) -> dict[str, Any]:
    """A plan's figure for its card, once its sheet's counts allow it.

    ``path`` is the plan's sheet (:func:`resolve_sheet`'s rules; its refusal
    is raised as it is). The figure is ``<stem>_figure.svg`` beside it, which
    the plan wrote under the sheet's stem, read through
    :func:`~olmoearth_agent.reporting.figure_file.read_figure_svg`. Returns
    ``{"svg": ...}``. ``ValueError`` with :data:`FIGURE_LATER` while
    :func:`figure_shown` holds the figure (counts that cannot be read
    included), and with :data:`FIGURE_GONE` when no figure a tool wrote is
    there.
    """
    real = resolve_sheet(path)
    try:
        progress: dict[str, Any] | None = sheet_progress(str(real))
    except Exception:  # noqa: BLE001 - counts not read: the figure is held
        progress = None
    if not figure_shown(progress):
        raise ValueError(FIGURE_LATER)
    stem = real.name[: -len(SHEET_SUFFIX)]
    svg = read_figure_svg(str(real.with_name(stem + FIGURE_SUFFIX)))
    if svg is None:
        raise ValueError(FIGURE_GONE)
    return {"svg": svg}


#: The most rows the projects card lists under "In progress"
#: (:func:`sheets_in_progress`), and the most sheets, newest first, it reads
#: to find them.
IN_PROGRESS_MAX = 5
IN_PROGRESS_READ = 25

#: What a listed sheet's labels are for, by its job (:func:`sheet_job`); a
#: certificate plan's in the reader test's words (A/projects-7, 13 Nov 2026).
IN_PROGRESS_JOBS = {
    "estimate": "Checking how wrong a map is",
    # WORDING: r4p2
    "certify": "Labelling for a zone test (it may certify nothing)",
    "which_map": "Saying which map is more accurate",
}


def resume_message(name: str, path: str) -> str:
    """What a listed sheet's Resume sends Claude as the user's words: open
    the sheet ``name`` at ``path`` again with the sheet tool."""
    return (
        f"Continue labelling my label sheet {name}: open it with "
        f"olmoearth_label_sheet, labels_csv_path {path}."
    )


def _sheet_files() -> list[tuple[str, float]]:
    """Each plan's sheet at the top of the write root and of the workspace
    root, with when it was last written, newest first.

    One :func:`os.scandir` of each root's real path, a root taken once when
    the two are one folder; no subfolder is entered (the planner writes its
    sheets at the top of the write root). A sheet is a file named
    ``*_to_label.csv``, not hidden, and a regular file itself: a link is left
    out. A root that cannot be read gives none.
    """
    roots: list[str] = []
    for root in (write_root, workspace_root):
        try:
            real = os.path.realpath(root())
        except (OSError, RuntimeError, ValueError):
            continue
        if real not in roots:
            roots.append(real)
    found: dict[str, float] = {}
    for root_path in roots:
        try:
            with os.scandir(root_path) as entries:
                for entry in entries:
                    name = entry.name
                    if name.startswith(".") or not name.endswith(SHEET_SUFFIX):
                        continue
                    try:
                        if entry.is_file(follow_symlinks=False):
                            mtime = entry.stat(follow_symlinks=False).st_mtime
                            found.setdefault(entry.path, mtime)
                    except OSError:
                        continue
        except OSError:
            continue
    return sorted(found.items(), key=lambda item: (-item[1], item[0]))


def sheets_in_progress() -> dict[str, Any]:
    """The plans' sheets on this computer that are part labelled, for the
    projects card's "In progress": counts only.

    The sheets are :func:`_sheet_files`', newest first. The newest
    :data:`IN_PROGRESS_READ` are read, each through :func:`resolve_sheet`
    and without the lock: a save replaces the file whole, so a read sees the
    file before it or after it, and a sheet only listed gets no lock file
    or secret beside it. A sheet counts when some window and not every
    window has a label cell filled (0 < k < n) and a design file is beside
    it. The design is read only for the rows listed, at most
    :data:`IN_PROGRESS_MAX`; each row's job (:func:`sheet_job`) and counts
    come from it, counted as :func:`sheet_progress` counts them
    (:func:`_counted_as_shown`).

    The filter counts without the design, so on a seen sheet it can differ
    from the counts shown, rarely: a class code typed in a spreadsheet,
    with ``wrong`` not yet worked out, passes the filter but is not
    labelled. A row whose counts then say no window labelled is left out
    (and not replaced, so no more than :data:`IN_PROGRESS_MAX` designs are
    read); a seen sheet whose every label cell is filled, some not yet
    worked out, is not listed, though its plan card says it is part way.

    Returns ``{"sheets": [...], "more": bool}``. Each row: ``path`` and the
    same path as ``labels_csv_path`` (the name the projects panel tells
    Claude it by, with the sheet's whole file name), ``name``, ``job``,
    ``job_label`` (:data:`IN_PROGRESS_JOBS`),
    ``progress`` (:func:`progress_words`' line), ``saved`` (when the file
    was last written, UTC, ISO 8601) and ``resume_message``
    (:func:`resume_message`). ``more``: more such sheets were found than
    are listed, or sheets past the newest :data:`IN_PROGRESS_READ` were not
    read. A sheet that cannot be read is skipped. Never a row, a place, a
    class or a label. Sheets of any Studio account on this computer are
    listed: a sheet names no account.
    """
    files = _sheet_files()
    more = len(files) > IN_PROGRESS_READ
    picked: list[tuple[Path, float, list[str], list[list[str]]]] = []
    for path, mtime in files[:IN_PROGRESS_READ]:
        try:
            real = resolve_sheet(path)
            header, body = _read(real)
            raw = _counts(header, body)
            if not 0 < raw["n_labelled"] < raw["n_rows"]:
                continue
            if not _design_path(real).is_file():
                continue
        except (OSError, ValueError, csv.Error):
            continue
        if len(picked) == IN_PROGRESS_MAX:
            more = True
            break
        picked.append((real, mtime, header, body))
    rows: list[dict[str, Any]] = []
    for real, mtime, header, body in picked:
        design = _design_of(real)[1]
        if design is None:
            continue
        kind, _rule_now, counts = _counted_as_shown(header, body, design)
        job = sheet_job(design, kind)
        if job is None or not 0 < counts["n_labelled"] < counts["n_rows"]:
            continue
        line = progress_words(
            counts["n_labelled"], counts["n_rows"], counts["n_unjudged"]
        )[0]
        rows.append(
            {
                "path": str(real),
                "labels_csv_path": str(real),
                "name": real.name,
                "job": job,
                "job_label": IN_PROGRESS_JOBS[job],
                "progress": line,
                "saved": datetime.fromtimestamp(mtime, UTC).isoformat(
                    timespec="seconds"
                ),
                "resume_message": resume_message(real.name, str(real)),
            }
        )
    return {"sheets": rows, "more": more}


def read_sheet(
    path: str, face: str | None = None, *, unblind: bool = False
) -> dict[str, Any]:
    """The page copy of a sheet: its windows by handle, the answers, the counts.

    ``path``, ``name``, ``kind`` (``two_map`` for a sheet with ``class_a``
    and ``class_b``, else ``one_map``), ``label_rule`` (:func:`_rule`),
    ``blind`` (:func:`_opens_blind`), ``design_path`` (the design file
    beside the sheet, which the estimate takes with it),
    ``takes_unjudged``, the counts (:func:`_counts`) and the job and its
    finish button (:func:`_finish`) and ``version`` (:func:`_version`, read
    with the rows), as :func:`sheet_progress` gives them;
    ``sheet_id`` (:func:`_sheet_id`); ``face``, the column the page writes,
    and ``faces``, those offered now (:func:`_faces`); ``answers`` and
    ``open`` (:func:`_answers`); ``has_place``, whether the sheet has places
    (``lon`` and ``lat``); ``windows`` (:func:`_page_windows`), in the order
    of their handles. No ``face_choices`` and no ``has_stratum``: the panel
    has no face switch and shows no stratum (the owner's O1 and the last
    round's F11, 14 Nov 2026). ``reading_rule``, ``rule_scope``,
    ``rule_source``, ``rule_limit`` and ``graded`` as :func:`sheet_progress`
    gives them (the panel prints the rule under its instruction from the first
    open, and says a graded sheet's answers should not change). ``unit``, the
    word the face says for one window (:func:`window_unit`: ``"square"`` or
    ``"point"``),
    and ``pictures_possible``, whether its windows can have pictures at all
    (:func:`pictures_possible`, from the design's structure, and for a scores
    file that names a Studio result that small file's own top level, never a
    raster; the panel then offers no Pictures button and does not speak of
    pictures). ``period_text``, the period the map covers as the class face
    names it (:func:`_period_text`: ``"1 Apr 2024 to 30 Jun 2024"``, or
    ``None``). ``rule_note``
    for a named scores map's sheet labelled right or wrong first;
    ``mix_problem`` and ``clear_column`` when the two label columns would be
    refused (:func:`_mix_problem`).

    Never a window index, place, row, column, order, stratum or design.
    ``face`` (default: the first offered) not offered now is refused
    (:data:`FACE_WRONG_REFUSED`, :data:`FACE_CLASS_REFUSED`). ``unblind``
    (the reviewer's press on a two-map sheet's blind switch) adds the two
    maps' classes on ``pair`` (``unblinded``); on any other face it is
    ignored, so a class map's class seen is never shown beside the map's
    class (the owner's O1).
    """
    real = resolve_sheet(path)
    if face is not None and face not in FACES:
        raise ValueError("face is one of " + ", ".join(FACES))
    design_path, design = _design_of(real)
    with _sheet_lock(real):
        secret = _sheet_secret(real)
        header, body, version = _read_stamped(real, secret)
        sheet_id, handles = _windows(real, header, body, secret)
    rows = [dict(zip(header, row)) for row in body]
    kind = _kind(header)
    rule = _rule(design, kind, header, body)
    offered = _faces(rule, kind, header, body)
    face = face or offered[0]
    if face not in offered:
        raise ValueError(_face_refused(face, _face_choices(rule, kind, header)))
    seen_values = _seen_answer_values(design) if rule == "seen" else None
    answers, open_ = _answers(face, design, kind, rows)
    # The two-map sheet's switch alone (the owner's O1 of 14 Nov 2026).
    unblinded = bool(unblind) and face == "pair"
    data: dict[str, Any] = {
        "path": str(real),
        "name": real.name,
        "kind": kind,
        "label_rule": rule,
        "blind": _opens_blind(rule, header, body),
        "design_path": str(design_path) if design_path else None,
        "takes_unjudged": takes_unjudged(design),
        **_counts(header, body, seen_values),
        **_finish(real, kind, design_path, design),
        "version": version,
        "sheet_id": sheet_id,
        "face": face,
        "faces": offered,
        "answers": answers,
        "open": open_,
        "has_place": {"lon", "lat"} <= set(header),
        "windows": _page_windows(face, rows, handles, design=design, unblind=unblinded),
        "unit": window_unit(design),
        "pictures_possible": pictures_possible(design),
        "period_text": _period_text(design),
    }
    data.update(_rule_fields(real, design, rows, kind))
    data["graded"] = _graded_field(
        real, version, header=header, body=body, design=design
    )
    if unblinded:
        data["unblinded"] = True
    if rule == "wrong" and "wrong" in header and _seen_codes(design) is not None:
        # A named scores map's sheet would be seen, but rows were labelled
        # right or wrong first (:func:`_rule`).
        data["rule_note"] = RULE_NOTE
    mix = _mix_problem(face, rule, design, header, body)
    if mix is not None:
        data["mix_problem"], data["clear_column"] = mix
    return data


def window_row(
    path: str, sheet_id: Any, handle: Any
) -> tuple[Path, dict[str, str], list[str]]:
    """One window's row, its sheet and the sheet's class names, for the pictures.

    The row as written (window index, row, col, place): never returned to a
    page. ``sheet_id`` must be the sheet's as it is now
    (:data:`CHANGED_UNDER_PAGE`) and ``handle`` one of its windows
    (:data:`UNKNOWN_HANDLE`); both are checked before anything else is read.
    The class names are a scores map's names by code, else the classes the
    design and the rows show (:func:`_classes`).
    """
    real = resolve_sheet(path)
    with _sheet_lock(real):
        header, body = _read(real)
        handles = _checked(real, header, body, sheet_id)
    index = handles.get(handle) if isinstance(handle, str) else None
    if index is None:
        raise ValueError(UNKNOWN_HANDLE)
    rows = [dict(zip(header, row)) for row in body]
    row = next(r for r in rows if int(r["window_index"]) == index)
    design = _design_of(real)[1]
    return real, row, _rule_names(design, rows, _kind(header))


def _rule_names(
    design: dict[str, Any] | None, rows: list[dict[str, str]], kind: str
) -> list[str]:
    """The class names the labelling rule is chosen by: a scores map's names by
    code, else the classes the design and the rows show (:func:`_classes`)."""
    named = list(_class_names(design).values())
    return named or _classes(design, rows, kind)


def _coordinate(value: str | None, limit: float) -> float | None:
    """A longitude or latitude cell as a number within ``limit``, or ``None``."""
    try:
        number = float(str(value or "").strip())
    except ValueError:
        return None
    return number if math.isfinite(number) and abs(number) <= limit else None


def degrees(value: float) -> str:
    """A coordinate as the map link (and the look-first CSV, r4-k K8) writes
    it: at most 6 decimals."""
    return f"{value:.6f}".rstrip("0").rstrip(".")


def osm_url(lon: float, lat: float) -> str:
    """The OpenStreetMap link of one place, a marker at zoom 17: the sheet's
    Street map (:func:`window_link`) and a look-first square's or point's
    (``review.review_link``, r4-k K7). ``lon`` and ``lat`` are checked by
    the caller (finite, within 180 and 90 degrees); the link goes only to the
    user's browser, never into a page's text or the model's."""
    x, y = degrees(lon), degrees(lat)
    return f"https://www.openstreetmap.org/?mlat={y}&mlon={x}#map=17/{y}/{x}"


def window_link(path: str, sheet_id: Any, handle: Any) -> dict[str, str]:
    """One window's OpenStreetMap link, at the reviewer's press: ``{"url"}``.

    The page copy holds no place (:func:`read_sheet`); this gives one
    window's, checked as :func:`window_row` checks it, when the user presses
    Street map (:func:`osm_url`). :data:`NO_PLACE` when its row holds no
    place.
    """
    _, row, _ = window_row(path, sheet_id, handle)
    lon, lat = _coordinate(row.get("lon"), 180.0), _coordinate(row.get("lat"), 90.0)
    if lon is None or lat is None:
        raise ValueError(NO_PLACE)
    return {"url": osm_url(lon, lat)}


def _label_value(value: Any, *, kind: str, takes: bool) -> str:
    """A ``reference_class`` cell as written; ``ValueError`` for one that cannot be.

    Cannot tell (``?``) is taken on a two-map sheet, and on a one-map sheet
    where ``takes`` (its rule is the class seen, ``"class"``, and it takes
    Cannot tell, :func:`takes_unjudged`: the last round's S4, which the
    estimate bounds both ways); refused elsewhere (:data:`UNJUDGED_NOT_HERE`).
    """
    text = "" if value is None else str(value).strip()
    if len(text) > MAX_LABEL_CHARS or any(ch in text for ch in "\r\n\x00"):
        raise ValueError("a class label is one short line of text")
    if text == UNJUDGED and kind != "two_map" and not takes:
        raise ValueError(UNJUDGED_NOT_HERE)
    return text


def _wrong_value(value: Any) -> str:
    """A ``wrong`` cell: ``''``, ``'0'``, ``'1'`` or ``'?'`` (cannot tell)."""
    text = "" if value is None else str(value).strip()
    if text not in ("", "0", "1", UNJUDGED):
        raise ValueError(
            "wrong is 1 where the map is wrong, 0 where it is right, ? where you "
            "cannot tell, or blank"
        )
    return text


def _seen_answers_text(codes: list[str], takes: bool) -> str:
    """Why a ``reference_class`` is refused on a seen sheet: the codes it takes."""
    return (
        "reference_class here is one of the codes "
        + ", ".join(codes)
        + (", or ? (cannot tell)" if takes else "")
        + ", or blank."
    )


def _map_code(population: Any, index: int) -> int:
    """The map's class of a window, from the design (``population.map_class``).

    ``ValueError`` when the design's map gives the window none (out of its
    range, below 0, or not a whole number). The sheet's ``map_class`` cell
    is never read.
    """
    on_map = population.get("map_class") if isinstance(population, dict) else None
    code = (
        on_map[index] if isinstance(on_map, list) and 0 <= index < len(on_map) else None
    )
    if not isinstance(code, int) or isinstance(code, bool) or code < 0:
        raise ValueError(f"window {index} has no class on this design's map")
    return code


def _seen_cells(
    label: dict[str, Any],
    cells: list[str],
    index: int,
    *,
    at: tuple[int, int],
    design: dict[str, Any] | None,
    codes: list[str],
) -> tuple[int, bool]:
    """Write one label of a seen sheet: the code seen and ``wrong`` from it.

    ``at`` is the ``(reference_class, wrong)`` columns. ``wrong`` is ``''``
    for a blank answer, ``?`` for Cannot tell, else 1 where the code is not
    the design's map class there and 0 where it is (:func:`_map_code`).
    Returns how many ``reference_class`` cells changed (0 or 1) and whether
    any cell did.
    """
    if "wrong" in label:
        raise ValueError(SEEN_NO_WRONG)
    if "reference_class" not in label:
        return 0, False
    raw = label["reference_class"]
    value = "" if raw is None else str(raw).strip()
    takes = takes_unjudged(design)
    if value == UNJUDGED and not takes:
        raise ValueError(_NO_UNJUDGED)
    if value not in ("", UNJUDGED, *codes):
        raise ValueError(_seen_answers_text(codes, takes))
    wrong = value
    if value not in ("", UNJUDGED):
        on_map = _map_code((design or {}).get("population"), index)
        wrong = str(int(int(value) != on_map))
    at_ref, at_wrong = at
    changed = cells[at_ref] != value
    dirty = changed or cells[at_wrong] != wrong
    cells[at_ref], cells[at_wrong] = value, wrong
    return int(changed), dirty


def _rederive_seen(
    body: list[list[str]],
    *,
    at: tuple[int, int, int],
    design: dict[str, Any] | None,
    codes: list[str],
) -> bool:
    """Work ``wrong`` out again on every row of a seen sheet that holds an answer.

    ``at`` is the ``(window_index, reference_class, wrong)`` columns. A row
    whose ``reference_class`` is one of ``codes`` gets ``wrong`` from the
    design's map class there (:func:`_map_code`, never the sheet's cell),
    one that says Cannot tell (``?``, where the sheet takes it) gets ``?``;
    so a code typed in a spreadsheet, or a ``wrong`` edited by hand, is
    graded as the panel's own answer would be. Any other row, and a window
    the design's map gives no class, is left as it is. Returns whether any
    cell changed.
    """
    at_index, at_ref, at_wrong = at
    population = (design or {}).get("population")
    takes = takes_unjudged(design)
    changed = False
    for row in body:
        value = row[at_ref].strip()
        if value == UNJUDGED and takes:
            wrong = UNJUDGED
        elif value in codes:
            try:
                on_map = _map_code(population, int(row[at_index]))
            except ValueError:
                continue
            wrong = str(int(int(value) != on_map))
        else:
            continue
        if row[at_wrong] != wrong:
            row[at_wrong] = wrong
            changed = True
    return changed


def _write_sheet(real: Path, header: list[str], body: list[list[str]]) -> bytes:
    """Replace the sheet whole: a new file beside it, then moved over it.

    Returns the bytes written (CSV, ``\\r\\n`` line ends, UTF-8), from which
    the save's answer works out the sheet's new version (:func:`_version`).
    """
    text = io.StringIO(newline="")
    writer = csv.writer(text)
    writer.writerow(header)
    writer.writerows(body)
    raw = text.getvalue().encode("utf-8")
    fd, staged = tempfile.mkstemp(prefix="." + real.name + ".", dir=real.parent)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(raw)
        os.chmod(staged, real.stat().st_mode & 0o777)
        os.replace(staged, real)
    except BaseException:
        Path(staged).unlink(missing_ok=True)
        raise
    return raw


def save_labels(
    path: str,
    labels: list[dict[str, Any]],
    *,
    rule: str | None = None,
    check_rule: bool = False,
) -> dict[str, Any]:
    """Write ``labels`` into the sheet's label columns; return the sheet's progress.

    Each label is ``{"window_index": int, "reference_class"?: str,
    "wrong"?: "0"|"1"|"?"|""}`` ("?": cannot tell); a key left out leaves its
    cell as it is. A window
    the sheet does not list is refused, and so is a ``wrong`` on a sheet
    without that column. The sheet is replaced whole (a new file beside it,
    made with a name of its own, then moved over it), so a reader finds it as
    it was or as written; the read and the write are one step under
    :func:`_sheet_lock`.

    On a seen sheet (:func:`_rule`) a label is the code seen alone: a
    ``wrong`` is refused, and ``wrong`` is written from the code and the
    design's map class (:func:`_seen_cells`); ``cells_changed`` counts the
    ``reference_class`` cells. Every save of a seen sheet also works
    ``wrong`` out again on each row that holds an answer, not only on the
    rows sent (:func:`_rederive_seen`). ``n_labelled`` counts as
    :func:`_counts` does. With ``check_rule`` (a panel's save, ``rule`` the
    label rule it drew), a sheet whose rule is no longer ``rule`` is refused
    (:data:`CHANGED_UNDER_PAGE`) and left as it is. The graded mark beside the
    sheet is kept true to the write (:func:`_write_keeping_mark`): a save
    into a blank sheet first removes it (an earlier sheet's of this name),
    unless the app blanked this sheet itself; a save that empties a graded
    sheet keeps it, marked changed.
    """
    real = resolve_sheet(path)
    if not isinstance(labels, list):
        raise ValueError("labels must be a list")
    design = _design_of(real)[1]
    with _sheet_lock(real):
        secret = _sheet_secret(real)
        header, body, read_version = _read_stamped(real, secret)
        kind = _kind(header)
        now = _rule(design, kind, header, body)
        if check_rule and rule != now:
            raise ValueError(CHANGED_UNDER_PAGE)
        blank = not _holds_answers(header, body)
        held = None if blank else _held_mark(real, header, body, design, read_version)
        changed, dirty, _redone = _apply(
            real, header, body, labels, kind=kind, now=now, design=design
        )
        if changed or dirty:
            _write_keeping_mark(
                real,
                header,
                body,
                secret=secret,
                blank=blank,
                read_version=read_version,
                held=held,
            )
    return _saved(real, header, body, now, design, changed)


def _saved(
    real: Path,
    header: list[str],
    body: list[list[str]],
    now: str | None,
    design: dict[str, Any] | None,
    changed: int,
) -> dict[str, Any]:
    """A save's answer: ``path``, ``cells_changed``, ``n_rows`` and
    ``n_labelled`` (as :func:`_counts` counts)."""
    answers = _seen_answer_values(design) if now == "seen" else None
    return {
        "path": str(real),
        "cells_changed": changed,
        "n_rows": len(body),
        "n_labelled": _counts(header, body, answers)["n_labelled"],
    }


def _apply(
    real: Path,
    header: list[str],
    body: list[list[str]],
    labels: list[Any],
    *,
    kind: str,
    now: str | None,
    design: dict[str, Any] | None,
) -> tuple[int, bool, bool]:
    """Write ``labels`` into ``body`` in place, under :func:`_sheet_lock`; the
    body of :func:`save_labels`, which :func:`save_answers` shares.

    Each label is ``{"window_index", "reference_class"?, "wrong"?}``.
    Returns how many label cells changed (on a seen sheet, the
    ``reference_class`` cells), whether any cell did (a seen sheet's
    ``wrong`` worked out again included), and whether working ``wrong`` out
    again on a seen sheet changed a row (:func:`_rederive_seen`): the rows
    sent are already worked out by then, so only rows not sent count.
    ``ValueError`` leaves the file as it is: nothing is written until every
    label is taken.
    """
    takes = takes_unjudged(design)
    # A class map's sheet takes Cannot tell as its class seen (S4).
    ref_takes = now == "class" and takes
    codes = _seen_codes(design) or []
    at_index = header.index("window_index")
    at_ref = header.index("reference_class")
    at_wrong = header.index("wrong") if "wrong" in header else None
    row_of: dict[int, list[str]] = {}
    for row in body:
        try:
            row_of[int(row[at_index])] = row
        except ValueError as exc:
            raise ValueError(
                f"{real.name} has a window_index that is not a number"
            ) from exc
    changed = 0
    dirty = False
    for label in labels:
        if not isinstance(label, dict):
            raise ValueError("each label is an object with window_index")
        try:
            index = int(label["window_index"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("each label needs a window_index") from exc
        cells = row_of.get(index)
        if cells is None:
            raise ValueError(f"window {index} is not on {real.name}")
        if now == "seen" and at_wrong is not None:
            count, touched = _seen_cells(
                label,
                cells,
                index,
                at=(at_ref, at_wrong),
                design=design,
                codes=codes,
            )
            changed += count
            dirty = dirty or touched
            continue
        if "reference_class" in label:
            value = _label_value(label["reference_class"], kind=kind, takes=ref_takes)
            changed += cells[at_ref] != value
            cells[at_ref] = value
        if "wrong" in label:
            if at_wrong is None:
                raise ValueError(
                    f"{real.name} has no wrong column; label the class seen"
                )
            value = _wrong_value(label["wrong"])
            if value == UNJUDGED and not takes:
                raise ValueError(_NO_UNJUDGED)
            changed += cells[at_wrong] != value
            cells[at_wrong] = value
    redone = False
    if now == "seen" and at_wrong is not None:
        redone = _rederive_seen(
            body, at=(at_index, at_ref, at_wrong), design=design, codes=codes
        )
        dirty = dirty or redone
    return changed, dirty, redone


def _clearable(
    column: Any,
    *,
    face: str,
    rule: str | None,
    kind: str,
    design: dict[str, Any] | None,
    header: list[str],
    body: list[list[str]],
) -> str:
    """The column a panel may clear whole, or ``ValueError`` (:data:`NOT_CLEARED`).

    The column :func:`_mix_problem` names, or either label column of a class
    map's sheet (to move between its two faces).
    """
    mix = _mix_problem(face, rule, design, header, body)
    if mix is not None and column == mix[1]:
        return str(column)
    two = rule == "class" and kind == "one_map" and "wrong" in header
    if two and column in ("wrong", "reference_class"):
        return str(column)
    raise ValueError(NOT_CLEARED)


def _answer_labels(
    answers: Any,
    handles: dict[str, int],
    *,
    face: str,
    allowed: list[dict[str, Any]],
    open_: bool,
) -> list[dict[str, Any]]:
    """A panel's ``answers`` (``[{"handle", "answer"}]``) as labels by window.

    An unknown handle is refused (:data:`UNKNOWN_HANDLE`), a window index
    (:data:`OLDER_PANEL`), and on a face that takes listed answers alone a
    value not listed (:data:`NOT_AN_ANSWER`); blank clears.
    """
    if not isinstance(answers, list):
        raise ValueError("answers must be a list of {handle, answer}")
    values = {a["value"] for a in allowed}
    column = FACE_COLUMN[face]
    labels: list[dict[str, Any]] = []
    for item in answers:
        if isinstance(item, dict) and "window_index" in item:
            raise ValueError(OLDER_PANEL)
        if not isinstance(item, dict) or set(item) != {"handle", "answer"}:
            raise ValueError("each answer is {handle, answer}")
        handle, value = item["handle"], item["answer"]
        index = handles.get(handle) if isinstance(handle, str) else None
        if index is None:
            raise ValueError(UNKNOWN_HANDLE)
        if not isinstance(value, str):
            raise ValueError("an answer is text")
        value = value.strip()
        # A face that takes listed answers alone refuses any other; an open
        # face takes a class typed in, but Cannot tell only where it is listed.
        if value and value not in values and (not open_ or value == UNJUDGED):
            listed = ", ".join(f"{a['label']} ({a['value']})" for a in allowed)
            raise ValueError(
                NOT_AN_ANSWER.format(value=value[:MAX_LABEL_CHARS], labels=listed)
            )
        labels.append({"window_index": index, column: value})
    return labels


def save_answers(
    path: str,
    *,
    sheet_id: Any,
    face: Any,
    rule: Any,
    answers: Any = None,
    clear_column: Any = None,
    version: Any = None,
) -> dict[str, Any]:
    """A panel's save: answers by handle on the face it drew; the sheet's progress.

    Under one :func:`_sheet_lock`: ``sheet_id`` must be the sheet's as it is
    now and ``rule`` its label rule (both :data:`CHANGED_UNDER_PAGE`: the
    page must open the sheet again); then ``version``, when sent, the
    sheet's version now (:func:`_version`; :class:`StaleSheet` when the sheet
    was saved elsewhere since the panel read it: the panel reads it again and
    sends its changes anew, so the first save wins); ``face`` one it offers now
    (:data:`FACE_WRONG_REFUSED`, :data:`FACE_CLASS_REFUSED`). Each answer
    ``{"handle", "answer"}`` goes into the face's column
    (:data:`FACE_COLUMN`), checked as :func:`_answer_labels` checks it, and
    written as :func:`save_labels` writes (:func:`_apply`, with the graded
    mark kept true to the write, :func:`_write_keeping_mark`: a clear that
    empties a graded sheet keeps its mark, marked changed). ``clear_column``
    (``wrong`` or ``reference_class``, :func:`_clearable`) blanks that column
    on every window first. Any refusal leaves the file as it was. Returns
    :func:`save_labels`' answer, with ``mix_problem`` and ``clear_column``
    as :func:`read_sheet` gives them now (``None`` when none) and
    ``version``: the version of the bytes written, or of the file as it is
    when nothing changed; and ``rederived: true`` when the save also changed
    rows that were not sent (a seen sheet's ``wrong`` worked out again,
    :func:`_rederive_seen`), so the panel reads the sheet again to hold what
    that version names (the key is left out otherwise, and names no handle
    or row).
    """
    real = resolve_sheet(path)
    if version is not None and not isinstance(version, str):
        raise ValueError(BAD_VERSION)
    design = _design_of(real)[1]
    with _sheet_lock(real):
        secret = _sheet_secret(real)
        header, body, now_version = _read_stamped(real, secret)
        handles = _checked(real, header, body, sheet_id, secret)
        kind = _kind(header)
        now = _rule(design, kind, header, body)
        if rule != now:
            raise ValueError(CHANGED_UNDER_PAGE)
        if version is not None and not hmac.compare_digest(
            version.encode("utf-8"), now_version.encode("utf-8")
        ):
            raise StaleSheet(STALE_SHEET)
        if not isinstance(face, str) or face not in FACES:
            raise ValueError("face is one of " + ", ".join(FACES))
        if face not in _faces(now, kind, header, body):
            raise ValueError(_face_refused(face, _face_choices(now, kind, header)))
        rows = [dict(zip(header, row)) for row in body]
        allowed, open_ = _answers(face, design, kind, rows)
        labels = _answer_labels(
            [] if answers is None else answers,
            handles,
            face=face,
            allowed=allowed,
            open_=open_,
        )
        if clear_column is not None:
            column = _clearable(
                clear_column,
                face=face,
                rule=now,
                kind=kind,
                design=design,
                header=header,
                body=body,
            )
            cleared: list[dict[str, Any]] = [
                {"window_index": i, column: ""} for i in handles.values()
            ]
            labels = cleared + labels
        blank = not _holds_answers(header, body)
        held = None if blank else _held_mark(real, header, body, design, now_version)
        changed, dirty, redone = _apply(
            real, header, body, labels, kind=kind, now=now, design=design
        )
        if changed or dirty:
            now_version = _write_keeping_mark(
                real,
                header,
                body,
                secret=secret,
                blank=blank,
                read_version=now_version,
                held=held,
            )
        mix = _mix_problem(face, now, design, header, body)
    out = _saved(real, header, body, now, design, changed)
    out["mix_problem"], out["clear_column"] = mix if mix is not None else (None, None)
    out["version"] = now_version
    if redone:
        # Rows not sent changed (a seen sheet's wrong worked out on a code a
        # spreadsheet typed): the version names content the panel does not
        # hold, so it reads the sheet again. No handle or row is named.
        out["rederived"] = True
    return out


# --------------------------------------------------------------------------
# The rule typed before the draw, the graded mark, the squares graded
# (round 4, pass 2: the owner's three extras of 13 Nov 2026)


def _hidden_path(real: Path, suffix: str) -> Path:
    """A hidden file beside the sheet: ``.<sheet name><suffix>``. Neither
    ``.csv`` nor ``.json``, so no tool that reads a path the model chooses
    can take it, and never listed as a sheet (:func:`_sheet_files`)."""
    return real.with_name("." + real.name + suffix)


def _read_hidden(place: Path) -> str | None:
    """A hidden file's text, never through a link; ``None`` when it is missing,
    longer than :data:`HIDDEN_MAX_BYTES` or not UTF-8. ``ValueError`` when a
    link stands at its place."""
    try:
        fd = os.open(place, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except FileNotFoundError:
        return None
    except OSError as exc:  # ELOOP: a link at its place
        if place.is_symlink():
            raise ValueError(f"{place.name} is a link; it is not read") from exc
        raise
    with os.fdopen(fd, "rb") as fh:
        raw = fh.read(HIDDEN_MAX_BYTES + 1)
    if len(raw) > HIDDEN_MAX_BYTES:
        return None
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return None


def _write_hidden(real: Path, place: Path, text: str) -> None:
    """Write a hidden file beside the sheet whole, mode 0600: a new file made
    beside it, then moved over its place, so a reader finds it as it was or
    as written. A link at its place is refused (``ValueError``), never
    followed."""
    if place.is_symlink():
        raise ValueError(f"{place.name} is a link; it is not written")
    fd, staged = tempfile.mkstemp(prefix=place.name + ".", dir=real.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.chmod(staged, 0o600)
        os.replace(staged, place)
    except BaseException:
        Path(staged).unlink(missing_ok=True)
        raise


def _now_text() -> str:
    """Now, UTC, as ISO 8601 to the second."""
    return datetime.now(UTC).isoformat(timespec="seconds")


def clean_rule(text: Any) -> str:
    """A labelling rule the user typed, as kept: its outer spaces trimmed, one
    printable line of :data:`RULE_MIN_CHARS` to :data:`RULE_MAX_CHARS`
    characters (no line break, tab or control character). ``ValueError``
    (:data:`RULE_TEXT`) otherwise.

    Before the checks, as text pasted from a document holds them: every
    Unicode space separator (category Zs: a no-break space, U+202F, an
    ideographic space) becomes a plain space, and every format character
    (category Cf: a zero-width space, a soft hyphen, a byte-order mark) is
    dropped. A line or paragraph separator, a line break and a tab are not
    spaces of that kind, so they are still refused. Lengths are counted in
    code points, after that.
    """
    if not isinstance(text, str):
        raise ValueError(RULE_TEXT)
    spaced = "".join(
        " " if unicodedata.category(ch) == "Zs" else ch
        for ch in text
        if unicodedata.category(ch) != "Cf"
    )
    cleaned = spaced.strip()
    if not RULE_MIN_CHARS <= len(cleaned) <= RULE_MAX_CHARS:
        raise ValueError(RULE_TEXT)
    if not cleaned.isprintable():
        raise ValueError(RULE_TEXT)
    return cleaned


def _clean_moment(value: Any) -> str:
    """When a rule was typed, as given: printable text of at most 40
    characters (the server's ISO 8601). ``ValueError`` (:data:`BAD_MOMENT`)."""
    if not isinstance(value, str):
        raise ValueError(BAD_MOMENT)
    text = value.strip()
    if not 0 < len(text) <= 40 or not text.isprintable():
        raise ValueError(BAD_MOMENT)
    return text


def typed_rule_scope(typed_at: str) -> str:
    """What is said with a typed rule (:data:`TYPED_RULE_SCOPE`): the day it was
    typed ("13 Nov 2026") when ``typed_at`` reads as ISO 8601, else as given."""
    try:
        when = datetime.fromisoformat(str(typed_at).strip().replace("Z", "+00:00"))
    except ValueError:
        day = str(typed_at)
    else:
        day = f"{when.day} {when.strftime('%b %Y')}"
    return TYPED_RULE_SCOPE.format(date=day)


def set_typed_rule(path: str, text: Any, *, typed_at: Any) -> dict[str, str]:
    """Keep the labelling rule the user typed on the Check card before its
    squares were picked, beside the sheet (``.<sheet>.rule``, mode 0600,
    written whole, no link followed). Returns ``{"text", "typed_at"}``.

    The server pins every typed rule here, the same way: the gate's, for a map
    with no owner's rule, and, by the owner's option A of 13 Nov 2026, the
    one typed in the box beside the owner's kelp rule on a map of presence
    and absence that is not of kelp, which then replaces the owner's rule for
    this sheet (:func:`labelling_rule`, ``imagery.rule_for``). The text is
    checked by :func:`clean_rule` (``ValueError``
    :data:`RULE_TEXT`), ``typed_at`` by :func:`_clean_moment`. Refused
    (:data:`RULE_AFTER_ANSWER`) once the sheet holds any answer (a label
    cell filled), so the rule is always fixed before any square is checked;
    until then, a second press writes it again. ``ValueError`` or ``OSError``
    only; nothing is written on a refusal.
    """
    real = resolve_sheet(path)
    rule = clean_rule(text)
    when = _clean_moment(typed_at)
    kept = {"format": RULE_FORMAT, "text": rule, "typed_at": when}
    with _sheet_lock(real):
        header, body = _read(real)
        if _counts(header, body)["n_labelled"]:
            raise ValueError(RULE_AFTER_ANSWER)
        _write_hidden(real, _hidden_path(real, RULE_SUFFIX), json.dumps(kept))
    return {"text": rule, "typed_at": when}


def _typed_rule_of(real: Path) -> dict[str, str] | None:
    """The typed rule beside a resolved sheet, checked as it was kept; ``None``
    when there is none, or none that reads (a link at its place included)."""
    try:
        text = _read_hidden(_hidden_path(real, RULE_SUFFIX))
    except (OSError, ValueError):
        return None
    if text is None:
        return None
    try:
        kept = json.loads(text)
    except ValueError:
        return None
    if not isinstance(kept, dict) or kept.get("format") != RULE_FORMAT:
        return None
    try:
        return {
            "text": clean_rule(kept.get("text")),
            "typed_at": _clean_moment(kept.get("typed_at")),
        }
    except ValueError:
        return None


def typed_rule(path: str) -> dict[str, str] | None:
    """The labelling rule the user typed for this sheet before the draw
    (:func:`set_typed_rule`): ``{"text", "typed_at"}``, or ``None``.
    ``ValueError`` only for a path that is no plan's sheet (:func:`resolve_sheet`)."""
    return _typed_rule_of(resolve_sheet(path))


def _labelling_rule(
    real: Path,
    design: dict[str, Any] | None,
    rows: list[dict[str, str]],
    kind: str,
) -> dict[str, str] | None:
    """:func:`labelling_rule` from what the caller read already."""
    # imagery imports this module, so it is imported here, when first asked.
    from olmoearth_agent.mcp_app import imagery

    text, scope, source = imagery.rule_for(real, _rule_names(design, rows, kind))
    if text is None or scope is None or source is None:
        return None
    return {"text": text, "scope": scope, "source": source}


def labelling_rule(path: str) -> dict[str, str] | None:
    """The labelling rule of a sheet: ``{"text", "scope", "source"}``, or ``None``.

    In this order (the owner's option A of 13 Nov 2026): ``source``
    ``"typed"``, the one the user typed before the draw and the server pinned
    beside the sheet (:func:`typed_rule`, with :func:`typed_rule_scope`, never
    ``imagery.RULE_SCOPE``), on any map, so on a map of presence and absence
    it replaces the owner's; else ``"owner"``, the app owner's rule for a map
    of presence and absence (``imagery.READING_RULE`` and
    ``imagery.RULE_SCOPE``), kept when the box was left empty; else ``None``
    (a map of other classes with no rule typed: the Check card's gate). The
    same rule the pictures print (``imagery.rule_for``).
    """
    real = resolve_sheet(path)
    with _sheet_lock(real):
        header, body = _read(real)
    rows = [dict(zip(header, row)) for row in body]
    return _labelling_rule(real, _design_of(real)[1], rows, _kind(header))


def _rule_fields(
    real: Path,
    design: dict[str, Any] | None,
    rows: list[dict[str, str]],
    kind: str,
) -> dict[str, str | None]:
    """``reading_rule``, ``rule_scope``, ``rule_source`` and ``rule_limit`` of
    the page copy and the counts (:func:`labelling_rule`); each ``None`` with
    no rule, or when the rule cannot be read.

    ``rule_limit`` is ``imagery.RULE_KELP_ONLY`` for the owner's rule (source
    ``"owner"``), else ``None``: the sheet's rule block says it after the
    scope, where no box says the rest (the last round's F9), but on a face
    that takes Cannot tell and can show pictures, where the cloud line takes
    its place (r4-m M8). ``rule_note`` is another field's name (a scores
    map's right-or-wrong note).
    """
    # imagery imports this module, so it is imported here, when first asked.
    from olmoearth_agent.mcp_app import imagery

    try:
        rule = _labelling_rule(real, design, rows, kind)
    except (OSError, ValueError):
        rule = None
    source = rule["source"] if rule else None
    return {
        "reading_rule": rule["text"] if rule else None,
        "rule_scope": rule["scope"] if rule else None,
        "rule_source": source,
        "rule_limit": imagery.RULE_KELP_ONLY if source == "owner" else None,
    }


def sheet_version(path: str) -> str:
    """The sheet's version now (:func:`_version`, the keyed digest of its bytes
    the panels compare): what a grading names (:func:`record_grading`)."""
    real = resolve_sheet(path)
    with _sheet_lock(real):
        return _version(_sheet_secret(real), real.read_bytes())


def _design_digest(design: dict[str, Any] | None) -> str | None:
    """What a graded mark is bound to: a digest (SHA-256, hex) of the design's
    ``to_label``, the windows the sheet was written from, in order, with their
    cells; ``None`` with no design file beside the sheet, or one that lists
    none.

    The design's content, not its file (the lead's ruling of 13 Nov 2026,
    round 4, findings 23 and R6): the plan tool writes the design file again,
    with the same content, when it keeps a labelled sheet whose design file
    was lost (``tools.estimation._labelled_plan``), and the digest stays the
    same. A fresh sheet planned again with the same arguments lists the same
    windows too: it is told apart by holding no answer (:func:`_mark_holds`).
    """
    listed = (design or {}).get("to_label")
    if not isinstance(listed, list):
        return None
    text = json.dumps(listed, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _holds_answers(header: list[str], body: list[list[str]]) -> bool:
    """Whether any label cell of the sheet is filled (``reference_class`` or
    ``wrong``, Cannot tell included): a sheet with none is blank, as a plan
    writes it."""
    return _counts(header, body)["n_labelled"] > 0


def _read_mark(real: Path) -> dict[str, Any] | None:
    """The graded mark beside the sheet, checked for its form; ``None`` when
    there is none or it does not read as one. Whether it is this sheet's,
    :func:`_mark_holds` says. ``ValueError`` when a link stands at its place."""
    text = _read_hidden(_hidden_path(real, GRADED_SUFFIX))
    if text is None:
        return None
    try:
        mark = json.loads(text)
    except ValueError:
        return None
    if not isinstance(mark, dict) or mark.get("format") != GRADED_FORMAT:
        return None
    for key in ("first", "last"):
        part = mark.get(key)
        if not isinstance(part, dict):
            return None
        if not all(isinstance(part.get(field), str) for field in ("version", "at")):
            return None
    gradings = mark.get("gradings")
    if not isinstance(gradings, int) or isinstance(gradings, bool) or gradings < 1:
        return None
    if not isinstance(mark.get("changed"), bool):
        return None
    return mark


def _mark_holds(
    mark: dict[str, Any] | None,
    *,
    answered: bool,
    digest: str | None,
    version: str | None = None,
) -> bool:
    """Whether a graded mark (:func:`_read_mark`) is this sheet's (the lead's
    ruling of 13 Nov 2026, round 4, R6 with finding 23).

    Not while the sheet holds no answer at all (``answered`` false,
    :func:`_holds_answers`): a blank sheet is a fresh one, planned again under
    this name, and its first save removes the mark (:func:`_drop_mark`);
    unless the app itself blanked this sheet (the last round's Q1): the mark's
    ``blanked`` is the sheet's version now (``version``, :func:`_version`),
    written when a save emptied the graded sheet (:func:`_write_keeping_mark`).
    Not when the mark names no design digest (one written before the digest
    was kept), or names another design's than the design file beside the
    sheet (``digest``, :func:`_design_digest`). A design file lost since the
    grading (``digest`` None), or none when it was graded, leaves the mark the
    sheet's: a fresh plan writes its design file with its blank sheet, and a
    sheet that holds answers is no fresh one.
    """
    if mark is None:
        return False
    if not answered:
        blanked = mark.get("blanked")
        if not (isinstance(blanked, str) and isinstance(version, str)):
            return False
        if not _same_version(blanked, version):
            return False
    first = mark["first"]
    if "design_digest" not in first:
        return False
    kept = first["design_digest"]
    if kept is not None and not isinstance(kept, str):
        return False
    return kept is None or digest is None or kept == digest


def _drop_mark(real: Path) -> None:
    """Remove the graded mark beside the sheet, if any: the first save into a
    blank sheet does (:func:`save_labels`, :func:`save_answers`), since a mark
    beside a blank sheet is an earlier sheet's of this name, unless the app
    blanked the sheet itself (:func:`_blanked_at`). A link at its place is
    removed itself, never followed. Called under :func:`_sheet_lock`."""
    _hidden_path(real, GRADED_SUFFIX).unlink(missing_ok=True)


def _held_mark(
    real: Path,
    header: list[str],
    body: list[list[str]],
    design: dict[str, Any] | None,
    version: str,
) -> dict[str, Any] | None:
    """The graded mark beside the sheet when it is this sheet's
    (:func:`_mark_holds`, with the rows and the version read), else ``None``;
    a mark that cannot be read (a link at its place included) is none. Called
    under :func:`_sheet_lock`."""
    try:
        mark = _read_mark(real)
    except (OSError, ValueError):
        return None
    holds = _mark_holds(
        mark,
        answered=_holds_answers(header, body),
        digest=_design_digest(design),
        version=version,
    )
    return mark if holds else None


def _blanked_at(real: Path, version: str) -> bool:
    """Whether the graded mark beside the sheet says the app blanked it at
    ``version`` (its ``blanked``, :func:`_write_keeping_mark`): then the
    first save into the blank sheet keeps the mark. False when there is no
    mark, or none that reads. Called under :func:`_sheet_lock`."""
    try:
        mark = _read_mark(real)
    except (OSError, ValueError):
        return False
    blanked = mark.get("blanked") if mark is not None else None
    return isinstance(blanked, str) and _same_version(blanked, version)


def _write_keeping_mark(
    real: Path,
    header: list[str],
    body: list[list[str]],
    *,
    secret: bytes,
    blank: bool,
    read_version: str,
    held: dict[str, Any] | None,
) -> str:
    """Write the sheet whole (:func:`_write_sheet`) and keep its graded mark
    true to it; returns the version written (:func:`_version`). Called under
    :func:`_sheet_lock`, with the rows already changed in place.

    ``blank``: the sheet held no answer when read (at ``read_version``). The
    first save into a blank sheet removes a graded mark beside it, an earlier
    sheet's of this name (:func:`_drop_mark`), unless the mark says the app
    blanked this sheet at that version (:func:`_blanked_at`). ``held``: the
    mark that was this sheet's before the save (:func:`_held_mark`). A save
    that empties the graded sheet (a column cleared from the panel, or every
    answer cleared one by one) keeps it, with ``changed`` true and
    ``blanked`` the version written (the lead's ruling Q1 of 13 Nov 2026), so
    answers given again after the result was seen grade as changed. When that
    mark cannot be written, the save stands and the mark is left as it was:
    no mark while the sheet holds no answer, as before the ruling.
    """
    if blank and not _blanked_at(real, read_version):
        _drop_mark(real)
    written = _version(secret, _write_sheet(real, header, body))
    if held is not None and not _holds_answers(header, body):
        kept = {**held, "changed": True, "blanked": written}
        with contextlib.suppress(OSError, ValueError):
            _write_hidden(
                real,
                _hidden_path(real, GRADED_SUFFIX),
                json.dumps(kept, sort_keys=True),
            )
    return written


def _answers_digest(header: list[str], body: list[list[str]]) -> str | None:
    """A digest (SHA-256, hex) of the sheet's answers: each window's
    ``(window_index, reference_class, wrong)``, each cell trimmed, sorted by
    window; ``None`` when a window index is no whole number.

    What the graded mark compares to tell whether the answers changed since
    the first grading (the lead's ruling Q2 of 13 Nov 2026): the label cells,
    not the file's bytes, so a sheet saved again from a spreadsheet (a
    byte-order mark, other line ends or quoting) with no answer changed is no
    change. The version (:func:`_version`, the bytes) stays for the panels'
    one live copy and the grading's race.
    """
    at_index = header.index("window_index")
    at_ref = header.index("reference_class")
    at_wrong = header.index("wrong") if "wrong" in header else None
    cells: list[tuple[int, str, str]] = []
    for row in body:
        try:
            index = int(row[at_index].strip())
        except ValueError:
            return None
        wrong = row[at_wrong].strip() if at_wrong is not None else ""
        cells.append((index, row[at_ref].strip(), wrong))
    text = json.dumps(sorted(cells), separators=(",", ":"))
    return hashlib.sha256(b"olmoearth-answers\0" + text.encode("utf-8")).hexdigest()


def _changed_since_first(
    mark: dict[str, Any], version: str, answers: str | None
) -> bool:
    """Whether the sheet's answers differ from those first graded: by the
    answers' digest the mark kept (``first.answers``, :func:`_answers_digest`)
    when both are known, else by the version (a mark written before the
    digest was kept, or answers that could not be read)."""
    kept = mark["first"].get("answers")
    if isinstance(kept, str) and answers is not None:
        return kept != answers
    return not _same_version(version, str(mark["first"]["version"]))


def record_grading(
    path: str, version: str, *, design_path: str | None = None
) -> dict[str, Any]:
    """Mark the sheet graded at ``version`` (:func:`sheet_version`, read right
    after the estimate): ``{first_at, changed, gradings}``.

    The mark is a hidden file beside the sheet (``.<sheet>.graded``, mode
    0600, written whole under the sheet's lock, no link followed):
    ``{"format": GRADED_FORMAT, "first": {version, at, design_path,
    design_digest, answers}, "last": {version, at}, "gradings", "changed"}``,
    and ``blanked`` once a save emptied the graded sheet
    (:func:`_write_keeping_mark`). The first grading sets ``first``, with the
    digest of the design's content (:func:`_design_digest`), which binds the
    mark to this sheet, and the digest of the answers graded
    (:func:`_answers_digest`, read from the sheet when its version now is
    ``version``; else ``None``); each one sets ``last`` and counts.
    ``changed`` turns true once a grading's answers differ from the first's
    (:func:`_changed_since_first`: by the answers' digest, the lead's ruling
    Q2 of 13 Nov 2026, else by the version), and stays true. A mark that does
    not read as one, or is not this sheet's (:func:`_mark_holds`: another
    design's, or beside a sheet that holds no answer and that the app did not
    blank), is started again. ``ValueError`` (:data:`BAD_VERSION`) for a
    version that is not one; ``ValueError`` or ``OSError`` only.
    """
    real = resolve_sheet(path)
    if not isinstance(version, str) or not _VERSION_RE.fullmatch(version):
        raise ValueError(BAD_VERSION)
    if design_path is not None and not isinstance(design_path, str):
        raise ValueError("design_path is the design file's path, or None")
    at = _now_text()
    place = _hidden_path(real, GRADED_SUFFIX)
    digest = _design_digest(_design_of(real)[1])
    with _sheet_lock(real):
        header, body, now = _read_stamped(real, _sheet_secret(real))
        # The answers graded, when the sheet is still the version graded.
        answers = _answers_digest(header, body) if _same_version(now, version) else None
        mark = _read_mark(real)
        holds = _mark_holds(
            mark, answered=_holds_answers(header, body), digest=digest, version=now
        )
        if not holds:
            mark = None
        if mark is None:
            mark = {
                "format": GRADED_FORMAT,
                "first": {
                    "version": version,
                    "at": at,
                    "design_path": design_path,
                    "design_digest": digest,
                    "answers": answers,
                },
                "last": {"version": version, "at": at},
                "gradings": 1,
                "changed": False,
            }
        else:
            changed = bool(mark["changed"]) or _changed_since_first(
                mark, version, answers
            )
            mark = {
                **mark,
                "last": {"version": version, "at": at},
                "gradings": int(mark["gradings"]) + 1,
                "changed": changed,
            }
        _write_hidden(real, place, json.dumps(mark, sort_keys=True))
    return {
        "first_at": str(mark["first"]["at"]),
        "changed": bool(mark["changed"]),
        "gradings": int(mark["gradings"]),
    }


def _same_version(a: str, b: str) -> bool:
    """Whether two versions are one, compared in constant time."""
    return hmac.compare_digest(a.encode("utf-8"), b.encode("utf-8"))


def graded_state(path: str) -> dict[str, Any] | None:
    """Whether the sheet was graded: ``{first_at, changed_since_first,
    changed, answered}``, or ``None`` before its first grading.

    ``changed_since_first``: its answers now are not those first graded
    (:func:`_changed_since_first`: by the answers' digest, else by the
    version; answers changed since, graded again or not); ``changed``: the
    mark's own, a later grading graded other answers, or a save emptied the
    graded sheet (:func:`record_grading`, :func:`_write_keeping_mark`);
    ``answered``: whether the sheet holds any answer now
    (:func:`_holds_answers`), false on a graded sheet a save emptied (the
    mark kept, the lead's ruling Q1), so a part cleared since is not counted
    as checked (the last round's F4, contract B2 to B1).
    ``None`` too for a mark that is not this sheet's (:func:`_mark_holds`:
    while the sheet holds no answer and the app did not blank it, or another
    design's). Read only: nothing is written. ``ValueError`` or ``OSError``
    only.
    """
    real = resolve_sheet(path)
    digest = _design_digest(_design_of(real)[1])
    with _sheet_lock(real):
        mark = _read_mark(real)
        if mark is None:
            return None
        raw = real.read_bytes()
        version = _version(_sheet_secret(real), raw)
    header, body = _parse(real, raw)
    answered = _holds_answers(header, body)
    holds = _mark_holds(mark, answered=answered, digest=digest, version=version)
    if not holds:
        return None
    answers = _answers_digest(header, body)
    return {
        "first_at": str(mark["first"]["at"]),
        "changed_since_first": _changed_since_first(mark, version, answers),
        "changed": bool(mark["changed"]),
        "answered": answered,
    }


def _graded_field(
    real: Path,
    version: str,
    *,
    header: list[str],
    body: list[list[str]],
    design: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """The page copy's and the counts' ``graded``: ``{first_at,
    changed_since_first}`` for the rows read (``header``, ``body``) at
    ``version`` (:func:`_changed_since_first`), or ``None`` before a first
    grading, when the mark cannot be read, or when it is not this sheet's
    (:func:`_mark_holds`, with whether the rows hold an answer, ``version``
    and the design read with them)."""
    try:
        mark = _read_mark(real)
    except (OSError, ValueError):
        return None
    holds = _mark_holds(
        mark,
        answered=_holds_answers(header, body),
        digest=_design_digest(design),
        version=version,
    )
    if mark is None or not holds:
        return None
    return {
        "first_at": str(mark["first"]["at"]),
        "changed_since_first": _changed_since_first(
            mark, version, _answers_digest(header, body)
        ),
    }


#: A sheet labelled with the class seen, as graded: its judged squares as
#: ``(code, wrong)``, its classes as ``(code, name, outside)`` in order
#: (``outside``: a class typed that the map does not write), and how many
#: squares say Cannot tell (:func:`_seen_squares`, :func:`_class_squares`).
_Judged = tuple[list[tuple[str, bool]], list[tuple[str, str, bool]], int]


def _seen_squares(
    design: dict[str, Any] | None, rows: list[dict[str, str]]
) -> _Judged | None:
    """A seen sheet's judged squares as ``(code, wrong)``, its classes as
    ``(code, name, outside)`` in the answers' order (:func:`_seen_codes`;
    ``outside`` false, since a seen sheet answers with the design's codes
    alone), and how many squares say Cannot tell (``?`` in both cells); or
    ``None`` when a square holds neither a code the design names with a
    ``wrong`` of 0 or 1 nor Cannot tell (a blank, or a code typed with its
    ``wrong`` not yet worked out). ``wrong`` is the cell the estimate graded."""
    codes = _seen_codes(design) or []
    names = _class_names(design)
    squares: list[tuple[str, bool]] = []
    unjudged = 0
    for row in rows:
        ref = (row.get("reference_class") or "").strip()
        wrong = (row.get("wrong") or "").strip()
        if ref == UNJUDGED and wrong == UNJUDGED:
            unjudged += 1
            continue
        if ref not in codes or wrong not in ("0", "1"):
            return None
        squares.append((ref, wrong == "1"))

    def named(code: str) -> str:
        """The code's name, its first letter a capital, as the answers say it."""
        return names[code][:1].upper() + names[code][1:]

    order = [(code, named(code), False) for code in codes]
    return squares, order, unjudged


def _class_squares(
    design: dict[str, Any] | None, rows: list[dict[str, str]]
) -> _Judged | None:
    """A class map's sheet labelled with the class seen, as ``(code, wrong)``
    per judged square, its classes as ``(code, name, outside)`` and how many
    squares say Cannot tell, or ``None``.

    Graded as the estimate grades it (``tools.estimation``): the class seen
    matched to the design's classes as the map writes them, case, outer
    spaces and ``3.0`` against ``3`` aside, and wrong where it is not the
    design's map class there (``population.map_class``, never the sheet's
    cell). A class the map does not write is a class of its own after the
    map's (``outside`` true), and wrong. The classes: the design's, in its
    order, then any such class in the order first seen; each class's code is
    its name as written. Cannot tell (``?`` in ``reference_class``, with
    ``wrong`` blank or ``?``: the last round's S4) is counted apart, never a
    class. ``None`` when a square has no class seen, says Cannot tell in one
    column only, its window has no class on the design's map, or the design
    lists no classes.
    """
    from olmoearth_agent.tools.estimation import _class_key

    population = (design or {}).get("population")
    population = population if isinstance(population, dict) else {}
    classes = population.get("classes")
    on_map = population.get("map_class")
    if not isinstance(classes, list) or not classes or not isinstance(on_map, list):
        return None
    written = [str(c) for c in classes]
    code_of = {_class_key(value): at for at, value in enumerate(classes)}
    outside: dict[str, str] = {}
    squares: list[tuple[str, bool]] = []
    unjudged = 0
    for row in rows:
        ref = (row.get("reference_class") or "").strip()
        wrong = (row.get("wrong") or "").strip()
        if ref == UNJUDGED:
            if wrong not in ("", UNJUDGED):
                return None
            unjudged += 1
            continue
        if not ref or wrong == UNJUDGED:
            return None
        try:
            index = int(str(row.get("window_index")).strip())
        except ValueError:
            return None
        mapped = on_map[index] if 0 <= index < len(on_map) else None
        if not isinstance(mapped, int) or isinstance(mapped, bool) or mapped < 0:
            return None
        key = _class_key(ref)
        at = code_of.get(key)
        if at is None:
            code = outside.setdefault(key, ref)
            squares.append((code, True))
        else:
            squares.append((written[at], at != mapped))
    order = [(name, name, False) for name in dict.fromkeys(written)]
    order += [(code, code, True) for code in outside.values() if code not in written]
    return squares, order, unjudged


def _judged(path: str, version: Any) -> tuple[str, _Judged] | None:
    """A sheet labelled with the class seen, as graded at ``version``: its face
    (``"seen"``, a named scores map's; ``"class"``, a class map's) and its
    squares (:data:`_Judged`). ``None`` when ``version`` is not the sheet's
    version now (answers changed since the estimate read them), for a sheet
    not labelled with the class seen (right or wrong, or two maps), or when a
    square cannot be counted (:func:`_seen_squares`, :func:`_class_squares`).
    ``ValueError`` or ``OSError`` for a sheet that cannot be read."""
    real = resolve_sheet(path)
    design = _design_of(real)[1]
    with _sheet_lock(real):
        header, body, now = _read_stamped(real, _sheet_secret(real))
    if not isinstance(version, str) or not _same_version(version, now):
        return None
    kind = _kind(header)
    rule = _rule(design, kind, header, body)
    rows = [dict(zip(header, row)) for row in body]
    if rule == "seen":
        face, got = "seen", _seen_squares(design, rows)
    elif rule == "class" and kind == "one_map":
        face, got = "class", _class_squares(design, rows)
    else:
        return None
    return None if got is None else (face, got)


def _class_rows(
    squares: list[tuple[str, bool]], order: list[tuple[str, str, bool]]
) -> list[dict[str, Any]]:
    """Each class of ``order`` as ``{"code", "name", "seen", "agreed",
    "outside"}``: on how many judged squares the reviewer saw it, on how many
    of those the map agreed, and whether the map writes it at all."""
    rows: list[dict[str, Any]] = []
    for code, name, outside in order:
        mine = [wrong for seen, wrong in squares if seen == code]
        rows.append(
            {
                "code": code,
                "name": name,
                "seen": len(mine),
                "agreed": mine.count(False),
                "outside": outside,
            }
        )
    return rows


def graded_squares(path: str, version: str) -> dict[str, Any] | None:
    """The squares of a graded sheet, for the result card's figure and counts.

    ``{n, n_unjudged, face, cells, classes}``: ``n`` squares, judged and
    Cannot tell; ``n_unjudged``, those that say Cannot tell; ``face``
    ``"seen"`` (a named scores map's sheet) or ``"class"`` (a class map's,
    labelled with the class seen); ``cells``, one ``{"seen": code, "wrong":
    bool}`` per judged square, grouped by class in the order of ``classes``,
    right before wrong, then one ``{"seen": None, "wrong": False,
    "unjudged": True}`` per Cannot tell square (the owner's O2 of 14 Nov
    2026: the card draws it as an empty dashed square, contract B2 to B1);
    ``classes``, each ``{"code", "name", "seen", "agreed", "outside"}``
    (:func:`_class_rows`) over the judged squares only: how many squares the
    reviewer saw it in, on how many of those the map agreed, and whether it
    is a class typed that the map does not write. Every class the sheet
    answers with is listed, seen or not. Counts only, from the sheet as
    graded: no place, window index or order.

    ``None`` when ``version`` is not the sheet's version now (answers
    changed since the estimate read them), when a square has no answer, when
    no square is judged (every one Cannot tell: no grid), or for a sheet not
    labelled with the class seen (right or wrong, or two maps).
    ``ValueError`` or ``OSError`` only.
    """
    got = _judged(path, version)
    if got is None:
        return None
    face, (squares, order, unjudged) = got
    if not squares:
        return None
    classes = _class_rows(squares, order)
    cells: list[dict[str, Any]] = []
    for row in classes:
        code, right = row["code"], row["agreed"]
        cells.extend({"seen": code, "wrong": False} for _ in range(right))
        cells.extend({"seen": code, "wrong": True} for _ in range(row["seen"] - right))
    cells.extend(
        {"seen": None, "wrong": False, "unjudged": True} for _ in range(unjudged)
    )
    return {
        "n": len(squares) + unjudged,
        "n_unjudged": unjudged,
        "face": face,
        "cells": cells,
        "classes": classes,
    }


def class_counts(path: str, version: str) -> dict[str, Any] | None:
    """The counts of a graded sheet labelled with the class seen, Cannot tell
    counted apart: what the result card says the check did not cover (the
    last round's R2 and M1, contract B2 to B1).

    ``{n, n_unjudged, classes}``: ``n``, every square of the sheet (judged or
    Cannot tell); ``n_unjudged``, those that say Cannot tell; ``classes``,
    each ``{"code", "name", "seen", "agreed", "outside"}`` over the judged
    squares only (:func:`_class_rows`): every class the sheet answers with,
    seen or not, and ``outside`` true for a class typed that the map does not
    write. Counts only: no place, window index or order.

    ``None`` when ``version`` is not the sheet's version now, for a
    right-or-wrong or two-map sheet, when a square holds no answer, when the
    sheet has no square, or when it cannot be read (never raises on a sheet
    that cannot be read).
    """
    try:
        got = _judged(path, version)
    except (OSError, ValueError, csv.Error):
        return None
    if got is None:
        return None
    _face, (squares, order, unjudged) = got
    n = len(squares) + unjudged
    if not n:
        return None
    return {"n": n, "n_unjudged": unjudged, "classes": _class_rows(squares, order)}


def result_holds_answers(result_id: Any) -> bool:
    """Whether any plan's sheet of this Studio result already holds an answer
    (the Check card's settled rule, the last round's Q6 and Q9: a rule typed
    on the card must not stand beside a sheet labelled under another).

    Among :func:`_sheet_files` (the tops of the two roots), the sheets whose
    names start with ``design_`` and the result's slug as the plan tool names
    them (``tools.review_set.slug(result_id, 12)``, then ``_``), told by name
    alone; at most ``check.SCAN_MAX`` of those are read, without the lock (a
    save replaces a sheet whole). True for the first that holds a label cell
    (:func:`_counts`, Cannot tell included) and whose design file beside it
    names this result (``population.source.result_id``): a slug cut to 12
    characters can be another result's. A sheet or design that cannot be
    read is skipped. Never raises: anything unexpected is no answer.
    """
    try:
        # check imports this module, so it is imported here, when first asked.
        from olmoearth_agent.mcp_app import check

        if not isinstance(result_id, str) or not result_id.strip():
            return False
        wanted = result_id.strip()
        prefix = "design_" + slug(wanted, 12) + "_"
        files = [path for path, _mtime in _sheet_files()]
        named = [p for p in files if os.path.basename(p).startswith(prefix)]
        limit = check.SCAN_MAX
        for path in named[:limit]:
            try:
                real = resolve_sheet(path)
                header, body = _read(real)
                if not _counts(header, body)["n_labelled"]:
                    continue
                named_id = _population_source(_design_of(real)[1]).get("result_id")
            except (OSError, ValueError, csv.Error):
                continue
            if isinstance(named_id, str) and named_id.strip() == wanted:
                return True
    except Exception:  # noqa: BLE001 - never raises: unknown is no answer
        return False
    return False
