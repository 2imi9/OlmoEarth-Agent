# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Pictures of a label sheet's window, for the reviewer (the sheet panel).

The owner agreed on 11 Nov 2026 (docs/Usage.md, "In Claude, with views"):

- The MCP server fetches small dated Sentinel-2 pictures of a window from a
  public source (:mod:`s2_fetch`): Earth Search's catalogue
  (:data:`CATALOGUE_HOST`) receives the map's extent, rounded outward to 0.1
  degree, and its period, once per sheet; the public bucket
  (:data:`BUCKET_HOST`) receives byte-range requests for parts of the image
  files. Both see this computer's IP address. This is the one exception to
  "no STAC or Earth Engine fetches". It is off until the user switches it on
  in the panel, after reading :data:`CONSENT_ASK` (three short lines since
  round r5-k K7; the hosts by name in its fold, :data:`CONSENT_FULL`); the
  answer is kept in the output root (:func:`consent_path`) with what it was
  given to (:func:`consent_version`), so a changed question asks again.
- Where the window is comes from files on disk, never from the page, which
  names the window by its handle alone (``sheet.window_row``): the sheet,
  its design (``population.source.scores_path``), the scores file
  (``grid``, ``patch_px``, ``raster.path``, ``date_window``) and the map
  raster's header (CRS, transform). The window's centre is checked against
  the sheet's own row before anything is fetched.
- Which dates: for each calendar month of the map's period, the scene with
  the least scene cloud whose data covers the picture (:data:`PICK_RULE`).
  The pick never looks at the window's map class, stratum or order, so a
  blind sheet stays blind. Pictures per date: near-infrared false colour and
  true colour, a context view with the window outlined and a detail view, on
  the map's own grid by nearest neighbour, one fixed stretch
  (:data:`STRETCH`, frozen on a public scene unrelated to any project).
- The labelling rule is printed with the pictures and kept in the record
  (:func:`rule_for`; the record names whose rule it is, ``rule_source``): the
  rule the user typed on the Check card before its squares were picked
  (``sheet.typed_rule``, the owner's rulings of 13 Nov 2026), on any map, so
  on a map of presence and absence whose maker meant something else by
  Present and Absent it replaces the app's (option A); else the app's
  (:data:`READING_RULE`, general for any map of presence and absence since
  round r5-l, what the map is meant to find since round r5-m; a sheet made
  with the earlier kelp rule or r5-l's words keeps those words,
  :func:`app_rule_of`) for a map of presence and absence; else none, and the
  pane says one must be agreed.
- What a label then is (:data:`LABEL_IS`): a visual reading of Sentinel-2,
  the map's own sensor; the estimate's ``must_state`` says so
  (:data:`LABEL_MUST_STATE`, added by the server when the sheet's record
  shows pictures were served).
- A record beside the sheet (:func:`record_path`, JSON lines) keeps what was
  shown: the search, and per window the scenes per month, the rule and the
  stretch. The pictures themselves are not kept.
- A public test scene (:func:`public_test_scene`), unrelated to any project, lets
  the user see the fetch work before the sheet's extent leaves the computer.

A review set's listed window (:func:`review_image`, round 4) is pictured on
the same map path: the result panel names it by the review's opaque key and
its rank (:mod:`review`), the scores file it was ranked from places it (its
raster, window size, period and the window's centre), and its record lies
beside the review's (``review_sets/<key>.imagery.jsonl``); nothing is
labelled from it, so it carries no rule. A review set of single sampled
pixels (a Studio result's grid) gets no pictures. Its windows' centres are
read where the street map and the CSV read them (``review.window_centre``,
round r4-k K7).

:func:`example_scene` (round r4-k K9) renders a fixed public place the way
a sheet's square is rendered, for a cluster run that makes the two example
pictures beside the app's rule (from a kelp canopy map); the app never calls
it.

What the view gets travels in the tool's ``structuredContent`` only; the
model reads ``{"ok": true}``, a failure included. Errors are a closed set of
fixed sentences: no coordinate, path, host or scene id. A failure another
try can get past (a slow fetch, the catalogue, the image files) carries
``again``, and the pane offers "Try again" (:data:`RETRY_SENTENCES`, round
r5-a L1). An answer for a square on which no month of the period has a
scene with data carries ``no_image``, and the pane says one line in place of
its empty months (:func:`_no_image`, r5-a L2); a search the catalogue cut
short is told by the record's search line alone.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import importlib
import json
import logging
import math
import os
import tempfile
import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from olmoearth_agent.mcp_app import review, s2_fetch, sheet
from olmoearth_agent.reporting.figures import png
from olmoearth_agent.security.paths import workspace_root
from olmoearth_agent.tools import review_set

try:  # pragma: no cover - Windows has no fcntl
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

#: The two hosts the fetch talks to, named in the question to the user.
CATALOGUE_HOST = s2_fetch.CATALOGUE_HOST
BUCKET_HOST = s2_fetch.BUCKET_HOST

#: The question the panel asks once, before any fetch: who receives what.
#: Round r5-k K7 (the owner's choice N1, accepted 18 Nov 2026): its face is
#: three short lines, what is sent (:data:`CONSENT_WHAT`: the map's extent,
#: rounded outward to 0.1 degree by :func:`s2_fetch.rounded`, and its months,
#: one search per month), to whom (:data:`CONSENT_TO`: the catalogue, and the
#: file store that gets the byte-range requests; both see the IP address)
#: and that nothing goes to Studio or Claude (:data:`CONSENT_NOT`). The first
#: line, the question and what is sent, takes two rows of the panel at 780
#: px; the other two, at most 90 characters, one row each (the round's
#: review, D1). The two hosts by name and what each gets are in the
#: question's fold (:data:`CONSENT_FULL`, "What is sent, in full" in the
#: panel), with the test scene's question, which calls them the same
#: catalogue and the same file store (review D3). Before, one paragraph of
#: about 11 lines named both hosts (the reader test's A/sheet-7, 13 Nov 2026;
#: round 4's D2 named sheets and review sets). The words changed, so the hash
#: (:func:`consent_version`) did, and everyone is asked once more.
# WORDING: r5k K7
CONSENT_WHAT = (
    "Show satellite (Sentinel-2) pictures of each square? To fetch them, this "
    "computer sends the box around the map, rounded out to 0.1 degree, and the "
    "map's months."
)
# WORDING: r5k K7
CONSENT_TO = (
    "They go to Earth Search, a public catalogue, and its file store; both see "
    "your IP address."
)
# WORDING: r5k K7
CONSENT_NOT = "Nothing is sent to Studio or to Claude."
#: The face's three lines, in order, and the question as the panel gets it
#: (``ask``): the three lines joined by line breaks, drawn one line each.
CONSENT_LINES = (CONSENT_WHAT, CONSENT_TO, CONSENT_NOT)
CONSENT_ASK = "\n".join(CONSENT_LINES)
#: The question's fold (``ask_full``): each host by name and what it gets
#: (:mod:`s2_fetch`: the catalogue, the extent and the month of each search,
#: kept in the record, so the squares of a sheet or a review set are searched
#: once; the file store, byte-range requests for parts of four band files),
#: and that no credential is sent. The test scene's question
#: (:data:`TEST_ASK`, :data:`REVIEW_TEST_ASK`) follows it in the fold.
# WORDING: r5k K7
CONSENT_FULL = (
    f"Earth Search's catalogue is {CATALOGUE_HOST}: it gets the box and the "
    "months, once for all the squares. Its file store is "
    f"{BUCKET_HOST}: it gets the requests for parts of image files. No key and "
    "no account are used."
)

#: Whether this build fetches pictures. True since the fetch landed; an answer
#: given to the stub ("-stub") does not count.
FETCHES = True


def consent_version() -> str:
    """What an answer was given to: the question's own words, its face
    (:data:`CONSENT_ASK`) and its fold (:data:`CONSENT_FULL`), as a hash,
    and whether the build fetches, so a changed question, or the first build
    that fetches, asks again. A yes is kept with this version
    (:func:`set_on`) and counts while the version is the same
    (:func:`is_on`): round r5-k K7 changed the words, so a yes given before
    lapses once, and a yes given to these words counts."""
    asked = CONSENT_ASK + "\n\n" + CONSENT_FULL
    digest = hashlib.sha256(asked.encode("utf-8")).hexdigest()[:16]
    return f"{digest}-{'fetch' if FETCHES else 'stub'}"


#: The app's labelling rule for a map of presence and absence, printed above
#: the pictures. Round r5-l (the owner, 19 Nov 2026: the instructions are
#: general, no kelp in what people are asked): it reads for any map of present
#: and absent. Round r5-m B1: it says what the map is meant to find, which
#: cannot be read as the map's own answer at the square (r5-l said "what the
#: map shows", read as that answer, which the blind sheet hides). The rule the
#: owner agreed on 11 Nov 2026 for the kelp trial (:data:`KELP_READING_RULE`)
#: and r5-l's words (:data:`R5L_READING_RULE`) are kept word for word for the
#: records made with them.
# WORDING: r5m B1
# WORDING: r5l
READING_RULE = (
    "Present: on at least one clear date shown, what the map is meant to find "
    "covers more than half of the outlined square. Absent: on every clear date "
    "shown, it covers half or less."
)

#: The app's rule before round r5-l, agreed by the owner on 11 Nov 2026 for a
#: map of presence and absence (the kelp trial), word for word. Records
#: already written keep their words: a draw, a sheet or a result made with
#: this rule still shows and repeats it (:func:`app_rule_of`,
#: ``check.result_app_rule``); a new part of a map already drawn with it, or
#: with a sheet labelled under it, takes it; nothing else drawn since is
#: given it, and a press that drew nothing keeps it for no map.
KELP_READING_RULE = (
    "Present: on at least one clear date shown, canopy fills more than half of "
    "the outlined square. Absent: on every clear date shown it fills half or less."
)

#: The app's rule of round r5-l, word for word, kept for the records made with
#: it (round r5-m B1: the owner's trial runs of staging): a draw, a sheet or a
#: result that holds these words keeps them and the general scope
#: (:data:`RULE_SCOPE`, :func:`app_rule_scope`), read as r5-l reads the kelp
#: rule's records (:data:`APP_RULES`, :func:`app_rule_of`); nothing new is
#: given them.
R5L_READING_RULE = (
    "Present: on at least one clear date shown, what the map shows covers more "
    "than half of the outlined square. Absent: on every clear date shown, it "
    "covers half or less."
)

#: What a label read from these pictures is (the pane and the record). Word
#: for word as the reader test kept it, but "square size" for "window size":
#: the pane is a face (the owner's ruling 6 of 13 Nov 2026), and only a
#: sheet of squares has pictures.
# WORDING: squares
LABEL_IS = (
    "These labels are a visual reading of Sentinel-2 pictures from the map's own "
    "period, the sensor the map was made from. What the pictures cannot show "
    "(submerged, sparse, smaller than a pixel, under glint) is missed by both, so "
    "the result is agreement with this reading at this square size, not ground truth."
)

#: The estimate's must_state when the sheet's labels were read off pictures.
LABEL_MUST_STATE = (
    "The labels are a visual reading of Sentinel-2 pictures from the map's own "
    "period: the result is agreement with that reading, not ground truth."
)

#: Said with the app's rule: who set it and for what (the reader test's
#: A/sheet-6, 13 Nov 2026; shortened by round r4-h C5; round r5-l: for any
#: map of present and absent, no date, no kelp): the Check card, the sheet,
#: the pictures and the model say it, and nothing after it (round r5-l took
#: away the line that any other map needs its own rule). A rule the user
#: typed before the draw has its own (``sheet.TYPED_RULE_SCOPE``), the kelp
#: rule its own (:data:`KELP_RULE_SCOPE`, :func:`app_rule_scope`).
# WORDING: r5l
RULE_SCOPE = "This rule came with the app, for any map of present and absent."

#: The scope said with :data:`KELP_READING_RULE`, word for word as round
#: r4-h C5 left it, for the records made with that rule (:func:`app_rule_scope`).
# WORDING: r4h C5
KELP_RULE_SCOPE = "This rule came with the app (set 11 Nov 2026 for kelp canopy maps)."

#: The app's rules by their words, each with the scope said with it: the
#: general rule, the kelp rule a record made before round r5-l holds, and
#: r5-l's words, which a record made before round r5-m holds, with the
#: general scope (round r5-m B1).
APP_RULES: dict[str, str] = {
    READING_RULE: RULE_SCOPE,
    KELP_READING_RULE: KELP_RULE_SCOPE,
    R5L_READING_RULE: RULE_SCOPE,
}

#: How the dates are chosen (the record; the caption says it shorter).
PICK_RULE = (
    "For each calendar month of the map's period, the scene with the least scene "
    "cloud whose data footprint covers the picture and whose data covers at least "
    f"{round(s2_fetch.DATA_SHARE * 100)}% of the outlined square, up to "
    f"{s2_fetch.TRIES_PER_MONTH} scenes tried; the same rule for every window, "
    "blind to its map class, stratum and order."
)

#: The fixed stretch, frozen on a public scene unrelated to any project
#: (12 Nov 2026). A change makes a new version.
STRETCH: dict[str, Any] = {
    "version": "s2c1-linear@1",
    "collection": s2_fetch.COLLECTION,
    "reflectance": s2_fetch.REFLECTANCE,
    "nir_false_colour": {"bands": ["nir", "red", "green"], "range": [0.0, 0.30]},
    "true_colour": {"bands": ["red", "green", "blue"], "range": [0.0, 0.20]},
    "no_data": "magenta",
    "resampling": "nearest neighbour, onto the map's own grid",
}

#: The pane's caption under the dates.
CAPTION = (
    "One date for each month of the map's period: the scene with the least scene "
    "cloud that has data here (dates in UTC). Near-infrared false colour shows "
    "vegetation red and water dark. Magenta: no data."
)

#: Said in front of :data:`CAPTION` on a sheet's own pictures: how big the
#: outlined square is, from its scores file (:func:`window_size`,
#: :func:`square_size_text`). The half of the reader test's A/check-4 that
#: puts the size "in the pictures" (round 4 pass 2, finding 8); CAPTION itself
#: stays word for word. ``{metres}`` is :data:`SQUARE_METRES`, or nothing off
#: a UTM grid.
# WORDING: squares
SQUARE_SIZE = "A square is {pixels} x {pixels} map pixels{metres}. "
# WORDING: squares
SQUARE_METRES = ", about {metres} m a side"

#: The test scene: a public place unrelated to any project (the Cape
#: Peninsula's Atlantic shore), a fixed period, the kelp map's window size.
TEST_PLACE = (18.3507, -33.9864)  # lon, lat
TEST_EPSG = 32734
TEST_PERIOD = ("2024-01-01", "2024-03-31")
TEST_PATCH = 11
#: The test scene's opening, why it is there (the reader test's A/sheet-8,
#: 13 Nov 2026), before what it sends.
# WORDING: r4p2
TEST_WHY = (
    "Not sure yet? See how the pictures look on a public test place first. "
    "Nothing about your map is sent."
)
#: The test scene's question beside a sheet. Since round r5-k K7 the panel
#: draws it in the question's fold, right after :data:`CONSENT_FULL`, so it
#: calls the two hosts "the same catalogue" and "the same file store" (the
#: round's review D3: each host is named once, by one name; the face says
#: both see the IP address, and :data:`TEST_WHY` that nothing of the map is
#: sent); the test scene's button follows the fold.
# WORDING: r5k K7
TEST_ASK = (
    f"{TEST_WHY} It is a public place unrelated to this sheet (the Cape "
    "Peninsula's Atlantic shore, January to March 2024): that place's box, "
    "rounded, goes to the same catalogue, and requests for parts of image files "
    "to the same file store. Pictures stay off for the sheet."
)
#: The test scene's caption beside a sheet: "square", the sheet face's word
#: (round 4 pass 2, the lead's ruling W; a sheet that can have pictures is
#: one of squares, ``sheet.window_unit``). Round r5-l: no "as on the kelp
#: map" (the sheet beside it may be of any map).
# WORDING: squares; r5l
TEST_CAPTION = (
    "A public test scene, not a square of this sheet: nothing here is labelled. "
    "The outlined square is 11 pixels. " + CAPTION
)
#: The test scene's question, asked beside a review set's window (in the
#: question's fold, as :data:`TEST_ASK`).
# WORDING: r4p2; r5k K7
REVIEW_TEST_ASK = (
    f"{TEST_WHY} It is a public place unrelated to this review set (the Cape "
    "Peninsula's Atlantic shore, January to March 2024): that place's box, "
    "rounded, goes to the same catalogue, and requests for parts of image files "
    "to the same file store. Pictures stay off for it."
)
#: The test scene's caption beside a review set's window (round 4's low
#: findings 1, 6 and 9): no sheet, no label and no kelp map, which a review
#: set has none of.
# WORDING: squares
REVIEW_TEST_CAPTION = (
    "A public test scene, not a square of this review set: nothing here is "
    "ranked. The outlined square is 11 pixels; this review set's squares may "
    "differ in size. " + CAPTION
)

#: The closed set of errors the panel may show.
NOT_READABLE = "The sheet could not be read."
#: A handle that names no window of the sheet (``sheet.UNKNOWN_HANDLE``).
NOT_ON_SHEET = sheet.UNKNOWN_HANDLE
#: A page drawn before the sheet changed (``sheet.CHANGED_UNDER_PAGE``), and
#: a panel of an older server, which names a window by its index.
CHANGED = sheet.CHANGED_UNDER_PAGE
OLDER_PANEL = sheet.OLDER_PANEL
#: Why a sheet's pictures are refused when its plan names no map raster that
#: places its squares (a sheet that can show pictures is one of squares:
#: ``sheet.pictures_possible``). No "window" on a face (the owner's ruling 6
#: of 13 Nov 2026); the sheet's map button is "Street map" (reader test 2's
#: A/F21).
# WORDING: r4h A/F21
NO_FOOTPRINT = (
    "Pictures need a plan drawn from a scores file, whose map raster places the "
    "square; this sheet's plan has none. Street map still works."
)
#: Why a square's pictures are refused when the raster and the sheet disagree.
# WORDING: squares
MISMATCH = (
    "The square's place in the map raster does not match its row on the sheet, "
    "so no pictures are shown."
)
NO_PERIOD = "The map's scores file records no period, so the dates to show are unknown."
TWO_PERIODS = "The sheet's two maps cover different periods; pictures need one period."
LONG_PERIOD = "The map's period is longer than 12 months; pictures cover at most 12."
NEEDS_EXTRA = "Pictures need the inferencex extra (numpy and rasterio)."
CATALOGUE_FAILED = (
    "Earth Search's catalogue did not answer as expected. Try again later."
)
FETCH_FAILED = (
    "The image files could not be read completely, so no date is shown (a label is "
    "never made on part of the dates). Try again."
)
TOO_SLOW = (
    "Fetching took longer than 40 seconds. Try again: image parts already fetched "
    "are kept for a while."
)
NO_RECORD = (
    "The record of what is shown could not be written beside the sheet, so the "
    "pictures are not shown."
)
UNEXPECTED = "The pictures could not be made (an unexpected error, logged)."

# A review set's own sentences (its windows come from a scores file, not a
# sheet; the rest of the closed set is shared).
# WORDING: held for the reader test
#: The map raster the review set's scores file names cannot be used here.
REVIEW_NO_FOOTPRINT = (
    "The map raster this review set's scores file names cannot be read here, so "
    "no pictures are shown."
)
#: A window whose place in the raster disagrees with the scores file (a
#: review set with pictures is one of a scores file's squares; a Studio
#: result's points get none). No "window" on a face (the owner's ruling 6 of
#: 13 Nov 2026).
# WORDING: squares
REVIEW_MISMATCH = (
    "The square's place in the map raster does not match its scores file, so no "
    "pictures are shown."
)
# WORDING: held for the reader test
#: The record beside the review set could not be written.
REVIEW_NO_RECORD = (
    "The record of what is shown could not be written beside the review set, so "
    "the pictures are not shown."
)
# WORDING: r4p2
#: :data:`FETCH_FAILED` beside a review set's window or its test scene, without
#: the label clause: nothing is labelled there (round 4's low findings 1, 6, 9).
REVIEW_FETCH_FAILED = (
    "The image files could not be read completely, so no date is shown. Try again."
)

_FETCH_SENTENCE = {
    "catalogue": CATALOGUE_FAILED,
    "fetch": FETCH_FAILED,
    "budget": FETCH_FAILED,
    "deadline": TOO_SLOW,
}
#: A fetch's failure beside a review set: :data:`REVIEW_FETCH_FAILED` in place
#: of :data:`FETCH_FAILED`.
_REVIEW_FETCH_SENTENCE = {
    **_FETCH_SENTENCE,
    "fetch": REVIEW_FETCH_FAILED,
    "budget": REVIEW_FETCH_FAILED,
}

#: The refusals another try can get past (r5-a L1): a slow fetch, the
#: catalogue, the image files. A pane's answer with one of them carries
#: ``again`` (true), and the pane then offers "Try again", which repeats the
#: call for the same square. Any other refusal (the sheet changed, no map
#: raster, the period) would be the same on a second try: no ``again``.
RETRY_SENTENCES = frozenset(
    {CATALOGUE_FAILED, FETCH_FAILED, REVIEW_FETCH_FAILED, TOO_SLOW}
)

MAX_MONTHS = 12
#: Picture calls running at once (the design: two at a time), and how long a
#: call waits for its turn or for another window's search of the same sheet.
_calls = threading.BoundedSemaphore(2)
WAIT_SECONDS = 40.0
_CONTEXT = 7  # the context picture, in windows a side (the window in the middle)
_DETAIL = 3  # the detail picture, in windows a side
_SCREEN = 240  # about this many screen pixels a side per picture
_NAMES = {"nir": "near-infrared false colour", "true": "true colour"}


class _Refused(Exception):
    """A window that cannot be pictured; the message is one fixed sentence."""


def _ask_answer(test_ask: str = TEST_ASK) -> dict[str, Any]:
    """The answer while pictures are off: the question's face (``ask``, its
    three lines, :data:`CONSENT_ASK`), its fold (``ask_full``,
    :data:`CONSENT_FULL`), and the test scene's (a sheet's, or a review
    set's :data:`REVIEW_TEST_ASK`), which the panel draws in the fold too
    (round r5-k K7)."""
    return {
        "ok": True,
        "needs_consent": True,
        "ask": CONSENT_ASK,
        "ask_full": CONSENT_FULL,
        "test_ask": test_ask,
    }


def _failed(said: str) -> dict[str, Any]:
    """A fetch's refusal, for the pane: its fixed sentence, with ``again``
    (true) when another try can get past it (:data:`RETRY_SENTENCES`, r5-a
    L1); the key is left out otherwise."""
    out: dict[str, Any] = {"ok": False, "error": said}
    if said in RETRY_SENTENCES:
        out["again"] = True
    return out


def _no_image(results: list[tuple[s2_fetch.Scene | None, Any, int]]) -> bool:
    """Whether no month of the map's period has a scene with data on the
    square (r5-a L2): every month's scene is ``None`` (:func:`_dates` then
    lists empty months only). The pane says one fixed line in place of that
    list. A search cut short (``complete`` false in the record's search
    line) is told by the record alone, never by the pane."""
    return all(scene is None for scene, _, _ in results)


# ------------------------------------------------------------------ consent


def consent_path() -> Path:
    """Where the user's answer is kept: the output root, beside the sheets."""
    return workspace_root() / ".imagery_on.json"


def is_on() -> bool:
    """Whether the user switched pictures on, for the current question."""
    try:
        data = json.loads(consent_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return (
        isinstance(data, dict)
        and data.get("on") is True
        and data.get("version") == consent_version()
    )


def set_on(on: bool) -> dict[str, Any]:
    """Keep the user's answer (on: written; off: removed) and return the status."""
    path = consent_path()
    if on:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".imagery_on.", dir=str(path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump({"on": True, "version": consent_version()}, fh)
            os.replace(tmp, path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise
    else:
        with contextlib.suppress(FileNotFoundError):
            path.unlink()
    return status()


def status() -> dict[str, Any]:
    """The switch and the question, for the panel."""
    return {"on": is_on(), "ask": CONSENT_ASK, "version": consent_version()}


def reading_rule(names: Iterable[Any]) -> str | None:
    """The rule for a map of presence and absence (its class names include a
    present and an absent), else ``None``: the app's rule is for such a map
    (:data:`READING_RULE`, general since round r5-l; which of the app's rules
    a sheet shows, :func:`app_rule_of` says), and another map's rule has to
    be agreed before the first label. The names
    are a scores map's ``class_names`` or a class map's ``classes``; which
    are present and absent is the sheet's (``sheet.PRESENT_NAMES``,
    ``sheet.ABSENT_NAMES``), whose seen answers put them first."""
    seen = {str(n).strip().lower() for n in names}
    present, absent = sheet.PRESENT_NAMES, sheet.ABSENT_NAMES
    return READING_RULE if seen & present and seen & absent else None


def rule_for(
    real: Path, names: Iterable[Any]
) -> tuple[str | None, str | None, str | None]:
    """The labelling rule of a resolved sheet: ``(text, scope, source)``.

    In this order (the owner's option A of 13 Nov 2026): the rule the user
    typed on the Check card before its squares were picked, pinned beside the
    sheet at the press (``sheet.typed_rule``: ``sheet.typed_rule_scope``,
    never :data:`RULE_SCOPE`, ``"typed"``), on any map, so on a map of
    presence and absence whose maker meant something else it replaces the
    app's; else the app's rule for a map of presence and absence
    (:func:`reading_rule`), kept when nothing was typed: the one the sheet's
    records hold (:func:`app_rule_of`: :data:`READING_RULE`, or
    :data:`KELP_READING_RULE` for a sheet made with it before round r5-l)
    with its own scope (:func:`app_rule_scope`), ``"owner"``; else
    ``(None, None, None)`` (a map of other classes with no rule typed: the
    Check card's gate). ``names`` as :func:`reading_rule` takes them. The
    pictures' pane and record, and the page copy of the sheet
    (``sheet.labelling_rule``), all read the rule here.
    """
    try:
        typed = sheet.typed_rule(str(real))
    except (OSError, ValueError):
        typed = None
    if typed is not None:
        return typed["text"], sheet.typed_rule_scope(typed["typed_at"]), "typed"
    if reading_rule(names) is not None:
        text = app_rule_of(real)
        return text, app_rule_scope(text), "owner"
    return None, None, None


def app_rule_scope(text: Any) -> str:
    """The scope said with one of the app's rules, chosen by its words (round
    r5-l): :data:`KELP_RULE_SCOPE` with :data:`KELP_READING_RULE`, which a
    record made before the round holds; :data:`RULE_SCOPE` with any other."""
    return KELP_RULE_SCOPE if text == KELP_READING_RULE else RULE_SCOPE


def _recorded_app_rule(real: Path) -> str | None:
    """The last of the app's rules (:data:`APP_RULES`) the sheet's pictures'
    record printed (its context lines in order, :func:`_header`), or ``None``
    when it printed none: the rule beside the sheet's latest pictures. A sheet
    that holds answers prints its own rule again at each open, so its last
    line stays its own; a fresh sheet planned again under an earlier sheet's
    name appends its rule after the earlier sheet's lines. Never raises."""
    found: str | None = None
    for entry in _read_record(record_path(real)):
        context = entry.get("context")
        text = context.get("reading_rule") if isinstance(context, dict) else None
        if isinstance(text, str) and text in APP_RULES:
            found = text
    return found


def _answered(real: Path) -> bool:
    """Whether a resolved sheet holds an answer (``sheet._holds_answers``),
    read without the sheet's lock, parsed as ``sheet.graded_app_rule`` parses
    it (a save replaces a sheet whole); a sheet that cannot be read or parsed
    counts as holding one. Never raises."""
    try:
        found = sheet.resolve_sheet(str(real))
        header, body = sheet._parse(found, found.read_bytes())
        return sheet._holds_answers(header, body)
    except Exception:  # noqa: BLE001 - unread, the pictures' record is kept
        return True


def app_rule_of(real: Path) -> str:
    """The app's rule for a resolved sheet whose rule is the app's (a map of
    presence and absence, no rule typed), as what its records hold says
    (round r5-l: records already written keep their words, and none is
    written again): :data:`KELP_READING_RULE` for a sheet made with that
    earlier rule, :data:`R5L_READING_RULE` for one made with r5-l's words
    (round r5-m B1: any rule of :data:`APP_RULES` its records hold), else
    :data:`READING_RULE`.

    In this order: while the sheet holds an answer (:func:`_answered`), the
    last of the app's rules its pictures' record printed
    (:func:`_recorded_app_rule`; beside a blank sheet the record can be an
    earlier sheet's of the same name); the one its graded mark kept
    (``sheet.graded_app_rule``: the rule the sheet took at its first grading,
    the draw's among them, or the kelp rule for a mark written before the
    round); the one the Check card's draw of the sheet was drawn with
    (``check.sheet_app_rule``: the rule the draw pinned, or the kelp rule for
    a draw recorded before the round). A sheet none of these speaks for (a
    sheet planned in the chat and never pictured or graded, one planned
    since, or a blank one no draw or mark from before the round speaks for)
    takes :data:`READING_RULE`. Never raises.
    """
    # check imports more than this module needs; imported here, when asked.
    from olmoearth_agent.mcp_app import check

    recorded = _recorded_app_rule(real)
    if recorded is not None and _answered(real):
        return recorded
    try:
        graded = sheet.graded_app_rule(str(real))
    except Exception:  # noqa: BLE001 - the rule stands without the mark's word
        graded = None
    if isinstance(graded, str) and graded in APP_RULES:
        return graded
    try:
        drawn = check.sheet_app_rule(str(real))
    except Exception:  # noqa: BLE001 - the rule stands without the draw's word
        logger.warning("a sheet's draw record was not read", exc_info=True)
        drawn = None
    if isinstance(drawn, str) and drawn in APP_RULES:
        return drawn
    return READING_RULE


# ------------------------------------------------------------ the map's grid


@dataclass(frozen=True)
class _Map:
    """What the pictures need of a map: its grid, raster header and period."""

    scores_name: str
    raster_sha256: str | None
    crs: str
    transform: tuple[float, float, float, float, float, float]
    grid: tuple[int, int]
    patch: int
    start: datetime
    end: datetime
    extent: tuple[float, float, float, float]


_maps: dict[tuple[str, int, int], _Map] = {}
_maps_lock = threading.Lock()


def _inside_roots(real: str) -> bool:
    return any(
        os.path.commonpath([root, real]) == root for root in review_set.read_roots()
    )


def _load_map(path: Any) -> _Map:
    """A scores file's map, cached by the file's identity; :class:`_Refused`."""
    try:
        real = os.path.realpath(str(path))
        st = os.stat(real)
    except (OSError, ValueError) as exc:
        raise _Refused(NO_FOOTPRINT) from exc
    key = (real, st.st_mtime_ns, st.st_size)
    with _maps_lock:
        if key in _maps:
            return _maps[key]
    try:
        loaded = review_set.load_scores_file(real)
        meta = loaded.meta
        grid = loaded.grid or tuple(int(v) for v in meta["grid"])
        patch = int(meta["patch_px"])
        raster = meta.get("raster") or {}
        raster_real = os.path.realpath(str(raster["path"]))
    except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
        raise _Refused(NO_FOOTPRINT) from exc
    if (
        not raster_real.lower().endswith((".tif", ".tiff"))
        or not _inside_roots(raster_real)
        or not os.path.isfile(raster_real)
        or patch < 1
    ):
        raise _Refused(NO_FOOTPRINT)
    window = meta.get("date_window")
    if not isinstance(window, dict) or not window.get("start") or not window.get("end"):
        raise _Refused(NO_PERIOD)
    try:
        start = s2_fetch.parse_moment(window["start"], end=False)
        end = s2_fetch.parse_moment(window["end"], end=True)
    except (TypeError, ValueError) as exc:
        raise _Refused(NO_PERIOD) from exc
    rasterio = importlib.import_module("rasterio")
    warp = importlib.import_module("rasterio.warp")
    try:
        with rasterio.open(raster_real) as src:  # the header only, on disk
            crs, transform = src.crs, src.transform
            height, width = src.height, src.width
            bounds = src.bounds
    except Exception as exc:  # noqa: BLE001 - any unreadable raster is the same refusal
        raise _Refused(NO_FOOTPRINT) from exc
    shape = raster.get("shape")
    if (
        crs is None
        or grid[0] * patch > height
        or grid[1] * patch > width
        or (isinstance(shape, list) and shape[-2:] != [height, width])
    ):
        raise _Refused(NO_FOOTPRINT)
    try:
        extent = s2_fetch.rounded(
            warp.transform_bounds(crs, "EPSG:4326", *bounds, densify_pts=21)
        )
    except ValueError as exc:  # across the antimeridian
        raise _Refused(NO_FOOTPRINT) from exc
    found = _Map(
        scores_name=os.path.basename(real),
        raster_sha256=(
            raster.get("sha256") if isinstance(raster.get("sha256"), str) else None
        ),
        crs=crs.to_wkt(),
        transform=tuple(transform)[:6],
        grid=(int(grid[0]), int(grid[1])),
        patch=patch,
        start=start,
        end=end,
        extent=extent,
    )
    with _maps_lock:
        _maps[key] = found
    return found


#: The EPSG codes of WGS 84's UTM zones, north and south, whose unit is the
#: metre: the only grids on which a square's side is given in metres.
_UTM_ZONES = (range(32601, 32661), range(32701, 32761))


def window_size(scores_path: Any) -> dict[str, Any] | None:
    """A scores file's window, as the Check card says its size: ``{"pixels",
    "metres"}``, or ``None``.

    ``pixels``: the file's ``patch_px`` (a window is that many map pixels a
    side). ``metres``: the raster's pixel width (``|a|`` of its transform)
    times ``patch_px``, only when the raster the file names lies inside the
    read roots and is on a UTM zone's grid (EPSG 32601-32660, 32701-32760),
    whose unit is the metre; else ``None``. ``None`` for a file outside the
    read roots, one that cannot be read, or one with no whole ``patch_px`` of
    1 or more. Reads the file and the raster's header only; never raises.
    """
    try:
        real = os.path.realpath(str(scores_path))
        if not _inside_roots(real):
            return None
        meta = review_set.load_scores_file(real).meta
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        return None
    patch = meta.get("patch_px")
    if isinstance(patch, float) and patch.is_integer():
        patch = int(patch)
    if not isinstance(patch, int) or isinstance(patch, bool) or patch < 1:
        return None
    return {"pixels": patch, "metres": _side_metres(meta.get("raster"), patch)}


def _side_metres(raster: Any, patch: int) -> float | None:
    """A window's side in metres on a UTM grid (:func:`window_size`), or ``None``."""
    try:
        path = os.path.realpath(str(raster["path"]))
        if (
            not path.lower().endswith((".tif", ".tiff"))
            or not _inside_roots(path)
            or not os.path.isfile(path)
        ):
            return None
        rasterio = importlib.import_module("rasterio")
        with rasterio.open(path) as src:  # the header only, on disk
            epsg = src.crs.to_epsg() if src.crs is not None else None
            width = abs(float(src.transform.a))
    except Exception:  # noqa: BLE001 - a raster that cannot be read gives no metres
        return None
    if not isinstance(epsg, int) or not any(epsg in zone for zone in _UTM_ZONES):
        return None
    if not math.isfinite(width) or width <= 0:
        return None
    return round(width * patch, 3)


def square_size_text(size: Any) -> str:
    """:data:`SQUARE_SIZE` for a :func:`window_size` answer, said in front of
    a sheet's :data:`CAPTION`; ``""`` when the size is not known.

    The metres (:data:`SQUARE_METRES`) are rounded to whole metres, half up,
    with thousands commas, as the Check card writes them; left out when
    :func:`window_size` gives none (a grid off UTM) or none above 0.
    """
    if not isinstance(size, dict):
        return ""
    pixels = size.get("pixels")
    if not isinstance(pixels, int) or isinstance(pixels, bool) or pixels < 1:
        return ""
    metres = size.get("metres")
    said = ""
    if (
        isinstance(metres, (int, float))
        and not isinstance(metres, bool)
        and math.isfinite(metres)
        and metres > 0
    ):
        said = SQUARE_METRES.format(metres=f"{math.floor(metres + 0.5):,}")
    return SQUARE_SIZE.format(pixels=pixels, metres=said)


def _map_of(design: dict[str, Any] | None) -> _Map:
    """The design's map (a two-map design: map a, both periods equal), from
    the scores files ``sheet.picture_sources`` reads off the design, the
    structure the page copy's ``pictures_possible`` is read from."""
    paths = sheet.picture_sources(design)
    if len(paths) == 1:
        return _load_map(paths[0])
    if len(paths) == 2:
        first, second = _load_map(paths[0]), _load_map(paths[1])
        if (first.start, first.end) != (second.start, second.end):
            raise _Refused(TWO_PERIODS)
        return first
    raise _Refused(NO_FOOTPRINT)


def _window_box(
    found: _Map, row: dict[str, Any]
) -> tuple[s2_fetch.Box, tuple[int, int, int, int], list[float]]:
    """The picture around the row's window on the map's grid, the outlined
    square in it, and how far the row's centre is from the window's (pixels).
    :class:`_Refused` when the row and the raster disagree."""
    affine = importlib.import_module("affine")
    warp = importlib.import_module("rasterio.warp")
    p, cols = found.patch, found.grid[1]
    try:
        index = int(str(row["window_index"]).strip())
        r, c = divmod(index, cols)
        if str(row.get("row", "")).strip() and int(str(row["row"]).strip()) != r:
            raise _Refused(MISMATCH)
        if str(row.get("col", "")).strip() and int(str(row["col"]).strip()) != c:
            raise _Refused(MISMATCH)
        lon, lat = float(row["lon"]), float(row["lat"])
    except (KeyError, TypeError, ValueError) as exc:
        raise _Refused(MISMATCH) from exc
    if r >= found.grid[0]:
        raise _Refused(MISMATCH)
    t = affine.Affine(*found.transform)
    (x,), (y,) = warp.transform("EPSG:4326", found.crs, [lon], [lat])
    px, py = ~t * (x, y)
    dx, dy = px - (c * p + p / 2), py - (r * p + p / 2)
    if abs(dx) > 0.5 or abs(dy) > 0.5:
        raise _Refused(MISMATCH)
    half = (_CONTEXT // 2) * p
    origin = t * affine.Affine.translation(c * p - half, r * p - half)
    side = _CONTEXT * p
    box = s2_fetch.Box(found.crs, tuple(origin)[:6], side, side)
    return box, (half, half, p, p), [round(dx, 3), round(dy, 3)]


# ------------------------------------------------------------------- record


def record_path(real: Path) -> Path:
    """The record beside the sheet, or beside a review set's record (JSON
    lines; not ``.json``, so no tool that reads a model-chosen ``.json`` can
    take it): ``a.csv`` and ``<key>.json`` give ``a.imagery.jsonl`` and
    ``<key>.imagery.jsonl``."""
    return real.with_name(real.stem + ".imagery.jsonl")


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_record(path: Path) -> list[dict[str, Any]]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    out = []
    for line in text.splitlines():
        with contextlib.suppress(ValueError):
            item = json.loads(line)
            if isinstance(item, dict):
                out.append(item)
    return out


def _append(path: Path, entries: list[dict[str, Any]]) -> None:
    """Append lines to the record, under a file lock (other processes)."""
    text = "".join(json.dumps(e, sort_keys=True) + "\n" for e in entries)
    with open(path, "a", encoding="utf-8") as fh:
        if fcntl is not None:
            fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        finally:
            if fcntl is not None:
                fcntl.flock(fh, fcntl.LOCK_UN)


def _header(
    real: Path,
    rule: str | None,
    scope: str | None = None,
    source: str | None = None,
) -> dict[str, Any]:
    """A sheet's record's context line: the rule printed (``rule``), what it
    says of it (``scope``: the app's rule's own, :func:`app_rule_scope`, when
    not given) and whose it is (``rule_source``: ``"owner"``, ``"typed"``, or
    ``None`` with no rule)."""
    if rule and scope is None and source in (None, "owner"):
        scope, source = app_rule_scope(rule), "owner"
    return {
        "context": {
            "record": "olmoearth-agent label-sheet pictures",
            "format": 1,
            "sheet": real.name,
            "reading_rule": rule,
            "rule_scope": scope if rule else None,
            "rule_source": source if rule else None,
            "label_is": LABEL_IS,
            "pick_rule": PICK_RULE,
            "stretch": STRETCH,
            "source": {
                "search": s2_fetch.SEARCH_URL,
                "collection": s2_fetch.COLLECTION,
            },
            "consent_version": consent_version(),
        }
    }


def _review_header(key: str) -> dict[str, Any]:
    """A review set's record's context line: no labelling rule and no label
    (nothing is labelled from a review set's pictures)."""
    return {
        "context": {
            "record": "olmoearth-agent review-set pictures",
            "format": 1,
            "review": key,
            "pick_rule": PICK_RULE,
            "stretch": STRETCH,
            "source": {
                "search": s2_fetch.SEARCH_URL,
                "collection": s2_fetch.COLLECTION,
            },
            "consent_version": consent_version(),
        }
    }


def served_any(path: Any) -> bool:
    """Whether pictures were served for a sheet (its record holds a serving)."""
    try:
        real = sheet.resolve_sheet(str(path))
    except (OSError, ValueError):
        return False
    return any("served" in e for e in _read_record(record_path(real)))


_sheet_locks: dict[str, threading.Lock] = {}
_sheet_locks_guard = threading.Lock()


def _lock_for(real: Path) -> threading.Lock:
    with _sheet_locks_guard:
        return _sheet_locks.setdefault(str(real), threading.Lock())


def _scenes(
    real: Path | None,
    found_extent: tuple[float, float, float, float],
    months: list[tuple[str, datetime, datetime]],
    client: Any,
    budget: s2_fetch.Budget,
) -> tuple[list[list[s2_fetch.Scene]], list[dict[str, Any]]]:
    """Each month's scenes: from the record when this extent and period were
    searched before (the catalogue hears of a sheet once), else searched now.
    Returns the scenes and the record lines to append (a new search)."""
    key = {
        "extent": list(found_extent),
        "period": [s2_fetch.iso(months[0][1]), s2_fetch.iso(months[-1][2])],
        "collection": s2_fetch.COLLECTION,
    }
    if real is not None:
        for entry in reversed(_read_record(record_path(real))):
            search = entry.get("search")
            if not isinstance(search, dict) or search.get("key") != key:
                continue
            got: list[list[s2_fetch.Scene]] = []
            for month in search.get("months") or []:
                kept = [
                    s2_fetch.Scene.from_record(s) for s in month.get("scenes") or []
                ]
                got.append([s for s in kept if s is not None])
            if len(got) == len(months):
                return got, []
    found: list[list[s2_fetch.Scene]] = []
    asked: list[dict[str, Any]] = []
    for label, lo, hi in months:
        scenes, query = s2_fetch.search_month(client, budget, found_extent, lo, hi)
        found.append(scenes)
        asked.append(
            {"month": label, "query": query, "scenes": [s.as_record() for s in scenes]}
        )
    return found, [{"search": {"key": key, "at": _now(), "months": asked}}]


# ---------------------------------------------------------------- rendering


def _date_label(scene: s2_fetch.Scene) -> str:
    when = s2_fetch.parse_moment(scene.datetime, end=False)
    cloud = "under 1%" if scene.cloud < 1 else f"{round(scene.cloud)}%"
    return f"{when.day} {when.strftime('%b %Y')}, scene cloud {cloud}"


def _pictures(stack: Any, square: tuple[int, int, int, int]) -> list[dict[str, Any]]:
    """The four pictures of one date: two renderings, context and detail."""
    p = square[2]
    lo, hi = (_CONTEXT - _DETAIL) // 2 * p, (_CONTEXT + _DETAIL) // 2 * p
    out = []
    for kind, spec in (
        ("nir", STRETCH["nir_false_colour"]),
        ("true", STRETCH["true_colour"]),
    ):
        rgb = s2_fetch.to_rgb(stack, tuple(spec["bands"]), spec["range"][1])
        views = (("context", rgb, square), ("detail", rgb[lo:hi, lo:hi], (p, p, p, p)))
        for view, crop, (bx, by, bw, bh) in views:
            side = int(crop.shape[0])
            scale = max(1, _SCREEN // side)
            cells = [tuple(px) for px in crop.reshape(-1, 3).tolist()]
            data = png(cells, side, side, 1)
            out.append(
                {
                    "kind": kind,
                    "view": view,
                    "label": f"{_NAMES[kind]}, {view}",
                    "src": "data:image/png;base64,"
                    + base64.b64encode(data).decode("ascii"),
                    "native": side,
                    "scale": scale,
                    "width": side * scale,
                    "height": side * scale,
                    "box": {"x": bx, "y": by, "w": bw, "h": bh},
                }
            )
    return out


def _dates(
    months: list[tuple[str, datetime, datetime]],
    results: list[tuple[s2_fetch.Scene | None, Any, int]],
    square: tuple[int, int, int, int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """The pane's dates and the record's months."""
    dates: list[dict[str, Any]] = []
    served: list[dict[str, Any]] = []
    for (label, _, _), (scene, stack, tried) in zip(months, results):
        if scene is None:
            dates.append(
                {
                    "month": label,
                    "label": f"{label}: no scene with data here",
                    "pictures": [],
                }
            )
            served.append({"month": label, "scene": None, "tried": tried})
            continue
        dates.append(
            {
                "month": label,
                "label": _date_label(scene),
                "pictures": _pictures(stack, square),
            }
        )
        served.append({"month": label, "tried": tried, **scene.as_record()})
    return dates, served


def _extra_missing() -> bool:
    try:
        for name in ("numpy", "rasterio", "rasterio.warp", "affine"):
            importlib.import_module(name)
    except ImportError:
        return True
    return False


# --------------------------------------------------------------------- tools


def window_image(
    path: str,
    handle: Any,
    *,
    sheet_id: Any,
    on: bool,
    still_on: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """The pictures of one window of a sheet, for the panel; never raises.

    The window is named by its handle on the page's copy of the sheet
    (``sheet_id``, ``sheet.read_sheet``), never by its index or place. Off
    (``on`` false): the question to ask (:data:`CONSENT_ASK`), and the test
    scene's (:data:`TEST_ASK`). On: the window, checked against the sheet
    (:data:`CHANGED`, :data:`NOT_ON_SHEET`) and the map raster before any
    fetch, with the labelling rule and its scope (:func:`rule_for`; nothing
    after the scope since round r5-l), what a label is, and the dated pictures
    under :data:`CAPTION`, led by how big the outlined square is
    (:func:`square_size_text`, left out when its scores file gives no size);
    the answer names the window by its handle. A failure is ``{"ok": false,
    "error": <one of the fixed sentences>}``, with ``again`` (true) when
    another try can get past it (:data:`RETRY_SENTENCES`, r5-a L1). When no
    month of the period has a scene with data on the square, the answer
    carries ``no_image`` (true) beside its empty months (r5-a L2).
    ``still_on`` (the server passes :func:`is_on`) is asked again after any
    wait: pictures switched off meanwhile send nothing.
    """
    if not on:
        return _ask_answer()
    try:
        return _window_image(path, handle, sheet_id, still_on)
    except (
        Exception
    ):  # noqa: BLE001 - the panel gets a fixed sentence; the log the rest
        logger.exception("window pictures failed")
        return {"ok": False, "error": UNEXPECTED}


def _window_image(
    path: str, handle: Any, sheet_id: Any, still_on: Callable[[], bool] | None
) -> dict[str, Any]:
    """:func:`window_image` with pictures on: the window found by its handle."""
    try:
        real, row, names = sheet.window_row(path, sheet_id, handle)
    except ValueError as exc:
        said = str(exc)
        known = said in (CHANGED, NOT_ON_SHEET)
        return {"ok": False, "error": said if known else NOT_READABLE}
    except OSError:
        return {"ok": False, "error": NOT_READABLE}
    index = int(str(row["window_index"]).strip())
    if _extra_missing():
        return {"ok": False, "error": NEEDS_EXTRA}
    rule, scope, source = rule_for(real, names)
    head = _header(real, rule, scope, source)
    _, design = sheet.design_of(real)
    try:
        found = _map_of(design)
        box, square, offset = _window_box(found, row)
        months = s2_fetch.months(found.start, found.end)
    except _Refused as exc:
        return {"ok": False, "error": str(exc)}
    except ValueError:
        return {"ok": False, "error": NO_PERIOD}
    if len(months) > MAX_MONTHS:
        return {"ok": False, "error": LONG_PERIOD}
    if not _calls.acquire(timeout=WAIT_SECONDS):  # two picture calls at a time
        return _failed(TOO_SLOW)
    client = s2_fetch.client_factory()
    try:
        lock = _lock_for(real)
        if not lock.acquire(timeout=WAIT_SECONDS):  # one search per sheet
            return _failed(TOO_SLOW)
        try:
            if still_on is not None and not still_on():  # switched off meanwhile
                return _ask_answer()
            # The search and the reads have their own deadlines: time spent
            # waiting for another window's search is not this window's.
            scenes, searched = _scenes(
                real, found.extent, months, client, s2_fetch.Budget()
            )
            if searched:
                _write(real, rule, searched, head)
        finally:
            lock.release()
        if still_on is not None and not still_on():
            return _ask_answer()
        fetch = s2_fetch.Fetch(client, s2_fetch.Budget())
        results = s2_fetch.dated(fetch, scenes, box, square)
    except s2_fetch.FetchError as exc:
        return _failed(_FETCH_SENTENCE.get(exc.kind, FETCH_FAILED))
    except OSError:
        return {"ok": False, "error": NO_RECORD}
    finally:
        client.close()
        _calls.release()
    dates, served = _dates(months, results, square)
    try:
        _write(
            real,
            rule,
            [
                {
                    "served": {
                        "window_index": index,
                        "at": _now(),
                        "months": served,
                        "picture": {"side": box.width, "square": list(square)},
                        "centre_offset_px": offset,
                        "stretch": STRETCH["version"],
                        "scores": found.scores_name,
                        "raster_sha256": found.raster_sha256,
                    }
                }
            ],
            head,
        )
    except OSError:
        return {"ok": False, "error": NO_RECORD}
    # How big the outlined square is, from the scores file that placed it (the
    # first of the design's, as _map_of took it), in front of the caption.
    placed_by = sheet.picture_sources(design)
    size = window_size(placed_by[0]) if placed_by else None
    answer: dict[str, Any] = {
        "ok": True,
        "handle": handle,
        "synthetic": False,
        "reading_rule": rule,
        "rule_scope": scope,
        "rule_source": source,
        "label_is": LABEL_IS,
        "caption": square_size_text(size) + CAPTION,
        "dates": dates,
        "pictures": [],
    }
    # No month has a scene with data on this square (r5-a L2): the pane says
    # one fixed line in place of the empty months. The key is left out
    # otherwise, so an answer with a picture is as it was.
    if _no_image(results):
        answer["no_image"] = True
    return answer


def _write(
    real: Path,
    rule: str | None,
    entries: list[dict[str, Any]],
    head: dict[str, Any] | None = None,
) -> None:
    """Append to the record, after a context line when the record is new or
    its rule, stretch or question changed. ``head``: the context line (a
    sheet's :func:`_header` when not given)."""
    path = record_path(real)
    head = head if head is not None else _header(real, rule)
    last = next((e for e in reversed(_read_record(path)) if "context" in e), None)
    _append(path, ([head] if last != json.loads(json.dumps(head)) else []) + entries)


#: The sheet's refusals that a review set says in its own words.
_REVIEW_SENTENCE = {NO_FOOTPRINT: REVIEW_NO_FOOTPRINT, MISMATCH: REVIEW_MISMATCH}


def _review_window_row(found: _Map, scores_real: str, index: int) -> dict[str, Any]:
    """A review set's listed window as the row :func:`_window_box` takes: its
    index, row and column on the map's grid, and the centre its scores file
    records for it (``centres_lon_lat``, one per row: the window's place in
    ``windows``, or its index when the file has none), read as the street
    map and the CSV read it (``review.window_centre``, r4-k K7).
    :class:`_Refused` (:data:`REVIEW_MISMATCH`) when the file holds no
    centre for it."""
    try:
        lon, lat = review.window_centre(scores_real, index)
    except ValueError as exc:
        raise _Refused(REVIEW_MISMATCH) from exc
    r, c = divmod(index, found.grid[1])
    return {"window_index": index, "row": r, "col": c, "lon": lon, "lat": lat}


def review_image(
    key: Any,
    rank: Any,
    *,
    on: bool,
    still_on: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """The pictures of one listed window of a review set, for the panel; never
    raises.

    The window is named by the review's key and its rank
    (:func:`review.review_row` checks both, and that the scores file is the
    one the key was made from, before anything else: a refusal is one of
    its fixed sentences). Off (``on`` false): the question to ask, with the
    review set's test-scene question (:data:`REVIEW_TEST_ASK`). On: the map
    path, the label sheet's (the scores file's raster, window size and
    period, the window's centre checked against the raster before any
    fetch), then the dated pictures, one search per review set and its
    record beside the review's (``review_sets/<key>.imagery.jsonl``). No
    labelling rule and no "what a label is": nothing is labelled here.
    ``again`` on a failure another try can get past and ``no_image`` when no
    month has a scene with data on the square, as for a sheet (r5-a L1, L2).
    ``still_on`` is asked again after any wait, as for a sheet.
    """
    try:
        return _review_image(key, rank, on, still_on)
    except (
        Exception
    ):  # noqa: BLE001 - the panel gets a fixed sentence; the log the rest
        logger.exception("review set pictures failed")
        return {"ok": False, "error": UNEXPECTED}


def _review_image(
    key: Any, rank: Any, on: bool, still_on: Callable[[], bool] | None
) -> dict[str, Any]:
    """:func:`review_image`: the window found by the review's key and rank."""
    try:
        real, scores_real, index = review.review_row(key, rank)
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}
    if not on:
        return _ask_answer(REVIEW_TEST_ASK)
    if _extra_missing():
        return {"ok": False, "error": NEEDS_EXTRA}
    try:
        found = _load_map(scores_real)
        row = _review_window_row(found, scores_real, index)
        box, square, offset = _window_box(found, row)
        months = s2_fetch.months(found.start, found.end)
    except _Refused as exc:
        said = str(exc)
        return {"ok": False, "error": _REVIEW_SENTENCE.get(said, said)}
    except ValueError:
        return {"ok": False, "error": NO_PERIOD}
    if len(months) > MAX_MONTHS:
        return {"ok": False, "error": LONG_PERIOD}
    head = _review_header(str(key))
    if not _calls.acquire(timeout=WAIT_SECONDS):  # two picture calls at a time
        return _failed(TOO_SLOW)
    client = s2_fetch.client_factory()
    try:
        lock = _lock_for(real)
        if not lock.acquire(timeout=WAIT_SECONDS):  # one search per review set
            return _failed(TOO_SLOW)
        try:
            if still_on is not None and not still_on():  # switched off meanwhile
                return _ask_answer(REVIEW_TEST_ASK)
            scenes, searched = _scenes(
                real, found.extent, months, client, s2_fetch.Budget()
            )
            if searched:
                _write(real, None, searched, head)
        finally:
            lock.release()
        if still_on is not None and not still_on():
            return _ask_answer(REVIEW_TEST_ASK)
        fetch = s2_fetch.Fetch(client, s2_fetch.Budget())
        results = s2_fetch.dated(fetch, scenes, box, square)
    except s2_fetch.FetchError as exc:
        return _failed(_REVIEW_FETCH_SENTENCE.get(exc.kind, REVIEW_FETCH_FAILED))
    except OSError:
        return {"ok": False, "error": REVIEW_NO_RECORD}
    finally:
        client.close()
        _calls.release()
    dates, served = _dates(months, results, square)
    try:
        _write(
            real,
            None,
            [
                {
                    "served": {
                        "review": key,
                        "rank": rank,
                        "window_index": index,
                        "at": _now(),
                        "months": served,
                        "picture": {"side": box.width, "square": list(square)},
                        "centre_offset_px": offset,
                        "stretch": STRETCH["version"],
                        "scores": found.scores_name,
                        "raster_sha256": found.raster_sha256,
                    }
                }
            ],
            head,
        )
    except OSError:
        return {"ok": False, "error": REVIEW_NO_RECORD}
    answer: dict[str, Any] = {
        "ok": True,
        "rank": rank,
        "synthetic": False,
        "reading_rule": None,
        "rule_scope": None,
        "label_is": None,
        "caption": CAPTION,
        "dates": dates,
        "pictures": [],
    }
    if _no_image(results):  # as a sheet's square (r5-a L2)
        answer["no_image"] = True
    return answer


_test_cache: dict[str, Any] = {}


def public_test_scene(*, review: bool = False) -> dict[str, Any]:
    """The public test scene's pictures, for the panel; never raises. Pressed
    by the user on the button after the question's fold, which holds
    :data:`TEST_ASK` (round r5-k K7); nothing of a sheet is used or sent, and
    no record is written (there is no sheet). ``review``: pressed beside a
    review set's window (after the fold with :data:`REVIEW_TEST_ASK`): its caption
    is :data:`REVIEW_TEST_CAPTION` and a fetch that fails says
    :data:`REVIEW_FETCH_FAILED`, with no sheet or label in either."""
    try:
        return _public_test_scene(review=review is True)
    except (
        Exception
    ):  # noqa: BLE001 - the panel gets a fixed sentence; the log the rest
        logger.exception("test scene failed")
        return {"ok": False, "error": UNEXPECTED}


def _fixed_box(
    place: tuple[float, float], epsg: int, patch: int
) -> tuple[s2_fetch.Box, tuple[int, int, int, int]]:
    """The picture around a fixed place, as a sheet's square is pictured: a
    10 m grid of the UTM zone ``epsg`` snapped to whole 10 m, the context
    view :data:`_CONTEXT` squares a side with the square of ``patch``
    pixels outlined in its middle. The public test scene's and the example
    pictures' (:func:`example_scene`)."""
    warp = importlib.import_module("rasterio.warp")
    side = _CONTEXT * patch
    (x,), (y,) = warp.transform("EPSG:4326", f"EPSG:{epsg}", [place[0]], [place[1]])
    x0 = round(x / 10) * 10 - (side // 2) * 10
    y0 = round(y / 10) * 10 + (side // 2) * 10
    box = s2_fetch.Box(
        f"EPSG:{epsg}", (10.0, 0.0, float(x0), 0.0, -10.0, float(y0)), side, side
    )
    half = (_CONTEXT // 2) * patch
    return box, (half, half, patch, patch)


def _public_test_scene(*, review: bool = False) -> dict[str, Any]:
    """:func:`public_test_scene`: the test place's dates, its scenes searched
    once per process."""
    if _extra_missing():
        return {"ok": False, "error": NEEDS_EXTRA}
    box, square = _fixed_box(TEST_PLACE, TEST_EPSG, TEST_PATCH)
    months = s2_fetch.months(
        s2_fetch.parse_moment(TEST_PERIOD[0], end=False),
        s2_fetch.parse_moment(TEST_PERIOD[1], end=True),
    )
    if not _calls.acquire(timeout=WAIT_SECONDS):
        return _failed(TOO_SLOW)
    client = s2_fetch.client_factory()
    try:
        scenes = _test_cache.get("scenes")
        if scenes is None:
            scenes, _ = _scenes(
                None, s2_fetch.rounded(box.lonlat()), months, client, s2_fetch.Budget()
            )
            _test_cache["scenes"] = scenes
        results = s2_fetch.dated(
            s2_fetch.Fetch(client, s2_fetch.Budget()), scenes, box, square
        )
    except s2_fetch.FetchError as exc:
        if review:
            said = _REVIEW_FETCH_SENTENCE.get(exc.kind, REVIEW_FETCH_FAILED)
        else:
            said = _FETCH_SENTENCE.get(exc.kind, FETCH_FAILED)
        return _failed(said)
    finally:
        client.close()
        _calls.release()
    dates, _ = _dates(months, results, square)
    return {
        "ok": True,
        "test_scene": True,
        "synthetic": False,
        "reading_rule": None,
        "rule_scope": None,
        "label_is": None,
        "caption": REVIEW_TEST_CAPTION if review else TEST_CAPTION,
        "dates": dates,
        "pictures": [],
    }


# ---------------------------------------------------------- example pictures

#: A fixed place :func:`example_scene` cannot render: not a place on Earth,
#: not on a UTM zone's grid, or no whole square size.
# WORDING: r4k K9
EXAMPLE_REFUSED = (
    "An example needs a place on Earth, a UTM zone's grid and a whole square size."
)


def example_scene(
    place: tuple[float, float],
    *,
    epsg: int,
    period: tuple[str, str],
    patch: int = TEST_PATCH,
) -> dict[str, Any]:
    """The pictures of one fixed public place, rendered as a sheet's square
    is (r4-k K9): for a later cluster run that makes the two example
    pictures beside the app's rule (``views/examples``, which the
    owner picks and checks before any ships). Never called by the app;
    nothing of a sheet, a map or a review is read or sent, and no record is
    written. Never raises: a failure is ``{"ok": False, "error": <a fixed
    sentence>}`` (the log keeps its kind only, never the place).

    ``place`` is ``(lon, lat)``; ``epsg`` the UTM zone (EPSG 32601-32660 or
    32701-32760) whose 10 m grid the pictures are drawn on; ``period`` the
    first and last day (ISO dates, at most :data:`MAX_MONTHS` months);
    ``patch`` the outlined square's side in pixels (the kelp map's 11, as
    the test scene's). The same path as a sheet's square: one least-cloud
    scene per month (:data:`PICK_RULE`), near-infrared false colour and true
    colour, each a context view with the square outlined and a detail view
    (:func:`_fixed_box`, :func:`_pictures`), one fixed stretch
    (:data:`STRETCH`). The answer: ``dates`` as a sheet's pane has them,
    ``served`` (per month the scene picked and how many were tried, for the
    note the owner keeps beside the pictures), ``stretch``, ``pick_rule`` and
    ``picture`` (the context view's side and the outlined square).
    """
    try:
        return _example_scene(place, epsg, period, patch)
    except Exception as exc:  # noqa: BLE001 - a fixed sentence for the caller
        # The failure's kind only: never the place or a path.
        logger.error("example scene failed (%s)", type(exc).__name__)
        return {"ok": False, "error": UNEXPECTED}


def _example_scene(
    place: tuple[float, float], epsg: int, period: tuple[str, str], patch: int
) -> dict[str, Any]:
    """:func:`example_scene`, its place, grid and square size checked first,
    then its period (at most :data:`MAX_MONTHS` months)."""
    try:
        lon, lat = (float(v) for v in place)
    except (TypeError, ValueError):
        return {"ok": False, "error": EXAMPLE_REFUSED}
    on_earth = (
        math.isfinite(lon) and math.isfinite(lat) and abs(lon) <= 180 and abs(lat) <= 90
    )
    utm = (
        isinstance(epsg, int)
        and not isinstance(epsg, bool)
        and any(epsg in zone for zone in _UTM_ZONES)
    )
    whole = isinstance(patch, int) and not isinstance(patch, bool) and patch >= 1
    if not (on_earth and utm and whole):
        return {"ok": False, "error": EXAMPLE_REFUSED}
    if _extra_missing():
        return {"ok": False, "error": NEEDS_EXTRA}
    try:
        months = s2_fetch.months(
            s2_fetch.parse_moment(period[0], end=False),
            s2_fetch.parse_moment(period[1], end=True),
        )
    except (TypeError, ValueError, IndexError):
        return {"ok": False, "error": NO_PERIOD}
    if not months:
        return {"ok": False, "error": NO_PERIOD}
    if len(months) > MAX_MONTHS:
        return {"ok": False, "error": LONG_PERIOD}
    box, square = _fixed_box((lon, lat), epsg, patch)
    if not _calls.acquire(timeout=WAIT_SECONDS):
        return {"ok": False, "error": TOO_SLOW}
    client = s2_fetch.client_factory()
    try:
        scenes, _ = _scenes(
            None, s2_fetch.rounded(box.lonlat()), months, client, s2_fetch.Budget()
        )
        results = s2_fetch.dated(
            s2_fetch.Fetch(client, s2_fetch.Budget()), scenes, box, square
        )
    except s2_fetch.FetchError as exc:
        said = _REVIEW_FETCH_SENTENCE.get(exc.kind, REVIEW_FETCH_FAILED)
        return {"ok": False, "error": said}
    finally:
        client.close()
        _calls.release()
    dates, served = _dates(months, results, square)
    return {
        "ok": True,
        "example": True,
        "synthetic": False,
        "dates": dates,
        "served": served,
        "stretch": STRETCH["version"],
        "pick_rule": PICK_RULE,
        "picture": {"side": box.width, "square": list(square)},
    }
