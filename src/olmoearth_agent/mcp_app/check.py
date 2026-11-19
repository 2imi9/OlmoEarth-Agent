# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""The Check card's records: each zone's label counts, its one draw, the log.

A Studio result's files hold one map per UTM zone
(``olmoearth_scores_from_studio``: one scores file per zone). A check is one
zone of one result (:func:`check_key`). The card offers, per zone, the label
counts the package plans before any label (:func:`offers`: the plan tool's
own ``plan.plan_error_rate`` call and its own sentence), none picked; the
user's press draws that zone's sample once, through the plan tool, with the
design (:data:`DESIGN`), the seed (:data:`SEED`) and the count fixed and
logged at the press.

The records live under ``review_set.write_root()/checks/`` (the folder 0700,
every file 0600, each written whole: a new file beside it, moved over it):

- ``<key>.json``: the offer (:data:`CHECK_FORMAT`), written again at each
  read of the result (:func:`offer_zones`), with what the cards name the
  zone by: the map's name and period, the zone's part name
  (:func:`part_names`) and share, whether the map has no labelling rule of
  the owner's (``rule_needed``), and whether the owner's rule may be
  replaced by the user's own before the map's first part is drawn
  (``rule_owner``, a map of presence and absence; missing: no);
- ``<key>.draw.json``: the draw (:data:`DRAW_FORMAT`), its settings pinned at
  the first press (:func:`claim_draw`; the user's labelling rule among them
  for a map with none), then ``drawn`` (:func:`finish_draw`) or ``failed``
  (:func:`fail_draw`);
- ``<key>.lock``: held, with this process's lock, around each read and
  rewrite of the draw, so a second server process waits its turn;
- ``log.jsonl``: one line per zone of each press (:func:`append_log`),
  stamped with the press's time. A press the server refuses whole (no list
  of draws, or too many) draws nothing and writes no line, and a line that
  cannot be written does not stop the draw: the draw record, not the log,
  holds the settings pinned at the press;
- ``planned/<sha256>.json``: the label counts planned for a number of
  windows (:func:`_planned`, round r5-i B11: :data:`PLANNED_FORMAT`, the
  inputs and the answer's JSON text whole), read before planning, so a
  card opened again in a new server does not plan them again; one older
  than :data:`PLANNED_KEEP_DAYS` days is removed when another is written.

The planner runs outside the lock, with the draw in state ``drawing`` and
the process id and start time written in it; a draw left ``drawing`` for
:data:`STALE_SECONDS` is drawn again with its pinned settings. Only
:func:`offer_zones`'s answer and :func:`draw_view` are for the page: no
scores path, raster digest, window, place, map class or figure
(:func:`sheet_context`, :func:`map_rule` and :func:`result_settled` are the
server's: they read the records back to name a sheet's part and the map's
rule, and to say whether that rule is settled). The server
leaves a drawn plan's ``facts`` and ``forbidden_claims`` out of the card's
view data; the model's text carries them, and the host also hands that
text to the page: the plan's generic contract wording, with no window,
class or place.
"""

from __future__ import annotations

import contextlib
import csv
import functools
import hashlib
import json
import logging
import os
import re
import stat
import tempfile
import threading
import time
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

try:  # POSIX: a lock another process (a second server) also honours
    import fcntl
except ImportError:  # pragma: no cover - Windows has no fcntl
    fcntl = None  # type: ignore[assignment]

from olmoearth_agent.mcp_app import sheet
from olmoearth_agent.security.paths import safe_path
from olmoearth_agent.tools import estimation, inferencex
from olmoearth_agent.tools import statistical_rules as rules
from olmoearth_agent.tools.review_set import write_root

logger = logging.getLogger(__name__)

#: The format of a check's offer record (``<key>.json``).
CHECK_FORMAT = "olmoearth-agent/check@1"

#: The format of a check's draw record (``<key>.draw.json``).
DRAW_FORMAT = "olmoearth-agent/check-draw@1"

#: The folder under the write root that holds the check records.
CHECKS_DIR = "checks"

#: The log of presses, one JSON line each, in :data:`CHECKS_DIR`.
LOG_NAME = "log.jsonl"

#: The folder in :data:`CHECKS_DIR` that keeps the planned label counts
#: (:func:`_planned`, round r5-i B11): one file per inputs, named by their
#: sha256 (:func:`_planned_place`).
PLANNED_DIR = "planned"

#: The format of a kept planned answer (``planned/<sha256>.json``).
PLANNED_FORMAT = "olmoearth-agent/check-planned@1"

#: Days a kept planned answer stays in :data:`PLANNED_DIR` after it was last
#: written (:func:`_prune_planned`, round r5-i B11): its inputs hold a digest
#: of the app's modules, so each app update keeps new files and an older
#: release's would stay for ever; one of this release older than this is
#: planned again once.
PLANNED_KEEP_DAYS = 30

#: The interval widths (high minus low) the card's counts are planned for:
#: widths of 20%, 10% and 5%, each at the package's planning rate.
CHECK_WIDTHS = (0.20, 0.10, 0.05)

#: The design every check's sample is drawn with: a simple random sample,
#: the one the counts' sentence is about.
DESIGN = "random"

#: The seed every check's sample is drawn with (the plan tool's default).
SEED = 0

#: Seconds after which a draw still ``drawing`` is taken as stopped and is
#: drawn again with its pinned settings.
STALE_SECONDS = 600

#: The largest record (bytes) read back.
MAX_RECORD_BYTES = 1024 * 1024

#: The longest error kept in a draw record.
MAX_ERROR_CHARS = 4000

#: The most of a plan's ``forbidden_claims`` a draw record keeps.
MAX_CLAIMS = 32

#: The most of a plan's ``facts`` a draw record keeps.
MAX_FACTS = 32

#: A check's key: 24 lowercase hexadecimal characters (:func:`check_key`).
KEY_PATTERN = re.compile(r"[0-9a-f]{24}")

#: A zone's native file in a Studio result's files (``studio_scores``).
ZONE_FILE = re.compile(r"result_epsg(?P<epsg>[0-9]+)_(?P<i>[0-9]+)_native\.tif")

#: Why a key the page sent is refused, before any file is touched.
BAD_KEY = "check_key is not one of this card's checks"

#: Why a press on a check with no offer record is refused.
UNKNOWN_CHECK = "This check is not known here; ask Claude to open the Check card again."

#: Why a press with a count the card did not offer is refused. The card shows
#: it under a number it offered itself when it was drawn (the offer on the
#: server is newer), so the words say what to do and not "count" (round r5-b,
#: J25).
# WORDING: r5b J25
NOT_OFFERED = "This card is out of date. Ask Claude to open the Check card again."

#: Why no press is offered when the records cannot be written.
NOT_WRITABLE = (
    "The agent's output folder ({root}) cannot be written, so no sample can be "
    "drawn here."
)

#: Why a record that is not one of ours, or cannot be read, is refused.
UNREADABLE = "This check's record cannot be read; ask Claude about it."

#: Why a draw is finished or failed that no press started.
NOT_CLAIMED = "This zone's draw was not started here."

#: Why a press is refused once Studio's files under a drawn zone changed.
RASTER_CHANGED = (
    "Studio's files for this zone changed after its sample was drawn, so this "
    "card does not draw it again; ask Claude about it."
)

#: Why a press is refused whole when it picks a part of a map that has no
#: labelling rule and the press brings none (the card's gate): nothing is
#: drawn or logged.
# WORDING: squares
RULE_NEEDED = (
    "No labelling rule is set for this map: write in one sentence what counts "
    "and what does not before any square is picked."
)

#: Why a press is refused whole when a part's offer record was written
#: before the offers said whether the map has a labelling rule (no
#: ``rule_needed``): opening the card again writes it; nothing is drawn or
#: logged.
# WORDING: r4p2
OLDER_CARD = "This card is from an older version of the app. Open the Check card again."

#: Why a press is refused whole when it sends a labelling rule for a map
#: whose rule is settled (:func:`result_settled`, or a rule a part pinned):
#: the owner's ruling of 13 Nov 2026, option A; nothing is drawn or logged.
# WORDING: squares
RULE_SETTLED = "This map's labelling rule was set when its first part was picked."

#: The most records one scan reads (:func:`sheet_context`, :func:`map_rule`,
#: :func:`result_settled`).
SCAN_MAX = 2000

#: The error of a draw left ``drawing`` past :data:`STALE_SECONDS`.
STOPPED = "The draw stopped before it finished."

#: The error of a plan that answered with no label sheet.
NO_SHEET = "The plan wrote no label sheet."

#: Why a drawn zone's sheet cannot be read when ``csv`` refuses it (a cell
#: past its field limit, a quote never closed): the zone says so, the card
#: still shows every zone.
SHEET_NOT_CSV = (
    "{name} cannot be read as CSV ({error}); save it again as CSV UTF-8, "
    "comma-separated"
)

#: A zone offering no count was refused where its files were read: it is not
#: read as a map (``refused_at``, :func:`offer_zones`).
REFUSED_READ = "read"

#: A zone offering no count was refused by the planner: no count is planned
#: for it, or the package is missing (``refused_at``, :func:`offer_zones`).
REFUSED_PLAN = "plan"

#: The start of a plan error that the same settings give again: a refusal
#: (a sheet holding labels the plan would clear, a budget above the windows),
#: arguments the plan does not take, or a scores file gone. Any other error
#: (the disk, memory, a time limit) may pass on a second press.
FINAL_ERRORS = (
    "ValueError:",
    "PathTraversalError:",
    "FileNotFoundError:",
    "NotADirectoryError:",
    "IsADirectoryError:",
    "KeyError:",
    "TypeError:",
    "invalid arguments",
)

#: The keys of a draw the page gets (:func:`draw_view`). ``failed_at`` (a
#: failed draw's time, :func:`_failed`; ``None`` otherwise): a retry that
#: fails again waits behind the failed draws that waited longer (r4-l L2).
VIEW_KEYS = (
    "check_key",
    "state",
    "pressed_at",
    "budget",
    "design",
    "seed",
    "labels_csv_path",
    "must_state",
    "error",
    "retry",
    "failed_at",
)

#: Held while a draw is read and rewritten in this process; :func:`_draw_lock`
#: adds a file lock for other processes.
_DRAW_LOCK = threading.Lock()

#: Held while a line is added to the log in this process.
_LOG_LOCK = threading.Lock()


def _now() -> float:
    """The time now, in seconds since the epoch."""
    return time.time()


def _stamp(moment: float) -> str:
    """``moment`` as a UTC time to the second, ISO 8601 (``...Z``)."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(moment))


def stamp_now() -> str:
    """The time now as :func:`_stamp` writes it: a press's time, taken once
    before its claims, for its draw records and log lines (:func:`log_entry`)."""
    return _stamp(_now())


def check_key(result_id: str, zone_file: str) -> str:
    """The key of one zone of one result: 24 hex characters of a SHA-256.

    ``result_id`` is the result's id in the form the resolver gives it, and
    ``zone_file`` the zone's native file name in the result's files.
    """
    text = f"{str(result_id).strip()}\n{str(zone_file).strip()}"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]


def _valid_key(key: Any) -> str:
    """``key`` when it is a check key (:data:`KEY_PATTERN`); else ``ValueError``."""
    if not isinstance(key, str) or KEY_PATTERN.fullmatch(key) is None:
        raise ValueError(BAD_KEY)
    return key


def checks_dir() -> Path:
    """The folder of the check records, made (0700) when missing.

    ``ValueError`` (:data:`NOT_WRITABLE`) when it cannot be made or written,
    and a ``PathTraversalError`` when it resolves outside the write root.
    """
    root = write_root()
    folder = safe_path(CHECKS_DIR, root=root)
    try:
        folder.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(folder, 0o700)
    except OSError as exc:
        raise ValueError(NOT_WRITABLE.format(root=root)) from exc
    if not os.access(folder, os.W_OK | os.X_OK):
        raise ValueError(NOT_WRITABLE.format(root=root))
    return folder


def _read_json(path: Path) -> dict[str, Any] | None:
    """A record's contents; ``None`` when it does not exist.

    ``ValueError`` (:data:`UNREADABLE`) for one that is not a JSON object of
    at most :data:`MAX_RECORD_BYTES`. A link is not followed.
    """
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise ValueError(UNREADABLE) from exc
    try:
        with os.fdopen(fd, "rb") as fh:
            raw = fh.read(MAX_RECORD_BYTES + 1)
        if len(raw) > MAX_RECORD_BYTES:
            raise ValueError(UNREADABLE)
        data = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(UNREADABLE) from exc
    if not isinstance(data, dict):
        raise ValueError(UNREADABLE)
    return data


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Replace a record whole: a new 0600 file beside it, then moved over it."""
    fd, staged = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, sort_keys=True)
        os.chmod(staged, 0o600)
        os.replace(staged, path)
    except BaseException:
        Path(staged).unlink(missing_ok=True)
        raise


@contextlib.contextmanager
def _draw_lock(folder: Path, key: str) -> Iterator[None]:
    """Hold one check's draw: this process's lock and a file lock.

    The file lock is on ``<key>.lock`` beside the records, never on the draw
    itself, which each rewrite replaces with a new file.
    """
    with _DRAW_LOCK:
        if fcntl is None:  # pragma: no cover - Windows
            yield
            return
        flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(folder / f"{key}.lock", flags, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def zone_label(file: str, files: Sequence[str] = ()) -> str:
    """A zone's name from its file's EPSG code: ``"UTM zone 10N"``.

    32601 to 32660 are the northern UTM zones and 32701 to 32760 the
    southern ones; any other code is ``"EPSG <code>"``. When two of
    ``files`` share the code, the name ends ``" (file i+1)"``, ``i`` from the
    file's name. A name that is not a zone's native file is kept as it is.
    """
    match = ZONE_FILE.fullmatch(str(file))
    if match is None:
        return str(file)
    code = int(match["epsg"])
    if 32601 <= code <= 32660:
        label = f"UTM zone {code - 32600}N"
    elif 32701 <= code <= 32760:
        label = f"UTM zone {code - 32700}S"
    else:
        label = f"EPSG {code}"
    same = {
        name
        for name in map(str, files)
        if (other := ZONE_FILE.fullmatch(name)) is not None
        and int(other["epsg"]) == code
    }
    if len(same | {str(file)}) > 1:
        label += f" (file {int(match['i']) + 1})"
    return label


def _utm(file: str) -> tuple[int, str] | None:
    """A zone file's UTM zone number and hemisphere (``"N"`` or ``"S"``):
    32601 to 32660 are the northern zones, 32701 to 32760 the southern ones;
    ``None`` for any other code or a name that is not a zone's native file."""
    match = ZONE_FILE.fullmatch(str(file))
    if match is None:
        return None
    code = int(match["epsg"])
    if 32601 <= code <= 32660:
        return code - 32600, "N"
    if 32701 <= code <= 32760:
        return code - 32700, "S"
    return None


def part_names(files: Sequence[str]) -> list[str | None]:
    """What the cards call each zone of a result, in the files' order.

    One file is the whole map: ``[None]`` (no part is named, no parts
    figure). A compass word only where it is certain, for exactly two UTM
    files: the same zone number in both hemispheres, ``"North part"`` and
    ``"South part"``; the same hemisphere, zone numbers one apart and
    neither zone 1 nor 60 (the two ends of the numbering, which meet at the
    antimeridian), ``"West part"`` for the lower number and ``"East part"``.
    Anything else (more parts, a code that is not UTM, the same code twice,
    zones further apart or across the hemispheres) is ``"Part 1"`` ..
    ``"Part n"``: a wrong compass word is worse than none.
    """
    names = [str(f) for f in files]
    out: list[str | None]
    if len(names) < 2:
        out = [None for _ in names]
        return out
    if len(names) == 2:
        first, second = _utm(names[0]), _utm(names[1])
        if first is not None and second is not None and first != second:
            (one, side_one), (two, side_two) = first, second
            if one == two:
                # WORDING: squares
                out = [
                    "North part" if s == "N" else "South part"
                    for s in (side_one, side_two)
                ]
                return out
            if (
                side_one == side_two
                and abs(one - two) == 1
                and not {one, two} & {1, 60}
            ):
                lower = min(one, two)
                # WORDING: squares
                out = ["West part" if z == lower else "East part" for z in (one, two)]
                return out
    # WORDING: squares
    out = [f"Part {i}" for i in range(1, len(names) + 1)]
    return out


def _whole(value: Any) -> int | None:
    """``value`` as a whole number (an int, or a float with no fraction), or ``None``."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


#: The app's own modules that word :func:`_planned`'s answer beside the
#: package's numbers (the counts' sentences, the offer's order): a kept
#: answer names their digest, so a release that words it otherwise plans
#: again (round r5-i B11).
_PLANNED_SOURCES = (__file__, estimation.__file__, rules.__file__)

#: The keys of each count of a planned answer (:func:`_plan_counts`).
_OFFER_KEYS = frozenset({"labels", "width", "sentence", "rate_note"})

#: The name of a kept planned answer (:func:`_planned_place`): the sha256 of
#: its inputs; nothing else in :data:`PLANNED_DIR` is ever removed.
_PLANNED_NAME = re.compile(r"[0-9a-f]{64}\.json")


@functools.lru_cache(maxsize=1)
def _planned_words() -> str | None:
    """The sha256 of the app's modules that word a planned answer
    (:data:`_PLANNED_SOURCES`), read once per process; ``None`` when one
    cannot be read, and then nothing is kept on disk."""
    digest = hashlib.sha256()
    try:
        for source in _PLANNED_SOURCES:
            digest.update(Path(str(source)).read_bytes())
    except OSError:
        return None
    return digest.hexdigest()


def _planned_inputs(n_valid: int, widths: tuple[float, ...]) -> dict[str, Any] | None:
    """What a planned answer depends on, as its kept file names it (round
    r5-i B11): the windows (``n_valid``), the ``widths``, the
    olmoearth-inferencex version installed (``inferencex``) and the digest
    of the app's modules that word it (``words``, :func:`_planned_words`);
    ``None`` when the version or the digest is not known: nothing is kept."""
    version = inferencex.version()
    words = _planned_words()
    if version is None or words is None:
        return None
    return {
        "n_valid": n_valid,
        "widths": list(widths),
        "inferencex": version,
        "words": words,
    }


def _planned_place(
    n_valid: int, widths: tuple[float, ...]
) -> tuple[Path, dict[str, Any]] | None:
    """Where a planned answer is kept, with its inputs
    (:func:`_planned_inputs`): ``<checks>/planned/<sha256 of the
    inputs>.json``, the folder made (0700) when missing; ``None`` when
    nothing is kept (the inputs not known, or the folder cannot be made or
    is a link: logged by the error's kind)."""
    inputs = _planned_inputs(n_valid, widths)
    if inputs is None:
        return None
    text = json.dumps(inputs, sort_keys=True)
    name = hashlib.sha256(text.encode("utf-8")).hexdigest()
    try:
        folder = checks_dir() / PLANNED_DIR
        folder.mkdir(mode=0o700, exist_ok=True)
        if folder.is_symlink() or not folder.is_dir():
            raise ValueError(UNREADABLE)
        os.chmod(folder, 0o700)
    except (OSError, ValueError) as exc:
        logger.info("planned counts are not kept on disk (%s)", type(exc).__name__)
        return None
    return folder / f"{name}.json", inputs


def _planned_shape(text: str, inputs: dict[str, Any]) -> bool:
    """Whether ``text`` is an answer of the shape planning gives for
    ``inputs`` (:func:`_plan_counts`): ``{"offer", "refused"}``; each count
    :data:`_OFFER_KEYS`, its ``labels`` a whole number from 1 to the
    windows, ascending and each once, its ``width`` one of the widths, its
    ``sentence`` text and its ``rate_note`` text or ``None``; ``refused``
    ``None`` beside a count, else text or ``None``."""
    try:
        answer = json.loads(text)
    except (ValueError, RecursionError):
        return False
    if not isinstance(answer, dict) or set(answer) != {"offer", "refused"}:
        return False
    offer, refused = answer["offer"], answer["refused"]
    if not isinstance(offer, list) or (offer and refused is not None):
        return False
    if refused is not None and not isinstance(refused, str):
        return False
    counts: list[int] = []
    for item in offer:
        if not isinstance(item, dict) or set(item) != _OFFER_KEYS:
            return False
        labels, note = item["labels"], item["rate_note"]
        if isinstance(labels, bool) or not isinstance(labels, int):
            return False
        if not 1 <= labels <= inputs["n_valid"]:
            return False
        if item["width"] not in inputs["widths"]:
            return False
        if not isinstance(item["sentence"], str):
            return False
        if note is not None and not isinstance(note, str):
            return False
        counts.append(labels)
    return counts == sorted(set(counts))


def _kept_planned(path: Path, inputs: dict[str, Any]) -> str | None:
    """The JSON text of the planned answer kept at ``path``, when it is one
    for exactly ``inputs`` (:data:`PLANNED_FORMAT`) of the shape planning
    gives (:func:`_planned_shape`); ``None`` when there is none, or it does
    not parse, names other inputs or holds another shape (it is then planned
    again and replaced)."""
    try:
        record = _read_json(path)
    except ValueError:
        return None
    if record is None:
        return None
    if record.get("format") != PLANNED_FORMAT or record.get("inputs") != inputs:
        return None
    answer = record.get("answer")
    if not isinstance(answer, str) or not _planned_shape(answer, inputs):
        return None
    return answer


def _keep_planned(path: Path, inputs: dict[str, Any], text: str) -> None:
    """Keep a planned answer at ``path`` (:data:`PLANNED_FORMAT`, its inputs
    and its JSON text whole), written as every record is
    (:func:`_write_json`: a new 0600 file beside it, moved over it, so no
    reader sees half of it); one that cannot be written is logged by the
    error's kind, and the answer stands. Once it is written, the kept
    answers older than :data:`PLANNED_KEEP_DAYS` days are removed
    (:func:`_prune_planned`)."""
    record = {"format": PLANNED_FORMAT, "inputs": inputs, "answer": text}
    try:
        _write_json(path, record)
    except (OSError, ValueError) as exc:
        logger.info("planned counts were not kept on disk (%s)", type(exc).__name__)
        return
    _prune_planned(path)


def _prune_planned(keep: Path) -> None:
    """Remove the kept planned answers beside ``keep`` (the one just written)
    last written more than :data:`PLANNED_KEEP_DAYS` days ago (round r5-i
    B11): only regular files named as :func:`_planned_place` names them
    (:data:`_PLANNED_NAME`), never ``keep``, a link, a folder, a staged file
    or any other name, so two releases in use never remove each other's
    fresh answers. A file that cannot be read or removed is logged by the
    error's kind alone and stays; nothing here fails the answer."""
    limit = time.time() - PLANNED_KEEP_DAYS * 86400
    try:
        entries = list(keep.parent.iterdir())
    except OSError as exc:
        logger.info("old planned counts were not looked at (%s)", type(exc).__name__)
        return
    for entry in entries:
        if entry.name == keep.name or not _PLANNED_NAME.fullmatch(entry.name):
            continue
        try:
            info = entry.lstat()
            if not stat.S_ISREG(info.st_mode) or info.st_mtime >= limit:
                continue
            entry.unlink(missing_ok=True)
        except OSError as exc:
            logger.info("old planned counts were not removed (%s)", type(exc).__name__)


@functools.lru_cache(maxsize=128)
def _planned(n_valid: int, widths: tuple[float, ...]) -> str:
    """:func:`offers`'s answer as JSON text, cached per ``(n_valid, widths)``
    in this process and, round r5-i B11, on this computer.

    The package is loaded first, as before (``InferencexMissingError``
    without it). Then the answer kept under the checks folder for the same
    inputs (:func:`_planned_place`: the windows, the widths, the package's
    version and the app's wording) is read and given as kept, the same JSON
    text planning gave (:func:`_plan_counts`), so a card opened again in a
    new server does not plan its counts again (44 s of a 67 s reopen,
    measured on a two-part map). One that does not parse, names other
    inputs or holds another shape is planned again and replaced; with
    nothing kept (no version known, a folder that cannot be made) it plans
    as before.
    """
    plan_pkg = inferencex.load("plan")
    place = _planned_place(n_valid, widths)
    if place is not None:
        kept = _kept_planned(*place)
        if kept is not None:
            return kept
    text = _plan_counts(plan_pkg, n_valid, widths)
    if place is not None:
        _keep_planned(*place, text)
    return text


def _plan_counts(plan_pkg: Any, n_valid: int, widths: tuple[float, ...]) -> str:
    """The planned answer's JSON text (:func:`_planned`), as before round
    r5-i.

    Each width is planned as the plan tool plans it
    (``estimation._label_budget_advice``): ``plan.plan_error_rate`` at the
    package's planning rate, kept under ``estimation._ADVICE_KEYS``.
    """
    by_count: dict[int, dict[str, Any]] = {}
    refused: str | None = None
    for width in sorted(widths, reverse=True):
        try:
            planned = estimation._jsonable(
                plan_pkg.plan_error_rate(n_valid, width, None)
            )
        except ValueError as exc:
            planned = {"n_population": n_valid, "width": width, "refusal": str(exc)}
        advice = {k: planned[k] for k in estimation._ADVICE_KEYS if k in planned}
        labels = _whole(advice.get("labels"))
        if labels is None or not 1 <= labels <= n_valid:
            if refused is None:
                fact = estimation._label_budget_fact(
                    {**advice, "labels": None}, DESIGN, 0
                )
                refused = str(fact["sentence"])
            continue
        if labels in by_count and by_count[labels]["width"] <= width:
            continue
        fact = estimation._label_budget_fact(advice, DESIGN, labels)
        by_count[labels] = {
            "labels": labels,
            "width": width,
            "sentence": str(fact["sentence"]),
            "rate_note": advice.get("rate_note"),
        }
    offer = [by_count[count] for count in sorted(by_count)]
    return json.dumps({"offer": offer, "refused": None if offer else refused})


def offers(n_valid: int, widths: Sequence[float] = CHECK_WIDTHS) -> dict[str, Any]:
    """The label counts a card offers for a zone of ``n_valid`` windows.

    For each width (:data:`CHECK_WIDTHS`), the count ``plan.plan_error_rate``
    recommends for a 95% interval no wider than it, on a map assumed wrong at
    the package's planning rate, each with the plan tool's own sentence for
    it (``estimation._label_budget_fact`` under the random design, unchanged:
    "this plan draws L" is true of the chip's count L) and the advice's
    ``rate_note``, which the sentence points to. ``offer`` holds counts of
    at most ``n_valid``, each once (the narrowest width's sentence where two
    widths plan one count), in ascending order: ``{"labels", "width",
    "sentence", "rate_note"}``. When no count survives, ``offer`` is empty
    and ``refused`` is the planner's sentence saying why. Cached per
    ``(n_valid, widths)``, in this process and on disk (:func:`_planned`,
    round r5-i B11); ``InferencexMissingError`` without the package.
    """
    count = _whole(n_valid)
    if count is None:
        raise ValueError("n_valid must be a whole number of windows")
    planned = _planned(count, tuple(float(w) for w in widths))
    answer: dict[str, Any] = json.loads(planned)
    return answer


def _share_values(inner: dict[str, Any]) -> dict[str, float]:
    """Each zone's share of the result's valid pixels, by file, as a fraction
    (the ``studio_zones`` fact, which a result of several zones carries)."""
    for item in inner.get("facts") or []:
        if isinstance(item, dict) and item.get("id") == "studio_zones":
            return {
                str(zone.get("file")): float(zone["share"])
                for zone in item.get("zones") or []
                if isinstance(zone, dict)
                and isinstance(zone.get("share"), (int, float))
                and not isinstance(zone.get("share"), bool)
            }
    return {}


def _shares(inner: dict[str, Any]) -> dict[str, str]:
    """Each zone's share of the result's valid pixels, by file, as a percent
    (:func:`_share_values`, written as the result's sentences write it)."""
    return {file: rules.percent(value) for file, value in _share_values(inner).items()}


def _period(value: Any) -> dict[str, str] | None:
    """A result's period as the offer record keeps it: its start and end as
    given (``date_window``, at most 40 characters each), or ``None``."""
    if not isinstance(value, dict):
        return None
    start, end = value.get("start"), value.get("end")
    if isinstance(start, str) and isinstance(end, str) and start and end:
        return {"start": start[:40], "end": end[:40]}
    return None


def _stale(draw: dict[str, Any], now: float) -> bool:
    """Whether a draw ``drawing`` was started :data:`STALE_SECONDS` or more
    ago (or at a time that cannot be read, or later than now)."""
    started = draw.get("started_at")
    if not isinstance(started, (int, float)) or isinstance(started, bool):
        return True
    return not 0 <= now - float(started) < STALE_SECONDS


def _texts(value: Any) -> list[str]:
    """A record's ``must_state`` as a list of text lines (anything else: none)."""
    if isinstance(value, str):
        return [value]
    if not isinstance(value, list):
        return []
    return [str(line) for line in value if isinstance(line, str)]


def _claims(value: Any) -> list[dict[str, Any]]:
    """A plan's ``forbidden_claims`` as the plan gave them: each entry that is
    an object, unchanged, at most :data:`MAX_CLAIMS` (anything else: none)."""
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, dict)][:MAX_CLAIMS]


def _facts(value: Any) -> list[dict[str, Any]]:
    """A plan's ``facts`` as the plan gave them: each entry that is an object,
    unchanged, at most :data:`MAX_FACTS` (anything else: none)."""
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, dict)][:MAX_FACTS]


def _pinned(draw: dict[str, Any]) -> dict[str, Any]:
    """A draw's settings pinned at its first press, as the card names them: the
    count (``budget``) and when it was pressed (a press retries with these)."""
    return {"budget": _whole(draw.get("budget")), "pressed_at": draw.get("pressed_at")}


def _sheet_words(path: Any) -> dict[str, Any]:
    """A drawn sheet's progress line and button words (``sheet.sheet_progress``:
    counts only), or why the sheet cannot be read (``sheet_error``).

    The sheet's own refusals (a ``ValueError``, a ``UnicodeDecodeError``
    among them), the disk's errors and ``csv``'s (:data:`SHEET_NOT_CSV`) are
    each this zone's: one odd sheet never fails the card's other zones.
    """
    try:
        progress = sheet.sheet_progress(str(path))
    except (ValueError, OSError) as exc:
        return {"progress": None, "open_label": None, "sheet_error": str(exc)}
    except csv.Error as exc:
        error = SHEET_NOT_CSV.format(name=Path(str(path)).name, error=exc)
        return {"progress": None, "open_label": None, "sheet_error": error}
    # The counts too (never a row): the card's heading says every square is
    # answered once each drawn sheet is labelled whole.
    return {
        "progress": progress["progress"],
        "open_label": progress["open_label"],
        "n_rows": _whole(progress.get("n_rows")),
        "n_labelled": _whole(progress.get("n_labelled")),
    }


def _stale_at(draw: dict[str, Any]) -> str | None:
    """When a draw left ``drawing`` past :data:`STALE_SECONDS` began to wait,
    as :func:`_stamp` writes it: the time it was claimed (``started_at``,
    which each retry's claim moves on, while ``pressed_at`` keeps the first
    press's, so a draw that stops again lets the others have their turn),
    else its ``pressed_at``, else ``None``."""
    started = draw.get("started_at")
    if isinstance(started, (int, float)) and not isinstance(started, bool):
        try:
            return _stamp(float(started))
        except (OverflowError, OSError, ValueError):
            pass  # NaN, or too far off for gmtime: the press's time below
    pressed = draw.get("pressed_at")
    return pressed if isinstance(pressed, str) else None


def _zone_state(folder: Path, key: str, digest: Any) -> dict[str, Any]:
    """A zone's state on the card, from its draw record.

    ``"to draw"`` with no draw; ``"busy"`` while another press draws it;
    ``"failed"`` with its ``error``, whether a press retries it (``retry``),
    the count and time pinned at its first press (``budget``,
    ``pressed_at``: a retry draws with them) and when it failed
    (``failed_at``, as :func:`_failed` writes it, ``None`` when the record
    holds none), as for a draw left ``drawing`` past :data:`STALE_SECONDS`,
    whose ``failed_at`` is when it was claimed (:func:`_stale_at`). The card
    retries first the failed draws that have waited longest, by this time
    (r4-l L2: a time, not a place, so it may reach the view data; the card
    never tells it to Claude or shows it); a record that cannot be read has
    none. ``"drawn"`` with ``draw``: when
    it was pressed, its count, design and seed, its sheet, the plan's
    ``facts``, ``must_state`` and ``forbidden_claims`` (for the model; the
    server leaves the facts and the claims out of the view data) and the
    sheet's progress words. ``raster_changed`` when the draw's raster digest is not the
    offer's ``digest``.
    """
    try:
        draw = _read_json(folder / f"{key}.draw.json")
    except ValueError:
        return {"state": "failed", "error": UNREADABLE, "retry": False}
    if draw is None:
        return {"state": "to draw"}
    if draw.get("format") != DRAW_FORMAT:
        return {"state": "failed", "error": UNREADABLE, "retry": False}
    out: dict[str, Any]
    state = draw.get("state")
    if state == "drawn":
        out = {
            "state": "drawn",
            "draw": {
                "pressed_at": draw.get("pressed_at"),
                "budget": draw.get("budget"),
                "design": draw.get("design"),
                "seed": draw.get("seed"),
                "labels_csv_path": draw.get("labels_csv_path"),
                "facts": _facts(draw.get("facts")),
                "must_state": _texts(draw.get("must_state")),
                "forbidden_claims": _claims(draw.get("forbidden_claims")),
                **_sheet_words(draw.get("labels_csv_path")),
            },
        }
    elif state == "drawing" and not _stale(draw, _now()):
        out = {"state": "busy"}
    elif state == "drawing":
        out = {
            "state": "failed",
            "error": STOPPED,
            "retry": True,
            **_pinned(draw),
            "failed_at": _stale_at(draw),
        }
    elif state == "failed":
        failed_at = draw.get("failed_at")
        out = {
            "state": "failed",
            "error": draw.get("error"),
            "retry": bool(draw.get("retry")),
            **_pinned(draw),
            "failed_at": failed_at if isinstance(failed_at, str) else None,
        }
    else:
        return {"state": "failed", "error": UNREADABLE, "retry": False}
    if draw.get("raster_sha256") != digest:
        out["raster_changed"] = True
    return out


def offer_zones(
    inner: dict[str, Any],
    *,
    map_name: str | None = None,
    rule_needed: bool = False,
    rule_owner: bool = False,
) -> list[dict[str, Any]]:
    """Write each zone's offer record and return the zones as the card shows them.

    ``inner`` is ``olmoearth_scores_from_studio``'s result. One entry per
    zone, in its order: ``check_key``, ``zone`` (:func:`zone_label`),
    ``file``, ``part`` (:func:`part_names`: ``None`` for a result of one
    zone), ``share`` (of the result's valid pixels, as a percent) and
    ``share_value`` (the same, a fraction; both ``None`` for a result of one
    zone), ``n_valid`` (its windows), then ``offer`` (:func:`offers`)
    or ``refused`` and ``refused_at``: :data:`REFUSED_READ` for a zone not
    read as a map, :data:`REFUSED_PLAN` for one with no count planned (the
    planner's sentence, or the package missing), and :func:`_zone_state`'s
    ``state`` and the rest. Each zone's offer record
    (``<key>.json``: :data:`CHECK_FORMAT`, the result and prediction ids,
    the zone's file, scores path, raster digest and windows, the offer and
    when it was made; ``map_name`` as the server read it, the result's
    ``period``, every zone's ``files``, this zone's ``part``,
    ``share_value`` and ``n_parts``, ``rule_needed``, a map with no
    labelling rule of the owner's, and ``rule_owner``, a map whose owner's
    rule the user may replace with its own before its first part is drawn)
    is written again whole; a refused zone's offers no count. The answer
    holds no scores path or raster digest. Blocking: the server calls it in
    a worker thread.
    """
    result_id = str(inner.get("result_id") or "").strip()
    if not result_id:
        raise ValueError("the result has no result_id")
    prediction_id = inner.get("prediction_id")
    zones = [z for z in inner.get("zones") or [] if isinstance(z, dict)]
    files = [str(z.get("file") or "").strip() for z in zones]
    named = [file for file in files if file]
    parts = dict(zip(named, part_names(named)))
    shares = _shares(inner) if len(zones) > 1 else {}
    values = _share_values(inner) if len(zones) > 1 else {}
    period = _period(inner.get("date_window"))
    name = map_name if isinstance(map_name, str) and map_name else None
    folder = checks_dir()
    out: list[dict[str, Any]] = []
    for entry, file in zip(zones, files):
        if not file:
            continue
        key = check_key(result_id, file)
        held = entry.get("pooled")
        pooled: dict[str, Any] = held if isinstance(held, dict) else {}
        raster = pooled.get("raster_check")
        digest = raster.get("sha256") if isinstance(raster, dict) else None
        n_valid = _whole(pooled.get("n_valid"))
        scores_path = entry.get("scores_path")
        offer: list[dict[str, Any]] = []
        refused: str | None
        refused_at = REFUSED_READ
        if not isinstance(scores_path, str) or not scores_path:
            refused = str(entry.get("refused") or "this zone was not read as a map")
        elif n_valid is None or not isinstance(digest, str) or not digest:
            # Shown on the card's face: no "window" there (the owner's ruling 6).
            # WORDING: squares
            refused = "this zone's scores file gives no count of its squares"
        else:
            refused_at = REFUSED_PLAN
            try:
                planned = offers(n_valid)
            except inferencex.InferencexMissingError:
                planned = {
                    "offer": [],
                    "refused": inferencex.missing_extra("Planning label counts")[
                        "reason"
                    ],
                }
            offer, refused = planned["offer"], planned["refused"]
        record = {
            "format": CHECK_FORMAT,
            "check_key": key,
            "result_id": result_id,
            "prediction_id": prediction_id,
            "zone_file": file,
            "scores_path": scores_path if isinstance(scores_path, str) else None,
            "raster_sha256": digest if isinstance(digest, str) else None,
            "n_valid": n_valid,
            "offer": offer,
            "refused": None if offer else refused,
            "offered_at": _stamp(_now()),
            "map_name": name,
            "period": period,
            "files": named,
            "part": parts.get(file),
            "share_value": values.get(file),
            "n_parts": len(named),
            "rule_needed": bool(rule_needed),
            "rule_owner": bool(rule_owner),
        }
        _write_json(folder / f"{key}.json", record)
        zone: dict[str, Any] = {
            "check_key": key,
            "zone": zone_label(file, files),
            "file": file,
            "part": parts.get(file),
            "share": shares.get(file),
            "share_value": values.get(file),
            "n_valid": n_valid,
        }
        if offer:
            zone["offer"] = offer
        else:
            zone["refused"] = refused
            zone["refused_at"] = refused_at
        zone.update(_zone_state(folder, key, record["raster_sha256"]))
        out.append(zone)
    return out


def read_offer(key: str) -> dict[str, Any]:
    """A check's offer record; ``ValueError`` (:data:`UNKNOWN_CHECK`) when none
    is written here, or it is not this key's. The key is checked first."""
    key = _valid_key(key)
    record = _read_json(checks_dir() / f"{key}.json")
    if (
        record is None
        or record.get("format") != CHECK_FORMAT
        or record.get("check_key") != key
        or check_key(str(record.get("result_id")), str(record.get("zone_file"))) != key
    ):
        raise ValueError(UNKNOWN_CHECK)
    return record


def _offered(offer: dict[str, Any]) -> set[int]:
    """The counts an offer record offers."""
    counts: set[int] = set()
    for item in offer.get("offer") or []:
        count = _whole(item.get("labels")) if isinstance(item, dict) else None
        if count is not None:
            counts.add(count)
    return counts


def claim_draw(
    key: str,
    budget: Any,
    pressed_at: str | None = None,
    *,
    reading_rule: str | None = None,
    rule_typed_at: str | None = None,
) -> dict[str, Any]:
    """Decide what a press on a check does, and record it, under the draw's lock.

    ``reading_rule`` is the labelling rule the server took for this press
    (the user's sentence, already checked, or the one an earlier part of the
    map pinned: :func:`map_rule`): a first press pins it with
    ``rule_typed_at``, the time it was typed (``rule_typed_at`` when given,
    an earlier part's for its pinned rule, so every part of the map keeps
    the day it was typed: pass-3 finding Q7; else this press's time, for a
    rule sent with this press), and a retry of a draw that pinned none pins
    it then the same way; a draw that pinned one keeps it.

    ``ValueError`` for a key that is not a check's (before any file is
    touched) or a check with no offer here (:data:`UNKNOWN_CHECK`); then the
    draw record is read. :data:`NOT_OFFERED` for a first press with a count
    the offer does not hold, and for a retry with neither its pinned count
    nor one the offer holds, or while the newest offer holds no count (the
    package missing, say: the retry waits for a read that offers counts
    again). Otherwise ``{"outcome", "draw"}``, and ``error`` where the press
    draws nothing:

    - ``"first"``: no draw yet. The draw is written ``drawing`` with the
      press's time (``pressed_at`` when given: the server takes it once per
      press, before its lock, :func:`stamp_now`; else now), the process id, the raster digest of the offer and its
      request pinned: ``{scores_path, design, budget, seed}`` (:data:`DESIGN`,
      :data:`SEED`, this count);
    - ``"busy"``: ``drawing``, started under :data:`STALE_SECONDS` ago;
    - ``"changed"``: the offer's raster digest is not the draw's (Studio's
      files changed): refused (:data:`RASTER_CHANGED`);
    - ``"repeat"``: already ``drawn``: the stored draw, whatever count was
      sent now;
    - ``"failed"``: ``failed`` with an error the same settings give again:
      not drawn again;
    - ``"retry"``: ``failed`` with an error that may pass, or ``drawing``
      for :data:`STALE_SECONDS` or more: written ``drawing`` again, the
      pinned request unchanged (never the count sent now; the pinned count
      was checked against the offer at the first press).

    The caller runs the plan with ``draw["request"]`` on ``"first"`` and
    ``"retry"`` only, then :func:`finish_draw` or :func:`fail_draw`.
    """
    key = _valid_key(key)
    count = _whole(budget)
    folder = checks_dir()
    with _draw_lock(folder, key):
        offer = read_offer(key)
        offered = _offered(offer)
        path = folder / f"{key}.draw.json"
        draw = _read_json(path)
        now = _now()
        if draw is None:
            if count is None or count not in offered:
                raise ValueError(NOT_OFFERED)
            draw = {
                "format": DRAW_FORMAT,
                "check_key": key,
                "result_id": offer.get("result_id"),
                "prediction_id": offer.get("prediction_id"),
                "zone_file": offer.get("zone_file"),
                "raster_sha256": offer.get("raster_sha256"),
                "state": "drawing",
                "pressed_at": pressed_at or _stamp(now),
                "started_at": now,
                "pid": os.getpid(),
                "attempts": 1,
                "design": DESIGN,
                "seed": SEED,
                "budget": count,
                "request": {
                    "scores_path": offer.get("scores_path"),
                    "design": DESIGN,
                    "budget": count,
                    "seed": SEED,
                },
            }
            if isinstance(reading_rule, str) and reading_rule:
                draw["reading_rule"] = reading_rule
                draw["rule_typed_at"] = _typed_at(rule_typed_at, draw["pressed_at"])
            _write_json(path, draw)
            return {"outcome": "first", "draw": draw}
        if draw.get("format") != DRAW_FORMAT or draw.get("check_key") != key:
            raise ValueError(UNREADABLE)
        state = draw.get("state")
        if state == "drawing" and not _stale(draw, now):
            return {"outcome": "busy", "draw": draw}
        if draw.get("raster_sha256") != offer.get("raster_sha256"):
            return {"outcome": "changed", "draw": draw, "error": RASTER_CHANGED}
        if state == "drawn":
            return {"outcome": "repeat", "draw": draw}
        if state == "failed" and not draw.get("retry"):
            return {"outcome": "failed", "draw": draw, "error": draw.get("error")}
        if state not in ("failed", "drawing"):
            raise ValueError(UNREADABLE)
        pinned = _whole(draw.get("budget"))
        if count is None or not offered or (count != pinned and count not in offered):
            raise ValueError(NOT_OFFERED)
        kept = {
            k: v for k, v in draw.items() if k not in ("error", "retry", "failed_at")
        }
        attempts = _whole(draw.get("attempts")) or 1
        draw = {
            **kept,
            "state": "drawing",
            "started_at": now,
            "pid": os.getpid(),
            "attempts": attempts + 1,
        }
        pinned_rule = draw.get("reading_rule")
        if isinstance(reading_rule, str) and reading_rule and not pinned_rule:
            draw["reading_rule"] = reading_rule
            draw["rule_typed_at"] = _typed_at(rule_typed_at, pressed_at or _stamp(now))
        _write_json(path, draw)
        return {"outcome": "retry", "draw": draw}


def _typed_at(given: Any, pressed: str) -> str:
    """When a pinned rule was typed: ``given`` (an earlier part's time for its
    pinned rule) when it is text, else ``pressed``, the press's time."""
    return given if isinstance(given, str) and given else pressed


def unpin_rule(key: str) -> dict[str, Any] | None:
    """Take a pinned rule off a check's draw record (``reading_rule`` and
    ``rule_typed_at``), under the draw's lock, and return the record; ``None``
    when the check has no draw record here. For a draw whose rule could not
    be written beside its sheet because the sheet already held answers
    (pass-3 finding Q9): :func:`map_rule` then no longer reports it, so the
    card says the rule the sheet's answers followed. ``ValueError`` for a
    key that is not a check's (before any file is touched) or a record that
    cannot be read (:data:`UNREADABLE`)."""
    key = _valid_key(key)
    folder = checks_dir()
    with _draw_lock(folder, key):
        path = folder / f"{key}.draw.json"
        draw = _read_json(path)
        if draw is None:
            return None
        if draw.get("format") != DRAW_FORMAT or draw.get("check_key") != key:
            raise ValueError(UNREADABLE)
        if "reading_rule" not in draw and "rule_typed_at" not in draw:
            return draw
        kept = {
            k: v for k, v in draw.items() if k not in ("reading_rule", "rule_typed_at")
        }
        _write_json(path, kept)
        return kept


def _retryable(error: str, request: Any) -> bool:
    """Whether a press may pass where the plan failed with ``error``: not for
    an error the same settings give again (:data:`FINAL_ERRORS`), nor once
    the pinned scores file is gone."""
    path = request.get("scores_path") if isinstance(request, dict) else None
    if not isinstance(path, str) or not os.path.isfile(path):
        return False
    return not error.startswith(FINAL_ERRORS)


def _plan_outcome(
    envelope: Any, request: Any
) -> tuple[dict[str, Any] | None, str, bool]:
    """What a plan's envelope gives a draw: its fields, or an error and whether
    a press may retry it.

    A drawn plan answered ``ok`` with a ``labels_csv_path``; its fields are
    that path, ``design_path``, ``n_population``, its ``facts``,
    ``must_state`` and ``forbidden_claims`` as it gave them (:func:`_facts`,
    :func:`_claims`; the model reads them with the zone, the view data does
    not). A plan that answered
    ``available: false`` (the package missing) or with no sheet is a failed
    draw a press does not retry.
    """
    if not isinstance(envelope, dict) or envelope.get("ok") is not True:
        error = envelope.get("error") if isinstance(envelope, dict) else None
        text = str(error or "The plan failed.")
        return None, text, _retryable(text, request)
    result = envelope.get("result")
    if not isinstance(result, dict):
        return None, NO_SHEET, False
    if result.get("available") is False:
        return None, str(result.get("reason") or "The plan is not available."), False
    labels = result.get("labels_csv_path")
    if not isinstance(labels, str) or not labels:
        return None, NO_SHEET, False
    must = result.get("must_state") or envelope.get("must_state") or []
    must = [must] if isinstance(must, str) else must
    claims = result.get("forbidden_claims") or envelope.get("forbidden_claims")
    facts = result.get("facts") or envelope.get("facts")
    design_path = result.get("design_path")
    return (
        {
            "labels_csv_path": labels,
            "design_path": design_path if isinstance(design_path, str) else None,
            "n_population": _whole(result.get("n_population")),
            "facts": _facts(facts),
            "must_state": [str(s) for s in must] if isinstance(must, list) else [],
            "forbidden_claims": _claims(claims),
        },
        "",
        False,
    )


def _claimed(folder: Path, key: str) -> tuple[Path, dict[str, Any]]:
    """A check's draw record, which a press started; ``ValueError`` otherwise."""
    path = folder / f"{key}.draw.json"
    draw = _read_json(path)
    if draw is None:
        raise ValueError(NOT_CLAIMED)
    if draw.get("format") != DRAW_FORMAT or draw.get("check_key") != key:
        raise ValueError(UNREADABLE)
    return path, draw


def finish_draw(key: str, envelope: Any) -> dict[str, Any]:
    """Record the plan a press ran (its envelope); return the draw record.

    A drawn plan (:func:`_plan_outcome`) makes the draw ``drawn``, with its
    sheet, design file, windows, ``facts``, ``must_state``,
    ``forbidden_claims`` and the time; any other
    answer makes it ``failed`` (:func:`fail_draw`), retried by a press only
    when the error may pass. A draw already ``drawn`` is kept as it is.
    """
    key = _valid_key(key)
    folder = checks_dir()
    with _draw_lock(folder, key):
        path, draw = _claimed(folder, key)
        if draw.get("state") == "drawn":
            return draw
        fields, error, retry = _plan_outcome(envelope, draw.get("request"))
        if fields is None:
            draw = _failed(draw, error, retry)
        else:
            kept = {
                k: v
                for k, v in draw.items()
                if k not in ("error", "retry", "failed_at")
            }
            draw = {**kept, **fields, "state": "drawn", "drawn_at": _stamp(_now())}
        _write_json(path, draw)
        return draw


def _failed(draw: dict[str, Any], error: str, retry: bool) -> dict[str, Any]:
    """``draw`` made ``failed``: its error (cut to :data:`MAX_ERROR_CHARS`),
    whether a press retries it, and the time."""
    return {
        **draw,
        "state": "failed",
        "error": str(error)[:MAX_ERROR_CHARS],
        "retry": bool(retry),
        "failed_at": _stamp(_now()),
    }


def fail_draw(key: str, error: str, *, retry: bool) -> dict[str, Any]:
    """Record that a press's draw failed with ``error``; return the draw record.

    ``retry``: whether a later press draws it again with the pinned
    settings; ``False`` for a failure the same settings give again, whose
    error the card shows whole, pointing to Claude. A draw already
    ``drawn`` (by another process) is kept as it is.
    """
    key = _valid_key(key)
    folder = checks_dir()
    with _draw_lock(folder, key):
        path, draw = _claimed(folder, key)
        if draw.get("state") == "drawn":
            return draw
        draw = _failed(draw, error, retry)
        _write_json(path, draw)
        return draw


def draw_view(
    record: dict[str, Any], *, outcome: str | None = None, error: str | None = None
) -> dict[str, Any]:
    """What the page gets of a draw: :data:`VIEW_KEYS` and the press's ``outcome``.

    ``error``, when given, replaces the record's (a press refused without a
    rewrite). Never the request, the scores path, the raster digest, a
    window, a class or a figure.
    """
    view = {k: record.get(k) for k in VIEW_KEYS}
    if not isinstance(view.get("must_state"), list):
        view["must_state"] = []
    view["outcome"] = outcome
    if error is not None:
        view["error"] = error
    return view


def log_entry(
    key: Any,
    *,
    outcome: str,
    asked: Any,
    draw: dict[str, Any] | None = None,
    offer: dict[str, Any] | None = None,
    pressed_at: str | None = None,
) -> dict[str, Any]:
    """One zone's line of a press for the log (:func:`append_log`).

    ``pressed_at`` is the press's time when given (the server takes it once
    per press, before its lock and every claim, :func:`stamp_now`), else
    now; the result, the zone
    and, from a draw, its pinned count and its state after the press;
    ``asked`` is the count the page sent; ``outcome`` is
    :func:`claim_draw`'s, ``"refused"``, or ``"cancelled"`` (a press
    stopped mid-plan).
    """
    record = draw or offer or {}
    return {
        "pressed_at": pressed_at or _stamp(_now()),
        "check_key": key,
        "result_id": record.get("result_id"),
        "zone_file": record.get("zone_file"),
        "budget": draw.get("budget") if draw else _whole(asked),
        "asked": asked,
        "design": DESIGN,
        "seed": SEED,
        "outcome": outcome,
        "state": draw.get("state") if draw else None,
    }


def _short(value: Any, limit: int = 200) -> str | None:
    """``value`` as text of at most ``limit`` characters, or ``None``."""
    return None if value is None else str(value)[:limit]


def append_log(entry: dict[str, Any]) -> None:
    """Add one zone's line of a press to ``log.jsonl`` (0600), under a lock.

    Only the log's own fields are written, each checked: a key that is not
    a check's is written as ``null``, counts as whole numbers or ``null``,
    the rest as short text. ``pressed_at`` is now unless the entry has it.
    """
    key = entry.get("check_key")
    line = {
        "pressed_at": _short(entry.get("pressed_at")) or _stamp(_now()),
        "check_key": (
            key if isinstance(key, str) and KEY_PATTERN.fullmatch(key) else None
        ),
        "result_id": _short(entry.get("result_id")),
        "zone_file": _short(entry.get("zone_file")),
        "budget": _whole(entry.get("budget")),
        "asked": _whole(entry.get("asked")),
        "design": _short(entry.get("design"), 32),
        "seed": _whole(entry.get("seed")),
        "outcome": _short(entry.get("outcome"), 32),
        "state": _short(entry.get("state"), 32),
    }
    data = (json.dumps(line, sort_keys=True) + "\n").encode("utf-8")
    folder = checks_dir()
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0)
    with _LOG_LOCK:
        fd = os.open(folder / LOG_NAME, flags, 0o600)
        try:
            if fcntl is not None:
                fcntl.flock(fd, fcntl.LOCK_EX)
            written = 0
            while written < len(data):
                written += os.write(fd, data[written:])
        finally:
            os.close(fd)


# --------------------------------------------------------------------------- what the cards read back


def _existing_checks_dir() -> Path | None:
    """The folder of the check records when it exists, else ``None``; never
    made here (a read never writes)."""
    try:
        folder = safe_path(CHECKS_DIR, root=write_root())
    except (OSError, ValueError):
        return None
    return folder if folder.is_dir() else None


def _draw_records(folder: Path) -> Iterator[dict[str, Any]]:
    """The draw records in ``folder``, by file name, at most :data:`SCAN_MAX`
    read; one that cannot be read, or is not a draw record, is skipped."""
    try:
        names = sorted(
            entry.name
            for entry in os.scandir(folder)
            if entry.name.endswith(".draw.json") and not entry.name.startswith(".")
        )
    except OSError:
        return
    for name in names[:SCAN_MAX]:
        try:
            record = _read_json(folder / name)
        except ValueError:
            continue
        if record is not None and record.get("format") == DRAW_FORMAT:
            yield record


#: An offer record's file name (``<key>.json``).
_OFFER_NAME = re.compile(r"[0-9a-f]{24}\.json")


def _result_files(folder: Path, result_id: str, zone_file: str) -> list[str] | None:
    """Every zone's file of a result, from another of its offer records that
    keeps them (``files``, holding ``zone_file``), at most :data:`SCAN_MAX`
    read; ``None`` when none does (records written before the files were
    kept)."""
    try:
        names = sorted(
            entry.name
            for entry in os.scandir(folder)
            if _OFFER_NAME.fullmatch(entry.name)
        )
    except OSError:
        return None
    for name in names[:SCAN_MAX]:
        key = name.removesuffix(".json")
        record = _record_quiet(folder, key, ".json", CHECK_FORMAT)
        if record is None or str(record.get("result_id") or "").strip() != result_id:
            continue
        listed = record.get("files")
        if not isinstance(listed, list):
            continue
        files = [str(f) for f in listed if isinstance(f, str) and f]
        if zone_file in files:
            return files
    return None


def _record_quiet(
    folder: Path, key: Any, suffix: str, form: str
) -> dict[str, Any] | None:
    """One check's record (``<key><suffix>``) of format ``form``, or ``None``
    for a key that is not one, a record missing, unreadable or not this key's."""
    if not isinstance(key, str) or KEY_PATTERN.fullmatch(key) is None:
        return None
    try:
        record = _read_json(folder / f"{key}{suffix}")
    except ValueError:
        return None
    if record is None or record.get("format") != form or record.get("check_key") != key:
        return None
    return record


def _same_file(path: Any, real: str) -> bool:
    """Whether a record's sheet path is the sheet whose real path is ``real``."""
    if not isinstance(path, str) or not path:
        return False
    if path == real:
        return True
    try:
        return os.path.realpath(path) == real
    except (OSError, ValueError):
        return False


def map_rule(result_id: Any) -> dict[str, str] | None:
    """The labelling rule a part of this result pinned at its press (the
    user's sentence for a map with no rule of the owner's): ``{text,
    typed_at}``, the earliest pinned, or ``None``. Read from the draw records
    (at most :data:`SCAN_MAX`); never raises."""
    rid = str(result_id or "").strip()
    folder = _existing_checks_dir()
    if not rid or folder is None:
        return None
    found: list[tuple[str, str]] = []
    for draw in _draw_records(folder):
        text = draw.get("reading_rule")
        if str(draw.get("result_id") or "").strip() != rid:
            continue
        if isinstance(text, str) and text:
            found.append((str(draw.get("rule_typed_at") or ""), text))
    if not found:
        return None
    typed_at, text = min(found)
    return {"text": text, "typed_at": typed_at}


def result_settled(result_id: Any) -> bool:
    """Whether a part of this result is drawn, so its labelling rule is
    settled (the owner's ruling of 13 Nov 2026, option A: no rule replaces
    the owner's after that): a draw record of the result ``drawn``, or
    ``drawing`` and not stale (being drawn now; the card shows it busy).
    Read from the draw records (at most :data:`SCAN_MAX`); never raises."""
    rid = str(result_id or "").strip()
    folder = _existing_checks_dir()
    if not rid or folder is None:
        return False
    now = _now()
    for draw in _draw_records(folder):
        if str(draw.get("result_id") or "").strip() != rid:
            continue
        state = draw.get("state")
        if state == "drawn" or (state == "drawing" and not _stale(draw, now)):
            return True
    return False


def sheet_context(labels_path: Any) -> dict[str, Any] | None:
    """Which part of which map a Check card's sheet is, for the cards; or
    ``None`` for a sheet no Check card drew here.

    The sheet is found among the drawn records (at most :data:`SCAN_MAX`
    read) by its real path. ``{result_id, zone_file, part, n_parts,
    parts_known, map_name, period, parts}``: ``part`` is this zone's name
    (:func:`part_names`, ``None`` for a map of one part), ``map_name`` and
    ``period`` as its offer record keeps them, and ``parts`` each zone of the
    result in order, ``{part, share_value, labels_csv_path, this}`` (the
    sheet of a zone drawn here, else ``None``; for the server's use, never
    the page's). The map's files are its offer record's; for a record
    written before the files were kept, another offer record's of the same
    result that keeps them (:func:`_result_files`); with neither, this part
    alone, and ``parts_known`` is false (how many parts the map has is not
    known). Never raises.
    """
    if not isinstance(labels_path, str) or not labels_path or "\x00" in labels_path:
        return None
    folder = _existing_checks_dir()
    if folder is None:
        return None
    try:
        real = os.path.realpath(labels_path)
    except (OSError, ValueError):
        return None
    mine: dict[str, Any] | None = None
    for draw in _draw_records(folder):
        if draw.get("state") == "drawn" and _same_file(
            draw.get("labels_csv_path"), real
        ):
            mine = draw
            break
    if mine is None:
        return None
    result_id = str(mine.get("result_id") or "").strip()
    zone_file = str(mine.get("zone_file") or "").strip()
    offer = _record_quiet(folder, mine.get("check_key"), ".json", CHECK_FORMAT)
    listed = offer.get("files") if offer else None
    files = (
        [str(f) for f in listed if isinstance(f, str) and f]
        if isinstance(listed, list)
        else []
    )
    known = zone_file in files
    if not known:
        found = _result_files(folder, result_id, zone_file)
        known = found is not None
        files = found if found is not None else [zone_file]
    names = part_names(files)
    parts: list[dict[str, Any]] = []
    for file, name in zip(files, names):
        this = file == zone_file
        key = check_key(result_id, file)
        its_offer = offer if this else _record_quiet(folder, key, ".json", CHECK_FORMAT)
        its_draw = (
            mine if this else _record_quiet(folder, key, ".draw.json", DRAW_FORMAT)
        )
        path = its_draw.get("labels_csv_path") if its_draw is not None else None
        drawn = its_draw is not None and its_draw.get("state") == "drawn"
        share = its_offer.get("share_value") if its_offer is not None else None
        parts.append(
            {
                "part": name,
                "share_value": (
                    float(share)
                    if isinstance(share, (int, float)) and not isinstance(share, bool)
                    else None
                ),
                "labels_csv_path": (
                    path if drawn and isinstance(path, str) and path else None
                ),
                "this": this,
            }
        )
    map_name = offer.get("map_name") if offer else None
    return {
        "result_id": result_id,
        "zone_file": zone_file,
        "part": names[files.index(zone_file)],
        "n_parts": len(files),
        "parts_known": known,
        "map_name": map_name if isinstance(map_name, str) and map_name else None,
        "period": _period(offer.get("period")) if offer else None,
        "parts": parts,
    }
