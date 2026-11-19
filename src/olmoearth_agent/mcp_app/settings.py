# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""The Settings box's three rows (round r5-n S2, the owner's design of 19 Nov 2026).

Every panel ends with a quiet Settings text button (``views/common.js``,
``OE.settingsBlock``). Its box reads the rows here through the app-only
``olmoearth_app_settings`` (:func:`state`) and changes one, at the user's
press, through the app-only ``olmoearth_app_setting`` (:func:`change`).
Both are quiet: the model reads ``{"ok": true}`` from them on every path.

1. Ai2's Asta research skills: a switch, off by default, kept in
   :func:`settings_path` (``<output root>/settings.json``, mode 600, written
   by an atomic rename). The file holds the switch alone: no key, no id, no
   place. A missing, unreadable or broken file reads as the defaults
   (:data:`DEFAULTS`). On, the start card shows one more job tile, whose
   press is the only thing in the app that asks Claude for Asta; turning the
   switch on or off changes nothing else and tells Claude nothing.
2. Satellite pictures: whether the user said yes to the pictures question
   (:func:`imagery.is_on`, the current question's version) and on which day
   (the answer file's time, this computer's date). "Forget my yes" removes
   that answer (:func:`imagery.set_on` with ``False``), so the next square
   asks again; the question's version is unchanged.
3. Maps kept on this computer: how many Studio results' files are kept and
   still read in place of a download (:mod:`studio_scores`, round r5-h H2:
   a record that names its folder, downloaded less than
   :func:`studio_scores.keep_hours` ago). Reading the row first removes the
   copies past that limit, as the next download would, so "None kept." is
   true. "Remove kept copies" removes every kept copy whose record names its
   folder, an expired one too (:func:`studio_scores.forget_kept` for each); a
   folder whose record cannot be read is left as it is, as the prune does.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from olmoearth_agent.mcp_app import imagery
from olmoearth_agent.security.paths import safe_path, workspace_root
from olmoearth_agent.tools import studio_scores
from olmoearth_agent.tools.review_set import write_root

logger = logging.getLogger(__name__)

#: The settings file's name, in the output root.
SETTINGS_FILE = "settings.json"

#: The settings file's format.
SETTINGS_FORMAT = 1

#: What a missing or broken settings file reads as: the research skills
#: switch off.
DEFAULTS: dict[str, Any] = {"asta": False}

#: The changes ``olmoearth_app_setting`` takes, by ``name``: the switch
#: (with ``on``), forget the pictures' yes, remove the kept copies.
NAMES = ("asta", "forget_pictures", "remove_kept")

#: A day as the box says it ("19 Nov 2026"), month by month.
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

# WORDING: r5n
#: The pictures' row: allowed, on the day of the yes; or not yet.
PICTURES_ALLOWED = "You allowed them on {day}."
# WORDING: r5n
PICTURES_NOT_ALLOWED = "Not allowed yet."

# WORDING: r5n
#: The kept copies' row: how many, and for how long; or none.
KEPT_MANY = "{n} maps kept for {hours} h, so they open without downloading again."
# WORDING: r5n
KEPT_ONE = "1 map kept for {hours} h, so it opens without downloading again."
# WORDING: r5n
KEPT_NONE = "None kept."

# WORDING: r5n
#: A change the box does not know, and a switch given no true or false.
UNKNOWN_SETTING = "There is no such setting."
# WORDING: r5n
ON_TRUE_OR_FALSE = "on must be true or false"

#: One change of the settings file at a time in this process.
_lock = threading.Lock()


def settings_path() -> Path:
    """Where the settings are kept: ``settings.json`` in the output root."""
    return workspace_root() / SETTINGS_FILE


def read() -> dict[str, Any]:
    """The kept settings: :data:`DEFAULTS`, with each kept value of the
    right kind over them. A missing, unreadable or broken file, and a value
    of another kind, read as the defaults; never raises."""
    out = dict(DEFAULTS)
    try:
        loaded: Any = json.loads(settings_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return out
    if isinstance(loaded, dict) and isinstance(loaded.get("asta"), bool):
        out["asta"] = loaded["asta"]
    return out


def asta_on() -> bool:
    """Whether the research skills switch is on (off by default)."""
    return read()["asta"] is True


def _write(values: dict[str, Any]) -> None:
    """Write the settings file: mode 600, by an atomic rename, leaving no
    part file on failure. It holds the format and ``values`` alone."""
    path = settings_path()
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    data = json.dumps({"format": SETTINGS_FORMAT, **values}, indent=2) + "\n"
    part = path.with_name(f".{path.name}.{uuid.uuid4().hex[:8]}.part")
    try:
        fd = os.open(part, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(data)
        os.replace(part, path)
    except BaseException:
        part.unlink(missing_ok=True)
        raise


def set_asta(on: bool) -> None:
    """Turn the research skills switch on or off (the user's press). Nothing
    else changes, and nothing is told to Claude."""
    with _lock:
        values = read()
        values["asta"] = bool(on)
        _write(values)


def _day(seconds: float) -> str:
    """A time (seconds since the epoch) as this computer's day, "19 Nov 2026"."""
    when = time.localtime(seconds)
    return f"{when.tm_mday} {_MONTHS[when.tm_mon - 1]} {when.tm_year}"


def pictures() -> dict[str, Any]:
    """The pictures' row: ``allowed`` (the user's yes to the current
    question, :func:`imagery.is_on`) and the row's words (``said``), with
    the day the yes was kept (its file's time). A yes whose file goes before
    its time is read is not allowed."""
    not_allowed: dict[str, Any] = {"allowed": False, "said": PICTURES_NOT_ALLOWED}
    if not imagery.is_on():
        return not_allowed
    try:
        since = imagery.consent_path().stat().st_mtime
    except OSError:
        return not_allowed
    return {"allowed": True, "said": PICTURES_ALLOWED.format(day=_day(since))}


def _plain_folder(path: Path) -> bool:
    """Whether path is a folder and not a link (false when it cannot be read)."""
    try:
        return not path.is_symlink() and path.is_dir()
    except OSError:
        return False


def _kept() -> list[tuple[str, float | None]]:
    """Each kept copy whose record names its folder (a result id) and when
    it was downloaded (``None`` when its record says no readable time), in
    the folders' order. A folder that is a link, not a folder, or whose
    record cannot be read is left out; never raises."""
    try:
        base = safe_path(studio_scores.KEPT_DIR, root=write_root())
        folders = sorted(base.iterdir()) if base.is_dir() else []
    except (OSError, ValueError):
        return []
    out: list[tuple[str, float | None]] = []
    for folder in folders:
        if not _plain_folder(folder):
            continue
        record = studio_scores._kept_record(folder)
        if record is None or record.get("result_id") != folder.name:
            continue
        out.append((folder.name, studio_scores._downloaded_at(record)))
    return out


def _hours_text(hours: float) -> str:
    """Hours as the row says them: "24", "1.5"."""
    return str(int(hours)) if float(hours).is_integer() else f"{hours:g}"


def kept() -> dict[str, Any]:
    """The kept copies' row: ``count``, the copies still read in place of a
    download (downloaded less than :func:`studio_scores.keep_hours` ago; none
    while that is 0), and the row's words (``said``). The copies no read can
    use any more are removed first, as the next download would remove them
    (:func:`studio_scores._prune_kept`: those kept that many hours ago or
    longer; every readable copy while that is 0), so no expired copy stays
    on this computer behind "None kept.". Blocking: the handler calls it in
    a worker thread."""
    hours = studio_scores.keep_hours()
    try:
        # keep_hours is above 0 or 0.0: at 0 the limit is now and "" names
        # no folder, so every readable copy goes, as keep() prunes then.
        studio_scores._prune_kept(hours, "")
    except (OSError, ValueError):
        logger.debug("old kept copies were not pruned", exc_info=True)
    now = studio_scores._now()
    count = 0
    if hours > 0:
        for _result_id, moment in _kept():
            if moment is not None and 0 <= now - moment < hours * 3600:
                count += 1
    if count == 0:
        said = KEPT_NONE
    elif count == 1:
        said = KEPT_ONE.format(hours=_hours_text(hours))
    else:
        said = KEPT_MANY.format(n=count, hours=_hours_text(hours))
    return {"count": count, "said": said}


def remove_kept() -> int:
    """Remove every kept copy whose record names its folder, an expired one
    too (:func:`studio_scores.forget_kept`, which logs a failure by its kind
    and never raises); how many were asked to go."""
    ids = [result_id for result_id, _moment in _kept()]
    for result_id in ids:
        studio_scores.forget_kept(result_id)
    return len(ids)


def state() -> dict[str, Any]:
    """The box's three rows, as ``olmoearth_app_settings`` answers them: the
    switch (``asta.on``), the pictures (:func:`pictures`) and the kept copies
    (:func:`kept`). No key, no id, no path and no place."""
    return {"asta": {"on": asta_on()}, "pictures": pictures(), "kept": kept()}


def change(args: dict[str, Any]) -> dict[str, Any]:
    """One change at the user's press (``olmoearth_app_setting``), then the
    three rows again (:func:`state`): ``{"name": "asta", "on": bool}`` turns
    the switch on or off; ``{"name": "forget_pictures"}`` removes the
    pictures' yes, so the next square asks again (the question's version is
    unchanged); ``{"name": "remove_kept"}`` removes the kept copies. Any
    other name, or a switch given no true or false, is refused
    (``ValueError``) and changes nothing."""
    name = args.get("name")
    if name == "asta":
        on = args.get("on")
        if not isinstance(on, bool):
            raise ValueError(ON_TRUE_OR_FALSE)
        set_asta(on)
    elif name == "forget_pictures":
        imagery.set_on(False)
    elif name == "remove_kept":
        remove_kept()
    else:
        raise ValueError(UNKNOWN_SETTING)
    return state()
