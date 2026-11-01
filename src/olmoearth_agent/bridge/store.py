# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""A bridge run's directory: what one step reads and writes.

- ``brief.json``: the brief, the history and the run's settings, written once
  by ``begin`` (the skill index is kept as it was then, as one
  ``olmoearth-agent`` run builds its prompt once).
- ``responses.jsonl``: the model's responses so far, one line per model call,
  each with the fingerprint of the request it answered.
- ``tool_cache.jsonl``: each tool handler invocation's outcome, in call order,
  with the model call whose response made it.
- ``events.jsonl``: the loop's events, as the last step's replay yielded them.
- ``pending.json``: what the loop waits for (a model call), or its final answer.
- ``provenance.json``: the run's provenance log, as the last step rebuilt it.
- ``step.lock``: there only while a step runs: who runs it (host, pid, Slurm
  job) and since when (:meth:`RunDir.locked`).

A step appends to the tool cache as each new handler returns; it writes the
response it was given, then the events, the provenance and ``pending.json``
last, only once it has stopped cleanly. A step that failed has saved no
response and left ``pending.json`` as it was; the tool calls it ran stay in
the cache as provisional records (``replay.ToolCache``). A step stopped
after saving the response but before writing ``pending.json`` is finished by
the next ``step`` (``session.step``).

A step holds the run's lock from its first read of the run to its last
write, so two steps never replay one run at once: after a dropped ssh
connection the first may still run on the cluster, and a second is refused
while the lock is held, which ``show`` reports. A lock older than
:data:`STALE_LOCK_SECONDS` (srun's limit of 20 minutes, and a margin) is a
dead step's and is taken over. The lock is a file made with ``O_EXCL``, not
``flock``: successive srun steps run on different compute nodes, and
``flock`` on a parallel filesystem may be node-local or not supported.

An append cut off before its line ended (a killed step, a full disk) leaves
a last line with no ``"\\n"``: that record was never saved, so a read drops
it, in the file too (:meth:`RunDir._lines`), and the next append starts a
line of its own. A handler whose record was cut off runs again on the next
step; a response cut off counts as not saved.

The directory is made readable by its owner only (0700), also when it existed
empty, and a symlink is refused: its files hold the account's names and the
tools' whole results.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import socket
import time
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from olmoearth_agent.bridge.errors import BridgeError
from olmoearth_agent.bridge.scripted import RecordedResponse

logger = logging.getLogger(__name__)

__all__ = ["STALE_LOCK_SECONDS", "RunDir"]

#: A step's lock older than this many seconds is a dead step's: the srun step
#: that runs a command is killed at 20 minutes (``scripts/oe-bridge.sh``).
STALE_LOCK_SECONDS = 25 * 60


class RunDir:
    """The files of one bridge run (see the module docstring)."""

    BRIEF = "brief.json"
    RESPONSES = "responses.jsonl"
    TOOL_CACHE = "tool_cache.jsonl"
    EVENTS = "events.jsonl"
    PENDING = "pending.json"
    PROVENANCE = "provenance.json"
    LOCK = "step.lock"

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def file(self, name: str) -> Path:
        """The path of one of the run's files."""
        return self.path / name

    # --- the lock ---------------------------------------------------------

    @contextlib.contextmanager
    def locked(self) -> Iterator[None]:
        """Hold the run's lock for the block: no other step changes the run meanwhile.

        The lock file records the host, the pid, ``SLURM_JOB_ID`` and the UTC
        start time, and is removed at the block's end, whatever ends it, when
        it is still this block's.

        Raises
        ------
        BridgeError
            When another step holds a lock younger than
            :data:`STALE_LOCK_SECONDS`.
        """
        holder = {
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "job": os.environ.get("SLURM_JOB_ID"),
            "started": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        text = json.dumps(holder)
        self._take_lock(text)
        try:
            yield
        finally:
            self._release_lock(text)

    def read_lock(self) -> dict[str, Any] | None:
        """Who holds the run's lock and since when, or ``None`` when no step does.

        A lock older than :data:`STALE_LOCK_SECONDS` is a dead step's and reads
        as ``None``; one whose text cannot be read gives ``{}``.
        """
        age = self._lock_age()
        if age is None or age >= STALE_LOCK_SECONDS:
            return None
        try:
            held = json.loads(self.file(self.LOCK).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return held if isinstance(held, dict) else {}

    @staticmethod
    def running_text(held: dict[str, Any]) -> str:
        """A held lock in words: since when, which Slurm job, which host."""
        return (
            f"a step is still running on this run (since "
            f"{held.get('started') or 'an unknown time'}, job "
            f"{held.get('job') or 'unknown'}, host {held.get('host') or 'unknown'})"
        )

    def _lock_age(self) -> float | None:
        """Seconds since the lock was taken, or ``None`` when there is none."""
        try:
            return time.time() - self.file(self.LOCK).stat().st_mtime
        except FileNotFoundError:
            return None

    def _take_lock(self, text: str) -> None:
        """Create the lock file with ``text``; a stale one is taken over."""
        path = self.file(self.LOCK)
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
        try:
            fd = os.open(path, flags, 0o600)
        except FileExistsError:
            held = self.read_lock()
            if held is not None:
                raise BridgeError(
                    f"{self.running_text(held)}: wait for it to end, then run show"
                ) from None
            logger.warning(
                "bridge: took over %s, older than %d s (a dead step's lock)",
                path,
                STALE_LOCK_SECONDS,
            )
            with contextlib.suppress(FileNotFoundError):
                path.unlink()
            try:
                fd = os.open(path, flags, 0o600)
            except FileExistsError:
                raise BridgeError(
                    "another step took this run's lock meanwhile: wait for it to "
                    "end, then run show"
                ) from None
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)

    def _release_lock(self, text: str) -> None:
        """Remove the lock if it is still the one ``text`` wrote."""
        path = self.file(self.LOCK)
        with contextlib.suppress(FileNotFoundError):
            if path.read_text(encoding="utf-8") == text:
                path.unlink()

    # --- writing ---------------------------------------------------------

    def create(self, brief: dict[str, Any]) -> None:
        """Make the run's directory with its ``brief.json``; it must be new or empty.

        The directory is made readable by its owner only (mode 0700), also
        when it existed empty; a symlink is refused, since where it points and
        that directory's mode are not this run's to set.
        """
        if self.path.is_symlink():
            raise BridgeError(
                f"{self.path} is a symlink: begin needs a real run directory"
            )
        if self.path.exists() and (not self.path.is_dir() or any(self.path.iterdir())):
            raise BridgeError(
                f"{self.path} exists and is not an empty directory: begin needs a "
                "new run directory"
            )
        self.path.mkdir(mode=0o700, parents=True, exist_ok=True)
        # mkdir leaves an existing directory's mode as it was.
        os.chmod(self.path, 0o700)
        self._write(self.BRIEF, json.dumps(brief, indent=2, ensure_ascii=False))

    def append_response(self, response: RecordedResponse) -> None:
        """Add the model's response to ``responses.jsonl``."""
        self._append(self.RESPONSES, response.to_record())

    def append_tool_record(self, record: dict[str, Any]) -> None:
        """Add one handler outcome to ``tool_cache.jsonl``."""
        self._append(self.TOOL_CACHE, record)

    def write_tool_cache(self, records: list[dict[str, Any]]) -> None:
        """Replace ``tool_cache.jsonl`` with ``records`` (provisional ones dropped)."""
        lines = [json.dumps(r, ensure_ascii=False) for r in records]
        self._write(self.TOOL_CACHE, "".join(line + "\n" for line in lines))

    def write_events(self, events: list[dict[str, Any]]) -> None:
        """Replace ``events.jsonl`` with the events of this step's replay."""
        lines = [json.dumps(e, ensure_ascii=False, default=str) for e in events]
        self._write(self.EVENTS, "".join(line + "\n" for line in lines))

    def write_pending(self, view: dict[str, Any]) -> None:
        """Replace ``pending.json`` with this step's view."""
        self._write(self.PENDING, json.dumps(view, indent=2, ensure_ascii=False))

    def write_provenance(self, text: str) -> None:
        """Replace ``provenance.json`` (``ProvenanceLog.to_json``)."""
        self._write(self.PROVENANCE, text)

    def _write(self, name: str, text: str) -> None:
        """Write a whole file at once: a reader sees the old file or the new one."""
        target = self.file(name)
        partial = target.with_name(target.name + ".partial")
        partial.write_text(text, encoding="utf-8")
        os.replace(partial, target)

    def _append(self, name: str, record: dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False)
        with self.file(name).open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()

    # --- reading ---------------------------------------------------------

    def read_brief(self) -> dict[str, Any]:
        """The run's ``brief.json``; :class:`BridgeError` when there is no run."""
        path = self.file(self.BRIEF)
        if not path.is_file():
            raise BridgeError(f"no bridge run at {self.path}: start one with begin")
        brief = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(brief, dict):
            raise BridgeError(f"{path} is not a bridge run's brief")
        return brief

    def read_responses(self) -> list[RecordedResponse]:
        """The model's responses so far, in order."""
        return [RecordedResponse.from_record(r) for r in self._lines(self.RESPONSES)]

    def read_tool_cache(self) -> list[dict[str, Any]]:
        """The handler outcomes so far, in call order."""
        return self._lines(self.TOOL_CACHE)

    def read_pending(self) -> dict[str, Any] | None:
        """The last step's view, or ``None`` before the first step ended."""
        path = self.file(self.PENDING)
        if not path.is_file():
            return None
        view = json.loads(path.read_text(encoding="utf-8"))
        return view if isinstance(view, dict) else None

    def _lines(self, name: str) -> list[dict[str, Any]]:
        """A JSON-lines file's records; an unfinished last line is dropped.

        Lines end at ``"\\n"`` only: a JSON string may hold U+2028 or U+0085
        (``json.dumps`` leaves them as they are), which ``splitlines()``
        splits at. The file is split as bytes, so a line cut inside a
        multi-byte character is dropped with its line, not met as a decode
        error. A last segment with no ``"\\n"`` is an append that never
        finished: it is dropped, and the file rewritten without it, so the
        next append is not glued onto it, whether or not it parses.
        """
        path = self.file(name)
        if not path.is_file():
            return []
        segments = path.read_bytes().split(b"\n")
        tail = segments.pop()  # b"" when the file ends with "\n"
        if tail.strip():
            logger.warning(
                "bridge: dropped the unfinished last line of %s (an append cut "
                "off: that record was never saved)",
                path,
            )
            self._write(name, b"".join(s + b"\n" for s in segments).decode("utf-8"))
        out: list[dict[str, Any]] = []
        for n, raw in enumerate(segments, 1):
            line = raw.decode("utf-8")
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise BridgeError(f"{path} line {n} is not a JSON object")
            out.append(record)
        return out
