# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""A bridge run's directory: what one step reads and writes.

- ``brief.json``: the brief, the history and the run's settings, written once
  by ``begin`` (the skill index and memory block are kept as they were then,
  as one ``olmoearth-agent`` run builds its prompt once).
- ``responses.jsonl``: the model's responses so far, one line per model call,
  each with the fingerprint of the request it answered.
- ``tool_cache.jsonl``: each tool handler invocation's outcome, in call order,
  with the model call whose response made it.
- ``events.jsonl``: the loop's events, as the last step's replay yielded them.
- ``pending.json``: what the loop waits for (a model call), or its final answer.
- ``provenance.json``: the run's provenance log, as the last step rebuilt it.

A step appends to the tool cache as each new handler returns; it writes the
response it was given, then the events, the provenance and ``pending.json``
last, only once it has stopped cleanly. A step that failed has saved no
response and left ``pending.json`` as it was; the tool calls it ran stay in
the cache as provisional records (``replay.ToolCache``). A step stopped
after saving the response but before writing ``pending.json`` is finished by
the next ``step`` (``session.step``).

The directory is made private to its owner (0700): its files hold the
account's names and the tools' whole results.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from olmoearth_agent.bridge.errors import BridgeError
from olmoearth_agent.bridge.scripted import RecordedResponse

__all__ = ["RunDir"]


class RunDir:
    """The files of one bridge run (see the module docstring)."""

    BRIEF = "brief.json"
    RESPONSES = "responses.jsonl"
    TOOL_CACHE = "tool_cache.jsonl"
    EVENTS = "events.jsonl"
    PENDING = "pending.json"
    PROVENANCE = "provenance.json"

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def file(self, name: str) -> Path:
        """The path of one of the run's files."""
        return self.path / name

    # --- writing ---------------------------------------------------------

    def create(self, brief: dict[str, Any]) -> None:
        """Make the run's directory with its ``brief.json``; it must be new or empty.

        A new directory is made readable by its owner only (mode 0700).
        """
        if self.path.exists() and (not self.path.is_dir() or any(self.path.iterdir())):
            raise BridgeError(
                f"{self.path} exists and is not an empty directory: begin needs a "
                "new run directory"
            )
        self.path.mkdir(mode=0o700, parents=True, exist_ok=True)
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
        path = self.file(name)
        if not path.is_file():
            return []
        out: list[dict[str, Any]] = []
        # Lines end at "\n" only: a JSON string may hold U+2028 or U+0085
        # (json.dumps leaves them as they are), which splitlines() splits at.
        for n, line in enumerate(path.read_text(encoding="utf-8").split("\n"), 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise BridgeError(f"{path} line {n} is not a JSON object")
            out.append(record)
        return out
