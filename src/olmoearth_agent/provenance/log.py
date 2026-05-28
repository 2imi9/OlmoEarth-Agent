# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Append-only provenance log for an agent run.

One :class:`~olmoearth_agent.types.ProvenanceManifest` entry per tool
call. Records the tool name, a hash of the arguments, and an
id-only summary of the result (no raw geometry, per operational rule
§3.1). Emits a JSON manifest and a runnable replay script.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any

from olmoearth_agent.types import ProvenanceManifest

# Keys we lift out of a tool result into the manifest's typed fields.
_ID_KEYS = ("prediction_id", "id", "model_id")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hash_args(arguments: dict[str, Any]) -> str:
    """Stable sha256 of the call arguments (sorted keys)."""
    blob = json.dumps(arguments, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _summarize_result(result: dict[str, Any]) -> dict[str, Any]:
    """Reduce a tool result to ids / status / counts — never raw geometry."""
    payload = result.get("result", result)
    if not isinstance(payload, dict):
        return {"ok": result.get("ok")}
    keep = {}
    for key in ("id", "name", "status", "total", "project_count", "model_id"):
        if key in payload:
            keep[key] = payload[key]
    keep["ok"] = result.get("ok")
    return keep


class ProvenanceLog:
    """Collects manifest entries for one agent run."""

    def __init__(self, run_id: str | None = None) -> None:
        self.run_id = run_id or str(uuid.uuid4())
        self.entries: list[ProvenanceManifest] = []

    def record_tool_call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        result: dict[str, Any],
    ) -> ProvenanceManifest:
        """Append one entry for a dispatched tool call."""
        payload = result.get("result", {})
        prediction_id = None
        model_id = None
        if isinstance(payload, dict):
            prediction_id = payload.get("prediction_id") or (
                payload.get("id") if tool_name.endswith("prediction") else None
            )
            model_id = payload.get("model_id")
        entry = ProvenanceManifest(
            run_id=self.run_id,
            timestamp=_utc_now(),
            api_call=tool_name,
            request_hash=_hash_args(arguments),
            response_summary=_summarize_result(result),
            prediction_id=prediction_id,
            model_id=model_id,
        )
        self.entries.append(entry)
        return entry

    def to_dict(self) -> dict[str, Any]:
        """Full manifest as a JSON-able dict."""
        return {
            "run_id": self.run_id,
            "entry_count": len(self.entries),
            "entries": [asdict(e) for e in self.entries],
        }

    def to_json(self, *, indent: int = 2) -> str:
        """Full manifest as a JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def replay_script(self) -> str:
        """Emit a runnable Python script that re-issues the recorded calls.

        The script re-dispatches each recorded tool call (by name + the
        original arguments are NOT stored verbatim — only their hash — so
        the script is a skeleton that documents the sequence and leaves
        argument reconstruction to the operator). This is enough to audit
        *what* the run did and in what order.
        """
        lines = [
            "# Auto-generated replay skeleton — OlmoEarth Agent provenance",
            f"# run_id: {self.run_id}",
            f"# {len(self.entries)} tool call(s)",
            "#",
            "# Each step lists the tool, the argument hash, and the result",
            "# summary. Re-run by dispatching these tools in order through a",
            "# fresh ToolRegistry (see olmoearth_agent.harness.LeadAgent).",
            "",
        ]
        for i, e in enumerate(self.entries, 1):
            ids = e.prediction_id or e.model_id or "-"
            lines.append(
                f"# {i:>2}. {e.api_call}  "
                f"args_sha256={e.request_hash[:12]}  "
                f"result={e.response_summary}  ids={ids}"
            )
        return "\n".join(lines) + "\n"
