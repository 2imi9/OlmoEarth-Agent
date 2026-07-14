# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Spill oversized tool results to disk before they enter the LLM context.

Shippy's lesson: tool output belongs in a local JSON file, not piped inline —
large payloads blow context (and, there, pipe buffers). Our agent loop feeds
every tool result back to the model as a JSON message; one big
``olmoearth_fetch_results`` payload can eat most of a 16k-token local-model
context. :func:`compact_result_for_llm` bounds that: results over a threshold
are written whole to ``<workspace>/tool_results/`` and the model receives a
compact envelope (preview + structural sketch + the saved path) instead.

Only the *LLM-bound* serialization is compacted. The UI event stream and the
provenance log keep the full result, so rendering and audit are unaffected.

The threshold is ``OLMOEARTH_TOOL_RESULT_SPILL_BYTES`` (bytes of serialized
JSON; ``0`` disables spilling entirely).
"""

from __future__ import annotations

import json
import logging
import os
import re
import uuid
from pathlib import Path
from typing import Any

from olmoearth_agent.security.paths import workspace_root

logger = logging.getLogger(__name__)

#: Env var overriding the spill threshold (serialized bytes; 0 disables).
SPILL_BYTES_ENV = "OLMOEARTH_TOOL_RESULT_SPILL_BYTES"

#: Default threshold: ~5k tokens of JSON, sized so two oversized results
#: cannot consume a 16k-token local-model context on their own.
DEFAULT_SPILL_BYTES = 20_000

_PREVIEW_CHARS = 1_500
_SANITIZE = re.compile(r"[^A-Za-z0-9_-]+")


def spill_threshold() -> int:
    """The active spill threshold in bytes (env override, else default)."""
    raw = os.environ.get(SPILL_BYTES_ENV, "").strip()
    if not raw:
        return DEFAULT_SPILL_BYTES
    try:
        return int(raw)
    except ValueError:
        return DEFAULT_SPILL_BYTES


def compact_result_for_llm(tool_name: str, result: Any) -> str:
    """Serialize ``result`` for the LLM, spilling oversized payloads to disk.

    Under the threshold the result is returned as-is (exact prior behavior).
    Over it, the full JSON is written to a file under the confined workspace
    root and a compact envelope — preview, structural sketch, saved path, and
    recovery guidance — is returned instead. Never raises: if the write fails,
    the envelope simply carries no path (still bounded).
    """
    text = json.dumps(result)
    limit = spill_threshold()
    if limit <= 0 or len(text) <= limit:
        return text

    saved: str | None = None
    stem = _SANITIZE.sub("_", tool_name)[:60] or "tool"
    target_dir = workspace_root() / "tool_results"
    path = target_dir / f"{stem}_{uuid.uuid4().hex[:8]}.json"
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        saved = str(path)
    except OSError:  # disk trouble must not break the agent turn
        logger.warning("could not spill %s result to %s", tool_name, path, exc_info=True)

    envelope: dict[str, Any] = {
        "ok": result.get("ok") if isinstance(result, dict) else None,
        "truncated": True,
        "full_result_bytes": len(text),
        "saved_to": saved,
        "shape": _shape(result),
        "preview": text[:_PREVIEW_CHARS],
        "note": (
            "Result too large for the context window; this is a preview plus a "
            "structural sketch. The complete JSON is at 'saved_to' — reference "
            "that path in your answer or pass it to tools that accept file "
            "inputs. Do not repeat the call expecting the full payload inline."
            if saved
            else "Result too large for the context window and could not be "
            "written to disk; this preview and structural sketch are all "
            "that is available. Do not repeat the call."
        ),
    }
    return json.dumps(envelope)


def _spill_dir_hint() -> Path:
    """Where spilled results land (exposed for tests and docs)."""
    return workspace_root() / "tool_results"


def _shape(value: Any, depth: int = 0) -> Any:
    """A tiny structural sketch of a JSON value: keys, lengths, scalar types.

    Gives the model enough orientation (what fields exist, how many records)
    to reason about the spilled payload without re-reading it.
    """
    if depth >= 3:
        return "..."
    if isinstance(value, dict):
        return {k: _shape(v, depth + 1) for k, v in list(value.items())[:12]}
    if isinstance(value, list):
        return {
            "list_length": len(value),
            "first_item": _shape(value[0], depth + 1) if value else None,
        }
    if isinstance(value, str):
        return value if len(value) <= 40 else f"str[{len(value)} chars]"
    return value
