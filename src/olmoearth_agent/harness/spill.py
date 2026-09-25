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

The preview is the result's summary, not its first characters: its scalar
fields (counts, shares, ids) and its summary blocks, within a budget, and the
output contract whole (``facts``, ``must_state``, ``forbidden_claims``,
``evidence_scope``), so a spilled result still says what an answer must state.
The raw JSON prefix is the preview when the tool's result is not a dict (a
list, say) or has no such fields.
The note says the full result is saved and that the path is for the user who
asks for the file; it does not ask the model to put the path in its answer,
where a saved file is a claim about what was done (exp86 round 7, brief 8:
"the full list is saved in <path>" named a file that held no such list).

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

#: A result's output contract, copied whole into a spilled envelope.
CONTRACT_KEYS = ("facts", "must_state", "forbidden_claims", "evidence_scope")

#: Budget, in serialized characters, for a summary preview's other fields.
_SUMMARY_CHARS = 3_000

#: A string field longer than this is left out of a summary preview.
_SHORT_STR = 300

#: Nested blocks a summary preview keeps (by name) when they fit the budget.
_SUMMARY_BLOCKS = ("summary", "narration", "stats", "spatial", "ceiling", "dates")

#: The note on a spilled result that was saved.
SAVED_NOTE = (
    "Result too large for the context window. 'preview' holds its summary "
    "fields and 'shape' its structure; any facts, must_state, forbidden_claims "
    "and evidence_scope are copied whole. The full result is saved to a file "
    "('saved_to'): mention that path only if the user asks for the file, and do "
    "not describe its contents beyond what this preview shows. Do not repeat "
    "the call expecting the full payload inline."
)

#: The note on a spilled result that could not be saved.
UNSAVED_NOTE = (
    "Result too large for the context window and could not be written to disk; "
    "this preview and structural sketch are all that is available. Do not "
    "repeat the call."
)


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
        logger.warning(
            "could not spill %s result to %s", tool_name, path, exc_info=True
        )

    body = _body(result)
    summary = _summary_preview(body) if body is not None else None
    envelope: dict[str, Any] = {
        "ok": result.get("ok") if isinstance(result, dict) else None,
        "truncated": True,
        "full_result_bytes": len(text),
        "saved_to": saved,
        "shape": _shape(result),
        "preview": summary if summary else text[:_PREVIEW_CHARS],
        "note": SAVED_NOTE if saved else UNSAVED_NOTE,
    }
    if body is not None:
        for key in CONTRACT_KEYS:
            if key in body:
                envelope[key] = body[key]
    return json.dumps(envelope)


def _body(result: Any) -> dict[str, Any] | None:
    """The tool's own result when it is a dict: the dispatch envelope's
    ``result``, or a dict passed bare (a failed call's envelope, say).

    ``None`` when the tool's result is not a dict (a list, a string): its
    preview is then the raw JSON prefix, as before, not the envelope's
    ``ok`` alone.
    """
    if not isinstance(result, dict):
        return None
    if "result" in result:
        inner = result["result"]
        return inner if isinstance(inner, dict) else None
    return result


def _short(value: Any) -> bool:
    """A scalar, or a string short enough to preview whole."""
    if value is None or isinstance(value, (bool, int, float)):
        return True
    return isinstance(value, str) and len(value) <= _SHORT_STR


def _summary_preview(body: dict[str, Any]) -> dict[str, Any]:
    """The result's aggregate fields, within a budget: scalars and short strings
    first (counts, shares, ids), then summary blocks whose name says so.

    Empty when the result has no such field (a bare list of records, say); the
    caller then previews the raw JSON prefix.
    """
    out: dict[str, Any] = {}
    used = 0
    for key, value in body.items():
        if key in CONTRACT_KEYS or not _short(value):
            continue
        size = len(json.dumps({key: value}))
        if used + size > _SUMMARY_CHARS:
            continue
        out[key] = value
        used += size
    for key, value in body.items():
        if key in CONTRACT_KEYS or not isinstance(value, dict):
            continue
        if not any(word in key for word in _SUMMARY_BLOCKS):
            continue
        size = len(json.dumps({key: value}))
        if used + size <= _SUMMARY_CHARS:
            out[key] = value
            used += size
    return out


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
