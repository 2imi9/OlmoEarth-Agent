# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Unit tests for oversized-tool-result spilling (harness/spill.py)."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pytest

from olmoearth_agent.harness.agent import LeadAgent
from olmoearth_agent.harness.spill import (
    DEFAULT_SPILL_BYTES,
    SPILL_BYTES_ENV,
    compact_result_for_llm,
    spill_threshold,
)
from olmoearth_agent.llm.types import ChatResponse, Message, ToolCall, ToolSpec
from olmoearth_agent.security.paths import OUTPUT_ROOT_ENV
from olmoearth_agent.tools.registry import RegisteredTool, ToolContext, ToolRegistry

_EMPTY_SCHEMA = {"type": "object", "properties": {}, "required": []}


@pytest.fixture(autouse=True)
def _workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv(OUTPUT_ROOT_ENV, str(tmp_path))
    return tmp_path


def test_threshold_default_and_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(SPILL_BYTES_ENV, raising=False)
    assert spill_threshold() == DEFAULT_SPILL_BYTES
    monkeypatch.setenv(SPILL_BYTES_ENV, "123")
    assert spill_threshold() == 123
    monkeypatch.setenv(SPILL_BYTES_ENV, "not-a-number")
    assert spill_threshold() == DEFAULT_SPILL_BYTES


def test_small_result_passes_through_verbatim() -> None:
    result = {"ok": True, "result": {"x": 1}}
    assert compact_result_for_llm("echo", result) == json.dumps(result)


def test_large_result_is_spilled_to_workspace(tmp_path: Path) -> None:
    records = [{"id": i, "geom": "x" * 50} for i in range(1000)]
    result = {"ok": True, "result": {"records": records}}
    text = compact_result_for_llm("olmoearth_fetch_results", result)

    envelope = json.loads(text)
    assert envelope["truncated"] is True
    assert envelope["ok"] is True
    assert len(text) < len(json.dumps(result))
    # Full payload is on disk, inside the confined workspace root.
    saved = Path(envelope["saved_to"])
    assert saved.is_file()
    assert tmp_path.resolve() in saved.resolve().parents
    assert json.loads(saved.read_text(encoding="utf-8")) == result
    # The sketch tells the model how many records exist without inlining them.
    assert envelope["shape"]["result"]["records"]["list_length"] == 1000
    # No summary field to preview: the raw JSON prefix stands in.
    assert envelope["preview"] == json.dumps(result)[:1500]


def test_the_preview_is_the_results_summary_and_the_contract_travels_whole() -> None:
    """exp86: the preview was the first 1,500 characters of the JSON, whatever
    they held. It is now the aggregate fields, and the output contract (facts,
    must_state, forbidden_claims, evidence_scope) is copied whole."""
    rows = [{"window_index": i, "class_a": 0, "class_b": 1} for i in range(2000)]
    body = {
        "differing": rows,
        "n_windows": 16384,
        "n_differing": 3807,
        "share_differing": 0.232361,
        "listing_order": "the first differing windows in window order",
        "long_text": "x" * 5000,
        "margin_summary": {"median_margin": 4.47, "lowest_margin": 0.1},
        "class_changes": [{"class_a": 4, "class_b": 2, "n": 1409}],
        "facts": [{"id": "dominant_change", "sentence": "4 -> 2: 1,409 of 3,807"}],
        "must_state": ["No side is right without labels."],
        "forbidden_claims": [{"id": "winner_without_labels", "why": "no labels"}],
        "evidence_scope": "exp58 measured it elsewhere.",
    }
    envelope = json.loads(
        compact_result_for_llm("olmoearth_compare_review", {"ok": True, "result": body})
    )
    preview = envelope["preview"]
    assert preview["n_differing"] == 3807 and preview["share_differing"] == 0.232361
    assert preview["margin_summary"] == body["margin_summary"]
    assert "long_text" not in preview and "differing" not in preview
    for key in ("facts", "must_state", "forbidden_claims", "evidence_scope"):
        assert envelope[key] == body[key]
        assert key not in preview


def test_the_spill_note_does_not_ask_for_the_path_in_the_answer() -> None:
    """exp86 round 7: an answer cited a saved file for a list it did not hold.
    The path is for the user who asks for the file, not for the answer."""
    result = {"ok": True, "result": {"blob": "x" * 50_000, "n": 1}}
    envelope = json.loads(compact_result_for_llm("echo", result))
    note = envelope["note"]
    assert "reference that path in your answer" not in note
    assert "only if the user asks for the file" in note
    assert "do not describe its contents beyond what this preview shows" in note
    assert envelope["preview"] == {"n": 1}


@pytest.mark.parametrize(
    "body",
    [
        [{"id": i, "geom": "x" * 50} for i in range(1000)],
        "x" * 50_000,
    ],
    ids=["list", "string"],
)
def test_a_result_that_is_not_a_dict_previews_the_raw_json_prefix(body: Any) -> None:
    """exp87 review: a tool whose result is a list (or a string) got the
    dispatch envelope's ``ok`` as its whole preview. Its preview is the raw
    JSON prefix, as before the summary preview."""
    result = {"ok": True, "result": body}
    envelope = json.loads(compact_result_for_llm("echo", result))
    assert envelope["preview"] == json.dumps(result)[:1500]
    assert "facts" not in envelope and "must_state" not in envelope


def test_a_failed_tool_envelope_previews_its_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(SPILL_BYTES_ENV, "50")
    result = {"ok": False, "error": "ValueError: bad", "hint": "fix it"}
    envelope = json.loads(compact_result_for_llm("echo", result))
    assert envelope["preview"]["error"] == "ValueError: bad"
    assert envelope["ok"] is False


def test_zero_threshold_disables_spilling(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(SPILL_BYTES_ENV, "0")
    result = {"ok": True, "result": {"blob": "x" * 100_000}}
    assert compact_result_for_llm("echo", result) == json.dumps(result)


class _RecordingLLM:
    """Scripted responses; records the messages of every chat() call."""

    def __init__(self, responses: Iterable[ChatResponse]) -> None:
        self._responses = list(responses)
        self.seen: list[list[Message]] = []

    async def chat(
        self, messages: list[Message], *, tools: Any = None, **_kw: Any
    ) -> ChatResponse:
        self.seen.append(list(messages))
        return self._responses.pop(0)


@pytest.mark.asyncio
async def test_agent_loop_spills_for_llm_but_streams_full_result() -> None:
    big = {"records": [{"id": i, "geom": "y" * 50} for i in range(1000)]}

    async def fetch(_args: dict[str, Any], _ctx: ToolContext) -> dict[str, Any]:
        return big

    registry = ToolRegistry()
    registry.register(
        RegisteredTool(
            spec=ToolSpec(name="fetch", description="f", parameters=_EMPTY_SCHEMA),
            handler=fetch,
        )
    )
    llm = _RecordingLLM(
        [
            ChatResponse(
                content=None,
                tool_calls=[ToolCall(id="c1", name="fetch", arguments={})],
                finish_reason="tool_calls",
            ),
            ChatResponse(content="done", tool_calls=[], finish_reason="stop"),
        ]
    )
    agent = LeadAgent(llm, registry, studio=None)  # type: ignore[arg-type]

    events = [e async for e in agent.run_stream("get results")]
    tool_results = [e for e in events if e["type"] == "tool_result"]
    assert len(tool_results) == 1
    # The UI event keeps the full result (rendering is unaffected)...
    assert tool_results[0]["result"]["result"] == big
    # ...but the tool message the LLM sees on the next turn is the compact
    # envelope pointing at the spilled file.
    tool_msg = next(m for m in llm.seen[1] if m.role == "tool")
    envelope = json.loads(tool_msg.content or "")
    assert envelope["truncated"] is True
    assert Path(envelope["saved_to"]).is_file()
    assert len(tool_msg.content or "") < len(json.dumps(big))
