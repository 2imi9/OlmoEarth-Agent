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
    assert envelope["preview"]


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
