# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Tests for ``LeadAgent.run_stream`` — the streaming event source."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pytest

from olmoearth_agent.harness.agent import LeadAgent
from olmoearth_agent.llm.types import ChatResponse, Message, ToolCall, ToolSpec
from olmoearth_agent.tools.registry import RegisteredTool, ToolContext, ToolRegistry

_EMPTY_SCHEMA = {"type": "object", "properties": {}, "required": []}


class _FakeLLM:
    """Returns a scripted sequence of ChatResponses."""

    def __init__(self, responses: Iterable[ChatResponse]) -> None:
        self._responses = list(responses)

    async def chat(
        self, messages: list[Message], *, tools: Any = None, **_kw: Any
    ) -> ChatResponse:
        return self._responses.pop(0)


def _echo_registry() -> ToolRegistry:
    async def echo(args: dict[str, Any], _ctx: ToolContext) -> dict[str, Any]:
        return args

    registry = ToolRegistry()
    registry.register(RegisteredTool(ToolSpec("echo", "echo", _EMPTY_SCHEMA), echo))
    return registry


async def _collect(agent: LeadAgent, brief: str, **kw: Any) -> list[dict[str, Any]]:
    return [event async for event in agent.run_stream(brief, **kw)]


@pytest.mark.asyncio
async def test_stream_emits_thinking_call_result_final_in_order() -> None:
    responses = [
        ChatResponse(
            content=None,
            tool_calls=[ToolCall(id="c1", name="echo", arguments={"x": 1})],
            thinking="I should echo first.",
            finish_reason="tool_calls",
        ),
        ChatResponse(content="all done", tool_calls=[], finish_reason="stop"),
    ]
    agent = LeadAgent(
        _FakeLLM(responses),  # type: ignore[arg-type]
        _echo_registry(),
        studio=None,  # type: ignore[arg-type]
    )
    events = await _collect(agent, "do it")
    assert [e["type"] for e in events] == [
        "thinking",
        "tool_call",
        "tool_result",
        "final",
    ]
    assert events[0]["text"] == "I should echo first."
    assert events[1]["name"] == "echo"
    assert events[1]["arguments"] == {"x": 1}
    assert events[2]["ok"] is True
    assert events[2]["result"]["result"] == {"x": 1}
    assert events[3]["content"] == "all done"


@pytest.mark.asyncio
async def test_stream_no_tools_emits_only_final() -> None:
    agent = LeadAgent(
        _FakeLLM(  # type: ignore[arg-type]
            [ChatResponse(content="hi", tool_calls=[], finish_reason="stop")]
        ),
        _echo_registry(),
        studio=None,  # type: ignore[arg-type]
    )
    events = await _collect(agent, "hello")
    assert [e["type"] for e in events] == ["final"]
    assert events[0]["content"] == "hi"


@pytest.mark.asyncio
async def test_stream_skips_thinking_when_absent() -> None:
    responses = [
        ChatResponse(
            content=None,
            tool_calls=[ToolCall(id="c1", name="echo", arguments={})],
            finish_reason="tool_calls",
        ),
        ChatResponse(content="done", tool_calls=[], finish_reason="stop"),
    ]
    agent = LeadAgent(
        _FakeLLM(responses),  # type: ignore[arg-type]
        _echo_registry(),
        studio=None,  # type: ignore[arg-type]
    )
    events = await _collect(agent, "go")
    assert "thinking" not in [e["type"] for e in events]


@pytest.mark.asyncio
async def test_stream_hits_max_turns() -> None:
    loop_response = ChatResponse(
        content=None,
        tool_calls=[ToolCall(id="c", name="echo", arguments={})],
        finish_reason="tool_calls",
    )
    agent = LeadAgent(
        _FakeLLM([loop_response] * 5),  # type: ignore[arg-type]
        _echo_registry(),
        studio=None,  # type: ignore[arg-type]
    )
    events = await _collect(agent, "loop", max_turns=2)
    types = [e["type"] for e in events]
    assert events[-1] == {"type": "max_turns", "turns": 2}
    assert types.count("tool_call") == 2
    assert types.count("tool_result") == 2
