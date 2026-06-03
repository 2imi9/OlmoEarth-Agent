# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Tests for the grounded verify-and-retry gate in ``LeadAgent.run_stream``."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pytest

from olmoearth_agent.harness.agent import LeadAgent
from olmoearth_agent.harness.state import ThreadState
from olmoearth_agent.llm.types import ChatResponse, Message, ToolCall, ToolSpec
from olmoearth_agent.tools.registry import RegisteredTool, ToolContext, ToolRegistry

_SCHEMA = {
    "type": "object",
    "properties": {"value": {"type": "string"}},
    "required": [],
}


class _FakeLLM:
    """Returns a scripted sequence of ChatResponses, ignoring the messages."""

    def __init__(self, responses: Iterable[ChatResponse]) -> None:
        self._responses = list(responses)

    async def chat(
        self, messages: list[Message], *, tools: Any = None, **_kw: Any
    ) -> ChatResponse:
        return self._responses.pop(0)


def _value_registry() -> ToolRegistry:
    """A ``calc`` tool whose verifier demands ``value == 'good'``."""

    async def calc(args: dict[str, Any], _ctx: ToolContext) -> dict[str, Any]:
        return {"value": args.get("value")}

    def verify(payload: dict[str, Any]) -> tuple[bool, str]:
        if payload.get("value") == "good":
            return (True, "")
        return (False, "value must be 'good'")

    registry = ToolRegistry()
    registry.register(
        RegisteredTool(ToolSpec("calc", "echo value", _SCHEMA), calc, verify=verify)
    )
    return registry


def _call(value: str, cid: str) -> ChatResponse:
    return ChatResponse(
        content=None,
        tool_calls=[ToolCall(id=cid, name="calc", arguments={"value": value})],
        finish_reason="tool_calls",
    )


def _final(text: str) -> ChatResponse:
    return ChatResponse(content=text, tool_calls=[], finish_reason="stop")


async def _collect(agent: LeadAgent, brief: str, **kw: Any) -> list[dict[str, Any]]:
    return [event async for event in agent.run_stream(brief, **kw)]


@pytest.mark.asyncio
async def test_verify_fail_triggers_reflection_then_model_corrects() -> None:
    state = ThreadState()
    agent = LeadAgent(
        _FakeLLM([_call("bad", "c1"), _call("good", "c2"), _final("done")]),  # type: ignore[arg-type]
        _value_registry(),
        studio=None,  # type: ignore[arg-type]
        state=state,
    )
    events = await _collect(agent, "go")
    verify_events = [e for e in events if e["type"] == "verify"]
    assert len(verify_events) == 1
    assert verify_events[0]["ok"] is False
    assert "good" in verify_events[0]["reason"]
    # one reflection recorded and fed back, then the model corrected and finished
    assert len(state.reflections) == 1
    assert events[-1]["type"] == "final"
    assert events[-1]["content"] == "done"
    assert [e["type"] for e in events].count("tool_call") == 2


@pytest.mark.asyncio
async def test_passing_verifier_emits_no_verify_event() -> None:
    state = ThreadState()
    agent = LeadAgent(
        _FakeLLM([_call("good", "c1"), _final("ok")]),  # type: ignore[arg-type]
        _value_registry(),
        studio=None,  # type: ignore[arg-type]
        state=state,
    )
    events = await _collect(agent, "go")
    assert not any(e["type"] == "verify" for e in events)
    assert state.reflections == []


@pytest.mark.asyncio
async def test_tool_without_verifier_is_unaffected() -> None:
    async def echo(args: dict[str, Any], _ctx: ToolContext) -> dict[str, Any]:
        return args

    registry = ToolRegistry()
    registry.register(RegisteredTool(ToolSpec("echo", "echo", _SCHEMA), echo))
    state = ThreadState()
    agent = LeadAgent(
        _FakeLLM(  # type: ignore[arg-type]
            [
                ChatResponse(
                    content=None,
                    tool_calls=[ToolCall(id="c", name="echo", arguments={})],
                    finish_reason="tool_calls",
                ),
                _final("ok"),
            ]
        ),
        registry,
        studio=None,  # type: ignore[arg-type]
        state=state,
    )
    events = await _collect(agent, "go")
    assert not any(e["type"] == "verify" for e in events)
    assert state.reflections == []


@pytest.mark.asyncio
async def test_retry_budget_caps_reflections() -> None:
    # The model never corrects; the per-run budget bounds reflections to 1.
    state = ThreadState()
    agent = LeadAgent(
        _FakeLLM([_call("bad", f"c{i}") for i in range(5)]),  # type: ignore[arg-type]
        _value_registry(),
        studio=None,  # type: ignore[arg-type]
        state=state,
    )
    events = await _collect(agent, "go", max_turns=3, max_verify_retries=1)
    verify_events = [e for e in events if e["type"] == "verify"]
    assert len(verify_events) == 3  # one failing result per turn
    assert len(state.reflections) == 1  # but only one reflection is fed back
    assert events[-1]["type"] == "max_turns"


@pytest.mark.asyncio
async def test_zero_budget_disables_reflection() -> None:
    state = ThreadState()
    agent = LeadAgent(
        _FakeLLM([_call("bad", "c1"), _final("done")]),  # type: ignore[arg-type]
        _value_registry(),
        studio=None,  # type: ignore[arg-type]
        state=state,
    )
    events = await _collect(agent, "go", max_verify_retries=0)
    # the failure is still observed, but no reflection is fed back
    assert any(e["type"] == "verify" for e in events)
    assert state.reflections == []
