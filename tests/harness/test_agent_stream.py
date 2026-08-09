# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Tests for ``LeadAgent.run_stream``: the streaming event source."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pytest

from olmoearth_agent.harness.agent import LeadAgent
from olmoearth_agent.harness.response_policy import ResponsePolicy
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


@pytest.mark.asyncio
async def test_stream_seeds_history_before_brief() -> None:
    captured: dict[str, Any] = {}

    class _CapturingLLM:
        async def chat(
            self, messages: list[Message], *, tools: Any = None, **_kw: Any
        ) -> ChatResponse:
            captured["messages"] = list(messages)
            return ChatResponse(content="ok", tool_calls=[], finish_reason="stop")

    history = [
        Message(role="user", content="How many projects do I have?"),
        Message(role="assistant", content="You have 5."),
    ]
    agent = LeadAgent(
        _CapturingLLM(),  # type: ignore[arg-type]
        _echo_registry(),
        studio=None,  # type: ignore[arg-type]
    )
    await _collect(agent, "Which relate to water quality?", history=history)

    seen = [(m.role, m.content) for m in captured["messages"]]
    assert seen[0][0] == "system"
    assert seen[1] == ("user", "How many projects do I have?")
    assert seen[2] == ("assistant", "You have 5.")
    assert seen[3] == ("user", "Which relate to water quality?")


class _CapturingLLM:
    """Captures the messages of the first ``chat`` call, then answers."""

    def __init__(self) -> None:
        self.messages: list[Message] = []

    async def chat(
        self, messages: list[Message], *, tools: Any = None, **_kw: Any
    ) -> ChatResponse:
        self.messages = list(messages)
        return ChatResponse(content="ok", tool_calls=[], finish_reason="stop")


@pytest.mark.asyncio
async def test_forced_skill_pins_run_via_system_prompt() -> None:
    llm = _CapturingLLM()
    agent = LeadAgent(
        llm,  # type: ignore[arg-type]
        _echo_registry(),
        studio=None,  # type: ignore[arg-type]
        forced_skill="change-detection",
    )
    await _collect(agent, "did forest cover decline?")
    system = llm.messages[0]
    assert system.role == "system"
    assert "FORCED SKILL" in system.content
    assert "change-detection" in system.content


@pytest.mark.asyncio
async def test_no_forced_skill_leaves_prompt_unpinned() -> None:
    llm = _CapturingLLM()
    agent = LeadAgent(
        llm,  # type: ignore[arg-type]
        _echo_registry(),
        studio=None,  # type: ignore[arg-type]
    )
    await _collect(agent, "hello")
    assert "FORCED SKILL" not in llm.messages[0].content


class _PolicyLLM:
    """Scripted LLM that records every policy-sensitive chat argument."""

    def __init__(self, responses: Iterable[ChatResponse]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def chat(
        self, messages: list[Message], *, tools: Any = None, **kwargs: Any
    ) -> ChatResponse:
        self.calls.append({"messages": list(messages), "tools": tools, **kwargs})
        return self.responses.pop(0)


@pytest.mark.asyncio
async def test_concise_policy_appends_contract_and_bounds_turn() -> None:
    llm = _PolicyLLM(
        [ChatResponse(content="Prediction p1 completed.", finish_reason="stop")]
    )
    policy = ResponsePolicy(concise_max_words=80, concise_max_tokens=2048)
    agent = LeadAgent(
        llm,  # type: ignore[arg-type]
        _echo_registry(),
        studio=None,  # type: ignore[arg-type]
        response_policy=policy,
    )

    events = await _collect(agent, "Check prediction p1")

    assert events[-1]["content"] == "Prediction p1 completed."
    assert llm.calls[0]["max_tokens"] == 2048
    system = llm.calls[0]["messages"][0].content
    assert "RESPONSE CONTRACT (concise default)" in system
    assert "80 words" in system


@pytest.mark.asyncio
async def test_explicit_detail_gets_larger_budget_without_rewrite() -> None:
    llm = _PolicyLLM(
        [ChatResponse(content="A detailed but valid answer.", finish_reason="stop")]
    )
    policy = ResponsePolicy(concise_max_words=80, concise_max_tokens=2048)
    agent = LeadAgent(
        llm,  # type: ignore[arg-type]
        _echo_registry(),
        studio=None,  # type: ignore[arg-type]
        response_policy=policy,
    )

    await _collect(agent, "Give me a detailed report")

    assert len(llm.calls) == 1
    assert llm.calls[0]["max_tokens"] == 8192
    system = llm.calls[0]["messages"][0].content
    assert "RESPONSE CONTRACT (detailed" in system


@pytest.mark.asyncio
async def test_overlong_final_gets_one_no_tools_editor_pass() -> None:
    draft = (
        "Prediction p1 completed. Output results/p1.json. Extra explanatory "
        "words make this draft longer than the intentionally tiny test budget."
    )
    rewrite = "Prediction p1 completed. Output: results/p1.json."
    llm = _PolicyLLM(
        [
            ChatResponse(content=draft, finish_reason="stop"),
            ChatResponse(content=rewrite, finish_reason="stop"),
        ]
    )
    policy = ResponsePolicy(
        concise_max_words=8,
        concise_max_tokens=2048,
        repair_max_tokens=256,
    )
    agent = LeadAgent(
        llm,  # type: ignore[arg-type]
        _echo_registry(),
        studio=None,  # type: ignore[arg-type]
        response_policy=policy,
    )

    events = await _collect(agent, "Check prediction p1")

    assert events[-1] == {
        "type": "final",
        "turn": 1,
        "content": rewrite,
        "compacted": True,
    }
    assert len(llm.calls) == 2
    assert llm.calls[1]["tools"] is None
    assert llm.calls[1]["mode"] == "instruct_general"
    assert llm.calls[1]["max_tokens"] == 256
    assert llm.calls[1]["preserve_thinking"] is False


@pytest.mark.asyncio
async def test_failed_editor_keeps_completed_answer() -> None:
    draft = "one two three four five six seven eight nine ten eleven twelve"

    class _FailingEditor(_PolicyLLM):
        async def chat(
            self, messages: list[Message], *, tools: Any = None, **kwargs: Any
        ) -> ChatResponse:
            if self.calls:
                raise RuntimeError("editor unavailable")
            return await super().chat(messages, tools=tools, **kwargs)

    llm = _FailingEditor([ChatResponse(content=draft, finish_reason="stop")])
    agent = LeadAgent(
        llm,  # type: ignore[arg-type]
        _echo_registry(),
        studio=None,  # type: ignore[arg-type]
        response_policy=ResponsePolicy(concise_max_words=8),
    )

    events = await _collect(agent, "Check prediction p1")

    assert events[-1]["content"] == draft
    assert "compacted" not in events[-1]


@pytest.mark.asyncio
async def test_editor_that_drops_identifier_keeps_completed_answer() -> None:
    draft = (
        "Prediction pred-42 completed and produced many extra explanatory words "
        "that make this response exceed its intentionally tiny test budget."
    )
    llm = _PolicyLLM(
        [
            ChatResponse(content=draft, finish_reason="stop"),
            ChatResponse(content="Prediction completed.", finish_reason="stop"),
        ]
    )
    agent = LeadAgent(
        llm,  # type: ignore[arg-type]
        _echo_registry(),
        studio=None,  # type: ignore[arg-type]
        response_policy=ResponsePolicy(concise_max_words=8),
    )

    events = await _collect(agent, "Check prediction pred-42")

    assert events[-1]["content"] == draft
    assert "compacted" not in events[-1]
