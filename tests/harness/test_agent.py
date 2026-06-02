# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Unit tests for the lead-agent loop with a fake LLM."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pytest

from olmoearth_agent.harness.agent import LOCAL_BUDGET_CLAUSE, LeadAgent
from olmoearth_agent.llm.types import ChatResponse, Message, ToolCall, ToolSpec
from olmoearth_agent.tools.registry import RegisteredTool, ToolContext, ToolRegistry

_EMPTY_SCHEMA = {"type": "object", "properties": {}, "required": []}


class _FakeLLM:
    """Returns a scripted sequence of ChatResponses."""

    def __init__(self, responses: Iterable[ChatResponse]) -> None:
        self._responses = list(responses)
        self.turns = 0

    async def chat(
        self,
        messages: list[Message],
        *,
        tools: Any = None,
        **_kw: Any,
    ) -> ChatResponse:
        self.turns += 1
        return self._responses.pop(0)


def _registry_with_echo() -> ToolRegistry:
    async def echo(args: dict[str, Any], _ctx: ToolContext) -> dict[str, Any]:
        return args

    registry = ToolRegistry()
    registry.register(
        RegisteredTool(
            spec=ToolSpec(name="echo", description="echo", parameters=_EMPTY_SCHEMA),
            handler=echo,
        )
    )
    return registry


@pytest.mark.asyncio
async def test_agent_calls_tool_then_answers() -> None:
    responses = [
        ChatResponse(
            content=None,
            tool_calls=[ToolCall(id="c1", name="echo", arguments={"x": 1})],
            finish_reason="tool_calls",
        ),
        ChatResponse(content="all done", tool_calls=[], finish_reason="stop"),
    ]
    agent = LeadAgent(
        _FakeLLM(responses),  # type: ignore[arg-type]
        _registry_with_echo(),
        studio=None,  # type: ignore[arg-type]
    )
    result = await agent.run("do the thing")
    assert result.final_content == "all done"
    assert result.turns == 2
    assert result.tool_calls == [("echo", True)]
    assert result.hit_max_turns is False


@pytest.mark.asyncio
async def test_agent_answers_without_tools() -> None:
    responses = [ChatResponse(content="hello", tool_calls=[], finish_reason="stop")]
    agent = LeadAgent(
        _FakeLLM(responses),  # type: ignore[arg-type]
        _registry_with_echo(),
        studio=None,  # type: ignore[arg-type]
    )
    result = await agent.run("hi")
    assert result.final_content == "hello"
    assert result.turns == 1
    assert result.tool_calls == []


def test_local_flag_appends_budget_clause() -> None:
    reg = _registry_with_echo()
    local_agent = LeadAgent(
        _FakeLLM([]), reg, studio=None, local=True  # type: ignore[arg-type]
    )
    cloud_agent = LeadAgent(
        _FakeLLM([]), reg, studio=None, local=False  # type: ignore[arg-type]
    )
    assert LOCAL_BUDGET_CLAUSE in local_agent.system_prompt
    assert "limited output budget" in local_agent.system_prompt
    assert LOCAL_BUDGET_CLAUSE not in cloud_agent.system_prompt


@pytest.mark.asyncio
async def test_agent_records_provenance() -> None:
    responses = [
        ChatResponse(
            content=None,
            tool_calls=[ToolCall(id="c1", name="echo", arguments={"x": 1})],
            finish_reason="tool_calls",
        ),
        ChatResponse(content="done", tool_calls=[], finish_reason="stop"),
    ]
    agent = LeadAgent(
        _FakeLLM(responses),  # type: ignore[arg-type]
        _registry_with_echo(),
        studio=None,  # type: ignore[arg-type]
    )
    await agent.run("go")
    assert len(agent.state.provenance.entries) == 1
    assert agent.state.provenance.entries[0].api_call == "echo"


@pytest.mark.asyncio
async def test_agent_hits_max_turns() -> None:
    # Always returns a tool call -> never terminates on its own.
    loop_response = ChatResponse(
        content=None,
        tool_calls=[ToolCall(id="c", name="echo", arguments={})],
        finish_reason="tool_calls",
    )
    agent = LeadAgent(
        _FakeLLM([loop_response] * 5),  # type: ignore[arg-type]
        _registry_with_echo(),
        studio=None,  # type: ignore[arg-type]
    )
    result = await agent.run("loop forever", max_turns=3)
    assert result.hit_max_turns is True
    assert result.final_content is None
    assert result.turns == 3
    assert len(result.tool_calls) == 3
