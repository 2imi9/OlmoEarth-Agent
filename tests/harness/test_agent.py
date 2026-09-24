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
        _FakeLLM([]),
        reg,
        studio=None,
        local=True,  # type: ignore[arg-type]
    )
    cloud_agent = LeadAgent(
        _FakeLLM([]),
        reg,
        studio=None,
        local=False,  # type: ignore[arg-type]
    )
    assert LOCAL_BUDGET_CLAUSE in local_agent.system_prompt
    assert "limited output budget" in local_agent.system_prompt
    assert LOCAL_BUDGET_CLAUSE not in cloud_agent.system_prompt


def test_default_prompt_has_run_order_and_guards() -> None:
    from olmoearth_agent.harness.agent import DEFAULT_SYSTEM_PROMPT as p

    # The ordered recipe + key dependency guards are present, so the model
    # discovers a model_id before submitting and polls before fetching.
    assert "Typical order" in p
    assert "olmoearth_load_context" in p
    assert "olmoearth_request_aoi" in p
    assert "project_id AND area_id AND model_id" in p
    assert "poll until status is completed before" in p
    assert "Do not repeat a tool call that already succeeded" in p


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
async def test_agent_hits_max_turns_and_still_answers() -> None:
    # Returns a tool call on every turn that offers tools -> never stops on its
    # own; the call after the cap offers none, and the model answers.
    loop_response = ChatResponse(
        content=None,
        tool_calls=[ToolCall(id="c", name="echo", arguments={})],
        finish_reason="tool_calls",
    )
    answer = ChatResponse(content="partial answer", tool_calls=[], finish_reason="stop")
    llm = _FakeLLM([loop_response] * 3 + [answer])
    agent = LeadAgent(
        llm,  # type: ignore[arg-type]
        _registry_with_echo(),
        studio=None,  # type: ignore[arg-type]
    )
    result = await agent.run("loop forever", max_turns=3)
    assert result.hit_max_turns is True
    assert result.final_content == "partial answer"
    assert result.turns == 4 and llm.turns == 4
    assert len(result.tool_calls) == 3


class _RecordingLLM(_FakeLLM):
    """A scripted LLM that records the tool names offered on every turn."""

    def __init__(self, responses: Iterable[ChatResponse]) -> None:
        super().__init__(responses)
        self.offered: list[list[str]] = []

    async def chat(
        self,
        messages: list[Message],
        *,
        tools: Any = None,
        **kw: Any,
    ) -> ChatResponse:
        self.offered.append([t.name for t in tools or []])
        return await super().chat(messages, tools=tools, **kw)


def _registry_with_deferred_skill() -> ToolRegistry:
    """echo (core), a deferred tool under olmoearth-rslearn, and load_skill."""
    from olmoearth_agent.skills.loader import SkillLoader
    from olmoearth_agent.tools.skill_tools import build_skill_tools

    registry = _registry_with_echo()

    async def fit(_args: dict[str, Any], _ctx: ToolContext) -> dict[str, Any]:
        return {"fitted": True}

    registry.register(
        RegisteredTool(
            spec=ToolSpec(name="rs_tool", description="rs", parameters=_EMPTY_SCHEMA),
            handler=fit,
        ),
        group="olmoearth-rslearn",
    )
    empty = SkillLoader(root="/nonexistent-skills-dir")
    registry.register_all(build_skill_tools(empty, registry=registry))
    return registry


@pytest.mark.asyncio
async def test_deferred_tools_arrive_the_turn_after_their_skill_loads() -> None:
    responses = [
        ChatResponse(
            content=None,
            tool_calls=[
                ToolCall(
                    id="c1",
                    name="olmoearth_load_skill",
                    arguments={"name": "olmoearth-rslearn"},
                )
            ],
            finish_reason="tool_calls",
        ),
        ChatResponse(
            content=None,
            tool_calls=[ToolCall(id="c2", name="rs_tool", arguments={})],
            finish_reason="tool_calls",
        ),
        ChatResponse(content="done", tool_calls=[], finish_reason="stop"),
    ]
    llm = _RecordingLLM(responses)
    agent = LeadAgent(
        llm,  # type: ignore[arg-type]
        _registry_with_deferred_skill(),
        studio=None,  # type: ignore[arg-type]
    )
    result = await agent.run("write me an rslearn model.yaml")
    assert "rs_tool" not in llm.offered[0]  # core only on the first turn
    assert "rs_tool" in llm.offered[1]  # loaded by olmoearth_load_skill
    assert result.tool_calls == [("olmoearth_load_skill", True), ("rs_tool", True)]


@pytest.mark.asyncio
async def test_a_forced_skill_has_its_deferred_tools_from_turn_one() -> None:
    llm = _RecordingLLM(
        [ChatResponse(content="ok", tool_calls=[], finish_reason="stop")]
    )
    agent = LeadAgent(
        llm,  # type: ignore[arg-type]
        _registry_with_deferred_skill(),
        studio=None,  # type: ignore[arg-type]
        forced_skill="rslearn",
    )
    await agent.run("configure it")
    assert "rs_tool" in llm.offered[0]
    assert "FORCED SKILL" in agent.system_prompt
    # A forced skill with no deferred group changes nothing.
    other = _RecordingLLM(
        [ChatResponse(content="ok", tool_calls=[], finish_reason="stop")]
    )
    plain = LeadAgent(
        other,  # type: ignore[arg-type]
        _registry_with_deferred_skill(),
        studio=None,  # type: ignore[arg-type]
        forced_skill="predict",
    )
    await plain.run("run it")
    assert "rs_tool" not in other.offered[0]
