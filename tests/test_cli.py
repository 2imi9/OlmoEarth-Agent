# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Tests for the CLI entrypoint (arg parsing + assembly with mocks)."""

from __future__ import annotations

from typing import Any

import pytest

from olmoearth_agent import cli
from olmoearth_agent.harness.agent import AgentResult
from olmoearth_agent.llm.types import ChatResponse, Message, ToolCall, ToolSpec
from olmoearth_agent.tools.registry import RegisteredTool, ToolContext, ToolRegistry

_EMPTY = {"type": "object", "properties": {}, "required": []}


def test_parse_args_defaults() -> None:
    ns = cli._parse_args(["do a thing"])
    assert ns.brief == "do a thing"
    assert ns.max_turns == 8
    assert ns.show_trace is False


def test_parse_args_flags() -> None:
    ns = cli._parse_args(["task", "--max-turns", "3", "--show-trace"])
    assert ns.max_turns == 3
    assert ns.show_trace is True


class _FakeLLM:
    def __init__(self, responses: list[ChatResponse]) -> None:
        self._responses = list(responses)

    async def chat(
        self, messages: list[Message], *, tools: Any = None, **_kw: Any
    ) -> ChatResponse:
        return self._responses.pop(0)


class _FakeStudio:
    async def aclose(self) -> None: ...


@pytest.mark.asyncio
async def test_run_brief_with_injected_deps() -> None:
    registry = ToolRegistry()

    async def echo(args: dict[str, Any], _ctx: ToolContext) -> dict[str, Any]:
        return args

    registry.register(
        RegisteredTool(ToolSpec("echo", "echo", _EMPTY), echo)
    )
    llm = _FakeLLM(
        [
            ChatResponse(
                content=None,
                tool_calls=[ToolCall(id="c1", name="echo", arguments={"x": 1})],
                finish_reason="tool_calls",
            ),
            ChatResponse(content="done", tool_calls=[], finish_reason="stop"),
        ]
    )
    result = await cli.run_brief(
        "go",
        llm=llm,  # type: ignore[arg-type]
        studio=_FakeStudio(),  # type: ignore[arg-type]
        registry=registry,
        skill_index="",
    )
    assert result.final_content == "done"
    assert result.tool_calls == [("echo", True)]


def test_main_returns_0_and_prints_answer(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    async def fake_run(_brief: str, **_kw: Any) -> AgentResult:
        return AgentResult(final_content="the answer", turns=1, tool_calls=[])

    monkeypatch.setattr(cli, "run_brief", fake_run)
    code = cli.main(["how many projects?"])
    assert code == 0
    assert "the answer" in capsys.readouterr().out


def test_main_returns_1_on_no_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_run(_brief: str, **_kw: Any) -> AgentResult:
        return AgentResult(
            final_content=None, turns=8, tool_calls=[], hit_max_turns=True
        )

    monkeypatch.setattr(cli, "run_brief", fake_run)
    assert cli.main(["loop"]) == 1
