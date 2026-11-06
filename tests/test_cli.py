# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
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

    registry.register(RegisteredTool(ToolSpec("echo", "echo", _EMPTY), echo))
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


_FIGURE = "/w/design_scores_ab12_random_300_s0_figure.svg"
_CAPTION = "Which windows to label. (a) The 300 windows drawn by the random design."


@pytest.mark.asyncio
async def test_run_brief_keeps_each_figure_a_tool_wrote() -> None:
    registry = ToolRegistry()

    async def plan(_args: dict[str, Any], _ctx: ToolContext) -> dict[str, Any]:
        return {"n_drawn": 300, "figure_path": _FIGURE, "figure_caption": _CAPTION}

    async def echo(args: dict[str, Any], _ctx: ToolContext) -> dict[str, Any]:
        return args

    registry.register(RegisteredTool(ToolSpec("plan", "plan", _EMPTY), plan))
    registry.register(RegisteredTool(ToolSpec("echo", "echo", _EMPTY), echo))
    llm = _FakeLLM(
        [
            ChatResponse(
                content=None,
                tool_calls=[
                    ToolCall(id="c1", name="plan", arguments={}),
                    ToolCall(id="c2", name="echo", arguments={"x": 1}),
                ],
                finish_reason="tool_calls",
            ),
            ChatResponse(content="done", tool_calls=[], finish_reason="stop"),
        ]
    )
    figures: dict[str, str] = {}
    result = await cli.run_brief(
        "go",
        llm=llm,  # type: ignore[arg-type]
        studio=_FakeStudio(),  # type: ignore[arg-type]
        registry=registry,
        skill_index="",
        figures=figures,
    )
    assert result.final_content == "done"
    assert result.tool_calls == [("plan", True), ("echo", True)]
    assert figures == {_FIGURE: _CAPTION}  # the echo wrote none


def test_figure_of_reads_only_a_successful_tool_result_naming_a_figure() -> None:
    ok = {"ok": True, "result": {"figure_path": _FIGURE, "figure_caption": _CAPTION}}
    event = {"type": "tool_result", "ok": True, "result": ok}
    assert cli.figure_of(event) == (_FIGURE, _CAPTION)
    bare = {"ok": True, "result": {"figure_path": _FIGURE}}
    event = {"type": "tool_result", "ok": True, "result": bare}
    assert cli.figure_of(event) == (_FIGURE, "")
    failed = {"ok": False, "error": "ValueError: no scores"}
    for none in (
        {"type": "tool_result", "ok": False, "result": failed},
        {"type": "tool_result", "ok": True, "result": {"ok": True, "result": [1]}},
        {"type": "tool_call", "name": "plan", "arguments": {}},
        {"type": "final", "content": _FIGURE},
    ):
        assert cli.figure_of(none) is None, none


def test_main_prints_each_figure_after_the_answer(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    async def fake_run(
        _brief: str, *, figures: dict[str, str], **_kw: Any
    ) -> AgentResult:
        figures[_FIGURE] = _CAPTION
        return AgentResult(
            final_content="Label these 300 windows.",
            turns=2,
            tool_calls=[("olmoearth_plan_label_sample", True)],
        )

    monkeypatch.setattr(cli, "run_brief", fake_run)
    assert cli.main(["how should I spend 300 labels?"]) == 0
    assert capsys.readouterr().out.splitlines() == [
        "Label these 300 windows.",
        f"figure: {_FIGURE}",
        _CAPTION,
    ]


def test_main_returns_0_and_prints_answer(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    async def fake_run(_brief: str, **_kw: Any) -> AgentResult:
        return AgentResult(final_content="the answer", turns=1, tool_calls=[])

    monkeypatch.setattr(cli, "run_brief", fake_run)
    code = cli.main(["how many projects?"])
    assert code == 0
    assert "the answer" in capsys.readouterr().out


def test_main_prints_an_answer_forced_by_the_turn_cap(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    async def fake_run(_brief: str, **_kw: Any) -> AgentResult:
        return AgentResult(
            final_content="what I have", turns=9, tool_calls=[], hit_max_turns=True
        )

    monkeypatch.setattr(cli, "run_brief", fake_run)
    code = cli.main(["loop"])
    captured = capsys.readouterr()
    assert code == 0
    assert captured.out.strip() == "what I have"
    assert "turn cap (8)" in captured.err


def test_main_returns_1_on_no_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_run(_brief: str, **_kw: Any) -> AgentResult:
        return AgentResult(
            final_content=None, turns=8, tool_calls=[], hit_max_turns=True
        )

    monkeypatch.setattr(cli, "run_brief", fake_run)
    assert cli.main(["loop"]) == 1
