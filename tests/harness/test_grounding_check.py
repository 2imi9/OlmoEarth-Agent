# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""The number check in the loop, the CLI and the web bridge.

exp86 round 3's one genuine fault (B8/cluster run 1): the answer stated
"5,565+ windows" between the budget cut and the median, a count no tool had
returned. A scripted model calls a tool that returns the scores' figures,
drafts that answer, and rewrites it when the harness asks. A number the
rewrite still states is shown marked ``[unverified: numbers]``.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

import pytest

from olmoearth_agent.harness.agent import (
    CHECK_ANSWER_ENV,
    CHECK_NUMBERS_ENV,
    TURN_CAP_PROMPT,
    LeadAgent,
    grounding_prompt,
)
from olmoearth_agent.llm.presets import DEFAULT_AGENT_MODE, REVISION_MODE
from olmoearth_agent.llm.types import ChatResponse, Message, ToolCall, ToolSpec
from olmoearth_agent.tools.registry import RegisteredTool, ToolContext, ToolRegistry

_EMPTY_SCHEMA = {"type": "object", "properties": {}, "required": []}
_SUMMARY = {
    "n_windows": 16384,
    "listed": 819,
    "margin_summary": {"margin_at_budget_cut": 0.994, "median_margin": 4.468},
}
_GROUNDED = "819 of 16,384 windows are listed; the cut is at 0.994, the median 4.468."
_DRAFT = (
    "819 windows are listed, and 5,565+ windows lie between the cut and the median."
)
_REVISION = "819 windows are listed; the cut is at 0.994 and the median at 4.468."
_STILL = "819 windows are listed, and 7,373 windows lie between the cut and the median."
_MARK = " [unverified: numbers]"


def _answer(content: str | None, thinking: str | None = None) -> ChatResponse:
    return ChatResponse(
        content=content, tool_calls=[], thinking=thinking, finish_reason="stop"
    )


def _summary_call(n: int = 1) -> ChatResponse:
    call = ToolCall(id=f"c{n}", name="margin_summary", arguments={})
    return ChatResponse(content=None, tool_calls=[call], finish_reason="tool_calls")


class _Scripted:
    """Returns scripted responses in order; records each call's messages and tools."""

    def __init__(self, responses: Iterable[ChatResponse]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[list[Message], Any]] = []
        self.modes: list[Any] = []

    async def chat(
        self, messages: list[Message], *, tools: Any = None, **kw: Any
    ) -> ChatResponse:
        self.calls.append((list(messages), tools))
        self.modes.append(kw.get("mode", DEFAULT_AGENT_MODE))
        return self.responses.pop(0)

    async def aclose(self) -> None:
        return None


class _ToolsUntilOffered:
    """Calls the tool while tools are offered; then gives ``answers`` in turn."""

    def __init__(self, answers: Iterable[str | None]) -> None:
        self.answers = list(answers)
        self.calls: list[tuple[list[Message], Any]] = []

    async def chat(
        self, messages: list[Message], *, tools: Any = None, **_kw: Any
    ) -> ChatResponse:
        self.calls.append((list(messages), tools))
        if tools:
            return _summary_call(len(self.calls))
        return _answer(self.answers.pop(0))

    async def aclose(self) -> None:
        return None


def _registry() -> ToolRegistry:
    async def summary(_args: dict[str, Any], _ctx: ToolContext) -> dict[str, Any]:
        return _SUMMARY

    registry = ToolRegistry()
    registry.register(
        RegisteredTool(ToolSpec("margin_summary", "margins", _EMPTY_SCHEMA), summary)
    )
    return registry


def _agent(llm: Any, **kw: Any) -> LeadAgent:
    return LeadAgent(llm, _registry(), studio=None, **kw)  # type: ignore[arg-type]


async def _events(agent: LeadAgent, brief: str = "which windows?", **kw: Any) -> Any:
    return [e async for e in agent.run_stream(brief, **kw)]


def _checks(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [e for e in events if e["type"] == "grounding_check"]


@pytest.fixture(autouse=True)
def _check_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(CHECK_NUMBERS_ENV, raising=False)
    monkeypatch.delenv(CHECK_ANSWER_ENV, raising=False)


# --- the loop --------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_grounded_answer_is_shown_without_another_call() -> None:
    llm = _Scripted([_summary_call(), _answer(_GROUNDED)])
    events = await _events(_agent(llm))
    assert _checks(events) == []
    assert len(llm.calls) == 2
    final = events[-1]
    assert final["type"] == "final" and final["content"] == _GROUNDED
    assert final["grounding_revised"] is False


@pytest.mark.asyncio
async def test_an_ungrounded_draft_is_rewritten_once() -> None:
    llm = _Scripted(
        [_summary_call(), _answer(_DRAFT), _answer(_REVISION, thinking="drop it")]
    )
    events = await _events(_agent(llm))
    types = [e["type"] for e in events]
    assert types == [
        "tool_call",
        "tool_result",
        "check",
        "grounding_check",
        "thinking",
        "final",
    ]
    assert events[2] == {
        "type": "check",
        "turn": 2,
        "check": "numbers",
        "violations": [{"check": "numbers", "text": _DRAFT, "detail": "5,565"}],
        "action": "revise",
        "draft": _DRAFT,
    }
    assert _checks(events) == [
        {
            "type": "grounding_check",
            "turn": 2,
            "unsupported": ["5,565"],
            "action": "revise",
            "draft": _DRAFT,
        }
    ]
    assert events[4]["text"] == "drop it" and events[4]["turn"] == 2
    final = events[-1]
    assert final["content"] == _REVISION
    assert final["grounding_revised"] is True and final["forced_by_turn_cap"] is False
    assert final["revised"] is True and final["marked"] == []
    assert final["turn"] == 2
    # One more call, without tools and with the rewrite's sampling: the
    # draft, then the harness's note (exp86's, when only numbers fired).
    assert len(llm.calls) == 3
    messages, tools = llm.calls[2]
    assert tools is None
    assert llm.modes == [DEFAULT_AGENT_MODE, DEFAULT_AGENT_MODE, REVISION_MODE]
    assert messages[-2].role == "assistant" and messages[-2].content == _DRAFT
    assert messages[-1].role == "user"
    assert messages[-1].content == grounding_prompt(["5,565"])
    assert "5,565" in (messages[-1].content or "")


@pytest.mark.asyncio
async def test_a_rewrite_still_ungrounded_is_shown_marked_and_not_asked_again() -> None:
    llm = _Scripted([_summary_call(), _answer(_DRAFT), _answer(_STILL)])
    events = await _events(_agent(llm))
    assert [(c["action"], c["unsupported"]) for c in _checks(events)] == [
        ("revise", ["5,565"]),
        ("shown", ["7,373"]),
    ]
    marked = [e for e in events if e["type"] == "check" and e["action"] == "marked"]
    assert [(e["check"], [v["detail"] for v in e["violations"]]) for e in marked] == [
        ("numbers", ["7,373"])
    ]
    assert len(llm.calls) == 3
    final = events[-1]
    # Shown, not deleted: the sentence carries the check's marker.
    assert final["content"] == _STILL + _MARK and final["grounding_revised"] is True
    assert final["marked"] == ["numbers"]
    assert [e["type"] for e in events].count("final") == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("empty", [None, "", "  \n"])
async def test_an_empty_rewrite_keeps_the_draft(empty: str | None) -> None:
    llm = _Scripted([_summary_call(), _answer(_DRAFT), _answer(empty)])
    events = await _events(_agent(llm))
    assert [(c["action"], c["unsupported"]) for c in _checks(events)] == [
        ("revise", ["5,565"]),
        ("shown", ["5,565"]),
    ]
    final = events[-1]
    assert final["content"] == _DRAFT + _MARK and final["grounding_revised"] is False
    assert final["revised"] is False and final["marked"] == ["numbers"]


@pytest.mark.asyncio
async def test_the_check_can_be_switched_off(monkeypatch: pytest.MonkeyPatch) -> None:
    llm = _Scripted([_summary_call(), _answer(_DRAFT)])
    events = await _events(_agent(llm, check_numbers=False))
    assert _checks(events) == [] and len(llm.calls) == 2
    assert events[-1]["content"] == _DRAFT
    assert events[-1]["grounding_revised"] is False

    monkeypatch.setenv(CHECK_NUMBERS_ENV, "0")
    llm = _Scripted([_summary_call(), _answer(_DRAFT)])
    events = await _events(_agent(llm))
    assert _checks(events) == [] and len(llm.calls) == 2


@pytest.mark.asyncio
async def test_the_answer_forced_at_the_turn_cap_is_checked() -> None:
    llm = _ToolsUntilOffered([_DRAFT, _REVISION])
    events = await _events(_agent(llm), max_turns=2)
    types = [e["type"] for e in events]
    assert types[-4:] == ["max_turns", "check", "grounding_check", "final"]
    assert events[-2]["turn"] == 3 and events[-2]["action"] == "revise"
    final = events[-1]
    assert final["content"] == _REVISION
    assert final["forced_by_turn_cap"] is True and final["grounding_revised"] is True
    assert len(llm.calls) == 4
    messages, tools = llm.calls[3]
    assert tools is None
    assert [m.role for m in messages[-3:]] == ["user", "assistant", "user"]
    assert messages[-3].content == TURN_CAP_PROMPT
    assert messages[-2].content == _DRAFT


@pytest.mark.asyncio
async def test_the_turn_cap_fallback_is_the_harness_text_and_not_checked() -> None:
    """The fallback states the cap (12), which no tool returned; it is not a
    model answer, so it is not sent back."""
    llm = _ToolsUntilOffered([None])
    events = await _events(_agent(llm), max_turns=12)
    assert _checks(events) == []
    assert "turn cap of 12" in events[-1]["content"]
    assert len(llm.calls) == 13


@pytest.mark.asyncio
async def test_the_brief_and_the_users_history_are_sources() -> None:
    answer = "Plan 350 labels over 12 strata, as you said."
    history = [
        Message(role="user", content="use 12 strata"),
        Message(role="assistant", content="Noted: 12 strata."),
    ]
    llm = _Scripted([_answer(answer)])
    events = await _events(_agent(llm), "plan 350 labels", history=history)
    assert _checks(events) == [] and len(llm.calls) == 1


@pytest.mark.asyncio
async def test_an_earlier_answer_and_the_saved_preferences_are_not_sources() -> None:
    """A number the model derived in an earlier answer ("~23%", exp86's
    69/300) is not evidence for the next one, and neither is the memory
    block, which is part of the system prompt."""
    history = [
        Message(role="user", content="how wrong is it?"),
        Message(role="assistant", content="About 23% of windows are wrong."),
    ]
    llm = _Scripted([_answer("As before, about 23% are wrong."), _answer("As before.")])
    events = await _events(_agent(llm), "and now?", history=history)
    assert _checks(events)[0]["unsupported"] == ["23%"]

    llm = _Scripted([_answer("The grid is 16."), _answer("The grid is as saved.")])
    agent = _agent(llm, memory_block="Saved preferences: grid 16")
    events = await _events(agent, "which grid?")
    assert _checks(events)[0]["unsupported"] == ["16"]

    # A figure only in the system prompt (a tool description's, say) is not
    # evidence: exp86 round 1 quoted "51-70%" from one.
    llm = _Scripted([_answer("either side wins 51-70%"), _answer("either side wins")])
    agent = _agent(llm, system_prompt="Tools: one side wins ~51-70% of windows.")
    events = await _events(agent, "compare")
    assert _checks(events)[0]["unsupported"] == ["51", "70%"]
    assert events[-1]["content"] == "either side wins"


@pytest.mark.asyncio
async def test_a_spilled_results_numbers_are_still_sources(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """The model sees a compacted envelope; the check reads the full result."""
    monkeypatch.setenv("OLMOEARTH_TOOL_RESULT_SPILL_BYTES", "200")
    monkeypatch.setenv("OLMOEARTH_OUTPUT_ROOT", str(tmp_path))
    big = {"windows": list(range(1000, 1400)), "deep": {"n_boundary": 8823}}

    async def fetch(_args: dict[str, Any], _ctx: ToolContext) -> dict[str, Any]:
        return big

    registry = ToolRegistry()
    registry.register(RegisteredTool(ToolSpec("fetch", "f", _EMPTY_SCHEMA), fetch))
    call = ToolCall(id="c1", name="fetch", arguments={})
    llm = _Scripted(
        [
            ChatResponse(content=None, tool_calls=[call], finish_reason="tool_calls"),
            _answer("8,823 windows touch class boundaries; window 1399 is last."),
        ]
    )
    agent = LeadAgent(llm, registry, studio=None)  # type: ignore[arg-type]
    events = await _events(agent)
    tool_message = next(m for m in llm.calls[1][0] if m.role == "tool")
    assert "8823" not in (tool_message.content or "")
    assert _checks(events) == []


@pytest.mark.asyncio
async def test_run_collects_the_check() -> None:
    llm = _Scripted([_summary_call(), _answer(_DRAFT), _answer(_STILL)])
    result = await _agent(llm).run("which windows?")
    assert result.final_content == _STILL + _MARK
    assert result.grounding_revised is True and result.revised is True
    assert [c["action"] for c in result.grounding_checks] == ["revise", "shown"]
    assert [(c["check"], c["action"]) for c in result.checks] == [
        ("numbers", "revise"),
        ("numbers", "marked"),
    ]
    assert result.marked == ["numbers"]
    assert result.turns == 2


# --- the CLI and the web bridge --------------------------------------------


def _patch_cli(monkeypatch: pytest.MonkeyPatch, llm: Any) -> Any:
    from olmoearth_agent import cli

    class _Studio:
        @classmethod
        def from_env(cls) -> _Studio:
            return cls()

        async def aclose(self) -> None:
            return None

    class _Skills:
        def index(self) -> str:
            return ""

    monkeypatch.setattr(cli, "OlmoEarthLLM", lambda: llm)
    monkeypatch.setattr(cli, "StudioClient", _Studio)
    monkeypatch.setattr(cli, "build_default_registry", _registry)
    monkeypatch.setattr(cli, "SkillLoader", _Skills)
    monkeypatch.setattr(cli, "preferences_block", lambda: "")
    return cli


def test_the_cli_prints_the_rewrite(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    llm = _Scripted([_summary_call(), _answer(_DRAFT), _answer(_REVISION)])
    cli = _patch_cli(monkeypatch, llm)
    assert cli.main(["which windows?", "--show-trace"]) == 0
    out = capsys.readouterr()
    assert out.out.strip() == _REVISION
    assert "[numbers] in no tool result, rewrite asked: 5,565" in out.err
    assert "no tool returned" not in out.err


def test_the_cli_names_numbers_still_unsupported_on_stderr(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    llm = _Scripted([_summary_call(), _answer(_DRAFT), _answer(_STILL)])
    cli = _patch_cli(monkeypatch, llm)
    assert cli.main(["which windows?"]) == 0
    out = capsys.readouterr()
    assert out.out.strip() == _STILL + _MARK  # the sentence is marked
    assert "(numbers in the answer that no tool returned: 7,373)" in out.err
    assert "[numbers]" not in out.err  # the rewrite request is trace-only


def test_the_web_bridge_streams_the_check(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from olmoearth_agent import serve

    monkeypatch.setattr(serve, "preferences_block", lambda: "")
    with TestClient(serve.app) as client:
        serve.app.state.llm = _Scripted(
            [_summary_call(), _answer(_DRAFT), _answer(_STILL)]
        )
        serve.app.state.registry = _registry()
        serve.app.state.skill_index = ""
        resp = client.post(
            "/api/run",
            json={"brief": "which windows?"},
            headers={"X-Olmoearth-Key": "k"},
        )
    assert resp.status_code == 200
    events = [
        json.loads(line[len("data: ") :])
        for line in resp.text.splitlines()
        if line.startswith("data: ")
    ]
    assert [e["type"] for e in events] == [
        "tool_call",
        "tool_result",
        "check",
        "grounding_check",
        "check",
        "grounding_check",
        "final",
        "done",
    ]
    assert [(e["action"], e["unsupported"]) for e in _checks(events)] == [
        ("revise", ["5,565"]),
        ("shown", ["7,373"]),
    ]
    assert events[-2]["content"] == _STILL + _MARK
    assert events[-2]["grounding_revised"] is True
