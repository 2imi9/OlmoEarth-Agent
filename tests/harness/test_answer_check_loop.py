# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""The answer checks in the loop, the CLI and the web bridge.

A scripted comparison tool returns exp86 B7's facts (1,247 of 1,570
differing windows flip 0 -> 1); the scripted model drafts the round 6
answer that read the direction backwards and claimed a saved list, and
rewrites it when the harness asks, once for every check.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

import pytest

from olmoearth_agent.harness.agent import (
    CHECK_ANSWER_ENV,
    CHECK_NUMBERS_ENV,
    LeadAgent,
    revision_prompt,
)
from olmoearth_agent.harness.checks import MARK
from olmoearth_agent.llm.presets import DEFAULT_AGENT_MODE, REVISION_MODE
from olmoearth_agent.llm.types import ChatResponse, Message, ToolCall, ToolSpec
from olmoearth_agent.tools.registry import RegisteredTool, ToolContext, ToolRegistry

_EMPTY_SCHEMA = {"type": "object", "properties": {}, "required": []}
_FACT = {
    "id": "dominant_change",
    "from_class": 0,
    "to_class": 1,
    "n": 1247,
    "share": 0.794268,
    "sentence": "1,247 of the 1,570 differing windows change 0 -> 1.",
}
_COMPARE = {
    "n_differing": 1570,
    "n_differing_listed": 30,
    "classes": {"0": "not water", "1": "water"},
    "facts": [_FACT],
}
_BACKWARDS = "Most differing windows flip class 1 -> 0."
_SAVED = "The full differing-window list was saved."
_FIXED = "Most differing windows flip 0 -> 1: 1,247 of the 1,570."


def _answer(content: str | None) -> ChatResponse:
    return ChatResponse(content=content, tool_calls=[], finish_reason="stop")


def _compare_call() -> ChatResponse:
    call = ToolCall(id="c1", name="compare", arguments={})
    return ChatResponse(content=None, tool_calls=[call], finish_reason="tool_calls")


class _Scripted:
    """Returns scripted responses in order; records messages, tools and mode."""

    def __init__(self, responses: Iterable[ChatResponse]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[list[Message], Any, Any]] = []

    async def chat(
        self, messages: list[Message], *, tools: Any = None, **kw: Any
    ) -> ChatResponse:
        self.calls.append((list(messages), tools, kw.get("mode", DEFAULT_AGENT_MODE)))
        return self.responses.pop(0)

    async def aclose(self) -> None:
        return None


def _registry() -> ToolRegistry:
    async def compare(_args: dict[str, Any], _ctx: ToolContext) -> dict[str, Any]:
        return _COMPARE

    registry = ToolRegistry()
    registry.register(
        RegisteredTool(ToolSpec("compare", "compare two maps", _EMPTY_SCHEMA), compare)
    )
    return registry


def _agent(llm: Any, **kw: Any) -> LeadAgent:
    return LeadAgent(llm, _registry(), studio=None, **kw)  # type: ignore[arg-type]


async def _events(agent: LeadAgent, brief: str = "compare A and B") -> Any:
    return [e async for e in agent.run_stream(brief)]


def _check_events(events: list[dict[str, Any]]) -> list[tuple[str, str, int]]:
    return [
        (e["check"], e["action"], len(e["violations"]))
        for e in events
        if e["type"] == "check"
    ]


@pytest.fixture(autouse=True)
def _checks_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(CHECK_NUMBERS_ENV, raising=False)
    monkeypatch.delenv(CHECK_ANSWER_ENV, raising=False)


@pytest.mark.asyncio
async def test_every_violation_goes_into_one_rewrite() -> None:
    draft = f"{_BACKWARDS} {_SAVED} About 1,111 of them sit together."
    llm = _Scripted([_compare_call(), _answer(draft), _answer(_FIXED)])
    events = await _events(_agent(llm))
    assert [e["type"] for e in events] == [
        "tool_call",
        "tool_result",
        "check",
        "check",
        "check",
        "grounding_check",
        "final",
    ]
    assert _check_events(events) == [
        ("numbers", "revise", 1),
        ("direction", "revise", 1),
        ("actions", "revise", 1),
    ]
    direction = events[3]
    assert direction["violations"][0]["text"] == _BACKWARDS
    assert direction["draft"] == draft and direction["turn"] == 2
    assert events[5]["unsupported"] == ["1,111"]
    # One call more, without tools, with the rewrite's sampling.
    assert len(llm.calls) == 3
    messages, tools, mode = llm.calls[2]
    assert tools is None and mode == REVISION_MODE
    assert messages[-2].role == "assistant" and messages[-2].content == draft
    note = messages[-1].content or ""
    assert note == revision_prompt(
        {
            c: e["violations"]
            for e in events
            if e["type"] == "check"
            for c in [e["check"]]
        }
    )
    for piece in ("1,111", _BACKWARDS, _SAVED, "Call no tools"):
        assert piece in note
    final = events[-1]
    assert final["content"] == _FIXED
    assert final["revised"] is True and final["grounding_revised"] is True
    assert final["marked"] == []


@pytest.mark.asyncio
async def test_a_sentence_still_flagged_is_marked_not_deleted() -> None:
    draft = f"{_BACKWARDS} {_SAVED}"
    still = f"Maps differ on 1,570 windows. {_BACKWARDS}"
    llm = _Scripted([_compare_call(), _answer(draft), _answer(still)])
    events = await _events(_agent(llm))
    assert _check_events(events) == [
        ("direction", "revise", 1),
        ("actions", "revise", 1),
        ("direction", "marked", 1),
    ]
    assert not [e for e in events if e["type"] == "grounding_check"]
    final = events[-1]
    assert final["content"] == f"{still} {MARK.format(check='direction')}"
    assert final["revised"] is True and final["grounding_revised"] is False
    assert final["marked"] == ["direction"]
    assert len(llm.calls) == 3  # asked once, never twice


@pytest.mark.asyncio
async def test_an_empty_rewrite_keeps_the_draft_marked() -> None:
    llm = _Scripted([_compare_call(), _answer(_BACKWARDS), _answer("  ")])
    events = await _events(_agent(llm))
    final = events[-1]
    assert final["content"] == f"{_BACKWARDS} [unverified: direction]"
    assert final["revised"] is False and final["marked"] == ["direction"]


@pytest.mark.asyncio
async def test_a_clean_answer_costs_no_call() -> None:
    llm = _Scripted([_compare_call(), _answer(_FIXED)])
    events = await _events(_agent(llm))
    assert _check_events(events) == [] and len(llm.calls) == 2
    assert events[-1]["revised"] is False and events[-1]["marked"] == []


@pytest.mark.asyncio
async def test_the_listing_claim_depends_on_the_surface() -> None:
    """On the web the tool's listing is on screen beside the answer."""
    claim = "The first 30 windows are listed above."
    llm = _Scripted([_compare_call(), _answer(claim), _answer(claim)])
    events = await _events(_agent(llm))
    assert _check_events(events) == [("actions", "revise", 1), ("actions", "marked", 1)]

    llm = _Scripted([_compare_call(), _answer(claim)])
    events = await _events(_agent(llm, surface="web"))
    assert _check_events(events) == [] and events[-1]["content"] == claim


@pytest.mark.asyncio
async def test_on_the_web_an_earlier_answer_is_a_source_for_its_numbers() -> None:
    """A figure shown and checked in an earlier turn may be restated on the web;
    on the command line only the tool results and the user's words count."""
    history = [
        Message(role="user", content="compare A and B"),
        Message(role="assistant", content="About 1,111 windows sit together."),
    ]
    answer = "As before, about 1,111 windows sit together."
    llm = _Scripted([_compare_call(), _answer(answer)])
    events = [
        e
        async for e in _agent(llm, surface="web").run_stream(
            "and now?", history=history
        )
    ]
    assert _check_events(events) == [] and events[-1]["content"] == answer

    llm = _Scripted([_compare_call(), _answer(answer), _answer(answer)])
    events = [e async for e in _agent(llm).run_stream("and now?", history=history)]
    assert _check_events(events) == [("numbers", "revise", 1), ("numbers", "marked", 1)]


class _NamesTheSpill(_Scripted):
    """Answers with the path the harness spilled the tool's result to."""

    async def chat(
        self, messages: list[Message], *, tools: Any = None, **kw: Any
    ) -> ChatResponse:
        if messages[-1].role == "tool":
            saved = json.loads(messages[-1].content or "{}").get("saved_to")
            self.responses = [_answer(f"The full result was saved to `{saved}`.")]
        return await super().chat(messages, tools=tools, **kw)


@pytest.mark.asyncio
async def test_the_spill_file_the_answer_names_is_a_file_this_run_wrote(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    from olmoearth_agent.harness.spill import SPILL_BYTES_ENV
    from olmoearth_agent.security.paths import OUTPUT_ROOT_ENV

    monkeypatch.setenv(OUTPUT_ROOT_ENV, str(tmp_path))
    monkeypatch.setenv(SPILL_BYTES_ENV, "50")
    llm = _NamesTheSpill([_compare_call()])
    events = await _events(_agent(llm))
    final = events[-1]["content"]
    assert "tool_results/compare_" in final
    assert _check_events(events) == [] and events[-1]["marked"] == []


def test_the_surface_is_cli_or_web() -> None:
    with pytest.raises(ValueError, match="surface"):
        _agent(_Scripted([]), surface="tty")


@pytest.mark.asyncio
async def test_the_checks_but_the_numbers_can_be_switched_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    draft = f"{_BACKWARDS} About 1,111 of them."
    llm = _Scripted([_compare_call(), _answer(draft), _answer(_FIXED)])
    events = await _events(_agent(llm, check_answer=False))
    assert _check_events(events) == [("numbers", "revise", 1)]

    monkeypatch.setenv(CHECK_ANSWER_ENV, "off")
    llm = _Scripted([_compare_call(), _answer(draft), _answer(_FIXED)])
    events = await _events(_agent(llm))
    assert _check_events(events) == [("numbers", "revise", 1)]

    monkeypatch.setenv(CHECK_NUMBERS_ENV, "0")
    llm = _Scripted([_compare_call(), _answer(draft)])
    events = await _events(_agent(llm))
    assert _check_events(events) == [] and len(llm.calls) == 2


@pytest.mark.asyncio
async def test_run_collects_the_checks() -> None:
    llm = _Scripted([_compare_call(), _answer(_BACKWARDS), _answer(_BACKWARDS)])
    result = await _agent(llm).run("compare A and B")
    assert [(c["check"], c["action"]) for c in result.checks] == [
        ("direction", "revise"),
        ("direction", "marked"),
    ]
    assert result.marked == ["direction"] and result.revised is True
    assert result.grounding_checks == [] and result.grounding_revised is False
    assert result.final_content == f"{_BACKWARDS} [unverified: direction]"


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


def test_the_cli_prints_the_marked_answer_and_names_the_check(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    llm = _Scripted([_compare_call(), _answer(_BACKWARDS), _answer(_BACKWARDS)])
    cli = _patch_cli(monkeypatch, llm)
    assert cli.main(["compare A and B", "--show-trace"]) == 0
    out = capsys.readouterr()
    assert out.out.strip() == f"{_BACKWARDS} [unverified: direction]"
    assert "[direction] 1 sentence(s) flagged, rewrite asked" in out.err
    assert "(the direction check still flags 1 statement(s)" in out.err


def test_the_web_bridge_streams_the_checks_on_the_web_surface(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from olmoearth_agent import serve

    monkeypatch.setattr(serve, "preferences_block", lambda: "")
    answer = f"{_BACKWARDS} The first 30 windows are listed above."
    with TestClient(serve.app) as client:
        serve.app.state.llm = _Scripted(
            [_compare_call(), _answer(answer), _answer(answer)]
        )
        serve.app.state.registry = _registry()
        serve.app.state.skill_index = ""
        resp = client.post(
            "/api/run",
            json={"brief": "compare A and B"},
            headers={"X-Olmoearth-Key": "k"},
        )
    assert resp.status_code == 200
    events = [
        json.loads(line[len("data: ") :])
        for line in resp.text.splitlines()
        if line.startswith("data: ")
    ]
    # The direction is checked; "listed above" is not, on the web.
    assert _check_events(events) == [
        ("direction", "revise", 1),
        ("direction", "marked", 1),
    ]
    final = next(e for e in events if e["type"] == "final")
    assert final["content"] == (
        f"{_BACKWARDS} [unverified: direction] The first 30 windows are listed above."
    )
