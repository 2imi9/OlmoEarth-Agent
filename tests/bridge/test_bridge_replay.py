# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""The replay bridge against the loop it replays (``olmoearth_agent.bridge``).

A scripted model calls a fake registry's tools (one answers, one refuses
with an output contract), drafts an answer with a number no tool returned,
and rewrites it when the harness asks. The same responses go once through
``LeadAgent`` with a stub model, as ``tests/harness/test_answer_check_loop.py``
drives it, and once through the bridge, one step at a time, each step a
fresh session reading the run directory, as a new process would. Nothing
here touches the network. Further tests cover a step that fails partway
(its tool calls kept for the same response, dropped for another), a step
stopped between saving the response and writing its view, divergence, the
rewrite call, spill names and skill loading.
"""

from __future__ import annotations

import asyncio
import json
import os
import stat
import time
import uuid
from collections.abc import Callable, Iterable
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import Any, cast

import pytest

from olmoearth_agent.bridge.errors import BridgeError, BridgeHalt
from olmoearth_agent.bridge.replay import replayed_error
from olmoearth_agent.bridge.scripted import ModelCall, RecordedResponse, ScriptedLLM
from olmoearth_agent.bridge.session import begin, render_text, show, step
from olmoearth_agent.bridge.store import STALE_LOCK_SECONDS
from olmoearth_agent.harness import spill
from olmoearth_agent.harness.agent import (
    CHECK_ANSWER_ENV,
    CHECK_NUMBERS_ENV,
    LeadAgent,
    grounding_prompt,
)
from olmoearth_agent.harness.state import ThreadState
from olmoearth_agent.llm.presets import (
    CLAIM_CHECK_MODE,
    DEFAULT_AGENT_MODE,
    REVISION_MODE,
)
from olmoearth_agent.llm.types import ChatResponse, Message, ToolCall, ToolSpec
from olmoearth_agent.provenance.log import ProvenanceLog
from olmoearth_agent.security.paths import OUTPUT_ROOT_ENV
from olmoearth_agent.skills.registry import READ_ONLY_ENV
from olmoearth_agent.studio.client import StudioAccessError, StudioClient
from olmoearth_agent.tools.registry import (
    RegisteredTool,
    ToolContext,
    ToolRegistry,
    own_hint,
)

Factory = Callable[[], ToolRegistry]

_MAP_SCHEMA = {"type": "object", "properties": {"map": {"type": "string"}}}
_EMPTY_SCHEMA = {"type": "object", "properties": {}, "required": []}
_BRIEF = "How many windows does map a have?"
_MEASURED = {"map": "a", "n_windows": 1570}
_REFUSED = "budget 300 is over the 173 valid windows"
_DRAFT = "Map a has about 1,111 windows."
_FIXED = "Map a has 1,570 windows."


def _calls(*calls: tuple[str, dict[str, Any]]) -> dict[str, Any]:
    """A response that calls the ``(name, arguments)`` tools, in order."""
    return {"tool_calls": [{"name": n, "arguments": a} for n, a in calls]}


_MEASURE_A = _calls(("measure", {"map": "a"}))
_BOTH = _calls(("measure", {"map": "a"}), ("refuse", {}))


class _Refusal(ValueError):
    """A refusal with an output contract, as ``statistical_rules.refusal`` raises."""

    def __init__(self, message: str, contract: dict[str, Any]) -> None:
        super().__init__(message)
        self.contract = contract


def _registry(calls: dict[str, int]) -> ToolRegistry:
    """measure answers, refuse refuses; ``calls`` counts each handler's runs."""

    async def measure(args: dict[str, Any], _ctx: ToolContext) -> dict[str, Any]:
        calls["measure"] = calls.get("measure", 0) + 1
        return {**_MEASURED, "map": args.get("map", "a")}

    async def refuse(_args: dict[str, Any], _ctx: ToolContext) -> dict[str, Any]:
        calls["refuse"] = calls.get("refuse", 0) + 1
        raise _Refusal(_REFUSED, {"must_state": ["No plan was made."], "other": 1})

    measure_spec = ToolSpec("measure", "count a map's windows", _MAP_SCHEMA)
    refuse_spec = ToolSpec("refuse", "plan a label sample", _EMPTY_SCHEMA)
    registry = ToolRegistry()
    registry.register(RegisteredTool(measure_spec, measure))
    registry.register(RegisteredTool(refuse_spec, refuse))
    return registry


class _Studio:
    """A Studio client the fake tools never call."""

    @classmethod
    def from_env(cls) -> _Studio:
        return cls()

    async def aclose(self) -> None:
        return None


def _studio() -> StudioClient:
    return cast(StudioClient, _Studio())


class _Stub:
    """Returns scripted responses in order; keeps each request's fingerprint.

    The stub model of ``tests/harness/test_answer_check_loop.py`` (``_Scripted``),
    keeping each request as the bridge fingerprints it.
    """

    def __init__(self, responses: Iterable[ChatResponse]) -> None:
        self.responses = list(responses)
        self.fingerprints: list[str] = []

    async def chat(
        self, messages: list[Message], *, tools: Any = None, **kw: Any
    ) -> ChatResponse:
        call = ModelCall.of(
            len(self.fingerprints) + 1,
            messages,
            tools,
            kw.get("mode", DEFAULT_AGENT_MODE),
            kw.get("max_tokens"),
            kw.get("preserve_thinking", True),
        )
        self.fingerprints.append(call.sha256)
        return self.responses.pop(0)


def _as_chat(k: int, response: dict[str, Any]) -> ChatResponse:
    """A response as the live model would send it on model call ``k``."""
    calls = [
        ToolCall(id=f"call_{k}_{j}", name=c["name"], arguments=dict(c["arguments"]))
        for j, c in enumerate(response.get("tool_calls", []), 1)
    ]
    return ChatResponse(
        content=response.get("content"),
        tool_calls=calls,
        finish_reason="tool_calls" if calls else "stop",
    )


def _jsonl(path: Path) -> list[dict[str, Any]]:
    """A JSON-lines file's records; lines end at "\\n" only, as the bridge reads them."""
    lines = path.read_text(encoding="utf-8").split("\n")
    return [json.loads(line) for line in lines if line]


async def _begin(run: Path, factory: Factory) -> dict[str, Any]:
    """Begin a run of the brief, with no skill index."""
    return await begin(
        run,
        _BRIEF,
        skill_index="",
        registry_factory=factory,
        studio=_studio(),
    )


async def _step(run: Path, response: Any, factory: Factory) -> dict[str, Any]:
    """One step in a fresh session, as a new process makes it."""
    return await step(run, response, registry_factory=factory, studio=_studio())


async def _bridge(
    run: Path, responses: list[dict[str, Any]], factory: Factory
) -> list[dict[str, Any]]:
    """Begin, then one step per response."""
    first = await _begin(run, factory)
    return [first, *[await _step(run, r, factory) for r in responses]]


def _entries(entries: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Provenance entries without what each run draws anew: its id and the times."""
    return [
        {k: v for k, v in e.items() if k not in ("run_id", "timestamp")}
        for e in entries
    ]


@pytest.fixture(autouse=True)
def _settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv(CHECK_NUMBERS_ENV, raising=False)
    monkeypatch.delenv(CHECK_ANSWER_ENV, raising=False)
    monkeypatch.setenv(OUTPUT_ROOT_ENV, str(tmp_path / "outputs"))


@pytest.mark.asyncio
async def test_a_run_step_by_step_is_the_loop_run_at_once(tmp_path: Path) -> None:
    responses = [_BOTH, {"content": _DRAFT}, {"content": _FIXED}]

    direct_calls: dict[str, int] = {}
    stub = _Stub([_as_chat(k, r) for k, r in enumerate(responses, 1)])
    agent = LeadAgent(
        stub,  # type: ignore[arg-type]
        _registry(direct_calls),
        _studio(),
        state=ThreadState(provenance=ProvenanceLog(run_id="direct")),
        skill_index="",
        local=False,
        check_claims=False,
        surface="cli",
    )
    events = [e async for e in agent.run_stream(_BRIEF)]

    bridge_calls: dict[str, int] = {}
    run = tmp_path / "run"
    views = await _bridge(run, responses, partial(_registry, bridge_calls))

    # Every request the model got, byte for byte.
    recorded = _jsonl(run / "responses.jsonl")
    assert [r["request_sha256"] for r in recorded] == stub.fingerprints
    # The tool calls and their results, the checks and the final event.
    assert _jsonl(run / "events.jsonl") == json.loads(json.dumps(events, default=str))
    assert [e["type"] for e in events] == [
        "tool_call",
        "tool_result",
        "tool_call",
        "tool_result",
        "check",
        "grounding_check",
        "final",
    ]
    refused = events[3]["result"]
    assert refused["error"] == f"_Refusal: {_REFUSED}"
    assert refused["must_state"] == ["No plan was made."] and "other" not in refused
    # The provenance, less the run's id and the times.
    provenance = json.loads((run / "provenance.json").read_text())
    direct = json.loads(
        json.dumps([asdict(e) for e in agent.state.provenance.entries], default=str)
    )
    assert _entries(provenance["entries"]) == _entries(direct)
    # The answer.
    final = views[-1]
    assert final["status"] == "final" and final["content"] == events[-1]["content"]
    assert final["content"] == _FIXED and final["revised"] is True
    # Each handler ran once in each route, however many steps replayed it.
    assert direct_calls == bridge_calls == {"measure": 1, "refuse": 1}


@pytest.mark.asyncio
async def test_a_replay_serves_the_cache_and_runs_no_handler_again(
    tmp_path: Path,
) -> None:
    calls: dict[str, int] = {}
    factory = partial(_registry, calls)
    run = tmp_path / "run"
    await _begin(run, factory)
    assert calls == {}
    await _step(run, _MEASURE_A, factory)
    assert calls == {"measure": 1}
    await _step(run, _calls(("measure", {"map": "b"})), factory)
    assert calls == {"measure": 2}
    final = await _step(run, {"content": _FIXED}, factory)
    # Call 1 was replayed by two steps and call 2 by one: no handler ran again.
    assert calls == {"measure": 2}
    assert final["status"] == "final" and final["content"] == _FIXED
    records = _jsonl(run / "tool_cache.jsonl")
    assert [(r["call"], r["name"], r["arguments"]) for r in records] == [
        (1, "measure", {"map": "a"}),
        (2, "measure", {"map": "b"}),
    ]
    assert records[1]["result"] == {"map": "b", "n_windows": 1570}


@pytest.mark.asyncio
async def test_a_replay_that_differs_from_the_record_stops(tmp_path: Path) -> None:
    calls: dict[str, int] = {}
    factory = partial(_registry, calls)

    async def started(name: str) -> Path:
        run = tmp_path / name
        await _begin(run, factory)
        await _step(run, _MEASURE_A, factory)
        return run

    # The cache holds another call than the one the loop makes.
    run = await started("tools")
    cache = run / "tool_cache.jsonl"
    record = json.loads(cache.read_text())
    cache.write_text(json.dumps({**record, "arguments": {"map": "b"}}) + "\n")
    pending = (run / "pending.json").read_text()
    with pytest.raises(BridgeError, match="the run diverged at call 1"):
        await _step(run, {"content": _FIXED}, factory)
    assert (run / "pending.json").read_text() == pending
    assert len(_jsonl(run / "responses.jsonl")) == 1
    assert calls == {"measure": 1}  # the diverged call was not run

    # A recorded response answered another request than the loop now makes.
    run = await started("model")
    responses = run / "responses.jsonl"
    line = json.loads(responses.read_text())
    responses.write_text(json.dumps({**line, "request_sha256": "0" * 64}) + "\n")
    with pytest.raises(BridgeError, match="the run diverged at model call 1"):
        await _step(run, {"content": _FIXED}, factory)
    assert calls == {"measure": 2}  # one live call per run, none on a replay


def _flaky(calls: dict[str, int], broken: dict[str, bool]) -> ToolRegistry:
    """measure and refuse, and export, whose result is not JSON while broken."""
    registry = _registry(calls)

    async def export(_args: dict[str, Any], _ctx: ToolContext) -> dict[str, Any]:
        calls["export"] = calls.get("export", 0) + 1
        return {"saved_to": object() if broken["on"] else "outputs/a.json"}

    spec = ToolSpec("export", "export a map", _EMPTY_SCHEMA)
    registry.register(RegisteredTool(spec, export))
    return registry


_MEASURE_EXPORT = _calls(("measure", {"map": "a"}), ("export", {}))
_NO_TOOL = {"content": "No tool was called."}


async def _failed(run: Path, factory: Factory) -> None:
    """Begin, then a step that fails after its first tool call ran."""
    await _begin(run, factory)
    with pytest.raises(BridgeError, match=r"the result of call 2 \(export\) is not"):
        await _step(run, _MEASURE_EXPORT, factory)


@pytest.mark.asyncio
async def test_a_failed_step_s_tool_calls_serve_the_same_response_again(
    tmp_path: Path,
) -> None:
    calls: dict[str, int] = {}
    broken = {"on": True}
    factory = partial(_flaky, calls, broken)
    run = tmp_path / "run"
    await _begin(run, factory)
    pending = (run / "pending.json").read_text()
    with pytest.raises(BridgeError, match="is not JSON"):
        await _step(run, _MEASURE_EXPORT, factory)
    # No response saved and the pending call as it was; the call that ran is
    # kept, for model call 1.
    assert (run / "pending.json").read_text() == pending
    assert not (run / "responses.jsonl").exists()
    (record,) = _jsonl(run / "tool_cache.jsonl")
    assert (record["name"], record["model_call"]) == ("measure", 1)
    assert calls == {"measure": 1, "export": 1}

    broken["on"] = False
    second = await _step(run, _MEASURE_EXPORT, factory)
    assert second["model_call"] == 2 and "dropped_tool_calls" not in second
    # measure is served from the failed step's record; export runs again.
    assert calls == {"measure": 1, "export": 2}
    records = _jsonl(run / "tool_cache.jsonl")
    assert [(r["name"], r["model_call"]) for r in records] == [
        ("measure", 1),
        ("export", 1),
    ]
    final = await _step(run, {"content": _FIXED}, factory)
    assert final["status"] == "final" and final["content"] == _FIXED
    assert calls == {"measure": 1, "export": 2}


@pytest.mark.asyncio
async def test_a_failed_step_s_tool_calls_are_dropped_for_another_response(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    calls: dict[str, int] = {}
    factory = partial(_flaky, calls, {"on": True})

    # Another first call: the kept record is dropped there, the new call runs.
    run = tmp_path / "other"
    await _failed(run, factory)
    view = await _step(run, _calls(("measure", {"map": "b"})), factory)
    assert view["status"] == "model_call" and view["model_call"] == 2
    assert view["dropped_tool_calls"] == [{"call": 1, "name": "measure"}]
    assert "a Studio write among them may already have happened" in caplog.text
    records = _jsonl(run / "tool_cache.jsonl")
    assert [(r["name"], r["arguments"]) for r in records] == [("measure", {"map": "b"})]
    assert calls == {"measure": 2, "export": 1}
    # That response is saved: the next step replays it from the cache.
    final = await _step(run, {"content": _FIXED}, factory)
    assert final["status"] == "final" and "dropped_tool_calls" not in final
    assert calls == {"measure": 2, "export": 1}

    # An answer instead: the kept record is dropped at the step's end.
    run = tmp_path / "answer"
    await _failed(run, factory)
    final = await _step(run, _NO_TOOL, factory)
    assert final["status"] == "final" and final["content"] == _NO_TOOL["content"]
    assert final["dropped_tool_calls"] == [{"call": 1, "name": "measure"}]
    assert _jsonl(run / "tool_cache.jsonl") == []
    assert calls == {"measure": 3, "export": 2}


@pytest.mark.asyncio
async def test_a_step_stopped_before_its_view_is_finished_by_the_next(
    tmp_path: Path,
) -> None:
    calls: dict[str, int] = {}
    factory = partial(_registry, calls)
    run = tmp_path / "run"
    await _begin(run, factory)
    first = (run / "pending.json").read_text()
    second = await _step(run, _MEASURE_A, factory)

    # As if killed after it saved the response and before it wrote pending.json.
    (run / "pending.json").write_text(first)
    with pytest.raises(BridgeError, match="already has its response saved"):
        await _step(run, _calls(("measure", {"map": "b"})), factory)
    assert await _step(run, _MEASURE_A, factory) == second
    (run / "pending.json").write_text(first)
    assert await _step(run, None, factory) == second
    assert len(_jsonl(run / "responses.jsonl")) == 1
    assert calls == {"measure": 1}


@pytest.mark.asyncio
async def test_an_answer_that_fails_a_check_gets_a_call_without_tools(
    tmp_path: Path,
) -> None:
    calls: dict[str, int] = {}
    factory = partial(_registry, calls)
    run = tmp_path / "run"
    first = await _begin(run, factory)
    assert first["model_call"] == 1 and first["turn"] == 1
    assert first["tools"] == ["measure", "refuse"]
    assert first["mode"] == DEFAULT_AGENT_MODE
    assert first["system"]["changed"] is True and first["system"]["text"]
    assert first["messages"] == [{"role": "user", "content": _BRIEF}]

    second = await _step(run, _MEASURE_A, factory)
    assert second["system"] == {"sha256": first["system"]["sha256"], "changed": False}
    (result,) = second["messages"]
    assert result["role"] == "tool" and result["tool_call_id"] == "call_1_1"
    assert json.loads(result["content"]) == {"ok": True, "result": _MEASURED}

    rewrite = await _step(run, {"content": _DRAFT}, factory)
    assert rewrite["status"] == "model_call" and rewrite["model_call"] == 3
    assert rewrite["turn"] == 2  # a rewrite repeats the answer's turn
    assert rewrite["tools"] is None and rewrite["mode"] == REVISION_MODE
    assert rewrite["messages"] == [
        {"role": "user", "content": grounding_prompt(["1,111"])}
    ]
    assert "No tools are offered" in rewrite["respond"]

    # A tool call is no answer to a call that offers no tools.
    pending = (run / "pending.json").read_text()
    with pytest.raises(BridgeError, match="No tools are offered"):
        await _step(run, _MEASURE_A, factory)
    assert (run / "pending.json").read_text() == pending

    final = await _step(run, {"content": _FIXED}, factory)
    assert final["status"] == "final" and final["content"] == _FIXED
    assert final["revised"] is True and final["grounding_revised"] is True
    assert final["marked"] == [] and final["notes"] == []
    assert calls == {"measure": 1}


def _measure_only(calls: dict[str, int]) -> ToolRegistry:
    tool = _registry(calls).tool_of("measure")
    assert tool is not None
    registry = ToolRegistry()
    registry.register(tool)
    return registry


def test_the_final_notes_are_the_command_line_s(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from olmoearth_agent import cli

    responses = [_MEASURE_A, {"content": _DRAFT}, {"content": _DRAFT}]
    stub = _Stub([_as_chat(k, r) for k, r in enumerate(responses, 1)])

    class _Skills:
        def index(self) -> str:
            return ""

    factory = partial(_measure_only, {})
    monkeypatch.setattr(cli, "OlmoEarthLLM", lambda: stub)
    monkeypatch.setattr(cli, "StudioClient", _Studio)
    monkeypatch.setattr(cli, "build_default_registry", factory)
    monkeypatch.setattr(cli, "SkillLoader", _Skills)
    assert cli.main([_BRIEF, "--show-trace"]) == 0
    printed = capsys.readouterr()

    views = asyncio.run(_bridge(tmp_path / "run", responses, factory))
    final = views[-1]
    assert printed.out == f"{final['content']}\n"
    assert printed.err.splitlines() == final["trace"] + final["notes"]
    assert final["marked"] == ["numbers"]
    assert final["notes"] == ["(numbers in the answer that no tool returned: 1,111)"]


def _big(_calls: dict[str, int]) -> ToolRegistry:
    async def big(_args: dict[str, Any], _ctx: ToolContext) -> dict[str, Any]:
        return {**_MEASURED, "windows": [{"id": f"w{i}"} for i in range(40)]}

    registry = ToolRegistry()
    spec = ToolSpec("big", "list windows", _EMPTY_SCHEMA)
    registry.register(RegisteredTool(spec, big))
    return registry


@pytest.mark.asyncio
async def test_a_spilled_result_keeps_its_file_across_steps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(spill.SPILL_BYTES_ENV, "50")
    factory = partial(_big, {})
    run = tmp_path / "run"
    await _begin(run, factory)
    view = await _step(run, _calls(("big", {})), factory)
    saved = json.loads(view["messages"][0]["content"])["saved_to"]
    assert "tool_results/big_" in saved and Path(saved).is_file()

    # The next step replays the call and spills it again, under the same name:
    # the request is the one the model read, and the file is one the run wrote.
    answer = {"content": f"The full result was saved to `{saved}`."}
    final = await _step(run, answer, factory)
    assert final["status"] == "final" and final["marked"] == []
    assert [e for e in _jsonl(run / "events.jsonl") if e["type"] == "check"] == []
    assert spill.uuid is uuid  # put back after each step


def _named(_calls: dict[str, int]) -> ToolRegistry:
    async def name(_args: dict[str, Any], _ctx: ToolContext) -> dict[str, Any]:
        # JSON keeps U+2028 and U+0085 unescaped; str.splitlines() splits there.
        return {"name": "Karst Binary\x85", "n_windows": 1570}

    registry = ToolRegistry()
    spec = ToolSpec("name", "name a map", _EMPTY_SCHEMA)
    registry.register(RegisteredTool(spec, name))
    return registry


@pytest.mark.asyncio
async def test_a_result_holding_a_line_separator_replays(tmp_path: Path) -> None:
    factory = partial(_named, {})
    run = tmp_path / "run"
    await _begin(run, factory)
    await _step(run, _calls(("name", {})), factory)
    final = await _step(run, {"content": _FIXED}, factory)
    assert final["status"] == "final" and final["content"] == _FIXED
    (record,) = _jsonl(run / "tool_cache.jsonl")
    assert record["result"]["name"] == "Karst Binary\x85"


def _skill_registry(calls: dict[str, int]) -> ToolRegistry:
    """measure and refuse (core), rs_tool deferred under olmoearth-rslearn, load_skill."""
    from olmoearth_agent.skills.loader import SkillLoader
    from olmoearth_agent.tools.skill_tools import build_skill_tools

    registry = _registry(calls)

    async def fit(_args: dict[str, Any], _ctx: ToolContext) -> dict[str, Any]:
        calls["rs_tool"] = calls.get("rs_tool", 0) + 1
        return {"fitted": True}

    registry.register(
        RegisteredTool(ToolSpec("rs_tool", "rs", _EMPTY_SCHEMA), fit),
        group="olmoearth-rslearn",
    )
    empty = SkillLoader(root="/nonexistent-skills-dir")
    registry.register_all(build_skill_tools(empty, registry=registry))
    return registry


@pytest.mark.asyncio
async def test_a_loaded_skill_is_loaded_again_on_each_replay(tmp_path: Path) -> None:
    calls: dict[str, int] = {}
    factory = partial(_skill_registry, calls)
    run = tmp_path / "run"
    first = await _begin(run, factory)
    assert "rs_tool" not in first["tools"]
    assert "- rs_tool" not in first["system"]["text"]
    load = _calls(("olmoearth_load_skill", {"name": "olmoearth-rslearn"}))
    second = await _step(run, load, factory)
    assert "rs_tool" in second["tools"]
    assert second["system"]["changed"] is True
    assert "- rs_tool: rs." in second["system"]["text"]

    # The replay loads the group again (the handler runs again), so the loop's
    # requests are the ones the model answered and the card stays as it was.
    third = await _step(run, _calls(("rs_tool", {})), factory)
    assert "rs_tool" in third["tools"] and third["system"]["changed"] is False
    records = _jsonl(run / "tool_cache.jsonl")
    assert [(r["name"], r.get("rerun", False)) for r in records] == [
        ("olmoearth_load_skill", True),
        ("rs_tool", False),
    ]
    assert calls == {"rs_tool": 1}


_CANNOT = "write to Studio: this run's key is read-only (a test's line)"
# An answer with no number, so the answer checks ask for no rewrite.
_PLAIN = "No tool measured map a in this run."


def _read_only_registry(calls: dict[str, int]) -> ToolRegistry:
    """The fake registry with a ``cannot`` line, as a read-only key gives one."""
    registry = _registry(calls)
    registry.cannot["studio_write"] = _CANNOT
    return registry


@pytest.mark.asyncio
async def test_a_setting_s_card_line_reaches_the_model_through_the_replay(
    tmp_path: Path,
) -> None:
    # The live loop puts registry.cannot under what no tool can do; the
    # replay's copy of the registry must too, or the model reads another card
    # (found on aicr: OLMOEARTH_READ_ONLY's line was missing from the bridge's
    # system prompt while the write tools were left out).
    calls: dict[str, int] = {}
    stub = _Stub([_as_chat(1, {"content": _PLAIN})])
    agent = LeadAgent(
        stub,  # type: ignore[arg-type]
        _read_only_registry(calls),
        _studio(),
        state=ThreadState(provenance=ProvenanceLog(run_id="direct")),
        skill_index="",
        local=False,
        check_claims=False,
        surface="cli",
    )
    _ = [e async for e in agent.run_stream(_BRIEF)]

    run = tmp_path / "run"
    first = await _begin(run, partial(_read_only_registry, calls))
    assert f"- {_CANNOT}" in first["system"]["text"]
    final = await _step(run, {"content": _PLAIN}, partial(_read_only_registry, calls))
    assert final["status"] == "final" and final["revised"] is False
    recorded = _jsonl(run / "responses.jsonl")
    assert [r["request_sha256"] for r in recorded] == stub.fingerprints


def _refused_key_registry(calls: dict[str, int]) -> ToolRegistry:
    """measure and refuse, and write, which Studio refuses as a read-only key's."""
    registry = _registry(calls)

    async def write(_args: dict[str, Any], _ctx: ToolContext) -> dict[str, Any]:
        calls["write"] = calls.get("write", 0) + 1
        raise StudioAccessError("POST", "/projects", 403, "permission_error")

    spec = ToolSpec("write", "create a project", _EMPTY_SCHEMA)
    registry.register(RegisteredTool(spec, write))
    return registry


@pytest.mark.asyncio
async def test_a_refused_key_s_hint_reaches_the_model_through_the_replay(
    tmp_path: Path,
) -> None:
    # The registry gives a refused key's own hint (stop, report the refusal)
    # in place of the generic one (check the ids). The bridge raises every
    # handler error from its record, the live call's too, so the record keeps
    # the hint and the replayed error carries it.
    responses = [_calls(("write", {})), {"content": _PLAIN}]
    direct_calls: dict[str, int] = {}
    stub = _Stub([_as_chat(k, r) for k, r in enumerate(responses, 1)])
    agent = LeadAgent(
        stub,  # type: ignore[arg-type]
        _refused_key_registry(direct_calls),
        _studio(),
        state=ThreadState(provenance=ProvenanceLog(run_id="direct")),
        skill_index="",
        local=False,
        check_claims=False,
        surface="cli",
    )
    events = [e async for e in agent.run_stream(_BRIEF)]
    live = StudioAccessError("POST", "/projects", 403, "permission_error").hint
    (direct,) = [e["result"] for e in events if e["type"] == "tool_result"]
    assert direct["hint"] == live and "Do not retry it" in live
    assert "Check that referenced ids exist" not in direct["hint"]

    bridge_calls: dict[str, int] = {}
    factory = partial(_refused_key_registry, bridge_calls)
    run = tmp_path / "run"
    await _begin(run, factory)
    # The live call: the envelope the model reads is the live loop's.
    second = await _step(run, responses[0], factory)
    envelope = json.loads(second["messages"][0]["content"])
    assert envelope == json.loads(json.dumps(direct))
    (record,) = _jsonl(run / "tool_cache.jsonl")
    assert record["error"]["type"] == "StudioAccessError"
    assert record["error"]["hint"] == live
    # A later step serves it from the cache: the events are the live loop's.
    final = await _step(run, responses[1], factory)
    assert final["status"] == "final" and final["content"] == _PLAIN
    assert _jsonl(run / "events.jsonl") == json.loads(json.dumps(events, default=str))
    recorded = _jsonl(run / "responses.jsonl")
    assert [r["request_sha256"] for r in recorded] == stub.fingerprints
    assert direct_calls == bridge_calls == {"write": 1}


@pytest.mark.asyncio
async def test_only_a_refused_key_s_hint_is_recorded_and_replayed(
    tmp_path: Path,
) -> None:
    # The refusal with a contract keeps the generic hint, as live, and its
    # record keeps no hint.
    calls: dict[str, int] = {}
    factory = partial(_refused_key_registry, calls)
    run = tmp_path / "run"
    await _begin(run, factory)
    view = await _step(run, _calls(("refuse", {}), ("write", {})), factory)
    tool = [m for m in view["messages"] if m["role"] == "tool"]
    refused, written = (json.loads(m["content"]) for m in tool)
    assert "Check that referenced ids exist" in refused["hint"]
    assert written["hint"] == StudioAccessError.hint
    first, second = _jsonl(run / "tool_cache.jsonl")
    assert "hint" not in first["error"] and second["error"]["hint"]
    # A record cached before the hint was kept replays with none, so the
    # registry gives the generic hint, as it did when the record was made.
    old = replayed_error({"type": "StudioAccessError", "message": "refused"})
    assert old.hint is None and own_hint(old) is None
    kept = replayed_error({**second["error"]})
    assert own_hint(kept) == StudioAccessError.hint


@pytest.mark.asyncio
async def test_the_claim_check_s_call_is_refused() -> None:
    llm = ScriptedLLM([])
    with pytest.raises(BridgeHalt, match="claim check"):
        await llm.chat([Message(role="user", content="x")], mode=CLAIM_CHECK_MODE)
    assert llm.calls == [] and llm.pending is None


@pytest.mark.parametrize(
    ("response", "problem"),
    [
        ({}, "needs"),
        ({"answer": "x"}, "unknown response keys"),
        ({"content": 3}, "content must be a string"),
        ({"tool_calls": [{"name": "measure", "id": "c1"}]}, "unknown keys"),
        ({"tool_calls": [{"name": "measure", "arguments": "{"}]}, "not JSON"),
        ({"tool_calls": [{"arguments": {}}]}, "name must be"),
    ],
)
def test_a_malformed_response_is_refused(response: Any, problem: str) -> None:
    with pytest.raises(BridgeError, match=problem):
        RecordedResponse.parse(response)


@pytest.mark.asyncio
async def test_a_run_takes_one_response_per_pending_call(tmp_path: Path) -> None:
    factory = partial(_registry, {})
    run = tmp_path / "run"
    await _begin(run, factory)
    # The run's files hold account names and tool results: owner only.
    assert stat.S_IMODE(run.stat().st_mode) & 0o077 == 0
    with pytest.raises(BridgeError, match="not an empty directory"):
        await _begin(run, factory)
    with pytest.raises(BridgeError, match="waiting for a model response"):
        await _step(run, None, factory)
    await _step(run, _NO_TOOL, factory)
    with pytest.raises(BridgeError, match="the run has ended"):
        await _step(run, {"content": "Again."}, factory)
    with pytest.raises(BridgeError, match="the run has ended"):
        await _step(run, None, factory)


@pytest.mark.asyncio
async def test_a_run_directory_that_existed_empty_is_made_private(
    tmp_path: Path,
) -> None:
    # mkdir leaves an existing directory's mode as it was; begin sets 0700.
    run = tmp_path / "made"
    run.mkdir()
    os.chmod(run, 0o744)
    await _begin(run, partial(_registry, {}))
    assert stat.S_IMODE(run.stat().st_mode) == 0o700
    # A symlink is refused: where it points is not this run's to set.
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target)
    with pytest.raises(BridgeError, match="is a symlink"):
        await _begin(link, partial(_registry, {}))
    assert list(target.iterdir()) == []


_HELD = {
    "host": "c0101",
    "pid": 7,
    "job": "1605000",
    "started": "2026-10-31T12:00:00+00:00",
}


@pytest.mark.asyncio
async def test_a_second_step_is_refused_while_one_holds_the_run(
    tmp_path: Path,
) -> None:
    # After a dropped ssh connection the first step may still run on the
    # cluster; a step sent again meanwhile must not replay the run too.
    calls: dict[str, int] = {}
    factory = partial(_registry, calls)
    run = tmp_path / "run"
    first = await _begin(run, factory)
    lock = run / "step.lock"
    assert not lock.exists()  # begin's step released it
    lock.write_text(json.dumps(_HELD))
    pending = (run / "pending.json").read_text()
    running = (
        r"a step is still running on this run \(since 2026-10-31T12:00:00\+00:00, "
        r"job 1605000, host c0101\)"
    )
    with pytest.raises(BridgeError, match=running):
        await _step(run, _MEASURE_A, factory)
    assert calls == {} and (run / "pending.json").read_text() == pending
    assert not (run / "responses.jsonl").exists()
    assert json.loads(lock.read_text()) == _HELD  # the holder's, untouched
    # show says so: the call shown is the one that step is answering.
    shown = show(run)
    assert shown["step_running"] == _HELD
    assert shown["model_call"] == first["model_call"] == 1
    assert "do not send the response again" in render_text(shown)
    # A lock older than srun's limit is a dead step's: the next step takes
    # it over, and removes it at its end.
    old = time.time() - STALE_LOCK_SECONDS - 60
    os.utime(lock, (old, old))
    assert "step_running" not in show(run)
    second = await _step(run, _MEASURE_A, factory)
    assert second["model_call"] == 2 and calls == {"measure": 1}
    assert not lock.exists()


@pytest.mark.asyncio
async def test_a_response_for_another_call_is_refused_before_anything_runs(
    tmp_path: Path,
) -> None:
    # A response sent again after its call was answered (the first step
    # finished after all) would otherwise answer the next call.
    calls: dict[str, int] = {}
    factory = partial(_registry, calls)
    run = tmp_path / "run"
    await _begin(run, factory)
    second = await step(
        run, _MEASURE_A, call=1, registry_factory=factory, studio=_studio()
    )
    assert second["model_call"] == 2
    pending = (run / "pending.json").read_text()
    with pytest.raises(
        BridgeError,
        match="the response answers model call 1, and the pending call is model "
        "call 2: nothing ran",
    ):
        await step(run, _MEASURE_A, call=1, registry_factory=factory, studio=_studio())
    assert calls == {"measure": 1}
    assert len(_jsonl(run / "responses.jsonl")) == 1
    assert (run / "pending.json").read_text() == pending


@pytest.mark.asyncio
async def test_the_read_only_switch_is_kept_for_the_whole_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # It decides the tools offered and the card; the wrapper sets it unless
    # OE_BRIDGE_ALLOW_WRITES=1, and a step with the other setting is refused.
    monkeypatch.delenv(READ_ONLY_ENV, raising=False)
    calls: dict[str, int] = {}
    factory = partial(_registry, calls)
    run = tmp_path / "run"
    await _begin(run, factory)
    brief = json.loads((run / "brief.json").read_text())
    assert brief["settings"]["read_only"] is False
    monkeypatch.setenv(READ_ONLY_ENV, "1")
    with pytest.raises(
        BridgeError, match="read_only was False when the run began and is True now"
    ):
        await _step(run, _MEASURE_A, factory)
    assert calls == {}
    monkeypatch.delenv(READ_ONLY_ENV)
    assert (await _step(run, _MEASURE_A, factory))["model_call"] == 2


@pytest.mark.asyncio
async def test_a_run_recorded_with_a_memory_block_still_steps(tmp_path: Path) -> None:
    # A run begun before the preference memory went (31 October 2026)
    # recorded a memory_block, which is no longer read, and a missing
    # skill_index reads as "": such a run directory still loads and steps.
    # The replay checks the first call's fingerprint, system prompt included,
    # so a prompt that took the old block in would stop the step.
    calls: dict[str, int] = {}
    factory = partial(_registry, calls)
    run = tmp_path / "run"
    await _begin(run, factory)
    path = run / "brief.json"
    brief = json.loads(path.read_text())
    assert "memory_block" not in brief["settings"]
    brief["settings"]["memory_block"] = "Durable user preferences: grid 16"
    del brief["settings"]["skill_index"]
    path.write_text(json.dumps(brief))
    view = await _step(run, _MEASURE_A, factory)
    assert view["model_call"] == 2
    assert "Durable user preferences" not in json.dumps(view)


def _lines_ok(path: Path) -> bool:
    """Whether every line of a JSON-lines file is whole: it ends with "\\n"."""
    text = path.read_text(encoding="utf-8")
    return text.endswith("\n") and all(
        isinstance(json.loads(line), dict) for line in text.split("\n") if line
    )


@pytest.mark.asyncio
async def test_a_cut_off_tool_record_is_dropped_and_its_call_runs_again(
    tmp_path: Path,
) -> None:
    # A step killed while it appended a record leaves half a line, which
    # every later read would fail on; that record was never saved.
    calls: dict[str, int] = {}
    broken = {"on": True}
    factory = partial(_flaky, calls, broken)
    run = tmp_path / "run"
    await _failed(run, factory)
    cache = run / "tool_cache.jsonl"
    with cache.open("a", encoding="utf-8") as fh:
        fh.write('{"call": 2, "model_call": 1, "name": "export", "argu')
    broken["on"] = False
    second = await _step(run, _MEASURE_EXPORT, factory)
    assert second["model_call"] == 2 and "dropped_tool_calls" not in second
    assert calls == {"measure": 1, "export": 2}
    assert [r["name"] for r in _jsonl(cache)] == ["measure", "export"]
    assert _lines_ok(cache)


@pytest.mark.asyncio
async def test_a_cut_off_response_counts_as_not_saved(tmp_path: Path) -> None:
    # Killed while it appended the response, before it wrote pending.json:
    # the same response is taken as new, and its tool call is served.
    calls: dict[str, int] = {}
    factory = partial(_registry, calls)
    run = tmp_path / "run"
    await _begin(run, factory)
    pending = (run / "pending.json").read_text()
    second = await _step(run, _MEASURE_A, factory)
    responses = run / "responses.jsonl"
    line = responses.read_text(encoding="utf-8")
    responses.write_text(line[: len(line) // 2], encoding="utf-8")
    (run / "pending.json").write_text(pending)
    assert await _step(run, _MEASURE_A, factory) == second
    assert calls == {"measure": 1}
    assert _jsonl(responses) == [json.loads(line)]
    assert _lines_ok(responses)


@pytest.mark.asyncio
async def test_a_line_cut_inside_a_character_is_dropped_whole(
    tmp_path: Path,
) -> None:
    # Split as bytes: a cut inside U+0085's two bytes is no decode error.
    calls: dict[str, int] = {}
    factory = partial(_registry, calls)
    run = tmp_path / "run"
    await _begin(run, factory)
    cache = run / "tool_cache.jsonl"
    fragment = '{"call": 1, "model_call": 1, "name": "measure", "result": "Karst\x85'
    cache.write_bytes(fragment.encode("utf-8")[:-1])
    second = await _step(run, _MEASURE_A, factory)
    assert second["model_call"] == 2 and calls == {"measure": 1}
    (record,) = _jsonl(cache)
    assert record["result"] == _MEASURED
    assert _lines_ok(cache)


@pytest.mark.asyncio
async def test_the_view_carries_the_tool_specs_when_they_are_new(
    tmp_path: Path,
) -> None:
    # The capability card names required arguments only, and an argument a
    # spec does not declare is ignored: the model gets every spec on the first
    # call and whenever the tools offered change.
    factory = partial(_skill_registry, {})
    run = tmp_path / "run"
    first = await _begin(run, factory)
    assert first["tool_specs_changed"] is True
    specs = {s["name"]: s for s in first["tool_specs"]}
    assert [s["name"] for s in first["tool_specs"]] == first["tools"]
    measure = {"name": "measure", "description": "count a map's windows"}
    assert specs["measure"] == {**measure, "parameters": _MAP_SCHEMA}
    text = render_text(first)
    assert "-- tool specs (use only the argument names each declares) --" in text
    assert "- measure: count a map's windows" in text
    assert f"  parameters: {json.dumps(_MAP_SCHEMA)}" in text

    second = await _step(run, _MEASURE_A, factory)
    assert second["tool_specs_changed"] is False and "tool_specs" not in second
    assert "-- tool specs unchanged since the last call --" in render_text(second)

    load = _calls(("olmoearth_load_skill", {"name": "olmoearth-rslearn"}))
    third = await _step(run, load, factory)
    assert third["tool_specs_changed"] is True
    assert "rs_tool" in [s["name"] for s in third["tool_specs"]]
    assert "- rs_tool: rs" in render_text(third)


@pytest.mark.asyncio
async def test_a_run_whose_first_step_failed_is_started_without_a_response(
    tmp_path: Path,
) -> None:
    # begin writes brief.json before its first step; when that step fails
    # (OLMOEARTH_API_KEY unset, say), a step with no response starts the run.
    made: list[int] = []

    def factory() -> ToolRegistry:
        made.append(1)
        if len(made) == 2:  # begin's own step
            raise RuntimeError("OLMOEARTH_API_KEY is not set")
        return _registry({})

    run = tmp_path / "run"
    with pytest.raises(RuntimeError, match="OLMOEARTH_API_KEY is not set"):
        await _begin(run, factory)
    assert (run / "brief.json").is_file() and not (run / "step.lock").exists()
    with pytest.raises(BridgeError, match="with no response file"):
        show(run)
    with pytest.raises(BridgeError, match="the run has not started"):
        await _step(run, _MEASURE_A, factory)
    first = await _step(run, None, factory)
    assert first["status"] == "model_call" and first["model_call"] == 1
