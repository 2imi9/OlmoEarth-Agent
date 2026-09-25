# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""The middleware chain: LangChain 1.x's hook order, nesting, jumps and overrides.

The default chain's behaviour is pinned by the loop's own tests
(test_agent*.py, test_turn_cap_end_to_end.py, test_grounding_check.py,
test_answer_check_loop.py), which ran unchanged when the loop became a chain.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable
from typing import Any

import pytest

from olmoearth_agent.harness.agent import LeadAgent
from olmoearth_agent.harness.answer_checks import AnswerChecksMiddleware
from olmoearth_agent.harness.middleware import (
    AgentMiddleware,
    AgentState,
    MiddlewareChain,
    MiddlewareError,
    ModelRequest,
    ModelResponse,
    RunContext,
    Runtime,
    ToolCallRequest,
    ToolResult,
    hook_config,
)
from olmoearth_agent.harness.retry_hint import RetryHintMiddleware
from olmoearth_agent.harness.state import ThreadState
from olmoearth_agent.harness.turn_cap import TURN_CAP_PROMPT, TurnCapMiddleware
from olmoearth_agent.llm.presets import REVISION_MODE
from olmoearth_agent.llm.types import ChatResponse, Message, ToolCall, ToolSpec
from olmoearth_agent.tools.registry import RegisteredTool, ToolContext, ToolRegistry

_SCHEMA = {"type": "object", "properties": {}, "required": []}


def _answer(content: str | None, thinking: str | None = None) -> ChatResponse:
    return ChatResponse(
        content=content, tool_calls=[], thinking=thinking, finish_reason="stop"
    )


def _calls(*names: str) -> ChatResponse:
    calls = [ToolCall(id=f"c{i}", name=n, arguments={}) for i, n in enumerate(names)]
    return ChatResponse(content=None, tool_calls=calls, finish_reason="tool_calls")


class _Scripted:
    """Returns scripted responses in order, then answers "done"."""

    def __init__(self, responses: Iterable[ChatResponse] = ()) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[list[Message], Any, dict[str, Any]]] = []

    async def chat(
        self, messages: list[Message], *, tools: Any = None, **kw: Any
    ) -> ChatResponse:
        self.calls.append((list(messages), tools, kw))
        return self.responses.pop(0) if self.responses else _answer("done")


def _registry(log: list[str] | None = None) -> ToolRegistry:
    async def echo(args: dict[str, Any], _ctx: ToolContext) -> dict[str, Any]:
        if log is not None:
            log.append("tool")
        return dict(args)

    async def fail(_args: dict[str, Any], _ctx: ToolContext) -> None:
        raise ValueError("budget 300 is more than the 173 valid windows")

    registry = ToolRegistry()
    registry.register(RegisteredTool(ToolSpec("echo", "echo", _SCHEMA), echo))
    registry.register(RegisteredTool(ToolSpec("fail", "fail", _SCHEMA), fail))
    return registry


def _agent(llm: Any, middleware: list[AgentMiddleware] | None, **kw: Any) -> LeadAgent:
    return LeadAgent(
        llm, kw.pop("registry", None) or _registry(), studio=None, middleware=middleware, **kw  # type: ignore[arg-type]
    )


async def _events(
    agent: LeadAgent, brief: str = "go", **kw: Any
) -> list[dict[str, Any]]:
    return [e async for e in agent.run_stream(brief, **kw)]


class _Recorder(AgentMiddleware):
    """Logs every hook it runs as ``<label>.<hook>``."""

    def __init__(self, label: str, log: list[str]) -> None:
        self.label, self.log = label, log

    @property
    def name(self) -> str:
        return f"Recorder-{self.label}"

    async def abefore_agent(self, state: AgentState, runtime: Runtime) -> None:
        self.log.append(f"{self.label}.before_agent")

    async def abefore_model(self, state: AgentState, runtime: Runtime) -> None:
        self.log.append(f"{self.label}.before_model")

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        self.log.append(f"{self.label}>model")
        response = await handler(request)
        self.log.append(f"{self.label}<model")
        return response

    async def aafter_model(self, state: AgentState, runtime: Runtime) -> None:
        self.log.append(f"{self.label}.after_model")

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolResult]],
    ) -> ToolResult:
        self.log.append(f"{self.label}>tool")
        result = await handler(request)
        self.log.append(f"{self.label}<tool")
        return result

    async def aafter_agent(self, state: AgentState, runtime: Runtime) -> None:
        self.log.append(f"{self.label}.after_agent")


class _LoggingLLM(_Scripted):
    def __init__(self, log: list[str], responses: Iterable[ChatResponse]) -> None:
        super().__init__(responses)
        self.log = log

    async def chat(
        self, messages: list[Message], *, tools: Any = None, **kw: Any
    ) -> ChatResponse:
        self.log.append("model")
        return await super().chat(messages, tools=tools, **kw)


# --- order and nesting ------------------------------------------------------


@pytest.mark.asyncio
async def test_before_hooks_run_in_order_after_hooks_in_reverse_and_wraps_nest() -> (
    None
):
    log: list[str] = []
    llm = _LoggingLLM(log, [_calls("echo"), _answer("ok")])
    chain = [_Recorder("A", log), _Recorder("B", log)]
    events = await _events(_agent(llm, chain, registry=_registry(log)))
    assert events[-1]["content"] == "ok"
    turn = [
        "A.before_model",
        "B.before_model",
        "A>model",
        "B>model",
        "model",
        "B<model",
        "A<model",
        "B.after_model",
        "A.after_model",
    ]
    assert log == [
        "A.before_agent",
        "B.before_agent",
        *turn,
        "A>tool",
        "B>tool",
        "tool",
        "B<tool",
        "A<tool",
        *turn,
        "B.after_agent",
        "A.after_agent",
    ]


@pytest.mark.asyncio
async def test_a_wrap_may_retry_or_short_circuit_its_handler() -> None:
    class Retry(AgentMiddleware):
        async def awrap_model_call(
            self,
            request: ModelRequest,
            handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
        ) -> ModelResponse:
            first = await handler(request)
            if first.result.content == "flaky":
                return await handler(request)
            return first

    llm = _Scripted([_answer("flaky"), _answer("steady")])
    events = await _events(_agent(llm, [Retry()]))
    assert events[-1]["content"] == "steady" and len(llm.calls) == 2

    class Canned(AgentMiddleware):
        async def awrap_model_call(
            self,
            request: ModelRequest,
            handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
        ) -> ChatResponse:
            return _answer("from the cache")  # a bare ChatResponse is accepted

    llm = _Scripted()
    events = await _events(_agent(llm, [Canned()]))
    assert events[-1]["content"] == "from the cache" and llm.calls == []

    class Bad(AgentMiddleware):
        async def awrap_model_call(self, request: Any, handler: Any) -> Any:
            return "text"

    with pytest.raises(MiddlewareError, match="Bad.awrap_model_call"):
        await _events(_agent(_Scripted(), [Bad()]))

    class BadTool(AgentMiddleware):
        async def awrap_tool_call(self, request: Any, handler: Any) -> Any:
            return "text"

    with pytest.raises(MiddlewareError, match="BadTool.awrap_tool_call"):
        await _events(_agent(_Scripted([_calls("echo")]), [BadTool()]))


# --- overrides ----------------------------------------------------------------


def _request(**kw: Any) -> ModelRequest:
    runtime = Runtime(context=RunContext(brief="b"), emit=lambda _e: None)
    base: dict[str, Any] = {
        "messages": [Message(role="user", content="b")],
        "system_message": Message(role="system", content="sys"),
        "tools": [ToolSpec("echo", "echo", _SCHEMA)],
        "state": {"messages": []},
        "runtime": runtime,
    }
    return ModelRequest(**{**base, **kw})


def test_override_returns_a_copy_and_leaves_the_request() -> None:
    request = _request()
    changed = request.override(tools=None, model_settings={"mode": REVISION_MODE})
    assert changed.tools is None and changed.model_settings == {"mode": REVISION_MODE}
    assert request.tools is not None and request.model_settings == {}
    assert changed.messages is request.messages and changed.state is request.state
    prompt = request.override(system_prompt="other")
    assert prompt.system_message == Message(role="system", content="other")
    assert prompt.system_prompt == "other" and request.system_prompt == "sys"
    assert request.override(system_prompt=None).system_message is None
    with pytest.raises(ValueError, match="both"):
        request.override(system_prompt="x", system_message=None)
    with pytest.raises(AttributeError):
        request.tools = None  # type: ignore[misc]


@pytest.mark.asyncio
async def test_an_overridden_model_request_is_what_the_model_gets() -> None:
    class Rewrite(AgentMiddleware):
        async def awrap_model_call(
            self,
            request: ModelRequest,
            handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
        ) -> ModelResponse:
            return await handler(
                request.override(
                    system_prompt="short",
                    messages=[*request.messages, Message(role="user", content="more")],
                    tools=None,
                    model_settings={"mode": REVISION_MODE},
                )
            )

    llm = _Scripted([_answer("ok")])
    await _events(_agent(llm, [Rewrite()]), "brief")
    messages, tools, kw = llm.calls[0]
    assert [(m.role, m.content) for m in messages] == [
        ("system", "short"),
        ("user", "brief"),
        ("user", "more"),
    ]
    assert tools is None and kw == {"mode": REVISION_MODE}

    class Choose(AgentMiddleware):
        async def awrap_model_call(
            self,
            request: ModelRequest,
            handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
        ) -> ModelResponse:
            return await handler(request.override(tool_choice="required"))

    with pytest.raises(MiddlewareError, match="tool_choice"):
        await _events(_agent(_Scripted(), [Choose()]))


@pytest.mark.asyncio
async def test_an_overridden_tool_call_is_what_runs() -> None:
    seen: list[ToolCallRequest] = []

    class Args(AgentMiddleware):
        async def awrap_tool_call(
            self,
            request: ToolCallRequest,
            handler: Callable[[ToolCallRequest], Awaitable[ToolResult]],
        ) -> ToolResult:
            seen.append(request)
            call = request.tool_call
            fixed = ToolCall(id=call.id, name=call.name, arguments={"x": 2})
            return await handler(request.override(tool_call=fixed))

    llm = _Scripted([_calls("echo", "nosuch"), _answer("ok")])
    events = await _events(_agent(llm, [Args()]))
    results = [e for e in events if e["type"] == "tool_result"]
    assert results[0]["result"] == {"ok": True, "result": {"x": 2}}
    assert seen[0].tool_call.arguments == {} and seen[0].tool == ToolSpec(
        "echo", "echo", _SCHEMA
    )
    # A tool the registry does not know comes with no spec.
    assert seen[1].tool is None and results[1]["ok"] is False


# --- jumps --------------------------------------------------------------------


@pytest.mark.asyncio
async def test_before_model_may_end_the_run_before_the_call() -> None:
    log: list[str] = []

    class Stop(AgentMiddleware):
        @hook_config(can_jump_to=["end"])
        async def abefore_model(self, state: AgentState, runtime: Runtime) -> Any:
            return {"jump_to": "end", "answer": "stopped"}

    llm = _Scripted()
    events = await _events(_agent(llm, [_Recorder("R", log), Stop()]))
    assert llm.calls == [] and events == [
        {
            "type": "final",
            "turn": 1,
            "content": "stopped",
            "forced_by_turn_cap": False,
            "grounding_revised": False,
            "revised": False,
            "marked": [],
        }
    ]
    assert log == ["R.before_agent", "R.before_model", "R.after_agent"]


@pytest.mark.asyncio
async def test_after_model_may_send_the_run_back_to_the_model_in_the_same_turn() -> (
    None
):
    class Again(AgentMiddleware):
        @hook_config(can_jump_to=["model"])
        async def aafter_model(self, state: AgentState, runtime: Runtime) -> Any:
            if state.get("answer") == "draft":
                note = Message(role="user", content="once more")
                return {"messages": [note], "jump_to": "model"}
            return None

    llm = _Scripted([_calls("echo"), _answer("draft"), _answer("final")])
    events = await _events(_agent(llm, [Again()]))
    assert events[-1]["content"] == "final" and events[-1]["turn"] == 2
    messages, _tools, _kw = llm.calls[2]
    assert [(m.role, m.content) for m in messages[-2:]] == [
        ("assistant", "draft"),
        ("user", "once more"),
    ]


@pytest.mark.asyncio
async def test_a_jump_ends_its_phase_and_end_skips_the_tools() -> None:
    log: list[str] = []

    class End(AgentMiddleware):
        @hook_config(can_jump_to=["end"])
        async def aafter_model(self, state: AgentState, runtime: Runtime) -> Any:
            return {"jump_to": "end"}

    llm = _Scripted([_calls("echo")])
    # after_model runs in reverse: End (last in the list) runs first, and its
    # jump means the recorder's after_model never runs.
    agent = _agent(llm, [_Recorder("R", log), End()], registry=_registry(log))
    events = await _events(agent)
    assert [e["type"] for e in events] == ["final"]
    assert "tool" not in log and "R.after_model" not in log
    assert log[-1] == "R.after_agent"


@pytest.mark.asyncio
async def test_before_and_after_agent_may_jump() -> None:
    class Skip(AgentMiddleware):
        @hook_config(can_jump_to=["end"])
        async def abefore_agent(self, state: AgentState, runtime: Runtime) -> Any:
            return {"jump_to": "end"}

    llm = _Scripted()
    events = await _events(_agent(llm, [Skip()]))
    assert llm.calls == [] and events[-1]["content"] is None

    class OneMore(AgentMiddleware):
        @hook_config(can_jump_to=["model"])
        async def aafter_agent(self, state: AgentState, runtime: Runtime) -> Any:
            if state.get("answer") == "first":
                return {"jump_to": "model"}
            return None

    llm = _Scripted([_answer("first"), _answer("second")])
    events = await _events(_agent(llm, [OneMore()]))
    assert len(llm.calls) == 2 and events[-1]["content"] == "second"

    class Tools(AgentMiddleware):
        @hook_config(can_jump_to=["tools"])
        async def abefore_agent(self, state: AgentState, runtime: Runtime) -> Any:
            return {"jump_to": "tools"}

    llm = _Scripted([_answer("ok")])  # no calls to run: on to the model
    events = await _events(_agent(llm, [Tools()]))
    assert events[-1]["content"] == "ok" and events[-1]["turn"] == 1


@pytest.mark.asyncio
async def test_a_jump_must_be_declared() -> None:
    class Undeclared(AgentMiddleware):
        async def aafter_model(self, state: AgentState, runtime: Runtime) -> Any:
            return {"jump_to": "model"}

    with pytest.raises(MiddlewareError, match=r"Undeclared.aafter_model.*hook_config"):
        await _events(_agent(_Scripted(), [Undeclared()]))

    class Elsewhere(AgentMiddleware):
        @hook_config(can_jump_to=["end"])
        async def abefore_model(self, state: AgentState, runtime: Runtime) -> Any:
            return {"jump_to": "model"}

    with pytest.raises(MiddlewareError, match="without declaring"):
        await _events(_agent(_Scripted(), [Elsewhere()]))

    class Nowhere(AgentMiddleware):
        @hook_config(can_jump_to=["end"])
        async def abefore_model(self, state: AgentState, runtime: Runtime) -> Any:
            return {"jump_to": "start"}

    with pytest.raises(MiddlewareError, match="destinations"):
        await _events(_agent(_Scripted(), [Nowhere()]))

    with pytest.raises(ValueError, match="destinations"):
        hook_config(can_jump_to=["start"])  # type: ignore[list-item]

    class NotADict(AgentMiddleware):
        async def abefore_model(self, state: AgentState, runtime: Runtime) -> Any:
            return ["jump_to", "end"]

    with pytest.raises(MiddlewareError, match="not a dict"):
        await _events(_agent(_Scripted(), [NotADict()]))


def test_a_middleware_appears_once_and_is_a_middleware() -> None:
    with pytest.raises(ValueError, match="more than once"):
        _agent(_Scripted(), [RetryHintMiddleware(), RetryHintMiddleware()])
    with pytest.raises(TypeError, match="not an AgentMiddleware"):
        MiddlewareChain([object()])  # type: ignore[list-item]


# --- events -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_emitted_events_are_relayed_in_order_with_the_loops_own() -> None:
    class Announce(AgentMiddleware):
        async def abefore_model(self, state: AgentState, runtime: Runtime) -> None:
            runtime.emit({"type": "note", "where": "before_model"})

        async def awrap_tool_call(
            self,
            request: ToolCallRequest,
            handler: Callable[[ToolCallRequest], Awaitable[ToolResult]],
        ) -> ToolResult:
            runtime = request.runtime
            runtime.emit({"type": "note", "where": "tool"})
            return await handler(request)

        async def aafter_agent(self, state: AgentState, runtime: Runtime) -> None:
            runtime.emit({"type": "note", "where": "after_agent"})

    llm = _Scripted(
        [ChatResponse("", [ToolCall("c", "echo", {})], "hm"), _answer("ok")]
    )
    events = await _events(_agent(llm, [Announce()]))
    assert [(e["type"], e.get("where")) for e in events] == [
        ("note", "before_model"),
        ("thinking", None),
        ("tool_call", None),
        ("note", "tool"),
        ("tool_result", None),
        ("note", "before_model"),
        ("note", "after_agent"),
        ("final", None),
    ]


class _Waits:
    """A model call that returns only once the consumer has seen an event."""

    def __init__(self) -> None:
        self.seen = asyncio.Event()
        self.cancelled = False

    async def chat(
        self, messages: list[Message], *, tools: Any = None, **kw: Any
    ) -> ChatResponse:
        try:
            await self.seen.wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        return _answer("ok")


class _Starting(AgentMiddleware):
    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        request.runtime.emit({"type": "starting"})
        return await handler(request)


@pytest.mark.asyncio
async def test_an_event_emitted_during_a_model_call_arrives_before_it_returns() -> None:
    llm = _Waits()
    agent = _agent(llm, [_Starting()])

    async def consume() -> list[str]:
        kinds = []
        async for event in agent.run_stream("go"):
            kinds.append(event["type"])
            if event["type"] == "starting":
                llm.seen.set()  # the call can only return after this
        return kinds

    assert await asyncio.wait_for(consume(), timeout=5) == ["starting", "final"]


@pytest.mark.asyncio
async def test_stopping_the_stream_mid_call_cancels_the_call() -> None:
    llm = _Waits()
    stream = _agent(llm, [_Starting()]).run_stream("go")
    first = await asyncio.wait_for(stream.__anext__(), timeout=5)
    assert first["type"] == "starting"
    await asyncio.wait_for(stream.aclose(), timeout=5)  # type: ignore[attr-defined]
    assert llm.cancelled


# --- the built-in middleware ----------------------------------------------------


def test_the_default_chain_is_cap_retry_checks_with_the_agents_switches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OLMOEARTH_CHECK_NUMBERS", raising=False)
    monkeypatch.delenv("OLMOEARTH_CHECK_ANSWER", raising=False)
    agent = LeadAgent(_Scripted(), _registry(), studio=None, surface="web")  # type: ignore[arg-type]
    chain = agent.default_middleware()
    assert [type(m) for m in chain] == [
        TurnCapMiddleware,
        RetryHintMiddleware,
        AnswerChecksMiddleware,
    ]
    checks = chain[2]
    assert isinstance(checks, AnswerChecksMiddleware)
    assert checks.surface == "web" and checks.checks[0] == "numbers"
    agent.check_numbers = False
    later = agent.default_middleware()[2]
    assert isinstance(later, AnswerChecksMiddleware) and "numbers" not in later.checks
    with pytest.raises(ValueError, match="unknown checks"):
        AnswerChecksMiddleware(["spelling"])


@pytest.mark.asyncio
async def test_the_default_chain_given_explicitly_runs_the_same() -> None:
    def script() -> list[ChatResponse]:
        return [_calls("fail"), _calls("fail"), _calls("echo"), _answer("x")]

    implicit = await _events(_agent(_Scripted(script()), None), max_turns=2)
    agent = _agent(_Scripted(script()), None)
    explicit = _agent(_Scripted(script()), agent.default_middleware())
    assert await _events(explicit, max_turns=2) == implicit
    assert [e["type"] for e in implicit][-2:] == ["max_turns", "final"]


@pytest.mark.asyncio
async def test_without_the_retry_middleware_no_hint_is_added() -> None:
    llm = _Scripted([_calls("fail"), _calls("fail"), _answer("stopped")])
    events = await _events(_agent(llm, [TurnCapMiddleware()]))
    results = [e["result"] for e in events if e["type"] == "tool_result"]
    assert [r.get("same_error_count") for r in results] == [None, None]
    assert all("Stop retrying" not in r["hint"] for r in results)

    llm = _Scripted([_calls("fail"), _calls("fail"), _answer("stopped")])
    events = await _events(_agent(llm, [RetryHintMiddleware()]))
    results = [e["result"] for e in events if e["type"] == "tool_result"]
    assert [r.get("same_error_count") for r in results] == [None, 2]


@pytest.mark.asyncio
async def test_the_registry_leaves_the_hint_to_the_middleware_when_asked() -> None:
    registry = _registry()
    ctx = ToolContext(studio=None, state=ThreadState())  # type: ignore[arg-type]
    call = ToolCall(id="c", name="fail", arguments={})
    for _ in range(2):
        out = await registry.dispatch(call, ctx, retry_hint=False)
        assert "same_error_count" not in out
    assert ctx.state.tool_failures == {}
    # An unknown tool is never counted, by either path.
    unknown = ToolCall(id="u", name="nosuch", arguments={})
    for _ in range(2):
        assert "same_error_count" not in await registry.dispatch(unknown, ctx)
    assert ctx.state.tool_failures == {}


@pytest.mark.asyncio
async def test_a_cap_below_one_turn_forces_the_first_call() -> None:
    llm = _Scripted([_calls("echo")])
    events = await _events(_agent(llm, None), max_turns=0)
    assert [e["type"] for e in events] == ["max_turns", "final"]
    assert events[0]["turns"] == 0 and events[-1]["turn"] == 1
    assert "turn cap of 0" in events[-1]["content"]
    messages, tools, _kw = llm.calls[0]
    assert tools is None and messages[-1].content == TURN_CAP_PROMPT


@pytest.mark.asyncio
async def test_a_chain_without_the_turn_cap_stops_past_it() -> None:
    llm = _Scripted([_calls("echo")] * 5)
    with pytest.raises(MiddlewareError, match="turn cap of 2"):
        await _events(_agent(llm, []), max_turns=2)
    assert len(llm.calls) == 3  # turns 1-3; turn 4 is refused


@pytest.mark.asyncio
async def test_a_hook_that_always_jumps_back_is_stopped() -> None:
    from olmoearth_agent.harness.agent import MODEL_STEPS_PER_TURN

    class Forever(AgentMiddleware):
        @hook_config(can_jump_to=["model"])
        async def aafter_model(self, state: AgentState, runtime: Runtime) -> Any:
            return {"jump_to": "model"}

    llm = _Scripted()
    with pytest.raises(MiddlewareError, match="went back to the model"):
        await _events(_agent(llm, [Forever()]))
    assert len(llm.calls) == MODEL_STEPS_PER_TURN

    class Spin(AgentMiddleware):
        @hook_config(can_jump_to=["model"])
        async def abefore_model(self, state: AgentState, runtime: Runtime) -> Any:
            return {"jump_to": "model"}

    llm = _Scripted()
    with pytest.raises(MiddlewareError, match="went back to the model"):
        await _events(_agent(llm, [Spin()]))
    assert llm.calls == []
