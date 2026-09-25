# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Middleware for the lead-agent loop, with LangChain 1.x's hook interface.

On 25 September 2026 the owner chose to build this layer rather than move the
harness to LangChain. The interface is LangChain's, so that a later move is a
mechanical replacement. It follows ``langchain.agents.middleware.types``
(langchain 1.x) and ``langgraph.prebuilt.tool_node.ToolCallRequest``:

- The hooks have LangChain's async names and arguments:
  ``abefore_agent(state, runtime)``, ``abefore_model(state, runtime)``,
  ``awrap_model_call(request, handler)``, ``aafter_model(state, runtime)``,
  ``awrap_tool_call(request, handler)`` and ``aafter_agent(state, runtime)``.
- A node hook (``abefore_*``, ``aafter_*``) returns ``None`` or a dict of state
  updates. The dict may hold ``"jump_to"``: ``"model"``, ``"tools"`` or
  ``"end"``. A hook that jumps declares where with
  ``@hook_config(can_jump_to=[...])``. A jump the hook did not declare raises
  :class:`MiddlewareError`, and so does an unknown destination.
- The ``abefore_*`` hooks run in list order and the ``aafter_*`` hooks in
  reverse. The ``awrap_*`` hooks nest, with the first middleware outermost. A
  jump ends its phase: the hooks after it in that phase do not run.
- A wrap hook gets the request and a ``handler``. It may change the request
  with ``request.override(...)``, call the handler once, more than once or not
  at all, and change what the handler returns.

What differs from LangChain:

- Only the async hooks exist.
- The types are this package's own. A message is a
  :class:`~olmoearth_agent.llm.types.Message` and a model response a
  :class:`~olmoearth_agent.llm.types.ChatResponse`. A tool is its
  :class:`~olmoearth_agent.llm.types.ToolSpec`, and a wrapped tool call
  returns the registry's result envelope (``{"ok": ..., ...}``), not a
  ``ToolMessage``.
- :class:`ModelRequest` has no ``model`` and no ``response_format``. Its
  ``model_settings`` are the keyword arguments of ``OlmoEarthLLM.chat``: our
  sampling mode is ``{"mode": ...}``. ``tools=None`` offers the model no tools.
- A ``"messages"`` update appends. Our messages have no ids to replace by.
- The runtime's ``emit(event)`` stands in for LangGraph's ``stream_writer``.
  ``LeadAgent.run_stream`` relays each event in order, as it is emitted, even
  from inside a wrapped model or tool call.

The state is a dict (:class:`AgentState`): the conversation without the
system message, the run's :class:`~olmoearth_agent.harness.state.ThreadState`,
the turn and the answer so far, and each middleware's own keys. ``jump_to``
is ephemeral: it is read from a hook's return and never kept.
"""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import AsyncGenerator, Awaitable, Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Literal, TypedDict, TypeVar, cast

from olmoearth_agent.llm.types import ChatResponse, Message, ToolCall, ToolSpec

if TYPE_CHECKING:
    from olmoearth_agent.harness.checks import ToolRecord
    from olmoearth_agent.harness.state import ThreadState

__all__ = [
    "JUMP_TARGETS",
    "AgentMiddleware",
    "AgentState",
    "EventRelay",
    "JumpTo",
    "MiddlewareChain",
    "MiddlewareError",
    "ModelRequest",
    "ModelResponse",
    "RunContext",
    "Runtime",
    "ToolCallRequest",
    "ToolResult",
    "hook_config",
    "last_ai_message",
    "without_tool_calls",
]

#: Where a node hook may send the run next (LangChain's ``JumpTo``).
JumpTo = Literal["tools", "model", "end"]
JUMP_TARGETS: tuple[JumpTo, ...] = ("tools", "model", "end")

#: What a wrapped tool call returns: the registry's result envelope.
ToolResult = dict[str, Any]

_F = TypeVar("_F", bound=Callable[..., Any])


class MiddlewareError(RuntimeError):
    """A middleware broke the interface: an undeclared jump, a bad return."""


class AgentState(TypedDict, total=False):
    """The state every hook reads, and node hooks update.

    The run's keys, which the runner keeps:

    - ``messages``: the conversation without the system message (the history,
      the brief, then each model turn and tool result). A ``"messages"``
      update appends.
    - ``thread_state``: the run's :class:`ThreadState`, which the tools share.
    - ``turn``: the current turn, from 1. The turn moves on when the model is
      called at the start of the run or after the tools. A jump back to the
      model repeats the turn.
    - ``answer``: the text the run will show, first the last model
      response's content. A middleware may rewrite it.
    - ``tool_records``: every tool call with its full result envelope, as
      the answer checks read them (the model saw spilled results compacted).

    The final event's flags, which middleware set (False or empty when none
    does): ``forced_by_turn_cap``, ``revised``, ``grounding_revised`` and
    ``marked``.

    The built-in middleware's own keys: ``turn_cap``
    (:class:`~olmoearth_agent.harness.turn_cap.TurnCapMiddleware`) and
    ``answer_checks``, ``answer_checks_draft`` and ``answer_checks_numbers``
    (:class:`~olmoearth_agent.harness.answer_checks.AnswerChecksMiddleware`).
    """

    messages: list[Message]
    thread_state: ThreadState
    turn: int
    answer: str | None
    tool_records: list[ToolRecord]
    forced_by_turn_cap: bool
    revised: bool
    grounding_revised: bool
    marked: list[str]
    turn_cap: str
    answer_checks: str
    answer_checks_draft: str | None
    answer_checks_numbers: bool


@dataclass(frozen=True)
class RunContext:
    """The run's inputs, as ``LeadAgent.run_stream`` received them."""

    brief: str
    history: list[Message] = field(default_factory=list)
    max_turns: int = 8


@dataclass(frozen=True)
class Runtime:
    """What a hook gets besides the state: the run's inputs and ``emit``.

    ``emit(event)`` publishes one event (a JSON-serializable dict with a
    ``type``). ``LeadAgent.run_stream`` yields it in order, as it happens.
    """

    context: RunContext
    emit: Callable[[dict[str, Any]], None]


@dataclass(frozen=True, kw_only=True)
class ModelRequest:
    """One model call, as ``awrap_model_call`` sees it.

    ``messages`` exclude the system message, which is ``system_message``.
    ``tools`` are the specs offered, or ``None`` for none. ``model_settings``
    are the keyword arguments of ``OlmoEarthLLM.chat``, such as
    ``{"mode": REVISION_MODE}``. ``tool_choice`` is kept for LangChain's shape,
    but our client takes none: a request that sets it is refused.
    """

    messages: list[Message]
    system_message: Message | None
    tools: list[ToolSpec] | None
    state: AgentState
    runtime: Runtime
    tool_choice: Any | None = None
    model_settings: dict[str, Any] = field(default_factory=dict)

    @property
    def system_prompt(self) -> str | None:
        """The system message's text, or ``None``."""
        return None if self.system_message is None else self.system_message.content

    def override(self, **changes: Any) -> ModelRequest:
        """A copy with ``changes`` applied; this request is left as it is.

        ``system_prompt="..."`` sets ``system_message`` from text, as in
        LangChain.
        """
        if "system_prompt" in changes:
            if "system_message" in changes:
                raise ValueError("Cannot specify both system_prompt and system_message")
            text = changes.pop("system_prompt")
            changes["system_message"] = (
                None if text is None else Message(role="system", content=text)
            )
        return replace(self, **changes)


@dataclass(frozen=True)
class ModelResponse:
    """What a model call returns: our ``ChatResponse`` as ``result``."""

    result: ChatResponse
    structured_response: Any = None


@dataclass(frozen=True, kw_only=True)
class ToolCallRequest:
    """One tool call, as ``awrap_tool_call`` sees it.

    ``tool`` is the called tool's spec, or ``None`` when no tool of that name
    is registered (the registry then answers with an ``unknown tool`` error).
    """

    tool_call: ToolCall
    tool: ToolSpec | None
    state: AgentState
    runtime: Runtime

    def override(self, **changes: Any) -> ToolCallRequest:
        """A copy with ``changes`` applied; this request is left as it is."""
        return replace(self, **changes)


def hook_config(*, can_jump_to: Sequence[JumpTo] | None = None) -> Callable[[_F], _F]:
    """Declare where a node hook may jump (LangChain's ``hook_config``).

    Examples
    --------
    >>> class Stop(AgentMiddleware):
    ...     @hook_config(can_jump_to=["end"])
    ...     async def abefore_model(self, state, runtime):
    ...         return {"jump_to": "end"}
    """
    if can_jump_to is not None:
        unknown = [t for t in can_jump_to if t not in JUMP_TARGETS]
        if unknown:
            raise ValueError(
                f"can_jump_to holds {unknown}; the destinations are {list(JUMP_TARGETS)}"
            )

    def decorator(func: _F) -> _F:
        if can_jump_to is not None:
            func.__can_jump_to__ = list(can_jump_to)  # type: ignore[attr-defined]
        return func

    return decorator


class AgentMiddleware:
    """Base class: override the hooks a middleware needs.

    The defaults do nothing: a node hook returns ``None`` and a wrap hook
    calls its handler. Only the hooks a subclass overrides are run.
    """

    @property
    def name(self) -> str:
        """The middleware's name: its class name, unless overridden."""
        return type(self).__name__

    async def abefore_agent(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        """Once, before the first model call."""
        return None

    async def abefore_model(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        """Before each model call."""
        return None

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse | ChatResponse:
        """Around each model call; ``handler(request)`` makes the call."""
        return await handler(request)

    async def aafter_model(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        """After each model call; its response is the last assistant message."""
        return None

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolResult]],
    ) -> ToolResult:
        """Around each tool call; ``handler(request)`` runs the tool."""
        return await handler(request)

    async def aafter_agent(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        """Once, after the last model call, before the final event."""
        return None


def last_ai_message(state: AgentState) -> Message | None:
    """The last assistant message of the state, or ``None``."""
    for message in reversed(state.get("messages", [])):
        if message.role == "assistant":
            return message
    return None


def without_tool_calls(response: ModelResponse) -> ModelResponse:
    """The response with any tool call dropped (the model was offered none)."""
    if not response.result.tool_calls:
        return response
    return replace(response, result=replace(response.result, tool_calls=[]))


_NODE_HOOKS = ("abefore_agent", "abefore_model", "aafter_model", "aafter_agent")


def _overrides(middleware: AgentMiddleware, hook: str) -> bool:
    return getattr(type(middleware), hook) is not getattr(AgentMiddleware, hook)


def _as_model_response(value: Any, where: str) -> ModelResponse:
    if isinstance(value, ModelResponse):
        return value
    if isinstance(value, ChatResponse):
        return ModelResponse(result=value)
    raise MiddlewareError(
        f"{where} returned {type(value).__name__}, not a ModelResponse or ChatResponse"
    )


def _apply(state: AgentState, updates: dict[str, Any]) -> None:
    """Apply a node hook's updates: ``messages`` appends, any other key replaces."""
    target = cast("dict[str, Any]", state)
    for key, value in updates.items():
        if key == "jump_to":
            continue
        if key == "messages":
            target.setdefault("messages", []).extend(value)
        else:
            target[key] = value


class MiddlewareChain:
    """Runs a list of middleware in LangChain's order.

    ``abefore_*`` hooks run in list order, ``aafter_*`` hooks in reverse, and
    ``awrap_*`` hooks nest with the first middleware outermost. A node hook's
    jump ends its phase and is returned to the runner.
    """

    def __init__(self, middleware: Sequence[AgentMiddleware]) -> None:
        self.middleware = list(middleware)
        for m in self.middleware:
            if not isinstance(m, AgentMiddleware):
                raise TypeError(f"{m!r} is not an AgentMiddleware")
        names = [m.name for m in self.middleware]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ValueError(
                f"middleware {duplicates} appear more than once; each may appear once"
            )

        def using(hook: str) -> list[AgentMiddleware]:
            return [m for m in self.middleware if _overrides(m, hook)]

        self._before_agent = using("abefore_agent")
        self._before_model = using("abefore_model")
        self._after_model = using("aafter_model")[::-1]
        self._after_agent = using("aafter_agent")[::-1]
        self._model_wraps = using("awrap_model_call")
        self._tool_wraps = using("awrap_tool_call")

    async def before_agent(self, state: AgentState, runtime: Runtime) -> JumpTo | None:
        """The ``abefore_agent`` hooks, in list order; the jump, if any."""
        return await self._nodes("abefore_agent", self._before_agent, state, runtime)

    async def before_model(self, state: AgentState, runtime: Runtime) -> JumpTo | None:
        """The ``abefore_model`` hooks, in list order; the jump, if any."""
        return await self._nodes("abefore_model", self._before_model, state, runtime)

    async def after_model(self, state: AgentState, runtime: Runtime) -> JumpTo | None:
        """The ``aafter_model`` hooks, in reverse order; the jump, if any."""
        return await self._nodes("aafter_model", self._after_model, state, runtime)

    async def after_agent(self, state: AgentState, runtime: Runtime) -> JumpTo | None:
        """The ``aafter_agent`` hooks, in reverse order; the jump, if any."""
        return await self._nodes("aafter_agent", self._after_agent, state, runtime)

    async def call_model(
        self,
        request: ModelRequest,
        call: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """``call`` wrapped by every ``awrap_model_call``, the first outermost."""

        async def base(req: ModelRequest) -> ModelResponse:
            return _as_model_response(await call(req), "the model call")

        handler: Callable[[ModelRequest], Awaitable[ModelResponse]] = base
        for m in reversed(self._model_wraps):
            handler = self._model_layer(m, handler)
        return await handler(request)

    async def call_tool(
        self,
        request: ToolCallRequest,
        call: Callable[[ToolCallRequest], Awaitable[ToolResult]],
    ) -> ToolResult:
        """``call`` wrapped by every ``awrap_tool_call``, the first outermost."""
        handler = call
        for m in reversed(self._tool_wraps):
            handler = self._tool_layer(m, handler)
        return await handler(request)

    @staticmethod
    def _model_layer(
        m: AgentMiddleware, inner: Callable[[ModelRequest], Awaitable[ModelResponse]]
    ) -> Callable[[ModelRequest], Awaitable[ModelResponse]]:
        async def handler(request: ModelRequest) -> ModelResponse:
            result = await m.awrap_model_call(request, inner)
            return _as_model_response(result, f"{m.name}.awrap_model_call")

        return handler

    @staticmethod
    def _tool_layer(
        m: AgentMiddleware, inner: Callable[[ToolCallRequest], Awaitable[ToolResult]]
    ) -> Callable[[ToolCallRequest], Awaitable[ToolResult]]:
        async def handler(request: ToolCallRequest) -> ToolResult:
            result = await m.awrap_tool_call(request, inner)
            if not isinstance(result, dict):
                raise MiddlewareError(
                    f"{m.name}.awrap_tool_call returned {type(result).__name__}, "
                    "not a result envelope (a dict)"
                )
            return result

        return handler

    @staticmethod
    async def _nodes(
        hook: str,
        middleware: list[AgentMiddleware],
        state: AgentState,
        runtime: Runtime,
    ) -> JumpTo | None:
        for m in middleware:
            fn = getattr(m, hook)
            updates = await fn(state, runtime)
            if updates is None:
                continue
            if not isinstance(updates, dict):
                raise MiddlewareError(
                    f"{m.name}.{hook} returned {type(updates).__name__}, "
                    "not a dict of state updates or None"
                )
            jump = updates.get("jump_to")
            if jump is not None:
                if jump not in JUMP_TARGETS:
                    raise MiddlewareError(
                        f"{m.name}.{hook} jumps to {jump!r}; the destinations "
                        f"are {list(JUMP_TARGETS)}"
                    )
                if jump not in getattr(fn, "__can_jump_to__", ()):
                    raise MiddlewareError(
                        f"{m.name}.{hook} jumps to {jump!r} without declaring it: "
                        f"decorate the hook with @hook_config(can_jump_to=[{jump!r}])"
                    )
            _apply(state, updates)
            if jump is not None:
                return cast(JumpTo, jump)
        return None


class EventRelay:
    """Carries the events middleware emit to ``run_stream``, as they happen.

    :meth:`emit` queues an event. :meth:`run` awaits one step of the run in
    its own task and yields each queued event as soon as it is emitted, so a
    consumer sees an event emitted inside a model call before the call
    returns. The step's result is left in :attr:`value`. If the consumer
    stops early, or the run is cancelled, the step is cancelled and awaited.
    """

    def __init__(self) -> None:
        self._events: deque[dict[str, Any]] = deque()
        self._wake: asyncio.Future[None] | None = None
        #: The result of the last step :meth:`run` completed.
        self.value: Any = None

    def emit(self, event: dict[str, Any]) -> None:
        """Queue one event for ``run_stream`` to yield."""
        self._events.append(event)
        if self._wake is not None and not self._wake.done():
            self._wake.set_result(None)

    async def run(self, step: Awaitable[Any]) -> AsyncGenerator[dict[str, Any], None]:
        """Await ``step``, yielding every event emitted meanwhile, in order."""
        self.value = None
        task = asyncio.ensure_future(step)
        try:
            while True:
                while self._events:
                    yield self._events.popleft()
                if task.done():
                    break
                self._wake = asyncio.get_running_loop().create_future()
                try:
                    await asyncio.wait(
                        (task, self._wake), return_when=asyncio.FIRST_COMPLETED
                    )
                finally:
                    self._wake = None
            self.value = task.result()
        finally:
            if not task.done():
                task.cancel()
                await asyncio.wait((task,))
            if task.done() and not task.cancelled():
                task.exception()  # retrieved: raised above, or moot on an early stop
