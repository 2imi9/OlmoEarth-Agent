# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""The turn cap, as middleware: at the cap, one more call without tools.

exp86 round 1: three runs ended at the cap with nothing to show. When the
last turn (``max_turns``) still asks for tools, its tools run, and then one
more call, with no tools and the harness's note, asks the model to answer
from what it has. Any tool call the model still emits is not run. If that
call returns no text, the answer is the harness's :func:`turn_cap_fallback`.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from olmoearth_agent.harness.middleware import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
    Runtime,
    last_ai_message,
    without_tool_calls,
)
from olmoearth_agent.llm.types import ChatResponse, Message

__all__ = ["TURN_CAP_PROMPT", "TurnCapMiddleware", "turn_cap_fallback"]

#: The harness's message to the model when the turn cap is reached with the
#: model still asking for tools; the answer call that follows offers none.
TURN_CAP_PROMPT = (
    "Harness note: this run has reached its turn cap, and no more tools can be "
    "called. Answer the request now from the tool results above. Say plainly "
    "what you could not do or find out, and why (for example a tool's limit or "
    "an error that repeated). State only what the tools returned."
)

#: ``state["turn_cap"]``: the cap is reached, and the forced call is next.
REACHED = "reached"
#: ``state["turn_cap"]``: the forced call has been asked for.
ASKED = "asked"


def turn_cap_fallback(max_turns: int) -> str:
    """The answer when the model writes no text even when offered no tools."""
    return (
        f"No answer was written: the run reached its turn cap of {max_turns} "
        "turns while still calling tools, and the model returned no text when "
        "asked to answer without them. The tool calls and their results are in "
        "the run's trace."
    )


class TurnCapMiddleware(AgentMiddleware):
    """Forces a tool-less answer once the run's ``max_turns`` are used.

    The cap is ``runtime.context.max_turns`` (``run_stream``'s argument).
    ``aafter_model`` notes the cap when turn ``max_turns`` asks for tools;
    those tools run. ``abefore_model`` then emits ``max_turns``, adds the
    harness's note, and numbers the call turn ``max_turns + 1``;
    ``awrap_model_call`` offers it no tools and drops any tool call it
    returns. ``aafter_agent`` puts :func:`turn_cap_fallback` in place of an
    empty forced answer. The final event's ``forced_by_turn_cap`` is set.
    """

    async def abefore_agent(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        """With a cap below one turn, the first call is already the forced one."""
        if runtime.context.max_turns < 1:
            return {"turn_cap": REACHED}
        return None

    async def aafter_model(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        """Note the cap when its last turn still asks for tools."""
        last = last_ai_message(state)
        if (
            state.get("turn_cap") is None
            and last is not None
            and last.tool_calls
            and state.get("turn", 0) >= runtime.context.max_turns
        ):
            return {"turn_cap": REACHED}
        return None

    async def abefore_model(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        """Before the forced call: emit ``max_turns`` and add the harness's note."""
        if state.get("turn_cap") != REACHED:
            return None
        max_turns = runtime.context.max_turns
        runtime.emit(
            {"type": "max_turns", "turns": max_turns, "final_answer_forced": True}
        )
        return {
            "turn_cap": ASKED,
            "turn": max_turns + 1,
            "forced_by_turn_cap": True,
            "messages": [Message(role="user", content=TURN_CAP_PROMPT)],
        }

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse | ChatResponse:
        """Offer the forced call no tools; a tool call it returns is not run."""
        if request.state.get("turn_cap") != ASKED:
            return await handler(request)
        return without_tool_calls(await handler(request.override(tools=None)))

    async def aafter_agent(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        """An empty forced answer becomes the harness's fallback text."""
        if state.get("forced_by_turn_cap") and not (state.get("answer") or "").strip():
            return {"answer": turn_cap_fallback(runtime.context.max_turns)}
        return None
