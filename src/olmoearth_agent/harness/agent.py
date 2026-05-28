# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 OlmoEarth Agent contributors
"""The lead-agent loop: brief -> LLM -> tool calls -> result.

A single-agent ReAct-style loop (DeerFlow v2's lead-agent shape, minus
subagents for now). The LLM sees the tool registry's specs; each emitted
tool call is dispatched and its result fed back, until the model returns
a plain-text answer or the turn budget is exhausted.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from olmoearth_agent.llm.client import OlmoEarthLLM
from olmoearth_agent.llm.types import Message
from olmoearth_agent.studio.client import StudioClient
from olmoearth_agent.tools.registry import ToolContext, ToolRegistry
from olmoearth_agent.harness.state import ThreadState

DEFAULT_SYSTEM_PROMPT = (
    "You are the OlmoEarth Studio agent. You help Earth-observation "
    "researchers run studies on the OlmoEarth Studio platform by calling "
    "the provided tools.\n"
    "Rules:\n"
    "- Call olmoearth_load_context first to see the user's existing "
    "projects before creating anything.\n"
    "- Never invent project, area, dataset, model, or prediction IDs. "
    "Discover them with tools.\n"
    "- Do not create a project whose name already exists; reuse it.\n"
    "- Never print raw latitude/longitude or full GeoJSON in your replies; "
    "reference a saved path or an id instead.\n"
    "- When the task is complete, stop calling tools and reply with a short "
    "plain-text summary of what you did and the ids involved."
)


@dataclass
class AgentResult:
    """Outcome of one :meth:`LeadAgent.run`."""

    final_content: str | None
    turns: int
    tool_calls: list[tuple[str, bool]] = field(default_factory=list)
    hit_max_turns: bool = False
    state: ThreadState | None = None


class LeadAgent:
    """Drives a natural-language brief to completion via tool calls.

    Examples
    --------
    >>> import asyncio
    >>> async def main():
    ...     async with StudioClient.from_env() as studio:
    ...         agent = LeadAgent(OlmoEarthLLM(), build_registry(), studio)
    ...         result = await agent.run("List my OlmoEarth projects.")
    ...         print(result.final_content)
    >>> asyncio.run(main())  # doctest: +SKIP
    """

    def __init__(
        self,
        llm: OlmoEarthLLM,
        registry: ToolRegistry,
        studio: StudioClient,
        *,
        state: ThreadState | None = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> None:
        self.llm = llm
        self.registry = registry
        self.studio = studio
        self.state = state or ThreadState()
        self.system_prompt = system_prompt

    async def run(self, brief: str, *, max_turns: int = 8) -> AgentResult:
        """Run the agent loop until it answers or hits ``max_turns``.

        Parameters
        ----------
        brief
            The user's natural-language request.
        max_turns
            Hard cap on LLM round-trips, to bound cost and stop loops.

        Returns
        -------
        AgentResult
            Final text (``None`` if the cap was hit), turn count, and the
            ``(tool_name, ok)`` trace of every dispatched call.
        """
        messages: list[Message] = [
            Message(role="system", content=self.system_prompt),
            Message(role="user", content=brief),
        ]
        ctx = ToolContext(studio=self.studio, state=self.state)
        calls: list[tuple[str, bool]] = []

        for turn in range(1, max_turns + 1):
            self.state.turn_count = turn
            response = await self.llm.chat(messages, tools=self.registry.specs())

            if not response.tool_calls:
                return AgentResult(
                    final_content=response.content,
                    turns=turn,
                    tool_calls=calls,
                    state=self.state,
                )

            messages.append(
                Message(
                    role="assistant",
                    content=response.content,
                    tool_calls=response.tool_calls,
                )
            )
            for call in response.tool_calls:
                result = await self.registry.dispatch(call, ctx)
                calls.append((call.name, bool(result.get("ok"))))
                messages.append(
                    Message(
                        role="tool",
                        tool_call_id=call.id,
                        name=call.name,
                        content=json.dumps(result),
                    )
                )

        return AgentResult(
            final_content=None,
            turns=max_turns,
            tool_calls=calls,
            hit_max_turns=True,
            state=self.state,
        )
