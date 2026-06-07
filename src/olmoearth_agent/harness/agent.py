# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""The lead-agent loop: brief -> LLM -> tool calls -> result.

A single-agent ReAct-style loop (DeerFlow v2's lead-agent shape, minus
subagents for now). The LLM sees the tool registry's specs; each emitted
tool call is dispatched and its result fed back, until the model returns
a plain-text answer or the turn budget is exhausted.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from olmoearth_agent.harness.state import ThreadState
from olmoearth_agent.llm.client import OlmoEarthLLM
from olmoearth_agent.llm.types import Message
from olmoearth_agent.security import egress
from olmoearth_agent.studio.client import StudioClient
from olmoearth_agent.tools.registry import ToolContext, ToolRegistry

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = (
    "You are the OlmoEarth Studio agent. You help Earth-observation "
    "researchers run studies on the OlmoEarth Studio platform by calling "
    "the provided tools.\n"
    "Rules:\n"
    "- Call olmoearth_load_context first to see the user's existing "
    "projects before creating anything.\n"
    "- Never invent project, area, dataset, model, or prediction IDs. "
    "Discover them with tools.\n"
    "- When a task needs a geographic area of interest (AOI) and the brief "
    "gives none (no area_id, bbox, or polygon), call olmoearth_request_aoi "
    "to let the user draw it on a map, instead of asking them to type "
    "coordinates. If the brief already provides an area_id or bbox, use it.\n"
    "- To compare two prediction results numerically when there are no "
    "ground-truth labels, call olmoearth_compare_results (it reports "
    "model-vs-model agreement: difference, correlation, agreement fraction) "
    "rather than only describing them. Use olmoearth_classification_metrics "
    "only when ground-truth labels exist (accuracy needs truth).\n"
    "- Do not create a project whose name already exists; reuse it.\n"
    "- Never print raw latitude/longitude or full GeoJSON in your replies; "
    "reference a saved path or an id instead.\n"
    "- When the task is complete, stop calling tools and reply with a concise "
    "answer in GitHub-flavored Markdown (use tables, **bold**, and lists where "
    "they help) summarizing what you did and the ids involved.\n"
    "- Do NOT use emoji or decorative pictographs (no star / coloured-circle / "
    "question-mark emoji). Use plain markers only, ✓, ✗, ~, or words "
    "(strong / moderate / weak / unclear)."
)

# Appended to the system prompt when the run is on the local model (small,
# limited output budget, and less reliable at following length/style rules than
# a hosted model). Cloud backends omit it - they can afford longer answers and
# obey "be concise". Restates the no-emoji rule last, where recency helps a
# weak model honor it.
LOCAL_BUDGET_CLAUSE = (
    "\n\nIMPORTANT - you are running on a small local model with a limited "
    "output budget, so a long answer gets cut off mid-sentence. Keep every "
    "reply short: a 2-3 sentence summary plus at most a 3-5 row table or a few "
    "bullets. Report only the top few items and offer to expand if the user "
    "wants more; never dump exhaustive lists or large tables. Plain text only - "
    "no emoji or decorative pictographs (use the plain markers above)."
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
        skill_index: str = "",
        local: bool = False,
    ) -> None:
        self.llm = llm
        self.registry = registry
        self.studio = studio
        self.state = state or ThreadState()
        self.system_prompt = system_prompt
        if skill_index:
            # Progressive disclosure: list the vendored SKILL.md skills so the
            # model knows to call olmoearth_load_skill when a task matches.
            self.system_prompt += (
                "\n\nAvailable instruction skills (call olmoearth_load_skill "
                "with the name to get full steps):\n" + skill_index
            )
        if local:
            # The local model needs an explicit output-budget + brevity reminder
            # (a hosted model does not), or long answers truncate mid-sentence.
            self.system_prompt += LOCAL_BUDGET_CLAUSE

    def _record_external_endpoints(self) -> None:
        """Note the external Studio endpoint this run uses in the manifest.

        Best-effort audit trail (host only, never raises): the provenance log
        then answers "what external host did this run contact". Enforcement of
        the endpoint happens earlier, at client construction (see
        ``security/egress.py``); here we only record.
        """
        try:
            base = getattr(self.studio.config, "base_url", None)
            if base:
                self.state.provenance.record_egress(
                    egress.check_endpoint(base, "studio")
                )
        except Exception:  # provenance bookkeeping must never break a run
            logger.debug(
                "could not record studio endpoint in provenance", exc_info=True
            )

    async def run_stream(
        self,
        brief: str,
        *,
        max_turns: int = 8,
        history: list[Message] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Run the agent loop, yielding one event per step as it happens.

        This is the streaming source of truth; :meth:`run` consumes it to
        assemble an :class:`AgentResult`. Each yielded event is a small,
        JSON-serializable dict tagged with a ``type``:

        - ``thinking``    : the model's reasoning for a turn (``text``).
        - ``tool_call``   : a dispatched call (``name``, ``arguments``, ``id``).
        - ``tool_result`` : its outcome (``name``, ``ok``, ``result``, ``id``).
        - ``final``       : the plain-text answer (``content``).
        - ``max_turns``   : the cap was hit with no answer (``turns``).

        Parameters
        ----------
        brief
            The user's natural-language request.
        max_turns
            Hard cap on LLM round-trips, to bound cost and stop loops.
        history
            Prior conversation turns (user/assistant messages) to seed before
            the new ``brief``, so multi-turn follow-ups have context. Inserted
            between the system prompt and the new user message.
        """
        messages: list[Message] = [Message(role="system", content=self.system_prompt)]
        if history:
            messages.extend(history)
        messages.append(Message(role="user", content=brief))
        ctx = ToolContext(studio=self.studio, state=self.state)
        self._record_external_endpoints()

        for turn in range(1, max_turns + 1):
            self.state.turn_count = turn
            response = await self.llm.chat(messages, tools=self.registry.specs())

            if response.thinking:
                yield {"type": "thinking", "turn": turn, "text": response.thinking}

            if not response.tool_calls:
                yield {"type": "final", "turn": turn, "content": response.content}
                return

            messages.append(
                Message(
                    role="assistant",
                    content=response.content,
                    tool_calls=response.tool_calls,
                )
            )
            for call in response.tool_calls:
                yield {
                    "type": "tool_call",
                    "turn": turn,
                    "id": call.id,
                    "name": call.name,
                    "arguments": call.arguments,
                }
                result = await self.registry.dispatch(call, ctx)
                self.state.provenance.record_tool_call(
                    call.name, call.arguments, result
                )
                yield {
                    "type": "tool_result",
                    "turn": turn,
                    "id": call.id,
                    "name": call.name,
                    "ok": bool(result.get("ok")),
                    "result": result,
                }
                messages.append(
                    Message(
                        role="tool",
                        tool_call_id=call.id,
                        name=call.name,
                        content=json.dumps(result),
                    )
                )

        yield {"type": "max_turns", "turns": max_turns}

    async def run(
        self,
        brief: str,
        *,
        max_turns: int = 8,
        history: list[Message] | None = None,
    ) -> AgentResult:
        """Run the agent loop until it answers or hits ``max_turns``.

        Thin collector over :meth:`run_stream`: drains the streamed events
        and assembles an :class:`AgentResult`.

        Parameters
        ----------
        brief
            The user's natural-language request.
        max_turns
            Hard cap on LLM round-trips, to bound cost and stop loops.
        history
            Prior conversation turns to seed before ``brief`` (see
            :meth:`run_stream`).

        Returns
        -------
        AgentResult
            Final text (``None`` if the cap was hit), turn count, and the
            ``(tool_name, ok)`` trace of every dispatched call.
        """
        final_content: str | None = None
        turns = 0
        calls: list[tuple[str, bool]] = []
        hit_max_turns = False

        async for event in self.run_stream(brief, max_turns=max_turns, history=history):
            kind = event["type"]
            if kind == "tool_result":
                calls.append((event["name"], event["ok"]))
            elif kind == "final":
                final_content = event["content"]
                turns = event["turn"]
            elif kind == "max_turns":
                hit_max_turns = True
                turns = event["turns"]

        return AgentResult(
            final_content=final_content,
            turns=turns,
            tool_calls=calls,
            hit_max_turns=hit_max_turns,
            state=self.state,
        )
