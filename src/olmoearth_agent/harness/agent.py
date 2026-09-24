# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""The lead-agent loop: brief -> LLM -> tool calls -> result.

A single-agent ReAct-style loop (DeerFlow v2's lead-agent shape, minus
subagents for now). Each turn the LLM sees the registry's core tool specs
plus the deferred groups this run has loaded (``ToolRegistry.active_specs``);
each emitted tool call is dispatched and its result fed back, until the model
returns a plain-text answer or the turn budget is exhausted. When the last
turn still asks for tools, one more call, with no tools, asks the model to
answer from what it has, so a run never ends without an answer.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from olmoearth_agent.harness.soul import load_soul
from olmoearth_agent.harness.spill import compact_result_for_llm
from olmoearth_agent.harness.state import ThreadState
from olmoearth_agent.llm.client import OlmoEarthLLM
from olmoearth_agent.llm.types import Message
from olmoearth_agent.security import egress
from olmoearth_agent.studio.client import StudioClient
from olmoearth_agent.tools.registry import ToolContext, ToolRegistry

logger = logging.getLogger(__name__)

#: The agent's soul (persona + guardrails + workflow), loaded from the
#: versioned ``soul.md`` artifact next to this module — or the operator's
#: ``OLMOEARTH_SOUL_PATH`` override — at import time. Editing behavioral
#: boundaries is a markdown change, not a code change (see ``harness/soul.py``).
DEFAULT_SYSTEM_PROMPT = load_soul()

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


#: The harness's message to the model when the turn cap is reached with the
#: model still asking for tools; the answer call that follows offers none.
TURN_CAP_PROMPT = (
    "Harness note: this run has reached its turn cap, and no more tools can be "
    "called. Answer the request now from the tool results above. Say plainly "
    "what you could not do or find out, and why (for example a tool's limit or "
    "an error that repeated). State only what the tools returned."
)


def turn_cap_fallback(max_turns: int) -> str:
    """The answer when the model writes no text even when offered no tools."""
    return (
        f"No answer was written: the run reached its turn cap of {max_turns} "
        "turns while still calling tools, and the model returned no text when "
        "asked to answer without them. The tool calls and their results are in "
        "the run's trace."
    )


def _forced_skill_clause(skill: str) -> str:
    """A system-prompt directive pinning the run to one user-chosen skill.

    Server-side skill routing: the webui's "/" menu sends the picked skill as a
    structured request field, and the agent turns it into a strong steer here
    (instead of the client rewriting the user's brief). Works for both the
    vendored instruction skills - loaded via ``olmoearth_load_skill`` - and the
    Python tool-bundle skills, which are invoked directly. ``skill`` is the webui
    slug (e.g. ``change-detection``); the model resolves it against the skill
    index and tool names.
    """
    return (
        f"\n\nFORCED SKILL: The user explicitly selected the '{skill}' skill for "
        "this request. Use it: if it has a loadable instruction package, call "
        "olmoearth_load_skill for it before other tools; otherwise use its "
        "tool(s) directly. Do not switch to a different skill unless this one is "
        "clearly inapplicable to the brief - and if so, say which you used and why."
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
        forced_skill: str = "",
        memory_block: str = "",
        local: bool = False,
    ) -> None:
        self.llm = llm
        self.registry = registry
        self.studio = studio
        self.state = state or ThreadState()
        self.forced_skill = forced_skill
        self.system_prompt = system_prompt
        if skill_index:
            # Progressive disclosure: list the vendored SKILL.md skills so the
            # model knows to call olmoearth_load_skill when a task matches.
            self.system_prompt += (
                "\n\nAvailable instruction skills (call olmoearth_load_skill "
                "with the name to get full steps):\n" + skill_index
            )
        if memory_block:
            # Cross-thread memory: durable user preferences (rendered by
            # harness/memory.py, already framed as data-not-instructions).
            self.system_prompt += "\n\n" + memory_block
        if forced_skill:
            # Server-side skill routing: pin this run to the user-chosen skill.
            self.system_prompt += _forced_skill_clause(forced_skill)
            # A forced skill whose tools are deferred has them from turn one.
            group = f"olmoearth-{forced_skill}"
            if group in registry.groups():
                self.state.loaded_groups.add(group)
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
        - ``final``       : the plain-text answer (``content``), with
          ``forced_by_turn_cap`` true when the turn cap forced it.
        - ``max_turns``   : the cap was hit with the model still asking for
          tools (``turns``); a ``final`` follows, from one more call that
          offers no tools (``final_answer_forced``).

        Parameters
        ----------
        brief
            The user's natural-language request.
        max_turns
            Hard cap on tool-calling LLM round-trips, to bound cost and stop
            loops. Reaching it costs one more round-trip, without tools, for
            the answer.
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
            # Core specs plus the deferred groups loaded so far; a load_skill
            # call on this turn adds its group from the next turn on.
            tools = self.registry.active_specs(self.state.loaded_groups)
            response = await self.llm.chat(messages, tools=tools)

            if response.thinking:
                yield {"type": "thinking", "turn": turn, "text": response.thinking}

            if not response.tool_calls:
                yield {
                    "type": "final",
                    "turn": turn,
                    "content": response.content,
                    "forced_by_turn_cap": False,
                }
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
                        # Oversized results are spilled to a workspace file and
                        # replaced by a compact envelope so one big payload
                        # can't eat the context window. The UI event above and
                        # the provenance record keep the full result.
                        content=compact_result_for_llm(call.name, result),
                    )
                )

        # The last turn still asked for tools. One more call, with no tools,
        # asks for the answer, so the run never ends without one (exp86 round
        # 1: three runs ended at the cap with nothing to show). Any tool call
        # the model still emits is not run.
        yield {"type": "max_turns", "turns": max_turns, "final_answer_forced": True}
        turn = max_turns + 1
        self.state.turn_count = turn
        messages.append(Message(role="user", content=TURN_CAP_PROMPT))
        response = await self.llm.chat(messages, tools=None)
        if response.thinking:
            yield {"type": "thinking", "turn": turn, "text": response.thinking}
        content = response.content if (response.content or "").strip() else None
        yield {
            "type": "final",
            "turn": turn,
            "content": content or turn_cap_fallback(max_turns),
            "forced_by_turn_cap": True,
        }

    async def run(
        self,
        brief: str,
        *,
        max_turns: int = 8,
        history: list[Message] | None = None,
    ) -> AgentResult:
        """Run the agent loop until it answers (at ``max_turns``, without tools).

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
            Final text, turn count (``max_turns + 1`` when the cap forced the
            answer; ``hit_max_turns`` then says so), and the
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
