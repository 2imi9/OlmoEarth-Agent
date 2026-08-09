# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""The lead-agent loop: brief -> LLM -> tool calls -> result.

A single-agent ReAct-style loop (DeerFlow v2's lead-agent shape, minus
subagents for now). The LLM sees the tool registry's specs; each emitted
tool call is dispatched and its result fed back, until the model returns
a plain-text answer or the turn budget is exhausted.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from olmoearth_agent.harness.response_policy import ResponsePolicy
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

# Appended only for the local model. Final-answer length is controlled for every
# backend by :class:`ResponsePolicy`; this clause prevents the smaller model
# from spending its bounded completion budget narrating a plan before it acts.
LOCAL_BUDGET_CLAUSE = (
    "\n\nLOCAL MODEL: you have a limited output budget. Spend it selecting and "
    "calling tools, not narrating a plan or repeating tool results. Follow the "
    "run-specific RESPONSE CONTRACT appended to this prompt."
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
        response_policy: ResponsePolicy | None = None,
        local: bool = False,
    ) -> None:
        self.llm = llm
        self.registry = registry
        self.studio = studio
        self.state = state or ThreadState()
        self.forced_skill = forced_skill
        self.response_policy = response_policy or ResponsePolicy.from_env()
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
        if local:
            # The local model needs an explicit output-budget + brevity reminder
            # (a hosted model does not), or long answers truncate mid-sentence.
            self.system_prompt += LOCAL_BUDGET_CLAUSE

    async def _compact_final(self, brief: str, draft: str) -> tuple[str, bool]:
        """Losslessly shorten an over-budget final answer with tools disabled.

        This is an editorial pass, not another agent turn. If the provider
        fails, returns a tool call, or makes the answer longer, the original
        answer wins so concision enforcement can never turn a successful run
        into a failed one.
        """
        if not self.response_policy.needs_compaction(brief, draft):
            return draft, False
        editor_messages = [
            Message(
                role="system",
                content=(
                    "You are a lossless response editor. Shorten supplied text "
                    "without changing facts or performing actions."
                ),
            ),
            Message(
                role="user",
                content=self.response_policy.repair_prompt(draft, brief),
            ),
        ]
        try:
            edited = await self.llm.chat(
                editor_messages,
                tools=None,
                mode="instruct_general",
                max_tokens=self.response_policy.repair_max_tokens,
                preserve_thinking=False,
            )
        except Exception:  # an editorial failure must not fail a completed task
            logger.debug("final-answer compaction failed", exc_info=True)
            return draft, False
        candidate = (edited.content or "").strip()
        if (
            edited.tool_calls
            or not candidate
            or self.response_policy.word_count(candidate)
            >= self.response_policy.word_count(draft)
            or not self.response_policy.preserves_required_markers(draft, candidate)
        ):
            return draft, False
        return candidate, True

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
        - ``final``       : the plain-text answer (``content``); optional
          ``compacted=true`` records an over-budget editorial rewrite.
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
        run_prompt = self.system_prompt + "\n\n" + self.response_policy.contract(brief)
        messages: list[Message] = [Message(role="system", content=run_prompt)]
        if history:
            messages.extend(history)
        messages.append(Message(role="user", content=brief))
        ctx = ToolContext(studio=self.studio, state=self.state)
        self._record_external_endpoints()

        for turn in range(1, max_turns + 1):
            self.state.turn_count = turn
            response = await self.llm.chat(
                messages,
                tools=self.registry.specs(),
                max_tokens=self.response_policy.max_tokens(brief),
            )

            if response.thinking:
                yield {"type": "thinking", "turn": turn, "text": response.thinking}

            if not response.tool_calls:
                content = response.content
                compacted = False
                if content:
                    content, compacted = await self._compact_final(brief, content)
                event: dict[str, Any] = {
                    "type": "final",
                    "turn": turn,
                    "content": content,
                }
                if compacted:
                    event["compacted"] = True
                yield event
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
