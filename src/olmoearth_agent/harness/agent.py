# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""The lead-agent loop: brief -> LLM -> tool calls -> result.

A single-agent ReAct-style loop (DeerFlow v2's lead-agent shape, minus
subagents for now). Each turn the LLM sees the registry's core tool specs
plus the deferred groups this run has loaded (``ToolRegistry.active_specs``);
each emitted tool call is dispatched and its result fed back, until the model
returns a plain-text answer.

The loop's behaviours are middleware with LangChain 1.x's hook interface
(:mod:`olmoearth_agent.harness.middleware`); :meth:`LeadAgent.run_stream`
runs the chain. By default, in this order:

- :class:`~olmoearth_agent.harness.turn_cap.TurnCapMiddleware`: when the last
  turn still asks for tools, one more call, with no tools, asks the model to
  answer from what it has, so a run never ends without an answer.
- :class:`~olmoearth_agent.harness.retry_hint.RetryHintMiddleware`: from the
  second time a tool fails alike, its result tells the model to stop retrying.
- :class:`~olmoearth_agent.harness.answer_checks.AnswerChecksMiddleware`:
  before the answer is shown, it is checked against the run
  (:mod:`olmoearth_agent.harness.checks`): its numbers against the tool
  results and the user's messages (:mod:`olmoearth_agent.harness.grounding`),
  its directions, places and magnitudes against the facts the tools computed,
  its claims of files saved or items listed against what the run did, and the
  claims and statements the tools forbid or require. When any check finds a
  violation, one more call, with no tools, asks the model to rewrite the
  answer with every violation listed. The rewrite is checked again but never
  sent back: each sentence a check still flags is shown with ``[unverified:
  <check>]`` after it, and nothing is deleted. Beside the rules, the claim
  check (:mod:`~olmoearth_agent.harness.claim_check`) has the agent's own
  model read the answer against the run's record and the tools' capability
  card, for the unsupported claims and impossible offers no rule enumerates
  (exp86 round 9); it costs at most two model calls per run.
"""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator, Sequence
from contextlib import aclosing
from dataclasses import dataclass, field
from typing import Any

from olmoearth_agent.harness.answer_checks import AnswerChecksMiddleware
from olmoearth_agent.harness.checks import (
    CHECKS,
    CLAIMS,
    NUMBERS,
    SURFACES,
    ToolRecord,
    grounding_prompt,
    revision_prompt,
)
from olmoearth_agent.harness.middleware import (
    AgentMiddleware,
    AgentState,
    EventRelay,
    MiddlewareChain,
    MiddlewareError,
    ModelRequest,
    ModelResponse,
    RunContext,
    Runtime,
    ToolCallRequest,
)
from olmoearth_agent.harness.retry_hint import RetryHintMiddleware
from olmoearth_agent.harness.soul import load_soul, with_capability_card
from olmoearth_agent.harness.spill import spill_result_for_llm
from olmoearth_agent.harness.state import ThreadState
from olmoearth_agent.harness.turn_cap import (
    TURN_CAP_PROMPT,
    TurnCapMiddleware,
    turn_cap_fallback,
)
from olmoearth_agent.llm.client import OlmoEarthLLM
from olmoearth_agent.llm.types import Message, ToolCall
from olmoearth_agent.security import egress
from olmoearth_agent.studio.client import StudioClient
from olmoearth_agent.tools.registry import ToolContext, ToolRegistry

logger = logging.getLogger(__name__)

__all__ = [
    "CHECK_ANSWER_ENV",
    "CHECK_CLAIMS_ENV",
    "CHECK_NUMBERS_ENV",
    "DEFAULT_SYSTEM_PROMPT",
    "TURN_CAP_PROMPT",
    "AgentResult",
    "LeadAgent",
    "grounding_prompt",
    "revision_prompt",
    "turn_cap_fallback",
]

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


#: Set to ``0`` (or ``false``, ``no``, ``off``) to switch the answer's number
#: check off; ``LeadAgent(check_numbers=False)`` does the same for one agent.
CHECK_NUMBERS_ENV = "OLMOEARTH_CHECK_NUMBERS"
#: The same for the answer's other checks (direction, actions, forbidden
#: claims, required statements); ``LeadAgent(check_answer=False)``.
CHECK_ANSWER_ENV = "OLMOEARTH_CHECK_ANSWER"
#: The same for the claim check, the model's own reading of the answer
#: against the run (``harness/claim_check.py``); ``LeadAgent(check_claims=
#: False)``. Independent of the two above: it is the one check that costs
#: model calls (at most two per run).
CHECK_CLAIMS_ENV = "OLMOEARTH_CHECK_CLAIMS"
_FALSY = {"0", "false", "no", "off"}


#: How many times one turn may enter the model step, jumps back included: a
#: guard like LangGraph's recursion limit against a hook that always jumps.
#: The default chain enters it at most twice (the answer and its rewrite).
MODEL_STEPS_PER_TURN = 10


def _env_on(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() not in _FALSY


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
    #: The answer is the model's rewrite after the number check.
    grounding_revised: bool = False
    #: The run's ``grounding_check`` events (see :meth:`LeadAgent.run_stream`).
    grounding_checks: list[dict[str, Any]] = field(default_factory=list)
    #: The answer is the model's rewrite after any check.
    revised: bool = False
    #: The run's ``check`` events: one per check that fired and pass, and one
    #: per call of the claim check.
    checks: list[dict[str, Any]] = field(default_factory=list)
    #: The checks whose sentences the answer shows marked ``[unverified: ...]``.
    marked: list[str] = field(default_factory=list)


@dataclass
class _Run:
    """One run's chain, state and runtime, and the tool calls to run next."""

    chain: MiddlewareChain
    state: AgentState
    runtime: Runtime
    system: Message
    pending: list[ToolCall] = field(default_factory=list)
    #: Model steps entered in the current turn (jumps back included).
    steps: int = 0
    #: The call the registry last ran: a wrap hook may have replaced the
    #: model's call, and the records must hold the one that ran.
    executed: ToolCall | None = None


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
        check_numbers: bool = True,
        check_answer: bool = True,
        check_claims: bool = True,
        surface: str = "cli",
        middleware: Sequence[AgentMiddleware] | None = None,
    ) -> None:
        if surface not in SURFACES:
            raise ValueError(f"surface must be one of {SURFACES}, not {surface!r}")
        #: The run's middleware, in order; ``None`` runs
        #: :meth:`default_middleware`, built afresh for each run.
        self.middleware = None if middleware is None else list(middleware)
        if self.middleware is not None:
            MiddlewareChain(self.middleware)  # refuse a bad list now, not mid-run
        self.llm = llm
        self.registry = registry
        self.studio = studio
        self.state = state or ThreadState()
        self.forced_skill = forced_skill
        #: The prompt the card follows (the soul, unless one was passed).
        self._soul = system_prompt
        #: What follows the card: skill index, memory, forced skill, budget.
        tail = ""
        # Check the answer against the run before it is shown, unless
        # switched off here or by OLMOEARTH_CHECK_NUMBERS=0 (the numbers),
        # OLMOEARTH_CHECK_ANSWER=0 (the other rules) and
        # OLMOEARTH_CHECK_CLAIMS=0 (the claim check).
        self.check_numbers = check_numbers and _env_on(CHECK_NUMBERS_ENV)
        self.check_answer = check_answer and _env_on(CHECK_ANSWER_ENV)
        self.check_claims = check_claims and _env_on(CHECK_CLAIMS_ENV)
        # Where the answer is shown: on the web the tool results are shown
        # beside it, so "listed above" may point at them (harness/checks.py).
        self.surface = surface
        if skill_index:
            # Progressive disclosure: list the vendored SKILL.md skills so the
            # model knows to call olmoearth_load_skill when a task matches.
            tail += (
                "\n\nAvailable instruction skills (call olmoearth_load_skill "
                "with the name to get full steps):\n" + skill_index
            )
        if memory_block:
            # Cross-thread memory: durable user preferences (rendered by
            # harness/memory.py, already framed as data-not-instructions).
            tail += "\n\n" + memory_block
        if forced_skill:
            # Server-side skill routing: pin this run to the user-chosen skill.
            tail += _forced_skill_clause(forced_skill)
            # A forced skill whose tools are deferred has them from turn one.
            group = f"olmoearth-{forced_skill}"
            if group in registry.groups():
                self.state.loaded_groups.add(group)
        if local:
            # The local model needs an explicit output-budget + brevity reminder
            # (a hosted model does not), or long answers truncate mid-sentence.
            tail += LOCAL_BUDGET_CLAUSE
        self._prompt_tail = tail
        #: The loaded groups the prompt's capability card lists in full.
        self._card_groups = frozenset(self.state.loaded_groups)
        self.system_prompt = self._compose_system_prompt()

    def _compose_system_prompt(self) -> str:
        """The soul, the capability card of the loaded groups, then the clauses.

        The card follows the soul, whose rules point at it (exp86 round 9:
        offers of actions no tool can do); the run's clauses come after, the
        local budget clause last, where recency helps a weak model.
        """
        head = with_capability_card(
            self._soul, self.registry, sorted(self._card_groups)
        )
        return head + self._prompt_tail

    def _refresh_card(self, run: _Run) -> None:
        """Rebuild the system message when the run has loaded another group.

        ``olmoearth_load_skill`` (or dispatching a deferred tool) sends that
        group's specs from the next turn on; its tools join the card at the
        same time, so the card lists every tool the model is offered.
        """
        groups = frozenset(self.state.loaded_groups)
        if groups == self._card_groups:
            return
        self._card_groups = groups
        self.system_prompt = self._compose_system_prompt()
        run.system = Message(role="system", content=self.system_prompt)

    def _checks(self) -> list[str]:
        """The answer checks this agent runs, in order: the rules, then the claims."""
        rules = [
            name
            for name in CHECKS
            if (self.check_numbers if name == NUMBERS else self.check_answer)
        ]
        return rules + ([CLAIMS] if self.check_claims else [])

    def default_middleware(self) -> list[AgentMiddleware]:
        """The chain a run uses when ``middleware`` was not given.

        The turn cap outermost, then the retry hint, then the answer checks
        (the ones ``check_numbers``, ``check_answer`` and ``check_claims``
        leave on, on this agent's ``surface``; the claim check calls this
        agent's ``llm`` and reads its ``registry``'s card). Their hooks do
        not depend on each other's order; this order is the loop's before
        middleware.
        """
        return [
            TurnCapMiddleware(),
            RetryHintMiddleware(),
            AnswerChecksMiddleware(
                self._checks(),
                surface=self.surface,
                llm=self.llm,
                registry=self.registry,
            ),
        ]

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
          ``forced_by_turn_cap`` true when the turn cap forced it,
          ``revised`` true when it is the model's rewrite after the answer
          checks (``grounding_revised`` when the number check was among
          them), and ``marked`` naming the checks whose sentences it shows
          marked. Exactly one per run, always last.
        - ``max_turns``   : the cap was hit with the model still asking for
          tools (``turns``); a ``final`` follows, from one more call that
          offers no tools (``final_answer_forced``).
        - ``grounding_check`` : the answer states numbers (``unsupported``, as
          written) that no tool result of this run and no user message
          supports. ``action`` is ``"revise"`` when one more call, with no
          tools, asks the model to rewrite the answer without them, and
          ``"shown"`` when the answer about to be shown (the rewrite, or the
          draft if the rewrite was empty) still states some. A ``"revise"``
          event carries the answer it replaces (``draft``). At most one of
          each per run, before the ``final``; ``turn`` is the answer's turn.
          Off with ``check_numbers=False`` or ``OLMOEARTH_CHECK_NUMBERS=0``.
        - ``check``       : one per check that found violations in the
          answer (``check``: ``numbers``, ``direction``, ``actions``,
          ``forbidden_claims``, ``must_state`` or ``claims``;
          ``violations``: each ``{check, text, detail}``, ``text`` the
          sentence). ``action`` is
          ``"revise"`` on the draft (which the event carries as ``draft``):
          one call, with no tools and the rewrite sampling
          (``REVISION_MODE``), asks for a rewrite that fixes every
          violation of every check. It is ``"marked"`` on the answer about to
          be shown (the rewrite, or the draft if the rewrite was empty),
          where each sentence still flagged is followed by
          ``[unverified: <check>]``, and a required statement the answer
          still lacks is added at its end after its marker. It is
          ``"appended"`` when a required statement is all the draft lacks:
          it is added the same way, with no rewrite. The number check
          yields its ``grounding_check`` beside its ``check`` event. Off
          with ``check_answer=False`` or ``OLMOEARTH_CHECK_ANSWER=0``
          (every rule but the numbers). The claim check (``claims``: the
          agent's own model reads the answer against the run, one call on
          the draft and one on the rewrite, no tools) yields one ``check``
          event per call, flags or not: ``"revise"`` or ``"marked"`` with
          its flags (``violations`` also carry ``kind``: ``unsupported``,
          ``contradicts`` or ``offer``), else ``"passed"``, or
          ``"failed_open"`` when its call failed or its reply was no list
          (``error`` says which; nothing is flagged). Each carries the
          verifier's raw ``reply``, its ``version``, ``unmatched`` (quotes
          found in no sentence of the answer), ``reply_complete`` and
          ``finish_reason``. Off with ``check_claims=False`` or
          ``OLMOEARTH_CHECK_CLAIMS=0``.

        The loop runs this agent's middleware (``middleware``, else
        :meth:`default_middleware`) in LangChain's order: ``abefore_agent``
        once, then per model call ``abefore_model``, the call wrapped by
        every ``awrap_model_call`` and ``aafter_model``, then per tool call
        ``awrap_tool_call``, and ``aafter_agent`` before the ``final``. The
        ``tool_call``, ``tool_result`` and ``final`` events are yielded here
        and ``thinking`` by the model step; the middleware emit the others
        (``max_turns``: the turn cap; ``grounding_check`` and ``check``: the
        answer checks), and each is yielded in order as it is emitted.

        Parameters
        ----------
        brief
            The user's natural-language request.
        max_turns
            Hard cap on tool-calling LLM round-trips, to bound cost and stop
            loops. Reaching it costs one more round-trip, without tools, for
            the answer (``TurnCapMiddleware``). A chain that would start a
            turn after that one raises :class:`MiddlewareError`, and so does
            a turn that enters the model step more than
            :data:`MODEL_STEPS_PER_TURN` times.
        history
            Prior conversation turns (user/assistant messages) to seed before
            the new ``brief``, so multi-turn follow-ups have context. Inserted
            between the system prompt and the new user message.
        """
        chain = MiddlewareChain(
            self.default_middleware() if self.middleware is None else self.middleware
        )
        ctx = ToolContext(studio=self.studio, state=self.state)
        self._record_external_endpoints()
        relay = EventRelay()
        runtime = Runtime(
            context=RunContext(
                brief=brief, history=list(history or []), max_turns=max_turns
            ),
            emit=relay.emit,
        )
        state: AgentState = {
            "messages": [*(history or []), Message(role="user", content=brief)],
            "thread_state": self.state,
            "turn": 0,
            "answer": None,
            # The full result of every call, for the answer checks (the model
            # saw compacted ones; a spilled result's numbers are still the
            # tool's).
            "tool_records": [],
        }
        run = _Run(
            chain=chain,
            state=state,
            runtime=runtime,
            system=Message(role="system", content=self.system_prompt),
        )

        async def dispatch(request: ToolCallRequest) -> dict[str, Any]:
            # The chain's RetryHintMiddleware adds the stop-retrying hint.
            run.executed = request.tool_call
            return await self.registry.dispatch(
                request.tool_call, ctx, retry_hint=False
            )

        # Each step runs through the relay, which yields the events the
        # middleware emit during it, as they are emitted.
        async with aclosing(relay.run(chain.before_agent(state, runtime))) as events:
            async for event in events:
                yield event
        node, advance = relay.value or "model", True
        while True:
            if node == "model":
                async with aclosing(
                    relay.run(self._model_node(run, advance))
                ) as events:
                    async for event in events:
                        yield event
                node, advance = relay.value, False
            elif node == "tools":
                turn = state["turn"]
                for call in run.pending:
                    yield {
                        "type": "tool_call",
                        "turn": turn,
                        "id": call.id,
                        "name": call.name,
                        "arguments": call.arguments,
                    }
                    request = ToolCallRequest(
                        tool_call=call,
                        tool=self.registry.spec_of(call.name),
                        state=state,
                        runtime=runtime,
                    )
                    run.executed = call
                    step = chain.call_tool(request, dispatch)
                    async with aclosing(relay.run(step)) as events:
                        async for event in events:
                            yield event
                    result: dict[str, Any] = relay.value
                    # The records hold the call that ran, which a wrap hook
                    # may have changed; the events and the tool message keep
                    # the model's id and name, which pair them with its call.
                    ran = run.executed or call
                    # Oversized results are spilled to a workspace file and
                    # replaced by a compact envelope so one big payload can't
                    # eat the context window. The UI event below and the
                    # provenance record keep the full result; the spill file
                    # is one this run wrote, which the answer may name.
                    content, spilled_to = spill_result_for_llm(ran.name, result)
                    state["tool_records"].append(
                        ToolRecord(ran.name, ran.arguments, result, spilled_to)
                    )
                    self.state.provenance.record_tool_call(
                        ran.name, ran.arguments, result
                    )
                    yield {
                        "type": "tool_result",
                        "turn": turn,
                        "id": call.id,
                        "name": call.name,
                        "ok": bool(result.get("ok")),
                        "result": result,
                    }
                    state["messages"].append(
                        Message(
                            role="tool",
                            tool_call_id=call.id,
                            name=call.name,
                            content=content,
                        )
                    )
                run.pending = []
                node, advance = "model", True
            else:
                ending = chain.after_agent(state, runtime)
                async with aclosing(relay.run(ending)) as events:
                    async for event in events:
                        yield event
                if relay.value in (None, "end"):
                    break
                node, advance = relay.value, False

        yield {
            "type": "final",
            "turn": state["turn"],
            "content": state.get("answer"),
            "forced_by_turn_cap": bool(state.get("forced_by_turn_cap", False)),
            "grounding_revised": bool(state.get("grounding_revised", False)),
            "revised": bool(state.get("revised", False)),
            "marked": list(state.get("marked", [])),
        }

    async def _model_node(self, run: _Run, advance: bool) -> str:
        """One model call with its ``abefore_model`` and ``aafter_model`` hooks.

        Returns where the run goes next: a hook's jump, else the tools when
        the response calls any, else the end.
        """
        state, runtime = run.state, run.runtime
        if advance:
            state["turn"] += 1
            run.steps = 0
        run.steps += 1
        if run.steps > MODEL_STEPS_PER_TURN:
            raise MiddlewareError(
                f"turn {state['turn']} went back to the model "
                f"{MODEL_STEPS_PER_TURN} times: a hook that jumps to the model "
                "must stop"
            )
        jump = await run.chain.before_model(state, runtime)
        if jump is not None:
            return jump
        max_turns = runtime.context.max_turns
        if state["turn"] > max_turns + 1:
            raise MiddlewareError(
                f"turn {state['turn']} is past the turn cap of {max_turns}: no "
                "middleware ended the run (TurnCapMiddleware does)"
            )
        self.state.turn_count = state["turn"]
        self._refresh_card(run)
        request = ModelRequest(
            messages=list(state["messages"]),
            system_message=run.system,
            # Core specs plus the deferred groups loaded so far; a load_skill
            # call on this turn adds its group from the next turn on.
            tools=self.registry.active_specs(self.state.loaded_groups),
            state=state,
            runtime=runtime,
        )
        response = (await run.chain.call_model(request, self._call_model)).result
        if response.thinking:
            runtime.emit(
                {"type": "thinking", "turn": state["turn"], "text": response.thinking}
            )
        state["messages"].append(
            Message(
                role="assistant",
                content=response.content,
                tool_calls=response.tool_calls or None,
            )
        )
        state["answer"] = response.content
        run.pending = list(response.tool_calls)
        jump = await run.chain.after_model(state, runtime)
        return jump or ("tools" if run.pending else "end")

    async def _call_model(self, request: ModelRequest) -> ModelResponse:
        """The call every ``awrap_model_call`` wraps: ``OlmoEarthLLM.chat``."""
        if request.tool_choice is not None:
            raise MiddlewareError(
                "OlmoEarthLLM.chat takes no tool_choice; offer no tools with "
                "request.override(tools=None)"
            )
        messages = list(request.messages)
        if request.system_message is not None:
            messages.insert(0, request.system_message)
        response = await self.llm.chat(
            messages, tools=request.tools, **request.model_settings
        )
        return ModelResponse(result=response)

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
        grounding_revised = revised = False
        grounding_checks: list[dict[str, Any]] = []
        checks: list[dict[str, Any]] = []
        marked: list[str] = []

        async for event in self.run_stream(brief, max_turns=max_turns, history=history):
            kind = event["type"]
            if kind == "tool_result":
                calls.append((event["name"], event["ok"]))
            elif kind == "final":
                final_content = event["content"]
                turns = event["turn"]
                grounding_revised = bool(event.get("grounding_revised"))
                revised = bool(event.get("revised"))
                marked = list(event.get("marked") or [])
            elif kind == "max_turns":
                hit_max_turns = True
                turns = event["turns"]
            elif kind == "grounding_check":
                grounding_checks.append(event)
            elif kind == "check":
                checks.append(event)

        return AgentResult(
            final_content=final_content,
            turns=turns,
            tool_calls=calls,
            hit_max_turns=hit_max_turns,
            state=self.state,
            grounding_revised=grounding_revised,
            grounding_checks=grounding_checks,
            revised=revised,
            checks=checks,
            marked=marked,
        )
