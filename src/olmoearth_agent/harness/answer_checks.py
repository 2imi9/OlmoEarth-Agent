# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""The answer checks, as middleware: one rewrite, then marking.

exp86: in each of three rounds the only genuine fault was a number no tool
had returned; the round 6 and 7 audits found directions, places and actions
no tool supported. The answer is checked against the run
(:mod:`olmoearth_agent.harness.checks`) before it is shown. When any check
finds a violation, one more call, with no tools and the rewrite's sampling
(``REVISION_MODE``), asks the model to rewrite the answer with every
violation listed. The rewrite is checked again but never sent back: each
sentence a check still flags is shown with ``[unverified: <check>]`` after
it, and nothing is deleted. A required statement the answer only
paraphrased or left out costs no rewrite: it is appended as the tool states
it. The harness's turn-cap fallback is not checked.

Beside the rules, the claim check (``claims``,
:mod:`olmoearth_agent.harness.claim_check`): the agent's own model reads
the answer against the run's record and the capability card, for what the
rules cannot enumerate (exp86 round 9: the rules caught about half of new
wordings). Its flags join the rules' in the one rewrite, and it reads the
rewrite again. It costs at most two model calls per run, one on the draft
and one on the rewrite, and none when the answer is not checked.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from olmoearth_agent.harness.capabilities import capability_card
from olmoearth_agent.harness.checks import (
    CHECKS,
    CLAIMS,
    MUST_STATE,
    NUMBERS,
    SURFACES,
    RunEvidence,
    Violation,
    mark_answer,
    revision_prompt,
    run_checks,
    unsupported_of,
)
from olmoearth_agent.harness.claim_check import ChatClient, ClaimCheck, check_claims
from olmoearth_agent.harness.middleware import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
    Runtime,
    hook_config,
    last_ai_message,
    without_tool_calls,
)
from olmoearth_agent.llm.presets import REVISION_MODE
from olmoearth_agent.llm.types import ChatResponse, Message
from olmoearth_agent.tools.registry import ToolRegistry

__all__ = ["CLAIM_ACTIONS", "AnswerChecksMiddleware"]

#: A ``claims`` check event's ``action`` when the claim check flagged nothing:
#: ``"passed"`` (its reply was read) or ``"failed_open"`` (its call failed or
#: its reply was no list; ``error`` says which). With flags it is
#: ``"revise"`` or ``"marked"``, as for the rules.
PASSED = "passed"
FAILED_OPEN = "failed_open"
CLAIM_ACTIONS = ("revise", "marked", PASSED, FAILED_OPEN)

#: ``state["answer_checks"]``: the rewrite has been asked for.
REVISING = "revising"
#: ``state["answer_checks"]``: the rewrite was checked and marked; no more.
DONE = "done"


class AnswerChecksMiddleware(AgentMiddleware):
    """Checks each answer (a response without tool calls) before it is shown.

    ``checks`` are the names of :data:`~olmoearth_agent.harness.checks.CHECKS`
    to run, in that order, and :data:`~olmoearth_agent.harness.checks.CLAIMS`
    for the claim check, which runs after them; none runs nothing. The claim
    check calls ``llm`` (the agent's own client, as the loop calls it) and
    reads offers against the capability card of ``registry``. ``surface``
    is where the answer is shown (``"cli"`` or ``"web"``; see
    ``RunEvidence``).

    ``aafter_model`` runs the checks. Only a required statement missing: it
    emits ``check`` events with ``action`` ``"appended"`` and appends the
    statements. Any other violation: it emits ``check`` events (``"revise"``,
    with the ``draft``) and, when the numbers fired, a ``grounding_check``,
    adds the rewrite note and jumps to the model once. ``awrap_model_call``
    offers that call no tools and ``REVISION_MODE``. On the rewrite (or the
    draft, if the rewrite is empty) ``aafter_model`` emits ``"marked"`` and
    ``"shown"`` events for what still fails and marks it.

    The claim check emits one ``check`` event per call, flags or not, since
    its reply cannot be replayed as a rule can: ``"revise"`` or ``"marked"``
    with its flags, else ``"passed"`` or ``"failed_open"``; each carries the
    verifier's raw ``reply``, its ``version``, the quotes found in no
    sentence (``unmatched``), ``reply_complete`` and ``finish_reason``, and
    ``error`` when it failed open.
    """

    def __init__(
        self,
        checks: Sequence[str] = tuple(CHECKS),
        *,
        surface: str = "cli",
        llm: ChatClient | None = None,
        registry: ToolRegistry | None = None,
    ) -> None:
        unknown = [c for c in checks if c not in CHECKS and c != CLAIMS]
        if unknown:
            raise ValueError(
                f"unknown checks {unknown}; the checks are {[*CHECKS, CLAIMS]}"
            )
        if surface not in SURFACES:
            raise ValueError(f"surface must be one of {SURFACES}, not {surface!r}")
        if CLAIMS in checks and llm is None:
            raise ValueError("the claims check needs the agent's LLM client (llm=)")
        self.checks = list(checks)
        self.surface = surface
        self.llm = llm
        self.registry = registry

    @property
    def _rules(self) -> list[str]:
        """The rule checks to run, in order (every check but the claims)."""
        return [c for c in self.checks if c != CLAIMS]

    async def _claims(
        self, state: AgentState, answer: str, evidence: RunEvidence
    ) -> ClaimCheck | None:
        """The claim check's reading of ``answer``, or ``None`` when it is off."""
        if CLAIMS not in self.checks or self.llm is None:
            return None
        loaded = getattr(state.get("thread_state"), "loaded_groups", ())
        card = "" if self.registry is None else capability_card(self.registry, loaded)
        return await check_claims(self.llm, answer, evidence, card)

    async def _found(
        self, state: AgentState, answer: str, evidence: RunEvidence
    ) -> tuple[dict[str, list[Violation]], ClaimCheck | None]:
        """Every check's violations (the rules', then the claims'), and the reading."""
        found = run_checks(answer, evidence, self._rules)
        claim = await self._claims(state, answer, evidence)
        if claim is not None and claim.violations:
            found[CLAIMS] = claim.violations
        return found, claim

    @staticmethod
    def _emit(
        runtime: Runtime,
        turn: int,
        found: dict[str, list[Violation]],
        claim: ClaimCheck | None,
        action: str,
        **extra: Any,
    ) -> None:
        """One ``check`` event per check in ``found``, with ``action``.

        The claim check, when it ran and flagged nothing, has its own event
        (``"passed"`` or ``"failed_open"``) after the others.
        """
        for name, violations in found.items():
            event = {
                "type": "check",
                "turn": turn,
                "check": name,
                "violations": violations,
                "action": action,
                **extra,
            }
            if name == CLAIMS and claim is not None:
                event.update(claim.audit())
            runtime.emit(event)
        if claim is not None and CLAIMS not in found:
            runtime.emit(
                {
                    "type": "check",
                    "turn": turn,
                    "check": CLAIMS,
                    "violations": [],
                    "action": FAILED_OPEN if claim.failed else PASSED,
                    **claim.audit(),
                }
            )

    def _evidence(self, state: AgentState, runtime: Runtime) -> RunEvidence:
        history = runtime.context.history
        return RunEvidence(
            tools=state.get("tool_records", []),
            # The user's own words; never the saved preferences. An earlier
            # assistant message is a source on the web only, where it was
            # checked when it was shown (a derived number is not a source on
            # the command line).
            user_messages=[runtime.context.brief]
            + [m.content for m in history if m.role == "user" and m.content],
            surface=self.surface,
            assistant_messages=[
                m.content for m in history if m.role == "assistant" and m.content
            ],
        )

    @hook_config(can_jump_to=["model"])
    async def aafter_model(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        """Check the answer: append, ask for one rewrite, or mark the rewrite."""
        phase = state.get("answer_checks")
        if not self.checks or phase == DONE:
            return None
        if phase == REVISING:
            return await self._mark(state, runtime)
        last = last_ai_message(state)
        answer = state.get("answer")
        if last is None or last.tool_calls or not (answer and answer.strip()):
            return None
        found, claim = await self._found(state, answer, self._evidence(state, runtime))
        turn = state.get("turn", 0)
        if not found:
            self._emit(runtime, turn, found, claim, PASSED)  # the claim check's alone
            return None
        if set(found) == {MUST_STATE}:
            # A required statement the answer only paraphrased or left out
            # costs no rewrite: the harness appends it as the tool states it
            # (replayed on exp86 rounds 6-7, the key-term test flagged four
            # correct declines worded differently).
            self._emit(runtime, turn, found, claim, "appended")
            return {"answer": mark_answer(answer, found), "marked": list(found)}
        self._emit(runtime, turn, found, claim, "revise", draft=answer)
        if NUMBERS in found:
            runtime.emit(
                {
                    "type": "grounding_check",
                    "turn": turn,
                    "unsupported": unsupported_of(found[NUMBERS]),
                    "action": "revise",
                    # The answer the rewrite replaces, so a trace shows what
                    # the check removed (exp86 round 4 could not tell).
                    "draft": answer,
                }
            )
        return {
            "answer_checks": REVISING,
            "answer_checks_draft": answer,
            "answer_checks_numbers": NUMBERS in found,
            "messages": [Message(role="user", content=revision_prompt(found))],
            "jump_to": "model",
        }

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse | ChatResponse:
        """The rewrite call: no tools, the rewrite's sampling, no tool call run."""
        if request.state.get("answer_checks") != REVISING:
            return await handler(request)
        settings = {**request.model_settings, "mode": REVISION_MODE}
        return without_tool_calls(
            await handler(request.override(tools=None, model_settings=settings))
        )

    async def _mark(self, state: AgentState, runtime: Runtime) -> dict[str, Any]:
        """Check the rewrite (or the draft, if it is empty) and mark what fails."""
        rewrite = state.get("answer")
        revised = bool((rewrite or "").strip())
        answer = rewrite if revised else state.get("answer_checks_draft")
        found, claim = await self._found(
            state, answer or "", self._evidence(state, runtime)
        )
        turn = state.get("turn", 0)
        self._emit(runtime, turn, found, claim, "marked")
        if NUMBERS in found:
            runtime.emit(
                {
                    "type": "grounding_check",
                    "turn": turn,
                    "unsupported": unsupported_of(found[NUMBERS]),
                    "action": "shown",
                }
            )
        updates: dict[str, Any] = {
            "answer_checks": DONE,
            "answer": answer,
            "revised": revised,
            "grounding_revised": revised and bool(state.get("answer_checks_numbers")),
        }
        if found:
            # Fail closed by marking: nothing is deleted.
            updates["answer"] = mark_answer(answer or "", found)
            updates["marked"] = list(found)
        return updates
