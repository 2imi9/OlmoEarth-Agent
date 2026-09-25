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
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from olmoearth_agent.harness.checks import (
    CHECKS,
    MUST_STATE,
    NUMBERS,
    SURFACES,
    RunEvidence,
    mark_answer,
    revision_prompt,
    run_checks,
    unsupported_of,
)
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

__all__ = ["AnswerChecksMiddleware"]

#: ``state["answer_checks"]``: the rewrite has been asked for.
REVISING = "revising"
#: ``state["answer_checks"]``: the rewrite was checked and marked; no more.
DONE = "done"


class AnswerChecksMiddleware(AgentMiddleware):
    """Checks each answer (a response without tool calls) before it is shown.

    ``checks`` are the names of :data:`~olmoearth_agent.harness.checks.CHECKS`
    to run, in that order; none runs nothing. ``surface`` is where the answer
    is shown (``"cli"`` or ``"web"``; see ``RunEvidence``).

    ``aafter_model`` runs the checks. Only a required statement missing: it
    emits ``check`` events with ``action`` ``"appended"`` and appends the
    statements. Any other violation: it emits ``check`` events (``"revise"``,
    with the ``draft``) and, when the numbers fired, a ``grounding_check``,
    adds the rewrite note and jumps to the model once. ``awrap_model_call``
    offers that call no tools and ``REVISION_MODE``. On the rewrite (or the
    draft, if the rewrite is empty) ``aafter_model`` emits ``"marked"`` and
    ``"shown"`` events for what still fails and marks it.
    """

    def __init__(
        self, checks: Sequence[str] = tuple(CHECKS), *, surface: str = "cli"
    ) -> None:
        unknown = [c for c in checks if c not in CHECKS]
        if unknown:
            raise ValueError(f"unknown checks {unknown}; the checks are {list(CHECKS)}")
        if surface not in SURFACES:
            raise ValueError(f"surface must be one of {SURFACES}, not {surface!r}")
        self.checks = list(checks)
        self.surface = surface

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
            return self._mark(state, runtime)
        last = last_ai_message(state)
        answer = state.get("answer")
        if last is None or last.tool_calls or not (answer and answer.strip()):
            return None
        found = run_checks(answer, self._evidence(state, runtime), self.checks)
        if not found:
            return None
        turn = state.get("turn", 0)
        if set(found) == {MUST_STATE}:
            # A required statement the answer only paraphrased or left out
            # costs no rewrite: the harness appends it as the tool states it
            # (replayed on exp86 rounds 6-7, the key-term test flagged four
            # correct declines worded differently).
            for name, violations in found.items():
                runtime.emit(
                    {
                        "type": "check",
                        "turn": turn,
                        "check": name,
                        "violations": violations,
                        "action": "appended",
                    }
                )
            return {"answer": mark_answer(answer, found), "marked": list(found)}
        for name, violations in found.items():
            runtime.emit(
                {
                    "type": "check",
                    "turn": turn,
                    "check": name,
                    "violations": violations,
                    "action": "revise",
                    "draft": answer,
                }
            )
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

    def _mark(self, state: AgentState, runtime: Runtime) -> dict[str, Any]:
        """Check the rewrite (or the draft, if it is empty) and mark what fails."""
        rewrite = state.get("answer")
        revised = bool((rewrite or "").strip())
        answer = rewrite if revised else state.get("answer_checks_draft")
        found = run_checks(answer or "", self._evidence(state, runtime), self.checks)
        turn = state.get("turn", 0)
        for name, violations in found.items():
            runtime.emit(
                {
                    "type": "check",
                    "turn": turn,
                    "check": name,
                    "violations": violations,
                    "action": "marked",
                }
            )
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
