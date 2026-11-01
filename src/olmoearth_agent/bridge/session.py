# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""One bridge step: a fresh ``LeadAgent`` replays the run, then waits or ends.

:func:`step` builds the agent as ``olmoearth-agent`` does (``cli.py``: the
default registry, ``StudioClient.from_env``, a fresh ``ThreadState`` and the
skill index), with :class:`ScriptedLLM` as its model,
the claim check off, not ``local`` (the budget clause is for the small local
model) and on the command line's surface. It runs ``run_stream`` as a task:
the recorded responses are replayed and the cached tool results served
(:mod:`olmoearth_agent.bridge.replay`), the response just given is played,
and the step stops at the first model call nothing answers yet, or at the
``final`` event.

What a step returns (:func:`pending_view`, :func:`final_view`) is what the
model needs next: for a model call, the messages it has not seen since its
last response (tool results as the loop feeds them, spilled when large; the
harness's notes), the system prompt when it changed, the tools offered, with
their specs on the first call and whenever the tools offered change (an
argument a spec does not declare is passed through and ignored by its
handler, so the model needs the names), and the sampling mode; for the end,
the answer with its flags and the notes ``olmoearth-agent`` prints beside
it.

A step holds the run's lock (:meth:`RunDir.locked`) from its first read to
its last write, so a step sent again after a dropped connection, while the
first still runs on the cluster, is refused, and ``show`` says a step is
running. A response may name the model call it answers (``call``), so one
sent again after that call was answered is refused, never played on the
next. Whether the run's Studio key is declared read-only
(``OLMOEARTH_READ_ONLY``) is recorded at ``begin`` and must stay so.

The one draw of chance in what the loop shows the model, besides what the
tools return (served from the cache), is the name of a spilled result's file
(``harness/spill.py``: ``uuid4``). During a step the n-th spill of a run
takes the same name each time (:func:`stable_spill_names`), so the path the
model read stays the path the run wrote. The provenance log's times are
drawn anew by each replay; the model reads them only through the tools that
report them, whose results are cached.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import uuid
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

from olmoearth_agent.bridge.errors import BridgeError, BridgeHalt
from olmoearth_agent.bridge.replay import ToolCache, replay_registry
from olmoearth_agent.bridge.scripted import RecordedResponse, ScriptedLLM
from olmoearth_agent.bridge.store import RunDir
from olmoearth_agent.harness import spill
from olmoearth_agent.harness.agent import AgentResult, LeadAgent
from olmoearth_agent.harness.state import ThreadState
from olmoearth_agent.llm.client import OlmoEarthLLM
from olmoearth_agent.llm.types import Message, Role, ToolSpec
from olmoearth_agent.provenance.log import ProvenanceLog
from olmoearth_agent.skills import SkillLoader, build_default_registry
from olmoearth_agent.skills.registry import READ_ONLY_ENV
from olmoearth_agent.studio.client import StudioClient
from olmoearth_agent.tools.registry import ToolRegistry, opted_in

logger = logging.getLogger(__name__)

__all__ = [
    "BRIDGE_VERSION",
    "DEFAULT_MAX_TURNS",
    "FINAL",
    "MODEL_CALL",
    "begin",
    "build_agent",
    "cli_notes",
    "cli_trace",
    "final_view",
    "pending_view",
    "render_text",
    "show",
    "stable_spill_names",
    "step",
    "system_prompt",
    "tool_specs",
]

#: The version of the run directory's layout, in ``brief.json``.
BRIDGE_VERSION = 1
#: As ``olmoearth-agent --max-turns``.
DEFAULT_MAX_TURNS = 8
#: The surface the answer is checked for: the command line's (``checks.py``).
SURFACE = "cli"
#: ``pending.json``'s ``status`` while the loop waits for the model.
MODEL_CALL = "model_call"
#: ``pending.json``'s ``status`` once the loop has answered.
FINAL = "final"

RESPOND_WITH_TOOLS = (
    'Reply with {"tool_calls": [{"name": "<tool>", "arguments": {...}}]} '
    '(several calls run in order, as one turn) or with {"content": "<answer>"}.'
)
RESPOND_WITHOUT_TOOLS = (
    'No tools are offered on this call: reply with {"content": "<answer>"} only.'
)

RegistryFactory = Callable[[], ToolRegistry]


# --- the agent -------------------------------------------------------------


def build_agent(
    llm: ScriptedLLM,
    registry: ToolRegistry,
    studio: StudioClient | None,
    brief: dict[str, Any],
) -> LeadAgent:
    """A ``LeadAgent`` as ``olmoearth-agent`` builds one, for the run ``brief``.

    ``cli.py``'s ``run_brief`` with the run's skill index, the claim check
    off, ``local`` off and the command line's surface. The provenance log
    takes the run's id, so each step's log names the run. A run begun before
    the preference memory went (31 October 2026) also recorded a
    ``memory_block``; it is no longer read, so its directory still loads, and
    a missing ``skill_index`` reads as ``''``.
    """
    settings = brief["settings"]
    return LeadAgent(
        cast(OlmoEarthLLM, llm),
        registry,
        cast(StudioClient, studio),
        state=ThreadState(provenance=ProvenanceLog(run_id=str(brief["run_id"]))),
        skill_index=str(settings.get("skill_index", "")),
        local=False,
        check_claims=False,
        surface=SURFACE,
    )


def _history(raw: Any) -> list[dict[str, str]]:
    """Prior turns for ``brief.json``: user and assistant messages with text."""
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise BridgeError('history is a list of {"role", "content"} objects')
    out: list[dict[str, str]] = []
    for n, turn in enumerate(raw, 1):
        if (
            not isinstance(turn, dict)
            or turn.get("role") not in ("user", "assistant")
            or not isinstance(turn.get("content"), str)
        ):
            raise BridgeError(
                f"history turn {n} must be "
                '{"role": "user" or "assistant", "content": "..."}'
            )
        out.append({"role": turn["role"], "content": turn["content"]})
    return out


def _messages(history: Iterable[dict[str, str]]) -> list[Message]:
    """The prior turns as the loop's messages."""
    return [Message(role=cast(Role, t["role"]), content=t["content"]) for t in history]


class _SpillNames:
    """Stands in for :mod:`uuid` in ``harness/spill.py`` during a step.

    The loop names a spilled result's file with ``uuid4().hex[:8]``. A replay
    that drew a new name would show the model another path than the one it
    read, and the answer checks would not count the path it names as a file
    this run wrote. Here the n-th spill of a run gets the same name on every
    step: the digest of the run's id and n.
    """

    def __init__(self, run_id: str) -> None:
        self._run_id = run_id
        self._count = 0

    def uuid4(self) -> uuid.UUID:
        """The name of the run's next spill, as a UUID."""
        self._count += 1
        digest = hashlib.sha256(f"{self._run_id}:{self._count}".encode()).digest()
        return uuid.UUID(bytes=digest[:16], version=4)


@contextmanager
def stable_spill_names(run_id: str) -> Iterator[None]:
    """Within the block, the n-th result the loop spills is named alike each step.

    The only place the bridge reaches into the harness: ``spill.uuid`` is
    replaced for the block and put back after it.
    """
    saved = spill.uuid
    setattr(spill, "uuid", _SpillNames(run_id))
    try:
        yield
    finally:
        setattr(spill, "uuid", saved)


async def _drive(
    agent: LeadAgent, llm: ScriptedLLM, brief: dict[str, Any]
) -> list[dict[str, Any]]:
    """Run the loop to the model call nothing answers yet, or to its end.

    The loop runs as a task; when :class:`ScriptedLLM` signals the pending
    call, the task is cancelled and awaited. A :class:`BridgeHalt` or any
    other error of the loop is raised here.
    """
    events: list[dict[str, Any]] = []

    async def consume() -> None:
        # Never left early: a cancel or an error ends the generator in place.
        async for event in agent.run_stream(
            str(brief["brief"]),
            max_turns=int(brief["settings"]["max_turns"]),
            history=_messages(brief.get("history") or []),
        ):
            # Kept as they come, so a cancelled loop leaves those so far.
            events.append(event)
            logger.debug("bridge: loop event %s", event.get("type"))

    loop = asyncio.ensure_future(consume())
    waiting = asyncio.ensure_future(llm.waiting.wait())
    tasks: list[asyncio.Future[Any]] = [loop, waiting]
    try:
        await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.wait(tasks)
    error = None if loop.cancelled() else loop.exception()
    if llm.pending is None and error is not None:
        raise error
    return events


def _answered(
    pending: dict[str, Any] | None, responses: list[RecordedResponse]
) -> bool:
    """Whether the pending call's response is saved already.

    A step saves the response it was given, then writes ``pending.json``
    last; one stopped in between leaves the pending call's response as the
    last one saved, with the request's fingerprint.
    """
    return bool(
        pending is not None
        and pending.get("status") == MODEL_CALL
        and responses
        and pending.get("model_call") == len(responses)
        and responses[-1].request_sha256 == pending.get("request_sha256")
    )


def _check_switches(agent: LeadAgent, settings: dict[str, Any]) -> None:
    """The answer checks and the read-only switch must stay as they were.

    A run begun before ``read_only`` was recorded has no such setting, and
    only a divergence shows a change of it.
    """
    for name, variable in (
        ("check_numbers", "OLMOEARTH_CHECK_NUMBERS"),
        ("check_answer", "OLMOEARTH_CHECK_ANSWER"),
    ):
        now = bool(getattr(agent, name))
        if now != bool(settings.get(name)):
            raise BridgeError(
                f"{name} was {settings.get(name)} when the run began and is {now} "
                f"now ({variable}): keep it as it was for the whole run"
            )
    if "read_only" in settings:
        now = opted_in(READ_ONLY_ENV)
        if now != bool(settings["read_only"]):
            raise BridgeError(
                f"read_only was {settings['read_only']} when the run began and is "
                f"{now} now ({READ_ONLY_ENV}, which scripts/oe-bridge.sh sets "
                "unless OE_BRIDGE_ALLOW_WRITES=1): keep it as it was for the whole "
                "run"
            )


# --- the commands ----------------------------------------------------------


async def begin(
    run_dir: str | Path,
    brief: str,
    *,
    max_turns: int = DEFAULT_MAX_TURNS,
    history: Any = None,
    skill_index: str | None = None,
    registry_factory: RegistryFactory | None = None,
    studio: StudioClient | None = None,
) -> dict[str, Any]:
    """Start a run in ``run_dir`` and play it to its first model call.

    Parameters
    ----------
    run_dir
        The run's directory; it must not exist or be empty.
    brief
        The user's request, word for word.
    max_turns
        The turn cap, as ``olmoearth-agent --max-turns``.
    history
        Prior turns, ``[{"role": "user" | "assistant", "content": ...}]``.
    skill_index
        The system prompt's skill index; by default read now
        (``SkillLoader().index()``) and kept for the whole run.
    registry_factory
        Builds the tool registry each step; ``build_default_registry``.
    studio
        A Studio client to use (not closed here); by default each step makes
        one with ``StudioClient.from_env`` and closes it.

    Returns
    -------
    dict
        The first step's view (:func:`pending_view`).
    """
    if not brief.strip():
        raise BridgeError("the brief is empty")
    if skill_index is None:
        skill_index = SkillLoader().index()
    record: dict[str, Any] = {
        "bridge": BRIDGE_VERSION,
        "run_id": uuid.uuid4().hex,
        "created": datetime.now(timezone.utc).isoformat(),
        "brief": brief,
        "history": _history(history),
        "settings": {
            "max_turns": int(max_turns),
            "surface": SURFACE,
            "local": False,
            "check_claims": False,
            "skill_index": skill_index,
            # It decides the tools offered and the capability card: a later
            # step with another setting is refused (_check_switches).
            "read_only": opted_in(READ_ONLY_ENV),
        },
    }
    factory = registry_factory or build_default_registry
    probe = build_agent(ScriptedLLM([]), factory(), None, record)
    record["settings"]["check_numbers"] = probe.check_numbers
    record["settings"]["check_answer"] = probe.check_answer
    RunDir(run_dir).create(record)
    return await step(run_dir, registry_factory=factory, studio=studio)


async def step(
    run_dir: str | Path,
    response: Any = None,
    *,
    call: int | None = None,
    registry_factory: RegistryFactory | None = None,
    studio: StudioClient | None = None,
) -> dict[str, Any]:
    """Give the pending model call its ``response``, replay, and stop at the next.

    Parameters
    ----------
    run_dir
        The run's directory (made by :func:`begin`).
    response
        The model's response to the pending call: ``{"tool_calls": [{"name",
        "arguments"}]}`` or ``{"content": "..."}``. ``None`` only to start a
        run whose first step failed, or to finish a step that saved its
        response and stopped before it ended.
    call
        The model call ``response`` answers (the view's ``model_call``), or
        ``None``; another pending call is refused before anything runs, so a
        response sent again after its call was answered is never played on
        the next.
    registry_factory, studio
        As for :func:`begin`.

    Returns
    -------
    dict
        :func:`pending_view` for the next model call, or :func:`final_view`;
        with ``dropped_tool_calls`` when the step dropped provisional records.

    Raises
    ------
    BridgeError
        No run, a step running on the run (its lock held), a malformed
        response, a response for another call, a run that is not waiting for
        one, or a replay that diverged from the recorded run.

    Notes
    -----
    A step that fails saves no response and leaves ``pending.json`` as it
    was. The tool calls it ran stay in the cache as provisional records: the
    same response sent again reuses them, and another response drops them
    (:class:`~olmoearth_agent.bridge.replay.ToolCache`; a Studio write among
    them may already have happened). A step stopped after it saved the
    response and before it wrote ``pending.json`` is finished by the next
    step, given no response or the same one. The whole step holds the run's
    lock (:meth:`~olmoearth_agent.bridge.store.RunDir.locked`).
    """
    run = RunDir(run_dir)
    run.read_brief()  # no run: refused before a lock is made in no directory
    with run.locked():
        return await _step(
            run, response, call=call, registry_factory=registry_factory, studio=studio
        )


def _unchanged(run: RunDir, saved: int, pending: dict[str, Any] | None) -> None:
    """Refuse to save when the run's responses or pending call changed meanwhile.

    The lock keeps a second step out; this check keeps ``responses.jsonl`` and
    ``pending.json`` consistent even so (a lock taken over from a step that
    was not dead, say).
    """
    now = run.read_pending()
    before = None if pending is None else pending.get("request_sha256")
    after = None if now is None else now.get("request_sha256")
    if len(run.read_responses()) != saved or before != after:
        raise BridgeError(
            "another step saved a response or a view on this run meanwhile; this "
            "step saved nothing: run show"
        )


async def _step(
    run: RunDir,
    response: Any,
    *,
    call: int | None,
    registry_factory: RegistryFactory | None,
    studio: StudioClient | None,
) -> dict[str, Any]:
    """:func:`step` under the run's lock."""
    brief = run.read_brief()
    pending = run.read_pending()
    responses = run.read_responses()
    saved = len(responses)
    unfinished = _answered(pending, responses)
    given: RecordedResponse | None = None
    if pending is not None and pending.get("status") != MODEL_CALL:
        raise BridgeError("the run has ended: show prints its answer")
    if response is None:
        if pending is not None and not unfinished:
            raise BridgeError(
                "the run is waiting for a model response: pass one (show prints "
                "the pending call)"
            )
    else:
        if pending is None:
            raise BridgeError(
                "the run has not started: run step without a response first "
                "(scripts/oe-bridge.sh step RUN, with no response file)"
            )
        if call is not None and call != pending.get("model_call"):
            raise BridgeError(
                f"the response answers model call {call}, and the pending call is "
                f"model call {pending.get('model_call')}: nothing ran; run show "
                "and answer the pending call"
            )
        parsed = RecordedResponse.parse(
            response, request_sha256=str(pending["request_sha256"])
        )
        if parsed.tool_calls and not pending.get("tools"):
            raise BridgeError(RESPOND_WITHOUT_TOOLS)
        if unfinished:
            if parsed.to_record() != responses[-1].to_record():
                raise BridgeError(
                    f"model call {saved} already has its response saved (the step "
                    "that took it stopped before it ended): send the same "
                    "response again, or run step without one"
                )
        elif pending.get("model_call") != saved + 1:
            raise BridgeError(
                f"{run.RESPONSES} holds {saved} responses but the pending "
                f"call is model call {pending.get('model_call')}"
            )
        else:
            given = parsed
            responses.append(given)

    cache = ToolCache(run, saved=saved, model_call=len(responses))
    llm = ScriptedLLM(responses)
    registry = replay_registry((registry_factory or build_default_registry)(), cache)
    client = StudioClient.from_env() if studio is None else studio
    try:
        agent = build_agent(llm, registry, client, brief)
        _check_switches(agent, brief["settings"])
        with stable_spill_names(str(brief["run_id"])):
            events = await _drive(agent, llm, brief)
    except BridgeHalt as halt:
        raise BridgeError(str(halt)) from None
    finally:
        if studio is None:
            await client.aclose()

    if cache.position < cache.committed:
        raise BridgeError(
            f"the run diverged: the replay made {cache.position} of the "
            f"{cache.committed} tool calls the saved responses made"
        )
    if llm.pending is not None:
        view = pending_view(brief, llm, agent)
    elif events and events[-1].get("type") == "final":
        if len(llm.calls) != len(responses):
            raise BridgeError(
                f"the run diverged: the loop ended after {len(llm.calls)} model "
                f"calls and {len(responses)} responses are recorded"
            )
        view = final_view(brief, events, agent.state)
    else:  # pragma: no cover - run_stream always ends with its final event
        raise BridgeError("the loop stopped without a final event")
    _unchanged(run, saved, pending)
    cache.drop_unread()
    if cache.dropped:
        view["dropped_tool_calls"] = cache.dropped

    if given is not None:
        run.append_response(given)
    run.write_events(events)
    run.write_provenance(agent.state.provenance.to_json())
    run.write_pending(view)
    return view


def show(run_dir: str | Path) -> dict[str, Any]:
    """The last step's view again (``pending.json``); nothing runs.

    While a step holds the run's lock, the view carries ``step_running`` (who
    and since when): the pending call shown is then the one that step is
    answering, and the response is not to be sent again.
    """
    run = RunDir(run_dir)
    run.read_brief()
    held = run.read_lock()
    view = run.read_pending()
    if view is None:
        if held is not None:
            raise BridgeError(
                f"{run.running_text(held)}: the run's first step has not ended; "
                "wait for it to end, then run show"
            )
        raise BridgeError(
            "the run has no pending call yet: its first step did not end; run "
            "step without a response to start it (scripts/oe-bridge.sh step RUN, "
            "with no response file), or begin again under a new run name"
        )
    if held is not None:
        view = {**view, "step_running": held}
    return view


def system_prompt(
    *,
    registry_factory: RegistryFactory | None = None,
    skill_index: str | None = None,
) -> str:
    """The system prompt a new run's first model call gets.

    The soul, the capability card of the registry's tools and the skill
    index, as :func:`begin` would build them now.
    """
    if skill_index is None:
        skill_index = SkillLoader().index()
    record: dict[str, Any] = {
        "run_id": "prompt",
        "settings": {"skill_index": skill_index},
    }
    factory = registry_factory or build_default_registry
    return build_agent(ScriptedLLM([]), factory(), None, record).system_prompt


def _spec_view(spec: ToolSpec) -> dict[str, Any]:
    """One tool spec as JSON: its name, description and JSON Schema."""
    return {
        "name": spec.name,
        "description": spec.description,
        "parameters": spec.parameters,
    }


def tool_specs(
    run_dir: str | Path | None = None,
    *,
    registry_factory: RegistryFactory | None = None,
) -> list[dict[str, Any]]:
    """The specs of the tools offered: on a new run's first call, or the pending one.

    With ``run_dir``, the tools offered on the run's pending call (none when
    it offers none, or when the run has ended).
    """
    registry = (registry_factory or build_default_registry)()
    if run_dir is None:
        names = [s.name for s in registry.active_specs()]
    else:
        view = show(run_dir)
        offered = view.get("tools") if view.get("status") == MODEL_CALL else None
        names = [str(n) for n in offered or []]
    specs = [registry.spec_of(name) for name in names]
    return [_spec_view(s) for s in specs if s is not None]


# --- the views -------------------------------------------------------------


def _message_view(message: Message) -> dict[str, Any]:
    """One message as the model receives it, without the empty fields."""
    out: dict[str, Any] = {"role": message.role}
    if message.name is not None:
        out["name"] = message.name
    if message.tool_call_id is not None:
        out["tool_call_id"] = message.tool_call_id
    if message.tool_calls:
        out["tool_calls"] = [
            {"id": c.id, "name": c.name, "arguments": c.arguments}
            for c in message.tool_calls
        ]
    out["content"] = message.content
    return out


def _sha256(text: str | None) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def pending_view(
    brief: dict[str, Any], llm: ScriptedLLM, agent: LeadAgent
) -> dict[str, Any]:
    """What the model needs to answer the pending call.

    ``messages`` are the ones it has not seen: everything after its last
    response (all of them on the first call). ``system`` holds the system
    prompt's text on the first call and whenever it changed (a loaded group
    adds its tools to the card), else only its sha256. ``tool_specs`` holds
    the offered tools' whole specs (name, description, JSON Schema) on the
    first call and whenever the tools offered changed, which
    ``tool_specs_changed`` says: the capability card names required
    arguments only, and an argument a spec does not declare is passed
    through and ignored, so a misnamed one would run the tool on its default.
    """
    call = llm.pending
    if call is None:
        raise BridgeError("no model call is pending")
    previous = llm.calls[-2] if len(llm.calls) > 1 else None
    # The model's last response is one assistant message after the
    # conversation of its call; the conversation only grows.
    seen = 0 if previous is None else len(previous.messages) + 1
    changed = previous is None or previous.system != call.system
    system: dict[str, Any] = {"sha256": _sha256(call.system), "changed": changed}
    if changed:
        system["text"] = call.system
    specs_changed = previous is None or previous.tool_names != call.tool_names
    view: dict[str, Any] = {
        "status": MODEL_CALL,
        "run_id": brief["run_id"],
        "model_call": call.index,
        "turn": agent.state.turn_count,
        "max_turns": brief["settings"]["max_turns"],
        "mode": call.mode,
        "tools": call.tool_names,
        "tool_specs_changed": specs_changed,
        "system": system,
        "messages": [_message_view(m) for m in call.messages[seen:]],
        "request_sha256": call.sha256,
        "respond": RESPOND_WITH_TOOLS if call.tools else RESPOND_WITHOUT_TOOLS,
    }
    if specs_changed and call.tools:
        view["tool_specs"] = [_spec_view(t) for t in call.tools]
    return view


def _collect(events: Iterable[dict[str, Any]], state: ThreadState) -> AgentResult:
    """The events as an :class:`AgentResult`, as ``LeadAgent.run`` assembles it."""
    result = AgentResult(final_content=None, turns=0, state=state)
    for event in events:
        kind = event["type"]
        if kind == "tool_result":
            result.tool_calls.append((event["name"], event["ok"]))
        elif kind == "final":
            result.final_content = event["content"]
            result.turns = event["turn"]
            result.grounding_revised = bool(event.get("grounding_revised"))
            result.revised = bool(event.get("revised"))
            result.marked = list(event.get("marked") or [])
        elif kind == "max_turns":
            result.hit_max_turns = True
            result.turns = event["turns"]
        elif kind == "grounding_check":
            result.grounding_checks.append(event)
        elif kind == "check":
            result.checks.append(event)
    return result


def cli_notes(result: AgentResult, max_turns: int) -> list[str]:
    """The notes ``olmoearth-agent`` prints to stderr beside the answer (``cli.py``)."""
    if result.final_content is None:
        return ["(no answer, hit the turn cap)"]
    notes: list[str] = []
    if result.hit_max_turns:
        notes.append(
            f"(reached the turn cap ({max_turns}): the answer was written "
            "from what the tools returned, without further tool calls)"
        )
    notes += [
        "(numbers in the answer that no tool returned: "
        + ", ".join(check.get("unsupported", []))
        + ")"
        for check in result.grounding_checks
        if check.get("action") == "shown"
    ]
    notes += [
        f"(the {check.get('check')} check still flags "
        f"{len(check.get('violations', []))} statement(s) of the answer; "
        "each is marked [unverified])"
        for check in result.checks
        if check.get("action") == "marked" and check.get("check") != "numbers"
    ]
    return notes


def cli_trace(result: AgentResult) -> list[str]:
    """What ``olmoearth-agent --show-trace`` prints to stderr (``cli.py``)."""
    lines = [f"  [{'ok' if ok else 'FAIL'}] {name}" for name, ok in result.tool_calls]
    lines += [
        "  [numbers] in no tool result, rewrite asked: "
        + ", ".join(check.get("unsupported", []))
        for check in result.grounding_checks
        if check.get("action") == "revise"
    ]
    lines += [
        f"  [{check.get('check')}] {len(check.get('violations', []))} "
        "sentence(s) flagged, rewrite asked"
        for check in result.checks
        if check.get("action") == "revise" and check.get("check") != "numbers"
    ]
    if result.state is not None:
        lines.append(
            f"  ({result.turns} turn(s), "
            f"{len(result.state.provenance.entries)} provenance entries)"
        )
    return lines


def final_view(
    brief: dict[str, Any], events: list[dict[str, Any]], state: ThreadState
) -> dict[str, Any]:
    """The run's answer, its flags, and ``olmoearth-agent``'s notes and trace."""
    final = events[-1]
    result = _collect(events, state)
    return {
        "status": FINAL,
        "run_id": brief["run_id"],
        "content": result.final_content,
        "turns": result.turns,
        "forced_by_turn_cap": bool(final.get("forced_by_turn_cap")),
        "hit_max_turns": result.hit_max_turns,
        "revised": result.revised,
        "grounding_revised": result.grounding_revised,
        "marked": result.marked,
        "notes": cli_notes(result, int(brief["settings"]["max_turns"])),
        "trace": cli_trace(result),
    }


def _message_text(message: dict[str, Any]) -> list[str]:
    """One message of a view, as ``--text`` prints it."""
    role = message.get("role")
    if role == "tool":
        name, call_id = message.get("name"), message.get("tool_call_id")
        lines = [f"-- tool result: {name} ({call_id}) --"]
    else:
        lines = [f"-- {role} --"]
    lines += [
        f"[tool call {call.get('id')}] {call.get('name')} {call.get('arguments')}"
        for call in message.get("tool_calls") or []
    ]
    if message.get("content") is not None:
        lines.append(str(message["content"]))
    return lines


def _dropped_text(view: dict[str, Any]) -> list[str]:
    """The bridge's note on the provisional tool calls a step dropped, if any."""
    dropped = view.get("dropped_tool_calls") or []
    if not dropped:
        return []
    names = ", ".join(str(d.get("name")) for d in dropped)
    return [
        f"(bridge: dropped {len(dropped)} tool call(s) ({names}) that an earlier "
        "step made for this model call before it failed; a Studio write among "
        "them may already have happened)"
    ]


def _running_text(view: dict[str, Any]) -> list[str]:
    """The bridge's note that a step still runs on the run, if ``show`` saw one."""
    held = view.get("step_running")
    if not isinstance(held, dict):
        return []
    return [
        f"(bridge: {RunDir.running_text(held)}: the view below is from before "
        "it; wait for it to end, then run show again, and do not send the "
        "response again meanwhile)"
    ]


def _specs_text(view: dict[str, Any]) -> list[str]:
    """The offered tools' specs as ``--text`` prints them, when the view has them."""
    specs = view.get("tool_specs")
    if not specs:
        if view.get("tools"):
            return ["-- tool specs unchanged since the last call --"]
        return []
    lines = ["-- tool specs (use only the argument names each declares) --"]
    for spec in specs:
        parameters = json.dumps(spec.get("parameters"), ensure_ascii=False)
        lines += [
            f"- {spec.get('name')}: {spec.get('description')}",
            f"  parameters: {parameters}",
        ]
    return lines


def render_text(view: dict[str, Any]) -> str:
    """The readable form of a view (``--text``)."""
    if view.get("status") == FINAL:
        content = view.get("content")
        flags = [f for f in ("revised", "forced_by_turn_cap") if view.get(f)]
        if view.get("marked"):
            flags.append("marked: " + ", ".join(view["marked"]))
        lines = [
            *_running_text(view),
            f"== final answer (run {view.get('run_id')}) ==",
            "(no answer)" if content is None else str(content),
            "== " + ("; ".join(flags) if flags else "not revised, not marked") + " ==",
            *view.get("notes", []),
            *_dropped_text(view),
        ]
        return "\n".join(lines)
    tools = view.get("tools")
    system = view.get("system") or {}
    lines = [
        *_running_text(view),
        f"== model call {view.get('model_call')} "
        f"(turn {view.get('turn')} of {view.get('max_turns')}) ==",
        *_dropped_text(view),
        f"mode: {view.get('mode')}",
        "tools: " + (", ".join(tools) if tools else "none"),
        *_specs_text(view),
    ]
    if system.get("changed"):
        lines.append(f"-- system prompt (sha256 {system.get('sha256')}) --")
        lines.append(str(system.get("text")))
    else:
        lines.append(f"-- system prompt unchanged (sha256 {system.get('sha256')}) --")
    for message in view.get("messages", []):
        lines += _message_text(message)
    lines.append(f"== {view.get('respond')} ==")
    return "\n".join(lines)
