# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Tool registry: maps tool names to JSON-Schema specs and async handlers.

A skill contributes a *bundle* of :class:`RegisteredTool` objects to the
registry. The lead agent exposes ``registry.active_specs(loaded)`` to the LLM
on every turn and routes each emitted ``ToolCall`` back through
:meth:`ToolRegistry.dispatch`. This is the seam every skill plugs into
(``SKILLS.md``).

Deferred loading. A bundle registered with a ``group`` (a catalog skill's
name) is *deferred*: its specs are not sent on every turn, only once the run
has loaded that group, which ``olmoearth_load_skill`` does when the model
loads the skill, and the web UI's forced skill does before the first turn.
Every other tool is *core* and is sent on every turn; the default registry
defers none. A deferred tool stays callable: dispatching one loads its group,
so its spec is sent from the next turn on.

Opt-in tools. A tool behind an opt-in setting is not registered at all unless
the operator sets that environment variable to a truthy value
(:func:`opted_in`): ``OLMOEARTH_OPT_IN_TOOLS`` for the tools left out of the
default registry (``skills.registry.OPT_IN_TOOLS``). Unlike a deferred tool,
an opt-in tool that is off cannot be called or loaded by the model, and no
text the default agent sends it (tool descriptions, capability card, soul,
SKILL.md packages) names one of ``OPT_IN_TOOLS``;
``tests/skills/test_skill_registry.py`` checks that. The opposite switch, ``OLMOEARTH_READ_ONLY`` (the same values), leaves
out the tools that write to Studio (``skills.registry.STUDIO_WRITE_TOOLS``),
and :attr:`ToolRegistry.cannot` puts the reason on the capability card.

A handler's exception becomes a failed envelope with a generic hint; a
:class:`~olmoearth_agent.studio.client.StudioAccessError` (Studio refused the
key), or the bridge's replay of one, brings its own hint instead
(:func:`own_hint`), and any exception's ``contract`` keys are put beside the
error.
"""

from __future__ import annotations

import os
import re
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from olmoearth_agent.llm.types import ToolCall, ToolSpec
from olmoearth_agent.studio.client import StudioAccessError, StudioClient
from olmoearth_agent.tools.validate import schema_summary, validate_arguments

if TYPE_CHECKING:
    from olmoearth_agent.harness.state import ThreadState


@dataclass
class ToolContext:
    """Everything a tool handler needs to do its work."""

    studio: StudioClient
    state: ThreadState


#: The hint on a tool's failure the second time in a run it fails the same way.
STOP_RETRYING_HINT = (
    "This tool has now failed {count} times in this run with this same error "
    "(numbers aside). Stop retrying it: tell the user what the error says the "
    "limit or problem is, and answer with what you have."
)

_NUMBER = re.compile(r"\d+(?:\.\d+)?")

#: The values that switch an opt-in setting on (case and spaces aside).
OPT_IN_VALUES = frozenset({"1", "true", "yes", "on"})


def opted_in(variable: str) -> bool:
    """True only if the operator set the opt-in variable ``variable`` on.

    The one switch of every opt-in tool bundle (see the module docstring):
    ``1``, ``true``, ``yes`` or ``on``; unset, empty or anything else is off.

    Parameters
    ----------
    variable
        The environment variable's name, e.g. ``OLMOEARTH_OPT_IN_TOOLS``.

    Returns
    -------
    bool
        Whether the bundle behind ``variable`` is registered.

    Examples
    --------
    >>> opted_in("OLMOEARTH_OPT_IN_TOOLS")  # doctest: +SKIP
    False
    """
    return os.environ.get(variable, "").strip().lower() in OPT_IN_VALUES


def note_failure(envelope: dict[str, Any], name: str, thread_state: Any) -> None:
    """Count a failed envelope in the run; from the second alike, tell the model to stop.

    exp86 round 1: the model met one refusal ("budget 300 ... 173 valid
    windows") five times, with a new grid each time, and never answered.
    "Alike" is the same tool with the same error once its numbers are masked.
    The counts live on the run's state (``ThreadState.tool_failures``), so a
    new run starts afresh. A successful envelope, or a state without the
    counts (as in some unit tests), changes nothing.
    """
    if envelope.get("ok") is not False:
        return
    failures = getattr(thread_state, "tool_failures", None)
    if not isinstance(failures, dict):
        return
    key = (name, _NUMBER.sub("N", str(envelope.get("error", ""))))
    count = failures.get(key, 0) + 1
    failures[key] = count
    if count >= 2:
        envelope["same_error_count"] = count
        envelope["hint"] = STOP_RETRYING_HINT.format(count=count)


#: A tool handler: receives parsed arguments + context, returns any
#: JSON-serializable result.
Handler = Callable[[dict[str, Any], ToolContext], Awaitable[Any]]


#: The output contract's keys a refusal may carry onto its failed envelope.
_CONTRACT_KEYS = ("facts", "must_state", "forbidden_claims")


def own_hint(exc: BaseException) -> str | None:
    """The hint a handler's exception brings in place of the generic one, if any.

    A :class:`~olmoearth_agent.studio.client.StudioAccessError` brings one
    (Studio refused the request: no id is wrong and a retry cannot help), and
    so does the bridge's replay of one, which carries the recorded hint and
    marks its class ``replays_hint`` (``bridge.replay.ReplayedToolError``;
    the bridge imports this module, so its class is not imported here). No
    other exception's ``hint`` attribute is read.
    """
    if not (
        isinstance(exc, StudioAccessError)
        or getattr(type(exc), "replays_hint", False) is True
    ):
        return None
    hint = getattr(exc, "hint", None)
    return hint if isinstance(hint, str) and hint else None


@dataclass(frozen=True)
class Capability:
    """What one tool does, what it needs, and what it cannot do, in words.

    The system prompt's capability card is assembled from these
    (:func:`olmoearth_agent.harness.capabilities.capability_card`). exp86
    round 9: the model answers only from its tools, and 7 of the audit's 16
    material findings were offers of actions no tool can do or whose
    preconditions did not hold: a review set of a regression band with no
    threshold, "I'll set up a direct model run", labels on flagged windows
    turned into "a proper error-rate estimate", a comparison "rerun with
    labels_date", which takes no labels. Nothing the model saw said what the
    tools cannot do, so it filled the gap from general knowledge.

    Declared beside the handler, so a change to one is made beside the
    other. Every entry of ``needs`` and ``cannot`` is a condition a test
    calls the tool in (``tests/tools/test_tool_capabilities.py``), getting the
    refusal or the absence it states, so the card cannot drift from the code.

    Attributes
    ----------
    does
        What the tool does, in one line; its required arguments come from the
        spec, not from here.
    needs
        Preconditions the tool enforces, each a short phrase (e.g. "threshold,
        for a regression band not in [0, 1]"); without one it refuses.
    cannot
        What the tool refuses or does not do, each a short phrase (e.g. "say
        which map is right: it takes no labels").
    covers
        Ids of :data:`olmoearth_agent.harness.capabilities.NO_TOOL_CAN` this
        tool can do after all; the card's list of what no tool can do drops
        them (e.g. a tool that read any file would cover ``files``).
    """

    does: str
    needs: tuple[str, ...] = ()
    cannot: tuple[str, ...] = ()
    covers: frozenset[str] = frozenset()


@dataclass
class RegisteredTool:
    """A JSON-Schema spec paired with its async handler and its capability."""

    spec: ToolSpec
    handler: Handler
    #: The tool's line on the capability card; every tool of the default
    #: registry has one (``tests/tools/test_tool_capabilities.py``).
    capability: Capability | None = None


class ToolRegistry:
    """Holds the agent's callable tools.

    Names are unique; registering a duplicate name overwrites the prior
    entry (so a skill can intentionally override a default tool).
    """

    def __init__(self) -> None:
        self._tools: dict[str, RegisteredTool] = {}
        #: Deferred tools only: tool name -> the group that loads it.
        self._group_of: dict[str, str] = {}
        #: What a setting of this process rules out that no tool's declaration
        #: says, by id -> its line (e.g. a read-only Studio key; see
        #: ``skills.registry.build_default_registry``). The capability card
        #: lists each under what no tool can do
        #: (:func:`olmoearth_agent.harness.capabilities.beyond_tools`).
        self.cannot: dict[str, str] = {}

    def register(self, tool: RegisteredTool, *, group: str | None = None) -> None:
        """Add (or replace) one tool; ``group`` makes it deferred (see module doc)."""
        name = tool.spec.name
        self._tools[name] = tool
        if group:
            self._group_of[name] = group
        else:
            self._group_of.pop(name, None)

    def register_all(
        self, tools: Iterable[RegisteredTool], *, group: str | None = None
    ) -> None:
        """Add (or replace) a bundle of tools, deferred under ``group`` if given."""
        for tool in tools:
            self.register(tool, group=group)

    def names(self) -> list[str]:
        """Registered tool names (core and deferred), in insertion order."""
        return list(self._tools)

    def specs(self) -> list[ToolSpec]:
        """Every registered tool spec, core and deferred."""
        return [t.spec for t in self._tools.values()]

    def groups(self) -> dict[str, list[str]]:
        """Deferred groups: group name -> its tool names, in insertion order."""
        out: dict[str, list[str]] = {}
        for name in self._tools:
            group = self._group_of.get(name)
            if group:
                out.setdefault(group, []).append(name)
        return out

    def group_of(self, name: str) -> str | None:
        """The group a deferred tool belongs to, or ``None`` for a core tool."""
        return self._group_of.get(name)

    def spec_of(self, name: str) -> ToolSpec | None:
        """The spec of the tool called ``name``, or ``None`` if none is registered."""
        tool = self._tools.get(name)
        return None if tool is None else tool.spec

    def capability_of(self, name: str) -> Capability | None:
        """The declared capability of the tool called ``name``, or ``None``."""
        tool = self._tools.get(name)
        return None if tool is None else tool.capability

    def tool_of(self, name: str) -> RegisteredTool | None:
        """The registered tool called ``name`` (spec, handler, capability), or ``None``.

        The replay bridge (:mod:`olmoearth_agent.bridge`) copies a registry
        with each handler wrapped, keeping the order, groups and specs.
        """
        return self._tools.get(name)

    def active_specs(self, loaded: Iterable[str] = ()) -> list[ToolSpec]:
        """The specs to send on one turn: every core tool plus ``loaded`` groups."""
        groups = set(loaded)
        return [
            t.spec
            for name, t in self._tools.items()
            if name not in self._group_of or self._group_of[name] in groups
        ]

    async def dispatch(
        self, call: ToolCall, ctx: ToolContext, *, retry_hint: bool = True
    ) -> dict[str, Any]:
        """Execute one tool call, returning a JSON-able result envelope.

        Never raises: an unknown tool, malformed arguments, or a handler
        exception is returned as ``{"ok": False, "error": ...}`` so the agent
        loop can feed the failure back to the model and let it recover.

        Arguments are validated against the tool's declared JSON Schema
        *before* the handler runs (see :mod:`olmoearth_agent.tools.validate`),
        so a missing/mistyped argument comes back as a self-documenting
        rejection naming the argument and the expected shape — not as a bare
        ``KeyError`` from handler internals.

        The second time in a run a tool fails with the same error (numbers
        aside), the envelope's ``hint`` tells the model to stop retrying and
        report the limit to the user, and ``same_error_count`` counts it
        (:func:`note_failure`). ``retry_hint=False``
        leaves that out: the lead agent passes it, and its
        ``RetryHintMiddleware`` adds the hint around this call instead.
        """
        tool = self._tools.get(call.name)
        if tool is None:
            return {
                "ok": False,
                "error": f"unknown tool: {call.name!r}",
                "available": self.names(),
                "hint": "Call one of the available tools exactly by name.",
            }
        envelope = await self._execute(tool, call, ctx)
        if retry_hint:
            note_failure(envelope, call.name, getattr(ctx, "state", None))
        return envelope

    async def _execute(
        self, tool: RegisteredTool, call: ToolCall, ctx: ToolContext
    ) -> dict[str, Any]:
        """Validate the arguments and run the handler; never raises."""
        group = self._group_of.get(call.name)
        loaded = getattr(getattr(ctx, "state", None), "loaded_groups", None)
        if group and isinstance(loaded, set):
            # A deferred tool called before its skill was loaded (e.g. a name
            # remembered from an earlier conversation): run it, and send its
            # group's specs from the next turn on.
            loaded.add(group)
        problems = validate_arguments(call.arguments, tool.spec.parameters)
        if problems:
            return {
                "ok": False,
                "error": f"invalid arguments for {call.name}: " + "; ".join(problems),
                "expected_arguments": schema_summary(tool.spec.parameters),
                "hint": "Fix the named arguments to match "
                "'expected_arguments' and call the tool again.",
            }
        try:
            result = await tool.handler(call.arguments, ctx)
        except Exception as exc:  # noqa: BLE001 - surfaced to the model, not swallowed
            failed: dict[str, Any] = {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "tool": call.name,
                "hint": "The tool ran but failed. Check that referenced ids "
                "exist (discover them with search/list tools) and that "
                "argument values are valid; do not retry with identical "
                "arguments.",
            }
            hint = own_hint(exc)
            if hint is not None:
                # Studio refused the key (HTTP 401/403), live or replayed by
                # the bridge: no id is wrong and a retry cannot help, so the
                # error's own hint replaces the above.
                failed["hint"] = hint
            # A refusal can carry the output contract's keys (e.g. the labels
            # a budget leaves over; statistical_rules.refusal): stated beside
            # the error, which is unchanged.
            contract = getattr(exc, "contract", None)
            if isinstance(contract, dict):
                failed.update(
                    {k: v for k, v in contract.items() if k in _CONTRACT_KEYS}
                )
            return failed
        return {"ok": True, "result": result}
