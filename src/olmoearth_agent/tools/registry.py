# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Tool registry: maps tool names to JSON-Schema specs and async handlers.

A skill contributes a *bundle* of :class:`RegisteredTool` objects to the
registry. The lead agent exposes ``registry.active_specs(loaded)`` to the LLM
on every turn and routes each emitted ``ToolCall`` back through
:meth:`ToolRegistry.dispatch`. This is the seam every skill plugs into
(``SKILLS.md``).

Deferred loading. A bundle registered with a ``group`` (a skill name such as
``olmoearth-rslearn``) is *deferred*: its specs are not sent on every turn,
only once the run has loaded that group, which ``olmoearth_load_skill`` does
when the model loads the skill, and the web UI's forced skill does before the
first turn. Every other tool is *core* and is sent on every turn. A deferred
tool stays callable: dispatching one loads its group, so its spec is sent from
the next turn on.
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from olmoearth_agent.llm.types import ToolCall, ToolSpec
from olmoearth_agent.studio.client import StudioClient
from olmoearth_agent.tools.validate import schema_summary, validate_arguments

if TYPE_CHECKING:
    from olmoearth_agent.harness.state import ThreadState


@dataclass
class ToolContext:
    """Everything a tool handler needs to do its work."""

    studio: StudioClient
    state: ThreadState


#: A tool handler: receives parsed arguments + context, returns any
#: JSON-serializable result.
Handler = Callable[[dict[str, Any], ToolContext], Awaitable[Any]]


#: The hint on a tool's failure the second time in a run it fails the same way.
STOP_RETRYING_HINT = (
    "This tool has now failed {count} times in this run with this same error "
    "(numbers aside). Stop retrying it: tell the user what the error says the "
    "limit or problem is, and answer with what you have."
)

_NUMBER = re.compile(r"\d+(?:\.\d+)?")

#: The output contract's keys a refusal may carry onto its failed envelope.
_CONTRACT_KEYS = ("facts", "must_state", "forbidden_claims")


def _record_failure(envelope: dict[str, Any], name: str, ctx: ToolContext) -> None:
    """Count this failure in the run; from the second alike, tell the model to stop.

    "Alike" is the same tool with the same error once its numbers are masked:
    in exp86 round 1 the model met one refusal ("budget 300 ... 173 valid
    windows") five times, with a new grid each time, and never answered.
    The counts live on the run's state, so a new run starts afresh.
    """
    failures = getattr(getattr(ctx, "state", None), "tool_failures", None)
    if not isinstance(failures, dict):
        return
    key = (name, _NUMBER.sub("N", str(envelope.get("error", ""))))
    count = failures.get(key, 0) + 1
    failures[key] = count
    if count >= 2:
        envelope["same_error_count"] = count
        envelope["hint"] = STOP_RETRYING_HINT.format(count=count)


@dataclass
class RegisteredTool:
    """A JSON-Schema spec paired with its async handler."""

    spec: ToolSpec
    handler: Handler


class ToolRegistry:
    """Holds the agent's callable tools.

    Names are unique; registering a duplicate name overwrites the prior
    entry (so a skill can intentionally override a default tool).
    """

    def __init__(self) -> None:
        self._tools: dict[str, RegisteredTool] = {}
        #: Deferred tools only: tool name -> the group that loads it.
        self._group_of: dict[str, str] = {}

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

    def active_specs(self, loaded: Iterable[str] = ()) -> list[ToolSpec]:
        """The specs to send on one turn: every core tool plus ``loaded`` groups."""
        groups = set(loaded)
        return [
            t.spec
            for name, t in self._tools.items()
            if name not in self._group_of or self._group_of[name] in groups
        ]

    async def dispatch(self, call: ToolCall, ctx: ToolContext) -> dict[str, Any]:
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
        report the limit to the user, and ``same_error_count`` counts it.
        """
        tool = self._tools.get(call.name)
        if tool is None:
            return {
                "ok": False,
                "error": f"unknown tool: {call.name!r}",
                "available": self.names(),
                "hint": "Call one of the available tools exactly by name.",
            }
        group = self._group_of.get(call.name)
        loaded = getattr(getattr(ctx, "state", None), "loaded_groups", None)
        if group and isinstance(loaded, set):
            # A deferred tool called before its skill was loaded (e.g. a name
            # remembered from an earlier conversation): run it, and send its
            # group's specs from the next turn on.
            loaded.add(group)
        problems = validate_arguments(call.arguments, tool.spec.parameters)
        if problems:
            invalid: dict[str, Any] = {
                "ok": False,
                "error": f"invalid arguments for {call.name}: " + "; ".join(problems),
                "expected_arguments": schema_summary(tool.spec.parameters),
                "hint": "Fix the named arguments to match "
                "'expected_arguments' and call the tool again.",
            }
            _record_failure(invalid, call.name, ctx)
            return invalid
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
            # A refusal can carry the output contract's keys (e.g. the labels
            # a budget leaves over; statistical_rules.refusal): stated beside
            # the error, which is unchanged.
            contract = getattr(exc, "contract", None)
            if isinstance(contract, dict):
                failed.update(
                    {k: v for k, v in contract.items() if k in _CONTRACT_KEYS}
                )
            _record_failure(failed, call.name, ctx)
            return failed
        return {"ok": True, "result": result}
