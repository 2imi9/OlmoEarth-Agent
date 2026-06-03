# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Tool registry: maps tool names to JSON-Schema specs and async handlers.

A skill contributes a *bundle* of :class:`RegisteredTool` objects to the
registry. The lead agent exposes ``registry.specs()`` to the LLM and
routes each emitted ``ToolCall`` back through :meth:`ToolRegistry.dispatch`.
This is the seam every skill plugs into (``SKILLS.md``).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from olmoearth_agent.llm.types import ToolCall, ToolSpec
from olmoearth_agent.studio.client import StudioClient

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

#: A grounded verifier: given a handler's (non-throwing) result payload, return
#: ``(ok, reason)``. ``ok=False`` means the result is semantically wrong despite
#: ``dispatch`` having wrapped it as ``{"ok": True}``; ``reason`` is a one-line
#: natural-language note the agent loop feeds back as a reflection before one
#: retry. Verifiers MUST be pure + deterministic and read a *real external
#: signal* (an oracle, a schema, the result's own fields) -- never the model
#: judging itself (see ``docs/self-improvement-proposal.md`` 1.1).
Verifier = Callable[[dict[str, Any]], tuple[bool, str]]


@dataclass
class RegisteredTool:
    """A JSON-Schema spec paired with its async handler.

    ``verify`` is optional: when set, the lead agent runs it on the handler's
    result and, on failure, appends a one-line reflection and re-prompts the
    model once (a grounded verify-and-retry gate). Default ``None`` -> the tool
    behaves exactly as before, so adding the field is a no-op for every existing
    tool.
    """

    spec: ToolSpec
    handler: Handler
    verify: Verifier | None = None


class ToolRegistry:
    """Holds the agent's callable tools.

    Names are unique; registering a duplicate name overwrites the prior
    entry (so a skill can intentionally override a default tool).
    """

    def __init__(self) -> None:
        self._tools: dict[str, RegisteredTool] = {}

    def register(self, tool: RegisteredTool) -> None:
        """Add (or replace) one tool."""
        self._tools[tool.spec.name] = tool

    def register_all(self, tools: Iterable[RegisteredTool]) -> None:
        """Add (or replace) a bundle of tools."""
        for tool in tools:
            self.register(tool)

    def get(self, name: str) -> RegisteredTool | None:
        """The registered tool with ``name`` (or ``None``).

        Lets the agent loop reach a tool's :attr:`RegisteredTool.verify`
        predicate after :meth:`dispatch` has run.
        """
        return self._tools.get(name)

    def names(self) -> list[str]:
        """Registered tool names, in insertion order."""
        return list(self._tools)

    def specs(self) -> list[ToolSpec]:
        """All tool specs, to hand to the LLM."""
        return [t.spec for t in self._tools.values()]

    async def dispatch(
        self, call: ToolCall, ctx: ToolContext
    ) -> dict[str, Any]:
        """Execute one tool call, returning a JSON-able result envelope.

        Never raises: an unknown tool or a handler exception is returned
        as ``{"ok": False, "error": ...}`` so the agent loop can feed the
        failure back to the model and let it recover.
        """
        tool = self._tools.get(call.name)
        if tool is None:
            return {
                "ok": False,
                "error": f"unknown tool: {call.name!r}",
                "available": self.names(),
            }
        try:
            result = await tool.handler(call.arguments, ctx)
        except Exception as exc:  # noqa: BLE001 - surfaced to the model, not swallowed
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        return {"ok": True, "result": result}
