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

    def register(self, tool: RegisteredTool) -> None:
        """Add (or replace) one tool."""
        self._tools[tool.spec.name] = tool

    def register_all(self, tools: Iterable[RegisteredTool]) -> None:
        """Add (or replace) a bundle of tools."""
        for tool in tools:
            self.register(tool)

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
