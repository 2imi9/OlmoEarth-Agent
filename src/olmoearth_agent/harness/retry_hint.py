# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""The stop-retrying hint on a tool's second alike failure, as middleware.

exp86 round 1: the model met one refusal ("budget 300 ... 173 valid
windows") five times, with a new grid each time, and never answered. From
the second time in a run a tool fails with the same error, numbers aside,
its envelope tells the model to stop retrying and report the limit.

The rule (:func:`~olmoearth_agent.tools.registry.note_failure`) lived only in
``ToolRegistry.dispatch`` until the harness had middleware; its text and
trigger are unchanged. ``dispatch`` still applies it when called directly
(``retry_hint=True``, the default); the lead agent passes
``retry_hint=False`` and leaves it to :class:`RetryHintMiddleware`.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from olmoearth_agent.harness.middleware import (
    AgentMiddleware,
    ToolCallRequest,
    ToolResult,
)
from olmoearth_agent.tools.registry import STOP_RETRYING_HINT, note_failure

__all__ = ["STOP_RETRYING_HINT", "RetryHintMiddleware", "note_failure"]


class RetryHintMiddleware(AgentMiddleware):
    """Adds the stop-retrying hint to a registered tool's repeated failure.

    A call to a tool the registry does not know is not counted, as before:
    its envelope already lists the tools there are.
    """

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolResult]],
    ) -> ToolResult:
        """Run the tool; count its failure and add the hint from the second."""
        result = await handler(request)
        if request.tool is not None and isinstance(result, dict):
            note_failure(
                result, request.tool_call.name, request.state.get("thread_state")
            )
        return result
