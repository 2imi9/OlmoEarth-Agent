# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""The stop-retrying hint on a tool's second alike failure, as middleware.

exp86 round 1: the model met one refusal ("budget 300 ... 173 valid
windows") five times, with a new grid each time, and never answered. From
the second time in a run a tool fails with the same error, numbers aside,
its envelope tells the model to stop retrying and report the limit.

The rule lived in ``ToolRegistry.dispatch`` until the harness had
middleware; its text and trigger are unchanged. ``dispatch`` still applies
it when called directly (``retry_hint=True``, the default); the lead agent
passes ``retry_hint=False`` and leaves it to :class:`RetryHintMiddleware`.
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from typing import Any

from olmoearth_agent.harness.middleware import (
    AgentMiddleware,
    ToolCallRequest,
    ToolResult,
)

__all__ = ["STOP_RETRYING_HINT", "RetryHintMiddleware", "note_failure"]

#: The hint on a tool's failure the second time in a run it fails the same way.
STOP_RETRYING_HINT = (
    "This tool has now failed {count} times in this run with this same error "
    "(numbers aside). Stop retrying it: tell the user what the error says the "
    "limit or problem is, and answer with what you have."
)

_NUMBER = re.compile(r"\d+(?:\.\d+)?")


def note_failure(envelope: dict[str, Any], name: str, thread_state: Any) -> None:
    """Count a failed envelope in the run; from the second alike, tell the model to stop.

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
