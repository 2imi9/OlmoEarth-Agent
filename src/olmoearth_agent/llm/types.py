# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Dataclass types for the LLM client.

These mirror the OpenAI Chat Completions surface but stay independent of
the SDK so callers don't need to import ``openai`` types directly. The
client converts to and from these in
:mod:`olmoearth_agent.llm.client`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["system", "user", "assistant", "tool"]
FinishReason = Literal["stop", "tool_calls", "length", "content_filter"]


@dataclass
class ToolCall:
    """One function call the model emitted.

    ``arguments`` is the JSON-decoded payload; callers should validate it
    against the corresponding :class:`ToolSpec` before dispatch.
    """

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class Message:
    """One conversation turn.

    Use ``role="tool"`` together with ``tool_call_id`` (matching a prior
    :class:`ToolCall`) to feed a function result back to the model.
    """

    role: Role
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None
    name: str | None = None


@dataclass
class ToolSpec:
    """OpenAI-style function-tool declaration.

    ``parameters`` is a JSON Schema object (typically
    ``{"type": "object", "properties": ..., "required": [...]}``). The
    server validates arguments against this schema only loosely; callers
    should re-validate on receipt.
    """

    name: str
    description: str
    parameters: dict[str, Any]


@dataclass
class ChatResponse:
    """Parsed completion from the OpenAI-compatible LLM server.

    ``thinking`` carries the model's reasoning when it is in thinking mode
    (default for agent runs): the inline ``<think>...</think>`` block, or
    the field a reasoning parser put it in (``reasoning``, else
    ``reasoning_content``). It is informational only (the harness logs it as
    a ``thinking`` event) and is never fed back into the conversation: a
    :class:`Message` has no field for it, and the model's
    ``preserve_thinking`` flag handles that internally.
    """

    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    thinking: str | None = None
    finish_reason: FinishReason | None = None
    usage: dict[str, int] | None = None
