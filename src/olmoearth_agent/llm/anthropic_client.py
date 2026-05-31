# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Native Anthropic (Claude) backend for the agent's LLM seam.

A drop-in alternative to :class:`~olmoearth_agent.llm.client.OlmoEarthLLM`:
the same ``.chat()`` surface and :class:`ChatResponse` return, but it talks
to the Anthropic Messages API through the ``anthropic`` SDK instead of an
OpenAI-compatible endpoint. Use it to run briefs on Claude (bring your own
Anthropic API key) while keeping the local model the default.

``anthropic`` is an optional dependency (the ``claude`` extra). It is
imported lazily inside :class:`AnthropicLLM`, so this module - and its pure
message/tool converters, which is where the real logic lives - import fine
without it installed (and stay unit-testable offline).
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from olmoearth_agent.llm.config import ServingConfig
from olmoearth_agent.llm.presets import DEFAULT_AGENT_MODE, PRESETS, SamplingMode
from olmoearth_agent.llm.types import (
    ChatResponse,
    FinishReason,
    Message,
    ToolCall,
    ToolSpec,
)

#: Anthropic's public API base; exposed so a caller can point at a proxy.
DEFAULT_ANTHROPIC_BASE_URL = "https://api.anthropic.com"

#: Default Claude model when the caller doesn't pin one (balanced for agent
#: tool loops). Override per request from the web UI.
DEFAULT_CLAUDE_MODEL = "claude-sonnet-4-6"

#: Anthropic Messages requires an explicit output budget; Claude's ceiling is
#: far above the local model's, but a moderate default keeps runs cheap.
DEFAULT_MAX_OUTPUT_TOKENS = 8192

#: Sampling keys the Messages API accepts (the OpenAI client also sends
#: ``presence_penalty``/``frequency_penalty``, which Anthropic does not take).
_ANTHROPIC_SAMPLING = frozenset({"temperature", "top_p", "top_k"})

#: Anthropic ``stop_reason`` -> our :data:`FinishReason`.
_STOP_REASON_MAP: dict[str, FinishReason] = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "tool_use": "tool_calls",
    "max_tokens": "length",
}


def _split_system(messages: list[Message]) -> tuple[str | None, list[Message]]:
    """Pull ``system`` turns out into one Anthropic top-level ``system`` string.

    Anthropic takes the system prompt as a dedicated parameter rather than a
    message role, so we concatenate any system messages and return the rest.
    """
    system_parts = [m.content for m in messages if m.role == "system" and m.content]
    rest = [m for m in messages if m.role != "system"]
    return ("\n\n".join(system_parts) or None), rest


def _assistant_blocks(message: Message) -> list[dict[str, Any]]:
    """Assistant turn -> Anthropic content blocks (text + ``tool_use``)."""
    blocks: list[dict[str, Any]] = []
    if message.content:
        blocks.append({"type": "text", "text": message.content})
    blocks.extend(
        {
            "type": "tool_use",
            "id": call.id,
            "name": call.name,
            "input": call.arguments or {},
        }
        for call in message.tool_calls or []
    )
    # Anthropic rejects an empty content array; keep at least one block.
    return blocks or [{"type": "text", "text": ""}]


def _messages_to_anthropic(messages: list[Message]) -> list[dict[str, Any]]:
    """Convert internal messages to Anthropic user/assistant turns.

    Consecutive ``role="tool"`` results are merged into a single following
    ``user`` turn: Anthropic wants every ``tool_result`` for an assistant's
    ``tool_use`` batch grouped in one user message, in order.
    """
    out: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []

    def flush() -> None:
        if pending:
            out.append({"role": "user", "content": list(pending)})
            pending.clear()

    for m in messages:
        if m.role == "tool":
            pending.append(
                {
                    "type": "tool_result",
                    "tool_use_id": m.tool_call_id or "",
                    "content": m.content or "",
                }
            )
            continue
        flush()
        if m.role == "assistant":
            out.append({"role": "assistant", "content": _assistant_blocks(m)})
        else:  # user (system already split out upstream)
            out.append({"role": "user", "content": m.content or ""})
    flush()
    return out


def _tools_to_anthropic(tools: Iterable[ToolSpec]) -> list[dict[str, Any]]:
    """:class:`ToolSpec` -> Anthropic tool declaration (``input_schema``)."""
    return [
        {"name": t.name, "description": t.description, "input_schema": t.parameters}
        for t in tools
    ]


def _sampling_for_anthropic(mode: SamplingMode) -> dict[str, Any]:
    """Subset of a sampling preset that Anthropic's Messages API accepts."""
    return {k: v for k, v in PRESETS[mode].items() if k in _ANTHROPIC_SAMPLING}


def _parse_anthropic_payload(
    content_blocks: list[dict[str, Any]],
    stop_reason: str | None,
    usage: dict[str, Any] | None,
) -> ChatResponse:
    """Turn a ``Message.model_dump()`` payload into a :class:`ChatResponse`.

    Operates on plain dicts (not SDK objects) so it is unit-testable without
    the ``anthropic`` package installed.
    """
    text_parts: list[str] = []
    thinking_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    for i, block in enumerate(content_blocks):
        btype = block.get("type")
        if btype == "text":
            text_parts.append(block.get("text") or "")
        elif btype == "thinking":
            thinking_parts.append(block.get("thinking") or "")
        elif btype == "tool_use":
            tool_calls.append(
                ToolCall(
                    id=block.get("id") or f"call_{i}",
                    name=block.get("name") or "",
                    arguments=block.get("input") or {},
                )
            )
    usage_out: dict[str, int] | None = None
    if usage:
        prompt = int(usage.get("input_tokens") or 0)
        completion = int(usage.get("output_tokens") or 0)
        usage_out = {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
        }
    return ChatResponse(
        content="".join(text_parts) or None,
        tool_calls=tool_calls,
        thinking="\n".join(p for p in thinking_parts if p) or None,
        finish_reason=_STOP_REASON_MAP.get(stop_reason or ""),
        usage=usage_out,
    )


class AnthropicLLM:
    """Async Claude backend with the :class:`OlmoEarthLLM` ``.chat()`` surface.

    Pass an Anthropic ``api_key`` (pay-per-token) or an ``auth_token``
    (OAuth bearer). The Qwen-specific knobs (``preserve_thinking``,
    ``top_k`` via ``extra_body``, ``<think>`` parsing) don't apply here;
    ``preserve_thinking`` is accepted for interface parity but ignored, and
    extended thinking is left off (Claude reasons inline for tool loops).
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_CLAUDE_MODEL,
        api_key: str | None = None,
        auth_token: str | None = None,
        base_url: str | None = None,
        timeout_seconds: float = 600.0,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    ) -> None:
        try:
            from anthropic import AsyncAnthropic
        except ModuleNotFoundError as exc:  # pragma: no cover - import guard
            raise RuntimeError(
                "The 'anthropic' package is required for the Claude backend. "
                "Install it with: uv sync --extra claude"
            ) from exc
        if not api_key and not auth_token:
            raise RuntimeError("AnthropicLLM needs an api_key or an auth_token.")
        # A ServingConfig keeps health/introspection uniform with OlmoEarthLLM
        # (serve.py reads ``llm.config.model`` / ``llm.config.endpoint``).
        self.config = ServingConfig(
            endpoint=base_url or DEFAULT_ANTHROPIC_BASE_URL,
            model=model,
            api_key=api_key or "OAUTH",
            timeout_seconds=timeout_seconds,
            max_output_tokens=max_output_tokens,
        )
        client_kwargs: dict[str, Any] = {"timeout": timeout_seconds}
        if base_url:
            client_kwargs["base_url"] = base_url
        if api_key:
            client_kwargs["api_key"] = api_key
        if auth_token:
            client_kwargs["auth_token"] = auth_token
        self._client = AsyncAnthropic(**client_kwargs)

    async def chat(
        self,
        messages: Iterable[Message],
        *,
        tools: Iterable[ToolSpec] | None = None,
        mode: SamplingMode = DEFAULT_AGENT_MODE,
        max_tokens: int | None = None,
        preserve_thinking: bool = True,  # noqa: ARG002 - interface parity
    ) -> ChatResponse:
        """Send a Messages request and return a parsed :class:`ChatResponse`."""
        system, rest = _split_system(list(messages))
        payload: dict[str, Any] = {
            "model": self.config.model,
            "max_tokens": max_tokens or self.config.max_output_tokens,
            "messages": _messages_to_anthropic(rest),
            **_sampling_for_anthropic(mode),
        }
        if system:
            payload["system"] = system
        if tools is not None:
            tool_list = _tools_to_anthropic(tools)
            if tool_list:
                payload["tools"] = tool_list
        completion = await self._client.messages.create(**payload)
        data = completion.model_dump()
        return _parse_anthropic_payload(
            data.get("content") or [],
            data.get("stop_reason"),
            data.get("usage"),
        )
