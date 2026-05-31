# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Async OpenAI-compatible client for the vLLM-served Qwen3.6 backbone.

The client is intentionally small: build a payload, dispatch via the
``openai`` SDK, parse the completion. Everything agent-specific
(provenance manifest writes, retry policy, multi-turn loop management)
lives in higher layers: see ``PLAN.md`` §6 P3 for the roadmap.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from typing import Any, Protocol

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion

from olmoearth_agent.llm.config import ServingConfig
from olmoearth_agent.llm.presets import (
    DEFAULT_AGENT_MODE,
    PRESETS,
    SamplingMode,
)
from olmoearth_agent.llm.types import (
    ChatResponse,
    FinishReason,
    Message,
    ToolCall,
    ToolSpec,
)

# OpenAI Chat Completions exposes a fixed set of top-level sampling
# parameters; everything else (top_k, chat_template_kwargs, custom vLLM
# extras) has to ride in ``extra_body`` which vLLM forwards to the
# generation backend.
_OPENAI_TOP_LEVEL_SAMPLING = frozenset(
    {
        "temperature",
        "top_p",
        "presence_penalty",
        "frequency_penalty",
    }
)

# Match a leading ``<think>...</think>`` block; the model emits this when
# thinking mode is enabled (the default for agent runs).
_THINK_RE = re.compile(r"<think>(.*?)</think>\s*", re.DOTALL)

# Fallback parsers for when the server emits a tool call as *text* in the
# message content instead of via the structured ``tool_calls`` field. The
# llama.cpp server does this whenever it is launched without a matching
# ``--tool-call-parser`` (see docs/serving.md), and even with one it can
# fall back intermittently. We recover the call so the agent loop does not
# silently treat an intended tool call as a final answer.
#
#   Hermes-XML form:  <function=NAME><parameter=KEY>VALUE</parameter>...</function>
#   Hermes-JSON form: <tool_call>{"name": "...", "arguments": {...}}</tool_call>
_TEXT_FUNC_RE = re.compile(r"<function=([\w.\-]+)\s*>(.*?)</function>", re.DOTALL)
_TEXT_PARAM_RE = re.compile(r"<parameter=([\w.\-]+)\s*>(.*?)</parameter>", re.DOTALL)
_TEXT_TOOLCALL_JSON_RE = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL
)
_TOOLCALL_TAG_RE = re.compile(r"</?tool_call>")


class Tracer(Protocol):
    """Plug point for the provenance middleware (PR #7).

    Both hooks are synchronous to keep the trace path free of awaits;
    they must not block. PR #7 will register a Tracer that appends one
    ``ProvenanceManifest`` entry per request/response pair.
    """

    def on_request(self, payload: dict[str, Any]) -> None:
        """Called immediately before the request is sent."""

    def on_response(self, payload: dict[str, Any]) -> None:
        """Called after a successful response is parsed."""


class _NullTracer:
    """No-op tracer used when the caller does not supply one."""

    def on_request(self, payload: dict[str, Any]) -> None:  # noqa: D401
        """Discard the request payload."""

    def on_response(self, payload: dict[str, Any]) -> None:  # noqa: D401
        """Discard the response payload."""


def _extract_thinking(content: str | None) -> tuple[str | None, str | None]:
    """Split a leading ``<think>...</think>`` block off ``content``.

    Parameters
    ----------
    content
        Raw assistant-message content from the model.

    Returns
    -------
    tuple of (thinking, remainder)
        ``thinking`` is the inner text of the first ``<think>`` block (or
        ``None`` if absent). ``remainder`` is the rest of the content,
        with leading whitespace trimmed; it is ``None`` if the entire
        message was thinking with no follow-up.
    """
    if content is None:
        return None, None
    match = _THINK_RE.match(content)
    if match is None:
        return None, content
    thinking = match.group(1).strip()
    remainder = content[match.end() :].lstrip()
    return thinking, remainder or None


def _coerce_arg(raw: str) -> Any:
    """Parse a text tool-call argument as JSON, falling back to the string."""
    text = raw.strip()
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return text


def _extract_text_tool_calls(content: str) -> tuple[list[ToolCall], str | None]:
    """Recover tool calls the server emitted as text in ``content``.

    Returns the recovered calls and the leftover prose (``None`` if the
    whole message was tool-call markup). Returns ``([], content)`` when no
    tool-call markup is present, so the caller can no-op safely.
    """
    calls: list[ToolCall] = []
    func_matches = list(_TEXT_FUNC_RE.finditer(content))
    if func_matches:
        for i, match in enumerate(func_matches):
            arguments = {
                key: _coerce_arg(value)
                for key, value in _TEXT_PARAM_RE.findall(match.group(2))
            }
            calls.append(
                ToolCall(id=f"call_{i}", name=match.group(1), arguments=arguments)
            )
        cleaned = _TEXT_FUNC_RE.sub("", content)
    else:
        for i, match in enumerate(_TEXT_TOOLCALL_JSON_RE.finditer(content)):
            try:
                obj = json.loads(match.group(1))
            except json.JSONDecodeError:
                continue
            name = obj.get("name")
            if name:
                calls.append(
                    ToolCall(
                        id=f"call_{i}",
                        name=name,
                        arguments=obj.get("arguments") or {},
                    )
                )
        cleaned = _TEXT_TOOLCALL_JSON_RE.sub("", content) if calls else content
    if not calls:
        return [], content
    cleaned = _TOOLCALL_TAG_RE.sub("", cleaned).strip()
    return calls, (cleaned or None)


def _message_to_openai(message: Message) -> dict[str, Any]:
    """Convert an internal :class:`Message` to the OpenAI wire format."""
    out: dict[str, Any] = {"role": message.role}
    if message.content is not None:
        out["content"] = message.content
    if message.tool_calls:
        out["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": json.dumps(call.arguments),
                },
            }
            for call in message.tool_calls
        ]
    if message.tool_call_id is not None:
        out["tool_call_id"] = message.tool_call_id
    if message.name is not None:
        out["name"] = message.name
    return out


def _tool_to_openai(tool: ToolSpec) -> dict[str, Any]:
    """Convert a :class:`ToolSpec` to the OpenAI function-tool format."""
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        },
    }


class OlmoEarthLLM:
    """Async client for the OlmoEarth Agent's vLLM-served LLM backbone.

    Pinned to ``unsloth/Qwen3.6-35B-A3B-GGUF`` defaults: ``thinking_general``
    sampling, ``preserve_thinking=True`` for multi-turn runs, and the
    32K-token output budget recommended by the model card.

    Examples
    --------
    >>> import asyncio
    >>> from olmoearth_agent.llm import OlmoEarthLLM, Message
    >>> async def main():
    ...     client = OlmoEarthLLM()  # reads LLM_ENDPOINT / LLM_MODEL from env
    ...     response = await client.chat([Message(role="user", content="hi")])
    ...     print(response.content)
    >>> asyncio.run(main())  # doctest: +SKIP
    """

    def __init__(
        self,
        config: ServingConfig | None = None,
        *,
        tracer: Tracer | None = None,
        openai_compat: bool = False,
    ) -> None:
        self.config = config or ServingConfig.from_env()
        self._tracer: Tracer = tracer or _NullTracer()
        # When True, send only standard OpenAI params: drop the Qwen/vLLM
        # extras (top_k via extra_body, chat_template_kwargs.preserve_thinking)
        # that hosted providers like OpenAI and Gemini reject. Used by the
        # bridge for the ChatGPT/Gemini backends.
        self._openai_compat = openai_compat
        self._client = AsyncOpenAI(
            base_url=self.config.endpoint,
            api_key=self.config.api_key,
            timeout=self.config.timeout_seconds,
        )

    async def aclose(self) -> None:
        """Close the underlying OpenAI client and release its connection pool."""
        await self._client.close()

    async def chat(
        self,
        messages: Iterable[Message],
        *,
        tools: Iterable[ToolSpec] | None = None,
        mode: SamplingMode = DEFAULT_AGENT_MODE,
        max_tokens: int | None = None,
        preserve_thinking: bool = True,
    ) -> ChatResponse:
        """Send a chat completion request and return a parsed response.

        Parameters
        ----------
        messages
            Conversation history. Order matters; the last message
            should usually be ``role="user"`` or ``role="tool"``.
        tools
            Function-tool declarations available to the model. Omit
            for plain text completions.
        mode
            Sampling preset; see :mod:`olmoearth_agent.llm.presets`.
            Default is ``thinking_general``.
        max_tokens
            Override the server's max output budget. Defaults to
            :attr:`ServingConfig.max_output_tokens`.
        preserve_thinking
            When True (default for agent runs), forwards
            ``chat_template_kwargs.preserve_thinking=True`` so the
            model retains thinking context across turns.

        Returns
        -------
        ChatResponse
            Content (with the ``<think>`` block split off into
            :attr:`ChatResponse.thinking`), parsed tool calls, finish
            reason, and usage counters.
        """
        payload = self._build_payload(
            messages=messages,
            tools=tools,
            mode=mode,
            max_tokens=max_tokens,
            preserve_thinking=preserve_thinking,
        )
        self._tracer.on_request(payload)
        completion = await self._client.chat.completions.create(**payload)
        response = self._parse_completion(completion)
        self._tracer.on_response({"completion_id": completion.id, "response": response})
        return response

    def _build_payload(
        self,
        *,
        messages: Iterable[Message],
        tools: Iterable[ToolSpec] | None,
        mode: SamplingMode,
        max_tokens: int | None,
        preserve_thinking: bool,
    ) -> dict[str, Any]:
        """Assemble the OpenAI Chat Completions payload."""
        sampling = PRESETS[mode]
        extra_body: dict[str, Any] = {}
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": [_message_to_openai(m) for m in messages],
            "max_tokens": max_tokens or self.config.max_output_tokens,
        }
        for key, value in sampling.items():
            if key in _OPENAI_TOP_LEVEL_SAMPLING:
                payload[key] = value
            elif not self._openai_compat:
                extra_body[key] = value
        if preserve_thinking and not self._openai_compat:
            extra_body["chat_template_kwargs"] = {"preserve_thinking": True}
        if extra_body:
            payload["extra_body"] = extra_body
        if tools is not None:
            tool_list = [_tool_to_openai(t) for t in tools]
            if tool_list:
                payload["tools"] = tool_list
        return payload

    @staticmethod
    def _parse_completion(completion: ChatCompletion) -> ChatResponse:
        """Turn an OpenAI ``ChatCompletion`` into a :class:`ChatResponse`."""
        choice = completion.choices[0]
        message = choice.message
        # When the server is started with `--reasoning-parser qwen3`, vLLM
        # splits the <think> block into a `reasoning_content` field and
        # leaves `content` clean. Otherwise the block is inline and we
        # extract it ourselves. Handle both so the client works regardless
        # of how the server was launched.
        server_reasoning = getattr(message, "reasoning_content", None)
        if server_reasoning:
            thinking: str | None = server_reasoning
            content: str | None = message.content
        else:
            thinking, content = _extract_thinking(message.content)
        tool_calls: list[ToolCall] = []
        if message.tool_calls:
            for raw in message.tool_calls:
                if raw.type != "function":
                    # Future-proofing: ignore non-function tool kinds we
                    # don't model yet (e.g. computer-use, code interpreter).
                    continue
                arguments_text = raw.function.arguments or "{}"
                try:
                    arguments = json.loads(arguments_text)
                except json.JSONDecodeError:
                    # Model returned malformed JSON. Surface the raw
                    # string so the caller can decide what to do; do
                    # not silently drop the tool call.
                    arguments = {"__raw_arguments": arguments_text}
                tool_calls.append(
                    ToolCall(
                        id=raw.id,
                        name=raw.function.name,
                        arguments=arguments,
                    )
                )
        usage: dict[str, int] | None = None
        if completion.usage is not None:
            usage = {
                "prompt_tokens": completion.usage.prompt_tokens,
                "completion_tokens": completion.usage.completion_tokens,
                "total_tokens": completion.usage.total_tokens,
            }
        finish_reason: FinishReason | None = (
            choice.finish_reason  # type: ignore[assignment]
            if choice.finish_reason
            in {"stop", "tool_calls", "length", "content_filter"}
            else None
        )
        # Recover a tool call the server emitted as text rather than via the
        # structured field (see _extract_text_tool_calls). Only when the
        # structured channel is empty, so a well-behaved server is untouched.
        if not tool_calls and content:
            recovered, cleaned = _extract_text_tool_calls(content)
            if recovered:
                tool_calls = recovered
                content = cleaned
                finish_reason = "tool_calls"
        return ChatResponse(
            content=content,
            tool_calls=tool_calls,
            thinking=thinking,
            finish_reason=finish_reason,
            usage=usage,
        )
