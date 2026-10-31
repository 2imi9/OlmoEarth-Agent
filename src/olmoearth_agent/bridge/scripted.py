# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""The agent's model, played back from a run's recorded responses.

:class:`ScriptedLLM` stands where ``OlmoEarthLLM`` stands in ``LeadAgent``:
the same ``chat()`` seam (``llm/types.py``). Its k-th call returns the k-th
response the model gave (``responses.jsonl``), with the tool-call ids
``call_<k>_<j>``, so a replay feeds the loop exactly what the live run fed
it. The first call beyond the recorded responses is the model's next turn:
it keeps the request (:class:`ModelCall`), signals the session and blocks;
the session shows the request and cancels the loop.

Each recorded response keeps the fingerprint of the request it answered
(:func:`request_fingerprint`: the system prompt, the messages, the offered
tools' specs and the call's settings). A replayed request with another
fingerprint means the loop is no longer the one the model answered, and the
replay stops ("the run diverged at model call k").
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from olmoearth_agent.bridge.errors import BridgeError, BridgeHalt
from olmoearth_agent.llm.presets import (
    CLAIM_CHECK_MODE,
    DEFAULT_AGENT_MODE,
    SamplingMode,
)
from olmoearth_agent.llm.types import ChatResponse, Message, ToolCall, ToolSpec

__all__ = [
    "ModelCall",
    "RecordedResponse",
    "ScriptedLLM",
    "request_fingerprint",
    "tool_call_id",
]

#: The keys a model response may have.
RESPONSE_KEYS = ("content", "tool_calls", "thinking")
#: The keys of one tool call in a response.
CALL_KEYS = ("name", "arguments")


def tool_call_id(call: int, index: int) -> str:
    """The id of the ``index``-th tool call (from 1) of model call ``call``.

    Examples
    --------
    >>> tool_call_id(3, 1)
    'call_3_1'
    """
    return f"call_{call}_{index}"


def request_fingerprint(
    system: str | None,
    messages: Sequence[Message],
    tools: Sequence[ToolSpec] | None,
    mode: str,
    max_tokens: int | None,
    preserve_thinking: bool,
) -> str:
    """The sha256 of one model request: everything the model is given.

    The system prompt, every message (with tool calls and ids), the offered
    tools' whole specs (``None`` when none is offered) and the call's
    settings, as canonical JSON.
    """
    payload = {
        "system": system,
        "messages": [asdict(m) for m in messages],
        "tools": None if tools is None else [asdict(t) for t in tools],
        "mode": mode,
        "max_tokens": max_tokens,
        "preserve_thinking": preserve_thinking,
    }
    blob = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _arguments(raw: Any, where: str) -> dict[str, Any]:
    """A tool call's arguments: a JSON object, or a string holding one."""
    if raw is None:
        return {}
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise BridgeError(f"{where}: arguments are not JSON ({exc})") from exc
    if not isinstance(raw, dict):
        raise BridgeError(f"{where}: arguments must be a JSON object")
    return raw


@dataclass(frozen=True)
class RecordedResponse:
    """One response of the model, as ``responses.jsonl`` keeps it.

    ``tool_calls`` are ``(name, arguments)`` pairs in the model's order;
    ``request_sha256`` is the fingerprint of the request it answered.
    """

    content: str | None = None
    tool_calls: tuple[tuple[str, dict[str, Any]], ...] = ()
    thinking: str | None = None
    request_sha256: str | None = None

    @classmethod
    def parse(cls, data: Any, *, request_sha256: str | None = None) -> RecordedResponse:
        """Read a response the model wrote; raise :class:`BridgeError` if malformed.

        It is a JSON object with ``tool_calls`` (a list of ``{"name",
        "arguments"}``), or ``content`` (the answer's text), or both, and an
        optional ``thinking``.
        """
        if not isinstance(data, dict):
            raise BridgeError(
                'a response is a JSON object: {"tool_calls": [...]} or '
                '{"content": "..."}'
            )
        unknown = sorted(set(data) - set(RESPONSE_KEYS))
        if unknown:
            raise BridgeError(
                f"unknown response keys {unknown}; the keys are {list(RESPONSE_KEYS)}"
            )
        content = data.get("content")
        if content is not None and not isinstance(content, str):
            raise BridgeError("content must be a string")
        thinking = data.get("thinking")
        if thinking is not None and not isinstance(thinking, str):
            raise BridgeError("thinking must be a string")
        raw_calls = data.get("tool_calls") or []
        if not isinstance(raw_calls, list):
            raise BridgeError("tool_calls must be a list")
        calls: list[tuple[str, dict[str, Any]]] = []
        for j, raw in enumerate(raw_calls, 1):
            where = f"tool call {j}"
            if not isinstance(raw, dict):
                raise BridgeError(f'{where} must be {{"name": ..., "arguments": {{}}}}')
            extra = sorted(set(raw) - set(CALL_KEYS))
            if extra:
                raise BridgeError(
                    f"{where}: unknown keys {extra}; a call has {list(CALL_KEYS)} "
                    "(the bridge gives the ids)"
                )
            name = raw.get("name")
            if not isinstance(name, str) or not name.strip():
                raise BridgeError(f"{where}: name must be a tool's name")
            calls.append((name, _arguments(raw.get("arguments"), where)))
        if not calls and content is None:
            raise BridgeError('a response needs "tool_calls" or "content"')
        return cls(
            content=content,
            tool_calls=tuple(calls),
            thinking=thinking,
            request_sha256=request_sha256,
        )

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> RecordedResponse:
        """Read one line of ``responses.jsonl`` (a response and its fingerprint)."""
        body = {k: v for k, v in record.items() if k != "request_sha256"}
        sha = record.get("request_sha256")
        return cls.parse(body, request_sha256=sha if isinstance(sha, str) else None)

    def to_record(self) -> dict[str, Any]:
        """The line ``responses.jsonl`` keeps: the response and its fingerprint."""
        out: dict[str, Any] = {}
        if self.tool_calls:
            out["tool_calls"] = [
                {"name": name, "arguments": arguments}
                for name, arguments in self.tool_calls
            ]
        if self.content is not None:
            out["content"] = self.content
        if self.thinking is not None:
            out["thinking"] = self.thinking
        if self.request_sha256 is not None:
            out["request_sha256"] = self.request_sha256
        return out

    def chat_response(self, call: int) -> ChatResponse:
        """The response as the loop receives it on model call ``call`` (from 1)."""
        calls = [
            ToolCall(id=tool_call_id(call, j), name=name, arguments=dict(arguments))
            for j, (name, arguments) in enumerate(self.tool_calls, 1)
        ]
        return ChatResponse(
            content=self.content,
            tool_calls=calls,
            thinking=self.thinking,
            finish_reason="tool_calls" if calls else "stop",
        )


@dataclass(frozen=True)
class ModelCall:
    """One request the loop made of the model, as the bridge received it."""

    #: The call's place in the run, from 1.
    index: int
    #: The system message's text (``None`` if the request had none).
    system: str | None
    #: The conversation without the system message, in order.
    messages: tuple[Message, ...]
    #: The offered tools' specs, or ``None`` when none is offered.
    tools: tuple[ToolSpec, ...] | None
    mode: str
    max_tokens: int | None
    preserve_thinking: bool
    #: :func:`request_fingerprint` of the request.
    sha256: str

    @classmethod
    def of(
        cls,
        index: int,
        messages: Iterable[Message],
        tools: Iterable[ToolSpec] | None,
        mode: str,
        max_tokens: int | None,
        preserve_thinking: bool,
    ) -> ModelCall:
        """The call from ``chat``'s arguments (the system message comes first)."""
        listed = list(messages)
        system: str | None = None
        if listed and listed[0].role == "system":
            system = listed[0].content
            listed = listed[1:]
        specs = None if tools is None else tuple(tools)
        return cls(
            index=index,
            system=system,
            messages=tuple(listed),
            tools=specs,
            mode=mode,
            max_tokens=max_tokens,
            preserve_thinking=preserve_thinking,
            sha256=request_fingerprint(
                system, listed, specs, mode, max_tokens, preserve_thinking
            ),
        )

    @property
    def tool_names(self) -> list[str] | None:
        """The offered tools' names, or ``None`` when none is offered."""
        return None if self.tools is None else [t.name for t in self.tools]


class ScriptedLLM:
    """``LeadAgent``'s model, played back; the next call waits for the model.

    Parameters
    ----------
    responses
        The model's responses so far, one per model call, in order.

    Attributes
    ----------
    calls
        Every request received in this replay, in order.
    pending
        The first request beyond ``responses``: the call the model has to
        answer next; ``None`` until the loop makes it.
    waiting
        Set when ``pending`` is.
    """

    def __init__(self, responses: Sequence[RecordedResponse]) -> None:
        self.responses = list(responses)
        self.calls: list[ModelCall] = []
        self.pending: ModelCall | None = None
        self.waiting = asyncio.Event()

    async def chat(
        self,
        messages: Iterable[Message],
        *,
        tools: Iterable[ToolSpec] | None = None,
        mode: SamplingMode = DEFAULT_AGENT_MODE,
        max_tokens: int | None = None,
        preserve_thinking: bool = True,
    ) -> ChatResponse:
        """The recorded response to this call, or wait for the model's.

        The claim check's call (``CLAIM_CHECK_MODE``) is refused: the bridge
        runs with ``check_claims=False``, and a claim check that failed open
        on a refusal would pass the answer unread.
        """
        if mode == CLAIM_CHECK_MODE:
            raise BridgeHalt(
                "the claim check called the model; the bridge runs without it "
                "(check_claims=False) and has no model call to give it"
            )
        if self.pending is not None:
            raise BridgeHalt("a second model call came while one was pending")
        call = ModelCall.of(
            len(self.calls) + 1, messages, tools, mode, max_tokens, preserve_thinking
        )
        self.calls.append(call)
        if call.index <= len(self.responses):
            recorded = self.responses[call.index - 1]
            if (
                recorded.request_sha256 is not None
                and recorded.request_sha256 != call.sha256
            ):
                raise BridgeHalt(
                    f"the run diverged at model call {call.index}: the loop's "
                    "request is not the one the recorded response answered"
                )
            return recorded.chat_response(call.index)
        self.pending = call
        self.waiting.set()
        # Never resolved: the session cancels the loop once it has the call.
        loop = asyncio.get_running_loop()
        blocked: asyncio.Future[ChatResponse] = loop.create_future()
        return await blocked
