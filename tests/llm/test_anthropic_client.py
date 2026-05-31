# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Unit tests for the native Anthropic (Claude) backend.

The pure converters need no network and no real API call; the one
``.chat()`` test stubs the SDK client so it stays offline too.
"""

from __future__ import annotations

from typing import Any

import pytest

from olmoearth_agent.llm.anthropic_client import (
    AnthropicLLM,
    _messages_to_anthropic,
    _parse_anthropic_payload,
    _sampling_for_anthropic,
    _split_system,
    _tools_to_anthropic,
)
from olmoearth_agent.llm.presets import DEFAULT_AGENT_MODE
from olmoearth_agent.llm.types import Message, ToolCall, ToolSpec


def test_split_system_concatenates_and_filters() -> None:
    system, rest = _split_system(
        [
            Message(role="system", content="A"),
            Message(role="user", content="hi"),
            Message(role="system", content="B"),
        ]
    )
    assert system == "A\n\nB"
    assert [m.role for m in rest] == ["user"]


def test_split_system_none_when_absent() -> None:
    system, rest = _split_system([Message(role="user", content="hi")])
    assert system is None
    assert len(rest) == 1


def test_messages_to_anthropic_tool_use_and_merged_results() -> None:
    out = _messages_to_anthropic(
        [
            Message(role="user", content="run it"),
            Message(
                role="assistant",
                content="ok",
                tool_calls=[
                    ToolCall(id="t1", name="search", arguments={"q": "x"}),
                    ToolCall(id="t2", name="fetch", arguments={}),
                ],
            ),
            Message(role="tool", tool_call_id="t1", content="r1"),
            Message(role="tool", tool_call_id="t2", content="r2"),
            Message(role="user", content="thanks"),
        ]
    )
    assert [m["role"] for m in out] == ["user", "assistant", "user", "user"]
    asst = out[1]["content"]
    assert asst[0] == {"type": "text", "text": "ok"}
    assert asst[1]["type"] == "tool_use" and asst[1]["id"] == "t1"
    assert asst[2]["name"] == "fetch"
    results = out[2]["content"]  # the two tool results merged into one user turn
    assert [b["type"] for b in results] == ["tool_result", "tool_result"]
    assert results[0]["tool_use_id"] == "t1" and results[0]["content"] == "r1"
    assert results[1]["tool_use_id"] == "t2"


def test_assistant_with_no_content_still_valid() -> None:
    out = _messages_to_anthropic([Message(role="assistant")])
    assert out[0]["content"] == [{"type": "text", "text": ""}]


def test_tools_to_anthropic_uses_input_schema() -> None:
    spec = ToolSpec(
        name="f", description="d", parameters={"type": "object", "properties": {}}
    )
    assert _tools_to_anthropic([spec]) == [
        {
            "name": "f",
            "description": "d",
            "input_schema": {"type": "object", "properties": {}},
        }
    ]


def test_sampling_drops_unsupported_keys() -> None:
    s = _sampling_for_anthropic(DEFAULT_AGENT_MODE)
    assert set(s).issubset({"temperature", "top_p", "top_k"})
    assert "presence_penalty" not in s and "frequency_penalty" not in s


def test_parse_payload_text_tool_thinking_usage() -> None:
    resp = _parse_anthropic_payload(
        [
            {"type": "thinking", "thinking": "hmm"},
            {"type": "text", "text": "Hello"},
            {"type": "tool_use", "id": "tc1", "name": "search", "input": {"q": "x"}},
        ],
        "tool_use",
        {"input_tokens": 10, "output_tokens": 3},
    )
    assert resp.content == "Hello"
    assert resp.thinking == "hmm"
    assert resp.finish_reason == "tool_calls"
    assert resp.tool_calls == [ToolCall(id="tc1", name="search", arguments={"q": "x"})]
    assert resp.usage == {
        "prompt_tokens": 10,
        "completion_tokens": 3,
        "total_tokens": 13,
    }


def test_parse_payload_end_turn_maps_to_stop() -> None:
    resp = _parse_anthropic_payload(
        [{"type": "text", "text": "done"}], "end_turn", None
    )
    assert resp.finish_reason == "stop"
    assert resp.tool_calls == []
    assert resp.usage is None


def test_constructor_requires_credentials() -> None:
    with pytest.raises(RuntimeError, match="api_key or an auth_token"):
        AnthropicLLM(model="claude-x")


def test_constructor_sets_config() -> None:
    llm = AnthropicLLM(model="claude-x", api_key="sk-test")
    assert llm.config.model == "claude-x"
    assert llm.config.endpoint  # base url recorded for health/introspection


class _FakeMessages:
    def __init__(self, captured: dict[str, Any]) -> None:
        self._captured = captured

    async def create(self, **payload: Any) -> Any:
        self._captured.update(payload)

        class _Resp:
            @staticmethod
            def model_dump() -> dict[str, Any]:
                return {
                    "content": [{"type": "text", "text": "hi from claude"}],
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 5, "output_tokens": 2},
                }

        return _Resp()


class _FakeClient:
    def __init__(self, captured: dict[str, Any]) -> None:
        self.messages = _FakeMessages(captured)


@pytest.mark.asyncio
async def test_chat_builds_payload_and_parses() -> None:
    llm = AnthropicLLM(model="claude-x", api_key="sk-test")
    captured: dict[str, Any] = {}
    llm._client = _FakeClient(captured)  # type: ignore[assignment]
    resp = await llm.chat(
        [
            Message(role="system", content="sys"),
            Message(role="user", content="hi"),
        ],
        tools=[ToolSpec(name="f", description="d", parameters={"type": "object"})],
    )
    assert captured["model"] == "claude-x"
    assert captured["system"] == "sys"
    assert captured["messages"][0]["role"] == "user"
    assert captured["tools"][0]["name"] == "f"
    assert "max_tokens" in captured
    assert resp.content == "hi from claude"
    assert resp.finish_reason == "stop"
    assert resp.usage == {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}
