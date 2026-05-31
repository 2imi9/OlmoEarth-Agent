# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Smoke tests for the LLM client.

Unit tests run against a mock OpenAI-compatible endpoint (fast, no GPU
required). The integration test at the bottom hits a live LLM server
and is skipped unless ``LLM_ENDPOINT`` is set in the environment.
"""

from __future__ import annotations

import json
import os
from typing import Any, Callable

import pytest

from olmoearth_agent.llm import (
    Message,
    OlmoEarthLLM,
    ServingConfig,
    ToolSpec,
)


@pytest.mark.asyncio
async def test_simple_chat_round_trip(
    serving_config: ServingConfig,
    mock_chat_response: Callable[..., None],
) -> None:
    """A plain user message returns parsed content."""
    mock_chat_response(content="hello back")
    client = OlmoEarthLLM(serving_config)
    response = await client.chat([Message(role="user", content="hello")])
    assert response.content == "hello back"
    assert response.tool_calls == []
    assert response.finish_reason == "stop"
    assert response.usage == {
        "prompt_tokens": 1,
        "completion_tokens": 1,
        "total_tokens": 2,
    }


@pytest.mark.asyncio
async def test_thinking_trace_extracted(
    serving_config: ServingConfig,
    mock_chat_response: Callable[..., None],
) -> None:
    """Leading ``<think>...</think>`` is split into ``ChatResponse.thinking``."""
    mock_chat_response(
        content="42",
        thinking="The user asked for the answer.",
    )
    client = OlmoEarthLLM(serving_config)
    response = await client.chat([Message(role="user", content="what?")])
    assert response.thinking == "The user asked for the answer."
    assert response.content == "42"


@pytest.mark.asyncio
async def test_server_side_reasoning_content(
    serving_config: ServingConfig,
    mock_chat_response: Callable[..., None],
) -> None:
    """When the server splits reasoning (``--reasoning-parser qwen3``).

    The client reads ``message.reasoning_content`` directly instead of
    regex-extracting an inline ``<think>`` block.
    """
    mock_chat_response(
        content="the answer is 42",
        reasoning_content="server-side reasoning trace",
    )
    client = OlmoEarthLLM(serving_config)
    response = await client.chat([Message(role="user", content="q")])
    assert response.thinking == "server-side reasoning trace"
    assert response.content == "the answer is 42"


@pytest.mark.asyncio
async def test_tool_call_round_trip(
    serving_config: ServingConfig,
    mock_chat_response: Callable[..., None],
) -> None:
    """A function-tool response is parsed into ``ToolCall`` instances."""
    mock_chat_response(
        content=None,
        tool_calls=[
            {
                "id": "call_0",
                "type": "function",
                "function": {
                    "name": "get_current_time",
                    "arguments": json.dumps({"timezone": "UTC"}),
                },
            }
        ],
    )
    tools = [
        ToolSpec(
            name="get_current_time",
            description="Return the current time in the given timezone.",
            parameters={
                "type": "object",
                "properties": {"timezone": {"type": "string"}},
                "required": ["timezone"],
            },
        )
    ]
    client = OlmoEarthLLM(serving_config)
    response = await client.chat(
        [Message(role="user", content="what time is it?")],
        tools=tools,
    )
    assert response.finish_reason == "tool_calls"
    assert len(response.tool_calls) == 1
    call = response.tool_calls[0]
    assert call.id == "call_0"
    assert call.name == "get_current_time"
    assert call.arguments == {"timezone": "UTC"}


@pytest.mark.asyncio
async def test_text_tool_call_hermes_xml_recovered(
    serving_config: ServingConfig,
    mock_chat_response: Callable[..., None],
) -> None:
    """A tool call emitted as ``<function=..><parameter=..>`` text is recovered.

    The llama.cpp server falls back to this when launched without a
    matching ``--tool-call-parser`` (docs/serving.md). The client must
    recover the call so the agent loop does not mistake it for an answer.
    """
    content = (
        "<tool_call> <function=olmoearth_similarity_search>\n"
        "<parameter=query>\n[0.9, 0.1, 0.0]\n</parameter>\n"
        "<parameter=k>\n3\n</parameter>\n"
        '<parameter=ids>\n["site_A", "site_B"]\n</parameter>\n'
        "</function>\n</tool_call>"
    )
    mock_chat_response(content=content, finish_reason="stop")
    client = OlmoEarthLLM(serving_config)
    response = await client.chat([Message(role="user", content="find similar")])
    assert response.finish_reason == "tool_calls"
    assert len(response.tool_calls) == 1
    call = response.tool_calls[0]
    assert call.name == "olmoearth_similarity_search"
    assert call.arguments == {
        "query": [0.9, 0.1, 0.0],
        "k": 3,
        "ids": ["site_A", "site_B"],
    }
    assert response.content is None


@pytest.mark.asyncio
async def test_text_tool_call_hermes_json_recovered(
    serving_config: ServingConfig,
    mock_chat_response: Callable[..., None],
) -> None:
    """A ``<tool_call>{json}</tool_call>`` text emission is recovered too."""
    content = (
        '<tool_call>{"name": "get_current_time", '
        '"arguments": {"timezone": "UTC"}}</tool_call>'
    )
    mock_chat_response(content=content, finish_reason="stop")
    client = OlmoEarthLLM(serving_config)
    response = await client.chat([Message(role="user", content="time?")])
    assert response.finish_reason == "tool_calls"
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].name == "get_current_time"
    assert response.tool_calls[0].arguments == {"timezone": "UTC"}


@pytest.mark.asyncio
async def test_plain_content_not_parsed_as_tool_call(
    serving_config: ServingConfig,
    mock_chat_response: Callable[..., None],
) -> None:
    """Ordinary content with no tool-call markup is left untouched."""
    mock_chat_response(content="The trend is increasing.", finish_reason="stop")
    client = OlmoEarthLLM(serving_config)
    response = await client.chat([Message(role="user", content="trend?")])
    assert response.tool_calls == []
    assert response.content == "The trend is increasing."
    assert response.finish_reason == "stop"


# Note: the OpenAI SDK merges ``extra_body`` into the top level of the
# serialized HTTP request (that is its documented purpose), so the
# captured request body has ``top_k`` and ``chat_template_kwargs`` at the
# top level, NOT nested under an ``extra_body`` key. vLLM reads them from
# there. The assertions below check the on-the-wire shape accordingly.


@pytest.mark.asyncio
async def test_request_carries_preserve_thinking(
    serving_config: ServingConfig,
    mock_chat_response: Callable[..., None],
    last_request_body: Callable[[], dict[str, Any]],
) -> None:
    """Default agent mode forwards ``preserve_thinking=True`` on the wire."""
    mock_chat_response(content="ok")
    client = OlmoEarthLLM(serving_config)
    await client.chat([Message(role="user", content="ping")])
    body = last_request_body()
    assert body["chat_template_kwargs"]["preserve_thinking"] is True


@pytest.mark.asyncio
async def test_request_carries_sampling_params(
    serving_config: ServingConfig,
    mock_chat_response: Callable[..., None],
    last_request_body: Callable[[], dict[str, Any]],
) -> None:
    """The thinking_general preset lands on the wire correctly.

    ``temperature`` / ``top_p`` / ``presence_penalty`` are native OpenAI
    fields; ``top_k`` is vLLM-specific and rides through ``extra_body``,
    which the SDK flattens to the top level.
    """
    mock_chat_response(content="ok")
    client = OlmoEarthLLM(serving_config)
    await client.chat([Message(role="user", content="ping")])
    body = last_request_body()
    assert body["temperature"] == 1.0
    assert body["top_p"] == 0.95
    assert body["presence_penalty"] == 1.5
    assert body["top_k"] == 20


@pytest.mark.asyncio
async def test_preserve_thinking_can_be_disabled(
    serving_config: ServingConfig,
    mock_chat_response: Callable[..., None],
    last_request_body: Callable[[], dict[str, Any]],
) -> None:
    """``preserve_thinking=False`` omits the chat_template_kwargs flag."""
    mock_chat_response(content="ok")
    client = OlmoEarthLLM(serving_config)
    await client.chat(
        [Message(role="user", content="ping")],
        preserve_thinking=False,
    )
    body = last_request_body()
    assert "chat_template_kwargs" not in body


@pytest.mark.asyncio
async def test_tracer_receives_request_and_response(
    serving_config: ServingConfig,
    mock_chat_response: Callable[..., None],
) -> None:
    """The Tracer hook fires once per call on both request and response."""
    mock_chat_response(content="ok")

    requests_seen: list[dict[str, Any]] = []
    responses_seen: list[dict[str, Any]] = []

    class _Recorder:
        def on_request(self, payload: dict[str, Any]) -> None:
            requests_seen.append(payload)

        def on_response(self, payload: dict[str, Any]) -> None:
            responses_seen.append(payload)

    client = OlmoEarthLLM(serving_config, tracer=_Recorder())
    await client.chat([Message(role="user", content="ping")])
    assert len(requests_seen) == 1
    assert len(responses_seen) == 1
    assert requests_seen[0]["model"] == serving_config.model
    assert responses_seen[0]["completion_id"] == "chatcmpl-mock"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_live_function_call_round_trip() -> None:
    """Run against a real LLM server.

    Set ``LLM_ENDPOINT`` to a running OpenAI-compatible server (the
    llama.cpp server from docs/serving.md) to exercise the actual
    function-calling path on the Qwen3.6 backbone. Skipped otherwise so
    unit-only runs stay fast.

    IMPORTANT: the server must be started with tool-calling enabled
    (``--enable-auto-tool-choice --tool-call-parser ...``) or it will
    return the tool call as plain text and this test will fail at the
    ``finish_reason == "tool_calls"`` assertion. See ``docs/serving.md``
    "Function calling" for the full command.
    """
    if not os.environ.get("LLM_ENDPOINT"):
        pytest.skip("Set LLM_ENDPOINT to run live tests")

    client = OlmoEarthLLM()
    tools = [
        ToolSpec(
            name="get_current_time",
            description="Return the current UTC time as ISO-8601.",
            parameters={
                "type": "object",
                "properties": {},
                "required": [],
            },
        )
    ]
    response = await client.chat(
        [
            Message(
                role="user",
                content=(
                    "What time is it right now? Call the get_current_time "
                    "tool: do not guess."
                ),
            )
        ],
        tools=tools,
    )
    assert response.finish_reason == "tool_calls", (
        f"Expected the model to call the tool but got "
        f"finish_reason={response.finish_reason!r}, "
        f"content={response.content!r}"
    )
    assert any(
        call.name == "get_current_time" for call in response.tool_calls
    ), f"No get_current_time call in {response.tool_calls!r}"
