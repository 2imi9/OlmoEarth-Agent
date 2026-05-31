# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Test fixtures for the LLM client.

The mock vLLM endpoint uses ``pytest-httpx`` to intercept OpenAI SDK
calls. It implements just enough of the Chat Completions protocol to
verify request building, response parsing, tool calls, and thinking-
trace extraction without booting a real vLLM server.
"""

from __future__ import annotations

import json
from typing import Any, Callable

import pytest
from pytest_httpx import HTTPXMock

from olmoearth_agent.llm.config import ServingConfig

MOCK_ENDPOINT = "http://mock-llm:8000/v1"
MOCK_MODEL = "unsloth/Qwen3.6-35B-A3B-GGUF:UD-IQ4_XS"


@pytest.fixture
def serving_config() -> ServingConfig:
    """ServingConfig pointing at the mock endpoint."""
    return ServingConfig(endpoint=MOCK_ENDPOINT, model=MOCK_MODEL)


@pytest.fixture
def mock_chat_response(
    httpx_mock: HTTPXMock,
) -> Callable[..., None]:
    """Factory that registers one mocked Chat Completions response.

    The default response is a simple "ok" assistant message. Pass
    ``tool_calls=[...]`` to simulate a function-call response, or
    ``thinking="..."`` to prepend a ``<think>...</think>`` block.
    """

    def _register(
        *,
        content: str | None = "ok",
        tool_calls: list[dict[str, Any]] | None = None,
        thinking: str | None = None,
        reasoning_content: str | None = None,
        finish_reason: str = "stop",
    ) -> None:
        # `reasoning_content` simulates a server started with
        # `--reasoning-parser qwen3` (clean content + separate field);
        # `thinking` simulates an inline `<think>` block in content.
        if reasoning_content is not None:
            full_content: str | None = content
        elif thinking and content:
            full_content = f"<think>{thinking}</think>\n{content}"
        elif thinking:
            full_content = f"<think>{thinking}</think>"
        else:
            full_content = content
        message: dict[str, Any] = {"role": "assistant", "content": full_content}
        if reasoning_content is not None:
            message["reasoning_content"] = reasoning_content
        if tool_calls:
            message["tool_calls"] = tool_calls
            message["content"] = None
            finish_reason = "tool_calls"
        body = {
            "id": "chatcmpl-mock",
            "object": "chat.completion",
            "created": 0,
            "model": MOCK_MODEL,
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "total_tokens": 2,
            },
        }
        httpx_mock.add_response(
            url=f"{MOCK_ENDPOINT}/chat/completions",
            method="POST",
            json=body,
        )

    return _register


@pytest.fixture
def last_request_body(httpx_mock: HTTPXMock) -> Callable[[], dict[str, Any]]:
    """Return the most recent request body as parsed JSON."""

    def _last() -> dict[str, Any]:
        requests = httpx_mock.get_requests()
        if not requests:
            msg = "No HTTP requests captured"
            raise AssertionError(msg)
        return json.loads(requests[-1].content)

    return _last
