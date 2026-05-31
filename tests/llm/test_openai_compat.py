# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""The ``openai_compat`` flag on OlmoEarthLLM (used for ChatGPT/Gemini).

Hosted OpenAI-compatible providers reject the Qwen/vLLM extras
(``top_k`` via ``extra_body``, ``chat_template_kwargs``); the flag drops
them. These build the payload only - no network.
"""

from __future__ import annotations

from typing import Any

from olmoearth_agent.llm import OlmoEarthLLM, ServingConfig
from olmoearth_agent.llm.presets import DEFAULT_AGENT_MODE
from olmoearth_agent.llm.types import Message


def _payload(*, openai_compat: bool) -> dict[str, Any]:
    llm = OlmoEarthLLM(
        ServingConfig(model="m", api_key="k", endpoint="http://x/v1"),
        openai_compat=openai_compat,
    )
    return llm._build_payload(
        messages=[Message(role="user", content="hi")],
        tools=None,
        mode=DEFAULT_AGENT_MODE,
        max_tokens=None,
        preserve_thinking=True,
    )


def test_openai_compat_drops_vendor_extras() -> None:
    payload = _payload(openai_compat=True)
    assert "extra_body" not in payload  # no top_k / chat_template_kwargs
    assert payload["model"] == "m"
    assert payload["messages"] and "max_tokens" in payload


def test_default_keeps_vendor_extras() -> None:
    payload = _payload(openai_compat=False)
    assert "extra_body" in payload
    assert payload["extra_body"]["chat_template_kwargs"] == {"preserve_thinking": True}
