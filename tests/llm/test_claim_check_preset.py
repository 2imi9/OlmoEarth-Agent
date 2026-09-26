# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""The claim check's sampling preset, as it reaches the wire (no network).

The claim check (``harness/claim_check.py``) reads one answer against one
record: near-zero temperature, no presence penalty, no thinking, and a
bounded reply. Its thinking switch is a chat-template argument, which the
agent's ``preserve_thinking`` must not overwrite.
"""

from __future__ import annotations

from typing import Any

from olmoearth_agent.llm import OlmoEarthLLM, ServingConfig
from olmoearth_agent.llm.anthropic_client import _sampling_for_anthropic
from olmoearth_agent.llm.presets import (
    CLAIM_CHECK_MAX_TOKENS,
    CLAIM_CHECK_MODE,
    PRESETS,
)
from olmoearth_agent.llm.types import Message


def _payload(*, preserve_thinking: bool, openai_compat: bool = False) -> dict[str, Any]:
    llm = OlmoEarthLLM(
        ServingConfig(model="m", api_key="k", endpoint="http://x/v1"),
        openai_compat=openai_compat,
    )
    return llm._build_payload(
        messages=[Message(role="user", content="hi")],
        tools=None,
        mode=CLAIM_CHECK_MODE,
        max_tokens=CLAIM_CHECK_MAX_TOKENS,
        preserve_thinking=preserve_thinking,
    )


def test_the_claim_check_preset_is_low_variance_and_does_not_think() -> None:
    preset = PRESETS[CLAIM_CHECK_MODE]
    assert preset["temperature"] <= 0.2 and "presence_penalty" not in preset
    payload = _payload(preserve_thinking=False)
    assert payload["temperature"] == preset["temperature"]
    assert payload["max_tokens"] == CLAIM_CHECK_MAX_TOKENS <= 4096
    assert payload["extra_body"] == {
        "top_k": 20,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def test_preserve_thinking_is_merged_into_the_presets_template_switches() -> None:
    payload = _payload(preserve_thinking=True)
    assert payload["extra_body"]["chat_template_kwargs"] == {
        "enable_thinking": False,
        "preserve_thinking": True,
    }
    assert PRESETS[CLAIM_CHECK_MODE]["chat_template_kwargs"] == {
        "enable_thinking": False
    }  # the preset itself is left as it is


def test_hosted_backends_get_no_template_switch() -> None:
    assert "extra_body" not in _payload(preserve_thinking=True, openai_compat=True)
    assert _sampling_for_anthropic(CLAIM_CHECK_MODE) == {
        "temperature": 0.1,
        "top_p": 0.8,
        "top_k": 20,
    }
