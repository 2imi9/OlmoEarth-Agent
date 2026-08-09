# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Hosted NVIDIA NIM smoke test for the concise response contract.

This test never uses the local LLM or GPU. It is skipped unless
``NVIDIA_API_KEY`` is set and calls NVIDIA's hosted OpenAI-compatible API
directly. Override ``NVIDIA_NIM_MODEL`` to compare another hosted chat model.
"""

from __future__ import annotations

import os

import pytest

from olmoearth_agent.harness import LeadAgent, ResponsePolicy
from olmoearth_agent.llm import OlmoEarthLLM, ServingConfig
from olmoearth_agent.tools.registry import ToolRegistry

_NIM_ENDPOINT = "https://integrate.api.nvidia.com/v1"
_NIM_MODEL = os.environ.get("NVIDIA_NIM_MODEL", "nvidia/nemotron-3-nano-30b-a3b")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_hosted_nim_honors_concise_final_contract() -> None:
    """Require a short, loss-preserving final answer from hosted NIM."""
    api_key = os.environ.get("NVIDIA_API_KEY", "").strip()
    if not api_key:
        pytest.skip("needs NVIDIA_API_KEY; local GPU fallback is forbidden")

    policy = ResponsePolicy(concise_max_words=120, concise_max_tokens=4096)
    llm = OlmoEarthLLM(
        ServingConfig(
            endpoint=_NIM_ENDPOINT,
            model=_NIM_MODEL,
            api_key=api_key,
            timeout_seconds=180,
        ),
        openai_compat=True,
    )
    agent = LeadAgent(
        llm,
        ToolRegistry(),
        studio=None,  # type: ignore[arg-type]
        response_policy=policy,
    )
    try:
        result = await agent.run(
            "Summarize this completed tool result for the user. Do not perform "
            "another action. Trusted result data: prediction_id=pred-nim-42; "
            "status=completed; artifact=artifacts/pred-nim-42.tif; "
            "valid_pixels=1842; warning=cloud cover is 7 percent.",
            max_turns=2,
        )
    finally:
        await llm.aclose()

    content = result.final_content or ""
    assert content, "hosted NIM returned no final answer"
    assert policy.word_count(content) <= policy.concise_max_words, content
    assert "pred-nim-42" in content, content
    assert "completed" in content.lower(), content
    assert "artifacts/pred-nim-42.tif" in content, content
    assert "1842" in content.replace(",", ""), content
    assert "7" in content and "cloud" in content.lower(), content
