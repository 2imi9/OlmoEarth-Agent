# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Live end-to-end test: real LLM + real Studio API through the harness.

Skipped unless both ``LLM_ENDPOINT`` (a running OpenAI-compatible LLM
server, e.g. the llama.cpp 4-bit GGUF per docs/serving.md) and
``OLMOEARTH_API_KEY`` are set. This is the slice that proves the whole
spine: natural-language brief -> LLM tool call -> live Studio API ->
answer.
"""

from __future__ import annotations

import os

import pytest

_HAVE_LLM = bool(os.environ.get("LLM_ENDPOINT"))
_HAVE_KEY = bool(os.environ.get("OLMOEARTH_API_KEY"))


@pytest.mark.integration
@pytest.mark.asyncio
async def test_live_agent_counts_projects() -> None:
    if not (_HAVE_LLM and _HAVE_KEY):
        pytest.skip("needs LLM_ENDPOINT + OLMOEARTH_API_KEY")

    from olmoearth_agent.harness import LeadAgent
    from olmoearth_agent.llm import OlmoEarthLLM
    from olmoearth_agent.skills import build_default_registry
    from olmoearth_agent.studio import StudioClient

    async with StudioClient.from_env() as studio:
        agent = LeadAgent(OlmoEarthLLM(), build_default_registry(), studio)
        result = await agent.run(
            "How many OlmoEarth Studio projects do I have? "
            "Use the tools to find out, then state the number.",
            max_turns=6,
        )

    # The model should have ended with a text answer...
    assert result.final_content is not None, (
        f"agent hit max turns without answering; calls={result.tool_calls}"
    )
    # ...and at least one Studio tool call must have succeeded.
    assert any(ok for _name, ok in result.tool_calls), (
        f"no successful tool call; trace={result.tool_calls}"
    )
