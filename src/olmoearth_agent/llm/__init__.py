# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""LLM client for the 4-bit GGUF ``unsloth/Qwen3.6-35B-A3B-GGUF`` model.

Talks to any OpenAI-compatible server; the documented stack is the
llama.cpp server in ``docs/serving.md`` (see ``docs/CANON.md`` C1/C3).
The public surface is :class:`OlmoEarthLLM` and the small dataclass
types in :mod:`olmoearth_agent.llm.types`.
"""

from olmoearth_agent.llm.anthropic_client import AnthropicLLM
from olmoearth_agent.llm.client import OlmoEarthLLM, Tracer
from olmoearth_agent.llm.config import ServingConfig
from olmoearth_agent.llm.presets import (
    DEFAULT_AGENT_MODE,
    PRESETS,
    SamplingMode,
)
from olmoearth_agent.llm.types import (
    ChatResponse,
    Message,
    ToolCall,
    ToolSpec,
)

__all__ = [
    "AnthropicLLM",
    "ChatResponse",
    "DEFAULT_AGENT_MODE",
    "Message",
    "OlmoEarthLLM",
    "PRESETS",
    "SamplingMode",
    "ServingConfig",
    "ToolCall",
    "ToolSpec",
    "Tracer",
]
