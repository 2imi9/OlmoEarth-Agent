# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 OlmoEarth Agent contributors
"""LLM serving client for ``unsloth/Qwen3.6-35B-A3B-NVFP4`` via vLLM.

See ``docs/serving.md`` for the ``vllm serve`` command and the four
sampling presets from the model card. The public surface is
:class:`OlmoEarthLLM` and the small dataclass types in
:mod:`olmoearth_agent.llm.types`.
"""

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
