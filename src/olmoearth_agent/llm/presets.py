# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Sampling presets from the Qwen3.6-35B-A3B model card.

The four modes correspond to the "Best Practices" section of
https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF. The agent's default
is ``thinking_general`` paired with ``preserve_thinking=True`` so
reasoning context carries across multi-turn function-call runs.
"""

from __future__ import annotations

from typing import Any, Literal

SamplingMode = Literal[
    "thinking_general",
    "thinking_coding",
    "instruct_general",
    "instruct_reason",
]

PRESETS: dict[SamplingMode, dict[str, Any]] = {
    "thinking_general": {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "presence_penalty": 1.5,
    },
    "thinking_coding": {
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
    },
    "instruct_general": {
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "presence_penalty": 1.5,
    },
    "instruct_reason": {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "presence_penalty": 1.5,
    },
}

DEFAULT_AGENT_MODE: SamplingMode = "thinking_general"
