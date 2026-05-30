# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Configuration for connecting to a running OpenAI-compatible LLM server.

The agent reads ``LLM_ENDPOINT``, ``LLM_MODEL``, and ``LLM_API_KEY`` from
the environment (see ``.env.example``). Defaults assume the local
llama.cpp server from ``docs/serving.md`` serving the 4-bit GGUF. See
``docs/CANON.md`` (C1, C3, C5) for the canonical values.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

#: Default served model id: the 4-bit GGUF (see docs/CANON.md C1).
DEFAULT_MODEL = "unsloth/Qwen3.6-35B-A3B-GGUF:UD-IQ4_XS"

#: Default OpenAI-compatible endpoint for the local llama.cpp server.
DEFAULT_ENDPOINT = "http://localhost:8000/v1"

#: Long-default request timeout. Agent runs with tool calls and 32K output
#: budgets can routinely take minutes; raise per-call if you serve at >1M
#: context via YaRN.
DEFAULT_TIMEOUT_SECONDS = 600.0

#: Default output budget per the model card's "Adequate Output Length"
#: guidance (32,768 tokens for most queries).
DEFAULT_MAX_OUTPUT_TOKENS = 32768


@dataclass
class ServingConfig:
    """Static configuration for one vLLM endpoint.

    Build via :meth:`from_env` in production; pass an instance directly
    in tests so they don't depend on process environment.
    """

    endpoint: str = DEFAULT_ENDPOINT
    model: str = DEFAULT_MODEL
    api_key: str = "EMPTY"
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS

    @classmethod
    def from_env(cls) -> ServingConfig:
        """Construct a :class:`ServingConfig` from environment variables.

        Returns
        -------
        ServingConfig
            ``endpoint`` reads ``LLM_ENDPOINT``, ``model`` reads
            ``LLM_MODEL``, ``api_key`` reads ``LLM_API_KEY`` (defaults to
            ``"EMPTY"`` since the local server needs no auth; the OpenAI
            SDK rejects empty strings).
        """
        return cls(
            endpoint=os.environ.get("LLM_ENDPOINT", DEFAULT_ENDPOINT),
            model=os.environ.get("LLM_MODEL", DEFAULT_MODEL),
            api_key=os.environ.get("LLM_API_KEY", "EMPTY"),
        )
