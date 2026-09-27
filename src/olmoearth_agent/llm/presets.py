# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Sampling presets from the Qwen3.6-35B-A3B model card, and the harness's own.

The first four modes correspond to the "Best Practices" section of
https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF. The agent's default
is ``thinking_general`` paired with ``preserve_thinking=True`` so
reasoning context carries across multi-turn function-call runs.
``instruct_verify`` is the harness's claim check (:data:`CLAIM_CHECK_MODE`).

A preset's keys other than the OpenAI sampling fields ride in the request's
``extra_body`` (``llm/client.py``): ``top_k``, and ``chat_template_kwargs``,
which vLLM and llama.cpp pass to the chat template.
"""

from __future__ import annotations

from typing import Any, Literal

SamplingMode = Literal[
    "thinking_general",
    "thinking_coding",
    "instruct_general",
    "instruct_reason",
    "instruct_verify",
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
    "instruct_verify": {
        "temperature": 0.1,
        "top_p": 0.8,
        "top_k": 20,
        "chat_template_kwargs": {"enable_thinking": False},
    },
}

DEFAULT_AGENT_MODE: SamplingMode = "thinking_general"

#: The sampling of the harness's rewrite after the answer checks
#: (``LeadAgent.run_stream``): no presence penalty, which pushes a model to
#: new words when the task is to keep most of a draft and change a few
#: sentences, and a lower temperature. The tool-calling loop keeps
#: :data:`DEFAULT_AGENT_MODE`.
REVISION_MODE: SamplingMode = "thinking_coding"

#: The sampling of the harness's claim check (``harness/claim_check.py``): the
#: same answer read against the same record should be flagged alike, so the
#: temperature is near 0; there is no presence penalty, since the reply copies
#: the answer's sentences exactly; and no thinking (Qwen's template switch
#: ``enable_thinking``), since the reply is a short JSON list and the check's
#: cost is bounded per run. A server whose template ignores the switch still
#: splits the thinking off the reply (``llm/client.py``).
CLAIM_CHECK_MODE: SamplingMode = "instruct_verify"
#: The claim check's output budget: a JSON list of the flagged sentences, each
#: with a reason of at most 25 words, fits well within it. A reply cut at the
#: budget keeps its complete items (``harness/claim_check.py``).
CLAIM_CHECK_MAX_TOKENS = 4096
