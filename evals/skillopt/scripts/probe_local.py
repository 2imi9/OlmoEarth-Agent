"""Probe the local Qwen server through SkillOpt's two model paths.

Confirms that the target (qwen) and optimizer (OpenAI-compatible) paths both
return non-empty content from the local llama.cpp Qwen3.6 server.
"""

import os

os.environ["SKILLOPT_OPENAI_COMPAT_EXTRA_BODY"] = (
    '{"chat_template_kwargs": {"enable_thinking": false}}'
)

from skillopt.model import (
    chat_optimizer,
    chat_target,
    configure_azure_openai,
    configure_qwen_chat,
    set_optimizer_deployment,
    set_reasoning_effort,
    set_target_deployment,
)
from skillopt.model.backend_config import set_optimizer_backend, set_target_backend

MODEL = "unsloth/Qwen3.6-35B-A3B-GGUF:UD-IQ4_XS"
ENDPOINT = "http://localhost:8000/v1"

set_optimizer_backend("openai_chat")
set_target_backend("qwen_chat")
configure_azure_openai(endpoint=ENDPOINT, auth_mode="openai_compatible")
configure_qwen_chat(
    base_url=ENDPOINT,
    max_tokens=512,
    enable_thinking=False,
    temperature=0.3,
    timeout_seconds=300,
)
set_target_deployment(MODEL)
set_optimizer_deployment(MODEL)
set_reasoning_effort(None)

t, tu = chat_target(
    "You are terse.",
    "Reply with exactly: TARGET_OK",
    max_completion_tokens=64,
    stage="probe",
)
print("TARGET content:", repr((t or "")[:200]))
print("TARGET usage:", tu)

o, ou = chat_optimizer(
    "You are terse.",
    "Reply with exactly: OPT_OK",
    max_completion_tokens=256,
    stage="probe",
)
print("OPT content:", repr((o or "")[:200]))
print("OPT usage:", ou)

ok = bool((t or "").strip()) and bool((o or "").strip())
print("PROBE_RESULT:", "PASS" if ok else "FAIL")
