# Serving Qwen3.6-35B-A3B-NVFP4 with vLLM

The OlmoEarth Agent talks to a vLLM-served LLM over the OpenAI Chat
Completions protocol. This doc covers the supported `vllm serve`
command, environment variables the agent reads, sampling defaults, and
the YaRN incantation for long-context serving.

## Quick start

```bash
# Install (one-time)
uv pip install vllm --torch-backend=auto

# Serve (the canonical command, also pinned in PLAN.md §4)
vllm serve unsloth/Qwen3.6-35B-A3B-NVFP4 \
  --trust-remote-code \
  --dtype bfloat16 \
  --max-model-len 4096
```

The server listens on `http://localhost:8000/v1` and speaks OpenAI Chat
Completions. Point the agent at it:

```bash
export VLLM_ENDPOINT=http://localhost:8000/v1
export VLLM_MODEL=unsloth/Qwen3.6-35B-A3B-NVFP4
# VLLM_API_KEY is optional; defaults to "EMPTY" which vLLM accepts.
```

`pytest tests/llm/test_smoke.py -m integration` then exercises the live
function-calling path (`get_current_time` toy tool); without
`VLLM_ENDPOINT` set, those tests skip.

## Hardware requirements

- **NVIDIA Blackwell GPU.** B100, B200, GB200/GB300 NVL72, RTX PRO 6000
  Blackwell, RTX 50-series, or DGX Spark (GB10). Hopper (H100 / H200)
  does NOT natively accelerate NVFP4 — it will run but lose the speed
  advantage that motivated the choice.
- **`vllm>=0.19.0`** — the version the model card recommends for this
  NVFP4 checkpoint.
- **VRAM: the NVFP4 weights are ~20 GB.** That fits a B200 (180 GB) or
  any datacenter Blackwell with room to spare. It does **NOT** fit
  comfortably on a 24 GB card: vLLM reserves a memory pool at
  `--gpu-memory-utilization`, loads the ~20 GB weights into it, then
  runs a memory-profiling forward pass to size the KV cache. On 24 GB
  the residual (~2 GB) is too little and the profiling step stalls.
  **Verified 2026-05-28 on an RTX 5090 Laptop (24 GB): NVFP4 + vLLM
  0.19.x hangs at profiling.** For ≤24 GB local dev, use the 4-bit
  GGUF path below instead.

## Local development on ≤24 GB VRAM (4-bit GGUF via llama.cpp)

When NVFP4 won't fit, serve a 4-bit GGUF with llama.cpp. Same
OpenAI-compatible protocol, so the agent's client code is unchanged —
only `VLLM_ENDPOINT` / `VLLM_MODEL` differ.

**Verified working** on an RTX 5090 Laptop (24 GB) on 2026-05-28:
`UD-IQ4_XS` (17.7 GB) loads to ~18.6 GB used / ~5.5 GB free, and the
agent's function-call round-trip passes end-to-end.

```bash
docker run -d --name oe-llama --gpus all -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  ghcr.io/ggml-org/llama.cpp:server-cuda \
  -hf unsloth/Qwen3.6-35B-A3B-GGUF:UD-IQ4_XS \
  --host 0.0.0.0 --port 8000 \
  --jinja -ngl 999 -c 4096 --no-mmap
```

- **`--jinja`** is required for tool calling — it activates the GGUF's
  chat template so the model emits parseable `tool_calls`. (No
  `--tool-call-parser` flag needed; llama.cpp uses the template.)
- **`--no-mmap`** loads the file fully (avoids slow mmap-over-virtiofs
  on Docker Desktop / WSL, which is what stalls large loads).
- **`-ngl 999`** offloads all layers to GPU; **`-c 4096`** context.
- 4-bit GGUF sizes for a 24 GB card (see the
  [GGUF model card](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF)
  hardware table): `UD-IQ4_XS` 17.7 GB (safe), `UD-Q4_K_S` 20.9 GB
  (tight), `UD-Q4_K_M` 22.1 GB (too tight with KV cache on 24 GB).

Point the agent at it:

```bash
export VLLM_ENDPOINT=http://localhost:8000/v1
export VLLM_MODEL=unsloth/Qwen3.6-35B-A3B-GGUF:UD-IQ4_XS
uv run pytest tests/llm -m integration -v   # live function-call test
```

This is a **local-dev accommodation, not a change to the production
stack** — production stays vLLM + NVFP4 on a datacenter Blackwell where
the weights fit with KV headroom.

## Configuration knobs

| Flag | Default | Notes |
|---|---|---|
| `--max-model-len` | 4096 | Raise after checking GPU memory. Qwen3.6's native context is 262144; long-context serving needs YaRN (below). |
| `--trust-remote-code` | required | Qwen3.6's tokenizer ships custom code. |
| `--dtype` | `bfloat16` | Activations + KV cache; weights remain NVFP4. |
| `--host` / `--port` | `0.0.0.0:8000` | Bind address. The agent's default `VLLM_ENDPOINT` assumes `localhost:8000`. |

## Function calling (required for the agent)

The base `vllm serve` command above does **not** parse tool calls — the
model emits them as text and the OpenAI `tool_calls` field stays empty.
The agent's whole job is function calling, so the server must be
launched with tool-call parsing enabled:

```bash
vllm serve unsloth/Qwen3.6-35B-A3B-NVFP4 \
  --trust-remote-code \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --reasoning-parser qwen3
```

Notes:
- `--enable-auto-tool-choice` + `--tool-call-parser` are what turn the
  model's emitted calls into structured OpenAI `tool_calls`. **The live
  integration test (`tests/llm/test_smoke.py::test_live_function_call_round_trip`)
  asserts `finish_reason == "tool_calls"` and will fail without these.**
- **The agent's function-call path is VERIFIED** (2026-05-28) end-to-end
  via the llama.cpp local-dev path above — the `OlmoEarthLLM` client
  correctly produced a structured `create_project(...)` tool call from a
  natural-language brief. The client code is proven against any
  OpenAI-compatible tool-calling server.
- **The vLLM `--tool-call-parser` NAME is still UNVERIFIED for this
  model** — we could not stand up vLLM NVFP4 on the 24 GB test box (see
  Hardware requirements). On a datacenter Blackwell, confirm the parser
  with `vllm serve --help | grep -A3 tool-call-parser`. The Qwen card
  uses `qwen3_coder` for SGLang; vLLM ≥0.19.0 ships `hermes`,
  `qwen3_coder`, `qwen3_xml`. Pick the Qwen3.6 one.
- `--reasoning-parser qwen3` splits the `<think>` block into a separate
  `reasoning_content` field. Optional: the client
  (`OlmoEarthLLM._parse_completion`) reads `reasoning_content` when
  present and otherwise extracts the inline `<think>` block itself, so it
  works with or without this flag.

## Long context (YaRN)

Qwen3.6 supports up to 1.01M tokens with YaRN scaling. The model card's
canonical incantation:

```bash
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 vllm serve unsloth/Qwen3.6-35B-A3B-NVFP4 \
  --trust-remote-code \
  --dtype bfloat16 \
  --max-model-len 1010000 \
  --hf-overrides '{"text_config": {"rope_parameters": {"mrope_interleaved": true, "mrope_section": [11, 11, 10], "rope_type": "yarn", "rope_theta": 10000000, "partial_rotary_factor": 0.25, "factor": 4.0, "original_max_position_embeddings": 262144}}}'
```

Per the model card: enable YaRN only when you need it. Static YaRN
hurts short-context quality — set `factor` to the smallest value that
covers your typical context length.

## Agent-mode defaults

The agent client (`src/olmoearth_agent/llm/client.py`) opens every chat
with:

- **`chat_template_kwargs.preserve_thinking=True`** — keeps thinking
  context across turns. Per the model card: improves multi-turn
  decision consistency, optimizes KV cache utilization. Passed via the
  OpenAI SDK's `extra_body`.
- **`thinking_general` sampling preset** — `T=1.0, top_p=0.95, top_k=20,
  presence_penalty=1.5`.

Switch presets per call with `OlmoEarthLLM.chat(..., mode="instruct_general")`.
The four presets in `src/olmoearth_agent/llm/presets.py` mirror the
"Best Practices" table on the model card.

## Local Docker

For environment isolation (the GPU still needs to be Blackwell):

```bash
docker compose -f docker/vllm.compose.yml up
```

The compose file pins `vllm/vllm-openai:v0.19.0`, mounts the HF cache,
and exposes port 8000. Set `HF_TOKEN` in `.env` first so the container
can pull the model weights.

## References

- Model card: <https://huggingface.co/unsloth/Qwen3.6-35B-A3B-NVFP4>
- Base model: <https://huggingface.co/Qwen/Qwen3.6-35B-A3B>
- vLLM docs: <https://docs.vllm.ai/>
- Ziming's vLLM fork: <https://github.com/2imi9/vllm>
- NVFP4 explainer: <https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/>
