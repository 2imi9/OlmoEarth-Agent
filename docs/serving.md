# Serving Qwen3.6-35B-A3B (4-bit GGUF) with llama.cpp

The OlmoEarth Agent talks to an OpenAI-compatible LLM server. The
supported stack is **llama.cpp serving the 4-bit GGUF**
`unsloth/Qwen3.6-35B-A3B-GGUF:UD-IQ4_XS`. Canonical values live in
[`docs/CANON.md`](CANON.md) (C1 model, C3 server, C4 quant, C5 env vars).

> Why GGUF and not NVFP4? The NVFP4 weights (~20 GB) don't leave KV-cache
> headroom on a 24 GB card (verified: it stalls at memory profiling). The
> 4-bit GGUF (~17.7 GB) fits with room to spare and was verified
> end-to-end. NVFP4 + a larger-context server may return as a datacenter
> option later, but it is not the path today (CANON C4/C7).

## Quick start (Docker)

```bash
docker compose -f docker/llama.compose.yml up
# or, equivalently, a one-off container:
docker run -d --name oe-llama --gpus all -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  ghcr.io/ggml-org/llama.cpp:server-cuda \
  -hf unsloth/Qwen3.6-35B-A3B-GGUF:UD-IQ4_XS \
  --host 0.0.0.0 --port 8000 \
  --jinja -ngl 999 -c 8192 --no-mmap
```

The server listens on `http://localhost:8000/v1` (OpenAI Chat
Completions). Point the agent at it:

```bash
export LLM_ENDPOINT=http://localhost:8000/v1
export LLM_MODEL=unsloth/Qwen3.6-35B-A3B-GGUF:UD-IQ4_XS
# LLM_API_KEY is optional; defaults to "EMPTY" which the server accepts.
```

Then `uv run pytest -m integration` exercises the live function-calling
path; without `LLM_ENDPOINT` set, those tests skip.

## Flags that matter

| Flag | Why |
|---|---|
| `--jinja` | **Required for tool calling** — activates the GGUF chat template so the model emits parseable `tool_calls`. Without it, tool calls come back as text and the integration tests fail at `finish_reason == "tool_calls"`. |
| `--no-mmap` | Loads the file fully instead of mmap'ing — avoids the slow mmap-over-virtiofs path that stalls large loads on Docker Desktop / WSL. |
| `-ngl 999` | Offload all layers to the GPU. |
| `-c 8192` | **Total** context, which llama.cpp splits across parallel slots (it defaults to ~4 → ~2048 tokens each). Fine for short chats, but a full `SKILL.md` loaded mid-conversation overflows a 2048-token slot. For skill-heavy or long-context runs, use one big slot: `--parallel 1 -c 16384`. |

## Hardware

- **NVIDIA GPU**, Blackwell recommended (RTX 50-series / B-series). The
  4-bit GGUF (~17.7 GB) fits a 24 GB card (verified on an RTX 5090
  Laptop) with ~5 GB free for KV cache.
- llama.cpp `server-cuda` image (`ghcr.io/ggml-org/llama.cpp:server-cuda`).
- Other 4-bit GGUF sizes (see the
  [GGUF model card](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF)):
  `UD-IQ4_XS` 17.7 GB (safe on 24 GB), `UD-Q4_K_S` 20.9 GB (tight),
  `UD-Q4_K_M` 22.1 GB (too tight on 24 GB with KV cache).

## Agent-mode defaults

The agent client (`src/olmoearth_agent/llm/client.py`) opens every chat
with:

- **`chat_template_kwargs.preserve_thinking=True`** — keeps thinking
  context across turns (Qwen3.6 model card: improves multi-turn decision
  consistency, optimizes KV cache). Passed via the OpenAI SDK's
  `extra_body`, which the server reads from the top level of the request.
- **`thinking_general` sampling preset** — `T=1.0, top_p=0.95, top_k=20,
  presence_penalty=1.5`.

Switch presets per call with `OlmoEarthLLM.chat(..., mode="instruct_general")`.
The four presets in `src/olmoearth_agent/llm/presets.py` mirror the
"Best Practices" table on the model card.

## References

- GGUF model card: <https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF>
- Base model: <https://huggingface.co/Qwen/Qwen3.6-35B-A3B>
- llama.cpp server: <https://github.com/ggml-org/llama.cpp>
- Canonical facts: [`docs/CANON.md`](CANON.md)
