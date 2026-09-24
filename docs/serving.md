# Serving Qwen3.6-35B-A3B (4-bit GGUF) with llama.cpp

The OlmoEarth Agent talks to an OpenAI-compatible LLM server. The
supported stack is **llama.cpp serving the 4-bit GGUF**
`unsloth/Qwen3.6-35B-A3B-GGUF:UD-IQ4_XS`. Canonical values live in
[`docs/CANON.md`](CANON.md) (C1 model, C3 server, C4 quant, C5 env vars).

> **Prefer a hosted model?** You don't need to serve anything or download 17.7 GB - see [Cloud API (skip the local model)](#cloud-api-skip-the-local-model) below (CANON C10). The rest of this guide covers the local default.

> Why GGUF and not NVFP4? The NVFP4 weights (~20 GB) don't leave KV-cache
> headroom on a 24 GB card (verified: it stalls at memory profiling). The
> 4-bit GGUF (~17.7 GB) fits with room to spare and was verified
> end-to-end. NVFP4 + a larger-context server may return as a datacenter
> option later, but it is not the path today (CANON C4/C7).

## Cloud API (skip the local model)

The agent's reasoning backbone can be a hosted **Claude**, **ChatGPT**, or
**Gemini** model instead of the local Qwen3.6. This path needs **no Docker and
no download** - just the web UI bridge:

```bash
make setup      # init vendored skills + uv sync --all-extras
make bridge     # live UI on http://localhost:8088 (no `make serve`)
```

Then, in the UI: paste your Studio key, open **Settings -> LLM backend**, pick a
provider, and paste that provider's API key. The bridge forwards the key to the
provider **per request** (via the `X-LLM-Backend` / `X-LLM-Key` headers) and
never stores or logs it, exactly like the Studio key. `GET /api/llm/models`
autodetects the provider's current model ids for the dropdown.

- **Claude** uses the native Anthropic SDK - install the extra once:
  `uv sync --extra claude`.
- **ChatGPT** and **Gemini** use the OpenAI-compatible client (Gemini via its
  `.../v1beta/openai/` base URL); no extra needed.

`make bridge` starts the UI even when no local model is running, so the cloud
path stands on its own. If you are on the default **local** backend and the
local model is not up, the UI shows a one-line nudge to either start it
(`make serve`) or switch to a cloud provider - it does not fail silently.

## Optional CPU reliability gateway

For shared or failure-prone hosted-model traffic, SGLang Model Gateway can sit
between the existing OpenAI-compatible client and the provider. The optional
profile adds retries, circuit breaking, and model-call telemetry without using a
local model or GPU. It does not replace the agent harness, and its MCP and
conversation-storage features remain disabled. See
[`docs/sglang-gateway.md`](sglang-gateway.md) for the tested boundary and the
digest-pinned compose command.

## Quick start (Docker)

`make up` does the whole local bring-up in one command (setup, then start the
server below and block until healthy, then serve the live UI). `make serve`
runs just the server step. The raw invocations:

```bash
docker compose -f docker/llama.compose.yml up
# or, equivalently, a one-off container:
docker run -d --name oe-llama --gpus all -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  ghcr.io/ggml-org/llama.cpp:server-cuda \
  -hf unsloth/Qwen3.6-35B-A3B-GGUF:UD-IQ4_XS \
  --host 0.0.0.0 --port 8000 \
  --jinja -ngl 999 -c 16384 --parallel 1 --no-mmap
```

> The compose bind-mounts your host `~/.cache/huggingface` (the same cache the
> one-off `docker run` above uses), so the ~17.7 GB GGUF downloads **once** and
> is reused on every later `make serve` -- no re-download. `make serve` (via
> [`scripts/serve-llm.sh`](../scripts/serve-llm.sh)) resolves and exports
> `HF_CACHE_DIR` for you, including the `C:/Users/<you>/.cache/huggingface` form
> Docker Desktop needs on Windows. If you instead run `docker compose ... up` by
> hand from PowerShell/cmd (where `$HOME` is unset), pass it explicitly:
> `HF_CACHE_DIR=C:/Users/<you>/.cache/huggingface docker compose -f docker/llama.compose.yml up`.
> Set `HF_PROXY=http://host.docker.internal:7897` only for a genuine first-run
> download behind a host proxy (`127.0.0.1` won't work from inside the container).
> Upgrading from the old named-volume setup leaves an orphan -- reclaim it with
> `docker volume rm docker_hf_cache`.

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
| `--jinja` | **Required for first-class tool calling**: activates the GGUF chat template so the model emits structured `tool_calls` directly. Without it the call arrives as text and the client falls back to text-recovery (`client.py` `_extract_text_tool_calls`) to still surface it as `finish_reason == "tool_calls"`; the live function-calling test is least flaky with it on. |
| `--no-mmap` | Loads the file fully instead of mmap'ing, avoiding the slow mmap-over-virtiofs path that stalls large loads on Docker Desktop / WSL. |
| `-ngl 999` | Offload all layers to the GPU. |
| `-c 16384 --parallel 1` | **Total** context as one big slot. llama.cpp splits `-c` across parallel slots (default ~4), so the old `-c 8192` default gave ~2048 tokens per slot — a single `SKILL.md` loaded mid-conversation overflowed it, and multi-call tool sessions (e.g. several litsearch results) need the headroom even unsplit. One 16384-token slot is the verified skill-heavy configuration; the agent is single-user, so serial slots cost nothing. |

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

- **`chat_template_kwargs.preserve_thinking=True`**: keeps thinking
  context across turns (Qwen3.6 model card: improves multi-turn decision
  consistency, optimizes KV cache). Passed via the OpenAI SDK's
  `extra_body`, which the server reads from the top level of the request.
- **`thinking_general` sampling preset**: `T=1.0, top_p=0.95, top_k=20,
  presence_penalty=1.5`.

Switch presets per call with `OlmoEarthLLM.chat(..., mode="instruct_general")`.
The four presets in `src/olmoearth_agent/llm/presets.py` mirror the
"Best Practices" table on the model card.

## References

- GGUF model card: <https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF>
- Base model: <https://huggingface.co/Qwen/Qwen3.6-35B-A3B>
- llama.cpp server: <https://github.com/ggml-org/llama.cpp>
- Canonical facts: [`docs/CANON.md`](CANON.md)
