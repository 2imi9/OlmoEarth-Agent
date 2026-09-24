# Serving the agent's LLM

The OlmoEarth Agent talks to an OpenAI-compatible LLM server. The local
default, for a laptop, is **llama.cpp serving the 4-bit GGUF**
`unsloth/Qwen3.6-35B-A3B-GGUF:UD-IQ4_XS`; most of this guide covers it. A
second setup, tested on 24 September 2026, serves a larger model with vLLM on
a GPU cluster: see [Serving on a GPU cluster with vLLM](#serving-on-a-gpu-cluster-with-vllm).
Canonical values live in [`docs/CANON.md`](CANON.md) (C1 model, C3 server,
C4 quant, C5 env vars, C11 cluster option).

> **Prefer a hosted model?** You don't need to serve anything or download 17.7 GB - see [Cloud API (skip the local model)](#cloud-api-skip-the-local-model) below (CANON C10). The rest of this guide covers the local default.

> Why GGUF and not NVFP4 on the laptop? The Qwen3.6 NVFP4 weights (~20 GB)
> don't leave KV-cache headroom on a 24 GB card (verified: it stalls at
> memory profiling). The 4-bit GGUF (~17.7 GB) fits with room to spare and was
> verified end-to-end (CANON C4/C7). On a data-centre GPU, NVFP4 under vLLM
> works: see the cluster section.

## Cloud API (skip the local model)

The agent's reasoning backbone can be a hosted **Claude**, **ChatGPT**, or
**Gemini** model instead of the local Qwen3.6. This path needs **no Docker and
no download** - just the web UI bridge:

```bash
make setup      # uv sync --all-extras
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

## Serving on a GPU cluster with vLLM

Tested on 24 September 2026: the agent ran end to end, through the web UI, on
`nvidia/Qwen3.8-27B-NVFP4` served by vLLM on one GPU of a shared cluster, and
reached from a laptop through an SSH tunnel. The laptop default above does not
change; use this when a bigger model or a longer context is worth a cluster
job.

**Model and hardware.** `nvidia/Qwen3.8-27B-NVFP4` (NVFP4 weights of
Qwen3.8-27B). It was served on an RTX PRO 6000 and, in an earlier session, on
a B200 (both Blackwell). Start-up took about 3.5 minutes; a single stream
with reasoning on ran at about 63 tokens/s, so a turn with tool calls can take
tens of seconds.

**vLLM.** On nodes with an NVIDIA driver but no CUDA toolkit, a pip-installed
vLLM failed at start-up (FlashInfer compiles kernels and needs `nvcc`). The
vLLM OpenAI container image, which bundles the toolkit, run with Apptainer
(`apptainer exec --nv <image>.sif vllm serve ...`) worked. Inside the job:

```bash
export VLLM_API_KEY="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
vllm serve nvidia/Qwen3.8-27B-NVFP4 \
  --port 8000 \
  --max-model-len 65536 \
  --kv-cache-dtype fp8_e4m3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3
```

| Flag | Why |
|---|---|
| `--enable-auto-tool-choice --tool-call-parser qwen3_coder` | Qwen3.8's chat template writes tool calls as `<function=...>` blocks (the qwen3_coder format, not `<tool_call>{json}`); the parser returns them as OpenAI `tool_calls` with object arguments, which the agent loop needs. |
| `--reasoning-parser qwen3` | Returns the thinking block in a separate reasoning field, so the answer text the agent parses has no `<think>` block in it. |
| `--kv-cache-dtype fp8_e4m3` | FP8 KV cache: about half the KV memory of 16-bit, so the long context fits beside the weights. |
| `--max-model-len 65536` | Room for loaded `SKILL.md` bodies and many tool results in one conversation. |

**Access token.** vLLM listens on all interfaces by default, so on a shared
cluster anyone who can reach the node can reach the port. With
`VLLM_API_KEY` set, vLLM answers `/v1` requests only with
`Authorization: Bearer <token>`. Generate the token inside the job, keep it
out of the repository, logs and chat, and pass it to the agent as
`LLM_API_KEY`.

**SSH tunnel.** Forward a laptop port to the compute node through the
cluster's login host (placeholders in angle brackets):

```bash
ssh -N -o ServerAliveInterval=30 -o ExitOnForwardFailure=yes \
  -L 8000:<compute-node>:8000 <user>@<login-host>
```

The login host closed the first tunnel mid-session; wrapping the command in a
loop that reconnects (`while true; do ssh ...; sleep 5; done`) fixed it.

**Point the agent at it.**

```bash
export LLM_ENDPOINT=http://localhost:8000/v1
export LLM_MODEL=nvidia/Qwen3.8-27B-NVFP4
export LLM_API_KEY=<the token from VLLM_API_KEY>
curl -s -H "Authorization: Bearer $LLM_API_KEY" "$LLM_ENDPOINT/models"   # lists the model
make bridge        # web UI on the default "local" backend; or: make agent Q="..."
```

Known gaps: the web UI's "local model is not up" hint probes `/v1/models`
without the key, so with a key set it can show although the model answers.
When the cluster job's time runs out the model disappears mid-conversation:
plan a session inside the job's window, and release the GPU when done.

## References

- GGUF model card: <https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF>
- Base model: <https://huggingface.co/Qwen/Qwen3.6-35B-A3B>
- llama.cpp server: <https://github.com/ggml-org/llama.cpp>
- NVFP4 model card: <https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4>
- vLLM tool calling and reasoning parsers: <https://docs.vllm.ai/>
- Canonical facts: [`docs/CANON.md`](CANON.md)
