# CANON: single source of truth for cross-document facts

Some facts (the model name, the serving stack, env-var names) are
restated across many files: `README.md`, `PLAN.md`, `SKILLS.md`,
`AGENTS.md`, `docs/serving.md`, `.env.example`, the docker compose, and
code defaults. When the same fact lives in many places it **drifts**:
one file says one thing, another says something stale.

**This file is the canonical value for every such fact.** When a fact
changes, update it *here first*, then run the alignment protocol below to
fix every other reference. Treat any document that contradicts this file
as the bug.

_Last aligned: 2026-09-24._

## Canonical facts

| # | Fact | Canonical value |
|---|---|---|
| C1 | **LLM model (local default)** | `unsloth/Qwen3.6-35B-A3B-GGUF`, quant tag `UD-IQ4_XS` (served id `unsloth/Qwen3.6-35B-A3B-GGUF:UD-IQ4_XS`); `DEFAULT_MODEL` in `llm/config.py` |
| C2 | Base model family | `Qwen/Qwen3.6-35B-A3B`: 35B total / 3B active, hybrid Gated-DeltaNet + MoE |
| C3 | **Serving stack (local default)** | llama.cpp `ghcr.io/ggml-org/llama.cpp:server-cuda`, OpenAI-compatible API, `--jinja` enables tool calling. The local default for a laptop stays llama.cpp. |
| C4 | **Quantization (local default)** | 4-bit GGUF (`UD-IQ4_XS`, ~17.7 GB). NVFP4 is not used on the laptop (C7); it is the cluster option's format (C11). |
| C5 | LLM env vars | `LLM_ENDPOINT` (default `http://localhost:8000/v1`), `LLM_MODEL`, `LLM_API_KEY` |
| C6 | Studio API | base `https://olmoearth.allenai.org/api/v1`; Bearer `OLMOEARTH_API_KEY`; response envelope `{records, meta, errors}` |
| C7 | Local GPU constraint | RTX 5090 Laptop, 24 GB (Blackwell). This is why the Qwen3.6 NVFP4 weights (~20 GB, no KV headroom) were dropped for GGUF on the laptop. |
| C8 | Sampling default | `thinking_general` preset + `chat_template_kwargs.preserve_thinking=True` for multi-turn agent runs (Qwen3.6 model card) |
| C9 | Skill catalog | 18 skills in `SKILLS.md` (Prep / Configure / Run / Analyze / Integrate / Report). 17 implemented in-repo; skill #5 `olmoearth-change-detection` also has an out-of-process JEPA engine backed by `2imi9/olmoearth-jepa-change`. The instruction packages for #1-#3 and #17 (data-prep, studio-job-config, embeddings, rslearn) ship in `src/olmoearth_agent/skills/packages/`. Earlier catalog versions split data-prep into #1/#2, embeddings into #4/#17, and change-detection into #6/#19; those were merged. |
| C10 | **Hosted LLM backends (optional)** | Local Qwen3.6 (C1) is the **default**; the bridge also accepts bring-your-own-key **Claude** (native Anthropic SDK), **ChatGPT**, and **Gemini** (OpenAI-compatible), selected per request (`X-LLM-Backend` / `X-LLM-Key` / `X-LLM-Model`; `GET /api/llm/models` autodetects). Keys are forwarded per request, never stored server-side. |
| C11 | **GPU-cluster option (tested 2026-09-24)** | `nvidia/Qwen3.8-27B-NVFP4` served by vLLM with an access token (`VLLM_API_KEY`), reached through an SSH tunnel and set in the agent with C5's variables. Flags and setup: `docs/serving.md`. |

## Banned as "the current approach"

These must NOT appear as what we use today (historical mentions in
`CHANGELOG.md` and `docs/archive/` are fine: they are dated records, not
claims about the present):

- **NVFP4 or vLLM as the local default**: the laptop path is llama.cpp +
  GGUF (C3/C4). Both are fine to name as the GPU-cluster option (C11).
- **TensorRT-LLM**: not used.
- **`VLLM_ENDPOINT` / `VLLM_MODEL` / `VLLM_API_KEY` as the agent's
  settings**: the agent reads `LLM_*` (C5). `VLLM_API_KEY` appears only as
  the vLLM server's own variable (C11).

## Alignment protocol

Run this whenever a canonical fact changes (or to audit drift):

```bash
# 1. Update the value in the table above.
# 2. Find stale references to the OLD value (history is exempt):
grep -rin "<old value>" --include="*.md" --include="*.py" --include="*.yml" . \
  | grep -v CHANGELOG.md | grep -v docs/archive/
# 3. Fix every hit to match this file.
# 4. Re-grep to confirm zero stale references (outside CHANGELOG history).
# 5. Run the tests to confirm code defaults still work:
uv run pytest -q
```

## Documents governed by this canon

`README.md` | `PLAN.md` | `SKILLS.md` | `AGENTS.md` | `CONTRIBUTING.md`
 | `docs/serving.md` | `.env.example` | `docker/llama.compose.yml` |
`src/olmoearth_agent/llm/{config,__init__,client,presets}.py`

`CHANGELOG.md` is **history**: never rewritten to match canon; new
entries simply follow it.
