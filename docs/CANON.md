# CANON — single source of truth for cross-document facts

Some facts (the model name, the serving stack, env-var names) are
restated across many files: `README.md`, `PLAN.md`, `SKILLS.md`,
`AGENTS.md`, `docs/serving.md`, `.env.example`, the docker compose, and
code defaults. When the same fact lives in many places it **drifts** —
one file says one thing, another says something stale.

**This file is the canonical value for every such fact.** When a fact
changes, update it *here first*, then run the alignment protocol below to
fix every other reference. Treat any document that contradicts this file
as the bug.

_Last aligned: 2026-05-28._

## Canonical facts

| # | Fact | Canonical value |
|---|---|---|
| C1 | **LLM model (served)** | `unsloth/Qwen3.6-35B-A3B-GGUF`, quant tag `UD-IQ4_XS` (served id `unsloth/Qwen3.6-35B-A3B-GGUF:UD-IQ4_XS`) |
| C2 | Base model family | `Qwen/Qwen3.6-35B-A3B` — 35B total / 3B active, hybrid Gated-DeltaNet + MoE |
| C3 | **Serving stack** | llama.cpp `ghcr.io/ggml-org/llama.cpp:server-cuda`, OpenAI-compatible API, `--jinja` enables tool calling |
| C4 | **Quantization** | 4-bit GGUF (`UD-IQ4_XS`, ~17.7 GB). **NVFP4 is NOT used.** |
| C5 | LLM env vars | `LLM_ENDPOINT` (default `http://localhost:8000/v1`), `LLM_MODEL`, `LLM_API_KEY` |
| C6 | Studio API | base `https://olmoearth.allenai.org/api/v1`; Bearer `OLMOEARTH_API_KEY`; response envelope `{records, meta, errors}` |
| C7 | Local GPU constraint | RTX 5090 Laptop, 24 GB (Blackwell). This is why NVFP4 (~20 GB weights, no KV headroom) was dropped for GGUF. |
| C8 | Sampling default | `thinking_general` preset + `chat_template_kwargs.preserve_thinking=True` for multi-turn agent runs (Qwen3.6 model card) |
| C9 | Skill catalog | 16 skills in `SKILLS.md`; vendored #1–#4 via submodule `vendor/olmoearth-skills` |

## Banned as "the current approach"

These must NOT appear as what we use today (historical mentions in
`CHANGELOG.md` are fine — that file is a dated record, not a claim about
the present):

- **NVFP4** / `unsloth/Qwen3.6-35B-A3B-NVFP4` — dropped (C4/C7).
- **"served via vLLM"** as the path — serving is llama.cpp (C3). vLLM
  may return as a *datacenter scaling* option later, but it is not the
  documented stack.
- **TensorRT-LLM** — not used.
- **`VLLM_ENDPOINT` / `VLLM_MODEL` / `VLLM_API_KEY`** — renamed to
  `LLM_*` (C5).

## Alignment protocol

Run this whenever a canonical fact changes (or to audit drift):

```bash
# 1. Update the value in the table above.
# 2. Find stale references to the OLD value (skip the vendor submodule):
grep -rin "NVFP4" --include="*.md" --include="*.py" --include="*.yml" . \
  | grep -v vendor/ | grep -v CHANGELOG.md
# 3. Fix every hit to match this file.
# 4. Re-grep to confirm zero stale references (outside CHANGELOG history).
# 5. Run the tests to confirm code defaults still work:
uv run pytest -q
```

## Documents governed by this canon

`README.md` · `PLAN.md` · `SKILLS.md` · `AGENTS.md` · `CONTRIBUTING.md`
· `docs/serving.md` · `.env.example` · `docker/llama.compose.yml` ·
`src/olmoearth_agent/llm/{config,__init__,client,presets}.py`

`CHANGELOG.md` is **history** — never rewritten to match canon; new
entries simply follow it.
