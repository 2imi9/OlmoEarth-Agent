# OlmoEarth Agent

A tool that drives the [OlmoEarth Studio](https://allenai.org/blog/olmoearth) platform from natural-language briefs. Same shape as Google's Google Earth Agent: a compact catalog of functions (Studio API, EO data fetch, geometry utilities) plus a sandboxed Python interpreter, with operational constraints built in.

## Status

**v0.4.** Text-only LLM ([unsloth/Qwen3.6-35B-A3B-GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF), 4-bit `UD-IQ4_XS`) served via **llama.cpp** with function calling. Multimodal stack parked. 16-skill catalog in [`SKILLS.md`](SKILLS.md); skills ship one PR at a time. Harness + several skills are implemented and verified live — see [`CHANGELOG.md`](CHANGELOG.md). Canonical facts: [`docs/CANON.md`](docs/CANON.md).

See [`PLAN.md`](PLAN.md) for the tool catalog, harness dataclasses, operational rules, and roadmap; [`SKILLS.md`](SKILLS.md) for the per-skill spec.

## Run it

```bash
uv sync --all-extras
git submodule update --init          # vendored skills (#1–#4)

# 1. Serve the LLM (4-bit GGUF via llama.cpp) — see docs/serving.md:
docker compose -f docker/llama.compose.yml up -d

# 2. Point the agent at the LLM + your Studio key:
export LLM_ENDPOINT=http://localhost:8000/v1
export OLMOEARTH_API_KEY=...          # Studio UI → profile → API Keys

# 3. Run a brief:
uv run olmoearth-agent "How many OlmoEarth Studio projects do I have?"
uv run olmoearth-agent --show-trace "Which of my projects relate to water quality?"
```

`--show-trace` prints the tool-call trace + provenance count to stderr; the answer goes to stdout. Also runnable as `python -m olmoearth_agent "..."`.

## What's in this repo

- [`PLAN.md`](PLAN.md) — Tool catalog, harness dataclasses, operational rules, underlying stack, roadmap.
- [`SKILLS.md`](SKILLS.md) — 16-skill catalog (Prep / Configure / Run / Analyze / Integrate / Report).
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — Contributor workflow (DCO sign-off, branch naming, AI-assistance policy).
- [`AGENTS.md`](AGENTS.md) — Onboarding context for coding agents working on this codebase.
- [`CHANGELOG.md`](CHANGELOG.md) — Keep a Changelog v1.1.0.
- `pyproject.toml`, `.pre-commit-config.yaml`, `.env.example` — Project scaffold.
- `LICENSE` — Apache 2.0.

## Tool surface (summary)

The agent exposes:

- `system:python` — sandboxed Python interpreter with `pandas`, `geopandas`, `xarray`, `rioxarray`, `shapely`, `pystac_client`, `planetary_computer`, `rslearn`, `olmoearth_projects` preloaded. No `import` statements.
- `system:search`, `system:fetch` — web search and documented-endpoint HTTP GET.
- `olmoearth.*` — Studio API wrappers: `load_context`, `resolve_to_aoi`, `search_dataset_spec`, `get_data_in_locations`, `create_project`/`create_area`/`create_dataset`/`create_labelset`/`upload_labels`, `submit_prediction`/`poll_prediction`/`fetch_results`/`save_view`.
- `eo.*` — STAC search, asset signing, AOI windowing.
- `utils.*` — geometry helpers, equal-frequency binning, spatial cross-validation split.

Full catalog with arguments and return types in [`PLAN.md` §1](PLAN.md).

## Studio API

- Docs: https://docs.olmoearth.allenai.org/
- Auth: https://docs.olmoearth.allenai.org/authentication/ — Bearer token; max 10 keys per account
- Live OpenAPI spec: https://olmoearth.allenai.org/api/v1/openapi.json
- Resources: Areas, Projects, Datasets, Labelsets, Labels, Annotations, Tasks, Predictions, PredictionResults, Users

## Underlying stack (reference only)

| Layer | Reference |
|---|---|
| LLM | [unsloth/Qwen3.6-35B-A3B-GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF) (4-bit `UD-IQ4_XS`) via [llama.cpp](https://github.com/ggml-org/llama.cpp). Text + function calling. See [`docs/serving.md`](docs/serving.md). |
| Harness | [ByteDance DeerFlow v2](https://github.com/bytedance/deer-flow) |
| Skills | 16 skills in [`SKILLS.md`](SKILLS.md), packaged per [agentskills.io](https://agentskills.io) ([NVIDIA AI-Q](https://docs.nvidia.com/aiq-blueprint/latest/integration/agent-skills.html) impl reference) |
| Parked | Multimodal stack (Prismatic VLM + adapters + OlmoEarth embedding stream) and train-time self-improvement loops — see `PLAN.md` §7 |

## License

Apache 2.0 — see [`LICENSE`](LICENSE).
