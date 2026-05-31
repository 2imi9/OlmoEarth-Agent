<div align="center">

<img src="webui/assets/OlmoEarth-logo.png" alt="OlmoEarth" width="360">&nbsp;&nbsp;<img src="webui/assets/agent-tag.png" alt="Agent" height="42">

**Drive [OlmoEarth Studio](https://allenai.org/blog/olmoearth) from natural-language briefs - on a local LLM, or your own Claude / ChatGPT / Gemini.**

[![License](https://img.shields.io/badge/License-OlmoEarth%20Artifact-1f6feb.svg)](LICENSE)
[![Skills](https://img.shields.io/badge/skills-15-F0529C.svg)](SKILLS.md)
[![LLM](https://img.shields.io/badge/LLM-local%20Qwen3.6%20%2B%20hosted-0FCB8C.svg)](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB.svg)](pyproject.toml)

</div>

<div align="center">

![OlmoEarth Agent live demo](webui/demo/olmoearth-agent-demo.gif)

</div>

<div align="center"><sub>Connect a Studio key, send a brief, and the agent loop streams its reasoning, tool calls, results, and a plain-English answer — in a multi-turn chat with saved history, a collapsible Studio project tree, and Markdown answers. &nbsp;·&nbsp; <a href="webui/">Open the web UI →</a> &nbsp;·&nbsp; <a href="webui/demo/olmoearth-agent-demo.mp4">walkthrough (MP4)</a></sub></div>

---

OlmoEarth Agent turns a natural-language brief into real geospatial work on [OlmoEarth Studio](https://allenai.org/blog/olmoearth). It reasons about the ask, calls tools over a sandboxed Python interpreter, submits and polls predictions, and reports honest results: a provenance manifest per call, mandatory spatial cross-validation on auto-correlated AOIs, and no raw coordinates in chat. It runs on a local **Qwen3.6-35B-A3B** model served with llama.cpp by default (no hosted LLM required), and can optionally use your own **Claude, ChatGPT, or Gemini** key via the web UI's model-backend picker.

## Quick start

> **Prerequisites:** [Docker](https://docs.docker.com/get-docker/) (to serve the LLM), [uv](https://docs.astral.sh/uv/), and an `OLMOEARTH_API_KEY` (Studio UI → profile → **API Keys**).

```bash
make setup                                              # git submodule init + uv sync --all-extras
make serve                                              # 4-bit Qwen3.6 GGUF via llama.cpp (docker)
make agent Q="How many OlmoEarth Studio projects do I have?"
make web                                                # styled web UI on http://localhost:8080
```

Set your key first (`export OLMOEARTH_API_KEY=...`); `LLM_ENDPOINT` defaults to `http://localhost:8000/v1`. `make help` lists every target; `make down` stops the LLM. No `make` (e.g. Windows + Git Bash)? [`./scripts/quickstart.sh`](scripts/quickstart.sh) runs setup + serve and prints the exact `uv run` commands; see [`docs/serving.md`](docs/serving.md) for the raw `docker compose` invocations.

## What it does

The LLM reads the brief, plans, and emits a tool call; the harness runs it in a sandboxed geospatial Python interpreter (`geopandas`, `xarray`, `shapely`, `pystac_client`, `rslearn`, … preloaded, no `import` statements), feeds the result back, and iterates until it can answer. State persists across turns. The operational rules — trailing-12-month windows, cost guards on fine-tunes, mandatory spatial CV on auto-correlated AOIs, a provenance manifest per call — are enforced by the harness, not the model.

The capability set ships as **15 skills**, grouped by EO-workflow stage.

<details>
<summary><strong>The 15 skills</strong>, by workflow stage (Prep / Configure / Run / Analyze / Integrate / Report)</summary>

| # | Skill | What it does | Stage |
|---|---|---|---|
| 1 | `olmoearth-studio-upload` | Labels (GeoJSON/CSV/Shapefile) → Studio-importable file with MIME / 10K / multi-metric guards | **Prep** |
| 2 | `olmoearth-rslearn-config` | Labels → `rslearn` `dataset.json` + Lightning YAML with a 7-criteria audit | **Prep** |
| 3 | `olmoearth-studio-job-config` | Task description → Studio wizard answers, 14 presets + cross-field validator | **Configure** |
| 4 | `olmoearth-embeddings` | Task profile → embeddings-vs-fine-tune decision + a runnable notebook | **Configure** |
| 5 | `olmoearth-predict` | Core run primitive: submit / poll / pixel-value / features / files | **Run** |
| 6 | `olmoearth-change-detect` | Two-or-more-date trajectory diff (refuses naïve 2-date diffs) | **Run** |
| 7 | `olmoearth-baseline-compare` | Studio vs. a baseline foundation model, side-by-side on transfer regions | **Run** |
| 8 | `olmoearth-evaluate` | Spatial-block CV + NNDM-LOO over `/prediction-results` | **Analyze** |
| 9 | `olmoearth-similarity` | FAISS over fine-tuned OlmoEarth Base embeddings | **Analyze** |
| 10 | `olmoearth-uncertainty` | Repeated pixel-value + Meyer-Pebesma Area of Applicability | **Analyze** |
| 11 | `olmoearth-cloud-mask-audit` | CFMask / s2cloudless / Sen2Cor / MAJA ensemble disagreement | **Analyze** |
| 12 | `olmoearth-qgis-bridge` | Tile URLs → QGIS WMTS + COG with a sidecar uncertainty raster | **Integrate** |
| 13 | `olmoearth-data-export` | Export Studio projects + predictions to JSON, grouped by project or status | **Integrate** |
| 14 | `olmoearth-provenance` | Manifest wrapper around every API call; emits a replay script | **Report** |
| 15 | `olmoearth-case-narrative` | Stakeholder writeup with live tiles + a freshness gate | **Report** |

</details>

See [**`docs/SHOWCASE.md`**](docs/SHOWCASE.md) for every skill in action with real outputs — each driven by the live Qwen3.6 backbone (real reasoning, function calls, and results, not mockups). Per-skill specs are in [`SKILLS.md`](SKILLS.md); the function catalog, harness dataclasses, and operational rules are in [`PLAN.md`](PLAN.md).

## Web UI

```bash
make web                      # static demo  → http://localhost:8080
uv run olmoearth-agent-serve  # live bridge  → http://127.0.0.1:8088
```

A styled front-end: a multi-turn chat with saved history (localStorage), a collapsible Studio project tree (project → model → predictions → results), Markdown-rendered answers, and a per-turn "Reasoning & tools" disclosure. Served statically it's a scripted demo; served by the bridge (`olmoearth-agent-serve`) it streams the real `LeadAgent` over SSE. Bring-your-own-key: paste a Studio API key (top bar → "Add API key"), kept client-side. See [`webui/`](webui/) for source and design notes.

## Stack

<details>
<summary><strong>The stack</strong>, layer by layer</summary>

| Layer | What |
|---|---|
| **LLM** | **Default:** [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) (35B total / 3B active, hybrid Gated-DeltaNet + MoE), served locally as a 4-bit GGUF ([`unsloth/Qwen3.6-35B-A3B-GGUF`](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF) `UD-IQ4_XS`, ~17.7 GB). **Optional:** bring your own **Claude / ChatGPT / Gemini** key, selected in the web UI (model autodetect; key never stored) |
| **Serving** | [llama.cpp](https://github.com/ggml-org/llama.cpp) (`ghcr.io/ggml-org/llama.cpp:server-cuda`), OpenAI-compatible API, `--jinja` for tool calling: see [`docs/serving.md`](docs/serving.md) |
| **Harness** | Sandboxed Python interpreter + a compact function catalog (`system.*`, `olmoearth.*`, `eo.*`, `utils.*`) with operational constraints enforced ([`PLAN.md`](PLAN.md)) |
| **Skills** | 15-skill catalog; vendored #1-#4 via submodule `vendor/olmoearth-skills` |
| **Studio API** | `https://olmoearth.allenai.org/api/v1`, Bearer `OLMOEARTH_API_KEY` ([docs](https://docs.olmoearth.allenai.org/)) |

</details>

## Docs & links

- [**PLAN.md**](PLAN.md): the function catalog, harness dataclasses, and operational rules
- [**SKILLS.md**](SKILLS.md): the full 15-skill catalog (Prep / Configure / Run / Analyze / Integrate / Report)
- [**docs/SHOWCASE.md**](docs/SHOWCASE.md): every skill run live, with real outputs
- [**docs/CANON.md**](docs/CANON.md): the canonical facts the repo holds itself to
- [**CONTRIBUTING.md**](CONTRIBUTING.md) · [**CHANGELOG.md**](CHANGELOG.md) · [**AGENTS.md**](AGENTS.md)

## License

[OlmoEarth Artifact License](LICENSE) (Ai2) - free use with restrictions: no military/defense/surveillance or extractive uses; cite Ai2 and propagate the terms downstream. Matches [`allenai/olmoearth_pretrain`](https://github.com/allenai/olmoearth_pretrain/blob/main/LICENSE).

<div align="center"><sub>Built on OlmoEarth Studio by Ai2. A research demo, not an official product.</sub></div>
