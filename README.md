<div align="center">

<img src="webui/assets/OlmoEarth-logo.png" alt="OlmoEarth" width="360">&nbsp;&nbsp;<img src="webui/assets/agent-tag.png" alt="Agent" height="42">

**Drive [OlmoEarth Studio](https://allenai.org/blog/olmoearth) from natural-language briefs: a local-LLM analog to Google's Earth Agent.**

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-1f6feb.svg)](LICENSE)
[![Skills](https://img.shields.io/badge/skills-15-F0529C.svg)](SKILLS.md)
[![LLM](https://img.shields.io/badge/LLM-Qwen3.6--35B--A3B%20%C2%B7%20local-0FCB8C.svg)](docs/serving.md)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB.svg)](pyproject.toml)

</div>

<div align="center">

![OlmoEarth Agent live demo](webui/demo/olmoearth-agent-demo.gif)

</div>

<div align="center"><em>A brief in, the agent loop streamed out: reasoning, a tool call, the result, and a plain-English answer. &nbsp;·&nbsp; <a href="webui/">Open the web UI →</a></em></div>

<div align="center"><sub>A multi-turn <strong>chat</strong> with saved history, a collapsible Studio <strong>project tree</strong>, and Markdown answers (full walkthrough: <a href="webui/demo/olmoearth-agent-demo.mp4">MP4</a>). Served by the bridge (<code>olmoearth-agent-serve</code>), it streams the real agent.</sub></div>

---

OlmoEarth Agent turns a natural-language brief into real geospatial work on **OlmoEarth Studio**. It's a compact analog to Google's Earth Agent: a small catalog of functions (Studio HTTP API, EO data fetch, geometry utilities) running over a sandboxed Python interpreter, with operational constraints built in. The agent reasons about the ask, calls the right tools, submits and polls predictions, and reports **honest results**: every API call is wrapped in a provenance manifest, spatial cross-validation is mandatory on auto-correlated AOIs, and raw coordinates never leak into chat. It runs entirely on a **local Qwen3.6-35B-A3B** model served via llama.cpp, with no hosted LLM required.

## Contents

- [Quick start](#quick-start)
- [What it does](#what-it-does)
- [Web UI](#web-ui)
- [Stack](#stack)
- [Docs & links](#docs--links)
- [License](#license)

## Quick start

> **Prerequisites:** [Docker](https://docs.docker.com/get-docker/) (to serve the LLM), [uv](https://docs.astral.sh/uv/), and an `OLMOEARTH_API_KEY` (Studio UI → profile → **API Keys**).

```bash
make setup                                              # uv sync + vendored skills (#1-#4)
make serve                                              # 4-bit Qwen3.6 GGUF via llama.cpp (docker)
make agent Q="How many OlmoEarth Studio projects do I have?"
make web                                                # styled web UI on http://localhost:8080
```

Set your key first (`export OLMOEARTH_API_KEY=...`); `LLM_ENDPOINT` defaults to `http://localhost:8000/v1`. Run `make help` for the full target list, and `make down` to stop the LLM when you're done.

**No `make`** (e.g. Windows + Git Bash)? Run [`./scripts/quickstart.sh`](scripts/quickstart.sh): it does setup + serve, then prints the exact `uv run olmoearth-agent "…"` and web-UI commands to copy. See [`docs/serving.md`](docs/serving.md) for the raw `docker compose` + `uv run` invocations.

<div align="right"><a href="#contents">↑ back to top</a></div>

## What it does

The loop is straightforward: the LLM reads the brief, plans, and emits a tool call; the harness runs it in a **sandboxed geospatial Python interpreter** (`pandas`, `geopandas`, `xarray`, `rioxarray`, `shapely`, `pystac_client`, `planetary_computer`, `rslearn`, … preloaded, no `import` statements), feeds the result back, and iterates until it can answer. State persists across turns, and the operational rules (default trailing-12-month windows, cost guards on fine-tunes, mandatory spatial CV on auto-correlated AOIs, a provenance manifest per call) are enforced by the harness, not left to the model.

The capability set ships as **15 skills**, grouped by where they sit in an EO workflow:

| # | Skill | What it does | Stage |
|---|---|---|---|
| 1 | `olmoearth-studio-upload` | Labels (GeoJSON/CSV/Shapefile) → Studio-importable file with MIME / 10K / multi-metric guards | **Prep** |
| 2 | `olmoearth-rslearn-config` | Labels → `rslearn` `dataset.json` + Lightning YAML with a 7-criteria audit | **Prep** |
| 3 | `olmoearth-studio-job-config` | Task description → Studio wizard answers, 14 presets + cross-field validator | **Configure** |
| 4 | `olmoearth-embeddings` | Task profile → embeddings-vs-fine-tune decision + a runnable notebook | **Configure** |
| 5 | `olmoearth-predict` | Core run primitive: submit / poll / pixel-value / features / files | **Run** |
| 6 | `olmoearth-change-detect` | Two-or-more-date trajectory diff (refuses naïve 2-date diffs) | **Run** |
| 7 | `olmoearth-baseline-compare` | Studio vs. AlphaEarth, side-by-side on transfer regions | **Run** |
| 8 | `olmoearth-evaluate` | Spatial-block CV + NNDM-LOO over `/prediction-results` | **Analyze** |
| 9 | `olmoearth-similarity` | FAISS over fine-tuned OlmoEarth Base embeddings | **Analyze** |
| 10 | `olmoearth-uncertainty` | Repeated pixel-value + Meyer-Pebesma Area of Applicability | **Analyze** |
| 11 | `olmoearth-cloud-mask-audit` | CFMask / s2cloudless / Sen2Cor / MAJA ensemble disagreement | **Analyze** |
| 12 | `olmoearth-qgis-bridge` | Tile URLs → QGIS WMTS + COG with a sidecar uncertainty raster | **Integrate** |
| 13 | `olmoearth-data-export` | Export Studio projects + predictions to JSON, grouped by project or status | **Integrate** |
| 14 | `olmoearth-provenance` | Manifest wrapper around every API call; emits a replay script | **Report** |
| 15 | `olmoearth-case-narrative` | Stakeholder writeup with live tiles + a freshness gate | **Report** |

See [**`docs/SHOWCASE.md`**](docs/SHOWCASE.md) for **every skill in action, with real outputs**: each one, in catalog order, driven by the live Qwen3.6 backbone (real reasoning, real function calls, real results, not mockups). Per-skill specs are in [`SKILLS.md`](SKILLS.md); the function catalog, harness dataclasses, and operational rules are in [`PLAN.md`](PLAN.md).

<div align="right"><a href="#contents">↑ back to top</a></div>

## Web UI

```bash
make web                      # static demo  → http://localhost:8080
uv run olmoearth-agent-serve  # LIVE bridge  → http://127.0.0.1:8088
```

A styled front-end (dark-teal canvas, OlmoEarth-pink `#F0529C`, inspired by Ai2 **Asta**): a multi-turn **chat with saved history** (localStorage), a collapsible **Studio project tree** (project → model → predictions → results), **Markdown-rendered answers**, and a per-turn "Reasoning & tools" disclosure. Served statically it's a scripted demo; served by the **bridge** (`olmoearth-agent-serve`) it streams the real `LeadAgent` over SSE. **Bring-your-own-key**: paste a Studio API key (top bar → "Add API key"), kept client-side. See [`webui/`](webui/) for source + design notes.

<div align="right"><a href="#contents">↑ back to top</a></div>

## Stack

| Layer | What |
|---|---|
| **LLM** | [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) (35B total / 3B active, hybrid Gated-DeltaNet + MoE), served **locally** as a 4-bit GGUF ([`unsloth/Qwen3.6-35B-A3B-GGUF`](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF) `UD-IQ4_XS`, ~17.7 GB) |
| **Serving** | [llama.cpp](https://github.com/ggml-org/llama.cpp) (`ghcr.io/ggml-org/llama.cpp:server-cuda`), OpenAI-compatible API, `--jinja` for tool calling: see [`docs/serving.md`](docs/serving.md) |
| **Harness** | Sandboxed Python interpreter + a compact function catalog (`system.*`, `olmoearth.*`, `eo.*`, `utils.*`) with operational constraints enforced ([`PLAN.md`](PLAN.md)) |
| **Skills** | 15-skill catalog; vendored #1-#4 via submodule `vendor/olmoearth-skills` |
| **Studio API** | `https://olmoearth.allenai.org/api/v1`, Bearer `OLMOEARTH_API_KEY` ([docs](https://docs.olmoearth.allenai.org/)) |

<div align="right"><a href="#contents">↑ back to top</a></div>

## Docs & links

- [**PLAN.md**](PLAN.md): the function catalog, harness dataclasses, and the operational rules
- [**SKILLS.md**](SKILLS.md): the full 15-skill catalog (Prep / Configure / Run / Analyze / Integrate / Report)
- [**docs/SHOWCASE.md**](docs/SHOWCASE.md): every skill run live, with real outputs
- [**docs/CANON.md**](docs/CANON.md): the canonical facts the repo holds itself to
- [**CONTRIBUTING.md**](CONTRIBUTING.md) · [**CHANGELOG.md**](CHANGELOG.md) · [**AGENTS.md**](AGENTS.md)

## License

[Apache License 2.0](LICENSE).

<div align="center"><sub>Built on OlmoEarth Studio by Ai2. Web UI styling inspired by Ai2 Asta. This is a research-demo project, not an official product page.</sub></div>
