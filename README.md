<div align="center">

<img src="webui/assets/OlmoEarth-logo.png" alt="OlmoEarth" width="360">&nbsp;&nbsp;<img src="webui/assets/agent-tag.png" alt="Agent" height="42">

**Drive [OlmoEarth Studio](https://allenai.org/blog/olmoearth) from natural-language briefs - on a local LLM, or your own cloud API.**

[![License](https://img.shields.io/badge/License-OlmoEarth%20Artifact-1f6feb.svg)](LICENSE)
[![Skills](https://img.shields.io/badge/skills-13%20default%20%2B%205%20opt--in-F0529C.svg)](SKILLS.md)
[![LLM](https://img.shields.io/badge/LLM-local%20Qwen3.6%20%2B%20hosted-0FCB8C.svg)](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB.svg)](pyproject.toml)

</div>

<div align="center">

![OlmoEarth Agent live demo](webui/demo/olmoearth-agent-demo.gif)

</div>

<div align="center"><sub>Connect a Studio key, send a brief, and the agent loop streams its reasoning, tool calls, results, and a plain-English answer -- in a multi-turn chat with saved history, a collapsible Studio project tree, and Markdown answers. Need an area of interest? <strong>Draw it on a map</strong> right in the chat and it is stored as a Studio area. &nbsp; | &nbsp; <a href="webui/demo/olmoearth-agent-demo.mp4">walkthrough (MP4)</a></sub></div>

<div align="center"><sub>Demo recorded on <a href="https://github.com/2imi9/OlmoEarth-Agent/releases/tag/v1.2.0">v1.2.0</a>.</sub></div>

---

OlmoEarth Agent runs [OlmoEarth Studio](https://allenai.org/blog/olmoearth) from a natural-language brief and checks what Studio returns where it can. It finds a model, submits and polls predictions, and compares results. For a map that carries scores (a Studio result whose band is a `[0, 1]` score, a regression band given a threshold, or a direct model run's scores), it then says which windows a reviewer should check first, which windows to label, and how wrong the map is, with the interval that sample earns (the verification layer, through [olmoearth-inferencex](https://github.com/2imi9/olmoearth_inferenceX)). A Studio class map carries no score, so it gets no review set and no certified zone; labels on a random or class-design sample still give its error rate and per-class accuracy over its sampled grid points (at most 16 x 16, one pixel each), not every pixel. Two maps of the same area and time get the which-of-two-maps route: two Studio results (class maps, no score needed), or two scores files of one window grid from `olmoearth_scores_from_file` (not plain class-map rasters). Labels on the windows where they differ say which map is more accurate when the exact sign test (or, past a fifth of those windows, the interval) can tell; otherwise they name neither. The difference in accuracy comes with a 95% interval. Two Studio results are compared on sampled grid points (at most 12 x 12, one pixel each), so often every differing point gets labelled, and the verdict covers those points, not every pixel. For scores files with many differing windows, olmoearth-inferencex suggests about 100 labels. Every tool call goes into a provenance manifest. Tools for training models outside Studio, and most tools whose inputs no Studio run produces, are opt-in. It runs on a local **Qwen3.6-35B-A3B** model served with llama.cpp by default (no hosted LLM required), and can optionally use your own **cloud API** key via the web UI's model-backend picker.

## Quick start

> **Prerequisites:** [uv](https://docs.astral.sh/uv/) and an OlmoEarth Studio API key (Studio UI -> profile -> **API Keys**). The **local model** path also needs [Docker](https://docs.docker.com/get-docker/); the **cloud-API** path needs neither Docker nor a download.

Pick the backbone for the agent's reasoning - both drive the same live UI:

**A. Cloud API** (Claude / ChatGPT / Gemini) - no Docker, no 17.7 GB download:

```bash
make setup      # uv sync --all-extras
make bridge     # live web UI on http://localhost:8088 (no local model needed)
```

Open **http://localhost:8088**, paste your Studio key, then go to **Settings -> LLM backend**, pick a provider, and paste that provider's API key. Both keys stay client-side and are sent per request, never stored server-side. Send a brief.

**B. Local model** (fully offline, free) - one command auto-starts the LLM and the UI:

```bash
make setup      # uv sync --all-extras
make up         # start the 4-bit Qwen3.6 LLM (first run pulls ~17.7 GB), then the live UI
```

Open **http://localhost:8088**, paste your Studio key, and send a brief. (`make up` is just `make serve` then `make bridge`; run them separately if you prefer.)

**Prefer the terminal?** Run a one-shot brief against the local model instead of the web UI:

```bash
export OLMOEARTH_API_KEY=...   # Studio UI -> profile -> API Keys
make serve && make agent       # make agent runs a sample brief; ask your own with Q="..."
```

`make help` lists every target; `make down` stops the LLM; `make web` serves a backend-free **demo** UI on `:8080`. `LLM_ENDPOINT` defaults to `http://localhost:8000/v1`. No `make` (e.g. Windows + Git Bash)? [`./scripts/quickstart.sh`](scripts/quickstart.sh) runs setup + serve and prints the exact `uv run` commands; see [`docs/serving.md`](docs/serving.md) for the cloud-API setup and the raw `docker compose` invocations.

## What it does

The LLM reads the brief, plans, and emits a tool call; the harness dispatches it as a function-call tool, feeds the result back, and iterates until it can answer. (An opt-in `system:python` subprocess is available for light glue between calls.) The harness records a provenance manifest for every tool call. The agent's persona and guardrails live in a versioned soul artifact (`harness/soul.md`; swap via `OLMOEARTH_SOUL_PATH`), followed in the system prompt by a capability card built from the tools' code (what each tool does, needs and cannot do, and what no tool can do; `harness/capabilities.py`); tool arguments are schema-validated at dispatch; oversized tool results spill to workspace files instead of the context window; the answer is checked against the run before it is shown (its numbers against the tool results and the user's messages, its directions, places and magnitudes against the facts the tools computed, its claims of files saved or items listed against what the run did, and the claims the tools forbid or require; and, by the agent's own model, against the run's record for statements no tool output supports and offers no tool can carry out), every problem found goes into one request to rewrite the answer, and a sentence that still fails is shown marked `[unverified: <check>]` (`harness/checks.py`, `harness/claim_check.py`; off with `OLMOEARTH_CHECK_NUMBERS=0` and `OLMOEARTH_CHECK_ANSWER=0`; the model's own reading is opt-in with `OLMOEARTH_CHECK_CLAIMS=1`); and durable user preferences (default project / area / sources, saved via `olmoearth_remember`) carry across conversations.

> **Production posture:** the per-capability egress allowlist defaults to `audit` (log-only) so a first run never breaks. For any deployment beyond your own laptop, set `OLMOEARTH_EGRESS=enforce` -- unknown hosts are then refused, not just logged (add self-hosted endpoints via `OLMOEARTH_EGRESS_ALLOW`).

The capability set ships as **18 skills**, grouped by EO-workflow stage. Four of them (#1-#3 and #17) are instruction packages (`SKILL.md`) that ship inside the Python package and load on demand with `olmoearth_load_skill`.

The agent registers 28 tools by default: the Studio run loop, the comparison and verification tools, and the plumbing (provenance, export, QGIS, memory). All 28 are sent on every turn. Another 15 opt-in tools stay in the code but off the default list: they train models outside Studio, need inputs no Studio run produces, repeat what Ai2's Explorer and Asta already do, or name a winner with no test (#6's baseline comparison: which map is more accurate is the two-map route below). Skills #6, #8, #10, #15 and #16 are opt-in whole, and so is #18's rule grader. To register them, start the agent or the bridge with `OLMOEARTH_OPT_IN_TOOLS=1` (e.g. `OLMOEARTH_OPT_IN_TOOLS=1 make bridge`); [`SKILLS.md`](SKILLS.md#opt-in-tools) gives each one's reason. Without the setting, the web UI's "/" menu still lists the five opt-in skills, and the bridge runs a brief that picks one without pinning it.

| Verification route | Needs | Gives |
|---|---|---|
| Which windows to check first | a map with scores | the review set at a budget (`olmoearth_review_set`, `olmoearth_review_set_from_result`) |
| How wrong one map is | a map with scores, labels on a planned sample (about 300 for a scores file; a Studio result's grid holds at most 256 windows) | the error rate with the interval its design earns, per-class accuracy, a certified zone (`olmoearth_plan_label_sample`, then `olmoearth_estimate_map_error` / `olmoearth_certify_zone`) |
| A map from a direct OlmoEarth run, not Studio | `oe-inferencex from-olmoearth`'s `scores_<EPSG>.tif` and `olmoearth_output.json`, or `score_area.py`'s raster and `manifest.json`, holding probabilities or logits (class ids are refused, with how to get probabilities). For a certificate: a `seed_from_map` random plan and its labels, fixed once given | every window of the run's raster, for the review set, the error estimate and the two-map route (`olmoearth_scores_from_file`). A certified zone packed into a certificate anyone holding the map can check, and that check (`olmoearth_write_certificate`, `olmoearth_verify_certificate`). The certificate restates the zone; it adds no stronger guarantee |
| How wrong a Studio class map is | a classification result (classes, no scores), labels on a random sample or a sample stratified by map class (`design='random'` or `'class'`; olmoearth-inferencex's label budget for a random sample's interval width) | the error rate and per-class accuracy over the sampled grid points (at most 16 x 16, one pixel each), not every pixel: the interval valid at every sample size first, then the design's (`olmoearth_plan_label_sample`, then `olmoearth_estimate_map_error`); no review set and no certified zone without probabilities |
| Which of two maps is more accurate | two maps of the same area and time (two Studio results, or two scores files of one window grid), labels on the windows where they differ (about 100 for scores files; two Studio results are sampled on at most 12 x 12 points, so often every differing point) | which map is more accurate, or neither, by the exact sign test (the interval past a fifth of the differing windows), and the difference in accuracy with a 95% interval, over the windows compared (for Studio results, the sampled points) (`olmoearth_plan_label_sample` with `design='disagreement'`, then `olmoearth_estimate_map_error`); refused when the dates differ (a Studio prediction's start and end times, or `date_a` / `date_b` for scores files); with no dates given, the result says so |

See it on a real map: `make verify-demo` runs the review set, estimate, certified zone and certificate tools on the expert-annotated Dynamic World tile that olmoearth-inferencex ships. It needs the `inferencex` extra (`make setup`, Python 3.11 to 3.13), no Studio key and no LLM. [docs/VERIFICATION_DEMO.md](docs/VERIFICATION_DEMO.md) shows each result and whether each interval covers the true error rate of the tile's annotated 40 m windows (not Dynamic World's 10 m accuracy).

<details>
<summary><strong>The 18 skills</strong>, by workflow stage (Prep / Configure / Run / Analyze / Integrate / Report); <em>opt-in</em> marks what needs <code>OLMOEARTH_OPT_IN_TOOLS=1</code></summary>

| # | Skill | What it does | Stage |
|---|---|---|---|
| 1 | `olmoearth-data-prep` | Labels (GeoJSON/CSV/Shapefile) -> Studio-importable file (MIME / 10K / multi-metric guards) AND `rslearn` `dataset.json` + Lightning YAML, with a 7-criteria audit | **Prep** |
| 2 | `olmoearth-studio-job-config` | Task description -> Studio wizard answers, 14 presets + cross-field validator | **Configure** |
| 3 | `olmoearth-embeddings` | Embeddings-vs-fine-tune **guidance + a runnable notebook**; Ai2's precomputed [OlmoEarth v1.3 embeddings](https://source.coop/olmoearth/olmoearth-v1-3-embeddings) (one vector per 10 m pixel per year) replace computing them yourself for an annual task in an area and year they cover. The **one-call** `olmoearth_automate` (decide + propose a config; optional HuggingFace introspection) is *opt-in* | **Configure** |
| 4 | `olmoearth-predict` | Core run primitive: submit / poll / fetch results, sample the output at a point, compare results with no ground truth | **Run** |
| 5 | `olmoearth-change-detection` | Change over time: one model's dated Studio results traced as a series (`olmoearth_compare_results`, mode `series`: date-ordered, legend-calibrated shifts; estimate movement, not verified ground change). The trajectory of your own dated values (`olmoearth_change_detect`, refuses fewer than 3 dates) is *opt-in* | **Run** |
| 6 | `olmoearth-baseline-compare` | *Opt-in.* Studio vs. a baseline foundation model, side-by-side on transfer regions; its winner is a point metric with no test, so which map is more accurate is #18's two-map route (both maps as Studio results, or as scores files of one window grid) | **Run** |
| 7 | `olmoearth-evaluate` | Per-class accuracy against your labels; spatial-block CV and NNDM-LOO are *opt-in* | **Analyze** |
| 8 | `olmoearth-similarity` | *Opt-in.* Exact top-K kNN over supplied embeddings (e.g. OlmoEarth Base) | **Analyze** |
| 9 | `olmoearth-uncertainty` | Ensemble-disagreement confidence across Studio results; the Meyer-Pebesma Area-of-Applicability (OOD) flag is *opt-in* | **Analyze** |
| 10 | `olmoearth-cloud-mask-audit` | *Opt-in.* CFMask / s2cloudless / Sen2Cor / MAJA ensemble disagreement | **Analyze** |
| 11 | `olmoearth-qgis-bridge` | Tile URLs -> a QGIS `.qlr` layer + GDAL_WMS descriptor + XYZ URLs + SLD ramp + legend + a local `gdal_translate` COG recipe (key never embedded) | **Integrate** |
| 12 | `olmoearth-data-export` | Export Studio projects + predictions to JSON, grouped by project or status | **Integrate** |
| 13 | `olmoearth-provenance` | Manifest wrapper around every API call; emits a replay script | **Report** |
| 14 | `olmoearth-case-narrative` | Stakeholder writeup with live tiles + a freshness gate. It states the accuracy the verification tools (#18) measured in the same request, as they returned it: estimate, intervals, labels, design, seed. If none was measured, it says so. An accuracy figure the model typed is refused | **Report** |
| 15 | `olmoearth-litsearch` | *Opt-in.* arXiv + OpenAlex literature search + DOI/arXiv-id resolution to ground citations; optional [Asta](https://github.com/allenai/asta-plugins) full-text ranked search when its CLI is installed | **Report** |
| 16 | `olmoearth-negative-sampler` | *Opt-in.* Presence-only labels -> trainable set: generates a buffered, spatially-thinned (optionally embedding-dissimilar) negative class so the data-prep audit passes | **Prep** |
| 17 | `olmoearth-rslearn` | Guidance for running rslearn (the data + training engine under OlmoEarth): the 4-stage pipeline and fit/predict, as commands for you to run. The torch-free tools to **recommend** a full setup from a plain-language goal and **validate** a config (shapes / label-type / bands) before a multi-hour run are *opt-in* | **Configure** |
| 18 | `olmoearth-review-set` | **Which windows to check first, without labels.** Ranks windows by the model's own top-1-minus-top-2 margin at your review budget (boundary-first optional), reports the attainable ceiling `min(1, budget/error_rate)` so a capture is read against what was reachable, and (*opt-in*) grades any candidate audit rule against the margin and a no-model control with a per-group sign test. Bring-your-own scores: Studio returns hard classes only. Which of two class maps of one time is more accurate needs no scores: labels on a random sample of the windows where they differ (`design='disagreement'`). A direct model run's files (`score_area.py`'s or `from-olmoearth`'s), the label-sample plan, the error estimate, the certified zone and its certificate, the two-map route and `olmoearth_compare_review`'s reading of dates need the `inferencex` extra (`pip install 'olmoearth-agent[inferencex]'`, olmoearth-inferencex 1.20.1 or later, Python 3.11 to 3.13; `make setup` installs it there) | **Analyze** |

</details>

See [**`docs/SHOWCASE.md`**](docs/SHOWCASE.md) for captured runs of skills #1-#16 with real outputs, driven by the live Qwen3.6 backbone (captured with an earlier tool set; some of them call tools that are now opt-in, see [`SKILLS.md`](SKILLS.md#opt-in-tools)). Per-skill specs are in [`SKILLS.md`](SKILLS.md); the function catalog, harness dataclasses, and operational rules are in [`PLAN.md`](PLAN.md).

## Stack

<details>
<summary><strong>The stack</strong>, layer by layer</summary>

| Layer | What |
|---|---|
| **LLM** | **Default:** [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) (35B total / 3B active, hybrid Gated-DeltaNet + MoE), served locally as a 4-bit GGUF ([`unsloth/Qwen3.6-35B-A3B-GGUF`](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF) `UD-IQ4_XS`, ~17.7 GB). **Optional:** bring your own **cloud API** key, selected in the web UI (model autodetect; key never stored) |
| **Serving** | [llama.cpp](https://github.com/ggml-org/llama.cpp) (`ghcr.io/ggml-org/llama.cpp:server-cuda`), OpenAI-compatible API, `--jinja` for tool calling: see [`docs/serving.md`](docs/serving.md) |
| **Harness** | A catalog of function-call tools (more behind `OLMOEARTH_OPT_IN_TOOLS=1`), an **opt-in** `system:python` subprocess for light glue (`OLMOEARTH_RUN_PYTHON=1`), and a provenance manifest per call ([`PLAN.md`](PLAN.md)) |
| **Skills** | 18-skill catalog ([`SKILLS.md`](SKILLS.md)); the instruction packages for #1-#3 and #17 are in [`src/olmoearth_agent/skills/packages/`](src/olmoearth_agent/skills/packages/) |
| **Studio API** | `https://olmoearth.allenai.org/api/v1`, Bearer `OLMOEARTH_API_KEY` ([docs](https://docs.olmoearth.allenai.org/)) |

</details>

## Docs & links

- [**PLAN.md**](PLAN.md): the function catalog, harness dataclasses, and operational rules
- [**SKILLS.md**](SKILLS.md): the 18-skill catalog (Prep / Configure / Run / Analyze / Integrate / Report)
- [**docs/SHOWCASE.md**](docs/SHOWCASE.md): captured runs of skills #1-#16
- [**docs/CANON.md**](docs/CANON.md): the canonical facts the repo holds itself to
- [**CONTRIBUTING.md**](CONTRIBUTING.md) | [**CHANGELOG.md**](CHANGELOG.md) | [**AGENTS.md**](AGENTS.md)

## License

[OlmoEarth Artifact License](LICENSE) (Ai2) - free use with restrictions: no military/defense/surveillance or extractive uses; cite Ai2 and propagate the terms downstream. Matches [`allenai/olmoearth_pretrain`](https://github.com/allenai/olmoearth_pretrain/blob/main/LICENSE).

<div align="center"><sub>Built in collaboration with Ai2, on OlmoEarth Studio. A research demo, not an official product.</sub></div>
