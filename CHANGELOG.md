# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog v1.1.0](https://keepachangelog.com/en/1.1.0/); this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

PR titles for each merged change appear under the appropriate section below.
See [`CONTRIBUTING.md`](CONTRIBUTING.md#7-documentation) for the convention.

## [Unreleased]

### Added
- **Skill #5 `olmoearth-predict` (core run loop)** — `tools/predict.py`:
  `olmoearth_search_predictions` (discover reusable `model_id`s) and
  `olmoearth_submit_prediction`; poll via the foundational
  `olmoearth_get_prediction`. `StudioClient` gains `search_predictions`
  and `submit_prediction` (the six `PredictionWrite` required fields:
  name, project_id, area_id, model_id, start_time, end_time).
  **Resolves the PLAN.md §4 `model_id` gap for the reuse case**:
  `predictions/search` returns each prediction's `model_id`, so a client
  discovers a reusable id by searching. Read paths verified live
  2026-05-28 (12 real predictions, all with model_ids); an agent run
  found PA Karst model_ids and recovered from a failed tool call mid-run.
  `submit` implemented + unit-tested (not live-created, to avoid side
  effects). Result sub-tools (pixel-value/features/files) are a follow-up
  within this skill. 3 new tests.
- **Skill #14 `olmoearth-provenance` (`src/olmoearth_agent/provenance/`)** —
  implements operational rule §3.13. `ProvenanceLog` lives on
  `ThreadState`; the lead agent records one `ProvenanceManifest` entry
  per dispatched tool call (tool name, sha256 of args, id-only result
  summary — never raw geometry). `to_json()` + `replay_script()` emit
  an auditable manifest and a replay skeleton. Tool bundle
  `olmoearth_provenance_summary` lets the agent report what it did.
  Added `ProvenanceManifest` to `types.py` (was spec-only in PLAN §2).
  Verified live 2026-05-28: agent run recorded `load_context` +
  `provenance_summary` with hashes and result summaries. 7 new tests.
- **Harness core (`src/olmoearth_agent/{types,studio,tools,harness,skills}`)** —
  the structure all 16 skills plug into:
  - `types.py` — harness dataclasses + `ApiEnvelope[T]` (the live
    `{records, meta, errors}` Studio response wrapper found 2026-05-28).
  - `studio/client.py` — async `StudioClient` (httpx, Bearer auth,
    envelope unwrap) with `users_me`, `search_projects`, `create_project`,
    `get_prediction`, `load_context`. Endpoints verified against the live
    API.
  - `tools/registry.py` — `ToolRegistry` + `ToolContext`; dispatch never
    raises (errors return to the model). `tools/studio.py` — the
    foundational `olmoearth_*` tool bundle (load_context / search_projects
    / create_project / get_prediction).
  - `harness/agent.py` — `LeadAgent` ReAct loop (DeerFlow v2 lead-agent
    shape): brief → LLM → tool dispatch → result, with a turn cap and the
    operational rules in the system prompt. `harness/state.py` —
    `ThreadState`.
  - `skills/registry.py` — manifest slotting all 16 skills (number,
    category, status, tools) + `build_default_registry()`.
- **Verified live end-to-end 2026-05-28**: `LeadAgent` →
  `OlmoEarthLLM` (Qwen3.6 4-bit GGUF via llama.cpp) → `StudioClient`
  (live Studio API) → answer. The agent called `olmoearth_load_context`
  and correctly filtered the user's real projects by topic.
- `tests/{studio,tools,harness,skills}` — 16 unit tests (mock HTTP +
  fake LLM) + 1 live integration test (`tests/harness/test_live.py`,
  needs `VLLM_ENDPOINT` + `OLMOEARTH_API_KEY`).
- `pyproject.toml` runtime dep: `httpx>=0.27`.
- **LLM serving client (`src/olmoearth_agent/llm/`)** — async OpenAI-
  compatible wrapper around the vLLM-served Qwen3.6-35B-A3B-NVFP4
  backbone. `OlmoEarthLLM.chat(messages, tools=..., mode=...)` returns
  a parsed `ChatResponse` with content, extracted `<think>` trace,
  tool calls, finish reason, and usage. Four sampling presets from
  the model card; default `thinking_general` with
  `chat_template_kwargs.preserve_thinking=True` for multi-turn agent
  runs. Synchronous `Tracer` protocol exposes request/response hooks
  for the provenance middleware (lands in PR #7).
- `docs/serving.md` — vLLM serve command, hardware requirements,
  **function-calling serve flags** (`--enable-auto-tool-choice
  --tool-call-parser` — required or tool calls come back as text;
  parser name flagged UNVERIFIED pending live confirmation), YaRN
  long-context recipe, agent-mode defaults.
- Client robustness: `_parse_completion` reads server-split
  `reasoning_content` (when served with `--reasoning-parser qwen3`)
  and otherwise extracts the inline `<think>` block — works either way.
- `docs/serving.md` — "Local development on ≤24 GB VRAM" section: 4-bit
  GGUF (`UD-IQ4_XS`) via llama.cpp `server-cuda` with `--jinja` for tool
  calling. **Function-call path verified end-to-end 2026-05-28** on an
  RTX 5090 Laptop (24 GB): NVFP4+vLLM stalls at memory profiling on
  24 GB (residual KV headroom too small), but the 4-bit GGUF loads to
  ~18.6 GB and the agent's `create_project(...)` tool call round-trips.
  Production stack stays vLLM+NVFP4 on datacenter Blackwell; this is a
  local-dev accommodation (same OpenAI protocol, client code unchanged).
- `docker/vllm.compose.yml` — pinned `vllm/vllm-openai:v0.19.0` for
  local dev (still requires Blackwell host).
- `tests/llm/` — mock-endpoint smoke tests via `pytest-httpx`: simple
  chat, `<think>` extraction, tool-call round-trip, preserve_thinking
  forwarding, top_k routing through `extra_body`, tracer hooks. Plus
  one live integration test (`@pytest.mark.integration`) that hits a
  real `vllm serve` instance when `VLLM_ENDPOINT` is set.
- `pyproject.toml` runtime dep: `openai>=1.50`. Dev deps:
  `pytest-asyncio>=0.24`, `pytest-httpx>=0.30`. Pytest config now
  pins `asyncio_mode = "strict"`.
- `.env.example`: `VLLM_ENDPOINT`, `VLLM_MODEL` (defaults pointing at
  local `vllm serve` on `http://localhost:8000/v1`).
- `PLAN.md` §4: explicit Studio API spec-version pin (`openapi.json` v0.1.0,
  pre-1.0) and verified findings on Firebase auth, `PredictionResultAccessLevel`,
  `PredictionUpdate` rename-only, free-form `PredictionRead.progress`,
  `*-management/` doc stubs, and the `TaskStatus`-vs-`PredictionStatus` enum split.
- `PLAN.md` §1: new `olmoearth.create_label` tool row (the Studio API treats
  labelset metadata and individual label classes as separate POSTs).
- `PLAN.md` §2: new `LabelsetSpec`, `LabelDef`, `DataPrepLabelSchema` dataclasses
  (clean separation between Studio-API schemas and the OlmoEarth dataset-prep
  layer's field names).
- **`SKILLS.md`** — detailed 16-skill catalog (Prep / Configure / Run /
  Analyze / Integrate / Report). Each skill has what / why / tools-composed
  with academic citations (Ploton 2020 spatial CV, Meyer-Pebesma 2021 AOA,
  Skakun CMIX 2022 cloud masks, WorldCereal 2025 lessons, IAMAP, NASA
  Similarity Search, etc.). Skill #16 (`roger-annotation-bridge`) added
  alongside the 15 from Ziming's source spec.
- `SKILLS.md`: "Existing implementations (upstream source)" section
  pinning skills #1–#4 to [`2imi9/OlmoEarth-Skills`](https://github.com/2imi9/OlmoEarth-Skills)
  — upstream unifies skills #1 + #2 as `olmoearth-data-prep`
  (split-vs-unify decision deferred to first end-to-end skill PR).
  Skill #16 target pinned to [`2imi9/Roger-Studio`](https://github.com/2imi9/Roger-Studio).
- `SKILLS.md`: "Vendoring policy" section — submodule vs copy-with-
  provenance choice deferred to first vendoring PR.
- `PLAN.md` §4 Skills row: references upstream
  [`2imi9/OlmoEarth-Skills`](https://github.com/2imi9/OlmoEarth-Skills)
  as canonical home for skills #1–#4.
- `PLAN.md` §1: three new global tools — `olmoearth.pixel_value`,
  `olmoearth.features_search`, `olmoearth.fetch_embedding` (used by skills
  #4, #5, #9, #10).
- `PLAN.md` §2: four new dataclasses — `PixelValueResult`, `FeatureMatch`,
  `EmbeddingVector`, `ProvenanceManifest`.
- `PLAN.md` §3: new operational rule **13 — Provenance manifest** (every
  `olmoearth.*` API call writes a `ProvenanceManifest` entry via
  `provenance_middleware` from skill #14).
- `PLAN.md` §7: new "Future work (parked)" section documenting the
  multimodal stack and train-time self-improvement tracks that are
  explicitly deferred.

### Changed
- `PLAN.md` bumped to **v0.4**. Scope narrowed to text-only LLM
  (`unsloth/Qwen3.6-35B-A3B-NVFP4`) with function calling. Multimodal
  stack (Prismatic / Q-Former / OlmoEarth embedding stream / NVFP4
  fine-tuning) moved to §7.1 Future work; train-time self-improvement
  moved to §7.2.
- `PLAN.md` §4 "Underlying stack" table collapsed from 7 rows to 5:
  dropped "Vision–language model" + "Geospatial encoder stream" +
  "Self-improvement"; added explicit "LLM" row pinning Qwen3.6-35B-A3B-NVFP4
  served via **vLLM ≥0.19.0** (upstream-recommended path for this NVFP4
  checkpoint; full `vllm serve` command in §4). Agent sessions use
  `chat_template_kwargs.preserve_thinking=True` per the model card.
- `PLAN.md` §7 "Future work" trimmed from per-bullet ref lists to two
  prose paragraphs. Noted that Qwen3.6 ships a native vision encoder,
  so re-opening §7.1 means using/replacing that tower, not training
  one from scratch.
- `SKILLS.md` "Existing implementations" + "Vendoring policy" sections
  tightened (split-vs-unify and submodule-vs-copy decisions left as
  one-line options rather than verbose A/B writeups).
- `PLAN.md` §6 roadmap rewritten from 7 generic phases (P0–P6) to
  skill-first: P0–P2 done (scaffold / gap closure / this rewrite),
  P3 = LLM serving + harness MVP, then one PR per skill ordered by
  case-study demand. Skills 14 (provenance) and 8 (evaluate) flagged
  for early landing because they're cross-cutting.
- `PLAN.md` §8 (was §7) references trimmed: in-scope LLM refs only;
  parked refs live in §7.1/§7.2.
- `PLAN.md` §3 rule list grew from 12 → 13 (provenance manifest).

### Changed (continued from PR #3)
- `PLAN.md` bumped to v0.3 (PR #3 increment, superseded by v0.4 here).
- `PLAN.md` §4: rewritten "Studio gaps" subsection from v0.1/v0.2's three
  UNVERIFIED items to verified findings — webhook absence CLOSED, fine-tune
  `model_id` field CONFIRMED but provenance still UNVERIFIED, rate limits
  CLOSED-as-undocumented.
- `PLAN.md` §5 example: uses `LabelsetSpec` + `create_label` flow and notes
  that every Studio Prediction requires a `model_id`.

### Changed
- `PLAN.md` bumped to v0.3.
- `PLAN.md` §4: rewritten "Studio gaps" subsection from v0.1/v0.2's three
  UNVERIFIED items to verified findings — webhook absence CLOSED, fine-tune
  `model_id` field CONFIRMED but provenance still UNVERIFIED, rate limits
  CLOSED-as-undocumented.
- `PLAN.md` §5 example: uses `LabelsetSpec` + `create_label` flow and notes
  that every Studio Prediction requires a `model_id`.

### Fixed
- `PLAN.md` §2 `PredictionStatus.state` enum corrected against the live
  `components.schemas.PredictionStatus`: `queued`→`pending`, `succeeded`→
  `completed`, added `cancelled` as a fifth terminal state.
- `PLAN.md` §2 `LabelSchema` retired — the v0.2 shape (`sample_category`,
  `es_label`, `oe_labels`) is the OlmoEarth dataset-prep / rslearn layer's
  schema, NOT the Studio API. Renamed to `DataPrepLabelSchema` and
  documented as a different layer; `LabelsetSpec` + `LabelDef` replace it
  for the API surface.
- `PLAN.md` §2 `PredictionRef.kind` docstring clarifies it's a client-side
  dispatch key, not a Studio API field (the API has no `kind` / `task_type`).

## [0.0.2] - 2026-05-27

### Added
- `CONTRIBUTING.md` modeled on NVIDIA earth2studio's developer workflow
  (DCO sign-off required, pre-commit mandatory, PR title into CHANGELOG,
  90% coverage gate, Black / Ruff / mypy / interrogate / NumPy docstrings /
  Apache-2.0 + SPDX headers).
- `AGENTS.md` following the [agents.md](https://agents.md/) spec for
  coding-agent onboarding.
- AI-assisted contribution policy: disclosure in PR description; no AI
  co-author trailers in commits (per OpenInfra Foundation convention).
- Branch naming convention: `2imi9/feature-<slug>`; no direct commits to `main`.

### Changed
- LICENSE copyright line updated to "OlmoEarth Agent contributors".

## [0.0.1] - 2026-05-26

### Added
- Initial `PLAN.md` and `README.md` defining the tool catalog, harness data
  classes, operational rules, and underlying-stack references — modeled on
  Google's Google Earth Agent shape (catalog + dataclasses + numbered rules).
- Apache-2.0 LICENSE.
- `.gitignore` covering Python build artifacts, virtual environments, test
  caches, type-checker caches, secrets (`.env` family, key files), model and
  data artifacts (safetensors, checkpoints, GeoTIFF, Zarr, Parquet), Hugging
  Face caches, and Claude Code agent state.
