# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog v1.1.0](https://keepachangelog.com/en/1.1.0/); this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

PR titles for each merged change appear under the appropriate section below.
See [`CONTRIBUTING.md`](CONTRIBUTING.md#7-documentation) for the convention.

## [Unreleased]

### Added
- Project scaffold: `pyproject.toml` (build backend, dev deps, Black / Ruff /
  mypy / interrogate / pytest configuration), `.pre-commit-config.yaml`
  (pinned hook versions modeled on earth2studio), `.env.example` template,
  empty `src/olmoearth_agent/` package skeleton with `py.typed` marker, and
  this `CHANGELOG.md` file.
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
  served via TensorRT-LLM/vLLM.
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
