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
