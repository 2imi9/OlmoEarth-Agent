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
