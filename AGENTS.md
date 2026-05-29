# AGENTS.md

Onboarding context for coding agents (Claude Code, Cursor, Codex, Aider, …) working on this repository. Follows the [agents.md](https://agents.md/) convention — a sibling to `README.md` that holds agent-specific guidance.

The agent's runtime contract lives in [`PLAN.md`](PLAN.md). This file is about how to *contribute to* the codebase, not how the agent itself behaves at runtime.

---

## What this project is

A tool that drives the [OlmoEarth Studio](https://allenai.org/blog/olmoearth) platform from natural-language briefs. Tool catalog + harness dataclasses + operational rules — same shape as Google's Google Earth Agent. Not a "research framework", not "auto-research" anything.

---

## Quick commands

```bash
# Setup
uv sync
uv run pre-commit install

# Lint / format / type-check
uv run pre-commit run --all-files

# Tests
uv run pytest                       # unit only
uv run pytest -m integration        # needs OLMOEARTH_API_KEY

# Doc build (when docs/ exists)
uv run mkdocs serve
```

---

## Code layout

```
.
├── PLAN.md              # Source of truth: tool catalog, dataclasses, rules
├── README.md            # Public-facing summary
├── CONTRIBUTING.md      # Contributor workflow
├── AGENTS.md            # This file
├── LICENSE              # Apache-2.0
├── pyproject.toml       # (future) project + tool config
├── src/olmoearth_agent/
│   ├── types.py         # Harness dataclasses (PLAN.md §2)
│   ├── tools/           # Tool implementations (PLAN.md §1)
│   │   ├── system.py    # python / search / fetch
│   │   ├── olmoearth.py # Studio API wrappers
│   │   ├── eo.py        # STAC + tiling
│   │   └── utils.py     # geometry helpers
│   ├── harness/         # Lead agent + middleware (DeerFlow v2 pattern)
│   ├── skills/          # SKILL.md packages (agentskills.io spec)
│   └── mcp/             # MCP server: OlmoEarth Studio + adjacent
└── tests/
    ├── test_operational_rules.py  # One test per PLAN.md §3 rule
    └── ...
```

---

## Conventions you must follow

- **Branch naming:** `2imi9/feature-<short-kebab-slug>`. No direct commits to `main`.
- **DCO sign-off:** `git commit -s -m "..."` is required on every commit. The sign-off names a human; AI tools do not certify the DCO.
- **No AI co-author trailers in commits.** Don't add `Co-Authored-By: Claude` or similar. Disclose AI involvement in the PR description if it was substantial. See [`CONTRIBUTING.md` §8](CONTRIBUTING.md#8-ai-assisted-contributions).
- **PR title is release-note quality** (it goes into `CHANGELOG.md`).
- **`pre-commit` is required** before review.
- **Operational rules in [`PLAN.md`](PLAN.md) §3 are not suggestions.** A change that breaks one of them must update both the rule and the corresponding test in `tests/test_operational_rules.py`.

---

## Code style (full detail in `CONTRIBUTING.md` §5)

- Black formatter, 88-char line length.
- Ruff: `["E", "F", "S", "I", "PERF"]`. `E501` deferred to Black.
- mypy with `disallow_untyped_defs = true`. Type hints required on public functions.
- NumPy-style docstrings; interrogate `fail-under = 90`.
- Python 3.11+; PEP 604 `|` over `Union`/`Optional`.
- Apache-2.0 header + SPDX identifier on every source file.

---

## Architectural anchors

When choosing how to structure a change, defer to:

- [ByteDance DeerFlow v2](https://github.com/bytedance/deer-flow) for the harness shape — LangGraph lead agent + subagents-as-tools + middleware chain + MCP-first tools.
- [agentskills.io](https://agentskills.io) / [NVIDIA AI-Q Agent Skills](https://docs.nvidia.com/aiq-blueprint/latest/integration/agent-skills.html) for `SKILL.md` packaging.
- [OlmoEarth Studio OpenAPI](https://olmoearth.allenai.org/api/v1/openapi.json) as the source of truth for API endpoints (codegen, not hand-written wrappers).
- [`unsloth/Qwen3.6-35B-A3B-GGUF`](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF) (4-bit `UD-IQ4_XS`) served via llama.cpp for the LLM. Canonical facts in [`docs/CANON.md`](docs/CANON.md); keep docs aligned with it. (Multimodal/Prismatic is parked — `PLAN.md` §7.)

Anything that conflicts with `PLAN.md` is a bug in `PLAN.md` — open a PR that updates it rather than working around it.

---

## Pitfalls specific to this codebase

1. **Geospatial output safety.** `PLAN.md` §3.1 forbids raw lat/lon, WKT, or full GeoJSON in chat responses. If a tool returns geometries, write them to a file and reference the path.
2. **Studio long-running jobs are async.** `submit_prediction` returns a `PredictionRef`; never block on training. Downstream tools must accept the ref and poll.
3. **Sandbox bans `import`.** The `system:python` interpreter has libraries preloaded — do not generate code with `import` statements; rely on the preloaded names.
4. **Studio API key cap is 10 per account.** Never programmatically rotate keys; tests use a single key from env.
5. **Studio has no `/models` or `/jobs` resource.** All async work is `Predictions` (request) + `PredictionResults` (output incl. XYZ tiles and MVT vectors). Do not invent endpoints that aren't in [`openapi.json`](https://olmoearth.allenai.org/api/v1/openapi.json).

---

## When in doubt

Open a draft PR with the question in the body. The reviewer surface is small; getting a fast nod beats spending an afternoon guessing.
