# AGENTS.md

Onboarding context for coding agents (Claude Code, Cursor, Codex, Aider, ...) working on this repository. Follows the [agents.md](https://agents.md/) convention: a sibling to `README.md` that holds agent-specific guidance.

The agent's runtime contract lives in [`PLAN.md`](PLAN.md). This file is about how to *contribute to* the codebase, not how the agent itself behaves at runtime.

---

## What this project is

A tool that drives the [OlmoEarth Studio](https://allenai.org/blog/olmoearth) platform from natural-language briefs. Tool catalog + harness dataclasses + operational rules.

---

## Quick commands

```bash
# Setup
uv sync --all-extras
uv run pre-commit install

# Lint / format / type-check
uv run pre-commit run --all-files

# Tests
uv run pytest                       # unit only
uv run pytest -m integration        # needs OLMOEARTH_API_KEY
```

---

## Code layout

```
.
├── PLAN.md              # Source of truth: tool catalog, dataclasses, rules
├── README.md            # Public-facing summary
├── CONTRIBUTING.md      # Contributor workflow
├── AGENTS.md            # This file
├── LICENSE              # OlmoEarth Artifact License (Ai2)
├── pyproject.toml       # project + tool config (black/ruff/mypy/pytest)
├── src/olmoearth_agent/
│   ├── types.py         # Studio envelope, context and provenance dataclasses
│   ├── cli.py           # `olmoearth-agent` CLI entrypoint
│   ├── serve.py         # `olmoearth-agent-serve` web bridge (serves webui/)
│   ├── llm/             # LLM clients: local Qwen + Claude/ChatGPT/Gemini
│   ├── studio/          # OlmoEarth Studio API client
│   ├── tools/           # Tool implementations (PLAN.md §1)
│   ├── harness/         # Lead agent + middleware
│   ├── bridge/          # `olmoearth-agent-bridge`: a Claude Code session as the model (docs/CLAUDE_AS_MODEL.md)
│   ├── skills/          # catalog (registry.py), loader, SKILL.md packages (packages/)
│   ├── analysis/        # computations behind the tools (comparison, review set, ...)
│   ├── reporting/       # case narrative
│   ├── provenance/      # tool-call manifest + replay
│   └── security/        # egress allowlist + SSRF guard (PLAN.md §3.4)
└── tests/               # unit + integration (mirrors the package layout)
```

---

## Conventions you must follow

The full rules are in [`CONTRIBUTING.md`](CONTRIBUTING.md); the ones agents most often miss:

- **Branch naming:** `2imi9/feature-<short-kebab-slug>`. No direct commits to `main`.
- **DCO sign-off** names a human; AI tools do not certify the DCO ([§4](CONTRIBUTING.md#4-commit-conventions)).
- **No AI co-author trailers in commits** ([§8](CONTRIBUTING.md#8-ai-assisted-contributions)).
- **`pre-commit` and `uv run pytest` must pass** before review. There is no CI to run them for you.
- **Operational rules in [`PLAN.md`](PLAN.md) §3 are not suggestions.** A change that breaks one must update the rule and, where one exists, its test; rule tests live beside the tool they guard (e.g. `tests/tools/test_compare_results.py::test_no_mode_returns_a_coordinate` for §3.1). The rules and clauses no code applied were retired in place on 31 October 2026 (the others keep their numbers); the rest describe what the code does.

---

## Code style

Formatter, linter, type checker, docstring threshold and licence headers: [`CONTRIBUTING.md` §5](CONTRIBUTING.md#5-code-style).

---

## Architectural anchors

When choosing how to structure a change, defer to:

- The harness is a single-agent tool-calling loop (`harness/agent.py`), shaped after [ByteDance DeerFlow v2](https://github.com/bytedance/deer-flow)'s lead agent without its subagents or LangGraph. Its behaviours (turn cap, retry hint, answer checks) are middleware with LangChain 1.x's hook interface (`harness/middleware.py`); add one as an `AgentMiddleware`, not as code in the loop.
- [agentskills.io](https://agentskills.io) / [NVIDIA AI-Q Agent Skills](https://docs.nvidia.com/aiq-blueprint/latest/integration/agent-skills.html) for `SKILL.md` packaging.
- [OlmoEarth Studio OpenAPI](https://olmoearth.allenai.org/api/v1/openapi.json) as the reference for API endpoints; `studio/client.py` is a hand-written client checked against it and live probes.
- [`unsloth/Qwen3.6-35B-A3B-GGUF`](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF) (4-bit `UD-IQ4_XS`) served via llama.cpp is the default local LLM; any OpenAI-compatible server works (see [`docs/serving.md`](docs/serving.md)). Canonical facts in [`docs/CANON.md`](docs/CANON.md); keep docs aligned with it. (Multimodal/Prismatic is parked, `PLAN.md` §7.)

Anything that conflicts with `PLAN.md` is a bug in `PLAN.md`: open a PR that updates it rather than working around it.

---

## Pitfalls specific to this codebase

1. **Geospatial output safety.** `PLAN.md` §3.1 forbids raw lat/lon, WKT, or full GeoJSON in chat responses. If a tool returns geometries, write them to a file and reference the path.
2. **Studio long-running jobs are async.** `submit_prediction` returns a prediction id; never block on training. Downstream tools must accept the ref and poll.
3. **The agent runs no code the model writes.** No tool executes a Python snippet or a shell command; the one child process is olmoearth-inferencex's own command line (`tools/inferencex.run_cli`, with the agent's keys scrubbed from its environment). Training outside Studio (a local rslearn pipeline) is outside the agent (`harness/soul.md`): it says so and hands over no pipeline code; the studio-job-config package's `scripts/recommend.py` is the user's to run.
4. **Studio API key cap is 10 per account.** Never programmatically rotate keys; tests use a single key from env.
5. **Every tool a catalog row lists is in the default registry.** No tool is opt-in (`skills/registry.py`). The one setting that leaves tools out is `OLMOEARTH_READ_ONLY` (pitfall 7), read by `tools/registry.opted_in`.
6. **`/models` is live but undocumented; there is no `/jobs` resource.** All async work is `Predictions` (request) + `PredictionResults` (output incl. XYZ tiles and MVT vectors). `/models` is absent from [`openapi.json`](https://olmoearth.allenai.org/api/v1/openapi.json) (v0.1.0) yet the live API answers `POST /models/search` and `GET /models/{id}` (verified 2026-05-30, still outside the spec on 2026-10-31; used to label a prediction's model as fine-tuned vs an embeddings run, and for the training selector `olmoearth_studio_label_inventory` replays). Once `/models` answers 405, or 404 as for a path no route matches, the client asks it no more. Don't invent *other* endpoints - confirm against the spec or a live probe first.
7. **A Studio key can be read-only, and nothing shows it.** `/users/me` lists the user's permissions, not the key's access level. `OLMOEARTH_READ_ONLY=1` leaves out the two write tools (`skills/registry.py`, `STUDIO_WRITE_TOOLS`) and puts the reason on the capability card. Drawing an area (`olmoearth_request_aoi`) stays: the web UI still saves it with `POST /areas`, and `serve.api_create_area` answers a refused save with 403. A refused request raises `StudioAccessError` (HTTP 401/403, never retried). Writes with a read-only key have never been tried live.
8. **A result's `download_token` is a credential.** `StudioClient.prediction_result_files` masks it in every error; never put a real one in a tool result, a manifest, a log or a test fixture.

---

## When in doubt

Open a draft PR with the question in the body. The reviewer surface is small; getting a fast nod beats spending an afternoon guessing.
