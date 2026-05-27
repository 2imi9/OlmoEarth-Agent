# OlmoEarth Studio Auto-Research Agent

An auto-research agent for the [OlmoEarth Studio](https://allenai.org/blog/olmoearth) platform. It decomposes Earth-science research questions into Studio actions (define areas, import labels, configure datasets, train or embed, evaluate, publish results), executes them through Studio's HTTP API and adjacent geospatial tools, and improves itself from its own traces.

## Status

**v0.1 — architecture plan only.** No runnable code yet. See [`PLAN.md`](PLAN.md) for the full architecture, verified references, and roadmap.

## What's in this repo right now

- [`PLAN.md`](PLAN.md) — Verified-link architecture plan: seven-layer design, data-flow walkthrough, eight-phase roadmap, open questions.
- `LICENSE` — Apache 2.0.
- `.gitignore` — Python / EO / agent-state.

## Architectural anchors

| Layer | Reference |
|---|---|
| Harness | [ByteDance DeerFlow v2](https://github.com/bytedance/deer-flow) — LangGraph lead-agent + subagents-as-tools + middleware chain + MCP-first tools |
| Skill packaging | [NVIDIA AI-Q Agent Skills](https://docs.nvidia.com/aiq-blueprint/latest/integration/agent-skills.html) on the open [agentskills.io](https://agentskills.io) spec |
| Vision–language model | [Prismatic VLMs](https://arxiv.org/abs/2402.07865) + [unsloth/Qwen3.6-35B-A3B-NVFP4](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-NVFP4) |
| Geospatial foundation | [OlmoEarth-v1-Large](https://huggingface.co/allenai/OlmoEarth-v1-Large) embeddings as parallel encoder stream |
| Self-improvement | [Stanford CS329A](https://cs329a.stanford.edu/) — three-loop taxonomy (inference-time / train-time / open-ended) |

## OlmoEarth Studio API

- Docs: https://docs.olmoearth.allenai.org/
- Auth: https://docs.olmoearth.allenai.org/authentication/ — Bearer token; max 10 keys per account
- Live OpenAPI spec: https://olmoearth.allenai.org/api/v1/openapi.json

## Roadmap

P0 scaffolding → P1 harness MVP → P2 model stack → P3 geo stream → P4 inference-time self-improvement → P5 eval+observability → P6 train-time self-improvement → P7 open-ended. Detail in [`PLAN.md` §8](PLAN.md).

## License

Apache 2.0 — see [`LICENSE`](LICENSE).
