# OlmoEarth Studio Auto-Research Agent — Architecture Plan

**Status:** v0.1 draft, 2026-05-26
**Scope:** Generic auto-research agent for the OlmoEarth Studio platform. Domain-agnostic across Studio workflows (label import → fine-tune / embed+LP / reference inference).
**Verification discipline:** Every external claim has a real URL. Anything unverified is flagged inline as **UNVERIFIED**.

---

## 1. Executive summary

We are building an agent that helps Earth-science researchers run end-to-end studies on the [OlmoEarth Studio](https://allenai.org/blog/olmoearth) platform without hand-driving every step. The agent decomposes a research question into Studio actions (define areas, import labels, configure dataset, train / embed, evaluate, publish results), executes them through Studio's HTTP API and adjacent geospatial tools, and improves itself over time from its own traces.

The architecture borrows four ideas from production agent stacks already shipping in 2026, and grounds the learning loop in Stanford's [CS329A](https://cs329a.stanford.edu/) taxonomy:

| Layer | Inspiration | What we take |
|---|---|---|
| Harness | [ByteDance DeerFlow v2](https://github.com/bytedance/deer-flow) | LangGraph lead-agent + subagents-as-tools + 14-middleware chain + MCP-first tool surface |
| Skill packaging | [NVIDIA AI-Q Agent Skills](https://docs.nvidia.com/aiq-blueprint/latest/integration/agent-skills.html) on the open [agentskills.io](https://agentskills.io) spec | `SKILL.md` progressive disclosure, signed `skill-card.md` trust metadata |
| Vision–language model | [Prismatic VLMs](https://arxiv.org/abs/2402.07865) (Karamcheti et al., ICML 2024) + [unsloth/Qwen3.6-35B-A3B-NVFP4](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-NVFP4) | Fused vision encoder → MLP projector → MoE backbone, plus a parallel stream for OlmoEarth geo embeddings |
| Self-improvement | [Stanford CS329A](https://cs329a.stanford.edu/) | Three loops — inference-time search, train-time RL on traces, open-ended evolution — with verifier-grounded eval |

The end state: a domain-aware research agent that scientists can hand a question like *"map alfalfa fields in the Klamath basin from 2022–2024 Sentinel-2"* and get back a runnable Studio project, a fine-tuned or embedded inference, and a defensible report — with the agent getting measurably better at this class of tasks over time.

---

## 2. System block diagram

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  Researcher (CLI / Studio web UI / notebook)                                  │
└───────────────────────┬──────────────────────────────────────────────────────┘
                        │  natural-language brief, files, AOIs
                        ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  L3  Skill Library            ─── agentskills.io SKILL.md + skill-card.md     │
│  • aoi-fetch    • label-prep    • studio-project    • train-or-embed          │
│  • evaluate     • report        • cost-guard        • case-study-runbooks     │
└───────────────────────┬──────────────────────────────────────────────────────┘
                        ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  L2  Agent Harness            ─── LangGraph (DeerFlow-style)                  │
│  ┌─────────────────────────────────────────────────────────────────────┐     │
│  │ Lead agent  (create_agent loop)                                     │     │
│  │   ├─ subagent: research        (multi-step exploration)             │     │
│  │   ├─ subagent: bash            (sandboxed shell)                    │     │
│  │   ├─ subagent: eo-tile-fetch   (Sentinel-2/Landsat window pulls)    │     │
│  │   └─ subagent: critic          (Reflexion + process reward score)   │     │
│  │                                                                     │     │
│  │ 14-middleware chain                                                 │     │
│  │   summarization · todo · loop-detection · memory ·                  │     │
│  │   clarification · safety · guardrail · cost-guard ·                 │     │
│  │   eo-dataset-validation · spatial-cv-checker · …                    │     │
│  └─────────────────────────────────────────────────────────────────────┘     │
└─────────┬────────────────────────────────────────┬───────────────────────────┘
          │ tool calls                              │ subagent spawns
          ▼                                         ▼
┌──────────────────────────┐         ┌──────────────────────────────────────┐
│ L4  MCP tool surface     │         │  ThreadState (shared LangGraph state)│
│  • olmoearth-studio MCP  │         │  artifacts · todos · viewed_images   │
│  • planetary-computer    │         │  uploaded_files · sandbox · run_id   │
│  • nldi / nhd / huc      │         └──────────────────────────────────────┘
│  • roger-studio          │
│  • hf-datasets / models  │
└──────────────────────────┘
          │
          ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  L1  Model Stack                                                              │
│  ┌────────────────────┐   ┌────────────────────┐                              │
│  │ Prismatic vision   │   │ OlmoEarth encoder  │                              │
│  │ (DINOv2 + SigLIP)  │   │ (OlmoEarth-v1.x)   │                              │
│  └─────────┬──────────┘   └─────────┬──────────┘                              │
│            ▼  patch tokens          ▼  geo embedding                          │
│      ┌─────────────┐           ┌─────────────┐                                │
│      │ MLP project │           │ MoVA / Q-Fmr│                                │
│      └─────┬───────┘           └──────┬──────┘                                │
│            └────────────┬──────────────┘                                      │
│                         ▼                                                     │
│         Qwen3.6-35B-A3B-NVFP4 (MoE, 256 experts, 262K ctx)                    │
│         Unsloth NVFP4 weights · LoRA adapters trained in BF16                 │
└───────────────────────┬──────────────────────────────────────────────────────┘
                        ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  L0  Runtime           Blackwell B100/B200 · TensorRT-LLM ≥0.17 · vLLM        │
│  L5  Self-improvement  STaR / Reflexion / Archon search / SWiRL              │
│  L6  Eval + observ.    OTEL · LangSmith · METR · GDPVal · DeepScholar-Bench  │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Reference catalog (all verified unless flagged)

### 3.1 OlmoEarth Studio
- Docs landing — https://docs.olmoearth.allenai.org/
- Authentication — https://docs.olmoearth.allenai.org/authentication/
- Live OpenAPI 3 spec — https://olmoearth.allenai.org/api/v1/openapi.json
- Interactive API browser (JS-rendered) — https://docs.olmoearth.allenai.org/api/
- Launch blog — https://allenai.org/blog/olmoearth
- v1.1 blog — https://allenai.org/blog/olmoearth-v1-1
- Foundation model weights — https://huggingface.co/allenai/OlmoEarth-v1-Large
- Sample fine-tuning configs + CLI — https://github.com/allenai/olmoearth_projects
- Pretraining / eval code — https://github.com/allenai/olmoearth_pretrain
- Area-management reference (illustrative endpoint doc) — https://docs.olmoearth.allenai.org/area-management/

**Auth model:** `Authorization: Bearer <api_key>`; max 10 keys/account; key shown once on creation in Studio UI.
**Resources exposed by v1 OpenAPI:** Areas, Projects, Datasets, Labelsets, Labels, Annotations, Tasks, Predictions, PredictionResults, Users.
**Key gap:** No `/models` or `/jobs` resources. Long-running training/inference is modeled as `Predictions` (request) + `PredictionResults` (outputs incl. XYZ raster tiles, MVT vector tiles). Webhook / push-notification mechanism is **UNVERIFIED** — assume polling on `GET /predictions/{id}` until proven otherwise.

### 3.2 Harness — DeerFlow v2 (ByteDance)
- Repo — https://github.com/bytedance/deer-flow (Apache-style OSS, default branch `main`, created 2025-05-07)
- Project site — https://deerflow.tech/
- Architecture doc — https://github.com/bytedance/deer-flow/blob/main/backend/docs/ARCHITECTURE.md
- LangGraph entry config — https://github.com/bytedance/deer-flow/blob/main/backend/langgraph.json
- Lead agent — https://github.com/bytedance/deer-flow/blob/main/backend/packages/harness/deerflow/agents/lead_agent/agent.py
- Factory + middleware chain — https://github.com/bytedance/deer-flow/blob/main/backend/packages/harness/deerflow/agents/factory.py
- Subagent executor — https://github.com/bytedance/deer-flow/blob/main/backend/packages/harness/deerflow/subagents/executor.py
- Task-tool (subagent-as-tool) — https://github.com/bytedance/deer-flow/blob/main/backend/packages/harness/deerflow/tools/builtins/task_tool.py
- MCP integration — https://github.com/bytedance/deer-flow/blob/main/backend/packages/harness/deerflow/mcp/tools.py
- MCP server doc — https://github.com/bytedance/deer-flow/blob/main/backend/docs/MCP_SERVER.md
- ThreadState — https://github.com/bytedance/deer-flow/blob/main/backend/packages/harness/deerflow/agents/thread_state.py
- Skill installer + tool-policy filter — https://github.com/bytedance/deer-flow/blob/main/backend/packages/harness/deerflow/skills/installer.py , https://github.com/bytedance/deer-flow/blob/main/backend/packages/harness/deerflow/skills/tool_policy.py

**Flag:** DeerFlow restructured to v2 in Feb 2026. Older blog posts describe a v1 static planner/researcher/coder/reporter graph that is no longer in `main`. Pin to a specific commit if you reference line numbers; cite v2 only.
**No paper exists** — cite the repo + ARCHITECTURE.md.
**No self-critique node** — DeerFlow's `reflection/` directory is Python `importlib` introspection, not LLM reflection. We add the critic ourselves (§7).

### 3.3 Skills — NVIDIA AI-Q on the agentskills.io spec
- Primary doc — https://docs.nvidia.com/aiq-blueprint/latest/integration/agent-skills.html
- AI-Q docs index — https://docs.nvidia.com/aiq-blueprint/latest/
- MCP tools — https://docs.nvidia.com/aiq-blueprint/latest/customization/mcp-tools.html
- Observability — https://docs.nvidia.com/aiq-blueprint/latest/deployment/observability.html
- AI-Q GitHub — https://github.com/NVIDIA-AI-Blueprints/aiq (Apache-2.0)
- NVIDIA skills catalog — https://github.com/NVIDIA/skills
- "Add a Deep Research Skill" blog (2026-05-20) — https://developer.nvidia.com/blog/add-a-specialized-deep-research-skill-to-agent-harnesses/
- "Verified Agent Skills" governance blog (2026-05-19) — https://developer.nvidia.com/blog/nvidia-verified-agent-skills-provide-capability-governance-for-ai-agents/
- Open spec — https://agentskills.io

**Key facts:**
- Skill = folder with required `SKILL.md` (YAML frontmatter: `name`, `version`, `description` minimum), optional `scripts/`, `references/`, `assets/`.
- NVIDIA additions: `skill-card.md` (trust metadata) + `skill.oms.sig` (OpenSSF Model Signing detached signature).
- **Progressive disclosure** invocation: agent boots loading only descriptions; full `SKILL.md` swaps in on match; bundled scripts run from there. **No JSON-Schema arg validation at the skill layer** — scripts do their own parsing.
- Discovery is **filesystem-based per harness** (`.claude/skills/`, `~/.config/opencode/skills/`, etc.). No central registry.
- Composition by shared env vars / handoff, not a workflow graph. Cross-skill dependencies are convention-only — **no `depends_on:` field** (flag).
- Built on Anthropic-originated agentskills.io spec, adopted by 30+ harnesses incl. Claude Code, Codex, Cursor, OpenCode, Gemini CLI.

### 3.4 Vision–language model stack
- Prismatic paper — https://arxiv.org/abs/2402.07865
- Prismatic code — https://github.com/TRI-ML/prismatic-vlms (MIT)
- Qwen3.6-35B-A3B upstream — https://huggingface.co/Qwen/Qwen3.6-35B-A3B (Apache-2.0, released 2026-04-16)
- Qwen3.6 GitHub — https://github.com/QwenLM/Qwen3.6
- Qwen3.6 release blog — https://qwen.ai/blog?id=qwen3.6-35b-a3b (note: JS-rendered, dates come from HF model card)
- NVFP4 quantized variant — https://huggingface.co/unsloth/Qwen3.6-35B-A3B-NVFP4 (Apache-2.0, UltraChat-calibrated, 16K ctx)
- NVIDIA NVFP4 blog — https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/
- Unsloth + Blackwell blog — https://developer.nvidia.com/blog/train-an-llm-on-an-nvidia-blackwell-desktop-with-unsloth-and-scale-it/
- Nemotron NVFP4-QAD report (Mar 2026) — https://research.nvidia.com/labs/nemotron/files/NVFP4-QAD-Report.pdf

**Architecture facts (Qwen3.6-35B-A3B, from HF card):**
- 35B total / 3B active; hidden 2048; 40 layers in hybrid `(3× Gated-DeltaNet→MoE + 1× Gated-Attention→MoE) × 10` layout.
- Attention: 16 Q heads, 2 KV heads (GQA), head dim 256; Gated-DeltaNet: 32 V / 16 QK heads, dim 128.
- 256 experts, 8 routed + 1 shared active per token; expert FFN dim 512; MTP enabled.
- 262K native context, ~1.01M via YaRN.
- Multimodal — ships a native vision tower with `--language-model-only` flag.

**NVFP4 facts:**
- 4-bit E2M1 in 16-element micro-blocks; per-block E4M3 FP8 scale; per-tensor FP32 secondary scale → ~4.5 bits/value effective.
- 3.5× smaller than FP16, 1.8× smaller than FP8.
- Requires Blackwell 5th-gen Tensor Cores (B100/B200, GB200/GB300 NVL72, RTX PRO 6000 Blackwell, RTX 50-series, DGX Spark GB10). **Hopper does NOT accelerate FP4 natively.**
- Serving stacks with mature NVFP4 paths as of Mar 2026: TensorRT-LLM ≥0.17, vLLM (dense + MoE), SGLang.
- Unsloth supports NVFP4 fine-tuning; up to 40B on a single Blackwell GPU.

**Adapter design references:**
- LLaVA-1.5 (MLP projector wins over linear) — https://arxiv.org/abs/2310.03744
- LLaVA-NeXT / haotian-liu/LLaVA — https://github.com/haotian-liu/LLaVA
- BLIP-2 Q-Former (token compression) — https://arxiv.org/abs/2301.12597
- Honeybee C-Abstractor / D-Abstractor (locality-preserving projector — directly relevant for satellite tiles) — https://arxiv.org/abs/2312.06742
- MoVA (multi-encoder mixture-of-vision-experts) — https://arxiv.org/abs/2404.13046
- MoME (mixture-of-modal-experts) — https://arxiv.org/abs/2407.12709
- MoE-LLaVA (VLM on MoE backbone precedent) — https://arxiv.org/abs/2401.15947
- AsyMoE (modality-aware expert allocation warnings) — https://arxiv.org/abs/2509.12715

**No published paper combines Prismatic with EO data** — this is a novel contribution, not a reproduction.

### 3.5 Self-improvement — Stanford CS329A
- Course site — https://cs329a.stanford.edu/
- Stanford Online mirror — https://online.stanford.edu/courses/cs329a-self-improving-ai-agents
- Mirhoseini lab — https://scalingintelligence.stanford.edu/
- Mirhoseini personal — https://www.azaliamirhoseini.com/

**Canonical papers we'll lean on (full ranking in §7):**
- Reflexion — https://arxiv.org/abs/2303.11366
- Self-Refine — https://arxiv.org/abs/2303.17651
- STaR — https://arxiv.org/abs/2203.14465
- Constitutional AI / RLAIF — https://arxiv.org/abs/2212.08073
- Lightman et al., "Let's Verify Step by Step" (process reward models) — https://arxiv.org/abs/2305.20050
- Math-Shepherd (PRM training) — https://arxiv.org/abs/2312.08935
- Large Language Monkeys (repeated sampling laws) — https://arxiv.org/abs/2407.21787
- Archon (inference-time search over pipelines) — https://arxiv.org/abs/2409.15254
- Snell et al., optimal test-time scaling — https://arxiv.org/abs/2408.03314
- LATS (tree search over reasoning) — https://arxiv.org/abs/2310.04406
- ADaPT (recursive decomposition) — https://arxiv.org/abs/2311.05772
- SWiRL (step-wise RL on synthetic tool-use trajectories) — https://arxiv.org/abs/2504.04736
- DeepSeekMath / GRPO — https://arxiv.org/abs/2402.03300
- DAPO — https://arxiv.org/abs/2503.14476
- Automated Design of Agentic Systems (ADAS) — https://arxiv.org/abs/2408.08435
- The AI Scientist — https://arxiv.org/abs/2408.06292
- AlphaEvolve — https://arxiv.org/abs/2506.13131
- ReAct (tool-use baseline) — https://arxiv.org/abs/2210.03629

**Evaluation references:**
- METR long-task time-horizon metric — https://arxiv.org/abs/2503.14499
- GDPVal (economically valuable real-world tasks) — https://openai.com/index/gdpval/
- KernelBench (verifier-grounded code agent eval) — https://arxiv.org/abs/2502.10517
- DeepScholar-Bench — referenced in CS329A Nov 17 lecture; arXiv ID **UNVERIFIED**

---

## 4. Layered architecture

### L0 — Runtime
- **Hardware target:** NVIDIA Blackwell. Dev box = single B200 (or RTX PRO 6000); production = GB200 NVL72 if we need throughput. Justified by NVFP4 hardware requirement; without Blackwell the quantization is software-emulated and loses its speed advantage.
- **Serving:** TensorRT-LLM ≥0.17 or vLLM (both have first-class NVFP4 MoE paths as of March 2026). SGLang is a fallback.
- **Training:** Unsloth — confirmed NVFP4 fine-tuning support per the NVIDIA + Unsloth blog. We freeze the NVFP4 base, train projectors + LoRA in BF16/FP16.
- **Orchestration:** Studio long-running predictions are async; we poll `GET /predictions/{id}` via `tenacity`-style backoff inside an `eo-job-await` middleware. **UNVERIFIED:** whether OlmoEarth Run emits webhook callbacks — if it does, the polling is a fallback.

### L1 — Model stack

Two parallel encoder streams feed one MoE language backbone:

1. **Prismatic vision tower** — fused DINOv2 + SigLIP per the Prismatic paper's design-space finding (fused > single, SigLIP > CLIP). RGB / pseudo-color Sentinel-2 / Landsat composites.
2. **OlmoEarth geo encoder** — `OlmoEarth-v1-Large` embeddings for raw multispectral / time-series stacks. This is where the platform's domain prior lives.

Both feed dedicated projectors. Defaults:
- **Vision → LLM:** 2-layer MLP+GELU per LLaVA-1.5. Patch tokens (256–1024).
- **Geo → LLM:** MLP first pass; upgrade to a Q-Former (BLIP-2-style) or MoVA gating block once we have enough paired data to train it. The Q-Former matters because OlmoEarth embeddings can be large (multispectral × time → many tokens) and compressing them with a learned querier outperforms linear projection at fixed budget.

Both projector outputs are concatenated as prefix tokens to the Qwen3.6-35B-A3B-NVFP4 input. Critical implementation notes:
- Disable Qwen3.6's native vision tower (`--language-model-only`) so the encoder choice stays under our control.
- LoRA adapters target both Gated-DeltaNet and Gated-Attention blocks (the hybrid layout means a single adapter family won't cover both).
- NVFP4 weights are frozen; only projectors + LoRA train. This is the supported Unsloth + Blackwell path.

**Expert allocation risk:** Qwen3.6 was pretrained on text/code/vision-text. Routing geo-embedding tokens through its 256 experts may concentrate on a small subset (per AsyMoE's warning). We log expert-usage histograms and consider modality-specific expert sets if collapse is observed.

### L2 — Agent harness

DeerFlow v2's pattern, ported:

- **Lead agent** built with `langchain.agents.create_agent` — no hand-wired StateGraph. Topology emerges from the middleware chain + tool list.
- **Subagents as tools.** Each subagent runs on an isolated event loop with its own context window. Built-ins to port:
  - `general-purpose` — multi-step exploration (DeerFlow has this).
  - `bash` — sandboxed shell (DeerFlow has this).
  - **New for EO:** `eo-tile-fetch` — Sentinel-2 / Landsat / NAIP window pulls via Planetary Computer or Earthdata. Heavy I/O, deserves an isolated context.
  - **New for self-improvement:** `critic` — Reflexion-style verbal reflection + a process-reward-model score (§7).
- **Middleware chain.** Port DeerFlow's 14 (summarization, todo, loop-detection, memory, clarification, safety, guardrail, …) and add Studio-specific ones:
  - `eo-dataset-validation` — equal-frequency binning, MIME-type sanity, sample_category / es_label / oe_labels schema checks (the 8 prep pitfalls the `olmoearth-data-prep` skill exists to prevent).
  - `spatial-cv-checker` — refuse random splits when AOIs are spatially auto-correlated.
  - `cost-guard` — refuse to launch a Studio fine-tune that the OpenAPI estimate puts above a session budget.
  - `eo-job-await` — async polling for `Predictions` long-running work.
- **Shared `ThreadState`** as DeerFlow's reducer-merged LangGraph state: `artifacts`, `todos`, `viewed_images`, `uploaded_files`, `sandbox`, plus EO-specific fields `aois`, `prediction_ids`, `studio_project_id`.

### L3 — Skill library

Adopt the agentskills.io spec verbatim. Each skill is a folder with:

```
<skill>/
├── SKILL.md            # YAML frontmatter (name, version, description) + Markdown body
├── scripts/            # invocable helpers, called from SKILL.md body
├── references/         # deep docs the agent pulls on demand
├── assets/             # templates, fixtures
├── skill-card.md       # trust metadata (NVIDIA-style: license, owner, known risks, refs)
└── skill.oms.sig       # OpenSSF Model Signing detached signature
```

Discovery: filesystem-based at `.olmoearth-agent/skills/` in the project root, plus a user-level `~/.olmoearth-agent/skills/`. Same boot pattern as Claude Code / Codex.

Initial skill set (each maps to a Studio workflow stage):

| Skill | When activated | Key scripts |
|---|---|---|
| `aoi-fetch` | "give me AOIs for X watershed / county / HUC" | NLDI / NHD / HUC lookups, GeoJSON output |
| `label-prep` | CSV/Shapefile/GeoJSON labels → Studio-importable | The 8-pitfall guard from existing [olmoearth-data-prep](C:\Users\Frank\.claude\skills\olmoearth-data-prep) skill |
| `studio-project` | "create a Studio project / dataset / labelset" | Wraps the Areas / Projects / Datasets / Labelsets API |
| `train-or-embed` | Picks FT vs embed+LP based on label volume + class balance | Generates rslearn / olmoearth_projects config; submits via `Predictions` |
| `evaluate` | Runs held-out metrics; flags spatial leakage; emits report | Uses Studio Annotations + tasks aggregates |
| `report` | Writes the human-readable Markdown/PDF | Combines artifacts |
| `cost-guard` | Estimates Studio fine-tune cost before submission | Uses OpenAPI schema sizes |
| `case-study-runbook-*` | Optional thin shells around Karst / Chesapeake / Potomac patterns | None — they delegate to the above |

The existing `olmoearth-data-prep` skill in `C:\Users\Frank\.claude\skills\` is the spiritual prototype for `label-prep`. We re-package it under the agentskills.io spec.

### L4 — Tool layer (MCP)

DeerFlow's MCP-first surface means every external system is a stdio-pooled MCP server. Initial servers:

| MCP server | Purpose | Notes |
|---|---|---|
| `olmoearth-studio` | Wraps `https://olmoearth.allenai.org/api/v1/` | Codegen from the live OpenAPI spec (`openapi.json`). Bearer auth header. |
| `planetary-computer` | Sentinel-2 / Landsat / NAIP STAC search + sign | Public STAC endpoint, no auth for queries |
| `earthdata` | NASA / NSIDC catalogs | Auth via Earthdata token |
| `nldi-nhd-huc` | Watershed AOI lookups | Built on USGS NLDI |
| `roger-studio` | Annotation export / import | If/when API stabilizes |
| `hf-hub` | Datasets + models pull/push | `huggingface_hub` |
| `geocoder` | Place name → bbox | Nominatim or equivalent |

Auth pattern per NVIDIA AI-Q: three modes — unauthenticated, service-account, per-user identity forwarding. Use service-account for Studio (single team API key) until we need per-user routing.

### L5 — Self-improvement loop

See §7 for the full design. Summary: three loops nested by latency.

### L6 — Eval + observability

- **Tracing:** OpenTelemetry exporter following the NVIDIA AI-Q pattern, with PII / coordinate redaction. Lat/lon and partner data are sensitive and must not appear in outbound traces.
- **Backends:** Phoenix or LangSmith locally; OTEL Collector to whatever bAI Labs lands on.
- **Eval datasets:** built from traces. The LangSmith "datasets from runs" pattern. Held-out OlmoEarth fine-tune metrics serve as outcome-level verifiers.
- **Benchmarks:** for the agent itself — METR long-task time-horizon (does an agent that took N hours yesterday take <N hours today?), KernelBench-style verifier-grounded scoring on the rslearn-config-writing subtask, and a custom DeepScholar-Bench-shaped EO research bench (to be built).

---

## 5. Data flow walkthrough

Walking through a representative call.

**Input:** *"Map alfalfa fields in the Klamath basin from 2022–2024 Sentinel-2. We have ~300 ground-truth points in this GeoJSON."*

1. **Clarification middleware** asks: *"Train your own classifier, or use OlmoEarth-v1 embeddings + a linear probe? Confidence on ground-truth point boundaries?"* — derived from CS329A's "ambiguity costs more than questions" framing.
2. **Lead agent** decomposes:
   - subagent: `aoi-fetch` → call NLDI MCP → return Klamath HUC-8 GeoJSON.
   - subagent: `label-prep` → run the 8-pitfall guard on the user's GeoJSON, normalize fields, write Studio-import-ready file.
   - `studio-project` skill → create Project, Area, Dataset, Labelset via Studio MCP (`POST /projects`, `POST /areas`, etc.).
   - `train-or-embed` skill → 300 points is below the FT threshold → pick embed+LP. Generate rslearn config. Submit via `POST /predictions`.
   - `eo-job-await` middleware polls `GET /predictions/{id}` until done.
   - `evaluate` skill → pull `PredictionResults` tile + vector outputs; cross-validate spatially; compute IoU.
   - `critic` subagent reads the eval and the trace; emits a process-reward score + a Reflexion-style verbal critique → stored in `artifacts['critique']` for the next session.
3. **Report skill** writes a Markdown report with embedded XYZ tile URLs from `prediction-results/tiles/{z}/{x}/{y}.png` and a confusion matrix.
4. **Cost-guard middleware** had refused the FT path at step 2 if the user had insisted — surfaces the estimate, asks again.
5. **OTEL exporter** writes the trace (redacted of lat/lon) to LangSmith. The trace becomes a training example for §7's loops.

---

## 6. The auto-research loop

Phrased as the agent's outer loop, not the human's:

```
while research_question_remains():
    plan = decompose(question)               # ReAct + ADaPT-style decomposition
    while plan.has_open_step():
        step = plan.next()
        candidate = sample_n(step, n=k)      # repeated sampling (Monkeys)
        verified  = verify(candidate)        # process reward + domain checks
        result    = pick_best(verified)
        plan.update(result)
        critic.reflect(step, result)         # Reflexion verbal critique
    summary = report(plan.artifacts)
    eval_score = score(summary, held_out)
    persist(trace, eval_score)               # feeds §7 train-time loops
```

Three patterns from CS329A make this work:

- **Repeated sampling + verifier (Monkeys, Archon).** Coverage scales with N when you have a domain verifier. Our verifiers: rslearn config schema validator, Studio API schema validator, spatial-CV-leakage detector, the `eo-dataset-validation` middleware.
- **Process reward models (Lightman, Math-Shepherd).** Score each step, not just the final outcome. Lets us assign credit to early errors (wrong label-prep) rather than blaming late ones (model didn't converge).
- **Reflexion verbal critique.** Cheap, no weight updates, per-session. Strong baseline before we invest in train-time loops.

---

## 7. Self-improvement design (CS329A-grounded)

Three loops, nested by latency. Order matters — start at L5a and only graduate when you have enough traces.

### L5a — Inference-time (per session, no weight updates)
- **Reflexion** ([arXiv:2303.11366](https://arxiv.org/abs/2303.11366)) — verbal self-critique into `artifacts['critique']`, replayed at the start of the next attempt.
- **Self-Refine** ([arXiv:2303.17651](https://arxiv.org/abs/2303.17651)) — same-model critique → revise on individual artifacts (rslearn configs, report drafts).
- **Repeated sampling + verifier** (Monkeys [arXiv:2407.21787](https://arxiv.org/abs/2407.21787); Archon [arXiv:2409.15254](https://arxiv.org/abs/2409.15254)) — k samples per step, verifier picks. Most leverage where a deterministic checker exists.
- **LATS / tree search** ([arXiv:2310.04406](https://arxiv.org/abs/2310.04406)) — only on subgoals with combinatorial structure (e.g. choosing a model + bands + temporal window combo).

### L5b — Train-time (across sessions, LoRA-level updates)
Once we have ≥1K trajectories with eval scores:
- **STaR** ([arXiv:2203.14465](https://arxiv.org/abs/2203.14465)) — fine-tune on the agent's own correct trajectories (filtered by verifier score). Cheap, well-understood.
- **SWiRL** ([arXiv:2504.04736](https://arxiv.org/abs/2504.04736)) — step-wise RL on synthetic / replayed multi-step tool-use trajectories. Directly designed for chained tool calls like Studio workflows.
- **GRPO / DAPO** ([arXiv:2402.03300](https://arxiv.org/abs/2402.03300), [arXiv:2503.14476](https://arxiv.org/abs/2503.14476)) — if/when we have a reward model worth optimizing against. Constitutional-AI principles ([arXiv:2212.08073](https://arxiv.org/abs/2212.08073)) seed the reward model for domain guardrails (don't leak coordinates; respect license terms on EO data).

### L5c — Open-ended (months, agent-designing-agent)
Aspirational; only pursue once the train-time loop has plateaued:
- **ADAS** ([arXiv:2408.08435](https://arxiv.org/abs/2408.08435)) — meta-agent searches over agent topologies. Could discover better skill orderings.
- **AlphaEvolve** ([arXiv:2506.13131](https://arxiv.org/abs/2506.13131)) — evolutionary search over rslearn configs / model recipes against held-out metrics.

### Evaluation
- **METR long-task horizon** ([arXiv:2503.14499](https://arxiv.org/abs/2503.14499)) — does an agent that took N hours last month take less now?
- **Custom EO research bench** — DeepScholar-Bench-shaped; ~20 frozen research questions with ground-truth artifacts. Run weekly, dashboard the trend.
- **Per-skill metrics** from the NVIDIA verified-skills blog: trigger accuracy, task completion rate, token efficiency. Tracked per-skill in OTEL.
- **Held-out OlmoEarth fine-tune metrics** as outcome-level verifier. Train on a fixed labeled split, freeze a held-out tile set, measure IoU.

---

## 8. Roadmap

| Phase | Duration | Deliverables | Exit criteria |
|---|---|---|---|
| **P0 — Scaffolding** | 2 weeks | Repo layout, `olmoearth-studio` MCP server codegen'd from OpenAPI, Bearer-auth handshake working, `OlmoEarth-v1-Large` loading via HF | One round-trip: `POST /projects` → `GET /projects/{id}` via the MCP from the agent |
| **P1 — Harness MVP** | 3 weeks | Port DeerFlow v2 lead-agent + 14 middlewares; add `eo-dataset-validation`, `cost-guard`, `eo-job-await`; ship `label-prep` skill (rewrap of `olmoearth-data-prep`) | Agent runs the Klamath walkthrough (§5) end-to-end with human approval gates |
| **P2 — Model stack** | 4 weeks | Prismatic vision tower wiring, MLP projector, LoRA adapters on Qwen3.6-35B-A3B-NVFP4 via Unsloth on B200; native vision tower disabled | Adapters train without OOM on B200; inference runs through TRT-LLM or vLLM |
| **P3 — Geo stream** | 3 weeks | OlmoEarth-v1 embedding projector; expert-usage logging; if collapse observed, modality-specific expert allocation | Held-out EO eval shows non-degraded vs. vision-only baseline |
| **P4 — Self-improvement L5a** | 2 weeks | `critic` subagent, Reflexion + Self-Refine, repeated sampling + verifier on rslearn config writing | Pass rate on rslearn-config generation up ≥10pp vs. single-sample |
| **P5 — Eval + observability** | 2 weeks | OTEL with PII/coord redaction; LangSmith dataset from traces; custom EO research bench v0 (20 questions) | Weekly trendline dashboard live |
| **P6 — Self-improvement L5b** | 4 weeks (only if traces accumulate) | STaR fine-tune on filtered traces; SWiRL on tool-use chains | METR-horizon metric improves on the EO bench |
| **P7 — Open-ended L5c** | Open-ended | ADAS / AlphaEvolve experiments | Discovered improvement holds out-of-sample |

P0–P3 is the buildable core. P4–P5 is where the agent starts learning. P6–P7 is aspirational and depends on P5's trace volume.

---

## 9. Open questions / unverified gaps

These are items where the public record is incomplete and we need a private check, an experiment, or a partner conversation:

1. **OlmoEarth Studio rate limits, payload caps, webhooks.** Not documented. Action: ask Ai2 contact; until then, polling-only.
2. **OlmoEarth Studio `/models` resource.** No public endpoint for naming a fine-tuned model when creating a Prediction. Action: dive into `openapi.json` schemas; confirm whether `Prediction.model_id` references a model the user trained or is implicit in the project.
3. **Qwen3.6-35B-A3B native vision tower's exact architecture.** Card mentions multimodality but not the encoder details. Action: read `config.json` on HF before committing to a Prismatic-only design.
4. **Prismatic with EO data.** No published precedent. Action: a small pilot with our existing labels before committing the full architecture.
5. **Qwen3.6 release blog scraping.** JS-rendered page; the release date (2026-04-16) came from HF card. Action: re-fetch in a browser before citing externally.
6. **DeepScholar-Bench paper ID.** Referenced in CS329A but I couldn't resolve a direct arXiv link. Action: WebSearch when CS329A publishes the Nov 17 reading list.
7. **NVIDIA `docs.nvidia.com/skills` standalone site.** Referenced from the skills README but returned 404. Action: re-check.
8. **NVIDIA verified-skills repo license.** GitHub returns `Other`; README says `Apache-2.0 AND CC-BY-4.0`. Confirm with NVIDIA legal before redistributing skill cards.
9. **Expert-collapse risk on Qwen3.6 MoE with geo tokens.** Empirical. Action: log expert-usage histograms in P3 and react if needed.

---

## 10. What this plan does NOT include (intentional)

- **A Python SDK for the Studio API.** We codegen from `openapi.json` into the MCP server rather than maintaining a hand-written SDK.
- **A custom training framework.** We use Unsloth for fine-tuning + LangGraph for orchestration. Both are well-supported in 2026 on Blackwell + NVFP4.
- **Domain-specialized weights for Karst / Chesapeake / Potomac.** Per scope decision, this is a generic agent. Case-study runbooks are thin shells.
- **A custom evaluation framework.** We reuse METR / GDPVal / KernelBench shapes and build a small EO research bench on top.
- **An agent UI.** Lead-agent input is CLI + Studio Web UI integration; we do not ship a separate frontend.

---

## 11. Verified-link index (one place)

Pasted here for ease of grepping later. Same content as §3, deduplicated and grouped.

**OlmoEarth Studio**
https://docs.olmoearth.allenai.org/ · https://docs.olmoearth.allenai.org/authentication/ · https://docs.olmoearth.allenai.org/api/ · https://olmoearth.allenai.org/api/v1/openapi.json · https://allenai.org/blog/olmoearth · https://allenai.org/blog/olmoearth-v1-1 · https://huggingface.co/allenai/OlmoEarth-v1-Large · https://github.com/allenai/olmoearth_projects · https://github.com/allenai/olmoearth_pretrain

**DeerFlow v2**
https://github.com/bytedance/deer-flow · https://deerflow.tech/ · https://github.com/bytedance/deer-flow/blob/main/backend/docs/ARCHITECTURE.md · https://github.com/bytedance/deer-flow/blob/main/backend/langgraph.json · https://github.com/bytedance/deer-flow/blob/main/backend/packages/harness/deerflow/agents/factory.py · https://github.com/bytedance/deer-flow/blob/main/backend/packages/harness/deerflow/agents/lead_agent/agent.py · https://github.com/bytedance/deer-flow/blob/main/backend/packages/harness/deerflow/subagents/executor.py · https://github.com/bytedance/deer-flow/blob/main/backend/packages/harness/deerflow/tools/builtins/task_tool.py · https://github.com/bytedance/deer-flow/blob/main/backend/packages/harness/deerflow/mcp/tools.py · https://github.com/bytedance/deer-flow/blob/main/backend/docs/MCP_SERVER.md · https://github.com/bytedance/deer-flow/blob/main/backend/packages/harness/deerflow/agents/thread_state.py · https://github.com/bytedance/deer-flow/blob/main/backend/packages/harness/deerflow/skills/installer.py · https://github.com/bytedance/deer-flow/blob/main/backend/packages/harness/deerflow/skills/tool_policy.py

**NVIDIA AI-Q + agentskills.io**
https://docs.nvidia.com/aiq-blueprint/latest/integration/agent-skills.html · https://docs.nvidia.com/aiq-blueprint/latest/ · https://docs.nvidia.com/aiq-blueprint/latest/customization/mcp-tools.html · https://docs.nvidia.com/aiq-blueprint/latest/deployment/observability.html · https://github.com/NVIDIA-AI-Blueprints/aiq · https://github.com/NVIDIA/skills · https://developer.nvidia.com/blog/add-a-specialized-deep-research-skill-to-agent-harnesses/ · https://developer.nvidia.com/blog/nvidia-verified-agent-skills-provide-capability-governance-for-ai-agents/ · https://agentskills.io

**Vision–language stack**
https://arxiv.org/abs/2402.07865 · https://github.com/TRI-ML/prismatic-vlms · https://huggingface.co/Qwen/Qwen3.6-35B-A3B · https://github.com/QwenLM/Qwen3.6 · https://qwen.ai/blog?id=qwen3.6-35b-a3b · https://huggingface.co/unsloth/Qwen3.6-35B-A3B-NVFP4 · https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/ · https://developer.nvidia.com/blog/train-an-llm-on-an-nvidia-blackwell-desktop-with-unsloth-and-scale-it/ · https://research.nvidia.com/labs/nemotron/files/NVFP4-QAD-Report.pdf · https://arxiv.org/abs/2310.03744 · https://github.com/haotian-liu/LLaVA · https://arxiv.org/abs/2301.12597 · https://arxiv.org/abs/2312.06742 · https://arxiv.org/abs/2404.13046 · https://arxiv.org/abs/2407.12709 · https://arxiv.org/abs/2401.15947 · https://arxiv.org/abs/2509.12715

**Stanford CS329A + papers**
https://cs329a.stanford.edu/ · https://online.stanford.edu/courses/cs329a-self-improving-ai-agents · https://scalingintelligence.stanford.edu/ · https://www.azaliamirhoseini.com/ · https://arxiv.org/abs/2303.11366 · https://arxiv.org/abs/2303.17651 · https://arxiv.org/abs/2203.14465 · https://arxiv.org/abs/2212.08073 · https://arxiv.org/abs/2305.20050 · https://arxiv.org/abs/2312.08935 · https://arxiv.org/abs/2407.21787 · https://arxiv.org/abs/2409.15254 · https://arxiv.org/abs/2408.03314 · https://arxiv.org/abs/2310.04406 · https://arxiv.org/abs/2311.05772 · https://arxiv.org/abs/2504.04736 · https://arxiv.org/abs/2402.03300 · https://arxiv.org/abs/2503.14476 · https://arxiv.org/abs/2408.08435 · https://arxiv.org/abs/2408.06292 · https://arxiv.org/abs/2506.13131 · https://arxiv.org/abs/2210.03629 · https://arxiv.org/abs/2503.14499 · https://openai.com/index/gdpval/ · https://arxiv.org/abs/2502.10517

---

*End of v0.1 — next pass should resolve the §9 unverified items and produce a P0 sprint plan.*
