# Agent runtime baseline

**Decision:** keep the in-repo typed harness, modeled after **DeerFlow v2**.
Adopt selected **OpenClaw** workspace patterns; do not adopt OpenClaw as the
runtime. Use **Shippy** as the reliability bar for domain-agent behavior.

## What each reference contributes

| Reference | We adopt | We do not adopt now |
|---|---|---|
| [DeerFlow v2](https://github.com/bytedance/deer-flow) | Lead-agent loop, ordered middleware seams, bounded turns, tool-output externalization, context management, subagents only when isolation or parallelism has measured value | LangGraph and subagents before a real workload requires them |
| [OpenClaw](https://github.com/openclaw/openclaw) | Versioned `soul.md`, Agent Skills-compatible markdown, config/env eligibility, workspace-local state, explicit capability allowlists | Always-on gateway, messaging channels, host shell, public skill marketplace, and its broad personal-assistant runtime |
| Shippy | Typed domain API, deterministic tools, explicit behavioral limits, artifacts instead of huge context payloads, and eval-gated releases | Maritime-specific workflows or infrastructure |

## Why not switch to OpenClaw

OlmoEarth Agent is a narrow Studio operator, not a general personal assistant.
Its strongest properties are its typed `StudioClient`, deterministic
`ToolRegistry`, geospatial output rules, provenance log, and EO-specific tests.
Replacing that runtime would add a gateway, session/channel machinery, dynamic
execution surfaces, and a larger skill supply-chain boundary without improving
the core Studio workflow.

OpenClaw remains a useful pattern source. Its official skills documentation
describes load-time filtering by config, environment, and binary presence, plus
per-agent allowlists. Those ideas fit this repo and should be adopted behind the
existing registry rather than by replacing it:
https://github.com/openclaw/openclaw/blob/main/docs/tools/skills.md

## Current runtime shape

```text
brief
  -> soul + eligible skill index + memory + response policy
  -> lead-agent tool loop (bounded turns)
  -> schema-validated deterministic tool dispatch
  -> compact/spilled tool result
  -> concise final answer
  -> provenance + eval artifacts
```

The response policy is a middleware-style seam, not another persona rule:

- concise by default for local and hosted models;
- explicit detailed requests get a larger budget;
- every agent turn has a hard output-token ceiling;
- an overlong final answer gets one tools-disabled, lossless editing pass;
- the rewrite is rejected if machine-checkable statuses, ids, paths, URLs, or
  numbers change;
- editing failure returns the original successful answer.

## Remote evaluation model

Hosted NVIDIA NIM is an evaluation target, not the runtime baseline. The
credential-gated smoke test uses `nvidia/nemotron-3-nano-30b-a3b` because
NVIDIA documents it for reasoning, instruction following, and tool calling.
It calls `https://integrate.api.nvidia.com/v1` directly and never starts or
falls back to a local GPU model. See `docs/serving.md` for the command.

## Reconsider the runtime only when

Re-evaluate OpenClaw or a fuller DeerFlow integration when at least one real
requirement appears: multiple messaging channels, unattended scheduled work,
per-user isolated workspaces, dynamic third-party plugin installation, or
parallel subagents with measured latency/context benefits. Framework adoption
is not itself an agent-quality improvement.
