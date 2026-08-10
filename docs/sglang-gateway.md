# SGLang Model Gateway evaluation

## Decision

SGLang Model Gateway is useful as an **optional serving Adapter** for a hosted
OpenAI-compatible model. It is not an agent-harness replacement and is not the
OlmoEarth Agent runtime baseline.

Keep the in-repo lead loop, typed `ToolRegistry`, skills, provenance, and
geospatial policy authoritative. Use the gateway only when centralized retries,
request limits, circuit breaking, or Prometheus/OpenTelemetry telemetry justify
another process between the agent and its model provider.

The default direct provider path remains simpler for one user and one hosted
model.

## Evidence

The evaluation used SGLang Model Gateway `v0.3.2`, the version in SGLang
`v0.5.17` and the current versioned official gateway image as of 2026-08-10.
A CPU-only Docker smoke test against a disposable OpenAI-compatible upstream
verified that the gateway:

- retried a first `503` response and returned the second successful response;
- forwarded the caller's bearer authorization to the upstream;
- preserved `/v1/chat/completions` and a native function-tool response;
- exposed Prometheus metrics; and
- worked with the existing `OlmoEarthLLM` client without code changes.

The container had no Docker GPU device request. No local model or local GPU was
used.

## What it helps

- A shared hosted-model endpoint that needs bounded retries and circuit breaking.
- Multi-user or bursty traffic that needs a queue and concurrency controls.
- Central model-call metrics, tracing, and request correlation.
- A future fleet of multiple inference workers where routing becomes real work.

## What it does not help

- It does not reduce the 37-tool schema catalog sent by the current harness.
- It does not make answers more concise or improve skill selection.
- Its agent-aware KV-cache work applies to SGLang inference workers, not a hosted
  NVIDIA NIM endpoint whose KV cache is controlled by NVIDIA.
- Its MCP loop and conversation storage duplicate the OlmoEarth lead loop and
  state ownership. They are deliberately disabled here.

OpenClaw-inspired capability projection and context accounting remain separate
harness changes. SGLang belongs below that Interface, at the model-serving seam.

## Start the CPU gateway

The compose file defaults to NVIDIA's hosted OpenAI-compatible base URL. Override
`SGLANG_UPSTREAM_URL` for another compatible provider. The URL must omit the
trailing `/v1` because the incoming request path is forwarded unchanged.

```bash
docker compose -f docker/sglang-gateway.compose.yml up -d
```

Point the existing client at the local gateway and keep the real provider key in
the client process. Compose does not inject or persist the key. The gateway sees
the bearer header in transit because it must forward that header to the provider.

```bash
export LLM_ENDPOINT=http://127.0.0.1:30000/v1
export LLM_MODEL=nvidia/nemotron-3-nano-30b-a3b
export LLM_API_KEY="$NVIDIA_API_KEY"

uv run olmoearth-agent "How many Studio projects do I have?"
```

PowerShell equivalent:

```powershell
$env:LLM_ENDPOINT = "http://127.0.0.1:30000/v1"
$env:LLM_MODEL = "nvidia/nemotron-3-nano-30b-a3b"
$env:LLM_API_KEY = $env:NVIDIA_API_KEY

uv run olmoearth-agent "How many Studio projects do I have?"
```

The gateway listens only on host loopback:

- OpenAI-compatible API: `http://127.0.0.1:30000/v1`
- Prometheus metrics: `http://127.0.0.1:29000/metrics`

## Operational boundary

The evaluation profile deliberately sets:

- `--backend openai`: proxy a hosted OpenAI-compatible provider;
- `--history-backend none`: the gateway is not the conversation authority;
- no `--mcp-config-path`: domain tools stay in `ToolRegistry`;
- two retry attempts instead of the gateway default of five; and
- a digest-pinned image, loopback-only ports, a read-only filesystem, dropped
  Linux capabilities, and `no-new-privileges`.

Retries can duplicate provider billing if the provider completed a request but
its response was lost. Set `SGLANG_GATEWAY_RETRIES=1` to disable retrying model
requests.

Stop and remove the gateway with:

```bash
docker compose -f docker/sglang-gateway.compose.yml down
```

## Adoption trigger

Promote this from evaluation-only when measurements show at least one of:

1. repeated transient provider failures that client retries do not handle;
2. sustained multi-user concurrency requiring a shared queue or rate limit;
3. a second model worker requiring routing; or
4. an operational requirement for centralized model-call telemetry.

Until then, use the direct hosted-provider path.

## Primary sources

- [SGLang Model Gateway documentation](https://github.com/sgl-project/sglang/blob/v0.5.17/docs/advanced_features/sgl_model_gateway.md)
- [SGLang Model Gateway source README](https://github.com/sgl-project/sglang/blob/v0.5.17/sgl-model-gateway/README.md)
- [SGLang v0.5.17 release](https://github.com/sgl-project/sglang/releases/tag/v0.5.17)
- [Official gateway image tags](https://hub.docker.com/r/lmsysorg/sgl-model-gateway/tags)
