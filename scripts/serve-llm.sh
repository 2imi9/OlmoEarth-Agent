#!/usr/bin/env bash
# Bring up the OlmoEarth Agent's LLM backend (4-bit GGUF via llama.cpp) and
# block until it answers on /health. Idempotent: re-running just re-attaches
# and re-checks. Canonical values: docs/CANON.md (C1 model, C3 server).
set -euo pipefail

COMPOSE_FILE="docker/llama.compose.yml"
HEALTH_URL="http://localhost:8000/health"
MODEL_ID="unsloth/Qwen3.6-35B-A3B-GGUF:UD-IQ4_XS"
MAX_WAIT=600   # seconds to wait for the model to load
SLEEP=5        # seconds between health polls

echo "==> Starting the LLM (llama.cpp, 4-bit GGUF) via ${COMPOSE_FILE}"
docker compose -f "${COMPOSE_FILE}" up -d

echo "==> Waiting for ${HEALTH_URL} (up to ${MAX_WAIT}s; first load pulls ~17.7 GB)"
elapsed=0
until curl -fsS "${HEALTH_URL}" >/dev/null 2>&1; do
  if [ "${elapsed}" -ge "${MAX_WAIT}" ]; then
    echo "!! LLM did not become healthy within ${MAX_WAIT}s." >&2
    echo "   Check logs: docker compose -f ${COMPOSE_FILE} logs -f" >&2
    exit 1
  fi
  printf '   ... not ready yet (%ds elapsed)\n' "${elapsed}"
  sleep "${SLEEP}"
  elapsed=$((elapsed + SLEEP))
done

echo "==> LLM is healthy."
echo "    Served model : ${MODEL_ID}"
echo "    OpenAI base  : http://localhost:8000/v1"
echo "    Point the agent at it with: export LLM_ENDPOINT=http://localhost:8000/v1"
