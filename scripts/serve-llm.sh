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

# Reuse the host Hugging Face cache so the ~17.7 GB GGUF downloads once.
# The compose file bind-mounts ${HF_CACHE_DIR}; resolve it to an absolute,
# Docker-Desktop-acceptable path here so a fresh clone needs no env vars.
HF_CACHE_DIR="${HF_CACHE_DIR:-${HF_HOME:-${HOME}/.cache/huggingface}}"
# On Windows (Git Bash/MSYS) convert /c/Users/... -> C:/Users/... ; Docker
# Desktop's mounter rejects the MSYS form. cygpath exists only on MSYS, so
# this is a no-op on Linux/macOS.
if command -v cygpath >/dev/null 2>&1; then
  HF_CACHE_DIR="$(cygpath -m "${HF_CACHE_DIR}")"
fi
export HF_CACHE_DIR
mkdir -p "${HF_CACHE_DIR}" 2>/dev/null || true
# Optional first-run proxy (empty => direct). Must target host.docker.internal,
# since 127.0.0.1 inside the container is the container itself.
export HF_PROXY="${HF_PROXY:-}"

echo "==> Starting the LLM (llama.cpp, 4-bit GGUF) via ${COMPOSE_FILE}"
echo "    HF cache (host, bind-mounted): ${HF_CACHE_DIR}"
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
