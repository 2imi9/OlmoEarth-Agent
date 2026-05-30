#!/usr/bin/env bash
# One-shot quickstart for the OlmoEarth Agent: init vendored skills, sync the
# Python env, and bring up the local LLM. After this finishes, set your Studio
# key and run a brief. Canonical values: docs/CANON.md.
set -euo pipefail

echo "############################################"
echo "# OlmoEarth Agent — quickstart"
echo "############################################"

echo ""
echo "==> [1/3] Initializing vendored skills (submodule #1-#4)"
git submodule update --init

echo ""
echo "==> [2/3] Syncing the Python environment (uv sync --all-extras)"
uv sync --all-extras

echo ""
echo "==> [3/3] Bringing up the local LLM (llama.cpp, 4-bit GGUF)"
./scripts/serve-llm.sh

echo ""
echo "############################################"
echo "# Setup complete. Run these next:"
echo "############################################"
echo ""
echo "  # 1. Point the agent at your OlmoEarth Studio account:"
echo "  export OLMOEARTH_API_KEY=...        # Studio UI -> profile -> API Keys"
echo ""
echo "  # 2. Run a brief:"
echo "  uv run olmoearth-agent \"How many OlmoEarth Studio projects do I have?\""
echo ""
echo "  # 3. (optional) Open the demo web UI on http://localhost:8080:"
echo "  make web"
echo ""
