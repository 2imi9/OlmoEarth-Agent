# OlmoEarth Agent: preset quickstart targets.
# Run on Git Bash / WSL / Linux / macOS. GNU make; recipes use real TABs.
# Canonical values live in docs/CANON.md. Keep these consistent with it.

# Default sample brief for `make agent`, override with: make agent Q="..."
Q ?= How many OlmoEarth Studio projects do I have?

# Compose file for the local LLM backend (4-bit GGUF via llama.cpp).
COMPOSE := docker/llama.compose.yml

# Web UI dev-server port. The LLM owns 8000, so the UI gets 8080.
WEB_PORT := 8080

# Live-agent bridge (FastAPI) port: the same UI wired to your real Studio account.
BRIDGE_PORT := 8088

.PHONY: help setup serve down agent web bridge up

help: ## List the available targets.
	@echo "OlmoEarth Agent make targets:"
	@echo "  make setup   - init submodules (vendored skills #1-#4) + uv sync --all-extras"
	@echo "  make up      - LOCAL one-command bring-up: setup + serve the LLM + live UI"
	@echo "  make serve   - start the llama.cpp LLM and wait for it to be healthy"
	@echo "  make down    - stop the LLM"
	@echo "  make agent   - run a brief: make agent Q=\"<your brief>\""
	@echo "  make web     - serve the static DEMO web UI on http://localhost:$(WEB_PORT) (no backend)"
	@echo "  make bridge  - serve the LIVE web UI on http://localhost:$(BRIDGE_PORT) (your Studio account)"
	@echo ""
	@echo "Two ways to run the live UI:"
	@echo "  - Local model (offline): make up        # auto-starts the LLM, then the UI"
	@echo "  - Cloud API (no download): make setup && make bridge"
	@echo "      then pick a provider (Claude/ChatGPT/Gemini) in the UI and paste a key"
	@echo ""
	@echo "Default brief (make agent): $(Q)"

setup: ## Init vendored skills + sync the Python env.
	git submodule update --init
	uv sync --all-extras

serve: ## Bring up the LLM and block until /health is green.
	./scripts/serve-llm.sh

down: ## Stop the LLM.
	docker compose -f $(COMPOSE) down

agent: ## Run a single brief through the agent (override with Q="...").
	uv run olmoearth-agent "$(Q)"

web: ## Serve the static DEMO web UI (no backend; port 8080, the LLM owns 8000).
	@echo "Serving the static DEMO web UI at http://localhost:$(WEB_PORT) (sample data - run 'make bridge' for the live agent)"
	python -m http.server $(WEB_PORT) --directory webui

bridge: ## Serve the LIVE web UI (port 8088). Local LLM optional: pick a cloud provider in the UI to skip 'make serve'.
	@echo "Serving the LIVE web UI at http://localhost:$(BRIDGE_PORT) - open it, paste your Studio key, send a brief"
	@echo "  - Local model? Bring it up first: 'make serve' (or 'make up' to do both in one command)."
	@echo "  - No local model / no 17.7 GB download? Open the UI, go to Settings -> LLM backend,"
	@echo "    pick a cloud provider (Claude / ChatGPT / Gemini) and paste a key. The UI nudges you if the local model is down."
	uv run olmoearth-agent-serve --port $(BRIDGE_PORT)

up: setup serve ## LOCAL one-command bring-up: setup, start the LLM, then serve the live UI.
	@echo ""
	@echo "==> LLM is up and healthy. Starting the live web UI on http://localhost:$(BRIDGE_PORT) ..."
	@echo "    (Ctrl-C stops the UI; the LLM keeps running - 'make down' to stop it.)"
	@echo "    Prefer a cloud API instead? Stop here and run 'make bridge' (no local model needed)."
	$(MAKE) bridge
