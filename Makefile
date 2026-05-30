# OlmoEarth Agent — preset quickstart targets.
# Run on Git Bash / WSL / Linux / macOS. GNU make; recipes use real TABs.
# Canonical values live in docs/CANON.md — keep these consistent with it.

# Default sample brief for `make agent` — override with: make agent Q="..."
Q ?= How many OlmoEarth Studio projects do I have?

# Compose file for the local LLM backend (4-bit GGUF via llama.cpp).
COMPOSE := docker/llama.compose.yml

# Web UI dev-server port. The LLM owns 8000, so the UI gets 8080.
WEB_PORT := 8080

.PHONY: help setup serve down agent web up

help: ## List the available targets.
	@echo "OlmoEarth Agent — make targets:"
	@echo "  make setup   - init submodules (vendored skills #1-#4) + uv sync --all-extras"
	@echo "  make serve   - start the llama.cpp LLM and wait for it to be healthy"
	@echo "  make down    - stop the LLM"
	@echo "  make agent   - run a brief: make agent Q=\"<your brief>\""
	@echo "  make web     - serve the web UI on http://localhost:$(WEB_PORT)"
	@echo "  make up      - setup + serve (one-shot bring-up)"
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

web: ## Serve the static web UI (port 8080 — the LLM owns 8000).
	@echo "Serving the OlmoEarth Agent web UI at http://localhost:$(WEB_PORT)"
	python -m http.server $(WEB_PORT) --directory webui

up: setup serve ## One-shot bring-up: setup, then serve, then print next steps.
	@echo ""
	@echo "LLM is up and healthy. Next steps:"
	@echo "  1. export OLMOEARTH_API_KEY=...   # Studio UI -> profile -> API Keys"
	@echo "  2. make agent Q=\"How many OlmoEarth Studio projects do I have?\""
	@echo "  3. make web                       # optional: the demo web UI"
