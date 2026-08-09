# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License

FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim@sha256:e5b65587bce7de595f299855d7385fe7fca39b8a74baa261ba1b7147afa78e58 AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

# Cache the locked runtime dependencies separately from project source edits.
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN uv sync --frozen --no-dev --extra serve --extra claude --no-install-project

COPY src ./src

# A non-editable install lets the final image copy only the completed virtual
# environment. The Claude extra makes every web UI backend available.
RUN uv sync --frozen --no-dev --extra serve --extra claude --no-editable


FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim@sha256:e5b65587bce7de595f299855d7385fe7fca39b8a74baa261ba1b7147afa78e58 AS runtime

ENV HOME=/tmp \
    OLMOEARTH_EGRESS=enforce \
    OLMOEARTH_OUTPUT_ROOT=/app/olmoearth_outputs \
    OLMOEARTH_RUN_PYTHON=0 \
    OLMOEARTH_SKILLS_DIR=/app/vendor/olmoearth-skills/skills \
    OLMOEARTH_WEBUI_DIR=/app/webui \
    PATH=/app/.venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY --from=builder /app/.venv ./.venv
RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin olmoearth \
    && mkdir -p /app/olmoearth_outputs \
    && chown -R olmoearth:olmoearth /app/olmoearth_outputs

COPY --chown=olmoearth:olmoearth webui ./webui
COPY --chown=olmoearth:olmoearth vendor/olmoearth-skills/skills \
    ./vendor/olmoearth-skills/skills

USER olmoearth

EXPOSE 8088

HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=6 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8088/api/health', timeout=2).close()"]

CMD ["olmoearth-agent-serve", "--host", "0.0.0.0", "--port", "8088"]
