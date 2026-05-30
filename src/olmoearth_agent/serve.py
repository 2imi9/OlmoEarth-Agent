# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 OlmoEarth Agent contributors
"""A thin HTTP bridge that puts the live agent behind the ``webui/`` shell.

This is the seam the ``webui`` README calls for: a small FastAPI app that
serves the static front-end *and* exposes two JSON/SSE endpoints so a
browser brief runs the real :class:`~olmoearth_agent.harness.LeadAgent`
instead of the canned demo.

Endpoints
---------
``GET  /api/health``   liveness + which LLM/Studio the bridge is wired to.
``GET  /api/projects`` the caller's real Studio projects (bring-your-own-key).
``POST /api/run``      stream a brief's agent loop as Server-Sent Events.

The caller's Studio API key rides in the ``X-Olmoearth-Key`` header (or
``Authorization: Bearer``); it is forwarded to Studio per request and
never stored or logged. The LLM backbone is the env-configured
(``LLM_*``) endpoint from ``docs/serving.md`` — bring it up with
``scripts/serve-llm.sh`` first.

Run it::

    uv run olmoearth-agent-serve            # 127.0.0.1:8088, serves webui/
    # then open http://127.0.0.1:8088

Requires the ``serve`` extra (``uv sync --extra serve``).
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from olmoearth_agent.harness import LeadAgent, ThreadState
from olmoearth_agent.llm import OlmoEarthLLM
from olmoearth_agent.skills import SkillLoader, build_default_registry
from olmoearth_agent.studio import StudioClient
from olmoearth_agent.studio.client import DEFAULT_BASE_URL, StudioConfig

#: Hard cap on agent round-trips a single browser request may trigger.
_MAX_TURNS_CEILING = 12


def _webui_dir() -> Path:
    """Locate the static ``webui/`` directory (env override, else repo root)."""
    override = os.environ.get("OLMOEARTH_WEBUI_DIR")
    if override:
        return Path(override)
    # src/olmoearth_agent/serve.py -> parents[2] is the repo root.
    return Path(__file__).resolve().parents[2] / "webui"


def _studio_base() -> str:
    """Studio API base URL (env override, else the canonical default)."""
    return os.environ.get("OLMOEARTH_BASE_URL", DEFAULT_BASE_URL)


def _studio_key(request: Request) -> str:
    """Pull the caller's Studio key from the request headers.

    Prefers ``X-Olmoearth-Key``; falls back to ``Authorization: Bearer``.
    Returns an empty string when absent.
    """
    key = request.headers.get("x-olmoearth-key", "").strip()
    if not key:
        auth = request.headers.get("authorization", "")
        if auth.lower().startswith("bearer "):
            key = auth[7:].strip()
    return key


def _sse(event: dict[str, Any]) -> str:
    """Format one event dict as a Server-Sent Events ``data:`` frame."""
    return f"data: {json.dumps(event)}\n\n"


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Build the shared, request-independent agent pieces once at startup.

    The LLM client and tool registry are stateless across requests; only
    the Studio client (carries the caller's key) and ``ThreadState`` are
    per-request. Constructing the LLM client opens no socket, so this
    succeeds even when the LLM server is not up yet.
    """
    app.state.llm = OlmoEarthLLM()
    app.state.registry = build_default_registry()
    app.state.skill_index = SkillLoader().index()
    yield


app = FastAPI(title="OlmoEarth Agent bridge", lifespan=_lifespan)


@app.get("/api/health")
async def api_health() -> dict[str, Any]:
    """Liveness probe; the front-end uses it to switch from demo to live."""
    llm: OlmoEarthLLM = app.state.llm
    return {
        "ok": True,
        "mode": "live",
        "llm_endpoint": llm.config.endpoint,
        "llm_model": llm.config.model,
        "studio_base": _studio_base(),
    }


@app.get("/api/projects")
async def api_projects(request: Request) -> dict[str, Any]:
    """Return the caller's real Studio projects via ``load_context``."""
    key = _studio_key(request)
    if not key:
        raise HTTPException(status_code=400, detail="missing Studio key")
    try:
        async with StudioClient(
            StudioConfig(api_key=key, base_url=_studio_base())
        ) as studio:
            ctx = await studio.load_context()
    except Exception as exc:  # surfaced to the caller, not swallowed
        raise HTTPException(
            status_code=502, detail=f"Studio call failed: {type(exc).__name__}"
        ) from exc
    return {
        "ok": True,
        "user_name": ctx.user_name,
        "organization": ctx.organization,
        "projects": [{"id": p.id, "name": p.name} for p in ctx.projects],
    }


@app.post("/api/run")
async def api_run(request: Request) -> StreamingResponse:
    """Stream a brief's agent loop as Server-Sent Events.

    Body: ``{"brief": str, "max_turns"?: int}``. Each ``run_stream`` event
    becomes one SSE frame; failures (e.g. the LLM server being down) are
    delivered as a final ``{"type": "error"}`` frame rather than a dropped
    connection, so the UI can show them.
    """
    key = _studio_key(request)
    if not key:
        raise HTTPException(status_code=400, detail="missing Studio key")
    try:
        body = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="invalid JSON body") from exc
    brief = str(body.get("brief", "")).strip()
    if not brief:
        raise HTTPException(status_code=400, detail="missing 'brief'")
    max_turns = max(1, min(_MAX_TURNS_CEILING, int(body.get("max_turns", 8))))

    llm: OlmoEarthLLM = app.state.llm
    registry = app.state.registry
    skill_index: str = app.state.skill_index

    async def event_stream() -> AsyncIterator[str]:
        async with StudioClient(
            StudioConfig(api_key=key, base_url=_studio_base())
        ) as studio:
            agent = LeadAgent(
                llm,
                registry,
                studio,
                state=ThreadState(),
                skill_index=skill_index,
            )
            try:
                async for event in agent.run_stream(brief, max_turns=max_turns):
                    yield _sse(event)
            except Exception as exc:  # don't drop the stream — report it
                yield _sse({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
            yield _sse({"type": "done"})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # disable proxy buffering for SSE
        },
    )


# Serve the static front-end at the root. Mounted last so the explicit
# /api/* routes above take precedence over the catch-all.
_WEBUI = _webui_dir()
if _WEBUI.is_dir():
    app.mount("/", StaticFiles(directory=str(_WEBUI), html=True), name="webui")


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint: run the bridge with uvicorn. Returns an exit code."""
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(
        prog="olmoearth-agent-serve",
        description="Serve the OlmoEarth Agent web UI wired to the live agent.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="bind host")
    parser.add_argument("--port", type=int, default=8088, help="bind port")
    args = parser.parse_args(argv)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
