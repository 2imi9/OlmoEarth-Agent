# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
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
(``LLM_*``) endpoint from ``docs/serving.md``; bring it up with
``scripts/serve-llm.sh`` first.

Run it::

    uv run olmoearth-agent-serve            # 127.0.0.1:8088, serves webui/
    # then open http://127.0.0.1:8088

Requires the ``serve`` extra (``uv sync --extra serve``).
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import os
import re
import time
from collections import OrderedDict
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from olmoearth_agent.analysis.aoi import geometry_bbox, validate_polygon_geometry
from olmoearth_agent.harness import LeadAgent, ThreadState
from olmoearth_agent.llm import AnthropicLLM, OlmoEarthLLM, ServingConfig
from olmoearth_agent.llm.anthropic_client import DEFAULT_ANTHROPIC_BASE_URL
from olmoearth_agent.llm.types import Message
from olmoearth_agent.security import egress
from olmoearth_agent.skills import SkillLoader, build_default_registry
from olmoearth_agent.studio import StudioClient
from olmoearth_agent.studio.client import DEFAULT_BASE_URL, StudioConfig

#: Hard cap on agent round-trips a single browser request may trigger.
_MAX_TURNS_CEILING = 12

#: Accepted shape of an optional ``forced_skill`` (the webui "/" menu slug). A
#: tight allow-list so an arbitrary string can't be injected into the prompt.
_SKILL_SLUG_RE = re.compile(r"^[a-z0-9-]{1,40}$")

#: How many prior conversation turns to forward to the agent (bounds prompt size).
_MAX_HISTORY_TURNS = 12

#: Bounded in-memory caches so a repeated map pan / difference scan does not
#: re-fetch the same auth-gated Studio raster tiles or pixel values (which are
#: slow through a proxy). Content is identical per result, so caching by the
#: resolved request is safe; the caller's key is hashed into the cache key so
#: distinct keys never share entries. Cleared on process restart.
_CACHE_MAX = 2048
_TILE_CACHE: OrderedDict[str, tuple[bytes, str]] = OrderedDict()
_PV_CACHE: OrderedDict[str, dict[str, Any]] = OrderedDict()


def _cache_get(cache: OrderedDict[str, Any], k: str) -> Any:
    if k in cache:
        cache.move_to_end(k)
        return cache[k]
    return None


def _cache_put(cache: OrderedDict[str, Any], k: str, v: Any) -> None:
    cache[k] = v
    cache.move_to_end(k)
    while len(cache) > _CACHE_MAX:
        cache.popitem(last=False)


def _key_hash(key: str) -> str:
    """A short, non-reversible tag for the caller's key, to namespace caches."""
    return hashlib.sha256(key.encode()).hexdigest()[:12]


#: Short-TTL cache for read endpoints (projects / areas / predictions / ...).
#: Survives client page reloads (unlike the in-page api.js cache), so reopening
#: the Projects tree / AOI modal after a refresh skips the slow Studio round
#: trip. Keyed by capability + key-hash + ids; TTL keeps it fresh enough.
_READ_CACHE: OrderedDict[str, tuple[float, dict[str, Any]]] = OrderedDict()
_READ_CACHE_MAX = 512
_READ_TTL_LONG = 300.0   # projects / areas / extent: rarely change in a session
_READ_TTL_SHORT = 30.0   # predictions / results: change as runs progress


def _read_cache_get(k: str) -> dict[str, Any] | None:
    hit = _READ_CACHE.get(k)
    if hit is None:
        return None
    expiry, payload = hit
    if time.monotonic() > expiry:
        _READ_CACHE.pop(k, None)
        return None
    _READ_CACHE.move_to_end(k)
    return payload


def _read_cache_put(k: str, payload: dict[str, Any], ttl: float) -> None:
    _READ_CACHE[k] = (time.monotonic() + ttl, payload)
    _READ_CACHE.move_to_end(k)
    while len(_READ_CACHE) > _READ_CACHE_MAX:
        _READ_CACHE.popitem(last=False)


def _read_cache_invalidate(substr: str) -> None:
    for k in [key for key in _READ_CACHE if substr in key]:
        _READ_CACHE.pop(k, None)


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


#: Pooled per-key StudioClient. A burst of pixel-value / tile fetches (the
#: difference scan samples ~grid^2 x2 points, each through the proxy) was opening
#: a brand-new ``StudioClient`` -- and thus a fresh TLS handshake -- per request,
#: which dominated the latency. Reusing one client per key keeps connections
#: alive, so only the first call pays for the handshake. ``httpx.AsyncClient`` is
#: safe for concurrent use, so the UI's 8-way concurrent sampling shares one
#: pooled client fine. A small LRU cap bounds growth; the lifespan closes them.
_STUDIO_POOL: OrderedDict[str, StudioClient] = OrderedDict()
_STUDIO_POOL_MAX = 8
_STUDIO_POOL_LOCK = asyncio.Lock()


async def _pooled_studio(key: str) -> StudioClient:
    """Return a reused, connection-pooled StudioClient for the caller's key."""
    h = _key_hash(key)
    client = _STUDIO_POOL.get(h)
    if client is not None:
        _STUDIO_POOL.move_to_end(h)
        return client
    async with _STUDIO_POOL_LOCK:
        client = _STUDIO_POOL.get(h)  # re-check under the lock
        if client is None:
            client = StudioClient(StudioConfig(api_key=key, base_url=_studio_base()))
            _STUDIO_POOL[h] = client
            while len(_STUDIO_POOL) > _STUDIO_POOL_MAX:
                _, evicted = _STUDIO_POOL.popitem(last=False)
                try:
                    await evicted.aclose()
                except Exception:  # noqa: BLE001, S110 - best-effort close of an evicted client
                    pass
        _STUDIO_POOL.move_to_end(h)
    return client


async def _close_studio_pool() -> None:
    """Close all pooled Studio clients (called on app shutdown)."""
    for client in list(_STUDIO_POOL.values()):
        try:
            await client.aclose()
        except Exception:  # noqa: BLE001, S110 - best-effort cleanup on shutdown
            pass
    _STUDIO_POOL.clear()


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


def _studio_detail(exc: Exception) -> str:
    """A useful 502 detail: the upstream HTTP status when there is one.

    httpx raises ``HTTPStatusError`` carrying a ``.response``; surfacing its
    status code (instead of only the exception class) turns an opaque 502
    into something debuggable.
    """
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    if status is not None:
        return f"Studio API returned HTTP {status}"
    return f"Studio call failed: {type(exc).__name__}"


def _sse(event: dict[str, Any]) -> str:
    """Format one event dict as a Server-Sent Events ``data:`` frame."""
    return f"data: {json.dumps(event)}\n\n"


def _history_from_body(body: dict[str, Any]) -> list[Message]:
    """Parse ``body['history']`` into seed ``Message``s for multi-turn runs.

    Accepts a list of ``{"role": "user"|"assistant", "content": str}``; ignores
    anything else and keeps only the last :data:`_MAX_HISTORY_TURNS` to bound
    the prompt.
    """
    raw = body.get("history")
    if not isinstance(raw, list):
        return []
    out: list[Message] = []
    for item in raw[-_MAX_HISTORY_TURNS:]:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = item.get("content")
        if role in ("user", "assistant") and isinstance(content, str) and content:
            out.append(Message(role=role, content=content))
    return out


def _model_summary(rec: dict[str, Any]) -> dict[str, Any]:
    """Distill a Studio model record into the fields the project tree shows.

    ``model_type`` (e.g. ``"fine_tuned"`` vs ``"embeddings"``) drives the
    node's type badge and icon; ``foundation`` is a friendly encoder
    descriptor (e.g. ``"OlmoEarth Nano v1"``) from the wizard answers,
    surfaced on hover.
    """
    wiz = rec.get("wizard_answers") or {}
    variant = str(wiz.get("encoder_variant") or "").strip()
    version = str(wiz.get("foundation_model_version") or "").strip()
    foundation = ""
    if variant:
        foundation = f"OlmoEarth {variant.title()}"
        if version:
            foundation = f"{foundation} {version}"
    return {
        "id": rec.get("id", ""),
        "name": rec.get("name") or "",
        "model_type": rec.get("model_type") or "",
        "foundation": foundation,
    }


async def _resolve_models(
    studio: StudioClient, predictions: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    """Best-effort ``model_id`` -> summary map for the tree's model level.

    Resolves the distinct model ids referenced by ``predictions`` against
    ``/models/search``. Enrichment only: any failure (or a model past the
    search page) just leaves that node to fall back to its id, so the tree
    still renders. The 200 cap mirrors the predictions page; an account with
    more models than that may show a few unlabeled nodes.
    """
    model_ids = {str(r["model_id"]) for r in predictions if r.get("model_id")}
    if not model_ids:
        return {}
    try:
        env = await studio.search_models(limit=200)
    except Exception:  # enrichment must never break the tree
        return {}
    by_id = {m.get("id"): m for m in env.records}
    return {mid: _model_summary(by_id[mid]) for mid in model_ids if mid in by_id}


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
    await _close_studio_pool()


app = FastAPI(title="OlmoEarth Agent bridge", lifespan=_lifespan)


@app.middleware("http")
async def _revalidate_static(request: Request, call_next: Any) -> Any:
    """Serve the static web UI with ``Cache-Control: no-cache``.

    The bridge serves the ``webui/`` ES modules with no build step, so a
    cached old module after an edit is a real footgun (it needs a manual
    hard refresh). ``no-cache`` makes the browser revalidate every load
    (304 when unchanged), keeping edits visible without disabling caching
    outright. API responses (``/api/*``, incl. the SSE stream) are left
    untouched.
    """
    response = await call_next(request)
    if not request.url.path.startswith("/api/"):
        response.headers.setdefault("Cache-Control", "no-cache")
    return response


def _claude_available() -> bool:
    """Whether the optional ``anthropic`` SDK is importable (the claude extra)."""
    return importlib.util.find_spec("anthropic") is not None


#: Hosted OpenAI-compatible providers: base URL + a sensible default model
#: (overridden by the model the user picks from autodetect). Claude is handled
#: separately via the native Anthropic client.
_OPENAI_PROVIDERS: dict[str, dict[str, str]] = {
    "openai": {"base_url": "https://api.openai.com/v1", "model": "gpt-4o"},
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "model": "gemini-2.0-flash",
    },
}


def _llm_detail(exc: Exception) -> str:
    """A debuggable detail for an upstream LLM-provider failure."""
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    if status is not None:
        return f"LLM provider returned HTTP {status}"
    return f"LLM provider call failed: {type(exc).__name__}"


def _llm_for_request(request: Request) -> Any:
    """Choose the LLM backend for this request (default: the local model).

    Hosted backends are selected via ``X-LLM-Backend`` (``claude`` |
    ``openai`` | ``gemini``) with a bring-your-own ``X-LLM-Key`` and an
    optional ``X-LLM-Model``. The key is forwarded per request and never
    stored or logged, mirroring the Studio key. ChatGPT/Gemini share the
    OpenAI-compatible client (``openai_compat=True`` drops the Qwen extras);
    Claude uses the native Anthropic client. Subscription/OAuth auth is not
    wired here.
    """
    backend = request.headers.get("x-llm-backend", "local").strip().lower()
    if backend in ("", "local"):
        return app.state.llm
    api_key = request.headers.get("x-llm-key", "").strip()
    if not api_key:
        raise HTTPException(status_code=400, detail="missing API key for backend")
    model = request.headers.get("x-llm-model", "").strip()
    if backend in ("claude", "anthropic"):
        # Guard the destination before the BYO key is handed to the client, so a
        # mis-pointed endpoint cannot become a credential-exfiltration channel.
        try:
            egress.validate_endpoint(DEFAULT_ANTHROPIC_BASE_URL, "llm-cloud")
        except egress.EgressError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        kwargs: dict[str, Any] = {"api_key": api_key}
        if model:
            kwargs["model"] = model
        try:
            return AnthropicLLM(**kwargs)
        except RuntimeError as exc:  # anthropic not installed, or no credentials
            raise HTTPException(status_code=503, detail=str(exc)) from exc
    provider = _OPENAI_PROVIDERS.get(backend)
    if provider is None:
        raise HTTPException(status_code=400, detail=f"unknown LLM backend: {backend}")
    try:
        egress.validate_endpoint(provider["base_url"], "llm-cloud")
    except egress.EgressError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    return OlmoEarthLLM(
        ServingConfig(
            endpoint=provider["base_url"],
            api_key=api_key,
            model=model or provider["model"],
        ),
        openai_compat=True,
    )


def _filter_models(backend: str, ids: list[str]) -> list[str]:
    """Sort + lightly filter listed model ids to the likely chat models."""
    needles = {"openai": ("gpt", "o1", "o3", "chatgpt"), "gemini": ("gemini",)}.get(
        backend
    )
    if needles:
        chat = [i for i in ids if any(n in i.lower() for n in needles)]
        ids = chat or ids  # fall back to everything if the filter is too tight
    return sorted(set(ids))


async def _list_models(backend: str, api_key: str) -> list[str]:
    """List a provider's current model ids for the web UI autodetect.

    Claude uses the Anthropic SDK; ChatGPT/Gemini use the OpenAI SDK against
    the provider's model-listing endpoint. Network errors propagate to the
    caller (surfaced as a 502).
    """
    if backend in ("claude", "anthropic"):
        try:
            from anthropic import AsyncAnthropic
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "The 'anthropic' package is required for the Claude backend. "
                "Install it with: uv sync --extra claude"
            ) from exc

        client = AsyncAnthropic(api_key=api_key)
        try:
            page = await client.models.list(limit=100)
            ids = [m.id for m in page.data]
        finally:
            await client.close()
        return _filter_models(backend, ids)
    provider = _OPENAI_PROVIDERS.get(backend)
    if provider is None:
        raise HTTPException(status_code=400, detail=f"unknown LLM backend: {backend}")
    from openai import AsyncOpenAI

    openai_client = AsyncOpenAI(base_url=provider["base_url"], api_key=api_key)
    try:
        resp = await openai_client.models.list()
        ids = [m.id for m in resp.data]
    finally:
        await openai_client.close()
    return _filter_models(backend, ids)


async def _local_llm_up(endpoint: str) -> bool:
    """Best-effort check that the configured local LLM endpoint answers.

    Used only so the web UI can nudge the user when the default (local)
    backend is selected but no model server is running (start it, or pick a
    cloud provider). A localhost model probe must never traverse a proxy, so
    ``trust_env=False``. Any failure -> ``False``; bounded by a short timeout
    and never raises.
    """
    url = endpoint.rstrip("/") + "/models"
    try:
        async with httpx.AsyncClient(timeout=2.0, trust_env=False) as client:
            resp = await client.get(url)
        return resp.status_code == 200
    except Exception:
        return False


@app.get("/api/health")
async def api_health() -> dict[str, Any]:
    """Liveness probe; the front-end uses it to switch from demo to live.

    ``llm_local_up`` is a best-effort reachability check of the configured
    local LLM endpoint, so the UI can tell a cloud-only user (no ``make
    serve``) to pick a provider instead of failing on the local default.
    """
    llm: OlmoEarthLLM = app.state.llm
    return {
        "ok": True,
        "mode": "live",
        "llm_endpoint": llm.config.endpoint,
        "llm_model": llm.config.model,
        "llm_local_up": await _local_llm_up(llm.config.endpoint),
        "studio_base": _studio_base(),
        "claude_available": _claude_available(),
    }


@app.get("/api/llm/models")
async def api_llm_models(request: Request) -> dict[str, Any]:
    """List the selected provider's model ids (web UI model autodetect).

    Reads ``X-LLM-Backend`` + ``X-LLM-Key``; the key is used for this one
    lookup and never stored. Returns ``{"ok": True, "models": [...]}``.
    """
    api_key = request.headers.get("x-llm-key", "").strip()
    if not api_key:
        raise HTTPException(status_code=400, detail="missing API key")
    backend = request.headers.get("x-llm-backend", "").strip().lower()
    if backend in ("", "local"):
        raise HTTPException(
            status_code=400,
            detail="select a hosted backend (claude, openai, or gemini) to list models",
        )
    try:
        models = await _list_models(backend, api_key)
    except HTTPException:
        raise
    except (ImportError, RuntimeError) as exc:  # e.g. the anthropic extra is missing
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=_llm_detail(exc)) from exc
    return {"ok": True, "models": models}


@app.get("/api/projects")
async def api_projects(request: Request) -> dict[str, Any]:
    """Return the caller's real Studio projects via ``load_context``."""
    key = _studio_key(request)
    if not key:
        raise HTTPException(status_code=400, detail="missing Studio key")
    ckey = f"projects|{_key_hash(key)}"
    cached = _read_cache_get(ckey)
    if cached is not None:
        return cached
    try:
        async with StudioClient(
            StudioConfig(api_key=key, base_url=_studio_base())
        ) as studio:
            ctx = await studio.load_context()
    except Exception as exc:  # surfaced to the caller, not swallowed
        raise HTTPException(status_code=502, detail=_studio_detail(exc)) from exc
    payload = {
        "ok": True,
        "user_name": ctx.user_name,
        "organization": ctx.organization,
        "projects": [{"id": p.id, "name": p.name} for p in ctx.projects],
    }
    _read_cache_put(ckey, payload, _READ_TTL_LONG)
    return payload


@app.post("/api/run")
async def api_run(request: Request) -> StreamingResponse:
    """Stream a brief's agent loop as Server-Sent Events.

    Body: ``{"brief": str, "max_turns"?: int, "history"?: list,
    "forced_skill"?: str}``. ``forced_skill`` (a webui "/" menu slug) pins the
    run to one skill server-side. Each ``run_stream`` event becomes one SSE
    frame; failures (e.g. the LLM server being down) are delivered as a final
    ``{"type": "error"}`` frame rather than a dropped connection, so the UI can
    show them.
    """
    key = _studio_key(request)
    if not key:
        raise HTTPException(status_code=400, detail="missing Studio key")
    try:
        body = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="invalid JSON body") from exc
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="JSON body must be an object")
    brief = str(body.get("brief", "")).strip()
    if not brief:
        raise HTTPException(status_code=400, detail="missing 'brief'")
    try:
        max_turns = max(1, min(_MAX_TURNS_CEILING, int(body.get("max_turns", 8))))
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="invalid 'max_turns'") from exc
    history = _history_from_body(body)
    # Optional server-side skill routing: the webui "/" menu sends the picked
    # skill slug here, pinning the run to it. Malformed values are ignored (not
    # rejected) so a stray slug never breaks a run.
    forced_skill = str(body.get("forced_skill", "")).strip().lower()
    if not _SKILL_SLUG_RE.match(forced_skill):
        forced_skill = ""

    llm = _llm_for_request(request)
    registry = app.state.registry
    skill_index: str = app.state.skill_index

    async def event_stream() -> AsyncIterator[str]:
        try:
            async with StudioClient(
                StudioConfig(api_key=key, base_url=_studio_base())
            ) as studio:
                agent = LeadAgent(
                    llm,
                    registry,
                    studio,
                    state=ThreadState(),
                    skill_index=skill_index,
                    forced_skill=forced_skill,
                    # The shared app.state.llm is the local model; a per-request
                    # hosted client (Claude/OpenAI/Gemini) is not. Only the local
                    # model gets the brevity/budget clause.
                    local=llm is app.state.llm,
                )
                try:
                    async for event in agent.run_stream(
                        brief, max_turns=max_turns, history=history
                    ):
                        yield _sse(event)
                except Exception as exc:  # don't drop the stream, report it
                    yield _sse(
                        {"type": "error", "message": f"{type(exc).__name__}: {exc}"}
                    )
                yield _sse({"type": "done"})
        finally:
            # Release the per-request hosted client (Claude/OpenAI/Gemini); the
            # shared local model is reused across requests and must stay open.
            if llm is not app.state.llm:
                await llm.aclose()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # disable proxy buffering for SSE
        },
    )


@app.post("/api/areas")
async def api_create_area(request: Request) -> dict[str, Any]:
    """Store a drawn AOI as a Studio area (the AOI-draw-in-chat flow).

    Body: ``{"name": str, "geom": <GeoJSON Polygon/MultiPolygon>,
    "project_id": str}``. The geometry is the polygon the user drew on the
    chat map; it is stored via ``POST /areas`` so the returned ``area_id``
    can drive a prediction (skill #5) and the derived ``bbox`` can drive a
    bbox-based skill. Returns ``{ok, area: {id, name, project_id, bbox}}``.
    """
    key = _studio_key(request)
    if not key:
        raise HTTPException(status_code=400, detail="missing Studio key")
    try:
        body = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="invalid JSON body") from exc
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="JSON body must be an object")
    name = str(body.get("name", "")).strip()
    project_id = str(body.get("project_id", "")).strip()
    geom = body.get("geom")
    if not name:
        raise HTTPException(status_code=400, detail="missing 'name'")
    if not project_id:
        raise HTTPException(status_code=400, detail="missing 'project_id'")
    try:
        geom = validate_polygon_geometry(geom)
        bbox = geometry_bbox(geom)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    try:
        async with StudioClient(
            StudioConfig(api_key=key, base_url=_studio_base())
        ) as studio:
            record = await studio.create_area(
                name=name, geom=geom, project_id=project_id
            )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=_studio_detail(exc)) from exc
    # The project's area list changed -> drop its cached read (server + clients).
    _read_cache_invalidate(f"areas|{_key_hash(key)}|{project_id}")
    return {
        "ok": True,
        "area": {
            "id": record.get("id"),
            "name": record.get("name"),
            "project_id": record.get("project_id") or project_id,
            "bbox": bbox,
        },
    }


@app.get("/api/projects/{project_id}/areas")
async def api_project_areas(project_id: str, request: Request) -> dict[str, Any]:
    """List a project's existing areas (so a drawn AOI can be reused)."""
    key = _studio_key(request)
    if not key:
        raise HTTPException(status_code=400, detail="missing Studio key")
    ckey = f"areas|{_key_hash(key)}|{project_id}"
    cached = _read_cache_get(ckey)
    if cached is not None:
        return cached
    try:
        async with StudioClient(
            StudioConfig(api_key=key, base_url=_studio_base())
        ) as studio:
            env = await studio.search_areas(project_id=project_id, limit=200)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=_studio_detail(exc)) from exc
    payload = {
        "ok": True,
        "areas": [
            {"id": r.get("id", ""), "name": r.get("name") or "(unnamed)"}
            for r in env.records
        ],
    }
    _read_cache_put(ckey, payload, _READ_TTL_LONG)
    return payload


@app.get("/api/areas/{area_id}")
async def api_get_area(area_id: str, request: Request) -> dict[str, Any]:
    """Fetch one stored area with its geometry + derived bbox (for reuse).

    The list endpoint omits ``geom`` (lightweight); when the user picks a
    saved area to reuse, the draw UI calls this to render it on the map and
    feed its bbox to bbox-based skills.
    """
    key = _studio_key(request)
    if not key:
        raise HTTPException(status_code=400, detail="missing Studio key")
    ckey = f"area|{_key_hash(key)}|{area_id}"
    cached = _read_cache_get(ckey)
    if cached is not None:
        return cached
    try:
        async with StudioClient(
            StudioConfig(api_key=key, base_url=_studio_base())
        ) as studio:
            record = await studio.get_area(area_id)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=_studio_detail(exc)) from exc
    geom = record.get("geom")
    try:
        bbox = geometry_bbox(geom) if geom else None
    except ValueError:
        bbox = None
    payload = {
        "ok": True,
        "area": {
            "id": record.get("id"),
            "name": record.get("name"),
            "project_id": record.get("project_id"),
            "geom": geom,
            "bbox": bbox,
        },
    }
    _read_cache_put(ckey, payload, _READ_TTL_LONG)
    return payload


@app.get("/api/projects/{project_id}/predictions")
async def api_project_predictions(project_id: str, request: Request) -> dict[str, Any]:
    """Predictions for a project: the tree's model + prediction levels.

    The front-end groups predictions by ``model_id`` into a synthetic
    "Model" level; we resolve each ``model_id`` to its Studio model record
    so that level can show the model's real name and whether it is a
    fine-tuned model or an embeddings run. (``/models`` is undocumented in
    openapi v0.1.0 but live; see :meth:`StudioClient.search_models`.)
    """
    key = _studio_key(request)
    if not key:
        raise HTTPException(status_code=400, detail="missing Studio key")
    ckey = f"preds|{_key_hash(key)}|{project_id}"
    cached = _read_cache_get(ckey)
    if cached is not None:
        return cached
    try:
        async with StudioClient(
            StudioConfig(api_key=key, base_url=_studio_base())
        ) as studio:
            env = await studio.search_predictions(project_id=project_id, limit=200)
            models = await _resolve_models(studio, env.records)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=_studio_detail(exc)) from exc
    predictions = [
        {
            "id": r.get("id", ""),
            "name": r.get("name") or "(unnamed)",
            "status": r.get("status") or "",
            "model_id": r.get("model_id") or "",
        }
        for r in env.records
    ]
    payload = {"ok": True, "predictions": predictions, "models": models}
    _read_cache_put(ckey, payload, _READ_TTL_SHORT)
    return payload


@app.get("/api/predictions/{prediction_id}/results")
async def api_prediction_results(
    prediction_id: str, request: Request
) -> dict[str, Any]:
    """Prediction-results for a prediction: the tree's leaf level."""
    key = _studio_key(request)
    if not key:
        raise HTTPException(status_code=400, detail="missing Studio key")
    ckey = f"results|{_key_hash(key)}|{prediction_id}"
    cached = _read_cache_get(ckey)
    if cached is not None:
        return cached
    try:
        async with StudioClient(
            StudioConfig(api_key=key, base_url=_studio_base())
        ) as studio:
            env = await studio.search_prediction_results(
                prediction_id=prediction_id, limit=200
            )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=_studio_detail(exc)) from exc
    results = [
        {
            "id": r.get("id", ""),
            "property_names": r.get("property_names") or [],
            "file_format": r.get("file_format") or "",
            "tiles": len(r.get("tile_urls") or []),
            "tile_url": (r.get("tile_urls") or [""])[0],
        }
        for r in env.records
    ]
    payload = {"ok": True, "results": results}
    _read_cache_put(ckey, payload, _READ_TTL_SHORT)
    return payload


@app.get("/api/results/{result_id}/extent")
async def api_result_extent(result_id: str, request: Request) -> dict[str, Any]:
    """Geographic extent of a prediction-result, so the chat can show the raster.

    Reads ``result_metadata.geometry`` (the result's bounds) and returns its
    bbox plus the first tile-URL template and property name. The web UI uses
    the bbox to fit the result map to the raster (otherwise a small AOI is
    invisible at world zoom).
    """
    key = _studio_key(request)
    if not key:
        raise HTTPException(status_code=400, detail="missing Studio key")
    ckey = f"extent|{_key_hash(key)}|{result_id}"
    cached = _read_cache_get(ckey)
    if cached is not None:
        return cached
    try:
        async with StudioClient(
            StudioConfig(api_key=key, base_url=_studio_base())
        ) as studio:
            rec = await studio.get_prediction_result(result_id)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=_studio_detail(exc)) from exc
    geom = (rec.get("result_metadata") or {}).get("geometry")
    try:
        bbox = geometry_bbox(geom) if geom else None
    except ValueError:
        bbox = None
    tiles = rec.get("tile_urls") or []
    props = rec.get("property_names") or []
    payload = {
        "ok": True,
        "extent": {
            "result_id": rec.get("id") or result_id,
            "bbox": bbox,
            "tile_url": tiles[0] if tiles else None,
            "property": props[0] if props else None,
        },
    }
    _read_cache_put(ckey, payload, _READ_TTL_LONG)
    return payload


@app.get("/api/pixel-value")
async def api_pixel_value(request: Request) -> dict[str, Any]:
    """Pointwise model output at one coordinate (for the difference scan).

    The web UI samples both results on a grid through here to paint a live
    difference map. ``result_id``, ``lon``, ``lat`` (and optional
    ``property``) are query params. A point off the raster (Studio 404) is
    returned as ``value: null`` so the scan just skips that cell.
    """
    key = _studio_key(request)
    if not key:
        raise HTTPException(status_code=400, detail="missing Studio key")
    q = request.query_params
    result_id = q.get("result_id", "").strip()
    lon, lat = q.get("lon"), q.get("lat")
    if not result_id or lon is None or lat is None:
        raise HTTPException(status_code=400, detail="result_id, lon, lat required")
    try:
        lon_f, lat_f = float(lon), float(lat)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="lon/lat must be numbers") from exc
    prop_key = q.get("property") or ""
    ckey = f"{_key_hash(key)}|{result_id}|{lon_f}|{lat_f}|{prop_key}"
    cached: dict[str, Any] | None = _cache_get(_PV_CACHE, ckey)
    if cached is not None:
        return cached
    try:
        studio = await _pooled_studio(key)
        rec = await studio.pixel_value(result_id, lon_f, lat_f)
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            out = {"ok": True, "value": None}  # off-raster / nodata
            _cache_put(_PV_CACHE, ckey, out)
            return out
        raise HTTPException(status_code=502, detail=_studio_detail(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=_studio_detail(exc)) from exc
    bands = rec.get("bands") or []
    prop = q.get("property")
    band = next((b for b in bands if b.get("property_name") == prop), bands[0] if bands else {})
    cls = band.get("classification")
    out = {
        "ok": True,
        "value": cls if cls is not None else band.get("raw_value"),
        "property": band.get("property_name"),
        "categorical": cls is not None,
    }
    _cache_put(_PV_CACHE, ckey, out)
    return out


@app.get("/api/tile/{z}/{x}/{y}")
async def api_tile(z: int, x: int, y: int, request: Request) -> Response:
    """Proxy one Studio raster tile, adding the caller's Bearer key.

    Browser map tiles (``<img>`` requests) cannot carry an Authorization
    header, and Studio tiles are auth-gated, so the web UI's result map
    fetches each tile through here (a ``fetch`` *can* send
    ``X-Olmoearth-Key``). The ``src`` query param is the Studio tile
    template (with ``{z}/{x}/{y}``); we substitute, resolve it against the
    Studio origin only (the egress guard blocks any other host -> no SSRF),
    fetch with the Bearer key, and return the image bytes. A non-200 from
    Studio is passed through bare so the map simply skips that tile.
    """
    key = _studio_key(request)
    if not key:
        raise HTTPException(status_code=400, detail="missing Studio key")
    src = request.query_params.get("src", "")
    if not src:
        raise HTTPException(status_code=400, detail="missing 'src' tile template")
    path = src.replace("{z}", str(z)).replace("{x}", str(x)).replace("{y}", str(y))
    origin = _studio_base().split("/api/")[0]  # https://olmoearth.allenai.org
    if path.startswith(("http://", "https://")):
        url = path
    else:
        url = origin + ("" if path.startswith("/") else "/") + path
    # Hard SSRF guard: the tile must come from the Studio host, independent of
    # the OLMOEARTH_EGRESS mode (which defaults to log-only) -- this proxy must
    # never become an open relay for arbitrary URLs.
    from urllib.parse import urlparse

    if urlparse(url).netloc != urlparse(origin).netloc:
        raise HTTPException(status_code=403, detail="tile src host not allowed")
    ckey = _key_hash(key) + "|" + url
    cached = _cache_get(_TILE_CACHE, ckey)
    if cached is not None:
        body, media = cached
        return Response(content=body, media_type=media)
    try:
        studio = await _pooled_studio(key)
        resp = await studio.get_bytes(url, timeout=20.0)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=_studio_detail(exc)) from exc
    if resp.status_code != 200:
        return Response(status_code=resp.status_code)
    media = resp.headers.get("content-type", "image/png")
    _cache_put(_TILE_CACHE, ckey, (resp.content, media))
    return Response(content=resp.content, media_type=media)


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
