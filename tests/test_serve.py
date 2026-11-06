# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Tests for the FastAPI bridge (serve.py) with a fake LLM: no live services."""

from __future__ import annotations

from collections.abc import Iterable
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from pytest_httpx import HTTPXMock

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from olmoearth_agent import serve  # noqa: E402
from olmoearth_agent.llm.types import ChatResponse, Message  # noqa: E402


@pytest.fixture(autouse=True)
def _clear_bridge_caches() -> Any:
    """The bridge's tile / pixel / read caches are module-global; clear them
    between tests so a cached response can't leak across cases."""
    serve._READ_CACHE.clear()
    serve._TILE_CACHE.clear()
    serve._PV_CACHE.clear()
    serve._STUDIO_POOL.clear()  # pooled clients are module-global too
    yield
    serve._STUDIO_POOL.clear()


class _FakeLLM:
    """Minimal stand-in for OlmoEarthLLM: scripted responses + a config."""

    def __init__(self, responses: Iterable[ChatResponse]) -> None:
        self._responses = list(responses)
        self.closed = False

    async def chat(
        self, messages: list[Message], *, tools: Any = None, **_kw: Any
    ) -> ChatResponse:
        return self._responses.pop(0)

    async def aclose(self) -> None:
        self.closed = True


def test_health_reports_live() -> None:
    with TestClient(serve.app) as client:
        resp = client.get("/api/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["mode"] == "live"
    assert "llm_endpoint" in body


def test_projects_requires_key() -> None:
    with TestClient(serve.app) as client:
        resp = client.get("/api/projects")
    assert resp.status_code == 400


def test_run_requires_key() -> None:
    with TestClient(serve.app) as client:
        resp = client.post("/api/run", json={"brief": "hi"})
    assert resp.status_code == 400


def test_run_requires_brief() -> None:
    with TestClient(serve.app) as client:
        resp = client.post("/api/run", json={}, headers={"X-Olmoearth-Key": "k"})
    assert resp.status_code == 400


def test_run_streams_final_answer() -> None:
    with TestClient(serve.app) as client:
        serve.app.state.llm = _FakeLLM(
            [ChatResponse(content="hello there", tool_calls=[], finish_reason="stop")]
        )
        resp = client.post(
            "/api/run",
            json={"brief": "say hi"},
            headers={"X-Olmoearth-Key": "k"},
        )
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]
    body = resp.text
    assert '"type": "final"' in body
    assert "hello there" in body
    assert '"type": "done"' in body


def test_run_streams_error_event_when_llm_fails() -> None:
    class _BoomLLM:
        async def chat(self, *_a: Any, **_k: Any) -> ChatResponse:
            raise RuntimeError("llm down")

    with TestClient(serve.app) as client:
        serve.app.state.llm = _BoomLLM()
        resp = client.post(
            "/api/run",
            json={"brief": "x"},
            headers={"X-Olmoearth-Key": "k"},
        )
    assert resp.status_code == 200
    body = resp.text
    assert '"type": "error"' in body
    assert "llm down" in body
    assert '"type": "done"' in body


def test_run_forwards_history() -> None:
    captured: dict[str, Any] = {}

    class _CapLLM:
        async def chat(
            self, messages: list[Message], *, tools: Any = None, **_kw: Any
        ) -> ChatResponse:
            captured["messages"] = list(messages)
            return ChatResponse(content="ok", tool_calls=[], finish_reason="stop")

    with TestClient(serve.app) as client:
        serve.app.state.llm = _CapLLM()
        resp = client.post(
            "/api/run",
            json={
                "brief": "follow up",
                "history": [
                    {"role": "user", "content": "first q"},
                    {"role": "assistant", "content": "first a"},
                ],
            },
            headers={"X-Olmoearth-Key": "k"},
        )
    assert resp.status_code == 200
    roles = [(m.role, m.content) for m in captured["messages"]]
    assert ("user", "first q") in roles
    assert ("assistant", "first a") in roles
    assert roles[-1] == ("user", "follow up")  # new brief is last


def test_run_local_flag_tracks_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class _CapAgent:
        def __init__(self, *_a: Any, **kw: Any) -> None:
            captured["local"] = kw.get("local")

        async def run_stream(self, *_a: Any, **_kw: Any) -> Any:
            yield {"type": "final", "turn": 1, "content": "ok"}

    monkeypatch.setattr(serve, "LeadAgent", _CapAgent)
    with TestClient(serve.app) as client:
        # Default (local) backend -> the shared local model -> brevity clause on.
        client.post("/api/run", json={"brief": "hi"}, headers={"X-Olmoearth-Key": "k"})
        assert captured["local"] is True
        captured.clear()
        # A hosted backend -> a per-request client -> no brevity clause.
        client.post(
            "/api/run",
            json={"brief": "hi"},
            headers={
                "X-Olmoearth-Key": "k",
                "X-LLM-Backend": "openai",
                "X-LLM-Key": "sk-x",
            },
        )
    assert captured["local"] is False


def test_run_auto_route_demotes_simple_brief_to_local(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class _CapAgent:
        def __init__(self, *_a: Any, **kw: Any) -> None:
            captured["local"] = kw.get("local")

        async def run_stream(self, *_a: Any, **_kw: Any) -> Any:
            yield {"type": "final", "turn": 1, "content": "ok"}

    monkeypatch.setattr(serve, "LeadAgent", _CapAgent)
    hosted = {
        "X-Olmoearth-Key": "k",
        "X-LLM-Backend": "openai",
        "X-LLM-Key": "sk-x",
        "X-LLM-Route": "auto",
    }
    with TestClient(serve.app) as client:
        # Simple lookup + auto -> demoted to the shared local model.
        client.post("/api/run", json={"brief": "list my projects"}, headers=hosted)
        assert captured["local"] is True
        captured.clear()
        # Complex brief + auto -> stays on the hosted backend.
        client.post(
            "/api/run",
            json={"brief": "create and train a flood model for the Mekong"},
            headers=hosted,
        )
        assert captured["local"] is False
        captured.clear()
        # A user-forced skill means a workflow: never demoted, even if the
        # brief itself reads like a lookup.
        client.post(
            "/api/run",
            json={"brief": "list my projects", "forced_skill": "predict"},
            headers=hosted,
        )
        assert captured["local"] is False
        captured.clear()
        # Without the auto header, the hosted selection is untouched.
        no_route = {k: v for k, v in hosted.items() if k != "X-LLM-Route"}
        client.post("/api/run", json={"brief": "list my projects"}, headers=no_route)
    assert captured["local"] is False


def test_run_forwards_forced_skill(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class _CapAgent:
        def __init__(self, *_a: Any, **kw: Any) -> None:
            captured["forced_skill"] = kw.get("forced_skill")

        async def run_stream(self, *_a: Any, **_kw: Any) -> Any:
            yield {"type": "final", "turn": 1, "content": "ok"}

    monkeypatch.setattr(serve, "LeadAgent", _CapAgent)
    with TestClient(serve.app) as client:
        # A valid slug is forwarded, lower-cased.
        client.post(
            "/api/run",
            json={"brief": "hi", "forced_skill": "Review-Set"},
            headers={"X-Olmoearth-Key": "k"},
        )
        assert captured["forced_skill"] == "review-set"
        captured.clear()
        # A malformed value is ignored (empty), not rejected with an error.
        resp = client.post(
            "/api/run",
            json={"brief": "hi", "forced_skill": "drop tables; --"},
            headers={"X-Olmoearth-Key": "k"},
        )
        assert resp.status_code == 200
        assert captured["forced_skill"] == ""
        captured.clear()
        # A well-formed but NON-EXISTENT slug is ignored (never injected into the
        # prompt directive), even though it passes the shape regex.
        resp = client.post(
            "/api/run",
            json={"brief": "hi", "forced_skill": "totally-fake-skill"},
            headers={"X-Olmoearth-Key": "k"},
        )
        assert resp.status_code == 200
        assert captured["forced_skill"] == ""
        captured.clear()
        # Absent -> empty (no pin).
        client.post("/api/run", json={"brief": "hi"}, headers={"X-Olmoearth-Key": "k"})
    assert captured["forced_skill"] == ""


def _forced_as_sent(monkeypatch: pytest.MonkeyPatch, slugs: Iterable[str]) -> list[str]:
    """The ``forced_skill`` the agent is built with, per slug the UI sends."""
    forwarded: list[str] = []

    class _CapAgent:
        def __init__(self, *_a: Any, **kw: Any) -> None:
            forwarded.append(kw.get("forced_skill"))

        async def run_stream(self, *_a: Any, **_kw: Any) -> Any:
            yield {"type": "final", "turn": 1, "content": "ok"}

    monkeypatch.setattr(serve, "LeadAgent", _CapAgent)
    with TestClient(serve.app) as client:
        for slug in slugs:
            client.post(
                "/api/run",
                json={"brief": "hi", "forced_skill": slug},
                headers={"X-Olmoearth-Key": "k"},
            )
    return forwarded


def test_studio_job_config_stays_forced_without_tools_of_its_own(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """studio-job-config lists no tool of its own, but its SKILL.md loads, so
    it stays forced."""
    sent = _forced_as_sent(monkeypatch, ("studio-job-config",))
    assert sent == ["studio-job-config"]


def test_the_rows_that_went_are_not_forced(monkeypatch: pytest.MonkeyPatch) -> None:
    """Change detection and uncertainty folded into predict's comparison tool,
    and evaluate, the QGIS bridge, the data export and the training-side
    packages (data prep, embeddings, rslearn) went: an old "/" menu slug of
    theirs pins nothing."""
    slugs = (
        "change-detection",
        "uncertainty",
        "evaluate",
        "qgis-bridge",
        "data-export",
        "data-prep",
        "embeddings",
        "rslearn",
    )
    assert _forced_as_sent(monkeypatch, slugs) == [""] * len(slugs)


def _offered_per_forced_skill(
    monkeypatch: pytest.MonkeyPatch, slugs: Iterable[str]
) -> dict[str, set[str]]:
    """The tool specs the first turn offers, per forced skill ("" = none)."""
    offered: dict[str, set[str]] = {}

    class _Probe(serve.LeadAgent):
        def __init__(self, *a: Any, **kw: Any) -> None:
            super().__init__(*a, **kw)
            specs = self.registry.active_specs(self.state.loaded_groups)
            offered[kw.get("forced_skill") or ""] = {s.name for s in specs}

        async def run_stream(self, *_a: Any, **_kw: Any) -> Any:
            yield {"type": "final", "turn": 1, "content": "ok"}

    monkeypatch.setattr(serve, "LeadAgent", _Probe)
    with TestClient(serve.app) as client:
        for slug in slugs:
            client.post(
                "/api/run",
                json={"brief": "set up training", "forced_skill": slug},
                headers={"X-Olmoearth-Key": "k"},
            )
    return offered


def test_forcing_studio_job_config_offers_no_tool_of_its_own(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The instruction skill has no deferred tools: forcing it offers the
    first turn the same tools as no forced skill.

    A forced skill's deferred group from turn one is tested on the agent
    (tests/harness/test_agent.py); no tool of the agent's is deferred.
    """
    offered = _offered_per_forced_skill(monkeypatch, ("studio-job-config", ""))
    assert offered["studio-job-config"] == offered[""]
    assert "olmoearth_compare_results" in offered[""]


def test_skills_endpoint_exposes_per_skill_stages() -> None:
    with TestClient(serve.app) as client:
        resp = client.get("/api/skills")
    assert resp.status_code == 200
    body = resp.json()
    # the ordered canonical stage list, for key -> label mapping in the UI
    stage_keys = [s["key"] for s in body["stages"]]
    assert stage_keys[0] == "context" and stage_keys[-1] == "report"
    skills = {s["slug"]: s for s in body["skills"]}
    # a run skill gets the full pipeline; an advisory skill skips submit/poll/fetch
    assert "submit" in skills["predict"]["stages"]
    assert "submit" not in skills["review-set"]["stages"]
    assert "fetch" not in skills["review-set"]["stages"]


def test_health_reports_claude_available() -> None:
    with TestClient(serve.app) as client:
        resp = client.get("/api/health")
    assert resp.status_code == 200
    assert isinstance(resp.json()["claude_available"], bool)


def test_health_reports_local_llm_up(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _up(_endpoint: str, _api_key: str | None = None) -> bool:
        return True

    monkeypatch.setattr(serve, "_local_llm_up", _up)
    with TestClient(serve.app) as client:
        body = client.get("/api/health").json()
    assert body["llm_local_up"] is True


def test_health_reports_local_llm_down(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _down(_endpoint: str, _api_key: str | None = None) -> bool:
        return False

    monkeypatch.setattr(serve, "_local_llm_up", _down)
    with TestClient(serve.app) as client:
        body = client.get("/api/health").json()
    assert body["llm_local_up"] is False


def test_local_llm_probe_false_when_unreachable() -> None:
    # A closed localhost port: the probe must swallow the error and report down,
    # never raise (so /api/health stays fast and green without a local model).
    import asyncio

    assert asyncio.run(serve._local_llm_up("http://127.0.0.1:1/v1")) is False


def test_local_llm_probe_sends_the_api_key(httpx_mock: HTTPXMock) -> None:
    # A vLLM server started with an access token answers /models only with the
    # key; without it the probe read "local model is not up" while the model
    # served the agent (24 September trial).
    import asyncio

    def _requires_token(request: httpx.Request) -> httpx.Response:
        if request.headers.get("authorization") == "Bearer tok-123":
            return httpx.Response(200, json={"data": [{"id": "m"}]})
        return httpx.Response(401, json={"error": "Unauthorized"})

    httpx_mock.add_callback(
        _requires_token, url="http://gpu-node:8000/v1/models", is_reusable=True
    )
    endpoint = "http://gpu-node:8000/v1"
    assert asyncio.run(serve._local_llm_up(endpoint, "tok-123")) is True
    assert asyncio.run(serve._local_llm_up(endpoint, "wrong")) is False
    assert asyncio.run(serve._local_llm_up(endpoint)) is False


def test_health_probes_with_the_configured_llm_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[str | None] = []

    async def _probe(_endpoint: str, api_key: str | None = None) -> bool:
        seen.append(api_key)
        return True

    monkeypatch.setenv("LLM_API_KEY", "tok-123")
    monkeypatch.setattr(serve, "_local_llm_up", _probe)
    with TestClient(serve.app) as client:
        body = client.get("/api/health").json()
    assert body["llm_local_up"] is True
    assert seen == ["tok-123"]


def test_run_claude_backend_requires_key() -> None:
    with TestClient(serve.app) as client:
        resp = client.post(
            "/api/run",
            json={"brief": "hi"},
            headers={"X-Olmoearth-Key": "k", "X-LLM-Backend": "claude"},
        )
    assert resp.status_code == 400
    assert "API key" in resp.json()["detail"]


def test_run_uses_claude_backend_when_selected(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def _fake_anthropic(**kwargs: Any) -> _FakeLLM:
        captured.update(kwargs)
        return _FakeLLM(
            [
                ChatResponse(
                    content="hi from claude", tool_calls=[], finish_reason="stop"
                )
            ]
        )

    monkeypatch.setattr(serve, "AnthropicLLM", _fake_anthropic)
    # The bridge's environment does not redirect a browser user's key.
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://models.example.invalid")
    with TestClient(serve.app) as client:
        resp = client.post(
            "/api/run",
            json={"brief": "hello"},
            headers={
                "X-Olmoearth-Key": "k",
                "X-LLM-Backend": "claude",
                "X-LLM-Key": "sk-ant-test",
                "X-LLM-Model": "claude-sonnet-4-6",
            },
        )
    assert resp.status_code == 200
    assert "hi from claude" in resp.text
    # the bring-your-own Anthropic key + model reached the backend constructor
    assert captured["api_key"] == "sk-ant-test"
    assert captured["model"] == "claude-sonnet-4-6"
    assert captured["base_url"] == "https://api.anthropic.com"


def test_run_uses_openai_backend_when_selected(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def _fake_openai(config: Any = None, *, openai_compat: bool = False) -> _FakeLLM:
        fake = _FakeLLM(
            [ChatResponse(content="hi from gpt", tool_calls=[], finish_reason="stop")]
        )
        if config is not None:  # the per-request build (not lifespan startup)
            captured["endpoint"] = config.endpoint
            captured["model"] = config.model
            captured["openai_compat"] = openai_compat
            captured["llm"] = fake
        return fake

    with TestClient(serve.app) as client:
        monkeypatch.setattr(serve, "OlmoEarthLLM", _fake_openai)
        resp = client.post(
            "/api/run",
            json={"brief": "hello"},
            headers={
                "X-Olmoearth-Key": "k",
                "X-LLM-Backend": "openai",
                "X-LLM-Key": "sk-openai",
                "X-LLM-Model": "gpt-4o",
            },
        )
    assert resp.status_code == 200
    assert "hi from gpt" in resp.text
    assert captured["endpoint"].startswith("https://api.openai.com")
    assert captured["model"] == "gpt-4o"
    assert captured["openai_compat"] is True
    assert captured["llm"].closed is True  # per-request hosted client is released


def test_run_rejects_non_object_body() -> None:
    with TestClient(serve.app) as client:
        resp = client.post(
            "/api/run", json=["not", "an", "object"], headers={"X-Olmoearth-Key": "k"}
        )
    assert resp.status_code == 400


def test_run_rejects_non_integer_max_turns() -> None:
    with TestClient(serve.app) as client:
        resp = client.post(
            "/api/run",
            json={"brief": "hi", "max_turns": "lots"},
            headers={"X-Olmoearth-Key": "k"},
        )
    assert resp.status_code == 400


def test_run_rejects_unknown_backend() -> None:
    with TestClient(serve.app) as client:
        resp = client.post(
            "/api/run",
            json={"brief": "hi"},
            headers={
                "X-Olmoearth-Key": "k",
                "X-LLM-Backend": "bogus",
                "X-LLM-Key": "x",
            },
        )
    assert resp.status_code == 400
    assert "unknown LLM backend" in resp.json()["detail"]


def test_llm_models_rejects_local_backend() -> None:
    with TestClient(serve.app) as client:
        resp = client.get(
            "/api/llm/models", headers={"X-LLM-Backend": "local", "X-LLM-Key": "k"}
        )
    assert resp.status_code == 400


def test_llm_models_runtime_error_is_503(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _boom(backend: str, api_key: str) -> list[str]:
        raise RuntimeError("install the anthropic extra")

    with TestClient(serve.app) as client:
        monkeypatch.setattr(serve, "_list_models", _boom)
        resp = client.get(
            "/api/llm/models", headers={"X-LLM-Backend": "claude", "X-LLM-Key": "k"}
        )
    assert resp.status_code == 503
    assert "anthropic" in resp.json()["detail"]


def test_llm_models_provider_error_is_502(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _boom(backend: str, api_key: str) -> list[str]:
        raise ValueError("upstream blew up")

    with TestClient(serve.app) as client:
        monkeypatch.setattr(serve, "_list_models", _boom)
        resp = client.get(
            "/api/llm/models", headers={"X-LLM-Backend": "openai", "X-LLM-Key": "k"}
        )
    assert resp.status_code == 502
    assert "ValueError" in resp.json()["detail"]


def test_llm_models_requires_key() -> None:
    with TestClient(serve.app) as client:
        resp = client.get("/api/llm/models", headers={"X-LLM-Backend": "openai"})
    assert resp.status_code == 400


def test_llm_models_lists_for_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _fake_list(backend: str, api_key: str) -> list[str]:
        assert backend == "gemini"
        return ["gemini-2.0-flash", "gemini-2.5-pro"]

    with TestClient(serve.app) as client:
        monkeypatch.setattr(serve, "_list_models", _fake_list)
        resp = client.get(
            "/api/llm/models",
            headers={"X-LLM-Backend": "gemini", "X-LLM-Key": "k"},
        )
    assert resp.status_code == 200
    assert resp.json()["models"] == ["gemini-2.0-flash", "gemini-2.5-pro"]


def _fake_sdk(events: list[tuple[str, str]]) -> type:
    """An SDK client stand-in that records the key and the URL it is given."""

    class _Models:
        async def list(self, **_kwargs: Any) -> Any:
            return SimpleNamespace(data=[SimpleNamespace(id="gpt-4o")])

    class _Client:
        def __init__(self, **kwargs: Any) -> None:
            given = f"{kwargs.get('api_key')} {kwargs.get('base_url')}"
            events.append(("client", given))
            self.models = _Models()

        async def close(self) -> None:
            return None

    return _Client


#: Each hosted backend, the SDK client the model list builds, and its endpoint.
_HOSTED = [
    ("claude", "anthropic.AsyncAnthropic", serve.DEFAULT_ANTHROPIC_BASE_URL),
    ("openai", "openai.AsyncOpenAI", serve._OPENAI_PROVIDERS["openai"]["base_url"]),
    ("gemini", "openai.AsyncOpenAI", serve._OPENAI_PROVIDERS["gemini"]["base_url"]),
]

#: An endpoint off the ``llm-cloud`` allowlist.
_OFF_LIST = "https://models.example.invalid/v1"


def _point_backend_at(monkeypatch: pytest.MonkeyPatch, backend: str, url: str) -> None:
    """Point a hosted backend's fixed endpoint at ``url`` for one test."""
    if backend == "claude":
        monkeypatch.setattr(serve, "DEFAULT_ANTHROPIC_BASE_URL", url)
    else:
        monkeypatch.setitem(serve._OPENAI_PROVIDERS[backend], "base_url", url)


@pytest.mark.parametrize(("backend", "sdk", "url"), _HOSTED)
def test_llm_models_checks_the_endpoint_before_the_key(
    monkeypatch: pytest.MonkeyPatch, backend: str, sdk: str, url: str
) -> None:
    """The model list checks the provider's endpoint for ``llm-cloud``, as a
    run does, then hands the SDK client the key and that endpoint: an
    ``ANTHROPIC_BASE_URL`` in the bridge's environment does not redirect it."""
    pytest.importorskip(sdk.split(".")[0])
    events: list[tuple[str, str]] = []
    real = serve.egress.validate_endpoint

    def _check(endpoint: str, capability: str, **kwargs: Any) -> str:
        events.append(("check", f"{capability} {endpoint}"))
        return real(endpoint, capability, **kwargs)

    monkeypatch.setenv("ANTHROPIC_BASE_URL", _OFF_LIST)
    with TestClient(serve.app) as client:
        # Patched after startup, so only the request's own calls are recorded.
        monkeypatch.setattr(serve.egress, "validate_endpoint", _check)
        monkeypatch.setattr(sdk, _fake_sdk(events))
        resp = client.get(
            "/api/llm/models", headers={"X-LLM-Backend": backend, "X-LLM-Key": "k"}
        )
    assert resp.status_code == 200, resp.text
    assert events == [("check", f"llm-cloud {url}"), ("client", f"k {url}")]


@pytest.mark.parametrize(("backend", "sdk"), [h[:2] for h in _HOSTED])
def test_llm_models_refuses_an_endpoint_off_the_allowlist(
    monkeypatch: pytest.MonkeyPatch, backend: str, sdk: str
) -> None:
    """In ``enforce`` mode a provider endpoint off the ``llm-cloud`` allowlist
    is refused with HTTP 403, and no SDK client is built with the key."""
    pytest.importorskip(sdk.split(".")[0])
    events: list[tuple[str, str]] = []
    _point_backend_at(monkeypatch, backend, _OFF_LIST)
    monkeypatch.delenv("OLMOEARTH_EGRESS_ALLOW", raising=False)
    with TestClient(serve.app) as client:
        monkeypatch.setenv("OLMOEARTH_EGRESS", "enforce")
        monkeypatch.setattr(sdk, _fake_sdk(events))
        resp = client.get(
            "/api/llm/models", headers={"X-LLM-Backend": backend, "X-LLM-Key": "k"}
        )
    assert resp.status_code == 403, resp.text
    assert "host not on 'llm-cloud' allowlist" in resp.json()["detail"]
    assert events == []


@pytest.mark.parametrize("backend", ["claude", "openai", "gemini"])
def test_run_refuses_a_backend_endpoint_off_the_allowlist(
    monkeypatch: pytest.MonkeyPatch, backend: str
) -> None:
    """A run checks the hosted backend's endpoint as the model list does: in
    ``enforce`` mode a host off the allowlist is refused with HTTP 403 before
    a client is built with the key."""
    built: list[Any] = []
    _point_backend_at(monkeypatch, backend, _OFF_LIST)
    monkeypatch.delenv("OLMOEARTH_EGRESS_ALLOW", raising=False)
    with TestClient(serve.app) as client:
        monkeypatch.setenv("OLMOEARTH_EGRESS", "enforce")
        monkeypatch.setattr(serve, "AnthropicLLM", lambda *a, **k: built.append(k))
        monkeypatch.setattr(serve, "OlmoEarthLLM", lambda *a, **k: built.append(a))
        resp = client.post(
            "/api/run",
            json={"brief": "hi"},
            headers={
                "X-Olmoearth-Key": "k",
                "X-LLM-Backend": backend,
                "X-LLM-Key": "sk",
            },
        )
    assert resp.status_code == 403, resp.text
    assert "host not on 'llm-cloud' allowlist" in resp.json()["detail"]
    assert built == []


class _FakeEnv:
    def __init__(self, records: list[dict[str, Any]]) -> None:
        self.records = records


class _FakeStudio:
    """Async-context StudioClient stand-in returning canned envelopes."""

    def __init__(self, *_a: Any, **_k: Any) -> None: ...

    async def __aenter__(self) -> "_FakeStudio":
        return self

    async def __aexit__(self, *_a: Any) -> None: ...

    async def aclose(self) -> None: ...

    async def get_bytes(self, url: str, *, timeout: float | None = None) -> Any:
        import httpx

        return httpx.Response(
            200, content=b"PNGDATA", headers={"content-type": "image/png"}
        )

    async def search_predictions(
        self, *, project_id: str, limit: int = 200
    ) -> _FakeEnv:
        return _FakeEnv(
            [
                {
                    "id": "p1",
                    "name": "Karst run",
                    "status": "completed",
                    "model_id": "m-aaa",
                },
                {
                    "id": "p2",
                    "name": "Karst run 2",
                    "status": "running",
                    "model_id": "m-aaa",
                },
                {"id": "p3", "name": "Other", "status": "failed", "model_id": "m-bbb"},
            ]
        )

    async def search_prediction_results(
        self, *, prediction_id: str, limit: int = 200
    ) -> _FakeEnv:
        return _FakeEnv(
            [
                {
                    "id": "r1",
                    "prediction_id": prediction_id,
                    "property_names": ["karst_score"],
                    "file_formats": ["png"],
                    "tile_urls": ["t1", "t2"],
                }
            ]
        )

    async def search_models(self, *, limit: int = 200, offset: int = 0) -> _FakeEnv:
        return _FakeEnv(
            [
                {
                    "id": "m-aaa",
                    "name": "Karst fine-tune",
                    "model_type": "fine_tuned",
                    "wizard_answers": {
                        "encoder_variant": "nano",
                        "foundation_model_version": "v1",
                    },
                },
                {
                    "id": "m-bbb",
                    "name": "Karst embeddings",
                    "model_type": "embeddings",
                    "wizard_answers": {"encoder_variant": "base"},
                },
            ]
        )

    async def create_area(
        self, *, name: str, geom: dict[str, Any], project_id: str
    ) -> dict[str, Any]:
        return {"id": "area-1", "name": name, "project_id": project_id}

    async def search_areas(self, *, project_id: str, limit: int = 200) -> _FakeEnv:
        return _FakeEnv(
            [{"id": "area-1", "name": "Saved AOI", "project_id": project_id}]
        )

    async def get_prediction_result(self, result_id: str) -> dict[str, Any]:
        return {
            "id": result_id,
            "tile_urls": ["/api/v1/prediction-results/r/tiles/{z}/{x}/{y}.png"],
            "property_names": ["score"],
            "result_metadata": {
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [-78.0, 40.0],
                            [-77.0, 40.0],
                            [-77.0, 41.0],
                            [-78.0, 41.0],
                            [-78.0, 40.0],
                        ]
                    ],
                },
            },
        }

    async def get_area(self, area_id: str) -> dict[str, Any]:
        return {
            "id": area_id,
            "name": "Saved AOI",
            "project_id": "p1",
            "geom": {
                "type": "Polygon",
                "coordinates": [
                    [[-2.0, -2.0], [2.0, -2.0], [2.0, 2.0], [-2.0, 2.0], [-2.0, -2.0]]
                ],
            },
        }


_SQUARE = {
    "type": "Polygon",
    "coordinates": [[[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0]]],
}


def test_create_area_stores_and_returns_bbox(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(serve, "StudioClient", _FakeStudio)
    with TestClient(serve.app) as client:
        resp = client.post(
            "/api/areas",
            json={"name": "Drawn AOI", "geom": _SQUARE, "project_id": "p1"},
            headers={"X-Olmoearth-Key": "k"},
        )
    assert resp.status_code == 200
    area = resp.json()["area"]
    assert area["id"] == "area-1"
    assert area["project_id"] == "p1"
    assert area["bbox"] == [-1.0, -1.0, 1.0, 1.0]


def test_create_area_refused_by_studio_is_a_403(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A read-only key's refusal reads as such, not as a 502 outage."""
    from olmoearth_agent.studio.client import StudioAccessError

    class _ReadOnlyStudio(_FakeStudio):
        async def create_area(
            self, *, name: str, geom: dict[str, Any], project_id: str
        ) -> dict[str, Any]:
            raise StudioAccessError("POST", "/areas", 403, "permission_error")

    monkeypatch.setattr(serve, "StudioClient", _ReadOnlyStudio)
    with TestClient(serve.app) as client:
        resp = client.post(
            "/api/areas",
            json={"name": "Drawn AOI", "geom": _SQUARE, "project_id": "p1"},
            headers={"X-Olmoearth-Key": "k"},
        )
    assert resp.status_code == 403
    detail = resp.json()["detail"]
    assert "Studio refused POST /areas (HTTP 403, permission_error)" in detail
    assert "read-only" in detail and "nothing was created" in detail


def test_create_area_requires_key() -> None:
    with TestClient(serve.app) as client:
        resp = client.post(
            "/api/areas", json={"name": "x", "geom": _SQUARE, "project_id": "p1"}
        )
    assert resp.status_code == 400


def test_create_area_rejects_non_polygon(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(serve, "StudioClient", _FakeStudio)
    with TestClient(serve.app) as client:
        resp = client.post(
            "/api/areas",
            json={
                "name": "x",
                "geom": {"type": "Point", "coordinates": [0, 0]},
                "project_id": "p1",
            },
            headers={"X-Olmoearth-Key": "k"},
        )
    assert resp.status_code == 400
    assert "Polygon" in resp.json()["detail"]


def test_create_area_requires_project_id(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(serve, "StudioClient", _FakeStudio)
    with TestClient(serve.app) as client:
        resp = client.post(
            "/api/areas",
            json={"name": "x", "geom": _SQUARE},
            headers={"X-Olmoearth-Key": "k"},
        )
    assert resp.status_code == 400


def test_read_cache_serves_repeat_without_requerying(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"n": 0}

    class _CountingCtx:
        def __init__(self, *_a: Any, **_k: Any) -> None: ...
        async def __aenter__(self) -> "_CountingCtx":
            return self

        async def __aexit__(self, *_a: Any) -> None: ...
        async def load_context(self) -> Any:
            calls["n"] += 1
            from olmoearth_agent.types import StudioContext

            return StudioContext(
                user_id="u", user_name="U", organization="O", projects=[]
            )

    monkeypatch.setattr(serve, "StudioClient", _CountingCtx)
    with TestClient(serve.app) as client:
        h = {"X-Olmoearth-Key": "k"}
        client.get("/api/projects", headers=h)
        client.get("/api/projects", headers=h)  # served from the read cache
    assert calls["n"] == 1


def test_create_area_invalidates_areas_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(serve, "StudioClient", _FakeStudio)
    with TestClient(serve.app) as client:
        h = {"X-Olmoearth-Key": "k"}
        client.get("/api/projects/p1/areas", headers=h)  # populate cache
        assert any("areas|" in key and "|p1" in key for key in serve._READ_CACHE)
        client.post(
            "/api/areas",
            json={"name": "x", "geom": _SQUARE, "project_id": "p1"},
            headers=h,
        )
        # the create dropped p1's cached area list
        assert not any("areas|" in key and "|p1" in key for key in serve._READ_CACHE)


def test_project_areas_list(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(serve, "StudioClient", _FakeStudio)
    with TestClient(serve.app) as client:
        resp = client.get("/api/projects/p1/areas", headers={"X-Olmoearth-Key": "k"})
    assert resp.status_code == 200
    areas = resp.json()["areas"]
    assert areas[0]["id"] == "area-1"
    assert areas[0]["name"] == "Saved AOI"


def test_get_area_returns_geom_and_bbox(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(serve, "StudioClient", _FakeStudio)
    with TestClient(serve.app) as client:
        resp = client.get("/api/areas/area-1", headers={"X-Olmoearth-Key": "k"})
    assert resp.status_code == 200
    area = resp.json()["area"]
    assert area["id"] == "area-1"
    assert area["geom"]["type"] == "Polygon"
    assert area["bbox"] == [-2.0, -2.0, 2.0, 2.0]


def test_get_area_requires_key() -> None:
    with TestClient(serve.app) as client:
        resp = client.get("/api/areas/area-1")
    assert resp.status_code == 400


def test_result_extent_returns_bbox(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(serve, "StudioClient", _FakeStudio)
    with TestClient(serve.app) as client:
        resp = client.get("/api/results/r1/extent", headers={"X-Olmoearth-Key": "k"})
    assert resp.status_code == 200
    ext = resp.json()["extent"]
    assert ext["bbox"] == [-78.0, 40.0, -77.0, 41.0]
    assert ext["tile_url"].endswith(".png")
    assert ext["property"] == "score"


def test_result_extent_requires_key() -> None:
    with TestClient(serve.app) as client:
        resp = client.get("/api/results/r1/extent")
    assert resp.status_code == 400


def test_pixel_value_proxy_requires_key() -> None:
    with TestClient(serve.app) as client:
        resp = client.get("/api/pixel-value?result_id=r1&lon=1&lat=2")
    assert resp.status_code == 400


def test_pixel_value_proxy_requires_coords() -> None:
    with TestClient(serve.app) as client:
        resp = client.get(
            "/api/pixel-value?result_id=r1", headers={"X-Olmoearth-Key": "k"}
        )
    assert resp.status_code == 400


def test_pixel_value_proxy_extracts_band(monkeypatch: pytest.MonkeyPatch) -> None:
    class _PV(_FakeStudio):
        async def pixel_value(
            self, result_id: str, lon: float, lat: float
        ) -> dict[str, Any]:
            return {
                "bands": [
                    {
                        "property_name": "karst",
                        "raw_value": 0.42,
                        "classification": None,
                    }
                ]
            }

    monkeypatch.setattr(serve, "StudioClient", _PV)
    with TestClient(serve.app) as client:
        resp = client.get(
            "/api/pixel-value?result_id=r1&lon=-77.6&lat=40.8",
            headers={"X-Olmoearth-Key": "k"},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["value"] == 0.42
    assert body["property"] == "karst"
    assert body["categorical"] is False


def test_pixel_value_proxy_blanks_the_nodata_sentinel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Studio returns no-data as a value (-1 on a [0, 1] band); the difference
    scan must skip that cell, not paint the sentinel as a model output."""

    class _PV(_FakeStudio):
        async def pixel_value(
            self, result_id: str, lon: float, lat: float
        ) -> dict[str, Any]:
            return {
                "bands": [
                    {
                        "band_index": 1,
                        "property_name": "sample_karst_score",
                        "raw_value": -1.0,
                        "classification": None,
                        "regression": {
                            "min_value": 0.0,
                            "max_value": 1.0,
                            "colormap_name": "viridis",
                        },
                    }
                ]
            }

    monkeypatch.setattr(serve, "StudioClient", _PV)
    with TestClient(serve.app) as client:
        resp = client.get(
            "/api/pixel-value?result_id=nd1&lon=-77.6&lat=40.8",
            headers={"X-Olmoearth-Key": "k"},
        )
    body = resp.json()
    assert body["value"] is None
    assert body["nodata"] is True
    assert body["property"] == "sample_karst_score"


def test_pixel_value_proxy_caches(monkeypatch: pytest.MonkeyPatch) -> None:
    serve._PV_CACHE.clear()
    calls = {"n": 0}

    class _Counting(_FakeStudio):
        async def pixel_value(
            self, result_id: str, lon: float, lat: float
        ) -> dict[str, Any]:
            calls["n"] += 1
            return {
                "bands": [
                    {"property_name": "k", "raw_value": 0.7, "classification": None}
                ]
            }

    monkeypatch.setattr(serve, "StudioClient", _Counting)
    with TestClient(serve.app) as client:
        h = {"X-Olmoearth-Key": "k"}
        u = "/api/pixel-value?result_id=cache1&lon=1.0&lat=2.0"
        a = client.get(u, headers=h)
        b = client.get(u, headers=h)  # identical -> should hit cache
    assert a.json()["value"] == 0.7
    assert b.json()["value"] == 0.7
    assert calls["n"] == 1  # Studio queried only once


def test_tile_proxy_requires_key() -> None:
    with TestClient(serve.app) as client:
        resp = client.get(
            "/api/tile/5/3/7?src=%2Ffoo%2F%7Bz%7D%2F%7Bx%7D%2F%7By%7D.png"
        )
    assert resp.status_code == 400


def test_tile_proxy_requires_src() -> None:
    with TestClient(serve.app) as client:
        resp = client.get("/api/tile/5/3/7", headers={"X-Olmoearth-Key": "k"})
    assert resp.status_code == 400


def test_tile_proxy_rejects_foreign_host() -> None:
    # An absolute src pointing off the Studio host must be blocked (SSRF guard).
    import urllib.parse

    src = urllib.parse.quote("https://evil.example.com/{z}/{x}/{y}.png", safe="")
    with TestClient(serve.app) as client:
        resp = client.get(
            f"/api/tile/5/3/7?src={src}", headers={"X-Olmoearth-Key": "k"}
        )
    assert resp.status_code == 403


def test_tile_proxy_fetches_via_pooled_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """A valid tile request fetches bytes through the pooled StudioClient."""
    import urllib.parse

    monkeypatch.setattr(serve, "StudioClient", _FakeStudio)
    src = urllib.parse.quote(
        "https://olmoearth.allenai.org/api/v1/prediction-results/r1/tiles/{z}/{x}/{y}.png",
        safe="",
    )
    with TestClient(serve.app) as client:
        resp = client.get(
            f"/api/tile/5/3/7?src={src}", headers={"X-Olmoearth-Key": "k"}
        )
    assert resp.status_code == 200
    assert resp.content == b"PNGDATA"
    assert resp.headers["content-type"].startswith("image/png")


def test_studio_client_pool_reuses_one_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """A burst of pixel-value calls for one key reuses a single pooled client
    (the fix: no fresh TLS handshake per point)."""
    created = {"n": 0}

    class _Counting(_FakeStudio):
        def __init__(self, *a: Any, **k: Any) -> None:
            super().__init__(*a, **k)
            created["n"] += 1

        async def pixel_value(
            self, result_id: str, lon: float, lat: float
        ) -> dict[str, Any]:
            return {
                "bands": [
                    {"property_name": "k", "raw_value": 0.5, "classification": None}
                ]
            }

    monkeypatch.setattr(serve, "StudioClient", _Counting)
    with TestClient(serve.app) as client:
        h = {"X-Olmoearth-Key": "k"}
        # distinct coords so neither call is a pixel-value cache hit
        client.get("/api/pixel-value?result_id=r1&lon=1.0&lat=2.0", headers=h)
        client.get("/api/pixel-value?result_id=r1&lon=3.0&lat=4.0", headers=h)
    assert created["n"] == 1  # one StudioClient created and reused across both


def test_static_assets_are_revalidated() -> None:
    """The web UI is served no-cache so edited ES modules are never stale."""
    with TestClient(serve.app) as client:
        resp = client.get("/")
    assert resp.headers.get("cache-control") == "no-cache"


def test_project_predictions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(serve, "StudioClient", _FakeStudio)
    with TestClient(serve.app) as client:
        resp = client.get(
            "/api/projects/proj1/predictions", headers={"X-Olmoearth-Key": "k"}
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    assert [p["id"] for p in data["predictions"]] == ["p1", "p2", "p3"]
    assert data["predictions"][0]["model_id"] == "m-aaa"
    assert data["predictions"][1]["status"] == "running"
    # The model_id level is enriched with the model's real name + type.
    models = data["models"]
    assert models["m-aaa"]["name"] == "Karst fine-tune"
    assert models["m-aaa"]["model_type"] == "fine_tuned"
    assert models["m-aaa"]["foundation"] == "OlmoEarth Nano v1"
    assert models["m-bbb"]["model_type"] == "embeddings"
    assert models["m-bbb"]["foundation"] == "OlmoEarth Base"  # no version -> name only


def test_project_predictions_model_enrichment_is_best_effort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failing /models lookup must not break the tree: predictions still load."""

    class _NoModelsStudio(_FakeStudio):
        async def search_models(self, *, limit: int = 200, offset: int = 0) -> _FakeEnv:
            raise RuntimeError("models endpoint down")

    monkeypatch.setattr(serve, "StudioClient", _NoModelsStudio)
    with TestClient(serve.app) as client:
        resp = client.get(
            "/api/projects/proj1/predictions", headers={"X-Olmoearth-Key": "k"}
        )
    assert resp.status_code == 200
    data = resp.json()
    assert [p["id"] for p in data["predictions"]] == ["p1", "p2", "p3"]
    assert data["models"] == {}  # enrichment degraded gracefully


def test_prediction_results(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(serve, "StudioClient", _FakeStudio)
    with TestClient(serve.app) as client:
        resp = client.get(
            "/api/predictions/p1/results", headers={"X-Olmoearth-Key": "k"}
        )
    assert resp.status_code == 200
    results = resp.json()["results"]
    assert results[0]["tiles"] == 2
    assert results[0]["property_names"] == ["karst_score"]
    # Studio's field is the list file_formats; it has no singular file_format.
    assert results[0]["file_formats"] == ["png"]
    assert "file_format" not in results[0]


def test_tree_endpoints_require_key() -> None:
    with TestClient(serve.app) as client:
        assert client.get("/api/projects/p1/predictions").status_code == 400
        assert client.get("/api/predictions/p1/results").status_code == 400
