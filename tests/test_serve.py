# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Tests for the FastAPI bridge (serve.py) with a fake LLM: no live services."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from olmoearth_agent import serve  # noqa: E402
from olmoearth_agent.llm.types import ChatResponse, Message  # noqa: E402


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


def test_health_reports_claude_available() -> None:
    with TestClient(serve.app) as client:
        resp = client.get("/api/health")
    assert resp.status_code == 200
    assert isinstance(resp.json()["claude_available"], bool)


def test_health_reports_local_llm_up(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _up(_endpoint: str) -> bool:
        return True

    monkeypatch.setattr(serve, "_local_llm_up", _up)
    with TestClient(serve.app) as client:
        body = client.get("/api/health").json()
    assert body["llm_local_up"] is True


def test_health_reports_local_llm_down(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _down(_endpoint: str) -> bool:
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


class _FakeEnv:
    def __init__(self, records: list[dict[str, Any]]) -> None:
        self.records = records


class _FakeStudio:
    """Async-context StudioClient stand-in returning canned envelopes."""

    def __init__(self, *_a: Any, **_k: Any) -> None: ...

    async def __aenter__(self) -> "_FakeStudio":
        return self

    async def __aexit__(self, *_a: Any) -> None: ...

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
                    "file_format": "png",
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

    async def search_areas(
        self, *, project_id: str, limit: int = 200
    ) -> _FakeEnv:
        return _FakeEnv([{"id": "area-1", "name": "Saved AOI", "project_id": project_id}])

    async def get_area(self, area_id: str) -> dict[str, Any]:
        return {
            "id": area_id,
            "name": "Saved AOI",
            "project_id": "p1",
            "geom": {
                "type": "Polygon",
                "coordinates": [[[-2.0, -2.0], [2.0, -2.0], [2.0, 2.0], [-2.0, 2.0], [-2.0, -2.0]]],
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


def test_project_areas_list(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(serve, "StudioClient", _FakeStudio)
    with TestClient(serve.app) as client:
        resp = client.get(
            "/api/projects/p1/areas", headers={"X-Olmoearth-Key": "k"}
        )
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
    assert results[0]["file_format"] == "png"


def test_tree_endpoints_require_key() -> None:
    with TestClient(serve.app) as client:
        assert client.get("/api/projects/p1/predictions").status_code == 400
        assert client.get("/api/predictions/p1/results").status_code == 400
