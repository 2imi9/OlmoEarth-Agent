# SPDX-License-Identifier: Apache-2.0
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

    async def chat(
        self, messages: list[Message], *, tools: Any = None, **_kw: Any
    ) -> ChatResponse:
        return self._responses.pop(0)


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
