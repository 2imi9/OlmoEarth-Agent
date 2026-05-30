# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Tests for the FastAPI bridge (serve.py) with a fake LLM — no live services."""

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
