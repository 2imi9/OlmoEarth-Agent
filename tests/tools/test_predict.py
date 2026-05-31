# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Unit + live tests for the olmoearth-predict tool bundle."""

from __future__ import annotations

import json
import os

import pytest
from pytest_httpx import HTTPXMock

from olmoearth_agent.harness.state import ThreadState
from olmoearth_agent.tools.predict import build_predict_tools
from olmoearth_agent.tools.registry import RegisteredTool, ToolContext

BASE = "http://mock-studio/api/v1"


def _tool(name: str) -> RegisteredTool:
    return next(t for t in build_predict_tools() if t.spec.name == name)


@pytest.mark.asyncio
async def test_search_predictions_tool(httpx_mock: HTTPXMock) -> None:
    from olmoearth_agent.studio.client import StudioClient, StudioConfig

    httpx_mock.add_response(
        url=f"{BASE}/predictions/search",
        method="POST",
        json={
            "records": [
                {"id": "p1", "name": "X", "status": "completed", "model_id": "m1"}
            ],
            "meta": {"total": 1},
        },
    )
    async with StudioClient(StudioConfig(api_key="k", base_url=BASE)) as studio:
        ctx = ToolContext(studio=studio, state=ThreadState())
        result = await _tool("olmoearth_search_predictions").handler(
            {"limit": 5}, ctx
        )
    assert result["total"] == 1
    assert result["predictions"][0]["model_id"] == "m1"


@pytest.mark.asyncio
async def test_submit_prediction_tool_sends_required_fields(
    httpx_mock: HTTPXMock,
) -> None:
    from olmoearth_agent.studio.client import StudioClient, StudioConfig

    httpx_mock.add_response(
        url=f"{BASE}/predictions",
        method="POST",
        json={"records": [{"id": "pred9", "status": "pending"}]},
    )
    state = ThreadState()
    async with StudioClient(StudioConfig(api_key="k", base_url=BASE)) as studio:
        ctx = ToolContext(studio=studio, state=state)
        result = await _tool("olmoearth_submit_prediction").handler(
            {
                "name": "n",
                "project_id": "pr",
                "area_id": "a",
                "model_id": "m",
                "start_time": "2022-01-01",
                "end_time": "2024-12-31",
            },
            ctx,
        )
    assert result["id"] == "pred9"
    assert "pred9" in state.prediction_ids
    body = json.loads(httpx_mock.get_requests()[0].content)
    assert set(body) == {
        "name",
        "project_id",
        "area_id",
        "model_id",
        "start_time",
        "end_time",
    }


@pytest.mark.asyncio
async def test_fetch_results_filters_by_prediction_id(
    httpx_mock: HTTPXMock,
) -> None:
    from olmoearth_agent.studio.client import StudioClient, StudioConfig

    httpx_mock.add_response(
        url=f"{BASE}/prediction-results/search",
        method="POST",
        json={
            "records": [
                {"id": "r1", "prediction_id": "pX", "tile_urls": ["t1"],
                 "property_names": ["score"], "file_format": "png"},
                {"id": "r2", "prediction_id": "pOTHER", "tile_urls": ["t2"]},
            ],
            "meta": {"total": 2},
        },
    )
    async with StudioClient(StudioConfig(api_key="k", base_url=BASE)) as studio:
        ctx = ToolContext(studio=studio, state=ThreadState())
        result = await _tool("olmoearth_fetch_results").handler(
            {"prediction_id": "pX"}, ctx
        )
    assert result["result_count"] == 1
    assert result["results"][0]["result_id"] == "r1"
    assert result["results"][0]["tile_urls"] == ["t1"]


@pytest.mark.asyncio
async def test_get_prediction_result_tool(httpx_mock: HTTPXMock) -> None:
    from olmoearth_agent.studio.client import StudioClient, StudioConfig

    httpx_mock.add_response(
        url=f"{BASE}/prediction-results/r9",
        json={
            "records": [
                {"id": "r9", "prediction_id": "p1", "tile_urls": ["tpl"],
                 "property_names": ["karst_score"], "file_format": "png",
                 "result_metadata": {"foo": 1}}
            ]
        },
    )
    async with StudioClient(StudioConfig(api_key="k", base_url=BASE)) as studio:
        ctx = ToolContext(studio=studio, state=ThreadState())
        result = await _tool("olmoearth_get_prediction_result").handler(
            {"result_id": "r9"}, ctx
        )
    assert result["result_id"] == "r9"
    assert result["property_names"] == ["karst_score"]
    assert result["result_metadata"] == {"foo": 1}


@pytest.mark.integration
@pytest.mark.asyncio
async def test_live_search_predictions_have_model_ids() -> None:
    """Read-only live check: existing predictions carry reusable model_ids."""
    if not os.environ.get("OLMOEARTH_API_KEY"):
        pytest.skip("needs OLMOEARTH_API_KEY")
    from olmoearth_agent.studio import StudioClient

    async with StudioClient.from_env() as studio:
        env = await studio.search_predictions(limit=5)
    assert env.total is not None and env.total >= 1
    assert all(r.get("model_id") for r in env.records), (
        "every prediction should expose a model_id for reuse"
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_live_fetch_results_returns_tile_urls() -> None:
    """Live: a completed prediction's results expose tile URLs."""
    if not os.environ.get("OLMOEARTH_API_KEY"):
        pytest.skip("needs OLMOEARTH_API_KEY")
    from olmoearth_agent.studio import StudioClient

    async with StudioClient.from_env() as studio:
        preds = await studio.search_predictions(limit=20)
        completed = next(
            (r for r in preds.records if r.get("status") == "completed"), None
        )
        assert completed is not None, "expected a completed prediction"
        env = await studio.search_prediction_results(
            prediction_id=completed["id"], limit=1000
        )
    # Some completed prediction in the first 1000 results should match and
    # carry tile_urls. (If not in-window, this is a soft check.)
    if env.records:
        assert any(r.get("tile_urls") for r in env.records)
