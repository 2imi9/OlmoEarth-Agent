# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Tests for the olmoearth-uncertainty tools (AOA + ensemble confidence)."""

from __future__ import annotations

import re
from urllib.parse import parse_qs, urlparse

import httpx
import pytest
from pytest_httpx import HTTPXMock

from olmoearth_agent.harness.state import ThreadState
from olmoearth_agent.llm.types import ToolCall
from olmoearth_agent.studio.client import StudioClient, StudioConfig
from olmoearth_agent.tools.registry import RegisteredTool, ToolContext, ToolRegistry
from olmoearth_agent.tools.uncertainty import build_uncertainty_tools

_TRAIN = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]]

BASE = "http://mock-studio/api/v1"


def _ctx() -> ToolContext:
    return ToolContext(studio=None, state=ThreadState())  # type: ignore[arg-type]


def _tool(name: str) -> RegisteredTool:
    return next(t for t in build_uncertainty_tools() if t.spec.name == name)


def _result_with_geom(rid: str, coords: list) -> dict:
    return {
        "records": [
            {"id": rid, "result_metadata": {"geometry": {"type": "Polygon",
             "coordinates": coords}}}
        ]
    }


_BOX_0_10 = [[[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]]


@pytest.mark.asyncio
async def test_tool_flags_in_distribution() -> None:
    tool = build_uncertainty_tools()[0]
    result = await tool.handler(
        {"train_features": _TRAIN, "new_features": [[0.5, 0.5]]}, _ctx()
    )
    assert result["inside_aoa"] == [True]
    assert result["verdict"] == "within-AOA"


@pytest.mark.asyncio
async def test_tool_flags_out_of_distribution() -> None:
    tool = build_uncertainty_tools()[0]
    result = await tool.handler(
        {"train_features": _TRAIN, "new_features": [[1000.0, 1000.0]]}, _ctx()
    )
    assert result["ood_fraction"] == 1.0
    assert result["verdict"] == "mostly-OOD"


@pytest.mark.asyncio
async def test_invalid_input_surfaces_to_model_via_dispatch() -> None:
    registry = ToolRegistry()
    registry.register_all(build_uncertainty_tools())
    result = await registry.dispatch(
        ToolCall(
            id="c1",
            name="olmoearth_area_of_applicability",
            arguments={"train_features": [[0.0, 0.0]], "new_features": [[1.0, 1.0]]},
        ),
        _ctx(),
    )
    assert result["ok"] is False
    assert "2" in result["error"]


@pytest.mark.asyncio
async def test_ensemble_uncertainty_regression_disagreement(
    httpx_mock: HTTPXMock,
) -> None:
    # Two results share the 0..10 extent; B = A + 2 everywhere -> real spread.
    httpx_mock.add_response(
        url=f"{BASE}/prediction-results/a1", json=_result_with_geom("a1", _BOX_0_10)
    )
    httpx_mock.add_response(
        url=f"{BASE}/prediction-results/b1", json=_result_with_geom("b1", _BOX_0_10)
    )

    def pixel(request: httpx.Request) -> httpx.Response:
        q = parse_qs(urlparse(str(request.url)).query)
        lon = float(q["lon"][0])
        offset = 2.0 if "/b1/" in str(request.url) else 0.0
        return httpx.Response(
            200,
            json={"records": [{"bands": [{"property_name": "score",
                   "raw_value": round(lon + offset, 6), "classification": None}]}]},
        )

    httpx_mock.add_callback(pixel, url=re.compile(r".*/pixel-value\?.*"), is_reusable=True)

    async with StudioClient(StudioConfig(api_key="k", base_url=BASE)) as studio:
        ctx = ToolContext(studio=studio, state=ThreadState())
        out = await _tool("olmoearth_ensemble_uncertainty").handler(
            {"result_ids": ["a1", "b1"], "grid": 2}, ctx
        )
    assert out["comparable"] is True
    assert out["value_type"] == "regression"
    assert out["n_results"] == 2
    assert out["n_points"] == 4  # 2x2 grid, all valid
    # every point had exactly the two ensemble members
    assert all(p["n"] == 2 for p in out["per_point"])
    # real spread -> confidence strictly below 1
    assert out["summary"]["mean_confidence"] is not None
    assert out["summary"]["mean_confidence"] < 1.0


@pytest.mark.asyncio
async def test_ensemble_uncertainty_categorical_auto_detect(
    httpx_mock: HTTPXMock,
) -> None:
    httpx_mock.add_response(
        url=f"{BASE}/prediction-results/a1", json=_result_with_geom("a1", _BOX_0_10)
    )
    httpx_mock.add_response(
        url=f"{BASE}/prediction-results/b1", json=_result_with_geom("b1", _BOX_0_10)
    )

    def pixel(request: httpx.Request) -> httpx.Response:
        cls = "wetland" if "/b1/" in str(request.url) else "forest"
        return httpx.Response(
            200,
            json={"records": [{"bands": [{"property_name": "landcover",
                   "raw_value": None, "classification": cls}]}]},
        )

    httpx_mock.add_callback(pixel, url=re.compile(r".*/pixel-value\?.*"), is_reusable=True)

    async with StudioClient(StudioConfig(api_key="k", base_url=BASE)) as studio:
        ctx = ToolContext(studio=studio, state=ThreadState())
        out = await _tool("olmoearth_ensemble_uncertainty").handler(
            {"result_ids": ["a1", "b1"], "grid": 2}, ctx
        )
    assert out["comparable"] is True
    assert out["value_type"] == "categorical"  # auto-detected from classification
    # two members always disagree -> max entropy, zero confidence
    assert out["summary"]["mean_entropy"] == pytest.approx(1.0)
    assert out["summary"]["mean_confidence"] == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_ensemble_uncertainty_needs_two_results() -> None:
    out = await _tool("olmoearth_ensemble_uncertainty").handler(
        {"result_ids": ["only-one"]}, _ctx()
    )
    assert out["comparable"] is False
    assert ">= 2" in out["reason"]


@pytest.mark.asyncio
async def test_ensemble_uncertainty_rejects_duplicate_ids() -> None:
    # A duplicate id would re-read ONE deterministic result and fake zero
    # disagreement; it must be rejected before any sampling (no studio needed).
    out = await _tool("olmoearth_ensemble_uncertainty").handler(
        {"result_ids": ["same", "same"]}, _ctx()
    )
    assert out["comparable"] is False
    assert "DISTINCT" in out["reason"]


@pytest.mark.asyncio
async def test_ensemble_uncertainty_mixed_band_types_clean_answer(
    httpx_mock: HTTPXMock,
) -> None:
    # a1 is regression, b1 is categorical at the same property: auto-detect
    # picks regression from a1, then a categorical label cannot be float()'d.
    # The tool must answer cleanly (comparable=False), not raise.
    httpx_mock.add_response(
        url=f"{BASE}/prediction-results/a1", json=_result_with_geom("a1", _BOX_0_10)
    )
    httpx_mock.add_response(
        url=f"{BASE}/prediction-results/b1", json=_result_with_geom("b1", _BOX_0_10)
    )

    def pixel(request: httpx.Request) -> httpx.Response:
        if "/b1/" in str(request.url):
            band = {"property_name": "x", "raw_value": None,
                    "classification": "forest"}
        else:
            band = {"property_name": "x", "raw_value": 1.0, "classification": None}
        return httpx.Response(200, json={"records": [{"bands": [band]}]})

    httpx_mock.add_callback(pixel, url=re.compile(r".*/pixel-value\?.*"), is_reusable=True)

    async with StudioClient(StudioConfig(api_key="k", base_url=BASE)) as studio:
        ctx = ToolContext(studio=studio, state=ThreadState())
        out = await _tool("olmoearth_ensemble_uncertainty").handler(
            {"result_ids": ["a1", "b1"], "grid": 2}, ctx
        )
    assert out["comparable"] is False
    assert "value_type" in out["reason"]


@pytest.mark.asyncio
async def test_ensemble_uncertainty_no_shared_extent(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url=f"{BASE}/prediction-results/a1",
        json=_result_with_geom("a1", [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]),
    )
    httpx_mock.add_response(
        url=f"{BASE}/prediction-results/b1",
        json=_result_with_geom("b1", [[[5, 5], [6, 5], [6, 6], [5, 6], [5, 5]]]),
    )
    async with StudioClient(StudioConfig(api_key="k", base_url=BASE)) as studio:
        ctx = ToolContext(studio=studio, state=ThreadState())
        out = await _tool("olmoearth_ensemble_uncertainty").handler(
            {"result_ids": ["a1", "b1"]}, ctx
        )
    assert out["comparable"] is False
    assert "common extent" in out["reason"]
