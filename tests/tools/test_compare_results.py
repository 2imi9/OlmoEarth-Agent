# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Unit test for olmoearth_compare_results (mocked Studio)."""

from __future__ import annotations

import re
from urllib.parse import parse_qs, urlparse

import httpx
import pytest
from pytest_httpx import HTTPXMock

from olmoearth_agent.harness.state import ThreadState
from olmoearth_agent.studio.client import StudioClient, StudioConfig
from olmoearth_agent.tools.predict import build_predict_tools
from olmoearth_agent.tools.registry import RegisteredTool, ToolContext

BASE = "http://mock-studio/api/v1"


def _tool(name: str) -> RegisteredTool:
    return next(t for t in build_predict_tools() if t.spec.name == name)


def _result_with_geom(rid: str) -> dict:
    return {
        "records": [
            {
                "id": rid,
                "result_metadata": {
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]],
                    }
                },
            }
        ]
    }


@pytest.mark.asyncio
async def test_compare_results_quantifies_divergence(httpx_mock: HTTPXMock) -> None:
    # Both results share the same 0..10 extent.
    httpx_mock.add_response(url=f"{BASE}/prediction-results/a1", json=_result_with_geom("a1"))
    httpx_mock.add_response(url=f"{BASE}/prediction-results/b1", json=_result_with_geom("b1"))

    # pixel-value: value = lon for A, lon + 0.05 for B (varies by point, B offset).
    def pixel(request: httpx.Request) -> httpx.Response:
        q = parse_qs(urlparse(str(request.url)).query)
        lon = float(q["lon"][0])
        offset = 0.05 if "/b1/" in str(request.url) else 0.0
        return httpx.Response(
            200,
            json={
                "records": [
                    {
                        "coordinates": {"lon": lon, "lat": 0},
                        "bands": [
                            {
                                "property_name": "sample_karst_score",
                                "raw_value": round(lon + offset, 6),
                                "classification": None,
                            }
                        ],
                    }
                ]
            },
        )

    httpx_mock.add_callback(
        pixel, url=re.compile(r".*/pixel-value\?.*"), is_reusable=True
    )

    async with StudioClient(StudioConfig(api_key="k", base_url=BASE)) as studio:
        ctx = ToolContext(studio=studio, state=ThreadState())
        out = await _tool("olmoearth_compare_results").handler(
            {"result_id_a": "a1", "result_id_b": "b1", "grid": 3, "tolerance": 0.1}, ctx
        )

    assert out["comparable"] is True
    assert out["kind"] == "regression"
    s = out["stats"]
    assert s["n_samples"] == 9  # 3x3 grid, all valid
    assert s["mean_diff_b_minus_a"] == 0.05  # B is uniformly +0.05
    assert s["correlation"] == 1.0  # perfectly correlated (B = A + const)
    assert s["agreement_fraction"] == 1.0  # 0.05 <= tolerance 0.1


@pytest.mark.asyncio
async def test_compare_results_no_overlap(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url=f"{BASE}/prediction-results/a1",
        json={
            "records": [
                {"id": "a1", "result_metadata": {"geometry": {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]}}}
            ]
        },
    )
    httpx_mock.add_response(
        url=f"{BASE}/prediction-results/b1",
        json={
            "records": [
                {"id": "b1", "result_metadata": {"geometry": {"type": "Polygon", "coordinates": [[[5, 5], [6, 5], [6, 6], [5, 6], [5, 5]]]}}}
            ]
        },
    )
    async with StudioClient(StudioConfig(api_key="k", base_url=BASE)) as studio:
        ctx = ToolContext(studio=studio, state=ThreadState())
        out = await _tool("olmoearth_compare_results").handler(
            {"result_id_a": "a1", "result_id_b": "b1"}, ctx
        )
    assert out["comparable"] is False
