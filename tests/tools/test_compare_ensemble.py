# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""olmoearth_compare_results in ensemble mode (formerly olmoearth_ensemble_uncertainty).

Every test of the former tool, ported to the merged tool with
``mode="ensemble"``. One change of output: the value type reads
``classification`` (not ``categorical``), as in every other mode.
"""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import parse_qs, urlparse

import httpx
import pytest
from pytest_httpx import HTTPXMock

from olmoearth_agent.harness.state import ThreadState
from olmoearth_agent.studio.client import StudioClient, StudioConfig
from olmoearth_agent.tools.compare import build_compare_tools
from olmoearth_agent.tools.registry import ToolContext

BASE = "http://mock-studio/api/v1"

_BOX_0_10 = [[[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]]


def _ctx() -> ToolContext:
    return ToolContext(studio=None, state=ThreadState())  # type: ignore[arg-type]


async def _ensemble(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    """Call the merged tool in ensemble mode."""
    (tool,) = build_compare_tools()
    result: dict[str, Any] = await tool.handler({**args, "mode": "ensemble"}, ctx)
    return result


def _result_with_geom(rid: str, coords: list) -> dict:
    return {
        "records": [
            {
                "id": rid,
                "result_metadata": {
                    "geometry": {"type": "Polygon", "coordinates": coords}
                },
            }
        ]
    }


@pytest.mark.asyncio
async def test_ensemble_regression_disagreement(httpx_mock: HTTPXMock) -> None:
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
            json={
                "records": [
                    {
                        "bands": [
                            {
                                "property_name": "score",
                                "raw_value": round(lon + offset, 6),
                                "classification": None,
                            }
                        ]
                    }
                ]
            },
        )

    httpx_mock.add_callback(
        pixel, url=re.compile(r".*/pixel-value\?.*"), is_reusable=True
    )

    async with StudioClient(StudioConfig(api_key="k", base_url=BASE)) as studio:
        ctx = ToolContext(studio=studio, state=ThreadState())
        out = await _ensemble({"result_ids": ["a1", "b1"], "grid": 2}, ctx)
    assert out["comparable"] is True
    assert out["mode"] == "ensemble"
    assert out["value_type"] == "regression"
    assert out["n_results"] == 2
    assert out["n_points"] == 4  # 2x2 grid, all valid
    # every point had exactly the two ensemble members
    assert all(p["n"] == 2 for p in out["per_point"])
    # each point is addressed as a grid window (rule 3.1), never a coordinate
    assert [p["window_index"] for p in out["per_point"]] == [0, 1, 2, 3]
    assert {(p["row"], p["col"]) for p in out["per_point"]} == {
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    }
    # real spread -> confidence strictly below 1
    assert out["summary"]["mean_confidence"] is not None
    assert out["summary"]["mean_confidence"] < 1.0
    assert "not accuracy" in out["narration"]["framing"]


@pytest.mark.asyncio
async def test_ensemble_categorical_auto_detect(httpx_mock: HTTPXMock) -> None:
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
            json={
                "records": [
                    {
                        "bands": [
                            {
                                "property_name": "landcover",
                                "raw_value": None,
                                "classification": cls,
                            }
                        ]
                    }
                ]
            },
        )

    httpx_mock.add_callback(
        pixel, url=re.compile(r".*/pixel-value\?.*"), is_reusable=True
    )

    async with StudioClient(StudioConfig(api_key="k", base_url=BASE)) as studio:
        ctx = ToolContext(studio=studio, state=ThreadState())
        out = await _ensemble({"result_ids": ["a1", "b1"], "grid": 2}, ctx)
    assert out["comparable"] is True
    # auto-detected from classification (reads "classification" in every mode)
    assert out["value_type"] == "classification"
    # two members always disagree -> max entropy, zero confidence
    assert out["summary"]["mean_entropy"] == pytest.approx(1.0)
    assert out["summary"]["mean_confidence"] == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_ensemble_needs_two_results() -> None:
    out = await _ensemble({"result_ids": ["only-one"]}, _ctx())
    assert out["comparable"] is False
    assert ">= 2" in out["reason"]


@pytest.mark.asyncio
async def test_ensemble_rejects_duplicate_ids() -> None:
    # A duplicate id would re-read ONE deterministic result and fake zero
    # disagreement; it must be rejected before any sampling (no studio needed).
    out = await _ensemble({"result_ids": ["same", "same"]}, _ctx())
    assert out["comparable"] is False
    assert "DISTINCT" in out["reason"]


@pytest.mark.asyncio
async def test_ensemble_mixed_band_types_clean_answer(httpx_mock: HTTPXMock) -> None:
    # a1 is regression, b1 is categorical at the same property: the tool must
    # answer cleanly (comparable=False), not raise.
    httpx_mock.add_response(
        url=f"{BASE}/prediction-results/a1", json=_result_with_geom("a1", _BOX_0_10)
    )
    httpx_mock.add_response(
        url=f"{BASE}/prediction-results/b1", json=_result_with_geom("b1", _BOX_0_10)
    )

    def pixel(request: httpx.Request) -> httpx.Response:
        if "/b1/" in str(request.url):
            band = {"property_name": "x", "raw_value": None, "classification": "forest"}
        else:
            band = {"property_name": "x", "raw_value": 1.0, "classification": None}
        return httpx.Response(200, json={"records": [{"bands": [band]}]})

    httpx_mock.add_callback(
        pixel, url=re.compile(r".*/pixel-value\?.*"), is_reusable=True
    )

    async with StudioClient(StudioConfig(api_key="k", base_url=BASE)) as studio:
        ctx = ToolContext(studio=studio, state=ThreadState())
        out = await _ensemble({"result_ids": ["a1", "b1"], "grid": 2}, ctx)
    assert out["comparable"] is False
    assert "value_type" in out["reason"]


@pytest.mark.asyncio
async def test_ensemble_forced_regression_over_labels_is_a_clean_answer(
    httpx_mock: HTTPXMock,
) -> None:
    # value_type='regression' pinned over class labels: prediction_confidence
    # cannot float() them, and the tool says so instead of raising.
    for rid in ("a1", "b1"):
        httpx_mock.add_response(
            url=f"{BASE}/prediction-results/{rid}",
            json=_result_with_geom(rid, _BOX_0_10),
        )

    def pixel(_request: httpx.Request) -> httpx.Response:
        band = {"property_name": "x", "raw_value": None, "classification": "forest"}
        return httpx.Response(200, json={"records": [{"bands": [band]}]})

    httpx_mock.add_callback(
        pixel, url=re.compile(r".*/pixel-value\?.*"), is_reusable=True
    )
    async with StudioClient(StudioConfig(api_key="k", base_url=BASE)) as studio:
        ctx = ToolContext(studio=studio, state=ThreadState())
        out = await _ensemble(
            {"result_ids": ["a1", "b1"], "grid": 2, "value_type": "regression"}, ctx
        )
    assert out["comparable"] is False
    assert "value_type" in out["reason"]


@pytest.mark.asyncio
async def test_ensemble_no_shared_extent(httpx_mock: HTTPXMock) -> None:
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
        out = await _ensemble({"result_ids": ["a1", "b1"]}, ctx)
    assert out["comparable"] is False
    assert "common extent" in out["reason"]
