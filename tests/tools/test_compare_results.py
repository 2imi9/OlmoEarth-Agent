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
    assert out["kind"] == "cross_model"  # default comparison mode
    assert out["value_type"] == "regression"  # data type (renamed from "kind")
    assert out["narration"]["labels"]["a"] == "model A"
    assert "model-vs-model agreement" in out["narration"]["framing"]
    s = out["stats"]
    assert s["n_samples"] == 9  # 3x3 grid, all valid
    assert s["mean_diff_b_minus_a"] == 0.05  # B is uniformly +0.05
    assert s["correlation"] == 1.0  # perfectly correlated (B = A + const)
    assert s["agreement_fraction"] == 1.0  # 0.05 <= tolerance 0.1


@pytest.mark.asyncio
async def test_compare_results_temporal_frames_change_over_time(
    httpx_mock: HTTPXMock,
) -> None:
    # Same model, two dates: A = earlier, B = later. B is uniformly +0.05, so
    # the narration should report a net *increase* over time, not "model agree".
    httpx_mock.add_response(url=f"{BASE}/prediction-results/a1", json=_result_with_geom("a1"))
    httpx_mock.add_response(url=f"{BASE}/prediction-results/b1", json=_result_with_geom("b1"))

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
                                "property_name": "sample_score",
                                "raw_value": round(lon + offset, 6),
                                "classification": None,
                            }
                        ],
                    }
                ]
            },
        )

    httpx_mock.add_callback(pixel, url=re.compile(r".*/pixel-value\?.*"), is_reusable=True)

    async with StudioClient(StudioConfig(api_key="k", base_url=BASE)) as studio:
        ctx = ToolContext(studio=studio, state=ThreadState())
        out = await _tool("olmoearth_compare_results").handler(
            {"result_id_a": "a1", "result_id_b": "b1", "kind": "temporal", "grid": 3},
            ctx,
        )

    assert out["comparable"] is True
    assert out["kind"] == "temporal"
    assert out["value_type"] == "regression"
    nar = out["narration"]
    assert nar["labels"] == {
        "a": "earlier", "b": "later", "diff": "change (later - earlier)"
    }
    assert "net increase of 0.05" in nar["headline"]
    assert "change over time" in nar["framing"]
    # the framing string (used in `method`) must not claim model agreement
    assert "not model-vs-model agreement" in nar["framing"]
    assert "not model-vs-model agreement" in out["method"]


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


# ---------------------------------------------------------------------------
# olmoearth_compare_group (N results)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_compare_group_pairwise_and_ensemble(httpx_mock: HTTPXMock) -> None:
    # Three results over the same 0..10 extent. B = A + 0.05 (within the 0.1
    # tolerance), C = A + 0.5 (out) -> consensus is 0, the divergent pair is
    # with C, and every hotspot carries a real lon/lat from the grid.
    for rid in ("a1", "b1", "c1"):
        httpx_mock.add_response(
            url=f"{BASE}/prediction-results/{rid}", json=_result_with_geom(rid)
        )

    offsets = {"a1": 0.0, "b1": 0.05, "c1": 0.5}

    def pixel(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        q = parse_qs(urlparse(url).query)
        lon = float(q["lon"][0])
        offset = next(v for k, v in offsets.items() if f"/{k}/" in url)
        return httpx.Response(
            200,
            json={
                "records": [
                    {
                        "coordinates": {"lon": lon, "lat": 0},
                        "bands": [
                            {
                                "property_name": "sample_score",
                                "raw_value": round(lon + offset, 6),
                                "classification": None,
                            }
                        ],
                    }
                ]
            },
        )

    httpx_mock.add_callback(pixel, url=re.compile(r".*/pixel-value\?.*"), is_reusable=True)

    async with StudioClient(StudioConfig(api_key="k", base_url=BASE)) as studio:
        ctx = ToolContext(studio=studio, state=ThreadState())
        out = await _tool("olmoearth_compare_group").handler(
            {"result_ids": ["a1", "b1", "c1"], "grid": 3, "tolerance": 0.1}, ctx
        )

    assert out["comparable"] is True
    assert out["result_ids"] == ["a1", "b1", "c1"]
    assert out["value_type"] == "regression"
    assert out["samples_requested"] == 27  # 3x3 grid x 3 results
    assert len(out["pairwise"]) == 3
    ab = next(
        p
        for p in out["pairwise"]
        if {p["result_id_a"], p["result_id_b"]} == {"a1", "b1"}
    )
    assert ab["stats"]["agreement_fraction"] == 1.0
    ens = out["ensemble"]
    assert ens["n_points_used"] == 9
    assert ens["consensus_fraction"] == 0.0  # C is 0.5 away everywhere
    assert "c1" in (
        out["most_divergent_pair"]["result_id_a"],
        out["most_divergent_pair"]["result_id_b"],
    )
    spots = ens["top_disagreement_points"]
    assert spots and all("lon" in s and "lat" in s for s in spots)
    assert "consensus across the group" in out["method"]


@pytest.mark.asyncio
async def test_compare_group_requires_shared_extent(httpx_mock: HTTPXMock) -> None:
    # a1/b1 overlap, but c1 is disjoint -> no extent shared by ALL results.
    for rid in ("a1", "b1"):
        httpx_mock.add_response(
            url=f"{BASE}/prediction-results/{rid}", json=_result_with_geom(rid)
        )
    httpx_mock.add_response(
        url=f"{BASE}/prediction-results/c1",
        json={
            "records": [
                {
                    "id": "c1",
                    "result_metadata": {
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [
                                [[50, 50], [60, 50], [60, 60], [50, 60], [50, 50]]
                            ],
                        }
                    },
                }
            ]
        },
    )
    async with StudioClient(StudioConfig(api_key="k", base_url=BASE)) as studio:
        ctx = ToolContext(studio=studio, state=ThreadState())
        out = await _tool("olmoearth_compare_group").handler(
            {"result_ids": ["a1", "b1", "c1"]}, ctx
        )
    assert out["comparable"] is False
    assert "shared" in out["reason"]


@pytest.mark.asyncio
async def test_compare_group_validates_id_count() -> None:
    ctx = ToolContext(studio=None, state=ThreadState())  # type: ignore[arg-type]
    too_few = await _tool("olmoearth_compare_group").handler(
        {"result_ids": ["only", "only"]}, ctx  # dedup -> 1 distinct id
    )
    assert too_few["comparable"] is False
    too_many = await _tool("olmoearth_compare_group").handler(
        {"result_ids": [f"r{i}" for i in range(7)]}, ctx
    )
    assert too_many["comparable"] is False
