# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Unit tests for olmoearth_compare_results, pair and group modes (mocked Studio).

The group tests are the former olmoearth_compare_group tests, ported to the
merged tool (``mode="group"``).
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
from olmoearth_agent.tools.registry import RegisteredTool, ToolContext

BASE = "http://mock-studio/api/v1"


def _tool() -> RegisteredTool:
    (tool,) = build_compare_tools()
    return tool


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
    httpx_mock.add_response(
        url=f"{BASE}/prediction-results/a1", json=_result_with_geom("a1")
    )
    httpx_mock.add_response(
        url=f"{BASE}/prediction-results/b1", json=_result_with_geom("b1")
    )

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
        out = await _tool().handler(
            {"result_ids": ["a1", "b1"], "grid": 3, "tolerance": 0.1}, ctx
        )

    assert out["comparable"] is True
    assert out["mode"] == "pair"  # auto: two results
    assert (out["result_id_a"], out["result_id_b"]) == ("a1", "b1")  # UI card keys
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
    httpx_mock.add_response(
        url=f"{BASE}/prediction-results/a1", json=_result_with_geom("a1")
    )
    httpx_mock.add_response(
        url=f"{BASE}/prediction-results/b1", json=_result_with_geom("b1")
    )

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

    httpx_mock.add_callback(
        pixel, url=re.compile(r".*/pixel-value\?.*"), is_reusable=True
    )

    async with StudioClient(StudioConfig(api_key="k", base_url=BASE)) as studio:
        ctx = ToolContext(studio=studio, state=ThreadState())
        out = await _tool().handler(
            {"result_ids": ["a1", "b1"], "kind": "temporal", "grid": 3},
            ctx,
        )

    assert out["comparable"] is True
    assert out["kind"] == "temporal"
    assert out["ordering"] == "given-order"  # no prediction dates to order by
    assert out["value_type"] == "regression"
    nar = out["narration"]
    assert nar["labels"] == {
        "a": "earlier",
        "b": "later",
        "diff": "change (later - earlier)",
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
                {
                    "id": "a1",
                    "result_metadata": {
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
                        }
                    },
                }
            ]
        },
    )
    httpx_mock.add_response(
        url=f"{BASE}/prediction-results/b1",
        json={
            "records": [
                {
                    "id": "b1",
                    "result_metadata": {
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [[[5, 5], [6, 5], [6, 6], [5, 6], [5, 5]]],
                        }
                    },
                }
            ]
        },
    )
    async with StudioClient(StudioConfig(api_key="k", base_url=BASE)) as studio:
        ctx = ToolContext(studio=studio, state=ThreadState())
        out = await _tool().handler({"result_ids": ["a1", "b1"]}, ctx)
    assert out["comparable"] is False
    assert "common extent" in out["reason"]


# ---------------------------------------------------------------------------
# mode="group" (N results; formerly olmoearth_compare_group)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_compare_group_pairwise_and_ensemble(httpx_mock: HTTPXMock) -> None:
    # Three results over the same 0..10 extent. B = A + 0.05 (within the 0.1
    # tolerance), C = A + 0.5 (out) -> consensus is 0, the divergent pair is
    # with C, and every hotspot is a grid window (row, col), never a lon/lat.
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

    httpx_mock.add_callback(
        pixel, url=re.compile(r".*/pixel-value\?.*"), is_reusable=True
    )

    async with StudioClient(StudioConfig(api_key="k", base_url=BASE)) as studio:
        ctx = ToolContext(studio=studio, state=ThreadState())
        out = await _tool().handler(
            {"result_ids": ["a1", "b1", "c1"], "grid": 3, "tolerance": 0.1}, ctx
        )

    assert out["comparable"] is True
    assert out["mode"] == "group"  # auto: 3 results, models unknown
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
    assert spots
    for spot in spots:  # rule 3.1: a window address, never a coordinate
        assert "lon" not in spot and "lat" not in spot
        assert spot["window_index"] == spot["row"] * 3 + spot["col"]
    assert "shared_extent_bbox" not in out
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
        out = await _tool().handler(
            {"result_ids": ["a1", "b1", "c1"], "mode": "group"}, ctx
        )
    assert out["comparable"] is False
    assert "common extent" in out["reason"]


@pytest.mark.asyncio
async def test_compare_group_validates_id_count() -> None:
    ctx = ToolContext(studio=None, state=ThreadState())  # type: ignore[arg-type]
    too_few = await _tool().handler(
        {"result_ids": ["only", "only"], "mode": "group"},
        ctx,  # dedup -> 1 id
    )
    assert too_few["comparable"] is False
    too_many = await _tool().handler(
        {"result_ids": [f"r{i}" for i in range(7)], "mode": "group"}, ctx
    )
    assert too_many["comparable"] is False
    assert "mode='group'" in too_many["reason"]  # the group cap is 6


# ---------------------------------------------------------------------------
# Mode choice, refusals, and rule 3.1 across every mode
# ---------------------------------------------------------------------------


def _dated_result(rid: str, pid: str, prop: str | None = None) -> dict:
    body = _result_with_geom(rid)
    body["records"][0]["prediction_id"] = pid
    if prop:
        body["records"][0]["property_names"] = [prop]
    return body


def _offset_pixel(offsets: dict[str, float], prop: str = "score") -> Any:
    def pixel(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        lon = float(parse_qs(urlparse(url).query)["lon"][0])
        offset = next(v for k, v in offsets.items() if f"/{k}/" in url)
        band = {
            "property_name": prop,
            "raw_value": round(lon / 100 + offset, 6),
            "classification": None,
        }
        return httpx.Response(200, json={"records": [{"bands": [band]}]})

    return pixel


@pytest.mark.asyncio
async def test_auto_pair_of_one_model_on_two_dates_is_temporal_and_ordered(
    httpx_mock: HTTPXMock,
) -> None:
    # Given later-first; the tool reads both predictions' dates and makes the
    # earlier result A, so "later - earlier" cannot silently flip sign.
    httpx_mock.add_response(
        url=f"{BASE}/prediction-results/late", json=_dated_result("late", "p2")
    )
    httpx_mock.add_response(
        url=f"{BASE}/prediction-results/early", json=_dated_result("early", "p1")
    )
    for pid, start in (("p1", "2025-01-01T00:00:00Z"), ("p2", "2025-06-01")):
        httpx_mock.add_response(
            url=f"{BASE}/predictions/{pid}",
            json={"records": [{"id": pid, "start_time": start, "model_id": "m1"}]},
        )
    httpx_mock.add_response(
        url=f"{BASE}/models/m1", json={"records": [{"id": "m1", "wizard_answers": {}}]}
    )
    httpx_mock.add_callback(
        _offset_pixel({"early": 0.0, "late": 0.2}),
        url=re.compile(r".*/pixel-value\?.*"),
        is_reusable=True,
    )
    async with StudioClient(StudioConfig(api_key="k", base_url=BASE)) as studio:
        ctx = ToolContext(studio=studio, state=ThreadState())
        out = await _tool().handler({"result_ids": ["late", "early"], "grid": 2}, ctx)
    assert out["mode"] == "pair"
    assert out["kind"] == "temporal"
    assert out["ordering"] == "chronological"
    assert (out["result_id_a"], out["result_id_b"]) == ("early", "late")
    assert out["dates"] == ["2025-01-01T00:00:00Z", "2025-06-01"]
    assert out["stats"]["mean_diff_b_minus_a"] == 0.2
    assert "net increase" in out["narration"]["headline"]


@pytest.mark.asyncio
async def test_temporal_pair_of_two_models_is_refused(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url=f"{BASE}/prediction-results/a1", json=_dated_result("a1", "p1")
    )
    httpx_mock.add_response(
        url=f"{BASE}/prediction-results/b1", json=_dated_result("b1", "p2")
    )
    for pid, model in (("p1", "m1"), ("p2", "m2")):
        httpx_mock.add_response(
            url=f"{BASE}/predictions/{pid}",
            json={"records": [{"id": pid, "model_id": model}]},
        )
    async with StudioClient(StudioConfig(api_key="k", base_url=BASE)) as studio:
        ctx = ToolContext(studio=studio, state=ThreadState())
        out = await _tool().handler(
            {"result_ids": ["a1", "b1"], "kind": "temporal"}, ctx
        )
    assert out["comparable"] is False
    assert "different models" in out["reason"]
    assert "cross_model" in out["reason"]
    assert not any("pixel-value" in str(r.url) for r in httpx_mock.get_requests())


@pytest.mark.asyncio
async def test_mode_counts_are_checked_before_any_studio_call() -> None:
    ctx = ToolContext(studio=None, state=ThreadState())  # type: ignore[arg-type]
    pair = await _tool().handler({"result_ids": ["a", "b", "c"], "mode": "pair"}, ctx)
    assert pair["comparable"] is False
    assert "exactly two" in pair["reason"]
    too_many = await _tool().handler({"result_ids": [f"r{i}" for i in range(9)]}, ctx)
    assert too_many["comparable"] is False
    assert "too many results" in too_many["reason"]


@pytest.mark.asyncio
async def test_auto_group_over_its_cap_points_to_ensemble(
    httpx_mock: HTTPXMock,
) -> None:
    # Seven results with no known model: not a series, too many for a group.
    ids = [f"r{i}" for i in range(7)]
    for rid in ids:
        httpx_mock.add_response(
            url=f"{BASE}/prediction-results/{rid}", json=_result_with_geom(rid)
        )
    async with StudioClient(StudioConfig(api_key="k", base_url=BASE)) as studio:
        ctx = ToolContext(studio=studio, state=ThreadState())
        out = await _tool().handler({"result_ids": ids}, ctx)
    assert out["comparable"] is False
    assert "mode='ensemble'" in out["reason"]


@pytest.mark.asyncio
async def test_group_of_different_properties_is_refused_then_allowed(
    httpx_mock: HTTPXMock,
) -> None:
    props = {"a1": "score", "b1": "score", "c1": "count"}
    for rid, prop in props.items():
        body = _result_with_geom(rid)
        body["records"][0]["property_names"] = [prop]
        httpx_mock.add_response(
            url=f"{BASE}/prediction-results/{rid}", json=body, is_reusable=True
        )

    def pixel(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        rid = next(r for r in props if f"/{r}/" in url)
        lon = float(parse_qs(urlparse(url).query)["lon"][0])
        band = {"property_name": props[rid], "raw_value": lon, "classification": None}
        return httpx.Response(200, json={"records": [{"bands": [band]}]})

    httpx_mock.add_callback(
        pixel, url=re.compile(r".*/pixel-value\?.*"), is_reusable=True
    )
    async with StudioClient(StudioConfig(api_key="k", base_url=BASE)) as studio:
        ctx = ToolContext(studio=studio, state=ThreadState())
        refused = await _tool().handler({"result_ids": list(props), "grid": 2}, ctx)
        assert not any("pixel-value" in str(r.url) for r in httpx_mock.get_requests())
        allowed = await _tool().handler(
            {
                "result_ids": list(props),
                "grid": 2,
                "allow_different_properties": True,
            },
            ctx,
        )
    assert refused["comparable"] is False
    assert refused["mode"] == "group"
    assert "different properties" in refused["reason"]
    assert [p["property_name"] for p in refused["properties"]] == [
        "score",
        "score",
        "count",
    ]
    assert allowed["comparable"] is True
    assert "only the correlation" in allowed["warning"]
    # a pair of one property keeps its statistics; a pair of two properties
    # keeps each map's mean and the correlation; no ensemble across quantities
    by_pair = {
        (p["result_id_a"], p["result_id_b"]): set(p["stats"])
        for p in allowed["pairwise"]
    }
    assert "agreement_fraction" in by_pair[("a1", "b1")]
    for pair in (("a1", "c1"), ("b1", "c1")):
        assert by_pair[pair] <= {"n_samples", "mean_a", "mean_b", "correlation", "note"}
    assert set(allowed["ensemble"]) == {"note"}
    assert allowed["most_divergent_pair"] is None
    assert "rmse_between_models" in allowed["statistics_left_out"]
    assert "ensemble" in allowed["statistics_left_out"]


def _keys(value: object) -> set[str]:
    """Every dict key anywhere inside a JSON-like value."""
    if isinstance(value, dict):
        return set(value) | {k for v in value.values() for k in _keys(v)}
    if isinstance(value, list):
        return {k for v in value for k in _keys(v)}
    return set()


_FORBIDDEN = {"lon", "lat", "bbox", "coordinates", "geometry", "shared_extent_bbox"}


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["pair", "group", "series", "ensemble"])
async def test_no_mode_returns_a_coordinate(mode: str, httpx_mock: HTTPXMock) -> None:
    """Rule 3.1: windows by (row, col) and index only, in every mode."""
    offsets = {"ra": 0.0, "rb": 0.3, "rc": 0.6}
    ids = ["ra", "rb"] if mode == "pair" else list(offsets)
    for day, rid in enumerate(ids, start=1):
        httpx_mock.add_response(
            url=f"{BASE}/prediction-results/{rid}",
            json=_dated_result(rid, f"p-{rid}", "score"),
        )
        httpx_mock.add_response(
            url=f"{BASE}/predictions/p-{rid}",
            json={
                "records": [
                    {
                        "id": f"p-{rid}",
                        "start_time": f"2025-0{day}-01",
                        "model_id": "m1",
                    }
                ]
            },
        )
    httpx_mock.add_response(
        url=f"{BASE}/models/m1", json={"records": [{"id": "m1", "wizard_answers": {}}]}
    )
    httpx_mock.add_callback(
        _offset_pixel(offsets), url=re.compile(r".*/pixel-value\?.*"), is_reusable=True
    )
    async with StudioClient(StudioConfig(api_key="k", base_url=BASE)) as studio:
        ctx = ToolContext(studio=studio, state=ThreadState())
        out = await _tool().handler(
            {"result_ids": ids, "mode": mode, "grid": 3, "tolerance": 0.05}, ctx
        )
    assert out["comparable"] is True
    assert out["mode"] == mode
    assert not (_keys(out) & _FORBIDDEN)
    assert out["shared_extent_km2"] > 0
