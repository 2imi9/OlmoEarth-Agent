# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Unit tests for the olmoearth_trace_shifts tool (tools/trace_shifts.py)."""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import parse_qs, urlparse

import httpx
import pytest
from pytest_httpx import HTTPXMock

from olmoearth_agent.harness.state import ThreadState
from olmoearth_agent.studio.client import StudioClient, StudioConfig
from olmoearth_agent.tools.registry import RegisteredTool, ToolContext
from olmoearth_agent.tools.trace_shifts import build_trace_shift_tools

BASE = "http://mock-studio/api/v1"

#: Per-result value offset; chosen so later dates have larger values, making
#: every expected stat an exact constant (values depend only on lon + offset).
_OFFSETS = {"ra": 0.0, "rb": 0.1, "rc": 0.2}
#: result -> prediction -> start_time; input order is deliberately NOT
#: chronological so the test proves the tool re-orders by date.
_PREDICTIONS = {"ra": ("pa", "2026-01-01T00:00:00Z"), "rb": ("pb", "2026-02-01T00:00:00Z"), "rc": ("pc", "2026-03-01T00:00:00Z")}


def _tool() -> RegisteredTool:
    (tool,) = build_trace_shift_tools()
    return tool


def _result_body(rid: str) -> dict[str, Any]:
    return {
        "records": [
            {
                "id": rid,
                "prediction_id": _PREDICTIONS[rid][0],
                "result_metadata": {
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [
                            [[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]
                        ],
                    }
                },
            }
        ]
    }


def _prediction_body(rid: str) -> dict[str, Any]:
    pid, start = _PREDICTIONS[rid]
    return {"records": [{"id": pid, "start_time": start, "model_id": "m1"}]}


def _pixel(request: httpx.Request) -> httpx.Response:
    url = str(request.url)
    lon = float(parse_qs(urlparse(url).query)["lon"][0])
    offset = next(v for k, v in _OFFSETS.items() if f"/{k}/" in url)
    return httpx.Response(
        200,
        json={
            "records": [
                {
                    "coordinates": {"lon": lon, "lat": 0},
                    "bands": [
                        {
                            "property_name": "sample_score",
                            "raw_value": round(lon / 100 + offset, 6),
                            "classification": None,
                        }
                    ],
                }
            ]
        },
    )


def _mock_studio(httpx_mock: HTTPXMock, rids: list[str]) -> None:
    for rid in rids:
        httpx_mock.add_response(
            url=f"{BASE}/prediction-results/{rid}", json=_result_body(rid)
        )
        httpx_mock.add_response(
            url=f"{BASE}/predictions/{_PREDICTIONS[rid][0]}",
            json=_prediction_body(rid),
        )
    httpx_mock.add_callback(
        _pixel, url=re.compile(r".*/pixel-value\?.*"), is_reusable=True
    )


@pytest.mark.asyncio
async def test_trace_orders_by_date_and_traces_shifts(
    httpx_mock: HTTPXMock,
) -> None:
    # Input order rb, rc, ra — chronological is ra (Jan), rb (Feb), rc (Mar).
    _mock_studio(httpx_mock, ["rb", "rc", "ra"])
    async with StudioClient(StudioConfig(api_key="k", base_url=BASE)) as studio:
        ctx = ToolContext(studio=studio, state=ThreadState())
        out = await _tool().handler(
            {
                "result_ids": ["rb", "rc", "ra"],
                "grid": 3,
                "tolerance": 0.05,
                "value_range": [0.0, 1.0],
            },
            ctx,
        )
    assert out["comparable"] is True
    assert out["ordering"] == "chronological"
    assert out["result_ids"] == ["ra", "rb", "rc"]
    assert out["dates"] == [_PREDICTIONS[r][1] for r in ("ra", "rb", "rc")]
    assert out["value_type"] == "regression"
    assert out["samples_requested"] == 27
    # Each step is +0.1 everywhere: perfect correlation, exact mean delta.
    assert len(out["steps"]) == 2
    for step in out["steps"]:
        assert step["stats"]["mean_diff_b_minus_a"] == 0.1
        assert step["stats"]["correlation"] == 1.0
    assert out["steps"][0]["from"] == {"result_id": "ra", "date": _PREDICTIONS["ra"][1]}
    assert out["steps"][0]["to"] == {"result_id": "rb", "date": _PREDICTIONS["rb"][1]}
    trajectory = out["trajectory"]
    assert trajectory["n_points_used"] == 9
    assert trajectory["mean_total_change"] == 0.2
    assert trajectory["mean_slope_per_step"] == 0.1
    assert trajectory["shifted_fraction"] == 1.0
    # Legend calibration: supplied 0-1 range -> net shift is 20% of range.
    assert out["calibration"]["source"] == "supplied"
    assert trajectory["mean_total_change_fraction_of_range"] == 0.2
    top = trajectory["top_shift_points"][0]
    assert "lon" in top and "lat" in top
    assert top["largest_step_to_date"] == _PREDICTIONS["rc"][1]
    assert "net increase" in out["narration"]["headline"]
    assert "not verified ground change" in out["method"]


@pytest.mark.asyncio
async def test_trace_falls_back_to_given_order_without_dates(
    httpx_mock: HTTPXMock,
) -> None:
    # rb's prediction lookup 404s -> its date is unknown -> given-order.
    httpx_mock.add_response(
        url=f"{BASE}/prediction-results/rb", json=_result_body("rb")
    )
    httpx_mock.add_response(
        url=f"{BASE}/predictions/pb", status_code=404, is_reusable=True
    )
    for rid in ("rc", "ra"):
        httpx_mock.add_response(
            url=f"{BASE}/prediction-results/{rid}", json=_result_body(rid)
        )
        httpx_mock.add_response(
            url=f"{BASE}/predictions/{_PREDICTIONS[rid][0]}",
            json=_prediction_body(rid),
        )
    httpx_mock.add_callback(
        _pixel, url=re.compile(r".*/pixel-value\?.*"), is_reusable=True
    )
    async with StudioClient(StudioConfig(api_key="k", base_url=BASE)) as studio:
        ctx = ToolContext(studio=studio, state=ThreadState())
        out = await _tool().handler(
            {"result_ids": ["rb", "rc", "ra"], "grid": 2}, ctx
        )
    assert out["comparable"] is True
    assert out["ordering"] == "given-order"
    assert out["result_ids"] == ["rb", "rc", "ra"]  # untouched input order
    assert out["dates"][0] is None
    assert "not verified chronological" in out["narration"]["framing"]


@pytest.mark.asyncio
async def test_trace_validates_result_count() -> None:
    ctx = ToolContext(studio=None, state=ThreadState())  # type: ignore[arg-type]
    too_few = await _tool().handler({"result_ids": ["ra", "ra", "rb"]}, ctx)
    assert too_few["comparable"] is False
    assert "olmoearth_compare_results" in too_few["reason"]
    too_many = await _tool().handler(
        {"result_ids": [f"r{i}" for i in range(9)]}, ctx
    )
    assert too_many["comparable"] is False
    assert "too many results" in too_many["reason"]


@pytest.mark.asyncio
async def test_trace_sorts_mixed_aware_and_naive_dates(
    httpx_mock: HTTPXMock,
) -> None:
    # rb's prediction reports a NAIVE date-only start_time; ra/rc are
    # tz-aware. All parse, so the chronological branch runs — naive is read
    # as UTC instead of crashing sorted() with aware-vs-naive TypeError.
    for rid in ("rb", "rc", "ra"):
        httpx_mock.add_response(
            url=f"{BASE}/prediction-results/{rid}", json=_result_body(rid)
        )
        body = _prediction_body(rid)
        if rid == "rb":
            body["records"][0]["start_time"] = "2026-02-01"
        httpx_mock.add_response(
            url=f"{BASE}/predictions/{_PREDICTIONS[rid][0]}", json=body
        )
    httpx_mock.add_callback(
        _pixel, url=re.compile(r".*/pixel-value\?.*"), is_reusable=True
    )
    async with StudioClient(StudioConfig(api_key="k", base_url=BASE)) as studio:
        ctx = ToolContext(studio=studio, state=ThreadState())
        out = await _tool().handler({"result_ids": ["rb", "rc", "ra"], "grid": 2}, ctx)
    assert out["comparable"] is True
    assert out["ordering"] == "chronological"
    assert out["result_ids"] == ["ra", "rb", "rc"]


@pytest.mark.asyncio
async def test_trace_refuses_mixed_models(httpx_mock: HTTPXMock) -> None:
    for rid in ("rb", "rc", "ra"):
        httpx_mock.add_response(
            url=f"{BASE}/prediction-results/{rid}", json=_result_body(rid)
        )
        body = _prediction_body(rid)
        if rid == "rc":
            body["records"][0]["model_id"] = "m2"  # different model
        httpx_mock.add_response(
            url=f"{BASE}/predictions/{_PREDICTIONS[rid][0]}", json=body
        )
    async with StudioClient(StudioConfig(api_key="k", base_url=BASE)) as studio:
        ctx = ToolContext(studio=studio, state=ThreadState())
        out = await _tool().handler({"result_ids": ["rb", "rc", "ra"]}, ctx)
    assert out["comparable"] is False
    assert "multiple models" in out["reason"]
    assert "olmoearth_compare_group" in out["reason"]


@pytest.mark.asyncio
async def test_trace_refuses_duplicate_dates(httpx_mock: HTTPXMock) -> None:
    for rid in ("rb", "rc", "ra"):
        httpx_mock.add_response(
            url=f"{BASE}/prediction-results/{rid}", json=_result_body(rid)
        )
        body = _prediction_body(rid)
        body["records"][0]["start_time"] = "2026-01-01T00:00:00Z"  # all same
        httpx_mock.add_response(
            url=f"{BASE}/predictions/{_PREDICTIONS[rid][0]}", json=body
        )
    async with StudioClient(StudioConfig(api_key="k", base_url=BASE)) as studio:
        ctx = ToolContext(studio=studio, state=ThreadState())
        out = await _tool().handler({"result_ids": ["rb", "rc", "ra"]}, ctx)
    assert out["comparable"] is False
    assert "distinct dates" in out["reason"]
    assert "olmoearth_ensemble_uncertainty" in out["reason"]


def _mock_metadata_only(httpx_mock: HTTPXMock, rids: list[str]) -> None:
    """Result + prediction mocks without the shared pixel callback, so a test
    can register its own (a second matching reusable callback would never
    fire — pytest-httpx serves the first registered match)."""
    for rid in rids:
        httpx_mock.add_response(
            url=f"{BASE}/prediction-results/{rid}", json=_result_body(rid)
        )
        httpx_mock.add_response(
            url=f"{BASE}/predictions/{_PREDICTIONS[rid][0]}",
            json=_prediction_body(rid),
        )


@pytest.mark.asyncio
async def test_trace_refuses_mixed_band_types(httpx_mock: HTTPXMock) -> None:
    # ra (chronologically first) returns numeric raw_values; rc returns class
    # labels -> a clean refusal instead of float('urban') mid-envelope.
    _mock_metadata_only(httpx_mock, ["rb", "rc", "ra"])

    def pixel_mixed(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "/rc/" in url:
            return httpx.Response(
                200,
                json={
                    "records": [
                        {
                            "bands": [
                                {
                                    "property_name": "sample_score",
                                    "raw_value": "urban",
                                    "classification": None,
                                }
                            ]
                        }
                    ]
                },
            )
        return _pixel(request)

    httpx_mock.add_callback(
        pixel_mixed, url=re.compile(r".*/pixel-value\?.*"), is_reusable=True
    )
    async with StudioClient(StudioConfig(api_key="k", base_url=BASE)) as studio:
        ctx = ToolContext(studio=studio, state=ThreadState())
        out = await _tool().handler({"result_ids": ["rb", "rc", "ra"], "grid": 2}, ctx)
    assert out["comparable"] is False
    assert "mix numeric and categorical" in out["reason"]


@pytest.mark.asyncio
async def test_trace_invalid_value_range_is_flagged_not_silent(
    httpx_mock: HTTPXMock,
) -> None:
    _mock_studio(httpx_mock, ["rb", "rc", "ra"])
    async with StudioClient(StudioConfig(api_key="k", base_url=BASE)) as studio:
        ctx = ToolContext(studio=studio, state=ThreadState())
        out = await _tool().handler(
            {"result_ids": ["rb", "rc", "ra"], "grid": 2, "value_range": [1.0, 0.0]},
            ctx,
        )
    assert out["comparable"] is True
    assert out["calibration"]["source"] == "observed"
    assert "value_range was invalid" in out["calibration"]["note"]


@pytest.mark.asyncio
async def test_trace_skips_empty_bands_when_picking_first(
    httpx_mock: HTTPXMock,
) -> None:
    # The chronologically-first sample is a nodata record with bands: [] —
    # it must not crash envelope building nor decide the value type.
    _mock_metadata_only(httpx_mock, ["rb", "rc", "ra"])
    seen: dict[str, bool] = {}

    def pixel_first_empty(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "/ra/" in url and not seen.get("ra"):
            seen["ra"] = True
            return httpx.Response(200, json={"records": [{"bands": []}]})
        return _pixel(request)

    httpx_mock.add_callback(
        pixel_first_empty, url=re.compile(r".*/pixel-value\?.*"), is_reusable=True
    )
    async with StudioClient(StudioConfig(api_key="k", base_url=BASE)) as studio:
        ctx = ToolContext(studio=studio, state=ThreadState())
        out = await _tool().handler({"result_ids": ["rb", "rc", "ra"], "grid": 2}, ctx)
    assert out["comparable"] is True
    assert out["value_type"] == "regression"
    assert out["property_name"] == "sample_score"
    assert out["model_check"] == "single-model"


@pytest.mark.asyncio
async def test_trace_reports_missing_overlap(httpx_mock: HTTPXMock) -> None:
    for rid in ("rb", "rc", "ra"):
        body = _result_body(rid)
        if rid == "rc":  # disjoint extent
            body["records"][0]["result_metadata"]["geometry"]["coordinates"] = [
                [[100, 100], [110, 100], [110, 110], [100, 110], [100, 100]]
            ]
        httpx_mock.add_response(url=f"{BASE}/prediction-results/{rid}", json=body)
    async with StudioClient(StudioConfig(api_key="k", base_url=BASE)) as studio:
        ctx = ToolContext(studio=studio, state=ThreadState())
        out = await _tool().handler({"result_ids": ["rb", "rc", "ra"]}, ctx)
    assert out["comparable"] is False
    assert "common overlapping extent" in out["reason"]
