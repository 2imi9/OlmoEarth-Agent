# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""No-data never enters a statistic: every grid-sampling tool (mocked Studio).

The regression fixture is the live trial of 24 September 2026: KarstBinary
against KarstNumber over Pennsylvania, on the 6x6 grid
``olmoearth_compare_results`` samples for a pair. Studio returned the no-data sentinel
``-1.0`` as a value at 11 of the 36 points, in both maps, and the tool
reported a correlation of 0.946 ("karst in largely the same places") where
the 25 valid points give -0.017. Values only; no coordinates were kept.
"""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import parse_qs, urlparse

import httpx
import pytest
from pytest_httpx import HTTPXMock

from olmoearth_agent.analysis.raster_compare import (
    band_is_nodata,
    compare_numeric,
    grid_points,
)
from olmoearth_agent.harness.state import ThreadState
from olmoearth_agent.studio.client import StudioClient, StudioConfig
from olmoearth_agent.tools.compare import build_compare_tools
from olmoearth_agent.tools.predict import build_predict_tools
from olmoearth_agent.tools.registry import RegisteredTool, ToolContext

BASE = "http://mock-studio/api/v1"

#: (KarstBinary sample_karst_score, KarstNumber sample_number) at each point of
#: the 6x6 grid, in grid_points order.
KARST_PAIRS: list[tuple[float, float]] = [
    (-1.0, -1.0),
    (-1.0, -1.0),
    (-1.0, -1.0),
    (-1.0, -1.0),
    (-1.0, -1.0),
    (-1.0, -1.0),
    (-1.0, -1.0),
    (0.014252796769142151, 0.6907918453216553),
    (0.012461096048355103, 0.699575662612915),
    (0.01856943964958191, 0.7247110605239868),
    (0.010601378977298737, 0.7226852178573608),
    (0.009470075368881226, 0.6882981061935425),
    (-1.0, -1.0),
    (0.006078377366065979, 0.6535265445709229),
    (0.010559536516666412, 0.7297954559326172),
    (0.010187044739723206, 0.6476298570632935),
    (0.013300128281116486, 0.6524789333343506),
    (0.04418563097715378, 0.6897921562194824),
    (0.031240344047546387, 0.6923239231109619),
    (0.013816550374031067, 0.6827501058578491),
    (0.0050364211201667786, 0.645200252532959),
    (0.40842896699905396, 0.6834670305252075),
    (0.016914527863264084, 0.6380941867828369),
    (0.8781579732894897, 0.6616921424865723),
    (0.021388303488492966, 0.716030478477478),
    (0.014908656477928162, 0.6848962306976318),
    (0.030392754822969437, 0.6966522932052612),
    (0.01479293406009674, 0.6436717510223389),
    (0.010435245931148529, 0.6241855621337891),
    (0.014537528157234192, 0.6763505935668945),
    (-1.0, -1.0),
    (-1.0, -1.0),
    (-1.0, -1.0),
    (0.04444808512926102, 0.711957573890686),
    (0.3993160128593445, 0.6659121513366699),
    (0.45876431465148926, 0.7216143608093262),
]

_BOX = [[[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]]
_UNIT = {"min_value": 0.0, "max_value": 1.0, "colormap_name": "viridis"}


def _band(prop: str, value: float) -> dict[str, Any]:
    """A pixel-value band as Studio returns it for a [0, 1] regression output."""
    return {
        "band_index": 1,
        "property_name": prop,
        "raw_value": value,
        "classification": None,
        "regression": dict(_UNIT),
    }


def _result(rid: str, prop: str, **extra: Any) -> dict[str, Any]:
    """A prediction-result record declaring one [0, 1] regression property."""
    record = {
        "id": rid,
        "property_names": [prop],
        "result_metadata": {
            "geometry": {"type": "Polygon", "coordinates": _BOX},
            "regression_fields": [{"property_name": prop, **_UNIT}],
        },
    }
    record.update(extra)
    return {"records": [record]}


def _pixel_callback(by_result: dict[str, tuple[str, list[float]]], grid: int) -> Any:
    """Serve ``by_result[rid] = (property, values in grid_points order)``."""
    points = grid_points([0.0, 0.0, 10.0, 10.0], grid)
    index = {pt: i for i, pt in enumerate(points)}

    def pixel(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        q = parse_qs(urlparse(url).query)
        key = (round(float(q["lon"][0]), 6), round(float(q["lat"][0]), 6))
        rid = next(r for r in by_result if f"/{r}/" in url)
        prop, values = by_result[rid]
        return httpx.Response(
            200, json={"records": [{"bands": [_band(prop, values[index[key]])]}]}
        )

    return pixel


def _predict_tool(name: str) -> RegisteredTool:
    return next(t for t in build_predict_tools() if t.spec.name == name)


async def _compare(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    """Call olmoearth_compare_results (every grid-sampling mode lives there)."""
    (tool,) = build_compare_tools()
    result: dict[str, Any] = await tool.handler(args, ctx)
    return result


def _nodata_filtered(pairs: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """The pairs the shared rule keeps, with the trial's band metadata."""
    return [
        (a, b)
        for a, b in pairs
        if not band_is_nodata(_band("a", a)) and not band_is_nodata(_band("b", b))
    ]


def test_karst_fixture_before_and_after_the_fix() -> None:
    """Keep both numbers reproducible: 0.946 as shipped, -0.017 without no-data."""
    before = compare_numeric(KARST_PAIRS, tolerance=0.1)
    assert before["n_samples"] == 36
    assert before["correlation"] == pytest.approx(0.9464)
    assert before["agreement_fraction"] == pytest.approx(0.3056)
    assert before["mean_a"] == pytest.approx(-0.235771)

    kept = _nodata_filtered(KARST_PAIRS)
    assert len(KARST_PAIRS) - len(kept) == 11
    after = compare_numeric(kept, tolerance=0.1)
    assert after["n_samples"] == 25
    assert after["correlation"] == pytest.approx(-0.0172)
    assert after["agreement_fraction"] == 0.0
    assert after["mean_a"] == pytest.approx(0.10049)


@pytest.mark.asyncio
async def test_compare_results_drops_the_sentinel_on_the_trial_grid(
    httpx_mock: HTTPXMock,
) -> None:
    prop = "sample_karst_score"
    httpx_mock.add_response(
        url=f"{BASE}/prediction-results/a1", json=_result("a1", prop)
    )
    httpx_mock.add_response(
        url=f"{BASE}/prediction-results/b1", json=_result("b1", prop)
    )
    callback = _pixel_callback(
        {
            "a1": (prop, [a for a, _ in KARST_PAIRS]),
            "b1": (prop, [b for _, b in KARST_PAIRS]),
        },
        6,
    )
    httpx_mock.add_callback(
        callback, url=re.compile(r".*/pixel-value\?.*"), is_reusable=True
    )
    async with StudioClient(StudioConfig(api_key="k", base_url=BASE)) as studio:
        ctx = ToolContext(studio=studio, state=ThreadState())
        out = await _compare({"result_ids": ["a1", "b1"], "grid": 6}, ctx)
    assert out["comparable"] is True
    assert out["n_nodata_dropped"] == 11
    assert out["n_failed_dropped"] == 0
    assert out["stats"]["n_samples"] == 25
    assert out["stats"]["correlation"] == pytest.approx(-0.0172)
    assert out["stats"]["agreement_fraction"] == 0.0
    assert out["property_a"]["max_value"] == 1.0
    assert "warning" not in out


@pytest.mark.asyncio
async def test_compare_results_counts_cells_not_only_samples(
    httpx_mock: HTTPXMock,
) -> None:
    """exp86 round 1 (brief 3, Studio run 2) on this very grid: the answer read
    samples_requested (72, one sample per map per cell) as the cell count and
    wrote "25 of 72 grid cells, 47 dropped"; it is 25 of 36, 11 dropped."""
    prop = "sample_karst_score"
    httpx_mock.add_response(
        url=f"{BASE}/prediction-results/a1", json=_result("a1", prop)
    )
    httpx_mock.add_response(
        url=f"{BASE}/prediction-results/b1", json=_result("b1", prop)
    )
    callback = _pixel_callback(
        {
            "a1": (prop, [a for a, _ in KARST_PAIRS]),
            "b1": (prop, [b for _, b in KARST_PAIRS]),
        },
        6,
    )
    httpx_mock.add_callback(
        callback, url=re.compile(r".*/pixel-value\?.*"), is_reusable=True
    )
    async with StudioClient(StudioConfig(api_key="k", base_url=BASE)) as studio:
        ctx = ToolContext(studio=studio, state=ThreadState())
        out = await _compare({"result_ids": ["a1", "b1"], "grid": 6}, ctx)
    assert out["grid"] == "6x6"
    assert out["n_cells"] == 36
    assert out["n_cells_compared"] == 25 and out["n_cells_dropped"] == 11
    assert out["samples_requested"] == 72
    note = out["sampling_note"]
    assert "36 cells" in note and "72" in note and "25 compared" in note


@pytest.mark.asyncio
async def test_compare_results_refuses_different_properties_before_sampling(
    httpx_mock: HTTPXMock,
) -> None:
    """The trial compared a binary score with a count and narrated a 0.4 gap."""
    httpx_mock.add_response(
        url=f"{BASE}/prediction-results/a1", json=_result("a1", "sample_karst_score")
    )
    httpx_mock.add_response(
        url=f"{BASE}/prediction-results/b1", json=_result("b1", "sample_number")
    )
    async with StudioClient(StudioConfig(api_key="k", base_url=BASE)) as studio:
        ctx = ToolContext(studio=studio, state=ThreadState())
        out = await _compare({"result_ids": ["a1", "b1"], "grid": 6}, ctx)
    assert out["comparable"] is False
    assert out["property_a"]["property_name"] == "sample_karst_score"
    assert out["property_b"]["property_name"] == "sample_number"
    assert out["property_a"]["min_value"] == 0.0
    assert "allow_different_properties" in out["reason"]
    assert not any("pixel-value" in str(r.url) for r in httpx_mock.get_requests())


@pytest.mark.asyncio
async def test_compare_results_allows_different_properties_with_a_warning(
    httpx_mock: HTTPXMock,
) -> None:
    """The trial's exact call, allowed: correlation only, and the fixed number."""
    httpx_mock.add_response(
        url=f"{BASE}/prediction-results/a1", json=_result("a1", "sample_karst_score")
    )
    httpx_mock.add_response(
        url=f"{BASE}/prediction-results/b1", json=_result("b1", "sample_number")
    )
    callback = _pixel_callback(
        {
            "a1": ("sample_karst_score", [a for a, _ in KARST_PAIRS]),
            "b1": ("sample_number", [b for _, b in KARST_PAIRS]),
        },
        6,
    )
    httpx_mock.add_callback(
        callback, url=re.compile(r".*/pixel-value\?.*"), is_reusable=True
    )
    async with StudioClient(StudioConfig(api_key="k", base_url=BASE)) as studio:
        ctx = ToolContext(studio=studio, state=ThreadState())
        out = await _compare(
            {
                "result_ids": ["a1", "b1"],
                "grid": 6,
                "allow_different_properties": True,
            },
            ctx,
        )
    assert out["comparable"] is True
    assert out["n_nodata_dropped"] == 11
    assert out["stats"]["correlation"] == pytest.approx(-0.0172)
    assert "only the correlation" in out["warning"]
    assert out["narration"]["headline"].startswith("different properties")
    # exp86 round 4: statistics that mix the two quantities are not returned,
    # so an answer cannot put them in a table
    assert set(out["stats"]) == {"n_samples", "mean_a", "mean_b", "correlation"}
    assert out["statistics_left_out"] == [
        "agreement_fraction",
        "max_abs_diff",
        "mean_abs_diff",
        "mean_diff_b_minus_a",
        "rmse_between_models",
        "tolerance",
    ]


@pytest.mark.asyncio
async def test_the_models_nodata_value_is_dropped_too(httpx_mock: HTTPXMock) -> None:
    """A band with no declared range: the model's wizard nodata_value decides."""
    for rid in ("a1", "b1"):
        record = {
            "id": rid,
            "prediction_id": f"p-{rid}",
            "result_metadata": {"geometry": {"type": "Polygon", "coordinates": _BOX}},
        }
        httpx_mock.add_response(
            url=f"{BASE}/prediction-results/{rid}", json={"records": [record]}
        )
        httpx_mock.add_response(
            url=f"{BASE}/predictions/p-{rid}",
            json={"records": [{"id": f"p-{rid}", "model_id": "m1"}]},
        )
    httpx_mock.add_response(
        url=f"{BASE}/models/m1",
        json={"records": [{"id": "m1", "wizard_answers": {"nodata_value": 255}}]},
    )
    values = [255.0 if i % 3 == 0 else float(i) for i in range(9)]

    def pixel(request: httpx.Request) -> httpx.Response:
        q = parse_qs(urlparse(str(request.url)).query)
        key = (round(float(q["lon"][0]), 6), round(float(q["lat"][0]), 6))
        i = grid_points([0.0, 0.0, 10.0, 10.0], 3).index(key)
        band = {
            "property_name": "count",
            "raw_value": values[i],
            "classification": None,
        }
        return httpx.Response(200, json={"records": [{"bands": [band]}]})

    httpx_mock.add_callback(
        pixel, url=re.compile(r".*/pixel-value\?.*"), is_reusable=True
    )
    async with StudioClient(StudioConfig(api_key="k", base_url=BASE)) as studio:
        ctx = ToolContext(studio=studio, state=ThreadState())
        out = await _compare({"result_ids": ["a1", "b1"], "grid": 3}, ctx)
    assert out["n_nodata_dropped"] == 3
    assert out["stats"]["n_samples"] == 6
    assert out["stats"]["max_abs_diff"] == 0.0


def _sentinel_values(n: int) -> list[float]:
    """Every third point is Studio's -1 sentinel; the rest rise from 0.1."""
    return [-1.0 if i % 3 == 0 else round(0.1 + 0.05 * i, 3) for i in range(n)]


@pytest.mark.asyncio
async def test_compare_group_counts_and_drops_nodata(httpx_mock: HTTPXMock) -> None:
    prop = "sample_karst_score"
    ids = ["g1", "g2", "g3"]
    for rid in ids:
        httpx_mock.add_response(
            url=f"{BASE}/prediction-results/{rid}", json=_result(rid, prop)
        )
    values = _sentinel_values(9)
    callback = _pixel_callback({rid: (prop, values) for rid in ids}, 3)
    httpx_mock.add_callback(
        callback, url=re.compile(r".*/pixel-value\?.*"), is_reusable=True
    )
    async with StudioClient(StudioConfig(api_key="k", base_url=BASE)) as studio:
        ctx = ToolContext(studio=studio, state=ThreadState())
        out = await _compare({"result_ids": ids, "grid": 3, "mode": "group"}, ctx)
    assert out["n_nodata_dropped"] == 9  # 3 sentinel points x 3 results
    assert out["nodata_by_result"] == {"g1": 3, "g2": 3, "g3": 3}
    assert out["ensemble"]["n_points_used"] == 6
    assert all(p["stats"]["n_samples"] == 6 for p in out["pairwise"])


@pytest.mark.asyncio
async def test_ensemble_uncertainty_drops_nodata_draws(httpx_mock: HTTPXMock) -> None:
    prop = "sample_karst_score"
    for rid in ("e1", "e2"):
        httpx_mock.add_response(
            url=f"{BASE}/prediction-results/{rid}", json=_result(rid, prop)
        )
    values = _sentinel_values(9)
    callback = _pixel_callback({"e1": (prop, values), "e2": (prop, values)}, 3)
    httpx_mock.add_callback(
        callback, url=re.compile(r".*/pixel-value\?.*"), is_reusable=True
    )
    async with StudioClient(StudioConfig(api_key="k", base_url=BASE)) as studio:
        ctx = ToolContext(studio=studio, state=ThreadState())
        out = await _compare(
            {"result_ids": ["e1", "e2"], "grid": 3, "mode": "ensemble"}, ctx
        )
    assert out["comparable"] is True
    assert out["n_nodata_dropped"] == 6
    assert out["n_points_dropped"] == 3


@pytest.mark.asyncio
async def test_ensemble_uncertainty_refuses_different_properties(
    httpx_mock: HTTPXMock,
) -> None:
    httpx_mock.add_response(
        url=f"{BASE}/prediction-results/e1", json=_result("e1", "sample_karst_score")
    )
    httpx_mock.add_response(
        url=f"{BASE}/prediction-results/e2", json=_result("e2", "sample_number")
    )
    async with StudioClient(StudioConfig(api_key="k", base_url=BASE)) as studio:
        ctx = ToolContext(studio=studio, state=ThreadState())
        out = await _compare({"result_ids": ["e1", "e2"], "mode": "ensemble"}, ctx)
    assert out["comparable"] is False
    assert "different properties" in out["reason"]
    assert not any("pixel-value" in str(r.url) for r in httpx_mock.get_requests())


@pytest.mark.asyncio
async def test_trace_shifts_drops_nodata(httpx_mock: HTTPXMock) -> None:
    prop = "sample_karst_score"
    dates = {"t1": "2024-01-01", "t2": "2024-02-01", "t3": "2024-03-01"}
    for rid, start in dates.items():
        httpx_mock.add_response(
            url=f"{BASE}/prediction-results/{rid}",
            json=_result(rid, prop, prediction_id=f"p-{rid}"),
        )
        httpx_mock.add_response(
            url=f"{BASE}/predictions/p-{rid}",
            json={
                "records": [{"id": f"p-{rid}", "start_time": start, "model_id": "m1"}]
            },
        )
    httpx_mock.add_response(
        url=f"{BASE}/models/m1", json={"records": [{"id": "m1", "wizard_answers": {}}]}
    )
    values = _sentinel_values(9)
    callback = _pixel_callback({rid: (prop, values) for rid in dates}, 3)
    httpx_mock.add_callback(
        callback, url=re.compile(r".*/pixel-value\?.*"), is_reusable=True
    )
    async with StudioClient(StudioConfig(api_key="k", base_url=BASE)) as studio:
        ctx = ToolContext(studio=studio, state=ThreadState())
        out = await _compare(
            {"result_ids": list(dates), "grid": 3, "mode": "series"}, ctx
        )
    assert out["comparable"] is True
    assert out["n_nodata_dropped"] == 9
    # Identical valid values at every date: no shift once no-data is out.
    assert all(step["stats"]["mean_diff_b_minus_a"] == 0.0 for step in out["steps"])


@pytest.mark.asyncio
async def test_single_point_pixel_value_reports_the_sentinel_as_nodata(
    httpx_mock: HTTPXMock,
) -> None:
    httpx_mock.add_response(
        url=re.compile(r".*/prediction-results/r1/pixel-value\?.*"),
        json={"records": [{"bands": [_band("sample_karst_score", -1.0)]}]},
    )
    async with StudioClient(StudioConfig(api_key="k", base_url=BASE)) as studio:
        ctx = ToolContext(studio=studio, state=ThreadState())
        out = await _predict_tool("olmoearth_pixel_value").handler(
            {"result_id": "r1", "lon": 1.0, "lat": 2.0}, ctx
        )
    assert out["available"] is False
    assert out["nodata"] is True
    assert out["value"] is None
    assert out["raw_value"] == -1.0
    assert out["declared_range"] == [0.0, 1.0]


@pytest.mark.asyncio
async def test_result_listings_describe_outputs_without_geometry(
    httpx_mock: HTTPXMock,
) -> None:
    body = _result("r9", "sample_karst_score", prediction_id="p1")
    body["records"][0]["result_metadata"]["start_datetime"] = "2025-01-01T00:00:00Z"
    httpx_mock.add_response(url=f"{BASE}/prediction-results/r9", json=body)
    async with StudioClient(StudioConfig(api_key="k", base_url=BASE)) as studio:
        ctx = ToolContext(studio=studio, state=ThreadState())
        out = await _predict_tool("olmoearth_get_prediction_result").handler(
            {"result_id": "r9"}, ctx
        )
    assert out["outputs"] == [
        {
            "property_name": "sample_karst_score",
            "value_type": "regression",
            "min_value": 0.0,
            "max_value": 1.0,
        }
    ]
    assert out["period"][0] == "2025-01-01T00:00:00Z"
    assert "'coordinates'" not in repr(out)
    assert out["result_metadata"]["geometry"] == "omitted (rule 3.1)"
    assert out["result_metadata"]["regression_fields"][0]["max_value"] == 1.0


@pytest.mark.asyncio
async def test_studio_class_objects_compare_by_label(httpx_mock: HTTPXMock) -> None:
    """Studio's classification is a {label, color} object; it reads as its label."""
    meta = {
        "geometry": {"type": "Polygon", "coordinates": _BOX},
        "classification_fields": [
            {
                "property_name": "lc",
                "allowed_values": [
                    {"value": 1, "label": "water", "color": [0, 0, 255]},
                    {"value": 2, "label": "forest", "color": [0, 255, 0]},
                ],
            }
        ],
    }
    for rid in ("c1", "c2"):
        httpx_mock.add_response(
            url=f"{BASE}/prediction-results/{rid}",
            json={
                "records": [
                    {"id": rid, "property_names": ["lc"], "result_metadata": meta}
                ]
            },
        )
    points = grid_points([0.0, 0.0, 10.0, 10.0], 2)

    def pixel(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        q = parse_qs(urlparse(url).query)
        i = points.index((round(float(q["lon"][0]), 6), round(float(q["lat"][0]), 6)))
        if i == 3:  # off-legend raw value: no class -> no-data
            band = {"property_name": "lc", "raw_value": 255, "classification": None}
        else:
            label = "forest" if (i == 0 and "/c2/" in url) else "water"
            band = {
                "property_name": "lc",
                "raw_value": 1,
                "classification": {"label": label, "color": [0, 0, 255, 255]},
            }
        return httpx.Response(200, json={"records": [{"bands": [band]}]})

    httpx_mock.add_callback(
        pixel, url=re.compile(r".*/pixel-value\?.*"), is_reusable=True
    )
    async with StudioClient(StudioConfig(api_key="k", base_url=BASE)) as studio:
        ctx = ToolContext(studio=studio, state=ThreadState())
        out = await _compare({"result_ids": ["c1", "c2"], "grid": 2}, ctx)
    assert out["value_type"] == "classification"
    assert out["n_nodata_dropped"] == 1
    assert out["stats"] == {
        "n_samples": 3,
        "agreement_fraction": 0.6667,
        "n_disagree": 1,
    }


@pytest.mark.asyncio
async def test_model_lookup_is_best_effort(httpx_mock: HTTPXMock) -> None:
    """A missing prediction or an unusable nodata_value never fails the tool."""
    from olmoearth_agent.tools.sampling import model_for_result, result_nodata_context

    httpx_mock.add_response(url=f"{BASE}/predictions/gone", status_code=404)
    httpx_mock.add_response(
        url=f"{BASE}/predictions/p2", json={"records": [{"id": "p2", "model_id": "m9"}]}
    )
    httpx_mock.add_response(
        url=f"{BASE}/models/m9",
        json={"records": [{"id": "m9", "wizard_answers": {"nodata_value": "n/a"}}]},
    )
    async with StudioClient(StudioConfig(api_key="k", base_url=BASE)) as studio:
        ctx = ToolContext(studio=studio, state=ThreadState())
        assert await model_for_result(ctx, {"prediction_id": "gone"}) is None
        assert await model_for_result(ctx, {}) is None
        nd = await result_nodata_context(ctx, {"prediction_id": "p2"})
    assert nd.model is not None and nd.model["model_id"] == "m9"
    assert nd.nodata_value is None
