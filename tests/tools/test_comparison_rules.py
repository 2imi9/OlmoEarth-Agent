# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""The claims a comparison's output forbids (the output contract's ids).

exp86's blind audit of rounds 6 and 7: answers offered "per-window error
rates, classification metrics" for a regression declared 0.2 to 1.2 with no
threshold, a label sample to find which of two properties' maps is closer,
and "a third dated map" to settle which of two dated maps is right. The
comparisons now name those claims in ``forbidden_claims``; their numbers are
unchanged.
"""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import parse_qs, urlparse

import httpx
import pytest
from pytest_httpx import HTTPXMock

from olmoearth_agent.analysis.raster_compare import grid_points
from olmoearth_agent.harness.state import ThreadState
from olmoearth_agent.llm.types import ToolCall
from olmoearth_agent.studio.client import StudioClient, StudioConfig
from olmoearth_agent.tools import statistical_rules as rules
from olmoearth_agent.tools.compare import build_compare_tools
from olmoearth_agent.tools.registry import ToolContext, ToolRegistry
from olmoearth_agent.tools.review_set import build_review_set_tools

BASE = "http://mock-studio/api/v1"


async def _result(
    name: str, args: dict[str, Any], ctx: ToolContext | None = None
) -> Any:
    registry = ToolRegistry()
    registry.register_all(build_review_set_tools())
    registry.register_all(build_compare_tools())
    out = await registry.dispatch(
        ToolCall(id="1", name=name, arguments=args),
        ctx or ToolContext(studio=None, state=ThreadState()),  # type: ignore[arg-type]
    )
    assert out["ok"] is True, out
    return out["result"]


def _ids(out: dict[str, Any]) -> set[str]:
    """The forbidden ids a result carries; the tools and the estimation rules
    each add theirs, so the order is not part of the contract."""
    return {c["id"] for c in out.get("forbidden_claims", [])}


def _why(out: dict[str, Any], claim_id: str) -> str:
    (why,) = [c["why"] for c in out["forbidden_claims"] if c["id"] == claim_id]
    return why


@pytest.mark.asyncio
async def test_a_comparison_across_dates_is_not_settled_by_another_date() -> None:
    """exp86 rounds 6 and 7 (B3/cluster): "run a third dated map"."""
    pytest.importorskip("oe_inferencex.compare")  # the dates reading
    a = [[0.2, 0.8], [0.7, 0.3], [0.4, 0.6]]
    b = [[0.2, 0.8], [0.3, 0.7], [0.4, 0.6]]

    async def compare(**dates: str) -> dict[str, Any]:
        args: dict[str, Any] = {"scores_a": a, "scores_b": b, **dates}
        result: dict[str, Any] = await _result("olmoearth_compare_review", args)
        return result

    apart = await compare(
        date_a="2022-01-01/2022-12-31", date_b="2023-01-01/2023-12-31"
    )
    assert apart["dates"]["status"] == "different_time"
    assert _ids(apart) == {rules.ANOTHER_DATE_SETTLES_IT, rules.WINNER_WITHOUT_LABELS}
    another = _why(apart, rules.ANOTHER_DATE_SETTLES_IT)
    assert "(2022-01-01/2022-12-31 and 2023-01-01/2023-12-31)" in another
    assert "cannot tell which" in another
    assert "different or overlapping periods" in another
    assert "more confident side" in _why(apart, rules.WINNER_WITHOUT_LABELS)
    overlap = await compare(
        date_a="2022-01-01/2022-12-31",
        date_b="2022-06-01/2023-05-31",
        labels_date="2022-07-01",
    )
    assert overlap["dates"]["status"] == "overlapping_time"
    assert _ids(overlap) == {
        rules.ANOTHER_DATE_SETTLES_IT,
        rules.WINNER_WITHOUT_LABELS,
    }
    # The reason covers the overlapping periods, not only disjoint ones.
    assert "different or overlapping periods" in _why(
        overlap, rules.ANOTHER_DATE_SETTLES_IT
    )
    assert "labels dated 2022-07-01" in _why(overlap, rules.WINNER_WITHOUT_LABELS)
    # No labels are taken at any dates: the same time, or none given, still
    # names no winner (and another date is not at issue there).
    for same in (
        await compare(date_a="2024-03-01", date_b="2024-03-01"),
        await compare(),
    ):
        assert _ids(same) == {rules.WINNER_WITHOUT_LABELS}
        assert "more confident side" in _why(same, rules.WINNER_WITHOUT_LABELS)
    # The counts the parity check reads are untouched.
    assert apart["n_differing"] == 1 and apart["share_differing"] == pytest.approx(
        1 / 3
    )


_BOX = [[[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]]
_RANGES = {"sample_karst_score": (0.0, 1.0), "sample_number": (0.2, 1.2)}


def _record(rid: str, prop: str) -> dict[str, Any]:
    lo, hi = _RANGES[prop]
    return {
        "records": [
            {
                "id": rid,
                "property_names": [prop],
                "result_metadata": {
                    "geometry": {"type": "Polygon", "coordinates": _BOX},
                    "regression_fields": [
                        {"property_name": prop, "min_value": lo, "max_value": hi}
                    ],
                },
            }
        ]
    }


def _mock_pair(
    httpx_mock: HTTPXMock, props: dict[str, str], grid: int, *, sampled: bool = True
) -> None:
    for rid, prop in props.items():
        httpx_mock.add_response(
            url=f"{BASE}/prediction-results/{rid}",
            json=_record(rid, prop),
            is_reusable=True,
        )
    if not sampled:  # refused from the records, before any pixel-value call
        return
    points = grid_points([0.0, 0.0, 10.0, 10.0], grid)
    index = {pt: i for i, pt in enumerate(points)}

    def pixel(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        q = parse_qs(urlparse(url).query)
        i = index[(round(float(q["lon"][0]), 6), round(float(q["lat"][0]), 6))]
        rid = next(r for r in props if f"/{r}/" in url)
        lo, hi = _RANGES[props[rid]]
        band = {
            "property_name": props[rid],
            "raw_value": lo + (hi - lo) * ((i * 7 + len(rid)) % 10) / 10,
            "classification": None,
            "regression": {"min_value": lo, "max_value": hi},
        }
        return httpx.Response(200, json={"records": [{"bands": [band]}]})

    httpx_mock.add_callback(
        pixel, url=re.compile(r".*/pixel-value\?.*"), is_reusable=True
    )


@pytest.mark.asyncio
async def test_two_properties_carry_no_combined_statistic_no_winner_and_no_error_rate(
    httpx_mock: HTTPXMock,
) -> None:
    """exp86 rounds 6 and 7 (B3/studio): KarstBinary against KarstNumber; the
    answers offered "per-window error rates, classification metrics" for the
    0.2-1.2 regression and a label sample to find which map is closer."""
    _mock_pair(httpx_mock, {"a1": "sample_karst_score", "b1": "sample_number"}, 4)
    async with StudioClient(StudioConfig(api_key="k", base_url=BASE)) as studio:
        ctx = ToolContext(studio=studio, state=ThreadState())
        refused = await _result(
            "olmoearth_compare_results", {"result_ids": ["a1", "b1"], "grid": 4}, ctx
        )
        allowed = await _result(
            "olmoearth_compare_results",
            {
                "result_ids": ["a1", "b1"],
                "grid": 4,
                "allow_different_properties": True,
            },
            ctx,
        )
    assert refused["comparable"] is False
    assert _ids(refused) == {
        rules.COMBINED_STATISTIC_ACROSS_PROPERTIES,
        rules.WINNER_WITHOUT_LABELS,
    }
    assert allowed["comparable"] is True
    assert _ids(allowed) == {
        rules.ERROR_RATE_FOR_UNTHRESHOLDED_REGRESSION,
        rules.COMBINED_STATISTIC_ACROSS_PROPERTIES,
        rules.WINNER_WITHOUT_LABELS,
    }
    no_rate = _why(allowed, rules.ERROR_RATE_FOR_UNTHRESHOLDED_REGRESSION)
    assert no_rate.startswith("'sample_number' (declared range [0.2, 1.2])")
    assert "sample_karst_score" not in no_rate  # a [0, 1] score is decided at 0.5
    combined = _why(allowed, rules.COMBINED_STATISTIC_ACROSS_PROPERTIES)
    assert "['sample_karst_score', 'sample_number']" in combined
    # One winner claim, the one that says what the correlation does say.
    assert "rise and fall together" in _why(allowed, rules.WINNER_WITHOUT_LABELS)
    # The statistics are the ones returned before the rules were added.
    assert set(allowed["stats"]) == {"n_samples", "mean_a", "mean_b", "correlation"}


@pytest.mark.asyncio
async def test_a_group_of_two_properties_refused_carries_the_claims(
    httpx_mock: HTTPXMock,
) -> None:
    _mock_pair(
        httpx_mock,
        {"a1": "sample_karst_score", "b1": "sample_karst_score", "c1": "sample_number"},
        3,
        sampled=False,
    )
    async with StudioClient(StudioConfig(api_key="k", base_url=BASE)) as studio:
        ctx = ToolContext(studio=studio, state=ThreadState())
        refused = await _result(
            "olmoearth_compare_results",
            {"result_ids": ["a1", "b1", "c1"], "mode": "group", "grid": 3},
            ctx,
        )
    assert refused["comparable"] is False and refused["mode"] == "group"
    assert _ids(refused) == {
        rules.COMBINED_STATISTIC_ACROSS_PROPERTIES,
        rules.WINNER_WITHOUT_LABELS,
    }


@pytest.mark.asyncio
async def test_two_unit_scores_of_one_property_name_no_winner(
    httpx_mock: HTTPXMock,
) -> None:
    """No labels are ever given to a comparison: one property's two maps
    carry the winner claim too, and nothing else ([0, 1] scores)."""
    _mock_pair(httpx_mock, {"a1": "sample_karst_score", "b1": "sample_karst_score"}, 3)
    async with StudioClient(StudioConfig(api_key="k", base_url=BASE)) as studio:
        ctx = ToolContext(studio=studio, state=ThreadState())
        out = await _result(
            "olmoearth_compare_results", {"result_ids": ["a1", "b1"], "grid": 3}, ctx
        )
    assert out["comparable"] is True and out["value_type"] == "regression"
    assert _ids(out) == {rules.WINNER_WITHOUT_LABELS}
    assert "no labels were used" in _why(out, rules.WINNER_WITHOUT_LABELS)


@pytest.mark.asyncio
async def test_one_unthresholded_property_has_no_error_rate(
    httpx_mock: HTTPXMock,
) -> None:
    _mock_pair(httpx_mock, {"a1": "sample_number", "b1": "sample_number"}, 3)
    async with StudioClient(StudioConfig(api_key="k", base_url=BASE)) as studio:
        ctx = ToolContext(studio=studio, state=ThreadState())
        out = await _result(
            "olmoearth_compare_results", {"result_ids": ["a1", "b1"], "grid": 3}, ctx
        )
    assert _ids(out) == {
        rules.ERROR_RATE_FOR_UNTHRESHOLDED_REGRESSION,
        rules.WINNER_WITHOUT_LABELS,
    }
    # one property, named once
    assert (
        _why(out, rules.ERROR_RATE_FOR_UNTHRESHOLDED_REGRESSION).count("sample_number")
        == 1
    )
