# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""olmoearth_review_set_from_result: a Studio result as review windows (mocked Studio).

The trial's brief 2 asked which windows of a KarstBinary prediction a reviewer
should check first. The model sampled pixels by hand and put the 0.97 and 0.99
cells first, the reverse of the margin ranking it claimed to follow. This tool
samples the result itself and ranks by the margin, so the least decided
windows lead and the most confident come last.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import httpx
import pytest
from pytest_httpx import HTTPXMock

from olmoearth_agent.analysis.raster_compare import grid_windows
from olmoearth_agent.harness.state import ThreadState
from olmoearth_agent.llm.types import ToolCall
from olmoearth_agent.studio.client import StudioClient, StudioConfig
from olmoearth_agent.tools.registry import ToolContext, ToolRegistry
from olmoearth_agent.tools.review_set import build_review_set_tools

BASE = "http://mock-studio/api/v1"
_BBOX = [0.0, 0.0, 8.0, 4.0]
_BOX = [[[0, 0], [8, 0], [8, 4], [0, 4], [0, 0]]]

#: A 4x4 grid of KarstBinary-like scores, row-major from the north-west;
#: -1 is Studio's no-data sentinel.
_GRID = [
    [0.97, 0.52, -1.0, 0.03],
    [0.31, 0.99, 0.05, 0.47],
    [-1.0, 0.60, 0.10, 0.88],
    [0.02, 0.45, 0.93, -1.0],
]

_BANNED_KEYS = {"lon", "lat", "bbox", "coordinates", "geometry", "centres_lon_lat"}


def _keys(obj: Any) -> set[str]:
    """Every dict key anywhere inside ``obj``."""
    if isinstance(obj, dict):
        out = set(obj)
        for v in obj.values():
            out |= _keys(v)
        return out
    if isinstance(obj, list):
        out = set()
        for v in obj:
            out |= _keys(v)
        return out
    return set()


def _record(rid: str, meta: dict[str, Any], prop: str = "sample_karst_score") -> dict:
    return {
        "records": [
            {
                "id": rid,
                "property_names": [prop],
                "result_metadata": {
                    "geometry": {"type": "Polygon", "coordinates": _BOX},
                    **meta,
                },
            }
        ]
    }


_BINARY_META = {
    "regression_fields": [
        {"property_name": "sample_karst_score", "min_value": 0.0, "max_value": 1.0}
    ]
}


def _mock_pixels(
    httpx_mock: HTTPXMock, values: list[list[float]], block: dict | None
) -> None:
    """Serve ``values[row][col]`` at each window centre of a 4x4 grid."""
    cells = {(lo, la): (r, c) for r, c, lo, la in grid_windows(_BBOX, 4)}

    def pixel(request: httpx.Request) -> httpx.Response:
        q = parse_qs(urlparse(str(request.url)).query)
        r, c = cells[(round(float(q["lon"][0]), 6), round(float(q["lat"][0]), 6))]
        band: dict[str, Any] = {
            "band_index": 1,
            "property_name": "sample_karst_score",
            "raw_value": values[r][c],
            "classification": None,
        }
        if block is not None:
            band["regression"] = block
        return httpx.Response(200, json={"records": [{"bands": [band]}]})

    httpx_mock.add_callback(
        pixel, url=re.compile(r".*/pixel-value\?.*"), is_reusable=True
    )


@pytest.fixture(autouse=True)
def _workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The tool writes the evidence text to the workspace; keep it temporary."""
    monkeypatch.setenv("OLMOEARTH_OUTPUT_ROOT", str(tmp_path / "workspace"))


def _tool() -> Any:
    return next(
        t
        for t in build_review_set_tools()
        if t.spec.name == "olmoearth_review_set_from_result"
    )


@pytest.mark.asyncio
async def test_binary_score_ranks_the_least_decided_windows_first(
    httpx_mock: HTTPXMock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OLMOEARTH_SCORES_ROOT", str(tmp_path))
    httpx_mock.add_response(
        url=f"{BASE}/prediction-results/kb", json=_record("kb", _BINARY_META)
    )
    _mock_pixels(httpx_mock, _GRID, {"min_value": 0.0, "max_value": 1.0})
    async with StudioClient(StudioConfig(api_key="k", base_url=BASE)) as studio:
        ctx = ToolContext(studio=studio, state=ThreadState())
        out = await _tool().handler(
            {"result_id": "kb", "grid": 4, "budgets": [0.25, 0.5]}, ctx
        )
    assert out["ranked"] is True
    assert out["score_kind"] == "binary_score"
    assert "P(positive)" in out["assumption"]
    assert "not probabilities of error" in out["assumption"]
    sampling = out["sampling"]
    assert (sampling["n_windows_sampled"], sampling["n_valid"]) == (16, 13)
    assert sampling["n_nodata_dropped"] == 3
    assert "not every pixel" in sampling["what_a_window_is"]
    assert [b["n_review"] for b in out["budgets"]] == [3, 6]

    review = out["review"]
    # 0.52, 0.47, 0.45, 0.60, 0.31 lead; the 0.97 / 0.99 / 0.02 cells never do.
    assert [(r["row"], r["col"]) for r in review[:5]] == [
        (0, 1),
        (1, 3),
        (3, 1),
        (2, 1),
        (1, 0),
    ]
    assert review[0]["window_index"] == 1 and review[0]["score"] == 0.52
    assert review[0]["margin"] == pytest.approx(0.04)
    assert [r["margin"] for r in review] == sorted(r["margin"] for r in review)
    confident = {(0, 0), (1, 1), (3, 0)}
    assert not confident & {(r["row"], r["col"]) for r in review}
    assert review[0]["predicted_class"] == 1 and review[4]["predicted_class"] == 0
    assert "never first" in out["signal"]
    assert any("not a sample" in c for c in out["caveats"])

    # Rule §3.1: windows by (row, col) and index only; locations stay in the file.
    assert not _BANNED_KEYS & _keys(out)
    saved = json.loads(Path(out["scores_path"]).read_text())
    assert saved["grid"] == [4, 4] and len(saved["windows"]) == 13
    assert saved["scores"][0] == pytest.approx([0.03, 0.97])
    assert len(saved["centres_lon_lat"]) == 13
    assert "olmoearth_plan_label_sample" in out["next_step"]


@pytest.mark.asyncio
async def test_the_saved_scores_feed_olmoearth_review_set(
    httpx_mock: HTTPXMock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The file keeps grid indices, so the generic tool names the same windows."""
    monkeypatch.setenv("OLMOEARTH_SCORES_ROOT", str(tmp_path))
    httpx_mock.add_response(
        url=f"{BASE}/prediction-results/kb", json=_record("kb", _BINARY_META)
    )
    _mock_pixels(httpx_mock, _GRID, {"min_value": 0.0, "max_value": 1.0})
    tools = {t.spec.name: t for t in build_review_set_tools()}
    async with StudioClient(StudioConfig(api_key="k", base_url=BASE)) as studio:
        ctx = ToolContext(studio=studio, state=ThreadState())
        first = await tools["olmoearth_review_set_from_result"].handler(
            {"result_id": "kb", "grid": 4, "budgets": [0.25]}, ctx
        )
        again = await tools["olmoearth_review_set"].handler(
            {"scores_path": first["scores_path"], "budget": 0.25}, ctx
        )
    assert [r["window_index"] for r in again["review"]] == [
        r["window_index"] for r in first["review"][:3]
    ]
    assert again["review"][0]["row"] == 0 and again["review"][0]["col"] == 1
    # The file says what its rows are, so the generic tool scopes it the same way.
    assert again["evidence_covers_this_case"] == "no"
    assert again["must_state"] == first["must_state"]


@pytest.mark.asyncio
async def test_a_studio_band_is_scoped_as_no_recorded_experiment_grades_it(
    httpx_mock: HTTPXMock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """exp86 round 7: answers applied the suite's result to a regression band the
    tool said no experiment grades. The one scope sentence says so and must be
    stated; the full evidence text is in a file, not inline."""
    monkeypatch.setenv("OLMOEARTH_SCORES_ROOT", str(tmp_path))
    httpx_mock.add_response(
        url=f"{BASE}/prediction-results/kb", json=_record("kb", _BINARY_META)
    )
    _mock_pixels(httpx_mock, _GRID, {"min_value": 0.0, "max_value": 1.0})
    async with StudioClient(StudioConfig(api_key="k", base_url=BASE)) as studio:
        ctx = ToolContext(studio=studio, state=ThreadState())
        out = await _tool().handler({"result_id": "kb", "grid": 4}, ctx)
    assert "evidence" not in out
    assert out["evidence_covers_this_case"] == "no"
    assert (
        "no recorded experiment grades a regression score read as a probability"
        in out["evidence_scope"]
    )
    assert out["must_state"] == [
        "No recorded experiment grades a regression score read as a probability."
    ]
    assert [f["id"] for f in out["facts"]] == ["margin_ratio"]
    assert [c["id"] for c in out["forbidden_claims"]] == ["error_rate_without_labels"]
    assert len(out["caveats"]) == 2
    detail = json.loads(Path(out["evidence_detail_path"]).read_text())
    assert "suite-margin-wins-every-task" in detail["ranking"]["claims"]


@pytest.mark.asyncio
async def test_a_classification_band_is_refused_before_sampling(
    httpx_mock: HTTPXMock,
) -> None:
    meta = {
        "classification_fields": [
            {
                "property_name": "landcover",
                "allowed_values": [
                    {"value": 1, "label": "water", "color": [0, 0, 255]}
                ],
            }
        ]
    }
    httpx_mock.add_response(
        url=f"{BASE}/prediction-results/lc", json=_record("lc", meta, prop="landcover")
    )
    async with StudioClient(StudioConfig(api_key="k", base_url=BASE)) as studio:
        ctx = ToolContext(studio=studio, state=ThreadState())
        out = await _tool().handler({"result_id": "lc"}, ctx)
    assert out["ranked"] is False
    assert "no margin can be recovered from a hard class" in out["reason"]
    assert out["use_instead"] == "olmoearth_compare_results"
    assert "mode='ensemble'" in out["reason"]
    assert not any("pixel-value" in str(r.url) for r in httpx_mock.get_requests())


@pytest.mark.asyncio
async def test_another_range_needs_a_threshold(
    httpx_mock: HTTPXMock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OLMOEARTH_SCORES_ROOT", str(tmp_path))
    meta = {
        "regression_fields": [
            {"property_name": "sample_karst_score", "min_value": 0.0, "max_value": 10.0}
        ]
    }
    httpx_mock.add_response(
        url=f"{BASE}/prediction-results/cnt",
        json=_record("cnt", meta),
        is_reusable=True,
    )
    counts = [[v * 10 if v >= 0 else -1.0 for v in row] for row in _GRID]
    _mock_pixels(httpx_mock, counts, {"min_value": 0.0, "max_value": 10.0})
    async with StudioClient(StudioConfig(api_key="k", base_url=BASE)) as studio:
        ctx = ToolContext(studio=studio, state=ThreadState())
        refused = await _tool().handler({"result_id": "cnt", "grid": 4}, ctx)
        ranked = await _tool().handler(
            {"result_id": "cnt", "grid": 4, "threshold": 5.0, "error_rate": 0.2}, ctx
        )
    assert refused["ranked"] is False
    assert "needs a decision threshold" in refused["reason"]
    assert refused["declared_range"] == [0.0, 10.0]
    # exp86 round 8 (brief 3 on Studio): the most ambiguous windows of a band
    # with no threshold were offered for review; the refusal forbids it.
    claims = {c["id"]: c["why"] for c in refused["forbidden_claims"]}
    assert set(claims) == {
        "review_set_for_unthresholded_regression",
        "error_rate_for_unthresholded_regression",
    }
    review = claims["review_set_for_unthresholded_regression"]
    assert review.startswith("'sample_karst_score' (declared range [0, 10])")
    assert "needs a threshold for this band" in review
    assert ranked["ranked"] is True
    assert "review_set_for_unthresholded_regression" not in {
        c["id"] for c in ranked["forbidden_claims"]
    }
    assert ranked["score_kind"] == "threshold_distance"
    assert ranked["review"][0]["window_index"] == 1  # 5.2, nearest the threshold
    assert ranked["review"][0]["margin"] == pytest.approx(0.02)
    ceiling = ranked["budgets"][-1]["attainable_ceiling"]
    assert ceiling == pytest.approx(min(1.0, 0.10 / 0.2))


@pytest.mark.asyncio
async def test_undeclared_metadata_is_read_from_the_sampled_band(
    httpx_mock: HTTPXMock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OLMOEARTH_SCORES_ROOT", str(tmp_path))
    httpx_mock.add_response(url=f"{BASE}/prediction-results/u1", json=_record("u1", {}))
    _mock_pixels(httpx_mock, _GRID, {"min_value": 0.0, "max_value": 1.0})
    registry = ToolRegistry()
    registry.register_all(build_review_set_tools())
    async with StudioClient(StudioConfig(api_key="k", base_url=BASE)) as studio:
        ctx = ToolContext(studio=studio, state=ThreadState())
        out = await registry.dispatch(
            ToolCall(
                id="1",
                name="olmoearth_review_set_from_result",
                arguments={"result_id": "u1", "grid": 4, "save_scores": False},
            ),
            ctx,
        )
    assert out["ok"] is True
    result = out["result"]
    assert result["sampling"]["n_nodata_dropped"] == 3
    assert result["declared_range"] == [0.0, 1.0]
    assert "scores_path" not in result


def _windows_file(path: Path, windows: list[int], values: list[float]) -> str:
    path.write_text(
        json.dumps(
            {"grid": [3, 3], "windows": windows, "scores": [[1 - v, v] for v in values]}
        )
    )
    return str(path)


@pytest.mark.asyncio
async def test_compare_review_maps_windows_back_to_the_grid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OLMOEARTH_SCORES_ROOT", str(tmp_path))
    windows = [0, 1, 2, 4, 5, 8]
    a = _windows_file(tmp_path / "a.json", windows, [0.1, 0.9, 0.2, 0.8, 0.3, 0.7])
    b = _windows_file(tmp_path / "b.json", windows, [0.1, 0.9, 0.2, 0.2, 0.3, 0.7])
    other = _windows_file(tmp_path / "c.json", [0, 1, 2, 3, 5, 8], [0.5] * 6)
    registry = ToolRegistry()
    registry.register_all(build_review_set_tools())
    ctx = ToolContext(studio=None, state=ThreadState())  # type: ignore[arg-type]
    out = await registry.dispatch(
        ToolCall(
            id="1",
            name="olmoearth_compare_review",
            arguments={
                "scores_path_a": a,
                "scores_path_b": b,
                "date_a": "2024-05-01",
            },
        ),
        ctx,
    )
    assert out["ok"] is True
    result = out["result"]
    assert [(d["window_index"], d["row"], d["col"]) for d in result["differing"]] == [
        (4, 1, 1)
    ]
    assert result["which_side_is_right"].startswith("not graded: only one map's date")
    mismatch = await registry.dispatch(
        ToolCall(
            id="2",
            name="olmoearth_compare_review",
            arguments={"scores_path_a": a, "scores_path_b": other},
        ),
        ctx,
    )
    assert mismatch["ok"] is False and "different windows" in mismatch["error"]
    boundary = await registry.dispatch(
        ToolCall(
            id="3",
            name="olmoearth_review_set",
            arguments={"scores_path": a, "order": "boundary_first"},
        ),
        ctx,
    )
    assert boundary["ok"] is False and "no-data windows" in boundary["error"]


@pytest.mark.asyncio
async def test_margin_summary_labels_its_fields_and_gives_the_unlisted_range(
    httpx_mock: HTTPXMock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """exp86 round 2 (brief 2, Studio run 3): "all other windows' scores ranged
    from ~0.96 to ~0.99 ... (median margin 0.965)": the median read as the
    lower end, where the lowest margin outside the listed windows was 0.786."""
    monkeypatch.setenv("OLMOEARTH_SCORES_ROOT", str(tmp_path))
    httpx_mock.add_response(
        url=f"{BASE}/prediction-results/kb", json=_record("kb", _BINARY_META)
    )
    _mock_pixels(httpx_mock, _GRID, {"min_value": 0.0, "max_value": 1.0})
    async with StudioClient(StudioConfig(api_key="k", base_url=BASE)) as studio:
        ctx = ToolContext(studio=studio, state=ThreadState())
        out = await _tool().handler(
            {"result_id": "kb", "grid": 4, "budgets": [0.25], "max_listed": 3}, ctx
        )
    ms = out["margin_summary"]
    # The 13 valid windows' margins |2s - 1|, ascending:
    # 0.04 0.06 0.10 | 0.20 0.38 0.76 0.80 0.86 0.90 0.94 0.94 0.96 0.98
    assert not {"min", "median", "max"} & set(ms)
    assert ms["n_windows"] == 13
    assert ms["lowest_margin"] == pytest.approx(0.04)
    assert ms["median_margin"] == pytest.approx(0.8)
    assert ms["highest_margin"] == pytest.approx(0.98)
    assert ms["listed"]["n"] == 3
    assert ms["listed"]["margin_range"] == pytest.approx([0.04, 0.1])
    assert ms["not_listed"]["n"] == 10
    assert ms["not_listed"]["margin_range"] == pytest.approx([0.2, 0.98])
    assert "not a lower end" in ms["reading"]
    assert "not_listed" in ms["reading"]


@pytest.mark.asyncio
async def test_margin_summary_of_a_fully_listed_review_set() -> None:
    """The same summary on olmoearth_review_set; nothing unlisted, no range."""
    tool = next(
        t for t in build_review_set_tools() if t.spec.name == "olmoearth_review_set"
    )
    out = await tool.handler(
        {"scores": [[0.9, 0.1], [0.4, 0.6], [0.55, 0.45]], "budget": 1.0},
        ToolContext(studio=None, state=ThreadState()),  # type: ignore[arg-type]
    )
    ms = out["margin_summary"]
    assert ms["listed"]["n"] == 3 and ms["not_listed"] == {"n": 0, "margin_range": None}
    assert ms["lowest_margin"] == pytest.approx(0.1)
    assert ms["median_margin"] == pytest.approx(0.2)
