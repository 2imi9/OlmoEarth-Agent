# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Design-based estimation through olmoearth-inferencex (the ``inferencex`` extra).

These run the real package (``uv sync --all-extras``); the missing-extra path
is simulated. The trial's brief 4 ("how wrong is this map, I can label 300
windows") got a hand-made mix of stratified and targeted windows and a
simple-random-sample interval that does not apply to it; these tools replace
that with the package's design, estimate, per-class accuracy and zone.
"""

from __future__ import annotations

import csv
import json
import random
import re
import sys
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import httpx
import pytest
from pytest_httpx import HTTPXMock

from olmoearth_agent.analysis.raster_compare import grid_windows
from olmoearth_agent.analysis.review_set import margins
from olmoearth_agent.harness.state import ThreadState
from olmoearth_agent.llm.types import ToolCall
from olmoearth_agent.studio.client import StudioClient, StudioConfig
from olmoearth_agent.tools.estimation import build_estimation_tools
from olmoearth_agent.tools.registry import ToolContext, ToolRegistry
from olmoearth_agent.tools.review_set import build_review_set_tools

estimate = pytest.importorskip("oe_inferencex.estimate")

BASE = "http://mock-studio/api/v1"
_BANNED_KEYS = {"lon", "lat", "bbox", "coordinates", "geometry", "centres_lon_lat"}


def _ctx() -> ToolContext:
    return ToolContext(studio=None, state=ThreadState())  # type: ignore[arg-type]


def _registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register_all(build_estimation_tools())
    registry.register_all(build_review_set_tools())
    return registry


async def _call(name: str, args: dict[str, Any], ctx: ToolContext | None = None) -> Any:
    out = await _registry().dispatch(
        ToolCall(id="1", name=name, arguments=args), ctx or _ctx()
    )
    return out


def _keys(obj: Any) -> set[str]:
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


def _map(n: int = 1200, seed: int = 7) -> tuple[list[list[float]], list[int]]:
    """Three-class logits with errors concentrated at low margin, and the truth."""
    rng = random.Random(seed)
    scores, wrong = [], []
    for _ in range(n):
        row = [rng.gauss(0.0, 1.5) for _ in range(3)]
        top = sorted(row, reverse=True)
        scores.append(row)
        wrong.append(1 if rng.random() < 0.6 * 2.718 ** -(top[0] - top[1]) else 0)
    return scores, wrong


@pytest.fixture(autouse=True)
def _scores_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("OLMOEARTH_SCORES_ROOT", str(tmp_path))
    return tmp_path


@pytest.mark.asyncio
async def test_confidence_design_then_the_estimate_it_earns() -> None:
    scores, truth = _map()
    plan = await _call("olmoearth_plan_label_sample", {"scores": scores, "budget": 300})
    assert plan["ok"] is True
    plan = plan["result"]
    assert plan["design"] == "confidence" and plan["n_population"] == 1200
    assert sum(plan["allocation"]) == 300
    assert any("not a sample" in r for r in plan["interval_rules"])
    assert any("does not apply to a stratified" in r for r in plan["interval_rules"])
    design = json.loads(Path(plan["design_path"]).read_text())
    indices = design["sample"]["indices"]
    assert [w["window_index"] for w in plan["windows"]] == indices[:50]
    wrong = [truth[i] for i in indices]

    out = (
        await _call(
            "olmoearth_estimate_map_error",
            {"design_path": plan["design_path"], "wrong": wrong},
        )
    )["result"]
    # Exactly the package's numbers on the same sample, nothing re-derived.
    direct = estimate.estimate_error_rate(
        estimate.sample_for_estimation(
            [m for m in margins(scores)],
            300,
            design="confidence",
            p1=design_p1(scores),
            seed=0,
        ),
        wrong,
    )
    assert out["estimate"] == pytest.approx(direct["estimate"])
    assert (out["low"], out["high"]) == pytest.approx((direct["low"], direct["high"]))
    assert out["method"] == direct["method"]
    assert "stratified" in out["method"]
    assert out["low"] <= sum(truth) / len(truth) <= out["high"]
    assert "as returned" in out["report"]


def design_p1(scores: list[list[float]]) -> list[float]:
    """Top-1 softmax probability, as the tool computes it for logits."""
    import math

    out = []
    for row in scores:
        top = max(row)
        out.append(1.0 / sum(math.exp(v - top) for v in row))
    return out


@pytest.mark.asyncio
async def test_per_class_accuracy_from_reference_classes() -> None:
    scores, truth = _map()
    plan = (
        await _call("olmoearth_plan_label_sample", {"scores": scores, "budget": 240})
    )["result"]
    design = json.loads(Path(plan["design_path"]).read_text())
    idx = design["sample"]["indices"]
    mc = design["population"]["map_class"]
    wrong = [truth[i] for i in idx]
    reference = [mc[i] if w == 0 else (mc[i] + 1) % 3 for i, w in zip(idx, wrong)]
    out = (
        await _call(
            "olmoearth_estimate_map_error",
            {
                "design_path": plan["design_path"],
                "wrong": wrong,
                "reference": reference,
            },
        )
    )["result"]
    per_class = out["per_class"]
    assert per_class["n_classes"] == 3
    assert set(per_class["per_class"]) == {"0", "1", "2"}
    assert 0.0 <= per_class["overall_accuracy"]["estimate"] <= 1.0


@pytest.mark.asyncio
async def test_labels_can_come_back_through_the_csv() -> None:
    scores, truth = _map(400)
    plan = (
        await _call("olmoearth_plan_label_sample", {"scores": scores, "budget": 60})
    )["result"]
    sheet = Path(plan["labels_csv_path"])
    rows = list(csv.DictReader(sheet.open()))
    assert len(rows) == 60 and rows[0]["wrong"] == ""
    # One blank label is refused, naming the count.
    rows_filled = [dict(r, wrong=str(truth[int(r["window_index"])])) for r in rows]
    rows_filled[5]["wrong"] = ""
    _write(sheet, rows_filled)
    blank = await _call(
        "olmoearth_estimate_map_error",
        {"design_path": plan["design_path"], "labels_path": str(sheet)},
    )
    assert blank["ok"] is False and "1 of 60" in blank["error"]
    rows_filled[5]["wrong"] = str(truth[int(rows_filled[5]["window_index"])])
    _write(sheet, rows_filled)
    out = await _call(
        "olmoearth_estimate_map_error",
        {"design_path": plan["design_path"], "labels_path": str(sheet)},
    )
    assert out["ok"] is True and out["result"]["n_labelled"] == 60


def _write(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@pytest.mark.asyncio
async def test_a_review_set_is_refused_as_a_sample() -> None:
    """The trial's error: pooling targeted windows inflates the error rate."""
    scores, truth = _map()
    marg = margins(scores)
    review = sorted(range(len(scores)), key=marg.__getitem__)[:100]
    out = await _call(
        "olmoearth_estimate_map_error",
        {
            "window_indices": review,
            "wrong": [truth[i] for i in review],
            "scores": scores,
        },
    )
    assert out["ok"] is False
    assert "enriched set, not a sample" in out["error"]
    rng = random.Random(3)
    drawn = rng.sample(range(len(scores)), 100)
    ok = await _call(
        "olmoearth_estimate_map_error",
        {"window_indices": drawn, "wrong": [truth[i] for i in drawn], "scores": scores},
    )
    assert ok["ok"] is True
    assert ok["result"]["method"].startswith("exact hypergeometric")


@pytest.mark.asyncio
async def test_certify_zone_needs_a_random_design() -> None:
    scores, truth = _map(2000, seed=11)
    stratified = (
        await _call("olmoearth_plan_label_sample", {"scores": scores, "budget": 300})
    )["result"]
    refused = (
        await _call(
            "olmoearth_certify_zone",
            {
                "design_path": stratified["design_path"],
                "wrong": [0] * 300,
                "alpha": 0.1,
            },
        )
    )["result"]
    assert refused["certified"] is False and "simple random sample" in refused["reason"]

    plan = (
        await _call(
            "olmoearth_plan_label_sample",
            {"scores": scores, "budget": 300, "design": "random"},
        )
    )["result"]
    design = json.loads(Path(plan["design_path"]).read_text())
    idx = design["sample"]["indices"]
    wrong = [truth[i] for i in idx]
    zone = (
        await _call(
            "olmoearth_certify_zone",
            {"design_path": plan["design_path"], "wrong": wrong, "alpha": 0.3},
        )
    )["result"]
    direct = estimate.certify_zone(
        [m for m in margins(scores)], idx, wrong, 0.3, delta=0.1, rule="prefix"
    )
    assert zone["coverage"] == direct["coverage"]
    assert zone["upper_bound"] == pytest.approx(direct["upper_bound"])
    if zone["certified"]:
        saved = json.loads(Path(zone["zone_path"]).read_text())
        assert len(saved["window_indices"]) == zone["n_zone"]
    assert "zone_indices_in_order" not in zone


@pytest.mark.asyncio
async def test_plan_from_a_sampled_studio_result(httpx_mock: HTTPXMock) -> None:
    """Result -> grid scores -> design: windows by (row, col); no-data stays out."""
    bbox = [0.0, 0.0, 6.0, 6.0]
    box = [[[0, 0], [6, 0], [6, 6], [0, 6], [0, 0]]]
    httpx_mock.add_response(
        url=f"{BASE}/prediction-results/kb",
        json={
            "records": [
                {
                    "id": "kb",
                    "property_names": ["sample_karst_score"],
                    "result_metadata": {
                        "geometry": {"type": "Polygon", "coordinates": box},
                        "regression_fields": [
                            {
                                "property_name": "sample_karst_score",
                                "min_value": 0.0,
                                "max_value": 1.0,
                            }
                        ],
                    },
                }
            ]
        },
        is_reusable=True,
    )
    cells = {(lo, la): r * 6 + c for r, c, lo, la in grid_windows(bbox, 6)}
    rng = random.Random(5)
    values = [-1.0 if i % 7 == 0 else round(rng.random(), 4) for i in range(36)]

    def pixel(request: httpx.Request) -> httpx.Response:
        q = parse_qs(urlparse(str(request.url)).query)
        i = cells[(round(float(q["lon"][0]), 6), round(float(q["lat"][0]), 6))]
        band = {
            "property_name": "sample_karst_score",
            "raw_value": values[i],
            "classification": None,
            "regression": {"min_value": 0.0, "max_value": 1.0},
        }
        return httpx.Response(200, json={"records": [{"bands": [band]}]})

    httpx_mock.add_callback(
        pixel, url=re.compile(r".*/pixel-value\?.*"), is_reusable=True
    )
    async with StudioClient(StudioConfig(api_key="k", base_url=BASE)) as studio:
        ctx = ToolContext(studio=studio, state=ThreadState())
        out = await _call(
            "olmoearth_plan_label_sample",
            {"result_id": "kb", "grid": 6, "budget": 12, "design": "random"},
            ctx,
        )
        too_many = await _call(
            "olmoearth_plan_label_sample",
            {"result_id": "kb", "grid": 6, "budget": 40},
            ctx,
        )
    assert out["ok"] is True
    plan = out["result"]
    assert plan["n_population"] == 36 - 6  # windows 0, 7, 14, 21, 28, 35 are no-data
    nodata = {0, 7, 14, 21, 28, 35}
    assert not nodata & {w["window_index"] for w in plan["windows"]}
    assert all(
        (w["row"], w["col"]) == divmod(w["window_index"], 6) for w in plan["windows"]
    )
    assert "P(positive)" in plan["assumption"]
    assert "sampled points" in plan["population"]["what_the_rate_will_be_of"]
    assert not _BANNED_KEYS & _keys(plan)
    # The reviewer's sheet carries the locations; the chat result does not.
    header = next(csv.reader(Path(plan["labels_csv_path"]).open()))
    assert {"lon", "lat", "wrong"} <= set(header)
    assert too_many["ok"] is False and "30 valid windows" in too_many["error"]


@pytest.mark.asyncio
async def test_plan_from_the_review_set_scores_file(tmp_path: Path) -> None:
    rows = [[1 - s, s] for s in (0.1, 0.9, 0.45, 0.8, 0.3, 0.6, 0.05, 0.95)]
    (tmp_path / "s.json").write_text(
        json.dumps(
            {"grid": [3, 3], "windows": [0, 1, 2, 3, 5, 6, 7, 8], "scores": rows}
        )
    )
    out = (
        await _call(
            "olmoearth_plan_label_sample",
            {"scores_path": str(tmp_path / "s.json"), "budget": 4, "design": "random"},
        )
    )["result"]
    assert out["n_population"] == 8
    assert 4 not in {w["window_index"] for w in out["windows"]}


@pytest.mark.asyncio
async def test_labels_without_a_design_from_a_studio_scores_file(
    tmp_path: Path,
) -> None:
    rows = [[1 - s, s] for s in (0.1, 0.9, 0.45, 0.8, 0.3, 0.6, 0.05, 0.95)]
    (tmp_path / "s.json").write_text(
        json.dumps(
            {
                "result_id": "kb",
                "grid": [3, 3],
                "windows": [0, 1, 2, 3, 5, 6, 7, 8],
                "scores": rows,
                "score_kind": "binary_score",
            }
        )
    )
    out = await _call(
        "olmoearth_estimate_map_error",
        {
            "window_indices": [0, 3, 6, 8],
            "wrong": [0, 1, 0, 0],
            "scores_path": str(tmp_path / "s.json"),
        },
    )
    assert out["ok"] is True
    result = out["result"]
    assert result["estimate"] == 0.25 and result["n_population"] == 8
    assert "sampled grid points" in result["report"]
    outside = await _call(
        "olmoearth_estimate_map_error",
        {"window_indices": [4], "wrong": [1], "scores_path": str(tmp_path / "s.json")},
    )
    assert outside["ok"] is False and "outside the valid map" in outside["error"]


@pytest.mark.asyncio
async def test_inline_scores_take_a_grid_as_n_or_rows_cols() -> None:
    scores, _truth = _map(6)
    for grid in ([2, 3], 3):
        args: dict[str, Any] = {"scores": scores, "budget": 3, "design": "random"}
        args["grid"] = grid
        out = await _call("olmoearth_plan_label_sample", args)
        if grid == 3:  # 3 x 3 = 9 cells for 6 rows: refused, not guessed
            assert out["ok"] is False and "does not match" in out["error"]
            continue
        assert all(
            (w["row"], w["col"]) == divmod(w["window_index"], 3)
            for w in out["result"]["windows"]
        )


@pytest.mark.asyncio
async def test_missing_extra_is_a_clear_answer_not_a_crash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Simulate an install without olmoearth-inferencex (and so without numpy)."""
    monkeypatch.setitem(sys.modules, "oe_inferencex.estimate", None)
    monkeypatch.setitem(sys.modules, "oe_inferencex.compare", None)
    for name, args in (
        ("olmoearth_plan_label_sample", {"scores": [[0.2, 0.8]], "budget": 1}),
        ("olmoearth_estimate_map_error", {"design_path": "x.json", "wrong": [1]}),
        ("olmoearth_certify_zone", {"design_path": "x.json", "alpha": 0.05}),
    ):
        out = await _call(name, args)
        assert out["ok"] is True, name
        assert out["result"]["available"] is False
        assert out["result"]["install"] == "pip install 'olmoearth-agent[inferencex]'"
    scores = [[0.2, 0.8], [0.7, 0.3]]
    cmp = (
        await _call(
            "olmoearth_compare_review",
            {"scores_a": scores, "scores_b": scores, "date_a": "2024-01-01"},
        )
    )["result"]
    assert cmp["dates"]["assessed"] is False
    assert "not assessed" in cmp["dates"]["reason"]


@pytest.mark.asyncio
async def test_compare_review_reads_the_dates_with_the_extra() -> None:
    a = [[0.2, 0.8], [0.7, 0.3], [0.4, 0.6]]
    b = [[0.2, 0.8], [0.3, 0.7], [0.4, 0.6]]
    apart = (
        await _call(
            "olmoearth_compare_review",
            {
                "scores_a": a,
                "scores_b": b,
                "date_a": "2024-03-01",
                "date_b": "2024-09-01",
            },
        )
    )["result"]
    assert apart["dates"]["status"] == "different_time"
    assert apart["dates"]["days_apart"] == 184
    assert apart["which_side_is_right"].startswith("not graded")
    assert "no labels_date" in apart["which_side_is_right"]
    same = (
        await _call(
            "olmoearth_compare_review",
            {
                "scores_a": a,
                "scores_b": b,
                "date_a": "2024-03-01",
                "date_b": "2024-03-01",
            },
        )
    )["result"]
    assert same["dates"]["status"] == "same_time"
    assert same["which_side_is_right"] == "not resolvable without labels"
    unstated = (
        await _call("olmoearth_compare_review", {"scores_a": a, "scores_b": b})
    )["result"]
    assert unstated["dates"]["status"] == "unstated"


def test_population_rows_are_validated() -> None:
    from olmoearth_agent.tools.estimation import population_from_rows

    with pytest.raises(ValueError, match="at least one"):
        population_from_rows([])
    with pytest.raises(ValueError, match=">= 2"):
        population_from_rows([[0.5]])
    with pytest.raises(ValueError, match="not finite"):
        population_from_rows([[0.5, float("nan")]])
    with pytest.raises(ValueError, match="needs its 'grid'"):
        population_from_rows([[0.4, 0.6]], windows=[0])
    with pytest.raises(ValueError, match="1 windows for 2"):
        population_from_rows([[0.4, 0.6], [0.3, 0.7]], grid=(2, 2), windows=[0])
    with pytest.raises(ValueError, match="distinct and inside"):
        population_from_rows([[0.4, 0.6], [0.3, 0.7]], grid=(2, 2), windows=[0, 9])
    pop = population_from_rows([[0.4, 0.6], [0.3, 0.7]], grid=(2, 2), windows=[3, 0])
    assert pop.n_valid == 2 and pop.map_class == [1, -1, -1, 1]
    assert pop.p1[3] == pytest.approx(0.6)


# --------------------------------------------------------------------------- exp86 round 1: the Studio grid
# Brief 4 on a Studio result (exp86 round 1, all three runs): the model asked
# for grids of 20 to 64 and got the same "173 valid windows" each time, because
# the grid was held to 16 without a word; [18, 18] and [30, 30], the form the
# schema advertised, crashed; and the budget error sent it to "a finer grid".

_KB_BBOX = [0.0, 0.0, 6.0, 6.0]


def _kb_value(lon: float, lat: float) -> float:
    """A deterministic [0, 1] score at a point, no-data (-1) on about a third."""
    import math

    frac = math.sin(lon * 12.9898 + lat * 78.233) * 43758.5453
    frac -= math.floor(frac)
    return -1.0 if frac < 0.33 else round(frac, 4)


def _kb_valid(grid: int) -> int:
    """Valid windows of the mocked result at ``grid`` x ``grid``."""
    return sum(
        1 for _r, _c, lo, la in grid_windows(_KB_BBOX, grid) if _kb_value(lo, la) >= 0
    )


def _mock_kb(httpx_mock: HTTPXMock) -> list[int]:
    """Serve result 'kb' (a [0, 1] score band); return a live count of pixel calls."""
    box = [[[0, 0], [6, 0], [6, 6], [0, 6], [0, 0]]]
    httpx_mock.add_response(
        url=f"{BASE}/prediction-results/kb",
        json={
            "records": [
                {
                    "id": "kb",
                    "property_names": ["sample_karst_score"],
                    "result_metadata": {
                        "geometry": {"type": "Polygon", "coordinates": box},
                        "regression_fields": [
                            {
                                "property_name": "sample_karst_score",
                                "min_value": 0.0,
                                "max_value": 1.0,
                            }
                        ],
                    },
                }
            ]
        },
        is_reusable=True,
    )
    calls = [0]

    def pixel(request: httpx.Request) -> httpx.Response:
        calls[0] += 1
        q = parse_qs(urlparse(str(request.url)).query)
        band = {
            "property_name": "sample_karst_score",
            "raw_value": _kb_value(float(q["lon"][0]), float(q["lat"][0])),
            "classification": None,
            "regression": {"min_value": 0.0, "max_value": 1.0},
        }
        return httpx.Response(200, json={"records": [{"bands": [band]}]})

    httpx_mock.add_callback(
        pixel, url=re.compile(r".*/pixel-value\?.*"), is_reusable=True
    )
    return calls


@pytest.mark.asyncio
async def test_a_studio_grid_above_16_is_capped_and_says_so(
    httpx_mock: HTTPXMock,
) -> None:
    calls = _mock_kb(httpx_mock)
    async with StudioClient(StudioConfig(api_key="k", base_url=BASE)) as studio:
        ctx = ToolContext(studio=studio, state=ThreadState())
        out = await _call(
            "olmoearth_plan_label_sample",
            {"result_id": "kb", "grid": 20, "budget": 10, "design": "random"},
            ctx,
        )
    assert out["ok"] is True, out
    sampling = out["result"]["sampling"]
    assert calls[0] == 256  # 16 x 16 pixel-value calls, not 20 x 20
    assert sampling["grid"] == "16x16"
    assert sampling["grid_requested"] == 20 and sampling["grid_capped"] is True
    assert "capped at 16" in sampling["grid_note"]
    assert sampling["n_valid"] == _kb_valid(16)
    assert not _BANNED_KEYS & _keys(out["result"])


@pytest.mark.asyncio
async def test_a_studio_grid_in_range_is_stated_uncapped(httpx_mock: HTTPXMock) -> None:
    _mock_kb(httpx_mock)
    async with StudioClient(StudioConfig(api_key="k", base_url=BASE)) as studio:
        ctx = ToolContext(studio=studio, state=ThreadState())
        out = await _call(
            "olmoearth_plan_label_sample",
            {"result_id": "kb", "grid": 6, "budget": 5, "design": "random"},
            ctx,
        )
    sampling = out["result"]["sampling"]
    assert sampling["grid"] == "6x6"
    assert sampling["grid_requested"] == 6 and sampling["grid_capped"] is False
    assert "grid_note" not in sampling


@pytest.mark.asyncio
async def test_a_studio_grid_takes_the_rows_cols_form(httpx_mock: HTTPXMock) -> None:
    """[N, N], the form the schema advertises, is N; a non-square grid is refused."""
    calls = _mock_kb(httpx_mock)
    async with StudioClient(StudioConfig(api_key="k", base_url=BASE)) as studio:
        ctx = ToolContext(studio=studio, state=ThreadState())
        square = await _call(
            "olmoearth_plan_label_sample",
            {"result_id": "kb", "grid": [8, 8], "budget": 5, "design": "random"},
            ctx,
        )
        capped = await _call(
            "olmoearth_plan_label_sample",
            {"result_id": "kb", "grid": [30, 30], "budget": 5, "design": "random"},
            ctx,
        )
        oblong = await _call(
            "olmoearth_plan_label_sample",
            {"result_id": "kb", "grid": [8, 12], "budget": 5},
            ctx,
        )
    assert square["ok"] is True, square
    assert square["result"]["sampling"]["grid"] == "8x8"
    assert capped["ok"] is True, capped
    assert capped["result"]["sampling"]["grid"] == "16x16"
    assert capped["result"]["sampling"]["grid_capped"] is True
    assert oblong["ok"] is False and "square" in oblong["error"]
    assert "TypeError" not in oblong["error"]
    assert calls[0] == 64 + 256  # the refusal samples nothing


def test_the_plan_schema_states_the_studio_grid_range() -> None:
    spec = next(
        t.spec
        for t in build_estimation_tools()
        if t.spec.name == "olmoearth_plan_label_sample"
    )
    assert "2-16" in spec.parameters["properties"]["grid"]["description"]


@pytest.mark.asyncio
async def test_the_review_set_from_a_result_states_a_capped_grid(
    httpx_mock: HTTPXMock,
) -> None:
    _mock_kb(httpx_mock)
    async with StudioClient(StudioConfig(api_key="k", base_url=BASE)) as studio:
        ctx = ToolContext(studio=studio, state=ThreadState())
        out = await _call(
            "olmoearth_review_set_from_result",
            {"result_id": "kb", "grid": 40, "save_scores": False},
            ctx,
        )
    sampling = out["result"]["sampling"]
    assert sampling["grid"] == "16x16" and sampling["grid_capped"] is True
    assert sampling["grid_requested"] == 40


# --------------------------------------------------------------------------- the budget error


@pytest.mark.asyncio
async def test_a_budget_above_a_capped_studio_grid_names_the_real_ceiling(
    httpx_mock: HTTPXMock,
) -> None:
    _mock_kb(httpx_mock)
    n16 = _kb_valid(16)
    async with StudioClient(StudioConfig(api_key="k", base_url=BASE)) as studio:
        ctx = ToolContext(studio=studio, state=ThreadState())
        out = await _call(
            "olmoearth_plan_label_sample",
            {"result_id": "kb", "grid": 30, "budget": 300},
            ctx,
        )
    assert out["ok"] is False
    error = out["error"]
    assert "16x16 = 256 points" in error
    assert f"{n16} valid windows" in error
    assert f"at most {n16} labels" in error
    assert "grid 30 was capped at 16" in error
    assert "olmoearth_scores_from_file" in error
    assert "finer grid" not in error


@pytest.mark.asyncio
async def test_a_budget_error_offers_a_finer_grid_only_when_one_can_help(
    httpx_mock: HTTPXMock,
) -> None:
    _mock_kb(httpx_mock)
    n10 = _kb_valid(10)
    async with StudioClient(StudioConfig(api_key="k", base_url=BASE)) as studio:
        ctx = ToolContext(studio=studio, state=ThreadState())
        reachable = await _call(
            "olmoearth_plan_label_sample",
            {"result_id": "kb", "budget": n10 + 1},
            ctx,
        )
        unreachable = await _call(
            "olmoearth_plan_label_sample",
            {"result_id": "kb", "budget": 300},
            ctx,
        )
    # The default grid, 10: a finer grid, up to 16, can still help a budget <= 256.
    assert "10x10 = 100 points" in reachable["error"]
    assert "a finer grid, up to 16" in reachable["error"]
    # 300 labels exceed even 16 x 16 = 256 points: no grid reaches them.
    assert "finer grid" not in unreachable["error"]
    assert "olmoearth_scores_from_file" in unreachable["error"]


@pytest.mark.asyncio
async def test_a_budget_above_inline_scores_states_the_ceiling() -> None:
    scores, _truth = _map(40)
    out = await _call("olmoearth_plan_label_sample", {"scores": scores, "budget": 50})
    assert out["ok"] is False
    assert "40 valid windows" in out["error"] and "at most 40 labels" in out["error"]
    assert "finer grid" not in out["error"]
