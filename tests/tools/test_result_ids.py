# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""A prediction id where a result id is asked (``tools/result_ids.py``; mocked Studio).

A user often has a prediction's id at hand where a tool asks for one of its
results. Studio then answers the result read with 404 (or a 200 with errors);
the tool asks whether the id is a prediction's and, when that prediction has
exactly one result, runs once more with it, saying so in ``resolved_ids``.
None of these tests reaches a network: Studio is ``pytest_httpx``'s mock.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import httpx
import pytest
from pytest_httpx import HTTPXMock

from olmoearth_agent.analysis.raster_compare import grid_windows
from olmoearth_agent.bridge.replay import ToolCache
from olmoearth_agent.bridge.store import RunDir
from olmoearth_agent.harness.state import ThreadState
from olmoearth_agent.llm.types import ToolCall
from olmoearth_agent.provenance.log import ProvenanceLog
from olmoearth_agent.skills import build_default_registry
from olmoearth_agent.studio.client import StudioClient, StudioConfig
from olmoearth_agent.tools import predict
from olmoearth_agent.tools.registry import ToolContext, ToolRegistry
from olmoearth_agent.tools.result_ids import RESULT_ID_ARGS
from olmoearth_agent.tools.review_set import slug
from olmoearth_agent.tools.statistical_rules import CARD_MAX_ROWS

BASE = "http://mock-studio/api/v1"

#: A prediction, its result, another result, and an id that is neither.
P = "0a5f8e2c-1d3b-4c6e-9f70-1a2b3c4d5e6f"
R = "7b9c1d2e-3f40-4a51-8b62-7c8d9e0f1a2b"
R2 = "c3d4e5f6-0718-4293-a4b5-c6d7e8f9a0b1"
X = "e1f2a3b4-c5d6-4e7f-8091-a2b3c4d5e6f7"

#: The tools whose result-id arguments are resolved (results-ids spec, B1).
RESOLVED = {
    "olmoearth_plan_label_sample": ("result_id", "result_id_b"),
    "olmoearth_review_set_from_result": ("result_id",),
    "olmoearth_scores_from_studio": ("result_id",),
    "olmoearth_compare_results": ("result_ids",),
    "olmoearth_get_prediction_result": ("result_id",),
    "olmoearth_pixel_value": ("result_id",),
}

_BOX = [[[0, 0], [8, 0], [8, 4], [0, 4], [0, 0]]]
_BBOX = [0.0, 0.0, 8.0, 4.0]
_SCORES = [
    [0.97, 0.52, -1.0, 0.03],
    [0.31, 0.99, 0.05, 0.47],
    [-1.0, 0.60, 0.10, 0.88],
    [0.02, 0.45, 0.93, -1.0],
]


@pytest.fixture(autouse=True)
def _roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every file a tool writes lands in this test's folder."""
    for env, folder in (
        ("OLMOEARTH_OUTPUT_ROOT", "workspace"),
        ("OLMOEARTH_SCORES_ROOT", "scores"),
    ):
        (tmp_path / folder).mkdir()
        monkeypatch.setenv(env, str(tmp_path / folder))


def _registry() -> ToolRegistry:
    """The default registry, as both dispatch paths use it."""
    return build_default_registry()


async def _dispatch(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """One call through ``ToolRegistry.dispatch`` against the mocked Studio."""
    async with StudioClient(StudioConfig(api_key="k", base_url=BASE)) as studio:
        ctx = ToolContext(studio=studio, state=ThreadState())
        return await _registry().dispatch(
            ToolCall(id="t1", name=name, arguments=arguments), ctx
        )


def _asked(httpx_mock: HTTPXMock) -> list[tuple[str, str]]:
    """Each request Studio was sent: its method and path."""
    return [(r.method, r.url.path) for r in httpx_mock.get_requests()]


def _record(rid: str, **extra: Any) -> dict[str, Any]:
    """A result record as ``GET /prediction-results/{id}`` answers it."""
    return {"records": [{"id": rid, "property_names": ["kelp"], **extra}]}


def _not_found(httpx_mock: HTTPXMock, path: str) -> None:
    """Studio answers ``GET path`` with 404."""
    httpx_mock.add_response(
        url=f"{BASE}{path}",
        status_code=404,
        json={"records": [], "errors": [{"code": "not_found_error"}]},
    )


def _a_prediction(
    httpx_mock: HTTPXMock, results: list[dict[str, Any]], status: str = "completed"
) -> None:
    """P is a prediction with ``results``: its record, then its results' search."""
    httpx_mock.add_response(
        url=f"{BASE}/predictions/{P}",
        json={"records": [{"id": P, "name": "kelp run", "status": status}]},
    )
    httpx_mock.add_response(
        url=f"{BASE}/prediction-results/search",
        method="POST",
        json={"records": results, "meta": {"total": len(results)}},
    )


def _result_of_p(rid: str, *props: str) -> dict[str, Any]:
    """A search record of one of P's results."""
    return {"id": rid, "prediction_id": P, "property_names": list(props or ["kelp"])}


@pytest.mark.asyncio
async def test_a_prediction_with_one_result_is_read_as_that_result(
    httpx_mock: HTTPXMock,
) -> None:
    """Upper case and spaces too: the id is written the API's way first."""
    _not_found(httpx_mock, f"/prediction-results/{P}")
    _a_prediction(httpx_mock, [_result_of_p(R)])
    httpx_mock.add_response(
        url=f"{BASE}/prediction-results/{R}", json=_record(R, prediction_id=P)
    )
    arguments = {"result_id": f"  {P.upper()} "}
    out = await _dispatch("olmoearth_get_prediction_result", arguments)
    assert out["ok"] is True
    result = out["result"]
    assert result["result_id"] == R and result["prediction_id"] == P
    assert result["resolved_ids"] == [
        {"argument": "result_id", "prediction_id": P, "result_id": R}
    ]
    assert arguments == {"result_id": f"  {P.upper()} "}  # the caller's dict
    assert _asked(httpx_mock) == [
        ("GET", f"/api/v1/prediction-results/{P}"),
        ("GET", f"/api/v1/predictions/{P}"),
        ("POST", "/api/v1/prediction-results/search"),
        ("GET", f"/api/v1/prediction-results/{R}"),
    ]
    search = json.loads(httpx_mock.get_requests()[2].content)
    assert search["prediction_id"] == {"eq": P}


@pytest.mark.asyncio
async def test_a_200_carrying_errors_starts_the_check_too(
    httpx_mock: HTTPXMock,
) -> None:
    """Studio's client raises "Studio API error" for an empty 200 with errors."""
    httpx_mock.add_response(
        url=f"{BASE}/prediction-results/{P}",
        json={"records": [], "errors": [{"code": "not_found_error"}]},
    )
    _a_prediction(httpx_mock, [_result_of_p(R)])
    httpx_mock.add_response(url=f"{BASE}/prediction-results/{R}", json=_record(R))
    out = await _dispatch("olmoearth_get_prediction_result", {"result_id": P})
    assert out["ok"] is True
    assert out["result"]["result_id"] == R
    assert out["result"]["resolved_ids"][0]["prediction_id"] == P


@pytest.mark.asyncio
async def test_a_prediction_with_several_results_is_refused_with_their_list(
    httpx_mock: HTTPXMock,
) -> None:
    _not_found(httpx_mock, f"/prediction-results/{P}")
    _a_prediction(
        httpx_mock,
        [_result_of_p(R, "kelp"), _result_of_p(R2, "landcover", "landcover_rank1")],
    )
    out = await _dispatch("olmoearth_get_prediction_result", {"result_id": P})
    assert out["ok"] is False
    error = out["error"]
    assert error.startswith("ValueError: result_id ")
    assert f"{P} is a prediction id, not a result id" in error
    assert f"Its 2 results: {R} (kelp); {R2} (landcover, landcover_rank1)" in error
    assert "Pass one of them as result_id" in error
    assert "or ask the user which" in error
    assert "more;" not in error  # two are listed whole
    # Neither result is read: which one is the caller's to say.
    assert ("GET", f"/api/v1/prediction-results/{R}") not in _asked(httpx_mock)
    assert len(_asked(httpx_mock)) == 3


@pytest.mark.asyncio
async def test_a_prediction_with_no_result_yet_names_its_status(
    httpx_mock: HTTPXMock,
) -> None:
    _not_found(httpx_mock, f"/prediction-results/{P}")
    _a_prediction(httpx_mock, [], status="running")
    out = await _dispatch("olmoearth_get_prediction_result", {"result_id": P})
    assert out["ok"] is False
    assert "is a prediction id, not a result id, and it has no results yet" in (
        out["error"]
    )
    assert "(status running)" in out["error"]
    assert "olmoearth_get_prediction gives its status" in out["error"]


@pytest.mark.asyncio
async def test_an_id_that_is_neither_keeps_the_error_studio_gave(
    httpx_mock: HTTPXMock,
) -> None:
    """The first outcome stands, word for word, when no prediction is found."""
    _not_found(httpx_mock, f"/prediction-results/{X}")
    _not_found(httpx_mock, f"/prediction-results/{X}")
    _not_found(httpx_mock, f"/predictions/{X}")
    async with StudioClient(StudioConfig(api_key="k", base_url=BASE)) as studio:
        ctx = ToolContext(studio=studio, state=ThreadState())
        with pytest.raises(httpx.HTTPStatusError) as raised:
            await predict._get_prediction_result({"result_id": X}, ctx)
    out = await _dispatch("olmoearth_get_prediction_result", {"result_id": X})
    assert out["ok"] is False
    assert out["error"] == f"HTTPStatusError: {raised.value}"
    assert "resolved_ids" not in json.dumps(out)
    assert _asked(httpx_mock)[1:] == [
        ("GET", f"/api/v1/prediction-results/{X}"),
        ("GET", f"/api/v1/predictions/{X}"),
    ]


@pytest.mark.asyncio
async def test_a_refused_key_during_the_check_is_the_refusal(
    httpx_mock: HTTPXMock,
) -> None:
    """A 403 on the prediction read is raised as it came: the sign-in still works."""
    _not_found(httpx_mock, f"/prediction-results/{P}")
    httpx_mock.add_response(
        url=f"{BASE}/predictions/{P}", status_code=403, json={"errors": []}
    )
    out = await _dispatch("olmoearth_get_prediction_result", {"result_id": P})
    assert out["ok"] is False
    assert out["error"].startswith(
        f"StudioAccessError: Studio refused GET /predictions/{P} (HTTP 403)"
    )
    assert out["hint"].startswith("Studio refused this read")


@pytest.mark.asyncio
async def test_a_studio_outage_never_asks_for_a_prediction(
    httpx_mock: HTTPXMock,
) -> None:
    """A 503 is retried by the client and is no sign of a prediction id."""
    for _ in range(3):
        httpx_mock.add_response(url=f"{BASE}/prediction-results/{P}", status_code=503)
    out = await _dispatch("olmoearth_get_prediction_result", {"result_id": P})
    assert out["ok"] is False
    assert "503" in out["error"]
    assert not [p for _m, p in _asked(httpx_mock) if "/predictions/" in p]
    assert len(_asked(httpx_mock)) == 3


@pytest.mark.asyncio
async def test_one_result_in_two_spellings_is_one_result_to_compare(
    httpx_mock: HTTPXMock,
) -> None:
    """compare_results de-duplicates the API's spelling, so R and R.upper() are one."""
    out = await _dispatch("olmoearth_compare_results", {"result_ids": [R.upper(), R]})
    assert out["ok"] is True
    assert "need >= 2 DISTINCT result_ids (got 1)" in out["result"]["reason"]
    assert httpx_mock.get_requests() == []


@pytest.mark.asyncio
async def test_a_two_map_plan_of_one_result_in_two_spellings_is_refused(
    httpx_mock: HTTPXMock,
) -> None:
    pytest.importorskip("oe_inferencex.plan")
    out = await _dispatch(
        "olmoearth_plan_label_sample",
        {
            "result_id": R.upper(),
            "result_id_b": R,
            "design": "disagreement",
            "budget": 10,
        },
    )
    assert out["ok"] is False
    assert f"map a and map b are the same result ({R})" in out["error"]
    assert httpx_mock.get_requests() == []


@pytest.mark.asyncio
async def test_an_id_with_a_character_no_id_has_is_refused_before_studio(
    httpx_mock: HTTPXMock,
) -> None:
    """No '/', '?', '#' or '%' of the model's reaches a request's path."""
    for bad in ("a/b", "r?x", f"{R}#x", "50%"):
        out = await _dispatch("olmoearth_get_prediction_result", {"result_id": bad})
        assert out["ok"] is False
        assert "result_id must be a Studio result id, a UUID" in out["error"]
        assert "nothing was asked of Studio" in out["error"]
    out = await _dispatch("olmoearth_compare_results", {"result_ids": ["ok-1", "r?x"]})
    assert out["ok"] is False
    assert "result_ids[1] must be a Studio result id" in out["error"]
    assert httpx_mock.get_requests() == []


@pytest.mark.asyncio
async def test_a_prediction_among_results_to_compare_is_replaced_in_its_place(
    httpx_mock: HTTPXMock,
) -> None:
    """Each id is checked in order; only the prediction's place changes."""
    for _ in range(2):  # the first run, then the run with P's result
        httpx_mock.add_response(url=f"{BASE}/prediction-results/{R2}", json=_record(R2))
    _not_found(httpx_mock, f"/prediction-results/{P}")
    _not_found(httpx_mock, f"/predictions/{R2}")
    _a_prediction(httpx_mock, [_result_of_p(R)])
    httpx_mock.add_response(url=f"{BASE}/prediction-results/{R}", json=_record(R))
    out = await _dispatch("olmoearth_compare_results", {"result_ids": [R2, P]})
    assert out["ok"] is True
    result = out["result"]
    # The records carry no extent: the compare refuses, on the ids it read.
    assert result["comparable"] is False and result["result_ids"] == [R2, R]
    assert result["resolved_ids"] == [
        {"argument": "result_ids[1]", "prediction_id": P, "result_id": R}
    ]


@pytest.mark.asyncio
async def test_the_pixel_value_of_a_prediction_is_read_from_its_one_result(
    httpx_mock: HTTPXMock,
) -> None:
    """The pixel value catches its errors: its 404 reason starts the check."""
    httpx_mock.add_response(
        url=re.compile(rf".*/prediction-results/{P}/pixel-value\?.*"), status_code=404
    )
    _a_prediction(httpx_mock, [_result_of_p(R)])
    band = {"property_name": "kelp", "raw_value": 0.73, "classification": None}
    httpx_mock.add_response(
        url=re.compile(rf".*/prediction-results/{R}/pixel-value\?.*"),
        json={"records": [{"bands": [band]}]},
    )
    out = await _dispatch(
        "olmoearth_pixel_value", {"result_id": P, "lon": 1.5, "lat": 2.5}
    )
    assert out["ok"] is True
    result = out["result"]
    assert result["available"] is True and result["value"] == 0.73
    assert result["result_id"] == R and result["prediction_id"] == P
    assert result["resolved_ids"][0]["result_id"] == R


@pytest.mark.asyncio
async def test_a_real_off_raster_404_keeps_its_answer(httpx_mock: HTTPXMock) -> None:
    """A result's own 404 (off its raster) costs one prediction read, no more."""
    httpx_mock.add_response(
        url=re.compile(rf".*/prediction-results/{R}/pixel-value\?.*"), status_code=404
    )
    _not_found(httpx_mock, f"/predictions/{R}")
    out = await _dispatch(
        "olmoearth_pixel_value", {"result_id": R, "lon": 1.5, "lat": 2.5}
    )
    assert out["ok"] is True
    result = out["result"]
    assert result["available"] is False
    assert "404 Not Found" in result["reason"]
    assert "resolved_ids" not in result and "prediction_id" not in result
    assert len(_asked(httpx_mock)) == 2


def _mock_result_r(httpx_mock: HTTPXMock, *, reusable: bool = False) -> None:
    """P's one result R, a [0, 1] score answered at each window of a 4x4 grid
    (R's record read once, or as often as asked with ``reusable``)."""
    _not_found(httpx_mock, f"/prediction-results/{P}")
    _a_prediction(httpx_mock, [_result_of_p(R, "sample_karst_score")])
    meta = {
        "geometry": {"type": "Polygon", "coordinates": _BOX},
        "regression_fields": [
            {"property_name": "sample_karst_score", "min_value": 0.0, "max_value": 1.0}
        ],
    }
    httpx_mock.add_response(
        url=f"{BASE}/prediction-results/{R}",
        json={
            "records": [
                {
                    "id": R,
                    "property_names": ["sample_karst_score"],
                    "result_metadata": meta,
                }
            ]
        },
        is_reusable=reusable,
    )
    cells = {(lo, la): (r, c) for r, c, lo, la in grid_windows(_BBOX, 4)}

    def pixel(request: httpx.Request) -> httpx.Response:
        """R's score at the window whose centre was asked."""
        assert f"/prediction-results/{R}/" in request.url.path
        q = parse_qs(urlparse(str(request.url)).query)
        r, c = cells[(round(float(q["lon"][0]), 6), round(float(q["lat"][0]), 6))]
        value = {
            "band_index": 1,
            "property_name": "sample_karst_score",
            "raw_value": _SCORES[r][c],
            "classification": None,
            "regression": {"min_value": 0.0, "max_value": 1.0},
        }
        return httpx.Response(200, json={"records": [{"bands": [value]}]})

    httpx_mock.add_callback(
        pixel, url=re.compile(r".*/pixel-value\?.*"), is_reusable=True
    )


def _args_hash(arguments: dict[str, Any]) -> str:
    """The provenance log's hash of a call's arguments (sorted-key JSON)."""
    blob = json.dumps(arguments, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


@pytest.mark.asyncio
async def test_a_review_set_of_a_prediction_through_the_registry(
    httpx_mock: HTTPXMock,
) -> None:
    """The handler gets R: its scores file is named from R, and its card says so."""
    _mock_result_r(httpx_mock)
    arguments = {"result_id": P, "grid": 4, "budgets": [0.25]}
    out = await _dispatch("olmoearth_review_set_from_result", arguments)
    assert out["ok"] is True, out
    result = out["result"]
    assert result["ranked"] is True
    assert result["resolved_ids"] == [
        {"argument": "result_id", "prediction_id": P, "result_id": R}
    ]
    assert Path(result["scores_path"]).name.startswith(f"scores_{slug(R, 12)}_")
    assert slug(P, 12) not in Path(result["scores_path"]).name
    saved = json.loads(Path(result["scores_path"]).read_text())
    assert saved["result_id"] == R
    assert result["card"][-1] == ["Result used", R, f"the one result of prediction {P}"]
    assert arguments == {"result_id": P, "grid": 4, "budgets": [0.25]}


@pytest.mark.asyncio
async def test_a_review_set_of_a_prediction_through_the_mcp_app(
    httpx_mock: HTTPXMock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Provenance hashes the call as the model made it; the result names R."""
    pytest.importorskip("mcp")
    from olmoearth_agent.mcp_app import auth, server

    monkeypatch.setenv("OLMOEARTH_TOOL_RESULT_SPILL_BYTES", "0")  # the text whole
    _mock_result_r(httpx_mock)
    keys = auth.KeyStore(
        {
            auth.KEY_ENV: "sk_test_result_ids_789",
            auth.KEY_FILE_ENV: str(tmp_path / "cfg" / "studio.env"),
        }
    )
    app = server.AppServer(None, keys=keys, base_url=BASE)
    arguments = {"result_id": P, "grid": 4, "budgets": [0.25]}
    try:
        called = await app.call_agent_tool(
            "olmoearth_review_set_from_result", arguments, views=True
        )
        entry = app.state.provenance.entries[-1]
    finally:
        await app.aclose()
    assert called.isError is False
    assert entry.api_call == "olmoearth_review_set_from_result"
    as_made = {"result_id": P, "grid": 4, "budgets": [0.25]}
    assert arguments == as_made
    assert entry.request_hash == _args_hash(as_made)
    assert entry.request_hash != _args_hash({**as_made, "result_id": R})
    assert entry.prediction_id == P
    texts = [getattr(c, "text", "") for c in called.content]
    # Round 4: a ranked review set of a Studio result says first that no
    # recorded experiment supports its order, and how few pixels it read.
    assert texts[0].startswith(
        "No recorded experiment supports this order on a Studio result"
    )
    body = json.loads(texts[-1])
    assert body["result"]["resolved_ids"][0]["result_id"] == R
    assert f"scores_{slug(R, 12)}_" in body["result"]["scores_path"]
    # The card is the panel's, not the model's.
    assert "Result used" not in "".join(texts)


@pytest.mark.asyncio
async def test_a_plan_of_a_prediction_is_repeated_with_its_result_id(
    httpx_mock: HTTPXMock,
) -> None:
    """The plan ran on R: its result names R as its result_id, so the provenance
    recipe repeats it with R (as the design file names it), not with P."""
    pytest.importorskip("oe_inferencex.plan")
    _mock_result_r(httpx_mock, reusable=True)
    arguments = {"result_id": P, "grid": 4, "budget": 5, "design": "random"}
    out = await _dispatch("olmoearth_plan_label_sample", arguments)
    assert out["ok"] is True, out
    result = out["result"]
    assert result["result_id"] == R and result["prediction_id"] == P
    design = json.loads(Path(result["design_path"]).read_text(encoding="utf-8"))
    assert design["population"]["source"]["result_id"] == R
    assert arguments["result_id"] == P  # the call as made
    log = ProvenanceLog()
    log.record_tool_call("olmoearth_plan_label_sample", arguments, out)
    (call,) = log.verification_calls()
    assert call["result_id"] == R
    script = log.replay_script()
    assert f"result_id={R}" in script and f"result_id={P}" not in script


@pytest.mark.asyncio
async def test_a_card_holds_the_tools_rows_then_one_row_per_id_resolved(
    httpx_mock: HTTPXMock,
) -> None:
    """At most six rows the tool writes, then "Result used" (Usage.md, the
    CHANGELOG); the panel keeps them all."""
    _mock_result_r(httpx_mock)
    out = await _dispatch(
        "olmoearth_review_set_from_result",
        {"result_id": P, "grid": 4, "budgets": [0.25, 0.5, 0.75]},
    )
    assert out["ok"] is True, out
    card = out["result"]["card"]
    assert len(card) == CARD_MAX_ROWS + 1
    assert [row[0] for row in card].count("Result used") == 1
    assert card[-1] == ["Result used", R, f"the one result of prediction {P}"]
    pytest.importorskip("mcp")
    from olmoearth_agent.mcp_app import server

    # The panel keeps every row, a two-map plan's two resolved ids too.
    assert server._card(card) == card
    assert server.CARD_VIEW_ROWS >= CARD_MAX_ROWS + 2


@pytest.mark.asyncio
async def test_a_replayed_resolved_call_asks_studio_nothing(
    httpx_mock: HTTPXMock, tmp_path: Path
) -> None:
    """The bridge caches the one handler call, resolution and all."""
    _not_found(httpx_mock, f"/prediction-results/{P}")
    _a_prediction(httpx_mock, [_result_of_p(R)])
    httpx_mock.add_response(url=f"{BASE}/prediction-results/{R}", json=_record(R))
    run = RunDir(tmp_path / "run")
    run.path.mkdir()
    name = "olmoearth_get_prediction_result"
    tool = _registry().tool_of(name)
    assert tool is not None
    async with StudioClient(StudioConfig(api_key="k", base_url=BASE)) as studio:
        ctx = ToolContext(studio=studio, state=ThreadState())
        live = ToolCache(run, saved=0, model_call=1)
        first = await live.wrap(name, tool.handler)({"result_id": P}, ctx)
        asked = len(httpx_mock.get_requests())
        replay = ToolCache(run, saved=1, model_call=2)
        again = await replay.wrap(name, tool.handler)({"result_id": P}, ctx)
    assert first["result_id"] == R and first["resolved_ids"][0]["prediction_id"] == P
    assert again == first
    assert asked == 4 and len(httpx_mock.get_requests()) == asked


def test_the_tools_that_take_a_result_id_are_the_ones_resolved() -> None:
    """Each named argument is a property of the tool's schema; nothing else is wrapped."""
    registry = _registry()
    wrapped = {}
    for name in registry.names():
        tool = registry.tool_of(name)
        assert tool is not None
        names = getattr(tool.handler, RESULT_ID_ARGS, None)
        if names is not None:
            wrapped[name] = names
    assert wrapped == RESOLVED
    for name, names in wrapped.items():
        spec = registry.spec_of(name)
        assert spec is not None
        assert set(names) <= set(spec.parameters["properties"]), name
    # No spec changed: the registry's size and descriptions are as before.
    assert len(registry.names()) == 25
