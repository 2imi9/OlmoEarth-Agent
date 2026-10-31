# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Unit + live tests for the olmoearth-predict tool bundle."""

from __future__ import annotations

import json
import os
import re

import pytest
from pytest_httpx import HTTPXMock

from olmoearth_agent.harness.state import ThreadState
from olmoearth_agent.tools.predict import build_predict_tools
from olmoearth_agent.tools.registry import RegisteredTool, ToolContext

BASE = "http://mock-studio/api/v1"


def _tool(name: str) -> RegisteredTool:
    return next(t for t in build_predict_tools() if t.spec.name == name)


@pytest.mark.asyncio
async def test_search_predictions_tool(httpx_mock: HTTPXMock) -> None:
    from olmoearth_agent.studio.client import StudioClient, StudioConfig

    httpx_mock.add_response(
        url=f"{BASE}/predictions/search",
        method="POST",
        json={
            "records": [
                {"id": "p1", "name": "X", "status": "completed", "model_id": "m1"}
            ],
            "meta": {"total": 1},
        },
    )
    httpx_mock.add_response(
        url=f"{BASE}/models/m1",
        json={
            "records": [
                {
                    "id": "m1",
                    "name": "KarstBinary",
                    "model_type": "fine_tuned",
                    "wizard_answers": {
                        "prediction_type": "per_pixel_regression",
                        "nodata_value": None,
                    },
                }
            ]
        },
    )
    async with StudioClient(StudioConfig(api_key="k", base_url=BASE)) as studio:
        ctx = ToolContext(studio=studio, state=ThreadState())
        result = await _tool("olmoearth_search_predictions").handler({"limit": 5}, ctx)
    assert result["total"] == 1
    assert result["predictions"][0]["model_id"] == "m1"
    # The model is described by what it outputs, read from the model record,
    # so a regression score is never narrated as a "confidence layer".
    model = result["models"]["m1"]
    assert model["prediction_type"] == "per_pixel_regression"
    assert model["model_type"] == "fine_tuned"
    assert "not a class probability" in result["models_note"]
    assert "Ai2 also publishes" in result["models_note"]


@pytest.mark.asyncio
async def test_search_predictions_says_which_models_were_fine_tuned_on_labels(
    httpx_mock: HTTPXMock,
) -> None:
    """exp86 round 9 (B3/studio run 1) answered "no ground-truth labels exist"
    while these records named a label field of the project. The listing says
    which models were fine-tuned on one, without the person who asked for it."""
    from olmoearth_agent.studio.client import StudioClient, StudioConfig

    httpx_mock.add_response(
        url=f"{BASE}/predictions/search",
        method="POST",
        json={
            "records": [
                {"id": "p1", "name": "X", "status": "completed", "model_id": "m1"},
                {"id": "p2", "name": "E", "status": "completed", "model_id": "m2"},
            ],
            "meta": {"total": 2},
        },
    )
    httpx_mock.add_response(
        url=f"{BASE}/models/m1",
        json={
            "records": [
                {
                    "id": "m1",
                    "name": "KarstBinary",
                    "model_type": "fine_tuned",
                    "requester_id": "person-7f3a",
                    "wizard_answers": {
                        "prediction_type": "per_pixel_regression",
                        "label_field_id": "lf-1",
                        "split_proportions": {
                            "train_prop": 0.75,
                            "val_prop": 0.25,
                            "test_prop": 0.0,
                        },
                    },
                }
            ]
        },
    )
    httpx_mock.add_response(
        url=f"{BASE}/models/m2",
        json={
            "records": [
                {
                    "id": "m2",
                    "name": "KarstEmbedding",
                    "model_type": "embeddings",
                    "wizard_answers": {"wizard_id": "embeddings_v1"},
                }
            ]
        },
    )
    async with StudioClient(StudioConfig(api_key="k", base_url=BASE)) as studio:
        ctx = ToolContext(studio=studio, state=ThreadState())
        result = await _tool("olmoearth_search_predictions").handler({}, ctx)
    trained = result["models"]["m1"]
    assert trained["trained_on_labels"] is True
    assert trained["label_field_id"] == "lf-1"
    assert trained["split"] == {"train_prop": 0.75, "val_prop": 0.25, "test_prop": 0.0}
    assert result["models"]["m2"]["trained_on_labels"] is None
    assert "label_field_id" not in result["models"]["m2"]
    (fact,) = result["facts"]
    assert fact["id"] == "labels_in_studio"
    assert fact["sentence"] == (
        "KarstBinary was fine-tuned in Studio on a label field of its project "
        "(train/val split 0.75/0.25), so labels for this project may exist in "
        "Studio; no tool of this run looked them up for this area."
    )
    assert "person-7f3a" not in json.dumps(result)


@pytest.mark.asyncio
async def test_search_predictions_survives_a_missing_model(
    httpx_mock: HTTPXMock,
) -> None:
    from olmoearth_agent.studio.client import StudioClient, StudioConfig

    httpx_mock.add_response(
        url=f"{BASE}/predictions/search",
        method="POST",
        json={"records": [{"id": "p1", "model_id": "gone"}], "meta": {"total": 1}},
    )
    httpx_mock.add_response(url=f"{BASE}/models/gone", status_code=404)
    async with StudioClient(StudioConfig(api_key="k", base_url=BASE)) as studio:
        ctx = ToolContext(studio=studio, state=ThreadState())
        result = await _tool("olmoearth_search_predictions").handler({}, ctx)
        # One missing model is not a missing /models route: no fact says so.
        assert studio.models_unavailable is False
    assert result["predictions"][0]["model_id"] == "gone"
    assert result["models"] == {}
    assert "facts" not in result


@pytest.mark.asyncio
async def test_search_predictions_by_project_pages_with_the_servers_total(
    httpx_mock: HTTPXMock,
) -> None:
    """Studio filters by project; its total of the project's predictions pages."""
    from olmoearth_agent.studio.client import StudioClient, StudioConfig

    httpx_mock.add_response(
        url=f"{BASE}/predictions/search",
        method="POST",
        match_json={
            "limit": 1,
            "offset": 0,
            "deleted_time": {"exists": False},
            "project_id": {"eq": "proj1"},
        },
        json={"records": [{"id": "p1", "status": "completed"}], "meta": {"total": 3}},
    )
    async with StudioClient(StudioConfig(api_key="k", base_url=BASE)) as studio:
        ctx = ToolContext(studio=studio, state=ThreadState())
        result = await _tool("olmoearth_search_predictions").handler(
            {"project_id": "proj1", "limit": 1}, ctx
        )
    assert (result["total"], result["next_offset"]) == (3, 1)
    assert [p["id"] for p in result["predictions"]] == ["p1"]


@pytest.mark.asyncio
async def test_search_predictions_says_when_model_records_cannot_be_read(
    httpx_mock: HTTPXMock,
) -> None:
    """A server without /models is not a model trained on no labels."""
    from olmoearth_agent.studio.client import StudioClient, StudioConfig

    httpx_mock.add_response(
        url=f"{BASE}/predictions/search",
        method="POST",
        json={"records": [{"id": "p1", "model_id": "m1"}], "meta": {"total": 1}},
    )
    httpx_mock.add_response(url=f"{BASE}/models/m1", status_code=405)
    async with StudioClient(StudioConfig(api_key="k", base_url=BASE)) as studio:
        ctx = ToolContext(studio=studio, state=ThreadState())
        result = await _tool("olmoearth_search_predictions").handler({}, ctx)
        assert studio.models_unavailable is True
    assert result["models"] == {}
    (fact,) = result["facts"]
    assert fact["id"] == "model_records_unavailable"
    assert "unknown, not absent" in fact["sentence"]


@pytest.mark.asyncio
async def test_a_refused_submit_tells_the_model_to_stop_not_to_check_ids(
    httpx_mock: HTTPXMock,
) -> None:
    """A read-only key's 403 on POST /predictions, through the registry."""
    from olmoearth_agent.llm.types import ToolCall
    from olmoearth_agent.studio.client import (
        StudioAccessError,
        StudioClient,
        StudioConfig,
    )
    from olmoearth_agent.tools.registry import ToolRegistry

    httpx_mock.add_response(
        url=f"{BASE}/predictions",
        method="POST",
        status_code=403,
        json={"errors": [{"code": "permission_error", "message": "read-only key"}]},
    )
    registry = ToolRegistry()
    registry.register_all(build_predict_tools())
    args = {
        "name": "n",
        "project_id": "pr",
        "area_id": "a",
        "model_id": "m",
        "start_time": "2022-01-01",
        "end_time": "2024-12-31",
    }
    state = ThreadState()
    async with StudioClient(StudioConfig(api_key="k", base_url=BASE)) as studio:
        out = await registry.dispatch(
            ToolCall(id="c1", name="olmoearth_submit_prediction", arguments=args),
            ToolContext(studio=studio, state=state),
        )
    assert out["ok"] is False
    assert out["error"].startswith("StudioAccessError: Studio refused POST /predict")
    assert "nothing was created or changed" in out["error"]
    assert out["hint"] == StudioAccessError.hint
    assert "Check that referenced ids exist" not in out["hint"]
    assert "nothing was created or changed" in out["must_state"][0]
    assert state.prediction_ids == []
    assert len(httpx_mock.get_requests()) == 1


@pytest.mark.asyncio
async def test_submit_prediction_tool_sends_required_fields(
    httpx_mock: HTTPXMock,
) -> None:
    from olmoearth_agent.studio.client import StudioClient, StudioConfig

    httpx_mock.add_response(
        url=f"{BASE}/predictions",
        method="POST",
        json={"records": [{"id": "pred9", "status": "pending"}]},
    )
    state = ThreadState()
    async with StudioClient(StudioConfig(api_key="k", base_url=BASE)) as studio:
        ctx = ToolContext(studio=studio, state=state)
        result = await _tool("olmoearth_submit_prediction").handler(
            {
                "name": "n",
                "project_id": "pr",
                "area_id": "a",
                "model_id": "m",
                "start_time": "2022-01-01",
                "end_time": "2024-12-31",
            },
            ctx,
        )
    assert result["id"] == "pred9"
    assert "pred9" in state.prediction_ids
    body = json.loads(httpx_mock.get_requests()[0].content)
    # PredictionWrite (spec of 31 October) has target_area_ids and no area_id,
    # which Studio would have dropped, leaving the prediction without an area.
    assert set(body) == {
        "name",
        "project_id",
        "target_area_ids",
        "model_id",
        "start_time",
        "end_time",
    }
    assert body["target_area_ids"] == ["a"]


@pytest.mark.asyncio
async def test_submit_prediction_tool_asks_for_top_k_scores(
    httpx_mock: HTTPXMock,
) -> None:
    from olmoearth_agent.studio.client import StudioClient, StudioConfig

    httpx_mock.add_response(
        url=f"{BASE}/predictions",
        method="POST",
        json={"records": [{"id": "pred9", "status": "pending"}]},
    )
    spec = _tool("olmoearth_submit_prediction").spec
    assert "probability_top_k" in spec.parameters["properties"]
    assert "probability_top_k" not in spec.parameters["required"]
    async with StudioClient(StudioConfig(api_key="k", base_url=BASE)) as studio:
        ctx = ToolContext(studio=studio, state=ThreadState())
        await _tool("olmoearth_submit_prediction").handler(
            {
                "name": "n",
                "project_id": "pr",
                "area_id": "a",
                "model_id": "m",
                "start_time": "2022-01-01",
                "end_time": "2024-12-31",
                "probability_top_k": 2,
            },
            ctx,
        )
    body = json.loads(httpx_mock.get_requests()[0].content)
    assert body["probability_output"] == {"top_k": 2}
    assert "area_id" not in body


@pytest.mark.asyncio
async def test_fetch_results_filters_by_prediction_id(
    httpx_mock: HTTPXMock,
) -> None:
    """Studio filters by prediction and fills result_metadata (live, 31 October)."""
    from olmoearth_agent.studio.client import StudioClient, StudioConfig

    httpx_mock.add_response(
        url=f"{BASE}/prediction-results/search",
        method="POST",
        match_json={
            "limit": 1000,
            "offset": 0,
            "prediction_id": {"eq": "pX"},
            "include": ["result_metadata"],
        },
        json={
            "records": [
                {
                    "id": "r1",
                    "prediction_id": "pX",
                    "tile_urls": ["t1"],
                    "property_names": ["score"],
                    "file_formats": ["png"],
                    "result_metadata": {
                        "start_datetime": "2024-06-01T00:00:00Z",
                        "end_datetime": "2024-09-01T00:00:00Z",
                        "regression_fields": [
                            {"property_name": "score", "min_value": 0, "max_value": 1}
                        ],
                    },
                }
            ],
            "meta": {"total": 1},
        },
    )
    async with StudioClient(StudioConfig(api_key="k", base_url=BASE)) as studio:
        ctx = ToolContext(studio=studio, state=ThreadState())
        # A model-chosen scan_limit over the spec's 1000 is clamped, not a 422.
        result = await _tool("olmoearth_fetch_results").handler(
            {"prediction_id": "pX", "scan_limit": 5000}, ctx
        )
    assert result["result_count"] == 1
    (one,) = result["results"]
    assert one["result_id"] == "r1"
    assert one["tile_urls"] == ["t1"]
    assert one["file_formats"] == ["png"]
    assert "file_format" not in one
    assert one["period"] == ["2024-06-01T00:00:00Z", "2024-09-01T00:00:00Z"]
    assert one["outputs"][0]["property_name"] == "score"


@pytest.mark.asyncio
async def test_get_prediction_result_tool(httpx_mock: HTTPXMock) -> None:
    from olmoearth_agent.studio.client import StudioClient, StudioConfig

    httpx_mock.add_response(
        url=f"{BASE}/prediction-results/r9",
        json={
            "records": [
                {
                    "id": "r9",
                    "prediction_id": "p1",
                    "tile_urls": ["tpl"],
                    "property_names": ["karst_score"],
                    "file_formats": ["png"],
                    "result_metadata": {"foo": 1},
                }
            ]
        },
    )
    async with StudioClient(StudioConfig(api_key="k", base_url=BASE)) as studio:
        ctx = ToolContext(studio=studio, state=ThreadState())
        result = await _tool("olmoearth_get_prediction_result").handler(
            {"result_id": "r9"}, ctx
        )
    assert result["result_id"] == "r9"
    assert result["property_names"] == ["karst_score"]
    assert result["result_metadata"] == {"foo": 1}
    assert result["file_formats"] == ["png"]


@pytest.mark.asyncio
async def test_pixel_value_regression(httpx_mock: HTTPXMock) -> None:
    from olmoearth_agent.studio.client import StudioClient, StudioConfig

    httpx_mock.add_response(
        url=re.compile(r".*/prediction-results/r1/pixel-value\?.*"),
        json={
            "records": [
                {
                    "coordinates": {"lon": 1.5, "lat": 2.5},
                    "bands": [
                        {
                            "property_name": "sample_karst_score",
                            "raw_value": 0.73,
                            "classification": None,
                        }
                    ],
                }
            ]
        },
    )
    async with StudioClient(StudioConfig(api_key="k", base_url=BASE)) as studio:
        ctx = ToolContext(studio=studio, state=ThreadState())
        out = await _tool("olmoearth_pixel_value").handler(
            {"result_id": "r1", "lon": 1.5, "lat": 2.5}, ctx
        )
    assert out["available"] is True
    assert out["value_type"] == "regression"
    assert out["value"] == 0.73
    assert out["property_name"] == "sample_karst_score"
    assert out["queried_point"] == {"lon": 1.5, "lat": 2.5}


@pytest.mark.asyncio
async def test_pixel_value_categorical_and_property_select(
    httpx_mock: HTTPXMock,
) -> None:
    from olmoearth_agent.studio.client import StudioClient, StudioConfig

    httpx_mock.add_response(
        url=re.compile(r".*/prediction-results/r2/pixel-value\?.*"),
        json={
            "records": [
                {
                    "bands": [
                        {
                            "property_name": "score",
                            "raw_value": 0.1,
                            "classification": None,
                        },
                        {
                            "property_name": "landcover",
                            "raw_value": None,
                            "classification": "forest",
                        },
                    ]
                }
            ]
        },
    )
    async with StudioClient(StudioConfig(api_key="k", base_url=BASE)) as studio:
        ctx = ToolContext(studio=studio, state=ThreadState())
        out = await _tool("olmoearth_pixel_value").handler(
            {"result_id": "r2", "lon": 0, "lat": 0, "property_name": "landcover"},
            ctx,
        )
    assert out["value_type"] == "classification"
    assert out["value"] == "forest"
    assert out["property_name"] == "landcover"
    # every band's value is surfaced
    assert {b["property_name"] for b in out["bands"]} == {"score", "landcover"}


@pytest.mark.asyncio
async def test_pixel_value_off_raster_is_unavailable(httpx_mock: HTTPXMock) -> None:
    from olmoearth_agent.studio.client import StudioClient, StudioConfig

    httpx_mock.add_response(
        url=re.compile(r".*/prediction-results/r3/pixel-value\?.*"),
        json={"records": [{"coordinates": {"lon": 9, "lat": 9}, "bands": []}]},
    )
    async with StudioClient(StudioConfig(api_key="k", base_url=BASE)) as studio:
        ctx = ToolContext(studio=studio, state=ThreadState())
        out = await _tool("olmoearth_pixel_value").handler(
            {"result_id": "r3", "lon": 9, "lat": 9}, ctx
        )
    assert out["available"] is False
    assert "no value" in out["reason"]


@pytest.mark.asyncio
async def test_pixel_value_null_band_value_is_unavailable_with_reason(
    httpx_mock: HTTPXMock,
) -> None:
    from olmoearth_agent.studio.client import StudioClient, StudioConfig

    # A band is present but carries neither a raw_value nor a classification:
    # report it like the other unavailable paths, with a reason, not a bare flag.
    httpx_mock.add_response(
        url=re.compile(r".*/prediction-results/r4/pixel-value\?.*"),
        json={
            "records": [
                {
                    "bands": [
                        {
                            "property_name": "score",
                            "raw_value": None,
                            "classification": None,
                        }
                    ]
                }
            ]
        },
    )
    async with StudioClient(StudioConfig(api_key="k", base_url=BASE)) as studio:
        ctx = ToolContext(studio=studio, state=ThreadState())
        out = await _tool("olmoearth_pixel_value").handler(
            {"result_id": "r4", "lon": 0, "lat": 0}, ctx
        )
    assert out["available"] is False
    assert out["value"] is None
    assert "no value" in out["reason"]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_live_search_predictions_have_model_ids() -> None:
    """Read-only live check: existing predictions carry reusable model_ids."""
    if not os.environ.get("OLMOEARTH_API_KEY"):
        pytest.skip("needs OLMOEARTH_API_KEY")
    from olmoearth_agent.studio import StudioClient

    async with StudioClient.from_env() as studio:
        env = await studio.search_predictions(limit=5)
    assert env.total is not None and env.total >= 1
    assert all(
        r.get("model_id") for r in env.records
    ), "every prediction should expose a model_id for reuse"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_live_fetch_results_returns_tile_urls() -> None:
    """Live: a completed prediction's results expose tile URLs."""
    if not os.environ.get("OLMOEARTH_API_KEY"):
        pytest.skip("needs OLMOEARTH_API_KEY")
    from olmoearth_agent.studio import StudioClient

    async with StudioClient.from_env() as studio:
        preds = await studio.search_predictions(limit=20)
        completed = next(
            (r for r in preds.records if r.get("status") == "completed"), None
        )
        assert completed is not None, "expected a completed prediction"
        env = await studio.search_prediction_results(
            prediction_id=completed["id"], limit=1000
        )
    # Some completed prediction in the first 1000 results should match and
    # carry tile_urls. (If not in-window, this is a soft check.)
    if env.records:
        assert any(r.get("tile_urls") for r in env.records)
