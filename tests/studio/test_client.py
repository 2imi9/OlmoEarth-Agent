# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Unit tests for the Studio client against a mocked HTTP endpoint."""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx
import pytest
from pytest_httpx import HTTPXMock, IteratorStream

from olmoearth_agent.studio.client import (
    MODELS_UNAVAILABLE,
    StudioAccessError,
    StudioClient,
    StudioConfig,
)

BASE = "http://mock-studio/api/v1"

#: A result's download token, as a test value; it must never appear in an
#: error text or a log line.
DL_VALUE = "dl-7f3a9c-test-only"


@pytest.fixture
def config() -> StudioConfig:
    return StudioConfig(api_key="sk_test", base_url=BASE)


@pytest.mark.asyncio
async def test_users_me_unwraps_envelope(
    config: StudioConfig, httpx_mock: HTTPXMock
) -> None:
    httpx_mock.add_response(
        url=f"{BASE}/users/me",
        json={"records": [{"id": "u1", "name": "Test"}], "meta": None, "errors": None},
    )
    async with StudioClient(config) as studio:
        me = await studio.users_me()
    assert me["name"] == "Test"


@pytest.mark.asyncio
async def test_search_projects_reports_total(
    config: StudioConfig, httpx_mock: HTTPXMock
) -> None:
    httpx_mock.add_response(
        url=f"{BASE}/projects/search",
        method="POST",
        json={
            "records": [{"id": "p1", "name": "A"}],
            "meta": {"total": 1},
            "errors": None,
        },
    )
    async with StudioClient(config) as studio:
        env = await studio.search_projects(limit=5)
    assert env.total == 1
    assert env.one is not None
    assert env.one["name"] == "A"


@pytest.mark.asyncio
async def test_load_context_combines_user_and_projects(
    config: StudioConfig, httpx_mock: HTTPXMock
) -> None:
    httpx_mock.add_response(
        url=f"{BASE}/users/me",
        json={
            "records": [
                {
                    "id": "u1",
                    "name": "Ziming",
                    "organizations": [{"name": "bAI Labs"}],
                }
            ]
        },
    )
    httpx_mock.add_response(
        url=f"{BASE}/projects/search",
        method="POST",
        json={"records": [{"id": "p1", "name": "Karst"}], "meta": {"total": 1}},
    )
    async with StudioClient(config) as studio:
        ctx = await studio.load_context()
    assert ctx.user_name == "Ziming"
    assert ctx.organization == "bAI Labs"
    assert len(ctx.projects) == 1
    assert ctx.projects[0].name == "Karst"


@pytest.mark.asyncio
async def test_create_area_posts_geom_and_returns_record(
    config: StudioConfig, httpx_mock: HTTPXMock
) -> None:
    geom = {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]}
    httpx_mock.add_response(
        url=f"{BASE}/areas",
        method="POST",
        json={"records": [{"id": "a1", "name": "AOI", "project_id": "p1"}]},
    )
    async with StudioClient(config) as studio:
        rec = await studio.create_area(name="AOI", geom=geom, project_id="p1")
    assert rec["id"] == "a1"
    request = httpx_mock.get_requests()[0]
    import json as _json

    sent = _json.loads(request.read())
    assert sent == {"name": "AOI", "geom": geom, "project_id": "p1"}


@pytest.mark.asyncio
async def test_search_areas_sends_the_project_filter(
    config: StudioConfig, httpx_mock: HTTPXMock
) -> None:
    """Studio filters areas by project itself (AreaSearchRequest.project_id).

    Until the 31 October spec the client filtered pages client-side; the
    server's filter replaces that scan.
    """
    httpx_mock.add_response(
        url=f"{BASE}/areas/search",
        method="POST",
        match_json={"limit": 50, "offset": 0, "project_id": {"eq": "p1"}},
        json={
            "records": [{"id": "a1", "name": "x", "project_id": "p1"}],
            "meta": {"total": 1},
        },
    )
    async with StudioClient(config) as studio:
        env = await studio.search_areas(project_id="p1")
    assert [r["id"] for r in env.records] == ["a1"]
    assert env.total == 1


@pytest.mark.asyncio
async def test_a_server_that_refuses_the_filter_gets_the_old_scan(
    config: StudioConfig, httpx_mock: HTTPXMock
) -> None:
    """An older server answered the filter field with 422: scan client-side."""
    httpx_mock.add_response(
        url=f"{BASE}/areas/search",
        method="POST",
        match_json={"limit": 50, "offset": 0, "project_id": {"eq": "p1"}},
        status_code=422,
        json={"detail": [{"msg": "Extra inputs are not permitted"}]},
    )
    httpx_mock.add_response(
        url=f"{BASE}/areas/search",
        method="POST",
        match_json={"limit": 50, "offset": 0},
        json={
            "records": [
                {"id": "a1", "name": "x", "project_id": "p1"},
                {"id": "a2", "name": "y", "project_id": "p2"},
            ],
            "meta": {"total": 2},
        },
    )
    async with StudioClient(config) as studio:
        env = await studio.search_areas(project_id="p1")
    assert [r["id"] for r in env.records] == ["a1"]
    # The server's total (2) counted every area; the scan counts its matches.
    assert env.total == 1


@pytest.mark.asyncio
async def test_get_area(config: StudioConfig, httpx_mock: HTTPXMock) -> None:
    geom = {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]}
    httpx_mock.add_response(
        url=f"{BASE}/areas/a1",
        json={"records": [{"id": "a1", "name": "AOI", "geom": geom}]},
    )
    async with StudioClient(config) as studio:
        rec = await studio.get_area("a1")
    assert rec["id"] == "a1"
    assert rec["geom"]["type"] == "Polygon"


@pytest.mark.asyncio
async def test_delete_area(config: StudioConfig, httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url=f"{BASE}/areas/a1",
        method="DELETE",
        json={"records": [{"id": "a1", "name": "AOI"}]},
    )
    async with StudioClient(config) as studio:
        rec = await studio.delete_area("a1")
    assert rec["id"] == "a1"


@pytest.mark.asyncio
async def test_auth_header_sent(config: StudioConfig, httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(url=f"{BASE}/users/me", json={"records": [{}]})
    async with StudioClient(config) as studio:
        await studio.users_me()
    request = httpx_mock.get_requests()[0]
    assert request.headers["Authorization"] == "Bearer sk_test"


def test_from_env_rejects_placeholder(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLMOEARTH_API_KEY", "replace-me")
    with pytest.raises(RuntimeError, match="OLMOEARTH_API_KEY"):
        StudioConfig.from_env()


@pytest.mark.asyncio
async def test_get_project(config: StudioConfig, httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url=f"{BASE}/projects/p1", json={"records": [{"id": "p1", "name": "X"}]}
    )
    async with StudioClient(config) as studio:
        rec = await studio.get_project("p1")
    assert rec["id"] == "p1"
    assert rec["name"] == "X"


@pytest.mark.asyncio
async def test_delete_project_unwraps_nested_record(
    config: StudioConfig, httpx_mock: HTTPXMock
) -> None:
    # DELETE returns 202 with the deleted record nested under "record".
    httpx_mock.add_response(
        url=f"{BASE}/projects/p1",
        method="DELETE",
        status_code=202,
        json={"records": [{"record": {"id": "p1", "name": "X"}}]},
    )
    async with StudioClient(config) as studio:
        rec = await studio.delete_project("p1")
    assert rec["id"] == "p1"


# --------------------------------------------------------------------------- search filters and limits


@pytest.mark.asyncio
async def test_search_predictions_sends_the_project_and_deleted_filters(
    config: StudioConfig, httpx_mock: HTTPXMock
) -> None:
    """The live API of 31 October filters by project and leaves deleted ones out."""
    httpx_mock.add_response(
        url=f"{BASE}/predictions/search",
        method="POST",
        match_json={
            "limit": 50,
            "offset": 0,
            "deleted_time": {"exists": False},
            "project_id": {"eq": "p1"},
        },
        json={"records": [{"id": "x1", "project_id": "p1"}], "meta": {"total": 7}},
    )
    async with StudioClient(config) as studio:
        env = await studio.search_predictions(project_id="p1")
    assert [r["id"] for r in env.records] == ["x1"]
    # The server's count of the project's predictions, not a page's length.
    assert env.total == 7


@pytest.mark.asyncio
async def test_search_predictions_without_a_project_leaves_deleted_ones_out(
    config: StudioConfig, httpx_mock: HTTPXMock
) -> None:
    httpx_mock.add_response(
        url=f"{BASE}/predictions/search",
        method="POST",
        match_json={"limit": 5, "offset": 0, "deleted_time": {"exists": False}},
        json={"records": [], "meta": {"total": 0}},
    )
    async with StudioClient(config) as studio:
        env = await studio.search_predictions(limit=5)
    assert env.records == []


@pytest.mark.asyncio
async def test_search_predictions_falls_back_to_the_scan_on_422(
    config: StudioConfig, httpx_mock: HTTPXMock
) -> None:
    httpx_mock.add_response(
        url=f"{BASE}/predictions/search",
        method="POST",
        match_json={
            "limit": 2,
            "offset": 0,
            "deleted_time": {"exists": False},
            "project_id": {"eq": "p1"},
        },
        status_code=422,
    )
    httpx_mock.add_response(
        url=f"{BASE}/predictions/search",
        method="POST",
        match_json={"limit": 2, "offset": 0},
        json={
            "records": [
                {"id": "x1", "project_id": "p1"},
                {"id": "x2", "project_id": "p2"},
            ],
            "meta": {"total": 3},
        },
    )
    httpx_mock.add_response(
        url=f"{BASE}/predictions/search",
        method="POST",
        match_json={"limit": 2, "offset": 2},
        json={"records": [{"id": "x3", "project_id": "p1"}], "meta": {"total": 3}},
    )
    async with StudioClient(config) as studio:
        env = await studio.search_predictions(project_id="p1", limit=2)
    assert [r["id"] for r in env.records] == ["x1", "x3"]
    assert env.total == 2


@pytest.mark.asyncio
async def test_the_fallback_scan_stops_at_the_specs_largest_offset(
    config: StudioConfig, httpx_mock: HTTPXMock
) -> None:
    """An older server's scan never sends an offset past 10000 (no 422 mid-scan)."""
    httpx_mock.add_response(
        url=f"{BASE}/predictions/search",
        method="POST",
        match_json={
            "limit": 1,
            "offset": 9999,
            "deleted_time": {"exists": False},
            "project_id": {"eq": "p1"},
        },
        status_code=422,
    )
    for start in (9999, 10000):
        httpx_mock.add_response(
            url=f"{BASE}/predictions/search",
            method="POST",
            match_json={"limit": 1, "offset": start},
            json={"records": [{"id": f"x{start}", "project_id": "p1"}]},
        )
    async with StudioClient(config) as studio:
        env = await studio.search_predictions(project_id="p1", limit=1, offset=9999)
    assert [r["id"] for r in env.records] == ["x9999", "x10000"]
    assert len(httpx_mock.get_requests()) == 3


@pytest.mark.asyncio
async def test_search_prediction_results_filters_and_includes_metadata(
    config: StudioConfig, httpx_mock: HTTPXMock
) -> None:
    """Without ``include`` a search record's result_metadata is null (measured)."""
    meta = {"start_datetime": "2024-06-01T00:00:00Z", "regression_fields": []}
    httpx_mock.add_response(
        url=f"{BASE}/prediction-results/search",
        method="POST",
        match_json={
            "limit": 200,
            "offset": 0,
            "prediction_id": {"eq": "pX"},
            "include": ["result_metadata"],
        },
        json={
            "records": [{"id": "r1", "prediction_id": "pX", "result_metadata": meta}]
        },
    )
    async with StudioClient(config) as studio:
        env = await studio.search_prediction_results(prediction_id="pX")
    assert env.records[0]["result_metadata"] == meta


@pytest.mark.parametrize(
    ("asked", "sent"), [(5000, 1000), (0, 1), (-4, 1), (1000, 1000)]
)
@pytest.mark.asyncio
async def test_a_project_search_limit_stays_within_the_specs_bounds(
    asked: int, sent: int, config: StudioConfig, httpx_mock: HTTPXMock
) -> None:
    """ProjectSearchRequest takes 1 to 1000; a model's limit never becomes a 422."""
    httpx_mock.add_response(
        url=f"{BASE}/projects/search",
        method="POST",
        match_json={"limit": sent, "offset": 0},
        json={"records": []},
    )
    async with StudioClient(config) as studio:
        await studio.search_projects(limit=asked, offset=-3)


@pytest.mark.asyncio
async def test_each_search_keeps_its_own_spec_bound(
    config: StudioConfig, httpx_mock: HTTPXMock
) -> None:
    """1001 on prediction-results was a 422 live; predictions take up to 10000."""
    httpx_mock.add_response(
        url=f"{BASE}/prediction-results/search",
        method="POST",
        match_json={
            "limit": 1000,
            "offset": 0,
            "prediction_id": {"eq": "pX"},
            "include": ["result_metadata"],
        },
        json={"records": []},
    )
    httpx_mock.add_response(
        url=f"{BASE}/predictions/search",
        method="POST",
        match_json={"limit": 10000, "offset": 0, "deleted_time": {"exists": False}},
        json={"records": []},
    )
    httpx_mock.add_response(
        url=f"{BASE}/labelsets/search",
        method="POST",
        match_json={"limit": 1000, "offset": 0},
        json={"records": []},
    )
    async with StudioClient(config) as studio:
        await studio.search_prediction_results(prediction_id="pX", limit=1001)
        await studio.search_predictions(limit=20000)
        await studio.search_labelsets(limit=5000)


@pytest.mark.asyncio
async def test_an_offset_past_the_specs_bound_is_refused_before_any_request(
    config: StudioConfig, httpx_mock: HTTPXMock
) -> None:
    """The spec caps these offsets at 10000: no 422, no fallback scan, no request."""
    httpx_mock.add_response(
        url=f"{BASE}/areas/search",
        method="POST",
        match_json={"limit": 50, "offset": 10000, "project_id": {"eq": "p1"}},
        json={"records": []},
    )
    httpx_mock.add_response(
        url=f"{BASE}/projects/search",
        method="POST",
        match_json={"limit": 50, "offset": 12000},
        json={"records": []},
    )
    async with StudioClient(config) as studio:
        with pytest.raises(ValueError, match="offset 12000 is past the 10000"):
            await studio.search_predictions(project_id="p1", offset=12000)
        with pytest.raises(ValueError, match="/annotations/search"):
            await studio.search_annotations({"project_id": {"eq": "p1"}}, offset=10001)
        # The bound itself is accepted, and a search the spec leaves unbounded
        # takes any offset.
        await studio.search_areas(project_id="p1", offset=10000)
        await studio.search_projects(offset=12000)
    assert len(httpx_mock.get_requests()) == 2


# --------------------------------------------------------------------------- a refused key


@pytest.mark.asyncio
async def test_a_refused_write_raises_studio_access_error_once(
    config: StudioConfig, httpx_mock: HTTPXMock
) -> None:
    """A 403: Studio's code and message, no retry, and nothing was created."""
    httpx_mock.add_response(
        url=f"{BASE}/projects",
        method="POST",
        status_code=403,
        json={
            "records": None,
            "meta": None,
            "errors": [{"code": "permission_error", "message": "API key is read-only"}],
        },
    )
    async with StudioClient(config) as studio:
        with pytest.raises(StudioAccessError) as caught:
            await studio.create_project(name="n", description="d")
    err = caught.value
    text = str(err)
    assert text.startswith("Studio refused POST /projects (HTTP 403, permission_error")
    assert ": API key is read-only)." in text
    assert "may be read-only or lack the permission" in text
    assert text.endswith("nothing was created or changed.")
    assert (err.status_code, err.code) == (403, "permission_error")
    (sentence,) = err.contract["must_state"]
    assert "nothing was created or changed" in sentence
    assert len(sentence.split()) <= 25
    assert len(httpx_mock.get_requests()) == 1


@pytest.mark.asyncio
async def test_a_refused_read_is_not_retried_and_names_no_query(
    config: StudioConfig, httpx_mock: HTTPXMock
) -> None:
    """A read retries transient errors, never a 401; the path loses its query."""
    httpx_mock.add_response(
        url=f"{BASE}/prediction-results/r1/pixel-value?lon=1.5&lat=2.5",
        status_code=401,
        text="unauthorized",
    )
    async with StudioClient(config) as studio:
        with pytest.raises(StudioAccessError) as caught:
            await studio.pixel_value("r1", 1.5, 2.5)
    text = str(caught.value)
    assert "Studio refused GET /prediction-results/r1/pixel-value (HTTP 401)." in text
    assert "lon=" not in text
    assert len(httpx_mock.get_requests()) == 1


@pytest.mark.asyncio
async def test_a_refused_delete_raises_studio_access_error(
    config: StudioConfig, httpx_mock: HTTPXMock
) -> None:
    httpx_mock.add_response(url=f"{BASE}/areas/a1", method="DELETE", status_code=403)
    async with StudioClient(config) as studio:
        with pytest.raises(StudioAccessError, match="DELETE /areas/a1"):
            await studio.delete_area("a1")


# --------------------------------------------------------------------------- submit


@pytest.mark.asyncio
async def test_submit_prediction_sends_target_area_ids_and_top_k(
    config: StudioConfig, httpx_mock: HTTPXMock
) -> None:
    """PredictionWrite has target_area_ids and probability_output, no area_id."""
    httpx_mock.add_response(
        url=f"{BASE}/predictions",
        method="POST",
        json={"records": [{"id": "pred1", "status": "pending"}]},
    )
    async with StudioClient(config) as studio:
        rec = await studio.submit_prediction(
            name="n",
            project_id="pr",
            area_id="a1",
            model_id="m",
            start_time="2024-01-01T00:00:00Z",
            end_time="2024-12-31T00:00:00Z",
            probability_top_k=2,
        )
    assert rec["id"] == "pred1"
    assert json.loads(httpx_mock.get_requests()[0].content) == {
        "name": "n",
        "project_id": "pr",
        "target_area_ids": ["a1"],
        "model_id": "m",
        "start_time": "2024-01-01T00:00:00Z",
        "end_time": "2024-12-31T00:00:00Z",
        "probability_output": {"top_k": 2},
    }


@pytest.mark.parametrize("top_k", [0, -1, True])
@pytest.mark.asyncio
async def test_submit_prediction_refuses_a_top_k_below_one(
    top_k: int, config: StudioConfig, httpx_mock: HTTPXMock
) -> None:
    async with StudioClient(config) as studio:
        with pytest.raises(ValueError, match="probability_top_k"):
            await studio.submit_prediction(
                name="n",
                project_id="pr",
                area_id="a1",
                model_id="m",
                start_time="2024-01-01",
                end_time="2024-12-31",
                probability_top_k=top_k,
            )
    assert httpx_mock.get_requests() == []


# --------------------------------------------------------------------------- result files


@pytest.mark.asyncio
async def test_prediction_result_files_returns_the_zip_and_masks_the_token_in_logs(
    config: StudioConfig, httpx_mock: HTTPXMock, caplog: pytest.LogCaptureFixture
) -> None:
    httpx_mock.add_response(
        url=f"{BASE}/prediction-results/files",
        match_params={"download_token": DL_VALUE},
        content=b"PK\x03\x04zip",
    )
    caplog.set_level(logging.INFO, logger="httpx")
    async with StudioClient(config) as studio:
        data = await studio.prediction_result_files(DL_VALUE)
    assert data == b"PK\x03\x04zip"
    # httpx logs every request's URL at INFO; the token is masked there.
    assert DL_VALUE not in caplog.text
    logged = [r.getMessage() for r in caplog.records if r.name == "httpx"]
    assert any("download_token=<redacted>" in m for m in logged), logged


@pytest.mark.asyncio
async def test_a_refused_download_names_no_token(
    config: StudioConfig, httpx_mock: HTTPXMock
) -> None:
    httpx_mock.add_response(
        url=f"{BASE}/prediction-results/files",
        match_params={"download_token": DL_VALUE},
        status_code=403,
        # The token where the message's 300-character cut would split it.
        json={
            "errors": [{"code": "permission_error", "message": "x" * 290 + DL_VALUE}]
        },
    )
    async with StudioClient(config) as studio:
        with pytest.raises(StudioAccessError) as caught:
            await studio.prediction_result_files(DL_VALUE)
    text = str(caught.value)
    assert DL_VALUE[:6] not in text
    assert "x<redacted>" in text
    assert "GET /prediction-results/files (HTTP 403, permission_error" in text
    assert len(httpx_mock.get_requests()) == 1


@pytest.mark.asyncio
async def test_a_failed_download_names_no_token(
    config: StudioConfig, httpx_mock: HTTPXMock
) -> None:
    httpx_mock.add_response(
        url=f"{BASE}/prediction-results/files",
        match_params={"download_token": DL_VALUE},
        status_code=404,
        json={"detail": "Prediction Result not found"},
    )
    httpx_mock.add_exception(
        httpx.ReadTimeout(f"timed out reading ...?download_token={DL_VALUE}"),
        url=f"{BASE}/prediction-results/files",
        match_params={"download_token": DL_VALUE},
    )
    async with StudioClient(config) as studio:
        with pytest.raises(RuntimeError) as missing:
            await studio.prediction_result_files(DL_VALUE)
        with pytest.raises(RuntimeError) as timed_out:
            await studio.prediction_result_files(DL_VALUE)
    assert type(missing.value) is RuntimeError
    assert "HTTP 404" in str(missing.value)
    assert "ReadTimeout" in str(timed_out.value)
    for err in (missing.value, timed_out.value):
        assert DL_VALUE not in str(err)
        # httpx's own exception, whose text and request hold the URL, is not
        # attached either way: an error reporter walking __context__ finds none.
        assert err.__cause__ is None
        assert err.__context__ is None


@pytest.mark.asyncio
async def test_a_download_over_the_cap_is_refused(
    config: StudioConfig, httpx_mock: HTTPXMock
) -> None:
    # A streamed body with no Content-Length (as a chunked answer has): only
    # the count while read can stop it.
    httpx_mock.add_response(
        url=f"{BASE}/prediction-results/files",
        match_params={"download_token": DL_VALUE},
        stream=IteratorStream([b"x" * 6, b"x" * 6]),
    )
    httpx_mock.add_response(
        url=f"{BASE}/prediction-results/files",
        match_params={"download_token": DL_VALUE},
        headers={"Content-Length": "5000000000"},
        content=b"PK",
    )
    async with StudioClient(config) as studio:
        with pytest.raises(RuntimeError, match="cap of 10 bytes while read"):
            await studio.prediction_result_files(DL_VALUE, max_bytes=10)
        # Content-Length over the default 2 GiB cap: refused before the body.
        with pytest.raises(RuntimeError, match="5000000000 bytes, over"):
            await studio.prediction_result_files(DL_VALUE)


@pytest.mark.asyncio
async def test_prediction_result_files_needs_a_token(
    config: StudioConfig, httpx_mock: HTTPXMock
) -> None:
    async with StudioClient(config) as studio:
        with pytest.raises(ValueError, match="download_token"):
            await studio.prediction_result_files("")
    assert httpx_mock.get_requests() == []


# --------------------------------------------------------------------------- annotations, labels, datasets


@pytest.mark.asyncio
async def test_annotation_records_lose_the_people_behind_them(
    config: StudioConfig, httpx_mock: HTTPXMock
) -> None:
    """annotator_id and an image's user_id are personal data: never returned."""
    filters = {"project_id": {"eq": "p1"}, "status": {"eq": "approved"}}
    httpx_mock.add_response(
        url=f"{BASE}/annotations/search",
        method="POST",
        match_json={**filters, "limit": 10000, "offset": 0},
        json={
            "records": [
                {
                    "id": "an1",
                    "status": "approved",
                    "annotator_id": "person-1",
                    "geom_wkt": "POINT (0 0)",
                    "annotation_images": [
                        {"id": "i1", "user_id": "person-2", "image_url": "/i1.png"}
                    ],
                }
            ]
        },
    )
    async with StudioClient(config) as studio:
        env = await studio.search_annotations(filters, limit=50000)
    (record,) = env.records
    assert "annotator_id" not in record
    assert record["annotation_images"] == [{"id": "i1", "image_url": "/i1.png"}]
    assert record["geom_wkt"] == "POINT (0 0)"
    assert "person-" not in json.dumps(env.records)


@pytest.mark.asyncio
async def test_annotations_aggregate_but_never_by_annotator(
    config: StudioConfig, httpx_mock: HTTPXMock
) -> None:
    body = {
        "metric": {"type": "count"},
        "group_by": [{"field": "status"}],
        "project_id": {"eq": "p1"},
    }
    httpx_mock.add_response(
        url=f"{BASE}/annotations/aggregate",
        method="POST",
        match_json=body,
        json={"records": [{"key": {"status": "approved"}, "value": 1056}]},
    )
    async with StudioClient(config) as studio:
        env = await studio.aggregate_annotations(body)
        with pytest.raises(ValueError, match="personal data"):
            await studio.aggregate_annotations(
                {"metric": {"type": "count"}, "group_by": [{"field": "annotator_id"}]}
            )
    assert env.records == [{"key": {"status": "approved"}, "value": 1056}]
    assert len(httpx_mock.get_requests()) == 1


@pytest.mark.asyncio
async def test_labels_and_datasets_pass_their_filters_through(
    config: StudioConfig, httpx_mock: HTTPXMock
) -> None:
    httpx_mock.add_response(
        url=f"{BASE}/labels/search",
        method="POST",
        match_json={"labelset_id": {"eq": "ls1"}, "limit": 100, "offset": 0},
        json={"records": [{"id": "l1", "name": "present", "color": "#00ff00"}]},
    )
    httpx_mock.add_response(
        url=f"{BASE}/datasets/search",
        method="POST",
        match_json={"project_id": {"eq": "p1"}, "limit": 100, "offset": 0},
        json={"records": [{"id": "d1", "name": "kelp", "requester_id": "person-3"}]},
    )
    httpx_mock.add_response(
        url=f"{BASE}/datasets/d1",
        json={"records": [{"id": "d1", "name": "kelp", "requester_id": "person-3"}]},
    )
    async with StudioClient(config) as studio:
        labels = await studio.search_labels({"labelset_id": {"eq": "ls1"}})
        datasets = await studio.search_datasets({"project_id": {"eq": "p1"}})
        dataset = await studio.get_dataset("d1")
    assert labels.records[0]["name"] == "present"
    assert datasets.records == [{"id": "d1", "name": "kelp"}]
    assert dataset == {"id": "d1", "name": "kelp"}


# --------------------------------------------------------------------------- /models


@pytest.mark.asyncio
async def test_a_missing_models_route_is_not_asked_again(
    config: StudioConfig, httpx_mock: HTTPXMock
) -> None:
    """/models is not in the spec: after a 404 for the route, no more calls."""
    httpx_mock.add_response(
        url=f"{BASE}/models/search",
        method="POST",
        status_code=404,
        json={"detail": "Not Found"},
    )
    async with StudioClient(config) as studio:
        with pytest.raises(httpx.HTTPStatusError):
            await studio.search_models()
        assert studio.models_unavailable is True
        with pytest.raises(RuntimeError) as caught:
            await studio.get_model("m1")
        with pytest.raises(RuntimeError):
            await studio.search_models()
    assert str(caught.value) == MODELS_UNAVAILABLE
    assert len(httpx_mock.get_requests()) == 1


@pytest.mark.parametrize(
    ("status", "body"), [(405, None), (404, {"detail": "Not Found"})]
)
@pytest.mark.asyncio
async def test_a_405_or_an_unrouted_404_from_one_model_marks_the_route_missing(
    status: int,
    body: dict[str, Any] | None,
    config: StudioConfig,
    httpx_mock: HTTPXMock,
) -> None:
    """A 405, or FastAPI's 404 for a path no route matches, is a missing route."""
    httpx_mock.add_response(url=f"{BASE}/models/m1", status_code=status, json=body)
    async with StudioClient(config) as studio:
        with pytest.raises(httpx.HTTPStatusError):
            await studio.get_model("m1")
        with pytest.raises(RuntimeError, match="no model record is read"):
            await studio.get_model("m2")
        assert studio.models_unavailable is True
    assert len(httpx_mock.get_requests()) == 1


@pytest.mark.parametrize(
    "body",
    [
        {
            "records": None,
            "meta": None,
            "errors": [{"code": "not_found_error", "message": "Model not found"}],
        },
        {"detail": "Model not found"},
        None,
    ],
    ids=["studio-envelope", "detail-naming-the-model", "no-body"],
)
@pytest.mark.asyncio
async def test_a_missing_model_record_leaves_other_lookups_running(
    body: dict[str, Any] | None, config: StudioConfig, httpx_mock: HTTPXMock
) -> None:
    """A 404 that may name one missing model never stops the lookups of others.

    How Studio answers a missing model is unverified: its not_found_error
    envelope, a detail naming the model and an empty body all leave the route
    in use, so one deleted model never hides the records of the rest.
    """
    httpx_mock.add_response(url=f"{BASE}/models/gone", status_code=404, json=body)
    httpx_mock.add_response(
        url=f"{BASE}/models/m1", json={"records": [{"id": "m1", "name": "Kelp"}]}
    )
    async with StudioClient(config) as studio:
        with pytest.raises(httpx.HTTPStatusError):
            await studio.get_model("gone")
        assert studio.models_unavailable is False
        record = await studio.get_model("m1")
    assert record["name"] == "Kelp"
