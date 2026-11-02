# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Tests for olmoearth_studio_label_inventory (tools/studio_labels.py).

The kelp case follows what was measured live on 2026-10-31 with a read-only
key: the model record's data selector (its training project's ``project_id``
``{"eq", "inc": null}``, every other filter null, ``sort_by`` creation_time
ascending), its label and split fields, its split proportions (test 0.0), and
the training project's 1056 approved MULTIPOLYGON annotations by oe_split and
kelp class, all created at 01:26 UTC, the model at 02:25 and updated at 03:10.
The embeddings record follows the live KarstEmbedding's shape (no selector)
and the no-split-field case the live KarstBinary's. Ids, people, coordinates
and the data types of kelp_value and task_name (the facts name those fields
only) are this test's own values.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest
from pytest_httpx import HTTPXMock

from olmoearth_agent.analysis.output_contract import MUST_STATE_MAX_WORDS
from olmoearth_agent.harness.state import ThreadState
from olmoearth_agent.llm.types import ToolCall
from olmoearth_agent.studio.client import StudioAccessError, StudioClient, StudioConfig
from olmoearth_agent.tools import studio_labels
from olmoearth_agent.tools.registry import ToolContext, ToolRegistry

BASE = "http://mock-studio/api/v1"
TOOL = "olmoearth_studio_label_inventory"
SEARCH = f"{BASE}/annotations/search"
PROJECT = "proj-kelp-test"
MODEL = "model-kelp-test"
PREDICTION = "pred-kelp-test"
LABEL_FIELD = "field-kelp-test"
SPLIT_FIELD = "field-oe-split-test"
#: The model record's checkpoint token and requester, as test values: neither
#: may reach a result.
CKPT_VALUE = "ckpt-5e1a-test-only"
REQUESTER = "person-requester-test"
#: A free-text metadata value (the task_name field): never returned.
FREE_NOTE = "free-text-note-test: surveyed from the pier at low tide"
#: What a selector may hold that must reach Studio and nothing else: a
#: person's id, a polygon's coordinates and a free-text filter value.
HIDDEN_PERSON = "person-annotator-hidden"
HIDDEN_TEXT = "free-text-filter-hidden"
HIDDEN_RING = [[-119.81, 34.41], [-119.79, 34.41], [-119.79, 34.43], [-119.81, 34.41]]

#: oe_split x kelp of the kelp training project's annotations (2026-10-31).
KELP_CELLS = (
    ("test", "absent", 378),
    ("test", "present", 179),
    ("train", "absent", 304),
    ("train", "present", 132),
    ("val", "absent", 44),
    ("val", "present", 19),
)


def _kelp_time(i: int) -> str:
    """The i-th annotation's creation time: all at 01:26 UTC, in order."""
    return f"2026-10-31T01:26:{i // 20:02d}.{i % 20:06d}Z"


def _entry(
    field_id: str,
    name: str,
    data_type: str,
    *,
    label: str | None = None,
    value: Any = None,
) -> dict[str, Any]:
    """One metadata_values entry, shaped as the spec's AnnotationMetadataRead."""
    return {
        "id": f"mv-{field_id}",
        "metadata_field_id": field_id,
        "name": name,
        "data_type": data_type,
        "display_name": None,
        "value": value,
        "label_id": f"label-id-{label}" if label else None,
        "label_name": label,
    }


def _annotation(i: int, split: str, kelp: str, created: str) -> dict[str, Any]:
    """One annotation record, shaped as the spec's AnnotationRead."""
    lon, lat = f"-119.8{i:04d}", f"34.4{i:04d}"
    present = int(kelp == "present")
    return {
        "id": f"ann-{i:05d}",
        "status": "approved",
        "project_id": PROJECT,
        "task_id": f"task-{i // 100}",
        "annotator_id": f"person-annotator-{i % 3}",
        "creation_time": created,
        "updated_time": created,
        "geom_wkt": f"MULTIPOLYGON ((({lon} {lat}, {lon} 34.5, -119.7 {lat}, "
        f"{lon} {lat})))",
        "annotation_images": [
            {
                "id": f"img-{i}",
                "user_id": "person-photographer",
                "latitude": 34.41,
                "longitude": -119.81,
                "image_url": f"/img-{i}.png",
            }
        ],
        "metadata_values": [
            _entry(LABEL_FIELD, "kelp", "labelset", label=kelp),
            _entry("field-kelp-value", "kelp_value", "number", value=present),
            _entry(SPLIT_FIELD, "oe_split", "labelset", label=split),
            _entry("field-task-name", "task_name", "text", value=FREE_NOTE),
        ],
    }


def _kelp_annotations() -> list[dict[str, Any]]:
    """The kelp training project's 1056 annotations, in creation-time order."""
    cells = [(split, kelp) for split, kelp, n in KELP_CELLS for _ in range(n)]
    return [_annotation(i, s, k, _kelp_time(i)) for i, (s, k) in enumerate(cells)]


def _selector(project: str, **filters: Any) -> dict[str, Any]:
    """A data selector shaped as the kelp model record's (live, 2026-10-31)."""
    nulls = dict.fromkeys(
        (
            "id",
            "task_id",
            "annotator_id",
            "status",
            "dataset_id",
            "image_id",
            "metadata_filters",
            "has_feedback",
            "has_images",
            "has_labels",
            "creation_time",
            "updated_time",
            "start_time",
            "end_time",
            "intersects_geometry",
            "intersects_area_id",
            "metadata_value",
            "limit",
        )
    )
    return {
        **nulls,
        "project_id": {"eq": project, "inc": None},
        "sort_by": "creation_time",
        "sort_direction": "asc",
        **filters,
    }


def _model(
    selectors: Any,
    *,
    created: str = "2026-10-31T02:25:55.862308Z",
    updated: str = "2026-10-31T03:10:57.823284Z",
    split_field: str | None = SPLIT_FIELD,
    test_prop: float = 0.0,
) -> dict[str, Any]:
    """A model record shaped as the kelp model's (live, 2026-10-31), test ids."""
    return {
        "id": MODEL,
        "name": "Santa Barbara Kelp Q2c Base Rerun 1280m Apr-Jun",
        "model_type": "fine_tuned",
        "status": "completed",
        "creation_time": created,
        "updated_time": updated,
        "checkpoint_token": CKPT_VALUE,
        "requester_id": REQUESTER,
        "project_id": PROJECT,
        "wizard_answers": {
            "wizard_id": "fine_tuned_v1",
            "project_id": PROJECT,
            "prediction_type": "per_pixel_classification",
            "label_field_id": LABEL_FIELD,
            "split_field_id": split_field,
            "split_proportions": {
                "train_prop": 0.75,
                "val_prop": 0.25 - test_prop,
                "test_prop": test_prop,
            },
            "patch_size": 4,
        },
        "data_selectors": selectors,
    }


def _search(
    records: list[dict[str, Any]], bodies: list[dict[str, Any]]
) -> Callable[[httpx.Request], httpx.Response]:
    """Answer /annotations/search as Studio would, keeping each request's body.

    A page of ``records`` (in creation-time order) at the body's offset and
    limit, from a ``creation_time`` ``gte`` floor when one is sent, with
    ``meta.total`` the count of what the request selects.
    """

    def answer(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        bodies.append(body)
        floor = (body.get("creation_time") or {}).get("gte")
        chosen = [r for r in records if floor is None or r["creation_time"] >= floor]
        start, size = body["offset"], body["limit"]
        page = chosen[start : start + size]
        payload = {"records": page, "meta": {"total": len(chosen)}, "errors": None}
        return httpx.Response(200, json=payload)

    return answer


def _mock_search(
    httpx_mock: HTTPXMock, records: list[dict[str, Any]], bodies: list[dict[str, Any]]
) -> None:
    """Answer every /annotations/search with :func:`_search`."""
    httpx_mock.add_callback(
        _search(records, bodies), url=SEARCH, method="POST", is_reusable=True
    )


def _mock_model(httpx_mock: HTTPXMock, record: dict[str, Any]) -> None:
    """Answer GET /models/{MODEL} with ``record``."""
    httpx_mock.add_response(url=f"{BASE}/models/{MODEL}", json={"records": [record]})


def _mock_kelp(httpx_mock: HTTPXMock, bodies: list[dict[str, Any]]) -> None:
    """The kelp prediction (optional), its model record and the annotation search."""
    prediction = {
        "id": PREDICTION,
        "model_id": MODEL,
        "project_id": PROJECT,
        "requester_id": REQUESTER,
        "status": "completed",
    }
    httpx_mock.add_response(
        url=f"{BASE}/predictions/{PREDICTION}",
        json={"records": [prediction]},
        is_optional=True,
    )
    _mock_model(httpx_mock, _model([_selector(PROJECT)]))
    _mock_search(httpx_mock, _kelp_annotations(), bodies)


def _studio() -> StudioClient:
    """A client of the mocked Studio."""
    return StudioClient(StudioConfig(api_key="k", base_url=BASE))


async def _call(studio: Any, **args: Any) -> dict[str, Any]:
    """Dispatch the tool through a registry, as the agent does."""
    registry = ToolRegistry()
    registry.register_all(studio_labels.build_studio_label_tools())
    call = ToolCall(id="c1", name=TOOL, arguments=args)
    ctx = ToolContext(studio=studio, state=ThreadState())
    return await registry.dispatch(call, ctx)


def _page(offset: int, **filters: Any) -> dict[str, Any]:
    """The search body a replayed kelp selector sends for one page."""
    return {
        "project_id": {"eq": PROJECT},
        **filters,
        "sort_by": "creation_time",
        "sort_direction": "asc",
        "limit": 1000,
        "offset": offset,
    }


def _selector_with_people() -> dict[str, Any]:
    """A selector holding a person's id, a polygon and a free-text filter."""
    return _selector(
        PROJECT,
        annotator_id={"eq": HIDDEN_PERSON, "neq": None},
        intersects_geometry={"type": "Polygon", "coordinates": [HIDDEN_RING]},
        metadata_filters=[
            {
                "metadata_field_id": "field-task-name",
                "label_id": None,
                "text": {"eq": HIDDEN_TEXT},
            }
        ],
    )


def _leaks(text: str) -> list[str]:
    """The selector's values (a person, a polygon, free text) found in ``text``."""
    values = (HIDDEN_PERSON, HIDDEN_TEXT, "-119.81", "-119.79", "34.41", "34.43")
    return [v for v in values if v in text]


# --------------------------------------------------------------------------- the kelp case


@pytest.mark.asyncio
async def test_the_kelp_models_training_annotations_are_counted_from_its_prediction(
    httpx_mock: HTTPXMock,
) -> None:
    """1056 approved MULTIPOLYGON annotations, by oe_split x kelp as measured."""
    bodies: list[dict[str, Any]] = []
    _mock_kelp(httpx_mock, bodies)
    async with _studio() as studio:
        envelope = await _call(studio, prediction_id=PREDICTION)
    assert envelope["ok"] is True, envelope
    out = envelope["result"]
    assert (out["source"], out["prediction_id"]) == ("prediction", PREDICTION)
    assert (out["model_id"], out["training_selector"]) == (MODEL, "replayed")
    assert out["selectors"] == [
        {"filters": ["project_id"], "not_replayed": [], "studio_total": 1056}
    ]
    assert (out["annotations_read"], out["studio_total"]) == (1056, 1056)
    assert out["complete"] is True
    assert out["read_note"] == (
        "Read 1056 of the 1056 that Studio counts (meta.total): every annotation "
        "selected was read, each counted once."
    )
    assert out["by_status"] == {"approved": 1056}
    assert out["by_geometry_type"] == {"MULTIPOLYGON": 1056}
    assert out["by_split"] == {"test": 557, "train": 436, "val": 63}
    assert out["by_class"] == {"absent": 726, "present": 330}
    assert out["split_by_class"] == {
        "test": {"absent": 378, "present": 179},
        "train": {"absent": 304, "present": 132},
        "val": {"absent": 44, "present": 19},
    }
    # All created at 01:26, before the model's record (02:25, updated 03:10).
    assert out["creation_time"] == {"first": _kelp_time(0), "last": _kelp_time(1055)}
    assert out["created_after_model_creation"] == 0
    assert out["updated_after_model_creation"] == 0
    assert out["created_after_model_update"] == 0
    assert out["times_note"] == studio_labels.TIMES_NOTE
    assert out["classes_note"] == studio_labels.CLASSES_NOTE
    model = out["model"]
    assert (model["model_type"], model["prediction_type"]) == (
        "fine_tuned",
        "per_pixel_classification",
    )
    assert model["split_proportions"] == {
        "train_prop": 0.75,
        "val_prop": 0.25,
        "test_prop": 0.0,
    }
    assert (model["patch_size"], model["training_project_id"]) == (4, PROJECT)
    assert model["label_field_named"] is model["split_field_named"] is True
    # The record names a split field and proportions with no test share.
    assert out["selector_notes"] == [studio_labels.SPLIT_NOTE]


@pytest.mark.asyncio
async def test_the_result_states_what_annotations_cannot_be_used_for(
    httpx_mock: HTTPXMock,
) -> None:
    """must_state: no design-based error rate; train/val fit the model; the
    kelp record's test share is 0, so its 557 test-tagged annotations were held
    out only if training followed the split field."""
    _mock_kelp(httpx_mock, [])
    async with _studio() as studio:
        out = (await _call(studio, prediction_id=PREDICTION))["result"]
    assert out["must_state"] == [
        studio_labels.NOT_A_SAMPLE,
        studio_labels.TRAIN_VAL_USED,
        studio_labels.TEST_SHARE_ZERO,
    ]
    assert [c["id"] for c in out["forbidden_claims"]] == ["accuracy_not_measured"]
    why = out["forbidden_claims"][0]["why"]
    # Holds where the stakeholder report emits the same id (one why per id).
    assert why.startswith("no error rate, accuracy or share wrong may be stated")
    assert "olmoearth_estimate_map_error" in why
    assert "olmoearth_plan_label_sample" in out["next_steps"][0]


@pytest.mark.parametrize(
    ("split_field", "test_prop", "expected"),
    [
        (
            SPLIT_FIELD,
            0.0,
            [
                studio_labels.NOT_A_SAMPLE,
                studio_labels.TRAIN_VAL_USED,
                studio_labels.TEST_SHARE_ZERO,
            ],
        ),
        (
            SPLIT_FIELD,
            0.1,
            [
                studio_labels.NOT_A_SAMPLE,
                studio_labels.TRAIN_VAL_USED,
                studio_labels.TEST_HELD_OUT_IF,
            ],
        ),
        (None, 0.0, [studio_labels.NOT_A_SAMPLE, studio_labels.NO_TEST_SHARE]),
        (None, 0.1, [studio_labels.NOT_A_SAMPLE, studio_labels.NO_SPLIT_FIELD]),
    ],
    ids=["split-field-no-test", "split-field-test", "karst-no-test", "no-field-test"],
)
@pytest.mark.asyncio
async def test_must_state_follows_the_split_field_and_the_test_share(
    httpx_mock: HTTPXMock,
    split_field: str | None,
    test_prop: float,
    expected: list[str],
) -> None:
    """With no split field and a test share of 0 (the live KarstBinary and
    KarstNumber), no annotation is recorded as held out; every annotation here
    predates the model, so nothing is said of later ones."""
    record = _model([_selector(PROJECT)], split_field=split_field, test_prop=test_prop)
    _mock_model(httpx_mock, record)
    _mock_search(httpx_mock, _kelp_annotations()[:3], [])
    async with _studio() as studio:
        envelope = await _call(studio, model_id=MODEL)
    assert envelope["ok"] is True, envelope
    assert envelope["result"]["must_state"] == expected


@pytest.mark.asyncio
async def test_the_replayed_selector_drops_its_null_keys_and_its_order(
    httpx_mock: HTTPXMock,
) -> None:
    """The live selector's project_id {"eq", "inc": null} goes out as {"eq"}."""
    bodies: list[dict[str, Any]] = []
    _mock_kelp(httpx_mock, bodies)
    async with _studio() as studio:
        envelope = await _call(studio, model_id=MODEL)
    assert envelope["ok"] is True, envelope
    assert bodies == [_page(0), _page(1000)]
    assert envelope["result"]["source"] == "model"


@pytest.mark.asyncio
async def test_the_split_by_class_table_keeps_its_largest_cells_apart(
    httpx_mock: HTTPXMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Past its bounds the table sums the smaller splits and classes."""
    monkeypatch.setattr(studio_labels, "_MAX_TABLE_SPLITS", 1)
    monkeypatch.setattr(studio_labels, "_MAX_TABLE_CLASSES", 1)
    _mock_kelp(httpx_mock, [])
    async with _studio() as studio:
        out = (await _call(studio, model_id=MODEL))["result"]
    other = studio_labels.OTHER
    assert out["split_by_class"] == {
        "test": {"absent": 378, other: 179},
        other: {"absent": 348, other: 151},
    }
    assert out["by_split"] == {"test": 557, "train": 436, "val": 63}


# --------------------------------------------------------------------------- privacy


@pytest.mark.asyncio
async def test_no_id_person_token_geometry_or_free_text_reaches_the_result(
    httpx_mock: HTTPXMock,
) -> None:
    """Annotations are counted, never listed (docs/Usage.md, rule 1, for places)."""
    _mock_kelp(httpx_mock, [])
    async with _studio() as studio:
        envelope = await _call(studio, prediction_id=PREDICTION)
    assert envelope["ok"] is True, envelope
    text = json.dumps(envelope)
    for key in (
        "annotator_id",
        "requester_id",
        "checkpoint_token",
        "user_id",
        "geom_wkt",
        "metadata_field_id",
        "label_id",
        "latitude",
        "longitude",
    ):
        assert key not in text, key
    for value in (
        CKPT_VALUE,
        REQUESTER,
        "person-",
        "ann-",
        "img-",
        "mv-",
        "task-",
        "label-id-",
        LABEL_FIELD,
        SPLIT_FIELD,
        FREE_NOTE,
    ):
        assert value not in text, value
    # No coordinate of the geometries or the images, and no WKT body.
    for number in ("-119.", "34.4", "34.5", "(("):
        assert number not in text, number


@pytest.mark.asyncio
async def test_selector_values_reach_studio_but_never_the_result(
    httpx_mock: HTTPXMock,
) -> None:
    """A person's id, a polygon and a free-text filter are sent as the selector
    holds them; the result names the filters' keys, never their values."""
    _mock_model(httpx_mock, _model([_selector_with_people()]))
    bodies: list[dict[str, Any]] = []
    _mock_search(httpx_mock, _kelp_annotations()[:3], bodies)
    async with _studio() as studio:
        envelope = await _call(studio, model_id=MODEL)
    assert envelope["ok"] is True, envelope
    (body,) = bodies
    assert body["annotator_id"] == {"eq": HIDDEN_PERSON}
    assert body["intersects_geometry"]["coordinates"] == [HIDDEN_RING]
    assert body["metadata_filters"] == [
        {"metadata_field_id": "field-task-name", "text": {"eq": HIDDEN_TEXT}}
    ]
    assert envelope["result"]["selectors"][0]["filters"] == [
        "annotator_id",
        "intersects_geometry",
        "metadata_filters",
        "project_id",
    ]
    assert _leaks(json.dumps(envelope)) == []


@pytest.mark.parametrize(
    ("status", "error"),
    [
        (200, "RuntimeError: Studio's annotation search failed (RuntimeError)"),
        (
            422,
            "RuntimeError: Studio's annotation search failed "
            "(HTTPStatusError, HTTP 422)",
        ),
        (
            403,
            "StudioAccessError: Studio refused POST /annotations/search "
            "(HTTP 403, permission_error).",
        ),
    ],
    ids=["errors-in-200", "422", "403"],
)
@pytest.mark.asyncio
async def test_a_failed_search_reaches_the_model_without_studios_text(
    httpx_mock: HTTPXMock, status: int, error: str
) -> None:
    """Studio's error text may quote the selector: only its type, status and
    code reach the error, and no __context__ keeps the request."""
    record = _model([_selector_with_people()])
    httpx_mock.add_response(
        url=f"{BASE}/models/{MODEL}", json={"records": [record]}, is_reusable=True
    )
    message = f"no annotator {HIDDEN_PERSON} near -119.81 34.41 for {HIDDEN_TEXT}"
    httpx_mock.add_response(
        url=SEARCH,
        method="POST",
        status_code=status,
        json={
            "records": [],
            "meta": None,
            "errors": [{"code": "permission_error", "message": message}],
        },
        is_reusable=True,
    )
    async with _studio() as studio:
        envelope = await _call(studio, model_id=MODEL)
        ctx = ToolContext(studio=studio, state=ThreadState())
        with pytest.raises(RuntimeError) as caught:
            await studio_labels._label_inventory({"model_id": MODEL}, ctx)
    assert envelope["ok"] is False
    assert envelope["error"].startswith(error), envelope["error"]
    assert _leaks(json.dumps(envelope)) == []
    if status == 403:
        # A refused search is a read: the read's hint, which names no
        # read-only key (a read-only key reads freely).
        read = StudioAccessError("POST", "/annotations/search", 403)
        assert envelope["hint"] == read.hint
        assert "read-only" not in envelope["hint"]
        assert "read-only" not in envelope["must_state"][0]
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


# --------------------------------------------------------------------------- paging


@pytest.mark.asyncio
async def test_the_read_stops_at_the_cap_and_says_how_many_of_the_total(
    httpx_mock: HTTPXMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(studio_labels, "_PAGE", 2)
    monkeypatch.setattr(studio_labels, "_CAP", 5)
    bodies: list[dict[str, Any]] = []
    _mock_search(httpx_mock, _kelp_annotations()[:9], bodies)
    async with _studio() as studio:
        envelope = await _call(studio, project_id=PROJECT)
    assert envelope["ok"] is True, envelope
    out = envelope["result"]
    assert (out["annotations_read"], out["studio_total"]) == (5, 9)
    assert out["complete"] is False
    assert out["read_note"].startswith(
        "Read 5 of the 9 that Studio counts (meta.total): the read stopped at "
        "this tool's cap of 5 annotations;"
    )
    assert sum(out["by_status"].values()) == 5
    assert [b["offset"] for b in bodies] == [0, 2, 4]


@pytest.mark.asyncio
async def test_a_read_that_ends_exactly_at_the_cap_is_complete(
    httpx_mock: HTTPXMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Four annotations, a cap of four, a full last page: Studio's total says
    nothing is left, so the read is complete and no third page is asked for."""
    monkeypatch.setattr(studio_labels, "_PAGE", 2)
    monkeypatch.setattr(studio_labels, "_CAP", 4)
    bodies: list[dict[str, Any]] = []
    _mock_search(httpx_mock, _kelp_annotations()[:4], bodies)
    async with _studio() as studio:
        envelope = await _call(studio, project_id=PROJECT)
    assert envelope["ok"] is True, envelope
    out = envelope["result"]
    assert (out["annotations_read"], out["complete"]) == (4, True)
    assert "every annotation selected was read" in out["read_note"]
    assert [b["offset"] for b in bodies] == [0, 2]


@pytest.mark.asyncio
async def test_a_selector_after_the_cap_still_has_its_total_read(
    httpx_mock: HTTPXMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The second selector is not paged, but a page of one gives its total, so
    the result still says how many of Studio's total were read."""
    monkeypatch.setattr(studio_labels, "_PAGE", 2)
    monkeypatch.setattr(studio_labels, "_CAP", 3)
    selectors = [_selector(PROJECT), _selector(PROJECT, status={"eq": "approved"})]
    _mock_model(httpx_mock, _model(selectors))
    bodies: list[dict[str, Any]] = []
    _mock_search(httpx_mock, _kelp_annotations()[:5], bodies)
    async with _studio() as studio:
        envelope = await _call(studio, model_id=MODEL)
    assert envelope["ok"] is True, envelope
    out = envelope["result"]
    assert [(b["offset"], b["limit"]) for b in bodies] == [(0, 2), (2, 2), (0, 1)]
    assert [s["studio_total"] for s in out["selectors"]] == [5, 5]
    assert (out["annotations_read"], out["studio_total"]) == (3, 10)
    assert out["complete"] is False
    assert out["read_note"].startswith(
        "Read 3 of the 10 that Studio counts (meta.total): the read stopped at "
        "this tool's cap of 3 annotations;"
    )
    # Not complete, so later annotations cannot be ruled out.
    assert out["must_state"][1] == studio_labels.TRAIN_VAL_BEFORE


@pytest.mark.asyncio
async def test_pages_that_repeat_and_skip_are_not_called_complete(
    httpx_mock: HTTPXMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Offset pages over equal creation times keep no fixed order: a page that
    repeats one annotation and skips another ends short of meta.total."""
    monkeypatch.setattr(studio_labels, "_PAGE", 2)
    kelp = _kelp_annotations()
    pages = {0: [kelp[0], kelp[1]], 2: [kelp[1], kelp[3]], 4: [kelp[4]]}

    def answer(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        payload = {"records": pages[body["offset"]], "meta": {"total": 5}}
        return httpx.Response(200, json=payload)

    httpx_mock.add_callback(answer, url=SEARCH, method="POST", is_reusable=True)
    async with _studio() as studio:
        envelope = await _call(studio, project_id=PROJECT)
    assert envelope["ok"] is True, envelope
    out = envelope["result"]
    assert (out["annotations_read"], out["studio_total"]) == (4, 5)
    assert out["complete"] is False
    assert out["read_note"] == (
        "Read 4 of the 5 that Studio counts (meta.total): Studio's pages ended "
        "before as many distinct annotations as it counts came back (offset pages "
        "over equal creation times may repeat some and skip others); the counts "
        "cover the annotations read only."
    )


@pytest.mark.asyncio
async def test_past_the_largest_offset_the_read_goes_on_from_the_last_creation_time(
    httpx_mock: HTTPXMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The annotations at the floor's time come twice and are counted once."""
    monkeypatch.setattr(studio_labels, "_PAGE", 2)
    monkeypatch.setattr(studio_labels, "_OFFSET_MAX", 2)
    bodies: list[dict[str, Any]] = []
    _mock_search(httpx_mock, _kelp_annotations()[:7], bodies)
    async with _studio() as studio:
        envelope = await _call(studio, project_id=PROJECT)
    assert envelope["ok"] is True, envelope
    out = envelope["result"]
    assert (out["annotations_read"], out["studio_total"]) == (7, 7)
    assert out["complete"] is True
    at3, at6 = {"gte": _kelp_time(3)}, {"gte": _kelp_time(6)}
    assert [(b["offset"], b.get("creation_time")) for b in bodies] == [
        (0, None),
        (2, None),
        (0, at3),
        (2, at3),
        (0, at6),
    ]


@pytest.mark.asyncio
async def test_a_floor_that_would_not_move_ends_the_read(
    httpx_mock: HTTPXMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Five annotations of one creation time, two to a page, no offset past 0:
    the floor stays at that time, so the read ends instead of looping."""
    monkeypatch.setattr(studio_labels, "_PAGE", 2)
    monkeypatch.setattr(studio_labels, "_OFFSET_MAX", 0)
    same = "2026-10-31T01:26:16.152409Z"
    records = [_annotation(i, "train", "absent", same) for i in range(5)]
    bodies: list[dict[str, Any]] = []
    _mock_search(httpx_mock, records, bodies)
    async with _studio() as studio:
        envelope = await _call(studio, project_id=PROJECT)
    assert envelope["ok"] is True, envelope
    out = envelope["result"]
    assert (out["annotations_read"], out["complete"]) == (2, False)
    assert "more annotations share one creation time" in out["read_note"]
    assert [b.get("creation_time") for b in bodies] == [None, {"gte": same}]


@pytest.mark.asyncio
async def test_the_page_bound_stops_the_read(
    httpx_mock: HTTPXMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(studio_labels, "_PAGE", 2)
    monkeypatch.setattr(studio_labels, "_MAX_PAGES", 2)
    bodies: list[dict[str, Any]] = []
    _mock_search(httpx_mock, _kelp_annotations()[:7], bodies)
    async with _studio() as studio:
        envelope = await _call(studio, project_id=PROJECT)
    assert envelope["ok"] is True, envelope
    out = envelope["result"]
    assert (out["annotations_read"], out["complete"]) == (4, False)
    assert out["read_note"].startswith(
        "Read 4 of the 7 that Studio counts (meta.total): the read stopped at "
        "this tool's 2 search pages;"
    )
    assert [b["offset"] for b in bodies] == [0, 2]


@pytest.mark.asyncio
async def test_an_annotation_two_selectors_select_is_counted_once(
    httpx_mock: HTTPXMock,
) -> None:
    """A selector's limit is named as not replayed; Studio's totals are summed."""
    selectors = [
        _selector(PROJECT),
        _selector(PROJECT, status={"eq": "approved", "neq": None}, limit=500),
    ]
    _mock_model(httpx_mock, _model(selectors))
    bodies: list[dict[str, Any]] = []
    _mock_search(httpx_mock, _kelp_annotations()[:6], bodies)
    async with _studio() as studio:
        envelope = await _call(studio, model_id=MODEL)
    assert envelope["ok"] is True, envelope
    out = envelope["result"]
    assert (out["annotations_read"], out["studio_total"]) == (6, 12)
    assert out["complete"] is True
    assert out["selectors"][1] == {
        "filters": ["project_id", "status"],
        "not_replayed": ["limit"],
        "studio_total": 6,
    }
    assert bodies == [_page(0), _page(0, status={"eq": "approved"})]
    assert "counted once here" in out["read_note"]
    assert "not_replayed" in out["selector_notes"][0]


@pytest.mark.asyncio
async def test_selectors_and_their_keys_are_bounded(
    httpx_mock: HTTPXMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Selectors past the bound are not replayed and are said to be left out; a
    selector names at most ten keys, and an offset of 0 narrows nothing."""
    monkeypatch.setattr(studio_labels, "_MAX_SELECTORS", 1)
    odd = {f"odd_key_{i:02d}": i + 1 for i in range(12)}
    selectors = [_selector(PROJECT, offset=0, **odd), _selector("proj-other-test")]
    _mock_model(httpx_mock, _model(selectors))
    bodies: list[dict[str, Any]] = []
    _mock_search(httpx_mock, _kelp_annotations()[:3], bodies)
    async with _studio() as studio:
        envelope = await _call(studio, model_id=MODEL)
    assert envelope["ok"] is True, envelope
    out = envelope["result"]
    assert bodies == [_page(0)]
    assert out["selectors_left_out"] == 1
    (selector,) = out["selectors"]
    assert selector["not_replayed"] == [
        *(f"odd_key_{i:02d}" for i in range(10)),
        "and 2 more",
    ]
    assert any(
        n.startswith("Only the first 1 of the record's 2 selectors were replayed")
        for n in out["selector_notes"]
    )


@pytest.mark.parametrize("unread", ["filterless", "past_the_bound"])
@pytest.mark.asyncio
async def test_a_selector_not_read_leaves_later_annotations_unknown(
    httpx_mock: HTTPXMock, monkeypatch: pytest.MonkeyPatch, unread: str
) -> None:
    """A selector with no filter, or past the bound, is not read: it may select
    annotations the counts miss, so the complete read of the others does not
    rule out later ones, and the hedged sentence is stated."""
    if unread == "filterless":
        selectors = [
            _selector(PROJECT),
            {"sort_by": "creation_time", "sort_direction": "asc"},
        ]
    else:
        monkeypatch.setattr(studio_labels, "_MAX_SELECTORS", 1)
        selectors = [_selector(PROJECT), _selector(PROJECT, status={"eq": "pending"})]
    _mock_model(httpx_mock, _model(selectors))
    _mock_search(httpx_mock, _kelp_annotations(), [])
    async with _studio() as studio:
        envelope = await _call(studio, model_id=MODEL)
    assert envelope["ok"] is True, envelope
    out = envelope["result"]
    assert out["complete"] is True
    assert out["created_after_model_creation"] == 0
    assert out["must_state"] == [
        studio_labels.NOT_A_SAMPLE,
        studio_labels.TRAIN_VAL_BEFORE,
        studio_labels.TEST_SHARE_ZERO,
    ]


def test_the_training_sentences_state_their_assumption() -> None:
    """That Studio trained on what the selector selects is assumed, not
    verified live: neither sentence states it as a fact."""
    assert studio_labels.TRAIN_VAL_USED.startswith(
        "If training used the record's selector,"
    )
    assert studio_labels.TRAIN_VAL_BEFORE.startswith(
        "Assuming training used the record's selector,"
    )


# --------------------------------------------------------------------------- times


@pytest.mark.asyncio
async def test_annotations_created_or_updated_after_the_model_are_counted_apart(
    httpx_mock: HTTPXMock,
) -> None:
    """Two before the model's creation, three between it and the update, one
    after, and one of the first two updated after the creation.

    With no split field the split is unrecorded; the record's test share of 0
    holds none out, and must_state says the later ones may not have trained it.
    """
    times = ["2026-10-31T01:00:00Z"] * 2 + ["2026-10-31T02:30:00Z"] * 3
    records = [
        _annotation(i, "train", "present", at)
        for i, at in enumerate([*times, "2026-10-31T03:30:00Z"])
    ]
    records[0]["updated_time"] = "2026-10-31T02:10:00Z"
    record = _model(
        [_selector(PROJECT)],
        created="2026-10-31T02:00:00Z",
        updated="2026-10-31T03:00:00Z",
        split_field=None,
    )
    _mock_model(httpx_mock, record)
    _mock_search(httpx_mock, records, [])
    async with _studio() as studio:
        envelope = await _call(studio, model_id=MODEL)
    assert envelope["ok"] is True, envelope
    out = envelope["result"]
    assert out["created_after_model_creation"] == 4
    assert out["updated_after_model_creation"] == 5
    assert out["created_after_model_update"] == 1
    assert out["creation_time"] == {
        "first": "2026-10-31T01:00:00Z",
        "last": "2026-10-31T03:30:00Z",
    }
    assert out["by_split"] == {studio_labels.UNRECORDED: 6}
    assert out["by_class"] == {"present": 6}
    assert out["must_state"] == [
        studio_labels.NOT_A_SAMPLE,
        studio_labels.NO_TEST_SHARE,
        studio_labels.LATER_UNKNOWN,
    ]
    assert "selector_notes" not in out


@pytest.mark.asyncio
async def test_a_split_model_with_later_annotations_hedges_what_trained_it(
    httpx_mock: HTTPXMock,
) -> None:
    """An annotation added after the model does not read as trained on (the
    review's case: 200 train annotations added after the model finished)."""
    records = _kelp_annotations()[:3]
    records[2]["creation_time"] = "2026-10-31T04:00:00Z"
    records[2]["updated_time"] = "2026-10-31T04:00:00Z"
    _mock_model(httpx_mock, _model([_selector(PROJECT)]))
    _mock_search(httpx_mock, records, [])
    async with _studio() as studio:
        out = (await _call(studio, model_id=MODEL))["result"]
    assert out["created_after_model_creation"] == 1
    assert out["must_state"][1] == studio_labels.TRAIN_VAL_BEFORE


# --------------------------------------------------------------------------- no model


@pytest.mark.asyncio
async def test_no_models_route_leaves_the_selector_unknown_and_counts_a_project(
    httpx_mock: HTTPXMock,
) -> None:
    """A server with no /models route: the training selector is unknown, never
    empty, and nothing is counted for it; a project's annotations are counted
    with no training statement but "unknown"."""
    httpx_mock.add_response(
        url=f"{BASE}/predictions/{PREDICTION}",
        json={"records": [{"id": PREDICTION, "model_id": MODEL}]},
    )
    httpx_mock.add_response(
        url=f"{BASE}/models/{MODEL}", status_code=404, json={"detail": "Not Found"}
    )
    bodies: list[dict[str, Any]] = []
    _mock_search(httpx_mock, _kelp_annotations(), bodies)
    async with _studio() as studio:
        first = await _call(studio, prediction_id=PREDICTION)
        assert studio.models_unavailable is True
        assert bodies == []
        second = await _call(studio, project_id=PROJECT)
    assert first["ok"] is True and second["ok"] is True, (first, second)
    unknown = first["result"]
    assert (unknown["training_selector"], unknown["model_id"]) == ("unknown", MODEL)
    assert "/models" in unknown["reason"]
    assert "annotations_read" not in unknown
    assert unknown["must_state"] == [
        studio_labels.SELECTOR_UNKNOWN,
        studio_labels.NOT_A_SAMPLE,
    ]
    assert [c["id"] for c in unknown["forbidden_claims"]] == ["accuracy_not_measured"]
    assert studio_labels.PROJECT_STEP in unknown["next_steps"]
    counted = second["result"]
    assert (counted["source"], counted["training_selector"]) == ("project", "unknown")
    assert counted["annotations_read"] == counted["studio_total"] == 1056
    assert counted["by_labelset_field"] == {
        "kelp": {"absent": 726, "present": 330},
        "oe_split": {"test": 557, "train": 436, "val": 63},
    }
    assert "by_split" not in counted and "by_class" not in counted
    assert counted["must_state"] == [
        studio_labels.NOT_A_SAMPLE,
        studio_labels.TRAINING_UNKNOWN,
    ]
    assert bodies[0] == _page(0)
    text = json.dumps(second)
    for training in (
        studio_labels.TRAIN_VAL_USED,
        studio_labels.TRAIN_VAL_BEFORE,
        studio_labels.TEST_HELD_OUT_IF,
        studio_labels.TEST_SHARE_ZERO,
    ):
        assert training not in text
    assert FREE_NOTE not in text


@pytest.mark.asyncio
async def test_a_refused_model_record_leaves_the_selector_unknown(
    httpx_mock: HTTPXMock,
) -> None:
    """A key refused on /models/{id} is a lookup failure: unknown, not an error."""
    httpx_mock.add_response(
        url=f"{BASE}/models/{MODEL}",
        status_code=403,
        json={
            "records": [],
            "meta": None,
            "errors": [{"code": "permission_error", "message": "no access"}],
        },
    )
    async with _studio() as studio:
        envelope = await _call(studio, model_id=MODEL)
    assert envelope["ok"] is True, envelope
    out = envelope["result"]
    assert out["training_selector"] == "unknown"
    assert out["reason"] == "Studio refused the model record for this key (HTTP 403)"
    assert out["must_state"] == [
        studio_labels.SELECTOR_UNKNOWN,
        studio_labels.NOT_A_SAMPLE,
    ]


@pytest.mark.asyncio
async def test_an_embeddings_model_is_not_said_to_be_trained_on_annotations(
    httpx_mock: HTTPXMock,
) -> None:
    """The live KarstEmbedding's record: model_type embeddings, no selector."""
    record = {
        "id": MODEL,
        "name": "KarstEmbedding",
        "model_type": "embeddings",
        "status": "completed",
        "creation_time": "2026-07-05T14:28:39.653852Z",
        "updated_time": "2026-07-05T14:28:39.653857Z",
        "checkpoint_token": CKPT_VALUE,
        "requester_id": None,
        "wizard_answers": {
            "wizard_id": "embeddings_v1",
            "project_id": PROJECT,
            "encoder_variant": "nano",
        },
        "data_selectors": None,
    }
    _mock_model(httpx_mock, record)
    async with _studio() as studio:
        envelope = await _call(studio, model_id=MODEL)
    assert envelope["ok"] is True, envelope
    out = envelope["result"]
    assert out["training_selector"] == "none recorded"
    assert (out["model"]["model_type"], out["model"]["prediction_type"]) == (
        "embeddings",
        None,
    )
    assert out["must_state"] == [studio_labels.NO_SELECTOR, studio_labels.NOT_A_SAMPLE]
    assert studio_labels.SELECTOR_UNKNOWN not in json.dumps(envelope)
    assert CKPT_VALUE not in json.dumps(envelope)


@pytest.mark.asyncio
async def test_studio_values_in_the_model_record_are_bounded(
    httpx_mock: HTTPXMock,
) -> None:
    """Text is cut, and a number that is not one (a bool, a string) is None."""
    record = _model(None)
    record.update({"name": "N" * 5000, "status": {"nested": "x"}})
    record["wizard_answers"].update(
        {
            "patch_size": "4",
            "split_proportions": {"train_prop": True, "val_prop": 0.25, "x": 1},
        }
    )
    _mock_model(httpx_mock, record)
    async with _studio() as studio:
        out = (await _call(studio, model_id=MODEL))["result"]
    model = out["model"]
    assert model["name"] == "N" * 120
    assert (model["status"], model["patch_size"]) == (None, None)
    assert model["split_proportions"] == {"train_prop": None, "val_prop": 0.25}


# --------------------------------------------------------------------------- inputs


@pytest.mark.parametrize(
    "args",
    [
        {},
        {"prediction_id": PREDICTION, "project_id": PROJECT},
        {"prediction_id": PREDICTION, "model_id": MODEL, "project_id": PROJECT},
        {"model_id": " ", "project_id": ""},
    ],
    ids=["none", "two", "three", "blank"],
)
@pytest.mark.asyncio
async def test_exactly_one_of_the_three_ids_is_taken(args: dict[str, Any]) -> None:
    """Refused before any Studio call (the client is None here)."""
    envelope = await _call(None, **args)
    assert envelope["ok"] is False
    assert (
        "give exactly one of prediction_id, model_id or project_id" in envelope["error"]
    )


@pytest.mark.parametrize(
    "args",
    [
        {"model_id": "m/../../users/me"},
        {"prediction_id": "p?download_token=x"},
        {"project_id": "a b"},
    ],
    ids=["path", "query", "space"],
)
@pytest.mark.asyncio
async def test_an_id_that_is_no_studio_id_is_refused(args: dict[str, Any]) -> None:
    """An id goes into a request path: refused before any Studio call."""
    envelope = await _call(None, **args)
    assert envelope["ok"] is False
    assert "is not a Studio id" in envelope["error"]


@pytest.mark.asyncio
async def test_a_predictions_model_id_that_is_no_studio_id_is_never_requested(
    httpx_mock: HTTPXMock,
) -> None:
    """The prediction record is Studio's data: its model id is checked too."""
    httpx_mock.add_response(
        url=f"{BASE}/predictions/{PREDICTION}",
        json={"records": [{"id": PREDICTION, "model_id": "m/../../users/me"}]},
    )
    async with _studio() as studio:
        envelope = await _call(studio, prediction_id=PREDICTION)
    assert envelope["ok"] is True, envelope
    out = envelope["result"]
    assert (out["training_selector"], out["model_id"]) == ("unknown", None)
    assert out["reason"] == "the prediction record's model id is not a Studio id"
    assert len(httpx_mock.get_requests()) == 1


@pytest.mark.parametrize(
    "sentence",
    [
        studio_labels.NOT_A_SAMPLE,
        studio_labels.TRAIN_VAL_USED,
        studio_labels.TRAIN_VAL_BEFORE,
        studio_labels.TEST_HELD_OUT_IF,
        studio_labels.TEST_SHARE_ZERO,
        studio_labels.NO_SPLIT_FIELD,
        studio_labels.NO_TEST_SHARE,
        studio_labels.LATER_UNKNOWN,
        studio_labels.TRAINING_UNKNOWN,
        studio_labels.SELECTOR_UNKNOWN,
        studio_labels.NO_SELECTOR,
    ],
)
def test_each_must_state_sentence_keeps_the_contract(sentence: str) -> None:
    """At most 25 words each (output_contract); a result states at most three."""
    assert len(sentence.split()) <= MUST_STATE_MAX_WORDS
