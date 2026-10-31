# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Tests for olmoearth_studio_label_inventory (tools/studio_labels.py).

The kelp case follows what was measured live on 2026-10-31 with a read-only
key: the model record's data selector (its training project's ``project_id``
``{"eq", "inc": null}``, every other filter null, ``sort_by`` creation_time
ascending), its label and split fields, and the training project's 1056
approved MULTIPOLYGON annotations by oe_split and kelp class, all created at
01:26 UTC, the model at 02:25 and updated at 03:10. Ids, people, coordinates
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
from olmoearth_agent.studio.client import StudioClient, StudioConfig
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
                "val_prop": 0.25,
                "test_prop": 0.0,
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
    httpx_mock.add_response(
        url=f"{BASE}/models/{MODEL}", json={"records": [_model([_selector(PROJECT)])]}
    )
    httpx_mock.add_callback(
        _search(_kelp_annotations(), bodies),
        url=SEARCH,
        method="POST",
        is_reusable=True,
    )


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
    assert out["created_after_model_update"] == 0
    assert out["times_note"] == studio_labels.TIMES_NOTE
    model = out["model"]
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
    """must_state: no design-based error rate; train/val fit the model; test only if."""
    _mock_kelp(httpx_mock, [])
    async with _studio() as studio:
        out = (await _call(studio, prediction_id=PREDICTION))["result"]
    assert out["must_state"] == [
        studio_labels.NOT_A_SAMPLE,
        studio_labels.TRAIN_VAL_USED,
        studio_labels.TEST_HELD_OUT_IF,
    ]
    assert [c["id"] for c in out["forbidden_claims"]] == ["accuracy_not_measured"]
    assert "olmoearth_estimate_map_error" in out["forbidden_claims"][0]["why"]
    assert "olmoearth_plan_label_sample" in out["next_steps"][0]


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
async def test_no_id_person_token_geometry_or_free_text_reaches_the_result(
    httpx_mock: HTTPXMock,
) -> None:
    """Annotations are counted, never listed (PLAN.md section 3.1 for places)."""
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


# --------------------------------------------------------------------------- paging


@pytest.mark.asyncio
async def test_the_read_stops_at_the_cap_and_says_how_many_of_the_total(
    httpx_mock: HTTPXMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(studio_labels, "_PAGE", 2)
    monkeypatch.setattr(studio_labels, "_CAP", 5)
    bodies: list[dict[str, Any]] = []
    httpx_mock.add_callback(
        _search(_kelp_annotations()[:9], bodies),
        url=SEARCH,
        method="POST",
        is_reusable=True,
    )
    async with _studio() as studio:
        envelope = await _call(studio, project_id=PROJECT)
    assert envelope["ok"] is True, envelope
    out = envelope["result"]
    assert (out["annotations_read"], out["studio_total"]) == (5, 9)
    assert out["complete"] is False
    assert out["read_note"].startswith(
        "Read 5 of the 9 that Studio counts (meta.total), stopping at this "
        "tool's cap of 5;"
    )
    assert sum(out["by_status"].values()) == 5
    assert [b["offset"] for b in bodies] == [0, 2, 4]


@pytest.mark.asyncio
async def test_past_the_largest_offset_the_read_goes_on_from_the_last_creation_time(
    httpx_mock: HTTPXMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The annotations at the floor's time come twice and are counted once."""
    monkeypatch.setattr(studio_labels, "_PAGE", 2)
    monkeypatch.setattr(studio_labels, "_OFFSET_MAX", 2)
    bodies: list[dict[str, Any]] = []
    httpx_mock.add_callback(
        _search(_kelp_annotations()[:7], bodies),
        url=SEARCH,
        method="POST",
        is_reusable=True,
    )
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
async def test_an_annotation_two_selectors_select_is_counted_once(
    httpx_mock: HTTPXMock,
) -> None:
    """A selector's limit is named as not replayed; Studio's totals are summed."""
    selectors = [
        _selector(PROJECT),
        _selector(PROJECT, status={"eq": "approved", "neq": None}, limit=500),
    ]
    httpx_mock.add_response(
        url=f"{BASE}/models/{MODEL}", json={"records": [_model(selectors)]}
    )
    bodies: list[dict[str, Any]] = []
    httpx_mock.add_callback(
        _search(_kelp_annotations()[:6], bodies),
        url=SEARCH,
        method="POST",
        is_reusable=True,
    )
    async with _studio() as studio:
        envelope = await _call(studio, model_id=MODEL)
    assert envelope["ok"] is True, envelope
    out = envelope["result"]
    assert (out["annotations_read"], out["studio_total"]) == (6, 12)
    assert out["selectors"][1] == {
        "filters": ["project_id", "status"],
        "not_replayed": ["limit"],
        "studio_total": 6,
    }
    assert bodies == [_page(0), _page(0, status={"eq": "approved"})]
    assert "counted once here" in out["read_note"]
    assert "not_replayed" in out["selector_notes"][0]


# --------------------------------------------------------------------------- times


@pytest.mark.asyncio
async def test_annotations_created_after_the_model_are_counted_apart(
    httpx_mock: HTTPXMock,
) -> None:
    """Two before the model's creation, three between it and the update, one after.

    With no split field named, the split is unrecorded and must_state says
    which annotations fit or tuned the model is unknown.
    """
    times = ["2026-10-31T01:00:00Z"] * 2 + ["2026-10-31T02:30:00Z"] * 3
    records = [
        _annotation(i, "train", "present", at)
        for i, at in enumerate([*times, "2026-10-31T03:30:00Z"])
    ]
    record = _model(
        [_selector(PROJECT)],
        created="2026-10-31T02:00:00Z",
        updated="2026-10-31T03:00:00Z",
        split_field=None,
    )
    httpx_mock.add_response(url=f"{BASE}/models/{MODEL}", json={"records": [record]})
    httpx_mock.add_callback(
        _search(records, []), url=SEARCH, method="POST", is_reusable=True
    )
    async with _studio() as studio:
        envelope = await _call(studio, model_id=MODEL)
    assert envelope["ok"] is True, envelope
    out = envelope["result"]
    assert out["created_after_model_creation"] == 4
    assert out["created_after_model_update"] == 1
    assert out["creation_time"] == {
        "first": "2026-10-31T01:00:00Z",
        "last": "2026-10-31T03:30:00Z",
    }
    assert out["by_split"] == {studio_labels.UNRECORDED: 6}
    assert out["by_class"] == {"present": 6}
    assert out["must_state"] == [
        studio_labels.NOT_A_SAMPLE,
        studio_labels.NO_SPLIT_FIELD,
    ]
    assert "selector_notes" not in out


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
    httpx_mock.add_callback(
        _search(_kelp_annotations(), bodies),
        url=SEARCH,
        method="POST",
        is_reusable=True,
    )
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
    for training in (studio_labels.TRAIN_VAL_USED, studio_labels.TEST_HELD_OUT_IF):
        assert training not in text
    assert FREE_NOTE not in text


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
        "give exactly one of prediction_id, model_id or project_id"
        in envelope["error"]
    )


@pytest.mark.parametrize(
    "sentence",
    [
        studio_labels.NOT_A_SAMPLE,
        studio_labels.TRAIN_VAL_USED,
        studio_labels.TEST_HELD_OUT_IF,
        studio_labels.NO_SPLIT_FIELD,
        studio_labels.TRAINING_UNKNOWN,
        studio_labels.SELECTOR_UNKNOWN,
    ],
)
def test_each_must_state_sentence_keeps_the_contract(sentence: str) -> None:
    """At most 25 words each (output_contract); a result states at most three."""
    assert len(sentence.split()) <= MUST_STATE_MAX_WORDS
