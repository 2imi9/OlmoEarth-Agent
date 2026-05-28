# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Unit tests for the provenance log."""

from __future__ import annotations

from olmoearth_agent.provenance.log import ProvenanceLog


def test_record_tool_call_appends_entry() -> None:
    log = ProvenanceLog(run_id="r1")
    entry = log.record_tool_call(
        "olmoearth_create_project",
        {"name": "X", "description": "d"},
        {"ok": True, "result": {"id": "p1", "name": "X"}},
    )
    assert len(log.entries) == 1
    assert entry.api_call == "olmoearth_create_project"
    assert entry.run_id == "r1"
    assert len(entry.request_hash) == 64  # sha256 hex
    assert entry.response_summary["id"] == "p1"


def test_args_hash_is_order_independent() -> None:
    log = ProvenanceLog()
    h1 = log.record_tool_call("t", {"a": 1, "b": 2}, {"ok": True}).request_hash
    h2 = log.record_tool_call("t", {"b": 2, "a": 1}, {"ok": True}).request_hash
    assert h1 == h2


def test_result_summary_drops_raw_geometry() -> None:
    log = ProvenanceLog()
    entry = log.record_tool_call(
        "t",
        {},
        {"ok": True, "result": {"id": "p1", "geometry": "POLYGON((...))", "name": "n"}},
    )
    assert "geometry" not in entry.response_summary
    assert entry.response_summary["id"] == "p1"
    assert entry.response_summary["name"] == "n"


def test_prediction_id_lifted_from_result() -> None:
    log = ProvenanceLog()
    entry = log.record_tool_call(
        "olmoearth_get_prediction",
        {"prediction_id": "x"},
        {"ok": True, "result": {"id": "pred1", "model_id": "m1", "status": "completed"}},
    )
    # get_prediction's "id" is a prediction id
    assert entry.prediction_id == "pred1"
    assert entry.model_id == "m1"


def test_to_json_and_replay_script() -> None:
    log = ProvenanceLog(run_id="r2")
    log.record_tool_call(
        "olmoearth_load_context", {}, {"ok": True, "result": {"project_count": 5}}
    )
    blob = log.to_json()
    assert '"run_id": "r2"' in blob
    assert '"entry_count": 1' in blob
    script = log.replay_script()
    assert "run_id: r2" in script
    assert "olmoearth_load_context" in script
