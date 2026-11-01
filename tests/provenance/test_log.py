# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Unit tests for the provenance log."""

from __future__ import annotations

import pytest

from olmoearth_agent.harness.state import ThreadState
from olmoearth_agent.provenance.log import (
    VERIFICATION_TOOLS,
    ProvenanceLog,
    VerificationCall,
    labels_digest,
)
from olmoearth_agent.provenance.tools import build_provenance_tools
from olmoearth_agent.security.egress import EgressDecision
from olmoearth_agent.tools.registry import ToolContext


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
        {
            "ok": True,
            "result": {"id": "pred1", "model_id": "m1", "status": "completed"},
        },
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


def test_record_egress_appends_audit_entry() -> None:
    log = ProvenanceLog(run_id="r3")
    record = log.record_egress(
        EgressDecision(
            allowed=True, host="olmoearth.allenai.org", capability="studio", reason="ok"
        )
    )
    assert record["host"] == "olmoearth.allenai.org"
    assert record["capability"] == "studio"
    assert record["allowed"] is True
    manifest = log.to_dict()
    assert manifest["egress_count"] == 1
    assert manifest["egress"][0]["host"] == "olmoearth.allenai.org"


# --------------------------------------------------------------------------- the verification calls

#: A plan, its estimate and its zone, as the tools return them (trimmed).
_PLAN_ARGS = {"scores": [[0.1, 0.9]] * 4, "budget": 3, "design": "random", "seed": 5}
_PLAN = {
    "available": True,
    "design": "random",
    "budget": 3,
    "seed": 5,
    "design_path": "/w/design_scores_ab12cd34_random_3_s5.json",
}
_ESTIMATE_ARGS = {"design_path": _PLAN["design_path"], "wrong": [0, 1, 0]}
_ESTIMATE = {
    "available": True,
    "design_path": _PLAN["design_path"],
    "seed": 5,
    "design": "random",
    "n_labelled": 3,
    "estimate": 1 / 3,
}
_ZONE_ARGS = {**_ESTIMATE_ARGS, "alpha": 0.1, "rule": "bonferroni"}


def _verified_log() -> ProvenanceLog:
    log = ProvenanceLog(run_id="r4")
    log.record_tool_call(
        "olmoearth_load_context", {}, {"ok": True, "result": {"project_count": 1}}
    )
    log.record_tool_call(
        "olmoearth_plan_label_sample", _PLAN_ARGS, {"ok": True, "result": _PLAN}
    )
    log.record_tool_call(
        "olmoearth_estimate_map_error",
        _ESTIMATE_ARGS,
        {"ok": True, "result": _ESTIMATE},
    )
    log.record_tool_call(
        "olmoearth_certify_zone", _ZONE_ARGS, {"ok": False, "error": "refused"}
    )
    return log


def test_the_verification_calls_are_kept_whole_with_their_seeds() -> None:
    """Each verification tool's call is kept with its arguments and result
    (the report states them); other tools' are not. The seed is the tool's:
    the plan's own, or the seed of the design file an estimate read."""
    log = _verified_log()
    assert [c.tool for c in log.verification] == list(VERIFICATION_TOOLS[:3])
    plan, estimate, zone = log.verification
    assert plan.ok and plan.result == _PLAN and plan.arguments == _PLAN_ARGS
    assert (plan.seed, estimate.seed, estimate.design) == (5, 5, "random")
    assert estimate.request_hash == log.entries[2].request_hash
    # A failed call is listed, with no result; its seed is none it returned.
    assert zone.ok is False and zone.result is None and zone.seed is None
    assert zone.repeat()["alpha"] == 0.1 and zone.repeat()["rule"] == "bonferroni"


def test_the_summary_lists_each_verification_call_with_what_repeats_it() -> None:
    """Skill #3: the design, budget and seed (and the inputs, inline arrays
    counted, not copied), so a reader can draw the same sample again."""
    calls = _verified_log().verification_calls()
    assert calls[0] == {
        "tool": "olmoearth_plan_label_sample",
        "ok": True,
        "design": "random",
        "budget": 3,
        "seed": 5,
        "design_path": _PLAN["design_path"],
        "scores": "inline, 4 rows",
    }
    assert calls[1]["seed"] == 5 and calls[1]["n_labelled"] == 3
    # Inline labels are counted and given by their digest, never copied.
    assert calls[1]["wrong"] == f"inline, 3 rows, sha256 {labels_digest([0, 1, 0])}"
    assert calls[2]["ok"] is False and "seed" not in calls[2]
    script = _verified_log().replay_script()
    assert "# Verification calls" in script
    assert "olmoearth_plan_label_sample  design=random  budget=3  seed=5" in script
    assert script.rstrip().endswith("(failed)")


def test_the_manifest_carries_the_verification_calls() -> None:
    manifest = _verified_log().to_dict()
    assert manifest["entry_count"] == 4
    tools = [c["tool"] for c in manifest["verification"]]
    assert tools == list(VERIFICATION_TOOLS[:3])
    assert '"verification"' in _verified_log().to_json()


@pytest.mark.asyncio
async def test_the_provenance_summary_tool_lists_the_seeds() -> None:
    state = ThreadState()
    state.provenance = _verified_log()
    ctx = ToolContext(studio=None, state=state)  # type: ignore[arg-type]
    out = await build_provenance_tools()[0].handler({}, ctx)
    assert out["calls"][1] == "olmoearth_plan_label_sample"
    assert [c["seed"] for c in out["verification_calls"][:2]] == [5, 5]
    assert "seed=5" in out["replay_script"]


def test_inline_labels_are_given_by_their_digest() -> None:
    """Two calls on one design file with other labels show it in the summary:
    the same inline labels give the same digest, other labels another; scores
    and window indices are counted only, and a labels file is named."""

    def call(**arguments: object) -> VerificationCall:
        return VerificationCall(
            "olmoearth_certify_zone", "t", "h", dict(arguments), True, None
        )

    first = call(design_path="/w/d.json", wrong=[0, 1, 0], alpha=0.05).repeat()
    again = call(design_path="/w/d.json", wrong=[0, 1, 0], alpha=0.1).repeat()
    other = call(design_path="/w/d.json", wrong=[0, 0, 1], alpha=0.1).repeat()
    assert first["wrong"] == again["wrong"] != other["wrong"]
    assert other["wrong"] == f"inline, 3 rows, sha256 {labels_digest([0, 0, 1])}"
    indexed = call(window_indices=[4, 9], scores=[[0.2, 0.8]] * 10).repeat()
    assert indexed["window_indices"] == "inline, 2 rows"
    assert indexed["scores"] == "inline, 10 rows"
    from_file = call(design_path="/w/d.json", labels_path="/w/l.csv").repeat()
    assert from_file["labels_path"] == "/w/l.csv" and "wrong" not in from_file


def test_the_certificate_calls_are_kept_with_their_plans_seed() -> None:
    """A certificate written and checked: each is kept whole, with the seed
    its plan took from the map (128 bits), the date, the certificate, the map
    copy, the window asked about and whether it verified."""
    assert VERIFICATION_TOOLS[3:] == (
        "olmoearth_write_certificate",
        "olmoearth_verify_certificate",
    )
    seed = 2**100 + 7
    cert = "/w/certificates/d/certificate/certificate.json"
    log = ProvenanceLog(run_id="r6")
    log.record_tool_call(
        "olmoearth_write_certificate",
        {"design_path": "/w/d.json", "wrong": [0, 0, 1], "date": "2024-06-01"},
        {
            "ok": True,
            "result": {
                "available": True,
                "design": "random",
                "seed": seed,
                "alpha": 0.2,
                "delta": 0.1,
                "rule": "prefix",
                "n_labelled": 3,
                "certificate_path": cert,
            },
        },
    )
    log.record_tool_call(
        "olmoearth_verify_certificate",
        {"certificate_path": cert, "map_path": "/w/copy.tif", "window": [2, 3]},
        {
            "ok": True,
            "result": {
                "available": True,
                "verified": False,
                "design": "random",
                "seed": seed,
                "certificate_path": cert,
            },
        },
    )
    written, checked = log.verification_calls()
    assert written == {
        "tool": "olmoearth_write_certificate",
        "ok": True,
        "design": "random",
        "seed": seed,
        "design_path": "/w/d.json",
        "alpha": 0.2,
        "delta": 0.1,
        "rule": "prefix",
        "date": "2024-06-01",
        "certificate_path": cert,
        "wrong": f"inline, 3 rows, sha256 {labels_digest([0, 0, 1])}",
        "n_labelled": 3,
    }
    assert checked == {
        "tool": "olmoearth_verify_certificate",
        "ok": True,
        "design": "random",
        "seed": seed,
        "certificate_path": cert,
        "map_path": "/w/copy.tif",
        "window": [2, 3],
        "verified": False,
    }
    assert f"seed={seed}" in log.replay_script()


def test_a_map_seeded_plan_is_listed_with_seed_from_map() -> None:
    """A plan drawn with seed_from_map is repeated with seed_from_map and its
    alpha, delta and rule, never with the seed alone (the plan tool refuses
    seed and seed_from_map together, and the seed alone draws another
    sample). The flag is read from the arguments: the plan's result holds a
    seed_from_map block of the same name."""
    seed = 2**100 + 7
    args = {
        "scores_path": "/w/scores.json",
        "budget": 60,
        "design": "random",
        "seed_from_map": True,
        "alpha": 0.2,
    }
    result = {
        "available": True,
        "design": "random",
        "budget": 60,
        "seed": seed,
        "design_path": "/w/d.json",
        "seed_from_map": {"seed": seed, "note": "n", "package_sheet": "/w/s.csv"},
    }
    log = ProvenanceLog(run_id="r7")
    log.record_tool_call(
        "olmoearth_plan_label_sample", args, {"ok": True, "result": result}
    )
    (plan,) = log.verification_calls()
    assert plan["seed_from_map"] is True and plan["seed"] == seed
    assert plan["alpha"] == 0.2
    assert list(plan)[:6] == ["tool", "ok", "design", "budget", "seed", "seed_from_map"]
    script = log.replay_script()
    assert f"seed={seed}  seed_from_map=True  scores_path=/w/scores.json" in script
    assert "drawn\n# again with seed_from_map" in script
    # A plan without it lists no flag.
    plain = VerificationCall(
        "olmoearth_plan_label_sample", "t", "h", {"seed_from_map": False}, True, None
    )
    assert "seed_from_map" not in plain.repeat()


def test_the_replay_script_keeps_every_value_on_its_comment_line() -> None:
    """A value given as an argument (a refused call's date) can hold a line
    break; written raw, it would end the '#' comment and run as code. It is
    written as a JSON string, its breaks escaped."""
    log = ProvenanceLog(run_id="r8")
    log.record_tool_call(
        "olmoearth_write_certificate",
        {"design_path": "/w/d.json", "date": "2024-06-01\nimport os\r\nos.getcwd()"},
        {"ok": False, "error": "refused"},
    )
    script = log.replay_script()
    body = script.splitlines()
    assert all(line == "" or line.startswith("#") for line in body), body
    assert '"2024-06-01\\nimport os\\r\\nos.getcwd()"' in script
    # A value with no line break is written as it is.
    assert "design_path=/w/d.json" in script
