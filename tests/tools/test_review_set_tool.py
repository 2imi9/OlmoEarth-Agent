# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Tests for the olmoearth-review-set tool bundle (skill #18)."""

from __future__ import annotations

import json

import pytest

from olmoearth_agent.harness.state import ThreadState
from olmoearth_agent.llm.types import ToolCall
from olmoearth_agent.tools.registry import ToolContext, ToolRegistry
from olmoearth_agent.tools.review_set import build_review_set_tools
from olmoearth_agent.tools.uncertainty import build_uncertainty_tools

#: 4x4 grid, two classes. Column 0 is the minority class AND carries the low
#: margin, so the same fixture exercises ranking and the boundary indicator.
_SCORES = [[0.1, 0.2] if i % 4 == 0 else [5.0, 0.1] for i in range(16)]


def _ctx() -> ToolContext:
    return ToolContext(studio=None, state=ThreadState())  # type: ignore[arg-type]


def _tools() -> dict[str, object]:
    return {t.spec.name: t for t in build_review_set_tools()}


def _registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register_all(build_review_set_tools())
    return registry


def test_bundle_exposes_the_three_catalogued_tools() -> None:
    assert set(_tools()) == {
        "olmoearth_review_set",
        "olmoearth_grade_review_rule",
        "olmoearth_review_budget_ceiling",
    }


@pytest.mark.asyncio
async def test_review_set_returns_the_low_margin_windows() -> None:
    tool = _tools()["olmoearth_review_set"]
    result = await tool.handler({"scores": _SCORES, "budget": 0.25}, _ctx())  # type: ignore[attr-defined]
    assert result["n_review"] == 4
    assert {r["window_index"] for r in result["review"]} == {0, 4, 8, 12}


@pytest.mark.asyncio
async def test_review_set_accepts_a_grid_and_reports_boundaries() -> None:
    tool = _tools()["olmoearth_review_set"]
    result = await tool.handler(
        {"scores": _SCORES, "budget": 0.25, "grid": [4, 4], "order": "boundary_first"},
        _ctx(),
    )  # type: ignore[attr-defined]
    assert result["order"] == "boundary_first"
    assert result["n_boundary_windows"] > 0


@pytest.mark.asyncio
async def test_review_set_echoes_caller_ids() -> None:
    tool = _tools()["olmoearth_review_set"]
    ids = [f"w{i}" for i in range(16)]
    result = await tool.handler(
        {"scores": _SCORES, "budget": 0.125, "ids": ids}, _ctx()
    )  # type: ignore[attr-defined]
    assert result["review"][0]["id"].startswith("w")


@pytest.mark.asyncio
async def test_grade_review_rule_compares_against_baseline_and_control() -> None:
    tool = _tools()["olmoearth_grade_review_rule"]
    errors = [1, 1, 0, 0, 0, 0]
    result = await tool.handler(
        {
            "signal": [0.9, 0.8, 0.1, 0.1, 0.0, 0.0],
            "errors": errors,
            "baseline": [0.0, 0.0, 0.9, 0.9, 0.9, 0.9],
            "control": [0.5] * 6,
        },
        _ctx(),
    )  # type: ignore[attr-defined]
    assert result["gradable"] is True
    assert result["verdict"]["candidate_beats_baseline_pooled"] is True
    assert result["verdict"]["candidate_beats_control_pooled"] is True


@pytest.mark.asyncio
async def test_budget_ceiling_scores_a_quoted_capture() -> None:
    tool = _tools()["olmoearth_review_budget_ceiling"]
    result = await tool.handler(
        {"budget": 0.10, "error_rate": 0.20, "observed_capture": 0.45}, _ctx()
    )  # type: ignore[attr-defined]
    assert result["attainable_ceiling"] == pytest.approx(0.5)
    assert result["share_of_ceiling"] == pytest.approx(0.9)
    assert result["multiple_of_random"] == pytest.approx(4.5)
    assert "warning" not in result


@pytest.mark.asyncio
async def test_budget_ceiling_flags_an_impossible_capture() -> None:
    tool = _tools()["olmoearth_review_budget_ceiling"]
    result = await tool.handler(
        {"budget": 0.05, "error_rate": 0.50, "observed_capture": 0.9}, _ctx()
    )  # type: ignore[attr-defined]
    assert "warning" in result


@pytest.mark.asyncio
async def test_bad_arguments_come_back_as_an_error_envelope() -> None:
    """A malformed call is reported, never raised out of dispatch."""
    result = await _registry().dispatch(
        ToolCall(id="1", name="olmoearth_review_set", arguments={"scores": [[1.0]]}),
        _ctx(),
    )
    assert result["ok"] is False
    assert ">= 2 classes" in result["error"]


@pytest.mark.asyncio
async def test_missing_required_argument_is_rejected_before_the_handler() -> None:
    result = await _registry().dispatch(
        ToolCall(
            id="2", name="olmoearth_grade_review_rule", arguments={"signal": [1.0]}
        ),
        _ctx(),
    )
    assert result["ok"] is False
    assert "errors" in result["error"]


@pytest.mark.asyncio
async def test_rule_3_1_no_coordinates_are_emitted() -> None:
    """Rule §3.1: the chat result carries window indices and ids, never coordinates."""
    result = await _registry().dispatch(
        ToolCall(
            id="3",
            name="olmoearth_review_set",
            arguments={"scores": _SCORES, "budget": 1.0, "grid": [4, 4]},
        ),
        _ctx(),
    )
    assert result["ok"] is True
    blob = repr(result["result"]).lower()
    for banned in ("latitude", "longitude", "geometry", "wkt", '"lat"', '"lon"'):
        assert banned not in blob
    assert all(not ({"lat", "lon"} & set(row)) for row in result["result"]["review"])


@pytest.mark.asyncio
async def test_review_set_carries_the_evidence_block_into_the_tool_result() -> None:
    result = await _registry().dispatch(
        ToolCall(id="4", name="olmoearth_review_set", arguments={"scores": _SCORES}),
        _ctx(),
    )
    evidence = result["result"]["evidence"]
    assert "suite-margin-wins-every-task" in evidence
    assert "shrug-signals-rejected" in evidence
    assert any("predictive entropy" in c.lower() for c in result["result"]["caveats"])


def test_skill_9_routes_the_error_ranking_question_to_skill_18() -> None:
    """#9 owns self-consistency and OOD; #18 owns which windows are wrong."""
    for tool in build_uncertainty_tools():
        assert "olmoearth_review_set" in tool.spec.description


def test_skill_18_routes_the_ood_question_back_to_skill_9() -> None:
    """The routing is mutual, so neither skill silently answers the other's question."""
    review = {t.spec.name: t for t in build_review_set_tools()}["olmoearth_review_set"]
    assert "olmoearth_area_of_applicability" in review.spec.description
    assert "olmoearth_ensemble_uncertainty" in review.spec.description


def test_skill_9_and_18_signals_are_comparable_on_identical_windows() -> None:
    """#9's ensemble dispersion and #18's margin can be graded head to head.

    This exercises the seam between the two skills: skill #9's own
    ``prediction_confidence`` produces the candidate, skill #18's margin is
    the baseline, and skill #18's grader scores both on the same windows.

    The data is simulated with a planted margin signal, so this asserts that
    the plumbing discriminates -- it is NOT evidence for the ordering. The
    evidence is the measured result cited in ``EVIDENCE``.
    """
    import math
    import random

    from olmoearth_agent.analysis.review_set import grade_rule, margins
    from olmoearth_agent.analysis.uncertainty import prediction_confidence

    random.seed(42)
    rows = cols = 40
    n_classes = 3
    scores, members, errors = [], [], []
    for i in range(rows * cols):
        r, c = divmod(i, cols)
        truth = 0 if r + c < 26 else (1 if r + c < 52 else 2)
        edge = min(abs(r + c - 26), abs(r + c - 52))
        conf = 1.0 - math.exp(-edge / 3.0)
        wrong = random.random() > (0.35 + 0.65 * conf)
        pred = (truth + 1) % n_classes if wrong else truth
        logits = [0.0] * n_classes
        logits[pred] = 0.6 + 5.0 * conf + random.gauss(0, 0.25)
        logits[(pred + 1) % n_classes] = 0.5 + random.gauss(0, 0.25)
        scores.append(logits)
        members.append(
            [
                (
                    pred
                    if random.random() < 0.4 + 0.6 * conf
                    else random.randrange(n_classes)
                )
                for _ in range(3)
            ]
        )
        errors.append(
            1 if max(range(n_classes), key=logits.__getitem__) != truth else 0
        )

    dispersion = [
        float(pt["entropy"])
        for pt in prediction_confidence(members, value_type="categorical")["per_point"]
    ]
    graded = grade_rule(
        dispersion,
        errors,
        baseline=[-m for m in margins(scores)],
        control=[random.random() for _ in range(rows * cols)],
        budgets=[0.10],
    )
    assert graded["gradable"] is True
    arms = graded["arms"]
    # Both real signals beat the no-model control; the grader separates them.
    assert arms["candidate"]["excess_aurc"] < arms["control"]["excess_aurc"]
    assert arms["baseline"]["excess_aurc"] < arms["control"]["excess_aurc"]
    assert graded["verdict"]["candidate_beats_baseline_pooled"] is False
    # Capture never exceeds the attainable ceiling, for any arm.
    ceiling = graded["ceiling_at_budget"]["0.1"]
    for arm in arms.values():
        assert arm["capture_at_budget"]["0.1"] <= ceiling + 1e-9


@pytest.mark.asyncio
async def test_review_set_reads_scores_from_a_json_file(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """A file under OLMOEARTH_SCORES_ROOT stands in for inline rows, grid included."""
    path = tmp_path / "scores.json"
    path.write_text(json.dumps({"grid": [4, 4], "scores": _SCORES}))
    monkeypatch.setenv("OLMOEARTH_SCORES_ROOT", str(tmp_path))
    tool = _tools()["olmoearth_review_set"]
    result = await tool.handler({"scores_path": str(path), "budget": 0.25}, _ctx())  # type: ignore[attr-defined]
    assert {r["window_index"] for r in result["review"]} == {0, 4, 8, 12}
    assert all(
        (r["row"], r["col"]) == divmod(r["window_index"], 4) for r in result["review"]
    )


@pytest.mark.asyncio
async def test_scores_path_outside_the_root_is_refused(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """A model-chosen path cannot read outside the allowed root."""
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "scores.json").write_text(json.dumps(_SCORES))
    root = tmp_path / "root"
    root.mkdir()
    monkeypatch.setenv("OLMOEARTH_SCORES_ROOT", str(root))
    result = await _registry().dispatch(
        ToolCall(
            id="3",
            name="olmoearth_review_set",
            arguments={"scores_path": str(elsewhere / "scores.json")},
        ),
        _ctx(),
    )
    assert result["ok"] is False
    assert "scores_path" in result["error"]


@pytest.mark.asyncio
async def test_review_set_without_scores_or_path_is_an_error_envelope() -> None:
    """Neither argument given is reported, never raised."""
    result = await _registry().dispatch(
        ToolCall(id="4", name="olmoearth_review_set", arguments={"budget": 0.1}),
        _ctx(),
    )
    assert result["ok"] is False
    assert "scores" in result["error"]
