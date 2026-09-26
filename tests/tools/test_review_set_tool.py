# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Tests for the olmoearth-review-set tool bundle (skill #18)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from olmoearth_agent.analysis.output_contract import (
    MUST_STATE_MAX,
    MUST_STATE_MAX_WORDS,
    word_count,
)
from olmoearth_agent.analysis.review_set import (
    MUST_STATE_MULTICLASS_LOGIT,
    MUST_STATE_NO_WINNER,
)
from olmoearth_agent.harness.state import ThreadState
from olmoearth_agent.llm.types import ToolCall
from olmoearth_agent.tools.compare import build_compare_tools
from olmoearth_agent.tools.registry import ToolContext, ToolRegistry
from olmoearth_agent.tools.review_set import build_review_set_tools
from olmoearth_agent.tools.uncertainty import build_uncertainty_tools

#: 4x4 grid, two classes. Column 0 is the minority class AND carries the low
#: margin, so the same fixture exercises ranking and the boundary indicator.
_SCORES = [[0.1, 0.2] if i % 4 == 0 else [5.0, 0.1] for i in range(16)]


@pytest.fixture(autouse=True)
def _workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The tools write files (the evidence text, a comparison's listing) to the
    workspace; keep them in a temporary one."""
    monkeypatch.setenv("OLMOEARTH_OUTPUT_ROOT", str(tmp_path / "workspace"))
    return tmp_path / "workspace"


def _ctx() -> ToolContext:
    return ToolContext(studio=None, state=ThreadState())  # type: ignore[arg-type]


def _tools() -> dict[str, object]:
    return {t.spec.name: t for t in build_review_set_tools()}


def _registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register_all(build_review_set_tools())
    return registry


def test_bundle_exposes_the_five_catalogued_tools() -> None:
    assert set(_tools()) == {
        "olmoearth_review_set",
        "olmoearth_review_set_from_result",
        "olmoearth_compare_review",
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
async def test_review_set_scopes_its_evidence_and_saves_the_full_text() -> None:
    """One sentence inline; the blocks it stands for are in a file, by path."""
    result = await _registry().dispatch(
        ToolCall(id="4", name="olmoearth_review_set", arguments={"scores": _SCORES}),
        _ctx(),
    )
    out = result["result"]
    assert "evidence" not in out and "caveats" not in out
    assert "24-task embedding suite" in out["evidence_scope"]
    detail = json.loads(Path(out["evidence_detail_path"]).read_text())
    assert "suite-margin-wins-every-task" in detail["ranking"]["claims"]
    assert "shrug-signals-rejected" in detail["ranking"]["claims"]
    assert any("predictive entropy" in c.lower() for c in detail["ranking"]["limits"])
    assert [f["id"] for f in out["facts"]] == ["margin_ratio"]


@pytest.mark.asyncio
async def test_an_unwritable_workspace_leaves_no_evidence_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Saving the evidence text never fails the ranking."""
    blocker = tmp_path / "file"
    blocker.write_text("not a directory")
    monkeypatch.setenv("OLMOEARTH_OUTPUT_ROOT", str(blocker))
    out = await _tools()["olmoearth_review_set"].handler({"scores": _SCORES}, _ctx())  # type: ignore[attr-defined]
    assert out["evidence_detail_path"] is None and out["n_review"] >= 1
    other = [list(r) for r in _SCORES]
    other[1] = [0.1, 5.0]
    cmp = await _tools()["olmoearth_compare_review"].handler(  # type: ignore[attr-defined]
        {"scores_a": _SCORES, "scores_b": other}, _ctx()
    )
    assert cmp["n_differing"] == 1 and cmp["differing_path"] is None
    assert "could not be saved" in cmp["listing_order"]


@pytest.mark.asyncio
async def test_review_set_states_the_warnings_its_scores_file_carries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """exp86 round 6: the provider's multi-class warning reached no brief-8
    answer. A file's warnings travel whole in scores_file_warnings, and the
    limit a known one states goes to must_state."""
    warning = (
        "multi-class logit margin: on Ai2's suite one minus the top probability "
        "ranked errors better on 14 of 16 multi-class tasks (exp76); pass form='top1'"
    )
    other = "3 pixels are not finite and were treated as no-data; pass nodata_mask"
    rows = [[0.0, 2.0, 0.0], [1.5, 0.0, 0.0], [0.0, 0.0, 0.3], [0.9, 0.0, 0.0]]
    (tmp_path / "s.json").write_text(
        json.dumps(
            {
                "grid": [2, 2],
                "scores": rows,
                "score_kind": "window_confidence",
                "score_type": "logit",
                "model": {
                    "repo": "allenai/OlmoEarth-v1-FT-AWF-Base",
                    "revision": "a347b15",
                },
                "package_warnings": [warning, other],
            }
        )
    )
    monkeypatch.setenv("OLMOEARTH_SCORES_ROOT", str(tmp_path))
    out = await _tools()["olmoearth_review_set"].handler(  # type: ignore[attr-defined]
        {"scores_path": str(tmp_path / "s.json"), "budget": 0.5}, _ctx()
    )
    assert out["evidence_covers_this_case"] == "in part"
    assert out["scores_file_warnings"] == [
        "The scores provider (olmoearth-inferencex) warns: " + warning,
        "The scores provider (olmoearth-inferencex) warns: " + other,
    ]
    # The scope's limit and the warning's are one sentence, stated once.
    assert out["must_state"] == [MUST_STATE_MULTICLASS_LOGIT]
    assert out["boundary_neighbours_means"].startswith("boundary_neighbours counts")


@pytest.mark.asyncio
async def test_must_state_never_carries_an_argument_no_agent_tool_takes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """exp87 review: the provider's multi-class warning ends "pass form='top1'",
    and no agent tool takes a form argument. must_state states the fact and
    what this ranking uses, in at most 25 words; the warning stays whole in
    scores_file_warnings."""
    warning = (
        "multi-class logit margin: on Ai2's suite one minus the top probability "
        "ranked errors better on 14 of 16 multi-class tasks (exp76); pass form='top1'"
    )
    rows = [[0.0, 2.0, 0.0], [1.5, 0.0, 0.0], [0.0, 0.0, 0.3], [0.9, 0.0, 0.0]]
    (tmp_path / "s.json").write_text(
        json.dumps(
            {
                "scores": rows,
                "score_kind": "window_confidence",
                "score_type": "logit",
                "model": {"repo": "someone/Other-Model"},
                "package_warnings": [warning],
            }
        )
    )
    monkeypatch.setenv("OLMOEARTH_SCORES_ROOT", str(tmp_path))
    out = await _tools()["olmoearth_review_set"].handler(  # type: ignore[attr-defined]
        {"scores_path": str(tmp_path / "s.json"), "budget": 0.5}, _ctx()
    )
    assert len(out["must_state"]) == 2 <= MUST_STATE_MAX
    assert "someone/Other-Model" in out["must_state"][0]
    assert out["must_state"][1] == MUST_STATE_MULTICLASS_LOGIT
    for sentence in out["must_state"]:
        assert "form" not in sentence and "pass" not in sentence
        assert word_count(sentence) <= MUST_STATE_MAX_WORDS
    assert out["scores_file_warnings"][0].endswith("pass form='top1'")


@pytest.mark.asyncio
async def test_review_set_places_a_files_rows_on_the_files_own_grid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """exp87 review: a model-passed grid placed a file's rows even where the
    file names its own grid and leaves windows out; a transposed grid put
    every window in the wrong row and column."""
    rows = [[0.1, 0.2], [5.0, 0.1], [0.3, 0.2], [4.0, 0.1]]
    (tmp_path / "s.json").write_text(
        json.dumps({"grid": [2, 4], "windows": [1, 3, 5, 6], "scores": rows})
    )
    monkeypatch.setenv("OLMOEARTH_SCORES_ROOT", str(tmp_path))
    out = await _tools()["olmoearth_review_set"].handler(  # type: ignore[attr-defined]
        {"scores_path": str(tmp_path / "s.json"), "budget": 1.0, "grid": [4, 2]},
        _ctx(),
    )
    placed = {r["window_index"]: (r["row"], r["col"]) for r in out["review"]}
    assert placed == {1: (0, 1), 3: (0, 3), 5: (1, 1), 6: (1, 2)}
    assert "set aside for the scores file's own (2x4)" in out["grid_note"]
    (tmp_path / "n.json").write_text(
        json.dumps({"windows": [1, 3], "scores": rows[:2]})
    )
    ungridded = await _tools()["olmoearth_review_set"].handler(  # type: ignore[attr-defined]
        {"scores_path": str(tmp_path / "n.json"), "budget": 1.0, "grid": [4, 2]},
        _ctx(),
    )
    assert all("row" not in r for r in ungridded["review"])
    assert "names no grid" in ungridded["grid_note"]


def test_skill_9_routes_the_error_ranking_question_to_skill_18() -> None:
    """#9 owns self-consistency and OOD; #18 owns which windows are wrong.

    #9's disagreement signal is the comparison tool's ensemble mode, so that
    tool routes the question to #18 too.
    """
    for tool in [*build_uncertainty_tools(), *build_compare_tools()]:
        assert "olmoearth_review_set" in tool.spec.description


def test_skill_18_routes_the_ood_question_back_to_skill_9() -> None:
    """The routing is mutual, so neither skill silently answers the other's question."""
    review = {t.spec.name: t for t in build_review_set_tools()}["olmoearth_review_set"]
    assert "olmoearth_area_of_applicability" in review.spec.description
    assert "olmoearth_compare_results (mode='ensemble')" in review.spec.description


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


@pytest.mark.asyncio
async def test_compare_review_counts_differences_and_declines_the_side_question() -> (
    None
):
    other = [list(r) for r in _SCORES]
    other[1], other[2] = [0.1, 5.0], [0.1, 5.0]  # two windows flip class on side B
    tool = _tools()["olmoearth_compare_review"]
    result = await tool.handler(
        {"scores_a": _SCORES, "scores_b": other, "grid": [4, 4]}, _ctx()
    )  # type: ignore[attr-defined]
    assert result["n_differing"] == 2
    assert {d["window_index"] for d in result["differing"]} == {1, 2}
    assert result["which_side_is_right"] == "not resolvable without labels"
    assert "51 to 70" in result["evidence_scope"]
    assert result["must_state"] == [MUST_STATE_NO_WINNER]
    assert [c["id"] for c in result["forbidden_claims"]] == [
        "winner_without_labels",
        "subset_labelling_sufficient",
    ]
    assert result["where"] in ("mostly on class boundaries", "spread across the scene")


@pytest.mark.asyncio
async def test_compare_review_lists_ten_inline_and_saves_every_differing_window(
    _workspace: Path,
) -> None:
    """exp86 rounds 6 and 7 read place and direction off an inline listing of 50.
    Ten are listed; all of them are in a file, in the same order."""
    n = 64
    a = [[5.0, 0.1] for _ in range(n)]
    b = [list(r) for r in a]
    for i in range(0, n, 2):
        b[i] = [0.1, 6.0]
    out = await _tools()["olmoearth_compare_review"].handler(  # type: ignore[attr-defined]
        {"scores_a": a, "scores_b": b, "grid": [8, 8], "max_listed": 50}, _ctx()
    )
    assert out["n_differing"] == out["n_differing_total"] == 32
    assert out["n_differing_listed"] == len(out["differing"]) == 10
    saved = json.loads(Path(out["differing_path"]).read_text())
    assert Path(out["differing_path"]).parent == _workspace
    assert saved["n_differing_total"] == 32
    assert [d["window_index"] for d in saved["differing"]] == list(range(0, n, 2))
    assert saved["differing"][:10] == out["differing"]
    assert "all 32 are in the file at differing_path" in out["listing_order"]
    assert "not a sample" in out["listing_order"]
    ids = [f["id"] for f in out["facts"]]
    assert ids == ["dominant_change", "more_confident_side", "concentration"]
    assert out["spatial"]["top_band_share"] == 0.25  # every row band holds 8 of 32
    concentration = out["facts"][2]
    assert concentration["grid"] == [8, 8] and concentration["n_differing"] == 32
    assert concentration["max_band"]["axis"] == "rows"
    fewer = await _tools()["olmoearth_compare_review"].handler(  # type: ignore[attr-defined]
        {"scores_a": a, "scores_b": b, "grid": [8, 8], "max_listed": 3}, _ctx()
    )
    assert (
        fewer["n_differing_listed"] == 3
        and fewer["differing_path"] == out["differing_path"]
    )


@pytest.mark.asyncio
async def test_compare_review_with_no_difference_saves_no_listing() -> None:
    out = await _tools()["olmoearth_compare_review"].handler(  # type: ignore[attr-defined]
        {"scores_a": _SCORES, "scores_b": _SCORES, "grid": [4, 4]}, _ctx()
    )
    assert out["n_differing"] == 0 and out["differing_path"] is None
    assert out["facts"] == [] and out["spatial"]["max_band"] is None
    assert out["listing_order"].startswith("no window differs")


@pytest.mark.asyncio
async def test_compare_review_names_classes_both_files_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    names = {"0": "water", "1": "forest"}
    other = [list(r) for r in _SCORES]
    other[1] = [0.1, 5.0]
    (tmp_path / "a.json").write_text(
        json.dumps({"grid": [4, 4], "scores": _SCORES, "classes": names})
    )
    (tmp_path / "b.json").write_text(
        json.dumps({"grid": [4, 4], "scores": other, "classes": names})
    )
    monkeypatch.setenv("OLMOEARTH_SCORES_ROOT", str(tmp_path))
    out = await _tools()["olmoearth_compare_review"].handler(  # type: ignore[attr-defined]
        {
            "scores_path_a": str(tmp_path / "a.json"),
            "scores_path_b": str(tmp_path / "b.json"),
        },
        _ctx(),
    )
    assert out["class_changes"][0]["class_name_a"] == "water"
    assert out["differing"][0]["class_name_b"] == "forest"
    assert (
        "class 0 (water) in map A to class 1 (forest) in map B"
        in out["facts"][0]["sentence"]
    )
    saved = json.loads(Path(out["differing_path"]).read_text())
    assert saved["scores_path_a"] == str(tmp_path / "a.json")


@pytest.mark.asyncio
async def test_compare_review_counts_every_class_change_not_only_the_listed_ones() -> (
    None
):
    """exp86 round 6: answers read a direction and a place off the first listed
    windows ("most flip 1 -> 0", when 1,247 of 1,570 flipped 0 -> 1). The counts
    over all differing windows are given, and the listing says what it is."""
    n = 40
    a = [[5.0, 0.1] for _ in range(n)]  # class 0 everywhere on side A
    b = [list(r) for r in a]
    for i in range(3):  # the first differing windows flip 0 -> 1 ...
        b[i] = [0.1, 5.0]
    a2 = [list(r) for r in a]
    for i in range(3, 30):  # ... but most differences are 1 -> 0 further down
        a2[i] = [0.1, 6.0]
    tool = _tools()["olmoearth_compare_review"]
    result = await tool.handler(
        {"scores_a": a2, "scores_b": b, "grid": [5, 8], "max_listed": 3}, _ctx()
    )  # type: ignore[attr-defined]
    assert result["n_differing"] == 30
    assert [(d["class_a"], d["class_b"]) for d in result["differing"]] == [(0, 1)] * 3
    assert result["class_changes"][0] == {
        "class_a": 1,
        "class_b": 0,
        "n": 27,
        "share_of_differing": 0.9,
    }
    assert result["class_changes"][1]["n"] == 3 and result["n_class_changes"] == 2
    assert result["a_more_confident_share_of_differing"] == 0.9
    assert "not a sample" in result["listing_order"]


@pytest.mark.asyncio
async def test_compare_review_reads_both_sides_from_files(
    tmp_path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    (tmp_path / "a.json").write_text(json.dumps({"grid": [4, 4], "scores": _SCORES}))
    (tmp_path / "b.json").write_text(json.dumps({"grid": [4, 4], "scores": _SCORES}))
    monkeypatch.setenv("OLMOEARTH_SCORES_ROOT", str(tmp_path))
    result = await _registry().dispatch(
        ToolCall(
            id="5",
            name="olmoearth_compare_review",
            arguments={
                "scores_path_a": str(tmp_path / "a.json"),
                "scores_path_b": str(tmp_path / "b.json"),
            },
        ),
        _ctx(),
    )
    assert result["ok"] is True and result["result"]["n_differing"] == 0


@pytest.mark.asyncio
async def test_compare_review_rejects_mismatched_window_counts() -> None:
    result = await _registry().dispatch(
        ToolCall(
            id="6",
            name="olmoearth_compare_review",
            arguments={"scores_a": _SCORES, "scores_b": _SCORES[:8]},
        ),
        _ctx(),
    )
    assert result["ok"] is False and "window counts" in result["error"]


@pytest.mark.asyncio
async def test_compare_review_across_dates_states_the_scope_limit() -> None:
    """Maps of different times: a difference may be change on the ground, and a
    third date does not settle which map was right (exp86 round 7 offered one)."""
    pytest.importorskip("oe_inferencex.compare")
    a = [[0.2, 0.8], [0.7, 0.3], [0.4, 0.6]]
    b = [[0.2, 0.8], [0.3, 0.7], [0.4, 0.6]]
    tool = _tools()["olmoearth_compare_review"]
    apart = await tool.handler(  # type: ignore[attr-defined]
        {"scores_a": a, "scores_b": b, "date_a": "2024-03-01", "date_b": "2024-09-01"},
        _ctx(),
    )
    assert apart["must_state"][0] == MUST_STATE_NO_WINNER
    assert "different times" in apart["must_state"][1]
    assert all(word_count(m) <= MUST_STATE_MAX_WORDS for m in apart["must_state"])
    assert {c["id"] for c in apart["forbidden_claims"]} == {
        "winner_without_labels",
        "another_date_settles_it",
        "subset_labelling_sufficient",
    }
    partly = await tool.handler(  # type: ignore[attr-defined]
        {"scores_a": a, "scores_b": b, "date_a": "2024-03-01"}, _ctx()
    )
    assert "Only one map's date" in partly["must_state"][1]
    assert [c["id"] for c in partly["forbidden_claims"]] == [
        "winner_without_labels",
        "subset_labelling_sufficient",
    ]
    same = await tool.handler(  # type: ignore[attr-defined]
        {"scores_a": a, "scores_b": b, "date_a": "2024-03-01", "date_b": "2024-03-01"},
        _ctx(),
    )
    assert same["must_state"] == [MUST_STATE_NO_WINNER]


@pytest.mark.asyncio
async def test_compare_review_places_files_on_their_own_grid_not_the_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """exp87 review: with a model-passed grid, compare_review placed the rows of
    files that leave windows out on that grid, not the files' own. A
    transposed grid passes every count check and moves every window."""
    rows_a = [[5.0, 0.1], [0.1, 5.0], [5.0, 0.1], [5.0, 0.1]]
    rows_b = [[0.1, 5.0], [0.1, 5.0], [5.0, 0.1], [0.1, 5.0]]
    windows = [0, 3, 5, 7]
    for name, rows in (("a", rows_a), ("b", rows_b)):
        (tmp_path / f"{name}.json").write_text(
            json.dumps({"grid": [2, 4], "windows": windows, "scores": rows})
        )
    monkeypatch.setenv("OLMOEARTH_SCORES_ROOT", str(tmp_path))
    args = {
        "scores_path_a": str(tmp_path / "a.json"),
        "scores_path_b": str(tmp_path / "b.json"),
    }
    tool = _tools()["olmoearth_compare_review"]
    own = await tool.handler(args, _ctx())  # type: ignore[attr-defined]
    asked = await tool.handler({**args, "grid": [4, 2]}, _ctx())  # type: ignore[attr-defined]
    for key in ("differing", "spatial", "facts"):
        assert asked[key] == own[key]
    assert [(d["window_index"], d["row"], d["col"]) for d in asked["differing"]] == [
        (0, 0, 0),
        (7, 1, 3),
    ]
    assert asked["spatial"]["grid"] == [2, 4]
    assert "set aside for the scores files' own (2x4)" in asked["grid_note"]
    assert "grid_note" not in own
    # Two files that name different grids are refused.
    (tmp_path / "c.json").write_text(
        json.dumps({"grid": [4, 2], "windows": windows, "scores": rows_b})
    )
    result = await _registry().dispatch(
        ToolCall(
            id="g",
            name="olmoearth_compare_review",
            arguments={**args, "scores_path_b": str(tmp_path / "c.json")},
        ),
        _ctx(),
    )
    assert result["ok"] is False and "different grids" in result["error"]


@pytest.mark.asyncio
async def test_compare_review_counts_only_the_grids_the_files_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """exp87 re-review: a whole-grid file that names no grid made the model's
    grid count as a file-named grid, so A (2x4) beside B (no grid) with a
    passed 4x2 was refused as 'different grids', and a passed grid equal to
    A's own was reported as not used."""
    rows_a = [[5.0, 0.1]] * 4 + [[0.1, 5.0]] * 4
    rows_b = [[0.1, 5.0]] * 8
    (tmp_path / "a.json").write_text(json.dumps({"grid": [2, 4], "scores": rows_a}))
    (tmp_path / "b.json").write_text(json.dumps({"scores": rows_b}))
    monkeypatch.setenv("OLMOEARTH_SCORES_ROOT", str(tmp_path))
    tool = _tools()["olmoearth_compare_review"]
    args = {
        "scores_path_a": str(tmp_path / "a.json"),
        "scores_path_b": str(tmp_path / "b.json"),
    }
    out = await tool.handler({**args, "grid": [4, 2]}, _ctx())  # type: ignore[attr-defined]
    assert out["spatial"]["grid"] == [2, 4]
    assert "set aside for the scores files' own (2x4)" in out["grid_note"]
    same = await tool.handler({**args, "grid": [2, 4]}, _ctx())  # type: ignore[attr-defined]
    assert same["spatial"]["grid"] == [2, 4] and "grid_note" not in same
    # windows listed in both, the grid named by one only, the same grid passed
    for name, rows, extra in (("c", rows_a, {"grid": [2, 4]}), ("d", rows_b, {})):
        (tmp_path / f"{name}.json").write_text(
            json.dumps({**extra, "windows": list(range(8)), "scores": rows})
        )
    both = await tool.handler(  # type: ignore[attr-defined]
        {
            "scores_path_a": str(tmp_path / "c.json"),
            "scores_path_b": str(tmp_path / "d.json"),
            "grid": [2, 4],
        },
        _ctx(),
    )
    assert both["spatial"]["grid"] == [2, 4] and "grid_note" not in both


@pytest.mark.asyncio
async def test_compare_review_drops_the_models_grid_for_a_file_without_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A file that leaves windows out and names no grid gives nothing to check
    the model's grid against, so no grid is used."""
    rows_a = [[5.0, 0.1], [0.1, 5.0]]
    rows_b = [[0.1, 5.0], [0.1, 5.0]]
    for name, rows in (("a", rows_a), ("b", rows_b)):
        (tmp_path / f"{name}.json").write_text(
            json.dumps({"windows": [2, 5], "scores": rows})
        )
    monkeypatch.setenv("OLMOEARTH_SCORES_ROOT", str(tmp_path))
    out = await _tools()["olmoearth_compare_review"].handler(  # type: ignore[attr-defined]
        {
            "scores_path_a": str(tmp_path / "a.json"),
            "scores_path_b": str(tmp_path / "b.json"),
            "grid": [3, 2],
        },
        _ctx(),
    )
    assert out["n_differing"] == 1 and out["differing"][0]["window_index"] == 2
    assert "row" not in out["differing"][0] and "spatial" not in out
    assert "name no grid" in out["grid_note"]


@pytest.mark.asyncio
async def test_every_review_set_result_keeps_must_state_inside_the_contract() -> None:
    """The contract: at most 3 must_state sentences of at most 25 words each,
    and each forbidden id once, whatever the case adds (dates included)."""
    pytest.importorskip("oe_inferencex.compare")
    a = [[0.2, 0.8], [0.7, 0.3], [0.4, 0.6], [0.9, 0.1]]
    b = [[0.2, 0.8], [0.3, 0.7], [0.4, 0.6], [0.1, 0.9]]
    tools = _tools()
    outs = [
        await tools["olmoearth_review_set"].handler({"scores": a}, _ctx()),  # type: ignore[attr-defined]
        await tools["olmoearth_compare_review"].handler(  # type: ignore[attr-defined]
            {"scores_a": a, "scores_b": b, "grid": [2, 2]}, _ctx()
        ),
        await tools["olmoearth_compare_review"].handler(  # type: ignore[attr-defined]
            {
                "scores_a": a,
                "scores_b": b,
                "date_a": "2024-03-01",
                "date_b": "2025-09-01",
            },
            _ctx(),
        ),
    ]
    for out in outs:
        assert 1 <= len(out["must_state"]) <= MUST_STATE_MAX
        assert all(word_count(m) <= MUST_STATE_MAX_WORDS for m in out["must_state"])
        ids = [c["id"] for c in out["forbidden_claims"]]
        assert len(ids) == len(set(ids))


def test_a_comparison_across_properties_is_marked_once() -> None:
    """exp87 review: forbidden ids are deduplicated per result; marking a
    comparison of different properties twice used to list each entry twice."""
    from olmoearth_agent.tools.compare import _mark_different

    out: dict = {}
    _mark_different(out)
    _mark_different(out)
    assert len(out["must_state"]) == 1
    # the forbidden claim comes from the statistical rules, which name the
    # properties; marking adds none, so none can be listed twice
    assert "forbidden_claims" not in out
