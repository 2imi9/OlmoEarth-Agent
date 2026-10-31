# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""A scores file pooled from a Studio result's own files, through the later tools.

``olmoearth_scores_from_studio`` writes a run directory whose manifest has
``source.kind`` ``studio_files`` and a ``studio`` block, and
``olmoearth_scores_from_file`` carries that block and its assumption into the
scores file it writes. live3 (31 October 2026) ran ``olmoearth_review_set`` and
``olmoearth_plan_label_sample`` on the kelp result's epsg32611 file and found
three things this module pins: the review set's evidence scope called the
Studio model "not an OlmoEarth model" by its id, although Studio fine-tunes
OlmoEarth foundation models (the kelp model's record names its
``foundation_model_version``); it reported ``score_type_detected`` "logit"
beside ``score_type`` "probability" for rows ``[m, 0]``; and the
uncalibrated-scores sentence reached the later results only as a warning or an
``assumption``, not in ``must_state``.

The files below are shaped as that tool writes one, with no raster: the review
set reads the JSON alone, and a random plan, its estimate and its
certification need only the package's ``estimate`` and ``certify`` modules,
which a test skips without.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import pytest

from olmoearth_agent.analysis.output_contract import (
    MUST_STATE_MAX,
    MUST_STATE_MAX_WORDS,
    word_count,
)
from olmoearth_agent.analysis.review_set import (
    MUST_STATE_STUDIO_FILES,
    MUST_STATE_UNNAMED_MODEL,
    RANKING_SOURCE,
    STUDIO_WINDOW_CONFIDENCE,
    must_state_other_model,
    ranking_evidence_scope,
)
from olmoearth_agent.harness.state import ThreadState
from olmoearth_agent.llm.types import ToolCall
from olmoearth_agent.tools import review_set as review_tools
from olmoearth_agent.tools import scores_file, studio_scores
from olmoearth_agent.tools import statistical_rules as rules
from olmoearth_agent.tools.estimation import build_estimation_tools
from olmoearth_agent.tools.registry import ToolContext, ToolRegistry
from olmoearth_agent.tools.review_set import build_review_set_tools

REVIEW = "olmoearth_review_set"
PLAN = "olmoearth_plan_label_sample"
GRADE = "olmoearth_estimate_map_error"
CERTIFY = "olmoearth_certify_zone"

#: The assumption a top-k zone's manifest records (``studio_scores._assumption``
#: for a top-2 result of two classes).
TOPK_ASSUMPTION = (
    "The scores are Studio's uncalibrated softmax scores, quantized to 1/254 "
    "and decoded as (value - 1) / 254, the files README's formula: they rank "
    "windows and are not probabilities that a class is right."
)

#: A Studio model id and result id, made up for these files.
MODEL_ID = "model-kelp-test"
RESULT_ID = "result-kelp-test"


@pytest.fixture(autouse=True)
def _roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("OLMOEARTH_SCORES_ROOT", str(tmp_path))
    monkeypatch.setenv("OLMOEARTH_OUTPUT_ROOT", str(tmp_path))
    return tmp_path


def _scores_file(
    root: Path,
    *,
    kind: str | None = studio_scores.TOPK,
    model: dict[str, Any] | None = None,
    n: int = 64,
) -> str:
    """A window-confidence scores file of an 8 x 8 grid, as scores_from_file writes it.

    ``kind`` is the ``studio`` block's kind (top-k scores or a ``[0, 1]``
    score); ``None`` leaves the block out, as a direct run's file has none.
    Each row is the window's pooled top-1 probability at its class and 0 at
    the other class.
    """
    rng = random.Random(5)
    confidence = [round(rng.uniform(0.5, 1.0), 6) for _ in range(n)]
    classes = [i % 2 for i in range(n)]
    rows = [[c, 0.0] if k == 0 else [0.0, c] for c, k in zip(confidence, classes)]
    payload: dict[str, Any] = {
        "format": "olmoearth-agent/scores@1",
        "provenance": scores_file.PROVENANCE,
        "score_kind": "window_confidence",
        "score_type": "probability",
        "values": "probabilities",
        "row_encoding": scores_file.ROW_ENCODING,
        "grid": [8, 8],
        "patch_px": 11,
        "scores": rows,
        "p1": confidence,
        "map_class": classes,
        "classes": {"0": "absent", "1": "present"},
        "model": model or {"repo": "allenai/OlmoEarth-v1-FT-AWF-Base"},
        "package_warnings": [],
    }
    if kind is not None:
        assumption = (
            TOPK_ASSUMPTION
            if kind == studio_scores.TOPK
            else studio_scores.UNIT_ASSUMPTION_SENTENCE
        )
        studio: dict[str, Any] = {
            "kind": kind,
            "property_name": "kelp",
            "assumption": assumption,
        }
        if kind == studio_scores.TOPK:
            studio.update(k=2, decode=studio_scores.TOPK_DECODE)
        # As scores_file.studio_model writes it: no revision, the result id.
        studio_model = {
            "repo": None,
            "revision": None,
            "id": MODEL_ID,
            "name": "Kelp Base",
            "result_id": RESULT_ID,
        }
        payload["provenance"] = scores_file.STUDIO_PROVENANCE
        payload["model"] = model or studio_model
        payload["studio"] = studio
        payload["assumption"] = assumption
        payload["warnings"] = [assumption]
    path = root / f"scores_run_{kind or 'direct'}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


async def _ok(name: str, args: dict[str, Any]) -> dict[str, Any]:
    registry = ToolRegistry()
    registry.register_all(build_review_set_tools())
    registry.register_all(build_estimation_tools())
    ctx = ToolContext(studio=None, state=ThreadState())  # type: ignore[arg-type]
    out = await registry.dispatch(ToolCall(id="1", name=name, arguments=args), ctx)
    assert out["ok"] is True, out
    result: dict[str, Any] = out["result"]
    return result


def _ids(result: dict[str, Any]) -> list[str]:
    return [c["id"] for c in result.get("forbidden_claims", [])]


# --------------------------------------------------------------------------- the evidence scope


def test_the_studio_files_scope_does_not_cover_the_case() -> None:
    """Measured on OlmoEarth models' class-score margins, not on a Studio
    model's pooled scores: "no", with its own short must_state, and never
    "not an OlmoEarth model"."""
    scope = ranking_evidence_scope(
        score_kind=STUDIO_WINDOW_CONFIDENCE, n_classes=2, model=MODEL_ID
    )
    assert scope["covers"] == "no"
    assert scope["sentence"] == (
        f"{RANKING_SOURCE}, not on a Studio model's top-k softmax scores (or [0, "
        "1] score) pooled to window confidence, so it does not cover this case."
    )
    assert "not an OlmoEarth model" not in scope["sentence"]
    assert "24" not in scope["sentence"] and "suite" not in scope["sentence"]
    assert scope["must_state"] == MUST_STATE_STUDIO_FILES
    assert word_count(MUST_STATE_STUDIO_FILES) <= MUST_STATE_MAX_WORDS
    # Every other case keeps its scope: a model outside the family, or none.
    other = ranking_evidence_scope(score_kind="logit", n_classes=2, model=MODEL_ID)
    assert f"come from {MODEL_ID}, not an OlmoEarth model" in other["sentence"]
    assert other["must_state"] == must_state_other_model(MODEL_ID)
    unnamed = ranking_evidence_scope(score_kind="logit", n_classes=2)
    assert unnamed["must_state"] == MUST_STATE_UNNAMED_MODEL


def test_the_studio_kind_of_a_file_is_read_from_its_studio_block() -> None:
    """The value repeated in review_set (studio_scores imports it) is the same."""
    assert review_tools.STUDIO_TOPK == studio_scores.TOPK
    kind = review_tools.file_studio_kind
    assert kind({"studio": {"kind": "top_k"}}) == "top_k"
    assert kind({"studio": {"kind": studio_scores.UNIT}}) == "unit_score"
    assert kind({"studio": {}}) == ""
    assert kind({"studio": "top_k"}) is None
    assert kind({}) is None
    assert rules.STUDIO_TOPK_SCORES_MUST_STATE == studio_scores.TOPK_SCORES_SENTENCE


@pytest.mark.asyncio
async def test_a_review_set_of_a_studio_top_k_file_is_scoped_and_states_its_scores(
    tmp_path: Path,
) -> None:
    """live3: "these scores come from <model id>, not an OlmoEarth model",
    score_type_detected "logit", and the uncalibrated sentence a warning only."""
    path = _scores_file(tmp_path)
    review = await _ok(REVIEW, {"scores_path": path, "budget": 0.1})
    assert review["evidence_covers_this_case"] == "no"
    assert review["evidence_scope"].startswith(RANKING_SOURCE)
    assert "not an OlmoEarth model" not in json.dumps(review)
    assert review["must_state"] == [
        MUST_STATE_STUDIO_FILES,
        rules.STUDIO_TOPK_SCORES_MUST_STATE,
    ]
    assert len(review["must_state"]) <= MUST_STATE_MAX
    assert rules.EVIDENCE_OUTSIDE_ITS_SCOPE in _ids(review)
    # The rows [m, 0] are an encoding: their type is the file's, not detected.
    assert review["score_type"] == "probability"
    assert "score_type_detected" not in review
    # The assumption still travels as the file's warning.
    assert f"The scores file warns: {TOPK_ASSUMPTION}" in review["scores_file_warnings"]
    assert review["provenance"] == scores_file.STUDIO_PROVENANCE


@pytest.mark.asyncio
async def test_a_review_set_of_a_studio_unit_score_file_is_scoped_only(
    tmp_path: Path,
) -> None:
    """A [0, 1] score's file gets the Studio scope; its assumption stays a
    warning, and no top-k sentence is stated."""
    path = _scores_file(tmp_path, kind=studio_scores.UNIT)
    review = await _ok(REVIEW, {"scores_path": path, "budget": 0.1})
    assert review["evidence_covers_this_case"] == "no"
    assert review["must_state"] == [MUST_STATE_STUDIO_FILES]
    assert "score_type_detected" not in review
    assert any(
        studio_scores.UNIT_ASSUMPTION_SENTENCE in w
        for w in review["scores_file_warnings"]
    )


@pytest.mark.asyncio
async def test_a_direct_runs_window_file_keeps_its_scope_without_detection(
    tmp_path: Path,
) -> None:
    """A direct run's window confidences keep the scope they had (an
    OlmoEarth model's two-class margin; another model's "not an OlmoEarth
    model"); only score_type_detected is left out. Inline rows keep it."""
    olmo = await _ok(REVIEW, {"scores_path": _scores_file(tmp_path, kind=None)})
    assert olmo["evidence_covers_this_case"] == "yes"
    assert olmo["must_state"] == []
    assert olmo["score_type"] == "probability"
    assert "score_type_detected" not in olmo
    other_root = tmp_path / "other"
    other_root.mkdir()
    other_path = _scores_file(
        other_root, kind=None, model={"repo": "someone/Other-Model"}
    )
    other = await _ok(REVIEW, {"scores_path": other_path})
    assert "someone/Other-Model, not an OlmoEarth model" in other["evidence_scope"]
    assert other["evidence_covers_this_case"] == "not known"
    inline = await _ok(REVIEW, {"scores": [[0.9, 0.1], [0.4, 0.6]], "budget": 1.0})
    assert inline["score_type_detected"] == "probability"


# --------------------------------------------------------------------------- the label designs


@pytest.mark.asyncio
async def test_a_plan_its_estimate_and_its_certification_state_the_top_k_scores(
    tmp_path: Path,
) -> None:
    """Every later result on a Studio top-k file states the uncalibrated
    sentence in must_state, where the pixel-value top-k route states its own."""
    pytest.importorskip("oe_inferencex.estimate")
    pytest.importorskip("oe_inferencex.certify.zone")
    path = _scores_file(tmp_path)
    plan = await _ok(
        PLAN, {"scores_path": path, "budget": 40, "design": "random", "seed": 2}
    )
    assert plan["must_state"] == [rules.STUDIO_TOPK_SCORES_MUST_STATE]
    assert plan["assumption"] == TOPK_ASSUMPTION
    design = json.loads(Path(plan["design_path"]).read_text(encoding="utf-8"))
    assert design["population"]["source"]["studio"]["kind"] == studio_scores.TOPK
    wrong = [0] * len(design["sample"]["indices"])
    graded = await _ok(GRADE, {"design_path": plan["design_path"], "wrong": wrong})
    assert graded["must_state"][-1] == rules.STUDIO_TOPK_SCORES_MUST_STATE
    assert len(graded["must_state"]) <= MUST_STATE_MAX
    zone = await _ok(
        CERTIFY, {"design_path": plan["design_path"], "wrong": wrong, "alpha": 0.2}
    )
    assert rules.STUDIO_TOPK_SCORES_MUST_STATE in zone["must_state"]
    assert len(zone["must_state"]) <= MUST_STATE_MAX


@pytest.mark.asyncio
async def test_a_plan_of_a_unit_score_or_direct_file_states_no_top_k_scores(
    tmp_path: Path,
) -> None:
    """A [0, 1] score's file keeps its assumption as the plan's assumption;
    a direct run's file states nothing of Studio's scores."""
    pytest.importorskip("oe_inferencex.estimate")
    unit = await _ok(
        PLAN,
        {
            "scores_path": _scores_file(tmp_path, kind=studio_scores.UNIT),
            "budget": 20,
            "design": "random",
        },
    )
    assert rules.STUDIO_TOPK_SCORES_MUST_STATE not in unit.get("must_state", [])
    assert unit["assumption"] == studio_scores.UNIT_ASSUMPTION_SENTENCE
    direct = await _ok(
        PLAN,
        {
            "scores_path": _scores_file(tmp_path, kind=None),
            "budget": 20,
            "design": "random",
        },
    )
    assert rules.STUDIO_TOPK_SCORES_MUST_STATE not in direct.get("must_state", [])
    design = json.loads(Path(direct["design_path"]).read_text(encoding="utf-8"))
    assert "studio" not in design["population"]["source"]
