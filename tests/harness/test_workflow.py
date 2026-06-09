# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Per-skill run-pipeline stage relevance (so the UI rail can skip dead steps)."""

from __future__ import annotations

from olmoearth_agent.harness.workflow import (
    WORKFLOW_STAGES,
    skill_workflow_stages,
    stage_for_tool,
)
from olmoearth_agent.skills.registry import SKILLS


def test_stage_for_tool_poll_vs_fetch_are_not_confused() -> None:
    # get_prediction is the poll; get_prediction_result is the fetch (substring trap)
    assert stage_for_tool("olmoearth_get_prediction") == "poll"
    assert stage_for_tool("olmoearth_get_prediction_result") == "fetch"
    assert stage_for_tool("olmoearth_submit_prediction") == "submit"
    assert stage_for_tool("olmoearth_request_aoi") == "aoi"
    assert stage_for_tool("olmoearth_compare_results") is None  # off-pipeline


def test_run_skill_gets_the_full_pipeline() -> None:
    predict = next(s for s in SKILLS if s.name == "olmoearth-predict")
    stages = skill_workflow_stages(predict.tools)
    # submitting a prediction implies the whole pipeline, AOI included
    assert stages == ["context", "aoi", "model", "submit", "poll", "fetch", "report"]


def test_advisory_skill_skips_submit_poll_fetch() -> None:
    neg = next(s for s in SKILLS if s.name == "olmoearth-negative-sampler")
    stages = skill_workflow_stages(neg.tools)
    assert "submit" not in stages and "poll" not in stages and "fetch" not in stages
    assert stages[0] == "context" and stages[-1] == "report"  # always bracketed


def test_stages_are_in_canonical_pipeline_order() -> None:
    order = [k for k, _l, _m in WORKFLOW_STAGES]
    for spec in SKILLS:
        if spec.number < 1:
            continue
        stages = skill_workflow_stages(spec.tools)
        assert stages == [k for k in order if k in stages]  # subsequence, ordered
        assert set(stages) <= set(order)
