# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Canonical run-pipeline stages + which ones a given skill actually touches.

The web UI shows a "Workflow" rail tracking a run through the OlmoEarth pipeline:
``Load context -> Define AOI -> Find model -> Submit run -> Poll status ->
Fetch results -> Report``. For an **advisory** skill (e.g. ``negative-sampler``,
which never submits a prediction) most of those stages are irrelevant and would
sit forever "pending", reading as a stuck run. This module is the source of
truth for which stages a skill's tools touch, so the UI can render only the
relevant subset. Mirror the keys/labels in ``webui/js/run.js`` (``WF_STAGES``).

Pure data + predicates; no torch, no I/O.
"""

from __future__ import annotations

from collections.abc import Callable

#: (key, label, predicate over a tool name). ``context`` and ``report`` bracket
#: every run; the middle five are the actual prediction pipeline. ``poll`` is an
#: EXACT match (``olmoearth_get_prediction``) so ``get_prediction_result`` falls
#: to ``fetch``, not ``poll``.
WORKFLOW_STAGES: list[tuple[str, str, Callable[[str], bool]]] = [
    ("context", "Load context", lambda n: "load_context" in n),
    ("aoi", "Define AOI", lambda n: "request_aoi" in n),
    ("model", "Find model", lambda n: "load_skill" in n or "search_predictions" in n),
    ("submit", "Submit run", lambda n: "submit_prediction" in n),
    ("poll", "Poll status", lambda n: n == "olmoearth_get_prediction" or "_poll" in n),
    (
        "fetch",
        "Fetch results",
        lambda n: any(
            s in n for s in ("fetch_results", "get_prediction_result", "get_results")
        ),
    ),
    ("report", "Report", lambda _n: False),  # completed by the final answer
]

#: Always present: a run implicitly loads context and ends in a report.
_ALWAYS = frozenset({"context", "report"})

#: When a skill submits a prediction, the whole run pipeline applies (you can't
#: submit without an AOI, poll without a submit, etc.).
_RUN_PIPELINE = frozenset({"context", "aoi", "model", "submit", "poll", "fetch", "report"})


def stage_for_tool(tool_name: str) -> str | None:
    """The pipeline stage a tool advances, or ``None`` if it's off-pipeline."""
    n = tool_name or ""
    for key, _label, matches in WORKFLOW_STAGES:
        if matches(n):
            return key
    return None


def skill_workflow_stages(tools: list[str] | None) -> list[str]:
    """The relevant stage keys for a skill, given its tools, in pipeline order.

    A **run** skill (one whose tools include ``submit_prediction``) gets the full
    pipeline; an **advisory** skill (e.g. negative-sampler) gets just the stages
    its tools touch, bracketed by ``context`` + ``report`` — so the rail never
    shows submit/poll/fetch it will never reach.
    """
    keys = set(_ALWAYS)
    for tool in tools or []:
        stage = stage_for_tool(tool)
        if stage is not None:
            keys.add(stage)
    if "submit" in keys:
        keys |= _RUN_PIPELINE
    return [key for key, _label, _m in WORKFLOW_STAGES if key in keys]
