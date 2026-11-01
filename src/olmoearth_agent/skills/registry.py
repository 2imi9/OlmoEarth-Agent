# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Skill catalog manifest, the harness's view of all 5 skills.

This is the structural "slot" for every skill in ``SKILLS.md``. Each
:class:`SkillSpec` records the skill's category, build status, and the
tools it contributes. :func:`build_default_registry` assembles the
tool bundles of the currently-implemented skills into a
:class:`ToolRegistry` the lead agent can use.

By default the agent is the Studio run loop and a verification layer (review
sets, label samples, error rates and certified zones); every tool a catalog
row lists is registered. ``OLMOEARTH_READ_ONLY`` (:data:`READ_ONLY_ENV`, read
by :func:`olmoearth_agent.tools.registry.opted_in`) declares this run's Studio
key read-only: the tools that write to Studio (:data:`STUDIO_WRITE_TOOLS`)
are left out, and the capability card says why.

As each skill is implemented, flip its ``status`` to ``"implemented"``
and add its ``build_*_tools`` bundle to :func:`build_default_registry`.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Literal

from olmoearth_agent.harness.capabilities import READ_ONLY_STUDIO
from olmoearth_agent.provenance.tools import build_provenance_tools
from olmoearth_agent.tools.certificate import build_certificate_tools
from olmoearth_agent.tools.compare import build_compare_tools
from olmoearth_agent.tools.estimation import build_estimation_tools
from olmoearth_agent.tools.narrative import build_narrative_tools
from olmoearth_agent.tools.predict import build_predict_tools
from olmoearth_agent.tools.registry import RegisteredTool, ToolRegistry, opted_in
from olmoearth_agent.tools.review_set import build_review_set_tools
from olmoearth_agent.tools.scores_file import build_scores_file_tools
from olmoearth_agent.tools.skill_tools import build_skill_tools
from olmoearth_agent.tools.studio import build_studio_tools
from olmoearth_agent.tools.studio_labels import build_studio_label_tools
from olmoearth_agent.tools.studio_scores import build_studio_scores_tools

SkillStatus = Literal["foundational", "implemented", "vendored", "planned"]

#: The setting that declares this run's Studio key read-only (``1``, ``true``,
#: ``yes`` or ``on``; see :func:`~olmoearth_agent.tools.registry.opted_in`).
#: Studio's keys can be read-only, and ``/users/me`` does not show it, so the
#: operator says so here.
#: Unset, the registry is unchanged.
READ_ONLY_ENV = "OLMOEARTH_READ_ONLY"

#: The tools that write to Studio (a project, a prediction), left out of the
#: default registry while :data:`READ_ONLY_ENV` is on. Drawing an area
#: (``olmoearth_request_aoi``) stays: the web UI saves it, and answers a
#: refused save with HTTP 403 (``serve.api_create_area``).
STUDIO_WRITE_TOOLS = frozenset(
    {"olmoearth_create_project", "olmoearth_submit_prediction"}
)


@dataclass(frozen=True)
class SkillSpec:
    """One row of the skill catalog (see ``SKILLS.md``).

    ``tools`` names every tool the skill contributes.
    """

    number: int
    name: str
    category: Literal["Foundational", "Configure", "Run", "Analyze", "Report"]
    status: SkillStatus
    summary: str
    tools: list[str] = field(default_factory=list)


#: The full catalog. ``implemented`` = the row's tools are built and wired in
#: this repo (row #0 included: the base Studio tools, not a SKILLS.md entry);
#: ``vendored`` = an instruction package moved into ``skills/packages/`` from
#: 2imi9/OlmoEarth-Skills (frozen upstream), served by olmoearth_load_skill.
#: No row is ``foundational`` or ``planned``.
SKILLS: list[SkillSpec] = [
    SkillSpec(
        0,
        "studio-core",
        "Foundational",
        "implemented",
        "Base Studio API tools every Run/Analyze skill builds on, and the count "
        "of the Studio annotations a model's training selector selects now, by "
        "split and class (no random design recorded: no error rate).",
        [
            "olmoearth_load_context",
            "olmoearth_create_project",
            "olmoearth_request_aoi",
            "olmoearth_get_prediction",
            "olmoearth_studio_label_inventory",
        ],
    ),
    # #1 is the one instruction package left (loaded by SkillLoader through
    # olmoearth_load_skill); data prep, embeddings and rslearn, the
    # training-side packages, were removed on 31 October 2026.
    SkillSpec(
        1,
        "olmoearth-studio-job-config",
        "Configure",
        "vendored",
        "Task description -> full Studio 'new model' wizard answers "
        "(name/type/foundation/label/training data/split/temporal/sources/area); "
        "14 presets + validator. Studio trains on Ai2 compute.",
        ["olmoearth_load_skill"],
    ),
    # #2 holds the comparison tool's four modes: a pair, a group, change over
    # time (series; once its own row) and ensemble disagreement (ensemble;
    # once its own row too).
    SkillSpec(
        2,
        "olmoearth-predict",
        "Run",
        "implemented",
        "Run loop: search predictions (find model_id), submit, poll, "
        "fetch results (tile URLs), sample the model output at a point, and "
        "compare results quantitatively with one tool "
        "(olmoearth_compare_results: grid-sampled, no ground truth): a pair "
        "of results, a group of 2-6 models (pairwise matrix + ensemble "
        "consensus), change over time as one model's dated results traced "
        "as a series (mode='series': date-ordered, legend-calibrated shifts; "
        "estimate movement, not verified ground change), or "
        "ensemble-disagreement confidence across distinct results "
        "(mode='ensemble'), the signal that works when Studio gives you hard "
        "classes only; when per-class scores exist, #5 ranks errors better "
        "(measured).",
        [
            "olmoearth_search_predictions",
            "olmoearth_submit_prediction",
            "olmoearth_get_prediction",
            "olmoearth_fetch_results",
            "olmoearth_get_prediction_result",
            "olmoearth_pixel_value",
            "olmoearth_compare_results",
        ],
    ),
    SkillSpec(
        3,
        "olmoearth-provenance",
        "Report",
        "implemented",
        "Manifest of every tool call; emits a replay skeleton (the calls in "
        "order, not re-run); lists each verification call's design and seed.",
        ["olmoearth_provenance_summary"],
    ),
    SkillSpec(
        4,
        "olmoearth-case-narrative",
        "Report",
        "implemented",
        "Stakeholder Markdown writeup with tile URLs + freshness gate; states "
        "the accuracy the request's verification tools measured, or that none "
        "was.",
        ["olmoearth_case_narrative"],
    ),
    # #5 closes the gap #2's ensemble mode cannot: ranking WHICH windows are
    # wrong, from the model's own top-1-minus-top-2 margin. Studio's
    # pixel-value returns only raw_value/classification per band (no logits),
    # which is exactly why the ensemble mode reaches for disagreement -- with
    # hard classes it is the only signal available. A class map with top-k
    # scores (a prediction submitted with probability_output) carries its
    # rank-1 and rank-2 softmax scores as bands of their own, and #5 reads its
    # margin from them. Given real scores the ordering flips: upstream
    # (2imi9/olmoearth_inferenceX-record) measured ensemble disagreement, cross-model
    # disagreement, two-view disagreement and feature-space typicality against
    # the margin on seven expert-labelled testbeds and none of them won. So the
    # two are complementary, not rivals: the ensemble mode for hard-class
    # Studio results, #5 whenever per-class scores exist.
    SkillSpec(
        5,
        "olmoearth-review-set",
        "Analyze",
        "implemented",
        "Label-free review set from the model's own top-1-minus-top-2 margin: "
        "which windows to check first at a budget (from caller scores, from a "
        "direct model run's files, score_area.py's scores raster and manifest "
        "or what oe-inferencex from-olmoearth writes, or from a Studio "
        "result whose band is a binary score in [0, 1], or a class map with "
        "top-k scores, sampled on a grid or read from its native-resolution "
        "files, one scores file per UTM zone), "
        "boundary-first ordering, and the attainable-ceiling arithmetic. Then "
        "how wrong the map is: a labelled sample drawn by a design, its error "
        "rate with the interval valid at every sample size first, then the one "
        "that design earns, per-class accuracy and a certified zone "
        "(the optional olmoearth-inferencex extra), which a direct run's map "
        "can carry as a certificate anyone holding the map can check. A "
        "review set is not a sample. Which of two class maps of one time is "
        "more accurate: labels on a random sample of the windows where they "
        "differ, graded by the exact sign test and an interval on the "
        "difference.",
        [
            "olmoearth_review_set",
            "olmoearth_review_set_from_result",
            "olmoearth_scores_from_file",
            "olmoearth_scores_from_studio",
            "olmoearth_compare_review",
            "olmoearth_review_budget_ceiling",
            "olmoearth_plan_label_sample",
            "olmoearth_estimate_map_error",
            "olmoearth_certify_zone",
            "olmoearth_write_certificate",
            "olmoearth_verify_certificate",
        ],
    ),
]


def skills_by_status(status: SkillStatus) -> list[SkillSpec]:
    """All catalog entries with the given build status."""
    return [s for s in SKILLS if s.status == status]


def build_default_registry() -> ToolRegistry:
    """Assemble a :class:`ToolRegistry` from every implemented skill bundle.

    Core bundles are sent to the LLM on every turn; every bundle below is
    core. A deferred bundle would be registered under its skill's name
    (``group=``) and sent only once that skill is loaded
    (``olmoearth_load_skill``) or forced from the web UI; see
    :mod:`olmoearth_agent.tools.registry`. A new skill adds its
    ``build_*_tools()`` bundle here and flips its catalog ``status``.

    While :data:`READ_ONLY_ENV` (``OLMOEARTH_READ_ONLY``) is on, the tools in
    :data:`STUDIO_WRITE_TOOLS` are left out, and the registry's ``cannot``
    gives the capability card its line
    (:data:`~olmoearth_agent.harness.capabilities.READ_ONLY_STUDIO`).

    Returns
    -------
    ToolRegistry
        The agent's tools for this process's settings.

    Examples
    --------
    >>> registry = build_default_registry()
    >>> "olmoearth_review_set" in registry.names()
    True
    """
    registry = ToolRegistry()
    # The tools left out: with a read-only Studio key, the ones that write
    # to Studio.
    skip: set[str] = set()
    if opted_in(READ_ONLY_ENV):
        skip |= STUDIO_WRITE_TOOLS
        registry.cannot["studio_write"] = READ_ONLY_STUDIO

    def add(tools: Iterable[RegisteredTool], group: str | None = None) -> None:
        """Register ``tools`` (deferred under ``group``), less any left out."""
        registry.register_all(
            (t for t in tools if t.spec.name not in skip),
            group=group,
        )

    # --- Core: sent on every turn. ---
    add(build_studio_tools())
    # The Studio annotations a model's training selector selects, counted
    # (read-only): core, since whether labels exist is asked before any review
    # set or estimate.
    add(build_studio_label_tools())
    add(build_predict_tools())
    # How Studio results differ (#2: a pair, a group, a series, an ensemble):
    # one tool.
    add(build_compare_tools())
    # Label-free error ranking (skill #5); complements #2's ensemble mode,
    # which is what remains reachable when Studio yields only hard classes.
    add(build_review_set_tools())
    # The scores provider (#5): a direct model run's raster + manifest as the
    # review set's and the estimation tools' input. Core, not deferred: it is
    # the entry point of a path whose other tools are all core, and deferring
    # it would put a load_skill turn before the first useful call.
    add(build_scores_file_tools())
    # A Studio result's own native-resolution files (top-k scores or a [0, 1]
    # score) as the same run directories and scores files (#5); core for the
    # same reason.
    add(build_studio_scores_tools())
    # How wrong is the map (skill #5, second half): design-based estimation
    # through the optional inferencex extra; the tools say so when it is absent.
    add(build_estimation_tools())
    # A certified zone of a direct run's map packed into a certificate, and its
    # check (#5): the same extra, no other dependency, so core like the rest.
    add(build_certificate_tools())
    add(build_narrative_tools())
    add(build_provenance_tools())
    # Skill loading, last among the bundles so its description lists any
    # deferred group.
    add(build_skill_tools(registry=registry))
    return registry
