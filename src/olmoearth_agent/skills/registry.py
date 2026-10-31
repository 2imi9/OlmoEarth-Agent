# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Skill catalog manifest, the harness's view of all 18 skills.

This is the structural "slot" for every skill in ``SKILLS.md``. Each
:class:`SkillSpec` records the skill's category, build status, and the
tools it contributes. :func:`build_default_registry` assembles the
tool bundles of the currently-implemented skills into a
:class:`ToolRegistry` the lead agent can use.

By default the agent is the Studio run loop and a verification layer (review
sets, label samples, error rates and certified zones). A tool a catalog row
lists under ``opt_in`` is left out of the default registry, with its one-line
reason; it stays in the code and is registered as before when
``OLMOEARTH_OPT_IN_TOOLS`` is set (:data:`OPT_IN_ENV`), the same switch as
``OLMOEARTH_RUN_PYTHON`` (:func:`olmoearth_agent.tools.registry.opted_in`).
``OLMOEARTH_READ_ONLY`` (:data:`READ_ONLY_ENV`, the same values) declares this
run's Studio key read-only: the tools that write to Studio
(:data:`STUDIO_WRITE_TOOLS`) are left out, and the capability card says why.

As each skill is implemented, flip its ``status`` to ``"implemented"``
and add its ``build_*_tools`` bundle to :func:`build_default_registry`.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Literal

from olmoearth_agent.harness.capabilities import READ_ONLY_STUDIO
from olmoearth_agent.provenance.tools import build_provenance_tools
from olmoearth_agent.tools.automate import build_automate_tools
from olmoearth_agent.tools.baseline_compare import build_baseline_compare_tools
from olmoearth_agent.tools.certificate import build_certificate_tools
from olmoearth_agent.tools.change_detect import build_change_detect_tools
from olmoearth_agent.tools.cloud_mask_audit import build_cloud_mask_audit_tools
from olmoearth_agent.tools.compare import build_compare_tools
from olmoearth_agent.tools.estimation import build_estimation_tools
from olmoearth_agent.tools.evaluate import build_evaluate_tools
from olmoearth_agent.tools.export import build_export_tools
from olmoearth_agent.tools.litsearch import build_litsearch_tools
from olmoearth_agent.tools.memory import build_memory_tools
from olmoearth_agent.tools.narrative import build_narrative_tools
from olmoearth_agent.tools.negative_sampler import build_negative_sampler_tools
from olmoearth_agent.tools.predict import build_predict_tools
from olmoearth_agent.tools.qgis import build_qgis_tools
from olmoearth_agent.tools.registry import RegisteredTool, ToolRegistry, opted_in
from olmoearth_agent.tools.review_set import build_review_set_tools
from olmoearth_agent.tools.rslearn import build_rslearn_tools
from olmoearth_agent.tools.scores_file import build_scores_file_tools
from olmoearth_agent.tools.similarity import build_similarity_tools
from olmoearth_agent.tools.skill_tools import build_skill_tools
from olmoearth_agent.tools.studio import build_studio_tools
from olmoearth_agent.tools.system import build_system_tools
from olmoearth_agent.tools.uncertainty import build_uncertainty_tools

SkillStatus = Literal["foundational", "implemented", "vendored", "planned"]

#: The one setting that registers the opt-in tools (``1``, ``true``, ``yes``
#: or ``on``; see :func:`olmoearth_agent.tools.registry.opted_in`).
OPT_IN_ENV = "OLMOEARTH_OPT_IN_TOOLS"

#: The setting that declares this run's Studio key read-only (the values of
#: :func:`~olmoearth_agent.tools.registry.opted_in`). Studio's keys can be
#: read-only, and ``/users/me`` does not show it, so the operator says so here.
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

    ``tools`` names every tool the skill contributes; ``opt_in`` maps those of
    them left out of the default registry to the one-line reason, and they
    are registered only when :data:`OPT_IN_ENV` is set.
    """

    number: int
    name: str
    category: Literal[
        "Foundational", "Prep", "Configure", "Run", "Analyze", "Integrate", "Report"
    ]
    status: SkillStatus
    summary: str
    tools: list[str] = field(default_factory=list)
    opt_in: dict[str, str] = field(default_factory=dict)

    @property
    def default_tools(self) -> list[str]:
        """The skill's tools the default registry holds (``tools`` less ``opt_in``)."""
        return [t for t in self.tools if t not in self.opt_in]

    @property
    def opt_in_only(self) -> bool:
        """True when every tool of the skill is opt-in, so none is on by default."""
        return bool(self.tools) and not self.default_tools


#: The full catalog. ``foundational`` = base Studio tools (not a SKILLS.md
#: entry); ``upstream`` = exists in 2imi9/OlmoEarth-Skills (vendor, don't
#: rebuild); ``planned`` = to build; ``implemented`` = wired in this repo.
SKILLS: list[SkillSpec] = [
    SkillSpec(
        0,
        "studio-core",
        "Foundational",
        "implemented",
        "Base Studio API tools every Run/Analyze skill builds on.",
        [
            "olmoearth_load_context",
            "olmoearth_create_project",
            "olmoearth_request_aoi",
            "olmoearth_get_prediction",
        ],
    ),
    # #1 unifies the former studio-upload + rslearn-config rows: they were two
    # catalog entries for the single vendored `olmoearth-data-prep` SKILL.md
    # package (loaded by SkillLoader). One package -> one entry.
    SkillSpec(
        1,
        "olmoearth-data-prep",
        "Prep",
        "vendored",
        "Labels -> Studio import (MIME/10K/multi-metric guards) AND rslearn "
        "dataset.json + Lightning YAML with a 7-criteria audit. The single "
        "vendored olmoearth-data-prep SKILL.md package.",
        ["olmoearth_load_skill"],
    ),
    SkillSpec(
        2,
        "olmoearth-studio-job-config",
        "Configure",
        "vendored",
        "Task description -> full Studio 'new model' wizard answers "
        "(name/type/foundation/label/training data/split/temporal/sources/area); "
        "14 presets + validator. Studio trains on Ai2 compute.",
        ["olmoearth_load_skill"],
    ),
    # #3 unifies the former embeddings + automate rows: both decide
    # embeddings-vs-fine-tune. The vendored SKILL.md is the guidance + notebook
    # generator (load via olmoearth_load_skill); olmoearth_automate is the
    # in-repo one-call version (decide + propose a config + HF introspection),
    # which reuses the same decision table, and is opt-in.
    SkillSpec(
        3,
        "olmoearth-embeddings",
        "Configure",
        "vendored",
        "Embeddings-vs-fine-tune decision: the vendored guidance + runnable "
        "notebook. Ai2's precomputed OlmoEarth v1.3 embeddings "
        "(source.coop/olmoearth/olmoearth-v1-3-embeddings; one vector per 10 m "
        "pixel per year) replace computing them yourself for an annual task in "
        "an area and year they cover; a task that needs dates within a year "
        "still needs the notebook. The one-call olmoearth_automate is opt-in.",
        ["olmoearth_load_skill", "olmoearth_automate"],
        opt_in={
            "olmoearth_automate": "for an annual task, Ai2's v1.3 embeddings and "
            "its Explorer replace computing embeddings",
        },
    ),
    SkillSpec(
        4,
        "olmoearth-predict",
        "Run",
        "implemented",
        "Run loop: search predictions (find model_id), submit, poll, "
        "fetch results (tile URLs), sample the model output at a point, and "
        "compare results quantitatively with one tool (grid-sampled, no "
        "ground truth): a pair of results, or a group of 2-6 models "
        "(pairwise matrix + ensemble consensus).",
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
    # #5: change over time is olmoearth_compare_results' series mode, which
    # superseded the caller-series trajectory tool olmoearth_change_detect
    # (opt-in). The JEPA latent-change detector in the separate torch repo
    # 2imi9/olmoearth-jepa-change is not connected: no tool here calls it.
    SkillSpec(
        5,
        "olmoearth-change-detection",
        "Run",
        "implemented",
        "Change over time: one model's dated Studio results traced as a series "
        "(olmoearth_compare_results, mode='series': date-ordered, "
        "legend-calibrated shifts; estimate movement, not verified ground "
        "change). The trajectory of the caller's own dated values "
        "(olmoearth_change_detect, refuses fewer than 3 dates) is opt-in.",
        ["olmoearth_change_detect", "olmoearth_compare_results"],
        opt_in={
            "olmoearth_change_detect": "superseded by olmoearth_compare_results, "
            "mode='series'",
        },
    ),
    # #6 names a winner from point metrics, with no design, interval or test;
    # which of two maps is more accurate is #18's two-map route
    # (olmoearth_plan_label_sample, design='disagreement', then
    # olmoearth_estimate_map_error).
    SkillSpec(
        6,
        "olmoearth-baseline-compare",
        "Run",
        "implemented",
        "Opt-in. OlmoEarth vs a baseline foundation model (e.g. AlphaEarth) "
        "side-by-side on transfer regions; bring-your-own exported "
        "embeddings/predictions (no live GEE connection). Its winner is a point "
        "metric with no test; #18's two-map route says which map is more "
        "accurate.",
        ["olmoearth_baseline_compare"],
        opt_in={
            "olmoearth_baseline_compare": "names a winner from point metrics, with "
            "no design, interval or test",
        },
    ),
    SkillSpec(
        7,
        "olmoearth-evaluate",
        "Analyze",
        "implemented",
        "Classification metrics against the user's labels; the "
        "random-vs-spatial CV inflation check and NNDM-LOO cross-validation "
        "(CAST port) are opt-in.",
        [
            "olmoearth_cv_inflation_check",
            "olmoearth_classification_metrics",
            "olmoearth_nndm_cv",
        ],
        opt_in={
            "olmoearth_cv_inflation_check": "a training-side check, run outside Studio",
            "olmoearth_nndm_cv": "a training-side check, run outside Studio",
        },
    ),
    SkillSpec(
        8,
        "olmoearth-similarity",
        "Analyze",
        "implemented",
        "Opt-in. Exact top-K kNN over supplied embeddings (e.g. OlmoEarth "
        "Base); geographic-prior warning.",
        ["olmoearth_similarity_search"],
        opt_in={
            "olmoearth_similarity_search": "Ai2's Explorer and AlphaEarth in "
            "Earth Engine search every pixel",
        },
    ),
    SkillSpec(
        9,
        "olmoearth-uncertainty",
        "Analyze",
        "implemented",
        "Ensemble-disagreement confidence (across distinct results: "
        "olmoearth_compare_results, mode='ensemble'), the signal that works "
        "when Studio gives you hard classes only; when per-class scores "
        "exist, #18 ranks errors better (measured). The Meyer-Pebesma "
        "Area-of-Applicability OOD flag is opt-in.",
        ["olmoearth_area_of_applicability", "olmoearth_compare_results"],
        opt_in={
            "olmoearth_area_of_applicability": "no tool produces its input, and "
            "it was measured not to rank errors better than the margin",
        },
    ),
    SkillSpec(
        10,
        "olmoearth-cloud-mask-audit",
        "Analyze",
        "implemented",
        "Opt-in. CFMask/s2cloudless/Sen2Cor/MAJA ensemble disagreement.",
        ["olmoearth_cloud_mask_audit"],
        opt_in={
            "olmoearth_cloud_mask_audit": "no tool produces its input; "
            "inferencex's condition layer records cloud where a run has SCL",
        },
    ),
    SkillSpec(
        11,
        "olmoearth-qgis-bridge",
        "Integrate",
        "implemented",
        "Tile URLs -> a QGIS .qlr layer-definition + a GDAL_WMS/XYZ descriptor "
        "+ XYZ URLs + an OGC SLD ramp style + a local gdal_translate COG recipe "
        "(the agent is GDAL-free; key never embedded).",
        ["olmoearth_qgis_bridge"],
    ),
    # #12 reframed from "wire external MCPs" to "export our own Studio
    # data, grouped" (more useful, self-contained). See CHANGELOG.
    SkillSpec(
        12,
        "olmoearth-data-export",
        "Integrate",
        "implemented",
        "Export Studio projects + predictions grouped (by project or "
        "status) to JSON files.",
        ["olmoearth_export_data"],
    ),
    SkillSpec(
        13,
        "olmoearth-provenance",
        "Report",
        "implemented",
        "Manifest wrapper over every tool call; emits replay script; lists "
        "each verification call's design and seed.",
        ["olmoearth_provenance_summary"],
    ),
    SkillSpec(
        14,
        "olmoearth-case-narrative",
        "Report",
        "implemented",
        "Stakeholder Markdown writeup with tile URLs + freshness gate; states "
        "the accuracy the request's verification tools measured, or that none "
        "was.",
        ["olmoearth_case_narrative"],
    ),
    SkillSpec(
        15,
        "olmoearth-litsearch",
        "Report",
        "implemented",
        "Opt-in. arXiv + OpenAlex literature search + DOI/arXiv-id resolution "
        "to ground citations (key-free; deduped across sources).",
        ["olmoearth_litsearch"],
        opt_in={
            "olmoearth_litsearch": "it costs a core slot every turn; it is the "
            "agent's own route to Ai2's Asta, not a copy of it",
        },
    ),
    SkillSpec(
        16,
        "olmoearth-negative-sampler",
        "Prep",
        "implemented",
        "Opt-in. Presence-only labels -> trainable set: generate a buffered, "
        "spatially-thinned (optionally embedding-dissimilar) negative class "
        "so the data-prep audit's negative-class check passes.",
        ["olmoearth_negative_sampler"],
        opt_in={
            "olmoearth_negative_sampler": "prepares labels for model training "
            "outside Studio",
        },
    ),
    # #17 promotes the vendored olmoearth-rslearn SKILL.md (it OPERATES rslearn,
    # the data + training engine under OlmoEarth) to a catalog row, and adds
    # four in-repo TORCH-FREE tools so a scientist who doesn't know rslearn can
    # be guided + checked (recommend, validate, compose, diagnose). The tools
    # are opt-in (training outside Studio); the SKILL.md loads by default.
    # Mirrors #3's vendored-SKILL.md + in-repo-tool pattern.
    SkillSpec(
        17,
        "olmoearth-rslearn",
        "Configure",
        "vendored",
        "Operate rslearn (the data + training engine under OlmoEarth): the vendored "
        "SKILL.md guides add_windows -> prepare -> ingest -> materialize -> model "
        "fit/predict, as commands the user runs. Four opt-in torch-free tools for "
        "non-experts -- recommend "
        "a full setup from a plain-language research goal (+ a pre/mid/post fusion "
        "strategy for multiple modalities), validate a config (encoder/decoder/head "
        "shapes, task<->label-type, bands) before training, compose a valid finetune "
        "model.yaml, and diagnose a failing prepare/ingest/materialize run.",
        [
            "olmoearth_load_skill",
            "olmoearth_rslearn_recommend",
            "olmoearth_rslearn_validate",
            "olmoearth_rslearn_compose",
            "olmoearth_rslearn_diagnose",
        ],
        opt_in={
            "olmoearth_rslearn_recommend": "model training outside Studio",
            "olmoearth_rslearn_validate": "model training outside Studio",
            "olmoearth_rslearn_compose": "model training outside Studio",
            "olmoearth_rslearn_diagnose": "model training outside Studio",
        },
    ),
    # #18 closes the gap #9 cannot: ranking WHICH windows are wrong, from the
    # model's own top-1-minus-top-2 margin. Studio's pixel-value returns only
    # raw_value/classification (no probabilities, no logits), which is exactly
    # why #9 reached for ensemble disagreement -- with hard classes it is the
    # only signal available. Given real scores the ordering flips: upstream
    # (2imi9/olmoearth_inferenceX-record) measured ensemble disagreement, cross-model
    # disagreement, two-view disagreement and feature-space typicality against
    # the margin on seven expert-labelled testbeds and none of them won. So the
    # two skills are complementary, not rivals: #9 for hard-class Studio
    # results and for the OOD regime, #18 whenever per-class scores exist.
    SkillSpec(
        18,
        "olmoearth-review-set",
        "Analyze",
        "implemented",
        "Label-free review set from the model's own top-1-minus-top-2 margin: "
        "which windows to check first at a budget (from caller scores, from a "
        "direct model run's files, score_area.py's scores raster and manifest "
        "or what oe-inferencex from-olmoearth writes, or from a Studio "
        "result whose band is a binary score in [0, 1], sampled on a grid), "
        "boundary-first ordering, the attainable-ceiling arithmetic, and "
        "(opt-in) a grader for any candidate audit rule. Then how wrong the "
        "map is: a labelled sample drawn by a design, its error rate with the "
        "interval valid at every sample size first, then the one that design "
        "earns, per-class accuracy and a certified zone "
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
            "olmoearth_compare_review",
            "olmoearth_grade_review_rule",
            "olmoearth_review_budget_ceiling",
            "olmoearth_plan_label_sample",
            "olmoearth_estimate_map_error",
            "olmoearth_certify_zone",
            "olmoearth_write_certificate",
            "olmoearth_verify_certificate",
        ],
        opt_in={
            "olmoearth_grade_review_rule": "needs per-unit error labels a Studio "
            "user rarely has",
        },
    ),
]


#: Every opt-in tool, with the one-line reason it left the default registry.
OPT_IN_TOOLS: dict[str, str] = {
    tool: reason for skill in SKILLS for tool, reason in skill.opt_in.items()
}

#: The one #7 tool sent on every turn; the spatial-CV tools are opt-in.
_EVALUATE_CORE = frozenset({"olmoearth_classification_metrics"})


def skills_by_status(status: SkillStatus) -> list[SkillSpec]:
    """All catalog entries with the given build status."""
    return [s for s in SKILLS if s.status == status]


def build_default_registry() -> ToolRegistry:
    """Assemble a :class:`ToolRegistry` from every implemented skill bundle.

    Core bundles are sent to the LLM on every turn. Deferred bundles are
    registered under their skill's name (``group=``) and sent only once that
    skill is loaded (``olmoearth_load_skill``) or forced from the web UI; see
    :mod:`olmoearth_agent.tools.registry`. A new skill adds its
    ``build_*_tools()`` bundle here and flips its catalog ``status``.

    The tools in :data:`OPT_IN_TOOLS` are left out unless :data:`OPT_IN_ENV`
    (``OLMOEARTH_OPT_IN_TOOLS``) is on; on, each is registered core or
    deferred under its skill, as the comments below say. While
    :data:`READ_ONLY_ENV` (``OLMOEARTH_READ_ONLY``) is on, the tools in
    :data:`STUDIO_WRITE_TOOLS` are left out too, and the registry's ``cannot``
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
    # The tools left out: the opt-in ones unless opted in, and with a
    # read-only Studio key the ones that write to Studio.
    skip: set[str] = set() if opted_in(OPT_IN_ENV) else set(OPT_IN_TOOLS)
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
    add(build_predict_tools())
    # How Studio results differ (#4, #5 series, #9 ensemble): one tool.
    add(build_compare_tools())
    # Opt-in: superseded by compare_results' series mode.
    add(build_change_detect_tools())
    # Accuracy against labels stays core (the comparison and review tools
    # route to it); #7's spatial-CV tools are opt-in, deferred below.
    evaluate = build_evaluate_tools()
    add(t for t in evaluate if t.spec.name in _EVALUATE_CORE)
    # Label-free error ranking (skill #18); complements #9's ensemble/OOD
    # signals, which are what remain reachable when Studio yields only
    # hard classes. Its rule grader is opt-in.
    add(build_review_set_tools())
    # The scores provider (#18): a direct model run's raster + manifest as the
    # review set's and the estimation tools' input. Core, not deferred: it is
    # the entry point of a path whose other tools are all core, and deferring
    # it would put a load_skill turn before the first useful call.
    add(build_scores_file_tools())
    # How wrong is the map (skill #18, second half): design-based estimation
    # through the optional inferencex extra; the tools say so when it is absent.
    add(build_estimation_tools())
    # A certified zone of a direct run's map packed into a certificate, and its
    # check (#18): the same extra, no other dependency, so core like the rest.
    add(build_certificate_tools())
    add(build_narrative_tools())
    # Opt-in: literature search (#15).
    add(build_litsearch_tools())
    add(build_export_tools())
    add(build_qgis_tools())
    add(build_provenance_tools())
    # --- Deferred: sent once their skill is loaded. ---
    # Self-run training (#3, #17), opt-in: only when the user trains the
    # model themselves; Studio trains on Ai2's compute (soul.md).
    add(build_automate_tools(), group="olmoearth-embeddings")
    add(build_rslearn_tools(), group="olmoearth-rslearn")
    # Caller-array tools (the opt-in #6, #7 spatial CV, #8, #9 AOA, #10):
    # their inputs are arrays the user supplies inline, which no agent tool
    # produces. With the opt-in tools off, no group is deferred.
    add(build_baseline_compare_tools(), group="olmoearth-baseline-compare")
    add(
        (t for t in evaluate if t.spec.name not in _EVALUATE_CORE),
        group="olmoearth-evaluate",
    )
    add(build_similarity_tools(), group="olmoearth-similarity")
    add(build_uncertainty_tools(), group="olmoearth-uncertainty")
    add(build_cloud_mask_audit_tools(), group="olmoearth-cloud-mask-audit")
    # Label preparation from the user's own presence-only file (#16), opt-in.
    add(build_negative_sampler_tools(), group="olmoearth-negative-sampler")
    # Skill loading, after the deferred bundles so its description lists them.
    add(build_skill_tools(registry=registry))
    # Cross-thread preference memory (remember/forget); core, not a skill.
    add(build_memory_tools())
    # Opt-in code execution (OLMOEARTH_RUN_PYTHON); an empty bundle otherwise.
    add(build_system_tools())
    return registry
