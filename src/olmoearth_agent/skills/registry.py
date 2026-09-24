# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Skill catalog manifest, the harness's view of all 18 skills.

This is the structural "slot" for every skill in ``SKILLS.md``. Each
:class:`SkillSpec` records the skill's category, build status, and the
tools it contributes. :func:`build_default_registry` assembles the
tool bundles of the currently-implemented skills into a
:class:`ToolRegistry` the lead agent can use.

As each skill is implemented, flip its ``status`` to ``"implemented"``
and add its ``build_*_tools`` bundle to :func:`build_default_registry`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from olmoearth_agent.provenance.tools import build_provenance_tools
from olmoearth_agent.tools.automate import build_automate_tools
from olmoearth_agent.tools.baseline_compare import build_baseline_compare_tools
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
from olmoearth_agent.tools.registry import ToolRegistry
from olmoearth_agent.tools.review_set import build_review_set_tools
from olmoearth_agent.tools.rslearn import build_rslearn_tools
from olmoearth_agent.tools.similarity import build_similarity_tools
from olmoearth_agent.tools.skill_tools import build_skill_tools
from olmoearth_agent.tools.studio import build_studio_tools
from olmoearth_agent.tools.system import build_system_tools
from olmoearth_agent.tools.uncertainty import build_uncertainty_tools

SkillStatus = Literal["foundational", "implemented", "vendored", "planned"]


@dataclass(frozen=True)
class SkillSpec:
    """One row of the skill catalog (see ``SKILLS.md``)."""

    number: int
    name: str
    category: Literal[
        "Foundational", "Prep", "Configure", "Run", "Analyze", "Integrate", "Report"
    ]
    status: SkillStatus
    summary: str
    tools: list[str] = field(default_factory=list)


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
    # which reuses the same decision table.
    SkillSpec(
        3,
        "olmoearth-embeddings",
        "Configure",
        "vendored",
        "Embeddings-vs-fine-tune decision: the vendored guidance + runnable "
        "notebook, plus the in-repo one-call olmoearth_automate (decide + "
        "config + optional HF-dataset introspection) reusing the same table.",
        ["olmoearth_load_skill", "olmoearth_automate"],
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
    # #5 unifies change-detect + the JEPA latent-change skill: both are change
    # detection, two engines. Engine A = the in-process Studio multi-date
    # trajectory diff (olmoearth_change_detect). Engine B = the out-of-process
    # JEPA latent-prediction residual detector in the separate torch repo
    # 2imi9/olmoearth-jepa-change (no heavy deps here; invoked out-of-process).
    SkillSpec(
        5,
        "olmoearth-change-detection",
        "Run",
        "implemented",
        "Change detection, two engines: in-process Studio multi-date (>=3) "
        "trajectory diff (refuses naive 2-date) plus date-ordered, "
        "legend-calibrated shift tracing across dated results "
        "(olmoearth_compare_results, mode='series'), and an out-of-process "
        "JEPA latent-prediction residual detector (separate torch repo).",
        ["olmoearth_change_detect", "olmoearth_compare_results"],
    ),
    SkillSpec(
        6,
        "olmoearth-baseline-compare",
        "Run",
        "implemented",
        "OlmoEarth vs a baseline foundation model (e.g. AlphaEarth) "
        "side-by-side on transfer regions; bring-your-own exported "
        "embeddings/predictions (no live GEE connection).",
        ["olmoearth_baseline_compare"],
    ),
    SkillSpec(
        7,
        "olmoearth-evaluate",
        "Analyze",
        "implemented",
        "Random-vs-spatial CV inflation check + classification metrics + "
        "NNDM-LOO cross-validation (CAST port) for an unbiased map-accuracy estimate.",
        [
            "olmoearth_cv_inflation_check",
            "olmoearth_classification_metrics",
            "olmoearth_nndm_cv",
        ],
    ),
    SkillSpec(
        8,
        "olmoearth-similarity",
        "Analyze",
        "implemented",
        "Exact top-K kNN over supplied embeddings (e.g. OlmoEarth Base; "
        "FAISS = scale-up follow-up); geographic-prior warning.",
        ["olmoearth_similarity_search"],
    ),
    SkillSpec(
        9,
        "olmoearth-uncertainty",
        "Analyze",
        "implemented",
        "Ensemble-disagreement confidence (across distinct results: "
        "olmoearth_compare_results, mode='ensemble') + Meyer-Pebesma "
        "Area-of-Applicability OOD flag. The signals that work when Studio "
        "gives you hard classes only; when per-class scores exist, #18 ranks "
        "errors better (measured).",
        ["olmoearth_area_of_applicability", "olmoearth_compare_results"],
    ),
    SkillSpec(
        10,
        "olmoearth-cloud-mask-audit",
        "Analyze",
        "implemented",
        "CFMask/s2cloudless/Sen2Cor/MAJA ensemble disagreement.",
        ["olmoearth_cloud_mask_audit"],
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
        "Manifest wrapper over every tool call; emits replay script.",
        ["olmoearth_provenance_summary"],
    ),
    SkillSpec(
        14,
        "olmoearth-case-narrative",
        "Report",
        "implemented",
        "Stakeholder Markdown writeup with tile URLs + freshness gate.",
        ["olmoearth_case_narrative"],
    ),
    SkillSpec(
        15,
        "olmoearth-litsearch",
        "Report",
        "implemented",
        "arXiv + OpenAlex literature search + DOI/arXiv-id resolution to "
        "ground citations (key-free; deduped across sources).",
        ["olmoearth_litsearch"],
    ),
    SkillSpec(
        16,
        "olmoearth-negative-sampler",
        "Prep",
        "implemented",
        "Presence-only labels -> trainable set: generate a buffered, "
        "spatially-thinned (optionally embedding-dissimilar) negative class "
        "so the data-prep audit's negative-class check passes.",
        ["olmoearth_negative_sampler"],
    ),
    # #17 promotes the vendored olmoearth-rslearn SKILL.md (it OPERATES rslearn,
    # the data + training engine under OlmoEarth) to a catalog row, and adds two
    # in-repo TORCH-FREE tools so a scientist who doesn't know rslearn can be
    # guided + checked: olmoearth_rslearn_recommend (goal -> an explained setup)
    # and olmoearth_rslearn_validate (catch config errors before a training run).
    # Mirrors #3's vendored-SKILL.md + in-repo-tool pattern.
    SkillSpec(
        17,
        "olmoearth-rslearn",
        "Configure",
        "vendored",
        "Operate rslearn (the data + training engine under OlmoEarth): the vendored "
        "SKILL.md runs add_windows -> prepare -> ingest -> materialize -> model "
        "fit/predict, plus four in-repo torch-free tools for non-experts -- recommend "
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
    ),
    # #18 closes the gap #9 cannot: ranking WHICH windows are wrong, from the
    # model's own top-1-minus-top-2 margin. Studio's pixel-value returns only
    # raw_value/classification (no probabilities, no logits), which is exactly
    # why #9 reached for ensemble disagreement -- with hard classes it is the
    # only signal available. Given real scores the ordering flips: upstream
    # (2imi9/olmoearth_inferenceX) measured ensemble disagreement, cross-model
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
        "which windows to check first at a budget (from caller scores, or from "
        "a Studio result whose band is a binary score in [0, 1], sampled on a "
        "grid), boundary-first ordering, the attainable-ceiling arithmetic, and "
        "a grader for any candidate audit rule. Then how wrong the map is: a "
        "labelled sample drawn by a design, its error rate with the interval "
        "that design earns, per-class accuracy and a certified zone (the "
        "optional olmoearth-inferencex extra). A review set is not a sample.",
        [
            "olmoearth_review_set",
            "olmoearth_review_set_from_result",
            "olmoearth_compare_review",
            "olmoearth_grade_review_rule",
            "olmoearth_review_budget_ceiling",
            "olmoearth_plan_label_sample",
            "olmoearth_estimate_map_error",
            "olmoearth_certify_zone",
        ],
    ),
]


#: The one #7 tool sent on every turn; the spatial-CV tools are deferred.
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
    """
    registry = ToolRegistry()
    # --- Core: sent on every turn. ---
    registry.register_all(build_studio_tools())
    registry.register_all(build_predict_tools())
    # How Studio results differ (#4, #5 series, #9 ensemble): one tool.
    registry.register_all(build_compare_tools())
    registry.register_all(build_change_detect_tools())
    # Accuracy against labels stays core (the comparison and review tools
    # route to it); #7's spatial-CV tools are deferred below.
    evaluate = build_evaluate_tools()
    registry.register_all(t for t in evaluate if t.spec.name in _EVALUATE_CORE)
    # Label-free error ranking (skill #18); complements #9's ensemble/OOD
    # signals, which are what remain reachable when Studio yields only
    # hard classes.
    registry.register_all(build_review_set_tools())
    # How wrong is the map (skill #18, second half): design-based estimation
    # through the optional inferencex extra; the tools say so when it is absent.
    registry.register_all(build_estimation_tools())
    registry.register_all(build_narrative_tools())
    registry.register_all(build_litsearch_tools())
    registry.register_all(build_export_tools())
    registry.register_all(build_qgis_tools())
    registry.register_all(build_provenance_tools())
    # --- Deferred: sent once their skill is loaded. ---
    # Self-run training (#3, #17): only when the user trains the model
    # themselves; Studio trains on Ai2's compute (soul.md).
    registry.register_all(build_automate_tools(), group="olmoearth-embeddings")
    registry.register_all(build_rslearn_tools(), group="olmoearth-rslearn")
    # Caller-array tools (#6, #7 spatial CV, #8, #9 AOA, #10): their inputs
    # are arrays the user supplies inline, which no agent tool produces.
    registry.register_all(
        build_baseline_compare_tools(), group="olmoearth-baseline-compare"
    )
    registry.register_all(
        (t for t in evaluate if t.spec.name not in _EVALUATE_CORE),
        group="olmoearth-evaluate",
    )
    registry.register_all(build_similarity_tools(), group="olmoearth-similarity")
    registry.register_all(build_uncertainty_tools(), group="olmoearth-uncertainty")
    registry.register_all(
        build_cloud_mask_audit_tools(), group="olmoearth-cloud-mask-audit"
    )
    # Label preparation from the user's own presence-only file (#16).
    registry.register_all(
        build_negative_sampler_tools(), group="olmoearth-negative-sampler"
    )
    # Skill loading, after the deferred bundles so its description lists them.
    registry.register_all(build_skill_tools(registry=registry))
    # Cross-thread preference memory (remember/forget); core, not a skill.
    registry.register_all(build_memory_tools())
    # Opt-in code execution (OLMOEARTH_RUN_PYTHON); an empty bundle otherwise.
    registry.register_all(build_system_tools())
    return registry
