# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Skill catalog manifest, the harness's view of all 17 skills.

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
from olmoearth_agent.tools.evaluate import build_evaluate_tools
from olmoearth_agent.tools.export import build_export_tools
from olmoearth_agent.tools.litsearch import build_litsearch_tools
from olmoearth_agent.tools.narrative import build_narrative_tools
from olmoearth_agent.tools.negative_sampler import build_negative_sampler_tools
from olmoearth_agent.tools.predict import build_predict_tools
from olmoearth_agent.tools.qgis import build_qgis_tools
from olmoearth_agent.tools.registry import ToolRegistry
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
            "olmoearth_search_projects",
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
        "Task description -> Studio wizard answers; 14 presets + validator.",
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
        "fetch results (tile URLs), and compare two results quantitatively "
        "(grid-sampled model-vs-model agreement, no ground truth).",
        [
            "olmoearth_search_predictions",
            "olmoearth_submit_prediction",
            "olmoearth_get_prediction",
            "olmoearth_fetch_results",
            "olmoearth_get_prediction_result",
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
        "trajectory diff (refuses naive 2-date), and an out-of-process JEPA "
        "latent-prediction residual detector (separate torch repo).",
        ["olmoearth_change_detect"],
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
        "Confidence + Meyer-Pebesma Area-of-Applicability OOD flag.",
        ["olmoearth_area_of_applicability"],
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
        "Tile URLs -> QGIS XYZ URLs + OGC SLD ramp style + load "
        "instructions. (COG export follows.)",
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
        ["olmoearth_litsearch", "olmoearth_litsearch_resolve"],
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
        "fit/predict, plus two in-repo torch-free tools for non-experts -- recommend "
        "a full setup from a plain-language research goal, and validate a config "
        "(encoder/decoder/head shapes, task<->label-type, bands) before training.",
        [
            "olmoearth_load_skill",
            "olmoearth_rslearn_recommend",
            "olmoearth_rslearn_validate",
            "olmoearth_rslearn_compose",
            "olmoearth_rslearn_diagnose",
        ],
    ),
]


def skills_by_status(status: SkillStatus) -> list[SkillSpec]:
    """All catalog entries with the given build status."""
    return [s for s in SKILLS if s.status == status]


def build_default_registry() -> ToolRegistry:
    """Assemble a :class:`ToolRegistry` from the implemented skill bundles.

    Today that is only the foundational Studio tools. As skills land,
    add their ``build_*_tools()`` bundle here (and flip the catalog
    ``status`` to ``"implemented"``).
    """
    registry = ToolRegistry()
    registry.register_all(build_studio_tools())
    registry.register_all(build_predict_tools())
    registry.register_all(build_baseline_compare_tools())
    registry.register_all(build_change_detect_tools())
    registry.register_all(build_cloud_mask_audit_tools())
    registry.register_all(build_evaluate_tools())
    registry.register_all(build_uncertainty_tools())
    registry.register_all(build_similarity_tools())
    registry.register_all(build_narrative_tools())
    registry.register_all(build_negative_sampler_tools())
    registry.register_all(build_litsearch_tools())
    registry.register_all(build_automate_tools())
    registry.register_all(build_rslearn_tools())
    registry.register_all(build_export_tools())
    registry.register_all(build_qgis_tools())
    registry.register_all(build_provenance_tools())
    registry.register_all(build_skill_tools())
    # Opt-in code execution (OLMOEARTH_RUN_PYTHON); an empty bundle otherwise.
    registry.register_all(build_system_tools())
    return registry
