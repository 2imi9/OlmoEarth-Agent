# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Skill catalog manifest — the harness's view of all 16 skills.

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
from olmoearth_agent.tools.baseline_compare import build_baseline_compare_tools
from olmoearth_agent.tools.change_detect import build_change_detect_tools
from olmoearth_agent.tools.cloud_mask_audit import build_cloud_mask_audit_tools
from olmoearth_agent.tools.evaluate import build_evaluate_tools
from olmoearth_agent.tools.export import build_export_tools
from olmoearth_agent.tools.narrative import build_narrative_tools
from olmoearth_agent.tools.predict import build_predict_tools
from olmoearth_agent.tools.qgis import build_qgis_tools
from olmoearth_agent.tools.registry import ToolRegistry
from olmoearth_agent.tools.similarity import build_similarity_tools
from olmoearth_agent.tools.skill_tools import build_skill_tools
from olmoearth_agent.tools.studio import build_studio_tools
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
        0, "studio-core", "Foundational", "implemented",
        "Base Studio API tools every Run/Analyze skill builds on.",
        ["olmoearth_load_context", "olmoearth_search_projects",
         "olmoearth_create_project", "olmoearth_get_prediction"],
    ),
    # #1 + #2 are unified upstream as the `olmoearth-data-prep` SKILL.md
    # package; vendored via git submodule + loaded by SkillLoader.
    SkillSpec(1, "olmoearth-studio-upload", "Prep", "vendored",
              "Labels -> Studio import (MIME/10K/multi-metric guards). "
              "In upstream olmoearth-data-prep SKILL.md.",
              ["olmoearth_load_skill"]),
    SkillSpec(2, "olmoearth-rslearn-config", "Prep", "vendored",
              "Labels -> rslearn dataset.json + Lightning YAML; 7-criteria "
              "audit. In upstream olmoearth-data-prep SKILL.md.",
              ["olmoearth_load_skill"]),
    SkillSpec(3, "olmoearth-studio-job-config", "Configure", "vendored",
              "Task description -> Studio wizard answers; 14 presets + validator.",
              ["olmoearth_load_skill"]),
    SkillSpec(4, "olmoearth-embeddings", "Configure", "vendored",
              "Embeddings-vs-fine-tune decision + runnable notebook.",
              ["olmoearth_load_skill"]),
    SkillSpec(5, "olmoearth-predict", "Run", "implemented",
              "Run loop: search predictions (find model_id), submit, poll, "
              "fetch results (tile URLs). pixel-value/features follow.",
              ["olmoearth_search_predictions", "olmoearth_submit_prediction",
               "olmoearth_get_prediction", "olmoearth_fetch_results",
               "olmoearth_get_prediction_result"]),
    SkillSpec(6, "olmoearth-change-detect", "Run", "implemented",
              "Multi-date (>=3) trajectory diff; refuses naive 2-date.",
              ["olmoearth_change_detect"]),
    SkillSpec(7, "olmoearth-baseline-compare", "Run", "implemented",
              "OlmoEarth vs AlphaEarth on transfer regions (bring-your-own "
              "exported GEE Satellite Embedding data; no live GEE connection).",
              ["olmoearth_baseline_compare"]),
    SkillSpec(8, "olmoearth-evaluate", "Analyze", "implemented",
              "Random-vs-spatial CV inflation check + classification "
              "metrics. NNDM-LOO follows.",
              ["olmoearth_cv_inflation_check", "olmoearth_classification_metrics"]),
    SkillSpec(9, "olmoearth-similarity", "Analyze", "implemented",
              "FAISS over OlmoEarth Base embeddings; geographic-prior warning.",
              ["olmoearth_similarity_search"]),
    SkillSpec(10, "olmoearth-uncertainty", "Analyze", "implemented",
              "Confidence + Meyer-Pebesma Area-of-Applicability OOD flag.",
              ["olmoearth_area_of_applicability"]),
    SkillSpec(11, "olmoearth-cloud-mask-audit", "Analyze", "implemented",
              "CFMask/s2cloudless/Sen2Cor/MAJA ensemble disagreement.",
              ["olmoearth_cloud_mask_audit"]),
    SkillSpec(12, "olmoearth-qgis-bridge", "Integrate", "implemented",
              "Tile URLs -> QGIS XYZ URLs + OGC SLD ramp style + load "
              "instructions. (COG export follows.)",
              ["olmoearth_qgis_bridge"]),
    # #13 reframed from "wire external MCPs" to "export our own Studio
    # data, grouped" (more useful, self-contained). See CHANGELOG.
    SkillSpec(13, "olmoearth-data-export", "Integrate", "implemented",
              "Export Studio projects + predictions grouped (by project or "
              "status) to JSON files.",
              ["olmoearth_export_data"]),
    SkillSpec(14, "olmoearth-provenance", "Report", "implemented",
              "Manifest wrapper over every tool call; emits replay script.",
              ["olmoearth_provenance_summary"]),
    SkillSpec(15, "olmoearth-case-narrative", "Report", "implemented",
              "Stakeholder Markdown writeup with tile URLs + freshness gate.",
              ["olmoearth_case_narrative"]),
    SkillSpec(16, "roger-annotation-bridge", "Prep", "planned",
              "Roger Studio annotations -> Studio labelset (schema UNVERIFIED).",
              ["roger_annotation_bridge"]),
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
    registry.register_all(build_export_tools())
    registry.register_all(build_qgis_tools())
    registry.register_all(build_provenance_tools())
    registry.register_all(build_skill_tools())
    return registry
