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
from olmoearth_agent.tools.registry import ToolRegistry
from olmoearth_agent.tools.studio import build_studio_tools

SkillStatus = Literal["foundational", "implemented", "upstream", "planned"]


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
    SkillSpec(1, "olmoearth-studio-upload", "Prep", "upstream",
              "Labels -> Studio import (MIME/10K/multi-metric guards).",
              ["olmoearth_upload_labels"]),
    SkillSpec(2, "olmoearth-rslearn-config", "Prep", "upstream",
              "Labels -> rslearn dataset.json + Lightning YAML; 7-criteria audit.",
              ["olmoearth_write_rslearn_config"]),
    SkillSpec(3, "olmoearth-studio-job-config", "Configure", "upstream",
              "Task description -> Studio wizard answers; 14 presets + validator.",
              ["olmoearth_studio_job_validate"]),
    SkillSpec(4, "olmoearth-embeddings", "Configure", "upstream",
              "Embeddings-vs-fine-tune decision + runnable notebook.",
              ["olmoearth_fetch_embedding"]),
    SkillSpec(5, "olmoearth-predict", "Run", "planned",
              "Core run primitive: submit/poll/pixel-value/features/files.",
              ["olmoearth_submit_prediction", "olmoearth_poll_prediction",
               "olmoearth_fetch_results", "olmoearth_pixel_value",
               "olmoearth_features_search"]),
    SkillSpec(6, "olmoearth-change-detect", "Run", "planned",
              "Multi-date (>=3) trajectory diff; refuses naive 2-date.",
              ["olmoearth_change_detect"]),
    SkillSpec(7, "olmoearth-baseline-compare", "Run", "planned",
              "OlmoEarth vs AlphaEarth on transfer regions (needs GEE MCP).",
              ["olmoearth_baseline_compare"]),
    SkillSpec(8, "olmoearth-evaluate", "Analyze", "planned",
              "Spatial-block CV + NNDM-LOO over prediction-results.",
              ["olmoearth_spatial_block_cv", "olmoearth_nndm_loo"]),
    SkillSpec(9, "olmoearth-similarity", "Analyze", "planned",
              "FAISS over OlmoEarth Base embeddings; geographic-prior warning.",
              ["olmoearth_similarity_search"]),
    SkillSpec(10, "olmoearth-uncertainty", "Analyze", "planned",
              "Confidence + Meyer-Pebesma Area-of-Applicability OOD flag.",
              ["olmoearth_area_of_applicability"]),
    SkillSpec(11, "olmoearth-cloud-mask-audit", "Analyze", "planned",
              "CFMask/s2cloudless/Sen2Cor/MAJA ensemble disagreement.",
              ["olmoearth_cloud_mask_audit"]),
    SkillSpec(12, "olmoearth-qgis-bridge", "Integrate", "planned",
              "Tile URLs -> QGIS WMTS + COG with sidecar uncertainty.",
              ["olmoearth_qgis_bridge"]),
    SkillSpec(13, "olmoearth-external-data", "Integrate", "planned",
              "GEE/PC/OSM/USGS/NOAA into an AOI (needs user-connected MCPs).",
              ["olmoearth_external_data"]),
    SkillSpec(14, "olmoearth-provenance", "Report", "implemented",
              "Manifest wrapper over every tool call; emits replay script.",
              ["olmoearth_provenance_summary"]),
    SkillSpec(15, "olmoearth-case-narrative", "Report", "planned",
              "Stakeholder writeup with live tiles + freshness gate.",
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
    registry.register_all(build_provenance_tools())
    return registry
