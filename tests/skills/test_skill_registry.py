# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Sanity tests for the skill catalog manifest."""

from __future__ import annotations

from olmoearth_agent.skills.registry import (
    SKILLS,
    build_default_registry,
    skills_by_status,
)


def test_catalog_covers_all_17_skills() -> None:
    numbered = [s for s in SKILLS if s.number >= 1]
    assert len(numbered) == 17
    assert {s.number for s in numbered} == set(range(1, 18))


def test_foundational_entry_is_implemented() -> None:
    implemented = skills_by_status("implemented")
    assert any(s.name == "studio-core" for s in implemented)


def test_vendored_skills_are_the_four_existing() -> None:
    vendored = {s.name for s in skills_by_status("vendored")}
    assert vendored == {
        "olmoearth-studio-upload",
        "olmoearth-rslearn-config",
        "olmoearth-studio-job-config",
        "olmoearth-embeddings",
    }


def test_provenance_skill_is_implemented() -> None:
    implemented = {s.name for s in skills_by_status("implemented")}
    assert "olmoearth-provenance" in implemented


def test_default_registry_exposes_foundational_and_provenance_tools() -> None:
    registry = build_default_registry()
    names = registry.names()
    for tool in (
        "olmoearth_load_context",
        "olmoearth_search_projects",
        "olmoearth_create_project",
        "olmoearth_get_prediction",
        "olmoearth_search_predictions",
        "olmoearth_submit_prediction",
        "olmoearth_fetch_results",
        "olmoearth_get_prediction_result",
        "olmoearth_baseline_compare",
        "olmoearth_change_detect",
        "olmoearth_cloud_mask_audit",
        "olmoearth_cv_inflation_check",
        "olmoearth_classification_metrics",
        "olmoearth_area_of_applicability",
        "olmoearth_similarity_search",
        "olmoearth_case_narrative",
        "olmoearth_export_data",
        "olmoearth_qgis_bridge",
        "olmoearth_provenance_summary",
        "olmoearth_litsearch",
        "olmoearth_litsearch_resolve",
        "olmoearth_automate",
        "olmoearth_list_skills",
        "olmoearth_load_skill",
    ):
        assert tool in names
    # every registered tool has a JSON-schema spec
    for spec in registry.specs():
        assert spec.parameters["type"] == "object"
