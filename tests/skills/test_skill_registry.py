# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Sanity tests for the skill catalog manifest."""

from __future__ import annotations

from olmoearth_agent.skills.registry import (
    SKILLS,
    build_default_registry,
    skills_by_status,
)


def test_catalog_covers_all_18_skills() -> None:
    numbered = [s for s in SKILLS if s.number >= 1]
    assert len(numbered) == 18
    assert {s.number for s in numbered} == set(range(1, 19))


def test_every_catalogued_tool_is_actually_registered() -> None:
    """A catalog row that names a tool the registry never builds is a lie."""
    registered = set(build_default_registry().names())
    for skill in SKILLS:
        if skill.status in ("implemented", "foundational"):
            for tool in skill.tools:
                if tool == "olmoearth_load_skill":
                    continue
                assert tool in registered, f"#{skill.number} {skill.name}: {tool}"


def test_foundational_entry_is_implemented() -> None:
    implemented = skills_by_status("implemented")
    assert any(s.name == "studio-core" for s in implemented)


def test_vendored_skills_match_the_four_packages() -> None:
    # Four SKILL.md packages = four dirs under src/olmoearth_agent/skills/packages/.
    vendored = {s.name for s in skills_by_status("vendored")}
    assert vendored == {
        "olmoearth-data-prep",
        "olmoearth-studio-job-config",
        "olmoearth-embeddings",
        "olmoearth-rslearn",
    }


def test_provenance_skill_is_implemented() -> None:
    implemented = {s.name for s in skills_by_status("implemented")}
    assert "olmoearth-provenance" in implemented


def test_default_registry_exposes_foundational_and_provenance_tools() -> None:
    registry = build_default_registry()
    names = registry.names()
    for tool in (
        "olmoearth_load_context",
        "olmoearth_create_project",
        "olmoearth_request_aoi",
        "olmoearth_get_prediction",
        "olmoearth_search_predictions",
        "olmoearth_submit_prediction",
        "olmoearth_fetch_results",
        "olmoearth_get_prediction_result",
        "olmoearth_pixel_value",
        "olmoearth_compare_results",
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
        "olmoearth_automate",
        "olmoearth_rslearn_recommend",
        "olmoearth_rslearn_validate",
        "olmoearth_rslearn_compose",
        "olmoearth_rslearn_diagnose",
        "olmoearth_negative_sampler",
        "olmoearth_load_skill",
    ):
        assert tool in names
    # every registered tool has a JSON-schema spec
    for spec in registry.specs():
        assert spec.parameters["type"] == "object"


def test_merged_and_removed_tools_are_gone() -> None:
    """One comparison tool; search_projects, litsearch_resolve, list_skills merged."""
    names = set(build_default_registry().names())
    for gone in (
        "olmoearth_compare_group",
        "olmoearth_trace_shifts",
        "olmoearth_ensemble_uncertainty",
        "olmoearth_search_projects",
        "olmoearth_litsearch_resolve",
        "olmoearth_list_skills",
    ):
        assert gone not in names
    for skill in SKILLS:
        for gone in ("olmoearth_compare_group", "olmoearth_trace_shifts"):
            assert gone not in skill.tools, f"#{skill.number} lists {gone}"


def test_deferred_groups_are_catalog_skills() -> None:
    """A group is loaded by its skill's name (load_skill, or the UI's forced slug)."""
    registry = build_default_registry()
    by_name = {s.name: s for s in SKILLS}
    for group, tools in registry.groups().items():
        assert group in by_name, group
        for tool in tools:
            assert tool in by_name[group].tools, f"{group} does not list {tool}"


def test_the_core_set_keeps_the_studio_review_estimation_and_compare_tools() -> None:
    registry = build_default_registry()
    core = {s.name for s in registry.active_specs()}
    for tool in (
        "olmoearth_load_context",
        "olmoearth_search_predictions",
        "olmoearth_submit_prediction",
        "olmoearth_get_prediction",
        "olmoearth_fetch_results",
        "olmoearth_pixel_value",
        "olmoearth_compare_results",
        "olmoearth_compare_review",
        "olmoearth_review_set",
        "olmoearth_review_set_from_result",
        "olmoearth_scores_from_file",
        "olmoearth_plan_label_sample",
        "olmoearth_estimate_map_error",
        "olmoearth_certify_zone",
        "olmoearth_classification_metrics",
        "olmoearth_load_skill",
    ):
        assert tool in core, tool
    for deferred in (
        "olmoearth_automate",
        "olmoearth_rslearn_recommend",
        "olmoearth_rslearn_validate",
        "olmoearth_rslearn_compose",
        "olmoearth_rslearn_diagnose",
        "olmoearth_cloud_mask_audit",
        "olmoearth_similarity_search",
        "olmoearth_area_of_applicability",
    ):
        assert deferred not in core, deferred


def test_a_default_turn_carries_about_half_the_spec_tokens() -> None:
    """Measured as the audit did: len(JSON of the specs) / 4.

    13,706 tokens per turn before the merge and the deferred groups; a
    default turn now sends the core set only.
    """
    import json
    from dataclasses import asdict

    specs = build_default_registry().active_specs()
    assert len(json.dumps([asdict(s) for s in specs])) / 4 < 8_000
