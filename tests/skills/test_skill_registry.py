# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Sanity tests for the skill catalog manifest."""

from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path

import pytest

from olmoearth_agent.harness.capabilities import capability_card
from olmoearth_agent.harness.soul import load_soul
from olmoearth_agent.skills.loader import SkillLoader
from olmoearth_agent.skills.registry import (
    OPT_IN_ENV,
    OPT_IN_TOOLS,
    READ_ONLY_ENV,
    SKILLS,
    STUDIO_WRITE_TOOLS,
    build_default_registry,
    skills_by_status,
)
from olmoearth_agent.tools.registry import ToolRegistry

_SKILLS_MD = Path(__file__).resolve().parents[2] / "SKILLS.md"

#: The opt-in tools removed on 31 October 2026 with their code (outside
#: inference examination, and not used): written out, so a catalog row or a
#: text the agent sends that still names one is caught.
REMOVED = (
    "olmoearth_automate",
    "olmoearth_cv_inflation_check",
    "olmoearth_nndm_cv",
    "olmoearth_rslearn_recommend",
    "olmoearth_rslearn_validate",
    "olmoearth_rslearn_compose",
    "olmoearth_rslearn_diagnose",
    "olmoearth_area_of_applicability",
    "olmoearth_grade_review_rule",
    "olmoearth_change_detect",
)

#: Every tool the default registry keeps (the plan's Keep rows), by name.
KEPT = (
    "olmoearth_load_context",
    "olmoearth_create_project",
    "olmoearth_request_aoi",
    "olmoearth_get_prediction",
    "olmoearth_studio_label_inventory",
    "olmoearth_search_predictions",
    "olmoearth_submit_prediction",
    "olmoearth_fetch_results",
    "olmoearth_get_prediction_result",
    "olmoearth_pixel_value",
    "olmoearth_export_data",
    "olmoearth_compare_results",
    "olmoearth_classification_metrics",
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
    "olmoearth_case_narrative",
    "olmoearth_qgis_bridge",
    "olmoearth_provenance_summary",
    "olmoearth_load_skill",
    "olmoearth_remember",
    "olmoearth_forget",
)


@pytest.fixture
def default_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """The opt-in setting is off (whatever the shell running the tests holds)."""
    monkeypatch.delenv(OPT_IN_ENV, raising=False)


def _opted_in(monkeypatch: pytest.MonkeyPatch, value: str = "1") -> ToolRegistry:
    """The registry built with the documented opt-in setting on."""
    monkeypatch.setenv(OPT_IN_ENV, value)
    return build_default_registry()


def _turn_chars(registry: ToolRegistry) -> int:
    """Characters of one default turn's specs, as the audit measured them."""
    return len(json.dumps([asdict(s) for s in registry.active_specs()]))


def test_catalog_rows_are_numbered_from_0_in_increasing_order() -> None:
    """Row #0 first, then each row's number above the one before it.

    Gaps are allowed while the trim removes rows; the renumbering closes them.
    """
    numbers = [s.number for s in SKILLS]
    assert numbers[0] == 0
    assert all(a < b for a, b in zip(numbers, numbers[1:]))


def test_every_catalogued_tool_is_actually_registered(default_env: None) -> None:
    """A catalog row that names a tool the registry never builds is a lie."""
    registered = set(build_default_registry().names())
    for skill in SKILLS:
        if skill.status in ("implemented", "foundational"):
            for tool in skill.default_tools:
                if tool == "olmoearth_load_skill":
                    continue
                assert tool in registered, f"#{skill.number} {skill.name}: {tool}"


# --------------------------------------------------------------------------- the trim


def test_no_catalog_row_opts_a_tool_in() -> None:
    """The last opt-in tools went on 31 October 2026, with their code."""
    assert OPT_IN_TOOLS == {}
    assert not [skill.number for skill in SKILLS if skill.opt_in]


@pytest.mark.parametrize("tool", REMOVED)
def test_each_removed_tool_is_gone_even_opted_in(
    tool: str, default_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    for registry in (build_default_registry(), _opted_in(monkeypatch)):
        assert tool not in registry.names()
        assert registry.group_of(tool) is None
    for skill in SKILLS:
        assert tool not in skill.tools, f"#{skill.number} lists {tool}"


@pytest.mark.parametrize(
    "value", ["1", "true", "Yes", " on ", "", "0", "false", "no", "off", "maybe"]
)
def test_the_opt_in_setting_adds_no_tool(
    value: str, default_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No tool is opt-in, so whatever the setting holds the registry is KEPT."""
    assert sorted(_opted_in(monkeypatch, value).names()) == sorted(KEPT)


@pytest.mark.parametrize("value", ["1", "true", "Yes", " on "])
def test_a_read_only_key_leaves_out_exactly_the_studio_write_tools(
    value: str, default_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """OLMOEARTH_READ_ONLY: no tool that would meet Studio's refusal is offered."""
    assert STUDIO_WRITE_TOOLS == {
        "olmoearth_create_project",
        "olmoearth_submit_prediction",
    }
    monkeypatch.setenv(READ_ONLY_ENV, value)
    registry = build_default_registry()
    assert sorted(registry.names()) == sorted(set(KEPT) - STUDIO_WRITE_TOOLS)
    assert "studio_write" in registry.cannot
    # The opt-in setting brings no write tool back.
    assert not STUDIO_WRITE_TOOLS & set(_opted_in(monkeypatch).names())


@pytest.mark.parametrize("value", ["", "0", "off", "maybe"])
def test_without_the_read_only_setting_the_registry_is_unchanged(
    value: str, default_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(READ_ONLY_ENV, value)
    registry = build_default_registry()
    assert sorted(registry.names()) == sorted(KEPT)
    assert registry.cannot == {}


@pytest.mark.parametrize("tool", KEPT)
def test_each_kept_tool_is_in_the_default_registry(
    tool: str, default_env: None
) -> None:
    assert tool in build_default_registry().names()


def test_the_default_registry_is_the_kept_tools_and_nothing_else(
    default_env: None,
) -> None:
    """30 tools."""
    assert sorted(build_default_registry().names()) == sorted(KEPT)
    assert len(KEPT) == 30


def test_no_tool_waits_for_its_skill(
    default_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No tool is deferred, opted in or not.

    olmoearth_load_skill then names no deferred group.
    """
    assert _opted_in(monkeypatch).groups() == {}
    monkeypatch.delenv(OPT_IN_ENV)
    registry = build_default_registry()
    assert registry.groups() == {}
    load = registry.spec_of("olmoearth_load_skill")
    assert load is not None
    assert "not sent until their skill is loaded" not in load.description
    for gone in (
        "olmoearth-rslearn",
        "olmoearth-evaluate",
        "olmoearth-uncertainty",
    ):
        assert gone not in load.description


def test_skills_md_gives_each_opt_in_tool_its_reason() -> None:
    """SKILLS.md's opt-in table: a row per tool, with the registry's reason."""
    text = _SKILLS_MD.read_text(encoding="utf-8")
    rows = [row for row in text.splitlines() if row.startswith("| `olmoearth_")]
    assert len(rows) == len(OPT_IN_TOOLS)
    for tool, reason in OPT_IN_TOOLS.items():
        assert any(f"`{tool}`" in row and reason in row for row in rows), tool
    assert f"{OPT_IN_ENV}=1" in text


def test_no_page_says_asta_covers_the_literature_search() -> None:
    """The literature search went as outside inference examination and unused.

    The owner's correction of 28 October 2026 and decision of 31 October: no
    page gives "Asta covers it" as the reason (the CHANGELOG keeps its
    history).
    """
    root = _SKILLS_MD.parent
    pages = [
        root / "README.md",
        _SKILLS_MD,
        root / "PLAN.md",
        root / "webui" / "js" / "skills.js",
    ]
    for path in pages:
        text = path.read_text(encoding="utf-8")
        for stale in ("Asta covers", "Explorer or Asta", "Explorer and Asta"):
            assert stale not in text, (path.name, stale)


def test_no_skill_is_opt_in_whole() -> None:
    """The five skills that were opt-in whole (#6, #8, #10, #15, #16) are gone."""
    assert not {s.name for s in SKILLS if s.opt_in_only}


def test_compare_results_is_the_tool_for_a_series(default_env: None) -> None:
    """change_detect is gone: compare_results' description takes series."""
    spec = build_default_registry().spec_of("olmoearth_compare_results")
    assert spec is not None
    assert "the tool for a time series" in spec.description
    assert "olmoearth_change_detect" not in spec.description


def _removed_tools_named(text: str) -> set[str]:
    """The removed tools ``text`` names, by name or by a ``prefix_*`` pattern."""
    named: set[str] = set()
    for token in re.findall(r"olmoearth_[a-z0-9_]+\*?", text):
        if token.endswith("*"):
            named |= {t for t in REMOVED if t.startswith(token[:-1])}
        elif token in REMOVED:
            named.add(token)
    return named


def test_the_name_check_finds_a_name_and_a_pattern() -> None:
    assert _removed_tools_named("see olmoearth_change_detect.") == {
        "olmoearth_change_detect"
    }
    assert len(_removed_tools_named("the olmoearth_rslearn_* tools")) == 4
    assert not _removed_tools_named("olmoearth_review_set and olmoearth_load_skill")


def test_nothing_the_agent_sends_names_a_removed_tool(
    default_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A removed tool is neither offered nor pointed at.

    The texts the model reads: every tool's description and parameter texts,
    the capability card with every group loaded, the soul, the skill index and
    every SKILL.md package (what ``olmoearth_load_skill`` returns).
    """
    monkeypatch.delenv("OLMOEARTH_SOUL_PATH", raising=False)
    monkeypatch.delenv("OLMOEARTH_SKILLS_DIR", raising=False)
    registry = build_default_registry()
    texts = {
        spec.name: spec.description + json.dumps(spec.parameters)
        for spec in registry.specs()
    }
    texts["capability card"] = capability_card(registry, registry.groups())
    texts["soul"] = load_soul()
    loader = SkillLoader()
    texts["skill index"] = loader.index(brief=False)
    for skill in loader.discover():
        texts[skill.name] = loader.load(skill.name)
    named = {where: _removed_tools_named(text) for where, text in texts.items()}
    assert not {where: n for where, n in named.items() if n}
    assert len(texts) > len(registry.names())


def test_a_read_only_agent_names_a_write_tool_only_where_the_card_answers_it(
    default_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """OLMOEARTH_READ_ONLY: no text points at a left-out write tool unanswered.

    The soul is one text for every run, and its typical order names
    olmoearth_submit_prediction; the read-only card's line names the two tools
    left out and says to skip that step. No tool text, skill index or SKILL.md
    package names either tool.
    """
    monkeypatch.delenv("OLMOEARTH_SOUL_PATH", raising=False)
    monkeypatch.delenv("OLMOEARTH_SKILLS_DIR", raising=False)
    monkeypatch.setenv(READ_ONLY_ENV, "1")
    registry = build_default_registry()
    texts = {
        spec.name: spec.description + json.dumps(spec.parameters)
        for spec in registry.specs()
    }
    loader = SkillLoader()
    texts["skill index"] = loader.index(brief=False)
    for skill in loader.discover():
        texts[skill.name] = loader.load(skill.name)

    def write_tools_named(text: str) -> set[str]:
        return set(re.findall(r"olmoearth_[a-z0-9_]+", text)) & STUDIO_WRITE_TOOLS

    named = {where: write_tools_named(text) for where, text in texts.items()}
    assert not {where: n for where, n in named.items() if n}
    assert write_tools_named(load_soul()) == {"olmoearth_submit_prediction"}
    card = capability_card(registry, registry.groups())
    head, beyond = card.split("No tool of this agent can:", 1)
    assert not write_tools_named(head)
    (line,) = [x for x in beyond.splitlines() if "read-only" in x]
    assert write_tools_named(line) == STUDIO_WRITE_TOOLS
    assert "skip the submit step of the typical order" in line


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


def test_default_registry_exposes_foundational_and_provenance_tools(
    default_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    names = build_default_registry().names()
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
        "olmoearth_classification_metrics",
        "olmoearth_case_narrative",
        "olmoearth_export_data",
        "olmoearth_qgis_bridge",
        "olmoearth_provenance_summary",
        "olmoearth_load_skill",
    ):
        assert tool in names
    # every registered tool, opt-in ones included, has a JSON-schema spec
    for spec in _opted_in(monkeypatch).specs():
        assert spec.parameters["type"] == "object"


@pytest.mark.parametrize("opt_in", [False, True])
def test_merged_and_removed_tools_are_gone(
    opt_in: bool, default_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One comparison tool; search_projects and list_skills merged; litsearch_resolve
    merged, then removed with the literature search."""
    registry = _opted_in(monkeypatch) if opt_in else build_default_registry()
    names = set(registry.names())
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


@pytest.mark.parametrize("opt_in", [False, True])
def test_deferred_groups_are_catalog_skills(
    opt_in: bool, default_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A group is loaded by its skill's name (load_skill, or the UI's forced slug)."""
    registry = _opted_in(monkeypatch) if opt_in else build_default_registry()
    by_name = {s.name: s for s in SKILLS}
    for group, tools in registry.groups().items():
        assert group in by_name, group
        for tool in tools:
            assert tool in by_name[group].tools, f"{group} does not list {tool}"


def test_the_core_set_keeps_the_studio_review_estimation_and_compare_tools(
    default_env: None,
) -> None:
    core = {s.name for s in build_default_registry().active_specs()}
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
        "olmoearth_scores_from_studio",
        "olmoearth_plan_label_sample",
        "olmoearth_estimate_map_error",
        "olmoearth_certify_zone",
        "olmoearth_classification_metrics",
        "olmoearth_studio_label_inventory",
        "olmoearth_load_skill",
    ):
        assert tool in core, tool


def test_a_default_turn_carries_under_8000_spec_tokens(
    default_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Measured as the audit did: len(JSON of the specs) / 4.

    13,706 tokens per turn before the merge and the deferred groups; under
    8,000 (about 7,700) once a default turn sent the core set only. The trim
    of 27 October 2026 takes three core specs off a default turn, counted by
    hand from their source as json.dumps writes them: olmoearth_change_detect
    1,206 characters, olmoearth_litsearch 2,186, olmoearth_grade_review_rule
    1,978, and 358 from olmoearth_load_skill's list of deferred groups, less 28
    added to olmoearth_compare_results (the tool for a time series): 5,700
    characters, 1,425 tokens. 8,000 - 1,425 = 6,575. The which-map step adds
    the two-map design, counted the same way: 704 characters to
    olmoearth_plan_label_sample (map b, the dates, the difference, the
    design), 298 to olmoearth_estimate_map_error (the class seen; the map it
    names, or neither, by the sign test or interval) and 76 to
    olmoearth_compare_results (its pointer to the route), less the 101 of
    olmoearth_load_skill's note on #6's group, now opt-in: 977 characters,
    244 tokens. 6,575 + 244 = 6,819, so the bound was 6,850. The class-map
    step adds, counted the same way, 476 characters to
    olmoearth_plan_label_sample (the class design in its enum and
    description, the sentence on a Studio class map, and min_per_class with
    its "(all of a smaller class)", width, expected_error_rate and
    class_share), 80 to olmoearth_estimate_map_error (the valid interval
    first) and 17 to olmoearth_certify_zone (a class map refused): 573
    characters, 143 tokens. 6,819 + 143 = 6,962; the bound moves by the same
    143, from 6,850 to 6,993, so it is 7,000. The verified-report step adds,
    counted the same way, 79 characters to olmoearth_case_narrative (the
    sentence on the accuracy it states, less the freshness gate's "important
    for disaster-response briefs" and "to produce the deliverable") and 32 to
    olmoearth_provenance_summary (", the verification calls' seeds,"): 111
    characters, 28 tokens. 6,962 + 28 = 6,990; the bound moves by the same
    28, from 6,993 to 7,021, so it is 7,050. The direct-run step adds,
    counted the same way, olmoearth_write_certificate's spec (914
    characters) and olmoearth_verify_certificate's (693) with their two
    separators (4), 428 to olmoearth_plan_label_sample (seed_from_map, alpha,
    delta and rule, 322, and the sentence on seed_from_map, 106), 28 to
    olmoearth_certify_zone (alpha's "a seed_from_map plan's own") and 186 to
    olmoearth_scores_from_file (from-olmoearth's files and the refusal of
    class ids): 2,253 characters, 563 tokens. 6,990 + 563 = 7,553; the bound
    moves by the same 563, from 7,050 to 7,613, so it is 7,650. The review of
    that step adds, counted the same way, 7 characters to
    olmoearth_write_certificate's description ("writes no certificate" for
    "writes nothing") and 24 to its date's ("a score_area.py run's date
    window" for "the run's"): 31 characters, 8 tokens. 7,553 + 8 = 7,561,
    under the same bound. The fixes of 28 October take, counted the same
    way, 34 characters off olmoearth_estimate_map_error's description (the
    interval valid at every sample size first for every design, 48, in
    place of the 80 of the sentence on a class map's or 'class' sample, and
    "each interval and its method" for "estimate, low, high and method", 2
    fewer): 8 tokens. 7,561 - 8 = 7,553, under the same bound. The fixes of
    agent trial 12 move, counted the same way, olmoearth_plan_label_sample
    alone: its description 41 characters shorter (seed_from_map's
    precondition in place of its old sentence, the design's parenthesis
    shortened but still "'random' when a certified zone is wanted", "The
    interval comes from olmoearth_estimate_map_error" and "it draws windows
    where they differ" left out), its design parameter 3 shorter
    ("'disagreement': two maps") and its seed_from_map parameter 62 longer
    (the precondition): 18 characters, 5 tokens. 7,553 + 5 = 7,558, under
    the same bound. The Studio-client step of 31 October adds, counted the
    same way, olmoearth_submit_prediction's probability_top_k parameter (234
    characters with its separator) and 1 to olmoearth_get_prediction_result
    ("file formats"), and takes 31 off olmoearth_fetch_results (Studio's own
    filter, 97 characters, in place of the client-side scan's sentence, 129,
    less 1 for "file formats"): 204 characters, 51 tokens. 7,558 + 51 =
    7,609, under the same bound. The top-k step adds, counted the same way,
    78 characters to olmoearth_review_set_from_result's description ("A
    class map with top-k scores ranks by rank-1 minus rank-2 score. ", and
    "a class map without top-k scores" for "a classification band") and 21
    each to olmoearth_plan_label_sample's and olmoearth_certify_zone's
    (" without top-k scores"): 120 characters, 30 tokens. 7,609 + 30 =
    7,639, under the same bound. The review of that step adds 2 characters
    to olmoearth_review_set_from_result's threshold ("Required for a
    regression band not in [0, 1]." for "Required unless the band is a [0, 1]
    score.", which said a class map with top-k scores needs one): 1 token,
    7,640, under the same bound. The label-inventory step of 31 October adds,
    counted the same way, olmoearth_studio_label_inventory's spec with its
    separator: after the review of that step, 467 characters, 117 tokens
    (its description 233 of them: "a model's training selector selects
    now" and "No random design recorded"). The Studio-files step adds
    olmoearth_scores_from_studio's spec (931 characters: its description 436,
    its four parameters result_id, patch, clip_out_of_range and max_mb) with
    its separator (2): 933 characters, 233 tokens. 7,640 + 117 + 233 =
    7,990; the bound moves by the same 350, from 7,650 to 8,000. The review
    of that step takes '"default": 4, ' (14 characters) off its patch
    parameter, whose default is now chosen per zone, and adds 18 to the
    parameter's description ("; default per zone"); olmoearth_submit_prediction's
    probability_top_k minimum goes from 1 to 2, the same length: 4
    characters, 1 token. 7,990 + 1 = 7,991, under the same bound. These are
    counts by hand, not measurements. The trim of 31 October 2026 took no
    default spec and left no opt-in tool, so an opted-in turn carries the
    default specs.
    """
    default = _turn_chars(build_default_registry())
    assert default / 4 < 8_000
    assert _turn_chars(_opted_in(monkeypatch)) == default
