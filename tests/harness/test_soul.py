# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Unit tests for the soul artifact loader (harness/soul.py)."""

from __future__ import annotations

from pathlib import Path

import pytest

from olmoearth_agent.harness.soul import (
    PACKAGED_SOUL,
    SOUL_PATH_ENV,
    load_soul,
    soul_path,
)


def test_packaged_soul_exists_and_loads() -> None:
    assert PACKAGED_SOUL.is_file()
    text = load_soul()
    assert text == PACKAGED_SOUL.read_text(encoding="utf-8").strip()
    # The soul's behavioral boundaries are an explicit, named section.
    assert "## Guardrails" in text
    assert "Never invent project, area, dataset, model, or prediction IDs" in text


def test_soul_routes_review_and_estimation_questions_to_their_tools() -> None:
    """The trial's model ranked hand-sampled pixels and invented an interval."""
    text = load_soul()
    assert "olmoearth_review_set_from_result" in text
    assert "olmoearth_plan_label_sample" in text
    assert "olmoearth_estimate_map_error" in text
    assert "a review set is not a sample" in text


def test_soul_routes_a_direct_model_runs_map_to_the_scores_provider() -> None:
    """A cluster run's raster goes through the provider, never Studio sampling."""
    text = load_soul()
    assert "direct model run" in text
    assert "olmoearth_scores_from_file" in text


def test_soul_states_numbers_as_the_tools_returned_them() -> None:
    """exp86 round 1: derived ratios ("3/46 ~ 6.5%"), a wrong subtraction ("47
    dropped") and a figure quoted from a tool's description ("51-70%")."""
    text = " ".join(load_soul().split())  # the rule wraps across lines
    assert "exactly as the tools returned them" in text
    assert "no ratio, difference or percentage of your own" in text
    assert "never quote a figure from a tool's description" in text
    # exp86 round 3: the rule alone was broken once a round; the harness checks.
    assert "checks your answer's numbers against the tool results" in text


def test_soul_states_as_fact_only_what_a_tool_returned() -> None:
    """exp86 round 9 (B3/studio): "no ground-truth labels exist", while Studio's
    model records showed both models fine-tuned on the project's label fields,
    which no tool reads."""
    text = " ".join(load_soul().split())
    assert (
        "State as fact only what a tool of this run returned or the user said." in text
    )
    assert "whether ground-truth labels exist, what a model was trained on" in text
    assert "say it is not known from this run" in text
    # The soul's own routing no longer presumes that no labels exist.
    assert "with no ground-truth labels" not in text
    assert "when ground-truth labels exist" not in text


def test_soul_proposes_only_what_the_capability_card_lists() -> None:
    """exp86 round 9: 7 of 16 material findings were offers no tool can carry
    out, or whose preconditions did not hold; the owner keeps offers, possible
    ones only."""
    text = " ".join(load_soul().split())
    assert "Propose next steps freely, but only actions the capability card" in text
    assert "each with its preconditions" in text
    assert "say plainly that no tool of this agent can do it" in text


def test_default_system_prompt_is_the_soul() -> None:
    from olmoearth_agent.harness.agent import DEFAULT_SYSTEM_PROMPT

    # The agent's base prompt is exactly the versioned artifact — editing
    # soul.md is the way to change behavior, no code edit involved.
    assert DEFAULT_SYSTEM_PROMPT == PACKAGED_SOUL.read_text(encoding="utf-8").strip()


def test_env_override_swaps_the_soul(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    custom = tmp_path / "custom_soul.md"
    custom.write_text("You are a test soul.\n", encoding="utf-8")
    monkeypatch.setenv(SOUL_PATH_ENV, str(custom))
    assert soul_path() == custom
    assert load_soul() == "You are a test soul."


def test_broken_override_falls_back_to_packaged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(SOUL_PATH_ENV, str(tmp_path / "missing.md"))
    # A bad operator path must not yield an agent with no boundaries.
    assert load_soul() == PACKAGED_SOUL.read_text(encoding="utf-8").strip()
