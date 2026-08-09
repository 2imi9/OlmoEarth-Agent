# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Tests for the run-specific concise response policy."""

from __future__ import annotations

from olmoearth_agent.harness.response_policy import ResponsePolicy


def test_auto_defaults_to_concise_and_honors_explicit_detail() -> None:
    policy = ResponsePolicy()

    assert policy.effective_style("List my projects") == "concise"
    assert policy.effective_style("Give me a detailed report") == "detailed"
    assert policy.effective_style("Explain this step-by-step") == "detailed"
    assert policy.effective_style("Write a full YAML config") == "detailed"
    assert policy.effective_style("List all projects") == "detailed"
    assert policy.effective_style("Return the complete table") == "detailed"


def test_pinned_concise_overrides_detail_phrase() -> None:
    policy = ResponsePolicy(style="concise")

    assert policy.effective_style("Give me a comprehensive report") == "concise"


def test_pinned_detailed_does_not_need_trigger_phrase() -> None:
    policy = ResponsePolicy(style="detailed")

    assert policy.effective_style("status?") == "detailed"


def test_contract_is_specific_and_action_oriented() -> None:
    policy = ResponsePolicy(concise_max_words=90)

    contract = policy.contract("Check prediction status")

    assert "90 words" in contract
    assert "Answer first" in contract
    assert "Do not recap" in contract


def test_only_overlong_concise_answer_needs_compaction() -> None:
    policy = ResponsePolicy(concise_max_words=5)

    assert policy.needs_compaction("status?", "one two three four five") is False
    assert policy.needs_compaction("status?", "one two three four five six") is True
    assert (
        policy.needs_compaction(
            "Give me a detailed report", "one two three four five six"
        )
        is False
    )
    assert (
        policy.needs_compaction("write code", "```python\none two three\n```") is False
    )


def test_editor_must_preserve_ids_paths_urls_and_numbers() -> None:
    policy = ResponsePolicy()
    draft = (
        "1. Prediction pred-nim-42 Completed with 1,842 pixels and 7% cloud. "
        "Artifact: artifacts/pred-nim-42.tif. "
        "Details: https://example.test/results/pred-nim-42."
    )
    lossless = (
        "pred-nim-42 completed: 1842 pixels, 7 percent cloud. "
        "artifacts/pred-nim-42.tif; https://example.test/results/pred-nim-42"
    )

    assert policy.preserves_required_markers(draft, lossless) is True
    assert policy.preserves_required_markers(draft, "Prediction completed.") is False
    assert (
        policy.preserves_required_markers(
            draft, lossless + " New result pred-invented-99."
        )
        is False
    )


def test_from_env_falls_back_and_clamps(monkeypatch: object) -> None:
    # pytest's monkeypatch fixture is duck-typed here to keep this unit module
    # independent of pytest imports at runtime.
    setter = getattr(monkeypatch, "setenv")
    setter("OLMOEARTH_RESPONSE_STYLE", "invalid")
    setter("OLMOEARTH_RESPONSE_MAX_WORDS", "2")
    setter("OLMOEARTH_AGENT_MAX_OUTPUT_TOKENS", "999999")

    policy = ResponsePolicy.from_env()

    assert policy.style == "auto"
    assert policy.concise_max_words == 40
    assert policy.concise_max_tokens == 32768


def test_from_env_uses_defaults_for_non_integer_budgets(monkeypatch: object) -> None:
    setter = getattr(monkeypatch, "setenv")
    setter("OLMOEARTH_RESPONSE_MAX_WORDS", "many")
    setter("OLMOEARTH_AGENT_MAX_OUTPUT_TOKENS", "lots")

    policy = ResponsePolicy.from_env()

    assert policy.concise_max_words == 120
    assert policy.concise_max_tokens == 4096
