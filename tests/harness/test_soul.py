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
