# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""README facts that must match the code: the skill and tool counts."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from olmoearth_agent.skills.registry import (
    READ_ONLY_ENV,
    SKILLS,
    build_default_registry,
)

_README = Path(__file__).resolve().parents[1] / "README.md"


def test_readme_skill_counts_match_the_catalog() -> None:
    """The badge's skill count, and every "N skills" / "N-skill" mention."""
    text = _README.read_text(encoding="utf-8")
    numbered = [s for s in SKILLS if s.number >= 1]
    badge = re.findall(r"badge/skills-(\d+)-[0-9A-Fa-f]{6}\.svg", text)
    assert badge == [str(len(numbered))]
    mentions = re.findall(r"\b(\d+)(?:-skill\b| skills\b)", text)
    assert mentions, "README no longer states the skill count"
    assert set(mentions) == {str(len(numbered))}


def test_readme_tool_counts_match_the_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The default tools and those sent every turn, by count.

    The opt-in setting went with the last opt-in tools, so the README names
    neither.
    """
    monkeypatch.delenv(READ_ONLY_ENV, raising=False)
    registry = build_default_registry()
    text = _README.read_text(encoding="utf-8")
    assert re.findall(r"\b(\d+) tools by default\b", text) == [
        str(len(registry.names()))
    ]
    assert re.findall(r"\b(\d+) are sent on every turn\b", text) == [
        str(len(registry.active_specs()))
    ]
    assert not re.findall(r"\b\d+ opt-in tools\b", text)
    assert "OLMOEARTH_OPT_IN_TOOLS" not in text
