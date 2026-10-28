# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""README facts that must match the code: the skill and tool counts."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from olmoearth_agent.skills.registry import (
    OPT_IN_ENV,
    OPT_IN_TOOLS,
    SKILLS,
    build_default_registry,
)

_README = Path(__file__).resolve().parents[1] / "README.md"


def test_readme_skill_counts_match_the_catalog() -> None:
    """The badge's default and opt-in skills, and every "N skills" / "N-skill" mention.

    A skill is opt-in when every tool it contributes is (``SkillSpec.opt_in_only``).
    """
    text = _README.read_text(encoding="utf-8")
    numbered = [s for s in SKILLS if s.number >= 1]
    opt_in = [s for s in numbered if s.opt_in_only]
    badge = re.findall(r"badge/skills-(\d+)%20default%20%2B%20(\d+)%20opt--in-", text)
    assert badge == [(str(len(numbered) - len(opt_in)), str(len(opt_in)))]
    mentions = re.findall(r"\b(\d+)(?:-skill\b| skills\b)", text)
    assert mentions, "README no longer states the skill count"
    assert set(mentions) == {str(len(numbered))}


def test_readme_tool_counts_match_the_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The default tools, those sent every turn, and the opt-in ones, by count."""
    monkeypatch.delenv(OPT_IN_ENV, raising=False)
    monkeypatch.delenv("OLMOEARTH_RUN_PYTHON", raising=False)
    registry = build_default_registry()
    text = _README.read_text(encoding="utf-8")
    assert re.findall(r"\b(\d+) tools by default\b", text) == [
        str(len(registry.names()))
    ]
    assert re.findall(r"\b(\d+) are sent on every turn\b", text) == [
        str(len(registry.active_specs()))
    ]
    assert re.findall(r"\b(\d+) opt-in tools\b", text) == [str(len(OPT_IN_TOOLS))]
    assert f"{OPT_IN_ENV}=1" in text
