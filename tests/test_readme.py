# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""README facts that must match the code: the skill count."""

from __future__ import annotations

import re
from pathlib import Path

from olmoearth_agent.skills.registry import SKILLS

_README = Path(__file__).resolve().parents[1] / "README.md"


def test_readme_skill_counts_match_the_catalog() -> None:
    """The badge and every "N skills" / "N-skill" mention equal the catalog."""
    text = _README.read_text(encoding="utf-8")
    catalog = len([s for s in SKILLS if s.number >= 1])
    badge = re.findall(r"badge/skills-(\d+)-", text)
    mentions = re.findall(r"\b(\d+)(?:-skill\b| skills\b)", text)
    assert badge == [str(catalog)]
    assert mentions, "README no longer states the skill count"
    assert set(mentions) == {str(catalog)}
