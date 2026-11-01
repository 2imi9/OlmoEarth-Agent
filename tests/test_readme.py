# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""README facts that must match the repo: the skill and tool counts, linked files."""

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

#: A Markdown link or image target, or an HTML ``src`` or ``href`` value.
_LINK = re.compile(r'\]\(([^)\s]+)\)|(?:src|href)="([^"]+)"')


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


@pytest.mark.parametrize("page", ["README.md", "webui/README.md"])
def test_readme_links_name_files_that_exist(page: str) -> None:
    """Every relative link and image names a file the repository holds.

    The web UI's demo GIFs and MP4s, their recorders and its screenshot went
    on 31 October 2026; a link left to one of them is caught here.
    """
    path = _README.parent / page
    targets = [a or b for a, b in _LINK.findall(path.read_text(encoding="utf-8"))]
    local = [t.split("#")[0] for t in targets if "://" not in t]
    local = [t for t in local if t and not t.startswith("mailto:")]
    assert local, f"{page} links no local file"
    missing = [t for t in local if not (path.parent / t).exists()]
    assert not missing, (page, missing)
