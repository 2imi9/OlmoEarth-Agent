# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""README and docs/Usage.md facts that must match the repo: counts and links."""

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
_USAGE = _README.parent / "docs" / "Usage.md"

#: A Markdown link or image target, or an HTML ``src`` or ``href`` value.
_LINK = re.compile(r'\]\(([^)\s]+)\)|(?:src|href)="([^"]+)"')


def _anchors(page: Path) -> set[str]:
    """The GitHub anchors of a Markdown page's ``#`` headings."""
    headings = re.findall(r"^#+ (.+)$", page.read_text(encoding="utf-8"), re.M)
    return {
        re.sub(r"[^\w\- ]", "", h.strip().lower()).replace(" ", "-") for h in headings
    }


def test_readme_skill_counts_match_the_catalog() -> None:
    """Every "N skills" / "N-skill" mention, and Usage's numbered skill headings.

    The README's skill badge went with the other badges; the web UI and other
    pages link Usage's skill headings by number and name.
    """
    numbered = [s for s in SKILLS if s.number >= 1]
    for page in (_README, _USAGE):
        text = page.read_text(encoding="utf-8")
        mentions = re.findall(r"\b(\d+)(?:-skill\b| skills\b)", text)
        assert mentions, f"{page.name} no longer states the skill count"
        assert set(mentions) == {str(len(numbered))}, page.name
    headings = re.findall(
        r"^### (\d+)\. `([a-z-]+)`$", _USAGE.read_text(encoding="utf-8"), re.M
    )
    assert headings == [(str(s.number), s.name) for s in numbered]


def test_readme_tool_counts_match_the_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The default tools and those sent every turn, by count; Usage's tool table.

    The opt-in setting went with the last opt-in tools, so neither page names
    it.
    """
    monkeypatch.delenv(READ_ONLY_ENV, raising=False)
    registry = build_default_registry()
    default = [str(len(registry.names()))]
    every_turn = [str(len(registry.active_specs()))]
    for page in (_README, _USAGE):
        text = page.read_text(encoding="utf-8")
        tools = re.findall(r"\b(\d+) tools by default\b", text)
        assert tools == default, page.name
        sent = re.findall(r"\b(\d+) are sent on every turn\b", text)
        assert sent == every_turn, page.name
        assert not re.findall(r"\b\d+ opt-in tools\b", text), page.name
        assert "OLMOEARTH_OPT_IN_TOOLS" not in text, page.name
    table = re.findall(
        r"^\| `(olmoearth_[a-z_]+)` \|", _USAGE.read_text(encoding="utf-8"), re.M
    )
    assert sorted(table) == sorted(registry.names())


@pytest.mark.parametrize("page", ["README.md", "docs/Usage.md"])
def test_readme_links_name_files_that_exist(page: str) -> None:
    """Every relative link and image names a file the repository holds.

    The web UI's demo GIFs and MP4s, their recorders and its screenshot went
    on 31 October 2026; a link left to one of them is caught here. A link to
    a heading of a Markdown page names a heading the page has.
    """
    path = _README.parent / page
    targets = [a or b for a, b in _LINK.findall(path.read_text(encoding="utf-8"))]
    local = [t.split("#")[0] for t in targets if "://" not in t]
    local = [t for t in local if t and not t.startswith("mailto:")]
    assert local, f"{page} links no local file"
    missing = [t for t in local if not (path.parent / t).exists()]
    assert not missing, (page, missing)
    for target in targets:
        name, _, anchor = target.partition("#")
        if not anchor or "://" in target or (name and not name.endswith(".md")):
            continue
        dest = path.parent / name if name else path
        assert anchor in _anchors(dest), (page, target)
