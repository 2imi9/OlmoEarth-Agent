# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Tests for the vendored-skill loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from olmoearth_agent.skills.loader import SkillLoader, _parse_frontmatter

_SKILL_MD = (
    "---\n"
    "name: my-skill\n"
    "description: does the thing,\n"
    "across two lines\n"
    "---\n"
    "# My Skill\n\nbody text\n"
)


def test_parse_frontmatter_multiline_description() -> None:
    fm = _parse_frontmatter(_SKILL_MD)
    assert fm["name"] == "my-skill"
    assert "does the thing" in fm["description"]
    assert "across two lines" in fm["description"]


def test_parse_frontmatter_absent_returns_empty() -> None:
    assert _parse_frontmatter("# no frontmatter\n") == {}


def _make_skill(root: Path, name: str) -> None:
    folder = root / name
    folder.mkdir(parents=True)
    (folder / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: desc for {name}\n---\n# {name}\nbody\n",
        encoding="utf-8",
    )


def test_loader_discovers_and_loads(tmp_path: Path) -> None:
    _make_skill(tmp_path, "skill-a")
    _make_skill(tmp_path, "skill-b")
    loader = SkillLoader(root=tmp_path)
    names = {s.name for s in loader.discover()}
    assert names == {"skill-a", "skill-b"}
    assert "desc for skill-a" in loader.index()
    assert "# skill-a" in loader.load("skill-a")


def test_loader_missing_dir_is_empty(tmp_path: Path) -> None:
    loader = SkillLoader(root=tmp_path / "does-not-exist")
    assert loader.discover() == []
    assert loader.index() == ""


def test_loader_unknown_skill_raises(tmp_path: Path) -> None:
    loader = SkillLoader(root=tmp_path)
    with pytest.raises(KeyError):
        loader.load("nope")


def test_vendored_submodule_when_initialized() -> None:
    """If the submodule is checked out, the 3 upstream skills are visible."""
    loader = SkillLoader()  # default vendor/ path
    names = {s.name for s in loader.discover()}
    if not names:
        pytest.skip("vendor/olmoearth-skills submodule not initialized")
    assert "olmoearth-data-prep" in names
    assert "olmoearth-embeddings" in names
    assert "olmoearth-studio-job-config" in names
