# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Tests for the skill-package loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from olmoearth_agent.skills.loader import (
    DEFAULT_SKILLS_DIR,
    SkillLoader,
    _parse_frontmatter,
)
from olmoearth_agent.skills.registry import skills_by_status

#: The four instruction packages that ship inside the agent package.
PACKAGED_SKILLS = {
    "olmoearth-data-prep",
    "olmoearth-studio-job-config",
    "olmoearth-embeddings",
    "olmoearth-rslearn",
}

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


def test_default_dir_is_inside_the_package() -> None:
    """The packages resolve beside the loader, so a wheel install finds them."""
    import olmoearth_agent

    package_dir = Path(olmoearth_agent.__file__).resolve().parent
    assert DEFAULT_SKILLS_DIR == package_dir / "skills" / "packages"
    assert (DEFAULT_SKILLS_DIR / "LICENSE").is_file()


def test_default_dir_holds_the_four_packages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No checkout step is needed: the four packages are always present."""
    monkeypatch.delenv("OLMOEARTH_SKILLS_DIR", raising=False)
    loader = SkillLoader()
    assert {s.name for s in loader.discover()} == PACKAGED_SKILLS
    for name in PACKAGED_SKILLS:
        assert loader.load(name).startswith("---\nname: ")


def test_packages_match_the_catalog_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every catalog row served by ``olmoearth_load_skill`` has a package."""
    monkeypatch.delenv("OLMOEARTH_SKILLS_DIR", raising=False)
    catalog = {s.name for s in skills_by_status("vendored")}
    assert catalog == {s.name for s in SkillLoader().discover()}
