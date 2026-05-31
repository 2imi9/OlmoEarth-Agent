# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Progressive-disclosure loader for vendored agentskills.io packages.

The upstream skills (#1-#4) live in ``2imi9/OlmoEarth-Skills``, vendored
here as a git submodule at ``vendor/olmoearth-skills``. Each is a
``SKILL.md`` instruction package (not a Python tool bundle), so the
harness consumes them the agentskills.io way: the LLM sees each skill's
``name`` + ``description`` up front (the index), and pulls the full
``SKILL.md`` body into context only when a task matches (via the
``olmoearth_load_skill`` tool).

Gracefully returns an empty index if the submodule is not initialized
(e.g. a clone without ``--recurse-submodules``).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

#: Default location of the vendored skills, relative to the repo root.
#: ``loader.py`` is at ``src/olmoearth_agent/skills/`` -> parents[3] = repo root.
DEFAULT_SKILLS_DIR = (
    Path(__file__).resolve().parents[3] / "vendor" / "olmoearth-skills" / "skills"
)

_KEY_RE = re.compile(r"^([A-Za-z_][\w-]*):\s?(.*)$")


def _parse_frontmatter(text: str) -> dict[str, str]:
    """Parse a leading ``---`` YAML-ish frontmatter block.

    Handles single-key-per-line with continuation lines (so a long
    multi-line ``description`` is captured whole). Returns ``{}`` if no
    frontmatter is present.
    """
    if not text.startswith("---"):
        return {}
    end = text.find("\n---", 3)
    if end == -1:
        return {}
    block = text[3:end].strip("\n")
    result: dict[str, str] = {}
    current_key: str | None = None
    parts: list[str] = []
    for line in block.splitlines():
        match = _KEY_RE.match(line)
        if match:
            if current_key is not None:
                result[current_key] = "\n".join(parts).strip()
            current_key = match.group(1)
            parts = [match.group(2)]
        else:
            parts.append(line)
    if current_key is not None:
        result[current_key] = "\n".join(parts).strip()
    return result


@dataclass(frozen=True)
class VendoredSkill:
    """One discovered ``SKILL.md`` package."""

    name: str
    description: str
    path: Path


class SkillLoader:
    """Discovers and loads vendored ``SKILL.md`` packages.

    Resolution order for the skills directory: explicit ``root`` arg →
    ``OLMOEARTH_SKILLS_DIR`` env → :data:`DEFAULT_SKILLS_DIR`.
    """

    def __init__(self, root: str | Path | None = None) -> None:
        chosen = root or os.environ.get("OLMOEARTH_SKILLS_DIR") or DEFAULT_SKILLS_DIR
        self.root = Path(chosen)

    def discover(self) -> list[VendoredSkill]:
        """Return all discoverable skills (empty if the dir is missing)."""
        if not self.root.is_dir():
            return []
        skills: list[VendoredSkill] = []
        for entry in sorted(self.root.iterdir()):
            skill_md = entry / "SKILL.md"
            if not skill_md.is_file():
                continue
            frontmatter = _parse_frontmatter(skill_md.read_text(encoding="utf-8"))
            name = frontmatter.get("name")
            if name:
                skills.append(
                    VendoredSkill(
                        name=name,
                        description=frontmatter.get("description", ""),
                        path=skill_md,
                    )
                )
        return skills

    def index(self, *, brief: bool = True) -> str:
        """Progressive-disclosure index, one line per skill.

        ``brief`` (default) truncates each description to its first
        sentence (≤160 chars) so the index fits a small context window
        when injected into a system prompt. The full descriptions remain
        available via the ``olmoearth_list_skills`` tool, and the full
        instructions via ``olmoearth_load_skill``. Pass ``brief=False``
        for the complete descriptions (large-context models).
        """
        lines = []
        for skill in self.discover():
            if brief:
                summary = skill.description.split(". ")[0][:160]
            else:
                summary = skill.description
            lines.append(f"- {skill.name}: {summary}")
        return "\n".join(lines)

    def load(self, name: str) -> str:
        """Return the full ``SKILL.md`` body for ``name``.

        Raises
        ------
        KeyError
            If no vendored skill has that name.
        """
        for skill in self.discover():
            if skill.name == name:
                return skill.path.read_text(encoding="utf-8")
        msg = f"no vendored skill named {name!r}"
        raise KeyError(msg)
