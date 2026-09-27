# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Progressive-disclosure loader for the agent's agentskills.io packages.

The four instruction skills (catalog #1-#3 and #17: data-prep,
studio-job-config, embeddings, rslearn) ship inside this package, under
``olmoearth_agent/skills/packages/``, so a wheel install carries them. They
were moved here from ``2imi9/OlmoEarth-Skills`` (now frozen); that
directory's ``LICENSE`` and ``README.md`` keep the licence and attribution.

Each is a ``SKILL.md`` instruction package (not a Python tool bundle), so
the harness consumes them the agentskills.io way: the LLM sees each
skill's ``name`` + ``description`` up front (the index), and pulls the full
``SKILL.md`` body into context only when a task matches (via the
``olmoearth_load_skill`` tool).

An explicit ``root`` or ``OLMOEARTH_SKILLS_DIR`` that does not exist gives
an empty index rather than an error.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

#: Default location of the skill packages: beside this module, inside the
#: installed package (so it resolves the same from a checkout or a wheel).
DEFAULT_SKILLS_DIR = Path(__file__).resolve().parent / "packages"

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
    """One discovered ``SKILL.md`` package.

    The name predates the move of the packages into this repository; it is
    kept because it is part of the public ``olmoearth_agent.skills`` API.
    """

    name: str
    description: str
    path: Path


class SkillLoader:
    """Discovers and loads the ``SKILL.md`` packages.

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
        when injected into a system prompt. The full instructions come
        from ``olmoearth_load_skill``. Pass ``brief=False`` for the complete
        descriptions (large-context models).
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
            If no skill package has that name.
        """
        for skill in self.discover():
            if skill.name == name:
                return skill.path.read_text(encoding="utf-8")
        msg = f"no skill package named {name!r}"
        raise KeyError(msg)
