# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""The built wheel must carry the instruction skills.

Builds the sdist and, from it, the wheel with ``uv build`` into a temporary
directory (the same route a source install takes), then checks both archives
and runs the loader from the unpacked wheel. Skipped when ``uv`` is not on
``PATH``.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]

#: The four instruction packages that must ship in the wheel.
_SKILLS = (
    "olmoearth-data-prep",
    "olmoearth-embeddings",
    "olmoearth-rslearn",
    "olmoearth-studio-job-config",
)
_PREFIX = "olmoearth_agent/skills/packages/"


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """Build the sdist and wheel once for this module; return their paths."""
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv is not on PATH")
    out = tmp_path_factory.mktemp("dist")
    subprocess.run(  # noqa: S603 - fixed argv, no shell
        [uv, "build", "--out-dir", str(out), str(_REPO_ROOT)],
        check=True,
        capture_output=True,
        text=True,
    )
    (sdist,) = out.glob("*.tar.gz")
    (wheel,) = out.glob("*.whl")
    return sdist, wheel


def test_wheel_contains_the_skill_packages(built: tuple[Path, Path]) -> None:
    """Every SKILL.md, its references and scripts, and the licence ship."""
    _, wheel = built
    names = set(zipfile.ZipFile(wheel).namelist())
    for skill in _SKILLS:
        assert f"{_PREFIX}{skill}/SKILL.md" in names, skill
    assert f"{_PREFIX}LICENSE" in names
    assert f"{_PREFIX}README.md" in names
    assert f"{_PREFIX}olmoearth-data-prep/scripts/audit.py" in names
    assert f"{_PREFIX}olmoearth-studio-job-config/references/presets.md" in names


def test_sdist_contains_the_skill_packages(built: tuple[Path, Path]) -> None:
    """A wheel built from the sdist can only ship what the sdist carries."""
    sdist, _ = built
    with tarfile.open(sdist) as archive:
        names = {name.split("/", 1)[1] for name in archive.getnames() if "/" in name}
    for skill in _SKILLS:
        assert f"src/{_PREFIX}{skill}/SKILL.md" in names, skill


def test_loader_finds_the_skills_in_the_unpacked_wheel(
    built: tuple[Path, Path], tmp_path: Path
) -> None:
    """The default loader, imported from the wheel, discovers all four."""
    _, wheel = built
    site = tmp_path / "site"
    zipfile.ZipFile(wheel).extractall(site)  # noqa: S202 - our own wheel
    probe = (
        "import json, sys\n"
        f"sys.path.insert(0, {str(site)!r})\n"
        "from olmoearth_agent.skills.loader import DEFAULT_SKILLS_DIR, SkillLoader\n"
        "names = sorted(s.name for s in SkillLoader().discover())\n"
        "print(json.dumps({'dir': str(DEFAULT_SKILLS_DIR), 'names': names}))\n"
    )
    env = {"PATH": "", "PYTHONDONTWRITEBYTECODE": "1"}
    result = subprocess.run(  # noqa: S603 - fixed argv, no shell
        [sys.executable, "-c", probe],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    report = json.loads(result.stdout)
    assert Path(report["dir"]).is_relative_to(site.resolve())
    assert report["names"] == list(_SKILLS)
