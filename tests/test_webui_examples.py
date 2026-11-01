# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""The web UI shows what the default agent does and counts what the catalog has.

The landing page's example runs and the demo mode's canned runs call only
tools of the default registry; the tagline's, the skills button's and the "/"
menu's skill counts, and the cards, are the catalog's. Every local file the page
links is in ``webui/``.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from olmoearth_agent.skills.registry import (
    READ_ONLY_ENV,
    SKILLS,
    build_default_registry,
)

_WEBUI = Path(__file__).resolve().parents[1] / "webui"

#: Link targets that name no file: an anchor, inline data, a mail address.
_NOT_A_FILE = ("#", "data:", "mailto:")


def _read(*parts: str) -> str:
    """One web UI file's text."""
    return _WEBUI.joinpath(*parts).read_text(encoding="utf-8")


def _example_calls() -> list[str]:
    """Every tool the landing page's transcripts and the demo's scenarios call."""
    page = re.findall(
        r"called <code>(olmoearth_[a-z0-9_]+)</code>", _read("index.html")
    )
    demo = re.findall(r"tool: '(olmoearth_[a-z0-9_]+)'", _read("js", "run.js"))
    return page + demo


def test_the_example_runs_call_only_default_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(READ_ONLY_ENV, raising=False)
    calls = _example_calls()
    assert "olmoearth_compare_results" in calls
    assert "olmoearth_plan_label_sample" in calls
    assert set(calls) <= set(build_default_registry().names())


def test_the_landing_page_and_the_slash_menu_count_the_catalog() -> None:
    numbered = [s for s in SKILLS if s.number >= 1]
    total = len(numbered)
    page = _read("index.html")
    tagline = re.findall(r"<strong>(\d+) skills</strong>", page)
    assert tagline == [str(total)]
    assert "opt-in" not in page
    assert re.findall(r"Browse the (\d+) skills", page) == [str(total)]
    slash = re.findall(r"pick one of the (\d+)\s+skills", _read("js", "slash.js"))
    assert slash == [str(total)]
    cards = _read("js", "skills.js")
    assert len(re.findall(r"\{ n: \d+,\s+slug: ", cards)) == total
    assert "optIn" not in cards


def test_the_page_links_only_files_it_ships() -> None:
    """Every local ``src`` and ``href`` of the page names a file in webui/.

    The architecture diagram its menu linked went on 31 October 2026.
    """
    targets = re.findall(r'(?:src|href)="([^"]+)"', _read("index.html"))
    local = [t for t in targets if "://" not in t and not t.startswith(_NOT_A_FILE)]
    assert "styles.css" in local
    missing = [t for t in local if not (_WEBUI / t).exists()]
    assert not missing, missing
