# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""The web UI shows what the default agent does and counts what the catalog has.

The landing page's example runs and the demo mode's canned runs call only
tools of the default registry, never an opt-in one; the tagline's, the skills
button's and the "/" menu's skill counts, and the cards' opt-in marks, are the
catalog's.
"""

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

_WEBUI = Path(__file__).resolve().parents[1] / "webui"


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
    monkeypatch.delenv(OPT_IN_ENV, raising=False)
    calls = _example_calls()
    assert "olmoearth_compare_results" in calls
    assert "olmoearth_plan_label_sample" in calls
    assert not set(calls) & set(OPT_IN_TOOLS), set(calls) & set(OPT_IN_TOOLS)
    assert set(calls) <= set(build_default_registry().names())


def test_the_landing_page_and_the_slash_menu_count_the_catalog() -> None:
    numbered = [s for s in SKILLS if s.number >= 1]
    total = len(numbered)
    whole = len([s for s in numbered if s.opt_in_only])
    page = _read("index.html")
    tagline = re.findall(
        r"<strong>(\d+) skills</strong>,\s*(\d+) on by default and (\d+) opt-in",
        page,
    )
    assert tagline == [(str(total), str(total - whole), str(whole))]
    assert re.findall(r"Browse the (\d+) skills", page) == [str(total)]
    slash = re.findall(r"pick one of the (\d+)\s+skills", _read("js", "slash.js"))
    assert slash == [str(total)]
    cards = _read("js", "skills.js")
    assert len(re.findall(r"\{ n: \d+,\s+slug: ", cards)) == total
    assert cards.count("optIn: true, ex:") == whole
