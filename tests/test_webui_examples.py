# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""The web UI shows what the default agent does and counts what the catalog has.

The landing page's example runs and the demo mode's canned runs call only
tools of the default registry; the tagline's, the skills button's and the "/"
menu's skill counts, and the cards, are the catalog's. Every local file the page
links is in ``webui/``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from olmoearth_agent.skills.registry import (
    READ_ONLY_ENV,
    SKILLS,
    build_default_registry,
)

_WEBUI = Path(__file__).resolve().parents[1] / "webui"

#: The run ``make verify-demo`` recorded, every call's arguments and envelope.
_VERIFICATION_DEMO_JSON = _WEBUI.parent / "docs" / "verification_demo.json"

#: Link targets that name no file: an anchor, inline data, a mail address.
_NOT_A_FILE = ("#", "data:", "mailto:")


def _read(*parts: str) -> str:
    """One web UI file's text."""
    return _WEBUI.joinpath(*parts).read_text(encoding="utf-8")


def _scenario(run_js: str, name: str) -> str:
    """The body of one demo scenario of ``run.js``'s ``SCENARIOS``."""
    block = re.search(rf"\n  {name}: \{{\n(.*?)\n  \}},", run_js, re.S)
    assert block, name
    return block.group(1)


def _recorded_caption(tool: str, design: str | None = None) -> str:
    """The figure caption the one recorded call of ``tool`` (and design) returned."""
    record = json.loads(_VERIFICATION_DEMO_JSON.read_text(encoding="utf-8"))
    captions = [
        call["envelope"]["result"]["figure_caption"]
        for call in record["calls"]
        if call["tool"] == tool
        and (design is None or call["envelope"]["result"].get("design") == design)
    ]
    assert len(captions) == 1, (tool, design, len(captions))
    caption = captions[0]
    assert isinstance(caption, str), (tool, design)
    return caption


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


def test_the_demo_verification_replays_quote_make_verify_demo() -> None:
    """The demo's review-set and error-rate runs replay docs/VERIFICATION_DEMO.md.

    Each figure the two canned runs show is a figure ``make verify-demo``
    recorded on the Dynamic World tile, and each answer says where it comes
    from; the README's walkthrough video shows these two runs.
    """
    run_js = _read("js", "run.js")
    verify_demo = _WEBUI.parent / "docs" / "VERIFICATION_DEMO.md"
    page = verify_demo.read_text(encoding="utf-8")
    quoted = {
        "review": ["791", "15,813", "532"],
        "error": ["19.0%", "14.8% to 23.9%", "39.6%", "25.9% to 54.6%", "15,813"],
    }
    for name, figures in quoted.items():
        block = _scenario(run_js, name)
        assert "make verify-demo" in block, name
        for figure in figures:
            assert figure in block, (name, figure)
            assert figure in page, (name, figure)
    # The review run's two shares: 532 of the 791 windows, and the tile's error rate.
    assert "532 of the 3,048 errors" in page
    assert "3,048 of 15,813 windows wrong: 19.3%" in page
    assert round(100 * 532 / 791) == 67 and "(67%)" in run_js
    assert "wrong 19% of the time" in run_js
    assert "48 the map calls this class" in page and "48 labelled windows" in run_js


def test_the_demo_verification_replays_show_the_recorded_figures() -> None:
    """The review-set and error-rate demo runs show the tool's figure and caption.

    Each scenario's ``figure_src`` is an SVG shipped in webui/ (opened in a
    new tab as a document, so it holds no script), and its ``figure_caption``
    is, character for character, the caption ``make verify-demo`` recorded in
    docs/verification_demo.json: the review set's, and the random design's
    error estimate's. ``demoEvents`` puts both where the renderer reads them.
    """
    run_js = _read("js", "run.js")
    expected = {
        "review": (
            "demo/figures/review_set.svg",
            _recorded_caption("olmoearth_review_set"),
        ),
        "error": (
            "demo/figures/error_rate.svg",
            _recorded_caption("olmoearth_estimate_map_error", design="random"),
        ),
    }
    for name, (source, caption) in expected.items():
        block = _scenario(run_js, name)
        assert re.findall(r"figure_src: '([^']+)'", block) == [source], name
        figure = _WEBUI / source
        assert figure.is_file(), figure
        assert "<script" not in figure.read_text(encoding="utf-8"), figure
        shown = re.findall(r'figure_caption: "([^"\\]*)"', block)
        assert shown == [caption], name
    assert "toolResult.figure_src = sc.figure_src;" in run_js
    assert "resultObj.figure_caption = sc.figure_caption;" in run_js


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
