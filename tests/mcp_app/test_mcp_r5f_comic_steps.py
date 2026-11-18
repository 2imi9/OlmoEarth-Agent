# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""The Check card's four steps as comic panels (owner, 18 Nov 2026): each step,
its picture and its sentence, sits in its own framed white box, and all four
panels are one width and one height, two rows of two on a narrow panel
included. The view is read as text; the browser harness measures the sizes."""

from __future__ import annotations

from olmoearth_agent.mcp_app import views


def _style() -> str:
    """The figures' style sheet as figures.js writes it."""
    return (views.VIEW_DIR / "figures.js").read_text(encoding="utf-8")


def test_each_step_is_a_framed_panel_of_one_size() -> None:
    """Every row of the steps' grid is as tall as the tallest (grid-auto-rows:
    1fr) and every column as wide as the others, so the four panels are one
    size; each panel has a visible edge, a white ground and its own padding;
    the picture sits centred inside the frame."""
    style = _style()
    grid = (
        ".oe-steps { list-style: none; margin: 12px 0 0; padding: 0; display: grid;\n"
        "  grid-template-columns: repeat(4, minmax(0, 1fr)); grid-auto-rows: 1fr; "
        "gap: 12px; }"
    )
    assert grid in style
    panel = (
        "  padding: 10px; background: var(--paper); border: 1.5px solid "
        "var(--text-2); border-radius: 6px; }"
    )
    assert panel in style
    pic = ".oe-steps .oe-step-pic { display: block; flex: none; align-self: center; }"
    assert pic in style
    narrow = (
        "@media (max-width: 560px) { .oe-steps { grid-template-columns: "
        "repeat(2, minmax(0, 1fr)); } }"
    )
    assert narrow in style


def test_the_panels_use_the_views_tokens_and_no_mint() -> None:
    """The frame and the ground are the views' own tokens; no mint (it is kept
    for a completed status' ring) and no colour written out."""
    style = _style()
    start, end = style.index(".oe-steps > li {"), style.index(".oe-steps .oe-step-pic")
    rule = style[start:end]
    assert "--mint" not in rule and "#" not in rule
