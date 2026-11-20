"""Round r5-p (owner, 19 Nov 2026, a screenshot of the Check card): each step's
sentence sits centred under its picture, as the picture does, in the Check
card's and the result card's steps (``OE.fig.steps`` in views/figures.js)."""

from __future__ import annotations

from olmoearth_agent.mcp_app import views


def _style() -> str:
    """The figures' stylesheet as written in figures.js."""
    return (views.VIEW_DIR / "figures.js").read_text(encoding="utf-8")


def test_each_step_sentence_is_centred_under_its_picture() -> None:
    """The step's panel centres its children and its text; the picture keeps
    its own centring."""
    style = _style()
    start = style.index(".oe-steps > li {")
    rule = style[start : style.index("}", start)]
    assert "align-items: center;" in rule
    assert "text-align: center;" in rule
    # The picture and its sentence, as one block, sit in the middle of the
    # panel's height too (the owner's second note: figure and text together).
    assert "justify-content: center;" in rule
    assert "align-items: flex-start" not in rule
    pic = ".oe-steps .oe-step-pic { display: block; flex: none; align-self: center; }"
    assert pic in style
