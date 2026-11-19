# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Round r5-j (18 Nov 2026), track SHEET, as specs/r5-j-round.md gives it:
the label sheet's items of the audit after r5-i
(specs/r5-i-audit/build_list.md B2, B7 and B8). The views are read as text,
as they are on disk (no test here runs the scripts).

- B2: the yellow outline of the square on its pictures (``OE.picture``,
  which the label sheet and the look-first pane share) has a 1 px black edge
  outside and inside it, drawn as a box-shadow, so the frame shows on cloud,
  sand or snow; its box is one pixel further out on each side (3 px outside
  the square, after review), so the inner line covers none of the square's
  pixels; its yellow border and the pictures' captions are as they were.
- B7: in the list, a row with no answer shows "(not labelled yet)" in its
  class menu, the word of the Unlabelled filter and of the counter (it showed
  "(class seen)", like an answer given); the option's value stays empty.
- B8: every control inside a label sheet except those in its footer keeps a
  scroll margin of 140 px at its bottom (since r5-k K5 the footer's own
  height and 8 px, which change with what the footer holds), so a control
  that takes the focus
  (by the keyboard, or by the sheet after an answer) is not left under the
  sheet's sticky footer; the footer's own controls keep none, since the
  footer cannot hide them and a margin there would only scroll the sheet
  (after review); the footer's own rules do not change, and nothing outside
  a sheet gets the margin.

The browser harness (scripts/mcp_views_e2e.py) draws them: 6b and 10b read
the outline's computed edge, 3 and 7 the list's empty option, and 3 and 12b
the focused Cannot tell against the footer at 380 px, and that the focus on
the footer's own button scrolls nothing.
"""

from __future__ import annotations

import re

from olmoearth_agent.mcp_app import views

#: B2: the outline's style before its box: the 2 px yellow border, then the
#: 1 px black edge outside and inside it.
OUTLINE_STYLE = (
    "style: 'position:absolute;box-sizing:border-box;pointer-events:none;"
    "border:2px solid #ffd600;box-shadow:0 0 0 1px #000,inset 0 0 0 1px #000;'"
)
#: B2, after review: the outline's box, 3 px outside the square on each side
#: (it was 2 px), so the inner black line lies outside the square.
OUTLINE_BOX = (
    "        + 'left:' + (Number(b.x) * s - 3) + 'px;top:' + (Number(b.y) * s - 3)"
    " + 'px;width:' + (Number(b.w) * s + 6) + 'px;height:' + (Number(b.h) * s + 6)"
    " + 'px' })\n"
)
#: B2, unchanged: the context picture's caption, by the unit word.
CAPTION = "'Context, ' + (unit === 'point' ? 'point' : 'square') + ' outlined'"

#: B7: the first option of a list row's class menu, its value empty.
EMPTY_OPTION = "OE.el('option', { value: '' }, '(not labelled yet)'),"
#: B7: the words it had.
OLD_EMPTY = "(class seen)"

#: B8: the sheet's controls, except the footer's own, keep 140 px below them
#: when scrolled into view; r5-k K5 makes it the footer's own height and 8 px
#: (--oe-foot, which the sheet sets from the footer; 170 px until it does),
#: since the footer's height changes with what it holds
#: (tests/mcp_app/test_mcp_r5k_sheet.py).
SCROLL_MARGIN = (
    "section.sheet :is(button, select, input, textarea, summary, [tabindex])"
    ":not(.sheet-footer *) { scroll-margin-bottom: "
    "calc(var(--oe-foot, 162px) + 8px); }\n"
)
#: B8, unchanged: the sticky footer's rule, the rule the margin follows.
FOOTER = (
    ".sheet-footer { position: sticky; bottom: 0; margin: 16px -16px -16px;"
    " padding: 12px 16px;\n"
    "  border-top: 1px solid var(--divider); background: var(--paper);"
    " border-radius: 0 0 4px 4px; }\n"
)


def _read(name: str) -> str:
    """A file of views/, as it is on disk."""
    return (views.VIEW_DIR / name).read_text(encoding="utf-8")


def _between(text: str, start: str, end: str) -> str:
    """The part of text from start up to the first end after it."""
    at = text.index(start)
    return text[at : text.index(end, at)]


def _script() -> str:
    """views/sheet.js, as it is on disk."""
    return _read("sheet.js")


def _editor() -> str:
    """The body of OE.sheetEditor."""
    return _between(_script(), "OE.sheetEditor = (app, data,", "\n};")


def _picture() -> str:
    """The body of OE.picture, one picture with its outline and caption."""
    return _between(_script(), "OE.picture = (", "\n};")


def _style() -> str:
    """The style sheet every view gets (common.js OE.STYLE), comments and all."""
    return _between(_read("common.js"), "OE.STYLE = `", "\n`;\n")


def _code(js: str) -> str:
    """JS text without its comments (a block, or the rest of a line after
    ``//``): what a word gone from the page must be gone from."""
    bare = re.sub(r"/\*.*?\*/", "", js, flags=re.S)
    return re.sub(r"//[^\n]*", "", bare)


def _rules(style: str) -> str:
    """A style sheet without its comments."""
    return re.sub(r"/\*.*?\*/", "", style, flags=re.S)


# --------------------------------------------------------------------------- B2


def test_b2_the_outline_has_a_black_edge_on_each_side() -> None:
    """B2: the outline of the square is its 2 px yellow border with a 1 px
    black line outside it and one inside it, both a box-shadow in the
    outline's own style, right after the border; its box is one pixel
    further out than it was (3 px outside the square on each side, after
    review), so the inner line covers none of the square's pixels."""
    picture = _picture()
    assert OUTLINE_STYLE + "\n" + OUTLINE_BOX in picture
    # One outline, one shadow: the picture's own box and image keep theirs.
    assert picture.count("#ffd600") == 1
    assert picture.count("box-shadow:") == 1
    assert "box-shadow:0 0 0 1px #000,inset 0 0 0 1px #000;" in picture
    # Box-shadow only: no margin, outline or padding added to the outline.
    style = _between(picture, "{ 'aria-hidden': 'true', style: '", " })")
    for moved in ("margin", "padding", "outline:", "transform"):
        assert moved not in style, moved
    # The picture's frame is as it was: its hairline outline and the image.
    assert (
        "OE.el('div', { style: 'position:relative;outline:1px solid var(--rule);"
        "border-radius:4px;width:' + w + 'px;height:' + h + 'px' }," in picture
    )


def test_b2_the_captions_and_their_users_do_not_change() -> None:
    """B2: the pictures' captions keep their words ("Context, square
    outlined" by unit, or "Detail"), and the dates' block still draws each
    picture with OE.picture, so the label sheet's pane and the look-first
    pane both get the edge."""
    picture = _picture()
    assert CAPTION in picture
    assert "(p.view === 'detail' ? 'Detail'" in picture
    dates = _between(_script(), "OE.datesBlock = (", "\n};")
    assert "OE.picture(p, String(d.label || ''), what, null, unit)" in dates
    assert "OE.datesBlock(" in _read("result.html")


# --------------------------------------------------------------------------- B7


def test_b7_a_row_with_no_answer_says_not_labelled_yet() -> None:
    """B7: a list row's class menu opens with "(not labelled yet)", value
    empty, under its WORDING line of this round; then the map's classes,
    Cannot tell in the face's words, and "Other class..." last; the old
    "(class seen)" is gone from the script's code."""
    editor = _editor()
    assert "    // WORDING: r5j B7\n    " + EMPTY_OPTION + "\n" in editor
    menu = _between(editor, "const select = OE.el('select', {", "return select;")
    assert menu.index(EMPTY_OPTION) < menu.index("options.map((c) => OE.el('option'")
    assert menu.index(EMPTY_OPTION) < menu.index("'Other class...'")
    assert OLD_EMPTY not in _code(_script())
    # Its word is the filter's and the counter's.
    script = _script()
    assert "['unlabelled', 'Unlabelled']" in script
    assert "OE.el('span', {}, 'labelled')" in script


def test_b7_is_the_one_wording_mark_of_this_round_in_the_sheet() -> None:
    """B7 is the one string of this round in sheet.js: one WORDING line of
    r5-j, just above the option it marks (B2 changes a style, not words)."""
    script = _script()
    assert script.count("// WORDING: r5j") == 1
    assert script.count("// WORDING: r5j B7\n") == 1


# --------------------------------------------------------------------------- B8


def test_b8_the_sheets_controls_keep_a_margin_over_the_footer() -> None:
    """B8: every control inside a label sheet except those in its footer (a
    button, a menu, a field, a fold's summary, anything with a tabindex: a
    list row's answers, the face) keeps the footer's height and 8 px below it
    when the browser brings it into view (140 px until r5-k K5, which takes
    the footer's height from the sheet's --oe-foot), so the sticky footer,
    whatever it holds, does not hide the focus. The rule follows the footer's
    own, which does not change."""
    style = _style()
    assert style.count(SCROLL_MARGIN) == 1
    assert FOOTER in style
    assert style.index(FOOTER) < style.index(SCROLL_MARGIN)
    # The footer's row keeps its rules, side by side (r5-b J29).
    row = ".sheet-footer > div.row { margin-left: auto; justify-content: flex-end; }"
    inner = ".sheet-footer > div.row > .row { justify-content: flex-end; }"
    assert row + "\n" + inner + "\n" in style
    assert style.index(SCROLL_MARGIN) < style.index(row)


def test_b8_only_a_sheets_controls_get_the_margin() -> None:
    """B8: the margin is the style sheet's one scroll margin and it is
    scoped to section.sheet (the label sheet, on its own or open in a Check
    or result card): no view's other controls move, and no scroll padding is
    set; the pictures question keeps its own margin, set where it is brought
    into view (r5-f F01), which an element's own style puts first."""
    rules = _rules(_style())
    assert rules.count("scroll-margin") == 1
    assert "scroll-padding" not in rules
    (line,) = [x for x in rules.split("\n") if "scroll-margin" in x]
    assert line.startswith("section.sheet :is(")
    for name in ("check.html", "result.html", "sheet.html", "start.html"):
        assert "scroll-margin" not in _read(name), name
    assert "preview.style.scrollMarginBottom = margin;" in _script()


def test_b8_the_footers_own_controls_keep_no_margin() -> None:
    """B8, after review: the rule leaves out the controls inside the sheet's
    footer (its selector ends with ``:not(.sheet-footer *)``): the footer is
    pinned at the panel's foot, so it cannot hide its own buttons (the finish
    button, Save labels, Reopen the sheet), and a margin there would only
    scroll the sheet each time one takes the focus (the finish button takes
    it after the last answer). Those buttons are drawn inside the footer."""
    rules = _rules(_style())
    (line,) = [x for x in rules.split("\n") if "scroll-margin" in x]
    selector = line.split(" {", 1)[0]
    assert selector.endswith(":not(.sheet-footer *)")
    assert selector.count(":not(") == 1
    # The finish button and Reopen the sheet go in done; Save labels beside it;
    # under them the line of the saves (r5-k K5), then the finish press's slot.
    assert (
        "OE.el('div', { class: 'sheet-footer spread' }, progress,"
        " OE.el('div', { class: 'row' }, done, saveButton), saveSaid,\n"
        "      finishType)" in _editor()
    )
