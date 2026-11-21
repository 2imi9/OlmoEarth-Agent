# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Round r5-r (20 Nov 2026), track SHEET, as specs/r5-r-round.md gives it. The
owner, after labelling a sheet live: "for labelling we need a button to the
next empty one, so we will not have this 99/100 scenario".

- N1: on the one-square view, beside Previous and Next, a third button "Next
  unlabelled" (``W.nextOpen``; "Next unlabelled square" for a screen reader,
  ``W.nextOpenLabel``): the next square with no answer after the one shown,
  wrapping round to the first from the top (``jumpOpen``). Shown only while
  a square has no answer (``updateOpen``); an outlined small button as the
  other two, the three in equal columns that wrap to a second row on a
  narrow panel (``PAGER``, one layout with or without the third), so they
  keep one size; the key U where no answer of the sheet has that key
  (``openKey``), named at the end of the keys line while a square has no
  answer; it shows a square as Next does (``pageBy``), so the focus stays
  where Next leaves it; grey while the square shown is the only one with no
  answer. The list keeps its Unlabelled filter.
- N2: while 1 to 3 squares are left and at least one is answered, the
  counter's row ends with "Unlabelled: Square 29" (or "Unlabelled: Squares
  7, 29 and 80"), each square a text button that opens it; the grey finish
  button's line reads "Answer every square first: go to the next
  unlabelled.", its last words a text button that does what N1 does.
- Every sheet kind: none of the new code asks which face or kind the sheet
  is, and the words follow the sheet's unit word ("point" on a sheet of
  points). Blind as before: the new code reads a square's place and whether
  it has an answer, never what the map says there.

The views are read as text, as they are on disk (no test here runs the
scripts); the words of ``OE.sheetWords`` are read back from the file
(:func:`_said`). The browser harness (scripts/mcp_views_e2e.py, which no test
runs) draws it: 10e_kelp_next_unlabelled, a sheet with 99 of 100 squares
answered, where a press of Next unlabelled, the key U, the footer's text
button and the counter's "Square 29" each show the empty square, at 780 and
at 380 px.

No sheet here holds a place.
"""

from __future__ import annotations

import re
from pathlib import Path

from olmoearth_agent.mcp_app import views

#: N1: the button's words and its name for a screen reader.
NEXT_OPEN = "Next unlabelled"
NEXT_OPEN_LABEL = "Next unlabelled {one}"
#: N1: the key's sentence, at the end of the face's keys line.
NEXT_OPEN_KEY = " U shows the next unlabelled {one}."
#: N2: the counter's line for one, two and three squares left.
LEFT_ONE = "Unlabelled: Square 29"
LEFT_TWO = "Unlabelled: Squares 7 and 29"
LEFT_THREE = "Unlabelled: Squares 7, 29 and 80"
#: N2: a named square's button for a screen reader.
GO_TO = "Go to square 29"
#: N2: the grey finish button's line, whole, and its text button's words.
NOT_YET = "Answer every {one} first: go to the next unlabelled."
GO_OPEN = "go to the next unlabelled"
#: This round's mark.
MARK = "// WORDING: r5r"

#: One token of a JS concatenation: a single-quoted string, a plus, or a name.
_TOKEN = re.compile(
    r"\s*(?:'((?:[^'\\]|\\.)*)'|(\+)|([A-Za-z_][A-Za-z0-9_.]*(?:\([^()]*\))?))"
)

#: The staging tree's root (CHANGELOG.md, docs/, spec/).
ROOT = Path(__file__).resolve().parents[2]


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


def _words() -> str:
    """The body of OE.sheetWords, the sheet's table of words."""
    return _between(_script(), "OE.sheetWords = (unit) => {", "\n};")


def _editor() -> str:
    """The body of OE.sheetEditor."""
    return _between(_script(), "OE.sheetEditor = (app, data,", "\n};")


def _function(name: str) -> str:
    """One function of OE.sheetEditor, from its first line to its end."""
    return _between(_editor(), f"  function {name}(", "\n  }\n")


def _code(js: str) -> str:
    """JS text without its comments (a block, or the rest of a line after
    ``//``), each run of joined single-quoted literals read as one and escaped
    quotes unescaped."""
    bare = re.sub(r"/\*.*?\*/", "", js, flags=re.S)
    bare = re.sub(r"//[^\n]*", "", bare)
    return re.sub(r"'\s*\+\s*'", "", bare).replace("\\'", "'")


def _concat(expr: str, names: dict[str, str]) -> str:
    """The text a JS concatenation of single-quoted strings and names makes,
    each name given its value in names; anything else fails."""
    out: list[str] = []
    text = expr.strip()
    at = 0
    while at < len(text):
        match = _TOKEN.match(text, at)
        assert match is not None, text[at:]
        literal, _plus, name = match.groups()
        if literal is not None:
            out.append(re.sub(r"\\(.)", r"\1", literal))
        elif name is not None:
            out.append(names[name])
        at = match.end()
    return "".join(out)


def _entry(block: str, start: str) -> str:
    """One entry's value in a JS object literal: from start to the comma that
    ends its last line."""
    at = block.index(start) + len(start)
    end = block.find(",\n", at)
    return (block[at:] if end < 0 else block[at:end]).strip().rstrip(",")


def _said(key: str, unit: str, **names: str) -> str:
    """An entry of OE.sheetWords as the face says it, for a unit word; an
    entry that takes a number (``(k) => ...``) is given it as ``k``."""
    one = "point" if unit == "point" else "square"
    value = re.sub(r"^\([a-z]+\) => ", "", _entry(_words(), f"    {key}: "))
    given = {"one": one, "many": one + "s", "One": one.capitalize(), **names}
    return _concat(value, given)


def _tags_above(key: str) -> list[str]:
    """The WORDING lines right above an entry of OE.sheetWords, nearest
    first, up to the first line that is not a WORDING line."""
    lines = _words().split("\n")
    (at,) = [i for i, line in enumerate(lines) if line.startswith(f"    {key}: ")]
    tags: list[str] = []
    for line in reversed(lines[:at]):
        if not line.strip().startswith("// WORDING: "):
            break
        tags.append(line.strip())
    return tags


def _left_line(places: list[int], unit: str = "square") -> str:
    """The counter's line for the squares left at places, put together as
    updateOpen puts it: the lead, then one square by name, or the plural word
    and the places joined by ", " with " and " before the last (the joiners
    are pinned in test_n2_the_counter_names_the_last_squares_left)."""
    lead = _said("leftLead", unit)
    if len(places) == 1:
        return lead + _said("leftOne", unit, k=str(places[0]))
    out = lead + _said("leftMany", unit)
    for i, k in enumerate(places):
        if i:
            out += " and " if i == len(places) - 1 else ", "
        out += str(k)
    return out


# --------------------------------------------------------------------------- N1


def test_n1_the_buttons_words() -> None:
    """N1: "Next unlabelled", named "Next unlabelled square" for a screen
    reader ("point" on a sheet of points), never Next's own name, and the
    key's sentence; each carries this round's mark."""
    assert _said("nextOpen", "square") == NEXT_OPEN
    for unit in ("square", "point"):
        label = _said("nextOpenLabel", unit)
        assert label == NEXT_OPEN_LABEL.format(one=unit)
        # Next's name ("Next square") is no part of it: one button per name.
        assert _said("nextLabel", unit) not in label
        assert _said("nextOpenKey", unit) == NEXT_OPEN_KEY.format(one=unit)
    for key in ("nextOpen", "nextOpenLabel", "nextOpenKey"):
        assert _tags_above(key) == [MARK], key
    # Previous and Next keep their names.
    assert _said("prevLabel", "square") == "Previous square"
    assert _said("nextLabel", "square") == "Next square"


def test_n1_the_button_is_made_as_previous_and_next_are() -> None:
    """N1: an outlined small button (class small, as Previous and Next: not
    a text button, not the dark one), made hidden, the third of the row in
    the face's heading; its press is jumpOpen, told that it came from inside
    the face."""
    editor = _editor()
    made = (
        "  const nextOpenButton = OE.el('button', {\n"
        "    class: 'small', 'data-focus': 'next-open', "
        "'aria-label': W.nextOpenLabel, 'aria-keyshortcuts': openKey,\n"
        "    onclick: () => jumpOpen(true),\n"
        "  }, W.nextOpen);\n"
        "  nextOpenButton.hidden = true;\n"
    )
    assert made in editor
    row = (
        "  const pager = OE.el('div', { class: 'row', style: PAGER }, "
        "prevButton, nextButton, nextOpenButton);\n"
    )
    assert row in editor
    assert editor.index("  }, 'Next');\n") < editor.index(made) < editor.index(row)
    assert "      OE.el('div', { class: 'face-top' }, faceLive, pager),\n" in editor
    # Previous and Next are made as before.
    for name, label, step in (("prev", "prevLabel", "-1"), ("next", "nextLabel", "1")):
        assert (
            f"class: 'small', 'data-focus': '{name}', 'aria-label': W.{label}, "
            f"onclick: () => pageBy({step}),"
        ) in editor


def test_n1_shown_only_while_a_square_has_no_answer() -> None:
    """N1: updateOpen, run by updateProgress (so after every answer and every
    render), shows the button while a square has no answer and hides it once
    none is left; a button that goes while it holds the focus hands it to the
    face."""
    update = _function("updateOpen")
    assert "    const left = openRows();\n    const show = left.length > 0;\n" in update
    assert "      nextOpenButton.hidden = !show;\n" in update
    assert "      if (had && !quietSave) OE.focusIfHere(faceEl);\n" in update
    editor = _editor()
    assert (
        "  const openRows = () => windows.filter((w) => !isLabelledRow(w));\n" in editor
    )
    progress = _function("updateProgress")
    assert progress.count("    updateOpen();\n") == 1
    at = progress.index("    updateOpen();\n")
    assert at < progress.index("    if (state.stale) {")


def test_n1_three_buttons_of_one_size_at_every_width() -> None:
    """N1 (design rule: same kind, same size, at 780 and 380 px): the row of
    Previous, Next and Next unlabelled is a grid of equal columns at least
    132 px wide that wrap (two to a row on a narrow panel), every row as tall
    as the tallest. Its inline style sets the row flow, which overrides the
    column flow common.js gives ``.face-top > .row`` (under that flow each
    button took a new column and the third never wrapped). One layout whether
    the third button shows or not (auto-fill keeps its column's place), so
    Previous and Next keep their size and place once every square is
    answered. The browser harness measures it (pager_rows)."""
    code = _code(_editor())
    pager = (
        "const PAGER = 'display:grid;grid-auto-flow:row;"
        "grid-template-columns:repeat(auto-fill,minmax(132px,1fr));"
        "grid-auto-rows:1fr;align-items:stretch;gap:8px;flex:0 1 412px;min-width:0';"
    )
    assert pager in code
    assert "PAGER_TWO" not in code and "PAGER_THREE" not in code
    # The layout is set once, where the row is made: no switch of it later.
    assert code.count("style: PAGER }") == 1
    assert "pager.style" not in code
    # Three columns and two gaps are the row's width, read from the style.
    col = re.search(r"minmax\((\d+)px", pager)
    gap = re.search(r"gap:(\d+)px", pager)
    basis = re.search(r"flex:0 1 (\d+)px", pager)
    assert col is not None and gap is not None and basis is not None
    assert int(basis.group(1)) == 3 * int(col.group(1)) + 2 * int(gap.group(1))
    # The rule of common.js the inline style must override.
    common = _read("common.js")
    assert (
        ".toggle, .face-top > .row { display: inline-grid; grid-auto-flow: column;"
        in common
    )


def test_n1_the_next_square_with_no_answer_wrapping_round() -> None:
    """N1: on the face jumpOpen shows the next square with no answer after
    the one shown, wrapping round (nextUnanswered, which never returns the
    square shown), or the first left when the square shown alone has none:
    the button and U (inFace) then do nothing, and the button is grey
    (aria-disabled, set by updateOpen outside its early return, so it follows
    the square shown). It pages there as Next does (showFromAnywhere, by
    pageBy): a press inside the face keeps the focus where it is, a press
    from outside it (the footer) gives the face the focus, so an answer's key
    works at once. With none left it does nothing."""
    jump = _function("jumpOpen")
    assert jump.startswith("  function jumpOpen(inFace) {\n")
    assert "    const left = openRows();\n    if (!left.length) return;\n" in jump
    face = (
        "    if (state.view === 'face') {\n"
        "      const next = nextUnanswered(state.at) || left[0];\n"
        "      if (inFace && next.handle === state.at) return;\n"
        "      showFromAnywhere(next.handle);\n"
        "      return;\n"
        "    }\n"
    )
    assert face in jump
    shown = _function("showFromAnywhere")
    assert "    pageBy(place(h) - place(state.at));\n" in shown
    assert (
        "    if (!faceEl.contains(document.activeElement)) "
        "{ OE.focusIfHere(faceEl); headInView(); }" in shown
    )
    update = _function("updateOpen")
    grey = (
        "    const only = state.view === 'face' && left.length === 1 "
        "&& left[0].handle === state.at;\n"
        "    nextOpenButton.setAttribute('aria-disabled', String(only));\n"
    )
    assert grey in update
    assert update.index(grey) < update.index("    if (key === leftShown) return;\n")
    following = _function("nextUnanswered")
    assert "    for (let step = 1; step < windows.length; step += 1) {\n" in following
    assert "      const w = windows[(at + step) % windows.length];\n" in following
    assert "      if (!isLabelledRow(w)) return w;\n" in following
    page = _function("pageBy")
    assert page.endswith("    goTo(windows[to].handle);\n    headInView();")
    # Showing a square is not answering: nothing here sets a label or saves.
    for name in ("jumpOpen", "openRow", "showFromAnywhere", "updateOpen"):
        body = _code(_function(name))
        for word in ("setLabel(", "save(", "schedule(", "state.labels.set("):
            assert word not in body, (name, word)


def test_n1_the_key_u_where_no_answer_has_it() -> None:
    """N1: the key is U unless an answer of the sheet has that key (a seen
    sheet's class whose name starts with U keeps it): the button then names
    no key and the keys line does not offer it. On the face the key is read
    after the arrow keys and after the answers' keys, before the lock is
    asked (showing is not answering), and does what the button does. The
    keys line offers it only while a square has no answer, and updateOpen
    writes the line again when the button comes or goes."""
    editor = _editor()
    assert (
        "  const openKey = answers.some((a) => a.keys.includes('U')) ? null : 'U';\n"
        in editor
    )
    face_key = _function("onFaceKey")
    block = (
        "    if (!hit && openKey && pressed === openKey && state.at) {\n"
        "      e.preventDefault();\n"
        "      jumpOpen(true);\n"
        "      return;\n"
        "    }\n"
        "    if (!hit) return;\n"
    )
    assert block in face_key
    arrows = face_key.index("if (k === 'ArrowLeft' || k === 'ArrowRight') {")
    answer = face_key.index("const hit = pressed && state.at ? answers.find(")
    assert arrows < answer < face_key.index(block)
    assert face_key.index(block) < face_key.index("if (blocked()) return;")
    hint = _function("faceHintText")
    assert "+ (openKey && openRows().length ? W.nextOpenKey : '');" in hint
    assert (
        "      if (state.view === 'face') "
        "{ hint.textContent = faceHintText(); hint.hidden = !hint.textContent; }\n"
        in _function("updateOpen")
    )
    # The list's keys line and its filter are as they were.
    assert "nextOpenKey" not in _function("listHintText")
    assert (
        "[['any', 'Any'], ['unlabelled', 'Unlabelled'], ['labelled', 'Labelled']], "
        "state.filter," in editor
    )


# --------------------------------------------------------------------------- N2


def test_n2_the_counters_line_in_words() -> None:
    """N2: "Unlabelled: Square 29", "Unlabelled: Squares 7 and 29",
    "Unlabelled: Squares 7, 29 and 80"; "Point" on a sheet of points; a named
    square's button reads "Go to square 29" to a screen reader."""
    assert _left_line([29]) == LEFT_ONE
    assert _left_line([7, 29]) == LEFT_TWO
    assert _left_line([7, 29, 80]) == LEFT_THREE
    assert _left_line([29], "point") == "Unlabelled: Point 29"
    assert _left_line([7, 29, 80], "point") == "Unlabelled: Points 7, 29 and 80"
    assert _said("goTo", "square", k="29") == GO_TO
    assert _said("goTo", "point", k="29") == "Go to point 29"
    for key in ("leftLead", "leftOne", "leftMany", "goTo"):
        assert _tags_above(key) == [MARK], key


def test_n2_the_counter_names_the_last_squares_left() -> None:
    """N2: the line is the last of the counter's row, made hidden; updateOpen
    shows it while 1 to 3 squares are left and at least one is answered (not
    the whole of a fresh sheet of three), each square a text button named
    W.goTo whose press opens it; the joiners are ", " and, before the last,
    " and "."""
    editor = _editor()
    made = "  const leftLine = OE.el('span', { class: 'small oe-left' });\n"
    assert made in editor
    assert "  leftLine.hidden = true;\n  progress.append(leftLine);\n" in editor
    counter = (
        "counterNum, OE.el('span', {}, 'labelled'), counterNote, savedNote, countSaid);"
    )
    assert editor.index(counter) < editor.index(made)
    update = _function("updateOpen")
    few = "    const few = left.length >= 1 && left.length <= 3 && left.length < n;\n"
    assert few in update
    assert "    leftLine.hidden = !few;\n" in update
    link = (
        "    const link = (w, words) => OE.el('button', {\n"
        "      class: 'link', style: LINE_LINK, 'data-focus': 'left:' + w.handle, "
        "'aria-label': W.goTo(place(w.handle)),\n"
        "      onclick: () => openRow(w.handle),\n"
        "    }, words);\n"
    )
    assert link in update
    put = (
        "    const parts = [W.leftLead];\n"
        "    if (left.length === 1) {\n"
        "      parts.push(link(left[0], W.leftOne(place(left[0].handle))));\n"
        "    } else {\n"
        "      parts.push(W.leftMany);\n"
        "      left.forEach((w, i) => {\n"
        "        if (i) parts.push(i === left.length - 1 ? ' and ' : ', ');\n"
        "        parts.push(link(w, String(place(w.handle))));\n"
        "      });\n"
        "    }\n"
        "    leftLine.replaceChildren(...parts);\n"
    )
    assert put in update
    # A named square opens: on the face as Next shows one, and the face takes
    # the focus (the press came from the footer), so an answer's key works at
    # once; from the list on the face, as a press of its row's number does.
    opened = _function("openRow")
    assert "    if (state.view === 'face') { showFromAnywhere(h); return; }\n" in opened
    assert "    setView('face', h);" in opened


def test_n2_the_grey_finish_buttons_line_ends_in_a_jump() -> None:
    """N2: "Answer every square first: go to the next unlabelled." ("point"
    on a sheet of points): W.notYet, a text button with W.goOpen whose press
    is jumpOpen, and a full stop. A sheet that already gave its result keeps
    its own line with no button (its answers are locked). The line is still
    the finish button's description."""
    for unit in ("square", "point"):
        line = _said("notYet", unit) + _said("goOpen", unit) + "."
        assert line == NOT_YET.format(one=unit)
    assert _said("goOpen", "square") == GO_OPEN
    assert _tags_above("goOpen") == [MARK]
    assert _tags_above("notYet") == ["// WORDING: r5f F02", MARK]
    editor = _editor()
    jump = (
        "  // WORDING: r5r\n"
        "  const goOpenLink = OE.el('button', {\n"
        "    class: 'link', style: LINE_LINK, 'data-focus': 'go-open', "
        "onclick: () => jumpOpen(),\n"
        "  }, W.goOpen);\n"
    )
    assert jump in editor
    line_made = "    gaveResult ? W.gradedBlank : [W.notYet, goOpenLink, '.']);\n"
    assert line_made in editor
    assert editor.index(jump) < editor.index(line_made)
    finish = _function("updateFinish")
    assert (
        "    if (waits) button.setAttribute('aria-describedby', notYetLine.id);\n"
        in finish
    )
    # In the list the jump gives the first row with no answer the focus, at
    # its answers; where the filter hides that row, it shows on the face.
    whole = _function("jumpOpen")
    listed = whole[whole.index("    const first = left[0];\n") :]
    assert (
        "    if (!visibleRows().includes(first)) "
        "{ setView('face', first.handle); return; }\n" in listed
    )
    assert "    render(firstAnswerFocusKey(first));\n" in listed


def test_n2_a_text_button_in_a_small_line() -> None:
    """N2: the text buttons of the two lines are in the line's size,
    underlined, 24 px tall (3 px above and below an 18 px line) and at least
    24 px wide: a square's number may be one digit."""
    code = _code(_editor())
    style = (
        "const LINE_LINK = 'font-size:inherit;line-height:inherit;padding:3px 4px;"
        "min-width:24px;text-decoration:underline;text-underline-offset:2px';"
    )
    assert style in code
    assert ".small { font-size: 12px; line-height: 18px; }" in _read("common.js")
    assert code.count("style: LINE_LINK") == 2


# ------------------------------------------------------- every kind, and blind


def test_the_new_code_asks_no_sheet_kind_and_reads_no_map_answer() -> None:
    """Every sheet kind (one map, two maps, class seen, points): none of the
    new functions asks which face, kind or unit the sheet has. Blind as
    before: they read a square's handle, its place and whether it has an
    answer, never what the map says there (the extras of an unblinded read,
    a map class, a pair), nor a stratum or an index."""
    for name in ("jumpOpen", "openRow", "updateOpen"):
        body = _code(_function(name))
        for kind in ("twoMap", "wrongFace", "seen", "closed", "face ===", "data."):
            assert kind not in body, (name, kind)
        for hidden in ("state.extra", "mapClass", "map_class", "pair", "stratum"):
            assert hidden not in body, (name, hidden)
        assert "index" not in body.replace("findIndex", ""), name
    # Whether a square has an answer is the counter's own test.
    editor = _editor()
    assert (
        "  const isLabelledRow = (w) => answered(state.labels.get(w.handle) || '');\n"
        in editor
    )


# ------------------------------------------------------------------- the marks


def test_every_r5r_string_of_the_sheet_carries_its_wording_comment() -> None:
    """Each entry of OE.sheetWords this round added or changed has this
    round's WORDING line, and so has the footer's text button where it is
    made; no other line of the script got one."""
    keys = (
        "nextOpen",
        "nextOpenLabel",
        "nextOpenKey",
        "leftLead",
        "leftOne",
        "leftMany",
        "goTo",
        "notYet",
        "goOpen",
    )
    for key in keys:
        assert MARK in _tags_above(key), key
    assert _words().count(MARK + "\n") == len(keys)
    assert _script().count(MARK + "\n") == len(keys) + 1


# --------------------------------------------------------------------- the docs


def test_r5r_sheet_the_changelog_and_the_docs_say_so() -> None:
    """The CHANGELOG has one plain line for each item under Added (in the
    Unreleased part, or in the version that takes it), docs/mcp-app.md says
    them where the label sheet's buttons are described, and spec/Usage.md
    gives the words."""
    log = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    newest = _between(log, "## [Unreleased]", "\n## [1.5.3]")
    assert "### Added\n" in newest
    for text in (
        '- Label sheet: a "Next unlabelled" button beside Previous and Next (key U, '
        "unless an answer on the sheet already uses U) shows the next square with "
        "no answer,",
        "- Label sheet: while 1 to 3 squares are left, the counter names them "
        '("Unlabelled: Square 29")',
        '"Answer every square first: go to the next unlabelled."',
    ):
        assert text in newest, text
    guide = (ROOT / "docs" / "mcp-app.md").read_text(encoding="utf-8")
    sheet = " ".join(_between(guide, "## Label sheet", "\n## ").split())
    assert (
        "**Next unlabelled**, beside Previous and Next (key U, unless an answer on "
        "the sheet already uses U)," in sheet
    )
    assert 'the counter names them ("Unlabelled: Square 29")' in sheet
    usage = " ".join((ROOT / "spec" / "Usage.md").read_text(encoding="utf-8").split())
    for said in (
        '"Answer every square first: go to the next unlabelled."',
        'a third button stands beside Previous and Next, "Next unlabelled"',
        '"Unlabelled: Square 29" or "Unlabelled: Squares 7, 29 and 80"',
        '"U shows the next unlabelled square."',
    ):
        assert said in usage, said
    assert '"Answer every square first."' not in usage
