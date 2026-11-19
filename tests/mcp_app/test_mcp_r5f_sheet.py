# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Round r5-f (18 Nov 2026), track SHEET, as specs/r5-f-round.md gives it: the
label sheet's items of the audit after the design round
(specs/r5-e-audit/build_list.md: F01, F02, F12 and F14). The views are read as
text, as they are on disk (no test here runs the scripts); the words of
``OE.sheetWords`` are read back from the file (:func:`_said`: a JS
concatenation of single-quoted strings and names read into its text), so a
missing space between two joined pieces fails here as it would on the face.

- F01: after a press of Pictures (Show in the list), when the pictures'
  answer is the question that needs a yes, its pane is scrolled into view,
  the least that shows it whole above the sheet's pinned footer when it fits,
  else its top and then the row of "Turn on pictures" above the footer
  (``questionInView``), also when the question was kept from an earlier
  press on the same square; the press of One square with the pane open does
  the same for a question kept for the square shown there. A pane that
  follows an answer, Previous or Next scrolls nothing; no focus moves and no
  word changes.
- F02: while a square has no answer the footer says, beside the grey finish
  button, "Answer every square first." (``W.notYet``), in the place
  ``W.everyDone`` takes once the sheet is ready, and the button names that
  line as its description (``aria-describedby``); not while the sheet is
  stale, not while only a save is pending, never on a sheet that already
  gave its result (r5-k K3 locked its answers: that place says
  ``W.gradedBlank`` there instead), and (where the build list's
  ``!complete()`` would say it
  untruly) not on a sheet with no square, nor on one whose every square is
  answered but which holds two sets of answers. The lead stays two lines.
  On a plan card with its sheet open the card's own line to answer every one
  goes (with track RESULT's F10), so it is said once.
- F12: a sheet with no pictures and no places ends its line "... without
  looking at the map. No records? Stop and tell Claude."; the tail it had
  after a semicolon is gone; a two-map sheet's says "the maps".
- F14: a class name on the face may break after each underscore
  (``OE.softBreaks`` in common.js: a ``wbr`` element between the pieces, the
  text as written), so "evergreen_needleleaf_woodland" is no longer cut in
  mid-word; the helper uses no lookbehind.
"""

from __future__ import annotations

import re

from olmoearth_agent.mcp_app import views

#: F02: beside the grey finish button while a square has no answer (W.notYet).
NOT_YET = "Answer every {one} first."
#: F02, unchanged: beside the ready finish button (W.everyDone), and there on
#: a sheet that already gave its result (W.everyDoneGraded).
EVERY_DONE = "After you press, do not change an answer."
EVERY_DONE_GRADED = "This sheet already gave its result."
#: F12: the first line of a sheet with no pictures and no places (W.whatBare).
WHAT_BARE = (
    "This sheet has no pictures and no places, so it cannot show where a {one} is. "
    "Answer only from your own records of each {one}, without looking at the map. "
    "No records? Stop and tell Claude."
)
#: F12: the same on a two-map sheet (W.whatBarePair).
WHAT_BARE_PAIR = (
    "This sheet has no pictures and no places, so it cannot show where a {one} is. "
    "Answer only from your own records of each {one}, without looking at the maps. "
    "No records? Stop and tell Claude."
)
#: F12: the tail the line had before, gone from the script's code.
OLD_TAIL = "with none, stop and tell Claude"
#: F12, unchanged (r5-e H1): the clause a two-map sheet's lead swaps
#: (W.hiddenOne) and what it puts in its place (W.hiddenPair).
HIDDEN_ONE = ", without looking at the map."
HIDDEN_PAIR = ", without looking at the two maps."
#: F14: the helper as common.js holds it: the text cut after each underscore,
#: a wbr element between two pieces, no piece empty.
SOFT_BREAKS = (
    "OE.softBreaks = (text) => {\n"
    "  const parts = String(text).split('_');\n"
    "  const out = [];\n"
    "  parts.forEach((part, i) => {\n"
    "    const piece = i < parts.length - 1 ? part + '_' : part;\n"
    "    if (!piece) return;\n"
    "    if (out.length) out.push(document.createElement('wbr'));\n"
    "    out.push(piece);\n"
    "  });\n"
    "  return out;\n"
    "};\n"
)

#: One token of a JS concatenation: a single-quoted string, a plus, or a name.
_TOKEN = re.compile(
    r"\s*(?:'((?:[^'\\]|\\.)*)'|(\+)|([A-Za-z_][A-Za-z0-9_.]*(?:\([^()]*\))?))"
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


def _words() -> str:
    """The body of OE.sheetWords, the sheet's table of words."""
    return _between(_script(), "OE.sheetWords = (unit) => {", "\n};")


def _editor() -> str:
    """The body of OE.sheetEditor."""
    return _between(_script(), "OE.sheetEditor = (app, data,", "\n};")


def _code(js: str) -> str:
    """JS text without its comments (a block, or the rest of a line after
    ``//``), each run of joined single-quoted literals read as one and escaped
    quotes unescaped: what a word gone from the page must be gone from."""
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
    """An entry of OE.sheetWords as the face says it, for a unit word."""
    one = "point" if unit == "point" else "square"
    value = re.sub(r"^\([a-z]+\) => ", "", _entry(_words(), f"    {key}: "))
    return _concat(value, {"one": one, "many": one + "s", **names})


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


# --------------------------------------------------------------------------- F01


def test_f01_the_pictures_question_is_brought_into_view() -> None:
    """The pane is scrolled only when the pictures' answer is the question
    (needs_consent): its scroll margin is the pinned footer's height and 8
    px, and the scroll is the least that shows it (block: nearest), so a pane
    already in view does not move; a pane taller than the panel shows its
    top, and a second scroll, the least again, brings the row of "Turn on
    pictures" (data-ask-yes) into view above the footer. It is called after
    the pane is drawn, for a new answer and for a question kept from an
    earlier press on the same square, only after a press of Pictures (Show
    in the list: togglePictures passes reveal); and after the press of One
    square for a question kept for the square shown there (setView). A pane
    that follows an answer in the list (followPictures), Previous, Next or an
    answer on the face (goTo's timer), the switch to List, Try again and
    pictures switched on pass no reveal, so they scroll nothing. Nothing else
    in the editor scrolls but headInView (r5-k K4: after an answer that moves
    on, or an arrow key, the new square's heading, the least scroll, and none
    while it is in view; tests/mcp_app/test_mcp_r5k_sheet.py), no focus moves
    (the pane is a polite live region,
    so a screen reader hears the question), and the question's button keeps
    its words."""
    editor = _editor()
    ask = _between(editor, "  function questionInView(answer) {", "\n  }\n")
    # After the check: the scroll waits two frames for the layout to settle.
    for line in (
        "    if (!answer || !answer.needs_consent || typeof preview.scrollIntoView "
        "!== 'function') return;\n",
        "    later(() => later(() => { if (preview.isConnected) "
        "bringQuestionUp(); }));",
    ):
        assert line in ask, line
    bring = _between(editor, "  function bringQuestionUp() {", "\n  }\n")
    for line in (
        "    const foot = wrap.querySelector('.sheet-footer');\n",
        "    const margin = ((foot ? foot.offsetHeight : 0) + 8) + 'px';\n",
        "    preview.style.scrollMarginBottom = margin;\n",
        "    preview.scrollIntoView({ block: 'nearest' });\n",
        "    const yes = preview.querySelector('[data-ask-yes]');\n",
        "    const below = at.bottom + 8 - foot.getBoundingClientRect().top;\n",
        "    if (below > 0) "
        "window.scrollBy(0, Math.min(below, Math.max(0, at.top)));\n",
        "    else if (at.top < 0) window.scrollBy(0, at.top - 8);",
    ):
        assert line in bring, line
    assert bring.index("preview.scrollIntoView(") < bring.index("window.scrollBy(")
    assert "focus" not in (ask + bring).lower()
    # The row the second scroll lifts: the question's own button, marked.
    row = (
        "        OE.el('div', { class: 'row', style: 'justify-content:flex-end', "
        "'data-ask-yes': '' },\n"
        "          OE.el('button', { onclick: () => setPictures(true) }, "
        "'Turn on pictures')),\n"
    )
    assert row in editor
    # Drawn first, then brought into view when asked to (reveal): a new
    # answer, and one kept.
    load = _between(
        editor, "  async function loadPictures(h, reveal = false) {", "\n  }\n"
    )
    kept = (
        "    if (state.images.has(h)) { renderPreview(); if (reveal) "
        "questionInView(state.images.get(h)); return; }\n"
    )
    drawn = (
        "    if (state.shown === h) { renderPreview(answer); if (reveal) "
        "questionInView(answer); }"
    )
    assert kept in load and load.endswith(drawn)
    # The press of Pictures (Show in the list) alone reveals; every other call
    # gives the one argument, the window.
    toggle = _between(editor, "  function togglePictures(h) {", "\n  }\n")
    assert toggle.endswith("\n    loadPictures(h, true);")
    code = _code(editor)
    assert re.findall(r"loadPictures\(([^()]*)\)", code) == [
        "h",
        "h",
        "state.shown",
        "h, true",
        "h, reveal = false",
        "h",
        "h",
    ]
    follow = _between(editor, "  function followPictures(h) {", "\n  }\n")
    assert "setTimeout(() => { if (follow()) loadPictures(h); }, 300);" in follow
    turn = _between(editor, "  function goTo(h, focusKey) {", "\n  }\n")
    assert "&& !state.images.has(h)) loadPictures(h);" in turn
    assert "questionInView" not in turn
    # One square with the pane open: a question kept for the square shown is
    # brought into view, after goTo drew the face; List leaves the scroll.
    view = _between(editor, "  function setView(v, at) {", "\n  }\n")
    face = (
        "      goTo(at || state.at, at ? 'face' : null);\n"
        "      if (state.shown !== null && state.shown === state.at) "
        "questionInView(state.images.get(state.at));\n"
        "      return;\n"
    )
    assert face in view
    assert view.count("questionInView(") == 1
    # Its three calls and itself; the two scrolls of the editor are its own, and
    # r5-k K5's focus listener scrolls a focused control above the footer.
    assert code.count("questionInView(") == 4
    assert code.count("scrollIntoView(") == 2 and code.count("window.scrollBy(") == 3
    assert code.count("window.scrollBy(0, Math.min(over, at.top));") == 1
    head = _between(editor, "  function headInView() {", "\n  }\n")
    assert "top.scrollIntoView({ block: 'nearest' });" in head
    # After the check: while the next pictures load the pane keeps its height,
    # so the page does not get shorter and the face does not jump on Next.
    preview_draw = _between(editor, "  function renderPreview(shown) {", "\n  }\n")
    keep = (
        "      if (preview.childNodes.length && !preview.style.minHeight) "
        "preview.style.minHeight = preview.offsetHeight + 'px';\n"
    )
    assert keep in preview_draw
    assert "    preview.style.minHeight = '';" in preview_draw
    live = (
        "const preview = OE.el('div', { class: 'stack', id: previewId, "
        "'aria-live': 'polite' });"
    )
    assert live in editor
    assert "{ onclick: () => setPictures(true) }, 'Turn on pictures'))," in editor
    # The footer the margin is measured from is the sheet's pinned one.
    assert "OE.el('div', { class: 'sheet-footer spread' }, progress," in editor
    assert ".sheet-footer { position: sticky; bottom: 0;" in _read("common.js")


# --------------------------------------------------------------------------- F02


def test_f02_the_footer_says_why_the_finish_button_is_grey() -> None:
    """W.notYet, for squares and for points, stands in the table between the
    two lines of the ready button; its span is made once, beside
    everyDoneLine's. updateFinish keeps the two everyDone lines and then puts
    the new line in or takes it out: shown only while the sheet is not ready,
    not stale, has a square with no answer (count() < n: a save that is only
    pending leaves count() at n, a sheet with no square has n of 0, and a
    sheet whose two sets of answers are in the way is complete in count); on
    a sheet that already gave its result, whose answers are locked (r5-k K3),
    the same line says W.gradedBlank instead, never to answer a square
    (tests/mcp_app/test_mcp_r5k_sheet.py). While it is shown the finish
    button's aria-describedby names it (its span has an id of its own), and
    otherwise the button names no description, so a screen reader that
    reaches the grey button hears why. A stale sheet's footer is drawn before
    updateFinish is reached, with its own words. The lead is still two
    lines."""
    for unit in ("square", "point"):
        assert _said("notYet", unit) == NOT_YET.format(one=unit)
    assert _said("everyDone", "square") == EVERY_DONE
    assert _said("everyDoneGraded", "square") == EVERY_DONE_GRADED
    words = _words()
    keys = ("everyDone", "notYet", "everyDoneGraded")
    at = [words.index(f"    {key}: ") for key in keys]
    assert at == sorted(at)
    editor = _editor()
    made = (
        "  // WORDING: r5f F02\n"
        "  // WORDING: r5k K3 (gradedBlank on a graded sheet)\n"
        "  const notYetLine = OE.el('span', { class: 'small', id: 'oe-not-yet-' "
        "+ Math.random().toString(36).slice(2) },\n"
        "    gaveResult ? W.gradedBlank : W.notYet);\n"
    )
    assert made in editor
    assert editor.index("  const everyDoneLine = ") < editor.index(made)
    assert "asksForAnswers" not in editor
    finish = _between(editor, "  function updateFinish() {", "\n  }\n")
    lines = (
        "    if (ok && !done.contains(everyDoneLine)) done.prepend(everyDoneLine);\n",
        "    else if (!ok && done.contains(everyDoneLine)) everyDoneLine.remove();\n",
        "    const waits = !ok && !state.stale && count() < n;\n",
        "    if (waits && !done.contains(notYetLine)) done.prepend(notYetLine);\n",
        "    else if (!waits && done.contains(notYetLine)) notYetLine.remove();\n",
        "    if (waits) button.setAttribute('aria-describedby', notYetLine.id);\n",
        "    else button.removeAttribute('aria-describedby');\n",
    )
    assert "".join(lines) in finish
    assert _code(editor).count("W.notYet") == 1
    # The finish button's description is that line's alone, set only in
    # updateFinish (2 uses). The other 5 are on other controls: Reopen the
    # sheet (1, the stale footer's line, r5-i B7), the answer controls
    # lockAnswers locks (2, described by the graded notice or the stale line,
    # r5-k K3), and the two-map switch syncPairLock locks (2, W.pairLocked,
    # r5-k K2).
    code = _code(_script())
    assert _code(finish).count("aria-describedby") == 2
    assert code.count("aria-describedby") == 7
    assert "reopenButton.setAttribute('aria-describedby', staleLine.id);" in code
    assert "c.setAttribute('aria-describedby', why.id);" in code
    assert "c.removeAttribute('aria-describedby');" in code
    assert (
        "if (lock) blind.setAttribute('aria-describedby', pairLockLine.id); "
        "else blind.removeAttribute('aria-describedby');" in code
    )
    # What ready and complete count, which the new line's test leans on.
    assert "const ready = () => complete() && !pending() && !state.stale;" in editor
    assert "return n > 0 && count() === n && !state.mix;" in editor
    # A stale sheet: its own line and button, and updateFinish is not reached.
    progress = _between(editor, "  function updateProgress() {", "\n  }\n")
    stale = _between(progress, "    if (state.stale) {", "\n    }\n")
    assert "done.replaceChildren(staleLine, reopenButton);" in stale
    assert stale.rstrip().endswith("return;") and "updateFinish()" not in stale
    # r5-k K5: the footer's height is read again after the finish press is set.
    assert progress.rstrip().endswith("updateFinish();\n    fitFoot();")
    # r5-e H1 holds: the lead has no third line, and its old words stay gone.
    lead = _between(editor, "  const leadWords = {", "\n  };\n")
    assert lead.endswith("\n    cannot: null,")
    assert "No result until every" not in _code(_script())


def test_f02_a_plan_card_with_its_sheet_open_says_it_once() -> None:
    """F02 with track RESULT's F10 (after review): once its sheet is open in
    a plan card, the card's lead loses its second line (the "next" paragraph
    OE.lead draws; the plan's "Answer every one on the sheet to get the
    result.") with the button and the count, since the sheet's footer says it
    beside its grey finish button from then on; the lead's first line
    stays."""
    page = _read("result.html")
    opened = _between(page, "        opened = true;\n", "\n        if (hadFocus)")
    for line in (
        "        button.remove();\n",
        "        count.remove();  // the editor counts from here on\n",
        "        const answerEvery = root.querySelector("
        "':scope > .oe-lead > p.next');\n",
        "        if (answerEvery) answerEvery.remove();\n",
    ):
        assert line in opened, line
    assert _code(opened).count(".remove()") == 3
    # The lead is #root's own child, a paragraph per line named by its key.
    assert "const root = document.getElementById('root');" in page
    assert "  const lead = OE.lead(data.lead);\n  if (lead) root.append(lead);" in page
    assert "OE.el('p', { class: k }, s)" in _read("common.js")


# --------------------------------------------------------------------------- F12


def test_f12_no_records_is_a_question_of_its_own() -> None:
    """whatBare and whatBarePair end with the way out as a question and its
    answer, for squares and for points; the first sentence is as it was, and
    the old tail is gone from the code. The line is still the lead's one
    first line, chosen as before; a two-map sheet's holds neither clause its
    lead swaps, so the swap of r5-b J28 leaves it alone."""
    for unit in ("square", "point"):
        bare = _said("whatBare", unit)
        assert bare == WHAT_BARE.format(one=unit)
        assert bare.startswith("This sheet has no pictures and no places, so it")
        pair = _said("whatBarePair", unit)
        assert pair == WHAT_BARE_PAIR.format(one=unit)
        assert HIDDEN_ONE not in pair and HIDDEN_PAIR not in pair
    assert OLD_TAIL not in _code(_script())
    editor = _editor()
    assert (
        "const whatBare = data.has_place ? W.whatPlaced : (twoMap ? "
        "W.whatBarePair : W.whatBare);" in editor
    )
    assert "what: wrongFace ? null : picturesPossible ? W.what : whatBare," in editor


# --------------------------------------------------------------------------- F14


def test_f14_a_class_name_may_break_after_an_underscore() -> None:
    """common.js holds OE.softBreaks right after OE.el, with no lookbehind
    and no regular expression at all; OE.el takes what it returns (an array
    is flattened, an element appended as it is). On the face an answer's
    label (the name, then its key) and a class typed in go through it; the
    label's expression is as it was inside the call. The list's menu of more
    classes holds plain text (an option takes no element), and the rule that
    breaks a word with no underscore inside its cell stays."""
    common = _read("common.js")
    assert SOFT_BREAKS in common
    made = common.index("OE.el = (tag, attrs = {}, ...children) => {")
    assert made < common.index("OE.softBreaks = ") < common.index("OE.viewData = ")
    assert "(?<" not in SOFT_BREAKS and "RegExp" not in SOFT_BREAKS
    assert "  for (const child of children.flat()) {\n" in common
    append = (
        "    node.append(child instanceof Node ? child : "
        "document.createTextNode(String(child)));\n"
    )
    assert append in common
    editor = _editor()
    answers = _between(editor, "  function faceAnswers(w) {", "\n  }\n")
    label = (
        "    }, OE.softBreaks(a.label + (a.keys.length ? ' (' + a.keys[0] + ')' : "
        "'')))),\n"
    )
    assert label in answers
    typed = "onclick: () => facePick(current) }, OE.softBreaks(current)) : null);"
    assert typed in answers
    assert _code(editor).count("OE.softBreaks(") == 2
    assert "rest.map((a) => OE.el('option', { value: a.value }, a.label))" in answers
    assert ".face .answers button { overflow-wrap: anywhere; }\n" in common


# ------------------------------------------------------------------- the marks


def test_every_r5f_string_of_the_sheet_carries_its_wording_comment() -> None:
    """Each entry of OE.sheetWords this round changed or added has a WORDING
    line of r5-f above the tag of its round (nearest the entry), and so has
    the footer's new line where its span is made; no other entry got one."""
    tags = {
        "whatBare": ["// WORDING: r4m M6; r5b J11", "// WORDING: r5f F12"],
        "whatBarePair": ["// WORDING: r5b J28", "// WORDING: r5f F12"],
        "notYet": ["// WORDING: r5f F02"],
    }
    for key, want in tags.items():
        assert _tags_above(key) == want, key
    ours = sum(1 for want in tags.values() for tag in want if "r5f" in tag)
    assert _words().count("// WORDING: r5f ") == ours
    # One more in the editor: the span of the footer's new line.
    assert _script().count("// WORDING: r5f ") == ours + 1
