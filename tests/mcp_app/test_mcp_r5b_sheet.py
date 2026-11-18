# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Round r5-b (17 Nov 2026), track SHEET, as specs/r5-b-round.md gives it: the
label sheet's items of the user-side audit's build list. The views are read as
text, as they are on disk (no test here runs the scripts); the words of
``OE.sheetWords`` are read back from the file (:func:`_said`: a JS
concatenation of single-quoted strings and names read into its text), so a
missing space between two joined pieces fails here as it would on the face.

- J05: Previous, Next, Pictures and Street map are outlined small buttons, and
  the line that says nothing is shown yet is at body size.
- J06: a graded sheet says so in plain words at body size, and its lead keeps
  its first line only.
- J11: a sheet with no pictures and no places says it cannot show where a
  point is, and what to answer from; one with places, squares or points, says
  where to look.
- J13: the lead is three lines (what, the Cannot tell line, no result until
  every answer); the separate lines under it are gone, and the line on reading
  satellite pictures is in the Details fold.
- J14: the rule is at body size, "Rule: " in bold, right above the answers.
- J15: once the sheet changed elsewhere no dead button reads "Not saved"; the
  footer says whether answers were lost and what to do, and after this panel's
  own removal of a set of answers whose sheet could not be read again, that
  the read failed (not that the sheet changed elsewhere).
- J21: a sheet with two sets of answers names each set by what it holds, never
  by its column; the server's sentence is in the Details fold.
- J24 (sheet.js): the colour switch says "False colour".
- J25 (sheet.html): a sheet that cannot be opened is a card with a next step,
  the raw reason in a fold.
- J26: that no rule was set is said on the face, in the rule's place, from the
  first open (not on a graded sheet; a two-map sheet's names the two maps); the
  pictures pane no longer says it.
- J28: a two-map sheet's switch, instruction and first line in plain words; the
  first line says the maps' answers are shown while the switch is unticked.
- J29 (a, sheet.js): the line beside the finish button has its object.

Round r5-e (track SHEET, tests/mcp_app/test_mcp_r5e_sheet.py) changed some of
these words: the lead is two lines (H1: no noResult, a shorter first line and
Cannot tell line, the hidden sentences now clauses), the rule's scope is in the
Details fold (H2), the finish line is its second sentence alone and the graded
notice leaves its reason to the fold (H5). The tests here pin the new words
where they pinned the old.
"""

from __future__ import annotations

import re

from olmoearth_agent.mcp_app import imagery, sheet, views

#: J06: the notice on a sheet already graded (W.graded); r5-e H5 moved its
#: middle clause, the reason, to the Details fold (W.gradedWhy).
GRADED = (
    "This sheet already gave its result. Do not change an answer now; a second "
    "opinion needs another person and a fresh sheet."
)
#: J11: the first line of a sheet with no pictures and no places (W.whatBare);
#: {one} is the sheet's unit word. r5-f F12: the way out is a question of its
#: own (it was the tail "; with none, stop and tell Claude.").
WHAT_BARE = (
    "This sheet has no pictures and no places, so it cannot show where a {one} is. "
    "Answer only from your own records of each {one}, without looking at the map. "
    "No records? Stop and tell Claude."
)
#: J13: the lead's third line (W.noResult); r5-e H1 removed it (the greyed
#: finish button and the counter say it). The script joined the unit word in
#: as a name, so the pieces around it are what must be gone.
NO_RESULT = ("No result until every", "You can change an answer until you press")
#: J13: in the Details fold of a sheet that can show pictures (W.reading).
READING = "Your answers are a reading of satellite pictures, not a visit on site."
#: J15: beside "Reopen the sheet" with no answer waiting; with some, the other
#: branch of W.changedElsewhere says how many were not saved.
CHANGED_NONE = "This sheet was changed somewhere else. Reopen the sheet to carry on."
#: J15: in its place after this panel removed a set of answers and could not
#: read the sheet again (W.notReadAgain).
NOT_READ_AGAIN = (
    "This panel could not read the sheet again. Reopen the sheet to carry on."
)
#: J25: a sheet that cannot be opened or read (sheet.html).
NOT_OPENED = "This label sheet could not be opened. Ask Claude to open it again."
#: J26: no rule on the page copy (W.noRule), for squares.
NO_RULE = (
    "No rule was set for this map. Decide your rule before the first answer and use "
    "it for every square."
)
#: J26: the same on a two-map sheet (W.noRulePair), for squares.
NO_RULE_PAIR = (
    "No rule was set for these two maps. Decide your rule before the first answer "
    "and use it for every square."
)
#: J28: what the lead's first line ends with on a one-map sheet (W.hiddenOne),
#: and what a two-map sheet's says in its place (W.hiddenPair), or while its
#: switch is unticked (W.shownPair); r5-e H1: clauses of the first line, no
#: longer sentences of their own.
HIDDEN_ONE = ", without looking at the map."
HIDDEN_PAIR = ", without looking at the two maps."
SHOWN_PAIR = ". The two maps' answers are shown now."
#: J29: beside the finish button once every answer is saved (W.everyDone);
#: r5-e H5: its second sentence alone.
EVERY_DONE = "After you press, do not change an answer."

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


# --------------------------------------------------------------------------- J05


def test_j05_the_faces_buttons_are_outlined() -> None:
    """Previous, Next, Pictures and Street map are outlined small buttons
    (they were bare words); the line that says nothing is shown yet is at
    body size and colour, in the words it had; the list's row buttons stay
    text buttons, and the answers stay above the pictures."""
    editor = _editor()
    pager = _between(editor, "  const prevButton = ", "  const faceExtra = ")
    below = _between(
        editor,
        "  const facePicturesButton = ",
        "  const facePictures = ",
    )
    for text in (
        "class: 'small', 'data-focus': 'prev', 'aria-label': W.prevLabel,",
        "class: 'small', 'data-focus': 'next', 'aria-label': W.nextLabel,",
    ):
        assert text in pager, text
    for text in (
        "class: 'small', 'aria-controls': previewId,",
        "class: 'small', onclick: () => { if (state.at) openMap(state.at); },",
    ):
        assert text in below, text
    assert "'link'" not in pager and "'link'" not in below
    # The pictures-off line: body size and colour (r5-e H1: it starts with the
    # press, no longer "Nothing to look at yet").
    off = "const picturesOff = picturesPossible ? OE.el('div', {}, W.picturesOff)"
    assert off + " : null;" in editor
    assert _said("picturesOff", "square") == (
        "Press Pictures to see satellite pictures; you are asked first what is sent."
    )
    # The list's row buttons stay text buttons.
    rows = _between(editor, "  function renderList(", "\n  }\n")
    for text in (
        "class: 'link', 'data-focus': 'n:' + h,",
        "class: 'link', 'data-focus': 'i:' + h,",
    ):
        assert text in rows, text
    # The order on the face stays: the answers, then Pictures and Street map.
    face = _between(editor, "    faceEl.append(", "  } else {")
    assert face.index("faceAnswerArea,") < face.index("facePicturesButton,")


# --------------------------------------------------------------------------- J06


def test_j06_a_graded_sheet_says_so_in_plain_words() -> None:
    """The notice in words that are on a button ("graded" was on none), at
    body size; while it shows, the lead keeps its first line only: no Cannot
    tell line (r5-e H1: and no third line on any sheet)."""
    assert _said("graded", "square") == GRADED
    editor = _editor()
    assert "{ class: 'card summary', role: 'note' }, W.graded) : null;" in editor
    assert "card summary small" not in editor
    for text in (
        "next: gradedNote ? null : cannotLineText || null,",
        "cannot: null,",
    ):
        assert text in editor, text
    code = _code(_script())
    assert "has been graded" not in code and "fresh blind sheet" not in code


# --------------------------------------------------------------------------- J11


def test_j11_a_sheet_with_no_pictures_and_no_places_says_so() -> None:
    """One line replaced, no control added: the sheet says it cannot show
    where a point is, what to answer from, and to stop with nothing to answer
    from; a sheet with places keeps its own first line, squares or points
    (the "no places" line is chosen by places alone: a sheet of squares with
    places draws Street map on every square)."""
    for unit in ("square", "point"):
        assert _said("whatBare", unit) == WHAT_BARE.format(one=unit)
    assert "    // WORDING: r4m M6; r5b J11\n    whatBare: " in _words()
    assert "say what is there" not in _code(_script())
    editor = _editor()
    assert (
        "const whatBare = data.has_place ? W.whatPlaced : (twoMap ? "
        "W.whatBarePair : W.whatBare);" in editor
    )
    assert "W.one === 'point' && data.has_place" not in editor
    assert "what: wrongFace ? null : picturesPossible ? W.what : whatBare," in editor


# --------------------------------------------------------------------------- J13


def test_j13_the_lead_is_three_lines() -> None:
    """What this is; the Cannot tell line (or a certificate's own; none on a
    two-map sheet). The separate lines under the lead are gone, with the words
    of the old second and third lines. r5-e H1: the third line (no result
    until every answer, W.noResult) is gone too: the lead is two lines."""
    words = _words()
    for gone in ("next", "nextBare", "cannot", "cannotBare", "changeLine", "noResult"):
        assert f"\n    {gone}: " not in words, gone
    for gone in NO_RESULT:
        assert gone not in _code(_script()), gone
    editor = _editor()
    lead = _between(editor, "  const leadWords = {", "\n  };\n")
    for text in (
        "what: wrongFace ? null : picturesPossible ? W.what : whatBare,",
        "next: gradedNote ? null : cannotLineText || null,",
        "cannot: null,",
    ):
        assert text in lead, text
    assert "W.noResult" not in editor and "finishWords" not in editor
    assert "OE.lead(leadWords)" in editor
    layout = _between(editor, "  wrap.append(", "  // One listener for the keys")
    for gone in ("cannotLine,", "changeLine,"):
        assert gone not in layout, gone
    for gone in ("const cannotLine = ", "const changeLine = ", "W.changeLine("):
        assert gone not in editor, gone
    code = _code(_script())
    for gone in ("Press Pictures, turn them on", "No result while you check"):
        assert gone not in code, gone
    # What stands before the answers, in order.
    order = [
        "OE.el('h1', {}, 'Label sheet'), partLine)",
        "gradedNote,",
        "leadLines,",
        "faceInstruction,",
        "ruleNote,",
        "switchRow,",
        "viewToggle, filterBox)",
        "notice,",
        "ruleBlock,",
        "noRuleLine,",
        "area,",
        "details),",
    ]
    places = [layout.index(text) for text in order]
    assert places == sorted(places)


def test_j13_the_reading_line_is_in_the_fold() -> None:
    """That the answers are a reading of satellite pictures, not a visit on
    site, is in the sheet's one Details fold, only on a sheet that can show
    pictures; it is no longer on the lead."""
    assert _said("reading", "square") == READING
    editor = _editor()
    note = "const readingNote = picturesPossible ? OE.el('div', { class: 'small' },"
    assert note + " W.reading) : null;" in editor
    fold = _between(editor, "  const foldLabel = 'Details';", "\n\n")
    assert "clearNote, examplesNote, readingNote, mixNote,\n" in fold
    assert editor.count("OE.fold(") == 1
    assert "W.reading" not in _between(editor, "  const leadWords = {", "\n  };\n")


# --------------------------------------------------------------------------- J14


def test_j14_the_rule_is_body_size_right_above_the_answers() -> None:
    """The rule's own line has no small class and says "Rule: " in bold; the
    line after it keeps its size; the block stands after the notice and
    directly before the answers (the face or the list). r5-e H2: the scope is
    no longer in the block but in the Details fold (scopeNote)."""
    editor = _editor()
    block = _between(editor, "  const ruleBlock = pageRule ?", "\n  const clearNote")
    assert "OE.el('div', {}, OE.el('strong', {}, 'Rule: '), pageRule)," in block
    assert "'Rule: ' + pageRule" not in editor
    assert "cloudOnFace ? OE.el('div', { class: 'small' }, W.cloudLine)" in block
    assert "data.rule_scope" not in block
    assert "OE.el('div', { class: 'small' }, data.rule_scope) : null;" in editor
    layout = _between(editor, "  wrap.append(", "  // One listener for the keys")
    above = "      notice,\n      ruleBlock,\n      noRuleLine,\n      area,\n"
    assert above in layout
    assert layout.index("leadLines,") < layout.index("ruleBlock,")


# --------------------------------------------------------------------------- J15


def test_j15_a_sheet_changed_elsewhere_shows_no_dead_button() -> None:
    """Once the sheet changed under the page: the Save button is hidden (it
    read "Not saved", greyed), the note beside the counter says how many
    answers are not saved, and the line beside "Reopen the sheet" says, at
    body size, what happened, whether answers were lost and what to do. The
    server's refusal prefix is not touched."""
    script = _script()
    editor = _editor()
    progress = _between(editor, "  function updateProgress() {", "\n  }\n")
    stale = _between(progress, "    if (state.stale) {", "      return;\n    }")
    for text in (
        "saveButton.hidden = true;",
        "savedNote.hidden = false;",
        "savedNote.textContent = left ? OE.counted(left, 'answer') + ' not saved'",
        " : 'Not saving';",
        "counterNote.textContent = '';",
        "staleLine.textContent = state.staleWhy === 'read' ? W.notReadAgain : "
        "W.changedElsewhere(left);",
        "done.replaceChildren(staleLine, reopenButton);",
    ):
        assert text in stale, text
    assert "'Not saved'" not in progress
    assert "saveButton.textContent = 'Not saved'" not in script
    assert "const staleLine = OE.el('span', { class: 'error' });" in editor
    assert "{ class: 'small error' }" not in progress
    words = _words()
    for text in (
        "'This sheet was changed somewhere else, so ' + k",
        "(k === 1 ? ' answer here was' : ' answers here were')",
        "' not saved. Reopen the sheet, then answer '",
        "(k === 1 ? 'that ' + one : 'those ' + many) + ' again.'",
        f": '{CHANGED_NONE}'),",
    ):
        assert text in words, text
    prefix = "This sheet changed since the panel opened it"
    # After this panel's own removal of a set of answers, a failed read again
    # is not a change elsewhere: markStale('read') there only, and the footer
    # says the read failed (W.notReadAgain).
    assert _said("notReadAgain", "square") == NOT_READ_AGAIN
    stale_mark = _between(editor, "  function markStale(why) {", "\n  }\n")
    assert "    state.staleWhy = why || 'changed';\n" in stale_mark
    clear = _between(
        editor,
        "  async function clearAndOpen(column, to) {",
        "\n  }\n",
    )
    removed = _between(clear, "replaceEditor(fresh, 'Removed: ", "reopenToast(")
    assert "markStale('read');" in removed
    assert _code(editor).count("markStale('read');") == 1
    assert f"OE.CHANGED_UNDER_PAGE = '{prefix}';" in script
    assert sheet.CHANGED_UNDER_PAGE.startswith(prefix + ";")


# --------------------------------------------------------------------------- J21


def test_j21_two_sets_of_answers_are_named_by_what_they_hold() -> None:
    """The notice says the sheet holds two sets of answers; its button, an
    outlined small one, says which set goes and which stays; the server's own
    sentence is in the Details fold; the toasts name the set removed. No
    string names a column."""
    editor = _editor()
    assert (
        "const answersOf = (c) => (c === 'wrong' ? 'right-or-wrong marks' : "
        "'classes picked');" in editor
    )
    progress = _between(editor, "  function updateProgress() {", "\n  }\n")
    for text in (
        "'This sheet holds two sets of answers, so no result can come from it. '),",
        "OE.el('button', { class: 'small', onclick: () => "
        "clearAndOpen(mix.column, face) },",
        "'Remove the ' + answersOf(mix.column) + ', keep the ' + answersOf(kept))",
        "const kept = mix.column === 'wrong' ? 'reference_class' : 'wrong';",
        "mixNote.hidden = !state.mix;",
        "'The estimate would refuse this sheet: ' + state.mix.problem + '.'",
    ):
        assert text in progress, text
    fold = _between(editor, "  const foldLabel = 'Details';", "\n\n")
    assert "mixNote,\n" in fold
    notice = _between(progress, "    if (state.mix) {", "\n    }\n")
    assert "mix.problem" not in notice
    clear = _between(
        editor,
        "  async function clearAndOpen(column, to) {",
        "\n  }\n",
    )
    for text in (
        "replaceEditor(fresh, 'Removed: the ' + answersOf(column) + '.');",
        "reopenToast('The ' + answersOf(column) + ' were removed, but the sheet "
        "could not be read again: '",
        "'Press the button again to remove the ' + answersOf(column) + '.', true);",
    ):
        assert text in clear, text
    code = _code(_script())
    for gone in (
        "Clear the ' + mix.column",
        "column was cleared",
        "Press the link again",
        "'The estimate would refuse this sheet: ' + mix.problem",
    ):
        assert gone not in code, gone


def test_j21_the_server_names_the_two_columns_the_panel_tells_apart() -> None:
    """The column the server says to clear is ``wrong`` (the panel's
    "right-or-wrong marks") or ``reference_class`` (its "classes picked"),
    whichever is not the one being labelled."""
    header = ["window_index", "map_class", "wrong", "reference_class"]
    body = [["0", "water", "1", ""], ["1", "trees", "", "trees"]]
    design = {"design": "random", "population": {"score_kind": "class_map"}}
    assert sheet._mix_problem("class", "class", design, header, body) == (
        "it also has 1 label in its wrong column",
        "wrong",
    )
    assert sheet._mix_problem("wrong", "class", design, header, body) == (
        "it also has 1 label in its reference_class column",
        "reference_class",
    )


# --------------------------------------------------------------------------- J24


def test_j24_the_colour_switch_says_false_colour() -> None:
    """The switch of the pictures (the sheet's pane and a review set's, both
    OE.datesBlock) says "False colour"; the caption under the pictures still
    says what the colours show."""
    dates = _between(_script(), "OE.datesBlock = (", "\n};")
    kinds = "const kinds = [['nir', 'False colour'], ['true', 'True colour']];"
    assert "  // WORDING: r5b J24\n  " + kinds in dates
    for name in ("sheet.js", "result.html"):
        assert "Near-infrared false colour" not in _code(_read(name)), name
    assert "Near-infrared false colour shows vegetation red" in imagery.CAPTION


# --------------------------------------------------------------------------- J25


def test_j25_a_sheet_that_cannot_be_opened_names_a_next_step() -> None:
    """Both dead ends of the sheet view (the tool's result is an error, the
    read of the rows failed) show a card with the title, one plain line that
    says what to do, and the raw reason in a closed Details fold."""
    page = _read("sheet.html")
    assert f"// WORDING: r5b J25\nconst NOT_OPENED = '{NOT_OPENED}';" in page
    card = _between(page, "function notOpened(reason) {", "\n}\n")
    for text in (
        "OE.el('h1', {}, 'Label sheet'),",
        "OE.el('div', {}, NOT_OPENED),",
        "why ? OE.fold('Details', OE.el('div', { class: 'small', "
        "style: 'overflow-wrap:anywhere' }, why)) : null));",
    ):
        assert text in card, text
    shown = _between(page, "async function show(app, params) {", "\n}\n")
    assert "notOpened(message);" in shown
    assert "notOpened(err && err.message);" in shown
    for gone in ("class: 'card error'", "The sheet could not be read: '"):
        assert gone not in page, gone


# --------------------------------------------------------------------------- J26


def test_j26_no_rule_is_said_on_the_face_from_the_first_open() -> None:
    """With no rule on the page copy, on a sheet that can show pictures, one
    body-size line in the rule's place says none was set and what to do; the
    pictures pane no longer says it, and a rule the pictures bring hides it.
    Not on a graded sheet (J06: nothing under its notice invites an answer);
    a two-map sheet's line names the two maps."""
    assert _said("noRule", "square") == NO_RULE
    assert _said("noRule", "point").endswith("use it for every point.")
    assert _said("noRulePair", "square") == NO_RULE_PAIR
    assert _said("noRulePair", "point").endswith("use it for every point.")
    editor = _editor()
    line = (
        "  const noRuleLine = !pageRule && picturesPossible && !gradedNote\n"
        "    ? OE.el('div', { class: 'oe-no-rule' }, twoMap ? W.noRulePair : "
        "W.noRule) : null;\n"
    )
    assert line in editor
    # gradedNote is declared before the line that reads it.
    assert editor.index("  const gradedNote = ") < editor.index(line)
    layout = _between(editor, "  wrap.append(", "  // One listener for the keys")
    assert "      ruleBlock,\n      noRuleLine,\n      area,\n" in layout
    pane = _between(editor, "  function renderPreview(shown) {", "\n  }\n")
    assert "W.noRule" not in pane
    assert "if (noRuleLine) noRuleLine.hidden = true;" in pane
    code = _code(editor)
    assert len(re.findall(r"\bW\.noRule\b", code)) == 1
    assert code.count("W.noRulePair") == 1
    assert "No labelling rule is set" not in _code(_script())


# --------------------------------------------------------------------------- J28


def test_j28_a_two_map_sheet_in_plain_words() -> None:
    """The switch says what it hides and to keep it on; the instruction names
    the button the face has (Cannot tell) and says "there", not "on the
    ground"; the lead's first line says both maps' answers are hidden, and
    that they are shown while the switch is unticked (by extraLines' own
    test); with no pictures and no places it names the maps too."""
    script = _script()
    assert (
        "// WORDING: r4i F11; r5b J28\n  const blindText = 'Hide what the two maps "
        "say (keep it on while you answer)';" in script
    )
    editor = _editor()
    told = _between(editor, "  function faceInstructionText() {", "\n  }\n")
    assert (
        "if (twoMap) return 'For each ' + one + ', pick the class you see there; "
        "press Cannot tell when you cannot judge it.';" in told
    )
    assert _said("hiddenOne", "square") == HIDDEN_ONE
    assert _said("hiddenPair", "square") == HIDDEN_PAIR
    # The one-map clause is the one the lead's first lines end with (r5-e H1:
    # a clause after a comma, no longer a sentence after a space).
    for key in ("what", "whatPlaced"):
        assert _said(key, "square").endswith(HIDDEN_ONE), key
    swap = "leadWords.what.replace(W.hiddenOne, W.hiddenPair)"
    assert "if (twoMap && leadWords.what) leadWords.what = " + swap + ";" in editor
    # The no-pictures, no-places line on a two-map sheet: the maps (r5-f F12:
    # the clause now ends its sentence, "the map." once in the line).
    for unit in ("square", "point"):
        bare = WHAT_BARE.format(one=unit)
        assert bare.count("the map.") == 1
        assert _said("whatBarePair", unit) == bare.replace("the map.", "the maps.")
    # Unticked, the first line says the answers are shown; ticked, hidden.
    assert _said("shownPair", "square") == SHOWN_PAIR
    for text in (
        "  const pairWhat = twoMap && leadLines ? leadLines.querySelector('p.what')"
        " : null;\n",
        "  const pairShown = pairHidden.replace(W.hiddenPair, W.shownPair);\n",
    ):
        assert text in editor, text
    drawn = _between(editor, "  function render(focusKey) {", "\n  }\n")
    assert (
        "    if (pairWhat) pairWhat.textContent = !state.blind && state.revealed ? "
        "pairShown : pairHidden;\n" in drawn
    )
    lines = _between(editor, "  function extraLines(h) {", "\n  }\n")
    assert "const unblinded = !state.blind && state.revealed;" in lines
    code = _code(script)
    for gone in ("Blind: hide which classes", "on the ground there", "pick ? when"):
        assert gone not in code, gone


# --------------------------------------------------------------------------- J29


def test_j29_the_finish_line_says_what_must_not_change() -> None:
    """(a), sheet.js: "Answers must not change after you press." had no
    object; the line now says not to change an answer (r5-e H5: that
    sentence alone)."""
    assert _said("everyDone", "square") == EVERY_DONE
    assert "    // WORDING: r4h S2 A/F17; r5b J29\n    everyDone: " in _words()
    assert "Answers must not change after you press" not in _code(_script())
    assert "All answered and saved." not in _code(_script())
    # r5-d R06: on a sheet that already gave its result, W.everyDoneGraded.
    line = "const everyDoneLine = OE.el('span', { class: 'small' },\n"
    assert line in _editor()
    assert "? W.everyDoneGraded : W.everyDone);" in _editor()


# ------------------------------------------------------------------- the marks


def test_every_r5b_string_of_the_sheet_carries_its_wording_comment() -> None:
    """Each entry of OE.sheetWords this round added or changed has its
    WORDING line right above it, and so has each string of the editor
    (noResult is gone since r5-e H1; r5-e's own lines stand above these)."""
    lines = _words().split("\n")
    tags = {
        "hiddenOne": "r5b J28",
        "hiddenPair": "r5b J28",
        "shownPair": "r5b J28",
        "reading": "r5b J13",
        "whatBare": "r4m M6; r5b J11",
        "whatBarePair": "r5b J28",
        "graded": "r5b J06",
        "changedElsewhere": "r5b J15",
        "notReadAgain": "r5b J15",
        "noRule": "r5b J26",
        "noRulePair": "r5b J26 J28",
        "everyDone": "r4h S2 A/F17; r5b J29",
    }
    for key, tag in tags.items():
        (at,) = [i for i, line in enumerate(lines) if line.startswith(f"    {key}: ")]
        assert lines[at - 1].strip() == f"// WORDING: {tag}", key
    editor = _editor()
    for text in (
        "// WORDING: r5b J21\n  const answersOf = ",
        "// WORDING: r5b J15\n      savedNote.textContent = ",
        "// WORDING: r5b J21\n      notice.append(",
        "// WORDING: r5b J21\n          replaceEditor(fresh, 'Removed: ",
        "// WORDING: r5b J21\n          reopenToast('The ' + answersOf(column)",
        "// WORDING: r5b J21\n      OE.toast('Not cleared: the sheet was saved",
        "// WORDING: r4h S1 S3; r5b J28 (the two-map sheet's)\n  function face",
    ):
        assert text in editor, text
