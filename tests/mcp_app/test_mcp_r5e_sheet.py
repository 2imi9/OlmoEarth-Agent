# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Round r5-e (17 Nov 2026), track SHEET, as specs/r5-e-round.md gives it: the
label sheet's items H1 to H5 of the design build list
(specs/r5-e-design/build_list.md). The views are read as text, as they are on
disk (no test here runs the scripts); the words of ``OE.sheetWords`` are read
back from the file (:func:`_said`: a JS concatenation of single-quoted strings
and names read into its text), so a missing space between two joined pieces
fails here as it would on the face.

- H1: the lead under "Label sheet" is two short lines: what to do in one
  clause ("Say what you see in each square's satellite pictures, without
  looking at the map."), and the Cannot tell line without its reason; the
  third line (no result until every answer) is gone; a two-map sheet still
  swaps its clause; the class line with the period and the pictures line
  start with what to do.
- H2: the rule, the cloud line on a face that has one and the example
  pictures sit in one box with a border (``card stack oe-rule``); who set the
  rule and when is in the Details fold, and so is, right after it (after
  review), the line that any other map needs its own rule: alone under the
  rule it had nothing to refer to.
- H3: "Other class..." is the last cell of the answers' group, a button as
  large as every answer, dashed (common.js ``.oe-other``), no longer a text
  link on a line of its own.
- H4: in the list, each row's pictures button reads "Show" under the head
  "Pictures" and, after review, "Hide" on the row whose pictures are open (a
  press there closes them); its name starts with the word it shows ("Show
  pictures: square k"); Street map is said once, in the open pane's head; a
  sheet with places and no pictures keeps its per-row Street map; while
  pictures are off, the line under the list names the list's own press.
- H5: the finish line is its second sentence alone, and the graded notice is
  two lines, its reason in the Details fold.
"""

from __future__ import annotations

import re

from olmoearth_agent.mcp_app import views

#: H1: the lead's first line on a sheet that can show pictures (W.what).
WHAT = (
    "Say what you see in each {one}'s satellite pictures, without looking at the "
    "map."
)
#: H1: the clause what and whatPlaced end with (W.hiddenOne), what a two-map
#: sheet says in its place (W.hiddenPair), and, while its switch is unticked,
#: what ends the sentence there instead (W.shownPair).
HIDDEN_ONE = ", without looking at the map."
HIDDEN_PAIR = ", without looking at the two maps."
SHOWN_PAIR = ". The two maps' answers are shown now."
#: H1: a sheet with places and no pictures (W.whatPlaced).
WHAT_PLACED = (
    "This sheet has no pictures. See where each {one} is, then say what was there "
    "from your own imagery or records, without looking at the map."
)
#: H1: the face's one Cannot tell line (W.cannotTellLine); the reason stays in
#: the fold (W.cannotTell).
CANNOT_TELL_LINE = "Cannot judge a {one}? Press Cannot tell; do not guess."
#: H1: the class face's line with the map's period (W.classWhat(period)).
CLASS_WITH_PERIOD = "Pick what was there during {period}, using the map's class names."
#: H1: the class face's line with no period known, unchanged and in the fold
#: (r5-d R08).
CLASS_NO_PERIOD = (
    "For each {one}, pick what was there in the period the map covers, using the "
    "map's class names; ask whoever made the map for that period."
)
#: H1: the line in the pane's place while pictures are off (W.picturesOff).
PICTURES_OFF = (
    "Press Pictures to see satellite pictures; you are asked first what is sent."
)
#: H4, after review: the same line in the list, where the press is a row's
#: Show under the head "Pictures" (W.picturesOffList).
PICTURES_OFF_LIST = (
    "Press Show under Pictures to see a {one}'s satellite pictures; you are asked "
    "first what is sent."
)
#: H5: beside the ready finish button (W.everyDone).
EVERY_DONE = "After you press, do not change an answer."
#: H5: the graded notice (W.graded) and its reason, in the fold (W.gradedWhy).
GRADED = (
    "This sheet already gave its result. Do not change an answer now; a second "
    "opinion needs another person and a fresh sheet."
)
GRADED_WHY = (
    "A change made after seeing where the map disagrees makes the result invalid."
)
#: R06 (r5-d), unchanged: beside the ready finish button of a graded sheet.
EVERY_DONE_GRADED = "This sheet already gave its result."
#: The period the fixtures of the older tests use.
PERIOD = "1 Apr 2024 to 30 Jun 2024"

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


# --------------------------------------------------------------------------- H1


def test_h1_the_lead_is_two_short_lines() -> None:
    """The first line says what to do in one clause, the second is the
    Cannot tell line without its reason (in the fold), and no third line is
    drawn: noResult and finishWords are gone. Blind labelling is unchanged:
    the face says "without looking at the map" and the fold keeps the blind
    line. The class line with the period and the pictures line start with
    what to do."""
    for unit in ("square", "point"):
        assert _said("what", unit) == WHAT.format(one=unit)
        assert _said("whatPlaced", unit) == WHAT_PLACED.format(one=unit)
        assert _said("cannotTellLine", unit) == CANNOT_TELL_LINE.format(one=unit)
    assert _said("picturesOff", "square") == PICTURES_OFF
    # classWhat: with the period, what to do first; without it, unchanged.
    block = _between(_words(), "    classWhat: (period) => (period", "),\n")
    known, unknown = block.split("\n      : ")
    with_period = known.split("\n      ? ")[1]
    assert _concat(with_period, {"one": "square", "period": PERIOD}) == (
        CLASS_WITH_PERIOD.format(period=PERIOD)
    )
    assert _concat(unknown, {"one": "point"}) == CLASS_NO_PERIOD.format(one="point")
    # What a guess costs stays in the fold's sentence (cannotTellRest).
    assert "Do not guess: a guess can make the result wrong." in _code(_words())
    # The lead: two lines at most; no third.
    editor = _editor()
    lead = _between(editor, "  const leadWords = {", "\n  };\n")
    # The slice ends right before the object's closing line, so this is its
    # whole last line: nothing is joined to it and no entry follows it.
    assert lead.endswith("\n    cannot: null,")
    assert "\n    noResult: " not in _words()
    assert "W.noResult" not in editor and "finishWords" not in editor
    code = _code(_script())
    for gone in (
        "No result until every",
        "You can change an answer until you press",
        "The map's answer is hidden",
        "answers are hidden",
        "Nothing to look at yet",
        "a Cannot tell makes the range wider",
    ):
        assert gone not in code, gone
    # The blind line stays in the fold (the class face's and a seen sheet's).
    assert "const sentLine = seen || face === 'class' ? W.blindSeen" in editor


def test_h1_a_two_map_sheet_still_swaps_its_clause() -> None:
    """The swap of r5-b J28 works on the new words: what and whatPlaced end
    with hiddenOne (once), so a two-map sheet's line ends "without looking at
    the two maps", and with the switch unticked the sentence ends there and
    the next says the two maps' answers are shown."""
    assert _said("hiddenOne", "square") == HIDDEN_ONE
    assert _said("hiddenPair", "square") == HIDDEN_PAIR
    assert _said("shownPair", "square") == SHOWN_PAIR
    shown = {
        "what": (
            "Say what you see in each square's satellite pictures. The two maps' "
            "answers are shown now."
        ),
        "whatPlaced": (
            "This sheet has no pictures. See where each square is, then say what was "
            "there from your own imagery or records. The two maps' answers are shown "
            "now."
        ),
    }
    for key, want in shown.items():
        said = _said(key, "square")
        assert said.endswith(HIDDEN_ONE) and said.count(HIDDEN_ONE) == 1, key
        pair = said.replace(HIDDEN_ONE, HIDDEN_PAIR)
        assert pair.endswith(HIDDEN_PAIR), key
        assert pair.replace(HIDDEN_PAIR, SHOWN_PAIR) == want, key
    editor = _editor()
    assert (
        "if (twoMap && leadWords.what) leadWords.what = "
        "leadWords.what.replace(W.hiddenOne, W.hiddenPair);" in editor
    )
    swap = "  const pairShown = pairHidden.replace(W.hiddenPair, W.shownPair);\n"
    assert swap in editor
    # whatBare has no hidden clause: a two-map sheet has its own whatBarePair.
    assert HIDDEN_ONE not in _said("whatBare", "square")


# --------------------------------------------------------------------------- H2


def test_h2_the_rule_sits_in_one_box_and_its_scope_in_the_fold() -> None:
    """The rule block is a card (a box with a border) that holds the rule, the
    cloud line on a face that has one (a plain small line) and the example
    pictures; the scope is a small line in the Details fold, after the class
    line with no period, and right after it there (after review) the
    kelp-only line: never without the scope, which names the maps the rule is
    for, and, as before, only where the face has no cloud line; the example
    captions, which check.html copies word for word, stay."""
    editor = _editor()
    block = _between(editor, "  const ruleBlock = pageRule ?", "\n  const clearNote")
    assert block.startswith(
        "  const ruleBlock = pageRule ? OE.el('div', { class: 'card stack oe-rule' },\n"
        "    OE.el('div', {}, OE.el('strong', {}, 'Rule: '), pageRule),\n"
        "    cloudOnFace ? OE.el('div', { class: 'small' }, W.cloudLine) : null,\n"
    )
    assert block.endswith("W.cloudLine) : null,\n    ruleExamples) : null;")
    assert "data.rule_scope" not in block and "faint" not in block
    assert "data.rule_limit" not in block
    assert (
        "  const scopeNote = typeof data.rule_scope === 'string' && data.rule_scope ? "
        "OE.el('div', { class: 'small' }, data.rule_scope) : null;\n" in editor
    )
    assert (
        "  const limitNote = pageRule && scopeNote && !cloudOnFace && typeof "
        "data.rule_limit === 'string' && data.rule_limit ? OE.el('div', { class: "
        "'small' }, data.rule_limit) : null;\n" in editor
    )
    assert editor.index("  const scopeNote = ") < editor.index("  const limitNote = ")
    fold = _between(editor, "  const foldLabel = 'Details';", "\n\n")
    assert "    classNoPeriod, scopeNote, limitNote, gradedWhy,\n" in fold
    layout = _between(editor, "  wrap.append(", "  // One listener for the keys")
    assert "scopeNote" not in layout and "limitNote" not in layout
    assert "      notice,\n      ruleBlock,\n      noRuleLine,\n      area,\n" in layout
    script = _script()
    assert "OE.EXAMPLE_PRESENT = 'Present under this rule';" in script
    assert "OE.EXAMPLE_ABSENT = 'Absent under this rule';" in script


# --------------------------------------------------------------------------- H3


def test_h3_other_class_is_a_cell_of_the_answers() -> None:
    """On a face that takes any class, "Other class..." is appended to the
    answers' group as a button of class oe-other (common.js draws it the size
    of every answer, dashed), no longer a text link in a line of its own; its
    focus key and its press are as they were."""
    editor = _editor()
    answers = _between(editor, "  function faceAnswers(w) {", "\n  }\n")
    assert (
        "      group.append(OE.el('button', {\n"
        "        type: 'button', class: 'oe-other', 'data-focus': 'other:' + h, "
        "onclick: () => { state.other = h; render('o:' + h); },\n"
        "      }, 'Other class...'));\n" in answers
    )
    assert "class: 'link', 'data-focus': 'other:'" not in editor
    assert "parts.push(OE.el('div', {}, OE.el('button'" not in answers
    assert "class: 'answers row', role: 'group'," in answers
    common = _read("common.js")
    rule = [line for line in common.split("\n") if "button.oe-other" in line]
    assert rule and "border-style: dashed" in rule[0]


# --------------------------------------------------------------------------- H4


def test_h4_street_map_is_said_once_in_the_list() -> None:
    """Each row's pictures button says what its press does under the head
    "Pictures": "Show", or "Hide" on the row whose pane is open (after review:
    the button is a toggle), and its name starts with that word
    (W.showPictures, W.hidePictures); markPictures keeps both in step while
    the rows stand and leaves the face's button, which has the same focus
    key, at "Pictures"; a row has Street map only on a sheet with places and
    no pictures; the open pane's head in the list holds the square's one
    Street map, with the row's focus key, so openMap still marks it busy; the
    face keeps its own; while pictures are off, the line in the pane's place
    names the press of the view shown (W.picturesOffList in the list)."""
    for unit in ("square", "point"):
        assert _said("showPictures", unit, k="3") == f"Show pictures: {unit} 3"
        assert _said("hidePictures", unit, k="3") == f"Hide pictures: {unit} 3"
        assert _said("picturesOffList", unit) == PICTURES_OFF_LIST.format(one=unit)
    editor = _editor()
    rows = _between(editor, "  function renderList(focusKey) {", "\n  }\n")
    show = (
        "'aria-label': (state.shown === h ? W.hidePictures : W.showPictures)(k), "
        "onclick: () => togglePictures(h),\n"
        "          }, state.shown === h ? 'Hide' : 'Show') : null,"
    )
    assert show in rows
    assert "}, 'Pictures') : null," not in rows and "}, 'Show') : null," not in rows
    assert "W.pictures(k)" not in rows
    # In place, while the rows stand: the word and the name of a row's button
    # only (the face's button is not in the table's body).
    mark = _between(editor, "  function markPictures() {", "\n  }\n")
    assert (
        "      const open = b.dataset.focus === 'i:' + state.shown;\n"
        "      b.setAttribute('aria-expanded', String(open));\n"
        "      if (!body.contains(b)) continue;\n"
        "      // WORDING: r5e H4\n"
        "      b.textContent = open ? 'Hide' : 'Show';\n"
        "      b.setAttribute('aria-label', (open ? W.hidePictures : W.showPictures)"
        "(place(b.dataset.focus.slice(2))));\n" in mark
    )
    assert "facePicturesButton.setAttribute('aria-label', W.pictures(k));" in editor
    assert "}, 'Pictures') : null;" in editor  # the face's button keeps its word
    assert "data.has_place && !picturesPossible ? OE.el('button', {" in rows
    assert (
        "placeColumn ? OE.el('th', {}, picturesPossible ? 'Pictures' : W.streetMap)"
        " : null," in rows
    )
    pane = _between(editor, "  function renderPreview(shown) {", "\n  }\n")
    # With no pane open and pictures off: the one node (never null here: the
    # test before the else), its words those of the view shown, set only when
    # they differ, so the live region says them once per switch.
    off = (
        "      if (state.picturesOn || !picturesOff) preview.replaceChildren();\n"
        "      else {\n"
        "        const off = state.view === 'list' ? W.picturesOffList : "
        "W.picturesOff;\n"
        "        if (picturesOff.textContent !== off) picturesOff.textContent = off;\n"
        "        if (preview.childNodes.length !== 1 || preview.firstChild !== "
        "picturesOff) preview.replaceChildren(picturesOff);\n"
        "      }\n"
    )
    assert off in pane
    assert pane.index("!picturesOff") < pane.index("picturesOff.textContent")
    head = _between(pane, "    const head = OE.el('div', { class: 'spread' },", ";\n")
    assert (
        "state.view === 'list' && data.has_place ? OE.el('button', { class: 'link', "
        "'data-focus': 'm:' + h, 'aria-disabled': String(linking.has(h)), "
        "'aria-label': W.openMapOf(k), onclick: () => openMap(h) }, W.streetMap)"
        " : null," in head
    )
    assert "'Close pictures'" in head
    assert head.index("W.streetMap") < head.index("'Close pictures'")
    # openMap finds the button by its focus key anywhere in the editor.
    mark = "wrap.querySelector('[data-focus=\"' + CSS.escape('m:' + h) + '\"]')"
    assert mark in editor
    # The face keeps its Street map beside Pictures.
    assert "const faceMapButton = data.has_place ? OE.el('button', {" in editor


# --------------------------------------------------------------------------- H5


def test_h5_the_footer_line_and_the_graded_notice_are_shorter() -> None:
    """Beside the ready finish button, the line on not changing an answer
    alone; on a graded sheet the notice is two lines, and its reason is a
    small line in the Details fold (only on a graded sheet); r5-d R06's line
    for a graded sheet's footer stays, the notice's own first sentence."""
    assert _said("everyDone", "square") == EVERY_DONE
    assert _said("graded", "square") == GRADED
    assert _said("gradedWhy", "square") == GRADED_WHY
    assert _said("everyDoneGraded", "square") == EVERY_DONE_GRADED
    assert GRADED.startswith(EVERY_DONE_GRADED + " ")
    editor = _editor()
    assert (
        "  const gradedWhy = gradedNote ? OE.el('div', { class: 'small' }, "
        "W.gradedWhy) : null;\n" in editor
    )
    assert editor.index("  const gradedNote = ") < editor.index("  const gradedWhy = ")
    assert editor.index("  const gradedWhy = ") < editor.index("  const foldParts = [")
    layout = _between(editor, "  wrap.append(", "  // One listener for the keys")
    assert "gradedWhy" not in layout
    code = _code(_script())
    assert "All answered and saved." not in code
    assert "a change made after seeing" not in code


# ------------------------------------------------------------------- the marks


def test_every_r5e_string_of_the_sheet_carries_its_wording_comment() -> None:
    """Each entry of OE.sheetWords this round changed or added has a WORDING
    line of r5-e above the tag of its round (nearest the entry), and so have
    the list's Show and Hide where they are drawn and kept in step; no other
    entry got one."""
    tags = {
        "what": ["// WORDING: r4m M6", "// WORDING: r5e H1"],
        "hiddenOne": ["// WORDING: r5b J28", "// WORDING: r5e H1"],
        "hiddenPair": ["// WORDING: r5b J28", "// WORDING: r5e H1"],
        "shownPair": ["// WORDING: r5b J28", "// WORDING: r5e H1"],
        "whatPlaced": ["// WORDING: r4h S3 E/sheet-1; r4m M6", "// WORDING: r5e H1"],
        "picturesOff": ["// WORDING: r5e H1"],
        "picturesOffList": ["// WORDING: r5e H4"],
        "showPictures": ["// WORDING: r5e H4"],
        "hidePictures": ["// WORDING: r5e H4"],
        "cannotTellLine": ["// WORDING: r4i F5", "// WORDING: r5e H1"],
        "classWhat": [
            "// WORDING: r4h S3 E/sheet-3",
            "// WORDING: r5e H1 (with the period)",
        ],
        "graded": ["// WORDING: r5b J06", "// WORDING: r5e H5"],
        "gradedWhy": ["// WORDING: r5e H5"],
        "everyDone": ["// WORDING: r4h S2 A/F17; r5b J29", "// WORDING: r5e H5"],
    }
    for key, want in tags.items():
        assert _tags_above(key) == want, key
    ours = sum(1 for want in tags.values() for tag in want if "r5e" in tag)
    assert _words().count("// WORDING: r5e ") == ours
    rows = _between(_editor(), "  function renderList(focusKey) {", "\n  }\n")
    assert "          // WORDING: r5e H4 (the column's head says" in rows
