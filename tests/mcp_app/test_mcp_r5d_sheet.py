# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Round r5-d (17 Nov 2026), track SHEET, as specs/r5-d-round.md gives it: the
label sheet's items of the last audit's build list
(specs/r5-final-audit/build_list.md). The views are read as text, as they are
on disk (no test here runs the scripts); the words of ``OE.sheetWords`` are
read back from the file (:func:`_said`: a JS concatenation of single-quoted
strings and names read into its text), so a missing space between two joined
pieces fails here as it would on the face.

- R02: the cloud line says a case ("If cloud covers the square on every date,
  press Cannot tell."), not a fact about the square shown; it is drawn where
  it was (the rule block, and the pictures pane's fallback rule lines).
- R06: on a sheet that already gave its result, the line beside the ready
  finish button says so ("This sheet already gave its result."), the
  notice's own first sentence, in place of the line that says an answer may
  change until the press.
- R08: the class face's line with no period known is in the Details fold,
  not a fourth line on the face; with the period it stays on the face. The
  words do not change.
- R13: the list's class menu says Cannot tell as the face's button does
  ("Cannot tell (?)"); its value stays ``?``.
"""

from __future__ import annotations

import re

from olmoearth_agent.mcp_app import sheet, views

#: R02: the cloud line, for squares (W.cloudLine).
CLOUD_LINE = "If cloud covers the square on every date, press Cannot tell."
#: R02: the same for points.
CLOUD_LINE_POINTS = "If cloud covers the point on every date, press Cannot tell."
#: R02: what the cloud line said before (r4-m M8), which read as a fact.
CLOUD_OLD = "No clear date of this"
#: R06: beside the ready finish button on a graded sheet (W.everyDoneGraded).
EVERY_DONE_GRADED = "This sheet already gave its result."
#: R06: the line beside it on every other sheet, unchanged (W.everyDone).
EVERY_DONE = "All answered and saved. After you press, do not change an answer."
#: R08: the class face's line with no period known (W.classWhat(null)), for
#: squares: the words of r4-h's E/sheet-3, unchanged.
CLASS_NO_PERIOD = (
    "For each square, pick what was there in the period the map covers, using the "
    "map's class names; ask whoever made the map for that period."
)
#: R13: the list's Cannot tell, the face's button's words (sheet.CANNOT_TELL
#: and its key).
LIST_CANNOT_TELL = "Cannot tell (?)"

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


def _tag_above(key: str) -> list[str]:
    """The comment lines right above an entry of OE.sheetWords, nearest
    first, up to the first line that is not a WORDING line."""
    lines = _words().split("\n")
    (at,) = [i for i, line in enumerate(lines) if line.startswith(f"    {key}: ")]
    tags: list[str] = []
    for line in reversed(lines[:at]):
        if not line.strip().startswith("// WORDING: "):
            break
        tags.append(line.strip())
    return tags


# --------------------------------------------------------------------------- R02


def test_r02_the_cloud_line_says_a_case_not_a_fact() -> None:
    """The cloud line is a condition, for squares and for points, with its
    WORDING line of this round above r4-m's; the old words, which read as
    said of the square shown before any picture was open, are gone from the
    code."""
    assert _said("cloudLine", "square") == CLOUD_LINE
    assert _said("cloudLine", "point") == CLOUD_LINE_POINTS
    assert _tag_above("cloudLine") == ["// WORDING: r4m M8", "// WORDING: r5d R02"]
    code = _code(_script())
    assert CLOUD_OLD not in code
    assert "cloud over it on every date" not in code


def test_r02_the_cloud_line_is_drawn_where_it_was() -> None:
    """Only the words change: the rule block says it on a face that takes
    Cannot tell and can show pictures, and the pictures pane's fallback rule
    lines on a face that takes Cannot tell; nowhere else."""
    editor = _editor()
    block = _between(editor, "  const ruleBlock = pageRule ?", "\n  const clearNote")
    assert "    cloudOnFace ? OE.el('div', { class: 'small' }, W.cloudLine)\n" in block
    pane = _between(editor, "    let ruleLines = null;", "\n    const dated = ")
    assert "takesQ ? OE.el('div', { class: 'small' }, W.cloudLine) : null);" in pane
    assert _code(editor).count("W.cloudLine") == 2
    assert "const cloudOnFace = takesQ && picturesPossible;" in editor


# --------------------------------------------------------------------------- R06


def test_r06_a_graded_sheets_footer_says_it_gave_its_result() -> None:
    """The line beside the ready finish button is W.everyDoneGraded on a
    sheet whose page copy says it was graded (the same test as the notice's),
    W.everyDone on every other; the new words are the notice's own first
    sentence, so the face says one thing."""
    assert _said("everyDoneGraded", "square") == EVERY_DONE_GRADED
    assert _said("everyDoneGraded", "point") == EVERY_DONE_GRADED
    assert _said("everyDone", "square") == EVERY_DONE
    assert _said("graded", "square").startswith(EVERY_DONE_GRADED + " ")
    assert _tag_above("everyDoneGraded") == ["// WORDING: r5d R06"]
    assert _tag_above("everyDone") == ["// WORDING: r4h S2 A/F17; r5b J29"]
    editor = _editor()
    assert (
        "  // WORDING: r5d R06\n"
        "  const everyDoneLine = OE.el('span', { class: 'small' },\n"
        "    data.graded && typeof data.graded === 'object' ? W.everyDoneGraded"
        " : W.everyDone);\n" in editor
    )
    assert "const gradedNote = data.graded && typeof data.graded === 'object'" in editor
    # The line still shows only once the sheet is ready, and goes when it is not.
    finish = _between(editor, "  function updateFinish() {", "\n  }\n")
    for text in (
        "if (ok && !done.contains(everyDoneLine)) done.prepend(everyDoneLine);",
        "else if (!ok && done.contains(everyDoneLine)) everyDoneLine.remove();",
    ):
        assert text in finish, text
    assert _code(editor).count("W.everyDoneGraded") == 1


# --------------------------------------------------------------------------- R08


def test_r08_the_class_line_with_no_period_is_in_the_fold() -> None:
    """With no period known the class face's instruction is empty (so render
    hides it) and the same words stand in the Details fold, first after the
    file name, before what Cannot tell does; with the period the line stays
    on the face. A two-map sheet's face is 'pair', never 'class'."""
    editor = _editor()
    told = _between(editor, "  function faceInstructionText() {", "\n  }\n")
    assert (
        "    // WORDING: r5d R08 (the class face's line, on the face only with the "
        "map's period)\n"
        "    if (face === 'class') return period ? W.classWhat(period) : '';\n" in told
    )
    assert "faceInstruction.hidden = !faceInstruction.textContent;" in editor
    assert (
        "  // WORDING: r5d R08 (where it stands)\n"
        "  const classNoPeriod = face === 'class' && !period ? OE.el('div', "
        "{ class: 'small' }, W.classWhat(null)) : null;\n" in editor
    )
    assert editor.index("  const period = ") < editor.index("  const classNoPeriod = ")
    fold = _between(editor, "  const foldLabel = 'Details';", "\n\n")
    assert (
        "data.name),\n    classNoPeriod,\n"
        "    instruction, hint, ...foldLines, clearNote, examplesNote, readingNote,"
        " mixNote,\n" in fold
    )
    assert "const twoMap = face === 'pair';" in editor
    # The words are r4-h's E/sheet-3, unchanged: classWhat's branch with no
    # period.
    block = _between(_words(), "    classWhat: (period) => (period", "),\n")
    _known, unknown = block.split("\n      : ")
    assert _concat(unknown, {"one": "square"}) == CLASS_NO_PERIOD


# --------------------------------------------------------------------------- R13


def test_r13_the_lists_cannot_tell_is_the_faces() -> None:
    """The list's class menu writes the answer ? as the face's button does
    (the server's label for it, then its key); the option's value stays
    OE.UNJUDGED, the server's '?'."""
    editor = _editor()
    assert LIST_CANNOT_TELL == sheet.CANNOT_TELL + " (" + sheet.UNJUDGED + ")"
    assert (
        "    // WORDING: r5d R13\n"
        "    options.map((c) => OE.el('option', { value: c }, c === OE.UNJUDGED ? "
        f"'{LIST_CANNOT_TELL}' : c)));" in editor
    )
    # The face's button: the answer's label, then its first key.
    assert "a.label + (a.keys.length ? ' (' + a.keys[0] + ')' : '')" in editor
    assert "OE.UNJUDGED = '" + sheet.UNJUDGED + "';" in _script()
    assert "? (cannot tell)" not in _code(_script())
