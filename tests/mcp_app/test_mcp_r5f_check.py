# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Round r5-f (18 Nov 2026), track CHECK, on the MCP App: the Check card's
items of the audit after the design round (specs/r5-e-audit/build_list.md:
F04, F09 and F11).

- F04: a picked part whose sheet cannot be read, or did not open on a press,
  says on its face what to do, in plain words; the server's reason (a file's
  name, a step no button of the card does) is in the part's lines in Details;
- F09: step 3 says what the one who looks must recognise: what the map
  shows (since round r5-m B1, what the map is meant to find);
- F11: a failed pick's line names the press by its button, or says the pick
  is not in this press and waits for a later one; with every square
  answered the heading says "the sheet" on a map in one part and "a part's
  sheet" on a map in parts.

The views are read as text (no test here runs the scripts); a JS
concatenation of single-quoted strings and names is read back into its text
(:func:`_concat`). The server's words on a sheet that is gone are read
through its own function.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytest.importorskip("mcp")

from olmoearth_agent.mcp_app import check, views  # noqa: E402

#: One token of a JS concatenation: a single-quoted string, a plus, or a name
#: (a call with no parentheses inside its own is one name).
_TOKEN = re.compile(
    r"\s*(?:'((?:[^'\\]|\\.)*)'|(\+)|([A-Za-z_][A-Za-z0-9_.]*(?:\([^()]*\))?))"
)

#: Words no face holds (round r5-e's rule R7, and the owner's ruling 6).
_OFF_FACE = re.compile(
    r"\b(?:windows?|UTM|zones?|intervals?|strat(?:um|a)|design|budget"
    r"|recommended)\b",
    re.I,
)

#: F04: the face's line of a picked part whose sheet cannot be read when the
#: card opens: on a map in parts (the part by its name), and in one part.
SHEET_UNREAD_WEST = (
    "Its sheet cannot be opened now. Tell Claude that the sheet of West part "
    "cannot be opened."
)
SHEET_UNREAD_ONE = (
    "Its sheet cannot be opened now. Tell Claude that the sheet cannot be opened."
)
#: F04: the face's line when a sheet did not open on a press of its button.
#: Round r5-i B1: it names the button by its label, in double quotes ("Open
#: the sheet" here; tests/mcp_app/test_mcp_r5i_check.py), where it said "the
#: button"; before any press, a sheet that could not be read when the card
#: opened says to press it (SHEET_TRY), and this round's red SHEET_UNREAD
#: comes only after a press failed.
SHEET_NOT_OPENED = (
    'The sheet could not be opened. Press "Open the sheet" to try again; if it '
    "fails again, tell Claude."
)
#: F04: the words the face said before, now the start of each line in Details
#: (the server's reason follows).
SHEET_UNREAD_WHY = "Its sheet cannot be read now: "
SHEET_NOT_OPENED_WHY = "The sheet could not be opened: "

#: F09: step 3, and the words it had ("it" pointed at nothing on the face).
#: Round r5-m B1: what the map is meant to find (it said "what the map
#: shows"; tests/mcp_app/test_mcp_r5m_check.py).
STEP_3 = (
    "3. Someone who can recognise what the map is meant to find looks at each one "
    "in a satellite picture."
)
STEP_3_OLD = "recognise it in a satellite picture"

#: F11: the press's own words, and a failed pick's two lines after "The pick
#: failed.", for 80 squares: the press retries it, or it waits its turn: not
#: in this press (the line under the press says "this press" too), on a later
#: press of the button, never "the next press" (past twice draws_max failed
#: picks, even the press after this one may not take it).
PRESS = "Pick the squares"
RETRY_80 = "Press Pick the squares to try again with 80 squares."
RETRY_LATER_80 = (
    "Not in this press: it waits for a later press of Pick the squares " "(80 squares)."
)
#: F11: the heading with every square answered and no sheet open, on a map in
#: one part and on a map in parts.
ALL_ANSWERED_ONE = (
    "Every square is answered. Open the sheet, then press Estimate the error rate."
)
#: Round r5-i B2: "each part's sheet in turn" on a map in parts (it said "a
#: part's sheet", which suggested one press gives the map's result). Round
#: r5-j B3: the press in each part's sheet, one result per part ("in turn,
#: then press" still read as one press).
ALL_ANSWERED_PARTS = (
    "Every square is answered. In each part's sheet, press Estimate the error "
    "rate: one result per part."
)


def _read(name: str) -> str:
    """A file of views/, as it is on disk."""
    return (views.VIEW_DIR / name).read_text(encoding="utf-8")


def _between(text: str, start: str, end: str) -> str:
    """The part of text from start up to the first end after it."""
    at = text.index(start)
    return text[at : text.index(end, at)]


def _after(text: str, start: str, end: str) -> str:
    """The part of text after start up to the first end after it."""
    at = text.index(start) + len(start)
    return text[at : text.index(end, at)]


def _concat(expr: str, names: dict[str, str] | None = None) -> str:
    """The text a JS concatenation of single-quoted strings and names makes,
    each name (or call) given its value in names; anything else fails."""
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
            out.append((names or {})[name])
        at = match.end()
    return "".join(out)


def _code(page: str) -> str:
    """The page with its comments taken out."""
    return re.sub(r"/\*.*?\*/|//[^\n]*", "", page, flags=re.S)


def _in_order(text: str, *pieces: str) -> bool:
    """Whether each piece is in text, each after the one before it."""
    at = [text.index(p) for p in pieces]
    return at == sorted(at)


# --------------------------------------------------------------------------- F04


def test_f04_a_part_whose_sheet_cannot_be_read_says_what_to_do() -> None:
    """F04: a picked part whose sheet cannot be read when the card opens says,
    in body type, that the sheet cannot be opened now and what to tell
    Claude, the part by its name on a map in parts; the face no longer shows
    the server's words (a raw file name, "plan the sample again") in a small
    line."""
    page = _read("check.html")
    assert "// WORDING: r5f F04\nconst SHEET_UNREAD = (name) => " in page
    named = "(name ? ' of ' + name : '')"
    unread = _after(page, "const SHEET_UNREAD = (name) => ", ";\n")
    assert unread.count(named) == 1
    for of, said in ((" of West part", SHEET_UNREAD_WEST), ("", SHEET_UNREAD_ONE)):
        assert _concat(unread.replace(named, "OF"), {"OF": of}) == said
        assert _OFF_FACE.search(said) is None, said
    zone = _between(page, "  function fillZone(m) {", "\n  }\n")
    # Round r5-i B1: before any press the line says to press the button
    # (SHEET_TRY, tests/mcp_app/test_mcp_r5i_check.py); this line, red, only
    # once a press failed too.
    line = (
        "      const unread = Boolean(d.sheet_error) && !d.progress;\n"
        "      if (unread) {\n"
        "        parts.push(m.sheetError ? OE.el('div', { class: 'error' }, "
        "SHEET_UNREAD(nParts > 1 ? label(m) : null))\n"
        "          : OE.el('div', {}, SHEET_TRY(nParts > 1 ? label(m) : null, "
        "button)));\n"
        "      }\n"
    )
    assert line in zone
    # No word of the server's on the part's face, and no small line there.
    for gone in ("bare(d.sheet_error)", "SHEET_UNREAD_WHY", "{ class: 'small' }"):
        assert gone not in zone, gone
    assert "plan the sample again" not in _code(page)


def test_f04_a_sheet_that_did_not_open_says_to_press_again() -> None:
    """F04: a sheet that did not open on a press of its button says, in the
    part's red line, to press the button again and to tell Claude if it
    fails again (unless its sheet could not be read when the card opened
    either: then that line, turned red, is the one); the thrown error's
    words are kept for Details (sheetWhy), and both go once the sheet opens,
    and so does the reason its sheet could not be read when the card opened
    (else the part would say its sheet cannot be opened beside "Its sheet is
    open below."). The open button stays. Round r5-i B1: the line names the
    button by the label it shows (SHEET_NOT_OPENED of the button's label,
    worded when the part is drawn: m.sheetError is a flag now)."""
    page = _read("check.html")
    assert "// WORDING: r5f F04\nconst SHEET_NOT_OPENED = (button) => " in page
    expr = _after(page, "const SHEET_NOT_OPENED = (button) => ", ";\n")
    said = _concat(expr, {"button": "Open the sheet"})
    assert said == SHEET_NOT_OPENED
    assert _OFF_FACE.search(said) is None
    model = _between(page, "function zoneModel(z, index) {", "\n}\n")
    assert "sheetError: null, sheetWhy: null, opening: false," in model
    swap = _between(
        page, "  async function openZone(m, { hadFocus = null } = {}) {", "\n  }\n"
    )
    assert _in_order(
        swap,
        "    } catch (err) {\n",
        "      m.sheetError = true;\n"
        "      m.sheetWhy = bare(err.message || 'it could not be read');\n"
        "      renderAll();\n"
        "      if (had) OE.focusIfHere(m.openButton);\n      return;\n    }\n",
        "    m.draw.sheet_error = null;\n    m.sheetWhy = null;\n"
        "    m.sheetError = null;\n",
        "const editor = OE.sheetEditor(app, sheet, {",
    )
    zone = _between(page, "  function fillZone(m) {", "\n  }\n")
    assert _in_order(
        zone,
        "      if (m.sheetError && !unread) parts.push(OE.el('div', "
        "{ class: 'error' }, SHEET_NOT_OPENED(button)));\n",
        "        m.openButton = OE.el('button', {",
    )
    assert "sheetWhy" not in zone and "err.message" not in zone


def test_f04_the_servers_reason_is_in_the_parts_details() -> None:
    """F04: the words the face said before, each with the server's reason,
    are among the part's lines in Details, after a failed pick's reason and
    before the planner's sentences; each of the four strings is defined once
    and used once, and a long file name breaks inside those lines (rule
    R2)."""
    page = _read("check.html")
    for name, said in (
        ("SHEET_UNREAD_WHY", SHEET_UNREAD_WHY),
        ("SHEET_NOT_OPENED_WHY", SHEET_NOT_OPENED_WHY),
    ):
        mark = "// WORDING: r5f F04 (in Details since r5-f)\n"
        assert f"{mark}const {name} = '{said}';\n" in page, name
    fold = _between(page, "  function fillFold() {", "\n  }\n")
    assert _in_order(
        fold,
        "        lines.push(OE.el('div', {}, PICK_FAILED + bare(m.error) + '.'));\n",
        "      const unreadWhy = m.kind === 'drawn' && m.draw && m.draw.sheet_error "
        "&& !m.draw.progress\n"
        "        ? bare(m.draw.sheet_error) : null;\n"
        "      if (unreadWhy) lines.push(OE.el('div', {}, "
        "SHEET_UNREAD_WHY + unreadWhy + '.'));\n"
        "      if (m.sheetWhy && m.sheetWhy !== unreadWhy) {\n"
        "        lines.push(OE.el('div', {}, "
        "SHEET_NOT_OPENED_WHY + m.sheetWhy + '.'));\n"
        "      }\n",
        "      if (m.kind !== 'drawn') {\n",
        "{ class: 'stack fold-part' }, lines));\n",
    )
    code = _code(page)
    for name in (
        "SHEET_UNREAD",
        "SHEET_NOT_OPENED",
        "SHEET_UNREAD_WHY",
        "SHEET_NOT_OPENED_WHY",
    ):
        assert len(re.findall(rf"\b{name}\b", code)) == 2, name
    assert "  .fold-part { overflow-wrap: anywhere; }\n" in page


def test_f04_a_failed_press_on_a_sheet_that_could_not_be_read_says_one_line() -> None:
    """F04, after review: a part whose sheet could not be read when the card
    opened, and whose button's press then failed too, says one line on its
    face: the one that names the part and says to tell Claude, now red, so
    the press visibly got its answer; not, beside it, the line that says to
    press again (the sheet failed twice). Details says the server's reason
    once when the press failed for the same reason, and both when they
    differ (a host's failure, say)."""
    page = _read("check.html")
    zone = _between(page, "  function fillZone(m) {", "\n  }\n")
    assert zone.count("const unread = ") == 1
    assert _in_order(
        zone,
        "      const unread = Boolean(d.sheet_error) && !d.progress;\n",
        "m.sheetError ? OE.el('div', { class: 'error' }, SHEET_UNREAD(",
        "      if (m.changed) parts.push(",
        "      if (m.sheetError && !unread) parts.push(",
    )
    # The face draws the press's red line in one place only, and only when
    # the line on a sheet that could not be read is not shown (round r5-i
    # B1: worded there, SHEET_NOT_OPENED of the button's label).
    assert zone.count("SHEET_NOT_OPENED(button)));") == 1
    fold = _between(page, "  function fillFold() {", "\n  }\n")
    assert fold.count("bare(m.draw.sheet_error)") == 1
    assert "m.sheetWhy !== unreadWhy" in fold
    # Both reasons go through bare(), so the same server words compare equal.
    swap = _between(
        page, "  async function openZone(m, { hadFocus = null } = {}) {", "\n  }\n"
    )
    assert "m.sheetWhy = bare(err.message || 'it could not be read');\n" in swap


def test_f04_counts_read_again_clear_a_sheet_that_could_not_be_read() -> None:
    """F04, after review: a part whose sheet could not be read when the card
    opened, and whose counts read later (the user comes back to the panel),
    no longer says its sheet cannot be opened beside the counts: refreshZone
    notes whether that line showed before it takes the counts, and, when the
    read gave progress words, drops the reason and draws the part and
    Details again, the focus kept on the part's open button. A failed press
    is another read: its line and its reason (sheetError, sheetWhy) stay."""
    page = _read("check.html")
    refresh = _between(page, "  async function refreshZone(m) {", "\n  }\n")
    assert _in_order(
        refresh,
        "    if (mine !== m.reads || !got || !m.draw) return;\n",
        "    const unread = Boolean(m.draw.sheet_error) && !m.draw.progress;\n",
        "    if (typeof got.progress === 'string') m.draw.progress = got.progress;\n",
        "    if (m.openButton) m.openButton.textContent = ",
        "    if (unread && typeof got.progress === 'string' && got.progress) {\n"
        "      const had = Boolean(m.openButton) && "
        "document.activeElement === m.openButton;\n"
        "      m.draw.sheet_error = null;\n"
        "      fillZone(m);\n"
        "      fillFold();\n"
        "      if (had) OE.focusIfHere(m.openButton);\n"
        "    }\n",
        "    fillAsk();\n",
    )
    assert "sheetError" not in refresh and "sheetWhy" not in refresh
    # Shown again, the card reads every drawn part's counts.
    assert (
        "models.filter((m) => m.kind === 'drawn').forEach((m) => refreshZone(m));"
        in page
    )


def test_f04_the_servers_words_on_a_missing_sheet_name_its_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """F04: what the server says of a picked part's sheet that is gone
    (check._sheet_words, the view's ``sheet_error``): the file's name and a
    step no button of the card does. The card keeps these words for Details;
    this round changes none of them."""
    out = tmp_path / "outputs"
    out.mkdir()
    monkeypatch.setenv("OLMOEARTH_SCORES_ROOT", str(out))
    monkeypatch.setenv("OLMOEARTH_OUTPUT_ROOT", str(out))
    words = check._sheet_words(str(out / "gone_to_label.csv"))
    assert words == {
        "progress": None,
        "open_label": None,
        "sheet_error": "gone_to_label.csv does not exist; plan the sample again",
    }


# --------------------------------------------------------------------------- F09


def test_f09_step_3_says_what_the_reader_must_recognise() -> None:
    """F09: step 3 reads "Someone who can recognise what the map is meant to
    find looks at each one in a satellite picture." (it said "recognise it";
    steps 1 and 2 speak of the map and the squares, so "it" pointed at
    nothing; round r5-m B1 put "what the map is meant to find" for F09's
    "what the map shows"); the other steps are as they were, and the four go
    to the steps figure."""
    page = _read("check.html")
    marks = (
        "// WORDING: r5f F09 (step 3)\n"
        "// WORDING: squares; r4h critic gap 2 (step 3)\n"
        "const STEPS = [\n"
    )
    assert marks in page
    steps = _between(page, "const STEPS = [", "];")
    listed = [
        "1. The map is cut into small squares.",
        "2. The app picks some at random.",
        STEP_3,
        "4. You get how often the map is wrong, with a range.",
    ]
    assert steps == "const STEPS = [\n" + "".join(f"  '{s}',\n" for s in listed)
    assert "fig('steps', [firstStep(models), ...STEPS.slice(1)])" in page
    assert STEP_3_OLD not in page
    assert " it " not in STEP_3 and _OFF_FACE.search(STEP_3) is None


# --------------------------------------------------------------------------- F11


def test_f11_a_failed_picks_line_says_which_press_and_whose() -> None:
    """F11: after "The pick failed." a part the press retries says "Press
    Pick the squares to try again with 80 squares.": the press by its
    button's own words, written out in the line; a part that waits (more
    picks failed than one press takes) says it is not in this press (the
    line under the press calls the press the user makes "this press") and
    waits for a later press of the same button, with its number; never "the
    next press", which past twice draws_max failed picks is untrue. Neither
    speaks as the system any more ("The press tries again ...", "A later
    press tries again ...")."""
    page = _read("check.html")
    assert f"// WORDING: squares\nconst PRESS = '{PRESS}';\n" in page
    counted = {"OE.counted(n, 'square')": "80 squares"}
    tags = (
        "// WORDING: r5f F11\n// WORDING: r5e\n// WORDING: r5b J25\n"
        "// WORDING: squares\nconst RETRY = "
    )
    assert tags in page
    retry = _concat(_after(page, "const RETRY = (n) => ", ";\n"), counted)
    assert retry == RETRY_80
    assert retry.startswith(f"Press {PRESS} to try again with ")
    tags = (
        "// WORDING: r5f F11\n// WORDING: r5e\n// WORDING: squares\n"
        "const RETRY_LATER = "
    )
    assert tags in page
    later = _concat(_after(page, "const RETRY_LATER = (n) => ", ";\n"), counted)
    assert later == RETRY_LATER_80
    for gone in ("The press tries again", "A later press tries again"):
        assert gone not in _code(page), gone
    # Drawn where they were: one line after the short red words.
    zone = _between(page, "  function fillZone(m) {", "\n  }\n")
    drawn = (
        "      parts.push(OE.el('div', {}, OE.el('span', { class: 'error' }, "
        "FAILED_SHORT), ' ',\n"
        "        now ? RETRY(m.budget) : RETRY_LATER(m.budget)));\n"
    )
    assert drawn in zone
    for said in (RETRY_80, RETRY_LATER_80):
        assert "Claude" not in said and _OFF_FACE.search(said) is None, said


def test_f11_the_heading_names_the_sheet_by_the_maps_parts() -> None:
    """F11: with every square answered and no sheet open the heading says
    "Open the sheet" on a map in one part (it said "a sheet" on a map with
    one) and "Open a part's sheet" on a map in parts: fillAsk passes whether
    the map has more than one part. Round r5-i B2: "Open each part's sheet in
    turn" on a map in parts, and after review only where more than one part
    has a sheet: fillAsk passes whether more than one part is drawn
    (tests/mcp_app/test_mcp_r5i_check.py). Round r5-j B3: "In each part's
    sheet, press Estimate the error rate: one result per part." on a map in
    parts; the line on one part keeps its words
    (tests/mcp_app/test_mcp_r5j_check.py)."""
    page = _read("check.html")
    tags = (
        "// WORDING: r5f F11\n// WORDING: r5b J17\n"
        "const ALL_ANSWERED = (several) => "
    )
    assert tags in page
    which = (
        "(several ? 'In each part\\'s sheet, press Estimate the error rate: one "
        "result per part.'\n    : 'Open the sheet, then press Estimate the error "
        "rate.')"
    )
    answered = _after(page, "const ALL_ANSWERED = (several) => ", ";\n")
    assert answered.count(which) == 1
    head = "Every square is answered. "
    for said in (ALL_ANSWERED_ONE, ALL_ANSWERED_PARTS):
        tail = said.removeprefix(head)
        assert "'" + tail.replace("'", "\\'") + "'" in which, tail
        assert _concat(answered.replace(which, "WHICH"), {"WHICH": tail}) == said
        assert _OFF_FACE.search(said) is None, said
    fill = _between(page, "  function fillAsk() {", "\n  }\n")
    line = (
        "    ask.textContent = drawable().length ? ASK "
        ": done ? ALL_ANSWERED(drawn.length > 1) : YOUR_SQUARES;\n"
    )
    assert line in fill
    assert "  const nParts = models.length;\n" in page
    assert "Open a sheet" not in page
    # Defined once, and the heading's one use.
    assert _code(page).count("ALL_ANSWERED") == 2


# --------------------------------------------------------------------------- marks


def test_every_item_of_the_track_is_marked_in_the_page() -> None:
    """Each new or changed string of this round carries ``// WORDING: r5f``
    with its item, above its older marks: SHEET_UNREAD, SHEET_NOT_OPENED and
    the two starts of the Details lines (F04), STEPS (F09), ALL_ANSWERED,
    RETRY and RETRY_LATER (F11)."""
    page = _read("check.html")
    marks = re.findall(r"// WORDING: r5f (F\d+)", page)
    for item, count in (("F04", 4), ("F09", 1), ("F11", 3)):
        assert marks.count(item) == count, item
    for name in (
        "SHEET_UNREAD",
        "SHEET_NOT_OPENED",
        "SHEET_UNREAD_WHY",
        "SHEET_NOT_OPENED_WHY",
        "STEPS",
        "ALL_ANSWERED",
        "RETRY",
        "RETRY_LATER",
    ):
        marked = r"// WORDING: r5f [^\n]*\n(?://[^\n]*\n)*const " + name + r" = "
        assert re.search(marked, page) is not None, name
