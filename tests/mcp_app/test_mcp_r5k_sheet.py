# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Round r5-k (18 Nov 2026), track SHEET, as specs/r5-k-round.md gives it:
the owner's design decisions after the r5-i audit, on the label sheet
(specs/r5-i-audit/owner_decisions.md O2, O19, O8 and O11). The views are read
as text, as they are on disk (no test here runs the scripts); the words of
``OE.sheetWords`` are read back from the file (:func:`_said`).

- K2 (O2): a two-map sheet's switch that shows what the two maps say stays
  locked (ticked, aria-disabled, the greyed look of a busy button) until
  every square is answered and saved and the sheet's result was asked for
  (the finish press whose message Claude took), with one line under its
  words that says so; a press or a key on it changes nothing. A real change
  of an answer after the press, made here or taken in, locks it again until
  a new press; a press whose message the host did not take keeps it locked,
  the line then saying it opens once the sheet is opened again after Claude
  grades it.
- K3 (O19): a sheet that gave its result, or that says it is not saving,
  takes no answer: its answer buttons, keys and list menus are read only
  (aria-disabled with the busy look, no handler acts), each described by the
  line that says why (the graded notice, now "This sheet gave its result;
  its answers are locked.", or the footer's line beside "Reopen the sheet");
  a class being typed keeps its text. A graded sheet's face asks for no
  answer (no instruction), offers no Remove button for a set of its answers,
  and says by its grey finish button why a square left blank (one blanked
  elsewhere) blocks it; a sheet Claude grades after the press is drawn again
  once the counts say so, its answers locked.
- K4 (O8): on a square's card the pictures come before the answers (the
  heading, Pictures and its pane, then the answers and Cannot tell), and the
  dates wrap onto as many rows as they need, each at its own size, instead
  of a sideways strip cut at the panel's edge (the look-first pane's too);
  an answer that moves on, or an arrow key, brings the new square's heading
  into view.
- K5 (O11), the sheet's part: a save that went through ("Saved to ...") is
  said beside the counter in the place of "Saved automatically"; a save that
  failed ("Not saved: ...") and Save labels' press with nothing waiting are
  a line in the sheet's footer under the row with Save labels; a save made
  elsewhere taken in ("This sheet was saved elsewhere; it shows those labels
  now.") is said beside the counter it changed; each keeps its words and its
  role (status, or alert for an error). The sheet's controls keep the
  footer's own height and 8 px over it (--oe-foot, set from the footer each
  time it changes; r5-j B8 kept 140 px).

The browser harness (scripts/mcp_views_e2e.py) draws them: 7 (K2), 3j, 3j2
(renamed 3j2_sheet_graded_locked), 3j2b (the graded sheet with a square
blanked elsewhere), 3j4, 3j5, 3j6, 3h2, 3w, 3c and 3h (K3), 10, 10d, 3j, 3k,
3s and 6b (K4), and 3, 3e, 3f, 3g, 3i, 7 and the Check card's swap (K5).
"""

from __future__ import annotations

import re

from olmoearth_agent.mcp_app import views

#: K2: the line under the locked switch's words (W.pairLocked), for squares.
PAIR_LOCKED = "Shown once every square is answered and the result is sent."
#: K2: in its place once the finish press's message was shown to copy
#: (W.pairLockedNotSent).
PAIR_NOT_SENT = "Shown when you open this sheet again after Claude grades it."
#: K3: by a graded sheet's grey finish button while a square has no answer
#: (W.gradedBlank), for squares.
GRADED_BLANK = "A square has no answer; a new result needs a fresh sheet."
#: K3: the line the spec gives, the graded notice's first sentence.
LOCKED = "This sheet gave its result; its answers are locked."
#: K3: the graded notice whole (W.graded).
GRADED = LOCKED + " A second opinion needs another person and a fresh sheet."
#: K3: what the notice said before, which the lock makes untrue.
OLD_GRADED = "Do not change an answer now"
#: K5: the words the moved statuses keep, as the script writes them ("Saved
#: to ..." is said beside the counter, counterSay, once the save is over).
SAVED = "savedWords = 'Saved to ' + data.name + '.'"
NOT_SAVED = "footerSay(saveSaid, 'Not saved' + (wrap.isConnected ? '' : ' to '"
KEEPS_CHANGING = (
    "footerSay(saveSaid, 'Not saved: the sheet keeps changing elsewhere. ' "
    "+ tryAgain(), true);"
)
ELSEWHERE = (
    "else footerSay(countSaid, 'This sheet was saved elsewhere; it shows those "
    "labels now.' + (keptNote ? ' ' + keptNote : ''));"
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


def _style() -> str:
    """The style sheet every view gets (common.js OE.STYLE), comments and all."""
    return _between(_read("common.js"), "OE.STYLE = `", "\n`;\n")


def _rules(style: str) -> str:
    """A style sheet without its comments."""
    return re.sub(r"/\*.*?\*/", "", style, flags=re.S)


def _code(js: str) -> str:
    """JS text without its comments (a block, or the rest of a line after
    ``//``), each run of joined single-quoted literals read as one: what a
    word gone from the page must be gone from."""
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


def _said(key: str, unit: str) -> str:
    """An entry of OE.sheetWords as the face says it, for a unit word."""
    one = "point" if unit == "point" else "square"
    value = re.sub(r"^\([a-z]+\) => ", "", _entry(_words(), f"    {key}: "))
    return _concat(value, {"one": one, "many": one + "s"})


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


# --------------------------------------------------------------------------- K2


def test_k2_the_line_beside_the_locked_switch() -> None:
    """K2: the line under the switch's words while it is locked says when it
    opens, in the sheet's unit word, under this round's WORDING mark; once a
    press's message was shown to copy (the host did not take it), its other
    form says it opens when the sheet is opened again after Claude grades
    it."""
    assert _said("pairLocked", "square") == PAIR_LOCKED
    assert _said("pairLocked", "point") == PAIR_LOCKED.replace("square", "point")
    assert _tags_above("pairLocked") == ["// WORDING: r5k K2"]
    assert _said("pairLockedNotSent", "square") == PAIR_NOT_SENT
    assert _said("pairLockedNotSent", "point") == PAIR_NOT_SENT
    assert _tags_above("pairLockedNotSent") == ["// WORDING: r5k K2"]


def test_k2_the_switch_opens_only_once_the_result_was_sent() -> None:
    """K2: the switch is locked unless the sheet is ready (every square
    answered and saved, not stale) and its result was asked for: the finish
    press whose message Claude took (resultSent, set only when OE.sendJob says
    the host took it, after the press's guard), or a sheet that already gave
    its result; then it works as before. A message the host did not take sets
    resultShown instead, which only changes the line (syncPairLock). A real
    change undoes the press: setLabel, right after its guard that the answer
    differs, and a take-in from elsewhere that showed a change (adopt, not
    after this page's own save), set both back, before the redraw."""
    editor = _editor()
    assert (
        "  let resultSent = false;\n"
        "  let resultShown = false;\n"
        "  const pairLocked = () => !(ready() && (resultSent || gaveResult));\n"
        in editor
    )
    assert "  const ready = () => complete() && !pending() && !state.stale;" in editor
    finish = _between(editor, "  async function finishPress(message) {", "\n  }\n")
    assert (
        "    const sent = await OE.sendJob(app, message, finishType);\n"
        "    if (sent) resultSent = true; else resultShown = true;\n"
        "    updateProgress();" in finish
    )
    sent = finish.index("resultSent = true;")
    assert finish.index("if (changed || state.stale || pending()") < sent
    assert editor.count("resultSent = true;") == 1
    assert editor.count("resultShown = true;") == 1
    set_label = _between(editor, "  function setLabel(h, value) {", "\n  }\n")
    guard = "    if (state.labels.get(h) === value) return;\n"
    assert set_label.index(guard) < set_label.index("    resultSent = false;\n")
    assert set_label.index("    resultShown = false;\n") < set_label.index(
        "    state.labels.set(h, value);"
    )
    adopt = _between(
        editor, "  function adopt(fresh, { mine = false } = {}) {", "\n  }\n"
    )
    undo = (
        "    if (differed && !mine) {\n"
        "      resultSent = false;\n"
        "      resultShown = false;\n"
        "    }\n"
    )
    assert adopt.index(undo) < adopt.index("    render(key || undefined);")
    # Its declaration, setLabel's and adopt's.
    assert editor.count("resultSent = false;") == 3


def test_k2_a_press_or_key_on_the_locked_switch_changes_nothing() -> None:
    """K2: a click on the switch (a press on it or its label, or Space) is
    cancelled while it is locked, so it stays ticked and no change asks the
    server for the maps' classes; setBlind refuses to untick it too."""
    editor = _editor()
    assert (
        "if (blind) blind.addEventListener('click', (e) => "
        "{ if (pairLocked()) e.preventDefault(); });" in editor
    )
    blind = _between(editor, "  async function setBlind(on) {", "\n  }\n")
    guard = (
        "    if (!on && pairLocked()) { blind.checked = true; state.blind = true; "
        "return; }\n"
    )
    assert guard in blind
    assert blind.index(guard) < blind.index("olmoearth_app_sheet")


def test_k2_the_locked_switch_is_greyed_with_its_line_beside_it() -> None:
    """K2: syncPairLock draws the lock as the counts stand: aria-disabled,
    the line under the switch's words as its description, shown only while
    locked (W.pairLockedNotSent once a press's message was shown to copy and
    every answer is still in and saved, else W.pairLocked), and the label
    greyed (oe-locked); locked again while the maps' answers show, it ticks
    the switch and hides them (true: the caller draws again). render runs it
    first, updateProgress after each change. The line stands in the switch's
    row, outside the label, on a row of its own 4 px under it (the row's gap
    4 px by 16 px) and from the label's words' left edge (after review: beside
    them it wrapped to a loose line 16 px below)."""
    editor = _editor()
    sync = _between(editor, "  function syncPairLock() {", "\n  }\n")
    for text in (
        "if (!blind) return false;",
        "const lock = pairLocked();",
        "blind.setAttribute('aria-disabled', String(lock));",
        "if (lock) blind.setAttribute('aria-describedby', pairLockLine.id); "
        "else blind.removeAttribute('aria-describedby');",
        "const said = lock && ready() && resultShown && !resultSent "
        "? W.pairLockedNotSent : W.pairLocked;",
        "if (pairLockLine.textContent !== said) pairLockLine.textContent = said;",
        "pairLockLine.hidden = !lock;",
        "switchLabel.classList.toggle('oe-locked', lock);",
        "if (!lock || state.blind) return false;",
        "blind.checked = true;",
        "state.blind = true;",
        "return true;",
    ):
        assert text in sync, text
    render = _between(editor, "  function render(focusKey) {", "\n  }\n")
    assert render.index("syncPairLock();") < render.index("if (pairWhat)")
    progress = _between(editor, "  function updateProgress() {", "\n  }\n")
    assert "    if (syncPairLock()) { render(); return; }\n" in progress
    row = (
        "  const switchRow = blind ? OE.el('div', { class: 'row', "
        "style: 'gap:4px 16px' }, switchLabel, pairLockLine) : null;"
    )
    assert row in editor
    line = (
        "  const pairLockLine = blind ? OE.el('span', {\n"
        "    class: 'small', id: 'oe-pair-lock-' + Math.random().toString(36)"
        ".slice(2), style: 'flex-basis:100%;padding-left:31px',\n"
        "  }, W.pairLocked) : null;\n"
    )
    assert line in editor
    # The switch's id and its ticked start are as they were; its words lost
    # "(keep it on while you answer)" in round r5-m B2
    # (tests/mcp_app/test_mcp_r5m_sheet.py).
    assert "type: 'checkbox', id: 'oe-blind', checked: true," in editor
    assert "  const blindText = 'Hide what the two maps say';\n" in editor
    assert "keep it on while you answer" not in _code(editor)


def test_k2_the_locked_look() -> None:
    """K2 (common.js): a locked switch's box at the busy opacity and its
    label's words in the busy colour, each rule on its own line; the line
    under them keeps the body's."""
    style = _style()
    assert (
        'input[type=checkbox][aria-disabled="true"] { cursor: default; '
        "opacity: .38; }\n" in style
    )
    assert "\nlabel.oe-locked { cursor: default; color: rgba(0,0,0,.26); }" in style


# --------------------------------------------------------------------------- K3


def test_k3_the_graded_notice_says_the_answers_are_locked() -> None:
    """K3: the graded notice starts with the spec's line, keeps its second
    sentence (the way to a second opinion) and no longer asks not to change
    an answer; its marks keep r5-b's and r5-e's, this round's on top. The
    footer's shorter line beside the finish button is as it was (r5-d R06)."""
    assert _said("graded", "square") == GRADED
    assert _said("graded", "square").startswith(LOCKED + " ")
    assert _tags_above("graded") == [
        "// WORDING: r5b J06",
        "// WORDING: r5e H5",
        "// WORDING: r5k K3",
    ]
    assert OLD_GRADED not in _code(_script())
    assert _said("everyDoneGraded", "square") == "This sheet already gave its result."


def test_k3_a_sheet_that_gave_its_result_or_does_not_save_is_locked() -> None:
    """K3: the lock holds on a graded sheet (the page copy's graded) and while
    the sheet stopped saving (state.stale); blocked, which every answer path
    asks, holds while it is locked as well as while a column is cleared.
    Round r5-m B2: it holds too once a two-map sheet showed what the two maps
    say (state.revealed; tests/mcp_app/test_mcp_r5m_sheet.py)."""
    editor = _editor()
    assert (
        "  const gaveResult = Boolean(data.graded && "
        "typeof data.graded === 'object');\n"
        "  const locked = () => gaveResult || state.stale || state.revealed;\n"
        in editor
    )
    assert "  const blocked = () => state.clearing || locked();\n" in editor
    set_label = _between(editor, "  function setLabel(h, value) {", "\n  }\n")
    assert "if (blocked()) return;" in set_label
    for name in ("onKey(e)", "onFaceKey(e)", "facePick(value)"):
        body = _between(editor, f"  function {name} {{", "\n  }\n")
        assert "blocked()" in body, name


def test_k3_the_answer_controls_are_marked_in_place() -> None:
    """K3: lockAnswers marks every answer control drawn (the face's answers,
    its list and a typed class with OK; each list row's answer cell)
    aria-disabled, described by the line that says why (the graded notice;
    round r5-m B2: on a two-map sheet whose maps' answers were shown, the
    lead's first line; else the footer's line beside "Reopen the sheet"), a
    field read only; it redraws nothing, so a class being typed keeps its
    text. updateProgress runs it after every render and when the sheet stops
    saving (before its stale branch); the notice has an id to be named by."""
    editor = _editor()
    lock = _between(editor, "  function lockAnswers() {", "\n  }\n")
    for text in (
        "const lock = locked();",
        "const why = gaveResult && gradedNote ? gradedNote : state.revealed && "
        "pairWhat ? pairWhat : staleLine;",
        "faceAnswerArea.querySelectorAll('button, select, input')",
        "body.querySelectorAll('td:last-child button, td:last-child select, "
        "td:last-child input')",
        "c.setAttribute('aria-disabled', 'true');",
        "c.setAttribute('aria-describedby', why.id);",
        "if (c.tagName === 'INPUT') c.readOnly = true;",
        "c.removeAttribute('aria-disabled');",
    ):
        assert text in lock, text
    assert "render(" not in lock and "replaceChildren" not in lock
    progress = _between(editor, "  function updateProgress() {", "\n  }\n")
    assert progress.index("lockAnswers();") < progress.index("if (state.stale) {")
    note_id = (
        "if (gradedNote) gradedNote.id = 'oe-graded-' + "
        "Math.random().toString(36).slice(2);"
    )
    assert note_id in editor
    stale_id = "const staleLine = OE.el('span', { class: 'error', id: 'oe-stale-'"
    assert stale_id in editor


def test_k3_no_press_key_or_menu_sets_an_answer_while_locked() -> None:
    """K3: each way an answer is given checks the lock: a typed class's OK or
    Enter (and Escape, so its text is not lost) in the list and on the face,
    a row's menu (no press opens it, a change goes back to the answer given,
    and onKey keeps its keys from changing it), the face's list of more
    classes (the same) and "Other class...", which opens no field. Paging is
    not answering: the face's arrow keys page before the lock is asked."""
    editor = _editor()
    assert editor.count("if (locked()) return;  // r5-k K3") == 2
    escape = (
        "if (locked()) { if (e.key === 'Enter' || e.key === 'Escape') "
        "e.preventDefault(); return; }"
    )
    assert editor.count(escape) == 2
    cell = _between(editor, "  function labelCell(w) {", "\n  }\n")
    assert "onmousedown: (e) => { if (locked()) e.preventDefault(); }," in cell
    back = "if (locked()) { select.value = state.labels.get(h) || ''; return; }"
    assert back in cell
    on_key = _between(editor, "  function onKey(e) {", "\n  }\n")
    keys = "if (locked() && t && t.tagName === 'SELECT' && !e.ctrlKey && !e.metaKey"
    assert keys in on_key
    face = _between(editor, "  function faceAnswers(w) {", "\n  }\n")
    goes_back = (
        "if (locked()) { list.value = rest.some((a) => a.value === current) "
        "? current : ''; return; }"
    )
    assert goes_back in face
    press = (
        "list.addEventListener('mousedown', (e) => "
        "{ if (locked()) e.preventDefault(); });"
    )
    assert press in face
    other = "onclick: () => { if (!locked()) { state.other = h; render('o:' + h); } },"
    assert other in face
    face_key = _between(editor, "  function onFaceKey(e) {", "\n  }\n")
    arrows = face_key.index("if (k === 'ArrowLeft' || k === 'ArrowRight') {")
    assert arrows < face_key.index("if (blocked()) return;")


def test_k3_the_locked_look() -> None:
    """K3 (common.js): a locked answer (aria-disabled) has every busy
    button's look, no hover; the answer given keeps its bold words and tick
    in grey with a grey line; a locked menu or field is greyed. The rules
    come after the chosen look, which they beat."""
    style = _style()
    busy = (
        '.face .answers button[aria-disabled="true"], '
        '.toggle button[aria-disabled="true"],\n'
        '.face .answers button[aria-disabled="true"]:hover, '
        '.toggle button[aria-disabled="true"]:hover {\n'
        "  cursor: default; background: transparent; color: rgba(0,0,0,.26); "
        "border-color: var(--divider); box-shadow: none; }\n"
    )
    chosen = (
        '.face .answers button[aria-pressed="true"][aria-disabled="true"], '
        '.toggle button[aria-pressed="true"][aria-disabled="true"] {\n'
        "  background: var(--wash); color: var(--grey); border-color: "
        "var(--disabled); box-shadow: inset 0 0 0 1px var(--disabled); }\n"
    )
    fields = (
        'select[aria-disabled="true"], input[type=text][aria-disabled="true"] '
        "{ cursor: default; color: var(--grey);\n  background: var(--wash); }\n"
    )
    for text in (busy, chosen, fields):
        assert text in style, text
    pressed = '.face .answers button[aria-pressed="true"] {'
    assert style.index(pressed) < style.index(busy) < style.index(chosen)
    assert style.index('.toggle button[aria-pressed="true"] {') < style.index(busy)


def test_k3_a_graded_face_asks_for_no_answer() -> None:
    """K3 (r5-b J06), after review: on a sheet that gave its result the face
    has no instruction to answer (the two-map, right-or-wrong and class lines
    asked for answers the lock refuses): faceInstructionText says nothing
    first thing, so render hides the line (r5-d R08's test, unchanged)."""
    editor = _editor()
    told = _between(editor, "  function faceInstructionText() {", "\n  }\n")
    first = (
        "  function faceInstructionText() {\n"
        "    // r5-k K3 (r5-b J06): a sheet that gave its result takes no answer, "
        "so nothing under its notice asks for one.\n"
        "    if (gaveResult) return '';\n"
    )
    assert told.startswith(first)
    assert "faceInstruction.hidden = !faceInstruction.textContent;" in editor


def test_k3_a_locked_sheet_offers_no_removal_of_its_answers() -> None:
    """K3, after review: a sheet whose answers are locked (graded, the only
    one locked where the notice is drawn) keeps the notice of its two sets of
    answers and the fold's line, but no Remove button; and clearAndOpen,
    which sends clear_column, refuses while it is locked."""
    editor = _editor()
    progress = _between(editor, "  function updateProgress() {", "\n  }\n")
    assert (
        "      mix.column && !locked() ? OE.el('button', { class: 'small', "
        "onclick: () => clearAndOpen(mix.column, face) },\n" in progress
    )
    assert "'This sheet holds two sets of answers, so no result" in progress
    clear = _between(editor, "  async function clearAndOpen(column, to) {", "\n  }\n")
    head = "  async function clearAndOpen(column, to) {\n"
    assert clear.startswith(head + "    if (state.clearing || locked()) return;\n")


def test_k3_a_graded_sheet_with_a_blank_square_says_why_it_cannot_finish() -> None:
    """K3, after review: a square blanked elsewhere after the grading leaves a
    graded sheet that cannot be finished and takes no answer; by its grey
    finish button the line in notYet's place says so and what a new result
    needs (W.gradedBlank, in the unit word, this round's mark), as the
    button's description, never "Answer every square first."."""
    assert _said("gradedBlank", "square") == GRADED_BLANK
    assert _said("gradedBlank", "point") == GRADED_BLANK.replace("square", "point")
    assert _tags_above("gradedBlank") == ["// WORDING: r5k K3"]
    editor = _editor()
    assert "    gaveResult ? W.gradedBlank : W.notYet);\n" in editor
    finish = _between(editor, "  function updateFinish() {", "\n  }\n")
    assert "    const waits = !ok && !state.stale && count() < n;\n" in finish
    assert (
        "    if (waits) button.setAttribute('aria-describedby', notYetLine.id);\n"
        in finish
    )


def test_k3_a_sheet_graded_after_the_press_is_drawn_again() -> None:
    """K3, after review: the counts (sheet_progress) say whether the sheet
    gave its result; one that did since this editor was drawn (Claude graded
    it after the finish press, its version unchanged) is read and drawn again
    from that read, so its answers lock, once nothing here waits to be
    saved; a sheet drawn as graded is not drawn again."""
    editor = _editor()
    check = _between(editor, "  function check({ force = false } = {}) {", "\n  }\n")
    graded = (
        "      const graded = Boolean(got && got.graded && typeof got.graded === "
        "'object') && !gaveResult;\n"
        "      if (!got || typeof got.version !== 'string' || (got.version === "
        "data.version && !graded)) return false;\n"
    )
    assert graded in check
    redraw = (
        "        if (graded && fresh && fresh.graded && typeof fresh.graded === "
        "'object' && !pending() && wrap.isConnected) {\n"
        "          replaceEditor(fresh);\n"
        "          return true;\n"
        "        }\n"
        "        const changed = fresh ? adopt(fresh) : true;\n"
    )
    assert redraw in check
    assert check.index("const fresh = await readFresh();") < check.index(redraw)


# --------------------------------------------------------------------------- K4


def test_k4_the_pictures_come_before_the_answers() -> None:
    """K4: the face holds, in order, its heading with Previous and Next,
    Pictures and Street map, the pictures pane, then what the square shows
    beside its answer and the answers; the declarations are as they were."""
    editor = _editor()
    face = _between(editor, "    faceEl.append(", "  } else {")
    order = [
        "OE.el('div', { class: 'face-top' }, faceLive,",
        "facePicturesButton, faceMapButton)",
        "facePictures,",
        "faceExtra,",
        "faceAnswerArea);",
    ]
    places = [face.index(text) for text in order]
    assert places == sorted(places)
    assert face.count("faceAnswerArea") == 1


def test_k4_the_dates_wrap_onto_rows() -> None:
    """K4 (common.js): the dates wrap, each at its own size and never wider
    than the face, and nothing scrolls sideways; the sliver rule that kept
    each date 40 px narrower is gone; the full screen grid is as it was."""
    rules = _rules(_style())
    assert (
        ".face-dates { display: flex; flex-wrap: wrap; gap: 12px; "
        "align-items: flex-start; }\n"
        ".face-dates > * { flex: none; max-width: 100%; }\n" in rules
    )
    assert "calc(100% - 40px)" not in rules
    dates = [line for line in rules.split("\n") if line.startswith(".face-dates")]
    assert dates and not any("overflow-x" in line for line in dates)
    assert (
        ".face.full .face-dates { display: grid; grid-template-columns: "
        "repeat(auto-fill, minmax(260px, 1fr)); overflow: visible; }\n" in rules
    )
    assert ".face.full .face-dates > * { max-width: none; }\n" in rules


def test_k4_the_dates_take_no_focus_of_their_own() -> None:
    """K4: the block of dates no longer scrolls, so it is no stop of its own
    (no tabindex); it is still a named region, and the look-first pane draws
    it through the same block."""
    dates = _between(_script(), "OE.datesBlock = (", "\n};")
    assert (
        "? { class: 'face-dates', role: 'region', 'aria-label': 'Dates, ' + what }"
        in dates
    )
    assert "tabindex" not in dates
    assert "OE.datesBlock(" in _read("result.html")


def test_k4_an_answer_that_moves_on_brings_the_heading_into_view() -> None:
    """K4, after review: the face is taller than the view now that the
    pictures stand above the answers, so after an answer that moves on to
    another square (faceAdvance, after goTo drew it and moved the focus) and
    after a move by the arrow keys or Previous and Next (pageBy), the new
    square's heading row is brought into view, the least scroll ('nearest':
    none while it is in view, so a press of Next, in that row, moves
    nothing); the last answer, which stays on its square, scrolls nothing."""
    editor = _editor()
    head = _between(editor, "  function headInView() {", "\n  }\n")
    assert "const top = faceEl.querySelector('.face-top');" in head
    assert "top.scrollIntoView({ block: 'nearest' });" in head
    advance = _between(editor, "  function faceAdvance() {", "\n  }\n")
    assert advance.endswith("    goTo(next.handle, 'face');\n    headInView();")
    last = _between(advance, "    if (!next) {", "\n    }\n")
    assert "headInView" not in last
    page = _between(editor, "  function pageBy(step) {", "\n  }\n")
    assert page.endswith("    goTo(windows[to].handle);\n    headInView();")
    assert _code(editor).count("headInView()") == 3


# --------------------------------------------------------------------------- K5


def test_k5_the_footer_holds_the_saves_lines() -> None:
    """K5: the counter's row holds its line (countSaid, last), and the footer
    the save's line (saveSaid) after the row with Save labels and before the
    finish press's slot; each is a polite status in place before its first
    words."""
    editor = _editor()
    assert (
        "  const countSaid = OE.el('span', { class: 'small muted oe-count-said', "
        "role: 'status', 'aria-live': 'polite' });\n" in editor
    )
    assert (
        "  const saveSaid = OE.el('div', { class: 'small muted oe-save-said', "
        "role: 'status', 'aria-live': 'polite' });\n" in editor
    )
    assert (
        "counterNum, OE.el('span', {}, 'labelled'), counterNote, savedNote, "
        "countSaid);" in editor
    )
    assert (
        "OE.el('div', { class: 'sheet-footer spread' }, progress, "
        "OE.el('div', { class: 'row' }, done, saveButton), saveSaid,\n"
        "      finishType));" in editor
    )


def test_k5_a_save_that_went_through_is_said_beside_the_counter() -> None:
    """K5, after review: "Saved to ..." stood under the finish button after
    every automatic save and stayed under Save labels beside "1 change not
    saved yet". savedNote, "Saved automatically" (its construction as it
    was), is a polite status; counterSay writes a save that went through
    there (it shows only while nothing waits) and empties the save's line, or
    tells the status line once the editor is off the page; a change that
    waits, or a save under way, empties the save's line unless it says a
    failure (Save labels stays shown with that)."""
    editor = _editor()
    assert (
        "  const savedNote = OE.el('span', { class: 'small muted' }, W.saved);\n"
        in editor
    )
    assert (
        "  savedNote.setAttribute('role', 'status');\n"
        "  savedNote.setAttribute('aria-live', 'polite');\n" in editor
    )
    say = _between(editor, "  function counterSay(text) {", "\n  }\n")
    for text in (
        "if (!wrap.isConnected) { OE.toast(text); return; }",
        "saveSaid.textContent = '';",
        "savedNote.textContent = text;",
    ):
        assert text in say, text
    progress = _between(editor, "  function updateProgress() {", "\n  }\n")
    assert "    savedNote.hidden = showSave;\n" in progress
    assert (
        "    if ((left || state.saving) && !saveSaid.classList.contains('error')) "
        "saveSaid.textContent = '';\n" in progress
    )
    # Its definition, after a save that went through, and after this page's
    # own save read again (adopt's mine).
    assert _code(editor).count("counterSay(") == 3


def test_k5_the_margin_follows_the_footers_height() -> None:
    """K5, after review: fitFoot keeps the footer's height on the sheet as
    --oe-foot (nothing while the footer is not drawn), at the end of
    updateProgress (its stale branch too, before its focus move), of
    footerSay and of counterSay, after the finish press (updateProgress), and
    from a ResizeObserver on the footer, which goes with the editor's other
    listeners when it is replaced."""
    editor = _editor()
    fit = _between(editor, "  function fitFoot() {", "\n  }\n")
    for text in (
        "const foot = wrap.querySelector(':scope > .sheet-footer');",
        "const tall = foot ? foot.offsetHeight : 0;",
        "if (tall) wrap.style.setProperty('--oe-foot', tall + 'px');",
    ):
        assert text in fit, text
    progress = _between(editor, "  function updateProgress() {", "\n  }\n")
    assert progress.endswith("    updateFinish();\n    fitFoot();")
    stale = _between(progress, "    if (state.stale) {", "      return;\n")
    assert stale.index("fitFoot();") < stale.index("OE.focusIfHere(reopenButton);")
    for name in ("footerSay(slot, text, isError = false)", "counterSay(text)"):
        body = _between(editor, f"  function {name} {{", "\n  }\n")
        assert body.endswith("    fitFoot();"), name
    watch = (
        "    const footWatch = new ResizeObserver(() => fitFoot());\n"
        "    footWatch.observe(wrap.querySelector(':scope > .sheet-footer'));\n"
        "    life.signal.addEventListener('abort', () => footWatch.disconnect());\n"
    )
    assert watch in editor
    assert editor.index("  const life = new AbortController();") < editor.index(watch)
    dispose = _between(editor, "  wrap.oeDispose = () => {", "\n  };")
    assert "    life.abort();\n" in dispose


def test_k5_footer_say_keeps_the_role_and_the_words() -> None:
    """K5: footerSay writes a status in its slot, polite, or an alert for an
    error (assertive, the error colour), as OE.toast says them; with the
    editor off the page (another sheet opened in its place) it has no
    footer, and the status line says it, as before."""
    start = "  function footerSay(slot, text, isError = false) {"
    say = _between(_editor(), start, "\n  }\n")
    for text in (
        "if (!wrap.isConnected) { OE.toast(text, isError); return; }",
        "slot.setAttribute('role', isError ? 'alert' : 'status');",
        "slot.setAttribute('aria-live', isError ? 'assertive' : 'polite');",
        "slot.classList.toggle('error', isError);",
        "slot.classList.toggle('muted', !isError);",
        "slot.textContent = text;",
    ):
        assert text in say, text


def test_k5_each_status_moves_with_its_words() -> None:
    """K5: "Every change is saved to ...", a save that failed and "the sheet
    keeps changing elsewhere" go to the save's line under Save labels (the
    last two as errors); "Saved to ..." (after review) beside the counter in
    the place of "Saved automatically" (counterSay, once the save is over and
    updateProgress has shown that place), and so does the line after this
    page's own save (rederived); a save made elsewhere taken in goes beside
    the counter; a save that went through clears the counter's line (its note
    rides on "Saved to"); none goes to the status line any more. The refusal
    of a sheet that changed under the page stays a toast (oeLeave clears
    it)."""
    editor = _editor()
    for text in (SAVED, NOT_SAVED, KEEPS_CHANGING, ELSEWHERE):
        assert text in editor, text
    every = (
        "if (!auto && !quiet) footerSay(saveSaid, 'Every change is saved to ' "
        "+ data.name + '.');"
    )
    assert every in editor
    assert "if (mine) counterSay(W.rereadMine(data.name));" in editor
    save = _between(editor, "  async function save(", "\n  }\n")
    assert save.index(SAVED) < save.index("countSaid.textContent = '';")
    shown = "      updateProgress();\n      if (savedWords) counterSay(savedWords);\n"
    assert shown in save and save.index(SAVED) < save.index(shown)
    assert "+ said + '. ' + tryAgain(), true);" in save
    code = _code(_script())
    for gone in (
        "OE.toast('Saved to ",
        "OE.toast('Every change is saved",
        "OE.toast('Not saved' +",
        "OE.toast('Not saved: the sheet keeps changing",
        "OE.toast(mine ? W.rereadMine",
    ):
        assert gone not in code, gone
    changed = "OE.toast('Not saved: this sheet changed since the panel opened it. '"
    assert changed in editor


def test_k5_a_stale_sheet_empties_the_lines() -> None:
    """K5: once the sheet stopped saving, the counter's and the save's lines
    are emptied: the line beside "Reopen the sheet" says what happened."""
    progress = _between(_editor(), "  function updateProgress() {", "\n  }\n")
    stale = _between(progress, "    if (state.stale) {", "      return;\n")
    assert "countSaid.textContent = '';" in stale
    assert "saveSaid.textContent = '';" in stale


def test_k5_the_lines_sit_in_the_footer_and_the_margin_grows() -> None:
    """K5 (common.js): the save's line takes a row of its own, its words at
    the right (under Save labels); empty, each line is drawn but out of the
    flow. The sheet's controls keep the footer's own height and 8 px over it
    (--oe-foot, which fitFoot sets; 170 px until it does; r5-j B8 kept 140
    px, and a fixed 170 px was smaller than the footer at 380 px with Save
    labels and a line of two rows, after review), the style sheet's one
    scroll margin, still not on the footer's own controls."""
    style = _style()
    assert (
        ".oe-save-said { flex-basis: 100%; text-align: right; "
        "overflow-wrap: anywhere; }\n" in style
    )
    assert (
        ".oe-save-said:empty, .oe-count-said:empty { position: absolute; "
        "width: 1px; height: 1px; margin: 0; overflow: hidden;\n"
        "  clip-path: inset(50%); }\n" in style
    )
    rules = _rules(style)
    assert rules.count("scroll-margin") == 1
    margin = (
        "section.sheet :is(button, select, input, textarea, summary, [tabindex])"
        ":not(.sheet-footer *) { scroll-margin-bottom: "
        "calc(var(--oe-foot, 162px) + 8px); }\n"
    )
    assert margin in rules


def test_every_r5k_string_of_the_sheet_carries_its_wording_mark() -> None:
    """Each entry of OE.sheetWords this track changed or added carries its
    WORDING mark of r5-k, once each: the graded notice and the line by a
    graded sheet's grey finish button (K3), and the switch's two lines
    (K2)."""
    words = _words()
    assert words.count("// WORDING: r5k K3\n") == 2
    assert words.count("// WORDING: r5k K2\n") == 2
