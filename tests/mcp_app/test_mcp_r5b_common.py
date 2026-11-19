# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Round r5-b (17 Nov 2026), track COMMON: what every view shares
(views/common.js), each item as specs/r5-audit/build_list.md gives it. The
files are read as text (no test here runs the scripts):

- J16: a chosen answer on the face is no longer filled dark as the view's
  one action is: the pale selected fill, a dark line, 700 and a tick (with
  empty alt text); a pressed button of a joined toggle is 700 with a dark
  line inside its edge;
- J19: the sign-in panel: its button "Open the sign-in page", a white card,
  its note a few plain lines (the default two; three on the Check card and a
  result card, each with its own first line); the server's note for the
  model names the same button;
- J20: a job button's slot says what became of its message: "Sent to the
  chat. Claude answers there.", or, in the error colour, "Claude did not get
  this message. Copy it into the chat:" (r5-f F08: it said "Type it in the
  chat:") over the message; the slot's line is
  its status region, in place (out of the layout) before the first press; a
  press that sends nothing clears the slot, and a message taken clears
  OE.say's own failure line from the status line;
- J24 (the CSS): the face's pictures strip keeps each date at its top and
  no wider than the strip, and, of several, the edge of the next in view;
- J29 (a) the footer's row with the finish button keeps to the right edge,
  (d) a field's grey hint is darker, (e) an error in the status line is an
  alert.
"""

from __future__ import annotations

import re

from olmoearth_agent.mcp_app import views

#: J19: the sign-in panel's button.
SIGN_IN_BUTTON = "Open the sign-in page"

#: J19: the note's line that says what the button opens, where the key is
#: typed and where Studio makes one.
SIGN_IN_OPENS = (
    "The button opens a sign-in page in your browser, on this computer. Enter your "
    "Studio API key there (in Studio: your profile, then API Keys)."
)

#: J19: where the key goes, as the Check card and a result card say it.
SIGN_IN_KEY_GOES = "The key goes to the OlmoEarth Agent here, not through Claude."

#: J19: the default note's second line until r5-e: where the key goes, and
#: that Studio checks it (r5-e C3: SIGN_IN_KEY_GOES in its place, so this
#: sentence is gone).
SIGN_IN_CHECKED = (
    "The key goes to the OlmoEarth Agent here, not through Claude, and is checked "
    "with Studio before it is used."
)

#: J19: the first line of the note on the Check card and on a result card.
CHECK_NEEDS = "This check needs OlmoEarth Studio."
STEP_NEEDS = "This step needs OlmoEarth Studio."

#: J19: the button's words before this round, gone from every view.
OLD_BUTTON = "Sign in in your browser"

#: J20: what a job button's slot says once Claude took the message, and when
#: it did not (r5-f F08: "Copy it into the chat:", before "Type it in the
#: chat:").
JOB_SENT = "Sent to the chat. Claude answers there."
JOB_NOT_SENT = "Claude did not get this message. Copy it into the chat:"

#: J20: OE.say's status line when the host declines a message (the words as
#: before, now named), which a message taken later clears.
NOT_TAKEN = "Claude did not take the message; type it in the chat instead."

#: Words no face holds (r4-i's rule), and no choice is marked recommended.
_OFF_FACE = re.compile(
    r"\b(?:windows?|budget|stratum|design|review set|recommended)\b", re.I
)


def _read(name: str) -> str:
    """A file of views/, as it is on disk."""
    return (views.VIEW_DIR / name).read_text(encoding="utf-8")


def _between(text: str, start: str, end: str) -> str:
    """The part of text from start up to the first end after it."""
    at = text.index(start)
    return text[at : text.index(end, at)]


def _code(script: str) -> str:
    """The text with its comments taken out (a block, or the rest of a line
    after two slashes)."""
    return re.sub(r"/\*.*?\*/|//[^\n]*", "", script, flags=re.S)


def _style() -> str:
    """The style sheet every view gets (common.js OE.STYLE), comments and all."""
    return _between(_read("common.js"), "OE.STYLE = `", "\n`;\n")


def _joined(python: str) -> str:
    """Python source with string literals that follow one another over lines
    read as one (the closing quote, the line break and the opening quote
    dropped)."""
    return re.sub(r'"\s*\n\s*f?"', "", python)


def _call_args(text: str, at: int) -> list[str]:
    """The arguments of the call whose opening bracket is text[at], each
    stripped: split at the commas outside any nested bracket and outside any
    quoted string ('...', "..." or `...`)."""
    args: list[str] = []
    depth = 0
    start = at + 1
    i = at
    while True:
        c = text[i]
        if c in "'\"`":
            i += 1
            while text[i] != c:
                i += 2 if text[i] == "\\" else 1
        elif c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
            if not depth:
                args.append(text[start:i].strip())
                return [a for a in args if a]
        elif c == "," and depth == 1:
            args.append(text[start:i].strip())
            start = i + 1
        i += 1


def test_j16_a_chosen_answer_is_not_filled_as_the_one_action() -> None:
    """J16: a chosen answer on the face has the pale selected fill, the
    text's colour, a dark line and 700, with a tick before its words (a
    pseudo-element with empty alt text, so both the button's own text and its
    accessible name stay as they were; a browser without the alt form keeps
    the first content, the tick alone); the dark fill is left to the
    contained button alone."""
    style = _style()
    chosen = (
        '.face .answers button[aria-pressed="true"] { background: var(--selected);'
        " color: var(--text); border-color: var(--text);\n"
        "  box-shadow: inset 0 0 0 1px var(--text); font-weight: 700; }"
    )
    assert chosen in style
    # A tick and a no-break space, then the same with empty alt text; the
    # backslashes are doubled in the script, since the style sheet is a
    # template string.
    tick = (
        '.face .answers button[aria-pressed="true"]::before'
        ' { content: "\\\\2713\\\\00a0"; content: "\\\\2713\\\\00a0" / ""; }'
    )
    assert tick in style
    # The dark fill: the contained button's, and no other rule's.
    assert style.count("background: var(--text)") == 1
    assert "button.primary { background: var(--text); color: var(--canvas);" in style
    assert style.count("color: var(--canvas)") == 1
    # An answer not chosen still rests as it is under the pointer (r4-h A/F22).
    assert (
        '.face .answers button:not([aria-pressed="true"]):not([aria-disabled="true"])'
        ":not(:disabled):hover {\n  background: transparent;" in style
    )


def test_j16_a_pressed_toggle_is_told_by_more_than_its_fill() -> None:
    """J16: on every joined toggle (OE.toggle, and the list's answers, which
    use the same class) the pressed button is 700 with a 2px dark line inside
    its edge, over the pale fill it had."""
    style = _style()
    pressed = (
        '.toggle button[aria-pressed="true"] { background: var(--selected);'
        " color: var(--text); font-weight: 700;\n"
        "  box-shadow: inset 0 0 0 2px var(--text); }"
    )
    assert pressed in style
    # The pressed rule comes after the hover rule, so a pressed button under
    # the pointer keeps its look.
    assert style.index(".toggle button:hover {") < style.index(pressed)
    toggle = _between(_read("common.js"), "OE.toggle = (", "\n};")
    assert "class: 'toggle' + (small ? ' small' : '')" in toggle
    assert "'aria-pressed': String(v === value)" in toggle


def test_j19_the_sign_in_panels_words() -> None:
    """J19: the button says what its press does; the default note is two
    plain lines; the card is white (no 'summary' class), and each line of the
    note is its own div."""
    common = _read("common.js")
    for text in (
        f"// WORDING: r5b J19\nOE.SIGN_IN_BUTTON = '{SIGN_IN_BUTTON}';",
        f"// WORDING: r5b J19\nOE.SIGN_IN_OPENS = '{SIGN_IN_OPENS}';",
        f"// WORDING: r5b J19\nOE.SIGN_IN_KEY_GOES = '{SIGN_IN_KEY_GOES}';",
        # r5-e C3: the default note's second line is SIGN_IN_KEY_GOES itself.
        "// WORDING: r5e\nOE.SIGN_IN_NOTE ="
        " [OE.SIGN_IN_OPENS, OE.SIGN_IN_KEY_GOES];",
    ):
        assert text in common, text
    assert SIGN_IN_CHECKED not in common
    panel = _between(common, "OE.signInPanel = ", "\n};")
    for text in (
        "const button = OE.el('button', { class: primary ? 'primary' : '' },"
        " OE.SIGN_IN_BUTTON);",
        # r5-g G2: a line may be an element (the start card's Request access
        # line), placed as given; a string is a div of its own, as before.
        "const given = (Array.isArray(note) ? note : [note])"
        ".filter((s) => (typeof s === 'string' && s) || s instanceof Node);",
        "const lines = given.length ? given : OE.SIGN_IN_NOTE;",
        "return OE.el('section', { class: 'card dialog stack' },",
        "OE.el('div', { class: 'oe-sign-note' },"
        " lines.map((s) => (s instanceof Node ? s : OE.el('div', {}, s)))),",
    ):
        assert text in panel, text
    assert "summary" not in _code(panel)
    assert ".oe-sign-note > div + div { margin-top: 4px; }" in _style()
    # The old words are gone: the button's, and the one long sentence.
    assert OLD_BUTTON not in common
    assert "served on this computer" not in common
    # No line is long: the old note was one sentence of 40 words.
    for line in (SIGN_IN_OPENS, SIGN_IN_KEY_GOES):
        assert max(len(s.split()) for s in line.split(". ")) <= 20, line
        assert not _OFF_FACE.search(line), line


def test_j19_the_check_card_and_a_result_card_say_it_in_three_lines() -> None:
    """J19: the Check card's and a result card's sign-in note is three plain
    lines: the card's own, then the panel's two (what the button opens, where
    the key goes); the start card keeps its own one sentence (r5-a's), then
    (r5-g G2) its Request access line."""
    check = _between(_read("check.html"), "function signIn(app, tool) {", "\n}\n")
    # Round r5-e, K6: the Check card's note is the panel's two lines; its own
    # first line said the sign-in's heading again.
    assert (
        "    // WORDING: r5e\n    // WORDING: r5b J19\n"
        "    note: [OE.SIGN_IN_OPENS, OE.SIGN_IN_KEY_GOES],\n" in check
    )
    assert CHECK_NEEDS not in check
    result = _between(_read("result.html"), "function show(app, params) {", "\n}\n")
    # Round r5-e, R2: a result card's note is the panel's two lines; its own
    # first line said the sign-in's heading again.
    assert (
        "      // WORDING: r5b J19; r5e R2\n"
        "      note: [OE.SIGN_IN_OPENS, OE.SIGN_IN_KEY_GOES],\n" in result
    )
    assert STEP_NEEDS not in result
    for page in (check, result):
        assert "This tool needs OlmoEarth Studio" not in page
        assert "served on this" not in page
        assert "goes from there" not in page
    # r5-g G2: the start card's note is its sentence, then its Request access line.
    assert "note: [KEY_NOTE, accessLine(app)]," in _read("start.html")


def test_j19_every_view_and_the_server_name_the_new_button() -> None:
    """J19: every view carries the new button and none the old one or the
    old sentence; the note the server gives the model when the sign-in panel
    opens names the button as the panel does."""
    for name in views.VIEWS:
        page = views.view_html(name)
        assert f"OE.SIGN_IN_BUTTON = '{SIGN_IN_BUTTON}';" in page, name
        assert OLD_BUTTON not in page, name
        assert "served on this computer" not in page, name
    server = (views.VIEW_DIR.parent / "server.py").read_text(encoding="utf-8")
    said = _joined(server)
    assert f"The user presses '{SIGN_IN_BUTTON}' there and enters" in said
    assert OLD_BUTTON not in said


def test_j20_a_job_buttons_slot_says_what_became_of_its_message() -> None:
    """J20: the slot's first child, its status line, is its one live part:
    once Claude took the message the line says so, muted, and when it did
    not, in the error colour, with the whole message under it, outside the
    line, so a screen reader hears the line and not the message's ids. Until
    a press the line is empty and the slot out of the layout, but drawn, so
    the line is in the accessibility tree before its first words."""
    common = _read("common.js")
    for text in (
        f"// WORDING: r5b J20\nOE.JOB_SENT = '{JOB_SENT}';",
        f"// WORDING: r5f F08\nOE.JOB_NOT_SENT = '{JOB_NOT_SENT}';",
        "OE.typeSlot = () => OE.el('div', { class: 'oe-type small' },"
        " OE.el('div', { role: 'status' }));",
    ):
        assert text in common, text
    # Each press empties the slot and keeps its line, then writes the line;
    # the message to type goes into the slot, after the line, never into it.
    clear = _between(common, "OE.clearSlot = (slot) => {", "\n};")
    assert clear.splitlines()[1:] == [
        "  const line = slot.firstElementChild;",
        "  while (line.nextSibling) line.nextSibling.remove();",
        "  line.textContent = '';",
        "  return line;",
    ]
    send = _between(common, "OE.sendJob = async (app, text, slot) => {", "\n};")
    # r5-d R07: with a slot, OE.say is quiet (the slot says it, no toast).
    assert send.splitlines()[1:] == [
        "  const sent = await OE.say(app, text, { quiet: Boolean(slot) });",
        "  if (slot) {",
        "    const line = OE.clearSlot(slot);",
        "    if (!sent) slot.append(OE.el('code', {}, text));",
        "    line.className = sent ? 'muted' : 'error';",
        "    line.textContent = sent ? OE.JOB_SENT : OE.JOB_NOT_SENT;",
        "  }",
        "  return sent;",
    ]
    # The old line, with no reason beside it, is gone from the code.
    assert "Type this in the chat:" not in _code(common)
    # The empty slot: out of the layout, but drawn (no display: none), and
    # the message is still selectable whole.
    style = _style()
    assert (
        ".oe-type:has(> [role=status]:empty) { position: absolute; width: 1px;"
        " height: 1px; margin: 0; overflow: hidden;\n  clip-path: inset(50%); }"
        in style
    )
    assert ".oe-type:empty" not in style
    assert "display: none" not in _between(style, ".oe-type {", ".oe-type code {")
    assert (
        "user-select: all; overflow-wrap: anywhere; white-space: pre-wrap; }" in style
    )
    for line in (JOB_SENT, JOB_NOT_SENT):
        assert not _OFF_FACE.search(line), line


def test_j20_a_message_taken_clears_only_the_failure_line_of_say() -> None:
    """J20: OE.say's toasts stay, its failure line named; once a later
    message is taken it clears the status line, and only while that still
    shows the failure line OE.say wrote last, so a card never says "Sent to
    the chat" beside "Claude did not take the message", and any other line
    (a save, a sign-in) stays. r5-d R07: a quiet call (a job button's, whose
    slot says it) writes no failure line; the success path is as it was."""
    common = _read("common.js")
    head = "OE.say = async (app, text, { quiet = false } = {}) => {"
    named = (
        f"// WORDING: r5b J20\nOE.NOT_TAKEN = '{NOT_TAKEN}';\n"
        "let saidFailed = null;  // the failure line OE.say last wrote\n"
    )
    assert named + head in common
    say = _between(common, head, "\n};")
    for text in (
        "      if (!quiet) {\n"
        "        saidFailed = OE.NOT_TAKEN;\n"
        "        OE.toast(saidFailed, true);\n"
        "      }\n"
        "      return false;\n",
        "    const bar = document.getElementById('oe-toast');\n"
        "    if (bar && saidFailed && bar.textContent === saidFailed) OE.toast('');\n"
        "    saidFailed = null;\n"
        "    return true;\n",
        "    if (!quiet) {\n"
        "      saidFailed = 'Could not send the message to Claude: ' + err.message;\n"
        "      OE.toast(saidFailed, true);\n"
        "    }\n"
        "    return false;\n",
    ):
        assert text in say, text
    # The success path clears no other line: one empty toast, behind the test.
    assert say.count("OE.toast(") == 3
    assert say.count("OE.toast('')") == 1
    assert NOT_TAKEN not in _code(say)


def test_j20_every_job_press_writes_into_a_slot_of_its_own() -> None:
    """J20, each call rather than each file: every OE.sendJob call in views/
    passes three arguments, the third a bare name that the same file binds to
    OE.typeSlot(), so no press sends with no slot (and seems to have done
    nothing); the files that send a job are the views with job buttons."""
    senders = []
    for path in sorted(views.VIEW_DIR.iterdir()):
        if path.suffix not in (".html", ".js") or path.name == "common.js":
            continue
        text = path.read_text(encoding="utf-8")
        at = text.find("OE.sendJob(")
        if at >= 0:
            senders.append(path.name)
        while at >= 0:
            where = f"{path.name}:{text.count(chr(10), 0, at) + 1}"
            args = _call_args(text, at + len("OE.sendJob"))
            assert len(args) == 3, (where, args)
            name = args[2]
            assert re.fullmatch(r"[A-Za-z_$][\w$]*", name), (where, name)
            bound = rf"\b(?:const|let)\s+{re.escape(name)}\s*=\s*OE\.typeSlot\(\)"
            assert re.search(bound, text), (where, name)
            at = text.find("OE.sendJob(", at + 1)
    assert {"check.html", "result.html", "sheet.js", "start.html"} <= set(senders)
    assert "studio.html" in senders


def test_j20_a_press_that_sends_nothing_leaves_no_word_of_the_last() -> None:
    """J20: on the projects card a prediction's two jobs share one slot, and
    the comparer's button has its own; each press clears it first
    (OE.clearSlot, which keeps the status line in place), so a press that
    sends nothing (several outputs: the list opens, its first line says so,
    r5-d R09; no class scores: a toast) does not leave "Sent to the chat"
    from an earlier press under it."""
    studio = _read("studio.html")
    row = _between(studio, "function predictionRow(", "\n}\n")
    assert "OE.clearSlot(slot);\n    const out = await read();\n" in row
    assert row.index("OE.clearSlot(slot);") < row.index("OE.toast(OE.NO_FIRST);")
    pair = _between(studio, "function comparer(", "\n}\n")
    assert "OE.clearSlot(slot);\n    const pair = [...picked.values()];\n" in pair
    assert pair.index("OE.clearSlot(slot);") < pair.index("OE.toast(")


def test_j24_the_pictures_strip_keeps_each_date_in_view() -> None:
    """J24 (the CSS): in the face's strip a date starts at the top (a month
    without a picture is not stretched beside the real ones) and is never
    wider than the strip (its row then wraps Detail under Context); the full
    screen grid is as it was, each date filling its cell. r5-k K4
    (tests/mcp_app/test_mcp_r5k_sheet.py): the dates wrap onto rows and
    nothing scrolls sideways, so the rule that kept each of several dates
    40px narrower, as a sign that the strip went on, is gone."""
    style = _style()
    strip = (
        ".face-dates { display: flex; flex-wrap: wrap; gap: 12px;"
        " align-items: flex-start; }\n"
        ".face-dates > * { flex: none; max-width: 100%; }\n"
    )
    assert strip in style
    grid = (
        ".face.full .face-dates { display: grid; grid-template-columns:"
        " repeat(auto-fill, minmax(260px, 1fr)); overflow: visible; }\n"
        ".face.full .face-dates > * { max-width: none; }\n"
    )
    assert grid in style
    assert ".face-dates > :not(:only-child) {" not in style
    assert style.index(strip) < style.index(grid)


def test_j29_the_footers_row_keeps_to_the_right_edge() -> None:
    """J29 (a): the footer's row that holds the finish button has an auto
    left margin and ends at the right, so it does not jump to mid-row when
    the line beside the button makes it wrap, and the row inside it (the
    line and the button) ends at the right too. The rule names div.row: the
    counter, the footer's first child, is a span of class row and stays at
    the left (the build list's '.sheet-footer > .row' would move it)."""
    style = _style()
    row = ".sheet-footer > div.row { margin-left: auto; justify-content: flex-end; }"
    inner = ".sheet-footer > div.row > .row { justify-content: flex-end; }"
    assert row + "\n" + inner + "\n" in style
    assert style.index(".sheet-footer { position: sticky;") < style.index(row)
    assert ".sheet-footer > .row {" not in style
    sheet = _read("sheet.js")
    assert "const progress = OE.el('span', { class: 'row'," in sheet
    assert "const done = OE.el('div', { class: 'row' });" in sheet
    assert "OE.el('div', { class: 'sheet-footer spread' }, progress," in sheet
    # r5-k K5: the line of the saves (saveSaid) under that row.
    assert "OE.el('div', { class: 'row' }, done, saveButton), saveSaid," in sheet


def test_j29_a_fields_grey_hint_is_darker() -> None:
    """J29 (d): the placeholder's colour is the text's at .7 (r5-f F03: 5.3:1
    on a white field; J29 set .6, 4.0:1; before, .42)."""
    style = _style()
    assert "input::placeholder { color: rgba(10,50,53,.7); }" in style
    assert "rgba(10,50,53,.6)" not in _code(style)
    assert "rgba(10,50,53,.42)" not in _code(style)


def test_j29_an_error_in_the_status_line_is_an_alert() -> None:
    """J29 (e): the status line is made a polite status region; a message
    sets its role and aria-live each time: an error alert and assertive, any
    other status and polite, before the text changes."""
    common = _read("common.js")
    region = _between(common, "OE.toastRegion = () => {", "\n};")
    made = "id: 'oe-toast', class: 'small', role: 'status', 'aria-live': 'polite',"
    assert made in region
    toast = _between(common, "OE.toast = (text, isError = false) => {", "\n};")
    lines = [line.strip() for line in toast.splitlines()[1:]]
    assert lines == [
        "const bar = OE.toastRegion();",
        "bar.setAttribute('role', isError ? 'alert' : 'status');",
        "bar.setAttribute('aria-live', isError ? 'assertive' : 'polite');",
        "bar.className = 'small ' + (isError ? 'error' : 'muted');",
        "bar.textContent = text;",
    ]
