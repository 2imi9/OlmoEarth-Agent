# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Round r5-d (17 Nov 2026), track START: the start card, the projects panel
and the shared job button, each item as specs/r5-final-audit/build_list.md
gives it. The views are read as text (no test here runs the scripts):

- R04: on a narrow panel the start card's caption says what the strip of
  three boxes shows ("How a check of a Studio map works."), not what the
  docs figure shows; on a wide panel the docs figure's caption stays word
  for word; each is a span of its own, and the panel's width picks the one
  shown at the width that picks the drawing;
- R07: a job button's press whose slot says what became of its message
  (OE.sendJob) writes no toast as well (OE.say's quiet), so a message the
  host did not take is said once; a press with no slot ("Tell Claude: this
  project", the sign-in card's "Continue in the chat", the map's area)
  keeps the toast; no word changes;
- R09: while a prediction's list of several outputs is open, the row's own
  job buttons are away (their press only opened the list), and they come
  back when it closes; the list's first line says there are several, once
  (no toast, no line in the row's slot), and the focus goes to the first
  output's first button; no word changes, and the line that said how many
  maps is gone;
- R12: the start card says where the key goes in the words of every other
  sign-in text (OE.SIGN_IN_KEY_GOES).
"""

from __future__ import annotations

import re

from olmoearth_agent.mcp_app import views

#: R04: the docs figure's caption on a wide panel (r5-c, the owner's figure),
#: word for word.
CAPTION = (
    "How OlmoEarth makes maps (a figure from the OlmoEarth docs). This app comes "
    "after: it checks how often a map is wrong."
)

#: R04: the caption on a narrow panel, where the strip of three boxes shows.
CAPTION_NARROW = "How a check of a Studio map works."

#: R07: OE.say's status line when the host declines a message (as r5-b J20
#: wrote it), and what a job button's slot says then (r5-f F08: "Copy it into
#: the chat:"; r5-b J20 wrote "Type it in the chat:").
NOT_TAKEN = "Claude did not take the message; type it in the chat instead."
JOB_NOT_SENT = "Claude did not get this message. Copy it into the chat:"

#: R09: the outputs list's first line where the run made several maps (r5-f
#: F13's words; r5-c S05 wrote "This run made more than one map. Use the
#: buttons on the one you mean."), the one line a row's press on such a run
#: leaves, and the end of the line that said how many maps (OUTPUTS.several),
#: which is gone.
OUTPUTS_FIRST = (
    "This run made more than one map; Studio names them only by the ids below. "
    "Not sure which is yours? Ask whoever made the run."
)
SEVERAL_GONE = "Pick one in the list under it."

#: R12: where the key goes, as every other sign-in text says it.
KEY_GOES = "The key goes to the OlmoEarth Agent here, not through Claude."

#: R12: the start card's note about the key, whole: since r5-g G2 it ends at
#: where the key goes, and what to do without a Studio account is the note's
#: next line, of its own (start.html accessLine); it said "No Studio account?
#: Ask whoever made the map how to get one of your own. Do not use their
#: key." after KEY_GOES.
KEY_NOTE = (
    "You need your own Studio API key: a secret code that Studio makes for you "
    "(in Studio, open your profile, then API Keys). The key goes to the OlmoEarth "
    "Agent here, not through Claude."
)

#: Words no face holds (r4-i's rule), and no choice is marked recommended.
_OFF_FACE = re.compile(r"\b(?:windows?|recommended)\b", re.I)


def _read(name: str) -> str:
    """A file of views/, as it is on disk."""
    return (views.VIEW_DIR / name).read_text(encoding="utf-8")


def _between(text: str, start: str, end: str) -> str:
    """The part of text from start up to the first end after it."""
    at = text.index(start)
    return text[at : text.index(end, at)]


def _in_order(text: str, *pieces: str) -> bool:
    """Whether each piece is in text, each after the one before it."""
    at = [text.index(p) for p in pieces]
    return at == sorted(at)


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


# --------------------------------------------------------------------------- R04


def test_r04_the_caption_has_a_narrow_form_of_its_own() -> None:
    """R04: overview.js keeps the docs figure's caption word for word and
    adds the narrow one, marked r5d, and returns both; neither has a digit,
    a per cent sign or a word a face never says, and the narrow one does not
    speak of the docs figure, which a narrow panel does not show."""
    script = _read("overview.js")
    assert f"  // WORDING: r5c docs figure\n  const CAPTION = '{CAPTION}';\n" in script
    assert (
        f"  // WORDING: r5d R04\n  const CAPTION_NARROW = '{CAPTION_NARROW}';\n"
        in script
    )
    # r5-k K8: it returns WIDE too, the width the start card places the figure by.
    assert (
        "  return { STYLE, style, figure, CAPTION, CAPTION_NARROW, WIDE };\n" in script
    )
    for text in (CAPTION, CAPTION_NARROW):
        assert re.search(r"[0-9%]", text) is None, text
        assert not _OFF_FACE.search(text), text
    assert "docs" not in CAPTION_NARROW and "figure" not in CAPTION_NARROW
    assert "Studio" in CAPTION_NARROW and "check" in CAPTION_NARROW


def test_r04_the_panels_width_picks_the_caption_as_it_picks_the_drawing() -> None:
    """R04: under 680 px the narrow caption shows and the wide one does not;
    from 680 px up, in the media rule that shows the docs figure and hides
    the strip, the wide one shows and the narrow one does not; the base rule
    comes before the media rule, so the media rule wins on a wide panel."""
    style = _between(_read("overview.js"), "  const STYLE = `", "`;")
    base = ".oe-overview-cap .cap-wide { display: none; }"
    assert base in style
    assert style.count("@media") == 1
    wide = _between(style, "@media (min-width: 680px) {", "\n}\n")
    for rule in (
        ".oe-overview img.wide { display: block; }",
        ".oe-overview svg.narrow { display: none; }",
        ".oe-overview-cap .cap-wide { display: inline; }",
        ".oe-overview-cap .cap-narrow { display: none; }",
    ):
        assert rule in wide, rule
    assert style.index(base) < style.index("@media (min-width: 680px) {")
    assert ".cap-narrow" not in style.replace(wide, "")


def test_r04_the_start_card_draws_each_caption_in_a_span_of_its_own() -> None:
    """R04: the caption's line, outside the figure, holds the wide caption's
    span, then the narrow one's, then the docs link; the start view carries
    both captions' words (overview.js inlined)."""
    block = _between(_read("start.html"), "function overviewBlock(app) {", "\n}\n")
    assert _in_order(
        block,
        "OE.overview.figure(),",
        "OE.el('div', { class: 'small muted oe-overview-cap' },",
        "OE.el('span', { class: 'cap-wide' }, OE.overview.CAPTION),",
        "OE.el('span', { class: 'cap-narrow' }, OE.overview.CAPTION_NARROW),",
        "' ', docs, address));",
    )
    assert "OE.overview.CAPTION, ' '" not in block
    page = views.view_html("start")
    assert CAPTION in page and CAPTION_NARROW in page


# --------------------------------------------------------------------------- R07


def test_r07_a_job_press_with_a_slot_writes_no_toast() -> None:
    """R07: OE.say takes quiet (false unless given); both of its failure
    branches write the failure line and the toast only when not quiet; a
    message taken still clears a failure line an earlier press without a
    slot wrote; a job button's press (OE.sendJob) is quiet whenever it has a
    slot, which says what became of the message. The words are as they
    were."""
    common = _read("common.js")
    head = "OE.say = async (app, text, { quiet = false } = {}) => {"
    assert common.count(head) == 1
    say = _between(common, head, "\n};")
    declined = (
        "    if (answer && answer.isError) {\n"
        "      if (!quiet) {\n"
        "        saidFailed = OE.NOT_TAKEN;\n"
        "        OE.toast(saidFailed, true);\n"
        "      }\n"
        "      return false;\n"
        "    }\n"
    )
    thrown = (
        "  } catch (err) {\n"
        "    if (!quiet) {\n"
        "      saidFailed = 'Could not send the message to Claude: ' + err.message;\n"
        "      OE.toast(saidFailed, true);\n"
        "    }\n"
        "    return false;\n"
    )
    assert declined in say and thrown in say
    cleared = "if (bar && saidFailed && bar.textContent === saidFailed) OE.toast('');"
    assert cleared in say
    assert say.count("OE.toast(") == 3
    send = _between(common, "OE.sendJob = async (app, text, slot) => {", "\n};")
    quiet = "  const sent = await OE.say(app, text, { quiet: Boolean(slot) });\n"
    assert quiet in send
    assert "OE.toast(" not in send
    for text in (NOT_TAKEN, JOB_NOT_SENT):
        assert f"'{text}';" in common, text


def test_r07_a_press_with_no_slot_keeps_the_toast() -> None:
    """R07: only OE.sendJob passes quiet; every other OE.say call (Tell
    Claude on the projects card, the sign-in card's Continue in the chat,
    the map's area) passes the text alone, so a message the host does not
    take is still said in the status line there, the one place that says
    it."""
    callers = []
    for path in sorted(views.VIEW_DIR.iterdir()):
        if path.suffix not in (".html", ".js"):
            continue
        text = path.read_text(encoding="utf-8")
        at = text.find("OE.say(")
        while at >= 0:
            where = f"{path.name}:{text.count(chr(10), 0, at) + 1}"
            args = _call_args(text, at + len("OE.say"))
            callers.append(path.name)
            if path.name == "common.js":
                assert args == ["app", "text", "{ quiet: Boolean(slot) }"], where
            else:
                assert len(args) == 2, (where, args)
                assert "quiet" not in args[1], where
            at = text.find("OE.say(", at + 1)
    assert callers.count("common.js") == 1
    assert {"studio.html", "sign_in.html"} <= set(callers)
    use = _between(_read("studio.html"), "const useButton = ", "\n\n")
    assert "OE.say(app, text);" in use


# --------------------------------------------------------------------------- R09


def test_r09_a_run_of_several_maps_drops_the_rows_own_jobs() -> None:
    """R09: once a completed prediction's outputs are read and there are
    several, fill() draws the list with OUTPUTS.first and each output's
    buttons and marks the run as several; while that list is open (show)
    the row's own job buttons are away, and they come back after the
    toggle when it closes, the toggle staying in place, so it keeps the
    focus; a lone output keeps the row's buttons (r5-c S05). fill() is
    called once, in read() after the results are awaited, and show() only
    from a press, so both run after the row's jobs and their line are
    made."""
    row = _between(_read("studio.html"), "function predictionRow(", "\n}\n")
    fill = _between(row, "const fill = (out) => {", "\n  };")
    assert _in_order(
        fill,
        "const rows = list.map((r) => resultRow(app, p, r, done && list.length > 1,",
        "results.replaceChildren(",
        "OUTPUTS.first), ...rows] : rows));",
        "several = done && list.length > 1;",
    )
    assert ".remove()" not in fill
    show = _between(row, "const show = (shown) => {", "\n  };")
    assert _in_order(
        show,
        "results.hidden = !shown;",
        "const away = shown && several;",
        "for (const b of jobs) {",
        "if (away) b.remove();",
        "else if (b.parentNode !== acts) acts.append(b);",
    )
    assert "replaceChildren" not in show and "toggle.remove" not in show
    acts = "  const acts = OE.el('div', { class: 'row pred-acts' }, toggle, jobs);\n"
    assert _in_order(
        row,
        "const jobs = !done ? [] : [",
        "hasJobs = jobs.length > 0;",
        acts,
        "    acts,\n",
    )
    assert row.count("fill(") == 1
    read = _between(row, "const read = async () => {", "\n  };")
    assert _in_order(read, "await getResults();", "fill(out);")
    assert row.count("show(") == 3
    toggled = _between(row, "toggle.addEventListener('click', async () => {", "\n  });")
    assert "show(results.hidden);" in toggled
    assert (
        _between(row, "const job = (key) => ", "\n  }, slot, 'pred:").count(
            "show(true);"
        )
        == 2
    )


def test_r09_a_rows_press_on_several_says_so_under_the_row() -> None:
    """R09: a row's press that finds several outputs opens the list under
    the row, whose first line (OUTPUTS.first) says there are several, once:
    no toast at the page's foot and no line in the row's slot, which the
    press emptied first; the focus moves, while this panel has it, to the
    first output's first button, the pressed one being away; "Where to look
    first" on several with no class scores opens the list too, so no press
    is left out of view. Closing the list never empties the slot, so a
    "Sent to the chat." an earlier press wrote there stays. No word
    changes; the line that said how many maps (OUTPUTS.several) is gone."""
    page = _read("studio.html")
    row = _between(page, "function predictionRow(", "\n}\n")
    job = _between(row, "const job = (key) => ", "\n  }, slot, 'pred:")
    assert job.index("OE.clearSlot(slot);") < job.index("const out = await read();")
    opened = "    if (list.length > 1) {\n      show(true);\n"
    several = _between(job, opened, "return;")
    assert "OE.focusIfHere(results.querySelector('button'));" in several
    assert "OE.toast(" not in several
    assert "textContent" not in several and "OE.clearSlot(" not in several
    assert "OUTPUTS.several" not in page and SEVERAL_GONE not in page
    first = _between(job, "if (key === 'first' && ", "return;")
    assert (
        "if (list.length > 1) { show(true); "
        "OE.focusIfHere(results.querySelector('button')); }" in first
    )
    assert _in_order(first, "show(true);", "OE.toast(OE.NO_FIRST);")
    outputs = _between(page, "const OUTPUTS = {", "\n};")
    assert f"  first: '{OUTPUTS_FIRST}',\n" in outputs
    assert "several" not in outputs
    show = _between(row, "const show = (shown) => {", "\n  };")
    assert "OE.clearSlot(" not in show and "slot.textContent" not in show


# --------------------------------------------------------------------------- R12


def test_r12_the_start_card_says_where_the_key_goes_as_every_sign_in_text() -> None:
    """R12: the start card's note says where the key goes in the words of
    OE.SIGN_IN_KEY_GOES (the Check card's and a result card's sign-in),
    marked r5d; the rest of the note is as it was, the line on a Studio
    account after it (r5-g G2: a line of its own, the note's next, so the
    sentence ends with KEY_GOES and is marked r5g too); no view keeps the
    old sentence, which said the key goes only to Studio."""
    page = _read("start.html")
    tag = "// WORDING: r5g\n// WORDING: r5d R12\n"
    assert f"{tag}const KEY_NOTE = '{KEY_NOTE}';" in page
    assert f"OE.SIGN_IN_KEY_GOES = '{KEY_GOES}';" in _read("common.js")
    assert KEY_NOTE.endswith(" " + KEY_GOES)
    assert KEY_NOTE.startswith("You need your own Studio API key: ")
    assert "No Studio account?" not in KEY_NOTE
    assert "note: [KEY_NOTE, accessLine(app)]," in page
    old = ("goes only to Studio", "never through Claude", "It stays on this computer")
    for name in views.VIEWS:
        html = views.view_html(name)
        for gone in old:
            assert gone not in html, (name, gone)
