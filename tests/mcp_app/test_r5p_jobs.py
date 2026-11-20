# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Round r5-p (19 Nov 2026), track JOBS, as specs/r5-p-round.md gives it.

- J1, Choose another job (the owner, labelling live: after deleting the
  queued message from Claude's message box he could no longer pick another
  job). The host never tells a panel that its message was taken back, so
  the user lifts the lock: once a job is sent, the locked line of the start
  card and of the projects panel ends with a "Choose another job" text
  button. A press turns every job button of the card back on (aria-disabled
  false, no data-oe-locked, the title it had before the lock or none),
  empties the job slots, puts the lead and hint lines back and gives the
  focus to the first job. Its title says what to do first with a message
  that still waits; a locked button's title names it.
- J2: the pressed job's line says where to press Send.
- J3: Settings' "Remove kept copies" needs two presses: the first makes it
  say "Press again to remove" for six seconds; the second, within that
  time, removes; Escape, or the focus moved away, puts it back.
- J4: olmoearth_submit_prediction's description says that it runs a model
  on Studio's compute and uses the account's credit, and to tell the user
  the model, the area and the dates and wait for their yes.

The views are read as text (no test here runs the scripts). The browser
harness (scripts/mcp_views_e2e.py) draws them: 19k_start_jobs_once,
11_start_card, 11g_start_settings_asta and 4_studio_projects press Choose
another job (jobs_unlocked), and 11g_start_settings_asta presses Remove kept copies
twice (settings_remove_twice).
"""

from __future__ import annotations

from pathlib import Path

from olmoearth_agent.mcp_app import views
from olmoearth_agent.tools.predict import build_predict_tools

ROOT = Path(__file__).resolve().parents[2]

#: J1: the text button's words and its title.
CHOOSE_ANOTHER = "Choose another job"
CHOOSE_ANOTHER_TITLE = (
    "If the first message still waits in the message box, delete it there first."
)
#: J1: what pointing at a locked job button says.
JOB_LOCKED_TITLE = "One job at a time: press Choose another job first, or ask Claude."
#: J2: the pressed job's line, and its words before this round.
JOB_SENT_ONE = "Sent to the chat. Press Send in the message box below."
JOB_SENT_ONE_OLD = "If it waits there, press Send."
#: J3: the button's words after its first press, and for how long (ms).
SET_REMOVE_AGAIN = "Press again to remove"
SET_REMOVE_WAIT = 6000
#: J4: the sentence the tool's description ends with.
RUN_CONFIRM = (
    "It runs a model on Studio's compute and uses the account's credit: before "
    "calling it, tell the user the model, the area and the dates, and wait for "
    "their yes."
)


def _read(name: str) -> str:
    """A file of views/, as it is on disk."""
    return (views.VIEW_DIR / name).read_text(encoding="utf-8")


def _between(text: str, start: str, end: str) -> str:
    """The text from ``start`` to the next ``end`` after it."""
    at = text.index(start)
    return text[at : text.index(end, at)]


def _in_order(text: str, *pieces: str) -> bool:
    """Whether each piece is in text, each after the one before it."""
    at = [text.index(p) for p in pieces]
    return at == sorted(at)


# ----------------------------------------------------------------------- J1


def test_j1_the_text_button_has_the_specs_words_and_title() -> None:
    """J1: OE.CHOOSE_ANOTHER and OE.CHOOSE_ANOTHER_TITLE hold the spec's
    words, marked r5p; OE.chooseAnotherButton makes a real button (so a
    click, Enter and Space press it) of the text-button class, with that
    title, whose press runs OE.unlockJobs and nothing else."""
    common = _read("common.js")
    for text in (
        f"// WORDING: r5p\nOE.CHOOSE_ANOTHER = '{CHOOSE_ANOTHER}';\n",
        f"// WORDING: r5p\nOE.CHOOSE_ANOTHER_TITLE = '{CHOOSE_ANOTHER_TITLE}';\n",
    ):
        assert text in common, text
    button = _between(common, "OE.chooseAnotherButton = () => ", ";\n")
    assert _in_order(
        button,
        "OE.el('button', {",
        "type: 'button', class: 'link oe-choose-another', "
        "title: OE.CHOOSE_ANOTHER_TITLE,",
        "onclick: () => OE.unlockJobs(),",
        "}, OE.CHOOSE_ANOTHER)",
    )
    # The one place that lifts the lock: no view calls it on its own.
    assert common.count("OE.unlockJobs(") == 1
    for name in ("start.html", "studio.html", "sheet.js", "result.html"):
        assert "OE.unlockJobs(" not in _read(name), name


def test_j1_a_locked_button_names_choose_another_job_in_its_title() -> None:
    """J1: pointing at a locked job button says OE.JOB_LOCKED_TITLE, the
    spec's words, marked r5p; the words of round r5-o are gone."""
    common = _read("common.js")
    assert f"// WORDING: r5p\nOE.JOB_LOCKED_TITLE = '{JOB_LOCKED_TITLE}';\n" in common
    assert "to start another, ask Claude" not in common


def test_j1_the_unlock_turns_every_locked_button_back_on() -> None:
    """J1: OE.unlockJobs does nothing with no lock; else it lifts the lock
    first, then every button the lock turned off (those made before it, and
    those made while it stood: both are kept in lockedButtons) loses
    data-oe-locked, gets back the title it had before the lock (none: no
    title) and is aria-disabled false, and is a job button of the next lock
    again; every job slot is emptied, the ones that said the one-job line
    even out of the page; the page is told last (oe-jobs-unlocked). The
    disabled property is never touched: a tile off for a missing sign-in
    stays off."""
    common = _read("common.js")
    unlock = _between(common, "OE.unlockJobs = () => {", "\n};")
    assert _in_order(
        unlock,
        "if (jobLockedBy === null) return;",
        "jobLockedBy = null;",
        "for (const r of lockedButtons) {",
        "b.removeAttribute('data-oe-locked');",
        "const title = titleBefore.get(b);",
        "titleBefore.delete(b);",
        "if (typeof title === 'string') b.setAttribute('title', title); "
        "else b.removeAttribute('title');",
        "b.setAttribute('aria-disabled', 'false');",
        "jobButtons.add(new WeakRef(b));",
        "lockedButtons.clear();",
        "for (const r of sentSlots) {",
        "if (s) OE.clearSlot(s);",
        "sentSlots.clear();",
        "document.querySelectorAll('[data-oe-job-slot]')"
        ".forEach((s) => OE.clearSlot(s));",
        "document.dispatchEvent(new Event('oe-jobs-unlocked'));",
    )
    assert ".disabled" not in unlock
    # Both ways a button goes off put it in lockedButtons.
    lock = _between(common, "OE.lockJobs = (key) => {", "\n};")
    assert "if (b) { OE.jobIdle(b); lockedButtons.add(new WeakRef(b)); }" in lock
    mark = _between(common, "OE.jobButton = (button, key, slot) => {", "\n};")
    assert _in_order(
        mark,
        "OE.jobIdle(button);",
        "lockedButtons.add(new WeakRef(button));",
    )


def test_j1_the_lock_keeps_each_title_and_the_slot_of_the_pressed_job() -> None:
    """J1: the lock writes its own title on a button, so OE.jobIdle keeps the
    one the button had (once: a second call while locked does not take the
    lock's own title for it), which the unlock puts back ("Run on an area"
    has a title of its own); the slot that says the one-job line is kept
    weakly (the pressed job's at the lock, and one drawn again while
    locked), so the unlock empties it even when a search took its row out
    of the page."""
    common = _read("common.js")
    for text in (
        "const titleBefore = new WeakMap();",
        "const sentSlots = new Set();",
        "const lockedButtons = new Set();",
    ):
        assert text in common, text
    idle = _between(common, "OE.jobIdle = (button) => {", "\n};")
    assert _in_order(
        idle,
        "if (locked && !titleBefore.has(button)) "
        "titleBefore.set(button, button.getAttribute('title'));",
        "if (locked) button.title = OE.JOB_LOCKED_TITLE;",
    )
    lock = _between(common, "OE.lockJobs = (key) => {", "\n};")
    assert _in_order(
        lock,
        "if (s.getAttribute('data-oe-job-slot') !== jobLockedBy) OE.clearSlot(s);",
        "else sentSlots.add(new WeakRef(s));",
        "document.dispatchEvent(new Event('oe-jobs-locked'));",
    )
    mark = _between(common, "OE.jobButton = (button, key, slot) => {", "\n};")
    assert _in_order(
        mark,
        "line.textContent = OE.JOB_SENT_ONE;",
        "sentSlots.add(new WeakRef(slot));",
    )
    assert "title: RUN_TITLE }" in _read("studio.html")


def test_j1_the_busy_rule_is_whole() -> None:
    """J1: a stray rule for a locked job button stood inside the rule of a
    busy button, where it styled nothing; it is taken out (no rule fades a
    locked button: it is aria-disabled, so the busy rule shows it off, and
    the start card keeps its tiles' words readable), and the busy rule is
    whole again. The text button inside a card's line is underlined and
    takes the line's size."""
    common = _read("common.js")
    assert (
        'button[aria-disabled="true"], button[aria-disabled="true"]:hover '
        "{ cursor: default; background: transparent;\n"
        "  color: rgba(0,0,0,.26); border-color: var(--divider); box-shadow: none; }\n"
    ) in common
    assert "button[data-oe-locked], button[data-oe-locked]:hover" not in common
    assert _in_order(
        common,
        'button.primary[aria-disabled="true"], '
        'button.primary[aria-disabled="true"]:hover',
        "\n.oe-lead button.oe-choose-another { font-size: inherit; "
        "line-height: inherit; padding: 0 2px; text-decoration: underline;",
        "\nbutton.link { border: 0;",
    )


def test_j1_the_start_cards_locked_line_ends_with_the_button() -> None:
    """J1: while the tiles are locked the start card's line is its locked
    words, a space, and the Choose another job button; a line that already
    holds the button keeps it (and its focus) when the words are set again
    (each change of the Settings switch runs lockedWords)."""
    start = _read("start.html")
    words = _between(start, "function lockedWords() {", "\n}\n")
    assert _in_order(
        words,
        "const words = (jobsDrawn.on ? START_NEXT_LOCKED : START_NEXT_LOCKED_OUT)"
        " + ' ';",
        "if (next.querySelector('button.oe-choose-another')) "
        "next.firstChild.textContent = words;",
        "else next.replaceChildren(words, OE.chooseAnotherButton());",
        "if (line) line.remove();",
    )
    back = _between(start, "function unlockedWords() {", "\n}\n")
    assert "OE.chooseAnotherButton()" not in back


def test_j1_the_start_card_puts_its_words_back_and_focuses_the_first_job() -> None:
    """J1: at the unlock (the event oe-jobs-unlocked) the start card's line
    says what a drawing with no lock says for the card as it is now (signed
    in START_NEXT_IN; signed out the lead's own words, or the Studio-jobs
    words while the Asta tile is shown: offWords), the line over the tiles
    stands there again signed out, and the focus goes to the first tile
    that can be pressed, or, with none (signed out, no Asta tile), to the
    card's one contained button; only while the panel's frame has the
    focus. The listener goes with the card."""
    start = _read("start.html")
    words = _between(start, "function unlockedWords() {", "\n}\n")
    assert _in_order(
        words,
        "if (!jobsDrawn) return;",
        "if (next) next.textContent = jobsDrawn.on ? START_NEXT_IN : START_LEAD.next;",
        "if (list && !jobsDrawn.on && !list.previousElementSibling) "
        "list.before(OE.el('div', {}, ''));",
        "offWords();",
        "OE.focusIfHere(root.querySelector('.jobs button.job:not(:disabled)') "
        "|| root.querySelector('button.primary:not(:disabled)'));",
    )
    for text in (
        "document.addEventListener('oe-jobs-unlocked', unlockedWords);",
        "OE.cleanups.add(() => document.removeEventListener("
        "'oe-jobs-unlocked', unlockedWords));",
    ):
        assert text in start, text
    # No words kept on the elements: what the unlock writes is worked out
    # from the card as it is, so a sign-in made while locked is not undone.
    assert "oeOpen" not in start and "removedLine" not in start


def test_j1_the_projects_panel_ends_its_locked_line_with_the_button() -> None:
    """J1: the projects panel's locked line is its words, a space and the
    button, written once (a line that holds the button keeps it); at the
    unlock the line says PROJECTS_LEAD's words again, every compare hint
    says what its boxes now give (its which-map button is on only while two
    are ticked), in the page or out of it, and the focus goes to the first
    job button that is shown and on, or to the first such button of the
    panel."""
    studio = _read("studio.html")
    words = _between(studio, "function lockedWords() {", "\n}\n")
    assert _in_order(
        words,
        "if (next && !next.querySelector('button.oe-choose-another')) {",
        "next.replaceChildren(PROJECTS_NEXT_LOCKED + ' ', OE.chooseAnotherButton());",
        "for (const hint of root.querySelectorAll('.compare-hint')) "
        "hint.textContent = COMPARE.how;",
        "for (const r of compareHints) {",
    )
    back = _between(studio, "function unlockedWords() {", "\n}\n")
    assert _in_order(
        back,
        "if (next) next.textContent = PROJECTS_LEAD.next;",
        "for (const r of compareHints) {",
        "hint.textContent = hintButton.get(hint).disabled "
        "? COMPARE.how : COMPARE.ready;",
        "const shown = (b) => !b.disabled && b.getClientRects().length > 0;",
        "[...root.querySelectorAll('button[data-oe-job]')].find(shown)",
        "|| [...root.querySelectorAll('button')].find(shown);",
        "if (first) OE.focusIfHere(first);",
    )
    comparer = _between(studio, "function comparer(app, project) {", "\n}\n")
    assert _in_order(
        comparer,
        "}, slot, 'which:' + project.id);",
        "compareHints.add(new WeakRef(hint));",
        "hintButton.set(hint, button);",
        "const sync = () => {",
    )
    for text in (
        "document.addEventListener('oe-jobs-unlocked', unlockedWords);",
        "OE.cleanups.add(() => document.removeEventListener("
        "'oe-jobs-unlocked', unlockedWords));",
    ):
        assert text in studio, text
    assert "oeOpen" not in studio


# ----------------------------------------------------------------------- J2


def test_j2_the_pressed_jobs_line_says_where_to_press_send() -> None:
    """J2: OE.JOB_SENT_ONE holds the spec's words, marked r5p, set in one
    place; the words of round r5-o are in no view."""
    common = _read("common.js")
    assert f"// WORDING: r5p\nOE.JOB_SENT_ONE = '{JOB_SENT_ONE}';\n" in common
    assert common.count("OE.JOB_SENT_ONE = ") == 1
    for name in ("common.js", "start.html", "studio.html"):
        assert JOB_SENT_ONE_OLD not in _read(name), name


# ----------------------------------------------------------------------- J3


def test_j3_remove_kept_copies_takes_two_presses() -> None:
    """J3: the button's first press only arms it: its words turn to
    OE.SET_REMOVE_AGAIN and a timer of OE.SET_REMOVE_WAIT (six seconds)
    starts; a press while armed disarms it and sends the one change the
    server knows (remove_kept); the timer's end puts the words back. After
    review: a press while a change is being answered does nothing (the
    guard stands before the armed test), and the second click of a
    double-click (detail 2) does not remove: it leaves the button armed."""
    common = _read("common.js")
    assert f"// WORDING: r5p\nOE.SET_REMOVE_AGAIN = '{SET_REMOVE_AGAIN}';\n" in common
    assert f"OE.SET_REMOVE_WAIT = {SET_REMOVE_WAIT};\n" in common
    block = _between(common, "OE.settingsBlock = (app) => {", "\n};\n")
    assert _in_order(
        block,
        "remove = OE.el('button', { type: 'button', class: 'small', "
        "'data-set': 'kept' }, OE.SET_REMOVE);",
        "let armed = 0;",
        "const disarm = () => {",
        "if (armed) clearTimeout(armed);",
        "remove.textContent = OE.SET_REMOVE;",
        "remove.addEventListener('click', (e) => {",
        "if (!armed) {",
        "remove.textContent = OE.SET_REMOVE_AGAIN;",
        "armed = setTimeout(disarm, OE.SET_REMOVE_WAIT);",
        "disarm();",
        "change(remove, { name: 'remove_kept' });",
    )
    # The first press ends before the change: it returns once armed. A press
    # while busy neither arms nor removes; a double-click's second click,
    # tested only once armed (so its first click still arms), does not remove.
    press = _between(block, "remove.addEventListener('click', (e) => {", "\n      });")
    assert press.count("return;") == 3
    assert _in_order(
        press,
        "(e) => {\n        if (busy) return;\n        if (!armed) {\n",
        "armed = setTimeout(disarm, OE.SET_REMOVE_WAIT);",
        "          return;\n        }\n        if (e.detail > 1) return;\n"
        "        disarm();\n        change(remove, { name: 'remove_kept' });",
    )
    # The change is sent from one place, behind the armed check.
    assert block.count("name: 'remove_kept'") == 1


def test_j3_escape_or_the_focus_moved_away_resets_the_button() -> None:
    """J3: the focus leaving the button disarms it; Escape on the armed
    button disarms it and is taken (preventDefault), so the box's own
    Escape, which skips a key already taken, leaves the box open; Escape on
    a button that is not armed is left to the box, which closes."""
    block = _between(_read("common.js"), "OE.settingsBlock = (app) => {", "\n};\n")
    assert _in_order(
        block,
        "remove.addEventListener('blur', disarm);",
        "remove.addEventListener('keydown', (e) => {",
        "if (e.key !== 'Escape' || !open || e.defaultPrevented) return;",
    )
    key = _between(block, "remove.addEventListener('keydown', (e) => {", "\n      });")
    assert _in_order(
        key,
        "if (e.key !== 'Escape' || !armed) return;",
        "e.preventDefault();",
        "disarm();",
    )


def test_j3_the_armed_words_are_said_to_a_screen_reader() -> None:
    """J3, after review: a screen reader does not read the focused button's
    new words, so a line nobody sees (span.oe-set-armed, aria-live polite,
    no role), put after the button when the box is drawn, says them at the
    first press and is emptied when the press is taken back. The box's own
    status line is not used: nothing new is seen in Settings."""
    common = _read("common.js")
    assert (
        "\n.oe-set-armed { position: absolute; width: 1px; height: 1px; margin: 0; "
        "overflow: hidden; clip-path: inset(50%); }\n"
    ) in common
    block = _between(common, "OE.settingsBlock = (app) => {", "\n};\n")
    assert _in_order(
        block,
        "let heard = null;",
        "heard = OE.el('span', { class: 'oe-set-armed', 'aria-live': 'polite' });",
        "remove.textContent = OE.SET_REMOVE;\n        heard.textContent = '';",
        "remove.textContent = OE.SET_REMOVE_AGAIN;\n"
        "          heard.textContent = OE.SET_REMOVE_AGAIN;",
        "row('kept', OE.SET_KEPT, text(kept.said), remove),",
        "if (remove) remove.after(heard);",
    )
    press = _between(block, "remove.addEventListener('click', (e) => {", "\n      });")
    assert "say(" not in press and "oe-set-said" not in press


def test_j3_nothing_else_in_settings_changes() -> None:
    """J3: the switch and Forget my yes still change at their first press
    (the server's change of the kept copies is covered by
    test_r5n_settings.py)."""
    block = _between(_read("common.js"), "OE.settingsBlock = (app) => {", "\n};\n")
    for text in (
        "toggle.addEventListener('click', () => "
        "change(toggle, { name: 'asta', on: !on }));",
        "forget.addEventListener('click', () => "
        "change(forget, { name: 'forget_pictures' }));",
    ):
        assert text in block, text


# ----------------------------------------------------------------------- J4


def test_j4_the_tool_says_to_confirm_before_it_runs_a_model() -> None:
    """J4: olmoearth_submit_prediction's description ends with the spec's
    sentence, after what it had (the fields it requires and how to poll)."""
    name = "olmoearth_submit_prediction"
    tool = next(t for t in build_predict_tools() if t.spec.name == name)
    text = tool.spec.description
    assert text.endswith(" " + RUN_CONFIRM)
    assert text.startswith("Submit a new prediction. Requires name, project_id, ")
    assert "poll it with olmoearth_get_prediction. It runs a model" in text
    assert text.count(RUN_CONFIRM) == 1


def test_j4_no_other_tool_of_the_file_changed_its_words() -> None:
    """J4: the sentence is on the one tool that starts a run; the tools that
    read predictions do not carry it."""
    for tool in build_predict_tools():
        if tool.spec.name != "olmoearth_submit_prediction":
            assert "wait for their yes" not in tool.spec.description, tool.spec.name


# ---------------------------------------------------------------------- docs


def test_the_changelog_and_the_panel_guide_say_each_item() -> None:
    """The CHANGELOG's Unreleased part has one plain line per item under
    Fixes, and docs/mcp-app.md says them where the job buttons, the Settings
    box and the run of a model are described."""
    log = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    unreleased = _between(log, "## [Unreleased]", "\n## [1.5.1]")
    assert "### Fixes\n" in unreleased
    for text in (
        "- Choose another job: once a job's message is in the chat, the line of the "
        "start card and of the projects panel ends with a",
        '- The pressed job now says "Sent to the chat. Press Send in the message '
        'box below."',
        '- Settings: "Remove kept copies" takes two presses; after the first it '
        'says "Press again to remove" for six seconds.',
        "- Before Claude runs a model on Studio's compute, it tells you the model, "
        "the area and the dates and waits for your yes.",
    ):
        assert text in unreleased, text
    guide = (ROOT / "docs" / "mcp-app.md").read_text(encoding="utf-8")
    start = _between(guide, "## Start card", "\n## ")
    assert "Press Send in the message box below." in start
    assert "press **Choose another job** at the end of" in start
    assert "delete it there first" in start
    projects = _between(guide, "## Projects panel", "\n## ")
    assert "press **Choose another\njob** at the end of the panel's line" in projects
    settings = _between(guide, "## Settings\n", "\n## ")
    assert "Press **Remove kept copies** twice to delete the kept maps" in settings
    assert '"Press again to remove" for six seconds' in settings
    run = _between(guide, "## Draw an area and run a model", "\n## ")
    assert "tells you the model, the area and the dates and waits\nfor your yes" in run
