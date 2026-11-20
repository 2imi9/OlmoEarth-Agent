# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Round r5-o (19 Nov 2026), track JOBS, as specs/r5-o-tune.md gives it.

- T2, one job at a time (the owner's screenshot: six job requests piled up
  in Claude's message box; "for user it can only select one each time"). A
  panel cannot take back a message the host queued, so once a job button's
  message reached the chat every job button of the card is off
  (aria-disabled, greyed, a press does nothing) and the pressed button's
  slot says ``OE.JOB_SENT_ONE``. It holds for the start card's tiles (the
  Asta tile too) and the projects panel's job buttons. A message the host
  did not take locks nothing. The lock is the page's, so a card drawn again
  keeps it. The other sends (the sheet's finish, the result card's buttons,
  the sign-in's Continue) keep ``OE.JOB_SENT`` and lock nothing.
- T3: beside a message Claude did not get, a Copy message button
  (``OE.copyButton`` with a label; the sign-in keeps "Copy address").

The views are read as text (no test here runs the scripts). The browser
harness (scripts/mcp_views_e2e.py) draws them: 11_start_card,
19k_start_jobs_once and 4_studio_projects (T2: the lock and its line at 780
and 380 px, a press of another job by a click and by Enter; T3: the Copy
message button, pressed), and 11g_start_settings_asta (T2: the Asta tile).

Round r5-p (19 Nov 2026) changed what T2 pins, and the pins here follow:
the pressed job's line says where to press Send (J2), and the user lifts
the lock with Choose another job (J1: the locked lines end with that text
button and no longer say to ask Claude for another job; a locked button's
title names it). Its own tests are in test_r5p_jobs.py.
"""

from __future__ import annotations

from pathlib import Path

from olmoearth_agent.mcp_app import views

ROOT = Path(__file__).resolve().parents[2]

#: T2: what the pressed job button's slot says once its message is in the chat.
#: Round r5-p J2: it says where to press Send.
JOB_SENT_ONE = "Sent to the chat. Press Send in the message box below."
#: What the other sends' slots say, as before.
JOB_SENT = "Sent to the chat. Claude answers there."
#: T2, after review: a locked job button's title. Round r5-p J1: it names
#: Choose another job.
JOB_LOCKED_TITLE = "One job at a time: press Choose another job first, or ask Claude."
#: T2, after review: the start card's line once its tiles are locked, signed in
#: and signed out, and the projects panel's. Round r5-p J1: each is followed
#: by the Choose another job text button, so no line says to ask Claude for
#: another job; signed out it says that the Studio jobs need a sign-in.
START_NEXT_LOCKED = "One job is in the chat."
START_NEXT_LOCKED_OUT = "One job is in the chat. Sign in for the Studio jobs."
PROJECTS_NEXT_LOCKED = 'One job is in the chat. "Show its maps" still opens a project.'


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


# ----------------------------------------------------------------------- T2


def test_t2_the_one_job_line_is_a_new_constant_and_the_old_line_stays() -> None:
    """T2: OE.JOB_SENT_ONE holds the spec's words (round r5-p J2: its new
    words, marked r5p); OE.JOB_SENT is as it was, for the other sends; the
    lock's state is the page's (module variables in common.js), unlocked at
    first."""
    common = _read("common.js")
    for text in (
        f"// WORDING: r5p\nOE.JOB_SENT_ONE = '{JOB_SENT_ONE}';\n",
        f"// WORDING: r5b J20\nOE.JOB_SENT = '{JOB_SENT}';\n",
        "let jobLockedBy = null;",
        "let jobSending = false;",
        "const jobButtons = new Set();",
    ):
        assert text in common, text
    assert common.count("OE.JOB_SENT_ONE = ") == 1


def test_t2_a_job_message_taken_locks_and_one_not_taken_does_not() -> None:
    """T2: OE.sendJob knows a job's slot by its mark (data-oe-job-slot). For
    such a slot nothing is sent while the jobs are locked or another job's
    message is with the host; the lock comes only once the host took the
    message (sent), so a message not taken locks nothing; the slot then says
    OE.JOB_SENT_ONE, and any other slot OE.JOB_SENT as before."""
    send = _between(
        _read("common.js"), "OE.sendJob = async (app, text, slot) => {", "\n};"
    )
    assert _in_order(
        send,
        "const job = slot ? slot.getAttribute('data-oe-job-slot') : null;",
        "if (job !== null && (jobLockedBy !== null || jobSending)) return false;",
        "if (job !== null) jobSending = true;",
        "const sent = await OE.say(app, text, { quiet: Boolean(slot) });",
        "if (job !== null) jobSending = false;",
        "if (sent && job !== null) OE.lockJobs(job);",
        "line.className = sent ? 'muted' : 'error';",
        "sent ? (job !== null ? OE.JOB_SENT_ONE : OE.JOB_SENT) : OE.JOB_NOT_SENT;",
        "return sent;",
    )
    assert send.count("OE.lockJobs(") == 1


def test_t2_a_job_button_is_marked_goes_off_and_is_made_off_once_locked() -> None:
    """T2: OE.jobButton marks the button (data-oe-job) and its slot
    (data-oe-job-slot, the first button's key on a slot two buttons share);
    OE.lockJobs keeps the pressed job's key and turns every job button made
    so far off, in the page or not (held weakly, after review, so a row that
    was dropped is not kept alive), and tells the page (oe-jobs-locked); a
    job button made after that (a card drawn again) is off from the start,
    and the slot of the pressed job says the one-job line again; OE.jobIdle
    sets aria-disabled from the lock, never the disabled property (the button
    keeps the focus); OE.jobLocked is true only for a job button."""
    common = _read("common.js")
    locked = _between(common, "OE.jobLocked = ", ";\n")
    assert "jobLockedBy !== null" in locked
    assert "button.hasAttribute('data-oe-job')" in locked
    idle = _between(common, "OE.jobIdle = (button) => {", "\n};")
    for text in (
        "const locked = OE.jobLocked(button);",
        "button.setAttribute('aria-disabled', String(locked));",
        "if (locked) button.setAttribute('data-oe-locked', 'true');",
        "if (locked) button.title = OE.JOB_LOCKED_TITLE;",
    ):
        assert text in idle, text
    assert ".disabled = " not in idle
    mark = _between(common, "OE.jobButton = (button, key, slot) => {", "\n};")
    assert _in_order(
        mark,
        "button.setAttribute('data-oe-job', String(key));",
        "const first = !slot.hasAttribute('data-oe-job-slot');",
        "if (first) slot.setAttribute('data-oe-job-slot', String(key));",
        "if (jobLockedBy === null) {",
        "for (const r of jobButtons) if (!r.deref()) jobButtons.delete(r);",
        "jobButtons.add(new WeakRef(button));",
        "return button;",
        "OE.jobIdle(button);",
        "if (first && String(key) === jobLockedBy) {",
        "line.className = 'muted';",
        "line.textContent = OE.JOB_SENT_ONE;",
        "\n  return button;",
    )
    lock = _between(common, "OE.lockJobs = (key) => {", "\n};")
    assert _in_order(
        lock,
        "jobLockedBy = String(key);",
        "for (const r of jobButtons) {",
        "const b = r.deref();",
        "if (b) { OE.jobIdle(b); lockedButtons.add(new WeakRef(b)); }",
        "jobButtons.clear();",
        "document.dispatchEvent(new Event('oe-jobs-locked'));",
    )
    # The key is set in three places: its declaration, the lock, and (round
    # r5-p J1) the unlock, which only the user's Choose another job runs.
    assert common.count("jobLockedBy = ") == 3
    unlock = _between(common, "OE.unlockJobs = () => {", "\n};")
    assert unlock.count("jobLockedBy = null;") == 1
    assert common.count("OE.unlockJobs(") == 1


def test_t2_the_start_cards_tiles_are_job_buttons() -> None:
    """T2: each tile of the start card (jobRow, so the Asta tile too) is a
    job button with its slot, keyed by the job's key; its press does nothing
    while the jobs are locked, and after a press it is on again only when
    they are not (OE.jobIdle); a locked tile's label and line stay readable."""
    page = _read("start.html")
    row = _between(page, "function jobRow(app, job, on, press = null) {", "\n}\n")
    assert _in_order(
        row,
        "const slot = OE.typeSlot();",
        "OE.jobButton(button, 'start:' + job.key, slot);",
        "if (busy || button.disabled || e.detail > 1) return;",
        "if (OE.jobLocked(button)) return;",
        "busy = true;",
        "OE.sendJob(app, job.message, slot)",
        "      busy = false;\n",
        "OE.jobIdle(button);",
    )
    assert "button.setAttribute('aria-disabled', 'false');" not in row
    tile = _between(page, "function astaRow(app) {", "\n}\n")
    assert "jobRow(app, ASTA_JOB, true, async (slot) => {" in tile
    assert "return OE.sendJob(app, ASTA_JOB.message, slot);" in tile
    assert (
        ".jobs button.job[data-oe-locked] .job-label, "
        ".jobs button.job[data-oe-locked] .job-line { color: var(--text-2); }"
    ) in page


def test_t2_the_projects_panels_job_buttons_are_job_buttons() -> None:
    """T2: every busyButton of the projects panel sends a job, so busyButton
    makes a job button with the slot its press writes in: check, look first
    (a prediction's row, an output's row), which-map, run on an area and
    resume a sheet each pass their slot; a press does nothing while the jobs
    are locked, and the button is on again after a press only when they are
    not."""
    page = _read("studio.html")
    busy = _between(page, "function busyButton(", "\n}\n")
    assert busy.startswith("function busyButton(attrs, text, fn, slot, key) {")
    assert _in_order(
        busy,
        "e.stopPropagation();",
        "if (busy || button.disabled || e.detail > 1) return;",
        "if (OE.jobLocked(button)) return;",
        "busy = true;",
        "try { await fn(); } finally {",
        "        busy = false;\n",
        "OE.jobIdle(button);",
        "return OE.jobButton(button, key, slot);",
    )
    # After review: the key is given at every call, never a name or the words.
    assert "aria-label" not in busy and "|| text" not in busy
    assert "button.setAttribute('aria-disabled', 'false');" not in busy
    for text in (
        "() => OE.sendJob(app, String(row.resume_message), slot), slot, "
        "'resume:' + String(row.resume_message));",
        "() => OE.sendJob(app, OE.JOB_TEXT[key]({ result_id: r.id, "
        "prediction_id: p.id, name: p.name }), slot), slot,\n"
        "    'result:' + project.id + ':' + p.id + ':' + r.id);",
        "await OE.sendJob(app, OE.JOB_TEXT[key]({ result_id: list[0].id, "
        "prediction_id: p.id, name: p.name }), slot);\n"
        "  }, slot, 'pred:' + project.id + ':' + p.id);",
        "await OE.sendJob(app, OE.JOB_TEXT.which(chosen[0], chosen[1]), slot);\n"
        "  }, slot, 'which:' + project.id);",
        "'Run on an area', () => OE.sendJob(app, OE.JOB_TEXT.run({ ...m, "
        "project_id: project.id }), slot), slot,\n"
        "      'run:' + project.id + ':' + m.id);",
    ):
        assert text in page, text
    # The five calls above are every call: the sixth "busyButton(" is its own line.
    assert page.count("busyButton(") == 6


def test_t2_the_other_sends_lock_nothing() -> None:
    """T2: the label sheet's finish, the result card's and the Check card's
    buttons and the sign-in's Continue are no job buttons: their pages mark
    none (no OE.jobButton, no data-oe-job), so their slots keep OE.JOB_SENT
    and their press locks nothing; "Tell Claude" on the projects panel goes
    through OE.say, which has no slot."""
    for name in ("sheet.js", "sheet.html", "result.html", "check.html", "sign_in.html"):
        page = _read(name)
        assert "OE.jobButton(" not in page, name
        assert "data-oe-job" not in page, name
        assert "OE.lockJobs(" not in page, name
    assert _read("start.html").count("OE.jobButton(") == 1
    assert _read("studio.html").count("OE.jobButton(") == 1
    use = _between(_read("studio.html"), "const useButton = ", "\n\n")
    assert "OE.say(app, text);" in use and "OE.sendJob(" not in use


def test_t2_a_job_key_is_built_from_ids_never_from_a_name() -> None:
    """T2, after review: a job's key is one per job of the page and the same
    at each drawing, built from ids: the project's with the prediction's,
    the output's or the model's, the project's alone for which-map, and a
    sheet's own resume request. It was the button's spoken name, so a row of
    the same name in another project, drawn after the lock, said its message
    was sent. busyButton takes the key from its caller and has no other."""
    page = _read("studio.html")
    for text in (
        "function predictionRow(app, project, p, compare, kind, model = null) {",
        "function resultRow(app, p, r, done, kind, project) {",
        "resultRow(app, p, r, done && list.length > 1, kind, project));",
        "(p) => predictionRow(app, project, p, compare,",
        "'resume:' + String(row.resume_message)",
        "'pred:' + project.id + ':' + p.id",
        "'result:' + project.id + ':' + p.id + ':' + r.id",
        "'run:' + project.id + ':' + m.id",
        "'which:' + project.id",
    ):
        assert text in page, text
    # The start card's tiles: the job's own key, one per tile.
    assert "OE.jobButton(button, 'start:' + job.key, slot);" in _read("start.html")
    # A slot says the one-job line again only when its key is the pressed
    # job's: a job button of another key, made after the lock, is off with no
    # line.
    mark = _between(
        _read("common.js"), "OE.jobButton = (button, key, slot) => {", "\n};"
    )
    assert _in_order(
        mark,
        "OE.jobIdle(button);",
        "if (first && String(key) === jobLockedBy) {",
        "line.textContent = OE.JOB_SENT_ONE;",
    )


def test_t2_a_locked_job_button_says_why_in_its_title() -> None:
    """T2, after review: the pressed button's slot may be folded away or far
    from the other buttons, so every locked job button carries the reason as
    its title (OE.jobIdle sets it: buttons locked by OE.lockJobs, and those
    made after the lock)."""
    common = _read("common.js")
    assert f"// WORDING: r5p\nOE.JOB_LOCKED_TITLE = '{JOB_LOCKED_TITLE}';\n" in common
    idle = _between(common, "OE.jobIdle = (button) => {", "\n};")
    assert "if (locked) button.title = OE.JOB_LOCKED_TITLE;" in idle


def test_t2_the_compare_hint_does_not_say_to_press_a_locked_button() -> None:
    """T2, after review: two maps ticked while the jobs are locked, the hint
    beside the which-map button, which is off, keeps saying how a comparison
    works (sync reads OE.jobLocked); a lock that comes while two are ticked
    puts those words back at once (the event OE.lockJobs sends). The boxes
    stay as they are, so Claude is still told what is ticked."""
    page = _read("studio.html")
    comparer = _between(page, "function comparer(app, project) {", "\n}\n")
    assert _in_order(
        comparer,
        "{ class: 'small faint compare-hint', 'aria-live': 'polite' }, COMPARE.how);",
        "hint.textContent = two ? COMPARE.ready : COMPARE.how;",
        "if (OE.jobLocked(button)) hint.textContent = COMPARE.how;",
        "for (const box of boxes) box.disabled = two && !box.checked;",
    )
    words = _between(page, "function lockedWords() {", "\n}\n")
    assert (
        "for (const hint of root.querySelectorAll('.compare-hint')) "
        "hint.textContent = COMPARE.how;"
    ) in words
    assert "document.addEventListener('oe-jobs-locked', lockedWords);" in page
    lock = _between(_read("common.js"), "OE.lockJobs = (key) => {", "\n};")
    assert "document.dispatchEvent(new Event('oe-jobs-locked'));" in lock


def test_t2_no_line_says_to_press_a_job_once_the_jobs_are_locked() -> None:
    """T2, after review: once the jobs are locked the start card's line says
    to ask Claude (signed out: to sign in first, then ask) and the line over
    its tiles goes, at each drawing, at each change of the Settings switch
    (offWords asks lockedWords first) and at the moment of the lock (the
    event); the projects panel's line says the same and that "Show its maps"
    still opens a project, when it is drawn and at the lock. Round r5-p J1:
    each of those lines ends with the Choose another job text button
    (OE.chooseAnotherButton), so no line says to ask Claude any more: signed
    out the start card's says that the Studio jobs need a sign-in."""
    start = _read("start.html")
    for name, text, mark in (
        ("START_NEXT_LOCKED", START_NEXT_LOCKED, "r5p"),
        ("START_NEXT_LOCKED_OUT", START_NEXT_LOCKED_OUT, "r5p"),
    ):
        assert f"// WORDING: {mark}\nconst {name} = '{text}';\n" in start, name
        assert "ress a job" not in text, name
    words = _between(start, "function lockedWords() {", "\n}\n")
    assert _in_order(
        words,
        "if (!OE.jobLocked()) return false;",
        "if (!jobsDrawn) return true;",
        "(jobsDrawn.on ? START_NEXT_LOCKED : START_NEXT_LOCKED_OUT) + ' ';",
        "else next.replaceChildren(words, OE.chooseAnotherButton());",
        "const line = list ? list.previousElementSibling : null;",
        "if (line) line.remove();",
        "\n  return true;",
    )
    off = _between(start, "function offWords() {", "\n}\n")
    assert _in_order(
        off,
        "if (lockedWords()) return;",
        "if (!jobsDrawn || jobsDrawn.on) return;",
    )
    assert "document.addEventListener('oe-jobs-locked', lockedWords);" in start
    studio = _read("studio.html")
    name = "PROJECTS_NEXT_LOCKED"
    assert f"// WORDING: r5p\nconst {name} = '{PROJECTS_NEXT_LOCKED}';\n" in studio
    words = _between(studio, "function lockedWords() {", "\n}\n")
    assert _in_order(
        words,
        "if (!OE.jobLocked()) return;",
        "root.querySelector(':scope > .oe-lead p.next');",
        "next.replaceChildren(PROJECTS_NEXT_LOCKED + ' ', OE.chooseAnotherButton());",
    )
    render = _between(studio, "function renderProjects(", "\n}\n")
    assert _in_order(render, "OE.lead(PROJECTS_LEAD),", "  fill();\n  lockedWords();")


# ----------------------------------------------------------------------- T3


def test_t3_a_message_not_taken_has_a_copy_message_button() -> None:
    """T3: OE.sendJob's not-taken branch shows the text as before, in its
    box, and under it OE.copyButton(text, code, 'Copy message'), marked
    r5o; a message taken shows neither."""
    send = _between(
        _read("common.js"), "OE.sendJob = async (app, text, slot) => {", "\n};"
    )
    assert _in_order(
        send,
        "const line = OE.clearSlot(slot);",
        "if (!sent) {",
        "const code = OE.el('code', {}, text);",
        "// WORDING: r5o\n",
        "slot.append(code, OE.copyButton(text, code, 'Copy message'));",
        "line.className = sent ? 'muted' : 'error';",
    )
    assert send.count("OE.copyButton(") == 1


def test_t3_the_copy_button_takes_its_label_and_the_sign_in_keeps_its_own() -> None:
    """T3: OE.copyButton has a label argument, "Copy address" when none is
    given, which is what the sign-in panel's call leaves it."""
    common = _read("common.js")
    copy = _between(
        common,
        "OE.copyButton = (text, shown, label = 'Copy address') => {",
        "\n};",
    )
    assert "OE.el('button', { type: 'button', class: 'link' }, label);" in copy
    assert "OE.copyButton(page.url, address)" in common
    assert common.count("'Copy address'") == 1


# ---------------------------------------------------------------------- docs


def test_the_changelog_and_the_panel_guide_say_both() -> None:
    """The CHANGELOG's Unreleased part has one plain line per item, and
    docs/mcp-app.md says them where they show: the start card and the
    projects panel (one job at a time), and "When a card stops or fails"
    (Copy message, and a message that waits in Claude's box)."""
    log = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    unreleased = _between(log, "## [Unreleased]", "\n## [1.5.0]")
    for text in (
        "- One job at a time: once a job's message is in the chat, the job buttons "
        "of the start card and the projects panel turn off,",
        "- Once the job buttons are off, no line of those two cards still says to "
        "press a job, and each button that is off says why when you point at it.",
        "- A message Claude did not get has a Copy message button under it.",
    ):
        assert text in unreleased, text
    guide = (ROOT / "docs" / "mcp-app.md").read_text(encoding="utf-8")
    start = _between(guide, "## Start card", "\n## ")
    assert "One job at a time: once a job's message is in the chat" in start
    # Round r5-p (J1, J2): the line's new words, and Choose another job.
    assert "Press Send in the message box below." in start
    assert "press **Choose another job** at the end of" in start
    projects = _between(guide, "## Projects panel", "\n## ")
    assert "to start another, press **Choose another\njob** at the end" in projects
    fails = _between(guide, "## When a card stops or fails", "\n**What it means.**")
    assert "with **Copy message** under it" in fails
    assert "is sent when you press Send." in fails
