# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Round r5-n (19 Nov 2026), track VIEWS, as specs/r5-n-round.md gives it:
the panels' bugs of the pre-release bug hunt
(specs/bughunt-2026-11-19/confirmed.json, area views).

- JS-1: a one-shot button sends its message once per press. One shared
  guard, common.js ``OE.oncePerPress``, keeps the button aria-disabled while
  its press runs (so it keeps the focus) and drops a press meanwhile, and
  the second click of a fast double-click (its detail 2) even when the host
  answered the first; a key's press (detail 0) counts. Every one-shot
  button the bug names goes through it: "Check how wrong this map is" (a
  look-first card), "Check each class the map draws" (a result card), "Run
  it again" (a result card and the Check card, after a sign-in), "Check
  this map on points" (the Check card of a class map without scores),
  "Continue in the chat" (the sign-in card) and "Tell Claude: this project"
  (the projects card). A double-click sent two messages, so Claude started
  the job twice. After review: the start card's jobs (jobRow, the Asta tile
  too) and the projects card's job buttons (busyButton) drop that second
  click too, and the label sheet's finish button stays busy until the host
  answers its message and drops it too (finishPress, finishControl).
- JS-2: on the Check card no press starts while a part's sheet is being
  opened: drawPicked returns, openZone makes the press aria-disabled as it
  starts (updatePress, whose line under the press then says why,
  SHEET_OPENING), and updatePress keeps it so while any part opens. A press
  made then had the sheet open over its answer and left parts with no press.
- JS-4: on a label sheet that turns stale, updateProgress hands the focus
  to "Reopen the sheet" from the finish row too (``done``), whose finish
  button it removes; the focus fell to the page.
- JS-5: Sign out is aria-disabled, not disabled, while it is answered, so
  it keeps the focus, and a sign-out that fails leaves it there; the
  sign-in card's account card drawn again after a Sign out puts the focus
  on its main button (the sign-in's, or "Use another key").

The views are read as text (no test here runs the scripts). The browser
harness (scripts/mcp_views_e2e.py) draws them: 19a, 19d to 19h and the
Tell Claude press of 4_studio_projects (JS-1: a double-click sends one
message, a press of Enter one more), 19k_start_jobs_once and a result row's
Check in 4_studio_projects (JS-1: the second click alone, after the host
answered the first, sends nothing), 19l_sheet_finish_once (JS-1: a press
while the finish message waits for the host sends nothing),
19i_check_no_press_while_a_sheet_opens (JS-2),
19j_sheet_finish_changed_elsewhere (JS-4), and 19b, 19c,
4j_studio_sign_out_refused and 11e_start_sign_out_refused (JS-5: the focus
after a Sign out, refused or taken).
"""

from __future__ import annotations

import re
from pathlib import Path

from olmoearth_agent.mcp_app import views

#: JS-1: each one-shot button's handler as its page writes it: the page, then
#: the handler's words. The messages are the ones they sent before.
ONE_SHOT = (
    (
        "result.html",
        "onclick: OE.oncePerPress(() => OE.sendJob(app, text, typed)) }, "
        "CHECK_WORDS.button)",
    ),
    (
        "result.html",
        "onclick: OE.oncePerPress(() => "
        "OE.sendJob(app, OE.JOB_TEXT.byClass(classCheck), typed)) },",
    ),
    (
        "result.html",
        "onclick: OE.oncePerPress(() => OE.sendJob(app, 'I have signed in to "
        "OlmoEarth Studio. Run ' + data.tool + ' again.', typed)),",
    ),
    (
        "check.html",
        "onclick: OE.oncePerPress(() => OE.sendJob(app, OE.JOB_TEXT.points("
        "{ result_id: resultId, prediction_id: predictionId }), typed)),",
    ),
    (
        "check.html",
        "onclick: OE.oncePerPress(() => OE.sendJob(app, 'I have signed in to "
        "OlmoEarth Studio. Run ' + tool + ' again.', typed)),",
    ),
    (
        "sign_in.html",
        "onclick: OE.oncePerPress(() => OE.say(app, 'I have signed in to "
        "OlmoEarth Studio. Continue.')),",
    ),
    (
        "studio.html",
        "onclick: OE.oncePerPress((e) => { e.stopPropagation(); "
        "const said = OE.say(app, text); if (also) also(); return said; }),",
    ),
)


def _read(name: str) -> str:
    """A file of views/, as it is on disk."""
    return (views.VIEW_DIR / name).read_text(encoding="utf-8")


def _between(text: str, start: str, end: str) -> str:
    """The part of text from start up to the first end after it."""
    at = text.index(start)
    return text[at : text.index(end, at)]


def _code(text: str) -> str:
    """The text with its comments taken out: JS and CSS block comments, JS
    line comments, and HTML comments."""
    return re.sub(r"/\*.*?\*/|<!--.*?-->|//[^\n]*", "", text, flags=re.S)


def _in_order(text: str, *pieces: str) -> bool:
    """Whether each piece is in text, each after the one before it."""
    at = [text.index(p) for p in pieces]
    return at == sorted(at)


# --------------------------------------------------------------------------- JS-1


def test_js1_one_guard_runs_a_press_once() -> None:
    """JS-1: OE.oncePerPress, defined once in common.js, is busy while fn
    runs, with the button aria-disabled (never disabled, which would drop
    the focus) and on again after, whatever fn does; a press while busy, a
    press of a disabled button and the second click of a double-click
    (detail above 1) do nothing; fn gets the event and is waited for."""
    common = _read("common.js")
    guard = _between(common, "OE.oncePerPress = (fn) => {", "\n};")
    assert _in_order(
        guard,
        "  let busy = false;\n",
        "  return async (e) => {\n",
        "    const button = e.currentTarget;\n",
        "    if (busy || button.disabled || e.detail > 1) return;\n",
        "    busy = true;\n",
        "    button.setAttribute('aria-disabled', 'true');\n",
        "    try { await fn(e); } finally {\n",
        "      busy = false;\n",
        "      button.setAttribute('aria-disabled', 'false');\n",
    )
    assert ".disabled = " not in guard
    assert _code(common).count("OE.oncePerPress = ") == 1


def test_js1_every_one_shot_button_goes_through_the_guard() -> None:
    """JS-1: the seven one-shot buttons the bug names go through
    OE.oncePerPress with the words they sent before; on the result card and
    the Check card no message goes to Claude any other way (every OE.sendJob
    there is the guarded call, and no handler calls it bare); the sign-in
    card's one OE.say and the projects card's Tell Claude are guarded too,
    the Tell button's own press still stopped before the card under it."""
    for name, handler in ONE_SHOT:
        assert handler in _read(name), (name, handler)
    for name, n in (("result.html", 3), ("check.html", 2)):
        code = _code(_read(name))
        assert code.count("OE.sendJob(") == n, name
        assert code.count("OE.oncePerPress(() => OE.sendJob(app, ") == n, name
        assert "onclick: () => OE.sendJob(" not in code, name
    sign_in = _code(_read("sign_in.html"))
    assert sign_in.count("OE.say(") == 1
    assert sign_in.count("OE.oncePerPress(") == 1
    use = _between(_read("studio.html"), "const useButton = ", "\n\n")
    assert use.count("OE.oncePerPress(") == 1 and use.count("OE.say(") == 1
    assert _in_order(use, "e.stopPropagation();", "OE.say(app, text);", "also();")


def test_js1_the_other_job_buttons_drop_a_double_click_too() -> None:
    """JS-1, after review: the start card's jobs (jobRow, the Asta tile's
    too) and the projects card's job buttons (busyButton), which took one
    press at a time already, drop the second click of a fast double-click
    too (detail above 1): a host that answered the first click before the
    second came got the message twice. busyButton still stops its own press
    first, so a click it drops does not reach the row under it."""
    row = _between(_read("start.html"), "function jobRow(", "\n}\n")
    assert _in_order(
        row,
        "  button.addEventListener('click', async (e) => {\n",
        "    if (busy || button.disabled || e.detail > 1) return;\n",
        "    busy = true;\n",
    )
    busy = _between(_read("studio.html"), "function busyButton(", "\n}\n")
    assert _in_order(
        busy,
        "    onclick: async (e) => {\n",
        "      e.stopPropagation();\n",
        "      if (busy || button.disabled || e.detail > 1) return;\n",
        "      busy = true;\n",
    )


def test_js1_the_finish_button_stays_busy_until_its_message_is_answered() -> None:
    """JS-1, after review: the label sheet's finish button is busy again
    (finishing, aria-disabled) once its press's check found nothing changed,
    with no wait between, and stays so until the host answers its message
    (updateFinish leaves aria-disabled alone meanwhile): it was on again
    while the message went, so a second press asked Claude to grade twice.
    Its click drops the second click of a fast double-click (detail above
    1); a key's press (detail 0) counts."""
    script = _read("sheet.js")
    finish = _between(script, "  async function finishPress(message) {", "\n  }\n")
    again = (
        "    } finally {\n"
        "      finishing = false;\n"
        "      finishButton.setAttribute('aria-disabled', String(!ready()));\n"
        "    }\n"
        "    if (changed || state.stale || pending() || !complete()) return;\n"
        "    // r5-n JS-1 (sheet finish): busy again, with no wait since the "
        "finally\n"
        "    // above, until the host answers the message below.\n"
        "    finishing = true;\n"
        "    finishButton.setAttribute('aria-disabled', 'true');\n"
    )
    assert _in_order(
        finish,
        again,
        "    const sent = await OE.sendJob(app, message, finishType);\n",
        "    updateProgress();\n"
        "    finishing = false;\n"
        "    finishButton.setAttribute('aria-disabled', String(!ready()));",
    )
    assert finish.endswith("String(!ready()));")
    update = _between(script, "  function updateFinish() {", "\n  }\n")
    held = "    if (!finishing) button.setAttribute('aria-disabled', String(!ok));\n"
    assert held in update
    control = _between(script, "  function finishControl() {", "\n  }\n")
    assert (
        "        onclick: (e) => { if (e.detail > 1) return; "
        "finishPress(message); },\n" in control
    )


# --------------------------------------------------------------------------- JS-2


def test_js2_no_press_starts_while_a_sheet_opens() -> None:
    """JS-2: drawPicked returns at once while any part's sheet is being
    opened, as while a press is answered; openZone draws the press again
    (updatePress) as it marks its part opening, before any wait, which makes
    it aria-disabled and says why under it; and updatePress keeps the press
    off while any part opens, so a count chosen meanwhile does not turn it
    on."""
    page = _read("check.html")
    press = _between(page, "  async function drawPicked() {", "\n  }\n")
    first = press.split("\n")[1]
    assert first == "    if (busy || models.some((x) => x.opening)) return;"
    swap = _between(
        page, "  async function openZone(m, { hadFocus = null } = {}) {", "\n  }\n"
    )
    assert _in_order(
        swap,
        "    if (busy || m.opening || !m.draw) return;\n",
        "    m.opening = true;\n",
        "    updatePress();\n",
        "await current.oeSettle()",
        "await OE.call(app, 'olmoearth_app_sheet'",
    )
    update = _between(page, "  function updatePress() {", "\n  }\n")
    assert (
        "    press.setAttribute('aria-disabled', String(busy || models.some((x) => "
        "x.opening) || !plan.send.length || plan.over > 0 || gateHeld()));\n"
    ) in update


def test_js2_the_line_under_the_press_says_a_sheet_is_opening() -> None:
    """JS-2, after review: the press greyed while a part's sheet is being
    opened says why in the line under it (pressWhy, the J08 line, which the
    press names as its description): SHEET_OPENING, marked as a string of
    this round, before every other reason but a rule lost."""
    page = _read("check.html")
    assert (
        "// WORDING: r5n\n"
        "const SHEET_OPENING = 'A sheet is opening. Wait for it.';\n" in page
    )
    why = _between(page, "  function pressWhy(plan) {", "\n  }\n")
    assert _in_order(
        why,
        "    if (ruleLost) return '';\n",
        "    if (models.some((x) => x.opening)) return SHEET_OPENING;\n",
        "    const chosen = plan.send.length > 0;\n",
    )
    assert len(re.findall(r"\bSHEET_OPENING\b", _code(page))) == 2
    assert "'aria-describedby': 'oe-check-limit oe-check-cannot'," in page


def test_js2_every_end_of_an_open_draws_the_press_again() -> None:
    """JS-2: the press openZone turned off comes back as the counts stand at
    every end that leaves it on the card: a sheet that could not be read and
    a sheet that opened both draw the card again (renderAll, which runs
    updatePress) once the part is no longer opening. The end where the open
    editor could not save first only comes with an editor open, and the card
    shows no press while one is (updatePress)."""
    page = _read("check.html")
    swap = _between(
        page, "  async function openZone(m, { hadFocus = null } = {}) {", "\n  }\n"
    )
    caught = _between(swap, "    } catch (err) {\n", "\n    }\n")
    assert _in_order(caught, "      m.opening = false;\n", "      renderAll();\n")
    opened = swap[swap.index("    m.opening = false;\n    // The sheet was read") :]
    assert "\n    renderAll();\n" in opened
    assert "const current = opened !== null ? " in swap
    update = _between(page, "  function updatePress() {", "\n  }\n")
    assert _in_order(
        update,
        "    if (opened !== null || !can.length) {\n",
        "      if (pressArea.isConnected) pressArea.remove();\n",
        "press.setAttribute('aria-disabled', ",
    )
    draw = _between(page, "  function renderAll() {", "\n  }\n")
    assert "updatePress();" in draw


# --------------------------------------------------------------------------- JS-4


def test_js4_the_finish_buttons_focus_goes_to_reopen_the_sheet() -> None:
    """JS-4: a sheet that turns stale counts the finish row (``done``) among
    the places whose focus goes to "Reopen the sheet", before that row's
    finish button is replaced; the finish press keeps its button focused
    while it checks (aria-disabled). ``done`` holds only the finish button,
    its two lines and, once stale, the stale line and Reopen the sheet; the
    line of the finish press's message (finishType) is outside it."""
    script = _read("sheet.js")
    progress = _between(script, "  function updateProgress() {", "\n  }\n")
    stale = _between(progress, "    if (state.stale) {", "      return;\n    }")
    assert _in_order(
        stale,
        "      const hadFocus = (!saveButton.hidden && "
        "saveButton.contains(document.activeElement))\n"
        "        || notice.contains(document.activeElement)\n"
        "        || done.contains(document.activeElement);\n",
        "      if (!done.contains(reopenButton)) "
        "done.replaceChildren(staleLine, reopenButton);\n",
        "      if (hadFocus && !quietSave) OE.focusIfHere(reopenButton);\n",
    )
    fills = re.findall(
        r"\bdone\.(?:replaceChildren|prepend|append|before|after)\(([^)]*)\)",
        _code(script),
    )
    assert sorted(fills) == sorted(
        ["staleLine, reopenButton", "button", "everyDoneLine", "notYetLine"]
    )
    assert (
        "OE.el('div', { class: 'row' }, done, saveButton), saveSaid,\n"
        "      finishType)"
    ) in script
    finish = _between(script, "  async function finishPress(message) {", "\n  }\n")
    assert _in_order(
        finish,
        "    finishButton.setAttribute('aria-disabled', 'true');\n",
        "await check({ force: true })",
    )


# --------------------------------------------------------------------------- JS-5


def test_js5_sign_out_keeps_the_focus_while_it_is_answered() -> None:
    """JS-5: Sign out is busy while its call runs, aria-disabled then (it
    was disabled, which drops the focus to the page) and on again after,
    refused or not; a press meanwhile does nothing. It looks busy as every
    busy button does, a text button too (common.js's style)."""
    common = _read("common.js")
    button = _between(common, "OE.signOutButton = ", "\n};")
    assert _in_order(
        button,
        "  let busy = false;\n  const out = OE.el('button', {\n",
        "    onclick: async () => {\n      if (busy) return;\n",
        "      busy = true;\n      out.setAttribute('aria-disabled', 'true');\n",
        "await OE.call(app, 'olmoearth_app_sign_out');",
        "reply(err.message, true, drawnState);",
        "      } finally { busy = false; "
        "out.setAttribute('aria-disabled', 'false'); }\n",
    )
    assert ".disabled" not in _code(button)
    for rule in (
        'button[aria-disabled="true"], button[aria-disabled="true"]:hover {',
        'button.link[aria-disabled="true"], button.link[aria-disabled="true"]:hover {',
    ):
        assert rule in common, rule


def test_js5_the_account_card_drawn_again_gives_its_main_button_the_focus() -> None:
    """JS-5: on the sign-in card, a Sign out that went through draws the card
    again, then puts the focus on its main button (it went nowhere): signed
    out, the sign-in panel's button, the contained one; with the app's own
    key still in effect, the account card has no Sign out and no contained
    button (no fresh sign-in), so the focus goes to its first button, "Use
    another key"."""
    page = _read("sign_in.html")
    card = _between(page, "function accountCard(", "\n}\n")
    assert (
        "onSignOut: (s) => { render(app, s); "
        "OE.focusIfHere(root.querySelector('button.primary') "
        "|| root.querySelector('button')); } }));"
    ) in card
    assert (
        "const another = OE.el('button', { class: 'link', 'aria-expanded': "
        "'false' }, 'Use another key');"
    ) in card
    assert "OE.el('div', { class: 'spread' }, another, actions)" in card
    assert "status.source === 'environment' ? null" in card
    render = _between(page, "function render(app, status, ", "\n}\n")
    assert (
        "root.append(OE.signInPanel(app, { onDone: (s) => "
        "render(app, s, { signedInNow: true }) }));"
    ) in render
    panel = _between(_read("common.js"), "OE.signInPanel = ", "\n};")
    assert "primary = true" in panel
    assert "{ class: primary ? 'primary' : '' }, OE.SIGN_IN_BUTTON" in panel


# --------------------------------------------------------------------------- docs


def test_changelog_and_usage_say_each_item() -> None:
    """The 1.5.0 detail file's marker (spec/changes/1.5.0.md) of this track is
    replaced by one plain line per
    item, each in the round's own list, found by the bug hunt's id it ends
    with (its own number, Jnn, is the round's); spec/Usage.md, which carries
    no round ids, says what changed for the user, pinned by its sentences."""
    stage = Path(__file__).resolve().parents[2]
    changelog = (stage / "spec" / "changes" / "1.5.0.md").read_text(encoding="utf-8")
    assert "@@R5N-VIEWS@@" not in changelog
    head = (
        "  - The pre-release bug fixes, Settings with Asta, and the app icon "
        "(round 5, n, 19 Nov 2026):\n"
    )
    section = changelog.split(head, 1)[1].split("\n- ", 1)[0]
    for item in ("JS-1", "JS-2", "JS-4", "JS-5"):
        lines = [
            line
            for line in section.split("\n")
            if line.endswith(f"(bug hunt: views/{item}).")
        ]
        assert len(lines) == 1, item
        assert re.match(r"    - J\d+: ", lines[0]), lines[0][:40]
    usage = " ".join((stage / "spec" / "Usage.md").read_text(encoding="utf-8").split())
    for said in (
        "the second click of a fast double-click does nothing either, even when "
        "the host answered the first",
        "Until the host answers that message the finish button stays greyed, keeps "
        "the keyboard focus and takes no second press",
        "While a Sign out is answered it is greyed, keeps the keyboard focus and "
        "takes no second press",
        "While a part's sheet is being opened the press is greyed and does nothing, "
        'and the line under it says "A sheet is opening. Wait for it."',
        '"Reopen the sheet" takes the finish button\'s place and its focus',
    ):
        assert said in usage, said
    assert "so does the finish button whose press found" not in usage
