# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Round r5-i (18 Nov 2026), track OTHER VIEWS, as specs/r5-i-round.md and
specs/r5-g-audit/build_list.md give it: items B4 to B8 of the re-audit's
build list. The views are read as text, as they are on disk (no test here
runs the scripts; the browser harness, scripts/mcp_views_e2e.py, draws them:
2 and 2b, 11_start_card and 11_start_card_access_380, 3_label_sheet, 4, 4d,
4e and 3h):

- B4 (result.html): a plan card's second lead line, once a read of the
  counts says every one is answered and the sheet is not open in the card,
  reads 'Every one is answered. Press "Open the sheet" to get the result.'
  (the button named in double quotes, never bare: rule R4); a read below
  the whole sheet puts the server's words back; an older read that answers
  last, or a read of a card drawn over since, touches nothing.
- B5 (common.js): a docs or Request access link the host did not open says
  so in its own slot ("The link was not opened. " + DOCS_BY_HAND, a status
  in place before the first press), with no toast (OE.open's quiet); every
  other link keeps the toast.
- B6 (studio.html): the closed outputs toggle says "Show outputs" (its
  aria-label follows), 100 px wide, the wider of its two labels.
- B7 (sheet.js): after a set of answers is removed and the sheet cannot be
  read again, the toast says only which answers went, as a plain message;
  the footer says what failed, as the description of "Reopen the sheet",
  which takes the focus of the Remove button pressed.
- B8 (start.html): signed out, "Sign in to start a job." is at body size in
  the body colour (no class).
"""

from __future__ import annotations

import re

from olmoearth_agent.mcp_app import sheet, views

#: B4: the plan card's second lead line at every one answered; the open
#: button's label (sheet.OPEN_DONE at a whole sheet) in double quotes.
PLAN_ALL_ANSWERED = 'Every one is answered. Press "Open the sheet" to get the result.'
#: B4: the server's second lead line (server.PLAN_LEAD's and
#: PLAN_LEAD_POINTS' "next"), which the card keeps on the line and puts back.
PLAN_NEXT = "Answer every one on the sheet to get the result."

#: B5: what the status line says when the host does not open a link (r4-k
#: K7), the docs link's words over the address, and the slot's first line.
LINK_NOT_OPENED = "The link was not opened."
DOCS_BY_HAND = "Open this address in your browser:"
LINK_DECLINED_LINE = "The link was not opened. Open this address in your browser:"

#: B6: the outputs toggle's two labels.
SHOW_OUTPUTS = "Show outputs"
HIDE_OUTPUTS = "Hide outputs"

#: B7: the toast's words after a removal whose read failed, for each set, and
#: the footer's line (W.notReadAgain, r5-b J15), which stays.
REMOVED_WRONG = "The right-or-wrong marks were removed."
REMOVED_CLASSES = "The classes picked were removed."
NOT_READ_AGAIN = (
    "This panel could not read the sheet again. Reopen the sheet to carry on."
)

#: B8: the line over the signed-out job tiles.
SIGN_IN_FIRST = "Sign in to start a job."

#: Words no face holds (round r5-e's rule R7, and the owner's ruling 6).
_OFF_FACE = re.compile(
    r"\b(?:windows?|UTM|zones?|intervals?|strat(?:um|a)|design|budget"
    r"|recommended)\b",
    re.I,
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
    after two slashes, but not the slashes of an address: after a colon, as
    in 'https://', or a slash, as in 'file:///')."""
    return re.sub(r"/\*.*?\*/|(?<![:/])//[^\n]*", "", script, flags=re.S)


def _in_order(text: str, *pieces: str) -> bool:
    """Whether each piece is in text, each after the one before it."""
    at = [text.index(p) for p in pieces]
    return at == sorted(at)


def _js(text: str) -> str:
    """text as a single-quoted JS string, its quotes escaped as the views do."""
    return "'" + text.replace("'", "\\'") + "'"


def _const(script: str, name: str) -> str:
    """The value of a one-line single-quoted JS constant (``name = '...';``)."""
    found = re.search(re.escape(name) + r" = '((?:[^'\\]|\\.)*)';\n", script)
    assert found, name
    return found.group(1).replace("\\'", "'")


# --------------------------------------------------------------------------- B4


def test_b4_the_plan_card_names_getting_the_result_at_every_one_answered() -> None:
    """B4: the constant, marked r5i B4, right after OPEN_FIRST; in refresh()
    the lead's p.next (the "next" paragraph OE.lead draws, #root's own
    child) keeps the server's words on itself (dataset.lead) and reads
    PLAN_ALL_ANSWERED while the read is complete, the server's words at any
    other count (a failed read is not complete); only while the sheet is not
    open here, and only after the checks that drop an older read and a read
    of a card no longer on the page (a new tool result drew another in its
    place: the new card's line is not this read's). The sheet's own read on
    the open press drops its answer the same way."""
    page = _read("result.html")
    assert "// WORDING: r4p2\nconst OPEN_FIRST = 'Open the sheet here';\n" in page
    assert (
        f"// WORDING: r5i B4\nconst PLAN_ALL_ANSWERED = {_js(PLAN_ALL_ANSWERED)};\n"
        in page
    )
    assert _in_order(
        page,
        "const OPEN_FIRST = ",
        "const PLAN_ALL_ANSWERED = ",
        "function sheetButton(",
    )
    button = _between(page, "function sheetButton(app, data, figureSlot) {", "\n}\n")
    refresh = _between(button, "  async function refresh() {", "\n  }\n")
    assert _in_order(
        refresh,
        "    if (mine !== reads) return;\n",
        "    if (!slot.isConnected) return;  // a card drawn in this one's place",
        "    tellCard(got, words);\n",
        "    const complete = rows > 0 && labelled >= rows;\n",
        "    const nextLine = root.querySelector(':scope > .oe-lead > p.next');\n",
        "    if (nextLine && !opened) {\n",
        "      nextLine.dataset.lead = nextLine.dataset.lead || "
        "nextLine.textContent;\n",
        "      nextLine.textContent = complete ? PLAN_ALL_ANSWERED : "
        "nextLine.dataset.lead;\n",
        "    const allowed = Boolean(got) && (got.blind === false || complete);\n",
        "      if (mine !== reads || !slot.isConnected) return;\n",
        "    showFigure(allowed && Boolean(svg), why);",
    )
    assert _code(refresh).count("slot.isConnected") == 2
    opener = _between(button, "    onclick: async () => {", "\n    },\n")
    assert _in_order(
        opener,
        "const sheet = await OE.call(app, 'olmoearth_app_sheet', "
        "{ labels_csv_path: data.labels_csv_path });\n",
        "        if (!slot.isConnected) return;\n",
        "        const editor = OE.sheetEditor(app, sheet, {",
    )
    # A failed read is not complete: its rows are NaN.
    assert (
        "const rows = got ? Number(got.n_rows) : NaN, labelled = got ? "
        "Number(got.n_labelled) : NaN;" in refresh
    )
    # Defined once, used once; the open still takes the line off (r5-f F02).
    code = _code(page)
    assert code.count("PLAN_ALL_ANSWERED") == 2
    assert code.count(PLAN_ALL_ANSWERED) == 1
    assert "if (answerEvery) answerEvery.remove();" in button
    # Plain words, no unit, so it holds for squares, points and a certificate.
    assert not _OFF_FACE.search(PLAN_ALL_ANSWERED)
    assert not re.search(r"\b(?:squares?|points?)\b", PLAN_ALL_ANSWERED)
    assert PLAN_ALL_ANSWERED != PLAN_NEXT and len(PLAN_ALL_ANSWERED.split(". ")) == 2
    # Rule R4: no button's words bare, outside double quotes (as the harness's
    # lead_check reads it); the one quoted label is the button's at a whole
    # sheet, the server's sheet.OPEN_DONE.
    bare = re.sub(r'"[^"]*"', " ", PLAN_ALL_ANSWERED)
    for label in (sheet.OPEN_FIRST, sheet.OPEN_AGAIN, sheet.OPEN_DONE):
        assert label not in bare, label
    assert re.findall(r'"([^"]*)"', PLAN_ALL_ANSWERED) == [sheet.OPEN_DONE]
    assert sheet.progress_words(3, 3, 0)[1] == sheet.OPEN_DONE
    assert sheet.progress_words(5, 5, 1)[1] == sheet.OPEN_DONE


# --------------------------------------------------------------------------- B5


def test_b5_a_declined_docs_or_access_link_says_so_in_its_own_slot() -> None:
    """B5: OE.docsLink and OE.accessLink ask OE.open to be quiet, and when the
    host does not open the link their slot (a status) says it, LINK_NOT_OPENED
    then DOCS_BY_HAND on one line, over the address; the line marked r5i B5.
    Their bodies stay the same but for the label and the address. The empty
    slot is not hidden, so the status is in place before the first press."""
    common = _read("common.js")
    assert _const(common, "OE.LINK_NOT_OPENED") == LINK_NOT_OPENED
    assert _const(common, "OE.DOCS_BY_HAND") == DOCS_BY_HAND
    assert LINK_NOT_OPENED + " " + DOCS_BY_HAND == LINK_DECLINED_LINE
    for head, url in (
        ("OE.docsLink = (app) => {", "OE.DOCS_URL"),
        ("OE.accessLink = (app) => {", "OE.ACCESS_URL"),
    ):
        link = _between(common, head, "\n};")
        assert _in_order(
            link,
            "const slot = OE.el('div', { class: 'oe-link-slot small', "
            "role: 'status' });",
            f"const opened = await OE.open(app, {url}, {{ quiet: true }});\n",
            "      // WORDING: r5i B5\n"
            "      slot.replaceChildren(...(opened ? [] : [OE.el('div', {}, "
            "OE.LINK_NOT_OPENED + ' ' + OE.DOCS_BY_HAND), "
            f"OE.el('code', {{}}, {url})]));",
            "return [button, slot];",
        ), head
        assert "OE.toast(" not in link, head
    docs = _between(common, "OE.docsLink = (app) => {", "\n};")
    access = _between(common, "OE.accessLink = (app) => {", "\n};")
    same = docs.replace("OE.docsLink", "OE.accessLink")
    same = same.replace("OE.DOCS_LABEL", "OE.ACCESS_LABEL")
    assert same.replace("OE.DOCS_URL", "OE.ACCESS_URL") == access
    # The empty slot is drawn (0 px tall), never display none: a live region
    # hidden until its first words arrive with them is often not announced,
    # so the status is kept in the accessibility tree before its first words.
    assert ".oe-link-slot:empty" not in common
    assert ".oe-link-slot code { user-select: all; overflow-wrap: anywhere; }" in common


def test_b5_only_those_two_links_are_quiet_and_every_other_keeps_its_toast() -> None:
    """B5: OE.open takes quiet (false unless given); a decline writes
    LINK_NOT_OPENED in the status line unless quiet, both ways it can fail;
    a link opened still clears that line. Only the docs and Request access
    links pass quiet: the sign-in page's open, the result's and the sheet's
    street map links (no slot of their own) call OE.open as before."""
    common = _read("common.js")
    head = "OE.open = async (app, url, { quiet = false } = {}) => {"
    assert common.count(head) == 1
    opener = _between(common, head, "\n};")
    assert opener.count("if (!quiet) OE.toast(OE.LINK_NOT_OPENED, true);") == 2
    assert opener.count("OE.toast(") == 3
    assert "if (bar && bar.textContent === OE.LINK_NOT_OPENED) OE.toast('');" in opener
    assert opener.count("return false;") == 2 and "return true;" in opener
    calls = {
        name: re.findall(r"OE\.open\(app, [^)]*\)", _code(_read(name)))
        for name in ("common.js", "result.html", "sheet.js", "start.html")
    }
    assert calls["common.js"] == [
        "OE.open(app, OE.DOCS_URL, { quiet: true })",
        "OE.open(app, OE.ACCESS_URL, { quiet: true })",
        "OE.open(app, page.url)",
    ]
    assert calls["result.html"] == ["OE.open(app, url)"]
    assert calls["sheet.js"] == ["OE.open(app, url)"]
    assert calls["start.html"] == []
    for name in views.VIEWS:
        html = views.view_html(name)
        assert html.count("{ quiet: true })") <= 2, name
        assert "OE.open(app, url, { quiet" not in html, name


# --------------------------------------------------------------------------- B6


def test_b6_the_closed_outputs_toggle_says_show_outputs() -> None:
    """B6: the label marked r5i B6 under the r4-p2 tag of the words; the
    toggle's aria-label is built from the label ("Show outputs of <run>"),
    closed and open; no toggle text "Outputs" is left in the code; the
    toggle is 100 px wide (it was 92 px, for "Hide outputs"), so the job
    buttons after it keep one left edge."""
    page = _read("studio.html")
    words = _between(page, "const OUTPUTS = {", "\n};")
    assert (
        "// WORDING: r4p2 (label, none, noneYet); r5c S05 (first)\n"
        "const OUTPUTS = {\n"
        "  // WORDING: r5i B6\n"
        f"  label: (shown) => (shown ? '{HIDE_OUTPUTS}' : '{SHOW_OUTPUTS}'),\n"
    ) in page
    assert "'Outputs'" not in _code(page) and "'Outputs'" not in words
    row = _between(page, "function predictionRow(", "\n}\n")
    assert _in_order(
        row,
        "const label = OUTPUTS.label;",
        "'aria-label': label(false) + ' of ' + p.name },",
        "toggle.textContent = label(shown);",
        "toggle.setAttribute('aria-label', label(shown) + ' of ' + p.name);",
    )
    rule = ".items .pred-acts > button.link { min-width: 100px; text-align: left; }"
    assert page.count(rule) == 1
    assert "min-width: 92px" not in page
    assert views.view_html("studio").count(f"'{SHOW_OUTPUTS}'") == 1


# --------------------------------------------------------------------------- B7


def test_b7_the_toast_after_a_removal_whose_read_failed_says_what_went() -> None:
    """B7: in clearAndOpen, a removal the server took whose sheet then cannot
    be read marks the copy stale with 'read' (the footer's W.notReadAgain,
    beside Reopen the sheet) and toasts only which answers went, marked r5i
    B7, as a plain message (reopenToast's isError false: a removal is no
    error; every other reopenToast stays an error); the read's error and
    "could not be read again" are on no line of that path; the footer's
    words are unchanged. The footer's line is the description of Reopen the
    sheet (its own id), and the stale footer hands that button the focus of
    the notice's Remove button, which it drops, as it hands a Save button's."""
    script = _read("sheet.js")
    clear = _between(script, "  async function clearAndOpen(column, to) {", "\n  }\n")
    caught = _between(
        clear, "        } catch (_) {\n", "\n        }\n      }\n    } finally {"
    )
    assert _in_order(
        caught,
        "          markStale('read');\n",
        "          // WORDING: r5i B7\n"
        "          reopenToast('The ' + answersOf(column) + ' were removed.', false);",
    )
    helper = _between(
        script, "  function reopenToast(text, isError = true) {", "\n  }\n"
    )
    assert helper.endswith("    state.reopenSaid = text;\n    OE.toast(text, isError);")
    assert len(re.findall(r"reopenToast\([^;]*, false\);", _code(script))) == 1
    # The footer's line describes Reopen the sheet; the focus goes there.
    assert (
        "  const staleLine = OE.el('span', { class: 'error', id: 'oe-stale-' + "
        "Math.random().toString(36).slice(2) });\n"
        "  reopenButton.setAttribute('aria-describedby', staleLine.id);\n"
    ) in script
    progress = _between(script, "  function updateProgress() {", "\n  }\n")
    stale = _between(progress, "    if (state.stale) {", "      return;\n    }")
    assert _in_order(
        stale,
        "      const hadFocus = (!saveButton.hidden && "
        "saveButton.contains(document.activeElement))\n"
        "        || notice.contains(document.activeElement);\n",
        "      notice.replaceChildren();\n",
        "      if (hadFocus && !quietSave) OE.focusIfHere(reopenButton);\n",
    )
    for gone in ("err", "bare(", "could not be read again", "Reopen it to go on"):
        assert gone not in _code(caught), gone
    assert "could not be read again" not in _code(clear)
    assert "Reopen it to go on" not in _code(clear)
    # The two sets' names give the two toasts.
    assert (
        "const answersOf = (c) => (c === 'wrong' ? 'right-or-wrong marks' : "
        "'classes picked');" in script
    )
    for kind, toast in (
        ("right-or-wrong marks", REMOVED_WRONG),
        ("classes picked", REMOVED_CLASSES),
    ):
        assert "The " + kind + " were removed." == toast
    assert f"    notReadAgain: {_js(NOT_READ_AGAIN)},\n" in script


# --------------------------------------------------------------------------- B8


def test_b8_sign_in_to_start_a_job_is_at_body_size_in_the_body_colour() -> None:
    """B8: signed out, the line over the job tiles has no class (body size,
    body colour; it was 'small faint'); its words and its place (the jobs
    card's first block, over the list) are unchanged; signed in, no line."""
    page = _read("start.html")
    render = _between(page, "function render(app, data) {", "\n}\n")
    assert _in_order(
        render,
        "root.append(OE.el('section', { class: 'card stack' },",
        f"on ? null : OE.el('div', {{}}, {_js(SIGN_IN_FIRST)}),",
        "OE.el('ul', { class: 'items jobs', 'aria-label': 'Jobs' }, "
        "jobRows(app, jobs, on))));",
    )
    assert render.count(SIGN_IN_FIRST) == 1
    assert f"{{ class: 'small faint' }}, {_js(SIGN_IN_FIRST)}" not in page
    assert not _OFF_FACE.search(SIGN_IN_FIRST)
