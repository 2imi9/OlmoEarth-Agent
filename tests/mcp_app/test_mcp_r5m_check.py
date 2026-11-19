# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Round r5-m (19 Nov 2026), track CHECK, as specs/r5-m-round.md gives it:
the Check card's items of the build list of the audit after r5-l
(specs/r5-l-audit/build_list.md).

- B1 (the Check card's part): the two captions under the example pictures
  and step 3 say "what the map is meant to find", the words of the app's
  general rule (``imagery.READING_RULE``); they said "what the map shows",
  which readers took for the map's own answer at the square, the one thing
  the blind sheet hides. "Present: what the map is meant to find covers more
  than half of the square on a clear date.", "Absent: what the map is meant
  to find covers half of the square or less on every clear date." and "3.
  Someone who can recognise what the map is meant to find looks at each one
  in a satellite picture.", each marked ``// WORDING: r5m B1``. No kelp, no
  number changes.
- B3 (the K5 regression of 13c): the line of a swap that could not save
  first (NOT_SWAPPED: "... press Save labels to try again, then open the
  other sheet.") is, after review, the open sheet's own, said once in its
  footer under Save labels (the editor's ``oeSay``), after the save's reason
  and in place of "Press Save labels to try again.", and goes once a save is
  taken; a toast of those words (an editor with no such hook) goes once a
  sheet opens. On the card of parts the page's one status line (#oe-toast)
  sits right under the press or the sheet open here, before "How this check
  works" and Details (r5-k K5's rule: a status sits under what caused it);
  it was at the page's foot under both folds. After review: show() puts it
  back under #root, before Settings (r5-n S1), before it draws another card;
  it no longer says
  the line under the press again (a press held by the rule or the box toasts
  nothing), and a press with no reply, which leaves no press and no sheet,
  is said in the page's hidden status; a card that show() replaced places
  no fold. No words change.
- B7: a chosen number of squares carries a tick before it, the same tick a
  chosen answer on the label sheet carries, drawn by the style with empty
  alt text, so each button's name stays its number. The rule sets the tick
  alone, so a count keeps its size (R1); after review, a count's line never
  breaks (nowrap), so a four-digit count with its tick stays one line in the
  narrowest columns.
- After review, the sheet track's B5 on this card: with every square
  answered and a drawn sheet that already gave its result, the heading names
  the button such a sheet has ("Show the result again"), or none where only
  some did or one changed since; the card reads that from the sheet's counts
  (``check._sheet_words``: ``graded``, ``graded_changed``).

The views are read as text (no test here runs the scripts). The browser
harness (scripts/mcp_views_e2e.py) draws them: the captions in
12j_check_example_pictures and 12t_check_examples_with_box, step 3 in
12_check_card and the cards that read the steps, the swap's line in
13c_check_swap_saves_first (in the sheet's footer, the status line empty;
gone after the press of Save labels) and 12_check_card (the status line
after the sheet, before the fold), the held press in 12e_check_no_rule_gate
and 12g_check_owner_rule_box, the press with no reply in
14d_check_press_no_answer and 14f_check_press_lost_rule (which also draws a
card that could not start after the Check card), the tick in 12_check_card
(the widest count at 780, 520, 490, 481 and 380 px), 12c_check_narrow_counts
and 12u_check_wide_counts_parts (four digits on a map in two parts), and
the graded heading in 13g_check_graded_sheet and 13h_check_parts_one_graded.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

from olmoearth_agent.mcp_app import check, imagery, views

#: B1: the words the captions and step 3 share with the app's rule.
MEANT = "what the map is meant to find"
#: B1: the words they had in round r5-l, read as the map's own answer.
OLD = "what the map shows"

#: B1: the two captions under the example pictures.
EXAMPLE_PRESENT = (
    "Present: what the map is meant to find covers more than half of the square "
    "on a clear date."
)
EXAMPLE_ABSENT = (
    "Absent: what the map is meant to find covers half of the square or less on "
    "every clear date."
)
#: B1: step 3 under the Check card's title.
STEP_3 = (
    "3. Someone who can recognise what the map is meant to find looks at each one "
    "in a satellite picture."
)

#: B3: the line of a swap that could not save first, as check.html declares
#: it, word for word as it was before the round.
NOT_SWAPPED_MADE = (
    "const NOT_SWAPPED = 'The sheet open here has a change that is not saved, so "
    "it stays open: press Save labels to try'\n"
    "  + ' again, then open the other sheet.';\n"
)
#: B3, after review: the sheet editor's hook that says it in its footer.
OE_SAY = (
    "  wrap.oeSay = (text) => {\n"
    "    const tail = tryAgain();\n"
    "    const was = saveSaid.classList.contains('error')"
    " ? saveSaid.textContent : '';\n"
    "    footerSay(saveSaid, was.endsWith(tail) ? was.slice(0, -tail.length) + text"
    " : text, true);\n"
    "  };\n"
)
#: B3, after review: a press held by the rule or the box.
HELD = "    if (gateHeld()) { updatePress(); OE.focusIfHere(ruleInput); return; }\n"
#: B3, after review: show() puts the page's one status line back under
#: #root, before Settings (r5-n S1).
BACK = (
    "  const bar = OE.toastRegion();\n"
    "  if (root.contains(bar)) root.after(bar);\n"
    "  root.replaceChildren(OE.el('div', { class: 'overline' }, 'Accuracy check'));\n"
)

#: B7: the tick's content, as check.html's style writes it (CSS escapes).
TICK_CONTENT = 'content: "\\2713\\00a0"; content: "\\2713\\00a0" / "";'
#: B7: the rule that draws it before a chosen count.
TICK = (
    '  .pick-group button.pick[aria-pressed="true"] .pick-count::before'
    ' { content: "\\2713\\00a0"; content: "\\2713\\00a0" / ""; }\n'
)
#: B7: the chosen count's look before this round, which the tick follows.
CHOSEN = (
    '  .pick-group button.pick[aria-pressed="true"] { background: var(--selected);'
    " border-color: var(--text);\n"
    "    box-shadow: inset 0 0 0 1px var(--text); }\n"
)
#: B7, after review: a count's line never breaks.
COUNT_LINE = (
    "  .pick-count { font-size: 14px; line-height: 21px; white-space: nowrap; }\n"
)

#: After review (B5 on this card): the graded sheet's finish button.
GRADED_FINISH = "Show the result again"
#: After review: the heading's four graded forms, as ALL_GRADED makes them.
ALL_GRADED_ONE = (
    "Every square is answered. Open the sheet, then press Show the result again."
)
ALL_GRADED_PARTS = (
    "Every square is answered. In each part's sheet, press Show the result again: "
    "one result per part."
)
ALL_GRADED_SOME = (
    "Every square is answered. Open each part's sheet to get its result: one "
    "result per part."
)
ALL_GRADED_ONE_UNNAMED = "Every square is answered. Open the sheet to get its result."
#: After review: ALL_GRADED as check.html writes it.
ALL_GRADED_MADE = (
    "// WORDING: r5m (after review: the sheet's B5 words, sheet.js W.gradedFinish)\n"
    "const GRADED_FINISH = 'Show the result again';\n"
    "// WORDING: r5m (after review: a graded sheet's heading)\n"
    "const ALL_GRADED = (several, named) => 'Every square is answered. '\n"
    "  + (named ? (several ? 'In each part\\'s sheet, press ' + GRADED_FINISH"
    " + ': one result per part.'\n"
    "    : 'Open the sheet, then press ' + GRADED_FINISH + '.')\n"
    "    : several ? 'Open each part\\'s sheet to get its result: one result per"
    " part.'\n"
    "      : 'Open the sheet to get its result.');\n"
)

#: Words no face holds (round r5-e's rule R7, and the owner's ruling 6).
_OFF_FACE = re.compile(
    r"\b(?:windows?|UTM|zones?|intervals?|strat(?:um|a)|design|budget"
    r"|recommended)\b",
    re.I,
)

#: A word that names the one kind of map the app's rule was first set for.
_KELP = re.compile(r"kelp|canopy", re.I)


def _read(name: str) -> str:
    """A file of views/, as it is on disk."""
    return (views.VIEW_DIR / name).read_text(encoding="utf-8")


def _between(text: str, start: str, end: str) -> str:
    """The part of text from start up to the first end after it."""
    at = text.index(start)
    return text[at : text.index(end, at)]


def _code(page: str) -> str:
    """The page with its comments taken out."""
    return re.sub(r"/\*.*?\*/|//[^\n]*", "", page, flags=re.S)


def _in_order(text: str, *pieces: str) -> bool:
    """Whether each piece is in text, each after the one before it."""
    at = [text.index(p) for p in pieces]
    return at == sorted(at)


def _check_card(page: str) -> str:
    """check.html's checkCard, whole."""
    return _between(page, "function checkCard(", "\n}\n")


def _open_zone(page: str) -> str:
    """check.html's openZone, whole but its closing brace."""
    return _between(
        page, "  async function openZone(m, { hadFocus = null } = {}) {", "\n  }\n"
    )


def _draw_picked(page: str) -> str:
    """check.html's drawPicked, whole but its closing brace."""
    return _between(page, "  async function drawPicked() {", "\n  }\n")


# --------------------------------------------------------------------------- B1


def test_b1_the_captions_say_what_the_map_is_meant_to_find() -> None:
    """B1: the two captions under the example pictures are the spec's words,
    each marked ``// WORDING: r5m B1`` above the tags it had (r5-l L2, r4-k
    K9); they name no map and no word a face does not say, and they are
    still the pictures' captions and alt text."""
    page = _read("check.html")
    for name, said in (
        ("EXAMPLE_PRESENT", EXAMPLE_PRESENT),
        ("EXAMPLE_ABSENT", EXAMPLE_ABSENT),
    ):
        tags = (
            "// WORDING: r5m B1\n// WORDING: r5l L2\n// WORDING: r4k K9\n"
            f"const {name} = '{said}';\n"
        )
        assert tags in page, name
        assert MEANT in said and OLD not in said, name
        assert _KELP.search(said) is None, name
        assert _OFF_FACE.search(said) is None, name
    assert EXAMPLE_PRESENT.startswith("Present: ")
    assert EXAMPLE_ABSENT.startswith("Absent: ")
    block = _between(page, "function examplesBlock(ex) {", "\n}\n")
    assert "one(ex.present, EXAMPLE_PRESENT), one(ex.absent, EXAMPLE_ABSENT)" in block
    assert "OE.el('img', { src, alt: caption })," in block


def test_b1_step_3_says_what_the_map_is_meant_to_find() -> None:
    """B1: step 3 reads "3. Someone who can recognise what the map is meant
    to find looks at each one in a satellite picture.", marked
    ``// WORDING: r5m B1 (step 3)`` above its old tags; the other steps are
    as they were, and the card of parts and the preparing card draw the four
    from the one list."""
    page = _read("check.html")
    marks = (
        "// WORDING: r5m B1 (step 3)\n"
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
    assert MEANT in STEP_3 and OLD not in STEP_3
    assert _KELP.search(STEP_3) is None and _OFF_FACE.search(STEP_3) is None
    assert "fig('steps', [firstStep(models), ...STEPS.slice(1)])" in page
    assert "fig('steps', [...STEPS])" in page


def test_b1_the_cards_words_follow_the_rule_and_are_served() -> None:
    """B1: the captions and step 3 say what the app's general rule says
    (``imagery.READING_RULE``: what the map is meant to find); the card's
    code no longer says "what the map shows" (comments may quote it); the
    marks of this round on the card are B1's three and, after review, the
    graded heading's two (B3 and B7 change no words); the check view as the
    server serves it holds the new words."""
    assert MEANT in imagery.READING_RULE
    page = _read("check.html")
    assert OLD not in _code(page)
    marks = re.findall(r"// WORDING: r5m[^\n]*", page)
    assert marks == [
        "// WORDING: r5m B1 (step 3)",
        "// WORDING: r5m (after review: the sheet's B5 words, sheet.js W.gradedFinish)",
        "// WORDING: r5m (after review: a graded sheet's heading)",
        "// WORDING: r5m B1",
        "// WORDING: r5m B1",
    ]
    served = views.view_html("check")
    for said in (EXAMPLE_PRESENT, EXAMPLE_ABSENT, STEP_3):
        assert f"'{said}'" in served, said


# --------------------------------------------------------------------------- B3


def test_b3_a_sheet_that_opens_clears_the_not_swapped_toast() -> None:
    """B3: once a sheet opens (its read succeeded), a toast that said the
    sheet open before stays open goes (an editor with no ``oeSay`` toasts
    it), as the part's own line on answers not saved does (r5-d R05), each
    only while the toast still says its words; a read that fails opens
    nothing and keeps it. The clear comes before the editor swap, which may
    write a toast of its own for the part swapped away."""
    page = _read("check.html")
    swap = _open_zone(page)
    clear = (
        "    const bar = document.getElementById('oe-toast');\n"
        "    if (bar && ((m.line && bar.textContent === m.line)"
        " || bar.textContent === NOT_SWAPPED)) OE.toast('');\n"
        "    m.line = null;\n"
    )
    assert swap.count(clear) == 1
    assert _in_order(
        swap,
        "      else OE.toast(NOT_SWAPPED, true);\n      return;\n    }\n",
        "    } catch (err) {\n",
        "      if (had) OE.focusIfHere(m.openButton);\n      return;\n    }\n",
        clear,
        "    const editor = OE.sheetEditor(app, sheet, {\n",
        "      OE.toast(before.line, true);\n",
    )
    caught = _between(swap, "    } catch (err) {\n", "      return;\n    }\n")
    assert "OE.toast(" not in caught


def test_b3_the_swaps_line_is_the_sheets_own() -> None:
    """B3, after review: a swap that could not save first says NOT_SWAPPED
    through the open sheet's ``oeSay``, once, in its footer's line under
    Save labels (an alert): after the failed save's reason, in place of
    tryAgain's "Press Save labels to try again.", else alone; the status
    line, which the toast took over before, is emptied; a toast only for an
    editor with no such hook. The line goes as the footer's save line does,
    once a save is taken (counterSay empties it), so the card's onSaved is
    back to reading the counts again, its one onSaved."""
    page = _read("check.html")
    swap = _open_zone(page)
    settle = _between(swap, "!(await current.oeSettle())) {\n", "      return;\n")
    assert _in_order(
        settle,
        "      m.opening = false;\n",
        "      if (typeof current.oeSay === 'function')"
        " { OE.toast(''); current.oeSay(NOT_SWAPPED); }\n",
        "      else OE.toast(NOT_SWAPPED, true);\n",
    )
    saved = (
        "    const editor = OE.sheetEditor(app, sheet, {\n"
        "      headerAction: OE.fullscreenButton(app, OE.sheetMode),"
        " onSaved: () => refreshZone(m),\n"
        "    });\n"
    )
    assert saved in swap
    assert _code(page).count("onSaved") == 1
    editor = _read("sheet.js")
    assert editor.count(OE_SAY) == 1
    assert _in_order(
        editor,
        "  wrap.oeSettle = async () => {\n",
        OE_SAY,
        "  wrap.oeFlush = async () => {\n",
    )
    again = _between(editor, "  function tryAgain() {", "\n  }\n")
    connected = "    if (wrap.isConnected) return 'Press Save labels to try again.';\n"
    assert connected in again
    said = _between(editor, "  function counterSay(text) {", "\n  }\n")
    assert "    saveSaid.textContent = '';\n" in said
    assert _code(editor).count("oeSay") == 1


def test_b3_the_toasts_words_are_unchanged() -> None:
    """B3: no words change: NOT_SWAPPED is as it was, said in one place (the
    swap that could not save first: in the sheet's footer, or as a toast for
    an editor with no ``oeSay``) and cleared from the status line once a
    sheet opens."""
    page = _read("check.html")
    assert NOT_SWAPPED_MADE in page
    code = _code(page)
    assert len(re.findall(r"\bNOT_SWAPPED\b", code)) == 4
    assert code.count("OE.toast(NOT_SWAPPED, true);") == 1
    assert code.count("current.oeSay(NOT_SWAPPED);") == 1
    assert "WORDING: r5m B3" not in page


def test_b3_the_status_line_sits_under_the_press_or_the_sheet() -> None:
    """B3 (r5-k K5's rule on this card): the card of parts puts the page's
    one status line right after its action (the press, or the sheet open
    here), so before Details, and the fold "How this check works" goes after
    that line: parts, press or sheet, status line, the fold, Details. The
    line is the node OE.toastRegion keeps (the one #oe-toast, made in the
    body when none is in the page), so moving it keeps one. After review:
    show() puts that line back under #root, before Settings (r5-n S1; made at
    the page's foot if missing) before it draws #root again, so every card
    starts with the region where OE.start put it, and checkCard moves the same
    node again."""
    page = _read("check.html")
    card = _check_card(page)
    assert _in_order(
        card,
        "\n  root.append(action, OE.fold(DETAILS, foldBody));\n",
        "\n  action.after(OE.toastRegion());\n",
        "\n  const how = OE.el('details', ",
        "\n  function foldHow() {\n",
        "    how.append(...moved);\n    OE.toastRegion().after(how);\n  }\n",
        "\n  renderAll();\n",
    )
    assert card.count("action.after(how)") == 0
    assert _code(page).count("OE.toastRegion()") == 3
    assert _code(card).count("OE.toastRegion()") == 2
    shown = _between(page, "function show(app, params) {", "\n}\n")
    assert shown.count(BACK) == 1
    assert _in_order(
        shown, "  OE.disposeEditors(root);\n", BACK, "  if (data.needs_sign_in) {\n"
    )
    common = _read("common.js")
    region = _between(common, "OE.toastRegion = () => {", "\n};")
    assert "  let bar = document.getElementById('oe-toast');\n" in region
    assert "  if (!bar) {\n" in region
    assert "    document.body.append(bar);\n" in region
    assert region.endswith("\n  return bar;")


def test_b3_a_card_that_was_replaced_places_no_fold() -> None:
    """B3, after review: foldHow places the fold after the status line, found
    by its id, so a card that show() replaced (its press's reply came late
    and ran renderAll) would put its fold in the new card: it does nothing
    once its action is off the page, nor takes the new card's strips card."""
    page = _read("check.html")
    fold = _between(page, "  function foldHow() {", "\n  }\n")
    guard = (
        "    if (how.isConnected || !action.isConnected "
        "|| !models.some((m) => m.kind === 'drawn')) return;\n"
    )
    assert _in_order(
        fold,
        guard,
        "    const moved = [steps, root.querySelector(':scope > .card.oe-strips')]",
        "    OE.toastRegion().after(how);",
    )


def test_b3_a_press_held_by_the_rule_toasts_nothing() -> None:
    """B3, after review (said once): the line under the press says what the
    rule or the box needs (pressWhy: RULE_FIRST, BOX_SHORT), and the status
    line sits right under it since B3, so a press held by the rule draws
    that line again and moves the focus to the box, with no toast of the same
    words (it stayed, too, once the rule was fixed)."""
    page = _read("check.html")
    press = _draw_picked(page)
    assert press.count(HELD) == 1
    assert _in_order(
        press,
        "    if (plan.over > 0) { updatePress(); return; }",
        HELD,
        "    const hadFocus = document.activeElement === press;\n",
    )
    for toast in ("OE.toast(gateMode", "OE.toast(RULE_FIRST", "OE.toast(BOX_SHORT"):
        assert toast not in _code(page), toast
    why = _between(page, "  function pressWhy(plan) {", "\n  }\n")
    assert "if (gateMode === 'replace' && gateHeld()) return BOX_SHORT;" in why
    assert "if (gateHeld()) return RULE_FIRST;" in why


def test_b3_a_press_with_no_reply_is_heard_not_toasted() -> None:
    """B3, after review: a press with no reply leaves no press and no sheet,
    so a status line under the empty action hung under the last part's card
    and read as that part's line; the part's line and the gate's say what
    happened on the face. The press empties the status line ("Picking the
    squares..." goes) and says PRESS_NO_ANSWER in the page's hidden status
    (#oe-check-said, which show empties), a status a screen reader hears."""
    page = _read("check.html")
    press = _draw_picked(page)
    lost = (
        "        OE.toast('');\n"
        "        if (heard) heard.textContent = PRESS_NO_ANSWER;\n"
        "      }\n      renderAll();\n"
    )
    assert press.count(lost) == 1
    assert _in_order(press, "ruleLost = true;", lost)
    assert "OE.toast(PRESS_NO_ANSWER" not in page
    assert "const PRESS_NO_ANSWER = 'The press got no answer back.';\n" in page
    assert "const heard = document.getElementById('oe-check-said');\n" in page
    assert (
        '<div id="oe-check-said" class="oe-said-hidden" role="status" '
        'aria-live="polite"></div>' in page
    )


# --------------------------------------------------------------------------- B7


def test_b7_a_chosen_count_carries_the_answers_tick() -> None:
    """B7: a chosen count (aria-pressed true) draws a tick and a no-break
    space before its number, inside its count's line, right after the chosen
    look (the pale fill and the dark line, unchanged); the tick's alt text is
    empty, so the button's name stays its number. It is the very tick a
    chosen answer of the label sheet carries (common.js, whose JS string
    doubles each backslash)."""
    page = _read("check.html")
    style = _between(page, "<style>", "</style>")
    assert style.count(TICK) == 1
    assert _in_order(style, CHOSEN, TICK, COUNT_LINE)
    assert TICK.endswith(" { " + TICK_CONTENT + " }\n")
    common = _read("common.js")
    doubled = TICK_CONTENT.replace("\\", "\\\\")
    answers = '.face .answers button[aria-pressed="true"]::before { ' + doubled
    assert answers in common
    # The button's own text is its number: the card's code writes no tick.
    assert "✓" not in page and "\\u2713" not in page
    assert (
        "OE.el('span', { class: 'pick-count' }, OE.el('strong', {}, "
        "Number(o.labels).toLocaleString('en-US')),\n" in page
    )


def test_b7_the_tick_does_not_widen_a_count() -> None:
    """B7 (R1: same kind, same size): the tick's rule sets its content alone
    (no width, padding, margin, border or font of its own), and the counts'
    least widths are as they were: a column of 132 px at least, a count's own
    least width, and each count a row of its own under 480 px. After review:
    a count's line never breaks (nowrap), so a chosen count of four digits
    stays one line in the narrowest columns, and the style's comment says
    what the browser check measures."""
    page = _read("check.html")
    style = _between(page, "<style>", "</style>")
    body = _between(style, ".pick-count::before {", "}")
    names = re.findall(r"([a-z-]+)\s*:", body.split("{", 1)[1])
    assert names == ["content", "content"]
    for rule in (
        "  .pick-group { --same-min: 132px; max-width: 100%; }\n",
        "    min-width: 132px; padding: 8px 12px; text-align: left;"
        " font-weight: 400; }\n",
        "    .pick-group button.pick { flex: 1 1 100%; min-width: 0;"
        " flex-direction: row; align-items: baseline;\n",
        "  @media (max-width: 480px) { .pick-group { --same-min: 100%; } }\n",
        COUNT_LINE,
    ):
        assert rule in style, rule
    assert style.count("white-space: nowrap") == 2  # the count's, the hidden status's
    note = _between(style, "  /* r5-m B7:", "*/")
    flat = " ".join(note.split())
    assert "four-digit" in flat and "481" in flat
    assert "a count of three digits with its tick fits" not in flat


def test_b7_the_served_check_view_holds_the_tick() -> None:
    """B7: the check view as the server serves it (views.view_html, which
    fills in placeholders only) holds the tick's rule with its CSS escapes
    as written, and the count's line that never breaks."""
    served = views.view_html("check")
    assert TICK in served
    assert COUNT_LINE in served


# --------------------------------------------------------------------------- B5 here


def test_the_heading_names_the_button_a_graded_sheet_has() -> None:
    """After review (the sheet track's B5: a sheet that already gave its
    result labels its finish button "Show the result again"): with every
    square answered and no sheet open, a drawn sheet that gave its result
    makes the heading name that button when every drawn sheet gave one and
    none changed since, and no button by its words when only some did or
    one changed since (its button reads otherwise); a map with no graded
    sheet keeps ALL_ANSWERED. The words are the sheet's own."""
    page = _read("check.html")
    assert ALL_GRADED_MADE in page
    editor = _read("sheet.js")
    assert f"    gradedFinish: '{GRADED_FINISH}',\n" in editor
    for said in (
        ALL_GRADED_ONE,
        ALL_GRADED_PARTS,
        ALL_GRADED_SOME,
        ALL_GRADED_ONE_UNNAMED,
    ):
        assert said.startswith("Every square is answered. "), said
        assert "Estimate" not in said and _OFF_FACE.search(said) is None, said
    for said in (ALL_GRADED_ONE, ALL_GRADED_PARTS):
        assert GRADED_FINISH in said, said
    for said in (ALL_GRADED_PARTS, ALL_GRADED_SOME):
        assert said.endswith(": one result per part."), said
    fill = _between(page, "  function fillAsk() {", "\n  }\n")
    assert _in_order(
        fill,
        "    ask.textContent = drawable().length ? ASK "
        ": done ? ALL_ANSWERED(drawn.length > 1) : YOUR_SQUARES;\n",
        "    const graded = drawn.filter((m) => m.draw && m.draw.graded === true);\n",
        "    const named = graded.length === drawn.length"
        " && !graded.some((m) => m.draw.graded_changed === true);\n",
        "    if (!drawable().length && done && graded.length)"
        " ask.textContent = ALL_GRADED(drawn.length > 1, named);\n",
        "    moreLine.hidden = ",
    )
    # Defined once, and the heading's one use.
    assert _code(page).count("ALL_GRADED") == 2
    refresh = _between(page, "  async function refreshZone(m) {", "\n  }\n")
    assert _in_order(
        refresh,
        "    if (Number.isInteger(got.n_labelled)) "
        "m.draw.n_labelled = got.n_labelled;\n",
        "    m.draw.graded = Boolean(got.graded && typeof got.graded === 'object');\n",
        "    m.draw.graded_changed = m.draw.graded"
        " && got.graded.changed_since_first === true;\n",
        "    fillAsk();\n",
    )


def _words_with(monkeypatch: pytest.MonkeyPatch, mark: Any) -> dict[str, Any]:
    """check._sheet_words of a sheet answered whole whose counts carry the
    graded mark given (sheet.sheet_progress's ``graded``)."""

    def progress(path: str) -> dict[str, Any]:
        """The sheet's counts, as sheet.sheet_progress gives them."""
        return {
            "progress": "20 of 20 labelled",
            "open_label": "Open the sheet",
            "n_rows": 20,
            "n_labelled": 20,
            "n_unjudged": 0,
            "graded": mark,
        }

    monkeypatch.setattr(check.sheet, "sheet_progress", progress)
    return check._sheet_words("/out/check_a_to_label.csv")


def test_a_drawn_sheets_words_say_whether_it_gave_its_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After review: a drawn zone's sheet words (the card's view data) say,
    beside its counts, whether its sheet gave its result (a graded mark,
    whatever it holds) and whether its answers changed since, two yes-or-no
    values, never the mark itself nor any grade."""
    first = {"first_at": "2026-11-13T10:04:00Z", "changed_since_first": False}
    changed = {**first, "changed_since_first": True}
    for mark, want in (
        (None, (False, False)),
        (first, (True, False)),
        (changed, (True, True)),
    ):
        words = _words_with(monkeypatch, mark)
        assert (words["graded"], words["graded_changed"]) == want, mark
        assert set(words) == {
            "progress",
            "open_label",
            "n_rows",
            "n_labelled",
            "graded",
            "graded_changed",
        }
        assert "2026-11-13" not in repr(words)


# --------------------------------------------------------------------------- the docs


def test_the_changelog_and_the_usage_say_this_tracks_items() -> None:
    """The 1.5.0 detail file's marker (spec/changes/1.5.0.md) of this track is
    replaced by one line per item
    (B1, B3, B7, and after review B5 on this card), found by their openings
    in the round's own list and by their words; spec/Usage.md gives step 3's
    new words, says where the status line goes and when, shows the chosen
    count's tick and gives the graded heading."""
    stage = Path(__file__).resolve().parents[2]
    changelog = (stage / "spec" / "changes" / "1.5.0.md").read_text(encoding="utf-8")
    assert "@@R5M-CHECK@@" not in changelog
    head = "The build list of the audit after r5-l (round 5, m, 19 Nov 2026):\n"
    section = changelog.split(head, 1)[1].split("\n- ", 1)[0]
    for line in (
        "\n    - B1: on the Check card the two captions under the example pictures"
        " say ",
        '\n    - B3: on a map in parts the line "The sheet open here has a change'
        " that is not saved",
        "\n    - B7: on the Check card a chosen number of squares carries a tick"
        ' before it ("✓ 20 squares")',
        "\n    - B5 on the Check card (after review): ",
    ):
        assert section.count(line) == 1, line
    for said in ('"' + STEP_3 + '"', '"' + ALL_GRADED_ONE + '"', "12u_check_"):
        assert said in section, said
    usage = " ".join((stage / "spec" / "Usage.md").read_text(encoding="utf-8").split())
    assert STEP_3[len("3. ") :] in usage
    # Where the status line goes and when: Usage.md carries no round ids, so
    # its sentence is pinned instead of the B3 marker.
    assert (
        "On a card of parts the status line sits right under the press or the "
        "sheet open in the card" in usage
    )
    assert "✓ 20 squares" in usage
    for said in (ALL_GRADED_ONE, ALL_GRADED_SOME, "graded_changed"):
        assert said in usage, said
    assert "(a press then says the same;" not in usage
    assert 'the toast "The press got no answer back."' not in usage
