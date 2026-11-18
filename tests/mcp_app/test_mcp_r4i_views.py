# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""The panels' part of the last fix round (r4-i), read from the view files (no
test here runs the scripts), each string as the round's string table gives it:

- F6: after a Check-card press that got no reply, the press is held and the
  gate shows one line, "Ask Claude to open the Check card again to see the
  rule.", until a reply or a reopened card gives the rule;
- F8: a review set's picture, with the tool's caption, is the first thing in
  the closed fold, and its one legend line follows it there (r4-j J4);
- O2: with Cannot tell answers the grid of checked squares is still drawn, a
  Cannot tell square empty and dashed, with the server's one line in place of
  the per-class lines;
- O3: the outlined button "Check each class the map draws" under the
  uncovered lines, its message (OE.JOB_TEXT.byClass, with r4-j J1's route
  that keeps to the part) and its Details line;
- F2: the rare classes' counted lines (uncovered_detail) in Details.

A JS concatenation of single-quoted strings and names is read back into its
text (:func:`_concat`), so a missing space between two joined pieces fails
here as it would on the card."""

from __future__ import annotations

import re

from olmoearth_agent.mcp_app import sheet, views

#: One token of a JS concatenation: a single-quoted string, a plus, or a name
#: (a call with no parentheses inside its own, such as unitOf(unit), is one
#: name).
_TOKEN = re.compile(
    r"\s*(?:'((?:[^'\\]|\\.)*)'|(\+)|([A-Za-z_][A-Za-z0-9_.]*(?:\([^()]*\))?))"
)

#: The Check card's gate line, and the toast of a press, while a press got no
#: reply (r4-i F6), in the words of r5-b J09: the card picks nothing more,
#: and the card opened again shows the rule and what was picked.
RULE_LOST = (
    "This card can pick no more squares. Ask Claude to open the Check card "
    "again: it shows the rule and what was picked."
)

#: The result card's second check (r4-i O3): the button, its Details line
#: ({many} the card's plural unit word) and the message its press sends.
CLASS_CHECK = "Check each class the map draws"
CLASS_CHECK_DETAIL = (
    "Check each class the map draws: a separate check with its own ranges. It "
    "picks {many} for each class the map draws, which need not be the classes you "
    "did not see. It is planned after this result was read, so it is its own "
    "check and is never pooled with this one."
)
#: The message on squares names the route that keeps to the part (r4-j J1):
#: the scores_path olmoearth_check_map gives for that part ({which}: "that
#: part", or "this map" with no part's file); on points, by result_id.
BY_CLASS_SQUARES = (
    "Check each class the map draws: a new check of Studio prediction result "
    "r1{part}, picked class by class (olmoearth_plan_label_sample on the "
    "scores_path that olmoearth_check_map gives for {which}). Its own check and "
    "ranges, never pooled with this one. First ask me how many squares to check."
)
BY_CLASS_POINTS = (
    "Check each class the map draws: a new check of Studio prediction result "
    "r1, picked class by class (olmoearth_plan_label_sample by result_id). Its "
    "own check and ranges, never pooled with this one. First ask me how many "
    "points to check."
)
#: A zone's file name as Studio gives it, and its words in the message.
PART_FILE = "result_epsg32610_0_native.tif"
IN_PART = " (the part in file result_epsg32610_0_native.tif)"

#: The ternaries of OE.JOB_TEXT.byClass, as written: the unit's (outer),
#: then, on squares, the part's file and the words for it.
UNIT_HEAD = "(r.unit === 'point'\n    ? "
UNIT_ELSE = "\n    : "
PART_CHOICE = "(r.part_file ? ' (the part in file ' + r.part_file + ')' : '')"
WHICH_CHOICE = "(r.part_file ? 'that part' : 'this map')"

#: The dashed square's CSS (r4-i O2); since r5-g G1 no legend swatch shares it.
CANNOT_CSS = (
    ".oe-sq.cannot { background: transparent; box-shadow: none;"
    " border: 1px dashed var(--text); box-sizing: border-box; }"
)

#: Words no face holds (the round's rule).
_OFF_FACE = re.compile(r"\b(?:windows?|budget|stratum|design|review set)\b", re.I)


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


def _entry(block: str, start: str) -> str:
    """One entry's value in a JS object literal: from start to the comma that
    ends its last line, or to the block's end."""
    at = block.index(start) + len(start)
    end = block.find(",\n", at)
    return (block[at:] if end < 0 else block[at:end]).strip().rstrip(",")


def _concat(expr: str, names: dict[str, str]) -> str:
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
            out.append(names[name])
        at = match.end()
    return "".join(out)


def _in_order(text: str, *pieces: str) -> bool:
    """Whether each piece is in text, each after the one before it."""
    at = [text.index(p) for p in pieces]
    return at == sorted(at)


def test_f6_a_press_with_no_reply_holds_the_press_and_says_one_line() -> None:
    """r4-i F6: while ruleLost the press is held (gateHeld is true first,
    before its two rule conditions, so the press is aria-disabled), the gate
    shows one line, RULE_LOST, and is not hidden (no rule and no box), and a
    press toasts RULE_LOST and returns before any call and without moving the
    focus; RULE_FIRST stays for the rule gate. A reply that carries the rule
    clears it, as before; the pictures' limit stays out of Details then.
    r5-b J09 rewords the line and leaves no part with numbers after such a
    press, so no press is left to hold: the guard stays first in the press.
    r5-b J08: a press with nothing chosen fills the line under the press
    (updatePress) and no longer toasts; a press held by 1 to 9 characters in
    the box beside the kelp rule toasts BOX_SHORT, not RULE_FIRST."""
    page = _read("check.html")
    assert "// WORDING: r5b J09\n// WORDING: r4i F6\nconst RULE_LOST = " in page
    assert _concat(_after(page, "const RULE_LOST = ", ";\n"), {}) == RULE_LOST
    held = _after(page, "  const gateHeld = () => ", ";\n")
    assert held.startswith("ruleLost || ")
    for text in (
        "(gateMode === 'ask' && ruleLength() < RULE_MIN)",
        "(gateMode === 'replace' && ruleLength() > 0 && ruleLength() < RULE_MIN)",
    ):
        assert text in held, text
    gate = _between(page, "  function fillGate() {", "\n  }\n")
    assert "const mode = ruleLost ? 'lost'\n" in gate
    assert "ruleLost ? 'none'" not in gate
    lost = (
        "    } else if (mode === 'lost') {\n"
        "      gate.replaceChildren(OE.el('div', {}, RULE_LOST));\n"
        "    } else {\n"
    )
    assert lost in gate
    assert "gate.hidden = mode === 'none';" in gate
    press = _between(page, "  async function drawPicked() {", "\n  }\n")
    line = "    if (gateHeld() && ruleLost) { OE.toast(RULE_LOST); return; }\n"
    assert line in press and "focus" not in line
    assert _in_order(
        press,
        "if (busy) return;",
        line,
        "if (!picked.length) { updatePress(); return; }",
        "if (gateHeld()) { OE.toast(gateMode === 'replace' ? BOX_SHORT : RULE_FIRST);"
        " OE.focusIfHere(ruleInput); return; }",
        "OE.call(app, 'olmoearth_app_check_draw'",
    )
    assert "ruleLost = false;\n      gateMode = '';" in press
    assert "ruleLost = true;" in press
    update = _between(page, "  function updatePress() {", "\n  }\n")
    assert (
        "press.setAttribute('aria-disabled', String(busy || !plan.send.length"
        " || plan.over > 0 || gateHeld()));" in update
    )
    fold = _between(page, "  function fillFold() {", "\n  }\n")
    assert "if (!ruleLost && rule.source !== 'owner')" in fold
    # Defined once, shown in the gate, toasted on a press.
    code = re.sub(r"/\*.*?\*/|//[^\n]*", "", page, flags=re.S)
    assert code.count("RULE_LOST") == 3
    assert _OFF_FACE.search(RULE_LOST) is None


def test_f8_a_review_sets_picture_is_folded_and_its_legend_after_it() -> None:
    """r4-i F8 (the round's fallback: the picture's own title, subtitle,
    legend and caption say "windows" and "review set"): no review picture on
    the face; the picture with the tool's caption (figureBlock's figcaption)
    is the first node of the fold, before the card table and the notes. r4-j
    J4: its one legend line is the fold's next node, only with a picture, and
    nothing of it is on the face (it explained no picture in view there);
    the check button and the plan's sheet come before the fold (r5-e R3)."""
    page = _read("result.html")
    assert "reviewPicture" not in page
    figure = _between(page, "function figureBlock(app, data,", "\n}\n")
    caption = (
        "data.figure.caption ? OE.el('figcaption', {}, data.figure.caption) : null,"
    )
    assert caption in figure
    assert page.count("OE.el('figcaption'") == 1
    show = _between(page, "function show(app, params) {", "\n}\n")
    # r5-e: the legend sits inside the picture's own box, over its buttons.
    pushed = (
        "const legend = OE.el('div', { class: 'small review-legend' },"
        " REVIEW_LEGEND(unit));\n"
        "    reviewFigure.insertBefore(legend, reviewFigure.querySelector(':scope > .row'));\n"
        "    inFold.push(reviewFigure);"
    )
    for text in (
        "const reviewFigure = review && data.figure && data.figure.svg"
        " ? figureBlock(app, data) : null;",
        "  if (reviewFigure) {\n",
        pushed,
        "else if (data.figure && data.figure.svg && !reviewFigure)"
        " put(figureBlock(app, data), false);",
    ):
        assert text in show, text
    # The review figure, and other tools' figure: no third figure.
    assert show.count("figureBlock(app, data)") == 2
    # The legend once, in the fold's list; never appended to the face.
    assert show.count("REVIEW_LEGEND(") == 1
    assert "root.append(OE.el('div', { class: 'small review-legend' }" not in show
    assert _in_order(
        show,
        "if (reviewFigure) {\n",
        pushed,
        "put(cardBlock(data.card), true);",
        "put(notesBlock(data.must_state), true);",
    )
    # Round r5-e, R3: the one action (the check button, the plan's sheet
    # button) right under the lines that lead to it, then one Details fold,
    # whose last node is the look-first list's numbers.
    assert _in_order(
        show,
        "if (lead) root.append(lead);",
        "const listBlock = data.review && typeof data.review === 'object'"
        " ? reviewBlock(app, data.review) : null;",
        "if (listBlock) root.append(listBlock);",
        "root.append(reasonBlock(data.reason));",
        pushed,
        "if (plan) inFold.push(planFile(data));",
        "if (listBlock && listBlock.numbers) inFold.push(listBlock.numbers);",
        "root.append(...onFace);",
        "if (ask) root.append(ask);",
        "if (plan) root.append(sheetButton(app, data, figureSlot));",
        "if (inFold.length) root.append(OE.fold(FOLD_LABEL, ...inFold));",
    )
    assert "REVIEW_LEGEND(" not in _between(show, "const lead = ", "reviewBlock(app")


def test_o2_the_grid_draws_a_cannot_tell_square_empty_and_dashed() -> None:
    """r4-i O2: a cell with unjudged true is drawn span.oe-sq.cannot (no fill
    class, never ' wrong'), left out of the fills' count and order; its row
    says "Cannot tell" (the server's {unjudged} entry, else CANNOT_KEY, the
    sheet's own word), with its count in the figure's aria-label. Round r5-g,
    G1: the squares are a tally, one row per kind in the order the cells
    come, so the Cannot tell row is last (its cells are); the legend, its
    dashed swatch and the outline's entry are gone with the grid
    (tests/mcp_app/test_mcp_r5g_result.py)."""
    script = _read("figures.js")
    assert "  // WORDING: r4i O2\n  const CANNOT_KEY = 'Cannot tell';" in script
    assert sheet.CANNOT_TELL == "Cannot tell"
    assert "  const isCannot = (c) => c.unjudged === true;" in script
    squares = _between(script, "  const squares = (", "\n  };")
    for text in (
        "const judged = cells.filter((c) => !isCannot(c));",
        "for (const c of judged) count.set(key(c.seen), (count.get(key(c.seen)) || 0)"
        " + 1);",
        "for (const v of [...classes.map((c) => c.code), ...judged.map((c) => c.seen)])"
        " {",
        "const wrong = !isCannot(cell) && isWrong(cell);",
        "mark: isCannot(cell) ? 'oe-sq cannot' : 'oe-sq ' + fill(cell.seen)"
        " + (wrong ? ' wrong' : ''),",
        "const cannot = legend.find((e) => e.unjudged === true && words(e.text));",
        "text: isCannot(cell) ? (cannot ? words(cannot.text) : CANNOT_KEY)",
        "const given = legend.find((e) => !e.wrong && !e.unjudged"
        " && key(e.code) === code);",
        "'aria-label': rows.map((r) => r.text + ' (' + r.n + ')').join('. '),",
    ):
        assert text in squares, text
    for gone in (
        "for (const c of cells) count.set(",
        "...cells.map((c) => c.seen)",
        "oe-swatch",
        "entries.push(",
    ):
        assert gone not in squares, gone
    assert CANNOT_CSS in script
    assert ".oe-swatch" not in script
    assert _OFF_FACE.search("Cannot tell") is None


def test_o2_the_result_card_says_no_split_in_place_of_the_class_lines() -> None:
    """r4-i O2: under the squares, the server's squares.no_split (a string)
    as p.small.oe-no-split; the heading is as before (n_checked counts the
    Cannot tell squares). Round r5-b, J23: the class lines with text are
    made beside the squares (p.small.oe-class-line) but go to the Details
    fold, after the rule used; only the no-split line stays under the
    squares, since it explains the dashed ones."""
    card = _between(_read("result.html"), "function resultCard(", "\n}\n")
    said = (
        "text(sq.no_split) ? OE.el('p', { class: 'small oe-no-split' },"
        " sq.no_split) : null));"
    )
    assert said in card
    assert "lines.push(" not in card and "...lines" not in card
    assert _in_order(
        card,
        "let classLines = [];",
        "classLines = (Array.isArray(sq.classes) ? sq.classes : [])",
        ".map((c) => OE.el('p', { class: 'small oe-class-line' }, c.line));",
        "RESULT_WORDS.squaresHead(n !== null ? n : sq.cells.length, unit)),"
        " OE.fig.squares(sq),\n      text(sq.no_split) ? OE.el(",
        "const folded = [];",
        "RESULT_WORDS.ruleUsed(",
        "classLines.forEach((line) => folded.push(line));",
        "fewDetail.forEach(",
    )


def test_o3_the_class_check_words_and_message() -> None:
    """r4-i O3, r4-j J1: the button's words and its Details line (squares and
    points), each marked; the message its press sends (OE.JOB_TEXT.byClass)
    names the result, on squares the part's file only when given and the
    route that keeps to it (the scores_path olmoearth_check_map gives for
    that part: olmoearth_plan_label_sample has no part argument), on points
    the planner by result_id, and asks how many squares (or points) first."""
    page = _read("result.html")
    words = _between(page, "const RESULT_WORDS = {", "\n};")
    assert f"  // WORDING: r4i O3\n  classCheck: '{CLASS_CHECK}'," in words
    assert "  // WORDING: r4i O3\n  classCheckDetail: (unit) => " in words
    detail = _entry(words, "classCheckDetail: (unit) => ")
    for unit in ("square", "point"):
        shown = _concat(detail, {"unitOf(unit)": unit})
        assert shown == CLASS_CHECK_DETAIL.format(many=unit + "s"), unit
    for face in (CLASS_CHECK, CLASS_CHECK_DETAIL):
        assert _OFF_FACE.search(face) is None, face
    jobs = _between(_read("common.js"), "OE.JOB_TEXT = {", "\n};")
    assert "  // WORDING: r4j J1\n  byClass: (r) => " in jobs
    expr = _after(jobs, "  byClass: (r) => ", ",\n")
    assert expr.startswith(UNIT_HEAD) and expr.endswith(")")
    points, squares = expr[len(UNIT_HEAD) : -1].split(UNIT_ELSE)
    names = {"r.result_id": "r1", "r.part_file": PART_FILE}
    assert _concat(points, names) == BY_CLASS_POINTS
    assert "part_file" not in points
    assert PART_CHOICE in squares and WHICH_CHOICE in squares
    with_part = "' (the part in file ' + r.part_file + ')'"
    for part, in_part, which in (
        (with_part, IN_PART, "that part"),
        ("''", "", "this map"),
    ):
        chosen = squares.replace(PART_CHOICE, part)
        chosen = chosen.replace(WHICH_CHOICE, "'" + which + "'")
        want = BY_CLASS_SQUARES.format(part=in_part, which=which)
        assert _concat(chosen, names) == want, in_part
    assert "OE.named" not in expr and "window" not in expr


def test_o3_the_class_check_button_under_the_uncovered_lines() -> None:
    """r4-i O3: only when r.class_check names the result (and the part's file,
    when a string) by ids matching the job buttons' pattern; right after the
    uncovered block, one outlined button (no 'primary') whose press goes
    through OE.sendJob with an OE.typeSlot beside it; its Details line only
    beside it, after the rare classes' counted lines and the unseen names.
    Round r5-b, J23: the rule used is no longer on the face under the
    button; it is the Details fold's first line."""
    page = _read("result.html")
    check = _between(page, "function classCheckOf(r, unit) {", "\n}\n")
    for text in (
        "const id = /^[A-Za-z0-9_.:-]{1,200}$/;",
        "const c = r.class_check && typeof r.class_check === 'object'"
        " ? r.class_check : null;",
        "if (!c || typeof c.result_id !== 'string' || !id.test(c.result_id))"
        " return null;",
        "const partFile = typeof c.part_file === 'string' ? c.part_file : null;",
        "if (partFile !== null && !id.test(partFile)) return null;",
        "return { result_id: c.result_id, part_file: partFile,",
    ):
        assert text in check, text
    card = _between(page, "function resultCard(", "\n}\n")
    button = _between(card, "  if (classCheck) {\n    const typed", "\n  }\n")
    for text in (
        "const typed = OE.typeSlot();",
        "OE.el('div', { class: 'row oe-class-check' },",
        "onclick: () => OE.sendJob(app, OE.JOB_TEXT.byClass(classCheck), typed) },",
        "RESULT_WORDS.classCheck),\n      typed));",
    ):
        assert text in button, text
    assert "primary" not in button
    assert "const classCheck = classCheckOf(r, unit);" in card
    assert _in_order(
        card,
        "oe-uncovered",
        "const classCheck = classCheckOf(r, unit);",
        "oe-class-check' },",
        "const folded = [];",
        "RESULT_WORDS.ruleUsed(",
    )
    detail = (
        "folded.push(OE.el('p', { class: 'small oe-class-check-detail' },"
        " RESULT_WORDS.classCheckDetail(classCheck.unit)));"
    )
    assert "  if (classCheck) {\n    " + detail in card
    assert _in_order(
        card,
        "const folded = [];",
        "fewDetail.forEach(",
        "if (unseenDetail) folded.push(",
        detail,
        "if (rows.length) folded.push(",
        "OE.fold(RESULT_WORDS.details",
    )


def test_f2_the_rare_classes_counted_lines_go_first_in_details() -> None:
    """r4-i F2: the server's uncovered_detail (each rare class's counted line,
    when the face names two or more without counts) is in the Details fold,
    each as p.small.oe-few-detail, and never on the face; first of the
    fold's lines about classes nobody or few saw (round r5-b, J23: the rule
    used and the class lines now stand before them)."""
    card = _between(_read("result.html"), "function resultCard(", "\n}\n")
    for text in (
        "const fewDetail = (Array.isArray(r.uncovered_detail) ? r.uncovered_detail"
        " : []).filter((s) => text(s));",
        "fewDetail.forEach((s) => folded.push(OE.el('p', { class: 'small"
        " oe-few-detail' }, s)));",
    ):
        assert text in card, text
    assert card.index("uncovered_detail") > card.index("const folded = [];")
    assert card.count("fewDetail") == 2
    assert _in_order(
        card,
        "const folded = [];",
        "fewDetail.forEach(",
        "if (rows.length) folded.push(",
    )
