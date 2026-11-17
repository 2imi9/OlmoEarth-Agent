# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""The panels' words of the r4-e round, read from the view files (no test here
runs the scripts): the result card counts in the result's unit, squares or
points (r4-d-1 findings R2 and R7); the classes no square had are said on the
face and their names may sit in Details (the peer's item 5); a review set's
numbers fold names its rows by the block's unit (R8); "windows" goes from the
faces (the owner's ruling 6); a sheet that can show no pictures says nothing
of dates (N).

And the last round's (r4-h) words of the Check card, the result card, the
review set and the job text, each as the plan's string table gives it.

A JS concatenation of single-quoted strings and names is read back into its
text (:func:`_concat`), so a missing space between two joined pieces fails
here as it would on the card. The sheet's words (sheet.js, another builder's
file in r4-h) are read with :func:`_holds`, however the script joins them."""

from __future__ import annotations

import re

from olmoearth_agent.mcp_app import server, views

#: One token of a JS concatenation: a single-quoted string, a plus, or a name
#: (a call with no parentheses inside its own, such as unitOf(unit), is one
#: name).
_TOKEN = re.compile(
    r"\s*(?:'((?:[^'\\]|\\.)*)'|(\+)|([A-Za-z_][A-Za-z0-9_.]*(?:\([^()]*\))?))"
)

#: result.html's unit word: "point", else "square".
UNIT_OF = "const unitOf = (unit) => (unit === 'point' ? 'point' : 'square');"

#: The plan card's line while its figure is held (sheet.FIGURE_LATER says the
#: same words; tests/mcp_app/test_mcp_sheet.py holds the two equal): no unit,
#: since a plan's sheet is of squares or of points.
FIGURE_LATER = (
    "The plan's figure shades the map by its confidence, so it shows here once "
    "the whole sheet is labelled."
)

#: The job button's check request, its ids given and no Studio name.
CHECK_REQUEST = (
    "Check how wrong this map is: open its Check card (olmoearth_check_map) for "
    "the Studio prediction result r1 (prediction_id p1); I will pick how many "
    "squares to check there."
)

#: Where the sheet file is, for a sheet of squares (as before) and of points.
SHEET_FILE_SQUARES = (
    "To see where the sampled squares are, open the label sheet file "
    "(check_a_to_label.csv) in a spreadsheet or GIS. It lists each square's place "
    "and whether the map was right there. These are sampled squares only, not all "
    "of the map's errors."
)
SHEET_FILE_POINTS = (
    "To see where the sampled points are, open the label sheet file "
    "(points_to_label.csv) in a spreadsheet or GIS. It lists each point's place "
    "and whether the map was right there. These are sampled points only, not all "
    "of the map's errors."
)

#: An unfinished sheet's line, of squares (the e2e's UNFINISHED) and of points.
UNFINISHED_SQUARES = (
    "Not finished yet: 3 of 80 squares on this sheet have no answer. Answer every "
    "square, then press Estimate the error rate again."
)
UNFINISHED_POINTS = (
    "Not finished yet: 2 of 20 points on this sheet have no answer. Answer every "
    "point, then press Estimate the error rate again."
)

#: Cannot tell on a sheet with pictures, and on one that can show none (r4-h
#: S1, A/F04: in the sheet's Details fold); {one} its unit word. r4-i F5 and
#: O2: with Cannot tell answers the result card still draws the grid, so the
#: cost names the per-class lines only.
CANNOT_TELL = (
    "Cannot tell is for a {one} you cannot judge from the dates shown. Do not "
    "guess: a guess can make the result wrong. A Cannot tell does not: you still "
    "get a result, with a wider range, but without the per-class lines."
)
CANNOT_TELL_BARE = CANNOT_TELL.replace(" from the dates shown", "")

#: The Check card's words of r4-h (the plan's string table).
STEP_3 = "3. Someone who can recognise it in a satellite picture looks at each one."
CANNOT = (
    "Your rule and number are fixed once you press. You can stop and come back "
    "later; there is no result until every square is done. A random pick lands "
    "mostly on common classes; a rare class may get no square."
)
RULE_ASK = (
    "First write, in one sentence, how you will decide what class a square is. It "
    "is saved and shown with the result."
)
RULE_HINT = "For example: the class that fills most of the outlined square"
RULE_OWN = (
    "Not a kelp map? Write its own rule here. Left empty, the kelp rule above is used."
)
CLASS_MAP_LINES = (
    "This map can be checked, on single points picked at random instead of squares.",
    "It holds classes only, without how sure the model was. You still get how often "
    "it is wrong, with a range, for a grid of at most 256 points over the map, not "
    "for every pixel.",
)

#: The request the Check card's points button sends (r4-h C6, E/check-1).
POINTS_REQUEST = (
    "Check this map on points: Studio prediction result r1 (prediction_id p1). "
    "First ask me how many points to check and which way to pick them."
)

#: The result card's parts lines (r4-h R3, A/F03 FINAL).
OTHER_PART = (
    "Check the other part too. Then report the parts' results side by side; do not "
    "average or add them into one rate."
)
OTHER_PARTS = (
    "Check the other 3 parts too. Then report the parts' results side by side; do "
    "not average or add them into one rate."
)

#: The legend line under a review set's picture (r4-h V1, D/D-card-1 FINAL).
LEGEND_POINTS = (
    "Shade is the order only, most sure to least sure among the points read, not an "
    "amount of doubt; each square stands for the one pixel read in it."
)
LEGEND_SQUARES = (
    "Shade is the order only, most sure to least sure among the squares read, not "
    "an amount of doubt."
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


def _flat(js: str) -> str:
    """JS text with each run of joined single-quoted literals read as one (the
    joins ``' + '``, over lines too, dropped) and escaped quotes unescaped."""
    return re.sub(r"'\s*\+\s*'", "", js).replace("\\'", "'")


def _holds(js: str, text: str) -> bool:
    """Whether js says text, however its pieces are joined: text is cut at
    each placeholder ({one}) and after each stop, colon or semicolon, and
    every piece must be in js read as :func:`_flat` reads it."""
    flat = _flat(js)
    pieces = re.split(r"\{[a-z]+\}|(?<=[.?:;])\s+", text)
    return all(p.strip() in flat for p in pieces if p.strip())


def test_the_concatenation_reader_reads_joined_pieces() -> None:
    """The reader itself: escapes, names, calls and line breaks."""
    expr = "'It\\'s ' + one\n    + ' of ' + unitOf(unit) + 's'"
    assert _concat(expr, {"one": "1", "unitOf(unit)": "point"}) == "It's 1 of points"
    js = "a: 'Cannot judge a ' + one + '? Press'\n    + ' Cannot tell.',"
    assert _holds(js, "Cannot judge a {one}? Press Cannot tell.")
    assert not _holds(js, "Cannot judge a {one}? Press Cannot see.")


def test_the_result_cards_words_follow_its_unit() -> None:
    """R2 and R7: the card counts what was checked in the result's unit
    (r.unit), so a sheet of points is never graded onto a card of squares;
    the words of a sheet of squares are as before. r4-h R4: the count says
    how many were Cannot tell ("80 squares checked, 2 Cannot tell")."""
    page = _read("result.html")
    assert UNIT_OF in page
    tag = "// WORDING: squares; r4h R4 (checked), r4h R3 (partsHead, otherPart)\n"
    assert tag + "const RESULT_WORDS = {" in page
    words = _between(page, "const RESULT_WORDS = {", "\n};")
    assert (
        "  checked: (n, unit, k) => OE.counted(n, unitOf(unit)) + ' checked'"
        " + (k > 0 ? ', ' + k + ' Cannot tell' : '')," in words
    )
    assert "OE.counted(n, 'square')" not in page
    assert re.search(r"\bwindows?\b", words, re.I) is None
    head = _after(words, "squaresHead: (n, unit) => (n === 1 ? ", "),\n")
    one, many = head.split("\n    : ")
    for unit, n, said_one, said_many in (
        ("square", "80", "The square you checked", "The 80 squares you checked"),
        ("point", "20", "The point you checked", "The 20 points you checked"),
    ):
        names = {"unitOf(unit)": unit, "n": n}
        assert _concat(one, names) == said_one
        assert _concat(many, names) == said_many
    sheet_file = _entry(words, "sheetFile: (name, unit) => ")
    for unit, name, said in (
        ("square", "check_a_to_label.csv", SHEET_FILE_SQUARES),
        ("point", "points_to_label.csv", SHEET_FILE_POINTS),
    ):
        assert _concat(sheet_file, {"unitOf(unit)": unit, "name": name}) == said
    unfinished = _entry(words, "unfinished: (left, n, unit) => ")
    for unit, left, n, said in (
        ("square", "3", "80", UNFINISHED_SQUARES),
        ("point", "2", "20", UNFINISHED_POINTS),
    ):
        names = {"unitOf(unit)": unit, "left": left, "n": n}
        assert _concat(unfinished, names) == said


def test_the_result_card_and_the_unfinished_line_read_the_unit() -> None:
    """The card takes the result view's unit for every count word; the
    unfinished line takes the unfinished counts' unit, else the view's."""
    page = _read("result.html")
    card = _between(page, "function resultCard(", "\n}\n")
    for text in (
        "const unit = unitOf(r.unit);",
        "RESULT_WORDS.checked(n, unit, unjudged)",
        "RESULT_WORDS.squaresHead(n !== null ? n : sq.cells.length, unit)",
        "RESULT_WORDS.sheetFile(r.sheet_name, unit)",
    ):
        assert text in card, text
    assert (
        "const unjudged = Number.isInteger(r.n_unjudged) && r.n_unjudged > 0"
        " ? r.n_unjudged : 0;" in card
    )
    show = _between(page, "function show(app, params) {", "\n}\n")
    assert "RESULT_WORDS.unfinished(left, of, unfinished.unit || data.unit)" in show


def test_the_classes_nobody_saw_are_said_under_the_headline() -> None:
    """The peer's item 5, moved by r4-h R2: a class seen too rarely or in no
    square has no class line of its own (its line is null and is not drawn);
    the server's uncovered lines say them directly under the headline, before
    the rule line and the grid; with more than five unseen classes their names
    (squares.unseen_detail) are in the Details fold, before its rows, grid or
    not (r4-i F2: after the rare classes' counted lines). Text only."""
    card = _between(_read("result.html"), "function resultCard(", "\n}\n")
    for text in (
        ".filter((c) => c && typeof c === 'object' && text(c.line))",
        "const unseenDetail = sq ? text(sq.unseen_detail) || null : null;",
        "if (unseenDetail) folded.push(OE.el('p', { class: 'small oe-unseen-detail' },"
        " unseenDetail));",
    ):
        assert text in card, text
    assert "unseen_line" not in card
    assert card.index("oe-uncovered") < card.index("RESULT_WORDS.squaresHead(")
    folded_at = card.index("if (unseenDetail) folded.push(")
    assert folded_at < card.index("if (rows.length) folded.push(")
    assert folded_at < card.index("OE.fold(RESULT_WORDS.details")
    assert "innerHTML" not in card


def test_the_review_sets_words_follow_the_blocks_unit() -> None:
    """R8: a Studio result's review set (its block's unit 'point') calls its
    rows points, in its cards, its pane and its numbers fold; a set of squares
    reads as before."""
    page = _read("result.html")
    assert "// WORDING: squares\nconst reviewWords = (unit) => {" in page
    assert "const REVIEW_WORDS = {" not in page
    review = _between(page, "const reviewWords = (unit) => {", "\n};")
    assert "const one = unitOf(unit);" in review
    starts = (
        "numbers: ",
        "heading: (k, total) => ",
        "back: (k) => ",
        "windowLabel: (k) => ",
        "firstShown: (n, total) => ",
    )
    for unit, said in (
        (
            {"one": "square", "One": "Square"},
            (
                "The numbers for these squares",
                "Square 3 of 7",
                "Back to square 3",
                "square 3",
                "The first 5 of 7 squares are shown here; the rest are in the full "
                "list.",
            ),
        ),
        (
            {"one": "point", "One": "Point"},
            (
                "The numbers for these points",
                "Point 3 of 7",
                "Back to point 3",
                "point 3",
                "The first 5 of 7 points are shown here; the rest are in the full "
                "list.",
            ),
        ),
    ):
        names = {**unit, "k": "3", "n": "5", "total": "7"}
        got = tuple(_concat(_entry(review, s), names) for s in starts)
        assert got == said


def test_the_numbers_fold_takes_the_blocks_unit() -> None:
    """C2: numbersFold(windows, unit), called with the block's unit both where
    the cards are drawn and where they are not."""
    page = _read("result.html")
    fold = _between(page, "function numbersFold(windows, unit) {", "\n}\n")
    assert "const words = reviewWords(unit);" in fold
    assert "OE.el('summary', {}, words.numbers)" in fold
    block = _between(page, "function reviewBlock(app, block) {", "\n}\n")
    for text in (
        "const unit = unitOf(block.unit);",
        "const REVIEW_WORDS = reviewWords(unit);",
        "{ colour: state.colour, choose, strip: true, unit }",
    ):
        assert text in block, text
    assert block.count("numbersFold(windows, unit)") == 2
    assert "numbersFold(windows)" not in page


def test_the_plan_cards_held_figure_line_says_no_window() -> None:
    """The owner's ruling 6: the line in a held figure's place says no
    "window" (and no unit: it is true of a sheet of squares or of points)."""
    page = _read("result.html")
    assert f'// WORDING: squares\nconst FIGURE_LATER = "{FIGURE_LATER}";' in page
    assert "shades each window" not in page


def test_the_check_request_says_squares() -> None:
    """The owner's ruling 6: the check request names squares to check."""
    jobs = _between(_read("common.js"), "OE.JOB_TEXT = {", "\n};")
    assert "  // WORDING: squares\n  check: (r) => " in jobs
    names = {"r.result_id": "r1", "r.prediction_id": "p1", "OE.named(r.name)": ""}
    assert _concat(_entry(jobs, "check: (r) => "), names) == CHECK_REQUEST
    assert "how many windows to label" not in jobs


def test_the_pictures_name_the_outlined_square_by_the_unit() -> None:
    """The owner's ruling 6: a context picture's caption says "square" (or
    "point"), never "window"; the editor and a review set's pane give their
    unit; the finish button's fallback message names the sheet's unit."""
    script = _read("sheet.js")
    picture = _between(script, "OE.picture = (", "\n};")
    for text in (
        "unit = 'square') => {",
        "'Context, ' + (unit === 'point' ? 'point' : 'square') + ' outlined'",
    ):
        assert text in picture, text
    assert "// WORDING: squares" in picture
    # The dates block (each date its own block, r4-h A/F24) passes its unit on.
    dates = _between(script, "OE.datesBlock = (", "\n};")
    assert "unit = 'square' } = {}) => {" in dates
    assert re.search(r"OE\.picture\([^;]*\bunit\)", dates) is not None
    editor = _between(script, "OE.sheetEditor = (app, data,", "\n};")
    for text in (
        "      unit: W.one,\n",
        "      // WORDING: squares\n      const message = data.finish_message || (",
        "'I have labelled every ' + W.one + ' of the sheet ' + data.name",
    ):
        assert text in editor, text
    for gone in ("window outlined", "labelled every window"):
        assert gone not in script, gone


def test_a_sheet_that_can_show_no_pictures_says_nothing_of_dates() -> None:
    """N: with pictures_possible false, the note that no date is marked clear
    (A/sheet-3) is left out, and Cannot tell reads "you cannot judge" with no
    "from the dates shown"; a sheet with pictures says the dates. r4-h S1
    (A/F04): both in the string table's words; r4-i F5 and O2: the cost no
    longer names the picture of squares (the grid stays)."""
    script = _read("sheet.js")
    words = _between(script, "OE.sheetWords = (unit) => {", "\n};")
    assert _holds(words, CANNOT_TELL)
    assert _holds(words, CANNOT_TELL_BARE)
    assert "but without the picture of" not in _flat(words)
    assert "dates" not in _entry(words, "cannotTellBare: ")
    editor = _between(script, "OE.sheetEditor = (app, data,", "\n};")
    chosen = r"picturesPossible\s*\?\s*W\.cannotTell\s*:\s*W\.cannotTellBare\b"
    assert re.search(chosen, editor) is not None
    assert re.search(r"picturesPossible \?[^;]*\bW\.clear\b", editor) is not None
    # The old words are gone (A/sheet-4's "Do not guess instead").
    assert "Do not guess instead" not in _flat(script)


def test_r4h_the_check_cards_steps_counts_and_press_line() -> None:
    """r4-h, the Check card before the press: step 1 names the square's size
    when every part's rounds the same (A/F13), step 3 who should look (critic
    gap 2); the counts are words only, no band and no "points" (C1); the line
    over the press says what it fixes and what a random pick misses (C2)."""
    page = _read("check.html")
    steps = _between(page, "const STEPS = [", "];")
    assert f"  '{STEP_3}'," in steps
    assert "  '1. The map is cut into small squares.'," in steps
    assert "// WORDING: squares; r4h critic gap 2 (step 3)\nconst STEPS = [" in page
    assert "You look at each one" not in page
    assert "// WORDING: r4h A/F13\nconst STEP_SIZED = " in page
    sized = _after(page, "const STEP_SIZED = (metres) => ", ";\n")
    said = _concat(sized, {"metres": "60"})
    assert said == "1. The map is cut into squares about 60 m a side."
    first = _between(page, "const firstStep = (models) => {", "\n};")
    for text in (
        "Math.round(m.square.metres) : null));",
        "metres.every((v) => v !== null && v === metres[0])",
        "? STEP_SIZED(metres[0].toLocaleString('en-US')) : STEPS[0];",
    ):
        assert text in first, text
    assert "fig('steps', [firstStep(models), ...STEPS.slice(1)])" in page
    assert (
        "// WORDING: r4h C1\nconst WIDTH_WORDS = [[0.20, 'Quick. A wide range.'],"
        " [0.10, 'Longer. A narrower range.'],\n  [0.05, 'A lot of work. A narrow"
        " range.']];" in page
    )
    counts = _between(page, "  function countsBlock(m) {", "\n  }\n")
    assert "points" not in counts and "fig(" not in counts
    assert "points" not in _after(page, "const WIDTH_WORDS = ", ";\n")
    for gone in ("BAND_LABEL", "'band'", "A rough answer", "A fair", "A sharp answer"):
        assert gone not in page, gone
    assert "// WORDING: r4h C2\nconst CANNOT = " in page
    assert _concat(_after(page, "const CANNOT = ", ";\n"), {}) == CANNOT
    assert "You check the squares yourself" not in page


def test_r4h_the_check_cards_rule_gate_and_box() -> None:
    """r4-h C4 (C/check-1 FINAL): no red line; the ask, the box's label "Your
    rule" and an example as the box's placeholder only (never sent: the rule
    sent is the box's value); the pictures' limit moves to Details. C5
    (B/check-own-rule-box FINAL): the box beside the owner's rule says what
    an empty box does, under a label in body type."""
    page = _read("check.html")
    assert "NO_RULE" not in page and "No labelling rule is set" not in page
    assert "// WORDING: r4h C4\nconst RULE_ASK = " in page
    assert _concat(_after(page, "const RULE_ASK = ", ";\n"), {}) == RULE_ASK
    assert "// WORDING: r4h C4\nconst RULE_LABEL = 'Your rule';" in page
    assert f"// WORDING: r4h C4\nconst RULE_HINT = '{RULE_HINT}';" in page
    # The example is the placeholder only: defined once, used once.
    assert page.count("RULE_HINT") == 2
    gate = _between(page, "  function fillGate() {", "\n  }\n")
    for text in (
        "ruleInput.setAttribute('placeholder', RULE_HINT)",
        "ruleInput.removeAttribute('placeholder')",
        "OE.el('label', { class: 'field-label', for: 'oe-rule' }, RULE_LABEL)",
        "OE.el('label', { class: 'own-rule', for: 'oe-rule' }, RULE_OWN)",
        # r4-i F6: a press with no reply shows the gate's one line.
        "const mode = ruleLost ? 'lost'",
    ):
        assert text in gate, text
    for gone in (
        "class: 'error'",
        "RULE_PICTURES",
        "field-label', for: 'oe-rule' }, RULE_OWN",
    ):
        assert gone not in gate, gone
    sent = _after(page, "  const ruleText = () => ", ";\n")
    assert "ruleInput.value" in sent and "placeholder" not in sent
    fold = _between(page, "  function fillFold() {", "\n  }\n")
    assert (
        "if (!ruleLost && rule.source !== 'owner') nodes.push(OE.el('p', {},"
        " RULE_PICTURES));" in fold
    )
    assert f"// WORDING: r4h C5\nconst RULE_OWN = '{RULE_OWN}';" in page
    assert ".gate label.own-rule { display: block; margin-bottom: 4px; }" in page
    assert "If this map is not of kelp" not in page


def test_r4h_the_check_cards_parts() -> None:
    """r4-h C3 (A/F06 FINAL): a part not picked under an open editor says how
    to pick it; A/F11: each part's heading carries its count of squares; V6:
    a part with no name of the server's is "Part i", never its zone; A/F08:
    the parts figure is drawn again as parts are picked and checked, each
    labelled so (the figure's states are figures.js's)."""
    page = _read("check.html")
    assert (
        "// WORDING: r4h C3\nconst LATER = 'To pick this part\\'s squares, ask Claude"
        " to open the Check card again.';" in page
    )
    assert (
        "const PART_HEAD = (name, n) => name + (Number.isInteger(n) && n >= 0\n"
        "  ? ' (' + n.toLocaleString('en-US') + (n === 1 ? ' square)' : ' squares)')"
        " : '');" in page
    )
    assert "OE.el('h3', {}, PART_HEAD(label(m), m.windows))" in page
    assert "// WORDING: r4h V6\nconst PART_N = (i) => 'Part ' + i;" in page
    assert "const label = (m) => m.part || PART_N(m.index + 1);" in page
    assert "m.part || m.name" not in page
    names = {"name": "East part"}
    shown = _concat(_after(page, "const PART_OPEN = (name) => ", ";\n"), names)
    assert shown == "East part: not picked yet"
    shown = _concat(_after(page, "const PART_PICKED = (name) => ", ";\n"), names)
    assert shown == "East part: picked, not checked yet"
    parts = _between(page, "  function fillParts() {", "\n  }\n")
    for text in (
        "if (nParts < 2) return;",
        "state: 'picked', label: complete(m.draw) ? name : PART_PICKED(name) };",
        "state: 'open', label: m.kind === 'unknown' ? name : PART_OPEN(name) };",
        "}), { mode: 'plan' });",
    ):
        assert text in parts, text
    for where in ("  function renderAll() {", "  async function refreshZone(m) {"):
        assert "    fillParts();\n" in _between(page, where, "\n  }\n"), where
    assert (
        "if (nParts > 1) root.append(OE.el('div', {}, PARTS_LINE(nParts)), partsSlot);"
        in page
    )


def test_r4h_the_check_cards_rule_follows_the_servers_reply() -> None:
    """r4-h Q4 and Q11: after every press the card shows the rule the server's
    reply gives (drawn again); a reply that refused a typed rule picked
    nothing and keeps the picks; a press with no reply shows no rule and no
    box (the parts send the user to Claude; r4-i F6: one line in their place,
    and the press held, tests/mcp_app/test_mcp_r4i_views.py); the card's own
    guess of the rule from what it sent is gone, and the counts' reads never
    touch the rule."""
    page = _read("check.html")
    press = _between(page, "  async function drawPicked() {", "\n  }\n")
    for text in (
        "got = await OE.call(app, 'olmoearth_app_check_draw', args,"
        " { timeout: PRESS_WAIT_MS });",
        "rule = ruleModel(got.rule);\n      ruleLost = false;\n      gateMode = '';",
        "const refused = got && typeof got.refused === 'string' && got.refused"
        " ? got.refused : null;",
        "OE.toast(NONE_PICKED + bare(refused) + '.', true);",
        "ruleLost = true;",
    ):
        assert text in press, text
    assert press.index("const refused =") < press.index("takeAnswer(m, answers[i])")
    for gone in (
        "rule.text = typed",
        "claimed",
        "TYPED_SCOPE",
        "rule.replaceable = false",
    ):
        assert gone not in press, gone
    assert "TYPED_SCOPE" not in page
    assert "// WORDING: squares\nconst NONE_PICKED = 'No square was picked: ';" in page
    assert "let rule = ruleModel(data.rule);" in page
    refresh = _between(page, "  async function refreshZone(m) {", "\n  }\n")
    assert "rule =" not in refresh and "reading_rule" not in refresh


def test_r4h_a_class_map_without_scores_offers_the_check_on_points() -> None:
    """r4-h C6 (E/check-2, E/check-1 FINALs): the refusal's face says the map
    can be checked on points, in two lines; one contained button sends the
    points request with the result's id (OE.JOB_TEXT.points, from the view's
    points_request); the tool's sentence is in a closed Details fold. The
    request names no Studio name and asks Claude to ask the two questions."""
    page = _read("check.html")
    lines = _after(page, "// WORDING: r4h C6\nconst CLASS_MAP = [\n", "\n];")
    said = tuple(_concat(s, {}) for s in lines.rstrip(",").split(",\n"))
    assert said == CLASS_MAP_LINES
    assert "// WORDING: r4h C6\nconst ON_POINTS = 'Check this map on points';" in page
    card = _between(page, "function classMapCard(app, data, reason) {", "\n}\n")
    for text in (
        "data.points_request",
        "class: 'primary',",
        "OE.sendJob(app, OE.JOB_TEXT.points({ result_id: resultId,"
        " prediction_id: predictionId }), typed)",
        "...CLASS_MAP.map((s) => OE.el('div', {}, s)),",
        "root.append(OE.fold(DETAILS, OE.el('div', { class: 'small' }, reason)));",
    ):
        assert text in card, text
    show = _between(page, "function show(app, params) {", "\n}\n")
    assert "if (!data.ok && reason) {\n    classMapCard(app, data, reason);" in show
    assert "This card needs a map with scores" not in page
    jobs = _between(_read("common.js"), "OE.JOB_TEXT = {", "\n};")
    expr = _after(jobs, "  // WORDING: r4h C6\n  points: (r) => ", ",\n")
    head, rest = expr.split("\n    + (r.prediction_id ? ", 1)
    middle, tail = rest.split(" : '')\n    + ", 1)
    names = {"r.result_id": "r1", "r.prediction_id": "p1"}
    with_id = _concat(head, names) + _concat(middle, names) + _concat(tail, {})
    assert with_id == POINTS_REQUEST
    without = _concat(head, names) + _concat(tail, {})
    assert without == POINTS_REQUEST.replace(" (prediction_id p1)", "")
    assert "OE.named" not in expr and "name" not in expr


def test_r4h_a_chosen_answer_alone_looks_chosen() -> None:
    """r4-h A/F22: no hover fill or border change on an answer not chosen (an
    enabled one rests as it is); a chosen answer is filled dark, its words
    the canvas colour, in 600."""
    common = _read("common.js")
    assert (
        '.face .answers button:not([aria-pressed="true"]):not([aria-disabled="true"])'
        ":not(:disabled):hover {\n  background: transparent; border-color:"
        " var(--outline); }" in common
    )
    assert (
        '.face .answers button[aria-pressed="true"] { background: var(--text);'
        " color: var(--canvas); border-color: var(--text);\n  font-weight: 600; }"
        in common
    )
    old = "background: var(--selected); border-color: var(--text); font-weight"
    assert old not in common


def test_r4h_the_result_cards_headline_and_line() -> None:
    """r4-h R1 and R4: the server's headline (rate and 95% range in one
    sentence) over the line figure (a dot, or with Cannot tell a dark bar
    from the rate's low to high end over the band), then the hedge; no
    "range:" line, no "Error rate:" line, no good-enough line (R5: the
    server's last line says it); no text but the headline without the
    figure helpers."""
    page = _read("result.html")
    rate = _between(page, "function rateBlock(r) {", "\n}\n")
    for text in (
        "if (!text(r.headline)) return null;",
        "OE.el('p', { class: 'big' }, r.headline),",
        "rateLow: finite(r.rate_low) ? r.rate_low : null,"
        " rateHigh: finite(r.rate_high) ? r.rate_high : null,",
        "rateText: rate !== null ? text(r.rate_text) : text(r.rate_range_text),"
        " rangeText: text(r.range_text),",
        "text(r.hedge) ? OE.el('p', { class: 'oe-hedge' }, r.hedge) : null);",
    ):
        assert text in rate, text
    at = [rate.index(t) for t in ("r.headline),", "    line,\n", "text(r.hedge)")]
    assert at == sorted(at)
    for gone in ("RESULT_WORDS.range", "errorRate", "goodEnough", "'range: '"):
        assert gone not in page, gone
    card = _between(page, "function resultCard(", "\n}\n")
    for text in (
        "const uncovered = (Array.isArray(r.uncovered) ? r.uncovered : [])"
        ".filter((s) => text(s));",
        "card.append(OE.el('div', { class: 'stack oe-uncovered' },"
        " uncovered.map((s) => OE.el('p', {}, s))));",
        "if (text(r.last_line)) card.append(OE.el('p', { class: 'small' },"
        " r.last_line));",
    ):
        assert text in card, text
    order = (
        "const rate = rateBlock(r);",
        "oe-uncovered",
        "RESULT_WORDS.ruleUsed(",
        "RESULT_WORDS.squaresHead(",
        "RESULT_WORDS.partsHead",
        "r.last_line",
        "OE.fold(RESULT_WORDS.details",
    )
    at = [card.index(t) for t in order]
    assert at == sorted(at)


def test_r4h_the_result_cards_parts_are_never_averaged() -> None:
    """r4-h R3 (B/result-parts-not-averaged, A/F03 FINALs): the parts' heading
    says one result per part, never averaged; the line while a part is not
    checked is the server's SAY_OTHER (SAY_OTHERS for k parts), word for
    word."""
    page = _read("result.html")
    words = _between(page, "const RESULT_WORDS = {", "\n};")
    assert "  partsHead: 'One result per part, never averaged'," in words
    assert "Which part this is" not in page
    after = _after(words, "otherPart: (k) => (k === 1\n    ? ", "),\n")
    one, many = after.split("\n    : ")
    assert _concat(one, {}) == OTHER_PART == server.SAY_OTHER.strip()
    shown = _concat(many, {"k": "3"})
    assert shown == OTHER_PARTS == server.SAY_OTHERS.format(k="3").strip()


def test_r4h_a_review_sets_legend_follows_its_picture_in_the_fold() -> None:
    """r4-h V1 (D/D-card-1 FINAL): a review tool's picture has one legend
    line, by the block's unit; r4-i F8 (the round's fallback): the picture,
    with the tool's caption, is the fold's first node; r4-j J4: the legend
    line is the fold's next node, after the window cards and never on the
    face; V7: a plan's and a review set's card table and notes always in the
    fold, lead or not; other figures as before. V3: the result card's lines
    are spaced."""
    page = _read("result.html")
    tools = "['olmoearth_review_set', 'olmoearth_review_set_from_result']"
    assert f"const REVIEW_TOOLS = {tools};" in page
    assert "// WORDING: r4h V1\nconst REVIEW_LEGEND = (unit) => " in page
    legend = _after(
        page, "const REVIEW_LEGEND = (unit) => (unitOf(unit) === 'point'\n  ? ", ");\n"
    )
    point, square = legend.split("\n  : ")
    assert _concat(point, {}) == LEGEND_POINTS
    assert _concat(square, {}) == LEGEND_SQUARES
    show = _between(page, "function show(app, params) {", "\n}\n")
    for text in (
        "const review = REVIEW_TOOLS.includes(data.tool);",
        "const reviewFigure = review && data.figure && data.figure.svg"
        " ? figureBlock(app, data) : null;",
        "inFold.push(reviewFigure, OE.el('div', { class: 'small review-legend' },"
        " REVIEW_LEGEND(unit)));",
        "const keepFolded = plan || review;",
        "const put = (node, keep) => { (lead || (keep && keepFolded) ? inFold"
        " : onFace).push(node); };",
        "if (Array.isArray(data.card) && data.card.length) put(cardBlock(data.card),"
        " true);",
        "else if (data.figure && data.figure.svg && !reviewFigure)"
        " put(figureBlock(app, data), false);",
        "if (Array.isArray(data.must_state) && data.must_state.length)"
        " put(notesBlock(data.must_state), true);",
        "if (inFold.length) root.append(OE.fold(FOLD_LABEL, ...inFold));",
    ):
        assert text in show, text
    assert "reviewPicture" not in page
    legend_at = show.index("REVIEW_LEGEND(unit)")
    assert show.index("reviewBlock(app, data.review)") < legend_at
    assert legend_at < show.index("OE.fold(FOLD_LABEL")
    margins = page.index(".oe-result p, .oe-result h2, .oe-result h3 { margin: 0; }")
    assert page.index("section.oe-result > * + * { margin-top: 12px; }") > margins
    assert ".oe-result .stack > * + * { margin-top: 8px; }" in page
