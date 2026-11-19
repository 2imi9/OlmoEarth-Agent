# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Round r5-j (18 Nov 2026), track CHECK, as specs/r5-j-round.md gives it:
the Check card's items of the audit after r5-i
(specs/r5-i-audit/build_list.md B3, B5, B6 and B9).

- B3: on a map in parts with every square answered and no sheet open, the
  heading says to press Estimate the error rate in each part's sheet, one
  result per part (it said to open each part's sheet in turn, then press,
  which read as one press for the map); on a map in one part it keeps its
  words, and the rule on which line shows does not change.
- B5: a card whose map is being prepared says, under its status line and
  outside it, that it fills in by itself and that the user can keep
  chatting; that line goes when the card cannot ask (no result id) and when
  it stopped waiting. Once it stopped waiting its line, still the status,
  is the failed card's red card and says, as that card does, that what was
  picked or saved before is kept. The turning mark sits by the line's first
  row, its turning edge in the text's colour.
- B6: a part's heading on a map in parts reads "West part (300 squares in
  all)": "(300 squares)" right over the count buttons read as a fourth count
  or as the work to do. The count buttons do not change.
- B9: on a map in parts each part's section is named by its heading
  (aria-labelledby, the heading's id), so a screen reader tells two parts'
  "Open the sheet" apart; a map in one part has no heading and no name.

The views are read as text (no test here runs the scripts); a JS
concatenation of single-quoted strings and names is read back into its text
(:func:`_concat`). The browser harness (scripts/mcp_views_e2e.py) draws
them: 13b, 13e and 13f the heading (B3), 12p, 12q and 12r the preparing card
(B5), the parts' headings of 12, 12n, 12p, 13 and 14 (B6), and 13b and 13e
each part's name (B9).
"""

from __future__ import annotations

import re

from olmoearth_agent.mcp_app import sheet, views

#: One token of a JS concatenation: a single-quoted string, a plus, or a name
#: (a call with no parentheses inside its own is one name).
_TOKEN = re.compile(
    r"\s*(?:'((?:[^'\\]|\\.)*)'|(\+)|([A-Za-z_][A-Za-z0-9_.]*(?:\([^()]*\))?))"
)

#: Words no face holds (round r5-e's rule R7, and the owner's ruling 6).
_OFF_FACE = re.compile(
    r"\b(?:windows?|UTM|zones?|intervals?|strat(?:um|a)|design|budget"
    r"|recommended)\b",
    re.I,
)

#: B3: the heading with every square answered and no sheet open, on a map in
#: parts (new) and in one part (unchanged), and the start they share.
ALL_ANSWERED_PARTS = (
    "Every square is answered. In each part's sheet, press Estimate the error "
    "rate: one result per part."
)
ALL_ANSWERED_ONE = (
    "Every square is answered. Open the sheet, then press Estimate the error rate."
)
ALL_ANSWERED_HEAD = "Every square is answered. "

#: B5: the line under the preparing card's status, outside it.
PREP_FILLS = (
    "This card fills in by itself when the map is ready; you can keep chatting."
)
#: B5: the line once the card stopped waiting, and the failed card's plain
#: line (round r5-d R01), whose middle sentence it takes.
PREP_TOO_LONG = (
    "Preparing this map took too long. Anything picked or saved before is kept. "
    "Ask Claude to open the Check card again."
)
CHECK_FAILED = (
    "The check could not start. Anything picked or saved before is kept. Ask "
    "Claude what to do next: it has the reason."
)
KEPT = "Anything picked or saved before is kept."

#: B6: a part's heading as check.html's PART_HEAD makes it, for 300 squares,
#: for one, for 1,234, and with no count known.
HEADS: tuple[tuple[str, int | None, str], ...] = (
    ("West part", 300, "West part (300 squares in all)"),
    ("Part 2", 1, "Part 2 (1 square in all)"),
    ("East part", 1234, "East part (1,234 squares in all)"),
    ("Part 3", None, "Part 3"),
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


def _concat(expr: str, names: dict[str, str] | None = None) -> str:
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
            out.append((names or {})[name])
        at = match.end()
    return "".join(out)


def _code(page: str) -> str:
    """The page with its comments taken out."""
    return re.sub(r"/\*.*?\*/|//[^\n]*", "", page, flags=re.S)


def _in_order(text: str, *pieces: str) -> bool:
    """Whether each piece is in text, each after the one before it."""
    at = [text.index(p) for p in pieces]
    return at == sorted(at)


def _card(page: str) -> str:
    """check.html's preparingCard, whole."""
    return _between(page, "function preparingCard(app, data) {", "\n}\n")


def _all_answered(page: str, several: bool) -> str:
    """ALL_ANSWERED's text as check.html makes it for several (more than one
    part has a sheet) or not: its start, then the branch of its ternary,
    each read back (:func:`_concat` reads no ternary)."""
    expr = _after(page, "const ALL_ANSWERED = (several) => ", ";\n")
    head, rest = expr.split("\n  + (several ? ", 1)
    parts, rest = rest.split("\n    : ", 1)
    one = rest.removesuffix(")")
    return _concat(head) + _concat(parts if several else one)


# --------------------------------------------------------------------------- B3


def test_b3_every_part_answered_says_one_result_per_part() -> None:
    """B3: on a map in parts the heading ties the press to each part's sheet
    and says that each gives one result; it no longer says to open each
    part's sheet in turn and then press. On a map in one part (or where only
    one part has a sheet) the line keeps its words. The press it names is the
    sheet's own finish button for an estimate."""
    page = _read("check.html")
    tags = (
        "// WORDING: r5j B3\n// WORDING: r5i B2\n// WORDING: r5f F11\n"
        "// WORDING: r5b J17\nconst ALL_ANSWERED = (several) => "
    )
    assert tags in page
    expr = (
        "const ALL_ANSWERED = (several) => 'Every square is answered. '\n"
        "  + (several ? 'In each part\\'s sheet, press Estimate the error rate: "
        "one result per part.'\n"
        "    : 'Open the sheet, then press Estimate the error rate.');\n"
    )
    assert expr in page
    assert _all_answered(page, True) == ALL_ANSWERED_PARTS
    assert _all_answered(page, False) == ALL_ANSWERED_ONE
    for said in (ALL_ANSWERED_PARTS, ALL_ANSWERED_ONE):
        assert said.startswith(ALL_ANSWERED_HEAD), said
        assert sheet.JOB_LABELS["estimate"] in said, said
        assert _OFF_FACE.search(said) is None, said
    assert ALL_ANSWERED_PARTS.endswith(": one result per part.")
    code = _code(page)
    assert "in turn" not in code
    assert "then press" not in ALL_ANSWERED_PARTS
    # Which line shows is unchanged: more than one part has a sheet.
    fill = _between(page, "  function fillAsk() {", "\n  }\n")
    line = (
        "    ask.textContent = drawable().length ? ASK "
        ": done ? ALL_ANSWERED(drawn.length > 1) : YOUR_SQUARES;\n"
    )
    assert line in fill
    assert code.count("ALL_ANSWERED") == 2


# --------------------------------------------------------------------------- B5


def test_b5_the_preparing_card_says_it_fills_in_outside_its_status() -> None:
    """B5: under the preparing line (the status a screen reader hears change)
    the card draws one more line, outside the status, that it fills in by
    itself and the user can keep chatting; the status holds the mark and the
    words alone, so a new stage is heard without that line. It is #root's
    last block, right after the status."""
    page = _read("check.html")
    assert f"// WORDING: r5j B5\nconst PREP_FILLS = '{PREP_FILLS}';\n" in page
    card = _card(page)
    row = (
        "  const row = OE.el('div', { class: 'oe-preparing', "
        "id: 'oe-check-preparing', role: 'status' }, mark, words);\n"
    )
    fills = "  const fills = OE.el('div', { id: 'oe-check-fills' }, PREP_FILLS);\n"
    assert _in_order(
        card,
        "  const words = OE.el('span', {}, PREPARING(stageOf(data.stage)));\n",
        "  const mark = OE.el('span', { class: 'oe-prep-mark', "
        "'aria-hidden': 'true' });\n",
        row,
        fills,
        "  root.append(row, fills);\n",
        "  const resultId = ",
    )
    # The fills line has no role of its own and is no child of the status: the
    # status is built with the mark and the words alone, and nothing is put
    # into it later.
    assert "row.append(" not in card and "words.append(" not in card
    assert card.count("PREP_FILLS") == 1
    assert _code(page).count("PREP_FILLS") == 2
    assert "%" not in PREP_FILLS and _OFF_FACE.search(PREP_FILLS) is None
    assert "Claude" not in PREP_FILLS


def test_b5_the_fills_line_goes_with_no_id_and_once_too_long() -> None:
    """B5: a view that names no result (the card cannot ask) and a card that
    stopped waiting drop the fills line with the mark, before the words
    change: neither card fills in by itself. A card that fills in or fails
    is drawn anew by show, which empties #root."""
    page = _read("check.html")
    card = _card(page)
    no_id = (
        "  if (!resultId) {\n    mark.remove();\n    fills.remove();\n"
        "    words.textContent = PREP_NO_ID;\n    return;\n  }\n"
    )
    assert no_id in card
    gone = _between(card, "  const tooLong = () => {\n", "\n  };\n")
    assert _in_order(
        gone, "    mark.remove();\n    fills.remove();\n", "words.textContent = "
    )
    assert card.count("fills.remove();") == 2
    show = _between(page, "function show(app, params) {", "\n}\n")
    assert "  root.replaceChildren(OE.el('div', { class: 'overline' }, " in show


def test_b5_the_card_that_stopped_waiting_is_the_red_card() -> None:
    """B5: once the card stopped waiting its line is the red card of the
    failed card (class 'card error', as show draws it; 12q), still the status
    with its id; its words say, in CHECK_FAILED's middle sentence, that what
    was picked or saved before is kept, then to ask Claude to open the Check
    card again (once)."""
    page = _read("check.html")
    tags = "// WORDING: r5j B5\n// WORDING: r5h\nconst PREP_TOO_LONG = "
    assert tags in page
    assert _concat(_after(page, "const PREP_TOO_LONG = ", ";\n")) == PREP_TOO_LONG
    assert _concat(_after(page, "const CHECK_FAILED = ", ";\n")) == CHECK_FAILED
    assert CHECK_FAILED.split(". ")[1] + "." == KEPT
    assert PREP_TOO_LONG.split(". ")[1] + "." == KEPT
    assert PREP_TOO_LONG.count("Ask Claude to open the Check card again.") == 1
    assert _OFF_FACE.search(PREP_TOO_LONG) is None and "%" not in PREP_TOO_LONG
    card = _card(page)
    gone = _between(card, "  const tooLong = () => {\n", "\n  };\n")
    assert _in_order(
        gone,
        "    stop();\n",
        "    mark.remove();\n",
        "    row.className = 'card error';\n",
        "    words.textContent = PREP_TOO_LONG;\n",
        "    OE.tellClaude(app, PREP_TOLD_LONG(resultId));",
    )
    assert card.count("row.className = ") == 1
    assert "setAttribute('role'" not in card and "removeAttribute(" not in card
    show = _between(page, "function show(app, params) {", "\n}\n")
    assert "{ class: own ? 'card error stack' : 'card error' }," in show
    assert ".card.error { border-color: rgba(208,47,46,.5); }" in _read("common.js")


def test_b5_the_mark_sits_by_the_first_row_in_the_texts_colour() -> None:
    """B5: the status lays its mark and words from the top (a line that wraps
    at 380 px keeps the mark by its first row, 4 px down), and the mark's
    turning edge is in the text's colour on its pale ring; it still turns
    slowly and stands still for a reader who asks for less motion."""
    page = _read("check.html")
    style = _between(page, "<style>", "</style>")
    for rule in (
        "  .oe-preparing { display: flex; align-items: flex-start; gap: 8px; }\n",
        "  .oe-prep-mark { flex: none; width: 12px; height: 12px; "
        "border-radius: 50%; border: 2px solid var(--divider);\n"
        "    border-top-color: var(--text); margin-top: 4px; "
        "animation: oe-prep-turn 1.6s linear infinite; }\n",
        "  @media (prefers-reduced-motion: reduce) "
        "{ .oe-prep-mark { animation: none; } }\n",
    ):
        assert rule in style, rule
    assert ".oe-preparing { display: flex; align-items: center;" not in style
    assert "border-top-color: var(--text-2)" not in style


# --------------------------------------------------------------------------- B6


def test_b6_a_parts_heading_counts_its_squares_in_all() -> None:
    """B6: a part's heading says its count of squares "in all" ("West part
    (300 squares in all)", "(1 square in all)" for one), the name alone when
    the count is not known; the count buttons' words do not change (no other
    string says "in all")."""
    page = _read("check.html")
    tags = "// WORDING: r5j B6\n// WORDING: r4h A/F11\nconst PART_HEAD = (name, n) => "
    assert tags in page
    expr = (
        "const PART_HEAD = (name, n) => name + (Number.isInteger(n) && n >= 0\n"
        "  ? ' (' + n.toLocaleString('en-US') + (n === 1 ? ' square in all)' "
        ": ' squares in all)') : '');\n"
    )
    assert expr in page
    # Each case is built from the page's own pieces: the opening, the count
    # with a comma per thousand (toLocaleString('en-US'), pinned in expr), then
    # the ternary's branch for one square or for more.
    head = _after(page, "const PART_HEAD = (name, n) => ", ";\n")
    opening = _concat(_after(head, "&& n >= 0\n  ? ", " + n.toLocaleString('en-US')"))
    branches = _after(head, "(n === 1 ? ", ") : '')").split(" : ")
    one, many = (_concat(b) for b in branches)
    for name, n, said in HEADS:
        made = name
        if n is not None:
            made += opening + f"{n:,}" + (one if n == 1 else many)
        assert made == said, said
        assert _OFF_FACE.search(said) is None, said
    for branch in (one, many):
        assert _OFF_FACE.search(branch) is None, branch
    code = _code(page)
    assert "' squares)'" not in code and "' square)'" not in code
    assert code.count("in all") == 2
    assert code.count("PART_HEAD(") == 1


# --------------------------------------------------------------------------- B9


def test_b9_each_part_is_named_by_its_heading() -> None:
    """B9: on a map in parts each part's heading has an id of its own (its
    place among the card's zones, unique on the card) and its section is
    named by it (aria-labelledby), so each part is a named region; a map in
    one part has no heading and its section no name (OE.el leaves out an
    attribute whose value is null). The buttons keep their names."""
    page = _read("check.html")
    loop = _between(page, "\n  for (const m of models) {\n", "\n  }\n")
    heading = (
        "    m.heading = nParts > 1 ? OE.el('h3', { id: 'oe-part-' + m.index }, "
        "PART_HEAD(label(m), m.windows)) : null;\n"
    )
    section = (
        "    root.append(OE.el('section', {\n"
        "      class: nParts > 1 ? 'part card stack' : 'part stack',\n"
        "      'aria-labelledby': nParts > 1 ? 'oe-part-' + m.index : null,\n"
        "    }, m.heading, m.body));"
    )
    assert _in_order(loop, heading, "    m.body = OE.el('div', { class: 'stack' });\n")
    assert loop.endswith(section)
    code = _code(page)
    assert code.count("'oe-part-' + m.index") == 2
    assert code.count("aria-labelledby") == 1
    # The index is the part's place among the zones: one per part.
    assert "  const models = zones.map((z, i) => zoneModel(z, i));\n" in page
    assert "function zoneModel(z, index) {" in page
    make = _between(
        _read("common.js"), "OE.el = (tag, attrs = {}, ...children) => {", "\n};\n"
    )
    skip = (
        "    if (value === undefined || value === null || value === false) "
        "continue;\n"
    )
    assert skip in make


# --------------------------------------------------------------------------- marks


def test_every_changed_string_of_the_track_is_marked_in_the_page() -> None:
    """Each new or changed string of this round carries ``// WORDING: r5j``
    with its item, above its older marks: ALL_ANSWERED (B3), PREP_TOO_LONG
    and PREP_FILLS (B5), PART_HEAD (B6). B9 changes no string."""
    page = _read("check.html")
    marks = re.findall(r"// WORDING: r5j (B\d+)", page)
    for item, count in (("B3", 1), ("B5", 2), ("B6", 1), ("B9", 0)):
        assert marks.count(item) == count, item
    for name in ("ALL_ANSWERED", "PREP_TOO_LONG", "PREP_FILLS", "PART_HEAD"):
        marked = r"// WORDING: r5j [^\n]*\n(?://[^\n]*\n)*const " + name + r" = "
        assert re.search(marked, page) is not None, name
