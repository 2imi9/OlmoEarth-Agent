# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Round r5-g (18 Nov 2026), track RESULT: the result card's squares as a
tally (views/figures.js squares(), views/result.html), item G1 as
specs/r5-g-round.md gives it. The owner's note of 18 Nov 2026, on a result of
4,000 squares drawn one by one (a wall several screens tall): one square per
kind, a multiplication sign with how many of that kind, the kind where the
map said otherwise marked in red.

- G1: one row per kind of square that occurs, in the order the cells come
  (a class seen where the map agreed, the same class where the map said
  otherwise, Cannot tell last), each row the kind's own square (the mark the
  grid drew), "× N" with thousands commas, and the row's words in plain
  language; a kind with no square has no row; the rows' counts are the cells
  counted, so they add up to the squares checked; red only on the rows where
  the map said otherwise; every row one height and the counts one column;
  the grid, r5-f F06's bar and the legend are gone at every number of
  squares; the heading stays, and the figure has one name for a screen
  reader.

The views are read as text (no test here runs the scripts): the rows are
worked out here as figures.js makes them (:func:`_tally`), over the cells
server._squares sends, and the script is read for the steps that make them.
The browser harness (scripts/mcp_views_e2e.py: 17, 17b, 17c, 17d, 17i, 17m,
17o, 17p and 17v; 17h and 3j draw no squares) draws them and measures the
rows.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

import pytest

pytest.importorskip("mcp")

from olmoearth_agent.mcp_app import server, sheet, views  # noqa: E402

#: Words no face holds (the design rules' R7), and no choice is recommended.
_OFF_FACE = re.compile(
    r"\b(windows?|UTM|zones?|intervals?|strat(um|a)|design|budget|recommended)\b",
    re.I,
)

#: The fills figures.js gives the classes seen, in turn (FILLS).
FILLS = ("f0", "f1", "f2")

#: G1: a row's words where the map agreed and where it said otherwise, after
#: what the reviewer saw (the server's legend words, LEGEND_SAW).
AGREED = "; the map agreed"
OTHERWISE = "; the map said otherwise"

#: G1: the tally's rules: one column of rows of one height (grid-auto-rows:
#: 1fr makes every row the tallest's), each row the square's column (24 px:
#: the 14 px square and its outline, 5 px on each side), the counts' column
#: (as wide as the script sets it, the same on every row) and the words; the
#: square centred in its column; the × at the column's left and the count at
#: its right, its digits of one width; the words wrap inside their column.
TALLY_RULES = (
    ".oe-sq-tally { list-style: none; margin: 0; padding: 0; display: grid;"
    " grid-template-columns: minmax(0, 1fr);\n"
    "  grid-auto-rows: 1fr; row-gap: 6px; }\n"
    ".oe-sq-tally > li { display: grid;"
    " grid-template-columns: 24px var(--oe-tally-n, 7ch) minmax(0, 1fr);\n"
    "  column-gap: 10px; align-items: center; min-height: 24px; }\n"
    ".oe-sq-tally > li > .oe-sq { justify-self: center; }\n"
    ".oe-tally-n { display: flex; justify-content: space-between; gap: 6px;"
    " white-space: nowrap;\n"
    "  font-variant-numeric: tabular-nums; font-weight: 600; }\n"
    ".oe-tally-words { min-width: 0; overflow-wrap: anywhere; }\n"
)

#: G1: the squares' own rules, as the grid had them, without the legend's
#: swatches: the size, the three fills, Cannot tell dashed and empty, and the
#: red outline set off from the square (r4-h V2, r4-i O2, r5-b J04).
SQUARE_RULES = (
    ".oe-sq { display: inline-block; flex: none; width: 14px; height: 14px;"
    " border-radius: 2px;\n"
    "  box-shadow: inset 0 0 0 1px var(--divider); }\n"
    ".oe-sq.f0 { background: var(--text); }\n"
    ".oe-sq.f1 { background: var(--paper);"
    " box-shadow: inset 0 0 0 2px var(--text); }\n"
    ".oe-sq.f2 { background: rgba(16,82,87,.45);"
    " box-shadow: inset 0 0 0 1px var(--text); }\n"
    ".oe-sq.cannot { background: transparent; box-shadow: none;"
    " border: 1px dashed var(--text); box-sizing: border-box; }\n"
    ".oe-sq.wrong { outline: 3px solid var(--error); outline-offset: 2px; }\n"
)

#: What the grid, the bar (r5-f F06) and the legend were drawn with: none of
#: it stays in figures.js.
GONE = (
    ".oe-sq-grid",
    ".oe-sq-bar",
    ".oe-sq-legend",
    "oe-swatch",
    "BAR_FROM",
    "WRONG_KEY",
    "Outlined in red",
    "const entries",
    "entries.push(",
    "cells.map((c) => OE.el('span'",
    "runs.push(",
    "e.wrong === true && words(e.text)",
)

#: One row as worked out here: its kind, the square's classes, the count as
#: the row writes it and the row's words.
Row = tuple[str, str, str, str]


def _read(name: str) -> str:
    """A file of views/, as it is on disk."""
    return (views.VIEW_DIR / name).read_text(encoding="utf-8")


def _between(text: str, start: str, end: str) -> str:
    """The part of text from start up to the first end after it."""
    at = text.index(start)
    return text[at : text.index(end, at)]


def _figure_style() -> str:
    """figures.js's own style sheet (its STYLE), as written."""
    return _between(_read("figures.js"), "  const STYLE = `", "`;")


def _squares_script() -> str:
    """figures.js's squares, whole."""
    return _between(_read("figures.js"), "  const squares = (", "\n  };")


def _const(script: str, name: str) -> str:
    """The text of figures.js's ``const NAME = '...';``, one single-quoted
    string, its escapes read."""
    found = re.search(
        r"\n  const " + re.escape(name) + r" = '((?:[^'\\]|\\.)*)';\n", script
    )
    assert found is not None, name
    return re.sub(r"\\(.)", r"\1", found.group(1))


def _said() -> dict[str, str]:
    """figures.js's words for the rows: SAW, AGREED, OTHERWISE, CANNOT_KEY
    and TIMES, as written."""
    script = _read("figures.js")
    names = ("SAW", "AGREED", "OTHERWISE", "CANNOT_KEY", "TIMES")
    return {name: _const(script, name) for name in names}


def _key(value: Any) -> str:
    """A code as figures.js keys it (``String(v ?? '')``)."""
    return "" if value is None else str(value)


def _words(value: Any) -> str:
    """figures.js's words(): a string with words in it, trimmed, else ''."""
    return value.strip() if isinstance(value, str) and value.strip() else ""


def _wrong(cell: dict[str, Any]) -> bool:
    """figures.js's isWrong: the map said otherwise (true or 1)."""
    value = cell.get("wrong")
    return value is True or (value == 1 and not isinstance(value, bool))


def _kind(cell: dict[str, Any]) -> str:
    """A cell's kind as figures.js keys it: Cannot tell, else the class seen
    and whether the map said otherwise."""
    if cell.get("unjudged") is True:
        return "cannot"
    return "seen " + _key(cell.get("seen")) + (" wrong" if _wrong(cell) else "")


def _seen(
    code: str, classes: list[Any], legend: list[Any], said: dict[str, str]
) -> str:
    """What the reviewer saw of a class, as figures.js's seenOf says it: the
    legend's words for the class, else SAW and the class's name, else its
    code."""
    for entry in legend:
        named = not entry.get("wrong") and not entry.get("unjudged")
        if named and _key(entry.get("code")) == code and _words(entry.get("text")):
            return _words(entry.get("text"))
    for row in classes:
        if _key(row.get("code")) == code:
            return said["SAW"] + (_words(row.get("name")) or code)
    return said["SAW"] + code


def _row(
    cell: dict[str, Any],
    n: int,
    order: list[str],
    view: dict[str, Any],
    said: dict[str, str],
) -> Row:
    """One row as figures.js writes it for the kind of cell, n of them: the
    square's classes, the count with thousands commas (toLocaleString
    'en-US'), the words."""
    count = f"{n:,}"
    legend = [e for e in view.get("legend") or [] if isinstance(e, dict)]
    if cell.get("unjudged") is True:
        given = [
            e for e in legend if e.get("unjudged") is True and _words(e.get("text"))
        ]
        words = _words(given[0]["text"]) if given else said["CANNOT_KEY"]
        return ("cannot", "oe-sq cannot", count, words)
    code = _key(cell.get("seen"))
    fill = FILLS[min(order.index(code), len(FILLS) - 1)]
    wrong = _wrong(cell)
    classes = [c for c in view.get("classes") or [] if isinstance(c, dict)]
    words = _seen(code, classes, legend, said)
    words += said["OTHERWISE"] if wrong else said["AGREED"]
    mark = "oe-sq " + fill + (" wrong" if wrong else "")
    return (_kind(cell), mark, count, words)


def _tally(view: dict[str, Any]) -> list[Row]:
    """The tally's rows as figures.js's squares makes them from the server's
    view: one per kind of cell, in the order its first cell comes, each with
    the cells of that kind counted; the fills go in turn to the classes that
    have cells, in the classes' order, then any code seen they do not list."""
    said = _said()
    cells = [c for c in view["cells"] if isinstance(c, dict)]
    classes = [c for c in view.get("classes") or [] if isinstance(c, dict)]
    judged = [c for c in cells if c.get("unjudged") is not True]
    seen = Counter(_key(c.get("seen")) for c in judged)
    codes = [c.get("code") for c in classes] + [c.get("seen") for c in judged]
    order = [k for k in dict.fromkeys(map(_key, codes)) if seen[k]]
    kinds: dict[str, list[Any]] = {}
    for cell in cells:
        kinds.setdefault(_kind(cell), [cell, 0])[1] += 1
    return [_row(cell, n, order, view, said) for cell, n in kinds.values()]


def _graded(groups: list[tuple[str, str, int, int]], cannot: int = 0) -> dict[str, Any]:
    """A seen sheet's counts as sheet.graded_squares writes them: for each
    class (code, name, right, wrong) its right cells, then its wrong ones,
    then one cell per Cannot tell square; every class listed, seen or not."""
    cells: list[dict[str, Any]] = []
    for code, _name, right, wrong in groups:
        cells += [{"seen": code, "wrong": False}] * right
        cells += [{"seen": code, "wrong": True}] * wrong
    cells += [{"seen": None, "wrong": False, "unjudged": True}] * cannot
    return {
        "n": len(cells),
        "n_unjudged": cannot,
        "face": "seen",
        "cells": cells,
        "classes": [
            {"code": code, "name": name, "seen": right + wrong, "agreed": right}
            for code, name, right, wrong in groups
        ],
    }


#: The harness's result cards, as the server's counts: 17 (two classes, one
#: rare), 17b (every square right, a class nobody saw), 17c (two Cannot
#: tell), 17i (four classes, two seen), 17o (three classes and one Cannot
#: tell, a class nobody saw), 17p (three classes, two rare) and 17v (4,000
#: squares, 1,200 Cannot tell).
CARDS = {
    "17": _graded([("0", "absent", 70, 7), ("1", "present", 0, 3)]),
    "17b": _graded([("0", "absent", 80, 0), ("1", "present", 0, 0)]),
    "17c": _graded([("0", "absent", 70, 5), ("1", "present", 0, 3)], cannot=2),
    "17i": _graded(
        [
            ("0", "Tree cover", 0, 0),
            ("1", "Shrubland", 0, 0),
            ("2", "Grassland", 30, 4),
            ("3", "Cropland", 40, 6),
        ]
    ),
    "17o": _graded(
        [
            ("0", "Water", 18, 2),
            ("1", "Trees", 10, 2),
            ("2", "Crops", 6, 1),
            ("3", "Snow", 0, 0),
        ],
        cannot=1,
    ),
    "17p": _graded([("0", "Water", 70, 5), ("1", "Kelp", 2, 1), ("2", "Rock", 0, 2)]),
    "17v": _graded([("0", "absent", 2400, 300), ("1", "present", 0, 100)], cannot=1200),
}


def _view(card: str) -> dict[str, Any]:
    """server._squares' view of a card's counts (no estimate rows)."""
    view = server._squares(CARDS[card], {})
    assert view is not None, card
    return view


# --------------------------------------------------------------------------- G1


def test_g1_the_4000_squares_are_four_rows() -> None:
    """G1: the card of 4,000 squares (2,400 seen absent where the map agreed,
    300 absent where it said otherwise, 100 present where it said otherwise,
    1,200 Cannot tell) is four rows, in the cells' order, each the kind's own
    square, its count with a thousands comma and its plain words; the counts
    add up to the 4,000 checked. r5-f F06 drew it as a bar, the grid before
    it as 4,000 marks."""
    rows = _tally(_view("17v"))
    assert [r[1:] for r in rows] == [
        ("oe-sq f0", "2,400", 'You saw "absent"' + AGREED),
        ("oe-sq f0 wrong", "300", 'You saw "absent"' + OTHERWISE),
        ("oe-sq f1 wrong", "100", 'You saw "present"' + OTHERWISE),
        ("oe-sq cannot", "1,200", "Cannot tell"),
    ]
    total = sum(int(r[2].replace(",", "")) for r in rows)
    assert total == CARDS["17v"]["n"] == 4000


def test_g1_the_80_squares_of_the_first_card() -> None:
    """G1: the first result card (80 squares: 70 seen absent where the map
    agreed, 7 where it said otherwise, 3 present, all where it said
    otherwise) is three rows; the rare class has its row, the outline needs
    no legend entry (each otherwise row says it itself: squares() reads the
    legend only for a class's words and for Cannot tell's), and the words
    name the class as the server's legend does ('You saw "absent"')."""
    rows = _tally(_view("17"))
    assert [r[1:] for r in rows] == [
        ("oe-sq f0", "70", 'You saw "absent"' + AGREED),
        ("oe-sq f0 wrong", "7", 'You saw "absent"' + OTHERWISE),
        ("oe-sq f1 wrong", "3", 'You saw "present"' + OTHERWISE),
    ]
    squares = _squares_script()
    assert (
        "legend.find((e) => !e.wrong && !e.unjudged && key(e.code) === code)" in squares
    )
    assert "legend.find((e) => e.unjudged === true && words(e.text))" in squares
    assert squares.count("legend.find(") == 2


@pytest.mark.parametrize("card", sorted(CARDS))
def test_g1_the_counts_are_the_cells_counted(card: str) -> None:
    """G1, no number changes: every row's count is the cells of its kind
    (for each class, where the map agreed its agreed squares, where it said
    otherwise its seen less agreed; Cannot tell the counts' n_unjudged), the
    counts add up to the squares checked, a kind with no square has no row
    (a class nobody saw has none, nor one the map was always right or always
    wrong on), and no two rows are one kind."""
    graded = CARDS[card]
    rows = _tally(_view(card))
    counts = {r[0]: int(r[2].replace(",", "")) for r in rows}
    assert len(counts) == len(rows)
    assert sum(counts.values()) == graded["n"] == len(graded["cells"])
    assert all(n > 0 for n in counts.values())
    want: dict[str, int] = {}
    for row in graded["classes"]:
        right, wrong = row["agreed"], row["seen"] - row["agreed"]
        if right:
            want["seen " + row["code"]] = right
        if wrong:
            want["seen " + row["code"] + " wrong"] = wrong
    if graded["n_unjudged"]:
        want["cannot"] = graded["n_unjudged"]
    assert counts == want
    # The rows come in the cells' order: class by class, agreed before
    # otherwise, Cannot tell last.
    assert list(counts) == list(want)
    assert len(rows) <= 2 * server.SQUARES_MAX_CLASSES + 1


def test_g1_the_rows_words_and_fills_on_a_map_of_three_classes() -> None:
    """G1: on a class sheet of three classes seen and one Cannot tell (17o),
    each class has its fill (f0, f1, f2, in the classes' order) on both of
    its rows, the class nobody saw (Snow) has none, and Cannot tell is the
    last row, in the sheet's own word; on a map of four classes where only
    two were seen (17i), the two take f0 and f1."""
    rows = _tally(_view("17o"))
    assert [r[1:] for r in rows] == [
        ("oe-sq f0", "18", 'You saw "Water"' + AGREED),
        ("oe-sq f0 wrong", "2", 'You saw "Water"' + OTHERWISE),
        ("oe-sq f1", "10", 'You saw "Trees"' + AGREED),
        ("oe-sq f1 wrong", "2", 'You saw "Trees"' + OTHERWISE),
        ("oe-sq f2", "6", 'You saw "Crops"' + AGREED),
        ("oe-sq f2 wrong", "1", 'You saw "Crops"' + OTHERWISE),
        ("oe-sq cannot", "1", sheet.CANNOT_TELL),
    ]
    four = _tally(_view("17i"))
    assert [r[1] for r in four] == [
        "oe-sq f0",
        "oe-sq f0 wrong",
        "oe-sq f1",
        "oe-sq f1 wrong",
    ]
    assert not any("Tree cover" in r[3] or "Shrubland" in r[3] for r in four)
    for card in CARDS:
        for row in _tally(_view(card)):
            assert _OFF_FACE.search(row[3]) is None, row


def test_g1_red_only_on_the_rows_where_the_map_said_otherwise() -> None:
    """G1: the red is the outline of a row's square where the map said
    otherwise (the error colour, as the grid's outline), the only red of the
    figure: one rule of the figures' style sheet names --error, the square's
    wrong rule; a row's square takes wrong exactly when its words say the
    map said otherwise (one test, wrong, makes both); nothing else in the
    squares is red. The red rows as drawn are measured in the harness
    (tally_check), not here."""
    style = _figure_style()
    assert style.count("var(--error)") == 1
    wrong = ".oe-sq.wrong { outline: 3px solid var(--error); outline-offset: 2px; }"
    assert wrong in style
    assert "#d02f2e" not in style.lower() and re.search(r"\bred\b", style) is None
    squares = _squares_script()
    assert "error" not in squares
    for text in (
        "const wrong = !isCannot(cell) && isWrong(cell);",
        "mark: isCannot(cell) ? 'oe-sq cannot' : 'oe-sq ' + fill(cell.seen)"
        " + (wrong ? ' wrong' : ''),",
        ": seenOf(key(cell.seen)) + (wrong ? OTHERWISE : AGREED),",
        "rows.map((r) => OE.el('li', {},",
    ):
        assert text in squares, text
    assert "OE.el('li', { class:" not in squares


def test_g1_the_tally_is_one_row_per_kind_and_no_mark_per_square() -> None:
    """G1: squares() keys each cell by its kind (Cannot tell; else the class
    seen and whether the map said otherwise), counts each kind in the order
    its first cell comes (a Map keeps that order), and draws one li per
    kind: the kind's square, the count's cell ("×" and the count, with
    thousands commas) and the words (with a place to break after an
    underscore, OE.softBreaks). The figure is named once for a screen
    reader, each row's words with its count; no grid, no bar, no legend and
    no span per square stay."""
    squares = _squares_script()
    for text in (
        "const kinds = new Map();",
        "for (const c of cells) {",
        "const k = isCannot(c) ? 'cannot' : 'seen ' + key(c.seen)"
        " + (isWrong(c) ? ' wrong' : '');",
        "if (kinds.has(k)) kinds.get(k).n += 1;",
        "else kinds.set(k, { cell: c, n: 1 });",
        "const rows = [...kinds.values()].map(({ cell, n }) => {",
        "n: n.toLocaleString('en-US'),",
        "text: isCannot(cell) ? (cannot ? words(cannot.text) : CANNOT_KEY)",
        "return SAW + ((c && words(c.name)) || code);",
        "OE.el('span', { class: r.mark }),",
        "OE.el('span', { class: 'oe-tally-n' }, OE.el('span', {}, TIMES),"
        " OE.el('span', {}, r.n)),",
        "OE.el('span', { class: 'oe-tally-words' },",
        "typeof OE.softBreaks === 'function' ? OE.softBreaks(r.text) : r.text",
        "class: 'oe-fig oe-squares', role: 'img', 'aria-label':"
        " rows.map((r) => r.text + ' (' + r.n + ')').join('. '),",
        "}, tally);",
    ):
        assert text in squares, text
    assert squares.count("'aria-label':") == 1
    assert squares.count("OE.el('li'") == 1
    script = _read("figures.js")
    for gone in GONE:
        assert gone not in script, gone
    assert "  return { STYLE, style, steps, rateLine, parts, squares };\n" in script


def test_g1_rows_of_one_height_and_the_counts_in_one_column() -> None:
    """G1 (R1, R2): the rows are one column of one height (grid-auto-rows:
    1fr), each with the square's column, the counts' column and the words;
    the counts' column is as wide as the longest count with its ×, set once
    on the list (--oe-tally-n), so it is the same on every row, the × at its
    left and the count at its right; the words wrap inside their column, so
    nothing runs out of the figure at 380 px; the list is in the figures'
    small type, as the parts' labels under it. The squares keep their size,
    fills, dashed Cannot tell and red outline, without the legend's
    swatches."""
    style = _figure_style()
    assert TALLY_RULES in style
    assert SQUARE_RULES in style
    assert style.index(SQUARE_RULES) < style.index(TALLY_RULES)
    squares = _squares_script()
    widest = "const widest = Math.max(...rows.map((r) => r.n.length)) + TIMES.length;"
    assert widest in squares
    column = "style: '--oe-tally-n: calc(' + widest + 'ch + 6px)' },"
    tally = "const tally = OE.el('ul', { class: 'oe-sq-tally small', "
    assert tally + column in squares


def test_g1_the_rows_words_are_marked() -> None:
    """G1: the row's words are marked // WORDING: r5g (where the map agreed,
    where it said otherwise, the ×); SAW, the fallback words of a class with
    no legend entry, is marked r5g too and copies the server's LEGEND_SAW
    (r4-p2); Cannot tell is the sheet's own word (r4-i O2)."""
    script = _read("figures.js")
    for text in (
        "  // WORDING: r5g\n  const AGREED = '" + AGREED + "';\n",
        "  // WORDING: r5g\n  const OTHERWISE = '" + OTHERWISE + "';\n",
        "  // WORDING: r5g\n  const TIMES = '× ';\n",
        "  // WORDING: r5g (the fallback when a class has no legend entry;"
        " the server's LEGEND_SAW, r4p2)\n  const SAW = 'You saw ';\n",
        "  // WORDING: r4i O2\n  const CANNOT_KEY = 'Cannot tell';\n",
    ):
        assert text in script, text
    said = _said()
    assert said["SAW"] + "{name}" == server.LEGEND_SAW
    assert said["CANNOT_KEY"] == sheet.CANNOT_TELL == server.LEGEND_CANNOT
    assert said["AGREED"] == AGREED and said["OTHERWISE"] == OTHERWISE
    assert said["TIMES"] == "× "


def test_g1_the_card_keeps_its_heading_over_the_squares() -> None:
    """G1: the card's heading stays ("The N squares you checked", over the
    tally); the card draws the figure as before (OE.fig.squares) and has no
    legend of its own; its comment says the squares are a tally. Since round
    r5-s W2 no line follows the tally (with Cannot tell it was the server's
    no-split line)."""
    page = _read("result.html")
    card = _between(page, "function resultCard(", "\n}\n")
    said = (
        "OE.el('h3', {}, RESULT_WORDS.squaresHead(n !== null ? n : sq.cells.length,"
        " unit)), OE.fig.squares(sq)));\n"
    )
    assert said in card
    assert "no_split" not in page
    assert "oe-sq" not in page
    start = "/* The estimate's result card (data.result"
    head = " ".join(_between(page, start, "function resultCard(").split())
    assert "the squares checked as a tally (r5-g G1" in head


def test_g1_the_figures_head_comment_says_tally() -> None:
    """G1: the figures' head comment describes the tally, no longer the grid
    with its legend or the bar above 200 squares."""
    script = _read("figures.js")
    head = " ".join(script[: script.index("OE.fig = (() => {")].split())
    assert "squares({cells, classes, legend}): the squares checked as a tally" in head
    assert "(r5-g G1, the owner's note of 18 Nov 2026" in head
    for gone in ("a legend of at most five entries", "are drawn as one bar"):
        assert gone not in head, gone
