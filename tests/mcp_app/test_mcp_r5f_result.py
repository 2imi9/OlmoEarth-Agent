# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Round r5-f (18 Nov 2026), track RESULT: what the audit after the design
round found on the result cards (views/figures.js, views/result.html and the
result words in server.py), each item as specs/r5-e-audit/build_list.md gives
it. Layout and words only: no number, range or digit changes. The server's
words through its own names; the views are read as text (no test here runs
the scripts):

- F05: in the result card's parts figure a part is never narrower than its
  label on one line (at its longest word, a part at the 12% floor wrapped its
  label to four lines, and the row's one height made the part beside it a
  tall dark slab); the parts that do not fit one row go on to the next, never
  wider than their shares there (a part alone on a later row grew over the
  whole row); stacked, at 560 px or less, a label may wrap again and a part
  is full width;
- F06: more than 200 squares checked are drawn as one bar of runs, a run for
  each stretch of squares that look the same, as wide as its count's share,
  under the same legend and with the same name for a screen reader (4,000
  marks were about three screens of squares); a run where the map said
  otherwise is never under 12 px, and the second class keeps its dark ring
  inside the red (else that run was the legend's outline sample); up to 200
  the grid is as before. Round r5-g, G1 replaced the bar and the grid with a
  tally, one row per kind of square (tests/mcp_app/test_mcp_r5g_result.py):
  here only that the bar is gone and that the server's cells still come
  grouped, which is the tally's order;
- F07: under the parts' heading the face's line says
  where the other part is checked: "Check the other part too. Ask Claude to
  open the Check card again."; the model's lines keep their words;
- F10: a plan card's lead says that the squares were picked at random and
  are now fixed, and what answering every one gives; with track SHEET's F02
  (after review) these are its two lines, and the second goes once the sheet
  is open in the card, whose footer says it.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("mcp")

from olmoearth_agent.mcp_app import auth, server, sheet, views  # noqa: E402

#: Words no face holds (the design rules' R7), and no choice is recommended.
_OFF_FACE = re.compile(
    r"\b(windows?|UTM|zones?|intervals?|strat(um|a)|design|budget|recommended)\b",
    re.I,
)

#: One token of a JS concatenation: a single-quoted string, a plus, or a name
#: (a call with no parentheses inside its own is one name).
_TOKEN = re.compile(
    r"\s*(?:'((?:[^'\\]|\\.)*)'|(\+)|([A-Za-z_][A-Za-z0-9_.]*(?:\([^()]*\))?))"
)

#: F05: the row of parts, which wraps; a part's least width, its label on one
#: line, and its most, its share (the --oe-share its own style gives, none
#: without it); the label's own rule, as it was; and the stacked rule, whose
#: second rule lets a stacked part be narrower than its label again and full
#: width whatever its share.
PARTS_ROW = (
    ".oe-parts { display: flex; flex-wrap: wrap; gap: 4px; align-items: stretch; }\n"
)
PART_LEAST = (
    ".oe-part { min-width: max-content; max-width: var(--oe-share, none);"
    " display: flex; flex-direction: column; gap: 4px; }\n"
)
PART_LABEL = ".oe-part-label { color: var(--text); overflow-wrap: break-word; }\n"
PARTS_STACKED = (
    "@media (max-width: 560px) { .oe-parts { display: grid;"
    " grid-template-columns: minmax(0, 1fr); grid-auto-rows: 1fr; }\n"
    "  .oe-parts > .oe-part { min-width: 0; max-width: none; } }\n"
)

#: F06: the bar's rules and the legend's, and the code that drew the bar;
#: since r5-g G1 none of them is in figures.js (a tally, one row per kind).
BAR_GONE = (
    ".oe-sq-bar",
    ".oe-sq-legend",
    ".oe-sq-grid",
    "const BAR_FROM",
    "runs.push(",
    "'flex: ' + r.n",
)

#: F07: the sentence the face adds, and its line for one other part and for
#: three.
ASK_CHECK = "Ask Claude to open the Check card again."
OTHER_ONE = "Check the other part too. Ask Claude to open the Check card again."
OTHER_THREE = "Check the other 3 parts too. Ask Claude to open the Check card again."
#: Round r5-s W8: the Details line on the parts, in words that describe.
SIDE_BY_SIDE = (
    "Each part has its own result: this check gives no range for a combined rate."
)

#: F10: the plan's two sentences, of squares and of points, and the words it
#: had; the second is the lead's "next" line since track SHEET's F02 (after
#: review).
PLAN_NEXT = "Answer every one on the sheet to get the result."
PLAN_SQUARES = (
    "The squares to check were picked at random and are now fixed. Answer every one "
    "on the sheet to get the result."
)
PLAN_POINTS = (
    "The points to check were picked at random and are now fixed. Answer every one "
    "on the sheet to get the result."
)
PLAN_BEFORE = "picked before any answer"


@pytest.fixture(autouse=True)
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The output root, the read roots' own, and no key."""
    out = tmp_path / "outputs"
    out.mkdir()
    monkeypatch.setenv("OLMOEARTH_OUTPUT_ROOT", str(out))
    monkeypatch.delenv("OLMOEARTH_SCORES_ROOT", raising=False)
    monkeypatch.delenv(auth.KEY_ENV, raising=False)
    monkeypatch.setenv(auth.KEY_FILE_ENV, str(tmp_path / "cfg" / "studio.env"))
    return out.resolve()


def _read(name: str) -> str:
    """A file of views/, as it is on disk."""
    return (views.VIEW_DIR / name).read_text(encoding="utf-8")


def _server_source() -> str:
    """server.py, as it is on disk."""
    return Path(server.__file__).read_text(encoding="utf-8")


def _code(text: str) -> str:
    """A view's text without its comments (``/* */`` and ``//`` to the end of
    the line), which may quote words a face no longer says."""
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


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
    ends its last line."""
    at = block.index(start) + len(start)
    end = block.find(",\n", at)
    return (block[at:] if end < 0 else block[at:end]).strip().rstrip(",")


def _concat(expr: str, names: dict[str, str]) -> str:
    """The text a JS concatenation of single-quoted strings and names makes,
    each name given its value in names; anything else fails."""
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


def _figure_style() -> str:
    """figures.js's own style sheet (its STYLE), as written."""
    return _between(_read("figures.js"), "  const STYLE = `", "`;")


def _squares_script() -> str:
    """figures.js's squares, whole."""
    return _between(_read("figures.js"), "  const squares = (", "\n  };")


def _runs(cells: list[dict[str, Any]]) -> list[tuple[str, int]]:
    """The bar's runs as figures.js makes them, over the server's cells in
    their order: a new run wherever the mark changes (Cannot tell; else the
    class seen and whether the map said otherwise), each with its count."""
    runs: list[tuple[str, int]] = []
    for cell in cells:
        if cell.get("unjudged") is True:
            mark = "cannot"
        else:
            mark = str(cell.get("seen")) + (" wrong" if cell.get("wrong") else "")
        if runs and runs[-1][0] == mark:
            runs[-1] = (mark, runs[-1][1] + 1)
        else:
            runs.append((mark, 1))
    return runs


# --------------------------------------------------------------------------- F05


def test_f05_a_part_is_never_narrower_than_its_label_on_one_line() -> None:
    """F05: the row of parts wraps, and a part's least width is its label on
    one line (max-content), so a small part's label is one line and the row's
    one height is one line's; parts whose labels do not fit one row go on to
    the next. Above its label's width a part is never wider than its share
    (its own --oe-share; with the wrap a part alone on a later row grew over
    the whole row, wider than the parts of larger shares), so a short last
    row stays short, at the left. At 560 px or less the parts are stacked as
    before, and there a part may be narrower than its label again and is
    full width whatever its share (its rule names the row, so it outweighs
    both wherever it stands), so a long label wraps inside the panel. The
    longest-word width (min-content) is gone."""
    style = _figure_style()
    assert PARTS_ROW + PART_LEAST in style
    assert PART_LABEL + PARTS_STACKED in style
    assert style.index(PART_LEAST) < style.index(PARTS_STACKED)
    for gone in (
        "min-content",
        ".oe-parts { display: flex; gap: 4px; align-items: stretch; }",
        "\n.oe-part { min-width: 0;",
    ):
        assert gone not in style, gone


def test_f05_no_number_word_or_fill_of_the_parts_changed() -> None:
    """F05 is layout only: the 12% floor and the widths by share are computed
    as before, and a part's share is both its grow factor and, as a percent,
    its --oe-share (its most width); each part's words are its label inside
    its own box, the figure's name for a screen reader is every label, "this
    result" keeps its dark fill (r4-h: a fill means a result) and the
    server's labels their words."""
    script = _read("figures.js")
    assert "  const FLOOR = 0.12;\n" in script
    parts = _between(script, "  const parts = (", "\n  };")
    for text in (
        "const w = widths(items.map(shareOf));",
        "const share = round2(w[i] * 100);",
        "style: 'flex: ' + share + ' 1 0%; --oe-share: ' + share + '%' },",
        "OE.el('span', { class: 'oe-part-label small' }, labels[i])));",
        "'aria-label': labels.join('; ') }, pieces);",
    ):
        assert text in parts, text
    filled = ".oe-part.this .oe-part-box { background: var(--text);"
    assert filled + " border-color: var(--text); }\n" in _figure_style()
    assert server.PART_THIS == "{part}: this result"
    assert server.PART_NOT == "{part}: not checked yet"


# --------------------------------------------------------------------------- F06


def test_f06_the_bar_is_gone_with_the_grid() -> None:
    """F06 drew more than 200 squares as one bar of runs, and up to 200 the
    grid of one mark per square. Round r5-g, G1 (the owner's note of 18 Nov
    2026) replaced both with a tally, one row per kind of square, at every
    number of squares: no bar, no grid, no legend and none of their rules
    stay, the squares keep their size, and the second class its 2 px dark
    ring (tests/mcp_app/test_mcp_r5g_result.py)."""
    squares = _squares_script()
    style = _figure_style()
    for gone in BAR_GONE:
        assert gone not in squares, gone
        assert gone not in style, gone
    assert "const kinds = new Map();" in squares
    ring = ".oe-sq.f1 { background: var(--paper);"
    assert ring + " box-shadow: inset 0 0 0 2px var(--text); }\n" in style
    assert "width: 14px; height: 14px; border-radius: 2px;" in style


def test_f06_the_servers_cells_come_in_runs() -> None:
    """F06: server._squares passes the graded cells on in the order it gets
    them. For the audit's 4,000-square card as sheet.graded_squares writes it
    (grouped class by class, right before wrong, Cannot tell last: 2,400 seen
    absent and right, 300 absent and wrong, 100 present and wrong, 1,200
    Cannot tell) that is four runs whose counts add up to every square, no
    more than two for each class drawn and one for Cannot tell on this
    input; since r5-g G1 these are the tally's four rows, in this order
    (tests/mcp_app/test_mcp_r5g_result.py draws them from the same
    fixture). graded_squares' own grouping, from answers given out of order, is
    pinned in tests/mcp_app/test_mcp_r4i_sheet.py, not here: by
    test_a_seen_sheets_cannot_tell_squares_come_after_the_judged_ones and
    test_a_class_sheets_cannot_tell_squares_come_after_the_judged_ones."""
    graded = {
        "n": 4000,
        "n_unjudged": 1200,
        "face": "seen",
        "cells": [{"seen": "0", "wrong": False}] * 2400
        + [{"seen": "0", "wrong": True}] * 300
        + [{"seen": "1", "wrong": True}] * 100
        + [{"seen": None, "wrong": False, "unjudged": True}] * 1200,
        "classes": [
            {"code": "0", "name": "Absent", "seen": 2700, "agreed": 2400},
            {"code": "1", "name": "Present", "seen": 100, "agreed": 0},
        ],
    }
    view = server._squares(graded, {})
    assert view is not None
    want = [
        ("0", 2400),
        ("0 wrong", 300),
        ("1 wrong", 100),
        ("cannot", 1200),
    ]
    runs = _runs(view["cells"])
    assert runs == want
    assert sum(n for _, n in runs) == len(view["cells"]) == 4000
    assert len(runs) <= 2 * server.SQUARES_MAX_CLASSES + 1


# --------------------------------------------------------------------------- F07


def test_f07_the_other_part_line_says_where_to_check_it() -> None:
    """F07: under the parts' heading the face's line is
    the first sentence of the server's SAY_OTHER (SAY_OTHERS for k parts) and
    then, in a sentence of its own, where that check is started: "Ask Claude
    to open the Check card again.", words the Check card itself uses, since no
    button of the result card checks a part; marked; drawn once, as before;
    no word kept off faces and no digit of its own."""
    page = _read("result.html")
    words = _between(page, "const RESULT_WORDS = {", "\n};")
    marks = "  // WORDING: r5f F07\n  // WORDING: r5e R4\n  // WORDING: r4m M2\n"
    assert marks + "  otherPart: (k) => (k === 1\n" in words
    after = _after(words, "otherPart: (k) => (k === 1\n    ? ", "),\n")
    one, many = after.split("\n    : ")
    assert _concat(one, {}) == OTHER_ONE
    assert _concat(many, {"k": "3"}) == OTHER_THREE
    for line, first in (
        (OTHER_ONE, server.SAY_OTHER),
        (OTHER_THREE, server.SAY_OTHERS.format(k="3")),
    ):
        assert line.endswith(" " + ASK_CHECK)
        assert first.strip().startswith(line.removesuffix(" " + ASK_CHECK) + " ")
        assert _OFF_FACE.search(line) is None, line
    assert not any(ch.isdigit() for ch in OTHER_ONE + _concat(many, {"k": ""}))
    card = _between(page, "function resultCard(app, data) {", "\n}\n")
    face = "other ? OE.el('p', { class: 'small' }, RESULT_WORDS.otherPart(other))"
    assert face + " : null" in card
    assert _code(page).count(ASK_CHECK) == 2
    # The Check card says the same sentence for a part not picked (its LATER).
    later = _concat(_after(_read("check.html"), "const LATER = ", ";\n"), {})
    assert later.lower().endswith(" " + ASK_CHECK.lower())


def test_f07_the_models_lines_keep_their_words() -> None:
    """F07 changes the card's face only: the model's lines (SAY_OTHER,
    SAY_OTHERS) are as r4-m M2 wrote them and do not name the Check card, and
    Details' sentence on the parts was the server's word for word (words that
    describe since round r5-s W8; the server's sentence the model's alone)."""
    assert server.SAY_OTHER == (
        " Check the other part too. Report the parts' results side by side; do not "
        "average or add them: this check gives no range for a combined rate."
    )
    assert server.SAY_OTHERS.format(k="2") == (
        " Check the other 2 parts too. Report the parts' results side by side; do "
        "not average or add them: this check gives no range for a combined rate."
    )
    for said in (server.SAY_OTHER, server.SAY_OTHERS, server.SAY_SIDE_BY_SIDE):
        assert "Check card" not in said
    words = _between(_read("result.html"), "const RESULT_WORDS = {", "\n};")
    side = _concat(_entry(words, "sideBySide: "), {})
    assert side == SIDE_BY_SIDE != server.SAY_SIDE_BY_SIDE.strip()
    assert "side by side" not in side and "do not" not in side


# --------------------------------------------------------------------------- F10


def test_f10_the_plan_lead_says_the_pick_is_random_and_fixed() -> None:
    """F10: a plan card's lead is two short sentences: the squares were
    picked at random and are now fixed, and answering every one on the sheet
    gives the result ("picked before any answer" did not say whose answer);
    "points" for a plan of points; with track SHEET's F02 (after review) the
    first is its "what" line and the second its "next" (the card takes that
    one off once its sheet is open there, whose footer says it), no
    "cannot"; marked; they hold no digit, no button's words and no word kept
    off faces."""
    for line, lead in (
        (PLAN_SQUARES, server.PLAN_LEAD),
        (PLAN_POINTS, server.PLAN_LEAD_POINTS),
    ):
        what = line.removesuffix(" " + PLAN_NEXT)
        assert lead == {"what": what, "next": PLAN_NEXT, "cannot": None}
    assert PLAN_POINTS == PLAN_SQUARES.replace("squares", "points")
    for line in (PLAN_SQUARES, PLAN_POINTS):
        assert line.count(". ") == 1 and line.endswith(".")
        assert PLAN_BEFORE not in line
        assert sheet.OPEN_FIRST not in line
        assert not any(ch.isdigit() for ch in line)
        assert _OFF_FACE.search(line) is None, line
    source = _server_source()
    mark = "    # WORDING: r5f F10 F02\n"
    assert source.count(mark + '    "what": "The ') == 2
    assert source.count(mark + f'    "next": "{PLAN_NEXT}",\n') == 2
    assert "# WORDING: squares; r5e R2\nPLAN_LEAD: dict[str, str | None] = {" in source


def test_f10_a_plan_with_a_sheet_carries_the_line() -> None:
    """F10: the plan tool's lead is that line beside its sheet, in squares or,
    for a plan of points, in points; a plan without a sheet has no lead."""
    assert server.LEAD["olmoearth_plan_label_sample"] == server.PLAN_LEAD
    lead = server._lead("olmoearth_plan_label_sample", {}, {"labels_csv_path": "x"})
    assert lead == server.PLAN_LEAD
    points = server._lead(
        "olmoearth_plan_label_sample", {}, {"labels_csv_path": "x", "unit": "point"}
    )
    assert points == server.PLAN_LEAD_POINTS
    none = server._lead("olmoearth_plan_label_sample", {}, {})
    assert none == {"what": None, "next": None, "cannot": None}
