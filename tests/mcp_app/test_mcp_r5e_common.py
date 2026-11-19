# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Round r5-e (17 Nov 2026), track COMMON: the style and helpers every view
shares (views/common.js), each item as specs/r5-e-design/build_list.md gives
it. The files are read as text (no test here runs the scripts):

- C1: controls of one kind side by side have one size: a square's answers
  are equal columns of one grid, every row as tall as the tallest (a tick no
  longer widens a chosen answer); a joined toggle and the face's Previous
  and Next are each as wide as the widest member; "Other class..." on the
  face is an answer cell with a dashed edge; after review: a list row's
  answers keep their words' widths beside the open pictures pane and stand
  one under the other on a panel under 480 px, a class name of one long word
  breaks inside its cell, and where one column fits the list of more classes
  takes one cell;
- C2: a card's lead (OE.lead) draws at most two lines, all in the body
  colour (no grey third line), and no card sends more than two;
- C3: the default sign-in note says where the key goes in the words every
  other sign-in text uses (OE.SIGN_IN_KEY_GOES), no longer that Studio
  checks the key;
- C4: everything about one square sits in the face's own box: a hairline
  edge with 12 px inside, the focus ring 2 px outside it; after review: the
  rule's box above it has the same 12 px, and on a panel of 420 px or less
  both keep 6 px at their sides.
"""

from __future__ import annotations

import re

import pytest

from olmoearth_agent.mcp_app import views

#: C1: the rule for every group of same-kind controls, after .row and .card.
SAME = (
    "/* r5-e: one rule for every group of same-kind controls: equal columns, every"
    " row as tall as the tallest. */\n"
    ".oe-same, .face .answers { display: grid; grid-template-columns:"
    " repeat(auto-fill, minmax(var(--same-min, 136px), 1fr)); grid-auto-rows: 1fr;"
    " gap: 8px; align-items: stretch; }\n"
    ".oe-same > *, .face .answers > * { min-width: 0; }\n"
)

#: C1: a joined group (a toggle) or a pair (Previous and Next on the face).
JOINED = (
    "/* r5-e: a joined group or pair: every member as wide as the widest. */\n"
    ".toggle, .face-top > .row { display: inline-grid; grid-auto-flow: column;"
    " grid-auto-columns: 1fr; }\n"
)

#: C1: the face's answer buttons, the list beside them, and "Other class...".
ANSWERS = (
    ".face .answers > select { grid-column: span 2; }\n"
    ".face .answers button { min-width: 0; padding: 10px 12px; }\n"
    ".face .answers button.oe-other { border-style: dashed; color: var(--text-2); }\n"
)

#: C1 (after review): beside the open pictures pane a list row's answers keep
#: their words' widths, so the row stays one line (H4); the last rule of the
#: pane's media block.
ROW_BESIDE_PANE = "  .sheet-work.pane td > .toggle { grid-auto-columns: auto; }"

#: C1 (after review): on a panel under 480 px a list row's answers stand one
#: under the other, each as wide as the widest.
ROW_NARROW = (
    "@media (max-width: 480px) {\n"
    "  td > .toggle { grid-auto-flow: row; grid-auto-rows: 1fr; }\n"
    "  td > .toggle button { margin-left: 0; margin-top: -1px; border-radius: 0; }\n"
    "  td > .toggle button:first-child { margin-top: 0; border-radius: 4px 4px 0 0; }\n"
    "  td > .toggle button:last-child { border-radius: 0 0 4px 4px; }\n"
    "}\n"
)

#: C1 (after review): a long word breaks inside its answer's cell.
LONG_WORD = ".face .answers button { overflow-wrap: anywhere; }\n"

#: C1 (after review): where only one column fits, the list takes one cell.
ONE_COLUMN = (
    "@media (max-width: 359.98px) { .face .answers > select { grid-column: auto; } }\n"
)

#: C3: where the key goes, as every sign-in text says it (r5-b J19).
KEY_GOES = "The key goes to the OlmoEarth Agent here, not through Claude."

#: C3: the end of the default note's old second line, gone.
CHECKED = "and is checked with Studio before it is used"

#: C4: the face's own box and its focus ring.
FACE_BOX = (
    ".face { border: 1px solid var(--rule); border-radius: 4px; padding: 12px; }\n"
    ".face:focus { outline: none; }\n"
    ".face:focus-visible { outline: 2px solid var(--text); outline-offset: 2px; }\n"
)

#: C4 (after review): the rule's box above the face keeps the face's inset.
RULE_BOX = ".card.oe-rule { padding: 12px; }\n"

#: C4 (after review): on a narrow panel the face and the rule's box keep 6 px
#: at their sides, so a date of the strip holds a 240 px picture.
FACE_NARROW = (
    "@media (max-width: 420px) { .face, .card.oe-rule { padding: 12px 6px; } }\n"
)

#: The keys of a lead (OE.lead, server.LEAD_LINES).
LEAD_KEYS = ("what", "next", "cannot")


def _read(name: str) -> str:
    """A file of views/, as it is on disk."""
    return (views.VIEW_DIR / name).read_text(encoding="utf-8")


def _between(text: str, start: str, end: str) -> str:
    """The part of text from start up to the first end after it."""
    at = text.index(start)
    return text[at : text.index(end, at)]


def _style() -> str:
    """The style sheet every view gets (common.js OE.STYLE), comments and all."""
    return _between(_read("common.js"), "OE.STYLE = `", "\n`;\n")


def _object(text: str, start: str) -> list[tuple[str, str]]:
    """The keys and values of the object literal that follows start in text
    (a script), each stripped: its comments left out (// to the end of the
    line, /* ... */), split at the commas outside any nested bracket and
    outside any quoted string ('...', "..." or `...`), each item at its first
    colon."""
    i = text.index("{", text.index(start)) + 1
    depth = 1
    items: list[str] = []
    item = ""
    while depth:
        c = text[i]
        if c in "'\"`":
            j = i + 1
            while text[j] != c:
                j += 2 if text[j] == "\\" else 1
            item += text[i : j + 1]
            i = j + 1
            continue
        if text.startswith("//", i):
            i = text.index("\n", i)
            continue
        if text.startswith("/*", i):
            i = text.index("*/", i) + 2
            continue
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
        if depth == 1 and c == ",":
            items.append(item)
            item = ""
        elif depth:
            item += c
        i += 1
    items.append(item)
    pairs = [s.strip().partition(":") for s in items if s.strip()]
    return [(key.strip(), value.strip()) for key, _, value in pairs]


def _lines_given(pairs: list[tuple[str, str]]) -> list[str]:
    """The lead keys an object literal gives a value other than null."""
    return [key for key, value in pairs if key in LEAD_KEYS and value != "null"]


# --------------------------------------------------------------------------- C1


def test_c1_same_kind_controls_have_one_rule() -> None:
    """C1: the shared rule makes a group of same-kind controls equal columns
    (a square's answers, and any group marked .oe-same, such as the Check
    card's counts), every row as tall as the tallest; it comes after .row,
    so the answers' grid wins over the row's flex, and after .card. A joined
    toggle and the face's Previous and Next pair are a grid of equal columns;
    the toggle's old inline-flex rule is gone."""
    style = _style()
    assert SAME in style
    assert JOINED in style
    at = style.index(SAME)
    assert style.index("\n.row { display: flex;") < at
    assert style.index("\n.card { background: var(--paper);") < at
    assert style.index(SAME) < style.index(JOINED)
    assert ".toggle { display: inline-flex; }" not in style
    assert re.search(r"\n\.toggle \{", style) is None
    # The toggle's join and its pressed look are as they were (r5-b J16).
    assert ".toggle button:first-child { margin-left: 0;" in style
    assert '.toggle button[aria-pressed="true"] {' in style


def test_c1_the_answers_on_the_face_take_one_cell_each() -> None:
    """C1: an answer button no longer has its own minimum width (a tick then
    made a chosen one wider); its padding is less, the list of more classes
    spans two cells, and "Other class..." is an answer cell with a dashed
    edge in the secondary colour; the old 96 px is gone. The rules reach the
    face's answers (a div 'answers row') and its pager (the row in
    .face-top), as sheet.js draws them, in every view."""
    style = _style()
    assert ANSWERS in style
    assert "min-width: 96px" not in style
    assert "padding: 10px 16px" not in style
    sheet = _read("sheet.js")
    assert "class: 'answers row'" in sheet
    assert "{ class: 'face-top' }" in sheet
    for name in views.VIEWS:
        page = views.view_html(name)
        assert JOINED in page, name
        assert SAME in page, name


def test_c1_a_list_rows_answers_where_the_row_has_no_room() -> None:
    """C1 (after review): a list row's answers, the one joined toggle in a
    table cell (sheet.js labelCell), give up side-by-side equal columns where
    they do not fit: beside the open pictures pane (a panel of 760 px or
    more) they keep their words' widths, so the row stays one line (H4); on
    a panel under 480 px they stand one under the other, each as wide as the
    widest. Both rules come after the joined toggle's, so they win, and
    neither names a toggle outside a table cell."""
    style = _style()
    pane = _between(style, "@media (min-width: 760px) {", "\n}\n")
    assert ROW_BESIDE_PANE in pane
    assert ROW_NARROW in style
    assert style.index(JOINED) < style.index(ROW_NARROW) < style.index(pane)
    assert re.search(r"\n\.toggle \{", style) is None
    sheet = _read("sheet.js")
    assert "class: 'toggle small', role: 'group', tabindex: '-1'," in sheet
    assert "OE.el('td', {}, labelCell(w))" in sheet


def test_c1_a_long_class_name_breaks_inside_its_cell() -> None:
    """C1 (after review): an answer's cell has the grid's width, not its
    words', so a class name of one long word breaks inside the button and
    does not run into the next cell; where only one column fits, the list of
    more classes takes one cell (a span of two would add a second column of
    another width). Both come after the answers' rules, so they win."""
    style = _style()
    assert LONG_WORD in style
    assert ONE_COLUMN in style
    assert style.index(ANSWERS) < style.index(LONG_WORD) < style.index(ONE_COLUMN)


# --------------------------------------------------------------------------- C2


def test_c2_the_lead_draws_at_most_two_lines_in_the_body_colour() -> None:
    """C2: OE.lead keeps the first two lines that have words; no rule colours
    a lead's line (the grey third line is gone), and the comment says so."""
    common = _read("common.js")
    lead = _between(common, "OE.lead = (lead) => {", "\n};")
    assert (
        ".filter(([, s]) => s).slice(0, 2);  // r5-e: at most two lines (owner: fewer)"
        in lead
    )
    assert "if (!lines.length) return null;" in lead
    assert "OE.el('p', { class: k }, s)" in lead
    style = _style()
    assert ".oe-lead > p.cannot" not in style
    assert re.search(r"\.oe-lead[^{]*\{[^}]*\bcolor:", style) is None
    assert (
        "/* A card's plain lines under its heading (OE.lead): at most two, in the"
        " body\n   colour; the rest goes in the card's Details fold (r5-e). */" in style
    )
    assert "three plain lines" not in style
    assert "one block of at most two lines" in common


def test_c2_no_card_of_a_view_sends_three_lines() -> None:
    """C2: the leads the views give themselves (the start card's, the
    projects card's, the sheet's) each name two lines or fewer (a key whose
    value is not null), so the cap is only a guard."""
    start = _object(_read("start.html"), "const START_LEAD = ")
    projects = _object(_read("studio.html"), "const PROJECTS_LEAD = ")
    sheet = _object(_read("sheet.js"), "const leadWords = ")
    for name, pairs in (("start", start), ("projects", projects), ("sheet", sheet)):
        assert pairs, name
        assert {key for key, _ in pairs} <= set(LEAD_KEYS), (name, pairs)
        assert 1 <= len(_lines_given(pairs)) <= 2, (name, pairs)


def test_c2_no_server_lead_has_three_lines() -> None:
    """C2: every lead the server sends a result view gives two lines or
    fewer: each LEAD entry, each PLAN_LEAD*, and the review sets' leads as
    _lead fills them (with their counts, pictured or not, boundary first, a
    Studio result's points with the check card under the list)."""
    pytest.importorskip("mcp")
    from olmoearth_agent.mcp_app import server

    def given(lead: dict[str, str | None]) -> int:
        """How many of the lead's lines have words."""
        assert set(lead) <= set(LEAD_KEYS), lead
        return sum(1 for key in LEAD_KEYS if isinstance(lead.get(key), str))

    assert tuple(server.LEAD_LINES) == LEAD_KEYS
    plans = {
        k: v
        for k, v in vars(server).items()
        if k.startswith("PLAN_LEAD") and isinstance(v, dict)
    }
    assert {"PLAN_LEAD", "PLAN_LEAD_POINTS"} <= set(plans)
    for name, lead in plans.items():
        assert given(lead) <= 2, name
    for tool, lead in server.LEAD.items():
        assert given(lead) <= 2, tool
    ranked = {"n_review": 2, "n_windows": 8}
    pictured = {"review": {"pictures": True}}
    studio = {
        "n_review": 7,
        "n_windows": 96,
        "sampling": {"grid": "10x10", "n_valid": 72},
    }
    asks = {"review": {"check_request": {"result_id": "r9", "prediction_id": "p9"}}}
    for tool, inner, data in (
        ("olmoearth_review_set", ranked, pictured),
        ("olmoearth_review_set", ranked, {}),
        ("olmoearth_review_set", {**ranked, "order": "boundary_first"}, pictured),
        ("olmoearth_review_set_from_result", studio, asks),
        ("olmoearth_review_set_from_result", studio, {}),
        ("olmoearth_plan_label_sample", {}, {"labels_csv_path": "x", "unit": "square"}),
        ("olmoearth_plan_label_sample", {}, {"labels_csv_path": "x", "unit": "point"}),
    ):
        lead = server._lead(tool, inner, data)
        assert 1 <= given(lead) <= 2, (tool, inner, data, lead)


# --------------------------------------------------------------------------- C3


def test_c3_the_default_sign_in_note_says_where_the_key_goes_once() -> None:
    """C3: the default note is the line on what the button opens, then
    SIGN_IN_KEY_GOES itself, marked r5e; the long second line that said
    Studio checks the key is gone from common.js and from every view; the
    panel still falls back to the default note."""
    common = _read("common.js")
    assert (
        "// WORDING: r5e\nOE.SIGN_IN_NOTE = [OE.SIGN_IN_OPENS, OE.SIGN_IN_KEY_GOES];"
        in common
    )
    assert f"// WORDING: r5b J19\nOE.SIGN_IN_KEY_GOES = '{KEY_GOES}';" in common
    assert CHECKED not in common
    assert "adds where the key goes, in the words every sign-in text" in common
    assert "that Studio checks it. A card" not in common
    panel = _between(common, "OE.signInPanel = ", "\n};")
    assert "const lines = given.length ? given : OE.SIGN_IN_NOTE;" in panel
    for name in views.VIEWS:
        assert CHECKED not in views.view_html(name), name


# --------------------------------------------------------------------------- C4


def test_c4_the_square_sits_in_the_faces_own_box() -> None:
    """C4: the face (the one square: its heading, Previous and Next, the
    map's class, the answers and the pictures) has a hairline edge in the
    table rules' grey, 4 px corners and 12 px inside; its focus ring is 2 px
    outside the edge (before, 4 px outside an edge that was not drawn)."""
    style = _style()
    assert FACE_BOX in style
    assert "--rule: rgb(224,224,224);" in style
    assert "outline-offset: 4px" not in style
    assert style.index(FACE_BOX) < style.index(".face-top {")


def test_c4_the_rules_box_and_a_narrow_panel_keep_one_inset() -> None:
    """C4 (after review): the rule's box right above the face has the face's
    12 px inside (it had a card's 16 px), so the rule's words and the
    square's heading start at one left edge; on a panel of 420 px or less
    both keep 6 px at their sides, so a date of the pictures strip still
    holds a 240 px picture. Both rules come after the face's box, so they
    win, and before the pager's rule."""
    style = _style()
    assert RULE_BOX in style
    assert FACE_NARROW in style
    assert style.index(FACE_BOX) < style.index(RULE_BOX) < style.index(FACE_NARROW)
    assert style.index(FACE_NARROW) < style.index(".face-top {")
    assert "class: 'card stack oe-rule'" in _read("sheet.js")


def test_c4_only_the_sheet_editor_draws_a_face() -> None:
    """C4: the box is drawn where the sheet editor draws its face (the label
    sheet view, and the editor in the Check card and in a result card), and
    no other script names the class face."""
    face = re.compile(r"class: '(?:[^']* )?face(?: [^']*)?'")
    for path in sorted(views.VIEW_DIR.iterdir()):
        if path.suffix not in (".js", ".html"):
            continue
        found = face.findall(path.read_text(encoding="utf-8"))
        if path.name == "sheet.js":
            assert found == ["class: 'face stack'"], found
        else:
            assert not found, path.name
