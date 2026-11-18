# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Round r5-e (17 Nov 2026), track RESULT: the result cards and the
look-first list (views/result.html, views/figures.js and the result words in
server.py), each item as specs/r5-e-design/build_list.md gives it. The
server's words through its own names; the views are read as text (no test
here runs the scripts):

- R1: every look-first card is one size (the owner's screenshot of 17 Nov
  2026: cards 10 and 12 wrapped "Map says: trees" and were taller): one
  fixed height, the rank in a column two digits wide, a caption cut at two
  lines; Street map is said once, outlined in the open square's pane beside
  Previous and Next (the two one width), no longer under every card; the
  points' rank buttons 1 to 9 and 10 to 50 are one width;
- R2: a plan's lead is one line, which no longer says the button's words
  again; the look-first lead is shorter ("The 14 squares the model was least
  sure of, ..." under the overline "Look first"), its points' form without
  the grid's clause; beside the check card under a Studio result's list no
  second line (the card says it, with its button); the result card's
  signed-out note is the panel's two lines;
- R3: the one contained action (a Studio look-first's check, a plan's sheet
  button) comes right under the lines that lead to it, before the card's one
  Details fold, whose last node is the look-first numbers (a closed fold of
  their own);
- R4: under "One result per part, never averaged" the face says "Check the
  other part too." and Details the side-by-side sentence with its reason
  (the server's SAY_SIDE_BY_SIDE); the area sentence of a map of more than
  two classes is the Details' last counted line, in words of its own there
  ("This check says ..."), no longer the end of the last line; the model's
  lines keep both;
- R5: in the result card's parts figure each part's words sit inside its own
  box, every box in the row one height; no part is narrower than its longest
  word, and on a panel of 560 px or less the parts are stacked.
"""

from __future__ import annotations

import copy
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

#: R2: the plan's one line, of squares and of points.
PLAN_SQUARES = (
    "The squares to check, picked before any answer: answer every one on the sheet."
)
PLAN_POINTS = (
    "The points to check, picked before any answer: answer every one on the sheet."
)

#: R2: the look-first lead of 14 squares among 280 read, and of 7 points
#: among 72 read (one point in the singular below).
LOOK_SQUARES = (
    "The 14 squares the model was least sure of, among the 280 read; the number 14 "
    "was chosen in advance, not a count of doubtful squares. No labels were used, "
    "so the map may be right or wrong there."
)
LOOK_POINTS = (
    "The 7 points the model was least sure of, among the 72 read; the number 7 was "
    "chosen in advance, not a count of doubtful points. No labels were used, so the "
    "map may be right or wrong there."
)

#: R4: the face's line under the parts, and its second sentence, in Details.
OTHER_ONE = "Check the other part too."
OTHER_THREE = "Check the other 3 parts too."
SIDE_BY_SIDE = (
    "Report the parts' results side by side; do not average or add them: this check "
    "gives no range for a combined rate."
)
#: R4: the area sentence as Details says it, a paragraph of its own there, so
#: it names its subject (the model's line keeps "It says ...").
AREA = "This check says nothing about the area of any one class."

#: R1: the look-first cards' style before this round, gone: the narrower
#: column, the least height in place of one height, the rank's free width,
#: and the cell that held a card and its Street map.
OLD_CARD_STYLE = (
    "minmax(140px, 1fr)",
    "min-height: 44px",
    "flex: none;",
    "window-cell",
)

#: Two review rows as the tool lists them (rank, index, row, column, margin,
#: the map's class).
ROWS = [
    {
        "rank": 1,
        "window_index": 5,
        "row": 1,
        "col": 1,
        "margin": 0.0123457,
        "predicted_class": 0,
        "class_name": "water",
    },
    {
        "rank": 2,
        "window_index": 2,
        "row": 0,
        "col": 2,
        "margin": 0.0456,
        "predicted_class": 1,
        "class_name": "trees",
    },
]


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


def _in_order(text: str, *pieces: str) -> bool:
    """Whether each piece is in text, each after the one before it."""
    at = [text.index(p) for p in pieces]
    return at == sorted(at)


def _px(pattern: str, rule: str) -> int:
    """The whole number of px that pattern's one group finds in a CSS rule."""
    found = re.search(pattern, rule)
    assert found is not None, pattern
    return int(found.group(1))


def _style() -> str:
    """result.html's own style block."""
    return _between(_read("result.html"), "<style>", "</style>")


def _review_block() -> str:
    """result.html's reviewBlock, whole."""
    return _between(_read("result.html"), "function reviewBlock(app, block) {", "\n}\n")


def _show() -> str:
    """result.html's show, whole."""
    return _between(_read("result.html"), "function show(app, params) {", "\n}\n")


def _inner(**extra: Any) -> dict[str, Any]:
    """An ok estimate's result as the tool returns it: a random design of 80
    labels, 10 of them wrong, its 95% range 6.1% to 21.7%."""
    return {
        "plan_design": "random",
        "n_labelled": 80,
        "n_unjudged": 0,
        "nominal_coverage": 0.95,
        "valid_interval": {
            "estimate": 0.125,
            "low": 0.061,
            "high": 0.217,
            "method": "exact hypergeometric interval at 95%",
        },
        **extra,
    }


def _app() -> server.AppServer:
    """An app server with no key, its Studio a mock address (never called)."""
    return server.AppServer(
        None, keys=auth.KeyStore(), base_url="http://mock-studio/api/v1"
    )


# --------------------------------------------------------------------------- R1


def test_r1_every_look_first_card_is_one_size() -> None:
    """R1: the cards' grid takes 160 px columns (two on a phone, as before);
    each card has one fixed height (two caption lines of 18 px, 8 + 8 padding,
    a 1 + 1 border: 54 px, read from the rules; a fixed height, since rows of
    1fr shrink once the strip scrolls); the strip's box is 399 px at most, six
    rows and half of the seventh (at 420 px it ended 6 px short of the seventh
    row and shaved its edge); the rank sits in a column two digits wide,
    right-aligned in tabular figures, so 1 and 12 take one width; a caption
    longer than two lines is cut (its whole words stay in the card's name and
    in the pane). The cell that held a card and its Street map is gone."""
    style = _style()
    for text in (
        ".review-strip { display: grid; grid-template-columns: repeat(auto-fill,"
        " minmax(160px, 1fr)); gap: 8px;\n    max-height: 399px; overflow: auto; }",
        ".review-strip button.window-card { height: 54px; display: flex;"
        " align-items: center; gap: 10px;",
        ".window-card .rank { flex: 0 0 2ch; text-align: right;"
        " font-variant-numeric: tabular-nums; font-size: 20px;\n"
        "    line-height: 28px; font-weight: 600; }",
        ".window-card .caption { min-width: 0; font-size: 12px; line-height: 18px;"
        " overflow-wrap: anywhere;\n    display: -webkit-box;"
        " -webkit-box-orient: vertical; -webkit-line-clamp: 2; overflow: hidden; }",
        "@media (max-width: 560px) {\n"
        "    .review-strip { grid-template-columns: repeat(2, minmax(0, 1fr)); }",
    ):
        assert text in style, text
    for gone in OLD_CARD_STYLE:
        assert gone not in style, gone
    # 18 + 18 lines, 8 + 8 padding and a 1 + 1 border: 54 px, read from the
    # rules (the border and the border-box sizing are common.js's).
    card = _between(style, ".review-strip button.window-card {", "}")
    caption = _between(style, ".window-card .caption {", "}")
    common = _read("common.js")
    base = _between(common, "\nbutton {", "}")
    height = _px(r"(?<![-\w])height: (\d+)px;", card)
    pad = _px(r"(?<![-\w])padding: (\d+)px", card)  # the first value: top and bottom
    line = _px(r"(?<![-\w])line-height: (\d+)px;", caption)
    border = _px(r"(?<![-\w])border: (\d+)px solid", base)
    assert "* { box-sizing: border-box; }" in common
    assert height == 2 * line + 2 * pad + 2 * border == 54
    # The strip's box: six rows and their six gaps, and half of the seventh.
    gap = _px(r"(?<![-\w])gap: (\d+)px;", _between(style, ".review-strip {", "}"))
    assert 6 * (height + gap) + height // 2 == 399
    assert "window-cell" not in _read("result.html")


def test_r1_street_map_is_said_once_in_the_open_squares_pane() -> None:
    """R1 (the owner's note 2): no Street map under each card, which said the
    same thing on every card, outside its box; the open square's pane has one,
    outlined (small), right after Previous and Next and before Close, named
    for its square (REVIEW_WORDS.mapOf) and pressed by its data-focus map:k,
    as openMap finds it; the test scene shows none (it is no square of the
    list). Each card is a button of its own in the strip."""
    block = _review_block()
    cards = _between(block, "  function cards() {", "\n  }\n")
    assert "mapButton(" not in cards
    assert (
        "strip.replaceChildren(...windows.slice(0, state.shown).map((w, i) =>"
        " OE.el('button', {" in cards
    )
    pane = _between(block, "  function render(shown) {", "\n  }\n")
    top = _between(pane, "    const top = ", "    const parts = [top];")
    assert top.count("mapButton(") == 1
    assert _in_order(
        top,
        "state.test ? button('back', REVIEW_WORDS.back(w.rank),",
        " : [\n",
        "OE.el('div', { class: 'review-pager' },",
        "button('previous', REVIEW_WORDS.previous,",
        "button('next', REVIEW_WORDS.next,",
        "mapButton(w.rank, REVIEW_WORDS.streetMap, 'small'),",
        "        ],\n",
        "REVIEW_WORDS.close",
    )
    made = _between(block, "  const mapButton = ", "  }, words);")
    named = "'data-focus': 'map:' + rank, 'aria-label': REVIEW_WORDS.mapOf(rank),"
    assert named in made
    assert "section.querySelectorAll('[data-focus=\"map:' + rank + '\"]')" in block
    # The words are r4-k K7's, unchanged: "Street map", once per open square.
    words = _between(_read("result.html"), "const reviewWords = (unit) => {", "\n};")
    assert "    streetMap: 'Street map',\n" in words


def test_r1_the_panes_previous_and_next_are_one_width() -> None:
    """R1 (design rule R1, the build list's own example: Previous 72 px
    against Next 49 px): in the open square's pane Previous and Next sit in
    one group, a grid of equal columns, so each is as wide as the wider, as on
    the label sheet's face; the group holds the two alone: Street map and
    Close stay the row's own children, each as wide as its words. The rule is
    result.html's own (common.js's names the sheet's face)."""
    rule = (
        "  .review-pane .review-pager { display: inline-grid;"
        " grid-auto-flow: column; grid-auto-columns: 1fr; gap: 8px; }\n"
    )
    assert rule in _style()
    pane = _between(_review_block(), "  function render(shown) {", "\n  }\n")
    top = _between(pane, "    const top = ", "    const parts = [top];")
    assert top.count("class: 'review-pager'") == 1
    group = _after(top, "OE.el('div', { class: 'review-pager' },", "\n          //")
    lines = [line.strip() for line in group.strip().splitlines()]
    assert len(lines) == 2
    assert lines[0].startswith("button('previous', REVIEW_WORDS.previous,")
    assert lines[0].endswith("{ disabled: i === 0 }),")
    assert lines[1].startswith("button('next', REVIEW_WORDS.next,")
    assert lines[1].endswith("{ disabled: i === n - 1 })),")
    # The row's own children after the group: Street map, then Close.
    assert _in_order(
        top,
        "{ disabled: i === n - 1 })),",
        "          mapButton(w.rank, REVIEW_WORDS.streetMap, 'small'),\n",
        "        OE.el('button', { type: 'button', class: 'link',"
        " 'data-focus': 'close', onclick: close }, REVIEW_WORDS.close)));",
    )


def test_r1_the_points_rank_buttons_are_one_width() -> None:
    """R1: the points' Street map buttons, each its rank, are at least 36 px
    wide in tabular figures, so 1 to 9 and 10 to 50 are one size (a two-digit
    rank in a small button is about 35 px)."""
    style = _style()
    assert (
        "  .row.review-links { gap: 4px; }\n"
        "  .row.review-links button.small { min-width: 36px;"
        " font-variant-numeric: tabular-nums; }\n" in style
    )
    row = _between(_review_block(), "    if (n && linkKey) {", "    return section;")
    assert "windows.map((w) => mapButton(w.rank, String(w.rank), 'small'))" in row


# --------------------------------------------------------------------------- R2


def test_r2_the_plan_lead_is_one_line() -> None:
    """R2: a plan's lead is one line, in squares or in points; its "next" said
    the button's own words again (sheet.OPEN_FIRST) and its "cannot" what the
    sheet's greyed finish button and counter show (r5-e H1 took the sheet's
    sentence too), so both are gone; marked; no face word kept off faces."""
    assert server.PLAN_LEAD == {"what": PLAN_SQUARES, "next": None, "cannot": None}
    assert server.PLAN_LEAD_POINTS == {
        "what": PLAN_POINTS,
        "next": None,
        "cannot": None,
    }
    assert server.LEAD["olmoearth_plan_label_sample"] == server.PLAN_LEAD
    lead = server._lead("olmoearth_plan_label_sample", {}, {"labels_csv_path": "x"})
    assert lead == server.PLAN_LEAD
    points = server._lead(
        "olmoearth_plan_label_sample", {}, {"labels_csv_path": "x", "unit": "point"}
    )
    assert points == server.PLAN_LEAD_POINTS
    for line in (PLAN_SQUARES, PLAN_POINTS):
        assert sheet.OPEN_FIRST not in line
        assert _OFF_FACE.search(line) is None, line
    source = _server_source()
    assert "# WORDING: squares; r5e R2\nPLAN_LEAD: dict[str, str | None] = {" in source
    assert (
        "# WORDING: squares; r5e R2\nPLAN_LEAD_POINTS: dict[str, str | None] = {"
        in source
    )


def test_r2_the_look_first_lead_is_shorter() -> None:
    """R2: the look-first lead no longer starts "Where to look first:" (the
    overline says "Look first"), keeps that the number was chosen in advance
    and is no count of doubtful squares (the reader test's D/D-card-3), and
    says in plain words that no label was used; a Studio result's points
    without the grid's clause; one square or point in the singular; marked."""
    squares = server._lead(
        "olmoearth_review_set",
        {"n_review": 14, "n_windows": 280},
        {"review": {"pictures": True}},
    )
    assert squares == {
        "what": LOOK_SQUARES,
        "next": server.REVIEW_LEAD_NEXT,
        "cannot": None,
    }
    points = server._lead(
        "olmoearth_review_set_from_result",
        {"n_review": 7, "n_windows": 96, "sampling": {"grid": "10x10", "n_valid": 72}},
        {"review": {"pictures": False}},
    )
    assert points == {
        "what": LOOK_POINTS,
        "next": server.REVIEW_LEAD_NEXT_NO_PICTURES_POINTS,
        "cannot": None,
    }
    assert "grid" not in str(points["what"]) and "square" not in str(points["what"])
    one = server._lead("olmoearth_review_set", {"n_review": 1, "n_windows": 280}, {})
    assert one["what"] == (
        "The 1 square the model was least sure of, among the 280 read; the number 1 "
        "was chosen in advance, not a count of doubtful squares. No labels were "
        "used, so the map may be right or wrong there."
    )
    for line in (LOOK_SQUARES, LOOK_POINTS):
        assert not line.startswith("Where to look first")
        assert _OFF_FACE.search(line) is None, line
    source = _server_source()
    assert "# WORDING: r4h V1, D/D-card-3; r5e R2\nREVIEW_LEAD_WHAT = (" in source
    assert "# WORDING: r4h V1, D/D-card-3; r5e R2\nREVIEW_LEAD_POINTS = (" in source


def test_r2_no_second_line_beside_the_check_card() -> None:
    """R2: a Studio result's list with a check request (the card under it
    says how to learn how wrong the map is, with its button) has no second
    line; without one it keeps it, and so does a scores file's list, which
    still points to the main job."""
    inner = {"n_review": 7, "sampling": {"n_valid": 72}}
    request = {"result_id": "r9", "prediction_id": None}
    beside = server._lead(
        "olmoearth_review_set_from_result",
        inner,
        {"review": {"pictures": False, "check_request": request}},
    )
    assert beside == {"what": LOOK_POINTS, "next": None, "cannot": None}
    alone = server._lead(
        "olmoearth_review_set_from_result", inner, {"review": {"pictures": False}}
    )
    assert alone["next"] == server.REVIEW_LEAD_NEXT_NO_PICTURES_POINTS
    scores = server._lead(
        "olmoearth_review_set",
        {"n_review": 2, "n_windows": 8},
        {"review": {"pictures": False}},
    )
    assert scores["next"] == server.REVIEW_LEAD_NEXT_NO_PICTURES
    # A boundary-first list beside the card: its third line alone.
    edge = server._lead(
        "olmoearth_review_set_from_result",
        {**inner, "order": "boundary_first"},
        {"review": {"pictures": False, "check_request": request}},
    )
    assert edge == {"what": None, "next": None, "cannot": server.REVIEW_LEAD_CANNOT}


@pytest.mark.asyncio
async def test_r2_a_studio_results_list_with_its_check_has_one_line(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R2 through the host's call: a Studio result's ranked list (its result
    names the result id) carries the check request in its block, and its lead
    the first line alone, the review block read before the lead."""
    app = _app()
    envelope = {
        "ok": True,
        "result": {
            "n_review": 2,
            "n_windows": 96,
            "sampling": {"grid": "10x10", "n_valid": 72},
            "result_id": "r9",
            "review": copy.deepcopy(ROWS),
            "must_state": ["A review set is not a sample."],
        },
    }

    async def run(_name: str, _arguments: dict[str, Any]) -> dict[str, Any]:
        """The ranked list, whatever is asked."""
        return copy.deepcopy(envelope)

    monkeypatch.setattr(app, "run_agent_tool", run)
    try:
        shown = await app.call_agent_tool(
            "olmoearth_review_set_from_result", {"result_id": "r9"}, views=True
        )
    finally:
        await app.aclose()
    data = (shown.meta or {})[server.VIEW_DATA_KEY]
    request = {"result_id": "r9", "prediction_id": None}
    assert data["review"]["check_request"] == request
    assert data["lead"] == {
        "what": "The 2 points the model was least sure of, among the 72 read; the "
        "number 2 was chosen in advance, not a count of doubtful points. No labels "
        "were used, so the map may be right or wrong there.",
        "next": None,
        "cannot": None,
    }


def test_r2_the_result_cards_sign_in_note_is_two_lines() -> None:
    """R2: signed out, the result card's note is the panel's two lines (what
    the button opens, where the key goes); its own first line, "This step
    needs OlmoEarth Studio.", said the sign-in's heading again; marked."""
    show = _show()
    assert (
        "      // WORDING: r5b J19; r5e R2\n"
        "      note: [OE.SIGN_IN_OPENS, OE.SIGN_IN_KEY_GOES],\n" in show
    )
    assert "This step needs OlmoEarth Studio." not in _read("result.html")


# --------------------------------------------------------------------------- R3


def test_r3_the_one_action_comes_before_the_one_details_fold() -> None:
    """R3: show draws the lead, the list, a refusal's reason and the face's
    other nodes, then the one contained action (a Studio look-first's check
    button, a plan's sheet button), then the card's one Details fold, last;
    the fold is drawn once."""
    show = _show()
    assert _in_order(
        show,
        "if (lead) root.append(lead);",
        "if (listBlock) root.append(listBlock);",
        "root.append(reasonBlock(data.reason));",
        "root.append(...onFace);",
        "const ask = data.review && typeof data.review === 'object'"
        " ? checkBlock(app, data.review.check_request) : null;",
        "if (ask) root.append(ask);",
        "if (plan) root.append(sheetButton(app, data, figureSlot));",
        "if (inFold.length) root.append(OE.fold(FOLD_LABEL, ...inFold));",
        "if (root.children.length === 1) root.append(OE.el('div', {}, 'Done.'));",
    )
    assert show.count("OE.fold(FOLD_LABEL") == 1
    assert "root.append(reviewBlock(app, data.review));" not in show
    # The check button and the plan's sheet button stay the contained kind.
    page = _read("result.html")
    check = _between(page, "function checkBlock(app, request) {", "\n}\n")
    assert "OE.el('button', { class: 'primary', onclick:" in check
    button = _between(page, "function sheetButton(app, data, figureSlot) {", "\n}\n")
    assert "    class: 'primary',\n" in button


def test_r3_the_numbers_are_the_last_node_of_details() -> None:
    """R3: the look-first list draws no second fold of its own: it keeps its
    numbers (a closed fold, "The numbers for these squares") on the block,
    with a key under the cards and without one in place of the cards; show
    pushes them last into the card's Details, after a plan's sheet file."""
    block = _review_block()
    assert block.count("numbersFold(windows, unit)") == 2
    assert "    } else if (n) section.numbers = numbersFold(windows, unit);" in block
    assert "  section.append(strip, more, pane, csvBlock());\n" in block
    assert "  section.numbers = numbersFold(windows, unit);" in block
    assert "section.append(numbersFold(" not in block
    assert "numbersFold(windows, unit));" not in block
    show = _show()
    assert _in_order(
        show,
        "if (plan) inFold.push(planFile(data));",
        "if (listBlock && listBlock.numbers) inFold.push(listBlock.numbers);",
        "if (inFold.length) root.append(OE.fold(FOLD_LABEL, ...inFold));",
    )
    page = _read("result.html")
    fold = _between(page, "function numbersFold(windows, unit) {", "\n}\n")
    assert "return OE.el('details', { class: 'oe-numbers' }," in fold
    assert "OE.el('summary', {}, words.numbers)" in fold


# --------------------------------------------------------------------------- R4


def test_r4_the_parts_line_says_one_sentence_and_details_the_reason() -> None:
    """R4: under "One result per part, never averaged" the face says the first
    sentence of the server's SAY_OTHER (SAY_OTHERS for k parts); the second,
    with its reason, is sideBySide, the server's SAY_SIDE_BY_SIDE word for
    word, in the card's Details once the map has two parts or more, after the
    rule and the class lines, before the rare classes' counted lines; marked."""
    page = _read("result.html")
    words = _between(page, "const RESULT_WORDS = {", "\n};")
    assert "  // WORDING: r5e R4\n  // WORDING: r4m M2\n  otherPart: (k) => " in words
    marked = "  // WORDING: r5e R4\n  // WORDING: r4m M2 (server SAY_SIDE_BY_SIDE"
    assert marked in words
    after = _after(words, "otherPart: (k) => (k === 1\n    ? ", "),\n")
    one, many = after.split("\n    : ")
    assert _concat(one, {}) == OTHER_ONE
    assert _concat(many, {"k": "3"}) == OTHER_THREE
    side = _concat(_entry(words, "sideBySide: "), {})
    assert side == SIDE_BY_SIDE == server.SAY_SIDE_BY_SIDE.strip()
    assert OTHER_ONE + " " + side == server.SAY_OTHER.strip()
    assert OTHER_THREE + " " + side == server.SAY_OTHERS.format(k="3").strip()
    card = _between(page, "function resultCard(app, data) {", "\n}\n")
    face = "other ? OE.el('p', { class: 'small' }, RESULT_WORDS.otherPart(other))"
    assert face + " : null" in card
    assert _in_order(
        card,
        "const folded = [];",
        "classLines.forEach((line) => folded.push(line));",
        "if (parts.length > 1) folded.push(OE.el('p',"
        " { class: 'small oe-side-by-side' }, RESULT_WORDS.sideBySide));",
        "fewDetail.forEach((s) => folded.push(OE.el('p',"
        " { class: 'small oe-few-detail' }, s)));",
        "if (folded.length) card.append(OE.fold(RESULT_WORDS.details, ...folded));",
    )
    for line in (OTHER_ONE, SIDE_BY_SIDE):
        assert _OFF_FACE.search(line) is None, line


def test_r4_the_area_sentence_is_in_details() -> None:
    """R4: on a map of more than two classes the area sentence is the last of
    the Details' counted lines (uncovered_detail), in words of its own there
    (server.AREA_DETAIL: a paragraph of its own names its subject, "This check
    says ...", where "It says ..." read as the map or the class line before
    it), and the last line is one sentence; a map of two classes has neither;
    the model's third line keeps its own sentence (LAST_LINE_AREA) under the
    same condition, as before."""
    two = {"population": {"class_names": {"0": "absent", "1": "present"}}}
    three = {"population": {"classes": ["water", "trees", "crops"]}}
    plain = server._estimate_result(_inner(), None, None, design=two)
    many = server._estimate_result(_inner(), None, None, design=three)
    assert plain is not None and many is not None
    assert plain["view"]["last_line"] == server.LAST_LINE_ANSWERS
    assert plain["view"]["uncovered_detail"] == []
    assert many["view"]["last_line"] == server.LAST_LINE_ANSWERS
    assert many["view"]["uncovered_detail"] == [AREA]
    assert server.AREA_DETAIL == AREA
    assert AREA not in many["view"]["last_line"]
    assert _OFF_FACE.search(AREA) is None
    # The model's lines keep both sentences (the say-first lines, unchanged).
    said = " It says nothing about the area of any one class."
    assert server.LAST_LINE_AREA == said
    assert many["say_first"][2] == plain["say_first"][2] + server.LAST_LINE_AREA
    source = _server_source()
    assert (
        "    area = [AREA_DETAIL] if n_classes is not None and n_classes > 2 else []\n"
        in source
    )
    assert "# WORDING: r5e R4\nAREA_DETAIL = " in source
    assert '"uncovered_detail": list(uncovered["few_detail"]) + area,' in source
    # The page folds every uncovered_detail line, as before.
    card = _between(_read("result.html"), "function resultCard(app, data) {", "\n}\n")
    assert (
        "const fewDetail = (Array.isArray(r.uncovered_detail) ? r.uncovered_detail"
        " : []).filter((s) => text(s));" in card
    )


def test_r4_the_counted_lines_come_before_the_area_sentence(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R4: a graded sheet of a map of three classes, two of them seen too
    rarely: under the headline their names; in Details each one's counted
    line, then the area sentence (the server's order, which the page keeps);
    the last line one sentence."""
    rare = {
        "n": 80,
        "n_unjudged": 0,
        "classes": [
            {"code": "a", "name": "A", "seen": 3, "agreed": 2, "outside": False},
            {"code": "b", "name": "B", "seen": 2, "agreed": 0, "outside": False},
            {"code": "c", "name": "C", "seen": 75, "agreed": 70, "outside": False},
        ],
    }
    monkeypatch.setattr(server.check, "sheet_context", lambda p: None)
    monkeypatch.setattr(server.imagery, "served_any", lambda p: False)
    monkeypatch.setattr(sheet, "labelling_rule", lambda p: None, raising=False)
    monkeypatch.setattr(sheet, "graded_state", lambda p: None, raising=False)
    monkeypatch.setattr(sheet, "graded_squares", lambda p, v: None, raising=False)
    monkeypatch.setattr(
        sheet, "class_counts", lambda p, v: copy.deepcopy(rare), raising=False
    )
    path = root / "west_to_label.csv"
    path.write_text("order,window_index,wrong,reference_class\n", encoding="utf-8")
    grading = {"first_at": "2026-11-13T10:04:00Z", "changed": False, "gradings": 1}
    mark = ("v" + "a" * 16, grading)
    got = server._estimate_result(_inner(design="random"), str(path), mark)
    assert got is not None
    view = got["view"]
    assert view["uncovered"] == [
        'Too few squares to say how well the map finds: "A", "B".'
    ]
    detail = view["uncovered_detail"]
    assert len(detail) == 3 and detail[-1] == AREA
    assert all(line.startswith("Only ") for line in detail[:2])
    assert view["last_line"] == server.LAST_LINE_ANSWERS


# --------------------------------------------------------------------------- R5


def test_r5_each_parts_words_sit_inside_its_box() -> None:
    """R5 (the owner's note 2: no loose words under a box): each part's label
    is a span inside its box, no longer a line under an empty 20 px bar; the
    row stretches its parts, each box fills its part and is at least 28 px
    tall, its words centred inside with 4 and 8 px of room, so every box in
    the row is one height; on a filled box (this result) the words are
    light. The state rules are as before."""
    script = _read("figures.js")
    parts = _between(script, "  const parts = (", "\n  };")
    assert (
        "OE.el('div', { class: 'oe-part-box' }, OE.el('span', { class: 'oe-part-label"
        " small' }, labels[i])));" in parts
    )
    assert "OE.el('div', { class: 'oe-part-label small' }" not in script
    for token in (
        ".oe-parts { display: flex; gap: 4px; align-items: stretch; }\n",
        ".oe-part-box { flex: 1; min-height: 28px; padding: 4px 8px; display: flex;"
        " align-items: center;\n  border: 1.5px solid var(--text-2);"
        " border-radius: 2px; background: var(--paper); }\n",
        ".oe-part.this .oe-part-label { color: var(--canvas); }\n",
        ".oe-part.this .oe-part-box { background: var(--text);"
        " border-color: var(--text); }\n",
        ".oe-part.not .oe-part-box { border-style: dashed; background: transparent; }",
        ".oe-part-label { color: var(--text); overflow-wrap: break-word; }",
    ):
        assert token in script, token
    assert "height: 20px" not in script
    assert "align-items: flex-start; }\n.oe-part " not in script
    # The figure's name for a screen reader is every label, as before.
    assert (
        "return OE.el('div', { class: 'oe-fig oe-parts', role: 'img',"
        " 'aria-label': labels.join('; ') }, pieces);" in parts
    )


def test_r5_no_part_is_narrower_than_its_longest_word() -> None:
    """R5 on a narrow panel with a small part (a part at the 12% floor had
    about 18 px for its words at 380 px, so its label broke letter by letter
    into about ten lines and both boxes grew to about 200 px): a part is never
    narrower than its label's longest word (min-content, which break-word,
    unlike anywhere, does not shrink), so the share sets the width above that;
    at 560 px or less (the steps' own breakpoint) the parts are stacked, each
    full width, in rows of one height. The row's own rule is as R5 made it,
    and the stacked rule comes after it."""
    script = _read("figures.js")
    for token in (
        ".oe-parts { display: flex; gap: 4px; align-items: stretch; }\n"
        ".oe-part { min-width: min-content; display: flex; flex-direction: column;"
        " gap: 4px; }\n",
        ".oe-part-label { color: var(--text); overflow-wrap: break-word; }\n"
        "@media (max-width: 560px) { .oe-parts { display: grid;"
        " grid-template-columns: minmax(0, 1fr); grid-auto-rows: 1fr; } }\n",
    ):
        assert token in script, token
    for gone in (
        ".oe-part { min-width: 0;",
        ".oe-part-label { color: var(--text); overflow-wrap: anywhere; }",
    ):
        assert gone not in script, gone
    row = script.index(".oe-parts { display: flex;")
    assert row < script.index("@media (max-width: 560px) { .oe-parts {")


def test_r5_the_result_card_draws_the_parts_figure() -> None:
    """R5: the result card draws the parts figure in its result mode, only
    with two parts or more, the line under it the face's one sentence (R4)."""
    card = _between(_read("result.html"), "function resultCard(app, data) {", "\n}\n")
    assert "OE.fig.parts(parts, { mode: 'result' })" in card
    drawn = "if (parts.length > 1 && OE.fig && typeof OE.fig.parts === 'function') {"
    assert drawn in card
    assert "oe-part-label" not in _code(_read("result.html"))
