# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Round r4-m (17 Nov 2026), each string as specs/r4-m-round.md gives it; the
views are read as text, as views.view_html() serves them (no test here runs
the scripts), and the server's words through its own functions:

- M1: the line under the headline for the classes nobody saw says what the
  rate covers: every square of the map (of this part; every point of the
  map's grid; of the map they were picked from when the part is not known),
  all classes together. It is drawn for a random design only, a simple
  random sample of the whole map or part, so no design keeps the old words;
- M2: the parts are reported side by side, never averaged or added, with the
  reason: this check gives no range for a combined rate (the card and every
  say-first line that says the rule);
- M3: the plan card's face shows no file path; its fold says the sheet is
  also a file on this computer, that it shows the map's class at each square
  (or point), so answering there is not blind; on a sheet labelled right or
  wrong, which shows the map's class itself, that the file shows each
  square's place, order and stratum, which the sheet here hides (review);
- M4: the Check card of a class map without scores says, before the count is
  fixed, that each point needs the reviewer's own imagery or records; so
  does the model's note, before its question on the count (review);
- M5: the box beside the kelp rule says what to write and that each square
  is judged by it;
- M6: every sheet face but right or wrong (which shows the map's class) says
  "The map's answer is hidden: answer without looking at the map.";
- M7: a class line starts from what the reader saw;
- M8: the sheet's rule block, on a face with Cannot tell and pictures, says
  that a square with no clear date is Cannot tell, in the kelp-only line's
  place (review: the page copy has the pictures' rule, so the pane's own
  rule lines are a fallback); the pane, where the page copy has no rule,
  shows the rule's Absent half and, on a face with Cannot tell, that line in
  place of the scope and kelp-only lines; a date is headed by its day, its
  scene cloud in the pane's own fold.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("mcp")

from olmoearth_agent.mcp_app import (  # noqa: E402
    auth,
    imagery,
    server,
    sheet,
    views,
)

BASE = "http://mock-studio/api/v1"

#: One token of a JS concatenation: a single-quoted string, a plus, or a name
#: (a call with no parentheses inside its own, such as unitOf(unit), is one
#: name), as tests/mcp_app/test_mcp_r4e_f3_views.py reads them.
_TOKEN = re.compile(
    r"\s*(?:'((?:[^'\\]|\\.)*)'|(\+)|([A-Za-z_][A-Za-z0-9_.]*(?:\([^()]*\))?))"
)

#: M2's reason, after "do not average or add them".
REASON = ": this check gives no range for a combined rate"

#: M3: the plan card's fold line, around the sheet's path.
PLAN_FILE = (
    "The sheet is also a file on this computer: {path}. It shows the map's class "
    "at each {unit}, so answering there is not blind; use the sheet here instead."
)

#: M3, after review: the line for a sheet labelled right or wrong.
PLAN_FILE_WRONG = (
    "The sheet is also a file on this computer: {path}. It shows each {unit}'s "
    "place, order and stratum, which the sheet here hides; use the sheet here "
    "instead."
)

#: M4, after review: what the model tells the user before the count.
CLASS_MAP_TELL = (
    "Tell the user first, in these words or plainer: 'Each point needs your own "
    "imagery or records from the map's dates; the app shows no pictures of "
    "points.' Then ask the user two questions"
)

#: M4: the Check card's second line on a class map without scores, as round
#: r5-e K7 cuts it (what the check gives is in its Details).
CLASS_MAP_SECOND = (
    "The app shows no pictures of points: each needs your own imagery or records "
    "from the map's dates."
)
#: M4: the sentence the model's note gives the user first (its own words since
#: r5-e K7, "in these words or plainer"; the card says the same in another order).
NOTE_TELL = (
    "Each point needs your own imagery or records from the map's dates; the app "
    "shows no pictures of points."
)

#: M5: the box's label beside the app's rule, as r5-b J18 shortens it (the map
#: maker's meaning kept; what an empty box does is a small line under the box),
#: worded for any map by round r5-l L2 (tests/mcp_app/test_mcp_r5l_views.py).
RULE_OWN = (
    "Did the map maker mean something else? Write their meaning of Present and "
    "Absent (optional)."
)

#: M6: what every sheet face says after its instruction; r5-e H1: a clause of
#: the first line ("..., without looking at the map."), no longer a sentence
#: that says the answer is hidden and not to look.
HIDDEN = ", without looking at the map."

#: M8: the cloud line (reworded after review: only the case the rule leaves
#: open; r5-d R02: as a case, not a fact about the square shown), the pane's
#: fold's summary and first line.
CLOUD_LINE = "If cloud covers the square on every date, press Cannot tell."
CLOUD_FOLD = "Scene cloud of each date"
CLOUD_WHOLE = "Scene cloud is for the whole satellite image, not this square."


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
    ends its last line."""
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


def _code(page: str) -> str:
    """The text with its comments taken out."""
    return re.sub(r"/\*.*?\*/|//[^\n]*", "", page, flags=re.S)


def _in_order(text: str, *pieces: str) -> bool:
    """Whether each piece is in text, each after the one before it."""
    at = [text.index(p) for p in pieces]
    return at == sorted(at)


def _words() -> str:
    """OE.sheetWords, the label sheet's string table."""
    return _between(_read("sheet.js"), "OE.sheetWords = (unit) => {", "\n};")


def _said(key: str, unit: str, **names: str) -> str:
    """An entry of OE.sheetWords as the face says it, for a unit word."""
    one = "point" if unit == "point" else "square"
    value = re.sub(r"^\([a-z]+\) => ", "", _entry(_words(), f"    {key}: "))
    return _concat(value, {"one": one, "many": one + "s", **names})


def _editor() -> str:
    """The label editor's code (OE.sheetEditor)."""
    return _between(_read("sheet.js"), "OE.sheetEditor = (app, data,", "\n};")


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


# --------------------------------------------------------------------------- M1


def test_m1_the_unseen_line_says_what_the_rate_covers() -> None:
    """Every square of the map, all classes together; of this part on a map
    of two parts or more; every point of the map's grid for a sheet of
    points; of the map they were picked from when the part is not known."""
    assert server.CLASS_UNSEEN_ALL == "every {unit} of the map, all classes together"
    assert server.CLASS_UNSEEN_ALL_POINTS == (
        "every point of the map's grid, all classes together"
    )
    assert server.CLASS_UNSEEN_ALL_PART == (
        "every {unit} of this part, all classes together"
    )
    assert server.CLASS_UNSEEN_ALL_UNKNOWN == (
        "every {unit} of the map they were picked from, all classes together"
    )
    names = ['"Snow"']
    # Round r5-j B4: "You did not see ... in any {unit}".
    start = 'You did not see "Snow" in any {unit}, so this check says nothing about '
    end = "how well the map finds it. The rate above is for {whole}."
    picked = "every square of the map they were picked from, all classes together"
    for unit, kw, whole in (
        ("square", {}, "every square of the map, all classes together"),
        ("point", {}, "every point of the map's grid, all classes together"),
        (
            "square",
            {"several": True},
            "every square of this part, all classes together",
        ),
        ("square", {"known": False}, picked),
        # The part not known wins over a part count (a record with no list).
        ("square", {"known": False, "several": True}, picked),
    ):
        line, detail = server._unseen_lines(names, unit, **kw)
        said = start.format(unit=unit) + end.format(whole=whole)
        assert line == said, (unit, kw)
        assert detail is None
    many = [f'"C{i}"' for i in range(6)]
    line, detail = server._unseen_lines(many, "square", m=8, known=False)
    assert line == (
        "You did not see 6 of the map's 8 classes in any square, so this check says "
        "nothing about how well the map finds them. The rate above is for every square "
        "of the map they were picked from, all classes together. Names under Details."
    )
    assert str(detail).startswith("Classes you saw in no square: ")
    # No word of the old "all squares together" anywhere in the server.
    source = Path(inspect.getsourcefile(server) or "").read_text(encoding="utf-8")
    assert '"all {units} together"' not in source
    assert '"all {units} of this part"' not in source


def test_m1_uncovered_passes_the_part_known_on() -> None:
    """_uncovered says the not-known words with known false; known true is
    the default (the map, or the part with several)."""
    counts = {
        "n": 9,
        "n_unjudged": 0,
        "classes": [
            {"code": "a", "name": "Water", "seen": 9, "agreed": 8},
            {"code": "b", "name": "Snow", "seen": 0, "agreed": 0},
        ],
    }
    unknown = server._uncovered(counts, "square", 9, several=False, known=False)
    assert unknown["lines"][-1].endswith(
        "The rate above is for every square of the map they were picked from, all "
        "classes together."
    )
    known = server._uncovered(counts, "square", 9, several=False)
    assert known["lines"][-1].endswith(
        "The rate above is for every square of the map, all classes together."
    )


def test_m1_the_line_shows_for_a_random_design_only() -> None:
    """The design check of M1: _estimate_result gives _uncovered the counts
    only for a random design (a simple random sample of every valid square
    of the map or part, or every point of the grid), so the whole-map words
    never stand beside a design whose rate is not the whole's; and it hands
    on whether the part is known, in the _uncovered call itself (not the
    _class_check call's own known=known)."""
    source = inspect.getsource(server._estimate_result)
    assert 'picked_at_random = inner.get("plan_design") == "random"' in source
    assert source.count("uncovered = _uncovered(") == 1
    call = _between(source, "uncovered = _uncovered(", "\n    )")
    assert "counts if picked_at_random and agreed else None" in call
    assert "known=known," in call


# --------------------------------------------------------------------------- M2


def test_m2_side_by_side_carries_its_reason_everywhere() -> None:
    """The card's line and the model's say-first lines: one other part,
    several, none open, and the Check card's line for a map in parts."""
    assert server.SAY_OTHER == (
        " Check the other part too. Report the parts' results side by side; do not "
        "average or add them" + REASON + "."
    )
    assert server.SAY_OTHERS.format(k="2") == (
        " Check the other 2 parts too. Report the parts' results side by side; do "
        "not average or add them" + REASON + "."
    )
    assert server.SAY_SIDE_BY_SIDE == (
        " Report the parts' results side by side; do not average or add them"
        + REASON
        + "."
    )
    assert server.SAY_CHECK_NONE.format(parts=server.SAY_CHECK_NONE_PARTS) == (
        "There is no result until every square of a sheet is done; each part gets "
        "its own result, never one rate for all parts. To cover the whole map, "
        "report each part's rate with its range, side by side; do not average or "
        "add them up" + REASON + "."
    )
    for said in (server.SAY_OTHER, server.SAY_OTHERS, server.SAY_SIDE_BY_SIDE):
        assert "into one rate" not in said and "Then report" not in said
    words = _between(_read("result.html"), "const RESULT_WORDS = {", "\n};")
    assert "  // WORDING: r4m M2\n  otherPart: (k) => (k === 1\n" in words
    after = _after(words, "otherPart: (k) => (k === 1\n    ? ", "),\n")
    one, many = after.split("\n    : ")
    # Round r5-e, R4: the first sentence on the card's face, the second (with
    # its reason, the server's SAY_SIDE_BY_SIDE word for word) in Details.
    # Round r5-f, F07: the face's line then says where, in a sentence of its
    # own that the model's lines do not have.
    ask = " Ask Claude to open the Check card again."
    side = _concat(_entry(words, "sideBySide: "), {})
    assert side == server.SAY_SIDE_BY_SIDE.strip()
    one_said = _concat(one, {})
    assert one_said.endswith(ask)
    assert one_said.removesuffix(ask) + " " + side == server.SAY_OTHER.strip()
    many_said = _concat(many, {"k": "3"})
    assert many_said.endswith(ask)
    many_first = many_said.removesuffix(ask) + " " + side
    assert many_first == server.SAY_OTHERS.format(k="3").strip()


# --------------------------------------------------------------------------- M3


def test_m3_the_plan_cards_face_has_no_file_path() -> None:
    """The sheet button's row holds the button and the count only; no "or
    fill the CSV" anywhere in the page."""
    page = _read("result.html")
    assert "or fill the CSV" not in page
    button = _between(page, "function sheetButton(app, data, figureSlot) {", "\n}\n")
    assert "OE.el('div', { class: 'row' }, button, count)," in button
    assert "csv-path" not in _code(button)
    assert "data.labels_csv_path))" not in button


def test_m3_the_fold_says_the_file_is_not_blind() -> None:
    """planFile: the fold's last line for a plan, before the face is drawn;
    in the sheet's own unit; the path in its mono span, wrapping anywhere. A
    sheet labelled right or wrong (the server's right_or_wrong) shows the
    map's class itself: its line says what the file adds there instead."""
    page = _read("result.html")
    head = _after(page, "// WORDING: r4m M3\nconst PLAN_FILE_HEAD = ", ";\n")
    tail = _after(page, "// WORDING: r4m M3\nconst PLAN_FILE_TAIL = (unit) => ", ";\n")
    wrong = _after(
        page, "// WORDING: r4m M3\nconst PLAN_FILE_TAIL_WRONG = (unit) => ", ";\n"
    )
    for unit in ("square", "point"):
        named = {"unitOf(unit)": unit}
        said = _concat(head, {}) + "/out/p_to_label.csv" + _concat(tail, named)
        assert said == PLAN_FILE.format(path="/out/p_to_label.csv", unit=unit)
        said = _concat(head, {}) + "/out/p_to_label.csv" + _concat(wrong, named)
        assert said == PLAN_FILE_WRONG.format(path="/out/p_to_label.csv", unit=unit)
    made = _between(page, "function planFile(data) {", "\n}\n")
    assert (
        "OE.el('p', { class: 'small csv-path oe-plan-file' }, PLAN_FILE_HEAD," in made
    )
    assert "OE.el('span', { class: 'mono' }, data.labels_csv_path), tail);" in made
    assert (
        "const tail = data.right_or_wrong === true ? PLAN_FILE_TAIL_WRONG(data.unit)"
        " : PLAN_FILE_TAIL(data.unit);" in made
    )
    show = _between(page, "function show(app, params) {", "\n}\n")
    # Round r5-e, R3: the sheet's button right under the lines, then Details.
    assert _in_order(
        show,
        "put(notesBlock(data.must_state), true);",
        "if (plan) inFold.push(planFile(data));",
        "root.append(...onFace);",
        "if (plan) root.append(sheetButton(app, data, figureSlot));",
        "if (inFold.length) root.append(OE.fold(FOLD_LABEL, ...inFold));",
    )
    assert ".csv-path { overflow-wrap: anywhere; }" in page


def test_m3_a_plans_view_data_names_its_unit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The server gives a plan with a sheet its unit (_points_plan, read once:
    the lead takes it from the view data), so the fold's line says squares
    or points as the sheet does, and whether the sheet is labelled right or
    wrong (its counts' label_rule; false when they cannot be read); a plan
    with no sheet gets neither."""
    app = server.AppServer(None, keys=auth.KeyStore(), base_url=BASE)
    asked: list[Any] = []

    def points(inner: dict[str, Any]) -> bool:
        asked.append(inner)
        return bool(inner.get("sampling"))

    monkeypatch.setattr(server, "_points_plan", points)
    plan = "olmoearth_plan_label_sample"
    sampled = {"labels_csv_path": "/out/p_to_label.csv", "sampling": {"grid": "10x10"}}
    data = app._view_data(plan, {"ok": True, "result": sampled}, None, {}, True)
    assert data["unit"] == sheet.UNIT_POINT
    assert data["lead"] == server.PLAN_LEAD_POINTS
    assert len(asked) == 1
    asked.clear()
    patches = {"labels_csv_path": "/out/s_to_label.csv"}
    data = app._view_data(plan, {"ok": True, "result": patches}, None, {}, True)
    assert data["unit"] == sheet.UNIT_SQUARE
    assert data["lead"] == server.PLAN_LEAD
    assert data["right_or_wrong"] is False
    assert len(asked) == 1
    for rule, wrong in (("wrong", True), ("seen", False), ("class", False)):
        counts = {"label_rule": rule}
        data = app._view_data(plan, {"ok": True, "result": patches}, counts, {}, True)
        assert data["right_or_wrong"] is wrong, rule
    none = app._view_data(plan, {"ok": True, "result": {}}, None, {}, True)
    assert "unit" not in none and "right_or_wrong" not in none
    # Called alone, the lead still reads the unit off the plan.
    asked.clear()
    lead = server._lead(plan, sampled, {"labels_csv_path": "x"})
    assert lead == server.PLAN_LEAD_POINTS and len(asked) == 1


# --------------------------------------------------------------------------- M4


def test_m4_a_class_maps_check_card_says_no_pictures_of_points() -> None:
    """The second line, before any count is fixed; C6's tag kept under M4's
    (and r5-e's over it: K7 cut the face to two lines); the sheet's own line
    stays as it was."""
    page = _read("check.html")
    lines = _after(
        page,
        "// WORDING: r4m M4 (the second line)\n// WORDING: r4h C6\nconst CLASS_MAP = [\n",
        "\n];",
    )
    said = tuple(_concat(s, {}) for s in lines.rstrip(",").split(",\n"))
    assert said == (
        "This map can be checked on points picked at random, instead of squares.",
        CLASS_MAP_SECOND,
    )
    assert "at most 256 points" not in page
    # r5-e H1: the sheet's line ends "..., without looking at the map."
    assert _said("whatPlaced", "point").startswith(
        "This sheet has no pictures. See where each point is, then say what was "
        "there from your own imagery or records"
    )


def test_m4_the_models_note_says_it_before_the_count() -> None:
    """In a client with no panel the model's note is all the user hears: it
    says the card's sentence before the question on the count (review of
    r4-m; word for word until r5-e K7 put the card's in another order), with
    its WORDING line."""
    note = server.CHECK_CLASS_MAP_NOTE
    assert CLASS_MAP_TELL in note
    assert f"'{NOTE_TELL}'" in note
    assert note.index(CLASS_MAP_TELL) < note.index("(1) 'How many points can you")
    source = Path(inspect.getsourcefile(server) or "").read_text(encoding="utf-8")
    tag = "# WORDING: r4h C6, E/said-refused-2; r4m M4\nCHECK_CLASS_MAP_NOTE = ("
    assert tag in source


# --------------------------------------------------------------------------- M5


def test_m5_the_rule_box_says_what_to_write() -> None:
    """The box's label in its new words, C5's tag kept under M5's; the box
    shows no grey hint. r5-b J18 shortens the label and sets no placeholder
    anywhere: the gate's example is a small line under its box, and under
    this box a small line says what an empty box does."""
    page = _read("check.html")
    assert "// WORDING: r4m M5\n// WORDING: r4h C5\nconst RULE_OWN = " in page
    assert _concat(_after(page, "const RULE_OWN = ", ";\n"), {}) == RULE_OWN
    assert "Write its own rule here" not in page
    gate = _between(page, "  function fillGate() {", "\n  }\n")
    assert "placeholder" not in gate
    assert "gate.replaceChildren(...ruleLines(), boxed(RULE_OWN, RULE_EMPTY));" in gate
    assert "What the map's makers count as Present" not in page


# --------------------------------------------------------------------------- M6


def test_m6_every_sheet_face_says_not_to_look_at_the_map() -> None:
    """what, whatBare and whatPlaced, for squares and for points, each with
    its WORDING line; the old sentence ends nowhere with a stop. The
    right-or-wrong face has no such line: it shows the map's class. r5-b J11
    gave whatBare (no pictures, no places) other words, which keep "without
    looking at the map". r5-e H1: what and whatPlaced end with the clause
    HIDDEN, one sentence each."""
    editor = _editor()
    assert "what: wrongFace ? null : picturesPossible ? W.what : whatBare," in editor
    for unit in ("square", "point"):
        one = "point" if unit == "point" else "square"
        assert _said("what", unit) == (
            f"Say what you see in each {one}'s satellite pictures{HIDDEN}"
        )
        # r5-b J11: a sheet with no pictures and no places says it cannot
        # show where a square is; it still says not to look at the map (r5-f
        # F12: the clause ends its sentence, and the way out follows it).
        bare = _said("whatBare", unit)
        assert bare.startswith("This sheet has no pictures and no places, so it")
        assert f"of each {one}, without looking at the map. No records? " in bare
        assert _said("whatPlaced", unit) == (
            f"This sheet has no pictures. See where each {one} is, then say what was "
            f"there from your own imagery or records{HIDDEN}"
        )
    lines = _words().split("\n")
    for key, tag in (
        ("what", "// WORDING: r4m M6"),
        ("whatBare", "// WORDING: r4m M6; r5b J11"),
        ("whatPlaced", "// WORDING: r4h S3 E/sheet-1; r4m M6"),
    ):
        (at,) = [i for i, line in enumerate(lines) if line.startswith(f"    {key}: ")]
        assert lines[at - 1].strip() == tag, key
    flat = re.sub(r"'\s*\+\s*'", "", _read("sheet.js")).replace("\\'", "'")
    assert "The map's answer is hidden.'" not in flat


# --------------------------------------------------------------------------- M7


def test_m7_a_class_line_starts_from_what_the_reader_saw() -> None:
    """With its range and without, in squares and in points, from the squares
    figure itself."""
    ranged = server.CLASS_LINE.format(
        name='"Bare ground"',
        k="50",
        units="squares",
        j="45",
        low="78.2%",
        high="96.7%",
    )
    assert ranged == (
        'Where you saw "Bare ground" (50 squares), the map also said "Bare ground" '
        "on 45; agreement range: 78.2% to 96.7%."
    )
    plain = server.CLASS_LINE_NO_RANGE.format(
        name='"Water"', k="8", units="squares", j="6"
    )
    assert plain == (
        'Where you saw "Water" (8 squares), the map also said "Water" on 6.'
    )
    got = {
        "face": "class",
        "cells": [],
        "classes": [{"code": "a", "name": "Grass", "seen": 6, "agreed": 5}],
    }
    shown = server._squares(got, {}, "point")
    assert shown is not None
    assert shown["classes"][0]["line"] == (
        'Where you saw "Grass" (6 points), the map also said "Grass" on 5.'
    )


# --------------------------------------------------------------------------- M8


def test_m8_the_words_of_the_pictures_pane() -> None:
    """The cloud line, the fold's summary and its first line, for squares,
    each with its WORDING line."""
    assert _said("cloudLine", "square") == CLOUD_LINE
    assert _said("cloudFold", "square") == CLOUD_FOLD
    assert _said("cloudWhole", "square") == CLOUD_WHOLE
    lines = _words().split("\n")
    for key in ("cloudLine", "cloudFold", "cloudWhole"):
        (at,) = [i for i, line in enumerate(lines) if line.startswith(f"    {key}: ")]
        assert lines[at - 1].strip() == "// WORDING: r4m M8", key


def test_m8_the_pane_shows_the_rules_absent_half_and_the_cloud_line() -> None:
    """In place of the scope and the kelp-only line: the rule's Present half,
    its Absent half word for word (ruleHalves), and the cloud line only on a
    face that takes Cannot tell (takesQ: a certificate plan's has none). Only
    where the page copy has no rule; the sheet's own rule block is the next
    test's."""
    editor = _editor()
    pane = _between(editor, "    let ruleLines = null;", "\n    const dated = ")
    for text in (
        "const [present, absent] = ruleHalves(answer.reading_rule);",
        "'Rule: ' + present",
        "absent ? OE.el('div', { class: 'small', style: 'font-weight:600' }, absent)"
        " : null,",
        "takesQ ? OE.el('div', { class: 'small' }, W.cloudLine) : null);",
    ):
        assert text in pane, text
    assert "rule_scope" not in pane and "rule_limit" not in pane
    halves = _between(editor, "  function ruleHalves(rule) {", "\n  }\n")
    assert "const at = text.search(/\\bAbsent\\s*:/i);" in halves
    assert (
        "return at > 0 ? [text.slice(0, at).trim(), text.slice(at).trim()]"
        " : [text, null];" in halves
    )
    assert "const takesQ = answers.some((a) => a.value === OE.UNJUDGED);" in editor
    # The app's rule has both halves, so the pane's split finds its Absent
    # (round r5-l L1: the general rule, and the kelp rule a record keeps).
    rule = imagery.READING_RULE
    at = rule.index("Absent:")
    assert rule[at:] == "Absent: on every clear date shown, it covers half or less."
    kelp = imagery.KELP_READING_RULE
    at = kelp.index("Absent:")
    assert kelp[at:] == "Absent: on every clear date shown it fills half or less."


def test_m8_the_rule_block_says_the_cloud_line_on_a_face_with_cannot_tell() -> None:
    """Review of r4-m: the page copy has the rule the pictures have (both
    imagery.rule_for), so the pane's own rule lines are a fallback. The
    sheet's rule block says the cloud line, after the rule, on a face that
    takes Cannot tell and can show pictures. r5-e H2: the scope is in the
    Details fold now, so the block (a box) holds the rule and, on such a face,
    the cloud line; r5-l L1: no line follows the scope there on which maps
    the rule is for (the panel reads no data.rule_limit)."""
    editor = _editor()
    assert _in_order(
        editor,
        "const takesQ = answers.some((a) => a.value === OE.UNJUDGED);",
        "  const cloudOnFace = takesQ && picturesPossible;\n  const ruleBlock",
    )
    block = _between(editor, "  const ruleBlock = pageRule ?", "\n  const clearNote")
    assert _in_order(
        block,
        "OE.el('strong', {}, 'Rule: '), pageRule),",
        "    cloudOnFace ? OE.el('div', { class: 'small' }, W.cloudLine) : null,\n",
        "    ruleExamples) : null;",
    )
    assert block.count("W.cloudLine") == 1
    assert "data.rule_scope" not in block and "data.rule_limit" not in block
    assert _in_order(
        editor,
        "  const scopeNote = typeof data.rule_scope === 'string'",
        "    classNoPeriod, scopeNote, gradedWhy,\n",
    )
    assert "data.rule_limit" not in editor and "limitNote" not in editor


def test_m8_a_dates_heading_leaves_its_scene_cloud_to_the_panes_fold() -> None:
    """OE.datesBlock takes a heading for each date's block (the pictures'
    names keep the whole label); the editor's dates are headed by their day
    (dateHead), and each label whole goes to the pane's own closed fold
    (cloudFold), after its first line; the window's pane and the test scene
    alike. The split follows the server's own label (imagery._date_label)."""
    script = _read("sheet.js")
    dates = _between(script, "OE.datesBlock = (", "\n};")
    assert "heading = null, unit = 'square' } = {}) => {" in dates
    assert "heading ? heading(label) : label" in dates
    assert "OE.picture(p, String(d.label || ''), what, null, unit)" in dates
    editor = _editor()
    assert "      heading: dateHead,\n      unit: W.one,\n" in editor
    assert "return /^(.+?), (scene cloud .+)$/.exec(label);" in editor
    fold = _between(editor, "  function cloudFold(dates) {", "\n  }\n")
    for text in (
        "if (!said.length) return null;",
        "OE.el('details', { class: 'oe-clouds small' }, OE.el('summary', {}, W.cloudFold),",
        # Round r5-p: each label whole, then its cloud mask line when it has one.
        "OE.el('div', {}, W.cloudWhole), ...said.map((d) => "
        "[OE.el('div', {}, String(d.label)),",
    ):
        assert text in fold, text
    # Not the sheet's own "Details" fold, nor the cards' oe-more.
    assert "OE.fold(" not in fold
    preview = _between(editor, "  function renderPreview(shown) {", "\n  }\n")
    assert "const clouds = dated ? cloudFold(answer.dates) : null;" in preview
    assert _in_order(preview, "datesBlock(answer.dates, W.label(k))", "...[clouds]")
    test = _between(editor, "  function renderTest() {", "\n  }\n")
    assert "...[cloudFold(dates)].filter(Boolean)," in test
    label = inspect.getsource(imagery._date_label)
    assert '}, scene cloud {cloud}"' in label
