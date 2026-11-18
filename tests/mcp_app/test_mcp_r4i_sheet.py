# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""The label sheet in the last fix round (r4i, 14 Nov 2026), builder B2's part.

From specs/r4-i-plan.md (sections 1, 2 and 4):

- O1 (the owner's ruling): a class map's sheet has no unblind switch and no
  "How to label" choice. The page copy sends no ``face_choices`` and no
  ``has_stratum``; an unblind on the class face is ignored; a two-map sheet
  keeps its switch, which sends the pair and never a stratum; the
  right-or-wrong face still sends the map's class; the server's face
  refusals stay.
- F11: no stratum on any face; the two-map switch's label.
- F5 and O2: the face's Cannot tell line and the fold's cost sentence.
- F9: ``imagery.RULE_KELP_ONLY`` as ``rule_limit`` on the page copy and the
  counts beside the owner's rule, shown after the rule's scope.
- O2: ``graded_squares`` with Cannot tell: the judged squares, then one
  empty cell per Cannot tell square (contract B2 to B1).
- F4: ``graded_state`` says whether the sheet holds an answer (contract B2
  to B1).

The pictures' ``rule_limit`` (F9) is held in test_mcp_r4p2_imagery.py, beside
the pictures' other rule tests. No test here runs the panel's script:
sheet.js is read as text, and a JS concatenation of single-quoted strings and
names is read into its text (:func:`_concat`), as the r4h tests read it.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

import pytest

from olmoearth_agent.mcp_app import imagery, sheet, views

ONE_MAP = [
    "order",
    "window_index",
    "row",
    "col",
    "map_class",
    "lon",
    "lat",
    "stratum",
    "wrong",
    "reference_class",
]

#: A two-map sheet with a stratum column.
TWO_MAP = [
    "order",
    "window_index",
    "row",
    "col",
    "class_a",
    "class_b",
    "stratum",
    "reference_class",
]

#: The 20 squares of each plan here.
ALL = tuple(range(20))

#: The map's class of each square on the seen fixture's design: present (1)
#: at squares 4 and 12, absent (0) elsewhere.
SEEN_MAP = [1 if w in (4, 12) else 0 for w in ALL]

#: A class map's classes as it writes them, and its class of each square:
#: square w is CLASSES[w % 3].
CLASSES = ["water", "trees", "crops"]
CLASS_MAP = [w % 3 for w in ALL]

#: A typed rule, and when it was typed.
TYPED = "Water: open water fills more than half of the outlined square."
TYPED_AT = "2026-11-13T10:15:00+00:00"


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    out = tmp_path / "outputs"
    out.mkdir()
    monkeypatch.setenv("OLMOEARTH_OUTPUT_ROOT", str(out))
    monkeypatch.delenv("OLMOEARTH_SCORES_ROOT", raising=False)
    return out


def _write(path: Path, header: list[str], rows: list[list[str]]) -> Path:
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)
    return path


def _sheet_rows(windows: tuple[int, ...], on_map: list[Any]) -> list[list[str]]:
    """The rows as a plan writes them: every label blank, a stratum of 0."""
    return [
        [str(order), str(w), "0", str(w), str(on_map[w]), "10.5", "50.2", "0", "", ""]
        for order, w in enumerate(windows, start=1)
    ]


def _seen(root: Path) -> Path:
    """A named scores map's plan of 20 squares (the kelp trial's): Present
    and Absent, under the owner's rule."""
    design = {
        "design": "random",
        "population": {
            "score_kind": "window_confidence",
            "class_names": {"0": "absent", "1": "present"},
            "map_class": SEEN_MAP,
        },
    }
    (root / "kelp.json").write_text(json.dumps(design), encoding="utf-8")
    return _write(root / "kelp_to_label.csv", ONE_MAP, _sheet_rows(ALL, SEEN_MAP))


def _class_map(root: Path) -> Path:
    """A class map's plan of 20 squares, labelled with the class seen."""
    design = {
        "design": "random",
        "population": {
            "score_kind": "class_map",
            "classes": CLASSES,
            "map_class": CLASS_MAP,
        },
    }
    (root / "land.json").write_text(json.dumps(design), encoding="utf-8")
    on_map = [CLASSES[c] for c in CLASS_MAP]
    return _write(root / "land_to_label.csv", ONE_MAP, _sheet_rows(ALL, on_map))


def _pair(root: Path) -> Path:
    """A two-map sheet whose rows hold a stratum; its design lists two
    classes and the maps show a third."""
    (root / "pair.json").write_text(
        json.dumps({"design": "disagreement", "classes": ["crop", "water"]}),
        encoding="utf-8",
    )
    return _write(
        root / "pair_to_label.csv",
        TWO_MAP,
        [
            ["1", "4", "0", "4", "water", "forest", "1", ""],
            ["2", "5", "0", "5", "crop", "water", "2", ""],
        ],
    )


def _by_window(path: Path) -> dict[int, str]:
    """Each window's handle by its index: the server's own map, never sent."""
    _, handles = sheet.window_handles(str(path))
    return {index: handle for handle, index in handles.items()}


def _answer(path: Path, face: str, answers: dict[int, str], **extra: Any) -> Any:
    """A panel's save of these answers by window, on ``face``, at the version
    the panel read."""
    read = sheet.read_sheet(str(path), face)
    at = _by_window(path)
    return sheet.save_answers(
        str(path),
        sheet_id=read["sheet_id"],
        face=face,
        rule=read["label_rule"],
        answers=[{"handle": at[w], "answer": a} for w, a in answers.items()],
        version=read["version"],
        **extra,
    )


def _class_row(name: str, seen: int, agreed: int) -> dict[str, Any]:
    """A class map's class as the counts list it: its code is its name."""
    return {
        "code": name,
        "name": name,
        "seen": seen,
        "agreed": agreed,
        "outside": False,
    }


# ------------------------------------------------------------- sheet.js as text

#: One token of a JS concatenation: a single-quoted string, a plus, or a name.
_TOKEN = re.compile(
    r"\s*(?:'((?:[^'\\]|\\.)*)'|(\+)|([A-Za-z_][A-Za-z0-9_.]*(?:\([^()]*\))?))"
)


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


def _script() -> str:
    return (views.VIEW_DIR / "sheet.js").read_text(encoding="utf-8")


def _between(text: str, start: str, end: str) -> str:
    at = text.index(start)
    return text[at : text.index(end, at)]


def _after(text: str, start: str, end: str) -> str:
    at = text.index(start) + len(start)
    return text[at : text.index(end, at)]


def _entry(block: str, start: str) -> str:
    """One entry's value in a JS object literal: from start to the comma that
    ends its last line."""
    at = block.index(start) + len(start)
    end = block.find(",\n", at)
    return (block[at:] if end < 0 else block[at:end]).strip().rstrip(",")


def _words() -> str:
    return _between(_script(), "OE.sheetWords = (unit) => {", "\n};")


def _editor() -> str:
    return _between(_script(), "OE.sheetEditor = (app, data,", "\n};")


def _said(key: str, unit: str) -> str:
    """An entry of OE.sheetWords as the face says it, for a unit word."""
    words = _words()
    one = "point" if unit == "point" else "square"
    tail = _after(words, "const cannotTellRest = ", ";\n")
    rest = _concat(tail, {"many": one + "s"})
    value = _entry(words, f"    {key}: ")
    return _concat(value, {"one": one, "many": one + "s", "cannotTellRest": rest})


def _code(js: str) -> str:
    """JS text without its comments: each ``/* ... */`` block and each line
    that is a ``//`` comment."""
    bare = re.sub(r"/\*.*?\*/", "", js, flags=re.DOTALL)
    lines = bare.split("\n")
    return "\n".join(line for line in lines if not line.strip().startswith("//"))


# ------------------------------------------------------------------ O1 and F11


def test_a_class_sheets_page_copy_has_no_switch_and_no_stratum(root: Path) -> None:
    """O1: a class map's page copy sends no face_choices and no has_stratum;
    an unblind on the class face is ignored (no unblinded, no map's class, no
    stratum); the right-or-wrong face, offered on a blank sheet, still sends
    the map's class, never a stratum, and ignores an unblind too."""
    path = _class_map(root)
    read = sheet.read_sheet(str(path))
    assert (read["face"], read["faces"]) == ("class", ["class", "wrong"])
    for gone in ("face_choices", "has_stratum", "unblinded"):
        assert gone not in read, gone
    shown = sheet.read_sheet(str(path), "class", unblind=True)
    assert "unblinded" not in shown and "has_stratum" not in shown
    assert all(set(w) == {"handle", "answer"} for w in shown["windows"])
    assert '"stratum"' not in json.dumps(shown)
    at = _by_window(path)
    wrong = sheet.read_sheet(str(path), "wrong", unblind=True)
    assert "unblinded" not in wrong
    assert {w["handle"]: w["map_class"] for w in wrong["windows"]} == {
        at[w]: CLASSES[CLASS_MAP[w]] for w in ALL
    }
    assert all(set(w) == {"handle", "answer", "map_class"} for w in wrong["windows"])


def test_a_two_map_sheet_keeps_its_switch_with_the_pair_and_no_stratum(
    root: Path,
) -> None:
    """O1 names class sheets: a two-map sheet's blind switch stays. Blind, its
    windows are a handle and an answer; its press sends each window's pair,
    and never the stratum its rows hold (F11)."""
    path = _pair(root)
    at = _by_window(path)
    blind = sheet.read_sheet(str(path))
    assert blind["face"] == "pair" and "unblinded" not in blind
    assert all(set(w) == {"handle", "answer"} for w in blind["windows"])
    shown = sheet.read_sheet(str(path), unblind=True)
    assert shown["unblinded"] is True
    assert {w["handle"]: w["pair"] for w in shown["windows"]} == {
        at[4]: ["water", "forest"],
        at[5]: ["crop", "water"],
    }
    assert all(set(w) == {"handle", "answer", "pair"} for w in shown["windows"])
    for gone in ("has_stratum", "face_choices"):
        assert gone not in shown, gone
    assert '"stratum"' not in json.dumps(shown)


def test_the_server_still_refuses_the_other_face_of_a_labelled_class_sheet(
    root: Path,
) -> None:
    """O1 keeps the refusals that guard saves: once a class seen is saved,
    right or wrong (which shows the map's class) is refused on a read and on
    a save, and the file stays as it was."""
    path = _class_map(root)
    _answer(path, "class", {0: "water"})
    read = sheet.read_sheet(str(path))
    assert read["faces"] == ["class"]
    with pytest.raises(ValueError) as refused:
        sheet.read_sheet(str(path), "wrong")
    assert str(refused.value) == sheet.FACE_WRONG_REFUSED
    at = _by_window(path)
    before = path.read_bytes()
    with pytest.raises(ValueError, match="offered only before any class seen"):
        sheet.save_answers(
            str(path),
            sheet_id=read["sheet_id"],
            face="wrong",
            rule="class",
            answers=[{"handle": at[1], "answer": "1"}],
        )
    assert path.read_bytes() == before


def test_the_panel_draws_no_switch_on_a_class_sheet() -> None:
    """O1 in sheet.js: no "How to label" choice, no class-face unblind, no
    face-switch refusal links, no stratum line or column; the blind checkbox
    is made under twoMap alone; a class face's Details line is the seen
    sheet's blind line; an answer waits only while a column is cleared; the
    server's face refusals are still known (a save can be refused so)."""
    script = _script()
    for gone in (
        "How to label",
        "Label right or wrong only",
        "Label the class seen",
        "Clear the classes seen",
        "Clear right or wrong",
        "Blind: hide the map",
        "switchFace",
        "modeSwitch",
        "face_choices",
        "faceNote",
        "state.switching",
        "has_stratum",
        "showStratum",
        "Stratum",
    ):
        assert gone not in script, gone
    editor = _editor()
    assert script.count("id: 'oe-blind'") == 1
    assert (
        "const blind = twoMap ? OE.el('input', {\n"
        "    type: 'checkbox', id: 'oe-blind', checked: true," in editor
    )
    assert "const switchRow = blind ? OE.el('div'," in editor
    assert (
        "const sentLine = seen || face === 'class' ? W.blindSeen : wrongFace ? "
        "W.wrongLine : blind ? W.codeLine : '';" in editor
    )
    assert "const blocked = () => state.clearing;" in editor
    extra = _between(editor, "  function extraLines(h) {", "\n  }\n")
    assert "if (wrongFace && extra.mapClass !== null) {" in extra
    assert "face === 'class'" not in extra and "stratum" not in extra
    rows = _between(editor, "  function renderList(focusKey) {", "\n  }\n")
    assert "const showMapClass = wrongFace;" in rows
    assert "stratum" not in _between(editor, "  const extraOf = (w) => ({", "});")
    face = sheet.FACE_WRONG_REFUSED.split(";")[0]
    assert f'OE.FACE_WRONG_REFUSED = "{face}";' in script
    progress = _between(editor, "  function updateProgress() {", "\n  }\n")
    assert "if (state.mix) {" in progress and "faceNote" not in progress


def test_no_face_string_of_the_panel_names_a_stratum() -> None:
    """F11: the two-map switch's label, with its WORDING line (in r5-b J28's
    plain words, which name no stratum either); past the comments and the
    right-or-wrong fold line (in Details, not reworded), no code of sheet.js
    says "stratum"."""
    script = _script()
    assert (
        "// WORDING: r4i F11; r5b J28\n  const blindText = 'Hide what the two maps "
        "say (keep it on while you answer)';" in script
    )
    wrong_line = _entry(_words(), "    wrongLine: ")
    assert "stratum" in wrong_line
    code = _code(script).replace(wrong_line, "")
    assert "stratum" not in code.lower()
    assert "the stratum" not in _code(script)


# ------------------------------------------------------------------------ F5


def test_the_sheets_cannot_tell_words_are_the_string_tables() -> None:
    """F5 and O2 (section 1): the face's line and the fold's sentence, for
    squares and for points, each with its WORDING line; the old words are
    gone."""
    rest = (
        " Do not guess: a guess can make the result wrong. A Cannot tell does "
        "not: you still get a result, with a wider range, but without the "
        "per-class lines."
    )
    for one in ("square", "point"):
        # r5-e H1: the face's line keeps the press and "do not guess"; what a
        # guess costs is the fold's sentence (cannotTell) alone.
        assert _said("cannotTellLine", one) == (
            f"Cannot judge a {one}? Press Cannot tell; do not guess."
        )
        assert _said("cannotTell", one) == (
            f"Cannot tell is for a {one} you cannot judge from the dates shown." + rest
        )
        assert _said("cannotTellBare", one) == (
            f"Cannot tell is for a {one} you cannot judge." + rest
        )
    words = _words()
    assert "// WORDING: r4i F5\n    cannotTellLine: " in words
    assert "// WORDING: r4i F5 O2\n  const cannotTellRest = " in words
    assert "// WORDING: r4i F5 O2\n    cannotTell: " in words
    assert "// WORDING: r4i F5 O2\n    cannotTellBare: " in words
    assert "it only makes the range wider" not in _script()
    assert "without the picture of" not in words
    # A sheet that takes no Cannot tell (a certificate's) keeps its own line.
    line = _between(_editor(), "  const cannotLineText = (() => {", "\n  })();")
    assert "if (takesQ) return W.cannotTellLine;" in line
    assert "'A certificate needs right or wrong on every ' + W.one + '.'" in line


# ------------------------------------------------------------------------ F9


def test_the_kelp_only_line_is_the_string_tables() -> None:
    """F9 (section 1), shortened by r4-j J6 (it always follows the scope,
    which names kelp canopy maps already), with its WORDING line; the Check
    card's scope (C5) stays word for word."""
    assert imagery.RULE_KELP_ONLY == "Any other map needs its own rule."
    source = Path(imagery.__file__).read_text(encoding="utf-8")
    assert "# WORDING: r4j J6\nRULE_KELP_ONLY = " in source
    assert imagery.RULE_SCOPE == (
        "This rule came with the app (set 11 Nov 2026 for kelp canopy maps)."
    )
    # Said after the scope: "kelp canopy maps" once.
    both = imagery.RULE_SCOPE + " " + imagery.RULE_KELP_ONLY
    assert both.count("kelp canopy maps") == 1


def test_the_page_copy_and_the_counts_say_the_owners_rule_is_for_kelp(
    root: Path,
) -> None:
    """F9: rule_limit is RULE_KELP_ONLY beside the owner's rule, in
    _rule_fields, the page copy and the counts; None beside a typed rule and
    with no rule at all."""
    kelp = _seen(root)
    real = sheet.resolve_sheet(str(kelp))
    design = sheet.design_of(real)[1]
    assert sheet._rule_fields(real, design, [], "one_map") == {
        "reading_rule": imagery.READING_RULE,
        "rule_scope": imagery.RULE_SCOPE,
        "rule_source": "owner",
        "rule_limit": imagery.RULE_KELP_ONLY,
    }
    for said in (sheet.read_sheet(str(kelp)), sheet.sheet_progress(str(kelp))):
        assert said["rule_source"] == "owner"
        assert said["rule_limit"] == imagery.RULE_KELP_ONLY
    sheet.set_typed_rule(str(kelp), TYPED, typed_at=TYPED_AT)
    for said in (sheet.read_sheet(str(kelp)), sheet.sheet_progress(str(kelp))):
        assert said["rule_source"] == "typed" and said["rule_limit"] is None
    land = _class_map(root)
    for said in (sheet.read_sheet(str(land)), sheet.sheet_progress(str(land))):
        assert said["reading_rule"] is None and said["rule_limit"] is None


def test_the_panel_shows_the_kelp_only_line_after_the_scope() -> None:
    """F9 in sheet.js: the panel shows data.rule_limit right after
    data.rule_scope, which names kelp canopy maps before it. r5-e H2 moved
    the scope from the rule block to the Details fold (scopeNote), and, after
    review, the kelp-only line with it (limitNote: alone under the rule,
    "any other map" had nothing to refer to); it is drawn only with the
    scope and, as before, only where the face has no cloud line. The
    pictures pane (a page copy with no rule, an answer with one) no longer
    repeats the scope and the kelp-only line: r4-m M8 puts the rule's Absent
    half and the cloud line there instead
    (tests/mcp_app/test_mcp_r4m_round.py)."""
    editor = _editor()
    block = _between(editor, "  const ruleBlock = pageRule ?", "\n  const clearNote")
    assert "OE.el('strong', {}, 'Rule: '), pageRule)," in block
    assert "data.rule_scope" not in block and "data.rule_limit" not in block
    scope = editor.index("  const scopeNote = typeof data.rule_scope === 'string'")
    limit = editor.index(
        "  const limitNote = pageRule && scopeNote && !cloudOnFace && typeof "
        "data.rule_limit === 'string' && data.rule_limit ? OE.el('div', "
        "{ class: 'small' }, data.rule_limit) : null;\n"
    )
    assert scope < limit
    fold = _between(editor, "  const foldLabel = 'Details';", "\n\n")
    assert "    classNoPeriod, scopeNote, limitNote, gradedWhy,\n" in fold
    pane = _between(editor, "    let ruleLines = null;", "\n    const dated = ")
    assert "answer.rule_scope" not in pane and "answer.rule_limit" not in pane


# ------------------------------------------------------------------------ O2


def test_a_seen_sheets_cannot_tell_squares_come_after_the_judged_ones(
    root: Path,
) -> None:
    """O2 (contract B2 to B1): 20 squares, 2 Cannot tell: n 20, n_unjudged
    2, the 18 judged by class (right before wrong), then 2 empty cells; the
    classes over the judged squares only, as class_counts gives them. None
    for another version, a square with no answer, or no judged square."""
    path = _seen(root)
    # Square 5 is seen present where the map says absent; 18 and 19 cannot tell.
    answers = {w: "1" if w in (4, 5, 12) else "0" for w in range(18)}
    _answer(path, "seen", {**answers, 18: "?", 19: "?"})
    version = sheet.sheet_version(str(path))
    got = sheet.graded_squares(str(path), version)
    assert got is not None
    assert (got["n"], got["n_unjudged"], got["face"]) == (20, 2, "seen")
    assert got["classes"] == [
        {"code": "1", "name": "Present", "seen": 3, "agreed": 2, "outside": False},
        {"code": "0", "name": "Absent", "seen": 15, "agreed": 15, "outside": False},
    ]
    unjudged = {"seen": None, "wrong": False, "unjudged": True}
    assert got["cells"] == [
        *[{"seen": "1", "wrong": False}] * 2,
        {"seen": "1", "wrong": True},
        *[{"seen": "0", "wrong": False}] * 15,
        unjudged,
        unjudged,
    ]
    counts = sheet.class_counts(str(path), version)
    assert counts == {"n": 20, "n_unjudged": 2, "classes": got["classes"]}
    for hidden in ("window_index", "10.5", "50.2", "stratum"):
        assert hidden not in json.dumps(got)
    assert sheet.graded_squares(str(path), "v" + "a" * 16) is None
    _answer(path, "seen", {0: ""})
    assert sheet.graded_squares(str(path), sheet.sheet_version(str(path))) is None
    _answer(path, "seen", {w: "?" for w in ALL})
    assert sheet.graded_squares(str(path), sheet.sheet_version(str(path))) is None


def test_a_class_sheets_cannot_tell_squares_come_after_the_judged_ones(
    root: Path,
) -> None:
    """O2 on a class map's sheet: the same shape, its classes the map's in
    its order. Each square is answered as the map says, but square 0 is seen
    as trees (the map says water) and squares 1 and 2 are Cannot tell."""
    path = _class_map(root)
    answers = {w: CLASSES[CLASS_MAP[w]] for w in ALL}
    answers.update({0: "trees", 1: "?", 2: "?"})
    _answer(path, "class", answers)
    version = sheet.sheet_version(str(path))
    unjudged = {"seen": None, "wrong": False, "unjudged": True}
    assert sheet.graded_squares(str(path), version) == {
        "n": 20,
        "n_unjudged": 2,
        "face": "class",
        "cells": [
            *[{"seen": "water", "wrong": False}] * 6,
            *[{"seen": "trees", "wrong": False}] * 6,
            {"seen": "trees", "wrong": True},
            *[{"seen": "crops", "wrong": False}] * 5,
            unjudged,
            unjudged,
        ],
        "classes": [
            _class_row("water", 6, 6),
            _class_row("trees", 7, 6),
            _class_row("crops", 5, 5),
        ],
    }
    _answer(path, "class", {w: "?" for w in ALL})
    assert sheet.graded_squares(str(path), sheet.sheet_version(str(path))) is None


def test_a_sheet_labelled_right_or_wrong_still_has_no_squares(root: Path) -> None:
    """O2 changes only the sheets labelled with the class seen: a class map's
    sheet labelled right or wrong, Cannot tell included, has none."""
    path = _class_map(root)
    _answer(path, "wrong", {w: "?" if w < 2 else "0" for w in ALL})
    assert sheet.graded_squares(str(path), sheet.sheet_version(str(path))) is None


# ------------------------------------------------------------------------ F4


def test_graded_state_says_whether_the_sheet_holds_an_answer(root: Path) -> None:
    """F4 (contract B2 to B1): a graded sheet says answered True; the same
    sheet emptied by the panel's clear (Q1: the mark kept, blanked) says
    answered False, the rest of the state as r4-h gave it; answered again,
    True."""
    path = _class_map(root)
    _answer(path, "class", {w: CLASSES[CLASS_MAP[w]] for w in ALL})
    first = sheet.record_grading(str(path), sheet.sheet_version(str(path)))
    assert sheet.graded_state(str(path)) == {
        "first_at": first["first_at"],
        "changed_since_first": False,
        "changed": False,
        "answered": True,
    }
    _answer(path, "class", {}, clear_column="reference_class")
    assert sheet.graded_state(str(path)) == {
        "first_at": first["first_at"],
        "changed_since_first": True,
        "changed": True,
        "answered": False,
    }
    _answer(path, "class", {0: "water"})
    state = sheet.graded_state(str(path))
    assert state is not None and state["answered"] is True
