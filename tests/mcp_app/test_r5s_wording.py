# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Round r5-s (20 Nov 2026), as specs/r5-s-wording.md gives it: a card's text
says what the user sees, and never sounds like one of our rules. The rules
and the statistical cautions stay in the code, in the model's text (the
say-first lines, the notes, the tool's own must_state) and in the docs.

- W1: the result card's heading over the parts is "Parts of this map";
- W2: no line under the squares stands in place of the class lines when some
  squares were Cannot tell (the view's ``squares.no_split`` is ``None``, and
  the card no longer reads it);
- W3: the card's last line is what the map was compared with, and no more;
  the model's third say-first line keeps the whole caution;
- W4: the Check card's Details say "Each part gets its own result.";
- W5: the message of "Check each class the map draws" says "It gets its own
  result.";
- W6: the Details line beside that button ends "so it gets its own result.";
- W7: the face keeps "Check the other part too." and where; the model's
  lines keep their second sentence;
- W8: Details' line on the parts describes (the model's sentence on how to
  report them is on no card), and the label sheet's line on its answers says
  what they are. Kept on purpose: "Out of date, do not use ...", the
  labelling rule's own lines and the start card's line on keys.

The views are read as text (no test here runs the scripts); the browser
harness (scripts/mcp_views_e2e.py) pins the same words on the drawn cards.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

mcp = pytest.importorskip("mcp")

from olmoearth_agent.mcp_app import server, views  # noqa: E402

#: This round's marks.
MARK = "// WORDING: r5s"
MARK_PY = "# WORDING: r5s"

#: The staging tree's root (CHANGELOG.md, docs/).
ROOT = Path(__file__).resolve().parents[2]

#: Every view file a card is drawn from.
VIEW_FILES = sorted(
    p.name for p in views.VIEW_DIR.iterdir() if p.suffix in (".html", ".js")
)

#: W1: the heading over the parts.
PARTS_HEAD = "Parts of this map"
#: W3: the card's last line, with pictures served and without.
LAST_PICTURES = "Compared with what you saw in the satellite pictures."
LAST_ANSWERS = "Compared with your answers."
#: W4: the Check card's Details line on a map in parts ({n}: how many).
SEVERAL = (
    "Studio saved this map as {n} files, one per UTM zone (a north-south strip of "
    "the world map grid). Each part gets its own result. To cover the whole map, "
    "check every part. Pick a count for each part you want checked."
)
#: W5: the sentence of the second check's message, on squares and on points.
OWN_RESULT = "It gets its own result."
#: W6: the Details line beside the second check's button ({many}: its unit).
CLASS_CHECK_DETAIL = (
    "Check each class the map draws: a separate check with its own ranges. It "
    "picks {many} for each class the map draws, which need not be the classes you "
    "did not see. It is planned after this result was read, so it gets its own "
    "result."
)
#: W8: Details' line on the parts, and the label sheet's line on its answers.
SIDE_BY_SIDE = (
    "Each part has its own result: this check gives no range for a combined rate."
)
READING = "Your answers are a reading of satellite pictures."

#: The words the cards said before this round: in no view file (code or
#: comment), and in no string the server sends for a card's face.
GONE = (
    "One result per part",
    "never averaged",
    "No split by class",
    "Checked against",
    "not a site visit",
    "not a visit on site",
    "no verdict",
    "no single rate",
    "Do not average",
    "do not average",
    "never pooled",
    "Its own check and ranges",
    "is its own check",
    "Report the parts",
)

#: The model's third say-first line of a result read from answers (W3: the
#: caution the card's last line no longer carries).
SAY_LIMITS = (
    "This compares the map with the answers given, not a visit on site; it does "
    "not say whether the map is good enough."
)

#: One token of a JS concatenation: a single-quoted string, a plus, or a name
#: (a call with no parentheses inside its own, such as unitOf(unit), is one
#: name).
_TOKEN = re.compile(
    r"\s*(?:'((?:[^'\\]|\\.)*)'|(\+)|([A-Za-z_][A-Za-z0-9_.]*(?:\([^()]*\))?))"
)


def _read(name: str) -> str:
    """A file of views/, as it is on disk."""
    return (views.VIEW_DIR / name).read_text(encoding="utf-8")


def _between(text: str, start: str, end: str) -> str:
    """The part of text from start up to the first end after it."""
    at = text.index(start)
    return text[at : text.index(end, at)]


def _after(text: str, start: str, end: str) -> str:
    """The part of text after start, up to the first end after it."""
    at = text.index(start) + len(start)
    return text[at : text.index(end, at)]


def _code(js: str) -> str:
    """JS text without its comments (a block, or the rest of a line after
    ``//``), each run of joined single-quoted literals read as one and escaped
    quotes unescaped."""
    bare = re.sub(r"/\*.*?\*/", "", js, flags=re.S)
    bare = re.sub(r"//[^\n]*", "", bare)
    return re.sub(r"'\s*\+\s*'", "", bare).replace("\\'", "'")


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


def _entry(block: str, start: str) -> str:
    """One entry's value in a JS object literal: from start to the comma that
    ends its last line."""
    at = block.index(start) + len(start)
    end = block.find(",\n", at)
    return (block[at:] if end < 0 else block[at:end]).strip().rstrip(",")


def _result_words() -> str:
    """The result card's table of words (result.html RESULT_WORDS)."""
    return _between(_read("result.html"), "const RESULT_WORDS = {", "\n};")


def _sheet_words() -> str:
    """The label sheet's table of words (sheet.js OE.sheetWords)."""
    return _between(_read("sheet.js"), "OE.sheetWords = (unit) => {", "\n};")


def _estimate(**extra: Any) -> dict[str, Any]:
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


# --------------------------------------------------------------------------- the views


def test_r5s_no_view_file_holds_the_old_words() -> None:
    """No view file says one of the lines this round took off the cards, in
    its code or in a comment (a JS string's escaped quote read as a quote)."""
    assert {"result.html", "check.html", "common.js", "sheet.js"} <= set(VIEW_FILES)
    for name in VIEW_FILES:
        page = _read(name).replace("\\'", "'")
        for gone in GONE:
            assert gone not in page, (name, gone)


def test_w1_the_parts_heading_names_what_is_under_it() -> None:
    """W1: RESULT_WORDS.partsHead is "Parts of this map", marked, and the
    card still draws it as the heading over the parts figure."""
    words = _result_words()
    assert f"  {MARK}\n  partsHead: '{PARTS_HEAD}',\n" in words
    card = _between(_read("result.html"), "function resultCard(app, data) {", "\n}\n")
    assert "OE.el('h3', {}, RESULT_WORDS.partsHead), OE.fig.parts(parts," in card


def test_w2_the_card_draws_no_line_in_place_of_the_class_lines() -> None:
    """W2: result.html no longer reads squares.no_split; under the squares'
    heading the card draws the figure and nothing more."""
    page = _read("result.html")
    assert "no_split" not in page and "oe-no-split" not in page
    card = _between(page, "function resultCard(app, data) {", "\n}\n")
    assert (
        "RESULT_WORDS.squaresHead(n !== null ? n : sq.cells.length, unit)),"
        " OE.fig.squares(sq)));\n" in card
    )


def test_w4_the_check_cards_details_say_each_part_gets_its_own_result() -> None:
    """W4: check.html SEVERAL keeps its first sentence (how many files) and
    then says "Each part gets its own result."; marked; it is still the
    Details' line of a map in two parts or more."""
    page = _read("check.html")
    assert f"{MARK}\n// WORDING: squares\nconst SEVERAL = (n) => " in page
    said = _concat(_after(page, "const SEVERAL = (n) => ", ";\n"), {"n": "2"})
    assert said == SEVERAL.format(n="2")
    assert said.startswith("Studio saved this map as 2 files, one per UTM zone (")
    assert " Each part gets its own result. " in said
    assert "if (nParts > 1) nodes.push(OE.el('p', {}, SEVERAL(nParts)));" in page


def test_w5_the_second_checks_message_says_it_gets_its_own_result() -> None:
    """W5: OE.JOB_TEXT.byClass says "It gets its own result." on points and on
    squares, right before it asks how many; marked above the entry's own
    mark."""
    jobs = _between(_read("common.js"), "OE.JOB_TEXT = {", "\n};")
    assert f'  {MARK} (both: "{OWN_RESULT}")\n  // WORDING: r4j J1\n' in jobs
    said = _code(_after(jobs, "  byClass: (r) => ", ",\n  // WORDING: "))
    assert said.count(OWN_RESULT) == 2
    assert f"by result_id). {OWN_RESULT} First ask me how many points" in said
    assert f"). {OWN_RESULT} First ask me how many squares to check." in said


def test_w6_the_second_checks_details_line_ends_with_its_own_result() -> None:
    """W6: RESULT_WORDS.classCheckDetail, of squares and of points, ends "so
    it gets its own result."; marked above the entry's own mark."""
    words = _result_words()
    marks = f'  {MARK} (its last clause: "so it gets its own result.")\n'
    assert marks + "  // WORDING: r4i O3\n  classCheckDetail: (unit) => " in words
    detail = _entry(words, "classCheckDetail: (unit) => ")
    for unit in ("square", "point"):
        shown = _concat(detail, {"unitOf(unit)": unit})
        assert shown == CLASS_CHECK_DETAIL.format(many=unit + "s"), unit
        assert shown.endswith("so it gets its own result.")


def test_w7_the_face_keeps_check_the_other_part_too() -> None:
    """W7: the face's line under the parts is "Check the other part too." and
    where that check is started, as before; the second sentence of the
    server's SAY_OTHER and SAY_OTHERS (how to report the parts) is the
    model's, on no card."""
    words = _result_words()
    after = _after(words, "otherPart: (k) => (k === 1\n    ? ", "),\n")
    one, many = after.split("\n    : ")
    ask = " Ask Claude to open the Check card again."
    assert _concat(one, {}) == "Check the other part too." + ask
    assert _concat(many, {"k": "3"}) == "Check the other 3 parts too." + ask
    told = "Report the parts' results side by side; do not average or add them"
    for said in (server.SAY_OTHER, server.SAY_OTHERS, server.SAY_SIDE_BY_SIDE):
        assert told in said, said
    assert server.SAY_OTHER.startswith(" Check the other part too. ")
    assert told not in _read("result.html").replace("\\'", "'")


def test_w8_details_line_on_the_parts_describes() -> None:
    """W8: RESULT_WORDS.sideBySide says that each part has its own result and
    why no one rate covers them, with the reason r4-m M2 gave; marked; drawn
    in Details on a map of two parts or more, as before."""
    page = _read("result.html")
    words = _result_words()
    marks = f"  {MARK}\n  // WORDING: r5e R4 (in Details)\n  sideBySide: '"
    assert marks in words
    side = _concat(_entry(words, "sideBySide: "), {})
    assert side == SIDE_BY_SIDE
    assert side.endswith(": this check gives no range for a combined rate.")
    for rule in ("side by side", "do not", "never", "average"):
        assert rule not in side, rule
    assert (
        "if (parts.length > 1) folded.push(OE.el('p',"
        " { class: 'small oe-side-by-side' }, RESULT_WORDS.sideBySide));" in page
    )


def test_w8_the_sheets_line_on_its_answers_says_what_they_are() -> None:
    """W8: OE.sheetWords' reading is "Your answers are a reading of satellite
    pictures."; marked above the entry's own mark."""
    words = _sheet_words()
    marks = f"    {MARK}\n    // WORDING: r5b J13\n    reading: '{READING}',\n"
    assert marks in words


def test_w8_the_warnings_and_the_labelling_rule_are_kept() -> None:
    """Kept on purpose: the withdrawn result's "Out of date, do not use" (the
    page's words and the server's), the label sheet's own rule lines (no
    guess; no change after the press) and the start card's line on keys."""
    result = _code(_result_words())
    assert "'Out of date, do not use (computed ' + at + ' from the" in result
    assert "'Out of date, do not use: ' + headline" in result
    assert server.OUT_OF_DATE_RATE.startswith("Out of date, do not use (computed ")
    assert server.OUT_OF_DATE_RATE_UNTIMED == "Out of date, do not use: {headline}"
    sheet_words = _code(_sheet_words())
    assert "Press Cannot tell; do not guess.'" in sheet_words
    assert "everyDone: 'After you press, do not change an answer.'," in sheet_words
    assert "Do not guess: a guess can make the result wrong." in sheet_words
    assert "Do not use someone else\\'s key.';" in _read("start.html")


# --------------------------------------------------------------------------- the server


def test_w2_the_server_sends_no_line_in_place_of_the_class_lines() -> None:
    """W2: the server has no no-split sentence; with a Cannot tell answer the
    squares' ``no_split`` is ``None``, no class has a line, and the legend
    still names the dashed square; without one, the same key, ``None``."""
    assert not hasattr(server, "NO_SPLIT") and not hasattr(server, "NO_SPLIT_ONE")
    got = {
        "n": 9,
        "n_unjudged": 1,
        "face": "class",
        "cells": [{"seen": "a", "wrong": False}] * 6
        + [{"seen": "a", "wrong": True}] * 2
        + [{"seen": None, "wrong": False, "unjudged": True}],
        "classes": [{"code": "a", "name": "Water", "seen": 8, "agreed": 6}],
    }
    for unit in ("square", "point"):
        one = server._squares(got, {}, unit)
        assert one is not None
        assert one["no_split"] is None and one["n_unjudged"] == 1
        assert [c["line"] for c in one["classes"]] == [None]
        assert {"unjudged": True, "text": server.LEGEND_CANNOT} in one["legend"]
        assert "No split" not in json.dumps(one)
    plain = server._squares({**got, "n_unjudged": 0, "cells": got["cells"][:8]}, {})
    assert plain is not None
    assert (plain["no_split"], plain["n_unjudged"]) == (None, 0)


def test_w3_the_last_line_says_what_the_map_was_compared_with() -> None:
    """W3: the card's last line, with pictures and without, is one plain
    sentence, marked; an estimate's view carries it, and nothing the view
    holds says one of the old lines."""
    assert server.LAST_LINE_PICTURES == LAST_PICTURES
    assert server.LAST_LINE_ANSWERS == LAST_ANSWERS
    source = Path(server.__file__).read_text(encoding="utf-8")
    assert f"{MARK_PY}\nLAST_LINE_PICTURES = " in source
    assert f"{MARK_PY}\nLAST_LINE_ANSWERS = " in source
    two = {"population": {"class_names": {"0": "absent", "1": "present"}}}
    told = server._estimate_result(_estimate(), None, None, design=two)
    assert told is not None
    assert told["view"]["last_line"] == LAST_ANSWERS
    shown = json.dumps(told["view"])
    for gone in GONE:
        assert gone not in shown, gone


def test_w3_the_models_lines_keep_the_cautions() -> None:
    """The model-facing text is as it was: the third say-first line says the
    comparison is no visit on site and no verdict on the map; the lines on
    the parts say to report them side by side and not to average them, with
    the reason; the Check card's note says the same before any result."""
    two = {"population": {"class_names": {"0": "absent", "1": "present"}}}
    told = server._estimate_result(_estimate(), None, None, design=two)
    assert told is not None
    assert told["say_first"][2] == SAY_LIMITS
    assert server.SAY_LIMITS == (
        "This compares the map with {seen}, not a visit on site; it {whole}does not "
        "say whether the map is good enough."
    )
    assert server.SAY_SEEN_PICTURES == "what was seen in satellite pictures"
    assert server.SAY_NOT_WHOLE == "is not the whole map's rate and "
    reason = ": this check gives no range for a combined rate"
    assert server.SAY_SIDE_BY_SIDE == (
        " Report the parts' results side by side; do not average or add them"
        + reason
        + "."
    )
    assert server.SAY_OTHER.endswith(server.SAY_SIDE_BY_SIDE)
    assert server.SAY_OTHERS.endswith(
        " Report the parts' results side by side; do not average or add them"
        + reason
        + "."
    )
    assert server.SAY_CHECK_NONE_PARTS == (
        "; each part gets its own result, never one rate for all parts. To cover the "
        "whole map, report each part's rate with its range, side by side; do not "
        "average or add them up" + reason
    )
    assert (
        "Each part is its own map: never give one rate for all parts, and do not "
        "average or add the parts' rates." in server.CHECK_PREPARING_RULES
    )


# --------------------------------------------------------------------------- the docs


def test_r5s_the_changelog_and_the_guide_say_the_new_words() -> None:
    """The CHANGELOG's Unreleased part has one line under Fixes; the panel
    guide quotes the result card's new last line and its parts heading, and
    keeps the cautions (the docs are where they stay)."""
    log = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    newest = _between(log, "## [Unreleased]", "\n## [1.5.3]")
    fixes = newest[newest.index("### Fixes\n") :]
    assert "- Cards say what you see in plain words: " in fixes
    for said in (PARTS_HEAD, LAST_PICTURES, LAST_ANSWERS, READING):
        assert f'"{said}"' in fixes, said
    assert '"Each part gets its own result."' in fixes
    guide = (ROOT / "docs" / "mcp-app.md").read_text(encoding="utf-8")
    result = " ".join(_between(guide, "## Result card", "\n## ").split())
    assert f'"{LAST_PICTURES}" (or "{LAST_ANSWERS}")' in result
    assert f'the heading "{PARTS_HEAD}"' in result
    assert "not a site visit, and no verdict on whether the map is good" in result
    assert "side by side; never average them." in result
    assert "not a site visit; the card gives no verdict" not in result
