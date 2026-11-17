# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""The repairs of round r4-j (14 Nov 2026) not pinned beside their r4-i tests,
each string as specs/r4-j-round.md gives it; the view files are read as text
(no test here runs the scripts):

- J2: after a Check-card press with no reply, a part's line says only that
  its squares may or may not be picked; the gate's RULE_LOST stays the one
  instruction to open the card again, and the press's toast is unchanged;
- J3: the Details name of the classes nobody saw says what the reader saw,
  since a Cannot tell square may hold such a class (the F1 rule);
- J5: the look-first card's overline says "Look first", never "Review set".

J1 (the class-check message) and J4 (the review legend in the fold) are
pinned in test_mcp_r4i_views.py, J6 (the kelp-only line) in
test_mcp_r4i_sheet.py and test_mcp_figure_cards.py, beside the tests they
change."""

from __future__ import annotations

import re
from pathlib import Path

from olmoearth_agent.mcp_app import server, views

#: One token of a JS concatenation: a single-quoted string, a plus, or a name.
_TOKEN = re.compile(r"\s*(?:'((?:[^'\\]|\\.)*)'|(\+)|([A-Za-z_][A-Za-z0-9_.]*))")

#: A part's line after a press that got no reply (r4-j J2).
NO_ANSWER = (
    "The press got no answer back for this part, so its squares may or may not "
    "be picked."
)
#: The gate's one line then (r4-i F6), the one instruction.
RULE_LOST = "Ask Claude to open the Check card again to see the rule."
#: The press's toast then, unchanged.
PRESS_NO_ANSWER = (
    "The press got no answer back, so this card cannot say what was picked: a "
    "part's squares may be picked, or still being picked. Ask Claude to open the "
    "Check card again; it shows each part."
)

#: The look-first card's overline (r4-j J5).
LOOK_FIRST = "Look first"
REVIEW_TOOLS = ("olmoearth_review_set", "olmoearth_review_set_from_result")

#: Words no face holds (r4-i's rule), the overline included.
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


def _concat(expr: str) -> str:
    """The text a JS concatenation of single-quoted strings makes; a name
    in it fails."""
    out: list[str] = []
    text = expr.strip()
    at = 0
    while at < len(text):
        match = _TOKEN.match(text, at)
        assert match is not None, text[at:]
        literal, _plus, name = match.groups()
        assert name is None, name
        if literal is not None:
            out.append(re.sub(r"\\(.)", r"\1", literal))
        at = match.end()
    return "".join(out)


def _code(page: str) -> str:
    """The page with its comments taken out."""
    return re.sub(r"/\*.*?\*/|//[^\n]*", "", page, flags=re.S)


def _in_order(text: str, *pieces: str) -> bool:
    """Whether each piece is in text, each after the one before it."""
    at = [text.index(p) for p in pieces]
    return at == sorted(at)


def test_j2_a_parts_line_after_a_press_with_no_reply_asks_nothing() -> None:
    """r4-j J2: a part whose press got no answer says only that its squares
    may or may not be picked (marked), never to open the card again: the
    gate's RULE_LOST, set on the same press, is the one instruction; the
    press's toast PRESS_NO_ANSWER is word for word as before."""
    page = _read("check.html")
    assert f"// WORDING: r4j J2\nconst NO_ANSWER = '{NO_ANSWER}';" in page
    for ask in ("Ask Claude", "open the Check card", "again"):
        assert ask not in NO_ANSWER, ask
    assert f"// WORDING: r4i F6\nconst RULE_LOST = '{RULE_LOST}';" in page
    toast = _after(page, "const PRESS_NO_ANSWER = ", ";\n")
    assert _concat(toast) == PRESS_NO_ANSWER
    # Defined once, and drawn only on a part whose press got no answer.
    assert len(re.findall(r"\bNO_ANSWER\b", _code(page))) == 2
    zone = _between(page, "  function fillZone(m) {", "\n  }\n")
    unknown = (
        "    } else if (m.kind === 'unknown') {\n"
        "      parts.push(OE.el('div', {}, NO_ANSWER));\n"
    )
    assert unknown in zone
    # The press that makes its parts 'unknown' sets ruleLost, so the gate's
    # line shows beside each such part's line.
    press = _between(page, "  async function drawPicked() {", "\n  }\n")
    assert _in_order(
        press,
        "m.kind = 'unknown';",
        "ruleLost = true;",
        "OE.toast(PRESS_NO_ANSWER, true);",
    )
    assert _OFF_FACE.search(NO_ANSWER) is None


def test_j3_the_unseen_classes_row_says_what_the_reader_saw() -> None:
    """r4-j J3: the fold row (and the Details line) for the classes nobody
    saw says "Classes you saw in no {unit}" (marked), never that they were
    in no square checked; with Cannot tell answers too."""
    assert server.CLASS_UNSEEN_ROW == "Classes you saw in no {unit}"
    assert server.CLASS_UNSEEN_DETAIL == "Classes you saw in no {unit}: {names}"
    source = Path(server.__file__).read_text(encoding="utf-8")
    marked = (
        "# WORDING: r4j J3 (CLASS_UNSEEN_DETAIL's words, as a fold row's label)\n"
        'CLASS_UNSEEN_ROW = "Classes you saw in no {unit}"'
    )
    assert marked in source
    assert "not in any" not in source
    unseen = [
        {"code": f"u{i}", "name": f"Class {i}", "seen": 0, "agreed": 0}
        for i in range(6)
    ]
    classes = [{"code": "a", "name": "Water", "seen": 70, "agreed": 60}, *unseen]
    names = ", ".join(f'"Class {i}"' for i in range(6))
    for unit, k in (("square", 0), ("square", 2), ("point", 2)):
        counts = {"n": 72, "n_unjudged": k, "classes": classes}
        got = server._uncovered(counts, unit, 72, several=False)
        assert got["detail"] == f"Classes you saw in no {unit}: {names}", (unit, k)


def test_j5_the_look_first_cards_overline_says_look_first() -> None:
    """r4-j J5: both review tools' overline (result.html KIND, shown by
    kindOf in the overline) is "Look first", each marked; no "Review set"
    is left as a kind, and no kind says a word kept off faces."""
    page = _read("result.html")
    kinds = _between(page, "const KIND = {", "\n};")
    for tool in REVIEW_TOOLS:
        assert f"  // WORDING: r4j J5\n  {tool}: '{LOOK_FIRST}',\n" in kinds, tool
    assert "Review set" not in kinds
    values = re.findall(r":\s*'([^']*)'", _code(kinds))
    assert values.count(LOOK_FIRST) == 2
    for value in values:
        assert _OFF_FACE.search(value) is None, value
    overline = (
        "root.replaceChildren(OE.el('div', { class: 'overline' }, kindOf(data.tool)));"
    )
    assert overline in page
    assert "const kindOf = (tool) => KIND[tool] || " in page
