# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Round r5-l (19 Nov 2026), track VIEWS, as specs/r5-l-round.md gives it:
L2, the faces say it generally. The owner, 19 Nov 2026: "our instructions
should be general; we shouldn't use kelp to state some question [we] ask for
people to eval".

- L2 (a): the Check card's box beside the app's rule asks "Did the map maker
  mean something else? Write their meaning of Present and Absent
  (optional)." (RULE_OWN; it asked "Not a kelp map? ..."); the line under
  the box says "Left empty, the rule above is used." (RULE_EMPTY); and 1 to
  9 characters in the box give "Write at least 10 characters in the box, or
  empty it to use the rule above." (BOX_SHORT) under the press and as the
  press's toast. No map is named in any of them.
- L2 (b): the two example pictures under the rule keep their pictures; their
  captions say what Present and Absent mean in general words (what the map
  shows covers more than half of the square on a clear date, or half of the
  square or less on every clear date), on the Check card and on the label
  sheet word for word, and one small grey line under them, in the same
  block, says where they come from: "Example pictures from a kelp canopy
  map.".
- L2 (c): no other line of the views' own words names kelp: once comments
  are taken out, that source line is the only text in views/*.html and
  views/*.js that says "kelp" (or "canopy").

The views are read as text (no test here runs the scripts). The browser
harness (scripts/mcp_views_e2e.py) draws them: 12_check_card and
12g_check_owner_rule_box (the box's label, the line under it and, with 5
characters typed, BOX_SHORT under the press and as the toast),
12j_check_example_pictures and 16d_sheet_example_pictures (the captions and
the source line under the pictures, on the card and on the sheet), and
12t_check_examples_with_box (the pictures with their source line, then the
box, on one card, with BOX_SHORT once 5 characters are typed).
"""

from __future__ import annotations

import re

from olmoearth_agent.mcp_app import views

#: L2 (a): the box's label, the line under the box, and the line under the
#: press (and the press's toast) with 1 to 9 characters in the box.
RULE_OWN = (
    "Did the map maker mean something else? Write their meaning of Present and "
    "Absent (optional)."
)
RULE_EMPTY = "Left empty, the rule above is used."
BOX_SHORT = (
    "Write at least 10 characters in the box, or empty it to use the rule above."
)

#: L2 (b): the two captions and the one line on where the pictures come from.
#: Round r5-m B1 words the captions of what the map is meant to find (they
#: said "what the map shows"; tests/mcp_app/test_mcp_r5m_check.py), marked
#: above the tags this test reads.
EXAMPLE_PRESENT = (
    "Present: what the map is meant to find covers more than half of the square "
    "on a clear date."
)
EXAMPLE_ABSENT = (
    "Absent: what the map is meant to find covers half of the square or less on "
    "every clear date."
)
EXAMPLES_SOURCE = "Example pictures from a kelp canopy map."

#: The words those lines had before this round, which no view holds now.
OLD = (
    "Not a kelp map?",
    "Left empty, the kelp rule above is used.",
    "or empty it to use the kelp rule.",
    "Present under this rule",
    "Absent under this rule",
)

#: Words no face holds (round r5-e's rule R7, and the owner's ruling 6).
_OFF_FACE = re.compile(
    r"\b(?:windows?|UTM|zones?|intervals?|strat(?:um|a)|design|budget"
    r"|recommended)\b",
    re.I,
)

#: A word that names the one kind of map the app's rule was first set for.
_KELP = re.compile(r"kelp|canopy", re.I)


def _read(name: str) -> str:
    """A file of views/, as it is on disk."""
    return (views.VIEW_DIR / name).read_text(encoding="utf-8")


def _between(text: str, start: str, end: str) -> str:
    """The part of text from start up to the first end after it."""
    at = text.index(start)
    return text[at : text.index(end, at)]


def _code(text: str) -> str:
    """The text with its comments taken out: JS and CSS block comments, JS
    line comments, and HTML comments."""
    return re.sub(r"/\*.*?\*/|<!--.*?-->|//[^\n]*", "", text, flags=re.S)


def _in_order(text: str, *pieces: str) -> bool:
    """Whether each piece is in text, each after the one before it."""
    at = [text.index(p) for p in pieces]
    return at == sorted(at)


def _views() -> list[str]:
    """The names of every page and script of views/, sorted."""
    return sorted(
        p.name for p in views.VIEW_DIR.iterdir() if p.suffix in (".html", ".js")
    )


# --------------------------------------------------------------------------- L2 (a)


def test_l2_the_box_beside_the_rule_names_no_map() -> None:
    """L2 (a): the box's label, the line under the box and the line under
    the press are the spec's words, each one plain string marked
    ``// WORDING: r5l L2`` above the tags it had; none names a map, none
    holds a word a face does not say, and the old words are gone."""
    page = _read("check.html")
    own = (
        "// WORDING: r5l L2\n// WORDING: r5b J18\n// WORDING: r4m M5\n"
        f"// WORDING: r4h C5\nconst RULE_OWN = '{RULE_OWN}';\n"
    )
    assert own in page
    empty = (
        "// WORDING: r5l L2\n// WORDING: r5b J18\n"
        f"const RULE_EMPTY = '{RULE_EMPTY}';\n"
    )
    assert empty in page
    short = (
        "// WORDING: r5l L2\n// WORDING: r5b J08\n"
        f"const BOX_SHORT = '{BOX_SHORT}';\n"
    )
    assert short in page
    for said in (RULE_OWN, RULE_EMPTY, BOX_SHORT):
        assert _KELP.search(said) is None, said
        assert _OFF_FACE.search(said) is None, said
    for gone in OLD[:3]:
        assert gone not in page, gone
    # Short, as r5-b J18 wanted (the label held 37 words then).
    assert len(RULE_OWN.split()) <= 15


def test_l2_the_box_words_stand_where_they_stood() -> None:
    """L2 (a): only the words change. The box keeps its label in body type
    and the line under it (r5-b J18), the press's line says BOX_SHORT while
    the box holds 1 to 9 characters (r5-b J08), and a press held by the box
    puts the focus in the box (round r5-m B3, after review: it toasted the
    same words too, right under that line)."""
    page = _read("check.html")
    gate = _between(page, "  function fillGate() {", "\n  }\n")
    assert "gate.replaceChildren(...ruleLines(), boxed(RULE_OWN, RULE_EMPTY));" in gate
    why = _between(page, "  function pressWhy(plan) {", "\n  }\n")
    assert "if (gateMode === 'replace' && gateHeld()) return BOX_SHORT;" in why
    press = _between(page, "  async function drawPicked() {", "\n  }\n")
    held = "if (gateHeld()) { updatePress(); OE.focusIfHere(ruleInput); return; }"
    assert held in press
    code = _code(page)
    for name in ("RULE_OWN", "RULE_EMPTY"):
        assert len(re.findall(rf"\b{name}\b", code)) == 2, name
    assert len(re.findall(r"\bBOX_SHORT\b", code)) == 2


# --------------------------------------------------------------------------- L2 (b)


def test_l2_the_check_cards_captions_are_general_and_the_source_is_under() -> None:
    """L2 (b) on the Check card: the two captions in general words, each
    marked above its old tag, still the pictures' alt text; one small line
    under the two pictures, in the same block (oe-examples-set) and right
    after them, says where they come from, 4 px under the captions; the
    place line stays in Details, as before."""
    page = _read("check.html")
    for name, said in (
        ("EXAMPLE_PRESENT", EXAMPLE_PRESENT),
        ("EXAMPLE_ABSENT", EXAMPLE_ABSENT),
    ):
        tags = f"// WORDING: r5l L2\n// WORDING: r4k K9\nconst {name} = '{said}';\n"
        assert tags in page, name
    source = f"// WORDING: r5l L2\nconst EXAMPLES_SOURCE = '{EXAMPLES_SOURCE}';\n"
    assert source in page
    assert EXAMPLE_PRESENT.startswith("Present: ")
    assert EXAMPLE_ABSENT.startswith("Absent: ")
    for said in (EXAMPLE_PRESENT, EXAMPLE_ABSENT):
        assert _KELP.search(said) is None, said
        assert "rule" not in said, said
    for said in (EXAMPLE_PRESENT, EXAMPLE_ABSENT, EXAMPLES_SOURCE):
        assert _OFF_FACE.search(said) is None, said
    for gone in OLD[3:]:
        assert gone not in page, gone
    block = _between(page, "function examplesBlock(ex) {", "\n}\n")
    assert "OE.el('img', { src, alt: caption })," in block
    assert _in_order(
        block,
        "return OE.el('div', { class: 'oe-examples-set' },",
        "OE.el('div', { class: 'oe-examples' }, one(ex.present, EXAMPLE_PRESENT),"
        " one(ex.absent, EXAMPLE_ABSENT)),",
        "OE.el('div', { class: 'small muted oe-examples-from' }, EXAMPLES_SOURCE));",
    )
    assert "  .oe-examples-from { margin-top: 4px; }\n" in page
    # Defined once and said once: under the pictures, never in Details.
    assert _code(page).count("EXAMPLES_SOURCE") == 2
    fold = _between(page, "  function fillFold() {", "\n  }\n")
    assert "EXAMPLES_SOURCE" not in fold
    assert (
        "if (examplesShown()) nodes.push(OE.el('p', {}, EXAMPLES_FROM(examples.year)));"
        in fold
    )


def test_l2_the_sheet_says_the_same_captions_and_source_line() -> None:
    """L2 (b) on the label sheet: the same two captions and the same source
    line, word for word, each marked; the pictures and the line are one
    block (oe-examples-set) at the end of the rule's box, the line 4 px
    under the pictures and grey (muted), as on the Check card; each figure
    is 120 px wide, as on the Check card, so a caption of a sentence wraps
    under its picture."""
    script = _read("sheet.js")
    for name, said in (
        ("OE.EXAMPLE_PRESENT", EXAMPLE_PRESENT),
        ("OE.EXAMPLE_ABSENT", EXAMPLE_ABSENT),
    ):
        tags = f"// WORDING: r5l L2\n// WORDING: r4k K9\n{name} = '{said}';\n"
        assert tags in script, name
    source = f"// WORDING: r5l L2\nOE.EXAMPLES_SOURCE = '{EXAMPLES_SOURCE}';\n"
    assert source in script
    for gone in OLD[3:]:
        assert gone not in script, gone
    pictures = _between(script, "OE.examplePictures = (ex) => ", ";\n")
    assert _in_order(
        pictures,
        "OE.el('div', { class: 'oe-examples-set' },",
        "OE.el('div', { class: 'row oe-examples',",
        "[[ex.present, OE.EXAMPLE_PRESENT], [ex.absent, OE.EXAMPLE_ABSENT]]",
        "OE.el('figure', { style: 'margin:0;width:120px;max-width:100%' },",
        "OE.el('figcaption', { class: 'small', 'aria-hidden': 'true' }, caption)",
        "OE.el('div', { class: 'small muted oe-examples-from', style: "
        "'margin-top:4px' }, OE.EXAMPLES_SOURCE)",
    )
    # The block still ends the rule's box (r4-k K9, r5-e H2).
    editor = _between(script, "OE.sheetEditor = (app, data,", "\n};")
    rule = _between(editor, "  const ruleBlock = pageRule ?", "\n  const clearNote")
    assert rule.endswith("W.cloudLine) : null,\n    ruleExamples) : null;")
    assert _code(script).count("OE.EXAMPLES_SOURCE") == 2


def test_l2_the_two_cards_word_the_pictures_alike() -> None:
    """L2 (b): the Check card and the sheet never word the example pictures
    differently: the captions and the source line are the same strings in
    check.html and sheet.js."""
    page = _read("check.html")
    script = _read("sheet.js")
    for said in (EXAMPLE_PRESENT, EXAMPLE_ABSENT, EXAMPLES_SOURCE):
        assert f"'{said}'" in page, said
        assert f"'{said}'" in script, said


# --------------------------------------------------------------------------- L2 (c)


def test_l2_no_line_of_the_views_own_words_names_kelp_but_the_source() -> None:
    """L2 (c): with comments taken out, the only text of views/*.html and
    views/*.js that says "kelp" or "canopy" is the pictures' source line,
    once on the Check card and once on the sheet (where the pictures come
    from, never the question a person is asked)."""
    named: dict[str, list[str]] = {}
    for name in _views():
        lines = [
            line.strip()
            for line in _code(_read(name)).splitlines()
            if _KELP.search(line)
        ]
        if lines:
            named[name] = lines
    assert named == {
        "check.html": [f"const EXAMPLES_SOURCE = '{EXAMPLES_SOURCE}';"],
        "sheet.js": [f"OE.EXAMPLES_SOURCE = '{EXAMPLES_SOURCE}';"],
    }


def test_l2_the_box_and_the_captions_hold_no_map_name_on_any_view() -> None:
    """L2 (c): none of the old instruction words that named kelp is left in
    any view, comments included where they quote a face (the old label, the
    old line under the box, the old line under the press, the old
    captions)."""
    for name in _views():
        text = _read(name)
        for gone in OLD:
            assert gone not in text, (name, gone)


def test_l2_no_view_reads_the_kelp_only_line() -> None:
    """L1 and L2 (c): the server sends no rule_limit since round r5-l (the
    line that any other map needs its own rule), so no view reads one or
    keeps the sheet's line for it (limitNote), comments included."""
    for name in _views():
        text = _read(name)
        for gone in ("data.rule_limit", "answer.rule_limit", "limitNote"):
            assert gone not in text, (name, gone)
