# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Round r5-k (18 Nov 2026), track CHECK, as specs/r5-k-round.md gives it:
K6, the owner's choice on O22 (specs/r5-i-audit/owner_decisions.md and
specs/r5-g-audit/owner_decisions.md: "fold the steps and picture into 'How
this check works' once a part is picked").

- K6: before any part of the Check card is picked, the card shows the four
  framed step panels and the strips picture as before. Once a part is drawn
  (its squares picked, by a press on the card or before it opened), both
  go, the same nodes, into one closed fold, "How this check works", placed
  after the parts and the press or the sheet open in the card, right before
  Details, so the parts and their sheets come first; the fold opens to show
  the same panels and picture. A map in one part folds its steps alone; the
  plain parts line of parts that are not one per strip holds no picture and
  stays where it is. A card whose map is being prepared keeps its steps.

The view is read as text (no test here runs the scripts). The browser
harness (scripts/mcp_views_e2e.py) draws it: 12_check_card before the press
(no fold, the steps on the face) and after it (the fold closed, then opened
at 780 and 380 px, then closed again), 13_check_repeat (a part drawn before
the card opened; the fold, opened, stays open through a press and its sheet
opening), 13b_check_all_drawn (a map in one part: the steps alone),
13f_check_one_sheet_of_parts (parts that are not one per strip, one drawn:
the steps alone in the fold, the plain parts line on the face, under the
question's line), 12p_check_preparing_ready (the card that filled in, no
part drawn: no fold), and 14_check_failed_draw and 14d_check_press_no_answer
(presses that drew nothing: no fold).
"""

from __future__ import annotations

import re

from olmoearth_agent.mcp_app import views

#: K6: the fold's words.
HOW = "How this check works"

#: Words no face holds (round r5-e's rule R7, and the owner's ruling 6).
_OFF_FACE = re.compile(
    r"\b(?:windows?|UTM|zones?|intervals?|strat(?:um|a)|design|budget"
    r"|recommended)\b",
    re.I,
)


def _read(name: str) -> str:
    """A file of views/, as it is on disk."""
    return (views.VIEW_DIR / name).read_text(encoding="utf-8")


def _between(text: str, start: str, end: str) -> str:
    """The part of text from start up to the first end after it."""
    at = text.index(start)
    return text[at : text.index(end, at)]


def _code(page: str) -> str:
    """The page with its comments taken out."""
    return re.sub(r"/\*.*?\*/|//[^\n]*", "", page, flags=re.S)


def _in_order(text: str, *pieces: str) -> bool:
    """Whether each piece is in text, each after the one before it."""
    at = [text.index(p) for p in pieces]
    return at == sorted(at)


def _check_card(page: str) -> str:
    """check.html's checkCard, whole."""
    return _between(page, "function checkCard(", "\n}\n")


def _fold_how(page: str) -> str:
    """check.html's foldHow, whole but its closing brace."""
    return _between(page, "  function foldHow() {", "\n  }\n")


# --------------------------------------------------------------------------- K6


def test_k6_the_folds_words_are_plain_and_marked() -> None:
    """K6: the fold is named "How this check works", one plain string marked
    ``// WORDING: r5k K6``, with no word a face does not say; it is used
    once, as the fold's summary."""
    page = _read("check.html")
    assert "// WORDING: r5k K6\nconst HOW = 'How this check works';\n" in page
    assert re.findall(r"// WORDING: r5k (K\d+)", page).count("K6") == 1
    assert _OFF_FACE.search(HOW) is None
    assert len(re.findall(r"\bHOW\b", _code(page))) == 2
    assert "OE.el('summary', {}, HOW)" in page


def test_k6_the_fold_is_one_closed_details_of_its_own_class() -> None:
    """K6: the fold is a details element of its own class and id, made
    without the open attribute (so closed), its summary the fold's words, and
    no statement of the card's code opens it (only the user does). It is not
    made by OE.fold (class oe-more), so Details stays the card's one oe-more
    fold and its last block; it is spaced as Details is (8 px under its
    summary, 12 px between what it holds)."""
    page = _read("check.html")
    made = (
        "  const how = OE.el('details', { class: 'oe-how', id: 'oe-check-how' }, "
        "OE.el('summary', {}, HOW));\n"
    )
    assert made in page
    assert "open" not in made
    opens = r"\bhow\.(?:open\b|(?:set|toggle)Attribute\(\s*['\"]open)"
    assert re.search(opens, _code(_check_card(page))) is None
    assert "OE.fold(HOW" not in page
    for rule in (
        "  .oe-how > summary + * { margin-top: 8px; }\n",
        "  .oe-how > * + * { margin-top: 12px; }\n",
    ):
        assert rule in page, rule


def test_k6_the_steps_and_the_strips_card_fold_once_a_part_is_drawn() -> None:
    """K6: foldHow does nothing until a part is drawn, nor a second time (the
    fold, made once, stays where it is, and open if the user opened it);
    then it moves the step panels and the strips card, the same nodes, into
    the fold, and puts the fold after the card's action (the press, or the
    sheet open here), so before Details: since round r5-m B3 right after the
    status line, which sits right under the action
    (tests/mcp_app/test_mcp_r5m_check.py). Only the strips card moves: the
    plain parts line (class card alone, no picture) stays on the face."""
    page = _read("check.html")
    fold = _fold_how(page)
    # Round r5-m B3, after review: nor once show() replaced the card (its
    # action off the page; tests/mcp_app/test_mcp_r5m_check.py).
    assert (
        "    if (how.isConnected || !action.isConnected "
        "|| !models.some((m) => m.kind === 'drawn')) return;\n" in fold
    )
    assert (
        "    const moved = [steps, root.querySelector(':scope > .card.oe-strips')]"
        ".filter(Boolean);\n" in fold
    )
    assert "    if (!moved.length) return;\n" in fold
    assert _in_order(
        fold, "    how.append(...moved);\n", "    OE.toastRegion().after(how);"
    )
    plain = (
        "root.append(OE.el('div', { class: 'card', id: 'oe-check-parts-line' }, "
        "OE.el('div', {}, PARTS_PLAIN(nParts))));"
    )
    assert plain in page
    assert "oe-strips" not in plain


def test_k6_every_render_folds_after_the_action_is_drawn() -> None:
    """K6: every render of the card (renderAll: on open, after a press was
    answered, after a sheet opened) runs foldHow once the press or the
    editor is in place (updatePress) and before Details and Claude's line,
    so a part drawn by a press folds the steps at once, and a card opened
    with a part drawn shows the fold from the start. Nothing else calls it."""
    page = _read("check.html")
    render = _between(page, "  function renderAll() {", "\n  }\n")
    assert _in_order(
        render,
        "    updatePress();\n",
        "    foldHow();\n",
        "    fillFold();\n",
        "    tellCard();",
    )
    # Its definition, and the one call in renderAll.
    assert _code(page).count("foldHow()") == 2


def test_k6_before_any_pick_the_face_is_as_before() -> None:
    """K6: the step panels and the strips card are drawn where they were, on
    the face (the steps under the title and the map's line, before the rule;
    the strips card under the question, before the parts), and the fold is
    made only once the face is built, outside the page until foldHow puts it
    there, so a card with no part drawn has no fold."""
    page = _read("check.html")
    card = _check_card(page)
    assert _in_order(
        card,
        "\n  const steps = fig('steps', [firstStep(models), ...STEPS.slice(1)]);\n",
        "\n  if (steps) root.append(steps);\n",
        "\n  root.append(gate);\n",
        "\n  root.append(moreLine);\n",
        "\n    root.append(OE.el('div', { class: 'card oe-strips', ",
        "\n  for (const m of models) {\n",
        "\n  root.append(action, OE.fold(DETAILS, foldBody));\n",
        "\n  action.after(OE.toastRegion());\n",
        "\n  const how = OE.el('details', ",
        "\n  function foldHow() {\n",
        "\n  renderAll();\n",
    )
    # r5-m B3: the fold goes after the status line, which is right under the
    # action (tests/mcp_app/test_mcp_r5m_check.py).
    assert card.count("OE.toastRegion().after(how)") == 1
    assert card.count("action.after(how)") == 0
    assert card.count("root.append(how") == 0


def test_k6_a_preparing_card_keeps_its_steps() -> None:
    """K6: a card whose map is being prepared (round r5-h) draws its four
    steps as before and has no fold: nothing on it is picked yet."""
    page = _read("check.html")
    card = _between(page, "function preparingCard(app, data) {", "\n}\n")
    steps = (
        "  const steps = fig('steps', [...STEPS]);\n"
        "  if (steps) root.append(steps);\n"
    )
    assert steps in card
    assert re.search(r"\bHOW\b|foldHow|oe-how", card) is None


def test_k6_the_served_check_view_holds_the_fold() -> None:
    """K6: the check view as the server serves it (views.view_html) holds the
    fold's words, its making and its spacing."""
    page = views.view_html("check")
    assert "const HOW = 'How this check works';" in page
    assert "id: 'oe-check-how'" in page
    assert ".oe-how > summary + * { margin-top: 8px; }" in page
