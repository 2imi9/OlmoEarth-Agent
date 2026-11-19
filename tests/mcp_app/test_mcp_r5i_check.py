# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Round r5-i (18 Nov 2026), track CHECK, on the MCP App: the Check card's
items of the re-audit (specs/r5-g-audit/build_list.md B1 to B3) and the
card's side of B10 (specs/r5-i-round.md).

- B1: a picked part whose sheet could not be read when the card opened
  says, before any press, to press its button, by the button's own label in
  double quotes; the red line that sends the user to Claude comes only once
  a press failed too, and the red line of a press that failed names the
  button the same way; a part drawn again while a press is answered or its
  sheet opens keeps its open button off;
- B2: with every square answered the heading says to open each part's sheet
  in turn where more than one part has a sheet, else the sheet;
- B3: a part that cannot be read as a map says on its face, in plain words,
  that the app could not cut its map into squares; the server's reason
  (patch, px, zone) is in the part's lines in Details;
- B10: a preparing card of parts whose status answers ready passes the
  status's ``model_text`` (what olmoearth_check_map would have answered in
  time) to Claude once, silently, after the card's parts line: whole, in a
  text block of its own after the panel's facts, under its own lead
  (READY_TOLD), never cut and outside CONTEXT_MAX, so must_state and
  forbidden_claims reach Claude; none of it is on the card, and a card drawn
  in its place drops it.

The views are read as text (no test here runs the scripts); a JS
concatenation of single-quoted strings and names is read back into its text
(:func:`_concat`). The browser harness (scripts/mcp_views_e2e.py: 14, 14b,
14i, 12o, 13e, 13f, 12p, 12s, 12q) draws the cards.
"""

from __future__ import annotations

import json
import re
from typing import Any

import pytest

pytest.importorskip("mcp")

from olmoearth_agent.mcp_app import imagery, server, views  # noqa: E402

#: One token of a JS concatenation: a single-quoted string, a plus, or a name
#: (a call with no parentheses inside its own is one name).
_TOKEN = re.compile(
    r"\s*(?:'((?:[^'\\]|\\.)*)'|(\+)|([A-Za-z_][A-Za-z0-9_.]*(?:\([^()]*\))?))"
)

#: Words no face holds (round r5-e's rule R7, and the owner's ruling 6).
_OFF_FACE = re.compile(
    r"\b(?:windows?|UTM|zones?|intervals?|strat(?:um|a)|design|budget"
    r"|recommended)\b",
    re.I,
)

#: B1: the line of a part whose sheet could not be read when the card opened,
#: before any press: on a map in parts (the part by its name) and in one part.
SHEET_TRY_WEST = (
    "The sheet of West part could not be read when this card opened. Press "
    '"Open the sheet" to try again; if it does not open, tell Claude.'
)
SHEET_TRY_ONE = (
    "The sheet could not be read when this card opened. Press "
    '"Open the sheet" to try again; if it does not open, tell Claude.'
)
#: B1: the red line of a press whose sheet did not open, for a button that
#: reads "Open the sheet here" (a sheet with no answer yet).
SHEET_NOT_OPENED_HERE = (
    'The sheet could not be opened. Press "Open the sheet here" to try again; '
    "if it fails again, tell Claude."
)
#: B1: the red line once a press failed too (round r5-f F04's words).
SHEET_UNREAD_WEST = (
    "Its sheet cannot be opened now. Tell Claude that the sheet of West part "
    "cannot be opened."
)

#: B2: the heading with every square answered and no sheet open, on a map in
#: parts and in one part (unchanged). Round r5-j B3: on a map in parts the
#: press is in each part's sheet, one result per part
#: (tests/mcp_app/test_mcp_r5j_check.py).
ALL_ANSWERED_PARTS = (
    "Every square is answered. In each part's sheet, press Estimate the error "
    "rate: one result per part."
)
ALL_ANSWERED_ONE = (
    "Every square is answered. Open the sheet, then press Estimate the error rate."
)

#: B3: a part that cannot be read as a map, on its face.
NOT_READ_FACE = (
    "This part cannot be checked: the app could not cut its map into squares."
)

#: B10: what goes before the server's text, in its own block of Claude's
#: update (unchanged after review: the server's preparing note names the
#: block by these opening words).
READY_TOLD = (
    "Its map is read. What olmoearth_check_map answers for it, to take as that "
    "tool's answer: "
)
#: B10, after review: what the card sends in its place when the server's text
#: is longer than OE.CONTEXT_WHOLE_MAX, for the result r1.
READY_TOLD_LONG_R1 = (
    "Its map is read. What olmoearth_check_map answers for it is too long for "
    "this card to hand on: call olmoearth_check_map once for Studio result_id "
    "r1 before you state any result of it."
)

#: B10, after review: a map of two parts not drawn, as the server reads it
#: (server._check_ready's input), with a result id of 36 characters, long
#: scores paths, two must_state items and a forbidden claim.
RID = "0b6f2a3c-1d4e-4f5a-8b9c-0d1e2f3a4b5c"
PID = "99999999-8888-4777-8666-555555555555"
FILES = ("result_epsg32610_0_native.tif", "result_epsg32611_1_native.tif")
MUST = [
    "Each UTM zone is its own map: one zone's estimate is not the result's.",
    "These are Studio's native-resolution files.",
]
CLAIMS = [{"claim": "the result is certified", "why": "nothing certifies it"}]


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


def _concat(expr: str, names: dict[str, str] | None = None) -> str:
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
            out.append((names or {})[name])
        at = match.end()
    return "".join(out)


def _code(page: str) -> str:
    """The page with its comments taken out."""
    return re.sub(r"/\*.*?\*/|//[^\n]*", "", page, flags=re.S)


def _in_order(text: str, *pieces: str) -> bool:
    """Whether each piece is in text, each after the one before it."""
    at = [text.index(p) for p in pieces]
    return at == sorted(at)


def _sheet_try(page: str, name: str | None, button: str) -> str:
    """SHEET_TRY's text as check.html makes it for a part's name (None: a map
    in one part) and the open button's label."""
    expr = _after(page, "const SHEET_TRY = (name, button) => ", ";\n")
    named = "(name ? ' of ' + name : '')"
    assert expr.count(named) == 1
    of = " of " + name if name else ""
    return _concat(expr.replace(named, "OF"), {"OF": of, "button": button})


def _fit(text: str, room: int, cut: str) -> str:
    """OE.fitContext (common.js) in Python: text whole when it fits in room,
    else cut back to its last clause boundary ("; " or ". ") with cut after
    it, else '' (left out)."""
    if len(text) <= room:
        return text
    limit = room - len(cut)
    if limit <= 0:
        return ""
    head = text[: limit + 2]
    at = max(head.rfind("; "), head.rfind(". "))
    return text[:at] + cut if at > 0 else ""


def _cleaned(text: str) -> str:
    """What OE.tellClaude does to a part's text before it fits it (or sends
    it whole): each run of control characters and Unicode line or paragraph
    breaks one space, the ends trimmed."""
    breaks = chr(0x2028) + chr(0x2029)
    return re.sub(r"[\x00-\x1f\x7f-\x9f" + breaks + "]+", " ", text).strip()


def _update(facts: list[str], whole: list[str], common: str) -> list[str]:
    """The text blocks of the update OE.tellClaude (common.js) sends for the
    parts fitted (in order, the label editor's first) and the parts sent
    whole: CONTEXT_HEAD and each fitted part in the room the ones before it
    left within CONTEXT_MAX (none: no such block), then each whole part
    cleaned, never cut, left out over CONTEXT_WHOLE_MAX."""
    head = _after(common, "OE.CONTEXT_HEAD = '", "';\n")
    room = int(_after(common, "OE.CONTEXT_MAX = ", ";\n")) - len(head)
    most = int(_after(common, "OE.CONTEXT_WHOLE_MAX = ", ";\n"))
    cut = _after(common, "OE.CONTEXT_CUT = '", "';\n")
    said: list[str] = []
    for text in facts:
        space = 1 if said else 0
        one = _fit(_cleaned(text), room - space, cut)
        if one:
            said.append(one)
            room -= space + len(one)
    blocks = [head + " ".join(said)] if said else []
    return blocks + [t for t in (_cleaned(w) for w in whole) if len(t) <= most]


def _ready_text() -> str:
    """What a ready status's model_text holds for a map of two parts not
    drawn (:data:`RID`, the owner's rule set): olmoearth_check_map's answer
    in time with panels as the model reads it, server._check_ready's own
    text (server._answer_text), which uses nothing of its server."""
    scores = "/srv/olmoearth/outputs/studio_scores/" + RID + "/"
    ready: dict[str, Any] = {
        "inner": {
            "prediction_id": PID,
            "kind": "top_k",
            "property_name": "kelp",
            "forbidden_claims": CLAIMS,
            "zones": [{"file": f, "scores_path": scores + f + ".npz"} for f in FILES],
        },
        "zones": [
            {
                "zone": zone,
                "part": part,
                "file": file,
                "n_valid": n_valid,
                "share": share,
                "state": "to draw",
            }
            for zone, part, file, n_valid, share in (
                ("UTM zone 10N", "West part", FILES[0], 300, "60%"),
                ("UTM zone 11N", "East part", FILES[1], 200, "40%"),
            )
        ],
        "clipped": {},
        "rule": {"set": True, "source": "owner", "replaceable": True},
        "stated": MUST,
        "result_id": RID,
        "labelling_rule": {"source": "owner", "text": imagery.READING_RULE},
        "copied": None,
        "view": {},
    }
    app = server.AppServer.__new__(server.AppServer)
    return server._answer_text(app._check_ready(ready, views=True))


# --------------------------------------------------------------------------- B1


def test_b1_a_sheet_that_could_not_be_read_says_to_press_first() -> None:
    """B1: before any press, a part whose sheet could not be read when the
    card opened says that, and to press its button, by the label drawn on the
    button (in double quotes), and to tell Claude only if it does not open;
    in body type, the part by its name on a map in parts. The red line that
    sends the user to Claude (SHEET_UNREAD, round r5-f F04's words) shows only
    once a press failed too (m.sheetError)."""
    page = _read("check.html")
    assert "// WORDING: r5i B1\nconst SHEET_TRY = (name, button) => " in page
    for name, said in (("West part", SHEET_TRY_WEST), (None, SHEET_TRY_ONE)):
        assert _sheet_try(page, name, "Open the sheet") == said
        assert _OFF_FACE.search(said) is None, said
    retried = _sheet_try(page, None, "Continue labelling")
    assert 'Press "Continue labelling" to try again' in retried
    zone = _between(page, "  function fillZone(m) {", "\n  }\n")
    drawn = (
        "      const button = d.open_label || 'Open the sheet';\n"
        "      const unread = Boolean(d.sheet_error) && !d.progress;\n"
        "      if (unread) {\n"
        "        parts.push(m.sheetError ? OE.el('div', { class: 'error' }, "
        "SHEET_UNREAD(nParts > 1 ? label(m) : null))\n"
        "          : OE.el('div', {}, SHEET_TRY(nParts > 1 ? label(m) : null, "
        "button)));\n"
        "      }\n"
    )
    assert drawn in zone
    # The button below says the same label the line names.
    assert _in_order(
        zone,
        drawn,
        "        m.openButton = OE.el('button', { class: primary === m ? "
        "'primary' : '', onclick: () => openZone(m) },\n          button);\n",
    )
    assert "d.open_label || 'Open the sheet'" not in zone.replace(drawn, "")
    code = _code(page)
    assert len(re.findall(r"\bSHEET_TRY\b", code)) == 2
    # SHEET_UNREAD's words are as round r5-f wrote them.
    unread = _after(page, "const SHEET_UNREAD = (name) => ", ";\n")
    named = "(name ? ' of ' + name : '')"
    west = _concat(unread.replace(named, "OF"), {"OF": " of West part"})
    assert west == SHEET_UNREAD_WEST


def test_b1_a_sheet_that_did_not_open_names_its_button() -> None:
    """B1: the red line of a press whose sheet did not open names the button
    by its label in double quotes, the label the part draws on it; a read of
    the counts that changes that label draws the part again (m.named), so the
    line never names words the button no longer says; m.sheetError is a flag
    set where the press failed and cleared once the sheet opens; "Press the
    button" is gone."""
    page = _read("check.html")
    tags = "// WORDING: r5i B1\n// WORDING: r5f F04\nconst SHEET_NOT_OPENED = "
    assert tags in page
    expr = _after(page, "const SHEET_NOT_OPENED = (button) => ", ";\n")
    said = _concat(expr, {"button": "Open the sheet here"})
    assert said == SHEET_NOT_OPENED_HERE
    assert _OFF_FACE.search(said) is None
    assert "Press the button" not in _code(page)
    zone = _between(page, "  function fillZone(m) {", "\n  }\n")
    line = (
        "      if (m.sheetError && !unread) parts.push(OE.el('div', "
        "{ class: 'error' }, SHEET_NOT_OPENED(button)));\n"
    )
    assert line in zone
    swap = _between(
        page, "  async function openZone(m, { hadFocus = null } = {}) {", "\n  }\n"
    )
    assert _in_order(
        swap,
        "    } catch (err) {\n",
        "      m.sheetError = true;\n",
        "      renderAll();\n",
        "    m.sheetError = null;\n",
    )
    # The label a line names (m.named, set where the part is drawn) is kept;
    # a read of the counts that changes the button's label draws the part
    # again, the focus kept on its button, so the line says the new label.
    named = (
        "      m.named = (unread && !m.sheetError) || (!unread && m.sheetError) "
        "? button : null;\n"
    )
    assert _in_order(zone, line, named, "        m.openButton = OE.el('button', {")
    assert "    m.named = null;\n" in zone
    model = _between(page, "function zoneModel(z, index) {", "\n}\n")
    assert "    named: null,\n" in model
    refresh = _between(page, "  async function refreshZone(m) {", "\n  }\n")
    follows = (
        "    if (m.named && m.named !== (m.draw.open_label || 'Open the sheet')) {\n"
        "      const had = Boolean(m.openButton) && "
        "document.activeElement === m.openButton;\n"
        "      fillZone(m);\n"
        "      if (had) OE.focusIfHere(m.openButton);\n"
        "    }\n"
    )
    assert _in_order(
        refresh,
        "    if (m.openButton) m.openButton.textContent = ",
        "    if (unread && typeof got.progress === 'string' && got.progress) {\n",
        follows,
        "    fillAsk();\n",
    )
    assert "sheetError" not in refresh
    code = _code(page)
    assert len(re.findall(r"\bSHEET_NOT_OPENED\b", code)) == 2
    assert len(re.findall(r"m\.sheetError = ", code)) == 2
    # The Details lines keep their words (round r5-f F04).
    assert "const SHEET_NOT_OPENED_WHY = 'The sheet could not be opened: ';" in page
    assert "const SHEET_UNREAD_WHY = 'Its sheet cannot be read now: ';" in page


def test_b1_a_part_drawn_again_mid_open_keeps_its_button_off() -> None:
    """B1, after review: fillZone draws a part's open button off
    (aria-disabled) while a press is answered or that part's sheet opens, as
    the press and openZone set it, so a read of the counts that draws the
    part again (refreshZone: m.named, or r5-f F04's unread redraw) never shows
    a live button that does nothing; both flags are cleared before every
    renderAll. Where the open editor's save fails, openZone sets the button on
    the card now (a redraw may have replaced the one pressed), still off while
    a press that started meanwhile is answered."""
    page = _read("check.html")
    zone = _between(page, "  function fillZone(m) {", "\n  }\n")
    assert _in_order(
        zone,
        "        m.openButton = OE.el('button', { class: primary === m ? "
        "'primary' : '', onclick: () => openZone(m) },\n          button);\n",
        "        if (busy || m.opening) m.openButton.setAttribute('aria-disabled', "
        "'true');\n",
        "        parts.push(OE.el('div', { class: 'row' }, m.openButton));\n",
    )
    swap = _between(
        page, "  async function openZone(m, { hadFocus = null } = {}) {", "\n  }\n"
    )
    settle = _after(swap, "!(await current.oeSettle())) {\n", "      return;\n")
    assert _in_order(
        settle,
        "      m.opening = false;\n",
        "      if (m.openButton) m.openButton.setAttribute('aria-disabled', "
        "String(busy));\n",
        # Round r5-m B3, after review: said in the sheet's footer (oeSay),
        # a toast only for an editor with no such hook.
        "      else OE.toast(NOT_SWAPPED, true);\n",
    )
    assert "button.setAttribute('aria-disabled', 'false')" not in swap


# --------------------------------------------------------------------------- B2


def test_b2_every_part_answered_says_each_parts_sheet_in_turn() -> None:
    """B2: on a map in parts with every square answered and no sheet open the
    heading says to open each part's sheet in turn, then press Estimate the
    error rate (one press gives one part's result, never the map's); the
    line on a map in one part is unchanged. After review: "each part's sheet
    in turn" only where more than one part has a sheet (drawn), so a map in
    parts whose other parts cannot be checked or failed for good says "the
    sheet" (harness 13f). Round r5-j B3: on a map in parts the line says to
    press Estimate the error rate in each part's sheet, one result per part
    (tests/mcp_app/test_mcp_r5j_check.py); the rule on which line shows is
    unchanged."""
    page = _read("check.html")
    tags = (
        "// WORDING: r5i B2\n// WORDING: r5f F11\n// WORDING: r5b J17\n"
        "const ALL_ANSWERED = (several) => "
    )
    assert tags in page
    which = (
        "(several ? 'In each part\\'s sheet, press Estimate the error rate: one "
        "result per part.'\n    : 'Open the sheet, then press Estimate the error "
        "rate.')"
    )
    answered = _after(page, "const ALL_ANSWERED = (several) => ", ";\n")
    assert answered.count(which) == 1
    head = "Every square is answered. "
    for said in (ALL_ANSWERED_PARTS, ALL_ANSWERED_ONE):
        tail = said.removeprefix(head)
        assert "'" + tail.replace("'", "\\'") + "'" in which, tail
        assert _concat(answered.replace(which, "WHICH"), {"WHICH": tail}) == said
        assert _OFF_FACE.search(said) is None, said
    assert "a part\\'s sheet" not in page
    fill = _between(page, "  function fillAsk() {", "\n  }\n")
    assert "ALL_ANSWERED(drawn.length > 1)" in fill
    assert "nParts" not in fill
    assert "    const drawn = models.filter((m) => m.kind === 'drawn');\n" in fill


# --------------------------------------------------------------------------- B3


def test_b3_a_part_not_read_as_a_map_says_plain_words() -> None:
    """B3: a part that cannot be read as a map says on its face that the app
    could not cut its map into squares, never the server's reason (patch,
    px, a map's size, or a refusal that names a zone); that reason, after
    NOT_READ, is among the part's lines in Details, right after the plan
    refusal's line. No number changes."""
    page = _read("check.html")
    tags = "// WORDING: r5i B3\n// WORDING: squares\nconst NOT_READ_FACE = "
    assert tags in page
    assert f"const NOT_READ_FACE = '{NOT_READ_FACE}';\n" in page
    assert _OFF_FACE.search(NOT_READ_FACE) is None
    assert not re.search(r"\d|\bpx\b|patch", NOT_READ_FACE)
    # NOT_READ stays, for Details only.
    assert "const NOT_READ = 'This part cannot be checked: ';\n" in page
    zone = _between(page, "  function fillZone(m) {", "\n  }\n")
    face = (
        "    } else if (m.refusedAt === 'read') {\n"
        "      // r5-i B3: plain words on the face; the server's reason is in "
        "Details (fillFold).\n"
        "      parts.push(OE.el('div', {}, NOT_READ_FACE));\n"
    )
    assert face in zone
    assert re.search(r"\bm\.refused\b", _code(zone)) is None
    assert "NOT_READ +" not in zone
    fold = _between(page, "  function fillFold() {", "\n  }\n")
    assert _in_order(
        fold,
        "      if (m.refusedAt === 'plan' && m.refused) lines.push(OE.el('div', "
        "{ class: 'small muted' }, String(m.refused)));\n",
        "      if (m.refusedAt === 'read' && m.refused) lines.push(OE.el('div', "
        "{ class: 'small muted' }, NOT_READ + bare(m.refused) + '.'));\n",
        "      if (m.kind === 'drawn' && m.draw) {\n",
    )
    code = _code(page)
    assert len(re.findall(r"\bNOT_READ\b", code)) == 2
    assert len(re.findall(r"\bNOT_READ_FACE\b", code)) == 2


# --------------------------------------------------------------------------- B10


def test_b10_a_ready_answer_passes_its_text_to_claude_once() -> None:
    """B10: once the status answers ready, after the card is drawn (its parts
    line, tellCard) and a screen reader is told, the card passes the
    answer's model_text to Claude, after READY_TOLD, as the update's part
    'check' (OE.tellClaude): one call, in the ready branch only, never for a
    failed answer. After review: sent whole (a block of its own, never cut),
    and only for a card of parts, so a ready card with nothing to check,
    which tells Claude that it ended without a check (PREP_TOLD_FAILED), is
    never also told "Its map is read"; over OE.CONTEXT_WHOLE_MAX the card
    sends READY_TOLD_LONG in its place, never a cut text. model_text is read
    only as a string with text (else nothing is sent), and is used nowhere
    else: nothing of it is drawn."""
    page = _read("check.html")
    card = _between(page, "function preparingCard(app, data) {", "\n}\n")
    ended = _between(
        card,
        "    if ((state === 'ready' && view) || state === 'failed') {\n",
        "\n    }\n",
    )
    sent = (
        "      const told = state === 'ready' && parts ? modelTextOf(got) : null;\n"
        "      if (told) {\n"
        "        const said = READY_TOLD + told;\n"
        "        OE.tellClaude(app, said.length <= OE.CONTEXT_WHOLE_MAX ? said "
        ": READY_TOLD_LONG(resultId), 'check',\n"
        "          { whole: true });\n"
        "      }\n"
        "      return;"
    )
    assert _in_order(
        ended,
        "      show(app, { structuredContent: drawn });\n",
        "      const parts = drawn.ok === true && Array.isArray(drawn.zones)\n",
        "      if (heard) heard.textContent = ",
        "      if (!parts) OE.tellClaude(app, PREP_TOLD_FAILED(",
        sent,
    )
    assert card.count("'check',") == 1 and "'check')" not in card
    reader = (
        "const modelTextOf = (got) => (got && typeof got.model_text === 'string' "
        "&& got.model_text.trim()\n  ? got.model_text : null);\n"
    )
    assert reader in page
    code = _code(page)
    assert code.count("model_text") == reader.count("model_text")
    assert len(re.findall(r"\bmodelTextOf\b", code)) == 2
    assert len(re.findall(r"\bREADY_TOLD\b", code)) == 2
    assert len(re.findall(r"\bREADY_TOLD_LONG\b", code)) == 2
    assert len(re.findall(r"\btold\b", _code(card))) == 3
    for drawn in ("OE.el(", "textContent", "OE.toast("):
        assert drawn not in _code(_after(card, "const told = ", "return;")), drawn


def test_b10_ready_told_names_the_tool_whose_answer_follows() -> None:
    """B10: READY_TOLD, marked to Claude, says the map is read and that what
    follows is olmoearth_check_map's answer for it, to take as that tool's
    answer (the preparing note, server.CHECK_PREPARING_NOTE, says not to call
    the tool again for that result: a second call draws a second Check card
    in Claude Desktop); it holds no id, name or place (the parts line before
    it names the result, and the server's text its result_id). After
    review, READY_TOLD_LONG, sent in its place when the text is longer than
    OE.CONTEXT_WHOLE_MAX, starts with the same words, so the note names both
    alike, and asks for one call for that result by its id before any of its
    results is stated."""
    page = _read("check.html")
    assert "// WORDING: r5i B10 (to Claude)\nconst READY_TOLD = " in page
    said = _concat(_after(page, "const READY_TOLD = ", ";\n"))
    assert said == READY_TOLD
    assert "olmoearth_check_map" in said and "that tool's answer" in said
    long_tag = "// WORDING: r5i B10 (to Claude)\nconst READY_TOLD_LONG = (id) => "
    assert long_tag in page
    expr = _after(page, "const READY_TOLD_LONG = (id) => ", ";\n")
    long_said = _concat(expr, {"OE.ctxId(id)": "r1"})
    assert long_said == READY_TOLD_LONG_R1
    start = "Its map is read. What olmoearth_check_map answers for it"
    assert said.startswith(start) and long_said.startswith(start)


def test_b10_show_drops_the_text_before_any_card() -> None:
    """B10: show drops the part 'check' before it draws any card (after the
    old card's editors are disposed), so a card drawn in its place for its
    own tool result never carries an older result's text; OE.tellClaude
    removes a part for '' (from the parts sent whole too) and sends nothing
    when no part is left (a first card's show sends no update), and it puts
    only the label editor's part before the others."""
    page = _read("check.html")
    show = _between(page, "function show(app, params) {", "\n}\n")
    assert _in_order(
        show,
        "  OE.disposeEditors(root);\n",
        "  OE.tellClaude(app, '', 'check');\n",
        "  root.replaceChildren(",
        "  if (data.needs_sign_in) {\n",
    )
    assert show.count("OE.tellClaude(") == 1
    assert _code(page).count("'check')") == 1
    assert _code(page).count("'check',") == 1
    common = _read("common.js")
    body = _between(common, "OE.tellClaude = ", "\n};")
    assert _in_order(
        body,
        "    OE.contextWhole.delete(part);\n",
        "    else OE.contextParts.delete(part);\n",
        "    if (!OE.contextParts.size && !OE.contextWhole.size) return;\n",
        "sort((a, b) => Number(b === 'sheet') - Number(a === 'sheet'))",
        "    if (whole === OE.contextSent) return;\n",
    )


def test_b10_a_part_sent_whole_is_a_block_of_its_own_never_cut() -> None:
    """B10, after review: OE.tellClaude with opts.whole keeps a string part
    in OE.contextWhole, out of the parts fitted into CONTEXT_MAX (fitContext
    runs on those alone), and sends it, cleaned, as a text block of its own
    after the facts' block (CONTEXT_HEAD and the fitted parts), with no head:
    in every later update too, since each update carries every part; a whole
    part longer than OE.CONTEXT_WHOLE_MAX is left out, never cut. An update
    of whole parts alone has no facts' block; the same-as-last check reads
    every block."""
    common = _read("common.js")
    assert "OE.contextWhole = new Map();\nOE.CONTEXT_WHOLE_MAX = 12000;\n" in common
    body = _between(common, "OE.tellClaude = ", "\n};")
    assert body.startswith("OE.tellClaude = (app, text, part = 'panel', opts = {}) =>")
    stored = (
        "    else if (cleaned(text) && opts && opts.whole === true) {\n"
        "      OE.contextParts.delete(part);\n"
        "      OE.contextWhole.set(part, cleaned(text));\n"
        "    } else if (cleaned(text)) OE.contextParts.set(part, cleaned(text));\n"
    )
    sent = (
        "    const blocks = [...OE.contextWhole.values()]"
        ".filter((value) => value.length <= OE.CONTEXT_WHOLE_MAX);\n"
        "    if (!said.length && !blocks.length) return;\n"
        "    const content = [...(said.length ? [OE.CONTEXT_HEAD + said.join(' ')] "
        ": []), ...blocks]\n"
        "      .map((value) => ({ type: 'text', text: value }));\n"
        "    const whole = JSON.stringify(content);\n"
    )
    assert _in_order(
        body,
        stored,
        "    const order = [...OE.contextParts.keys()]",
        "OE.fitContext(",
        sent,
        "Promise.resolve(app.updateModelContext({ content })).catch(() => {});",
    )
    assert body.count("OE.fitContext(") == 1 and body.count("OE.contextWhole") == 4


def test_b10_the_contract_reaches_claude_whole_beside_the_longest_facts() -> None:
    """B10, after review: olmoearth_check_map's answer in time for a map of
    two parts (server._check_ready's own text, as a ready status's model_text
    holds it) is longer than CONTEXT_MAX, so fitted into the facts it lost
    must_state. Sent whole after READY_TOLD, in a block of its own, it keeps
    every say-first line, SAY_FIRST_NOTE, the card's note, every must_state
    item and every forbidden claim, beside the parts line alone and beside an
    open sheet's part with its paths and a parts line with a
    labels_csv_path, while the facts' block stays within CONTEXT_MAX; it is
    within CONTEXT_WHOLE_MAX, so the card sends it, not READY_TOLD_LONG."""
    page = _read("check.html")
    common = _read("common.js")
    head = _after(common, "OE.CONTEXT_HEAD = '", "';\n")
    most = int(_after(common, "OE.CONTEXT_MAX = ", ";\n"))
    whole_most = int(_after(common, "OE.CONTEXT_WHOLE_MAX = ", ";\n"))
    told = _concat(_after(page, "const READY_TOLD = ", ";\n"))
    model_text = _ready_text()
    say, model = [json.loads(line) for line in model_text.split("\n")]
    assert say["say_first"] == server._check_say_first(2, views=True)
    assert model["must_state"] == MUST and model["forbidden_claims"] == CLAIMS
    said = told + model_text
    assert most < len(said) <= whole_most
    contract = [*say["say_first"], server.SAY_FIRST_NOTE, server.CHECK_CARD_NOTE]
    sheet_csv = "/srv/olmoearth/outputs/checks/" + RID + "/west_part_to_label.csv"
    sheet = (
        f"Label sheet west_part_to_label.csv (labels_csv_path {sheet_csv}, "
        f"design_path {sheet_csv[:-4]}.json): 0 of 20 windows labelled, all saved."
    )
    card = f"Check card of Studio result_id {RID} (2 zones, each its own map): "
    alone = card + "UTM zone 10N: not drawn; UTM zone 11N: not drawn."
    drawn = card + (
        "UTM zone 10N: 0 of 20 labelled, sample drawn (20 squares, design random, "
        f"seed 0), label sheet west_part_to_label.csv (labels_csv_path {sheet_csv}); "
        "UTM zone 11N: not drawn."
    )
    for facts in ([alone], [sheet, drawn]):
        blocks = _update(facts, [said], common)
        assert len(blocks) == 2, facts
        assert blocks[0].startswith(head) and len(blocks[0]) <= most
        assert blocks[1] == _cleaned(said) and blocks[1].startswith(told)
        for text in [*contract, *MUST]:
            assert server._json(text)[1:-1] in blocks[1], text[:60]
        for claim in CLAIMS:
            assert server._json(claim) in blocks[1], claim


# --------------------------------------------------------------------------- marks


def test_every_item_of_the_track_is_marked_in_the_page() -> None:
    """Each new or changed string of this round carries ``// WORDING: r5i``
    with its item, above its older marks: SHEET_TRY and SHEET_NOT_OPENED
    (B1), ALL_ANSWERED (B2), NOT_READ_FACE (B3), READY_TOLD and, after
    review, READY_TOLD_LONG (B10)."""
    page = _read("check.html")
    marks = re.findall(r"// WORDING: r5i (B\d+)", page)
    for item, count in (("B1", 2), ("B2", 1), ("B3", 1), ("B10", 2)):
        assert marks.count(item) == count, item
    for name in (
        "SHEET_TRY",
        "SHEET_NOT_OPENED",
        "ALL_ANSWERED",
        "NOT_READ_FACE",
        "READY_TOLD",
        "READY_TOLD_LONG",
    ):
        marked = r"// WORDING: r5i [^\n]*\n(?://[^\n]*\n)*const " + name + r" = "
        assert re.search(marked, page) is not None, name
