# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Round r5-b (17 Nov 2026), track CHECK, on the MCP App: the Check card's
items of the user-side audit (specs/r5-audit/build_list.md) and X1 of
specs/r5-b-round.md.

- J03: a count's words follow its place among the part's counts (the one
  number, the least work, the most, one between), never the planned width;
- J08: one line under the press says why it is off, or which parts it
  leaves out; a part left for a later card says first that it is not picked;
- J09: after a press with no reply the card picks nothing more: no part
  keeps its numbers, and the gate's line and the part's line say where the
  answer will be;
- J10: answers an editor could not save are said on their part's own line,
  by the part's name, with the next step;
- J17: the heading stops asking once nothing can be chosen; the parts
  figure says how far a picked part is answered and which part cannot be
  picked here; the line under an open sheet names the other part; the
  repeat line says the number chosen now is not used;
- J18: two sentences over the press, the parts line right under the
  heading, a short label on the box beside the kelp rule, and the gate's
  example under its box, never a placeholder;
- J25: the parts' dead ends say what to tell Claude, and "count" on a face
  is "number";
- J29 (c): under 480 px each count is a row of its own;
- X1: a map of numbers that is not a score from 0 to 1 gets two plain lines
  on the card's face, and Claude keeps the tool's sentence.

The cut line and the come-back line of round r5-a stay. The views are read
as text (no test here runs the scripts); a JS concatenation of single-quoted
strings and names is read back into its text (:func:`_concat`). The server's
words are read through its own functions.
"""

from __future__ import annotations

import copy
import inspect
import json
import re
from pathlib import Path
from typing import Any

import pytest
from pytest_httpx import HTTPXMock

pytest.importorskip("mcp")
from mcp import types  # noqa: E402

from olmoearth_agent.mcp_app import auth, check, server, views  # noqa: E402
from olmoearth_agent.tools import studio_scores  # noqa: E402

BASE = "http://mock-studio/api/v1"
#: A Studio key, as a test value.
KEY = "sk_test_r5b_key_321"
#: A result's download token, as a test value.
DOWNLOAD = "dl-test-value-r5b"
R = "11111111-2222-4333-8444-555555555555"
P = "99999999-8888-4777-8666-555555555555"
#: The number's property name, as Studio's record declares it.
NAME = "sample_canopy_height"

#: One token of a JS concatenation: a single-quoted string, a plus, or a name
#: (a call with no parentheses inside its own is one name).
_TOKEN = re.compile(
    r"\s*(?:'((?:[^'\\]|\\.)*)'|(\+)|([A-Za-z_][A-Za-z0-9_.]*(?:\([^()]*\))?))"
)

#: Words no face holds (the owner's ruling 6 and r4-i's rule).
_OFF_FACE = re.compile(r"\b(?:windows?|budget|stratum|design|review set)\b", re.I)

#: The registry's hint on a failed tool, written for the model.
MODEL_HINT = (
    "The tool ran but failed. Check that referenced ids exist (discover them "
    "with search/list tools) and that argument values are valid; do not retry "
    "with identical arguments."
)

#: J08: the lines under the press.
RULE_AND_NUMBER = "Write your rule and choose a number of squares first."
CHOOSE = "Choose a number of squares above."
BOX_SHORT = (
    "Write at least 10 characters in the box, or empty it to use the kelp rule."
)
LATER = (
    "Not picked yet. To pick this part's squares, ask Claude to open the Check "
    "card again."
)

#: J09: the gate's line and a part's line after a press with no reply; the
#: press's toast, word for word as before.
RULE_LOST = (
    "This card can pick no more squares. Ask Claude to open the Check card "
    "again: it shows the rule and what was picked."
)
NO_ANSWER = (
    "No answer came back, so this part's squares may or may not be picked. The "
    "Check card opened again will say."
)
PRESS_NO_ANSWER = (
    "The press got no answer back, so this card cannot say what was picked: a "
    "part's squares may be picked, or still being picked. Ask Claude to open the "
    "Check card again; it shows each part."
)

#: J17: the heading once no part offers a number.
ALL_ANSWERED = (
    "Every square is answered. Open a sheet, then press Estimate the error rate."
)
YOUR_SQUARES = "Your squares"

#: J18: the line over the press, the box's label and the line under the box.
CANNOT = (
    "Your rule and number are fixed once you press. You can stop and come back "
    "later; there is no result until every square is done."
)
RULE_OWN = (
    "Not a kelp map? Write the map maker's meaning of Present and Absent (optional)."
)
RULE_EMPTY = "Left empty, the kelp rule above is used."
RULE_HINT = "For example: the class that fills most of the outlined square"

#: J25: the parts' dead ends.
FINAL = (
    "This part cannot be picked again here. Tell Claude that the pick failed for "
    "this part."
)
CHANGED = (
    "Studio's files for this part changed after its squares were picked, so they "
    "are not picked again here. Tell Claude that this part's files changed."
)
NO_COUNT = (
    "No number is planned for this part. Ask Claude to plan it with a number of "
    "your own."
)
NOT_OFFERED = "This card is out of date. Ask Claude to open the Check card again."

#: X1: the card's two plain lines for a number declared from 0.2 to 1.2.
NUMBER_MAP = (
    'This map gives a number at each pixel ("sample_canopy_height", from 0.2 to '
    "1.2), not classes and not a score from 0 to 1. This check works on a map of "
    "classes."
)
NUMBER_MAP_NEXT = (
    "To check it, tell Claude the value from which a pixel counts as a yes. The "
    "check then asks you about single points, without pictures."
)
#: X1: the first line for a field whose record declares no range: no claim
#: on its range.
NUMBER_MAP_BARE = (
    'This map gives a number at each pixel ("sample_canopy_height", no range '
    "declared), not classes. This check works on a map of classes."
)
#: What the card's face must not hold then (the spec's X1).
NOT_ON_THE_FACE = ("olmoearth_", "probability to pool", "threshold", "this tool")


@pytest.fixture(autouse=True)
def _roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The write root (scores, outputs and the check records): the test's own."""
    out = tmp_path / "outputs"
    out.mkdir()
    monkeypatch.setenv("OLMOEARTH_OUTPUT_ROOT", str(out))
    monkeypatch.setenv("OLMOEARTH_SCORES_ROOT", str(out))
    monkeypatch.delenv(auth.KEY_ENV, raising=False)
    monkeypatch.setenv(auth.KEY_FILE_ENV, str(tmp_path / "cfg" / "studio.env"))
    return out


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


def _app() -> server.AppServer:
    """The app's server on the mocked Studio's address."""
    return server.AppServer(None, keys=auth.KeyStore(), base_url=BASE)


def _view(result: types.CallToolResult) -> dict[str, Any]:
    """The view data of a tool's result."""
    view: dict[str, Any] = (result.meta or {})[server.VIEW_DATA_KEY]
    return view


def _model(result: types.CallToolResult) -> dict[str, Any]:
    """The model's copy of a Check card's result: its last text block."""
    texts = [c.text for c in result.content if isinstance(c, types.TextContent)]
    body: dict[str, Any] = json.loads(texts[-1])
    return body


def _fake_tool(
    app: server.AppServer, monkeypatch: pytest.MonkeyPatch, studio: dict[str, Any]
) -> None:
    """``app.run_agent_tool`` answering ``studio`` for the Studio files."""

    async def run(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """The tool's envelope, as run_agent_tool gives it."""
        assert name == server.SCORES_TOOL, name
        return copy.deepcopy(studio)

    monkeypatch.setattr(app, "run_agent_tool", run)


def _number_record(
    low: float | None = 0.2, high: float | None = 1.2, name: str = NAME
) -> dict[str, Any]:
    """A result's record, as Studio's GET by id gives one, that declares one
    regression field from ``low`` to ``high`` (``None``: no range declared)
    and no class map."""
    declared: dict[str, Any] = {"property_name": name, "band_index": 1}
    if low is not None and high is not None:
        declared.update(min_value=low, max_value=high)
    return {
        "id": R,
        "prediction_id": P,
        "prediction_name": "Canopy height test",
        "property_names": [name],
        "download_token": DOWNLOAD,
        "result_metadata": {
            "start_datetime": "2023-01-01T00:00:00Z",
            "end_datetime": "2023-12-31T00:00:00Z",
            "classification_fields": [],
            "regression_fields": [declared],
        },
    }


def _tool_refusal(record: dict[str, Any]) -> str:
    """The Studio files tool's refusal of ``record``, as the registry writes a
    failed tool's error: the exception's name, then its words."""
    with pytest.raises(ValueError) as refused:
        studio_scores.files_plan(record)
    return f"ValueError: {refused.value}"


def _failed(error: str) -> dict[str, Any]:
    """A failed envelope of the Studio files tool, as the registry writes it."""
    return {
        "ok": False,
        "error": error,
        "tool": server.SCORES_TOOL,
        "hint": MODEL_HINT,
    }


# --------------------------------------------------------------------------- J03


def test_j03_a_counts_words_follow_its_place_among_the_parts_counts() -> None:
    """J03: the words are chosen by the count's place among the part's
    offers (one offer; the first; the last; any between), marked, and the
    words by planned width are gone, so a part's one number never reads
    "Longer" and 1,100 squares never "Quick"."""
    page = _read("check.html")
    words = (
        "// WORDING: r5b J03\n"
        "const COUNT_WORDS = (i, n) => "
        "(n === 1 ? 'The one number planned for this part.'\n"
        "  : i === 0 ? 'Least work. The widest range.'\n"
        "    : i === n - 1 ? 'Most work. The narrowest range.' "
        ": 'More work. A narrower range.');\n"
    )
    assert words in page
    counts = _between(page, "  function countsBlock(m) {", "\n  }\n")
    assert "const buttons = m.offer.map((o, i) => {" in counts
    drawn = "{ class: 'pick-words small' }, COUNT_WORDS(i, m.offer.length)));"
    assert drawn in counts
    assert "o.width" not in counts and "fig(" not in counts
    for gone in (
        "WIDTH_WORDS",
        "widthWords",
        "Quick. A wide range.",
        "Longer. A narrower range.",
        "A lot of work. A narrow range.",
    ):
        assert gone not in page, gone
    # Defined once, used once.
    assert _code(page).count("COUNT_WORDS") == 2


def test_j03_the_server_sends_a_parts_counts_in_ascending_order() -> None:
    """J03 reads a count's place as its size among the part's counts: the
    server's offer is in ascending order (check.offers says so, and builds
    the list sorted by count)."""
    source = inspect.getsource(check)
    assert "offer = [by_count[count] for count in sorted(by_count)]" in source
    assert "in ascending order" in (check.offers.__doc__ or "")


# --------------------------------------------------------------------------- J08


def test_j08_the_line_under_the_press_says_why_it_is_off() -> None:
    """J08: within the most parts one press picks, the line under the press
    (oe-check-limit, which describes the press) says, in body type, that the
    rule and a number are missing, that the rule is too short (beside the
    kelp rule: that the optional box holds too few characters, never "Write
    the labelling rule first"), that no number is chosen, or which parts the
    press leaves out; a press with nothing chosen fills that line and no
    longer toasts."""
    page = _read("check.html")
    pairs = (
        ("RULE_AND_NUMBER", RULE_AND_NUMBER),
        ("CHOOSE", CHOOSE),
        ("BOX_SHORT", BOX_SHORT),
    )
    for name, said in pairs:
        assert f"// WORDING: r5b J08\nconst {name} = '{said}';\n" in page, name
    left_out = (
        "// WORDING: r5b J08\n"
        "const LEFT_OUT = (names) => "
        "(names.length > 3 ? names.length + ' parts have'\n"
        "  : names.length === 1 ? names[0] + ' has'\n"
        "    : names.slice(0, -1).join(', ') + ' and ' "
        "+ names[names.length - 1] + ' have')\n"
        "  + ' no number chosen: this press leaves ' "
        "+ (names.length === 1 ? 'it' : 'them') + ' out.';\n"
    )
    assert left_out in page
    why = _between(page, "  function pressWhy(plan) {", "\n  }\n")
    assert _in_order(
        why,
        "if (ruleLost) return '';",
        "const chosen = plan.send.length > 0;",
        "if (gateMode === 'ask' && ruleLength() < RULE_MIN && !chosen) "
        "return RULE_AND_NUMBER;",
        "if (gateMode === 'replace' && gateHeld()) return BOX_SHORT;",
        "if (gateHeld()) return RULE_FIRST;",
        "if (!chosen) return CHOOSE;",
        "const left = nParts > 1 ? models.filter((m) => m.kind === 'pick' "
        "&& m.pick === null).map(label) : [];",
        "return left.length ? LEFT_OUT(left) : '';",
    )
    update = _between(page, "  function updatePress() {", "\n  }\n")
    assert _in_order(
        update,
        "const words = limitWords(plan);",
        "const why = words ? '' : pressWhy(plan);",
        "limitLine.hidden = !words && !why;",
        "limitLine.className = words "
        "? 'small ' + (plan.over > 0 ? 'error' : 'muted') : '';",
        "limitLine.textContent = words || why;",
    )
    # The press is described by the line, and the box's keys run the update.
    described = "'aria-describedby': 'oe-check-limit oe-check-cannot',"
    assert described in page
    assert "'aria-describedby': 'oe-rule-note', oninput: () => updatePress()," in page
    press = _between(page, "  async function drawPicked() {", "\n  }\n")
    assert "    if (!picked.length) { updatePress(); return; }" in press
    assert "Pick how many squares to check first." not in page
    # A press held by the box toasts the line's own words.
    held = "OE.toast(gateMode === 'replace' ? BOX_SHORT : RULE_FIRST);"
    assert held in press
    for said in (RULE_AND_NUMBER, CHOOSE, BOX_SHORT, "no number chosen: this press"):
        assert _OFF_FACE.search(said) is None, said


def test_j08_a_part_left_for_a_later_card_says_it_is_not_picked() -> None:
    """J08: a part left without a number says first that it is not picked,
    then how to pick it, in body colour (no faint class)."""
    page = _read("check.html")
    tags = "// WORDING: r5b J08\n// WORDING: r4h C3\nconst LATER = "
    assert tags in page
    assert _concat(_after(page, "const LATER = ", ";\n")) == LATER
    zone = _between(page, "  function fillZone(m) {", "\n  }\n")
    later = (
        "    } else if (m.kind === 'later') {\n"
        "      parts.push(OE.el('div', {}, LATER));\n"
    )
    assert later in zone
    assert "class: 'faint'" not in zone


# --------------------------------------------------------------------------- J09


def test_j09_after_a_press_with_no_reply_no_part_keeps_its_numbers() -> None:
    """J09: the press that got no reply marks its parts as maybe picked, and
    every other part that still offered numbers, or a retry, is left for the
    card opened again, so no numbers and no grey press stay; the gate's line
    and the part's line say where the answer will be; the press's toast is
    word for word as before."""
    page = _read("check.html")
    tags = "// WORDING: r5b J09\n// WORDING: r4i F6\nconst RULE_LOST = "
    assert tags in page
    assert _concat(_after(page, "const RULE_LOST = ", ";\n")) == RULE_LOST
    tags = "// WORDING: r5b J09\n// WORDING: r4j J2\nconst NO_ANSWER = "
    assert tags in page
    assert _concat(_after(page, "const NO_ANSWER = ", ";\n")) == NO_ANSWER
    toast = _concat(_after(page, "const PRESS_NO_ANSWER = ", ";\n"))
    assert toast == PRESS_NO_ANSWER
    press = _between(page, "  async function drawPicked() {", "\n  }\n")
    assert _in_order(
        press,
        "picked.forEach((m) => { m.kind = 'unknown'; m.pick = null; "
        "m.line = null; });",
        "ruleLost = true;",
        "models.forEach((o) => {\n"
        "          if (o.kind === 'pick' || o.kind === 'retry') "
        "{ o.kind = 'later'; o.pick = null; o.line = null; }\n"
        "        });",
        "OE.toast(PRESS_NO_ANSWER, true);",
        "renderAll();",
    )
    # With no part left to pick, the press goes (updatePress) and the focus
    # moves on from it (focusFallback).
    update = _between(page, "  function updatePress() {", "\n  }\n")
    gone = (
        "    if (opened !== null || !can.length) {\n"
        "      if (pressArea.isConnected) pressArea.remove();\n"
    )
    assert gone in update
    assert "if (hadFocus && !press.isConnected) focusFallback();" in press
    for said in (RULE_LOST, NO_ANSWER):
        assert _OFF_FACE.search(said) is None, said
    # The part's line still asks nothing of Claude: the gate's line does.
    assert "Ask Claude" not in NO_ANSWER and "Ask Claude" in RULE_LOST


# --------------------------------------------------------------------------- J10


def test_j10_answers_not_saved_are_said_on_their_parts_own_line() -> None:
    """J10: when an editor that stopped saving gives way, its part gets a red
    line of its own (fillZone draws a part's line) and the toast the same
    words: how many answers, for which part by its name, and what to do; no
    file name. The line goes when that part's sheet opens again."""
    page = _read("check.html")
    words = (
        "// WORDING: r5b J10\n"
        "const NOT_SAVED_PART = (k, name) => OE.counted(k, 'answer') + ' for ' "
        "+ name + (k === 1 ? ' was' : ' were')\n"
        "  + ' not saved: its sheet was changed somewhere else. "
        "Open its sheet again and answer '\n"
        "  + (k === 1 ? 'that square' : 'those squares') + ' again.';\n"
    )
    assert words in page
    swap = _between(
        page, "  async function openZone(m, { hadFocus = null } = {}) {", "\n  }\n"
    )
    assert _in_order(
        swap,
        "const dropped = current && typeof current.oeUnsaved === 'function'"
        " ? current.oeUnsaved() : 0;",
        "    m.sheetError = null;\n    m.line = null;",
        "action.replaceChildren(editor);\n    renderAll();\n",
        "    if (dropped > 0 && before && before.draw) {\n",
        "      before.line = NOT_SAVED_PART(dropped, label(before));\n"
        "      fillZone(before);\n"
        "      OE.toast(before.line, true);\n"
        "    } else if (current && typeof current.oeLeave === 'function') {\n",
    )
    assert "sheetName(before.draw)" not in page
    assert "that sheet changed since the panel opened it" not in page
    zone = _between(page, "  function fillZone(m) {", "\n  }\n")
    line = "if (m.line) parts.push(OE.el('div', { class: 'error' }, m.line));"
    assert line in zone
    # Defined once, used once.
    assert _code(page).count("NOT_SAVED_PART") == 2


# --------------------------------------------------------------------------- J17


def test_j17_the_heading_follows_what_the_card_holds() -> None:
    """J17: one heading node, its words set with every redraw and after each
    read of a sheet's counts: the question while a part offers a number or a
    retry; that every square is answered, and how to get the result, when no
    sheet is open here, a part is picked, every picked part is answered
    whole and no part may still be picked elsewhere; else "Your squares"."""
    page = _read("check.html")
    for name, said in (("ALL_ANSWERED", ALL_ANSWERED), ("YOUR_SQUARES", YOUR_SQUARES)):
        assert f"// WORDING: r5b J17\nconst {name} = '{said}';\n" in page, name
    assert "  const ask = OE.el('p', { class: 'ask' }, ASK);\n" in page
    assert "  root.append(gate);\n  root.append(ask);\n" in page
    fill = _between(page, "  function fillAsk() {", "\n  }\n")
    assert _in_order(
        fill,
        "const drawn = models.filter((m) => m.kind === 'drawn');",
        "const pending = models.some((m) => m.kind === 'unknown' "
        "|| m.kind === 'busy' || m.kind === 'later');",
        "const done = opened === null && drawn.length > 0 && !pending "
        "&& drawn.every((m) => complete(m.draw));",
        "ask.textContent = drawable().length ? ASK "
        ": done ? ALL_ANSWERED : YOUR_SQUARES;",
    )
    for where in ("  function renderAll() {", "  async function refreshZone(m) {"):
        assert "    fillAsk();\n" in _between(page, where, "\n  }\n"), where
    for said in (ALL_ANSWERED, YOUR_SQUARES):
        assert _OFF_FACE.search(said) is None, said


def test_j17_the_parts_captions_the_other_part_and_the_repeat() -> None:
    """J17: a picked part's caption says how far it is answered (k of n by
    its sheet's counts, "all answered" once whole, "picked, not checked yet"
    only while the counts are not read); a part the card cannot pick says
    so; the line under an open sheet names the one other part; a press on a
    part picked before says that pick stays and the number chosen now is not
    used."""
    page = _read("check.html")
    names = {
        "name": "West part",
        "k.toLocaleString('en-US')": "5",
        "n.toLocaleString('en-US')": "20",
    }
    for start, said in (
        ("const PART_ANSWERED = (name, k, n) => ", "West part: 5 of 20 answered"),
        ("const PART_DONE = (name) => ", "West part: all answered"),
        ("const PART_CANNOT = (name) => ", "West part: cannot be picked here"),
        ("const PART_PICKED = (name) => ", "West part: picked, not checked yet"),
        ("const PART_OPEN = (name) => ", "West part: not picked yet"),
    ):
        assert _concat(_after(page, start, ";\n"), names) == said, start
    for name in ("PART_ANSWERED", "PART_DONE", "PART_CANNOT"):
        assert f"// WORDING: r5b J17\nconst {name} = " in page, name
    parts = _between(page, "  function fillParts() {", "\n  }\n")
    assert _in_order(
        parts,
        "const said = complete(d) ? PART_DONE(name)\n"
        "          : countsRead(d) ? PART_ANSWERED(name, d.n_labelled, d.n_rows) "
        ": PART_PICKED(name);",
        "return { name, share: m.shareValue, state: 'picked', label: said };",
        "const said = m.kind === 'unknown' ? name\n"
        "        : m.kind === 'refused' || m.kind === 'failed' ? PART_CANNOT(name) "
        ": PART_OPEN(name);",
        "return { name, share: m.shareValue, state: 'open', label: said };",
        "}), { mode: 'plan' });",
    )
    read = (
        "const countsRead = (d) => Boolean(d) && Number.isInteger(d.n_rows) "
        "&& Number.isInteger(d.n_labelled) && d.n_rows > 0;\n"
    )
    assert read in page
    whole = "const complete = (d) => countsRead(d) && d.n_labelled >= d.n_rows;\n"
    assert whole in page
    other = (
        "// WORDING: r5b J17\n// WORDING: squares\n"
        "const OTHER_OPEN = (names) => "
        "(names.length === 1 ? names[0] + ' is not checked yet.'\n"
        "  : names.length + ' other parts are not checked yet.');\n"
    )
    assert other in page
    fill = _between(page, "  function fillOther() {", "\n  }\n")
    named = "otherLine.textContent = open.length ? OTHER_OPEN(open.map(label)) : '';"
    assert named in fill
    assert "The other part is not checked yet." not in page
    repeat = _concat(
        _after(page, "const REPEATED = (n, day) => ", ";\n"),
        {"day": "11 Nov 2026", "OE.counted(n, 'square')": "70 squares"},
    )
    assert repeat == (
        "Already picked on 11 Nov 2026: 70 squares. That pick stays; the number "
        "chosen now is not used."
    )
    assert "// WORDING: r5b J17\n// WORDING: squares\nconst REPEATED = " in page


# --------------------------------------------------------------------------- J18


def test_j18_two_sentences_over_the_press_and_the_parts_line_first() -> None:
    """J18: the line over the press drops its third sentence (Details keeps
    the rare-class note), and "This map is saved in N parts" stands right
    under the heading, before the numbers it explains; the parts figure
    stays where it was, after the parts."""
    page = _read("check.html")
    tags = "// WORDING: r5b J18\n// WORDING: r4h C2\nconst CANNOT = "
    assert tags in page
    assert _concat(_after(page, "const CANNOT = ", ";\n")) == CANNOT
    assert "a rare class may get no square" not in page
    fold = _between(page, "  function fillFold() {", "\n  }\n")
    assert "nodes.push(OE.el('p', {}, FIXED), OE.el('p', {}, RARE));" in fold
    line = (
        "  root.append(ask);\n"
        "  if (nParts > 1) root.append(OE.el('div', { id: 'oe-check-parts-line' }, "
        "PARTS_LINE(nParts)));\n"
        "  for (const m of models) {\n"
    )
    assert line in page
    figure = (
        "  if (nParts > 1) root.append(partsSlot);\n"
        "  root.append(action, otherLine, OE.fold(DETAILS, foldBody));\n"
    )
    assert figure in page
    assert _code(page).count("PARTS_LINE(") == 1


def test_j18_the_rule_boxes_say_less_and_keep_their_example_in_view() -> None:
    """J18: the box beside the kelp rule has a short label and, under the
    box, a small line on what an empty box does; the gate that asks for a
    rule has its label in body type and its example in a small line under
    the box, which describes the box; no placeholder is set anywhere."""
    page = _read("check.html")
    tags = (
        "// WORDING: r5b J18\n// WORDING: r4m M5\n// WORDING: r4h C5\n"
        "const RULE_OWN = "
    )
    assert tags in page
    assert _concat(_after(page, "const RULE_OWN = ", ";\n")) == RULE_OWN
    assert f"// WORDING: r5b J18\nconst RULE_EMPTY = '{RULE_EMPTY}';\n" in page
    assert f"// WORDING: r4h C4\nconst RULE_HINT = '{RULE_HINT}';\n" in page
    gate = _between(page, "  function fillGate() {", "\n  }\n")
    boxed = (
        "    const boxed = (words, under) => OE.el('div', {},\n"
        "      OE.el('label', { class: 'own-rule', for: 'oe-rule' }, words), "
        "ruleInput,\n"
        "      OE.el('div', { class: 'small muted under-box', id: 'oe-rule-note' }, "
        "under));\n"
    )
    assert boxed in gate
    asked = (
        "gate.replaceChildren(OE.el('div', {}, RULE_ASK), "
        "boxed(RULE_LABEL, RULE_HINT));"
    )
    assert asked in gate
    assert "gate.replaceChildren(...ruleLines(), boxed(RULE_OWN, RULE_EMPTY));" in gate
    assert "placeholder" not in _code(page)
    assert "field-label" not in page
    assert ".gate label.own-rule { display: block; margin-bottom: 4px; }" in page
    assert ".gate .under-box { margin-top: 4px; }" in page
    # The rule sent is the box's value, as before.
    sent = _after(page, "  const ruleText = () => ", ";\n")
    assert "ruleInput.value" in sent
    assert len(RULE_OWN.split()) < 15


# --------------------------------------------------------------------------- J25


def test_j25_the_parts_dead_ends_say_what_to_tell_claude() -> None:
    """J25: no line of a part stops at "ask Claude about it"; a face says
    "number", never "count"; a part that cannot be checked says so."""
    page = _read("check.html")
    for name, said in (("FINAL", FINAL), ("CHANGED", CHANGED), ("NO_COUNT", NO_COUNT)):
        assert f"// WORDING: r5b J25\n// WORDING: squares\nconst {name} = " in page
        assert _concat(_after(page, f"const {name} = ", ";\n")) == said, name
    not_read = (
        "// WORDING: r5b J25\n// WORDING: squares\n"
        "const NOT_READ = 'This part cannot be checked: ';\n"
    )
    assert not_read in page
    retry = _concat(
        _after(page, "const RETRY = (n) => ", ";\n"),
        {"PRESS": "Pick the squares", "OE.counted(n, 'square')": "80 squares"},
    )
    assert retry == (
        "Press Pick the squares to try again with the same number: 80 squares."
    )
    assert "// WORDING: r5b J25\n// WORDING: squares\nconst RETRY = " in page
    for gone in (
        "ask Claude about it",
        "No count is planned",
        "not read as a map",
        "with the same count",
    ):
        assert gone not in page, gone


def test_j25_a_count_the_card_no_longer_offers_says_it_is_out_of_date() -> None:
    """J25: the server's refusal of a count the newest offer does not hold
    says what to do, never "That count was not offered": the card shows it
    under a number it offered itself. The card joins it to "Not picked: "."""
    assert check.NOT_OFFERED == NOT_OFFERED
    assert "count" not in check.NOT_OFFERED and "offered" not in check.NOT_OFFERED
    source = inspect.getsource(check)
    assert f'# WORDING: r5b J25\nNOT_OFFERED = "{NOT_OFFERED}"\n' in source
    assert source.count("raise ValueError(NOT_OFFERED)") == 2
    page = _read("check.html")
    answer = _between(page, "  function takeAnswer(m, a) {", "\n  }\n")
    refused = (
        "if (a.outcome === 'refused') { m.line = NOT_PICKED + "
        "bare(a.error || 'the press was refused') + '.'; return; }"
    )
    assert refused in answer


# --------------------------------------------------------------------------- J29 (c)


def test_j29c_under_480_px_each_count_is_a_row_of_its_own() -> None:
    """J29 (c): on a narrow panel a count takes the whole line, its number
    at the left and its words at the right, so three counts no longer break
    two and one with the third stretched."""
    page = _read("check.html")
    rule = (
        "  @media (max-width: 480px) {\n"
        "    .pick-group button.pick { flex: 1 1 100%; min-width: 0; "
        "flex-direction: row; align-items: baseline;\n"
        "      justify-content: space-between; }\n"
        "  }\n"
    )
    assert rule in page
    assert "calc(50% - 8px)" not in page
    wide = (
        "  .pick-group button.pick { display: flex; flex-direction: column; "
        "align-items: flex-start; gap: 4px;\n"
    )
    assert wide in page


# --------------------------------------------------------------------------- X1


def test_x1_the_tools_refusal_of_a_number_map_is_read() -> None:
    """X1: the Studio files tool's own refusal of a result that declares one
    regression field from 0.2 to 1.2 starts and ends as the server expects,
    and its field is read: name, lowest and highest as the tool writes
    them, in plain digits. With no declared range (or an end that is no
    finite number), a line that claims no range; a range the tool's text
    rounds to [0, 1] keeps the tool's sentence (the card's line would
    contradict itself)."""
    error = _tool_refusal(_number_record())
    plain = error.removeprefix("ValueError: ")
    assert plain.startswith(server._NUMBER_MAP_START)
    assert plain.endswith(server._NUMBER_MAP_END)
    assert "olmoearth_plan_label_sample" in plain and "threshold" in plain
    field = server._number_map(error)
    assert field is not None
    assert field == {"name": NAME, "low": "0.2", "high": "1.2"}
    assert server._number_map_words(field) == NUMBER_MAP
    # The plain words too (no exception's name), and with spaces around.
    assert server._number_map(f"  {plain} ") == field
    wide = server._number_map(_tool_refusal(_number_record(-50.0, 1500.0)))
    assert wide == {"name": NAME, "low": "-50", "high": "1500"}
    bare = server._number_map(_tool_refusal(_number_record(None, None)))
    assert bare is not None
    assert bare == {"name": NAME, "low": None, "high": None}
    assert server._number_map_words(bare) == NUMBER_MAP_BARE
    assert "0 to 1" not in NUMBER_MAP_BARE
    endless = server._number_map(_tool_refusal(_number_record(0.0, float("inf"))))
    assert endless == {"name": NAME, "low": None, "high": None}
    # The tool writes 0.9999999 as "1": its sentence stays on the card.
    assert server._number_map(_tool_refusal(_number_record(0.0, 0.9999999))) is None
    # The tool writes 1000000 as "1e+06": the card says it in plain digits.
    big = server._number_map(_tool_refusal(_number_record(0.0, 1000000.0)))
    assert big == {"name": NAME, "low": "0", "high": "1000000"}
    assert "from 0 to 1000000), not classes" in server._number_map_words(big)
    tiny = server._number_map(_tool_refusal(_number_record(-0.0000001, 5.0)))
    assert tiny == {"name": NAME, "low": "-0.0000001", "high": "5"}
    # A name with a single quote in it: Python writes it in double quotes.
    quoted = server._number_map(_tool_refusal(_number_record(name="tree's height")))
    assert quoted == {"name": "tree's height", "low": "0.2", "high": "1.2"}


def test_x1_the_words_are_two_plain_lines_marked() -> None:
    """X1: the card's two lines as the spec gives them, each marked; no tool
    name, no "probability to pool", no "threshold" and no off-face word."""
    shown = server.CHECK_NUMBER_FIELD.format(name=NAME, low="0.2", high="1.2")
    assert server.CHECK_NUMBER_MAP.format(shown=shown) == NUMBER_MAP
    assert server.CHECK_NUMBER_MAP_NEXT == NUMBER_MAP_NEXT
    assert server.CHECK_NUMBER_MAP_BARE.format(name=NAME) == NUMBER_MAP_BARE
    for said in (NUMBER_MAP, NUMBER_MAP_BARE, NUMBER_MAP_NEXT):
        assert _OFF_FACE.search(said) is None, said
        for word in NOT_ON_THE_FACE:
            assert word not in said, word
    source = inspect.getsource(server)
    for name in (
        "CHECK_NUMBER_MAP = (",
        "CHECK_NUMBER_FIELD = ",
        "CHECK_NUMBER_MAP_BARE = (",
        "CHECK_NUMBER_MAP_NEXT = (",
    ):
        assert f"# WORDING: r5b X1\n{name}" in source, name
    assert "CHECK_NUMBER_FIELD_BARE" not in source


def test_x1_no_other_refusal_is_read_as_a_number_map() -> None:
    """X1: only that one refusal with exactly one field is taken. Several
    regression fields, no field, a class map beside a regression field, a
    class map without scores, a name Python writes with a backslash, a
    property name that holds the refusal's own words, and any other error
    keep the tool's sentence."""
    two = _number_record()
    two["result_metadata"]["regression_fields"].append(
        {"property_name": "second", "min_value": 0.0, "max_value": 5.0}
    )
    none = _number_record()
    none["result_metadata"]["regression_fields"] = []
    beside = _number_record()
    beside["result_metadata"]["classification_fields"] = [
        {"property_name": "landcover", "band_index": 1}
    ]
    classes = _number_record()
    classes["result_metadata"]["regression_fields"] = []
    classes["result_metadata"]["classification_fields"] = [
        {"property_name": "landcover", "band_index": 1}
    ]
    slashed = _number_record(name="up\\down")
    both = _number_record(name='it\'s "so"')
    for record in (two, none, beside, classes, slashed, both):
        error = _tool_refusal(record)
        assert server._number_map(error) is None, error
    # One field whose name looks like the end of one field and the start of
    # another is read as the one field it is, its whole name kept.
    sly = server._number_map(_tool_refusal(_number_record(name="a' [0, 2], 'b")))
    assert sly == {"name": "a' [0, 2], 'b", "low": "0.2", "high": "1.2"}
    start, end = server._NUMBER_MAP_START, server._NUMBER_MAP_END
    for error in (
        "",
        "ValueError: the check failed",
        "RuntimeError: GET /prediction-results/files failed (ReadTimeout)",
        start + end,
        f"{start}'a' [0, 2]",
        f"'a' [0, 2]{end}",
        f"{start}'a' [0, 2], 'b' [0, 3]{end}",
        f"{start}no output field{end}",
        f"{start}'' [0, 2]{end}",
        f"{start}'a' [0, 2]{end} and more",
    ):
        assert server._number_map(error) is None, error
    assert server._number_map(f"{start}'a' [0, 2]{end}") == {
        "name": "a",
        "low": "0",
        "high": "2",
    }


@pytest.mark.asyncio
async def test_x1_a_number_maps_check_card_says_two_plain_lines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """X1: the Check card of such a result gets the two plain lines as its
    view's ``error`` and ``next``, no hint, no sign-in, no zones and no
    offer; nothing of the tool's sentence reaches the view; Claude reads the
    tool's sentence and hint as the tool gave them."""
    offered: list[Any] = []
    monkeypatch.setattr(check, "offer_zones", lambda *a, **k: offered.append(a))
    error = _tool_refusal(_number_record())
    app = _app()
    _fake_tool(app, monkeypatch, _failed(error))
    for panels in (True, False):
        result = await app.check_map_tool({"result_id": R}, views=panels)
        assert result.isError is True
        view = _view(result)
        assert view == {
            "tool": server.CHECK_TOOL,
            "ok": False,
            "needs_sign_in": False,
            "error": NUMBER_MAP,
            "next": NUMBER_MAP_NEXT,
            "hint": None,
        }
        for word in NOT_ON_THE_FACE:
            assert word not in view["error"] + " " + view["next"], word
        text = json.dumps(view)
        for word in ("plan_label_sample", "to pool", "Error", "search/list tools"):
            assert word not in text, word
        model = _model(result)
        assert (model["error"], model["hint"]) == (error, MODEL_HINT)
        assert "olmoearth_plan_label_sample" in model["error"]
        assert "note" not in model and "next" not in model
    assert offered == []
    await app.aclose()


@pytest.mark.asyncio
async def test_x1_any_other_failed_card_is_as_it_was(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """X1 changes one refusal only: a result with two regression fields keeps
    the tool's sentence on its card (no ``next``), and signed out, the card
    offers the sign-in with its hint, never the two lines."""
    two = _number_record()
    two["result_metadata"]["regression_fields"].append(
        {"property_name": "second", "min_value": 0.0, "max_value": 5.0}
    )
    error = _tool_refusal(two)
    app = _app()
    _fake_tool(app, monkeypatch, _failed(error))
    view = _view(await app.check_map_tool({"result_id": R}, views=True))
    assert view["error"] == error.removeprefix("ValueError: ")
    assert "next" not in view and view["hint"] is None

    signed_out = {
        **_failed(_tool_refusal(_number_record())),
        "needs_sign_in": True,
        "hint": auth.NOT_SIGNED_IN,
    }
    _fake_tool(app, monkeypatch, signed_out)
    out = _view(await app.check_map_tool({"result_id": R}, views=True))
    assert out["needs_sign_in"] is True and out["hint"] == auth.NOT_SIGNED_IN
    assert "next" not in out
    await app.aclose()


@pytest.mark.asyncio
async def test_x1_the_real_tool_on_a_mocked_studio(
    httpx_mock: HTTPXMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """X1 end to end: the real Studio files tool, run through the registry on
    a mocked Studio whose record declares one regression field from 0.2 to
    1.2, refuses before any download, and the Check card's face gets the
    two plain lines while Claude's text keeps the tool's sentence."""
    monkeypatch.setenv(auth.KEY_ENV, KEY)
    httpx_mock.add_response(
        url=f"{BASE}/prediction-results/{R}",
        json={"records": [_number_record()], "meta": None, "errors": None},
        is_reusable=True,
    )
    app = _app()
    result = await app.check_map_tool({"result_id": R}, views=True)
    assert result.isError is True
    view = _view(result)
    assert (view["error"], view["next"]) == (NUMBER_MAP, NUMBER_MAP_NEXT)
    assert view["hint"] is None and view["needs_sign_in"] is False
    for word in NOT_ON_THE_FACE:
        assert word not in view["error"] + " " + view["next"], word
    text = json.dumps(view)
    for word in ("plan_label_sample", "to pool", "Error", DOWNLOAD):
        assert word not in text, word
    model = _model(result)
    assert model["error"].startswith("ValueError: " + server._NUMBER_MAP_START)
    assert "olmoearth_plan_label_sample" in model["error"]
    assert DOWNLOAD not in json.dumps(model)
    await app.aclose()


def test_x1_the_page_draws_the_two_lines_and_nothing_else() -> None:
    """X1: the page shows the view's two lines whole, as text, in a plain
    card under the card's title: no press, no count, no fold and no copy of
    the server's words. Any other failed card is drawn as before."""
    page = _read("check.html")
    card = _between(page, "function numberMapCard(data, next) {", "\n}\n")
    assert _in_order(
        card,
        "root.append(OE.el('h2', {}, TITLE));",
        "root.append(OE.el('div', { class: 'card stack', id: 'oe-check-numbers' },",
        "OE.el('div', {}, data.error), OE.el('div', {}, next)));",
    )
    for gone in ("OE.fold(", "button", "class: 'card error'", "innerHTML"):
        assert gone not in card, gone
    show = _between(page, "function show(app, params) {", "\n}\n")
    assert _in_order(
        show,
        "if (data.needs_sign_in) {",
        "if (!data.ok && reason) {\n    classMapCard(app, data, reason);",
        "const next = typeof data.next === 'string' && data.next ? data.next : null;",
        "if (!data.ok && next && typeof data.error === 'string' && data.error) {\n"
        "    numberMapCard(data, next);",
        "if (!data.ok) {\n    root.append(OE.el('div', { class: 'card error' },",
    )
    assert "// WORDING: r5b X1 (the server's words, shown whole)\n" in page
    for copied in ("This map gives a number at each pixel", "counts as a yes"):
        assert copied not in page, copied


# --------------------------------------------------------------------------- what r5-a left


def test_the_cut_line_and_the_come_back_line_of_r5a_stay() -> None:
    """Round r5-a's two lines are on the card as it left them: the score
    map's cut right over the gate, and the come-back line right under the
    press, hidden while a part shows as picked or as pressed with no reply."""
    page = _read("check.html")
    cut = (
        "  if (cut) root.append(OE.el('div', { id: 'oe-check-cut' }, cut));\n"
        "  root.append(gate);\n"
    )
    assert cut in page
    again = (
        "  const againLine = OE.el('div', { class: 'small muted', "
        "id: 'oe-check-again' }, PICKED_BEFORE);\n"
    )
    assert again in page
    start = "  const pressArea = OE.el('div', { class: 'stack' },"
    area = _between(page, start, ";\n")
    assert _in_order(
        area,
        "OE.el('div', { id: 'oe-check-cannot' }, CANNOT),",
        "OE.el('div', { class: 'row' }, press),",
        "againLine,",
        "limitLine)",
    )
    # The slice stops right before the call's ";": the line under the press is last.
    assert area.endswith("    limitLine)")
    update = _between(page, "  function updatePress() {", "\n  }\n")
    hidden = (
        "    againLine.hidden = models.some("
        "(m) => m.kind === 'drawn' || m.kind === 'unknown');\n"
    )
    assert hidden in update


def test_every_item_of_the_track_is_marked_in_the_page() -> None:
    """Each new or changed string of this round carries its WORDING mark."""
    page = _read("check.html")
    marks = re.findall(r"// WORDING: r5b (J\d+|X1)", page)
    for item, count in (
        ("J03", 1),
        ("J08", 5),
        ("J09", 2),
        ("J10", 1),
        ("J17", 7),
        ("J18", 3),
        ("J25", 5),
        ("X1", 1),
    ):
        assert marks.count(item) == count, item
