# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Round r5-h (18 Nov 2026), track VIEW, on the MCP App: the Check card while
its map is being prepared (specs/r5-h-round.md H1, the card's side).

- H1: a card whose view data says ``preparing`` draws its title, the map's
  line when known (from the view, or once a status answer gives it), the
  four step panels and one line, "Preparing this map: <stage>.", beside a
  quiet mark that holds no number; no press, no count, no rule and no box.
  It asks ``olmoearth_app_check_status`` 2 s after each answer, and again
  after an ask that fails; once ready it draws the status's view with the
  function that draws a card that opened ready (``show``), which tells
  Claude its parts as that card does; once failed, the failed card, and
  Claude is told the words that card shows, quoted as data; a screen reader
  hears either end; from 10 minutes after it was drawn it asks no more and
  says the map took too long and to ask Claude to open the Check card again.

The views are read as text (no test here runs the scripts); a JS
concatenation of single-quoted strings and names is read back into its text
(:func:`_concat`). The browser harness (scripts/mcp_views_e2e.py: 12p, 12q
and 12r) draws the card and answers its asks.
"""

from __future__ import annotations

import json
import re

import pytest

pytest.importorskip("mcp")

from olmoearth_agent.mcp_app import server, views  # noqa: E402

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

#: The status tool the preparing card asks (server.APP_TOOLS).
STATUS_TOOL = "olmoearth_app_check_status"

#: The preparation's stages as the round names them (a few fixed plain lines
#: of the server's, set as the preparation goes).
STAGES = (
    "Downloading the map from Studio",
    "Reading part 1 of 2",
    "Planning how many squares to offer",
)

#: The card's line once it stopped waiting, and when the view names no result.
#: Round r5-j B5: the line once it stopped waiting says, as the failed card
#: does, that what was picked or saved before is kept
#: (tests/mcp_app/test_mcp_r5j_check.py).
PREP_TOO_LONG = (
    "Preparing this map took too long. Anything picked or saved before is kept. "
    "Ask Claude to open the Check card again."
)
PREP_NO_ID = (
    "This card cannot tell when the map is ready. Ask Claude to open the Check "
    "card again."
)

#: What the card tells Claude, for result r1: while it waits, once it stopped
#: waiting, and once the preparation ended without a check.
PREP_TOLD = (
    "Check card of Studio result_id r1: its map is still being prepared; the "
    "card fills in by itself."
)
PREP_TOLD_LONG = (
    "Check card of Studio result_id r1: the card stopped waiting for its map "
    "after 10 minutes; it asks the user to have Claude open the Check card "
    "again (olmoearth_check_map)."
)
#: The server's plain error for a result whose files could not be read
#: (server._check_failed, r5-a K4), as the failed card shows it in Details.
READ_FAILED = "GET /prediction-results/files failed (ReadTimeout); no file was read"
PREP_TOLD_FAILED = (
    "Check card of Studio result_id r1: its map's preparation ended without a "
    "check to offer; the card's words, as data: " + json.dumps(READ_FAILED) + "."
)
PREP_TOLD_FAILED_NONE = (
    "Check card of Studio result_id r1: its map's preparation ended without a "
    "check to offer; the card gives no reason."
)

#: What a screen reader hears once the card filled in, or ended without one.
PREP_SAID_READY = "The map is ready to check."
PREP_SAID_ENDED = "The map could not be made ready to check."


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


def _card(page: str) -> str:
    """check.html's preparingCard, whole."""
    return _between(page, "function preparingCard(app, data) {", "\n}\n")


def _preparing(stage: str | None) -> str:
    """PREPARING's text as check.html makes it: the stage after a colon, its
    closing stops dropped (bare), and one stop at the end."""
    said = "Preparing this map"
    if stage:
        said += ": " + re.sub(r"\.+$", "", stage.strip())
    return said + "."


def _told_failed(page: str, why: str | None, rid: str = "r1") -> str:
    """PREP_TOLD_FAILED's text as check.html makes it for result rid, with
    why (cardWords' quoted words) or with none: its pieces around the
    ternary read back (:func:`_concat` reads no ternary)."""
    expr = _after(page, "const PREP_TOLD_FAILED = (id, why) => ", ";\n")
    head, rest = expr.split("\n  + (why ? ", 1)
    told, rest = rest.split(" : ", 1)
    none, tail = rest.split(") + ", 1)
    names = {"OE.ctxId(id)": rid, "why": why or ""}
    middle = _concat(told, names) if why else _concat(none)
    return _concat(head, names) + middle + _concat(tail)


# --------------------------------------------------------------------------- H1


def test_h1_a_preparing_view_gets_its_own_card_before_any_part() -> None:
    """H1: show stops any earlier preparing card's asks first and empties
    what a screen reader last heard of one, then, after the sign-in and
    every failed card (which a preparing view, ok, never reaches), draws the
    preparing card for a view that says preparing, before the card of parts;
    nothing else of show changes."""
    page = _read("check.html")
    show = _between(page, "function show(app, params) {", "\n}\n")
    head = (
        "function show(app, params) {\n"
        "  stopAsking();\n"
        "  if (heard) heard.textContent = '';\n"
        "  const data = OE.viewData(params);\n"
        "  OE.disposeEditors(root);\n"
    )
    assert show.startswith(head)
    branch = (
        "  if (data.preparing === true) {\n"
        "    preparingCard(app, data);\n"
        "    return;\n"
        "  }\n"
    )
    assert _in_order(
        show,
        "  if (data.needs_sign_in) {\n",
        "  if (!data.ok) {\n    // The check could not start (r5-d R01)",
        "    return;\n  }\n  const zones = ",
        "  const must = Array.isArray(data.must_state) ? data.must_state : [];\n",
        branch,
        "  if (zones.length) {\n",
    )
    assert show.count("preparingCard(") == 1
    assert "let stopAsking = () => {};\n" in page


def test_h1_the_preparing_card_draws_the_steps_and_one_line() -> None:
    """H1: the preparing card draws the title, the map's line when the view
    has it (mapLineOf, as the card that opens ready builds it), the four
    step panels with the steps as written (no part's square is known yet:
    step 1 unsized), then one line, a status, the mark before the words; the
    mark is an empty span hidden from a screen reader. No press, count,
    rule, box, fold, toast or focus in it. Round r5-j B5: the line that the
    card fills in by itself follows the status, outside it
    (tests/mcp_app/test_mcp_r5j_check.py)."""
    page = _read("check.html")
    card = _card(page)
    line = (
        "  const row = OE.el('div', { class: 'oe-preparing', "
        "id: 'oe-check-preparing', role: 'status' }, mark, words);\n"
    )
    assert _in_order(
        card,
        "  const title = OE.el('h2', {}, TITLE);\n  root.append(title);\n",
        "  const mapLine = mapLineOf(data.map);\n  let lined = Boolean(mapLine);\n",
        "  if (lined) root.append(OE.el('div', { class: 'small muted' }, "
        "mapLine));\n",
        "  const steps = fig('steps', [...STEPS]);\n  if (steps) root.append(steps);\n",
        "  const words = OE.el('span', {}, PREPARING(stageOf(data.stage)));\n",
        "  const mark = OE.el('span', { class: 'oe-prep-mark', "
        "'aria-hidden': 'true' });\n",
        line,
        "  root.append(row, fills);\n",
    )
    assert card.count("root.append(") == 4
    for gone in (
        "countsBlock",
        "ruleInput",
        "gate",
        "OE.fold(",
        "'button'",
        "'input'",
        "OE.toast(",
        "focus",
        "checkCard(",
    ):
        assert gone not in card, gone
    # The steps as the card that opens ready writes them, step 3 r5-f F09's.
    steps = _between(page, "const STEPS = [", "];")
    assert "'1. The map is cut into small squares.',\n" in steps


def test_h1_the_mark_is_quiet_and_holds_no_number() -> None:
    """H1: the mark is a small ring that turns slowly beside the words, in
    the card's own style; it stands still for a reader who asks for less
    motion; nothing on the line, whichever of the server's stages it says
    (server.CHECK_STAGE_*), is a number pretending to be a percentage or a
    word kept off the face. Round r5-j B5: the mark sits by the line's first
    row (4 px down) with its turning edge in the text's colour
    (tests/mcp_app/test_mcp_r5j_check.py)."""
    page = _read("check.html")
    style = _between(page, "<style>", "</style>")
    for rule in (
        "  .oe-preparing { display: flex; align-items: flex-start; gap: 8px; }\n",
        "  .oe-prep-mark { flex: none; width: 12px; height: 12px; "
        "border-radius: 50%; border: 2px solid var(--divider);\n"
        "    border-top-color: var(--text); margin-top: 4px; "
        "animation: oe-prep-turn 1.6s linear infinite; }\n",
        "  @keyframes oe-prep-turn { to { transform: rotate(360deg); } }\n",
        "  @media (prefers-reduced-motion: reduce) "
        "{ .oe-prep-mark { animation: none; } }\n",
    ):
        assert rule in style, rule
    # The stages the card will show are the server's; this test's STAGES are
    # three of them.
    stages = [
        value.format(number=1, parts=2)
        for key, value in vars(server).items()
        if key.startswith("CHECK_STAGE_") and isinstance(value, str)
    ]
    assert server.CHECK_STAGE_DOWNLOAD == STAGES[0]
    assert server.CHECK_STAGE_READ_PART.format(number=1, parts=2) == STAGES[1]
    assert server.CHECK_STAGE_PLAN == STAGES[2]
    assert set(STAGES) <= set(stages)
    for said in [_preparing(s) for s in stages] + [_preparing(None)]:
        assert "%" not in said and _OFF_FACE.search(said) is None, said
    assert "%" not in PREP_TOO_LONG and "%" not in PREP_NO_ID


def test_h1_the_line_says_the_servers_stage_in_plain_words() -> None:
    """H1: the line is "Preparing this map: <stage>." with the server's stage
    as given (its closing stops dropped), "Preparing this map." without one;
    a status's new stage replaces it, and the line is written only when its
    words change (a status is heard each time it is written, and the card
    asks every 2 s); once the card stops waiting it says the map took too
    long and to ask Claude to open the Check card again, and a view with no
    result id says so at once. Each string is marked r5h. Round r5-j B5: the
    line once it stopped waiting, in two pieces, says that what was picked
    or saved before is kept; the view with no result id drops the line that
    the card fills in by itself (tests/mcp_app/test_mcp_r5j_check.py)."""
    page = _read("check.html")
    expr = (
        "const PREPARING = (stage) => 'Preparing this map' + "
        "(stage ? ': ' + bare(stage) : '') + '.';\n"
    )
    assert "// WORDING: r5h\n" + expr in page
    bare = "const bare = (text) => String(text ?? '').trim().replace(/\\.+$/, '');\n"
    assert bare in page
    for name, said in (("PREP_TOO_LONG", PREP_TOO_LONG), ("PREP_NO_ID", PREP_NO_ID)):
        assert f"// WORDING: r5h\nconst {name} = '" in page, name
        assert _concat(_after(page, f"const {name} = ", ";\n")) == said, name
        assert said.endswith(" Ask Claude to open the Check card again."), name
        assert _OFF_FACE.search(said) is None, name
    card = _card(page)
    assert "const stageOf = (s) => (typeof s === 'string' && s.trim()" in card
    new = (
        "      const next = PREPARING(stageOf(got.stage));\n"
        "      if (words.textContent !== next) words.textContent = next;\n"
    )
    assert _in_order(card, "    if (state === 'preparing') {\n", new)
    # Its three writes: no result id, a new stage, too long.
    assert card.count("words.textContent = ") == 3
    no_id = (
        "  if (!resultId) {\n    mark.remove();\n    fills.remove();\n"
        "    words.textContent = PREP_NO_ID;\n"
    )
    assert no_id in card


def test_h1_the_card_asks_the_status_2_s_after_each_answer() -> None:
    """H1: the card asks olmoearth_app_check_status with its result's id 2 s
    after it was drawn and 2 s after each answer (one ask at a time, never
    an interval), each ask waiting 15 s at most; an ask that fails (thrown:
    a refusal or no answer) is an answer it cannot read, and it is asked
    again."""
    page = _read("check.html")
    for const in (
        "const PREP_ASK_MS = 2000;\n",
        "const PREP_ASK_WAIT_MS = 15000;\n",
        "const PREP_LIMIT_MS = 600000;\n",
    ):
        assert const in page, const
    card = _card(page)
    call = (
        "      got = await OE.call(app, 'olmoearth_app_check_status', "
        "{ result_id: resultId }, { timeout: PREP_ASK_WAIT_MS });\n"
        "    } catch (_) { got = null; }\n"
    )
    assert call in card
    assert card.count("OE.call(") == 1
    assert card.count("setTimeout(ask, PREP_ASK_MS);") == 2
    assert "setInterval" not in page
    assert _in_order(
        card,
        "  const ask = async () => {\n",
        call,
        "    const state = got && typeof got.state === 'string' ? got.state : null;\n",
        "    if ((state === 'ready' && view) || state === 'failed') {\n",
        "    if (state === 'preparing') {\n",
    )
    # The next ask after this answer; then, the card's last line, the first.
    assert card.endswith(
        "    timer = setTimeout(ask, PREP_ASK_MS);\n  };\n"
        "  timer = setTimeout(ask, PREP_ASK_MS);"
    )


def test_h1_ready_is_drawn_by_the_function_of_a_card_that_opened_ready() -> None:
    """H1: ready, the card stops asking and hands the status's view to show
    as an app tool's result (OE.viewData reads its structuredContent), so the
    card of parts is drawn by the one call of checkCard, which tells Claude
    its parts (renderAll, tellCard) as on a card that opened ready. A status
    view that says preparing again is not drawn (it would start this card
    anew)."""
    page = _read("check.html")
    card = _card(page)
    view = (
        "    const view = got && got.view && typeof got.view === 'object' "
        "&& got.view.preparing !== true ? got.view : null;\n"
    )
    ended = (
        "    if ((state === 'ready' && view) || state === 'failed') {\n"
        "      stop();\n"
        "      const drawn = view || { tool: 'olmoearth_check_map', ok: false };\n"
        "      show(app, { structuredContent: drawn });\n"
    )
    assert _in_order(card, view, ended)
    assert card.count("show(app, ") == 1
    common = _read("common.js")
    reads = (
        "OE.viewData = (params) => (params && params._meta && "
        "params._meta['olmoearth/view'])\n"
        "  || (params && params.structuredContent) || {};\n"
    )
    assert reads in common
    code = _code(page)
    assert len(re.findall(r"(?<!function )\bcheckCard\(app, ", code)) == 1
    render = _between(page, "  function renderAll() {", "\n  }\n")
    assert "    tellCard();" in render


def test_h1_failed_draws_the_failed_card_and_tells_claude() -> None:
    """H1: failed, the card stops asking and show draws the status's view, the
    failed card's (r5-a K4, r5-d R01), or with none a card that could not
    start; Claude is told that the preparation ended without a check, with
    the words that card shows (cardWords), unless the card drawn is a card
    of parts, which tells its own (show's test for a part: an object among
    the zones); the same for a ready view with no part to check. Either way
    a screen reader hears it (#oe-check-said), keyed on the card drawn, not
    on the state."""
    page = _read("check.html")
    card = _card(page)
    ended = _between(
        card,
        "    if ((state === 'ready' && view) || state === 'failed') {\n",
        "\n    }\n",
    )
    for text in (
        "      const parts = drawn.ok === true && Array.isArray(drawn.zones)\n"
        "        && drawn.zones.some((z) => z && typeof z === 'object');\n",
        "      if (heard) heard.textContent = parts ? PREP_SAID_READY "
        ": PREP_SAID_ENDED;\n",
        "      if (!parts) OE.tellClaude(app, "
        "PREP_TOLD_FAILED(resultId, cardWords(drawn)));\n",
        "      return;",
    ):
        assert text in ended, text
    assert _in_order(
        ended, "stop();", "show(app,", "heard.textContent = ", "OE.tellClaude(app,"
    )
    # A view with ok false is drawn as the card that could not start.
    show = _between(page, "function show(app, params) {", "\n}\n")
    assert "OE.el('div', {}, CHECK_FAILED)" in show


def test_h1_ten_minutes_after_it_was_drawn_the_card_stops_waiting() -> None:
    """H1: before each ask and after each answer the card compares the time
    since it was drawn with 10 minutes; from then it asks no more, its mark
    goes, its line says PREP_TOO_LONG (still the status, so a screen reader
    hears it) and Claude is told that it stopped waiting. Round r5-j B5: the
    line that the card fills in by itself goes too, and the line is the
    failed card's red card (tests/mcp_app/test_mcp_r5j_check.py)."""
    page = _read("check.html")
    card = _card(page)
    gone = _between(card, "  const tooLong = () => {\n", "\n  };\n")
    assert gone == (
        "  const tooLong = () => {\n"
        "    stop();\n"
        "    mark.remove();\n"
        "    fills.remove();\n"
        "    // r5-j B5: the red card of the card that could not start (show's "
        "failed card), still the status.\n"
        "    row.className = 'card error';\n"
        "    words.textContent = PREP_TOO_LONG;\n"
        "    OE.tellClaude(app, PREP_TOLD_LONG(resultId));"
    )
    limit = "if (Date.now() - since >= PREP_LIMIT_MS) { tooLong(); return; }"
    assert card.count(limit) == 2
    # From when the card was drawn; once before the ask, once after its answer.
    since = card.index("  const since = Date.now();\n")
    assert since < card.index(limit) < card.index("OE.call(") < card.rindex(limit)


def test_h1_the_asks_stop_when_another_card_is_drawn_or_the_panel_closes() -> None:
    """H1: the card's stop ends its asks (its pending ask's timer cleared, an
    answer that comes after it dropped) and leaves OE.cleanups; it is the
    stop show runs before it draws any card (stopAsking) and one Claude's
    closing of the panel runs (OE.cleanups, r4-k K1)."""
    page = _read("check.html")
    card = _card(page)
    stop = _between(card, "  const stop = () => {\n", "\n  };\n")
    for text in (
        "stopped = true;",
        "clearTimeout(timer);",
        "OE.cleanups.delete(stop);",
    ):
        assert text in stop, text
    assert "  stopAsking = stop;\n  OE.cleanups.add(stop);\n" in card
    held = "    if (stopped) return;\n"
    assert card.count(held) == 2
    # One before the ask, one after its answer.
    assert card.index(held) < card.index("OE.call(") < card.rindex(held)


def test_h1_what_the_preparing_card_tells_claude() -> None:
    """H1: facts and ids (the result's id through OE.ctxId; never a Studio
    name, a stage or a place): while it waits, that the card fills in by
    itself; once it stopped waiting, what follows; once the preparation
    ended without a check, the words the card shows for it, quoted as data
    (cardWords), or that it gives none: the card's "it has the reason" holds,
    and nothing says that a second olmoearth_check_map says why (it prepares
    the map anew). The Check card's first update is still tellCard's in the
    page's order (test_mcp_server reads it there)."""
    page = _read("check.html")
    names = {"OE.ctxId(id)": "r1"}
    for name, said in (
        ("PREP_TOLD", PREP_TOLD),
        ("PREP_TOLD_LONG", PREP_TOLD_LONG),
    ):
        assert f"// WORDING: r5h (to Claude)\nconst {name} = (id) => " in page, name
        expr = _after(page, f"const {name} = (id) => ", ";\n")
        assert _concat(expr, names) == said, name
        for left_out in ("dataName", ".name", "stage", "map.", "OE.named"):
            assert left_out not in expr, (name, left_out)
    failed = "// WORDING: r5h (to Claude)\nconst PREP_TOLD_FAILED = (id, why) => "
    assert failed in page
    expr = _after(page, "const PREP_TOLD_FAILED = (id, why) => ", ";\n")
    for left_out in ("dataName", ".name", "stage", "map.", "OE.named", "says why"):
        assert left_out not in expr, left_out
    assert _told_failed(page, json.dumps(READ_FAILED)) == PREP_TOLD_FAILED
    assert _told_failed(page, None) == PREP_TOLD_FAILED_NONE
    card = _card(page)
    assert "  OE.tellClaude(app, PREP_TOLD(resultId));\n" in card
    # Round r5-i B10 adds a fourth: a ready answer's model_text, part 'check'
    # (tests/mcp_app/test_mcp_r5i_check.py).
    assert card.count("OE.tellClaude(app, ") == 4
    assert page.index("OE.tellClaude(app, ") > page.index("function tellCard(")
    assert page.index("OE.tellClaude(app, ") < page.index("function preparingCard(")


def test_h1_claude_gets_the_cards_words_quoted_as_data_within_the_limit() -> None:
    """H1: the words a card drawn from a status shows for why it holds no
    check are a refusal's reason, else the server's plain error (the failed
    card's Details); Claude gets them much as OE.dataName gives a name:
    quoted, one line (every run of control characters, line breaks, U+2028
    and U+2029 among them, and spaces one space: \\s covers the two), at most
    300 characters, so the update, with the longest id and the head, stays
    within CONTEXT_MAX and OE.fitContext never cuts inside the quotes."""
    page = _read("check.html")
    words = _between(page, "const cardWords = (view) => {\n", "\n};\n")
    for text in (
        "  const said = [view.reason, view.error].find("
        "(s) => typeof s === 'string' && s.trim());\n",
        "    ? JSON.stringify(said.replace("
        "/[\\s\\u0000-\\u001f\\u007f-\\u009f]+/g, ' ')"
        ".trim().slice(0, 300))\n",
        "    : null;",
    ):
        assert text in words, text
    common = _read("common.js")
    head = _after(common, "OE.CONTEXT_HEAD = '", "';\n")
    most = int(_after(common, "OE.CONTEXT_MAX = ", ";\n"))
    longest = _told_failed(page, json.dumps("y" * 300), "x" * 255)
    assert len(head) + len(longest) <= most


def test_h1_the_map_line_comes_with_a_status_once_known() -> None:
    """H1: the server knows the map's name and period only once its files are
    read, which on a large map is after the card opened, and sends them with
    each preparing answer (map): the card draws them under its title once,
    built as the card that opens ready builds its line (mapLineOf), outside
    the status line, so a screen reader still hears only the stage."""
    page = _read("check.html")
    builds = (
        "const mapLineOf = (map) => (map && typeof map === 'object'\n"
        "  ? [map.name, map.period_text].filter((s) => typeof s === 'string' && s)"
        ".join(', ') : '');\n"
    )
    assert builds in page
    ready = (
        "  const mapLine = [map.name, map.period_text].filter((s) => "
        "typeof s === 'string' && s).join(', ');\n"
    )
    assert ready in _between(page, "function checkCard(", "\n}\n")
    card = _card(page)
    later = (
        "      const later = lined ? '' : mapLineOf(got.map);\n"
        "      if (later) {\n"
        "        title.after(OE.el('div', { class: 'small muted' }, later));\n"
        "        lined = true;\n"
        "      }\n"
    )
    assert _in_order(
        card,
        "  let lined = Boolean(mapLine);\n",
        "    if (state === 'preparing') {\n",
        later,
        "    timer = setTimeout(ask, PREP_ASK_MS);\n  };\n",
    )
    assert card.count("mapLineOf(") == 2 and card.count(".after(") == 1


def test_h1_a_screen_reader_hears_the_card_fill_in_or_end() -> None:
    """H1: the page holds one status outside #root, on no face (1 px,
    clipped, as an empty job slot), silent until a preparing card's map is
    ready with a card of parts (PREP_SAID_READY) or ends with any other card
    (PREP_SAID_ENDED), since the preparing line, the card's status, goes with
    the card; show empties it before it draws any card. Its words are marked
    r5h and hold no '%' and no word kept off the face; no toast, no focus."""
    page = _read("check.html")
    said = (
        '<div class="muted">Waiting for the result...</div></main>\n'
        '<div id="oe-check-said" class="oe-said-hidden" role="status" '
        'aria-live="polite"></div>\n<script>\n'
    )
    assert said in page
    assert page.count('id="oe-check-said"') == 1
    style = _between(page, "<style>", "</style>")
    hidden = (
        "  .oe-said-hidden { position: absolute; width: 1px; height: 1px; "
        "margin: 0; overflow: hidden; clip-path: inset(50%);\n"
        "    white-space: nowrap; }\n"
    )
    assert hidden in style
    assert "const heard = document.getElementById('oe-check-said');\n" in page
    for name, words in (
        ("PREP_SAID_READY", PREP_SAID_READY),
        ("PREP_SAID_ENDED", PREP_SAID_ENDED),
    ):
        assert f"// WORDING: r5h\nconst {name} = '{words}';\n" in page, name
        assert "%" not in words and _OFF_FACE.search(words) is None, name
    code = _code(page)
    # Round r5-m B3, after review: a third, the card of parts' press with no
    # reply (PRESS_NO_ANSWER; tests/mcp_app/test_mcp_r5m_check.py).
    assert code.count("heard.textContent = ") == 3
    assert code.count("heard.textContent = PRESS_NO_ANSWER;") == 1
    assert code.count("PREP_SAID_READY") == 2 and code.count("PREP_SAID_ENDED") == 2


def test_h1_the_status_tool_is_a_views_tool_called_with_its_schema() -> None:
    """H1: olmoearth_app_check_status is one of the views' tools (APP_TOOLS,
    which a panel may call: r4-k K4), with its handler, and the card's one
    call sends the schema's arguments: the result's id."""
    assert STATUS_TOOL in server.APP_TOOLS
    handler = "_app_" + STATUS_TOOL.removeprefix("olmoearth_app_")
    assert callable(getattr(server.AppServer, handler, None)), handler
    schema = server.APP_TOOLS[STATUS_TOOL][1]
    assert "result_id" in schema["properties"]
    assert set(schema.get("required", [])) <= {"result_id"}
    pattern = r"OE\.call\(app, '" + STATUS_TOOL + r"', \{ ([a-z_]+): "
    calls = re.findall(pattern, _read("check.html"))
    assert calls == ["result_id"]
