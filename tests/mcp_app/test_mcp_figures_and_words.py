# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""The start card, the projects card and the slim line in the reader test's
words, and the figures (views/figures.js) that draw only what they are given:
read from the files, since no test here runs the scripts.

The sheet's words (sheet.js, another builder's file in r4-h) are read as
text: :func:`_holds` finds a sentence of the plan's string table however the
script joins it (over lines, around a unit word such as ``one``)."""

from __future__ import annotations

import re

from olmoearth_agent.mcp_app import sheet, views

#: The start card's sentence about the key (B/S4), then what to do without a
#: Studio account (r4-h P1: A/F15, E/start-3).
KEY_NOTE = (
    "You need your own Studio API key: a secret code that Studio makes for you "
    "(in Studio, open your profile, then API Keys). It stays on this computer and "
    "goes only to Studio, never through Claude. No Studio account? Ask whoever "
    "made the map how to get one of your own. Do not use their key."
)

#: The start card's second lead line (r4-h P1, D/D-start-1), signed out
#: (r5-c S03: signed in, START_NEXT_IN in start.html takes its place).
START_NEXT = (
    "Sign in below first; then press a job, or type what you want in the chat."
)

#: The projects card's second lead line (r5-a V3: it names the button that
#: opens a project; before, r4-h A/F30, the project's name; r5-c S04: the
#: button names in quotes, one sentence).
PROJECTS_NEXT = (
    'Next: press "Show its maps" on the project that holds your map, then '
    '"Check how wrong it is" beside the map.'
)

#: The sheet's words of the r4-h string table that this file pins (S1, S2,
#: S3; {one} is the sheet's unit word, {finish} its finish button's words).
CODE_LINE = "While you label, this panel is not sent a {one}'s number, place or order."
# r5-b J29: the finish line's second sentence has its object.
EVERY_DONE = "All answered and saved. After you press, do not change an answer."
# r5-b J13: the lead's third line, which took in the line on changing an answer.
NO_RESULT = (
    "No result until every {one} is answered. You can change an answer until you "
    "press {finish}."
)
BLIND_SEEN = (
    "Blind: this panel is not sent what the map says for a {one}, or where the "
    "{one} is, so you answer without seeing the map's answer."
)
# r5-b J11: a sheet with no pictures and no places cannot show where a point is.
WHAT_BARE = (
    "This sheet has no pictures and no places, so it cannot show where a {one} is. "
    "Answer only from your own records of each {one}, without looking at the map; "
    "with none, stop and tell Claude."
)
# r4-m M6: whoever answers must not look at the map either.
WHAT_PLACE = (
    "This sheet has no pictures. See where each {one} is, then say what was there "
    "from your own imagery or records. The map's answer is hidden: answer without "
    "looking at the map."
)

#: The slim line on top of every view (B/S6).
SUB = (
    "This panel is the OlmoEarth Agent, an app inside Claude. It works with your "
    "OlmoEarth Studio data. It is not Studio itself."
)


def _read(name: str) -> str:
    """A file of views/, as it is on disk."""
    return (views.VIEW_DIR / name).read_text(encoding="utf-8")


def _between(text: str, start: str, end: str) -> str:
    """The part of text from start up to the first end after it."""
    at = text.index(start)
    return text[at : text.index(end, at)]


def _flat(js: str) -> str:
    """JS text with each run of joined single-quoted literals read as one (the
    joins ``' + '``, over lines too, dropped) and escaped quotes unescaped."""
    return re.sub(r"'\s*\+\s*'", "", js).replace("\\'", "'")


def _pieces(text: str) -> list[str]:
    """A sentence of the string table cut where a script may join a name to
    it: at each placeholder ({one}) and after each stop, colon or semicolon;
    trimmed, the empty ones dropped."""
    parts = re.split(r"\{[a-z]+\}|(?<=[.?:;])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def _holds(js: str, text: str) -> bool:
    """Whether js says text, however its pieces are joined (:func:`_pieces`)."""
    flat = _flat(js)
    return all(piece in flat for piece in _pieces(text))


def _code(js: str) -> str:
    """JS text without its comments (a block, or the rest of a line after
    ``//``), joined as :func:`_flat` joins it: what a word gone from the
    page must be gone from."""
    bare = re.sub(r"/\*.*?\*/", "", js, flags=re.S)
    return _flat(re.sub(r"//[^\n]*", "", bare))


def test_the_sentence_reader_reads_joined_pieces() -> None:
    """The reader itself: a sentence over two lines, around a unit word and
    with an escaped quote holds; a changed word does not."""
    js = (
        "x: 'While you label, this panel is not sent a ' + one + '\\'s number,'\n"
        "    + ' place or order.',"
    )
    assert _holds(js, CODE_LINE)
    assert not _holds(js.replace("place", "spot"), CODE_LINE)
    assert _pieces("Cannot judge a {one}? Press Cannot tell.") == [
        "Cannot judge a",
        "?",
        "Press Cannot tell.",
    ]


def test_every_view_says_what_the_panel_is() -> None:
    assert f"OE.SUB = '{SUB}';" in _read("common.js")
    for name in views.VIEWS:
        assert SUB in views.view_html(name), name


def test_the_start_card_says_where_a_key_is_made() -> None:
    """B/S4, and r4-h P1: what to do without a Studio account, at the end of
    the note (after "never through Claude."), never with their key."""
    page = _read("start.html")
    assert f"// WORDING: r4h P1\nconst KEY_NOTE = '{KEY_NOTE}';" in page
    said = KEY_NOTE.index("never through Claude.")
    assert said < KEY_NOTE.index("No Studio account?")
    assert "note: KEY_NOTE," in page
    # r5-b J19: the panel's default note is two plain lines, the first saying
    # where Studio makes a key; its button says what its press does.
    common = _read("common.js")
    assert (
        "OE.SIGN_IN_OPENS = 'The button opens a sign-in page in your browser, on "
        "this computer. Enter your Studio API key there (in Studio: your profile, "
        "then API Keys).';" in common
    )
    assert "OE.SIGN_IN_BUTTON = 'Open the sign-in page';" in common
    sign_in = _between(common, "OE.signInPanel = ", "\n};")
    assert "OE.SIGN_IN_BUTTON);" in sign_in
    assert "const lines = given.length ? given : OE.SIGN_IN_NOTE;" in sign_in
    assert "Sign in in your browser" not in common


def test_the_start_card_has_its_three_lines_and_no_fold() -> None:
    """r4-h P1 (D/D-start-1): the second line names the jobs below and a
    question typed in the chat; the hidden job is not named."""
    page = _read("start.html")
    lead = _between(page, "const START_LEAD = {", "\n};")
    for line in (
        "what: 'Jobs this app can do with your Studio maps. The first checks how "
        "wrong a map is.'",
        f"next: '{START_NEXT}'",
        # r5-c S03: the third line has a subject.
        "cannot: 'This card knows nothing about your map yet; only your own answers "
        "can show how wrong it is.'",
    ):
        assert line in lead, line
    assert "// WORDING: r4h P1\nconst START_LEAD = {" in page
    assert "press Check how wrong a map is" not in page
    # r5-c S03: signed in, the next line no longer asks to sign in.
    assert "OE.lead(on ? { ...START_LEAD, next: START_NEXT_IN } : START_LEAD)" in page
    assert "OE.fold(" not in page


def test_sign_out_says_who_can_remove_an_installed_key() -> None:
    """r5-c S01: the app's own key stays in effect after Sign out: the status
    line says the user is still signed in, and who can remove that key."""
    common = _read("common.js")
    assert (
        "'Still signed in: this app was installed with its own key. Only whoever "
        "installed it can remove it.'" in common
    )
    assert "still set in the server" not in common
    assert "Signed out here. A key put in" not in common


def test_the_lead_draws_its_lines_with_their_keys() -> None:
    common = _read("common.js")
    lead = _between(common, "OE.lead = (lead) => {", "\n};")
    assert "if (!lines.length) return null;" in lead
    assert "OE.el('p', { class: k }, s)" in lead
    assert ".oe-lead > p { margin: 0; }" in common


def test_the_projects_card_words() -> None:
    page = _read("studio.html")
    for text in (
        # r5-c S04: one name for the thing (maps), Studio's word after it, and
        # the third line's subject.
        "Your Studio projects. Each holds the maps its models made over an area "
        "and a period; Studio calls each run a prediction.",
        f"next: '{PROJECTS_NEXT}',",
        "This app cannot tell which of several similar projects holds your map. "
        "Ask whoever made them.",
        "OE.lead(PROJECTS_LEAD)",
        "OE.el('h1', {}, 'Projects in this Studio account')",
        "label: (kind) => 'Tell Claude: this ' + kind,",
        "title: 'Tells Claude which one you mean. Starts no job.',",
        "label: (shown) => (shown ? 'Hide outputs' : 'Outputs'),",
        # r5-c S05: the outputs list's first line, only where there are several.
        "first: 'This run made more than one map. Use the buttons on the one you "
        "mean.',",
        # r5-c S06: the note under the heading; the boxes' sentence moved out of
        # COMPARE.how into COMPARE.tick.
        "Each prediction is a model run over an area and a period; its output is "
        "the map.",
        "Embeddings are not a map these jobs can check or rank.",
        "tick: 'Tick two finished maps of the same area and dates to say which is "
        "more accurate.',",
        "how: 'You label a sample of the places where the two maps differ. The "
        "answer says which map is right more often there, or that these labels "
        "cannot tell; it does not say how accurate either map is.',",
        "Two ticked. Press the button to start.",
        "Starts nothing yet: you first draw or pick the area and give the dates. "
        "Needs a Studio key that can write.",
    ):
        assert text in page, text
    for gone in (
        "Use in chat",
        "Hide results",
        "Select two completed",
        "Open the project that holds your map",
        "Press the name of the project",
        # r5-c S04, S05: the words before this round.
        "Inside each are its predictions",
        "What this run produced",
    ):
        assert gone not in page, gone
    # Tell Claude starts no job: its press says the ids and the quoted name only.
    use = _between(page, "const useButton = ", "\n\n")
    assert "OE.say(app, text)" in use and "sendJob" not in use
    # A result's property names are shown as text only, never sent.
    row = _between(page, "function resultRow(", "\n}\n")
    assert "names.join(', ')" in row
    assert "OE.tellClaude" not in row and "OE.say" not in row
    lead_tag = "// WORDING: r5c S04\n"
    assert lead_tag + "const PROJECTS_LEAD = {" in page


def test_an_open_projects_models_sit_in_a_closed_fold() -> None:
    """r4-h A/F33: the Models box (Run on an area) serves making a new map,
    not the check, so an open project shows it in a closed fold named "Make a
    new map"."""
    page = _read("studio.html")
    assert "// WORDING: r4h A/F33\nconst MODELS_FOLD = 'Make a new map';" in page
    summary = _between(page, "function projectSummary(", "\n}\n")
    assert "const modelsBox = modelsCard(app, project, out);" in summary
    assert "modelsBox ? OE.fold(MODELS_FOLD, modelsBox) : null" in summary
    assert summary.count("modelsCard(") == 1
    fold = _between(_read("common.js"), "OE.fold = ", ";\n")
    assert "OE.el('details', { class: 'oe-more' }" in fold and "open" not in fold


def test_the_glyph_words() -> None:
    glyph = _read("glyph.js")
    assert "classes (drawn here without their colours' + noGrain + ')'" in glyph
    assert (
        "'embeddings (numbers for other programs; these jobs cannot check or rank "
        "it)'" in glyph
    )
    assert (
        "'months of the year its predictions cover (all years and areas counted "
        "together)'" in glyph
    )
    assert "colours not read'" not in glyph and "folded together'" not in glyph
    short = _between(glyph, "OE.glyph.outputWords = (m) => {", "\n};")
    assert "colour" not in short and "these jobs" not in short


def test_figures_live_inside_oe_fig_and_never_update_by_themselves() -> None:
    script = _read("figures.js")
    assert script.count("OE.fig = (() => {") == 1
    assert script.rstrip().endswith("})();")
    # The scripts are inlined into one module: no name of its own at the top.
    top = re.compile(r"^(?:const|let|var|function|async function|class)\b", re.M)
    assert top.search(script) is None
    for name in ("steps", "rateLine", "parts", "squares"):
        assert f"  const {name} = (" in script, name
    # r4-h C1: the count buttons draw no band, so the figure is gone.
    assert "  const band = (" not in script and ".oe-band" not in script
    assert "return { STYLE, style, steps, rateLine, parts, squares };" in script
    for banned in (
        "OE.call(",
        "callServerTool",
        "setTimeout",
        "setInterval",
        "addEventListener",
        "fetch(",
    ):
        assert banned not in script, banned
    for placeholder in ("@EXT_APPS", "@LEAFLET", "@DRAW", "_DATA_URL@", "@SCRIPTS@"):
        assert placeholder not in script, placeholder
    raw = {chr(0x2028), chr(0x2029), "\x00", "\x0b", "\x0c", "\x1b", "\x7f"}
    assert not raw & set(script)


def test_figures_have_the_plans_shapes() -> None:
    script = _read("figures.js")
    rate = _between(script, "  const rateLine = (", "\n  };")
    assert "'0%'" in rate and "'100%'" in rate
    assert "role: 'img', 'aria-label': label" in rate
    assert "const FLOOR = 0.12;" in script
    parts = _between(script, "  const parts = (", "\n  };")
    assert "if (items.length < 2) return null;" in parts
    squares = _between(script, "  const squares = (", "\n  };")
    # At most three class entries, then one for Cannot tell (r4-i O2) and one
    # for the outline: five in all.
    assert ".slice(0, FILLS.length)" in squares
    assert squares.count("entries.push(") == 2
    assert "const FILLS = ['f0', 'f1', 'f2'];" in script
    # Round r5-b, J04: the second class hollow, the third a mid teal with a
    # dark ring (before: mint, and a pale grey that vanished on the card).
    for token in (
        ".oe-sq.f0, .oe-swatch.f0 { background: var(--text); }",
        ".oe-sq.f1, .oe-swatch.f1 { background: var(--paper);"
        " box-shadow: inset 0 0 0 2px var(--text); }",
        ".oe-sq.f2, .oe-swatch.f2 { background: rgba(16,82,87,.45);"
        " box-shadow: inset 0 0 0 1px var(--text); }",
        "width: 14px; height: 14px",
    ):
        assert token in script, token


def test_the_rate_line_says_no_range_words_and_draws_a_cannot_tell_bar() -> None:
    """r4-h R1: no "range:" words under the line (the headline says the range);
    its aria-label is "{rateText}; 95% range {rangeText}". R4: with no rate
    (Cannot tell answers) and the rate's own two ends in [0, 1], a dark bar
    (class rate-bar, var(--text), 6 high, at least 3 wide) over the pale band,
    labelled rateText above its middle; never without the band."""
    script = _read("figures.js")
    rate = _between(script, "  const rateLine = (", "\n  };")
    assert "range: " not in rate and "'range" not in rate
    assert "OE.el('div', { class: 'small' }" not in rate
    said = "const said = rangeText ? '95% range ' + rangeText : '';"
    assert f"    // WORDING: r4h R1\n    {said}" in rate
    assert "const label = [rateText, said].filter(Boolean).join('; ');" in rate
    for text in (
        "const rateLow = fraction(o.rateLow);",
        "const rateHigh = fraction(o.rateHigh);",
        "const barred = rate === null && ranged && rateLow !== null"
        " && rateHigh !== null && rateHigh >= rateLow;",
        "const rateText = rate !== null || barred ? words(o.rateText) : '';",
        "const across = Math.max(3, xOf(rateHigh) - x0);",
        "class: 'rate-bar', x: round2(mid - across / 2), y: RL.y - 3,"
        " width: round2(across), height: 6,",
        "if (rateText) marks.push(labelAt(mid, rateText));",
    ):
        assert text in rate, text
    # The bar is drawn after the band, so it sits over it.
    assert rate.index("class: 'band'") < rate.index("class: 'rate-bar'")
    assert ".oe-fig .rate-bar { fill: var(--text); }" in script
    # Round r5-b, J29 (d): the band at 45% (it was 30%, too pale).
    assert ".oe-fig .band { fill: var(--text-2); fill-opacity: .45; }" in script


def test_the_parts_figure_draws_the_check_cards_states() -> None:
    """r4-h A/F08: on the Check card (mode 'plan') a part not picked is dashed
    and a picked one a solid outline, neither filled (a fill means a result on
    the result card); every part is named by its label, else its name; the
    result card's states are as before."""
    script = _read("figures.js")
    assert (
        "const STATES = { result: ['this', 'graded', 'not'],"
        " plan: ['open', 'picked'] };" in script
    )
    parts = _between(script, "  const parts = (", "\n  };")
    for text in (
        "const known = STATES[options && options.mode === 'result'"
        " ? 'result' : 'plan'];",
        "return words(p.label) || name;",
        "const state = known.includes(p.state) ? p.state : 'plan';",
    ):
        assert text in parts, text
    for token in (
        ".oe-part.open .oe-part-box { border-style: dashed; background: transparent; }",
        ".oe-part.picked .oe-part-box { background: transparent; }",
        ".oe-part.not .oe-part-box { border-style: dashed; background: transparent; }",
        ".oe-part.this .oe-part-box { background: var(--text);"
        " border-color: var(--text); }",
    ):
        assert token in script, token


def test_the_red_outline_stands_off_the_dark_squares() -> None:
    """r4-h V2: the outline where the map said otherwise is 3 px, set 2 px off
    the square, and the grid's gap is 8 px so outlines stay apart."""
    script = _read("figures.js")
    assert (
        ".oe-sq.wrong, .oe-swatch.wrong { outline: 3px solid var(--error);"
        " outline-offset: 2px; }" in script
    )
    grid = ".oe-sq-grid { display: flex; flex-wrap: wrap; gap: 8px; padding: 5px; }"
    assert grid in script
    assert "outline: 2px solid var(--error)" not in script


def test_squares_fill_only_the_classes_seen() -> None:
    """Round 4 pass 2, findings 4 and 21: the fills go in turn to the classes
    that have cells, so on a map of four or more classes (only codes 2 and 3
    seen, say) two seen classes never share a fill."""
    squares = _between(_read("figures.js"), "  const squares = (", "\n  };")
    counted_at = squares.index("const count = new Map();")
    assert counted_at < squares.index("const order = [];")
    assert "if (count.get(k) && !order.includes(k)) order.push(k);" in squares
    assert "if (!order.includes(key(v))) order.push(key(v));" not in squares


def test_the_result_cards_fixes_of_pass_2() -> None:
    page = _read("result.html")
    # Finding 13: the plan card's button before a read is the server's words.
    assert sheet.OPEN_FIRST == "Open the sheet here"
    assert f"// WORDING: r4p2\nconst OPEN_FIRST = '{sheet.OPEN_FIRST}';" in page
    assert "}, OPEN_FIRST);" in page and "let words = OPEN_FIRST," in page
    assert "Label these windows" not in page
    # Finding 17: one other part, or k of them (r4-h R3: reported side by
    # side, never averaged; tests/mcp_app/test_mcp_r4e_f3_views.py reads them
    # against the server's SAY_OTHER and SAY_OTHERS).
    words = _between(page, "const RESULT_WORDS = {", "\n};")
    for text in (
        "otherPart: (k) => (k === 1\n    ? 'Check the other part too.",
        "    : 'Check the other ' + k + ' parts too.",
    ):
        assert text in words, text
    for gone in ("for its own result", "each for its own result"):
        assert gone not in words, gone
    card = _between(page, "function resultCard(", "\n}\n")
    assert "parts.filter((p) => p.state === 'not').length" in card
    assert "RESULT_WORDS.otherPart(other)" in card
    # Finding 18: a refusal's reason stays on the face, never in the fold.
    show = _between(page, "function show(app, params) {", "\n}\n")
    assert "root.append(reasonBlock(data.reason));" in show
    assert "rest.push(reasonBlock(" not in page
    assert show.index("root.append(reasonBlock(") < show.index("OE.fold(FOLD_LABEL")
    # (K): where Studio makes a key, as in the default note; since r5-b J19
    # three plain lines, the card's own, then the panel's two (common.js).
    assert (
        "note: ['This step needs OlmoEarth Studio.', OE.SIGN_IN_OPENS,"
        " OE.SIGN_IN_KEY_GOES]," in show
    )


def test_the_sheet_names_its_windows_by_its_unit_word() -> None:
    """The lead's ruling W: every window word on the sheet's face, list and
    pane is the page copy's unit ('square', or 'point' for a plan drawn by
    result_id); tool names, fields and what Claude is told keep 'window'."""
    script = _read("sheet.js")
    assert "// WORDING: squares\nOE.sheetWords = (unit) => {" in script
    words = _between(script, "OE.sheetWords = (unit) => {", "\n};")
    assert "const one = unit === 'point' ? 'point' : 'square';" in words
    for text in (
        "title: (k, n) => One + ' ' + k + ' of ' + n,",
        "oneView: 'One ' + one,",
        "howToShow: 'How to show the ' + many,",
        "pictures: (k) => 'Pictures: ' + one + ' ' + k,",
        "openRow: (k) => 'Open ' + one + ' ' + k,",
        "back: (k) => 'Back to ' + one + ' ' + k,",
        "notOfSheet: 'Not a ' + one + ' of this sheet: nothing here is labelled.',",
    ):
        assert text in words, text
    # r4-h S1 (C/sheet-3) and S2 (A/F17): the code line and the finish line
    # in the plan's words, by the unit word.
    for said in (CODE_LINE, EVERY_DONE):
        assert _holds(script, said), said
    for gone in (
        "comes to this panel under a code of its own",
        "is labelled and saved.",
    ):
        assert gone not in words, gone
    assert "const W = OE.sheetWords(data.unit);" in script
    editor = _between(script, "OE.sheetEditor = (app, data,", "\n};")
    for gone in (
        "'Window '",
        "Pictures: window",
        "'Open window '",
        "Back to window",
        "Not a window of this sheet",
        "Each window comes to this panel",
        "'One window'",
        "How to show the windows",
        "in window '",
        "for window '",
        "No windows",
        "Every window is labelled and saved.'",
        "OE.SHEET_WORDS",
    ):
        assert gone not in editor, gone
    # What Claude is told keeps "window".
    told = _between(script, "  function contextText() {", "\n  }\n")
    assert "' windows labelled'" in told


def test_a_sheet_that_can_show_no_pictures_promises_none() -> None:
    """Finding 10: with the page copy's pictures_possible false, the lead's
    first line takes its plain form, no line says pictures are off and no
    Pictures button is drawn; a page copy without the field shows them.
    r4-h S3 (E/sheet-1): a sheet of points with a place says first that it
    has no pictures and what to judge from; without a place, since r5-b J11,
    that it cannot show where a point is. r5-b J13: the lead's other two
    lines (Cannot tell; no result until every answer) speak of no picture,
    so they have one form."""
    script = _read("sheet.js")
    for said in (WHAT_BARE, WHAT_PLACE, NO_RESULT):
        assert _holds(script, said), said
    editor = _between(script, "OE.sheetEditor = (app, data,", "\n};")
    for text in (
        "const picturesPossible = data.pictures_possible !== false;",
        "const picturesOff = picturesPossible ? OE.el(",
        "const facePicturesButton = picturesPossible ? OE.el('button', {",
        "W.whatBare",
        "W.noResult(finishWords)",
    ):
        assert text in editor, text
    rows = _between(editor, "  function renderList(", "\n  }\n")
    assert "picturesPossible ? OE.el('button', {" in rows


def test_the_seen_instruction_does_not_repeat_the_hidden_answer() -> None:
    """Finding 14: the seen face's instruction drops "The map's class is not
    sent to this panel"; the lead and the blind line say it. r4-h S2
    (B/sheet-change-answer): "Right or wrong is worked out when each label is
    saved." gives way to the line that an answer can change until the finish
    button is pressed; S1 (B/sheet-blind-line): the blind line in its words."""
    script = _read("sheet.js")
    code = _code(script)
    assert "class is not sent to this panel" not in code
    assert "Right or wrong is worked out when each label is saved." not in code
    assert _holds(script, NO_RESULT)
    # r4-m M6: with who must not look at the map.
    assert _holds(
        script, "The map's answer is hidden: answer without looking at the map."
    )
    assert _holds(script, BLIND_SEEN)
    assert "so you label without seeing" not in code
