# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Round r5-c (17 Nov 2026), track PROJECTS: the projects panel
(views/studio.html) and its glyph (views/glyph.js), each item as
specs/r5-c-audit/build_list.md gives it. The views are read as text (no test
here runs the scripts):

- S02: an undated prediction's year slot says "no dates"; a model whose
  record's grain is the window is "per-tile" on its row and in the Models
  list; no word the glyph draws says "window";
- S04: the card's next line: the buttons' names in quotes in one sentence
  (r5-e S5: now the card's only line, without "Next: ");
- S05: a lone output draws no buttons of its own (the row's use it), and only
  a list of several opens with its first line, which a row's press on such
  a run leaves as the one line (r5-d R09: the line that said how many maps,
  a toast and then the row's own, is gone);
- S06: the note under the Predictions heading, and what the tick boxes do
  after it only where boxes are drawn; COMPARE.how lost that sentence
  (r5-e S5: COMPARE.tick and COMPARE.how in shorter words);
- S07: a prediction row's Outputs and jobs are a line of their own under the
  lead; at 480 px or less the lead wraps, the name at full width;
- S08: the values lines say which card tells; a failed row says why it has
  no button;
- S11: dates as "10 Nov 2026"; the In progress card is white;
- S12: a long project name shows on up to two lines;
- S14: a project with no prediction says "No maps yet" beside its ring.
"""

from __future__ import annotations

import re

from olmoearth_agent.mcp_app import views

#: S04: the projects card's lines (r5-e S5: one line, without "Next: ").
PROJECTS_LEAD = {
    "next": (
        'Press "Show its maps" on the project that holds your map, then '
        '"Check how wrong it is" beside the map.'
    ),
}

#: S05: the outputs list's first line (the toast for a run with n maps that
#: S05 kept went in r5-d R09).
OUTPUTS_FIRST = "This run made more than one map. Use the buttons on the one you mean."

#: S06: the note under the Predictions heading, what a tick box does, and how
#: a comparison works.
PREDICTIONS_NOTE = (
    "Each prediction is a model run over an area and a period; "
    "its output is the map."
)
#: r5-e S5: COMPARE.tick and COMPARE.how, shorter; the outcome that the
#: labels cannot tell stays, since the tool can return it.
COMPARE_TICK = "Tick two finished maps of the same area and dates to compare them."
COMPARE_HOW = (
    "You label only where the two maps differ. You see which is right more often "
    "there, or that your labels cannot tell, but not how accurate either map is."
)

#: S08: the lines this round changed or added under a prediction's row.
ROW_VALUES = (
    "Shows a number at each pixel. Check works if that number is a score from 0 to "
    "1; its Check card tells you."
)
ROW_VALUES_ANY = (
    "Shows a number. Check works if that number is a score from 0 to 1; its Check "
    "card tells you."
)
ROW_FAILED = "This run failed in Studio. There is no map to check."

#: Words no face holds (r4-i's rule), and no choice is marked recommended.
_OFF_FACE = re.compile(
    r"\b(?:windows?|squares?|budget|stratum|design|review set|recommended)\b", re.I
)


def _read(name: str) -> str:
    """A file of views/, as it is on disk."""
    return (views.VIEW_DIR / name).read_text(encoding="utf-8")


def _between(text: str, start: str, end: str) -> str:
    """The part of text from start up to the first end after it."""
    at = text.index(start)
    return text[at : text.index(end, at)]


def _code(script: str) -> str:
    """The text with its comments taken out (a block, or the rest of a line
    after two slashes)."""
    return re.sub(r"/\*.*?\*/|//[^\n]*", "", script, flags=re.S)


def _style() -> str:
    """The projects panel's own style sheet (its first style element)."""
    return _between(_read("studio.html"), "<style>", "</style>")


def _in_order(text: str, *pieces: str) -> bool:
    """Whether each piece is in text, each after the one before it."""
    at = [text.index(p) for p in pieces]
    return at == sorted(at)


def test_s02_no_dates_and_per_tile_and_no_window_on_the_face() -> None:
    """S02: the year slot of an undated prediction says "no dates" (its ring
    keeps the spoken name "Dates not recorded"); a grain of window reads
    "per-tile" in both the caption and the row's words, and a missing grain
    "per pixel or per tile not recorded"; the only literal holding "window"
    is the record's own grain value, compared and never shown."""
    glyph = _read("glyph.js")
    row = _between(glyph, "OE.glyph.predictionRow = (p) => {", "\n};")
    assert "    // WORDING: r5c S02\n" in row
    assert "'aria-hidden': 'true' }, p.years || 'no dates'));" in row
    assert "'not recorded'" not in row
    assert "const label = p.span ? 'Covers ' + p.span : 'Dates not recorded';" in row
    for start in (
        "OE.glyph.outputText = (m) => {",
        "OE.glyph.outputWords = (m) => {",
    ):
        block = _between(glyph, start, "\n};")
        assert (
            "const grain = m.grain === 'pixel' ? 'per-pixel ' : m.grain === 'window'"
            " ? 'per-tile ' : '';" in block
        ), start
    text = _between(glyph, "OE.glyph.outputText = (m) => {", "\n};")
    assert (
        "const noGrain = m.grain ? '' : ', per pixel or per tile not recorded';" in text
    )
    tag = "// WORDING: r4p2; r5c S02 (per-tile, per tile)\n"
    assert tag + "OE.glyph.outputText" in glyph
    words = _between(glyph, "OE.glyph.outputWords = (m) => {", "\n};")
    assert "  // WORDING: r5c S02\n  const grain = " in words
    code = _code(glyph)
    literals = re.findall(r"'((?:[^'\\\n]|\\.)*)'", code)
    shown = [s for s in literals if re.search(r"\bwindows?\b", s, re.I)]
    assert shown == ["window", "window", "window"]  # the tile's n and the two grains
    assert "per-window" not in code and "per window" not in code


def test_s04_the_projects_cards_lead() -> None:
    """S04: the buttons' names in quotes in one sentence, a short plain line
    with no barred word. r5-e S5: it is the card's only line (no "what", no
    "cannot", no "Next: "), marked r5e."""
    page = _read("studio.html")
    lead = _between(page, "// WORDING: r5e\nconst PROJECTS_LEAD = {", "\n};")
    for key, line in PROJECTS_LEAD.items():
        assert f"  {key}: '{line}',\n" in lead + "\n", key
        assert not _OFF_FACE.search(line), line
    assert lead.count("\n  ") == 1
    assert "OE.lead(PROJECTS_LEAD)" in page
    # One sentence for what to do: the buttons are named in quotes.
    assert PROJECTS_LEAD["next"].count(". ") == 0
    assert '"Show its maps"' in PROJECTS_LEAD["next"]
    assert '"Check how wrong it is"' in PROJECTS_LEAD["next"]
    for gone in (
        "Inside each are its predictions",
        "Their outputs are the maps",
        "Then press Check how wrong it is",
        "It cannot tell which",
        "holds the map you use",
    ):
        assert gone not in page, gone


def test_s05_a_lone_output_has_no_buttons_of_its_own() -> None:
    """S05: the outputs list's rows get their jobs only when the run made
    more than one map, and only then does the list open with its first line;
    a job pressed on the row of a run with several still opens the list,
    whose first line says so (r5-d R09: nothing else does); the words are
    marked r5c."""
    page = _read("studio.html")
    outputs = _between(page, "const OUTPUTS = {", "\n};")
    assert f"  first: '{OUTPUTS_FIRST}',\n" in outputs
    assert "several" not in outputs
    tag = "// WORDING: r4p2 (label, none, noneYet); r5c S05 (first)\n"
    assert tag + "const OUTPUTS = {" in page
    row = _between(page, "function predictionRow(", "\n}\n")
    fill = _between(row, "const fill = (out) => {", "\n  };")
    assert (
        "const rows = list.map((r) => resultRow(app, p, r, done && list.length > 1,"
        " kind));" in fill
    )
    assert (
        ": hasJobs && list.length > 1 ? [OE.el('li', { class: 'small faint' },"
        " OUTPUTS.first), ...rows] : rows));" in fill
    )
    # A row's job on a run with several: the list opens, and (r5-d R09) its
    # first line alone says so: no toast, no line in the row's slot.
    job = _between(row, "const job = (key) => ", "\n  });")
    assert _in_order(
        job,
        "    if (list.length > 1) {\n      show(true);\n",
        "await OE.sendJob(app, OE.JOB_TEXT[key]({ result_id: list[0].id,",
    )
    assert "OUTPUTS.several" not in page
    result = _between(page, "function resultRow(", "\n}\n")
    assert "const jobs = !done ? [] : [" in result
    for gone in ("What this run produced", "pick a job on one of them below"):
        assert gone not in page, gone
    assert not _OFF_FACE.search(OUTPUTS_FIRST)


def test_s06_the_tick_boxes_are_said_under_the_heading() -> None:
    """S06: the note under the Predictions heading is one sentence; where
    boxes are drawn (compare) COMPARE.tick follows it; the hint under the
    list keeps how a comparison works, without the boxes' sentence."""
    page = _read("studio.html")
    note = f"// WORDING: r5c S06\nconst PREDICTIONS_NOTE = '{PREDICTIONS_NOTE}';"
    assert note in page
    compare = _between(page, "const COMPARE = {", "\n};")
    assert f"  tick: '{COMPARE_TICK}',\n" in compare
    assert f"  how: '{COMPARE_HOW}',\n" in compare
    assert "  ready: 'Two ticked. Press the button to start.',\n" in compare + "\n"
    tag = "// WORDING: r4p2 (ready); r5e (tick, how: shorter, r5-e S5)\n"
    assert tag + "const COMPARE = {" in page
    summary = _between(page, "function projectSummary(", "\n}\n")
    assert (
        "compare ? compare.footer() : null, compare ? PREDICTIONS_NOTE + ' ' + "
        "COMPARE.tick : PREDICTIONS_NOTE)," in summary
    )
    # The note is drawn under the heading, before the rows and their boxes.
    card = _between(page, "function summaryCard(", "\n}\n")
    assert _in_order(
        card,
        "OE.el('h2', {}, title),",
        "note ? OE.el('div', { class: 'small muted' }, note) : null,",
        # r5-e S5: 'Recent' only when 'See all (n)' follows.
        "sorted.length ? [more ? caption : null, ul, more, footer]",
    )
    comparer = _between(page, "function comparer(app, project) {", "\n}\n")
    assert "'aria-live': 'polite' }, COMPARE.how);" in comparer
    assert "hint.textContent = two ? COMPARE.ready : COMPARE.how;" in comparer
    assert "Tick two finished maps of the same area and dates. You then" not in page
    assert "a prediction can have more than one output" not in page
    for line in (PREDICTIONS_NOTE, COMPARE_TICK, COMPARE_HOW):
        assert not _OFF_FACE.search(line), line


def test_s07_outputs_have_a_line_of_their_own_and_a_narrow_lead_wraps() -> None:
    """S07: the prediction row's lead is marked pred; its Outputs and jobs are
    a div of their own right after the spread (no longer inside it); the CSS
    gives that line a 4 px gap and, at 480 px or less, wraps the lead so the
    name's column takes the full width and the year slot keeps no width."""
    page = _read("studio.html")
    row = _between(page, "function predictionRow(", "\n}\n")
    lead = "OE.el('span', { class: 'row lead pred' }, box, OE.glyph.predictionRow(p),"
    assert lead in row
    # r5-d R09: the line is made before the row (show puts the row's jobs
    # away from it and back) and stands right after the spread.
    assert _in_order(
        row,
        "const acts = OE.el('div', { class: 'row pred-acts' }, toggle, jobs);",
        "OE.el('div', { class: 'spread' },",
        "p.status ? OE.chip(p.status, { ring: done }) : null))),\n    acts,\n",
        "says ? OE.el('div', { class: 'small faint pred-says' }, says) : null,",
        "slot, results);",
    )
    assert "OE.el('span', { class: 'row' }, toggle, jobs)" not in row
    style = _style()
    assert ".items .pred-acts { margin-top: 4px; }" in style
    assert (
        "  @media (max-width: 480px) {\n"
        "    .items .lead.pred { flex-wrap: wrap; }\n"
        "    .items .lead.pred > .lead-text { flex-basis: 100%; }\n"
        "    .items .lead.pred .glyph-year { min-width: 0; }\n"
        "  }\n" in style
    )
    # The rule is the panel's own: the model rows' lead keeps its one line.
    models = _between(page, "function modelsCard(", "\n}\n")
    assert "OE.el('span', { class: 'row lead' }, g.tile," in models


def test_s08_the_values_lines_and_the_failed_row() -> None:
    """S08: the values lines say the number, when Check works and which card
    tells; a failed row (not completed, status 'failed') says why it has no
    button, and a row still running says nothing; the row passes its
    status."""
    page = _read("studio.html")
    table = _between(page, "const ROW_SAYS = {", "\n};")
    for key, line in (
        ("values", ROW_VALUES),
        ("valuesAny", ROW_VALUES_ANY),
        ("failed", ROW_FAILED),
    ):
        assert f"  {key}: '{line}',\n" in table + "\n", key
        assert not _OFF_FACE.search(line), line
    assert ROW_VALUES_ANY == ROW_VALUES.replace(" at each pixel", "")
    assert "its card says so" not in page
    # r5-e S7: the classes line is gone, so the tag no longer names it.
    tag = "// WORDING: r5a V3 (detection, unknown, unread); r5c S08 "
    assert tag + "(values, valuesAny, failed); r5e (classes: none)\n" in page
    says = _between(
        page, "const rowSays = (kind, done, grain, unread, status) => ", ";\n"
    )
    assert _in_order(
        says,
        "kind === 'embeddings' ? NOT_A_MAP",
        "!done ? (status === 'failed' ? ROW_SAYS.failed : null)",
        "unread ? ROW_SAYS.unread",
    )
    row = _between(page, "function predictionRow(", "\n}\n")
    assert (
        "const says = rowSays(kind, done, model && model.grain, unread, p.status);"
        in row
    )


def test_s11_dates_name_the_month_and_the_in_progress_card_is_white() -> None:
    """S11: a sheet's last save and a project's creation are written "10 Nov
    2026" (the day, the month's short name from a fixed table, the year), by
    one function, which reads no locale data; the compact rows' Created column
    drops "Created " as before; no date is month/day/year any more; the In
    progress card is not the cream summary card."""
    page = _read("studio.html")
    # The month's name from the fixed table: the browser's en-GB names
    # September "Sept".
    assert (
        "const dayOf = (d) => d.getDate() + ' ' + OE.glyph.MONTHS[d.getMonth()] + "
        "' ' + d.getFullYear();" in page
    )
    saved = _between(page, "const savedOn = (iso) => {", "\n};")
    assert "  // WORDING: r5c S11\n" in saved
    assert "? ' Last saved ' + dayOf(d) + '.' : '';" in saved
    created = _between(page, "const created = (iso) => {", "\n};")
    assert "  // WORDING: r5c S11\n" in created
    assert "? 'Created ' + dayOf(d) : '';" in created
    assert "(d.getMonth() + 1) + '/'" not in page
    assert "toLocaleString" not in page
    assert page.index("const dayOf = ") < page.index("const savedOn = ")
    row = _between(page, "function projectRow(app, p) {", "\n}\n")
    assert "created(p.creation_time).replace('Created ', '') || '-'" in row
    card = _between(page, "function inProgressCard(app, s) {", "\n}\n")
    assert "OE.el('div', { class: 'card in-progress', hidden: true });" in card
    assert "summary in-progress" not in page
    # The In progress card's own rules still apply (they key on in-progress).
    assert ".in-progress .items { display: grid;" in _style()


def test_s12_a_long_project_name_shows_on_two_lines() -> None:
    """S12: the project's name in its title button is clamped at two lines,
    wrapping anywhere, no longer cut to one line with an ellipsis; the whole
    name stays in the button's title."""
    style = _style()
    rule = _between(style, ".title .name-text {", "}")
    for text in (
        "min-width: 0;",
        "overflow: hidden;",
        "overflow-wrap: anywhere;",
        "display: -webkit-box;",
        "-webkit-line-clamp: 2;",
        "-webkit-box-orient: vertical;",
    ):
        assert text in rule, text
    assert "text-overflow" not in rule and "nowrap" not in rule
    title = _between(_read("studio.html"), "const titleButton = (name) => ", ";\n")
    assert "class: 'title', title: name," in title
    assert "OE.el('span', { class: 'name-text' }, name)" in title


def test_s14_a_project_with_no_prediction_says_no_maps_yet() -> None:
    """S14: beside the empty ring of a project with no prediction, "No maps
    yet"; "Show its maps" stays on every project. r5-e S7: the ring's long
    caption is said once above the cards (studio.html RING_SAYS), so the
    other projects say nothing there unless they differ."""
    glyph = _read("glyph.js")
    card = _between(glyph, "OE.glyph.projectCard = (g) => {", "\n};")
    assert (
        "const note = g.count === 0 ? 'No maps yet' : !colors ? 'dates not "
        "recorded'" in card
    )
    assert "months of the year its predictions cover" not in card
    assert "// WORDING: r4p2 (B/P3); r5c S14" in card
    page = _read("studio.html")
    for start in ("function projectCard(app, p) {", "function projectRow(app, p) {"):
        assert "const show = showButton(p.name);" in _between(page, start, "\n}\n")


def test_r5c_every_changed_string_is_marked() -> None:
    """Every string this track changed sits under a WORDING: r5c mark (the
    rule of the round), so each can be found with one search; those that
    round r5-e changed again (S04's line, S06's tick and how: r5-e S5) are
    marked r5e."""
    page = _read("studio.html")
    glyph = _read("glyph.js")
    for tag in ("S06", "S11"):
        assert f"// WORDING: r5c {tag}" in page, tag
    assert "r5c S05 (first)" in page and "r5c S08 (values" in page
    assert "// WORDING: r5e\nconst PROJECTS_LEAD = {" in page
    assert "r5e (tick, how: shorter, r5-e S5)" in page
    for tag in ("S02", "S14"):
        assert f"r5c {tag}" in glyph, tag
