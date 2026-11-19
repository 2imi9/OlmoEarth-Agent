# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Round r5-e (17 Nov 2026), track START: the start card, the browser
sign-in page and the projects panel, each item as
specs/r5-e-design/build_list.md gives it. The views are read as text (no
test here runs the scripts); the sign-in page through its own template:

- S1: the start card has one line under its title, what to do next (the
  title says what the card is; what a check cannot tell is said where
  labelling happens);
- S2: each job's line sits inside its button, and the job buttons are one
  column of tiles of one size; each line is at most 85 characters, and the
  check's and the run's lines are the strip's plain words; signed out, why
  the tiles are off stands over them, inside the jobs card;
- S3: the overview figure, its caption and the docs link sit in one white
  box, as wide as the figure;
- S4: the browser sign-in page says what to paste and where the key goes
  in two short paragraphs, no longer that Studio checks the key;
- S5: the projects panel has one line under its heading; the tick boxes'
  line and how a comparison works are shorter; "Recent" shows only when
  "See all (n)" follows;
- S6: a closed project card's bottom row sits at its foot, and a row's job
  buttons start at one left edge whether its outputs are open or not;
- S7: what is the same on every item is said once: the ring's caption above
  the cards (shown while a card has its ring), no "No description", no line
  under a class map's row;
- S8: the narrow strip's example map is a finished Studio map from the docs
  (a JPEG of views/brand, in the start view alone), no longer a tile drawn
  in made-up colours.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from olmoearth_agent.mcp_app import auth, views

#: S1: the start card's one line, signed out and signed in.
START_NEXT = "Sign in first, then press a job or ask Claude in the chat."
START_NEXT_IN = "Press a job, or ask Claude in the chat."

#: S2: the check's and the run's lines (the strip's words), and the longest a
#: job's line may be inside its button (two lines at 380 px).
CHECK_LINE = (
    "You check random squares of the map; you get how often it is wrong, with a "
    "range."
)
RUN_LINE = (
    "Makes a new map once you pick a model, an area and dates. Needs a key that can "
    "write."
)
WHICH_LINE = "Label only where the two maps differ; see which is right more often."
LINE_MAX = 85

#: S4: the browser sign-in page's two paragraphs, their words joined.
PAGE_FIRST = (
    "Paste your Studio API key (in Studio: your profile, then API Keys). It goes "
    "from this computer to the OlmoEarth Agent, not through Claude."
)
PAGE_SECOND = (
    "A read-only key can check maps; to save an area or run a model, use one that "
    "can write."
)

#: S5: the projects panel's one line, and the shorter tick and compare lines.
PROJECTS_NEXT = (
    'Press "Show its maps" on the project that holds your map, then '
    '"Check how wrong it is" beside the map.'
)
COMPARE_TICK = "Tick two finished maps of the same area and dates to compare them."
COMPARE_HOW = (
    "You label only where the two maps differ. You see which is right more often "
    "there, or that your labels cannot tell, but not how accurate either map is."
)

#: S7: what the ring on a project shows, said once above the cards (as the JS
#: string has it: the apostrophe escaped). It names the ring on a project: an
#: open project's prediction rows have rings of another kind.
RING_SAYS = (
    "The ring on each project shows the months of the year that project\\'s "
    "predictions cover, all years and areas counted together."
)

#: S8: the example map's file, its size, and the largest it may be.
MAP_FILE = "studio-map-example.jpg"
MAP_SIZE = (160, 160)
MAP_MAX_BYTES = 30 * 1024

#: JPEG markers that start a frame (they carry its height and width).
_SOF = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}

#: Words no face holds (r4-i's rule, R7), and no choice is marked recommended.
_OFF_FACE = re.compile(
    r"\b(?:windows?|UTM|zones?|intervals?|strat(?:um|a)|design|budget|recommended)\b",
    re.I,
)

#: A colour written out: a hash and three to eight hex digits.
_HEX = re.compile(r"#[0-9a-fA-F]{3,8}\b")


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


def _in_order(text: str, *pieces: str) -> bool:
    """Whether each piece is in text, each after the one before it."""
    at = [text.index(p) for p in pieces]
    return at == sorted(at)


def _style(name: str) -> str:
    """A page's own style sheet (its first style element)."""
    return _between(_read(name), "<style>", "</style>")


def _rule(style: str, selector: str) -> str:
    """The body of the first rule of style whose selector is selector."""
    return _between(style, selector + " {", "}")


def _words(html: str) -> str:
    """The words of an HTML fragment: its tags dropped, its spaces joined."""
    return " ".join(re.sub(r"<[^>]+>", " ", html).split())


def _jpeg_size(data: bytes) -> tuple[int, int]:
    """A JPEG's width and height, read from its first start-of-frame marker."""
    assert data[:2] == b"\xff\xd8", data[:4]
    at = 2
    while at + 4 <= len(data):
        assert data[at] == 0xFF, at
        marker = data[at + 1]
        if marker == 0xFF:  # a fill byte before the marker
            at += 1
            continue
        if marker == 0x01 or 0xD0 <= marker <= 0xD8:  # markers with no length
            at += 2
            continue
        if marker in _SOF:
            height = int.from_bytes(data[at + 5 : at + 7], "big")
            width = int.from_bytes(data[at + 7 : at + 9], "big")
            return width, height
        at += 2 + int.from_bytes(data[at + 2 : at + 4], "big")
    raise AssertionError("no start-of-frame marker")


# --------------------------------------------------------------------------- S1


def test_s1_the_start_card_says_one_line_what_to_do_next() -> None:
    """S1: START_LEAD holds the next line alone (no "what", which said the
    title again, no "cannot"), and START_NEXT_IN its signed-in form, each
    marked r5e; the card draws its lead as before, with no fold; neither
    line holds a word a face never says."""
    page = _read("start.html")
    lead = f"// WORDING: r5e\nconst START_LEAD = {{ next: '{START_NEXT}' }};\n"
    assert lead in page
    assert f"// WORDING: r5e\nconst START_NEXT_IN = '{START_NEXT_IN}';\n" in page
    render = _between(page, "function render(app, data) {", "\n}\n")
    assert (
        "const lead = OE.lead(on ? { ...START_LEAD, next: START_NEXT_IN } "
        ": START_LEAD);" in render
    )
    for gone in (
        "Jobs this app can do with your Studio maps",
        "This card knows nothing about your map yet",
        "Sign in below first",
        "Press a job below",
        "three plain lines",
    ):
        assert gone not in page, gone
    assert "OE.fold(" not in page and "<details" not in page
    for line in (START_NEXT, START_NEXT_IN):
        assert not _OFF_FACE.search(line), line
        assert len(line) <= 70, line  # one line at 780 px


# --------------------------------------------------------------------------- S2


def test_s2_each_jobs_line_is_inside_its_button() -> None:
    """S2: a job's button holds two spans, its label and its line; its name
    is the label alone (aria-label) and the line its description
    (aria-describedby, the line's id from the job's key); the row holds the
    button and the slot, no loose line under it."""
    page = _read("start.html")
    row = _between(page, "function jobRow(app, job, on, press = null) {", "\n}\n")
    for text in (
        "const lineId = 'job-line-' + job.key;",
        "OE.el('button', { type: 'button', class: 'job' + (main && on ? ' primary' "
        ": ''), disabled: !on, 'aria-label': job.label, 'aria-describedby': lineId },",
        "OE.el('span', { class: 'job-label' }, job.label), "
        "OE.el('span', { class: 'job-line', id: lineId }, job.line));",
        "return OE.el('li', { class: main ? 'main' : null }, button, slot);",
    ):
        assert text in row, text
    assert "small muted line" not in page and "}, job.label);" not in row
    assert _in_order(row, "const lineId = ", "const button = ", "return OE.el('li'")


def test_s2_the_job_tiles_are_one_column_of_one_size() -> None:
    """S2: the jobs are one column at every width (no two-column grid, no
    subgrid, no 560 px rule); each tile is full width and at least 86 px
    tall, its label over its line from one left edge; the line keeps a
    readable colour on a tile that is off (signed out), and so does the label
    (r5-f F03: it took the disabled grey, 1.9:1 on white)."""
    style = _style("start.html")
    grid = ".items.jobs { display: grid; grid-template-columns: minmax(0, 1fr); }"
    assert grid in style
    assert ".jobs li { padding: 4px 0; }" in style
    tile = _rule(style, ".jobs li > button.job")
    for part in (
        "width: 100%;",
        "min-height: 86px;",
        "display: flex;",
        "flex-direction: column;",
        "justify-content: center;",
        "align-items: flex-start;",
        "gap: 2px;",
        "padding: 10px 16px;",
        "text-align: left;",
    ):
        assert part in tile, part
    assert ".jobs .job-label { font-weight: 600; }" in style
    assert (
        ".jobs .job-line { font-size: 12px; line-height: 18px; font-weight: 400; }"
        in style
    )
    assert (
        ".jobs button.job:disabled .job-label, .jobs button.job:disabled .job-line "
        "{ color: var(--text-2); }" in style
    )
    assert (
        ".jobs li.jobs-head > h2 { font-size: 12px; line-height: 18px; "
        "font-weight: 600; color: var(--text-2); }" in style
    )
    for gone in ("subgrid", "max-content", "@media (max-width: 560px)", "> .line"):
        assert gone not in style, gone


def test_s2_every_jobs_line_fits_two_lines_in_its_tile() -> None:
    """S2: each job's line is at most 85 characters (two lines at 380 px
    inside its tile); the check's says the strip's words, no longer the parts
    a map is saved in, and the run's what its press waits for; both marked
    r5e; the line on which of two maps is as it was."""
    pytest.importorskip("mcp")
    from olmoearth_agent.mcp_app import server

    lines = {job["key"]: job["line"] for job in server.START_JOBS}
    for key, line in lines.items():
        assert len(line) <= LINE_MAX, (key, len(line))
    assert lines["check"] == CHECK_LINE and lines["run"] == RUN_LINE
    assert lines["which"] == WHICH_LINE
    for line in (CHECK_LINE, RUN_LINE):
        assert not _OFF_FACE.search(line), line
        assert not re.search(r"\d|%", line), line
    assert "part" not in CHECK_LINE
    source = Path(server.__file__).read_text(encoding="utf-8")
    assert (
        '        # WORDING: r5e\n        "line": "You check random squares of the '
        'map; you get how often it is "\n' in source
    )
    assert (
        '        # WORDING: r5e\n        "line": "Makes a new map once you pick a '
        'model, an area and dates. Needs "\n' in source
    )
    for gone in ("each part it is saved in", "Pressing this starts nothing"):
        assert gone not in source, gone


def test_s2_why_the_tiles_are_off_stands_over_them() -> None:
    """S2 (R2, after review): signed out, 'Sign in to start a job.' is the
    jobs card's first block, over the tiles, not a loose line under the last
    tile's box; signed in it is not drawn and the card holds the list alone.
    Its words are as they were; since r5-i B8 it is at body size in the body
    colour (no class)."""
    page = _read("start.html")
    render = _between(page, "function render(app, data) {", "\n}\n")
    assert _in_order(
        render,
        "root.append(OE.el('section', { class: 'card stack' },",
        "on ? null : OE.el('div', {}, 'Sign in to start a job.'),",
        "OE.el('ul', { class: 'items jobs', 'aria-label': 'Jobs' }, "
        "jobRows(app, jobs, on))));",
    )
    assert render.count("Sign in to start a job.") == 1


# --------------------------------------------------------------------------- S3


def test_s3_the_figure_caption_and_link_sit_in_one_white_box() -> None:
    """S3: the start card's wrapper is the white box (the views' paper, a
    hairline divider, 12 px corners) that holds the figure, its caption and
    the docs link; the docs picture keeps no box of its own; the caption
    stays outside the role=img element, as before. The box is as wide as its
    figure, not the panel: the strip's 440 px, and from 680 px the docs
    figure's 748 px, each plus the box's 24 px of padding and 2 px of edge
    (no blank white band beside the figure); the strip's two light boxes are
    the canvas colour, not white boxes inside the white box."""
    style = _style("start.html")
    wrap = _rule(style, ".oe-overview-wrap")
    for part in (
        "background: var(--paper);",
        "border: 1px solid var(--divider);",
        "border-radius: 12px;",
        "padding: 8px 12px 10px;",
        "max-width: 466px;",
    ):
        assert part in wrap, part
    wide = "@media (min-width: 680px) { .oe-overview-wrap { max-width: 774px; } }"
    assert _in_order(style, ".oe-overview-wrap {", wide)
    assert ".oe-overview-cap { margin-top: 6px; }" in style
    assert ".oe-overview-cap { margin-top: 4px; }" not in style
    overview = _between(_read("overview.js"), "  const STYLE = `", "`;")
    assert (
        ".oe-overview img.wide { display: none; width: 100%; max-width: 748px; "
        "height: auto; border-radius: 8px; }" in overview
    )
    assert "border: 1px solid" not in overview and "background:" not in overview
    assert ".oe-overview svg.narrow { max-width: 440px; }" in overview
    assert (
        ".oe-overview .box { fill: var(--canvas); stroke: var(--divider); "
        "stroke-width: 1; }" in overview
    )
    block = _between(_read("start.html"), "function overviewBlock(app) {", "\n}\n")
    assert _in_order(
        block,
        "OE.el('div', { class: 'oe-overview-wrap' },",
        "OE.overview.figure(),",
        "OE.el('div', { class: 'small muted oe-overview-cap' },",
        "' ', docs, address));",
    )


# --------------------------------------------------------------------------- S4


def test_s4_the_sign_in_page_says_it_in_two_short_paragraphs() -> None:
    """S4: the page's two paragraphs before its field say what to paste and
    where Studio makes a key, where the key goes, and which key does what;
    it no longer says that the page runs on this computer or that Studio
    checks the key (a wrong key shows its own error); marked r5e; the field
    and its label are as they were."""
    form = auth._FORM.format(error="")
    head = _between(form, "</h1>", "<form")
    paragraphs = re.findall(r"<p>(.*?)</p>", head, re.S)
    assert [_words(p) for p in paragraphs] == [PAGE_FIRST, PAGE_SECOND]
    for gone in (
        "This page runs on this computer",
        "checked with Studio",
        "is enough to check maps",
    ):
        assert gone not in form, gone
    assert '<label for="api-key" class="field">Studio API key</label>' in form
    source = Path(auth.__file__).read_text(encoding="utf-8")
    assert (
        "#: WORDING: r5e (the two paragraphs); r5c S09 (the field's label and "
        "placeholder)\n_FORM = " in source
    )
    for text in (PAGE_FIRST, PAGE_SECOND):
        assert not _OFF_FACE.search(text), text
    # About four lines at the page's 480 px: under the six it had.
    assert len(PAGE_FIRST) + len(PAGE_SECOND) <= 240


# --------------------------------------------------------------------------- S5


def test_s5_the_projects_panel_says_one_line() -> None:
    """S5: PROJECTS_LEAD holds the next line alone, without "Next: " (no
    "what", which said the heading and the Predictions note again, no
    "cannot", which the server tells Claude), marked r5e."""
    page = _read("studio.html")
    lead = _between(page, "// WORDING: r5e\nconst PROJECTS_LEAD = {", "\n};")
    assert lead.endswith(f"\n  next: '{PROJECTS_NEXT}',")
    assert lead.count("\n  ") == 1
    assert "OE.lead(PROJECTS_LEAD)" in page
    for gone in (
        "Your Studio projects. Each holds",
        "This app cannot tell which of several similar projects",
        "Next: press",
    ):
        assert gone not in page, gone
    assert not _OFF_FACE.search(PROJECTS_NEXT)


def test_s5_the_tick_and_compare_lines_are_shorter() -> None:
    """S5: what the tick boxes are for and how a comparison works, in fewer
    words; that the labels can fail to tell stays (the tool can return it);
    marked r5e."""
    page = _read("studio.html")
    compare = _between(page, "const COMPARE = {", "\n};")
    assert f"  tick: '{COMPARE_TICK}',\n" in compare
    assert f"  how: '{COMPARE_HOW}',\n" in compare
    assert "// WORDING: r4p2 (ready); r5e (tick, how: shorter, r5-e S5)\n" in page
    assert "cannot tell" in COMPARE_HOW and "not how accurate" in COMPARE_HOW
    for gone in ("to say which is more accurate", "You label a sample of the places"):
        assert gone not in page, gone
    for line in (COMPARE_TICK, COMPARE_HOW):
        assert not _OFF_FACE.search(line), line
        assert not re.search(r"\bsquares?\b", line), line


def test_s5_recent_shows_only_when_see_all_follows() -> None:
    """S5: a summary card's 'Recent' caption is drawn only when it has a
    'See all (n)' button (more than five items); a short list stands under
    its heading alone."""
    card = _between(_read("studio.html"), "function summaryCard(", "\n}\n")
    assert "sorted.length ? [more ? caption : null, ul, more, footer]" in card
    assert "[caption, ul, more, footer]" not in card
    assert (
        "const more = sorted.length > RECENT ? OE.el('button', { class: 'link', "
        "'aria-expanded': 'false' }) : null;" in card
    )


# --------------------------------------------------------------------------- S6


def test_s6_closed_cards_end_on_one_line_and_jobs_start_at_one_edge() -> None:
    """S6: a closed project card is a column whose bottom row takes the
    space left (margin-top: auto), so cards in one grid row end their
    buttons on one line; an open card keeps block layout; in the column the
    ring gives up its own top margin (margins do not collapse there), so it
    sits as far under the description closed as open and does not move when
    a card opens; the outputs text button is as wide as the wider of its
    labels (r5-i B6: "Show outputs", 100 px; it was 92 px for "Hide
    outputs"), so the job buttons after it start at one left edge. No word
    changes."""
    style = _style("studio.html")
    assert _in_order(
        style,
        ".project.open { grid-column: 1 / -1; cursor: default; }",
        ".project:not(.open) { display: flex; flex-direction: column; }",
        ".project:not(.open) > .spread { margin-top: auto; }",
        ".project:not(.open) > .glyph-slot > .glyph-card { margin-top: 0; }",
    )
    # The margins that rule evens out: the description's 8 px under it (below)
    # and the ring's own 4 px over it, which fold into 8 px in an open card.
    ring = _between(_read("glyph.js"), "\n.glyph-card {", "}")
    assert "margin: 4px 0 8px;" in ring
    assert _in_order(
        style,
        ".items .pred-acts { margin-top: 4px; }",
        ".items .pred-acts > button.link { min-width: 100px; text-align: left; }",
        "@media (max-width: 480px) {",
    )
    card = _between(_read("studio.html"), "function projectCard(app, p) {", "\n}\n")
    assert "OE.el('div', { class: 'spread' }, when, OE.el('span'," in card
    desc = _rule(style, ".project .desc")
    assert "margin: 4px 0 8px;" in desc and "min-height: 42px;" in desc


# --------------------------------------------------------------------------- S7


def test_s7_the_rings_caption_is_said_once_above_the_cards() -> None:
    """S7: a project card's note beside its ring is drawn only where that
    card differs ("No maps yet", "dates not recorded", "n without dates");
    the ring's caption is RING_SAYS (it names the ring on a project), made
    once by the panel, after the search, in one block with the cards (6 px
    over them: not a loose line under the search field); it is made hidden
    and shown only while a card in view has its ring (ringLine, called when
    the cards are filled and when the rings' read answers, which never moves
    the cards); none with compact rows; the spoken months per step stay."""
    glyph = _read("glyph.js")
    card = _between(glyph, "OE.glyph.projectCard = (g) => {", "\n};")
    assert (
        "const note = g.count === 0 ? 'No maps yet' : !colors ? 'dates not recorded' "
        ": undated > 0 ? undated + ' without dates' : null;" in card
    )
    assert "note ? OE.el('div', { class: 'small faint' }, note) : null," in card
    assert "'Predictions per month: ' + stepWords(g.months)" in card
    assert "months of the year its predictions cover" not in glyph
    assert "r5e (the ring's caption said once, above the cards)" in card
    page = _read("studio.html")
    assert f"// WORDING: r5e\nconst RING_SAYS = '{RING_SAYS}';\n" in page
    assert "months of the year its predictions cover" not in RING_SAYS
    projects = _between(page, "function renderProjects(", "\n}\n")
    made = (
        "const ringSays = !compact && list.length ? OE.el('div', { class: 'small "
        "muted', id: 'oe-ring-says', hidden: true }, RING_SAYS) : null;"
    )
    shown = (
        "const ringLine = () => { if (ringSays) ringSays.hidden = "
        "!target.querySelector('.glyph-card'); };"
    )
    block = (
        "ringSays ? OE.el('div', { class: 'ring-block' }, ringSays, container) "
        ": container,\n  ].filter(Boolean));"
    )
    assert _in_order(projects, made, shown, "const fill = () => {")
    assert _in_order(projects, "inProgressCard(app, onScreen), search,", block)
    fill = _between(projects, "const fill = () => {", "\n  };")
    assert fill.rstrip().endswith("ringLine();")
    then = _between(projects, ".then((out) => {", "\n      })")
    assert _in_order(then, "it.setGlyph(", "ringLine();")
    assert "fill()" not in _code(then)
    assert projects.count("ringLine();") == 2
    rule = ".ring-block > #oe-ring-says { margin-bottom: 6px; }"
    assert rule in _style("studio.html")
    assert _code(page).count("RING_SAYS") == 2
    assert not _OFF_FACE.search(RING_SAYS)


def test_s7_no_description_and_no_line_under_a_class_row() -> None:
    """S7: a project with no description shows none (no 'No description'
    under every such name; the description keeps its height, so the cards
    line up); a class map's row has no line under it (its Check beside it
    says it), while values, detection, unknown, unread, failed and
    embeddings rows keep theirs."""
    page = _read("studio.html")
    desc = "const descText = (p) => (p.description ? String(p.description) : '');"
    assert desc in page
    assert "No description" not in _code(page)
    assert "min-height: 42px;" in _rule(_style("studio.html"), ".project .desc")
    table = _between(page, "const ROW_SAYS = {", "\n};")
    assert "classes:" not in table and "Shows classes." not in page
    keys = re.findall(r"^  (\w+): '", table, re.M)
    assert keys == ["values", "valuesAny", "detection", "unknown", "unread", "failed"]
    says = _between(
        page, "const rowSays = (kind, done, grain, unread, status) => ", ";\n"
    )
    assert (
        ": kind === 'classes' ? null : kind === 'detection' ? ROW_SAYS.detection "
        ": ROW_SAYS.unknown)" in says
    )
    assert "includes(kind)" not in says
    assert "kind === 'embeddings' ? NOT_A_MAP" in says


# --------------------------------------------------------------------------- S8


def test_s8_the_example_map_is_a_jpeg_in_the_start_view_only() -> None:
    """S8: the narrow strip's example map is one BRAND entry, a JPEG in
    views/brand of 160 x 160 under 30 KB, inlined into the start card's page
    alone as a JPEG data URL; no page holds its placeholder; the start view
    still holds one PNG (the AGENT tag). Each docs picture's data URL is in
    the start page once (a placeholder named in a comment would be filled in
    too, and carry the picture a second time)."""
    assert views.BRAND["MAP_EXAMPLE_DATA_URL"] == "brand/" + MAP_FILE
    path = views.VIEW_DIR / views.BRAND["MAP_EXAMPLE_DATA_URL"]
    data = path.read_bytes()
    assert data[:3] == b"\xff\xd8\xff" and len(data) < MAP_MAX_BYTES
    assert _jpeg_size(data) == MAP_SIZE
    url = views.brand_data_url("MAP_EXAMPLE_DATA_URL")
    assert url.startswith("data:image/jpeg;base64,")
    for name in views.VIEWS:
        page = views.view_html(name)
        assert (url in page) is (name == "start"), name
        assert "@MAP_EXAMPLE_DATA_URL@" not in page, name
    start = views.view_html("start")
    assert start.count(url) == 1
    assert start.count(views.brand_data_url("OVERVIEW_DATA_URL")) == 1
    assert start.count("data:image/png;base64,") == 1


def test_s8_the_strip_draws_the_picture_where_the_tile_was() -> None:
    """S8: overview.js takes the picture as MAP_EXAMPLE and draws it 40 x 40
    at the tile's place in the first box (sliced to fill its square), with a
    thin edge over it; the made-up tile, its colours and its patches are
    gone, so no colour is written out; the picture sits inside the strip's
    aria-hidden drawing; the box keeps its words, and the caption and the
    figure's name are as they were (no number, place or class name added)."""
    script = _read("overview.js")
    code = _code(script)
    assert "  const MAP_EXAMPLE = '@MAP_EXAMPLE_DATA_URL@';\n" in script
    narrow = _between(script, "  const narrow = () => ", "  const figure = () => {")
    first = _between(narrow, "inbox('box', 1, 1, 100, 94, 8,", "arrow('M103")
    assert _in_order(
        first,
        "text('t', 11, 21, WORDS.narrowStudio[0])",
        "text('s', 11, 37, WORDS.narrowStudio[1])",
        "svg('image', { href: MAP_EXAMPLE, x: 11, y: 46, width: 40, height: 40, "
        "preserveAspectRatio: 'xMidYMid slice' }),",
        "svg('rect', { class: 'tile-edge', x: 11, y: 46, width: 40, height: 40, "
        "rx: 2 })),",
    )
    assert narrow.count("href: MAP_EXAMPLE") == 1
    assert len(re.findall(r"\bMAP_EXAMPLE\b", code)) == 2
    assert "'aria-hidden': 'true', focusable: 'false' }," in narrow
    for gone in ("TILE", "PATCHES", "tile("):
        assert gone not in code, gone
    assert _HEX.findall(script) == []
    assert "data:" not in code and "xlink" not in code
    assert "WORDS.narrowStudio" in first


# --------------------------------------------------------------------------- all


def test_r5e_every_changed_string_is_marked() -> None:
    """Every string this track added or changed sits under a WORDING: r5e mark
    (the rule of the round), so each can be found with one search."""
    start = _read("start.html")
    studio = _read("studio.html")
    glyph = _read("glyph.js")
    assert start.count("// WORDING: r5e\n") == 2
    for name in ("PROJECTS_LEAD", "RING_SAYS"):
        assert f"// WORDING: r5e\nconst {name} = " in studio, name
    assert "r5e (tick, how" in studio and "r5e (classes: none)" in studio
    assert "r5e (the ring's caption" in glyph
    pytest.importorskip("mcp")
    from olmoearth_agent.mcp_app import server

    source = Path(server.__file__).read_text(encoding="utf-8")
    jobs = _between(source, "START_JOBS: tuple[dict[str, Any], ...] = (", "\n)\n")
    assert jobs.count("# WORDING: r5e") == 2
