# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Round r5-f (18 Nov 2026), track START, as specs/r5-f-round.md gives it: the
start card, the projects panel, the area map, the shared job button and the
view shell, items of the audit after the design round
(specs/r5-e-audit/build_list.md: F03, F08, F13, F15, F16 and F17). The views
are read as text, as they are on disk (no test here runs the scripts):

- F03: signed out, a job tile's label keeps the line's readable colour (it
  took the disabled grey, 1.9:1 on white); a field's grey hint is the text's
  colour at .7 (at .6 it was 4.0:1 on white, under 4.5:1);
- F08: a job button's slot says "Copy it into the chat:" over a message the
  host did not take (it said "Type it in the chat:"); the box under the line
  still selects whole on one click;
- F13: a run of several maps opens its list with a line that says Studio
  names them only by the ids under it, and whom to ask;
- F15: the area map's one instruction is its first line under the map, at
  body size, before the fields and the saved areas;
- F16: every view's head gives the page OE.STYLE's canvas and text colours
  and a system sans-serif before any script runs, so a card that cannot
  start is not default serif text on a white page;
- F17 (harness only): the stand-ins the browser harness now uses are the
  app's own words, pinned here at their source.
"""

from __future__ import annotations

import re

from olmoearth_agent.mcp_app import imagery, sheet, views

#: F08: what a job button's slot says over a message the host did not take.
JOB_NOT_SENT = "Claude did not get this message. Copy it into the chat:"

#: F13: the first line of a run's list of several outputs.
OUTPUTS_FIRST = (
    "This run made more than one map; Studio names them only by the ids below. "
    "Not sure which is yours? Ask whoever made the run."
)

#: F15: the area map's instruction before anything is drawn (unchanged words).
MAP_WHERE = (
    "Draw a rectangle or polygon with the tools on the map, or pick a saved area."
)

#: F16: the two rules each view's head carries before its scripts.
STATIC_HTML = "html { background: #faf2e9; }"
STATIC_BODY = (
    "body { margin: 0; padding: 0 16px 16px; font: 14px/21px system-ui, "
    "-apple-system, 'Segoe UI', sans-serif; color: #0a3235; }"
)
#: The pages that already had a style element (the rules are put first in
#: it); the label sheet and the sign-in view get one of their own.
STYLED = ("start", "check", "result", "studio", "map")

#: F17: the app's words the harness's stand-ins now are: the first sentence
#: of the pictures question, the opening both test questions share, what a
#: certificate plan's sheet is for, the refusal of a part smaller than one
#: square, and the heading of a sheet of a map in parts.
ASK_FIRST = "Show satellite (Sentinel-2) pictures of each square?"
TEST_WHY = (
    "Not sure yet? See how the pictures look on a public test place first. "
    "Nothing about your map is sent."
)
CERTIFY_JOB = "Labelling for a zone test (it may certify nothing)"
PATCH_REFUSAL = 'f"patch {patch} px is larger than the {height} x {width} map"'
PART_HEADING = 'SHEET_PART_HEADING = "{part} of this map ({index} of {n})"'

#: The start of the watchdog's classic script (r4-k K2), and of the module.
WATCHDOG_START = "<script>\n// WORDING: r4k K2\n"
MODULE_START = '<script type="module">'

#: Words no face holds (r4-i's rule, R7), and no choice is marked recommended.
_OFF_FACE = re.compile(
    r"\b(?:windows?|UTM|zones?|intervals?|strat(?:um|a)|design|budget|recommended)\b",
    re.I,
)


def _read(name: str) -> str:
    """A file of views/, as it is on disk."""
    return (views.VIEW_DIR / name).read_text(encoding="utf-8")


def _source(*parts: str) -> str:
    """A source file of the package (olmoearth_agent/<parts>), as text."""
    return views.VIEW_DIR.parent.parent.joinpath(*parts).read_text(encoding="utf-8")


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


def _first_style(page: str) -> str:
    """The text inside a page's first style element."""
    start = page.index("<style>") + len("<style>")
    return page[start : page.index("</style>", start)]


def _flat(css: str) -> str:
    """CSS with its comments taken out and its spaces joined."""
    return " ".join(re.sub(r"/\*.*?\*/", " ", css, flags=re.S).split())


def _contrast(rgb: tuple[int, int, int], alpha: float = 1.0) -> float:
    """The contrast ratio (WCAG 2.1) of a colour drawn at alpha over white."""
    lit = []
    for c in rgb:
        s = (alpha * c + (1 - alpha) * 255) / 255
        lit.append(s / 12.92 if s <= 0.04045 else ((s + 0.055) / 1.055) ** 2.4)
    lum = 0.2126 * lit[0] + 0.7152 * lit[1] + 0.0722 * lit[2]
    return 1.05 / (lum + 0.05)


# --------------------------------------------------------------------------- F03


def test_f03_a_tile_that_is_off_keeps_its_label_readable() -> None:
    """F03: signed out, a job tile's label has the line's secondary text
    colour, not button:disabled's grey; the tile still reads as off (that
    rule, with its transparent fill, is unchanged). A field's grey hint is
    the text's colour at .7, at least 4.5:1 on a white field."""
    page_style = _first_style(_read("start.html"))
    assert (
        ".jobs button.job:disabled .job-label, .jobs button.job:disabled .job-line "
        "{ color: var(--text-2); }" in page_style
    )
    # One rule for both: the line's own rule of r5-e S2 took the label in.
    assert page_style.count(".jobs button.job:disabled .job-line") == 1
    common = _read("common.js")
    assert "--text-2: #105257;" in common
    assert (
        "button:disabled, button:disabled:hover { cursor: default; "
        "background: transparent; color: rgba(0,0,0,.26);" in common
    )
    # The label's colour against the grey it took, and the hint's against
    # the .6 it had, each on white.
    assert _contrast((16, 82, 87)) >= 4.5
    assert _contrast((0, 0, 0), 0.26) < 3
    assert "input::placeholder { color: rgba(10,50,53,.7); }" in common
    assert "rgba(10,50,53,.6)" not in _code(common)
    assert _contrast((10, 50, 53), 0.7) >= 4.5 > _contrast((10, 50, 53), 0.6)


# --------------------------------------------------------------------------- F08


def test_f08_the_job_slot_says_copy_and_its_box_selects_whole() -> None:
    """F08: over a message the host did not take, the slot's line says to
    copy it into the chat, marked r5f; the message is still drawn in a box
    that selects whole on one click; no view says "Type it in" any more."""
    common = _read("common.js")
    assert f"// WORDING: r5f F08\nOE.JOB_NOT_SENT = '{JOB_NOT_SENT}';" in common
    # Round r5-o: a job button's own sent line (T2), a Copy message button (T3).
    assert (
        "line.textContent = sent ? (job !== null ? OE.JOB_SENT_ONE : OE.JOB_SENT)"
        " : OE.JOB_NOT_SENT;"
    ) in common
    assert "const code = OE.el('code', {}, text);" in common
    assert "slot.append(code, OE.copyButton(text, code, 'Copy message'));" in common
    box = _between(common, ".oe-type code {", "}")
    assert "user-select: all;" in box
    for name in views.VIEWS:
        page = _code(views.view_html(name))
        assert "Type it in" not in page, name
    assert views.view_html("start").count(JOB_NOT_SENT) == 1


# --------------------------------------------------------------------------- F13


def test_f13_the_outputs_line_says_studio_names_the_maps_by_their_ids() -> None:
    """F13: the first line of a run's list of several outputs says that
    Studio names them only by the ids under it and whom to ask, marked r5f,
    said once over the rows; "the one you mean" is gone."""
    page = _read("studio.html")
    outputs = _between(page, "const OUTPUTS = {", "\n};")
    assert f"  // WORDING: r5f F13\n  first: '{OUTPUTS_FIRST}',\n" in outputs
    assert "[OE.el('li', { class: 'small faint' }, OUTPUTS.first), ...rows]" in page
    assert "the one you mean" not in _code(page)
    assert views.view_html("studio").count(OUTPUTS_FIRST) == 1
    assert not _OFF_FACE.search(OUTPUTS_FIRST)


# --------------------------------------------------------------------------- F15


def test_f15_the_area_maps_instruction_is_first_at_body_size() -> None:
    """F15: the instruction line is drawn at body size (no "small") and is
    the controls' first child, before the fields, the saved areas and the
    button's row; its words are as before."""
    page = _read("map.html")
    assert "const where = OE.el('div', { class: 'muted' });" in page
    assert "small" not in _between(page, "const where = ", ";")
    show = _between(page, "function show(connected, params) {", "\n}\n")
    controls = _between(show, "controls.replaceChildren(", ");")
    assert controls.startswith("controls.replaceChildren(\n    where,\n")
    assert controls.count("where") == 1
    assert _in_order(
        controls,
        "where,",
        "field('Project', projectSelect)",
        "areaBlock,",
        "saveButton",
    )
    update = _between(page, "function update() {", "\n}\n")
    assert f"'{MAP_WHERE}'" in update
    # The sign-in, when it opens, still goes above the instruction.
    assert "controls.prepend(panel);" in page


# --------------------------------------------------------------------------- F16


def test_f16_every_view_has_the_canvas_and_a_sans_serif_before_its_scripts() -> None:
    """F16: each of the seven pages holds the two rules in its head, before
    the watchdog and the module (none of whose rules run when the module is
    blocked): first in the page's style where it had one, a style of their
    own in the label sheet and the sign-in view. Their colours and the body's
    margin, padding, size and line height are OE.STYLE's, which takes over
    once the module runs; the watchdog's line is unchanged."""
    look = _between(_read("common.js"), "OE.STYLE = `", "\n`;\n")
    assert "--canvas: #faf2e9;" in look and "--text: #0a3235;" in look
    assert "html { background: var(--canvas); }" in look
    body = _between(look, "\nbody {", "}")
    for part in (
        "margin: 0; padding: 0 16px 16px; background: var(--canvas); "
        "color: var(--text);",
        "font-size: 14px; line-height: 21px;",
    ):
        assert part in body, part
    want = STATIC_HTML + " " + STATIC_BODY
    assert len(views.VIEWS) == 7
    for name, spec in views.VIEWS.items():
        page = _read(str(spec["file"]))
        head = page[: page.index("</head>")]
        assert head.count("<style>") == 1, name
        first = _flat(_first_style(page))
        if name in STYLED:
            assert first.startswith(want + " "), name
        else:
            assert first == want, name
        at = page.index(STATIC_HTML)
        assert at < page.index("</head>") < page.index(WATCHDOG_START), name
        assert page.index(WATCHDOG_START) < page.index(MODULE_START), name
        assert page.count(STATIC_HTML) == 1, name
        # Served as written (views.py fills in only its placeholders).
        assert want in _flat(views.view_html(name)), name
        assert (
            "if(r)r.textContent='This card could not start here. Carry on in the "
            "chat.';" in page
        ), name


# --------------------------------------------------------------------------- F17


def test_f17_the_harness_stand_ins_are_the_apps_own_words() -> None:
    """F17 (harness only; scripts/mcp_views_e2e.py reads no file under tests):
    its pictures question is the first sentence of imagery.CONSENT_ASK, which
    says "square"; its test question is how both of the app's test questions
    open, with no "review set"; a certificate plan's row says what
    sheet.IN_PROGRESS_JOBS says; a part smaller than one square is refused in
    scores_file's words; a sheet of a map in parts carries the server's
    heading, which the sheet draws under "Label sheet"."""
    assert imagery.CONSENT_ASK.startswith(ASK_FIRST + " ")
    assert "window" not in ASK_FIRST
    assert imagery.TEST_WHY == TEST_WHY
    for ask in (imagery.TEST_ASK, imagery.REVIEW_TEST_ASK):
        assert ask.startswith(TEST_WHY + " ")
    assert "review set" not in TEST_WHY
    assert sheet.IN_PROGRESS_JOBS["certify"] == CERTIFY_JOB
    assert PATCH_REFUSAL in _source("tools", "scores_file.py")
    assert PART_HEADING in _source("mcp_app", "server.py")
    assert (
        "data.part.heading ? OE.el('div', { class: 'small' }, data.part.heading)"
        in _read("sheet.js")
    )
