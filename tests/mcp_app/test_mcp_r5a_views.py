# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Round r5-a (17 Nov 2026), track VIEWS, each string as specs/r5-a-round.md
gives it; the views are read as text (no test here runs the scripts), the
server's note through its own names:

- V1: the start card's overview figure (views/overview.js): one name; on a
  wide panel the OlmoEarth docs' own overview figure, inlined (owner, 17 Nov
  2026), on a narrow one a strip of three boxes drawn in code with the views'
  tokens; one text name in plain words, no digit and no per cent sign; its
  caption and the docs link, outside the figure;
- V2: the account in the slim top line (OE.line, OE.setAccount), on the
  start card and the projects card; no "Signed in to OlmoEarth Studio" line
  on either;
- V3: the projects card leads to the next press: its next line, "Show its
  maps" on every project, "Tell Claude: this project" as a text button that
  also opens the project, the predictions before the areas, one line under
  each completed prediction, and the model's note (PROJECTS_PRESS_NOTE).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytest.importorskip("mcp")

from olmoearth_agent.mcp_app import server, views  # noqa: E402
from olmoearth_agent.skills import build_default_registry  # noqa: E402

#: V1: the figure's text name (its title, then what it shows), in plain words
#: since the start card shows the docs figure itself (owner, 17 Nov 2026).
TITLE = "How OlmoEarth makes a map, and where this app comes in."
DESC = (
    "Your labelled examples and satellite pictures go into OlmoEarth Studio. There "
    "you teach a model and run it, again and again. Studio makes maps, such as "
    "crops, mangrove loss and fire risk. This app then checks a map: you look at "
    "squares picked at random and get how often the map is wrong, with a range."
)

#: V1: the one small line under the figure, and the docs link after it.
CAPTION = (
    "How OlmoEarth makes maps (a figure from the OlmoEarth docs). This app comes "
    "after: it checks how often a map is wrong."
)
#: r5-d R04: the caption on a narrow panel, where the strip shows.
CAPTION_NARROW = "How a check of a Studio map works."
DOCS_LABEL = "What is OlmoEarth? (docs)"
DOCS_URL = "https://docs.olmoearth.allenai.org/"

#: V1: every word the narrow strip draws, in the order overview.js lists them
#: (a label over several lines is several entries).
FIGURE_WORDS = [
    "Studio",
    "makes a map",
    "You check",
    "random squares",
    "You get",
    "how often it is",
    "wrong, with a",
    "range",
]

#: V3: the projects card's lead line (r5-c S04: the button names in quotes,
#: one sentence; r5-e S5: the card's only line, no "Next: ").
PROJECTS_NEXT = (
    'Press "Show its maps" on the project that holds your map, then '
    '"Check how wrong it is" beside the map.'
)

#: V3: the line under a completed prediction's row, by its model's kind
#: (values: a values model whose grain is pixel; valuesAny: any other values
#: model, the spec's sentence without "at each pixel"; unknown: the kind is
#: not recorded; unread: the model it names was not read); r5-c S08: the
#: values lines say which card tells, and a failed row says why it has none;
#: r5-e S7: a class map's row has no line (its Check button says it).
ROW_SAYS = {
    "values": (
        "Shows a number at each pixel. Check works if that number is a score from 0 "
        "to 1; its Check card tells you."
    ),
    "valuesAny": (
        "Shows a number. Check works if that number is a score from 0 to 1; its "
        "Check card tells you."
    ),
    "detection": "Shows detections, not a map of classes. Check cannot run on it.",
    "unknown": (
        "What it shows is not recorded. " "Its Check card says whether it can run."
    ),
    "unread": (
        "What it shows was not read here. Its Check card says whether it can run."
    ),
    "failed": "This run failed in Studio. There is no map to check.",
}

#: Words no face holds (r4-i's rule).
_OFF_FACE = re.compile(r"\b(?:windows?|budget|stratum|design|review set)\b", re.I)

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


def _strings(block: str) -> list[str]:
    """Every single-quoted JS string of block, in order, unescaped."""
    found = re.findall(r"'((?:[^'\\]|\\.)*)'", block)
    return [re.sub(r"\\(.)", r"\1", s) for s in found]


def _in_order(text: str, *pieces: str) -> bool:
    """Whether each piece is in text, each after the one before it."""
    at = [text.index(p) for p in pieces]
    return at == sorted(at)


# --------------------------------------------------------------------------- V1


def test_v1_only_the_start_view_inlines_the_overview() -> None:
    """V1: overview.js is inlined into the start view, after common.js, and
    into no other view; the card draws the figure under its line (r5-e S1:
    one line, where it had three), before the sign-in and the jobs, with no
    fold."""
    assert views.VIEWS["start"]["scripts"] == ("common.js", "overview.js")
    for name, spec in views.VIEWS.items():
        held = "OE.overview = (() => {" in views.view_html(name)
        assert held is (name == "start"), name
        assert ("overview.js" in spec["scripts"]) is (name == "start"), name
    page = _read("start.html")
    render = _between(page, "function render(app, data) {", "\n}\n")
    assert _in_order(
        render,
        "root.replaceChildren(OE.el('h1', {}, 'What this app can do'));",
        # r5-c S03: signed in, its next line no longer asks to sign in.
        "const lead = OE.lead(on ? { ...START_LEAD, next: START_NEXT_IN } "
        ": START_LEAD);",
        "root.append(overviewBlock(app));",
        "root.append(OE.signInPanel(app, {",
        "OE.el('ul', { class: 'items jobs', 'aria-label': 'Jobs' }",
    )
    assert "OE.fold(" not in page and "<details" not in page


def test_v1_the_overview_lives_inside_one_name_and_never_updates() -> None:
    """V1: one IIFE, OE.overview, with no name of its own at the top of the
    module it is inlined into; no call, timer, listener or load; it returns
    its style, the figure and the caption (r5-d R04: its two forms); its
    placeholders are the docs figure's and (r5-e S8) the narrow strip's
    example map's."""
    script = _read("overview.js")
    assert script.count("OE.overview = (() => {") == 1
    assert script.rstrip().endswith("})();")
    top = re.compile(r"^(?:const|let|var|function|async function|class)\b", re.M)
    assert top.search(script) is None
    assert "  return { STYLE, style, figure, CAPTION, CAPTION_NARROW };\n" in script
    for banned in (
        "OE.call(",
        "callServerTool",
        "setTimeout",
        "setInterval",
        "addEventListener",
        "fetch(",
        "tellClaude",
        "OE.open(",
    ):
        assert banned not in script, banned
    assert re.findall(r"@[A-Z_]+@", _code(script)) == [
        "@OVERVIEW_DATA_URL@",
        "@MAP_EXAMPLE_DATA_URL@",
    ]
    for placeholder in ("@EXT_APPS", "@LEAFLET", "@DRAW", "@SCRIPTS@"):
        assert placeholder not in script, placeholder
    raw = {chr(0x2028), chr(0x2029), "\x00", "\x0b", "\x0c", "\x1b", "\x7f"}
    assert not raw & set(script)


def test_v1_the_wide_figure_is_the_docs_figure_inlined() -> None:
    """V1 (owner, 17 Nov 2026: the docs' assets may be used): on a wide panel
    the start card shows the overview figure of the OlmoEarth docs, a JPEG
    in views/brand inlined as a data URL into the start view alone, so no
    page loads it from elsewhere; it is aria-hidden (the wrapper names it)."""
    path = views.VIEW_DIR / views.BRAND["OVERVIEW_DATA_URL"]
    data = path.read_bytes()
    assert path.name == "olmoearth-overview.jpg"
    assert data[:3] == b"\xff\xd8\xff" and 50_000 < len(data) < 300_000
    url = views.brand_data_url("OVERVIEW_DATA_URL")
    assert url.startswith("data:image/jpeg;base64,")
    assert views.brand_data_url("AGENT_TAG_DATA_URL").startswith("data:image/png;")
    for name in views.VIEWS:
        assert (url in views.view_html(name)) is (name == "start"), name
    script = _read("overview.js")
    assert "  const PICTURE = '@OVERVIEW_DATA_URL@';\n" in script
    wide = _between(script, "  const wide = () => OE.el('img', {", "\n  });")
    for attr in ("class: 'wide'", "src: PICTURE", "alt: ''", "'aria-hidden': 'true'"):
        assert attr in wide, attr
    assert "width: '1500', height: '843'" in wide


def test_v1_the_narrow_strip_is_drawn_in_code_with_the_views_tokens() -> None:
    """V1: the narrow strip is drawn in code: no marker and no id; no colour
    is written out (r5-e S8: the made-up tile and its colours, TILE, gave way
    to a docs picture, a placeholder views.py fills in), every fill and
    stroke is a token; no mint: the dot is the canvas colour. The only
    address in the script is the SVG namespace."""
    script = _read("overview.js")
    code = _code(script)
    for banned in (
        "data:",
        "url(",
        "marker",
        "<title",
        "<defs",
        "https:",
        ".png",
        "rgb(",
        "rgba(",
        "--mint",
        "xlink",
    ):
        assert banned not in code, banned
    assert script.count("http") == 1
    assert "const NS = 'http://www.w3.org/2000/svg';" in script
    assert re.search(r"\bid:\s", code) is None
    assert _HEX.findall(script) == []
    for gone in ("const TILE", "const PATCHES", "const tile ="):
        assert gone not in script, gone
    style = _between(script, "  const STYLE = `", "`;")
    for rule in (
        ".oe-overview text { font-family: var(--font); fill: var(--text); }",
        # r5-e S3: the canvas colour, inside the start card's white box.
        ".oe-overview .box { fill: var(--canvas); stroke: var(--divider); ",
        ".oe-overview .ln { fill: none; stroke: var(--text-2); ",
        ".oe-overview .app { fill: var(--text); }",
        ".oe-overview .dark text { fill: var(--canvas); }",
        ".oe-overview .dot { fill: var(--canvas); stroke: var(--text); ",
    ):
        assert rule in style, rule


def test_v1_one_picture_one_drawing_and_one_text_name() -> None:
    """V1: the docs picture and the narrow drawing, each aria-hidden, inside
    one role=img element named by the title and the description; the
    drawing shows below 680 px, the picture from 680 px up, on a white card
    (r5-e S3: the start card's box, .oe-overview-wrap, which holds the
    caption too; the picture keeps no box of its own)."""
    script = _read("overview.js")
    assert script.count("viewBox: '0 0 348 96'") == 1
    assert "viewBox: '0 0 748 238'" not in script
    assert script.count("'aria-hidden': 'true'") == 2
    figure = _between(script, "  const figure = () => {", "\n  };")
    named = "role: 'img', 'aria-label': TITLE + ' ' + DESC }, wide(), narrow());"
    assert "style();" in figure and named in figure
    assert f"  // WORDING: r5c docs figure\n  const TITLE = '{TITLE}';\n" in script
    assert f"  // WORDING: r5c docs figure\n  const DESC = '{DESC}';\n" in script
    style = _between(script, "  const STYLE = `", "`;")
    assert "img.wide { display: none; width: 100%; max-width: 748px;" in style
    assert "border: 1px solid" not in style
    page_style = _between(_read("start.html"), "<style>", "</style>")
    box = _between(page_style, ".oe-overview-wrap {", "}")
    assert "background: var(--paper); border: 1px solid var(--divider);" in box
    wide = _between(style, "@media (min-width: 680px) {", "\n}\n")
    assert ".oe-overview img.wide { display: block; }" in wide
    assert ".oe-overview svg.narrow { display: none; }" in wide


def test_v1_the_figures_words() -> None:
    """V1: the narrow strip's words are plain: Studio makes a map, you check
    random squares, you get how often it is wrong, with a range; no digit
    and no per cent sign in them, the title, the description or the
    caption; none of the words a face never says; each label is drawn."""
    script = _read("overview.js")
    assert "  // WORDING: r5a V1\n  const WORDS = {" in script
    table = _between(script, "  const WORDS = {", "\n  };")
    words = _strings(table)
    assert words == FIGURE_WORDS
    said = " ".join(words)
    for part in ("Studio", "squares", "range"):
        assert part in said, part
    for text in (*words, TITLE, DESC, CAPTION, CAPTION_NARROW):
        assert re.search(r"[0-9%]", text) is None, text
        assert _OFF_FACE.search(text) is None, text
        assert "What this app can do" not in text, text
    for jargon in ("foundation model", "inference", "fine-tun", "embedding"):
        assert jargon not in (DESC + CAPTION + CAPTION_NARROW).lower(), jargon
    drawn = _between(script, "  const narrow = () => ", "  const figure = () => {")
    keys = re.findall(r"^    ([A-Za-z]+): ", table, re.M)
    assert keys == ["narrowStudio", "narrowCheck", "narrowGet"]
    for key in keys:
        assert f"WORDS.{key}" in drawn, key


def test_v1_the_strips_boxes_are_kept() -> None:
    """V1: the narrow strip's boxes, whose geometry was checked in a browser,
    are drawn where the draft has them (x, y, width, height, corner); an
    arrow's head is part of the arrow's own path."""
    script = _read("overview.js")
    narrow = _between(script, "  const narrow = () => ", "  const figure = () => {")
    for box in (
        "inbox('box', 1, 1, 100, 94, 8,",
        "inbox('box', 123, 1, 100, 94, 8,",
        "inbox('app', 245, 1, 102, 94, 8,",
    ):
        assert narrow.count(box) == 1, box
    assert narrow.count("arrow('M") == 2
    arrow = _between(script, "  const arrow = (d, x, y, dir) => {", "\n  };")
    assert "return svg('path', {" in arrow and "class: 'ln', d: `${d}M" in arrow


def test_v1_the_caption_and_the_docs_link_sit_outside_the_figure() -> None:
    """V1: one small caption under the figure, then a text button that opens
    the OlmoEarth docs through OE.open (the host opens it, never the page);
    when the host does not, the address shows to copy (r5-i B5: under its
    own line that the link was not opened, with no toast: quiet)."""
    script = _read("overview.js")
    assert f"  // WORDING: r5c docs figure\n  const CAPTION = '{CAPTION}';\n" in script
    common = _read("common.js")
    assert f"OE.DOCS_URL = '{DOCS_URL}';\n" in common
    assert f"// WORDING: r5a V1\nOE.DOCS_LABEL = '{DOCS_LABEL}';\n" in common
    link = _between(common, "OE.docsLink = (app) => {", "\n};")
    for text in (
        "OE.el('button', { type: 'button', class: 'link' }, OE.DOCS_LABEL);",
        "button.setAttribute('aria-disabled', 'true');",
        "const opened = await OE.open(app, OE.DOCS_URL, { quiet: true });",
        "OE.el('code', {}, OE.DOCS_URL)",
        "return [button, slot];",
    ):
        assert text in link, text
    for banned in ("window.open", "location", "OE.el('a'", "primary", "focus"):
        assert banned not in link, banned
    page = _read("start.html")
    block = _between(page, "function overviewBlock(app) {", "\n}\n")
    assert _in_order(
        block,
        "const [docs, address] = OE.docsLink(app);",
        "OE.el('div', { class: 'oe-overview-wrap' },",
        "OE.overview.figure(),",
        "{ class: 'small muted oe-overview-cap' }",
        # r5-d R04: the caption's two forms, one span each, then the link.
        "OE.el('span', { class: 'cap-wide' }, OE.overview.CAPTION),",
        "OE.el('span', { class: 'cap-narrow' }, OE.overview.CAPTION_NARROW),",
        "' ', docs, address));",
    )


# --------------------------------------------------------------------------- V2


def test_v2_the_top_line_holds_the_account_item() -> None:
    """V2: OE.line is the tag, the sentence and, right of it, an empty
    account item, which is not drawn while it is empty; no second image and
    none of the old brand bar's class names."""
    common = _read("common.js")
    line = _between(common, "OE.line = () => ", ";\n")
    for text in (
        "OE.el('header', { class: 'oe-line' },",
        "OE.el('div', { class: 'oe-line-text' },",
        "OE.el('span', { class: 'oe-said' }, OE.SUB),",
        "OE.el('span', { id: 'oe-account', class: 'oe-account' })",
    ):
        assert text in line, text
    assert line.count("OE.el('img'") == 1
    assert ".oe-account:empty { display: none; }" in common
    assert ".oe-said { flex: 1 1 260px; min-width: 0; }" in common
    for gone in ("oe-sub", "oe-brand", "OE.accountBar"):
        assert gone not in common, gone


def test_v2_set_account_fills_and_empties_the_item() -> None:
    """V2: signed in, who (short) and a Sign out text button, the whole
    sentence as the item's name; signed out, empty and unnamed; it never
    moves the focus."""
    common = _read("common.js")
    body = _between(common, "OE.setAccount = ", "\n};")
    assert _in_order(
        body,
        "const item = document.getElementById('oe-account');",
        "if (!item) return null;",
        "if (!signedIn) {",
        "item.replaceChildren();",
        "item.removeAttribute('role');",
        "item.removeAttribute('aria-label');",
        "const said = OE.accountLine(account);",
        # r5-c S01: no Sign out for the key the app was installed with.
        "const out = source === 'environment' ? null : "
        "OE.signOutButton(app, { onSignOut });",
        "item.setAttribute('role', 'group');",
        "item.setAttribute('aria-label', said);",
        "OE.el('span', { class: 'who', title: said }, "
        "OE.accountShort(account, source)), out]",
        "return out;",
    )
    assert "focus" not in body
    head = "// WORDING: r5a V2\nOE.accountShort = (account, source = null) => {"
    assert head in common
    short = _between(common, "OE.accountShort = ", "\n};")
    for text in (
        "if (!account) return 'Signed in';",
        "const who = account.user_name || 'Your account';",
        "return who + org;",
    ):
        assert text in short, text
    assert "Signed in to" not in short


def test_v2_the_two_cards_set_the_account_and_drop_their_signed_in_line() -> None:
    """V2: the start card and the projects card call OE.setAccount at each
    drawing; neither says "Signed in to OlmoEarth Studio as ..." in its body
    any more; each card's sign-out gives the focus on (the projects card's:
    to the sign-in's button, or to the new Sign out button and, when the read
    answers, from that one to the one drawn in its place); the other five
    views leave the item empty."""
    start = _read("start.html")
    render = _between(start, "function render(app, data) {", "\n}\n")
    # r5-c S01: each card passes the key's source; after Sign out the start
    # card's focus goes to its one contained button.
    set_call = (
        "OE.setAccount(app, { signedIn: on, account: status.account, "
        "source: status.source, onSignOut: (s) => signedOut(app, data, s) });"
    )
    assert _in_order(render, set_call, "root.replaceChildren(", "if (!on) {")
    assert "if (on) {" not in render
    out = _between(start, "function signedOut(app, data, result) {", "\n}\n")
    assert _in_order(
        out,
        "render(app, { ...data, status: result });",
        "OE.focusIfHere(root.querySelector('button.primary'));",
    )
    studio = _read("studio.html")
    projects = _between(studio, "function renderProjects(", "\n}\n")
    signed = (
        "OE.setAccount(app, { signedIn: true, account, source, "
        "onSignOut: (r) => signedOut(app, r) });"
    )
    assert signed in projects
    assert "    OE.el('h1', {}, 'Projects in this Studio account'),\n" in projects
    sign_in = _between(studio, "function renderSignIn(app) {", "\n}\n")
    assert "OE.setAccount(app, { signedIn: false });" in sign_in
    left = _between(studio, "async function signedOut(app, result) {", "\n}\n")
    assert "OE.setAccount(app, { signedIn: true, account: result.account," in left
    assert _in_order(
        left,
        "renderSignIn(app);",
        "OE.focusIfHere(root.querySelector('button.primary'));",
        "const held = OE.setAccount(app, { signedIn: true, account: result.account,",
        "OE.focusIfHere(held);",
        "const out = await OE.call(app, 'olmoearth_app_projects');",
        "if (mine !== generation) return;",
        # r5-c S01: with no Sign out drawn (the app's own key), to the search.
        "const had = left();",
        "renderProjects(app, out.projects, result.account, result.source);",
        "if (had) OE.focusIfHere(held ? document.querySelector('#oe-account button')",
        "if (err.needsSignIn) {",
        "if (had) OE.focusIfHere(root.querySelector('button.primary'));",
    )
    assert left.count("const had = left();") == 2
    # A card opened on a failed read names no account either.
    shown = _between(studio, "async function show(app, params) {", "\n}\n")
    assert _in_order(
        shown,
        "if (!data.ok) {",
        "OE.setAccount(app, { signedIn: false });",
        "listsNone(app, 'Studio could not be read');",
    )
    for page in (start, studio):
        for gone in ("OE.accountBar(", "OE.accountLine(", "Signed in to OlmoEarth"):
            assert gone not in page, gone
    for name in (
        "sheet.html",
        "sheet.js",
        "map.html",
        "sign_in.html",
        "result.html",
        "check.html",
        "figures.js",
        "glyph.js",
        "overview.js",
    ):
        assert "OE.setAccount(" not in _read(name), name


# --------------------------------------------------------------------------- V3


def test_v3_the_next_line_names_the_show_button_every_project_has() -> None:
    """V3 (A1, A2): the card's next line names the button; every project, as
    a card and as a compact row, has it; its accessible name starts with its
    label; the opener sets its words and aria-expanded as it sets the
    project's name's."""
    page = _read("studio.html")
    assert f"  next: '{PROJECTS_NEXT}',\n" in page
    tag = "// WORDING: r5e\nconst PROJECTS_LEAD = {"  # r5-e S5
    assert tag in page
    assert "// WORDING: r5a V3\nconst SHOW = { label: (open) => " in page
    assert "(open ? 'Hide its maps' : 'Show its maps') };\n" in page
    button = _between(page, "const showButton = (name) => ", ";\n")
    for text in (
        "type: 'button', class: 'small', 'aria-expanded': 'false',",
        "'aria-label': SHOW.label(false) + ': ' + name,",
        "}, SHOW.label(false))",
    ):
        assert text in button, text
    state = _between(page, "const showState = (button, name, open) => {", "\n};")
    for text in (
        "button.textContent = SHOW.label(open);",
        "button.setAttribute('aria-label', SHOW.label(open) + ': ' + name);",
        "button.setAttribute('aria-expanded', String(open));",
    ):
        assert text in state, text
    for start in ("function projectCard(app, p) {", "function projectRow(app, p) {"):
        block = _between(page, start, "\n}\n")
        for text in (
            "const show = showButton(p.name);",
            "showState(show, p.name, open);",
            "show.addEventListener('click', ",
            "OE.el('span', { class: 'row acts' }, tell, show)",
        ):
            assert text in block, (start, text)
    opener = _between(page, "function opener(app, project, title, ", "\n}\n")
    assert "title.setAttribute('aria-expanded', String(value));" in opener
    assert "onState(value);" in opener


def test_v3_tell_claude_this_project_is_a_text_button_that_opens_it() -> None:
    """V3 (A3, C2): on a project the Tell button is drawn as a text button,
    and its press, after its message, opens the project when it is closed;
    an area's stays small and outlined and opens nothing."""
    page = _read("studio.html")
    use = _between(page, "const useButton = ", "\n\n")
    for text in (
        "(app, text, kind, name, cls = 'small', also = null) =>",
        "class: cls, 'aria-label': TELL.label(kind) + ' (' + name + ')',",
        "e.stopPropagation(); OE.say(app, text); if (also) also();",
    ):
        assert text in use, text
    assert "sendJob" not in use and "primary" not in use
    for start in ("function projectCard(app, p) {", "function projectRow(app, p) {"):
        block = _between(page, start, "\n}\n")
        for text in (
            "let isOpen = false;",
            "let toggle = () => {};",
            "'project', p.name,\n",
            "'link', () => { if (!isOpen) toggle(); });",
            "toggle = opener(app, p, title, slot, (open) => {",
            "isOpen = open;",
        ):
            assert text in block, (start, text)
    summary = _between(page, "function projectSummary(", "\n}\n")
    assert "OE.named(a.name), 'area', a.name)))," in summary
    assert "'link'" not in summary


def test_v3_an_open_project_lists_its_predictions_before_its_areas() -> None:
    """V3 (B3): the Predictions box, then the Areas box, then the models'
    closed fold."""
    page = _read("studio.html")
    summary = _between(page, "function projectSummary(", "\n}\n")
    assert _in_order(
        summary,
        "summaryCard('Predictions', out.predictions,",
        "summaryCard('Areas', out.areas,",
        "modelsBox ? OE.fold(MODELS_FOLD, modelsBox) : null);",
    )


def test_v3_each_completed_prediction_says_what_it_shows() -> None:
    """V3 (B2): one line under a completed prediction's row, by its model's
    kind; "at each pixel" only of a values model whose grain is pixel; a
    model the prediction names that was not read says that, not that its
    kind is not recorded; an embeddings row keeps its own line whatever its
    status; a row not completed says nothing, but a failed one says it failed
    (r5-c S08); no line names a square or a window or calls a map
    recommended; no row's Check is the contained button (B1 is not built).
    r5-e S7: a class map's row has no line (it was the same on every class
    row, and the Check beside it says it)."""
    page = _read("studio.html")
    table = _between(page, "const ROW_SAYS = {", "\n};")
    assert (
        "// WORDING: r5a V3 (detection, unknown, unread); r5c S08 (values, "
        "valuesAny, failed); r5e (classes: none)\nconst ROW_SAYS = {" in page
    )
    assert "classes:" not in table
    for kind, line in ROW_SAYS.items():
        assert f"  {kind}: '{line}',\n" in table + "\n", kind
        assert re.search(r"squares?|windows?|recommend", line, re.I) is None, line
    assert table.count("\n  ") == len(ROW_SAYS)
    assert ROW_SAYS["valuesAny"] == ROW_SAYS["values"].replace(" at each pixel", "")
    says = _between(
        page, "const rowSays = (kind, done, grain, unread, status) => ", ";\n"
    )
    assert _in_order(
        says,
        "kind === 'embeddings' ? NOT_A_MAP",
        "!done ? (status === 'failed' ? ROW_SAYS.failed : null)",
        "unread ? ROW_SAYS.unread",
        "kind === 'values' ? (grain === 'pixel' ? ROW_SAYS.values",
        "ROW_SAYS.valuesAny)",
        "kind === 'classes' ? null : kind === 'detection' ? ROW_SAYS.detection",
        "ROW_SAYS.unknown",
    )
    row = _between(page, "function predictionRow(", "\n}\n")
    assert "const unread = !!p.model_id && !model;" in row
    assert (
        "const says = rowSays(kind, done, model && model.grain, unread, p.status);"
        in row
    )
    assert "says ? OE.el('div', { class: 'small faint pred-says' }, says)" in row
    for name in (
        "function predictionRow(",
        "function resultRow(",
        "function busyButton(",
    ):
        assert "primary" not in _between(page, name, "\n}\n"), name


def test_v3_the_model_is_told_what_to_press() -> None:
    """V3 (A4): the note before the projects starts with what to press, in
    the panel's own button names, then keeps its old text whole; only a
    press opens a project; a detections or embeddings map has no Check
    button; the one line names the press for the user's job (check, compare,
    make a new map) and where the last two show; the rows say what each map
    outputs, so Claude does not list the predictions again unless asked, and
    the note says that this comes before the first half of the rule kept
    after it (RULE_9), whose rest holds always. It is tool output: the
    server's instructions do not hold it."""
    note = server.PROJECTS_PRESS_NOTE
    whole = server.PROJECTS_PANEL_NOTE
    assert whole.startswith(note + " The projects panel may tell you silently ")
    assert server.RULE_9 in whole and whole.endswith(" " + server.CONTEXT_RULE)
    for text in (
        "the user picks by pressing, not by typing",
        "except a detections or embeddings one (its row says so)",
        "you cannot open one from the chat",
        "answer in one line",
        "with the press for their job: to check a map, ",
        "; to compare two maps, tick two finished maps of one open project, then ",
        "; to make a new map, open ",
        "show only in a project with two or more finished predictions",
        "only in a project whose predictions name a model",
        "do not list the project's predictions in the chat",
        "unless they ask for a list or say no panel shows",
        "say it yourself only when you do list them",
        "The rest of that rule holds always",
    ):
        assert text in note, text
    for name in (
        server.PANEL_SHOW_MAPS,
        server.PANEL_CHECK,
        server.PANEL_TELL_PROJECT,
        server.PANEL_WHICH,
        server.PANEL_NEW_MAP,
        server.PANEL_RUN,
    ):
        assert f"'{name}'" in note, name
    assert re.search(r"\bwindows?\b|recommend", note, re.I) is None
    # The names are the page's own.
    page = _read("studio.html")
    assert f" : '{server.PANEL_SHOW_MAPS}') }};\n" in page
    assert f"check: '{server.PANEL_CHECK}'" in page
    assert "label: (kind) => 'Tell Claude: this ' + kind," in page
    assert server.PANEL_TELL_PROJECT == "Tell Claude: this " + "project"
    assert f"busyButton({{ disabled: true }}, '{server.PANEL_WHICH}'," in page
    assert f"const MODELS_FOLD = '{server.PANEL_NEW_MAP}';" in page
    assert f"'{server.PANEL_RUN}', () => OE.sendJob(app, OE.JOB_TEXT.run(" in page
    # Where the compare and new-map presses show, as the note says.
    summary = _between(page, "function projectSummary(", "\n}\n")
    assert "const compare = completed >= 2 ? comparer(app, project) : null;" in summary
    models = _between(page, "function modelsCard(", "\n}\n")
    assert "if (!named_.size) return null;" in models
    source = Path(server.__file__).read_text(encoding="utf-8")
    assert "# WORDING: r5a V3\nPROJECTS_PRESS_NOTE = (\n" in source
    assert note not in server.server_instructions(build_default_registry())
