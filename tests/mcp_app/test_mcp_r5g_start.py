# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Round r5-g (18 Nov 2026), track START, as specs/r5-g-round.md gives it:
item G2, the start card points a person with no Studio account to OlmoEarth's
"Request access" (owner, 18 Nov 2026). The views are read as text, as they
are on disk (no test here runs the scripts; the browser harness,
scripts/mcp_views_e2e.py, presses the button):

- G2: signed out, the start card's sign-in note is its sentence about the
  key, then a line of its own: "No Studio account? ", a "Request access"
  text button, " on olmoearth.allenai.org. Do not use someone else's key.";
  the button opens https://olmoearth.allenai.org/ through OE.open (the host
  opens it, never the page) and, when the host does not, shows the address
  to copy, as the docs link does (OE.accessLink, in OE.docsLink's shape); a
  later press that opens it also clears OE.open's line that the link was not
  opened (every link of a view, the docs link's too); signed in, no such
  line (only the sign-in panel holds it); a text button, underlined as the
  docs link is, never a second contained button.
"""

from __future__ import annotations

import re

from olmoearth_agent.mcp_app import views

#: G2: the start card's sentence about the key, which now ends where the key
#: goes (r5-d R12's words); what to do without an account left it.
KEY_NOTE = (
    "You need your own Studio API key: a secret code that Studio makes for you "
    "(in Studio, open your profile, then API Keys). The key goes to the OlmoEarth "
    "Agent here, not through Claude."
)

#: G2: the note's last line as a reader sees it, and its three parts: the
#: words before the text button, its label (the site's own words for its
#: button), the words after it.
ACCESS_LINE = (
    "No Studio account? Request access on olmoearth.allenai.org. "
    "Do not use someone else's key."
)
ACCESS_BEFORE = "No Studio account? "
ACCESS_LABEL = "Request access"
ACCESS_AFTER = " on olmoearth.allenai.org. Do not use someone else's key."

#: G2: the address the button opens (the OlmoEarth site, whose "Request
#: access" is at the top of this page, read 18 Nov 2026).
ACCESS_URL = "https://olmoearth.allenai.org/"

#: What the status line says when the host does not open a link (OE.open,
#: r4-k K7's words); a later link opened clears it.
LINK_NOT_OPENED = "The link was not opened."

#: The note's old last sentences, which no view keeps.
OLD = (
    "Ask whoever made the map how to get one of your own.",
    "Do not use their key.",
)

#: Words no face holds (r4-i's rule), and no choice is marked recommended.
_OFF_FACE = re.compile(
    r"\b(?:windows?|UTM|zones?|intervals?|strat(?:um|a)|design|budget|recommended)\b",
    re.I,
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


def _in_order(text: str, *pieces: str) -> bool:
    """Whether each piece is in text, each after the one before it."""
    at = [text.index(p) for p in pieces]
    return at == sorted(at)


def _js(text: str) -> str:
    """text as a single-quoted JS string, its quotes escaped as the views do."""
    return "'" + text.replace("'", "\\'") + "'"


def _first_style(page: str) -> str:
    """The text inside a page's first style element."""
    start = page.index("<style>") + len("<style>")
    return page[start : page.index("</style>", start)]


def _flat(css: str) -> str:
    """CSS with its comments taken out and its spaces joined."""
    return " ".join(re.sub(r"/\*.*?\*/", " ", css, flags=re.S).split())


# --------------------------------------------------------------------------- G2


def test_g2_the_signed_out_note_ends_with_a_request_access_line() -> None:
    """G2: the note's sentence ends where the key goes; its last line is the
    words before the button, OE.accessLink's button and its slot, the words
    after it, each string marked r5g; the line is drawn into the sign-in's
    note, inside the signed-out branch only (a signed-in user has an
    account); no view keeps the old last sentences."""
    page = _read("start.html")
    tags = "// WORDING: r5g\n// WORDING: r5d R12\n"
    assert f"{tags}const KEY_NOTE = {_js(KEY_NOTE)};\n" in page
    assert f"// WORDING: r5g\nconst ACCESS_BEFORE = {_js(ACCESS_BEFORE)};\n" in page
    assert f"// WORDING: r5g\nconst ACCESS_AFTER = {_js(ACCESS_AFTER)};\n" in page
    assert ACCESS_BEFORE + ACCESS_LABEL + ACCESS_AFTER == ACCESS_LINE
    line = _between(page, "function accessLine(app) {", "\n}\n")
    assert _in_order(
        line,
        "const [access, address] = OE.accessLink(app);",
        "return OE.el('div', { class: 'oe-access' }, ACCESS_BEFORE, access, "
        "ACCESS_AFTER, address);",
    )
    render = _between(page, "function render(app, data) {", "\n}\n")
    signed_out = _between(render, "  if (!on) {\n", "\n  }\n")
    assert _in_order(
        signed_out,
        "root.append(OE.signInPanel(app, {",
        "acceptCurrent: true,",
        "note: [KEY_NOTE, accessLine(app)],",
        "onDone: (s) => {",
    )
    # Its definition and its one call; the helper is called there alone.
    code = _code(page)
    assert code.count("accessLine(") == 2
    assert code.count("OE.accessLink(") == 1
    assert "accessLine(" not in _code(render.replace(signed_out, ""))
    for name in views.VIEWS:
        html = views.view_html(name)
        for gone in OLD:
            assert gone not in html, (name, gone)
        if name != "start":
            assert "OE.accessLink(" not in _code(html), name
    # None of the words a face never says; the new line's sentences short.
    for words in (KEY_NOTE, ACCESS_LINE):
        assert not _OFF_FACE.search(words), words
    assert max(len(s.split()) for s in ACCESS_LINE.split(". ")) <= 8


def test_g2_the_access_link_opens_the_site_through_the_host() -> None:
    """G2: OE.accessLink is a text button that asks the host to open
    OlmoEarth's site (OE.open; the page never opens it) and its slot, which
    shows the address to copy, under the docs link's words, when the host
    does not open it, and is emptied by a later press that opens it: the
    shape of OE.docsLink, body for body, but for its label and address.
    OE.open's line that the link was not opened is cleared by a later link
    opened, and only that line (as OE.say clears its own, r5-b J20)."""
    common = _read("common.js")
    assert f"OE.ACCESS_URL = '{ACCESS_URL}';\n" in common
    assert f"// WORDING: r5g\nOE.ACCESS_LABEL = '{ACCESS_LABEL}';\n" in common
    head = "OE.accessLink = (app) => {"
    link = _between(common, head, "\n};")
    assert _in_order(
        link,
        "const slot = OE.el('div', { class: 'oe-link-slot small' });",
        "const button = OE.el('button', { type: 'button', class: 'link' }, "
        "OE.ACCESS_LABEL);",
        "if (busy) return;",
        "button.setAttribute('aria-disabled', 'true');",
        "const opened = await OE.open(app, OE.ACCESS_URL);",
        "slot.replaceChildren(...(opened ? [] : [OE.el('div', {}, OE.DOCS_BY_HAND), "
        "OE.el('code', {}, OE.ACCESS_URL)]));",
        "button.setAttribute('aria-disabled', 'false');",
        "return [button, slot];",
    )
    for banned in (
        "window.open",
        "location",
        "OE.el('a'",
        "primary",
        "focus",
        "DOCS_URL",
        "DOCS_LABEL",
    ):
        assert banned not in link, banned
    docs_head = "OE.docsLink = (app) => {"
    docs = _between(common, docs_head, "\n};")[len(docs_head) :]
    same = docs.replace("OE.DOCS_LABEL", "OE.ACCESS_LABEL")
    assert same.replace("OE.DOCS_URL", "OE.ACCESS_URL") == link[len(head) :]
    # OE.open asks the host and answers false when the host declined.
    opener = _between(common, "OE.open = async (app, url) => {", "\n};")
    assert "await app.openLink({ url });" in opener
    assert opener.count("return false;") == 2 and "return true;" in opener
    # Both failures write LINK_NOT_OPENED, said once, under r4-k K7's mark; a
    # link opened clears the status line only while it still says that, so
    # a later press that opens the site never leaves it there; any other
    # line stays.
    named = f"// WORDING: r4k K7\nOE.LINK_NOT_OPENED = '{LINK_NOT_OPENED}';\n"
    assert named + "OE.open = async (app, url) => {" in common
    assert common.count(f"'{LINK_NOT_OPENED}'") == 1
    assert LINK_NOT_OPENED not in _code(opener)
    assert _in_order(
        opener,
        "if (answer && answer.isError) { OE.toast(OE.LINK_NOT_OPENED, true); "
        "return false; }",
        "const bar = document.getElementById('oe-toast');",
        "if (bar && bar.textContent === OE.LINK_NOT_OPENED) OE.toast('');",
        "return true;",
        "} catch (_) {\n    OE.toast(OE.LINK_NOT_OPENED, true);\n    return false;",
    )
    assert opener.count("OE.toast(") == 3 and opener.count("OE.toast('')") == 1
    # The address is public and fixed: no id, query or place in it.
    assert re.fullmatch(r"https://olmoearth\.allenai\.org/", ACCESS_URL)
    # The slot is drawn only when it holds the address, which selects whole.
    assert ".oe-link-slot:empty { display: none; }" in common
    assert ".oe-link-slot code { user-select: all; overflow-wrap: anywhere; }" in common


def test_g2_the_sign_in_panel_places_an_element_line_as_given() -> None:
    """G2: the sign-in panel's note takes a line that is an element (the
    start card's access line, with its button and slot inside it) as it is,
    a string as a div of its own as before, after the same 4 px gap; the
    panel's one contained button is still the sign-in's; no other card
    draws an element line."""
    common = _read("common.js")
    panel = _between(common, "OE.signInPanel = ", "\n};")
    assert _in_order(
        panel,
        "const button = OE.el('button', { class: primary ? 'primary' : '' }, "
        "OE.SIGN_IN_BUTTON);",
        "const given = (Array.isArray(note) ? note : [note]).filter((s) => "
        "(typeof s === 'string' && s) || s instanceof Node);",
        "const lines = given.length ? given : OE.SIGN_IN_NOTE;",
        "OE.el('div', { class: 'oe-sign-note' }, lines.map((s) => "
        "(s instanceof Node ? s : OE.el('div', {}, s)))),",
        "status, link,",
    )
    assert ".oe-sign-note > div + div { margin-top: 4px; }" in common
    for name in views.VIEWS:
        if name != "start":
            assert "accessLine(" not in _code(views.view_html(name)), name


def test_g2_request_access_is_an_underlined_text_button_in_its_line() -> None:
    """G2: "Request access" is a text button (class link, never a second
    contained button) at its line's size and line height, the body's, with
    no padding, so the line keeps its height and its spacing; underlined as
    the docs link is (r5-c S13), whose own rule is unchanged."""
    style = _flat(_first_style(_read("start.html")))
    rule = _between(style, ".oe-access button.link {", "}")
    for part in (
        "font-size: 14px;",
        "line-height: 21px;",
        "padding: 0;",
        "text-decoration: underline;",
        "text-underline-offset: 2px;",
    ):
        assert part in rule, part
    docs = _between(style, ".oe-overview-cap button.link {", "}")
    for part in ("font-size: 12px;", "padding: 0 4px;", "text-decoration: underline;"):
        assert part in docs, part
    common = _read("common.js")
    look = _between(common, "OE.STYLE = `", "\n`;\n")
    assert "font-size: 14px; line-height: 21px;" in _between(look, "\nbody {", "}")
    link = _between(common, "OE.accessLink = (app) => {", "\n};")
    assert "class: 'link' }" in link and "primary" not in link


def test_g2_every_changed_string_is_marked_and_said_once() -> None:
    """G2: each string this track added or changed sits under a WORDING: r5g
    mark (three in start.html, the label's in common.js); the label and the
    address are written once, in common.js, and the line's closing words
    once, in start.html."""
    start = _read("start.html")
    assert start.count("// WORDING: r5g\n") == 3
    for name in ("KEY_NOTE", "ACCESS_BEFORE", "ACCESS_AFTER"):
        mark = rf"// WORDING: r5g\n(?:// WORDING: [^\n]*\n)*const {name} = "
        assert re.search(mark, start), name
    assert "// WORDING: r4h P1" not in start
    page = views.view_html("start")
    assert page.count(f"'{ACCESS_LABEL}'") == 1
    assert page.count(ACCESS_URL) == 1
    assert page.count("on olmoearth.allenai.org. Do not use") == 1
