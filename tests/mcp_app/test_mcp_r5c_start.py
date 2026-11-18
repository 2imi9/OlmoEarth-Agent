# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Round r5-c (17 Nov 2026), track START: the start card, the panel's
sign-in card, the browser sign-in page and the top line's account item, each
item as specs/r5-c-audit/build_list.md gives it. The views are read as text
(no test here runs the scripts); the sign-in page through its own templates:

- S01: with the key the app was installed with (source 'environment') the
  top line says "Signed in with the app's own key" and draws no Sign out;
  the sign-out message says the user is still signed in and who can remove
  that key, chosen by the key in effect; the start card and the projects
  card pass the key's source (the projects card from its first drawing) and
  move the focus on to what is drawn; a reopened start card redraws when
  another key or account is in effect; Claude is not told "signed out"
  while a key stays in effect; the panel's account card names the key in
  its "Key" row and draws no Sign out for that key (the server's side of
  the source and of the sign-in tool's note: test_mcp_server.py);
- S03: the start card's next line asks to sign in only while signed out,
  and its third line has a subject;
- S09: the browser sign-in page opens with what to paste, its key field has
  a label of its own, and its card is white;
- S13: the docs link inside the start card's caption is underlined.
"""

from __future__ import annotations

import re
from pathlib import Path

from olmoearth_agent.mcp_app import auth, views

#: S01: what the top line says with the app's own key, in place of a name.
OWN_KEY_SHORT = "Signed in with the app's own key"

#: S01: the sign-out message while the app's own key is still in effect.
STILL_SIGNED_IN = (
    "Still signed in: this app was installed with its own key. Only whoever "
    "installed it can remove it."
)

#: S01: what the projects card tells Claude after Sign out while a key stays
#: in effect: its projects being read, then their read failed (never "signed
#: out", which the toast and the top line contradict).
STILL_READING = (
    "still signed in: a key stays in effect after Sign out; its projects are being "
    "read"
)
STILL_READ_FAILED = (
    "still signed in: a key stays in effect after Sign out; its projects could not "
    "be read"
)

#: S01: the account card's "Key" row, by the key's source. A key entered on
#: the sign-in page reads so whether or not it was also remembered: the
#: status does not say, so the row cannot say "this session only".
KEY_ROWS = {
    "environment": "The app's own key, set when it was installed (OLMOEARTH_API_KEY)",
    "key-file": "The saved key file",
    "signed-in": "Entered on the sign-in page",
}

#: S01: two keys for the key store (never real ones).
INSTALLED = "sk_test_installed_key_1"
TYPED = "sk_test_typed_key_2"

#: S03: the start card's next line signed out, the same line signed in, and
#: its third line.
START_NEXT_OUT = (
    "Sign in below first; then press a job, or type what you want in the chat."
)
START_NEXT_IN = "Press a job below, or type what you want in the chat."
START_CANNOT = (
    "This card knows nothing about your map yet; only your own answers can show "
    "how wrong it is."
)

#: S09: the sign-in page's first paragraph, its words joined.
PAGE_FIRST = (
    "Paste your Studio API key below. To make one, open your profile in Studio, "
    "then API Keys. This page runs on this computer: the key goes from here to "
    "the OlmoEarth Agent, not through Claude, and is checked with Studio before "
    "it is used."
)

#: Words no face holds (r4-i's rule), and no choice is marked recommended.
_OFF_FACE = re.compile(r"\b(?:windows?|recommended)\b", re.I)


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


def _words(html: str) -> str:
    """The words of an HTML fragment: its tags dropped, its spaces joined."""
    return " ".join(re.sub(r"<[^>]+>", " ", html).split())


# --------------------------------------------------------------------------- S01


def test_s01_the_top_line_draws_no_sign_out_for_the_apps_own_key() -> None:
    """S01 (a): OE.setAccount takes the key's source; with the app's own key
    the item says so, keeps the whole sentence as its name and tooltip,
    draws no Sign out and returns null; another key keeps its name and its
    Sign out. It never moves the focus."""
    common = _read("common.js")
    body = _between(common, "OE.setAccount = ", "\n};")
    assert body.startswith(
        "OE.setAccount = (app, { signedIn = false, account = null, source = null, "
        "onSignOut } = {}) => {"
    )
    assert _in_order(
        body,
        "if (!signedIn) {",
        "const said = OE.accountLine(account);",
        "const out = source === 'environment' ? null : "
        "OE.signOutButton(app, { onSignOut });",
        "item.setAttribute('role', 'group');",
        "item.setAttribute('aria-label', said);",
        "OE.el('span', { class: 'who', title: said }, "
        "OE.accountShort(account, source)), out]",
        ".filter(Boolean));",
        "return out;",
    )
    assert "focus" not in _code(body)
    short = _between(common, "OE.accountShort = ", "\n};")
    assert short.startswith("OE.accountShort = (account, source = null) => {")
    own = "if (source === 'environment') return " + _js(OWN_KEY_SHORT) + ";"
    assert _in_order(
        short,
        "  // WORDING: r5c S01\n  " + own,
        "if (!account) return 'Signed in';",
        "return who + org;",
    )
    assert "// WORDING: r5a V2\nOE.accountShort = (account, source = null)" in common
    assert not _OFF_FACE.search(OWN_KEY_SHORT)


def test_s01_the_sign_out_message_says_the_user_is_still_signed_in() -> None:
    """S01 (b): with the app's own key still in effect, the status line says
    the user is still signed in and who can remove that key; no view keeps
    the old sentence, which said "Signed out here." beside a top line that
    said signed in. The message is chosen by the key in effect (the result's
    signed_in and source), not by environment_key_remains, which an
    OLMOEARTH_API_KEY that is no key also sets
    (test_s01_an_own_key_that_is_no_key_leaves_the_user_signed_out)."""
    button = _between(_read("common.js"), "OE.signOutButton = ", "\n};")
    assert "OE.toast(result.environment_key_remains" not in button
    assert _in_order(
        button,
        "// WORDING: r5c S01 (the first; the second: r4p2)",
        "OE.toast(result.signed_in && result.source === 'environment'",
        "? " + _js(STILL_SIGNED_IN),
        ": 'Signed out; the saved key was removed.');",
    )
    for name in views.VIEWS:
        page = views.view_html(name)
        assert "Signed out here. A key put in" not in page, name
        assert "is still in use; whoever installed" not in page, name
    assert max(len(s.split()) for s in STILL_SIGNED_IN.split(". ")) <= 12


def test_s01_the_start_card_passes_the_source_and_moves_the_focus_on() -> None:
    """S01 (c): the start card gives the top line the key's source at each
    drawing; after Sign out the focus goes to the card's one contained button
    (the sign-in's; the main job's while the app's own key stays in effect),
    never to a Sign out the top line no longer draws. Reopened from the
    history, the card redraws when its top line would read otherwise (signed
    in or out, the app's own key or another, another account), not only when
    the signed-in state flips, and moves no focus."""
    start = _read("start.html")
    render = _between(start, "function render(app, data) {", "\n}\n")
    assert (
        "OE.setAccount(app, { signedIn: on, account: status.account, "
        "source: status.source, onSignOut: (s) => signedOut(app, data, s) });"
        in render
    )
    out = _between(start, "function signedOut(app, data, result) {", "\n}\n")
    assert _in_order(
        out,
        "render(app, { ...data, status: result });",
        "OE.focusIfHere(root.querySelector('button.primary'));",
    )
    assert "#oe-account" not in out and "signedIn(result)" not in out
    drawn = _between(start, "const drawnAs = (s) => ", ";\n")
    assert drawn == (
        "const drawnAs = (s) => (signedIn(s)\n"
        "  ? OE.accountShort(s.account, s.source) + '|' + OE.accountShort(s.account)"
        " : null)"
    )
    again = _between(start, "async function recheck(app, data) {", "\n}\n")
    assert _in_order(
        again,
        "await OE.call(app, 'olmoearth_app_status');",
        "if (mine !== generation || !now) return;",
        "if (drawnAs(now) !== drawnAs(data.status)) "
        "render(app, { ...data, status: now });",
    )
    assert "signedIn(now) !== signedIn(data.status)" not in again
    assert "focus" not in _code(again)


def test_s01_the_projects_card_passes_the_source_and_moves_the_focus_on() -> None:
    """S01 (d): renderProjects takes the key's source for the top line, from
    the card's first drawing (the tool's view data carries it); after Sign
    out with a key still in effect the sign-out's source is passed on; with
    no Sign out drawn the focus goes to the search once the projects are
    drawn (to the new Sign out only when one is drawn), or to the error card
    when they cannot be read; a failed read of the projects no longer says
    "Signed out here." on the card, and Claude is not told "signed out"
    while a key stays in effect."""
    studio = _read("studio.html")
    projects = _between(studio, "function renderProjects(", "\n}\n")
    assert projects.startswith(
        "function renderProjects(app, projects, account, source = null) {"
    )
    assert (
        "OE.setAccount(app, { signedIn: true, account, source, "
        "onSignOut: (r) => signedOut(app, r) });" in projects
    )
    signed_out = _between(studio, "async function signedOut(app, result) {", "\n}\n")
    assert _in_order(
        signed_out,
        "const held = OE.setAccount(app, { signedIn: true, account: result.account, "
        "source: result.source,",
        "OE.focusIfHere(held);",
        "const left = () => (held ? document.activeElement === held",
        ": !document.activeElement || document.activeElement === document.body);",
        "const out = await OE.call(app, 'olmoearth_app_projects');",
        "const had = left();",
        "renderProjects(app, out.projects, result.account, result.source);",
        "if (had) OE.focusIfHere(held ? document.querySelector('#oe-account button')"
        " : root.querySelector('input.search'));",
        "if (err.needsSignIn) {",
        "if (had) OE.focusIfHere(root.querySelector('button.primary'));",
        "const had = !held && left();",
        "listsNone(app, " + _js(STILL_READ_FAILED) + ");",
        "const card = OE.el('div', { class: 'card error', tabindex: '-1' },",
        "'The projects of the key still in effect could not be read: ' + err.message",
        "root.replaceChildren(card);",
        "if (had) OE.focusIfHere(card);",
    )
    assert _in_order(
        signed_out,
        "OE.focusIfHere(held);",
        "listsNone(app, " + _js(STILL_READING) + ");",
        "const left = () =>",
    )
    assert signed_out.count("const had = left();") == 2
    assert "document.activeElement === held;" not in signed_out
    assert "'Signed out here." not in signed_out
    assert "signed out" not in _code(signed_out).lower()
    for told in (STILL_READING, STILL_READ_FAILED):
        assert told.startswith("still signed in: "), told
    sign_in = _between(studio, "function renderSignIn(app) {", "\n}\n")
    call = "renderProjects(app, out.projects, status.account, status.source);"
    assert call in sign_in
    shown = _between(studio, "async function show(app, params) {", "\n}\n")
    assert (
        "const view = renderProjects(app, data.projects, data.account, data.source);"
        in shown
    )


def test_s01_the_account_card_names_the_key_and_keeps_its_sign_out_off() -> None:
    """S01 (e): the panel's account card has a "Key" row that names the app's
    own key as such; a key entered on the sign-in page reads so (remembered
    or not); the card draws no Sign out for the app's own key."""
    page = _read("sign_in.html")
    source = _between(page, "const SOURCE = {", "\n};") + "\n"
    for name, said in (
        ("environment", KEY_ROWS["environment"]),
        ("'key-file'", KEY_ROWS["key-file"]),
        ("'signed-in'", KEY_ROWS["signed-in"]),
    ):
        assert f"  {name}: {_js(said)},\n" in source, name
    assert source.count("// WORDING: r5c S01") == 2
    assert (
        "        // WORDING: r5c S01\n"
        "        ['Key', SOURCE[status.source] || null],\n" in page
    )
    card = _between(page, "function accountCard(", "\n}\n")
    assert (
        "const actions = OE.el('div', { class: 'row' }, "
        "status.source === 'environment' ? null\n"
        "    : OE.signOutButton(app, { cls: 'small', "
        "onSignOut: (s) => render(app, s) }));" in card
    )
    for gone in ("Key source", "server\\'s configuration", "for this session"):
        assert gone not in _code(page), gone
    assert "OE.setAccount(" not in page


def test_s01_premise_page_key_reads_signed_in_own_key_survives_sign_out(
    tmp_path: Path,
) -> None:
    """S01's premise, unchanged auth.KeyStore behaviour that its views rely
    on (this test passes on the tree before round r5-c too): a key entered
    on the sign-in page and remembered reads source 'signed-in' (so the
    account card's "Key" row cannot say "For this session only" and says
    "Entered on the sign-in page"); after Sign out the app's own key is
    still in effect (the server's sign-out result is the sign-out's facts,
    then the status): signed in, source 'environment', no account."""
    store = auth.KeyStore(
        {auth.KEY_ENV: INSTALLED, auth.KEY_FILE_ENV: str(tmp_path / "s.env")}
    )
    store.sign_in(TYPED, auth.Account("Ana", "Lab", 2), remember=True)
    assert store.status()["source"] == "signed-in"
    assert auth.read_key_file(tmp_path / "s.env") == TYPED
    out = {**store.sign_out(), **store.status()}
    assert out["environment_key_remains"] is True
    assert out["signed_in"] is True and out["source"] == "environment"
    assert out["account"] is None
    assert INSTALLED not in repr(out) and TYPED not in repr(out)


def test_s01_an_own_key_that_is_no_key_leaves_the_user_signed_out(
    tmp_path: Path,
) -> None:
    """S01 (b), why the sign-out message reads the key in effect: an
    OLMOEARTH_API_KEY that is no key (here the placeholder .env.example
    ships) is skipped as a key, yet it still sets environment_key_remains;
    after Sign out no key is in effect, the cards show the sign-in, and the
    message must say signed out, not "Still signed in"."""
    store = auth.KeyStore(
        {auth.KEY_ENV: auth._PLACEHOLDER, auth.KEY_FILE_ENV: str(tmp_path / "s.env")}
    )
    store.sign_in(TYPED, auth.Account("Ana", "Lab", 2), remember=True)
    out = {**store.sign_out(), **store.status()}
    assert out["signed_in"] is False and out["source"] is None
    assert out["environment_key_remains"] is True
    assert TYPED not in repr(out)


# --------------------------------------------------------------------------- S03


def test_s03_the_start_cards_lines_say_what_to_do_signed_out_and_in() -> None:
    """S03: signed out, the next line asks to sign in first; signed in, it
    names the jobs and the chat only; the third line has a subject and no
    "error"; the card keeps its three lines and no fold."""
    page = _read("start.html")
    lead = _between(page, "const START_LEAD = {", "\n};") + "\n"
    assert "// WORDING: r4h P1\nconst START_LEAD = {" in page
    for line in (
        f"  // WORDING: r5c S03\n  next: {_js(START_NEXT_OUT)},\n",
        f"  // WORDING: r5c S03\n  cannot: {_js(START_CANNOT)},\n",
    ):
        assert line in lead, line
    assert re.findall(r"^  (\w+): '", lead, re.M) == ["what", "next", "cannot"]
    next_in = f"// WORDING: r5c S03\nconst START_NEXT_IN = {_js(START_NEXT_IN)};\n"
    assert next_in in page
    render = _between(page, "function render(app, data) {", "\n}\n")
    assert _in_order(
        render,
        "const on = signedIn(status);",
        "const lead = OE.lead(on ? { ...START_LEAD, next: START_NEXT_IN } "
        ": START_LEAD);",
        "if (lead) root.append(lead);",
    )
    for gone in ("Sign in if asked", "Nothing about your map yet", "finds the error"):
        assert gone not in page, gone
    assert "OE.fold(" not in page and "<details" not in page
    for line in (START_NEXT_OUT, START_NEXT_IN, START_CANNOT):
        assert not _OFF_FACE.search(line), line


# --------------------------------------------------------------------------- S09


def test_s09_the_sign_in_page_says_what_to_paste_and_labels_its_field() -> None:
    """S09: the page opens with what to paste and where Studio makes a key,
    then where the key goes; the key field is named by a label of its own,
    which stays in view once the key is pasted (the placeholder does not), so
    the field has no aria-label of its own; the card is white on the cream
    page, and the page still loads nothing from elsewhere."""
    form = auth._FORM.format(error="")
    assert _words(_between(form, "<p>", "</p>")) == PAGE_FIRST
    assert "Your Studio API key (Studio: profile, API Keys)" not in form
    assert _in_order(
        form,
        '<form method="post" autocomplete="off">',
        '<label for="api-key" class="field">Studio API key</label>',
        '<input type="password" id="api-key" name="api_key" '
        'placeholder="Paste the key here"',
        'aria-describedby="key-error" required autofocus spellcheck="false">',
        '<label><input type="checkbox" name="remember"',
    )
    assert "aria-label" not in _between(form, '<input type="password"', ">")
    assert form.count('id="api-key"') == 1
    page = auth._PAGE.format(brand="", body=form)
    style = _between(page, "<style>", "</style>")
    assert "body { margin: 0; min-height: 100vh; background: #faf2e9;" in style
    main = _between(style, "main {", "}")
    assert "background: #fff;" in main and "#faf2e9" not in main
    assert (
        "label.field { display: block; margin: 0 0 4px; color: #0a3235; "
        "font-weight: 600; }" in style
    )
    assert style.index("label {") < style.index("label.field {")
    assert not re.search(r"https?://|@import|<link|<script", page)


# --------------------------------------------------------------------------- S13


def test_s13_the_docs_link_in_the_caption_is_underlined() -> None:
    """S13: the docs link inside the caption's line is told apart by an
    underline, not by its weight alone (WCAG 1.4.1); its size stays the
    caption's."""
    rule = _between(_read("start.html"), ".oe-overview-cap button.link {", "}")
    for part in (
        "font-size: 12px;",
        "line-height: 18px;",
        "padding: 0 4px;",
        "text-decoration: underline;",
        "text-underline-offset: 2px;",
    ):
        assert part in rule, part
