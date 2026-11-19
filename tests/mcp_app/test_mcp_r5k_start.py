# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Round r5-k (18 Nov 2026), track START, as specs/r5-k-round.md gives it
(the owner accepted each decision on 18 Nov 2026). The views are read as
text, as they are on disk (no test here runs the scripts; the browser
harness, scripts/mcp_views_e2e.py, draws them: 11_start_card,
11b_start_card_signed_in, 11e_start_sign_out_refused,
11f_start_sign_out_long_name, 4f_studio_account_paths, 4g_studio_sign_out,
4i_studio_sign_out_unusable_own_key and 4j_studio_sign_out_refused):

- K8 (O17): on a panel at least 680 px wide the start card shows its jobs
  first: its title, its line, the sign-in while signed out (it stays first),
  the jobs (the dark main job, then "Other jobs"), then the OlmoEarth docs
  figure with its caption and docs link; below 680 px the three-step strip
  stays under the line, as before. The figure's box moves in the page (no
  CSS order), at each drawing and when the panel's width crosses 680 px, by
  the width overview.js shows the docs figure from (OE.overview.WIDE).
- K5 (O11), its start part: the sign-out's reply ("Signed out; the saved key
  was removed." or "Still signed in: ...") is said under Sign out, in the
  top line (OE.accountReply, #oe-account-reply), on the start card and the
  projects card, no longer in the status line at the page's foot; a plain
  reply is a polite status, an error an alert; its words are as they were;
  it goes once the top line shows something it is not about (a later
  sign-in, another account's name), an error at the card's next drawing.
  After review: it stands under where Sign out stood by that button's own
  box at the press (OE.signOutAt, OE.placeAccountReply), no longer by a
  width of the panel, and follows when the width changes; a Sign out empties
  the status line while it still says the sign-in made in the panel. The
  account card's Sign out (sign_in.html) keeps the status line.
"""

from __future__ import annotations

import re

from olmoearth_agent.mcp_app import views

#: K8: the width from which the docs figure shows, as a media query.
WIDE = "(min-width: 680px)"

#: K5: the sign-out's two replies (r5-c S01, r4p2), word for word.
STILL_SIGNED_IN = (
    "Still signed in: this app was installed with its own key. Only whoever "
    "installed it can remove it."
)
SIGNED_OUT = "Signed out; the saved key was removed."


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


def _style() -> str:
    """The style sheet every view gets (common.js OE.STYLE), comments and all."""
    return _between(_read("common.js"), "OE.STYLE = `", "\n`;\n")


# --------------------------------------------------------------------------- K8


def test_k8_overview_names_the_width_it_shows_the_docs_figure_from() -> None:
    """K8: overview.js names the width from which the docs figure shows
    (WIDE) and returns it; its one media rule, which shows the docs figure
    and hides the strip, is that width; the start card's white box widens
    for the docs figure at that width too. Still no listener in overview.js:
    the start card places the figure."""
    script = _read("overview.js")
    assert f"  const WIDE = {_js(WIDE)};\n" in script
    assert (
        "  return { STYLE, style, figure, CAPTION, CAPTION_NARROW, WIDE };\n" in script
    )
    style = _between(script, "  const STYLE = `", "`;")
    assert style.count("@media") == 1
    assert f"@media {WIDE} {{\n  .oe-overview img.wide {{ display: block; }}" in style
    assert "addEventListener" not in script and "matchMedia" not in script
    page = _between(_read("start.html"), "<style>", "</style>")
    assert f"@media {WIDE} {{ .oe-overview-wrap {{ max-width: 774px; }} }}" in page


def test_k8_the_start_card_places_the_figure_after_the_jobs_when_wide() -> None:
    """K8: placeOverview moves the figure's box, and nothing else: from WIDE
    to the end of the card (after the jobs), below it right after the card's
    line; a focus inside the box (the docs link), which the move drops, is
    given back only while the panel's frame has it (OE.focusIfHere). One
    listener, on the width's media query, calls it when the width crosses
    WIDE; no CSS order on the page."""
    page = _read("start.html")
    assert "const wideMedia = window.matchMedia(OE.overview.WIDE);\n" in page
    place = _between(page, "function placeOverview() {", "\n}\n")
    assert _in_order(
        place,
        "const wrap = root.querySelector(':scope > .oe-overview-wrap');",
        "if (!wrap) return;",
        "const had = wrap.contains(document.activeElement) "
        "? document.activeElement : null;",
        "if (wideMedia.matches) {",
        "if (root.lastElementChild !== wrap) root.append(wrap);",
        "} else {",
        "const line = root.querySelector(':scope > .oe-lead') "
        "|| root.querySelector(':scope > h1');",
        "if (line && line.nextElementSibling !== wrap) line.after(wrap);",
        "if (had && document.activeElement !== had) OE.focusIfHere(had);",
    )
    moves = re.findall(r"\.(?:append|prepend|after|before|insertBefore)\(", place)
    assert moves == [".append(", ".after("]
    assert ".focus(" not in place
    code = _code(page)
    assert code.count("wideMedia.addEventListener('change', placeOverview);") == 1
    assert code.count("addEventListener(") == 2  # the job tile's press, and this
    # When Claude closes the card, the listener goes (r4-k K1: OE.cleanups).
    assert (
        "OE.cleanups.add(() => wideMedia.removeEventListener('change', "
        "placeOverview));" in code
    )
    style = _between(page, "<style>", "</style>")
    assert re.search(r"\border\s*:", style) is None


def test_k8_render_places_the_figure_at_each_drawing() -> None:
    """K8: render draws the narrow order (title, line, figure, the sign-in
    while signed out, the jobs), then places the figure; with no jobs too,
    before it returns. The sign-in is drawn before the jobs, so it stays
    first when signed out at every width."""
    page = _read("start.html")
    render = _between(page, "function render(app, data) {", "\n}\n")
    assert _in_order(
        render,
        "root.replaceChildren(OE.el('h1', {}, 'What this app can do'));",
        "if (lead) root.append(lead);",
        "root.append(overviewBlock(app));",
        "root.append(OE.signInPanel(app, {",
        "root.append(OE.el('div', { class: 'faint' }, "
        "'Ask Claude what this app can do.'));",
        "    placeOverview();\n    return;\n",
        "root.append(OE.el('section', { class: 'card stack' },",
        "OE.el('ul', { class: 'items jobs', 'aria-label': 'Jobs' }, "
        "jobRows(app, jobs, on))));",
        "\n  placeOverview();",
    )
    assert render.count("placeOverview();") == 2
    assert render.rstrip().endswith("placeOverview();")


def test_k8_the_figure_keeps_its_caption_and_link_and_one_dark_button() -> None:
    """K8: the figure's box is as it was (the figure, its caption's two
    forms, the docs link and its slot); the main job is the one contained
    button signed in, the sign-in's signed out; the job tiles keep one size
    (R1: full width, at least 86 px tall) and their order (the main job,
    then "Other jobs" over the rest)."""
    page = _read("start.html")
    block = _between(page, "function overviewBlock(app) {", "\n}\n")
    assert _in_order(
        block,
        "const [docs, address] = OE.docsLink(app);",
        "OE.el('div', { class: 'oe-overview-wrap' },",
        "OE.overview.figure(),",
        "OE.el('span', { class: 'cap-wide' }, OE.overview.CAPTION),",
        "OE.el('span', { class: 'cap-narrow' }, OE.overview.CAPTION_NARROW), ' ', "
        "docs, address));",
    )
    code = _code(page)
    assert code.count("primary") == 2
    assert "class: 'job' + (main && on ? ' primary' : '')" in code
    assert "OE.focusIfHere(root.querySelector('button.primary'));" in code
    style = _between(page, "<style>", "</style>")
    assert (
        ".jobs li > button.job { width: 100%; min-height: 86px; display: flex; "
        "flex-direction: column; justify-content: center;" in style
    )
    rows = _between(page, "function jobRows(app, jobs, on) {", "\n}\n")
    assert _in_order(
        rows,
        "jobRow(app, main, on),",
        "OE.el('h2', {}, 'Other jobs')",
        "...others.map((job) => jobRow(app, job, on)),",
    )


# --------------------------------------------------------------------------- K5


def test_k5_the_top_line_holds_the_reply_under_the_account_item() -> None:
    """K5: the top line's words are the sentence, the account item and,
    after them, the sign-out's reply: a status in place (role status,
    polite), empty until Sign out is pressed. Its row is its own, under the
    sentence and the item, its words at the line's right end by default;
    after review no width of the panel moves them (no media rule): where
    Sign out stood does (test_k5_the_reply_stands_under_where_sign_out_stood).
    Empty, it is drawn but out of the line's flow (never display none), so a
    screen reader hears its first words."""
    common = _read("common.js")
    line = _between(common, "OE.line = () => ", ";\n")
    assert _in_order(
        line,
        "OE.el('span', { class: 'oe-said' }, OE.SUB),",
        "OE.el('span', { id: 'oe-account', class: 'oe-account' }),",
        "OE.el('div', { id: 'oe-account-reply', class: 'oe-account-reply muted', "
        "role: 'status', 'aria-live': 'polite' })",
    )
    style = _style()
    for rule in (
        ".oe-account-reply { flex: 1 0 100%; min-width: 0; text-align: right; "
        "overflow-wrap: anywhere; }",
        ".oe-account-reply:empty { position: absolute; }",
    ):
        assert rule in style, rule
    assert _in_order(style, ".oe-account button.link {", ".oe-account-reply {")
    reply_rules = [r for r in style.split("\n") if ".oe-account-reply" in r]
    assert all("display: none" not in r for r in reply_rules), reply_rules
    assert not [r for r in reply_rules if "@media" in r], reply_rules
    assert "max-width: 559px" not in style


def test_k5_the_reply_stands_under_where_sign_out_stood() -> None:
    """K5, after review: where the reply's words go follows Sign out's own
    box, not a width of the panel. The press measures Sign out against the
    line's words while it is drawn, before its call (OE.signOutAt: its gap to
    their right end, its right edge from their left end, the item's width)
    and gives that to the reply; OE.placeAccountReply first clears its own
    alignment, then ends the words at Sign out's right edge (right, that gap
    at their right) when Sign out stood at least as far from the left edge
    as the words are wide, else starts them at the left edge. When the
    width changes (a resize), a Sign out still drawn is measured again; one
    that is gone is put where the line's wrap would put the item (the
    sentence's flex-basis, the gap and the item's width at the press)."""
    common = _read("common.js")
    at = _between(common, "OE.signOutAt = ", "\n};")
    assert _in_order(
        at,
        "const t = out.closest('.oe-line-text').getBoundingClientRect(), "
        "b = out.getBoundingClientRect();",
        "return { fromRight: Math.max(0, t.right - b.right), "
        "right: b.right - t.left,",
        "item: out.closest('.oe-account').getBoundingClientRect().width };",
    )
    place = _between(common, "OE.placeAccountReply = ", "\n};")
    assert _in_order(
        place,
        "reply.style.textAlign = reply.style.paddingRight = '';",
        "if (!reply.textContent || !replyAt) return;",
        "const out = resized ? line.querySelector('#oe-account button') : null;",
        "let at = out ? OE.signOutAt(out) : replyAt;",
        "if (resized && !out) {",
        "item = Math.min(replyAt.item, width);",
        "at = px(line.querySelector('.oe-said'), 'flexBasis') "
        "+ px(line, 'columnGap') + item <= width",
        "? { fromRight: 0, right: width } "
        ": { fromRight: width - item, right: item };",
        "range.selectNodeContents(reply);",
        "if (at.right >= wide) {",
        "reply.style.textAlign = 'right';",
        "reply.style.paddingRight = at.fromRight + 'px';",
        "} else reply.style.textAlign = 'left';",
    )
    assert "focus" not in _code(place)
    code = _code(common)
    resize = "window.addEventListener('resize', () => OE.placeAccountReply(true));"
    assert code.count(resize) == 1
    assert code.count("OE.placeAccountReply(") == 2  # the reply's write, the resize
    button = _between(common, "OE.signOutButton = ", "\n};")
    assert _in_order(
        button,
        "const at = inLine ? OE.signOutAt(out) : null;",
        "await OE.call(app, 'olmoearth_app_sign_out');",
    )


def test_k5_the_reply_is_a_status_and_an_error_an_alert() -> None:
    """K5: OE.accountReply writes the reply: a plain one polite (role
    status), an error an alert (assertive, the error colour), as OE.toast
    does; it records the top line's state the words are about, whether they
    are an error and where Sign out stood (none once emptied), then places
    them; a page with no top line says it in the status line, and says
    nothing there to empty it. OE.accountState names that state: after
    review, with any key but the app's own, the name the line shows, so
    another account is another state."""
    common = _read("common.js")
    state = _between(common, "OE.accountState = ", ";\n")
    assert state == (
        "OE.accountState = (signedIn, source = null, account = null) => "
        "(!signedIn ? 'out'\n"
        "  : (source === 'environment' ? 'own' : "
        "'in|' + OE.accountShort(account)))"
    )
    for name in ("replyFor", "replyAt", "drawnState", "signInSaid"):
        assert f"let {name} = null;" in common, name
    assert "let replyIsError = false;" in common
    reply = _between(common, "OE.accountReply = ", "\n};")
    assert reply.startswith(
        "OE.accountReply = (text, isError = false, state = null, at = null) => {"
    )
    assert _in_order(
        reply,
        "const reply = document.getElementById('oe-account-reply');",
        "if (!reply) { if (text) OE.toast(text, isError); return; }",
        "replyFor = text ? state : null;",
        "replyIsError = Boolean(text) && isError;",
        "replyAt = text ? at : null;",
        "reply.setAttribute('role', isError ? 'alert' : 'status');",
        "reply.setAttribute('aria-live', isError ? 'assertive' : 'polite');",
        "reply.className = 'oe-account-reply ' + (isError ? 'error' : 'muted');",
        "reply.textContent = text;",
        "OE.placeAccountReply();",
    )
    assert "focus" not in _code(reply)


def test_k5_a_sign_out_empties_the_status_lines_sign_in() -> None:
    """K5, after review: the sign-in made in the panel says its line in the
    status line at the page's foot and keeps it (signInSaid); the reply of a
    Sign out that went through empties that line while it still says that
    sign-in, and only then, so "Signed in to OlmoEarth Studio as ..." never
    stays under a signed-out card. An error's reply leaves it; Sign out keeps
    its one OE.toast (the account card's)."""
    common = _read("common.js")
    panel = _between(common, "OE.signInPanel = ", "\n};")
    assert _in_order(
        panel,
        "signInSaid = OE.accountLine(now.account);",
        "OE.toast(signInSaid);",
        "if (onDone) onDone(now);",
    )
    assert "OE.toast(OE.accountLine(" not in _code(panel)
    reply = _between(common, "OE.accountReply = ", "\n};")
    assert _in_order(
        reply,
        "if (!reply) { if (text) OE.toast(text, isError); return; }",
        "if (text && !isError) {",
        "const bar = document.getElementById('oe-toast');",
        "if (bar && signInSaid && bar.textContent === signInSaid) OE.toast('');",
        "signInSaid = null;",
        "replyFor = text ? state : null;",
    )
    assert _code(reply).count("OE.toast(") == 2  # no top line; the sign-in's line
    assert _code(common).count("signInSaid") == 6


def test_k5_sign_out_in_the_top_line_replies_under_itself() -> None:
    """K5: Sign out pressed in the top line says its reply under itself
    (OE.accountReply, with the state the line then shows, its name too, and
    where Sign out stood), its words as they were and chosen as before (by
    the key in effect); an error the same way, as an alert, about what the
    line shows now (drawnState: the press changed nothing); a Sign out
    outside the top line (the account card, sign_in.html) keeps the status
    line. The sign-out's words have one place, common.js, and reach the
    status line only by that account card's branch (after review: the old
    OE.toast(result. ...) shape is gone)."""
    common = _read("common.js")
    button = _between(common, "OE.signOutButton = ", "\n};")
    assert _in_order(
        button,
        "const inLine = Boolean(out.closest('.oe-line'));",
        "const reply = (text, isError, state) => (inLine "
        "? OE.accountReply(text, isError, state, at) : OE.toast(text, isError));",
        "out.disabled = true;",
        "const result = await OE.call(app, 'olmoearth_app_sign_out');",
        "const words = result.signed_in && result.source === 'environment'",
        "? " + _js(STILL_SIGNED_IN),
        ": " + _js(SIGNED_OUT) + ";",
        "reply(words, false, "
        "OE.accountState(result.signed_in, result.source, result.account));",
        "if (onSignOut) onSignOut(result);",
        "reply(err.message, true, drawnState);",
    )
    code = _code(button)
    assert code.count("OE.toast(") == 1
    assert "OE.toast(result." not in code  # the old shape (r5-j)
    assert "OE.toast(words" not in code
    assert code.count("words") == 2  # declared, then given only to reply(words, ...)
    # Scoped to the button: common.js has another `const words = `.
    chosen = _between(button, "const words = ", ";\n")
    assert "OE.signOutButton(app, { cls: 'small', " in _read("sign_in.html")
    for words in (STILL_SIGNED_IN, SIGNED_OUT):
        assert _js(words) in chosen, words
        assert _code(common).count(_js(words)) == 1, words
        # The view files on disk, not views.view_html: each page holds common.js.
        homes = [
            p.name
            for p in sorted(views.VIEW_DIR.iterdir())
            if p.suffix in (".html", ".js") and _js(words) in _read(p.name)
        ]
        assert homes == ["common.js"], (words, homes)
    # The reply is the sign-out's alone: no view of its own writes it.
    for path in sorted(views.VIEW_DIR.iterdir()):
        if path.suffix in (".html", ".js") and path.name != "common.js":
            assert "OE.accountReply(" not in _read(path.name), path.name


def test_k5_the_reply_goes_once_the_top_line_shows_something_else() -> None:
    """K5: OE.setAccount empties the reply whenever the state it draws is
    not the one the reply is about: the sign-out's own redraw (signed out,
    or the app's own key) keeps it; a later sign-in, another key or (after
    review) another account's name empties it, so "Signed out" never stands
    beside a name; an error's reply goes at the card's next drawing, the
    same account too. It records what the line shows (drawnState, which a
    refused press's reply is about), does so before it draws, and moves no
    focus."""
    common = _read("common.js")
    body = _between(common, "OE.setAccount = ", "\n};")
    assert _in_order(
        body,
        "const item = document.getElementById('oe-account');",
        "if (!item) return null;",
        "const state = OE.accountState(signedIn, source, account);",
        "if (replyIsError || replyFor !== state) OE.accountReply('');",
        "drawnState = state;",
        "if (!signedIn) {",
        "item.replaceChildren();",
    )
    assert "focus" not in _code(body)
    assert _code(common).count("drawnState") == 3  # declared, drawn, refused


def test_k5_the_start_and_projects_cards_keep_their_sign_out_paths() -> None:
    """K5: the start card and the projects card give the top line the key's
    source at each drawing, as before, so the reply of a sign-out that
    leaves the app's own key in effect stays ('own'), and the projects
    card's sign-in after a refused key empties it ('out'); the projects card
    no longer says the toast says it."""
    start = _read("start.html")
    render = _between(start, "function render(app, data) {", "\n}\n")
    assert (
        "OE.setAccount(app, { signedIn: on, account: status.account, "
        "source: status.source, onSignOut: (s) => signedOut(app, data, s) });" in render
    )
    studio = _read("studio.html")
    left = _between(studio, "async function signedOut(app, result) {", "\n}\n")
    assert _in_order(
        left,
        "renderSignIn(app);",
        "const held = OE.setAccount(app, { signedIn: true, account: result.account, "
        "source: result.source,",
        "renderProjects(app, out.projects, result.account, result.source);",
        "if (err.needsSignIn) {",
    )
    assert left.count("renderSignIn(app);") == 2
    sign_in = _between(studio, "function renderSignIn(app) {", "\n}\n")
    assert "OE.setAccount(app, { signedIn: false });" in sign_in
    assert "as the toast says" not in studio
    assert studio.count("as the reply under Sign out says") == 2
