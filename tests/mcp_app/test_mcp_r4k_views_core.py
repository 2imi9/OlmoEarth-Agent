# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""The views-core part of round r4-k (16 Nov 2026), read from the view files
(no test here runs the scripts), each string as specs/r4-k-round.md gives it:

- K1: when Claude closes a card (ui/resource-teardown), OE.start's
  onteardown saves every label editor's last answers quietly (oeFlush),
  waiting at most 3 s, then drops the editors' timers, runs OE.cleanups (the
  sign-in's poll) and says 'teardown' in the host report; the editor's quiet
  save shows no toast, runs no onSaved and moves no focus, and oeSettle is
  as it was;
- K2: a stopped call replaces the waiting line only while no result came,
  never with the host's reason; a bridge error is one toast; a card that
  cannot connect says so, and every page runs the same watchdog, a classic
  script, before its module;
- K3: one host report after connect and one on teardown, names only; a
  context update goes unless the host says false;
- K5: the Full screen button works both ways and follows the host's
  context; OE.displayMode is common.js's, the area map's own mode folded
  into it; the callers are unchanged;
- K8: OE.canDownload;
- K9: two example pictures beside the owner's rule, only with the server's
  examples, and the line on where they come from in the sheet's one fold.

A JS concatenation of single-quoted strings and names is read back into its
text (:func:`_concat`), so a missing space between two pieces fails here as
it would on the card."""

from __future__ import annotations

import re

from olmoearth_agent.mcp_app import views

#: One token of a JS concatenation: a single-quoted string, a plus, or a name.
_TOKEN = re.compile(r"\s*(?:'((?:[^'\\]|\\.)*)'|(\+)|([A-Za-z_][A-Za-z0-9_.]*))")

#: The card's lines when it cannot go on (r4-k K2).
CANCELLED = "This step was stopped before it finished. Ask Claude to run it again."
NOT_STARTED = "This card could not start here. Carry on in the chat."
CARD_ERROR = "Something went wrong in this card."

#: The watchdog's statement, the same in every page (r4-k K2), and the whole
#: classic script this round's view-core pages carry before their module.
WATCHDOG = (
    "setTimeout(function(){if(!window.OE_STARTED){var r=document.getElementById"
    "('root');if(r)r.textContent='This card could not start here. Carry on in the"
    " chat.';}},10000);"
)
WATCHDOG_BLOCK = (
    "<script>\n// WORDING: r4k K2\n" + WATCHDOG + '\n</script>\n<script type="module">'
)
#: The pages this part of the round puts the watchdog in (the result and Check
#: cards are the other builder's; every page is checked for the statement).
CORE_PAGES = ("sheet", "studio", "start", "sign-in", "map")

#: The Full screen button's two labels (r4-k K5).
FULL_SCREEN = "Full screen"
BACK_TO_CHAT = "Back to the chat"

#: The example pictures' captions and the line on where they come from (r4-k
#: K9; {year} the examples' year). Round r5-l L2 words the captions for any
#: map (tests/mcp_app/test_mcp_r5l_views.py); round r5-m B1 says what the map
#: is meant to find (tests/mcp_app/test_mcp_r5m_sheet.py).
EXAMPLE_PRESENT = (
    "Present: what the map is meant to find covers more than half of the square "
    "on a clear date."
)
EXAMPLE_ABSENT = (
    "Absent: what the map is meant to find covers half of the square or less on "
    "every clear date."
)
EXAMPLES_NOTE = (
    "A public place, not your map. Contains modified Copernicus Sentinel data "
    "{year}."
)

#: The host report's arguments, names only (r4-k K3).
REPORT_KEYS = {"view", "event", "caps", "modes", "context_keys", "container", "host"}

#: Words no face holds (r4-i's rule).
_OFF_FACE = re.compile(r"\b(?:windows?|budget|stratum|design|review set)\b", re.I)

#: Words of a place, which no line of a card, toast or report may hold.
_PLACE = re.compile(r"\b(?:lat|lon|lng|latitude|longitude|bbox|coordinates?)\b", re.I)


def _read(name: str) -> str:
    """A file of views/, as it is on disk."""
    return (views.VIEW_DIR / name).read_text(encoding="utf-8")


def _page(view: str) -> str:
    """A view's page file, as it is on disk (before views.py fills it in)."""
    return _read(str(views.VIEWS[view]["file"]))


def _between(text: str, start: str, end: str) -> str:
    """The part of text from start up to the first end after it."""
    at = text.index(start)
    return text[at : text.index(end, at)]


def _after(text: str, start: str, end: str) -> str:
    """The part of text after start up to the first end after it."""
    at = text.index(start) + len(start)
    return text[at : text.index(end, at)]


def _code(page: str) -> str:
    """The text with its comments taken out."""
    return re.sub(r"/\*.*?\*/|//[^\n]*", "", page, flags=re.S)


def _in_order(text: str, *pieces: str) -> bool:
    """Whether each piece is in text, each after the one before it."""
    at = [text.index(p) for p in pieces]
    return at == sorted(at)


def _concat(expr: str, names: dict[str, str]) -> str:
    """The text a JS concatenation of single-quoted strings and names makes,
    each name given its value in names; anything else fails."""
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
            out.append(names[name])
        at = match.end()
    return "".join(out)


def _start() -> str:
    """common.js's OE.start, from its head to its closing brace."""
    return _between(
        _read("common.js"),
        "OE.start = async (name, { onResult, onInput, onContext } = {}) => {",
        "\n};\n",
    )


def _editor() -> str:
    """sheet.js's label editor, OE.sheetEditor, whole."""
    return _between(_read("sheet.js"), "OE.sheetEditor = (app, data,", "\n};")


def test_k1_the_teardown_saves_quietly_then_cleans_up() -> None:
    """r4-k K1: onteardown is set before connect; it races every editor's
    oeFlush against TEARDOWN_WAIT_MS (3 s), then drops the editors' timers
    (OE.disposeEditors, which only pages with sheet.js define), runs and
    clears OE.cleanups, fires the 'teardown' report without awaiting it and
    answers {}; it never toasts or moves the focus."""
    common = _read("common.js")
    assert "OE.cleanups = new Set();" in common
    assert "OE.TEARDOWN_WAIT_MS = 3000;" in common
    start = _start()
    assert start.index("app.onteardown = async () => {") < start.index(
        "await app.connect();"
    )
    teardown = _between(start, "  app.onteardown = async () => {", "\n  };\n")
    assert _in_order(
        teardown,
        "document.querySelectorAll('section.sheet')",
        "typeof e.oeFlush === 'function'",
        "await Promise.race([",
        "e.oeFlush()).catch(() => false)",
        "setTimeout(resolve, OE.TEARDOWN_WAIT_MS)",
        "clearTimeout(timer);",
        "if (typeof OE.disposeEditors === 'function') OE.disposeEditors(document);",
        "for (const stop of [...OE.cleanups]) {",
        "try { stop(); } catch (_)",
        "OE.cleanups.clear();",
        "report('teardown');",
        "return {};",
    )
    code = _code(teardown)
    assert "await report" not in code
    for gone in ("toast", "focus", "oeSettle"):
        assert gone not in code, gone
    # The sign-in's poll timer stops with the card.
    sign_in = _between(common, "OE.signInPanel = ", "\n};")
    assert _in_order(
        sign_in,
        "const stop = () => { if (timer) clearTimeout(timer); timer = null; run += 1; };",
        "OE.cleanups.add(stop);",
    )


def test_k1_the_editors_flush_is_quiet_and_bounded() -> None:
    """r4-k K1 in sheet.js: oeFlush waits at most FLUSH_WAIT_MS (2.5 s) for a
    save or read in flight, then saves quietly; oeSettle is word for word as
    before; the hidden and pagehide flush stays. A quiet save shows no toast,
    runs no onSaved, moves no focus and reads nothing again."""
    editor = _editor()
    assert (
        "  wrap.oeSettle = async () => {\n"
        "    while (state.saving || state.syncing) await wait(100);\n"
        "    if (state.stale) return true;\n"
        "    const ok = await settle();\n"
        "    return ok || state.stale;\n"
        "  };\n" in editor
    )
    assert "  const FLUSH_WAIT_MS = 2500;\n" in editor
    flush = _between(editor, "  wrap.oeFlush = async () => {", "\n  };\n")
    assert _in_order(
        flush,
        "const until = Date.now() + FLUSH_WAIT_MS;",
        "while ((state.saving || state.syncing) && Date.now() < until)"
        " await wait(100);",
        "if (state.stale || state.saving || state.syncing) return false;",
        "if (pending()) await save({ auto: true, quiet: true });",
        "return !pending() && !state.stale;",
    )
    assert "toast" not in flush and "focus" not in flush
    # The fallback flush when the panel is hidden or unloaded is kept.
    assert "if (document.visibilityState === 'hidden') flush();" in editor
    assert (
        "window.addEventListener('pagehide', flush, { signal: life.signal });" in editor
    )
    assert "async function save({ auto = false, quiet = false } = {}) {" in editor
    save = _between(editor, "  async function save(", "\n  }\n")
    for guarded in (
        # r4-l L3: through reopenToast (OE.toast, as an error), which keeps it.
        "if (!quiet) reopenToast('Not saved: this sheet changed since the panel"
        " opened it. Reopen the sheet to go on.');",
        # r5-k K5: a save's own statuses are the footer's line (footerSay),
        # still never on a quiet save.
        "if (!auto && !quiet) footerSay(saveSaid, 'Every change is saved to ' "
        "+ data.name + '.');",
        "      } else if (!quiet) {\n        footerSay(saveSaid, 'Not saved' + ",
        "        if (!quiet) {\n          OE.toast('Not saved: this sheet changed",
    ):
        assert guarded in save, guarded
    saved = _between(save, "      if (!quiet) {\n        // r5-k K5", "\n      }\n")
    # r5-k K5 (after review): said beside the counter (counterSay), once the
    # save is over; still never on a quiet save.
    assert "savedWords = 'Saved to ' + data.name + '.'" in saved
    assert "      if (savedWords) counterSay(savedWords);\n" in save
    assert "keptNote = '';" in saved and "if (onSaved) onSaved();" in saved
    assert save.count("onSaved") == 2
    assert _in_order(
        save,
        "quietSave = quiet;",
        "updateProgress();",
        "quietSave = false;",
        "if (quiet) return;",
        "if (stale) await resync();",
    )
    # No focus moves while a quiet save runs (the footer's two focus moves).
    assert editor.index("let quietSave = false;") < editor.index(
        "function updateProgress() {"
    )
    progress = _between(editor, "  function updateProgress() {", "\n  }\n")
    assert "if (!quietSave && !showSave && !saveButton.hidden" in progress
    finish = _between(editor, "  function updateFinish() {", "\n  }\n")
    assert "if (ok && state.finishFrom !== null && !quietSave) {" in finish


def test_k2_a_stopped_call_an_error_and_a_failed_connect() -> None:
    """r4-k K2: the three lines, each marked; every handler is set before
    connect; the cancelled line replaces #root only while no result came
    (set when ontoolresult fires) and never reads the host's reason; a
    bridge error is one toast, once per page, a late answer after a wait
    that ended left out; connect is in try/catch, the failure says
    NOT_STARTED, and OE_STARTED is set right after it connects."""
    common = _read("common.js")
    for name, text in (
        ("CANCELLED", CANCELLED),
        ("NOT_STARTED", NOT_STARTED),
        ("CARD_ERROR", CARD_ERROR),
    ):
        assert f"// WORDING: r4k K2\nOE.{name} = '{text}';" in common, name
    start = _start()
    connect = start.index("await app.connect();")
    for handler in (
        "app.ontoolresult = (params) => {",
        "app.onhostcontextchanged = (ctx) => {",
        "app.ontoolcancelled = () => {",
        "app.onerror = (err) => {",
        "app.onteardown = async () => {",
    ):
        assert start.index(handler) < connect, handler
    result = _between(start, "  app.ontoolresult = (params) => {", "\n  };\n")
    assert "resultCame = true;" in result
    assert "return onResult ? onResult(params, app) : undefined;" in result
    cancelled = _between(start, "  app.ontoolcancelled = () => {", "\n  };\n")
    assert _in_order(
        cancelled,
        "if (resultCame) return;",
        "root.replaceChildren(OE.el('div', { class: 'card summary', role: 'status' },"
        " OE.CANCELLED));",
    )
    assert "reason" not in _code(start)
    error = _between(start, "  app.onerror = (err) => {", "\n  };\n")
    assert _in_order(
        error,
        "if (errorShown || /unknown message id/i.test(",
        "errorShown = true;",
        "OE.toast(OE.CARD_ERROR, true);",
    )
    assert _in_order(
        start,
        "  try {\n    await app.connect();\n  } catch (_) {\n",
        "if (root) root.textContent = OE.NOT_STARTED;",
        "return app;",
        "window.OE_STARTED = true;",
        "if (root && root.textContent === OE.NOT_STARTED && waiting.length)"
        " root.replaceChildren(...waiting);",
    )
    for text in (CANCELLED, NOT_STARTED, CARD_ERROR):
        for promise in ("list", "nothing was saved", "not saved"):
            assert promise not in text, (text, promise)
        assert _OFF_FACE.search(text) is None, text
        assert _PLACE.search(text) is None and not re.search(r"\d", text), text


def test_k2_every_page_runs_the_same_watchdog_before_its_module() -> None:
    """r4-k K2: each of the seven pages holds the watchdog's statement in a
    classic inline script (no type) marked r4k K2, before its module script;
    its words are OE.NOT_STARTED's. This part's five pages hold the whole
    block word for word, and views.py keeps it in the page it serves."""
    for name in views.VIEWS:
        page = _page(name)
        at = page.index(WATCHDOG)
        assert page.count(WATCHDOG) == 1, name
        assert at < page.index('<script type="module">'), name
        element = page[page.rindex("<script", 0, at) : page.index("</script>", at)]
        assert element.startswith("<script>"), name
        assert "// WORDING: r4k K2" in element, name
        assert "tellClaude" not in element, name
    for name in CORE_PAGES:
        assert WATCHDOG_BLOCK in _page(name), name
        assert WATCHDOG in views.view_html(name), name
    words = _after(WATCHDOG, "r.textContent='", "';")
    assert words == NOT_STARTED
    assert f"OE.NOT_STARTED = '{words}';" in _read("common.js")


def test_k3_one_host_report_after_connect_names_only() -> None:
    """r4-k K3: olmoearth_app_host_report gets exactly the contract's keys:
    the server's view name, the event, the names of the host's capabilities,
    display modes and context keys (each at most 40 names of at most 40
    characters from [A-Za-z0-9_.:-]), the container's kind and the host's
    name; a failure is swallowed. It goes once per event: 'start' right
    after connect, 'teardown' when the card closes."""
    common = _read("common.js")
    assert "OE.REPORT_MAX = 40;" in common
    assert (
        "OE.reportName = (value) => String(value ?? '')"
        ".replace(/[^A-Za-z0-9_.:-]+/g, '').slice(0, 40);" in common
    )
    names = _between(common, "OE.reportNames = (list) => ", ";\n")
    assert ".filter(Boolean).slice(0, OE.REPORT_MAX)" in names
    report = _between(common, "OE.hostReport = (app, view, event) => {", "\n};\n")
    args = _between(report, "OE.call(app, 'olmoearth_app_host_report', {", "})")
    assert set(re.findall(r"^ {6}([a-z_]+)\b", args, re.M)) == REPORT_KEYS
    for text in (
        "view: OE.reportName(view),",
        "caps: OE.reportNames(Object.keys(caps)),",
        "modes: OE.reportNames(ctx.availableDisplayModes),",
        "context_keys: OE.reportNames(Object.keys(ctx)),",
        "const container = box.height !== undefined ? 'height' : box.maxHeight !=="
        " undefined ? 'maxHeight' : 'none';",
        "const named = String(info.name ?? '')",
        "host: /[A-Za-z]/.test(named) ? named : '',",
        ".then(() => true, () => false);",
    ):
        assert text in report, text
    code = _code(report)
    for gone in ("toast", "focus", "toolInfo", "console"):
        assert gone not in code, gone
    assert _PLACE.search(code) is None
    start = _start()
    assert _in_order(
        start,
        "const reported = new Set();",
        "if (reported.has(event)) return Promise.resolve(false);",
        "reported.add(event);",
        "return OE.hostReport(app, view, event);",
        "window.OE_STARTED = true;",
        "OE.takeContext(app, null);",
        "report('start');",
    )
    assert "await report(" not in start
    # The view is the server's name for it (views.VIEWS), by the name each
    # page starts with.
    block = _between(common, "OE.VIEW_NAMES = {", "\n};")
    known = dict(re.findall(r"'(olmoearth-[a-z-]+)': '([a-z-]+)'", block))
    for name in views.VIEWS:
        started = re.findall(r"OE\.start\('([a-z-]+)'", _page(name))
        assert len(started) == 1, name
        assert known[started[0]] == name, name


def test_k3_a_context_update_goes_unless_the_host_says_false() -> None:
    """r4-k K3: OE.tellClaude sends where the host's capabilities carry
    updateModelContext true or no such key (the stable spec has none), and
    skips only where the key is there and false, or before the host
    connected; the catch, the 800-character cap and the dedup stay."""
    common = _read("common.js")
    body = _between(common, "OE.tellClaude = ", "\n};")
    assert _in_order(
        body,
        "if (!app || typeof app.updateModelContext !== 'function') return;",
        "if (!caps || typeof caps !== 'object') return;",
        "if ('updateModelContext' in caps && !caps.updateModelContext) return;",
        "if (whole === OE.contextSent) return;",
        ".catch(() => {});",
    )
    assert "if (!caps || !caps.updateModelContext) return;" not in body
    assert "OE.CONTEXT_MAX = 800;" in common
    assert "toast" not in body and "focus" not in body


def test_k5_the_full_screen_button_works_both_ways() -> None:
    """r4-k K5: inline the button asks for 'fullscreen' and says Full
    screen, in full screen it asks for 'inline' and says Back to the chat,
    each shown only where the host lists that mode; every button is
    relabelled or hidden at each context change (taken before onContext)
    and at connect; the host's answer is the mode now. OE.displayMode is
    common.js's; the area map's own mode is gone; the callers are as
    before."""
    common = _read("common.js")
    assert f"// WORDING: r4k K5\nOE.FULL_SCREEN = '{FULL_SCREEN}';" in common
    assert f"// WORDING: r4k K5\nOE.BACK_TO_CHAT = '{BACK_TO_CHAT}';" in common
    assert _code(common).count(f"'{FULL_SCREEN}'") == 1
    target = _between(common, "OE.fullscreenTarget = (app) => {", "\n};\n")
    for text in (
        "const full = OE.displayMode === 'fullscreen';",
        "const mode = full ? 'inline' : 'fullscreen';",
        "label: full ? OE.BACK_TO_CHAT : OE.FULL_SCREEN,"
        " offered: modes.includes(mode)",
    ):
        assert text in target, text
    sync = _between(common, "OE.syncFullscreen = (app) => {", "\n};\n")
    assert "button.textContent = target.label;" in sync
    assert "button.hidden = !target.offered;" in sync
    take = _between(common, "OE.takeContext = (app, changed) => {", "\n};\n")
    assert _in_order(
        take,
        "if (typeof mode === 'string' && mode) OE.displayMode = mode;",
        "OE.hostModes = modes.filter(",
        "OE.syncFullscreen(app);",
    )
    button = _between(
        common, "OE.fullscreenButton = (app, after, cls = 'small') => {", "\n};\n"
    )
    assert _in_order(
        button,
        "const target = OE.fullscreenTarget(app);",
        "if (asking || !target.offered) return;",
        "await app.requestDisplayMode({ mode: target.mode });",
        "if (mode) OE.displayMode = mode;",
        "OE.syncFullscreen(app);",
        "if (target.mode === 'fullscreen' && mode !== 'fullscreen')"
        " OE.toast('Full screen is not available here.', true);",
        "if (after) after(mode);",
        "OE.fullscreenButtons.add(button);",
        "return button;",
    )
    assert "return null" not in button
    start = _start()
    assert _in_order(
        start,
        "OE.takeContext(app, ctx || {});",
        "if (onContext) onContext(ctx || {}, app);",
    )
    assert "OE.displayMode = null;" in common
    sheet_js = _read("sheet.js")
    assert "OE.displayMode = null" not in sheet_js
    assert "requestDisplayMode" not in sheet_js
    page = _page("map")
    assert "displayMode = 'inline'" not in page and "let displayMode" not in page
    fit = _between(page, "function fitHeight() {", "\n}\n")
    assert "OE.displayMode === 'fullscreen'" in fit
    assert "if (OE.displayMode === 'fullscreen') fitHeight();" in page
    # The callers as before; the map lays out the mode it opened in.
    assert "OE.fullscreenButton(app, fitHeight, 'map-overlay')" in page
    assert (
        "onContext: (ctx) => { if (ctx.displayMode) fitHeight(ctx.displayMode); },"
        in page
    )
    assert _in_order(page, "if (full) mapWrap.append(full);", "  fitHeight();\n")
    for name in ("sheet", "result", "check"):
        assert "OE.fullscreenButton(app, OE.sheetMode)" in _page(name), name


def test_k8_can_download_needs_the_method_and_the_hosts_flag() -> None:
    """r4-k K8: OE.canDownload is true only where the client has
    downloadFile and the host's capabilities carry downloadFile; false
    before the host connected and on any failure."""
    common = _read("common.js")
    can = _between(common, "OE.canDownload = (app) => {", "\n};\n")
    assert _in_order(
        can,
        "if (!app || typeof app.downloadFile !== 'function') return false;",
        "app.getHostCapabilities()",
        "return Boolean(caps && typeof caps === 'object' && caps.downloadFile);",
        "} catch (_) {\n    return false;",
    )


def test_k9_example_pictures_beside_the_owners_rule_only() -> None:
    """r4-k K9: the two captions and the fold line, each marked; the
    examples are taken only with the owner's rule, both pictures as data
    URLs and a whole-number year; the pictures go last in the rule block,
    the line in the Details fold (the sheet keeps its one fold), and nothing
    else on the face changes. The sheet view hands the editor the view's
    examples when the sheet read has none."""
    script = _read("sheet.js")
    assert f"// WORDING: r4k K9\nOE.EXAMPLE_PRESENT = '{EXAMPLE_PRESENT}';" in script
    assert f"// WORDING: r4k K9\nOE.EXAMPLE_ABSENT = '{EXAMPLE_ABSENT}';" in script
    note = _after(script, "// WORDING: r4k K9\nOE.EXAMPLES_NOTE = (year) => ", ";\n")
    assert _concat(note, {"year": "2024"}) == EXAMPLES_NOTE.format(year=2024)
    taken = _between(script, "OE.ruleExamples = (data) => {", "\n};\n")
    for text in (
        "data.rule_source === 'owner'",
        "const good = (src) => typeof src === 'string' && src.startsWith('data:image/');",
        "if (!ex || !good(ex.present) || !good(ex.absent)"
        " || !Number.isInteger(ex.year)) return null;",
    ):
        assert text in taken, text
    pictures = _between(script, "OE.examplePictures = (ex) => ", ";\n")
    for text in (
        "[[ex.present, OE.EXAMPLE_PRESENT], [ex.absent, OE.EXAMPLE_ABSENT]]",
        "OE.el('img', { src, alt: caption,",
        "OE.el('figcaption', { class: 'small', 'aria-hidden': 'true' }, caption)",
    ):
        assert text in pictures, text
    # The page's one PNG data URL stays the AGENT tag (test_mcp_server).
    assert "data:image/png;base64," not in script
    editor = _editor()
    assert _in_order(
        editor,
        "const examples = OE.ruleExamples(data);",
        "const ruleExamples = examples ? OE.examplePictures(examples) : null;",
        "const examplesNote = examples ? OE.el('div', { class: 'small' },"
        " OE.EXAMPLES_NOTE(examples.year)) : null;",
        "  const ruleBlock = pageRule ?",
    )
    rule = _between(editor, "  const ruleBlock = pageRule ?", "\n  const clearNote")
    # r5-e H2, after review: the line before the examples is the cloud line
    # alone (the scope is in the Details fold).
    assert rule.endswith("W.cloudLine) : null,\n    ruleExamples) : null;")
    fold = _between(editor, "  const foldLabel = 'Details';", "\n\n")
    assert "instruction, hint, ...foldLines, clearNote, examplesNote," in fold
    layout = _between(editor, "  wrap.append(", "  // One listener for the keys")
    assert "ruleExamples" not in layout and "examplesNote" not in layout
    assert editor.count("OE.fold(") == 1
    page = _page("sheet")
    assert (
        "const page = fresh && !fresh.examples && data.examples"
        " ? { ...fresh, examples: data.examples } : fresh;" in page
    )
    assert "OE.sheetEditor(app, page, { headerAction:" in page
    for text in (EXAMPLE_PRESENT, EXAMPLE_ABSENT, EXAMPLES_NOTE):
        assert _OFF_FACE.search(text) is None, text
        assert _PLACE.search(text) is None, text
