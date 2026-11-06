# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""The web UI shows what the default agent does and counts what the catalog has.

The landing page's example runs and the demo mode's canned runs call only
tools of the default registry; the tagline's, the skills button's and the "/"
menu's skill counts, and the cards, are the catalog's. Every local file the page
links is in ``webui/``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from olmoearth_agent.skills.registry import (
    READ_ONLY_ENV,
    SKILLS,
    build_default_registry,
)

_WEBUI = Path(__file__).resolve().parents[1] / "webui"

#: The run ``make verify-demo`` recorded, every call's arguments and envelope.
_VERIFICATION_DEMO_JSON = _WEBUI.parent / "docs" / "verification_demo.json"

#: Link targets that name no file: an anchor, inline data, a mail address.
_NOT_A_FILE = ("#", "data:", "mailto:")


def _read(*parts: str) -> str:
    """One web UI file's text."""
    return _WEBUI.joinpath(*parts).read_text(encoding="utf-8")


def _scenario(run_js: str, name: str) -> str:
    """The body of one demo scenario of ``run.js``'s ``SCENARIOS``."""
    block = re.search(rf"\n  {name}: \{{\n(.*?)\n  \}},", run_js, re.S)
    assert block, name
    return block.group(1)


def _recorded_caption(tool: str, design: str | None = None) -> str:
    """The figure caption the one recorded call of ``tool`` (and design) returned."""
    record = json.loads(_VERIFICATION_DEMO_JSON.read_text(encoding="utf-8"))
    captions = [
        call["envelope"]["result"]["figure_caption"]
        for call in record["calls"]
        if call["tool"] == tool
        and (design is None or call["envelope"]["result"].get("design") == design)
    ]
    assert len(captions) == 1, (tool, design, len(captions))
    caption = captions[0]
    assert isinstance(caption, str), (tool, design)
    return caption


def _function(source: str, signature: str) -> str:
    """One top-level JavaScript function's text, from its signature to its end."""
    start = source.index(signature)
    return source[start : source.index("\n}\n", start) + 2]


def _module_graph() -> set[str]:
    """The web UI modules index.html loads, following each module's imports."""
    page = _read("index.html")
    todo = re.findall(r'<script type="module" src="js/([\w-]+\.js)"', page)
    seen: set[str] = set()
    while todo:
        name = todo.pop()
        if name not in seen:
            seen.add(name)
            todo += re.findall(r"from '\./([\w-]+\.js)'", _read("js", name))
    return seen


def _example_calls() -> list[str]:
    """Every tool the landing page's transcripts and the demo's scenarios call."""
    page = re.findall(
        r"called <code>(olmoearth_[a-z0-9_]+)</code>", _read("index.html")
    )
    demo = re.findall(r"tool: '(olmoearth_[a-z0-9_]+)'", _read("js", "run.js"))
    return page + demo


def test_the_example_runs_call_only_default_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(READ_ONLY_ENV, raising=False)
    calls = _example_calls()
    assert "olmoearth_compare_results" in calls
    assert "olmoearth_plan_label_sample" in calls
    assert set(calls) <= set(build_default_registry().names())


def test_the_demo_verification_replays_quote_make_verify_demo() -> None:
    """The demo's review-set and error-rate runs replay docs/VERIFICATION_DEMO.md.

    Each figure the two canned runs show is a figure ``make verify-demo``
    recorded on the Dynamic World tile, and each answer says where it comes
    from; the README's walkthrough video shows these two runs.
    """
    run_js = _read("js", "run.js")
    verify_demo = _WEBUI.parent / "docs" / "VERIFICATION_DEMO.md"
    page = verify_demo.read_text(encoding="utf-8")
    quoted = {
        "review": ["791", "15,813", "532"],
        "error": ["19.0%", "14.8% to 23.9%", "39.6%", "25.9% to 54.6%", "15,813"],
    }
    for name, figures in quoted.items():
        block = _scenario(run_js, name)
        assert "make verify-demo" in block, name
        for figure in figures:
            assert figure in block, (name, figure)
            assert figure in page, (name, figure)
    # The review run's two shares: 532 of the 791 windows, and the tile's error rate.
    assert "532 of the 3,048 errors" in page
    assert "3,048 of 15,813 windows wrong: 19.3%" in page
    assert round(100 * 532 / 791) == 67 and "(67%)" in run_js
    assert "wrong 19% of the time" in run_js
    assert "48 the map calls this class" in page and "48 labelled windows" in run_js


def test_the_demo_verification_replays_show_the_recorded_figures() -> None:
    """The review-set and error-rate demo runs show the tool's figure and caption.

    Each scenario's ``figure_src`` is an SVG shipped in webui/ (opened in a
    new tab as a document, so it holds no script), and its ``figure_caption``
    is, character for character, the caption ``make verify-demo`` recorded in
    docs/verification_demo.json: the review set's, and the random design's
    error estimate's. ``demoEvents`` puts both where the renderer reads them.
    """
    run_js = _read("js", "run.js")
    expected = {
        "review": (
            "demo/figures/review_set.svg",
            _recorded_caption("olmoearth_review_set"),
        ),
        "error": (
            "demo/figures/error_rate.svg",
            _recorded_caption("olmoearth_estimate_map_error", design="random"),
        ),
    }
    for name, (source, caption) in expected.items():
        block = _scenario(run_js, name)
        assert re.findall(r"figure_src: '([^']+)'", block) == [source], name
        figure = _WEBUI / source
        assert figure.is_file(), figure
        assert "<script" not in figure.read_text(encoding="utf-8"), figure
        shown = re.findall(r'figure_caption: "([^"\\]*)"', block)
        assert shown == [caption], name
    assert "toolResult.figure_src = sc.figure_src;" in run_js
    assert "resultObj.figure_caption = sc.figure_caption;" in run_js


def test_the_landing_page_and_the_slash_menu_count_the_catalog() -> None:
    numbered = [s for s in SKILLS if s.number >= 1]
    total = len(numbered)
    page = _read("index.html")
    tagline = re.findall(r"<strong>(\d+) skills</strong>", page)
    assert tagline == [str(total)]
    assert "opt-in" not in page
    assert re.findall(r"Browse the (\d+) skills", page) == [str(total)]
    slash = re.findall(r"pick one of the (\d+)\s+skills", _read("js", "slash.js"))
    assert slash == [str(total)]
    cards = _read("js", "skills.js")
    assert len(re.findall(r"\{ n: \d+,\s+slug: ", cards)) == total
    assert "optIn" not in cards


def test_the_page_links_only_files_it_ships() -> None:
    """Every local ``src`` and ``href`` of the page names a file in webui/.

    The architecture diagram its menu linked went on 31 October 2026.
    """
    targets = re.findall(r'(?:src|href)="([^"]+)"', _read("index.html"))
    local = [t for t in targets if "://" not in t and not t.startswith(_NOT_A_FILE)]
    assert "styles.css" in local
    missing = [t for t in local if not (_WEBUI / t).exists()]
    assert not missing, missing


def test_the_figure_store_is_bounded_and_never_fails_a_render() -> None:
    """figstore.js keeps figures in IndexedDB, bounded, and every call resolves.

    A saved chat keeps no SVG in localStorage (about 5 MB), so the web UI keeps
    each live figure in IndexedDB (database 'oe_figures', store 'figures',
    keyed by the figure's path). Past 50 MB or 300 figures the least recently
    used go first. Where IndexedDB is missing, blocked or fails, storing does
    nothing and a lookup finds nothing. The suite runs no JavaScript: these
    check the source, and that the page's module graph loads the store.
    """
    assert "figstore.js" in _module_graph()
    store = _read("js", "figstore.js")
    for line in (
        "const DB_NAME = 'oe_figures';",
        "const STORE = 'figures';",
        "const MAX_BYTES = 50 * 1024 * 1024;",
        "const MAX_COUNT = 300;",
        "indexedDB.open(DB_NAME, 1)",
        "createObjectStore(STORE, { keyPath: 'key' })",
        "store.createIndex(BY_USE, ['at', 'bytes'])",
        "const rec = { key, svg, caption: text, bytes, at: Date.now() };",
        "if (total <= MAX_BYTES && count <= MAX_COUNT) break;",
        "store.put({ ...rec, at: Date.now() });",
    ):
        assert line in store, line
    for name in (
        "putFigure(key, svg, caption)",
        "getFigure(key)",
        "deleteFigures(keys)",
        "clearFigures()",
    ):
        assert re.search(rf" \*/\nexport function {re.escape(name)} \{{", store), name
    open_db = _function(store, "function openDb() {")
    assert "req.onerror = () => { settled = true; resolve(null); };" in open_db
    assert "} catch (e) {\n      resolve(null);" in open_db
    in_store = _function(store, "function inStore(mode, work, fallback, aborted) {")
    assert "tx.oncomplete = () => resolve(fallback);" in in_store
    assert (
        "tx.onabort = () => resolve(aborted ? aborted(tx.error) : fallback);"
        in in_store
    )
    assert ".catch(() => fallback)" in in_store
    # The transaction is declared outside the try, and a throw after a write
    # was queued aborts it before the fallback resolves.
    assert in_store.index("let tx = null;") < in_store.index("try {")
    catch = in_store[in_store.index("} catch (e) {") :]
    assert catch.index("tx.abort();") < catch.index("resolve(fallback);")
    # A lookup reads in a readonly transaction and answers on the read's
    # success; the time of use is written in its own transaction, not awaited.
    get = _function(store, "export function getFigure(key) {")
    assert "inStore('readonly'," in get
    assert "keep({ svg: rec.svg," in get
    assert "void touch(key);" in get
    assert "readwrite" not in get
    touch = _function(store, "function touch(key) {")
    assert "inStore('readwrite'," in touch
    assert "store.put({ ...rec, at: Date.now() });" in touch
    # Eviction never takes the figure just stored.
    evict = _function(store, "function evict(store, spare) {")
    assert "if (key === spare) continue;" in evict
    assert "evict(store, key);" in _function(
        store, "export function putFigure(key, svg, caption) {"
    )
    assert "store.index(BY_USE).openKeyCursor()" in _function(
        store, "function byAge(store, then) {"
    )
    assert "reject(" not in store


def test_the_figure_store_opens_its_version_and_is_not_held_by_a_block() -> None:
    """figstore.js creates its store only from version 0 and retries a block.

    The upgrade branches on the version it comes from, so a later version's
    upgrade does not create the store again. A call blocked by another tab
    (holding an older version open) gets null, but the null is not kept, so
    the next call opens again; a connection that opens after the promise
    answered null is closed, not leaked.
    """
    store = _read("js", "figstore.js")
    open_db = _function(store, "function openDb() {")
    assert "req.onupgradeneeded = (event) => {" in open_db
    assert "if (event.oldVersion < 1) {" in open_db
    assert open_db.index("if (event.oldVersion < 1) {") < open_db.index(
        "createObjectStore("
    )
    assert "req.onblocked = () => resolve(null);" not in open_db
    blocked = open_db[open_db.index("req.onblocked = () => {") :]
    blocked = blocked[: blocked.index("\n      };")]
    assert "if (opening === attempt) opening = null;" in blocked
    assert "resolve(null);" in blocked
    assert "if (settled) { db.close(); return; }" in open_db
    assert "opening = attempt;" in open_db
    # A closed or superseded connection clears only its own attempt.
    assert (
        "db.onversionchange = () => { db.close(); if (opening === attempt) opening = null; };"
        in open_db
    )
    assert "db.onclose = () => { if (opening === attempt) opening = null; };" in open_db


def test_a_figure_the_storage_refuses_frees_the_oldest_and_retries_once() -> None:
    """A put that aborts with QuotaExceededError makes room, then retries once.

    The browser's storage for the page can fill before the store's own caps.
    The put's transaction reports the error it aborted with; on
    QuotaExceededError the least recently used figures (by the [at, bytes]
    index) are deleted until the refused figure's bytes are freed or half the
    store, never the figure's own key, and the put is tried once more. Any
    further failure resolves quietly.
    """
    store = _read("js", "figstore.js")
    put = _function(store, "export function putFigure(key, svg, caption) {")
    assert "}, '', abortName);" in put
    assert (
        "if (failed !== 'QuotaExceededError' || generation !== clears) return undefined;"
        in put
    )
    assert "freeRoom(bytes, key)" in put
    assert put.count("put()") == 2, "one put and one retry"
    assert put.rstrip().endswith("}).catch(() => undefined);\n}")
    free = _function(store, "function freeRoom(need, spare) {")
    assert "byAge(store, (entries, total) => {" in free
    assert "const target = Math.min(need, total / 2);" in free
    assert "if (freed >= target) break;" in free
    assert "if (key === spare) continue;" in free
    assert "store.delete(key);" in free
    assert "return (error && error.name) || 'AbortError';" in _function(
        store, "function abortName(error) {"
    )
    # A clear during the retry stores nothing after it.
    assert "clears += 1;" in _function(store, "export function clearFigures() {")


def test_a_live_figure_is_kept_and_cleared_with_the_history() -> None:
    """chat.js keeps each live figure and clears the figures with the history.

    The live run's events go through ``onEvent``, which also hands an event's
    figure_svg to putFigure under the result's figure_path, not awaited; the
    SVG still never goes into localStorage (storedChat strips it). "Clear chat
    history" (main.js) calls chat.js's clearChatHistory, which clears the
    figures too. Deleting one chat deletes the figures no other chat shows
    (test_deleting_a_chat_deletes_the_figures_no_other_chat_shows).
    """
    chat = _read("js", "chat.js")
    assert (
        "import { putFigure, clearFigures, deleteFigures } from './figstore.js';"
        in chat
    )
    assert (
        "const onEvent = (ev) => { aTurn.events.push(ev); "
        "if (epoch === historyEpoch) keepFigure(ev); };"
    ) in chat
    keep = _function(chat, "function keepFigure(ev) {")
    assert "inner.figure_path" in keep and "inner.figure_caption" in keep
    assert "void putFigure(path, ev.figure_svg, caption);" in keep
    assert "await" not in keep
    assert "delete copy.figure_svg;" in _function(chat, "function storedChat(chat) {")
    clear = _function(chat, "export function clearChatHistory() {")
    assert "localStorage.removeItem(CHATS_LS)" in clear
    assert "clearFigures()" in clear
    main = _read("js", "main.js")
    assert "clearChatHistory()" in main
    assert "localStorage.removeItem('oe_chats')" not in main


def test_a_run_streaming_through_a_clear_writes_nothing_back() -> None:
    """A run that streams while the history is cleared stores and saves nothing.

    clearChatHistory bumps a module-level history epoch; handleSend captures it
    before the run starts, and once it has changed the run's events keep no
    figure (keepFigure) and its end saves no chat (persistChat), so neither a
    figure nor the chat comes back after the clear. The send button is still
    released.
    """
    chat = _read("js", "chat.js")
    assert re.search(r"\nlet historyEpoch = 0;", chat)
    clear = _function(chat, "export function clearChatHistory() {")
    assert "historyEpoch += 1;" in clear
    assert clear.index("historyEpoch += 1;") < clear.index("clearFigures()")
    send = _function(chat, "function handleSend(brief, attachments) {")
    assert "const epoch = historyEpoch;" in send
    assert send.index("const epoch = historyEpoch;") < send.index("runLive(")
    assert send.index("const epoch = historyEpoch;") < send.index("runDemo(")
    assert "if (epoch === historyEpoch) keepFigure(ev);" in send
    done = send[send.index("const done = () => {") :]
    done = done[: done.index("\n  };")]
    assert "sending = false; setSendingUI(false);" in done
    assert "if (epoch !== historyEpoch) return;" in done
    assert done.index("setSendingUI(false)") < done.index(
        "if (epoch !== historyEpoch) return;"
    )
    assert done.index("if (epoch !== historyEpoch) return;") < done.index(
        "persistChat(chat)"
    )


def test_deleting_a_chat_deletes_the_figures_no_other_chat_shows() -> None:
    """Deleting a chat deletes its figures that no remaining chat names.

    figstore.js's deleteFigures deletes figures by path; deleteChat collects
    the deleted chat's figure_path values (figurePaths) and deletes those that
    no remaining saved chat, nor the open chat, names, and only once the chats
    were saved without it.
    """
    store = _read("js", "figstore.js")
    delete_figures = _function(store, "export function deleteFigures(keys) {")
    assert "inStore('readwrite'," in delete_figures
    assert "for (const key of list) store.delete(key);" in delete_figures
    chat = _read("js", "chat.js")
    paths = _function(chat, "function figurePaths(chat) {")
    assert "const inner = ev && ev.result && ev.result.result;" in paths
    assert "paths.add(inner.figure_path)" in paths
    delete = _function(chat, "function deleteChat(id) {")
    assert "const left = chats.filter((c) => c.id !== id);" in delete
    assert "if (saveChats(left) && gone) {" in delete
    assert (
        "for (const c of left) figurePaths(c).forEach((p) => shown.add(p));" in delete
    )
    assert (
        "if (activeChat && activeChat.id !== id) "
        "figurePaths(activeChat).forEach((p) => shown.add(p));"
    ) in delete
    assert (
        "const orphans = [...figurePaths(gone)].filter((p) => !shown.has(p));" in delete
    )
    assert "void deleteFigures(orphans);" in delete
    save = _function(chat, "function saveChats(chats) {")
    assert save.count("return true;") == 2 and save.count("return false;") == 1


def test_a_reopened_chat_looks_its_figure_up() -> None:
    """viz.js shows a replayed figure from the store, built without innerHTML.

    A replayed result has figure_path but no figure_svg: renderFigure draws the
    caption and "Figure saved at" at once, then restoreFigure looks the path
    up (getFigure) and, found under the same caption, puts the frame a live
    figure gets (the <img> with the data: URL, linked through figurePageUrl)
    above the caption and the download after it, and removes the "Figure
    saved at" line. A copy stored under another caption (another
    determination's figure written to the same path) is not drawn, and the
    line stays. The figure code builds its elements with createElement and
    textContent.
    """
    viz = _read("js", "viz.js")
    assert "import { getFigure } from './figstore.js';" in viz
    restore = _function(viz, "async function restoreFigure(fig, where, path, caption)")
    assert "await getFigure(path)" in restore
    other = "if (caption && stored.caption && stored.caption !== caption) return;"
    assert other in restore
    assert restore.index("await getFigure(path)") < restore.index(other)
    assert restore.index(other) < restore.index("fig.insertBefore(")
    assert restore.index(other) < restore.index("where.remove();")
    assert "figureFrame(() => figurePageUrl(svg), svgDataUrl(svg)" in restore
    assert "fig.firstChild" in restore
    assert "where.remove();" in restore
    assert "figureDownload(path, () => svg)" in restore
    render = _function(viz, "function renderFigure(container, ev, inner) {")
    assert "figureFrame(() => figurePageUrl(svg), svgDataUrl(svg), caption)" in render
    assert "'Figure saved at ' + path" in render
    assert "void restoreFigure(fig, where, path, caption);" in render
    start = viz.index("function figureFrame(")
    figure_code = viz[start : viz.index("export function renderResultViz(")]
    assert "innerHTML" not in figure_code
    assert "insertAdjacentHTML" not in figure_code
    assert "img.src = src;" in figure_code
    assert "cap.textContent = caption;" in figure_code


def test_a_figure_makes_its_page_url_on_the_first_click() -> None:
    """A figure's full-size page (a blob: URL) is made when first opened.

    Drawing a figure (live or reopened) passes figureFrame a function that
    makes the page, not the page's URL, so reopening chats does not pile up
    blob: URLs; figureFrame makes it on the link's first press or click and
    keeps it for that figure. A demo figure links its shipped file as before.
    """
    viz = _read("js", "viz.js")
    frame = _function(viz, "function figureFrame(page, src, caption) {")
    assert "if (typeof page === 'function') {" in frame
    assert "let url = '';" in frame
    assert "if (!url) { url = page(); link.href = url; }" in frame
    assert "link.addEventListener('click', open);" in frame
    assert "link.addEventListener('pointerdown', open);" in frame
    assert "link.href = page;" in frame
    assert "createObjectURL" not in frame
    start = viz.index("function figureFrame(")
    figure_code = viz[start : viz.index("export function renderResultViz(")]
    # Every page URL is made lazily: once in renderFigure, once in restoreFigure.
    assert figure_code.count("figurePageUrl(svg)") == 2
    assert figure_code.count("() => figurePageUrl(svg)") == 2
    render = _function(viz, "function renderFigure(container, ev, inner) {")
    assert "figureFrame(demo, demo, caption)" in render
    page_url = _function(viz, "function figurePageUrl(svg) {")
    assert "URL.createObjectURL(" in page_url
    assert "createObjectURL" not in figure_code
