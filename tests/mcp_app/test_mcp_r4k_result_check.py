# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Round r4-k (16 Nov 2026), the result card's and the Check card's part, each
string as specs/r4-k-round.md gives it; the view files are read as text (no
test here runs the scripts):

- K2: the result and check pages carry the watchdog, a classic inline script
  before the module, that says the card could not start when the module has
  not started after 10 s (window.OE_STARTED, set by common.js's OE.start);
- K6: the rate line shows the 95% range's two ends as numbers under its band
  (the server's low_text and high_text, passed by result.html), one centred
  "a to b" label where the band is too narrow for two, and leaves out the 0%
  or 100% label where an end's label would touch it (the owner's choice of
  16 Nov 2026, which reverses r4-h R1);
- K7: a Street map button on each look-first card, and a row of them for
  points without picture cards in place of the numbers' table; a press asks
  the server by the review's key and the window's rank
  (olmoearth_app_review_link) and opens only an OpenStreetMap link, which
  never enters the page's text, a toast or what Claude is told;
- K8: one outlined button saves the look-first list with places as a CSV
  file: through app.downloadFile where OE.canDownload says the host takes
  downloads, else written by the server (save true) and named under the
  button; the file's text never enters the page or what Claude is told;
- K9: the Check card shows the server's two example pictures beside the
  owner's rule, with the label sheet's captions, and their place line in
  Details, only while the card shows the owner's rule.

A JS concatenation of single-quoted strings and names is read back into its
text (:func:`_concat`), so a missing space between two joined pieces fails
here as it would on the card."""

from __future__ import annotations

import re
from pathlib import Path

from olmoearth_agent.mcp_app import sheet, views

#: One token of a JS concatenation: a single-quoted string, a plus, or a name
#: (a call with no parentheses inside its own is one name).
_TOKEN = re.compile(
    r"\s*(?:'((?:[^'\\]|\\.)*)'|(\+)|([A-Za-z_][A-Za-z0-9_.]*(?:\([^()]*\))?))"
)

#: The watchdog's one statement (r4-k K2), the same in every page.
WATCHDOG = (
    "setTimeout(function(){if(!window.OE_STARTED){var r=document.getElementById("
    "'root');if(r)r.textContent='This card could not start here. Carry on in the "
    "chat.';}},10000);"
)
#: What it says.
WATCHDOG_TEXT = "This card could not start here. Carry on in the chat."

#: The look-first card's new words (r4-k K7, K8): (item, the entry's start in
#: reviewWords, its names, the text it makes).
REVIEW_ENTRIES: tuple[tuple[str, str, dict[str, str], str], ...] = (
    ("K7", "streetMap: ", {}, "Street map"),
    (
        "K7",
        "mapOf: (k) => ",
        {"one": "point", "k": "3"},
        "Street map: point 3 on OpenStreetMap",
    ),
    ("K7", "noLink: (k) => ", {"one": "point", "k": "3"}, "No map link for point 3."),
    (
        "K7",
        "notOpened: (why) => ",
        {"why": "it failed"},
        "The map was not opened: it failed.",
    ),
    ("K7", "linkNotMade: ", {}, "the link could not be made"),
    ("K7", "mapsLabel: ", {}, "Street map:"),
    ("K8", "csv: ", {}, "Save the list with places (CSV file)"),
    (
        "K8",
        "csvSaved: (file) => ",
        {"file": "f.csv"},
        "Saved on this computer as f.csv.",
    ),
    (
        "K8",
        "csvNotSaved: (why) => ",
        {"why": "it failed"},
        "The list was not saved: it failed.",
    ),
    ("K8", "listNotMade: ", {}, "the list could not be made"),
    ("K8", "notAllowed: ", {}, "The download was not allowed."),
    ("K8", "downloadFailed: ", {}, "the download failed"),
)

#: The example pictures' captions and their place line (r4-k K9).
EXAMPLE_PRESENT = "Present under this rule"
EXAMPLE_ABSENT = "Absent under this rule"
EXAMPLES_FROM = (
    "A public place, not your map. Contains modified Copernicus Sentinel data "
    "{year}."
)

#: A review's key as the server makes it (24 hex characters).
KEY = "0123456789abcdef01234567"

#: Words no face holds (r4-i's rule).
_OFF_FACE = re.compile(r"\b(?:windows?|budget|stratum|design|review set)\b", re.I)


def _read(name: str) -> str:
    """A file of views/, as it is on disk."""
    return (views.VIEW_DIR / name).read_text(encoding="utf-8")


def _between(text: str, start: str, end: str) -> str:
    """The part of text from start up to the first end after it."""
    at = text.index(start)
    return text[at : text.index(end, at)]


def _after(text: str, start: str, end: str) -> str:
    """The part of text after start up to the first end after it."""
    at = text.index(start) + len(start)
    return text[at : text.index(end, at)]


def _entry(block: str, start: str) -> str:
    """One entry's value in a JS object literal: from start to the comma that
    ends its last line."""
    at = block.index(start) + len(start)
    end = block.find(",\n", at)
    return (block[at:] if end < 0 else block[at:end]).strip().rstrip(",")


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


def _code_lines(text: str) -> list[str]:
    """The lines of text that are code, stripped: /* */ blocks and lines that
    are only a // comment left out (a // inside a string, such as a URL, is
    kept)."""
    bare = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    lines = [line.strip() for line in bare.splitlines()]
    return [line for line in lines if line and not line.startswith("//")]


def _in_order(text: str, *pieces: str) -> bool:
    """Whether each piece is in text, each after the one before it."""
    at = [text.index(p) for p in pieces]
    return at == sorted(at)


# --------------------------------------------------------------------------- K2


def test_k2_the_result_and_check_pages_carry_the_watchdog() -> None:
    """r4-k K2: each page has one classic inline script, marked, holding only
    the watchdog's statement, after the root and before the module script;
    the statement reads window.OE_STARTED (which common.js sets) and never
    sets it; its line promises nothing of Claude. The served page keeps it
    whole (views.view_html)."""
    assert WATCHDOG_TEXT in WATCHDOG
    for word in ("list", "square", "saved", "Claude"):
        assert word not in WATCHDOG_TEXT, word
    for name, view in (("result.html", "result"), ("check.html", "check")):
        page = _read(name)
        script = _between(page, "<script>\n", "</script>")
        assert script == "<script>\n// WORDING: r4k K2\n" + WATCHDOG + "\n", name
        assert page.count(WATCHDOG) == 1, name
        assert page.count("<script>") == 1, name
        assert _in_order(
            page,
            '<main id="root" class="stack">',
            "<script>\n// WORDING: r4k K2\n",
            '<script type="module">',
        ), name
        assert "OE_STARTED = " not in page and "OE_STARTED=" not in page, name
        assert WATCHDOG in views.view_html(view), view


# --------------------------------------------------------------------------- K6


#: figures.js's rules for the labels under the rate line, as written.
RATE_GEOMETRY = (
    "  const RL = { w: 320, h: 50, left: 16, right: 304, y: 24 };\n"
    "  const xOf = (v) => RL.left + v * (RL.right - RL.left);\n"
)
UNDER_RULES = (
    "  const UNDER_CH = 7;\n"
    "  const UNDER_GAP = 4;\n"
    "  const underAt = (x, text) => {\n"
    "    const half = (text.length * UNDER_CH) / 2;\n"
    "    const cx = Math.min(Math.max(x, half), RL.w - half);\n"
    "    return { text, cx, left: cx - half, right: cx + half };\n"
    "  };\n"
)
APART_RULE = (
    "  const apart = (a, b) => a.right + UNDER_GAP <= b.left"
    " || b.right + UNDER_GAP <= a.left;"
)
ENDS_RULE = (
    "      // WORDING: r4k K6\n"
    "      ends = apart(a, b) ? [a, b] : [underAt((xOf(low) + xOf(high)) / 2,"
    " rangeText || lowText + ' to ' + highText)];"
)
TICK_RULE = (
    "    const tick = (x, text) => (ends.every((e) => apart(e, underAt(x, text)))\n"
    "      ? svg('text', { class: 'tick-label', x, y: 44, 'text-anchor': 'middle' },"
    " text) : null);"
)
END_LABEL = (
    "marks.push(svg('text', { class: 'end-label', x: round2(e.cx), y: 44,"
    " 'text-anchor': 'middle' }, e.text));"
)

_Label = tuple[str, float, float]


def _under(x: float, text: str) -> _Label:
    """UNDER_RULES's label under x: its text and its left and right edges."""
    half = len(text) * 7 / 2
    cx = min(max(x, half), 320 - half)
    return (text, cx - half, cx + half)


def _apart(a: _Label, b: _Label) -> bool:
    """APART_RULE: the two labels keep at least 4 apart."""
    return a[2] + 4 <= b[1] or b[2] + 4 <= a[1]


def _x_of(v: float) -> float:
    """RATE_GEOMETRY's x of a fraction."""
    return 16 + v * (304 - 16)


def _labels(
    low: float, high: float, low_text: str, high_text: str, range_text: str
) -> tuple[list[str], list[str]]:
    """The labels rateLine draws under the line by the rules above: the
    ends' (two, or one centred), and the 0% and 100% labels it keeps."""
    ends: list[_Label] = []
    if low_text and high_text:
        a, b = _under(_x_of(low), low_text), _under(_x_of(high), high_text)
        mid = (_x_of(low) + _x_of(high)) / 2
        ends = [a, b] if _apart(a, b) else [_under(mid, range_text)]
    ticks = [
        text
        for x, text in ((16.0, "0%"), (304.0, "100%"))
        if all(_apart(e, _under(x, text)) for e in ends)
    ]
    return [e[0] for e in ends], ticks


def test_k6_the_rate_line_draws_the_ranges_ends_under_its_band() -> None:
    """r4-k K6: rateLine reads lowText and highText (the server's), with or
    without a rate (Cannot tell keeps the bar's label and gets the band's
    ends as well), each under its end as an end-label, or one label centred
    under the band (the server's rangeText) where the two would touch; the
    0% and 100% labels go only where an end's label would touch them; no
    "range:" words; the aria-label as before."""
    script = _read("figures.js")
    for text in (RATE_GEOMETRY, UNDER_RULES, APART_RULE):
        assert text in script, text
    assert ".oe-fig .end-label { font-size: 11px; font-weight: 600; }" in script
    rate = _between(script, "  const rateLine = (", "\n  };")
    for text in (
        "const lowText = ranged ? words(o.lowText) : '';",
        "const highText = ranged ? words(o.highText) : '';",
        "if (lowText && highText) {",
        "const a = underAt(xOf(low), lowText);",
        "const b = underAt(xOf(high), highText);",
        ENDS_RULE,
        TICK_RULE,
        "      tick(RL.left, '0%'),\n      tick(RL.right, '100%'),\n",
        END_LABEL,
    ):
        assert text in rate, text
    # The ends do not depend on a rate: Cannot tell keeps them (and the bar's
    # own label, rateText, as before).
    ends = _between(rate, "const lowText", "const marks = [")
    for word in ("barred", "rate !== null", "rateText"):
        assert word not in ends, word
    assert "if (rateText) marks.push(labelAt(mid, rateText));" in rate
    # The ends are drawn after the band, so they are never under it.
    assert rate.index("class: 'band'") < rate.index("class: 'end-label'")
    # The old shapes the earlier rounds pinned still hold.
    assert "range: " not in rate and "'range" not in rate
    assert "role: 'img', 'aria-label': label" in rate
    assert "    // WORDING: r4h R1\n    const said = rangeText ?" in rate


def test_k6_the_end_labels_at_0_100_and_a_narrow_band() -> None:
    """r4-k K6, by the rules pinned above: at a low end of 0% the 0% label
    goes and the two ends stay apart; at a high end of 100% the 100% label
    goes; a band one point wide gets one centred "a to b" label (the
    server's range text), which also clears the 0% label near it; a band in
    the middle keeps both end labels and both ticks; without the texts,
    nothing under the band."""
    assert _labels(0.0, 0.2, "0.0%", "20.0%", "0.0% to 20.0%") == (
        ["0.0%", "20.0%"],
        ["100%"],
    )
    assert _labels(0.8, 1.0, "80.0%", "100.0%", "80.0% to 100.0%") == (
        ["80.0%", "100.0%"],
        ["0%"],
    )
    assert _labels(0.10, 0.11, "10.0%", "11.0%", "10.0% to 11.0%") == (
        ["10.0% to 11.0%"],
        ["100%"],
    )
    assert _labels(0.3, 0.6, "30.0%", "60.0%", "30.0% to 60.0%") == (
        ["30.0%", "60.0%"],
        ["0%", "100%"],
    )
    assert _labels(0.3, 0.6, "", "", "30.0% to 60.0%") == ([], ["0%", "100%"])
    # Every label stays inside the 320-wide figure.
    for x, text in ((0.0, "0.0% to 1.0%"), (320.0, "99.0% to 100.0%")):
        _text, left, right = _under(x, text)
        assert 0 <= left and right <= 320, text


def test_k6_the_figures_comment_no_longer_cites_r1_as_in_force() -> None:
    """r4-k K6: the figures' head comment says the line now labels the
    band's ends (the owner's choice, reversing r4-h R1), not "No words under
    the line"."""
    script = _read("figures.js")
    head = script[: script.index("OE.fig = (() => {")]
    assert "No words under the line" not in head
    flat = " ".join(head.split())
    assert "lowText, highText}" in flat
    assert "r4-k K6: the owner's choice of 16 Nov 2026, which reverses r4-h R1" in flat


def test_k6_the_result_card_passes_the_servers_end_texts() -> None:
    """r4-k K6: rateBlock passes the server's low_text and high_text as
    lowText and highText, under the earlier pinned line; the page computes
    nothing."""
    page = _read("result.html")
    rate = _between(page, "function rateBlock(r) {", "\n}\n")
    passed = (
        "      rateText: rate !== null ? text(r.rate_text) : text(r.rate_range_text),"
        " rangeText: text(r.range_text),\n"
        "      // The band's two ends as numbers under it (r4-k K6): the server's"
        " texts.\n"
        "      lowText: text(r.low_text), highText: text(r.high_text),\n"
        "    })"
    )
    assert passed in rate
    for word in ("toFixed", "Math.", "* 100"):
        assert word not in rate, word


# --------------------------------------------------------------------------- K7, K8


def test_k7_k8_the_look_first_cards_words() -> None:
    """r4-k K7 and K8: each new word of the look-first card is marked and
    reads as the spec gives it; the Street map words are the label sheet's,
    word for word (sheet.js streetMap, openMapOf, noLink); none holds a word
    kept off faces."""
    page = _read("result.html")
    words = _between(page, "const reviewWords = (unit) => {", "\n};")
    for item, start, names, said in REVIEW_ENTRIES:
        assert f"    // WORDING: r4k {item}\n    {start}" in words, start
        assert _concat(_entry(words, start), names) == said, start
        assert _OFF_FACE.search(said) is None, said
    script = _read("sheet.js")
    sheet_words = _between(script, "OE.sheetWords = (unit) => {", "\n};")
    for one in ("square", "point"):
        names = {"one": one, "k": "3"}
        for ours, theirs in (
            ("mapOf: (k) => ", "openMapOf: (k) => "),
            ("noLink: (k) => ", "noLink: (k) => "),
            ("streetMap: ", "streetMap: "),
        ):
            mine = _concat(_entry(words, ours), names)
            assert mine == _concat(_entry(sheet_words, theirs), names), ours


def test_k7_a_street_map_press_asks_by_key_and_rank_and_shows_no_place() -> None:
    """r4-k K7: the key (24 hex) names the review for the links whether or
    not its windows have pictures; a press asks olmoearth_app_review_link by
    the key and the rank, busy while asked, and opens only an OpenStreetMap
    link (the server's) with OE.open; the link is in two lines only, never
    in the page's text, a toast or what Claude is told."""
    page = _read("result.html")
    assert "const OSM_LINK = 'https://www.openstreetmap.org/';" in page
    # The server's links start so (sheet.window_link's, and the helper the
    # review links share with it, wherever in the package it lives).
    package = Path(sheet.__file__).parent
    sources = [p.read_text(encoding="utf-8") for p in sorted(package.glob("*.py"))]
    assert any("https://www.openstreetmap.org/?mlat=" in s for s in sources)
    block = _between(page, "function reviewBlock(app, block) {", "\n}\n")
    for text in (
        "const linkKey = typeof block.key === 'string'"
        " && /^[0-9a-f]{24}$/.test(block.key) ? block.key : null;",
        "const key = block.pictures === true ? linkKey : null;",
        "if (linking.has(rank)) return;",
        "const got = await OE.call(app, 'olmoearth_app_review_link',"
        " { review_key: linkKey, rank });",
        "else OE.toast(REVIEW_WORDS.noLink(rank), true);",
        "OE.toast(REVIEW_WORDS.notOpened(bareText(err.message)"
        " || REVIEW_WORDS.linkNotMade), true);",
        "type: 'button', class: 'link', 'data-focus': 'map:' + rank,"
        " 'aria-label': REVIEW_WORDS.mapOf(rank),",
        "'aria-disabled': String(linking.has(rank)), onclick: () => openMap(rank),",
    ):
        assert text in block, text
    assert re.fullmatch(r"[0-9a-f]{24}", KEY) is not None
    with_url = [
        line for line in _code_lines(block) if re.search(r"\burl\b", line) is not None
    ]
    assert with_url == [
        "const url = got && typeof got.url === 'string' ? got.url : '';",
        "if (url.startsWith(OSM_LINK)) await OE.open(app, url);",
    ]
    assert "tellClaude" not in block
    for place in ("lon", "lat", "centre", "center"):
        assert re.search(rf"\b{place}\b", block) is None, place


def test_k7_each_card_has_its_street_map_and_points_get_a_row() -> None:
    """r4-k K7: each picture card sits in a cell with its Street map button
    under it (never inside the card's own button); points without picture
    cards whose record names their places get a row of Street map buttons,
    each its rank, in rank order, in place of the numbers' table; with no
    key at all, the numbers' table as before."""
    page = _read("result.html")
    assert (
        ".review-strip .window-cell { display: flex; flex-direction: column;"
        " gap: 2px; min-width: 0; }" in page
    )
    block = _between(page, "function reviewBlock(app, block) {", "\n}\n")
    cards = _between(block, "  function cards() {", "\n  }\n")
    for text in (
        "OE.el('div', { class: 'window-cell' },",
        "type: 'button', class: 'window-card', 'data-focus': 'card:' + w.rank,",
        "OE.el('span', { class: 'caption', 'aria-hidden': 'true' }, w.caption || '')),",
        "mapButton(w.rank, REVIEW_WORDS.streetMap))));",
    ):
        assert text in cards, text
    # The map button follows the card's button, closed before it.
    assert cards.index("w.caption || '')),") < cards.index("mapButton(")
    row = _between(block, "    if (n && linkKey) {", "    return section;")
    for text in (
        "REVIEW_WORDS.firstShown(n, total)",
        "OE.el('div', { class: 'row review-links', role: 'group',"
        " 'aria-label': REVIEW_WORDS.streetMap },",
        "OE.el('span', { class: 'small', 'aria-hidden': 'true' },"
        " REVIEW_WORDS.mapsLabel),",
        "windows.map((w) => mapButton(w.rank, String(w.rank)))), csvBlock());",
        "    } else if (n) section.append(numbersFold(windows, unit));\n",
    ):
        assert text in row, text
    assert row.index("csvBlock()") < row.index("numbersFold(")
    # The numbers' table: with no key, and under the picture cards, as before.
    assert block.count("numbersFold(windows, unit)") == 2
    assert (
        "section.append(strip, more, pane, csvBlock(), numbersFold(windows, unit));"
        in block
    )
    assert "primary" not in block


def test_k8_the_csv_button_and_its_two_paths() -> None:
    """r4-k K8: one outlined button; with OE.canDownload the server's text
    (save false) goes to app.downloadFile as a text/csv file named by the
    server (a name only, else a fixed one), a refusal toasts, and the host's
    own error is never shown; without it the server writes the file (save
    true) and the line under the button names it. The text is in three code
    lines only, never in the page, a toast or what Claude is told."""
    page = _read("result.html")
    assert (
        r"const csvName = (v) => (typeof v === 'string'"
        r" && /^[A-Za-z0-9_.-]{1,200}\.csv$/.test(v) ? v : null);" in page
    )
    named = r"[A-Za-z0-9_.-]{1,200}\.csv"
    assert re.fullmatch(named, f"look_first_not_a_sample_{KEY}.csv") is not None
    assert re.fullmatch(named, "review_sets/look_first.csv") is None
    block = _between(page, "function reviewBlock(app, block) {", "\n}\n")
    csv = _between(block, "  function csvBlock() {", "\n  }\n")
    assert _in_order(
        csv,
        "if (typeof OE.canDownload === 'function' && OE.canDownload(app)) {",
        "OE.call(app, 'olmoearth_app_review_csv',"
        " { review_key: linkKey, save: false });",
        "const name = csvName(got.name) || 'look_first_not_a_sample.csv';",
        "answer = await app.downloadFile({ contents: [{ type: 'resource', resource:"
        " { uri: 'file:///' + name, mimeType: 'text/csv', text } }] });",
        "throw new Error(REVIEW_WORDS.downloadFailed);",
        "if (answer && answer.isError) OE.toast(REVIEW_WORDS.notAllowed, true);",
        "} else {",
        "OE.call(app, 'olmoearth_app_review_csv',"
        " { review_key: linkKey, save: true });",
        "const file = csvName(got && got.file);",
        "said.textContent = REVIEW_WORDS.csvSaved(file);",
        "OE.toast(REVIEW_WORDS.csvNotSaved(bareText(err.message)"
        " || REVIEW_WORDS.listNotMade), true);",
    )
    assert "    // WORDING: r4k K8 (the server's file name, else this one)\n" in csv
    # The host's error in a download is replaced, never shown.
    assert "} catch (_) {\n              throw new Error(" in csv
    with_text = [
        line for line in _code_lines(csv) if re.search(r"\btext\b", line) is not None
    ]
    assert with_text == [
        "const text = got && typeof got.text === 'string' ? got.text : '';",
        "if (!text) throw new Error(REVIEW_WORDS.listNotMade);",
        "answer = await app.downloadFile({ contents: [{ type: 'resource', resource:"
        " { uri: 'file:///' + name, mimeType: 'text/csv', text } }] });",
    ]
    said = [line for line in _code_lines(csv) if "said." in line]
    assert said == ["said.textContent = REVIEW_WORDS.csvSaved(file);"]
    assert "primary" not in csv and "tellClaude" not in csv
    assert (
        "return OE.el('div', { class: 'stack review-csv' },"
        " OE.el('div', { class: 'row' }, button), said);" in csv
    )
    # Drawn under the picture cards and under the points' row: one definition,
    # two calls, both where a key names the review.
    assert block.count("csvBlock()") == 3


# --------------------------------------------------------------------------- K9


def test_k9_the_check_card_shows_the_example_pictures_beside_the_owners_rule() -> None:
    """r4-k K9: the captions and the place line are marked and read as the
    spec gives them; the server's examples are taken only as two PNG data
    URLs and a year; they show under the owner's rule (read only, or before
    the box for the map's own rule), never beside a typed rule nor after a
    press with no reply, and their place line goes in Details then only."""
    page = _read("check.html")
    assert f"// WORDING: r4k K9\nconst EXAMPLE_PRESENT = '{EXAMPLE_PRESENT}';" in page
    assert f"// WORDING: r4k K9\nconst EXAMPLE_ABSENT = '{EXAMPLE_ABSENT}';" in page
    line = _after(page, "// WORDING: r4k K9\nconst EXAMPLES_FROM = (year) => ", ";\n")
    assert _concat(line, {"year": "2024"}) == EXAMPLES_FROM.format(year=2024)
    for said in (EXAMPLE_PRESENT, EXAMPLE_ABSENT, EXAMPLES_FROM):
        assert _OFF_FACE.search(said) is None, said
    taken = _between(page, "function examplesOf(e) {", "\n}\n")
    for text in (
        "if (!e || typeof e !== 'object') return null;",
        "const year = Number.isInteger(e.year) && e.year >= 2014 && e.year <= 2100"
        " ? e.year : null;",
        "if (!EXAMPLE_PNG.test(e.present) || !EXAMPLE_PNG.test(e.absent)) return null;",
        "return { present: e.present, absent: e.absent, year };",
    ):
        assert text in taken, text
    png = r"const EXAMPLE_PNG = /^data:image\/png;base64,[A-Za-z0-9+/]+={0,2}$/;"
    assert png in page
    # The page's own source holds no PNG data URL (the rule writes its slash
    # escaped), so the served page keeps one only: the AGENT tag.
    assert "data:image/png;base64," not in page
    shown = _between(page, "function examplesBlock(ex) {", "\n}\n")
    for text in (
        "OE.el('img', { src, alt: caption }),",
        "OE.el('figcaption', { class: 'small', 'aria-hidden': 'true' }, caption));",
        "one(ex.present, EXAMPLE_PRESENT), one(ex.absent, EXAMPLE_ABSENT)",
    ):
        assert text in shown, text
    assert ".oe-examples { display: flex; flex-wrap: wrap; gap: 12px; }" in page
    card = _between(page, "function checkCard(", "\n}\n")
    for text in (
        "const examples = examplesOf(data.examples);",
        "const examplesShown = () => Boolean(examples) && !ruleLost && rule.set"
        " && rule.source === 'owner'\n    && Boolean(rule.text);",
    ):
        assert text in card, text
    gate = _between(page, "  function fillGate() {", "\n  }\n")
    lines = _between(gate, "const ruleLines = () => [", "].filter(Boolean);")
    assert _in_order(
        lines,
        "RULE_SHOWN(rule.text)",
        "rule.scope ?",
        "examplesShown() ? examplesBlock(examples) : null,",
    )
    assert _in_order(
        gate,
        "} else if (mode === 'replace') {",
        "gate.replaceChildren(...ruleLines(),",
        "RULE_OWN), ruleInput));",
        "} else if (mode === 'shown' || mode === 'owner') {",
        "gate.replaceChildren(...ruleLines());",
        "} else if (mode === 'lost') {",
    )
    fold = _between(page, "  function fillFold() {", "\n  }\n")
    assert _in_order(
        fold,
        "if (!ruleLost && rule.source !== 'owner') nodes.push(OE.el('p', {},"
        " RULE_PICTURES));",
        "if (examplesShown()) nodes.push(OE.el('p', {},"
        " EXAMPLES_FROM(examples.year)));",
    )
    code = re.sub(r"/\*.*?\*/|//[^\n]*", "", page, flags=re.S)
    # Defined once and used once each: in the gate, and in Details.
    assert code.count("examplesBlock(") == 2 and code.count("EXAMPLES_FROM") == 2


def test_k9_the_data_url_rule_takes_pngs_only() -> None:
    """r4-k K9: the rule examplesOf reads the pictures by (EXAMPLE_PNG, here
    in Python's words) takes a base64 PNG data URL and nothing else: no other
    image type, no link, no markup."""
    png = re.compile(r"^data:image/png;base64,[A-Za-z0-9+/]+={0,2}$")
    assert png.fullmatch("data:image/png;base64,iVBORw0KGgoAAAANSUhEUg==")
    for bad in (
        "data:image/svg+xml;base64,PHN2Zz4=",
        "https://example.org/present.png",
        'data:image/png;base64,iVBOR" onerror="x',
        "data:image/png;base64,",
    ):
        assert png.fullmatch(bad) is None, bad


def test_k9_the_sheet_shows_the_same_captions_and_place_line() -> None:
    """r4-k K9: the label sheet (sheet.js, built beside this card) carries
    the same two captions and the same place line, word for word, so the
    two cards never word the example pictures differently."""
    script = _read("sheet.js")
    joined = re.sub(r"'\s*\+\s*'", "", script)
    for text in (
        EXAMPLE_PRESENT,
        EXAMPLE_ABSENT,
        "A public place, not your map. Contains modified Copernicus Sentinel data ",
    ):
        assert text in joined, text
