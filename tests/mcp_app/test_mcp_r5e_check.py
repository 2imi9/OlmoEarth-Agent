# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Round r5-e (17 Nov 2026), track CHECK, on the MCP App: the Check card's
items of the round's design list (specs/r5-e-design/build_list.md, K1 to
K8), under its seven design rules.

- K1: a part's counts are one grid of equal buttons, each its number alone;
  one line under the question says what more squares give, once;
- K2: no parts figure on the Check card, and no line under an open sheet:
  each part of a map in parts is one white card holding its name and its
  counts or its state, said once;
- K3: why a map is in parts is the docs' picture of the world cut into
  strips (a JPEG of views/brand, in the Check card alone), with its one line
  beside it in the same card, where each part is a strip of its own; any
  other map in parts says only how many parts it has, with no picture;
- K4: the line over the press is one line at 780 px; the come-back line is
  in Details, while the card has a press and shows no part as picked;
- K5: the labelling rule, its scope, its example pictures and its box are
  one white card;
- K6: the signed-out card's note is the panel's two lines;
- K7: a class map's face is two lines; what the check gives is in Details;
- K8: a failed pick says "The pick failed." once on its part, the reason in
  Details, and whether this press or a later one tries it again; after a
  press with no reply the gate says once what to do, and takes the focus.

The views are read as text (no test here runs the scripts); a JS
concatenation of single-quoted strings and names is read back into its text
(:func:`_concat`). The picture is read as bytes.
"""

from __future__ import annotations

import re

import pytest

pytest.importorskip("mcp")

from olmoearth_agent.mcp_app import check, server, views  # noqa: E402

#: One token of a JS concatenation: a single-quoted string, a plus, or a name
#: (a call with no parentheses inside its own is one name).
_TOKEN = re.compile(
    r"\s*(?:'((?:[^'\\]|\\.)*)'|(\+)|([A-Za-z_][A-Za-z0-9_.]*(?:\([^()]*\))?))"
)

#: Words no face holds (the round's rule R7, and the owner's ruling 6).
_OFF_FACE = re.compile(
    r"\b(?:windows?|UTM|zones?|intervals?|strat(?:um|a)|design|budget"
    r"|recommended)\b",
    re.I,
)

#: K1: the one line under the question.
MORE_WORK = "More squares: more work, and a narrower range."
#: K3: the line beside the strips picture, for a map in 2 parts.
PARTS_LINE_2 = (
    "Studio saves a map in one part for each strip of the world it covers, like "
    "the pink strip here. This map has 2 parts; each gets its own check."
)
#: K3: the line of a map in parts that are not one per strip, for 4 parts.
PARTS_PLAIN_4 = "This map is saved in 4 parts. Each part gets its own check."
#: K4: the line over the press, and the come-back line (now in Details).
CANNOT = (
    "Once you press, your rule and number are fixed. You get a result once every "
    "square is answered."
)
PICKED_BEFORE = "Picked before? Press again: the same squares open."
#: K7: a class map's two face lines, and what the check gives (Details).
CLASS_MAP = (
    "This map can be checked on points picked at random, instead of squares.",
    "The app shows no pictures of points: each needs your own imagery or records "
    "from the map's dates.",
)
POINTS_GRID = (
    "You get how often the map is wrong over a grid of points spread across the "
    "map, with a range, not for every pixel."
)
#: K8: a part's short line, the retry's (this press, or a later one when more
#: picks failed than one press takes), the other parts' after a press with no
#: reply, the part pressed then, and the toast. The retry's two lines in round
#: r5-f F11's words: which press, and whose (they read "The press tries again
#: with 80 squares." and "A later press tries again with 80 squares."; the
#: one that waits says it is not in this press, as the line under the press
#: says "this press"; tests/mcp_app/test_mcp_r5f_check.py).
FAILED_SHORT = "The pick failed."
RETRY_80 = "Press Pick the squares to try again with 80 squares."
RETRY_LATER_80 = (
    "Not in this press: it waits for a later press of Pick the squares " "(80 squares)."
)
#: K8: the press's toast when a part was not picked (it promises no reason on
#: a part's face: a failed pick's reason is in Details).
NOT_ALL_PICKED = "Not every part's squares were picked; each part says so."
NOT_YET = "Not picked yet."
NO_ANSWER = "No answer came back: this part may or may not be picked."
PRESS_NO_ANSWER = "The press got no answer back."
#: The gate's line then (r5-b J09), unchanged: the one place that says what to do.
RULE_LOST = (
    "This card can pick no more squares. Ask Claude to open the Check card "
    "again: it shows the rule and what was picked."
)

#: The strips picture's file, its size (2x of 360 x 164 on the face) and its
#: largest size in bytes.
STRIPS_FILE = "world-strips.jpg"
STRIPS_SIZE = (720, 328)
STRIPS_MAX_BYTES = 120 * 1024

#: JPEG markers that start a frame (they carry its height and width): C0 to
#: CF but for DHT (C4), JPG (C8) and DAC (CC).
_SOF = set(range(0xC0, 0xD0)) - {0xC4, 0xC8, 0xCC}


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


def _concat(expr: str, names: dict[str, str] | None = None) -> str:
    """The text a JS concatenation of single-quoted strings and names makes,
    each name (or call) given its value in names; anything else fails."""
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
            out.append((names or {})[name])
        at = match.end()
    return "".join(out)


def _code(page: str) -> str:
    """The page with its comments taken out."""
    return re.sub(r"/\*.*?\*/|//[^\n]*", "", page, flags=re.S)


def _in_order(text: str, *pieces: str) -> bool:
    """Whether each piece is in text, each after the one before it."""
    at = [text.index(p) for p in pieces]
    return at == sorted(at)


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


def _file(code: int, i: int = 0) -> str:
    """A part's file as Studio names it, for an EPSG code."""
    return f"result_epsg{code}_{i}_native.tif"


def _strip_of(page: str, file: str) -> int | None:
    """What the page's stripOf gives for a part's file, worked out from the
    page's own text: its pattern for a strip file and its ranges of codes
    (the strip is the code's last two digits); None for any other file."""
    pattern = _after(page, "    const f = /", "/.exec(String(m.file || ''));\n")
    match = re.search(pattern, file)
    code = int(match.group(1)) if match else 0
    body = _between(page, "  const stripOf = (m) => {", "\n  };\n")
    ranges = re.findall(r"\(c >= (\d+) && c <= (\d+)\)", body)
    assert len(ranges) == 2, ranges
    inside = any(int(low) <= code <= int(high) for low, high in ranges)
    return code % 100 if inside else None


def _by_strip(page: str, files: list[str]) -> bool:
    """The page's byStrip for a map's files: each a strip, and none twice."""
    strips = [_strip_of(page, f) for f in files]
    return None not in strips and len(set(strips)) == len(files)


# --------------------------------------------------------------------------- K1


def test_k1_a_count_is_its_number_and_one_line_says_the_rest() -> None:
    """K1: each count's button holds its number alone (no words under it),
    the part's counts are one grid of equal columns (common.js .oe-same,
    132 px at least, a count's own least width, so three counts stay on one
    row inside a part's card from 481 px up; a row each under 480 px), and
    one line under the question, shown while a part offers more than one
    number, says what more squares give; the words by a count's place are
    gone."""
    page = _read("check.html")
    assert f"// WORDING: r5e\nconst MORE_WORK = '{MORE_WORK}';\n" in page
    counts = _between(page, "  function countsBlock(m) {", "\n  }\n")
    button = (
        "        OE.el('span', { class: 'pick-count' }, OE.el('strong', {}, "
        "Number(o.labels).toLocaleString('en-US')),\n"
        "          o.labels === 1 ? ' square' : ' squares'));\n"
    )
    assert button in counts
    assert "const buttons = m.offer.map((o) => {" in counts
    group = (
        "    m.group = OE.el('div', { class: 'pick-group oe-same', role: 'group',\n"
        "      'aria-label': SQUARES_GROUP(nParts > 1 ? label(m) : null) }, "
        "buttons);\n"
    )
    assert group in counts
    for gone in ("COUNT_WORDS", "pick-words", "Least work", "The one number"):
        assert gone not in page, gone
    assert "  .pick-group { --same-min: 132px; max-width: 100%; }\n" in page
    # No narrower than a count's own least width, or a count sticks out.
    assert "    min-width: 132px; padding: 8px 12px;" in page
    narrow = "  @media (max-width: 480px) { .pick-group { --same-min: 100%; } }\n"
    assert narrow in page
    assert ".pick-group { display: flex;" not in page
    more = "  const moreLine = OE.el('div', { id: 'oe-check-more' }, MORE_WORK);\n"
    assert more in page
    assert "  root.append(ask);\n  root.append(moreLine);\n" in page
    ask = _between(page, "  function fillAsk() {", "\n  }\n")
    hidden = (
        "    moreLine.hidden = !models.some((m) => m.kind === 'pick' "
        "&& m.offer.length > 1);"
    )
    assert hidden in ask
    # Defined once, used once.
    assert _code(page).count("MORE_WORK") == 2
    assert _OFF_FACE.search(MORE_WORK) is None


# --------------------------------------------------------------------------- K2


def test_k2_each_part_is_one_card_and_the_parts_figure_is_gone() -> None:
    """K2: a map in parts draws each part as one white card (its name and
    size over its counts, or its picked line and open button, or its state);
    a map in one part keeps a plain section. The parts figure, its captions
    and the line under an open sheet are gone, with their code; the result
    card keeps its own parts figure (figures.js, mode 'result')."""
    page = _read("check.html")
    section = (
        "    root.append(OE.el('section', { class: nParts > 1 ? 'part card stack' "
        ": 'part stack' }, m.heading, m.body));\n"
    )
    assert section in page
    assert "  root.append(action, OE.fold(DETAILS, foldBody));\n" in page
    for gone in (
        "partsSlot",
        "fillParts",
        "fig('parts'",
        "PART_OPEN",
        "PART_PICKED",
        "PART_ANSWERED",
        "PART_DONE",
        "PART_CANNOT",
        "otherLine",
        "fillOther",
        "OTHER_OPEN",
        "is not checked yet",
        ": not picked yet",
    ):
        assert gone not in page, gone
    # The heading still follows the sheets' counts (countsRead, complete).
    whole = "const complete = (d) => countsRead(d) && d.n_labelled >= d.n_rows;\n"
    assert whole in page
    for where in ("  function renderAll() {", "  async function refreshZone(m) {"):
        assert "    fillAsk();\n" in _between(page, where, "\n  }\n"), where
    # The result card draws its parts figure as before.
    assert "OE.fig.parts(" in _read("result.html")


# --------------------------------------------------------------------------- K3


def test_k3_the_strips_picture_is_a_jpeg_in_the_check_view_only() -> None:
    """K3: the strips picture is a JPEG in views/brand of 720 x 328 under
    120 KB, one BRAND entry, inlined into the Check card's page alone as a
    JPEG data URL (each page keeps its one PNG, the AGENT tag); the SVG is
    never inlined and figures.js holds no placeholder. Its file's metadata
    holds no place name. What it shows (recoloured, no text, no state shape)
    is checked by eye on the round's shots, not here."""
    assert views.BRAND["STRIPS_DATA_URL"] == "brand/" + STRIPS_FILE
    path = views.VIEW_DIR / views.BRAND["STRIPS_DATA_URL"]
    data = path.read_bytes()
    assert data[:3] == b"\xff\xd8\xff" and len(data) < STRIPS_MAX_BYTES
    assert _jpeg_size(data) == STRIPS_SIZE
    # The source's place name (its <title>, its file name) is in no metadata
    # segment. Drawn text cannot be read here: the pixels are compressed.
    assert b"minnesota" not in data.lower()
    url = views.brand_data_url("STRIPS_DATA_URL")
    assert url.startswith("data:image/jpeg;base64,")
    for name in views.VIEWS:
        page = views.view_html(name)
        assert (url in page) is (name == "check"), name
        assert page.count("data:image/png;base64,") == 1, name
    check_page = views.view_html("check")
    assert "land-shape" not in check_page
    assert "@STRIPS_DATA_URL@" not in check_page
    assert "_DATA_URL@" not in _read("figures.js")


def test_k3_the_parts_line_sits_beside_the_picture_in_one_card() -> None:
    """K3: for a map in parts that are each a strip of their own, right under
    the question and its line, one card holds the picture (aria-hidden, no
    alt text: its one line says what it shows) and the line, in plain words:
    strips and parts, never a zone; the picture 360 px wide at most, the line
    beside it or under it."""
    page = _read("check.html")
    tags = "// WORDING: r5e\n// WORDING: squares\nconst PARTS_LINE = (n) => "
    assert tags in page
    said = _concat(_after(page, "const PARTS_LINE = (n) => ", ";\n"), {"n": "2"})
    assert said == PARTS_LINE_2
    assert _OFF_FACE.search(said) is None
    assert "\nconst STRIPS = '@STRIPS_DATA_URL@';\n" in page
    card = (
        "  root.append(moreLine);\n"
        "  if (nParts > 1 && byStrip) {\n"
        "    root.append(OE.el('div', { class: 'card oe-strips', "
        "id: 'oe-check-parts-line' },\n"
        "      OE.el('img', { src: STRIPS, alt: '', 'aria-hidden': 'true', "
        "width: '360', height: '164' }),\n"
        "      OE.el('div', {}, PARTS_LINE(nParts))));\n"
        "  } else if (nParts > 1) {\n"
    )
    assert card in page
    for rule in (
        "  .card.oe-strips { display: flex; flex-wrap: wrap; gap: 12px; "
        "align-items: center; padding: 12px; }\n",
        "  .oe-strips img { display: block; width: 100%; max-width: 360px; "
        "height: auto; border-radius: 2px; }\n",
        "  .oe-strips > div { flex: 1 1 200px; min-width: 0; }\n",
    ):
        assert rule in page, rule
    assert len(re.findall(r"\bSTRIPS\b", _code(page))) == 2
    assert _code(page).count("PARTS_LINE(") == 1


def test_k3_parts_that_are_not_one_per_strip_get_the_plain_line() -> None:
    """K3: the strips picture and its sentence show only where each part is a
    strip of its own: every part's file a strip file, read as the server
    reads it (check._utm), and no strip twice. One strip on both sides of the
    equator (the server's "North part" and "South part": two parts in one
    strip of the picture), a file that is no strip, or a strip saved twice
    gets the line of r5-b J18 in a card with no picture: how many parts, and
    that each gets its own check."""
    page = _read("check.html")
    tags = "// WORDING: r5e\n// WORDING: squares\nconst PARTS_PLAIN = (n) => "
    assert tags in page
    said = _concat(_after(page, "const PARTS_PLAIN = (n) => ", ";\n"), {"n": "4"})
    assert said == PARTS_PLAIN_4
    assert _OFF_FACE.search(said) is None
    assert page.count("This map is saved in") == 1
    strip = (
        "  const stripOf = (m) => {\n"
        "    const f = /^result_epsg(\\d+)_\\d+_native\\.tif$/"
        ".exec(String(m.file || ''));\n"
        "    const c = f ? Number(f[1]) : 0;\n"
        "    return (c >= 32601 && c <= 32660) || (c >= 32701 && c <= 32760) "
        "? c % 100 : null;\n"
        "  };\n"
        "  const byStrip = models.every((m) => stripOf(m) !== null) "
        "&& new Set(models.map(stripOf)).size === nParts;\n"
    )
    assert strip in page
    plain = (
        "    root.append(OE.el('div', { class: 'card', id: 'oe-check-parts-line' }, "
        "OE.el('div', {}, PARTS_PLAIN(nParts))));\n"
        "  }\n"
        "  for (const m of models) {\n"
    )
    assert _in_order(page, "  } else if (nParts > 1) {\n", plain)
    # Drawn once, with no picture and no strips class.
    assert _code(page).count("PARTS_PLAIN(") == 1
    assert "oe-strips" not in plain and "img" not in plain
    cases = [
        ([_file(32610), _file(32611, 1)], True),  # two strips side by side
        ([_file(32610), _file(32733, 1)], True),  # two strips, north and south
        ([_file(32610), _file(32733, 1), _file(32735, 2)], True),
        ([_file(32637), _file(32737, 1)], False),  # one strip, both sides
        ([_file(32610), _file(3857, 1)], False),  # a file that is no strip
        ([_file(32661), _file(32761, 1)], False),  # the polar grids: no strips
        ([_file(32610), _file(32610, 1)], False),  # a strip saved twice
        ([_file(32610), _file(32733, 1), _file(32710, 2)], False),
        ([_file(32610), "other.tif"], False),
    ]
    for files, want in cases:
        assert _by_strip(page, files) is want, files
        # The page reads a file's strip as the server does.
        for name in files:
            utm = check._utm(name)
            assert _strip_of(page, name) == (utm[0] if utm else None), name
    # The one strip on both sides of the equator is the map the server names
    # by its sides: the strips sentence ("one part for each strip") is untrue
    # of it.
    sides = check.part_names([_file(32637), _file(32737, 1)])
    assert sides == ["North part", "South part"]


# --------------------------------------------------------------------------- K4


def test_k4_one_line_over_the_press_and_the_come_back_line_in_details() -> None:
    """K4: the line over the press is two short sentences (one line at
    780 px); under the press only the line that says why it is off; the
    come-back line is in Details, between how the pick is fixed and the
    rare-feature note, in the same words, under the rule it had on the face
    (r5-a L4): only while the card has a press and shows no part as picked
    or as pressed with no reply (beside a picked part the press picks another
    part's new squares; with no press there is nothing to press again)."""
    page = _read("check.html")
    tags = "// WORDING: r5e\n// WORDING: r5b J18\n// WORDING: r4h C2\nconst CANNOT = "
    assert tags in page
    assert _concat(_after(page, "const CANNOT = ", ";\n")) == CANNOT
    assert len(CANNOT) < 100
    for gone in ("come back later", "You can stop", "againLine", "oe-check-again"):
        assert gone not in page, gone
    start = "  const pressArea = OE.el('div', { class: 'stack' },"
    area = _between(page, start, ";\n")
    assert _in_order(
        area,
        "OE.el('div', { id: 'oe-check-cannot' }, CANNOT),",
        "OE.el('div', { class: 'row' }, press),",
        "limitLine)",
    )
    assert area.endswith("    limitLine)")
    assert f"// WORDING: r5a L4\nconst PICKED_BEFORE = '{PICKED_BEFORE}';\n" in page
    fold = _between(page, "  function fillFold() {", "\n  }\n")
    pushed = (
        "    const again = drawable().length > 0 && !models.some((m) => "
        "m.kind === 'drawn' || m.kind === 'unknown');\n"
        "    nodes.push(OE.el('p', {}, FIXED));\n"
        "    if (again) nodes.push(OE.el('p', {}, PICKED_BEFORE));\n"
        "    nodes.push(OE.el('p', {}, RARE));\n"
    )
    assert pushed in fold
    # Defined once, in Details once.
    assert page.count("PICKED_BEFORE") == 2


# --------------------------------------------------------------------------- K5


def test_k5_the_rule_and_its_box_are_one_white_card() -> None:
    """K5: the gate (the rule, its scope, its example pictures, the box and
    its line under it; after a press with no reply, its one line) is one
    white card, hidden while it holds nothing; no word changes."""
    page = _read("check.html")
    gate = "  const gate = OE.el('div', { class: 'gate card stack', hidden: true });\n"
    assert gate in page
    assert "'gate stack'" not in page
    fill = _between(page, "  function fillGate() {", "\n  }\n")
    assert "gate.hidden = mode === 'none';" in fill
    assert "gate.replaceChildren(OE.el('div', {}, RULE_LOST));" in fill


# --------------------------------------------------------------------------- K6


def test_k6_the_signed_out_note_is_the_panels_two_lines() -> None:
    """K6: signed out, the Check card's note is the panel's two lines (what
    the button opens, where the key goes); its first line, which said the
    heading again, is gone; the button is as before."""
    page = _read("check.html")
    sign = _between(page, "function signIn(app, tool) {", "\n}\n")
    note = (
        "    // WORDING: r5e\n    // WORDING: r5b J19\n"
        "    note: [OE.SIGN_IN_OPENS, OE.SIGN_IN_KEY_GOES],\n"
    )
    assert note in sign
    assert "This check needs OlmoEarth Studio." not in page
    assert "acceptCurrent: true," in sign and "'Run it again'" in sign


# --------------------------------------------------------------------------- K7


def test_k7_a_class_maps_face_is_two_lines_the_rest_in_details() -> None:
    """K7: a class map without scores says on its face that it can be checked
    on points and, before any count is fixed, that the app shows no pictures
    of points (r4-m M4); what the check gives is first in its closed Details,
    over the tool's sentence. The model's note keeps its own words (the
    server's CHECK_CLASS_MAP_NOTE: "in these words or plainer")."""
    page = _read("check.html")
    start = (
        "// WORDING: r5e\n// WORDING: r4m M4 (the second line)\n"
        "// WORDING: r4h C6\nconst CLASS_MAP = [\n"
    )
    lines = _after(page, start, "\n];")
    said = tuple(_concat(s) for s in lines.rstrip(",").split(",\n"))
    assert said == CLASS_MAP
    for line in said:
        assert len(line) < 110 and _OFF_FACE.search(line) is None, line
    tag = "// WORDING: r4m M4 (in Details since r5-e)\nconst POINTS_GRID = "
    assert tag in page
    assert _concat(_after(page, "const POINTS_GRID = ", ";\n")) == POINTS_GRID
    card = _between(page, "function classMapCard(app, data, reason) {", "\n}\n")
    fold = (
        "  root.append(OE.fold(DETAILS, OE.el('div', { class: 'small' }, "
        "POINTS_GRID), OE.el('div', { class: 'small' }, reason)));"
    )
    assert fold in card
    assert "...CLASS_MAP.map((s) => OE.el('div', {}, s))," in card
    assert "single points picked at random" not in page
    assert "Each point needs your own imagery" in server.CHECK_CLASS_MAP_NOTE


# --------------------------------------------------------------------------- K8


def test_k8_a_failed_pick_says_so_once_and_its_reason_is_in_details() -> None:
    """K8: a retry is one line ("The pick failed." in red, then what this
    press does, or, for a failed pick the press does not take because more
    failed than one press takes, that a later press tries it); a pick that
    failed for good says "The pick failed." over what to tell Claude; the
    server's reason is in the part's Details lines, after its clipped line,
    never on the face, so the press's toast says each part says so, and
    promises no reason there."""
    page = _read("check.html")
    assert f"// WORDING: r5e\nconst FAILED_SHORT = '{FAILED_SHORT}';\n" in page
    tags = "// WORDING: r5e\n// WORDING: r5b J25\n// WORDING: squares\nconst RETRY = "
    assert tags in page
    retry = _concat(
        _after(page, "const RETRY = (n) => ", ";\n"),
        {"OE.counted(n, 'square')": "80 squares"},
    )
    assert retry == RETRY_80
    tags = "// WORDING: r5e\n// WORDING: squares\nconst RETRY_LATER = "
    assert tags in page
    later = _concat(
        _after(page, "const RETRY_LATER = (n) => ", ";\n"),
        {"OE.counted(n, 'square')": "80 squares"},
    )
    assert later == RETRY_LATER_80
    zone = _between(page, "  function fillZone(m) {", "\n  }\n")
    # The part asks the press's own plan whether this press takes it.
    assert _in_order(
        zone,
        "    } else if (m.kind === 'retry') {\n",
        "      const now = sendPlan().send.includes(m);\n"
        "      parts.push(OE.el('div', {}, OE.el('span', { class: 'error' }, "
        "FAILED_SHORT), ' ',\n"
        "        now ? RETRY(m.budget) : RETRY_LATER(m.budget)));\n",
        "    } else if (m.kind === 'failed') {\n"
        "      if (m.error) parts.push(OE.el('div', { class: 'error' }, "
        "FAILED_SHORT));\n",
    )
    assert "PICK_FAILED" not in zone and "m.error ||" not in zone
    fold = _between(page, "  function fillFold() {", "\n  }\n")
    assert _in_order(
        fold,
        "      if (m.clipped) lines.push(OE.el('div', {}, m.clipped));\n",
        "      if ((m.kind === 'retry' || m.kind === 'failed') && m.error) {\n"
        "        lines.push(OE.el('div', {}, PICK_FAILED + bare(m.error) + '.'));\n"
        "      }\n",
    )
    # The server's words, defined once and said in Details only.
    assert _code(page).count("PICK_FAILED") == 2
    assert _code(page).count("FAILED_SHORT") == 3
    for gone in ("to try again with the same number", "Press ' + PRESS"):
        assert gone not in page, gone
    # The press's toast: a part not picked says so; no reason is promised.
    toast = (
        "    // WORDING: r5e\n"
        "    // WORDING: squares\n"
        "    OE.toast(drawnNow.length < picked.length ? '"
        + NOT_ALL_PICKED.replace("'", "\\'")
        + "'\n      : 'The squares are picked.');\n"
    )
    assert toast in _between(page, "  async function drawPicked() {", "\n  }\n")
    assert "each part says why" not in page


def test_k8_after_a_press_with_no_reply_the_gate_alone_says_what_to_do() -> None:
    """K8: after a press with no reply the part pressed and the toast say
    only what happened, every other part only that it is not picked; the
    gate's RULE_LOST, unchanged, is the one line that sends the user to
    Claude, and the focus, when the press had it, goes to the gate (it stands
    before every part, so a keyboard or a screen reader lands on the one
    instruction; the ring shows for a keyboard only). A part left while an
    editor is open keeps LATER."""
    page = _read("check.html")
    assert f"// WORDING: r5e\nconst NOT_YET = '{NOT_YET}';\n" in page
    tags = (
        "// WORDING: r5e\n// WORDING: r5b J09\n// WORDING: r4j J2\n"
        "const NO_ANSWER = "
    )
    assert tags in page
    assert _concat(_after(page, "const NO_ANSWER = ", ";\n")) == NO_ANSWER
    assert "// WORDING: r5e\nconst PRESS_NO_ANSWER = " in page
    toast = _concat(_after(page, "const PRESS_NO_ANSWER = ", ";\n"))
    assert toast == PRESS_NO_ANSWER
    assert _concat(_after(page, "const RULE_LOST = ", ";\n")) == RULE_LOST
    zone = _between(page, "  function fillZone(m) {", "\n  }\n")
    later = (
        "    } else if (m.kind === 'later') {\n"
        "      // After a press with no reply the gate's RULE_LOST says what to do "
        "(r5-e K8).\n"
        "      parts.push(OE.el('div', {}, ruleLost ? NOT_YET : LATER));\n"
    )
    assert later in zone
    assert _code(page).count("NOT_YET") == 2
    # The focus after the press with no reply: the gate's line, not a part
    # below it (a press the server refused whole keeps the old fallback).
    press = _between(page, "  async function drawPicked() {", "\n  }\n")
    assert _in_order(
        press,
        "        OE.toast(PRESS_NO_ANSWER, true);\n      }\n      renderAll();\n",
        "      if (hadFocus && !press.isConnected) {\n"
        "        if (ruleLost) { gate.setAttribute('tabindex', '-1'); "
        "OE.focusIfHere(gate); } else focusFallback();\n"
        "      }\n"
        "      return;\n",
        "    const refused = got && typeof got.refused === 'string'",
    )
    for rule in (
        "  .gate:focus { outline: none; }\n",
        "  .gate:focus-visible { outline: 2px solid var(--text); "
        "outline-offset: 2px; }\n",
    ):
        assert rule in page, rule
    for said in (
        NO_ANSWER,
        PRESS_NO_ANSWER,
        NOT_YET,
        FAILED_SHORT,
        RETRY_80,
        RETRY_LATER_80,
        NOT_ALL_PICKED,
    ):
        assert "Claude" not in said and "Check card" not in said, said
        assert _OFF_FACE.search(said) is None, said
    assert "Ask Claude" in RULE_LOST


# --------------------------------------------------------------------------- rule R2


def test_a_long_word_breaks_inside_the_gate_and_a_parts_card() -> None:
    """Rule R2: free text in the white cards (a typed rule in the gate, the
    server's words on a part that cannot be checked) breaks a word too long
    for its line, so it never runs out of its card; a map in one part (a
    plain section) does not scroll sideways on such a word either. Since
    round r5-f F04 the server's words on a sheet that cannot be read are in
    Details, whose lines break the same way (test_mcp_r5f_check.py)."""
    page = _read("check.html")
    assert "  .gate, section.part { overflow-wrap: anywhere; }\n" in page


# --------------------------------------------------------------------------- marks


def test_every_item_of_the_track_is_marked_in_the_page() -> None:
    """Each new or changed string of this round carries ``// WORDING: r5e``:
    MORE_WORK (K1), PARTS_LINE and PARTS_PLAIN (K3), CANNOT (K4), the sign-in
    note (K6), CLASS_MAP (K7), NOT_YET, NO_ANSWER, PRESS_NO_ANSWER,
    FAILED_SHORT, RETRY, RETRY_LATER and the press's toast (K8, the toast
    pinned with its mark in the K8 test); the moved POINTS_GRID says it is
    in Details since r5-e."""
    page = _read("check.html")
    assert page.count("// WORDING: r5e\n") == 13
    for name in (
        "MORE_WORK",
        "CANNOT",
        "PARTS_LINE",
        "PARTS_PLAIN",
        "NOT_YET",
        "CLASS_MAP",
        "NO_ANSWER",
        "PRESS_NO_ANSWER",
        "FAILED_SHORT",
        "RETRY",
        "RETRY_LATER",
    ):
        marked = r"// WORDING: r5e\n(?://[^\n]*\n)*const " + name + r" = "
        assert re.search(marked, page) is not None, name
    assert "// WORDING: r4m M4 (in Details since r5-e)\nconst POINTS_GRID = " in page
