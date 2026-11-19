# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Round r5-d (17 Nov 2026), track RESULT: the result cards' words (server.py
and views/result.html), each item as specs/r5-final-audit/build_list.md gives
it. The server's words through its own names; the views are read as text (no
test here runs the scripts).

- R03: with Cannot tell the headline says which of the rate's two figures is
  which: "... is wrong on 10.0% of squares, or 12.5% if the 2 Cannot tell
  were wrong; 95% range ...". The pair is the package's (confirmed below on
  the package itself: every Cannot tell counted right, then every one
  counted wrong); every case is worked by hand from its counts; no figure of
  the headline or the view changes, only the words around them. J02's
  "between" stays for a rate of two figures with no Cannot tell. After
  review: the page writes the card's counts with thousands commas from
  1,000, as the headline does ("4,000 squares checked, 1,200 Cannot tell").
- R10: the unfinished card says where to answer and press: "Answer the rest
  on the label sheet, then press Estimate the error rate there.", in the
  server's copy and the page's, word for word for every count (thousands
  commas, "1 of 1 square"). After review: the button it names is the
  sheet's own finish button, as the sheet draws it ("Certify the zone" on a
  certificate plan's sheet, "Say which map is more accurate" on a two-map
  sheet's, "Ask Claude to grade it" with no design file beside the sheet);
  "the rest" reads for one square left or many.
- R14: the fold of a plan card and of a look-first list is named "Details",
  as every other card's fold.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("mcp")

from olmoearth_agent.mcp_app import auth, server, sheet, views  # noqa: E402
from olmoearth_agent.tools import statistical_rules as rules  # noqa: E402

#: The package's method words of a random design's interval (any text).
METHOD = "exact hypergeometric interval at 95%"

#: R03, worked by hand: 8 of 80 squares wrong is 10.0%; with the 2 Cannot
#: tell wrong too, 10 of 80 is 12.5% (the build list's example, inline
#: labels, so the subject is the map).
TWO = (
    "The map is wrong on 10.0% of squares, or 12.5% if the 2 Cannot tell were wrong; "
    "95% range 6.1% to 21.7%."
)

#: 5 of 40 wrong is 12.5%; with the one Cannot tell wrong too, 6 of 40 is
#: 15.0% (the harness's 17o); one is said in the singular.
ONE = (
    "The map is wrong on 12.5% of squares, or 15.0% if the Cannot tell was wrong; "
    "95% range 5.0% to 26.8%."
)

#: 2 of 20 points wrong is 10.0%; with the one Cannot tell wrong too, 3 of 20
#: is 15.0%.
ONE_POINT = (
    "The map is wrong on 10.0% of points, or 15.0% if the Cannot tell was wrong; "
    "95% range 2.0% to 38.0%."
)

#: 400 of 4,000 wrong is 10.0%; with the 1,200 Cannot tell wrong too, 1,600
#: of 4,000 is 40.0% (the count with its thousands comma).
MANY = (
    "The map is wrong on 10.0% of squares, or 40.0% if the 1,200 Cannot tell were "
    "wrong; 95% range 8.0% to 42.0%."
)

#: 3 of 2,000 wrong is 0.15%; with the 5 Cannot tell wrong too, 8 of 2,000 is
#: 0.4% (two significant figures below 1%, as statistical_rules.percent).
SMALL = (
    "The map is wrong on 0.15% of squares, or 0.4% if the 5 Cannot tell were wrong; "
    "95% range 0.03% to 0.9%."
)

#: None of 80 wrong is 0.0%; with the 2 Cannot tell wrong, 2 of 80 is 2.5%.
NONE_WRONG = (
    "The map is wrong on 0.0% of squares, or 2.5% if the 2 Cannot tell were wrong; "
    "95% range 0.0% to 7.3%."
)

#: Round r5-b J02's headline with Cannot tell, as server.py wrote it before
#: R03: the same figures, the words compared in the test of no change.
J02_FORM = (
    "{subject} is wrong on between {low} and {high} of {units} ({k} Cannot tell); "
    "95% range {range}."
)

#: R10: the unfinished card's line, of squares and of points, on an
#: estimate's plan's sheet.
UNFINISHED_SQUARES = (
    "Not finished yet: 3 of 80 squares on this sheet have no answer. Answer the rest "
    "on the label sheet, then press Estimate the error rate there."
)
UNFINISHED_POINTS = (
    "Not finished yet: 2 of 20 points on this sheet have no answer. Answer the rest "
    "on the label sheet, then press Estimate the error rate there."
)
#: The same on a certificate plan's sheet, whose finish button certifies.
UNFINISHED_CERTIFY = (
    "Not finished yet: 3 of 80 squares on this sheet have no answer. Answer the rest "
    "on the label sheet, then press Certify the zone there."
)

#: R10, after review: the designs beside a sheet and the finish button each
#: sheet draws (sheet.sheet_job): a plain plan's, a certificate plan's
#: (seed_from_map) and a two-map plan's.
DESIGNS: list[tuple[dict[str, Any], str]] = [
    (
        {"design": "random", "population": {"score_kind": "window_confidence"}},
        "Estimate the error rate",
    ),
    (
        {
            "design": "random",
            "population": {"score_kind": "window_confidence"},
            "certificate": {"seed_rule": "from_map", "alpha": 0.05},
        },
        "Certify the zone",
    ),
    (
        {"design": "disagreement", "classes": ["crop", "water"]},
        "Say which map is more accurate",
    ),
]

#: An estimate refused for squares left unanswered, as the tool says it.
REFUSED: dict[str, Any] = {
    "ok": False,
    "error": "ValueError: 3 of 80 design windows have no 'wrong' label yet",
    "hint": "The tool ran but failed.",
}

#: A figure as a headline writes it: "10.0%", "0.15%", "95%", "1,200".
_FIGURE = re.compile(r"\d[\d,]*(?:\.\d+)?%?")

#: One token of a JS concatenation: a single-quoted string, a plus, or a name
#: (a call with no parentheses inside its own is one name).
_TOKEN = re.compile(
    r"\s*(?:'((?:[^'\\]|\\.)*)'|(\+)|([A-Za-z_][A-Za-z0-9_.]*(?:\([^()]*\))?))"
)


@pytest.fixture(autouse=True)
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The output root, the read roots' own, and no key."""
    out = tmp_path / "outputs"
    out.mkdir()
    monkeypatch.setenv("OLMOEARTH_OUTPUT_ROOT", str(out))
    monkeypatch.delenv("OLMOEARTH_SCORES_ROOT", raising=False)
    monkeypatch.delenv(auth.KEY_ENV, raising=False)
    monkeypatch.setenv(auth.KEY_FILE_ENV, str(tmp_path / "cfg" / "studio.env"))
    return out.resolve()


def _read(name: str) -> str:
    """A file of views/, as it is on disk."""
    return (views.VIEW_DIR / name).read_text(encoding="utf-8")


def _server_source() -> str:
    """server.py, as it is on disk."""
    return Path(server.__file__).read_text(encoding="utf-8")


def _code(text: str) -> str:
    """A view's text without its comments (``/* */``, ``<!-- -->`` and ``//``
    to the end of the line), which may quote words a face no longer says."""
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def _between(text: str, start: str, end: str) -> str:
    """The part of text from start up to the first end after it."""
    at = text.index(start)
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


def _inner(n: int, wrong: int, cannot: int, low: float, high: float) -> dict[str, Any]:
    """An ok estimate's result as the tool returns it, for a random design of
    ``n`` labels, ``wrong`` of them wrong and ``cannot`` Cannot tell: with
    Cannot tell no single rate, and the package's pair as the rate's two
    ends, ``wrong / n`` (every Cannot tell right) then ``(wrong + cannot) /
    n`` (every one wrong); the 95% range from ``low`` to ``high``."""
    valid: dict[str, Any] = {"low": low, "high": high, "method": METHOD}
    if cannot:
        valid["estimate"] = None
        valid["estimate_range"] = [wrong / n, (wrong + cannot) / n]
    else:
        valid["estimate"] = wrong / n
    return {
        "plan_design": "random",
        "n_labelled": n,
        "n_unjudged": cannot,
        "nominal_coverage": 0.95,
        "valid_interval": valid,
    }


def _view(inner: dict[str, Any], unit: str = "square") -> dict[str, Any]:
    """``_estimate_result``'s view for ``inner``, of inline labels (no sheet,
    so the subject is the map), which must be one, never ``None``."""
    got = server._estimate_result(inner, None, None, unit)
    assert got is not None
    return dict(got["view"])


def _app() -> server.AppServer:
    """An app server with no key, its Studio a mock address (never called)."""
    return server.AppServer(
        None, keys=auth.KeyStore(), base_url="http://mock-studio/api/v1"
    )


def _sheet(root: Path, design: dict[str, Any] | None) -> Path:
    """A plan's sheet of one map under the root, with ``design`` beside it
    (``<stem>.json``, as sheet.design_of reads it), or none."""
    path = root / "west_to_label.csv"
    path.write_text(
        "order,window_index,wrong,reference_class\n1,4,,\n", encoding="utf-8"
    )
    if design is not None:
        (root / "west.json").write_text(json.dumps(design), encoding="utf-8")
    return path


async def _refused_view(
    monkeypatch: pytest.MonkeyPatch, arguments: dict[str, Any]
) -> dict[str, Any]:
    """The view data of an estimate the tool refused for squares left
    unanswered (:data:`REFUSED`, the tool stood in for), called as the host
    calls it, with panels."""
    app = _app()

    async def run(name: str, args: dict[str, Any]) -> dict[str, Any]:
        """The refusal, whatever is asked."""
        return dict(REFUSED)

    monkeypatch.setattr(app, "run_agent_tool", run)
    try:
        result = await app.call_agent_tool(server.ESTIMATE_TOOL, arguments, views=True)
    finally:
        await app.aclose()
    return dict((result.meta or {})[server.VIEW_DATA_KEY])


#: R03's cases: (labels, wrong, Cannot tell, unit, the 95% range's ends, the
#: headline), each worked by hand at its constant above.
CASES = [
    (80, 8, 2, "square", 0.061, 0.217, TWO),
    (40, 5, 1, "square", 0.05, 0.268, ONE),
    (20, 2, 1, "point", 0.02, 0.38, ONE_POINT),
    (4000, 400, 1200, "square", 0.08, 0.42, MANY),
    (2000, 3, 5, "square", 0.0003, 0.009, SMALL),
    (80, 0, 2, "square", 0.0, 0.073, NONE_WRONG),
]


# --------------------------------------------------------------------------- R03


def test_r03_the_pair_is_every_cannot_tell_right_then_every_one_wrong() -> None:
    """R03, confirmed in the package before building (the build list's
    condition): the valid interval's estimate_range is the rate with every
    Cannot tell counted right, then with every one counted wrong
    (oe_inferencex.estimate.valid.whole_map_valid). By hand: 80 labels of a
    random sample of 1,000 windows, 8 wrong and 2 Cannot tell, give 8 / 80 =
    0.1 and 10 / 80 = 0.125, the rates of the same labels with the two read
    as right and as wrong. tests/tools/test_estimation_unjudged.py holds the
    same through the tool on the random, confidence and class designs."""
    valid = pytest.importorskip("oe_inferencex.estimate.valid")
    sample = {"design": "random", "n_population": 1000, "indices": list(range(80))}
    wrong = [1] * 8 + [0] * 72
    unjudged = [False] * 8 + [True] * 2 + [False] * 70
    got = valid.whole_map_valid(sample, wrong, unjudged)
    assert got["estimate"] is None
    assert got["estimate_range"] == [0.1, 0.125]
    as_right = valid.whole_map_valid(sample, wrong)
    as_wrong = valid.whole_map_valid(sample, [1] * 10 + [0] * 70)
    assert (as_right["estimate"], as_wrong["estimate"]) == (0.1, 0.125)
    assert got["low"] <= 0.1 and got["high"] >= 0.125


@pytest.mark.parametrize(("n", "wrong", "cannot", "unit", "low", "high", "said"), CASES)
def test_r03_the_headline_says_which_figure_is_which(
    n: int, wrong: int, cannot: int, unit: str, low: float, high: float, said: str
) -> None:
    """R03: each case worked by hand: the subject is wrong on the rate with
    every Cannot tell right, or on the rate with every one wrong "if the k
    Cannot tell were wrong" ("if the Cannot tell was wrong" for one), then
    the 95% range; "to" once, for the range; no "between" and no brackets."""
    view = _view(_inner(n, wrong, cannot, low, high), unit)
    assert view["headline"] == said
    assert view["headline"].count(" to ") == 1
    for gone in ("between", "(", "Error rate"):
        assert gone not in view["headline"], gone
    assert view["say"] is None and view["rate"] is None


@pytest.mark.parametrize(("n", "wrong", "cannot", "unit", "low", "high", "said"), CASES)
def test_r03_no_figure_changes(
    n: int, wrong: int, cannot: int, unit: str, low: float, high: float, said: str
) -> None:
    """R03 changes the words only: the headline holds J02's figures in J02's
    order (the rate's two ends, the count, 95%, the range's two ends), and
    one Cannot tell, said in the singular, drops its "1" alone; the view's
    figures and texts are the package's pair and the range as before; the
    model's first line is as before (SAY_RATE_RANGED)."""
    view = _view(_inner(n, wrong, cannot, low, high), unit)
    lo, hi = wrong / n, (wrong + cannot) / n
    rate_range = f"{rules.percent(lo)} to {rules.percent(hi)}"
    range_text = f"{rules.percent(low)} to {rules.percent(high)}"
    before = J02_FORM.format(
        subject="The map",
        low=rules.percent(lo),
        high=rules.percent(hi),
        units=unit + "s",
        k=f"{cannot:,}",
        range=range_text,
    )
    figures = _FIGURE.findall(before)
    if cannot == 1:
        assert figures[2] == "1"
        figures = figures[:2] + figures[3:]
    assert _FIGURE.findall(view["headline"]) == figures
    assert (view["rate_low"], view["rate_high"]) == (lo, hi)
    assert view["rate_range_text"] == rate_range
    # Round r5-j B1: the bar's label, the same two figures by the headline's
    # "or", and both in the headline, in this order.
    bar = f"{rules.percent(lo)} or {rules.percent(hi)}"
    assert view["rate_bar_text"] == bar
    assert _FIGURE.findall(bar) == _FIGURE.findall(view["headline"])[:2]
    assert (view["low"], view["high"], view["range_text"]) == (low, high, range_text)
    assert (view["low_text"], view["high_text"]) == (
        rules.percent(low),
        rules.percent(high),
    )
    assert view["n_unjudged"] == cannot and view["n_checked"] == n
    got = server._estimate_result(_inner(n, wrong, cannot, low, high), None, None, unit)
    assert got is not None
    first = server.SAY_RATE_RANGED.format(
        n=f"{n:,}", units=unit + "s", part="", rate_range=rate_range, range=range_text
    )
    # Points are a grid over a whole Studio result: the whole map, no hedge.
    hedge = "" if unit == "point" else " " + server.SAY_PART_HEDGE
    assert got["say_first"][0] == first + hedge


def test_r03_between_stays_without_cannot_tell_and_the_words_are_marked() -> None:
    """R03: a rate of two figures with no Cannot tell keeps J02's "between"
    form (HEADLINE_BETWEEN, its words as before); a single rate keeps its
    everyday sentence. The new words are marked, hold the sheet's own word
    (sheet.CANNOT_TELL), and no "between" and no "to" of their own."""
    bare = _inner(80, 8, 2, 0.061, 0.217)
    bare["n_unjudged"] = 0
    assert _view(bare)["headline"] == (
        "The map is wrong on between 10.0% and 12.5% of squares; 95% range 6.1% to "
        "21.7%."
    )
    assert _view(_inner(80, 10, 0, 0.061, 0.217))["headline"] == (
        "The map is wrong on about 1 square in 8: 12.5%, 95% range 6.1% to 21.7%."
    )
    assert server.HEADLINE_OR_CANNOT == (
        "{subject} is wrong on {low} of {units}, or {high} if {which} wrong; 95% "
        "range {range}."
    )
    assert server.HEADLINE_WHICH_ONE == "the Cannot tell was"
    assert server.HEADLINE_WHICH == "the {k} Cannot tell were"
    for words in (
        server.HEADLINE_OR_CANNOT,
        server.HEADLINE_WHICH_ONE,
        server.HEADLINE_WHICH,
    ):
        assert "between" not in words and " to " not in words, words
    for which in (server.HEADLINE_WHICH_ONE, server.HEADLINE_WHICH):
        assert sheet.CANNOT_TELL in which
    source = _server_source()
    for marked in (
        "# WORDING: r5d R03\nHEADLINE_OR_CANNOT = (\n",
        '# WORDING: r5d R03\nHEADLINE_WHICH_ONE = "the Cannot tell was"\n',
        '# WORDING: r5d R03\nHEADLINE_WHICH = "the {k} Cannot tell were"\n',
        "# WORDING: r5b J02; r5d R03 (no Cannot tell here)\nHEADLINE_BETWEEN = (\n",
    ):
        assert marked in source, marked
    assert "HEADLINE_CANNOT" not in source


def test_r03_the_page_draws_the_servers_headline_whole() -> None:
    """R03: the card computes no headline of its own; it draws the server's
    sentence as it comes, so the new words reach the face unchanged."""
    page = _read("result.html")
    rate = _between(page, "function rateBlock(r) {", "\n}\n")
    assert "OE.el('p', { class: 'big' }, r.headline)," in rate
    code = _code(page)
    for word in ("Cannot tell were", "Cannot tell was", "wrong on between"):
        assert word not in code, word


def test_r03_the_card_writes_its_counts_as_the_headline_does() -> None:
    """R03, after review: from 1,000 the card's subtitle and its squares'
    heading write their counts with thousands commas, as the headline ("if
    the 1,200 Cannot tell were wrong") and the model's line ("From 4,000
    squares checked") do: "4,000 squares checked, 1,200 Cannot tell", "The
    4,000 squares you checked". Under 1,000 as before; one is singular;
    marked. The words of common.js (OE.counted) are not changed."""
    page = _read("result.html")
    words = _between(page, "const RESULT_WORDS = {", "\n};")
    head = "n.toLocaleString('en-US') + ' ' + unitsOf(unit, n) + ' checked'"
    tail = "', ' + k.toLocaleString('en-US') + ' Cannot tell'"
    marked = "  // WORDING: r5d R03 (counts)\n  checked: (n, unit, k) => "
    assert marked in words
    assert _entry(words, "checked: (n, unit, k) => ") == (
        head + "\n    + (k > 0 ? " + tail + " : '')"
    )
    for unit, n, k, said in (
        ("square", 80, 2, "80 squares checked, 2 Cannot tell"),
        ("point", 40, 1, "40 points checked, 1 Cannot tell"),
        ("square", 1, 0, "1 square checked"),
        ("square", 4000, 1200, "4,000 squares checked, 1,200 Cannot tell"),
    ):
        names = {
            "n.toLocaleString('en-US')": f"{n:,}",
            "unitsOf(unit, n)": server._units(unit, n),
            "k.toLocaleString('en-US')": f"{k:,}",
        }
        got = _concat(head, names) + (_concat(tail, names) if k > 0 else "")
        assert got == said
    many = "'The ' + n.toLocaleString('en-US') + ' ' + unitOf(unit) + 's you checked'"
    assert "  // WORDING: r5d R03 (counts)\n  squaresHead: (n, unit) => " in words
    assert "\n    : " + many + "),\n" in words
    for n, said in (
        (80, "The 80 squares you checked"),
        (4000, "The 4,000 squares you checked"),
    ):
        names = {"n.toLocaleString('en-US')": f"{n:,}", "unitOf(unit)": "square"}
        assert _concat(many, names) == said
    assert "OE.counted(n, unitOf(unit))" not in page


# --------------------------------------------------------------------------- R10


def test_r10_the_unfinished_line_names_the_label_sheet() -> None:
    """R10: the unfinished card has no button, so its second sentence says
    where the answers and the press are: the label sheet and its own finish
    button (``press``; the estimate's, sheet.JOB_LABELS["estimate"], when
    none is given). The counts in the sheet's unit, with thousands commas,
    the unit singular for a sheet of one; "the rest" for one left or many;
    marked; "again" and "them" are gone."""
    assert server.UNFINISHED == (
        "Not finished yet: {left} of {n} {units} on this sheet have no answer. Answer "
        "the rest on the label sheet, then press {press} there."
    )
    for gone in ("again", "{unit}", " them "):
        assert gone not in server.UNFINISHED, gone
    error = "ValueError: 3 of 80 design windows have no 'wrong' label yet"
    assert server._unfinished(error) == {
        "left": 3,
        "n": 80,
        "unit": "square",
        "press": sheet.JOB_LABELS["estimate"],
        "text": UNFINISHED_SQUARES,
    }
    points = server._unfinished(
        "ValueError: 2 of 20 design windows have no 'wrong' label yet", "point"
    )
    assert points == {
        "left": 2,
        "n": 20,
        "unit": "point",
        "press": "Estimate the error rate",
        "text": UNFINISHED_POINTS,
    }
    certify = server._unfinished(error, "square", sheet.JOB_LABELS["certify"])
    assert certify is not None
    assert certify["press"] == "Certify the zone"
    assert certify["text"] == UNFINISHED_CERTIFY
    many = server._unfinished("ValueError: 1200 of 4000 design windows have no label")
    assert many is not None
    assert many["text"].startswith("Not finished yet: 1,200 of 4,000 squares on ")
    one = server._unfinished("ValueError: 1 of 1 design windows have no label")
    assert one is not None
    assert one["text"].startswith("Not finished yet: 1 of 1 square on this sheet ")
    assert server._unfinished("ValueError: something else") is None
    assert "# WORDING: squares; r5d R10\nUNFINISHED = (\n" in _server_source()


def test_r10_the_page_says_the_servers_words() -> None:
    """R10: the page's copy (RESULT_WORDS.unfinished) is the server's line
    (server._unfinished's text) word for word, in squares and in points, for
    one left, for a sheet of one (the unit singular, as the server's _units)
    and from 1,000 (thousands commas, as the server's), naming the button it
    is given, else the estimate's; marked. The card that draws it is a plain
    card with no button, as before, and passes the server's press only when
    it is words."""
    page = _read("result.html")
    units = "const unitsOf = (unit, n) => unitOf(unit) + (n === 1 ? '' : 's');"
    assert units in page
    words = _between(page, "const RESULT_WORDS = {", "\n};")
    start = "unfinished: (left, n, unit, press = 'Estimate the error rate') => "
    assert "  // WORDING: r5d R10\n  " + start in words
    entry = _entry(words, start)
    for unit, left, n in (
        ("square", 3, 80),
        ("point", 2, 20),
        ("square", 1, 80),
        ("square", 1, 1),
        ("point", 1, 1),
        ("square", 1200, 4000),
    ):
        for press in (sheet.JOB_LABELS["estimate"], sheet.JOB_LABELS["certify"]):
            error = f"ValueError: {left} of {n} design windows have no label"
            said = server._unfinished(error, unit, press)
            assert said is not None
            names = {
                "left.toLocaleString('en-US')": f"{left:,}",
                "n.toLocaleString('en-US')": f"{n:,}",
                "unitsOf(unit, n)": server._units(unit, n),
                "press": press,
            }
            assert _concat(entry, names) == said["text"], (unit, left, n, press)
    for gone in ("Answer every", "error rate again", "Answer them"):
        assert gone not in page, gone
    show = _between(page, "function show(app, params) {", "\n}\n")
    for line in (
        "const press = typeof unfinished.press === 'string' && unfinished.press"
        " ? unfinished.press : undefined;",
        "root.append(OE.el('div', { class: 'card' }, RESULT_WORDS.unfinished(left, of,"
        " unfinished.unit || data.unit, press)));\n      return;",
    ):
        assert line in show, line


@pytest.mark.asyncio
@pytest.mark.parametrize(("design", "press"), DESIGNS)
async def test_r10_the_card_names_the_sheets_own_finish_button(
    root: Path, monkeypatch: pytest.MonkeyPatch, design: dict[str, Any], press: str
) -> None:
    """R10, after review: the button the unfinished card names is the one the
    sheet draws (sheet.sheet_progress's finish_label, sheet.JOB_LABELS):
    "Estimate the error rate" on a plain plan's sheet, "Certify the zone" on
    a certificate plan's (seed_from_map), "Say which map is more accurate"
    on a two-map plan's; the counts and the unit as before, and no hint."""
    path = _sheet(root, design)
    assert sheet.sheet_progress(str(path))["finish_label"] == press
    view = await _refused_view(monkeypatch, {"labels_path": str(path)})
    assert view["unfinished"] == {
        "left": 3,
        "n": 80,
        "unit": "square",
        "press": press,
        "text": server.UNFINISHED.format(
            left="3", n="80", units="squares", press=press
        ),
    }
    assert view["hint"] is None
    if press == sheet.JOB_LABELS["certify"]:
        assert view["unfinished"]["text"] == UNFINISHED_CERTIFY


@pytest.mark.asyncio
async def test_r10_a_sheet_with_no_design_beside_it_names_its_own_button(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R10, after review: a sheet with no design file beside it, graded with a
    design_path, draws sheet.NO_JOB_LABEL ("Ask Claude to grade it") as its
    finish button, so the card names that, not the estimate's. Inline labels
    (no sheet) and a sheet that cannot be read name the estimate's button."""
    path = _sheet(root, None)
    elsewhere = root / "plan.json"
    elsewhere.write_text(json.dumps(DESIGNS[0][0]), encoding="utf-8")
    args = {"design_path": str(elsewhere), "labels_path": str(path)}
    view = await _refused_view(monkeypatch, args)
    assert view["unfinished"]["press"] == sheet.NO_JOB_LABEL == "Ask Claude to grade it"
    assert view["unfinished"]["text"].endswith(
        " Answer the rest on the label sheet, then press Ask Claude to grade it there."
    )
    assert server._finish_label(None) is None
    assert server._finish_label(str(root / "gone_to_label.csv")) is None
    inline = await _refused_view(monkeypatch, {"wrong": [1, 0, 0]})
    assert inline["unfinished"]["press"] == sheet.JOB_LABELS["estimate"]
    assert inline["unfinished"]["text"] == UNFINISHED_SQUARES


# --------------------------------------------------------------------------- R14


def test_r14_the_plan_and_look_first_fold_is_named_details() -> None:
    """R14: the fold of a plan card and of a look-first list (FOLD_LABEL)
    reads "Details", as the result card's fold, the label sheet's and the
    Check card's; marked; drawn once, as before; "More about this result" is
    no string of any view."""
    page = _read("result.html")
    assert "// WORDING: r4p2; r5d R14\nconst FOLD_LABEL = 'Details';\n" in page
    words = _between(page, "const RESULT_WORDS = {", "\n};")
    assert "  details: 'Details',\n" in words
    assert page.count("OE.fold(FOLD_LABEL, ...inFold)") == 1
    assert "if (inFold.length) root.append(OE.fold(FOLD_LABEL, ...inFold));" in page
    assert "const foldLabel = 'Details';" in _read("sheet.js")
    assert "const DETAILS = 'Details';" in _read("check.html")
    for name in sorted(p.name for p in views.VIEW_DIR.iterdir() if p.is_file()):
        if name.endswith((".html", ".js")):
            assert "More about this result" not in _code(_read(name)), name
