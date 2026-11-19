# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Round r5-j (18 Nov 2026), track RESULT, as specs/r5-j-round.md gives it:
the result card's items of the audit after r5-i
(specs/r5-i-audit/build_list.md B1 and B4). The server's words through its
own names; the views are read as text, as they are on disk (no test here
runs the scripts).

- B1: with Cannot tell, the dark bar of the result's line figure is
  labelled with the headline's two figures joined by the headline's own
  word, "10.0% or 40.0%" (the server's new ``rate_bar_text``), so the figure
  is named "10.0% or 40.0%; 95% range 8.0% to 42.0%" and "to" is the 95%
  range's alone; it read "10.0% to 40.0%" (``rate_range_text``), which a
  reader quoted as the range. With no Cannot tell but a rate of two figures
  the label is "between 10.0% and 12.5%", the word of that headline. No
  number changes: ``rate_range_text``, Claude's text (``SAY_RATE_RANGED``)
  and the Details rows are as they were; the page falls back to
  ``rate_range_text`` when no ``rate_bar_text`` is sent. After review: near
  either end of the line a label that long (23 characters) runs inward from
  that end (figures.js ``labelAt``), so the figure's edge never cuts it; a
  label of 14 characters or fewer is placed as before.
- B4: the face's line for the classes no square had opens with what the
  reader did not see: 'You did not see "Snow" in any square, ...' and "You
  did not see 6 of the map's 8 classes in any square, ..." (three
  constants); it read 'You saw "Snow" in no square' and "You saw 6 of the
  map's 8 classes in no square", whose start a quick reader took for the
  classes seen. Still what the reader saw, not what the squares hold
  (r4-j J3). The Details row (``CLASS_UNSEEN_ROW``, ``CLASS_UNSEEN_DETAIL``)
  and Claude's line (``SAY_SAW_NONE``) keep their words.

The browser harness (scripts/mcp_views_e2e.py) reads no file under tests;
its fixtures copy the server's words, pinned here: 17c, 17o and 17v read the
bar's label and the figure's name, 17z and 17za (and 17c) where the bar's
label is drawn, and 17b, 17i, 17j, 17m and 17o the unseen line.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("mcp")

from olmoearth_agent.mcp_app import auth, server, views  # noqa: E402
from olmoearth_agent.tools import statistical_rules as rules  # noqa: E402

#: A figure of a sentence: a count or a percent ("1,200", "10.0%", "95%").
_FIGURE = re.compile(r"\d[\d,]*(?:\.\d+)?%?")

#: B1: the page's new rateText, under its mark, and the line after it.
RATE_TEXT = (
    "      // WORDING: r5j B1\n"
    "      rateText: rate !== null ? text(r.rate_text)"
    " : (text(r.rate_bar_text) || text(r.rate_range_text)),\n"
    "      rangeText: text(r.range_text),\n"
)
#: B1: the rateText the page passed before (the rate's own range, "a to b").
OLD_RATE_TEXT = "rateText: rate !== null ? text(r.rate_text) : text(r.rate_range_text),"
#: B1 (after review): figures.js's label above the line, run inward from an
#: end when it is long, as written (RATE_CH a character, a little over).
LABEL_RULES = (
    "  const RATE_CH = 7.5;\n"
    "  const labelAt = (x, text) => {\n"
    "    // r5-j B1\n"
    "    const edge = Math.max(60, (text.length * RATE_CH) / 2 + 6);\n"
    "    const anchor = x < edge ? 'start' : x > RL.w - edge ? 'end' : 'middle';\n"
    "    const tx = anchor === 'start' ? Math.max(0, x - 6)"
    " : anchor === 'end' ? Math.min(RL.w, x + 6) : x;\n"
)

#: B4: the three face lines, word for word.
UNSEEN_ONE = (
    "You did not see {names} in any {unit}, so this check says nothing about how "
    "well the map finds it. The rate above is for {all}."
)
UNSEEN = (
    "You did not see {names} in any {unit}, so this check says nothing about how "
    "well the map finds them. The rate above is for {all}."
)
UNSEEN_MANY = (
    "You did not see {k} of the map's {m} classes in any {unit}, so this check "
    "says nothing about how well the map finds them. The rate above is for {all}. "
    "Names under Details."
)

#: B4: the harness's unseen lines (UNSEEN_PRESENT 17b, UNSEEN_TWO 17i,
#: UNSEEN_MANY 17j, the points' line 17m, UNSEEN_SNOW 17o), word for word.
HARNESS_PRESENT = (
    'You did not see "present" in any square, so this check says nothing about how '
    "well the map finds it. The rate above is for every square of this part, all "
    "classes together."
)
HARNESS_TWO = (
    'You did not see "Tree cover", "Shrubland" in any square, so this check says '
    "nothing about how well the map finds them. The rate above is for every square "
    "of the map, all classes together."
)
HARNESS_MANY = (
    "You did not see 6 of the map's 8 classes in any square, so this check says "
    "nothing about how well the map finds them. The rate above is for every square "
    "of the map, all classes together. Names under Details."
)
HARNESS_POINT = (
    'You did not see "Shrub" in any point, so this check says nothing about how well '
    "the map finds it. The rate above is for every point of the map's grid, all "
    "classes together."
)
HARNESS_SNOW = (
    'You did not see "Snow" in any square, so this check says nothing about how '
    "well the map finds it. The rate above is for every square of the map, all "
    "classes together."
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


def _between(text: str, start: str, end: str) -> str:
    """The part of ``text`` from ``start`` up to the next ``end``."""
    at = text.index(start)
    return text[at : text.index(end, at)]


def _server_source() -> str:
    """server.py, as it is on disk."""
    return Path(server.__file__).read_text(encoding="utf-8")


def _inner(
    n: int, wrong: int, cannot: int, low: float, high: float, **valid: Any
) -> dict[str, Any]:
    """An ok estimate's result as the tool returns it, for a random design of
    ``n`` labels, ``wrong`` of them wrong and ``cannot`` Cannot tell: with
    Cannot tell no single rate and the package's pair as the rate's two
    ends (every Cannot tell right, then every one wrong); the 95% range from
    ``low`` to ``high``; ``valid`` replaces any of the interval's keys."""
    interval: dict[str, Any] = {"low": low, "high": high, "method": "exact"}
    if cannot:
        interval["estimate"] = None
        interval["estimate_range"] = [wrong / n, (wrong + cannot) / n]
    else:
        interval["estimate"] = wrong / n
    interval.update(valid)
    return {
        "plan_design": "random",
        "n_labelled": n,
        "n_unjudged": cannot,
        "nominal_coverage": 0.95,
        "valid_interval": interval,
    }


def _result(inner: dict[str, Any], unit: str = "square") -> dict[str, Any]:
    """``_estimate_result``'s answer for ``inner``, of inline labels (no
    sheet, so the subject is the map), which must be one, never ``None``."""
    got = server._estimate_result(inner, None, None, unit)
    assert got is not None
    return dict(got)


# --------------------------------------------------------------------------- B1


@pytest.mark.parametrize(
    ("n", "wrong", "cannot", "low", "high", "bar", "named"),
    [
        # 17c: 8 of 80 wrong, and with the 2 Cannot tell wrong too, 10 of 80.
        (80, 8, 2, 0.061, 0.217, "10.0% or 12.5%", "6.1% to 21.7%"),
        # 17v: 400 of 4,000 wrong, and 1,600 with the 1,200 Cannot tell.
        (4000, 400, 1200, 0.08, 0.42, "10.0% or 40.0%", "8.0% to 42.0%"),
        # 17o: 5 of 40 wrong, and 6 with the one Cannot tell.
        (40, 5, 1, 0.05, 0.268, "12.5% or 15.0%", "5.0% to 26.8%"),
    ],
)
def test_b1_the_bar_is_labelled_with_the_headlines_or(
    n: int,
    wrong: int,
    cannot: int,
    low: float,
    high: float,
    bar: str,
    named: str,
) -> None:
    """B1: with Cannot tell, rate_bar_text is the headline's two figures, in
    its order, joined by its own "or" (HEADLINE_OR_CANNOT); no "to" of its
    own, so the figure's name, "{bar}; 95% range {range}", says "to" once,
    for the 95% range."""
    view = _result(_inner(n, wrong, cannot, low, high))["view"]
    assert view["rate_bar_text"] == bar
    first, last = bar.split(" or ")
    assert (first, last) == (
        rules.percent(wrong / n),
        rules.percent((wrong + cannot) / n),
    )
    assert f"is wrong on {first} of squares, or {last} if " in view["headline"]
    assert _FIGURE.findall(bar) == _FIGURE.findall(view["headline"])[:2]
    assert " to " not in bar and "between" not in bar
    assert view["range_text"] == named
    name = view["rate_bar_text"] + "; 95% range " + view["range_text"]
    assert name.count(" to ") == 1


def test_b1_with_no_cannot_tell_the_bar_says_between() -> None:
    """B1: a rate of two figures with no Cannot tell keeps the headline's
    "between ... and ..." (HEADLINE_BETWEEN), in the bar's label too."""
    inner = _inner(80, 8, 2, 0.061, 0.217)
    inner["n_unjudged"] = 0
    view = _result(inner)["view"]
    assert view["rate_bar_text"] == "between 10.0% and 12.5%"
    assert "wrong on between 10.0% and 12.5% of squares;" in view["headline"]
    assert view["rate_range_text"] == "10.0% to 12.5%"


@pytest.mark.parametrize(
    ("span", "low", "high", "bar", "headline"),
    [
        # 17z: the bar's middle at 16.3% of the line.
        (
            [0.162, 0.164],
            0.1,
            0.25,
            "between 16.2% and 16.4%",
            "The map is wrong on between 16.2% and 16.4% of squares; 95% range "
            "10.0% to 25.0%.",
        ),
        # 17za: at 84.0% of it.
        (
            [0.839, 0.841],
            0.75,
            0.9,
            "between 83.9% and 84.1%",
            "The map is wrong on between 83.9% and 84.1% of squares; 95% range "
            "75.0% to 90.0%.",
        ),
    ],
)
def test_b1_the_harness_between_labels_are_the_servers(
    span: list[float], low: float, high: float, bar: str, headline: str
) -> None:
    """B1 (after review): the harness's 17z and 17za, a rate of two figures
    with no Cannot tell (a stratum given no label: the package's
    estimate_range and no estimate), carry the server's bar label and
    headline, word for word."""
    inner = _inner(80, 13, 0, low, high, estimate=None, estimate_range=span)
    view = _result(inner)["view"]
    assert (view["rate_bar_text"], view["headline"]) == (bar, headline)
    assert (view["rate_low"], view["rate_high"]) == tuple(span)


def test_b1_a_rate_of_its_own_has_no_bar_and_no_label() -> None:
    """B1: a single rate is a dot labelled with its own text: no bar, so no
    rate_bar_text; nor with no rate and no ends of its own."""
    view = _result(_inner(80, 10, 0, 0.061, 0.217))["view"]
    assert (view["rate_text"], view["rate_range_text"]) == ("12.5%", None)
    assert view["rate_bar_text"] is None
    alone = _result(_inner(80, 8, 2, 0.061, 0.217, estimate_range=None))["view"]
    assert (alone["rate_range_text"], alone["rate_bar_text"]) == (None, None)


def test_b1_no_number_changes_and_claudes_text_stays() -> None:
    """B1 changes the bar's words only: the rate's own ends, rate_range_text
    ("a to b"), the 95% range and Claude's first line (SAY_RATE_RANGED) are
    as before, for squares and for points."""
    for unit, units in (("square", "squares"), ("point", "points")):
        got = _result(_inner(80, 8, 2, 0.061, 0.217), unit)
        view = got["view"]
        assert (view["rate_low"], view["rate_high"]) == (0.1, 0.125)
        assert view["rate_range_text"] == "10.0% to 12.5%"
        assert view["rate_bar_text"] == "10.0% or 12.5%"
        assert (view["low_text"], view["high_text"]) == ("6.1%", "21.7%")
        first = server.SAY_RATE_RANGED.format(
            n="80",
            units=units,
            part="",
            rate_range="10.0% to 12.5%",
            range="6.1% to 21.7%",
        )
        assert got["say_first"][0].startswith(first)
        assert "10.0% or 12.5%" not in " ".join(got["say_first"])


def test_b1_the_server_gives_and_marks_the_bars_label() -> None:
    """B1: the server builds rate_bar_text beside rate_range_text, from the
    same two figures, and puts it in the view; both places marked."""
    source = _server_source()
    built = (
        "            # WORDING: r5j B1\n"
        "            rate_bar_text = (\n"
        '                f"{rules.percent(first)} or {rules.percent(last)}"\n'
        "                if n_unjudged\n"
        '                else f"between {rules.percent(first)} and '
        '{rules.percent(last)}"\n'
        "            )\n"
    )
    ranged = (
        '            rate_range_text = f"{rules.percent(first)} to '
        '{rules.percent(last)}"\n'
    )
    assert ranged + built in source
    given = '        # WORDING: r5j B1\n        "rate_bar_text": rate_bar_text,\n'
    assert given in source
    assert '        "rate_range_text": rate_range_text,\n' in source


def test_b1_the_page_labels_the_bar_with_the_servers_text() -> None:
    """B1: rateBlock passes rate_bar_text as the bar's label when there is no
    rate (rate_range_text only when the server sent none), marked; the line
    figure names itself "{rateText}; 95% range {rangeText}", so the name is
    "10.0% or 12.5%; 95% range ..."; the page computes nothing."""
    page = _read("result.html")
    rate = _between(page, "function rateBlock(r) {", "\n}\n")
    assert RATE_TEXT in rate
    assert OLD_RATE_TEXT not in page
    for word in ("toFixed", "Math.", "* 100", "' or '"):
        assert word not in rate, word
    figures = _read("figures.js")
    assert "const label = [rateText, said].filter(Boolean).join('; ');" in figures
    assert "const said = rangeText ? '95% range ' + rangeText : '';" in figures


def _label_edges(x: float, text: str) -> tuple[float, float]:
    """LABEL_RULES's label above x: its left and right edges, each character
    RATE_CH (7.5) wide, a little over its drawn width."""
    width = len(text) * 7.5
    edge = max(60.0, width / 2 + 6)
    if x < edge:
        left = max(0.0, x - 6)
        return left, left + width
    if x > 320 - edge:
        right = min(320.0, x + 6)
        return right - width, right
    return x - width / 2, x + width / 2


def test_b1_the_bars_label_is_run_inward_only_when_long() -> None:
    """B1 (after review): figures.js runs the label above the line inward from
    an end within half its width and 6 of it (LABEL_RULES), so a label of 14
    characters or fewer keeps the edge of 60 it had ("10.0% or 12.5%",
    "12.5%"); the bar's "between ... and ..." of 17z and 17za (23 characters,
    its middle at x about 63 and 258), which a centred label ran past the
    figure's edge, is now run inward. The browser harness measures it."""
    script = _read("figures.js")
    assert LABEL_RULES in script
    assert ".oe-fig .rate-label { font-size: 12px; font-weight: 600; }" in script
    for short in ("10.0% or 12.5%", "10.0% to 12.5%", "12.5%", "100.0%"):
        assert max(60.0, len(short) * 7.5 / 2 + 6) == 60.0, short
    for low, high in ((0.162, 0.164), (0.839, 0.841)):
        text = f"between {rules.percent(low)} and {rules.percent(high)}"
        x = 16 + (low + high) / 2 * (304 - 16)
        half = len(text) * 7.5 / 2
        # Centred on x before (60 <= x <= 260), past the edge.
        assert 60 <= x <= 260 and (x - half < 0 or x + half > 320), (text, x)
        left, right = _label_edges(x, text)
        assert 0 <= left and right <= 320, (text, x, left, right)


@pytest.mark.parametrize("v", [0.0, 0.05, 0.163, 0.3, 0.5, 0.7, 0.84, 0.95, 1.0])
def test_b1_a_long_bar_label_stays_inside_the_figure(v: float) -> None:
    """B1 (after review): wherever the bar's middle falls on the line, its
    "between ... and ..." label (21 to 25 characters) stays inside the
    320-wide figure by LABEL_RULES."""
    for text in (
        f"between {rules.percent(v)} and {rules.percent(v)}",
        "between 100.0% and 100.0%",
    ):
        left, right = _label_edges(16 + v * (304 - 16), text)
        assert 0 <= left and right <= 320, (text, v, left, right)


# --------------------------------------------------------------------------- B4


def test_b4_the_three_face_lines_say_what_was_not_seen() -> None:
    """B4: each face line opens "You did not see ... in any {unit}", never
    "You saw ... in no {unit}"; the rest of each line is as before."""
    assert server.CLASS_UNSEEN_ONE == UNSEEN_ONE
    assert server.CLASS_UNSEEN == UNSEEN
    assert server.CLASS_UNSEEN_MANY == UNSEEN_MANY
    for line in (UNSEEN_ONE, UNSEEN, UNSEEN_MANY):
        assert line.startswith("You did not see ")
        assert "You saw" not in line and " in no " not in line


@pytest.mark.parametrize(
    ("names", "unit", "options", "line"),
    [
        (['"present"'], "square", {"several": True}, HARNESS_PRESENT),
        (['"Tree cover"', '"Shrubland"'], "square", {}, HARNESS_TWO),
        ([f'"C{i}"' for i in range(6)], "square", {"m": 8}, HARNESS_MANY),
        (['"Shrub"'], "point", {}, HARNESS_POINT),
        (['"Snow"'], "square", {}, HARNESS_SNOW),
    ],
)
def test_b4_the_harness_lines_are_the_servers(
    names: list[str], unit: str, options: dict[str, Any], line: str
) -> None:
    """B4: the harness's unseen lines (17b, 17i, 17j, 17m, 17o) are the
    server's, word for word; with more than five, the names wait in Details
    under the Details row's own words."""
    got, detail = server._unseen_lines(names, unit, **options)
    assert got == line
    if len(names) > server.CLASS_UNSEEN_LISTED:
        assert str(detail).startswith("Classes you saw in no square: ")
    else:
        assert detail is None


def test_b4_details_and_claudes_words_stay() -> None:
    """B4 changes the face's three lines only: the Details row and line and
    Claude's line keep "you saw ... in no ..." (Claude's line counted from
    what the reader saw, r4-i F1); the lines are marked."""
    assert server.CLASS_UNSEEN_ROW == "Classes you saw in no {unit}"
    assert server.CLASS_UNSEEN_DETAIL == "Classes you saw in no {unit}: {names}"
    assert server.SAY_SAW_NONE == "You saw {names} in no {unit}."
    source = _server_source()
    for name in ("CLASS_UNSEEN_ONE", "CLASS_UNSEEN", "CLASS_UNSEEN_MANY"):
        assert f"# WORDING: r4h R2, C/result-1; r5j B4\n{name} = (\n" in source, name


def test_b4_the_uncovered_lines_carry_the_new_line() -> None:
    """B4: a sheet's counts with a class no square had give the new line as
    the card's uncovered line (directly under the headline), and the many
    form keeps its thousands commas."""
    counts = {
        "n": 80,
        "n_unjudged": 0,
        "classes": [
            {"code": "0", "name": "Water", "seen": 80, "agreed": 70},
            {"code": "1", "name": "Snow", "seen": 0, "agreed": 0},
        ],
    }
    under = server._uncovered(counts, "square", 80, several=False)
    assert under["lines"] == [HARNESS_SNOW]
    many = [f'"C{i}"' for i in range(1200)]
    line, _detail = server._unseen_lines(many, "square", m=1210)
    assert str(line).startswith(
        "You did not see 1,200 of the map's 1,210 classes in any square, "
    )
