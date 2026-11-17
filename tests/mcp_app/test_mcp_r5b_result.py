# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Round r5-b (17 Nov 2026), track RESULT: the words and figures of the result
cards, each item as specs/r5-audit/build_list.md gives it. The server's words
through its own names; the views are read as text (no test here runs the
scripts).

- J01: up to 0.5 the rate in words is the nearer of "about 1 square in N"
  and "about N squares in 10" (0.41 is 4 in 10, not 1 in 2); a tie keeps "1
  in N". The boundary rates are worked by hand below. After review: beside
  its 95% range, only a phrase that lies inside the range as printed is
  said, the nearer such; with none, the headline gives the rate plainly.
- J02: with Cannot tell the headline keeps the everyday form: the subject is
  wrong on between the rate's two ends, with how many were Cannot tell; "to"
  is the 95% range's alone.
- J22: a rate of 0 is written "0.0% wrong" after "right on every square".
- J01, J02 and J22 change the words only: every number and text the view
  carries beside the headline is as before, and so is the model's line.
- J07: a sheet whose answers changed after its first result says so directly
  under the title's line, above the rate, in red at body size; "graded" is
  gone from the line, in the server's copy and the page's.
- J12: the server's last line is the rate block's last child, at body size.
- J23: the rule used and the class lines are in Details, off the face.
- J27: the look-first CSV's line tells the latest press only, a failure
  there in red and not a toast; the points' Street map buttons are outlined.
- J04: the second class seen is a hollow dark square, the third a mid teal
  with a dark ring; no mint; the outline's legend sample is the ring alone.
- J29 (b), (d): the steps keep 12 px above them; the 0% and 100% labels and
  the range's band are darker.
- J24: the look-first pane draws its dates through OE.datesBlock (sheet.js),
  so result.html holds none of the strip's words or rules.
"""

from __future__ import annotations

import copy
import math
import re
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("mcp")

from olmoearth_agent.mcp_app import auth, server, sheet, views  # noqa: E402
from olmoearth_agent.tools import statistical_rules as rules  # noqa: E402

#: The package's method words of a random design's interval (any text).
METHOD = "exact hypergeometric interval at 95%"

#: A sheet's version as sheet.sheet_version gives it.
VERSION = "v" + "a" * 16

#: The graded mark an estimate wrote: this version, first graded now.
MARK = (VERSION, {"first_at": "2026-11-13T10:04:00Z", "changed": False})

#: The 95% range every estimate here carries, as the card writes it.
RANGE = "6.1% to 21.7%"

#: J07: the card's line for answers changed after a first result.
GRADED = (
    "Answers on this sheet changed after its first result (13 Nov 2026, 10:04 UTC). "
    "Answers changed after seeing where the map disagrees make this result invalid."
)

#: J02: the build list's sentence, for the West part with 2 Cannot tell.
BETWEEN_WEST = (
    "The West part is wrong on between 10.0% and 12.5% of squares (2 Cannot tell); "
    "95% range 6.1% to 21.7%."
)

#: J02: the same for a map whose part is not known (inline labels).
BETWEEN_MAP = (
    "The map is wrong on between 10.0% and 12.5% of squares (2 Cannot tell); 95% "
    "range 6.1% to 21.7%."
)

#: J22: the build list's sentence, for the West part with every square right.
ZERO_WEST = (
    "The West part was right on every square you checked: 0.0% wrong, 95% range "
    "0.0% to 4.5%."
)

#: The Cannot tell square's rule in figures.js (r4-i O2), which J04 keeps.
CANNOT_CSS = (
    ".oe-sq.cannot, .oe-swatch.cannot { background: transparent; box-shadow: none;"
    " border: 1px dashed var(--text); box-sizing: border-box; }"
)

#: J04: the three fills and the outline's legend sample, in figures.js's order.
FILL_RULES = (
    ".oe-sq.f0, .oe-swatch.f0 { background: var(--text); }\n"
    ".oe-sq.f1, .oe-swatch.f1 { background: var(--paper);"
    " box-shadow: inset 0 0 0 2px var(--text); }\n"
    ".oe-sq.f2, .oe-swatch.f2 { background: rgba(16,82,87,.45);"
    " box-shadow: inset 0 0 0 1px var(--text); }\n"
    ".oe-swatch.blank { background: transparent; box-shadow: none; }\n"
)

#: One token of a JS concatenation: a single-quoted string, a plus, or a name
#: (a call with no parentheses inside its own is one name).
_TOKEN = re.compile(
    r"\s*(?:'((?:[^'\\]|\\.)*)'|(\+)|([A-Za-z_][A-Za-z0-9_.]*(?:\([^()]*\))?))"
)

#: What a sentence of say_rate names: "1 square in 8", "4 squares in 10".
_SAID = re.compile(r"about (\d[\d,]*) (?:square|point)s? in (\d[\d,]*)\.$")


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


def _between(text: str, start: str, end: str) -> str:
    """The part of text from start up to the first end after it."""
    at = text.index(start)
    return text[at : text.index(end, at)]


def _in_order(text: str, *pieces: str) -> bool:
    """Whether each piece is in text, each after the one before it."""
    at = [text.index(p) for p in pieces]
    return at == sorted(at)


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


def _said_value(sentence: str) -> float:
    """The rate a sentence of say_rate names: "about 1 square in 8." is 1/8,
    "about 4 squares in 10." is 4/10."""
    match = _SAID.search(sentence)
    assert match is not None, sentence
    top, bottom = (int(g.replace(",", "")) for g in match.groups())
    return top / bottom


def _printed(value: float) -> float:
    """A fraction as the card prints it (statistical_rules.percent), read
    back: 0.4186 is printed "41.9%", so 0.419."""
    return float(rules.percent(value)[:-1]) / 100


def _inside(share: float, low: float, high: float) -> bool:
    """Whether a share lies inside the range from low to high as printed."""
    return _printed(low) - 1e-9 <= share <= _printed(high) + 1e-9


def _inner(rate: float | None = 0.125, **extra: Any) -> dict[str, Any]:
    """An ok estimate's result as the tool returns it: 80 labels on a random
    design, the rate, and its valid interval 6.1% to 21.7%."""
    inner: dict[str, Any] = {
        "plan_design": "random",
        "n_labelled": 80,
        "n_unjudged": 0,
        "nominal_coverage": 0.95,
        "valid_interval": {
            "estimate": rate,
            "low": 0.061,
            "high": 0.217,
            "method": METHOD,
        },
    }
    inner.update(extra)
    return inner


def _ranged(k: int = 2) -> dict[str, Any]:
    """The same with ``k`` Cannot tell answers: no single rate, the rate's
    own two ends 10.0% and 12.5%."""
    inner = _inner(None, n_unjudged=k)
    inner["valid_interval"]["estimate_range"] = [0.1, 0.125]
    return inner


def _zero() -> dict[str, Any]:
    """Every square right: a rate of 0, its valid interval 0.0% to 4.5%."""
    inner = _inner(0.0)
    inner["valid_interval"].update({"low": 0.0, "high": 0.045})
    return inner


def _context(root: Path) -> dict[str, Any]:
    """check.sheet_context of the West part's sheet, of a map in two parts."""
    this = {
        "part": "West part",
        "share_value": 0.467,
        "labels_csv_path": str(root / "west_to_label.csv"),
        "this": True,
    }
    other = {
        "part": "East part",
        "share_value": 0.533,
        "labels_csv_path": str(root / "east_to_label.csv"),
        "this": False,
    }
    return {
        "result_id": "r1",
        "zone_file": "result_epsg32610_0_native.tif",
        "part": "West part",
        "n_parts": 2,
        "parts_known": True,
        "map_name": "Kelp Q2",
        "period": {"start": "2024-04-01T00:00:00Z", "end": "2024-06-30T00:00:00Z"},
        "parts": [this, other],
    }


def _stand_in(monkeypatch: pytest.MonkeyPatch, ctx: dict[str, Any] | None) -> None:
    """The sheet's functions stood in for a direct ``_estimate_result`` on a
    sheet: no counts, no squares, no rule, no other part graded, no pictures,
    and ``ctx`` as the Check card's record of the sheet."""
    monkeypatch.setattr(server.check, "sheet_context", lambda p: copy.deepcopy(ctx))
    monkeypatch.setattr(server.imagery, "served_any", lambda p: False)
    monkeypatch.setattr(sheet, "labelling_rule", lambda p: None, raising=False)
    monkeypatch.setattr(sheet, "graded_state", lambda p: None, raising=False)
    monkeypatch.setattr(sheet, "graded_squares", lambda p, v: None, raising=False)
    monkeypatch.setattr(sheet, "class_counts", lambda p, v: None, raising=False)


def _result(inner: dict[str, Any], *args: Any) -> dict[str, Any]:
    """``_estimate_result``'s answer for ``inner`` (the sheet's path, its
    graded mark and the unit follow), which must be one, never ``None``."""
    got = server._estimate_result(inner, *args)
    assert got is not None
    return got


# --------------------------------------------------------------------------- J01


@pytest.mark.parametrize(
    ("rate", "said"),
    [
        # The build list's cases. 0.41: N = floor(1/0.41 + 0.5) = 2 and
        # tenths = floor(4.1 + 0.5) = 4; 4/10 is 0.01 away, 1/2 is 0.09 away.
        (0.41, "about 4 squares in 10."),
        # 0.40: N = floor(2.5 + 0.5) = 3; 4/10 is the rate, 1/3 is 0.067 away.
        (0.40, "about 4 squares in 10."),
        # 0.30: N = floor(3.33 + 0.5) = 3; 3/10 is the rate, 1/3 is 0.033 away.
        (0.30, "about 3 squares in 10."),
        # Ties keep "1 in N": 5/10 = 1/2, and 2/10 = 1/5.
        (0.50, "about 1 square in 2."),
        (0.20, "about 1 square in 5."),
        # 0.25: tenths = floor(2.5 + 0.5) = 3; 3/10 is 0.05 away, 1/4 the rate.
        (0.25, "about 1 square in 4."),
        # Under 2 in 10 only "1 in N" is said: tenths = 1 for both.
        (0.125, "about 1 square in 8."),
        (0.10, "about 1 square in 10."),
        # Around them. 0.449: 4/10 is 0.049 away, 1/2 is 0.051 away.
        (0.449, "about 4 squares in 10."),
        # 0.45: tenths = floor(4.5 + 0.5) = 5 and N = 2, the same rate: a tie.
        (0.45, "about 1 square in 2."),
        (0.44, "about 4 squares in 10."),
        # 0.37: 4/10 is 0.03 away, 1/3 is 0.037 away; 0.36: 0.04 against 0.027.
        (0.37, "about 4 squares in 10."),
        (0.36, "about 1 square in 3."),
        (0.35, "about 1 square in 3."),
        # 0.28: N = floor(3.57 + 0.5) = 4; 3/10 is 0.02 away, 1/4 is 0.03 away.
        (0.28, "about 3 squares in 10."),
        (0.26, "about 1 square in 4."),
        # 0.19: N = 5 and tenths = 2, the same rate: a tie.
        (0.19, "about 1 square in 5."),
        (0.0004, "about 1 square in 2,500."),
        # Above 0.5 as before.
        (0.51, "about 5 squares in 10."),
        (0.6, "about 6 squares in 10."),
    ],
)
def test_j01_the_nearer_phrase_at_the_boundary_rates(rate: float, said: str) -> None:
    """J01: each boundary rate worked by hand; in points and with a part as
    the subject the same phrase is chosen, and beside a wide range too."""
    assert server.say_rate(rate) == "The map is wrong on " + said
    points = "The West part is wrong on " + said.replace("square", "point")
    assert server.say_rate(rate, "point", "West part") == points
    assert server.say_rate(rate, within=(0.0, 1.0)) == "The map is wrong on " + said


@pytest.mark.parametrize(
    ("rate", "low", "high", "said"),
    [
        # The harness's 17s: 4 in 10 lies inside 38.9% to 43.2%.
        (0.41, 0.389, 0.432, "The map is wrong on about 4 squares in 10."),
        # 0.44: 4 in 10, the nearer, and 1 in 2 both lie outside 41.9% to 46.1%.
        (0.44, 0.419, 0.461, None),
        # 0.47: a tie, 1 in 2 (and 5 in 10), outside 44.8% to 49.2%.
        (0.47, 0.448, 0.492, None),
        # 0.36: 1 in 3, the nearer, lies outside 35.0% to 41.0%; 4 in 10 inside.
        (0.36, 0.35, 0.41, "The map is wrong on about 4 squares in 10."),
        # Above a half: 6 in 10 lies outside 52.8% to 57.2%.
        (0.55, 0.528, 0.572, None),
        # 0, 1 and more than 9 in 10 are said whatever the range.
        (0.0, 0.0, 0.045, "The map was right on every square you checked."),
        (1.0, 0.955, 1.0, "The map was wrong on every square you checked."),
        (0.96, 0.887, 0.991, "The map is wrong on more than 9 squares in 10."),
    ],
)
def test_j01_beside_its_range_only_a_phrase_inside_it(
    rate: float, low: float, high: float, said: str | None
) -> None:
    """J01, after review: beside its 95% range the sentence names a share
    inside the range, the nearer of the two phrases that does, or there is
    none (each worked by hand)."""
    assert server.say_rate(rate, within=(low, high)) == said


def test_j01_the_phrase_said_lies_inside_the_range_printed_beside_it() -> None:
    """J01, after review: beside a range 2 points either side of the rate
    (clamped to [0, 1]), every rate from 0.001 to 0.999 has no sentence or
    one whose share lies inside the range as printed; the phrase said
    without a range is kept wherever it lies inside. "More than 9 in 10",
    from 0.95 up, is said whatever the range: its top is above 9 in 10."""
    for k in range(1, 1000):
        rate = k / 1000
        low, high = max(0.0, rate - 0.02), min(1.0, rate + 0.02)
        said, free = server.say_rate(rate, within=(low, high)), server.say_rate(rate)
        assert free is not None, rate
        if rate >= server.SAY_OVER_NINE_FROM:
            assert said == free and free.endswith("more than 9 squares in 10.")
            assert high > 0.9, rate
        elif _inside(_said_value(free), low, high):
            assert said == free, (rate, said)
        elif said is not None:
            assert _inside(_said_value(said), low, high), (rate, said)


def test_j01_up_to_a_half_the_phrase_said_is_the_nearer_one() -> None:
    """J01, every rate from 0.001 to 0.500: the rate the sentence names is
    never farther from the exact rate than "1 in N" is, nor than "N in 10"
    is from 2 in 10 up; from 0.2 up it is within 5 points of the rate
    (before, 0.41 read "1 square in 2", 9 points off and outside the range
    printed beside it). "1 square in 2" is said from 0.45 up only."""
    for k in range(1, 501):
        rate = k / 1000
        said = server.say_rate(rate)
        assert said is not None, rate
        off = abs(_said_value(said) - rate)
        one_in = 1 / math.floor(1 / rate + 0.5)
        tenths = math.floor(10 * rate + 0.5)
        assert off <= abs(one_in - rate) + 1e-9, (rate, said)
        if tenths >= 2:
            assert off <= abs(tenths / 10 - rate) + 1e-9, (rate, said)
        if k >= 200:
            assert off <= 0.05 + 1e-9, (rate, said)
        if k < 450:
            assert not said.endswith("1 square in 2."), (rate, said)


def test_j01_changes_the_words_only() -> None:
    """J01: the harness's rate of 0.41 (17s). The sentence says 4 squares in
    10; the exact rate, the range and every number of the view are the
    estimate's, and the model's line carries the same sentence."""
    inner = _inner(0.41, n_labelled=2000)
    inner["valid_interval"].update({"low": 0.389, "high": 0.432})
    got = _result(inner, None, None)
    view = got["view"]
    assert view["say"] == "The map is wrong on about 4 squares in 10."
    assert view["headline"] == (
        "The map is wrong on about 4 squares in 10: 41.0%, 95% range 38.9% to 43.2%."
    )
    assert (view["rate"], view["low"], view["high"]) == (0.41, 0.389, 0.432)
    assert (view["rate_text"], view["range_text"]) == ("41.0%", "38.9% to 43.2%")
    assert (view["low_text"], view["high_text"]) == ("38.9%", "43.2%")
    assert view["n_checked"] == 2000
    first = (
        "The map is wrong on about 4 squares in 10 (41.0%, 95% range 38.9% to "
        "43.2%), from 2,000 squares checked."
    )
    assert got["say_first"][0] == first + " " + server.SAY_PART_HEDGE
    marked = (
        "# WORDING: r4h R1; r5b J01 (said up to 0.5 too, where it is the nearer)\n"
        "SAY_N_IN_TEN = "
    )
    assert marked in _server_source()


def test_j01_a_rate_with_no_phrase_inside_its_range_is_said_plainly() -> None:
    """J01, after review: 0.44 beside 41.9% to 46.1% has no everyday
    sentence (4 in 10 and 1 in 2 both lie outside the range): the headline
    gives the rate plainly (HEADLINE_RANGED), and so does the model's line
    (SAY_RATE_PLAIN); every number is the estimate's."""
    inner = _inner(0.44, n_labelled=2000)
    inner["valid_interval"].update({"low": 0.419, "high": 0.461})
    got = _result(inner, None, None)
    view = got["view"]
    assert view["say"] is None
    assert view["headline"] == "Error rate: 44.0%, 95% range 41.9% to 46.1%."
    assert (view["rate"], view["low"], view["high"]) == (0.44, 0.419, 0.461)
    assert (view["rate_text"], view["range_text"]) == ("44.0%", "41.9% to 46.1%")
    first = (
        "The error rate is 44.0% (95% range 41.9% to 46.1%), from 2,000 squares "
        "checked."
    )
    assert got["say_first"][0] == first + " " + server.SAY_PART_HEDGE


# --------------------------------------------------------------------------- J02


def test_j02_the_headline_with_cannot_tell_says_between() -> None:
    """J02: the subject is wrong on between the rate's two ends of squares
    (or points), with how many were Cannot tell; "to" is used once, for the
    95% range. Inline labels: the subject is the map."""
    words = (
        "{subject} is wrong on between {low} and {high} of {units}{cannot}; 95% "
        "range {range}."
    )
    assert server.HEADLINE_BETWEEN == words
    assert server.HEADLINE_CANNOT == " ({k} Cannot tell)"
    assert sheet.CANNOT_TELL in server.HEADLINE_CANNOT
    view = _result(_ranged(), None, None)["view"]
    assert view["headline"] == BETWEEN_MAP
    assert view["headline"].count(" to ") == 1
    assert "Error rate" not in view["headline"]
    points = _result(_ranged(1), None, None, "point")["view"]
    assert points["headline"] == (
        "The map is wrong on between 10.0% and 12.5% of points (1 Cannot tell); 95% "
        "range 6.1% to 21.7%."
    )
    many = _result(_ranged(1200), None, None)["view"]
    assert "of squares (1,200 Cannot tell); 95% range" in many["headline"]
    source = _server_source()
    assert "# WORDING: r5b J02\nHEADLINE_BETWEEN = (\n" in source
    assert '# WORDING: r5b J02\nHEADLINE_CANNOT = " ({k} Cannot tell)"\n' in source


def test_j02_the_part_is_the_subject(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """J02: on a map in two parts the part is the sentence's subject, as in
    the build list: "The West part is wrong on between ..."."""
    _stand_in(monkeypatch, _context(root))
    path = str(root / "west_to_label.csv")
    view = _result(_ranged(), path, MARK)["view"]
    assert view["title_text"] == "Result: West part of Kelp Q2"
    assert view["headline"] == BETWEEN_WEST
    assert view["hedge"] is None


def test_j02_changes_the_words_only() -> None:
    """J02: the rate's two ends, the 95% range, their texts and the count of
    Cannot tell answers are as before; no rate in words; the model's first
    line is as before (SAY_RATE_RANGED). With no Cannot tell counted, no
    brackets. A rate with no everyday sentence keeps HEADLINE_RANGED, and a
    result with no rate at all its range-only line."""
    got = _result(_ranged(), None, None)
    view = got["view"]
    assert (view["say"], view["rate"], view["rate_text"]) == (None, None, None)
    assert (view["rate_low"], view["rate_high"]) == (0.1, 0.125)
    assert view["rate_range_text"] == "10.0% to 12.5%"
    assert (view["low"], view["high"]) == (0.061, 0.217)
    assert view["range_text"] == RANGE
    assert (view["low_text"], view["high_text"]) == ("6.1%", "21.7%")
    assert view["n_unjudged"] == 2
    first = (
        "From 80 squares checked, the error rate is a range, 10.0% to 12.5%, with "
        "the 95% range 6.1% to 21.7%."
    )
    assert got["say_first"][0] == first + " " + server.SAY_PART_HEDGE
    bare = _result(_ranged(0), None, None)["view"]
    assert bare["headline"] == (
        "The map is wrong on between 10.0% and 12.5% of squares; 95% range 6.1% to "
        "21.7%."
    )
    # A rate too small for a sentence (say_rate gives none): the plain form.
    tiny = _result(_inner(5e-324), None, None)["view"]
    assert tiny["say"] is None and tiny["rate_text"]
    assert tiny["headline"] == server.HEADLINE_RANGED.format(
        of="", rate_range=tiny["rate_text"], range=RANGE
    )
    assert tiny["headline"].startswith("Error rate: ")
    # No rate and no ends of its own: the range alone, as before.
    alone = _result(_inner(None), None, None)["view"]
    assert alone["headline"] == server.SAY_RANGE_ONLY.format(
        n="80", units="squares", part="", range=RANGE
    )


# --------------------------------------------------------------------------- J22


def test_j22_a_rate_of_zero_is_written_as_wrong(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """J22: after "right on every square you checked" the number says what
    it counts, "0.0% wrong"; no other rate gets the word; the view's rate
    and its text, the range and the model's line are as before."""
    assert server.HEADLINE_ZERO_WRONG == " wrong"
    got = _result(_zero(), None, None)
    view = got["view"]
    assert view["headline"] == (
        "The map was right on every square you checked: 0.0% wrong, 95% range 0.0% "
        "to 4.5%."
    )
    assert (view["rate"], view["rate_text"]) == (0.0, "0.0%")
    assert (view["low"], view["high"]) == (0.0, 0.045)
    assert view["range_text"] == "0.0% to 4.5%"
    assert view["say"] == "The map was right on every square you checked."
    first = (
        "The map was right on every square you checked (0.0%, 95% range 0.0% to "
        "4.5%), from 80 squares checked."
    )
    assert got["say_first"][0] == first + " " + server.SAY_PART_HEDGE
    whole = _zero()
    whole["valid_interval"]["estimate"] = 0
    assert _result(whole, None, None)["view"]["headline"] == view["headline"]
    for rate, start in (
        (1.0, "The map was wrong on every square you checked: 100.0%, 95% range "),
        (0.125, "The map is wrong on about 1 square in 8: 12.5%, 95% range "),
    ):
        other = _result(_inner(rate), None, None)["view"]
        assert other["headline"] == start + RANGE + ".", rate
    # A small rate beside a range that holds it (J01, after review: the words
    # name a share inside the range printed beside them).
    small = _inner(0.0004)
    small["valid_interval"].update({"low": 0.0001, "high": 0.003})
    assert _result(small, None, None)["view"]["headline"] == (
        "The map is wrong on about 1 square in 2,500: 0.04%, 95% range 0.01% to 0.3%."
    )
    _stand_in(monkeypatch, _context(root))
    west = _result(_zero(), str(root / "west_to_label.csv"), MARK)["view"]
    assert west["headline"] == ZERO_WEST
    marked = '# WORDING: r5b J22\nHEADLINE_ZERO_WRONG = " wrong"\n'
    assert marked in _server_source()


# --------------------------------------------------------------------------- J07


def test_j07_the_changed_answers_line_in_both_copies(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """J07: "Answers on this sheet changed after its first result (...)" and
    the second sentence as before, in the server's line and the page's own
    copy, each marked; "graded" is in neither. The model's note is as it
    was."""
    assert server.GRADED_LINE.format(first_at="13 Nov 2026, 10:04 UTC") == GRADED
    assert "graded" not in GRADED
    page = _read("result.html")
    words = _between(page, "const RESULT_WORDS = {", "\n};")
    assert "  // WORDING: r5b J07\n  gradedBefore: (at) => " in words
    entry = _entry(words, "gradedBefore: (at) => ")
    assert _concat(entry, {"at": "13 Nov 2026, 10:04 UTC"}) == GRADED
    assert "first graded" not in page
    assert "# WORDING: r4p2; r5b J07\nGRADED_LINE = (\n" in _server_source()
    _stand_in(monkeypatch, _context(root))
    again = (VERSION, {"first_at": "2026-11-13T10:04:00Z", "changed": True})
    got = _result(_inner(), str(root / "west_to_label.csv"), again)
    assert got["view"]["graded_before"]["line"] == GRADED
    note = server.GRADED_AGAIN_NOTE.format(first_at="2026-11-13T10:04:00Z")
    assert note in got["notes"]
    # The headline and the figures are not withdrawn (the owner's call).
    assert got["view"]["headline"].startswith("The West part is wrong on about 1 ")


def test_j07_the_line_stands_above_the_rate_at_body_size() -> None:
    """J07: the card draws the line right after the title's line and before
    the rate block, as p.error (no 'small'), where it was the fourth line."""
    card = _between(_read("result.html"), "function resultCard(", "\n}\n")
    line = (
        "card.append(OE.el('p', { class: 'error oe-graded-again' },"
        " RESULT_WORDS.gradedBefore(text(before.first_at) || 'earlier')));"
    )
    assert line in card
    assert "small error" not in card
    assert _in_order(
        card,
        "RESULT_WORDS.checked(n, unit, unjudged)",
        "if (before && before.changed === true) {",
        line,
        "const rate = rateBlock(r);",
        "oe-uncovered",
        "RESULT_WORDS.squaresHead(",
    )
    assert card.count("RESULT_WORDS.gradedBefore(") == 1


# --------------------------------------------------------------------------- J12


def test_j12_the_last_line_is_under_the_range_at_body_size() -> None:
    """J12: the server's last line is the rate block's last child (after the
    hedge), p.oe-last with no 'small'; the card appends it itself only when
    no rate block was drawn; no rule of the page makes it small."""
    page = _read("result.html")
    rate = _between(page, "function rateBlock(r) {", "\n}\n")
    last = "text(r.last_line) ? OE.el('p', { class: 'oe-last' }, r.last_line) : null);"
    assert rate.endswith("    " + last)
    assert _in_order(rate, "r.headline),", "    line,\n", "text(r.hedge)", last)
    card = _between(page, "function resultCard(", "\n}\n")
    kept = (
        "if (!rate && text(r.last_line)) card.append(OE.el('p', { class: 'oe-last' },"
        " r.last_line));"
    )
    assert kept in card
    assert card.count("r.last_line") == 2
    assert "{ class: 'small' }, r.last_line" not in page
    assert re.search(r"\.oe-last\s*\{", page) is None
    # The sentence itself is the server's, unchanged.
    assert server.LAST_LINE_PICTURES == (
        "Checked against satellite pictures, not a site visit; no verdict on "
        "whether the map is good enough."
    )


# --------------------------------------------------------------------------- J23


def test_j23_the_rule_and_the_class_lines_are_in_details() -> None:
    """J23: the "Rule you used" line and each class line are pushed first
    into the Details fold; the face, which ends where the fold's lines start,
    appends neither; under the squares only the no-split line stays."""
    page = _read("result.html")
    card = _between(page, "function resultCard(", "\n}\n")
    face, fold = card.split("  const folded = [];\n")
    rule = (
        "folded.push(OE.el('p', { class: 'small oe-rule-used' },"
        " RESULT_WORDS.ruleUsed(r.rule.text) + (whose ? ' ' + whose : '')));"
    )
    assert rule in fold
    assert "RESULT_WORDS.ruleUsed(" not in face and "ruleTyped" not in face
    assert page.count("RESULT_WORDS.ruleUsed(") == 1
    # The class lines are made beside the squares and kept for the fold.
    assert _in_order(
        face,
        "let classLines = [];",
        ".filter((c) => c && typeof c === 'object' && text(c.line))",
        ".map((c) => OE.el('p', { class: 'small oe-class-line' }, c.line));",
        "OE.fig.squares(sq),\n      text(sq.no_split) ? OE.el('p', { class: 'small"
        " oe-no-split' }, sq.no_split) : null));",
    )
    assert "classLines" not in face.split("OE.fig.squares(sq),")[1]
    assert "...classLines" not in card and "...lines" not in card
    # The fold's order: the rule, the class lines, then what it held before.
    assert _in_order(
        fold,
        rule,
        "classLines.forEach((line) => folded.push(line));",
        "fewDetail.forEach(",
        "if (unseenDetail) folded.push(",
        "if (rows.length) folded.push(",
        "if (folded.length) card.append(OE.fold(RESULT_WORDS.details, ...folded));",
    )


# --------------------------------------------------------------------------- J27


def test_j27_the_csv_line_tells_the_latest_press_only() -> None:
    """J27: a press empties the line under the button before it asks; the
    file's name and a failure are both written there (the failure in red,
    class 'small csv-file error'); the failure is no longer a toast, the
    refusal of a download still is. The failure is announced at once, as
    OE.toast does for an error (J29 (e): role alert, aria-live assertive),
    set before the words; anything else is a polite status."""
    page = _read("result.html")
    block = _between(page, "function reviewBlock(app, block) {", "\n}\n")
    csv = _between(block, "  function csvBlock() {", "\n  }\n")
    urgency = (
        "said.setAttribute('role', failed ? 'alert' : 'status');",
        "said.setAttribute('aria-live', failed ? 'assertive' : 'polite');",
    )
    for text in (
        "const said = OE.el('div', { class: 'small csv-file',"
        " 'aria-live': 'polite' });",
        "const say = (words, failed = false) => {",
        *urgency,
        "said.className = failed ? 'small csv-file error' : 'small csv-file';",
        "said.textContent = words;",
        "          // WORDING: r5b J27 (the failure under the button, in red, not a"
        " toast)\n",
    ):
        assert text in csv, text
    assert _in_order(
        csv,
        "const say = (words, failed = false) => {",
        *urgency,
        "said.textContent = words;",
    )
    assert _in_order(
        csv,
        "busy = true;",
        "say('');",
        "try {",
        "say(REVIEW_WORDS.csvSaved(file));",
        "} catch (err) {",
        "say(REVIEW_WORDS.csvNotSaved(bareText(err.message)"
        " || REVIEW_WORDS.listNotMade), true);",
        "} finally {",
    )
    assert csv.count("OE.toast(") == 1
    assert "OE.toast(REVIEW_WORDS.notAllowed, true);" in csv
    assert "OE.toast(REVIEW_WORDS.csvNotSaved(" not in page
    # The words themselves are r4-k K8's, unchanged.
    words = _between(page, "const reviewWords = (unit) => {", "\n};")
    made = _concat(_entry(words, "csvNotSaved: (why) => "), {"why": "it failed"})
    assert made == "The list was not saved: it failed."


def test_j27_the_points_street_map_buttons_are_outlined() -> None:
    """J27: the Street map button takes its look as an argument, a text
    button by default (under a picture card); the row of points' buttons
    asks for 'small', the outlined small button."""
    page = _read("result.html")
    block = _between(page, "function reviewBlock(app, block) {", "\n}\n")
    made = (
        "  const mapButton = (rank, words, cls = 'link') => OE.el('button', {\n"
        "    type: 'button', class: cls, 'data-focus': 'map:' + rank,"
    )
    assert made in block
    assert block.count("mapButton(") == 2
    assert "mapButton(w.rank, REVIEW_WORDS.streetMap))));" in block
    row = _between(block, "    if (n && linkKey) {", "    return section;")
    assert "windows.map((w) => mapButton(w.rank, String(w.rank), 'small')))," in row


# --------------------------------------------------------------------------- J04


def test_j04_the_second_and_third_class_are_told_by_more_than_a_pale_fill() -> None:
    """J04: the first class seen is filled dark; the second is hollow (paper
    with a 2 px dark ring), the third a mid teal with a 1 px dark ring; no
    mint in the figures (the look keeps it for a completed status' ring);
    the legend's sample for the red outline has nothing inside. Cannot tell
    stays dashed and empty, the red outline as it was."""
    script = _read("figures.js")
    style = _between(script, "  const STYLE = `", "`;")
    assert FILL_RULES in style
    assert "--mint" not in script
    for gone in (
        "{ background: var(--mint); }",
        ".oe-swatch.f2 { background: var(--wash-2); }",
        ".oe-swatch.blank { background: var(--paper); }",
    ):
        assert gone not in style, gone
    # The three fills differ from each other.
    fills = re.findall(r"\.oe-sq\.f\d, \.oe-swatch\.f\d \{ ([^}]*)\}", style)
    assert len(fills) == 3 and len(set(fills)) == 3
    assert CANNOT_CSS in style
    assert style.index(FILL_RULES) < style.index(CANNOT_CSS)
    wrong = ".oe-sq.wrong, .oe-swatch.wrong { outline: 3px solid var(--error);"
    assert wrong + " outline-offset: 2px; }" in style
    # The legend's outline entry is still the blank swatch with the outline.
    assert "entries.push({ cls: 'blank wrong'," in script
    head = script[: script.index("OE.fig = (() => {")]
    assert "the second --mint" not in head and "r5-b J04" in head


# --------------------------------------------------------------------------- J29


def test_j29_the_steps_margin_and_the_darker_labels_and_band() -> None:
    """J29 (b): the steps keep 12 px above them (this style sheet comes after
    common.js's, so a margin of 0 beat .stack's gap); (d): the 0% and 100%
    labels are --text-2 and the range's band is 45%."""
    script = _read("figures.js")
    style = _between(script, "  const STYLE = `", "`;")
    assert ".oe-steps { list-style: none; margin: 12px 0 0; padding: 0;" in style
    assert ".oe-fig .tick-label { font-size: 11px; fill: var(--text-2); }\n" in style
    assert ".oe-fig .band { fill: var(--text-2); fill-opacity: .45; }\n" in style
    for gone in ("fill-opacity: .3;", "fill: var(--grey)", "margin: 0; padding: 0;"):
        assert gone not in style, gone
    # The axis keeps its grey stroke.
    assert ".oe-fig .axis { stroke: var(--grey); stroke-width: 1; }\n" in style
    assert "r5-b, J29" in script[: script.index("OE.fig = (() => {")]


# --------------------------------------------------------------------------- J24


def test_j24_the_look_first_pane_draws_its_dates_through_the_sheets_block() -> None:
    """J24, the part in result.html: the look-first pane draws its dates and
    the colour switch through OE.datesBlock (sheet.js), as a strip, and has
    no words or rules of the strip of its own, so the strip's fix
    (common.js) and the switch's shorter label (sheet.js) reach it without a
    change here."""
    page = _read("result.html")
    block = _between(page, "function reviewBlock(app, block) {", "\n}\n")
    assert block.count("OE.datesBlock(") == 2
    assert block.count("{ colour: state.colour, choose, strip: true") == 2
    for own in ("Near-infrared", "alse colour", "face-dates", "True colour"):
        assert own not in page, own
