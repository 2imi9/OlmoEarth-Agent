# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Round r5-k (18 Nov 2026), track RESULT, as specs/r5-k-round.md gives it:
K1, the owner's choice on O4, and the result card's part of K5, the owner's
choice on O11 (specs/r5-i-audit/owner_decisions.md and
specs/r5-g-audit/owner_decisions.md; the owner accepted each on 18 Nov).

- K1: a result whose sheet's answers changed after its first result (the
  server's graded mark says so, and keeps saying so: ``changed`` stays
  true) is withdrawn. The card's title says "This result is out of date";
  its face shows no rate and no range, only what was checked (the part and
  the map the title named, the period, the count) and one line, in the
  words of the sheet's own notice (r5-k K3), naming no button: "The answers
  on this sheet changed after its first result, so this result does not
  count. A new result needs another person and a fresh sheet; ask Claude
  for one." (the spec's "Press Estimate the error rate again" led back to
  this card; the round's review; round r5-m B4 added "; ask Claude for
  one", test_mcp_r5m_result.py). The rate and range (the headline, whole),
  marked out of date with the time this grading computed them from the
  changed answers, and the first result's date (server.GRADED_LINE) are in
  its Details. Claude says the same: the line it says first is that the
  result (the part's, on a map of two parts) is out of date and withdrawn,
  the first block's note says to state no must_state item and no rate or
  range from it, and the next note the same. No number is recomputed or
  changed: the view keeps every one.
- K5 (the result card's part): a look-first Street map press that fails
  (no OpenStreetMap link, a link the host did not open, a refusal) says so
  in one line right under the row of buttons it came from (the open
  square's pane, or the points' row), in red and as an alert, no longer in
  the status line at the card's foot (6h); its words are the toast's. It
  tells the latest press only (a late failure of an earlier press says
  nothing), stays in place while the pane redraws (an alert put back in the
  page is read out again) and is emptied when the pane closes.

The views are read as text (no test here runs the scripts). The browser
harness (scripts/mcp_views_e2e.py) draws them: 17d_estimate_graded_again
(the withdrawn card, its Details opened at the end), 6h (squares 2 and 3:
the line under the pane's Street map; square 2 closed and opened again
shows none; square 3's line kept in place while its pictures arrive) and
6i (point 3: the line under the points' row, emptied by point 4's press;
point 5's late failure after point 6's press says nothing).
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any

import pytest

mcp = pytest.importorskip("mcp")
from mcp import types  # noqa: E402

from olmoearth_agent.mcp_app import auth, server, sheet, views  # noqa: E402

BASE = "http://mock-studio/api/v1"
ESTIMATE = "olmoearth_estimate_map_error"

#: The package's method words of a random design's interval (any text).
METHOD = "exact hypergeometric interval at 95%"

#: A sheet's version as sheet.sheet_version gives it.
VERSION = "v" + "a" * 16

#: The first grading's time, as the graded mark keeps it, and as the card
#: writes it; and this grading's own (record_grading's last_at, the mark's
#: last.at), when a withdrawn result's rate was computed.
FIRST_RAW = "2026-11-13T10:04:00Z"
FIRST_AT = "13 Nov 2026, 10:04 UTC"
LAST_RAW = "2026-11-18T09:12:00Z"
LAST_AT = "18 Nov 2026, 09:12 UTC"

#: The graded mark of a sheet graded once (calm), and of one whose answers
#: changed after its first result (again).
CALM = (VERSION, {"first_at": FIRST_RAW, "changed": False, "gradings": 1})
AGAIN = (
    VERSION,
    {"first_at": FIRST_RAW, "last_at": LAST_RAW, "changed": True, "gradings": 2},
)

#: K1: the spec's title, and the line in the words of the sheet's own notice
#: (r5-k K3), naming no button: the mark stays changed, so a new press of
#: the sheet's finish button gives this card again (the round's review).
#: Round r5-m B4: the line ends "; ask Claude for one." (it said no next
#: step); Claude's own line (SAY_WEST, SAY_ONE) is unchanged.
OUT_OF_DATE = "This result is out of date"
LINE = (
    "The answers on this sheet changed after its first result, so this result "
    "does not count. A new result needs another person and a fresh sheet; ask "
    "Claude for one."
)
#: What Claude says first: the West part's result (a map of two parts), and
#: a map of one part's or one whose part is not known.
SAY_WEST = (
    "This result for the West part is out of date and withdrawn: the answers on "
    "this sheet changed after its first result. A new result needs another "
    "person and a fresh sheet."
)
SAY_ONE = (
    "This result is out of date and withdrawn: the answers on this sheet changed "
    "after its first result. A new result needs another person and a fresh sheet."
)
#: The server's GRADED_LINE with the first result's date (r5-b J07).
GRADED = (
    "Answers on this sheet changed after its first result (13 Nov 2026, 10:04 UTC). "
    "Answers changed after seeing where the map disagrees make this result invalid."
)
#: The West part's headline at 12.5% (server.HEADLINE).
HEADLINE = (
    "The West part is wrong on about 1 square in 8: 12.5%, 95% range 6.1% to 21.7%."
)
#: That headline in Details, marked out of date with the time this grading
#: computed it from the changed answers (never the first result's date).
OLD_RATE = (
    "Out of date, do not use (computed 18 Nov 2026, 09:12 UTC from the changed "
    "answers): " + HEADLINE
)

#: The app owner's rule, as sheet.labelling_rule gives it.
OWNER_RULE = {
    "text": "Kelp: canopy covers most of the square.",
    "source": "owner",
    "scope": "For kelp maps only.",
}

#: One token of a JS concatenation: a single-quoted string, a plus, or a name.
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


def _between(text: str, start: str, end: str) -> str:
    """The part of text from start up to the first end after it."""
    at = text.index(start)
    return text[at : text.index(end, at)]


def _in_order(text: str, *pieces: str) -> bool:
    """Whether each piece is in text, each after the one before it."""
    at = [text.index(p) for p in pieces]
    return at == sorted(at)


def _code(page: str) -> str:
    """The page with its comments taken out (a // after a colon, as in an
    address, is kept)."""
    return re.sub(r"/\*.*?\*/|(?<![:/])//[^\n]*", "", page, flags=re.S)


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


def _inner(**extra: Any) -> dict[str, Any]:
    """An ok estimate's result as the tool returns it: 80 labels on a random
    design, 12.5% wrong, its valid interval 6.1% to 21.7%."""
    inner: dict[str, Any] = {
        "available": True,
        "plan_design": "random",
        "n_labelled": 80,
        "n_unjudged": 0,
        "nominal_coverage": 0.95,
        "valid_interval": {
            "estimate": 0.125,
            "low": 0.061,
            "high": 0.217,
            "method": METHOD,
        },
        "must_state": ["The error rate is estimated at 12.5%."],
    }
    inner.update(extra)
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


def _stand_in(
    monkeypatch: pytest.MonkeyPatch,
    root: Path,
    *,
    rule: dict[str, Any] | None = None,
    finish: str | None = "Estimate the error rate",
) -> None:
    """The sheet's functions stood in for a direct ``_estimate_result`` on
    the West part's sheet: no counts, no squares, no other part graded, no
    pictures, ``rule`` as its labelling rule, and ``finish`` as its finish
    button (sheet.sheet_progress's finish_label; ``None``: the sheet cannot
    be read)."""
    ctx = _context(root)

    def progress(path: str) -> dict[str, Any]:
        if finish is None:
            raise OSError("the sheet cannot be read")
        return {"finish_label": finish}

    monkeypatch.setattr(server.check, "sheet_context", lambda p: copy.deepcopy(ctx))
    monkeypatch.setattr(server.imagery, "served_any", lambda p: False)
    monkeypatch.setattr(
        sheet, "labelling_rule", lambda p: copy.deepcopy(rule), raising=False
    )
    monkeypatch.setattr(sheet, "graded_state", lambda p: None, raising=False)
    monkeypatch.setattr(sheet, "graded_squares", lambda p, v: None, raising=False)
    monkeypatch.setattr(sheet, "class_counts", lambda p, v: None, raising=False)
    monkeypatch.setattr(sheet, "sheet_progress", progress, raising=False)


def _result(root: Path, graded: Any) -> dict[str, Any]:
    """``_estimate_result``'s answer for the West part's sheet with the
    graded mark ``graded``, which must be one, never ``None``."""
    got = server._estimate_result(_inner(), str(root / "west_to_label.csv"), graded)
    assert got is not None
    return got


# --------------------------------------------------------------------------- K1


def test_k1_the_servers_words_and_their_marks() -> None:
    """K1: the card's title, its one line (no button: the sheet's notice's
    words), what was checked, the rate marked out of date with the time it
    was computed (and without one), what Claude says first and the first
    block's note, each marked r5k K1; the model's next note says the result
    is withdrawn and to state no rate and no range from it, marked too."""
    assert server.RESULT_OUT_OF_DATE == OUT_OF_DATE
    assert server.WITHDRAWN_LINE == LINE
    assert server.WITHDRAWN_OF == "{part} of {name}"
    assert server.OUT_OF_DATE_RATE.format(at=LAST_AT, headline=HEADLINE) == OLD_RATE
    assert server.OUT_OF_DATE_RATE_UNTIMED.format(headline=HEADLINE) == (
        "Out of date, do not use: " + HEADLINE
    )
    assert server.SAY_WITHDRAWN.format(part=server._for_part("West part")) == SAY_WEST
    assert server.SAY_WITHDRAWN.format(part=server._for_part(None)) == SAY_ONE
    assert server._for_part("Part 2") == " for Part 2"
    # No button named, with panels or without: pressed again, the sheet's
    # finish button gives this card again (the mark stays changed).
    for said in (server.WITHDRAWN_LINE, server.SAY_WITHDRAWN):
        assert "Press" not in said and "{press}" not in said
    # The first block's note asks for no must_state item (SAY_FIRST_NOTE
    # asks for each, and an estimate's first is the rate and its range).
    first = server.SAY_WITHDRAWN_NOTE
    assert "then each must_state" not in first
    assert "state no item of its must_state" in first
    assert "this overrides the rule to state each must_state item" in first
    note = server.GRADED_AGAIN_NOTE.format(first_at=FIRST_RAW)
    assert note.startswith("This result is withdrawn: ")
    assert "State no rate and no range from it" in note
    assert "this overrides every must_state item" in note
    assert "say so with the rate" not in note
    source = _server_source()
    for name in (
        "RESULT_OUT_OF_DATE = ",
        "WITHDRAWN_OF = ",
        "OUT_OF_DATE_RATE = (",
        "OUT_OF_DATE_RATE_UNTIMED = ",
        "SAY_WITHDRAWN = (",
        "SAY_WITHDRAWN_NOTE = (",
    ):
        assert "# WORDING: r5k K1\n" + name in source, name
    # Round r5-m B4 changed the line (its last words), marked as K1's follow-up.
    assert "# WORDING: r5k K1; r5m B4 (K1 follow-up)\nWITHDRAWN_LINE = (" in source
    # Round r5-m B4 added how to plan a fresh sheet to Claude's note.
    mark = "# WORDING: r4p2; r5k K1; r5m B4 (K1 follow-up)\nGRADED_AGAIN_NOTE = (\n"
    assert mark in source
    # The first result's date and the reason stay the server's GRADED_LINE.
    assert server.GRADED_LINE.format(first_at=FIRST_AT) == GRADED


def test_k1_the_page_says_the_servers_words() -> None:
    """K1: result.html's RESULT_WORDS holds the same texts, word for word,
    each marked r5k K1: the title, the line (one text, no button in it), and
    the rate marked out of date with the time it was computed, and without
    one."""
    words = _between(_read("result.html"), "const RESULT_WORDS = {", "\n};")
    assert "  // WORDING: r5k K1\n  outOfDate: " in words
    assert _concat(_entry(words, "  outOfDate: "), {}) == OUT_OF_DATE
    # Round r5-m B4 changed the line (its last words), marked as K1's follow-up.
    assert "  // WORDING: r5k K1; r5m B4 (K1 follow-up)\n  withdrawn: '" in words
    assert _concat(_entry(words, "  withdrawn: "), {}) == LINE
    timed = "outOfDateRate: (headline, at) => "
    assert "  // WORDING: r5k K1\n  " + timed in words
    rate = _concat(_entry(words, timed), {"headline": HEADLINE, "at": LAST_AT})
    assert rate == OLD_RATE
    untimed = "outOfDateRateUntimed: (headline) => "
    assert "  // WORDING: r5k K1\n  " + untimed in words
    bare = _concat(_entry(words, untimed), {"headline": HEADLINE})
    assert bare == server.OUT_OF_DATE_RATE_UNTIMED.format(headline=HEADLINE)


def test_k1_a_changed_sheet_withdraws_the_result(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """K1: answers changed after the first result: the view carries
    withdrawn (the title, the line, what was checked as the title names it,
    this grading's time and the headline marked out of date with it, never
    the first result's date); every other field and number is the one a
    calm sheet's result gives (no number changes); Claude says the withdrawn
    line alone, the part named, with no rate, under the withdrawn note, and
    gets the next note alone (no rule note asks for the rule "with the
    rate")."""
    _stand_in(monkeypatch, root, rule=OWNER_RULE)
    calm = _result(root, CALM)
    got = _result(root, AGAIN)
    view = got["view"]
    assert view["headline"] == HEADLINE
    assert view["withdrawn"] == {
        "title": OUT_OF_DATE,
        "line": LINE,
        "of": "West part of Kelp Q2",
        "computed_at": LAST_AT,
        "old_rate": OLD_RATE,
    }
    # The time in Details is this grading's (last_at), not the first's.
    assert FIRST_AT not in view["withdrawn"]["old_rate"]
    assert view["graded_before"]["changed"] is True
    assert view["graded_before"]["line"] == GRADED
    drop = ("graded_before", "withdrawn")
    kept = {k: v for k, v in view.items() if k not in drop}
    assert kept == {k: v for k, v in calm["view"].items() if k not in drop}
    assert calm["view"]["withdrawn"] is None
    assert got["say_first"] == [SAY_WEST]
    assert "%" not in " ".join(got["say_first"])
    assert got["note"] == server.SAY_WITHDRAWN_NOTE
    assert got["notes"] == [server.GRADED_AGAIN_NOTE.format(first_at=FIRST_RAW)]
    # A calm sheet's result is as before: the rate first, the rule note.
    assert calm["say_first"][0].startswith("The West part is wrong on about 1 ")
    assert calm["note"] == server.SAY_FIRST_NOTE
    # r5-l L1: "the app's rule" (it said "the app owner's rule").
    owner = "The sheet records the app's rule"
    assert any(n.startswith(owner) for n in calm["notes"])
    assert not any(server.GRADED_AGAIN_NOTE.split("{")[0] in n for n in calm["notes"])


@pytest.mark.parametrize(
    "finish",
    ["Estimate the error rate", "Certify the zone", "Ask Claude to grade it", None],
)
def test_k1_the_line_names_no_button(
    root: Path, monkeypatch: pytest.MonkeyPatch, finish: str | None
) -> None:
    """K1 (the round's review): the card's line and Claude's name no button,
    whatever the sheet's finish button is (sheet.sheet_progress's
    finish_label; ``None``: the sheet cannot be read). The mark stays
    changed, so pressing it again gives this card again; "Certify the zone"
    runs another tool, and the card has no button of its own."""
    _stand_in(monkeypatch, root, finish=finish)
    got = _result(root, AGAIN)
    gone = got["view"]["withdrawn"]
    assert "press" not in gone
    assert gone["line"] == LINE
    assert got["say_first"] == [SAY_WEST]
    for said in (gone["line"], *got["say_first"]):
        assert "Press" not in said
        assert finish is None or finish not in said


def test_k1_the_card_and_claude_name_what_was_checked(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """K1 (the round's review): what was checked, as the result's title
    names it: the part and the map on a map of two parts (above), the map
    alone on a map of one part, nothing when which part is not known (the
    neutral title); Claude's line names the part only on a map of two parts
    or more, so a map of one part's line is as it was."""
    _stand_in(monkeypatch, root)
    one = {**_context(root), "n_parts": 1, "part": None}
    monkeypatch.setattr(server.check, "sheet_context", lambda p: copy.deepcopy(one))
    got = _result(root, AGAIN)
    assert got["view"]["withdrawn"]["of"] == "Kelp Q2"
    assert got["say_first"] == [SAY_ONE]
    unknown = {**_context(root), "parts_known": False}
    monkeypatch.setattr(server.check, "sheet_context", lambda p: copy.deepcopy(unknown))
    got = _result(root, AGAIN)
    assert got["view"]["title_text"] == server.RESULT_SQUARES
    assert got["view"]["withdrawn"]["of"] is None
    assert got["say_first"] == [SAY_ONE]


def test_k1_a_change_seen_in_the_race_withdraws_too(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """K1: no mark written by this grading (a save landed while the
    estimate read the sheet), but the mark already there shows the answers
    changed since the first grading (server._mark_changed): withdrawn the
    same way."""
    _stand_in(monkeypatch, root)
    west = str(root / "west_to_label.csv")
    state = {"first_at": FIRST_RAW, "changed_since_first": True, "changed": False}
    monkeypatch.setattr(
        sheet, "graded_state", lambda p: dict(state) if p == west else None
    )
    got = _result(root, None)
    gone = got["view"]["withdrawn"]
    assert gone["line"] == LINE
    # This grading wrote no mark: no time of its own, and never the first's.
    assert gone["computed_at"] is None
    assert gone["old_rate"] == "Out of date, do not use: " + HEADLINE
    assert got["say_first"] == [SAY_WEST]
    assert got["note"] == server.SAY_WITHDRAWN_NOTE
    assert got["notes"] == [server.GRADED_AGAIN_NOTE.format(first_at=FIRST_RAW)]


def test_k1_a_second_press_on_a_changed_sheet_is_withdrawn_again(root: Path) -> None:
    """K1 (the round's review), with the sheet's own functions: a sheet
    graded, an answer changed, graded again (withdrawn), then graded once
    more on the same answers, as pressing the sheet's finish button again
    does: withdrawn again, with the same line and the same line said first,
    since the graded mark keeps ``changed`` (r4-p2). So neither may promise
    a new result from this sheet. Each Details time is that grading's own
    (record_grading's ``last_at``)."""
    windows = (7, 3, 5)
    listed = [
        {"order": o, "window_index": w, "row": 0, "col": w, "map_class": 0}
        for o, w in enumerate(windows)
    ]
    population = {
        "score_kind": "class_map",
        "classes": ["water", "trees"],
        "map_class": [0] * 10,
    }
    design = {"design": "random", "population": population, "to_label": listed}
    design_path = root / "land.json"
    design_path.write_text(json.dumps(design), encoding="utf-8")
    # A sheet with no place columns (no place is written here).
    path = root / "land_to_label.csv"
    header = "order,window_index,row,col,map_class,stratum,wrong,reference_class"
    rows = [f"{o},{w},0,{w},water,0,," for o, w in enumerate(windows, start=1)]
    path.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")
    sheet.save_labels(
        str(path), [{"window_index": w, "reference_class": "water"} for w in windows]
    )
    first = sheet.record_grading(
        str(path), sheet.sheet_version(str(path)), design_path=str(design_path)
    )
    assert first["changed"] is False
    assert first["last_at"] == first["first_at"]
    # One answer changed after the first result, then graded twice.
    sheet.save_labels(str(path), [{"window_index": 7, "reference_class": "trees"}])
    version = sheet.sheet_version(str(path))
    results = []
    for _ in range(2):
        mark = sheet.record_grading(str(path), version, design_path=str(design_path))
        assert mark["changed"] is True
        got = server._estimate_result(_inner(), str(path), (version, mark))
        assert got is not None
        gone = got["view"]["withdrawn"]
        assert gone["line"] == LINE
        assert gone["computed_at"] == server._when_text(mark["last_at"])
        results.append(got)
    assert results[1]["say_first"] == results[0]["say_first"] == [SAY_ONE]
    assert results[1]["note"] == server.SAY_WITHDRAWN_NOTE


async def _estimate_call(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    grading: dict[str, Any],
    *,
    views: bool = True,
) -> tuple[list[str], dict[str, Any]]:
    """The estimate through the tool's call on the West part's sheet, graded
    with ``grading`` (what sheet.record_grading answers), with panels or
    without: the text blocks Claude reads, and the result view."""
    sheet_path = root / "west_to_label.csv"
    header = "order,window_index,wrong,reference_class\n"
    sheet_path.write_text(header, encoding="utf-8")
    app = server.AppServer(None, keys=auth.KeyStore(), base_url=BASE)
    envelope = {"ok": True, "result": _inner()}

    async def run(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        return copy.deepcopy(envelope)

    _stand_in(monkeypatch, root)
    monkeypatch.setattr(app, "run_agent_tool", run)
    monkeypatch.setattr(sheet, "sheet_version", lambda p: VERSION, raising=False)
    monkeypatch.setattr(
        sheet,
        "record_grading",
        lambda p, v, design_path=None: copy.deepcopy(grading),
        raising=False,
    )
    args = {"design_path": str(root / "west.json"), "labels_path": str(sheet_path)}
    result = await app.call_agent_tool(ESTIMATE, args, views=views)
    await app.aclose()
    texts = [c.text for c in result.content if isinstance(c, types.TextContent)]
    return texts, (result.meta or {})[server.VIEW_DATA_KEY]["result"]


@pytest.mark.asyncio
async def test_k1_claude_is_told_the_result_is_withdrawn(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """K1 through the tool's call: the first text block Claude reads holds
    the withdrawn line as the one line to say first, with the withdrawn note
    (no "then each must_state item": the first item is the rate and its
    range), the next the note to state no rate and no range; the panel's view
    carries the withdrawn block with every number kept."""
    texts, view = await _estimate_call(root, monkeypatch, AGAIN[1])
    first = json.loads(texts[0])
    assert first["say_first"] == [SAY_WEST]
    assert first["note"] == server.SAY_WITHDRAWN_NOTE
    assert "then each must_state" not in first["note"]
    assert texts[1] == server.GRADED_AGAIN_NOTE.format(first_at=FIRST_RAW)
    assert view["withdrawn"]["title"] == OUT_OF_DATE
    assert (view["rate"], view["low"], view["high"]) == (0.125, 0.061, 0.217)


@pytest.mark.asyncio
async def test_k1_a_calm_result_keeps_the_first_note(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """K1, the control: a sheet whose answers did not change gets the rate's
    lines first with SAY_FIRST_NOTE, as before."""
    texts, view = await _estimate_call(root, monkeypatch, CALM[1])
    first = json.loads(texts[0])
    assert first["note"] == server.SAY_FIRST_NOTE
    assert first["say_first"][0].startswith("The West part is wrong on about 1 ")
    assert view["withdrawn"] is None


@pytest.mark.asyncio
async def test_k1_without_panels_claude_names_no_button(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """K1 (the round's review) in a client without panels: no label sheet's
    panel and no button there, so the line said first names no button and
    no sheet's button (it is the one line both clients get)."""
    texts, _view = await _estimate_call(root, monkeypatch, AGAIN[1], views=False)
    first = json.loads(texts[0])
    assert first["say_first"] == [SAY_WEST]
    assert first["note"] == server.SAY_WITHDRAWN_NOTE
    for said in first["say_first"]:
        assert "Press" not in said
        for button in ("Estimate the error rate", "Certify the zone"):
            assert button not in said


def test_k1_the_card_draws_the_withdrawn_card_in_place_of_the_result() -> None:
    """K1: resultCard draws withdrawnCard for a result whose graded_before
    says changed; withdrawnCard's face is the title, what was checked (the
    server's withdrawn.of, the part and the map, then the period and the
    count) and the one line (no button), then a closed Details fold holding
    the headline marked out of date (the server's line with the time it was
    computed, else the same words here, with withdrawn.computed_at or
    without a time), then the first result's date and why (GRADED_LINE);
    no rate block, figure, tally, parts, button, rows or notes. resultCard
    no longer draws the red line over the headline (r5-b J07)."""
    page = _read("result.html")
    card = _between(page, "function resultCard(app, data) {", "\n}\n")
    assert _in_order(
        card,
        "const r = data.result;",
        "if (withdrawnOf(r)) return withdrawnCard(data);",
        "const rate = rateBlock(r);",
    )
    assert "gradedBefore" not in card and "oe-graded-again" not in card
    which = _between(page, "function withdrawnOf(r) {", "\n}\n")
    assert "return before && before.changed === true ? before : null;" in which
    gone = _between(page, "function withdrawnCard(data) {", "\n}\n")
    assert _in_order(
        gone,
        "OE.el('section', { class: 'card stack oe-result oe-withdrawn' });",
        "card.append(OE.el('h2', {}, RESULT_WORDS.outOfDate));",
        "const sub = [text(gone.of), text(r.period_text),",
        "RESULT_WORDS.checked(n, unit, unjudged)",
        "card.append(OE.el('p', { class: 'oe-withdrawn-line' },"
        " RESULT_WORDS.withdrawn));",
        "const headline = text(r.headline);",
        "const at = text(gone.computed_at);",
        "const old = text(gone.old_rate)",
        "RESULT_WORDS.outOfDateRate(headline, at)",
        "RESULT_WORDS.outOfDateRateUntimed(headline)",
        "if (old) folded.push(OE.el('p', { class: 'small oe-out-of-date' }, old));",
        "folded.push(OE.el('p', { class: 'small oe-graded-again' },"
        " RESULT_WORDS.gradedBefore(text(before.first_at) || 'earlier')));",
        "card.append(OE.fold(RESULT_WORDS.details, ...folded));",
        "return card;",
    )
    # No button named any more (the round's review): no press field read.
    assert "press" not in gone
    # Four blocks: the title, what was checked, the line, the closed fold.
    assert _code(gone).count("card.append(") == 4
    for absent in (
        "rateBlock(",
        "OE.fig",
        "classCheckOf(",
        "OE.sendJob(",
        "cardBlock(",
        "figureBlock(",
        "notesBlock(",
        "'button'",
        "primary",
        "last_line",
        "hedge",
    ):
        assert absent not in gone, absent
    # The headline (rate and range) only in the fold, marked out of date.
    assert _code(gone).count("r.headline") == 1
    assert _code(gone).count("folded.push(") == 2


# --------------------------------------------------------------------------- K5


def _review_block() -> str:
    """result.html's reviewBlock, as it is on disk."""
    return _between(_read("result.html"), "function reviewBlock(app, block) {", "\n}\n")


def test_k5_a_street_map_failure_is_said_under_its_button() -> None:
    """K5: a Street map press that fails writes no toast any more: no
    OpenStreetMap link, a link the host did not open (OE.open quiet) and a
    refusal each go to fail, which says the toast's words in the one line
    (sayMap) only for the latest press (mapPress, counted after the busy
    guard: a late failure of an earlier press, made before a later one,
    says nothing; the round's review) and only while the button pressed is
    still drawn; a press empties that line first, so it tells the latest
    press only."""
    block = _review_block()
    assert "  let mapPress = 0;" in block
    assert _code(block).count("++mapPress") == 1
    opener = _between(block, "  async function openMap(rank) {", "\n  }\n")
    assert "OE.toast(" not in opener
    assert _in_order(
        opener,
        "if (linking.has(rank)) return;",
        "const mine = ++mapPress;",
        "const fail = (words) => {",
        "if (mine === mapPress && section.querySelector("
        "'[data-focus=\"map:' + rank + '\"]')) sayMap(rank, words);",
        "mark(true);",
        "sayMap(rank);",
        "const got = await OE.call(app, 'olmoearth_app_review_link',"
        " { review_key: linkKey, rank });",
        "if (!url.startsWith(OSM_LINK)) fail(REVIEW_WORDS.noLink(rank));",
        "else if (!(await OE.open(app, url, { quiet: true })))"
        " fail(OE.LINK_NOT_OPENED);",
        "fail(REVIEW_WORDS.notOpened(bareText(err.message)"
        " || REVIEW_WORDS.linkNotMade));",
    )
    # The words are the toast's (r4-k K7), unchanged.
    words = _read("result.html")
    for said in (
        "noLink: (k) => 'No map link for ' + one + ' ' + k + '.',",
        "notOpened: (why) => 'The map was not opened: ' + why + '.',",
    ):
        assert said in words, said


def test_k5_the_line_is_an_alert_when_it_says_a_failure() -> None:
    """K5: the one line starts as an empty polite status (kept in place, so
    its first words are heard); with words it is red and an alert, announced
    at once (as OE.toast says an error); emptied, a polite status again."""
    block = _review_block()
    made = (
        "const mapSaid = OE.el('div', { class: 'small review-map-said',"
        " role: 'status', 'aria-live': 'polite' });"
    )
    assert made in block
    say = _between(block, "  const sayMap = (rank, words = '') => {", "\n  };")
    for text in (
        "mapFor = words ? rank : null;",
        "mapSaid.setAttribute('role', words ? 'alert' : 'status');",
        "mapSaid.setAttribute('aria-live', words ? 'assertive' : 'polite');",
        "mapSaid.className = words ? 'small error review-map-said'"
        " : 'small review-map-said';",
        "mapSaid.textContent = words;",
    ):
        assert text in say, text
    assert "hidden" not in say


def test_k5_the_pane_draws_the_line_right_under_its_buttons() -> None:
    """K5: the open square's pane puts the line right after its row of
    buttons (heading, Previous, Next, Street map, Close), before anything
    else in the pane; it keeps its words only for the square whose press
    left them, and empties for another square or the test scene. While the
    pane stays open the line is never taken out of the page (the round's
    review: put back in as an alert, it is read out again on every redraw,
    a colour picked or the pictures come); everything else is drawn anew
    around it. Closed, the pane empties it (its button is gone), so a square
    opened again shows no earlier failure."""
    pane = _between(_review_block(), "  function render(shown) {", "\n  }\n")
    assert _in_order(
        pane,
        "mapButton(w.rank, REVIEW_WORDS.streetMap, 'small'),",
        "const parts = [top];",
        "if (state.test || mapFor !== w.rank) sayMap(null);",
        "parts.push(mapSaid);",
        "if (state.test) {",
    )
    assert pane.count("parts.push(mapSaid);") == 1
    assert pane.startswith(
        "  function render(shown) {\n"
        "    if (state.open === null) { sayMap(null); pane.hidden = true;"
        " pane.replaceChildren(); return; }\n"
    )
    assert _in_order(
        pane,
        "pane.hidden = false;",
        "if (mapSaid.parentNode === pane) {",
        "for (const c of [...pane.childNodes]) if (c !== mapSaid) c.remove();",
        "const at = parts.indexOf(mapSaid);",
        "mapSaid.before(...parts.slice(0, at));",
        "mapSaid.after(...parts.slice(at + 1));",
        "} else pane.replaceChildren(...parts);",
        "if (had) {",
    )
    assert _code(pane).count("pane.replaceChildren(...parts)") == 1


def test_k5_the_points_row_has_the_line_right_under_it() -> None:
    """K5: points without picture cards: the line right after their row of
    Street map buttons, before the CSV button; the line is the block's one
    (one per card)."""
    block = _review_block()
    row = _between(block, "    if (n && linkKey) {", "    return section;")
    assert (
        "windows.map((w) => mapButton(w.rank, String(w.rank), 'small'))),"
        " mapSaid, csvBlock());" in row
    )
    assert _code(block).count("const mapSaid = ") == 1


def test_k5_the_line_sits_close_and_takes_no_room_while_empty() -> None:
    """K5: the line 4 px under its row (a card's lines are 12 apart), and
    no room at all while it is empty, never display none (a live region
    hidden until its words arrive is often not announced)."""
    page = _read("result.html")
    style = _between(page, "<style>", "</style>")
    assert "  .stack > .review-map-said { margin-top: 4px; }\n" in style
    assert "  .stack > .review-map-said:empty { margin-top: 0; }\n" in style
    assert not re.search(r"review-map-said[^{]*\{[^}]*display:\s*none", style)
