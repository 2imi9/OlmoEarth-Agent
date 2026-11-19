# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Round r5-m (19 Nov 2026), track RESULT, as specs/r5-m-round.md gives it:
B4 of the build list of the audit after r5-l (specs/r5-l-audit/build_list.md),
a follow-up of round r5-k K1.

- B4: the withdrawn card's one line ends with how to get a new result: "The
  answers on this sheet changed after its first result, so this result does
  not count. A new result needs another person and a fresh sheet; ask Claude
  for one." It said no next step, and the card has no button (17d: all three
  result lenses missed the step). The server's WITHDRAWN_LINE and
  result.html's RESULT_WORDS.withdrawn say it word for word, each marked
  "WORDING: r5k K1; r5m B4 (K1 follow-up)". Unchanged: the card's title, its
  Details, the sheet's graded notice and what Claude says first
  (SAY_WITHDRAWN, which does not tell the user to ask Claude); its note
  (GRADED_AGAIN_NOTE) now names how to plan a fresh sheet, only at the
  user's ask: the same plan with another seed, since the same seed and
  count open this locked sheet again. The card still shows no rate and no
  range, Claude is still told to state none, and no number changes.

The views are read as text (no test here runs the scripts). The browser
harness (scripts/mcp_views_e2e.py) draws the card in
17d_estimate_graded_again, whose steps (result_again_steps) read the line
whole and its last words.
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
VERSION = "v" + "b" * 16

#: The first grading's time, as the graded mark keeps it; and this grading's
#: own (the mark's last_at), as the card writes it.
FIRST_RAW = "2026-11-13T10:04:00Z"
LAST_RAW = "2026-11-18T09:12:00Z"
LAST_AT = "18 Nov 2026, 09:12 UTC"

#: The graded mark of a sheet graded once (calm), and of one whose answers
#: changed after its first result (again).
CALM = (VERSION, {"first_at": FIRST_RAW, "changed": False, "gradings": 1})
AGAIN = (
    VERSION,
    {"first_at": FIRST_RAW, "last_at": LAST_RAW, "changed": True, "gradings": 2},
)

#: Round r5-k K1's line, and B4's: the same words, then how to get a new
#: result in place of the full stop.
K1_LINE = (
    "The answers on this sheet changed after its first result, so this result "
    "does not count. A new result needs another person and a fresh sheet."
)
NEXT = "; ask Claude for one."
LINE = (
    "The answers on this sheet changed after its first result, so this result "
    "does not count. A new result needs another person and a fresh sheet; ask "
    "Claude for one."
)
#: B4's mark, on the server's line and on the page's.
MARK = "WORDING: r5k K1; r5m B4 (K1 follow-up)"
#: B4's mark on Claude's note, which names how to plan a fresh sheet.
NOTE_MARK = "WORDING: r4p2; r5k K1; r5m B4 (K1 follow-up)"

#: What Claude says first, as r5-k K1 wrote it (unchanged): the West part's
#: result (a map of two parts), and a map of one part's.
SAY_WEST = (
    "This result for the West part is out of date and withdrawn: the answers on "
    "this sheet changed after its first result. A new result needs another "
    "person and a fresh sheet."
)
SAY_ONE = (
    "This result is out of date and withdrawn: the answers on this sheet changed "
    "after its first result. A new result needs another person and a fresh sheet."
)
OUT_OF_DATE = "This result is out of date"
#: The West part's headline at 12.5% (server.HEADLINE), and in Details,
#: marked out of date with the time this grading computed it.
HEADLINE = (
    "The West part is wrong on about 1 square in 8: 12.5%, 95% range 6.1% to 21.7%."
)
OLD_RATE = (
    "Out of date, do not use (computed 18 Nov 2026, 09:12 UTC from the changed "
    "answers): " + HEADLINE
)

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


def _inner() -> dict[str, Any]:
    """An ok estimate's result as the tool returns it: 80 labels on a random
    design, 12.5% wrong, its valid interval 6.1% to 21.7%."""
    return {
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
        "map_name": "Map Q2",
        "period": {"start": "2024-04-01T00:00:00Z", "end": "2024-06-30T00:00:00Z"},
        "parts": [this, other],
    }


def _stand_in(monkeypatch: pytest.MonkeyPatch, root: Path) -> None:
    """The sheet's functions stood in for a direct ``_estimate_result`` on
    the West part's sheet: no counts, no squares, no other part graded, no
    pictures, no labelling rule, and "Estimate the error rate" as its finish
    button (sheet.sheet_progress's finish_label)."""
    ctx = _context(root)
    monkeypatch.setattr(server.check, "sheet_context", lambda p: copy.deepcopy(ctx))
    monkeypatch.setattr(server.imagery, "served_any", lambda p: False)
    monkeypatch.setattr(sheet, "labelling_rule", lambda p: None, raising=False)
    monkeypatch.setattr(sheet, "graded_state", lambda p: None, raising=False)
    monkeypatch.setattr(sheet, "graded_squares", lambda p, v: None, raising=False)
    monkeypatch.setattr(sheet, "class_counts", lambda p, v: None, raising=False)
    monkeypatch.setattr(
        sheet,
        "sheet_progress",
        lambda p: {"finish_label": "Estimate the error rate"},
        raising=False,
    )


def _result(root: Path, graded: Any) -> dict[str, Any]:
    """``_estimate_result``'s answer for the West part's sheet with the
    graded mark ``graded``."""
    got = server._estimate_result(_inner(), str(root / "west_to_label.csv"), graded)
    assert got is not None
    return got


async def _estimate_call(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    grading: dict[str, Any],
    *,
    with_views: bool,
) -> tuple[list[str], dict[str, Any] | None]:
    """The estimate through the tool's call on the West part's sheet, graded
    with ``grading`` (what sheet.record_grading answers), with panels or
    without: the text blocks Claude reads, and the result view (``None``
    when the call carries none)."""
    sheet_path = root / "west_to_label.csv"
    header = "order,window_index,wrong,reference_class\n"
    sheet_path.write_text(header, encoding="utf-8")
    app = server.AppServer(None, keys=auth.KeyStore(), base_url=BASE)
    envelope = {"ok": True, "result": _inner()}

    async def run(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """The agent's tool, stood in: the ok estimate."""
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
    result = await app.call_agent_tool(ESTIMATE, args, views=with_views)
    await app.aclose()
    texts = [c.text for c in result.content if isinstance(c, types.TextContent)]
    data = (result.meta or {}).get(server.VIEW_DATA_KEY) or {}
    return texts, data.get("result")


# --------------------------------------------------------------------------- B4


def test_b4_the_servers_line_says_how_to_get_a_new_result() -> None:
    """B4: the server's line is K1's with "; ask Claude for one." in place
    of its full stop, marked as K1's follow-up; it still names no button
    (the card has none: the sheet's finish button pressed again gives this
    card again) and says no rate and no range."""
    assert server.WITHDRAWN_LINE == LINE
    assert LINE == K1_LINE.removesuffix(".") + NEXT
    assert server.WITHDRAWN_LINE.endswith("a fresh sheet" + NEXT)
    for absent in (
        "Press",
        "press",
        "Estimate the error rate",
        "Certify the zone",
        "%",
        "range",
    ):
        assert absent not in server.WITHDRAWN_LINE, absent
    assert "# " + MARK + "\nWITHDRAWN_LINE = (\n" in _server_source()


def test_b4_the_page_says_the_servers_words() -> None:
    """B4: result.html's RESULT_WORDS.withdrawn is the server's line word for
    word, with the same mark; the withdrawn card draws it as its one line and
    has no button of its own, so asking Claude is the way it names."""
    page = _read("result.html")
    words = _between(page, "const RESULT_WORDS = {", "\n};")
    assert "  // " + MARK + "\n  withdrawn: '" in words
    assert _concat(_entry(words, "  withdrawn: "), {}) == LINE
    gone = _between(page, "function withdrawnCard(data) {", "\n}\n")
    drawn = (
        "card.append(OE.el('p', { class: 'oe-withdrawn-line' },"
        " RESULT_WORDS.withdrawn));"
    )
    assert drawn in gone
    for absent in ("'button'", "OE.sendJob(", "OE.tellClaude", "primary"):
        assert absent not in gone, absent


def test_b4_what_claude_says_first_is_unchanged() -> None:
    """B4 leaves the line Claude says first as r5-k K1 wrote it (the card's
    title and that a new result needs another person and a fresh sheet,
    with no "ask Claude": Claude says it), still marked r5k K1, with its
    note; Claude is still told to state no rate and no range from the
    result; the card's title is unchanged."""
    assert server.SAY_WITHDRAWN.format(part=server._for_part("West part")) == SAY_WEST
    assert server.SAY_WITHDRAWN.format(part=server._for_part(None)) == SAY_ONE
    assert "Claude" not in server.SAY_WITHDRAWN
    source = _server_source()
    assert "# WORDING: r5k K1\nSAY_WITHDRAWN = (\n" in source
    assert "# WORDING: r5k K1\nSAY_WITHDRAWN_NOTE = (\n" in source
    first = server.SAY_WITHDRAWN_NOTE
    assert "state no item of its must_state and no rate or range from it" in first
    note = server.GRADED_AGAIN_NOTE.format(first_at=FIRST_RAW)
    assert "State no rate and no range from it" in note
    assert server.RESULT_OUT_OF_DATE == OUT_OF_DATE


def test_b4_claudes_note_says_how_to_plan_a_fresh_sheet() -> None:
    """B4: the card's line says to ask Claude for a new result, so Claude's
    note (GRADED_AGAIN_NOTE) names how to plan one, only when the user asks:
    olmoearth_plan_label_sample with the plan's own arguments and another
    seed, for a zone the Check card drew too (a plan is deterministic and
    keeps a sheet that holds labels, so the same seed and count open this
    locked sheet again). It still says to state no rate and no range and
    overrides every must_state item; what follows that holds no number and
    no range (read on the template: the formatted {first_at} has digits)."""
    template = server.GRADED_AGAIN_NOTE
    assert server.PLAN_TOOL == "olmoearth_plan_label_sample"
    for said in (
        "If the user asks for a new result, call olmoearth_plan_label_sample ",
        "a seed other than this sheet's seed",
        "a zone the Check card drew too",
        "open this locked sheet again",
        "only when the user asks",
    ):
        assert said in template, said
    assert "State no rate and no range from it" in template
    assert "this overrides every must_state item" in template
    after = template.split("State no rate and no range from it", 1)[1]
    assert re.search(r"\d", after) is None, after
    assert "%" not in after and "range" not in after and "{" not in after, after
    assert "# " + NOTE_MARK + "\nGRADED_AGAIN_NOTE = (\n" in _server_source()


def test_b4_a_changed_sheets_view_carries_the_line_and_every_number(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """B4 through ``_estimate_result``: a sheet whose answers changed after
    its first result gives the withdrawn block with B4's line (the title,
    what was checked and the rate marked out of date as before); every other
    field and number is a calm sheet's (no number changes); Claude says the
    unchanged line first, with no rate, under the withdrawn note. A calm
    sheet's notes name no plan of a fresh sheet."""
    _stand_in(monkeypatch, root)
    calm = _result(root, CALM)
    got = _result(root, AGAIN)
    view = got["view"]
    assert view["withdrawn"] == {
        "title": OUT_OF_DATE,
        "line": LINE,
        "of": "West part of Map Q2",
        "computed_at": LAST_AT,
        "old_rate": OLD_RATE,
    }
    drop = ("graded_before", "withdrawn")
    kept = {k: v for k, v in view.items() if k not in drop}
    assert kept == {k: v for k, v in calm["view"].items() if k not in drop}
    assert calm["view"]["withdrawn"] is None
    assert got["say_first"] == [SAY_WEST]
    assert "%" not in " ".join(got["say_first"])
    assert got["note"] == server.SAY_WITHDRAWN_NOTE
    assert got["notes"] == [server.GRADED_AGAIN_NOTE.format(first_at=FIRST_RAW)]
    assert not any(server.PLAN_TOOL in n for n in calm["notes"])


@pytest.mark.asyncio
@pytest.mark.parametrize("with_views", [True, False])
async def test_b4_claude_states_no_rate_with_panels_or_without(
    root: Path, monkeypatch: pytest.MonkeyPatch, with_views: bool
) -> None:
    """B4 through the tool's call, with panels and without: the first text
    block Claude reads holds the unchanged withdrawn line (no "ask Claude")
    with the withdrawn note, the next the note to state no rate and no
    range, which names how to plan a fresh sheet; with panels the view
    carries B4's line and keeps every number."""
    texts, view = await _estimate_call(
        root, monkeypatch, AGAIN[1], with_views=with_views
    )
    first = json.loads(texts[0])
    assert first["say_first"] == [SAY_WEST]
    assert first["note"] == server.SAY_WITHDRAWN_NOTE
    assert texts[1] == server.GRADED_AGAIN_NOTE.format(first_at=FIRST_RAW)
    assert server.PLAN_TOOL in texts[1]
    assert "ask Claude" not in " ".join(first["say_first"])
    if with_views:
        assert view is not None
        assert view["withdrawn"]["line"] == LINE
        assert (view["rate"], view["low"], view["high"]) == (0.125, 0.061, 0.217)


def test_b4_the_changelog_and_the_usage_say_the_line() -> None:
    """B4: the 1.5.0 detail file's marker (spec/changes/1.5.0.md) is replaced
    by the round's line, which
    gives the new words; spec/Usage.md gives them as the withdrawn card's
    line, and no longer K1's line that ended at "a fresh sheet."; both say
    Claude plans a new sample with another seed, since the same seed and
    count open the locked sheet again."""
    stage = Path(__file__).resolve().parents[2]
    changelog = (stage / "spec" / "changes" / "1.5.0.md").read_text(encoding="utf-8")
    assert "@@R5M-RESULT@@" not in changelog
    line = "\n    - B4: a withdrawn result's line ends with how to get a new result: "
    assert line in changelog
    assert '"' + LINE + '"' in changelog
    usage = (stage / "spec" / "Usage.md").read_text(encoding="utf-8")
    flat = " ".join(usage.split())
    assert '"' + LINE + '"' in flat
    assert '"' + K1_LINE + '"' not in flat
    for text in (changelog, flat):
        assert "`olmoearth_plan_label_sample` with the same map, design" in text
        assert "since the same seed and count" in text
