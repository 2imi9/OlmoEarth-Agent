# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""The figure-first Check and result cards: what the server gives them.

The Check card's map name, period, parts, square size and labelling-rule
gate (the press refused without a rule, the rule pinned and reused); the
estimate's result card (the rate in words beside the exact rate and range,
the squares by class with the tool's own ranges, the parts, the graded
mark, the fold's rows, the last line) and the lines the model says first;
the speech rounding; rule 9 in the instructions; a review set's reasons for
no pictures, its count and its Check request; the review pane's test scene;
the views' figures; the Studio refusal that names the plan tool. Sheet
functions another module provides are stood in for; no network.
"""

from __future__ import annotations

import copy
import json
import math
import re
import unicodedata
from pathlib import Path
from typing import Any

import pytest

mcp = pytest.importorskip("mcp")
from mcp import types  # noqa: E402

from olmoearth_agent.mcp_app import (  # noqa: E402
    auth,
    check,
    review,
    server,
    sheet,
    views,
)
from olmoearth_agent.skills import build_default_registry  # noqa: E402
from olmoearth_agent.tools import statistical_rules as rules  # noqa: E402
from olmoearth_agent.tools import studio_scores  # noqa: E402

BASE = "http://mock-studio/api/v1"
R = "11111111-2222-4333-8444-555555555555"
P = "99999999-8888-4777-8666-555555555555"
FILES = ("result_epsg32610_0_native.tif", "result_epsg32611_1_native.tif")
CANNED = {
    "offer": [
        {"labels": 20, "width": 0.2, "sentence": "Twenty labels.", "rate_note": "r50"},
        {"labels": 40, "width": 0.1, "sentence": "Forty labels.", "rate_note": "r50"},
    ],
    "refused": None,
}
ESTIMATE = "olmoearth_estimate_map_error"
METHOD = (
    "simple random sample; valid at every sample size: one stratum is labelled in "
    "part, and the interval is its exact hypergeometric interval at 95%"
)
TYPED = "Kelp: canopy fills more than half of the square."
#: check.sheet_context itself, before any test stands in for it.
REAL_SHEET_CONTEXT = check.sheet_context


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


@pytest.fixture
def canned(monkeypatch: pytest.MonkeyPatch) -> None:
    """Counts of 20 and 40 offered for every zone, without the package."""

    def fake(n_valid: int, widths: Any = check.CHECK_WIDTHS) -> dict[str, Any]:
        """A fresh copy of :data:`CANNED`."""
        return copy.deepcopy(CANNED)

    monkeypatch.setattr(check, "offers", fake)


def _app() -> server.AppServer:
    return server.AppServer(None, keys=auth.KeyStore(), base_url=BASE)


def _texts(result: types.CallToolResult) -> list[str]:
    return [c.text for c in result.content if isinstance(c, types.TextContent)]


def _view(result: types.CallToolResult) -> dict[str, Any]:
    return (result.meta or {})[server.VIEW_DATA_KEY]


def _words(text: str) -> str:
    return " ".join(text.split())


# --------------------------------------------------------------------------- speech rounding


@pytest.mark.parametrize(
    ("rate", "said"),
    [
        (0, "The map was right on every square you checked."),
        (0.125, "The map is wrong on about 1 square in 8."),
        # Round r5-b, J01: 0.4 and 0.3 are nearer "N in 10" than "1 in 3".
        (0.4, "The map is wrong on about 4 squares in 10."),
        (0.5, "The map is wrong on about 1 square in 2."),
        (0.3, "The map is wrong on about 3 squares in 10."),
        (0.0004, "The map is wrong on about 1 square in 2,500."),
        (0.6, "The map is wrong on about 6 squares in 10."),
        (0.96, "The map is wrong on more than 9 squares in 10."),
        (1, "The map was wrong on every square you checked."),
        (None, None),
        (math.nan, None),
        (-0.1, None),
        (1.2, None),
        (True, None),
        ("0.1", None),
        (5e-324, None),
    ],
)
def test_say_rate(rate: Any, said: str | None) -> None:
    """One everyday sentence per rate; never one for no rate. Up to 0.5 the
    nearer of "1 in N" and "N in 10" (round r5-b, J01)."""
    assert server.say_rate(rate) == said


@pytest.mark.parametrize(
    ("rate", "square", "point"),
    [
        (0.5, "about 1 square in 2.", "about 1 point in 2."),
        (0.51, "about 5 squares in 10.", "about 5 points in 10."),
        (0.94, "about 9 squares in 10.", "about 9 points in 10."),
        (0.949, "about 9 squares in 10.", "about 9 points in 10."),
        (0.95, "more than 9 squares in 10.", "more than 9 points in 10."),
        (0.999, "more than 9 squares in 10.", "more than 9 points in 10."),
    ],
)
def test_say_rate_near_the_top_never_rounds_to_ten(
    rate: float, square: str, point: str
) -> None:
    """The peer's item 3: above 0.5 "about N in 10" never says 10; from 0.95
    up to (not at) 1, "more than 9 in 10"; in squares or in points."""
    assert server.say_rate(rate) == "The map is wrong on " + square
    assert server.say_rate(rate, "square") == "The map is wrong on " + square
    assert server.say_rate(rate, "point") == "The map is wrong on " + point
    assert "10 squares" not in str(server.say_rate(rate))


def test_say_rate_at_zero_and_one_in_points() -> None:
    """Exactly 1 mirrors the zero sentence, in either unit; an unknown unit
    word is a square."""
    say = server.say_rate
    assert say(1.0, "point") == "The map was wrong on every point you checked."
    assert say(0, "point") == "The map was right on every point you checked."
    assert say(1.0) == "The map was wrong on every square you checked."
    assert say(0.125, "point") == "The map is wrong on about 1 point in 8."
    assert say(0.125, "window") == "The map is wrong on about 1 square in 8."
    assert server.SAY_OVER_NINE_FROM == 0.95
    assert not hasattr(server, "SAY_IN_TEN_MAX")


def test_say_rate_names_the_part_as_its_subject() -> None:
    """r4-h R1: on a map in parts the sentence's subject is the part ("The
    West part", "Part 2"), so it cannot be passed on as the whole map's;
    "The map" when no part is named."""
    say = server.say_rate
    assert say(0.125, "square", "West part") == (
        "The West part is wrong on about 1 square in 8."
    )
    assert say(0.125, "point", "Part 2") == "Part 2 is wrong on about 1 point in 8."
    assert say(0, "square", "East part") == (
        "The East part was right on every square you checked."
    )
    assert say(1.0, "square", "Part 3") == (
        "Part 3 was wrong on every square you checked."
    )
    assert say(0.96, "square", "West part") == (
        "The West part is wrong on more than 9 squares in 10."
    )
    assert say(0.6, "square", None) == "The map is wrong on about 6 squares in 10."
    assert server.SAY_SUBJECT_MAP == "The map"


def test_the_period_and_times_in_words() -> None:
    """Dates as given, as the cards write them."""
    window = {"start": "2024-04-01T00:00:00Z", "end": "2024-06-30T00:00:00Z"}
    assert server._period_text(window) == "1 Apr 2024 to 30 Jun 2024"
    assert server._period_text({"start": "2024-04-01"}) is None
    assert server._period_text({"start": "soon", "end": "2024-06-30"}) is None
    assert server._period_text(None) is None
    assert server._when_text("2026-11-13T10:04:00Z") == "13 Nov 2026, 10:04 UTC"
    assert server._when_text("2026-11-13") == "13 Nov 2026"
    assert server._when_text("later") == "later"
    assert server._when_text(None) is None


# --------------------------------------------------------------------------- the instructions


def test_rule_9_follows_the_first_working_rule_inside_the_part_kept() -> None:
    """The reader test's B/C1 rule follows the soul's first working rule; the
    wizard's step list makes room; rule 9, the review route and the end of
    the estimate rule each end before character 4,000 (the cut is about
    4,100); the projects panel's note carries the rule too."""
    text = server.server_instructions(build_default_registry())
    first = text.index(server._FIRST_RULE_END) + len(server._FIRST_RULE_END)
    assert text[first : first + 3] == "\n- "
    assert text.index(server.RULE_9) == first + 3
    assert "(model type" not in text
    assert "walk its wizard. Do NOT talk" in _words(text)
    for end in (
        server.RULE_9,
        server.SOUL_REVIEW_ROUTE,
        "does not apply to a stratified design.",
    ):
        assert text.index(end) + len(end) < 4000, end
    assert "never call one recommended" in server.RULE_9
    assert server.RULE_9 in server.PROJECTS_PANEL_NOTE
    assert server.PROJECTS_PANEL_NOTE.endswith(" " + server.CONTEXT_RULE)


# --------------------------------------------------------------------------- the Check card


def _studio(root: Path, labels: tuple[str, ...] = ("water", "trees")) -> dict[str, Any]:
    """olmoearth_scores_from_studio's envelope for a map in two UTM zones, 10N
    and 11N, with these classes and the period."""
    zones = []
    for i, file in enumerate(FILES):
        path = root / f"zone{i}_scores.json"
        path.write_text(json.dumps({"scores": [[0.2, 0.8]] * 30}), "utf-8")
        zones.append(
            {
                "file": file,
                "scores_path": str(path),
                "pooled": {"n_valid": 300, "raster_check": {"sha256": str(i) * 64}},
            }
        )
    shares = [{"file": FILES[0], "share": 0.467}, {"file": FILES[1], "share": 0.533}]
    inner = {
        "available": True,
        "result_id": R,
        "prediction_id": P,
        "kind": "topk",
        "property_name": "landcover",
        "classes": [
            {"channel": c, "value": c, "label": n} for c, n in enumerate(labels)
        ],
        "date_window": {"start": "2024-04-01T00:00:00Z", "end": "2024-06-30T00:00:00Z"},
        "zones": zones,
        "facts": [{"id": "studio_zones", "sentence": "Two.", "zones": shares}],
        "must_state": ["Zones are parts."],
        "forbidden_claims": [],
    }
    return {"ok": True, "result": inner}


def _fake_tools(
    app: server.AppServer,
    monkeypatch: pytest.MonkeyPatch,
    studio: dict[str, Any],
    root: Path,
) -> list[tuple[str, dict[str, Any]]]:
    """``app.run_agent_tool`` answering ``studio`` for the files and a drawn
    plan (a sheet per scores file) for a plan; the calls, in order."""
    calls: list[tuple[str, dict[str, Any]]] = []

    async def run(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        calls.append((name, dict(arguments)))
        if name == server.SCORES_TOOL:
            return copy.deepcopy(studio)
        stem = Path(str(arguments.get("scores_path"))).stem
        return {
            "ok": True,
            "result": {
                "available": True,
                "labels_csv_path": str(root / f"{stem}_to_label.csv"),
                "design_path": str(root / f"{stem}.json"),
                "n_population": 300,
                "must_state": ["Stated with the plan."],
            },
        }

    monkeypatch.setattr(app, "run_agent_tool", run)
    return calls


class _Studio:
    """A Studio client that knows one prediction's name."""

    async def get_prediction(self, prediction_id: str) -> dict[str, Any]:
        assert prediction_id == P
        return {"id": P, "name": " Kelp\u0007   Q2 "}


@pytest.mark.asyncio
async def test_the_check_card_names_the_map_its_parts_and_its_squares(
    root: Path, monkeypatch: pytest.MonkeyPatch, canned: None
) -> None:
    """The view gets the map's name (Studio's, cleaned) and period, how many
    parts, each part's name, share as a fraction and square size; the model
    gets each part's name and its share as a size field (never "share"),
    never the map's name; its first block is the say-first lines, worded
    for the chat without panels (no card is open), whose note asks for the
    counts in squares and parts."""
    app = _app()
    _fake_tools(app, monkeypatch, _studio(root, ("absent", "present")), root)

    async def studio() -> _Studio:
        return _Studio()

    monkeypatch.setattr(app, "studio", studio)
    monkeypatch.setattr(
        server.imagery,
        "window_size",
        lambda path: {"pixels": 11, "metres": 110.0},
        raising=False,
    )
    result = await app.check_map_tool({"result_id": R}, views=True)
    view = _view(result)
    assert view["map"] == {
        "name": "Kelp Q2",
        "period_text": "1 Apr 2024 to 30 Jun 2024",
    }
    assert view["n_parts"] == 2
    assert [z["part"] for z in view["zones"]] == ["West part", "East part"]
    assert [z["share_value"] for z in view["zones"]] == [0.467, 0.533]
    assert [z["square"] for z in view["zones"]] == [{"pixels": 11, "metres": 110.0}] * 2
    # A map of presence and absence: the owner's rule, set, with the box for
    # the map's own while no part is drawn (option A).
    assert view["rule"]["set"] is True and view["rule"]["source"] == "owner"
    assert view["rule"]["replaceable"] is True
    texts = _texts(result)
    assert len(texts) == 2
    assert json.loads(texts[0]) == {
        "say_first": [
            server.SAY_CHECK_OPEN,
            "Pick how many squares to check in each of its 2 parts; more squares give "
            "a narrower range. The count is your choice.",
            "There is no result until every square of a sheet is done; each part gets "
            "its own result, never one rate for all parts. To cover the whole map, "
            "report each part's rate with its range, side by side; do not average or "
            "add them up: this check gives no range for a combined rate.",
        ],
        "note": server.SAY_FIRST_NOTE,
    }
    model = json.loads(texts[-1])
    assert [z["part"] for z in model["zones"]] == ["West part", "East part"]
    assert [z["share_of_pixels_with_data"] for z in model["zones"]] == [
        "46.7%",
        "53.3%",
    ]
    assert all("share" not in z for z in model["zones"])
    assert model["note"].startswith(server.CHECK_CARD_NOTE)
    assert server.SHARE_SIZE_NOTE in model["note"]
    assert "Do not repeat the card; then wait." in model["note"]
    assert server.CHECK_OWNER_RULE_NOTE in model["note"]
    assert server.CHECK_TEXT_OWNER_RULE_NOTE not in model["note"]  # the box asks
    # The owner's rule reaches the model with its scope (Q5).
    assert model["labelling_rule"] == {
        "source": "owner",
        "text": server.imagery.READING_RULE,
        "scope": server.imagery.RULE_SCOPE,
    }
    assert "Kelp" not in "".join(texts)  # the map's Studio name: the panel's only
    plain = await app.check_map_tool({"result_id": R}, views=False)
    assert len(_texts(plain)) == 2
    assert json.loads(_texts(plain)[0]) == {
        "say_first": [
            "You check some small squares of this map yourself, picked at random.",
            "Pick how many squares to check in each of its 2 parts; more squares give "
            "a narrower range. The count is your choice.",
            "There is no result until every square of a sheet is done; each part gets "
            "its own result, never one rate for all parts. To cover the whole map, "
            "report each part's rate with its range, side by side; do not average or "
            "add them up: this check gives no range for a combined rate.",
        ],
        "note": server.SAY_FIRST_NOTE,
    }
    told = json.loads(_texts(plain)[-1])["note"]
    assert told.startswith(server.CHECK_TEXT_NOTE)
    assert server.CHECK_OWNER_RULE_NOTE not in told  # no card, no box
    # Without panels, a map of presence and absence: the model asks whether
    # the map is of kelp before any plan, and for its own rule if not (Q5).
    assert server.CHECK_TEXT_OWNER_RULE_NOTE in told
    assert server.CHECK_TEXT_OWNER_RULE_NOTE == (
        "The app owner's labelling rule is for kelp canopy maps. Before any plan, "
        "ask the user whether this map is of kelp; if it is not, ask for the map's "
        "own rule in one sentence and repeat that rule with the result instead of "
        "the kelp rule."
    )
    assert json.loads(_texts(plain)[-1])["labelling_rule"] == model["labelling_rule"]
    assert server.SHARE_SIZE_NOTE in told
    assert "how many squares they can check in each part" in told
    assert "only when the user asks" in told
    assert "Kelp" not in "".join(_texts(plain))
    await app.aclose()


def test_one_part_says_no_parts() -> None:
    """A map of one part: the say-first lines name no part; without panels
    the first line names no card."""
    assert server._check_say_first(1) == [
        server.SAY_CHECK_OPEN,
        "Pick how many squares to check; more squares give a narrower range. The "
        "count is your choice.",
        "There is no result until every square of a sheet is done.",
    ]
    chat = server._check_say_first(1, views=False)
    assert chat[0] == server.SAY_CHECK_CHAT and "Check card" not in chat[0]
    assert chat[1:] == server._check_say_first(1)[1:]


@pytest.mark.asyncio
async def test_a_map_name_that_cannot_be_read_is_left_out(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Signed out, or an id that is not a UUID: no name, no failure."""
    app = _app()
    assert await app._map_name(P) is None  # signed out: Studio refuses
    assert await app._map_name("../x") is None
    assert await app._map_name(None) is None
    await app.aclose()


@pytest.mark.asyncio
async def test_the_rule_gate_refuses_a_press_without_a_rule_then_pins_one(
    root: Path, monkeypatch: pytest.MonkeyPatch, canned: None
) -> None:
    """A map with no rule of the owner's: the card asks for one (the view's
    rule not set, the model's note says so); a press without a rule is
    refused whole, drawing and logging nothing; a press whose rule the
    server refuses (too short, a tab) is refused whole with that refusal's
    own words, never as if no rule were sent; a press with the user's
    sentence (a pasted no-break space and zero-width space in it) draws,
    pins it as kept and writes it beside the sheet; the card then shows it
    read only, and another part's press with no usable rule takes the
    pinned one."""
    app = _app()
    _fake_tools(app, monkeypatch, _studio(root), root)

    def clean(text: Any) -> str:
        """sheet.clean_rule as it behaves: space separators made plain spaces
        and format characters dropped, then one printable line of 10 to 300
        characters (no tab or line break), else its refusal."""
        if not isinstance(text, str):
            raise ValueError(sheet.RULE_TEXT)
        kept = "".join(
            " " if unicodedata.category(ch) == "Zs" else ch
            for ch in text
            if unicodedata.category(ch) != "Cf"
        ).strip()
        if not 10 <= len(kept) <= 300 or not kept.isprintable():
            raise ValueError(sheet.RULE_TEXT)
        return kept

    typed: list[tuple[str, str, str]] = []

    def set_typed(path: str, text: str, *, typed_at: str) -> dict[str, Any]:
        """sheet.set_typed_rule, recorded."""
        typed.append((path, text, typed_at))
        return {"text": text, "typed_at": typed_at}

    monkeypatch.setattr(sheet, "clean_rule", clean, raising=False)
    monkeypatch.setattr(sheet, "set_typed_rule", set_typed, raising=False)
    card = await app.check_map_tool({"result_id": R}, views=True)
    view = _view(card)
    assert view["rule"] == {
        "set": False,
        "text": None,
        "source": None,
        "scope": None,
        "replaceable": False,
    }
    assert check.read_offer(view["zones"][0]["check_key"])["rule_owner"] is False
    model = json.loads(_texts(card)[-1])
    assert server.CHECK_NO_RULE_NOTE in model["note"]
    assert server.CHECK_OWNER_RULE_NOTE not in model["note"]
    assert model["labelling_rule"] is None
    west, east = (z["check_key"] for z in view["zones"])
    with pytest.raises(ValueError, match="No labelling rule is set for this map"):
        await app.app_tool(
            "olmoearth_app_check_draw", {"draws": [{"check_key": west, "budget": 20}]}
        )
    assert not (root / "checks" / f"{west}.draw.json").exists()
    assert not (root / "checks" / check.LOG_NAME).exists()
    # A rule the server refuses, and no earlier part pinned one: refused with
    # the refusal's own words (the rule's bounds), never "no rule is set".
    for refused in ("short", "Kelp canopy\tfills more than half"):
        with pytest.raises(ValueError, match=re.escape(sheet.RULE_TEXT)):
            await app.app_tool(
                "olmoearth_app_check_draw",
                {"draws": [{"check_key": west, "budget": 20}], "reading_rule": refused},
            )
    assert not (root / "checks" / f"{west}.draw.json").exists()
    assert not (root / "checks" / check.LOG_NAME).exists()
    # Pasted from a document: a no-break space and a zero-width space.
    no_break, zero_width = chr(0x00A0), chr(0x200B)
    pasted = TYPED.replace(" ", no_break, 1).replace("canopy", f"can{zero_width}opy")
    out = await app.app_tool(
        "olmoearth_app_check_draw",
        {"draws": [{"check_key": west, "budget": 20}], "reading_rule": f"  {pasted}  "},
    )
    (drawn,) = out["draws"]
    assert (drawn["outcome"], drawn["state"]) == ("first", "drawn")
    record = json.loads((root / "checks" / f"{west}.draw.json").read_text("utf-8"))
    assert record["reading_rule"] == TYPED
    assert record["rule_typed_at"] == record["pressed_at"]
    assert typed == [(drawn["labels_csv_path"], TYPED, record["pressed_at"])]
    again = await app.check_map_tool({"result_id": R}, views=True)
    day = server._day_text(record["pressed_at"])
    assert _view(again)["rule"] == {
        "set": True,
        "text": TYPED,
        "source": "typed",
        "scope": f"Typed by you before the squares were picked, {day}.",
        "replaceable": False,
    }
    told = json.loads(_texts(again)[-1])
    assert server.CHECK_NO_RULE_NOTE not in told["note"]
    assert told["labelling_rule"] == {"source": "typed", "text": TYPED}
    other = await app.app_tool(
        "olmoearth_app_check_draw",
        {"draws": [{"check_key": east, "budget": 40}], "reading_rule": "bad"},
    )
    assert other["draws"][0]["state"] == "drawn"
    pinned = json.loads((root / "checks" / f"{east}.draw.json").read_text("utf-8"))
    assert pinned["reading_rule"] == TYPED
    assert [t[1] for t in typed] == [TYPED, TYPED]
    await app.aclose()


def _log_lines(root: Path) -> int:
    """How many lines the checks' log holds (0 with no log)."""
    log = root / "checks" / check.LOG_NAME
    return len(log.read_text("utf-8").splitlines()) if log.exists() else 0


def _typed_beside(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str, str]]:
    """sheet.set_typed_rule stood in for (the fake plan writes no sheet): the
    rules written beside sheets, in order."""
    typed: list[tuple[str, str, str]] = []

    def set_typed(path: str, text: str, *, typed_at: str) -> dict[str, Any]:
        typed.append((path, text, typed_at))
        return {"text": text, "typed_at": typed_at}

    monkeypatch.setattr(sheet, "set_typed_rule", set_typed, raising=False)
    return typed


@pytest.mark.asyncio
async def test_an_empty_box_keeps_the_owners_rule_and_a_part_drawn_settles_it(
    root: Path, monkeypatch: pytest.MonkeyPatch, canned: None
) -> None:
    """Option A, a map of presence and absence: the card shows the owner's
    kelp rule and its scope with the box (replaceable; the offer records
    rule_owner and no rule_needed). A press with the box empty draws with the
    kelp rule: nothing pinned, nothing written beside the sheet; its reply
    carries the map's rule as the card now shows it (Q4). Once a part is
    drawn the rule is settled: the card shows the kelp rule with no box, and
    a press that still sends a rule draws and logs nothing, its reply the
    refusal's one sentence and the rule (Q4, Q11)."""
    app = _app()
    _fake_tools(app, monkeypatch, _studio(root, ("absent", "present")), root)
    typed = _typed_beside(monkeypatch)
    view = _view(await app.check_map_tool({"result_id": R}, views=True))
    kelp = {
        "set": True,
        "text": server.imagery.READING_RULE,
        "source": "owner",
        "scope": server.imagery.RULE_SCOPE,
        "replaceable": True,
    }
    assert view["rule"] == kelp
    west, east = (z["check_key"] for z in view["zones"])
    offer = check.read_offer(west)
    assert (offer["rule_needed"], offer["rule_owner"]) == (False, True)
    out = await app.app_tool(
        "olmoearth_app_check_draw", {"draws": [{"check_key": west, "budget": 20}]}
    )
    assert out["draws"][0]["state"] == "drawn"
    assert out["rule"] == {**kelp, "replaceable": False}
    record = json.loads((root / "checks" / f"{west}.draw.json").read_text("utf-8"))
    assert "reading_rule" not in record and "rule_typed_at" not in record
    assert typed == []
    assert check.result_settled(R) is True and check.map_rule(R) is None
    again = await app.check_map_tool({"result_id": R}, views=True)
    assert _view(again)["rule"]["source"] == "owner"
    assert _view(again)["rule"]["replaceable"] is False
    assert server.CHECK_OWNER_RULE_NOTE not in json.loads(_texts(again)[-1])["note"]
    logged = _log_lines(root)
    refused = await app.app_tool(
        "olmoearth_app_check_draw",
        {"draws": [{"check_key": east, "budget": 40}], "reading_rule": TYPED},
    )
    assert refused == {
        "draws": [],
        "refused": check.RULE_SETTLED,
        "rule": {**kelp, "replaceable": False},
    }
    assert not (root / "checks" / f"{east}.draw.json").exists()
    assert _log_lines(root) == logged
    assert check.RULE_SETTLED == (
        "This map's labelling rule was set when its first part was picked."
    )
    # The other part, pressed with no rule: the kelp rule again.
    other = await app.app_tool(
        "olmoearth_app_check_draw", {"draws": [{"check_key": east, "budget": 40}]}
    )
    assert other["draws"][0]["state"] == "drawn"
    pinned = json.loads((root / "checks" / f"{east}.draw.json").read_text("utf-8"))
    assert "reading_rule" not in pinned and typed == []
    await app.aclose()


@pytest.mark.asyncio
async def test_a_rule_typed_in_the_box_replaces_the_owners_for_this_map(
    root: Path, monkeypatch: pytest.MonkeyPatch, canned: None
) -> None:
    """Option A: a rule of 1 to 9 characters in the box is refused whole with
    the gate's own words (nothing drawn or logged); a rule of 10 or more is
    cleaned as the gate's is, pinned with the draw and written beside the
    sheet; the card then shows it read only with its typed scope, never the
    owner's scope; another part pressed with no rule takes it; the same rule
    sent again passes (Q11), another one is answered settled with the rule."""
    app = _app()
    _fake_tools(app, monkeypatch, _studio(root, ("absent", "present")), root)
    typed = _typed_beside(monkeypatch)
    view = _view(await app.check_map_tool({"result_id": R}, views=True))
    west, east = (z["check_key"] for z in view["zones"])
    for refused in ("short", "​nine char"):
        with pytest.raises(ValueError, match=re.escape(sheet.RULE_TEXT)):
            await app.app_tool(
                "olmoearth_app_check_draw",
                {"draws": [{"check_key": west, "budget": 20}], "reading_rule": refused},
            )
    assert not (root / "checks" / f"{west}.draw.json").exists()
    assert _log_lines(root) == 0
    out = await app.app_tool(
        "olmoearth_app_check_draw",
        {"draws": [{"check_key": west, "budget": 20}], "reading_rule": f"  {TYPED} "},
    )
    (drawn,) = out["draws"]
    assert drawn["state"] == "drawn"
    record = json.loads((root / "checks" / f"{west}.draw.json").read_text("utf-8"))
    assert record["reading_rule"] == TYPED
    assert record["rule_typed_at"] == record["pressed_at"]
    assert typed == [(drawn["labels_csv_path"], TYPED, record["pressed_at"])]
    again = await app.check_map_tool({"result_id": R}, views=True)
    rule = _view(again)["rule"]
    assert rule == {
        "set": True,
        "text": TYPED,
        "source": "typed",
        "scope": sheet.typed_rule_scope(record["pressed_at"]),
        "replaceable": False,
    }
    assert rule["scope"] != server.imagery.RULE_SCOPE
    told = json.loads(_texts(again)[-1])
    assert told["labelling_rule"] == {"source": "typed", "text": TYPED}
    assert server.CHECK_OWNER_RULE_NOTE not in told["note"]
    other = await app.app_tool(
        "olmoearth_app_check_draw", {"draws": [{"check_key": east, "budget": 40}]}
    )
    assert other["draws"][0]["state"] == "drawn"
    assert other["rule"] == rule  # the reply says the rule the part was drawn with
    pinned = json.loads((root / "checks" / f"{east}.draw.json").read_text("utf-8"))
    assert pinned["reading_rule"] == TYPED
    assert [t[1] for t in typed] == [TYPED, TYPED]
    # The pinned rule sent again (a card that did not hear the press): it
    # passes and repeats the draw; another rule is answered settled.
    same = await app.app_tool(
        "olmoearth_app_check_draw",
        {"draws": [{"check_key": east, "budget": 40}], "reading_rule": f" {TYPED}"},
    )
    assert same["draws"][0]["outcome"] == "repeat" and same["rule"] == rule
    other_rule = await app.app_tool(
        "olmoearth_app_check_draw",
        {
            "draws": [{"check_key": east, "budget": 40}],
            "reading_rule": "Seagrass: leaves cover most of the square.",
        },
    )
    assert other_rule == {"draws": [], "refused": check.RULE_SETTLED, "rule": rule}
    await app.aclose()


@pytest.mark.asyncio
async def test_an_offer_without_rule_owner_keeps_the_owners_rule(
    root: Path, monkeypatch: pytest.MonkeyPatch, canned: None
) -> None:
    """An offer record of a map of presence and absence written before the box
    (no rule_owner): no box, so a rule the press sends is not taken; the
    owner's rule stands."""
    app = _app()
    _fake_tools(app, monkeypatch, _studio(root, ("absent", "present")), root)
    typed = _typed_beside(monkeypatch)
    view = _view(await app.check_map_tool({"result_id": R}, views=True))
    west = view["zones"][0]["check_key"]
    place = root / "checks" / f"{west}.json"
    record = json.loads(place.read_text("utf-8"))
    del record["rule_owner"]
    place.write_text(json.dumps(record), "utf-8")
    out = await app.app_tool(
        "olmoearth_app_check_draw",
        {"draws": [{"check_key": west, "budget": 20}], "reading_rule": TYPED},
    )
    assert out["draws"][0]["state"] == "drawn"
    drawn = json.loads((root / "checks" / f"{west}.draw.json").read_text("utf-8"))
    assert "reading_rule" not in drawn and typed == []
    await app.aclose()


@pytest.mark.asyncio
async def test_an_offer_written_before_the_rule_field_refuses_the_press(
    root: Path, monkeypatch: pytest.MonkeyPatch, canned: None
) -> None:
    """An offer record written before the offers said whether the map has a
    labelling rule (no rule_needed, a card shown before the update): a press,
    with a rule or without, is refused whole with the sentence to open the
    card again, which writes the field; nothing is planned, drawn or logged."""
    app = _app()
    calls = _fake_tools(app, monkeypatch, _studio(root), root)
    view = _view(await app.check_map_tool({"result_id": R}, views=True))
    west = view["zones"][0]["check_key"]
    place = root / "checks" / f"{west}.json"
    record = json.loads(place.read_text("utf-8"))
    del record["rule_needed"]
    place.write_text(json.dumps(record), "utf-8")
    for extra in ({}, {"reading_rule": TYPED}):
        with pytest.raises(ValueError, match=re.escape(check.OLDER_CARD)):
            await app.app_tool(
                "olmoearth_app_check_draw",
                {"draws": [{"check_key": west, "budget": 20}], **extra},
            )
    assert not (root / "checks" / f"{west}.draw.json").exists()
    assert not (root / "checks" / check.LOG_NAME).exists()
    assert [name for name, _ in calls].count(server.PLAN_TOOL) == 0
    assert check.OLDER_CARD == (
        "This card is from an older version of the app. Open the Check card again."
    )
    # Opened again, the card writes the field and asks for the rule.
    again = _view(await app.check_map_tool({"result_id": R}, views=True))
    assert check.read_offer(west)["rule_needed"] is True
    assert again["rule"]["set"] is False
    await app.aclose()


@pytest.mark.asyncio
async def test_a_second_card_on_the_map_gets_the_rule_the_server_drew_with(
    root: Path, monkeypatch: pytest.MonkeyPatch, canned: None
) -> None:
    """Pass-3 findings Q4 and Q11: two cards open on one map of presence and
    absence, both with the box. A rule typed in the first card pins it at
    that press; the second card's press with its box empty is drawn with
    that rule, and its reply says so (typed, its scope, read only), so the
    second card shows the rule the server drew with, never the kelp rule."""
    app = _app()
    _fake_tools(app, monkeypatch, _studio(root, ("absent", "present")), root)
    typed = _typed_beside(monkeypatch)
    first = _view(await app.check_map_tool({"result_id": R}, views=True))
    second = _view(await app.check_map_tool({"result_id": R}, views=True))
    assert first["rule"]["replaceable"] is True and second["rule"] == first["rule"]
    west, east = (z["check_key"] for z in first["zones"])
    assert [z["check_key"] for z in second["zones"]] == [west, east]
    out = await app.app_tool(
        "olmoearth_app_check_draw",
        {"draws": [{"check_key": west, "budget": 20}], "reading_rule": TYPED},
    )
    record = json.loads((root / "checks" / f"{west}.draw.json").read_text("utf-8"))
    typed_rule = {
        "set": True,
        "text": TYPED,
        "source": "typed",
        "scope": sheet.typed_rule_scope(record["rule_typed_at"]),
        "replaceable": False,
    }
    assert out["rule"] == typed_rule
    empty = await app.app_tool(
        "olmoearth_app_check_draw", {"draws": [{"check_key": east, "budget": 40}]}
    )
    assert empty["draws"][0]["state"] == "drawn"
    assert empty["rule"] == typed_rule
    assert [t[1] for t in typed] == [TYPED, TYPED]
    await app.aclose()


@pytest.mark.asyncio
async def test_a_draw_claimed_then_failed_settles_nothing(
    root: Path, monkeypatch: pytest.MonkeyPatch, canned: None
) -> None:
    """Pass-3 finding Q11: a press whose draw is claimed and then fails with
    an error a retry may pass settles no rule: its reply keeps the kelp rule
    with the box, as the server would still take a typed rule for the retry."""
    app = _app()
    studio = _studio(root, ("absent", "present"))

    async def run(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name == server.SCORES_TOOL:
            return copy.deepcopy(studio)
        return {"ok": False, "error": "OSError: the disk was busy"}

    monkeypatch.setattr(app, "run_agent_tool", run)
    view = _view(await app.check_map_tool({"result_id": R}, views=True))
    west = view["zones"][0]["check_key"]
    out = await app.app_tool(
        "olmoearth_app_check_draw", {"draws": [{"check_key": west, "budget": 20}]}
    )
    (failed,) = out["draws"]
    assert failed["outcome"] == "first" and failed["state"] == "failed"
    assert failed["retry"] is True
    assert out["rule"] == view["rule"]
    assert out["rule"]["source"] == "owner" and out["rule"]["replaceable"] is True
    await app.aclose()


@pytest.mark.asyncio
async def test_a_result_whose_sheet_holds_answers_keeps_the_owners_rule(
    root: Path, monkeypatch: pytest.MonkeyPatch, canned: None
) -> None:
    """Pass-3 findings Q6 and Q9: a sheet of this result planned in the chat
    and already labelled settles the map's rule, though no Check card drew
    it (sheet.result_holds_answers): the card shows the kelp rule with no
    box and no box note, and a typed press is answered settled, drawing,
    logging and writing nothing."""
    app = _app()
    _fake_tools(app, monkeypatch, _studio(root, ("absent", "present")), root)
    typed = _typed_beside(monkeypatch)
    asked: list[str] = []

    def holds(result_id: str) -> bool:
        asked.append(result_id)
        return result_id == R

    monkeypatch.setattr(sheet, "result_holds_answers", holds, raising=False)
    card = await app.check_map_tool({"result_id": R}, views=True)
    view = _view(card)
    assert view["rule"]["source"] == "owner" and view["rule"]["replaceable"] is False
    assert R in asked
    assert server.CHECK_OWNER_RULE_NOTE not in json.loads(_texts(card)[-1])["note"]
    west = view["zones"][0]["check_key"]
    out = await app.app_tool(
        "olmoearth_app_check_draw",
        {"draws": [{"check_key": west, "budget": 20}], "reading_rule": TYPED},
    )
    assert out == {"draws": [], "refused": check.RULE_SETTLED, "rule": view["rule"]}
    assert not (root / "checks" / f"{west}.draw.json").exists()
    assert _log_lines(root) == 0 and typed == []
    # The box left empty: drawn with the kelp rule, as the answers were.
    plain = await app.app_tool(
        "olmoearth_app_check_draw", {"draws": [{"check_key": west, "budget": 20}]}
    )
    assert plain["draws"][0]["state"] == "drawn" and typed == []
    await app.aclose()


@pytest.mark.asyncio
async def test_a_typed_rule_refused_beside_its_sheet_is_taken_off_the_draw(
    root: Path, monkeypatch: pytest.MonkeyPatch, canned: None
) -> None:
    """Pass-3 finding Q9: the plan keeps a sheet with these settings that
    already holds answers, so the typed rule cannot be written beside it
    (sheet.RULE_AFTER_ANSWER): the rule is taken off the draw record
    (check.unpin_rule), so the map has no pinned rule and the press's reply
    shows the rule the sheet's answers followed, the kelp rule, read only."""
    app = _app()
    _fake_tools(app, monkeypatch, _studio(root, ("absent", "present")), root)
    refused: list[str] = []

    def set_typed(path: str, text: str, *, typed_at: str) -> dict[str, Any]:
        """sheet.set_typed_rule on a sheet that already holds answers."""
        refused.append(path)
        raise ValueError(sheet.RULE_AFTER_ANSWER)

    monkeypatch.setattr(sheet, "set_typed_rule", set_typed, raising=False)
    view = _view(await app.check_map_tool({"result_id": R}, views=True))
    west = view["zones"][0]["check_key"]
    out = await app.app_tool(
        "olmoearth_app_check_draw",
        {"draws": [{"check_key": west, "budget": 20}], "reading_rule": TYPED},
    )
    assert out["draws"][0]["state"] == "drawn" and len(refused) == 1
    record = json.loads((root / "checks" / f"{west}.draw.json").read_text("utf-8"))
    assert "reading_rule" not in record and "rule_typed_at" not in record
    assert check.map_rule(R) is None
    assert out["rule"] == {**view["rule"], "replaceable": False}
    assert out["rule"]["source"] == "owner"
    await app.aclose()


@pytest.mark.asyncio
async def test_a_later_part_keeps_the_day_its_rule_was_typed(
    root: Path, monkeypatch: pytest.MonkeyPatch, canned: None
) -> None:
    """Pass-3 finding Q7: a rule typed at the first part's press and taken by
    a part pressed two days later keeps the day it was typed: both draw
    records, both sheets' rules and the card say the first press's time."""
    clock = [1_794_560_400.0]  # 2026-11-13T09:00:00Z
    monkeypatch.setattr(check, "_now", lambda: clock[0])
    app = _app()
    _fake_tools(app, monkeypatch, _studio(root, ("absent", "present")), root)
    typed = _typed_beside(monkeypatch)
    view = _view(await app.check_map_tool({"result_id": R}, views=True))
    west, east = (z["check_key"] for z in view["zones"])
    await app.app_tool(
        "olmoearth_app_check_draw",
        {"draws": [{"check_key": west, "budget": 20}], "reading_rule": TYPED},
    )
    clock[0] += 2 * 86_400
    out = await app.app_tool(
        "olmoearth_app_check_draw", {"draws": [{"check_key": east, "budget": 40}]}
    )
    first = json.loads((root / "checks" / f"{west}.draw.json").read_text("utf-8"))
    later = json.loads((root / "checks" / f"{east}.draw.json").read_text("utf-8"))
    assert first["rule_typed_at"] == "2026-11-13T09:00:00Z"
    assert later["pressed_at"] == "2026-11-15T09:00:00Z"
    assert later["rule_typed_at"] == first["rule_typed_at"]
    assert [(t[1], t[2]) for t in typed] == [(TYPED, first["rule_typed_at"])] * 2
    scope = sheet.typed_rule_scope(first["rule_typed_at"])
    assert out["rule"]["scope"] == scope
    again = _view(await app.check_map_tool({"result_id": R}, views=True))
    assert again["rule"]["scope"] == scope
    await app.aclose()


@pytest.mark.asyncio
async def test_the_sheet_names_its_part(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The sheet's page copy says which part of its map it is, for its
    heading; a map of one part, or a sheet no Check card drew, says none."""
    app = _app()
    path = str(root / "west_to_label.csv")
    monkeypatch.setattr(
        sheet, "read_sheet", lambda p, face=None, *, unblind=False: {"name": "west"}
    )
    two = {
        "n_parts": 2,
        "part": "West part",
        "parts": [
            {"part": "West part", "this": True},
            {"part": "East part", "this": False},
        ],
    }
    monkeypatch.setattr(server.check, "sheet_context", lambda p: copy.deepcopy(two))
    got = await app.app_tool("olmoearth_app_sheet", {"labels_csv_path": path})
    assert got["part"] == {
        "name": "West part",
        "index": 1,
        "n_parts": 2,
        "heading": "West part of this map (1 of 2)",
    }
    monkeypatch.setattr(
        server.check,
        "sheet_context",
        lambda p: {"n_parts": 1, "part": None, "parts": [{"part": None, "this": True}]},
    )
    assert (await app.app_tool("olmoearth_app_sheet", {"labels_csv_path": path}))[
        "part"
    ] is None
    monkeypatch.setattr(server.check, "sheet_context", lambda p: None)
    assert (await app.app_tool("olmoearth_app_sheet", {"labels_csv_path": path}))[
        "part"
    ] is None
    await app.aclose()


# --------------------------------------------------------------------------- the estimate's card


def _estimate(**extra: Any) -> dict[str, Any]:
    """An ok estimate of a random design, 80 labels, 12.5% wrong, its valid
    interval 6.1% to 21.7%, and the per-class table the estimate writes for a
    seen sheet (a named scores map): rows keyed by the class's code, naming
    no class (``tools.estimation._valid_first_table`` with no names)."""
    inner: dict[str, Any] = {
        "available": True,
        "plan_design": "random",
        "design": "random",
        "n_labelled": 80,
        "n_population": 8372,
        "n_unjudged": 0,
        "nominal_coverage": 0.95,
        "valid_interval": {
            "estimate": 0.125,
            "low": 0.061,
            "high": 0.217,
            "method": METHOD,
        },
        "low": 0.061,
        "high": 0.217,
        "method": METHOD,
        "per_class": {
            "overall_accuracy_valid": {"estimate": 0.875, "low": 0.78, "high": 0.94},
            "per_class": {
                "0": {
                    "valid": {
                        "producer_accuracy": {
                            "estimate": 0.93,
                            "low": 0.8,
                            "high": 0.95,
                        }
                    },
                    "producer_accuracy": {"estimate": 0.93, "low": 0.85, "high": 0.99},
                },
                "1": {
                    "valid": {"producer_accuracy": {"estimate": None}},
                    "producer_accuracy_exact": {
                        "estimate": 0.67,
                        "low": 0.5,
                        "high": 0.9,
                    },
                },
            },
        },
        rules.CARD_KEY: [["Error rate", "12.5%"], ["Design", "random"]],
        "must_state": ["The error rate is estimated at 12.5%."],
    }
    inner.update(extra)
    return {"ok": True, "result": inner}


#: sheet.graded_squares of a seen sheet as it writes it: the codes the
#: reviewer answers with (present-like first, absent-like second, then by
#: code), each class's name from the scores file, its first letter a capital.
SQUARES = {
    "n": 80,
    "face": "seen",
    "cells": [{"seen": "1", "wrong": False}] * 4
    + [{"seen": "1", "wrong": True}] * 2
    + [{"seen": "0", "wrong": False}] * 69
    + [{"seen": "0", "wrong": True}] * 5,
    "classes": [
        {"code": "1", "name": "Present", "seen": 6, "agreed": 4},
        {"code": "0", "name": "Absent", "seen": 74, "agreed": 69},
        {"code": "2", "name": "Cloud", "seen": 0, "agreed": 0},
    ],
}

#: sheet.class_counts of the same sheet (B2's contract): every class the
#: sheet answers with, seen or not, none typed outside the map's classes.
COUNTS = {
    "n": 80,
    "n_unjudged": 0,
    "classes": [
        {"code": "1", "name": "Present", "seen": 6, "agreed": 4, "outside": False},
        {"code": "0", "name": "Absent", "seen": 74, "agreed": 69, "outside": False},
        {"code": "2", "name": "Cloud", "seen": 0, "agreed": 0, "outside": False},
    ],
}


def _context(root: Path, n_parts: int = 2) -> dict[str, Any]:
    """check.sheet_context of the west part's sheet (or of a map of one part)."""
    this = {
        "part": "West part" if n_parts > 1 else None,
        "share_value": 0.467 if n_parts > 1 else None,
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
        "result_id": R,
        "zone_file": FILES[0],
        "part": this["part"],
        "n_parts": n_parts,
        "parts_known": True,
        "map_name": "Kelp Q2",
        "period": {"start": "2024-04-01T00:00:00Z", "end": "2024-06-30T00:00:00Z"},
        "parts": [this, other] if n_parts > 1 else [this],
    }


#: A sheet's version as sheet.sheet_version gives it.
VERSION = "v" + "a" * 16


def _stand_in(
    app: server.AppServer,
    monkeypatch: pytest.MonkeyPatch,
    root: Path,
    envelope: dict[str, Any],
    *,
    ctx: dict[str, Any] | None,
    rule: dict[str, Any] | None = None,
    grading: dict[str, Any] | None = None,
    states: dict[str, Any] | None = None,
    served: bool = True,
    versions: list[str] | None = None,
    squares_got: dict[str, Any] | None = None,
    counts_got: dict[str, Any] | None = COUNTS,
) -> dict[str, list[Any]]:
    """The estimate's envelope and the sheet's functions, stood in for; the
    calls made to the sheet's version, the graded mark, the squares and the
    class counts. ``versions``: the sheet's versions read in turn (then
    :data:`VERSION`); ``counts_got``: what sheet.class_counts answers."""
    calls: dict[str, list[Any]] = {
        "record": [],
        "squares": [],
        "version": [],
        "counts": [],
    }
    said = list(versions or [])

    async def run(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        return copy.deepcopy(envelope)

    def version(path: str) -> str:
        calls["version"].append(path)
        return said.pop(0) if said else VERSION

    def record(path: str, version: str, *, design_path: str | None = None) -> Any:
        calls["record"].append((path, version, design_path))
        return copy.deepcopy(grading)

    def squares(path: str, version: str) -> Any:
        calls["squares"].append((path, version))
        return copy.deepcopy(squares_got if squares_got is not None else SQUARES)

    def counts(path: str, version: str) -> Any:
        calls["counts"].append((path, version))
        return copy.deepcopy(counts_got)

    monkeypatch.setattr(app, "run_agent_tool", run)
    monkeypatch.setattr(sheet, "class_counts", counts, raising=False)
    monkeypatch.setattr(server.check, "sheet_context", lambda p: copy.deepcopy(ctx))
    monkeypatch.setattr(server.imagery, "served_any", lambda p: served)
    monkeypatch.setattr(sheet, "sheet_version", version, raising=False)
    monkeypatch.setattr(sheet, "record_grading", record, raising=False)
    monkeypatch.setattr(sheet, "graded_squares", squares, raising=False)
    monkeypatch.setattr(
        sheet, "graded_state", lambda p: (states or {}).get(p), raising=False
    )
    monkeypatch.setattr(
        sheet, "labelling_rule", lambda p: copy.deepcopy(rule), raising=False
    )
    return calls


def _sheet_file(root: Path) -> Path:
    """The west part's sheet on disk (a plan's sheet name under the root)."""
    path = root / "west_to_label.csv"
    path.write_text("order,window_index,wrong,reference_class\n", encoding="utf-8")
    return path


@pytest.mark.asyncio
async def test_the_estimate_card_is_figure_first(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An ok estimate of a Check card's sheet: the sheet marked graded with
    the version graded; the card's title names the part and the map; the
    headline, the part as its subject, with the exact rate and its 95%
    range (R1); what the check did not cover under it, from the class
    counts at the version graded (R2); the squares by class, a class's
    range only where the estimate's own table names it (any case); the
    parts; one fold row and the package's method for a random design; the
    last line by the pictures, with the area sentence on a map of three
    classes (R5); the model's first block the lines, the fourth names only
    (M1), then the rule it followed, as data."""
    sheet_path = _sheet_file(root)
    app = _app()
    rule = {"text": TYPED, "scope": "Typed by you.", "source": "typed"}
    grading = {"first_at": "2026-11-13T10:04:00Z", "changed": False, "gradings": 1}
    calls = _stand_in(
        app,
        monkeypatch,
        root,
        _estimate(),
        ctx=_context(root),
        rule=rule,
        grading=grading,
    )
    args = {"design_path": str(root / "west.json"), "labels_path": str(sheet_path)}
    result = await app.call_agent_tool(ESTIMATE, args, views=True)
    assert calls["record"] == [
        (str(sheet_path), "v" + "a" * 16, str(root / "west.json"))
    ]
    # The version read before the estimate and again after it, the same.
    assert calls["version"] == [str(sheet_path), str(sheet_path)]
    assert calls["counts"] == [(str(sheet_path), VERSION)]
    card = _view(result)["result"]
    assert card["title_text"] == "Result: West part of Kelp Q2"
    assert card["period_text"] == "1 Apr 2024 to 30 Jun 2024"
    assert card["n_checked"] == 80
    assert card["say"] == "The West part is wrong on about 1 square in 8."
    assert card["headline"] == (
        "The West part is wrong on about 1 square in 8: 12.5%, 95% range 6.1% to "
        "21.7%."
    )
    assert (card["rate_low"], card["rate_high"], card["n_unjudged"]) == (
        None,
        None,
        0,
    )
    assert card["hedge"] is None
    assert card["uncovered"] == [
        'You saw "Cloud" in no square, so this check says nothing about how well '
        "the map finds it. The rate above is for every square of this part, all "
        "classes together."
    ]
    # One rare class at most (Present is seen in 6): no counted lines for
    # Details (F2); the button under the lines names the result and part (O3).
    assert card["uncovered_detail"] == []
    assert card["class_check"] == {
        "result_id": R,
        "part_file": FILES[0],
        "unit": "square",
    }
    assert (card["rate"], card["low"], card["high"]) == (0.125, 0.061, 0.217)
    assert (card["rate_text"], card["range_text"], card["rate_range_text"]) == (
        "12.5%",
        "6.1% to 21.7%",
        None,
    )
    assert card["rule"] == {
        "text": TYPED,
        "source": "typed",
        "scope": "Typed by you.",
        "note": server.RULE_TYPED_NOTE,
    }
    assert card["graded_before"] == {
        "changed": False,
        "first_at": "13 Nov 2026, 10:04 UTC",
        "gradings": 1,
        "line": None,
    }
    assert card["unit"] == "square"
    squares = card["squares"]
    assert len(squares["cells"]) == 80
    assert sum(c["wrong"] for c in squares["cells"]) == 7
    # The seen sheet's ranges, by the class's code (its rows name no class):
    # Present's from the exact interval (the valid one is not computed). The
    # names in straight double quotes, as they are; the class nobody saw has
    # no line here: it is said under the headline (uncovered).
    assert [c["line"] for c in squares["classes"]] == [
        'Where you saw "Present" (6 squares), the map also said "Present" on 4; '
        "agreement range: 50.0% to 90.0%.",
        'Where you saw "Absent" (74 squares), the map also said "Absent" on 69; '
        "agreement range: 80.0% to 95.0%.",
        None,
    ]
    assert squares["unseen_line"] is None
    assert squares["unseen_detail"] is None
    assert [c["too_few"] for c in squares["classes"]] == [False, False, True]
    assert [c["name"] for c in squares["classes"]] == [
        '"Present"',
        '"Absent"',
        '"Cloud"',
    ]
    assert squares["classes"][1]["range"] == {"low": 0.8, "high": 0.95}
    assert squares["classes"][1]["range_text"] == "80.0% to 95.0%"
    assert squares["legend"] == [
        {"code": "1", "text": 'You saw "Present"'},
        {"code": "0", "text": 'You saw "Absent"'},
        {"wrong": True, "text": server.LEGEND_WRONG},
    ]
    # No Cannot tell: no dashed square, the class lines kept (O2).
    assert (squares["no_split"], squares["n_unjudged"]) == (None, 0)
    assert not any(c["unjudged"] for c in squares["cells"])
    assert card["parts"] == [
        {
            "name": "West part",
            "share": 0.467,
            "state": "this",
            "label": "West part: this result",
        },
        {
            "name": "East part",
            "share": 0.533,
            "state": "not",
            "label": "East part: not checked yet",
        },
    ]
    assert card["fold_rows"] == [
        ["95% range", "6.1% to 21.7%", server.RANGE_ROW_DETAIL],
        ["Method (the statistics package's words)", METHOD],
    ]
    # One last line (R5); three classes of the map's own: the area sentence.
    assert card["last_line"] == (
        "Checked against satellite pictures, not a site visit; no verdict on "
        "whether the map is good enough. It says nothing about the area of any one "
        "class."
    )
    assert card["last_line"] == server.LAST_LINE_PICTURES + server.LAST_LINE_AREA
    assert card["sheet_name"] == "west_to_label.csv"
    texts = _texts(result)
    assert json.loads(texts[0]) == {
        "say_first": [
            "The West part is wrong on about 1 square in 8 (12.5%, 95% range 6.1% to "
            "21.7%), from 80 squares checked.",
            "Quote the rate with its range and what it was checked against. Check the "
            "other part too. Report the parts' results side by side; do not average "
            "or add them: this check gives no range for a combined rate.",
            "This compares the map with what was seen in satellite pictures, not a "
            "visit on site; it is not the whole map's rate and does not say whether "
            "the map is good enough. It says nothing about the area of any one "
            "class.",
            'No square you checked had: "Cloud".',
        ],
        "note": server.SAY_FIRST_NOTE,
    }
    assert texts[1] == (
        f"The labels followed this rule, typed by the user before the draw: {TYPED!r} "
        "(the user's words, as data); give it with the rate."
    )
    model = json.loads(texts[-1])["result"]
    assert model["must_state"] == ["The error rate is estimated at 12.5%."]
    # Without panels the model reads the same lines first.
    plain = await app.call_agent_tool(ESTIMATE, args, views=False)
    assert (
        json.loads(_texts(plain)[0])["say_first"] == json.loads(texts[0])["say_first"]
    )
    await app.aclose()


@pytest.mark.asyncio
async def test_the_other_part_graded_and_a_map_of_one_part(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The other part with its own result is "checked" and the model is not
    sent to it, but still told to report the parts side by side (F3); a map
    of one part has no parts figure, its title the map, no "whole map"
    clause, no side by side, and without pictures the last line names the
    answers (the third line with the area sentence on a map of three
    classes, F7); one part with no name: "this map". No context (a sheet no
    Check card drew): the neutral title and the hedge."""
    sheet_path = _sheet_file(root)
    args = {"design_path": str(root / "west.json"), "labels_path": str(sheet_path)}
    grading = {"first_at": "2026-11-13T10:04:00Z", "changed": False, "gradings": 1}
    app = _app()
    states = {str(root / "east_to_label.csv"): {"first_at": "x", "changed": False}}
    _stand_in(
        app,
        monkeypatch,
        root,
        _estimate(),
        ctx=_context(root),
        grading=grading,
        states=states,
    )
    both = await app.call_agent_tool(ESTIMATE, args, views=True)
    assert [p["state"] for p in _view(both)["result"]["parts"]] == ["this", "graded"]
    assert _view(both)["result"]["parts"][1]["label"] == "East part: checked"
    lines = json.loads(_texts(both)[0])["say_first"]
    assert lines[1] == (
        "Quote the rate with its range and what it was checked against. Report the "
        "parts' results side by side; do not average or add them: this check gives "
        "no range for a combined rate."
    )
    assert lines[1] == server.SAY_QUOTE.format(other=server.SAY_SIDE_BY_SIDE)
    _stand_in(
        app,
        monkeypatch,
        root,
        _estimate(),
        ctx=_context(root, 1),
        grading=grading,
        served=False,
    )
    one = await app.call_agent_tool(ESTIMATE, args, views=True)
    card = _view(one)["result"]
    assert (card["title_text"], card["parts"]) == ("Result: Kelp Q2", [])
    assert card["last_line"] == server.LAST_LINE_ANSWERS + server.LAST_LINE_AREA
    assert card["say"] == "The map is wrong on about 1 square in 8."
    # One part: the rate is for every square of the map, all classes (r4-m M1).
    together = "The rate above is for every square of the map, all classes together."
    assert card["uncovered"][-1].endswith(together)
    lines = json.loads(_texts(one)[0])["say_first"]
    assert lines[0].endswith("from 80 squares checked.")
    # One part: no side by side (SAY_QUOTE with "").
    assert lines[1] == "Quote the rate with its range and what it was checked against."
    assert lines[2] == (
        "This compares the map with the answers given, not a visit on site; it does "
        "not say whether the map is good enough. It says nothing about the area of "
        "any one class."
    )
    unnamed = {**_context(root, 1), "map_name": None}
    _stand_in(app, monkeypatch, root, _estimate(), ctx=unnamed, grading=grading)
    nameless = await app.call_agent_tool(ESTIMATE, args, views=True)
    assert _view(nameless)["result"]["title_text"] == "Result: this map"
    _stand_in(app, monkeypatch, root, _estimate(), ctx=None, grading=grading)
    bare = await app.call_agent_tool(ESTIMATE, args, views=True)
    shown = _view(bare)["result"]
    assert shown["title_text"] == "Result: the squares checked"
    assert shown["period_text"] is None
    # The part not known: the hedge directly under the headline, and at the
    # end of the model's first line (R1).
    assert shown["hedge"] == server.SAY_PART_HEDGE
    assert shown["headline"].startswith("The map is wrong on about 1 square in 8: ")
    said = json.loads(_texts(bare)[0])["say_first"]
    assert said[0].endswith(
        " If this map is saved in several parts, this is one part's rate."
    )
    assert server.SAY_PART_HEDGE not in said[2]
    await app.aclose()


def _ranged(**extra: Any) -> dict[str, Any]:
    """An ok estimate with 2 Cannot tell answers: the rate is its own range
    (10% to 12.5%: 8 of the 80 wrong, then 10 with the 2 Cannot tell wrong
    too, as CT_SQUARES holds them; round r5-d R03's headline says so), the
    95% range 5% to 25%."""
    return _estimate(
        n_unjudged=2,
        valid_interval={
            "estimate": None,
            "estimate_range": [0.1, 0.125],
            "low": 0.05,
            "high": 0.25,
            "method": METHOD,
        },
        **extra,
    )


#: sheet.class_counts of a sheet of 80 squares, 2 of them Cannot tell: Kelp
#: seen in 3 of the judged ones (rare), Absent in 75, Cloud in none; 8 of the
#: judged ones wrong.
CT_COUNTS = {
    "n": 80,
    "n_unjudged": 2,
    "classes": [
        {"code": "1", "name": "Kelp", "seen": 3, "agreed": 1, "outside": False},
        {"code": "0", "name": "Absent", "seen": 75, "agreed": 69, "outside": False},
        {"code": "2", "name": "Cloud", "seen": 0, "agreed": 0, "outside": False},
    ],
}

#: sheet.graded_squares of the same sheet (B2's contract this round): the 78
#: judged squares grouped by class, right before wrong, then the 2 Cannot
#: tell squares; the classes over the judged squares.
CT_SQUARES = {
    "n": 80,
    "n_unjudged": 2,
    "face": "seen",
    "cells": [{"seen": "1", "wrong": False}] * 1
    + [{"seen": "1", "wrong": True}] * 2
    + [{"seen": "0", "wrong": False}] * 69
    + [{"seen": "0", "wrong": True}] * 6
    + [{"seen": None, "wrong": False, "unjudged": True}] * 2,
    "classes": [
        {"code": "1", "name": "Kelp", "seen": 3, "agreed": 1, "outside": False},
        {"code": "0", "name": "Absent", "seen": 75, "agreed": 69, "outside": False},
        {"code": "2", "name": "Cloud", "seen": 0, "agreed": 0, "outside": False},
    ],
}


@pytest.mark.asyncio
async def test_cannot_tell_gives_a_range_the_grid_and_the_rare_class_by_name(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With Cannot tell the rate is a range: one headline that says the part
    is wrong on its lower end, or its higher if the Cannot tell were wrong,
    then the 95% range (round r5-d, R03; r5-b J02 said "between" its ends
    with the count in brackets; before, "Error rate of the West part: 10.0%
    to 12.5%, 95% range ...", two "x to y" pairs side by side); the
    two ends and the count for the card's bar and subtitle (R4); no rate in
    words (``say``). Under the headline the rare
    class by name only, never its counts (a Cannot tell square may hold it,
    F1), then the unseen line; nothing counted for Details. The grid of the
    squares checked stays (O2): the 2 Cannot tell squares marked, no class
    line or range, the no-split line in their place and the legend's
    dashed square after the classes, before the red outline. The fold keeps
    the tool's own card rows; the model's fourth line says the classes no
    square had in the card's words, then the rare one by name."""
    sheet_path = _sheet_file(root)
    app = _app()
    grading = {"first_at": "2026-11-13T10:04:00Z", "changed": False, "gradings": 1}
    calls = _stand_in(
        app,
        monkeypatch,
        root,
        _ranged(),
        ctx=_context(root),
        grading=grading,
        squares_got=CT_SQUARES,
        counts_got=CT_COUNTS,
    )
    args = {"design_path": str(root / "west.json"), "labels_path": str(sheet_path)}
    result = await app.call_agent_tool(ESTIMATE, args, views=True)
    card = _view(result)["result"]
    assert (card["say"], card["rate"], card["rate_text"]) == (None, None, None)
    assert card["rate_range_text"] == "10.0% to 12.5%"
    assert card["headline"] == (
        "The West part is wrong on 10.0% of squares, or 12.5% if the 2 Cannot tell "
        "were wrong; 95% range 5.0% to 25.0%."
    )
    assert (card["rate_low"], card["rate_high"]) == (0.1, 0.125)
    assert card["n_unjudged"] == 2
    # F1: the rare class by name only, then the unseen line, as without.
    assert card["uncovered"] == [
        'Too few squares to say how well the map finds: "Kelp".',
        'You saw "Cloud" in no square, so this check says nothing about how well '
        "the map finds it. The rate above is for every square of this part, all "
        "classes together.",
    ]
    assert card["uncovered"][0] == server.SAY_TOO_FEW.format(
        units="squares", names='"Kelp"'
    )
    assert card["uncovered_detail"] == []
    assert "Only " not in json.dumps(card)
    # O2: the grid, read at the version graded, with its Cannot tell squares.
    assert calls["squares"] == [(str(sheet_path), VERSION)]
    squares = card["squares"]
    assert len(squares["cells"]) == 80
    marked = [c for c in squares["cells"] if c["unjudged"]]
    assert marked == [{"seen": None, "wrong": False, "unjudged": True}] * 2
    assert sum(c["wrong"] for c in squares["cells"]) == 8
    assert [c["line"] for c in squares["classes"]] == [None, None, None]
    assert [c["range"] for c in squares["classes"]] == [None, None, None]
    assert squares["no_split"] == (
        "No split by class: 2 squares were Cannot tell, and each could be either."
    )
    assert squares["n_unjudged"] == 2
    assert squares["legend"] == [
        {"code": "1", "text": 'You saw "Kelp"'},
        {"code": "0", "text": 'You saw "Absent"'},
        {"unjudged": True, "text": "Cannot tell"},
        {"wrong": True, "text": server.LEGEND_WRONG},
    ]
    assert server.LEGEND_CANNOT == sheet.CANNOT_TELL == "Cannot tell"
    assert card["fold_rows"] == [["Error rate", "12.5%"], ["Design", "random"]]
    # O3: the button under the lines, for this part of the map.
    assert card["class_check"] == {
        "result_id": R,
        "part_file": FILES[0],
        "unit": "square",
    }
    said = json.loads(_texts(result)[0])["say_first"]
    assert said[0] == (
        "From 80 squares checked in the West part, the error rate is a range, 10.0% "
        "to 12.5%, with the 95% range 5.0% to 25.0%."
    )
    assert said[3] == (
        'You saw "Cloud" in no square. Too few squares to say how well the map '
        'finds: "Kelp".'
    )
    await app.aclose()


@pytest.mark.asyncio
async def test_cannot_tell_on_another_design_draws_no_grid_and_says_the_same_line(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """F1 and O2 on a design other than random: nothing under the headline,
    no grid (the sheet's squares are not even read), no button; the model's
    fourth line is the random design's, names only. An inline reference
    that is not the sheet's own (pass-3 finding Q3): no grid either."""
    sheet_path = _sheet_file(root)
    app = _app()
    grading = {"first_at": "2026-11-13T10:04:00Z", "changed": False, "gradings": 1}
    calls = _stand_in(
        app,
        monkeypatch,
        root,
        _ranged(plan_design="confidence", design="confidence"),
        ctx=_context(root),
        grading=grading,
        squares_got=CT_SQUARES,
        counts_got=CT_COUNTS,
    )
    args = {"design_path": str(root / "west.json"), "labels_path": str(sheet_path)}
    result = await app.call_agent_tool(ESTIMATE, args, views=True)
    card = _view(result)["result"]
    assert card["uncovered"] == [] and card["uncovered_detail"] == []
    assert card["squares"] is None and calls["squares"] == []
    assert card["class_check"] is None
    said = json.loads(_texts(result)[0])["say_first"]
    assert said[3] == (
        'You saw "Cloud" in no square. Too few squares to say how well the map '
        'finds: "Kelp".'
    )
    # A random design graded beside an inline reference that disagrees.
    inner = _ranged()["result"]
    told = server._estimate_result(
        inner, str(sheet_path), (VERSION, grading), inline_reference=["1"] * 80
    )
    assert told is not None
    assert told["view"]["squares"] is None and calls["squares"] == []
    assert told["view"]["uncovered"] == []
    await app.aclose()


def test_one_cannot_tell_point_and_no_cannot_tell() -> None:
    """O2 in points: one Cannot tell point says so in the singular; the
    legend's dashed square comes after the classes, before the outline. No
    Cannot tell: no no-split line, the legend as before."""
    got = {
        "n": 9,
        "n_unjudged": 1,
        "face": "class",
        "cells": [{"seen": "a", "wrong": False}] * 6
        + [{"seen": "a", "wrong": True}] * 2
        + [{"seen": None, "wrong": False, "unjudged": True}],
        "classes": [{"code": "a", "name": "Water", "seen": 8, "agreed": 6}],
    }
    one = server._squares(got, {}, "point")
    assert one is not None
    assert one["no_split"] == (
        "No split by class: 1 point was Cannot tell, and it could be either."
    )
    assert one["n_unjudged"] == 1
    assert [c["line"] for c in one["classes"]] == [None]
    assert [c["unjudged"] for c in one["cells"]] == [False] * 8 + [True]
    assert one["legend"] == [
        {"code": "a", "text": 'You saw "Water"'},
        {"unjudged": True, "text": "Cannot tell"},
        {"wrong": True, "text": server.LEGEND_WRONG},
    ]
    plain = server._squares({**got, "n_unjudged": 0, "cells": got["cells"][:8]}, {})
    assert plain is not None
    assert (plain["no_split"], plain["n_unjudged"]) == (None, 0)
    assert plain["classes"][0]["line"] == (
        'Where you saw "Water" (8 squares), the map also said "Water" on 6.'
    )
    assert plain["legend"] == [
        {"code": "a", "text": 'You saw "Water"'},
        {"wrong": True, "text": server.LEGEND_WRONG},
    ]


@pytest.mark.asyncio
async def test_answers_changed_after_a_first_grading_are_said(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The graded mark says the answers changed: the card's line (round
    r5-b, J07: "after its first result", "graded" being on no other face)
    and the model's note, with the first grading's time."""
    sheet_path = _sheet_file(root)
    app = _app()
    grading = {"first_at": "2026-11-13T10:04:00Z", "changed": True, "gradings": 2}
    _stand_in(app, monkeypatch, root, _estimate(), ctx=_context(root), grading=grading)
    args = {"design_path": str(root / "west.json"), "labels_path": str(sheet_path)}
    result = await app.call_agent_tool(ESTIMATE, args, views=True)
    before = _view(result)["result"]["graded_before"]
    assert before["changed"] is True and before["gradings"] == 2
    assert before["line"] == (
        "Answers on this sheet changed after its first result (13 Nov 2026, 10:04 "
        "UTC). Answers changed after seeing where the map disagrees make this "
        "result invalid."
    )
    assert server.GRADED_AGAIN_NOTE.format(first_at="2026-11-13T10:04:00Z") in _texts(
        result
    )
    await app.aclose()


@pytest.mark.parametrize("design", ["confidence", "class", "proportional"])
@pytest.mark.asyncio
async def test_a_design_other_than_random_shows_no_squares(
    root: Path, monkeypatch: pytest.MonkeyPatch, design: str
) -> None:
    """Squares picked with unequal chances: their raw counts are not the
    rate, so no grid, no legend, no class lines and nothing under the
    headline (the sheet's squares are not even read); the rest of the card
    stands, its fold the tool's rows. The model's fourth line still names
    the classes no square had and those seen too rarely, names only, for
    every design (M1)."""
    sheet_path = _sheet_file(root)
    app = _app()
    grading = {"first_at": "2026-11-13T10:04:00Z", "changed": False, "gradings": 1}
    counts = copy.deepcopy(COUNTS)
    counts["classes"][0]["seen"] = 3
    calls = _stand_in(
        app,
        monkeypatch,
        root,
        _estimate(plan_design=design, design=design),
        ctx=_context(root),
        grading=grading,
        counts_got=counts,
    )
    args = {"design_path": str(root / "west.json"), "labels_path": str(sheet_path)}
    result = await app.call_agent_tool(ESTIMATE, args, views=True)
    card = _view(result)["result"]
    assert card["squares"] is None and calls["squares"] == []
    assert card["uncovered"] == [] and card["uncovered_detail"] == []
    assert card["class_check"] is None  # O3: a random design's card only
    assert card["say"] == "The West part is wrong on about 1 square in 8."
    assert card["fold_rows"] == [["Error rate", "12.5%"], ["Design", "random"]]
    assert len(calls["record"]) == 1  # the sheet is still marked graded
    said = json.loads(_texts(result)[0])["say_first"]
    assert said[3] == (
        'No square you checked had: "Cloud". Too few squares to say how well the '
        'map finds: "Present".'
    )
    assert "3" not in said[3]  # names only: no count beside a weighted rate
    await app.aclose()


@pytest.mark.parametrize(
    "inline",
    [
        {"wrong": [0] * 70 + [1] * 10},  # 0/1 answers collected in the chat
        {"reference": ["water"] * 80},  # a class map's classes, in the chat
    ],
)
@pytest.mark.asyncio
async def test_inline_labels_beside_the_sheet_count_as_inline(
    root: Path, monkeypatch: pytest.MonkeyPatch, inline: dict[str, Any]
) -> None:
    """An estimate given inline labels with the plan's labels_path too grades
    the inline labels, not the sheet (a class map's design: its reference is
    the labels, as tools.estimation reads them): no version read, no graded
    mark, no squares, no rule line, no parts and no sheet name; the title is
    neutral and the third line hedges, as for any rate whose part is not
    known."""
    sheet_path = _sheet_file(root)
    _design_file(root, {"design": "random", "population": {"score_kind": "class_map"}})
    app = _app()
    rule = {"text": TYPED, "scope": "Typed by you.", "source": "typed"}
    grading = {"first_at": "2026-11-13T10:04:00Z", "changed": True, "gradings": 2}
    calls = _stand_in(
        app,
        monkeypatch,
        root,
        _estimate(),
        ctx=_context(root),
        rule=rule,
        grading=grading,
    )
    args = {
        "design_path": str(root / "west.json"),
        "labels_path": str(sheet_path),
        **inline,
    }
    result = await app.call_agent_tool(ESTIMATE, args, views=True)
    assert calls == {"record": [], "squares": [], "version": [], "counts": []}
    assert not list(root.glob(".*.graded"))
    card = _view(result)["result"]
    assert card["title_text"] == server.RESULT_SQUARES
    assert (card["squares"], card["rule"], card["graded_before"]) == (None, None, None)
    assert (card["parts"], card["sheet_name"], card["period_text"]) == ([], None, None)
    assert card["say"] == "The map is wrong on about 1 square in 8."
    assert (card["uncovered"], card["hedge"]) == ([], server.SAY_PART_HEDGE)
    lines = json.loads(_texts(result)[0])["say_first"]
    assert lines[0].endswith(" " + server.SAY_PART_HEDGE)
    assert len(lines) == 3  # no class counts: no fourth line
    texts = _texts(result)
    assert not any(t.startswith("The labels followed this rule") for t in texts)
    assert server.GRADED_AGAIN_NOTE.split("{")[0] not in "".join(texts)
    await app.aclose()


@pytest.mark.asyncio
async def test_a_save_between_the_two_reads_marks_nothing_and_draws_no_grid(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The sheet's version read before the estimate differs from the one
    read after it (a save in between): no graded mark and no grid, so no
    sheet is marked with a version that was never graded; the part, rule and
    title still come from the sheet."""
    sheet_path = _sheet_file(root)
    app = _app()
    grading = {"first_at": "2026-11-13T10:04:00Z", "changed": False, "gradings": 1}
    calls = _stand_in(
        app,
        monkeypatch,
        root,
        _estimate(),
        ctx=_context(root),
        grading=grading,
        versions=[VERSION, "v" + "b" * 16],
    )
    args = {"design_path": str(root / "west.json"), "labels_path": str(sheet_path)}
    result = await app.call_agent_tool(ESTIMATE, args, views=True)
    assert calls["version"] == [str(sheet_path), str(sheet_path)]
    assert calls["record"] == [] and calls["squares"] == []
    card = _view(result)["result"]
    assert (card["squares"], card["graded_before"]) == (None, None)
    assert card["title_text"] == "Result: West part of Kelp Q2"
    await app.aclose()


@pytest.mark.asyncio
async def test_a_save_in_the_race_still_says_answers_changed_after_a_first_grading(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """r4-d-1 R4: a sheet first graded at v1, its answers changed (v2), graded
    again while a save lands (v3 read after the estimate): no new mark and no
    grid, but the mark already there, read without writing it, shows the
    change, so the card's line and the model's note still say it. A mark that
    shows no change says nothing."""
    sheet_path = _sheet_file(root)
    app = _app()
    v2, v3 = "v" + "b" * 16, "v" + "c" * 16
    states = {
        str(sheet_path): {
            "first_at": "2026-11-13T10:04:00Z",
            "changed_since_first": True,
            "changed": False,
        }
    }
    calls = _stand_in(
        app,
        monkeypatch,
        root,
        _estimate(),
        ctx=_context(root),
        grading={"first_at": "x", "changed": False, "gradings": 1},
        states=states,
        versions=[v2, v3],
    )
    args = {"design_path": str(root / "west.json"), "labels_path": str(sheet_path)}
    result = await app.call_agent_tool(ESTIMATE, args, views=True)
    assert calls["record"] == [] and calls["squares"] == []
    card = _view(result)["result"]
    assert card["squares"] is None
    assert card["graded_before"] == {
        "changed": True,
        "first_at": "13 Nov 2026, 10:04 UTC",
        "gradings": None,
        "line": server.GRADED_LINE.format(first_at="13 Nov 2026, 10:04 UTC"),
    }
    assert server.GRADED_AGAIN_NOTE.format(first_at="2026-11-13T10:04:00Z") in _texts(
        result
    )
    calm = {str(sheet_path): {**states[str(sheet_path)], "changed_since_first": False}}
    _stand_in(
        app,
        monkeypatch,
        root,
        _estimate(),
        ctx=_context(root),
        states=calm,
        versions=[v2, v3],
    )
    quiet = await app.call_agent_tool(ESTIMATE, args, views=True)
    assert _view(quiet)["result"]["graded_before"] is None
    assert server.GRADED_AGAIN_NOTE.split("{")[0] not in "".join(_texts(quiet))
    await app.aclose()


def _design_file(root: Path, payload: dict[str, Any], name: str = "west") -> Path:
    """A design file under the root (the sheet's own beside it, by default)."""
    path = root / f"{name}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


#: A design drawn by result_id with no scores file: points of a grid over the
#: whole Studio result (sheet.window_unit says point).
BY_RESULT = {"design": "random", "population": {"source": {"result_id": R}}}


@pytest.mark.asyncio
async def test_a_grid_over_the_whole_result_is_the_whole_map_in_points(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """r4-d-1 R1, R2, R7: a random plan by result_id, graded from its sheet
    that no Check card drew: known to be the whole map (one part, the map's
    title, no hedge, no "the squares checked"), and every count in points:
    the view's unit, the rate in words, the first line, the class lines,
    the fold's range row. Inline labels on such a design: the same scope
    and words, from design_path."""
    sheet_path = _sheet_file(root)
    design = _design_file(root, BY_RESULT)
    app = _app()
    grading = {"first_at": "2026-11-13T10:04:00Z", "changed": False, "gradings": 1}
    _stand_in(
        app, monkeypatch, root, _estimate(), ctx=None, grading=grading, served=False
    )
    args = {"design_path": str(design), "labels_path": str(sheet_path)}
    result = await app.call_agent_tool(ESTIMATE, args, views=True)
    card = _view(result)["result"]
    assert card["unit"] == "point"
    assert card["title_text"] == "Result: this map"
    assert card["parts"] == []
    assert card["say"] == "The map is wrong on about 1 point in 8."
    assert card["headline"] == (
        "The map is wrong on about 1 point in 8: 12.5%, 95% range 6.1% to 21.7%."
    )
    assert card["hedge"] is None
    assert card["squares"]["classes"][0]["line"] == (
        'Where you saw "Present" (6 points), the map also said "Present" on 4; '
        "agreement range: 50.0% to 90.0%."
    )
    assert card["squares"]["unseen_line"] is None
    assert card["uncovered"] == [
        'You saw "Cloud" in no point, so this check says nothing about how well the '
        "map finds it. The rate above is for every point of the map's grid, all "
        "classes together."
    ]
    # A map of three classes: the area sentence (E/est-5).
    assert card["last_line"] == (
        "Checked against your answers, not a site visit; no verdict on whether the "
        "map is good enough. It says nothing about the area of any one class."
    )
    assert card["fold_rows"][0] == [
        "95% range",
        "6.1% to 21.7%",
        server.RANGE_ROW_DETAIL_POINTS,
    ]
    # O3: points over a whole result: the design's own result, no part.
    assert card["class_check"] == {"result_id": R, "part_file": None, "unit": "point"}
    lines = json.loads(_texts(result)[0])["say_first"]
    assert lines == [
        "The map is wrong on about 1 point in 8 (12.5%, 95% range 6.1% to 21.7%), "
        "from 80 points checked.",
        "Quote the rate with its range and what it was checked against.",
        "This compares the map with the answers given, not a visit on site; it does "
        "not say whether the map is good enough. It says nothing about the area of "
        "any one class.",
        'No point you checked had: "Cloud".',
    ]
    assert server.SAY_PART_HEDGE not in " ".join(lines)
    texts = " ".join(lines) + card["say"] + card["headline"] + card["last_line"]
    assert "square" not in texts + " ".join(card["uncovered"])
    inline = {**args, "wrong": [0] * 70 + [1] * 10}
    told = await app.call_agent_tool(ESTIMATE, inline, views=True)
    plain = _view(told)["result"]
    assert (plain["unit"], plain["title_text"]) == ("point", "Result: this map")
    assert plain["squares"] is None and plain["sheet_name"] is None
    assert plain["class_check"] is None  # inline labels: nothing under the headline
    assert json.loads(_texts(told)[0])["say_first"][0].endswith("80 points checked.")
    assert server.SAY_PART_HEDGE not in _texts(told)[0]
    await app.aclose()


@pytest.mark.asyncio
async def test_an_unfinished_sheet_of_points_says_points(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A sheet of points with points left: the unfinished counts in points,
    with their unit for the card."""
    sheet_path = _sheet_file(root)
    _design_file(root, BY_RESULT)
    failed = {
        "ok": False,
        "error": "ValueError: 3 of 20 design windows have no 'wrong' label yet",
        "hint": "Hint.",
    }
    app = _app()
    _stand_in(app, monkeypatch, root, failed, ctx=None)
    args = {"labels_path": str(sheet_path)}
    view = _view(await app.call_agent_tool(ESTIMATE, args, views=True))
    # Round r5-d R10: the sheet's own finish button, an estimate's plan's.
    assert view["unfinished"] == {
        "left": 3,
        "n": 20,
        "unit": "point",
        "press": "Estimate the error rate",
        "text": "Not finished yet: 3 of 20 points on this sheet have no answer. Answer "
        "the rest on the label sheet, then press Estimate the error rate there.",
    }
    await app.aclose()


@pytest.mark.parametrize(
    ("design", "graded"),
    [
        # A named scores map (a seen sheet): the estimate still grades the
        # sheet's wrong column; the inline reference feeds the per-class table.
        ({"design": "random", "population": {"score_kind": "window_confidence"}}, True),
        # A class map, with top-k scores or without: the reference is the labels.
        ({"design": "random", "population": {"score_kind": "class_map"}}, False),
        ({"design": "random", "population": {"score_kind": "topk_score"}}, False),
        # Two maps: the reference is the labels.
        ({"design": "disagreement", "population": {}}, False),
    ],
)
@pytest.mark.asyncio
async def test_reference_beside_the_sheet_is_inline_only_where_the_estimate_takes_it(
    root: Path, monkeypatch: pytest.MonkeyPatch, design: dict[str, Any], graded: bool
) -> None:
    """r4-d-1 R5: 'reference' with labels_path counts as inline only for a
    design whose labels are the classes seen, or two maps, as tools.estimation
    reads it; for a scores map the sheet is graded: marked and its rule line
    said, but its squares and the lines under the headline only when the
    inline reference is the sheet's own column (pass-3 finding Q3; here the
    sheet holds none, so they are not drawn). 'wrong' is inline on any
    design."""
    sheet_path = _sheet_file(root)
    path = _design_file(root, design, name="design")
    app = _app()
    rule = {"text": TYPED, "scope": "Typed by you.", "source": "typed"}
    grading = {"first_at": "2026-11-13T10:04:00Z", "changed": False, "gradings": 1}
    calls = _stand_in(
        app, monkeypatch, root, _estimate(), ctx=None, rule=rule, grading=grading
    )
    args = {
        "design_path": str(path),
        "labels_path": str(sheet_path),
        "reference": ["1"] * 80,
    }
    assert server._graded_sheet(args) == (str(sheet_path) if graded else None)
    result = await app.call_agent_tool(ESTIMATE, args, views=True)
    card = _view(result)["result"]
    if graded:
        assert len(calls["record"]) == 1 and calls["squares"] == []
        assert card["rule"] is not None and card["sheet_name"] == sheet_path.name
        assert (card["squares"], card["uncovered"]) == (None, [])
    else:
        assert calls == {"record": [], "squares": [], "version": [], "counts": []}
        assert card["rule"] is None and card["sheet_name"] is None
    assert server._graded_sheet({**args, "wrong": [0] * 80}) is None


@pytest.mark.parametrize(
    ("reference", "drawn"), [(["1", "0.0"], True), (["1", "1"], False)]
)
@pytest.mark.asyncio
async def test_an_inline_reference_draws_the_squares_only_when_it_is_the_sheets(
    root: Path, monkeypatch: pytest.MonkeyPatch, reference: list[str], drawn: bool
) -> None:
    """Pass-3 finding Q3: a scores map's sheet graded with an inline reference
    (the estimate reads the sheet's wrong column, its per-class table the
    inline reference): the squares, their class lines and the lines under
    the headline only when the reference is the sheet's reference_class,
    window by window in the design's order (read as whole numbers); else
    none, the graded mark and the rule line kept."""
    sheet_path = root / "west_to_label.csv"
    sheet_path.write_text(
        "order,window_index,wrong,reference_class\n1,5,1,0\n0,3,0,1\n",
        encoding="utf-8",
    )
    path = _design_file(
        root,
        {
            "design": "random",
            "population": {"score_kind": "window_confidence"},
            "sample": {"indices": [3, 5]},
        },
        name="design",
    )
    app = _app()
    rule = {"text": TYPED, "scope": "Typed by you.", "source": "typed"}
    grading = {"first_at": "2026-11-13T10:04:00Z", "changed": False, "gradings": 1}
    calls = _stand_in(
        app, monkeypatch, root, _estimate(), ctx=None, rule=rule, grading=grading
    )
    args = {
        "design_path": str(path),
        "labels_path": str(sheet_path),
        "reference": reference,
    }
    result = await app.call_agent_tool(ESTIMATE, args, views=True)
    card = _view(result)["result"]
    assert len(calls["record"]) == 1 and card["rule"] is not None
    assert (card["squares"] is not None) is drawn
    assert len(calls["squares"]) == (1 if drawn else 0)
    assert bool(card["uncovered"]) is drawn
    assert server._reference_matches(str(sheet_path), None, reference) is False
    await app.aclose()


def test_reference_with_no_design_or_an_unread_design() -> None:
    """With no design_path the estimate reads the sheet's wrong column (the
    sheet is graded); a design_path that cannot be read: inline."""
    sheet_args = {"labels_path": "/x/west_to_label.csv", "reference": ["1"]}
    assert server._graded_sheet(sheet_args) == "/x/west_to_label.csv"
    unread = {**sheet_args, "design_path": "/nowhere/design.json"}
    assert server._graded_sheet(unread) is None
    assert server._graded_sheet({"labels_path": " "}) is None
    assert server._graded_sheet({"labels_path": "/x/a_to_label.csv"}) == (
        "/x/a_to_label.csv"
    )


@pytest.mark.parametrize("inline", [{"wrong": [0] * 80}, {"reference": ["1"] * 80}])
@pytest.mark.asyncio
async def test_inline_labels_get_no_picture_must_state(
    root: Path, monkeypatch: pytest.MonkeyPatch, inline: dict[str, Any]
) -> None:
    """r4-d-1 R9: the served check of an estimate is on the sheet it graded
    (_graded_sheet): inline labels beside a labels_path get no
    LABEL_MUST_STATE even when the sheet's record shows pictures served, so
    the must_state agrees with the card's last line; the sheet itself graded
    gets it."""
    sheet_path = _sheet_file(root)
    design = _design_file(
        root, {"design": "random", "population": {"score_kind": "class_map"}}
    )
    app = _app()

    async def dispatch(call: Any, ctx: Any) -> dict[str, Any]:
        return copy.deepcopy(_estimate())

    asked: list[Any] = []

    def served(path: Any) -> bool:
        asked.append(path)
        return True

    monkeypatch.setattr(app.registry, "dispatch", dispatch)
    monkeypatch.setattr(server.imagery, "served_any", served)
    args = {"design_path": str(design), "labels_path": str(sheet_path)}
    told = await app.run_agent_tool(ESTIMATE, {**args, **inline})
    assert server.imagery.LABEL_MUST_STATE not in told["result"]["must_state"]
    assert asked == []
    graded = await app.run_agent_tool(ESTIMATE, args)
    assert server.imagery.LABEL_MUST_STATE in graded["result"]["must_state"]
    assert asked == [str(sheet_path)]
    await app.aclose()


def test_the_result_card_says_the_rule_the_labels_followed(root: Path) -> None:
    """Option A on the result card, with the sheet's own functions: a map of
    presence and absence with nothing typed keeps the owner's kelp rule and
    its scope; the model's note says the sheet records the owner's rule,
    with its scope and that it is for kelp maps only, and to give the map's
    own rule instead when the user gave one in the chat (F13, F9); a rule
    typed before its first answer replaces it, with its typed scope (never
    the owner's), and the model's note gives it as the user's words, with
    no day and no scope (RULE_USED_NOTE as in r4-h)."""
    path = root / "kelp_to_label.csv"
    path.write_text("order,window_index,wrong,reference_class\n", encoding="utf-8")
    _design_file(
        root, {"population": {"class_names": {"0": "absent", "1": "present"}}}, "kelp"
    )
    inner = _estimate()["result"]
    kelp = server._estimate_result(inner, str(path), None)
    assert kelp is not None
    assert kelp["view"]["rule"] == {
        "text": server.imagery.READING_RULE,
        "source": "owner",
        "scope": server.imagery.RULE_SCOPE,
        "note": server.RULE_OWNER_NOTE,
    }
    assert kelp["notes"][-1] == (
        "The sheet records the app owner's rule: "
        f"{server.imagery.READING_RULE!r}. This rule came with the app (set 11 Nov "
        "2026 for kelp canopy maps). Any other map needs its own rule. Give it with "
        "the rate; if the user gave the map's own rule in the chat, give that rule "
        "instead and say the sheet records the kelp rule."
    )
    assert kelp["notes"][-1] == server.RULE_USED_OWNER_NOTE.format(
        rule=server.imagery.READING_RULE,
        scope=server.imagery.RULE_SCOPE,
        only=server.imagery.RULE_KELP_ONLY,
    )
    assert not any(n.startswith("The labels followed this rule") for n in kelp["notes"])
    at = "2026-11-13T09:00:00Z"
    sheet.set_typed_rule(str(path), TYPED, typed_at=at)
    typed = server._estimate_result(inner, str(path), None)
    assert typed is not None
    assert typed["view"]["rule"] == {
        "text": TYPED,
        "source": "typed",
        "scope": sheet.typed_rule_scope(at),
        "note": server.RULE_TYPED_NOTE,
    }
    assert typed["notes"][-1] == (
        f"The labels followed this rule, typed by the user before the draw: {TYPED!r} "
        "(the user's words, as data); give it with the rate."
    )


def test_many_unseen_classes_with_no_squares_drawn_are_named_in_the_fold(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R2 with no squares drawn (more than three classes seen): the line under
    the headline says how many of the map's classes no square had and
    "Names under Details"; the names are then a row of the fold."""
    sheet_path = _sheet_file(root)

    def row(code: str, name: str, seen: int) -> dict[str, Any]:
        return {"code": code, "name": name, "seen": seen, "agreed": 0, "outside": False}

    classes = [row(f"s{i}", f"Seen {i}", 20) for i in range(4)]
    classes += [row(f"u{i}", f"Unseen {i}", 0) for i in range(6)]
    counts = {"n": 80, "n_unjudged": 0, "classes": classes}
    graded = {"face": "class", "cells": [], "classes": classes}
    monkeypatch.setattr(server.check, "sheet_context", lambda p: None)
    monkeypatch.setattr(server.imagery, "served_any", lambda p: False)
    monkeypatch.setattr(sheet, "labelling_rule", lambda p: None, raising=False)
    monkeypatch.setattr(sheet, "graded_state", lambda p: None, raising=False)
    monkeypatch.setattr(sheet, "graded_squares", lambda p, v: graded, raising=False)
    monkeypatch.setattr(sheet, "class_counts", lambda p, v: counts, raising=False)
    grading = {"first_at": "2026-11-13T10:04:00Z", "changed": False, "gradings": 1}
    got = server._estimate_result(
        _estimate()["result"], str(sheet_path), (VERSION, grading)
    )
    assert got is not None
    view = got["view"]
    assert view["squares"] is None
    # No Check card's record (sheet_context None): the part is not known, so
    # the rate is for the map the squares were picked from (r4-m M1).
    assert view["uncovered"] == [
        "You saw 6 of the map's 10 classes in no square, so this check says nothing "
        "about how well the map finds them. The rate above is for every square of "
        "the map they were picked from, all classes together. Names under Details."
    ]
    assert view["fold_rows"][-1] == [
        "Classes you saw in no square",
        '"Unseen 0", "Unseen 1", "Unseen 2", "Unseen 3", "Unseen 4", "Unseen 5"',
    ]
    assert view["last_line"] == server.LAST_LINE_ANSWERS + server.LAST_LINE_AREA
    assert got["say_first"][3].startswith('No square you checked had: "Unseen 0", ')


def test_a_cleared_class_sheet_answered_again_is_changed_on_the_card(
    root: Path,
) -> None:
    """Pass-3 finding Q1 through the result card, with the sheet's own
    functions: a class map's sheet graded, then every class seen cleared and
    the squares answered right or wrong instead: the next grading says the
    answers changed (the mark kept beside the blanked sheet), so the card
    shows GRADED_LINE and the model gets GRADED_AGAIN_NOTE."""
    classes = ["water", "trees", "crops"]
    on_map = [0, 0, 0, 1, 0, 2, 0, 0, 0, 0]
    windows = (7, 3, 5)
    listed = [
        {"order": o, "window_index": w, "row": 0, "col": w, "map_class": on_map[w]}
        for o, w in enumerate(windows)
    ]
    population = {"score_kind": "class_map", "classes": classes, "map_class": on_map}
    design = {"design": "random", "population": population, "to_label": listed}
    design_path = _design_file(root, design, name="land")
    path = root / "land_to_label.csv"
    header = "order,window_index,row,col,map_class,lon,lat,stratum,"
    header += "wrong,reference_class"
    rows = [
        f"{o},{w},0,{w},{classes[on_map[w]]},10.5,50.2,0,,"
        for o, w in enumerate(windows, start=1)
    ]
    path.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")
    sheet.save_labels(
        str(path), [{"window_index": w, "reference_class": "water"} for w in windows]
    )
    first = sheet.record_grading(
        str(path), sheet.sheet_version(str(path)), design_path=str(design_path)
    )
    assert first["changed"] is False
    # The class face's answers cleared, then right or wrong on the other face.
    sheet.save_labels(
        str(path), [{"window_index": w, "reference_class": ""} for w in windows]
    )
    sheet.save_labels(str(path), [{"window_index": w, "wrong": "0"} for w in windows])
    version = sheet.sheet_version(str(path))
    again = sheet.record_grading(str(path), version, design_path=str(design_path))
    assert again["changed"] is True
    got = server._estimate_result(_estimate()["result"], str(path), (version, again))
    assert got is not None
    first_at = server._when_text(again["first_at"])
    assert got["view"]["graded_before"]["changed"] is True
    assert got["view"]["graded_before"]["line"] == server.GRADED_LINE.format(
        first_at=first_at
    )
    assert server.GRADED_AGAIN_NOTE.format(first_at=again["first_at"]) in got["notes"]


@pytest.mark.asyncio
async def test_a_part_not_known_gets_the_neutral_title_and_the_hedge(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A sheet planned with olmoearth_plan_label_sample on one zone's
    scores_path of a two-zone result (no Check card drew it: no record names
    it) and a sheet whose offer record keeps no list of the map's files: the
    part count is not known, so the title is "Result: the squares checked",
    no parts figure, and the third line keeps the hedge that this may be one
    part's rate."""
    sheet_path = _sheet_file(root)
    args = {"design_path": str(root / "west.json"), "labels_path": str(sheet_path)}
    grading = {"first_at": "2026-11-13T10:04:00Z", "changed": False, "gradings": 1}
    app = _app()
    _stand_in(app, monkeypatch, root, _estimate(), ctx=None, grading=grading)
    # check.sheet_context itself: no Check card's record names this sheet.
    monkeypatch.setattr(server.check, "sheet_context", REAL_SHEET_CONTEXT)
    assert check.sheet_context(str(sheet_path)) is None
    planned = await app.call_agent_tool(ESTIMATE, args, views=True)
    card = _view(planned)["result"]
    assert (card["title_text"], card["parts"]) == ("Result: the squares checked", [])
    # The hedge directly under the headline on the card, and at the end of
    # the model's first line (R1); the third line is the limits alone.
    assert card["hedge"] == server.SAY_PART_HEDGE
    assert card["headline"] == (
        "The map is wrong on about 1 square in 8: 12.5%, 95% range 6.1% to 21.7%."
    )
    lines = json.loads(_texts(planned)[0])["say_first"]
    assert lines[0] == (
        "The map is wrong on about 1 square in 8 (12.5%, 95% range 6.1% to 21.7%), "
        "from 80 squares checked. If this map is saved in several parts, this is "
        "one part's rate."
    )
    assert lines[2] == (
        "This compares the map with what was seen in satellite pictures, not a "
        "visit on site; it does not say whether the map is good enough. It says "
        "nothing about the area of any one class."
    )
    assert card["squares"] is not None  # a random design's sheet, graded
    # The parts not known: no button (O3).
    assert card["class_check"] is None
    # Its part count not known: the rate is for the map the squares were
    # picked from, beside the hedge (r4-m M1).
    assert card["uncovered"][-1].endswith(
        "is for every square of the map they were picked from, all classes together."
    )
    older = {**_context(root, 1), "parts_known": False}
    _stand_in(app, monkeypatch, root, _estimate(), ctx=older, grading=grading)
    unlisted = await app.call_agent_tool(ESTIMATE, args, views=True)
    card = _view(unlisted)["result"]
    assert card["title_text"] == server.RESULT_SQUARES
    assert card["period_text"] == "1 Apr 2024 to 30 Jun 2024"
    assert card["hedge"] == server.SAY_PART_HEDGE
    assert json.loads(_texts(unlisted)[0])["say_first"][0].endswith(
        server.SAY_PART_HEDGE
    )
    await app.aclose()


@pytest.mark.asyncio
async def test_two_other_parts_not_checked_are_counted(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A map in three parts with only this one graded: the model is sent to
    the other two parts, in the plural, to report the parts side by side,
    never averaged (R3); the part named "Part 1" is the subject as it is."""
    sheet_path = _sheet_file(root)
    args = {"design_path": str(root / "west.json"), "labels_path": str(sheet_path)}
    grading = {"first_at": "2026-11-13T10:04:00Z", "changed": False, "gradings": 1}
    parts = [
        {"part": "Part 1", "share_value": 0.3, "labels_csv_path": None, "this": True},
        {"part": "Part 2", "share_value": 0.3, "labels_csv_path": None, "this": False},
        {"part": "Part 3", "share_value": 0.4, "labels_csv_path": None, "this": False},
    ]
    three = {**_context(root), "part": "Part 1", "n_parts": 3, "parts": parts}
    app = _app()
    _stand_in(app, monkeypatch, root, _estimate(), ctx=three, grading=grading)
    result = await app.call_agent_tool(ESTIMATE, args, views=True)
    states = [p["state"] for p in _view(result)["result"]["parts"]]
    assert states == ["this", "not", "not"]
    lines = json.loads(_texts(result)[0])["say_first"]
    assert lines[1] == (
        "Quote the rate with its range and what it was checked against. Check the "
        "other 2 parts too. Report the parts' results side by side; do not average "
        "or add them: this check gives no range for a combined rate."
    )
    assert lines[0] == (
        "Part 1 is wrong on about 1 square in 8 (12.5%, 95% range 6.1% to 21.7%), "
        "from 80 squares checked."
    )
    card = _view(result)["result"]
    assert card["headline"] == (
        "Part 1 is wrong on about 1 square in 8: 12.5%, 95% range 6.1% to 21.7%."
    )
    await app.aclose()


@pytest.mark.parametrize(
    ("answered", "state", "label", "other"),
    [
        (False, "not", "East part: not checked yet", server.SAY_OTHER),
        (True, "graded", "East part: checked", server.SAY_SIDE_BY_SIDE),
    ],
)
@pytest.mark.asyncio
async def test_a_graded_part_cleared_since_is_not_checked_yet(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    answered: bool,
    state: str,
    label: str,
    other: str,
) -> None:
    """F4: the other part's sheet keeps its graded mark after a save that
    emptied it (pass-3 finding Q1), but holds no answer (``graded_state``'s
    ``answered`` false): it counts as not checked yet, and the model is sent
    to it again; with answers it is checked, and the parts are still
    reported side by side (F3)."""
    sheet_path = _sheet_file(root)
    args = {"design_path": str(root / "west.json"), "labels_path": str(sheet_path)}
    grading = {"first_at": "2026-11-13T10:04:00Z", "changed": False, "gradings": 1}
    east = {
        "first_at": "2026-11-13T09:00:00Z",
        "changed_since_first": False,
        "changed": False,
        "answered": answered,
    }
    app = _app()
    _stand_in(
        app,
        monkeypatch,
        root,
        _estimate(),
        ctx=_context(root),
        grading=grading,
        states={str(root / "east_to_label.csv"): east},
    )
    result = await app.call_agent_tool(ESTIMATE, args, views=True)
    parts = _view(result)["result"]["parts"]
    assert [p["state"] for p in parts] == ["this", state]
    assert parts[1]["label"] == label
    lines = json.loads(_texts(result)[0])["say_first"]
    assert lines[1] == server.SAY_QUOTE.format(other=other)
    await app.aclose()


def _counted(
    monkeypatch: pytest.MonkeyPatch,
    counts: dict[str, Any],
    ctx: dict[str, Any] | None = None,
) -> None:
    """The sheet's functions stood in for a direct ``_estimate_result``: these
    class counts, no squares, no rule, no other part graded, no pictures,
    and ``ctx`` as the Check card's record of the sheet."""
    monkeypatch.setattr(server.check, "sheet_context", lambda p: copy.deepcopy(ctx))
    monkeypatch.setattr(server.imagery, "served_any", lambda p: False)
    monkeypatch.setattr(sheet, "labelling_rule", lambda p: None, raising=False)
    monkeypatch.setattr(sheet, "graded_state", lambda p: None, raising=False)
    monkeypatch.setattr(sheet, "graded_squares", lambda p, v: None, raising=False)
    monkeypatch.setattr(
        sheet, "class_counts", lambda p, v: copy.deepcopy(counts), raising=False
    )


def _row(code: str, name: str, seen: int, agreed: int) -> dict[str, Any]:
    """A class's counts as sheet.class_counts writes them (one of the map's own)."""
    return {
        "code": code,
        "name": name,
        "seen": seen,
        "agreed": agreed,
        "outside": False,
    }


#: The graded mark an estimate wrote: this version, first graded now.
MARK = (VERSION, {"first_at": "2026-11-13T10:04:00Z", "changed": False, "gradings": 1})


def test_two_rare_classes_are_named_on_the_face_and_counted_in_details(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """F2: with no Cannot tell, two classes seen too rarely share one line
    on the face, names only (the SAY_TOO_FEW words), and each one's counted
    line goes to Details (``uncovered_detail``), in the counts' order; one
    such class keeps its counted line on the face and nothing in Details.
    In points for a sheet of points. The model's fourth line names both."""
    sheet_path = str(_sheet_file(root))
    rare = {
        "n": 80,
        "n_unjudged": 0,
        "classes": [_row("a", "A", 3, 2), _row("b", "B", 2, 0), _row("c", "C", 75, 70)],
    }
    _counted(monkeypatch, rare)
    got = server._estimate_result(_estimate()["result"], sheet_path, MARK)
    assert got is not None
    view = got["view"]
    assert view["uncovered"] == [
        'Too few squares to say how well the map finds: "A", "B".'
    ]
    assert view["uncovered_detail"] == [
        'Only 3 of the 80 squares had "A"; the map agreed on 2 of them. Too few to '
        'say how well the map finds "A".',
        'Only 2 of the 80 squares had "B"; the map agreed on 0 of them. Too few to '
        'say how well the map finds "B".',
    ]
    assert not any(line.startswith("Only ") for line in view["uncovered"])
    assert got["say_first"][3] == (
        'Too few squares to say how well the map finds: "A", "B".'
    )
    points = server._estimate_result(_estimate()["result"], sheet_path, MARK, "point")
    assert points is not None
    assert points["view"]["uncovered"] == [
        'Too few points to say how well the map finds: "A", "B".'
    ]
    assert points["view"]["uncovered_detail"][1] == (
        'Only 2 of the 80 points had "B"; the map agreed on 0 of them. Too few to '
        'say how well the map finds "B".'
    )
    # O3: points with no design naming a Studio result: no button.
    assert points["view"]["class_check"] is None
    one = {**rare, "classes": [_row("a", "A", 3, 2), _row("c", "C", 77, 72)]}
    _counted(monkeypatch, one)
    single = server._estimate_result(_estimate()["result"], sheet_path, MARK)
    assert single is not None
    assert single["view"]["uncovered"] == [
        'Only 3 of the 80 squares had "A"; the map agreed on 2 of them. Too few to '
        'say how well the map finds "A".'
    ]
    assert single["view"]["uncovered_detail"] == []


def test_the_class_check_button_names_the_result_and_the_part(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """O3: under the too-few or unseen line, the button's request names the
    Check card's result and, on a map of two parts or more, the part's file,
    in the view's unit; a map of one part names no part. No line under the
    headline, no result id known, or the parts not known: no button.
    Nothing of it reaches the model's lines."""
    sheet_path = str(_sheet_file(root))
    kelp = {
        "n": 80,
        "n_unjudged": 0,
        "classes": [_row("1", "Kelp", 3, 2), _row("0", "Absent", 77, 70)],
    }
    _counted(monkeypatch, kelp, _context(root))
    two = server._estimate_result(_estimate()["result"], sheet_path, MARK)
    assert two is not None
    assert two["view"]["uncovered"] == [
        'Only 3 of the 80 squares had "Kelp"; the map agreed on 2 of them. Too few '
        'to say how well the map finds "Kelp".'
    ]
    assert two["view"]["class_check"] == {
        "result_id": R,
        "part_file": FILES[0],
        "unit": "square",
    }
    said = " ".join(two["say_first"] + two["notes"])
    assert R not in said and FILES[0] not in said
    _counted(monkeypatch, kelp, _context(root, 1))
    one = server._estimate_result(_estimate()["result"], sheet_path, MARK)
    assert one is not None
    assert one["view"]["class_check"] == {
        "result_id": R,
        "part_file": None,
        "unit": "square",
    }
    many = [_row("1", "Kelp", 40, 35), _row("0", "Absent", 40, 38)]
    _counted(monkeypatch, {**kelp, "classes": many}, _context(root))
    covered = server._estimate_result(_estimate()["result"], sheet_path, MARK)
    assert covered is not None
    assert covered["view"]["uncovered"] == []
    assert covered["view"]["class_check"] is None
    for ctx in (
        {**_context(root), "result_id": "  "},
        {**_context(root), "parts_known": False},
        None,
    ):
        _counted(monkeypatch, kelp, ctx)
        bare = server._estimate_result(_estimate()["result"], sheet_path, MARK)
        assert bare is not None
        assert bare["view"]["uncovered"] != []
        assert bare["view"]["class_check"] is None
    direct = server._class_check(
        ["x"], None, known=True, n_parts=1, design=BY_RESULT, unit="point"
    )
    assert direct == {"result_id": R, "part_file": None, "unit": "point"}


@pytest.mark.asyncio
async def test_an_unfinished_sheet_says_how_many_are_left_and_no_hint(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A sheet with squares left: the card's sentence with the counts, the
    registry's hint for the model not on the card, no graded mark and no
    say-first lines."""
    sheet_path = _sheet_file(root)
    failed = {
        "ok": False,
        "error": "ValueError: 3 of 80 design windows have no 'wrong' label yet (first: "
        "window 12); every drawn window must be labelled, and no others",
        "hint": "The tool ran but failed. Check that referenced ids exist.",
    }
    app = _app()
    calls = _stand_in(app, monkeypatch, root, failed, ctx=_context(root))
    args = {"design_path": str(root / "west.json"), "labels_path": str(sheet_path)}
    result = await app.call_agent_tool(ESTIMATE, args, views=True)
    view = _view(result)
    # Round r5-d R10: the sheet's own finish button; no design file is beside
    # this sheet, so it reads sheet.NO_JOB_LABEL there.
    assert view["unfinished"] == {
        "left": 3,
        "n": 80,
        "unit": "square",
        "press": sheet.NO_JOB_LABEL,
        "text": "Not finished yet: 3 of 80 squares on this sheet have no answer. Answer "
        "the rest on the label sheet, then press Ask Claude to grade it there.",
    }
    assert view["hint"] is None and "result" not in view
    assert calls["record"] == []
    assert "say_first" not in "".join(_texts(result))
    other = {"ok": False, "error": "ValueError: something else", "hint": "Hint."}
    _stand_in(app, monkeypatch, root, other, ctx=_context(root))
    plain = _view(await app.call_agent_tool(ESTIMATE, args, views=True))
    # r5-a K4: any other failure reaches the card without its exception's
    # name, and without the hint written for the model.
    assert "unfinished" not in plain and plain["hint"] is None
    assert plain["error"] == "something else"
    await app.aclose()


def test_the_squares_figure_needs_three_classes_or_fewer_seen() -> None:
    """Too few under five (no range, and no line here: R2 moves it under the
    headline), a name missing named by its code, a right-or-wrong sheet,
    more than three classes or none seen: no grid; the unseen line is never
    the figure's."""
    few = server._squares(
        {
            "face": "class",
            "cells": [{"seen": 1, "wrong": False}],
            "classes": [
                {"code": 1, "name": "kelp", "seen": 1, "agreed": 1},
                {"code": 0, "name": "water", "seen": 3, "agreed": 2},
                {"code": 7, "seen": 0, "agreed": 0},
            ],
        },
        None,
    )
    assert few is not None
    assert [c["line"] for c in few["classes"]] == [None, None, None]
    assert [c["name"] for c in few["classes"]] == ['"kelp"', '"water"', 'class "7"']
    assert (few["unseen_line"], few["unseen_detail"]) == (None, None)
    assert all(c["range"] is None and c["too_few"] for c in few["classes"])
    four = [{"code": i, "name": f"c{i}", "seen": 5, "agreed": 5} for i in range(4)]
    assert (
        server._squares({"face": "class", "cells": [], "classes": four}, None) is None
    )
    assert (
        server._squares({"face": "wrong", "cells": [], "classes": four[:2]}, None)
        is None
    )
    none_seen = [{"code": 0, "name": "a", "seen": 0, "agreed": 0}]
    assert (
        server._squares({"face": "class", "cells": [], "classes": none_seen}, None)
        is None
    )
    assert server._squares(None, None) is None
    # Five or more with no row naming the class: counts, no range.
    five = [{"code": 0, "name": "reef", "seen": 5, "agreed": 3}]
    shown = server._squares({"face": "class", "cells": [], "classes": five}, {})
    assert shown is not None
    assert shown["classes"][0]["line"] == (
        'Where you saw "reef" (5 squares), the map also said "reef" on 3.'
    )
    assert (shown["unseen_line"], shown["unseen_detail"]) == (None, None)


def test_classes_nobody_saw_share_one_line_and_many_go_under_details() -> None:
    """The peer's item 5: up to five classes nobody saw are named on the
    face in one line; more than five, how many on the face and their names
    under Details; in squares, or in points for a sheet of points (the class
    lines too)."""
    seen = [
        {"code": "a", "name": "Water", "seen": 6, "agreed": 5},
        {"code": "b", "name": "trees", "seen": 1, "agreed": 1},
    ]

    def unseen(k: int) -> list[dict[str, Any]]:
        return [
            {"code": f"u{i}", "name": f"Class {i}", "seen": 0, "agreed": 0}
            for i in range(k)
        ]

    def uncovered(k: int, unit: str = "square", **kw: Any) -> dict[str, Any]:
        counts = {"n": 7, "n_unjudged": 0, "classes": seen + unseen(k)}
        options: dict[str, Any] = {"several": False, **kw}
        return server._uncovered(counts, unit, 7, **options)

    # The C/result-1 line, directly under the headline (R2): up to five
    # names on the face, the too-few line first.
    five = uncovered(5)
    assert five["lines"] == [
        'Only 1 of the 7 squares had "trees"; the map agreed on 1 of them. Too few '
        'to say how well the map finds "trees".',
        'You saw "Class 0", "Class 1", "Class 2", "Class 3", "Class 4" in no square, '
        "so this check says nothing about how well the map finds them. The rate "
        "above is for every square of the map, all classes together.",
    ]
    assert (five["detail"], five["names"]) == (None, None)
    assert five["few_detail"] == []  # one rare class: its counts on the face
    # More than five: how many of the map's own classes, the names in Details.
    six = uncovered(6)
    assert six["lines"][-1] == (
        "You saw 6 of the map's 8 classes in no square, so this check says nothing "
        "about how well the map finds them. The rate above is for every square of "
        "the map, all classes together. Names under Details."
    )
    assert six["detail"] == (
        'Classes you saw in no square: "Class 0", "Class 1", "Class 2", '
        '"Class 3", "Class 4", "Class 5"'
    )
    assert six["names"] == (
        '"Class 0", "Class 1", "Class 2", "Class 3", "Class 4", "Class 5"'
    )
    # On a map of several parts, "of this part"; in points for a sheet of
    # points, the too-few line too.
    points = uncovered(6, "point", several=True)
    assert points["lines"] == [
        'Only 1 of the 7 points had "trees"; the map agreed on 1 of them. Too few '
        'to say how well the map finds "trees".',
        "You saw 6 of the map's 8 classes in no point, so this check says nothing "
        "about how well the map finds them. The rate above is for every point of "
        "this part, all classes together. Names under Details.",
    ]
    assert str(points["detail"]).startswith("Classes you saw in no point: ")
    one = uncovered(1, "point")
    assert one["lines"][-1] == (
        'You saw "Class 0" in no point, so this check says nothing about how well '
        "the map finds it. The rate above is for every point of the map's grid, all "
        "classes together."
    )
    # A class typed outside the map's classes is not one of the map's own:
    # not counted among them, and no too-few line (the map never finds it).
    typed = {"code": "x", "name": "Kelp", "seen": 2, "agreed": 0, "outside": True}
    counts = {"n": 9, "n_unjudged": 0, "classes": seen + unseen(6) + [typed]}
    outside = server._uncovered(counts, "square", 9, several=False)
    assert "of the map's 8 classes" in outside["lines"][-1]
    assert not any('"Kelp"' in line for line in outside["lines"])
    assert '"Kelp"' not in str(outside["names"])
    # Cannot tell in the counts (F1): the rare class by name only, never its
    # counts, on the face or in Details; the unseen line kept.
    marked = {"n": 8, "n_unjudged": 1, "classes": seen + unseen(1)}
    told = server._uncovered(marked, "square", 8, several=False)
    assert told["lines"] == [
        'Too few squares to say how well the map finds: "trees".',
        'You saw "Class 0" in no square, so this check says nothing about how well '
        "the map finds it. The rate above is for every square of the map, all "
        "classes together.",
    ]
    assert told["few_detail"] == []
    assert server._uncovered(None, "square", 7, several=False) == {
        "lines": [],
        "few_detail": [],
        "detail": None,
        "names": None,
    }
    # The figure's own rows say none of it.
    cells = [{"seen": "a", "wrong": False}] * 5 + [{"seen": "b", "wrong": False}]
    figure = server._squares(
        {"face": "class", "cells": cells, "classes": seen + unseen(6)}, {}, "point"
    )
    assert figure is not None
    assert [c["line"] for c in figure["classes"][:2]] == [
        'Where you saw "Water" (6 points), the map also said "Water" on 5.',
        None,
    ]
    assert figure["unseen_line"] is None
    assert str(figure["unseen_detail"]).startswith("Classes you saw in no point: ")
    # Never lower-cased, never waiting for a feature word.
    assert all("water" not in line for line in five["lines"])


def test_the_fourth_line_names_the_classes_seen_too_rarely_or_not_at_all() -> None:
    """r4-h M1 (E/said-est-1): names only, the classes no square had, then
    those seen in one to four squares (of the map's own: a class typed
    outside them is in neither); either alone; none with no counts or when
    both are empty. With Cannot tell (F1): the classes no square had in the
    card's own words ("You saw ... in no square."), the rare ones still
    named."""
    counts = {
        "n": 80,
        "n_unjudged": 0,
        "classes": [
            {"code": "1", "name": "Present", "seen": 3, "agreed": 0},
            {"code": "0", "name": "Absent", "seen": 75, "agreed": 72},
            {"code": "2", "name": "Cloud", "seen": 0, "agreed": 0},
            {"code": "9", "name": "Kelp", "seen": 2, "agreed": 0, "outside": True},
        ],
    }
    say = server._say_uncovered
    assert say(counts, "square") == (
        'No square you checked had: "Cloud". Too few squares to say how well the '
        'map finds: "Present".'
    )
    assert say(counts, "point") == (
        'No point you checked had: "Cloud". Too few points to say how well the map '
        'finds: "Present".'
    )
    few = {**counts, "classes": counts["classes"][:2]}
    assert say(few, "point") == (
        'Too few points to say how well the map finds: "Present".'
    )
    assert say({**few, "n_unjudged": 2}, "point") == (
        'Too few points to say how well the map finds: "Present".'
    )
    assert say({**counts, "n_unjudged": 2}, "square") == (
        'You saw "Cloud" in no square. Too few squares to say how well the map '
        'finds: "Present".'
    )
    assert server.SAY_SAW_NONE == "You saw {names} in no {unit}."
    unseen = {**counts, "n_unjudged": 1, "classes": counts["classes"][1:3]}
    assert say(unseen, "point") == 'You saw "Cloud" in no point.'
    plenty = {"classes": [{"code": "0", "name": "Absent", "seen": 80, "agreed": 74}]}
    assert say(plenty, "square") is None
    assert say(None, "square") is None


@pytest.mark.parametrize(
    ("rate", "headline"),
    [
        (0.0, "The map was right on every square you checked: 0.0% wrong, 95% range"),
        (1.0, "The map was wrong on every square you checked: 100.0%, 95% range"),
        (0.95, "The map is wrong on more than 9 squares in 10: 95.0%, 95% range"),
    ],
)
def test_the_headline_at_zero_one_and_near_the_top(rate: float, headline: str) -> None:
    """R1: the headline carries the rate in words with the exact rate and the
    95% range at 0, 1 and 0.95 too (at 0 the rate is written "0.0% wrong",
    round r5-b, J22: it follows "right on every square"); inline labels (no
    sheet): the part is not known, so the hedge goes with it."""
    inner = _estimate()["result"]
    inner["valid_interval"] = {**inner["valid_interval"], "estimate": rate}
    got = server._estimate_result(inner, None, None)
    assert got is not None
    view = got["view"]
    assert view["headline"] == f"{headline} 6.1% to 21.7%."
    assert view["hedge"] == server.SAY_PART_HEDGE and view["uncovered"] == []
    assert got["say_first"][0].endswith(" " + server.SAY_PART_HEDGE)
    assert "95% range 6.1% to 21.7%" in got["say_first"][0]


def test_the_last_line_says_the_area_only_for_three_classes_or_more() -> None:
    """R5: one last line, pictures or answers; the area sentence only when the
    map's own classes number more than two (the counts' own, else the
    design's population classes); the model's third line (the limits) gets
    it under the same condition (F7)."""
    area = " It says nothing about the area of any one class."
    inner = _estimate()["result"]
    two = {"population": {"class_names": {"0": "absent", "1": "present"}}}
    three = {"population": {"classes": ["water", "trees", "crops"]}}
    plain = server._estimate_result(inner, None, None, design=two)
    assert plain is not None
    assert plain["view"]["last_line"] == server.LAST_LINE_ANSWERS
    assert not plain["say_first"][2].endswith(area)
    assert plain["say_first"][2] == (
        "This compares the map with the answers given, not a visit on site; it does "
        "not say whether the map is good enough."
    )
    many = server._estimate_result(inner, None, None, design=three)
    assert many is not None
    assert many["view"]["last_line"] == (
        server.LAST_LINE_ANSWERS + " It says nothing about the area of any one class."
    )
    assert many["say_first"][2].endswith(area)
    assert many["say_first"][2] == plain["say_first"][2] + server.LAST_LINE_AREA
    assert server.LAST_LINE_AREA == area
    assert server.LAST_LINE_PICTURES == (
        "Checked against satellite pictures, not a site visit; no verdict on "
        "whether the map is good enough."
    )
    assert server._design_classes(None) is None
    assert server._design_classes({"population": {"classes": []}}) is None


def _class_row(name: Any, low: float, high: float) -> dict[str, Any]:
    """A class map's per-class row as the estimate writes it: naming its
    class as the map writes it, its valid producer's accuracy first."""
    accuracy = {"estimate": (low + high) / 2, "low": low, "high": high}
    return {"class": name, "valid": {"producer_accuracy": accuracy}}


def test_a_class_named_by_a_code_reads_class_n() -> None:
    """A class map whose design lists its classes as codes (0, 1, 2, 3): the
    lines and the legend say class "3", never a bare 3 that reads as a count;
    the class seen too rarely and the classes nobody saw are said under the
    headline (R2); the range is still matched on the class as the sheet
    wrote it."""
    got = {
        "face": "class",
        "cells": [{"seen": "1", "wrong": True}] * 2
        + [{"seen": "3", "wrong": False}] * 12,
        "classes": [
            {"code": "0", "name": "0", "seen": 0, "agreed": 0},
            {"code": "1", "name": "1", "seen": 2, "agreed": 1},
            {"code": "2", "name": "2", "seen": 0, "agreed": 0},
            {"code": "3", "name": "3", "seen": 12, "agreed": 10},
        ],
    }
    table = {str(c): _class_row(c, 0.6, 0.99) for c in range(3)}
    table["3"] = _class_row(3, 0.7, 0.95)
    shown = server._squares(got, {"per_class": table})
    assert shown is not None
    assert [c["line"] for c in shown["classes"]] == [
        None,
        None,
        None,
        'Where you saw class "3" (12 squares), the map also said class "3" on 10; '
        "agreement range: 70.0% to 95.0%.",
    ]
    assert shown["unseen_line"] is None
    under = server._uncovered(
        {"n": 14, "n_unjudged": 0, "classes": got["classes"]},
        "square",
        14,
        several=False,
    )
    assert under["lines"] == [
        'Only 2 of the 14 squares had class "1"; the map agreed on 1 of them. Too '
        'few to say how well the map finds class "1".',
        'You saw class "0", class "2" in no square, so this check says nothing '
        "about how well the map finds them. The rate above is for every square of "
        "the map, all classes together.",
    ]
    assert [c["name"] for c in shown["classes"]] == [
        'class "0"',
        'class "1"',
        'class "2"',
        'class "3"',
    ]
    assert shown["legend"] == [
        {"code": "1", "text": 'You saw class "1"'},
        {"code": "3", "text": 'You saw class "3"'},
        {"wrong": True, "text": server.LEGEND_WRONG},
    ]


def test_a_class_maps_range_is_matched_by_its_name_never_by_a_code_key() -> None:
    """A class map's rows are keyed by their place in the table and name their
    class: classes written 10, 20 and 30 take the rows naming them, not the
    rows keyed "10" or "20"; a scores map's rows name no class and are taken
    by the class's code; a row keyed by the code that names another class is
    never taken."""
    got = {
        "face": "class",
        "cells": [],
        "classes": [
            {"code": "10", "name": "10", "seen": 0, "agreed": 0},
            {"code": "20", "name": "20", "seen": 8, "agreed": 6},
            {"code": "30", "name": "30", "seen": 5, "agreed": 5},
        ],
    }
    per_class = {
        "per_class": {
            "0": _class_row(10, 0.1, 0.2),
            "1": _class_row(20, 0.4, 0.9),
            "2": {"class": 30, "producer_accuracy_exact": {"low": 0.5, "high": 1.0}},
            "20": _class_row(7, 0.01, 0.02),
        }
    }
    shown = server._squares(got, per_class)
    assert shown is not None
    assert [c["line"] for c in shown["classes"][1:]] == [
        'Where you saw class "20" (8 squares), the map also said class "20" on 6; '
        "agreement range: 40.0% to 90.0%.",
        'Where you saw class "30" (5 squares), the map also said class "30" on 5; '
        "agreement range: 50.0% to 100.0%.",
    ]
    accuracy = {"low": 0.3, "high": 0.6}
    by_code = {"per_class": {"1": {"valid": {"producer_accuracy": accuracy}}}}
    assert server._class_range(by_code, "Kelp", "1") == {"low": 0.3, "high": 0.6}
    assert server._class_range(by_code, "Kelp", "0") is None
    named = {"per_class": {"1": _class_row("Water", 0.3, 0.6)}}
    assert server._class_range(named, "Kelp", "1") is None
    assert server._class_range(named, " water ", "9") == {"low": 0.3, "high": 0.6}


def test_the_plan_lead_needs_its_sheet() -> None:
    """A plan's three lines only beside its sheet, naming the panel's button
    (sheet.OPEN_FIRST); a plan drawn on a grid over a Studio result (its
    sampling) says points, not squares."""
    with_sheet = server._lead(
        "olmoearth_plan_label_sample", {}, {"labels_csv_path": "x"}
    )
    assert with_sheet == server.PLAN_LEAD
    assert sheet.OPEN_FIRST == "Open the sheet here"
    assert with_sheet["next"] == "Press Open the sheet here and answer every square."
    assert server._lead("olmoearth_plan_label_sample", {}, {}) == {
        "what": None,
        "next": None,
        "cannot": None,
    }
    sampled = {"sampling": {"grid": "10x10", "n_valid": 96}}
    points = server._lead(
        "olmoearth_plan_label_sample", sampled, {"labels_csv_path": "x"}
    )
    assert points == {
        "what": "The plan: which points to check, picked before any answer.",
        "next": "Press Open the sheet here and answer every point.",
        "cannot": "No result until every point is answered.",
    }
    assert "square" not in " ".join(str(v) for v in points.values())


def test_the_plan_lead_reads_the_unit_off_its_design(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """r4-d-1 R3: a plan whose result carries no sampling (one drawn from the
    scores file olmoearth_review_set_from_result saved) says points when its
    design file does (sheet.window_unit, the single source of the unit), so the
    plan card and the sheet agree; a design of squares, or none, keeps squares."""
    shown = {"labels_csv_path": "x"}
    by_result = _design_file(root, BY_RESULT, name="points")
    lead = server._lead(
        "olmoearth_plan_label_sample", {"design_path": str(by_result)}, shown
    )
    assert lead == server.PLAN_LEAD_POINTS
    zone = {
        "design": "random",
        "population": {
            "source": {"scores_path": "/x/zone.json", "result_id": R, "studio": {}}
        },
    }
    squares = _design_file(root, zone, name="zone")
    lead = server._lead(
        "olmoearth_plan_label_sample", {"design_path": str(squares)}, shown
    )
    assert lead == server.PLAN_LEAD
    unread = {"design_path": str(root / "gone.json")}
    assert server._lead("olmoearth_plan_label_sample", unread, shown) == (
        server.PLAN_LEAD
    )
    # The unit is read through sheet.window_unit, never beside it.
    seen: list[Any] = []

    def unit(design: Any) -> str:
        seen.append(design)
        return sheet.UNIT_POINT

    monkeypatch.setattr(sheet, "window_unit", unit)
    lead = server._lead(
        "olmoearth_plan_label_sample", {"design_path": str(squares)}, shown
    )
    assert lead == server.PLAN_LEAD_POINTS and seen == [zone]


# --------------------------------------------------------------------------- a review set


ROWS = [
    {
        "rank": 1,
        "window_index": 5,
        "row": 1,
        "col": 1,
        "margin": 0.01,
        "class_name": "water",
    },
    {
        "rank": 2,
        "window_index": 2,
        "row": 0,
        "col": 2,
        "margin": 0.02,
        "class_name": "trees",
    },
]
CENTRES = [[12.3 + 0.001 * i, 48.7 - 0.001 * i] for i in range(8)]


def _scores_file(path: Path, *, raster: bool = True, period: bool = True) -> Path:
    """A scores file of 8 windows, recording its raster, window size and
    centres (``raster``) and its period (``period``)."""
    payload: dict[str, Any] = {
        "format": "olmoearth-agent/scores@1",
        "grid": [2, 4],
        "scores": [[0.5, 0.5]] * 8,
        "centres_lon_lat": CENTRES,
    }
    if raster:
        payload.update(
            patch_px=4,
            raster={
                "file": "m.tif",
                "path": str(path.parent / "m.tif"),
                "shape": [8, 16],
            },
        )
    if period:
        payload["date_window"] = {"start": "2024-04-01", "end": "2024-06-30"}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _inner(**extra: Any) -> dict[str, Any]:
    return {"n_review": 2, "n_windows": 8, "review": copy.deepcopy(ROWS), **extra}


def test_why_no_pictures_by_cause(root: Path, tmp_path: Path) -> None:
    """Each reason by its cause: a file with its map but no period; one with
    no map; one outside the read roots, or one that cannot be read; a Studio
    result's sampled points, whatever its file holds."""
    tool = "olmoearth_review_set"
    undated = _scores_file(root / "undated.json", period=False)
    block = review.review_view(tool, _inner(), {"scores_path": str(undated)})
    assert block is not None and block["why_none"] == review.NO_PERIOD_RECORDED
    assert block["pictures"] is False and block["key"] is None
    placeless = _scores_file(root / "placeless.json", raster=False)
    block = review.review_view(tool, _inner(), {"scores_path": str(placeless)})
    assert block is not None and block["why_none"] == review.NO_PLACE_RECORDED
    outside = _scores_file(tmp_path / "elsewhere" / "far.json")
    block = review.review_view(tool, _inner(), {"scores_path": str(outside)})
    assert block is not None and block["why_none"] == review.NOT_READ_HERE
    broken = root / "broken.json"
    broken.write_text("not json", encoding="utf-8")
    block = review.review_view(tool, _inner(), {"scores_path": str(broken)})
    assert block is not None and block["why_none"] == review.NOT_READ_HERE
    studio = review.review_view(
        "olmoearth_review_set_from_result",
        _inner(result_id=R, prediction_id=P, scores_path=str(placeless)),
        {"result_id": R},
    )
    assert studio is not None and studio["why_none"] == review.STUDIO_POINTS
    assert studio["check_request"] == {"result_id": R, "prediction_id": P}
    # The panel words a Studio result's windows as points, a file's as squares.
    assert (studio["unit"], block["unit"]) == ("point", "square")
    assert review.where_of({"date_window": {"start": "a", "end": "b"}}, 0) == (
        "none",
        review.NO_PLACE_RECORDED,
    )
    placed = json.loads(_scores_file(root / "placed.json").read_text("utf-8"))
    assert review.where_of(placed, 8) == ("map", None)
    for sentence in (
        review.NO_PERIOD_RECORDED,
        review.STUDIO_POINTS,
        review.NOT_READ_HERE,
    ):
        assert "window" not in sentence and "/" not in sentence


def test_a_review_set_counts_its_whole_set_and_a_direct_run_has_no_check_request(
    root: Path,
) -> None:
    """140 in the set, 50 listed: the block says both; a direct run's set
    carries no Check request; a Studio result's with no prediction id names
    the result alone."""
    rows = [
        {"rank": i, "window_index": i, "row": 0, "col": i, "margin": 0.01 * i}
        for i in range(1, 51)
    ]
    block = review.review_view(
        "olmoearth_review_set",
        {"n_review": 140, "n_windows": 2800, "review": rows},
        {"scores": [[0.5, 0.5]] * 3},
    )
    assert block is not None
    assert (block["n_review"], block["n_listed"]) == (140, 50)
    assert block["why_none"] == review.NO_SCORES_FILE
    assert "check_request" not in block
    studio = review.review_view(
        "olmoearth_review_set_from_result",
        {"budgets": [{"n_review": 7}], "review": rows[:7], "result_id": f" {R} "},
        {"result_id": R},
    )
    assert studio is not None and studio["n_review"] == 7
    assert studio["check_request"] == {"result_id": R, "prediction_id": None}
    assert studio["why_none"] == review.STUDIO_POINTS


@pytest.mark.asyncio
async def test_the_review_panes_test_scene_is_in_review_words(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``review: true`` with the test scene asks for the review set's words;
    the sheet's pane asks as before."""
    asked: list[dict[str, Any]] = []

    def scene(**kwargs: Any) -> dict[str, Any]:
        asked.append(kwargs)
        return {"ok": True, "caption": "c"}

    monkeypatch.setattr(server.imagery, "public_test_scene", scene)
    app = _app()
    await app.app_tool(
        "olmoearth_app_window_image", {"test_scene": True, "review": True}
    )
    await app.app_tool("olmoearth_app_window_image", {"test_scene": True})
    assert asked == [{"review": True}, {}]
    await app.aclose()


# --------------------------------------------------------------------------- the views and Studio


def test_the_check_and_result_views_inline_the_figures() -> None:
    """The two figure-first cards load the shared figures between common.js
    and sheet.js."""
    for name in ("check", "result"):
        assert views.VIEWS[name]["scripts"] == ("common.js", "figures.js", "sheet.js")
        assert "OE.fig =" in views.view_html(name), name
    page = views.view_html("check")
    for word in ("Pick the squares", "How many squares do you want to check?"):
        assert word in page


def test_a_class_map_beside_a_regression_is_sent_to_the_plan_tool() -> None:
    """The third refusal names the plan tool for the class map and a threshold
    for the regression band, not the look-first job."""
    record = {
        "id": R,
        "prediction_id": P,
        "property_names": ["landcover", "score"],
        "result_metadata": {
            "classification_fields": [
                {
                    "property_name": "landcover",
                    "allowed_values": [{"value": 0, "label": "absent"}],
                    "band_index": 1,
                }
            ],
            "regression_fields": [
                {
                    "property_name": "score",
                    "band_index": 2,
                    "min_value": 0.0,
                    "max_value": 1.0,
                }
            ],
        },
    }
    with pytest.raises(ValueError, match="beside the regression field") as info:
        studio_scores.files_plan(record)
    text = str(info.value)
    assert "olmoearth_review_set_from_result" not in text
    assert (
        "olmoearth_plan_label_sample plans labels (design 'random' or 'class')" in text
    )
    assert text.endswith("the regression band needs a threshold and its property_name.")


def test_the_zone_sentences_say_part_within_the_contracts_words() -> None:
    """The reader test's A/check-10 and A/check-11: one part, not the whole
    map; at most 25 words each (the output contract's cap)."""
    from olmoearth_agent.analysis.output_contract import (
        MUST_STATE_MAX_WORDS,
        word_count,
    )

    for sentence in (studio_scores.ZONES_SENTENCE, studio_scores.NATIVE_SENTENCE):
        assert word_count(sentence) <= MUST_STATE_MAX_WORDS
        assert "part" in sentence and "Note:" not in sentence
    assert studio_scores.ZONES_SENTENCE.endswith("not the whole map.")
    assert "olmoearth_pixel_value" not in studio_scores.NATIVE_SENTENCE
