# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""The Check card's parts, the map's name and rule, and the records read back.

``mcp_app/check.py`` names each zone of a result as the cards call it
(:func:`check.part_names`), keeps the map's name, period, the zone's share
and whether the map has an owner's labelling rule in each offer record, pins
the user's rule with a draw (:func:`check.claim_draw`), and reads the
records back: the rule an earlier part pinned (:func:`check.map_rule`) and
which part of which map a sheet is (:func:`check.sheet_context`, which says
when the part count is not known). No network, no package.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from olmoearth_agent.mcp_app import check

OFFER = {
    "offer": [
        {"labels": 20, "width": 0.2, "sentence": "Twenty.", "rate_note": "r"},
        {"labels": 40, "width": 0.1, "sentence": "Forty.", "rate_note": "r"},
    ],
    "refused": None,
}


def _file(code: int, i: int = 0) -> str:
    """A zone's native file name for an EPSG code."""
    return f"result_epsg{code}_{i}_native.tif"


@pytest.fixture(autouse=True)
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The write root (scores and outputs), a folder of the test's own."""
    out = tmp_path / "out"
    out.mkdir()
    monkeypatch.setenv("OLMOEARTH_SCORES_ROOT", str(out))
    monkeypatch.setenv("OLMOEARTH_OUTPUT_ROOT", str(out))
    return out.resolve()


@pytest.fixture
def canned(monkeypatch: pytest.MonkeyPatch) -> None:
    """Counts offered without the package: 20 and 40, for every zone."""

    def fake(n_valid: int, widths: Any = check.CHECK_WIDTHS) -> dict[str, Any]:
        """A fresh copy of :data:`OFFER`."""
        answer: dict[str, Any] = json.loads(json.dumps(OFFER))
        return answer

    monkeypatch.setattr(check, "offers", fake)


def _inner(root: Path, files: tuple[str, ...], rid: str = "res-1") -> dict[str, Any]:
    """A scores_from_studio result of these zones, each with a scores file, its
    shares 47% and 53% (two zones) and the period."""
    zones = []
    for i, file in enumerate(files):
        path = root / f"{rid}_scores_{i}.json"
        path.write_text("{}", encoding="utf-8")
        zones.append(
            {
                "file": file,
                "scores_path": str(path),
                "pooled": {"n_valid": 600, "raster_check": {"sha256": f"{i}" * 64}},
            }
        )
    shares = (0.467, 0.533) if len(files) == 2 else tuple(1 / len(files) for _ in files)
    facts = (
        [
            {
                "id": "studio_zones",
                "sentence": "Several zones.",
                "zones": [{"file": f, "share": s} for f, s in zip(files, shares)],
            }
        ]
        if len(files) > 1
        else []
    )
    return {
        "result_id": rid,
        "prediction_id": "pred-1",
        "zones": zones,
        "facts": facts,
        "date_window": {"start": "2024-04-01T00:00:00Z", "end": "2024-06-30T00:00:00Z"},
    }


def _drawn(root: Path, key: str, name: str) -> Path:
    """Finish a claimed draw as drawn, with a sheet file of its own."""
    sheet = root / f"{name}_to_label.csv"
    sheet.write_text("order,window_index,wrong\n", encoding="utf-8")
    answer = {
        "ok": True,
        "result": {
            "available": True,
            "labels_csv_path": str(sheet),
            "design_path": str(root / f"{name}.json"),
            "n_population": 600,
            "must_state": ["Stated."],
        },
    }
    assert check.finish_draw(key, answer)["state"] == "drawn"
    return sheet


# --------------------------------------------------------------------------- part names


@pytest.mark.parametrize(
    ("files", "names"),
    [
        ((_file(32610), _file(32710, 1)), ["North part", "South part"]),
        ((_file(32610), _file(32611, 1)), ["West part", "East part"]),
        ((_file(32611), _file(32610, 1)), ["East part", "West part"]),  # order kept
        ((_file(32659), _file(32660, 1)), ["Part 1", "Part 2"]),  # zone 60
        ((_file(32660), _file(32601, 1)), ["Part 1", "Part 2"]),  # the antimeridian
        ((_file(32610), _file(32711, 1)), ["Part 1", "Part 2"]),  # number and side
        ((_file(32610), _file(32612, 1)), ["Part 1", "Part 2"]),  # two apart
        (
            (_file(32610), _file(32611, 1), _file(32612, 2)),
            ["Part 1", "Part 2", "Part 3"],
        ),
        ((_file(3857), _file(32610, 1)), ["Part 1", "Part 2"]),  # not UTM
        ((_file(32610), _file(32610, 1)), ["Part 1", "Part 2"]),  # the same EPSG
        ((_file(32610),), [None]),  # one file: the whole map
        (("scores.tif", _file(32610, 1)), ["Part 1", "Part 2"]),  # not a zone file
    ],
)
def test_part_names(files: tuple[str, ...], names: list[str | None]) -> None:
    """A compass word only for two UTM files where it is certain; else Part n."""
    assert check.part_names(files) == names


def test_no_files_no_names() -> None:
    """No zone: no name."""
    assert check.part_names([]) == []


# --------------------------------------------------------------------------- the offer


def test_offer_zones_names_the_parts_and_keeps_the_map(
    root: Path, canned: None
) -> None:
    """Each zone carries its part's name and its share as a fraction; the offer
    record keeps the map's name, period, every zone's file, the part, the
    share, how many parts and whether the map needs the user's rule."""
    files = (_file(32610), _file(32611, 1))
    zones = check.offer_zones(_inner(root, files), map_name="Kelp Q2", rule_needed=True)
    assert [z["part"] for z in zones] == ["West part", "East part"]
    assert [z["share_value"] for z in zones] == [0.467, 0.533]
    assert [z["share"] for z in zones] == ["46.7%", "53.3%"]
    record = check.read_offer(zones[0]["check_key"])
    assert record["map_name"] == "Kelp Q2"
    assert record["period"] == {
        "start": "2024-04-01T00:00:00Z",
        "end": "2024-06-30T00:00:00Z",
    }
    assert record["files"] == list(files)
    assert (record["part"], record["share_value"], record["n_parts"]) == (
        "West part",
        0.467,
        2,
    )
    assert record["rule_needed"] is True
    assert record["rule_owner"] is False  # no owner's rule to replace
    # The page's zones still hold no scores path or digest.
    text = json.dumps(zones)
    assert "scores_path" not in text and "0" * 64 not in text


def test_one_zone_is_the_whole_map(root: Path, canned: None) -> None:
    """One zone: no part name, no share, the rule not needed by default."""
    (zone,) = check.offer_zones(_inner(root, (_file(32610),)))
    assert zone["part"] is None and zone["share_value"] is None
    record = check.read_offer(zone["check_key"])
    assert (record["n_parts"], record["rule_needed"], record["map_name"]) == (
        1,
        False,
        None,
    )
    assert record["rule_owner"] is False  # by default: no box


def test_a_map_of_presence_and_absence_records_its_rule_owner(
    root: Path, canned: None
) -> None:
    """The owner's ruling of 13 Nov 2026, option A: a map whose owner's rule
    the user may replace keeps rule_needed False and records rule_owner."""
    (zone,) = check.offer_zones(
        _inner(root, (_file(32610),)), rule_needed=False, rule_owner=True
    )
    record = check.read_offer(zone["check_key"])
    assert (record["rule_needed"], record["rule_owner"]) == (False, True)
    assert "rule_owner" not in zone  # the page's zone: as before


def test_a_part_drawn_or_being_drawn_settles_the_maps_rule(
    root: Path, canned: None
) -> None:
    """result_settled: no records, no draw, a failed draw or one left drawing
    past STALE_SECONDS, or another result's draw: not settled; a draw being
    drawn now, or drawn: settled."""
    assert check.result_settled("res-1") is False  # no records folder yet
    first, second = check.offer_zones(
        _inner(root, (_file(32610), _file(32611, 1))), rule_owner=True
    )
    (other,) = check.offer_zones(_inner(root, (_file(32610),), rid="res-2"))
    assert check.result_settled("res-1") is False
    check.claim_draw(other["check_key"], 20)
    _drawn(root, other["check_key"], "other")
    assert check.result_settled("res-1") is False
    assert check.result_settled(" res-2 ") is True
    check.claim_draw(first["check_key"], 20)
    assert check.result_settled("res-1") is True  # being drawn now
    check.finish_draw(first["check_key"], {"ok": False, "error": "OSError: I/O"})
    assert check.result_settled("res-1") is False  # failed: nothing drawn
    check.claim_draw(second["check_key"], 40)
    later = check._now() + check.STALE_SECONDS + 1
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(check, "_now", lambda: later)
        assert check.result_settled("res-1") is False  # left drawing: stopped
    _drawn(root, second["check_key"], "second")
    assert check.result_settled("res-1") is True
    assert check.result_settled("") is False and check.result_settled(None) is False


# --------------------------------------------------------------------------- the rule


def test_a_first_press_pins_the_rule_and_a_retry_keeps_it(
    root: Path, canned: None
) -> None:
    """The rule the server took is pinned with the press's time; a retry keeps
    the rule pinned, whatever rule it is given; a press with none pins none."""
    first, second = check.offer_zones(
        _inner(root, (_file(32610), _file(32611, 1))), rule_needed=True
    )
    pressed = "2026-11-13T09:00:00Z"
    claim = check.claim_draw(
        first["check_key"], 20, pressed, reading_rule="Kelp: canopy over half."
    )
    draw = claim["draw"]
    assert (draw["reading_rule"], draw["rule_typed_at"]) == (
        "Kelp: canopy over half.",
        pressed,
    )
    stored = json.loads(
        (root / "checks" / f"{first['check_key']}.draw.json").read_text("utf-8")
    )
    assert stored["reading_rule"] == "Kelp: canopy over half."
    check.finish_draw(first["check_key"], {"ok": False, "error": "OSError: I/O"})
    again = check.claim_draw(
        first["check_key"],
        20,
        "2026-11-13T10:00:00Z",
        reading_rule="Another rule here.",
    )
    assert again["outcome"] == "retry"
    assert again["draw"]["reading_rule"] == "Kelp: canopy over half."
    assert again["draw"]["rule_typed_at"] == pressed
    plain = check.claim_draw(second["check_key"], 40)["draw"]
    assert "reading_rule" not in plain and "rule_typed_at" not in plain


def test_a_rule_taken_from_an_earlier_part_keeps_the_time_it_was_typed(
    root: Path, canned: None
) -> None:
    """Pass-3 finding Q7: a rule pinned with the time it was typed (an
    earlier part's) keeps that time on a first press and on a retry; with no
    time given, the press's own."""
    first, second = check.offer_zones(
        _inner(root, (_file(32610), _file(32611, 1))), rule_owner=True
    )
    typed_at = "2026-11-13T09:00:00Z"
    claim = check.claim_draw(
        first["check_key"],
        20,
        "2026-11-15T09:00:00Z",
        reading_rule="Seagrass: leaves cover most.",
        rule_typed_at=typed_at,
    )
    assert claim["draw"]["rule_typed_at"] == typed_at
    assert claim["draw"]["pressed_at"] == "2026-11-15T09:00:00Z"
    # A retry of a draw that pinned none pins the rule with its given time.
    check.claim_draw(second["check_key"], 40, "2026-11-14T00:00:00Z")
    check.finish_draw(second["check_key"], {"ok": False, "error": "OSError: I/O"})
    retry = check.claim_draw(
        second["check_key"],
        40,
        "2026-11-16T00:00:00Z",
        reading_rule="Seagrass: leaves cover most.",
        rule_typed_at=typed_at,
    )
    assert retry["outcome"] == "retry"
    assert retry["draw"]["rule_typed_at"] == typed_at
    assert check.map_rule("res-1") == {
        "text": "Seagrass: leaves cover most.",
        "typed_at": typed_at,
    }


def test_unpin_rule_takes_the_rule_off_the_draw_only(root: Path, canned: None) -> None:
    """Pass-3 finding Q9: check.unpin_rule drops a draw's pinned rule and the
    time it was typed, under the draw's lock, and keeps the rest of the
    record; then map_rule no longer reports it. A check with no draw record:
    None; a key that is not a check's: refused before any file."""
    first, second = check.offer_zones(
        _inner(root, (_file(32610), _file(32611, 1))), rule_owner=True
    )
    check.claim_draw(
        first["check_key"], 20, "2026-11-13T09:00:00Z", reading_rule="Kelp rule here."
    )
    _drawn(root, first["check_key"], "first")
    assert check.map_rule("res-1") is not None
    kept = check.unpin_rule(first["check_key"])
    assert kept is not None
    assert "reading_rule" not in kept and "rule_typed_at" not in kept
    stored = json.loads(
        (root / "checks" / f"{first['check_key']}.draw.json").read_text("utf-8")
    )
    assert stored == kept
    assert (stored["state"], stored["budget"]) == ("drawn", 20)
    assert check.map_rule("res-1") is None
    assert check.unpin_rule(first["check_key"]) == kept  # nothing left to drop
    assert check.unpin_rule(second["check_key"]) is None  # no draw record
    with pytest.raises(ValueError, match=check.BAD_KEY):
        check.unpin_rule("../x")


def test_map_rule_is_the_earliest_pinned_of_the_result(
    root: Path, canned: None
) -> None:
    """An earlier part's pinned rule is the map's; another result's is not."""
    assert check.map_rule("res-1") is None  # no records folder yet
    first, second = check.offer_zones(
        _inner(root, (_file(32610), _file(32611, 1))), rule_needed=True
    )
    (other,) = check.offer_zones(_inner(root, (_file(32610),), rid="res-2"))
    check.claim_draw(
        other["check_key"], 20, "2026-11-01T00:00:00Z", reading_rule="Other."
    )
    assert check.map_rule("res-1") is None
    check.claim_draw(
        second["check_key"], 40, "2026-11-13T10:00:00Z", reading_rule="Later rule."
    )
    check.claim_draw(
        first["check_key"], 20, "2026-11-13T09:00:00Z", reading_rule="Earlier rule."
    )
    assert check.map_rule(" res-1 ") == {
        "text": "Earlier rule.",
        "typed_at": "2026-11-13T09:00:00Z",
    }
    assert check.map_rule("res-2") == {
        "text": "Other.",
        "typed_at": "2026-11-01T00:00:00Z",
    }
    assert check.map_rule("") is None


# --------------------------------------------------------------------------- the context


def test_sheet_context_names_the_part_and_the_other_parts(
    root: Path, canned: None
) -> None:
    """A drawn sheet is found by its real path: its result, part, map's name
    and period, and each part in order with its share and, once drawn, its
    sheet; a sheet no card drew, or a path that is not a string, is None."""
    files = (_file(32610), _file(32611, 1))
    west, east = check.offer_zones(
        _inner(root, files), map_name="Kelp Q2", rule_needed=False
    )
    check.claim_draw(west["check_key"], 20)
    sheet = _drawn(root, west["check_key"], "west")
    got = check.sheet_context(str(sheet))
    assert got is not None
    assert (got["result_id"], got["zone_file"], got["part"], got["n_parts"]) == (
        "res-1",
        files[0],
        "West part",
        2,
    )
    assert got["parts_known"] is True
    assert got["map_name"] == "Kelp Q2"
    assert got["period"]["start"] == "2024-04-01T00:00:00Z"
    # Round r5-n (numbers/F2): each part says whether it was read as a map.
    assert got["parts"] == [
        {
            "part": "West part",
            "share_value": 0.467,
            "labels_csv_path": str(sheet),
            "this": True,
            "readable": True,
        },
        {
            "part": "East part",
            "share_value": 0.533,
            "labels_csv_path": None,
            "this": False,
            "readable": True,
        },
    ]
    # Once the other part is drawn, its sheet is named for the server.
    check.claim_draw(east["check_key"], 40)
    other = _drawn(root, east["check_key"], "east")
    again = check.sheet_context(str(sheet))
    assert again is not None and again["parts"][1]["labels_csv_path"] == str(other)
    from_east = check.sheet_context(str(other))
    assert from_east is not None and from_east["part"] == "East part"
    assert [p["this"] for p in from_east["parts"]] == [False, True]
    stray = root / "stray_to_label.csv"
    stray.write_text("x\n", encoding="utf-8")
    assert check.sheet_context(str(stray)) is None
    assert check.sheet_context(None) is None
    assert check.sheet_context("") is None


def test_sheet_context_of_one_part(root: Path, canned: None) -> None:
    """A map of one part: no part name, one part listed."""
    (zone,) = check.offer_zones(_inner(root, (_file(32610),)))
    check.claim_draw(zone["check_key"], 20)
    sheet = _drawn(root, zone["check_key"], "only")
    got = check.sheet_context(str(sheet))
    assert got is not None
    assert (got["part"], got["n_parts"], got["parts_known"]) == (None, 1, True)
    assert [p["this"] for p in got["parts"]] == [True]


def _drop_files(root: Path, key: str) -> None:
    """Rewrite a check's offer record as one written before the files were
    kept (no ``files``)."""
    place = root / "checks" / f"{key}.json"
    record = json.loads(place.read_text("utf-8"))
    del record["files"]
    place.write_text(json.dumps(record), encoding="utf-8")


def test_an_offer_record_without_its_files_does_not_say_one_part(
    root: Path, canned: None
) -> None:
    """An offer record written before the files were kept: the map's files
    come from another offer record of the same result that keeps them, so
    the part is named; with none, the sheet stands alone and the part count
    is not known (parts_known false), never taken for a map of one part."""
    files = (_file(32610), _file(32611, 1))
    west, east = check.offer_zones(_inner(root, files), map_name="Kelp Q2")
    check.claim_draw(west["check_key"], 20)
    sheet = _drawn(root, west["check_key"], "west")
    _drop_files(root, west["check_key"])
    got = check.sheet_context(str(sheet))
    assert got is not None
    assert (got["part"], got["n_parts"], got["parts_known"]) == ("West part", 2, True)
    assert [p["part"] for p in got["parts"]] == ["West part", "East part"]
    # Another result's record that keeps its files says nothing of this one.
    check.offer_zones(_inner(root, (_file(32610),), rid="res-2"))
    _drop_files(root, east["check_key"])
    alone = check.sheet_context(str(sheet))
    assert alone is not None
    assert (alone["part"], alone["n_parts"], alone["parts_known"]) == (None, 1, False)
    assert [p["this"] for p in alone["parts"]] == [True]
    assert alone["map_name"] == "Kelp Q2"


def test_a_drawn_zone_carries_its_sheets_counts(
    root: Path, canned: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A drawn zone's counts go with its progress words (counts only), so the
    card's heading can say when every square is answered."""
    (zone,) = check.offer_zones(_inner(root, (_file(32610),)))
    check.claim_draw(zone["check_key"], 20)
    _drawn(root, zone["check_key"], "counted")

    def progress(path: str) -> dict[str, Any]:
        """The sheet's counts, as sheet.sheet_progress gives them."""
        return {
            "progress": "7 of 20 labelled",
            "open_label": "Continue labelling",
            "n_rows": 20,
            "n_labelled": 7,
            "n_unjudged": 0,
        }

    monkeypatch.setattr(check.sheet, "sheet_progress", progress)
    (again,) = check.offer_zones(_inner(root, (_file(32610),)))
    draw = again["draw"]
    assert (draw["progress"], draw["n_rows"], draw["n_labelled"]) == (
        "7 of 20 labelled",
        20,
        7,
    )
