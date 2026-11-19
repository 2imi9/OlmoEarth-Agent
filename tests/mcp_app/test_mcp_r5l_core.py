# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Round r5-l, track CORE (L1, 19 Nov 2026): the app's rule is general.

The owner, 19 Nov 2026: the instructions are general; kelp is not used to
state what people are asked. So:

- ``imagery.READING_RULE`` is general, for any map of present and absent
  (round r5-m B1: it says what the map is meant to find; r5-l's words, "what
  the map shows", are ``imagery.R5L_READING_RULE``, kept for the records made
  with them: tests/mcp_app/test_mcp_r5m_sheet.py); ``imagery.RULE_SCOPE`` says
  it came with the app for any such map, with no date and no kelp.
- ``imagery.RULE_KELP_ONLY`` and ``rule_limit`` are gone: the page copy, the
  counts, the pictures' answer and the estimate's note to Claude say only the
  rule and its scope.
- Records already written keep their words: a draw, a sheet or a result made
  with the earlier kelp rule still shows and repeats that rule and its old
  scope line (``imagery.KELP_READING_RULE``, ``imagery.KELP_RULE_SCOPE``),
  chosen by what the record holds, in this order: the rule its pictures'
  record printed last (read only while the sheet holds an answer: beside a
  blank sheet it can be an earlier sheet's of the same name), the rule its
  graded mark kept (none on a mark written before the round: the kelp rule),
  the rule its Check card's draw pinned (none on a draw recorded before the
  round: the kelp rule). A map's card, and each new part of it, takes the rule
  a part was drawn with (a press that drew nothing does not count), else the
  kelp rule when a sheet of the map holds answers under it. Nothing is
  written again.
- Claude's notes: the card's box is for the map maker's meaning; without
  panels Claude gives the rule and asks whether the map maker meant something
  else; no note asks whether a map is of kelp.
- ``imagery.TEST_CAPTION`` no longer says "as on the kelp map".

No sheet here holds a place: the sheets have no lon or lat column.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

import pytest

mcp = pytest.importorskip("mcp")

from olmoearth_agent.mcp_app import check, imagery, server, sheet  # noqa: E402

#: The spec's general rule and scope, word for word (the rule as round r5-m
#: B1 words it: what the map is meant to find).
GENERAL_RULE = (
    "Present: on at least one clear date shown, what the map is meant to find "
    "covers more than half of the outlined square. Absent: on every clear date "
    "shown, it covers half or less."
)
#: The general rule as round r5-l worded it, kept for the records made with it
#: (round r5-m B1), with the general scope.
R5L_RULE = (
    "Present: on at least one clear date shown, what the map shows covers more "
    "than half of the outlined square. Absent: on every clear date shown, it "
    "covers half or less."
)
GENERAL_SCOPE = "This rule came with the app, for any map of present and absent."
#: The rule and scope before the round, word for word.
OLD_RULE = (
    "Present: on at least one clear date shown, canopy fills more than half of "
    "the outlined square. Absent: on every clear date shown it fills half or less."
)
OLD_SCOPE = "This rule came with the app (set 11 Nov 2026 for kelp canopy maps)."
KELP_ONLY = "Any other map needs its own rule."

#: A sheet's columns, with no place.
HEADER = [
    "order",
    "window_index",
    "row",
    "col",
    "map_class",
    "stratum",
    "wrong",
    "reference_class",
]
#: The map's class of each of 20 windows: present at 4 and 12.
ON_MAP = [1 if w in (4, 12) else 0 for w in range(20)]
WINDOWS = (4, 9, 12, 3)


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The output root (the tools' write root and the checks' folder's)."""
    out = tmp_path / "outputs"
    out.mkdir()
    monkeypatch.setenv("OLMOEARTH_OUTPUT_ROOT", str(out))
    monkeypatch.delenv("OLMOEARTH_SCORES_ROOT", raising=False)
    return out.resolve()


def _source(module: Any) -> str:
    """A module's source, as text."""
    return Path(module.__file__).read_text(encoding="utf-8")


def _sheet(
    root: Path,
    stem: str = "present",
    *,
    result_id: str | None = None,
    names: tuple[str, str] = ("absent", "present"),
) -> Path:
    """A plan of a map of presence and absence: its design (class names
    absent and present, or ``names``; the Studio result it names, when
    given) and its sheet of four windows, no answer, no place."""
    population: dict[str, Any] = {
        "score_kind": "window_confidence",
        "class_names": {"0": names[0], "1": names[1]},
        "map_class": ON_MAP,
    }
    if result_id is not None:
        population["source"] = {"result_id": result_id}
    design = {"design": "random", "population": population}
    (root / f"{stem}.json").write_text(json.dumps(design), encoding="utf-8")
    path = root / f"{stem}_to_label.csv"
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(HEADER)
        for order, w in enumerate(WINDOWS, start=1):
            writer.writerow(
                [str(order), str(w), "0", str(w), str(ON_MAP[w]), "0", "", ""]
            )
    return path


def _answer_all(path: Path) -> None:
    """Every square answered, as the panel saves answers."""
    sheet.save_labels(
        str(path),
        [{"window_index": w, "reference_class": str(ON_MAP[w])} for w in WINDOWS],
    )


def _said(path: Path) -> list[tuple[Any, ...]]:
    """The rule as labelling_rule, the page copy and the counts carry it."""
    rule = sheet.labelling_rule(str(path)) or {}
    read, progress = sheet.read_sheet(str(path)), sheet.sheet_progress(str(path))
    keys = ("reading_rule", "rule_scope", "rule_source")
    return [
        (rule.get("text"), rule.get("scope"), rule.get("source")),
        tuple(read[k] for k in keys),
        tuple(progress[k] for k in keys),
    ]


def _draw_record(root: Path, key: str, result_id: str, **fields: Any) -> Path:
    """A Check card's draw record, as check.claim_draw and finish_draw leave
    one drawn (``fields`` added: the sheet, the app's rule, a typed rule)."""
    folder = root / "checks"
    folder.mkdir(mode=0o700, exist_ok=True)
    record = {
        "format": check.DRAW_FORMAT,
        "check_key": key,
        "result_id": result_id,
        "state": "drawn",
        "pressed_at": "2026-11-17T09:00:00Z",
        **fields,
    }
    place = folder / f"{key}.draw.json"
    place.write_text(json.dumps(record), encoding="utf-8")
    os.chmod(place, 0o600)
    return place


def _offer(
    root: Path,
    result_id: str,
    zone: str = "result_epsg32610_0_native.tif",
    **fields: Any,
) -> str:
    """A Check card's offer of one zone of ``result_id`` (20 labels offered,
    a map of presence and absence unless ``fields`` say otherwise), as
    check.offer_zones writes one; its check key."""
    key = check.check_key(result_id, zone)
    folder = root / "checks"
    folder.mkdir(mode=0o700, exist_ok=True)
    scores = root / f"scores_{key}.json"
    scores.write_text("{}", encoding="utf-8")
    record = {
        "format": check.CHECK_FORMAT,
        "check_key": key,
        "result_id": result_id,
        "zone_file": zone,
        "scores_path": str(scores),
        "raster_sha256": "a" * 64,
        "offer": [{"labels": 20}],
        "refused": None,
        "rule_needed": False,
        "rule_owner": True,
        **fields,
    }
    place = folder / f"{key}.json"
    place.write_text(json.dumps(record), encoding="utf-8")
    os.chmod(place, 0o600)
    return key


def _undrawn(root: Path, key: str, result_id: str, state: str, **fields: Any) -> Path:
    """A draw record written before the round (no app_rule) that drew
    nothing: ``failed`` (retried by a press, unless ``fields`` say not) or
    ``drawing`` and stale (started long ago), with its pinned request."""
    stamp = {"started_at": 1.0} if state == "drawing" else {"retry": True}
    return _draw_record(
        root,
        key,
        result_id,
        state=state,
        budget=20,
        raster_sha256="a" * 64,
        request={"scores_path": None, "design": "random", "budget": 20, "seed": 0},
        **{**stamp, **fields},
    )


def _pictured(path: Path, *rules: str) -> Path:
    """The sheet's pictures' record, its context lines printing ``rules`` in
    order (as each open of the pictures with another rule appends one)."""
    record = imagery.record_path(path.resolve())
    with open(record, "w", encoding="utf-8") as fh:
        for rule in rules:
            context = {
                "record": "olmoearth-agent label-sheet pictures",
                "reading_rule": rule,
                "rule_scope": imagery.APP_RULES[rule],
            }
            fh.write(json.dumps({"context": context}) + "\n")
    return record


def _graded_before_the_round(path: Path) -> Path:
    """The sheet graded, its mark as a grading before the round wrote it (no
    app_rule in its first grading); the mark's place."""
    sheet.record_grading(str(path), sheet.sheet_version(str(path)))
    place = path.with_name("." + path.name + sheet.GRADED_SUFFIX)
    mark = json.loads(place.read_text(encoding="utf-8"))
    del mark["first"]["app_rule"]
    place.write_text(json.dumps(mark, sort_keys=True), encoding="utf-8")
    os.chmod(place, 0o600)
    return place


GENERAL = (GENERAL_RULE, GENERAL_SCOPE, "owner")
OLD = (OLD_RULE, OLD_SCOPE, "owner")


# --------------------------------------------------------------------------- the words


def test_l1_the_apps_rule_and_its_scope_are_general() -> None:
    """L1: the rule is general (since round r5-m B1 it says what the map is
    meant to find, under its WORDING: r5m B1 mark, above r5-l's) and its scope
    names no map and no date, each under its WORDING: r5l mark."""
    assert imagery.READING_RULE == GENERAL_RULE
    assert imagery.RULE_SCOPE == GENERAL_SCOPE
    for text in (imagery.READING_RULE, imagery.RULE_SCOPE):
        assert "kelp" not in text.lower() and "canopy" not in text.lower()
    assert "2026" not in imagery.RULE_SCOPE
    source = _source(imagery)
    assert "# WORDING: r5l\nREADING_RULE = (" in source
    assert "# WORDING: r5l\nRULE_SCOPE = " in source


def test_l1_the_earlier_rule_is_kept_word_for_word() -> None:
    """L1: the kelp rule and its scope stay as named constants, for the
    records made with them, and the scope is chosen by the rule's words.
    Round r5-m B1: r5-l's own words are one of the app's rules too, with the
    general scope."""
    assert imagery.KELP_READING_RULE == OLD_RULE
    assert imagery.KELP_RULE_SCOPE == OLD_SCOPE
    assert imagery.APP_RULES == {
        GENERAL_RULE: GENERAL_SCOPE,
        OLD_RULE: OLD_SCOPE,
        R5L_RULE: GENERAL_SCOPE,
    }
    assert imagery.app_rule_scope(R5L_RULE) == GENERAL_SCOPE
    assert imagery.app_rule_scope(OLD_RULE) == OLD_SCOPE
    assert imagery.app_rule_scope(GENERAL_RULE) == GENERAL_SCOPE
    assert imagery.app_rule_scope(None) == GENERAL_SCOPE
    assert "# WORDING: r4h C5\nKELP_RULE_SCOPE = " in _source(imagery)


def test_l1_the_kelp_only_line_and_rule_limit_are_gone() -> None:
    """L1: no "Any other map needs its own rule." and no rule_limit in the
    app's Python: the rule and its scope are all that is said."""
    assert not hasattr(imagery, "RULE_KELP_ONLY")
    for module in (imagery, sheet, check, server):
        source = _source(module)
        assert KELP_ONLY not in source, module.__name__
        assert '"rule_limit"' not in source, module.__name__


def test_l1_the_test_scenes_caption_names_no_kelp_map() -> None:
    """L1: the sheet's test scene says how big its square is, and no more."""
    assert imagery.TEST_CAPTION == (
        "A public test scene, not a square of this sheet: nothing here is "
        "labelled. The outlined square is 11 pixels. " + imagery.CAPTION
    )
    assert "kelp" not in imagery.TEST_CAPTION
    assert "# WORDING: squares; r5l\nTEST_CAPTION = (" in _source(imagery)


def test_l1_claudes_notes_ask_for_the_map_makers_meaning() -> None:
    """L1: with panels, the card's box is for the map maker's meaning, which
    then replaces the app's rule, and Claude does not write it for the user;
    without panels, Claude gives the rule and asks whether the map maker meant
    something else, then for that meaning in one sentence, repeated with the
    result. No note asks whether a map is of kelp."""
    assert server.CHECK_OWNER_RULE_NOTE == (
        "The card shows the app's general labelling rule with a box beside it: if "
        "the map maker meant something else by Present and Absent, the user writes "
        "that meaning there before the first press, and it then replaces the app's "
        "rule for this map. Do not write it for them unless they ask."
    )
    assert server.CHECK_TEXT_OWNER_RULE_NOTE == (
        "This map takes the labelling rule that came with the app (labelling_rule, "
        "with its scope). Before any plan, give the user that rule and ask whether "
        "the map maker meant something else by Present and Absent; if so, ask for "
        "their meaning in one sentence and repeat it with the result instead of the "
        "app's rule."
    )
    notes = [
        value
        for name, value in vars(server).items()
        if name.isupper() and isinstance(value, str)
    ]
    for note in notes:
        assert "of kelp" not in note and "kelp rule" not in note, note
    source = _source(server)
    assert "# WORDING: r5l\nCHECK_OWNER_RULE_NOTE = (" in source
    assert "# WORDING: r5l\nCHECK_TEXT_OWNER_RULE_NOTE = (" in source


def test_l1_the_estimates_note_says_the_rule_and_its_scope_only() -> None:
    """L1: the estimate's note to Claude gives the rule and the scope said with
    it (the kelp rule's own for a sheet made with it), and no line after the
    scope; the card's note beside the app's rule has no date, and the kelp
    rule keeps its old note."""
    assert "{only}" not in server.RULE_USED_OWNER_NOTE
    general = server.RULE_USED_OWNER_NOTE.format(
        rule=GENERAL_RULE, scope=imagery.app_rule_scope(GENERAL_RULE)
    )
    assert general.startswith(f"The sheet records the app's rule: {GENERAL_RULE!r}. ")
    assert GENERAL_SCOPE + " Give it with the rate;" in general
    assert "kelp" not in general and KELP_ONLY not in general
    old = server.RULE_USED_OWNER_NOTE.format(
        rule=OLD_RULE, scope=imagery.app_rule_scope(OLD_RULE)
    )
    assert OLD_SCOPE + " Give it with the rate;" in old and KELP_ONLY not in old
    assert server.RULE_OWNER_NOTE == "(the rule that came with the app)"
    assert server.RULE_OWNER_NOTE_KELP == "(the app owner's rule, 11 Nov 2026)"
    source = _source(server)
    assert "only=imagery.RULE_KELP_ONLY" not in source
    assert 'scope=imagery.app_rule_scope(rule_view["text"])' in source
    assert 'if rule["text"] == imagery.KELP_READING_RULE' in source


# --------------------------------------------------------------------------- a sheet


def test_l1_a_new_sheet_takes_the_general_rule_with_no_rule_limit(
    root: Path,
) -> None:
    """L1: a sheet of presence and absence with nothing typed and no record
    takes the general rule and its scope, in labelling_rule, the page copy,
    the counts and the pictures' rule; neither the page copy nor the counts
    carry rule_limit."""
    path = _sheet(root)
    assert _said(path) == [GENERAL, GENERAL, GENERAL]
    assert imagery.rule_for(path.resolve(), ["absent", "present"]) == GENERAL
    assert "rule_limit" not in sheet.read_sheet(str(path))
    assert "rule_limit" not in sheet.sheet_progress(str(path))
    assert KELP_ONLY not in json.dumps(sheet.read_sheet(str(path)))


def test_l1_a_sheet_whose_pictures_printed_the_kelp_rule_keeps_it(
    root: Path,
) -> None:
    """L1: the pictures' record of a sheet pictured and answered before the
    round holds the kelp rule: the sheet keeps it and its old scope,
    everywhere, and the record is not written again. The record's last rule
    is the one: kelp then general says general, general then kelp says kelp."""
    path = _sheet(root)
    _answer_all(path)
    record = _pictured(path, OLD_RULE)
    before = record.read_bytes()
    assert _said(path) == [OLD, OLD, OLD]
    assert "rule_limit" not in sheet.read_sheet(str(path))
    assert record.read_bytes() == before
    _pictured(path, OLD_RULE, GENERAL_RULE)
    assert _said(path)[0] == GENERAL
    _pictured(path, GENERAL_RULE, OLD_RULE)
    assert _said(path)[0] == OLD


def test_l1_a_blank_sheet_does_not_take_the_rule_of_the_record_beside_it(
    root: Path,
) -> None:
    """L1 (review F2): a fresh sheet planned again under an earlier sheet's
    name finds that sheet's pictures' record, which printed the kelp rule:
    while it holds no answer the record is not read, so it takes the general
    rule, and reading it writes nothing. Its first pictures append the general
    rule's line, which still stands once an answer is saved."""
    path = _sheet(root)
    record = _pictured(path, OLD_RULE)
    before = record.read_bytes()
    assert imagery.app_rule_of(path.resolve()) == GENERAL_RULE
    assert _said(path) == [GENERAL, GENERAL, GENERAL]
    assert record.read_bytes() == before
    # Its first pictures: the context line of the rule it takes, appended.
    real = path.resolve()
    imagery._write(real, imagery.rule_for(real, ["absent", "present"])[0], [])
    assert record.read_bytes() != before
    assert imagery._recorded_app_rule(real) == GENERAL_RULE
    first = [{"window_index": WINDOWS[0], "reference_class": "0"}]
    sheet.save_labels(str(path), first)
    assert _said(path)[0] == GENERAL


def test_l1_a_blank_sheet_drawn_since_the_round_agrees_with_its_card(
    root: Path,
) -> None:
    """L1 (review F2): a blank sheet a Check card drew since the round (its
    draw pins the general rule), beside an earlier sheet's kelp record, says
    the general rule, as its card does."""
    path = _sheet(root)
    _pictured(path, OLD_RULE)
    _draw_record(
        root,
        "a" * 24,
        "r-since",
        labels_csv_path=os.path.realpath(path),
        app_rule=GENERAL_RULE,
    )
    assert _said(path) == [GENERAL, GENERAL, GENERAL]
    assert server._check_rule(False, "r-since")["text"] == GENERAL_RULE


def test_l1_a_sheet_drawn_before_the_round_keeps_the_kelp_rule(root: Path) -> None:
    """L1: a sheet a Check card drew before the round (its draw record pins no
    app_rule) keeps the kelp rule and its old scope; a sheet drawn since (the
    record pins the general rule) takes the general rule."""
    old = _sheet(root, "old")
    new = _sheet(root, "new")
    _draw_record(root, "a" * 24, "r-old", labels_csv_path=os.path.realpath(old))
    _draw_record(
        root,
        "b" * 24,
        "r-new",
        labels_csv_path=os.path.realpath(new),
        app_rule=GENERAL_RULE,
    )
    assert check.sheet_app_rule(str(old)) == OLD_RULE
    assert check.sheet_app_rule(str(new)) == GENERAL_RULE
    assert check.sheet_app_rule(str(root / "other_to_label.csv")) is None
    assert _said(old) == [OLD, OLD, OLD]
    assert _said(new) == [GENERAL, GENERAL, GENERAL]


def test_l1_a_sheet_graded_before_the_round_keeps_the_kelp_rule(root: Path) -> None:
    """L1: a graded mark written since the round keeps the app's rule the
    sheet took (app_rule); a mark written before has none, and its sheet keeps
    the kelp rule. The mark is read, never written again, to say so."""
    path = _sheet(root)
    _answer_all(path)
    sheet.record_grading(str(path), sheet.sheet_version(str(path)))
    place = path.with_name("." + path.name + sheet.GRADED_SUFFIX)
    mark = json.loads(place.read_text(encoding="utf-8"))
    assert mark["first"]["app_rule"] == GENERAL_RULE
    assert sheet.graded_app_rule(str(path)) == GENERAL_RULE
    assert _said(path)[0] == GENERAL
    # The same mark as a grading before the round wrote it: no app_rule.
    del mark["first"]["app_rule"]
    place.write_text(json.dumps(mark, sort_keys=True), encoding="utf-8")
    os.chmod(place, 0o600)
    before = place.read_bytes()
    assert sheet.graded_app_rule(str(path)) == OLD_RULE
    assert _said(path) == [OLD, OLD, OLD]
    assert place.read_bytes() == before
    # Graded again: the first grading's rule stays the kelp rule.
    sheet.record_grading(str(path), sheet.sheet_version(str(path)))
    again = json.loads(place.read_text(encoding="utf-8"))
    assert "app_rule" not in again["first"] and again["gradings"] == 2
    assert _said(path)[0] == OLD


def test_l1_a_graded_mark_comes_before_a_later_draw(root: Path) -> None:
    """L1 (review F1): a sheet labelled and graded before the round, kept by
    the plan for a Check card press made since (its draw pins the general
    rule), keeps the kelp rule its mark holds; neither record is written
    again. The other way round, a mark that kept the general rule beside a
    draw that reads as kelp gives the general rule."""
    path = _sheet(root)
    _answer_all(path)
    place = _graded_before_the_round(path)
    drawn = _draw_record(
        root,
        "a" * 24,
        "r-kept",
        labels_csv_path=os.path.realpath(path),
        app_rule=GENERAL_RULE,
    )
    mark, draw = place.read_bytes(), drawn.read_bytes()
    assert imagery.app_rule_of(path.resolve()) == OLD_RULE
    assert sheet.labelling_rule(str(path)) == {
        "text": OLD_RULE,
        "scope": OLD_SCOPE,
        "source": "owner",
    }
    assert _said(path) == [OLD, OLD, OLD]
    assert (place.read_bytes(), drawn.read_bytes()) == (mark, draw)
    # The reverse: a mark written since the round, a draw from before it.
    other = _sheet(root, "other")
    _answer_all(other)
    sheet.record_grading(str(other), sheet.sheet_version(str(other)))
    _draw_record(root, "b" * 24, "r-other", labels_csv_path=os.path.realpath(other))
    assert check.sheet_app_rule(str(other)) == OLD_RULE
    assert imagery.app_rule_of(other.resolve()) == GENERAL_RULE
    assert _said(other) == [GENERAL, GENERAL, GENERAL]


def test_l1_a_mark_keeps_no_apps_rule_for_a_sheet_that_takes_none(
    root: Path,
) -> None:
    """L1 (review F4): the graded mark of a sheet that takes no app's rule (a
    map with the gate, or a rule typed beside the sheet) keeps app_rule None,
    which reads back as no rule; a mark with no app_rule at all is one from
    before the round, read as the kelp rule."""
    gate = _sheet(root, "gate", names=("water", "land"))
    _answer_all(gate)
    sheet.record_grading(str(gate), sheet.sheet_version(str(gate)))
    typed = _sheet(root, "typed")
    rule = "Seagrass: seagrass covers more than half of the outlined square."
    sheet.set_typed_rule(str(typed), rule, typed_at="2026-11-19T10:00:00+00:00")
    _answer_all(typed)
    sheet.record_grading(str(typed), sheet.sheet_version(str(typed)))
    for path in (gate, typed):
        place = path.with_name("." + path.name + sheet.GRADED_SUFFIX)
        mark = json.loads(place.read_text(encoding="utf-8"))
        assert "app_rule" in mark["first"] and mark["first"]["app_rule"] is None
        assert sheet.graded_app_rule(str(path)) is None
    assert sheet.labelling_rule(str(gate)) is None
    assert sheet.labelling_rule(str(typed)) == {
        "text": rule,
        "scope": sheet.typed_rule_scope("2026-11-19T10:00:00+00:00"),
        "source": "typed",
    }
    present = _sheet(root)
    _answer_all(present)
    _graded_before_the_round(present)
    assert sheet.graded_app_rule(str(present)) == OLD_RULE


def test_l1_a_typed_rule_still_comes_first(root: Path) -> None:
    """L1 keeps option A: a rule typed before the draw replaces the app's
    rule, whichever the records hold."""
    path = _sheet(root)
    _draw_record(root, "c" * 24, "r-old", labels_csv_path=os.path.realpath(path))
    typed = "Seagrass: seagrass covers more than half of the outlined square."
    sheet.set_typed_rule(str(path), typed, typed_at="2026-11-19T10:00:00+00:00")
    text, scope, source = imagery.rule_for(path.resolve(), ["absent", "present"])
    assert (text, source) == (typed, "typed")
    assert scope == sheet.typed_rule_scope("2026-11-19T10:00:00+00:00")


# --------------------------------------------------------------------------- a draw


def test_l1_a_map_pressed_before_the_round_keeps_the_kelp_rule_on_its_card(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """L1: the Check card of a map a part of which was pressed before the
    round shows the kelp rule and its old scope (settled); a new part of it is
    drawn with the same rule; a map never pressed shows the general rule and
    its scope, with the box."""
    monkeypatch.setattr(sheet, "result_holds_answers", lambda result_id: False)
    _draw_record(root, "d" * 24, "r-old")
    assert check.map_app_rule("r-old") == OLD_RULE
    assert check._app_rule_for("r-old") == OLD_RULE
    assert server._check_rule(False, "r-old") == {
        "set": True,
        "text": OLD_RULE,
        "source": "owner",
        "scope": OLD_SCOPE,
        "replaceable": False,
    }
    assert check.map_app_rule("r-new") is None
    assert check._app_rule_for("r-new") == GENERAL_RULE
    assert server._check_rule(False, "r-new") == {
        "set": True,
        "text": GENERAL_RULE,
        "source": "owner",
        "scope": GENERAL_SCOPE,
        "replaceable": True,
    }


@pytest.mark.parametrize(
    ("old_key", "new_key"), [("e" * 24, "f" * 24), ("f" * 24, "e" * 24)]
)
def test_l1_the_earliest_press_of_a_map_chooses_its_rule(
    root: Path, old_key: str, new_key: str
) -> None:
    """L1: of a map's draw records, the earliest press says which rule the map
    takes: a part pressed since the round with the general rule, after one
    pressed before, does not change the map's. Both orders of the records'
    file names (review C4), so neither the first nor the last read wins."""
    _draw_record(root, old_key, "r-mixed")
    late = _draw_record(root, new_key, "r-mixed", app_rule=GENERAL_RULE)
    record = json.loads(late.read_text(encoding="utf-8"))
    record["pressed_at"] = "2026-11-20T09:00:00Z"
    late.write_text(json.dumps(record), encoding="utf-8")
    assert check.map_app_rule("r-mixed") == OLD_RULE


@pytest.mark.parametrize("state", ["failed", "failed-final", "drawing"])
def test_l1_a_press_that_drew_nothing_keeps_no_kelp_rule(
    root: Path, monkeypatch: pytest.MonkeyPatch, state: str
) -> None:
    """L1 (review F3, C1): a map whose only press before the round drew
    nothing (failed, or left drawing too long) shows the general rule with
    the box, and the note beside it is true; a retry of that press pins the
    general rule, as a first press would."""
    monkeypatch.setattr(sheet, "result_holds_answers", lambda result_id: False)
    key = _offer(root, "r-fail")
    if state == "failed-final":
        _undrawn(root, key, "r-fail", "failed", retry=False)
    else:
        _undrawn(root, key, "r-fail", state)
    assert check.map_app_rule("r-fail") is None
    assert check._app_rule_for("r-fail") == GENERAL_RULE
    assert server._check_rule(False, "r-fail") == {
        "set": True,
        "text": GENERAL_RULE,
        "source": "owner",
        "scope": GENERAL_SCOPE,
        "replaceable": True,
    }
    if state != "failed-final":
        claim = check.claim_draw(key, 20)
        assert claim["outcome"] == "retry"
        assert claim["draw"]["app_rule"] == GENERAL_RULE


@pytest.mark.parametrize("state", ["failed", "drawing"])
def test_l1_a_part_drawn_before_the_round_keeps_its_map_on_the_kelp_rule(
    root: Path, monkeypatch: pytest.MonkeyPatch, state: str
) -> None:
    """L1 (review F3, C1): beside a press that drew nothing, a part of the same
    map drawn before the round keeps the map on the kelp rule, on the card and
    in the retried press's record: one map keeps one rule."""
    monkeypatch.setattr(sheet, "result_holds_answers", lambda result_id: False)
    _draw_record(root, "d" * 24, "r-part")
    key = _offer(root, "r-part")
    _undrawn(root, key, "r-part", state)
    assert check.map_app_rule("r-part") == OLD_RULE
    assert server._check_rule(False, "r-part") == {
        "set": True,
        "text": OLD_RULE,
        "source": "owner",
        "scope": OLD_SCOPE,
        "replaceable": False,
    }
    claim = check.claim_draw(key, 20)
    assert claim["outcome"] == "retry" and claim["draw"]["app_rule"] == OLD_RULE


def test_l1_a_first_press_pins_the_apps_rule(root: Path) -> None:
    """L1: claim_draw's first press writes the app's rule the part is drawn
    with into the draw record (the map's own, else the general rule), read
    before the draw's lock, beside a typed rule where one is pinned; an offer
    with no rule_owner (written before the box) still pins it."""
    source = _source(check)
    assert '"app_rule": None if offer.get("rule_needed") else app_rule,' in source
    assert "return result_app_rule(result_id) or imagery.READING_RULE" in source
    assert source.index("app_rule = _app_rule_for(") < source.index(
        "    with _draw_lock(folder, key):\n        offer = read_offer(key)\n"
        "        offered = _offered(offer)"
    )
    key = _offer(root, "r-first")
    typed = "Seagrass: seagrass covers more than half of the outlined square."
    claim = check.claim_draw(key, 20, reading_rule=typed)
    assert claim["outcome"] == "first"
    assert claim["draw"]["app_rule"] == GENERAL_RULE
    assert claim["draw"]["reading_rule"] == typed
    older = _offer(root, "r-older")
    place = root / "checks" / f"{older}.json"
    record = json.loads(place.read_text(encoding="utf-8"))
    del record["rule_owner"]
    place.write_text(json.dumps(record), encoding="utf-8")
    assert check.claim_draw(older, 20)["draw"]["app_rule"] == GENERAL_RULE


def test_l1_a_map_with_the_gate_pins_no_apps_rule(root: Path) -> None:
    """L1 (review F4): a press on a map with no rule of the owner's (the gate)
    pins app_rule None, the key there, even when a part of the map was drawn
    before the round; such a record reads as no app's rule."""
    _draw_record(root, "d" * 24, "r-gate")
    key = _offer(root, "r-gate", rule_needed=True, rule_owner=False)
    typed = "Water: open water covers more than half of the outlined square."
    claim = check.claim_draw(key, 20, reading_rule=typed)
    assert claim["outcome"] == "first"
    assert "app_rule" in claim["draw"] and claim["draw"]["app_rule"] is None
    assert check._draw_app_rule(claim["draw"]) is None
    assert check._draw_app_rule({"state": "drawn"}) == OLD_RULE
    # A map with the gate drawn since: its drawn record pins no app's rule.
    _draw_record(root, "c" * 24, "r-gate-2", app_rule=None, labels_csv_path="x")
    assert check.map_app_rule("r-gate-2") is None
    retried = _offer(root, "r-gate-3", rule_needed=True, rule_owner=False)
    _undrawn(root, retried, "r-gate-3", "failed")
    again = check.claim_draw(retried, 20, reading_rule=typed)
    assert again["outcome"] == "retry" and again["draw"]["app_rule"] is None


# --------------------------------------------------------------------------- a map's sheets


def _chat_sheet(root: Path, result_id: str, stem: str = "a1b2c3d4") -> Path:
    """A sheet of ``result_id`` planned in the chat, named as the plan tool
    names it (``design_`` and the result's slug), every square answered."""
    path = _sheet(root, f"design_{result_id}_{stem}_random_4_s0", result_id=result_id)
    _answer_all(path)
    return path


def test_l1_a_map_whose_chat_sheet_was_graded_before_the_round_keeps_kelp(
    root: Path,
) -> None:
    """L1 (review CORE-2): a map with no Check card draw whose sheet planned in
    the chat was labelled and graded before the round shows the kelp rule,
    settled, on its card, and its first part pressed pins the kelp rule."""
    _graded_before_the_round(_chat_sheet(root, "r-chat"))
    assert sheet.answered_sheets("r-chat")
    assert check.result_app_rule("r-chat") == OLD_RULE
    assert server._check_rule(False, "r-chat") == {
        "set": True,
        "text": OLD_RULE,
        "source": "owner",
        "scope": OLD_SCOPE,
        "replaceable": False,
    }
    key = _offer(root, "r-chat")
    claim = check.claim_draw(key, 20)
    assert claim["outcome"] == "first" and claim["draw"]["app_rule"] == OLD_RULE


def test_l1_a_map_whose_chat_sheet_was_pictured_before_the_round_keeps_kelp(
    root: Path,
) -> None:
    """L1 (review CORE-2): the same for a chat sheet answered beside pictures
    that printed the kelp rule before the round."""
    _pictured(_chat_sheet(root, "r-pics"), OLD_RULE)
    assert server._check_rule(False, "r-pics")["text"] == OLD_RULE
    assert server._check_rule(False, "r-pics")["replaceable"] is False


def test_l1_a_chat_sheet_no_record_speaks_for_leaves_the_general_rule(
    root: Path,
) -> None:
    """L1 (review CORE-2): a chat sheet labelled but never pictured or graded
    holds no record of the kelp rule: the map shows the general rule (settled,
    as the sheet holds answers); and a part drawn since with the general rule
    is not overridden by a kelp chat sheet of the same map."""
    _chat_sheet(root, "r-plain")
    assert check.result_app_rule("r-plain") is None
    assert server._check_rule(False, "r-plain") == {
        "set": True,
        "text": GENERAL_RULE,
        "source": "owner",
        "scope": GENERAL_SCOPE,
        "replaceable": False,
    }
    _graded_before_the_round(_chat_sheet(root, "r-both"))
    _draw_record(root, "a" * 24, "r-both", app_rule=GENERAL_RULE)
    assert check.result_app_rule("r-both") == GENERAL_RULE
    assert server._check_rule(False, "r-both")["text"] == GENERAL_RULE


# --------------------------------------------------------------------------- the docs


def test_l1_the_changelog_and_the_usage_say_the_rule_is_general() -> None:
    """L1: the 1.5.0 detail file's marker (spec/changes/1.5.0.md) is replaced
    by the round's line, and
    spec/Usage.md gives the general rule and its scope."""
    stage = Path(__file__).resolve().parents[2]
    changelog = (stage / "spec" / "changes" / "1.5.0.md").read_text(encoding="utf-8")
    assert "@@R5L-CORE@@" not in changelog
    assert "\n    - L1: " in changelog
    usage = (stage / "spec" / "Usage.md").read_text(encoding="utf-8")
    flat = " ".join(usage.split())
    assert GENERAL_SCOPE in flat
    assert "`imagery.RULE_KELP_ONLY`) follows the scope" not in flat
