# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""The sheet's three extras of 13 Nov 2026 (round 4, pass 2).

The labelling rule the user types before the draw (kept beside the sheet,
refused once an answer is saved, printed by the page copy; text pasted from
a document keeps its words; by the owner's option A a typed rule replaces
the owner's kelp rule, and an empty box keeps it), the graded mark (the
version first and last graded, ``changed`` sticky, bound to the design's
content and no mark while the sheet holds no answer, so a fresh sheet under
the same name is not graded while a design file written again keeps it), and
the graded sheet's squares (counts of the classes seen and of where the map
agreed, Cannot tell squares last since r4i's O2, none on a version the
estimate did not read). Then the page copy's unit and whether pictures can
be shown (contract C1, with a Studio result's sampled points file, finding
R3), the plan card's first button (contract C2), and no "window" on the
faces (the owner's ruling 6).
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import pytest

from olmoearth_agent.analysis.review_set import regression_scores
from olmoearth_agent.mcp_app import sheet
from olmoearth_agent.tools.review_set import SampledScores

ONE_MAP = [
    "order",
    "window_index",
    "row",
    "col",
    "map_class",
    "lon",
    "lat",
    "stratum",
    "wrong",
    "reference_class",
]
TWO_MAP = [
    "order",
    "window_index",
    "row",
    "col",
    "class_a",
    "class_b",
    "reference_class",
]

#: The map's class of each of 20 windows on the seen fixture's design: present
#: (1) at windows 4 and 12, absent (0) elsewhere.
SEEN_MAP = [1 if w in (4, 12) else 0 for w in range(20)]

#: A typed rule, and when it was typed.
TYPED = "Water: open water fills more than half of the outlined square."
TYPED_AT = "2026-11-13T10:15:00+00:00"


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    out = tmp_path / "outputs"
    out.mkdir()
    monkeypatch.setenv("OLMOEARTH_OUTPUT_ROOT", str(out))
    monkeypatch.delenv("OLMOEARTH_SCORES_ROOT", raising=False)
    return out


def _write(path: Path, header: list[str], rows: list[list[str]]) -> Path:
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)
    return path


def _rows(windows: tuple[int, ...], on_map: list[Any]) -> list[list[str]]:
    return [
        [str(order), str(w), "0", str(w), str(on_map[w]), "10.5", "50.2", "0", "", ""]
        for order, w in enumerate(windows, start=1)
    ]


def _seen(root: Path, windows: tuple[int, ...] = (4, 9, 12, 3)) -> Path:
    """A named scores map's plan (the kelp trial's): Present and Absent."""
    design = {
        "design": "random",
        "population": {
            "score_kind": "window_confidence",
            "class_names": {"0": "absent", "1": "present"},
            "map_class": SEEN_MAP,
        },
    }
    (root / "kelp.json").write_text(json.dumps(design), encoding="utf-8")
    return _write(root / "kelp_to_label.csv", ONE_MAP, _rows(windows, SEEN_MAP))


def _to_label(windows: tuple[int, ...]) -> list[dict[str, Any]]:
    """The windows drawn, as the plan tool lists them in its design."""
    return [
        {
            "order": order,
            "window_index": w,
            "row": 0,
            "col": w,
            "map_class": SEEN_MAP[w],
        }
        for order, w in enumerate(windows)
    ]


def _planned(root: Path, windows: tuple[int, ...] = (4, 9, 12, 3)) -> Path:
    """:func:`_seen`'s plan as the plan tool writes it: its design lists the
    windows drawn (``to_label``), whose digest the graded mark is bound to."""
    path = _seen(root, windows)
    design = json.loads((root / "kelp.json").read_text(encoding="utf-8"))
    design["to_label"] = _to_label(windows)
    (root / "kelp.json").write_text(json.dumps(design), encoding="utf-8")
    return path


def _digest(to_label: list[dict[str, Any]]) -> str:
    text = json.dumps(to_label, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


#: A class map's classes as it writes them, and its class of 10 windows.
CLASSES = ["water", "trees", "crops"]
CLASS_MAP = [0, 0, 0, 1, 0, 2, 0, 0, 0, 0]


def _class_map(root: Path) -> Path:
    """A class map's plan, labelled with the class seen: windows 7 (water), 3
    (trees) and 5 (crops)."""
    design = {
        "design": "random",
        "population": {
            "score_kind": "class_map",
            "classes": CLASSES,
            "map_class": CLASS_MAP,
        },
    }
    (root / "land.json").write_text(json.dumps(design), encoding="utf-8")
    on_map = [CLASSES[c] for c in CLASS_MAP]
    return _write(root / "land_to_label.csv", ONE_MAP, _rows((7, 3, 5), on_map))


def _scores_map(root: Path) -> Path:
    """A scores map's plan with no class names: labelled right or wrong."""
    design = {"design": "random", "population": {"score_kind": "window_confidence"}}
    (root / "conf.json").write_text(json.dumps(design), encoding="utf-8")
    return _write(root / "conf_to_label.csv", ONE_MAP, _rows((1, 2), [0] * 20))


def _label_seen(path: Path, answers: dict[int, str]) -> None:
    sheet.save_labels(
        str(path),
        [{"window_index": w, "reference_class": code} for w, code in answers.items()],
    )


def _set_cell(path: Path, window: str, column: str, value: str) -> None:
    """Edit one cell of the sheet, as a spreadsheet would."""
    with open(path, encoding="utf-8", newline="") as fh:
        rows = list(csv.reader(fh))
    at, at_window = rows[0].index(column), rows[0].index("window_index")
    for row in rows[1:]:
        if row[at_window] == window:
            row[at] = value
    _write(path, rows[0], rows[1:])


def _class_row(
    name: str, seen: int, agreed: int, *, outside: bool = False
) -> dict[str, Any]:
    """A class map's class as graded_squares lists it: its code is its name."""
    return {
        "code": name,
        "name": name,
        "seen": seen,
        "agreed": agreed,
        "outside": outside,
    }


# ------------------------------------------------------------- the typed rule


def test_a_typed_rule_is_one_printable_line_of_10_to_300_characters() -> None:
    assert sheet.clean_rule("  " + "a" * 10 + "  ") == "a" * 10
    assert sheet.clean_rule("b" * 300) == "b" * 300
    bad_rules = ("a" * 9, "c" * 301, "one line\nand another", "a\ttabbed rule", 42)
    for bad in (*bad_rules, None):
        with pytest.raises(ValueError, match=re.escape(sheet.RULE_TEXT)):
            sheet.clean_rule(bad)


def test_a_rule_pasted_from_a_document_keeps_its_words(root: Path) -> None:
    """Round 4 pass 2, findings 6 and 22: a Unicode space separator (no-break,
    narrow no-break, ideographic) becomes a plain space and a format character
    (zero-width space, soft hyphen, byte-order mark) is dropped, before the
    checks; a line break, a line separator and a tab are still refused, and
    the length is counted in code points."""
    # The characters are named, not typed, so this file stays plain ASCII.
    nbsp, narrow, ideographic = chr(0x00A0), chr(0x202F), chr(0x3000)
    zwsp, shy, bom = chr(0x200B), chr(0x00AD), chr(0xFEFF)
    line_separator, wave = chr(0x2028), chr(0x1F30A)
    plain = "Present: panels cover more than half of the square"
    pasted = f"Present:{nbsp}panels cover more than half of the square"
    assert sheet.clean_rule(pasted) == plain
    kelp = f"Kelp{narrow}canopy visible at the surface counts"
    assert sheet.clean_rule(kelp) == "Kelp canopy visible at the surface counts"
    assert sheet.clean_rule(ideographic + "a" * 10 + nbsp) == "a" * 10
    hidden = f"{bom}Kelp{zwsp} canopy{shy} counts here"
    assert sheet.clean_rule(hidden) == "Kelp canopy counts here"
    assert sheet.clean_rule("a" + nbsp * 298 + "b") == "a" + " " * 298 + "b"
    assert sheet.clean_rule(wave * 10) == wave * 10
    for bad in (
        "a" * 9 + zwsp,
        zwsp * 12,
        nbsp * 12,
        f"one line{line_separator}and another",
        "a rule\nwith a break",
        "a\ttabbed rule",
        wave * 5,
    ):
        with pytest.raises(ValueError, match=re.escape(sheet.RULE_TEXT)):
            sheet.clean_rule(bad)
    # Kept as cleaned, and read back the same.
    path = _class_map(root)
    kept = sheet.set_typed_rule(str(path), pasted, typed_at=TYPED_AT)
    assert kept["text"] == plain
    assert sheet.typed_rule(str(path)) == kept


def test_a_typed_rule_is_kept_beside_the_sheet_until_an_answer_is_saved(
    root: Path,
) -> None:
    """Kept whole in .<sheet>.rule, mode 0600; written again before any
    answer; refused once the sheet holds one, with the file as it was."""
    path = _class_map(root)
    assert sheet.typed_rule(str(path)) is None
    kept = sheet.set_typed_rule(str(path), f"  {TYPED} ", typed_at=TYPED_AT)
    assert kept == {"text": TYPED, "typed_at": TYPED_AT}
    place = path.with_name("." + path.name + sheet.RULE_SUFFIX)
    assert place.is_file() and (place.stat().st_mode & 0o777) == 0o600
    assert sheet.typed_rule(str(path)) == kept
    again = TYPED.replace("half", "a half")
    assert sheet.set_typed_rule(str(path), again, typed_at=TYPED_AT)["text"] == again
    sheet.save_labels(str(path), [{"window_index": 7, "reference_class": "water"}])
    before = place.read_bytes()
    with pytest.raises(ValueError, match=re.escape(sheet.RULE_AFTER_ANSWER)):
        sheet.set_typed_rule(str(path), TYPED, typed_at=TYPED_AT)
    assert place.read_bytes() == before
    for bad_moment in (None, "", "x" * 41, "13 Nov\n2026"):
        with pytest.raises(ValueError):
            sheet.set_typed_rule(str(_scores_map(root)), TYPED, typed_at=bad_moment)
    assert not list(root.glob(".conf_to_label.csv.rule*"))


def test_a_link_at_the_rules_place_is_never_followed(
    root: Path, tmp_path: Path
) -> None:
    path = _class_map(root)
    target = tmp_path / "elsewhere.txt"
    planted = json.dumps(
        {"format": sheet.RULE_FORMAT, "text": TYPED, "typed_at": TYPED_AT}
    )
    target.write_text(planted, encoding="utf-8")
    place = path.with_name("." + path.name + sheet.RULE_SUFFIX)
    place.symlink_to(target)
    assert sheet.typed_rule(str(path)) is None
    with pytest.raises(ValueError, match="is a link"):
        sheet.set_typed_rule(str(path), TYPED, typed_at=TYPED_AT)
    assert target.read_text(encoding="utf-8") == planted


def _rule_said(path: Path) -> list[tuple[Any, ...]]:
    """The rule as the page copy and the counts carry it."""
    read, progress = sheet.read_sheet(str(path)), sheet.sheet_progress(str(path))
    keys = ("reading_rule", "rule_scope", "rule_source")
    return [tuple(read[k] for k in keys), tuple(progress[k] for k in keys)]


def test_a_typed_rule_replaces_the_owners_and_an_empty_box_keeps_it(
    root: Path,
) -> None:
    """The owner's option A (13 Nov 2026): on a map of presence and absence,
    nothing typed (the box left empty) keeps the owner's kelp rule with its
    scope; a rule typed and pinned beside the sheet at the press replaces it,
    with the typed scope and never RULE_SCOPE, in labelling_rule, the page
    copy, the counts and the pictures' rule (imagery.rule_for)."""
    from olmoearth_agent.mcp_app import imagery

    kelp = _seen(root)
    names = ["absent", "present"]
    owner = (imagery.READING_RULE, imagery.RULE_SCOPE, "owner")
    assert sheet.labelling_rule(str(kelp)) == {
        "text": imagery.READING_RULE,
        "scope": imagery.RULE_SCOPE,
        "source": "owner",
    }
    assert _rule_said(kelp) == [owner, owner]
    assert imagery.rule_for(kelp, names) == owner
    sheet.set_typed_rule(str(kelp), TYPED, typed_at=TYPED_AT)
    scope = "Typed by you before the squares were picked, 13 Nov 2026."
    typed = (TYPED, scope, "typed")
    assert sheet.labelling_rule(str(kelp)) == {
        "text": TYPED,
        "scope": scope,
        "source": "typed",
    }
    assert _rule_said(kelp) == [typed, typed]
    assert imagery.rule_for(kelp, names) == typed
    copy = json.dumps(sheet.read_sheet(str(kelp)))
    assert imagery.READING_RULE not in copy and imagery.RULE_SCOPE not in copy


def test_a_map_of_other_classes_has_a_rule_only_once_one_is_typed(
    root: Path,
) -> None:
    """Not a map of presence and absence: no rule until the gate's is typed,
    then the typed one; the page copy and the counts carry it."""
    land = _class_map(root)
    assert sheet.labelling_rule(str(land)) is None
    read = sheet.read_sheet(str(land))
    assert (read["reading_rule"], read["rule_scope"], read["rule_source"]) == (
        None,
        None,
        None,
    )
    sheet.set_typed_rule(str(land), TYPED, typed_at=TYPED_AT)
    scope = "Typed by you before the squares were picked, 13 Nov 2026."
    assert sheet.typed_rule_scope(TYPED_AT) == scope
    assert sheet.labelling_rule(str(land)) == {
        "text": TYPED,
        "scope": scope,
        "source": "typed",
    }
    read = sheet.read_sheet(str(land))
    progress = sheet.sheet_progress(str(land))
    for key, want in (
        ("reading_rule", TYPED),
        ("rule_scope", scope),
        ("rule_source", "typed"),
    ):
        assert read[key] == progress[key] == want, key
    # A typed_at that is not ISO 8601 is said as given.
    assert sheet.typed_rule_scope("today").endswith("picked, today.")


# --------------------------------------------------------------- the graded mark


def test_the_graded_mark_keeps_the_first_version_and_changed_stays_true(
    root: Path,
) -> None:
    path = _seen(root)
    _label_seen(path, {4: "1", 9: "1", 12: "0", 3: "0"})
    assert sheet.graded_state(str(path)) is None
    assert sheet.read_sheet(str(path))["graded"] is None
    first = sheet.sheet_version(str(path))
    assert re.fullmatch(r"v[a-z2-7]{16}", first)
    assert first == sheet.read_sheet(str(path))["version"]
    out = sheet.record_grading(str(path), first, design_path=str(root / "kelp.json"))
    assert (out["changed"], out["gradings"]) == (False, 1)
    place = path.with_name("." + path.name + sheet.GRADED_SUFFIX)
    assert (place.stat().st_mode & 0o777) == 0o600
    mark = json.loads(place.read_text(encoding="utf-8"))
    assert mark["format"] == sheet.GRADED_FORMAT
    assert mark["first"]["version"] == mark["last"]["version"] == first
    assert mark["first"]["design_path"] == str(root / "kelp.json")
    state = sheet.graded_state(str(path))
    assert state == {
        "first_at": out["first_at"],
        "changed_since_first": False,
        "changed": False,
        "answered": True,
    }
    # The page copy and the counts say it was graded, and nothing changed since.
    graded = {"first_at": out["first_at"], "changed_since_first": False}
    assert sheet.read_sheet(str(path))["graded"] == graded
    assert sheet.sheet_progress(str(path))["graded"] == graded
    # Graded again on the same answers: counted, not changed.
    again = sheet.record_grading(str(path), first)
    assert (again["changed"], again["gradings"], again["first_at"]) == (
        False,
        2,
        out["first_at"],
    )
    # An answer changed after the grading: the copy says so before any new grading.
    _label_seen(path, {3: "1"})
    assert sheet.read_sheet(str(path))["graded"]["changed_since_first"] is True
    second = sheet.sheet_version(str(path))
    assert second != first
    changed = sheet.record_grading(str(path), second)
    assert (changed["changed"], changed["gradings"]) == (True, 3)
    # Changed back and graded on the first version: changed stays true.
    _label_seen(path, {3: "0"})
    assert sheet.sheet_version(str(path)) == first
    sticky = sheet.record_grading(str(path), first)
    assert (sticky["changed"], sticky["gradings"]) == (True, 4)
    assert sheet.graded_state(str(path)) == {
        "first_at": out["first_at"],
        "changed_since_first": False,
        "changed": True,
        "answered": True,
    }


def test_a_grading_needs_the_sheets_own_version(root: Path, tmp_path: Path) -> None:
    path = _seen(root)
    for bad in ("", "v123", "V" + "a" * 16, "s" + "a" * 16):
        with pytest.raises(ValueError, match=re.escape(sheet.BAD_VERSION)):
            sheet.record_grading(str(path), bad)
    place = path.with_name("." + path.name + sheet.GRADED_SUFFIX)
    assert not place.exists()
    # A mark that does not read as one is started again; a link is refused.
    place.write_text("not a mark", encoding="utf-8")
    version = sheet.sheet_version(str(path))
    assert sheet.record_grading(str(path), version)["gradings"] == 1
    place.unlink()
    target = tmp_path / "mark.json"
    target.write_text("{}", encoding="utf-8")
    place.symlink_to(target)
    with pytest.raises(ValueError, match="is a link"):
        sheet.record_grading(str(path), version)
    assert target.read_text(encoding="utf-8") == "{}"
    assert sheet.read_sheet(str(path))["graded"] is None


def _graded_place(path: Path) -> Path:
    return path.with_name("." + path.name + sheet.GRADED_SUFFIX)


def test_a_fresh_sheet_planned_again_under_the_same_name_is_not_graded(
    root: Path,
) -> None:
    """Round 4, R6 with finding 23: the mark keeps a digest of the design's
    to_label, not the design file's stamp. The labelled sheet moved aside and
    the plan written again with the same arguments (the same windows, so the
    same digest, the same file names, the design file written anew), the old
    mark beside the blank sheet is no mark (no notice on the page copy or the
    counts), the first save into the blank sheet removes it, and the first
    grading of the new sheet starts its own, unchanged."""
    path = _planned(root)
    _label_seen(path, {4: "1", 9: "1", 12: "0", 3: "0"})
    sheet.record_grading(str(path), sheet.sheet_version(str(path)))
    design = root / "kelp.json"
    mark = json.loads(_graded_place(path).read_text(encoding="utf-8"))
    assert mark["first"]["design_digest"] == _digest(_to_label((4, 9, 12, 3)))
    assert "design_stamp" not in mark["first"]
    # Answers changed on the same sheet: still its mark, which says so.
    _label_seen(path, {3: "1"})
    assert sheet.graded_state(str(path))["changed_since_first"] is True
    # Moved aside; the plan writes a blank sheet and its design file again.
    path.rename(root / "kelp_first_reviewer.csv")
    blank = _planned(root)
    assert blank == path and _graded_place(blank).is_file()
    st = design.stat()
    os.utime(design, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
    assert sheet.graded_state(str(blank)) is None
    assert sheet.read_sheet(str(blank))["graded"] is None
    assert sheet.sheet_progress(str(blank))["graded"] is None
    # A save that writes nothing leaves the file; the first that writes removes it.
    sheet.save_labels(str(blank), [{"window_index": 4, "reference_class": ""}])
    assert _graded_place(blank).is_file()
    _label_seen(blank, {4: "1"})
    assert not _graded_place(blank).exists()
    _label_seen(blank, {9: "0", 12: "1", 3: "0"})
    out = sheet.record_grading(str(blank), sheet.sheet_version(str(blank)))
    assert (out["changed"], out["gradings"]) == (False, 1)
    assert sheet.graded_state(str(blank)) == {
        "first_at": out["first_at"],
        "changed_since_first": False,
        "changed": False,
        "answered": True,
    }


def test_a_design_file_written_again_with_its_content_keeps_the_mark(
    root: Path,
) -> None:
    """Round 4, R6: graded, the design file lost, the plan run again (which
    keeps the labelled sheet and writes its design file again, the same
    content in other bytes, with another stamp), an answer changed and graded
    again: the mark is still the sheet's, and says the answers changed. A
    design file that lists other windows beside the sheet is another
    design's: no mark."""
    path = _planned(root)
    _label_seen(path, {4: "1", 9: "1", 12: "0", 3: "0"})
    out = sheet.record_grading(str(path), sheet.sheet_version(str(path)))
    unchanged = {"first_at": out["first_at"], "changed_since_first": False}
    design = root / "kelp.json"
    held = json.loads(design.read_text(encoding="utf-8"))
    design.unlink()
    # Lost: the sheet holds answers, so it is no fresh sheet; still its mark.
    assert sheet.graded_state(str(path)) == {
        **unchanged,
        "changed": False,
        "answered": True,
    }
    design.write_text(json.dumps(held, indent=2), encoding="utf-8")
    assert sheet.read_sheet(str(path))["graded"] == unchanged
    assert sheet.sheet_progress(str(path))["graded"] == unchanged
    _label_seen(path, {3: "1"})
    assert sheet.graded_state(str(path)) == {
        "first_at": out["first_at"],
        "changed_since_first": True,
        "changed": False,
        "answered": True,
    }
    again = sheet.record_grading(str(path), sheet.sheet_version(str(path)))
    assert (again["changed"], again["gradings"], again["first_at"]) == (
        True,
        2,
        out["first_at"],
    )
    # Another design's windows beside this sheet: not its mark.
    design.write_text(
        json.dumps({**held, "to_label": held["to_label"][::-1]}), encoding="utf-8"
    )
    assert sheet.graded_state(str(path)) is None
    assert sheet.read_sheet(str(path))["graded"] is None


def test_a_mark_beside_a_sheet_with_no_answer_is_no_mark(root: Path) -> None:
    """A fresh blank sheet beside an old mark (one the app did not blank, as
    a plan written again leaves it): no mark while it holds no answer; the
    panel's first save into it removes the mark, so its next grading is a
    first. Every answer cleared from a graded sheet by the app's own saves is
    no fresh sheet (the lead's ruling Q1 of 13 Nov 2026): the mark is kept,
    marked changed and blanked at the version written (test_mcp_r4e_last_sheet)."""
    path = _planned(root)
    _label_seen(path, {4: "1", 9: "1", 12: "0", 3: "0"})
    sheet.record_grading(str(path), sheet.sheet_version(str(path)))
    # The answers cleared by hand, as a spreadsheet would: not the app's blank.
    for window in ("4", "9", "12", "3"):
        for column in ("reference_class", "wrong"):
            _set_cell(path, window, column, "")
    assert _graded_place(path).is_file()
    assert "blanked" not in json.loads(_graded_place(path).read_text("utf-8"))
    assert sheet.graded_state(str(path)) is None
    read = sheet.read_sheet(str(path))
    assert read["graded"] is None and read["n_labelled"] == 0
    handle = read["windows"][0]["handle"]
    sheet.save_answers(
        str(path),
        sheet_id=read["sheet_id"],
        face=read["face"],
        rule=read["label_rule"],
        answers=[{"handle": handle, "answer": "1"}],
        version=read["version"],
    )
    assert not _graded_place(path).exists()


def test_a_mark_without_the_design_digest_is_no_mark(root: Path) -> None:
    """A mark with no digest (one written with the design file's stamp, before
    the digest was kept) or one that is not text reads as none and is started
    again; a sheet with no design file keeps a mark whose digest is None, and
    a design file written beside it later, the sheet's answers kept, leaves it
    the sheet's."""
    path = _planned(root)
    _label_seen(path, {4: "1", 9: "1", 12: "0", 3: "0"})
    version = sheet.sheet_version(str(path))
    sheet.record_grading(str(path), version)
    place = _graded_place(path)
    mark = json.loads(place.read_text(encoding="utf-8"))
    stamped = {k: v for k, v in mark["first"].items() if k != "design_digest"}
    stamped["design_stamp"] = {"mtime_ns": 1, "size": 2}
    for first in (stamped, {**mark["first"], "design_digest": 7}):
        place.write_text(json.dumps({**mark, "first": first}), encoding="utf-8")
        assert sheet.graded_state(str(path)) is None
        assert sheet.read_sheet(str(path))["graded"] is None
    assert sheet.record_grading(str(path), version)["gradings"] == 1
    # No design file beside the sheet.
    bare = _write(root / "bare_to_label.csv", ONE_MAP, _rows((1, 2), [0] * 20))
    sheet.save_labels(
        str(bare),
        [{"window_index": 1, "wrong": "0"}, {"window_index": 2, "wrong": "1"}],
    )
    sheet.record_grading(str(bare), sheet.sheet_version(str(bare)))
    kept = json.loads(_graded_place(bare).read_text(encoding="utf-8"))
    assert kept["first"]["design_digest"] is None
    assert sheet.graded_state(str(bare)) is not None
    (root / "bare.json").write_text(
        json.dumps(
            {
                "design": "random",
                "population": {"score_kind": "x"},
                "to_label": [{"order": 0, "window_index": 1}],
            }
        ),
        encoding="utf-8",
    )
    assert sheet.graded_state(str(bare)) is not None


# ------------------------------------------------------------ the squares graded


def test_a_seen_sheets_squares_are_counted_by_class_right_before_wrong(
    root: Path,
) -> None:
    """Present first, Absent second (the answers' order); each class's
    squares right before wrong; per class, how many were seen and on how
    many the map agreed. Window 9 was seen present where the map says
    absent, window 12 absent where it says present."""
    path = _seen(root)
    _label_seen(path, {4: "1", 9: "1", 12: "0", 3: "0"})
    version = sheet.sheet_version(str(path))
    got = sheet.graded_squares(str(path), version)
    assert got == {
        "n": 4,
        "n_unjudged": 0,
        "face": "seen",
        "cells": [
            {"seen": "1", "wrong": False},
            {"seen": "1", "wrong": True},
            {"seen": "0", "wrong": False},
            {"seen": "0", "wrong": True},
        ],
        "classes": [
            {"code": "1", "name": "Present", "seen": 2, "agreed": 1, "outside": False},
            {"code": "0", "name": "Absent", "seen": 2, "agreed": 1, "outside": False},
        ],
    }
    dumped = json.dumps(got)
    for hidden in ("window_index", "10.5", "50.2", "stratum", '"9"', '"12"'):
        assert hidden not in dumped
    # Another version than the one graded: none.
    assert sheet.graded_squares(str(path), "v" + "a" * 16) is None
    # Cannot tell on a square: counted after the judged ones, with no class
    # (the owner's O2 of 14 Nov 2026); the classes over the judged squares.
    _label_seen(path, {3: "?"})
    assert sheet.graded_squares(str(path), sheet.sheet_version(str(path))) == {
        "n": 4,
        "n_unjudged": 1,
        "face": "seen",
        "cells": [
            {"seen": "1", "wrong": False},
            {"seen": "1", "wrong": True},
            {"seen": "0", "wrong": True},
            {"seen": None, "wrong": False, "unjudged": True},
        ],
        "classes": [
            {"code": "1", "name": "Present", "seen": 2, "agreed": 1, "outside": False},
            {"code": "0", "name": "Absent", "seen": 1, "agreed": 0, "outside": False},
        ],
    }
    # A square with no answer: none.
    _label_seen(path, {3: ""})
    assert sheet.graded_squares(str(path), sheet.sheet_version(str(path))) is None


def test_a_class_seen_on_no_square_is_listed_with_none(root: Path) -> None:
    path = _seen(root)
    _label_seen(path, {4: "0", 9: "0", 12: "0", 3: "0"})
    got = sheet.graded_squares(str(path), sheet.sheet_version(str(path)))
    assert got is not None
    assert got["classes"] == [
        {"code": "1", "name": "Present", "seen": 0, "agreed": 0, "outside": False},
        {"code": "0", "name": "Absent", "seen": 4, "agreed": 2, "outside": False},
    ]
    assert [c["wrong"] for c in got["cells"]] == [False, False, True, True]


def test_a_class_maps_squares_are_graded_as_the_estimate_grades_them(
    root: Path,
) -> None:
    """The class seen matched to the design's classes as the map writes
    them (case aside), wrong where it is not the design's map class there;
    a class the map does not write comes after the map's, is wrong, and is
    marked outside (the last round's contract B2 to B1)."""
    path = _class_map(root)
    sheet.save_labels(
        str(path),
        [
            {"window_index": 7, "reference_class": "Water"},
            {"window_index": 3, "reference_class": "water"},
            {"window_index": 5, "reference_class": "shrubs"},
        ],
    )
    got = sheet.graded_squares(str(path), sheet.sheet_version(str(path)))
    assert got == {
        "n": 3,
        "n_unjudged": 0,
        "face": "class",
        "cells": [
            {"seen": "water", "wrong": False},
            {"seen": "water", "wrong": True},
            {"seen": "shrubs", "wrong": True},
        ],
        "classes": [
            _class_row("water", 2, 1),
            _class_row("trees", 0, 0),
            _class_row("crops", 0, 0),
            _class_row("shrubs", 1, 0, outside=True),
        ],
    }


def test_a_sheet_labelled_right_or_wrong_has_no_squares_graded(root: Path) -> None:
    path = _scores_map(root)
    sheet.save_labels(
        str(path),
        [{"window_index": 1, "wrong": "0"}, {"window_index": 2, "wrong": "1"}],
    )
    assert sheet.graded_squares(str(path), sheet.sheet_version(str(path))) is None
    # A class map's sheet labelled right or wrong only: no class seen.
    land = _class_map(root)
    sheet.save_labels(
        str(land),
        [{"window_index": w, "wrong": "0"} for w in (7, 3, 5)],
    )
    assert sheet.graded_squares(str(land), sheet.sheet_version(str(land))) is None


def test_the_hidden_files_are_never_listed_as_sheets(root: Path) -> None:
    path = _class_map(root)
    sheet.set_typed_rule(str(path), TYPED, typed_at=TYPED_AT)
    sheet.record_grading(str(path), sheet.sheet_version(str(path)))
    hidden = sorted(p.name for p in root.iterdir() if p.name.startswith("."))
    assert f".{path.name}{sheet.RULE_SUFFIX}" in hidden
    assert f".{path.name}{sheet.GRADED_SUFFIX}" in hidden
    assert all(not name.endswith(sheet.SHEET_SUFFIX) for name in hidden)
    for name in hidden:
        with pytest.raises(ValueError):
            sheet.resolve_sheet(os.path.join(root, name))


# ------------------------------------------- the page copy's unit and pictures


def _sourced(root: Path, name: str, source: dict[str, Any]) -> Path:
    """A class map's plan whose population names this source."""
    design = {
        "design": "random",
        "population": {
            "score_kind": "class_map",
            "classes": CLASSES,
            "map_class": CLASS_MAP,
            "source": source,
        },
    }
    (root / f"{name}.json").write_text(json.dumps(design), encoding="utf-8")
    on_map = [CLASSES[c] for c in CLASS_MAP]
    return _write(root / f"{name}_to_label.csv", ONE_MAP, _rows((7, 3, 5), on_map))


def _pair(root: Path, name: str, map_a: dict[str, Any], map_b: dict[str, Any]) -> Path:
    """A two-map plan (which map is more accurate) of these two maps."""
    design = {
        "design": "disagreement",
        "classes": ["crop", "water"],
        "population": {"source": {"map_a": map_a, "map_b": map_b}},
    }
    (root / f"{name}.json").write_text(json.dumps(design), encoding="utf-8")
    return _write(
        root / f"{name}_to_label.csv",
        TWO_MAP,
        [
            ["1", "4", "0", "4", "water", "crop", ""],
            ["2", "5", "0", "5", "crop", "water", ""],
        ],
    )


def test_the_page_copy_says_squares_or_points_and_whether_pictures_can_show(
    root: Path,
) -> None:
    """Contract C1: ``unit`` is "point" for a plan drawn from a Studio result
    (a result_id and no scores file), for one map or two, else "square";
    ``pictures_possible`` is true for a plan drawn from a scores file, or two
    maps each with one. Read off the design alone: the scores files named
    here do not exist and are never opened."""
    assert (sheet.UNIT_SQUARE, sheet.UNIT_POINT) == ("square", "point")
    scores = str(root / "missing_scores.json")
    cases = (
        (_sourced(root, "file", {"scores_path": scores}), "square", True),
        # A scores file pooled from a Studio result's own files names both.
        (
            _sourced(root, "pooled", {"scores_path": scores, "result_id": "r1"}),
            "square",
            True,
        ),
        # A class map without top-k scores: points of a grid over the result.
        (
            _sourced(root, "studio", {"result_id": "r1", "property_name": "lc"}),
            "point",
            False,
        ),
        (_sourced(root, "inline", {"input": "inline scores"}), "square", False),
        (
            _pair(root, "files", {"scores_path": scores}, {"scores_path": scores}),
            "square",
            True,
        ),
        (
            _pair(root, "results", {"result_id": "r1"}, {"result_id": "r2"}),
            "point",
            False,
        ),
    )
    for path, unit, possible in cases:
        read = sheet.read_sheet(str(path))
        assert (read["unit"], read["pictures_possible"]) == (unit, possible), path
    # No design file beside the sheet: squares, and no pictures.
    bare = _write(root / "bare_to_label.csv", ONE_MAP, _rows((1, 2), [0] * 20))
    read = sheet.read_sheet(str(bare))
    assert (read["unit"], read["pictures_possible"]) == ("square", False)
    # Either map of two drawn from a Studio result: points; pictures need both
    # maps' scores files.
    half = {
        "population": {
            "source": {"map_a": {"scores_path": scores}, "map_b": {"result_id": "r"}}
        }
    }
    assert sheet.window_unit(half) == "point"
    assert sheet.pictures_possible(half) is False
    for odd in (None, {}, {"population": []}, {"population": {"source": "x"}}):
        assert sheet.window_unit(odd) == "square"
        assert sheet.pictures_possible(odd) is False


def _points_scores(
    folder: Path, name: str, result_id: Any = "r1", **extra: Any
) -> Path:
    """A Studio result's sampled points as olmoearth_review_set_from_result
    saves them, by the tools' own writer (``SampledScores.to_file``): one
    pixel at each point of a 4 x 4 grid, each point's centre, no window size
    or raster. ``extra`` keys are set over the file's (``None`` drops one)."""
    values = [round(0.05 + 0.06 * i, 3) for i in range(16)]
    rows, kind, assumption = regression_scores(values, min_value=0.0, max_value=1.0)
    sampled = SampledScores(
        result_id=result_id,
        property_name="kelp",
        score_kind=kind,
        assumption=assumption,
        threshold=0.5,
        declared_range=(0.0, 1.0),
        grid=4,
        windows=list(range(16)),
        values=values,
        scores=rows,
        centres=[(18.3 + c / 100, -33.9 - r / 100) for r in range(4) for c in range(4)],
        n_sampled=16,
        n_nodata=0,
        n_failed=0,
    )
    merged = {**sampled.to_file(), **extra}
    payload = {key: value for key, value in merged.items() if value is not None}
    path = folder / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _one(source: dict[str, Any]) -> dict[str, Any]:
    """A design whose population names this source."""
    return {"population": {"source": source}}


def test_a_plan_from_a_studio_results_sampled_points_file_is_of_points(
    root: Path, tmp_path: Path
) -> None:
    """Round 4, finding R3 (the lead's task G): a plan drawn from the scores
    file olmoearth_review_set_from_result saves, a Studio result's sampled
    points (one pixel each, no raster), is of points with no pictures, read
    off that small file's own top level (the source names the result the plan
    copied from it). A file that is not one, or not this result's, stays of
    squares with pictures offered, as before; the file is read again once it
    changes."""
    points = _points_scores(root, "scores_r1_kelp_g4.json")
    source = {
        "scores_path": str(points),
        "result_id": "r1",
        "property_name": "kelp",
        "threshold": 0.5,
    }
    plan = _sourced(root, "sampled", source)
    read = sheet.read_sheet(str(plan))
    assert (read["unit"], read["pictures_possible"]) == ("point", False)
    design = sheet.design_of(sheet.resolve_sheet(str(plan)))[1]
    assert sheet.window_unit(design) == "point"
    assert sheet.picture_sources(design) == []
    # Two maps, each a Studio result's sampled points: points, no pictures.
    other = _points_scores(root, "scores_r2_kelp_g4.json", result_id="r2")
    pair = _one(
        {
            "map_a": {"scores_path": str(points), "result_id": "r1"},
            "map_b": {"scores_path": str(other), "result_id": "r2"},
        }
    )
    assert sheet.window_unit(pair) == "point"
    assert sheet.picture_sources(pair) == []
    # Files that are not a Studio result's sampled points: squares.
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    big = "x" * sheet.POINTS_FILE_MAX_BYTES
    not_points = [
        _points_scores(root, "patch.json", patch_px=11),
        _points_scores(root, "raster.json", raster={"path": "map.tif"}),
        _points_scores(root, "made.json", provenance="read off a map raster"),
        _points_scores(root, "format.json", format="olmoearth-agent/scores@2"),
        _points_scores(root, "grid.json", grid=[4, 5]),
        _points_scores(root, "centres.json", centres_lon_lat=None),
        _points_scores(root, "unnamed.json", result_id=None),
        _points_scores(root, "big.json", padding=big),
        _points_scores(outside, "outside.json"),
        root / "gone.json",
    ]
    for path in not_points:
        one = _one({"scores_path": str(path), "result_id": "r1"})
        assert sheet.window_unit(one) == "square", path
        assert sheet.picture_sources(one) == [str(path)], path
    # The points file of another result, or a source that names no result.
    for named in ({"result_id": "r2"}, {}):
        one = _one({"scores_path": str(points), **named})
        assert sheet.window_unit(one) == "square", named
        assert sheet.pictures_possible(one) is True, named
    # Changed into a file read off a raster: read again, squares.
    payload = json.loads(points.read_text(encoding="utf-8"))
    points.write_text(json.dumps({**payload, "patch_px": 11}), encoding="utf-8")
    assert sheet.window_unit(_one(source)) == "square"
    assert sheet.pictures_possible(_one(source)) is True


def test_a_check_cards_zone_stays_of_squares_and_its_file_is_not_read(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A Check card's zone is a scores file read off a map raster, which may
    name its Studio result: the plan copies its provenance (and its Studio
    block), so its sheet is of squares with pictures offered, and its file is
    never read for this, even one shaped like a sampled points file."""

    def never(_real: str) -> str | None:
        raise AssertionError("a zone's scores file was read for its unit")

    monkeypatch.setattr(sheet, "_points_result", never)
    lookalike = _points_scores(root, "scores_run_zone_p11.json")
    zone = {
        "scores_path": str(lookalike),
        "result_id": "r1",
        "provenance": "Studio's files",
        "studio": {"zone": "EPSG:32611"},
        "model": {"name": "kelp"},
    }
    for drop in (None, "studio", "provenance"):
        source = {k: v for k, v in zone.items() if k != drop}
        assert sheet.window_unit(_one(source)) == "square", drop
        assert sheet.picture_sources(_one(source)) == [str(lookalike)], drop
    read = sheet.read_sheet(str(_sourced(root, "zone", zone)))
    assert (read["unit"], read["pictures_possible"]) == ("square", True)


def test_the_plan_cards_first_button_names_no_unit() -> None:
    """Contract C2 (round 4 pass 2, finding 13): the button before any label."""
    assert sheet.OPEN_FIRST == "Open the sheet here"
    assert sheet.progress_words(0, 3, 0) == ("0 of 3 labelled", sheet.OPEN_FIRST)
    assert "window" not in sheet.OPEN_FIRST and "square" not in sheet.OPEN_FIRST


def test_the_sheets_face_sentences_say_no_window() -> None:
    """The owner's ruling 6 (13 Nov 2026): "windows" goes from the card faces.
    A handle not on the sheet and the held figure name no unit (a sheet is of
    squares or of points); a row with no place is a square's (a Studio
    result's points always carry theirs); the finish message names the
    sheet's own unit word (r5-a L3: "I have answered every square")."""
    assert sheet.UNKNOWN_HANDLE == "That square or point is not on this sheet."
    assert sheet.NO_PLACE == "This square has no place on the sheet to show on a map."
    assert sheet.FIGURE_LATER == (
        "The plan's figure shades the map by its confidence, so it shows here once "
        "the whole sheet is labelled."
    )
    message = sheet.finish_message(
        "estimate", unit=sheet.UNIT_SQUARE, call="c", labels_path="p"
    )
    assert message == (
        "I have answered every square of this sheet. Estimate the error rate. "
        "Estimate the map's error rate from it: c."
    )
    for said in (sheet.UNKNOWN_HANDLE, sheet.NO_PLACE, sheet.FIGURE_LATER, message):
        assert "window" not in said.lower(), said
