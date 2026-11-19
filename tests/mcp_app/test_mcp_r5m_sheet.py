# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Round r5-m (19 Nov 2026), track SHEET, as specs/r5-m-round.md gives it:
the label sheet's items of the build list of the audit after r5-l
(specs/r5-l-audit/build_list.md).

- B1 (the rule and the sheet's part): the app's general rule says "what the
  map is meant to find" (``imagery.READING_RULE``), and so do the two
  captions under the sheet's example pictures (``OE.EXAMPLE_PRESENT``,
  ``OE.EXAMPLE_ABSENT``, the Check card's words); r5-l said "what the map
  shows", which readers took for the map's own answer at the square, the one
  thing the blind sheet hides. The scope and every number are unchanged.
  Records already written keep their words: r5-l's words are kept word for
  word (``imagery.R5L_READING_RULE``), one of ``imagery.APP_RULES`` with the
  general scope, so a sheet, a graded mark or a draw made with them keeps
  them, and so does the map of a chat sheet answered under them
  (``check.result_app_rule``), unless a chat sheet of that map holds the
  kelp rule, the earlier one, which the map keeps. Nothing is written again.
- B2: on a two-map sheet the answers lock once the switch is unticked after
  the finish press and the two maps' answers show (``state.revealed``), and
  stay locked if it is ticked again; the lead's first line then says
  ". The two maps' answers are shown now, so your answers are locked." and
  is the locked answers' description; ticked again (or locked again by an
  answer taken in), it says ". The two maps' answers were shown, so your
  answers are locked." (``W.seenPair``), and on a sheet with no pictures and
  no places either takes the place of its tail (``W.barePairTail``); the
  face's instruction and the line that no rule was set go; the switch reads
  "Hide what the two maps say" (no "(keep it on while you answer)"). A lock
  in the view only.
- B5: a sheet that already gave its result names its finish button, once
  it is ready, "Show the result again" (``W.gradedFinish``), or "See why the
  result does not count" (``W.gradedChanged``) where its answers changed
  since that result (its press gives the withdrawn card then); not ready (a
  blank square, two sets of answers) it keeps the job's label, grey
  (``finishLabel``); the message it sends is unchanged, and what the panel
  tells Claude names the button's words.
- B6: a graded sheet that holds two sets of answers says "This sheet now
  holds two sets of answers." (no "so no result can come from it"); a sheet
  that gave no result keeps the old words.

The views are read as text (no test here runs the scripts). The browser
harness (scripts/mcp_views_e2e.py) draws them: 10_kelp_face, 3j and
16d_sheet_example_pictures (the rule and the captions; 16d checks that no
"what the map shows" is left on the sheet), 3j8_sheet_graded_r5l_rule (a
sheet graded with r5-l's words keeps them), 7_two_map_sheet (B2: the change
after the press moved before the first untick, then the locked answers, the
lead's line ticked and unticked, no face instruction and no line that no
rule was set, and the switch's words), 7b_two_map_sheet_bare (B2 on a
two-map sheet with no pictures and no places), the graded sheets 3j, 3j2,
3j2b, 3j4, 3j5, 3j6, 3j7, 3h2 and 3w (B5's button; grey on 3j2b and 3h2, the
job's label), 3j9_sheet_graded_changed (B5's words where the answers changed)
and 3h2_sheet_graded_mix (B6).

No sheet here holds a place: the sheets have no lon or lat column.
"""

from __future__ import annotations

import csv
import json
import os
import re
from pathlib import Path
from typing import Any

import pytest

mcp = pytest.importorskip("mcp")

from olmoearth_agent.mcp_app import check, imagery, server, sheet, views  # noqa: E402

#: B1: the general rule as this round words it, word for word.
NEW_RULE = (
    "Present: on at least one clear date shown, what the map is meant to find "
    "covers more than half of the outlined square. Absent: on every clear date "
    "shown, it covers half or less."
)
#: B1: the general rule as round r5-l worded it, kept for its records.
R5L_RULE = (
    "Present: on at least one clear date shown, what the map shows covers more "
    "than half of the outlined square. Absent: on every clear date shown, it "
    "covers half or less."
)
#: The kelp rule and its scope, word for word as r5-l kept them.
KELP_RULE = (
    "Present: on at least one clear date shown, canopy fills more than half of "
    "the outlined square. Absent: on every clear date shown it fills half or less."
)
KELP_SCOPE = "This rule came with the app (set 11 Nov 2026 for kelp canopy maps)."
#: The general scope, unchanged by the round.
SCOPE = "This rule came with the app, for any map of present and absent."
#: B1: the words the rule and the captions share, and r5-l's.
MEANT = "what the map is meant to find"
OLD_WORDS = "what the map shows"
#: B1: the sheet's two captions, the Check card's words.
EXAMPLE_PRESENT = (
    "Present: what the map is meant to find covers more than half of the square "
    "on a clear date."
)
EXAMPLE_ABSENT = (
    "Absent: what the map is meant to find covers half of the square or less on "
    "every clear date."
)
#: B2: the end of a two-map sheet's first line while its maps' answers show
#: (W.shownPair), the line whole on a sheet that can show pictures, and the
#: switch's words.
SHOWN_PAIR = ". The two maps' answers are shown now, so your answers are locked."
SHOWN_WHAT = (
    "Say what you see in each square's satellite pictures. The two maps' answers "
    "are shown now, so your answers are locked."
)
BLIND_TEXT = "Hide what the two maps say"
#: B2, after review: the end of the first line once the maps' answers were
#: shown and the switch is ticked again (W.seenPair), and the line whole.
SEEN_PAIR = ". The two maps' answers were shown, so your answers are locked."
SEEN_WHAT = (
    "Say what you see in each square's satellite pictures. The two maps' answers "
    "were shown, so your answers are locked."
)
#: B2, after review: a two-map sheet with no pictures and no places, its
#: first line once the maps' answers were shown (its tail swapped).
BARE_RECORDS = (
    "This sheet has no pictures and no places, so it cannot show where a {one} is. "
    "Answer only from your own records of each {one}"
)
#: B5: a graded sheet's finish button (W.gradedFinish), and its words where
#: the answers changed since the first result (W.gradedChanged).
GRADED_FINISH = "Show the result again"
GRADED_CHANGED = "See why the result does not count"
#: B6: a graded sheet's notice of its two sets of answers, and a sheet's that
#: gave no result (as before).
MIX_GRADED = "This sheet now holds two sets of answers. "
MIX_NOT_GRADED = "This sheet holds two sets of answers, so no result can come from it. "

#: One token of a JS concatenation: a single-quoted string, a plus, or a name.
_TOKEN = re.compile(
    r"\s*(?:'((?:[^'\\]|\\.)*)'|(\+)|([A-Za-z_][A-Za-z0-9_.]*(?:\([^()]*\))?))"
)

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
#: The names a map of presence and absence gives its classes.
NAMES = ["absent", "present"]


# --------------------------------------------------------------------------- the views, read


def _read(name: str) -> str:
    """A file of views/, as it is on disk."""
    return (views.VIEW_DIR / name).read_text(encoding="utf-8")


def _between(text: str, start: str, end: str) -> str:
    """The part of text from start up to the first end after it."""
    at = text.index(start)
    return text[at : text.index(end, at)]


def _script() -> str:
    """views/sheet.js, as it is on disk."""
    return _read("sheet.js")


def _words() -> str:
    """The body of OE.sheetWords, the sheet's table of words."""
    return _between(_script(), "OE.sheetWords = (unit) => {", "\n};")


def _editor() -> str:
    """The body of OE.sheetEditor."""
    return _between(_script(), "OE.sheetEditor = (app, data,", "\n};")


def _code(js: str) -> str:
    """JS text without its comments (a block, or the rest of a line after
    ``//``), each run of joined single-quoted literals read as one: what a
    word gone from the page must be gone from."""
    bare = re.sub(r"/\*.*?\*/", "", js, flags=re.S)
    bare = re.sub(r"//[^\n]*", "", bare)
    return re.sub(r"'\s*\+\s*'", "", bare).replace("\\'", "'")


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


def _entry(block: str, start: str) -> str:
    """One entry's value in a JS object literal: from start to the comma that
    ends its last line."""
    at = block.index(start) + len(start)
    end = block.find(",\n", at)
    return (block[at:] if end < 0 else block[at:end]).strip().rstrip(",")


def _said(key: str, unit: str) -> str:
    """An entry of OE.sheetWords as the face says it, for a unit word."""
    one = "point" if unit == "point" else "square"
    value = re.sub(r"^\([a-z]+\) => ", "", _entry(_words(), f"    {key}: "))
    return _concat(value, {"one": one, "many": one + "s"})


def _tags_above(key: str) -> list[str]:
    """The WORDING lines right above an entry of OE.sheetWords, nearest
    first, up to the first line that is not a WORDING line."""
    lines = _words().split("\n")
    (at,) = [i for i, line in enumerate(lines) if line.startswith(f"    {key}: ")]
    tags: list[str] = []
    for line in reversed(lines[:at]):
        if not line.strip().startswith("// WORDING: "):
            break
        tags.append(line.strip())
    return tags


# --------------------------------------------------------------------------- records


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


def _sheet(root: Path, stem: str = "present", *, result_id: str | None = None) -> Path:
    """A plan of a map of presence and absence: its design (class names absent
    and present; the Studio result it names, when given) and its sheet of four
    windows, no answer, no place."""
    population: dict[str, Any] = {
        "score_kind": "window_confidence",
        "class_names": {"0": NAMES[0], "1": NAMES[1]},
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


def _draw_record(root: Path, key: str, result_id: str, **fields: Any) -> Path:
    """A Check card's draw record, as check.claim_draw and finish_draw leave
    one drawn (``fields`` added: the sheet, the app's rule)."""
    folder = root / "checks"
    folder.mkdir(mode=0o700, exist_ok=True)
    record = {
        "format": check.DRAW_FORMAT,
        "check_key": key,
        "result_id": result_id,
        "state": "drawn",
        "pressed_at": "2026-11-19T09:00:00Z",
        **fields,
    }
    place = folder / f"{key}.draw.json"
    place.write_text(json.dumps(record), encoding="utf-8")
    os.chmod(place, 0o600)
    return place


def _offer(root: Path, result_id: str) -> str:
    """A Check card's offer of one zone of ``result_id`` (20 labels offered, a
    map of presence and absence), as check.offer_zones writes one; its key."""
    zone = "result_epsg32610_0_native.tif"
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
    }
    place = folder / f"{key}.json"
    place.write_text(json.dumps(record), encoding="utf-8")
    os.chmod(place, 0o600)
    return key


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


def _chat_sheet(root: Path, result_id: str) -> Path:
    """A sheet of ``result_id`` planned in the chat, named as the plan tool
    names it (``design_`` and the result's slug), every square answered."""
    stem = f"design_{result_id}_a1b2c3d4_random_4_s0"
    path = _sheet(root, stem, result_id=result_id)
    _answer_all(path)
    return path


#: B1: the labelling rule of a sheet whose records hold r5-l's words.
R5L = {"text": R5L_RULE, "scope": SCOPE, "source": "owner"}


# --------------------------------------------------------------------------- B1: the rule


def test_b1_the_apps_rule_says_what_the_map_is_meant_to_find() -> None:
    """B1: the general rule says what the map is meant to find, never what the
    map shows; its Absent half, its scope and its numbers are as they were,
    and it names no kind of map; its mark is this round's, above r5-l's."""
    assert imagery.READING_RULE == NEW_RULE
    assert MEANT in imagery.READING_RULE and OLD_WORDS not in imagery.READING_RULE
    at = NEW_RULE.index("Absent:")
    assert NEW_RULE[at:] == R5L_RULE[R5L_RULE.index("Absent:") :]
    assert imagery.RULE_SCOPE == SCOPE
    for text in (imagery.READING_RULE, imagery.RULE_SCOPE):
        assert "kelp" not in text.lower() and "canopy" not in text.lower()
    assert re.findall(r"\d", NEW_RULE) == re.findall(r"\d", R5L_RULE)
    assert "# WORDING: r5m B1\n# WORDING: r5l\nREADING_RULE = (" in _source(imagery)
    assert imagery.reading_rule(NAMES) == NEW_RULE
    assert imagery.reading_rule(["water", "land"]) is None


def test_b1_r5l_words_are_kept_for_the_records_made_with_them() -> None:
    """B1: r5-l's words stay word for word as a named constant, one of the
    app's rules with the general scope; the kelp rule and its scope are as
    r5-l kept them."""
    assert imagery.R5L_READING_RULE == R5L_RULE
    assert imagery.KELP_READING_RULE == KELP_RULE
    assert imagery.APP_RULES == {
        NEW_RULE: SCOPE,
        KELP_RULE: KELP_SCOPE,
        R5L_RULE: SCOPE,
    }
    assert imagery.app_rule_scope(R5L_RULE) == SCOPE
    assert imagery.app_rule_scope(NEW_RULE) == SCOPE
    assert imagery.app_rule_scope(KELP_RULE) == KELP_SCOPE


def test_b1_a_sheet_with_no_record_takes_the_new_words(root: Path) -> None:
    """B1: a sheet of presence and absence with nothing typed and no record
    takes the rule's new words and the general scope, in its labelling rule,
    its page copy and its counts."""
    path = _sheet(root)
    assert imagery.rule_for(path.resolve(), NAMES) == (NEW_RULE, SCOPE, "owner")
    assert sheet.labelling_rule(str(path)) == {
        "text": NEW_RULE,
        "scope": SCOPE,
        "source": "owner",
    }
    assert sheet.read_sheet(str(path))["reading_rule"] == NEW_RULE
    assert sheet.sheet_progress(str(path))["reading_rule"] == NEW_RULE


def test_b1_a_sheet_pictured_with_r5l_words_keeps_them(root: Path) -> None:
    """B1: a sheet answered beside pictures that printed r5-l's words keeps
    them and the general scope, in its labelling rule, its page copy and its
    counts; its record is not written again."""
    path = _sheet(root)
    _answer_all(path)
    record = _pictured(path, R5L_RULE)
    before = record.read_bytes()
    real = path.resolve()
    assert imagery.app_rule_of(real) == R5L_RULE
    assert imagery.rule_for(real, NAMES) == (R5L_RULE, SCOPE, "owner")
    assert sheet.labelling_rule(str(path)) == R5L
    assert sheet.read_sheet(str(path))["reading_rule"] == R5L_RULE
    assert sheet.sheet_progress(str(path))["rule_scope"] == SCOPE
    assert record.read_bytes() == before


def test_b1_a_sheet_graded_with_r5l_words_keeps_them(root: Path) -> None:
    """B1: a graded mark written since the round keeps the new words; a mark
    that kept r5-l's words (a grading of the staging trial) keeps its sheet on
    them, with the general scope; the mark is read, never written again."""
    path = _sheet(root)
    _answer_all(path)
    sheet.record_grading(str(path), sheet.sheet_version(str(path)))
    place = path.with_name("." + path.name + sheet.GRADED_SUFFIX)
    mark = json.loads(place.read_text(encoding="utf-8"))
    assert mark["first"]["app_rule"] == NEW_RULE
    mark["first"]["app_rule"] = R5L_RULE
    place.write_text(json.dumps(mark, sort_keys=True), encoding="utf-8")
    os.chmod(place, 0o600)
    before = place.read_bytes()
    assert sheet.graded_app_rule(str(path)) == R5L_RULE
    assert imagery.app_rule_of(path.resolve()) == R5L_RULE
    assert sheet.labelling_rule(str(path)) == R5L
    assert place.read_bytes() == before


def test_b1_a_map_drawn_with_r5l_words_keeps_them(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """B1: a draw record that pinned r5-l's words keeps its sheet and its map
    on them: the sheet's rule, the map's card (settled, the general scope) and
    the rule a new part is drawn with."""
    monkeypatch.setattr(sheet, "result_holds_answers", lambda result_id: False)
    path = _sheet(root)
    _draw_record(
        root,
        "d" * 24,
        "r-words",
        labels_csv_path=os.path.realpath(path),
        app_rule=R5L_RULE,
    )
    assert check.sheet_app_rule(str(path)) == R5L_RULE
    assert imagery.app_rule_of(path.resolve()) == R5L_RULE
    assert sheet.labelling_rule(str(path)) == R5L
    assert check.map_app_rule("r-words") == R5L_RULE
    assert check._app_rule_for("r-words") == R5L_RULE
    assert server._check_rule(False, "r-words") == {
        "set": True,
        "text": R5L_RULE,
        "source": "owner",
        "scope": SCOPE,
        "replaceable": False,
    }
    # A map never pressed takes the new words, with the box.
    assert check._app_rule_for("r-fresh") == NEW_RULE
    assert server._check_rule(False, "r-fresh")["text"] == NEW_RULE


def test_b1_a_map_whose_chat_sheet_holds_r5l_words_keeps_them(root: Path) -> None:
    """B1: a map with no Check card draw whose sheet planned in the chat was
    answered beside pictures that printed r5-l's words shows them, settled,
    on its card, and its first part pressed pins them (one map keeps one
    rule, as r5-l kept the kelp rule); a chat sheet no record speaks for
    leaves the map's rule to its draws."""
    _pictured(_chat_sheet(root, "r-talk"), R5L_RULE)
    assert check.result_app_rule("r-talk") == R5L_RULE
    assert server._check_rule(False, "r-talk") == {
        "set": True,
        "text": R5L_RULE,
        "source": "owner",
        "scope": SCOPE,
        "replaceable": False,
    }
    claim = check.claim_draw(_offer(root, "r-talk"), 20)
    assert claim["outcome"] == "first" and claim["draw"]["app_rule"] == R5L_RULE
    _chat_sheet(root, "r-blank")
    assert check.result_app_rule("r-blank") is None
    assert server._check_rule(False, "r-blank")["text"] == NEW_RULE
    source = _source(check)
    assert "            if kept == imagery.KELP_READING_RULE:\n" in source
    assert "            return imagery.R5L_READING_RULE\n" in source


def test_b1_a_map_whose_chat_sheets_hold_both_kept_rules_keeps_the_kelp_rule(
    root: Path,
) -> None:
    """B1, after review: a map with no draw whose two chat sheets were
    answered, one under the kelp rule and one under r5-l's words, keeps the
    kelp rule, the earlier one (one map keeps its first rule), wherever its
    sheet sits among the sheets (newest first): on its card, with the kelp
    rule's scope, and for its first part pressed."""
    kelp = _sheet(root, "design_r-both_a1b2c3d4_random_4_s0", result_id="r-both")
    words = _sheet(root, "design_r-both_e5f6a7b8_random_4_s1", result_id="r-both")
    for path, rule in ((kelp, KELP_RULE), (words, R5L_RULE)):
        _answer_all(path)
        _pictured(path, rule)
    for newer, older in ((words, kelp), (kelp, words)):
        os.utime(older, (1_700_000_000, 1_700_000_000))
        os.utime(newer, (1_800_000_000, 1_800_000_000))
        assert check.result_app_rule("r-both") == KELP_RULE
        said = server._check_rule(False, "r-both")
        assert said["text"] == KELP_RULE and said["scope"] == KELP_SCOPE
    claim = check.claim_draw(_offer(root, "r-both"), 20)
    assert claim["outcome"] == "first" and claim["draw"]["app_rule"] == KELP_RULE


# --------------------------------------------------------------------------- B1: the captions


def test_b1_the_sheets_captions_say_what_the_map_is_meant_to_find() -> None:
    """B1 on the label sheet: the two captions under the example pictures say
    what the map is meant to find, the rule's words, each marked above r5-l's
    and r4-k's tags; no code of sheet.js says "what the map shows" any more,
    and the source line is as it was."""
    script = _script()
    for name, said in (
        ("OE.EXAMPLE_PRESENT", EXAMPLE_PRESENT),
        ("OE.EXAMPLE_ABSENT", EXAMPLE_ABSENT),
    ):
        tags = (
            "// WORDING: r5m B1\n// WORDING: r5l L2\n// WORDING: r4k K9\n"
            f"{name} = '{said}';\n"
        )
        assert tags in script, name
        assert MEANT in said and MEANT in imagery.READING_RULE
        assert "kelp" not in said.lower() and "rule" not in said
    assert OLD_WORDS not in _code(script)
    assert (
        "// WORDING: r5l L2\nOE.EXAMPLES_SOURCE = "
        "'Example pictures from a kelp canopy map.';\n" in script
    )


# --------------------------------------------------------------------------- B2


def test_b2_the_answers_lock_once_the_two_maps_answers_were_shown() -> None:
    """B2: the lock holds once a two-map sheet showed what the two maps say
    (state.revealed), beside a graded sheet and one that stopped saving;
    revealed is set only once the unblind read came back, and nothing sets it
    back, so ticking the switch again keeps the answers locked; every answer
    path asks blocked, which asks locked."""
    editor = _editor()
    assert (
        "  const locked = () => gaveResult || state.stale || state.revealed;\n"
        in editor
    )
    assert "  const blocked = () => state.clearing || locked();\n" in editor
    assert "    revealed: false,\n" in editor
    assert editor.count("state.revealed = ") == 1
    blind = _between(editor, "  async function setBlind(on) {", "\n  }\n")
    assert blind.index("unblind: true") < blind.index("state.revealed = true;")
    assert "if (on || state.revealed) { state.blind = on; render(); return; }" in blind
    set_label = _between(editor, "  function setLabel(h, value) {", "\n  }\n")
    assert set_label.startswith(
        "  function setLabel(h, value) {\n    if (blocked()) return;"
    )


def test_b2_the_first_line_says_the_answers_are_locked_and_describes_them() -> None:
    """B2: with the maps' answers shown, the two-map sheet's first line ends
    "The two maps' answers are shown now, so your answers are locked.", under
    this round's mark above r5-e's and r5-b's; the line has an id, and the
    locked answers name it as their description (a graded sheet's notice
    first, the stale line last)."""
    assert _said("shownPair", "square") == SHOWN_PAIR
    assert _tags_above("shownPair") == [
        "// WORDING: r5b J28",
        "// WORDING: r5e H1",
        "// WORDING: r5m B2",
    ]
    what = _said("what", "square")
    assert what.replace(_said("hiddenOne", "square"), SHOWN_PAIR) == SHOWN_WHAT
    editor = _editor()
    lock = _between(editor, "  function lockAnswers() {", "\n  }\n")
    assert (
        "const why = gaveResult && gradedNote ? gradedNote : state.revealed && "
        "pairWhat ? pairWhat : staleLine;" in lock
    )
    assert "c.setAttribute('aria-describedby', why.id);" in lock
    made = (
        "  if (pairWhat) pairWhat.id = 'oe-pair-what-' + "
        "Math.random().toString(36).slice(2);\n"
    )
    assert made in editor
    assert (
        editor.index("  const pairWhat = ")
        < editor.index(made)
        < editor.index("  const pairHidden = ")
    )
    drawn = _between(editor, "  function render(focusKey) {", "\n  }\n")
    assert (
        "    if (pairWhat) pairWhat.textContent = !state.revealed ? pairHidden : "
        "state.blind ? pairSeen : pairShown;\n" in drawn
    )


def test_b2_ticked_again_the_first_line_still_says_the_answers_are_locked() -> None:
    """B2, after review: once the maps' answers were shown the first line
    says the answers are locked, the switch ticked again too (or locked again
    by an answer taken in from elsewhere): there it says they were shown
    (W.seenPair, this round's mark alone), never the hidden form, which asked
    for answers the lock refuses. Both locked forms come from one swap of the
    hidden clause, or, on a two-map sheet with no pictures and no places
    (whatBarePair, which has no such clause), of its tail (W.barePairTail,
    its own words), so that line says it too and asks nothing more."""
    assert _said("seenPair", "square") == SEEN_PAIR
    assert _tags_above("seenPair") == ["// WORDING: r5m B2"]
    what = _said("what", "square")
    assert what.replace(_said("hiddenOne", "square"), SEEN_PAIR) == SEEN_WHAT
    for unit in ("square", "point"):
        bare = _said("whatBarePair", unit)
        tail = _said("barePairTail", unit)
        assert bare.endswith(tail) and bare.count(tail) == 1, unit
        assert _said("hiddenPair", unit) not in bare, unit
        for end in (SHOWN_PAIR, SEEN_PAIR):
            locked = bare.replace(tail, end)
            assert locked == BARE_RECORDS.format(one=unit) + end, unit
            assert "without looking" not in locked and "No records" not in locked
    editor = _editor()
    swap = (
        "  const pairSwap = (s) => (pairHidden.includes(W.hiddenPair)\n"
        "    ? pairHidden.replace(W.hiddenPair, s) : "
        "pairHidden.replace(W.barePairTail, s));\n"
        "  const pairShown = pairSwap(W.shownPair);\n"
        "  const pairSeen = pairSwap(W.seenPair);\n"
    )
    assert swap in editor
    assert editor.index("  const pairHidden = ") < editor.index(swap)
    # The locked answers keep the line as their description (lockAnswers).
    lock = _between(editor, "  function lockAnswers() {", "\n  }\n")
    assert "state.revealed && pairWhat ? pairWhat : staleLine;" in lock


def test_b2_nothing_on_a_two_map_sheet_whose_maps_answers_were_shown_asks() -> None:
    """B2, after review: once the maps' answers were shown (the answers
    locked), the face's instruction says nothing, right after a graded
    sheet's (r5-k K3) and before the two-map line, so render hides it; and
    the line that no rule was set before the first answer is hidden, as on a
    graded sheet (r5-b J06), for good in this panel."""
    editor = _editor()
    told = _between(editor, "  function faceInstructionText() {", "\n  }\n")
    k3 = "    if (gaveResult) return '';\n"
    mine = (
        "    // r5-m B2: a two-map sheet whose maps' answers were shown takes no "
        "answer (locked), so nothing asks for one.\n"
        "    if (state.revealed) return '';\n"
    )
    assert told.index(k3) < told.index(mine) < told.index("    if (twoMap) return ")
    drawn = _between(editor, "  function render(focusKey) {", "\n  }\n")
    assert "    faceInstruction.hidden = !faceInstruction.textContent;\n" in drawn
    assert "    if (noRuleLine && state.revealed) noRuleLine.hidden = true;\n" in drawn


def test_b2_the_switch_says_what_it_hides_and_asks_nothing() -> None:
    """B2: the switch's words are "Hide what the two maps say", under this
    round's mark above the old tags; "keep it on while you answer" is gone
    from the code; the switch's lock (K2) is as it was."""
    script = _script()
    assert (
        "  // WORDING: r5m B2\n  // WORDING: r4i F11; r5b J28\n"
        f"  const blindText = '{BLIND_TEXT}';\n" in script
    )
    assert "keep it on" not in _code(script)
    editor = _editor()
    assert (
        "  const pairLocked = () => !(ready() && (resultSent || gaveResult));\n"
        in editor
    )


# --------------------------------------------------------------------------- B5


def test_b5_a_graded_sheets_finish_button_shows_the_result_again() -> None:
    """B5: a sheet that already gave its result names its finish button, once
    it is ready, "Show the result again" (W.gradedFinish, this round's mark,
    right after everyDoneGraded), or, after review, "See why the result does
    not count" (W.gradedChanged) where its answers changed since that result
    (the page copy's graded changed_since_first: its press gives the withdrawn
    card); a sheet not graded, and a graded one not ready (grey), keep the
    job's label (finishLabel); the button is made with the job's label and
    updateFinish names it, the node kept; the message the press sends is
    unchanged; what the panel tells Claude names the button's words, said
    only when the sheet is ready."""
    assert _said("gradedFinish", "square") == GRADED_FINISH
    assert _said("gradedFinish", "point") == GRADED_FINISH
    assert _tags_above("gradedFinish") == ["// WORDING: r5m B5"]
    assert _said("gradedChanged", "square") == GRADED_CHANGED
    assert _said("gradedChanged", "point") == GRADED_CHANGED
    assert _tags_above("gradedChanged") == ["// WORDING: r5m B5"]
    words = _words()
    assert (
        words.index("    everyDoneGraded: ")
        < words.index("    gradedFinish: ")
        < words.index("    gradedChanged: ")
        < words.index("    gradedBlank: ")
    )
    editor = _editor()
    label = _between(editor, "  function finishLabel(ok) {", "\n  }\n")
    assert (
        "    if (!gaveResult || !ok) return data.finish_label || "
        "'Ask Claude to grade it';\n"
        "    return data.graded.changed_since_first === true ? W.gradedChanged : "
        "W.gradedFinish;" in label
    )
    control = _between(editor, "  function finishControl() {", "\n  }\n")
    assert "      }, data.finish_label || 'Ask Claude to grade it');\n" in control
    assert "gaveResult" not in control and "W.graded" not in control
    assert "const message = data.finish_message || (" in control
    assert "onclick: () => finishPress(message)," in control
    update = _between(editor, "  function updateFinish() {", "\n  }\n")
    assert (
        "    const ok = ready();\n"
        "    // WORDING: r5m B5 (the button's words as the sheet stands, "
        "finishLabel; the node stays, so it keeps the focus)\n"
        "    const label = finishLabel(ok);\n"
        "    if (button.textContent !== label) button.textContent = label;\n"
    ) in update
    told = _between(editor, "  function contextText() {", "\n  }\n")
    assert "    else if (complete() && !left) {\n" in told
    assert "its finish button reads \"' + finishLabel(true) + '\".';" in told
    assert "  const ready = () => complete() && !pending() && !state.stale;\n" in editor
    assert "W.graded" not in _between(
        editor, "  async function finishPress(message) {", "\n  }\n"
    )


def test_b5_an_unfinished_estimate_of_a_graded_sheet_names_its_grey_button(
    root: Path,
) -> None:
    """B5, after review: a graded sheet with a square blanked elsewhere is not
    ready, so its grey finish button keeps the job's label (finishLabel), and
    the card of an estimate refused for that square names those words
    (``server._finish_label``, as the sheet draws them; ``_unfinished``)."""
    path = _sheet(root)
    _answer_all(path)
    sheet.record_grading(str(path), sheet.sheet_version(str(path)))
    with open(path, encoding="utf-8", newline="") as fh:
        rows = list(csv.reader(fh))
    for column in ("wrong", "reference_class"):
        rows[-1][rows[0].index(column)] = ""
    with open(path, "w", encoding="utf-8", newline="") as fh:
        csv.writer(fh).writerows(rows)
    progress = sheet.sheet_progress(str(path))
    assert isinstance(progress["graded"], dict)
    assert progress["n_labelled"] < progress["n_rows"]
    press = server._finish_label(str(path))
    assert press == progress["finish_label"] == sheet.JOB_LABELS["estimate"]
    error = "1 of 4 design windows have no label."
    unfinished = server._unfinished(error, sheet.UNIT_SQUARE, press)
    assert unfinished is not None and unfinished["press"] == press
    assert GRADED_FINISH not in unfinished["text"]


# --------------------------------------------------------------------------- B6


def test_b6_a_graded_sheet_with_two_sets_says_only_what_it_holds() -> None:
    """B6: on a sheet that gave its result the notice of two sets of answers
    says only that the sheet now holds them; a sheet that gave none keeps the
    old words; the marks keep r5-b's under this round's; the Remove button's
    guard and the Details sentence are as they were."""
    editor = _editor()
    progress = _between(editor, "  function updateProgress() {", "\n  }\n")
    made = (
        "      // WORDING: r5m B6\n"
        "      // WORDING: r5b J21\n"
        "      notice.append(OE.el('div', {}, OE.el('span', { class: 'error' }, "
        "gaveResult\n"
        f"        ? '{MIX_GRADED}'\n"
        f"        : '{MIX_NOT_GRADED}'),\n"
    )
    assert made in progress
    assert "no result" not in MIX_GRADED
    assert (
        "      mix.column && !locked() ? OE.el('button', { class: 'small', "
        "onclick: () => clearAndOpen(mix.column, face) },\n" in progress
    )
    assert (
        "'The estimate would refuse this sheet: ' + state.mix.problem + '.'" in progress
    )


# --------------------------------------------------------------------------- the docs


def test_r5m_sheet_the_changelog_and_the_usage_say_so() -> None:
    """The CHANGELOG's marker is replaced by one line per item, this round's
    own (an earlier round's "    - B1: " to "    - B9: " lines are there too),
    with the words quoted, and docs/Usage.md gives the new words."""
    stage = Path(__file__).resolve().parents[2]
    changelog = (stage / "CHANGELOG.md").read_text(encoding="utf-8")
    assert "@@R5M-SHEET@@" not in changelog
    for line in (
        "\n    - B1: the app's labelling rule for a map of presence and absence says "
        "what the map is meant to find: ",
        "\n    - B2: on a two-map sheet the answers are locked once the switch is "
        "unticked",
        "\n    - B5: on a label sheet that already gave its result the finish button "
        'reads "Show the result again"',
        "\n    - B6: on a label sheet that gave its result and now holds two sets of "
        "answers",
    ):
        assert line in changelog, line
    for quoted in (
        f'"{NEW_RULE}"',
        f'"{SHOWN_WHAT}"',
        f'"{SEEN_WHAT}"',
        f'"{BLIND_TEXT}"',
        f'"{GRADED_FINISH}"',
        f'"{GRADED_CHANGED}"',
        f'"{MIX_GRADED.strip()}"',
        "`imagery.R5L_READING_RULE`",
    ):
        assert quoted in changelog, quoted
    usage = " ".join((stage / "docs" / "Usage.md").read_text(encoding="utf-8").split())
    for said in (
        NEW_RULE,
        EXAMPLE_PRESENT,
        EXAMPLE_ABSENT,
        SHOWN_WHAT,
        SEEN_WHAT,
        f'"{BLIND_TEXT}"',
        GRADED_FINISH,
        GRADED_CHANGED,
        MIX_GRADED.strip(),
        "imagery.R5L_READING_RULE",
    ):
        assert said in usage, said
