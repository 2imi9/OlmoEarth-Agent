# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""The answer checks (harness/checks.py), one detector at a time.

The flagged cases are the kinds of statement the exp86 round 6 and 7 audits
confirmed false (a direction read backwards, a place read off a listing, a
file said to be saved, an alpha chosen after the bounds); each has a
companion the detector must leave alone, since a false alarm costs a
rewrite and a marked sentence.
"""

from __future__ import annotations

import time
from typing import Any

import pytest

from olmoearth_agent.harness.checks import (
    ACTIONS,
    CHECKS,
    DIRECTION,
    FORBIDDEN,
    FORBIDDEN_DETECTORS,
    MUST_STATE,
    NUMBERS,
    RunEvidence,
    ToolRecord,
    check_actions,
    check_direction,
    check_forbidden_claims,
    check_must_state,
    check_numbers,
    grounding_prompt,
    mark_answer,
    revision_prompt,
    run_checks,
    sentences,
)


def _record(name: str, result: dict[str, Any], **arguments: Any) -> ToolRecord:
    return ToolRecord(name, arguments, {"ok": True, "result": result})


def _run(*records: ToolRecord, surface: str = "cli", brief: str = "") -> RunEvidence:
    return RunEvidence(tools=list(records), user_messages=[brief], surface=surface)


def _flagged(check: Any, answer: str, run: RunEvidence) -> list[str]:
    return [v["text"] for v in check(answer, run)]


# --- sentences ---------------------------------------------------------------


def test_sentences_split_lines_and_full_stops_not_abbreviations() -> None:
    text = (
        "Intro, e.g. 1,000 labels. Next one? Yes; ok.\n"
        "\n"
        "| Metric | Value |\n"
        "|---|---:|\n"
        "| Differing | 3,807 (23.2%) |\n"
        "- a bullet vs. another. the tail stays"
    )
    assert [s.text for s in sentences(text)] == [
        "Intro, e.g. 1,000 labels.",
        "Next one?",
        "Yes;",
        "ok.",
        "| Metric | Value |",
        "| Differing | 3,807 (23.2%) |",
        "- a bullet vs. another. the tail stays",
    ]
    for s in sentences(text):
        assert text[s.start : s.end] == s.text


# --- numbers -----------------------------------------------------------------


def test_the_number_check_names_the_sentence_and_the_number() -> None:
    run = _run(_record("summary", {"n_windows": 16384}))
    answer = "16,384 windows. About 5,565 remain; 16,384 in all."
    assert check_numbers(answer, run) == [
        {"check": NUMBERS, "text": "About 5,565 remain;", "detail": "5,565"}
    ]


# --- direction: dominant change ------------------------------------------------

_WATER = {"0": "not water", "1": "water"}


def _flips(**fact: Any) -> RunEvidence:
    """exp86 B7: 1,247 of 1,570 differing windows flip 0 -> 1, 323 flip back."""
    facts = [
        {
            "id": "dominant_change",
            "from_class": 0,
            "to_class": 1,
            "n": 1247,
            "share": 0.794,
            "reverse_n": 323,
            "reverse_share": 0.206,
            "tied_with_reverse": False,
            "sentence": "1,247 of 1,570 differing windows (79.4%) change 0 -> 1.",
            **fact,
        }
    ]
    return _run(
        _record("scores", {"classes": _WATER}),
        _record("compare", {"n_differing": 1570, "facts": facts}),
    )


@pytest.mark.parametrize(
    "sentence",
    [
        "Most differing windows flip class 1 -> 0 (A positive, B negative).",
        "The changes are mostly water -> not water.",
        "Differences are dominated by windows that flip from class 1 to class 0.",
        "Most flips go A=1 → B=0.",
        "Most windows turn from water into not water.",
        "The main change: water becomes not water.",
        "The dominant change is water to not water.",
    ],
)
def test_a_dominant_change_stated_backwards_is_flagged(sentence: str) -> None:
    violations = check_direction(sentence, _flips())
    assert [v["text"] for v in violations] == [sentence]
    assert "opposite direction" in violations[0]["detail"]
    assert "1,247 of 1,570" in violations[0]["detail"]


@pytest.mark.parametrize(
    "sentence",
    [
        "Most differing windows flip class 0 -> 1.",
        "Most changes are not water -> water.",
        "Class flips are mostly 1→0 or 0→1 along cluster edges.",
        "323 windows flip 1 -> 0.",  # a minor change, stated as such
        "Window 12 flips 1 -> 0 and window 13 as well.",
        "Most differing windows sit in the south.",
        # A bare digit is never a class: "from 1 to 0" may be anything.
        "Differences are dominated by windows that flip from 1 to 0.",
        "Most differences span rows 1 to 0 of the tile.",
        # Not a change between the maps.
        "Most of the map is water, from the lake to not water uplands.",
        # A region's change, not the whole comparison's.
        "In the south the main change is class 1 -> 0.",
        "Most windows along the river flip water -> not water.",
    ],
)
def test_a_dominant_change_stated_forwards_or_not_as_dominant_passes(
    sentence: str,
) -> None:
    assert check_direction(sentence, _flips()) == []


_AWF = {
    "0": "woodland_forest",
    "2": "shrubland_savanna",
    "4": "grassland_barren",
    "6": "montane_forest",
}


def _awf(*facts: dict[str, Any], **result: Any) -> RunEvidence:
    return _run(
        _record("scores", {"classes": _AWF, "grid": [128, 128]}),
        _record("compare", {"n_differing": 3807, "facts": list(facts), **result}),
    )


_DOMINANT = {
    "id": "dominant_change",
    "from_class": 4,
    "to_class": 2,
    "n": 1409,
    "share": 0.37,
    "reverse_n": 266,
    "reverse_share": 0.07,
    "tied_with_reverse": False,
}


def test_another_pair_named_the_dominant_one_is_flagged() -> None:
    """Round 6, B3/cluster run 2: the largest pair was 4 -> 2 at 37%."""
    run = _awf(_DOMINANT)
    flagged = "The dominant contrast is **montane_forest vs woodland_forest**."
    assert _flagged(check_direction, flagged, run) == [flagged]
    for fine in (
        "The dominant change is grassland_barren -> shrubland_savanna (37.0%).",
        "Top transitions: grassland_barren -> shrubland_savanna (37.0%), "
        "shrubland_savanna -> grassland_barren (7.0%).",
        "The largest contrast is grassland vs shrubland.",
        "montane_forest vs woodland_forest is the second pair.",
        "The main change covers 0 / 6 of the listed rows.",  # digits, not classes
    ):
        assert check_direction(fine, run) == [], fine
    backwards = "The dominant change is shrubland_savanna -> grassland_barren."
    assert _flagged(check_direction, backwards, run) == [backwards]


@pytest.mark.parametrize(
    "sentence",
    [
        # What the maps are made of, not what changes between them.
        "Both maps are dominated by montane_forest and woodland_forest.",
        "The main classes are montane_forest and woodland_forest.",
        "The dominant classes in C1 are montane_forest / woodland_forest.",
        # A digit range is no class pair ("2 to 4 rows"), nor a change.
        "The main differing zone is 2 to 4 rows deep.",
        "The dominant change spans 2 to 4 rows.",
        # A pair ranked after the dominant one.
        "The second largest change is montane_forest -> woodland_forest.",
    ],
)
def test_a_class_pair_not_about_change_is_not_a_dominant_change_claim(
    sentence: str,
) -> None:
    assert check_direction(sentence, _awf(_DOMINANT)) == [], sentence


@pytest.mark.parametrize(
    "sentence",
    [
        "The main change is montane_forest -> woodland_forest.",
        "Most flips are between montane_forest and woodland_forest, the dominant "
        "transition.",
        "The dominant contrast is montane_forest vs woodland_forest.",
    ],
)
def test_another_pair_named_the_dominant_change_is_flagged(sentence: str) -> None:
    assert _flagged(check_direction, sentence, _awf(_DOMINANT)) == [sentence]


def test_the_digit_of_class_n_is_a_class_and_a_bare_one_is_not() -> None:
    run = _awf(_DOMINANT)
    assert _flagged(check_direction, "The main change is class 2 -> 4.", run)
    assert _flagged(check_direction, "Most windows flip from class 2 to 4.", run)
    assert not check_direction("The main change is class 4 -> 2.", run)
    assert not check_direction("Most windows flip from 2 to 4 rows north.", run)


def test_a_reverse_tied_with_the_top_count_is_no_contradiction() -> None:
    backwards = "The dominant change is shrubland_savanna -> grassland_barren."
    tied = {**_DOMINANT, "reverse_n": 1409, "tied_with_reverse": True}
    assert check_direction(backwards, _awf(tied)) == []
    # Tied with another direction: the fact's sentence says so and names both.
    other = {
        **_DOMINANT,
        "sentence": "4 -> 2 and 6 -> 0 are tied at 1,409 windows each.",
    }
    named = "The dominant change is montane_forest -> woodland_forest."
    assert check_direction(named, _awf(other)) == []
    assert _flagged(check_direction, named, _awf(_DOMINANT)) == [named]


# --- direction: the more confident side ----------------------------------------


def _sides(side: str = "A", share_a: float = 0.8) -> RunEvidence:
    fact = {
        "id": "more_confident_side",
        "side": side,
        "share_a": share_a,
        "share_b": round(1 - share_a, 6),
        "share_equal": 0.0,
    }
    return _run(
        _record(
            "compare",
            {"facts": [fact]},
            scores_path_a="F4/emsr279-11_s1_pre.json",
            scores_path_b="F4/emsr279-11_s1_post.json",
        )
    )


@pytest.mark.parametrize(
    "sentence",
    [
        "B (post) is usually the more confident side (mean margin 0.158 vs 0.389).",
        # A hedge in another clause leaves the claim standing.
        "B is usually the more confident side, but confidence does not pick a winner.",
        "Map B is the more confident map.",
        "post is mostly more confident.",
        "The more confident side is B.",
    ],
)
def test_the_other_side_named_more_confident_is_flagged(sentence: str) -> None:
    assert _flagged(check_direction, sentence, _sides()) == [sentence]


@pytest.mark.parametrize(
    "sentence",
    [
        "A is usually the more confident side.",
        "B is more confident on 20.0% of them.",  # the minority, stated as such
        "B is more confident on average (mean margin 0.4 vs 0.2).",
        "Neither side is more confident.",
        "A more confident side is right only on some windows.",
    ],
)
def test_the_side_the_tool_found_or_a_hedge_passes(sentence: str) -> None:
    assert check_direction(sentence, _sides()) == []


@pytest.mark.parametrize(
    "sentence",
    [
        # A region, not the whole comparison (the round 6 audit's review).
        "B is more confident in the north.",
        "B is usually more confident along the river, where A hesitates.",
        "Map B is mostly more confident in rows 3-5.",
        "B is more confident in the cluster at the east edge.",
    ],
)
def test_a_side_more_confident_in_a_region_is_no_claim_about_the_side(
    sentence: str,
) -> None:
    assert check_direction(sentence, _sides()) == []


def test_neither_side_more_confident_flags_either_side_named() -> None:
    run = _sides("neither", 0.45)
    for flagged in (
        "A is usually the more confident side.",
        "Map B is more confident.",
    ):
        violations = check_direction(flagged, run)
        assert [v["text"] for v in violations] == [flagged], flagged
    assert check_direction("Neither side is more confident.", run) == []


def test_side_labels_come_from_the_calls_that_named_the_inputs() -> None:
    """C1 and C2 are the run directories whose scores files went in as A and B."""
    fact = {
        "id": "more_confident_side",
        "side": "B",
        "share_a": 0.421592,
        "share_b": 0.578408,
        "share_equal": 0.0,
    }
    run = _run(
        _record("scores", {"scores_path": "/w/s_a7c4_p4.json"}, run_dir="C1/awf_2023"),
        _record("scores", {"scores_path": "/w/s_eddc_p4.json"}, run_dir="C2/awf_2022"),
        _record(
            "compare",
            {"facts": [fact]},
            scores_path_a="/w/s_a7c4_p4.json",
            scores_path_b="/w/s_eddc_p4.json",
        ),
    )
    assert _flagged(check_direction, "C1 is usually more confident.", run)
    assert _flagged(check_direction, "The 2023 map is mostly more confident.", run)
    assert not check_direction("C2 is usually more confident.", run)
    both = "C2 is more confident overall, but C1 is more confident on 42.2% of them."
    assert not check_direction(both, run)


# --- direction: concentration --------------------------------------------------


def _where(
    top: float = 0.394, axis: str = "rows", band: int = 0, share: float = 0.394
) -> dict[str, Any]:
    """A ``concentration`` fact on AWF's 128 x 128 grid, 3,807 windows differing."""
    return {
        "id": "concentration",
        "grid": [128, 128],
        "n_differing": 3807,
        "top_band_share": top,
        "max_band": {
            "axis": axis,
            "band": band,
            "of_grid": f"{25 * band}-{25 * (band + 1)}%",
            "share": share,
        },
    }


def test_a_place_the_bands_contradict_is_flagged() -> None:
    run = _awf(_where(top=0.05, band=2, share=0.47))
    for flagged in (
        "Most differences sit along the north edge.",
        "The differing windows are concentrated in the top rows.",
        "Differences are dominated by a strip in the north edge (north border).",
    ):
        assert _flagged(check_direction, flagged, run) == [flagged], flagged
    for fine in (
        "A small cluster sits at the north edge.",
        "Most differences are in rows 50-75%, e.g. near the north edge.",
        "Few differences are at the north edge.",
        "The north edge holds 51 of the 3,807 differing windows.",
        # "The north" may be the northern half, which the top band does not settle.
        "Most differences are in the north.",
        # Another clause places them; this one only names the edge.
        "Most differences are mid-grid; the north edge has a few.",
    ):
        assert check_direction(fine, run) == [], fine


def test_row_band_0_is_called_north_only_when_the_rows_run_north_to_south() -> None:
    """The fix-r8 review: the violation said "where the northmost band holds
    1.6%" of F4's 400 chips stacked in dataset order, which carry no
    georeference (row_order None). A fact from before row_order is north-up."""
    claim = "The differing windows are concentrated in the top rows."
    for order, says in (
        (None, "row band 0 (the grid's first rows)"),
        ("south_to_north", "the south edge here"),
        ("north_to_south", "the northmost band"),
    ):
        fact = {**_where(top=0.016, band=2, share=0.47), "row_order": order}
        (v,) = check_direction(claim, _awf(fact))
        assert says in v["detail"], (order, v["detail"])
        if order != "north_to_south":
            assert "northmost" not in v["detail"], v["detail"]
    legacy = check_direction(claim, _awf(_where(top=0.016, band=2, share=0.47)))
    assert "the northmost band" in legacy[0]["detail"]
    # Row 0 south: a north edge is the last row band, which top_band_share
    # does not count, so a compass claim is not read against it.
    south = _awf(
        {**_where(top=0.016, band=2, share=0.47), "row_order": "south_to_north"}
    )
    assert check_direction("Most differences sit along the north edge.", south) == []


def test_another_edge_contradicts_only_the_band_that_holds_the_most() -> None:
    """``max_band`` rows band 1: the south edge (band 3) cannot hold most."""
    run = _awf(_where(top=0.3, band=1, share=0.41))
    flagged = "Most differing windows cluster along the south edge."
    assert _flagged(check_direction, flagged, run) == [flagged]
    # The bands of the other axis say nothing about the east edge.
    assert check_direction("Most flips are along the east edge.", run) == []
    same = _awf(_where(top=0.1, band=3, share=0.5))
    assert check_direction(flagged, same) == []


def test_a_strip_too_narrow_for_most_differences_is_flagged() -> None:
    """Round 6, B3/cluster run 1: row 0 holds 51 of 3,807 differing windows; a
    row of 128 windows cannot hold most of them."""
    run = _awf(_where())
    for flagged in (
        "Dominated by a long strip in the north edge (row 0, cols ~3-48).",
        "The top-50 differing windows sit mostly in row 0 (north edge).",
        "The most confidence-split zone runs along row 0 (cols 3-51).",
    ):
        violations = check_direction(flagged, run)
        assert [v["text"] for v in violations] == [flagged], flagged
        assert "at most 128 windows of the 3807" in violations[0]["detail"]
    for fine in (
        "Most differences lie in rows 10-40.",  # 31 rows hold 3,968 windows
        "The dominant contrast is grassland vs shrubland (e.g. rows 0-1).",
        "A tight cluster sits at rows 1442-1449.",  # not a claim about most
        "Most differing windows are named by row (row 0 = north).",  # a legend
        "Most are small clusters: e.g. row 3, rows 10-12 and a block at rows 20-24.",
        "Rows 3-5 have low margins on both sides, the most suspect.",  # superlative
        # A window's own coordinates are no strip (the review of 65eb6af).
        "Most differences are grassland -> shrubland; the highest-ranked window "
        "is at row 12, col 40.",
        "The highest-ranked window sits mostly in row 12.",
        "Most differing windows are listed below, the first at (row 12, col 40).",
        # The bulk is placed in another clause than the row, or the row holds
        # a count of them.
        "Most differences are grassland -> shrubland, while row 0 holds the "
        "top-ranked ones.",
        "Most differences are grassland -> shrubland, and row 0 has 51 of them.",
    ):
        assert check_direction(fine, run) == [], fine


def test_the_strips_a_claim_names_are_summed() -> None:
    """Round 4, B7/files run 3: "mostly at" strips that hold too few windows."""
    fact = {
        "id": "concentration",
        "grid": [5600, 14],
        "n_differing": 1570,
        "top_band_share": 0.2,
        "max_band": {"axis": "rows", "band": 1, "of_grid": "25-50%", "share": 0.4},
    }
    run = _run(_record("compare", {"facts": [fact]}))
    for flagged in (
        "1,570 windows flip (mostly at rows ~344-345, rows 1134-1141 and rows 1442-1449).",
        # "1,570 of 78,028" counts all the differing windows, not a minor part.
        "1,570 of 78,028 windows flip class (mostly at rows ~344-345, rows "
        "1134-1141 and rows 1442-1449).",
    ):
        violations = check_direction(flagged, run)
        assert [v["text"] for v in violations] == [flagged], flagged
        assert "which hold at most 252 windows of the 1570" in violations[0]["detail"]
    minor = "Most flips cluster along rows 1442-1449, 112 of the 1,570."
    assert check_direction(minor, run) == []


# --- direction: magnitude and unused labels -----------------------------------


_RATIO = {
    "id": "margin_ratio",
    "listed_low": 13.0,
    "listed_high": 41.0,
    "review_set_low": 4.5,
    "review_set_high": 41.0,
    "versus": "median",
}


def test_a_magnitude_outside_the_margin_ratio_is_flagged() -> None:
    """Round 6, B8/cluster run 1: "3-4 orders of magnitude"; the ratio is 13-41."""
    run = _run(_record("review_set", {"facts": [_RATIO]}))
    flagged = "i.e. roughly 3–4 orders of magnitude more uncertain than typical windows"
    assert _flagged(check_direction, flagged, run) == [flagged]
    assert _flagged(check_direction, "These are 100x less confident.", run)
    assert _flagged(
        check_direction, "Their margins are 2x smaller than the median margin.", run
    )
    for fine in (
        "These are about 20x less confident than the median window.",
        "An order of magnitude below the median margin.",
        # Within the review set's range (4.5), though below the listed one's.
        "The review set is about 5x less confident than the median window.",
        "A review set overstates errors 1.8-5.8x.",  # not a margin ratio
    ):
        assert check_direction(fine, run) == [], fine


@pytest.mark.parametrize(
    "sentence",
    [
        # A confidence word and a magnitude that is not a ratio of confidence.
        "These windows are less confident, and a review set overstates errors 1.8-5.8x.",
        "Margins are low here; label 3 times as many windows as you plan.",
        "The median margin is 4.47, and 200 times as many windows sit elsewhere.",
    ],
)
def test_a_magnitude_that_is_no_ratio_of_confidence_is_not_read(sentence: str) -> None:
    run = _run(_record("review_set", {"facts": [_RATIO]}))
    assert check_direction(sentence, run) == [], sentence


def test_a_count_of_unused_labels_other_than_the_tools_is_flagged() -> None:
    fact = {"id": "unused_labels", "n": 127, "requested": 300, "planned": 173}
    run = _run(_record("plan", {"facts": [fact]}))
    for flagged in (
        "All 300 labels are planned.",
        "100 labels are left over.",
        "The remaining 150 labels have nowhere to go.",
    ):
        assert _flagged(check_direction, flagged, run) == [flagged], flagged
    for fine in (
        "Your extra 127 labels have nowhere to go.",
        "127 labels are left unused.",
        "The plan uses 173 of the 300 labels.",
    ):
        assert check_direction(fine, run) == [], fine


@pytest.mark.parametrize(
    "item",
    [
        "w{i} (row {r}), ",
        "(row {r}, cols 1-3) montane_forest vs woodland_forest [B is more "
        "confident in rows {i}-{j}, not A], ",
    ],
)
def test_a_listing_in_one_sentence_is_checked_in_linear_time(item: str) -> None:
    """A 12 KB listing written as one sentence, with every fact to check.

    Before each sentence was read once, every row of such a listing re-read
    its brackets and clause: about 2 s for the first listing.
    """
    facts = [
        _DOMINANT,
        {"id": "more_confident_side", "side": "A", "share_a": 0.6, "share_b": 0.4},
        _where(top=0.3, band=1, share=0.4),
        _RATIO,
        {"id": "unused_labels", "n": 127, "requested": 300, "planned": 173},
    ]
    run = _awf(*facts)
    parts: list[str] = []
    while sum(map(len, parts)) < 12_000:
        i = len(parts)
        parts.append(item.format(i=i, j=i + 1, r=i % 128))
    listing = "Differing windows, mostly along the north edge: " + "".join(parts)
    started = time.perf_counter()
    check_direction(listing, run)
    assert time.perf_counter() - started < 0.5
    started = time.perf_counter()
    run_checks(listing, run)
    assert time.perf_counter() - started < 0.5


def test_no_facts_no_direction_check() -> None:
    run = _run(_record("compare", {"n_differing": 10}))
    assert check_direction("Most windows flip 1 -> 0 along the north edge.", run) == []


# --- actions -------------------------------------------------------------------

_SCORES = "/runs/B8/cluster/1/workspace/scores_run_AWF_a7c40be9_p4.json"


def test_a_save_no_tool_made_is_flagged() -> None:
    """Round 6, B7/files run 3: nothing was saved."""
    run = _run(_record("compare", {"n_differing": 1570}))
    flagged = "The full differing-window list was saved; I've shown the first few."
    violations = check_actions(flagged, run)
    assert [v["text"] for v in violations] == [
        "The full differing-window list was saved;"
    ]
    assert "no tool of this run reported writing a file" in violations[0]["detail"]


def test_a_list_said_to_be_in_a_scores_file_is_flagged() -> None:
    """Round 7, B8/cluster run 1: the scores file is not the review list."""
    run = _run(
        _record("scores_from_file", {"scores_path": _SCORES}, run_dir="C1/awf"),
        _record("review_set", {"n_review": 819}, scores_path=_SCORES),
    )
    flagged = f"The full list is saved in the scores file at `{_SCORES}`."
    assert _flagged(check_actions, flagged, run) == [flagged]
    for fine in (
        f"Scores saved at `{_SCORES}`.",
        "Scores saved at `scores_run_AWF…_p4.json` for follow-up.",
        "Pass the saved scores_path to the planner.",
        "I can save the list if you want.",
        "Nothing was saved.",
    ):
        assert check_actions(fine, run) == [], fine


def test_a_file_a_tool_only_read_is_not_a_save() -> None:
    run = _run(
        _record("estimate", {"design_path": "F2/d.json"}, design_path="F2/d.json")
    )
    assert _flagged(check_actions, "The design was saved to F2/d.json.", run)
    listed = _run(_record("plan", {"labels_csv_path": "/w/design_to_label.csv"}))
    assert not check_actions("The labeling list was saved to the CSV.", listed)


def test_a_path_the_tool_reports_writing_is_written_though_it_was_given() -> None:
    """The review of 65eb6af: ``out_path`` chosen by the caller is still a write."""
    sampled = _run(
        _record(
            "negatives",
            {"out_path": "/w/negatives.geojson", "n": 40},
            out_path="/w/negatives.geojson",
        )
    )
    assert sampled.written_files == [("out_path", "/w/negatives.geojson")]
    assert (
        check_actions("The negatives were saved to negatives.geojson.", sampled) == []
    )
    folds = _run(
        _record("cv", {"folds_path": "/w/folds.json"}, output_path="/w/folds.json")
    )
    assert folds.written_files == [("folds_path", "/w/folds.json")]
    echoed = _run(
        _record("estimate", {"design_path": "F2/d.json"}, design_path="F2/d.json")
    )
    assert echoed.written_files == []


def test_the_harness_spill_file_is_a_written_file() -> None:
    """A result too large for the context is saved whole; the model is told where."""
    spill = "/w/tool_results/olmoearth_review_set_ab12cd34.json"
    run = RunEvidence(
        tools=[
            ToolRecord(
                "olmoearth_review_set",
                {"scores_path": _SCORES},
                {"ok": True, "result": {"n_review": 819}},
                spilled_to=spill,
            )
        ]
    )
    assert run.written_files == [("saved_to", spill)]
    for fine in (
        f"The full result was saved to `{spill}`.",
        "The full review list is saved at `…/olmoearth_review_set_ab12cd34.json`.",
    ):
        assert check_actions(fine, run) == [], fine
    assert _flagged(check_actions, f"The full list is saved in `{_SCORES}`.", run)


_WS = "/runs/B8/cluster/2/workspace"


def _reviewed() -> RunEvidence:
    """exp86 round 8, B8/cluster: a scores file, then a review set that lists 50
    of its 819 windows and writes only its evidence text."""
    return _run(
        _record(
            "olmoearth_scores_from_file",
            {"scores_path": f"{_WS}/scores_run_AWF_a7c40be9_p4.json"},
            run_dir="C1/awf",
        ),
        _record(
            "olmoearth_review_set",
            {
                "n_review": 819,
                "evidence_detail_path": f"{_WS}/review_set_evidence.json",
            },
            scores_path=f"{_WS}/scores_run_AWF_a7c40be9_p4.json",
        ),
    )


@pytest.mark.parametrize(
    "sentence",
    [
        # exp86 round 8, B8/cluster runs 2 and 3
        "The rest of the list and the evidence file are saved under the run's "
        "workspace (`review_set_evidence.json`).",
        "Full list (819 windows) and per-window scores are saved in the run's "
        "workspace (`review_set_evidence.json`).",
        "All 819 windows are listed in `.../workspace/review_set_evidence.json`.",
        "Full ranking: `review_set_evidence.json`.",
        # a file no tool of the run named
        "The full list is in `review_windows.csv`.",
    ],
)
def test_a_list_said_to_be_in_a_file_no_tool_names_as_a_list_is_flagged(
    sentence: str,
) -> None:
    """The evidence file is named by its key (evidence_detail_path), not by the
    "review" in its name, and a list claimed in a file needs a list's key."""
    violations = check_actions(sentence, _reviewed())
    assert [v["text"] for v in violations] == [sentence]
    assert "holds the list" in violations[0]["detail"]


def test_what_the_evidence_and_scores_files_hold_may_be_said_saved() -> None:
    for fine in (
        "The evidence file is saved as `review_set_evidence.json`.",
        "Scores are saved at `scores_run_AWF_a7c40be9_p4.json`.",
        "Per-window scores for all 16,384 windows are saved in "
        "`scores_run_AWF_a7c40be9_p4.json`.",
    ):
        assert check_actions(fine, _reviewed()) == [], fine


def test_a_list_in_the_file_a_tool_names_as_the_list_passes() -> None:
    """exp86 round 8, B3/cluster and B7/files: the differing windows are in the
    file at differing_path, and the answers said so in four ways."""
    differing = "/runs/B3/cluster/1/workspace/compare_differing_df789c0586.json"
    run = _run(_record("compare_review", {"differing_path": differing}))
    for fine in (
        "Full list of the 3,807 differing windows: "
        "`.../workspace/compare_differing_df789c0586.json`.",
        "All 3,807 differing windows are listed in "
        "`.../workspace/compare_differing_df789c0586.json`.",
        "full list in `compare_differing_df789c0586.json`",
        "All 1,570 differing windows (with margins) are saved at differing_path "
        "(`.../compare_differing_df789c0586.json`).",
        # not a claim
        "I can list the windows in `windows.csv` if you want.",
        "Compared F4/emsr279-11_s1_pre.json against F4/emsr279-11_s1_post.json "
        "over their shared windows.",
    ):
        assert check_actions(fine, run) == [], fine
    assert _flagged(check_actions, "The full list is in `other_windows.json`.", run)


_NOTE = (
    "review lists the first 50 of the 819 windows in the review set at budget "
    "0.05, in review order (not a sample of them); the full list of 819, in the "
    "same order, is in the CSV at review_list_path (rank, window_index, row, "
    "col, margin and the class or score; no coordinates); "
    "review_set_evidence.json (evidence_detail_path) holds evidence text only, "
    "no windows."
)


def _listed_short() -> RunEvidence:
    """exp86 round 8, B8/cluster, on the fixed tools: the review set writes its
    full list to a CSV, and its listing_note names the evidence file in prose."""
    return _run(
        _record(
            "olmoearth_review_set",
            {
                "n_review": 819,
                "evidence_detail_path": f"{_WS}/review_set_evidence.json",
                "review_list_path": f"{_WS}/review_list_49faf61fce.csv",
                "review_list_rows": 819,
                "listing_note": _NOTE,
            },
            scores_path=f"{_WS}/scores_run_AWF_a7c40be9_p4.json",
        ),
        _record(
            "olmoearth_compare_review",
            {
                "differing_path": f"{_WS}/compare_differing_df789c0586.json",
                "listing_order": "the 10 listed windows are the first differing "
                "windows in window order (from row 0), not a sample of them. "
                "review_set_evidence.json (evidence_detail_path) holds evidence "
                "text only, no windows",
            },
        ),
    )


def test_a_note_that_names_a_file_is_no_list_file() -> None:
    """The fix-r8 review: list_files read the tools' listing_note, a prose note
    under a key that names a listing, as a path, so the evidence file it
    mentions passed as a list. Only a string that is a path counts."""
    run = _listed_short()
    assert run.list_files == {
        f"{_WS}/review_list_49faf61fce.csv",
        f"{_WS}/compare_differing_df789c0586.json",
    }
    assert all(" " not in path for _, path in run.written_files)
    for flagged in (
        # exp86 round 8, B8/cluster runs 2 and 3
        "The rest of the list and the evidence file are saved under the run's "
        "workspace (`review_set_evidence.json`).",
        "Full list (819 windows) and per-window scores are saved in the run's "
        "workspace (`review_set_evidence.json`).",
        "The full ranked list is in `review_set_evidence.json`.",
        "All 3,807 differing windows are in `review_set_evidence.json`.",
    ):
        assert _flagged(check_actions, flagged, run) == [flagged], flagged
    for fine in (
        "The full ranked list of 819 is saved in `review_list_49faf61fce.csv`.",
        "All 3,807 differing windows are listed in `compare_differing_df789c0586.json`.",
        "The evidence file is saved as `review_set_evidence.json`.",
    ):
        assert check_actions(fine, run) == [], fine
    # A path with spaces is still a path; a note is not.
    spaced = _run(_record("export", {"out_path": "/Users/me/My Data/review list.csv"}))
    assert spaced.written_files == [("out_path", "/Users/me/My Data/review list.csv")]


def test_items_said_to_be_listed_must_be_in_the_answer_on_the_cli() -> None:
    """Round 6, B4/studio run 3: "The first 30 windows are listed above"."""
    run = _run(_record("plan", {"n_windows_listed": 50}))
    claim = "1. Fill the CSV. The first 30 windows are listed above if you need picks."
    violations = check_actions(claim, run)
    assert [v["text"] for v in violations] == [
        "The first 30 windows are listed above if you need picks."
    ]
    assert "the answer lists 0 there" in violations[0]["detail"]
    rows = "\n".join(f"- window {i}" for i in range(30))
    assert check_actions(rows + "\nThe first 30 windows are listed above.", run) == []
    table = "| a | b |\n|---|---|\n| 1 | 2 |\n\nSee the table above."
    assert check_actions(table, run) == []
    assert _flagged(check_actions, "See the table above.", run)
    # A listing the answer places in the tool's output is not the answer's.
    assert check_actions("The first 50 are listed in the tool output.", run) == []
    # On the web the tool's own listing is on screen above the answer.
    web = _run(_record("plan", {"n_windows_listed": 50}), surface="web")
    assert check_actions(claim, web) == []


def test_on_the_web_items_listed_below_or_the_first_n_are_still_checked() -> None:
    """Nothing but the answer is below it, on either surface."""
    web = _run(_record("plan", {"n_windows_listed": 50}), surface="web")
    for flagged in (
        "The windows to label are listed below.",
        "The first 10 windows are listed.",
    ):
        assert _flagged(check_actions, flagged, web) == [flagged], flagged
    rows = "\n".join(f"- window {i}" for i in range(10))
    assert check_actions("The first 10 windows are listed below.\n" + rows, web) == []
    assert check_actions(rows + "\nThe first 10 windows are listed.", web) == []


@pytest.mark.parametrize(
    "sentence",
    [
        "As shown above, B is the more confident side.",
        "See above for why the review set is not a sample.",
        "The margins shown above are the model's own.",
    ],
)
def test_a_pointer_to_prose_is_no_listing_claim(sentence: str) -> None:
    run = _run(_record("plan", {"n_windows_listed": 50}))
    assert check_actions(sentence, run) == [], sentence


# --- forbidden claims ------------------------------------------------------------


def _forbidding(claim_id: str, **arguments: Any) -> RunEvidence:
    result = {"forbidden_claims": [{"id": claim_id, "why": "the tool says so"}]}
    return _run(_record("tool", result, **arguments))


_FORBIDDEN_CASES: dict[str, tuple[list[str], list[str]]] = {
    "post_hoc_alpha": (
        [
            "A looser α (e.g. 0.15) would certify the top ~15% zone.",
            "A looser alpha (e.g. 0.15) would certify the top zone, but the guarantee "
            "covers only an alpha fixed before, so it isn't covered.",
            "Want me to re-test with a different alpha?",
            "At alpha 0.2 it would certify 90% of the map.",
        ],
        [
            "No zone is certified at alpha = 0.05.",
            "It could be certifiable at alpha 0.05 with more labels.",
            "Choosing alpha after seeing the bounds is not valid.",
            "You would need many more labels (and any looser alpha must be fixed "
            "before seeing the labels).",
            # Bonferroni's per-level alpha explains the rule in force.
            "Bonferroni splits it over 5 levels, so each level is certified at "
            "alpha 0.01.",
            "At 3 levels each is tested at α = 0.0167, which no zone passes.",
        ],
    ),
    "rule_switch_after_failure": (
        ["Or run the same test under the Bonferroni rule?"],
        [
            "Stricter rules (bonferroni) would only reject more.",
            "No rule certifies anything here.",
        ],
    ),
    "certify_from_nonrandom_design": (
        ["I can certify a zone from these labels next."],
        [
            "A certified zone needs a random design.",
            "For a certified zone, I'd plan a new random sample.",
            "To certify a zone I would need a probability sample of the map.",
            "I can plan a simple random sample so a zone can be certified next.",
        ],
    ),
    "error_rate_without_labels": (
        ["The map's error rate is about 12%.", "About 12% of windows are wrong."],
        ["An error rate needs labels.", "A review set would overstate errors 3x."],
    ),
    "error_rate_for_unthresholded_regression": (
        [
            "Then estimate each map's error rate honestly.",
            "I'll score both against it (per-window error rates, classification metrics).",
        ],
        [
            "A regression score has no error rate without a threshold.",
            "Give me a threshold and I can estimate an error rate.",
            "Agreement metrics here would be wrong to call accuracy.",
        ],
    ),
    "subset_labelling_sufficient": (
        [
            "Is 30 enough to start labeling?",
            "Labelling just the first 10 is enough.",
            # exp86 round 8, B3/studio run 3 (kept unmarked by the rewrite)
            "You name a decision threshold for each regression band → I design a "
            "targeted labeling sample from the lower-confidence windows of each "
            "map, you label them, and I compute defensible error rates per map.",
            # round 6, B4/cluster run 2; round 8, B5/files run 3
            "Note: because this is a targeted (low-confidence-first) design, the "
            "estimate stays unbiased.",
            "This confidence (targeted) design gives an honest whole-map estimate.",
            "Label the 50 most uncertain windows and I'll compute an unbiased "
            "error rate.",
            "A targeted sample of low-confidence windows gives a valid estimate of "
            "the map's error.",
            "Labelling only the review set yields an honest accuracy.",
            "An honest error rate comes from labelling the most uncertain windows.",
        ],
        [
            "Label all 173 windows in order.",
            "30 windows are not enough.",
            "A confidence-stratified design gives an honest whole-map estimate.",
            "A review set is not a sample, so it gives no honest error rate.",
            "Labelling only the least confident windows biases the estimate.",
            "The design oversamples the least confident windows and weights them, "
            "so the estimate stays unbiased.",
            # the honest estimate is the label sample's; the review set is another offer
            "I can design a label sample so N labels give an honest error estimate "
            "for one or both maps, or run a per-class review set to flag where each "
            "map looks uncertain.",
            "This review set holds errors, so its rate would overstate the map's; "
            "use olmoearth_plan_label_sample if you want an honest error estimate.",
            # round 5, B4/cluster run 2: the stratified design named in another clause
            "I designed a confidence-stratified sample: windows are ranked by "
            "margin, so the most suspect windows get labelled first while the "
            "estimate stays unbiased for the whole map.",
            # round 3, B3/cluster run 3
            "The more confident side is right only 51–70% of the time — not enough "
            "to pick a winner by confidence.",
        ],
    ),
    "winner_without_labels": (
        ["C2 is more accurate.", "Map B is the better map."],
        [
            "The more confident side is right on only about half of such windows.",
            "Neither map is right by these results alone.",
            "With labels, we could say whether C2 is more accurate.",
        ],
    ),
    "combined_statistic_across_properties": (
        ["The RMSE between them is 0.6.", "They differ by 0.59 on average."],
        [
            "Mean difference / RMSE: not meaningful here.",
            "The correlation is 0.5.",
            "Means differ by design (A 0.10, B 0.68), but they measure no gap.",
            "The mean gap (~0.58), RMSE (0.62) and 0% agreement must not be read "
            "as one map being higher.",
        ],
    ),
    "another_date_settles_it": (
        [
            "Or run a third date and look at the trajectory.",
            "A third dated map would settle which is right.",
            "Label a sample against a reference for each year, or run a third date "
            "and look at the trajectory.",
        ],
        [
            "Another date does not settle it.",
            "Provide labels for each date.",
            "What would settle it: references (photointerpretation, or another "
            "acquisition per date) to grade each map.",
            # Map A and map B, by their dates.
            "Labels would show which is right: check the later map against 2023 "
            "imagery and the earlier map against 2022 imagery.",
            "A reference dated to either the earlier image or the later image "
            "would settle it.",
        ],
    ),
    # exp86 round 8 (B3/studio): r = -0.02 over 25 cells, 95% interval -0.41
    # to 0.38; a 12 x 12 grid of the same pair gives 0.50.
    "spatial_pattern_from_one_correlation": (
        [
            "Their correlation is -0.017, i.e. essentially zero — the two maps do "
            "not rise and fall together anywhere in the sampled region.",
            "The two models do not agree spatially at all: one is high where the "
            "other is indifferent, and vice versa.",
            "A moderate degree of overlap, meaning large parts of the AOI show "
            "similar relative patterns while much of the rest disagrees.",
            "They disagree everywhere.",
            "Nowhere do the two maps move together.",
            "Where KarstBinary is high, KarstNumber tends to be low.",
            "The two maps agree in some regions and disagree in others.",
            "They co-vary locally but not overall.",
            "High values of one coincide with low values of the other.",
            "The maps diverge most in the north.",
            # round 1, B3/studio run 2: the comparison named in the label
            "**Where they differ:** effectively everywhere sampleable.",
        ],
        [
            # one map's level, not the two maps together (the fix-r8 review)
            "KarstBinary is low almost everywhere (mean 0.096).",
            "KarstNumber sits high in most parts of the AOI.",
            "The correlation is -0.017, essentially zero over 25 cells.",
            "A single correlation cannot say where the maps agree or differ.",
            "The correlation pools all 25 cells, so it says nothing about where "
            "they differ.",
            "If you want to see the divergence spatially, I can turn both into "
            "layer packs so you can overlay them.",
            # a breakdown a tool computed by place
            "Spatially, differences cluster in the middle rows: rows 2800-4199 "
            "hold 46.8% of the differing windows.",
            'Saying they "agree nowhere" would be wrong to claim from one number.',
            # round 4, B3/cluster run 3: a share of the differences
            "A large part of the disagreement sits right at class edges.",
            # round 1, B7/files run 2
            "Any differing window has two explanations: it changed on the ground "
            "(this is likely a large part of the 2%), or one map is wrong.",
            # round 3, B3/studio run 1: how much, not where
            "The two maps agree spatially only moderately (r = 0.497).",
        ],
    ),
    "agreement_from_uncertain_correlation": (
        [
            "Only whether they rise and fall together is comparable — and with "
            "correlation ≈ -0.0172, they effectively do not agree spatially: one "
            "going high says essentially nothing about the other.",
            "Across those the correlation is −0.0172, i.e. the two surfaces "
            "essentially do not rise and fall together.",
            "The two maps show a correlation of -0.0172 — essentially no "
            "relationship, i.e. their values rise and fall independently.",
            "The near-zero correlation suggests they're capturing different aspects "
            "of karst.",
            "The two maps are unrelated.",
            "The maps are essentially independent.",
            "KarstBinary and KarstNumber do not co-vary.",
            "The maps strongly agree.",
            "They're uncorrelated.",
            "They are not correlated.",
            "One tells you nothing about the other.",
            "The two maps disagree completely.",
            "The two predictions have nothing in common.",
            # round 3, B3/studio run 2
            "They barely agree spatially: correlation -0.017 across the 25 cells.",
            # the interval stated beside the claim does not withdraw it
            "With its 95% interval of -0.41 to 0.38, the maps do not co-vary.",
            # beside the interval, not what it holds (the fix-r8 review)
            "The 95% interval spans zero and the maps are unrelated.",
        ],
        [
            # the tools' own correlation fact and must_state (the fix-r8 review)
            "The correlation is -0.0172 over 25 cells, 95% interval -0.41 to 0.38 "
            "(Fisher's z): the interval holds both no relation and a moderate one, "
            "so this sample cannot say whether the maps co-vary; a correlation says "
            "nothing about where the maps agree or differ.",
            "Between results a1 and b1, the correlation is 0.1 over 25 cells, 95% "
            "interval -0.31 to 0.48 (Fisher's z): the interval holds 0 (no "
            "relation), so this sample cannot say whether the maps co-vary;",
            "The correlation is 0.62 over 144 cells, 95% interval 0.51 to 0.71 "
            "(Fisher's z): the maps' values tend to rise and fall together across "
            "the sampled cells; a correlation says nothing about where the maps "
            "agree or differ.",
            "One correlation says nothing about where two maps agree or differ, and "
            "one whose 95% interval spans zero cannot say whether they co-vary.",
            # rewordings of them
            "The 95% interval, -0.41 to 0.38, includes both no relationship and a "
            "moderate one.",
            "It is unclear whether the two maps co-vary: 25 cells are too few.",
            "The correlation is -0.017, essentially zero over 25 cells.",
            "With 25 cells the interval runs from -0.41 to 0.38, so it cannot say "
            "whether they co-vary.",
            "Only whether they rise and fall together is meaningful between them.",
            # round 3, B3/studio run 1; round 5, B3/studio run 1
            "Only how they move together is meaningful.",
            "The comparison only reports how much they rise and fall together.",
            "They measure different quantities, so no difference or RMSE is "
            "computed.",
            # a map against its reference is a labelling instruction
            "Set wrong = 1 where the map's class does not agree with the ground "
            "truth, else 0.",
            "Each map is scored against the labels independent of the other.",
        ],
    ),
    "review_set_for_unthresholded_regression": (
        [
            # a threshold mentioned, not required (the fix-r8 review)
            "Without a threshold, I can still build a review set of each map's "
            "lowest-margin windows.",
            "There is no decision threshold for KarstNumber, but I can rank its "
            "windows by margin for review.",
            "If you want, I can next design a label sample so N labels give an "
            "honest error estimate for one or both maps, or run a per-class review "
            "set to flag where each map looks uncertain.",
            "Run olmoearth_review_set_from_result on each result to flag its own "
            "most-ambiguous (lowest-margin) windows — where each model is likely "
            "wrong.",
            "Alternatively, I can build a human review set (lowest-confidence "
            "windows first) for either map.",
            "I can rank each map's windows by margin so you review the least "
            "certain first.",
            "Next, flag the windows closest to the decision boundary for review.",
            "Want me to build a review list of the most ambiguous cells in each map?",
        ],
        [
            "Once a threshold is named, I can build a review set for each map.",
            "Name a threshold first, and then I can rank the windows by margin.",
            "The review set needs a threshold for this band.",
            "Name a threshold and I can build a review set for each map.",
            "No review set applies to a regression band without a threshold.",
            "A regression value has no margin, so a review set does not apply here.",
            # reports of a review set that ran, a statement, a table row
            "The review set holds 819 windows (margins 0.108 to 0.994).",
            "A targeted review set would overstate the true error rate.",
            "The margin should not be read as a reason to open a window first.",
            "| Rank | Window (row, col) | Predicted class | Margin |",
        ],
    ),
    "one_reference_settles_two_dates": (
        [
            # exp86 round 8, B3/cluster runs 1 and 2, B7/files run 1
            "If you have reference labels for either date (ideally one set per "
            "year, with their dates), I can grade which map is right at the "
            "differing windows.",
            "If you have labels for either date, pass them (e.g. as labels_date) "
            "and I can grade each map against its own reference to separate true "
            "change from errors.",
            "To settle it you'd need expert-labeled references for both periods (or "
            "for at least one, plus a date-matched second inference).",
            # round 7, B7/files run 1: a hedge far along the clause is no negation
            "To settle it, provide labels for one or both dates, or treat this as a "
            "change-detection layer rather than a contest between the two maps.",
            "Provide reference labels dated to one of the two years for a verdict.",
            "Labels from just one of the dates would tell us which map is right.",
            "Labels for one date would settle which map is right.",
            "One reference and a second model run would be enough to decide.",
        ],
        [
            # the tools' must_state and which_side_is_right, and rewordings
            # (the fix-r8 review)
            "Maps of different times may differ by real change; labels for one "
            "date grade only that date's map; each map needs its own date's "
            "reference.",
            "Not graded: the maps describe different times, so a window where they "
            "differ may have changed on the ground, and each map needs a reference "
            "of its own date; labels dated 2023-06-01 grade only a map of that date.",
            "Labels for one date can grade only the map of that date.",
            "A reference for one date only grades that date's map, so each map "
            "needs its own.",
            "Only a reference dated to each map would separate them.",
            "Labels dated to each map's date would settle it.",
            "You'd need references for both dates to say which is right.",
            "Labels for one date cannot settle which map is right.",
            # round 2, B7/files run 1
            "No labels_date reference was given, so even with labels for only one "
            "date, grading would be blocked.",
            "Labels for one date would measure which map matches the ground at that "
            "date, counting the other wrong wherever the ground changed.",
        ],
    ),
    "evidence_outside_its_scope": (
        [
            # exp86 round 8, B3/cluster run 3; round 7, B3/cluster run 1
            "The 2022 map has a larger logit margin on 57.8% of differing windows, "
            "but upstream evidence shows the more confident side was only right "
            "51–70% of the time in comparable cases.",
            "Upstream evidence shows the more confident side is right on only about "
            "half or less of such windows.",
            "Measured evidence behind this rule: on Ai2's published suite the "
            "model's own margin beat every no-model control on all 24 scored tasks.",
            "Upstream results carry over to this pair.",
            "The margin's lead on Ai2's suite means it is the right ranker here.",
            "Published results for cases like this show confidence is a weak guide.",
        ],
        [
            # "measurement" alone is the run's (the fix-r8 review)
            "Why these: the tool ranks by margin, the measurement of how undecided "
            "the model is on each window.",
            "No recorded experiment grades this case.",
            "No recorded experiment grades which of these two maps is right.",
            "Upstream, the more confident side was right on 51-70% of differing "
            "windows on flood maps, which does not cover this pair.",
            "The evidence covers this case only in part.",
            # the run's own evidence, and a measure of this run
            "Why these: a near-zero margin means the pixel-level evidence can't "
            "separate the two classes.",
            "Why this ordering: here the margin is measured by the mean top-1 "
            "minus top-2 logit.",
        ],
    ),
    "certification_guaranteed": (
        [
            # exp86 round 8, B5/files run 2
            "Happy to plan that if you want a guaranteed-certifiable region.",
            "A random design of 900 labels will certify the top 25%.",
            "With enough random labels you are guaranteed a certified zone.",
            "This plan ensures the top zone can be certified.",
            "300 more random labels are sufficient to certify the high-confidence "
            "half.",
            "Planning a random design means you will get a certified region.",
            # a bound beside the promise is no bound of the test (the fix-r8 review)
            "A random design guarantees you a certified zone, and the guarantee "
            "holds only at alpha 0.05.",
        ],
        [
            # the test's own guarantee (the fix-r8 review)
            "The random-design test guarantees that the certified zone's error is "
            "at most alpha, except with probability delta.",
            "For a certified zone, olmoearth_certify_zone with the same design_path "
            "and labels, at an alpha (and delta and rule) fixed now, before any "
            "label is seen: the guarantee covers only that alpha.",
            "Certification needs a random design.",
            "A random design makes certification possible, never certain.",
            "With a random design, a zone could be certified if its errors are few.",
            "No design guarantees a certified zone.",
            "The guarantee holds only for an alpha fixed before the labels are seen.",
            "To get a certified zone you would need a new random-design label plan "
            "fixed in advance.",
            "Want me to size up how many labels a random design would need to "
            "certify the top 25% of the map?",
            # round 3, B6/files run 1
            "Want me to estimate how many additional labels would plausibly certify "
            "a 15–20% zone?",
            # another alpha, not a design (post_hoc_alpha reads it)
            "A looser alpha (e.g. 0.15) would certify the top zone, but the "
            "guarantee covers only an alpha fixed before seeing the labels.",
        ],
    ),
}


def test_every_forbidden_id_has_a_detector_and_cases() -> None:
    assert set(FORBIDDEN_DETECTORS) == set(_FORBIDDEN_CASES)


@pytest.mark.parametrize("claim_id", sorted(_FORBIDDEN_CASES))
def test_a_forbidden_claim_is_flagged_and_its_negation_is_not(claim_id: str) -> None:
    run = _forbidding(claim_id, alpha=0.05)
    flagged, fine = _FORBIDDEN_CASES[claim_id]
    for sentence in flagged:
        violations = check_forbidden_claims(sentence, run)
        assert [v["text"] for v in violations] == [sentence], sentence
        assert violations[0]["detail"] == f"{claim_id} (tool): the tool says so"
    for sentence in fine:
        assert check_forbidden_claims(sentence, run) == [], sentence


def test_the_alpha_the_user_asked_for_is_not_post_hoc() -> None:
    result = {"forbidden_claims": [{"id": "post_hoc_alpha", "why": "x"}]}
    run = RunEvidence(
        tools=[_record("certify", result, alpha=0.05)],
        user_messages=["Certify a zone at alpha 0.1 if one qualifies."],
    )
    assert check_forbidden_claims("At alpha 0.1 no zone would certify.", run) == []
    assert check_forbidden_claims("At alpha 0.05 no zone certifies.", run) == []
    # Per level under Bonferroni, of the alpha the user asked for.
    assert (
        check_forbidden_claims("Each of 4 levels is certified at alpha 0.025.", run)
        == []
    )
    assert _flagged(check_forbidden_claims, "At alpha 0.2 it would certify.", run)
    assert _flagged(check_forbidden_claims, "At alpha 0.03 it would certify.", run)


def test_a_looser_alpha_is_flagged_though_written_with_one_decimal() -> None:
    """exp87 re-review: a half-unit tolerance on "0.1" (0.05) let the canonical
    looser-alpha offer through when the tool ran at 0.05."""
    result = {"forbidden_claims": [{"id": "post_hoc_alpha", "why": "x"}]}
    run = RunEvidence(tools=[_record("certify", result, alpha=0.05)], user_messages=[])
    for sentence in (
        "Want me to rerun at alpha 0.1?",
        "Want me to re-test at \u03b1 = 0.1?",
        "At alpha 0.1 the top zone would certify.",
    ):
        assert _flagged(check_forbidden_claims, sentence, run) == [sentence], sentence
    assert check_forbidden_claims("At alpha 0.05 no zone certifies.", run) == []


@pytest.mark.parametrize(
    "sentence",
    [
        "Both maps are dominated by montane_forest and woodland_forest, which change "
        "little.",
        "Most windows change little, and both maps are dominated by montane_forest "
        "and woodland_forest.",
        "The main classes are montane_forest and woodland_forest in both maps, and "
        "few windows change.",
        "The largest classes in map A are montane_forest and woodland_forest, and in "
        "map B they switch order.",
    ],
)
def test_composition_beside_a_change_word_is_no_dominant_change_claim(
    sentence: str,
) -> None:
    assert check_direction(sentence, _awf(_DOMINANT)) == [], sentence


def test_a_quantity_after_around_or_near_is_no_place() -> None:
    """exp87 re-review: "around 60%" and "near the median" read as regions and
    exempted wrong claims."""
    flagged = "B is usually more confident, around 60% of the time."
    assert _flagged(check_direction, flagged, _sides()) == [flagged]
    for sentence in (
        "Most differing windows flip class 1 -> 0, around 1,247 of them.",
        "Most flips go water -> not water, near the median margin.",
    ):
        assert _flagged(check_direction, sentence, _flips()) == [sentence], sentence
    fine = "A is more confident near the class boundaries."
    assert check_direction(fine, _sides()) == []


_ROUND_8_IDS = (
    "spatial_pattern_from_one_correlation",
    "agreement_from_uncertain_correlation",
    "review_set_for_unthresholded_regression",
    "one_reference_settles_two_dates",
    "evidence_outside_its_scope",
    "certification_guaranteed",
)


@pytest.mark.parametrize(
    "sentence",
    [
        "The correlation is -0.017, essentially zero over 25 cells.",
        "Only a reference dated to each map would separate them.",
        "Labels dated to each map's date would settle it.",
        "Certification needs a random design.",
        "No recorded experiment grades this case.",
        "The review set needs a threshold for this band.",
    ],
)
def test_the_rules_stated_correctly_pass_every_round_8_detector(sentence: str) -> None:
    """The correct statements of the exp86 round 8 fixes, under all six ids at once."""
    result = {"forbidden_claims": [{"id": i, "why": "x"} for i in _ROUND_8_IDS]}
    assert check_forbidden_claims(sentence, _run(_record("tool", result))) == []


def _dated(**result: Any) -> RunEvidence:
    """A comparison of a 2023 and a 2022 map (exp86 B3/cluster)."""
    dates = {"a": "2023-01-01/2023-12-31", "b": "2022-01-01/2022-12-31"}
    forbidden = [{"id": "one_reference_settles_two_dates", "why": "x"}]
    return _run(
        _record(
            "compare_review", {"dates": dates, "forbidden_claims": forbidden, **result}
        )
    )


def test_a_year_of_one_map_named_alone_is_a_reference_for_one_date() -> None:
    """Read against the comparison's own dates: 2023 is one map's year."""
    flagged = "A reference for 2023 alone would settle which map is right."
    assert _flagged(check_forbidden_claims, flagged, _dated()) == [flagged]
    for fine in (
        "References for 2023 and 2022 would settle which map is right.",
        "Labels for each map's year (2023 for C1) would settle which is right.",
        # the rule, in a year's words (the fix-r8 review)
        "Labels dated 2023 would grade the 2023 map only, so they cannot say "
        "which map is right.",
        # a year of neither map is another_date_settles_it's, not this id's
        "A reference for 2019 would settle which map is right.",
    ):
        assert check_forbidden_claims(fine, _dated()) == [], fine
    # without the comparison's dates a bare year is not read
    undated = _forbidding("one_reference_settles_two_dates")
    assert check_forbidden_claims(flagged, undated) == []


def test_a_name_only_the_evidence_holds_is_the_evidence_put_in_the_case() -> None:
    """exp86 round 8, B7/files run 2: the pair called "the two Sen1Floods11 flood
    maps", a name only the tool's evidence_scope held."""
    scope = (
        "On 15 pairs of Sen1Floods11 flood maps with expert labels (exp58) the "
        "more confident side was right on 51 to 70 percent of differing windows, "
        "which does not cover this pair."
    )
    result = {
        "evidence_scope": scope,
        "facts": [{"id": "more_confident_side", "sentence": "... (exp58)."}],
        "forbidden_claims": [{"id": "evidence_outside_its_scope", "why": "x"}],
    }
    brief = "Map A (F4/emsr279-11_s1_pre.json) and map B: where do they differ?"
    run = _run(_record("compare_review", result), brief=brief)
    flagged = (
        "Compared all 78,028 shared windows between the two Sen1Floods11 flood maps."
    )
    assert _flagged(check_forbidden_claims, flagged, run) == [flagged]
    for fine in (
        # the evidence cited with its scope, and a name the case holds too
        "In the closest study (exp58, 15 Sen1Floods11 pairs) the more confident "
        "side was right on 51-70%, and that study does not cover this pair.",
        "The confident side is right on 51-70% upstream (exp58).",
        "Compared all 78,028 shared windows of the emsr279 maps.",
    ):
        assert check_forbidden_claims(fine, run) == [], fine
    named = _run(_record("compare_review", result), brief="Two Sen1Floods11 chips.")
    assert check_forbidden_claims(flagged, named) == []


def _correlations(*facts: tuple[str, float, str]) -> RunEvidence:
    """A group of Studio results, each pair's correlation a fact, one uncertain."""
    return _run(
        _record(
            "olmoearth_compare_results",
            {
                "facts": [
                    {
                        "id": "correlation",
                        "r": r,
                        "co_varies": sign,
                        "result_id_a": a,
                        "result_id_b": "b1",
                        "sentence": "...",
                    }
                    for a, r, sign in facts
                ],
                "forbidden_claims": [
                    {"id": "agreement_from_uncertain_correlation", "why": "x"}
                ],
            },
        )
    )


def test_a_correlation_stated_with_the_sign_the_tool_found_is_no_claim() -> None:
    """The fix-r8 review: in a group with one uncertain pair, the certain
    pair's own reading ("the maps' values tend to rise and fall together")
    was flagged. A correlation stated with an interval that excludes 0, or
    with the r of a pair whose interval does, is the tool's reading."""
    run = _correlations(("a1", 0.62, "positive"), ("c1", 0.05, "unknown"))
    for fine in (
        "For a1 and b1, the maps' values rise and fall together (r = 0.62).",
        "a1 and b1: the maps move together, 95% interval 0.51 to 0.71.",
    ):
        assert check_forbidden_claims(fine, run) == [], fine
    for flagged in (
        "For c1 and b1, the maps' values rise and fall together (r = 0.05).",
        "The maps c1 and b1 are unrelated.",
        # an interval that spans 0 withdraws nothing
        "c1 and b1 do not co-vary (95% interval -0.21 to 0.30).",
    ):
        assert _flagged(check_forbidden_claims, flagged, run) == [flagged], flagged


def _karst(threshold_claim: bool = True) -> RunEvidence:
    """exp86 round 8, B3/studio: KarstBinary (sample_karst_score, [0, 1]) and
    KarstNumber (sample_number, 0.2 to 1.2, no threshold)."""
    why = (
        "'sample_number' (declared range [0.2, 1.2]): a margin-based review set "
        "(olmoearth_review_set_from_result) ranks a band's windows by their "
        "distance from a decision threshold, and this band has none"
    )
    return _run(
        _record(
            "olmoearth_get_prediction_result",
            {
                "prediction_name": "KarstBinary--01-01-2025--12-31-2025--PA regular",
                "property_names": ["sample_karst_score"],
            },
        ),
        _record(
            "olmoearth_get_prediction_result",
            {
                "prediction_name": "KarstNumber--01-01-2025--12-31-2025--PA regular",
                "property_names": ["sample_number"],
            },
        ),
        _record(
            "olmoearth_compare_results",
            {
                "property_a": {"property_name": "sample_karst_score"},
                "property_b": {"property_name": "sample_number"},
                "forbidden_claims": [
                    {"id": "review_set_for_unthresholded_regression", "why": why}
                ],
            },
        ),
    )


def test_a_review_set_is_forbidden_only_for_the_band_the_reason_names() -> None:
    """The fix-r8 review: the id is emitted for KarstNumber (no threshold), and
    an offer for KarstBinary, a [0, 1] score the tools decide at 0.5, was
    flagged. An offer that names only another band of the run passes; one
    that names the band, or no band, is read."""
    run = _karst()
    for fine in (
        "For KarstBinary, a 0-1 score, I can build a review set of the windows "
        "nearest 0.5.",
        "I can build a review set of KarstBinary's lowest-margin windows.",
        "For sample_karst_score I can list the lowest-margin windows for review.",
    ):
        assert check_forbidden_claims(fine, run) == [], fine
    for flagged in (
        "For KarstNumber I can build a review set of its lowest-margin windows.",
        "I can build a review set of each map's lowest-margin windows.",
        "I can build a review set for KarstBinary and KarstNumber.",
        # another band named, but the offer is for every map
        "KarstBinary is a 0-1 score, and I can build a review set of each map's "
        "lowest-margin windows.",
    ):
        assert _flagged(check_forbidden_claims, flagged, run) == [flagged], flagged


@pytest.mark.parametrize(
    "item",
    [
        # exp86 round 8, B8/cluster: a review listing in one clause, no offer
        "w{i} (row {r}, col {c}) margin 0.{i:03d}, ",
        # each cue in one long clause, which a detector reads per match
        "w{i} agrees everywhere in places row {r}, ",
        "w{i} targeted honest estimate from the review set {r}, ",
        "w{i} for either date labels settle which map, guaranteed certifiable "
        "random design, upstream such windows, they do not co-vary, ",
    ],
)
def test_a_long_listing_is_checked_in_linear_time_with_every_claim_forbidden(
    item: str,
) -> None:
    """The fix-r8 review: the review-set detector counted quotes from the
    sentence's start and searched its whole clause once per "margin", 14.7 s
    for one 130 KB listing line; the spatial one sliced and searched the
    clause once per place (about a minute each on these lines). Every
    forbidden id of the contract is emitted, and all the detectors read the
    line within a second."""
    ids = list(FORBIDDEN_DETECTORS)
    record = _record(
        "tool",
        {
            "forbidden_claims": [
                {"id": i, "why": "'sample_number' (declared range [0.2, 1.2]): x"}
                for i in ids
            ],
            "dates": {"a": "2023-01-01/2023-12-31", "b": "2022-01-01/2022-12-31"},
            "property_a": {"property_name": "sample_karst_score"},
        },
    )
    run = _run(record)
    parts: list[str] = []
    while sum(map(len, parts)) < 130_000:
        i = len(parts)
        parts.append(item.format(i=i, r=i % 128, c=i % 97))
    listing = "Windows " + "".join(parts)
    started = time.perf_counter()
    for detector in FORBIDDEN_DETECTORS.values():
        detector(listing, record, run)
    assert time.perf_counter() - started < 1.0
    started = time.perf_counter()
    run_checks(listing, run)
    assert time.perf_counter() - started < 1.0


def test_a_claim_no_tool_forbade_is_not_checked() -> None:
    run = _run(_record("tool", {"forbidden_claims": [{"id": "not_a_known_id"}]}))
    assert check_forbidden_claims("A looser alpha would certify it.", run) == []
    failed = ToolRecord(
        "tool",
        {},
        {"ok": False, "result": {"forbidden_claims": [{"id": "post_hoc_alpha"}]}},
    )
    assert (
        check_forbidden_claims("A looser alpha would certify it.", _run(failed)) == []
    )


# --- must_state -------------------------------------------------------------------

_DATES = (
    "The maps describe different dates, so a window where they differ may have "
    "changed on the ground."
)


def _stating() -> RunEvidence:
    result = {"n_differing": 3807, "must_state": [_DATES]}
    return _run(_record("olmoearth_compare_review", result))


def test_a_required_statement_the_answer_lacks_is_flagged() -> None:
    answer = "3,807 windows differ. Map B is more confident."
    violations = check_must_state(answer, _stating())
    assert [v["text"] for v in violations] == [_DATES]
    assert violations[0]["detail"].startswith("olmoearth_compare_review requires it")


def test_a_required_statement_in_other_words_passes() -> None:
    answer = (
        "3,807 windows differ. The two maps describe different dates, so any "
        "difference may be a real change on the ground or an error."
    )
    assert check_must_state(answer, _stating()) == []


def test_only_the_contracts_must_state_sentences_are_read(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """At most three per result, of at most 25 words; scope detail is not one."""
    long = (
        "The review set covers every window whose margin is below the cut, drawn "
        "from the scores file of the run, in the order of the margins, and it "
        "excludes the windows the Studio result did not sample."
    )
    short = [f"Limit {w} applies to this result." for w in ("alpha", "beta", "gamma")]
    result = {"n_review": 819, "must_state": [long, *short, "Limit delta applies."]}
    run = _run(_record("olmoearth_review_set", result))
    texts = [v["text"] for v in check_must_state("819 windows to review.", run)]
    assert texts == short
    assert "a must_state sentence of 37 words is not checked" in caplog.text


def test_an_answer_that_does_not_report_the_result_is_not_held_to_it() -> None:
    assert check_must_state("I could not compare the maps.", _stating()) == []


# --- all of them -----------------------------------------------------------------


def test_run_checks_keeps_only_the_checks_that_fired_in_order() -> None:
    run = _forbidding("post_hoc_alpha", alpha=0.05)
    answer = "It certifies 5,565 windows. A looser alpha would certify it."
    found = run_checks(answer, run)
    assert list(found) == [NUMBERS, FORBIDDEN]
    assert list(CHECKS) == [NUMBERS, DIRECTION, ACTIONS, FORBIDDEN, MUST_STATE]
    assert run_checks(answer, run, [FORBIDDEN]).keys() == {FORBIDDEN}


def test_a_check_that_fails_is_skipped_not_raised(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def broken(answer: str, run: RunEvidence) -> list[Any]:
        raise RuntimeError("a checker bug")

    monkeypatch.setitem(CHECKS, DIRECTION, broken)
    run = _run(_record("summary", {"n": 16384}))
    assert run_checks("About 5,565 windows.", run) == {
        NUMBERS: [{"check": NUMBERS, "text": "About 5,565 windows.", "detail": "5,565"}]
    }
    assert "answer check 'direction' failed" in caplog.text


def test_a_url_in_a_result_is_not_a_written_file() -> None:
    run = _run(_record("fetch", {"tiles": "https://tiles.example.org/{z}/{x}/{y}.png"}))
    assert run.written_files == []
    assert _flagged(check_actions, "The tiles were saved.", run)


def test_the_rewrite_request_lists_every_violation() -> None:
    numbers_only = {NUMBERS: [{"check": NUMBERS, "text": "x", "detail": "5,565"}]}
    assert revision_prompt(numbers_only) == grounding_prompt(["5,565"])
    found = {
        NUMBERS: [{"check": NUMBERS, "text": "About 5,565.", "detail": "5,565"}],
        DIRECTION: [{"check": DIRECTION, "text": "Most flip 1 -> 0.", "detail": "d"}],
        MUST_STATE: [{"check": MUST_STATE, "text": _DATES, "detail": "m"}],
    }
    prompt = revision_prompt(found)
    assert prompt.startswith("Harness note:")
    assert "Call no tools" in prompt
    assert "5,565" in prompt
    assert '- "Most flip 1 -> 0." (d)' in prompt
    assert _DATES in prompt


def test_marking_keeps_every_sentence_and_marks_the_flagged_ones() -> None:
    answer = (
        "Intro.\n"
        "- Most flip 1 -> 0. The list was saved.\n"
        "\n"
        "| a | b |\n"
        "|---|---|\n"
        "| 5,565 | x |\n"
    )
    found = {
        NUMBERS: [{"check": NUMBERS, "text": "| 5,565 | x |", "detail": "5,565"}],
        DIRECTION: [{"check": DIRECTION, "text": "- Most flip 1 -> 0.", "detail": ""}],
        ACTIONS: [
            {"check": ACTIONS, "text": "The list was saved.", "detail": ""},
            {"check": ACTIONS, "text": "- Most flip 1 -> 0.", "detail": ""},
        ],
        MUST_STATE: [{"check": MUST_STATE, "text": _DATES, "detail": ""}],
    }
    marked = mark_answer(answer, found)
    assert marked == (
        "Intro.\n"
        "- Most flip 1 -> 0. [unverified: direction] [unverified: actions] "
        "The list was saved. [unverified: actions]\n"
        "\n"
        "| a | b |\n"
        "|---|---|\n"
        "| 5,565 | x [unverified: numbers] |\n"
        "\n"
        f"Note from the tool: {_DATES}"
    )
    assert mark_answer(answer, {}) == answer
