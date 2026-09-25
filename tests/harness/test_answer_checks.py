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
        ["Is 30 enough to start labeling?", "Labelling just the first 10 is enough."],
        ["Label all 173 windows in order.", "30 windows are not enough."],
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
