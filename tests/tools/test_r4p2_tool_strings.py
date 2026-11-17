# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""The tools' sentences the reader test of 13 Nov 2026 reworded (round 4, pass 2).

Strings only: each sentence is built by the tool from the numbers it already
had, in the audited FINAL words (A/check-6, A/check-9, A/estimate-5, B/R3,
B/R4); nothing is computed anew. The refusal of B/C4 and the next step of
B/C3 are held where those tools are tested (test_tool_capabilities,
test_review_set_from_result).

Round r4-i (14 Nov 2026, F10): the planner's sentence says a range's width as
a percentage from its low end to its high end, never in "points", which now
name Studio grid points only.
"""

from __future__ import annotations

from typing import Any

import pytest

from olmoearth_agent.analysis.output_contract import MUST_STATE_MAX_WORDS, word_count
from olmoearth_agent.tools import statistical_rules as rules
from olmoearth_agent.tools.estimation import (
    CLASS_DESIGN_BUDGET_NOTE,
    _label_budget_fact,
    _missed_must_state,
)
from olmoearth_agent.tools.review_set import _review_from_result_card

#: A planner's advice as ``plan.plan_error_rate`` gives it, no rate stated.
ADVICE: dict[str, Any] = {
    "n_population": 8372,
    "width": 0.2,
    "labels": 96,
    "probability_at_labels": 0.9512,
    "error_rate": 0.5,
    "error_rate_stated": False,
}


def test_the_plan_sentence_with_no_rate_stated_is_the_reader_tests() -> None:
    """A/check-6: before any label, the windows that hold data, the 95% range
    end to end with its probability, the package's planning figure named as
    no finding about the map, and how many windows this plan picks."""
    fact = _label_budget_fact(dict(ADVICE), "random", 96)
    # The width as r4i F10 says it: "20%", never "20 points".
    assert fact["sentence"] == (
        "Planned before any label, for a random sample of the 8372 windows that "
        "hold data: with 96 labels, the 95% range will be no wider than 20% "
        "from its low end to its high end, with probability 0.951. To plan this, "
        "the statistics package (olmoearth-inferencex) assumes a map wrong 50.0% "
        "of the time. That is a planning figure, not a finding about your map; "
        "the note below says more. This plan picks 96 windows."
    )
    assert "see rate_note" not in fact["sentence"]
    # The class design adds its note after it, as before.
    classed = _label_budget_fact(dict(ADVICE), "class", 96)["sentence"]
    assert classed == fact["sentence"] + CLASS_DESIGN_BUDGET_NOTE


def test_a_stated_rate_keeps_its_own_words() -> None:
    """A/check-6 rewrites only the branch with no rate stated; r4i F10 changes
    only the width's words in each branch."""
    stated = {**ADVICE, "error_rate": 0.2, "error_rate_stated": True}
    assert _label_budget_fact(stated, "random", 61)["sentence"] == (
        "Planned before any label by the package, for a random sample of the 8372 "
        "valid windows of a map wrong 20.0% of the time: 96 labels give a 95% "
        "interval no wider than 20% from its low end to its high end with "
        "probability 0.951; this plan draws 61."
    )
    refused = _label_budget_fact({**ADVICE, "labels": None}, "random", 0)
    assert refused["sentence"].startswith("No label budget is advised for a 95%")


#: The three branches of the ``label_budget`` fact: the planner refuses, the
#: caller states a rate, the package's planning rate (the Check card's).
BRANCHES: dict[str, dict[str, Any]] = {
    "refused": {**ADVICE, "labels": None},
    "stated": {**ADVICE, "error_rate": 0.2, "error_rate_stated": True},
    "default": dict(ADVICE),
}


@pytest.mark.parametrize("design", ["random", "class"])
@pytest.mark.parametrize("branch", sorted(BRANCHES))
def test_the_plan_sentence_says_a_width_without_points(
    branch: str, design: str
) -> None:
    """r4i F10: "points" now names Studio grid points only, so no branch of the
    planner's sentence says a range's width in points; each says it as a
    percentage from the range's low end to its high end (width 0.2 -> 20%)."""
    sentence = _label_budget_fact(dict(BRANCHES[branch]), design, 61)["sentence"]
    assert " points" not in sentence
    assert "no wider than 20% from its low end to its high end" in sentence
    assert "% from its low end to its high end" in sentence


def test_the_refusal_says_the_width_without_points() -> None:
    """r4i F10, the refusal branch word for word, with the package's reason."""
    refused = _label_budget_fact(dict(BRANCHES["refused"]), "random", 0)
    assert refused["sentence"] == (
        "No label budget is advised for a 95% interval no wider than 20% from its "
        "low end to its high end over the 8372 valid windows: the package says it "
        "advises none."
    )
    why = {**BRANCHES["refused"], "refusal": "width must be in (0, 1)."}
    assert _label_budget_fact(why, "random", 0)["sentence"] == (
        "No label budget is advised for a 95% interval no wider than 20% from its "
        "low end to its high end over the 8372 valid windows: the package says "
        "width must be in (0, 1)."
    )
    # Other widths: the number as before, the words the same.
    for width, said in ((0.1, "10%"), (0.05, "5%")):
        advice = {**BRANCHES["refused"], "width": width}
        sentence = _label_budget_fact(advice, "random", 0)["sentence"]
        assert f"no wider than {said} from its low end to its high end" in sentence
        assert " points" not in sentence


def test_the_studio_scores_sentence_is_the_reader_tests_in_25_words() -> None:
    """A/check-9, without its "Note:": raw scores in 254 steps that rank
    windows and are no probabilities, within the contract's 25 words."""
    sentence = rules.STUDIO_TOPK_SCORES_MUST_STATE
    assert sentence == (
        "Studio stores the model's raw scores (how sure it was) in 254 steps: they "
        "rank windows and are not probabilities that a class is right."
    )
    assert not sentence.startswith("Note:")
    assert word_count(sentence) == 25 <= MUST_STATE_MAX_WORDS
    assert rules.add_contract({}, must_state=[sentence])["must_state"] == [sentence]


PRODUCER = {"estimate": 0.69, "low": 0.602, "high": 0.768}


def test_a_missed_class_is_named_as_written_in_the_reader_tests_words() -> None:
    """A/estimate-5: the class as the map writes it, "class 4" for a bare code,
    twice, with its producer's interval; the 25-word guard holds."""
    assert _missed_must_state(4, PRODUCER) == (
        "Class 4: of the windows you would label class 4, the map finds only "
        "60.2% to 76.8%; it misses the rest."
    )
    assert _missed_must_state("4", PRODUCER) == _missed_must_state(4, PRODUCER)
    assert _missed_must_state("Kelp canopy", PRODUCER) == (
        "Kelp canopy: of the windows you would label Kelp canopy, the map finds "
        "only 60.2% to 76.8%; it misses the rest."
    )
    four = "Bare / sparse vegetation"
    sentence = _missed_must_state(four, PRODUCER)
    assert sentence is not None and word_count(sentence) == MUST_STATE_MAX_WORDS
    assert _missed_must_state("one two three four five", PRODUCER) is None
    for gone in ("producer's accuracy", "class's ground"):
        assert gone not in str(_missed_must_state(4, PRODUCER))


def test_a_result_review_sets_card_rows_say_the_share_checked() -> None:
    """B/R3 and B/R4: one row per budget, "If you check 1% of the windows",
    "the first k windows", "how decided" for the margin, and "(k of N;
    outlined in the picture)" only on the largest budget's row, which the
    figure outlines."""
    out = {
        "budgets": [
            {"budget": 0.01, "n_review": 1, "margin_cut": 0.297},
            {"budget": 0.05, "n_review": 4, "margin_cut": 0.554},
            {"budget": 0.1, "n_review": 7, "margin_cut": 0.745},
        ],
        "sampling": {"grid": "10x10", "n_valid": 72},
        "property_name": "sample_karst_score",
        "signal": "margin = distance from the decision",
    }
    rows = [row for row in _review_from_result_card(out) if row is not None]
    assert rows[:3] == [
        [
            "If you check 1% of the windows",
            "the first 1 window; how decided: up to 0.297",
        ],
        [
            "If you check 5% of the windows",
            "the first 4 windows; how decided: up to 0.554",
        ],
        [
            "If you check 10% of the windows",
            "the first 7 windows (7 of 72; outlined in the picture); how decided: "
            "up to 0.745",
        ],
    ]
    assert [row[0] for row in rows[3:]] == ["Sampled grid", "Property", "Ranked by"]
    # More budgets than the card lists (round 4 pass 2, finding 24): the first
    # two and the largest, the one the figure outlines, which alone says so.
    more = {
        **out,
        "budgets": [
            *out["budgets"],
            {"budget": 0.2, "n_review": 14, "margin_cut": 0.9},
        ],
    }
    shown = [row for row in _review_from_result_card(more) if row is not None]
    assert shown[:3] == [
        rows[0],
        rows[1],
        [
            "If you check 20% of the windows",
            "the first 14 windows (14 of 72; outlined in the picture); how "
            "decided: up to 0.9",
        ],
    ]
    assert [row[0] for row in shown[3:]] == ["Sampled grid", "Property", "Ranked by"]
    assert sum("outlined in the picture" in row[1] for row in shown) == 1
    # Five budgets: still three rows, the last the largest.
    five = {
        **more,
        "budgets": [
            *more["budgets"],
            {"budget": 0.3, "n_review": 21, "margin_cut": 0.95},
        ],
    }
    listed = [row[0] for row in _review_from_result_card(five) if row is not None]
    assert listed[:3] == [
        "If you check 1% of the windows",
        "If you check 5% of the windows",
        "If you check 30% of the windows",
    ]
    # The same budget twice: only the last entry, the one outlined, says so.
    twice = {
        **out,
        "budgets": [
            {"budget": 0.05, "n_review": 4, "margin_cut": 0.554},
            {"budget": 0.05, "n_review": 4, "margin_cut": 0.554},
        ],
    }
    pair = [row for row in _review_from_result_card(twice) if row is not None]
    assert ["outlined" in row[1] for row in pair[:2]] == [False, True]
    # No valid count: no "(k of N; ...)".
    bare = {**out, "sampling": None}
    plain = [row for row in _review_from_result_card(bare) if row is not None]
    assert plain[2] == [
        "If you check 10% of the windows",
        "the first 7 windows; how decided: up to 0.745",
    ]
