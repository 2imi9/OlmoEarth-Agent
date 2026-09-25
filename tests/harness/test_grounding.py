# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""The number check's reader (harness/grounding.py).

The cases are the numbers exp86's answers wrote, rounds 1 to 3: the invented
ones must be reported, and the ones a tool did return, however the answer
wrote them, must not.
"""

from __future__ import annotations

from typing import Any

import pytest

from olmoearth_agent.harness.grounding import (
    NumberPool,
    tokenize,
    unsupported_numbers,
)

_MINUS = "\u2212"
_ELLIPSIS = "\u2026"
_EN_DASH = "\u2013"


def _values(text: str) -> list[float]:
    return [t.value for t in tokenize(text)]


# --- reported --------------------------------------------------------------


def test_an_invented_count_is_reported_as_written() -> None:
    """Round 3, B8/cluster run 1: 5,565 is in no tool output (the truth, 7,373,
    is not either)."""
    pool = [
        {
            "n_windows": 16384,
            "listed": 819,
            "margin_summary": {"margin_at_budget_cut": 0.994, "median_margin": 4.468},
            "windows": [5800, 5979],
        }
    ]
    answer = "5,565+ windows still remain below the median but above the cut."
    assert unsupported_numbers(answer, pool) == ["5,565"]


def test_a_count_the_model_did_itself_is_reported() -> None:
    """Round 2, B6: "delta/18" counted from a list of levels."""
    levels = [{"coverage": round(0.05 * i, 2)} for i in range(1, 19)]
    assert unsupported_numbers("bonferroni uses delta/18", [{"levels": levels}]) == [
        "18"
    ]


def test_a_count_the_tool_states_supports_it() -> None:
    levels = [{"coverage": round(0.05 * i, 2)} for i in range(1, 19)]
    pool = [{"levels_tested": {"n_levels": 18, "levels": levels}}]
    assert unsupported_numbers("bonferroni uses delta/18", pool) == []


def test_a_percent_that_rounds_to_no_value_is_reported() -> None:
    assert unsupported_numbers("~23% wrong", [{"error_rate": 0.25}]) == ["23%"]


def test_a_count_off_by_one_is_reported() -> None:
    """Round 1, B3/studio run 2: "47 dropped", a wrong subtraction."""
    pool = [{"samples_a": 72, "samples_b": 71, "pairs": 25, "dropped": 46}]
    assert unsupported_numbers("47 dropped as no-data", pool) == ["47"]


def test_a_figure_from_no_output_is_reported() -> None:
    """Round 1: "51-70%" quoted from a tool description, and a threshold."""
    pool = [{"agreement": 0.74, "per_class": {"4": 0.93, "7": 0.95}}]
    answer = f"either side wins ~51{_EN_DASH}70%; classes 4 and 7 are >90%"
    assert unsupported_numbers(answer, pool) == ["51", "70%", "90%"]


def test_a_round_thousand_is_one_number_not_its_parts() -> None:
    """Round 5, B6/files run 3: "e.g. 1,000+ labels", in no tool output, passed
    as the parts 1 and 0. Only a bracketed pair, a window, splits."""
    pool = [{"n_labels": 300, "levels": [{"n_inside": 46, "n_wrong": 3}]}]
    assert unsupported_numbers("e.g. 1,000+ labels", pool) == ["1,000"]
    assert unsupported_numbers("2,000 more labels", pool) == ["2,000"]
    assert unsupported_numbers("window (1,000)", [{"row": 1, "col": 0}]) == []


def test_each_number_is_reported_once_in_order() -> None:
    assert unsupported_numbers("12 then 34 then 12", []) == ["12", "34"]


# --- supported -------------------------------------------------------------


@pytest.mark.parametrize(
    ("answer", "pool"),
    [
        # U+2212 is a minus; the value is compared at the decimals written.
        (f"correlation **{_MINUS}0.017**", [{"correlation": -0.0172}]),
        ("16,384 windows", [{"n_windows": 16384}]),
        # A window written with a thousands separator.
        ("window (24,108)", [{"row": 24, "col": 108}]),
        ("~23% wrong", [{"error_rate": 0.2298}]),
        ("23% wrong", [{"error_pct": 23.2}]),
        ("6.4M graded units", [{"n": 6412345}]),
        ("6.4M graded units", ["6,435,473 graded units"]),
        ("window 1821 first", [{"windows": [1821, 1321, 3037]}]),
        ("the 2023 map", [{"date_window": "2023-01-01/2023-12-31"}]),
        ("commit a7c40be9", [{"revision": "a7c40be91234abcd5678ef90"}]),
        ("revision a347b15", ["a347b1546ab881c92aa125400ed5acc8126394ca"]),
        (f"model 5aafb53d{_ELLIPSIS}704", ["5aafb53d-fe88-429e-a645-4f8364990704"]),
        (f"result 419c{_ELLIPSIS}", [{"id": "419c3e7f-0000-4000-8000-000000000000"}]),
        ("olmoearth-inferencex v1.3.0", ["olmoearth-inferencex>=1.3.0"]),
        # A float rounded to an integer, and an integer as a percent.
        ("~10,091 km2 of overlap", [{"shared_extent_km2": 10091.4}]),
        ("an accuracy of 85", [{"accuracy": 0.85}]),
        # A range's percent sign, and a range of percents written bare.
        (f"95% CI 32{_EN_DASH}59%", [{"low": 0.3187, "high": 0.5912, "level": 0.95}]),
        (f"[15.7{_EN_DASH}23.6]", [{"low": 0.15735, "high": 0.23625}]),
        ("p = 1.2e-5", [{"p": 1.2345e-05}]),
        # Numbers inside a tool's strings count.
        ("the cut is at 0.994", [{"evidence": "cut at margin 0.994 of 16,384"}]),
        # The brief is a source like any other.
        ("awf_namanga_2023_1042061", ["C1/awf_namanga_2023_1042061"]),
        ("128x128 grid", [{"grid": [128, 128]}]),
    ],
)
def test_a_number_a_source_supports_is_not_reported(
    answer: str, pool: list[Any]
) -> None:
    assert unsupported_numbers(answer, pool) == []


def test_small_integers_are_exempt() -> None:
    assert unsupported_numbers("3 of 5 levels; step 8", []) == []
    assert unsupported_numbers("9 levels", []) == ["9"]
    # A small percent is still a claim.
    assert unsupported_numbers("5% of the map", []) == ["5%"]


def test_numbers_in_a_url_are_not_read() -> None:
    answer = "see https://example.org/runs/12345/page?id=678 for it"
    assert unsupported_numbers(answer, []) == []


def test_the_sign_is_not_compared() -> None:
    assert unsupported_numbers(f"{_MINUS}5,565 or -5,565", [5565]) == []


def test_booleans_are_not_numbers() -> None:
    assert unsupported_numbers("100%", [{"ok": True}]) == ["100%"]


# --- the tokenizer ---------------------------------------------------------


def test_digit_runs_glued_to_letters_are_read() -> None:
    assert _values("128x128") == [128, 128]
    assert _values("a347b15") == [347, 15]
    assert _values("EMSR279-11") == [279, 11]
    assert _values("2023-01-01") == [2023, 1, 1]


def test_signs_and_forms_are_read_as_written() -> None:
    tokens = tokenize(f"{_MINUS}0.017, 16,384, ~23%, 6.4M, 1.2e-5 and 16k")
    assert [t.text for t in tokens] == [
        f"{_MINUS}0.017",
        "16,384",
        "23%",
        "6.4M",
        "1.2e-5",
        "16k",
    ]
    assert tokens[1].value == 16384 and tokens[1].parts == (16, 384)
    assert tokens[2].percent and tokens[3].scale == 1e6 and tokens[5].scale == 1e3
    assert tokens[4].exponent == -5


def test_a_bare_hex_head_is_not_an_exponent() -> None:
    """``7e91`` of a shortened hash is 7 and 91, as in the full hash."""
    assert _values("7e91") == [7, 91]
    assert unsupported_numbers("run 7e91", ["7e91ab04"]) == []


def test_a_pool_is_reusable() -> None:
    pool = NumberPool([{"n": 7373}])
    assert pool.unsupported("7,373 windows") == []
    assert pool.unsupported("5,565 windows") == ["5,565"]
