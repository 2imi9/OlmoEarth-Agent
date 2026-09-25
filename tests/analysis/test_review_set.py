# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Tests for the label-free review-set ranking (skill #18)."""

from __future__ import annotations

import random

import pytest

from olmoearth_agent.analysis.review_set import (
    BOUNDARY_NEIGHBOURS_MEANS,
    EVIDENCE,
    EVIDENCE_LIMITS,
    attainable_ceiling,
    aurc_expected,
    auroc,
    boundary_counts,
    capture_at_budget,
    compare_scores,
    detect_score_type,
    dominant_change_fact,
    evidence_detail,
    excess_aurc,
    grade_rule,
    margin_ratio_fact,
    margins,
    more_confident_fact,
    oracle_aurc,
    predicted_classes,
    ranking_evidence_scope,
    review_set,
    sign_test_p,
    spatial_breakdown,
)

# --------------------------------------------------------------------------- primitives


def test_margin_is_top1_minus_top2() -> None:
    assert margins([[3.0, 1.0, 0.5]]) == [2.0]
    assert margins([[1.0, 1.0]]) == [0.0]


def test_predicted_class_is_argmax_first_wins() -> None:
    assert predicted_classes([[0.1, 0.9], [0.5, 0.5]]) == [1, 0]


def test_score_type_detection() -> None:
    assert detect_score_type([[0.6, 0.4]]) == "probability"
    assert detect_score_type([[3.0, 1.0]]) == "logit"
    assert detect_score_type([[0.6, 0.6]]) == "logit"  # does not sum to 1


def test_boundary_counts_on_a_known_grid() -> None:
    # 3x3, centre is class 1, everything else class 0 -> centre has 8
    # differing neighbours; each corner touches only the centre.
    classes = [0, 0, 0, 0, 1, 0, 0, 0, 0]
    counts = boundary_counts(classes, 3, 3)
    assert counts[4] == 8
    assert counts[0] == 1
    assert counts[1] == 1


def test_boundary_counts_rejects_wrong_grid() -> None:
    with pytest.raises(ValueError, match="grid 2x2"):
        boundary_counts([0, 1, 0], 2, 2)


def test_uniform_grid_has_no_boundary() -> None:
    assert boundary_counts([0] * 9, 3, 3) == [0] * 9


# --------------------------------------------------------------------------- estimators


def test_perfect_ranker_has_zero_excess_aurc() -> None:
    errors = [0, 0, 1, 1]
    assert excess_aurc(errors, errors) == pytest.approx(0.0, abs=1e-12)


def test_oracle_aurc_closed_form_matches_aurc_of_errors() -> None:
    errors = [0, 1, 0, 1, 1]
    assert oracle_aurc(len(errors), 3) == pytest.approx(
        aurc_expected(errors, errors), abs=1e-12
    )


def test_anti_ranker_is_worse_than_perfect() -> None:
    errors = [0, 0, 1, 1]
    anti = [1, 1, 0, 0]
    assert excess_aurc(anti, errors) > excess_aurc(errors, errors)


def test_auroc_perfect_reversed_and_tied() -> None:
    assert auroc([0.0, 1.0], [0, 1]) == pytest.approx(1.0)
    assert auroc([1.0, 0.0], [0, 1]) == pytest.approx(0.0)
    assert auroc([0.0, 0.0], [0, 1]) == pytest.approx(0.5)


def test_auroc_undefined_without_both_classes() -> None:
    assert auroc([0.1, 0.2], [0, 0]) is None
    assert auroc([0.1, 0.2], [1, 1]) is None


def test_capture_is_bounded_by_the_attainable_ceiling() -> None:
    random.seed(3)
    n = 200
    errors = [1 if i % 5 == 0 else 0 for i in range(n)]  # 20% error rate
    signal = [random.random() for _ in range(n)]
    got = capture_at_budget(signal, errors, [0.10])[0.10]
    ceiling = attainable_ceiling(0.10, 0.2)
    assert ceiling == pytest.approx(0.5)
    assert got <= ceiling + 1e-9


def test_tie_aware_statistics_do_not_depend_on_input_order() -> None:
    # The nine-level boundary indicator ties heavily; a coarse score must not
    # be credited or penalised for raster order.
    random.seed(11)
    n = 300
    signal = [float(random.randint(0, 8)) for _ in range(n)]
    errors = [1 if random.random() < 0.25 else 0 for _ in range(n)]
    base_aurc = aurc_expected(signal, errors)
    base_cap = capture_at_budget(signal, errors, [0.1])[0.1]
    base_auc = auroc(signal, errors)
    for _ in range(5):
        idx = list(range(n))
        random.shuffle(idx)
        s = [signal[i] for i in idx]
        e = [errors[i] for i in idx]
        assert aurc_expected(s, e) == pytest.approx(base_aurc, abs=1e-12)
        assert capture_at_budget(s, e, [0.1])[0.1] == pytest.approx(base_cap, abs=1e-12)
        assert auroc(s, e) == pytest.approx(base_auc, abs=1e-12)


def test_attainable_ceiling_formula_and_validation() -> None:
    assert attainable_ceiling(0.10, 0.20) == pytest.approx(0.5)
    assert attainable_ceiling(0.50, 0.20) == pytest.approx(1.0)  # clipped
    assert attainable_ceiling(0.10, 0.0) == 0.0
    with pytest.raises(ValueError, match="budget must be"):
        attainable_ceiling(0.0, 0.2)
    with pytest.raises(ValueError, match="error_rate must be"):
        attainable_ceiling(0.1, 1.5)


def test_sign_test_matches_the_binomial_tail() -> None:
    assert sign_test_p(10, 10) == pytest.approx(1 / 1024)
    assert sign_test_p(0, 10) == pytest.approx(1.0)
    assert sign_test_p(5, 10) == pytest.approx(0.623046875)
    assert sign_test_p(3, 0) == 1.0


# --------------------------------------------------------------------------- review_set


def _planted(n: int = 40, period: int = 5) -> tuple[list[list[float]], list[int]]:
    """Scores whose margin is low exactly where the model errs."""
    scores, errors = [], []
    for i in range(n):
        bad = i % period == 0
        scores.append([0.2 if bad else 5.0, 0.1, 0.0])
        errors.append(1 if bad else 0)
    return scores, errors


def test_review_set_puts_the_low_margin_windows_first() -> None:
    scores, errors = _planted()
    out = review_set(scores, budget=0.20)
    assert out["n_review"] == 8
    picked = {row["window_index"] for row in out["review"]}
    assert all(errors[i] == 1 for i in picked)


def test_review_set_reports_ceiling_when_error_rate_given() -> None:
    scores, errors = _planted()
    out = review_set(scores, budget=0.10, error_rate=sum(errors) / len(errors))
    assert out["ceiling"]["attainable_ceiling"] == pytest.approx(0.5)


def test_review_set_states_its_evidence_in_one_scoped_sentence() -> None:
    """exp86 rounds 6 and 7: answers widened the inline evidence blocks to cases
    no experiment grades. A ranking carries one sentence instead: what was
    measured, on what, and whether it covers this case."""
    scores, _ = _planted()
    out = review_set(scores, budget=0.1)
    assert "evidence" not in out and "caveats" not in out
    scope = out["evidence_scope"]
    assert "24-task embedding suite" in scope and "window level" in scope
    assert "OlmoEarth family" in scope
    assert scope.count(". ") == 0 and scope.endswith(".")
    # Inline scores name no model, so coverage is not known and must be stated.
    assert out["evidence_covers_this_case"] == "not known"
    assert out["must_state"] == [scope]
    assert [c["id"] for c in out["forbidden_claims"]] == ["error_rate_without_labels"]


def test_the_full_evidence_text_is_kept_for_the_file() -> None:
    detail = evidence_detail()
    assert detail["ranking"]["claims"] == EVIDENCE
    assert detail["ranking"]["limits"] == list(EVIDENCE_LIMITS)
    assert "51 to 70" in detail["comparison"]["reading"]
    assert "exp76" in detail["ranking"]["multi_class"]


@pytest.mark.parametrize(
    ("kind", "n_classes", "model", "covers", "phrase"),
    [
        ("binary_score", 2, None, "no", "regression score read as a probability"),
        ("threshold_distance", 2, None, "no", "distance from a decision threshold"),
        ("logit", 2, None, "not known", "do not name the model"),
        ("logit", 2, "someone/Other-Model", "not known", "not an OlmoEarth model"),
        ("logit", 2, "allenai/OlmoEarth-v1-FT-AWF-Base", "yes", "two-class margin"),
        ("probability", 9, "allenai/OlmoEarth-v1-Base", "yes", "14 of the suite's 16"),
        ("logit", 9, "allenai/OlmoEarth-v1-FT-AWF-Base", "in part", "all 16"),
        (
            "window_confidence_probability",
            9,
            "allenai/OlmoEarth-v1-FT-AWF-Base",
            "yes",
            "top-1 probability over 9 classes",
        ),
    ],
)
def test_the_scope_sentence_says_whether_the_evidence_covers_the_case(
    kind: str, n_classes: int, model: str | None, covers: str, phrase: str
) -> None:
    scope = ranking_evidence_scope(score_kind=kind, n_classes=n_classes, model=model)
    assert scope["covers"] == covers
    assert phrase in scope["sentence"]
    assert scope["sentence"].startswith("Ai2's 24-task embedding suite")


def test_a_ranking_of_a_file_names_its_model_and_kind() -> None:
    """The scope follows what the rows are: a provider's multi-class logits are
    covered in part and stated; a two-class OlmoEarth map is covered in kind."""
    rows = [[0.0, 2.0, 0.0], [1.5, 0.0, 0.0], [0.0, 0.0, 0.3]]
    out = review_set(
        rows,
        budget=1.0,
        score_type="logit",
        score_kind="window_confidence",
        model="allenai/OlmoEarth-v1-FT-AWF-Base",
    )
    assert out["evidence_covers_this_case"] == "in part"
    assert out["must_state"] == [out["evidence_scope"]]
    two = review_set(
        [[0.2, 0.8], [0.6, 0.4]], budget=1.0, model="allenai/OlmoEarth-v1-Base"
    )
    assert two["evidence_covers_this_case"] == "yes" and two["must_state"] == []
    band = review_set([[0.4, 0.6], [0.9, 0.1]], budget=1.0, score_kind="binary_score")
    assert band["evidence_covers_this_case"] == "no"
    assert "regression score read as a probability" in band["must_state"][0]


def test_margin_ratio_is_the_median_over_the_listed_margins() -> None:
    """exp86 round 6 (brief 8): "3-4 orders of magnitude more uncertain" where the
    median margin was 13 to 41 times the listed ones. The ratio is a fact."""
    scores = [
        [1.0 + m, 1.0] for m in (0.1, 0.2, 0.4, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0)
    ]
    out = review_set(scores, budget=0.3)
    (fact,) = out["facts"]
    assert fact["id"] == "margin_ratio" and fact["versus"] == "median"
    assert (fact["low"], fact["high"]) == (10.0, 40.0)
    assert "10.00 to 40.00 times" in fact["sentence"]
    assert "3 listed windows (0.1 to 0.4)" in fact["sentence"]


def test_margin_ratio_with_a_zero_margin_has_no_upper_ratio() -> None:
    summary = {
        "n_windows": 5,
        "median_margin": 2.0,
        "listed": {"n": 2, "margin_range": [0.0, 0.5]},
    }
    fact = margin_ratio_fact(summary)
    assert fact is not None and fact["low"] == 4.0 and fact["high"] is None
    assert "no upper ratio" in fact["sentence"]
    zero = margin_ratio_fact(
        {**summary, "listed": {"n": 1, "margin_range": [0.0, 0.0]}}
    )
    assert (
        zero is not None
        and zero["low"] is None
        and "every listed margin is 0" in (zero["sentence"])
    )
    assert (
        margin_ratio_fact({**summary, "listed": {"n": 0, "margin_range": None}}) is None
    )


def test_boundary_neighbours_are_explained_where_they_are_returned() -> None:
    scores = [[5.0, 0.0] for _ in range(9)]
    scores[4] = [0.0, 5.0]
    with_grid = review_set(scores, budget=1.0, grid=(3, 3))
    assert with_grid["boundary_neighbours_means"] == BOUNDARY_NEIGHBOURS_MEANS
    assert "PREDICTED" in BOUNDARY_NEIGHBOURS_MEANS
    assert "not a measure of error" in BOUNDARY_NEIGHBOURS_MEANS
    assert "boundary_neighbours_means" not in review_set(scores, budget=1.0)


# --------------------------------------------------------------------------- comparison


def _flip(
    n: int, rows: list[int], a_class: int = 0, b_class: int = 1
) -> tuple[list[list[float]], list[list[float]]]:
    """A and B agree on class 0 except at ``rows`` (window indices)."""
    a = [[5.0, 0.0, 0.0] for _ in range(n)]
    b = [list(r) for r in a]
    for i in rows:
        a[i] = [0.0, 0.0, 0.0]
        a[i][a_class] = 3.0
        b[i] = [0.0, 0.0, 0.0]
        b[i][b_class] = 2.0
    return a, b


def test_class_pairs_give_both_directions_of_each_pair() -> None:
    """exp86 round 7: "A <-> B 37%" was read as both directions."""
    a, b = _flip(20, list(range(10)), a_class=2, b_class=1)
    a2, b2 = _flip(20, [12, 13], a_class=1, b_class=2)
    for i in (12, 13):
        a[i], b[i] = a2[i], b2[i]
    out = compare_scores(a, b)
    (pair,) = out["class_pairs"]
    assert pair["pair"] == "1<->2" and pair["classes"] == [1, 2] and pair["n"] == 12
    assert pair["directions"] == [
        {"class_a": 2, "class_b": 1, "n": 10},
        {"class_a": 1, "class_b": 2, "n": 2},
    ]
    assert out["n_class_pairs"] == 1 and out["n_class_changes"] == 2
    assert "both directions" in out["class_pairs_reading"]


def test_the_comparison_facts_are_computed_by_code() -> None:
    a, b = _flip(20, list(range(10)), a_class=2, b_class=1)
    a2, b2 = _flip(20, [12, 13], a_class=1, b_class=2)
    for i in (12, 13):
        a[i], b[i] = a2[i], b2[i]
    names = {"0": "water", "1": "forest", "2": "grass"}
    out = compare_scores(a, b, class_names=names)
    facts = {f["id"]: f for f in out["facts"]}
    dominant = facts["dominant_change"]
    assert (dominant["from_class"], dominant["to_class"]) == (2, 1)
    assert dominant["direction"] == "A->B"
    assert (dominant["n"], dominant["reverse_n"]) == (10, 2)
    assert dominant["share"] == pytest.approx(10 / 12, abs=1e-6)
    assert (
        "class 2 (grass) in map A to class 1 (forest) in map B" in dominant["sentence"]
    )
    assert "the reverse" in dominant["sentence"]
    side = facts["more_confident_side"]
    # A's margin is 3 and B's 2 on every differing window.
    assert (side["side"], side["share"]) == ("A", 1.0)
    assert "does not make A right" in side["sentence"]
    assert out["b_more_confident_share_of_differing"] == 0.0
    assert out["class_changes"][0]["class_name_a"] == "grass"
    assert "concentration" not in facts  # no grid, no spatial breakdown


def test_the_comparison_scope_and_contract() -> None:
    a, b = _flip(4, [1])
    out = compare_scores(a, b)
    assert "evidence" not in out and "caveats" not in out
    assert "51 to 70 percent" in out["evidence_scope"]
    assert out["evidence_covers_this_case"] == "no"
    assert out["must_state"] == [out["evidence_scope"]]
    assert [c["id"] for c in out["forbidden_claims"]] == ["winner_without_labels"]


def test_spatial_breakdown_counts_every_differing_window_by_band() -> None:
    """exp86: "a long strip along the north edge" for 1.3% of the differences.
    The shares per row and column band are over every differing window."""
    rows, cols = 8, 8
    # Rows 0-1 hold 2 differing windows; rows 4-5 hold 12.
    diff = [0, 1] + [r * cols + c for r in (4, 5) for c in range(6)]
    a, b = _flip(rows * cols, diff)
    out = compare_scores(a, b, grid=(rows, cols), max_listed=3)
    spatial = out["spatial"]
    assert [band["rows"] for band in spatial["row_bands"]] == [
        [0, 1],
        [2, 3],
        [4, 5],
        [6, 7],
    ]
    assert [band["n_differing"] for band in spatial["row_bands"]] == [2, 0, 12, 0]
    assert spatial["row_bands"][2]["share_of_differing"] == pytest.approx(
        12 / 14, abs=1e-6
    )
    assert spatial["row_bands"][2]["share_of_windows"] == 0.25
    assert spatial["top_band"]["where"] == "rows 4-5"
    assert spatial["top_band_share"] == spatial["row_bands"][2]["share_of_differing"]
    fact = next(f for f in out["facts"] if f["id"] == "concentration")
    assert fact["where"] == "rows 4-5" and fact["of_grid"] == "50-75%"
    assert fact["share"] == fact["top_band_share"] == spatial["top_band_share"]
    assert (
        "Rows 4-5 (50-75% of the grid's 8 rows, from row 0) hold 85.7%"
        in fact["sentence"]
    )
    assert out["n_differing_listed"] == 3  # the listing is still the first ones
    assert "boundary_means" in out and "PREDICTED" in out["boundary_means"]


def test_spatial_breakdown_places_windows_of_a_partial_grid() -> None:
    """A scores file that leaves no-data windows out: rows are placed by their
    grid index, and each band's share of the compared windows is its own."""
    out = spatial_breakdown([5, 6, 7], [0, 1, 5, 6, 7, 15], (4, 4))
    assert [b["n_windows"] for b in out["row_bands"]] == [2, 3, 0, 1]
    assert [b["n_differing"] for b in out["row_bands"]] == [0, 3, 0, 0]
    assert out["top_band"]["where"] == "row 1" and out["top_band_share"] == 1.0
    assert [b["n_differing"] for b in out["col_bands"]] == [0, 1, 1, 1]
    empty = spatial_breakdown([], [0, 1], (1, 2))
    assert empty["top_band"] is None and empty["n_bands"] == [1, 2]


def test_compare_places_rows_of_a_partial_grid() -> None:
    a, b = _flip(3, [1, 2])
    out = compare_scores(a, b, grid=(3, 3), windows=[0, 4, 8], max_listed=None)
    assert [(d["window_index"], d["row"], d["col"]) for d in out["differing"]] == [
        (4, 1, 1),
        (8, 2, 2),
    ]
    assert "boundary_share_overall" not in out  # needs every window
    assert out["spatial"]["top_band_share"] == 0.5
    with pytest.raises(ValueError, match="outside"):
        compare_scores(a, b, grid=(2, 2), windows=[0, 1, 9])
    with pytest.raises(ValueError, match="grid cells"):
        compare_scores(a, b, grid=(3, 3), windows=[0, 1])


def test_review_set_boundary_first_needs_a_grid() -> None:
    scores, _ = _planted(n=9)
    with pytest.raises(ValueError, match="needs grid"):
        review_set(scores, order="boundary_first")


def test_boundary_first_puts_boundary_windows_ahead_of_lower_margin_ones() -> None:
    # 5x5, all class 0 except the top-left corner. The boundary block is then
    # {0, 1, 5, 6}; the grid centre (12) is interior. Give the centre the
    # lowest margin in the whole grid: confidence order must open it first,
    # boundary-first must not.
    scores = [[5.0, 0.0] for _ in range(25)]
    scores[0] = [0.0, 5.0]  # corner: the other class, high margin
    scores[12] = [5.0, 4.99]  # interior: the lowest margin anywhere
    out = review_set(scores, budget=1.0, grid=(5, 5), order="boundary_first")
    order = [r["window_index"] for r in out["review"]]
    assert out["n_boundary_windows"] == 4
    assert set(order[:4]) == {0, 1, 5, 6}
    assert order[4] == 12  # first interior window, because it is lowest-margin
    conf = review_set(scores, budget=1.0, grid=(5, 5), order="confidence")
    assert [r["window_index"] for r in conf["review"]][0] == 12


def test_boundary_first_degenerates_to_confidence_when_all_windows_border() -> None:
    # Every window touching a boundary means the first sort key is constant,
    # so the order must fall back to ascending margin - not stay arbitrary.
    scores = [[5.0, 0.0] for _ in range(9)]
    scores[4] = [0.0, 5.0]  # centre differs -> all 9 windows border it
    scores[0] = [5.0, 4.9]  # lowest margin
    out = review_set(scores, budget=1.0, grid=(3, 3), order="boundary_first")
    assert out["n_boundary_windows"] == 9
    assert [r["window_index"] for r in out["review"]][0] == 0


def test_review_set_scopes_multiclass_probabilities() -> None:
    """The suite's margin is of probabilities (exp70), and on its multi-class
    tasks one minus the top probability ranked errors slightly better (exp76)."""
    out = review_set(
        [[0.5, 0.3, 0.2], [0.4, 0.4, 0.2]],
        budget=1.0,
        model="allenai/OlmoEarth-v1-Base",
    )
    assert out["score_type_detected"] == "probability"
    assert out["evidence_covers_this_case"] == "yes"
    assert "probability margin over 3 classes" in out["evidence_scope"]
    assert "exp76" in out["evidence_scope"]


def test_review_set_validates_its_inputs() -> None:
    with pytest.raises(ValueError, match=">= 2 classes"):
        review_set([[1.0]])
    with pytest.raises(ValueError, match="budget must be"):
        review_set([[1.0, 0.0]], budget=0.0)
    with pytest.raises(ValueError, match="budget must be"):
        review_set([[1.0, 0.0]], budget=1.5)
    with pytest.raises(ValueError, match="same length"):
        review_set([[1.0, 0.0], [1.0]])
    with pytest.raises(ValueError, match="ids length"):
        review_set([[1.0, 0.0]], ids=["a", "b"])
    with pytest.raises(ValueError, match="order must be"):
        review_set([[1.0, 0.0]], order="whatever")
    with pytest.raises(ValueError, match="non-finite"):
        review_set([[float("nan"), 0.0]])
    with pytest.raises(ValueError, match="is empty"):
        review_set([])


def test_review_set_always_reviews_at_least_one_window() -> None:
    scores, _ = _planted(n=10)
    out = review_set(scores, budget=0.001)
    assert out["n_review"] == 1
    assert out["realised_budget"] == pytest.approx(0.1)


def test_review_set_caps_the_inline_list_but_reports_the_full_count() -> None:
    scores, _ = _planted(n=100)
    out = review_set(scores, budget=1.0, max_listed=5)
    assert out["n_review"] == 100
    assert out["n_review_listed"] == 5


def test_review_set_emits_no_coordinates() -> None:
    scores, _ = _planted(n=9)
    out = review_set(scores, budget=1.0, grid=(3, 3), ids=[f"w{i}" for i in range(9)])
    for row in out["review"]:
        assert not {"lat", "lon", "latitude", "longitude", "geometry"} & set(row)


# --------------------------------------------------------------------------- grade_rule


def test_grade_rule_scores_a_good_candidate_against_a_useless_one() -> None:
    random.seed(5)
    n = 300
    errors = [1 if random.random() < 0.2 else 0 for _ in range(n)]
    good = [0.9 + random.random() * 0.1 if e else random.random() * 0.1 for e in errors]
    noise = [random.random() for _ in range(n)]
    out = grade_rule(good, errors, baseline=noise, control=noise)
    assert out["gradable"] is True
    assert out["verdict"]["candidate_beats_baseline_pooled"] is True
    assert out["verdict"]["candidate_beats_control_pooled"] is True
    assert out["arms"]["candidate"]["auroc"] > 0.95


def test_grade_rule_reports_the_ceiling_next_to_capture() -> None:
    errors = [1 if i % 4 == 0 else 0 for i in range(100)]
    out = grade_rule([float(e) for e in errors], errors, budgets=[0.1])
    assert out["ceiling_at_budget"]["0.1"] == pytest.approx(0.4)
    assert out["arms"]["candidate"]["capture_at_budget"]["0.1"] <= 0.4 + 1e-9


def test_grade_rule_refuses_a_degenerate_label_set() -> None:
    out = grade_rule([0.1, 0.2, 0.3], [0, 0, 0])
    assert out["gradable"] is False
    assert "both errors and non-errors" in out["reason"]


def test_grade_rule_honours_higher_is_suspect_false() -> None:
    errors = [0, 0, 1, 1]
    low_means_suspect = [1.0, 1.0, 0.0, 0.0]
    flipped = grade_rule(low_means_suspect, errors, higher_is_suspect=False)
    straight = grade_rule(low_means_suspect, errors, higher_is_suspect=True)
    assert flipped["arms"]["candidate"]["auroc"] == pytest.approx(1.0)
    assert straight["arms"]["candidate"]["auroc"] == pytest.approx(0.0)


def test_grade_rule_sign_test_uses_groups_as_the_unit_of_replication() -> None:
    # Ten scenes; the candidate wins every one.
    signal, errors, groups = [], [], []
    for g in range(10):
        for i in range(10):
            err = i < 3
            errors.append(1 if err else 0)
            signal.append(1.0 if err else 0.0)  # perfect
            groups.append(f"scene{g}")
    baseline = [1.0 - s for s in signal]  # exactly wrong
    out = grade_rule(signal, errors, baseline=baseline, groups=groups)
    pg = out["per_group"]
    assert pg["n_groups"] == 10
    assert pg["candidate_wins"] == 10
    assert pg["baseline_wins"] == 0
    assert pg["sign_test_p_one_sided"] == pytest.approx(1 / 1024, abs=1e-12)


def test_grade_rule_skips_groups_with_no_errors() -> None:
    signal = [0.9, 0.1, 0.9, 0.1]
    errors = [1, 0, 0, 0]
    groups = ["a", "a", "b", "b"]  # group b has no errors
    out = grade_rule(signal, errors, baseline=[0.1, 0.9, 0.1, 0.9], groups=groups)
    assert out["per_group"]["n_skipped_groups"] == 1
    assert out["per_group"]["n_gradable_groups"] == 1


def test_grade_rule_needs_a_baseline_for_the_sign_test() -> None:
    with pytest.raises(ValueError, match="needs a baseline"):
        grade_rule([0.1, 0.9], [0, 1], groups=["a", "b"])


def test_grade_rule_validates_lengths() -> None:
    with pytest.raises(ValueError, match="signal length"):
        grade_rule([0.1], [0, 1])
    with pytest.raises(ValueError, match="baseline length"):
        grade_rule([0.1, 0.9], [0, 1], baseline=[0.1])
    with pytest.raises(ValueError, match="control length"):
        grade_rule([0.1, 0.9], [0, 1], control=[0.1])
    with pytest.raises(ValueError, match="groups length"):
        grade_rule([0.1, 0.9], [0, 1], baseline=[0.1, 0.9], groups=["a"])


def test_a_monotone_reparametrisation_of_the_baseline_ties_exactly() -> None:
    # Guards the estimator: rescaling a signal must not change its ranking.
    random.seed(9)
    errors = [1 if random.random() < 0.3 else 0 for _ in range(120)]
    base = [random.random() for _ in range(120)]
    out = grade_rule([3.0 * b + 7.0 for b in base], errors, baseline=base)
    assert out["verdict"]["excess_aurc_delta"] == pytest.approx(0.0, abs=1e-12)


def test_ties_are_broken_as_the_companion_package_breaks_them() -> None:
    """Exact ties go to the higher window index first, as olmoearth-inferencex's
    ``assess.review_order`` does (an ascending stable sort, reversed); the agent broke
    them the other way until 2026-09-24, so a tied map gave a different review set."""
    scores = [[0.6, 0.4], [0.6, 0.4], [0.9, 0.1], [0.6, 0.4]]
    out = review_set(scores, budget=0.75, error_rate=None, max_listed=4)
    assert [w["window_index"] for w in out["review"]] == [3, 1, 0]
    np = pytest.importorskip("numpy")
    assess = pytest.importorskip("oe_inferencex.assess")
    margin = np.array([abs(r[0] - r[1]) for r in scores])
    package = [int(i) for i in assess.review_order(-margin)][:3]
    assert package == [3, 1, 0]


def test_no_side_is_named_more_confident_on_a_tie() -> None:
    assert more_confident_fact(0.5, 0.5, 10) is None
    assert more_confident_fact(None, None, 0) is None
    fact = more_confident_fact(0.3, 0.6, 10)
    assert fact is not None and (fact["side"], fact["share"]) == ("B", 0.6)
    assert "equal on 10.0%" in fact["sentence"]


def test_a_tied_dominant_change_says_so() -> None:
    changes = [
        {"class_a": 0, "class_b": 1, "n": 5, "share_of_differing": 0.5},
        {"class_a": 1, "class_b": 0, "n": 5, "share_of_differing": 0.5},
    ]
    fact = dominant_change_fact(changes, {(0, 1): 5, (1, 0): 5}, 10, None)
    assert fact is not None
    assert fact["sentence"].startswith("The most common change (tied with 1 other)")
    assert (fact["reverse_n"], fact["reverse_share"]) == (5, 0.5)
    assert dominant_change_fact([], {}, 0, None) is None
