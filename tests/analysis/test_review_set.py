# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Tests for the label-free review-set ranking (skill #18)."""

from __future__ import annotations

import random

import pytest

from olmoearth_agent.analysis.review_set import (
    attainable_ceiling,
    aurc_expected,
    auroc,
    boundary_counts,
    capture_at_budget,
    detect_score_type,
    excess_aurc,
    grade_rule,
    margins,
    oracle_aurc,
    predicted_classes,
    review_set,
    sign_test_p,
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


def test_review_set_carries_evidence_and_caveats() -> None:
    scores, _ = _planted()
    out = review_set(scores, budget=0.1)
    assert "shrug-signals-rejected" in out["evidence"]
    assert "embedding-dissimilarity-rejected" in out["evidence"]
    assert any("general theorem" in c for c in out["caveats"])
    assert any("Bolivia" in c for c in out["caveats"])


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


def test_review_set_warns_on_multiclass_probabilities() -> None:
    out = review_set([[0.5, 0.3, 0.2], [0.4, 0.4, 0.2]], budget=1.0)
    assert out["score_type_detected"] == "probability"
    assert any(
        "softmax margin can order windows differently" in c for c in out["caveats"]
    )


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
