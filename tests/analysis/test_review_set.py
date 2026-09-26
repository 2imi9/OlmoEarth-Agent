# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Tests for the label-free review-set ranking (skill #18)."""

from __future__ import annotations

import random

import pytest

from olmoearth_agent.analysis.output_contract import (
    MUST_STATE_MAX,
    MUST_STATE_MAX_WORDS,
    add_forbidden,
    add_must_state,
    word_count,
)
from olmoearth_agent.analysis.review_set import (
    BOUNDARY_NEIGHBOURS_MEANS,
    BOUNDARY_SHARE_MEANS,
    EVIDENCE,
    EVIDENCE_LIMITS,
    MUST_STATE_BINARY_SCORE,
    MUST_STATE_MULTICLASS_LOGIT,
    MUST_STATE_NO_WINNER,
    MUST_STATE_UNNAMED_MODEL,
    NOT_COVERED,
    RANKING_SOURCE,
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
    margin_summary,
    margins,
    more_confident_fact,
    must_state_other_model,
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
    # Inline scores name no model, so coverage is not known and must be stated,
    # in one short sentence: the scope itself stays in evidence_scope, naming
    # its source without the suite's figures (exp86 round 8).
    assert scope.startswith(RANKING_SOURCE) and "OlmoEarth models" in scope
    assert "24" not in scope and "suite" not in scope
    assert scope.count(". ") == 0 and scope.endswith(".")
    assert out["evidence_covers_this_case"] == "not known"
    assert out["must_state"] == [MUST_STATE_UNNAMED_MODEL]
    assert [c["id"] for c in out["forbidden_claims"]] == [
        "error_rate_without_labels",
        "evidence_outside_its_scope",
    ]


def test_not_listed_says_it_counts_after_the_tools_own_listing() -> None:
    """exp86 round 9 (B2/studio run 1): 5 of 8 listed windows shown, and the
    tool's "83 not listed" called "the other 83 windows" (86 were not shown,
    three of them below the stated range). Each count names its ranks."""
    marg = [0.01, 0.04, 0.12, 0.19, 0.22, 0.26, 0.39, 0.47, 0.52, 0.8, 0.9, 0.95]
    ranked = sorted(range(len(marg)), key=marg.__getitem__)
    summary = margin_summary(marg, ranked, k=9, n_listed=8)
    assert summary["listed"]["ranks"] == [1, 8]
    assert summary["review_set"]["ranks"] == [1, 9]
    not_listed = summary["not_listed"]
    assert not_listed["n"] == 4 and not_listed["ranks"] == [9, 12]
    assert not_listed["after"] == "the tool's 8 listed windows"
    reading = summary["reading"]
    assert "The 4 windows ranked after the tool's 8 listed (ranks 9 to 12" in reading
    assert "a table of fewer than 8 rows leaves out more windows" in reading
    assert "no higher than 0.52" in reading
    none = margin_summary(marg, ranked, k=9, n_listed=0)
    assert none["not_listed"]["ranks"] == [1, 12] and none["listed"]["ranks"] is None
    assert "No window is listed" in none["reading"]


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
        ("logit", 9, "allenai/OlmoEarth-v1-FT-AWF-Base", "in part", "14 of 16"),
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
    if covers in NOT_COVERED:
        # exp86 round 8: the source only, none of its figures or its dataset.
        assert scope["sentence"].startswith(RANKING_SOURCE)
        assert "24" not in scope["sentence"] and "suite" not in scope["sentence"]
    elif covers == "in part":
        # The suite measured the probability margin: none of its 24-task
        # figures, only exp76's about the logit margin itself.
        assert "24" not in scope["sentence"]
        assert "not the logit margin" in scope["sentence"]
    else:
        assert scope["sentence"].startswith("Ai2's 24-task embedding suite")
    # A limit goes to must_state as its own short sentence, never the scope.
    if covers == "yes":
        assert scope["must_state"] is None
    else:
        assert scope["must_state"] != scope["sentence"]
        assert word_count(scope["must_state"]) <= MUST_STATE_MAX_WORDS


def test_must_state_is_short_and_never_the_long_scope_sentence() -> None:
    """exp87 review: must_state carried the 60-word evidence_scope sentence. The
    contract holds a must_state sentence to 25 words; the scope stays whole in
    evidence_scope (and the file)."""
    cases = [
        ("binary_score", 2, None),
        ("threshold_distance", 2, None),
        ("logit", 2, None),
        ("logit", 2, "someone/Other-Model"),
        ("logit", 9, "allenai/OlmoEarth-v1-FT-AWF-Base"),
        ("window_confidence_logit", 9, "allenai/OlmoEarth-v1-FT-AWF-Base"),
    ]
    for kind, n_classes, model in cases:
        rows = [[0.2 + 0.1 * i] + [0.0] * (n_classes - 1) for i in range(4)]
        out = review_set(rows, budget=0.5, score_kind=kind, model=model)
        assert 1 <= len(out["must_state"]) <= MUST_STATE_MAX
        for sentence in out["must_state"]:
            assert word_count(sentence) <= MUST_STATE_MAX_WORDS
            assert sentence != out["evidence_scope"]
            assert "form=" not in sentence and "pass " not in sentence.lower()
        assert word_count(out["evidence_scope"]) > MUST_STATE_MAX_WORDS
    band = review_set([[0.4, 0.6], [0.9, 0.1]], budget=1.0, score_kind="binary_score")
    assert band["must_state"] == [MUST_STATE_BINARY_SCORE]
    logits = review_set(
        [[0.0, 2.0, 0.0], [1.5, 0.0, 0.0]],
        budget=1.0,
        score_type="logit",
        model="allenai/OlmoEarth-v1-FT-AWF-Base",
    )
    assert logits["must_state"] == [MUST_STATE_MULTICLASS_LOGIT]
    # A model name that is not one word does not lengthen the sentence.
    odd = must_state_other_model("a model with " + "many words " * 20)
    assert word_count(odd) <= MUST_STATE_MAX_WORDS and "another model" in odd
    assert "someone/Other-Model" in must_state_other_model("someone/Other-Model")


def test_add_must_state_holds_the_contracts_limits() -> None:
    out: dict = {}
    add_must_state(out, ["One.", "Two.", "One."])
    assert out["must_state"] == ["One.", "Two."]
    add_must_state(out, ["Three.", "Four."])
    assert out["must_state"] == ["One.", "Two.", "Three."]  # at most three
    with pytest.raises(ValueError, match="words"):
        add_must_state({}, [" ".join(["word"] * (MUST_STATE_MAX_WORDS + 1))])


def test_add_forbidden_lists_each_id_once() -> None:
    out = {"forbidden_claims": [{"id": "winner_without_labels", "why": "a"}]}
    add_forbidden(
        out,
        [
            {"id": "winner_without_labels", "why": "b"},
            {"id": "another_date_settles_it", "why": "c"},
            {"id": "another_date_settles_it", "why": "d"},
        ],
    )
    assert [(c["id"], c["why"]) for c in out["forbidden_claims"]] == [
        ("winner_without_labels", "a"),
        ("another_date_settles_it", "c"),
    ]


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
    assert out["must_state"] == [MUST_STATE_MULTICLASS_LOGIT]
    two = review_set(
        [[0.2, 0.8], [0.6, 0.4]], budget=1.0, model="allenai/OlmoEarth-v1-Base"
    )
    assert two["evidence_covers_this_case"] == "yes" and two["must_state"] == []
    band = review_set([[0.4, 0.6], [0.9, 0.1]], budget=1.0, score_kind="binary_score")
    assert band["evidence_covers_this_case"] == "no"
    assert band["must_state"] == [MUST_STATE_BINARY_SCORE]


def test_margin_ratio_is_the_median_over_the_listed_margins() -> None:
    """exp86 round 6 (brief 8): "3-4 orders of magnitude more uncertain" where the
    median margin was 13 to 41 times the listed ones. The ratio is a fact."""
    scores = [
        [1.0 + m, 1.0] for m in (0.1, 0.2, 0.4, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0)
    ]
    out = review_set(scores, budget=0.3)
    (fact,) = out["facts"]
    assert set(fact) == {
        "id",
        "listed_n",
        "listed_low",
        "listed_high",
        "review_set_low",
        "review_set_high",
        "versus",
        "sentence",
    }
    assert fact["id"] == "margin_ratio" and fact["versus"] == "median"
    assert (fact["listed_low"], fact["listed_high"]) == (10.0, 40.0)
    # Every window at the budget is listed: the two ranges are one.
    assert (fact["review_set_low"], fact["review_set_high"]) == (10.0, 40.0)
    assert "10.00 to 40.00 times" in fact["sentence"]
    assert "all 3 windows in the review set, every one listed (0.1 to 0.4)" in (
        fact["sentence"]
    )


def test_margin_ratio_names_its_windows_and_gives_the_first_ten_their_own() -> None:
    """exp86 round 8 (brief 8 on the cluster, run 2): an answer showed ten of
    the 50 listed windows and gave them the 50's ratio, "13 to 41 times"; the
    ten shown were 20.68 to 41.31 times below the median. The fact says how
    many windows its listed ratio covers and gives the first ten theirs."""
    marg = [0.1 * (i + 1) for i in range(60)] + [10.0] * 940
    scores = [[1.0 + m, 1.0] for m in marg]
    out = review_set(scores, budget=0.1, max_listed=50)
    (fact,) = out["facts"]
    assert fact["listed_n"] == 50
    assert (fact["listed_low"], fact["listed_high"]) == (2.0, 100.0)
    assert (fact["first_n"], fact["first_low"], fact["first_high"]) == (
        10,
        10.0,
        100.0,
    )
    assert (fact["review_set_low"], fact["review_set_high"]) == (1.0, 100.0)
    assert (
        "2.00 to 100.00 times the margins of the 50 listed windows (0.1 to 5), "
        "10.00 to 100.00 times those of the first 10 listed (0.1 to 1) and 1.00 "
        "to 100.00 times those of all 100 windows in the review set (0.1 to 10)"
    ) in fact["sentence"]
    # Ten or fewer listed: the listed ratio is the first ten's; no second one.
    few = review_set(scores, budget=0.1, max_listed=10)
    (fact,) = few["facts"]
    assert fact["listed_n"] == 10 and "first_n" not in fact
    assert "first 10 listed" not in fact["sentence"]


def test_margin_ratio_gives_the_whole_review_sets_range_beside_the_listed() -> None:
    """exp87 review: the listed range was the only one, so an answer about the
    review set read the first few windows' range as the whole set's."""
    scores = [
        [1.0 + m, 1.0] for m in (0.1, 0.2, 0.4, 0.8, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0)
    ]
    out = review_set(scores, budget=0.4, max_listed=2)
    assert out["margin_summary"]["review_set"] == {
        "n": 4,
        "margin_range": [0.1, 0.8],
        "ranks": [1, 4],
    }
    (fact,) = out["facts"]
    assert (fact["listed_low"], fact["listed_high"]) == (20.0, 40.0)
    assert (fact["review_set_low"], fact["review_set_high"]) == (5.0, 40.0)
    assert (
        "20.00 to 40.00 times the margins of the 2 listed windows (0.1 to 0.2) and "
        "5.00 to 40.00 times those of all 4 windows in the review set (0.1 to 0.8)"
    ) in fact["sentence"]
    none_listed = review_set(scores, budget=0.4, max_listed=0)
    (fact,) = none_listed["facts"]
    assert fact["listed_low"] is None and fact["listed_high"] is None
    assert fact["review_set_low"] == 5.0 and "none is listed" in fact["sentence"]


def test_margin_ratio_with_a_zero_margin_has_no_upper_ratio() -> None:
    summary = {
        "n_windows": 5,
        "median_margin": 2.0,
        "listed": {"n": 2, "margin_range": [0.0, 0.5]},
        "review_set": {"n": 3, "margin_range": [0.0, 1.0]},
    }
    fact = margin_ratio_fact(summary)
    assert fact is not None
    assert (fact["listed_low"], fact["listed_high"]) == (4.0, None)
    assert (fact["review_set_low"], fact["review_set_high"]) == (2.0, None)
    assert "at least 4.00 times" in fact["sentence"]
    assert "no ratio bounds it" in fact["sentence"]
    zero = margin_ratio_fact(
        {
            **summary,
            "listed": {"n": 1, "margin_range": [0.0, 0.0]},
            "review_set": {"n": 1, "margin_range": [0.0, 0.0]},
        }
    )
    assert zero is not None and zero["listed_low"] is None
    assert "no finite multiple of" in zero["sentence"]
    assert (
        margin_ratio_fact({**summary, "review_set": {"n": 0, "margin_range": None}})
        is None
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
    assert set(dominant) == {
        "id",
        "from_class",
        "to_class",
        "n",
        "share",
        "reverse_n",
        "reverse_share",
        "tied_with_reverse",
        "sentence",
    }
    assert (dominant["from_class"], dominant["to_class"]) == (2, 1)
    assert (dominant["n"], dominant["reverse_n"]) == (10, 2)
    assert dominant["share"] == pytest.approx(10 / 12, abs=1e-6)
    assert dominant["tied_with_reverse"] is False
    assert (
        "class 2 (grass) in map A to class 1 (forest) in map B" in dominant["sentence"]
    )
    assert "the reverse" in dominant["sentence"]
    side = facts["more_confident_side"]
    assert set(side) == {"id", "side", "share_a", "share_b", "share_equal", "sentence"}
    # A's margin is 3 and B's 2 on every differing window.
    assert (side["side"], side["share_a"], side["share_b"]) == ("A", 1.0, 0.0)
    assert side["share_equal"] == 0.0
    assert "does not make A right" in side["sentence"]
    assert out["b_more_confident_share_of_differing"] == 0.0
    assert out["class_changes"][0]["class_name_a"] == "grass"
    assert "concentration" not in facts  # no grid, no spatial breakdown


def test_the_comparison_scope_and_contract() -> None:
    a, b = _flip(4, [1])
    out = compare_scores(a, b)
    assert "evidence" not in out and "caveats" not in out
    # exp86 round 8: the scope names its source and that it does not cover the
    # pair, never exp58's figures or its dataset.
    assert "exp58" in out["evidence_scope"]
    assert "does not cover this pair" in out["evidence_scope"]
    assert "51 to 70" not in out["evidence_scope"]
    assert "Sen1Floods11" not in out["evidence_scope"]
    assert out["evidence_covers_this_case"] == "no"
    assert out["must_state"] == [MUST_STATE_NO_WINNER]
    assert [c["id"] for c in out["forbidden_claims"]] == [
        "winner_without_labels",
        "evidence_outside_its_scope",
    ]


def test_spatial_breakdown_counts_every_differing_window_by_band() -> None:
    """exp86: "a long strip along the north edge" for 1.3% of the differences.
    The shares per row and column band are over every differing window."""
    rows, cols = 8, 8
    # Rows 0-1 hold 2 differing windows; rows 4-5 hold 12.
    diff = [0, 1] + [r * cols + c for r in (4, 5) for c in range(6)]
    a, b = _flip(rows * cols, diff)
    out = compare_scores(a, b, grid=(rows, cols), max_listed=3)
    spatial = out["spatial"]
    assert [band["band"] for band in spatial["row_bands"]] == [0, 1, 2, 3]
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
    assert spatial["max_band"]["axis"] == "rows" and spatial["max_band"]["band"] == 2
    # top_band_share is the NORTHMOST row band's share, not the largest band's.
    assert spatial["top_band_share"] == pytest.approx(2 / 14, abs=1e-6)
    assert out["n_differing_listed"] == 3  # the listing is still the first ones
    assert "boundary_means" in out and "PREDICTED" in out["boundary_means"]


def test_a_comparisons_winner_claim_says_no_labels_were_given_to_it() -> None:
    """exp86 round 9 (B3/studio run 1): "no labels were used" read as "no
    ground-truth labels exist"."""
    a = [[5.0, 0.1], [0.1, 5.0], [5.0, 0.1]]
    b = [[5.0, 0.1], [5.0, 0.1], [5.0, 0.1]]
    out = compare_scores(a, b)
    (why,) = [
        c["why"] for c in out["forbidden_claims"] if c["id"] == "winner_without_labels"
    ]
    assert why.startswith("no labels were given to this comparison")
    assert "nothing here says whether labels for these maps exist" in why


def test_a_boundary_share_says_it_does_not_show_whether_regions_flip() -> None:
    """exp86 round 9 (B3/cluster run 3): "78.3% of the differing windows sit on
    class boundaries (vs. 53.9%) - mostly boundary reclassification rather than
    wholesale area flips". A window with one differing neighbour is on a
    boundary, so a block that flips whole has every window on one."""
    rows, cols = 8, 8
    # a 4 x 4 block flips whole in map B; map A's classes form two halves
    a = [[5.0, 0.1] if c < 4 else [0.1, 5.0] for r in range(rows) for c in range(cols)]
    block = {r * cols + c for r in range(2, 6) for c in range(2, 6)}
    b = [row[::-1] if i in block else list(row) for i, row in enumerate(a)]
    out = compare_scores(a, b, grid=(rows, cols), max_listed=3)
    assert BOUNDARY_SHARE_MEANS == out["boundary_means"]
    assert "do not show whether whole regions flip" in out["boundary_means"]
    assert "not whether whole regions flip" in out["where"]
    assert "mostly" not in out["where"]
    (fact,) = [f for f in out["facts"] if f["id"] == "boundary_share"]
    assert fact["share_of_differing"] == out["boundary_share_of_differing"]
    assert fact["share_overall"] == out["boundary_share_overall"]
    assert fact["sentence"].startswith(
        f"{out['boundary_share_of_differing'] * 100:.1f}% of the 16 differing windows"
    )
    assert "does not show whether whole regions flip" in fact["sentence"]
    same = compare_scores(a, a, grid=(rows, cols), max_listed=3)
    assert "boundary_share" not in {f["id"] for f in same["facts"]}


def test_the_concentration_fact_is_the_contracts() -> None:
    """exp87 review: top_band_share was the largest band's share, so a claim
    about the north edge was checked against the wrong band. It is row band 0's
    (the northmost); the largest band is max_band, beside it."""
    rows, cols = 8, 8
    diff = [0, 1] + [r * cols + c for r in (4, 5) for c in range(6)]
    a, b = _flip(rows * cols, diff)
    out = compare_scores(a, b, grid=(rows, cols), max_listed=3)
    fact = next(f for f in out["facts"] if f["id"] == "concentration")
    assert set(fact) == {
        "id",
        "grid",
        "n_differing",
        "top_band_share",
        "max_band",
        "row_order",
        "sentence",
    }
    assert fact["grid"] == [8, 8] and fact["n_differing"] == 14
    assert fact["top_band_share"] == pytest.approx(2 / 14, abs=1e-6)
    assert fact["max_band"] == {
        "axis": "rows",
        "band": 2,
        "of_grid": "50-75%",
        "share": pytest.approx(12 / 14, abs=1e-6),
    }
    # Inline scores carry no georeference: row band 0 is the grid's first rows,
    # never "the northmost" (exp86 round 8, brief 7 on files).
    assert fact["row_order"] is None
    assert fact["sentence"] == (
        "Of the 14 differing windows, 14.3% lie in row band 0 (rows 0-1, the "
        "grid's first rows, band 0 of 4), which holds 25.0% of all compared "
        "windows; the most in any band, 85.7%, lie in rows 4-5 (50-75% of the "
        "grid's 8 rows, band 2), which holds 25.0% of all compared windows. The "
        "scores carry no georeference, so the rows need not run north to south."
    )
    # Georeferenced scores whose row 0 is north: the northmost band.
    north = compare_scores(
        a, b, grid=(rows, cols), max_listed=3, row_order="north_to_south"
    )
    fact = next(f for f in north["facts"] if f["id"] == "concentration")
    assert fact["row_order"] == "north_to_south"
    assert fact["sentence"].startswith(
        "Of the 14 differing windows, 14.3% lie in the northmost row band (rows "
        "0-1, band 0 of 4), which holds 25.0% of all compared windows;"
    )
    assert "georeference" not in fact["sentence"]
    assert "the north edge" in north["spatial"]["reading"]


def test_the_concentration_fact_on_columns_on_the_north_band_and_on_a_tie() -> None:
    rows, cols = 4, 4
    # Column 3 holds three differing windows, row 0 one of them.
    a, b = _flip(rows * cols, [3, 7, 11])
    fact = next(
        f
        for f in compare_scores(a, b, grid=(rows, cols))["facts"]
        if f["id"] == "concentration"
    )
    assert fact["max_band"]["axis"] == "cols" and fact["max_band"]["band"] == 3
    assert fact["top_band_share"] == pytest.approx(1 / 3, abs=1e-6)
    assert "lie in column 3 (75-100% of the grid's 4 columns, band 3)" in (
        fact["sentence"]
    )
    # Row band 0 holds the most, so the sentence says it is the most.
    north = next(
        f
        for f in compare_scores(*_flip(16, [0, 1, 2]), grid=(4, 4))["facts"]
        if f["id"] == "concentration"
    )
    assert north["top_band_share"] == 1.0
    assert north["max_band"]["axis"] == "rows" and north["max_band"]["band"] == 0
    assert "that is the most of any of the 4 row and 4 column bands." in (
        north["sentence"]
    )
    # An even spread over the rows: the tie is counted, never read as a place.
    even = next(
        f
        for f in compare_scores(*_flip(16, [0, 5, 10, 15]), grid=(4, 4))["facts"]
        if f["id"] == "concentration"
    )
    assert even["max_band"]["band"] == 0 and even["top_band_share"] == 0.25
    assert "(tied with 7 other bands)" in even["sentence"]


def test_spatial_breakdown_places_windows_of_a_partial_grid() -> None:
    """A scores file that leaves no-data windows out: rows are placed by their
    grid index, and each band's share of the compared windows is its own."""
    out = spatial_breakdown([5, 6, 7], [0, 1, 5, 6, 7, 15], (4, 4))
    assert [b["n_windows"] for b in out["row_bands"]] == [2, 3, 0, 1]
    assert [b["n_differing"] for b in out["row_bands"]] == [0, 3, 0, 0]
    assert out["max_band"]["rows"] == [1, 1] and out["max_band"]["share"] == 1.0
    assert out["top_band_share"] == 0.0
    assert [b["n_differing"] for b in out["col_bands"]] == [0, 1, 1, 1]
    empty = spatial_breakdown([], [0, 1], (1, 2))
    assert empty["max_band"] is None and empty["n_bands"] == [1, 2]
    assert empty["top_band_share"] is None


def test_compare_places_rows_of_a_partial_grid() -> None:
    a, b = _flip(3, [1, 2])
    out = compare_scores(a, b, grid=(3, 3), windows=[0, 4, 8], max_listed=None)
    assert [(d["window_index"], d["row"], d["col"]) for d in out["differing"]] == [
        (4, 1, 1),
        (8, 2, 2),
    ]
    assert "boundary_share_overall" not in out  # needs every window
    assert out["spatial"]["top_band_share"] == 0.0  # row 0 holds none of them
    assert out["spatial"]["max_band"]["share"] == 0.5
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
    """The side is decided on counts, and a tie is "neither", stated."""
    assert more_confident_fact(0, 0, 0) is None
    tie = more_confident_fact(4, 4, 10)
    assert tie is not None and tie["side"] == "neither"
    assert (tie["share_a"], tie["share_b"], tie["share_equal"]) == (0.4, 0.4, 0.2)
    assert tie["sentence"].startswith("Neither map is more often the more confident")
    assert "equal on 20.0%" in tie["sentence"]
    fact = more_confident_fact(3, 6, 10)
    assert fact is not None and fact["side"] == "B"
    assert (fact["share_a"], fact["share_b"], fact["share_equal"]) == (0.3, 0.6, 0.1)
    assert "equal on 10.0%" in fact["sentence"]


def test_the_more_confident_side_is_decided_on_counts_not_rounded_shares() -> None:
    """exp87 review: shares rounded to 6 decimals can be equal where the counts
    are not (and were compared as floats)."""
    n = 10_000_001
    fact = more_confident_fact(5_000_001, 5_000_000, n)
    assert fact is not None
    assert fact["share_a"] == fact["share_b"]  # equal once rounded
    assert fact["side"] == "A"


def test_a_tied_dominant_change_says_so() -> None:
    fact = dominant_change_fact({(0, 1): 5, (1, 0): 5}, 10, None)
    assert fact is not None
    assert fact["tied_with_reverse"] is True
    assert (fact["from_class"], fact["to_class"]) == (0, 1)
    assert (fact["reverse_n"], fact["reverse_share"]) == (5, 0.5)
    assert fact["sentence"] == (
        "The most common change is a tie between a direction and its reverse, 5 of "
        "the 10 differing windows (50.0%) each: class 0 in map A to class 1 in map "
        "B, and class 1 in map A to class 0 in map B."
    )
    assert dominant_change_fact({}, 0, None) is None


def test_a_dominant_change_tied_with_another_direction_names_both() -> None:
    """A tie beyond the ten listed class changes still counts: the fact reads
    every direction, not the listing."""
    pairs = {(0, 1): 4, (2, 3): 4, (1, 0): 1}
    pairs.update({(10 + i, 20 + i): 1 for i in range(12)})
    fact = dominant_change_fact(pairs, 21, None)
    assert fact is not None and fact["tied_with_reverse"] is False
    assert fact["sentence"] == (
        "The most common change is a tie between 2 directions, 4 of the 21 "
        "differing windows (19.0%) each: class 0 in map A to class 1 in map B, and "
        "class 2 in map A to class 3 in map B; for the first, the reverse, class 1 "
        "in map A to class 0 in map B, is 1 (4.8%)."
    )
    many = {(0, 1): 3, (1, 0): 3, (2, 0): 3}
    fact = dominant_change_fact(many, 9, None)
    assert fact is not None and fact["tied_with_reverse"] is True
    assert "a tie between 3 directions" in fact["sentence"]
    assert fact["sentence"].endswith("class 2 in map A to class 0 in map B.")


# --------------------------------------------------------------------------- evidence outside its scope


def _strings(value: object) -> list[str]:
    """Every string anywhere inside a JSON-like value."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [t for v in value.values() for t in _strings(v)]
    if isinstance(value, list):
        return [t for v in value for t in _strings(v)]
    return []


#: exp58's figures and dataset, and the suite's, as the fields carried them.
_UPSTREAM = ("51 to 70", "Sen1Floods11", "15 pairs", "24-task", "all 24", "exp78")


def test_no_field_carries_evidence_that_does_not_cover_the_case() -> None:
    """exp86 round 8: "the more confident side was only right 51-70% of the
    time in comparable cases" (brief 3 on the cluster) and "the two
    Sen1Floods11 flood maps" (brief 7 on files), for a pair whose scope said
    the evidence does not cover it. Where it does not, no field carries its
    figures or dataset, and the answer may not apply them."""
    a, b = _flip(16, [1, 5, 9])
    comparison = compare_scores(a, b, grid=(4, 4))
    band = review_set(
        [[0.4, 0.6], [0.9, 0.1], [0.3, 0.7]], budget=1.0, score_kind="binary_score"
    )
    for out in (comparison, band):
        assert out["evidence_covers_this_case"] in NOT_COVERED
        for text in _strings(out):
            assert not any(figure in text for figure in _UPSTREAM), text
        (claim,) = [
            c
            for c in out["forbidden_claims"]
            if c["id"] == "evidence_outside_its_scope"
        ]
        assert "in comparable cases" in claim["why"]
        assert "upstream evidence shows" in claim["why"]
    assert "exp58" in _strings(comparison["forbidden_claims"])[-1]
    # The file keeps the record whole.
    assert "51 to 70" in evidence_detail()["comparison"]["reading"]


def test_evidence_that_covers_the_case_in_part_states_only_its_own_figure() -> None:
    """Multi-class logits of an OlmoEarth model: the suite measured the
    probability margin, not the logit margin this ranking uses. The fix-r8
    review found its 24-task finding in evidence_scope (B8/cluster, all three
    runs): no field the answer reads carries it, exp76's figure about the
    logit margin itself stays, and evidence_outside_its_scope names what the
    suite did not measure."""
    out = review_set(
        [[0.0, 2.0, 0.0], [1.5, 0.0, 0.0]],
        budget=1.0,
        score_type="logit",
        model="allenai/OlmoEarth-v1-FT-AWF-Base",
    )
    assert out["evidence_covers_this_case"] == "in part"
    read = [out["evidence_scope"], *out["must_state"]] + [
        f["sentence"] for f in out["facts"]
    ]
    for text in read:
        assert not any(f in text for f in ("24-task", "all 24", "all 16")), text
    assert "14 of 16" in out["evidence_scope"] and "exp76" in out["evidence_scope"]
    assert out["must_state"] == [MUST_STATE_MULTICLASS_LOGIT]
    (claim,) = [
        c for c in out["forbidden_claims"] if c["id"] == "evidence_outside_its_scope"
    ]
    assert "logit margin, which the suite did not measure" in claim["why"]
    assert "14 of 16" in claim["why"]
    # The file keeps the suite's finding whole.
    assert "all 24 tasks" in evidence_detail()["ranking"]["measured"]
