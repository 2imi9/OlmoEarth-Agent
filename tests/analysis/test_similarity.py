# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Tests for embedding similarity search (skill #9)."""

from __future__ import annotations

import pytest

from olmoearth_agent.analysis.similarity import (
    geographic_prior_check,
    similarity_search,
)

CORPUS = [[1.0, 0.0], [0.0, 1.0], [0.9, 0.1], [-1.0, 0.0]]


def test_cosine_top_k_ordering() -> None:
    out = similarity_search([1.0, 0.0], CORPUS, k=2, metric="cosine")
    assert out["metric"] == "cosine"
    assert out["matches"][0]["index"] == 0  # identical direction
    assert out["matches"][0]["score"] == 1.0
    assert out["matches"][1]["index"] == 2  # [0.9, 0.1] is next-closest


def test_euclidean_ordering() -> None:
    out = similarity_search(
        [0.0, 0.0], [[3.0, 4.0], [1.0, 0.0], [0.0, 2.0]], k=2, metric="euclidean"
    )
    assert [m["index"] for m in out["matches"]] == [1, 2]  # distances 5, 1, 2


def test_self_match_is_top() -> None:
    out = similarity_search([0.0, 1.0], CORPUS, k=1)
    assert out["matches"][0]["index"] == 1
    assert out["matches"][0]["score"] == 1.0


def test_ids_are_echoed() -> None:
    out = similarity_search([1.0, 0.0], CORPUS, ids=["a", "b", "c", "d"], k=1)
    assert out["matches"][0]["id"] == "a"


def test_k_is_clamped_to_corpus_size() -> None:
    out = similarity_search([1.0, 0.0], CORPUS, k=99)
    assert out["k"] == 4
    assert len(out["matches"]) == 4


def test_zero_query_vector_does_not_crash() -> None:
    out = similarity_search([0.0, 0.0], CORPUS, k=2, metric="cosine")
    assert all(m["score"] == 0.0 for m in out["matches"])


def test_empty_corpus_rejected() -> None:
    with pytest.raises(ValueError, match="corpus must be non-empty"):
        similarity_search([1.0, 0.0], [])


def test_empty_query_rejected() -> None:
    with pytest.raises(ValueError, match="query vector must be non-empty"):
        similarity_search([], CORPUS)


def test_dimension_mismatch_rejected() -> None:
    with pytest.raises(ValueError, match="match the query dimension"):
        similarity_search([1.0, 0.0], [[1.0, 0.0, 0.0]])


def test_ids_length_mismatch_rejected() -> None:
    with pytest.raises(ValueError, match="ids length"):
        similarity_search([1.0, 0.0], CORPUS, ids=["a"])


def test_invalid_k_rejected() -> None:
    with pytest.raises(ValueError, match="k must be"):
        similarity_search([1.0, 0.0], CORPUS, k=0)


def test_invalid_metric_rejected() -> None:
    with pytest.raises(ValueError, match="metric must be"):
        similarity_search([1.0, 0.0], CORPUS, metric="manhattan")


def test_geographic_prior_warns_when_clustered() -> None:
    out = geographic_prior_check([0.0, 0.0], [[0.1, 0.1], [0.0, 0.1], [-0.1, 0.0]])
    assert out["warning"] is True
    assert out["clustered_fraction"] == 1.0


def test_geographic_prior_no_warning_when_dispersed() -> None:
    out = geographic_prior_check(
        [0.0, 0.0], [[50.0, 50.0], [-50.0, -50.0], [100.0, 0.0]]
    )
    assert out["warning"] is False
    assert out["clustered_fraction"] == 0.0


def test_geographic_prior_empty_results() -> None:
    out = geographic_prior_check([0.0, 0.0], [])
    assert out["n_results"] == 0
    assert out["warning"] is False


def test_geographic_prior_rejects_bad_query_coord() -> None:
    with pytest.raises(ValueError, match="query_coord"):
        geographic_prior_check([0.0], [[1.0, 1.0]])


def test_geographic_prior_rejects_bad_result_coord() -> None:
    with pytest.raises(ValueError, match="result coord"):
        geographic_prior_check([0.0, 0.0], [[1.0, 1.0, 1.0]])
