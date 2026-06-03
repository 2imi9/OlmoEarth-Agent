# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Tests for pseudo-absence sampling (skill #18)."""

from __future__ import annotations

import pytest

from olmoearth_agent.analysis.negative_sampler import (
    bounding_box,
    candidate_grid,
    sample_negatives,
)
from olmoearth_agent.evaluation.spatial_cv import haversine_km


def test_bounding_box_expands_by_margin() -> None:
    box = bounding_box([(0.0, 0.0), (2.0, 4.0)], margin_deg=0.5)
    assert box == (-0.5, -0.5, 2.5, 4.5)


def test_bounding_box_single_point_gets_area() -> None:
    box = bounding_box([(10.0, 20.0)], margin_deg=0.1)
    assert box[2] > box[0] and box[3] > box[1]


def test_bounding_box_rejects_empty() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        bounding_box([])


def test_bounding_box_degenerate_rejected() -> None:
    with pytest.raises(ValueError, match="degenerate"):
        bounding_box([(1.0, 1.0)], margin_deg=0.0)


def test_candidate_grid_auto_sizes_to_target() -> None:
    grid = candidate_grid((0.0, 0.0, 4.0, 4.0), n_target=16)
    # ~16 points, all strictly inside the bbox.
    assert 9 <= len(grid) <= 25
    assert all(0.0 < lon < 4.0 and 0.0 < lat < 4.0 for lon, lat in grid)


def test_candidate_grid_explicit_step() -> None:
    grid = candidate_grid((0.0, 0.0, 1.0, 1.0), n_target=4, step_deg=0.5)
    assert len(grid) == 4  # 2x2 cells


def test_candidate_grid_guards_runaway() -> None:
    with pytest.raises(ValueError, match="grid would have"):
        candidate_grid((0.0, 0.0, 100.0, 100.0), n_target=4, step_deg=0.01)


def test_default_count_is_balanced_one_to_one() -> None:
    positives = [(0.0, 0.0), (1.0, 1.0), (2.0, 0.5)]
    out = sample_negatives(positives, exclusion_km=0.0)
    assert out["n_requested"] == 3
    assert out["n_negatives"] == 3
    assert out["positive_to_negative_ratio"] == 1.0
    assert out["candidate_source"] == "grid"
    assert out["bbox"] is not None


def test_negatives_respect_exclusion_buffer() -> None:
    positives = [(0.0, 0.0), (0.5, 0.5)]
    out = sample_negatives(positives, exclusion_km=5.0, n_negatives=4)
    for lon, lat in out["negatives"]:
        nearest = min(haversine_km((lon, lat), p) for p in positives)
        assert nearest >= 5.0


def test_negatives_respect_min_separation() -> None:
    positives = [(0.0, 0.0)]
    out = sample_negatives(
        positives, exclusion_km=1.0, min_separation_km=50.0, n_negatives=5
    )
    negs = out["negatives"]
    for i in range(len(negs)):
        for j in range(i + 1, len(negs)):
            assert haversine_km(tuple(negs[i]), tuple(negs[j])) >= 50.0


def test_all_candidates_buffered_out_warns() -> None:
    out = sample_negatives([(0.0, 0.0)], candidates=[(0.001, 0.0)], exclusion_km=1.0)
    assert out["n_negatives"] == 0
    assert out["ranking"] == "none"
    assert out["warnings"]


def test_embedding_dissimilarity_picks_least_similar() -> None:
    # Centroid points to [1, 0]; candidate 1 is the opposite direction.
    out = sample_negatives(
        [(0.0, 0.0)],
        positive_embeddings=[[1.0, 0.0]],
        candidates=[(10.0, 10.0), (20.0, 20.0)],
        candidate_embeddings=[[1.0, 0.0], [-1.0, 0.0]],
        n_negatives=1,
        exclusion_km=1.0,
    )
    assert out["ranking"] == "embedding_dissimilarity"
    assert out["negatives"] == [[20.0, 20.0]]


def test_spatial_dispersion_seeds_farthest_from_positives() -> None:
    out = sample_negatives(
        [(0.0, 0.0)],
        candidates=[(1.0, 0.0), (50.0, 50.0)],
        n_negatives=1,
        exclusion_km=1.0,
    )
    assert out["ranking"] == "spatial_dispersion"
    assert out["negatives"] == [[50.0, 50.0]]


def test_short_placement_warns() -> None:
    # Only two candidates survive but five are requested.
    out = sample_negatives(
        [(0.0, 0.0)],
        candidates=[(10.0, 0.0), (20.0, 0.0)],
        n_negatives=5,
        exclusion_km=1.0,
        min_separation_km=0.0,
    )
    assert out["n_negatives"] == 2
    assert any("placed 2 of 5" in w for w in out["warnings"])


def test_deterministic() -> None:
    positives = [(0.0, 0.0), (1.0, 2.0), (3.0, 1.0), (2.0, 3.0)]
    a = sample_negatives(positives, exclusion_km=1.0)
    b = sample_negatives(positives, exclusion_km=1.0)
    assert a == b


def test_empty_positives_rejected() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        sample_negatives([])


def test_bad_positive_pair_rejected() -> None:
    with pytest.raises(ValueError, match="lon, lat"):
        sample_negatives([(0.0, 0.0, 0.0)])  # type: ignore[list-item]


def test_negative_exclusion_rejected() -> None:
    with pytest.raises(ValueError, match="exclusion_km"):
        sample_negatives([(0.0, 0.0)], exclusion_km=-1.0)


def test_invalid_n_negatives_rejected() -> None:
    with pytest.raises(ValueError, match="n_negatives"):
        sample_negatives([(0.0, 0.0)], n_negatives=0)


def test_invalid_oversample_rejected() -> None:
    with pytest.raises(ValueError, match="oversample"):
        sample_negatives([(0.0, 0.0)], oversample=0)


def test_candidate_embeddings_without_candidates_rejected() -> None:
    with pytest.raises(ValueError, match="requires explicit candidates"):
        sample_negatives([(0.0, 0.0)], candidate_embeddings=[[1.0]])


def test_positive_embeddings_length_mismatch_rejected() -> None:
    with pytest.raises(ValueError, match="positive_embeddings length"):
        sample_negatives([(0.0, 0.0)], positive_embeddings=[[1.0], [2.0]])


def test_candidate_embeddings_length_mismatch_rejected() -> None:
    with pytest.raises(ValueError, match="candidate_embeddings length"):
        sample_negatives(
            [(0.0, 0.0)],
            candidates=[(10.0, 10.0)],
            candidate_embeddings=[[1.0], [2.0]],
        )


def test_embedding_dimension_mismatch_rejected() -> None:
    with pytest.raises(ValueError, match="dimensionality must match"):
        sample_negatives(
            [(0.0, 0.0)],
            positive_embeddings=[[1.0, 0.0, 0.0]],
            candidates=[(10.0, 10.0)],
            candidate_embeddings=[[1.0, 0.0]],
            exclusion_km=1.0,
        )
