# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Embedding similarity search (skill #9, ``olmoearth-similarity``).

Pure Python, no heavy deps. :func:`similarity_search` is an exact
brute-force top-K nearest-neighbour search over embedding vectors
(cosine or Euclidean). FAISS indexing is the scale-up follow-up; exact
kNN is correct and dependency-free at the sizes the agent passes in.

:func:`geographic_prior_check` is the honesty guard: it warns when the
top matches cluster geographically near the query, because then the
"similarity" may just be reflecting *location* (same region / biome)
rather than genuine feature resemblance: the classic similarity-search
failure mode. It returns a summary only (no raw coordinates), per
operational rule §3.1.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from olmoearth_agent.evaluation.spatial_cv import haversine_km

#: Default number of neighbours to return.
DEFAULT_K = 5

#: Default radius (km) within which top matches count as "clustered".
DEFAULT_PRIOR_RADIUS_KM = 100.0

_PLACES = 6


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return float(sum(x * y for x, y in zip(a, b)))


def _norm(a: Sequence[float]) -> float:
    return float(sum(x * x for x in a) ** 0.5)


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    na, nb = _norm(a), _norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return _dot(a, b) / (na * nb)


def _euclidean(a: Sequence[float], b: Sequence[float]) -> float:
    return float(sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5)


def similarity_search(
    query: Sequence[float],
    corpus: Sequence[Sequence[float]],
    *,
    ids: Sequence[str] | None = None,
    k: int = DEFAULT_K,
    metric: str = "cosine",
) -> dict[str, Any]:
    """Return the top-K corpus vectors most similar to the query.

    Parameters
    ----------
    query
        The query embedding vector.
    corpus
        Candidate embedding vectors (all the query's dimensionality).
    ids
        Optional labels parallel to ``corpus``; echoed in the matches.
    k
        Number of neighbours to return (clamped to the corpus size).
    metric
        ``"cosine"`` (higher = more similar) or ``"euclidean"`` (lower =
        more similar).

    Returns
    -------
    dict
        ``k``, ``metric``, and ``matches`` (each ``index``, ``score``, and
        ``id`` if ``ids`` was given), best first. Ties break by index.

    Raises
    ------
    ValueError
        If the corpus is empty, dimensions mismatch, ``ids`` is the wrong
        length, ``k < 1``, or ``metric`` is unknown.
    """
    if not corpus:
        raise ValueError("corpus must be non-empty.")
    dim = len(query)
    if dim == 0:
        raise ValueError("query vector must be non-empty.")
    for vec in corpus:
        if len(vec) != dim:
            raise ValueError("all corpus vectors must match the query dimension.")
    if ids is not None and len(ids) != len(corpus):
        raise ValueError("ids length must match corpus length.")
    if k < 1:
        raise ValueError("k must be >= 1.")
    if metric not in ("cosine", "euclidean"):
        raise ValueError("metric must be 'cosine' or 'euclidean'.")

    if metric == "cosine":
        scored = [(_cosine(query, vec), i) for i, vec in enumerate(corpus)]
        scored.sort(key=lambda s: (-s[0], s[1]))
    else:
        scored = [(_euclidean(query, vec), i) for i, vec in enumerate(corpus)]
        scored.sort(key=lambda s: (s[0], s[1]))

    matches = []
    for score, i in scored[:k]:
        match: dict[str, Any] = {"index": i, "score": round(score, _PLACES)}
        if ids is not None:
            match["id"] = str(ids[i])
        matches.append(match)
    return {"k": min(k, len(corpus)), "metric": metric, "matches": matches}


def geographic_prior_check(
    query_coord: Sequence[float],
    result_coords: Sequence[Sequence[float]],
    *,
    radius_km: float = DEFAULT_PRIOR_RADIUS_KM,
) -> dict[str, Any]:
    """Warn when the top matches cluster geographically near the query.

    Parameters
    ----------
    query_coord
        The query's ``[lon, lat]`` in degrees.
    result_coords
        ``[lon, lat]`` for each returned match.
    radius_km
        Matches within this distance of the query count as clustered.

    Returns
    -------
    dict
        ``n_results``, ``radius_km``, ``clustered_fraction``,
        ``mean_distance_km``, a ``warning`` boolean (set when a majority
        cluster within ``radius_km``), and a plain-language ``message``.

    Raises
    ------
    ValueError
        If a coordinate is not a ``[lon, lat]`` pair.
    """
    if len(query_coord) != 2:
        raise ValueError("query_coord must be a [lon, lat] pair.")
    if any(len(c) != 2 for c in result_coords):
        raise ValueError("each result coord must be a [lon, lat] pair.")
    if not result_coords:
        return {
            "n_results": 0,
            "warning": False,
            "message": "No result coordinates supplied; no geographic prior to assess.",
        }

    q = (float(query_coord[0]), float(query_coord[1]))
    distances = [haversine_km(q, (float(c[0]), float(c[1]))) for c in result_coords]
    within = sum(1 for d in distances if d <= radius_km)
    fraction = within / len(distances)
    warning = fraction >= 0.5
    if warning:
        message = (
            f"{within}/{len(distances)} top matches fall within {radius_km:g} km "
            "of the query: the similarity may reflect geographic proximity "
            "(same region/biome), not feature resemblance."
        )
    else:
        message = (
            "Top matches are geographically dispersed; the similarity is "
            "unlikely to be a pure geographic prior."
        )
    return {
        "n_results": len(result_coords),
        "radius_km": radius_km,
        "clustered_fraction": round(fraction, _PLACES),
        "mean_distance_km": round(sum(distances) / len(distances), _PLACES),
        "warning": warning,
        "message": message,
    }
