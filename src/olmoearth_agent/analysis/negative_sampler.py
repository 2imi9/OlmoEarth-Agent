# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Pseudo-absence (negative / background) sampling for presence-only labels.

Skill #18 (``olmoearth-negative-sampler``). Pure Python, no GDAL/geopandas.

A *presence-only* label set -- every point is a positive sighting of one
class -- is not trainable on its own: a classifier has no counter-examples, so
it learns to predict the positive class everywhere (pitfall #8 in the vendored
``olmoearth-data-prep``, whose ``audit.py`` hard-FAILs such a set for a missing
negative class). This module generates the missing class as **buffered,
spatially-thinned pseudo-absences** (Barbet-Massin et al. 2012,
doi:10.1111/j.2041-210X.2011.00172.x):

1. *Buffer* -- candidate background locations within ``exclusion_km`` of any
   positive are dropped; a pseudo-absence sitting on top of a presence is just
   label noise.
2. *Thin* -- accepted negatives are kept at least ``min_separation_km`` apart so
   they do not cluster (the same anti-clustering the audit's spatial check
   rewards).
3. *Rank* -- when the caller supplies embeddings for the candidates *and* the
   positives, candidates are ordered by **environmental dissimilarity** to the
   positive centroid -- the inverse of skill #9's similarity search -- so the
   negatives are drawn from genuinely different conditions. Without embeddings,
   selection falls back to deterministic farthest-point spatial dispersion.

Pseudo-absences are a *heuristic*, never verified absences; the dominant failure
mode is **contamination** -- a generated background point landing on an unmapped
positive. Two features keep that honest:

- An optional **contamination guard** (``contamination_threshold``): when
  embeddings are supplied, candidates whose cosine similarity to *any* positive
  meets or exceeds the threshold are dropped as likely hidden positives, before
  ranking. The buffer guards against *spatial* contamination; this guards against
  *environmental* contamination.
- A **quality report** on every result: the chosen negatives' nearest-positive
  distance (min / mean) and, with embeddings, their max similarity-to-positive
  distribution (lower = more credible), so the user can judge -- and tune -- the
  result instead of trusting it blindly.

Coordinates are ``(lon, lat)`` in degrees; distances are great-circle km
(haversine, shared with :mod:`olmoearth_agent.evaluation.spatial_cv`). The
algorithm is fully deterministic -- same inputs, same negatives -- so a
generated dataset is reproducible. Selection is returned as a list of negative
coordinates plus a summary; turning labels into GeoJSON features and files is
the tool layer's job (:mod:`olmoearth_agent.tools.negative_sampler`).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from olmoearth_agent.evaluation.spatial_cv import Point, haversine_km

#: Default exclusion buffer around positives, in km. Not a universal value --
#: it is the spatial scale below which a "background" point is too close to a
#: presence to be a credible absence. Surfaced in the result so the caller can
#: tune it to their sensor resolution / autocorrelation range.
DEFAULT_EXCLUSION_KM = 1.0

#: Default bbox margin (degrees) when the study area is derived from the
#: positives' own extent rather than supplied explicitly.
DEFAULT_MARGIN_DEG = 0.05

#: Default candidate oversampling factor for the auto-generated grid: aim for
#: ``oversample * n_negatives`` candidates so buffering + thinning have room.
DEFAULT_OVERSAMPLE = 8

#: Honest caveat attached to every result's quality report.
PSEUDO_ABSENCE_NOTE = (
    "Pseudo-absences are a heuristic, not verified absences: spot-check that none "
    "coincide with an unmapped positive. Lower similarity_to_positive means more "
    "credible negatives."
)

_MIN_GRID_SIDE = 2
_MAX_GRID_POINTS = 40_000  # guard against a runaway grid on a huge AOI
_PLACES = 6

#: A bounding box ``(min_lon, min_lat, max_lon, max_lat)`` in degrees.
BBox = tuple[float, float, float, float]


def _is_pair(value: Any) -> bool:
    return (
        isinstance(value, Sequence) and not isinstance(value, str) and len(value) == 2
    )


def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity of two equal-length vectors (0.0 if either is zero)."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def bounding_box(
    points: Sequence[Point], margin_deg: float = DEFAULT_MARGIN_DEG
) -> BBox:
    """Return the ``(min_lon, min_lat, max_lon, max_lat)`` extent of ``points``.

    The box is expanded by ``margin_deg`` on every side, which also gives a
    single point (or a collinear set) a non-degenerate area to sample around.

    Raises
    ------
    ValueError
        If ``points`` is empty, a point is not a ``(lon, lat)`` pair, or the
        margin is not positive enough to give a degenerate set an area.
    """
    if not points:
        raise ValueError("points must be non-empty.")
    if any(not _is_pair(p) for p in points):
        raise ValueError("each point must be a [lon, lat] pair.")
    lons = [float(p[0]) for p in points]
    lats = [float(p[1]) for p in points]
    box = (
        min(lons) - margin_deg,
        min(lats) - margin_deg,
        max(lons) + margin_deg,
        max(lats) + margin_deg,
    )
    if box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError("degenerate bounding box; use a positive margin_deg.")
    return box


def _linspace_centers(lo: float, hi: float, n: int) -> list[float]:
    """``n`` evenly-spaced cell centers spanning the open interval ``(lo, hi)``."""
    step = (hi - lo) / n
    return [lo + step * (i + 0.5) for i in range(n)]


def candidate_grid(
    bbox: BBox, *, n_target: int, step_deg: float | None = None
) -> list[Point]:
    """A regular lon/lat grid of candidate background points over ``bbox``.

    If ``step_deg`` is given it sets the spacing directly; otherwise the grid is
    auto-sized to roughly ``n_target`` points, preserving the bbox aspect ratio.

    Raises
    ------
    ValueError
        On a degenerate bbox, a non-positive ``step_deg``, ``n_target < 1``, or a
        grid that would exceed the internal point cap.
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    width, height = max_lon - min_lon, max_lat - min_lat
    if width <= 0 or height <= 0:
        raise ValueError("bbox must have positive width and height.")
    if n_target < 1:
        raise ValueError("n_target must be >= 1.")

    if step_deg is not None:
        if step_deg <= 0:
            raise ValueError("step_deg must be > 0.")
        n_cols = max(_MIN_GRID_SIDE, math.ceil(width / step_deg))
        n_rows = max(_MIN_GRID_SIDE, math.ceil(height / step_deg))
    else:
        ratio = width / height
        n_rows = max(_MIN_GRID_SIDE, round(math.sqrt(n_target / ratio)))
        n_cols = max(_MIN_GRID_SIDE, round(n_target / n_rows))

    if n_cols * n_rows > _MAX_GRID_POINTS:
        raise ValueError(
            f"grid would have {n_cols * n_rows} points (> {_MAX_GRID_POINTS}); "
            "widen step_deg or lower n_target."
        )
    lon_centers = _linspace_centers(min_lon, max_lon, n_cols)
    lat_centers = _linspace_centers(min_lat, max_lat, n_rows)
    return [(lon, lat) for lat in lat_centers for lon in lon_centers]


def _min_distance_to(point: Point, others: Sequence[Point]) -> float:
    """Smallest haversine distance from ``point`` to any of ``others`` (inf if empty)."""
    best = math.inf
    for other in others:
        d = haversine_km(point, other)
        if d < best:
            best = d
    return best


def _max_similarity_to(
    embedding: Sequence[float], positives: Sequence[Sequence[float]]
) -> float:
    """Largest cosine similarity from ``embedding`` to any positive embedding."""
    return max(_cosine_similarity(embedding, pe) for pe in positives)


def _select_by_dissimilarity(
    survivors: list[int],
    pool: Sequence[Point],
    centroid: list[float],
    cand_embeddings: Sequence[Sequence[float]],
    n_negatives: int,
    min_separation_km: float,
) -> list[int]:
    """Pick negatives least similar to the positive centroid, thinning as we go."""
    order = sorted(
        survivors, key=lambda i: (_cosine_similarity(centroid, cand_embeddings[i]), i)
    )
    accepted: list[int] = []
    for i in order:
        if min_separation_km > 0 and any(
            haversine_km(pool[i], pool[j]) < min_separation_km for j in accepted
        ):
            continue
        accepted.append(i)
        if len(accepted) >= n_negatives:
            break
    return accepted


def _select_by_dispersion(
    survivors: list[int],
    pool: Sequence[Point],
    positives: Sequence[Point],
    n_negatives: int,
    min_separation_km: float,
) -> list[int]:
    """Farthest-point selection: seed at the most-absent point, then maximize spread."""
    dist_to_positives = {i: _min_distance_to(pool[i], positives) for i in survivors}
    # Seed with the survivor farthest from every positive (ties -> lowest index).
    first = max(survivors, key=lambda i: (dist_to_positives[i], -i))
    accepted = [first]
    pending = [i for i in survivors if i != first]
    while len(accepted) < n_negatives and pending:
        best_idx: int | None = None
        best_sep = -math.inf
        for i in pending:
            sep = min(haversine_km(pool[i], pool[j]) for j in accepted)
            if min_separation_km > 0 and sep < min_separation_km:
                continue
            if sep > best_sep:
                best_sep, best_idx = sep, i
        if best_idx is None:
            break  # every remaining candidate is inside the thinning radius
        accepted.append(best_idx)
        pending.remove(best_idx)
    return accepted


def sample_negatives(
    positives: Sequence[Point],
    *,
    candidates: Sequence[Point] | None = None,
    bbox: BBox | None = None,
    n_negatives: int | None = None,
    exclusion_km: float = DEFAULT_EXCLUSION_KM,
    min_separation_km: float | None = None,
    positive_embeddings: Sequence[Sequence[float]] | None = None,
    candidate_embeddings: Sequence[Sequence[float]] | None = None,
    margin_deg: float = DEFAULT_MARGIN_DEG,
    grid_step_deg: float | None = None,
    oversample: int = DEFAULT_OVERSAMPLE,
    contamination_threshold: float | None = None,
) -> dict[str, Any]:
    """Generate buffered, thinned pseudo-absences for a presence-only label set.

    Parameters
    ----------
    positives
        The presence locations, ``(lon, lat)`` in degrees.
    candidates
        Optional explicit background candidate locations. When omitted, a grid
        is generated over ``bbox`` (or the positives' own extent + ``margin_deg``).
    bbox
        Optional ``(min_lon, min_lat, max_lon, max_lat)`` study area for the grid.
    n_negatives
        Target number of negatives. Defaults to ``len(positives)`` (a balanced
        1:1 set, which clears the data-prep audit's per-class and class-ratio
        checks).
    exclusion_km
        Drop any candidate within this distance of *any* positive.
    min_separation_km
        Keep accepted negatives at least this far apart. Defaults to
        ``exclusion_km``; set ``0`` to disable thinning.
    positive_embeddings, candidate_embeddings
        Optional parallel embedding vectors. When *both* are supplied (and
        ``candidates`` is explicit), negatives are ranked by environmental
        dissimilarity to the positive centroid (inverts skill #9). Otherwise
        selection uses farthest-point spatial dispersion.
    margin_deg
        Bbox margin when the study area is derived from the positives' extent.
    grid_step_deg
        Optional explicit grid spacing (degrees); else auto-sized.
    oversample
        Candidate-grid oversampling factor relative to ``n_negatives``.
    contamination_threshold
        Optional cosine value in ``[-1, 1]``. When embeddings are supplied, any
        candidate whose max similarity to a positive is ``>=`` this value is
        dropped as a likely *unmapped positive* (environmental contamination)
        before ranking. ``None`` (default) disables the guard; the quality
        report still reports the similarity distribution so a threshold can be
        chosen. No effect without embeddings.

    Returns
    -------
    dict
        ``negatives`` (the selected ``[lon, lat]`` list) plus a JSON-able
        summary: counts (including ``n_contamination_excluded``), the ranking
        mode actually used, the buffer / thinning radii, the study bbox (for a
        grid), the positive:negative balance, a ``quality`` report (buffer-km +
        similarity-to-positive stats + an honest caveat), and any ``warnings``.

    Raises
    ------
    ValueError
        On empty/malformed positives, negative radii, ``n_negatives < 1``,
        ``oversample < 1``, a ``contamination_threshold`` out of ``[-1, 1]``, or
        embedding inputs that do not align with their coordinates.
    """
    if not positives:
        raise ValueError("positives must be non-empty.")
    if any(not _is_pair(p) for p in positives):
        raise ValueError("each positive must be a [lon, lat] pair.")
    pos: list[Point] = [(float(p[0]), float(p[1])) for p in positives]

    if exclusion_km < 0:
        raise ValueError("exclusion_km must be >= 0.")
    if oversample < 1:
        raise ValueError("oversample must be >= 1.")
    if contamination_threshold is not None and not (
        -1.0 <= float(contamination_threshold) <= 1.0
    ):
        raise ValueError("contamination_threshold must be a cosine value in [-1, 1].")
    n_target_neg = len(pos) if n_negatives is None else int(n_negatives)
    if n_target_neg < 1:
        raise ValueError("n_negatives must be >= 1.")
    sep_km = exclusion_km if min_separation_km is None else float(min_separation_km)
    if sep_km < 0:
        raise ValueError("min_separation_km must be >= 0.")

    if positive_embeddings is not None and len(positive_embeddings) != len(pos):
        raise ValueError("positive_embeddings length must match positives length.")

    # ---- Build the candidate pool -----------------------------------------
    warnings: list[str] = []
    used_bbox: BBox | None = None
    if candidates is not None:
        if any(not _is_pair(c) for c in candidates):
            raise ValueError("each candidate must be a [lon, lat] pair.")
        pool: list[Point] = [(float(c[0]), float(c[1])) for c in candidates]
        if candidate_embeddings is not None and len(candidate_embeddings) != len(pool):
            raise ValueError(
                "candidate_embeddings length must match candidates length."
            )
        cand_emb = (
            [list(map(float, e)) for e in candidate_embeddings]
            if candidate_embeddings is not None
            else None
        )
        source = "provided"
    else:
        if candidate_embeddings is not None:
            raise ValueError("candidate_embeddings requires explicit candidates.")
        used_bbox = bbox if bbox is not None else bounding_box(pos, margin_deg)
        pool = candidate_grid(
            used_bbox, n_target=oversample * n_target_neg, step_deg=grid_step_deg
        )
        cand_emb = None
        source = "grid"

    if not pool:
        raise ValueError("no candidate background points to sample from.")

    # ---- Buffer: drop candidates too close to any positive ----------------
    survivors: list[int] = []
    for i, c in enumerate(pool):
        if exclusion_km <= 0 or _min_distance_to(c, pos) >= exclusion_km:
            survivors.append(i)
    n_excluded = len(pool) - len(survivors)
    if not survivors:
        warnings.append(
            "every candidate fell inside the exclusion buffer; lower exclusion_km "
            "or supply candidates farther from the positives."
        )
        return _summary(
            pos,
            [],
            pool,
            n_target_neg,
            n_excluded,
            0,
            "none",
            exclusion_km,
            sep_km,
            source,
            used_bbox,
            warnings,
        )

    # ---- Rank + select ----------------------------------------------------
    n_contam = 0
    sims_accepted: list[float] | None = None
    if cand_emb is not None and positive_embeddings is not None:
        dim = len(cand_emb[survivors[0]])
        if any(len(e) != dim for e in positive_embeddings):
            raise ValueError(
                "positive_embeddings dimensionality must match candidate_embeddings."
            )
        pos_embs = [[float(x) for x in e] for e in positive_embeddings]
        # Max similarity of each surviving candidate to ANY positive.
        max_sim = {i: _max_similarity_to(cand_emb[i], pos_embs) for i in survivors}

        # Contamination guard: drop candidates that resemble a positive too
        # closely (likely unmapped positives) before ranking.
        if contamination_threshold is not None:
            thr = float(contamination_threshold)
            kept = [i for i in survivors if max_sim[i] < thr]
            n_contam = len(survivors) - len(kept)
            survivors = kept
            if not survivors:
                warnings.append(
                    f"all {n_contam} buffered candidates met the contamination "
                    f"threshold ({thr:g}); they resemble known positives too "
                    "closely. Lower the threshold or supply more diverse candidates."
                )
                return _summary(
                    pos,
                    [],
                    pool,
                    n_target_neg,
                    n_excluded,
                    0,
                    "embedding_dissimilarity",
                    exclusion_km,
                    sep_km,
                    source,
                    used_bbox,
                    warnings,
                    n_contamination_excluded=n_contam,
                )

        centroid = [sum(e[d] for e in pos_embs) / len(pos_embs) for d in range(dim)]
        accepted = _select_by_dissimilarity(
            survivors, pool, centroid, cand_emb, n_target_neg, sep_km
        )
        ranking = "embedding_dissimilarity"
        sims_accepted = [max_sim[i] for i in accepted]
    else:
        if contamination_threshold is not None:
            warnings.append(
                "contamination_threshold was set but no embeddings were supplied, "
                "so the (embedding-based) contamination guard had no effect."
            )
        accepted = _select_by_dispersion(survivors, pool, pos, n_target_neg, sep_km)
        ranking = "spatial_dispersion"

    if len(accepted) < n_target_neg:
        warnings.append(
            f"placed {len(accepted)} of {n_target_neg} requested negatives "
            "(buffer/thinning/contamination/extent too tight); loosen exclusion_km "
            "or min_separation_km, raise contamination_threshold, widen the AOI, or "
            "provide more candidates."
        )
    return _summary(
        pos,
        accepted,
        pool,
        n_target_neg,
        n_excluded,
        len(survivors),
        ranking,
        exclusion_km,
        sep_km,
        source,
        used_bbox,
        warnings,
        n_contamination_excluded=n_contam,
        sims_accepted=sims_accepted,
    )


def _quality_report(
    accepted_points: list[Point],
    positives: Sequence[Point],
    sims: list[float] | None,
) -> dict[str, Any]:
    """Inspectable credibility stats for the chosen negatives (+ honest caveat)."""
    report: dict[str, Any] = {"note": PSEUDO_ABSENCE_NOTE}
    if not accepted_points:
        return report
    buffer_dists = [_min_distance_to(p, positives) for p in accepted_points]
    report["min_buffer_km"] = round(min(buffer_dists), 3)
    report["mean_buffer_km"] = round(sum(buffer_dists) / len(buffer_dists), 3)
    if sims:
        report["similarity_to_positive"] = {
            "min": round(min(sims), _PLACES),
            "mean": round(sum(sims) / len(sims), _PLACES),
            "max": round(max(sims), _PLACES),
        }
    return report


def _summary(
    positives: list[Point],
    accepted: list[int],
    pool: Sequence[Point],
    n_requested: int,
    n_excluded: int,
    n_survivors: int,
    ranking: str,
    exclusion_km: float,
    min_separation_km: float,
    source: str,
    used_bbox: BBox | None,
    warnings: list[str],
    *,
    n_contamination_excluded: int = 0,
    sims_accepted: list[float] | None = None,
) -> dict[str, Any]:
    """Assemble the JSON-able result envelope."""
    negatives = [
        [round(pool[i][0], _PLACES), round(pool[i][1], _PLACES)] for i in accepted
    ]
    accepted_points = [pool[i] for i in accepted]
    balance = round(len(positives) / len(accepted), 3) if accepted else None
    return {
        "n_positives": len(positives),
        "n_negatives": len(accepted),
        "n_requested": n_requested,
        "negatives": negatives,
        "candidate_source": source,
        "n_candidates": len(pool),
        "n_excluded_by_buffer": n_excluded,
        "n_contamination_excluded": n_contamination_excluded,
        "n_survivors": n_survivors,
        "ranking": ranking,
        "exclusion_km": exclusion_km,
        "min_separation_km": min_separation_km,
        "bbox": list(used_bbox) if used_bbox is not None else None,
        "positive_to_negative_ratio": balance,
        "quality": _quality_report(accepted_points, positives, sims_accepted),
        "warnings": warnings,
    }
