# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Ziming Qi
"""
audit.py — Run the 7 OE quality criteria against a labels GeoJSON.

Supports all three verified schemas:
- Studio import: properties.sample_category   (what you upload to Studio)
- Studio export: properties.es_label          (what Studio writes for olmoearth_run)
- rslearn / AWF: properties.oe_labels.category (older fine-tune flow)

Usage:
    python audit.py path/to/labels.geojson
    python audit.py path/to/dir_containing_geojson/

Exits 0 if all criteria PASS or WARN, 1 if any FAIL.

Stdlib only. shapely is optional; if missing, polygon validity check is skipped
with a WARN rather than a FAIL.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

# Thresholds chosen to match what's been useful across the Karst / Chesapeake /
# Potomac case studies. Adjust if your project has reasoned exceptions.
MIN_VOLUME_FAIL = 50
MIN_VOLUME_WARN = 200
MIN_PER_CLASS_FAIL = 10
MIN_PER_CLASS_WARN = 30
MAX_CLASS_RATIO_WARN = 10
SPATIAL_CLUSTER_WARN = 0.7

NEGATIVE_CLASS_NAMES = {
    "other", "background", "negative", "stable", "none", "no_event", "non_event"
}


def _load_geojson(path: Path) -> list[dict]:
    if path.is_dir():
        candidates = sorted(path.glob("**/*.geojson"))
        if not candidates:
            raise SystemExit(f"No .geojson found under {path}")
        path = candidates[0]
        print(f"Auditing first .geojson found: {path}\n")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("features", [])


def _get_label(feature: dict):
    """Pull the class label, supporting all 3 verified schemas.

    - Studio import: ``properties.sample_category``
    - Studio export: ``properties.es_label``
    - rslearn / AWF: ``properties.oe_labels.category``

    Returns whichever is present (in that priority order), or None.
    """
    props = feature.get("properties") or {}
    if props.get("sample_category") is not None:
        return props["sample_category"]
    if props.get("es_label") is not None:
        return props["es_label"]
    oe_labels = props.get("oe_labels")
    if isinstance(oe_labels, dict) and oe_labels.get("category") is not None:
        return oe_labels["category"]
    return None


def _detect_schema(features: list[dict]) -> tuple[str | None, int]:
    """Return (schema_name, count) for the most-populated of the three verified
    schemas, or (None, 0) if no recognized label field is present.

    Schema names: 'studio_import' | 'studio_export' | 'rslearn'.
    """
    studio_import = sum(
        1 for f in features
        if (f.get("properties") or {}).get("sample_category") is not None
    )
    studio_export = sum(
        1 for f in features
        if (f.get("properties") or {}).get("es_label") is not None
    )
    rslearn = sum(
        1 for f in features
        if isinstance((f.get("properties") or {}).get("oe_labels"), dict)
        and (f["properties"]["oe_labels"]).get("category") is not None
    )
    counts = {
        "studio_import": studio_import,
        "studio_export": studio_export,
        "rslearn": rslearn,
    }
    best = max(counts, key=counts.get)
    if counts[best] == 0:
        return None, 0
    return best, counts[best]


def _coords(feature: dict):
    geom = feature.get("geometry") or {}
    if geom.get("type") == "Point":
        c = geom.get("coordinates")
        if isinstance(c, list) and len(c) >= 2:
            return float(c[0]), float(c[1])
    return None


def _kmeans_lite(points, k=4, iters=10):
    """Tiny k-means in pure stdlib for the spatial clustering check."""
    import random
    if not points or len(points) < k:
        return [0] * len(points)
    random.seed(0)
    centroids = random.sample(points, k)
    assignments = [0] * len(points)
    for _ in range(iters):
        for i, p in enumerate(points):
            best, best_d = 0, float("inf")
            for j, c in enumerate(centroids):
                d = (p[0] - c[0]) ** 2 + (p[1] - c[1]) ** 2
                if d < best_d:
                    best_d, best = d, j
            assignments[i] = best
        new_centroids = []
        for j in range(k):
            members = [p for p, a in zip(points, assignments) if a == j]
            if members:
                cx = sum(m[0] for m in members) / len(members)
                cy = sum(m[1] for m in members) / len(members)
                new_centroids.append((cx, cy))
            else:
                new_centroids.append(centroids[j])
        centroids = new_centroids
    return assignments


def check_volume(features):
    n = len(features)
    if n < MIN_VOLUME_FAIL:
        return "FAIL", f"{n} samples (< {MIN_VOLUME_FAIL})"
    if n < MIN_VOLUME_WARN:
        return "WARN", f"{n} samples (< {MIN_VOLUME_WARN}, more is better within reason)"
    return "PASS", f"{n} samples"


def check_schema(features):
    if not features:
        return "FAIL", "no features"
    schema, count = _detect_schema(features)
    if schema is None:
        return (
            "FAIL",
            "no recognized label field — expected properties.sample_category "
            "(Studio import), properties.es_label (Studio export), or "
            "properties.oe_labels.category (rslearn / AWF)",
        )
    field = {
        "studio_import": "sample_category",
        "studio_export": "es_label",
        "rslearn": "oe_labels.category",
    }[schema]
    missing = sum(1 for f in features if _get_label(f) is None)
    if missing > 0:
        return (
            "WARN",
            f"{missing}/{len(features)} features missing label "
            f"({schema} schema detected via properties.{field}; {count} have it)",
        )
    return (
        "PASS",
        f"all {len(features)} features have properties.{field} ({schema} schema)",
    )


def check_class_distribution(features):
    labels = [_get_label(f) for f in features if _get_label(f) is not None]
    if not labels:
        return "FAIL", "no labels found"
    counts = Counter(labels)
    if len(counts) < 2:
        only = next(iter(counts))
        return "WARN", f"only 1 class present ({only!r}); a classifier needs >= 2"
    most = max(counts.values())
    least = min(counts.values())
    ratio = most / least
    if ratio > MAX_CLASS_RATIO_WARN:
        return (
            "WARN",
            f"max/min class ratio = {ratio:.1f} (> {MAX_CLASS_RATIO_WARN}); "
            f"consider equal-frequency binning to rebalance",
        )
    return "PASS", f"{len(counts)} classes, max/min ratio = {ratio:.1f}"


def check_per_class_volume(features):
    labels = [_get_label(f) for f in features if _get_label(f) is not None]
    if not labels:
        return "FAIL", "no labels found"
    counts = Counter(labels)
    bad = [(l, c) for l, c in counts.items() if c < MIN_PER_CLASS_FAIL]
    warn = [(l, c) for l, c in counts.items() if MIN_PER_CLASS_FAIL <= c < MIN_PER_CLASS_WARN]
    if bad:
        names = ", ".join(f"{l!r}={c}" for l, c in bad)
        return "FAIL", f"classes under {MIN_PER_CLASS_FAIL} samples: {names}"
    if warn:
        names = ", ".join(f"{l!r}={c}" for l, c in warn)
        return "WARN", f"classes under {MIN_PER_CLASS_WARN} samples: {names}"
    return "PASS", f"all classes have >= {MIN_PER_CLASS_WARN} samples"


def check_negative_class(features):
    labels = {str(_get_label(f)).lower() for f in features if _get_label(f) is not None}
    found = labels & NEGATIVE_CLASS_NAMES
    if found:
        return "PASS", f"negative class present: {sorted(found)}"
    return (
        "FAIL",
        f"no negative class found (looked for: {sorted(NEGATIVE_CLASS_NAMES)}); "
        f"add a 'stable' / 'background' class to prevent false positives",
    )


def check_spatial_distribution(features):
    points = [_coords(f) for f in features]
    points = [p for p in points if p is not None]
    if len(points) < 20:
        return "WARN", f"too few points ({len(points)}) to assess spatial clustering"
    assignments = _kmeans_lite(points, k=4)
    counts = Counter(assignments)
    largest = max(counts.values()) / len(points)
    if largest > SPATIAL_CLUSTER_WARN:
        return (
            "WARN",
            f"{largest:.0%} of points fall in 1 of 4 k-means clusters "
            f"(> {SPATIAL_CLUSTER_WARN:.0%}); spread labels across the AOI",
        )
    return "PASS", f"largest of 4 clusters holds {largest:.0%} of points"


def check_polygon_cleanliness(features):
    polygons = [
        f for f in features
        if (f.get("geometry") or {}).get("type") in ("Polygon", "MultiPolygon")
    ]
    if not polygons:
        return "PASS", "no polygons (point dataset)"
    try:
        from shapely.geometry import shape
    except ImportError:
        return (
            "WARN",
            f"{len(polygons)} polygons present but shapely not installed; "
            f"`pip install shapely` to enable validity checks",
        )
    invalid = []
    for f in polygons:
        try:
            g = shape(f["geometry"])
            if not g.is_valid:
                invalid.append(f.get("id", "?"))
        except Exception as e:
            invalid.append(f"err: {e}")
    if invalid:
        return (
            "FAIL",
            f"{len(invalid)}/{len(polygons)} invalid polygons "
            f"(self-intersecting or malformed)",
        )
    return "PASS", f"all {len(polygons)} polygons valid"


CHECKS = [
    ("Volume", check_volume),
    ("Schema (sample_category / es_label / oe_labels.category)", check_schema),
    ("Class distribution", check_class_distribution),
    ("Per-class volume", check_per_class_volume),
    ("Negative class", check_negative_class),
    ("Spatial distribution", check_spatial_distribution),
    ("Polygon cleanliness", check_polygon_cleanliness),
]


def main():
    parser = argparse.ArgumentParser(
        description="Run the 7 OE quality criteria on a labels GeoJSON.",
    )
    parser.add_argument("path", help="GeoJSON file or directory containing one")
    args = parser.parse_args()

    features = _load_geojson(Path(args.path))
    print(f"Loaded {len(features)} features\n")
    print("-" * 72)

    any_fail = False
    for name, fn in CHECKS:
        try:
            status, detail = fn(features)
        except Exception as e:
            status, detail = "ERROR", repr(e)
        markers = {
            "PASS": "[PASS]", "WARN": "[WARN]", "FAIL": "[FAIL]", "ERROR": "[ERR ]",
        }
        print(f"{markers[status]} {name}: {detail}")
        if status in ("FAIL", "ERROR"):
            any_fail = True

    print("-" * 72)
    print("Result:", "FAIL" if any_fail else "OK")
    sys.exit(1 if any_fail else 0)


if __name__ == "__main__":
    main()
