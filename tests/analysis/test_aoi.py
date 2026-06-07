# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Tests for the pure AOI geometry helpers."""

from __future__ import annotations

import pytest

from olmoearth_agent.analysis.aoi import (
    bbox_to_polygon,
    geometry_bbox,
    validate_polygon_geometry,
)

_POLY = {
    "type": "Polygon",
    "coordinates": [
        [[-122.34, 47.62], [-122.30, 47.62], [-122.30, 47.65], [-122.34, 47.65], [-122.34, 47.62]]
    ],
}
_MULTIPOLY = {
    "type": "MultiPolygon",
    "coordinates": [
        [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]]],
        [[[2.0, 2.0], [3.0, 2.0], [3.0, 3.5], [2.0, 3.5], [2.0, 2.0]]],
    ],
}


def test_validate_accepts_polygon_and_multipolygon() -> None:
    assert validate_polygon_geometry(_POLY) is _POLY
    assert validate_polygon_geometry(_MULTIPOLY) is _MULTIPOLY


@pytest.mark.parametrize(
    "geom",
    [
        {"type": "Point", "coordinates": [0.0, 0.0]},
        {"type": "LineString", "coordinates": [[0.0, 0.0], [1.0, 1.0]]},
        {"type": "Polygon"},  # no coordinates
        {"type": "Polygon", "coordinates": []},  # empty
        "not-a-dict",
        {"coordinates": [[[0, 0]]]},  # no type
    ],
)
def test_validate_rejects_non_polygons(geom: object) -> None:
    with pytest.raises(ValueError):
        validate_polygon_geometry(geom)


def test_geometry_bbox_polygon() -> None:
    assert geometry_bbox(_POLY) == [-122.34, 47.62, -122.30, 47.65]


def test_geometry_bbox_multipolygon_spans_all_parts() -> None:
    assert geometry_bbox(_MULTIPOLY) == [0.0, 0.0, 3.0, 3.5]


def test_geometry_bbox_empty_raises() -> None:
    with pytest.raises(ValueError):
        geometry_bbox({"type": "Polygon", "coordinates": []})


def test_bbox_to_polygon_round_trips_through_bbox() -> None:
    geom = bbox_to_polygon(-1.0, -2.0, 3.0, 4.0)
    assert geom["type"] == "Polygon"
    validate_polygon_geometry(geom)
    # ring is closed (first == last)
    ring = geom["coordinates"][0]
    assert ring[0] == ring[-1]
    assert geometry_bbox(geom) == [-1.0, -2.0, 3.0, 4.0]
