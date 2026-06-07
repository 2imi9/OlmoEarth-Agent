# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Pure-Python helpers for area-of-interest (AOI) GeoJSON geometries.

The AOI-draw-in-chat flow captures a polygon the user drew on a map and
hands it to Studio (``POST /areas``) and to bbox-driven skills. These
helpers validate that geometry and derive its bounding box, with no GDAL
or numpy dependency (the agent stays torch-free / no-GDAL): a drawn AOI is
a small GeoJSON ``Polygon`` / ``MultiPolygon``, so plain coordinate
arithmetic is enough.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

#: Studio areas accept only these geometry types ("Areas must be polygons").
POLYGON_TYPES = ("Polygon", "MultiPolygon")


def _iter_positions(coords: Any) -> Iterator[tuple[float, float]]:
    """Yield ``(lon, lat)`` pairs from an arbitrarily nested coordinate array.

    A GeoJSON position is ``[lon, lat]`` (optionally with elevation); rings
    nest them in lists (Polygon = list of rings; MultiPolygon = list of
    polygons). Recursing to the first numeric pair handles every depth.
    """
    if (
        isinstance(coords, (list, tuple))
        and len(coords) >= 2
        and all(isinstance(c, (int, float)) for c in coords[:2])
    ):
        yield (float(coords[0]), float(coords[1]))
        return
    if isinstance(coords, (list, tuple)):
        for item in coords:
            yield from _iter_positions(item)


def validate_polygon_geometry(geom: Any) -> dict[str, Any]:
    """Return ``geom`` if it is a valid GeoJSON Polygon/MultiPolygon, else raise.

    Studio areas reject non-polygon geometries, so we fail fast (before the
    network call) with a message the agent can relay.

    Raises
    ------
    ValueError
        If ``geom`` is not a dict, is not a Polygon/MultiPolygon, or has no
        usable coordinates.
    """
    if not isinstance(geom, dict):
        raise ValueError("geometry must be a GeoJSON object")
    gtype = geom.get("type")
    if gtype not in POLYGON_TYPES:
        raise ValueError(
            f"geometry type must be Polygon or MultiPolygon, got {gtype!r}"
        )
    coords = geom.get("coordinates")
    if not isinstance(coords, (list, tuple)) or not coords:
        raise ValueError("geometry has no coordinates")
    if next(_iter_positions(coords), None) is None:
        raise ValueError("geometry has no numeric positions")
    return geom


def geometry_bbox(geom: Any) -> list[float]:
    """Return ``[min_lon, min_lat, max_lon, max_lat]`` for a GeoJSON geometry.

    Works for any geometry type (walks every position), but the AOI flow
    only feeds it validated polygons. The bbox is what bbox-driven skills
    (e.g. ``--bbox`` latent-change) consume.

    Raises
    ------
    ValueError
        If the geometry contains no positions.
    """
    lons: list[float] = []
    lats: list[float] = []
    for lon, lat in _iter_positions(geom.get("coordinates") if isinstance(geom, dict) else geom):
        lons.append(lon)
        lats.append(lat)
    if not lons:
        raise ValueError("geometry has no positions to bound")
    return [min(lons), min(lats), max(lons), max(lats)]


def bbox_to_polygon(
    min_lon: float, min_lat: float, max_lon: float, max_lat: float
) -> dict[str, Any]:
    """Build a closed GeoJSON ``Polygon`` rectangle from a bbox.

    Convenience for callers that have a bbox (e.g. a typed extent) and need a
    polygon ``geom`` for ``POST /areas``. Ring is counter-clockwise and
    explicitly closed (first == last), per the GeoJSON spec.
    """
    return {
        "type": "Polygon",
        "coordinates": [
            [
                [min_lon, min_lat],
                [max_lon, min_lat],
                [max_lon, max_lat],
                [min_lon, max_lat],
                [min_lon, min_lat],
            ]
        ],
    }
