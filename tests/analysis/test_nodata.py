# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""The shared no-data rule, declared outputs and grid windows."""

from __future__ import annotations

import math

from olmoearth_agent.analysis.raster_compare import (
    band_is_nodata,
    declared_fields,
    declared_range,
    grid_points,
    grid_windows,
)

#: The band record Studio returned at a no-data point of the trial's KarstBinary result.
_KARST_NODATA = {
    "band_index": 1,
    "property_name": "sample_karst_score",
    "raw_value": -1.0,
    "classification": None,
    "regression": {"min_value": 0.0, "max_value": 1.0, "colormap_name": "viridis"},
}


def _band(value: float | None, **extra: object) -> dict[str, object]:
    band: dict[str, object] = {"property_name": "p", "raw_value": value}
    band["classification"] = None
    band.update(extra)
    return band


def test_the_trial_sentinel_is_nodata_although_the_model_declares_none() -> None:
    assert band_is_nodata(_KARST_NODATA, nodata_value=None) is True
    assert band_is_nodata({**_KARST_NODATA, "raw_value": 0.42}) is False
    # The declared range is inclusive.
    assert band_is_nodata({**_KARST_NODATA, "raw_value": 0.0}) is False
    assert band_is_nodata({**_KARST_NODATA, "raw_value": 1.0}) is False
    assert band_is_nodata({**_KARST_NODATA, "raw_value": 1.2}) is True


def test_missing_nan_and_model_nodata_value_are_nodata() -> None:
    assert band_is_nodata(None) is True
    assert band_is_nodata({}) is True
    assert band_is_nodata(_band(None)) is True
    assert band_is_nodata(_band(math.nan)) is True
    # No declared range: only the model's nodata_value can mark a sentinel.
    assert band_is_nodata(_band(255.0)) is False
    assert band_is_nodata(_band(255.0), nodata_value=255) is True
    assert band_is_nodata(_band(3.0), nodata_value=255) is False


def test_the_result_metadata_range_is_the_fallback() -> None:
    declared = {"value_type": "regression", "min_value": 0.0, "max_value": 10.0}
    assert band_is_nodata(_band(-1.0), declared=declared) is True
    assert band_is_nodata(_band(7.5), declared=declared) is False
    assert declared_range(_band(7.5), declared) == (0.0, 10.0)
    # The band's own block wins over the result's declaration.
    band = _band(7.5, regression={"min_value": 0.0, "max_value": 1.0})
    assert declared_range(band, declared) == (0.0, 1.0)


def test_classification_bands() -> None:
    labelled = _band(3.0, classification={"label": "forest", "color": [0, 1, 0, 255]})
    assert band_is_nodata(labelled) is False
    declared = {"value_type": "classification", "n_classes": 2}
    # A raw value with no class on a declared classification band is off-legend.
    assert band_is_nodata(_band(255.0), declared=declared) is True


def test_declared_fields_reads_result_metadata_without_geometry() -> None:
    record = {
        "result_metadata": {
            "geometry": {"type": "Point", "coordinates": [1.0, 2.0]},
            "regression_fields": [
                {"property_name": "sample_karst_score", "min_value": 0, "max_value": 1}
            ],
            "classification_fields": [
                {
                    "property_name": "landcover",
                    "allowed_values": [
                        {"value": 1, "label": "water", "color": [0, 0, 255]},
                        {"value": 2, "label": "forest", "color": [0, 255, 0]},
                    ],
                    "confidence_property_name": "landcover_conf",
                }
            ],
        }
    }
    fields = declared_fields(record)
    assert fields["sample_karst_score"] == {
        "value_type": "regression",
        "min_value": 0.0,
        "max_value": 1.0,
    }
    assert fields["landcover"]["classes"] == ["water", "forest"]
    assert fields["landcover"]["confidence_property_name"] == "landcover_conf"
    assert "coordinates" not in repr(fields)
    assert declared_fields({}) == {}


def test_grid_windows_are_row_major_from_the_north_west() -> None:
    bbox = [0.0, 0.0, 4.0, 2.0]
    windows = grid_windows(bbox, 2)
    assert [(r, c) for r, c, _lon, _lat in windows] == [(0, 0), (0, 1), (1, 0), (1, 1)]
    # Row 0 is the northern row, col 0 the western column.
    assert windows[0][3] > windows[2][3]
    assert windows[0][2] < windows[1][2]
    # The same cell centres grid_points samples, only reordered.
    assert {(lo, la) for _r, _c, lo, la in windows} == set(grid_points(bbox, 2))
