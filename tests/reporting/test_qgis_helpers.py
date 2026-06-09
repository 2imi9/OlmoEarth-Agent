# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Tests for the QGIS bridge helpers."""

from __future__ import annotations

import xml.etree.ElementTree as ET

from olmoearth_agent.reporting.qgis import (
    build_legend,
    build_raster_sld,
    resolve_xyz_url,
)

_SLD_NS = "http://www.opengis.net/sld"


def test_build_legend_continuous_has_value_color_stops() -> None:
    leg = build_legend("sample_karst_score", vmin=0.0, vmax=1.0)
    assert leg["kind"] == "continuous"
    assert leg["property"] == "sample_karst_score"
    assert leg["vmin"] == 0.0 and leg["vmax"] == 1.0
    # ascending value stops, each with a color, spanning the range
    vals = [s["value"] for s in leg["stops"]]
    assert vals == sorted(vals)
    assert vals[0] == 0.0 and vals[-1] == 1.0
    assert all(s["color"].startswith("#") for s in leg["stops"])


def test_build_legend_matches_the_sld_colors() -> None:
    # the in-chat legend must use the same ramp the QGIS export bakes into the SLD
    leg = build_legend("score", vmin=0.0, vmax=2.0)
    sld = build_raster_sld("score", vmin=0.0, vmax=2.0)
    for stop in leg["stops"]:
        assert f'color="{stop["color"]}"' in sld


def test_build_legend_categorical_echoes_classes() -> None:
    leg = build_legend(
        "landcover",
        classes=[
            {"value": 1, "color": "#1f78b4", "label": "water"},
            {"value": 2, "color": "#33a02c"},  # label defaults to the value
        ],
    )
    assert leg["kind"] == "categorical"
    assert leg["entries"][0] == {"value": 1, "color": "#1f78b4", "label": "water"}
    assert leg["entries"][1]["label"] == "2"


def test_resolve_relative_url() -> None:
    url = resolve_xyz_url(
        "/api/v1/prediction-results/abc/tiles/{z}/{x}/{y}.png?property_name=s",
        base_url="https://olmoearth.allenai.org",
    )
    assert url == (
        "https://olmoearth.allenai.org/api/v1/prediction-results/abc/"
        "tiles/{z}/{x}/{y}.png?property_name=s"
    )
    assert "{z}/{x}/{y}" in url  # QGIS XYZ placeholders preserved


def test_resolve_absolute_url_passthrough() -> None:
    abs_url = "https://host/tiles/{z}/{x}/{y}.png"
    assert resolve_xyz_url(abs_url) == abs_url


def test_build_raster_sld_is_valid_xml() -> None:
    sld = build_raster_sld("karst_score", vmin=0.0, vmax=1.0)
    root = ET.fromstring(sld)  # noqa: S314 - parses our own generated SLD
    assert root.tag == f"{{{_SLD_NS}}}StyledLayerDescriptor"
    # default YlOrRd ramp has 5 stops
    entries = root.findall(f".//{{{_SLD_NS}}}ColorMapEntry")
    assert len(entries) == 5
    quantities = [float(e.get("quantity")) for e in entries]
    assert quantities[0] == 0.0
    assert quantities[-1] == 1.0
    assert quantities == sorted(quantities)  # monotonic


def test_build_raster_sld_custom_range() -> None:
    sld = build_raster_sld("layer", vmin=10.0, vmax=20.0)
    root = ET.fromstring(sld)  # noqa: S314 - parses our own generated SLD
    entries = root.findall(f".//{{{_SLD_NS}}}ColorMapEntry")
    quantities = [float(e.get("quantity")) for e in entries]
    assert quantities[0] == 10.0
    assert quantities[-1] == 20.0
