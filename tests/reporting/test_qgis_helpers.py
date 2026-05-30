# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Tests for the QGIS bridge helpers."""

from __future__ import annotations

import xml.etree.ElementTree as ET

from olmoearth_agent.reporting.qgis import build_raster_sld, resolve_xyz_url

_SLD_NS = "http://www.opengis.net/sld"


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
    root = ET.fromstring(sld)  # noqa: S314 — parses our own generated SLD
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
    root = ET.fromstring(sld)  # noqa: S314 — parses our own generated SLD
    entries = root.findall(f".//{{{_SLD_NS}}}ColorMapEntry")
    quantities = [float(e.get("quantity")) for e in entries]
    assert quantities[0] == 10.0
    assert quantities[-1] == 20.0
