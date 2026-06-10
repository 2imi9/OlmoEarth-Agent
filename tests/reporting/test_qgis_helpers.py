# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Tests for the QGIS bridge helpers."""

from __future__ import annotations

import xml.etree.ElementTree as ET

from olmoearth_agent.reporting.qgis import (
    AUTH_PLACEHOLDER,
    build_gdal_wms_xml,
    build_legend,
    build_qlr,
    build_raster_sld,
    cog_recipe,
    resolve_xyz_url,
)

_SLD_NS = "http://www.opengis.net/sld"

_TILE = (
    "https://olmoearth.allenai.org/api/v1/prediction-results/abc/"
    "tiles/{z}/{x}/{y}.png?property_name=karst_score"
)


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


def test_build_qlr_is_valid_xml_with_wms_provider() -> None:
    qlr = build_qlr("karst layer", _TILE, crs="EPSG:3857")
    root = ET.fromstring(qlr)  # noqa: S314 - parses our own generated XML
    assert root.tag == "qlr"
    maplayer = root.find(".//maplayer")
    assert maplayer is not None and maplayer.get("type") == "raster"
    assert maplayer.findtext("provider") == "wms"
    assert maplayer.findtext(".//authid") == "EPSG:3857"


def test_build_qlr_never_embeds_the_real_key() -> None:
    qlr = build_qlr("layer", _TILE)
    # the unresolved sentinel stands in for the key; no live bearer value
    assert AUTH_PLACEHOLDER in qlr
    # the tile url is url-encoded inside the datasource so its & / {} survive
    assert "%7Bz%7D" in qlr  # {z} encoded


def test_build_qlr_sanitizes_layer_id() -> None:
    # spaces / punctuation in the name must not produce an invalid id token
    qlr = build_qlr("My Karst (2026)!", _TILE)
    root = ET.fromstring(qlr)  # noqa: S314 - our own XML
    layer_id = root.findtext(".//maplayer/id")
    assert layer_id == "My_Karst_2026_xyz"


def test_build_gdal_wms_xml_rewrites_placeholders_and_is_valid() -> None:
    xml = build_gdal_wms_xml(_TILE, crs="EPSG:3857")
    root = ET.fromstring(xml)  # noqa: S314 - our own XML
    assert root.tag == "GDAL_WMS"
    assert root.find(".//Service").get("name") == "TMS"
    server_url = root.findtext(".//ServerUrl")
    assert "${z}/${x}/${y}" in server_url  # GDAL placeholder form
    assert "{z}/{x}/{y}" not in server_url
    assert root.findtext(".//YOrigin") == "top"
    assert root.findtext("Projection") == "EPSG:3857"


def test_cog_recipe_is_an_honest_user_run_command() -> None:
    recipe = cog_recipe()
    assert "gdal_translate" in recipe["command"]
    assert "-of COG" in recipe["command"]
    # honest about why the agent can't produce a COG itself
    assert "GDAL" in recipe["rationale"]
    assert "full-precision" in recipe["rationale"]
    # the key is supplied at run time via an env var, not written to a file
    assert "$OLMOEARTH_API_KEY" in recipe["command"]
