# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Tests for the olmoearth_qgis_bridge tool."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest

from olmoearth_agent.harness.state import ThreadState
from olmoearth_agent.reporting.qgis import AUTH_PLACEHOLDER
from olmoearth_agent.tools.qgis import build_qgis_tools
from olmoearth_agent.tools.registry import ToolContext


@pytest.mark.asyncio
async def test_qgis_bridge_tool() -> None:
    tool = build_qgis_tools()[0]
    ctx = ToolContext(studio=None, state=ThreadState())  # type: ignore[arg-type]
    result = await tool.handler(
        {
            "tile_urls": [
                "/api/v1/prediction-results/abc/tiles/{z}/{x}/{y}.png?property_name=s"
            ],
            "layer_name": "karst",
        },
        ctx,
    )
    assert result["layer_name"] == "karst"
    assert result["xyz_urls"][0].startswith("https://olmoearth.allenai.org/")
    assert "{z}/{x}/{y}" in result["xyz_urls"][0]
    # SLD is valid XML
    ET.fromstring(result["sld"])  # noqa: S314 - parses our own generated SLD
    # Auth guidance is present, but the key is never embedded -- the
    # instructions point at the placeholder sentinel, not a baked-in token.
    assert any(AUTH_PLACEHOLDER in line for line in result["instructions"])
    # provider files are present and well-formed, key NOT embedded
    qlr_root = ET.fromstring(result["qlr"])  # noqa: S314 - our own XML
    assert qlr_root.tag == "qlr"
    gdal_root = ET.fromstring(result["gdal_wms_xml"])  # noqa: S314 - our own XML
    assert gdal_root.tag == "GDAL_WMS"
    # the honest COG recipe is a user-run command, not a fabricated file
    assert "gdal_translate" in result["cog_recipe"]["command"]
    assert "-of COG" in result["cog_recipe"]["command"]


@pytest.mark.asyncio
async def test_qgis_bridge_does_not_embed_the_key() -> None:
    tool = build_qgis_tools()[0]
    ctx = ToolContext(studio=None, state=ThreadState())  # type: ignore[arg-type]
    result = await tool.handler(
        {
            "tile_urls": [
                "https://olmoearth.allenai.org/api/v1/prediction-results/abc/"
                "tiles/{z}/{x}/{y}.png?property_name=s"
            ],
            "layer_name": "karst",
        },
        ctx,
    )
    # the unresolved sentinel is present; no real bearer value is baked in
    assert "__OLMOEARTH_API_KEY__" in result["qlr"]
    # GDAL descriptor rewrites {z}/{x}/{y} -> ${z}/${x}/${y}
    assert "${z}/${x}/${y}" in result["gdal_wms_xml"]
    assert "{z}/{x}/{y}" not in result["gdal_wms_xml"]
