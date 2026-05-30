# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Tests for the olmoearth_qgis_bridge tool."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest

from olmoearth_agent.harness.state import ThreadState
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
    ET.fromstring(result["sld"])  # noqa: S314 — parses our own generated SLD
    assert any("Authorization: Bearer" in line for line in result["instructions"])
