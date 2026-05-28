# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 OlmoEarth Agent contributors
"""The olmoearth-qgis-bridge tool bundle (skill #12).

Turn a prediction result's tile templates into QGIS-ready XYZ URLs + an
OGC SLD style + load instructions, so a GIS analyst can pull the layer
into QGIS without leaving the desktop.
"""

from __future__ import annotations

from typing import Any

from olmoearth_agent.llm.types import ToolSpec
from olmoearth_agent.reporting.qgis import (
    DEFAULT_BASE_URL,
    build_raster_sld,
    resolve_xyz_url,
)
from olmoearth_agent.tools.registry import RegisteredTool, ToolContext


async def _qgis_bridge(args: dict[str, Any], _ctx: ToolContext) -> dict[str, Any]:
    base_url = args.get("base_url", DEFAULT_BASE_URL)
    layer_name = args.get("layer_name", "olmoearth_layer")
    tile_urls = args["tile_urls"]
    xyz_urls = [resolve_xyz_url(t, base_url) for t in tile_urls]
    sld = build_raster_sld(
        layer_name,
        vmin=float(args.get("vmin", 0.0)),
        vmax=float(args.get("vmax", 1.0)),
    )
    return {
        "layer_name": layer_name,
        "xyz_urls": xyz_urls,
        "sld": sld,
        "instructions": [
            "QGIS → Layer → Add Layer → Add XYZ Layer; paste an xyz_url.",
            "Tiles require auth: add an HTTP header "
            "'Authorization: Bearer <OLMOEARTH_API_KEY>' to the XYZ "
            "connection (QGIS ≥3: Authentication config, or the "
            "connection's HTTP header field). Do not hard-code the key.",
            "Save the sld text to a .sld file and apply via Layer "
            "Properties → Symbology → Style → Load Style.",
        ],
    }


def build_qgis_tools() -> list[RegisteredTool]:
    """Return the olmoearth-qgis-bridge tool bundle."""
    return [
        RegisteredTool(
            spec=ToolSpec(
                name="olmoearth_qgis_bridge",
                description=(
                    "Turn a prediction result's tile_urls into QGIS-ready "
                    "XYZ layer URLs + an OGC SLD color-ramp style + load "
                    "instructions. Use after fetch_results to hand a GIS "
                    "analyst a desktop-loadable layer. vmin/vmax set the "
                    "ramp range (default 0..1 for a score layer)."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "tile_urls": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "layer_name": {"type": "string"},
                        "base_url": {"type": "string"},
                        "vmin": {"type": "number", "default": 0.0},
                        "vmax": {"type": "number", "default": 1.0},
                    },
                    "required": ["tile_urls"],
                },
            ),
            handler=_qgis_bridge,
        ),
    ]
