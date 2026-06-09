# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""The olmoearth-qgis-bridge tool bundle (skill #11).

Turn a prediction result's tile templates into QGIS-ready XYZ URLs + an
OGC SLD style + load instructions, so a GIS analyst can pull the layer
into QGIS without leaving the desktop.
"""

from __future__ import annotations

from typing import Any

from olmoearth_agent.llm.types import ToolSpec
from olmoearth_agent.reporting.qgis import (
    DEFAULT_BASE_URL,
    build_legend,
    build_raster_sld,
    resolve_xyz_url,
)
from olmoearth_agent.tools.registry import RegisteredTool, ToolContext


def _property_from_tiles(tile_urls: list[str], fallback: str) -> str:
    """The score/class property a tile template renders (``?property_name=...``)."""
    for t in tile_urls:
        if "property_name=" in t:
            return t.split("property_name=", 1)[1].split("&", 1)[0]
    return fallback


async def _qgis_bridge(args: dict[str, Any], _ctx: ToolContext) -> dict[str, Any]:
    base_url = args.get("base_url", DEFAULT_BASE_URL)
    layer_name = args.get("layer_name", "olmoearth_layer")
    tile_urls = args["tile_urls"]
    vmin, vmax = float(args.get("vmin", 0.0)), float(args.get("vmax", 1.0))
    xyz_urls = [resolve_xyz_url(t, base_url) for t in tile_urls]
    sld = build_raster_sld(layer_name, vmin=vmin, vmax=vmax)
    legend = build_legend(
        _property_from_tiles(tile_urls, layer_name),
        vmin=vmin,
        vmax=vmax,
        classes=args.get("classes"),
    )
    return {
        "layer_name": layer_name,
        "xyz_urls": xyz_urls,
        "sld": sld,
        "legend": legend,
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
                    "XYZ layer URLs + an OGC SLD color-ramp style + a structured "
                    "`legend` (value->color stops, matching the SLD) + load "
                    "instructions. Use after fetch_results to hand a GIS analyst a "
                    "desktop-loadable layer and a legend to read the map by. "
                    "vmin/vmax set the ramp range (default 0..1 for a score "
                    "layer); pass `classes` for a categorical layer."
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
                        "classes": {
                            "type": "array",
                            "items": {"type": "object"},
                            "description": "Categorical layer classes "
                            "([{value, color?, label?}]); omit for a continuous score.",
                        },
                    },
                    "required": ["tile_urls"],
                },
            ),
            handler=_qgis_bridge,
        ),
    ]
