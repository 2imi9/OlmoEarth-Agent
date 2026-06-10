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
    AUTH_PLACEHOLDER,
    DEFAULT_BASE_URL,
    build_gdal_wms_xml,
    build_legend,
    build_qlr,
    build_raster_sld,
    cog_recipe,
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
    crs = args.get("crs", "EPSG:3857")
    vmin, vmax = float(args.get("vmin", 0.0)), float(args.get("vmax", 1.0))
    xyz_urls = [resolve_xyz_url(t, base_url) for t in tile_urls]
    sld = build_raster_sld(layer_name, vmin=vmin, vmax=vmax)
    legend = build_legend(
        _property_from_tiles(tile_urls, layer_name),
        vmin=vmin,
        vmax=vmax,
        classes=args.get("classes"),
    )
    primary = xyz_urls[0] if xyz_urls else ""
    qlr = build_qlr(layer_name, primary, crs=crs) if primary else None
    gdal_wms_xml = build_gdal_wms_xml(primary, crs=crs) if primary else None
    return {
        "layer_name": layer_name,
        "xyz_urls": xyz_urls,
        "sld": sld,
        "legend": legend,
        # Pure-text provider files (Bearer key NOT embedded -- carries the
        # AUTH_PLACEHOLDER sentinel the user resolves at load time).
        "qlr": qlr,
        "gdal_wms_xml": gdal_wms_xml,
        # The agent is GDAL-free, so a true COG is a user-run recipe, not output.
        "cog_recipe": cog_recipe(),
        "instructions": [
            "Fastest: save the `qlr` text to a .qlr file and load it via QGIS -> "
            "Layer -> Add from Layer Definition File (one drop, no manual setup).",
            "Or QGIS -> Layer -> Add Layer -> Add XYZ Layer; paste an xyz_url.",
            "Tiles require auth: replace the "
            f"'{AUTH_PLACEHOLDER}' placeholder in the .qlr with your key, or "
            "(recommended) wire a QGIS Authentication config so the token stays "
            "in the encrypted qgis-auth.db. Do not commit a file with the real key.",
            "Save the sld text to a .sld file and apply via Layer Properties -> "
            "Symbology -> Style -> Load Style.",
            "To materialize a Cloud-Optimized GeoTIFF on disk, save `gdal_wms_xml` "
            "to a .xml file and run the `cog_recipe.command` locally (the agent "
            "has no GDAL; see cog_recipe.rationale).",
        ],
    }


def build_qgis_tools() -> list[RegisteredTool]:
    """Return the olmoearth-qgis-bridge tool bundle."""
    return [
        RegisteredTool(
            spec=ToolSpec(
                name="olmoearth_qgis_bridge",
                description=(
                    "Turn a prediction result's tile_urls into a QGIS-loadable "
                    "layer pack: XYZ layer URLs, a ready-to-drop QGIS layer "
                    "definition (`qlr`), a GDAL WMS/XYZ descriptor (`gdal_wms_xml`), "
                    "an OGC SLD color-ramp style, a structured `legend` (value->"
                    "color stops matching the SLD), and load instructions. The "
                    "Bearer key is never embedded in the generated files. Because "
                    "the agent is GDAL-free, a true Cloud-Optimized GeoTIFF is "
                    "returned as a user-run `cog_recipe` (a gdal_translate command "
                    "+ rationale), not produced here. Use after fetch_results to "
                    "hand a GIS analyst a desktop-loadable layer. vmin/vmax set "
                    "the ramp range (default 0..1); pass `classes` for a "
                    "categorical layer; `crs` defaults to EPSG:3857."
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
                        "crs": {
                            "type": "string",
                            "default": "EPSG:3857",
                            "description": "CRS authid for the provider files "
                            "(Studio tiles are web mercator).",
                        },
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
