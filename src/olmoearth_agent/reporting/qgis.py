# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""QGIS bridge helpers (skill #11).

Turn a prediction result's tile templates into QGIS-ready XYZ URLs and an
OGC SLD style. Pure functions, no QGIS dependency. The generated SLD is
standard OGC SLD 1.0 (loadable by QGIS, GeoServer, etc.); the URL
resolution preserves the ``{z}/{x}/{y}`` template QGIS expects.

NOTE: actually loading the layer in QGIS is the user's confirmation step;
what's verified here is that the SLD is well-formed and the URL resolves.
"""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import quote
from xml.sax.saxutils import escape, quoteattr

DEFAULT_BASE_URL = "https://olmoearth.allenai.org"

# YlOrRd 5-stop ramp (ColorBrewer): sensible default for a 0..1 score raster.
_YLORRD = ("#ffffb2", "#fecc5c", "#fd8d3c", "#f03b20", "#bd0026")

#: Unresolved sentinel for the Bearer key in a generated provider file. The key
#: is NEVER written into a shared file (a .qlr is plaintext XML that gets
#: emailed / committed); the user substitutes this or wires a QGIS authcfg.
AUTH_PLACEHOLDER = "__OLMOEARTH_API_KEY__"


def resolve_xyz_url(tile_template: str, base_url: str = DEFAULT_BASE_URL) -> str:
    """Resolve a (possibly relative) tile template to an absolute XYZ URL.

    QGIS XYZ layers want an absolute URL containing the literal
    ``{z}/{x}/{y}`` placeholders, which the Studio templates already use.
    """
    if tile_template.startswith(("http://", "https://")):
        return tile_template
    return base_url.rstrip("/") + "/" + tile_template.lstrip("/")


def _ramp_entries(
    vmin: float, vmax: float, ramp: tuple[str, ...]
) -> list[tuple[float, str]]:
    n = len(ramp)
    if n == 1:
        return [(vmin, ramp[0])]
    span = vmax - vmin
    return [(vmin + span * i / (n - 1), color) for i, color in enumerate(ramp)]


def build_legend(
    property_name: str,
    *,
    vmin: float = 0.0,
    vmax: float = 1.0,
    ramp: tuple[str, ...] = _YLORRD,
    classes: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Structured legend for a result raster — the value->color mapping a viewer
    needs to read a score/class map for analysis.

    Shares the same color ramp as :func:`build_raster_sld`, so a QGIS export and
    an in-chat legend agree. For a **continuous** score layer it returns the ramp
    stops (``value -> color``); for a **categorical** layer pass ``classes``
    (``[{value, color?, label?}]``) and it echoes them with stable keys. Pure
    data (no I/O), so the bridge / web UI can render a swatch strip from it.
    """
    if classes is not None:
        return {
            "kind": "categorical",
            "property": property_name,
            "entries": [
                {
                    "value": c.get("value"),
                    "color": c.get("color", "#888888"),
                    "label": str(c.get("label", c.get("value", ""))),
                }
                for c in classes
            ],
        }
    return {
        "kind": "continuous",
        "property": property_name,
        "vmin": vmin,
        "vmax": vmax,
        "ramp": "YlOrRd",
        "stops": [
            {"value": round(quantity, 6), "color": color, "label": f"{quantity:.3g}"}
            for quantity, color in _ramp_entries(vmin, vmax, ramp)
        ],
    }


def build_raster_sld(
    layer_name: str,
    *,
    vmin: float = 0.0,
    vmax: float = 1.0,
    ramp: tuple[str, ...] = _YLORRD,
    opacity: float = 0.8,
) -> str:
    """Build an OGC SLD 1.0 raster color-ramp style.

    Suitable for a continuous score layer (e.g. ``sample_karst_score``).
    Returns well-formed SLD XML.
    """
    entries = "\n".join(
        f'          <ColorMapEntry color={quoteattr(color)} '
        f'quantity="{quantity:.6g}" label="{quantity:.3g}"/>'
        for quantity, color in _ramp_entries(vmin, vmax, ramp)
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<StyledLayerDescriptor version="1.0.0" '
        'xmlns="http://www.opengis.net/sld" '
        'xmlns:ogc="http://www.opengis.net/ogc" '
        'xmlns:xlink="http://www.w3.org/1999/xlink" '
        'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">\n'
        "  <NamedLayer>\n"
        f"    <Name>{layer_name}</Name>\n"
        "    <UserStyle>\n"
        f"      <Title>{layer_name} (ramp)</Title>\n"
        "      <FeatureTypeStyle>\n"
        "        <Rule>\n"
        "          <RasterSymbolizer>\n"
        f"            <Opacity>{opacity:g}</Opacity>\n"
        '            <ColorMap type="ramp">\n'
        f"{entries}\n"
        "            </ColorMap>\n"
        "          </RasterSymbolizer>\n"
        "        </Rule>\n"
        "      </FeatureTypeStyle>\n"
        "    </UserStyle>\n"
        "  </NamedLayer>\n"
        "</StyledLayerDescriptor>\n"
    )


def build_qlr(
    layer_name: str,
    xyz_url: str,
    *,
    crs: str = "EPSG:3857",
    max_zoom: int = 19,
) -> str:
    """Build a QGIS Layer Definition file (.qlr) for an authenticated XYZ raster.

    A ``.qlr`` is pure-text XML the user can drag into QGIS to load the layer in
    one drop (QGIS 3.x: Layer -> Add from Layer Definition File). The Bearer key
    is **not** baked in -- the datasource carries the unresolved
    :data:`AUTH_PLACEHOLDER` sentinel, which the user replaces at load time or,
    better, supplies via a QGIS authentication config (``authcfg``) so the token
    stays in the encrypted ``qgis-auth.db``. Returns well-formed XML.
    """
    safe = re.sub(r"\W+", "_", layer_name).strip("_") or "olmoearth_layer"
    layer_id = f"{safe}_xyz"
    enc = quote(xyz_url, safe="")
    datasource = (
        f"type=xyz&url={enc}&zmax={max_zoom}&zmin=0"
        f"&http-header:Authorization=Bearer {AUTH_PLACEHOLDER}"
    )
    return (
        "<qlr>\n"
        "  <layer-tree-group>\n"
        f"    <layer-tree-layer name={quoteattr(layer_name)} providerKey=\"wms\" "
        f"source={quoteattr(datasource)} checked=\"Qt::Checked\" expanded=\"1\" "
        f"id={quoteattr(layer_id)}>\n"
        "      <customproperties/>\n"
        "    </layer-tree-layer>\n"
        "  </layer-tree-group>\n"
        '  <maplayer type="raster">\n'
        f"    <id>{escape(layer_id)}</id>\n"
        f"    <datasource>{escape(datasource)}</datasource>\n"
        f"    <layername>{escape(layer_name)}</layername>\n"
        "    <provider>wms</provider>\n"
        f"    <srs><spatialrefsys><authid>{escape(crs)}</authid>"
        "</spatialrefsys></srs>\n"
        "    <pipe>\n"
        '      <rasterrenderer type="singlebandcolordata" band="1" opacity="1"/>\n'
        "    </pipe>\n"
        "    <blendMode>0</blendMode>\n"
        "  </maplayer>\n"
        "</qlr>\n"
    )


def build_gdal_wms_xml(
    xyz_url: str,
    *,
    crs: str = "EPSG:3857",
    tile_format: str = "png",
    bands: int = 4,
    max_zoom: int = 19,
) -> str:
    """Build a GDAL WMS/TMS minidriver descriptor for an authenticated XYZ endpoint.

    GDAL reads this XML as a raster (``gdal_translate descriptor.xml out.tif``,
    or open it directly in QGIS-via-GDAL). XYZ tiles map to ``Service name="TMS"``
    with ``<YOrigin>top</YOrigin>``; GDAL uses ``${z}/${x}/${y}`` placeholders, so
    the Studio ``{z}/{x}/{y}`` template is rewritten. The ``DataWindow`` is the
    EPSG:3857 web-mercator world extent (Studio tiles are web mercator). GDAL_WMS
    has no element for an ``Authorization`` header, so the **Bearer token is
    supplied out-of-band at run time** via ``--config GDAL_HTTP_HEADERS`` (GDAL
    >= 3.7) -- never written into this file. Returns well-formed XML.
    """
    gdal_url = (
        xyz_url.replace("{z}", "${z}").replace("{x}", "${x}").replace("{y}", "${y}")
    )
    return (
        "<GDAL_WMS>\n"
        '  <Service name="TMS">\n'
        f"    <ServerUrl>{escape(gdal_url)}</ServerUrl>\n"
        f"    <TileFormat>{escape(tile_format)}</TileFormat>\n"
        "  </Service>\n"
        "  <DataWindow>\n"
        "    <UpperLeftX>-20037508.34</UpperLeftX>\n"
        "    <UpperLeftY>20037508.34</UpperLeftY>\n"
        "    <LowerRightX>20037508.34</LowerRightX>\n"
        "    <LowerRightY>-20037508.34</LowerRightY>\n"
        f"    <TileLevel>{int(max_zoom)}</TileLevel>\n"
        "    <TileCountX>1</TileCountX>\n"
        "    <TileCountY>1</TileCountY>\n"
        "    <YOrigin>top</YOrigin>\n"
        "  </DataWindow>\n"
        f"  <Projection>{escape(crs)}</Projection>\n"
        "  <BlockSizeX>256</BlockSizeX>\n"
        "  <BlockSizeY>256</BlockSizeY>\n"
        f"  <BandsCount>{int(bands)}</BandsCount>\n"
        "  <MaxConnections>5</MaxConnections>\n"
        "  <Cache/>\n"
        "  <UserAgent>OlmoEarth-Agent</UserAgent>\n"
        "</GDAL_WMS>\n"
    )


def cog_recipe(*, descriptor: str = "layer.xml", out: str = "out_cog.tif") -> dict[str, Any]:
    """An honest, user-run recipe to materialize a COG (the agent has no GDAL).

    A true Cloud-Optimized GeoTIFF needs internal tiling, embedded overview
    pyramids, and a specific IFD layout that only the GeoTIFF/GDAL stack writes
    correctly -- impossible to produce honestly in this torch-free, GDAL-free
    agent, which has only tile URLs (not pixels). So instead of faking a COG, we
    hand the user a one-line ``gdal_translate`` command to run locally on the
    GDAL_WMS descriptor. The Bearer token is injected at run time, never written
    into a file.
    """
    command = (
        'gdal_translate --config GDAL_HTTP_HEADERS '
        '"Authorization: Bearer $OLMOEARTH_API_KEY" '
        f"{descriptor} {out} -of COG -co COMPRESS=DEFLATE -co BLOCKSIZE=512 "
        "-co OVERVIEWS=AUTO"
    )
    rationale = (
        "This agent is torch-free and ships without GDAL/rasterio, so it cannot "
        "itself write a valid Cloud-Optimized GeoTIFF -- a real COG requires "
        "internal tiling, embedded overview pyramids, and a specific IFD layout "
        "only the GeoTIFF/GDAL stack produces. Run this gdal_translate command "
        "locally on the GDAL_WMS descriptor; supply the Bearer token at run time "
        "(never written into a generated file). Note: a COG built from the "
        "public tile endpoint is a VISUAL raster, not the model's full-precision "
        "scores -- those need the original prediction GeoTIFF, which the tile "
        "endpoint does not expose."
    )
    return {
        "command": command,
        "descriptor": descriptor,
        "output": out,
        "rationale": rationale,
    }
