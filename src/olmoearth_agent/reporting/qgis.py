# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 OlmoEarth Agent contributors
"""QGIS bridge helpers (skill #12).

Turn a prediction result's tile templates into QGIS-ready XYZ URLs and an
OGC SLD style. Pure functions — no QGIS dependency. The generated SLD is
standard OGC SLD 1.0 (loadable by QGIS, GeoServer, etc.); the URL
resolution preserves the ``{z}/{x}/{y}`` template QGIS expects.

NOTE: actually loading the layer in QGIS is the user's confirmation step;
what's verified here is that the SLD is well-formed and the URL resolves.
"""

from __future__ import annotations

from xml.sax.saxutils import quoteattr

DEFAULT_BASE_URL = "https://olmoearth.allenai.org"

# YlOrRd 5-stop ramp (ColorBrewer) — sensible default for a 0..1 score raster.
_YLORRD = ("#ffffb2", "#fecc5c", "#fd8d3c", "#f03b20", "#bd0026")


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
