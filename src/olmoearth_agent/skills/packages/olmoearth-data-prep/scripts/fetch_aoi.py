# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Ziming Qi
"""
fetch_aoi.py — Fetch real watershed / HUC polygons (not bboxes) for OE labels.

Sources:
- NLDI (Network-Linked Data Index): upstream basin polygon by NHDPlus COMID.
  Use this for stations / outlets — gives the precise upstream contributing area.
- WBD (Watershed Boundary Dataset): HUC-12 subbasin polygon by HUC-12 code.
  Use this for named subbasin / event-scale work.

Usage:
    python fetch_aoi.py --nldi-comid 12345 --out basin.geojson
    python fetch_aoi.py --huc12 020503060101 --out huc.geojson

Stdlib only.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request

NLDI_BASE = "https://api.water.usgs.gov/nldi/linked-data"
WBD_HUC12_LAYER = (
    "https://hydro.nationalmap.gov/arcgis/rest/services/wbd/MapServer/6/query"
)
USER_AGENT = "OlmoEarth-Skills/0.1 (https://github.com/BAIGroup/OlmoEarth-Skills)"


def _get(url: str, retries: int = 3, backoff: float = 2.0) -> dict:
    """GET JSON with retry + exponential backoff."""
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read())
        except Exception as e:
            last_exc = e
            if attempt < retries - 1:
                time.sleep(backoff ** attempt)
    raise SystemExit(f"GET failed after {retries} retries ({url}): {last_exc}")


def nldi_basin(comid: str) -> dict:
    """Upstream basin polygon for an NHDPlus COMID, as a GeoJSON FeatureCollection."""
    url = f"{NLDI_BASE}/comid/{comid}/basin?simplified=true"
    return _get(url)


def huc12_polygon(huc12: str) -> dict:
    """HUC-12 subbasin polygon from the WBD ArcGIS REST endpoint."""
    params = {
        "where": f"HUC12='{huc12}'",
        "outFields": "HUC12,NAME,STATES,AREASQKM",
        "returnGeometry": "true",
        "f": "geojson",
    }
    url = WBD_HUC12_LAYER + "?" + urllib.parse.urlencode(params)
    return _get(url)


def main():
    parser = argparse.ArgumentParser(
        description="Fetch real watershed / HUC polygons (NLDI basin or WBD HUC-12).",
    )
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--nldi-comid",
        help="NHDPlus COMID; returns upstream basin polygon",
    )
    g.add_argument(
        "--huc12",
        help="HUC-12 code (12 digits); returns HUC-12 subbasin polygon",
    )
    parser.add_argument(
        "--out",
        default="-",
        help="Output path (default: stdout)",
    )
    args = parser.parse_args()

    if args.nldi_comid:
        result = nldi_basin(args.nldi_comid)
    elif args.huc12:
        result = huc12_polygon(args.huc12)
    else:
        parser.error("must pass --nldi-comid or --huc12")
        return

    out = json.dumps(result, indent=2)
    if args.out == "-":
        print(out)
    else:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(out)
        print(f"Wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
