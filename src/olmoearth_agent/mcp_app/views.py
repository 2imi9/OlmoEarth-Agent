# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""The MCP App's views: HTML pages Claude renders beside a tool's result.

Each view is a ``ui://olmoearth-agent/<name>.html`` resource of MIME type
``text/html;profile=mcp-app`` (the MCP Apps extension, ``io.modelcontextprotocol/ui``).
The pages live in ``views/``; :func:`view_html` inlines the shared script
(``common.js``, ``figures.js`` for the Check card's and the result card's
small figures, ``OE.fig``, ``sheet.js`` for the views that edit a label
sheet, ``glyph.js`` for the Studio panel's property glyph, and
``overview.js`` for the start card's overview figure, ``OE.overview``, drawn
in code: round r5-a) and
the pinned URL and integrity hash of the MCP Apps client, so each resource is
one self-contained page.

Scripts come only from the pinned URLs below, each with its Subresource
Integrity hash; Studio's fonts, Manrope and Roboto Mono, come from Google
Fonts; the views inline only the AGENT tag, as a data URL, in their one slim
line on top (the browser sign-in page keeps the wordmark and tag). Each
view's ``csp`` declares just the origins it loads from; the host blocks every
other origin. No view can reach the network
otherwise: Studio is called by the server's tools, never by a page.

Round r4-k (K9): two example pictures beside the owner's kelp rule, a square
that counts as Present and one that counts as Absent, are inlined as data
URLs too (:func:`examples_for`), only into the data of the views that show
the owner's rule (the label sheet and the Check card), only when the rule
is the owner's and all three files of ``views/examples/`` exist. None ship
yet (the owner picks them), so nothing shows.
"""

from __future__ import annotations

import base64
import json
import threading
from functools import cache
from pathlib import Path
from typing import Any

#: MIME type of an MCP App view (the extension's ``RESOURCE_MIME_TYPE``).
VIEW_MIME = "text/html;profile=mcp-app"

#: The extension id a host advertises in ``capabilities.extensions``.
UI_EXTENSION = "io.modelcontextprotocol/ui"

#: The folder holding the pages and their shared scripts.
VIEW_DIR = Path(__file__).with_name("views")

#: The MCP Apps client (``App``), pinned: npm ``@modelcontextprotocol/ext-apps``
#: 2.0.4 (27 October 2026, MIT), and its SHA-384 as fetched on 10 November 2026.
EXT_APPS_URL = (
    "https://unpkg.com/@modelcontextprotocol/ext-apps@2.0.4/dist/src/app-with-deps.js"
)
EXT_APPS_SRI = "sha384-BcYSOlnbm0uMq2/ZWLkwX+z8gt95koY6ne5g0MbXXA4P65fJUagNbjh94Yx+CuKH"

#: Leaflet 1.7.1 and Leaflet.draw 1.0.4, the pair the web UI loads
#: (``webui/js/leaflet.js``), with their SHA-384 as fetched on 10 November 2026.
LEAFLET = {
    "LEAFLET_JS": "https://cdn.jsdelivr.net/npm/leaflet@1.7.1/dist/leaflet.js",
    "LEAFLET_JS_SRI": "sha384-RFZC58YeKApoNsIbBxf4z6JJXmh+geBSgkCQXFyh+4tiFSJmJBt+2FbjxW7Ar16M",
    "LEAFLET_CSS": "https://cdn.jsdelivr.net/npm/leaflet@1.7.1/dist/leaflet.css",
    "LEAFLET_CSS_SRI": "sha384-VzLXTJGPSyTLX6d96AxgkKvE/LRb7ECGyTxuwtpjHnVWVZs2gp5RDjeM/tgBnVdM",
    "DRAW_JS": "https://cdn.jsdelivr.net/npm/leaflet-draw@1.0.4/dist/leaflet.draw.js",
    "DRAW_JS_SRI": "sha384-JP5UPxIO2Tm2o79Fb0tGYMa44jkWar53aBoCbd8ah0+LcCDoohTIYr+zIXyfGIJN",
    "DRAW_CSS": "https://cdn.jsdelivr.net/npm/leaflet-draw@1.0.4/dist/leaflet.draw.css",
    "DRAW_CSS_SRI": "sha384-NZLkVuBRMEeB4VeZz27WwTRvlhec30biQ8Xx7zG7JJnkvEKRg5qi6BNbEXo9ydwv",
}

#: The OpenStreetMap tile server the map view draws its basemap from.
OSM_TILES = "https://tile.openstreetmap.org"

#: Google Fonts, for Studio's Manrope and Roboto Mono: the stylesheet's origin
#: and the font files'.
GOOGLE_FONTS = ["https://fonts.googleapis.com", "https://fonts.gstatic.com"]

#: The origins every view loads from: the MCP Apps client and the font.
BASE_DOMAINS = ["https://unpkg.com", *GOOGLE_FONTS]

#: The OlmoEarth wordmark and the AGENT tag (``webui/assets``), and the overview
#: figure of the OlmoEarth docs (used with Ai2's permission; the start card
#: shows it, ``views/overview.js``), inlined as data URLs so no page loads an
#: image from elsewhere: placeholder -> file. The views inline the AGENT tag
#: and the start view the figure; the browser sign-in page keeps the wordmark
#: and tag (:func:`brand_html`).
BRAND = {
    "LOGO_DATA_URL": "brand/olmoearth-logo.png",
    "AGENT_TAG_DATA_URL": "brand/agent-tag.png",
    "OVERVIEW_DATA_URL": "brand/olmoearth-overview.jpg",
    # r5-e S8: a finished Studio map from the docs, cut from inside its
    # prediction layer (no place, base map or legend): the start card's narrow
    # strip draws it (views/overview.js), 160 x 160.
    "MAP_EXAMPLE_DATA_URL": "brand/studio-map-example.jpg",
    # r5-e K3: the docs' picture of the world cut into strips, one in pink,
    # recoloured to the views' greys, its words, numbers and state taken out
    # (no place, no text): the Check card shows it beside the line on a map's
    # parts (views/check.html), 720 x 328 for 360 x 164 on the face.
    "STRIPS_DATA_URL": "brand/world-strips.jpg",
}

#: The media type of a brand image, by its file's suffix.
_BRAND_TYPES = {".png": "image/png", ".jpg": "image/jpeg"}

#: The example pictures beside the owner's kelp rule (r4-k K9): the folder,
#: its two pictures (key in the view's ``examples`` -> file) and the note
#: beside them (``{"year": N, "place": "a public place"}``; only the year
#: reaches a view, for the fold's Copernicus line).
EXAMPLES_DIR = VIEW_DIR / "examples"
EXAMPLE_PICTURES = {"present": "present.png", "absent": "absent.png"}
EXAMPLES_NOTE = "examples.json"

#: The views that show the owner's rule, so the only ones whose data carry
#: the examples: the label sheet's editor and the Check card.
EXAMPLE_VIEWS = frozenset({"sheet", "check"})

#: The largest example picture (bytes), and the largest note, read.
EXAMPLE_MAX_BYTES = 256 * 1024
EXAMPLES_NOTE_MAX_BYTES = 4096

#: The years an example's Sentinel-2 data can be from (Sentinel-2A flew in 2015).
EXAMPLE_YEARS = range(2015, 2101)

#: A PNG file's first eight bytes.
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

#: The examples as read, by the files' identity (one folder; a few entries).
_examples_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
_examples_lock = threading.Lock()


def uri(name: str) -> str:
    """The resource URI of the view called ``name``."""
    return f"ui://olmoearth-agent/{name}.html"


#: Each view: its file, the scripts inlined into it, its title and the
#: origins its page loads from (the resource's ``_meta.ui.csp``).
VIEWS: dict[str, dict[str, Any]] = {
    "sign-in": {
        "file": "sign_in.html",
        "scripts": ("common.js",),
        "title": "OlmoEarth Studio sign-in",
        "resource_domains": BASE_DOMAINS,
    },
    "result": {
        "file": "result.html",
        "scripts": ("common.js", "figures.js", "sheet.js"),
        "title": "OlmoEarth result and figure",
        "resource_domains": BASE_DOMAINS,
    },
    "sheet": {
        "file": "sheet.html",
        "scripts": ("common.js", "sheet.js"),
        "title": "OlmoEarth label sheet",
        "resource_domains": BASE_DOMAINS,
    },
    "studio": {
        "file": "studio.html",
        "scripts": ("common.js", "glyph.js"),
        "title": "OlmoEarth Studio projects",
        "resource_domains": BASE_DOMAINS,
    },
    "map": {
        "file": "map.html",
        "scripts": ("common.js",),
        "title": "OlmoEarth area map",
        "resource_domains": [*BASE_DOMAINS, "https://cdn.jsdelivr.net", OSM_TILES],
    },
    "start": {
        "file": "start.html",
        "scripts": ("common.js", "overview.js"),
        "title": "OlmoEarth Agent start",
        "resource_domains": BASE_DOMAINS,
    },
    "check": {
        "file": "check.html",
        "scripts": ("common.js", "figures.js", "sheet.js"),
        "title": "OlmoEarth accuracy check",
        "resource_domains": BASE_DOMAINS,
    },
}


#: The app's own icon, which Claude may show over a view in the side panel and full screen
#: (Claude's MCP Apps design guidelines, "Declare an app icon for Claude"): the OlmoEarth mark,
#: a 128 x 128 PNG under 10 KB, declared inline in the server's serverInfo (server.py).
APP_ICON = "brand/app-icon.png"


@cache
def app_icon_data_url() -> str:
    """:data:`APP_ICON` as a ``data:image/png;base64,...`` URL (Claude takes no URL and no SVG)."""
    data = base64.b64encode((VIEW_DIR / APP_ICON).read_bytes()).decode("ascii")
    return "data:image/png;base64," + data


@cache
def brand_data_url(key: str) -> str:
    """A brand image (:data:`BRAND`) as a ``data:<type>;base64,...`` URL."""
    path = VIEW_DIR / BRAND[key]
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{_BRAND_TYPES[path.suffix]};base64," + data


def brand_html() -> str:
    """The OlmoEarth wordmark and the AGENT tag as two ``<img>`` tags (data URLs)."""
    return (
        f'<img src="{brand_data_url("LOGO_DATA_URL")}" alt="OlmoEarth">'
        f'<img src="{brand_data_url("AGENT_TAG_DATA_URL")}" alt="Agent">'
    )


def _read_small(path: Path, limit: int) -> bytes | None:
    """A regular file's bytes when it is at most ``limit`` bytes, else
    ``None`` (missing, not a file, too large or unreadable)."""
    try:
        if not path.is_file() or path.stat().st_size > limit:
            return None
        data = path.read_bytes()
    except OSError:
        return None
    return data if len(data) <= limit else None


def examples_data(folder: Path | None = None) -> dict[str, Any] | None:
    """The two example pictures beside the owner's kelp rule (r4-k K9), as a
    view takes them: ``{"present": <data URL>, "absent": <data URL>,
    "year": N}``; ``None`` unless all three files exist and hold what they
    should.

    ``folder`` (:data:`EXAMPLES_DIR` when not given) holds
    :data:`EXAMPLE_PICTURES`, each a PNG of at most
    :data:`EXAMPLE_MAX_BYTES`, and :data:`EXAMPLES_NOTE`, a JSON object whose
    ``year`` is a whole year in :data:`EXAMPLE_YEARS`. Only the year leaves
    the note: the place it names is never sent. Read again only when a
    file's size or time changes; never raises.
    """
    where = folder if folder is not None else EXAMPLES_DIR
    files = [where / name for name in EXAMPLE_PICTURES.values()]
    files.append(where / EXAMPLES_NOTE)
    try:
        stats = [f.stat() for f in files]
    except OSError:
        return None
    ident: tuple[Any, ...] = (str(where), *((s.st_size, s.st_mtime_ns) for s in stats))
    with _examples_lock:
        known = _examples_cache.get(ident)
    if known is not None:
        return dict(known)
    note_bytes = _read_small(where / EXAMPLES_NOTE, EXAMPLES_NOTE_MAX_BYTES)
    try:
        note = json.loads(note_bytes.decode("utf-8")) if note_bytes else None
    except (UnicodeDecodeError, ValueError):
        note = None
    year = note.get("year") if isinstance(note, dict) else None
    if isinstance(year, bool) or not isinstance(year, int) or year not in EXAMPLE_YEARS:
        return None
    out: dict[str, Any] = {}
    for key, name in EXAMPLE_PICTURES.items():
        data = _read_small(where / name, EXAMPLE_MAX_BYTES)
        if data is None or not data.startswith(_PNG_SIGNATURE):
            return None
        out[key] = "data:image/png;base64," + base64.b64encode(data).decode("ascii")
    out["year"] = year
    with _examples_lock:
        if len(_examples_cache) >= 4:
            _examples_cache.clear()
        _examples_cache[ident] = out
    return dict(out)


def examples_for(
    view: str, rule_source: Any, folder: Path | None = None
) -> dict[str, Any] | None:
    """The examples for one view's data (:func:`examples_data`), or ``None``:
    only for a view that shows the owner's rule (:data:`EXAMPLE_VIEWS`), only
    when the rule shown is the owner's (``rule_source`` ``"owner"``: never a
    rule the user typed, nor none), and only when every file exists. The
    caller leaves the ``examples`` key out on ``None``."""
    if view not in EXAMPLE_VIEWS or rule_source != "owner":
        return None
    return examples_data(folder)


def view_meta(name: str) -> dict[str, Any]:
    """The view's ``_meta`` (its CSP), for its listing and its contents."""
    return {
        "ui": {
            "csp": {"resourceDomains": list(VIEWS[name]["resource_domains"])},
            "prefersBorder": True,
        }
    }


@cache
def view_html(name: str) -> str:
    """The view's page with its scripts and pinned URLs filled in.

    ``/*@SCRIPTS@*/`` becomes the view's shared scripts, and each
    ``@NAME@`` placeholder (``@EXT_APPS_URL@``, ``@LEAFLET_JS@``, ...) its
    value; a placeholder left over is an error, so a page never ships with
    one.
    """
    spec = VIEWS[name]
    page: str = (VIEW_DIR / str(spec["file"])).read_text(encoding="utf-8")
    scripts = "\n".join(
        (VIEW_DIR / s).read_text(encoding="utf-8") for s in spec["scripts"]
    )
    page = page.replace("/*@SCRIPTS@*/", scripts)
    values = {"EXT_APPS_URL": EXT_APPS_URL, "EXT_APPS_SRI": EXT_APPS_SRI, **LEAFLET}
    for key in BRAND:
        values[key] = brand_data_url(key)
    for key, value in values.items():
        page = page.replace(f"@{key}@", value)
    if (
        "@EXT_APPS" in page
        or "@LEAFLET" in page
        or "@DRAW" in page
        or "_DATA_URL@" in page
        or "/*@SCRIPTS@*/" in page
    ):
        raise RuntimeError(f"view {name!r} has a placeholder left over")
    return page
