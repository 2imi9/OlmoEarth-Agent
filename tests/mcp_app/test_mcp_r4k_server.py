# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Round r4-k (16 Nov 2026), the server's side, each string as
specs/r4-k-round.md gives it:

- K3: olmoearth_app_host_report (app-only) checks a panel's report, logs one
  INFO line of names and counts, and answers ok; a report not of its shape
  is refused quietly (the model reads ok); strings that are not names are
  dropped and counted. The client's extension keys and the UI extension's
  mimeTypes are logged once.
- K4: where panels show, the agent's tools and the app's own for the model
  are listed visibility ["model"], the views' tools ["app"]; no mimeTypes
  rule; every panel's call names a views' tool.
- K6: the result view gives the range's two ends, low_text and high_text,
  formatted as range_text.
- K7: a Studio result's points get a record (where "points") but never
  pictures; olmoearth_app_review_link gives a square's or point's street
  map link by key and rank, in the sheet's link shape (sheet.osm_url).
- K8: olmoearth_app_review_csv: the header row first, rows in rank order,
  the note on every row, no rate column; the file name says what it is;
  save mode writes it beside the records (0600) and names it only.
- K9: the example pictures reach the sheet's and the Check card's data only
  beside the owner's rule and only when all three files exist;
  imagery.example_scene, for a cluster run, is never called by the app.
"""

from __future__ import annotations

import base64
import copy
import csv
import inspect
import io
import json
import logging
import re
from pathlib import Path
from typing import Any

import pytest

mcp = pytest.importorskip("mcp")
from mcp import types  # noqa: E402

from olmoearth_agent.mcp_app import (  # noqa: E402
    auth,
    imagery,
    review,
    server,
    sheet,
    views,
)
from olmoearth_agent.skills import build_default_registry  # noqa: E402

BASE = "http://mock-studio/api/v1"
R = "11111111-2222-4333-8444-555555555555"
P = "99999999-8888-4777-8666-555555555555"
STUDIO = review.STUDIO_TOOL
DIRECT = "olmoearth_review_set"

#: Eight windows' centres, one per row of a scores file.
CENTRES = [[12.345678 + 0.001 * i, 48.765432 - 0.001 * i] for i in range(8)]

#: A Studio result's sampled windows on a 4 x 4 grid (each row's grid index).
STUDIO_WINDOWS = [0, 2, 5, 7, 8, 10, 13, 15]

#: Two listed windows, as the review tools list them (no coordinate).
ROWS = [
    {
        "rank": 1,
        "window_index": 5,
        "row": 1,
        "col": 1,
        "margin": 0.01,
        "predicted_class": 0,
        "class_name": "water",
    },
    {
        "rank": 2,
        "window_index": 2,
        "row": 0,
        "col": 2,
        "margin": 0.02,
        "predicted_class": 1,
        "class_name": "trees",
    },
]

#: The CSV's note column (r4-k K8).
NOTE = "look-first list, not a random sample; no error rate comes from it"

#: A small file that starts as a PNG does.
PNG = b"\x89PNG\r\n\x1a\n" + bytes(range(24))

#: Words no face holds (r4-i's rule).
_OFF_FACE = re.compile(r"\b(?:windows?|budget|stratum|design|review set)\b", re.I)

#: A panel's host report as the contract gives it (r4-k K3).
REPORT = {
    "view": "olmoearth-check",
    "event": "start",
    "caps": ["openLinks", "serverTools", "downloadFile", "updateModelContext"],
    "modes": ["inline", "fullscreen"],
    "context_keys": ["theme", "displayMode", "containerDimensions"],
    "container": "maxHeight",
    "host": "Claude Desktop",
}


@pytest.fixture(autouse=True)
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The output root (the read roots' own and the write root), no key."""
    out = tmp_path / "outputs"
    out.mkdir()
    monkeypatch.setenv("OLMOEARTH_OUTPUT_ROOT", str(out))
    monkeypatch.delenv("OLMOEARTH_SCORES_ROOT", raising=False)
    monkeypatch.delenv(auth.KEY_ENV, raising=False)
    monkeypatch.setenv(auth.KEY_FILE_ENV, str(tmp_path / "cfg" / "studio.env"))
    return out.resolve()


def _app() -> server.AppServer:
    """An app with the agent's own registry and no Studio key."""
    return server.AppServer(None, keys=auth.KeyStore(), base_url=BASE)


async def _list(srv: Any) -> list[types.Tool]:
    """The server's tools as a client lists them."""
    result = await srv.request_handlers[types.ListToolsRequest](
        types.ListToolsRequest(method="tools/list")
    )
    return result.root.tools


async def _call(srv: Any, name: str, arguments: dict[str, Any]) -> types.CallToolResult:
    """One tool call as a client makes it."""
    request = types.CallToolRequest(
        method="tools/call",
        params=types.CallToolRequestParams(name=name, arguments=arguments),
    )
    result = await srv.request_handlers[types.CallToolRequest](request)
    return result.root


def _texts(result: types.CallToolResult) -> list[str]:
    """What the model reads of a result."""
    return [c.text for c in result.content if isinstance(c, types.TextContent)]


def _view(result: types.CallToolResult) -> dict[str, Any]:
    """A result's view data."""
    return (result.meta or {})[server.VIEW_DATA_KEY]


def _host_lines(caplog: pytest.LogCaptureFixture) -> list[str]:
    """The host log's lines."""
    return [r.getMessage() for r in caplog.records if r.name == server.HOST_LOG.name]


def _map_scores(
    root: Path,
    *,
    centres: list[Any] | None = None,
    name: str = "scores_run_m_p4.json",
) -> Path:
    """A direct run's scores file of 8 windows on a 2 x 4 grid that records
    its map (raster path, window size, period and one centre per row)."""
    payload = {
        "format": "olmoearth-agent/scores@1",
        "grid": [2, 4],
        "scores": [[0.5, 0.5]] * 8,
        "patch_px": 4,
        "raster": {"file": "m.tif", "path": str(root / "m.tif"), "shape": [8, 16]},
        "date_window": {"start": "2024-04-01", "end": "2024-06-30"},
        "centres_lon_lat": CENTRES if centres is None else centres,
    }
    path = root / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _studio_scores(root: Path, *, centres: bool = True) -> Path:
    """A Studio result's scores file as olmoearth_review_set_from_result
    saves it: the sampled windows, one row each, and each row's centre; no
    raster, window size or period."""
    payload: dict[str, Any] = {
        "format": "olmoearth-agent/scores@1",
        "result_id": R,
        "grid": [4, 4],
        "windows": STUDIO_WINDOWS,
        "scores": [[0.5, 0.5]] * 8,
    }
    if centres:
        payload["centres_lon_lat"] = CENTRES
    path = root / "scores_11111111-222_class_g4.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _direct(
    root: Path, rows: list[dict[str, Any]] | None = None, **file: Any
) -> dict[str, Any]:
    """A direct run's review block, its record written."""
    inner = {"n_review": 2, "review": copy.deepcopy(ROWS if rows is None else rows)}
    path = _map_scores(root, **file)
    block = review.review_view(DIRECT, inner, {"scores_path": str(path)})
    assert block is not None
    return block


def _studio(root: Path, *, centres: bool = True) -> tuple[dict[str, Any], Path]:
    """A Studio result's review block, and its scores file."""
    path = _studio_scores(root, centres=centres)
    inner = {
        "n_review": 2,
        "review": copy.deepcopy(ROWS),
        "result_id": R,
        "prediction_id": P,
        "scores_path": str(path),
    }
    block = review.review_view(STUDIO, inner, {"result_id": R})
    assert block is not None
    return block, path


def _examples(
    folder: Path, *, year: Any = 2024, present: bytes = PNG, absent: bytes = PNG
) -> Path:
    """The three example files in folder; the examples' cache emptied."""
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "present.png").write_bytes(present)
    (folder / "absent.png").write_bytes(absent)
    note = {"year": year, "place": "a public place"}
    (folder / "examples.json").write_text(json.dumps(note), encoding="utf-8")
    views._examples_cache.clear()
    return folder


# --------------------------------------------------------------------------- K3


@pytest.mark.asyncio
async def test_k3_the_host_report_is_app_only_logged_and_answers_ok(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """r4-k K3: the report tool is a views' tool (visibility ["app"]) of the
    contract's keys; a report of its shape gets one INFO line of names
    only and {"ok": true}, which is all the model reads; a teardown report
    with nothing but its view and event is logged with empty lists."""
    monkeypatch.setattr(server, "client_renders_views", lambda _srv: True)
    srv = server.build_server(_app())
    tool = {t.name: t for t in await _list(srv)}[server.HOST_REPORT_TOOL]
    assert server.HOST_REPORT_TOOL == "olmoearth_app_host_report"
    assert tool.meta["ui"]["visibility"] == ["app"]
    assert "resourceUri" not in tool.meta["ui"]
    schema = tool.inputSchema
    assert set(schema["properties"]) == {
        "view",
        "event",
        "caps",
        "modes",
        "context_keys",
        "container",
        "host",
    }
    assert schema["required"] == ["view", "event"]
    assert schema["properties"]["event"]["enum"] == ["start", "teardown"]
    assert schema["properties"]["container"]["enum"] == ["height", "maxHeight", "none"]
    for names in ("caps", "modes", "context_keys"):
        assert schema["properties"][names]["maxItems"] == 40
    with caplog.at_level(logging.INFO, logger=server.HOST_LOG.name):
        told = await _call(srv, server.HOST_REPORT_TOOL, REPORT)
        bare = {"view": "olmoearth-label-sheet", "event": "teardown"}
        torn = await _call(srv, server.HOST_REPORT_TOOL, bare)
    assert told.isError is not True and told.structuredContent == {"ok": True}
    assert _texts(told) == ['{"ok": true}']
    assert torn.isError is not True and torn.structuredContent == {"ok": True}
    assert _host_lines(caplog) == [
        "panel host report: view=olmoearth-check event=start host=Claude Desktop "
        "container=maxHeight "
        "caps=[openLinks,serverTools,downloadFile,updateModelContext] "
        "modes=[inline,fullscreen] "
        "context_keys=[theme,displayMode,containerDimensions] dropped=0",
        "panel host report: view=olmoearth-label-sheet event=teardown host=- "
        "container=none caps=[] modes=[] context_keys=[] dropped=0",
    ]


@pytest.mark.parametrize(
    "bad",
    [
        {"view": "olmoearth-check", "event": "stop"},
        {"view": "", "event": "start"},
        {"view": "48.7654", "event": "start"},
        {"event": "start"},
        {"view": "olmoearth-check", "event": "start", "caps": "openLinks"},
        {"view": "olmoearth-check", "event": "start", "caps": ["openLinks", 3]},
        {"view": "olmoearth-check", "event": "start", "modes": ["inline"] * 41},
        {"view": "olmoearth-check", "event": "start", "container": "tall"},
        {"view": "olmoearth-check", "event": "start", "container": None},
        {"view": "olmoearth-check", "event": "start", "host": 3},
        {"view": "olmoearth-check", "event": "start", "host": "x" * 65},
        {"view": "olmoearth-check", "event": "start", "host": "-33.98, 18.35"},
    ],
)
@pytest.mark.asyncio
async def test_k3_a_report_not_of_its_shape_is_refused_quietly(
    bad: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """r4-k K3: a report whose view, event, lists, container or host is not
    of the contract's shape is refused with one fixed sentence for the
    panel; the model reads {"ok": true}, and nothing is logged."""
    monkeypatch.setattr(server, "client_renders_views", lambda _srv: True)
    srv = server.build_server(_app())
    with caplog.at_level(logging.INFO, logger=server.HOST_LOG.name):
        out = await _call(srv, server.HOST_REPORT_TOOL, bad)
    assert out.isError is True
    assert _texts(out) == ['{"ok": true}']
    assert out.structuredContent is not None
    assert out.structuredContent["error"] == server.HOST_REPORT_REFUSED
    assert _host_lines(caplog) == []
    assert server.HOST_REPORT_TOOL in server.QUIET_APP_TOOLS
    assert server.IMAGERY_TOOLS < server.QUIET_APP_TOOLS


@pytest.mark.asyncio
async def test_k3_strings_that_are_not_names_are_dropped_and_counted(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """r4-k K3: a string with no letter (a number, a coordinate) is not a
    name: it is dropped, never logged, and counted; a key the contract does
    not name is ignored."""
    monkeypatch.setattr(server, "client_renders_views", lambda _srv: True)
    srv = server.build_server(_app())
    report = {
        **REPORT,
        "caps": ["openLinks", "12.345678", "-"],
        "context_keys": ["theme"],
        "place": "12.345678,48.765432",
    }
    with caplog.at_level(logging.INFO, logger=server.HOST_LOG.name):
        out = await _call(srv, server.HOST_REPORT_TOOL, report)
    assert out.structuredContent == {"ok": True}
    (line,) = _host_lines(caplog)
    assert "caps=[openLinks]" in line and "context_keys=[theme]" in line
    assert line.endswith(" dropped=2")
    assert "12.3" not in line and "48.7" not in line and "place" not in line


def test_k3_the_clients_extensions_are_logged_once(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """r4-k K3: the first time the server reads a client's capabilities it
    logs the extension keys and the UI extension's mimeTypes, once; K4: no
    rule on mimeTypes, so a list without the view's MIME type, or none,
    still counts as a client with panels."""

    class Ctx:
        def __init__(self, caps: types.ClientCapabilities) -> None:
            self.session = type(
                "S", (), {"client_params": type("P", (), {"capabilities": caps})()}
            )()

    class Srv:
        def __init__(self, caps: types.ClientCapabilities) -> None:
            self.request_context = Ctx(caps)

    monkeypatch.setattr(server, "_LOGGED_ONCE", set())
    caps = types.ClientCapabilities(
        extensions={
            views.UI_EXTENSION: {"mimeTypes": [views.VIEW_MIME]},
            "io.example/other": {},
        }
    )
    with caplog.at_level(logging.INFO, logger=server.HOST_LOG.name):
        assert server.client_renders_views(object()) is False  # nothing read
        assert server.client_renders_views(Srv(caps)) is True
        assert server.client_renders_views(Srv(caps)) is True
        assert server.client_renders_views(Srv(types.ClientCapabilities())) is False
    assert _host_lines(caplog) == [
        "client extensions: [io.modelcontextprotocol/ui,io.example/other]; "
        "the UI extension's mimeTypes: [text/html;profile=mcp-app]"
    ]
    monkeypatch.setattr(server, "_LOGGED_ONCE", set())
    caplog.clear()
    other = types.ClientCapabilities(
        extensions={views.UI_EXTENSION: {"mimeTypes": ["text/plain"]}}
    )
    bare = types.ClientCapabilities(extensions={views.UI_EXTENSION: {}})
    with caplog.at_level(logging.INFO, logger=server.HOST_LOG.name):
        assert server.client_renders_views(Srv(bare)) is True
        assert server.client_renders_views(Srv(other)) is True
    assert _host_lines(caplog) == [
        "client extensions: [io.modelcontextprotocol/ui]; "
        "the UI extension's mimeTypes: none"
    ]


# --------------------------------------------------------------------------- K4


@pytest.mark.asyncio
async def test_k4_where_panels_show_the_models_tools_are_model_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """r4-k K4: to a client with panels every agent tool and every app tool
    for the model carries visibility ["model"] beside its view (meta only,
    descriptions unchanged), every views' tool ["app"]; without panels no
    tool carries a visibility and no views' tool is listed."""
    plain = {t.name: t for t in await _list(server.build_server(_app()))}
    monkeypatch.setattr(server, "client_renders_views", lambda _srv: True)
    tools = {t.name: t for t in await _list(server.build_server(_app()))}
    registry = list(build_default_registry().names())
    for name in [*registry, *server.OWN_TOOLS]:
        meta = tools[name].meta
        assert meta is not None and meta["ui"]["visibility"] == ["model"], name
        assert tools[name].description == plain[name].description, name
        view = (
            server.TOOL_VIEW.get(name)
            if name in registry
            else server.OWN_TOOLS[name][0]
        )
        if view:
            assert meta["ui"]["resourceUri"] == views.uri(view), name
            assert meta["ui/resourceUri"] == views.uri(view), name
        else:
            assert "resourceUri" not in meta["ui"] and "ui/resourceUri" not in meta
    for name in server.APP_TOOLS:
        assert tools[name].meta["ui"]["visibility"] == ["app"], name
        assert name not in plain, name
    for tool in plain.values():
        assert tool.meta is None or "visibility" not in tool.meta["ui"], tool.name
    for new in (
        server.HOST_REPORT_TOOL,
        "olmoearth_app_review_link",
        "olmoearth_app_review_csv",
    ):
        assert new in server.APP_TOOLS


def test_k4_every_panel_call_names_a_views_tool() -> None:
    """r4-k K4: with the agent's tools model-only, a panel may call only the
    views' tools: every literal OE.call in views/ names one of APP_TOOLS."""
    called: set[str] = set()
    pattern = re.compile(r"OE\.call\(\s*\w+\s*,\s*'([A-Za-z0-9_]+)'")
    for page in sorted(views.VIEW_DIR.iterdir()):
        if page.suffix in (".html", ".js"):
            called |= set(pattern.findall(page.read_text(encoding="utf-8")))
    assert {"olmoearth_app_sheet", "olmoearth_app_review_image"} <= called
    assert called <= set(server.APP_TOOLS), called - set(server.APP_TOOLS)


# --------------------------------------------------------------------------- K6


@pytest.mark.parametrize(
    ("low", "high", "low_text", "high_text"),
    [
        (0.061, 0.217, "6.1%", "21.7%"),
        (0.0, 0.045, "0.0%", "4.5%"),
        (0.955, 1.0, "95.5%", "100.0%"),
        (0.004, 0.006, "0.4%", "0.6%"),
    ],
)
def test_k6_the_result_view_gives_the_ranges_two_ends(
    low: float, high: float, low_text: str, high_text: str
) -> None:
    """r4-k K6: low_text and high_text beside range_text, formatted as it
    is, at 0%, at 100% and on a narrow band: the page computes nothing."""
    inner = {
        "valid_interval": {"estimate": (low + high) / 2, "low": low, "high": high},
        "n_labelled": 80,
        "n_unjudged": 0,
    }
    got = server._estimate_result(inner, None, None)
    assert got is not None
    view = got["view"]
    assert (view["low_text"], view["high_text"]) == (low_text, high_text)
    assert view["range_text"] == f"{low_text} to {high_text}"


def test_k6_with_cannot_tell_the_ends_are_given_and_the_rates_range_stays() -> None:
    """r4-k K6: with Cannot tell answers the view still gives the 95% range's
    ends; the rate's own range (rate_range_text) is as before."""
    inner = {
        "valid_interval": {
            "estimate": None,
            "low": 0.08,
            "high": 0.24,
            "estimate_range": [0.1, 0.15],
        },
        "n_labelled": 80,
        "n_unjudged": 3,
    }
    got = server._estimate_result(inner, None, None)
    assert got is not None
    view = got["view"]
    assert view["rate"] is None and view["rate_range_text"] == "10.0% to 15.0%"
    assert (view["low_text"], view["high_text"]) == ("8.0%", "24.0%")
    source = Path(server.__file__).read_text(encoding="utf-8")
    for key, end in (("low_text", "low"), ("high_text", "high")):
        line = f'        "{key}": rules.percent({end}),\n'
        assert "        # WORDING: r4k K6\n" + line in source, key


# --------------------------------------------------------------------------- K7


def test_k7_a_studio_results_points_get_a_record_but_no_pictures(root: Path) -> None:
    """r4-k K7: a Studio result's points whose scores file records each
    point's centre get a record (where "points", the same key recipe, the
    captions for the CSV) and a key, but no pictures and the same reason;
    without centres, no key and no record. No place or path is in the
    block; the pictures refuse a points key before anything is fetched."""
    bare, _ = _studio(root, centres=False)
    assert bare["key"] is None and bare["where"] == "none"
    assert bare["why_none"] == review.STUDIO_POINTS
    assert not (root / "review_sets").exists()
    block, path = _studio(root)
    assert re.fullmatch(r"[0-9a-f]{24}", block["key"])
    assert block["where"] == "points" and block["pictures"] is False
    assert block["why_none"] == review.STUDIO_POINTS and block["unit"] == "point"
    dumped = json.dumps(block)
    for lon, lat in CENTRES:
        assert f"{lon:.4f}" not in dumped and f"{lat:.4f}" not in dumped
    assert str(path) not in dumped and str(root) not in dumped
    st = path.stat()
    assert block["key"] == review.review_key(
        STUDIO, str(path), st.st_size, st.st_mtime_ns, [[1, 5], [2, 2]]
    )
    record_path = root / "review_sets" / f"{block['key']}.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    assert record["where"] == "points" and record["tool"] == STUDIO
    assert record["windows"] == [
        {"rank": 1, "window_index": 5},
        {"rank": 2, "window_index": 2},
    ]
    assert record["captions"] == {"1": "Map says: water", "2": "Map says: trees"}
    assert record_path.stat().st_mode & 0o777 == 0o600
    with pytest.raises(ValueError, match=re.escape(review.NO_PLACE_RECORDED)):
        review.review_row(block["key"], 1)
    assert review.review_row(block["key"], 1, points=True)[1:] == (str(path), 5)
    for on in (True, False):
        assert imagery.review_image(block["key"], 1, on=on) == {
            "ok": False,
            "error": review.NO_PLACE_RECORDED,
        }
    # A direct run's squares keep their pictures; their record has captions.
    square = _direct(root)
    assert square["where"] == "map" and square["pictures"] is True
    kept = json.loads(
        (root / "review_sets" / f"{square['key']}.json").read_text(encoding="utf-8")
    )
    assert kept["where"] == "map" and set(kept["captions"]) == {"1", "2"}


@pytest.mark.asyncio
async def test_k7_the_street_map_link_by_key_and_rank(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """r4-k K7: olmoearth_app_review_link (app-only) gives a listed square's
    or point's OpenStreetMap link, the centre its scores file records (a
    Studio result's by its place in the file's windows); the model reads
    {"ok": true}. A key of no review, a rank not listed and a scores file
    changed since are refused with their sentences, none with a place."""
    monkeypatch.setattr(server, "client_renders_views", lambda _srv: True)
    srv = server.build_server(_app())
    tool = {t.name: t for t in await _list(srv)}["olmoearth_app_review_link"]
    assert tool.meta["ui"]["visibility"] == ["app"]
    assert tool.inputSchema["required"] == ["review_key", "rank"]
    assert tool.inputSchema["properties"] == {
        "review_key": {"type": "string", "pattern": "^[0-9a-f]{24}$"},
        "rank": {"type": "integer", "minimum": 1},
    }
    square = _direct(root)
    studio, _ = _studio(root)

    async def link(key: str, rank: int) -> types.CallToolResult:
        """The panel's press."""
        args = {"review_key": key, "rank": rank}
        return await _call(srv, "olmoearth_app_review_link", args)

    first = await link(square["key"], 1)
    assert first.isError is not True
    assert first.structuredContent == {
        "url": "https://www.openstreetmap.org/?mlat=48.760432&mlon=12.350678"
        "#map=17/48.760432/12.350678"
    }
    assert first.structuredContent["url"] == sheet.osm_url(*CENTRES[5])
    assert _texts(first) == ['{"ok": true}']
    second = await link(square["key"], 2)
    assert second.structuredContent == {"url": sheet.osm_url(*CENTRES[2])}
    point = await link(studio["key"], 1)
    at = STUDIO_WINDOWS.index(5)
    assert point.structuredContent == {"url": sheet.osm_url(*CENTRES[at])}
    assert _texts(point) == ['{"ok": true}']
    refused = {
        review.REVIEW_UNKNOWN: await link("0" * 24, 1),
        review.NOT_LISTED: await link(square["key"], 3),
    }
    _map_scores(root, centres=[[0.0, 0.0], *CENTRES[1:]])  # rewritten since
    refused[review.REVIEW_CHANGED] = await link(square["key"], 1)
    for said, out in refused.items():
        assert out.isError is True, said
        assert out.structuredContent is not None
        assert out.structuredContent["error"] == said
    for out in (first, second, point, *refused.values()):
        text = "".join(_texts(out))
        for lon, lat in CENTRES:
            assert f"{lon:.4f}" not in text and f"{lat:.4f}" not in text


def test_k7_one_link_shape_for_the_sheet_and_the_look_first_cards() -> None:
    """r4-k K7: osm_url, factored out of sheet.window_link, is the one link
    shape: the sheet's Street map and the look-first cards' both use it;
    the pictures read a review's centre where the link and the CSV do."""
    assert sheet.osm_url(12.3456, 48.7654) == (
        "https://www.openstreetmap.org/?mlat=48.7654&mlon=12.3456"
        "#map=17/48.7654/12.3456"
    )
    assert sheet.degrees(12.5) == "12.5" and sheet.degrees(-33.98640049) == "-33.9864"
    assert "osm_url(lon, lat)" in inspect.getsource(sheet.window_link)
    assert "sheet.osm_url(lon, lat)" in inspect.getsource(review.review_link)
    window_row = inspect.getsource(imagery._review_window_row)
    assert "review.window_centre(scores_real, index)" in window_row
    assert "openstreetmap.org" not in inspect.getsource(review)


def test_k7_a_centre_that_is_not_a_place_on_earth_is_refused(root: Path) -> None:
    """r4-k K7: window_centre gives the centre the file records for a
    window (a Studio result's by its place in the file's windows), and
    refuses one beyond 180 and 90 degrees, a window the file does not
    hold, or a file with no centre per row."""
    far = _map_scores(root, centres=[[200.0, 10.0], *CENTRES[1:]], name="far.json")
    for index in (0, 8):
        with pytest.raises(ValueError, match=re.escape(review.NO_CENTRE)):
            review.window_centre(str(far), index)
    assert review.window_centre(str(far), 1) == tuple(CENTRES[1])
    studio = _studio_scores(root)
    assert review.window_centre(str(studio), 13) == tuple(CENTRES[6])
    with pytest.raises(ValueError, match=re.escape(review.NO_CENTRE)):
        review.window_centre(str(studio), 1)
    placeless = _studio_scores(root, centres=False)
    with pytest.raises(ValueError, match=re.escape(review.NO_CENTRE)):
        review.window_centre(str(placeless), 13)


def test_k7_k8_a_listed_centre_missing_refuses_the_link_and_the_list(
    root: Path,
) -> None:
    """r4-k K7 and K8: a listed window whose centre is not a number: its
    link is refused (NO_CENTRE), another's is given, and the list is not
    built (NO_CENTRES)."""
    holes: list[Any] = [list(c) for c in CENTRES]
    holes[5] = [None, None]
    block = _direct(root, centres=holes)
    with pytest.raises(ValueError, match=re.escape(review.NO_CENTRE)):
        review.review_link(block["key"], 1)
    assert review.review_link(block["key"], 2) == {"url": sheet.osm_url(*CENTRES[2])}
    with pytest.raises(ValueError, match=re.escape(review.NO_CENTRES)):
        review.review_csv(block["key"])


def test_k7_k8_new_sentences_are_marked_and_face_words_only() -> None:
    """r4-k K7 and K8: each new sentence and the CSV's words are marked, and
    no refusal holds a word no face may hold; nor does the host report's."""
    source = Path(review.__file__).read_text(encoding="utf-8")
    for item, start in (
        ("K7", "NO_CENTRE = ("),
        ("K8", "NO_CENTRES = ("),
        ("K8", 'CSV_NOT_SAVED = "'),
        ("K8", 'CSV_NAME = "'),
        ("K8", 'CSV_NOTE = "'),
        ("K8", "CSV_HEADER = ("),
    ):
        assert f"# WORDING: r4k {item}\n{start}" in source, start
    for said in (
        review.NO_CENTRE,
        review.NO_CENTRES,
        review.CSV_NOT_SAVED,
        server.HOST_REPORT_REFUSED,
    ):
        assert _OFF_FACE.search(said) is None, said
    marked = Path(server.__file__).read_text(encoding="utf-8")
    assert '# WORDING: r4k K3\nHOST_REPORT_REFUSED = "' in marked


# --------------------------------------------------------------------------- K8


def test_k8_the_csv_of_places_header_first_rank_order_and_its_note(
    root: Path,
) -> None:
    """r4-k K8: the header row first, the rows in rank order whatever the
    listing's order, each with its rank, its square's index, what the map
    says, its centre and the note; no rate column; the file name says it is
    not a sample. Save mode writes the same bytes beside the records (0600,
    the folder 0700) and gives the file's name only."""
    block = _direct(root, rows=list(reversed(ROWS)))
    key = block["key"]
    got = review.review_csv(key)
    assert set(got) == {"name", "text"}
    assert got["name"] == f"look_first_not_a_sample_{key}.csv"
    assert review.CSV_NOTE == NOTE
    rows = list(csv.reader(io.StringIO(got["text"], newline="")))
    assert rows[0] == ["rank", "square", "map_says", "lon", "lat", "note"]
    lon5, lat5 = (sheet.degrees(v) for v in CENTRES[5])
    lon2, lat2 = (sheet.degrees(v) for v in CENTRES[2])
    assert rows[1:] == [
        ["1", "5", "Map says: water", lon5, lat5, NOTE],
        ["2", "2", "Map says: trees", lon2, lat2, NOTE],
    ]
    assert "rate" not in ",".join(rows[0])
    assert got["text"].startswith("rank,square,map_says,lon,lat,note\r\n")
    assert not (root / "review_sets" / got["name"]).exists()
    saved = review.review_csv(key, save=True)
    assert saved == {"name": got["name"], "file": got["name"]}
    written = root / "review_sets" / got["name"]
    assert written.read_bytes() == got["text"].encode("utf-8")
    assert written.stat().st_mode & 0o777 == 0o600
    assert written.parent.stat().st_mode & 0o777 == 0o700


def test_k8_a_studio_results_points_list_says_point(root: Path) -> None:
    """r4-k K8: a Studio result's points: the second column is "point", each
    centre the file's row for that point."""
    block, _ = _studio(root)
    text = review.review_csv(block["key"])["text"]
    rows = list(csv.reader(io.StringIO(text, newline="")))
    assert rows[0] == ["rank", "point", "map_says", "lon", "lat", "note"]
    first = [sheet.degrees(v) for v in CENTRES[STUDIO_WINDOWS.index(5)]]
    second = [sheet.degrees(v) for v in CENTRES[STUDIO_WINDOWS.index(2)]]
    assert rows[1][:3] == ["1", "5", "Map says: water"] and rows[1][3:5] == first
    assert rows[2][:3] == ["2", "2", "Map says: trees"] and rows[2][3:5] == second
    assert rows[1][5] == rows[2][5] == NOTE


@pytest.mark.asyncio
async def test_k8_the_csv_tool_tells_the_model_nothing(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """r4-k K8: olmoearth_app_review_csv (app-only) gives the panel the name
    and text, or with save the file's name; the model reads {"ok": true}.
    save must be a boolean; a key of no review and a changed scores file
    are refused with their sentences, none with a place."""
    monkeypatch.setattr(server, "client_renders_views", lambda _srv: True)
    srv = server.build_server(_app())
    tool = {t.name: t for t in await _list(srv)}["olmoearth_app_review_csv"]
    assert tool.meta["ui"]["visibility"] == ["app"]
    assert tool.inputSchema["required"] == ["review_key"]
    assert tool.inputSchema["properties"]["save"] == {"type": "boolean"}
    key = _direct(root)["key"]
    name = f"look_first_not_a_sample_{key}.csv"
    out = await _call(srv, "olmoearth_app_review_csv", {"review_key": key})
    assert _texts(out) == ['{"ok": true}']
    assert out.structuredContent is not None
    assert out.structuredContent["name"] == name
    assert "12.350678" in out.structuredContent["text"]
    args = {"review_key": key, "save": True}
    saved = await _call(srv, "olmoearth_app_review_csv", args)
    assert saved.structuredContent == {"name": name, "file": name}
    assert _texts(saved) == ['{"ok": true}']
    args = {"review_key": key, "save": "yes"}
    wrong = await _call(srv, "olmoearth_app_review_csv", args)
    assert wrong.isError is True
    unknown = await _call(srv, "olmoearth_app_review_csv", {"review_key": "f" * 24})
    assert unknown.structuredContent is not None
    assert unknown.structuredContent["error"] == review.REVIEW_UNKNOWN
    _map_scores(root, centres=[[0.0, 0.0], *CENTRES[1:]])  # rewritten since
    changed = await _call(srv, "olmoearth_app_review_csv", {"review_key": key})
    assert changed.structuredContent is not None
    assert changed.structuredContent["error"] == review.REVIEW_CHANGED
    for result in (wrong, unknown, changed):
        assert result.isError is True
        assert "12.35" not in "".join(_texts(result))


def test_k8_a_list_that_cannot_be_saved_says_so_and_logs_no_path(
    root: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """r4-k K8: a file that cannot be written gives one fixed sentence, and
    the log line names the failure's kind only, never a path."""
    key = _direct(root)["key"]

    def fail(path: Path, text: str) -> None:
        """A write that fails, naming its path."""
        raise OSError(f"cannot write {path}")

    monkeypatch.setattr(review, "_write_text", fail)
    with caplog.at_level(logging.WARNING, logger=review.logger.name):
        with pytest.raises(ValueError, match=re.escape(review.CSV_NOT_SAVED)):
            review.review_csv(key, save=True)
    said = [r.getMessage() for r in caplog.records if r.name == review.logger.name]
    assert said == ["a look-first list was not saved (OSError)"]
    assert str(root) not in caplog.text


def test_k8_a_cell_is_never_read_as_a_formula() -> None:
    """r4-k K8: a text cell is printable, at most 80 characters, and never
    starts as a spreadsheet formula."""
    for start in ("=SUM(A1)", "+1", "-2", "@a"):
        assert review._cell(start) == "'" + start
    assert review._cell("Map says: water\x07") == "Map says: water"
    assert len(review._cell("x" * 200)) == 80


# --------------------------------------------------------------------------- K9


def test_k9_the_examples_need_all_three_files_and_the_owners_rule(
    tmp_path: Path,
) -> None:
    """r4-k K9: the two pictures as PNG data URLs and the year, only for the
    sheet and the Check card, only beside the owner's rule, only when both
    pictures and the note exist and hold what they should; the place the
    note names never reaches a view."""
    folder = tmp_path / "examples"
    assert views.examples_for("sheet", "owner", folder) is None
    _examples(folder)
    url = "data:image/png;base64," + base64.b64encode(PNG).decode("ascii")
    want = {"present": url, "absent": url, "year": 2024}
    assert views.examples_for("sheet", "owner", folder) == want
    assert views.examples_for("check", "owner", folder) == want
    assert views.EXAMPLE_VIEWS == {"sheet", "check"}
    assert views.EXAMPLES_DIR == views.VIEW_DIR / "examples"
    for view in ("result", "studio", "map", "start", "sign-in"):
        assert views.examples_for(view, "owner", folder) is None, view
    for source in ("typed", None, "", "Owner"):
        assert views.examples_for("sheet", source, folder) is None, source
    for name in ("present.png", "absent.png", "examples.json"):
        _examples(folder)
        (folder / name).unlink()
        views._examples_cache.clear()
        assert views.examples_for("sheet", "owner", folder) is None, name
    for year in (True, "2024", 1999, 2014, None, 2024.0):
        _examples(folder, year=year)
        assert views.examples_for("sheet", "owner", folder) is None, year
    _examples(folder, absent=b"GIF89a" + bytes(10))
    assert views.examples_for("sheet", "owner", folder) is None
    _examples(folder, present=PNG + bytes(views.EXAMPLE_MAX_BYTES))
    assert views.examples_for("sheet", "owner", folder) is None
    _examples(folder)
    shown = views.examples_for("sheet", "owner", folder)
    assert shown is not None and "place" not in shown


@pytest.mark.asyncio
async def test_k9_the_sheets_data_carries_the_examples_beside_the_owners_rule(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """r4-k K9: olmoearth_app_sheet's answer (the sheet editor's data) gains
    examples only when the sheet's rule is the owner's and the files exist;
    otherwise the key is left out."""
    monkeypatch.setattr(views, "EXAMPLES_DIR", _examples(tmp_path / "examples"))
    source: dict[str, Any] = {"rule": "owner"}

    def read(path: str, face: Any, *, unblind: bool = False) -> dict[str, Any]:
        """A page copy whose rule is source's."""
        return {"path": path, "rule_source": source["rule"], "windows": []}

    monkeypatch.setattr(server.sheet, "read_sheet", read)
    monkeypatch.setattr(server, "_sheet_part", lambda _path: None)
    app = _app()
    args = {"labels_csv_path": "a_to_label.csv"}
    owned = await app.app_tool("olmoearth_app_sheet", args)
    assert owned["examples"]["year"] == 2024
    assert owned["examples"]["absent"].startswith("data:image/png;base64,")
    for rule in ("typed", None):
        source["rule"] = rule
        assert "examples" not in await app.app_tool("olmoearth_app_sheet", args)
    source["rule"] = "owner"
    monkeypatch.setattr(views, "EXAMPLES_DIR", tmp_path / "none")
    assert "examples" not in await app.app_tool("olmoearth_app_sheet", args)
    await app.aclose()


@pytest.mark.asyncio
async def test_k9_the_check_cards_data_carries_the_examples_beside_the_owners_rule(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """r4-k K9: the Check card's data gains examples for a map of presence
    and absence (the owner's rule), and none for a map of other classes,
    which has no rule until the user types one."""
    monkeypatch.setattr(views, "EXAMPLES_DIR", _examples(tmp_path / "examples"))
    names = {"classes": ["absent", "present"]}
    app = _app()

    async def run(name: str, _arguments: dict[str, Any]) -> dict[str, Any]:
        """The result's files, read: no zone."""
        assert name == server.SCORES_TOOL
        inner = {
            "available": True,
            "result_id": R,
            "prediction_id": P,
            "kind": "topk",
            "classes": [{"label": n} for n in names["classes"]],
            "zones": [],
            "must_state": [],
            "forbidden_claims": [],
        }
        return {"ok": True, "result": inner}

    async def no_name(_prediction_id: Any) -> None:
        """No name read from Studio."""
        return None

    monkeypatch.setattr(app, "run_agent_tool", run)
    monkeypatch.setattr(app, "_map_name", no_name)
    monkeypatch.setattr(server.check, "offer_zones", lambda _inner, **_kw: [])
    owned = _view(await app.check_map_tool({"result_id": R}, views=True))
    assert owned["rule"]["source"] == "owner"
    assert owned["examples"]["year"] == 2024
    assert owned["examples"]["present"].startswith("data:image/png;base64,")
    names["classes"] = ["water", "trees"]
    other = _view(await app.check_map_tool({"result_id": R}, views=True))
    assert other["rule"]["set"] is False and "examples" not in other
    await app.aclose()


def test_k9_example_scene_is_for_a_cluster_run_never_the_app() -> None:
    """r4-k K9: nothing of the app calls imagery.example_scene; a place that
    is not on Earth, a grid that is not a UTM zone's or a square size that
    is not whole is refused before anything is fetched, and so is a period
    of more than 12 months."""
    call = re.compile(r"\bexample_scene\(")
    for path in (server.__file__, review.__file__, sheet.__file__, views.__file__):
        assert call.search(Path(path).read_text(encoding="utf-8")) is None, path
    for page in views.VIEW_DIR.iterdir():
        if page.is_file():
            assert "example_scene" not in page.read_text(encoding="utf-8"), page
    period = ("2024-01-01", "2024-03-31")
    for place, epsg, patch in (
        ((200.0, 0.0), 32734, 11),
        ((18.35, -95.0), 32734, 11),
        ((float("nan"), 0.0), 32734, 11),
        ((18.35, -33.98), 4326, 11),
        ((18.35, -33.98), True, 11),
        ((18.35, -33.98), 32734, 0),
        ((18.35, -33.98), 32734, True),
    ):
        got = imagery.example_scene(place, epsg=epsg, period=period, patch=patch)
        assert got == {"ok": False, "error": imagery.EXAMPLE_REFUSED}, place
    long = imagery.example_scene(
        (18.35, -33.98), epsg=32734, period=("2024-01-01", "2025-06-30")
    )
    assert long["ok"] is False
    assert long["error"] in (imagery.LONG_PERIOD, imagery.NEEDS_EXTRA)
    unread = imagery.example_scene((18.35, -33.98), epsg=32734, period=("soon", "x"))
    assert unread["error"] in (imagery.NO_PERIOD, imagery.NEEDS_EXTRA)


def test_k9_the_fixed_box_is_the_test_scenes() -> None:
    """r4-k K9: the example's picture is drawn as the test scene's is: 7
    squares a side on a 10 m grid, the square outlined in the middle."""
    pytest.importorskip("rasterio")
    box, square = imagery._fixed_box(
        imagery.TEST_PLACE, imagery.TEST_EPSG, imagery.TEST_PATCH
    )
    side = 7 * imagery.TEST_PATCH
    assert (box.width, box.height) == (side, side)
    assert square == (33, 33, 11, 11)
    assert box.crs == f"EPSG:{imagery.TEST_EPSG}"
    assert box.transform[0] == 10.0 and box.transform[4] == -10.0
