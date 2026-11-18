# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""The MCP App server: its tools and views, sign-in, and what reaches the model."""

from __future__ import annotations

import asyncio
import copy
import json
import random
import re
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import pytest
from pytest_httpx import HTTPXMock

mcp = pytest.importorskip("mcp")
from mcp import types  # noqa: E402

from olmoearth_agent.harness.soul import PACKAGED_SOUL  # noqa: E402
from olmoearth_agent.llm.types import ToolSpec  # noqa: E402
from olmoearth_agent.mcp_app import auth, check, server, sheet, views  # noqa: E402
from olmoearth_agent.skills import build_default_registry  # noqa: E402
from olmoearth_agent.tools import statistical_rules as rules  # noqa: E402
from olmoearth_agent.tools.registry import RegisteredTool, ToolRegistry  # noqa: E402

BASE = "http://mock-studio/api/v1"
KEY = "sk_test_mcp_key_456"


@pytest.fixture(autouse=True)
def _roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    out = tmp_path / "outputs"
    out.mkdir()
    monkeypatch.setenv("OLMOEARTH_OUTPUT_ROOT", str(out))
    monkeypatch.delenv("OLMOEARTH_SCORES_ROOT", raising=False)
    monkeypatch.delenv(auth.KEY_ENV, raising=False)
    monkeypatch.setenv(auth.KEY_FILE_ENV, str(tmp_path / "cfg" / "studio.env"))
    return out


def _app(registry: ToolRegistry | None = None) -> server.AppServer:
    return server.AppServer(registry, keys=auth.KeyStore(), base_url=BASE)


async def _list(srv: Any) -> list[types.Tool]:
    result = await srv.request_handlers[types.ListToolsRequest](
        types.ListToolsRequest(method="tools/list")
    )
    return result.root.tools


async def _call(srv: Any, name: str, arguments: dict[str, Any]) -> types.CallToolResult:
    request = types.CallToolRequest(
        method="tools/call",
        params=types.CallToolRequestParams(name=name, arguments=arguments),
    )
    result = await srv.request_handlers[types.CallToolRequest](request)
    return result.root


def _text(result: types.CallToolResult) -> str:
    return "".join(c.text for c in result.content if isinstance(c, types.TextContent))


def _view(result: types.CallToolResult) -> dict[str, Any]:
    return (result.meta or {})[server.VIEW_DATA_KEY]


@pytest.mark.asyncio
async def test_lists_every_agent_tool_with_its_view_and_no_app_tool_without_the_extension() -> (
    None
):
    srv = server.build_server(_app())
    tools = {t.name: t for t in await _list(srv)}
    registry_names = set(build_default_registry().names())
    assert set(server.TOOL_VIEW) <= registry_names  # the allow-list names real tools
    for name in registry_names:
        assert name in tools
        if name not in server.TOOL_VIEW:
            assert tools[name].meta is None  # an intermediate step opens no panel
            continue
        assert tools[name].meta["ui"]["resourceUri"] == views.uri(
            server.TOOL_VIEW[name]
        )
        assert (
            tools[name].meta["ui/resourceUri"] == tools[name].meta["ui"]["resourceUri"]
        )
    for quiet in ("olmoearth_search_predictions", "olmoearth_studio_label_inventory"):
        assert tools[quiet].meta is None
    assert tools["olmoearth_plan_label_sample"].meta["ui"]["resourceUri"] == views.uri(
        "result"
    )
    assert tools["olmoearth_request_aoi"].meta["ui"]["resourceUri"] == views.uri("map")
    assert tools["olmoearth_load_context"].meta["ui"]["resourceUri"] == views.uri(
        "studio"
    )
    assert tools["olmoearth_sign_in"].meta["ui"]["resourceUri"] == views.uri("sign-in")
    assert tools["olmoearth_start"].meta["ui"]["resourceUri"] == views.uri("start")
    assert tools["olmoearth_new_task"].meta is None  # no view: a plain tool
    assert tools["olmoearth_label_sheet"].meta["ui"]["resourceUri"] == views.uri(
        "sheet"
    )
    # The Check card is the app's own tool: the registry keeps its 25 tools.
    assert tools["olmoearth_check_map"].meta["ui"]["resourceUri"] == views.uri("check")
    assert server.CHECK_TOOL not in registry_names and len(registry_names) == 25
    assert not [n for n in tools if n.startswith("olmoearth_app_")]


@pytest.mark.asyncio
async def test_request_aoi_is_described_as_the_map_panel_answers() -> None:
    """The panel sends ids and the quoted name; the description must not promise a bbox."""
    specs = {s.name: s for s in build_default_registry().specs()}
    for name, (old, _new) in server.DESCRIPTION_EDITS.items():
        assert old in specs[name].description  # the edit still finds its words
    srv = server.build_server(_app())
    listed = {t.name: t for t in await _list(srv)}["olmoearth_request_aoi"].description
    assert listed is not None
    assert "bbox arrive" not in listed
    assert "area_id, project_id and name" in listed and "no coordinates" in listed


@pytest.mark.asyncio
async def test_app_tools_are_listed_app_only_to_a_client_with_the_extension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(server, "client_renders_views", lambda _srv: True)
    srv = server.build_server(_app())
    tools = {t.name: t for t in await _list(srv)}
    for name in server.APP_TOOLS:
        assert tools[name].meta["ui"]["visibility"] == ["app"]
        assert "resourceUri" not in tools[name].meta["ui"]
    press = tools["olmoearth_app_check_draw"].inputSchema
    assert press["required"] == ["draws"]
    # The projects card's "In progress" takes nothing: it reads this computer.
    listed = tools["olmoearth_app_sheets_in_progress"].inputSchema
    assert listed == {"type": "object", "properties": {}}
    assert press["properties"]["draws"]["maxItems"] == server.CHECK_DRAWS_MAX == 16
    # The sheet's tools name a window by its handle, never by its index.
    for name in (
        "olmoearth_app_sheet",
        "olmoearth_app_save_labels",
        "olmoearth_app_window_image",
        "olmoearth_app_window_link",
    ):
        assert "window_index" not in json.dumps(tools[name].inputSchema), name
    pictures = tools["olmoearth_app_window_image"].inputSchema["properties"]
    assert {"labels_csv_path", "sheet_id", "handle", "test_scene", "review"} == set(
        pictures
    )
    # The Check card's press carries the user's labelling rule when it has one:
    # the gate's, or the map's own in place of the owner's (option A).
    assert press["properties"]["reading_rule"] == {"type": "string", "maxLength": 300}
    assert "reading_rule" not in press["required"]
    told = tools["olmoearth_app_check_draw"].description or ""
    assert "in place of the owner's rule before the map's first part is drawn" in told
    save = tools["olmoearth_app_save_labels"].inputSchema
    assert {"labels_csv_path", "sheet_id", "face", "rule"} <= set(save["required"])
    assert "labels" not in save["properties"] and "answers" in save["properties"]
    # The sheet's version the panel read: sent when it has one, never required.
    assert save["properties"]["version"] == {"type": "string"}
    assert "version" not in save["required"]
    link = tools["olmoearth_app_window_link"].inputSchema
    assert link["required"] == ["labels_csv_path", "sheet_id", "handle"]
    assert set(tools["olmoearth_app_sheet"].inputSchema["properties"]) == {
        "labels_csv_path",
        "face",
        "unblind",
    }


def test_client_renders_views_reads_the_extension_capability() -> None:
    class Ctx:
        def __init__(self, caps: types.ClientCapabilities) -> None:
            self.session = type(
                "S", (), {"client_params": type("P", (), {"capabilities": caps})()}
            )()

    class Srv:
        def __init__(self, caps: types.ClientCapabilities) -> None:
            self.request_context = Ctx(caps)

    with_ui = types.ClientCapabilities(
        extensions={views.UI_EXTENSION: {"mimeTypes": [views.VIEW_MIME]}}
    )
    assert server.client_renders_views(Srv(with_ui)) is True
    assert server.client_renders_views(Srv(types.ClientCapabilities())) is False
    assert server.client_renders_views(object()) is False


@pytest.mark.asyncio
async def test_views_are_complete_pages_with_their_csp() -> None:
    srv = server.build_server(_app())
    listed = await srv.request_handlers[types.ListResourcesRequest](
        types.ListResourcesRequest(method="resources/list")
    )
    uris = {str(r.uri) for r in listed.root.resources}
    assert uris == {views.uri(name) for name in views.VIEWS}
    for name in views.VIEWS:
        read = await srv.request_handlers[types.ReadResourceRequest](
            types.ReadResourceRequest(
                method="resources/read",
                params=types.ReadResourceRequestParams(uri=views.uri(name)),
            )
        )
        content = read.root.contents[0]
        assert content.mimeType == views.VIEW_MIME
        assert (
            content.meta["ui"]["csp"]["resourceDomains"]
            == views.VIEWS[name]["resource_domains"]
        )
        page = content.text
        assert page.startswith("<!doctype html>")
        assert views.EXT_APPS_URL in page and views.EXT_APPS_SRI in page
        assert (
            "@EXT_APPS" not in page
            and "/*@SCRIPTS@*/" not in page
            and "@LEAFLET" not in page
        )
        assert "OE.start(" in page
        # One slim line: the AGENT tag inlined as a data URL, nothing fetched, no
        # wordmark, and the sentence that this panel is the agent's, in Claude
        # (the reader test's B/S6, OE.SUB).
        assert page.count("data:image/png;base64,") == 1 and "_DATA_URL@" not in page
        assert views.brand_data_url("AGENT_TAG_DATA_URL") in page
        assert views.brand_data_url("LOGO_DATA_URL") not in page
        assert "This panel is the OlmoEarth Agent, an app inside Claude." in page
        assert set(views.GOOGLE_FONTS) <= set(views.VIEWS[name]["resource_domains"])


def test_view_pages_hold_no_raw_line_separator_or_control_character() -> None:
    """A raw U+2028/U+2029 is a line terminator in JavaScript: inside a regex
    literal it stops the whole inlined module from parsing (no test here runs
    the scripts, so the bytes are checked). Write them as escapes."""
    banned = {chr(0x2028), chr(0x2029), "\x00", "\x0b", "\x0c", "\x1b", "\x7f"}
    for name in views.VIEWS:
        page = views.view_html(name)
        found = sorted({f"U+{ord(c):04X}" for c in page if c in banned})
        assert not found, f"{name}: raw {found}"


def test_label_editors_follow_the_hosts_display_mode() -> None:
    """Every view that draws the label editor (sheet.js) hands its Full screen
    answer and the host's context changes to OE.sheetMode, so the face's
    dates follow full screen and back; the editor itself never asks for a
    display mode (Full screen is the page's header action, at the user's
    press)."""
    editors = [n for n, spec in views.VIEWS.items() if "sheet.js" in spec["scripts"]]
    assert sorted(editors) == ["check", "result", "sheet"]
    for name in editors:
        page = views.view_html(name)
        assert "OE.fullscreenButton(app, OE.sheetMode)" in page, name
        assert "onContext: (ctx) => OE.sheetMode(ctx.displayMode)" in page, name
    script = (views.VIEW_DIR / "sheet.js").read_text(encoding="utf-8")
    assert "requestDisplayMode" not in script and "OE.sheetMode = " in script


def _words(text: str) -> str:
    return " ".join(text.split())


def test_instructions_are_the_agents_prompt_without_the_harness_claim() -> None:
    # The sentence is in the prompt, so its removal cannot silently stop matching.
    assert server._HARNESS_CHECK in _words(PACKAGED_SOUL.read_text(encoding="utf-8"))
    text = server.server_instructions(build_default_registry())
    assert server._HARNESS_CHECK not in _words(text)
    assert "never quote a figure from a tool's description as a finding" in _words(text)
    assert "olmoearth_sign_in" in text and server.MCP_RULES in text
    assert (
        "olmoearth_plan_label_sample" in text
    )  # the capability card follows the prompt


def test_instructions_send_no_studio_result_to_the_look_first_job() -> None:
    """soul.md's review-set rule, which sat inside the part Claude Code keeps
    (about 4,100 characters), routed "where is my map likely wrong" to
    olmoearth_review_set_from_result on any Studio result: here it is
    rewritten (soul.md itself unchanged) so a direct run's scores file is
    ranked, and on a Studio result the limits come first, the Check card is
    offered (not for embeddings), and the grid runs only at the user's ask.
    The new rule ends well inside the part kept."""
    soul = _words(PACKAGED_SOUL.read_text(encoding="utf-8"))
    assert server._SOUL_REVIEW_ROUTE in soul  # the edit still finds its words
    text = server.server_instructions(build_default_registry())
    assert server._SOUL_REVIEW_ROUTE not in _words(text)
    route = server.SOUL_REVIEW_ROUTE
    assert route in text and text.index(route) + len(route) < 4000
    assert "no recorded experiment supports such an order" in route
    assert "olmoearth_check_map" in route and "not for embeddings" in route
    assert "only if the user still asks" in route
    # MCP_RULES carries no copy of the rule: it lies past the part kept.
    assert "olmoearth_review_set_from_result" not in server.MCP_RULES


def test_the_rules_claude_code_keeps_still_end_before_character_4000() -> None:
    """The r4-h plan's section 8: Claude Code keeps about the first 4,100
    characters of the instructions, so rule 9, the review route and the
    soul's "How wrong a map is" rule (to its last words) each end before
    character 4,000; this round's model text is tool output, not
    instructions, and never moves them."""
    text = server.server_instructions(build_default_registry())
    for end in (
        server.RULE_9,
        server.SOUL_REVIEW_ROUTE,
        "does not apply to a stratified design.",
    ):
        assert end in text, end
        assert text.index(end) + len(end) < 4000, end
    for note in (server.CHECK_CLASS_MAP_NOTE, server.CHECK_TEXT_OWNER_RULE_NOTE):
        assert note.split("{")[0] not in text


@pytest.mark.asyncio
async def test_review_from_result_description_says_its_limits_before_running() -> None:
    """The listed description says, before any run, that no recorded
    experiment supports the order on a Studio result and that it reads a
    small grid of single pixels; offers the Check card; runs only at the
    user's ask; and says neither job applies to embeddings. The registry's
    own description (the agent's loop) is unchanged."""
    specs = {s.name: s for s in build_default_registry().specs()}
    old, new = server.DESCRIPTION_EDITS["olmoearth_review_set_from_result"]
    assert old in specs["olmoearth_review_set_from_result"].description
    assert server.REVIEW_FROM_RESULT_LIMITS not in (
        specs["olmoearth_review_set_from_result"].description
    )
    srv = server.build_server(_app())
    listed = {t.name: t for t in await _list(srv)}
    text = listed["olmoearth_review_set_from_result"].description
    assert text is not None and server.REVIEW_FROM_RESULT_LIMITS in text
    words = _words(server.REVIEW_FROM_RESULT_LIMITS)
    assert words.startswith("Before running it, tell the user")
    assert "no recorded experiment supports this order on a Studio result" in words
    assert "small grid of single pixels (about 100 points" in words
    assert "offer olmoearth_check_map instead" in words
    assert "run it only if the user still asks" in words
    assert "Neither job applies to an embeddings result." in words


@pytest.mark.asyncio
async def test_a_studio_tool_signed_out_says_to_sign_in_and_sends_nothing(
    httpx_mock: HTTPXMock,
) -> None:
    srv = server.build_server(_app())
    result = await _call(srv, "olmoearth_load_context", {})
    assert result.isError is True
    body = json.loads(_text(result))
    assert body["needs_sign_in"] is True and "olmoearth_sign_in" in body["hint"]
    assert _view(result)["needs_sign_in"] is True
    assert not httpx_mock.get_requests()


@pytest.mark.asyncio
async def test_a_key_studio_refuses_offers_the_sign_in(
    httpx_mock: HTTPXMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    from olmoearth_agent.studio.client import _refusal_cause

    assert server.KEY_REFUSED in _refusal_cause("GET", "/users/me", 401)[0]
    monkeypatch.setenv(auth.KEY_ENV, KEY)
    httpx_mock.add_response(
        url=f"{BASE}/users/me", status_code=401, json={"errors": []}
    )
    srv = server.build_server(_app())
    result = await _call(srv, "olmoearth_load_context", {})
    assert result.isError is True
    assert _view(result)["needs_sign_in"] is True
    assert KEY not in result.model_dump_json(by_alias=True)


def _studio_ok(httpx_mock: HTTPXMock, *, times: int = 1) -> None:
    for _ in range(times):
        httpx_mock.add_response(
            url=f"{BASE}/users/me",
            json={
                "records": [
                    {"id": "u1", "name": "Ana", "organizations": [{"name": "Lab"}]}
                ]
            },
        )
        httpx_mock.add_response(
            url=f"{BASE}/projects/search",
            method="POST",
            json={"records": [{"id": "p1", "name": "Wetlands"}], "meta": {"total": 1}},
        )


async def _post_key(url: str, key: str) -> str:
    """Submit ``key`` on the sign-in page, as the user's browser does."""
    port = int(url.split(":")[2].split("/")[0])
    path = url.split(str(port), 1)[1]
    form = f"api_key={key}&remember=1".encode()
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    writer.write(
        (
            f"POST {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n"
            f"Origin: http://127.0.0.1:{port}\r\n"
            "Content-Type: application/x-www-form-urlencoded\r\n"
            f"Content-Length: {len(form)}\r\n\r\n"
        ).encode()
        + form
    )
    await writer.drain()
    page = (await reader.read()).decode()
    writer.close()
    return page


@pytest.mark.asyncio
async def test_no_tool_takes_the_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """The key is typed on the 127.0.0.1 page only: no tool's schema takes one."""
    monkeypatch.setattr(server, "client_renders_views", lambda _srv: True)
    srv = server.build_server(_app())
    for tool in await _list(srv):
        props = (tool.inputSchema or {}).get("properties") or {}
        assert not {"api_key", "key", "token"} & set(props), tool.name
    assert "olmoearth_app_sign_in" not in server.APP_TOOLS


@pytest.mark.asyncio
async def test_signing_in_on_the_page_then_using_studio(
    httpx_mock: HTTPXMock, tmp_path: Path
) -> None:
    _studio_ok(httpx_mock, times=2)
    app = _app()
    srv = server.build_server(app)
    began = await _call(srv, "olmoearth_app_begin_sign_in", {})
    url = began.structuredContent["url"]
    assert (
        url.startswith("http://127.0.0.1:") and began.structuredContent["seconds"] > 0
    )
    page = await _post_key(url, KEY)
    assert "Signed in as Ana (Lab)" in page and KEY not in page
    assert auth.read_key_file(tmp_path / "cfg" / "studio.env") == KEY
    status = await _call(srv, "olmoearth_app_status", {})
    assert status.structuredContent["signed_in"] is True
    assert status.structuredContent["account"]["user_name"] == "Ana"
    context = await _call(srv, "olmoearth_load_context", {})
    assert context.isError is False
    assert _view(context)["projects"][0]["name"] == "Wetlands"
    # r5-c S01: the panel's data names the key's source: a key of the page's.
    assert _view(context)["source"] == "signed-in"
    told = await _call(srv, "olmoearth_sign_in", {})
    # Without panels the note offers no panel and no Sign out (r5-c S01).
    assert json.loads(_text(told))["note"] == "Signed in to OlmoEarth Studio."
    for result in (began, status, context, told):
        assert KEY not in result.model_dump_json(by_alias=True)
    assert all(
        r.headers["Authorization"] == f"Bearer {KEY}" for r in httpx_mock.get_requests()
    )
    await app.aclose()


@pytest.mark.asyncio
async def test_without_panels_the_sign_in_is_a_link_and_the_sheet_a_file(
    _roots: Path,
) -> None:
    app = _app()
    srv = server.build_server(app)
    told = json.loads(_text(await _call(srv, "olmoearth_sign_in", {})))
    assert told["signed_in"] is False
    assert told["sign_in_url"].startswith("http://127.0.0.1:")
    assert told["sign_in_url"] in told["note"] and auth.KEY_ENV in told["note"]
    (_roots / "p_to_label.csv").write_text(
        "order,window_index,row,col,map_class,stratum,wrong,reference_class\n"
        "1,7,0,7,water,0,,\n",
        encoding="utf-8",
    )
    (_roots / "p.json").write_text(
        json.dumps(
            {"design": "confidence", "population": {"score_kind": "class_scores"}}
        ),
        encoding="utf-8",
    )
    opened = json.loads(
        _text(
            await _call(
                srv,
                "olmoearth_label_sheet",
                {"labels_csv_path": str(_roots / "p_to_label.csv")},
            )
        )
    )
    assert opened["label_rule"] == "wrong"
    assert "no panel" in opened["note"] and "wrong column" in opened["note"]
    assert opened["design_path"] in opened["note"]
    await app.aclose()


@pytest.mark.asyncio
async def test_with_panels_the_sign_in_is_the_panel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(server, "client_renders_views", lambda _srv: True)
    app = _app()
    srv = server.build_server(app)
    told = json.loads(_text(await _call(srv, "olmoearth_sign_in", {})))
    assert "panel is open" in told["note"]
    # A link too, should the panel not appear; the panel's button opens the same page.
    assert told["sign_in_url"] == app.sign_in_page.url
    assert told["sign_in_url"] in told["note"]
    began = await _call(srv, "olmoearth_app_begin_sign_in", {})
    assert began.structuredContent["url"] == told["sign_in_url"]
    assert began.structuredContent["sign_ins"] == 0
    await app.aclose()


@pytest.mark.asyncio
async def test_with_panels_the_sign_in_note_offers_no_sign_out_for_the_apps_own_key(
    httpx_mock: HTTPXMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """r5-c S01: with the key the app was installed with (OLMOEARTH_API_KEY)
    the panel's account card draws no Sign out, so the sign-in tool's note
    offers none and says who can remove that key; a key entered on the
    sign-in page keeps "can sign out". The key appears nowhere."""
    monkeypatch.setenv(auth.KEY_ENV, KEY)
    monkeypatch.setattr(server, "client_renders_views", lambda _srv: True)
    _studio_ok(httpx_mock)
    app = _app()
    srv = server.build_server(app)
    result = await _call(srv, "olmoearth_sign_in", {})
    told = json.loads(_text(result))
    assert told["signed_in"] is True and told["source"] == "environment"
    assert "can sign out" not in told["note"]
    assert "installed" in told["note"] and "no Sign out" in told["note"]
    assert KEY not in result.model_dump_json(by_alias=True)
    # A key entered on the sign-in page (it comes before the environment's).
    app.keys.sign_in(KEY, auth.Account("Ana", "Lab", 1), remember=False)
    page_key = json.loads(_text(await _call(srv, "olmoearth_sign_in", {})))
    assert page_key["source"] == "signed-in"
    assert page_key["note"] == (
        "Signed in to OlmoEarth Studio. The panel shows the account and can sign out."
    )
    await app.aclose()


@pytest.mark.asyncio
async def test_the_projects_panel_data_names_the_keys_source(
    httpx_mock: HTTPXMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """r5-c S01: the projects tool's panel data carries the account and the
    key's source, so the panel's first drawing draws no Sign out for the
    app's own key (OLMOEARTH_API_KEY); the key appears nowhere."""
    monkeypatch.setenv(auth.KEY_ENV, KEY)
    _studio_ok(httpx_mock, times=2)  # the tool's read, then the account's check
    app = _app()
    srv = server.build_server(app)
    context = await _call(srv, "olmoearth_load_context", {})
    assert context.isError is False
    view = _view(context)
    assert view["source"] == "environment"
    assert view["account"]["user_name"] == "Ana"
    assert KEY not in context.model_dump_json(by_alias=True)
    await app.aclose()


@pytest.mark.asyncio
async def test_start_card_signed_out_names_the_four_jobs(
    httpx_mock: HTTPXMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Signed out, the start card names the jobs offered and the sign-in, and
    asks Studio nothing; without panels its text says to name them, with
    panels not to. While FIRST_IN_STUDIO is False (the default, until the
    owner rules D1), "Find the windows to look at first" is not offered."""
    assert server.FIRST_IN_STUDIO is False
    offered = server.start_jobs()
    assert [j["key"] for j in offered] == ["check", "which", "run"]
    srv = server.build_server(_app())
    told = await _call(srv, "olmoearth_start", {})
    assert told.isError is False
    body = json.loads(_text(told))
    assert body["ok"] is True and body["signed_in"] is False
    assert [j["job"] for j in body["jobs"]] == [j["label"] for j in offered]
    assert [j["needs"] for j in body["jobs"]] == [j["needs"] for j in offered]
    assert "no panel" in body["note"] and "olmoearth_sign_in" in body["note"]
    card = _view(told)
    assert card["status"]["signed_in"] is False
    assert set(card["status"]) == set(server.START_STATUS_KEYS)  # no key file path
    assert len(card["jobs"]) == 3
    keys = ("key", "label", "line", "message")
    assert card["jobs"] == [
        {**{k: j[k] for k in keys}, "main": j["key"] == "check"} for j in offered
    ]
    assert all(j["key"] != "first" for j in card["jobs"])
    assert not httpx_mock.get_requests()
    monkeypatch.setattr(server, "client_renders_views", lambda _srv: True)
    shown = json.loads(_text(await _call(srv, "olmoearth_start", {})))
    assert "Do not list the jobs again" in shown["note"]
    assert "the card offers the sign-in" in shown["note"]
    assert "olmoearth_sign_in" not in shown["note"]
    assert not httpx_mock.get_requests()


@pytest.mark.asyncio
async def test_start_card_signed_in(
    httpx_mock: HTTPXMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With a key Studio accepts, the card says signed in, with the account, and the
    key appears nowhere in the result."""
    monkeypatch.setenv(auth.KEY_ENV, KEY)
    _studio_ok(httpx_mock)
    monkeypatch.setattr(server, "client_renders_views", lambda _srv: True)
    app = _app()
    srv = server.build_server(app)
    told = await _call(srv, "olmoearth_start", {})
    body = json.loads(_text(told))
    assert body["signed_in"] is True
    assert body["note"] == server.START_CARD_NOTE  # no sign-in sentence
    card = _view(told)
    assert card["status"]["signed_in"] is True and not card["status"]["key_problem"]
    assert card["status"]["account"]["user_name"] == "Ana"
    assert KEY not in told.model_dump_json(by_alias=True)
    await app.aclose()


@pytest.mark.asyncio
async def test_prompts_are_the_start_jobs() -> None:
    """The server's MCP prompts are the start card's jobs: no arguments, and each
    gives the same request the card's button sends, as the user's message."""
    from mcp.server.lowlevel import NotificationOptions

    srv = server.build_server(_app())
    listed = await srv.request_handlers[types.ListPromptsRequest](
        types.ListPromptsRequest(method="prompts/list")
    )
    prompts = listed.root.prompts
    assert [(p.name, p.title, p.description) for p in prompts] == [
        (j["prompt"], j["label"], j["line"]) for j in server.start_jobs()
    ]
    assert "windows_to_look_at_first" not in {p.name for p in prompts}
    assert all(not p.arguments for p in prompts)
    for job in server.start_jobs():
        got = await srv.request_handlers[types.GetPromptRequest](
            types.GetPromptRequest(
                method="prompts/get",
                params=types.GetPromptRequestParams(name=job["prompt"]),
            )
        )
        (message,) = got.root.messages
        assert message.role == "user"
        assert isinstance(message.content, types.TextContent)
        assert message.content.text == job["message"]
    with pytest.raises(ValueError, match="unknown prompt"):
        await srv.request_handlers[types.GetPromptRequest](
            types.GetPromptRequest(
                method="prompts/get",
                params=types.GetPromptRequestParams(name="no_such_job"),
            )
        )
    assert srv.get_capabilities(NotificationOptions(), {}).prompts is not None


def test_start_job_strings() -> None:
    """Four jobs, short one-line labels and lines, each message a request with a
    verb that ends at the projects card; the start card is the app's own tool.
    The check is the main job (``main`` on it only); its line says what the
    user gets in plain words, with the panel's words ("squares", "range"),
    and promises no number before any label. The run's line (the reader
    test's B/S5) says what the press waits for. r5-e S2: each line sits inside
    its button, so it is at most 85 characters (two lines at 380 px); the
    check's line no longer brings in the parts a map is saved in."""
    jobs = server.START_JOBS
    assert len(jobs) == 4
    assert [j["key"] for j in jobs] == ["check", "first", "which", "run"]
    assert len({j["prompt"] for j in jobs}) == 4
    text_keys = {"key", "prompt", "label", "line", "needs", "message"}
    for i, job in enumerate(jobs):
        assert set(job) == (text_keys | {"main"} if i == 0 else text_keys)
        assert len(job["label"]) <= 72 and "\n" not in job["label"]
        # r5-e S2: a line inside its button, two lines at 380 px at most.
        assert len(job["line"]) <= 85 and "\n" not in job["line"]
        assert job["message"].split()[0] in {"Check", "Find", "Say", "Run"}
        assert "Show my Studio projects so I can pick" in job["message"]
        for key in text_keys:
            text = job[key]
            assert isinstance(text, str)
            assert chr(0x2028) not in text and chr(0x2029) not in text
    assert jobs[0]["key"] == "check" and jobs[0]["main"] is True
    line = jobs[0]["line"]
    assert not re.search(r"\d", line) and "%" not in line
    assert "squares" in line and "wrong" in line and "range" in line
    assert "random" in line and "window" not in line  # the panel's one word
    assert "part" not in line  # r5-e S2: a first-time user cannot place it
    # The look-first job, when offered, names the one case it is for.
    assert "class map" in jobs[1]["line"] and "top-k" in jobs[1]["needs"]
    assert "can write" in jobs[3]["line"]
    assert "once you pick a model, an area and dates" in jobs[3]["line"]
    assert "olmoearth_start" not in build_default_registry().names()
    assert server.OWN_TOOLS["olmoearth_start"][0] == "start"
    assert "olmoearth_start (the start card)" in _words(server.MCP_RULES)


@pytest.mark.asyncio
async def test_start_card_names_the_main_job_first() -> None:
    """The model's jobs give what each gives (its line) and mark the main one;
    the no-panel note names the main job first; the card's jobs mark it too."""
    srv = server.build_server(_app())
    told = await _call(srv, "olmoearth_start", {})
    body = json.loads(_text(told))
    first = body["jobs"][0]
    assert first["job"] == server.START_JOBS[0]["label"]
    assert first["gives"] == server.START_JOBS[0]["line"] and first["main"] is True
    assert all("main" not in j for j in body["jobs"][1:])
    assert all(j["gives"] for j in body["jobs"])
    assert "main job first" in body["note"]
    card = _view(told)
    assert card["jobs"][0]["main"] is True
    assert [j["main"] for j in card["jobs"][1:]] == [False] * (len(card["jobs"]) - 1)


@pytest.mark.asyncio
async def test_first_in_studio_decides_whether_the_look_first_job_is_offered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One constant decides it: False (the default) leaves the job off the
    start card and the menu, and the projects panel reads False; True offers
    it on both (the code for True is kept), and the panel reads True."""

    async def handler(_args: dict[str, Any], _ctx: Any) -> dict[str, Any]:
        return {"projects": [{"id": "p1", "name": "Wetlands"}], "total": 1}

    registry = ToolRegistry()
    registry.register(
        RegisteredTool(
            ToolSpec(
                "olmoearth_load_context",
                "Projects.",
                {"type": "object", "properties": {}},
            ),
            handler,
        )
    )
    app = _app(registry)
    srv = server.build_server(app)
    shown = await app.call_agent_tool("olmoearth_load_context", {}, views=True)
    assert _view(shown)["first_in_studio"] is False
    monkeypatch.setattr(server, "FIRST_IN_STUDIO", True)
    assert [j["key"] for j in server.start_jobs()] == ["check", "first", "which", "run"]
    card = _view(await _call(srv, "olmoearth_start", {}))
    assert [j["key"] for j in card["jobs"]] == ["check", "first", "which", "run"]
    listed = await srv.request_handlers[types.ListPromptsRequest](
        types.ListPromptsRequest(method="prompts/list")
    )
    assert "windows_to_look_at_first" in {p.name for p in listed.root.prompts}
    again = await app.call_agent_tool("olmoearth_load_context", {}, views=True)
    assert _view(again)["first_in_studio"] is True
    await app.aclose()


@pytest.mark.asyncio
async def test_a_view_tool_signed_out_asks_for_the_sign_in() -> None:
    srv = server.build_server(_app())
    result = await _call(srv, "olmoearth_app_projects", {})
    assert result.isError is True
    assert result.structuredContent["needs_sign_in"] is True


@pytest.mark.asyncio
async def test_the_projects_view_tool_gives_description_and_creation_time(
    httpx_mock: HTTPXMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(auth.KEY_ENV, KEY)
    httpx_mock.add_response(
        url=f"{BASE}/projects/search",
        json={
            "records": [
                {
                    "id": "p1",
                    "name": "Fields",
                    "description": "  Crop map  " + "x" * 400,
                    "creation_time": "2026-10-01T12:00:00Z",
                    "requester_id": "u1",
                },
                {"id": "p2", "name": ""},
            ]
        },
    )
    app = _app()
    out = await app.app_tool("olmoearth_app_projects", {})
    first, second = out["projects"]
    assert first["description"].startswith("Crop map")
    assert len(first["description"]) == 300
    assert first["creation_time"] == "2026-10-01T12:00:00Z"
    assert set(first) == {"id", "name", "description", "creation_time"}  # no person ids
    assert second == {
        "id": "p2",
        "name": "(unnamed)",
        "description": "",
        "creation_time": "",
    }
    await app.aclose()


@pytest.mark.asyncio
async def test_the_project_items_view_tool_gives_creation_times_and_no_geometry(
    httpx_mock: HTTPXMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(auth.KEY_ENV, KEY)
    httpx_mock.add_response(
        url=f"{BASE}/areas/search",
        json={
            "records": [
                {
                    "id": "a1",
                    "name": "Delta",
                    "project_id": "p1",
                    "creation_time": "2026-10-02T00:00:00Z",
                    "geom": {"type": "Point", "coordinates": [10.0, 50.0]},
                }
            ]
        },
    )
    httpx_mock.add_response(
        url=f"{BASE}/predictions/search",
        json={
            "records": [
                {
                    "id": "pr1",
                    "name": "Flood run",
                    "project_id": "p1",
                    "status": "completed",
                    "model_id": "m1",
                    "requester_id": "u1",
                    "start_time": "2024-04-01T00:00:00Z",
                    "end_time": "2024-07-01T00:00:00Z",
                    "result": {"download_token": "secret", "tile_urls": ["x"]},
                }
            ]
        },
    )
    httpx_mock.add_response(
        url=f"{BASE}/models/search",
        json={
            "records": [
                {
                    "id": "m1",
                    "name": "Crops",
                    "model_type": "fine_tuned",
                    "requester_id": "u1",
                    "checkpoint_token": "secret",
                    "wizard_answers": {
                        "prediction_type": "per_pixel_classification",
                        "encoder_variant": "tiny",
                        "imagery_sources": ["sentinel2_l2a"],
                    },
                },
                {"id": "m9", "name": "Elsewhere", "wizard_answers": {}},
            ]
        },
    )
    app = _app()
    out = await app.app_tool("olmoearth_app_project_items", {"project_id": "p1"})
    assert out["areas"] == [
        {"id": "a1", "name": "Delta", "creation_time": "2026-10-02T00:00:00Z"}
    ]
    assert out["predictions"] == [
        {
            "id": "pr1",
            "name": "Flood run",
            "status": "completed",
            "model_id": "m1",
            "creation_time": "",
            "months": [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            "years": "2024",
            "span": "Apr to Jun 2024",
        }
    ]
    # Only the project's models, as glyph fields: no person, token or selector.
    assert out["models_read"] is True
    assert [m["id"] for m in out["models"]] == ["m1"]
    model = out["models"][0]
    assert (model["grain"], model["kind"], model["size"]) == (
        "pixel",
        "classes",
        "Tiny",
    )
    assert model["sensors"] == ["S2"]
    text = json.dumps(out)
    assert "secret" not in text and "u1" not in text and "tile_urls" not in text
    await app.aclose()


@pytest.mark.asyncio
async def test_project_items_says_when_the_models_were_not_read(
    httpx_mock: HTTPXMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(auth.KEY_ENV, KEY)
    httpx_mock.add_response(url=f"{BASE}/areas/search", json={"records": []})
    httpx_mock.add_response(
        url=f"{BASE}/predictions/search",
        json={"records": [{"id": "pr1", "name": "Run", "model_id": "m1"}]},
    )
    httpx_mock.add_response(url=f"{BASE}/models/search", status_code=405)
    app = _app()
    out = await app.app_tool("olmoearth_app_project_items", {"project_id": "p1"})
    assert out["models"] == [] and out["models_read"] is False
    assert out["predictions"][0]["months"] is None  # no dates: "not recorded"
    await app.aclose()


@pytest.mark.asyncio
async def test_project_glyphs_count_each_projects_predictions_by_month(
    httpx_mock: HTTPXMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(auth.KEY_ENV, KEY)
    for _ in range(2):
        httpx_mock.add_response(
            url=f"{BASE}/predictions/search",
            json={
                "records": [
                    {"id": "a", "start_time": "2024-01-01", "end_time": "2024-02-01"},
                    {"id": "b", "start_time": "2024-01-10", "end_time": "2024-01-20"},
                ]
            },
        )
    app = _app()
    out = await app.app_tool(
        "olmoearth_app_project_glyphs", {"project_ids": ["p1", "p2", "p1"]}
    )
    assert set(out["glyphs"]) == {"p1", "p2"}  # duplicates read once
    ring = out["glyphs"]["p1"]
    assert ring["months"] == [2] + [0] * 11  # two predictions in January: step 2
    assert (ring["count"], ring["dated"], ring["capped"]) == (2, 2, False)
    with pytest.raises(ValueError):
        await app.app_tool(
            "olmoearth_app_project_glyphs", {"project_ids": [f"p{i}" for i in range(9)]}
        )
    await app.aclose()


@pytest.mark.asyncio
async def test_a_tool_that_catches_the_sign_in_error_still_asks_for_it() -> None:
    """olmoearth_pixel_value answers ok with the error as its reason."""
    srv = server.build_server(_app())
    result = await _call(
        srv, "olmoearth_pixel_value", {"result_id": "r1", "lon": 10.0, "lat": 50.0}
    )
    assert result.meta is None  # the pixel value opens no panel
    body = json.loads(_text(result))
    assert body["needs_sign_in"] is True and "olmoearth_sign_in" in body["hint"]


@pytest.mark.asyncio
async def test_a_tool_that_catches_a_refused_key_names_the_sign_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A caught 401 (the reason of an ok result) leads to olmoearth_sign_in too."""
    monkeypatch.setenv(auth.KEY_ENV, KEY)

    async def handler(_args: dict[str, Any], _ctx: Any) -> dict[str, Any]:
        return {
            "available": False,
            "reason": f"{server.KEY_REFUSED} (HTTP 401) on GET /results/r1",
        }

    registry = ToolRegistry()
    registry.register(
        RegisteredTool(
            ToolSpec(
                "olmoearth_fake_point", "A point.", {"type": "object", "properties": {}}
            ),
            handler,
        )
    )
    app = _app(registry)
    srv = server.build_server(app)
    result = await _call(srv, "olmoearth_fake_point", {})
    body = json.loads(_text(result))
    assert body["needs_sign_in"] is True
    assert server.SIGN_IN_AGAIN in body["hint"]
    assert (await app.account())["key_problem"]  # the refusal is remembered
    await app.aclose()


@pytest.mark.asyncio
async def test_a_new_task_and_a_new_key_start_a_new_run_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _app()
    srv = server.build_server(app)
    await _call(srv, "olmoearth_load_context", {})  # signed out: one failed call
    first = app.state
    assert first.provenance.entries and first.tool_failures
    told = await _call(srv, "olmoearth_new_task", {})
    assert told.isError is False and app.state is not first
    assert not app.state.provenance.entries and not app.state.tool_failures
    await _call(srv, "olmoearth_load_context", {})
    before_key = app.state
    # A first sign-in keeps the task's records (review wf_e14d1b98-18f).
    app.keys.sign_in(KEY, auth.Account("Ana", "Lab", 1), remember=False)
    await app.studio()
    assert app.state is before_key and app.state.provenance.entries
    # Another account's key starts a new run state.
    app.keys.sign_in("sk_other_key_789", auth.Account("Bo", "Lab", 1), remember=False)
    await app.studio()
    assert app.state is not before_key and not app.state.provenance.entries
    await app.aclose()


@pytest.mark.asyncio
async def test_a_refused_key_is_asked_about_once(
    httpx_mock: HTTPXMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A key Studio refused is not signed in, and not checked again on every poll."""
    monkeypatch.setenv(auth.KEY_ENV, KEY)
    httpx_mock.add_response(
        url=f"{BASE}/users/me", status_code=401, json={"errors": []}
    )
    app = _app()
    app.keys.account = auth.Account(
        "Ana", "Lab", 1
    )  # an account read before the refusal
    srv = server.build_server(app)
    refused = await _call(srv, "olmoearth_load_context", {})
    assert _view(refused)["needs_sign_in"] is True and app.keys.account is None
    for _ in range(3):
        status = await _call(srv, "olmoearth_app_status", {})
        assert status.structuredContent["key_problem"]
    told = json.loads(_text(await _call(srv, "olmoearth_sign_in", {})))
    assert told["key_problem"] and "sign_in_url" in told
    assert len(httpx_mock.get_requests()) == 1
    await app.aclose()


def test_only_a_caught_error_reason_asks_for_the_sign_in() -> None:
    mark = auth.NOT_SIGNED_IN_MARK
    assert server._caught_sign_in({"available": False, "reason": "failed: " + mark})
    assert not server._caught_sign_in(
        {"projects": [{"name": mark + " (a project name)"}]}
    )
    assert not server._caught_sign_in({"available": True, "reason": mark})
    assert not server._caught_sign_in(None)


def test_a_relative_output_root_is_made_absolute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    env = {"OLMOEARTH_OUTPUT_ROOT": "rel-outputs"}
    root = server._enter_output_root(env)
    assert root == (tmp_path / "rel-outputs").resolve()
    assert env["OLMOEARTH_OUTPUT_ROOT"] == str(root)


def test_the_server_works_in_its_output_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Started in /, as Claude Desktop may start it, the read guard gets no / root."""
    from olmoearth_agent.tools.review_set import read_roots

    monkeypatch.chdir("/")
    root = tmp_path / "new-outputs"
    env = {"OLMOEARTH_OUTPUT_ROOT": str(root)}
    monkeypatch.setenv("OLMOEARTH_OUTPUT_ROOT", str(root))
    entered = server._enter_output_root(env)
    assert entered == root.resolve() and Path.cwd() == root.resolve()
    assert (root.stat().st_mode & 0o777) == 0o700
    assert "/" not in read_roots()


@pytest.mark.asyncio
async def test_a_figure_reaches_the_view_not_the_model(_roots: Path) -> None:
    """A plan whose sheet is labelled right or wrong (a scores map's, which
    shows the map's class): its figure reaches the view at once, never the
    model's text."""
    svg = '<svg xmlns="http://www.w3.org/2000/svg"><text>figure</text></svg>'
    figure = _roots / "plan_x_figure.svg"
    figure.write_text(svg, encoding="utf-8")
    (_roots / "plan_x.json").write_text(
        json.dumps(
            {"design": "random", "population": {"score_kind": "window_confidence"}}
        ),
        encoding="utf-8",
    )
    (_roots / "plan_x_to_label.csv").write_text(
        "order,window_index,row,col,map_class,stratum,wrong,reference_class\n"
        "1,7,0,7,1,0,,\n",
        encoding="utf-8",
    )

    card = [
        ["Squares to label", "5"],
        ["Population", "400 windows", "the valid windows"],
    ]

    async def handler(_args: dict[str, Any], _ctx: Any) -> dict[str, Any]:
        return {
            "figure_path": str(figure),
            "figure_caption": "Fig. 1",
            "labels_csv_path": str(_roots / "plan_x_to_label.csv"),
            "n_population": 400,
            "must_state": ["the rate is of the valid windows"],
            "card": card,
        }

    registry = ToolRegistry()
    registry.register(
        RegisteredTool(
            ToolSpec(
                "olmoearth_plan_label_sample",
                "A plan.",
                {"type": "object", "properties": {}},
            ),
            handler,
        )
    )
    app = _app(registry)
    srv = server.build_server(app)
    result = await _call(srv, "olmoearth_plan_label_sample", {})
    view = _view(result)
    assert view["figure"] == {"svg": svg, "caption": "Fig. 1"}
    assert view["labels_csv_path"].endswith("plan_x_to_label.csv")
    # The card's rows reach the view as the tool wrote them; no table of fields.
    assert view["card"] == card
    assert "fields" not in view and "reason" not in view
    assert view["must_state"] == ["the rate is of the valid windows"]
    assert "<svg" not in _text(result)
    # The model's copy of the result leaves the card out; the rest stays.
    model_result = json.loads(result.content[-1].text)["result"]
    assert model_result["figure_caption"] == "Fig. 1"
    assert model_result["n_population"] == 400
    assert "card" not in model_result
    assert "Squares to label" not in _text(result)
    # This test client shows no panels: no note about the panel's sheet.
    assert server.PLAN_SHEET_NOTE not in _text(result)
    # With panels, the model reads first that the plan's panel opens the sheet.
    with_panels = await app.call_agent_tool(
        "olmoearth_plan_label_sample", {}, views=True
    )
    assert with_panels.content[0].text == server.PLAN_SHEET_NOTE
    assert json.loads(with_panels.content[1].text)["ok"] is True
    assert "card" not in json.loads(with_panels.content[1].text)["result"]
    await app.aclose()


@pytest.mark.asyncio
async def test_the_view_keeps_well_formed_card_rows_and_a_refusals_reason() -> None:
    """Card rows of another shape are dropped and the card is capped; a refusal
    the tool returned shows its reason, and a reason beside no refusal flag
    does not."""
    good = [[f"Row {i}", f"value {i}"] for i in range(10)]
    bad: list[Any] = [
        ["only a name"],
        ["Name", 5],
        ["Name", "value", "detail", "extra"],
        "not a row",
        ["Long", "x" * (server.CARD_VIEW_CHARS + 1)],
    ]
    answers: dict[str, dict[str, Any]] = {
        "olmoearth_review_set": {"card": [*bad, ["Kept", "v", "d"], *good]},
        "olmoearth_certify_zone": {"certified": False, "reason": "needs random"},
        "olmoearth_estimate_map_error": {"reason": "not a refusal", "card": bad},
        "olmoearth_plan_label_sample": {"card": "not a list"},
    }
    registry = ToolRegistry()
    for name, answer in answers.items():

        async def handler(
            _args: dict[str, Any], _ctx: Any, answer: dict[str, Any] = answer
        ) -> dict[str, Any]:
            return answer

        registry.register(
            RegisteredTool(
                ToolSpec(name, "A tool.", {"type": "object", "properties": {}}),
                handler,
            )
        )
    app = _app(registry)
    srv = server.build_server(app)
    ranked = _view(await _call(srv, "olmoearth_review_set", {}))
    assert ranked["card"] == [["Kept", "v", "d"], *good[: server.CARD_VIEW_ROWS - 1]]
    refused = _view(await _call(srv, "olmoearth_certify_zone", {}))
    assert refused["reason"] == "needs random" and "card" not in refused
    estimated = _view(await _call(srv, "olmoearth_estimate_map_error", {}))
    assert "reason" not in estimated and "card" not in estimated
    planned = _view(await _call(srv, "olmoearth_plan_label_sample", {}))
    assert "card" not in planned
    await app.aclose()


@pytest.mark.asyncio
async def test_a_tool_without_a_panel_sends_no_view_and_names_the_sign_in() -> None:
    """An intermediate step opens no panel, so its text leads to the sign-in."""
    srv = server.build_server(_app())
    result = await _call(srv, "olmoearth_get_prediction", {"prediction_id": "p1"})
    assert result.meta is None  # no view data for a tool with no panel
    body = json.loads(_text(result))
    assert body["needs_sign_in"] is True and "olmoearth_sign_in" in body["hint"]


@pytest.mark.asyncio
async def test_a_refused_key_on_a_tool_without_a_panel_names_the_sign_in(
    httpx_mock: HTTPXMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(auth.KEY_ENV, KEY)
    httpx_mock.add_response(
        url=f"{BASE}/predictions/p1", status_code=401, json={"errors": []}
    )
    srv = server.build_server(_app())
    result = await _call(srv, "olmoearth_get_prediction", {"prediction_id": "p1"})
    assert result.isError is True and result.meta is None
    body = json.loads(_text(result))
    assert body["needs_sign_in"] is True
    assert server.SIGN_IN_AGAIN in body["hint"]
    assert "Do not retry" not in body["hint"]  # replaced, not contradicted
    assert KEY not in result.model_dump_json(by_alias=True)


@pytest.mark.asyncio
async def test_label_sheet_rows_and_places_stay_out_of_the_models_text(
    _roots: Path,
) -> None:
    (_roots / "p_to_label.csv").write_text(
        "order,window_index,row,col,map_class,lon,lat,stratum,wrong,reference_class\n"
        "1,7,0,7,water,12.3456,48.7654,0,,\n",
        encoding="utf-8",
    )
    srv = server.build_server(_app())
    path = str(_roots / "p_to_label.csv")
    result = await _call(srv, "olmoearth_label_sheet", {"labels_csv_path": path})
    assert result.isError is False
    # The tool's result carries the sheet's counts only, in its text and as the
    # view's data: no row and no place anywhere in it.
    whole = result.model_dump_json(by_alias=True)
    assert "48.7654" not in whole and "12.3456" not in whole
    view = _view(result)
    assert "rows" not in view and "columns" not in view
    assert set(view) == {
        "path",
        "name",
        "kind",
        "design_path",
        "label_rule",
        "blind",
        "takes_unjudged",
        "n_rows",
        "n_labelled",
        "n_unjudged",
        "progress",
        "open_label",
        "job",
        "finish_label",
        "finish_call",
        "finish_message",
        "version",
        # The sheet's labelling rule and graded mark (the r4 pass-2 contract).
        "reading_rule",
        "rule_scope",
        "rule_source",
        # The owner's rule is for kelp canopy maps only (r4i F9).
        "rule_limit",
        "graded",
    }
    assert view["progress"] == "0 of 1 labelled"
    assert view["open_label"] == "Open the sheet here"
    # The model's text has no version: it is the panel's, for its saves.
    assert view["version"] not in _text(result)
    # The view reads the page copy from the file: a handle per window, no row
    # and no place; the place comes at the press on Street map, by handle.
    page = await _call(srv, "olmoearth_app_sheet", {"labels_csv_path": path})
    copy_ = page.structuredContent
    assert copy_ is not None and "rows" not in copy_
    assert "48.7654" not in json.dumps(copy_) and "12.3456" not in json.dumps(copy_)
    (window,) = copy_["windows"]
    link = await _call(
        srv,
        "olmoearth_app_window_link",
        {
            "labels_csv_path": path,
            "sheet_id": copy_["sheet_id"],
            "handle": window["handle"],
        },
    )
    assert link.structuredContent == {
        "url": "https://www.openstreetmap.org/?mlat=48.7654&mlon=12.3456"
        "#map=17/48.7654/12.3456"
    }
    assert "48.7654" not in _text(link)
    by_handle = {
        "labels_csv_path": path,
        "sheet_id": copy_["sheet_id"],
        "face": copy_["face"],
        "rule": copy_["label_rule"],
    }
    saved = await _call(
        srv,
        "olmoearth_app_save_labels",
        {**by_handle, "answers": [{"handle": window["handle"], "answer": "1"}]},
    )
    assert saved.isError is False and saved.structuredContent["n_labelled"] == 1
    # A panel of an older server sends labels by window index: refused.
    older = await _call(
        srv,
        "olmoearth_app_save_labels",
        {"labels_csv_path": path, "labels": [{"window_index": 7, "wrong": "0"}]},
    )
    assert older.isError is True and sheet.OLDER_PANEL in _text(older)
    stale = await _call(
        srv,
        "olmoearth_app_window_link",
        {
            "labels_csv_path": path,
            "sheet_id": "s" + "a" * 16,
            "handle": window["handle"],
        },
    )
    assert stale.isError is True and sheet.CHANGED_UNDER_PAGE in _text(stale)
    refused = await _call(
        srv, "olmoearth_label_sheet", {"labels_csv_path": "/etc/passwd"}
    )
    assert refused.isError is True


@pytest.mark.asyncio
async def test_the_plan_card_reads_how_far_its_sheet_is_labelled(
    _roots: Path,
) -> None:
    """olmoearth_app_sheet_progress: the model reads {"ok": true}; the view gets
    the counts and the button's words, never a row; a sheet that is not a
    plan's is refused."""
    path = _roots / "p_to_label.csv"
    path.write_text(
        "order,window_index,row,col,map_class,lon,lat,stratum,wrong,reference_class\n"
        "1,7,0,7,water,12.3456,48.7654,0,,\n"
        "2,3,0,3,trees,12.3457,48.7655,0,,\n",
        encoding="utf-8",
    )
    srv = server.build_server(_app())
    args = {"labels_csv_path": str(path)}
    first = await _call(srv, "olmoearth_app_sheet_progress", args)
    assert first.isError is False
    assert json.loads(_text(first)) == {"ok": True}
    got = first.structuredContent
    assert got is not None and "rows" not in got
    assert (got["n_rows"], got["n_labelled"]) == (2, 0)
    assert (got["progress"], got["open_label"]) == (
        "0 of 2 labelled",
        "Open the sheet here",
    )
    assert "48.765" not in first.model_dump_json(by_alias=True)
    page = (await _call(srv, "olmoearth_app_sheet", args)).structuredContent
    assert page is not None
    sheet_id, handles = sheet.window_handles(str(path))
    (at3,) = [h for h, i in handles.items() if i == 3]
    await _call(
        srv,
        "olmoearth_app_save_labels",
        {
            **args,
            "sheet_id": sheet_id,
            "face": page["face"],
            "rule": page["label_rule"],
            "answers": [{"handle": at3, "answer": "?"}],
        },
    )
    again = (await _call(srv, "olmoearth_app_sheet_progress", args)).structuredContent
    assert again is not None
    assert (again["progress"], again["open_label"]) == (
        "1 of 2 labelled, 1 cannot tell",
        "Continue labelling",
    )
    refused = await _call(
        srv, "olmoearth_app_sheet_progress", {"labels_csv_path": "/etc/passwd"}
    )
    assert refused.isError is True
    missing = await _call(srv, "olmoearth_app_sheet_progress", {})
    assert missing.isError is True


@pytest.mark.asyncio
async def test_the_projects_card_reads_the_sheets_in_progress(_roots: Path) -> None:
    """olmoearth_app_sheets_in_progress: the model reads {"ok": true}; the
    projects card gets the part-labelled sheets, counts only, each with the
    request its Resume sends; no argument, no Studio call."""
    (_roots / "kelp_to_label.csv").write_text(
        "order,window_index,row,col,map_class,lon,lat,stratum,wrong,reference_class\n"
        "1,7,0,7,water,12.3456,48.7654,0,0,\n"
        "2,3,0,3,trees,12.3457,48.7655,0,,\n",
        encoding="utf-8",
    )
    (_roots / "kelp.json").write_text(
        json.dumps({"design": "random", "population": {"score_kind": "class_map"}}),
        encoding="utf-8",
    )
    assert server.APP_TOOLS["olmoearth_app_sheets_in_progress"][1] == server._NO_ARGS
    srv = server.build_server(_app())
    got = await _call(srv, "olmoearth_app_sheets_in_progress", {})
    assert got.isError is False
    assert json.loads(_text(got)) == {"ok": True}
    body = got.structuredContent
    assert body is not None and body["more"] is False
    (row,) = body["sheets"]
    assert (row["name"], row["progress"]) == ("kelp_to_label.csv", "1 of 2 labelled")
    # The path the projects panel tells Claude the sheet by.
    assert row["labels_csv_path"] == row["path"]
    assert row["job_label"] == "Checking how wrong a map is"
    assert row["resume_message"] == sheet.resume_message(row["name"], row["path"])
    assert "olmoearth_label_sheet" in row["resume_message"]
    dumped = got.model_dump_json(by_alias=True)
    assert "48.765" not in dumped and "water" not in dumped
    assert "window_index" not in json.dumps(body)


@pytest.mark.asyncio
@pytest.mark.parametrize("panels", [False, True])
async def test_a_certificate_sheet_is_graded_by_certify_at_the_plans_alpha(
    _roots: Path, monkeypatch: pytest.MonkeyPatch, panels: bool
) -> None:
    """The note after a certificate plan's sheet names certify_zone at the
    alpha the plan fixed, never the estimate; the panel gets the same job."""
    monkeypatch.setattr(server, "client_renders_views", lambda _srv: panels)
    (_roots / "c_to_label.csv").write_text(
        "order,window_index,row,col,map_class,stratum,wrong,reference_class\n"
        "1,7,0,7,1,0,,\n",
        encoding="utf-8",
    )
    (_roots / "c.json").write_text(
        json.dumps(
            {
                "design": "random",
                "population": {"score_kind": "window_confidence"},
                "certificate": {"seed_rule": "from_map", "alpha": 0.1},
            }
        ),
        encoding="utf-8",
    )
    srv = server.build_server(_app())
    result = await _call(
        srv,
        "olmoearth_label_sheet",
        {"labels_csv_path": str(_roots / "c_to_label.csv")},
    )
    told = json.loads(_text(result))
    call = (
        f"call olmoearth_certify_zone with design_path {told['design_path']}, "
        f"labels_path {told['labels_csv_path']} and alpha 0.1 (fixed by the plan)"
    )
    assert call in told["note"] and "estimate_map_error" not in told["note"]
    assert ("panel is open" in told["note"]) is panels
    view = _view(result)
    assert (view["job"], view["finish_label"]) == ("certify", "Certify the zone")
    assert "olmoearth_certify_zone at the plan's alpha" in server.MCP_RULES


def _seen_sheet(root: Path) -> Path:
    """A named scores map's plan (its design names the classes and holds the
    map's class of every window): a sheet labelled with the class seen."""
    (root / "k.json").write_text(
        json.dumps(
            {
                "design": "confidence",
                "population": {
                    "score_kind": "window_confidence",
                    "class_names": {"0": "absent", "1": "present"},
                    "map_class": [0] * 7 + [1],
                },
            }
        ),
        encoding="utf-8",
    )
    path = root / "k_to_label.csv"
    path.write_text(
        "order,window_index,row,col,map_class,lon,lat,stratum,wrong,reference_class\n"
        "1,7,0,7,1,12.3456,48.7654,0,,\n",
        encoding="utf-8",
    )
    return path


@pytest.mark.asyncio
@pytest.mark.parametrize("panels", [False, True])
async def test_a_seen_sheet_sends_no_map_class_and_without_panels_asks_for_wrong(
    _roots: Path, monkeypatch: pytest.MonkeyPatch, panels: bool
) -> None:
    """A seen sheet: the panel gets no map class; with panels the model is
    told the sheet is blind; without, the file route still fills wrong (the
    estimate reads it, and only the panel's save works it out)."""
    monkeypatch.setattr(server, "client_renders_views", lambda _srv: panels)
    path = _seen_sheet(_roots)
    srv = server.build_server(_app())
    result = await _call(srv, "olmoearth_label_sheet", {"labels_csv_path": str(path)})
    told = json.loads(_text(result))
    assert told["label_rule"] == "seen"
    view = _view(result)
    assert view["label_rule"] == "seen" and "rows" not in view
    page = await _call(srv, "olmoearth_app_sheet", {"labels_csv_path": str(path)})
    read = page.structuredContent
    assert read is not None and read["face"] == "seen"
    assert all(set(w) == {"handle", "answer"} for w in read["windows"])
    assert '"map_class"' not in json.dumps(read) and "48.7654" not in json.dumps(read)
    if panels:
        assert server.PLAN_SEEN_NOTE in told["note"]
    else:
        assert "no panel" in told["note"] and "wrong column" in told["note"]
        assert server.PLAN_SEEN_NOTE not in told["note"]


@pytest.mark.asyncio
async def test_a_plan_with_a_seen_sheet_tells_the_model_it_is_blind(
    _roots: Path,
) -> None:
    """With panels, the note before a plan whose sheet is labelled with the
    class seen also says not to ask for right or wrong; a plan whose sheet
    cannot be read gets the plain note."""
    path = _seen_sheet(_roots)

    async def handler(args: dict[str, Any], _ctx: Any) -> dict[str, Any]:
        return {"labels_csv_path": args["labels_csv_path"], "must_state": ["m"]}

    registry = ToolRegistry()
    registry.register(
        RegisteredTool(
            ToolSpec(
                "olmoearth_plan_label_sample",
                "A plan.",
                {"type": "object", "properties": {}},
            ),
            handler,
        )
    )
    app = _app(registry)
    seen = await app.call_agent_tool(
        "olmoearth_plan_label_sample", {"labels_csv_path": str(path)}, views=True
    )
    assert seen.content[0].text == (
        server.PLAN_SHEET_NOTE + " " + server.PLAN_SEEN_NOTE
    )
    missing = await app.call_agent_tool(
        "olmoearth_plan_label_sample",
        {"labels_csv_path": str(_roots / "gone_to_label.csv")},
        views=True,
    )
    assert missing.content[0].text == server.PLAN_SHEET_NOTE
    await app.aclose()


#: A plan's listed windows, as the plan tool writes them: each window's index,
#: row, column, order, map class and stratum; a two-map plan's, each map's class.
LISTED = [
    {
        "order": 1,
        "window_index": 7,
        "row": 0,
        "col": 7,
        "map_class": "present",
        "stratum": 0,
    },
    {
        "order": 2,
        "window_index": 3,
        "row": 0,
        "col": 3,
        "class_a": "water",
        "class_b": "trees",
    },
]


@pytest.mark.asyncio
async def test_with_panels_a_plans_text_lists_no_window(_roots: Path) -> None:
    """The host hands a result's text to its panel too, and the plan's panel
    labels blind by handle: with panels, a plan with a sheet comes without its
    listed windows (no window's index, row, column, order, map class or
    stratum, nor two maps' classes), the seen note's path included, and the
    rest of the plan as it was. Without panels the text keeps the list (the
    file route fills the CSV, or passes the list)."""
    path = _seen_sheet(_roots)

    async def handler(args: dict[str, Any], _ctx: Any) -> dict[str, Any]:
        return {
            "labels_csv_path": args["labels_csv_path"],
            "design_path": str(_roots / "k.json"),
            "n_population": 8,
            "windows": copy.deepcopy(LISTED),
            "n_windows_listed": 2,
            "must_state": ["m"],
        }

    registry = ToolRegistry()
    registry.register(
        RegisteredTool(
            ToolSpec(
                "olmoearth_plan_label_sample",
                "A plan.",
                {"type": "object", "properties": {}},
            ),
            handler,
        )
    )
    app = _app(registry)
    args = {"labels_csv_path": str(path)}
    shown = await app.call_agent_tool("olmoearth_plan_label_sample", args, views=True)
    assert shown.content[0].text == (
        server.PLAN_SHEET_NOTE + " " + server.PLAN_SEEN_NOTE
    )
    told = json.loads(shown.content[1].text)
    plan = told["result"]
    assert told["ok"] is True
    assert "windows" not in plan and "n_windows_listed" not in plan
    assert (plan["labels_csv_path"], plan["n_population"], plan["must_state"]) == (
        str(path),
        8,
        ["m"],
    )
    whole = shown.model_dump_json(by_alias=True)
    for left_out in (
        "window_index",
        "map_class",
        "stratum",
        "class_a",
        "class_b",
        "n_windows_listed",
    ):
        assert left_out not in whole, left_out
    plain = await app.call_agent_tool("olmoearth_plan_label_sample", args, views=False)
    kept = json.loads(_text(plain))["result"]
    assert kept["windows"] == LISTED and kept["n_windows_listed"] == 2
    await app.aclose()


@pytest.mark.asyncio
async def test_a_plans_figure_is_held_on_the_server_until_its_blind_sheet_is_labelled(
    _roots: Path,
) -> None:
    """A seen sheet with no label: the plan's view gets the figure's caption and
    held, never its SVG, and so does a plan whose sheet cannot be read; the
    panel-only olmoearth_app_plan_figure refuses with the card's sentence. Once
    every window is labelled, the plan's view and that tool both give the SVG,
    which never reaches the model's text."""
    path = _seen_sheet(_roots)
    svg = '<svg xmlns="http://www.w3.org/2000/svg"><text>confidence</text></svg>'
    (_roots / "k_figure.svg").write_text(svg, encoding="utf-8")

    async def handler(args: dict[str, Any], _ctx: Any) -> dict[str, Any]:
        return {
            "figure_path": str(_roots / "k_figure.svg"),
            "figure_caption": "Fig. 1",
            "labels_csv_path": args["labels_csv_path"],
        }

    registry = ToolRegistry()
    registry.register(
        RegisteredTool(
            ToolSpec(
                "olmoearth_plan_label_sample",
                "A plan.",
                {"type": "object", "properties": {}},
            ),
            handler,
        )
    )
    app = _app(registry)
    srv = server.build_server(app)
    asked = {"labels_csv_path": str(path)}
    blank = await _call(srv, "olmoearth_plan_label_sample", asked)
    assert _view(blank)["figure"] == {"caption": "Fig. 1", "held": True}
    assert "<svg" not in blank.model_dump_json(by_alias=True)
    with_panels = await app.call_agent_tool(
        "olmoearth_plan_label_sample", asked, views=True
    )
    assert "<svg" not in with_panels.model_dump_json(by_alias=True)
    assert with_panels.content[0].text.endswith(server.PLAN_SEEN_NOTE)
    refused = await _call(srv, "olmoearth_app_plan_figure", asked)
    assert refused.isError is True and refused.structuredContent is not None
    assert refused.structuredContent["error"] == sheet.FIGURE_LATER
    assert "<svg" not in refused.model_dump_json(by_alias=True)
    gone = await _call(
        srv,
        "olmoearth_plan_label_sample",
        {"labels_csv_path": str(_roots / "gone_to_label.csv")},
    )
    assert _view(gone)["figure"] == {"caption": "Fig. 1", "held": True}
    # Every window labelled with the class seen (the panel's save): the figure.
    page = (await _call(srv, "olmoearth_app_sheet", asked)).structuredContent
    assert page is not None
    saved = await _call(
        srv,
        "olmoearth_app_save_labels",
        {
            **asked,
            "sheet_id": page["sheet_id"],
            "face": "seen",
            "rule": "seen",
            "answers": [{"handle": page["windows"][0]["handle"], "answer": "1"}],
        },
    )
    assert saved.isError is False and saved.structuredContent["n_labelled"] == 1
    labelled = await _call(srv, "olmoearth_plan_label_sample", asked)
    assert _view(labelled)["figure"] == {"svg": svg, "caption": "Fig. 1"}
    assert "<svg" not in _text(labelled)
    given = await _call(srv, "olmoearth_app_plan_figure", asked)
    assert given.isError is False and given.structuredContent == {"svg": svg}
    assert json.loads(_text(given)) == {"ok": True}  # the SVG is the panel's alone
    (_roots / "k_figure.svg").unlink()
    lost = await _call(srv, "olmoearth_app_plan_figure", asked)
    assert lost.isError is True and sheet.FIGURE_GONE in _text(lost)
    missing = await _call(srv, "olmoearth_app_plan_figure", {})
    assert missing.isError is True
    outside = await _call(
        srv, "olmoearth_app_plan_figure", {"labels_csv_path": "/etc/passwd"}
    )
    assert outside.isError is True and "<svg" not in _text(outside)
    await app.aclose()


@pytest.mark.asyncio
async def test_a_panels_save_under_an_old_rule_is_refused(_roots: Path) -> None:
    """The panel sends the rule it drew: a seen sheet's save goes through; once
    the sheet reads another way (its design file gone), it is refused and the
    file is left as it was."""
    path = _seen_sheet(_roots)
    srv = server.build_server(_app())
    page = (
        await _call(srv, "olmoearth_app_sheet", {"labels_csv_path": str(path)})
    ).structuredContent
    assert page is not None
    answers = [{"handle": page["windows"][0]["handle"], "answer": "1"}]
    args = {
        "labels_csv_path": str(path),
        "sheet_id": page["sheet_id"],
        "face": "seen",
        "rule": "seen",
        "answers": answers,
    }
    saved = await _call(srv, "olmoearth_app_save_labels", args)
    assert saved.isError is False and saved.structuredContent["cells_changed"] == 1
    assert path.read_text(encoding="utf-8").splitlines()[1].endswith(",0,1")
    (_roots / "k.json").unlink()
    before = path.read_bytes()
    refused = await _call(srv, "olmoearth_app_save_labels", args)
    assert refused.isError is True
    assert "changed since the panel opened it" in _text(refused)
    assert path.read_bytes() == before
    odd = await _call(srv, "olmoearth_app_save_labels", {**args, "rule": 3})
    assert odd.isError is True and path.read_bytes() == before
    for left_out in ("sheet_id", "face", "rule"):
        partial = {k: v for k, v in args.items() if k != left_out}
        missing = await _call(srv, "olmoearth_app_save_labels", partial)
        assert missing.isError is True and path.read_bytes() == before
    nothing = {k: v for k, v in args.items() if k != "answers"}
    empty = await _call(srv, "olmoearth_app_save_labels", nothing)
    assert empty.isError is True and "answers or clear_column" in _text(empty)


@pytest.mark.asyncio
async def test_a_save_from_a_panel_holding_an_older_version_is_stale(
    _roots: Path,
) -> None:
    """A panel's save names the sheet's version it read: at the version now
    it writes and answers the new one, which the counts then give; once the
    sheet was saved elsewhere since, the save is refused with ``stale`` (the
    panel reads the sheet again) and the file left as it was. A version that
    is not text, and a sheet drawn again, are refused but not as stale."""
    path = _seen_sheet(_roots)
    srv = server.build_server(_app())
    page = (
        await _call(srv, "olmoearth_app_sheet", {"labels_csv_path": str(path)})
    ).structuredContent
    assert page is not None and isinstance(page["version"], str)
    handle = page["windows"][0]["handle"]
    args = {
        "labels_csv_path": str(path),
        "sheet_id": page["sheet_id"],
        "face": "seen",
        "rule": "seen",
        "answers": [{"handle": handle, "answer": "1"}],
        "version": page["version"],
    }
    saved = await _call(srv, "olmoearth_app_save_labels", args)
    assert saved.isError is False and saved.structuredContent is not None
    newer = saved.structuredContent["version"]
    counts = await _call(
        srv, "olmoearth_app_sheet_progress", {"labels_csv_path": str(path)}
    )
    assert counts.structuredContent is not None
    assert newer != page["version"] and counts.structuredContent["version"] == newer
    before = path.read_bytes()
    again = {**args, "answers": [{"handle": handle, "answer": "0"}]}
    stale = await _call(srv, "olmoearth_app_save_labels", again)
    assert stale.isError is True and stale.structuredContent is not None
    assert stale.structuredContent["stale"] is True
    assert stale.structuredContent["error"] == sheet.STALE_SHEET
    assert path.read_bytes() == before
    for change in ({"version": 3}, {"sheet_id": "s" + "a" * 16}):
        refused = await _call(srv, "olmoearth_app_save_labels", {**again, **change})
        assert refused.isError is True, change
        assert "stale" not in (refused.structuredContent or {}), change
    assert path.read_bytes() == before
    current = await _call(srv, "olmoearth_app_save_labels", {**again, "version": newer})
    assert current.isError is False and path.read_bytes() != before


@pytest.mark.asyncio
async def test_save_area_writes_one_studio_area(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url=f"{BASE}/areas",
        method="POST",
        json={"records": [{"id": "a1", "name": "Delta", "project_id": "p1"}]},
    )
    app = _app()
    app.keys.sign_in(KEY, auth.Account("Ana", "Lab", 1), remember=False)
    srv = server.build_server(app)
    ring = [[10.0, 50.0], [10.1, 50.0], [10.1, 50.1], [10.0, 50.1], [10.0, 50.0]]
    result = await _call(
        srv,
        "olmoearth_app_save_area",
        {
            "name": "Delta",
            "project_id": "p1",
            "geom": {"type": "Polygon", "coordinates": [ring]},
        },
    )
    assert result.isError is False
    assert result.structuredContent["area"]["id"] == "a1"
    assert result.structuredContent["area"]["bbox"] == [10.0, 50.0, 10.1, 50.1]
    sent = json.loads(httpx_mock.get_requests()[0].content)
    assert sent == {
        "name": "Delta",
        "project_id": "p1",
        "geom": {"type": "Polygon", "coordinates": [ring]},
    }


def test_defaults_set_an_output_folder_and_a_spill_threshold() -> None:
    env: dict[str, str] = {}
    server._defaults(env)
    assert env["OLMOEARTH_OUTPUT_ROOT"].endswith("olmoearth-agent-outputs")
    assert env["OLMOEARTH_TOOL_RESULT_SPILL_BYTES"] == str(server.DEFAULT_SPILL_BYTES)
    kept = {
        "OLMOEARTH_OUTPUT_ROOT": "/data/out",
        "OLMOEARTH_TOOL_RESULT_SPILL_BYTES": "0",
    }
    server._defaults(kept)
    assert kept == {
        "OLMOEARTH_OUTPUT_ROOT": "/data/out",
        "OLMOEARTH_TOOL_RESULT_SPILL_BYTES": "0",
    }


# --------------------------------------------------------------------------- the Check card

R = "11111111-2222-4333-8444-555555555555"
P = "99999999-8888-4777-8666-555555555555"
ZONE_FILES = ("result_epsg32610_0_native.tif", "result_epsg32733_1_native.tif")
ZONE_MUST = [
    "Each UTM zone is its own map: one zone's estimate is not the result's.",
    "These are Studio's native-resolution files.",
]
ZONE_CLAIMS = [{"claim": "the result is certified", "why": "nothing certifies it"}]
#: A top-k result's classes, as olmoearth_scores_from_studio lists them.
PRESENT_ABSENT = [
    {"channel": 0, "value": 0, "label": "absent"},
    {"channel": 1, "value": 1, "label": "present"},
    {"channel": 2, "value": 2, "label": "cloud"},
]
CANNED = {
    "offer": [
        {"labels": 20, "width": 0.2, "sentence": "Twenty labels.", "rate_note": "r50"},
        {"labels": 40, "width": 0.1, "sentence": "Forty labels.", "rate_note": "r50"},
    ],
    "refused": None,
}

Plan = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


def _zone_logits(n: int, seed: int) -> list[list[float]]:
    """Three-class logits for ``n`` windows of a zone."""
    rng = random.Random(seed)
    return [[rng.gauss(0.0, 1.5) for _ in range(3)] for _ in range(n)]


def _studio_answer(root: Path, sizes: tuple[int, ...] = (300, 200)) -> dict[str, Any]:
    """olmoearth_scores_from_studio's envelope for a result of two UTM zones,
    each zone's scores file written under the root, as the tool writes it."""
    root = root.resolve()
    zones = []
    for i, (file, n) in enumerate(zip(ZONE_FILES, sizes)):
        path = root / f"zone{i}_scores.json"
        path.write_text(json.dumps({"scores": _zone_logits(n, 4 + i)}), "utf-8")
        zones.append(
            {
                "file": file,
                "scores_path": str(path),
                "patch": 4,
                "valid_pixels": 16 * n,
                "pooled": {
                    "n_valid": n,
                    "n_windows": n,
                    "raster_check": {"sha256": str(i) * 64},
                },
            }
        )
    shares = [{"file": f, "share": s} for f, s in zip(ZONE_FILES, (0.6, 0.4))]
    inner = {
        "available": True,
        "result_id": R,
        "prediction_id": P,
        "kind": "topk",
        "property_name": "landcover",
        # A map of presence and absence: the owner's labelling rule is set,
        # so a press needs no rule of the user's.
        "classes": copy.deepcopy(PRESENT_ABSENT),
        "date_window": {"start": "2024-04-01T00:00:00Z", "end": "2024-06-30T00:00:00Z"},
        "n_zones": len(zones),
        "zones": zones,
        "facts": [{"id": "studio_zones", "sentence": "Two.", "zones": shares}],
        "must_state": list(ZONE_MUST),
        "forbidden_claims": copy.deepcopy(ZONE_CLAIMS),
    }
    return {"ok": True, "result": inner}


def _fake_tools(
    app: server.AppServer,
    monkeypatch: pytest.MonkeyPatch,
    studio: dict[str, Any],
    plan: Plan | None = None,
) -> list[tuple[str, dict[str, Any]]]:
    """``app.run_agent_tool`` answering ``studio`` for the Studio files and
    ``plan`` (else the real plan tool, through the registry) for a plan;
    the calls made, in order."""
    real = app.run_agent_tool
    calls: list[tuple[str, dict[str, Any]]] = []

    async def run(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """The tool's envelope, as run_agent_tool gives it."""
        calls.append((name, dict(arguments)))
        if name == server.SCORES_TOOL:
            return copy.deepcopy(studio)
        if plan is not None:
            return await plan(dict(arguments))
        return await real(name, arguments)

    monkeypatch.setattr(app, "run_agent_tool", run)
    return calls


@pytest.fixture
def canned_offers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Counts of 20 and 40 offered for every zone, without the package."""

    def fake(n_valid: int, widths: Any = check.CHECK_WIDTHS) -> dict[str, Any]:
        """A fresh copy of :data:`CANNED`."""
        return copy.deepcopy(CANNED)

    monkeypatch.setattr(check, "offers", fake)


def _drawn_plan(root: Path, delay: float = 0.0) -> Plan:
    """A plan that draws (after ``delay`` seconds), with the windows and the
    figure a draw never keeps."""

    async def plan(args: dict[str, Any]) -> dict[str, Any]:
        """A drawn plan's envelope."""
        await asyncio.sleep(delay)
        return {
            "ok": True,
            "result": {
                "available": True,
                "labels_csv_path": str(root / f"b{args.get('budget')}_to_label.csv"),
                "design_path": str(root / "b.json"),
                "n_population": 300,
                "windows": [{"window_index": 3, "map_class": "water", "lon": 1.5}],
                "figure_path": str(root / "b_figure.svg"),
                "must_state": ["Stated with the plan."],
            },
        }

    return plan


def _answer(plan_answer: dict[str, Any]) -> Plan:
    """A plan that always gives ``plan_answer``."""

    async def plan(args: dict[str, Any]) -> dict[str, Any]:
        """``plan_answer``, copied."""
        return copy.deepcopy(plan_answer)

    return plan


def _press(*draws: tuple[str, int]) -> dict[str, Any]:
    """The press's arguments: ``(check_key, budget)`` per zone."""
    return {"draws": [{"check_key": k, "budget": b} for k, b in draws]}


def _model(result: types.CallToolResult) -> dict[str, Any]:
    """The model's copy of a Check card's result: its last text block (where
    panels show, the say-first lines come first)."""
    texts = [c.text for c in result.content if isinstance(c, types.TextContent)]
    body: dict[str, Any] = json.loads(texts[-1])
    return body


@pytest.mark.asyncio
async def test_the_check_card_offers_each_zone_and_the_model_gets_no_key(
    _roots: Path, monkeypatch: pytest.MonkeyPatch, canned_offers: None
) -> None:
    """Two zones, each with its counts and none picked; the view has no scores
    path or raster digest; the model reads the zones, the result's must_state
    and forbidden claims as the tool gave them, and the card's note, never a
    check's key or the counts' card."""
    app = _app()
    studio = _studio_answer(_roots)
    calls = _fake_tools(app, monkeypatch, studio)
    result = await app.check_map_tool({"result_id": f"  {R} "}, views=True)
    assert result.isError is False
    # r5-a K2: the card asks for a 0-to-1 score's stray values to be clipped.
    assert calls == [(server.SCORES_TOOL, {"result_id": R, "clip_out_of_range": True})]
    view = _view(result)
    assert (view["tool"], view["ok"], view["needs_sign_in"]) == (
        server.CHECK_TOOL,
        True,
        False,
    )
    assert view["must_state"] == ZONE_MUST and view["resolved"] is None
    # The page keeps a press within the server's limit.
    assert view["draws_max"] == server.CHECK_DRAWS_MAX == 16
    zones = view["zones"]
    assert [z["zone"] for z in zones] == ["UTM zone 10N", "UTM zone 33S"]
    assert [z["n_valid"] for z in zones] == [300, 200]
    assert all(z["state"] == "to draw" and "draw" not in z for z in zones)
    assert all([o["labels"] for o in z["offer"]] == [20, 40] for z in zones)
    shown = json.dumps(view)
    for left_out in ("scores_path", "raster", "0" * 64, str(_roots), "landcover"):
        assert left_out not in shown
    model = _model(result)
    keys = [z["check_key"] for z in zones]
    text = _text(result)
    assert "check_key" not in text and not any(k in text for k in keys)
    assert all("offer" not in z and "counts" not in z for z in model["zones"])
    assert "Twenty labels." not in text and "rate_note" not in text
    assert model["must_state"] == ZONE_MUST
    assert model["forbidden_claims"] == ZONE_CLAIMS
    assert (model["result_id"], model["prediction_id"], model["n_zones"]) == (R, P, 2)
    # The card tells the model, silently, which result it shows: its id only.
    assert view["result_id"] == R
    assert "resolved_ids" not in model
    # A map of presence and absence with no part drawn: the card shows the
    # owner's rule with the box for the map's own (option A), and says so.
    assert view["rule"]["replaceable"] is True
    assert model["note"] == server.CHECK_CARD_NOTE + " " + server.CHECK_OWNER_RULE_NOTE
    # The model waits for the zones the card draws; a zone it does not draw
    # is planned only at the user's ask, with the user's count.
    card = _words(server.CHECK_CARD_NOTE)
    assert "For a zone the card draws, do not pick a count for the user" in card
    assert "is planned only if the user asks, with a count the user gives" in card
    assert [z["scores_path"] for z in model["zones"]] == [
        z["scores_path"] for z in studio["result"]["zones"]
    ]
    assert [z["windows"] for z in model["zones"]] == [300, 200]
    assert all(z["state"] == "to draw" for z in model["zones"])
    await app.aclose()


@pytest.mark.asyncio
async def test_a_press_draws_once_through_the_plan_tool(
    _roots: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The press draws a random sample at seed 0 of the count pressed with the
    real plan tool; a second press with another count is a repeat (the same
    sheet, the design file unchanged, one plan call in provenance); the page
    gets no window, class or figure; the card then tells the model the zone
    is drawn."""
    pytest.importorskip("oe_inferencex.estimate")
    app = _app()
    _fake_tools(app, monkeypatch, _studio_answer(_roots))
    srv = server.build_server(app)
    card = _view(await app.check_map_tool({"result_id": R}, views=True))
    zone = card["zones"][0]
    counts = [o["labels"] for o in zone["offer"]]
    assert counts == sorted(set(counts)) and all(1 <= c <= 300 for c in counts)
    first = await _call(
        srv, "olmoearth_app_check_draw", _press((zone["check_key"], counts[0]))
    )
    assert first.isError is False and json.loads(_text(first)) == {"ok": True}
    assert first.structuredContent is not None
    (drawn,) = first.structuredContent["draws"]
    assert (drawn["outcome"], drawn["state"]) == ("first", "drawn"), drawn
    assert (drawn["design"], drawn["seed"], drawn["budget"]) == ("random", 0, counts[0])
    assert drawn["error"] is None and drawn["pressed_at"].endswith("Z")
    progress = sheet.sheet_progress(drawn["labels_csv_path"])
    assert progress["n_rows"] == counts[0]
    design_path = Path(progress["design_path"])
    design = json.loads(design_path.read_text(encoding="utf-8"))
    assert (design["design"], design["seed"], design["budget"]) == (
        "random",
        0,
        counts[0],
    )
    before = design_path.read_bytes()
    again = await _call(
        srv, "olmoearth_app_check_draw", _press((zone["check_key"], counts[-1]))
    )
    assert again.structuredContent is not None
    (repeat,) = again.structuredContent["draws"]
    assert (repeat["outcome"], repeat["state"], repeat["budget"]) == (
        "repeat",
        "drawn",
        counts[0],
    )
    assert repeat["labels_csv_path"] == drawn["labels_csv_path"]
    assert design_path.read_bytes() == before
    planned = [
        e for e in app.state.provenance.entries if e.api_call == server.PLAN_TOOL
    ]
    assert len(planned) == 1
    for answer in (drawn, repeat):
        assert set(answer) == {*check.VIEW_KEYS, "outcome"}
        # The plan's own sentences and the sheet's path aside: no window list,
        # class, place or figure.
        kept = {
            k: v
            for k, v in answer.items()
            if k not in ("labels_csv_path", "must_state")
        }
        shown = json.dumps(kept)
        for left_out in ("map_class", "figure", "windows", "window_index", "lon"):
            assert left_out not in shown
        assert all(isinstance(s, str) for s in answer["must_state"])
    log = _roots / "checks" / check.LOG_NAME
    rows = [json.loads(line) for line in log.read_text("utf-8").splitlines()]
    assert [(r["outcome"], r["budget"], r["asked"]) for r in rows] == [
        ("first", counts[0], counts[0]),
        ("repeat", counts[0], counts[-1]),
    ]
    later = await app.check_map_tool({"result_id": R}, views=True)
    (zone_now, other) = _view(later)["zones"]
    assert zone_now["state"] == "drawn" and other["state"] == "to draw"
    assert zone_now["draw"]["progress"] == f"0 of {counts[0]} labelled"
    note = _model(later)["note"]
    assert note.startswith(server.CHECK_CARD_NOTE)
    assert (
        f"UTM zone 10N: its sample was drawn at {drawn['pressed_at']} "
        f"({counts[0]} windows, design 'random', seed 0); 0 of {counts[0]} "
        "labelled. It is not drawn again"
    ) in note
    # The same settings open the same sheet; the plan's contract is the zone's.
    assert server.CHECK_SAME_SHEET_NOTE.format(budget=counts[0]) in note
    told = _model(later)["zones"][0]
    assert told["labels_csv_path"] == drawn["labels_csv_path"]
    assert told["must_state"] == drawn["must_state"]
    assert any(
        c.get("id") == "subset_labelling_sufficient" for c in told["forbidden_claims"]
    )
    # The plan's facts too: every one of the sheet's rows labelled.
    assert any(f.get("id") == "prefix_is_random_sample" for f in told["facts"])
    assert all(
        not {"forbidden_claims", "facts"} & set(z.get("draw") or {})
        for z in _view(later)["zones"]
    )
    await app.aclose()


@pytest.mark.asyncio
async def test_two_presses_at_once_draw_once(
    _roots: Path, monkeypatch: pytest.MonkeyPatch, canned_offers: None
) -> None:
    """Two presses of one zone at the same time: one plan call; the other press
    gets the stored draw, with the count pinned by the first."""
    app = _app()
    calls = _fake_tools(
        app, monkeypatch, _studio_answer(_roots), _drawn_plan(_roots, delay=0.05)
    )
    srv = server.build_server(app)
    zone = _view(await app.check_map_tool({"result_id": R}, views=True))["zones"][0]
    one, two = await asyncio.gather(
        _call(srv, "olmoearth_app_check_draw", _press((zone["check_key"], 20))),
        _call(srv, "olmoearth_app_check_draw", _press((zone["check_key"], 40))),
    )
    answers = [
        r.structuredContent["draws"][0] for r in (one, two) if r.structuredContent
    ]
    assert sorted(a["outcome"] for a in answers) == ["first", "repeat"]
    assert {a["budget"] for a in answers} == {20}
    assert len({a["labels_csv_path"] for a in answers}) == 1
    assert [name for name, _ in calls].count(server.PLAN_TOOL) == 1
    await app.aclose()


@pytest.mark.asyncio
async def test_a_press_is_refused_per_zone_for_a_count_not_offered_or_a_bad_key(
    _roots: Path, monkeypatch: pytest.MonkeyPatch, canned_offers: None
) -> None:
    """A count the card did not offer and a key that is not a check's are
    refused zone by zone, with no plan, each zone's line logged; a press that
    is not a list of draws, or of more than 16, is refused whole and logs
    nothing."""
    app = _app()
    calls = _fake_tools(app, monkeypatch, _studio_answer(_roots), _drawn_plan(_roots))
    srv = server.build_server(app)
    zones = _view(await app.check_map_tool({"result_id": R}, views=True))["zones"]
    key = zones[0]["check_key"]
    out = await _call(
        srv,
        "olmoearth_app_check_draw",
        _press((key, 30), ("../../x", 20), ("f" * 24, 20), (zones[1]["check_key"], 40)),
    )
    assert out.isError is False and out.structuredContent is not None
    refused, bad, unknown, drawn = out.structuredContent["draws"]
    assert (refused["outcome"], refused["error"]) == ("refused", check.NOT_OFFERED)
    assert refused["check_key"] == key and refused["state"] is None
    assert (bad["outcome"], bad["check_key"], bad["error"]) == (
        "refused",
        None,
        check.BAD_KEY,
    )
    assert (unknown["outcome"], unknown["error"]) == ("refused", check.UNKNOWN_CHECK)
    assert (drawn["outcome"], drawn["state"], drawn["budget"]) == ("first", "drawn", 40)
    assert [name for name, _ in calls].count(server.PLAN_TOOL) == 1
    assert not (_roots / "checks" / f"{key}.draw.json").exists()
    log = _roots / "checks" / check.LOG_NAME
    rows = [json.loads(line) for line in log.read_text("utf-8").splitlines()]
    assert [r["outcome"] for r in rows] == ["refused", "refused", "refused", "first"]
    for args in (
        {},
        {"draws": "lots"},
        {"draws": []},
        _press(*[(key, 20)] * (server.CHECK_DRAWS_MAX + 1)),
    ):
        whole = await _call(srv, "olmoearth_app_check_draw", args)
        assert whole.isError is True, args
    assert len(log.read_text("utf-8").splitlines()) == len(rows)
    await app.aclose()


@pytest.mark.asyncio
async def test_a_plan_with_no_sheet_or_a_final_error_is_not_drawn_again(
    _roots: Path, monkeypatch: pytest.MonkeyPatch, canned_offers: None
) -> None:
    """A plan answering available: false, and one refused with an error the
    same settings give again, are failed draws with no retry: a second press
    runs no plan and shows the error whole, without its exception's name
    (r5-a K4: the card's copy; the draw record keeps the name)."""
    for plan_answer, error in (
        (
            {"ok": True, "result": {"available": False, "reason": "Needs the extra."}},
            "Needs the extra.",
        ),
        (
            {"ok": False, "error": "ValueError: this sheet holds labels"},
            "this sheet holds labels",
        ),
    ):
        app = _app()
        calls = _fake_tools(
            app, monkeypatch, _studio_answer(_roots), _answer(plan_answer)
        )
        srv = server.build_server(app)
        zone = _view(await app.check_map_tool({"result_id": R}, views=True))["zones"][0]
        key = zone["check_key"]
        first = await _call(srv, "olmoearth_app_check_draw", _press((key, 20)))
        assert first.structuredContent is not None
        (failed,) = first.structuredContent["draws"]
        assert (failed["outcome"], failed["state"], failed["retry"]) == (
            "first",
            "failed",
            False,
        )
        assert failed["error"] == error
        second = await _call(srv, "olmoearth_app_check_draw", _press((key, 40)))
        assert second.structuredContent is not None
        (kept,) = second.structuredContent["draws"]
        assert (kept["outcome"], kept["error"], kept["budget"]) == ("failed", error, 20)
        assert [name for name, _ in calls].count(server.PLAN_TOOL) == 1
        card = _view(await app.check_map_tool({"result_id": R}, views=True))
        assert card["zones"][0]["state"] == "failed"
        assert card["zones"][0]["retry"] is False
        (_roots / "checks" / f"{key}.draw.json").unlink()
        await app.aclose()


@pytest.mark.asyncio
async def test_a_failure_that_may_pass_is_retried_with_the_pinned_count(
    _roots: Path, monkeypatch: pytest.MonkeyPatch, canned_offers: None
) -> None:
    """A plan that failed with an error a second press may not meet: the next
    press runs the plan again with the request pinned at the first press,
    whatever count it sends."""
    app = _app()
    answers = [{"ok": False, "error": "MemoryError: out of memory"}]

    async def plan(args: dict[str, Any]) -> dict[str, Any]:
        """A failure first, then a drawn plan."""
        if answers:
            return answers.pop()
        return await _drawn_plan(_roots)(args)

    calls = _fake_tools(app, monkeypatch, _studio_answer(_roots), plan)
    srv = server.build_server(app)
    zone = _view(await app.check_map_tool({"result_id": R}, views=True))["zones"][0]
    first = await _call(
        srv, "olmoearth_app_check_draw", _press((zone["check_key"], 20))
    )
    assert first.structuredContent is not None
    (failed,) = first.structuredContent["draws"]
    assert (failed["state"], failed["retry"]) == ("failed", True)
    again = await _call(
        srv, "olmoearth_app_check_draw", _press((zone["check_key"], 40))
    )
    assert again.structuredContent is not None
    (retried,) = again.structuredContent["draws"]
    assert (retried["outcome"], retried["state"], retried["budget"]) == (
        "retry",
        "drawn",
        20,
    )
    plans = [args for name, args in calls if name == server.PLAN_TOOL]
    assert plans == [plans[0], plans[0]]
    assert plans[0] == {
        "scores_path": str((_roots.resolve() / "zone0_scores.json")),
        "design": "random",
        "budget": 20,
        "seed": 0,
    }
    await app.aclose()


@pytest.mark.asyncio
async def test_a_press_cancelled_mid_plan_is_recorded_stopped_and_logged(
    _roots: Path, monkeypatch: pytest.MonkeyPatch, canned_offers: None
) -> None:
    """A press cancelled while its plan runs (the page's call timed out or was
    dropped): the cancel goes on, the zone's draw is recorded stopped, which
    the next press retries with the count pinned, the press is logged
    'cancelled' at its press time, and the press's lock is free again."""
    monkeypatch.setattr(check, "_now", lambda: 1_700_000_000.0)
    pressed = "2023-11-14T22:13:20Z"

    async def cancelled(args: dict[str, Any]) -> dict[str, Any]:
        """The plan's call cancelled under it."""
        raise asyncio.CancelledError

    app = _app()
    studio = _studio_answer(_roots)
    _fake_tools(app, monkeypatch, studio, cancelled)
    zone = _view(await app.check_map_tool({"result_id": R}, views=True))["zones"][0]
    key = zone["check_key"]
    with pytest.raises(asyncio.CancelledError):
        await app.app_tool("olmoearth_app_check_draw", _press((key, 20)))
    assert not app._check_lock.locked()
    record = json.loads((_roots / "checks" / f"{key}.draw.json").read_text("utf-8"))
    assert (record["state"], record["error"], record["retry"]) == (
        "failed",
        check.STOPPED,
        True,
    )
    assert (record["budget"], record["pressed_at"]) == (20, pressed)
    log = _roots / "checks" / check.LOG_NAME
    rows = [json.loads(line) for line in log.read_text("utf-8").splitlines()]
    assert [(r["outcome"], r["state"], r["budget"], r["asked"]) for r in rows] == [
        ("cancelled", "failed", 20, 20)
    ]
    assert rows[0]["pressed_at"] == pressed
    shown = _view(await app.check_map_tool({"result_id": R}, views=True))["zones"][0]
    assert (shown["state"], shown["retry"], shown["budget"]) == ("failed", True, 20)
    _fake_tools(app, monkeypatch, studio, _drawn_plan(_roots))
    again = await app.app_tool("olmoearth_app_check_draw", _press((key, 20)))
    (retried,) = again["draws"]
    assert (retried["outcome"], retried["state"], retried["budget"]) == (
        "retry",
        "drawn",
        20,
    )
    await app.aclose()


@pytest.mark.asyncio
async def test_a_press_is_logged_at_its_press_time(
    _roots: Path, monkeypatch: pytest.MonkeyPatch, canned_offers: None
) -> None:
    """The press's time is taken once, before its lock and its claims: both
    zones of a two-zone press, their draw records and their log lines carry
    it, though the first zone's plan takes five minutes; not the time a plan
    finished or the second zone was claimed."""
    clock = [1_700_000_000.0]
    monkeypatch.setattr(check, "_now", lambda: clock[0])
    drawn = _drawn_plan(_roots)

    async def slow(args: dict[str, Any]) -> dict[str, Any]:
        """A plan that takes five minutes."""
        clock[0] += 300
        return await drawn(args)

    app = _app()
    _fake_tools(app, monkeypatch, _studio_answer(_roots), slow)
    zones = _view(await app.check_map_tool({"result_id": R}, views=True))["zones"]
    keys = [z["check_key"] for z in zones]
    out = await app.app_tool(
        "olmoearth_app_check_draw", _press((keys[0], 20), (keys[1], 40))
    )
    pressed = "2023-11-14T22:13:20Z"
    assert [(a["state"], a["pressed_at"]) for a in out["draws"]] == [
        ("drawn", pressed),
        ("drawn", pressed),
    ]
    assert clock[0] == 1_700_000_600.0
    for key in keys:
        path = _roots / "checks" / f"{key}.draw.json"
        assert json.loads(path.read_text("utf-8"))["pressed_at"] == pressed
    log = _roots / "checks" / check.LOG_NAME
    rows = [json.loads(line) for line in log.read_text("utf-8").splitlines()]
    assert [(r["check_key"], r["outcome"], r["pressed_at"]) for r in rows] == [
        (keys[0], "first", pressed),
        (keys[1], "first", pressed),
    ]
    await app.aclose()


@pytest.mark.asyncio
async def test_a_class_map_is_pointed_to_the_plan_tool(
    _roots: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A class map without top-k scores is refused: the model gets the error,
    hint and contract and a note with the two plain questions of the reader
    test's E/said-refused-2, word for word (how many points; the whole map
    alone, or each class too), which it maps itself to design 'random' or
    'class' for olmoearth_plan_label_sample, never saying 'design',
    'budget', 'window' or a tool name to the user (C6); the card gets the
    refusal's words for its fold and the points request its button sends,
    no press. The refusal is known by its contract and fixed end: another
    refusal whose quoted property name holds the words gets the error and
    hint, and no class-map note."""
    claims = [rules.zone_for_a_class_map()]
    refusal = {
        "ok": False,
        "error": "ValueError: the result's 'landcover' is a class map without "
        "top-k scores: its files hold classes only, with no score to read. "
        + rules.CLASS_MAP_REFUSAL,
        "hint": "The tool ran but failed.",
        "forbidden_claims": copy.deepcopy(claims),
    }
    app = _app()
    _fake_tools(app, monkeypatch, refusal)
    result = await app.check_map_tool({"result_id": R}, views=True)
    assert result.isError is True
    model = _model(result)
    assert model["note"] == server.CHECK_CLASS_MAP_NOTE.format(result_id=R)
    note = model["note"]
    for question in (
        "(1) 'How many points can you check? More points give a narrower range; "
        "the count is fixed before you start.'",
        "(2) 'Do you want only the figure for the whole map, or also a figure for "
        "each class the map shows? For the whole map alone the points are picked "
        "anywhere, so they mostly land on the common classes; for each class too, "
        "some are picked from every class the map shows.'",
    ):
        assert question in note
    assert "do not say 'design', 'budget', 'window' or a tool name" in note
    assert f"olmoearth_plan_label_sample with result_id {R}," in note
    assert "design 'random' for the whole map alone or 'class' for each" in note
    assert "Pick neither for them" in note
    assert (model["error"], model["hint"]) == (refusal["error"], refusal["hint"])
    assert model["forbidden_claims"] == claims
    view = _view(result)
    assert view["reason"].startswith("The result's 'landcover' is a class map")
    assert view["points_request"] == {"result_id": R, "prediction_id": None}
    assert "error" not in view and "zones" not in view
    assert "map" not in view  # signed out: Studio read nothing (F12)
    # A result id the resolver read from a prediction's: the result's own.
    resolved = {
        **refusal,
        "result": {"resolved_ids": [{"prediction_id": P, "result_id": "r-7"}]},
    }
    assert server._points_request(resolved, R) == {
        "result_id": "r-7",
        "prediction_id": P,
    }
    echo = {
        "ok": False,
        "error": "ValueError: the result declares 2 class maps with top-k scores "
        "('x is a class map without top-k scores', 'b'); name one",
        "hint": "The tool ran but failed.",
    }
    for other in (echo, {**refusal, "forbidden_claims": []}):
        _fake_tools(app, monkeypatch, other)
        out = await app.check_map_tool({"result_id": R}, views=True)
        assert out.isError is True and "note" not in _model(out)
        # r5-a K4: the card's error has no exception name and no model hint;
        # the model reads both as the tool gave them.
        assert (_view(out)["error"], _view(out)["hint"]) == (
            str(other["error"]).removeprefix("ValueError: "),
            None,
        )
        assert (_model(out)["error"], _model(out)["hint"]) == (
            other["error"],
            other["hint"],
        )
        assert "reason" not in _view(out) and "points_request" not in _view(out)
        assert "map" not in _view(out)
    await app.aclose()


#: A download token as Studio's result record carries it (a test value).
DOWNLOAD = "dl-test-value-7f3e"


class _RefusalStudio:
    """A Studio client that knows one result's record (its prediction, its
    period and a download token) and its prediction's name; or, ``down``,
    one that answers nothing. The ids asked for, in order."""

    def __init__(self, *, down: bool = False) -> None:
        self.down = down
        self.results: list[str] = []
        self.predictions: list[str] = []

    async def get_prediction_result(self, result_id: str) -> dict[str, Any]:
        """The result's record, as ``GET /prediction-results/{id}`` gives it."""
        self.results.append(result_id)
        if self.down:
            raise RuntimeError("Studio did not answer")
        return {
            "id": result_id,
            "prediction_id": P,
            "result_metadata": {
                "start_datetime": "2024-04-01T00:00:00Z",
                "end_datetime": "2024-06-30T23:59:59Z",
            },
            "download_token": DOWNLOAD,
        }

    async def get_prediction(self, prediction_id: str) -> dict[str, Any]:
        """The prediction's record, with its name."""
        self.predictions.append(prediction_id)
        if self.down:
            raise RuntimeError("Studio did not answer")
        return {"id": prediction_id, "name": "Land cover 2024"}


@pytest.mark.asyncio
async def test_a_class_map_refusal_names_the_map_and_its_period(
    _roots: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """F12: the class-map refusal card gets the map's name and period from
    the server: the result's record read once (its prediction and its
    period; the download token dropped), then the prediction's name, the ok
    path's lookup; neither the token nor the name reaches the model's text
    or the view beyond the map line. Studio not answering: no map line, the
    refusal otherwise as before. A prediction id the resolver read is the
    one named."""
    refusal = {
        "ok": False,
        "error": "ValueError: the result's 'landcover' is a class map without "
        "top-k scores: its files hold classes only, with no score to read. "
        + rules.CLASS_MAP_REFUSAL,
        "hint": "The tool ran but failed.",
        "forbidden_claims": [rules.zone_for_a_class_map()],
    }
    app = _app()
    _fake_tools(app, monkeypatch, refusal)
    up = _RefusalStudio()

    async def studio() -> _RefusalStudio:
        """The Studio client that answers."""
        return up

    monkeypatch.setattr(app, "studio", studio)
    result = await app.check_map_tool({"result_id": R}, views=True)
    assert result.isError is True
    view = _view(result)
    assert view["map"] == {
        "name": "Land cover 2024",
        "period_text": "1 Apr 2024 to 30 Jun 2024",
    }
    assert (up.results, up.predictions) == ([R], [P])
    assert view["points_request"] == {"result_id": R, "prediction_id": None}
    assert view["reason"].startswith("The result's 'landcover' is a class map")
    assert _model(result)["note"] == server.CHECK_CLASS_MAP_NOTE.format(result_id=R)
    told = _text(result)
    assert DOWNLOAD not in told and "Land cover 2024" not in told
    assert DOWNLOAD not in json.dumps(view)
    # Studio not answering: no map line, the refusal as before.
    down = _RefusalStudio(down=True)

    async def silent() -> _RefusalStudio:
        """The Studio client that answers nothing."""
        return down

    monkeypatch.setattr(app, "studio", silent)
    failed = await app.check_map_tool({"result_id": R}, views=True)
    shown = _view(failed)
    assert "map" not in shown
    assert shown["points_request"] == {"result_id": R, "prediction_id": None}
    assert shown["reason"] == view["reason"]
    assert "error" not in shown and "zones" not in shown
    assert down.results == [R] and down.predictions == []
    # The resolver's prediction id is the one named, not the record's.
    other = "77777777-6666-4555-8444-333333333333"
    resolved = {
        **refusal,
        "result": {"resolved_ids": [{"prediction_id": other, "result_id": R}]},
    }
    _fake_tools(app, monkeypatch, resolved)
    monkeypatch.setattr(app, "studio", studio)
    asked: list[Any] = []

    async def map_name(prediction_id: Any) -> str | None:
        """The ok path's lookup, stood in for: the ids it is asked for."""
        asked.append(prediction_id)
        return "Land cover 2024"

    monkeypatch.setattr(app, "_map_name", map_name)
    named = await app.check_map_tool({"result_id": P}, views=True)
    assert asked == [other]
    assert _view(named)["map"] == {
        "name": "Land cover 2024",
        "period_text": "1 Apr 2024 to 30 Jun 2024",
    }
    assert _view(named)["points_request"] == {"result_id": R, "prediction_id": other}
    await app.aclose()


@pytest.mark.asyncio
async def test_the_model_reads_each_drawn_plans_contract_and_the_zones_not_drawn(
    _roots: Path, monkeypatch: pytest.MonkeyPatch, canned_offers: None
) -> None:
    """A drawn zone's entry carries its plan's facts, must_state and forbidden
    claims as the plan gave them (the view data gets no facts or claims), and
    its note says the same settings open the same sheet; a zone whose draw
    failed for good says so in its entry and a note; once Studio's files
    change under the drawn zone, its entry and a note say so, and the
    same-sheet note goes."""
    claims = [{"id": "subset_labelling_sufficient", "why": "Label every window."}]
    facts = [{"id": "prefix_is_random_sample", "sentence": "Label every row.", "n": 20}]

    async def plan(args: dict[str, Any]) -> dict[str, Any]:
        """Zone 33S refused for good; zone 10N drawn, with the plan's contract."""
        if str(args.get("scores_path")).endswith("zone1_scores.json"):
            return {"ok": False, "error": "ValueError: this sheet holds labels"}
        drawn = await _drawn_plan(_roots)(args)
        drawn["result"]["forbidden_claims"] = copy.deepcopy(claims)
        drawn["result"]["facts"] = copy.deepcopy(facts)
        return drawn

    app = _app()
    studio = _studio_answer(_roots)
    _fake_tools(app, monkeypatch, studio, plan)
    zones = _view(await app.check_map_tool({"result_id": R}, views=True))["zones"]
    pressed = await app.app_tool(
        "olmoearth_app_check_draw",
        _press((zones[0]["check_key"], 20), (zones[1]["check_key"], 40)),
    )
    drawn, failed = pressed["draws"]
    assert (drawn["state"], failed["state"], failed["retry"]) == (
        "drawn",
        "failed",
        False,
    )
    later = await app.check_map_tool({"result_id": R}, views=True)
    model = _model(later)
    one, two = model["zones"]
    assert (one["state"], one["must_state"]) == ("drawn", ["Stated with the plan."])
    assert (one["forbidden_claims"], one["facts"]) == (claims, facts)
    assert "raster_changed" not in one and "failed_for_good" not in one
    assert "facts" not in two
    assert (two["state"], two["failed_for_good"]) == ("failed", True)
    assert two["error"] == "ValueError: this sheet holds labels"
    note = model["note"]
    assert note.startswith(server.CHECK_CARD_NOTE)
    assert server.CHECK_SAME_SHEET_NOTE.format(budget=20) in note
    assert server.CHECK_FINAL_NOTE.format(zone="UTM zone 33S") in note
    assert "UTM zone 10N: Studio's files" not in note
    assert "its entry carries its plan's facts, must_state and" in note
    assert all(
        not {"forbidden_claims", "facts"} & set(z.get("draw") or {})
        for z in _view(later)["zones"]
    )
    changed = copy.deepcopy(studio)
    changed["result"]["zones"][0]["pooled"]["raster_check"]["sha256"] = "f" * 64
    _fake_tools(app, monkeypatch, changed, plan)
    moved = _model(await app.check_map_tool({"result_id": R}, views=True))
    assert moved["zones"][0]["raster_changed"] is True
    assert moved["zones"][0]["state"] == "drawn"
    assert server.CHECK_CHANGED_NOTE.format(zone="UTM zone 10N") in moved["note"]
    assert server.CHECK_SAME_SHEET_NOTE.format(budget=20) not in moved["note"]
    await app.aclose()


@pytest.mark.asyncio
async def test_without_panels_the_model_asks_for_a_count_per_zone(
    _roots: Path, monkeypatch: pytest.MonkeyPatch, canned_offers: None
) -> None:
    """In a client without panels (the server's own routing), the model reads
    first the say-first lines worded for the chat, then the note, which says
    to ask for a count of squares per part (each count's sentence only on
    request), and each zone carries its counts, each count's sentence and
    the note they point to; still no check key."""
    app = _app()
    _fake_tools(app, monkeypatch, _studio_answer(_roots))
    srv = server.build_server(app)
    result = await _call(srv, server.CHECK_TOOL, {"result_id": R})
    assert result.isError is False
    model = _model(result)
    # A map of presence and absence: the owner's rule, with the note that asks
    # whether the map is of kelp (no box without panels; pass-3 finding Q5).
    assert model["note"] == (
        server.CHECK_TEXT_NOTE + " " + server.CHECK_TEXT_OWNER_RULE_NOTE
    )
    assert model["labelling_rule"]["scope"] == server.imagery.RULE_SCOPE
    note = _words(model["note"])
    assert "how many squares they can check in each part" in note
    assert "only when the user asks" in note
    assert "olmoearth_plan_label_sample with that zone's scores_path" in note
    first = json.loads(result.content[0].text)
    assert first["note"] == server.SAY_FIRST_NOTE
    assert first["say_first"][0] == server.SAY_CHECK_CHAT
    for zone in model["zones"]:
        assert zone["counts"] == [20, 40]
        assert zone["sentences"] == ["Twenty labels.", "Forty labels."]
        assert zone["rate_notes"] == ["r50"]
    assert "check_key" not in _text(result)
    await app.aclose()


@pytest.mark.asyncio
async def test_a_prediction_with_one_result_names_the_result_used(
    _roots: Path, monkeypatch: pytest.MonkeyPatch, canned_offers: None
) -> None:
    """The resolver's resolved_ids pass to the model as they are, and the card
    names the result used."""
    studio = _studio_answer(_roots)
    studio["result"]["resolved_ids"] = [
        {"argument": "result_id", "prediction_id": P, "result_id": R}
    ]
    app = _app()
    calls = _fake_tools(app, monkeypatch, studio)
    result = await app.check_map_tool({"result_id": P}, views=True)
    assert calls == [(server.SCORES_TOOL, {"result_id": P, "clip_out_of_range": True})]
    assert _model(result)["resolved_ids"] == studio["result"]["resolved_ids"]
    assert _view(result)["resolved"] == (
        f"Result used: {R}, the one result of prediction {P}"
    )
    await app.aclose()


@pytest.mark.asyncio
async def test_a_check_that_cannot_be_read_offers_no_press(
    _roots: Path, monkeypatch: pytest.MonkeyPatch, canned_offers: None
) -> None:
    """No result id: refused with no call. Signed out: the card offers the
    sign-in. The extra missing: its reason, no zone. Records that cannot be
    written: the error, no zone."""
    app = _app()
    calls = _fake_tools(app, monkeypatch, _studio_answer(_roots))
    blank = await app.check_map_tool({"result_id": "  "}, views=True)
    assert blank.isError is True and calls == []
    assert _view(blank)["error"] == "missing result_id"
    signed_out = {
        "ok": False,
        "error": "NotSignedIn: " + auth.NOT_SIGNED_IN_MARK,
        "needs_sign_in": True,
        "hint": auth.NOT_SIGNED_IN,
    }
    _fake_tools(app, monkeypatch, signed_out)
    out = await app.check_map_tool({"result_id": R}, views=True)
    assert out.isError is True and _model(out)["needs_sign_in"] is True
    assert _view(out)["needs_sign_in"] is True and "zones" not in _view(out)
    missing = {"ok": True, "result": {"available": False, "reason": "No extra."}}
    _fake_tools(app, monkeypatch, missing)
    unread = await app.check_map_tool({"result_id": R}, views=True)
    assert unread.isError is False
    assert _model(unread) == {
        "ok": True,
        "available": False,
        "result_id": R,
        "reason": "No extra.",
    }
    assert (_view(unread)["reason"], _view(unread)["zones"]) == ("No extra.", [])

    def unwritable() -> Path:
        """The checks folder refused, as when the output folder is read-only."""
        raise ValueError(check.NOT_WRITABLE.format(root=_roots))

    _fake_tools(app, monkeypatch, _studio_answer(_roots))
    monkeypatch.setattr(check, "checks_dir", unwritable)
    stuck = await app.check_map_tool({"result_id": R}, views=True)
    assert stuck.isError is True
    assert "cannot be written" in _view(stuck)["error"]
    assert "zones" not in _view(stuck)
    await app.aclose()


def test_the_check_notes_name_the_press_settings() -> None:
    """The notes' design and seed are the ones a press pins."""
    assert (check.DESIGN, check.SEED) == ("random", 0)
    for note in (server.CHECK_CARD_NOTE, server.CHECK_TEXT_NOTE):
        assert "design 'random'" in note and "seed 0" in note
    assert "olmoearth_check_map (the Check card)" in _words(server.MCP_RULES)
    description = server.OWN_TOOLS[server.CHECK_TOOL][1]
    assert "design 'random', seed 0, once per zone" in description


def test_the_context_rule_goes_with_every_note_of_a_panel_that_tells() -> None:
    """The panels that tell the model what they show (context updates) come
    with the rule that those are facts, not requests, and no grade before
    every window is labelled: the plan's note, the projects' note, the Check
    card's note (with its blind rule: no window's map class, no ranking while
    the sheets are labelled) and the descriptions of the sheet tool and the
    Check card's. Without panels the Check card's note keeps the no-grade
    half. MCP_RULES stays as it was (the instructions are cut short in some
    clients)."""
    rule = server.CONTEXT_RULE
    assert rule == server.CONTEXT_FACTS + " " + server.NO_GRADE_RULE
    assert "facts, not requests" in server.CONTEXT_FACTS
    assert server.NO_GRADE_RULE.startswith(
        "Never grade a sheet or give a rate before every window"
    )
    assert server.PLAN_SHEET_NOTE.endswith(" " + rule)
    # The panel's button as the panel names it (sheet.OPEN_FIRST).
    assert server.PLAN_SHEET_NOTE.startswith(
        "The panel beside this result has an 'Open the sheet here' button"
    )
    assert '("Open the sheet here")' in server.MCP_RULES
    assert "Label these windows" not in server.MCP_RULES
    assert server.PROJECTS_PANEL_NOTE.endswith(" " + rule)
    assert "by id" in server.PROJECTS_PANEL_NOTE
    assert "'this one'" in server.PROJECTS_PANEL_NOTE
    assert "which label sheets are in progress" in server.PROJECTS_PANEL_NOTE
    # The panel names each sheet in progress by its file name and path: the
    # note names no id the panel does not send.
    assert "each by its file name and labels_csv_path" in server.PROJECTS_PANEL_NOTE
    assert server.CHECK_CARD_NOTE.endswith(" " + rule)
    card = _words(server.CHECK_CARD_NOTE)
    assert "the card tells you silently which zones are drawn" in card
    assert "labels_csv_path" in card
    assert "Do not tell the user any window's map class" in card
    assert "do not rank or sort these zones' windows while their sheets" in card
    assert server.CHECK_TEXT_NOTE.endswith(" " + server.NO_GRADE_RULE)
    assert server.CONTEXT_FACTS not in server.CHECK_TEXT_NOTE
    description = server.OWN_TOOLS["olmoearth_label_sheet"][1]
    assert (
        "Its panel tells you silently how many windows are labelled; never grade "
        "before every window is." in description
    )
    checks = _words(server.OWN_TOOLS[server.CHECK_TOOL][1])
    assert (
        "Its card tells you silently how many windows of each zone's sheet are "
        "labelled; never grade before every window is." in checks
    )
    assert rule not in server.MCP_RULES


@pytest.mark.asyncio
async def test_the_projects_panel_note_comes_first_where_panels_show() -> None:
    """With panels, the projects' result starts with the note that the panel
    may tell the project open and the predictions picked, by id; without
    panels, or when the read failed, no such note."""
    failing = {"fail": False}

    async def handler(_args: dict[str, Any], _ctx: Any) -> dict[str, Any]:
        if failing["fail"]:
            raise RuntimeError("Studio is down")
        return {"projects": [{"id": "p1", "name": "Wetlands"}], "total": 1}

    registry = ToolRegistry()
    registry.register(
        RegisteredTool(
            ToolSpec(
                "olmoearth_load_context",
                "Projects.",
                {"type": "object", "properties": {}},
            ),
            handler,
        )
    )
    app = _app(registry)
    with_panels = await app.call_agent_tool("olmoearth_load_context", {}, views=True)
    assert with_panels.content[0].text == server.PROJECTS_PANEL_NOTE
    assert json.loads(with_panels.content[1].text)["ok"] is True
    without = await app.call_agent_tool("olmoearth_load_context", {}, views=False)
    assert server.PROJECTS_PANEL_NOTE not in _text(without)
    assert server.CONTEXT_RULE not in _text(without)
    failing["fail"] = True
    failed = await app.call_agent_tool("olmoearth_load_context", {}, views=True)
    assert failed.isError is True
    assert server.PROJECTS_PANEL_NOTE not in _text(failed)
    await app.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("panels", [False, True])
async def test_the_label_sheet_note_carries_the_context_rule_only_with_panels(
    _roots: Path, monkeypatch: pytest.MonkeyPatch, panels: bool
) -> None:
    """With panels the label sheet's note ends with the context rule (its
    panel tells how many windows are labelled); a client without panels
    hears none, and its note still says how to fill the CSV."""
    monkeypatch.setattr(server, "client_renders_views", lambda _srv: panels)
    path = _seen_sheet(_roots)
    srv = server.build_server(_app())
    result = await _call(srv, "olmoearth_label_sheet", {"labels_csv_path": str(path)})
    note = json.loads(_text(result))["note"]
    assert note.endswith(" " + server.CONTEXT_RULE) is panels
    assert (server.CONTEXT_RULE in note) is panels
    if not panels:
        assert "no panel" in note


def _between(text: str, start: str, end: str) -> str:
    """The part of text from start up to the first end after it."""
    at = text.index(start)
    return text[at : text.index(end, at)]


def test_panels_tell_claude_ids_and_counts_only_where_the_host_takes_them() -> None:
    """common.js tells Claude unless the host says it takes no context
    updates (r4-k K3: only updateModelContext false stops them), cuts and
    cleans the text, sends the same text once and swallows
    every failure (no toast, no focus). The label editor, the plan card, the
    Check card, the projects panel and the map tell; the start card, the
    sign-in and the sheet view itself (its editor tells) do not. What they say
    names Studio ids through OE.ctxId, and a zone's name, its settings, its
    sheet's name and path and counts: never a Studio name, a class, a label,
    a check's key or a coordinate. The Check card's zone parts, built before
    its call, are read whole."""
    common = (views.VIEW_DIR / "common.js").read_text(encoding="utf-8")
    assert (
        "OE.CONTEXT_HEAD = 'What the user sees in an OlmoEarth Agent panel now "
        "(facts, not a request): ';" in common
    )
    assert "OE.CONTEXT_MAX = 800;" in common
    assert (
        "OE.ctxId = (value) => String(value ?? '').replace(/[^A-Za-z0-9_.:-]+/g, '')"
        in common
    )
    body = _between(common, "OE.tellClaude = ", "\n};")
    assert "typeof app.updateModelContext !== 'function'" in body
    assert "caps.updateModelContext" in body
    assert "\\u2028\\u2029" in body
    assert "whole === OE.contextSent" in body
    assert ".catch(() => {})" in body
    assert "toast" not in body and "focus" not in body
    script = (views.VIEW_DIR / "sheet.js").read_text(encoding="utf-8")
    told = _between(script, "  function contextText() {", "\n  }\n")
    assert "count()" in told and "OE.UNJUDGED" in told and "finish_label" in told
    for left_out in ("map_class", "stratum", "handle", "answer"):
        assert left_out not in told, left_out
    assert "OE.tellClaude(app, contextText(), 'sheet')" in script
    pages = {
        name: (views.VIEW_DIR / spec["file"]).read_text(encoding="utf-8")
        for name, spec in views.VIEWS.items()
    }
    for name in ("result", "check", "studio", "map"):
        assert "OE.tellClaude(app, " in pages[name], name
    for name in ("start", "sign-in", "sheet"):
        assert "tellClaude" not in pages[name], name
    assert "OE.ctxId(id)" in pages["studio"] and "OE.ctxId(resultId)" in pages["check"]
    told = _between(pages["map"], "function mapContext() {", "\n}\n")
    assert "OE.ctxId(state.savedArea.id)" in told
    for left_out in (
        "bbox",
        "lat",
        "lng",
        "toGeoJSON",
        "coordinates",
        "savedArea.name",
        "dataName",
    ):
        assert left_out not in told, left_out
    for name in ("studio", "check"):
        told = _between(pages[name], "OE.tellClaude(app, ", ";\n")
        assert "dataName" not in told and ".name)" not in told, name
    # The Check card's whole update: tellCard from its start to the end of its
    # call, the zone parts it builds included, not the call alone.
    page = pages["check"]
    start = page.index("function tellCard(")
    told = page[start : page.index(";\n", page.index("OE.tellClaude(app, ", start))]
    assert "models" in told and "OE.ctxId(resultId)" in told
    for left_out in (
        "dataName",
        "prediction_name",
        "property_name",
        "project",
        "resolved",
        "map_class",
        "reference_class",
        "stratum",
        "handle",
        "answer",
        "bbox",
        "lat",
        "lng",
        "lon",
        "coordinates",
        "check_key",
        "m.key",
        "offer",
    ):
        assert re.search(rf"\b{re.escape(left_out)}\b", told) is None, left_out


def test_the_projects_panel_lists_the_sheets_in_progress_by_name_and_counts() -> None:
    """studio.html's "In progress": filled from the panel-only tool behind the
    page's drawing number, with no focus; each Resume sends the server's
    request whole; what the panel tells Claude is each sheet's whole file
    name, its labels_csv_path (each row carries it) and counts, never its job
    words or Resume request."""
    page = (views.VIEW_DIR / views.VIEWS["studio"]["file"]).read_text(encoding="utf-8")
    card = _between(page, "function inProgressCard(app, s) {", "\n}\n")
    assert "OE.call(app, 'olmoearth_app_sheets_in_progress')" in card
    assert "mine !== generation" in card and "focus" not in card
    assert "Older sheets in progress are not listed." in card
    row = _between(page, "function sheetRow(app, row) {", "\n}\n")
    assert "OE.sendJob(app, String(row.resume_message), slot)" in row
    assert "'aria-label': 'Resume: ' + row.name" in row and "primary" not in row
    told = _between(page, "function tellScreen(app, s) {", "\n}\n")
    assert "'label sheets in progress: '" in told
    # Each sheet by the path the model can pass back, as PROJECTS_PANEL_NOTE
    # says, and its name not cut at 64 characters: OE.ctxId keeps up to 255,
    # a file name's longest.
    assert "labels_csv_path" in told
    common = (views.VIEW_DIR / "common.js").read_text(encoding="utf-8")
    ctx_id = _between(common, "OE.ctxId = ", ";\n")
    assert ".slice(0, 255)" in ctx_id and ".slice(0, 64)" not in ctx_id
    for left_out in ("job_label", "resume_message", "r.progress"):
        assert left_out not in told, left_out
    assert "olmoearth_app_sheets_in_progress" in server.APP_TOOLS


#: The fixture's window centres: values that must never reach a view, the
#: page or the model.
REVIEW_CENTRES = [[12.345678 + 0.001 * i, 48.765432 - 0.001 * i] for i in range(8)]

#: A ranked review set's listed rows, as olmoearth_review_set lists them (no
#: coordinate); the second's class name is long and holds a control character.
REVIEW_ROWS = [
    {
        "rank": 1,
        "window_index": 5,
        "row": 1,
        "col": 1,
        "margin": 0.0123457,
        "predicted_class": 0,
        "class_name": "water",
    },
    {
        "rank": 2,
        "window_index": 2,
        "row": 0,
        "col": 2,
        "margin": 0.0456,
        "predicted_class": 1,
        "class_name": "trees\x07" + "x" * 80,
    },
]


def _review_scores(root: Path, *, placed: bool = True) -> Path:
    """A scores file of 8 windows on a 2 x 4 grid; placed, it records its map
    (raster path, window size, period, one centre per row) as
    olmoearth_scores_from_file writes it, else only rows, as a grid of single
    pixels does."""
    payload: dict[str, Any] = {
        "format": "olmoearth-agent/scores@1",
        "grid": [2, 4],
        "scores": [[0.5, 0.5]] * 8,
    }
    if placed:
        payload.update(
            patch_px=4,
            raster={"file": "m.tif", "path": str(root / "m.tif"), "shape": [8, 16]},
            date_window={"start": "2024-04-01", "end": "2024-06-30"},
            centres_lon_lat=REVIEW_CENTRES,
        )
    else:
        payload.update(result_id="r9", centres_lon_lat=REVIEW_CENTRES)
    path = root / ("scores_run_m_p4.json" if placed else "scores_r9_class_g10.json")
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _ranked(**extra: Any) -> dict[str, Any]:
    """A ranked review set's envelope."""
    inner = {
        "n_review": 2,
        "n_windows": 8,
        "review": copy.deepcopy(REVIEW_ROWS),
        "must_state": ["A review set is not a sample."],
        "evidence_covers_this_case": "yes",
        **extra,
    }
    return {"ok": True, "result": inner}


def _answering(
    app: server.AppServer, monkeypatch: pytest.MonkeyPatch, envelope: dict[str, Any]
) -> None:
    """``app.run_agent_tool`` answering ``envelope`` (a fresh copy each call)."""

    async def run(_name: str, _arguments: dict[str, Any]) -> dict[str, Any]:
        return copy.deepcopy(envelope)

    monkeypatch.setattr(app, "run_agent_tool", run)


def _keys(value: Any) -> set[str]:
    """Every dict key anywhere in ``value``."""
    if isinstance(value, dict):
        return set(value) | set().union(*(_keys(v) for v in value.values()))
    if isinstance(value, list):
        return set().union(*(_keys(v) for v in value)) if value else set()
    return set()


@pytest.mark.asyncio
async def test_a_review_set_view_has_window_cards_not_rows(
    _roots: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A review set ranked from a scores file that records its map: with
    panels, the view gets one card per listed window ({rank, caption,
    numbers}) under an opaque key, no centre, path or place; the model's text
    starts with the panel's note and holds no row. Without panels the rows
    stay and no card is made."""
    scores = _review_scores(_roots)
    app = _app()
    _answering(app, monkeypatch, _ranked())
    args = {"scores_path": str(scores)}
    shown = await app.call_agent_tool("olmoearth_review_set", args, views=True)
    view = _view(shown)
    block = view["review"]
    assert re.fullmatch(r"[0-9a-f]{24}", block["key"])
    assert block["where"] == "map" and block["pictures"] is True
    assert block["why_none"] is None
    assert (block["n_review"], block["n_listed"]) == (2, 2)
    assert block["unit"] == "square"  # a scores file's windows are patches
    first, second = block["windows"]
    assert set(first) == {"rank", "caption", "numbers"}
    assert first == {
        "rank": 1,
        "caption": "Map says: water",
        "numbers": {"window_index": 5, "row": 1, "col": 1, "margin": 0.0123457},
    }
    # A class name is someone's text: printable characters only, at most 64,
    # and never bare (what the map says, to judge against the picture).
    assert second["caption"] == "Map says: " + ("trees" + "x" * 80)[:64]
    dumped = json.dumps(view)
    for centre in REVIEW_CENTRES:
        assert f"{centre[0]:.6f}"[:8] not in dumped
        assert f"{centre[1]:.6f}"[:8] not in dumped
    assert str(scores) not in dumped and str(_roots) not in dumped
    assert not {"lon", "lat", "centres_lon_lat", "scores_path", "raster"} & _keys(view)
    # The model reads the panel's note first and no listed row.
    assert shown.content[0].text == server.REVIEW_PANEL_NOTE
    model = json.loads(shown.content[1].text)
    assert "review" not in model["result"]
    assert model["result"]["must_state"] == ["A review set is not a sample."]
    assert block["key"] not in _text(shown)
    # The record, written whole (folder 0700, file 0600): the file's identity.
    record_path = _roots / "review_sets" / f"{block['key']}.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    assert record["scores_path"] == str(scores.resolve())
    assert record["windows"] == [
        {"rank": 1, "window_index": 5},
        {"rank": 2, "window_index": 2},
    ]
    assert record["period"] is None and record["where"] == "map"
    assert record_path.stat().st_mode & 0o777 == 0o600
    assert record_path.parent.stat().st_mode & 0o777 == 0o700
    # Without panels: the rows stay, no card, no note, no record made.
    plain = await app.call_agent_tool("olmoearth_review_set", args, views=False)
    assert "review" not in _view(plain)
    assert server.REVIEW_PANEL_NOTE not in _text(plain)
    assert json.loads(plain.content[0].text)["result"]["review"] == REVIEW_ROWS
    await app.aclose()


@pytest.mark.asyncio
async def test_inline_scores_get_no_pictures(
    _roots: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A call that passes scores is ranked from them, whatever scores_path also
    says: no key, no pictures, the reason in their place, and the model keeps
    its rows. A file that records no map keeps them too, with its reason."""
    scores = _review_scores(_roots)
    app = _app()
    _answering(app, monkeypatch, _ranked())
    inline = {"scores": [[0.5, 0.5]] * 8, "scores_path": str(scores)}
    shown = await app.call_agent_tool("olmoearth_review_set", inline, views=True)
    block = _view(shown)["review"]
    assert block["key"] is None and block["pictures"] is False
    assert block["why_none"] == server.review.NO_SCORES_FILE
    assert [w["rank"] for w in block["windows"]] == [1, 2]
    assert server.REVIEW_PANEL_NOTE not in _text(shown)
    assert json.loads(shown.content[0].text)["result"]["review"] == REVIEW_ROWS
    assert not (_roots / "review_sets").exists()
    bare = _review_scores(_roots, placed=False)
    unplaced = await app.call_agent_tool(
        "olmoearth_review_set", {"scores_path": str(bare)}, views=True
    )
    block = _view(unplaced)["review"]
    assert block["key"] is None and block["why_none"] == server.review.NO_PLACE_RECORDED
    assert json.loads(unplaced.content[0].text)["result"]["review"] == REVIEW_ROWS
    await app.aclose()


@pytest.mark.asyncio
async def test_a_studio_review_set_says_it_is_unsupported_first(
    _roots: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A ranked review set of a Studio result (the grid of single pixels, run
    only at the user's ask) starts with the note that no recorded experiment
    supports its order and how little it read, before any other note, with
    panels or without; its file records no map, so its rows stay. A direct
    run's review set gets no such note."""
    bare = _review_scores(_roots, placed=False)
    sampling = {"grid": "10x10", "n_valid": 72, "n_windows_sampled": 100}
    for kind in ("binary_score", "top_k"):
        app = _app()
        _answering(
            app,
            monkeypatch,
            _ranked(
                ranked=True,
                score_kind=kind,
                sampling=sampling,
                evidence_covers_this_case="no",
                scores_path=str(bare),
            ),
        )
        for views_on in (False, True):
            told = await app.call_agent_tool(
                "olmoearth_review_set_from_result", {"result_id": "r9"}, views=views_on
            )
            note = told.content[0].text
            assert note == server.REVIEW_UNSUPPORTED_NOTE.format(
                n_valid=72, grid="10x10"
            )
            assert "72 single pixels on a 10x10 grid" in note
            assert server.REVIEW_PANEL_NOTE not in _text(told)
            model = json.loads(told.content[-1].text)["result"]
            assert model["review"] == REVIEW_ROWS
            if views_on:
                block = _view(told)["review"]
                # r4-k K7: a Studio result's points whose file records each
                # centre get a record (street map and CSV), never pictures.
                assert block["where"] == "points" and block["pictures"] is False
                assert block["why_none"] == server.review.STUDIO_POINTS
        await app.aclose()
    app = _app()
    _answering(app, monkeypatch, _ranked())
    direct = await app.call_agent_tool(
        "olmoearth_review_set",
        {"scores_path": str(_review_scores(_roots))},
        views=True,
    )
    assert "No recorded experiment supports" not in _text(direct)
    refused = {"ok": True, "result": {"ranked": False, "reason": "needs a threshold"}}
    _answering(app, monkeypatch, refused)
    told = await app.call_agent_tool(
        "olmoearth_review_set_from_result", {"result_id": "r9"}, views=True
    )
    assert "No recorded experiment supports" not in _text(told)
    assert "review" not in _view(told)
    await app.aclose()


@pytest.mark.asyncio
async def test_the_review_image_tool_tells_the_model_nothing(
    _roots: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The panel asks for a window's pictures by the review's key and its rank,
    app-only; the model reads {"ok": true} whatever happens. A key that names
    no review, a rank not listed, or a scores file changed since it was
    ranked is refused with its sentence, before anything is fetched; while
    pictures are off the answer is the question, with the review set's own
    test-scene question."""
    monkeypatch.setattr(server, "client_renders_views", lambda _srv: True)
    scores = _review_scores(_roots)
    app = _app()
    _answering(app, monkeypatch, _ranked())
    shown = await app.call_agent_tool(
        "olmoearth_review_set", {"scores_path": str(scores)}, views=True
    )
    key = _view(shown)["review"]["key"]
    srv = server.build_server(app)
    tool = {t.name: t for t in await _list(srv)}["olmoearth_app_review_image"]
    assert tool.meta["ui"]["visibility"] == ["app"]
    assert tool.inputSchema["required"] == ["review_key", "rank"]
    assert tool.inputSchema["properties"]["review_key"]["pattern"] == "^[0-9a-f]{24}$"
    assert tool.inputSchema["properties"]["rank"] == {"type": "integer", "minimum": 1}
    assert "olmoearth_app_review_image" in server.IMAGERY_TOOLS

    async def ask(arguments: dict[str, Any]) -> types.CallToolResult:
        result = await _call(srv, "olmoearth_app_review_image", arguments)
        assert [c.text for c in result.content] == ['{"ok": true}']
        return result

    unknown = await ask({"review_key": "0" * 24, "rank": 1})
    assert unknown.structuredContent == {
        "ok": False,
        "error": server.review.REVIEW_UNKNOWN,
    }
    malformed = await ask({"review_key": "../etc/passwd", "rank": 1})
    assert malformed.structuredContent["error"] == server.review.REVIEW_UNKNOWN
    for rank in (3, 0, "1", True):
        unlisted = await ask({"review_key": key, "rank": rank})
        assert unlisted.structuredContent["error"] == server.review.NOT_LISTED
    asked = await ask({"review_key": key, "rank": 1})
    assert asked.structuredContent == {
        "ok": True,
        "needs_consent": True,
        "ask": server.imagery.CONSENT_ASK,
        "test_ask": server.imagery.REVIEW_TEST_ASK,
    }
    # The scores file rewritten since (another ranking writes the same name).
    scores.write_text(scores.read_text(encoding="utf-8") + " ", encoding="utf-8")
    changed = await ask({"review_key": key, "rank": 1})
    assert changed.structuredContent["error"] == server.review.REVIEW_CHANGED
    await app.aclose()


@pytest.mark.asyncio
async def test_the_leads_three_lines(
    _roots: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every result view carries a lead: a review set's lines (B/R11 with its
    counts, which already says no labels were used, so no third line; the
    next step, without pictures when the panel shows none; the third line
    only when the first is left out); the estimate, the certified zone and
    the Check card draw their own (no line here). The model reads no
    say_first for a review set."""
    assert set(server.LEAD) >= {
        name for name, view in server.TOOL_VIEW.items() if view == "result"
    }
    for tool in ("olmoearth_estimate_map_error", "olmoearth_certify_zone"):
        assert server._lead(tool) == {"what": None, "next": None, "cannot": None}
    assert server._lead(server.CHECK_TOOL) == {
        "what": None,
        "next": None,
        "cannot": None,
    }
    app = _app()
    _answering(app, monkeypatch, _ranked())
    shown = await app.call_agent_tool(
        "olmoearth_review_set", {"scores_path": str(_review_scores(_roots))}, views=True
    )
    # Round r5-e, R2: shorter, and no longer "Where to look first: ..." under
    # the overline "Look first".
    assert _view(shown)["lead"] == {
        "what": "The 2 squares the model was least sure of, among the 8 read; the "
        "number 2 was chosen in advance, not a count of doubtful squares. No labels "
        "were used, so the map may be right or wrong there.",
        "next": server.REVIEW_LEAD_NEXT,
        "cannot": None,
    }
    assert "say_first" not in _text(shown)
    bare = await app.call_agent_tool(
        "olmoearth_review_set",
        {"scores_path": str(_review_scores(_roots, placed=False))},
        views=True,
    )
    assert _view(bare)["lead"]["next"] == server.REVIEW_LEAD_NEXT_NO_PICTURES
    # A boundary-first set's windows are not the least sure: no first line,
    # so the third says no labels were used.
    _answering(app, monkeypatch, _ranked(order="boundary_first"))
    edge = await app.call_agent_tool(
        "olmoearth_review_set", {"scores_path": str(_review_scores(_roots))}, views=True
    )
    assert _view(edge)["lead"]["what"] is None
    assert _view(edge)["lead"]["cannot"] == server.REVIEW_LEAD_CANNOT
    common = (views.VIEW_DIR / "common.js").read_text(encoding="utf-8")
    lead = _between(common, "OE.lead = (lead) => {", "\n};")
    assert "if (!lines.length) return null;" in lead
    await app.aclose()


@pytest.mark.asyncio
async def test_a_studio_review_sets_lead_says_points_and_a_refusal_has_none(
    _roots: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A review set of a Studio result reads one pixel at each point of a
    grid: its first line says points, of how many read (its sampling's
    n_valid), never squares (round r5-e, R2: no longer on which grid; the
    fold's legend says each square stands for the one pixel read in it); one
    point is "point". A review set the tool refused (ranked false, with its
    reason) has no lead at all, so the view keeps the r4-b-2 layout: the
    reason on the face, no fold."""
    app = _app()
    sampled = _ranked(
        n_review=7, n_windows=96, sampling={"grid": "10x10", "n_valid": 72}
    )
    _answering(app, monkeypatch, sampled)
    shown = await app.call_agent_tool(
        "olmoearth_review_set_from_result", {"result_id": "r9"}, views=True
    )
    # No check_request here (the result names no result_id), so the next
    # line stays (r5-e R2 drops it only beside the check card).
    assert "check_request" not in _view(shown)["review"]
    assert _view(shown)["lead"] == {
        "what": "The 7 points the model was least sure of, among the 72 read; the "
        "number 7 was chosen in advance, not a count of doubtful points. No labels "
        "were used, so the map may be right or wrong there.",
        "next": server.REVIEW_LEAD_NEXT_NO_PICTURES_POINTS,
        "cannot": None,
    }
    # The next line and the block's unit say points too (r4-d-1 R8).
    assert server.REVIEW_LEAD_NEXT_NO_PICTURES_POINTS == (
        "To learn how wrong the map is, check a random pick of points."
    )
    assert _view(shown)["review"]["unit"] == "point"
    one = server._lead(
        "olmoearth_review_set_from_result",
        {"n_review": 1, "sampling": {"n_valid": 72}},
        {},
    )
    assert one["what"] == (
        "The 1 point the model was least sure of, among the 72 read; the number 1 "
        "was chosen in advance, not a count of doubtful points. No labels were "
        "used, so the map may be right or wrong there."
    )
    assert server._grid_text([16, 16]) == "16 x 16"
    assert server._grid_text("bad") is None
    refused = {
        "ok": True,
        "result": {
            "ranked": False,
            "reason": "This map's values are declared to run from 0.2 to 1.2; pass "
            "it as threshold.",
            "declared_range": [0.2, 1.2],
        },
    }
    _answering(app, monkeypatch, refused)
    again = await app.call_agent_tool(
        "olmoearth_review_set_from_result", {"result_id": "r9"}, views=True
    )
    told = _view(again)
    assert told["lead"] == {"what": None, "next": None, "cannot": None}
    assert told["reason"].startswith("This map's values are declared")
    plain = server._lead("olmoearth_review_set", {"reason": "needs a threshold"}, {})
    assert plain == dict.fromkeys(server.LEAD_LINES)
    await app.aclose()


def test_the_projects_card_offers_check_as_its_main_job_and_look_first_gated() -> None:
    """studio.html: Check is the row's first job, named as on the start card,
    and not offered for embeddings or detection; "Where to look first" only
    while the server offers it (first_in_studio) and, on a result, only for
    the tool's top-k base (names[0]); a prediction whose results hold no top-k
    scores says so neutrally and sends nothing."""
    page = (views.VIEW_DIR / views.VIEWS["studio"]["file"]).read_text(encoding="utf-8")
    common = (views.VIEW_DIR / "common.js").read_text(encoding="utf-8")
    assert "check: 'Check how wrong it is'" in page
    assert "Check accuracy" not in page
    assert "firstInStudio = data.first_in_studio === true;" in page
    assert "OE.NO_CHECK_KINDS = ['embeddings', 'detection'];" in common
    assert "OE.FIRST_KINDS = ['classes'];" in common
    top_k = _between(common, "OE.hasTopK = (names) => {", "\n};")
    assert "const first = list[0];" in top_k and "_rank1_prob" in top_k
    no_first = _between(common, "OE.NO_FIRST = ", ";\n")
    assert "classes only" not in no_first and "square" not in no_first
    assert "OE.toast(OE.NO_FIRST)" in page
