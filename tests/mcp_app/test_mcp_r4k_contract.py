# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Round r4-k (16 Nov 2026), the contract between the three parts built
apart (views-core, server, result and check cards), read from the files (no
test here runs the scripts):

- every call a page makes to a new views' tool (the host report, the street
  map link, the CSV of places) sends only the arguments the server's schema
  lists, and every one it requires; each has its handler;
- the host report's view is the server's own name of the view: every page's
  OE.start name maps to the view whose file it is;
- the watchdog's words are common.js's NOT_STARTED;
- the names the pages read from the server's data (low_text, high_text,
  examples) are the names the server writes.
"""

from __future__ import annotations

import re

import pytest

pytest.importorskip("mcp")

from olmoearth_agent.mcp_app import server, views  # noqa: E402

#: The views' tools this round adds, which the pages call.
NEW_TOOLS = (
    "olmoearth_app_host_report",
    "olmoearth_app_review_link",
    "olmoearth_app_review_csv",
)

#: An OE.call of a named tool with an object literal: the tool and the
#: literal's opening brace.
_CALL = re.compile(r"OE\.call\(\s*\w+\s*,\s*'([A-Za-z0-9_]+)'\s*,\s*\{")


def _read(name: str) -> str:
    """A file of views/, as it is on disk."""
    return (views.VIEW_DIR / name).read_text(encoding="utf-8")


def _literal(text: str, start: int) -> str:
    """The object literal whose opening brace is at start, braces balanced
    (no brace in a string inside it, as in every call checked here)."""
    depth = 0
    for at in range(start, len(text)):
        if text[at] == "{":
            depth += 1
        elif text[at] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1 : at]
    raise AssertionError("an unclosed object literal")


def _top_keys(body: str) -> set[str]:
    """The top-level keys of an object literal's body (a shorthand key
    included): each name at depth 0 before a colon, a comma or the end."""
    keys: set[str] = set()
    depth = 0
    piece = ""
    for ch in body + ",":
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            if piece.strip():  # a trailing comma leaves an empty piece
                match = re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*(?::|$)", piece)
                assert match is not None, piece
                keys.add(match.group(1))
            piece = ""
        else:
            piece += ch
    return keys


def _calls(tool: str) -> list[set[str]]:
    """The argument names of every literal call of tool in views/."""
    found: list[set[str]] = []
    for page in sorted(views.VIEW_DIR.iterdir()):
        if page.suffix not in (".html", ".js"):
            continue
        text = page.read_text(encoding="utf-8")
        found.extend(
            _top_keys(_literal(text, match.end() - 1))
            for match in _CALL.finditer(text)
            if match.group(1) == tool
        )
    return found


@pytest.mark.parametrize("tool", NEW_TOOLS)
def test_each_new_views_tool_is_called_with_its_schemas_arguments(tool: str) -> None:
    """Each new views' tool is listed app-only (APP_TOOLS), has its handler,
    and every page's call of it sends only the schema's arguments and every
    required one."""
    assert tool in server.APP_TOOLS
    handler = "_app_" + tool.removeprefix("olmoearth_app_")
    assert callable(getattr(server.AppServer, handler, None)), handler
    schema = server.APP_TOOLS[tool][1]
    allowed = set(schema["properties"])
    required = set(schema.get("required", []))
    calls = _calls(tool)
    assert calls, tool
    for keys in calls:
        assert keys <= allowed, (tool, keys - allowed)
        assert required <= keys, (tool, required - keys)


def test_the_host_report_sends_the_contracts_seven_names() -> None:
    """The host report (r4-k K3) sends the seven names the contract gives,
    which are the schema's properties too."""
    want = {"view", "event", "caps", "modes", "context_keys", "container", "host"}
    assert _calls("olmoearth_app_host_report") == [want]
    assert set(server.APP_TOOLS["olmoearth_app_host_report"][1]["properties"]) == want


def test_the_csv_is_asked_both_ways_and_the_link_by_key_and_rank() -> None:
    """The CSV button asks with save false (the host's download) and with save
    true (the server writes it); the Street map press by key and rank."""
    page = _read("result.html")
    assert "{ review_key: linkKey, save: false }" in page
    assert "{ review_key: linkKey, save: true }" in page
    assert "{ review_key: linkKey, rank }" in page


def test_every_pages_start_name_maps_to_its_own_view() -> None:
    """OE.VIEW_NAMES maps each page's OE.start name to the server's name of
    the view whose file that page is, so the host report's view is the
    server's (views.VIEWS)."""
    common = _read("common.js")
    at = common.index("OE.VIEW_NAMES = {")
    block = common[at : common.index("};", at)]
    names = dict(re.findall(r"'([a-z-]+)': '([a-z-]+)'", block))
    assert set(names.values()) == set(views.VIEWS)
    for view, spec in views.VIEWS.items():
        page = _read(str(spec["file"]))
        started = re.findall(r"await OE\.start\('([a-z-]+)'", page)
        assert len(started) == 1, view
        assert names[started[0]] == view, (view, started)


def test_the_watchdog_says_common_js_not_started_words() -> None:
    """The watchdog every page runs before its module says the words OE.start
    puts in #root when it cannot connect (OE.NOT_STARTED)."""
    common = _read("common.js")
    said = re.search(r"OE\.NOT_STARTED = '([^']+)';", common)
    assert said is not None
    for spec in views.VIEWS.values():
        page = _read(str(spec["file"]))
        assert f"r.textContent='{said.group(1)}';" in page, spec["file"]


def test_the_pages_read_the_names_the_server_writes() -> None:
    """The result card reads low_text and high_text, which the result view
    writes; the label sheet and the Check card read examples, which the
    sheet's answer and the Check card's view write."""
    source = (views.VIEW_DIR.parent / "server.py").read_text(encoding="utf-8")
    assert '"low_text": rules.percent(low),' in source
    assert '"high_text": rules.percent(high),' in source
    assert 'data["examples"] = examples' in source
    assert 'view["examples"] = examples' in source
    result = _read("result.html")
    assert "lowText: text(r.low_text), highText: text(r.high_text)" in result
    assert "data.examples" in _read("sheet.js")
    assert "examplesOf(data.examples)" in _read("check.html")
