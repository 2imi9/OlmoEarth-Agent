# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Round r5-n S5: the app's own icon, declared inline in the server's serverInfo.

Claude's MCP Apps design guidelines ("Declare an app icon for Claude"): one inline
``data:image/png;base64,`` icon, 64 to 128 px, under 10 KB; Claude takes no URL and no SVG.
"""

from __future__ import annotations

import base64
import struct

import pytest

from olmoearth_agent.mcp_app import views

pytest.importorskip("mcp")


def test_the_icon_is_a_small_inline_png() -> None:
    """The icon file is a PNG of 64 to 128 px a side and under 10 KB, given as a data URL."""
    url = views.app_icon_data_url()
    assert url.startswith("data:image/png;base64,")
    raw = base64.b64decode(url.split(",", 1)[1])
    assert raw[:8] == b"\x89PNG\r\n\x1a\n"
    assert len(raw) < 10 * 1024
    width, height = struct.unpack(">II", raw[16:24])
    assert 64 <= width <= 128 and 64 <= height <= 128


def test_the_server_declares_the_icon_in_its_server_info() -> None:
    """The low-level server carries one icon, the inline PNG, so the handshake's serverInfo holds it."""
    from olmoearth_agent.mcp_app import server

    built = server.build_server()
    icons = built.icons
    assert icons and len(icons) == 1
    assert icons[0].src == views.app_icon_data_url()
    assert icons[0].mimeType == "image/png"
