# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Tests for the foundational Studio tools (context paging, request-AOI)."""

from __future__ import annotations

import json

import pytest
from pytest_httpx import HTTPXMock

from olmoearth_agent.harness.state import ThreadState
from olmoearth_agent.llm.types import ToolCall
from olmoearth_agent.studio.client import StudioClient, StudioConfig
from olmoearth_agent.tools.registry import ToolContext, ToolRegistry
from olmoearth_agent.tools.studio import build_studio_tools

BASE = "http://mock-studio/api/v1"


def test_request_aoi_is_registered() -> None:
    names = [t.spec.name for t in build_studio_tools()]
    assert "olmoearth_request_aoi" in names


@pytest.mark.asyncio
async def test_request_aoi_returns_needs_aoi_directive() -> None:
    registry = ToolRegistry()
    registry.register_all(build_studio_tools())
    result = await registry.dispatch(
        ToolCall(
            id="c1",
            name="olmoearth_request_aoi",
            arguments={"purpose": "change detection", "suggested_name": "My AOI"},
        ),
        # request_aoi makes no Studio call, so a None client is fine.
        ctx=ToolContext(studio=None, state=None),  # type: ignore[arg-type]
    )
    assert result["ok"] is True
    inner = result["result"]
    assert inner["needs_aoi"] is True
    assert inner["purpose"] == "change detection"
    assert inner["suggested_name"] == "My AOI"


@pytest.mark.asyncio
async def test_load_context_pages_projects_with_total(httpx_mock: HTTPXMock) -> None:
    """The former olmoearth_search_projects is load_context's paging."""
    httpx_mock.add_response(
        url=f"{BASE}/users/me",
        json={"records": [{"id": "u1", "name": "Ziming", "organizations": []}]},
    )
    httpx_mock.add_response(
        url=f"{BASE}/projects/search",
        method="POST",
        json={
            "records": [
                {"id": "p3", "name": "Karst", "creation_time": "2026-09-01T00:00:00Z"}
            ],
            "meta": {"total": 5},
        },
    )
    registry = ToolRegistry()
    registry.register_all(build_studio_tools())
    assert "olmoearth_search_projects" not in registry.names()
    state = ThreadState()
    async with StudioClient(StudioConfig(api_key="k", base_url=BASE)) as studio:
        result = await registry.dispatch(
            ToolCall(
                id="c1",
                name="olmoearth_load_context",
                arguments={"limit": 1, "offset": 2},
            ),
            ctx=ToolContext(studio=studio, state=state),
        )
    assert result["ok"] is True
    out = result["result"]
    assert out["user_name"] == "Ziming"
    assert out["projects"] == [
        {"id": "p3", "name": "Karst", "creation_time": "2026-09-01T00:00:00Z"}
    ]
    assert (out["project_count"], out["total"], out["next_offset"]) == (1, 5, 3)
    assert state.studio_context is not None
    assert state.studio_context.projects[0].name == "Karst"
    (search,) = [r for r in httpx_mock.get_requests() if r.method == "POST"]
    assert json.loads(search.content) == {"limit": 1, "offset": 2}
