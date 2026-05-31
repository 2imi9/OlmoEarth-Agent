# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Unit tests for the Studio client against a mocked HTTP endpoint."""

from __future__ import annotations

import pytest
from pytest_httpx import HTTPXMock

from olmoearth_agent.studio.client import StudioClient, StudioConfig

BASE = "http://mock-studio/api/v1"


@pytest.fixture
def config() -> StudioConfig:
    return StudioConfig(api_key="sk_test", base_url=BASE)


@pytest.mark.asyncio
async def test_users_me_unwraps_envelope(
    config: StudioConfig, httpx_mock: HTTPXMock
) -> None:
    httpx_mock.add_response(
        url=f"{BASE}/users/me",
        json={"records": [{"id": "u1", "name": "Test"}], "meta": None, "errors": None},
    )
    async with StudioClient(config) as studio:
        me = await studio.users_me()
    assert me["name"] == "Test"


@pytest.mark.asyncio
async def test_search_projects_reports_total(
    config: StudioConfig, httpx_mock: HTTPXMock
) -> None:
    httpx_mock.add_response(
        url=f"{BASE}/projects/search",
        method="POST",
        json={
            "records": [{"id": "p1", "name": "A"}],
            "meta": {"total": 1},
            "errors": None,
        },
    )
    async with StudioClient(config) as studio:
        env = await studio.search_projects(limit=5)
    assert env.total == 1
    assert env.one is not None
    assert env.one["name"] == "A"


@pytest.mark.asyncio
async def test_load_context_combines_user_and_projects(
    config: StudioConfig, httpx_mock: HTTPXMock
) -> None:
    httpx_mock.add_response(
        url=f"{BASE}/users/me",
        json={
            "records": [
                {
                    "id": "u1",
                    "name": "Ziming",
                    "organizations": [{"name": "bAI Labs"}],
                }
            ]
        },
    )
    httpx_mock.add_response(
        url=f"{BASE}/projects/search",
        method="POST",
        json={"records": [{"id": "p1", "name": "Karst"}], "meta": {"total": 1}},
    )
    async with StudioClient(config) as studio:
        ctx = await studio.load_context()
    assert ctx.user_name == "Ziming"
    assert ctx.organization == "bAI Labs"
    assert len(ctx.projects) == 1
    assert ctx.projects[0].name == "Karst"


@pytest.mark.asyncio
async def test_auth_header_sent(
    config: StudioConfig, httpx_mock: HTTPXMock
) -> None:
    httpx_mock.add_response(url=f"{BASE}/users/me", json={"records": [{}]})
    async with StudioClient(config) as studio:
        await studio.users_me()
    request = httpx_mock.get_requests()[0]
    assert request.headers["Authorization"] == "Bearer sk_test"


def test_from_env_rejects_placeholder(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLMOEARTH_API_KEY", "replace-me")
    with pytest.raises(RuntimeError, match="OLMOEARTH_API_KEY"):
        StudioConfig.from_env()


@pytest.mark.asyncio
async def test_get_project(config: StudioConfig, httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url=f"{BASE}/projects/p1", json={"records": [{"id": "p1", "name": "X"}]}
    )
    async with StudioClient(config) as studio:
        rec = await studio.get_project("p1")
    assert rec["id"] == "p1"
    assert rec["name"] == "X"


@pytest.mark.asyncio
async def test_delete_project_unwraps_nested_record(
    config: StudioConfig, httpx_mock: HTTPXMock
) -> None:
    # DELETE returns 202 with the deleted record nested under "record".
    httpx_mock.add_response(
        url=f"{BASE}/projects/p1",
        method="DELETE",
        status_code=202,
        json={"records": [{"record": {"id": "p1", "name": "X"}}]},
    )
    async with StudioClient(config) as studio:
        rec = await studio.delete_project("p1")
    assert rec["id"] == "p1"
