# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""DOUBLE-GATED live write test: create -> get -> delete a throwaway project.

This is the only test that writes to a real OlmoEarth Studio account. It
requires BOTH ``OLMOEARTH_WRITE_TESTS=1`` and ``OLMOEARTH_API_KEY`` so it
can never run by accident during a normal integration run. It cleans up
after itself (deletes the project it created).

    OLMOEARTH_WRITE_TESTS=1 OLMOEARTH_API_KEY=... \
        uv run pytest tests/studio/test_write_live.py -m integration -v
"""

from __future__ import annotations

import os
import time

import pytest

_ENABLED = (
    os.environ.get("OLMOEARTH_WRITE_TESTS") == "1"
    and bool(os.environ.get("OLMOEARTH_API_KEY"))
)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_live_project_create_get_delete() -> None:
    if not _ENABLED:
        pytest.skip("set OLMOEARTH_WRITE_TESTS=1 + OLMOEARTH_API_KEY to run")

    import httpx

    from olmoearth_agent.studio import StudioClient

    name = f"zzz-agent-write-test-{int(time.time())}"
    async with StudioClient.from_env() as studio:
        created = await studio.create_project(
            name=name, description="throwaway; auto-deleted by the write test"
        )
        project_id = created["id"]
        assert created["name"] == name

        try:
            fetched = await studio.get_project(project_id)
            assert fetched["id"] == project_id
            assert fetched["name"] == name
        finally:
            deleted = await studio.delete_project(project_id)
            assert deleted.get("id") == project_id

        # Confirm it's gone.
        with pytest.raises(httpx.HTTPStatusError):
            await studio.get_project(project_id)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_live_area_create_get_delete() -> None:
    if not _ENABLED:
        pytest.skip("set OLMOEARTH_WRITE_TESTS=1 + OLMOEARTH_API_KEY to run")

    import httpx

    from olmoearth_agent.studio import StudioClient

    name = f"zzz-agent-area-test-{int(time.time())}"
    geom = {
        "type": "Polygon",
        "coordinates": [
            [[-122.34, 47.62], [-122.33, 47.62], [-122.33, 47.63], [-122.34, 47.62]]
        ],
    }
    async with StudioClient.from_env() as studio:
        project = await studio.create_project(
            name=name, description="throwaway; auto-deleted by the area write test"
        )
        project_id = project["id"]
        area_id = None
        try:
            area = await studio.create_area(
                name="zzz-area", geom=geom, project_id=project_id
            )
            area_id = area["id"]
            assert area_id
            fetched = await studio.get_area(area_id)
            assert fetched["id"] == area_id
            assert fetched["geom"]["type"] == "Polygon"
        finally:
            if area_id:
                await studio.delete_area(area_id)
            await studio.delete_project(project_id)

        if area_id:
            with pytest.raises(httpx.HTTPStatusError):
                await studio.get_area(area_id)
