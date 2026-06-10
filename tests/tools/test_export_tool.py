# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Tests for the olmoearth_export_data tool."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from olmoearth_agent.harness.state import ThreadState
from olmoearth_agent.tools.export import build_export_tools
from olmoearth_agent.tools.registry import ToolContext
from olmoearth_agent.types import ApiEnvelope


class _FakeStudio:
    def __init__(
        self, projects: list[dict[str, Any]], predictions: list[dict[str, Any]]
    ) -> None:
        self._projects = projects
        self._predictions = predictions

    async def search_projects(
        self, *, limit: int = 200, offset: int = 0
    ) -> ApiEnvelope[dict[str, Any]]:
        return ApiEnvelope(records=self._projects, meta={"total": len(self._projects)})

    async def search_predictions(
        self, *, project_id: str | None = None, limit: int = 1000, offset: int = 0
    ) -> ApiEnvelope[dict[str, Any]]:
        return ApiEnvelope(
            records=self._predictions, meta={"total": len(self._predictions)}
        )


def _ctx(studio: _FakeStudio) -> ToolContext:
    return ToolContext(studio=studio, state=ThreadState())  # type: ignore[arg-type]


@pytest.fixture(autouse=True)
def _confine_io_to_tmp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the path-traversal workspace root at this test's tmp_path so the
    tool's absolute tmp_path arguments resolve inside the workspace."""
    monkeypatch.setenv("OLMOEARTH_OUTPUT_ROOT", str(tmp_path))


@pytest.mark.asyncio
async def test_export_by_project(tmp_path: Path) -> None:
    studio = _FakeStudio(
        projects=[{"id": "p1", "name": "Proj One", "creation_time": "t"}],
        predictions=[
            {"id": "r1", "name": "pred", "status": "completed",
             "project_id": "p1", "model_id": "m1"}
        ],
    )
    tool = build_export_tools()[0]
    result = await tool.handler(
        {"group_by": "project", "out_dir": str(tmp_path)}, _ctx(studio)
    )
    assert result["groups"] == 1
    assert result["project_count"] == 1
    out = tmp_path / "project_proj-one.json"
    assert out.exists()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["project"]["id"] == "p1"
    assert data["predictions"][0]["id"] == "r1"


@pytest.mark.asyncio
async def test_export_by_status(tmp_path: Path) -> None:
    studio = _FakeStudio(
        projects=[],
        predictions=[
            {"id": "r1", "status": "completed", "project_id": "p1"},
            {"id": "r2", "status": "failed", "project_id": "p1"},
        ],
    )
    tool = build_export_tools()[0]
    result = await tool.handler(
        {"group_by": "status", "out_dir": str(tmp_path)}, _ctx(studio)
    )
    assert result["groups"] == 2
    assert (tmp_path / "status_completed.json").exists()
    assert (tmp_path / "status_failed.json").exists()


@pytest.mark.asyncio
async def test_export_rejects_out_dir_traversal(tmp_path: Path) -> None:
    # A model-controlled out_dir escaping the workspace root is refused.
    from olmoearth_agent.security.paths import PathTraversalError

    studio = _FakeStudio(projects=[], predictions=[])
    tool = build_export_tools()[0]
    with pytest.raises(PathTraversalError):
        await tool.handler(
            {"group_by": "status", "out_dir": "../escaped_exports"}, _ctx(studio)
        )
