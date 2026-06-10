# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Tests for the olmoearth-evaluate tool bundle."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from olmoearth_agent.harness.state import ThreadState
from olmoearth_agent.tools.evaluate import build_evaluate_tools
from olmoearth_agent.tools.registry import RegisteredTool, ToolContext


def _tool(name: str) -> RegisteredTool:
    return next(t for t in build_evaluate_tools() if t.spec.name == name)


def _ctx() -> ToolContext:
    return ToolContext(studio=None, state=ThreadState())  # type: ignore[arg-type]


@pytest.fixture(autouse=True)
def _confine_io_to_tmp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the path-traversal workspace root at this test's tmp_path so the
    tool's absolute tmp_path output paths resolve inside the workspace."""
    monkeypatch.setenv("OLMOEARTH_OUTPUT_ROOT", str(tmp_path))


@pytest.mark.asyncio
async def test_cv_inflation_check_tool() -> None:
    clusters = [(0.0, 0.0), (10.0, 0.0), (20.0, 0.0)]
    points = [[cx + 0.02 * k, cy + 0.02 * k] for cx, cy in clusters for k in range(5)]
    result = await _tool("olmoearth_cv_inflation_check").handler(
        {"points": points, "n_folds": 3}, _ctx()
    )
    assert result["inflation_risk"] in ("moderate", "high")
    assert result["spatial_to_random_ratio"] >= 1.5


@pytest.mark.asyncio
async def test_classification_metrics_tool() -> None:
    result = await _tool("olmoearth_classification_metrics").handler(
        {"y_true": [1, 1, 0, 0], "y_pred": [1, 0, 0, 0]}, _ctx()
    )
    assert result["accuracy"] == 0.75
    assert result["per_class"]["1"]["recall"] == 0.5


@pytest.mark.asyncio
async def test_nndm_cv_tool_returns_folds_inline() -> None:
    points = [[0.0, 0.0], [0.05, 0.0], [5.0, 5.0], [5.05, 5.0]]
    pred = [[x, y] for x in (-2.0, 2.0, 6.0) for y in (-2.0, 6.0)]
    result = await _tool("olmoearth_nndm_cv").handler(
        {"points": points, "pred_points": pred}, _ctx()
    )
    assert result["n_train"] == 4
    assert "cv_optimism_loo_km" in result
    assert "cv_optimism_nndm_km" in result
    assert len(result["folds"]) == 4
    assert result["folds"][0]["test"] == 0


@pytest.mark.asyncio
async def test_nndm_cv_tool_writes_folds_to_file(tmp_path: Path) -> None:
    points = [[0.0, 0.0], [0.05, 0.0], [5.0, 5.0], [5.05, 5.0]]
    pred = [[x, y] for x in (-2.0, 2.0, 6.0) for y in (-2.0, 6.0)]
    out = tmp_path / "folds.json"
    result = await _tool("olmoearth_nndm_cv").handler(
        {"points": points, "pred_points": pred, "output_path": str(out)}, _ctx()
    )
    # Folds go to the file, not the chat result. The tool returns the resolved
    # (workspace-confined) path.
    assert "folds" not in result
    assert result["folds_path"] == str(out.resolve())
    assert result["folds_written"] == 4
    written = json.loads(out.read_text(encoding="utf-8"))
    assert len(written["folds"]) == 4
    assert "phi_km" in written


@pytest.mark.asyncio
async def test_nndm_cv_tool_rejects_output_path_traversal(tmp_path: Path) -> None:
    # A model-controlled output_path escaping the workspace root is refused.
    from olmoearth_agent.security.paths import PathTraversalError

    points = [[0.0, 0.0], [0.05, 0.0], [5.0, 5.0], [5.05, 5.0]]
    pred = [[x, y] for x in (-2.0, 2.0, 6.0) for y in (-2.0, 6.0)]
    with pytest.raises(PathTraversalError):
        await _tool("olmoearth_nndm_cv").handler(
            {"points": points, "pred_points": pred,
             "output_path": "../escaped_folds.json"},
            _ctx(),
        )
