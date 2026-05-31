# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Tests for the opt-in olmoearth_run_python sandbox tool."""

from __future__ import annotations

from typing import Any

import pytest

from olmoearth_agent.tools.system import build_system_tools


def test_disabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OLMOEARTH_RUN_PYTHON", raising=False)
    assert build_system_tools() == []


def test_enabled_registers_one_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLMOEARTH_RUN_PYTHON", "1")
    tools = build_system_tools()
    assert len(tools) == 1
    assert tools[0].spec.name == "olmoearth_run_python"


def _handler(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setenv("OLMOEARTH_RUN_PYTHON", "1")
    return build_system_tools()[0].handler


@pytest.mark.asyncio
async def test_runs_and_captures_stdout(monkeypatch: pytest.MonkeyPatch) -> None:
    res = await _handler(monkeypatch)({"code": "print('hello', 1 + 1)"}, None)
    assert res["ok"] is True
    assert res["returncode"] == 0
    assert "hello 2" in res["stdout"]


@pytest.mark.asyncio
async def test_nonzero_exit_is_not_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    res = await _handler(monkeypatch)({"code": "raise SystemExit(3)"}, None)
    assert res["ok"] is False
    assert res["returncode"] == 3


@pytest.mark.asyncio
async def test_empty_code(monkeypatch: pytest.MonkeyPatch) -> None:
    res = await _handler(monkeypatch)({"code": "   "}, None)
    assert res["ok"] is False
    assert "no code" in res["error"]


@pytest.mark.asyncio
async def test_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLMOEARTH_RUN_PYTHON_TIMEOUT", "0.5")
    res = await _handler(monkeypatch)({"code": "import time; time.sleep(5)"}, None)
    assert res["ok"] is False
    assert "timed out" in res["error"]
