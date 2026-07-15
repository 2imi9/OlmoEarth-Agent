# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Unit tests for the tool registry and dispatch error handling."""

from __future__ import annotations

from typing import Any

import pytest

from olmoearth_agent.llm.types import ToolCall, ToolSpec
from olmoearth_agent.tools.registry import (
    RegisteredTool,
    ToolContext,
    ToolRegistry,
)

_EMPTY_SCHEMA = {"type": "object", "properties": {}, "required": []}


def _spec(name: str) -> ToolSpec:
    return ToolSpec(name=name, description="t", parameters=_EMPTY_SCHEMA)


@pytest.mark.asyncio
async def test_dispatch_unknown_tool_returns_error() -> None:
    registry = ToolRegistry()
    result = await registry.dispatch(
        ToolCall(id="c1", name="nope", arguments={}),
        ctx=ToolContext(studio=None, state=None),  # type: ignore[arg-type]
    )
    assert result["ok"] is False
    assert "unknown tool" in result["error"]


@pytest.mark.asyncio
async def test_dispatch_handler_exception_is_caught() -> None:
    async def boom(_args: dict[str, Any], _ctx: ToolContext) -> None:
        raise ValueError("bad input")

    registry = ToolRegistry()
    registry.register(RegisteredTool(spec=_spec("boom"), handler=boom))
    result = await registry.dispatch(
        ToolCall(id="c1", name="boom", arguments={}),
        ctx=ToolContext(studio=None, state=None),  # type: ignore[arg-type]
    )
    assert result["ok"] is False
    assert "ValueError: bad input" in result["error"]


@pytest.mark.asyncio
async def test_dispatch_success_wraps_result() -> None:
    async def echo(args: dict[str, Any], _ctx: ToolContext) -> dict[str, Any]:
        return args

    registry = ToolRegistry()
    registry.register(RegisteredTool(spec=_spec("echo"), handler=echo))
    result = await registry.dispatch(
        ToolCall(id="c1", name="echo", arguments={"x": 1}),
        ctx=ToolContext(studio=None, state=None),  # type: ignore[arg-type]
    )
    assert result == {"ok": True, "result": {"x": 1}}


@pytest.mark.asyncio
async def test_dispatch_rejects_missing_required_before_handler_runs() -> None:
    ran = False

    async def handler(_args: dict[str, Any], _ctx: ToolContext) -> str:
        nonlocal ran
        ran = True
        return "should not run"

    spec = ToolSpec(
        name="strict",
        description="t",
        parameters={
            "type": "object",
            "properties": {"project_id": {"type": "string"}},
            "required": ["project_id"],
        },
    )
    registry = ToolRegistry()
    registry.register(RegisteredTool(spec=spec, handler=handler))
    result = await registry.dispatch(
        ToolCall(id="c1", name="strict", arguments={}),
        ctx=ToolContext(studio=None, state=None),  # type: ignore[arg-type]
    )
    assert result["ok"] is False
    assert ran is False
    assert "missing required argument 'project_id'" in result["error"]
    # Self-documenting: the envelope restates the expected schema + a hint.
    assert result["expected_arguments"]["required"] == ["project_id"]
    assert "hint" in result


@pytest.mark.asyncio
async def test_dispatch_rejects_wrong_type_and_enum() -> None:
    async def handler(args: dict[str, Any], _ctx: ToolContext) -> dict[str, Any]:
        return args

    spec = ToolSpec(
        name="typed",
        description="t",
        parameters={
            "type": "object",
            "properties": {
                "limit": {"type": "integer"},
                "mode": {"type": "string", "enum": ["fast", "full"]},
            },
            "required": [],
        },
    )
    registry = ToolRegistry()
    registry.register(RegisteredTool(spec=spec, handler=handler))
    result = await registry.dispatch(
        ToolCall(id="c1", name="typed", arguments={"limit": "ten", "mode": "turbo"}),
        ctx=ToolContext(studio=None, state=None),  # type: ignore[arg-type]
    )
    assert result["ok"] is False
    assert "'limit'" in result["error"] and "'mode'" in result["error"]


@pytest.mark.asyncio
async def test_dispatch_exception_envelope_names_tool_and_hints() -> None:
    async def boom(_args: dict[str, Any], _ctx: ToolContext) -> None:
        raise RuntimeError("downstream failed")

    registry = ToolRegistry()
    registry.register(RegisteredTool(spec=_spec("boom"), handler=boom))
    result = await registry.dispatch(
        ToolCall(id="c1", name="boom", arguments={}),
        ctx=ToolContext(studio=None, state=None),  # type: ignore[arg-type]
    )
    assert result["ok"] is False
    assert result["tool"] == "boom"
    assert "hint" in result


def test_register_overwrites_by_name() -> None:
    registry = ToolRegistry()

    async def h(_a: dict[str, Any], _c: ToolContext) -> None: ...

    registry.register(RegisteredTool(spec=_spec("dup"), handler=h))
    registry.register(RegisteredTool(spec=_spec("dup"), handler=h))
    assert registry.names() == ["dup"]
