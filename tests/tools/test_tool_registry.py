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


def test_deferred_groups_and_active_specs() -> None:
    async def h(_a: dict[str, Any], _c: ToolContext) -> None: ...

    registry = ToolRegistry()
    registry.register(RegisteredTool(spec=_spec("core"), handler=h))
    registry.register_all(
        [
            RegisteredTool(spec=_spec("d1"), handler=h),
            RegisteredTool(spec=_spec("d2"), handler=h),
        ],
        group="g",
    )
    registry.register(RegisteredTool(spec=_spec("e1"), handler=h), group="other")
    assert registry.names() == ["core", "d1", "d2", "e1"]
    assert [s.name for s in registry.specs()] == ["core", "d1", "d2", "e1"]
    assert registry.groups() == {"g": ["d1", "d2"], "other": ["e1"]}
    assert registry.group_of("d1") == "g" and registry.group_of("core") is None
    assert [s.name for s in registry.active_specs()] == ["core"]
    assert [s.name for s in registry.active_specs({"g"})] == ["core", "d1", "d2"]
    # Re-registering without a group makes the tool core again.
    registry.register(RegisteredTool(spec=_spec("d2"), handler=h))
    assert [s.name for s in registry.active_specs()] == ["core", "d2"]


@pytest.mark.asyncio
async def test_dispatching_a_deferred_tool_runs_it_and_loads_its_group() -> None:
    from olmoearth_agent.harness.state import ThreadState

    async def echo(args: dict[str, Any], _ctx: ToolContext) -> dict[str, Any]:
        return args

    registry = ToolRegistry()
    registry.register(RegisteredTool(spec=_spec("late"), handler=echo), group="g")
    state = ThreadState()
    result = await registry.dispatch(
        ToolCall(id="c1", name="late", arguments={"x": 1}),
        ctx=ToolContext(studio=None, state=state),  # type: ignore[arg-type]
    )
    assert result == {"ok": True, "result": {"x": 1}}
    assert state.loaded_groups == {"g"}
    # A context without state (as in unit tests) still dispatches.
    again = await registry.dispatch(
        ToolCall(id="c2", name="late", arguments={}),
        ctx=ToolContext(studio=None, state=None),  # type: ignore[arg-type]
    )
    assert again["ok"] is True


def _plan_like_registry() -> ToolRegistry:
    """A tool that refuses every budget above a ceiling, whatever the grid."""

    async def plan(args: dict[str, Any], _ctx: ToolContext) -> None:
        if int(args.get("budget", 0)) > 173:
            raise ValueError(
                f"budget {args['budget']} is more than the 173 valid windows "
                f"(grid {args.get('grid')} was capped at 16)"
            )
        raise KeyError("something else")

    async def other(_args: dict[str, Any], _ctx: ToolContext) -> None:
        raise ValueError("budget 300 is more than the 173 valid windows")

    registry = ToolRegistry()
    registry.register(RegisteredTool(spec=_spec("plan"), handler=plan))
    registry.register(RegisteredTool(spec=_spec("other"), handler=other))
    return registry


@pytest.mark.asyncio
async def test_the_same_failure_twice_tells_the_model_to_stop_and_report() -> None:
    """exp86 round 1: the model retried the same refused plan five times, told
    only "do not retry with identical arguments", and never answered."""
    from olmoearth_agent.harness.state import ThreadState

    registry = _plan_like_registry()
    ctx = ToolContext(studio=None, state=ThreadState())  # type: ignore[arg-type]

    async def call(name: str, **args: Any) -> dict[str, Any]:
        return await registry.dispatch(ToolCall(id="c", name=name, arguments=args), ctx)

    first = await call("plan", budget=300, grid=20)
    assert first["ok"] is False
    assert "stop" not in first["hint"].lower()
    # New arguments, the same wall (numbers aside): stop and report it.
    second = await call("plan", budget=300, grid=30)
    assert "Stop retrying" in second["hint"]
    assert "tell the user" in second["hint"]
    assert second["same_error_count"] == 2
    third = await call("plan", budget=250, grid=40)
    assert third["same_error_count"] == 3 and "Stop retrying" in third["hint"]
    # A different error, or the same error from another tool, starts afresh.
    different = await call("plan", budget=1)
    assert "Stop retrying" not in different["hint"]
    assert "Stop retrying" not in (await call("other"))["hint"]
    # A new run (a new state) starts afresh too.
    fresh = ToolContext(studio=None, state=ThreadState())  # type: ignore[arg-type]
    again = await registry.dispatch(
        ToolCall(id="c", name="plan", arguments={"budget": 300}), fresh
    )
    assert "Stop retrying" not in again["hint"]


@pytest.mark.asyncio
async def test_repeated_invalid_arguments_also_say_stop() -> None:
    from olmoearth_agent.harness.state import ThreadState

    async def needs_x(_args: dict[str, Any], _ctx: ToolContext) -> None:
        return None

    schema = {"type": "object", "properties": {"x": {"type": "integer"}}}
    registry = ToolRegistry()
    registry.register(RegisteredTool(ToolSpec("t", "t", schema), needs_x))
    ctx = ToolContext(studio=None, state=ThreadState())  # type: ignore[arg-type]
    for expected in (False, True):
        out = await registry.dispatch(
            ToolCall(id="c", name="t", arguments={"x": "one"}), ctx
        )
        assert out["ok"] is False
        assert ("Stop retrying" in out["hint"]) is expected
