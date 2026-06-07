# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Foundational ``olmoearth.*`` tools that wrap the Studio API.

This is the minimal bundle every Studio-calling skill builds on:
context, project search, project creation, prediction status. Skills add
their own tools (predict, evaluate, …) as separate bundles registered
alongside these.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from olmoearth_agent.llm.types import ToolSpec
from olmoearth_agent.tools.registry import RegisteredTool, ToolContext


async def _load_context(_args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    studio_ctx = await ctx.studio.load_context()
    ctx.state.studio_context = studio_ctx
    return {
        "user_name": studio_ctx.user_name,
        "organization": studio_ctx.organization,
        "projects": [asdict(p) for p in studio_ctx.projects],
        "project_count": len(studio_ctx.projects),
    }


async def _search_projects(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    limit = int(args.get("limit", 50))
    offset = int(args.get("offset", 0))
    env = await ctx.studio.search_projects(limit=limit, offset=offset)
    return {
        "total": env.total,
        "projects": [
            {"id": r.get("id"), "name": r.get("name"),
             "creation_time": r.get("creation_time")}
            for r in env.records
        ],
    }


async def _create_project(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    record = await ctx.studio.create_project(
        name=args["name"], description=args.get("description", "")
    )
    return {"id": record.get("id"), "name": record.get("name")}


async def _request_aoi(args: dict[str, Any], _ctx: ToolContext) -> dict[str, Any]:
    """Signal the UI to surface the map draw widget; no Studio call.

    Returns a ``needs_aoi`` directive the web UI recognizes (run.js) and
    renders as an interactive "draw the area" prompt. The drawn polygon is
    stored as a Studio area and its ``area_id`` + ``bbox`` arrive on the
    user's next message, so the agent need not ask for coordinates in text.
    """
    return {
        "needs_aoi": True,
        "purpose": str(args.get("purpose", "")).strip(),
        "suggested_name": str(args.get("suggested_name", "")).strip(),
    }


async def _get_prediction(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    record = await ctx.studio.get_prediction(args["prediction_id"])
    return {
        "id": record.get("id"),
        "status": record.get("status"),
        "model_id": record.get("model_id"),
    }


def build_studio_tools() -> list[RegisteredTool]:
    """Return the foundational ``olmoearth.*`` tool bundle.

    Handlers read the :class:`StudioClient` from the per-call
    :class:`ToolContext`, so one bundle works for any authenticated user.
    """
    return [
        RegisteredTool(
            spec=ToolSpec(
                name="olmoearth_load_context",
                description=(
                    "Load the user's active OlmoEarth Studio context: "
                    "identity, organization, and available projects. Call "
                    "this first to discover what projects already exist."
                ),
                parameters={"type": "object", "properties": {}, "required": []},
            ),
            handler=_load_context,
        ),
        RegisteredTool(
            spec=ToolSpec(
                name="olmoearth_search_projects",
                description=(
                    "Search the user's OlmoEarth Studio projects. Returns "
                    "id, name, and creation_time for each, plus the total "
                    "count. Read-only."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "limit": {"type": "integer", "default": 50},
                        "offset": {"type": "integer", "default": 0},
                    },
                    "required": [],
                },
            ),
            handler=_search_projects,
        ),
        RegisteredTool(
            spec=ToolSpec(
                name="olmoearth_create_project",
                description=(
                    "Create a new OlmoEarth Studio project. Returns the new "
                    "project's id and name. Use only after confirming a "
                    "project with the same name does not already exist."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "description": {"type": "string"},
                    },
                    "required": ["name", "description"],
                },
            ),
            handler=_create_project,
        ),
        RegisteredTool(
            spec=ToolSpec(
                name="olmoearth_request_aoi",
                description=(
                    "Ask the user to DRAW an area of interest (AOI) on a map "
                    "in the chat. Call this when a task needs a geographic "
                    "area and none was provided (no area_id, bbox, or polygon "
                    "in the brief). The user draws a rectangle or polygon; it "
                    "is stored as a Studio area, and its area_id + bbox arrive "
                    "with their next message. Prefer this over asking the user "
                    "to type coordinates or a bbox. After calling it, stop and "
                    "tell the user to draw the area on the map."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "purpose": {
                            "type": "string",
                            "description": "What the AOI is for (e.g. 'change "
                            "detection over 2023-2024').",
                        },
                        "suggested_name": {
                            "type": "string",
                            "description": "Optional default name for the area.",
                        },
                    },
                    "required": [],
                },
            ),
            handler=_request_aoi,
        ),
        RegisteredTool(
            spec=ToolSpec(
                name="olmoearth_get_prediction",
                description=(
                    "Fetch the status of one prediction by id. Returns id, "
                    "status (pending/running/completed/failed/cancelled), and "
                    "model_id."
                ),
                parameters={
                    "type": "object",
                    "properties": {"prediction_id": {"type": "string"}},
                    "required": ["prediction_id"],
                },
            ),
            handler=_get_prediction,
        ),
    ]
