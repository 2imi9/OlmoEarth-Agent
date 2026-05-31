# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""The data-export tool bundle (skill #13, reframed).

Exports the user's Studio projects + predictions, grouped by project or
by status, as JSON files. Curated to ids / names / statuses / times:
no raw geometry (operational rule §3.1), so it's safe to share.
"""

from __future__ import annotations

import os
from typing import Any

from olmoearth_agent.llm.types import ToolSpec
from olmoearth_agent.reporting.export import curate, group_items, slugify, to_json
from olmoearth_agent.tools.registry import RegisteredTool, ToolContext

_PROJECT_FIELDS = ("id", "name", "description", "creation_time")
_PREDICTION_FIELDS = (
    "id",
    "name",
    "status",
    "model_id",
    "project_id",
    "start_time",
    "end_time",
    "creation_time",
)


def _write(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(to_json(obj))


async def _export_data(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    group_by = args.get("group_by", "project")
    out_dir = args.get("out_dir", "exports")
    os.makedirs(out_dir, exist_ok=True)

    projects = [
        curate(p, _PROJECT_FIELDS)
        for p in (await ctx.studio.search_projects(limit=200)).records
    ]
    predictions = [
        curate(p, _PREDICTION_FIELDS)
        for p in (await ctx.studio.search_predictions(limit=1000)).records
    ]

    files: list[str] = []
    if group_by == "status":
        for status, items in group_items(predictions, "status").items():
            path = os.path.join(out_dir, f"status_{slugify(status)}.json")
            _write(
                path,
                {
                    "status": status,
                    "prediction_count": len(items),
                    "predictions": items,
                },
            )
            files.append(path)
    else:  # default: by project
        by_project = group_items(predictions, "project_id")
        for project in projects:
            payload = {
                "project": project,
                "predictions": by_project.get(str(project.get("id")), []),
            }
            path = os.path.join(out_dir, f"project_{slugify(project.get('name'))}.json")
            _write(path, payload)
            files.append(path)

    return {
        "group_by": group_by,
        "out_dir": out_dir,
        "groups": len(files),
        "files": files,
        "project_count": len(projects),
        "prediction_count": len(predictions),
    }


def build_export_tools() -> list[RegisteredTool]:
    """Return the data-export tool bundle."""
    return [
        RegisteredTool(
            spec=ToolSpec(
                name="olmoearth_export_data",
                description=(
                    "Export the user's Studio projects + predictions to JSON "
                    "files, grouped by 'project' (default) or 'status'. "
                    "Curated metadata only (ids, names, statuses, times: no "
                    "raw geometry). Returns the written file paths and counts."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "group_by": {
                            "type": "string",
                            "enum": ["project", "status"],
                            "default": "project",
                        },
                        "out_dir": {"type": "string", "default": "exports"},
                    },
                    "required": [],
                },
            ),
            handler=_export_data,
        ),
    ]
