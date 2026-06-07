# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""The ``olmoearth-automate`` tool bundle (the in-repo facet of skill #3 olmoearth-embeddings).

One call that auto-decides **embeddings vs fine-tuning** for an Earth-observation
task and **proposes a config** (model size, classifier head, an embeddings
notebook command, a fine-tune schedule, and a hand-off to
``olmoearth-studio-job-config`` for the Studio wizard). Accepts either explicit
counts, a free-form task description, and/or a **Hugging Face dataset id** (whose
row count + ClassLabel classes are introspected via the public datasets-server).

Reuses the canonical decision logic from the vendored ``olmoearth-embeddings``
skill (``recommend.decide``); see ``analysis/automate.py``. Inputs are task
metadata only (no geometry) — provenance-safe.
"""

from __future__ import annotations

from typing import Any

from olmoearth_agent.analysis.automate import COMPUTE_TIERS, automate
from olmoearth_agent.llm.types import ToolSpec
from olmoearth_agent.tools.registry import RegisteredTool, ToolContext


async def _automate(args: dict[str, Any], _ctx: ToolContext) -> dict[str, Any]:
    return await automate(
        task=args.get("task"),
        num_samples=args.get("num_samples"),
        num_classes=args.get("num_classes"),
        compute=args.get("compute"),
        goal=args.get("goal"),
        hf_dataset=args.get("hf_dataset"),
    )


def build_automate_tools() -> list[RegisteredTool]:
    """Return the ``olmoearth-automate`` tool bundle."""
    return [
        RegisteredTool(
            spec=ToolSpec(
                name="olmoearth_automate",
                description=(
                    "Auto-decide embeddings vs fine-tuning for an OlmoEarth task "
                    "AND propose a full config (model size, classifier head, an "
                    "embeddings-notebook command, a fine-tune schedule, and a "
                    "hand-off to the Studio job wizard). Use when the user asks "
                    "'should I fine-tune or use embeddings', 'set this up for me', "
                    "'auto-configure', 'what model size', or points at a dataset "
                    "and wants a recommended pipeline. Provide what you know: a "
                    "free-text `task`, explicit `num_samples`/`num_classes`/"
                    "`compute`/`goal`, and/or a Hugging Face `hf_dataset` id (its "
                    "row count + label classes are read from the public "
                    "datasets-server). Returns the decision, rationale, a config, "
                    "and `ask_for` when key inputs are missing. Honest about "
                    "missing inputs; never invents dataset stats."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "task": {
                            "type": "string",
                            "description": "Free-form description, e.g. 'land cover, 9 classes, 200 samples, T4 GPU'.",
                        },
                        "num_samples": {
                            "type": "integer",
                            "description": "Labeled sample count.",
                        },
                        "num_classes": {
                            "type": "integer",
                            "description": "Number of label classes.",
                        },
                        "compute": {
                            "type": "string",
                            "enum": COMPUTE_TIERS,
                            "description": "Available compute tier (weakest→strongest).",
                        },
                        "goal": {
                            "type": "string",
                            "enum": [
                                "prototyping",
                                "production",
                                "similarity",
                                "no_labels",
                            ],
                        },
                        "hf_dataset": {
                            "type": "string",
                            "description": "Optional Hugging Face dataset id (e.g. 'org/name') to introspect for size + classes.",
                        },
                    },
                    "required": [],
                },
            ),
            handler=_automate,
        ),
    ]
