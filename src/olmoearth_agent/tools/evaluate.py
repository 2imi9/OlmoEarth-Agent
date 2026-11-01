# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""The ``olmoearth-evaluate`` tool bundle (skill #7).

Classification metrics from ground-truth labels the caller gives.
"""

from __future__ import annotations

from typing import Any

from olmoearth_agent.evaluation.metrics import classification_metrics
from olmoearth_agent.llm.types import ToolSpec
from olmoearth_agent.tools.registry import Capability, RegisteredTool, ToolContext

_LABEL_LIST = {
    "type": "array",
    "items": {"type": ["integer", "string"]},
    "description": "Class labels: integer codes or string names, one per pixel.",
}


async def _classification_metrics(
    args: dict[str, Any], _ctx: ToolContext
) -> dict[str, Any]:
    return classification_metrics(args["y_true"], args["y_pred"])


def build_evaluate_tools() -> list[RegisteredTool]:
    """Return the ``olmoearth-evaluate`` tool bundle."""
    return [
        RegisteredTool(
            spec=ToolSpec(
                name="olmoearth_classification_metrics",
                description=(
                    "Compute accuracy plus per-class precision / recall / "
                    "F1 / IoU, macro-F1, and mean IoU from aligned "
                    "ground-truth and predicted label lists."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "y_true": _LABEL_LIST,
                        "y_pred": _LABEL_LIST,
                    },
                    "required": ["y_true", "y_pred"],
                },
            ),
            handler=_classification_metrics,
            capability=Capability(
                does="accuracy and per-class precision, recall, F1 and IoU from "
                "ground-truth labels the caller gives",
                needs=("y_true and y_pred of the same length",),
            ),
        ),
    ]
