# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 OlmoEarth Agent contributors
"""The ``olmoearth-cloud-mask-audit`` tool bundle (skill #11).

Surfaces cloud-mask ensemble disagreement and, when given a model-error
mask, returns a bad-mask-vs-bad-model verdict. Operates on masks the
caller already has (the STAC + s2cloudless ``fetch_cloud_masks`` step is
a gated follow-up); returns summary statistics only, never per-pixel
geometry (operational rule §3.1).
"""

from __future__ import annotations

from typing import Any

from olmoearth_agent.analysis.cloud_mask import (
    DEFAULT_VERDICT_THRESHOLD,
    ensemble_disagree,
    verdict_classifier,
)
from olmoearth_agent.llm.types import ToolSpec
from olmoearth_agent.tools.registry import RegisteredTool, ToolContext

_MASKS_SCHEMA = {
    "type": "object",
    "description": (
        "Algorithm name -> aligned binary cloud mask (1 = cloud, 0 = "
        "clear). At least 2 masks, all the same length/order."
    ),
    "additionalProperties": {"type": "array", "items": {"type": "integer"}},
}

_BINARY_LIST = {"type": "array", "items": {"type": "integer"}}


async def _cloud_mask_audit(args: dict[str, Any], _ctx: ToolContext) -> dict[str, Any]:
    masks = {str(name): list(mask) for name, mask in args["masks"].items()}
    result: dict[str, Any] = {"ensemble": ensemble_disagree(masks)}
    model_error = args.get("model_error")
    if model_error is not None:
        result["attribution"] = verdict_classifier(
            masks,
            list(model_error),
            threshold=float(args.get("threshold", DEFAULT_VERDICT_THRESHOLD)),
        )
    return result


def build_cloud_mask_audit_tools() -> list[RegisteredTool]:
    """Return the ``olmoearth-cloud-mask-audit`` tool bundle."""
    return [
        RegisteredTool(
            spec=ToolSpec(
                name="olmoearth_cloud_mask_audit",
                description=(
                    "Audit an ensemble of cloud masks (CFMask, s2cloudless, "
                    "Sen2Cor, MAJA, or any others) for the same scene. "
                    "Returns where the algorithms agree vs disagree "
                    "(agreement/disagreement rates, per-algorithm cloud "
                    "fraction, pairwise disagreement, vote histogram) — it "
                    "surfaces disagreement, NOT a single ground-truth mask, "
                    "because algorithms diverge on thin/semi-transparent "
                    "cloud (Skakun CMIX 2022). If you also pass model_error "
                    "(1 where the prediction was wrong), it returns a verdict "
                    "on whether errors are cloud-mask-limited or "
                    "model-limited. Masks must be aligned binary arrays of "
                    "equal length. Read-only; returns summary stats only."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "masks": _MASKS_SCHEMA,
                        "model_error": {
                            **_BINARY_LIST,
                            "description": (
                                "Optional: 1 where the model's prediction was "
                                "wrong. Enables the bad-mask-vs-bad-model "
                                "verdict."
                            ),
                        },
                        "threshold": {
                            "type": "number",
                            "default": DEFAULT_VERDICT_THRESHOLD,
                        },
                    },
                    "required": ["masks"],
                },
            ),
            handler=_cloud_mask_audit,
        ),
    ]
