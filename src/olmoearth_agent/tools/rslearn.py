# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""The ``olmoearth-rslearn`` in-repo tool bundle (skill #17).

Two torch-free tools that let a scientist who doesn't know rslearn still set up a
correct OlmoEarth/rslearn experiment:

* ``olmoearth_rslearn_recommend`` — research goal -> a complete, explained setup
  (task, data layout, model composition, task knobs, fine-tune schedule).
* ``olmoearth_rslearn_validate`` — a dataset config.json + model.yaml -> the
  errors rslearn only surfaces hours into a training run (shape/channel/label-type
  mismatches), caught up front.

Logic lives in ``analysis/rslearn_advisor.py`` (pure metadata, no torch, no
network). These complement the vendored ``olmoearth-rslearn`` SKILL.md (loaded
via ``olmoearth_load_skill``), which teaches *running* the pipeline.
"""

from __future__ import annotations

from typing import Any

from olmoearth_agent.analysis.rslearn_advisor import (
    EMBEDDING_SIZES,
    TASKS,
    compose,
    diagnose,
    recommend,
    validate_config,
)
from olmoearth_agent.llm.types import ToolSpec
from olmoearth_agent.tools.registry import RegisteredTool, ToolContext


async def _recommend(args: dict[str, Any], _ctx: ToolContext) -> dict[str, Any]:
    return recommend(
        goal=args.get("goal"),
        task=args.get("task"),
        label_kind=args.get("label_kind"),
        num_classes=args.get("num_classes"),
        label_range=args.get("label_range"),
        sensor=args.get("sensor"),
        cloudy=args.get("cloudy"),
        temporal=args.get("temporal"),
        num_samples=args.get("num_samples"),
        model_size=args.get("model_size"),
    )


async def _validate(args: dict[str, Any], _ctx: ToolContext) -> dict[str, Any]:
    return validate_config(
        dataset_config=args.get("dataset_config"),
        model_config=args.get("model_config"),
    )


async def _compose(args: dict[str, Any], _ctx: ToolContext) -> dict[str, Any]:
    return compose(
        task=args.get("task"),
        goal=args.get("goal"),
        model_size=args.get("model_size", "base"),
        num_classes=args.get("num_classes"),
        class_names=args.get("class_names"),
        property_name=args.get("property_name", "category"),
        scale_factor=args.get("scale_factor"),
        dataset_path=args.get("dataset_path", "data/dataset"),
        bands=args.get("bands"),
        freeze_epochs=args.get("freeze_epochs", 10),
        total_epochs=args.get("total_epochs", 40),
        nodata_value=args.get("nodata_value"),
    )


async def _diagnose(args: dict[str, Any], _ctx: ToolContext) -> dict[str, Any]:
    return diagnose(summary=args.get("summary"), log_text=args.get("log_text"))


def build_rslearn_tools() -> list[RegisteredTool]:
    """Return the ``olmoearth-rslearn`` tool bundle (recommend + validate)."""
    return [
        RegisteredTool(
            spec=ToolSpec(
                name="olmoearth_rslearn_recommend",
                description=(
                    "Recommend a complete, EXPLAINED rslearn/OlmoEarth training setup "
                    "from a plain-language research goal — for users who don't know "
                    "rslearn. Use when someone says 'I want to map/detect/measure X "
                    "from satellite imagery', 'how do I set up training for ...', "
                    "'which task/model should I use', or 'help me fine-tune OlmoEarth "
                    "for my research'. Returns the rslearn task "
                    f"({', '.join(TASKS)}), the data layout (data_source + space_mode "
                    "+ compositing + label layer type), the model composition "
                    "(OlmoEarth encoder -> decoder -> head with the channel contract), "
                    "the task knobs (metrics / loss / nodata / scale_factor), and a "
                    "fine-tune schedule — each with a one-line why. Provide what you "
                    "know; missing inputs come back in `ask_for`. Pure guidance "
                    "(no training is run); pair with olmoearth_rslearn_validate."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "goal": {
                            "type": "string",
                            "description": "Plain-language research goal, e.g. 'map flooded buildings after a storm' or 'estimate canopy height per pixel'.",
                        },
                        "task": {
                            "type": "string",
                            "enum": list(TASKS),
                            "description": "Explicit rslearn task, if known (otherwise inferred from `goal`).",
                        },
                        "label_kind": {
                            "type": "string",
                            "description": "What the labels look like, e.g. 'polygons', 'points', 'a class raster', 'a value raster'.",
                        },
                        "num_classes": {"type": "integer", "description": "Number of label classes (for classification/segmentation/detection)."},
                        "label_range": {
                            "type": "array",
                            "items": {"type": "number"},
                            "description": "[min, max] of the value being predicted (regression tasks), used to suggest scale_factor.",
                        },
                        "sensor": {"type": "string", "description": "Imagery, e.g. 'Sentinel-2', 'Sentinel-1/SAR', 'Landsat', 'NAIP'."},
                        "cloudy": {"type": "boolean", "description": "Is the region/season cloud-prone? Affects compositing."},
                        "temporal": {
                            "type": "string",
                            "enum": ["single", "multi-month", "seasonal", "annual"],
                            "description": "Temporal scope of the imagery.",
                        },
                        "num_samples": {"type": "integer", "description": "Labeled sample count (drives encoder size + freeze schedule)."},
                        "model_size": {
                            "type": "string",
                            "enum": list(EMBEDDING_SIZES),
                            "description": "OlmoEarth encoder size, if the user has a preference.",
                        },
                    },
                    "required": [],
                },
            ),
            handler=_recommend,
        ),
        RegisteredTool(
            spec=ToolSpec(
                name="olmoearth_rslearn_validate",
                description=(
                    "Validate an rslearn dataset config.json and/or model.yaml BEFORE "
                    "training, catching the errors rslearn only surfaces hours into a "
                    "run: encoder embedding-dim vs decoder in_channels, decoder "
                    "out_channels vs num_classes, task vs label-type (SegmentationTask "
                    "needs a raster target, DetectionTask needs vector), model "
                    "inputs.layers/bands that don't exist in the dataset, bad dtype "
                    "values, data_source class_path / sort_by typos, and the Faster "
                    "R-CNN background-class +1 quirk. Use when the user has (or you "
                    "just assembled) an rslearn config and is about to run "
                    "add_windows/prepare/fit. Pass the PARSED objects. Returns "
                    "{ok, errors, warnings, checks}; never runs training."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "dataset_config": {
                            "type": "object",
                            "description": "The parsed rslearn dataset config.json (layers -> band_sets/data_source).",
                        },
                        "model_config": {
                            "type": "object",
                            "description": "The parsed rslearn model.yaml (model.init_args.encoder/decoder/task/inputs).",
                        },
                    },
                    "required": [],
                },
            ),
            handler=_validate,
        ),
        RegisteredTool(
            spec=ToolSpec(
                name="olmoearth_rslearn_compose",
                description=(
                    "Compose a complete, VALID OlmoEarth/rslearn finetune `model.yaml` "
                    "for the user. Use after `olmoearth_rslearn_recommend`, or when the "
                    "user says 'write the config', 'give me the YAML', or 'set up the "
                    "fine-tune'. Mirrors the proven OlmoEarth template (MultiTaskModel "
                    "+ OlmoEarth encoder -> Upsample/Conv or PoolingDecoder -> task head "
                    "+ FreezeUnfreeze), with the right decoder/head + channel counts for "
                    "the task. Returns the YAML string + the config dict. Supports "
                    "segmentation / per-pixel-regression / classification / regression; "
                    "detection is guided (Faster R-CNN anchors need tuning), not "
                    "auto-emitted. Then run olmoearth_rslearn_validate on it."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "task": {"type": "string", "enum": list(TASKS), "description": "rslearn task (or pass `goal` to infer)."},
                        "goal": {"type": "string", "description": "Plain-language goal, if task isn't given."},
                        "model_size": {"type": "string", "enum": list(EMBEDDING_SIZES), "description": "OlmoEarth encoder size (default base)."},
                        "num_classes": {"type": "integer", "description": "Class count (segmentation/classification)."},
                        "class_names": {"type": "array", "items": {"type": "string"}, "description": "Class names (classification)."},
                        "property_name": {"type": "string", "description": "Vector property holding the label (classification/regression; default 'category')."},
                        "scale_factor": {"type": "number", "description": "Label scale factor (regression tasks)."},
                        "dataset_path": {"type": "string", "description": "Path to the rslearn dataset dir (default data/dataset)."},
                        "bands": {"type": "array", "items": {"type": "string"}, "description": "Input bands (default the 12-band S2 L2A stack)."},
                        "freeze_epochs": {"type": "integer", "description": "Epochs with the encoder frozen before unfreezing (default 10)."},
                        "total_epochs": {"type": "integer", "description": "Total training epochs (default 40)."},
                        "nodata_value": {"type": "integer", "description": "NODATA value to mask out of the loss (raster tasks)."},
                    },
                    "required": [],
                },
            ),
            handler=_compose,
        ),
        RegisteredTool(
            spec=ToolSpec(
                name="olmoearth_rslearn_diagnose",
                description=(
                    "Diagnose a FAILING rslearn run (the `prepare` / `ingest` / "
                    "`materialize` stages) and return plain-English fixes. Use when a "
                    "scientist pastes an rslearn error or a stage summary and asks "
                    "'why did this fail' / 'no scenes found' / 'why are there 0 windows'. "
                    "Pass the parsed stage `summary` (with per-layer windows_prepared/"
                    "rejected/failed + error_messages) and/or the raw `log_text`. "
                    "Detects: no scenes / 0 windows, over-rejection (min_matches/cloud), "
                    "CRS mismatch, out-of-memory, disk, and auth errors. Inspection only."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "summary": {"type": "object", "description": "Parsed rslearn stage summary (per-layer counts + error_messages)."},
                        "log_text": {"type": "string", "description": "Raw rslearn CLI output / error text."},
                    },
                    "required": [],
                },
            ),
            handler=_diagnose,
        ),
    ]
