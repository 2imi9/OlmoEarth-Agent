# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""rslearn config advisor + validator (the in-repo facet of the olmoearth-rslearn skill).

Two torch-free functions that let a domain scientist who does NOT know rslearn
still set up a correct OlmoEarth/rslearn experiment:

* :func:`recommend` — a plain-language research goal (+ what's known about the
  labels / sensor / cloudiness / time scope) -> a complete, *explained* setup:
  the rslearn task, the data layout (data_source + query_config + compositing +
  bands), the model composition (encoder -> decoder -> head), the task knobs
  (metrics / loss / nodata / scale_factor), and a fine-tune schedule. Every
  choice carries a one-line *why*.
* :func:`validate_config` — a dataset ``config.json`` + a model ``model.yaml``
  (parsed to dicts) -> the errors rslearn only surfaces hours into a training
  run: encoder embedding-dim <-> decoder ``in_channels`` <-> ``out_channels`` =
  ``num_classes``; task <-> label-type (Segmentation needs raster, Detection
  needs vector); ``inputs.layers`` resolve to dataset layers; bands exist;
  ``data_source`` class_path + ``sort_by`` plausibility; the Faster R-CNN
  background-class +1 quirk.

**Torch-free by construction.** This module never imports ``rslearn`` (which
pulls torch and would break the agent's torch-free guarantee). Instead it
mirrors rslearn's facts as plain data, verified against the rslearn repo
(``docs/TasksAndModels.md``, ``rslearn/config/dataset.py``,
``rslearn/models/olmoearth_pretrain/model.py``) on 2026-06-08. rslearn evolves:
if it later rejects a value this module emits, re-check the repo and update the
tables here. Pure metadata in/out (no geometry, no network) -> provenance-safe.
"""

from __future__ import annotations

import re
from typing import Any

# --- rslearn facts, mirrored as data (see module docstring for provenance) ---

#: OlmoEarth encoder output embedding size per model size
#: (rslearn/models/olmoearth_pretrain/model.py: EMBEDDING_SIZES).
EMBEDDING_SIZES: dict[str, int] = {"nano": 128, "tiny": 192, "base": 768, "large": 1024}

#: Dataset query knobs (rslearn/config/dataset.py).
SPACE_MODES = ["CONTAINS", "INTERSECTS", "MOSAIC", "PER_PERIOD_MOSAIC", "SINGLE_COMPOSITE"]
TIME_MODES = ["WITHIN", "NEAREST", "BEFORE", "AFTER"]
COMPOSITING_METHODS = [
    "FIRST_VALID", "MEAN", "MEDIAN", "SPATIAL_MOSAIC_TEMPORAL_STACK",
    "TEMPORAL_MEAN", "TEMPORAL_MAX", "TEMPORAL_MIN",
]
DTYPES = ["uint8", "uint16", "uint32", "int32", "float32"]
LAYER_TYPES = ["raster", "vector"]

#: A representative slice of rslearn data_sources (≈40 exist); class_paths the
#: olmoearth-rslearn SKILL.md already documents.
DATA_SOURCES: dict[str, str] = {
    "sentinel2": "rslearn.data_sources.gcp_public_data.Sentinel2",
    "sentinel2_pc": "rslearn.data_sources.planetary_computer.Sentinel2",
    "sentinel1_pc": "rslearn.data_sources.planetary_computer.Sentinel1",
    "landsat_pc": "rslearn.data_sources.planetary_computer.Landsat",
    "naip_pc": "rslearn.data_sources.planetary_computer.NAIP",
    "local": "rslearn.data_sources.local_files.LocalFiles",
}
#: Metadata keys a data source is commonly sorted by (catches sort_by typos).
SORT_BY_KEYS = {"cloud_cover", "eo:cloud_cover", "properties.eo:cloud_cover"}

#: The five supervised tasks, each with the rslearn class, the label geometry it
#: needs, the encoder feature type it consumes, the recommended decoder + head,
#: and the per-task metric/loss knobs. (docs/TasksAndModels.md.)
TASKS: dict[str, dict[str, Any]] = {
    "segmentation": {
        "task_class": "rslearn.train.tasks.segmentation.SegmentationTask",
        "label_type": "raster",
        "feature": "FeatureMaps",
        "decoder": ["rslearn.models.unet.UNetDecoder"],
        "head": "rslearn.train.tasks.segmentation.SegmentationHead",
        "metrics": ["accuracy", "miou"],
        "needs_num_classes": True,
        "what": "classify every pixel into one of N classes (e.g. land-cover map).",
    },
    "per_pixel_regression": {
        "task_class": "rslearn.train.tasks.per_pixel_regression.PerPixelRegressionTask",
        "label_type": "raster",
        "feature": "FeatureMaps",
        "decoder": ["rslearn.models.unet.UNetDecoder"],
        "head": "rslearn.train.tasks.per_pixel_regression.PerPixelRegressionHead",
        "metrics": ["mse", "r2"],
        "needs_num_classes": False,
        "what": "predict a continuous value at every pixel (e.g. biomass, moisture).",
    },
    "classification": {
        "task_class": "rslearn.train.tasks.classification.ClassificationTask",
        "label_type": "vector",
        "feature": "FeatureVector",
        "decoder": ["rslearn.models.pooling_decoder.PoolingDecoder"],
        "head": "rslearn.train.tasks.classification.ClassificationHead",
        "metrics": ["accuracy", "f1"],
        "needs_num_classes": True,
        "what": "one label for the whole image/window (e.g. vessel type).",
    },
    "regression": {
        "task_class": "rslearn.train.tasks.regression.RegressionTask",
        "label_type": "vector",
        "feature": "FeatureVector",
        "decoder": ["rslearn.models.pooling_decoder.PoolingDecoder"],
        "head": "rslearn.train.tasks.regression.RegressionHead",
        "metrics": ["mse"],
        "needs_num_classes": False,
        "what": "one continuous value per image/window (e.g. vessel length).",
    },
    "detection": {
        "task_class": "rslearn.train.tasks.detection.DetectionTask",
        "label_type": "vector",
        "feature": "FeatureMaps",
        # Faster R-CNN IS the predictor (no separate head); FPN feeds it.
        "decoder": ["rslearn.models.fpn.Fpn", "rslearn.models.faster_rcnn.FasterRCNN"],
        "head": None,
        "metrics": ["mAP", "f1"],
        "needs_num_classes": True,
        "reserves_background": True,  # class 0 is background -> num_classes = real + 1
        "what": "draw bounding boxes with categories (e.g. count wind turbines).",
    },
}

#: Keyword -> task routing for the plain-language path.
_TASK_KEYWORDS: list[tuple[str, list[str]]] = [
    ("detection", ["detect", "bounding box", "count ", "locate", "find objects", "wind turbine", "vessel", "ship"]),
    ("per_pixel_regression", ["per-pixel regress", "per pixel regress", "continuous map", "biomass", "moisture", "height map", "canopy", "value at each pixel", "regress.*pixel"]),
    ("segmentation", ["segment", "land cover", "land-cover", "per-pixel class", "classify each pixel", "mask", "flood map", "crop type", "burn"]),
    ("regression", ["regress", "predict the length", "single value", "window-level value", "estimate the"]),
    ("classification", ["classify", "classification", "scene", "whole image", "window-level", "label the image", "what type"]),
]


def _route_task(goal: str) -> str | None:
    """Best-effort plain-language goal -> rslearn task key."""
    g = (goal or "").lower()
    for task, kws in _TASK_KEYWORDS:
        for kw in kws:
            if (".*" in kw and re.search(kw, g)) or (".*" not in kw and kw in g):
                return task
    return None


def _encoder_block(model_size: str) -> dict[str, Any]:
    dim = EMBEDDING_SIZES[model_size]
    return {
        "class_path": "rslearn.models.olmoearth_pretrain.model.OlmoEarth",
        "model_size": model_size,
        "embedding_dim": dim,
        "why": f"OlmoEarth {model_size.capitalize()} encoder emits {dim}-channel features; "
        "the decoder below must consume that channel count.",
    }


def _compositing_advice(cloudy: bool | None, temporal: str | None) -> dict[str, str]:
    """Pick a compositing method from cloud prevalence + temporal scope."""
    if temporal in {"seasonal", "annual", "multi-month"}:
        return {"method": "TEMPORAL_MEAN", "why": "averaging over the season smooths noise and gaps for a stable multi-month signal."}
    if cloudy:
        return {"method": "FIRST_VALID", "why": "FIRST_VALID keeps the least-cloudy valid pixel per window (pair with sort_by=cloud_cover); add a cloud-aware compositor if clouds dominate."}
    return {"method": "FIRST_VALID", "why": "the safe default: the first valid (least-cloudy) scene per window."}


def recommend(
    *,
    goal: str | None = None,
    task: str | None = None,
    label_kind: str | None = None,
    num_classes: int | None = None,
    label_range: list[float] | None = None,
    sensor: str | None = None,
    cloudy: bool | None = None,
    temporal: str | None = None,
    num_samples: int | None = None,
    model_size: str | None = None,
) -> dict[str, Any]:
    """Recommend a complete, explained rslearn setup from a research goal.

    All inputs are optional; whatever is missing is listed in ``ask_for``. Pure
    metadata logic — no network, no torch.
    """
    chosen = (task or "").strip().lower() or _route_task(goal or "")
    ask_for: list[str] = []
    if chosen not in TASKS:
        return {
            "task": None,
            "ask_for": ["a clearer goal or an explicit task: " + ", ".join(TASKS)],
            "note": "Couldn't infer the rslearn task from the goal. "
            "Say what the model should output: a per-pixel class map (segmentation), "
            "a per-pixel value (per_pixel_regression), bounding boxes (detection), "
            "or one label/value per image (classification / regression).",
            "tasks": {k: v["what"] for k, v in TASKS.items()},
        }

    t = TASKS[chosen]
    size = (model_size or "").strip().lower()
    if size not in EMBEDDING_SIZES:
        # Small data -> smaller encoder is plenty; large data -> base.
        size = "base" if (num_samples or 0) >= 2000 else "tiny"
    enc = _encoder_block(size)

    # Decoder/head, with the shape contract spelled out.
    out_channels = None
    if t["needs_num_classes"] and num_classes:
        out_channels = (num_classes + 1) if t.get("reserves_background") else num_classes
    model = {
        "framework": "SingleTaskModel",
        "encoder": enc,
        "decoder": t["decoder"],
        "head": t["head"],
        "feature_type": t["feature"],
        "out_channels": out_channels,
        "why": (
            f"{chosen} consumes {t['feature']} from the encoder, so use "
            f"{' -> '.join(p.rsplit('.', 1)[-1] for p in t['decoder'])}"
            + (f" -> {t['head'].rsplit('.', 1)[-1]}" if t["head"] else " (Faster R-CNN is the predictor)")
            + (
                f". Set the decoder's out_channels to {out_channels}"
                + (" (num_classes + 1: Faster R-CNN reserves class 0 for background)" if t.get("reserves_background") else " (= num_classes)")
                + "."
                if out_channels is not None
                else "."
            )
        ),
    }
    if t["needs_num_classes"] and not num_classes:
        ask_for.append("num_classes")

    # Task knobs.
    task_cfg: dict[str, Any] = {"class_path": t["task_class"], "metrics": t["metrics"]}
    notes: list[str] = []
    if chosen in {"regression", "per_pixel_regression"}:
        if label_range and len(label_range) == 2 and label_range[1] > 0:
            hi = float(label_range[1])
            sf = round(1.0 / hi, 6) if hi > 1.5 else 1.0
            task_cfg["scale_factor"] = sf
            notes.append(
                f"scale_factor={sf}: your labels span ~[{label_range[0]}, {label_range[1]}]; "
                "training on a ~[0,1] range is more stable (rslearn unscales the output)."
            )
        else:
            ask_for.append("label_range (min,max of the value you're predicting)")
    if t["label_type"] == "raster":
        task_cfg["nodata_value"] = None
        notes.append(
            "If your raster labels have missing/NODATA pixels, set nodata_value so "
            "they're masked out of the loss — otherwise the model trains on garbage there."
        )
    if chosen == "detection":
        notes.append("box_size is required for Point-geometry labels (a fixed box per point).")

    # Data layout.
    src_key = "sentinel2_pc"
    if sensor:
        s = sensor.lower()
        if "1" in s or "sar" in s:
            src_key = "sentinel1_pc"
        elif "landsat" in s:
            src_key = "landsat_pc"
        elif "naip" in s:
            src_key = "naip_pc"
    comp = _compositing_advice(cloudy, temporal)
    data = {
        "data_source": DATA_SOURCES[src_key],
        "sort_by": "cloud_cover",
        "space_mode": "MOSAIC" if temporal in {None, "single"} else "PER_PERIOD_MOSAIC",
        "compositing_method": comp["method"],
        "label_layer_type": t["label_type"],
        "why": f"{comp['why']} Labels are a '{t['label_type']}' layer because {chosen} "
        + ("reads class IDs / values from a raster." if t["label_type"] == "raster" else "reads category/value from vector features."),
    }

    # Fine-tune schedule.
    fine_tune = {
        "strategy": "MultiStageFineTuning",
        "stages": [
            {"freeze_encoder": True, "epochs": 10, "why": "stage 1: train the head on frozen OlmoEarth features (fast, stable)."},
            {"freeze_encoder": False, "unfreeze_lr_factor": 10, "epochs": 20, "why": "stage 2: unfreeze with a smaller encoder LR to adapt without forgetting."},
        ] if (num_samples or 0) >= 200 else [
            {"freeze_encoder": True, "epochs": 20, "why": "few labels: keep the encoder frozen and only train the head to avoid overfitting (consider the embeddings path via olmoearth-embeddings)."},
        ],
    }

    return {
        "task": chosen,
        "what_it_does": t["what"],
        "model": model,
        "task_config": task_cfg,
        "data": data,
        "fine_tune": fine_tune,
        "notes": notes,
        "ask_for": ask_for,
        "next": "Pass the assembled config.json + model.yaml to olmoearth_rslearn_validate before training, "
        "then run it with the olmoearth-rslearn skill (add_windows -> prepare -> ingest -> materialize -> model fit).",
    }


# --------------------------------------------------------------------------- #
# Validator
# --------------------------------------------------------------------------- #

_TASK_LABEL_TYPE = {
    "SegmentationTask": "raster",
    "PerPixelRegressionTask": "raster",
    "ClassificationTask": "vector",
    "RegressionTask": "vector",
    "DetectionTask": "vector",
}


def _last(path: str) -> str:
    return (path or "").rsplit(".", 1)[-1]


def _embedding_dim_for(encoder_args: dict[str, Any]) -> int | None:
    """Best-effort encoder output channels from a model_id/model_size hint."""
    blob = " ".join(str(v) for v in encoder_args.values()).lower()
    for size, dim in EMBEDDING_SIZES.items():
        if size in blob:
            return dim
    return None


def validate_config(
    dataset_config: dict[str, Any] | None = None,
    model_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Catch the rslearn config errors that otherwise fail hours into training.

    Both args are the parsed JSON/YAML dicts (no rslearn import). Returns
    ``{ok, errors, warnings, checks}`` — ``ok`` is False iff there are errors.
    """
    errors: list[str] = []
    warnings: list[str] = []
    checks: list[str] = []
    ds = dataset_config or {}
    mc = model_config or {}

    # ---- dataset layers: dtype + data_source + sort_by ----
    layers = (ds.get("layers") or {}) if isinstance(ds, dict) else {}
    layer_bands: dict[str, set[str]] = {}
    if layers:
        checks.append("dataset layers")
        for lname, lcfg in layers.items():
            if not isinstance(lcfg, dict):
                continue
            ltype = lcfg.get("type")
            if ltype and ltype not in LAYER_TYPES:
                errors.append(f"layer '{lname}': type {ltype!r} is not one of {LAYER_TYPES}.")
            bands: set[str] = set()
            for bs in lcfg.get("band_sets") or []:
                if isinstance(bs, dict):
                    dt = bs.get("dtype")
                    if dt and dt not in DTYPES:
                        errors.append(f"layer '{lname}' band_set dtype {dt!r} is not one of {DTYPES}.")
                    for b in bs.get("bands") or []:
                        bands.add(str(b))
            layer_bands[lname] = bands
            src = (lcfg.get("data_source") or {}) if isinstance(lcfg.get("data_source"), dict) else {}
            cp = src.get("class_path")
            if cp and not cp.startswith("rslearn.data_sources."):
                warnings.append(f"layer '{lname}' data_source class_path {cp!r} doesn't look like an rslearn.data_sources.* path.")
            init = (src.get("init_args") or {}) if isinstance(src.get("init_args"), dict) else {}
            sb = init.get("sort_by")
            if sb and sb not in SORT_BY_KEYS:
                warnings.append(f"layer '{lname}' sort_by={sb!r} is unusual (typo? expected one of {sorted(SORT_BY_KEYS)}); a bad key silently fails at ingest.")

    # ---- model: framework / encoder->decoder->head shapes ----
    model = (mc.get("model") or {}) if isinstance(mc, dict) else {}
    init = (model.get("init_args") or {}) if isinstance(model.get("init_args"), dict) else {}
    encoder = init.get("encoder") or []
    decoder = init.get("decoder") or []
    enc_dim: int | None = None
    if encoder:
        checks.append("encoder->decoder channel match")
        first = encoder[0] if isinstance(encoder[0], dict) else {}
        enc_dim = _embedding_dim_for(first.get("init_args") or {})
        # find a decoder component declaring in_channels
        for comp in decoder:
            if not isinstance(comp, dict):
                continue
            cargs = comp.get("init_args") or {}
            in_ch = cargs.get("in_channels")
            if isinstance(in_ch, int) and enc_dim is not None and in_ch != enc_dim:
                errors.append(
                    f"{_last(comp.get('class_path', '?'))}.in_channels={in_ch} but the OlmoEarth encoder "
                    f"emits {enc_dim} channels — they must match or rslearn errors on the first batch."
                )

    # ---- task <-> label type + out_channels = num_classes ----
    task = init.get("task") or mc.get("task") or {}
    if isinstance(task, dict) and task.get("class_path"):
        checks.append("task <-> label-type + num_classes")
        tname = _last(task["class_path"])
        targs = task.get("init_args") or {}
        want_label = _TASK_LABEL_TYPE.get(tname)
        # target layers in the dataset
        target_types = {
            (lc.get("type"))
            for lc in layers.values()
            if isinstance(lc, dict) and lc.get("is_target")
        }
        if want_label and target_types and want_label not in target_types:
            errors.append(
                f"{tname} needs a {want_label} target layer, but the target layer(s) are {sorted(t for t in target_types if t)}. "
                + ("Segmentation/per-pixel tasks read a raster label." if want_label == "raster" else "Classification/detection/regression read vector labels.")
            )
        num_classes = targs.get("num_classes") or (len(targs["classes"]) if isinstance(targs.get("classes"), list) else None)
        # out_channels of the decoder before the head must equal num_classes
        if num_classes:
            for comp in decoder:
                cargs = comp.get("init_args") or {} if isinstance(comp, dict) else {}
                oc = cargs.get("out_channels")
                if isinstance(oc, int):
                    expected = num_classes
                    if tname == "DetectionTask":
                        if oc <= num_classes:
                            warnings.append(
                                f"DetectionTask: decoder out/num_classes should reserve class 0 for background "
                                f"(num_classes = real classes + 1); saw {oc} vs {num_classes}."
                            )
                    elif oc != expected:
                        errors.append(
                            f"{_last(comp.get('class_path', '?'))}.out_channels={oc} but the task has {num_classes} classes — "
                            "they must match (the layer before the classification/segmentation head sizes the logits)."
                        )

    # ---- inputs.layers / bands resolve into the dataset ----
    inputs = init.get("inputs") or {}
    if inputs and layer_bands:
        checks.append("model inputs resolve to dataset layers")
        for iname, icfg in inputs.items():
            if not isinstance(icfg, dict):
                continue
            for lyr in icfg.get("layers") or []:
                if lyr not in layer_bands:
                    errors.append(f"model input '{iname}' references layer '{lyr}' that is not in the dataset config.")
                else:
                    errors.extend(
                        f"model input '{iname}' band '{b}' does not exist in dataset layer '{lyr}' "
                        f"(has: {sorted(layer_bands[lyr])})."
                        for b in icfg.get("bands") or []
                        if layer_bands[lyr] and str(b) not in layer_bands[lyr]
                    )

    if not checks:
        warnings.append("Nothing to validate: pass dataset_config (config.json) and/or model_config (model.yaml) as parsed objects.")

    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "checks": checks,
        "summary": (
            "No blocking errors found — still review the warnings."
            if not errors
            else f"{len(errors)} blocking error(s) would fail training; fix before running rslearn."
        ),
    }
