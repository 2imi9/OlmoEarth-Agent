# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""rslearn config advisor + validator (the in-repo facet of the olmoearth-rslearn skill).

Torch-free functions that let a domain scientist who does NOT know rslearn
still set up a correct OlmoEarth/rslearn experiment:

* :func:`recommend` — a plain-language research goal (+ what's known about the
  labels / sensor / cloudiness / time scope) -> a complete, *explained* setup:
  the rslearn task, the data layout (data_source + query_config + compositing +
  bands), the model composition (encoder -> decoder -> head), the task knobs
  (metrics / loss / nodata / scale_factor), and a fine-tune schedule. Every
  choice carries a one-line *why*. With 2+ ``modalities`` it also returns a
  :func:`recommend_fusion` pre/mid/post strategy.
* :func:`validate_config` — a dataset ``config.json`` + a model ``model.yaml``
  (parsed to dicts) -> the errors rslearn only surfaces hours into a training
  run: encoder embedding-dim <-> decoder ``in_channels`` <-> ``out_channels`` =
  ``num_classes``; task <-> label-type (Segmentation needs raster, Detection
  needs vector); ``inputs.layers`` resolve to dataset layers; bands exist;
  ``data_source`` class_path + ``sort_by`` plausibility; the Faster R-CNN
  background-class +1 quirk.
* :func:`compose` — emit a complete, valid finetune ``model.yaml`` (single-source,
  or multi-source fusion via ``modalities`` + ``fusion``).
* :func:`diagnose` — turn a failing ``prepare`` / ``ingest`` / ``materialize``
  run (a stage summary and/or log text) into plain-English fixes.

**Torch-free by construction.** This module never imports ``rslearn`` (which
pulls torch and would break the agent's torch-free guarantee). Instead it
mirrors rslearn's facts as plain data, verified against the rslearn repo
(``docs/TasksAndModels.md``, ``rslearn/config/dataset.py``,
``rslearn/models/olmoearth_pretrain/model.py``) on 2026-06-08. rslearn evolves:
if it later rejects a value this module emits, re-check the repo and update the
tables here. Pure metadata in/out (no geometry, no network) -> provenance-safe.
"""

from __future__ import annotations

import json
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
        "decoder": ["rslearn.models.upsample.Upsample", "rslearn.models.conv.Conv"],
        "head": "rslearn.train.tasks.segmentation.SegmentationHead",
        "metrics": ["accuracy", "miou"],
        "needs_num_classes": True,
        "what": "classify every pixel into one of N classes (e.g. land-cover map).",
    },
    "per_pixel_regression": {
        "task_class": "rslearn.train.tasks.per_pixel_regression.PerPixelRegressionTask",
        "label_type": "raster",
        "feature": "FeatureMaps",
        "decoder": ["rslearn.models.upsample.Upsample", "rslearn.models.conv.Conv"],
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
    modalities: list[str] | None = None,
) -> dict[str, Any]:
    """Recommend a complete, explained rslearn setup from a research goal.

    All inputs are optional; whatever is missing is listed in ``ask_for``. Pass
    ``modalities`` (2+) to also get a multi-source **fusion** recommendation
    (pre / mid / post) under the ``fusion`` key. Pure metadata logic — no network,
    no torch.
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

    fusion = None
    canon, _ = _normalize_modalities(modalities)
    if len(canon) >= 2:
        fusion = recommend_fusion(canon, task=chosen)

    return {
        "task": chosen,
        "what_it_does": t["what"],
        "model": model,
        "task_config": task_cfg,
        "data": data,
        "fine_tune": fine_tune,
        "fusion": fusion,
        "notes": notes,
        "ask_for": ask_for,
        "next": "Pass the assembled config.json + model.yaml to olmoearth_rslearn_validate before training, "
        "then run it with the olmoearth-rslearn skill (add_windows -> prepare -> ingest -> materialize -> model fit)."
        + (" For multi-source fusion, compose with olmoearth_rslearn_compose(modalities=..., fusion=...)." if fusion else ""),
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


# --------------------------------------------------------------------------- #
# Composer (emit a full, valid OlmoEarth finetune model.yaml)
# --------------------------------------------------------------------------- #

#: model_size -> rslearn ModelID (rslearn/models/olmoearth_pretrain/model.py).
_MODEL_ID = {
    "nano": "OLMOEARTH_V1_NANO", "tiny": "OLMOEARTH_V1_TINY",
    "base": "OLMOEARTH_V1_BASE", "large": "OLMOEARTH_V1_LARGE",
}
#: Default Sentinel-2 L2A 12-band stack (matches olmoearth-data-prep's template).
_S2_BANDS = ["B02", "B03", "B04", "B08", "B05", "B06", "B07", "B8A", "B11", "B12", "B01", "B09"]
_DECODER_KEY = {"segmentation": "segment", "per_pixel_regression": "regress",
                "classification": "classify", "regression": "regress"}

# --------------------------------------------------------------------------- #
# Multi-source fusion (pre / mid / post), grounded in rslearn's multi-input API
# --------------------------------------------------------------------------- #

#: Each modality -> the input KEY the OlmoEarth encoder reads it under, the
#: dataset layer's typical bands, a data_source class_path, and whether OlmoEarth
#: fuses it natively. ``native`` modalities are exactly OlmoEarth.MODALITY_NAMES
#: (rslearn/models/olmoearth_pretrain/model.py) -- the shared encoder iterates
#: those names over whatever inputs are present and fuses them internally via
#: cross-modal attention. A non-native modality (e.g. DEM) is NOT in MODALITY_NAMES,
#: so it can't ride the shared encoder; it needs a separate encoder path
#: (CrossAttentionFusionExtractor). Bands verified vs the rslearn clone 2026-06-09
#: (Sentinel-1 ``vv,vh``; WorldCover single ``B1``); landsat/OSM bands are
#: defaults to confirm against your own dataset config.
FUSION_MODALITIES: dict[str, dict[str, Any]] = {
    "sentinel2_l2a": {
        "bands": _S2_BANDS, "native": True,
        "data_source": "rslearn.data_sources.planetary_computer.Sentinel2",
        "what": "optical, 12-band Sentinel-2 L2A",
    },
    "sentinel1": {
        "bands": ["vv", "vh"], "native": True,
        "data_source": "rslearn.data_sources.planetary_computer.Sentinel1",
        "what": "SAR backscatter (sees through cloud and at night)",
    },
    "worldcover": {
        "bands": ["B1"], "native": True,
        "data_source": "rslearn.data_sources.worldcover.WorldCover",
        "what": "ESA WorldCover 10 m land-cover prior (static)",
    },
    "openstreetmap_raster": {
        "bands": ["B1"], "native": True,
        "data_source": "rslearn.data_sources.openstreetmap.OpenStreetMap",
        "what": "rasterized OpenStreetMap features (roads / buildings)",
    },
    "landsat": {
        "bands": ["B2", "B3", "B4", "B5", "B6", "B7"], "native": True,
        "data_source": "rslearn.data_sources.planetary_computer.LandsatC2L2",
        "what": "optical, Landsat (longer historical archive than S2)",
    },
    # Not an OlmoEarth pretrain modality -> needs its own encoder path.
    "dem": {
        "bands": ["B1"], "native": False,
        "data_source": "rslearn.data_sources.planetary_computer.CopDemGlo30",
        "what": "Copernicus DEM elevation (static; not an OlmoEarth pretrain modality)",
    },
}

#: Common aliases -> canonical modality key.
_MODALITY_ALIASES = {
    "s2": "sentinel2_l2a", "sentinel2": "sentinel2_l2a", "sentinel-2": "sentinel2_l2a",
    "sentinel2_l2a": "sentinel2_l2a", "optical": "sentinel2_l2a",
    "s1": "sentinel1", "sentinel-1": "sentinel1", "sar": "sentinel1", "sentinel1": "sentinel1",
    "worldcover": "worldcover", "landcover": "worldcover", "land_cover": "worldcover",
    "osm": "openstreetmap_raster", "openstreetmap": "openstreetmap_raster",
    "openstreetmap_raster": "openstreetmap_raster",
    "landsat": "landsat", "dem": "dem", "elevation": "dem", "srtm": "dem",
}

#: The OlmoEarth-native modalities (the shared encoder fuses these internally).
_NATIVE_MODALITIES = [m for m, v in FUSION_MODALITIES.items() if v["native"]]


def _normalize_modalities(
    modalities: list[str] | None,
) -> tuple[list[str], list[str]]:
    """Map modality aliases to canonical keys; return ``(known, unknown)`` (deduped)."""
    known: list[str] = []
    unknown: list[str] = []
    for m in modalities or []:
        canon = _MODALITY_ALIASES.get(str(m).strip().lower())
        if canon and canon not in known:
            known.append(canon)
        elif not canon and str(m).strip():
            unknown.append(str(m))
    return known, unknown


def recommend_fusion(
    modalities: list[str] | None,
    *,
    task: str | None = None,  # noqa: ARG001 - reserved for future task-aware advice
    robust_to_missing: bool = False,
) -> dict[str, Any]:
    """Recommend a multi-source fusion strategy (pre / mid / post) for a modality set.

    Grounded in rslearn's actual multi-input + fusion API:

    * **mid** (feature-level; the RECOMMENDED default) — one OlmoEarth encoder fed
      several modality inputs. The pretrained encoder fuses them internally via
      cross-modal attention (it iterates ``OlmoEarth.MODALITY_NAMES`` over whatever
      inputs are present). No extra fusion module to tune. Valid only when every
      modality is an OlmoEarth pretrain modality.
    * **cross_attention** (also mid/feature-level, separate paths) — distinct
      encoder paths fused by ``rslearn.models.xatt_fusion.CrossAttentionFusionExtractor``
      (a primary/query stream + context streams, with learned memory tokens and a
      ``context_dropout_prob`` for missing-context robustness). Use when a modality
      is NOT an OlmoEarth pretrain modality (e.g. DEM), or differs sharply in
      resolution/semantics, or you want an explicit primary/context relationship.
    * **pre** (input concatenation) — stack co-registered bands into a single input,
      one encoder over the stack. Simplest, but with OlmoEarth you forfeit the
      modality-specific pretrained embeddings, so it only makes sense for same-grid
      band *extensions*, not genuinely different sensors.
    * **post** (decision fusion / ensemble) — train one model per modality and
      ensemble their predictions. Most robust to a modality being missing at
      inference (a dropped modality just removes one vote).

    Pure metadata; no torch, no network.
    """
    canon, unknown = _normalize_modalities(modalities)
    if len(canon) < 2:
        return {
            "ok": False,
            "modalities": canon,
            "unknown_modalities": unknown,
            "note": "Fusion needs 2+ modalities. With one, call olmoearth_rslearn_compose "
            "directly. Known modalities: " + ", ".join(FUSION_MODALITIES),
        }

    non_native = [m for m in canon if not FUSION_MODALITIES[m]["native"]]
    primary = "sentinel2_l2a" if "sentinel2_l2a" in canon else canon[0]

    if robust_to_missing:
        strategy, why = "post", (
            "A modality may be missing at inference, so decision-level fusion is the "
            "most robust: train one model per modality and ensemble the predictions — "
            "a dropped modality just removes one vote instead of breaking the forward pass."
        )
    elif non_native:
        strategy, why = "cross_attention", (
            f"{', '.join(non_native)} is not an OlmoEarth pretrain modality "
            f"(MODALITY_NAMES = {', '.join(_NATIVE_MODALITIES)}), so it can't ride the "
            f"shared encoder. Use a CrossAttentionFusionExtractor with {primary} as the "
            "primary/query path and the other modality as a context path."
        )
    else:
        strategy, why = "mid", (
            "Every chosen modality is an OlmoEarth pretrain modality, so the simplest and "
            "strongest option is one OlmoEarth encoder fed all of them — it was pretrained "
            "to fuse them internally via cross-modal attention, with no extra fusion module "
            "to tune. compose() can emit this multi-input config directly."
        )

    alternatives = {
        "mid": "One OlmoEarth encoder, multiple modality inputs (internal cross-modal fusion).",
        "cross_attention": "Separate encoder paths fused by CrossAttentionFusionExtractor "
        "(primary + context); for non-native modalities or an explicit primary/context split.",
        "pre": "Concatenate co-registered bands at the input; only for same-grid band extensions.",
        "post": "One model per modality, ensemble predictions; most robust to a missing modality.",
    }
    notes = [
        f"primary modality: {primary} "
        + ("(Sentinel-2 is the natural query stream)" if primary == "sentinel2_l2a" else ""),
    ]
    if any(
        "temporal" in str(task or "").lower() or m in {"sentinel1", "sentinel2_l2a"}
        for m in canon
    ):
        notes.append(
            "If a modality is a time series, wrap its encoder in "
            "rslearn.models.simple_time_series.SimpleTimeSeries (temporal fusion is "
            "orthogonal to multi-modal fusion)."
        )
    if unknown:
        notes.append(f"unrecognized modalities ignored: {unknown}")

    return {
        "ok": True,
        "modalities": canon,
        "strategy": strategy,
        "why": why,
        "primary": primary,
        "alternatives": alternatives,
        "unknown_modalities": unknown,
        "notes": notes,
        "compose_hint": f"olmoearth_rslearn_compose(task=..., modalities={canon}, fusion='{strategy}')",
    }


def _modality_inputs(
    modalities: list[str], s2_bands: list[str] | None = None
) -> dict[str, dict[str, Any]]:
    """The model-input block for a set of native modalities (one entry each).

    Each modality is keyed by its OlmoEarth modality name and points at a dataset
    layer of the same short name; ``s2_bands`` overrides the Sentinel-2 band list.
    """
    inputs: dict[str, dict[str, Any]] = {}
    layer_name = {"sentinel2_l2a": "sentinel2", "sentinel1": "sentinel1",
                  "worldcover": "worldcover", "openstreetmap_raster": "openstreetmap",
                  "landsat": "landsat", "dem": "dem"}
    for m in modalities:
        spec = FUSION_MODALITIES[m]
        bands = (s2_bands or spec["bands"]) if m == "sentinel2_l2a" else spec["bands"]
        inputs[m] = {
            "data_type": "raster",
            "layers": [layer_name[m]],
            "bands": list(bands),
            "passthrough": True,
            "dtype": "FLOAT32",
        }
    return inputs


#: Fusion-strategy aliases -> canonical {mid, cross_attention, pre, post}.
_FUSION_ALIASES = {
    "mid": "mid", "shared": "mid", "shared_encoder": "mid", "feature": "mid",
    "cross_attention": "cross_attention", "xatt": "cross_attention",
    "cross-attention": "cross_attention", "late": "cross_attention",
    "pre": "pre", "early": "pre", "concat": "pre", "input": "pre",
    "post": "post", "ensemble": "post", "decision": "post",
}


def _normalize_fusion(fusion: str | None) -> str | None:
    """Map a fusion-strategy alias to its canonical key, or ``None`` if unset/unknown."""
    if not fusion:
        return None
    return _FUSION_ALIASES.get(str(fusion).strip().lower())


def _fusion_guidance(
    strategy: str,
    task: str,
    modalities: list[str],
    emb: int,
    size: str,
    rec: dict[str, Any],
    unknown: list[str],
) -> dict[str, Any]:
    """Grounded guidance for the strategies compose() does not auto-emit.

    ``cross_attention`` returns a real ``CrossAttentionFusionExtractor`` skeleton
    (class + channel dims grounded in rslearn's ``xatt_fusion``); ``pre`` and
    ``post`` return the recipe + caveats. No YAML, because each needs choices
    compose() can't make safely on its own.
    """
    primary = rec["primary"]
    others = [m for m in modalities if m != primary]
    base: dict[str, Any] = {
        "ok": True, "task": task, "model_size": size, "strategy": strategy,
        "modalities": modalities, "why": rec["why"], "yaml": None, "config": None,
        "alternatives": rec["alternatives"], "unknown_modalities": unknown,
    }
    if strategy == "cross_attention":
        base["guidance"] = {
            "encoder_class": "rslearn.models.xatt_fusion.CrossAttentionFusionExtractor",
            "primary_path": f"[OlmoEarth reading {primary}] (the query stream)",
            "context_paths": [f"[OlmoEarth reading {m}]" for m in others],
            "primary_output_channels": emb,
            "context_output_channels": [emb for _ in others],
            "missing_context_robustness": "set context_dropout_prob > 0 to train for a "
            "context modality being absent at inference.",
            "steps": [
                "Define one encoder path per modality (each an OlmoEarth FeatureExtractor "
                "restricted to that modality's input).",
                f"Wrap them in CrossAttentionFusionExtractor(primary_path=[...{primary}...], "
                f"context_paths=[[...{', '.join(others) or 'context'}...]], "
                f"primary_output_channels={emb}, context_output_channels={[emb for _ in others]}).",
                f"Feed the fused {emb}-channel output into the same decoder/head compose() "
                f"emits for {task} (decoder in_channels = {emb}).",
            ],
        }
        base["note"] = (
            "cross_attention needs per-path input wiring, so this is a grounded skeleton, "
            "not an auto-emitted YAML. Assemble the model.yaml and check it with "
            "olmoearth_rslearn_validate."
        )
    elif strategy == "pre":
        base["guidance"] = {
            "steps": [
                "Concatenate the co-registered bands of all modalities into ONE dataset "
                "layer / input band-set, then run a single encoder over the stack.",
            ],
            "caveat": "With OlmoEarth you forfeit the modality-specific pretrained "
            "embeddings, so input/pre-fusion only makes sense for same-grid band "
            "EXTENSIONS of one sensor. For genuinely different sensors prefer 'mid' "
            "(shared encoder): compose(..., fusion='mid').",
        }
        base["note"] = "pre-fusion is rarely best with a pretrained multi-modal encoder; see caveat."
    else:  # post
        base["guidance"] = {
            "steps": [
                f"Run compose() once per modality ({', '.join(modalities)}) to get one "
                "single-modality model.yaml each.",
                "Train each model independently.",
                "Ensemble the per-modality predictions at inference (average probabilities "
                "or vote) — a missing modality simply drops its vote.",
            ],
            "best_when": "you need robustness to a modality being missing at inference and "
            "each modality is individually informative.",
        }
        base["note"] = "post/decision fusion is N independent trainings, not one config."
    if unknown:
        base["notes"] = [f"unrecognized modalities ignored: {unknown}"]
    return base


def _decoder_chain(task: str, emb_dim: int, out_channels: int, patch_size: int) -> list[dict[str, Any]]:
    """The OlmoEarth decoder chain for a task (mirrors olmoearth-data-prep)."""
    head = TASKS[task]["head"]
    if task in {"segmentation", "per_pixel_regression"}:
        # dense output: upsample the patch features to pixels, then a 1x1 Conv -> head
        return [
            {"class_path": "rslearn.models.upsample.Upsample", "init_args": {"scale_factor": patch_size}},
            {"class_path": "rslearn.models.conv.Conv", "init_args": {
                "in_channels": emb_dim, "out_channels": out_channels, "kernel_size": 1,
                "activation": {"class_path": "torch.nn.Identity"}}},
            {"class_path": head},
        ]
    # window-level: pool to a vector, then head
    return [
        {"class_path": "rslearn.models.pooling_decoder.PoolingDecoder", "init_args": {
            "in_channels": emb_dim, "out_channels": out_channels, "num_conv_layers": 1, "num_fc_layers": 1}},
        {"class_path": head},
    ]


def compose(
    *,
    task: str | None = None,
    goal: str | None = None,
    model_size: str = "base",
    num_classes: int | None = None,
    class_names: list[str] | None = None,
    property_name: str = "category",
    scale_factor: float | None = None,
    dataset_path: str = "data/dataset",
    bands: list[str] | None = None,
    freeze_epochs: int = 10,
    total_epochs: int = 40,
    nodata_value: int | None = None,
    patch_size: int = 4,
    modalities: list[str] | None = None,
    fusion: str | None = None,
) -> dict[str, Any]:
    """Emit a complete, valid OlmoEarth/rslearn finetune ``model.yaml``.

    Mirrors the proven olmoearth-data-prep template — ``MultiTaskModel`` + a named
    decoder (Upsample/Conv for dense tasks, PoolingDecoder for window tasks) +
    ``FreezeUnfreeze`` — generalized across the four dense/window tasks. Returns
    ``{ok, yaml, config, notes, ask_for}``. Detection is *guided* rather than
    auto-emitted (Faster R-CNN anchors need tuning). Torch-free (dict + YAML only).

    Pass ``modalities`` (2+) to compose a **multi-source fusion** config. The
    ``fusion`` strategy (else auto-picked by :func:`recommend_fusion`) selects the
    rslearn mechanism: ``mid`` (the default) emits a full valid multi-input config —
    one OlmoEarth encoder fed every modality, fused internally; ``cross_attention``,
    ``pre``, and ``post`` return grounded *guidance* (a separate-path
    ``CrossAttentionFusionExtractor`` skeleton, input-concat caveats, or a
    train-per-modality-and-ensemble recipe) rather than an auto-emitted YAML.
    """
    chosen = (task or "").strip().lower() or _route_task(goal or "")
    if chosen not in TASKS:
        return {"ok": False, "task": None,
                "ask_for": ["task (or a clearer goal): " + ", ".join(TASKS)],
                "note": "Pick the rslearn task to compose a config for."}
    size = model_size.strip().lower()
    if size not in EMBEDDING_SIZES:
        size = "base"
    emb = EMBEDDING_SIZES[size]
    ask_for: list[str] = []
    notes: list[str] = []

    if chosen == "detection":
        return {
            "ok": False, "task": "detection",
            "note": "Detection (Faster R-CNN) needs FPN downsample_factors + anchor_sizes tuned to your "
            "object scale, so it isn't auto-emitted. Use olmoearth_rslearn_recommend for the shape, and the "
            "rslearn DetectionTask example (Fpn -> FasterRCNN with num_classes = real classes + 1 for "
            "background) as the template.",
            "ask_for": [],
        }

    # --- multi-source fusion: resolve the strategy, emit (mid) or guide (others) ---
    canon, unknown_modalities = _normalize_modalities(modalities)
    fusion_block: dict[str, Any] | None = None
    emit_modalities = ["sentinel2_l2a"]
    if len(canon) >= 2:
        strategy = _normalize_fusion(fusion) or recommend_fusion(
            canon, task=chosen, robust_to_missing=(fusion or "").strip().lower() in {"post", "ensemble"}
        )["strategy"]
        rec = recommend_fusion(canon, task=chosen, robust_to_missing=(strategy == "post"))
        non_native = [m for m in canon if not FUSION_MODALITIES[m]["native"]]
        # A non-native modality can't ride the shared OlmoEarth encoder -> steer to cross_attention.
        if strategy == "mid" and non_native:
            strategy = "cross_attention"
            rec["why"] = (
                f"{', '.join(non_native)} is not an OlmoEarth pretrain modality, so 'mid' "
                "(shared encoder) can't include it; using cross_attention instead. " + rec["why"]
            )
        if strategy != "mid":
            return _fusion_guidance(
                strategy, chosen, canon, emb, size, rec, unknown_modalities
            )
        emit_modalities = canon
        fusion_block = {
            "strategy": "mid", "modalities": canon, "why": rec["why"],
            "mechanism": "one OlmoEarth encoder fed all modality inputs; it fuses them "
            "internally via cross-modal attention (no extra fusion module).",
            "alternatives": rec["alternatives"],
        }
        if unknown_modalities:
            notes.append(f"unrecognized modalities ignored: {unknown_modalities}")
    elif len(canon) == 1:
        emit_modalities = canon  # single explicit modality (still one OlmoEarth input)

    if not any(FUSION_MODALITIES[m]["native"] for m in emit_modalities):
        return {
            "ok": False, "task": chosen, "modalities": emit_modalities,
            "note": "OlmoEarth's shared encoder needs at least one pretrain modality "
            f"({', '.join(_NATIVE_MODALITIES)}); {emit_modalities} alone can't be encoded "
            "by it. Add Sentinel-2, or use cross_attention with a separate encoder path.",
            "ask_for": [],
        }

    t = TASKS[chosen]
    raster = t["label_type"] == "raster"
    out_channels = 1
    if t["needs_num_classes"]:
        if not num_classes:
            ask_for.append("num_classes")
            num_classes = 2  # placeholder so the YAML is well-formed (flagged in ask_for)
        out_channels = num_classes
    key = _DECODER_KEY[chosen]

    task_init: dict[str, Any] = {}
    if chosen == "segmentation":
        task_init = {"num_classes": num_classes, "zero_is_invalid": False, "nodata_value": nodata_value}
    elif chosen == "per_pixel_regression":
        if scale_factor is None:
            ask_for.append("scale_factor (or a label_range via recommend to derive it)")
        task_init = {"scale_factor": scale_factor if scale_factor is not None else 1.0,
                     "metrics": ["mse", "r2"], "nodata_value": nodata_value}
    elif chosen == "classification":
        task_init = {"property_name": property_name,
                     "classes": class_names or [f"class_{i}" for i in range(num_classes or 2)]}
        if not class_names:
            notes.append("Replace the placeholder class names with your real category names.")
    elif chosen == "regression":
        if scale_factor is None:
            ask_for.append("scale_factor (or a label_range via recommend to derive it)")
        task_init = {"property_name": property_name,
                     "scale_factor": scale_factor if scale_factor is not None else 1.0, "metrics": ["mse"]}

    if raster:
        label_input = {"data_type": "raster", "layers": ["label"],
                       "bands": ["category"] if chosen == "segmentation" else ["value"],
                       "is_target": True, "dtype": "INT32" if chosen == "segmentation" else "FLOAT32"}
    else:
        label_input = {"data_type": "vector", "layers": ["label"], "is_target": True}

    monitor = f"val_{key}/accuracy" if chosen in {"segmentation", "classification"} else f"val_{key}/loss"
    mode = "max" if monitor.endswith("accuracy") else "min"

    config = {
        "model": {
            "class_path": "rslearn.train.lightning_module.RslearnLightningModule",
            "init_args": {
                "model": {
                    "class_path": "rslearn.models.multitask.MultiTaskModel",
                    "init_args": {
                        "encoder": [{
                            "class_path": "rslearn.models.olmoearth_pretrain.model.OlmoEarth",
                            "init_args": {"model_id": _MODEL_ID[size], "patch_size": patch_size},
                        }],
                        "decoders": {key: _decoder_chain(chosen, emb, out_channels, patch_size)},
                    },
                },
                "lr": 0.0001, "plateau": True, "plateau_factor": 0.2, "plateau_patience": 2,
            },
        },
        "data": {
            "class_path": "rslearn.train.data_module.RslearnDataModule",
            "init_args": {
                "path": dataset_path,
                "inputs": {**_modality_inputs(emit_modalities, bands), "label": label_input},
                "task": {
                    "class_path": "rslearn.train.tasks.multi_task.MultiTask",
                    "init_args": {
                        "tasks": {key: {"class_path": t["task_class"], "init_args": task_init}},
                        "input_mapping": {key: {"label": "targets"}},
                    },
                },
                "batch_size": 4, "num_workers": 4,
            },
        },
        "trainer": {
            "max_epochs": total_epochs, "accelerator": "gpu", "devices": 1, "log_every_n_steps": 10,
            "callbacks": [
                {"class_path": "lightning.pytorch.callbacks.ModelCheckpoint",
                 "init_args": {"dirpath": "checkpoints", "save_top_k": 1, "save_last": True,
                               "monitor": monitor, "mode": mode}},
                {"class_path": "rslearn.train.callbacks.freeze_unfreeze.FreezeUnfreeze",
                 "init_args": {"module_selector": ["model", "encoder", 0],
                               "unfreeze_at_epoch": freeze_epochs, "unfreeze_lr_factor": 10}},
            ],
        },
    }

    if fusion_block:
        notes.append(
            f"mid-fusion: one OlmoEarth encoder reads {len(emit_modalities)} modality inputs "
            f"({', '.join(emit_modalities)}) and fuses them internally — the decoder still sees a "
            f"single {emb}-channel feature map, so its channel contract is unchanged. Make sure your "
            "dataset config.json has a layer for each modality input."
        )
    notes.append(
        f"decoder in_channels = {emb} (OlmoEarth {size} embedding dim), out_channels = {out_channels} "
        f"({'num_classes' if t['needs_num_classes'] else '1 for regression'}); encoder frozen for "
        f"{freeze_epochs} epochs, then unfrozen at 10x LR. Validate this with olmoearth_rslearn_validate."
    )
    text: str | None
    try:
        import yaml  # type: ignore[import-untyped]
        text = yaml.safe_dump(config, sort_keys=False, default_flow_style=False)
    except Exception:  # noqa: BLE001 - YAML is best-effort; the dict is always returned
        text = None
        notes.append("PyYAML unavailable; returning the config as a dict only.")

    return {"ok": True, "task": chosen, "model_size": size, "yaml": text, "config": config,
            "modalities": emit_modalities, "fusion": fusion_block,
            "notes": notes, "ask_for": ask_for,
            "next": "Run olmoearth_rslearn_validate on this model.yaml + your dataset config.json, then "
            "`rslearn model fit --config model.yaml` (a long GPU job — see the olmoearth-rslearn skill)."}


# --------------------------------------------------------------------------- #
# Diagnoser (parse a failing prepare/ingest/materialize run -> fixes)
# --------------------------------------------------------------------------- #


def diagnose(summary: Any = None, log_text: str | None = None) -> dict[str, Any]:
    """Turn a failing rslearn run into plain-English fixes.

    Accepts a parsed summary (a ``PrepareDatasetWindowsSummary``-like dict with
    per-layer ``windows_prepared``/``windows_rejected``/``windows_failed`` +
    ``error_messages``) and/or raw log text. Pure dict/string inspection — no
    rslearn import. Returns ``{ok, diagnosis, fixes}``.
    """
    diagnosis: list[str] = []
    fixes: list[str] = []

    def _scan(d: Any) -> None:
        if isinstance(d, dict):
            prep, rej, fail = d.get("windows_prepared"), d.get("windows_rejected"), d.get("windows_failed")
            if prep == 0:
                diagnosis.append("0 windows prepared for a layer.")
                fixes.append("Widen --start/--end (the time range likely has no source scenes), or check the "
                             "AOI box / --src_crs — `prepare` found no scenes intersecting the windows.")
            if isinstance(rej, int) and isinstance(prep, int) and rej > prep > -1 and rej > 0:
                diagnosis.append(f"More windows rejected ({rej}) than prepared ({prep}).")
                fixes.append("min_matches may be too high or cloud filtering too strict — lower min_matches, "
                             "widen the time range, or confirm sort_by=cloud_cover.")
            if isinstance(fail, int) and fail > 0:
                diagnosis.append(f"{fail} windows failed (see error_messages).")
            for v in d.values():
                _scan(v)
        elif isinstance(d, list):
            for v in d:
                _scan(v)

    if summary is not None:
        _scan(summary)

    blob = (log_text or "")
    if isinstance(summary, (dict, list)):
        blob += " " + json.dumps(summary)
    low = blob.lower()
    if "no scenes" in low or "no items" in low or "0 scenes" in low:
        diagnosis.append("'no scenes found' in the output.")
        fixes.append("Widen --start/--end by 2+ weeks, or verify the AOI is within the data source's coverage.")
    if "crs" in low or "epsg" in low or " utm" in low:
        diagnosis.append("a CRS/EPSG mismatch is mentioned.")
        fixes.append("Set --src_crs to your box's CRS (the box is W,S,E,N lon/lat for EPSG:4326); a wrong CRS "
                     "lands the windows off-target (often in the ocean).")
    if "out of memory" in low or "oom" in low or ("cuda" in low and "memory" in low):
        diagnosis.append("an out-of-memory error is mentioned.")
        fixes.append("Lower batch_size / num_workers or grid_size, or use a smaller OlmoEarth model_id.")
    if "no space" in low or "disk" in low:
        diagnosis.append("a disk-space error is mentioned.")
        fixes.append("`materialize` needs room for cropped tiles — free disk or point --root at a larger volume.")
    if "401" in low or "403" in low or "unauthorized" in low or "credential" in low:
        diagnosis.append("an auth/credential error is mentioned.")
        fixes.append("Check the data-source credentials / Studio key; `ingest` is idempotent, so rerun after fixing.")

    if not diagnosis:
        diagnosis.append("No known failure pattern detected.")
        fixes.append("Paste the rslearn prepare/ingest/materialize summary JSON or the error lines for a targeted read.")

    ok = not any(k in d for d in diagnosis for k in ("0 windows", "no scenes", "failed", "error", "memory", "disk", "auth"))
    return {"ok": ok, "diagnosis": diagnosis, "fixes": fixes}
