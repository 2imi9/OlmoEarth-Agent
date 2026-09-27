# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Ziming Qi
"""
recommend.py — Recommend an OlmoEarth Studio "new model" wizard config from a task description.

Takes a plain-English task ("predict mangrove extent", "detect vessels") and emits
a filled config matching Studio's "new model" wizard (Studio trains on Ai2's
compute — none of these fields ask about the user's own hardware):

    {
      "output_type": "...",
      "foundation_model": "nano" | "tiny" | "base",
      "label_field": "...",
      "training_data": {"mode": "full"} | {"mode": "proportion", "fraction": 0.0-1.0},
      "data_split": {"method": "spatial"} | {"method": "metadata_field", "field": "..."},
      "time_frame": { "mode": "...", ... },
      "imagery_sources": ["sentinel2", ...],
      "patch_size_m": 160 | 320 | 640 | 1280,
      "rationale": { ... per-field one-liner ... }
    }

(The wizard's first step, Model name, is the user's free text — not emitted here.)

Usage:
    python recommend.py "predict mangrove extent in Indonesia"
    python recommend.py --task "vessel detection" --num-classes 1 --num-samples 8000
    python recommend.py --validate config.json

Stdlib only. Heuristics live in PRESETS / KEYWORD_RULES at the top so the
mapping is auditable in one screen.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# ----- Verified-from-Studio preset configs --------------------------------
# Each entry is a complete wizard answer plus a one-line rationale per field.
# Keep these in sync with references/presets.md.
# Patch default is Medium 640 m (the wizard's recommendation for most tasks);
# 1280 m for detection / large-region labels. adjust_for_signals() steps
# 640 -> 320 for small (<1,000-label) datasets.

_FULL = {"mode": "full"}
_SPATIAL = {"method": "spatial"}

PRESETS: dict[str, dict] = {
    "crop_type": {
        "output_type": "per_pixel_classification",
        "foundation_model": "tiny",
        "label_field": "oe_labels.category",
        "training_data": _FULL,
        "data_split": _SPATIAL,
        "time_frame": {"mode": "period", "period_months": 12, "start_months": [3, 4]},
        "imagery_sources": ["sentinel2"],
        "patch_size_m": 640,
    },
    "mangrove": {
        "output_type": "per_pixel_classification",
        "foundation_model": "tiny",
        "label_field": "oe_labels.category",
        "training_data": _FULL,
        "data_split": _SPATIAL,
        "time_frame": {"mode": "period", "period_months": 12, "start_months": list(range(1, 13))},
        "imagery_sources": ["sentinel2"],
        "patch_size_m": 640,
    },
    "land_cover": {
        "output_type": "per_pixel_classification",
        "foundation_model": "base",
        "label_field": "oe_labels.category",
        "training_data": _FULL,
        "data_split": _SPATIAL,
        "time_frame": {"mode": "period", "period_months": 12, "start_months": [1]},
        "imagery_sources": ["sentinel2"],
        "patch_size_m": 640,
    },
    "soil_moisture": {
        "output_type": "per_pixel_regression",
        "foundation_model": "tiny",
        "label_field": "oe_labels.moisture_pct",
        "training_data": _FULL,
        "data_split": _SPATIAL,
        "time_frame": {"mode": "single_moment_with_context",
                       "before_months": 3, "after_months": 0, "before_offset_days": 0},
        "imagery_sources": ["sentinel2", "sentinel1"],
        "patch_size_m": 640,
    },
    "tree_height": {
        "output_type": "per_pixel_regression",
        "foundation_model": "base",
        "label_field": "oe_labels.height_m",
        "training_data": _FULL,
        "data_split": _SPATIAL,
        "time_frame": {"mode": "period", "period_months": 12, "start_months": [6]},
        "imagery_sources": ["sentinel2"],
        "patch_size_m": 640,
    },
    "biomass": {
        "output_type": "window_regression",
        "foundation_model": "base",
        "label_field": "oe_labels.biomass_mg_per_ha",
        "training_data": _FULL,
        "data_split": _SPATIAL,
        "time_frame": {"mode": "period", "period_months": 12, "start_months": [6]},
        "imagery_sources": ["sentinel2", "sentinel1"],
        "patch_size_m": 640,
    },
    "ecosystem_type": {
        "output_type": "window_classification",
        "foundation_model": "tiny",
        "label_field": "oe_labels.ecosystem",
        "training_data": _FULL,
        "data_split": _SPATIAL,
        "time_frame": {"mode": "period", "period_months": 12, "start_months": [1]},
        "imagery_sources": ["sentinel2"],
        "patch_size_m": 640,
    },
    "vessel_detection": {
        "output_type": "point_detection",
        "foundation_model": "tiny",
        "label_field": "oe_labels.vessel_type",
        "training_data": _FULL,
        "data_split": _SPATIAL,
        "time_frame": {"mode": "single_moment", "observation_window_hours": 12},
        "imagery_sources": ["sentinel2", "sentinel1"],
        "patch_size_m": 1280,
    },
    "solar_array_detection": {
        "output_type": "point_detection",
        "foundation_model": "base",
        "label_field": "oe_labels.array_type",
        "training_data": _FULL,
        "data_split": _SPATIAL,
        "time_frame": {"mode": "period", "period_months": 12, "start_months": [1]},
        "imagery_sources": ["sentinel2"],
        "patch_size_m": 1280,
    },
    "oil_slick": {
        "output_type": "point_detection",
        "foundation_model": "tiny",
        "label_field": "oe_labels.slick",
        "training_data": _FULL,
        "data_split": _SPATIAL,
        "time_frame": {"mode": "single_moment", "observation_window_hours": 12},
        "imagery_sources": ["sentinel1", "sentinel2"],
        "patch_size_m": 1280,
    },
    "flood": {
        "output_type": "per_pixel_classification",
        "foundation_model": "base",
        "label_field": "oe_labels.flood_state",
        "training_data": _FULL,
        "data_split": _SPATIAL,
        "time_frame": {"mode": "single_moment_with_context",
                       "before_months": 1, "after_months": 2,
                       "before_offset_days": 7, "after_offset_days": 0},
        "imagery_sources": ["sentinel2", "sentinel1"],
        "patch_size_m": 640,
    },
    "drought": {
        "output_type": "per_pixel_regression",
        "foundation_model": "tiny",
        "label_field": "oe_labels.drought_index",
        "training_data": _FULL,
        "data_split": _SPATIAL,
        "time_frame": {"mode": "single_moment_with_context",
                       "before_months": 6, "after_months": 0, "before_offset_days": 0},
        "imagery_sources": ["sentinel2"],
        "patch_size_m": 640,
    },
    "burn_scar": {
        "output_type": "per_pixel_classification",
        "foundation_model": "tiny",
        "label_field": "oe_labels.burned",
        "training_data": _FULL,
        "data_split": _SPATIAL,
        "time_frame": {"mode": "single_moment_with_context",
                       "before_months": 1, "after_months": 1,
                       "before_offset_days": 0, "after_offset_days": 0},
        "imagery_sources": ["sentinel2"],
        "patch_size_m": 640,
    },
    "embeddings": {
        "output_type": "embeddings",
        "foundation_model": "tiny",
        "label_field": None,
        "training_data": None,
        "data_split": None,
        "time_frame": {"mode": "period", "period_months": 12, "start_months": [1]},
        "imagery_sources": ["sentinel2"],
        "patch_size_m": 640,
    },
}

# Keyword → preset. First match wins. Order matters — put more specific
# phrases first so "solar array" beats "array" inside a detection task, etc.
KEYWORD_RULES: list[tuple[str, str]] = [
    (r"\b(crop|maize|corn|wheat|soy|rice|sugar|cotton)\b", "crop_type"),
    (r"\bmangrove", "mangrove"),
    (r"\bland\s*cover|landcover|land-cover", "land_cover"),
    (r"\bsoil\s*moisture|moisture", "soil_moisture"),
    (r"\btree\s*height|canopy(?:\s+height|\s+density)?", "tree_height"),
    (r"\bbiomass|carbon\s*stock", "biomass"),
    (r"\becosystem(?:\s*type)?", "ecosystem_type"),
    (r"\bsolar(?:\s*array|\s*panel|\s*farm)?", "solar_array_detection"),
    (r"\boil\s*slick|oil\s*spill", "oil_slick"),
    (r"\bvessel|ship(?!ping)|boat", "vessel_detection"),
    (r"\bflood", "flood"),
    (r"\bdrought", "drought"),
    (r"\bburn(?:\s*scar)?|fire\s*scar|burned\s*area", "burn_scar"),
    (r"\bembed", "embeddings"),
]

RATIONALES: dict[str, dict[str, str]] = {
    "crop_type": {
        "output_type": "Crop type is one class per pixel within field boundaries → semantic segmentation.",
        "foundation_model": "Default Tiny; bump to Base if >5 classes or <2K samples.",
        "time_frame": "Crop type describes an annual property; March/April aligns to Northern-hemisphere growing season start.",
        "imagery_sources": "Optical-only is sufficient for crop spectral signatures; add S1 later if cloud cover > 30%.",
        "patch_size_m": "640 m — the wizard's recommended default; field-plus-neighbors context.",
    },
    "mangrove": {
        "output_type": "Mangrove vs non-mangrove per pixel → semantic segmentation.",
        "foundation_model": "Binary task; Tiny is plenty.",
        "time_frame": "Mangrove extent is stable year-round; any start month is valid.",
        "imagery_sources": "S2 spectral signature is highly distinctive for mangrove. Add S1 in heavily tidal areas.",
        "patch_size_m": "640 m default balances shoreline context and resolution.",
    },
    "land_cover": {
        "output_type": "Multi-class label per pixel → semantic segmentation.",
        "foundation_model": "5+ classes warrant Base for richer features.",
        "time_frame": "Annual land cover; January start as a neutral default.",
        "imagery_sources": "S2 alone resolves most cover types.",
        "patch_size_m": "640 m default; broad cover classes benefit from landscape context.",
    },
    "soil_moisture": {
        "output_type": "Continuous moisture per pixel → per-pixel regression.",
        "foundation_model": "Tiny suffices unless dataset is very small.",
        "time_frame": "3 months of before-context captures the drying trend that maps to current moisture.",
        "imagery_sources": "S1 is the dielectric/moisture signal; S2 carries NDVI as a vegetation proxy.",
        "patch_size_m": "640 m default; moisture varies at field scale and context helps.",
    },
    "tree_height": {
        "output_type": "Continuous height per pixel → per-pixel regression.",
        "foundation_model": "High label variance → Base for representation quality.",
        "time_frame": "Mid-year start captures peak leaf-on phenology.",
        "imagery_sources": "S2 alone for canopy; add S1 if forest is dense and radar penetration helps.",
        "patch_size_m": "640 m default; stand-level structure context.",
    },
    "biomass": {
        "output_type": "One value per region → window regression.",
        "foundation_model": "Regression with high variance → Base.",
        "time_frame": "Annual property; June start captures peak biomass season.",
        "imagery_sources": "S1 backscatter is a structure signal that helps biomass estimation.",
        "patch_size_m": "640 m matches typical biomass plot scale (and the wizard's default).",
    },
    "ecosystem_type": {
        "output_type": "One class per tile → window classification.",
        "foundation_model": "Few-class problem → Tiny.",
        "time_frame": "Annual property; January start as neutral default.",
        "imagery_sources": "S2 spectra resolve ecosystems.",
        "patch_size_m": "640 m default — the patch is the prediction tile; match the user's tile size if they name one.",
    },
    "vessel_detection": {
        "output_type": "Discrete moving objects → point/bbox detection.",
        "foundation_model": "Tiny default; Base if vessel-type taxonomy is fine-grained.",
        "time_frame": "Vessels move every hour; single moment ±12 h.",
        "imagery_sources": "S1 catches night detections and small wakes; S2 daylight confirmation.",
        "patch_size_m": "1280 m per Studio's detection recommendation — gives context for wakes and neighbors.",
    },
    "solar_array_detection": {
        "output_type": "Discrete stationary objects → point/bbox detection.",
        "foundation_model": "Base helps with fine-grained array-type distinctions.",
        "time_frame": "Solar arrays are stable installations → period mode, annual.",
        "imagery_sources": "S2 spectral signature (high reflectance) is the signal.",
        "patch_size_m": "1280 m per Studio's detection recommendation.",
    },
    "oil_slick": {
        "output_type": "Transient objects with bbox/point support → detection.",
        "foundation_model": "Tiny for a binary detection task.",
        "time_frame": "Slicks evolve hourly; single moment ±12 h.",
        "imagery_sources": "S1 primary (slick suppresses wind-roughness → dark patch); S2 secondary.",
        "patch_size_m": "1280 m for context — slicks have large extent and variable shape.",
    },
    "flood": {
        "output_type": "Flooded/not-flooded per pixel → semantic segmentation.",
        "foundation_model": "Class imbalance + spatial complexity → Base.",
        "time_frame": "Single moment with before/after context; 7-day before-offset skips event-day cloud cover.",
        "imagery_sources": "S1 is essential — floods happen under storm clouds that block S2.",
        "patch_size_m": "640 m captures floodplain context.",
    },
    "drought": {
        "output_type": "Continuous drought index per pixel → per-pixel regression.",
        "foundation_model": "Tiny suffices.",
        "time_frame": "6 months of before-context captures the dry-down trajectory.",
        "imagery_sources": "S2 NDVI/EVI carries the vegetation-stress signal.",
        "patch_size_m": "640 m default.",
    },
    "burn_scar": {
        "output_type": "Burned/unburned per pixel → semantic segmentation.",
        "foundation_model": "Binary task → Tiny.",
        "time_frame": "Before/after context lets the model contrast pre-burn and post-burn surfaces.",
        "imagery_sources": "S2 SWIR band is the burn signal.",
        "patch_size_m": "640 m default; fire perimeters benefit from surrounding context.",
    },
    "embeddings": {
        "output_type": "Feature vectors per pixel/tile, no supervised label → embeddings.",
        "foundation_model": "Tiny is the sweet spot for downstream consumers.",
        "time_frame": "12-month period captures seasonal variation in features.",
        "imagery_sources": "S2 alone for general-purpose embeddings.",
        "patch_size_m": "640 m default; match downstream consumer if known.",
    },
}

_DEFAULT_RATIONALES = {
    "training_data": "Train on the full dataset; a proportion is only for a quick first iteration on a very large dataset.",
    "data_split": "Spatial split (Studio's recommendation) keeps validation areas spatially separate — random splits leak neighbors and inflate accuracy.",
}

VALID_OUTPUT_TYPES = {
    "per_pixel_regression", "per_pixel_classification",
    "window_regression", "window_classification",
    "point_detection", "embeddings",
}
VALID_MODELS = {"nano", "tiny", "base"}
VALID_TRAINING_MODES = {"full", "proportion"}
VALID_SPLIT_METHODS = {"spatial", "metadata_field"}
VALID_TIME_MODES = {"period", "single_moment_with_context", "single_moment"}
VALID_PATCH = {160, 320, 640, 1280}
VALID_SOURCES = {"sentinel2", "sentinel1", "landsat"}


# ----- Recommendation -----------------------------------------------------

def match_preset(task: str) -> str | None:
    t = task.lower()
    for pattern, key in KEYWORD_RULES:
        if re.search(pattern, t):
            return key
    return None


def adjust_for_signals(cfg: dict, num_classes: int | None, num_samples: int | None) -> dict:
    """Tweak model size and patch size based on dataset signals the user provided."""
    cfg = json.loads(json.dumps(cfg))
    rationale = dict(RATIONALES.get(cfg["__preset_key"], {}))

    if num_classes is not None and cfg["output_type"] in {"per_pixel_classification",
                                                          "window_classification"}:
        if num_classes > 5 and cfg["foundation_model"] != "base":
            cfg["foundation_model"] = "base"
            rationale["foundation_model"] = (
                f"{num_classes} classes (>5) → Base for richer multi-class features."
            )
    if num_samples is not None:
        if num_samples < 2000 and cfg["foundation_model"] == "tiny":
            cfg["foundation_model"] = "base"
            rationale["foundation_model"] = (
                f"{num_samples} samples (<2K) → Base — representation quality dominates with small data."
            )
        elif num_samples > 20000 and cfg["foundation_model"] == "base":
            cfg["foundation_model"] = "tiny"
            rationale["foundation_model"] = (
                f"{num_samples} samples (>20K) → Tiny — large data lets a smaller model close the gap, faster to train."
            )
        # Small datasets also step the wizard's 640 m default down to 320 m
        # (detection stays at 1280 m — objects always need surrounding context).
        if num_samples < 1000 and cfg["patch_size_m"] == 640:
            cfg["patch_size_m"] = 320
            rationale["patch_size_m"] = (
                f"{num_samples} labels (<1,000) → step the 640 m default down to 320 m for a small dataset."
            )

    for field, text in _DEFAULT_RATIONALES.items():
        if cfg.get(field) is not None:
            rationale.setdefault(field, text)

    cfg["rationale"] = rationale
    cfg.pop("__preset_key", None)
    return cfg


def recommend(task: str, num_classes: int | None = None,
              num_samples: int | None = None) -> dict:
    key = match_preset(task)
    if key is None:
        raise SystemExit(
            f"Could not match task description to a known preset.\n"
            f"Task: {task!r}\n"
            f"Known presets: {sorted(PRESETS)}\n"
            f"Try rephrasing or feed --preset PRESET_NAME explicitly."
        )
    cfg = json.loads(json.dumps(PRESETS[key]))
    cfg["__preset_key"] = key
    cfg = adjust_for_signals(cfg, num_classes, num_samples)
    cfg["__preset_used"] = key
    return cfg


# ----- Validation ---------------------------------------------------------

def validate(cfg: dict) -> list[str]:
    """Return a list of problems with a Studio config. Empty list = OK."""
    problems: list[str] = []

    out = cfg.get("output_type")
    if out not in VALID_OUTPUT_TYPES:
        problems.append(f"output_type must be one of {sorted(VALID_OUTPUT_TYPES)}")

    if cfg.get("foundation_model") not in VALID_MODELS:
        problems.append(f"foundation_model must be one of {sorted(VALID_MODELS)}")

    # Label-dependent wizard steps (label field, training data, data split)
    # apply to fine-tuned model types only.
    if out == "embeddings":
        if cfg.get("label_field"):
            problems.append("embeddings output type does not use a label_field")
        if cfg.get("training_data") is not None:
            problems.append("embeddings output type does not use the training_data step")
        if cfg.get("data_split") is not None:
            problems.append("embeddings output type does not use the data_split step")
    elif out in VALID_OUTPUT_TYPES:
        td = cfg.get("training_data")
        if not isinstance(td, dict) or td.get("mode") not in VALID_TRAINING_MODES:
            problems.append(
                f"training_data.mode must be one of {sorted(VALID_TRAINING_MODES)}"
            )
        elif td.get("mode") == "proportion":
            frac = td.get("fraction")
            if not isinstance(frac, (int, float)) or not (0 < frac < 1):
                problems.append("training_data proportion requires 0 < fraction < 1")

        ds = cfg.get("data_split")
        if not isinstance(ds, dict) or ds.get("method") not in VALID_SPLIT_METHODS:
            problems.append(
                f"data_split.method must be one of {sorted(VALID_SPLIT_METHODS)}"
            )
        elif ds.get("method") == "metadata_field" and not ds.get("field"):
            problems.append(
                "data_split with metadata_field requires the field name whose "
                "values assign train/val/test"
            )

    tf = cfg.get("time_frame") or {}
    mode = tf.get("mode")
    if mode not in VALID_TIME_MODES:
        problems.append(f"time_frame.mode must be one of {sorted(VALID_TIME_MODES)}")

    if mode == "period":
        p = tf.get("period_months")
        if not isinstance(p, int) or not (1 <= p <= 12):
            problems.append("period_months must be an integer 1–12")
        if not tf.get("start_months"):
            problems.append("period mode requires at least one start month")
    elif mode == "single_moment_with_context":
        b = tf.get("before_months", 0)
        a = tf.get("after_months", 0)
        if (b or 0) == 0 and (a or 0) == 0:
            problems.append("single_moment_with_context requires before_months or after_months > 0")
        for name, v in (("before_months", b), ("after_months", a)):
            if v is not None and isinstance(v, int) and not (0 <= v <= 12):
                problems.append(f"{name} must be 0–12 (the wizard's timeline is up to 12 months each way)")
    elif mode == "single_moment":
        h = tf.get("observation_window_hours")
        if not isinstance(h, (int, float)) or h <= 0:
            problems.append("single_moment requires observation_window_hours > 0")

    srcs = cfg.get("imagery_sources") or []
    if not srcs:
        problems.append("imagery_sources must list at least one source")
    for s in srcs:
        if s not in VALID_SOURCES:
            problems.append(f"unknown imagery source: {s!r}")
    if "landsat" in srcs:
        problems.append("landsat is not yet available in Studio (per the wizard)")

    patch = cfg.get("patch_size_m")
    if patch not in VALID_PATCH:
        problems.append(f"patch_size_m must be one of {sorted(VALID_PATCH)}")

    # Cross-field sanity
    if out == "point_detection" and patch is not None and patch < 640:
        problems.append(
            f"detection with patch_size_m={patch} is too small — Studio recommends 1280 m"
        )
    if out == "point_detection" and mode == "period" and (tf.get("period_months") or 0) >= 6:
        # OK for stationary objects (solar) but warn for moving targets
        problems.append(
            "detection + multi-month period: only valid for stationary objects "
            "(solar arrays, oil tanks). For moving targets (vessels, slicks), use single_moment."
        )

    return problems


# ----- CLI ----------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Recommend an OlmoEarth Studio model-wizard config.")
    p.add_argument("task", nargs="?", help="Plain-English task description.")
    p.add_argument("--task", dest="task_flag", help="Alternative to positional task.")
    p.add_argument("--preset", help="Use a named preset directly (skip keyword matching).",
                   choices=sorted(PRESETS))
    p.add_argument("--num-classes", type=int, help="Number of label classes.")
    p.add_argument("--num-samples", type=int, help="Approximate number of labeled samples.")
    p.add_argument("--validate", metavar="CONFIG_JSON",
                   help="Validate an existing config JSON file and exit.")
    p.add_argument("--list-presets", action="store_true",
                   help="Print all preset keys and exit.")
    args = p.parse_args(argv)

    if args.list_presets:
        for k in sorted(PRESETS):
            print(k)
        return 0

    if args.validate:
        cfg = json.loads(Path(args.validate).read_text(encoding="utf-8-sig"))
        problems = validate(cfg)
        if not problems:
            print("OK — config passes Studio wizard validation.")
            return 0
        print("Problems found:")
        for prob in problems:
            print(f"  - {prob}")
        return 1

    task = args.task_flag or args.task
    if args.preset:
        cfg = json.loads(json.dumps(PRESETS[args.preset]))
        cfg["__preset_key"] = args.preset
        cfg = adjust_for_signals(cfg, args.num_classes, args.num_samples)
        cfg["__preset_used"] = args.preset
    elif task:
        cfg = recommend(task, args.num_classes, args.num_samples)
    else:
        p.print_help()
        return 2

    print(json.dumps(cfg, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
