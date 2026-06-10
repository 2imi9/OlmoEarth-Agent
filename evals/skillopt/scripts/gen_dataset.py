"""Generate the OlmoEarth job-config benchmark dataset.

Ground truth is computed by importing the skill's own oracle
(``recommend.py``: PRESETS + adjust_for_signals), so expected configs can
never drift from the skill's documented behavior. Task descriptions are
natural-language paraphrases (the target model must reason from the skill,
not pattern-match a preset key). Some items carry class/sample signals that
flip the recommended model size — the skill's model-size heuristic is a real
pitfall area, so the benchmark exercises it.

Writes split_dir/{train,val,test}/items.json.
"""
from __future__ import annotations

import copy
import importlib.util
import json
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
SKILLOPT_ROOT = os.path.dirname(HERE)
RECOMMEND_PATH = os.path.abspath(
    os.path.join(
        SKILLOPT_ROOT, "..", "OlmoEarth Agent", "vendor", "olmoearth-skills",
        "skills", "olmoearth-studio-job-config", "scripts", "recommend.py",
    )
)
OUT_DIR = os.path.join(SKILLOPT_ROOT, "data", "olmoearth_jobconfig_split")

spec = importlib.util.spec_from_file_location("oe_recommend", RECOMMEND_PATH)
rec = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rec)


def expected_for(preset_key: str, num_classes=None, num_samples=None) -> dict:
    cfg = copy.deepcopy(rec.PRESETS[preset_key])
    cfg["__preset_key"] = preset_key
    cfg = rec.adjust_for_signals(cfg, num_classes, num_samples)
    # PRESETS are authoritative; validate() emits an intentional soft-warning
    # for stationary-object detection (solar) using period mode, so we don't
    # treat its output as fatal here — just surface anything unexpected.
    problems = [p for p in rec.validate(cfg) if "stationary objects" not in p]
    if problems:
        print(f"  [warn] {preset_key}: {problems}")
    return {
        "output_type": cfg["output_type"],
        "foundation_model": cfg["foundation_model"],
        "time_frame": cfg["time_frame"],
        "imagery_sources": cfg["imagery_sources"],
        "patch_size_m": cfg["patch_size_m"],
    }


# (preset_key, task_description, num_classes, num_samples)
# Mix of: plain defaults, and model-size signal-flip cases that exercise the
# skill's heuristic (<2K samples or >5 classes -> Base; >20K -> Tiny). The
# foundation_model field is the weakest in the baseline, so it is sampled hard.
SPECS: list[tuple[str, str, int | None, int | None]] = [
    # crop_type (default tiny; flips to base on <2K samples or >5 classes)
    ("crop_type", "Map which crop is growing in each field across the Nile Delta for the 2024 growing season using Sentinel-2 time series.", None, None),
    ("crop_type", "We have about 1,500 labeled field polygons and want per-field crop classification across six crop types in Punjab.", 6, 1500),
    ("crop_type", "Per-pixel crop-type mapping for a smallholder region; eight crop classes, roughly 900 labeled parcels.", 8, 900),
    ("crop_type", "Crop-type segmentation over an irrigation district with a large archive — about 50,000 labeled fields, 4 crop classes.", 4, 50000),
    # mangrove (binary; tiny unless data is tiny)
    ("mangrove", "Delineate mangrove versus non-mangrove extent along the coast of Indonesia from optical imagery.", None, None),
    ("mangrove", "Produce an annual mangrove cover mask for a tidal estuary — a binary per-pixel classification.", None, None),
    ("mangrove", "Mangrove vs non-mangrove segmentation in a data-scarce delta; only ~600 labeled patches so far.", 2, 600),
    # land_cover (default base; flips to tiny on >20K samples)
    ("land_cover", "Produce an annual land-cover map (urban, water, forest, cropland, bare soil, grassland) for a large watershed.", 6, None),
    ("land_cover", "Continental-scale land-cover classification from Sentinel-2 with around 40,000 labeled samples.", 7, 40000),
    ("land_cover", "Six-class land cover for a province; very large training set, roughly 60,000 labeled pixels-blocks.", 6, 60000),
    # soil_moisture (per-pixel regression, before-context; tiny default)
    ("soil_moisture", "Estimate surface soil moisture as a continuous percentage per pixel for an agricultural region, using the months leading up to each observation.", None, None),
    ("soil_moisture", "Per-pixel soil-moisture regression for a drought-prone basin; radar plus optical.", None, None),
    ("soil_moisture", "Soil-moisture regression where we only have ~800 in-situ probe samples to train on.", None, 800),
    # tree_height (per-pixel regression; default base; flips tiny on >20K)
    ("tree_height", "Predict canopy height in metres per pixel over a temperate forest.", None, None),
    ("tree_height", "Estimate tree height across a managed plantation; we have roughly 35,000 lidar-derived samples.", None, 35000),
    # biomass (window regression; default base; flips tiny on >20K)
    ("biomass", "Estimate average above-ground biomass per 640 m region for a forest-carbon project.", None, None),
    ("biomass", "Regional mean biomass estimation with about 30,000 plot samples.", None, 30000),
    # ecosystem_type (window classification; tiny default; flips base on <2K or >5 classes)
    ("ecosystem_type", "Classify the dominant ecosystem type for each tile across a national park.", None, None),
    ("ecosystem_type", "One ecosystem label per 640 m tile; only about 1,200 labeled tiles available.", None, 1200),
    ("ecosystem_type", "Per-tile ecosystem classification with seven ecosystem classes across a basin.", 7, None),
    # vessel_detection (single_moment; tiny; flips base on <2K)
    ("vessel_detection", "Detect ships at sea in a scene, combining radar and optical for night and day.", None, None),
    ("vessel_detection", "Locate individual vessels as points in a busy harbor approach.", None, None),
    ("vessel_detection", "Vessel detection where labeled examples are scarce — about 1,100 annotated ships.", None, 1100),
    # solar_array_detection (detection + period, stationary; default base)
    ("solar_array_detection", "Find and localize solar farms across a state from imagery over the year.", None, None),
    ("solar_array_detection", "Detect utility and rooftop solar arrays — stationary objects — across a region.", None, None),
    # oil_slick (single_moment; S1 primary; tiny)
    ("oil_slick", "Detect oil slicks on the sea surface from a single radar-primary scene.", None, None),
    ("oil_slick", "Localize oil spills as discrete objects, using Sentinel-1 as the main sensor.", None, None),
    # flood (single_moment_with_context before+after; base; flips tiny on >20K)
    ("flood", "Map flood extent after a hurricane using before-and-after imagery; radar is essential under storm clouds.", None, None),
    ("flood", "Per-pixel flooded / not-flooded classification for a single storm event with surrounding context.", None, None),
    ("flood", "Flood mapping with a very large multi-event archive of about 45,000 labeled scenes.", 2, 45000),
    # drought (single_moment_with_context before only; tiny)
    ("drought", "Produce a per-pixel drought index using the preceding six months as context.", None, None),
    ("drought", "Continuous drought-severity regression for a rangeland from optical vegetation signals.", None, None),
    # burn_scar (single_moment_with_context before+after; tiny)
    ("burn_scar", "Map burned area after a wildfire using before-and-after Sentinel-2.", None, None),
    ("burn_scar", "Per-pixel burn-scar segmentation for a single fire event.", None, None),
    ("burn_scar", "Burn-scar mapping in a fire-prone region with a small label set — roughly 700 annotated scenes.", 2, 700),
    # embeddings (no label; tiny; S2 only)
    ("embeddings", "Generate general-purpose feature vectors over an AOI for downstream clustering and similarity search.", None, None),
    ("embeddings", "Extract unsupervised OlmoEarth embeddings for a region so we can cluster land types later — no labels yet.", None, None),
]


def build_items() -> list[dict]:
    items: list[dict] = []
    seen: dict[str, int] = {}
    for preset_key, desc, n_cls, n_smp in SPECS:
        seen[preset_key] = seen.get(preset_key, 0) + 1
        item_id = f"{preset_key}_{seen[preset_key]:02d}"
        expected = expected_for(preset_key, n_cls, n_smp)
        items.append(
            {
                "id": item_id,
                "task_description": desc,
                "task_type": expected["output_type"],
                "preset_key": preset_key,
                "expected": expected,
            }
        )
    return items


def write_split(name: str, items: list[dict]) -> None:
    path = os.path.join(OUT_DIR, name)
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "items.json"), "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)


def main() -> None:
    items = build_items()
    # Deterministic, preset-stratified round-robin so train/val/test each span
    # varied output types AND share the hard (signal-flip / context-mode)
    # variants — the test split must have real headroom, not only easy items.
    by_preset: dict[str, list[dict]] = {}
    for it in items:
        by_preset.setdefault(it["preset_key"], []).append(it)

    # cycle ~ 50% train / 20% val / 30% test, offset per preset so the first
    # (default) variant doesn't always land in train.
    cycle = ["train", "val", "test", "train", "test", "train", "val", "test"]
    rng = random.Random(42)
    buckets: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for offset, key in enumerate(sorted(by_preset)):
        group = by_preset[key]
        rng.shuffle(group)
        for i, it in enumerate(group):
            buckets[cycle[(i + offset) % len(cycle)]].append(it)

    train, val, test = buckets["train"], buckets["val"], buckets["test"]
    # Guarantee val isn't starved on small groups.
    rng.shuffle(test)
    while len(val) < 6 and len(test) > 10:
        val.append(test.pop())
    for split_items in (train, val, test):
        rng.shuffle(split_items)

    write_split("train", train)
    write_split("val", val)
    write_split("test", test)
    print(f"items={len(items)}  train={len(train)} val={len(val)} test={len(test)}")
    print(f"out_dir={OUT_DIR}")
    # Quick coverage readout
    for name, split in (("train", train), ("val", val), ("test", test)):
        types = sorted({it["task_type"] for it in split})
        print(f"  {name}: {len(split)} items, output_types={types}")


if __name__ == "__main__":
    main()
