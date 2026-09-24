# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Ziming Qi
"""
write_config.py — Emit OlmoEarth-ready output files from a labels GeoJSON.

Outputs (in --out-dir):

- config.json                            rslearn dataset config (style configurable)
- import.geojson + import.json           Studio import file (dual extension; Windows MIME)
- shards/region_NN.{geojson,json}        auto-split if > --max-per-shard records
- finetune.yaml                          Lightning fine-tune config (only with --finetune)

Two ``--config-style`` choices for ``config.json``:

- ``awf`` (default): three zoom-offset band_sets, single sentinel2 layer with
  ``query_config.PER_PERIOD_MOSAIC``. Mirrors the AWF tutorial cell 9.
- ``production``: twelve per-month layers (sentinel2_l2a_mo01..mo12) with
  ``alias: sentinel2_l2a`` + ``time_offset`` per layer, vector label layer.
  Matches ``allenai/olmoearth_projects:olmoearth_run_data/sample/dataset.json``
  — what ``olmoearth_run`` actually consumes.

Usage:
    python write_config.py labels.geojson out/
    python write_config.py labels.geojson out/ --config-style production
    python write_config.py labels.geojson out/ --finetune --num-classes 9

Stdlib only.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


# Production layer band order, verified against
# allenai/olmoearth_projects:olmoearth_run_data/sample/dataset.json.
# All 12 bands in one band_set (no zoom_offset split), alphabetical with B8A at end.
PRODUCTION_S2_BANDS = [
    "B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08",
    "B09", "B11", "B12", "B8A",
]


def make_rslearn_config(source: str = "pc", harmonize: bool = True) -> dict:
    """Return the AWF-tutorial-style 3-bandset Sentinel-2 rslearn config.

    Mirrors cell 9 of the AWF tutorial verbatim, with the cloud-cover sort key
    adjusted for the chosen data source (Planetary Computer uses
    ``eo:cloud_cover``; Element-84 Earth Search uses ``properties.eo:cloud_cover``).

    For the production layout (12 per-month layers with alias + time_offset)
    used by ``olmoearth_run``, see ``make_production_rslearn_config`` instead.
    """
    if source == "pc":
        class_path = "rslearn.data_sources.planetary_computer.Sentinel2"
        cache = "cache/planetary_computer"
        sort_by = "eo:cloud_cover"
    elif source == "e84":
        class_path = "rslearn.data_sources.earth_search.Sentinel2"
        cache = "cache/earth_search"
        sort_by = "properties.eo:cloud_cover"
    else:
        raise ValueError(f"unknown source {source!r} (use 'pc' or 'e84')")

    return {
        "layers": {
            "label": {
                "type": "raster",
                "band_sets": [{"bands": ["category"], "dtype": "int32"}],
            },
            "sentinel2": {
                "type": "raster",
                "band_sets": [
                    {"bands": ["B02", "B03", "B04", "B08"], "dtype": "uint16"},
                    {
                        "bands": ["B05", "B06", "B07", "B8A", "B11", "B12"],
                        "dtype": "uint16",
                        "zoom_offset": -1,
                    },
                    {
                        "bands": ["B01", "B09"],
                        "dtype": "uint16",
                        "zoom_offset": -2,
                    },
                ],
                "data_source": {
                    "class_path": class_path,
                    "ingest": False,
                    "init_args": {
                        "cache_dir": cache,
                        "harmonize": harmonize,
                        "sort_by": sort_by,
                    },
                    "query_config": {
                        "space_mode": "PER_PERIOD_MOSAIC",
                        "max_matches": 12,
                        "min_matches": 12,
                        "period_duration": "30d",
                    },
                },
            },
        }
    }


def make_production_rslearn_config(
    source: str = "pc",
    harmonize: bool = True,
    n_months: int = 12,
    period_duration_days: int = 30,
) -> dict:
    """Return the production rslearn dataset config used by ``olmoearth_run``.

    Layout: one layer per month (``sentinel2_l2a_mo01..mo<n_months>``), each
    aliased to ``sentinel2_l2a`` so the model sees a single concatenated input.
    Time offsets are spread around 0d in ``period_duration_days`` steps (the
    default 12-month / 30-day layout produces -180d, -150d, ..., +150d).

    Verified structurally against
    ``allenai/olmoearth_projects:olmoearth_run_data/sample/dataset.json``.
    """
    if source == "pc":
        ds_name = "rslearn.data_sources.planetary_computer.Sentinel2"
        cache = "cache/planetary_computer"
        sort_by = "eo:cloud_cover"
    elif source == "e84":
        ds_name = "rslearn.data_sources.earth_search.Sentinel2"
        cache = "cache/earth_search"
        sort_by = "properties.eo:cloud_cover"
    else:
        raise ValueError(f"unknown source {source!r} (use 'pc' or 'e84')")

    half = (n_months // 2) * period_duration_days
    offsets = [-half + i * period_duration_days for i in range(n_months)]

    layers: dict = {"label": {"type": "vector"}}
    for i, offset_days in enumerate(offsets, start=1):
        layers[f"sentinel2_l2a_mo{i:02d}"] = {
            "alias": "sentinel2_l2a",
            "band_sets": [
                {"bands": list(PRODUCTION_S2_BANDS), "dtype": "uint16"}
            ],
            "data_source": {
                "cache_dir": cache,
                "duration": f"{period_duration_days}d",
                "harmonize": harmonize,
                "ingest": False,
                "name": ds_name,
                "sort_by": sort_by,
                "time_offset": f"{offset_days}d",
            },
            "type": "raster",
        }

    return {"layers": layers}


def write_studio_import(features: list[dict], out_dir: Path, name: str = "import"):
    """Write features as both .geojson and .json (Windows MIME workaround)."""
    fc = {"type": "FeatureCollection", "features": features}
    geojson_path = out_dir / f"{name}.geojson"
    json_path = out_dir / f"{name}.json"
    with open(geojson_path, "w", encoding="utf-8") as f:
        json.dump(fc, f)
    shutil.copy2(geojson_path, json_path)
    return geojson_path, json_path


def auto_split_by_region(
    features: list[dict], max_per_shard: int
) -> list[list[dict]]:
    """Split features into shards of at most ``max_per_shard``.

    Sorts by longitude so each shard is geographically coherent — uploading 5 contiguous
    regions to Studio is much more useful than 5 random subsets, both for sanity-checking
    and for avoiding the 10K / 1-hour upload limit on each shard.
    """
    if len(features) <= max_per_shard:
        return [features]

    def lon(feature: dict) -> float:
        coords = (feature.get("geometry") or {}).get("coordinates") or [0, 0]
        return coords[0] if isinstance(coords, list) else 0.0

    sorted_feats = sorted(features, key=lon)
    n_shards = (len(sorted_feats) + max_per_shard - 1) // max_per_shard
    shard_size = (len(sorted_feats) + n_shards - 1) // n_shards
    return [
        sorted_feats[i : i + shard_size]
        for i in range(0, len(sorted_feats), shard_size)
    ]


FINETUNE_YAML = """\
# Auto-generated by olmoearth-data-prep. Edit num_classes, dataset path, and
# epoch counts as needed. The freeze/unfreeze pattern below matches the AWF
# tutorial: train decoder only for {freeze_epochs} epochs, then unfreeze the
# encoder for {unfreeze_epochs} more epochs at 10x learning rate.
model:
  class_path: rslearn.train.lightning_module.RslearnLightningModule
  init_args:
    model:
      class_path: rslearn.models.multitask.MultiTaskModel
      init_args:
        encoder:
          - class_path: rslearn.models.olmoearth_pretrain.model.OlmoEarth
            init_args:
              model_id: OLMOEARTH_V1_NANO
              patch_size: 4
        decoders:
          segment:
            - class_path: rslearn.models.upsample.Upsample
              init_args:
                scale_factor: 4
            - class_path: rslearn.models.conv.Conv
              init_args:
                in_channels: 128
                out_channels: {num_classes}
                kernel_size: 1
                activation:
                  class_path: torch.nn.Identity
            - class_path: rslearn.train.tasks.segmentation.SegmentationHead
    lr: 0.0001
    plateau: true
    plateau_factor: 0.2
    plateau_patience: 2
data:
  class_path: rslearn.train.data_module.RslearnDataModule
  init_args:
    path: {dataset_path}
    inputs:
      sentinel2_l2a:
        data_type: raster
        layers: [sentinel2]
        bands: [B02, B03, B04, B08, B05, B06, B07, B8A, B11, B12, B01, B09]
        passthrough: true
        dtype: FLOAT32
        load_all_item_groups: true
        load_all_layers: true
      label:
        data_type: raster
        layers: [label]
        bands: [category]
        is_target: true
        dtype: INT32
    task:
      class_path: rslearn.train.tasks.multi_task.MultiTask
      init_args:
        tasks:
          segment:
            class_path: rslearn.train.tasks.segmentation.SegmentationTask
            init_args:
              num_classes: {num_classes}
              zero_is_invalid: false
              nodata_value: {nodata_value}
        input_mapping:
          segment:
            label: targets
    batch_size: 4
    num_workers: 4
trainer:
  max_epochs: {total_epochs}
  accelerator: gpu
  devices: 1
  log_every_n_steps: 10
  callbacks:
    - class_path: lightning.pytorch.callbacks.ModelCheckpoint
      init_args:
        dirpath: checkpoints
        save_top_k: 1
        save_last: true
        monitor: val_segment/accuracy
        mode: max
    - class_path: rslearn.train.callbacks.freeze_unfreeze.FreezeUnfreeze
      init_args:
        module_selector: [model, encoder, 0]
        unfreeze_at_epoch: {freeze_epochs}
        unfreeze_lr_factor: 10
"""


def make_finetune_yaml(
    num_classes: int,
    dataset_path: str = "data/dataset",
    freeze_epochs: int = 10,
    unfreeze_epochs: int = 30,
    nodata_value: int = 9,
) -> str:
    return FINETUNE_YAML.format(
        num_classes=num_classes,
        dataset_path=dataset_path,
        freeze_epochs=freeze_epochs,
        unfreeze_epochs=unfreeze_epochs,
        total_epochs=freeze_epochs + unfreeze_epochs,
        nodata_value=nodata_value,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Emit OlmoEarth-ready output files from a labels GeoJSON.",
    )
    parser.add_argument("input", help="Input labels GeoJSON")
    parser.add_argument("out_dir", help="Output directory")
    parser.add_argument(
        "--source",
        default="pc",
        choices=["pc", "e84"],
        help="S2 data source: 'pc' (Planetary Computer) or 'e84' (Element-84 Earth Search)",
    )
    parser.add_argument(
        "--config-style",
        default="awf",
        choices=["awf", "production"],
        help=(
            "rslearn config layout. 'awf' (default): 3 zoom-offset band_sets, "
            "single sentinel2 layer w/ query_config.PER_PERIOD_MOSAIC — mirrors "
            "the AWF tutorial. 'production': 12 per-month layers (mo01..mo12) "
            "with alias='sentinel2_l2a' + time_offset, vector label layer — "
            "matches olmoearth_run's sample/dataset.json."
        ),
    )
    parser.add_argument(
        "--n-months",
        type=int,
        default=12,
        help="Months of imagery (production style only; default 12)",
    )
    parser.add_argument(
        "--max-per-shard",
        type=int,
        default=10000,
        help="Auto-split threshold (default 10000 — Studio's 1-hour upload limit)",
    )
    parser.add_argument(
        "--finetune",
        action="store_true",
        help="Also emit Lightning fine-tune YAML",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        help="Number of classes (required with --finetune)",
    )
    parser.add_argument(
        "--dataset-path",
        default="data/dataset",
        help="Path that finetune.yaml references for the materialized rslearn dataset",
    )
    parser.add_argument("--freeze-epochs", type=int, default=10)
    parser.add_argument("--unfreeze-epochs", type=int, default=30)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. rslearn config
    if args.config_style == "production":
        config = make_production_rslearn_config(
            source=args.source, n_months=args.n_months
        )
    else:
        config = make_rslearn_config(source=args.source)
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    print(
        f"Wrote {out_dir / 'config.json'} "
        f"(source={args.source}, style={args.config_style})"
    )

    # 2. Studio import (dual extension, auto-split if oversized)
    with open(args.input, encoding="utf-8") as f:
        data = json.load(f)
    features = data.get("features", [])

    shards = auto_split_by_region(features, args.max_per_shard)
    if len(shards) == 1:
        gj, js = write_studio_import(shards[0], out_dir, name="import")
        print(f"Wrote {gj.name} and {js.name} ({len(shards[0])} features)")
    else:
        shards_dir = out_dir / "shards"
        shards_dir.mkdir(exist_ok=True)
        for i, shard in enumerate(shards):
            gj, js = write_studio_import(
                shard, shards_dir, name=f"region_{i:02d}"
            )
            print(f"Wrote shards/{gj.name} and shards/{js.name} ({len(shard)} features)")
        print(
            f"Auto-split into {len(shards)} shards "
            f"(max-per-shard={args.max_per_shard}, sorted by longitude)"
        )

    # 3. Lightning fine-tune YAML
    if args.finetune:
        if args.num_classes is None:
            parser.error("--num-classes is required with --finetune")
        yaml_str = make_finetune_yaml(
            num_classes=args.num_classes,
            dataset_path=args.dataset_path,
            freeze_epochs=args.freeze_epochs,
            unfreeze_epochs=args.unfreeze_epochs,
        )
        (out_dir / "finetune.yaml").write_text(yaml_str)
        print(
            f"Wrote {out_dir / 'finetune.yaml'} "
            f"({args.freeze_epochs}+{args.unfreeze_epochs} epochs, "
            f"num_classes={args.num_classes})"
        )


if __name__ == "__main__":
    main()
