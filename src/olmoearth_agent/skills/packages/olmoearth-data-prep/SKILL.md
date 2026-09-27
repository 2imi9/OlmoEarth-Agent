---
name: olmoearth-data-prep
description: Convert raw geospatial labels (CSV / GeoJSON / Shapefile / station tables) into OlmoEarth-ready datasets that pass Studio import and avoid the 8 known prep pitfalls. Use whenever the user is preparing labels for OlmoEarth, building an rslearn dataset config, uploading to OlmoEarth Studio or Roger Studio, exporting Studio annotations for olmoearth_run, fetching real watershed AOIs (NLDI / NHD / HUC), splitting train/val for an EO model, troubleshooting Studio import errors (wrong field names, MIME-type rejection, 1-hour timeout), or asking about equal-frequency binning, spatial cross-validation, negative-class generation, or class imbalance for OlmoEarth fine-tuning. Trigger even when "OlmoEarth" isn't said explicitly — rslearn dataset configs, Sentinel-2 windows, Planetary Computer queries, sample_category / es_label / oe_labels schema questions, or watershed-vs-bbox decisions in an EO/labeling context all warrant this skill.
---

# OlmoEarth Data Prep

This skill captures the conventions for turning raw geospatial labels into datasets that OlmoEarth Studio accepts, rslearn fine-tuning consumes, and embeddings workflows can target — without re-learning the failure modes each time.

It bundles three small Python scripts (`scripts/audit.py`, `scripts/fetch_aoi.py`, `scripts/write_config.py`) that run standalone, plus three reference docs (schema, rslearn config template, pitfalls) loaded only when needed.

## When to use this skill

Trigger on any of:

- Converting raw labels (CSV / GeoJSON / Shapefile / station tables) into OlmoEarth-ready files
- Writing or modifying an rslearn `config.json` for Sentinel-1 / Sentinel-2 / Landsat
- Uploading to OlmoEarth Studio (or Roger Studio) and hitting an import error
- Fetching watershed / waterbody / HUC polygons for stations or events
- Splitting labels for fine-tuning (random vs spatial)
- Generating negative-class examples for classification
- Choosing embeddings configuration (foundation model size, resolution, temporal context, source)

## Workflow

Run these in order. Each step has a script, a reference doc, or both.

### 1. Validate schema

Labels must use OlmoEarth's literal field names. There are **three verified schemas** that are not interchangeable:

- **Studio import** (what you upload to Studio): `properties.sample_category` for the class label, plus `properties.task_name` and `properties.observation_time` framework fields and `sample_*`-prefixed project metadata.
- **Studio export raw** (what Studio's "Export Annotations" tab writes): `properties.es_label` (integer class index), plus `es_start_time`, `es_end_time`, `es_annotations_task_id`. **NOT directly consumable by `olmoearth_run`** — rename to schema #3 first via `olmoearth_projects/scripts/oer_annotation_creation.py` or a 5-line `jq` mapping (`es_label → oe_labels.<label_property>`, `es_start_time → oe_start_time`, etc.).
- **Production / rslearn standard** (what `olmoearth_run` actually reads, also what the AWF tutorial uses): `properties.oe_labels.{key}` where `{key}` matches `label_property` in `olmoearth_run.yaml` (typically `category`). Verified against `oer_annotation_creation.py` line 565 (`"oe_labels": labels`) and `sample/olmoearth_run.yaml` (`label_property: "category"`).

Mixing them is pitfall #1 and a frequent cause of silent Studio rejection or label-loading failures. See [`references/schema.md`](references/schema.md) for full details and anti-patterns.

### 2. Attach real AOIs (don't use bboxes)

For hydrology / watershed / event work, never use bboxes — fetch the actual basin polygon. Bboxes include unrelated cover that pollutes embeddings.

```bash
# Upstream basin for a station (precise — NLDI follows the flow network)
python scripts/fetch_aoi.py --nldi-comid 12345 --out basin.geojson

# HUC-12 subbasin (named — useful for event-scale or coarse work)
python scripts/fetch_aoi.py --huc12 020503060101 --out huc.geojson
```

### 3. Write outputs

`scripts/write_config.py` emits everything Studio and rslearn need:

- `config.json` — rslearn dataset config. Two layouts via `--config-style`:
  - `awf` (default): 3 zoom-offset band_sets, single `sentinel2` layer, mirrors the AWF tutorial. Best for tutorial-driven fine-tuning.
  - `production`: 12 per-month layers (`sentinel2_l2a_mo01..mo12`) with `alias: sentinel2_l2a` + per-layer `time_offset`, vector label. Verified structurally identical to `olmoearth_projects/olmoearth_run_data/sample/dataset.json` — what `olmoearth_run` actually consumes.
- `import.geojson` AND `import.json` — Studio import file in both extensions, because `.geojson` gets MIME-rejected as `application/octet-stream` on Windows browsers
- `shards/region_NN.{geojson,json}` — auto-split if record count > 10,000 (Studio's 1-hour upload limit), partitioned by longitude so each shard is geographically coherent
- `finetune.yaml` — Lightning fine-tune config (only with `--finetune`)

```bash
# AWF-tutorial style (default)
python scripts/write_config.py labels.geojson out/ --finetune --num-classes 9

# Production / olmoearth_run style
python scripts/write_config.py labels.geojson out/ --config-style production
```

See [`references/rslearn_config.md`](references/rslearn_config.md) for both layouts and the knobs that matter (PC vs E84 cloud-cover sort key, `zoom_offset` for AWF, `alias` + `time_offset` for production).

### 4. Audit against the 7 OE quality criteria

```bash
python scripts/audit.py out/import.geojson
```

The audit prints a `[PASS]` / `[WARN]` / `[FAIL]` line per criterion and exits non-zero on any FAIL. Run this before every Studio upload.

## The 7 OE quality criteria

From [docs.olmoearth.allenai.org/training-data-considerations](https://docs.olmoearth.allenai.org/training-data-considerations). The audit script checks each:

1. **Volume** — more is better, within reason. Warn under 200 samples; fail under 50.
2. **Schema** — one of `properties.sample_category` (Studio import), `properties.es_label` (Studio export raw — requires rename for olmoearth_run), or `properties.oe_labels.{key}` (production / rslearn standard — what olmoearth_run consumes) present on every feature.
3. **Class distribution** — categories should be roughly balanced; equal-frequency binning is preferred over quantile.
4. **Per-class volume** — every class needs enough samples; warn under 30 per class, fail under 10.
5. **Negative / non-target class** — required to prevent false positives.
6. **Spatial distribution** — labels spread across the AOI, not clustered. Random points within one region inflate apparent accuracy.
7. **Polygon cleanliness** — no self-intersecting geometry; Shapely `is_valid` must be True (skipped if shapely not installed).

## The 8 known pitfalls

Each one cost a debugging session in the Karst, Chesapeake, or Potomac case studies. The skill prevents each procedurally — see [`references/pitfalls.md`](references/pitfalls.md) for cause + fix + reference case.

| # | Pitfall | How this skill prevents it |
|---|---------|----------------------------|
| 1 | Wrong field names (`tag` / `label` / `class` / top-level `category`), confusing the three valid schemas (`sample_category` / `es_label` / `oe_labels.{key}`), OR forgetting to rename `es_*` → `oe_labels.{key}` before `olmoearth_run prepare_labeled_windows` | `audit.py` reports which schema it found; rename via `oer_annotation_creation.py` if you have a raw Studio export |
| 2 | Bbox AOIs instead of real watersheds | `fetch_aoi.py` (NLDI basin + WBD HUC-12) |
| 3 | Studio range-locking when uploading multiple metrics | `write_config.py` emits one import file per metric |
| 4 | `.geojson` rejected as `application/octet-stream` on Windows | `write_config.py` always emits both `.geojson` and `.json` |
| 5 | Quantile binning gave 96/2.5/1.3/0.1 % imbalance | `audit.py` warns on max/min class ratio > 10 |
| 6 | Random splits inflate reported accuracy | `write_config.py` sorts by longitude and assigns every Nth feature to val |
| 7 | 14K+ records timed out Studio (1-hour limit) | `write_config.py` auto-splits at 10K records |
| 8 | Class imbalance with no negative class | `audit.py` fails if no `other` / `background` / `stable` class found |

## Embeddings / fine-tuning configs

Embeddings workflow knobs:

| Parameter | Options |
|-----------|---------|
| Foundation model | Nano (128-dim), Tiny (192-dim), Base (768-dim) |
| Spatial resolution | 10 / 20 / 40 / 80 m |
| Temporal context | 1–12 months |
| Imagery sources | Sentinel-1, Sentinel-2, both |

For fine-tuning, the canonical pattern is:

- Phase 1: freeze encoder, train decoder only — 10 epochs.
- Phase 2: unfreeze, train full model with `unfreeze_lr_factor: 10` — 20–30 epochs.

`scripts/write_config.py --finetune --num-classes N` emits the Lightning YAML that matches this pattern.

## What this skill does NOT do

- Heavy raster I/O — that's `rslearn`'s job.
- Calling `rslearn dataset prepare/materialize` — surface those commands in the workflow but let the user / agent run them.
- Replacing Studio — Studio remains the source of truth for the literal schema; this skill points at it.
- Inferring task type from a CSV — ask the user (classification / regression-as-classification / segmentation / event detection).
- Auto-generating negative class samples — v0 detects the absence; the user must add them via Studio's tag group or manual sampling.

## Reference docs (loaded on demand)

- [`references/schema.md`](references/schema.md) — OE field literals; what's verified vs what to confirm against Studio.
- [`references/rslearn_config.md`](references/rslearn_config.md) — the canonical 3-bandset S2 config, annotated.
- [`references/pitfalls.md`](references/pitfalls.md) — the 8 pitfalls with cause + fix + reference case study.

## Bundled scripts (work standalone, no skill imports)

- [`scripts/audit.py`](scripts/audit.py) — 7-criteria audit on a GeoJSON.
- [`scripts/fetch_aoi.py`](scripts/fetch_aoi.py) — NLDI basin + WBD HUC-12 fetcher.
- [`scripts/write_config.py`](scripts/write_config.py) — emit rslearn config + Studio import + (optional) Lightning fine-tune YAML.
