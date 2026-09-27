---
name: olmoearth-rslearn
description: Operate the rslearn toolkit — the data + training engine under OlmoEarth (https://github.com/allenai/rslearn). Use whenever the user wants to RUN rslearn rather than just configure it: build/stage an rslearn dataset (the 4-stage add_windows → prepare → ingest → materialize pipeline), pull imagery from a data source (Sentinel-2 / Sentinel-1 / Landsat / NAIP / Google Earth Engine / Planetary Computer / local files), fine-tune or run inference on a foundation model (rslearn model fit / predict with a Lightning YAML), extract embeddings from a frozen encoder, tag a train/val split, or debug an rslearn run (no scenes found, CRS mismatch, cloud-cover sort, slow ingest/materialize). This is the layer that olmoearth-data-prep (which only WRITES config.json/YAML) and olmoearth-embeddings (which only EMITS a notebook) hand off to. Trigger on rslearn CLI questions, dataset config.json + model.yaml shapes, "how do I run/train/materialize ...", windows, band_sets, data_sources.*, RslearnDataModule / SegmentationTask / UNetDecoder, even when "rslearn" isn't named but the user is staging EO data or fine-tuning over it.
---

# OlmoEarth rslearn

[`rslearn`](https://github.com/allenai/rslearn) (AI2, Apache-2.0) is the
remote-sensing ML engine **underneath** OlmoEarth: it defines spatiotemporal
training **windows** (height × width × time boxes), imports raster/vector data
from online or local sources into those windows, fine-tunes a foundation-model
encoder on them, and runs inference on new locations/times. The same
`config.json` that [`olmoearth-data-prep`](../olmoearth-data-prep/SKILL.md)
emits is what rslearn consumes here.

This skill is the **"operate rslearn"** layer. The two adjacent skills stop
short of running anything:

- `olmoearth-data-prep` → **writes** the dataset `config.json` + Lightning YAML.
- **this skill** → **runs** the 4-stage data pipeline, then `model fit` / `predict`.
- `olmoearth-embeddings` → **emits a notebook** that drives a frozen encoder over
  a *materialized* rslearn dataset.

> **Execution reality.** rslearn's heavy stages (`ingest`, `materialize`,
> `model fit`) are minutes-to-hours and network/GPU-bound, and rslearn, GDAL and
> torch are **not** dependencies of this agent. So: use the
> `olmoearth_run_python` tool, when it is enabled, only for **light
> orchestration/inspection** (write a config, open a `Dataset`, tag a split,
> count windows). **Surface** the long CLI commands for the user to run — do not
> try to run `ingest`/`materialize`/`fit` inside the agent loop (they will time
> out).

## When to use

- Building or staging an rslearn **dataset** from a `config.json`
- Pulling imagery from a **data source** (S2 / S1 / Landsat / NAIP / GEE / PC / local)
- **Fine-tuning** (`model fit`) or running **inference** (`model predict`)
- Extracting **embeddings** from a frozen encoder over windows
- Tagging a **train/val split** on a materialized dataset
- **Debugging** an rslearn run (no scenes, CRS, cloud-cover sort, slow stages)

## Mental model: a dataset is a directory + `config.json`

A dataset is a folder with a `config.json` declaring named **layers**. Each
layer is `raster` or `vector`, has `band_sets`, and points at a `data_source`
by `class_path`:

```json
{
  "layers": {
    "sentinel2": {
      "type": "raster",
      "band_sets": [{ "dtype": "uint8", "bands": ["R", "G", "B"] }],
      "data_source": {
        "class_path": "rslearn.data_sources.gcp_public_data.Sentinel2",
        "init_args": { "index_cache_dir": "cache/sentinel2/", "sort_by": "cloud_cover", "use_rtree_index": false }
      }
    }
  }
}
```

Common `data_sources.*` (≈40 exist): `gcp_public_data.Sentinel2`,
`planetary_computer.*` (S2/S1/Landsat/NAIP), `aws_open_data.*`, `usgs.*`,
`google_earth_engine.*`, `xyz_tiles.*`, and `local_files.LocalFiles` for data
you already have. Pick **one** source per layer; `sort_by: cloud_cover` keeps
the least-cloudy scene per window.

## The 4-stage data pipeline

Run **in order** — each stage feeds the next. `--root $DS` is the dataset dir.

```bash
export DS=/path/to/dataset

# 1) add_windows — create the height×width×time boxes to learn over
rslearn dataset add_windows --root $DS --group default --utm --resolution 10 \
  --grid_size 128 --src_crs EPSG:4326 --box=-122.69,47.21,-121.50,47.94 \
  --start 2024-06-01T00:00:00+00:00 --end 2024-08-01T00:00:00+00:00 --name seattle

# 2) prepare — look up which source scenes intersect each window (metadata only)
rslearn dataset prepare --root $DS --workers 32 --batch-size 8

# 3) ingest — download those scenes into the tile cache (network-heavy, slow)
rslearn dataset ingest --root $DS --workers 32 --jobs-per-process 1

# 4) materialize — crop/mosaic the cached tiles to each window's exact bounds
rslearn dataset materialize --root $DS --workers 32
```

What each stage means + its usual failure:

| Stage | Does | Common failure → fix |
|---|---|---|
| `add_windows` | writes window geometries | wrong `--src_crs` → windows land in the ocean; box is `W,S,E,N` lon/lat |
| `prepare` | scene lookup (no download) | **"no scenes found"** → time range too narrow, or AOI outside the source's coverage; widen `--start/--end` |
| `ingest` | downloads scenes (cache) | network/credential errors; rerun is **idempotent** (skips cached) |
| `materialize` | crop/mosaic to windows | runs out of disk; needs `ingest` done first |

## Fine-tune + predict

`model fit/predict` are LightningCLI; the YAML wires an **encoder** (e.g.
OlmoEarth or SatlasPretrain) + **decoder** (`UNetDecoder`) + **task head**
(`SegmentationTask` / `ClassificationTask` / `DetectionTask`) + the
`RslearnDataModule` over your materialized dataset.

```bash
rslearn model fit     --config model.yaml                       # train (GPU)
rslearn model predict --config model.yaml --ckpt_path best.ckpt # inference → writes a result layer
```

Both stage their windows with the same `prepare/ingest/materialize` steps, then
`RslearnWriter` writes predictions back as a dataset layer. These are long/GPU
jobs — **surface them for the user**, don't run them in the agent loop.

## Light Python you CAN run (`olmoearth_run_python`)

How the tool works, so a snippet runs the first time:

- It exists only when the operator started the agent with
  `OLMOEARTH_RUN_PYTHON=1`. If it is not in your tool list, give the snippet to
  the user to run instead.
- Each call runs the snippet in a **fresh, isolated subprocess** (`python -I -c`)
  of the agent's own interpreter, in a throwaway temporary directory. Nothing is
  preloaded and **no state carries over** between calls: write normal `import`
  statements and use absolute paths.
- `rslearn` and `upath` import only if the user installed them in the agent's
  environment; an `ImportError` means they are absent, so surface the snippet.
- There is a wall-clock limit (30 s by default, `OLMOEARTH_RUN_PYTHON_TIMEOUT`)
  and stdout/stderr are truncated at 8,000 characters each; `print()` what you
  need back. Known credential variables are removed from its environment, but
  the subprocess is not network-isolated.

Tagging a split, for example:

```python
from rslearn.dataset import Dataset
from upath import UPath

ds = Dataset(UPath("/path/to/dataset/"))
windows = ds.load_windows(workers=32)
for w in windows:
    w.options["split"] = "val" if w.name.endswith(("0", "1")) else "train"
    w.save()
print(f"{len(windows)} windows; "
      f"{sum(w.options.get('split') == 'val' for w in windows)} val")
```

## Decision: embeddings vs fine-tuning

Defer to [`olmoearth-embeddings`](../olmoearth-embeddings/SKILL.md) for the
call, but the rslearn shortcut: **few labels / fast iteration** → freeze the
encoder, extract embeddings, kNN / linear-probe. **Many labels / max accuracy**
→ `model fit` end-to-end. Both run over the *same* materialized dataset.

## Handoff

1. `olmoearth-data-prep` writes `config.json` + the Lightning YAML (+ checks the 8 pitfalls).
2. **This skill** runs `add_windows → prepare → ingest → materialize`, then `model fit`/`predict`.
3. `olmoearth-embeddings` drives a frozen encoder over the materialized dataset for kNN/linear-probe.

> Provenance: command shapes + the data-source list verified against the rslearn
> repo `/docs` (CoreConcepts, WorkflowOverview, DatasetConfig, DataSources,
> ModelConfig, examples/IntroExample) in 2026-05. rslearn evolves — re-check the
> repo docs if a flag or `class_path` is rejected.
