# rslearn dataset config — AWF vs production layouts

There are **two valid layouts** for `config.json`, and they're not interchangeable. `scripts/write_config.py --config-style {awf,production}` emits either.

## Which one do I use?

| Goal | Style |
|------|-------|
| Replicate the AWF tutorial flow / quick fine-tune from a notebook | `awf` (default) |
| Build a project that runs through `olmoearth_run` (the production pipeline) | `production` |

Quick rule: if your destination is `olmoearth_projects/olmoearth_run_data/<project>/`, use `production`. If it's a one-off rslearn dataset for a Lightning fine-tune driven by the AWF tutorial, use `awf`.

## AWF style (`--config-style awf`)

Mirrors cell 9 of the AWF tutorial verbatim. One `sentinel2` layer, three band_sets at native resolutions via `zoom_offset`, with `query_config.PER_PERIOD_MOSAIC` to materialize 12 monthly mosaics. Raster label layer.

```json
{
  "layers": {
    "label": {
      "type": "raster",
      "band_sets": [{ "bands": ["category"], "dtype": "int32" }]
    },
    "sentinel2": {
      "type": "raster",
      "band_sets": [
        { "bands": ["B02", "B03", "B04", "B08"], "dtype": "uint16" },
        { "bands": ["B05", "B06", "B07", "B8A", "B11", "B12"], "dtype": "uint16", "zoom_offset": -1 },
        { "bands": ["B01", "B09"], "dtype": "uint16", "zoom_offset": -2 }
      ],
      "data_source": {
        "class_path": "rslearn.data_sources.planetary_computer.Sentinel2",
        "ingest": false,
        "init_args": {
          "cache_dir": "cache/planetary_computer",
          "harmonize": true,
          "sort_by": "eo:cloud_cover"
        },
        "query_config": {
          "space_mode": "PER_PERIOD_MOSAIC",
          "max_matches": 12,
          "min_matches": 12,
          "period_duration": "30d"
        }
      }
    }
  }
}
```

### AWF knobs

| Knob | Effect |
|------|--------|
| `zoom_offset: -1` | bandset is at 2× the base resolution (20 m if base is 10 m) |
| `zoom_offset: -2` | bandset is at 4× the base resolution (60 m at base 10 m) |
| `harmonize: true` | apply the S2 BOA offset (matters for 2022+ scenes; safe to leave on) |
| `space_mode: PER_PERIOD_MOSAIC` | one mosaic per period — controls temporal density |
| `period_duration: 30d` + `max_matches: 12` | 12 monthly mosaics |
| `min_matches: 12` | reject windows that can't fill all 12 months |

## Production style (`--config-style production`)

Verified structurally against `allenai/olmoearth_projects:olmoearth_run_data/sample/dataset.json`. Different shape entirely: **12 separate per-month layers**, each `aliased` to a single model input, with explicit `time_offset` per layer. Vector label layer.

```json
{
  "layers": {
    "label": { "type": "vector" },
    "sentinel2_l2a_mo01": {
      "alias": "sentinel2_l2a",
      "band_sets": [
        {
          "bands": ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B09", "B11", "B12", "B8A"],
          "dtype": "uint16"
        }
      ],
      "data_source": {
        "cache_dir": "cache/planetary_computer",
        "duration": "30d",
        "harmonize": true,
        "ingest": false,
        "name": "rslearn.data_sources.planetary_computer.Sentinel2",
        "sort_by": "eo:cloud_cover",
        "time_offset": "-180d"
      },
      "type": "raster"
    },
    "sentinel2_l2a_mo02": { "...": "time_offset: -150d, otherwise identical" },
    "...": "mo03..mo12 with time_offset stepping +30d each"
  }
}
```

### Production knobs

| Knob | Effect |
|------|--------|
| `alias: "sentinel2_l2a"` | All 12 monthly layers are read by the model as one concatenated input |
| `time_offset: "-180d"` ... `"+150d"` | Per-layer offset from the window's reference time, in 30d steps |
| `duration: "30d"` | Each layer pulls a 30-day mosaic |
| Single band_set per layer with all 12 bands | No `zoom_offset` split; rslearn handles native resolutions internally |
| `name` (not `class_path`) | Production uses the older `name` key in `data_source` |
| Label layer is just `{type: "vector"}` | No band_sets — labels are vector features attached to windows |

### Production CLI

```bash
# default: 12 months centered (-180d..+150d in 30d steps)
python scripts/write_config.py labels.geojson out/ --config-style production

# fewer months for shorter temporal context
python scripts/write_config.py labels.geojson out/ --config-style production --n-months 6
```

## Data source: Planetary Computer vs Element-84 Earth Search

Same trade-off in both styles. The cloud-cover sort key differs by source:

| Source | Data-source value (AWF: `class_path`, prod: `name`) | `sort_by` value |
|--------|-----------------------------------------------------|-----------------|
| Planetary Computer (PC) | `rslearn.data_sources.planetary_computer.Sentinel2` | `eo:cloud_cover` |
| Element-84 Earth Search (E84) | `rslearn.data_sources.earth_search.Sentinel2` | `properties.eo:cloud_cover` |

Using the wrong key silently sorts by file order instead of cloud cover, returning cloudy mosaics that hurt model accuracy. `--source pc|e84` selects the right one.

## Window geometry

Two standard sizes (same in both styles):

| Use case | Size | Resolution | Coverage |
|----------|------|------------|----------|
| Point-labeled fine-tuning | 63 × 63 | 10 m | 630 × 630 m — matches AWF tutorial; gives encoder 4-px-patch context with overlap |
| Pretraining-tile inference | 256 × 256 | 10 m | 2.56 × 2.56 km — matches OlmoEarth encoder training geometry exactly |

Studio's default 2 km grid is close to the 256 × 256 tile, so Studio tiles feed the encoder cleanly without reprojection-scale surprises.

## End-to-end flow (rslearn CLI, applies to both styles)

```bash
# Add windows from your labels GeoJSON
rslearn dataset add_windows \
    --root path/to/dataset \
    --group train \
    --fname train.geojson \
    --window_size 63 \
    --resolution 10 \
    --utm \
    --start 2023-01-01T00:00:00+00:00 \
    --end 2023-12-31T23:59:59+00:00

# Same for val (--group val, --fname val.geojson)

# Query the data source for matching scenes
rslearn dataset prepare --root path/to/dataset --workers 4

# Download and crop imagery (1-2 h for ~1500 windows)
rslearn dataset materialize --root path/to/dataset --workers 4 --no-use-initial-job
```

`prepare` only queries the catalog; `materialize` actually downloads.

For production projects driven through `olmoearth_run`, use `python -m olmoearth_projects.main olmoearth_run prepare_labeled_windows --project_path <project_dir>` instead — that wraps the rslearn calls and applies the project's labeled-window-preparer pipeline.

## Bandset ordering note (AWF style)

After loading and stacking the AWF-style 3-bandset config, the band order in the resulting tensor is:

```
[B02, B03, B04, B08, B05, B06, B07, B8A, B11, B12, B01, B09]
```

This is **10 m bands first, then 20 m, then 60 m** — not numeric order. The OlmoEarth normalization config (`olmoearth_pretrain.data.normalize.load_computed_config`) is keyed by band name, so as long as you keep the names with the data this works automatically. If you re-order, re-key the normalization too.

The production-style config emits bands in alphabetical order with `B8A` at the end (`[B01, B02, ..., B12, B8A]`) — different from the Lightning input order shown above. This is intentional: dataset.json layer order is independent of the order the model sees, which is set by the Lightning YAML's `bands:` list under `inputs.sentinel2_l2a`.
