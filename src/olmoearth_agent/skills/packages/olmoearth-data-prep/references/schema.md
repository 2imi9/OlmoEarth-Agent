# OlmoEarth Studio schema literals

There are **three verified schemas** in play and they're not interchangeable — confusing them is pitfall #1. Each comes from a different point in the OE pipeline.

## 1. Studio import schema (verified)

What you upload TO Studio for annotation review. Verified against the official `OlmoEarth_sample_file.geojson`:

```json
{
  "type": "Feature",
  "geometry": { "type": "Point", "coordinates": [-122.3321, 47.6062] },
  "properties": {
    "task_name": "task_name_value_1",
    "observation_time": "2025-06-01T00:00:00Z",
    "sample_category": "sample_category_value_1",
    "sample_number": 86,
    "sample_true_false": true
  }
}
```

| Field | Type | Role |
|-------|------|------|
| `properties.task_name` | string | Annotation task name |
| `properties.observation_time` | ISO 8601 datetime | When the observation was made |
| `properties.sample_category` | string | The class label (tag-group value in Studio) |
| `properties.sample_number` | integer | Number metadata (project-defined) |
| `properties.sample_true_false` | boolean | Boolean metadata (project-defined) |

**The `sample_*` prefix is the convention for project-defined metadata fields.** `task_name` and `observation_time` are framework fields without the prefix. Metadata types match Studio's annotation form primitives — Text / Number / Boolean / Enum / Multi-Value enum / Tag group.

## 2. Studio export raw schema (verified — needs rename for olmoearth_run)

What Studio's "Export Annotations" tab writes — the **raw** export. Verified against `olmoearth_projects/olmoearth_run_data/sample/annotation_features.geojson`:

```json
{
  "type": "Feature",
  "geometry": { "type": "Point", "coordinates": [-118.15, 33.74] },
  "properties": {
    "es_annotations_task_id": "164679b9-04ed-5b35-b438-9677104067fc",
    "es_start_time": "2024-02-24 05:57:00+00:00",
    "es_end_time": "2024-02-24 05:57:00+00:00",
    "es_label": 1
  }
}
```

| Field | Type | Role |
|-------|------|------|
| `properties.es_annotations_task_id` | UUID string | Links each feature to its parent annotation task |
| `properties.es_start_time` | ISO 8601 string with TZ | Start of the observation window |
| `properties.es_end_time` | ISO 8601 string with TZ | End of the observation window |
| `properties.es_label` | integer | Class label (integer index, not a string) |

`es_*` stands for "Earth System" Studio. **This file is NOT directly consumable by `olmoearth_run`** — `prepare_labeled_windows` reads the production schema (#3 below) via `label_property` from `olmoearth_run.yaml` (e.g. `label_property: "category"` reads `properties.oe_labels.category`).

**To use this with olmoearth_run, rename `es_*` → `oe_labels.{key}` first.** Two options:

- **Canonical**: run [`olmoearth_projects/scripts/oer_annotation_creation.py`](https://github.com/allenai/olmoearth_projects/blob/main/scripts/oer_annotation_creation.py) on your raw GIS data — it writes `properties.oe_labels` directly (line 565: `"oe_labels": labels`).
- **Quick patch** for an existing Studio export: a 5-line `jq`/Python pass mapping `es_label → oe_labels.<label_property>`, `es_start_time → oe_start_time`, `es_end_time → oe_end_time`, `es_annotations_task_id → oe_annotations_task_id`. Zero-index integer labels if your project's class IDs start at 1.

The companion `annotation_task_features.geojson` carries the task geometries that link via `es_annotations_task_id` (which becomes `oe_annotations_task_id` after rename).

## 3. Production schema — `oe_labels.{key}` (verified — what olmoearth_run consumes)

What `olmoearth_run` actually reads, and what the official AWF tutorial uses for direct rslearn fine-tuning. Verified against:

- `olmoearth_projects/scripts/oer_annotation_creation.py` line 565: `"oe_labels": labels` — the canonical script writes labels under `properties.oe_labels`.
- `olmoearth_projects/olmoearth_run_data/sample/olmoearth_run.yaml`: `label_property: "category"` — the labeled-window-preparer reads a property named `category`, which lines up with `oe_labels.category` (not `es_label`).
- AWF tutorial cell that reads labels: `sample["properties"].get("oe_labels", {}).get("category")`.

Structure:

```json
{
  "type": "Feature",
  "geometry": { "type": "Point", "coordinates": [-2.345, 36.789] },
  "properties": {
    "oe_labels": { "category": 1 },
    "oe_start_time": "2024-02-24T05:57:00+00:00",
    "oe_end_time": "2024-02-24T05:57:00+00:00",
    "oe_annotations_task_id": "164679b9-04ed-5b35-b438-9677104067fc"
  }
}
```

`oe_labels` is a dict. The key (e.g. `category`) must match `label_property` in `olmoearth_run.yaml`. Multi-label tasks can carry multiple keys: `{"category": 1, "severity": 3}`.

## Which one do I use?

| Goal | Schema |
|------|--------|
| Upload labels to Studio for annotation review | #1 Studio import (`sample_category`) |
| Use a file Studio just exported, as-is | #2 Studio export raw (`es_label`) — but you can't run `prepare_labeled_windows` on it directly; rename to #3 first |
| Run `olmoearth_run prepare_labeled_windows`, or run rslearn fine-tuning directly | #3 Production / rslearn standard (`oe_labels.{key}`) |

`scripts/audit.py` accepts any of the three and tells you which one it found.

## Anti-patterns the audit flags

| Wrong | #1 Studio import | #2 Studio export raw | #3 Production / rslearn |
|-------|-----------------|----------------------|------------------------|
| `properties.tag` | `properties.sample_category` | `properties.es_label` | `properties.oe_labels.{key}` |
| `properties.label` | same | same | same |
| `properties.class` | same | same | same |
| `properties.category` (top-level, no prefix or nesting) | same | same | nested under `oe_labels` |
| top-level `tag` (sibling of `geometry`) | move under `properties` | move under `properties` | move under `properties` |

## Project-defined metadata fields (Studio import)

The `sample_number` and `sample_true_false` in the verified import sample are placeholders for **project-defined** metadata. Real projects might have `sample_confidence`, `sample_severity`, `sample_observer_id`, etc. — they all share the `sample_*` prefix and map to the Annotation tags configured in Project settings.

If Studio rejects an upload with a "field not recognized" error:

1. Open Project settings → Annotation tags in Studio.
2. Verify the literal field name and prefix there.
3. Rename in your GeoJSON to match exactly.

## OE Studio Admin Guide pointer

The Nov 2025 OE Studio Admin Guide is the authoritative reference for:

- Metadata types: Text / Number / Boolean / Enum / Multi-Value enum / Tag group.
- Tag groups bundle enum tags (example: "Crop type" → grapes / tree fruit / trees / vegetables / water / wheat).
- Project setting: Label type = Point / Polygon / Line.
- Build Task Dataset area selection: Draw on map (< 20,000 km²) / Input coordinates / Upload CSV/GeoJSON.
- Default grid: 2 km — close to OE pretraining tile geometry of 2.56 km × 2.56 km @ 10 m/px.

For the export-side (`es_*`) schema and the labeled-window-preparer pipeline, see [olmoearth_projects/olmoearth_run_data/sample/README.md](https://github.com/allenai/olmoearth_projects/tree/main/olmoearth_run_data).

## Image vs Point/Polygon labels

Studio supports image annotations in addition to Point / Polygon / Line. This skill targets the geometric label flow — image annotation export uses a different schema and isn't covered here.
