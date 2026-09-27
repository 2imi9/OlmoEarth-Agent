# Studio model presets

Verified configs for common Earth-observation tasks, matching the "new model" wizard's steps. Each is a starting point — adjust the model size up/down by sample count and class count per [`model_sizes.md`](model_sizes.md), and step the 640 m patch default down to 320 m for small (<1,000-label) datasets per [`patch_sizes.md`](patch_sizes.md).

All presets default to full training data and the Spatial split (Studio's recommendation); see [`data_split.md`](data_split.md) for when a metadata-field split is right. Model name (the wizard's first step) is the user's free text and is not part of the config.

## Crop type mapping

```json
{
  "output_type": "per_pixel_classification",
  "foundation_model": "tiny",
  "label_field": "oe_labels.category",
  "training_data": { "mode": "full" },
  "data_split": { "method": "spatial" },
  "time_frame": { "mode": "period", "period_months": 12, "start_months": [3, 4] },
  "imagery_sources": ["sentinel2"],
  "patch_size_m": 640
}
```

Bump to Base if >5 crop classes or <2K samples. Move start months to Sep/Oct in Southern Hemisphere.

## Mangrove extent

```json
{
  "output_type": "per_pixel_classification",
  "foundation_model": "tiny",
  "label_field": "oe_labels.category",
  "training_data": { "mode": "full" },
  "data_split": { "method": "spatial" },
  "time_frame": { "mode": "period", "period_months": 12, "start_months": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] },
  "imagery_sources": ["sentinel2"],
  "patch_size_m": 640
}
```

Mangrove is stable year-round; any start month is valid. Add S1 in heavily tidal areas where waterline shifts confuse S2.

## Land cover

```json
{
  "output_type": "per_pixel_classification",
  "foundation_model": "base",
  "label_field": "oe_labels.category",
  "training_data": { "mode": "full" },
  "data_split": { "method": "spatial" },
  "time_frame": { "mode": "period", "period_months": 12, "start_months": [1] },
  "imagery_sources": ["sentinel2"],
  "patch_size_m": 640
}
```

Use Base for 5+ classes (urban, water, forest, cropland, bare, …). Broad cover classes benefit from the 640 m landscape context.

## Soil moisture

```json
{
  "output_type": "per_pixel_regression",
  "foundation_model": "tiny",
  "label_field": "oe_labels.moisture_pct",
  "training_data": { "mode": "full" },
  "data_split": { "method": "spatial" },
  "time_frame": { "mode": "single_moment_with_context", "before_months": 3, "after_months": 0, "before_offset_days": 0 },
  "imagery_sources": ["sentinel2", "sentinel1"],
  "patch_size_m": 640
}
```

S1 is the moisture signal (dielectric constant); S2 carries vegetation/NDVI as a proxy. 3 months of before-context captures the drying trend.

## Tree height / canopy density

```json
{
  "output_type": "per_pixel_regression",
  "foundation_model": "base",
  "label_field": "oe_labels.height_m",
  "training_data": { "mode": "full" },
  "data_split": { "method": "spatial" },
  "time_frame": { "mode": "period", "period_months": 12, "start_months": [6] },
  "imagery_sources": ["sentinel2"],
  "patch_size_m": 640
}
```

Mid-year start captures peak leaf-on phenology. Add S1 if forest is dense (radar penetrates canopy).

## Biomass (regional average)

```json
{
  "output_type": "window_regression",
  "foundation_model": "base",
  "label_field": "oe_labels.biomass_mg_per_ha",
  "training_data": { "mode": "full" },
  "data_split": { "method": "spatial" },
  "time_frame": { "mode": "period", "period_months": 12, "start_months": [6] },
  "imagery_sources": ["sentinel2", "sentinel1"],
  "patch_size_m": 640
}
```

640 m patch matches typical biomass plot scale (and the wizard's default). Base because regression with high variance benefits from richer features.

## Ecosystem type (regional)

```json
{
  "output_type": "window_classification",
  "foundation_model": "tiny",
  "label_field": "oe_labels.ecosystem",
  "training_data": { "mode": "full" },
  "data_split": { "method": "spatial" },
  "time_frame": { "mode": "period", "period_months": 12, "start_months": [1] },
  "imagery_sources": ["sentinel2"],
  "patch_size_m": 640
}
```

The patch is the prediction tile — 640 m by default; match the user's tile size if they name one.

## Vessel detection

```json
{
  "output_type": "point_detection",
  "foundation_model": "tiny",
  "label_field": "oe_labels.vessel_type",
  "training_data": { "mode": "full" },
  "data_split": { "method": "spatial" },
  "time_frame": { "mode": "single_moment", "observation_window_hours": 12 },
  "imagery_sources": ["sentinel2", "sentinel1"],
  "patch_size_m": 1280
}
```

S1 catches small vessels and night detections. Move to ±24 h if dataset is sparse. Use Base if vessel-class taxonomy is fine-grained (>3 classes).

## Solar array detection

```json
{
  "output_type": "point_detection",
  "foundation_model": "base",
  "label_field": "oe_labels.array_type",
  "training_data": { "mode": "full" },
  "data_split": { "method": "spatial" },
  "time_frame": { "mode": "period", "period_months": 12, "start_months": [1] },
  "imagery_sources": ["sentinel2"],
  "patch_size_m": 1280
}
```

Solar arrays are stable installations → period mode, not single-moment. Base helps with fine-grained array-type distinctions.

## Oil slick detection

```json
{
  "output_type": "point_detection",
  "foundation_model": "tiny",
  "label_field": "oe_labels.slick",
  "training_data": { "mode": "full" },
  "data_split": { "method": "spatial" },
  "time_frame": { "mode": "single_moment", "observation_window_hours": 12 },
  "imagery_sources": ["sentinel1", "sentinel2"],
  "patch_size_m": 1280
}
```

S1 is primary — slicks suppress wind-driven roughness → dark patch. S2 secondary for daylight confirmation.

## Flood damage / extent

```json
{
  "output_type": "per_pixel_classification",
  "foundation_model": "base",
  "label_field": "oe_labels.flood_state",
  "training_data": { "mode": "full" },
  "data_split": { "method": "spatial" },
  "time_frame": { "mode": "single_moment_with_context", "before_months": 1, "after_months": 2, "before_offset_days": 7, "after_offset_days": 0 },
  "imagery_sources": ["sentinel2", "sentinel1"],
  "patch_size_m": 640
}
```

Before-offset 7 days skips event-day cloud cover. S1 is essential — floods happen under storm clouds. After-context 2 months captures persistence vs recovery.

## Drought monitoring

```json
{
  "output_type": "per_pixel_regression",
  "foundation_model": "tiny",
  "label_field": "oe_labels.drought_index",
  "training_data": { "mode": "full" },
  "data_split": { "method": "spatial" },
  "time_frame": { "mode": "single_moment_with_context", "before_months": 6, "after_months": 0, "before_offset_days": 0 },
  "imagery_sources": ["sentinel2"],
  "patch_size_m": 640
}
```

6 months of before-context captures the dry-down trajectory. NDVI/EVI in S2 carries the signal; S1 not essential.

## Burn scar / fire scar

```json
{
  "output_type": "per_pixel_classification",
  "foundation_model": "tiny",
  "label_field": "oe_labels.burned",
  "training_data": { "mode": "full" },
  "data_split": { "method": "spatial" },
  "time_frame": { "mode": "single_moment_with_context", "before_months": 1, "after_months": 1, "before_offset_days": 0, "after_offset_days": 0 },
  "imagery_sources": ["sentinel2"],
  "patch_size_m": 640
}
```

S2 SWIR band is *the* burn signal. Before/after lets the model see contrast between pre-burn and post-burn surfaces.

## Embeddings (general purpose)

```json
{
  "output_type": "embeddings",
  "foundation_model": "tiny",
  "label_field": null,
  "training_data": null,
  "data_split": null,
  "time_frame": { "mode": "period", "period_months": 12, "start_months": [1] },
  "imagery_sources": ["sentinel2"],
  "patch_size_m": 640
}
```

Embeddings have no label, so the label-dependent steps (label field, training data, data split) don't apply. Tiny (192-dim) is the sweet spot for downstream consumers; Nano (128-dim) is fine for clustering only.

## Embeddings (water / wetland clustering)

```json
{
  "output_type": "embeddings",
  "foundation_model": "tiny",
  "label_field": null,
  "training_data": null,
  "data_split": null,
  "time_frame": { "mode": "period", "period_months": 12, "start_months": [1] },
  "imagery_sources": ["sentinel2", "sentinel1"],
  "patch_size_m": 640
}
```

Add S1 when the downstream task involves water-surface or wetland classification — radar adds the texture signal that pure S2 embeddings lack.
