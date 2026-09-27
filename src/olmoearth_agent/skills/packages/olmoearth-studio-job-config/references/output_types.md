# Model type — Studio's six options

Studio's "What should this model produce?" step frames the choice up front: **fine-tuned models predict from your labeled data; embeddings produce multi-purpose feature vectors.** Then it offers six choices, which split cleanly along two axes: granularity (per-pixel / window / point) and label kind (regression / classification / feature vector). The five fine-tuned types use the label-dependent wizard steps (label field, training data, data split); embeddings skip them.

```
                      regression          classification           feature vector
per pixel             per-pixel reg       per-pixel cls (segm)     —
window-level          window reg          window cls               —
points / boxes        —                   point / bbox detection   —
none of the above     —                   —                        embeddings
```

## When to pick each one

### Per-pixel regression
Predict a continuous numeric value for every pixel.

- Studio's examples: soil moisture, tree height
- Other fits: canopy height, AGB density, NDVI, leaf area index, snow depth, surface temperature
- Requires: a continuous value defined everywhere in the window (point labels need to be rasterized to a mask first — interpolate carefully or this becomes a coverage problem, not a regression problem)
- Patch size default: 640 m (320 m for fine-grained targets or small datasets)

### Per-pixel classification (semantic segmentation)
Predict a discrete class label for every pixel.

- Studio's examples: crop type, land cover, mangroves
- Other fits: water bodies, burn scars, deforestation, urban footprint, ice extent, wetlands
- Requires: full-coverage polygons (not points). If labels are points, this is the wrong output type — switch to detection or window classification.
- Patch size default: 640 m (320 m for fine-grained targets or small datasets)

### Window-level regression
Predict a single numeric value summarising a region.

- Studio's examples: average biomass in a region
- Other fits: tile-mean NDVI, average canopy density, total population estimate, mean LST
- Requires: one number per window. If you have a value per pixel, use per-pixel regression — window regression discards spatial detail.
- Patch size default: 640 m (the patch is the prediction window — match the window the user wants)

### Window-level classification
Predict a single class for a region/tile.

- Example: dominant ecosystem type per tile
- Other fits: dominant crop in a tile, dominant land-cover category, fire risk class (low/med/high), tile-level water presence (binary)
- Patch size default: 640 m (the patch is the prediction tile — match the user's tile size if they name one)

### Point or bounding box detection
Localize and classify discrete objects.

- Studio's examples: vessels, solar arrays, oil slicks
- Other fits: wind turbines, fishing vessels, aircraft, oil-storage tanks, livestock concentrations, fish farms
- Requires: point or bbox labels (not full-coverage masks).
- Patch size default: **1280 m (Studio's own recommendation)**.

### Embeddings
Generate dense feature vectors per pixel/tile, no supervised label.

- Use cases: clustering large regions, similarity search (find areas like this one), pre-computing features for a downstream non-OE model
- Output: one vector per monthly composite per pixel/tile across the configured time span
- Not a model task — there's no "test accuracy" to evaluate. Recommend only if the user explicitly wants embeddings as the deliverable.

## Disambiguation prompts

When the user is vague, ask:

1. "Is your label defined for every pixel in the window, or only at specific locations?"
   - Every pixel → per-pixel
   - Specific points → detection (or window-level if you only care about the aggregate)

2. "Do you want a number or a category?"
   - Number → regression
   - Category → classification

3. "Do you care about *where* in the window each object is, or just whether there's one?"
   - Where → detection
   - Just whether → window-level classification

4. "Are you predicting something on the image, or building features for something else?"
   - Predicting → fine-tuned model
   - Features for something else → embeddings

## Anti-patterns

- **Point labels + per-pixel classification** — the model will learn that 99% of the window is "background", because most pixels have no label. Either rasterize the points into polygons (per-pixel) or switch to point detection.
- **Mixed-coverage polygons + window-level classification** — if a window contains 60% forest and 40% urban, picking one class throws away the signal. Use per-pixel segmentation.
- **Object detection trained on full-coverage masks** — detection expects sparse positives. Convert masks to bboxes via connected-component analysis first.
- **Embeddings when you actually have labels** — if you have labels, fine-tune. Embeddings are for the unlabeled / weakly-labeled / downstream-model case.
