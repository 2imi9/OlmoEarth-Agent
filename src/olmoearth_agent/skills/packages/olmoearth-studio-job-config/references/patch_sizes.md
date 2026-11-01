# Surrounding area (patch size) — how much context the model sees

Studio's "How much surrounding area should the model use?" step offers four sizes — Extra-small 160 m / Small 320 m / **Medium 640 m (the wizard's recommendation for most tasks)** / Large 1280 m. The wizard renders this as a single dropdown, but it's two questions in disguise:

1. How large is the *thing* you're predicting? (the *object* or *region*)
2. How much surrounding context does the model need to identify it?

A larger patch helps with context but increases per-window compute (linearly with area) and reduces output resolution at inference (the window slides; bigger windows = coarser map).

## Defaults by model type

| Model type | Default patch |
|------------|---------------|
| Per-pixel regression | **640 m** (320 m for fine-grained targets or small datasets) |
| Per-pixel classification (segmentation) | **640 m** (320 m for fine-grained targets or small datasets) |
| Window-level regression | **640 m** — the patch is the prediction tile; match the tile the user wants |
| Window-level classification | **640 m** — same; match the user's tile size if they name one |
| Point / bbox detection | **1280 m** (objects need surrounding context) |
| Embeddings | **640 m**; match downstream consumer if known |

## The step-down rule: small datasets and fine-grained targets

Drop from the 640 m default to **320 m** when:

- The labeled dataset is **small (<1,000 labels)** — a big patch spends the few labels on mostly-context pixels; a smaller patch concentrates signal.
- The target is **fine-grained** — small fields, narrow shoreline bands, features that a 640 m window would drown in context.

Drop further to **160 m** only for narrow linear features (rivers, roads, hedgerows) where fine output resolution matters and context is mostly wasted area.

## The step-up rule: large regions and detection

Use **1280 m** when:

- **Detection** — always the starting point. A vessel is more identifiable with surrounding water + wake + neighbors in view; a solar array with adjacent panel rows; an oil slick with the surrounding smooth-water texture. Drop below 1280 m only when objects are dense (>10 per km²) and small (<5 m).
- **The label describes a large region** — broad land-cover transition zones, landscape-scale aggregates, labels digitized at coarse scale.

## Object size → patch size rule

Roughly, your patch should be **20–80× the object's longest dimension**:

| Object size | Recommended patch |
|-------------|-------------------|
| <5 m (cars, small vessels) | 320–640 m |
| 5–20 m (large vessels, small solar arrays) | 640–1280 m |
| 20–100 m (large solar arrays, ships, oil slicks) | 1280 m |
| >100 m (oil slicks, large concentrations) | 1280 m (Studio's max) |

If the object would barely span more than the patch, the model can't see context — increase the patch.

## Window-level: the patch is the prediction unit

If the user wants predictions for specific tiles ("ecosystem type per 640 m tile"), the patch must equal that tile size. Don't oversize — you'll predict at a coarser resolution than the user wants. For region-aggregate regression ("average biomass per square"), match the patch to the desired aggregation tile.

## Trade-off summary

```
Patch size       Per-window compute   Output resolution   Context   Best for
160 m            1×                   Finest              Minimal   Narrow linear features
320 m            4×                   Fine                Moderate  Fine-grained targets, small datasets
640 m            16×                  Medium              Good      RECOMMENDED default for most tasks
1280 m           64×                  Coarsest            Maximum   Detection, large-region labels
```

Per-window compute scales with the square of patch size (area), but you process *fewer* windows to cover the same AOI, so total job cost scales with AOI area — patch size mostly affects *output resolution* and *context*, not total cost. (All of this runs on Ai2's compute — it shapes the Studio job's cost and the map's resolution, never the user's hardware.)

## Anti-patterns

- **320 m everywhere out of habit** — the wizard recommends Medium 640 m for most tasks; 320 m is the *step-down* for fine-grained or small-dataset cases, not the default.
- **160 m for vessel detection** — the vessel is bigger than the patch; no context. Use 1280 m.
- **1280 m for fine-resolution land cover** — the output map comes out at 1280 m resolution, far coarser than S2's native 10 m. Use 640 m (or 320 m).
- **640 m with a few hundred labels** — mostly-context windows starve a small dataset. Step down to 320 m.
- **320 m for tiny dense objects (parking lots, individual panels)** — objects are barely > a pixel; the model can't learn them. Either go to 1280 m for context, or aggregate to window-level classification.
