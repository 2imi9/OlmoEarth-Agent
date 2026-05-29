# OlmoEarth Studio Job Config Recommender

OlmoEarth Studio's "new job" wizard has six decisions stacked on top of each other (output type → model size → label field → time frame → imagery sources → patch size), each with two-to-six options and non-obvious tradeoffs. This skill takes a plain-English task ("predict mangrove extent in Indonesia from S2", "detect oil slicks", "estimate biomass per region") and returns a filled-in recommendation with rationale for every field, plus the rslearn-side knobs the wizard implies but doesn't show.

## Quick decision rules (easy-to-miss steps)

The per-field sections below remain the source of truth. These short rules just make explicit the steps the wizard most often gets wrong — apply them, then justify in the `rationale`.

- **Model size — pick the section default, then ADJUST for data volume (the most-missed step).** After choosing the default model for the task, look at the stated dataset size: if there are **more than ~20,000 labeled samples, step the choice DOWN to `tiny`** (abundant data lets a smaller, faster model match Base); if there are **fewer than ~2,000 samples, or more than 5 classes, step UP to `base`** (representation quality dominates when data is small or complex). If no count is given, keep the default. Do not skip this adjustment.

- **`start_months` — default to `[1]`, never all twelve by reflex.** Use the growing-season start for crops (`[3,4]` Northern hemisphere, `[9,10]` Southern), mid-year `[6]` for peak-canopy properties (tree height, biomass), and the full `[1,2,3,4,5,6,7,8,9,10,11,12]` only for targets that are stable year-round (e.g. mangrove).

- **Context months (`single_moment_with_context`).** before → soil moisture `3`, drought `6`, flood `1`, burn scar `1`; after → flood `2`, burn scar `1`, post-event cause `1–2`, otherwise `0`. At least one of before/after must be greater than 0. For `single_moment`, set `observation_window_hours` (default `12`).

- **Imagery — `["sentinel2"]` only by default.** Add `"sentinel1"` just when the signal is texture/structure or needs cloud penetration: soil moisture, biomass, flood, oil slick, vessel detection. Do not add S1 for crop type, land cover, mangrove, tree height, ecosystem type, or embeddings.

- **Patch size.** 320 m for per-pixel and window tasks; 640 m for flood or landscape-scale context; 1280 m for detection (Studio's recommendation); 160 m only for narrow features or sparse points.


It bundles one script (`scripts/recommend.py`) that runs standalone and emits a JSON config matching the wizard's structure, plus reference docs for each decision (loaded only when the agent needs to justify a choice).

## When to use this skill

Trigger on any of:

- "I want to predict X — what Studio settings should I use?"
- A user pasting any part of the Studio job wizard ("What should this model produce?", "Which OlmoEarth foundation model should we fine-tune?", "What time frame is important context", "Which satellite imagery sources should the model use?", "How much surrounding area should the model use?")
- Naming a task without choosing a setting: "crop type mapping", "soil moisture", "tree height", "vessel detection", "solar array detection", "oil slick", "mangrove extent", "land cover", "biomass", "deforestation", "flood damage"
- Asking about a single field: "Nano vs Tiny vs Base", "do I need Sentinel-1", "3 vs 6 vs 12 month period", "before/after context offset", "1280m vs 320m window"

If the user is asking *how to prepare the labels* themselves (field names, schema, AOI fetching, audit), route to the sibling skill `olmoearth-data-prep` instead.

## Workflow

The wizard has six fields, decided in this order. Each downstream field depends on the previous one — never recommend in isolation.

### 1. Output type → drives everything else

Match the user's task to one of six outputs. The mapping is verifiable from the example tasks Studio itself names:

| User wants to predict… | Output type |
|------------------------|-------------|
| A number per pixel (soil moisture, tree height, canopy density) | **Per-pixel regression** |
| A category per pixel (crop type, land cover, mangrove vs non-mangrove) | **Per-pixel classification (semantic segmentation)** |
| One number per region/tile (average biomass, mean NDVI) | **Window-level regression** |
| One category per region/tile (ecosystem type in a 320 m tile) | **Window-level classification** |
| Discrete objects at specific points (vessels, solar arrays, oil slicks) | **Point or bounding box detection** |
| Reusable feature vectors for clustering / similarity / downstream model | **Embeddings** |

Rules of thumb when the user is ambiguous:

- "Per pixel" only if the label exists for every pixel — a fully painted mask. If the user has point labels for a continuous quantity, that's still per-pixel regression but only after rasterizing labels to a mask via interpolation (warn).
- If they have polygons of mixed cover, that's per-pixel classification, not window-level.
- Detection ≠ classification of a tile — detection localizes objects within the tile, classification gives one label for the whole tile.
- Embeddings are not a model task — they're a feature-extraction job. Recommend only if the user explicitly wants to feed vectors into something else.

See [`references/output_types.md`](references/output_types.md) for the full mapping and disambiguation prompts to ask the user.

### 2. Foundation model size

Three Vision Transformers, each a different speed/quality tradeoff:

| Model | Params | Pick when |
|-------|--------|-----------|
| **Nano** | ~1.4M | Tight compute budget, simple binary tasks, big AOIs (mangrove yes/no, water vs land), embeddings at scale |
| **Tiny** | ~6.2M | Default for most fine-tuning. Good balance of cost and representation. |
| **Base** | ~90M | Multi-class semantic segmentation (>5 classes), fine-grained detection (small objects), small/imbalanced datasets where representation quality matters more than throughput |

Heuristic: if the user has fewer than ~2K labeled samples or more than 5 classes, recommend **Base** — representation quality dominates. If they have 10K+ samples on a 2–3 class problem, **Tiny** is usually enough. **Nano** is only correct when compute is the binding constraint.

See [`references/model_sizes.md`](references/model_sizes.md) for the param-count vs accuracy curves and the upstream pretraining notes from `allenai/olmoearth_pretrain`.

### 3. Label field

The metadata field on the labels GeoJSON whose values the model learns. This is project-specific — read it from the user's data, don't invent it.

If the user has run the sibling `olmoearth-data-prep` skill, the schema audit already names it (`oe_labels.{key}` for production, `es_label` for Studio export, `sample_category` for Studio import). Confirm with the user that the wizard's "Which label should the model learn to predict?" dropdown shows that exact field.

If the user hasn't prepared data yet, stop here and route to `olmoearth-data-prep` first — pick this field after the schema is settled, not before.

### 4. Time frame — three modes

The wizard offers three temporal modes. Pick by the *nature* of the thing being predicted, not by what imagery happens to be available:

#### Mode A: A period of time
The prediction *describes a span*, not a moment. Use for:
- Crop type for a year
- Mangrove extent for a season
- Ecosystem type for a year
- Land cover annual

Then choose the period length:

| Length | Use for |
|--------|---------|
| **3 months (seasonal)** | Crop growth stages, irrigation cycles, quarterly inference |
| **6 months (half-year)** | Ecosystem state shifts, vegetation structure changes, wet/dry transitions |
| **12 months (annual)** | Land cover, annual ecosystem condition, long-term vegetation patterns |
| **Custom (1–12)** | Anything that aligns to a domain-specific cycle (e.g., 4 months for a sugarcane ratoon) |

Also ask **start month(s)**: which calendar months are valid starting points? Default to all 12 if the user has no preference; restrict (e.g., March/April for Northern-hemisphere agriculture) when they describe a growing-season-aligned task.

#### Mode B: A single moment with before-and/or-after context
The prediction *is about a specific date* but needs surrounding context. Use for:
- Soil moisture from preceding months (before context)
- Flood damage or forest-loss cause from following months (after context)
- Drought indicators (before)
- Post-event change detection (after)

Set before/after independently in monthly intervals up to 12 months each. At least one of before or after must be non-zero.

Use the **offset gap** (in days) when imagery very close to the observation date is noisy — e.g., set a 7-day before-offset for flood mapping to skip the day-of imagery that's saturated by cloud cover from the event itself.

#### Mode C: A single moment
The prediction is about *this image, right now*. Use for transient/moving objects:
- Oil spill detection (slick visible in one scene only)
- Vessel detection (ships move)
- Ship wakes
- Bright transient anomalies

Then set the **observation window** (symmetric, in hours): how close must imagery be to the label's timestamp? Default ±12 h (1 day total); widen to ±48 h or ±60 h only when imagery is sparse (e.g., S1 over a specific region). **The annotated data's timestamps must match the imagery you want to predict on** — pitfall: training labels timestamped to "today" but inference happens on a 3-day-old scene → mode-C mismatch.

See [`references/time_frames.md`](references/time_frames.md) for the full decision tree and worked examples.

### 5. Imagery sources

| Source | Default | Add when |
|--------|---------|----------|
| **Sentinel-2 (optical)** | ✅ Always | Multi-spectral visible + IR. Default for crop, vegetation, land cover, mangrove, embeddings. |
| **Sentinel-1 (radar)** | Optional | Cloudy regions (tropics, monsoon), surface-texture tasks (oil slicks, water roughness, ship wakes, soil moisture). Doesn't always improve accuracy — gives signal but adds training time. |
| **Landsat (optical)** | Not yet available in Studio | Long-term trends or pre-2015 history (Landsat 8 starts 2013; S2 starts 2015). Thermal bands help heat tasks. |

Heuristic: **always start with S2 alone**. Add S1 only when (a) optical is frequently obscured (cloud-cover > 30 % climatology), (b) the target signal is *texture* not *spectrum* (oil, wake, soil moisture), or (c) you've trained S2-only and are residual-debugging. Don't add S1 "just to have more bands" — every added modality slows training meaningfully.

See [`references/imagery_sources.md`](references/imagery_sources.md) for per-task source recommendations.

### 6. Patch size (surrounding area)

How much context the model sees around each label. Four options:

| Size | Per-pixel | Window-level | Detection |
|------|-----------|--------------|-----------|
| **Extra-Small 160 m** | Sparse point labels, narrow features (rivers, roads) | Rare | Rare |
| **Small 320 m** | **Default** for most segmentation | **Default** for ecosystem-type classification | Only for very small objects (~5 m, dense) |
| **Medium 640 m** | Larger contextual features (field-level crop type with neighbor effects) | Region-scale aggregates | Mid-sized objects (vehicles, small vessels) |
| **Large 1280 m** | Landscape-scale labels (broad land cover) | Coarse aggregates | **Default for detection** (per Studio's own recommendation) |

Rules:

- **Detection → 1280 m** unless the user is sure objects are dense and small. Studio explicitly recommends 1280 m.
- **Per-pixel classification → 320 m** unless labels reference inter-field/landscape context.
- **Window-level → 320 m** to match the wizard's own "320 × 320 m region" example unless the user wants coarser regions.
- The window slides at inference time, so larger windows = fewer total predictions (faster but lower spatial resolution).

See [`references/patch_sizes.md`](references/patch_sizes.md) for object-size-vs-patch-size rules and per-task recommendations.

## Recommendation output

The skill's deliverable is a single filled-in config matching the wizard's field names. Use this format (the bundled `scripts/recommend.py` emits the same):

```json
{
  "output_type": "per_pixel_classification",
  "foundation_model": "tiny",
  "label_field": "oe_labels.category",
  "time_frame": {
    "mode": "period",
    "period_months": 12,
    "start_months": [3, 4]
  },
  "imagery_sources": ["sentinel2"],
  "patch_size_m": 320,
  "rationale": {
    "output_type": "Crop type is one class per pixel within field boundaries → semantic segmentation",
    "foundation_model": "9 classes, ~3K labels → Base would be safer but Tiny is the cost-balanced default",
    "time_frame": "Crop type is an annual property; March/April aligns to Northern-hemisphere growing season start",
    "imagery_sources": "Optical-only is sufficient for crop spectral signatures; add S1 later if cloud cover > 30%",
    "patch_size_m": "320 m gives single-field context without smearing across neighbors"
  }
}
```

Always include the `rationale` — the user is making a budget decision (Base costs ~60× Nano), and a recommendation without reasoning isn't actionable.

## Bundled script

```bash
# Recommend from a one-line task description
python scripts/recommend.py "predict mangrove extent in Indonesia from S2"

# Or feed answers to specific wizard fields and let the script fill the rest
python scripts/recommend.py --task "vessel detection" --num-classes 1 --num-samples 8000

# Validate an existing config (catch mode-C with 1280 m for a 5 m vessel, etc.)
python scripts/recommend.py --validate config.json
```

The script is stdlib-only; the heuristics live in a single decision table at the top of the file so they're auditable in one screen.

## Common task presets

See [`references/presets.md`](references/presets.md) for verified-from-Studio-examples configs:

- **Crop type mapping** — per-pixel classification, Tiny, 12-month period, S2, 320 m
- **Mangrove extent** — per-pixel classification, Tiny, 12-month period, S2, 320 m
- **Land cover** — per-pixel classification, Base, 12-month period, S2, 320–640 m
- **Soil moisture** — per-pixel regression, Tiny, single-moment with 3 mo before context, S2 + S1, 320 m
- **Tree height / canopy** — per-pixel regression, Base, 12-month period, S2 (+ S1 optional), 320 m
- **Biomass (region average)** — window-level regression, Base, 12-month period, S2, 640 m
- **Ecosystem type (regional)** — window-level classification, Tiny, 12-month period, S2, 320 m
- **Vessel detection** — point/bbox detection, Tiny, single moment ±12 h, S2 + S1, 1280 m
- **Solar array detection** — point/bbox detection, Base, 12-month period (stable target), S2, 1280 m
- **Oil slick detection** — point/bbox detection, Tiny, single moment ±12 h, S1 + S2, 1280 m
- **Flood damage** — per-pixel classification, Base, single moment with 1–2 mo after context, S2 + S1, 320–640 m
- **Drought monitoring** — per-pixel or window regression, Tiny, single moment with 3 mo before, S2, 320 m
- **Embeddings (general purpose)** — embeddings, Tiny or Base, 12-month period, S2, depends on downstream task

## What this skill does NOT do

- Choose the **label field value** for the user — that's their project metadata. The skill names the field; the user picks the value.
- Generate the labels themselves — that's `olmoearth-data-prep`.
- Run training — Studio runs training. This skill stops at the "Create job" submit button.
- Pick optimizer / LR / batch size — those are Lightning-config knobs, not wizard knobs. See `olmoearth-data-prep --finetune` for the canonical Lightning YAML.
- Tell the user whether the task is *solvable* with their data — only the audit (sibling skill) can do that. This skill assumes labels are already validated.

## Reference docs (loaded on demand)

- [`references/output_types.md`](references/output_types.md) — six output types with disambiguation prompts.
- [`references/model_sizes.md`](references/model_sizes.md) — Nano/Tiny/Base tradeoff curves.
- [`references/time_frames.md`](references/time_frames.md) — period vs single-moment-with-context vs single-moment decision tree.
- [`references/imagery_sources.md`](references/imagery_sources.md) — when adding S1 helps vs hurts.
- [`references/patch_sizes.md`](references/patch_sizes.md) — object-size-to-patch-size rules.
- [`references/presets.md`](references/presets.md) — verified configs for ~12 common tasks.

## Bundled script (works standalone, no skill imports)

- [`scripts/recommend.py`](scripts/recommend.py) — task description → filled config JSON; also validates an existing config.
