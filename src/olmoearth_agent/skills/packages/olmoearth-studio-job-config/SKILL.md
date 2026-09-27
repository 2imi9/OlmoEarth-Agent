---
name: olmoearth-studio-job-config
description: Walk OlmoEarth Studio's "new model" wizard end to end (model name, model type, foundation model, label field, training data, data split, temporal context, image sources, surrounding area) from a plain-English task description. Use whenever the user is creating or configuring a Studio model and asking "which model type / fine-tuned vs embeddings / Nano vs Tiny vs Base / per-pixel vs window / how much before-after context / Sentinel-1 or just S2 / how much surrounding area / full data vs a proportion / spatial split or a metadata field" — or pasting fragments of the wizard (e.g. "What should this model produce?", "Which OlmoEarth foundation model should we fine-tune?", "What time frame is important context", "How should your data be split?", "How much surrounding area should the model use?"). Trigger even when "Studio" isn't said explicitly: any "I want to predict X from satellite imagery, what settings should I pick" question in an OlmoEarth / Sentinel-2 context warrants this skill. Studio trains on Ai2's compute — never ask the user about their own CPU/GPU. Pairs with `olmoearth-data-prep` (this skill chooses the config, that skill produces the data the config consumes).
---

# OlmoEarth Studio "New Model" Wizard Recommender

OlmoEarth Studio's "new model" wizard stacks nine decisions before the review screen (model name → model type → foundation model → label field → training data → data split → temporal context → image sources → surrounding area), each with non-obvious tradeoffs. This skill takes a plain-English task ("predict mangrove extent in Indonesia from S2", "detect oil slicks", "estimate biomass per region") and returns a filled-in recommendation with rationale for every step.

It bundles one script (`scripts/recommend.py`) that runs standalone and emits a JSON config matching the wizard's structure, plus reference docs for each decision (loaded only when the agent needs to justify a choice).

## Studio trains on Ai2's compute — read this first

Studio models train and run on Ai2's infrastructure. The user never provisions hardware, so:

- **Never ask about the user's local compute** (CPU vs GPU, Colab, T4 vs A100, RAM). It is a non-question for Studio users.
- **Never route a Studio model-creation request to a local training pipeline** (rslearn self-run, a local embeddings notebook). Those are for users who explicitly say they want to run training themselves, outside Studio.
- **Foundation model size is a quality vs Studio-job cost/speed tradeoff** — a bigger model costs more to train and is slower at inference *on Ai2's side*, not "do you have a GPU".
- **Embeddings vs fine-tuned is just the Model type step** (step 2 below), not a separate local-vs-cloud decision.

## Quick decision rules (easy-to-miss steps)

The per-step sections below remain the source of truth. These short rules just make explicit the steps the wizard most often gets wrong — apply them, then justify in the `rationale`.

- **Foundation model — pick the section default, then ADJUST for data volume (the most-missed step).** After choosing the default model for the task, look at the stated dataset size: if there are **more than ~20,000 labeled samples, step the choice DOWN to `tiny`** (abundant data lets a smaller, faster model match Base); if there are **fewer than ~2,000 samples, or more than 5 classes, step UP to `base`** (representation quality dominates when data is small or complex). If no count is given, keep the default. Do not skip this adjustment.
- **`start_months` — default to `[1]`, never all twelve by reflex.** Use the growing-season start for crops (`[3,4]` Northern hemisphere, `[9,10]` Southern), mid-year `[6]` for peak-canopy properties (tree height, biomass), and the full `[1,2,3,4,5,6,7,8,9,10,11,12]` only for targets that are stable year-round (e.g. mangrove).
- **Context months (`single_moment_with_context`).** before → soil moisture `3`, drought `6`, flood `1`, burn scar `1`; after → flood `2`, burn scar `1`, post-event cause `1–2`, otherwise `0`. At least one of before/after must be greater than 0. For `single_moment`, set `observation_window_hours` (default `12`).
- **Imagery — `["sentinel2"]` only by default.** Add `"sentinel1"` just when the signal is texture/structure or needs cloud penetration: soil moisture, biomass, flood, oil slick, vessel detection. Do not add S1 for crop type, land cover, mangrove, tree height, ecosystem type, or embeddings.
- **Surrounding area (patch) — Medium 640 m is the wizard's recommendation for most tasks.** Step DOWN to 320 m (or 160 m for narrow linear features) for fine-grained targets or **small (<1,000-label) datasets**; use 1280 m for large-region labels or **any detection task**.
- **Training data — full by default; data split — Spatial by default.** A proportion is only for a quick first iteration on a very large dataset; a metadata-field split only when the user brings a curated train/val/test assignment.

## When to use this skill

Trigger on any of:

- "I want to predict X — what Studio settings should I use?" / "create a model for X" / "configure a model in Studio"
- A user pasting any part of the Studio "new model" wizard ("What should this model produce?", "Which OlmoEarth foundation model should we fine-tune?", "How much of your training data should be used?", "How should your data be split?", "What time frame is important context", "Which satellite imagery sources should the model use?", "How much surrounding area should the model use?")
- Naming a task without choosing a setting: "crop type mapping", "soil moisture", "tree height", "vessel detection", "solar array detection", "oil slick", "mangrove extent", "land cover", "biomass", "deforestation", "flood damage"
- Asking about a single step: "fine-tuned or embeddings", "Nano vs Tiny vs Base", "do I need Sentinel-1", "3 vs 6 vs 12 month period", "before/after context offset", "spatial split or my own field", "1280 m vs 640 m window"

Routing:

- If the user is asking *how to prepare the labels* themselves (field names, schema, AOI fetching, audit), route to the sibling skill `olmoearth-data-prep` instead.
- Only if the user **explicitly** wants to run training themselves outside Studio (their own rslearn pipeline, their own machines) route to a self-run training skill. "Create/configure/build a model" in a Studio context always means this wizard.

## Workflow — the wizard, step by step

Nine steps, decided in this order. Each downstream step depends on the previous ones — never recommend in isolation.

### 1. Model name

Free text, purely organizational (no effect on training). Suggest a descriptive `<target>-<region>-<year-or-version>` name (e.g. `croptype-nile-delta-2024`) and let the user override — it's their catalog.

### 2. Model type — drives everything else

The wizard frames the choice up front: **"Fine-tuned models predict from your labeled data; Embeddings produce multi-purpose feature vectors."** Then it offers six outputs. Match the user's task:

| User wants to predict… | Model type |
|------------------------|------------|
| A number per pixel (soil moisture, tree height, canopy density) | **Per-pixel regression** |
| A category per pixel (crop type, land cover, mangrove vs non-mangrove) | **Per-pixel classification (semantic segmentation)** |
| One number per region/tile (average biomass, mean NDVI) | **Window-level regression** |
| One category per region/tile (dominant ecosystem type in a tile) | **Window-level classification** |
| Discrete objects at specific points (vessels, solar arrays, oil slicks) | **Point or bounding box detection** |
| Reusable feature vectors for clustering / similarity / a downstream model | **Embeddings** |

Rules of thumb when the user is ambiguous:

- "Per pixel" only if the label exists for every pixel — a fully painted mask. If the user has point labels for a continuous quantity, that's still per-pixel regression but only after rasterizing labels to a mask via interpolation (warn).
- If they have polygons of mixed cover, that's per-pixel classification, not window-level.
- Detection ≠ classification of a tile — detection localizes objects within the tile, classification gives one label for the whole tile.
- Embeddings are not a supervised task — they're feature extraction. Recommend only if the user explicitly wants vectors to feed into something else. Embeddings skip the label-dependent steps (label field, training data, data split).

See [`references/output_types.md`](references/output_types.md) for the full mapping and disambiguation prompts to ask the user.

### 3. Foundation model

Three Vision Transformers (parameter counts per `allenai/olmoearth_pretrain`), each a different quality vs Studio-cost/speed tradeoff:

| Model | Params | Pick when |
|-------|--------|-----------|
| **Nano** | ~1.4M | Cheapest/fastest Studio jobs: simple binary tasks, big AOIs (mangrove yes/no, water vs land), embeddings at scale |
| **Tiny** | ~6.2M | Default for most fine-tuning. Good balance of job cost and representation. |
| **Base** | ~90M | Multi-class semantic segmentation (>5 classes), fine-grained detection (small objects), small/imbalanced datasets where representation quality matters more than throughput |

Heuristic: fewer than ~2K labeled samples or more than 5 classes → **Base** (representation quality dominates). 10K+ samples on a 2–3 class problem → **Tiny** is usually enough. **Nano** only when job cost/turnaround is the binding constraint. This is a budget decision on Ai2's compute (Base trains ~60x slower than Nano) — never a question about the user's hardware.

See [`references/model_sizes.md`](references/model_sizes.md) for the tradeoff curves and the upstream pretraining notes.

### 4. Label field

The metadata field on the labels GeoJSON whose values the model learns. This is project-specific — read it from the user's data, don't invent it.

If the user has run the sibling `olmoearth-data-prep` skill, the schema audit already names it (`oe_labels.{key}` for production, `es_label` for Studio export, `sample_category` for Studio import). Confirm with the user that the wizard's "Which label should the model learn to predict?" dropdown shows that exact field.

If the user hasn't prepared data yet, stop here and route to `olmoearth-data-prep` first — pick this field after the schema is settled, not before. (Embeddings: skip this step.)

### 5. Training data — full vs a proportion

How much of the labeled dataset Studio uses for this training run.

- **Full (default).** Use all the training data. Right for nearly every real model.
- **A proportion.** Train on a fraction (e.g. 0.3) for a quick first iteration: sanity-check the config on a very large dataset before paying for the full run, or compare configs cheaply. Always re-train on full data for the model you keep.

### 6. Data split — Spatial is recommended

How Studio assigns train / validation / test.

- **Spatial (recommended).** Studio partitions by location, so validation areas are spatially separate from training areas. This is the default because neighboring EO samples are nearly identical (spatial autocorrelation) — a random split leaks neighbors across sets and inflates validation accuracy.
- **Use a metadata field.** A property on each labeled feature whose values assign `train` / `val` / `test`. Pick this only when the user brings a curated split — e.g. folds engineered by `olmoearth-data-prep`'s spatial cross-validation, or a split that must stay identical across tools/runs.

See [`references/data_split.md`](references/data_split.md) for the leakage rationale and anti-patterns. (Embeddings: skip this step.)

### 7. Temporal context — three modes

The wizard's "What time frame is important context for this model?" step. Pick by the *nature* of the thing being predicted, not by what imagery happens to be available:

#### Mode A: A period of time
The prediction *describes a span*, not a moment. Use for crop type for a year, mangrove extent for a season, ecosystem type for a year, annual land cover — and stable landscape properties in general.

| Length | Use for |
|--------|---------|
| **3 months (seasonal)** | Crop growth stages, irrigation cycles, quarterly inference |
| **6 months (half-year)** | Ecosystem state shifts, vegetation structure changes, wet/dry transitions |
| **12 months (annual)** | Land cover, annual ecosystem condition, stable landscape properties |
| **Custom (1–12)** | Anything that aligns to a domain-specific cycle (e.g., 4 months for a sugarcane ratoon) |

Also ask **start month(s)**: which calendar months are valid starting points? Restrict to the growing season for agriculture; the full 12 only for year-round-stable targets (see the quick rules).

#### Mode B: A single moment with before-and/or-after context
The prediction *is about a specific date* but needs surrounding imagery on a timeline: soil moisture from preceding months (before), flood damage or forest-loss cause from following months (after), drought (before), post-event change (after).

Set before/after independently in monthly intervals **up to 12 months each**; at least one must be non-zero. Use the **offset gap** (in days) to skip imagery too close to the observation date — e.g. a 7-day before-offset for flood mapping skips day-of imagery saturated by the event's own cloud cover.

#### Mode C: A single moment
The prediction is about *this image, right now* — transient or moving targets: oil slicks, vessels, ship wakes, bright transient anomalies.

Set the **observation window** (symmetric, in hours, ±12 to ±60): how close must imagery be to the label's timestamp? Default ±12 h; widen toward ±48–60 h only when imagery is sparse (e.g., a specific S1 footprint). **The annotated data's timestamps must match the imagery you want to predict on** — pitfall: training labels timestamped to "today" but inference on a 3-day-old scene is a mode-C mismatch.

Anti-pitfall worth repeating: a **stable landscape property** (geology, aquifer vulnerability, soil class) is Mode A with a 12-month period — *not* a single moment, even though any one scene "contains" it. The model benefits from the full seasonal stack.

See [`references/time_frames.md`](references/time_frames.md) for the full decision tree and worked examples.

### 8. Image sources

| Source | Default | Add when |
|--------|---------|----------|
| **Sentinel-2 (optical)** | Always | Multi-spectral visible + IR. Default for crop, vegetation, land cover, mangrove, embeddings. |
| **Sentinel-1 (radar)** | Optional | Cloudy regions (tropics, monsoon), surface-texture tasks (oil slicks, water roughness, ship wakes, soil moisture). Doesn't always improve accuracy — gives signal but adds training time. |
| **Landsat (optical)** | Not available yet | Listed in the wizard but not selectable. For pre-2015 history or thermal, fetch Landsat outside Studio. |

Heuristic: **always start with S2 alone**. Add S1 only when (a) optical is frequently obscured (cloud-cover > 30% climatology), (b) the target signal is *texture/structure* not *spectrum* (oil, wake, soil moisture, terrain), or (c) you've trained S2-only and are residual-debugging. Every added modality slows (Studio-side) training meaningfully.

See [`references/imagery_sources.md`](references/imagery_sources.md) for per-task source recommendations.

### 9. Surrounding area (patch size)

How much context the model sees around each label. Four options — **the wizard recommends Medium 640 m for most tasks**:

| Size | Use for |
|------|---------|
| **Extra-small 160 m** | Narrow linear features (rivers, roads); very fine-grained targets |
| **Small 320 m** | Fine-grained targets; **small (<1,000-label) datasets** where a big patch wastes the few labels |
| **Medium 640 m** | **Recommended default for most tasks** — per-pixel, window-level, and embeddings |
| **Large 1280 m** | Labels describing large regions (broad land cover, landscape aggregates) and **detection** (objects need surrounding context) |

Rules:

- **Detection → 1280 m** unless the user is sure objects are dense and small.
- **Most other tasks → 640 m.** Step down to 320 m for fine-grained targets or when the labeled dataset is small (<1,000 labels); 160 m only for narrow features.
- **Window-level**: the patch is the prediction unit — if the user names a tile size, match it; otherwise 640 m.
- The window slides at inference time, so larger patches = coarser output maps (fewer, bigger predictions).

See [`references/patch_sizes.md`](references/patch_sizes.md) for object-size-vs-patch-size rules and per-task recommendations.

### 10. Review and create

The wizard ends with a summary of every answer. Re-check the cross-field traps before "Create model" (or run `scripts/recommend.py --validate`): detection with a small patch, embeddings with a label field, mode-B with zero before AND after, Landsat selected, moving targets on a multi-month period. Then create — Studio queues and trains the model on Ai2's compute; there is nothing for the user to install or provision.

## Recommendation output

The skill's deliverable is a single filled-in config matching the wizard's steps. Use this format (the bundled `scripts/recommend.py` emits the same; model name is the user's free-text choice and not part of the emitted config):

```json
{
  "output_type": "per_pixel_classification",
  "foundation_model": "tiny",
  "label_field": "oe_labels.category",
  "training_data": {"mode": "full"},
  "data_split": {"method": "spatial"},
  "time_frame": {
    "mode": "period",
    "period_months": 12,
    "start_months": [3, 4]
  },
  "imagery_sources": ["sentinel2"],
  "patch_size_m": 640,
  "rationale": {
    "output_type": "Crop type is one class per pixel within field boundaries → semantic segmentation",
    "foundation_model": "9 classes, ~3K labels → Base would be safer but Tiny is the cost-balanced default",
    "training_data": "Full dataset — no reason to subsample at this size",
    "data_split": "Spatial split (Studio's recommendation) avoids neighbor leakage",
    "time_frame": "Crop type is an annual property; March/April aligns to Northern-hemisphere growing season start",
    "imagery_sources": "Optical-only is sufficient for crop spectral signatures; add S1 later if cloud cover > 30%",
    "patch_size_m": "640 m — the wizard's recommended default; field-plus-neighbors context"
  }
}
```

Always include the `rationale` — the user is making a budget decision on Studio's side (Base costs ~60x Nano to train), and a recommendation without reasoning isn't actionable.

## Bundled script

```bash
# Recommend from a one-line task description
python scripts/recommend.py "predict mangrove extent in Indonesia from S2"

# Or feed answers to specific wizard fields and let the script fill the rest
python scripts/recommend.py --task "vessel detection" --num-classes 1 --num-samples 8000

# Validate an existing config (catch detection with a small patch, embeddings with a label field, etc.)
python scripts/recommend.py --validate config.json
```

The script is stdlib-only; the heuristics live in a single decision table at the top of the file so they're auditable in one screen.

## Common task presets

See [`references/presets.md`](references/presets.md) for verified configs:

- **Crop type mapping** — per-pixel classification, Tiny, 12-month period, S2, 640 m
- **Mangrove extent** — per-pixel classification, Tiny, 12-month period, S2, 640 m
- **Land cover** — per-pixel classification, Base, 12-month period, S2, 640 m
- **Soil moisture** — per-pixel regression, Tiny, single-moment with 3 mo before context, S2 + S1, 640 m
- **Tree height / canopy** — per-pixel regression, Base, 12-month period, S2 (+ S1 optional), 640 m
- **Biomass (region average)** — window-level regression, Base, 12-month period, S2 + S1, 640 m
- **Ecosystem type (regional)** — window-level classification, Tiny, 12-month period, S2, 640 m
- **Vessel detection** — point/bbox detection, Tiny, single moment ±12 h, S2 + S1, 1280 m
- **Solar array detection** — point/bbox detection, Base, 12-month period (stable target), S2, 1280 m
- **Oil slick detection** — point/bbox detection, Tiny, single moment ±12 h, S1 + S2, 1280 m
- **Flood damage** — per-pixel classification, Base, single moment with 1 mo before / 2 mo after, S2 + S1, 640 m
- **Drought monitoring** — per-pixel regression, Tiny, single moment with 6 mo before, S2, 640 m
- **Burn scar** — per-pixel classification, Tiny, single moment with 1 mo before / 1 mo after, S2, 640 m
- **Embeddings (general purpose)** — embeddings, Tiny, 12-month period, S2, 640 m (match downstream consumer)

All presets default to full training data + spatial split.

## Worked example: a stable landscape property

Task shape: "score every pixel for a stable landscape property" — e.g. karst-aquifer vulnerability, soil erodibility, geological susceptibility. These trip the wizard in a specific way, so walk it explicitly:

1. **Model type:** per-pixel **regression** if the label is a continuous score (e.g. 0–1), per-pixel **classification** if it's classes — read the label field's type from the user's annotation schema, don't guess.
2. **Foundation model:** **Base** — the signal (terrain texture, drainage patterns, subtle vegetation response) is indirect and subtle; representation quality dominates. Step down only with a very large label set.
3. **Training data / data split:** full + **Spatial** — landscape properties are exactly where neighbor leakage inflates accuracy most.
4. **Temporal context:** **A period of time, 12 months** — the property is stable, so give the model the full seasonal stack. *Not* a single moment: any one scene "contains" the property, but the seasonal sequence is the signal.
5. **Image sources:** S2; add S1 when terrain/structural texture plausibly carries signal.
6. **Surrounding area:** **640 m** (the recommended default) — landscape context matters; step down to 320 m if the label set is small (<1,000).

## What this skill does NOT do

- Ask about, or plan for, the user's compute — Studio trains on Ai2's infrastructure. If the user explicitly wants self-run training outside Studio, that's a different (rslearn-side) workflow, not this wizard.
- Choose the **label field value** for the user — that's their project metadata. The skill names the field; the user picks the value.
- Generate the labels themselves — that's `olmoearth-data-prep`.
- Run training — Studio runs training. This skill stops at the "Create model" button.
- Tell the user whether the task is *solvable* with their data — only the audit (sibling skill) can do that. This skill assumes labels are already validated.

## Reference docs (loaded on demand)

- [`references/output_types.md`](references/output_types.md) — six model types with disambiguation prompts.
- [`references/model_sizes.md`](references/model_sizes.md) — Nano/Tiny/Base tradeoff curves.
- [`references/data_split.md`](references/data_split.md) — spatial vs metadata-field splits, leakage rationale.
- [`references/time_frames.md`](references/time_frames.md) — period vs single-moment-with-context vs single-moment decision tree.
- [`references/imagery_sources.md`](references/imagery_sources.md) — when adding S1 helps vs hurts.
- [`references/patch_sizes.md`](references/patch_sizes.md) — object-size-to-patch-size rules.
- [`references/presets.md`](references/presets.md) — verified configs for ~14 common tasks.

## Bundled script (works standalone, no skill imports)

- [`scripts/recommend.py`](scripts/recommend.py) — task description → filled config JSON; also validates an existing config.
