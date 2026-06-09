# OlmoEarth Agent Skills Catalog

Detailed spec for the 17 skills the agent ships with. Each skill is an [agentskills.io](https://agentskills.io)-spec package (`SKILL.md` + frontmatter + optional `scripts/`, `references/`, `assets/`, `skill-card.md`, `skill.oms.sig`).

`PLAN.md` is the runtime contract (tools, dataclasses, operational rules). This file is the *skill-layer* contract: what each skill does, why, the tools it composes (from `PLAN.md` §1 or skill-local), and the academic / engineering references that justify it.

**Status:** 17 skills — 16 implemented in-repo + the out-of-process JEPA change engine (v1.0 shipped the originals on 2026-05-31; `olmoearth-litsearch` and the `olmoearth_automate` facet added post-1.0; `olmoearth-negative-sampler` added post-1.1; the JEPA latent-change engine — **out-of-process**, backed by the separate heavy-ML repo [`2imi9/olmoearth-jepa-change`](https://github.com/2imi9/olmoearth-jepa-change) — added post-1.1, now folded into skill #5 `olmoearth-change-detection`; #17 `olmoearth-rslearn` added post-1.2 — a vendored "operate rslearn" SKILL.md plus two in-repo torch-free recommend/validate tools). See `CHANGELOG.md`.

## Existing implementations (upstream)

Three skills already exist in [`2imi9/OlmoEarth-Skills`](https://github.com/2imi9/OlmoEarth-Skills) (updated 2026-05-17). The agent vendors them rather than re-implementing. The three vendored `SKILL.md` packages map one-to-one to catalog skills #1-#3.

| Upstream | Catalog mapping |
|---|---|
| [`olmoearth-data-prep`](https://github.com/2imi9/OlmoEarth-Skills/tree/main/skills/olmoearth-data-prep) | Skill **#1**: 8 prep pitfalls + 7-criteria audit; labels -> Studio import AND rslearn config; recognizes all three schemas (`sample_category` / `es_label` / `oe_labels.{key}`). |
| [`olmoearth-studio-job-config`](https://github.com/2imi9/OlmoEarth-Skills/tree/main/skills/olmoearth-studio-job-config) | Skill **#2**: 14 verified presets + cross-field validator. |
| [`olmoearth-embeddings`](https://github.com/2imi9/OlmoEarth-Skills/tree/main/skills/olmoearth-embeddings) | Skill **#3**: embeddings-vs-fine-tune **guidance** + a Nano/Tiny/Base/Large notebook generator; the in-repo `olmoearth_automate` tool is the one-call automated version that reuses the same decision table. |

Skills #4-#16 are implemented in this repo (see `CHANGELOG.md`).

**Description convention.** Upstream uses trigger-heavy multi-sentence frontmatter ("Use whenever...", "Trigger even when...") because the description is the LLM's routing surface. Match it.

---

## Catalog

| # | Category | Name | What |
|---|---|---|---|
| 1 | Prep | [`olmoearth-data-prep`](#1-olmoearth-data-prep) | Labels (GeoJSON / CSV / Shapefile) -> Studio-importable file (MIME / 10K / multi-metric guards) AND an `rslearn` `dataset.json` + Lightning YAML, with a 7-criteria audit. |
| 2 | Configure | [`olmoearth-studio-job-config`](#2-olmoearth-studio-job-config) | Task description -> Studio wizard answers with 14 presets + cross-field validator. |
| 3 | Configure | [`olmoearth-embeddings`](#3-olmoearth-embeddings) | Embeddings-vs-fine-tune decision: **guidance + a runnable notebook** (you run it), plus the **one-call** `olmoearth_automate` (decide + propose a config; optional HF-dataset introspection). |
| 4 | Run | [`olmoearth-predict`](#4-olmoearth-predict) | The core run primitive: submit / poll / fetch results; pixel-value / features follow. |
| 5 | Run | [`olmoearth-change-detection`](#5-olmoearth-change-detection) | Change detection, two engines: Studio multi-date (>=3) trajectory diff (refuses naive 2-date), and an out-of-process JEPA latent-prediction pixel detector (separate repo). |
| 6 | Run | [`olmoearth-baseline-compare`](#6-olmoearth-baseline-compare) | Studio vs. a baseline foundation model (e.g. AlphaEarth), side-by-side on transfer regions. |
| 7 | Analyze | [`olmoearth-evaluate`](#7-olmoearth-evaluate) | Spatial-block CV + NNDM-LOO over `/prediction-results`. |
| 8 | Analyze | [`olmoearth-similarity`](#8-olmoearth-similarity) | Exact top-K kNN over supplied embeddings (e.g. OlmoEarth Base; FAISS = scale-up); geographic-prior warning. |
| 9 | Analyze | [`olmoearth-uncertainty`](#9-olmoearth-uncertainty) | Repeated pixel-value + Meyer-Pebesma Area of Applicability. |
| 10 | Analyze | [`olmoearth-cloud-mask-audit`](#10-olmoearth-cloud-mask-audit) | CFMask / s2cloudless / Sen2Cor / MAJA ensemble disagreement. |
| 11 | Integrate | [`olmoearth-qgis-bridge`](#11-olmoearth-qgis-bridge) | Tile URLs -> QGIS WMTS + COG with sidecar uncertainty raster. |
| 12 | Integrate | [`olmoearth-data-export`](#12-olmoearth-data-export) | Export Studio projects + predictions to JSON, grouped by project or status. |
| 13 | Report | [`olmoearth-provenance`](#13-olmoearth-provenance) | Manifest wrapper around every API call; emits replay script. |
| 14 | Report | [`olmoearth-case-narrative`](#14-olmoearth-case-narrative) | Stakeholder writeup with live tiles + freshness gate. |
| 15 | Report | [`olmoearth-litsearch`](#15-olmoearth-litsearch) | arXiv + OpenAlex literature search + DOI/arXiv-id resolution to ground citations. |
| 16 | Prep | [`olmoearth-negative-sampler`](#16-olmoearth-negative-sampler) | Presence-only labels -> trainable set: buffered, spatially-thinned (optionally embedding-dissimilar) negative class so the data-prep audit passes. |

### Example briefs

A realistic prompt that routes to each skill - what a user would actually type:

| # | Skill | Example brief |
|---|---|---|
| 1 | `olmoearth-data-prep` | "I have 3,000 field plots as a GeoJSON - get them into Studio without the Windows MIME error, then build the rslearn dataset.json." |
| 2 | `olmoearth-studio-job-config` | "I want per-pixel mangrove classification from Sentinel-2 - fill in the Studio job wizard." |
| 3 | `olmoearth-embeddings` | "I have 150 labels and a Colab T4 - should I fine-tune or use embeddings? Give me a notebook (or just set it up)." |
| 4 | `olmoearth-predict` | "Run a flood-extent prediction over this AOI for last month and return the result tiles." |
| 5 | `olmoearth-change-detection` | "Did forest cover decline across these four quarterly snapshots, or is it just noise?" |
| 6 | `olmoearth-baseline-compare` | "Compare OlmoEarth vs AlphaEarth for land cover in a region where AlphaEarth struggles." |
| 7 | `olmoearth-evaluate` | "My model reports 92% accuracy - re-check it with spatial cross-validation, not random splits." |
| 8 | `olmoearth-similarity` | "Find the 20 patches most similar to this illegal-mining site across the basin." |
| 9 | `olmoearth-uncertainty` | "Flag which parts of my prediction AOI fall outside the model's training distribution." |
| 10 | `olmoearth-cloud-mask-audit` | "My prediction looks wrong over this scene - bad cloud mask or bad model?" |
| 11 | `olmoearth-qgis-bridge` | "Give me a QGIS layer + SLD style for this prediction so I can open it on my desktop." |
| 12 | `olmoearth-data-export` | "Export all my Studio projects and their predictions to JSON, grouped by status." |
| 13 | `olmoearth-provenance` | "Produce a replay script + manifest so an auditor can reproduce this prediction." |
| 14 | `olmoearth-case-narrative` | "Write a stakeholder brief for this karst-vulnerability result with the live map tiles." |
| 15 | `olmoearth-litsearch` | "Find and cite the paper behind the Area-of-Applicability method I used." |
| 16 | `olmoearth-negative-sampler` | "My karst-site labels are presence-only and the audit fails for a missing negative class -- generate background samples." |

---

## Prep

### 1. `olmoearth-data-prep`

**Upstream:** the single vendored [`olmoearth-data-prep`](https://github.com/2imi9/OlmoEarth-Skills/tree/main/skills/olmoearth-data-prep) package (see the [`SKILL.md`](https://raw.githubusercontent.com/2imi9/OlmoEarth-Skills/main/skills/olmoearth-data-prep/SKILL.md) and `scripts/audit.py`, `scripts/write_config.py`). It covers both halves of label prep — Studio import and rslearn config — which earlier catalog versions split into two entries.

**In:** labels as GeoJSON / CSV / Shapefile (+ optional AOIs).
**Out:** a Studio-importable file AND/OR an `rslearn` `dataset.json` + Lightning YAML.

**What.** *Studio import:* `sample_category` schema enforcement, sharding at 10K records, dual file extension to bypass Windows MIME rejection, multi-metric file split when more than one numeric column is present. *rslearn config:* `oe_labels` schema, single-layer 3-bandset (AWF-style) or per-month production layout, the `es_label` rename, watershed AOIs from NLDI / HUC-12, and a 7-criteria audit before training starts. Recognizes all three label schemas (`sample_category` / `es_label` / `oe_labels.{key}`).

**Why.** Onboarding friction is the most repeated dropoff point for case providers. Studio uploads break silently: Windows rejects `.geojson` as `application/octet-stream`, and uploads above 10K records hit the 1-hour timeout. Config friction is likewise a documented barrier to operationalizing geospatial foundation models ([WorldCereal deployment lessons, arXiv:2508.00858](https://arxiv.org/abs/2508.00858), preprint): the `es_label` rename trap and layout confusion cause silent failures that only show up after a multi-hour training run.

**Tools composed.**
- `olmoearth_load_skill` (pulls the vendored `SKILL.md` into context on demand).
- Upstream skill-local: `validate_studio_mime`, `shard_at_10k`, `split_multi_metric`, `write_rslearn_config`, `audit_7_criteria`, `rename_es_label`; `eo.window_tile` / `olmoearth.resolve_to_aoi` (`PLAN.md` §1).
- Composes with skill #16 `olmoearth-negative-sampler`, whose output round-trips back through this skill's audit.

---

## Configure

### 2. `olmoearth-studio-job-config`

**Upstream:** [`2imi9/OlmoEarth-Skills/skills/olmoearth-studio-job-config`](https://github.com/2imi9/OlmoEarth-Skills/tree/main/skills/olmoearth-studio-job-config). Vendor as-is.

**In:** task description.
**Out:** Studio wizard answers.

**What.** Picks output type (per-pixel / window / detection / embeddings), model size (Nano / Tiny / Base), time-frame mode (period vs single-moment-with-context vs single-moment), imagery sources (S2 alone vs +S1), patch size (160 / 320 / 640 / 1280 m). 14 verified presets (crop / mangrove / land cover / soil moisture / biomass / vessel / solar / oil slick / flood / drought / burn scar / embeddings) plus a cross-field validator.

**Why.** Cross-field traps exist: detection with 320 m patch fails silently, Landsat is not yet available as a source, embeddings mode is incompatible with single-moment time frame. Researchers without OlmoEarth-specific intuition iterate on a misconfigured wizard for days. The validator catches the trap before the job is submitted.

**Tools composed.**
- `olmoearth_load_skill`; skill-local: `studio_job_validate`, `apply_preset`, `cross_field_check`.

---

### 3. `olmoearth-embeddings`

**Upstream:** [`2imi9/OlmoEarth-Skills/skills/olmoearth-embeddings`](https://github.com/2imi9/OlmoEarth-Skills/tree/main/skills/olmoearth-embeddings); grounded in the AWF Kenya tutorial's accuracy/time/VRAM table. Handles the small-dataset (<100 samples), limited-compute (T4 / Colab), similarity-search, and "no labels yet" cases.

**In:** task profile (label volume, class balance, target VRAM, target latency); or a free-form `task` / a Hugging Face `hf_dataset` id for the one-call path.
**Out:** an embeddings-vs-fine-tune decision + either a runnable `.ipynb` (the guidance path) or a ready config (the one-call path).

**What.** One embeddings-vs-fine-tune decision, offered two ways. **Guidance + notebook (vendored):** explains the decision grounded in the fine-tuned OlmoEarth benchmark table and emits a parameterized notebook that extracts Nano / Tiny / Base / Large embeddings and trains kNN + linear-probe heads for the **user** to run. **One-call automation (in-repo `olmoearth_automate`):** applies the same precedence rules programmatically — a faithful port of the vendored `recommend.decide`, kept in sync — and proposes an actionable config (model size, classifier head, an embeddings-notebook command, a fine-tune schedule, and a hand-off to skill #2 `olmoearth-studio-job-config`); given a Hugging Face dataset id it reads the row count + ClassLabel classes from the public datasets-server to fill its inputs. Inputs are task metadata only (no geometry), so results are provenance-safe.

**Why.** Embeddings-vs-fine-tune is the most frequent decision point in EO foundation model use. Practitioners without empirical guidance over-fit Large models on small samples or under-use embeddings when they would have sufficed. The measured accuracy / time / VRAM tradeoffs make the choice visible and actionable — a direct counter to AlphaEarth being shipped as annualized embeddings only. **No fabrication:** the one-call path reports `ask_for` when key inputs are missing rather than guessing, and never invents dataset stats.

**Tools composed.**
- `olmoearth_load_skill` (the vendored guidance + notebook generator) and `olmoearth_automate` (the in-repo one-call decision + config + optional HF-dataset introspection).
- `olmoearth.fetch_embedding` (`PLAN.md` §1); logic in `analysis/automate.py` (decision port + `propose_config` + `fetch_hf_dataset_profile`); `system:python` (opt-in subprocess) for light checks only — the generated notebook is meant for the **user** to run (it needs the geospatial/GPU stack the sandbox does not guarantee).

---

## Run

### 4. `olmoearth-predict`

**In:** Studio project + area + model_id + time range + config.
**Out:** `PredictionRef`, status, and result tiles / vectors / metrics.

**What.** Submit a prediction (`POST /predictions`), poll progress (`GET /predictions/{id}`), and fetch results (tiles / vectors / metrics). Pixel-value at points (`/pixel-value`) and feature-search by class (`/features/search`) are defined in `PLAN.md` §1 but not yet built as tools.

**Why.** The Studio API is async with a 5-state status enum (`pending / running / completed / failed / cancelled`) and a `download_token` flow. Without a wrapper, every case study re-implements the polling loop, retry on result-creation failures, and the tile-vs-raw-output decision. This is the core run primitive every other skill depends on.

**Tools composed.**
- `olmoearth_search_predictions`, `olmoearth_submit_prediction`, `olmoearth_get_prediction`, `olmoearth_fetch_results`, `olmoearth_get_prediction_result` (wrapping `PLAN.md` §1 submit / poll / fetch_results).
- `olmoearth_compare_results`: a **quantitative** two-result comparison with no ground truth. Samples both rasters on a grid (pointwise `pixel-value`) over their shared extent and returns mean / mean-absolute difference, RMSE, correlation, and an agreement fraction (regression) or class agreement (classification). A `kind` selects the framing: `cross_model` (default) = two different models over the same area (how much they *agree* -- divergence, not accuracy); `temporal` = one model's output at an earlier (A) vs a later (B) date (net *change* over time, later minus earlier). The returned `narration` (labels + headline) and the in-chat card / difference-scan captions adapt to the kind. This is the numeric counterpart to the side-by-side raster preview; for accuracy against labels use skill #7 `olmoearth-evaluate`.
- `features_search` (`PLAN.md` §1; not yet built as a tool).

**AOI input (draw-in-chat).** `submit_prediction` needs an `area_id`. Rather than ask the user to type a bbox, the agent calls the foundational `olmoearth_request_aoi` tool, which surfaces an interactive map in the web UI; the user **draws** a rectangle or polygon, it is stored as a Studio area (`POST /areas` -> `area_id`), and its `area_id` + bounding box are fed back on the next turn. The same drawn bbox also feeds the JEPA engine of skill #5.

**First skill to ship.** Foundation that #5, #6, #8, #9 all reuse.

---

### 5. `olmoearth-change-detection`

**In:** *(Engine A)* Studio project + area + >=3 time points (refuses 2). *(Engine B)* two co-registered 12-band Sentinel-2 GeoTIFFs (or AOI + 2 dates) + a trained predictor checkpoint.
**Out:** *(A)* change layer + trajectory metrics. *(B)* a georeferenced change-score heatmap GeoTIFF (CRS / transform preserved), percent-area-changed, top-k changed-region GeoJSON, and a summary-stats JSON.

One capability, two complementary engines.

**Engine A — Studio multi-date trajectory diff (in-process).** Turns a dated series of per-date layer summaries into trajectory metrics: step deltas, net change, the largest-change interval, a reversal count, and a trend label. Refuses fewer than 3 dates, because a 2-date diff cannot tell a steady trend from a flood that peaked then receded. Two-date diffs hide gradual drift, and annualized embedding products structurally cannot capture intra-annual signals such as cover-crop transitions, flood peaks, and harvest timing (consistent with [Ma et al. arXiv:2601.00857](https://arxiv.org/abs/2601.00857), preprint). Tool: `olmoearth_change_detect` (composes skill #4).

**Engine B — JEPA latent-prediction detector (out-of-process).** A change detector on **frozen** OlmoEarth embeddings: a lightweight head predicts the time-2 patch embedding from time-1, and the prediction residual is the change score (I-JEPA, Assran et al. CVPR 2023). Runs **out-of-process** in the standalone heavy-ML repo [`2imi9/olmoearth-jepa-change`](https://github.com/2imi9/olmoearth-jepa-change) (PyTorch + CUDA); the agent shells out to `python -m oejc.skill` and consumes the products. A naive embedding-difference (cosine) flags seasonal / illumination shifts as false change; the learned latent forward-model scores only deviation from the "normal" transition. On the OSCD test split (frozen OlmoEarth-v1-Base) it beats the cosine baseline by **+0.22 F1** (0.25 -> 0.47) and ~3x average precision, reaching **unsupervised-SOTA-level F1 0.54, label-free** (robust per-scene threshold) — integrity-verified (9x chance, permutation control, disjoint train / test cities). A Phase-2 gate study found current general VLMs cannot deliver calibrated, localized raster change comparison, so the agent needs this calibrated pixel-level tool. Full results + plan live in the separate repo's `RESULTS.md` / `PLAN.md`.

**Dependency.** Engine B's heavy PyTorch + CUDA + rasterio stack lives in the **separate** repo and is kept **out of this torch-free agent**; the agent invokes it out-of-process (CLI or container). No heavy dependencies are added here. A torch-free subprocess tool-wrapper (so the agent loop can call it live) is the natural follow-up.

---

### 6. `olmoearth-baseline-compare`

**In:** Studio project + area + transfer-region AOI.
**Out:** difference raster + per-metric comparison table.

**What.** Side-by-side OlmoEarth vs a baseline foundation model (AlphaEarth in the worked example) on transfer regions where the baseline is documented to underperform; bring-your-own exported embeddings/predictions for the baseline. Returns a difference raster plus a per-metric comparison table.

**Why.** AlphaEarth-based models are documented to underperform under cross-region transfer ([Ma et al. arXiv:2601.00857](https://arxiv.org/abs/2601.00857), preprint; specific magnitudes may revise). Fine-tuned OlmoEarth's "substantially outperformed" claim lands only if cases include side-by-side runs in regions where AlphaEarth actually fails, not on home-turf regions where it is competitive.

**Dependency.** The AlphaEarth side is data you export from the public Google Earth Engine "Satellite Embedding" dataset; the skill takes those bring-your-own predictions / scores and makes no live GEE or MCP connection. (AlphaEarth embeddings are now public via GEE, so no Trusted-Tester access is required.)

**Tools composed.**
- Skill #4.
- Skill-local: `compare_metrics`, `difference_raster`, exposed as the `olmoearth_baseline_compare` tool (operates on the two metric sets the caller already has).

---

## Analyze

### 7. `olmoearth-evaluate`

**In:** `ResultBundle` + ground truth.
**Out:** per-class metrics + spatial-block CV + NNDM-LOO results.

**What.** Spatial-block cross-validation, leave-one-region-out, nearest-neighbor-distance-matching LOO-CV for true map accuracy.

**Why.** [Ploton et al. (Nat. Commun. 11:4540, 2020)](https://www.nature.com/articles/s41467-020-18321-y) showed that standard random CV can report strong R2 on models with near-zero spatial predictive skill. [Meyer & Pebesma (MEE 12:1620, 2021)](https://besjournals.onlinelibrary.wiley.com/doi/10.1111/2041-210X.13650) formalized this. Without spatial CV, every case narrative the agent produces risks publishing inflated metrics. This is the most-cited methodological pain in EO ML literature.

**Tools composed.**
- `olmoearth_cv_inflation_check`: random-vs-spatial-block test-to-train distance ratio (the fast "is my accuracy inflated?" check).
- `olmoearth_nndm_cv`: NNDM Leave-One-Out folds for an unbiased estimate over the actual prediction area, a faithful pure-Python port of R [CAST](https://github.com/HannaMeyer/CAST)'s `nndm` ([Mila et al., MEE 13:1304, 2022](https://besjournals.onlinelibrary.wiley.com/doi/10.1111/2041-210X.13851)) -- verified against the reference algorithm (exact hand-trace + the `predpoints==trainpoints` invariant).
- `olmoearth_classification_metrics`: accuracy + per-class precision / recall / F1 / IoU.

---

### 8. `olmoearth-similarity`

**In:** query AOI / patch.
**Out:** top-K similar patches with similarity scores + geographic-prior warning.

**What.** Exact brute-force top-K kNN (cosine or Euclidean) over the embeddings the caller supplies (e.g. OlmoEarth Base). FAISS indexing is the scale-up follow-up; exact kNN is correct and dependency-free at the sizes the agent passes in. Returns top-K patches with similarity scores plus a geographic-prior warning when results cluster in the same biome as the query.

**Why.** [NASA Earthdata's Similarity Search tool](https://www.earthdata.nasa.gov/dashboard/services/similarity-search) helps scientists avoid manually inspecting large satellite-imagery regions. Public AlphaEarth demos show similarity search correlates well with independent risk models in match tasks. OlmoEarth has competitive embeddings (Base wins 15 of 24 kNN tasks per [arXiv:2511.13655](https://arxiv.org/abs/2511.13655)) but no skill exposing similarity search. This is the most-requested missing primitive.

**Tools composed.**
- `olmoearth.fetch_embedding` (`PLAN.md` §1) for sourcing embeddings; the skill as built takes bring-your-own embedding vectors (no in-sandbox fetch).
- The kNN search runs in-process inside the `olmoearth_similarity_search` tool handler (pure Python, no heavy deps), not the `system:python` sandbox.
- Skill-local: `similarity_search` (exact brute-force kNN), `geographic_prior_check`.

---

### 9. `olmoearth-uncertainty`

**In:** `PredictionRef` + AOI.
**Out:** confidence map + DI-weighted OOD flag per region.

**What.** Confidence maps plus a Meyer-Pebesma Dissimilarity-Index-weighted out-of-distribution flag. Warns when an AOI lies outside the trained embedding space.

**Why.** Softmax confidence is not OOD detection. Meyer & Pebesma's [Area of Applicability framework](https://besjournals.onlinelibrary.wiley.com/doi/10.1111/2041-210X.13650) (R [`CAST`](https://cran.r-project.org/package=CAST) package, on CRAN since 2018) is implemented across multiple peer-reviewed methods papers but absent from every EO foundation model platform. AlphaEarth's documented transfer failure under domain shift is exactly what AOA would have flagged.

**Tools composed.**
- `olmoearth.pixel_value` (`PLAN.md` §1, new in v0.4).
- Skill-local: `area_of_applicability`, `repeated_sampling`, `ood_flag`.

---

### 10. `olmoearth-cloud-mask-audit`

**In:** STAC items for an AOI + time range.
**Out:** per-tile cloud-mask disagreement raster + bad-mask-vs-bad-model verdict.

**What.** Ensemble of CFMask, s2cloudless, Sen2Cor, MAJA: flags whether a bad prediction is caused by a bad cloud mask or a bad model. Surfaces ensemble disagreement, not a single ground-truth mask.

**Why.** [Skakun et al. CMIX (RSE 274:112990, 2022)](https://doi.org/10.1016/j.rse.2022.112990) compared 10 cloud algorithms across 5 reference datasets and found large disagreement on thin and semi-transparent clouds. HLS v1.3 documentation flags Sentinel-2 cloud-mask quality as an open issue. Cloud-mask accuracy is also known to drop in snow / ice conditions (Yan et al. RSE 2025). Without an audit, practitioners cannot tell whether their model is wrong or just looking at cloud.

**Tools composed.**
- `eo.search_stac`, `eo.sign_assets` (`PLAN.md` §1).
- Skill-local: `fetch_cloud_masks`, `ensemble_disagree`, `verdict_classifier`.

---

## Integrate

### 11. `olmoearth-qgis-bridge`

**In:** `ResultBundle.tile_template` + uncertainty.
**Out:** QGIS WMTS connection file + COG + sidecar uncertainty raster + SLD style.

**What.** Writes a QGIS Processing-toolbox provider and an SLD style derived from the model class metadata. Outputs full-precision GeoTIFF, not just colorized tiles.

**Why.** Existing QGIS deep-learning plugins are documented as inference-only and compute-heavy ([IAMAP, arXiv:2508.00627](https://arxiv.org/abs/2508.00627), preprint). Google shipped a major Earth Engine QGIS plugin upgrade in late 2025 because GIS users will not leave their desktop. Most case providers (NGOs, government analysts, local agencies) live in QGIS or ArcGIS. This skill closes the hot-cold loop.

**Tools composed.**
- `olmoearth.fetch_results` (`PLAN.md` §1).
- Skill-local: `write_sld_style`, `cog_export`, `qgis_provider_xml`.

---

### 12. `olmoearth-data-export`

**REFRAMED 2026-05-28.** The original idea (wire third-party GEE / OSM /
USGS / NOAA MCPs *into* an AOI) needs external MCPs the user must connect
and isn't core. It was reframed to the more useful, self-contained
**export our own Studio data, grouped**, implemented as
`olmoearth_export_data` (projects + predictions -> JSON grouped by project
or status; `tools/export.py`). Verified live. The original
external-MCP-ingest idea is parked (re-open if a case needs it).

The original spec follows for reference:

**In:** AOI + data category (population / OSM context / weather / watershed / etc.).
**Out:** CRS-matched, date-aligned data bundle into the case.

**What.** Bring data from Google Earth Engine, Microsoft Planetary Computer, OSM Overpass, USGS Water Services, NOAA into a case with matched CRS and aligned dates. Depends on user-connected MCPs.

**Why.** Most cases need supplementary data (population, OSM context, weather, watershed) that live in separate platforms; the Cloud-Native Geospatial Forum community mirrored hundreds of TB of AlphaEarth out of Earth Engine for exactly this reason. Without a bridge, the case bounces between systems manually and provenance breaks.

**Dependency.** MCP availability varies by source; verify each connector before listing in a case.

**Tools composed.**
- New MCP servers (registered in `PLAN.md` §4): `eo.gee`, `eo.pc` (already planned), `eo.osm`, `eo.usgs_water`, `eo.noaa`.
- Skill-local: `crs_match`, `date_align`, `mcp_health_check`.

---

## Report

### 13. `olmoearth-provenance`

**In:** every API call (transparent wrap).
**Out:** manifest + single-command replay script.

**What.** Manifest of `prediction_id`, `model_id`, dataset hashes, time window, parameters, tile dates. Emits a single-command replay script so the case can be re-run by a third party.

**Why.** ML pipelines have a documented reproducibility gap ([Samuel et al. arXiv:2006.12117](https://arxiv.org/abs/2006.12117)). [Kedron & Holler (Remote Sensing 14:5471, 2022)](https://doi.org/10.3390/rs14215471) call for benchmark datasets to test replicability in remote sensing specifically. EUDR and REDD+ MRV impose audit requirements that need traceable provenance, not just outputs. Without provenance, NGO and regulator-facing cases cannot be defended.

**Tools composed.**
- Hooks into every `PLAN.md` §1 `olmoearth.*` call via `provenance_middleware` (added to `PLAN.md` §3 as Rule 13).
- Skill-local: `manifest_write`, `replay_script_emit`, `dataset_hash`.

**Operational rule.** Adds rule 13 to `PLAN.md` §3: *every Studio API call emits a manifest entry; sessions that disable provenance refuse to call `save_view`.*

---

### 14. `olmoearth-case-narrative`

**In:** `tile_urls` + provenance manifest.
**Out:** stakeholder writeup with live tiles + freshness gate.

**What.** Stakeholder writeup with embedded live map tiles, citations to dataset hashes, and a freshness gate that refuses to render stale tiles past a configurable window.

**Why.** Practitioners working under operational timelines often need to ship a working solution rather than an optimal one ([WorldCereal lessons, arXiv:2508.00858](https://arxiv.org/abs/2508.00858), preprint). NGO leadership, journalists, and policymakers need outputs tied to map tiles and policy briefs, not Jupyter notebooks. Freshness gating prevents stale-tile reports during disaster response when conditions change within hours.

**Tools composed.**
- Skill #13 (`olmoearth-provenance`) for manifest read.
- `olmoearth.fetch_results` for tile URLs.
- Skill-local: `freshness_gate`, `narrative_template`, `tile_embed`.

---

### 15. `olmoearth-litsearch`

**In:** a free-text query, or a single DOI / arXiv id.
**Out:** curated paper records (id, title, authors, year, venue, doi, arxiv_id, url, cited_by_count; abstract optional), deduped across sources.

**What.** Searches arXiv (Atom API) and OpenAlex (`/works`), and resolves a DOI or arXiv id to one record. Key-free -- OpenAlex is queried via the documented polite-pool `mailto` (set `OLMOEARTH_OPENALEX_MAILTO` to opt in). Round-robin blends the two sources, then dedups on DOI -> arXiv id -> normalized title. Returns bibliographic metadata only -- never full text / PDF bytes and never geometry -- so it is provenance-safe.

**Why.** The catalog already *cites* a body of EO literature (spatial CV, cloud masking, OlmoEarth / AlphaEarth embeddings, WorldCereal), but before this skill the agent could only lean on world-knowledge or hallucinate links -- the exact failure mode Google DeepMind's Science Skills report documents and that its own arXiv/OpenAlex skills fix. This grounds the case-narrative / research workflow in real, citable sources. **No fabrication:** never invents DOIs / ids / titles, reports empty results as empty, and returns a real `url` to cite for every record.

**Tools composed.**
- `olmoearth_litsearch` (unified arXiv + OpenAlex search) + `olmoearth_litsearch_resolve` (DOI / arXiv-id -> one record).
- Logic in `analysis/litsearch.py` (query-build / parse / cross-source dedup); shared `httpx` retry on transient {429, 5xx}.

---

## Prep (continued)

### 16. `olmoearth-negative-sampler`

**In:** a presence-only labels GeoJSON path (`positives_path`); optional `candidates_path`, `negative_label`, `n_negatives`, `exclusion_km`, `min_separation_km`, `contamination_threshold`.
**Out:** a combined GeoJSON (positives + generated negatives) written to `out_path`, plus a counts/ranking summary and a `quality` report (no raw coordinates in chat).

**What.** Generates the missing negative/background class for a presence-only label set as **buffered, spatially-thinned pseudo-absences** (Barbet-Massin et al. 2012): candidate background points within `exclusion_km` of any positive are dropped, accepted negatives are kept `min_separation_km` apart, and -- when the inputs carry per-feature `properties.embedding` vectors -- candidates are ranked by environmental *dissimilarity* to the positive centroid (the inverse of skill #8's similarity search). With `contamination_threshold` set, the embedding path also drops candidates whose similarity to *any* positive meets the threshold -- likely **unmapped positives** -- guarding against environmental contamination on top of the spatial buffer. Defaults to a balanced 1:1 set and writes the negatives under the same schema field the positives use (`sample_category` / `es_label` / `oe_labels.category`). Deterministic; no GDAL.

**Why.** A presence-only set is not trainable -- a classifier with no counter-examples predicts the positive class everywhere (the "false positives everywhere" failure, pitfall #8 in `olmoearth-data-prep`). That skill's `audit.py` *detects* the gap (it hard-FAILs `check_negative_class`) but its `--negative-class auto` was deferred, so the agent could previously only report the dataset as unusable and stop. This skill converts that dead-end into a finished artifact: the combined file round-trips straight back through the skill #1 `olmoearth-data-prep` audit and clears the negative-class check. **Honest by construction:** pseudo-absences are a heuristic, never verified absences, so every result carries a `quality` report (nearest-positive distance + similarity-to-positive stats) for the user to judge and tune; the negative label must be one the audit recognizes; and a placement shortfall (buffer / thinning / contamination / extent too tight) is surfaced as a warning rather than silently under-filled.

**Tools composed.**
- `olmoearth_negative_sampler` (file -> file; counts + path returned).
- Logic in `analysis/negative_sampler.py` (buffer + farthest-point/embedding-dissimilarity selection, reusing `evaluation.spatial_cv.haversine_km`); composes the data-prep audit's negative-class contract.

---

## Configure (continued)

### 17. `olmoearth-rslearn`

**In:** a plain-language research goal (+ optional `task`, `num_classes`, `label_range`, `sensor`, `cloudy`, `temporal`, `num_samples`, `model_size`); or, for validation, a parsed dataset `config.json` and/or model `model.yaml`.
**Out:** for *recommend*, a complete explained setup (task + data layout + `encoder -> decoder -> head` + task knobs + fine-tune schedule, each with a one-line *why*, and `ask_for` for missing inputs); for *validate*, `{ok, errors, warnings, checks}`.

**What.** [`rslearn`](https://github.com/allenai/rslearn) (AI2, Apache-2.0) is the data + training engine **under** OlmoEarth. The vendored `olmoearth-rslearn` SKILL.md teaches *running* it (the 4-stage `add_windows -> prepare -> ingest -> materialize` pipeline, then `model fit`/`predict`). On top of that, two **torch-free** in-repo tools let a domain scientist who does **not** know rslearn still set up a correct experiment. `olmoearth_rslearn_recommend` maps a goal to the right rslearn task (segmentation / per-pixel-regression / detection / classification / regression), a sensible OlmoEarth `encoder -> decoder -> head` composition with the channel contract spelled out, the data layout (`data_source` + `space_mode` + `compositing_method` + bands), the task knobs (metrics / loss / `nodata_value` / `scale_factor` from the label range), and a freeze->unfreeze schedule. `olmoearth_rslearn_validate` catches the errors rslearn only surfaces hours into a run: encoder embedding-dim vs decoder `in_channels`, decoder `out_channels` vs `num_classes`, task vs label-type (Segmentation/per-pixel need a raster target; Classification/Detection/Regression need vector), model `inputs.layers`/bands that aren't in the dataset, bad `dtype`/`sort_by`, and the Faster R-CNN background-class **+1** quirk.

**Why.** rslearn's config + `encoder->decoder->head` framework is a wall for the domain scientists OlmoEarth is for — they have a research question and labels, not ML-engineering fluency. These tools turn "read the docs and hand-write YAML" into "describe the goal, get a correct + explained setup, and have it checked before a multi-hour GPU run." **Torch-free by construction:** importing rslearn would pull torch (excluded from this agent), so the logic mirrors rslearn's facts (tasks, model sizes, config enums) as plain data in `analysis/rslearn_advisor.py`, verified against the rslearn repo with a version pointer — heavy training stays **out-of-process** (the JEPA pattern). Pure metadata in/out (no geometry, no network) → provenance-safe.

**Tools composed.**
- `olmoearth_load_skill` (load the vendored "operate rslearn" SKILL.md for the pipeline + `fit`/`predict`).
- `olmoearth_rslearn_recommend` (goal → explained setup; with 2+ `modalities`, also a pre/mid/post **fusion** recommendation), `olmoearth_rslearn_validate` (catch shape / label-type / band errors before training), `olmoearth_rslearn_compose` (emit a full valid finetune `model.yaml`; with 2+ `modalities` it emits a **multi-source fusion** config — `mid` = one OlmoEarth encoder fed every modality and fused internally via cross-modal attention, the OlmoEarth-native default; `cross_attention` / `pre` / `post` return grounded guidance), and `olmoearth_rslearn_diagnose` (parse a failing `prepare`/`ingest`/`materialize` run → plain-English fixes). Logic in `analysis/rslearn_advisor.py`; rslearn's tasks/models/config schemas — including the multi-input modalities (`OlmoEarth.MODALITY_NAMES`) and `CrossAttentionFusionExtractor` — mirrored as torch-free data, verified against the repo.

---

## Roadmap reference

Implementation order tracks `PLAN.md` §6. First skill to ship was **#4 `olmoearth-predict`** (the foundation that #5, #6, #8, #9 reuse). After that, prioritization is driven by the case-study queue, not this catalog order.

Candidate skills beyond the current 17 (prioritized) are researched in [`docs/eo-skills-shortlist.md`](docs/eo-skills-shortlist.md); its build-first pick, `olmoearth-negative-sampler`, shipped as #16.

## Adding a skill

A new skill is one PR with:
1. A folder under `skills/<skill-name>/` containing `SKILL.md` + `skill-card.md` + `scripts/` + optional `references/`, `assets/`. For skills already in [`2imi9/OlmoEarth-Skills`](https://github.com/2imi9/OlmoEarth-Skills), vendor the folder verbatim (git submodule or copy + provenance note in `skill-card.md`) rather than re-implementing.
2. Any new global tools added to `PLAN.md` §1.
3. A `tests/skills/test_<skill_name>.py` exercising the skill end-to-end against the live Studio API (mark with `@pytest.mark.integration`).
4. An entry in this `SKILLS.md` matching the per-skill template above.
5. A `CHANGELOG.md` line under `### Added`.

See [`CONTRIBUTING.md` §3](CONTRIBUTING.md#3-branch-and-pr-workflow) for branch naming.

## Vendoring policy

[`2imi9/OlmoEarth-Skills`](https://github.com/2imi9/OlmoEarth-Skills) is canonical for skills #1-#3; agent vendors at a pinned commit. Decide in the first skill PR:
- **A. Git submodule**: track an upstream SHA; cleanest provenance, harder for casual contributors.
- **B. Copy + provenance in `skill-card.md`**: simpler, drift risk if the bump is forgotten.
