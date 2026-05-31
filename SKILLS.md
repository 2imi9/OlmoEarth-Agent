# OlmoEarth Agent Skills Catalog

Detailed spec for the 15 skills the agent ships with. Each skill is an [agentskills.io](https://agentskills.io)-spec package (`SKILL.md` + frontmatter + optional `scripts/`, `references/`, `assets/`, `skill-card.md`, `skill.oms.sig`).

`PLAN.md` is the runtime contract (tools, dataclasses, operational rules). This file is the *skill-layer* contract: what each skill does, why, the tools it composes (from `PLAN.md` §1 or skill-local), and the academic / engineering references that justify it.

**Status:** v1.0, 2026-05-31. All 15 skills implemented; see `CHANGELOG.md`.

## Existing implementations (upstream)

Three skills already exist in [`2imi9/OlmoEarth-Skills`](https://github.com/2imi9/OlmoEarth-Skills) (updated 2026-05-17). The agent vendors them rather than re-implementing.

| Upstream | Catalog mapping |
|---|---|
| [`olmoearth-data-prep`](https://github.com/2imi9/OlmoEarth-Skills/tree/main/skills/olmoearth-data-prep) | Skills **#1 + #2 unified**: 8 prep pitfalls + 7-criteria audit; recognizes all three schemas (`sample_category` / `es_label` / `oe_labels.{key}`). |
| [`olmoearth-studio-job-config`](https://github.com/2imi9/OlmoEarth-Skills/tree/main/skills/olmoearth-studio-job-config) | Skill **#3**: 14 verified presets + cross-field validator. |
| [`olmoearth-embeddings`](https://github.com/2imi9/OlmoEarth-Skills/tree/main/skills/olmoearth-embeddings) | Skill **#4**: embeddings-vs-fine-tune decision + Nano/Tiny/Base/Large notebook. |

Skills #5-#15 are implemented in this repo (see `CHANGELOG.md`). Catalog #1/#2 are unified upstream as `olmoearth-data-prep` (matching the working implementation) but kept as two numbered entries here.

**Description convention.** Upstream uses trigger-heavy multi-sentence frontmatter ("Use whenever…", "Trigger even when…") because the description is the LLM's routing surface. Match it.

---

## Catalog

| # | Category | Name | What |
|---|---|---|---|
| 1 | Prep | [`olmoearth-studio-upload`](#1-olmoearth-studio-upload) | Labels (GeoJSON / CSV / Shapefile) → Studio-importable file with MIME / 10K / multi-metric guards. |
| 2 | Prep | [`olmoearth-rslearn-config`](#2-olmoearth-rslearn-config) | Labels → `rslearn` `dataset.json` + Lightning YAML with 7-criteria audit. |
| 3 | Configure | [`olmoearth-studio-job-config`](#3-olmoearth-studio-job-config) | Task description → Studio wizard answers with 14 presets + cross-field validator. |
| 4 | Configure | [`olmoearth-embeddings`](#4-olmoearth-embeddings) | Task profile → embeddings-vs-fine-tune decision + runnable notebook. |
| 5 | Run | [`olmoearth-predict`](#5-olmoearth-predict) | The core run primitive: submit / poll / pixel-value / features / files. |
| 6 | Run | [`olmoearth-change-detect`](#6-olmoearth-change-detect) | Two-or-more-date trajectory diff (refuses two-date naïve diff). |
| 7 | Run | [`olmoearth-baseline-compare`](#7-olmoearth-baseline-compare) | Studio vs. AlphaEarth side-by-side on transfer regions. |
| 8 | Analyze | [`olmoearth-evaluate`](#8-olmoearth-evaluate) | Spatial-block CV + NNDM-LOO over `/prediction-results`. |
| 9 | Analyze | [`olmoearth-similarity`](#9-olmoearth-similarity) | FAISS over fine-tuned OlmoEarth Base embeddings. |
| 10 | Analyze | [`olmoearth-uncertainty`](#10-olmoearth-uncertainty) | Repeated pixel-value + Meyer-Pebesma Area of Applicability. |
| 11 | Analyze | [`olmoearth-cloud-mask-audit`](#11-olmoearth-cloud-mask-audit) | CFMask / s2cloudless / Sen2Cor / MAJA ensemble disagreement. |
| 12 | Integrate | [`olmoearth-qgis-bridge`](#12-olmoearth-qgis-bridge) | Tile URLs → QGIS WMTS + COG with sidecar uncertainty raster. |
| 13 | Integrate | [`olmoearth-data-export`](#13-olmoearth-data-export) | Export Studio projects + predictions to JSON, grouped by project or status. |
| 14 | Report | [`olmoearth-provenance`](#14-olmoearth-provenance) | Manifest wrapper around every API call; emits replay script. |
| 15 | Report | [`olmoearth-case-narrative`](#15-olmoearth-case-narrative) | Stakeholder writeup with live tiles + freshness gate. |

### Example briefs

A realistic prompt that routes to each skill - what a user would actually type:

| # | Skill | Example brief |
|---|---|---|
| 1 | `olmoearth-studio-upload` | "I have 3,000 field plots as a GeoJSON - get them into Studio without the Windows MIME error." |
| 2 | `olmoearth-rslearn-config` | "Turn my labeled crop polygons + HUC-12 watershed AOIs into an rslearn dataset.json + Lightning YAML." |
| 3 | `olmoearth-studio-job-config` | "I want per-pixel mangrove classification from Sentinel-2 - fill in the Studio job wizard." |
| 4 | `olmoearth-embeddings` | "I have 150 labels and a Colab T4 - should I fine-tune or use embeddings? Give me a notebook." |
| 5 | `olmoearth-predict` | "Run a flood-extent prediction over this AOI for last month and return the result tiles." |
| 6 | `olmoearth-change-detect` | "Did forest cover decline across these four quarterly snapshots, or is it just noise?" |
| 7 | `olmoearth-baseline-compare` | "Compare OlmoEarth vs AlphaEarth for land cover in a region where AlphaEarth struggles." |
| 8 | `olmoearth-evaluate` | "My model reports 92% accuracy - re-check it with spatial cross-validation, not random splits." |
| 9 | `olmoearth-similarity` | "Find the 20 patches most similar to this illegal-mining site across the basin." |
| 10 | `olmoearth-uncertainty` | "Flag which parts of my prediction AOI fall outside the model's training distribution." |
| 11 | `olmoearth-cloud-mask-audit` | "My prediction looks wrong over this scene - bad cloud mask or bad model?" |
| 12 | `olmoearth-qgis-bridge` | "Give me a QGIS layer + SLD style for this prediction so I can open it on my desktop." |
| 13 | `olmoearth-data-export` | "Export all my Studio projects and their predictions to JSON, grouped by status." |
| 14 | `olmoearth-provenance` | "Produce a replay script + manifest so an auditor can reproduce this prediction." |
| 15 | `olmoearth-case-narrative` | "Write a stakeholder brief for this karst-vulnerability result with the live map tiles." |

---

## Prep

### 1. `olmoearth-studio-upload`

**Upstream:** unified with skill #2 inside [`olmoearth-data-prep`](https://github.com/2imi9/OlmoEarth-Skills/tree/main/skills/olmoearth-data-prep). See the [`SKILL.md`](https://raw.githubusercontent.com/2imi9/OlmoEarth-Skills/main/skills/olmoearth-data-prep/SKILL.md) and `scripts/audit.py`, `scripts/write_config.py`. Split-vs-unify decision tracked above.

**In:** labels as GeoJSON / CSV / Shapefile.
**Out:** Studio-importable file.

**What.** `sample_category` schema enforcement, sharding at 10K records, dual file extension to bypass Windows MIME rejection, multi-metric file split when more than one numeric column is present.

**Why.** Onboarding friction is the most repeated dropoff point for case providers. Studio uploads break silently: Windows rejects `.geojson` as `application/octet-stream`, and uploads above 10K records hit the 1-hour timeout. Without a defensive uploader, partner teams lose hours per case to format errors that surface only after retry.

**Tools composed.**
- `olmoearth.upload_labels` (`PLAN.md` §1).
- Skill-local: `validate_studio_mime`, `shard_at_10k`, `split_multi_metric`.

---

### 2. `olmoearth-rslearn-config`

**Upstream:** unified with skill #1 inside [`olmoearth-data-prep`](https://github.com/2imi9/OlmoEarth-Skills/tree/main/skills/olmoearth-data-prep). The upstream skill emits both AWF-style (1 sentinel2 layer with 3 zoom_offset bandsets) and production-style (12 per-month layers) `dataset.json` layouts.

**In:** labels + AOIs.
**Out:** `rslearn` `dataset.json` + Lightning YAML.

**What.** `oe_labels` schema, single-layer 3-bandset or per-month production layout, `es_label` rename, watershed AOIs from NLDI / HUC-12, 7-criteria audit before training starts.

**Why.** Config friction is a documented barrier to operationalizing geospatial foundation models ([WorldCereal deployment lessons, arXiv:2508.00858](https://arxiv.org/abs/2508.00858), preprint). The `es_label` rename trap and layout confusion cause silent failures that only show up after a multi-hour training run.

**Tools composed.**
- `eo.window_tile`, `olmoearth.resolve_to_aoi` (`PLAN.md` §1).
- Skill-local: `write_rslearn_config`, `audit_7_criteria`, `rename_es_label`.

---

## Configure

### 3. `olmoearth-studio-job-config`

**Upstream:** [`2imi9/OlmoEarth-Skills/skills/olmoearth-studio-job-config`](https://github.com/2imi9/OlmoEarth-Skills/tree/main/skills/olmoearth-studio-job-config). Vendor as-is.

**In:** task description.
**Out:** Studio wizard answers.

**What.** Picks output type (per-pixel / window / detection / embeddings), model size (Nano / Tiny / Base), time-frame mode (period vs single-moment-with-context vs single-moment), imagery sources (S2 alone vs +S1), patch size (160 / 320 / 640 / 1280 m). 14 verified presets (crop / mangrove / land cover / soil moisture / biomass / vessel / solar / oil slick / flood / drought / burn scar / embeddings) plus a cross-field validator.

**Why.** Cross-field traps exist: detection with 320 m patch fails silently, Landsat is not yet available as a source, embeddings mode is incompatible with single-moment time frame. Researchers without OlmoEarth-specific intuition iterate on a misconfigured wizard for days. The validator catches the trap before the job is submitted.

**Tools composed.**
- Skill-local: `studio_job_validate`, `apply_preset`, `cross_field_check`.

---

### 4. `olmoearth-embeddings`

**Upstream:** [`2imi9/OlmoEarth-Skills/skills/olmoearth-embeddings`](https://github.com/2imi9/OlmoEarth-Skills/tree/main/skills/olmoearth-embeddings). Vendor as-is; grounded in the AWF Kenya tutorial's accuracy/time/VRAM table. Handles the small-dataset (<100 samples), limited-compute (T4 / Colab), similarity-search, and "no labels yet" cases.

**In:** task profile (label volume, class balance, target VRAM, target latency).
**Out:** embeddings-vs-fine-tune decision + runnable `.ipynb`.

**What.** Decision grounded in the fine-tuned OlmoEarth benchmark table. Parameterized notebook extracts OlmoEarth Nano / Tiny / Base / Large embeddings and trains kNN + linear-probe heads.

**Why.** Embeddings-vs-fine-tune is the most frequent decision point in EO foundation model use. Practitioners without empirical guidance over-fit Large models on small samples or under-use embeddings when they would have sufficed. The measured accuracy / time / VRAM tradeoffs from fine-tuned OlmoEarth make the choice visible and actionable. This is also a direct counter to AlphaEarth being shipped as annualized embeddings only.

**Tools composed.**
- `olmoearth.fetch_embedding` (`PLAN.md` §1, new in v0.4).
- `system.python` for the notebook execution.
- Skill-local: `decision_matrix`, `knn_head`, `linear_probe_head`.

---

## Run

### 5. `olmoearth-predict`

**In:** Studio project + area + model_id + time range + config.
**Out:** `PredictionRef`, status, tiles, vectors, pixel values, feature matches.

**What.** Submit a prediction (`POST /predictions`), poll progress (`GET /predictions/{id}`), query pixel values at points (`/pixel-value`), search result features by class (`/features/search`), download raw outputs via token (`/files`).

**Why.** The Studio API is async with a 5-state status enum (`pending / running / completed / failed / cancelled`) and a `download_token` flow. Without a wrapper, every case study re-implements the polling loop, retry on result-creation failures, and the tile-vs-raw-output decision. This is the core run primitive every other skill depends on.

**Tools composed.**
- `olmoearth.submit_prediction`, `poll_prediction`, `fetch_results` (`PLAN.md` §1).
- `olmoearth.pixel_value`, `features_search` (`PLAN.md` §1, new in v0.4).

**First skill to ship.** Foundation that #6, #7, #9, #10 all reuse.

---

### 6. `olmoearth-change-detect`

**In:** Studio project + area + ≥3 time points (refuses 2).
**Out:** change layer + trajectory metrics.

**What.** Two POSTs to `/predictions` at t0 and t1, plus at least one intermediate time, produce a change layer. Forces a 3+-date trajectory for conservation and agriculture cases rather than a single before / after.

**Why.** Two-date diffs hide gradual drift. Annualized embedding products structurally cannot capture intra-annual signals such as cover-crop transitions, flood peaks, and harvest timing (consistent with [Ma et al. arXiv:2601.00857](https://arxiv.org/abs/2601.00857), preprint). Deforestation monitoring, crop calendars, and disaster workflows all need a trajectory. The skill enforces a minimum 3-date pattern so the agent does not publish false-change reports based on a single noisy pair.

**Tools composed.**
- Skill #5 (`olmoearth-predict`).
- Skill-local: `diff_layers`, `enforce_min_3_dates`.

---

### 7. `olmoearth-baseline-compare`

**In:** Studio project + area + transfer-region AOI.
**Out:** difference raster + per-metric comparison table.

**What.** Side-by-side OlmoEarth vs AlphaEarth on transfer regions where AlphaEarth is documented to underperform. Returns a difference raster plus a per-metric comparison table.

**Why.** AlphaEarth-based models are documented to underperform under cross-region transfer ([Ma et al. arXiv:2601.00857](https://arxiv.org/abs/2601.00857), preprint; specific magnitudes may revise). Fine-tuned OlmoEarth's "substantially outperformed" claim lands only if cases include side-by-side runs in regions where AlphaEarth actually fails, not on home-turf regions where it is competitive.

**Dependency.** Requires AlphaEarth access via the Earth Engine MCP. AlphaEarth has been Trusted-Tester gated through late 2025; verify access before running. UNVERIFIED if access has opened up; flag at skill load time.

**Tools composed.**
- Skill #5.
- Skill-local: `alphaearth_via_gee_mcp`, `compare_metrics`, `difference_raster`.

---

## Analyze

### 8. `olmoearth-evaluate`

**In:** `ResultBundle` + ground truth.
**Out:** per-class metrics + spatial-block CV + NNDM-LOO results.

**What.** Spatial-block cross-validation, leave-one-region-out, nearest-neighbor-distance-matching LOO-CV for true map accuracy.

**Why.** [Ploton et al. (Nat. Commun. 11:4540, 2020)](https://www.nature.com/articles/s41467-020-18321-y) showed that standard random CV can report strong R² on models with near-zero spatial predictive skill. [Meyer & Pebesma (MEE 12:1620, 2021)](https://besjournals.onlinelibrary.wiley.com/doi/10.1111/2041-210X.13650) formalized this. Without spatial CV, every case narrative the agent produces risks publishing inflated metrics. This is the most-cited methodological pain in EO ML literature.

**Tools composed.**
- `utils.spatial_train_val_split` (`PLAN.md` §1).
- Skill-local: `spatial_block_cv`, `nndm_loo`, `per_class_metrics`.

---

### 9. `olmoearth-similarity`

**In:** query AOI / patch.
**Out:** top-K similar patches with similarity scores + geographic-prior warning.

**What.** FAISS index over fine-tuned OlmoEarth Base embeddings. Returns top-K patches with similarity scores plus a geographic-prior warning when results cluster in the same biome as the query.

**Why.** [NASA Earthdata's Similarity Search tool](https://www.earthdata.nasa.gov/dashboard/services/similarity-search) helps scientists avoid manually inspecting large satellite-imagery regions. Public AlphaEarth demos show similarity search correlates well with independent risk models in match tasks. OlmoEarth has competitive embeddings (Base wins 15 of 24 kNN tasks per [arXiv:2511.13655](https://arxiv.org/abs/2511.13655)) but no skill exposing similarity search. This is the most-requested missing primitive.

**Tools composed.**
- `olmoearth.fetch_embedding` (`PLAN.md` §1, new in v0.4).
- `system.python` for FAISS (FAISS preloaded in sandbox).
- Skill-local: `faiss_index_build`, `geographic_prior_check`.

---

### 10. `olmoearth-uncertainty`

**In:** `PredictionRef` + AOI.
**Out:** confidence map + DI-weighted OOD flag per region.

**What.** Confidence maps plus a Meyer-Pebesma Dissimilarity-Index-weighted out-of-distribution flag. Warns when an AOI lies outside the trained embedding space.

**Why.** Softmax confidence is not OOD detection. Meyer & Pebesma's [Area of Applicability framework](https://besjournals.onlinelibrary.wiley.com/doi/10.1111/2041-210X.13650) (R [`CAST`](https://cran.r-project.org/package=CAST) package, on CRAN since 2018) is implemented across multiple peer-reviewed methods papers but absent from every EO foundation model platform. AlphaEarth's documented transfer failure under domain shift is exactly what AOA would have flagged.

**Tools composed.**
- `olmoearth.pixel_value` (`PLAN.md` §1, new in v0.4).
- Skill-local: `area_of_applicability`, `repeated_sampling`, `ood_flag`.

---

### 11. `olmoearth-cloud-mask-audit`

**In:** STAC items for an AOI + time range.
**Out:** per-tile cloud-mask disagreement raster + bad-mask-vs-bad-model verdict.

**What.** Ensemble of CFMask, s2cloudless, Sen2Cor, MAJA: flags whether a bad prediction is caused by a bad cloud mask or a bad model. Surfaces ensemble disagreement, not a single ground-truth mask.

**Why.** [Skakun et al. CMIX (RSE 274:112990, 2022)](https://doi.org/10.1016/j.rse.2022.112990) compared 10 cloud algorithms across 5 reference datasets and found large disagreement on thin and semi-transparent clouds. HLS v1.3 documentation flags Sentinel-2 cloud-mask quality as an open issue. Cloud-mask accuracy is also known to drop in snow / ice conditions (Yan et al. RSE 2025). Without an audit, practitioners cannot tell whether their model is wrong or just looking at cloud.

**Tools composed.**
- `eo.search_stac`, `eo.sign_assets` (`PLAN.md` §1).
- Skill-local: `fetch_cloud_masks`, `ensemble_disagree`, `verdict_classifier`.

---

## Integrate

### 12. `olmoearth-qgis-bridge`

**In:** `ResultBundle.tile_template` + uncertainty.
**Out:** QGIS WMTS connection file + COG + sidecar uncertainty raster + SLD style.

**What.** Writes a QGIS Processing-toolbox provider and an SLD style derived from the model class metadata. Outputs full-precision GeoTIFF, not just colorized tiles.

**Why.** Existing QGIS deep-learning plugins are documented as inference-only and compute-heavy ([IAMAP, arXiv:2508.00627](https://arxiv.org/abs/2508.00627), preprint). Google shipped a major Earth Engine QGIS plugin upgrade in late 2025 because GIS users will not leave their desktop. Most case providers (NGOs, government analysts, local agencies) live in QGIS or ArcGIS. This skill closes the hot-cold loop.

**Tools composed.**
- `olmoearth.fetch_results` (`PLAN.md` §1).
- Skill-local: `write_sld_style`, `cog_export`, `qgis_provider_xml`.

---

### 13. `olmoearth-data-export`

**REFRAMED 2026-05-28.** The original idea (wire third-party GEE / OSM /
USGS / NOAA MCPs *into* an AOI) needs external MCPs the user must connect
and isn't core. It was reframed to the more useful, self-contained
**export our own Studio data, grouped**, implemented as
`olmoearth_export_data` (projects + predictions → JSON grouped by project
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

### 14. `olmoearth-provenance`

**In:** every API call (transparent wrap).
**Out:** manifest + single-command replay script.

**What.** Manifest of `prediction_id`, `model_id`, dataset hashes, time window, parameters, tile dates. Emits a single-command replay script so the case can be re-run by a third party.

**Why.** ML pipelines have a documented reproducibility gap ([Samuel et al. arXiv:2006.12117](https://arxiv.org/abs/2006.12117)). [Kedron & Holler (Remote Sensing 14:5471, 2022)](https://doi.org/10.3390/rs14215471) call for benchmark datasets to test replicability in remote sensing specifically. EUDR and REDD+ MRV impose audit requirements that need traceable provenance, not just outputs. Without provenance, NGO and regulator-facing cases cannot be defended.

**Tools composed.**
- Hooks into every `PLAN.md` §1 `olmoearth.*` call via `provenance_middleware` (added to `PLAN.md` §3 as Rule 13).
- Skill-local: `manifest_write`, `replay_script_emit`, `dataset_hash`.

**Operational rule.** Adds rule 13 to `PLAN.md` §3: *every Studio API call emits a manifest entry; sessions that disable provenance refuse to call `save_view`.*

---

### 15. `olmoearth-case-narrative`

**In:** `tile_urls` + provenance manifest.
**Out:** stakeholder writeup with live tiles + freshness gate.

**What.** Stakeholder writeup with embedded live map tiles, citations to dataset hashes, and a freshness gate that refuses to render stale tiles past a configurable window.

**Why.** Practitioners working under operational timelines often need to ship a working solution rather than an optimal one ([WorldCereal lessons, arXiv:2508.00858](https://arxiv.org/abs/2508.00858), preprint). NGO leadership, journalists, and policymakers need outputs tied to map tiles and policy briefs, not Jupyter notebooks. Freshness gating prevents stale-tile reports during disaster response when conditions change within hours.

**Tools composed.**
- Skill #14 (`olmoearth-provenance`) for manifest read.
- `olmoearth.fetch_results` for tile URLs.
- Skill-local: `freshness_gate`, `narrative_template`, `tile_embed`.

---

## Roadmap reference

Implementation order tracks `PLAN.md` §6. First skill to ship is **#5 `olmoearth-predict`** (the foundation that #6, #7, #9, #10 reuse). After that, prioritization is driven by the case-study queue, not this catalog order.

## Adding a skill

A new skill is one PR with:
1. A folder under `skills/<skill-name>/` containing `SKILL.md` + `skill-card.md` + `scripts/` + optional `references/`, `assets/`. For skills already in [`2imi9/OlmoEarth-Skills`](https://github.com/2imi9/OlmoEarth-Skills), vendor the folder verbatim (git submodule or copy + provenance note in `skill-card.md`) rather than re-implementing.
2. Any new global tools added to `PLAN.md` §1.
3. A `tests/skills/test_<skill_name>.py` exercising the skill end-to-end against the live Studio API (mark with `@pytest.mark.integration`).
4. An entry in this `SKILLS.md` matching the per-skill template above.
5. A `CHANGELOG.md` line under `### Added`.

See [`CONTRIBUTING.md` §3](CONTRIBUTING.md#3-branch-and-pr-workflow) for branch naming.

## Vendoring policy

[`2imi9/OlmoEarth-Skills`](https://github.com/2imi9/OlmoEarth-Skills) is canonical for skills #1-#4; agent vendors at a pinned commit. Decide in the first skill PR:
- **A. Git submodule**: track an upstream SHA; cleanest provenance, harder for casual contributors.
- **B. Copy + provenance in `skill-card.md`**: simpler, drift risk if the bump is forgotten.
