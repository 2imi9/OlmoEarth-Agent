# OlmoEarth Agent Skills Catalog

Detailed spec for the 18 skills the agent ships with. Each skill is an [agentskills.io](https://agentskills.io)-spec package (`SKILL.md` + frontmatter + optional `scripts/`, `references/`, `assets/`, `skill-card.md`, `skill.oms.sig`).

`PLAN.md` is the runtime contract (tools, dataclasses, operational rules). This file is the *skill-layer* contract: what each skill does, why, the tools it composes (from `PLAN.md` §1 or skill-local), and the academic / engineering references that justify it.

**Status:** 18 skills implemented (v1.0 shipped #1–#15 on 2026-05-31; #16 `olmoearth-litsearch` + #17 `olmoearth-automate` added post-1.0; #18 `olmoearth-negative-sampler` added post-1.1). See `CHANGELOG.md`.

## Existing implementations (upstream)

Three skills already exist in [`2imi9/OlmoEarth-Skills`](https://github.com/2imi9/OlmoEarth-Skills) (updated 2026-05-17). The agent vendors them rather than re-implementing.

| Upstream | Catalog mapping |
|---|---|
| [`olmoearth-data-prep`](https://github.com/2imi9/OlmoEarth-Skills/tree/main/skills/olmoearth-data-prep) | Skills **#1 + #2 unified**: 8 prep pitfalls + 7-criteria audit; recognizes all three schemas (`sample_category` / `es_label` / `oe_labels.{key}`). |
| [`olmoearth-studio-job-config`](https://github.com/2imi9/OlmoEarth-Skills/tree/main/skills/olmoearth-studio-job-config) | Skill **#3**: 14 verified presets + cross-field validator. |
| [`olmoearth-embeddings`](https://github.com/2imi9/OlmoEarth-Skills/tree/main/skills/olmoearth-embeddings) | Skill **#4**: embeddings-vs-fine-tune **guidance** + a Nano/Tiny/Base/Large notebook generator. (Skill #17 `olmoearth-automate` is the one-call automated version that reuses this decision table.) |

Skills #5-#15 are implemented in this repo (see `CHANGELOG.md`). Catalog #1/#2 are unified upstream as `olmoearth-data-prep` (matching the working implementation) but kept as two numbered entries here.

**Description convention.** Upstream uses trigger-heavy multi-sentence frontmatter ("Use whenever…", "Trigger even when…") because the description is the LLM's routing surface. Match it.

---

## Catalog

| # | Category | Name | What |
|---|---|---|---|
| 1 | Prep | [`olmoearth-studio-upload`](#1-olmoearth-studio-upload) | Labels (GeoJSON / CSV / Shapefile) → Studio-importable file with MIME / 10K / multi-metric guards. |
| 2 | Prep | [`olmoearth-rslearn-config`](#2-olmoearth-rslearn-config) | Labels → `rslearn` `dataset.json` + Lightning YAML with 7-criteria audit. |
| 3 | Configure | [`olmoearth-studio-job-config`](#3-olmoearth-studio-job-config) | Task description → Studio wizard answers with 14 presets + cross-field validator. |
| 4 | Configure | [`olmoearth-embeddings`](#4-olmoearth-embeddings) | Embeddings-vs-fine-tune **guidance** + a generated runnable notebook (you run it). |
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
| 16 | Report | [`olmoearth-litsearch`](#16-olmoearth-litsearch) | arXiv + OpenAlex literature search + DOI/arXiv-id resolution to ground citations. |
| 17 | Configure | [`olmoearth-automate`](#17-olmoearth-automate) | **One call**: auto-decides embeddings vs fine-tune + proposes a config (reuses #4's logic); optional HF-dataset introspection. |
| 18 | Prep | [`olmoearth-negative-sampler`](#18-olmoearth-negative-sampler) | Presence-only labels → trainable set: buffered, spatially-thinned (optionally embedding-dissimilar) negative class so the data-prep audit passes. |

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
| 16 | `olmoearth-litsearch` | "Find and cite the paper behind the Area-of-Applicability method I used." |
| 17 | `olmoearth-automate` | "I have 200 labels and a T4 — should I fine-tune or use embeddings? Set it up." |
| 18 | `olmoearth-negative-sampler` | "My karst-site labels are presence-only and the audit fails for a missing negative class — generate background samples." |

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

**Versus #17.** This skill is the human-in-the-loop *guidance + notebook generator*: it explains the decision and emits a notebook the **user** runs. [`olmoearth-automate`](#17-olmoearth-automate) (#17) is the *one-call* automation that decides programmatically and proposes a config, reusing this skill's decision table.

**What.** Decision grounded in the fine-tuned OlmoEarth benchmark table. Parameterized notebook extracts OlmoEarth Nano / Tiny / Base / Large embeddings and trains kNN + linear-probe heads.

**Why.** Embeddings-vs-fine-tune is the most frequent decision point in EO foundation model use. Practitioners without empirical guidance over-fit Large models on small samples or under-use embeddings when they would have sufficed. The measured accuracy / time / VRAM tradeoffs from fine-tuned OlmoEarth make the choice visible and actionable. This is also a direct counter to AlphaEarth being shipped as annualized embeddings only.

**Tools composed.**
- `olmoearth.fetch_embedding` (`PLAN.md` §1, new in v0.4).
- `system.python` (opt-in subprocess) for light checks only — the generated notebook is meant for the **user** to run (it needs the geospatial/GPU stack the sandbox does not guarantee).
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
- FAISS runs inside the `olmoearth_similarity_search` tool handler — not the `system:python` sandbox (FAISS is not preloaded there).
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

### 16. `olmoearth-litsearch`

**In:** a free-text query, or a single DOI / arXiv id.
**Out:** curated paper records (id, title, authors, year, venue, doi, arxiv_id, url, cited_by_count; abstract optional), deduped across sources.

**What.** Searches arXiv (Atom API) and OpenAlex (`/works`), and resolves a DOI or arXiv id to one record. Key-free — OpenAlex is queried via the documented polite-pool `mailto` (set `OLMOEARTH_OPENALEX_MAILTO` to opt in). Round-robin blends the two sources, then dedups on DOI → arXiv id → normalized title. Returns bibliographic metadata only — never full text / PDF bytes and never geometry — so it is provenance-safe.

**Why.** The catalog already *cites* a body of EO literature (spatial CV, cloud masking, OlmoEarth / AlphaEarth embeddings, WorldCereal), but before this skill the agent could only lean on world-knowledge or hallucinate links — the exact failure mode Google DeepMind's Science Skills report documents and that its own arXiv/OpenAlex skills fix. This grounds the case-narrative / research workflow in real, citable sources. **No fabrication:** never invents DOIs / ids / titles, reports empty results as empty, and returns a real `url` to cite for every record.

**Tools composed.**
- `olmoearth_litsearch` (unified arXiv + OpenAlex search) + `olmoearth_litsearch_resolve` (DOI / arXiv-id → one record).
- Logic in `analysis/litsearch.py` (query-build / parse / cross-source dedup); shared `httpx` retry on transient {429, 5xx}.

---

### 17. `olmoearth-automate`

**In:** a free-form `task` (e.g. "land cover, 9 classes, 200 samples, T4 GPU"), explicit `num_samples` / `num_classes` / `compute` / `goal`, and/or a Hugging Face `hf_dataset` id.
**Out:** a decision (`embeddings` / `embeddings_then_fine_tune` / `fine_tune`), rationale, and a proposed config (model size, classifier head, an embeddings-notebook command, a fine-tune schedule, and a hand-off to `olmoearth-studio-job-config`).

**Versus #4.** [`olmoearth-embeddings`](#4-olmoearth-embeddings) (#4) is the *guidance + notebook generator* (the user runs the notebook). This skill is the *one-call* version: it decides programmatically and emits a ready config, can fill its inputs from a Hugging Face dataset, and reuses #4's decision table rather than duplicating it.

**What.** Applies the embeddings-vs-fine-tune precedence rules — a faithful port of the vendored `olmoearth-embeddings` `recommend.decide` (kept in sync) — then proposes an actionable config. Given a Hugging Face dataset id, it reads the row count + ClassLabel classes from the public datasets-server to fill the inputs. Inputs are task metadata only (no geometry), so results are provenance-safe.

**Why.** Picking embeddings vs fine-tune, the model size, and the classifier head is the most common configuration decision in EO foundation-model use, and the precedence (sample-size / compute / goal) is easy to get wrong by hand. This automates the call and proposes a runnable plan, routing Studio-side specifics to `olmoearth-studio-job-config`. **No fabrication:** reports `ask_for` when key inputs are missing rather than guessing, and never invents dataset stats.

**Tools composed.**
- `olmoearth_automate` (decision + config; optional HF-dataset introspection).
- Logic in `analysis/automate.py` (decision port + `propose_config` + `fetch_hf_dataset_profile`); reuses the vendored `olmoearth-embeddings` decision table.

---

### 18. `olmoearth-negative-sampler`

**In:** a presence-only labels GeoJSON path (`positives_path`); optional `candidates_path`, `negative_label`, `n_negatives`, `exclusion_km`, `min_separation_km`.
**Out:** a combined GeoJSON (positives + generated negatives) written to `out_path`, plus a counts/ranking summary (no raw coordinates in chat).

**What.** Generates the missing negative/background class for a presence-only label set as **buffered, spatially-thinned pseudo-absences** (Barbet-Massin et al. 2012): candidate background points within `exclusion_km` of any positive are dropped, accepted negatives are kept `min_separation_km` apart, and — when the inputs carry per-feature `properties.embedding` vectors — candidates are ranked by environmental *dissimilarity* to the positive centroid (the inverse of skill #9's similarity search). Defaults to a balanced 1:1 set and writes the negatives under the same schema field the positives use (`sample_category` / `es_label` / `oe_labels.category`). Deterministic; no GDAL.

**Why.** A presence-only set is not trainable — a classifier with no counter-examples predicts the positive class everywhere (the "false positives everywhere" failure, pitfall #8 in `olmoearth-data-prep`). That skill's `audit.py` *detects* the gap (it hard-FAILs `check_negative_class`) but its `--negative-class auto` was deferred, so the agent could previously only report the dataset as unusable and stop. This skill converts that dead-end into a finished artifact: the combined file round-trips straight back through the data-prep audit and clears the negative-class check. **Honest by construction:** the negative label must be one the audit recognizes, and a placement shortfall (buffer / thinning / extent too tight) is surfaced as a warning rather than silently under-filled.

**Tools composed.**
- `olmoearth_negative_sampler` (file → file; counts + path returned).
- Logic in `analysis/negative_sampler.py` (buffer + farthest-point/embedding-dissimilarity selection, reusing `evaluation.spatial_cv.haversine_km`); composes the data-prep audit's negative-class contract.

---

## Roadmap reference

Implementation order tracks `PLAN.md` §6. First skill to ship is **#5 `olmoearth-predict`** (the foundation that #6, #7, #9, #10 reuse). After that, prioritization is driven by the case-study queue, not this catalog order.

Candidate skills beyond the current 18 (prioritized) are researched in [`docs/eo-skills-shortlist.md`](docs/eo-skills-shortlist.md); its build-first pick, `olmoearth-negative-sampler`, shipped as #18.

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
