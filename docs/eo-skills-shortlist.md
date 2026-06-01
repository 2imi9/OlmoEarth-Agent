# Additional in-domain EO skills -- research + shortlist

Candidate skills to extend the OlmoEarth Agent catalog beyond the current 17,
all **in-domain** (Earth observation / geospatial / OlmoEarth Studio) and
**in-pattern** (pure-Python async `build_*_tools` -> `RegisteredTool`, logic in
`analysis/` or `reporting/`, reusing `httpx`, no heavy geospatial deps in the
sandbox -- pixel-heavy work is surfaced to Studio or the user). The GDM
science-skills bundle was assessed off-domain
([`docs/science-skills-assessment.md`](science-skills-assessment.md)); nothing here
imports it.

> **Method.** Read `SKILLS.md` (17-skill catalog) and `PLAN.md` (tool catalog,
> operational rules), verified coverage against the actual code
> (`src/olmoearth_agent/{analysis,reporting,evaluation,tools}/`,
> `skills/registry.py`), and grepped `src/` to confirm the gaps are genuinely
> absent (not just undocumented). The strongest Prep/Configure candidates were
> re-checked against the vendored `olmoearth-data-prep` (`SKILL.md`,
> `scripts/audit.py`, `references/pitfalls.md`).

## TL;DR

- **The 17 cover the six workflow stages well** -- Prep, Configure, Run, Analyze,
  Integrate, Report. Analyze is the best-covered (spatial-CV inflation, AOA/OOD,
  cloud-mask audit, similarity); the real gaps cluster at the **Prep edges**
  (turning a label set into a *trainable* dataset) and at **operationalization**
  (one-shot -> recurring; pixel map -> policy number).
- **Build first: `olmoearth-negative-sampler`.** It is the only candidate that
  resolves a *literally documented hard FAIL*: `data-prep`'s `audit.py`
  hard-fails a presence-only dataset for having no negative class, and its
  `SKILL.md` defers `--negative-class auto` to a planned version -- so today the
  agent can only tell the user their dataset is unusable and stop. This converts a
  dead-end into a finished dataset.
- **Then, by value:** area-adjusted accuracy with CIs (the MRV/regulatory
  standard), a recurring monitor/alert loop (the demo-to-operational divide),
  label<->imagery temporal alignment, a fine-tune recipe generator, and zonal
  stats to admin/watershed units.
- **Two items are not new skills:** `split-materializer` is best folded into
  `olmoearth-data-prep`'s split step, and downstream non-GIS export
  (STAC/GeoParquet/CSV) is already tracked as issue #59.
- **Honest sizing:** there are roughly eight genuinely distinct, well-justified
  candidates -- not a padded list. A smaller honest shortlist beats invented
  skills (cf. the project's "generalize skills, not examples" rule).

## 1. Coverage of the current 17 (verified against code)

| Stage | Covered by | Verdict |
|---|---|---|
| **Prep** | #1 studio-upload, #2 rslearn-config | Label ingest + dataset config solid; the *edges* (negatives, label/imagery alignment, materialized leakage-free split) are thin. |
| **Configure** | #3 job-config, #4 embeddings, #17 automate | Output-type / model-size / time-frame / source / patch all handled; the *how* of fine-tuning (loss, class weights, LR schedule) is not. |
| **Run** | #5 predict, #6 change-detect, #7 baseline-compare | Core async run loop is solid; everything is **one-shot** (no cadence/baseline-state). |
| **Analyze** | #8 evaluate, #9 similarity, #10 uncertainty, #11 cloud-mask-audit | Best-covered stage; stops short of an error matrix + **area-adjusted** accuracy, and of spatial rollups to reporting units. |
| **Integrate** | #12 qgis-bridge, #13 data-export | QGIS + raw JSON covered; non-QGIS export formats (STAC/GeoParquet/CSV) are the gap (issue #59). |
| **Report** | #14 provenance, #15 case-narrative, #16 litsearch | Provenance + stakeholder writeup + citation grounding all present. |

## 2. Prioritized shortlist

Ranked by (gap-fill value x real EO demand x in-pattern feasibility). "Source"
notes whether the candidate was verified against the vendored data-prep code
(*code*) or surfaced by the workflow-stage gap analysis (*coverage*).

| # | Skill | Stage | Fills | Effort | Priority | Source |
|---|---|---|---|---|---|---|
| 1 | **`olmoearth-negative-sampler`** | Prep | Generates the missing negative/background class (buffered, spatially-thinned, embedding-dissimilar pseudo-absence) so a presence-only label set becomes trainable. Resolves `audit.py`'s hard FAIL. | M | **High (build first)** | code |
| 2 | **`olmoearth-area-accuracy`** | Analyze | Confusion/error matrix + stratified reference design + **bias-corrected area estimates with 95% CIs** (Olofsson et al. RSE 2014) -- the MRV/REDD+ standard. #8 stops at per-pixel P/R/F1/IoU; map pixel-counts systematically mis-estimate area. | S-M | High | coverage |
| 3 | **`olmoearth-monitor`** | Run/Report | Register a standing AOI, re-run a prediction on a cadence, diff vs the prior baseline, emit an **alert only on a thresholded, spatially-coherent change**. Studio has no webhooks, so this is client-side scheduled polling + baseline state. The demo-to-operational divide. | M | High | coverage |
| 4 | **`olmoearth-label-imagery-align`** | Prep | Per-label gate: verify cloud-free Sentinel imagery actually exists in each label's window (STAC metadata only) before the dataset is built; tighten/flag the time offset. Catches a 2019-label/2023-pixel mismatch *before* a multi-hour ingest. | M | Med-High | code |
| 5 | **`olmoearth-finetune-tuner`** | Configure | Turns a `fine_tune` decision into a concrete Lightning recipe: loss + class weights for the actual imbalance, layer-wise LR decay / warmup / cosine, batch + epoch budget. #3 explicitly punts these; #4/#17 decide *whether*, not *how*. | M | Med-High | code |
| 6 | **`olmoearth-zonal-stats`** | Analyze/Report | Roll a prediction up to admin/watershed units (% class area, count, mean score, change per HUC-12 / county FIPS / protected area). `resolve_to_aoi` already yields the polygons. The missing pixel-map -> policy-number bridge. | M | Med | coverage |
| 7 | **`olmoearth-composite`** | Prep/Run | Rank + select best scenes per period from STAC metadata (cloud fraction, QA) and emit a cloud-free **composite recipe** + a gap/coverage report. #11 only *audits* mask disagreement; nothing produces the clean input. Pixel compositing stays in Studio/user (no GDAL). | M | Med | coverage |
| 8 | **`olmoearth-sample-design`** | Run/Configure | Stratified / systematic sample allocation by class proportion (Cochran/Olofsson sample-size for a target SE) and uncertainty-guided active-learning next-batch points to feed Studio annotation. Upstream of #2's split. | M | Med | coverage |

**Fold-in / already-tracked (not new skills):**
- `olmoearth-split-materializer` (a buffered, distance-guaranteed train/val/test
  split file) is a genuine gap but its cleanest home is *inside*
  `olmoearth-data-prep`'s split step (upgrading the current 1-D longitude sort),
  not a standalone catalog entry.
- Downstream non-GIS export (STAC Item/Collection, GeoParquet, CSV) is issue #59;
  prefer GeoParquet/GeoJSON/CSV over GeoPackage to stay dep-light.

## 3. Build-first: `olmoearth-negative-sampler`

The pick because it is the only candidate that turns a hard stop into a finished
artifact, verified in the vendored code:

- `references/pitfalls.md`: presence-only -> "false positives everywhere" is
  pitfall #8, hit across the Karst / Chesapeake cases.
- `scripts/audit.py` `check_negative_class` hard-**fails** when no
  `other`/`background`/`stable`/`none` class exists -- but never produces one.
- `SKILL.md`: "Auto-generating negative class samples -- v0 detects the absence;
  the user must add them," with `--negative-class auto` explicitly deferred.

**In-pattern fit.** `build_negative_sampler_tools -> RegisteredTool`, logic in
`analysis/`, composing `utils.add_buffer` + `to_geodataframe`/`from_geodataframe`
(sjoin + spatial thinning) + `olmoearth.fetch_embedding` for an OlmoEarth-native
"environmentally dissimilar" ranking (it *inverts* #9 similarity -- no FAISS, no
duplication). Emits labeled GeoJSON to a `FilePath` that round-trips straight back
into `data-prep`'s audit (operational rule 1: coords to file, not chat). No GDAL.
EO basis: buffered pseudo-absence is a textbook SDM/EO method (Barbet-Massin et
al. 2012, [doi:10.1111/j.2041-210X.2011.00172.x](https://doi.org/10.1111/j.2041-210X.2011.00172.x)).

## 4. Notes and caveats

- Every candidate keeps **pixel-level raster math out of the sandbox** (no
  guaranteed GDAL/rslearn) -- they operate on STAC metadata, label geometries,
  confusion matrices, or existing tool outputs, and emit recipes / selections /
  tables for Studio or the user to execute.
- AlphaEarth is now public via the GEE "Satellite Embedding" dataset; skill #7 is
  already reframed to bring-your-own exported data, so no new skill is needed
  there.
- The shortlist is intentionally Prep/Configure-heavy at the top because that is
  where the verified, documented holes are; the operationalization gaps (#3, #6)
  are the highest *strategic* value (demo -> deployed) but a step bigger in scope.

## 5. References

- Olofsson et al., "Good practices for estimating area and assessing accuracy of land change" (RSE 2014): <https://doi.org/10.1016/j.rse.2014.02.015>
- Barbet-Massin et al., pseudo-absence selection (MEE 2012): <https://doi.org/10.1111/j.2041-210X.2011.00172.x>
- LandCoverNet (label/imagery temporal correspondence): <https://arxiv.org/abs/2012.03111>
- Reduced Focal Loss (class imbalance): <https://arxiv.org/abs/1903.01347> · "Fine-tune Smarter, Not Harder" (layer-wise LR for EO foundation models): <https://arxiv.org/abs/2504.17397>
- Karasiak et al., spatial leakage in CV (ML 2021): <https://doi.org/10.1007/s10994-021-05972-1>
- In-repo: [`SKILLS.md`](../SKILLS.md), [`PLAN.md`](../PLAN.md), vendored `olmoearth-data-prep` (`vendor/olmoearth-skills/`), issue #59 (downloadable export artifacts).
