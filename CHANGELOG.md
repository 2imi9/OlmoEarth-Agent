# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog v1.1.0](https://keepachangelog.com/en/1.1.0/); this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

PR titles for each merged change appear under the appropriate section below.
See [`CONTRIBUTING.md`](CONTRIBUTING.md#7-documentation) for the convention.

## [Unreleased]

### Added
- **AOI draw-in-chat**: select an area of interest by **drawing** it on a map
  in the chat instead of typing a bbox. A composer "Draw AOI" button opens a
  Leaflet map (OSM basemap, rectangle/polygon tools; loaded from CDN, no build
  step); the drawn polygon is stored as an **OlmoEarth Studio area**
  (`POST /api/areas` -> new `StudioClient.create_area`, verified against the
  live `POST /api/v1/areas`) and attached to the next brief carrying its
  `area_id` (for `olmoearth_submit_prediction`) and bounding box (for
  bbox-based skills such as #19 `olmoearth-latent-change`). The agent surfaces
  the widget itself: a new foundational tool `olmoearth_request_aoi` lets it
  ask the user to draw an area when a task needs one and none was given,
  rendering an inline "Draw the area" button that seeds a follow-up turn.
  The modal can also **reuse a saved area**: pick one of the project's
  existing areas (`GET /api/areas/{id}`) to render it on the map and attach
  its `area_id` without creating a duplicate. Saved areas also appear as an
  **"Areas" branch under each project in the sidebar tree** and can be
  **dragged into the chat** to attach (like prediction results). The dev
  bridge now serves the static web UI with `Cache-Control: no-cache` so
  edited (no-build) ES modules are never served stale.
  Pure-Python geometry helpers in `analysis/aoi.py` (GeoJSON Polygon/
  MultiPolygon validation + bbox; no GDAL/numpy, agent stays torch-free).
  Resolves issue #59 (map/draw AOI) and the concrete sub-piece of #90.
- **Skill #19 `olmoearth-latent-change`** (catalog 18 -> 19): a JEPA latent-prediction
  change detector on **frozen** OlmoEarth embeddings, shipped as a thin
  **out-of-process** link to the new standalone heavy-ML repo
  [`2imi9/olmoearth-jepa-change`](https://github.com/2imi9/olmoearth-jepa-change)
  (PyTorch + CUDA, kept *out* of this torch-free agent). A lightweight head predicts the
  time-2 patch embedding from time-1; the prediction residual is the change score
  (I-JEPA, Assran et al. CVPR 2023). On the OSCD test split (frozen OlmoEarth-v1-Base)
  it beats the cosine baseline by **+0.22 F1** (0.25 -> 0.47) and ~3x AP, reaching
  unsupervised-SOTA-level **F1 0.54 label-free** (robust threshold), integrity-verified
  (9x chance, permutation control). A Phase-2 gate study found current general VLMs
  cannot deliver calibrated, localized raster change comparison, so the agent needs this
  calibrated pixel-level tool. **Catalog / skill-contract only here** (`SKILLS.md` #19);
  no heavy dependencies added. Complementary to #6 `olmoearth-change-detect` (Studio-API
  trajectory diff). Out = georeferenced heatmap GeoTIFF + %-area-changed + top-k GeoJSON.
- **Skill #18 `olmoearth-negative-sampler`** (catalog 17 -> 18): generates the
  missing negative / background class for a *presence-only* label set as
  buffered, spatially-thinned (optionally embedding-dissimilar) pseudo-absences,
  and writes a combined GeoJSON that round-trips back through the vendored
  `olmoearth-data-prep` audit -- clearing its negative-class check, which
  hard-FAILs a presence-only set. Resolves the documented dead-end where the
  audit detected the missing class but `--negative-class auto` was deferred
  upstream. In-process tool `olmoearth_negative_sampler`; pure-Python logic in
  `analysis/negative_sampler.py` (no GDAL), inverting skill #9's similarity
  ranking for the environmentally-dissimilar pseudo-absence selection (#75).
- **NNDM-LOO cross-validation for skill #8 `olmoearth-evaluate`** (tool
  `olmoearth_nndm_cv`): Nearest Neighbour Distance Matching Leave-One-Out CV
  (Mila et al. 2022) for an unbiased map-accuracy estimate over the actual
  prediction area, where the random-vs-spatial inflation check only flags the
  risk. Pure-Python `evaluation/nndm.py` (no GDAL; reuses `spatial_cv`'s
  haversine), a faithful port of R CAST's `nndm` verified against the reference
  algorithm (an exact hand-traced execution + the `predpoints==trainpoints`
  invariant from CAST's own tests). Reports the optimistic-bias before (LOO) vs
  after (NNDM) and writes the per-fold train/test/exclude indices inline or to a
  file. Closes the catalog's previously-aspirational NNDM-LOO claim (#92).

### Changed
- **Skill #18 `olmoearth-negative-sampler` -- accuracy refinement**: addresses
  the pseudo-absence accuracy concern (#92) with (1) an optional embedding-based
  **contamination guard** (`contamination_threshold`) that drops candidates
  resembling any positive too closely -- likely *unmapped positives* -- before
  ranking (guarding environmental contamination on top of the spatial buffer),
  and (2) a **quality report** on every result (nearest-positive-distance +
  similarity-to-positive stats + an honest caveat) so the negatives' credibility
  is inspectable and the thresholds tunable, rather than trusted blindly.

### Fixed
- **Skill #9 `olmoearth-similarity` catalog wording** corrected to match the
  implementation: it described "FAISS" but the tool runs an **exact brute-force
  top-K kNN** in-process (FAISS is the scale-up follow-up, per the analysis
  docstring). Aligned `README.md`, `SKILLS.md` (catalog row + #9 section, incl.
  removing a fictional `faiss_index_build` tool), `skills/registry.py`, and the
  `system:python` sandbox wording. Honest-results doc fix from the #92 audit.

### Security
- **In-process egress guard** (`security/egress.py`), informed by NVIDIA
  NemoClaw's network-policy model and SSRF guard
  ([`docs/nemoclaw-assessment.md`](docs/nemoclaw-assessment.md)). A
  per-capability host allowlist (`studio`, `llm-cloud`, `llm-local`, `litsearch`,
  `hf`) plus an SSRF block of private / loopback / link-local / cloud-metadata
  address ranges, validating every outbound endpoint before a request -- and any
  credential -- leaves the process. Wired into `StudioClient` (guards the
  `OLMOEARTH_BASE_URL`-overridable base URL before the Bearer key is bound), the
  hosted-LLM path in `serve._llm_for_request` (before the BYO key is handed over),
  and the litsearch / HF fetchers. Modes: `OLMOEARTH_EGRESS=audit` (default; log
  only, never breaks a deployment) / `enforce` (block, HTTP 403 on the LLM path) /
  `off`; `OLMOEARTH_EGRESS_ALLOW` allowlists a self-hosted Studio or LLM host.
  Decisions are recorded host-only in the provenance manifest
  (`ProvenanceLog.record_egress`).
- **Credential-scrubbed subprocess** for the opt-in `olmoearth_run_python` tool:
  the agent's Studio / LLM / cloud keys are removed from the child environment so
  executed snippets cannot read and exfiltrate them. Defence-in-depth, not a
  network sandbox; OS-essential env is preserved so it still launches. Advances
  the sandbox spec (#54).

## [1.1.0] - 2026-05-31

Post-1.0 checkpoint: two new skills (catalog 15 -> 17), the `make serve` cache
fix, and the sandbox / GDM / quick-start documentation pass.

### Added
- **Skill #16 `olmoearth-litsearch`**: arXiv + OpenAlex literature search and
  DOI / arXiv-id resolution, so the agent can ground EO/geospatial citations in
  real papers instead of world-knowledge or hallucinated links. In-process tool
  bundle (`olmoearth_litsearch`, `olmoearth_litsearch_resolve`), key-free
  (OpenAlex polite-pool `mailto` via `OLMOEARTH_OPENALEX_MAILTO`), deduped across
  sources; informed by a read of Google DeepMind's Science Skills (#62).
- **Skill #17 `olmoearth-automate`**: auto-decides embeddings vs fine-tuning for
  an EO task and proposes a config (model size, classifier head, embeddings
  notebook command, fine-tune schedule, Studio job-config hand-off). In-process
  tool `olmoearth_automate`; reuses the vendored `olmoearth-embeddings` decision
  table and can introspect a Hugging Face dataset (rows + classes) via the public
  datasets-server (#58).
- **`docs/science-skills-assessment.md`**: a multi-agent assessment of Google
  DeepMind's `science-skills` against this repo's skill/harness architecture. It
  re-confirms the bundle is off-domain (genomics / proteomics / chemistry) and
  that arXiv / OpenAlex literature search is the one transferable capability (now
  skill #16), and captures the report's 3-tier-test + LLM-autorater methodology
  as a target for the SkillOpt harness (#69).

### Changed
- **Quick start reworked around the live web UI** (`README.md`): the walkthrough
  leads with `make bridge` and the browser flow, and the LLM is positioned as a
  "local or cloud API" choice ("provider" wording became "cloud API") (#66).
- **README skill count corrected to 17** as `olmoearth-litsearch` (#16) and
  `olmoearth-automate` (#17) landed (badge, table, and prose) (#71).
- **`olmoearth-embeddings` (#4) and `olmoearth-automate` (#17) disambiguated**
  in `README.md` and `SKILLS.md` so the two Configure skills no longer read
  identically: #4 is the embeddings-vs-fine-tune *guidance* plus a runnable-
  notebook generator (you run the notebook); #17 is the *one-call* auto-decide
  + proposed-config tool (with optional Hugging Face dataset introspection) that
  reuses #4's decision logic (#73).
- **Corrected the `system:python` sandbox spec across docs** (`PLAN.md`,
  `AGENTS.md`, `SKILLS.md`): it is the opt-in subprocess (`OLMOEARTH_RUN_PYTHON=1`,
  `python -I`, no persisted state, geospatial stack not guaranteed), not a
  preloaded persistent interpreter (which is now a labelled design target) (#68).

### Fixed
- **`make serve` reuses the cached model** (`docker/llama.compose.yml`,
  `scripts/serve-llm.sh`): the compose now bind-mounts the host Hugging Face
  cache (`cygpath -m`-normalized on Windows) and offers opt-in `HF_PROXY` via
  `host.docker.internal`, so a second `make serve` loads the already-downloaded
  GGUF instead of re-pulling ~17.7 GB (#67).

## [1.0.0] - 2026-05-31

First tagged release: the agent runs all 15 skills live against OlmoEarth
Studio and a local or hosted LLM.

### Added
- **Multi-provider LLM backends**, selectable from the web UI: local Qwen3.6
  (default) plus bring-your-own-key **Claude** (native Anthropic SDK),
  **ChatGPT**, and **Gemini** (OpenAI-compatible). `GET /api/llm/models`
  autodetects each provider's current models; keys are forwarded per request
  and never stored server-side (#52).
- **Model-vs-embeddings labelling in the project tree**: each synthetic model
  node shows the model's real name and a type badge (Embeddings / Fine-tuned),
  resolved via the live (openapi-undocumented) `/models` endpoint (#51).
- **Per-skill example briefs** in `SKILLS.md` (a realistic prompt per skill).

### Fixed
- V1.0 readiness review: release the per-request hosted LLM client after each
  run (connection-pool leak); `/api/run` + `/api/llm/models` return clean
  4xx/5xx for bad input; web-UI subtab stale-closure, silent no-key->local,
  concurrent-send, and tree-retry edges; Studio client surfaces `errors`
  envelopes and paginates client-side filters.

### Changed
- **Relicensed from Apache-2.0 to the OlmoEarth Artifact License** (Ai2), matching
  `allenai/olmoearth_pretrain`: free use with restrictions (no military/defense/
  surveillance or extractive uses; cite Ai2 and propagate the terms downstream).
  Updated `LICENSE`, every source SPDX header (`LicenseRef-OlmoEarth-Artifact-License`),
  `pyproject.toml`, the README badge, `AGENTS.md`, and `CONTRIBUTING.md` (#64).
- **README: dropped the Google "Earth Agent" / AlphaEarth comparisons** and
  tightened wording (`README.md`): the tagline and intro no longer position the
  tool against another product, and skill #7 reads "vs. a baseline foundation
  model".
- **README made concise with collapsible sections**: the 15-skill catalog and
  the Stack table are now `<details>` blocks, so the page reads short at a
  glance and expands on demand.
- **Demo GIF now leads with connecting a Studio key** before the first question
  (`webui/demo/record_demo.py` reordered to key -> tree -> brief -> answer),
  matching the real flow; the GIF and MP4 were re-recorded.
- **ASCII-only typography across docs and code**: replaced every em dash, en
  dash, and horizontal bar (`U+2014` / `U+2013` / `U+2015`) with natural ASCII
  punctuation (commas, colons, periods, parentheses, or ` - `) across all
  in-scope Markdown docs, `src/` strings and docstrings, `webui/`, `scripts/`,
  and the `Makefile` (52 files). Number ranges and compound names became
  hyphens (`1-4`, `Meyer-Pebesma`); arrows (`->`) and code were left alone.
  Line counts are unchanged and 196 tests + ruff + mypy stay green. The
  vendored `olmoearth-skills` submodule and `evals/` fixtures are intentionally
  untouched.
- **README `AGENT` wordmark recolored to white** (`webui/assets/agent-tag.png`):
  the lettering goes from dark teal to white on the same OlmoEarth-pink pill,
  so it reads correctly on both light and dark GitHub themes. The pill shape,
  rounded corners, and transparent background are unchanged.
- **Agent-settings menu labels formalized** (`webui/index.html`): a consistent
  `Title (kind)` style for the Papers & resources links (the
  `Embeddings -> fine-tuning` arrow is gone), `Max steps per run` instead of
  `Max steps / run`, and a header on every section.
- **Landing example briefs follow the selected mode** (`webui/index.html`,
  `webui/app.js`): the "Run a prediction" / "Analyze results" / "Prep &
  configure" tabs now swap the suggested example briefs (not just the input
  placeholder), and selecting a tab reveals them.
- **Example briefs show by default** (`webui/app.js`): the landing reveals the
  suggested briefs on load instead of hiding them behind the toggle.

### Removed
- **Skill #16 `roger-annotation-bridge` dropped**: the planned Roger
  Studio → Studio labelset bridge is no longer part of the project, so the
  catalog is now **15 skills** (was 16). Removed its `SkillSpec` from
  `registry.py` and its section from `SKILLS.md`, and corrected the
  skill-count in `PLAN.md` / `README.md` / `docs/CANON.md` (C9). (Historic
  CHANGELOG entries that mention #16 are left as-is.)

### Fixed
- **Web UI project tree no longer 502s on expand** (`studio/client.py`,
  `serve.py`): `search_predictions` sent a `project_id` field that Studio's
  `/predictions/search` rejects with HTTP 422; it now drops that field and
  filters client-side, like the sibling `search_prediction_results`. The bridge
  also surfaces the real upstream status in the 502 detail instead of a generic
  "Studio call failed".
- **Studio reads retry transient failures** (`studio/client.py`): `load_context`
  (the projects panel) makes two Studio calls, so one transient 502/503/504 or
  timeout failed the whole connect (the panel showed "Couldn't load - HTTP 502"
  while the key was valid). Idempotent reads (GET and the search POSTs) now
  retry up to 3 times with backoff; creates are never retried.
- **"Show example briefs" button now works** (`webui/styles.css`,
  `webui/app.js`, `webui/index.html`): a `.examples { display: flex }` rule beat
  the `[hidden]` attribute, so the chips were always visible and the toggle did
  nothing. Added `.examples[hidden] { display: none }` and a Show/Hide label.
- **Text-emitted tool calls are now recovered (`llm/client.py`)**: when the
  llama.cpp server returns a tool call as plain text (Hermes-XML
  `<function=..><parameter=..>` or `<tool_call>{json}</tool_call>`) instead of
  via the structured `tool_calls` field (which it does without a matching
  `--tool-call-parser`, and intermittently even with one) the client now
  parses it back into a `ToolCall` and sets `finish_reason="tool_calls"`.
  Previously the agent loop mistook the markup for a final answer and silently
  skipped the call. Only triggers when the structured channel is empty, so a
  well-behaved server is untouched. 3 new tests.
- **Underspecified label-array tool schemas (`tools/baseline_compare.py`,
  `tools/evaluate.py`)**: `y_true` / `y_pred` / `*_pred` used
  `{"type": "array", "items": {}}` (items = any type). The grammar
  llama.cpp derives from that lets the model emit the integer arrays as `[]`
  or `[{"value": 1}, …]`, corrupting the call. Typed the items as
  `["integer", "string"]` (class codes or names) so grammar-constrained
  decoding produces clean arrays on the first try.
- **Circular import in `tools/skill_tools.py`**: the module-level
  `from olmoearth_agent.skills.loader import SkillLoader` formed a cycle
  (`skills.loader` → `skills/__init__` → `skills.registry` → `tools.skill_tools`)
  that raised `ImportError` whenever `tools.skill_tools` was imported before
  `skills.registry`. Moved the import inside `build_skill_tools` (lazy) so
  import order no longer matters.

### Added
- **Chat file upload** (`webui/index.html`, `webui/app.js`, `webui/styles.css`):
  attach files in the composer (button or drag-drop). Text files (GeoJSON,
  JSON, CSV, code, txt, md) and PDFs (text extracted client-side via lazy
  pdf.js) are appended to the brief as context; images are accepted but
  labelled "not readable" since the local model is text-only. Per-file and
  total size caps keep the prompt bounded. Frontend-only: the bridge and agent
  are unchanged.
- **Drag a Studio result into chat** (`webui/app.js`, `serve.py`): prediction-
  result nodes in the project tree are draggable; dropping one on the composer
  attaches the result as context (id, properties, format, tile URL). The bridge
  now forwards the result tile URL.
- **`docs/SHOWCASE.md` + `scripts/generate_showcase.py`**: a skills-in-action
  page where **all 15 skills, in catalog order, are driven by the live LLM**:
  each transcript is a captured run of the agent loop against the served
  Qwen3.6 backbone (brief → reasoning → function call → real result →
  answer), nothing fabricated. #5 (read-only) and #13 run against the live
  Studio API; #1-#4 load the vendored `SKILL.md` bodies via
  `olmoearth_load_skill`; #6-#15 are real computation. Falls back to a short
  note for #5/#13 when `OLMOEARTH_API_KEY` is absent. Linked from the README.
  Regenerate with `set -a; . ./.env; set +a;
  uv run python scripts/generate_showcase.py > docs/SHOWCASE.md`.
- **Skill #7 `olmoearth-baseline-compare` (`src/olmoearth_agent/analysis/baseline.py`)**:
  `compare_metrics` runs OlmoEarth vs AlphaEarth head-to-head on shared
  ground truth (reusing skill #8's `classification_metrics`): a per-metric
  table (accuracy / macro-F1 / mean-IoU) with deltas, a per-metric winner,
  and an `overall_winner`. `difference_raster` gives the cell-by-cell gap
  between two score layers (mean / mean-abs / max-abs difference + which
  layer is higher where). Tool `olmoearth_baseline_compare` (metrics
  always; difference raster when score layers are supplied). Substantiates
  an "outperforms AlphaEarth on transfer regions" claim (Ma et al.
  arXiv:2601.00857). **No live GEE connection / Earth Engine MCP**: the
  AlphaEarth side is data the user exported from the now-public GEE
  "Satellite Embedding" dataset, kept the repo's pure/no-deps design.
  13 new tests.
- **Skill #9 `olmoearth-similarity` (`src/olmoearth_agent/analysis/similarity.py`)**:
  `similarity_search` returns the top-K embedding vectors most similar to
  a query (exact brute-force kNN, cosine or Euclidean; FAISS-at-scale is
  the deferred follow-up). `geographic_prior_check` is the honesty guard:
  it **warns when the top matches cluster geographically near the query**,
  because then the "similarity" may reflect *location* (same region /
  biome) rather than genuine feature resemblance: the classic
  similarity-search failure mode (cf. NASA Earthdata Similarity Search;
  OlmoEarth Base wins 15/24 kNN tasks, arXiv:2511.13655). Tool
  `olmoearth_similarity_search` (optional `ids`, `metric`, and
  `query_coord` + `coords` to enable the geographic-prior warning);
  returns matches + a summary, no raw coordinates (rule §3.1). Pure
  Python; reuses `haversine_km` from the evaluate skill. 21 new tests.
- **Skill #10 `olmoearth-uncertainty` (`src/olmoearth_agent/analysis/uncertainty.py`)**:
  `area_of_applicability` implements the Meyer & Pebesma (2021, MEE
  12:1620) Area of Applicability: standardize (optionally
  importance-weight) the training features, compute each point's
  dissimilarity index (nearest-training distance / mean pairwise training
  distance), and flag points whose DI exceeds the training data's own
  outlier-adjusted threshold (`Q75 + 1.5·IQR` of leave-one-out DIs, as in
  the R `CAST` package) as **out-of-distribution**. `ood_flag` returns
  per-point flags + OOD fraction + verdict (within-AOA / partially-OOD /
  mostly-OOD). Tool `olmoearth_area_of_applicability`. The point:
  **softmax confidence is not OOD detection**: a model can be
  confidently wrong on data unlike its training set (AlphaEarth's
  documented transfer failure is exactly what AOA flags). Algorithm-
  agnostic, pure Python (no numpy); the repeated-sampling confidence map
  is the documented follow-up. 18 new tests.
- **Skill #11 `olmoearth-cloud-mask-audit` (`src/olmoearth_agent/analysis/cloud_mask.py`)**:
  `ensemble_disagree` summarizes where several aligned cloud masks
  (CFMask / s2cloudless / Sen2Cor / MAJA, or any others) agree vs
  disagree: agreement/disagreement rates, per-algorithm cloud fraction
  (which algorithm runs aggressive vs conservative), pairwise
  disagreement, and a vote histogram: it surfaces **disagreement, not a
  single ground-truth mask**, because algorithms diverge on thin /
  semi-transparent cloud (Skakun et al. CMIX, RSE 274:112990, 2022).
  `verdict_classifier` takes a model-error mask and returns a
  **bad-mask-vs-bad-model verdict** (cloud-mask-limited / model-limited /
  inconclusive). Tool `olmoearth_cloud_mask_audit` returns summary stats
  only, no per-pixel geometry (rule §3.1). Algorithm-agnostic, pure
  Python; the STAC + s2cloudless `fetch_cloud_masks` step (needs an AOI
  bbox + date, plus heavier deps) is the gated live-smoke follow-up.
  18 new tests.
- **Skill #6 `olmoearth-change-detect` (`src/olmoearth_agent/analysis/change_detect.py`)**:
  `enforce_min_3_dates` + `diff_layers` turn a dated series of per-date
  layer summaries (one `value` per prediction date, e.g. positive-class
  fraction or mean score over the AOI, from a skill #5 result) into
  trajectory metrics: per-step deltas, net change, the largest-change
  interval, a **reversal count**, and a trend label
  (increasing/decreasing/stable/oscillating). **Refuses fewer than 3
  distinct dates**: a two-date diff reports net change but cannot tell a
  steady trend from a reversal (a flood that peaked then receded reads as
  "no change"), so the skill enforces a 3+-date trajectory (`SKILLS.md`
  #6; Ma et al. arXiv:2601.00857). Trend/reversal logic runs on the raw
  deltas, so output rounding can never flip a sign. Tool
  `olmoearth_change_detect` composes skill #5 (`olmoearth-predict`); new
  `analysis/` package (home for the coming Analyze skills). 16 new tests.
- **Skill #12 `olmoearth-qgis-bridge` (`src/olmoearth_agent/reporting/qgis.py`)**:
  `resolve_xyz_url` (relative tile template → absolute QGIS XYZ URL,
  preserving `{z}/{x}/{y}`) + `build_raster_sld` (well-formed OGC SLD 1.0
  color-ramp, default 5-stop YlOrRd over a 0..1 score). Tool
  `olmoearth_qgis_bridge` turns a result's `tile_urls` into XYZ URLs +
  SLD + load instructions (Bearer-auth header note; key never embedded).
  Verified on real PA Karst tiles (resolved URL + valid 5-stop SLD).
  Loading in QGIS desktop is the user's confirmation step; COG export
  follows. 5 new tests.
- **Skill #13 reframed → `olmoearth-data-export`.** The original
  "external-data" (wire third-party GEE/OSM/USGS/NOAA MCPs) needed
  external MCPs and wasn't core; reframed to the self-contained "export
  our own Studio data, grouped." `olmoearth_export_data` writes the
  user's projects + predictions to JSON grouped by `project` (default)
  or `status`, curated to ids/names/statuses/times (no raw geometry).
  `src/olmoearth_agent/reporting/export.py` (pure helpers) +
  `tools/export.py`. Verified live 2026-05-28: 5 project files (12
  predictions linked) + 2 status files from the real account. 7 new
  tests. (`exports/` gitignored.)
- **CLI entrypoint: the agent is now runnable.** `olmoearth-agent
  "<brief>"` (and `python -m olmoearth_agent`) wire the LLM client,
  Studio client, default tool registry, and vendored-skill index into a
  `LeadAgent` and run a natural-language brief. `--show-trace` prints the
  tool-call trace + provenance count to stderr. `src/olmoearth_agent/
  cli.py` + `__main__.py`; `[project.scripts]` console entry. Verified
  live 2026-05-28: `olmoearth-agent "how many projects do I have?"` →
  load_context + search_projects → "You have 5 projects: …". 5 new tests.
- **Project write path verified + `get_project` / `delete_project`.**
  `StudioClient` gains `get_project` (`GET /projects/{id}`) and
  `delete_project` (`DELETE /projects/{id}` → 202; unwraps the deleted
  record nested under `record`). **Verified live 2026-05-28**: a
  throwaway project create → get → delete → 404 (cleaned up, no
  residue). New **double-gated** live test
  `tests/studio/test_write_live.py` (requires `OLMOEARTH_WRITE_TESTS=1`
  *and* `OLMOEARTH_API_KEY`, so it never writes by accident). This
  closes the last unverified core capability: the `POST /projects`
  write path. 3 new tests (2 unit, 1 gated live).
- **Skill #15 `olmoearth-case-narrative` (`src/olmoearth_agent/reporting/`)**:
  `build_narrative` (pure) assembles a stakeholder Markdown report from
  prediction results (tile URLs + properties) + the run's provenance,
  with a **freshness gate** that withholds and strikes through tiles
  older than a configurable window (so a disaster-response brief never
  shows stale imagery). Tool `olmoearth_case_narrative` reads provenance
  from `ThreadState`. Verified live 2026-05-28: agent produced a
  "PA Karst Demo" report. 7 new tests.

### Changed
- **Dropped NVFP4; serving consolidated on the 4-bit GGUF via llama.cpp.**
  NVFP4 (~20 GB) doesn't leave KV-cache headroom on a 24 GB card; the
  GGUF `unsloth/Qwen3.6-35B-A3B-GGUF:UD-IQ4_XS` (~17.7 GB) is the
  verified path. Swept every doc + code default: `README.md`, `PLAN.md`
  (scope / §4 / §6 roadmap / §7.1 / §8), `AGENTS.md`, `docs/serving.md`
  (rewritten around llama.cpp), `.env.example`, `llm/config.py` default
  + module docstrings, test mocks. `docker/vllm.compose.yml` →
  `docker/llama.compose.yml` (llama.cpp service).
- **Renamed LLM env vars**: `VLLM_ENDPOINT` / `VLLM_MODEL` /
  `VLLM_API_KEY` → `LLM_ENDPOINT` / `LLM_MODEL` / `LLM_API_KEY` (the
  endpoint is backend-neutral, no longer vLLM-specific). Verified live.

### Added
- **`docs/CANON.md`**: single source of truth for cross-document facts
  (model, serving stack, quantization, env vars, Studio API, skill
  count) plus a grep-based alignment protocol. Update a fact there
  first, then fix every reference. Prevents the doc drift that motivated
  this pass.
- **Skill #5 predict, result output path**: `olmoearth_fetch_results`
  (tile URLs / property names / file format for a prediction) and
  `olmoearth_get_prediction_result` (by result id). `StudioClient` gains
  `get_prediction_result` + `search_prediction_results`. The API's
  `PredictionResultSearchRequest` has no `prediction_id` filter
  (openapi v0.1.0), so `fetch_results` scans and filters client-side.
  Verified live 2026-05-28 against PA Karst results (tile URLs like
  `/api/v1/prediction-results/{id}/tiles/{z}/{x}/{y}.png?property_name=sample_karst_score`).
  pixel-value / features-search remain the last predict follow-up. 3 new tests.
- **Skill #8 `olmoearth-evaluate` (`src/olmoearth_agent/evaluation/`)**:
  honest map-accuracy tools, pure Python (no heavy deps). `spatial_cv.py`:
  haversine, `spatial_block_folds` (Roberts 2017), `random_folds`, and
  **`cv_inflation_diagnostic`**, the headline: compares mean
  test-to-train nearest-neighbour distance under random vs spatial-block
  CV and reports the inflation ratio + risk band (operationalizes Ploton
  2020 / Meyer-Pebesma 2021). `metrics.py`: per-class
  precision/recall/F1/IoU + accuracy/macro-F1/mean-IoU. Tools
  `olmoearth_cv_inflation_check` + `olmoearth_classification_metrics`.
  Verified live 2026-05-28: agent flagged clustered data with a 353×
  inflation ratio and explained why random CV would overstate accuracy.
  13 new tests. NNDM-LOO (Milà 2022) is the remaining follow-up.
- **Skills #1-#4 vendored** from [`2imi9/OlmoEarth-Skills`](https://github.com/2imi9/OlmoEarth-Skills)
  as a git submodule (`vendor/olmoearth-skills`, pinned `a96427e`). The
  three upstream `SKILL.md` packages (`olmoearth-data-prep` [unifies #1+#2],
  `olmoearth-studio-job-config` [#3], `olmoearth-embeddings` [#4]) are
  consumed via a new `SkillLoader` (`skills/loader.py`) that gives the
  harness agentskills.io-style progressive disclosure: `olmoearth_list_skills`
  (name+description index) and `olmoearth_load_skill` (full `SKILL.md` body).
  `LeadAgent` accepts a brief `skill_index` for its system prompt. Graceful
  when the submodule is not initialized. Verified live 2026-05-28: the agent
  loaded `olmoearth-data-prep` and reported its real steps. 9 new tests.
  (Clone with `git submodule update --init` to populate the vendored skills.)
- **Skill #5 `olmoearth-predict` (core run loop)**: `tools/predict.py`:
  `olmoearth_search_predictions` (discover reusable `model_id`s) and
  `olmoearth_submit_prediction`; poll via the foundational
  `olmoearth_get_prediction`. `StudioClient` gains `search_predictions`
  and `submit_prediction` (the six `PredictionWrite` required fields:
  name, project_id, area_id, model_id, start_time, end_time).
  **Resolves the PLAN.md §4 `model_id` gap for the reuse case**:
  `predictions/search` returns each prediction's `model_id`, so a client
  discovers a reusable id by searching. Read paths verified live
  2026-05-28 (12 real predictions, all with model_ids); an agent run
  found PA Karst model_ids and recovered from a failed tool call mid-run.
  `submit` implemented + unit-tested (not live-created, to avoid side
  effects). Result sub-tools (pixel-value/features/files) are a follow-up
  within this skill. 3 new tests.
- **Skill #14 `olmoearth-provenance` (`src/olmoearth_agent/provenance/`)**:
  implements operational rule §3.13. `ProvenanceLog` lives on
  `ThreadState`; the lead agent records one `ProvenanceManifest` entry
  per dispatched tool call (tool name, sha256 of args, id-only result
  summary, never raw geometry). `to_json()` + `replay_script()` emit
  an auditable manifest and a replay skeleton. Tool bundle
  `olmoearth_provenance_summary` lets the agent report what it did.
  Added `ProvenanceManifest` to `types.py` (was spec-only in PLAN §2).
  Verified live 2026-05-28: agent run recorded `load_context` +
  `provenance_summary` with hashes and result summaries. 7 new tests.
- **Harness core (`src/olmoearth_agent/{types,studio,tools,harness,skills}`)**,
  the structure all 16 skills plug into:
  - `types.py`: harness dataclasses + `ApiEnvelope[T]` (the live
    `{records, meta, errors}` Studio response wrapper found 2026-05-28).
  - `studio/client.py`: async `StudioClient` (httpx, Bearer auth,
    envelope unwrap) with `users_me`, `search_projects`, `create_project`,
    `get_prediction`, `load_context`. Endpoints verified against the live
    API.
  - `tools/registry.py`: `ToolRegistry` + `ToolContext`; dispatch never
    raises (errors return to the model). `tools/studio.py`: the
    foundational `olmoearth_*` tool bundle (load_context / search_projects
    / create_project / get_prediction).
  - `harness/agent.py`: `LeadAgent` ReAct loop (DeerFlow v2 lead-agent
    shape): brief → LLM → tool dispatch → result, with a turn cap and the
    operational rules in the system prompt. `harness/state.py`:
    `ThreadState`.
  - `skills/registry.py`: manifest slotting all 16 skills (number,
    category, status, tools) + `build_default_registry()`.
- **Verified live end-to-end 2026-05-28**: `LeadAgent` →
  `OlmoEarthLLM` (Qwen3.6 4-bit GGUF via llama.cpp) → `StudioClient`
  (live Studio API) → answer. The agent called `olmoearth_load_context`
  and correctly filtered the user's real projects by topic.
- `tests/{studio,tools,harness,skills}`: 16 unit tests (mock HTTP +
  fake LLM) + 1 live integration test (`tests/harness/test_live.py`,
  needs `VLLM_ENDPOINT` + `OLMOEARTH_API_KEY`).
- `pyproject.toml` runtime dep: `httpx>=0.27`.
- **LLM serving client (`src/olmoearth_agent/llm/`)**: async OpenAI-
  compatible wrapper around the vLLM-served Qwen3.6-35B-A3B-NVFP4
  backbone. `OlmoEarthLLM.chat(messages, tools=..., mode=...)` returns
  a parsed `ChatResponse` with content, extracted `<think>` trace,
  tool calls, finish reason, and usage. Four sampling presets from
  the model card; default `thinking_general` with
  `chat_template_kwargs.preserve_thinking=True` for multi-turn agent
  runs. Synchronous `Tracer` protocol exposes request/response hooks
  for the provenance middleware (lands in PR #7).
- `docs/serving.md`: vLLM serve command, hardware requirements,
  **function-calling serve flags** (`--enable-auto-tool-choice
  --tool-call-parser`, required or tool calls come back as text;
  parser name flagged UNVERIFIED pending live confirmation), YaRN
  long-context recipe, agent-mode defaults.
- Client robustness: `_parse_completion` reads server-split
  `reasoning_content` (when served with `--reasoning-parser qwen3`)
  and otherwise extracts the inline `<think>` block, works either way.
- `docs/serving.md`: "Local development on ≤24 GB VRAM" section: 4-bit
  GGUF (`UD-IQ4_XS`) via llama.cpp `server-cuda` with `--jinja` for tool
  calling. **Function-call path verified end-to-end 2026-05-28** on an
  RTX 5090 Laptop (24 GB): NVFP4+vLLM stalls at memory profiling on
  24 GB (residual KV headroom too small), but the 4-bit GGUF loads to
  ~18.6 GB and the agent's `create_project(...)` tool call round-trips.
  Production stack stays vLLM+NVFP4 on datacenter Blackwell; this is a
  local-dev accommodation (same OpenAI protocol, client code unchanged).
- `docker/vllm.compose.yml`: pinned `vllm/vllm-openai:v0.19.0` for
  local dev (still requires Blackwell host).
- `tests/llm/`: mock-endpoint smoke tests via `pytest-httpx`: simple
  chat, `<think>` extraction, tool-call round-trip, preserve_thinking
  forwarding, top_k routing through `extra_body`, tracer hooks. Plus
  one live integration test (`@pytest.mark.integration`) that hits a
  real `vllm serve` instance when `VLLM_ENDPOINT` is set.
- `pyproject.toml` runtime dep: `openai>=1.50`. Dev deps:
  `pytest-asyncio>=0.24`, `pytest-httpx>=0.30`. Pytest config now
  pins `asyncio_mode = "strict"`.
- `.env.example`: `VLLM_ENDPOINT`, `VLLM_MODEL` (defaults pointing at
  local `vllm serve` on `http://localhost:8000/v1`).
- `PLAN.md` §4: explicit Studio API spec-version pin (`openapi.json` v0.1.0,
  pre-1.0) and verified findings on Firebase auth, `PredictionResultAccessLevel`,
  `PredictionUpdate` rename-only, free-form `PredictionRead.progress`,
  `*-management/` doc stubs, and the `TaskStatus`-vs-`PredictionStatus` enum split.
- `PLAN.md` §1: new `olmoearth.create_label` tool row (the Studio API treats
  labelset metadata and individual label classes as separate POSTs).
- `PLAN.md` §2: new `LabelsetSpec`, `LabelDef`, `DataPrepLabelSchema` dataclasses
  (clean separation between Studio-API schemas and the OlmoEarth dataset-prep
  layer's field names).
- **`SKILLS.md`**: detailed 16-skill catalog (Prep / Configure / Run /
  Analyze / Integrate / Report). Each skill has what / why / tools-composed
  with academic citations (Ploton 2020 spatial CV, Meyer-Pebesma 2021 AOA,
  Skakun CMIX 2022 cloud masks, WorldCereal 2025 lessons, IAMAP, NASA
  Similarity Search, etc.). Skill #16 (`roger-annotation-bridge`) added
  alongside the 15 from Ziming's source spec.
- `SKILLS.md`: "Existing implementations (upstream source)" section
  pinning skills #1-#4 to [`2imi9/OlmoEarth-Skills`](https://github.com/2imi9/OlmoEarth-Skills),
  upstream unifies skills #1 + #2 as `olmoearth-data-prep`
  (split-vs-unify decision deferred to first end-to-end skill PR).
  Skill #16 target pinned to [`2imi9/Roger-Studio`](https://github.com/2imi9/Roger-Studio).
- `SKILLS.md`: "Vendoring policy" section, submodule vs copy-with-
  provenance choice deferred to first vendoring PR.
- `PLAN.md` §4 Skills row: references upstream
  [`2imi9/OlmoEarth-Skills`](https://github.com/2imi9/OlmoEarth-Skills)
  as canonical home for skills #1-#4.
- `PLAN.md` §1: three new global tools: `olmoearth.pixel_value`,
  `olmoearth.features_search`, `olmoearth.fetch_embedding` (used by skills
  #4, #5, #9, #10).
- `PLAN.md` §2: four new dataclasses: `PixelValueResult`, `FeatureMatch`,
  `EmbeddingVector`, `ProvenanceManifest`.
- `PLAN.md` §3: new operational rule **13: Provenance manifest** (every
  `olmoearth.*` API call writes a `ProvenanceManifest` entry via
  `provenance_middleware` from skill #14).
- `PLAN.md` §7: new "Future work (parked)" section documenting the
  multimodal stack and train-time self-improvement tracks that are
  explicitly deferred.

### Changed
- `PLAN.md` bumped to **v0.4**. Scope narrowed to text-only LLM
  (`unsloth/Qwen3.6-35B-A3B-NVFP4`) with function calling. Multimodal
  stack (Prismatic / Q-Former / OlmoEarth embedding stream / NVFP4
  fine-tuning) moved to §7.1 Future work; train-time self-improvement
  moved to §7.2.
- `PLAN.md` §4 "Underlying stack" table collapsed from 7 rows to 5:
  dropped "Vision-language model" + "Geospatial encoder stream" +
  "Self-improvement"; added explicit "LLM" row pinning Qwen3.6-35B-A3B-NVFP4
  served via **vLLM ≥0.19.0** (upstream-recommended path for this NVFP4
  checkpoint; full `vllm serve` command in §4). Agent sessions use
  `chat_template_kwargs.preserve_thinking=True` per the model card.
- `PLAN.md` §7 "Future work" trimmed from per-bullet ref lists to two
  prose paragraphs. Noted that Qwen3.6 ships a native vision encoder,
  so re-opening §7.1 means using/replacing that tower, not training
  one from scratch.
- `SKILLS.md` "Existing implementations" + "Vendoring policy" sections
  tightened (split-vs-unify and submodule-vs-copy decisions left as
  one-line options rather than verbose A/B writeups).
- `PLAN.md` §6 roadmap rewritten from 7 generic phases (P0-P6) to
  skill-first: P0-P2 done (scaffold / gap closure / this rewrite),
  P3 = LLM serving + harness MVP, then one PR per skill ordered by
  case-study demand. Skills 14 (provenance) and 8 (evaluate) flagged
  for early landing because they're cross-cutting.
- `PLAN.md` §8 (was §7) references trimmed: in-scope LLM refs only;
  parked refs live in §7.1/§7.2.
- `PLAN.md` §3 rule list grew from 12 → 13 (provenance manifest).

### Changed (continued from PR #3)
- `PLAN.md` bumped to v0.3 (PR #3 increment, superseded by v0.4 here).
- `PLAN.md` §4: rewritten "Studio gaps" subsection from v0.1/v0.2's three
  UNVERIFIED items to verified findings: webhook absence CLOSED, fine-tune
  `model_id` field CONFIRMED but provenance still UNVERIFIED, rate limits
  CLOSED-as-undocumented.
- `PLAN.md` §5 example: uses `LabelsetSpec` + `create_label` flow and notes
  that every Studio Prediction requires a `model_id`.

### Changed
- `PLAN.md` bumped to v0.3.
- `PLAN.md` §4: rewritten "Studio gaps" subsection from v0.1/v0.2's three
  UNVERIFIED items to verified findings: webhook absence CLOSED, fine-tune
  `model_id` field CONFIRMED but provenance still UNVERIFIED, rate limits
  CLOSED-as-undocumented.
- `PLAN.md` §5 example: uses `LabelsetSpec` + `create_label` flow and notes
  that every Studio Prediction requires a `model_id`.

### Fixed
- `PLAN.md` §2 `PredictionStatus.state` enum corrected against the live
  `components.schemas.PredictionStatus`: `queued`→`pending`, `succeeded`→
  `completed`, added `cancelled` as a fifth terminal state.
- `PLAN.md` §2 `LabelSchema` retired: the v0.2 shape (`sample_category`,
  `es_label`, `oe_labels`) is the OlmoEarth dataset-prep / rslearn layer's
  schema, NOT the Studio API. Renamed to `DataPrepLabelSchema` and
  documented as a different layer; `LabelsetSpec` + `LabelDef` replace it
  for the API surface.
- `PLAN.md` §2 `PredictionRef.kind` docstring clarifies it's a client-side
  dispatch key, not a Studio API field (the API has no `kind` / `task_type`).

## [0.0.2] - 2026-05-27

### Added
- `CONTRIBUTING.md` modeled on NVIDIA earth2studio's developer workflow
  (DCO sign-off required, pre-commit mandatory, PR title into CHANGELOG,
  90% coverage gate, Black / Ruff / mypy / interrogate / NumPy docstrings /
  Apache-2.0 + SPDX headers).
- `AGENTS.md` following the [agents.md](https://agents.md/) spec for
  coding-agent onboarding.
- AI-assisted contribution policy: disclosure in PR description; no AI
  co-author trailers in commits (per OpenInfra Foundation convention).
- Branch naming convention: `2imi9/feature-<slug>`; no direct commits to `main`.

### Changed
- LICENSE copyright line updated to "OlmoEarth Agent contributors".

## [0.0.1] - 2026-05-26

### Added
- Initial `PLAN.md` and `README.md` defining the tool catalog, harness data
  classes, operational rules, and underlying-stack references, modeled on
  Google's Google Earth Agent shape (catalog + dataclasses + numbered rules).
- Apache-2.0 LICENSE.
- `.gitignore` covering Python build artifacts, virtual environments, test
  caches, type-checker caches, secrets (`.env` family, key files), model and
  data artifacts (safetensors, checkpoints, GeoTIFF, Zarr, Parquet), Hugging
  Face caches, and Claude Code agent state.
