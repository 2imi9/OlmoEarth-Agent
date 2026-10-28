/* The 18-skill capability grid + the pop-out introduction modal + the in-app
   full-spec detail page. Self-contained: clicking a card opens the modal;
   "Full spec" opens the detail page; "Use this brief" drops the example into
   the composer. A card with optIn: true is a skill whose tools are all
   opt-in (OLMOEARTH_OPT_IN_TOOLS=1 on the bridge; SKILLS.md "Opt-in tools");
   other cards name their opt-in tools in their text. */

import { escapeHtml, autosize } from './util.js';

const ICONS = {
  upload:      '<path d="M12 16V4M7 9l5-5 5 5"/><path d="M5 20h14"/>',
  sliders:     '<path d="M4 8h8M16 8h4M4 16h4M12 16h8"/><circle cx="14" cy="8" r="2"/><circle cx="9" cy="16" r="2"/>',
  wand:        '<path d="M5 19l9-9"/><path d="M14 4l.9 2.6L17.5 7.5l-2.6.9L14 11l-.9-2.6L10.5 7.5l2.6-.9z"/>',
  branch:      '<circle cx="6" cy="6" r="2"/><circle cx="6" cy="18" r="2"/><circle cx="18" cy="12" r="2"/><path d="M6 8v8M8 6h4a4 4 0 014 4M8 18h4a4 4 0 004-4"/>',
  satellite:   '<circle cx="12" cy="12" r="2"/><path d="M8 8a5.5 5.5 0 000 8M16 8a5.5 5.5 0 010 8M5 5a10 10 0 000 14M19 5a10 10 0 010 14"/>',
  trend:       '<path d="M4 17l5-5 3 3 7-8"/><path d="M15 7h5v5"/>',
  compare:     '<path d="M12 3v18"/><path d="M6 7L2 14h8zM18 7l-4 7h8z"/><path d="M5 21h14"/>',
  barcheck:    '<path d="M4 20V11M10 20V4M16 20v-6"/><path d="M3 20h18"/>',
  search:      '<circle cx="11" cy="11" r="7"/><path d="M21 21l-4-4"/>',
  shield:      '<path d="M12 3l8 3v6c0 5-3.5 8-8 9-4.5-1-8-4-8-9V6z"/><path d="M9 12l2 2 4-4"/>',
  cloud:       '<path d="M7 18h10a4 4 0 000-8 6 6 0 00-11.5 2A3.5 3.5 0 007 18z"/>',
  layers:      '<path d="M12 3l9 5-9 5-9-5 9-5z"/><path d="M3 13l9 5 9-5"/>',
  database:    '<ellipse cx="12" cy="6" rx="8" ry="3"/><path d="M4 6v12c0 1.7 3.6 3 8 3s8-1.3 8-3V6"/><path d="M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3"/>',
  fingerprint: '<path d="M5 11a7 7 0 0114 0M8 12a4 4 0 018 0v2M12 13v5M8 15v3M16 15v2"/>',
  docspark:    '<path d="M7 3h7l5 5v13H7z"/><path d="M14 3v5h5"/><path d="M10.5 13l.7 1.8 1.8.7-1.8.7-.7 1.8-.7-1.8-1.8-.7 1.8-.7z"/>',
  scatter:     '<circle cx="6" cy="7" r="1.4"/><circle cx="17" cy="6" r="1.4"/><circle cx="9" cy="16" r="1.4"/><circle cx="18" cy="15" r="1.4"/><circle cx="12" cy="11" r="2.4"/><path d="M3 20h18"/>',
};

const SKILLS = [
  { n: 1,  slug: 'data-prep',         cat: 'Prep',      icon: 'upload',      desc: 'Labels → a Studio-importable file (MIME / 10K / multi-metric guards) AND an rslearn dataset.json + Lightning YAML, with a 7-criteria audit.', ex: 'I have 3,000 field plots as a GeoJSON - get them into Studio without the Windows MIME error, then build the rslearn dataset.json.' },
  { n: 2,  slug: 'studio-job-config', cat: 'Configure', icon: 'wand',        desc: 'Task description → Studio job-wizard answers; 14 presets + a cross-field validator.', ex: 'I want per-pixel mangrove classification from Sentinel-2 - fill in the Studio job wizard.' },
  { n: 3,  slug: 'embeddings',        cat: 'Configure', icon: 'branch',      desc: "Embeddings-vs-fine-tune: guidance + a runnable notebook; for an annual task, Ai2's precomputed v1.3 embeddings replace computing them yourself. The one-call decide-and-configure is opt-in.", ex: 'I have 150 labels and a Colab T4 - should I fine-tune or use embeddings? Give me a notebook.' },
  { n: 4,  slug: 'predict',           cat: 'Run',       icon: 'satellite',   desc: 'The core run loop: find a model, submit, poll, fetch result tiles, and compare results.', pink: true, ex: 'Run a flood-extent prediction over this AOI for last month and return the result tiles.' },
  { n: 5,  slug: 'change-detection',  cat: 'Run',       icon: 'trend',       desc: "Change over time: one model's dated Studio results traced as a series - estimate movement, not verified ground change.", pink: true, ex: 'Did forest cover decline across these four quarterly snapshots, or is it just noise?' },
  { n: 6,  slug: 'baseline-compare',  cat: 'Run',       icon: 'compare',     desc: 'OlmoEarth vs a baseline foundation model (e.g. AlphaEarth), head-to-head on transfer regions; its winner is untested, so which map is more accurate is #18\'s two-map route.', pink: true, optIn: true, ex: 'Compare OlmoEarth vs AlphaEarth for land cover in a region where AlphaEarth struggles.' },
  { n: 7,  slug: 'evaluate',          cat: 'Analyze',   icon: 'barcheck',    desc: 'Per-class metrics against your labels; the spatial CV inflation check and NNDM-LOO are opt-in.', ex: "Here are my 200 field labels and the map's classes at those points: what are its per-class precision and recall?" },
  { n: 8,  slug: 'similarity',        cat: 'Analyze',   icon: 'search',      desc: 'Top-K embedding search with a geographic-prior warning.', optIn: true, ex: 'Find the 20 patches most similar to this illegal-mining site across the basin.' },
  { n: 9,  slug: 'uncertainty',       cat: 'Analyze',   icon: 'shield',      desc: 'Ensemble disagreement across Studio results; the Area-of-Applicability out-of-distribution flag is opt-in.', ex: 'Where do my three runs of this model disagree most?' },
  { n: 10, slug: 'cloud-mask-audit',  cat: 'Analyze',   icon: 'cloud',       desc: 'CFMask / s2cloudless / Sen2Cor / MAJA disagreement: bad mask vs bad model.', optIn: true, ex: 'My prediction looks wrong over this scene - bad cloud mask or bad model?' },
  { n: 11, slug: 'qgis-bridge',       cat: 'Integrate', icon: 'layers',      desc: 'Tile URLs → a QGIS XYZ layer + an OGC SLD style, ready to load.', ex: 'Give me a QGIS layer + SLD style for this prediction so I can open it on my desktop.' },
  { n: 12, slug: 'data-export',       cat: 'Integrate', icon: 'database',    desc: 'Export Studio projects + predictions to JSON, grouped by project or status.', ex: 'Export all my Studio projects and their predictions to JSON, grouped by status.' },
  { n: 13, slug: 'provenance',        cat: 'Report',    icon: 'fingerprint', desc: 'A manifest over every tool call, plus a one-command replay script.', ex: 'Produce a replay script + manifest so an auditor can reproduce this prediction.' },
  { n: 14, slug: 'case-narrative',    cat: 'Report',    icon: 'docspark',    desc: 'A stakeholder Markdown brief with a freshness gate on stale tiles.', ex: 'Write a stakeholder brief for this karst-vulnerability result with the live map tiles.' },
  { n: 15, slug: 'litsearch',         cat: 'Report',    icon: 'search',      desc: 'arXiv + OpenAlex search with DOI / arXiv-id resolution to ground citations.', optIn: true, ex: 'Find and cite the paper behind the Area-of-Applicability method I used.' },
  { n: 16, slug: 'negative-sampler',  cat: 'Prep',      icon: 'scatter',     desc: 'Presence-only labels into a trainable set: buffered, thinned (optionally embedding-dissimilar) background class.', optIn: true, ex: 'My karst-site labels are presence-only and the audit fails for a missing negative class - generate background samples.' },
  { n: 17, slug: 'rslearn',           cat: 'Configure', icon: 'sliders',     desc: 'Run-the-pipeline guidance for rslearn; the tools that recommend a setup, compose and validate the finetune model.yaml, and diagnose a failed run are opt-in.', ex: 'I train my own model with rslearn: give me the commands to stage this flooded-buildings dataset from Sentinel-2 and fit the model.' },
  { n: 18, slug: 'review-set',        cat: 'Analyze',   icon: 'filter',      desc: "Which windows to check first, without labels: rank by the model's own top-1-minus-top-2 margin, at your review budget, boundary-first if you want it - plus the ceiling arithmetic; then how wrong the map is, from a labelled sample, and which of two maps of one time is more accurate, from labels where they differ. The grader for a candidate audit rule is opt-in.", ex: 'The map is done and I can only check 200 of these 4,000 windows. Which 200, and how much of the error will I actually catch?' },
];

/* The card's stage line: its category, and "opt-in" for an opt-in skill. */
const catLabel = (s) => (s.optIn ? `${s.cat} · opt-in` : s.cat);

/* Minimal {slug, desc, n} list for the composer's "/" skill slash-commands. */
export const SKILL_LIST = SKILLS.map((s) => ({ slug: s.slug, desc: s.optIn ? `${s.desc} (opt-in)` : s.desc, n: s.n }));

// Fuller "what it does + why" per skill, shown in the pop-out. The card shows
// the one-line desc above; this is the spec summary. Full academic spec: SKILLS.md.
const SKILL_SPECS = {
  1: `Labels → a Studio-importable file and/or an rslearn dataset.json + Lightning YAML. Enforces the sample_category / es_label / oe_labels schemas, shards at 10K records, works around the Windows .geojson MIME rejection, splits multi-metric files, handles the es_label rename and watershed AOIs (NLDI / HUC-12), and runs a 7-criteria audit before a multi-hour training run fails silently. The single vendored olmoearth-data-prep package.`,
  2: `Maps a task description to Studio wizard answers - output type, model size, time-frame mode, S2-vs-+S1 sources, patch size - from 14 verified presets, with a cross-field validator that catches traps (detection + 320m patch, embeddings + single-moment) before you submit.`,
  3: `The embeddings-vs-fine-tune decision grounded in the OlmoEarth accuracy / time / VRAM table: a parameterized notebook that extracts Nano/Tiny/Base/Large embeddings and trains kNN + linear-probe heads (you run it). Ai2 now publishes precomputed OlmoEarth v1.3 embeddings (source.coop/olmoearth/olmoearth-v1-3-embeddings): one vector per 10 m pixel per year, from a different model than the notebook's. They replace extracting embeddings yourself for an annual task in an area and year they cover; a task that needs dates within a year still needs the notebook. Opt-in (OLMOEARTH_OPT_IN_TOOLS=1): a one-call olmoearth_automate that decides programmatically and proposes a config (model size, head, notebook/fine-tune plan), optionally reading a Hugging Face dataset's rows + classes.`,
  4: `The core run primitive: discover a reusable model_id, submit a prediction, poll with backoff, then fetch results - XYZ raster tiles, MVT vectors, or pixel values - and compare results with no ground truth. Async-by-reference, and cost-guarded on fine-tunes. AOIs can be drawn on a map in the chat (olmoearth_request_aoi) and stored as a Studio area.`,
  5: `Change over time: olmoearth_compare_results in series mode traces one model's estimates across 3-8 dated results - chronological ordering discovered from each prediction, per-step pixel correlation, per-point trajectories, and shift sizes calibrated to the field's value range (estimate movement, not verified ground change). It refuses fewer than 3 dates, because a 2-date diff cannot tell a steady trend from a flood that peaked then receded. Opt-in: olmoearth_change_detect, the trajectory of a series of values you give. A JEPA latent-change detector lives in a separate repo and is not connected to this agent.`,
  6: `Opt-in (OLMOEARTH_OPT_IN_TOOLS=1): its winner is a point metric on the labels you give, with no design, no interval and no test, so it does not say which map is more accurate. Runs OlmoEarth vs a baseline foundation model head-to-head on shared ground truth: a per-metric table (accuracy / macro-F1 / mean-IoU) with deltas and an overall winner, plus a cell-by-cell difference raster. Which of two maps of one time is more accurate is #18's two-map route, for two Studio results or two scores files of one window grid: labels on a random sample of the windows where they differ, graded by the exact sign test or the interval (which may name neither map), with an interval on the difference.`,
  7: `Accuracy and per-class precision / recall / F1 / IoU against labels you give. Opt-in, as training-side checks run outside Studio: the random-vs-spatial-block CV inflation ratio and risk band (Ploton 2020 / Meyer-Pebesma 2021) and an NNDM-LOO estimate. For a Studio map's error rate with an interval, use #18's design-based estimation.`,
  8: `Opt-in (OLMOEARTH_OPT_IN_TOOLS=1): Ai2's Explorer and AlphaEarth in Earth Engine search every pixel. Returns the top-K embedding vectors most similar to a query (cosine or Euclidean kNN), with a geographic-prior warning when the matches cluster near the query - because then "similarity" may reflect location (same biome) rather than genuine feature resemblance.`,
  9: `Ensemble disagreement across two or more distinct Studio results (olmoearth_compare_results, mode ensemble): per-window spread as a confidence, the signal that works when Studio gives hard classes only. Opt-in: the Meyer-Pebesma Area of Applicability over feature vectors you supply, which no tool produces and which was measured not to rank errors better than the margin.`,
  10: `Opt-in (OLMOEARTH_OPT_IN_TOOLS=1): no tool produces its input. For a direct OlmoEarth run, olmoearth-inferencex's input-condition layer covers the question through its own CLI (from-olmoearth --conditions, then assess --condition), recording cloud only where the run's data hold Sentinel-2's SCL band; no agent tool does this, and a Studio result has no condition layer. Summarizes where several aligned cloud masks (CFMask / s2cloudless / Sen2Cor / MAJA) agree vs disagree, then returns a bad-mask-vs-bad-model verdict for a model-error region.`,
  11: `Resolves a prediction's relative tile template into an absolute QGIS XYZ URL and builds a well-formed OGC SLD color-ramp style, with Bearer-auth load instructions. Closes the hot-cloud-to-cold-desktop loop for the QGIS audience without embedding your key.`,
  12: `Exports your Studio projects and their predictions to JSON, grouped by project or status and curated to ids / names / statuses / times (no raw geometry). The self-contained alternative to wiring third-party data MCPs.`,
  13: `Records one manifest entry per tool call - tool name, sha256 of args, an id-only result summary, never raw geometry - and emits an auditable manifest plus a runnable replay skeleton. Serves EUDR / REDD+ MRV audit needs and the EO reproducibility gap.`,
  14: `Assembles a stakeholder Markdown report from prediction results and the run's provenance, with a freshness gate that withholds and strikes through tiles older than a configurable window - so a disaster-response brief never shows stale imagery.`,
  15: `Opt-in (OLMOEARTH_OPT_IN_TOOLS=1): Ai2's Asta covers literature search. Unified arXiv + OpenAlex search and DOI / arXiv-id resolution, deduped across sources and key-free (OpenAlex polite pool). Grounds EO citations in real papers instead of world-knowledge or hallucinated links, under a no-fabrication, cite-the-real-URL contract.`,
  16: `Opt-in (OLMOEARTH_OPT_IN_TOOLS=1): label preparation for model training outside Studio. Generates the missing negative/background class for a presence-only label set so it becomes trainable. Drops candidate points within a buffer of any positive, keeps the accepted negatives spatially thinned, and - when the inputs carry embeddings - ranks candidates by environmental dissimilarity to the positives (the inverse of #8). Writes a combined GeoJSON that round-trips straight back through the data-prep audit, converting its hard FAIL on a missing negative class into a PASS. Deterministic, no GDAL; surfaces a placement shortfall as a warning rather than under-filling silently.`,
  17: `rslearn - the data + training engine under OlmoEarth - made usable by non-experts. The vendored SKILL.md loads by default; the four tools below are opt-in (OLMOEARTH_OPT_IN_TOOLS=1), as they serve model training outside Studio. olmoearth_rslearn_recommend turns a plain-language goal into a complete, explained setup: the task (segmentation / per-pixel regression / detection / classification / regression), the data layout (data_source + space_mode + compositing + bands), the model composition (OlmoEarth encoder → decoder → head with the channel contract), the task knobs (metrics / loss / nodata / scale_factor), and a freeze→unfreeze schedule. olmoearth_rslearn_validate catches the errors rslearn only surfaces hours in - encoder-dim vs decoder in_channels, out_channels vs num_classes, task vs label-type, missing bands, the Faster R-CNN background-class +1. olmoearth_rslearn_compose emits the full finetune model.yaml (MultiTaskModel + the right decoder/head); olmoearth_rslearn_diagnose turns a failing prepare/ingest/materialize run into plain-English fixes. Torch-free guidance; the vendored SKILL.md guides the 4-stage data pipeline + model fit/predict, which the user runs.`,
  18: `The question a practitioner asks the moment a prediction map lands: which windows do I open first, and how much of the error do I catch if I only open 5% of them? olmoearth_review_set ranks windows by the model's own top-1-minus-top-2 margin (low margin = suspect), optionally boundary-first using the nine-level indicator (how many of the 8 neighbours were predicted as a different class). olmoearth_grade_review_rule (opt-in: it needs per-unit error labels a Studio user rarely has) is the honest broker: it scores any candidate suspicion signal against that margin and a no-model control, with the scene as the unit of replication and a one-sided exact sign test. olmoearth_review_budget_ceiling is the arithmetic guard - no rule can catch more than min(1, budget/error_rate), so a capture of 0.45 at a 10% budget with a 20% error rate is 90% of what was reachable, not '45%'. Measured, not asserted: on the tasks of Ai2's own published embedding suite the margin ranks errors better than a random order - its AUROC for errors is above 0.5 in all 3,560 cells of 16 encoders (OlmoEarth's four sizes among them) at ten probe seeds, and with OlmoEarth Base its 10% review set holds a median 21.4% of the errors against 10% for a random 10%. Its lead over the suite's two no-model controls (24 of 24 tasks) says little, since those controls are near chance there. Ensemble disagreement, cross-model disagreement and feature-space typicality were each measured against the margin and none ranked errors better. Bring-your-own scores - Studio returns hard classes only, which is exactly why skill #9 reaches for ensemble disagreement instead.`,
};

// Tools each skill composes, shown on the full-spec detail page. From PLAN.md section 1 + SKILLS.md.
const SKILL_TOOLS = {
  1: `olmoearth_load_skill · validate_studio_mime · shard_at_10k · split_multi_metric · write_rslearn_config · audit_7_criteria · rename_es_label`,
  2: `olmoearth_load_skill · Studio job-wizard presets (14) + a cross-field validator`,
  3: `olmoearth_load_skill (guidance + notebook) · opt-in: olmoearth_automate (decide + propose_config + fetch_hf_dataset_profile) · decision_matrix · knn_head · linear_probe_head`,
  4: `olmoearth_search_predictions · olmoearth_submit_prediction · olmoearth_get_prediction · olmoearth_fetch_results · olmoearth_get_prediction_result · olmoearth_pixel_value · olmoearth_compare_results (a pair or a group of results) · olmoearth_request_aoi`,
  5: `olmoearth_compare_results, mode series (date-ordered, legend-calibrated shift trace) · opt-in: olmoearth_change_detect (enforce_min_3_dates · diff_layers)`,
  6: `opt-in: olmoearth_baseline_compare · compare_metrics (reuses #7 classification_metrics) · difference_raster`,
  7: `olmoearth_classification_metrics · opt-in: olmoearth_cv_inflation_check · olmoearth_nndm_cv · spatial_block_folds · haversine`,
  8: `opt-in: olmoearth_similarity_search · similarity_search (brute-force kNN) · geographic_prior_check`,
  9: `olmoearth_compare_results, mode ensemble (disagreement across results) · opt-in: olmoearth_area_of_applicability · dissimilarity index (Meyer-Pebesma) · ood_flag`,
  10: `opt-in: olmoearth_cloud_mask_audit · ensemble_disagree · verdict_classifier`,
  11: `olmoearth_qgis_bridge · resolve_xyz_url · build_raster_sld`,
  12: `olmoearth_export_data (reads projects + predictions, curated)`,
  13: `olmoearth_provenance_summary · ProvenanceLog on ThreadState · replay_script`,
  14: `olmoearth_case_narrative · build_narrative (reads provenance + results)`,
  15: `opt-in: olmoearth_litsearch (search by query, or resolve a DOI / arXiv id; arXiv + OpenAlex, deduped)`,
  16: `opt-in: olmoearth_negative_sampler · analysis.negative_sampler sample_negatives (buffer + farthest-point / embedding-dissimilarity, reuses spatial_cv.haversine_km)`,
  17: `olmoearth_load_skill (guidance for the 4-stage pipeline + fit/predict, which the user runs) · opt-in: olmoearth_rslearn_recommend (goal → explained setup) · olmoearth_rslearn_validate (shape / label-type / band checks) · olmoearth_rslearn_compose (emit the finetune model.yaml) · olmoearth_rslearn_diagnose (failed run → fixes)`,
  18: `olmoearth_review_set (scores → the ordered review set at a budget) · olmoearth_review_set_from_result (a Studio result's [0, 1] score → the same margin ranking) · olmoearth_compare_review · olmoearth_review_budget_ceiling (min(1, budget/error_rate) and the share of it reached) · olmoearth_plan_label_sample → olmoearth_estimate_map_error / olmoearth_certify_zone (design-based error rate; optional inferencex extra) · olmoearth_plan_label_sample (design='disagreement') → olmoearth_estimate_map_error (which of two maps is more accurate: sample_disagreement, compare_from_disagreement, plan_which_map) · olmoearth_scores_from_file (a direct run's files: score_area.py's, or from-olmoearth's scores raster and olmoearth_output.json) · olmoearth_plan_label_sample (design='random', seed_from_map) → olmoearth_certify_zone → olmoearth_write_certificate / olmoearth_verify_certificate (a certificate anyone holding the map can check) · opt-in: olmoearth_grade_review_rule (candidate vs margin vs no-model control, per-group sign test)`,
};

// What each skill takes in and gives back, for the full-spec detail page.
// Concise In -> Out, mirroring the "In:/Out:" lines in SKILLS.md.
const SKILL_IO = {
  1:  { in: 'Labels as GeoJSON / CSV / Shapefile (+ optional AOIs).', out: 'A Studio-importable file and/or an rslearn dataset.json + Lightning YAML, through MIME / 10K / multi-metric guards and a 7-criteria audit.' },
  2:  { in: 'A free-text task description.', out: 'Studio job-wizard answers (output type, model size, time frame, sources, patch size) + a cross-field validation.' },
  3:  { in: 'Sample count, class count, compute tier, goal (opt-in: a HF dataset id).', out: 'An embeddings-vs-fine-tune recommendation + a runnable extraction notebook (opt-in: a proposed config).' },
  4:  { in: 'A model_id, an AOI (draw it on the map), and a time range.', out: 'A submitted prediction, then result tiles / pixel values (async by reference), and comparisons of results.' },
  5:  { in: '3-8 result ids of one model over one area (opt-in: 3+ dated values you give).', out: 'Per-step correlation + per-point trajectories + legend-calibrated shifts (opt-in: trajectory metrics; refuses <3 dates).' },
  6:  { in: '(opt-in) OlmoEarth + a baseline model on shared ground truth.', out: 'A per-metric table (accuracy / macro-F1 / mean-IoU) with deltas, an untested winner, and a difference raster.' },
  7:  { in: 'Your labels and the map\'s classes at the same points (opt-in: labelled points and prediction points).', out: 'Per-class precision / recall / F1 / IoU (opt-in: a random-vs-spatial CV inflation ratio + risk band and an NNDM-LOO estimate).' },
  8:  { in: 'A query embedding + a corpus (optionally parallel coords).', out: 'Top-K nearest matches + a geographic-prior warning when matches cluster near the query.' },
  9:  { in: 'Two or more Studio results of one quantity (opt-in: training-set + query embeddings).', out: 'Per-window ensemble spread as a confidence (opt-in: a Meyer-Pebesma Area-of-Applicability flag marking out-of-distribution points).' },
  10: { in: 'Aligned cloud masks (CFMask / s2cloudless / Sen2Cor / MAJA).', out: 'A disagreement summary + a bad-mask-vs-bad-model verdict for a suspect region.' },
  11: { in: "A prediction's relative tile template.", out: 'An absolute QGIS XYZ URL + an OGC SLD colour-ramp style + Bearer-auth load steps.' },
  12: { in: 'Your Studio projects + predictions.', out: 'Curated JSON files (ids / names / statuses / times; no geometry) grouped by project or status.' },
  13: { in: "The run's tool-call log on ThreadState.", out: 'An append-only manifest (tool, sha256 of args, id-only summary) + a runnable replay script.' },
  14: { in: 'Prediction results + the run provenance.', out: 'A stakeholder Markdown brief with live tiles and a freshness gate that strikes through stale imagery.' },
  15: { in: 'A free-text query, or a single DOI / arXiv id.', out: 'Curated paper records (title, authors, year, venue, url, citations), deduped across arXiv + OpenAlex.' },
  16: { in: 'A presence-only labels GeoJSON path (one positive class).', out: 'A combined GeoJSON with a buffered, spatially-thinned negative class that passes the data-prep audit.' },
  17: { in: 'A dataset and model to stage and train with rslearn (opt-in: a plain-language research goal + labels / sensor / classes / range, or a dataset config.json + model.yaml).', out: 'The rslearn pipeline and fit/predict commands, for you to run (opt-in: a complete, explained rslearn setup, or a validation report catching shape / label-type / band errors before a training run).' },
  18: { in: 'Per-window model scores (n_windows x n_classes logits or probabilities), a direct run\'s files (score_area.py\'s or from-olmoearth\'s) or a Studio result whose band is a [0, 1] score, a review budget, optionally a [rows, cols] grid; (opt-in) for grading, a candidate signal + a 0/1 error indicator; for estimation, a label budget, then the reviewer\'s 0/1 labels (for a Studio class map, the class seen in each window); for two maps of one time (class maps included), a label budget, then the class seen in each window where they differ.', out: 'The ordered review set with margins and boundary counts, the attainable-ceiling arithmetic, and the measured evidence + honest caveats; (opt-in) for grading, per-arm E-AURC / AUROC / capture with a head-to-head verdict and a per-group sign test; for estimation, the error rate with the interval valid at every sample size first, then the one its design earns (a review set is not a sample), and for a direct run\'s plan drawn with seed_from_map, a certificate anyone holding the map can check; for a Studio class map, the error rate and per-class accuracy the same way, and no review set or certified zone; for two maps, which is more accurate by the exact sign test, and the difference in accuracy with a 95% interval.' },
};

export function renderCards() {
  const grid = document.getElementById('capGrid');
  if (!grid) return;
  // Each card is a button: click (or Enter/Space) pops out an intro modal.
  grid.innerHTML = SKILLS.map((s) => `
    <article class="card${s.pink ? ' is-pink' : ''}" role="button" tabindex="0" aria-haspopup="dialog" data-n="${s.n}">
      <div class="card-top">
        <span class="card-ic"><svg viewBox="0 0 24 24" class="ic">${ICONS[s.icon] || ''}</svg></span>
        <span class="card-num">#${s.n}</span>
      </div>
      <div class="card-cat">${catLabel(s)}</div>
      <div class="card-name">olmoearth-${s.slug}</div>
      <p class="card-desc">${s.desc}</p>
      <div class="card-hint">
        <span>Example brief</span>
        <svg viewBox="0 0 24 24" class="ic card-chev"><path d="M5 12h14M13 6l6 6-6 6"/></svg>
      </div>
    </article>`).join('');
}

/* Clickable skill cards pop out an introduction modal (the grid stays uniform
   and in order). "Use this brief" drops the example into the composer. */
let _skillModalReturn = null;

function openSkillModal(s) {
  const modal = document.getElementById('skillModal');
  if (!modal) return;
  modal.innerHTML = `
    <div class="modal-backdrop" data-close></div>
    <div class="modal-card${s.pink ? ' is-pink' : ''}" role="dialog" aria-modal="true" aria-labelledby="skillModalName">
      <button class="modal-close" type="button" aria-label="Close" data-close>&times;</button>
      <div class="modal-top">
        <span class="card-ic"><svg viewBox="0 0 24 24" class="ic">${ICONS[s.icon] || ''}</svg></span>
        <span class="modal-num">#${s.n}</span>
      </div>
      <div class="card-cat">${catLabel(s)}</div>
      <h3 class="modal-name" id="skillModalName">olmoearth-${s.slug}</h3>
      <p class="modal-desc">${SKILL_SPECS[s.n] || s.desc}</p>
      <div class="modal-ex-label">Example brief</div>
      <p class="card-ex">${escapeHtml(s.ex)}</p>
      <div class="modal-actions">
        <button class="card-use" type="button" data-use>Use this brief
          <svg viewBox="0 0 24 24" class="ic"><path d="M5 12h14M13 6l6 6-6 6"/></svg>
        </button>
        <button class="card-use is-secondary" type="button" data-detail>Full spec
          <svg viewBox="0 0 24 24" class="ic"><path d="M7 3h7l5 5v13H7z"/><path d="M14 3v5h5"/></svg>
        </button>
      </div>
    </div>`;
  modal.dataset.n = s.n;
  modal.hidden = false;
  const closeBtn = modal.querySelector('.modal-close');
  if (closeBtn) closeBtn.focus();
}

function closeSkillModal() {
  const modal = document.getElementById('skillModal');
  if (!modal || modal.hidden) return;
  modal.hidden = true;
  modal.innerHTML = '';
  if (_skillModalReturn) { _skillModalReturn.focus(); _skillModalReturn = null; }
}

/* "Full spec" opens an in-app detail page (not a GitHub jump): the full spec,
   the tools the skill composes, and the example brief. */
function openSkillDetail(s) {
  const d = document.getElementById('skillDetail');
  if (!d) return;
  d.innerHTML = `
    <div class="detail-bar">
      <button class="detail-back" type="button" data-detail-close>
        <svg viewBox="0 0 24 24" class="ic"><path d="M19 12H5M11 18l-6-6 6-6"/></svg> Back to skills
      </button>
    </div>
    <div class="detail-body">
      <div class="detail-head${s.pink ? ' is-pink' : ''}">
        <span class="card-ic"><svg viewBox="0 0 24 24" class="ic">${ICONS[s.icon] || ''}</svg></span>
        <div class="detail-head-txt">
          <div class="card-cat">${catLabel(s)}</div>
          <h2 class="detail-name">olmoearth-${s.slug}</h2>
        </div>
        <span class="detail-num">#${s.n}</span>
      </div>
      <section class="detail-sec">
        <h3 class="detail-h">What it does &amp; why</h3>
        <p class="detail-p">${SKILL_SPECS[s.n] || s.desc}</p>
      </section>
      ${
        SKILL_IO[s.n]
          ? `<section class="detail-sec">
        <h3 class="detail-h">Inputs and outputs</h3>
        <div class="detail-io">
          <div class="detail-io-row"><span class="detail-io-k">In</span><span class="detail-io-v">${SKILL_IO[s.n].in}</span></div>
          <div class="detail-io-row"><span class="detail-io-k">Out</span><span class="detail-io-v">${SKILL_IO[s.n].out}</span></div>
        </div>
      </section>`
          : ''
      }
      <section class="detail-sec">
        <h3 class="detail-h">Stage</h3>
        <p class="detail-p detail-stage">${s.cat} &middot; skill #${s.n} of 18 in the OlmoEarth workflow (Prep &rarr; Configure &rarr; Run &rarr; Analyze &rarr; Integrate &rarr; Report).</p>
      </section>
      <section class="detail-sec">
        <h3 class="detail-h">Tools it composes</h3>
        <p class="detail-tools">${SKILL_TOOLS[s.n] || '-'}</p>
      </section>
      <section class="detail-sec">
        <h3 class="detail-h">Example brief</h3>
        <p class="card-ex">${escapeHtml(s.ex)}</p>
        <button class="card-use" type="button" data-use>Use this brief
          <svg viewBox="0 0 24 24" class="ic"><path d="M5 12h14M13 6l6 6-6 6"/></svg>
        </button>
      </section>
      <a class="detail-link" href="https://github.com/2imi9/OlmoEarth-Agent/blob/main/SKILLS.md#${s.n}-olmoearth-${s.slug}" target="_blank" rel="noopener">Full academic spec on GitHub →</a>
    </div>`;
  d.hidden = false;
  const back = d.querySelector('.detail-back');
  if (back) back.focus();
}

function closeSkillDetail() {
  const d = document.getElementById('skillDetail');
  if (!d || d.hidden) return;
  d.hidden = true;
  d.innerHTML = '';
  if (_skillModalReturn) { _skillModalReturn.focus(); _skillModalReturn = null; }
}

export function wireCards() {
  const grid = document.getElementById('capGrid');
  const modal = document.getElementById('skillModal');
  const detail = document.getElementById('skillDetail');
  if (!grid || !modal) return;
  const fillBrief = (root) => {
    const ex = root.querySelector('.card-ex');
    const input = document.getElementById('promptInput');
    if (ex && input) { input.value = ex.textContent.trim(); autosize(input); }
    if (input) {
      input.focus();
      input.classList.add('flash');
      setTimeout(() => input.classList.remove('flash'), 700);
    }
  };
  const openFor = (card) => {
    const s = SKILLS.find((x) => x.n === parseInt(card.dataset.n, 10));
    if (!s) return;
    _skillModalReturn = card;
    openSkillModal(s);
  };
  grid.addEventListener('click', (e) => {
    const card = e.target.closest('.card');
    if (card) openFor(card);
  });
  grid.addEventListener('keydown', (e) => {
    const card = e.target.closest('.card');
    if (card && (e.key === 'Enter' || e.key === ' ')) { e.preventDefault(); openFor(card); }
  });
  modal.addEventListener('click', (e) => {
    if (e.target.closest('[data-detail]')) {
      const s = SKILLS.find((x) => x.n === parseInt(modal.dataset.n, 10));
      const card = _skillModalReturn;
      closeSkillModal();
      _skillModalReturn = card;  // keep the focus-return target through the detail page
      if (s) openSkillDetail(s);
      return;
    }
    if (e.target.closest('[data-use]')) { fillBrief(modal); closeSkillModal(); return; }
    if (e.target.closest('[data-close]')) closeSkillModal();
  });
  if (detail) {
    detail.addEventListener('click', (e) => {
      if (e.target.closest('[data-use]')) { fillBrief(detail); closeSkillDetail(); return; }
      if (e.target.closest('[data-detail-close]')) closeSkillDetail();
    });
  }
  document.addEventListener('keydown', (e) => {
    if (e.key !== 'Escape') return;
    if (detail && !detail.hidden) closeSkillDetail();
    else closeSkillModal();
  });
}
