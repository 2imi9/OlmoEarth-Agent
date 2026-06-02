/* The 17-skill capability grid + the pop-out introduction modal + the in-app
   full-spec detail page. Self-contained: clicking a card opens the modal;
   "Full spec" opens the detail page; "Use this brief" drops the example into
   the composer. */

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
};

const SKILLS = [
  { n: 1,  slug: 'studio-upload',     cat: 'Prep',      icon: 'upload',      desc: 'Labels → a Studio-importable file with MIME, 10K-record, and multi-metric guards.', ex: 'I have 3,000 field plots as a GeoJSON - get them into Studio without the Windows MIME error.' },
  { n: 2,  slug: 'rslearn-config',    cat: 'Prep',      icon: 'sliders',     desc: 'Labels → an rslearn dataset.json + Lightning YAML, with a 7-criteria audit.', ex: 'Turn my labeled crop polygons + HUC-12 watershed AOIs into an rslearn dataset.json + Lightning YAML.' },
  { n: 3,  slug: 'studio-job-config', cat: 'Configure', icon: 'wand',        desc: 'Task description → Studio job-wizard answers; 14 presets + a cross-field validator.', ex: 'I want per-pixel mangrove classification from Sentinel-2 - fill in the Studio job wizard.' },
  { n: 4,  slug: 'embeddings',        cat: 'Configure', icon: 'branch',      desc: 'Embeddings-vs-fine-tune decision, plus a runnable extraction notebook.', ex: 'I have 150 labels and a Colab T4 - should I fine-tune or use embeddings? Give me a notebook.' },
  { n: 5,  slug: 'predict',           cat: 'Run',       icon: 'satellite',   desc: 'The core run loop: find a model, submit, poll, and fetch result tiles.', pink: true, ex: 'Run a flood-extent prediction over this AOI for last month and return the result tiles.' },
  { n: 6,  slug: 'change-detect',     cat: 'Run',       icon: 'trend',       desc: 'A multi-date (≥3) trajectory diff; it refuses a naive 2-date diff.', pink: true, ex: 'Did forest cover decline across these four quarterly snapshots, or is it just noise?' },
  { n: 7,  slug: 'baseline-compare',  cat: 'Run',       icon: 'compare',     desc: 'OlmoEarth vs AlphaEarth, head-to-head on transfer regions.', pink: true, ex: 'Compare OlmoEarth vs AlphaEarth for land cover in a region where AlphaEarth struggles.' },
  { n: 8,  slug: 'evaluate',          cat: 'Analyze',   icon: 'barcheck',    desc: 'A random-vs-spatial CV inflation check, plus per-class metrics.', ex: 'My model reports 92% accuracy - re-check it with spatial cross-validation, not random splits.' },
  { n: 9,  slug: 'similarity',        cat: 'Analyze',   icon: 'search',      desc: 'Top-K embedding search with a geographic-prior warning.', ex: 'Find the 20 patches most similar to this illegal-mining site across the basin.' },
  { n: 10, slug: 'uncertainty',       cat: 'Analyze',   icon: 'shield',      desc: 'A Meyer-Pebesma Area-of-Applicability out-of-distribution flag.', ex: 'Flag which parts of my prediction AOI fall outside the training distribution.' },
  { n: 11, slug: 'cloud-mask-audit',  cat: 'Analyze',   icon: 'cloud',       desc: 'CFMask / s2cloudless / Sen2Cor / MAJA disagreement: bad mask vs bad model.', ex: 'My prediction looks wrong over this scene - bad cloud mask or bad model?' },
  { n: 12, slug: 'qgis-bridge',       cat: 'Integrate', icon: 'layers',      desc: 'Tile URLs → a QGIS XYZ layer + an OGC SLD style, ready to load.', ex: 'Give me a QGIS layer + SLD style for this prediction so I can open it on my desktop.' },
  { n: 13, slug: 'data-export',       cat: 'Integrate', icon: 'database',    desc: 'Export Studio projects + predictions to JSON, grouped by project or status.', ex: 'Export all my Studio projects and their predictions to JSON, grouped by status.' },
  { n: 14, slug: 'provenance',        cat: 'Report',    icon: 'fingerprint', desc: 'A manifest over every tool call, plus a one-command replay script.', ex: 'Produce a replay script + manifest so an auditor can reproduce this prediction.' },
  { n: 15, slug: 'case-narrative',    cat: 'Report',    icon: 'docspark',    desc: 'A stakeholder Markdown brief with a freshness gate on stale tiles.', ex: 'Write a stakeholder brief for this karst-vulnerability result with the live map tiles.' },
  { n: 16, slug: 'litsearch',         cat: 'Report',    icon: 'search',      desc: 'arXiv + OpenAlex search with DOI / arXiv-id resolution to ground citations.', ex: 'Find and cite the paper behind the Area-of-Applicability method I used.' },
  { n: 17, slug: 'automate',          cat: 'Configure', icon: 'wand',        desc: 'One call: auto-decides embeddings vs fine-tune and proposes a config; optional HF introspection.', ex: 'I have 200 labels and a T4 - should I fine-tune or use embeddings? Set it up.' },
];

// Fuller "what it does + why" per skill, shown in the pop-out. The card shows
// the one-line desc above; this is the spec summary. Full academic spec: SKILLS.md.
const SKILL_SPECS = {
  1: `Enforces the sample_category schema, shards at 10K records, works around the Windows .geojson MIME rejection, and splits multi-metric files. Onboarding friction is the most repeated dropoff for case providers, and Studio uploads fail silently without these guards.`,
  2: `Writes an rslearn dataset.json (single-layer 3-bandset or per-month production layout) plus a Lightning YAML, handles the es_label rename trap and watershed AOIs (NLDI / HUC-12), and runs a 7-criteria audit before a multi-hour training run fails silently.`,
  3: `Maps a task description to Studio wizard answers - output type, model size, time-frame mode, S2-vs-+S1 sources, patch size - from 14 verified presets, with a cross-field validator that catches traps (detection + 320m patch, embeddings + single-moment) before you submit.`,
  4: `The embeddings-vs-fine-tune decision grounded in the OlmoEarth accuracy / time / VRAM table, plus a parameterized notebook that extracts Nano/Tiny/Base/Large embeddings and trains kNN + linear-probe heads. This is the guidance version - you run the notebook. For one-call automation, see #17.`,
  5: `The core run primitive: discover a reusable model_id, submit a prediction, poll with backoff, then fetch results - XYZ raster tiles, MVT vectors, pixel values, or features by class. Async-by-reference, and cost-guarded on fine-tunes.`,
  6: `Turns a dated series of per-date layer summaries into trajectory metrics: step deltas, net change, the largest-change interval, a reversal count, and a trend label. Refuses fewer than 3 dates, because a 2-date diff cannot tell a steady trend from a flood that peaked then receded.`,
  7: `Runs OlmoEarth vs a baseline foundation model head-to-head on shared ground truth: a per-metric table (accuracy / macro-F1 / mean-IoU) with deltas and an overall winner, plus a cell-by-cell difference raster, to substantiate a transfer-region claim honestly.`,
  8: `Compares test-to-train nearest-neighbour distance under random vs spatial-block CV and reports the accuracy-inflation ratio and risk band (Ploton 2020 / Meyer-Pebesma 2021), plus per-class precision / recall / F1 / IoU. The guard against random splits overstating accuracy on clustered data.`,
  9: `Returns the top-K embedding vectors most similar to a query (cosine or Euclidean kNN), with a geographic-prior warning when the matches cluster near the query - because then "similarity" may reflect location (same biome) rather than genuine feature resemblance.`,
  10: `Implements the Meyer-Pebesma Area of Applicability: each point's dissimilarity index vs the training set, flagging out-of-distribution points beyond the training data's own outlier threshold. The point: softmax confidence is not OOD detection - a model can be confidently wrong off-distribution.`,
  11: `Summarizes where several aligned cloud masks (CFMask / s2cloudless / Sen2Cor / MAJA) agree vs disagree, then returns a bad-mask-vs-bad-model verdict for a model-error region. Surfaces disagreement rather than one ground-truth mask, because algorithms diverge on thin, semi-transparent cloud.`,
  12: `Resolves a prediction's relative tile template into an absolute QGIS XYZ URL and builds a well-formed OGC SLD color-ramp style, with Bearer-auth load instructions. Closes the hot-cloud-to-cold-desktop loop for the QGIS audience without embedding your key.`,
  13: `Exports your Studio projects and their predictions to JSON, grouped by project or status and curated to ids / names / statuses / times (no raw geometry). The self-contained alternative to wiring third-party data MCPs.`,
  14: `Records one manifest entry per tool call - tool name, sha256 of args, an id-only result summary, never raw geometry - and emits an auditable manifest plus a runnable replay skeleton. Serves EUDR / REDD+ MRV audit needs and the EO reproducibility gap.`,
  15: `Assembles a stakeholder Markdown report from prediction results and the run's provenance, with a freshness gate that withholds and strikes through tiles older than a configurable window - so a disaster-response brief never shows stale imagery.`,
  16: `Unified arXiv + OpenAlex search and DOI / arXiv-id resolution, deduped across sources and key-free (OpenAlex polite pool). Grounds EO citations in real papers instead of world-knowledge or hallucinated links, under a no-fabrication, cite-the-real-URL contract.`,
  17: `One call that auto-decides embeddings vs fine-tune (porting #4's decision table) and proposes a config - model size, classifier head, embeddings-notebook command, fine-tune schedule, and a Studio job-config hand-off - and can read a Hugging Face dataset's rows + classes to fill its inputs. Reports what is missing rather than guessing.`,
};

// Tools each skill composes, shown on the full-spec detail page. From PLAN.md section 1 + SKILLS.md.
const SKILL_TOOLS = {
  1: `olmoearth.upload_labels · validate_studio_mime · shard_at_10k · split_multi_metric`,
  2: `eo.window_tile · olmoearth.resolve_to_aoi · write_rslearn_config · audit_7_criteria · rename_es_label`,
  3: `Studio job-wizard presets (14) + a cross-field validator`,
  4: `olmoearth.fetch_embedding · decision_matrix · knn_head · linear_probe_head · system.python (light checks)`,
  5: `olmoearth_search_predictions · olmoearth_submit_prediction · olmoearth_get_prediction · olmoearth_fetch_results · olmoearth_pixel_value · olmoearth_features_search`,
  6: `olmoearth_change_detect (composes #5 olmoearth-predict) · enforce_min_3_dates · diff_layers`,
  7: `olmoearth_baseline_compare · compare_metrics (reuses #8 classification_metrics) · difference_raster`,
  8: `olmoearth_cv_inflation_check · olmoearth_classification_metrics · spatial_block_folds · haversine`,
  9: `olmoearth_similarity_search · similarity_search (brute-force kNN) · geographic_prior_check`,
  10: `olmoearth_area_of_applicability · dissimilarity index (Meyer-Pebesma) · ood_flag`,
  11: `olmoearth_cloud_mask_audit · ensemble_disagree · verdict_classifier`,
  12: `olmoearth_qgis_bridge · resolve_xyz_url · build_raster_sld`,
  13: `olmoearth_export_data (reads projects + predictions, curated)`,
  14: `olmoearth_provenance_summary · ProvenanceLog on ThreadState · replay_script`,
  15: `olmoearth_case_narrative · build_narrative (reads provenance + results)`,
  16: `olmoearth_litsearch · olmoearth_litsearch_resolve (arXiv + OpenAlex, deduped)`,
  17: `olmoearth_automate · analysis.automate decide() + propose_config + fetch_hf_dataset_profile (reuses #4's table)`,
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
      <div class="card-cat">${s.cat}</div>
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
      <div class="card-cat">${s.cat}</div>
      <h3 class="modal-name" id="skillModalName">olmoearth-${s.slug}</h3>
      <p class="modal-desc">${SKILL_SPECS[s.n] || s.desc}</p>
      <div class="modal-ex-label">Example brief</div>
      <p class="card-ex">${escapeHtml(s.ex)}</p>
      <div class="modal-actions">
        <button class="card-use" type="button" data-use>Use this brief
          <svg viewBox="0 0 24 24" class="ic"><path d="M5 12h14M13 6l6 6-6 6"/></svg>
        </button>
        <button class="modal-link" type="button" data-detail>Full spec →</button>
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
          <div class="card-cat">${s.cat}</div>
          <h2 class="detail-name">olmoearth-${s.slug}</h2>
        </div>
        <span class="detail-num">#${s.n}</span>
      </div>
      <section class="detail-sec">
        <h3 class="detail-h">What it does &amp; why</h3>
        <p class="detail-p">${SKILL_SPECS[s.n] || s.desc}</p>
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
