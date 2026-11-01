/* The 8-skill capability grid + the pop-out introduction modal + the in-app
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
  shield:      '<path d="M12 3l8 3v6c0 5-3.5 8-8 9-4.5-1-8-4-8-9V6z"/><path d="M9 12l2 2 4-4"/>',
  fingerprint: '<path d="M5 11a7 7 0 0114 0M8 12a4 4 0 018 0v2M12 13v5M8 15v3M16 15v2"/>',
  docspark:    '<path d="M7 3h7l5 5v13H7z"/><path d="M14 3v5h5"/><path d="M10.5 13l.7 1.8 1.8.7-1.8.7-.7 1.8-.7-1.8-1.8-.7 1.8-.7z"/>',
};

const SKILLS = [
  { n: 1,  slug: 'data-prep',         cat: 'Prep',      icon: 'upload',      desc: 'Labels → a Studio-importable file (MIME / 10K / multi-metric guards) AND an rslearn dataset.json + Lightning YAML, with a 7-criteria audit.', ex: 'I have 3,000 field plots as a GeoJSON - get them into Studio without the Windows MIME error, then build the rslearn dataset.json.' },
  { n: 2,  slug: 'studio-job-config', cat: 'Configure', icon: 'wand',        desc: 'Task description → Studio job-wizard answers; 14 presets + a cross-field validator.', ex: 'I want per-pixel mangrove classification from Sentinel-2 - fill in the Studio job wizard.' },
  { n: 3,  slug: 'embeddings',        cat: 'Configure', icon: 'branch',      desc: "Embeddings-vs-fine-tune: guidance + a runnable notebook; for an annual task, Ai2's precomputed v1.3 embeddings replace computing them yourself.", ex: 'I have 150 labels and a Colab T4 - should I fine-tune or use embeddings? Give me a notebook.' },
  { n: 4,  slug: 'predict',           cat: 'Run',       icon: 'satellite',   desc: "The core run loop: find a model, submit, poll, fetch result tiles, and compare results: a pair, a group, one model's dated results as a series (estimate movement, not verified ground change), or ensemble disagreement.", pink: true, ex: 'Run a flood-extent prediction over this AOI for last month and return the result tiles.' },
  { n: 5,  slug: 'provenance',        cat: 'Report',    icon: 'fingerprint', desc: 'A manifest over every tool call, plus a one-command replay script.', ex: 'Produce a replay script + manifest so an auditor can reproduce this prediction.' },
  { n: 6,  slug: 'case-narrative',    cat: 'Report',    icon: 'docspark',    desc: 'A stakeholder Markdown brief with a freshness gate on stale tiles.', ex: 'Write a stakeholder brief for this karst-vulnerability result with the live map tiles.' },
  { n: 7,  slug: 'rslearn',           cat: 'Configure', icon: 'sliders',     desc: 'Run-the-pipeline guidance for rslearn: the commands to stage a dataset and fit a model, for you to run.', ex: 'I train my own model with rslearn: give me the commands to stage this flooded-buildings dataset from Sentinel-2 and fit the model.' },
  { n: 8,  slug: 'review-set',        cat: 'Analyze',   icon: 'shield',      desc: "Which windows to check first, without labels: rank by the model's own top-1-minus-top-2 margin, at your review budget, boundary-first if you want it - plus the ceiling arithmetic; then how wrong the map is, from a labelled sample, and which of two maps of one time is more accurate, from labels where they differ.", ex: 'The map is done and I can only check 200 of these 4,000 windows. Which 200, and how much of the error will I actually catch?' },
];

/* Minimal {slug, desc, n} list for the composer's "/" skill slash-commands. */
export const SKILL_LIST = SKILLS.map((s) => ({ slug: s.slug, desc: s.desc, n: s.n }));

// Fuller "what it does + why" per skill, shown in the pop-out. The card shows
// the one-line desc above; this is the spec summary. Full academic spec: SKILLS.md.
const SKILL_SPECS = {
  1: `Labels → a Studio-importable file and/or an rslearn dataset.json + Lightning YAML. Enforces the sample_category / es_label / oe_labels schemas, shards at 10K records, works around the Windows .geojson MIME rejection, splits multi-metric files, handles the es_label rename and watershed AOIs (NLDI / HUC-12), and runs a 7-criteria audit before a multi-hour training run fails silently. The single vendored olmoearth-data-prep package.`,
  2: `Maps a task description to Studio wizard answers - output type, model size, time-frame mode, S2-vs-+S1 sources, patch size - from 14 verified presets, with a cross-field validator that catches traps (detection + 320m patch, embeddings + single-moment) before you submit.`,
  3: `The embeddings-vs-fine-tune decision grounded in the OlmoEarth accuracy / time / VRAM table: a parameterized notebook that extracts Nano/Tiny/Base/Large embeddings and trains kNN + linear-probe heads (you run it). Ai2 now publishes precomputed OlmoEarth v1.3 embeddings (source.coop/olmoearth/olmoearth-v1-3-embeddings): one vector per 10 m pixel per year, from a different model than the notebook's. They replace extracting embeddings yourself for an annual task in an area and year they cover; a task that needs dates within a year still needs the notebook.`,
  4: `The core run primitive: discover a reusable model_id, submit a prediction, poll with backoff, then fetch results - XYZ raster tiles, MVT vectors, or pixel values - and compare results with no ground truth. Async-by-reference, and cost-guarded on fine-tunes. AOIs can be drawn on a map in the chat (olmoearth_request_aoi) and stored as a Studio area. Change over time: olmoearth_compare_results in series mode traces one model's estimates across 3-8 dated results - chronological ordering discovered from each prediction, per-step pixel correlation, per-point trajectories, and shift sizes calibrated to the field's value range (estimate movement, not verified ground change). It refuses fewer than 3 dates, because a 2-date diff cannot tell a steady trend from a flood that peaked then receded. Ensemble disagreement across two or more distinct Studio results (mode ensemble): per-window spread as a confidence, the signal that works when Studio gives hard classes only (a class map without top-k scores).`,
  5: `Records one manifest entry per tool call - tool name, sha256 of args, an id-only result summary, never raw geometry - and emits an auditable manifest plus a runnable replay skeleton. Serves EUDR / REDD+ MRV audit needs and the EO reproducibility gap.`,
  6: `Assembles a stakeholder Markdown report from prediction results and the run's provenance, with a freshness gate that withholds and strikes through tiles older than a configurable window - so a disaster-response brief never shows stale imagery.`,
  7: `rslearn - the data + training engine under OlmoEarth. The vendored SKILL.md guides the 4-stage data pipeline (add_windows → prepare → ingest → materialize) + model fit/predict, as commands the user runs; the agent runs no training.`,
  8: `The question a practitioner asks the moment a prediction map lands: which windows do I open first, and how much of the error do I catch if I only open 5% of them? olmoearth_review_set ranks windows by the model's own top-1-minus-top-2 margin (low margin = suspect), optionally boundary-first using the nine-level indicator (how many of the 8 neighbours were predicted as a different class). olmoearth_review_budget_ceiling is the arithmetic guard - no rule can catch more than min(1, budget/error_rate), so a capture of 0.45 at a 10% budget with a 20% error rate is 90% of what was reachable, not '45%'. Measured, not asserted: on the tasks of Ai2's own published embedding suite the margin ranks errors better than a random order - its AUROC for errors is above 0.5 in all 3,560 cells of 16 encoders (OlmoEarth's four sizes among them) at ten probe seeds, and with OlmoEarth Base its 10% review set holds a median 21.4% of the errors against 10% for a random 10%. Its lead over the suite's two no-model controls (24 of 24 tasks) says little, since those controls are near chance there. Ensemble disagreement, cross-model disagreement and feature-space typicality were each measured against the margin and none ranked errors better. Bring-your-own scores - Studio returns hard classes only, which is exactly why skill #4's comparison tool reaches for ensemble disagreement instead, unless the prediction was submitted with probability_output: a class map with top-k scores carries its rank-1 and rank-2 softmax scores (uncalibrated), and olmoearth_review_set_from_result ranks its windows by their difference.`,
};

// Tools each skill composes, shown on the full-spec detail page. From PLAN.md section 1 + SKILLS.md.
const SKILL_TOOLS = {
  1: `olmoearth_load_skill · validate_studio_mime · shard_at_10k · split_multi_metric · write_rslearn_config · audit_7_criteria · rename_es_label`,
  2: `olmoearth_load_skill · Studio job-wizard presets (14) + a cross-field validator`,
  3: `olmoearth_load_skill (guidance + notebook) · decision_matrix · knn_head · linear_probe_head`,
  4: `olmoearth_search_predictions · olmoearth_submit_prediction (probability_top_k asks for a class map's top-k scores; left out with OLMOEARTH_READ_ONLY=1) · olmoearth_get_prediction · olmoearth_fetch_results · olmoearth_get_prediction_result · olmoearth_pixel_value · olmoearth_compare_results (a pair or a group of results; mode series: a date-ordered, legend-calibrated shift trace; mode ensemble: disagreement across results) · olmoearth_request_aoi · olmoearth_studio_label_inventory (the Studio annotations a model's training selector selects now, by split and class; no error rate)`,
  5: `olmoearth_provenance_summary · ProvenanceLog on ThreadState · replay_script`,
  6: `olmoearth_case_narrative · build_narrative (reads provenance + results)`,
  7: `olmoearth_load_skill (guidance for the 4-stage pipeline + fit/predict, which the user runs)`,
  8: `olmoearth_review_set (scores → the ordered review set at a budget) · olmoearth_review_set_from_result (a Studio result's [0, 1] score, or a class map's top-k scores → the same margin ranking) · olmoearth_scores_from_studio (a Studio result's native-resolution files → one scores file per UTM zone) · olmoearth_compare_review · olmoearth_review_budget_ceiling (min(1, budget/error_rate) and the share of it reached) · olmoearth_plan_label_sample → olmoearth_estimate_map_error / olmoearth_certify_zone (design-based error rate; optional inferencex extra) · olmoearth_plan_label_sample (design='disagreement') → olmoearth_estimate_map_error (which of two maps is more accurate: sample_disagreement, compare_from_disagreement, plan_which_map) · olmoearth_scores_from_file (a direct run's files: score_area.py's, or from-olmoearth's scores raster and olmoearth_output.json) · olmoearth_plan_label_sample (design='random', seed_from_map) → olmoearth_certify_zone → olmoearth_write_certificate / olmoearth_verify_certificate (a certificate anyone holding the map can check)`,
};

// What each skill takes in and gives back, for the full-spec detail page.
// Concise In -> Out, mirroring the "In:/Out:" lines in SKILLS.md.
const SKILL_IO = {
  1:  { in: 'Labels as GeoJSON / CSV / Shapefile (+ optional AOIs).', out: 'A Studio-importable file and/or an rslearn dataset.json + Lightning YAML, through MIME / 10K / multi-metric guards and a 7-criteria audit.' },
  2:  { in: 'A free-text task description.', out: 'Studio job-wizard answers (output type, model size, time frame, sources, patch size) + a cross-field validation.' },
  3:  { in: 'Sample count, class count, compute tier, goal.', out: 'An embeddings-vs-fine-tune recommendation + a runnable extraction notebook.' },
  4:  { in: 'A model_id, an AOI (draw it on the map), and a time range; to compare, 2-8 result ids (3-8 of one model over one area for a series, two or more of one quantity for ensemble disagreement).', out: 'A submitted prediction, then result tiles / pixel values (async by reference), and comparisons of results: agreement, a series\' per-step correlation, per-point trajectories and legend-calibrated shifts, or per-window ensemble spread as a confidence.' },
  5:  { in: "The run's tool-call log on ThreadState.", out: 'An append-only manifest (tool, sha256 of args, id-only summary) + a runnable replay script.' },
  6:  { in: 'Prediction results + the run provenance.', out: 'A stakeholder Markdown brief with live tiles and a freshness gate that strikes through stale imagery.' },
  7:  { in: 'A dataset and model to stage and train with rslearn.', out: 'The rslearn pipeline and fit/predict commands, for you to run.' },
  8:  { in: 'Per-window model scores (n_windows x n_classes logits or probabilities), a direct run\'s files (score_area.py\'s or from-olmoearth\'s) or a Studio result whose band is a [0, 1] score or a class map with top-k scores (on a grid, or its native-resolution files), a review budget, optionally a [rows, cols] grid; for estimation, a label budget, then the reviewer\'s 0/1 labels (for a Studio class map, the class seen in each window); for two maps of one time (class maps included), a label budget, then the class seen in each window where they differ.', out: 'The ordered review set with margins and boundary counts, the attainable-ceiling arithmetic, and the measured evidence + honest caveats; for estimation, the error rate with the interval valid at every sample size first, then the one its design earns (a review set is not a sample), and for a direct run\'s plan drawn with seed_from_map, a certificate anyone holding the map can check; for a Studio class map, the error rate and per-class accuracy the same way, and, without top-k scores, no review set or certified zone; for two maps, which is more accurate by the exact sign test, and the difference in accuracy with a 95% interval.' },
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
          <div class="card-cat">${s.cat}</div>
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
        <p class="detail-p detail-stage">${s.cat} &middot; skill #${s.n} of ${SKILLS.length} in the OlmoEarth workflow (Prep &rarr; Configure &rarr; Run &rarr; Analyze &rarr; Report).</p>
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
