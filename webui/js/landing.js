/* Landing example briefs: each tab (Run a prediction / Analyze results / Prep &
   configure) suggests briefs that fit that stage of the workflow, so picking a
   mode also swaps the suggested examples. */

import { escapeHtml, autosize } from './util.js';

const EXAMPLE_BRIEFS = {
  run: [
    'How many OlmoEarth projects do I have, and which relate to water quality?',
    'Map karst aquifer vulnerability across central Pennsylvania.',
    'Detect new surface water in the Chesapeake over the last 12 months.',
  ],
  analyze: [
    'Compare my two prediction runs over the same area and show the difference map.',
    'Did karst-positive area trend up across these 4 quarterly snapshots?',
    'Compare OlmoEarth vs AlphaEarth on a transfer region.',
    'Flag predictions that fall outside my training data (area of applicability).',
  ],
  prep: [
    'I have 200 labels and a T4: embeddings or fine-tune?',
    'Audit my labels for the 8 data-prep pitfalls.',
    'Turn my GeoJSON labels into a Studio-ready dataset.',
  ],
};

function renderExamples(mode) {
  const box = document.getElementById('examples');
  if (!box) return;
  const briefs = EXAMPLE_BRIEFS[mode] || EXAMPLE_BRIEFS.run;
  box.innerHTML = briefs
    .map((b) => `<button class="ex" type="button">${escapeHtml(b)}</button>`)
    .join('');
}

function setExamplesShown(shown) {
  const box = document.getElementById('examples');
  const toggle = document.getElementById('exampleToggle');
  if (box) box.hidden = !shown;
  if (toggle) {
    toggle.setAttribute('aria-expanded', String(shown));
    const lbl = toggle.querySelector('.pill-lbl');
    if (lbl) lbl.textContent = shown ? 'Hide example briefs' : 'Show example briefs';
  }
}

export function wireTabs() {
  const input = document.getElementById('promptInput');
  document.querySelectorAll('.tab').forEach((tab) => {
    tab.addEventListener('click', () => {
      document.querySelectorAll('.tab').forEach((t) => t.classList.remove('is-active'));
      tab.classList.add('is-active');
      if (input && tab.dataset.ph) input.placeholder = tab.dataset.ph;
      renderExamples(tab.dataset.mode);
      setExamplesShown(true);  // surface the suggestions that fit this mode
    });
  });
}

export function wireExamples() {
  const toggle = document.getElementById('exampleToggle');
  const box = document.getElementById('examples');
  const input = document.getElementById('promptInput');
  const active = document.querySelector('.tab.is-active');
  renderExamples(active && active.dataset.mode ? active.dataset.mode : 'run');
  setExamplesShown(true);  // examples are visible by default; the toggle hides them
  if (toggle && box) {
    toggle.addEventListener('click', () => setExamplesShown(box.hidden));
    // Chips are re-rendered per mode, so delegate the click to the container.
    box.addEventListener('click', (e) => {
      const ex = e.target.closest('.ex');
      if (!ex || !input) return;
      input.value = ex.textContent.trim();
      autosize(input);
      input.focus();
    });
  }
}
