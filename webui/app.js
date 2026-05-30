/* OlmoEarth Agent web UI — interactions + the 15-skill capability grid.
   Static mock; no dependencies. */

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
  { n: 1,  slug: 'studio-upload',     cat: 'Prep',      icon: 'upload',      desc: 'Labels → a Studio-importable file with MIME, 10K-record, and multi-metric guards.' },
  { n: 2,  slug: 'rslearn-config',    cat: 'Prep',      icon: 'sliders',     desc: 'Labels → an rslearn dataset.json + Lightning YAML, with a 7-criteria audit.' },
  { n: 3,  slug: 'studio-job-config', cat: 'Configure', icon: 'wand',        desc: 'Task description → Studio job-wizard answers; 14 presets + a cross-field validator.' },
  { n: 4,  slug: 'embeddings',        cat: 'Configure', icon: 'branch',      desc: 'Embeddings-vs-fine-tune decision, plus a runnable extraction notebook.' },
  { n: 5,  slug: 'predict',           cat: 'Run',       icon: 'satellite',   desc: 'The core run loop: find a model, submit, poll, and fetch result tiles.', pink: true },
  { n: 6,  slug: 'change-detect',     cat: 'Run',       icon: 'trend',       desc: 'A multi-date (≥3) trajectory diff — it refuses a naive 2-date diff.', pink: true },
  { n: 7,  slug: 'baseline-compare',  cat: 'Run',       icon: 'compare',     desc: 'OlmoEarth vs AlphaEarth, head-to-head on transfer regions.', pink: true },
  { n: 8,  slug: 'evaluate',          cat: 'Analyze',   icon: 'barcheck',    desc: 'A random-vs-spatial CV inflation check, plus per-class metrics.' },
  { n: 9,  slug: 'similarity',        cat: 'Analyze',   icon: 'search',      desc: 'Top-K embedding search with a geographic-prior warning.' },
  { n: 10, slug: 'uncertainty',       cat: 'Analyze',   icon: 'shield',      desc: 'A Meyer–Pebesma Area-of-Applicability out-of-distribution flag.' },
  { n: 11, slug: 'cloud-mask-audit',  cat: 'Analyze',   icon: 'cloud',       desc: 'CFMask / s2cloudless / Sen2Cor / MAJA disagreement: bad mask vs bad model.' },
  { n: 12, slug: 'qgis-bridge',       cat: 'Integrate', icon: 'layers',      desc: 'Tile URLs → a QGIS XYZ layer + an OGC SLD style, ready to load.' },
  { n: 13, slug: 'data-export',       cat: 'Integrate', icon: 'database',    desc: 'Export Studio projects + predictions to JSON, grouped by project or status.' },
  { n: 14, slug: 'provenance',        cat: 'Report',    icon: 'fingerprint', desc: 'A manifest over every tool call, plus a one-command replay script.' },
  { n: 15, slug: 'case-narrative',    cat: 'Report',    icon: 'docspark',    desc: 'A stakeholder Markdown brief with a freshness gate on stale tiles.' },
];

function renderCards() {
  const grid = document.getElementById('capGrid');
  if (!grid) return;
  grid.innerHTML = SKILLS.map((s) => `
    <article class="card${s.pink ? ' is-pink' : ''}">
      <div class="card-top">
        <span class="card-ic"><svg viewBox="0 0 24 24" class="ic">${ICONS[s.icon] || ''}</svg></span>
        <span class="card-num">#${s.n}</span>
      </div>
      <div class="card-cat">${s.cat}</div>
      <div class="card-name">olmoearth-${s.slug}</div>
      <p class="card-desc">${s.desc}</p>
    </article>`).join('');
}

function wireTabs() {
  const input = document.getElementById('promptInput');
  document.querySelectorAll('.tab').forEach((tab) => {
    tab.addEventListener('click', () => {
      document.querySelectorAll('.tab').forEach((t) => t.classList.remove('is-active'));
      tab.classList.add('is-active');
      if (input && tab.dataset.ph) input.placeholder = tab.dataset.ph;
    });
  });
}

function wireExamples() {
  const toggle = document.getElementById('exampleToggle');
  const box = document.getElementById('examples');
  const input = document.getElementById('promptInput');
  if (toggle && box) {
    toggle.addEventListener('click', () => { box.hidden = !box.hidden; });
  }
  document.querySelectorAll('.ex').forEach((ex) => {
    ex.addEventListener('click', () => {
      if (!input) return;
      input.value = ex.textContent.trim();
      autosize(input);
      input.focus();
      input.scrollIntoView({ behavior: 'smooth', block: 'center' });
    });
  });
}

function autosize(el) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 200) + 'px';
}

function wirePrompt() {
  const form = document.getElementById('promptForm');
  const input = document.getElementById('promptInput');
  if (input) input.addEventListener('input', () => autosize(input));
  if (form) {
    form.addEventListener('submit', (e) => {
      e.preventDefault();
      const run = document.getElementById('run-example');
      if (run) run.scrollIntoView({ behavior: 'smooth' });
    });
  }
}

function wireMenu() {
  const btn = document.getElementById('menuBtn');
  const sidebar = document.getElementById('sidebar');
  if (btn && sidebar) btn.addEventListener('click', () => sidebar.classList.toggle('open'));
}

/* Bring-your-own-key: a user with a Studio assignment already has an OlmoEarth
   Studio API key, so they paste their own. Stored client-side only (localStorage).
   Real email-based login is future work — handled elsewhere, not here. */
function wireKey() {
  const LS = 'oe_studio_key';
  const connect = document.getElementById('keyConnect');
  const connected = document.getElementById('keyConnected');
  const form = document.getElementById('keyForm');
  const input = document.getElementById('keyInput');
  const mask = document.getElementById('keyMask');
  const dot = document.getElementById('statusDot');
  const status = document.getElementById('userStatus');
  const maskKey = (k) => (k.length <= 6 ? k : k.slice(0, 6) + '••••' + k.slice(-2));
  const get = () => { try { return localStorage.getItem(LS) || ''; } catch (e) { return ''; } };
  const set = (v) => { try { v ? localStorage.setItem(LS, v) : localStorage.removeItem(LS); } catch (e) {} };
  function render() {
    const k = get(); const on = !!k;
    if (connect) connect.hidden = on;
    if (connected) connected.hidden = !on;
    if (on && mask) mask.textContent = maskKey(k);
    if (dot) dot.classList.toggle('is-on', on);
    if (status) status.textContent = on ? 'Studio connected' : 'Not connected';
  }
  if (form) form.addEventListener('submit', (e) => {
    e.preventDefault();
    const v = (input.value || '').trim();
    if (!v) return;
    set(v); input.value = ''; render();
  });
  const dis = document.getElementById('keyDisconnect');
  if (dis) dis.addEventListener('click', () => { set(''); render(); });
  const top = document.getElementById('topKeyBtn');
  if (top) top.addEventListener('click', () => {
    document.getElementById('sidebar') && document.getElementById('sidebar').classList.add('open');
    if (input) { input.focus(); input.scrollIntoView({ behavior: 'smooth', block: 'center' }); }
  });
  render();
}

document.addEventListener('DOMContentLoaded', () => {
  renderCards();
  wireTabs();
  wireExamples();
  wirePrompt();
  wireMenu();
  wireKey();
});
