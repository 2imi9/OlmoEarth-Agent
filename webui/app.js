/* OlmoEarth Agent web UI — interactions + the 15-skill capability grid.
   Runs as a static mock by default; upgrades to the live agent when served
   by olmoearth_agent.serve (see the "Live bridge" section below). */

/* Live-mode flag, flipped on once /api/health answers. While false the page
   behaves exactly as the static demo. `renderKeyState` is wired by wireKey
   so detectBridge() can re-render the key card once the mode is known. */
const BRIDGE = { live: false, checked: false };
let renderKeyState = () => {};

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

/* ── Live run (demo) ─────────────────────────────────────────────────────
   The agent isn't wired to a backend yet, so a submitted brief plays a canned,
   streamed agent loop (thinking → tool call → result → answer). The content is
   real — it mirrors the olmoearth_change_detect / load_context showcase. */
const SCENARIOS = {
  change: {
    reasoning: "Four dates — that's ≥3, so a real trajectory diff is valid (a naive 2-date diff would hide the wobble). I'll call the change-detect skill.",
    tool: 'olmoearth_change_detect',
    args: '{\n  "series": [\n    {"date": "2024-03-01", "value": 0.12},\n    {"date": "2024-06-01", "value": 0.18},\n    {"date": "2024-09-01", "value": 0.15},\n    {"date": "2024-12-01", "value": 0.27}\n  ],\n  "metric": "karst_positive_fraction"\n}',
    result: [['trend', 'oscillating'], ['net_change', '+0.15'], ['reversals', '2'], ['max step', 'Sep→Dec +0.12']],
    answer: "Net positive area rose <strong>+0.15</strong> over the year, but it didn't trend cleanly upward — there were <strong>2 reversals</strong>, and the Sep→Dec quarter (<strong>+0.12</strong>) drove most of the gain. I'd call this <em>oscillating with an upward bias</em>, not a steady increase. Want the per-quarter chart or a stakeholder brief?",
  },
  projects: {
    reasoning: "I'll load the account context first (per the run rules — never invent ids), then count the projects it returns.",
    tool: 'olmoearth_load_context',
    args: '{}',
    result: [['name', 'demo-user'], ['project_count', '5'], ['ok', 'true']],
    answer: "You have <strong>5</strong> OlmoEarth Studio projects. Want me to list them, or filter to a topic (e.g. water quality)?",
  },
};

function pickScenario(brief) {
  return /project|how many|account|water quality/i.test(brief || '') ? SCENARIOS.projects : SCENARIOS.change;
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
}

function typeText(el, text, speed) {
  el.textContent = '';
  el.classList.add('caret');
  let i = 0;
  const tick = () => {
    if (i <= text.length) { el.textContent = text.slice(0, i); i += 1; setTimeout(tick, speed || 14); }
    else el.classList.remove('caret');
  };
  tick();
}

function runDemo(brief) {
  const rv = document.getElementById('runview');
  if (!rv) return;
  const sc = pickScenario(brief);
  const ex = document.getElementById('examples');
  if (ex) ex.hidden = true;
  const body = runShell(rv, brief);
  rv.scrollIntoView({ behavior: 'smooth', block: 'start' });

  setTimeout(() => {
    body.innerHTML = '<div class="think run-step" id="thinkRow"><span class="typing"><span></span><span></span><span></span></span><span id="thinkText" class="muted">Thinking…</span></div>';
  }, 450);

  setTimeout(() => {
    const row = document.getElementById('thinkRow');
    const t = document.getElementById('thinkText');
    if (row) { const dots = row.querySelector('.typing'); if (dots) dots.remove(); }
    if (t) { t.classList.remove('muted'); typeText(t, sc.reasoning); }
  }, 1700);

  setTimeout(() => {
    body.insertAdjacentHTML('beforeend',
      '<div class="toolcall run-step" id="tcCard">' +
        '<div class="tc-head"><span class="tc-dot running"></span><span id="tcState">calling</span> <code>' + sc.tool + '</code></div>' +
        '<pre class="tc-args">' + escapeHtml(sc.args) + '</pre>' +
        '<div class="tc-result" id="tcResult" hidden></div>' +
      '</div>');
    const c = document.getElementById('tcCard'); if (c) c.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }, 3100);

  setTimeout(() => {
    const dot = document.querySelector('#tcCard .tc-dot');
    const state = document.getElementById('tcState');
    const res = document.getElementById('tcResult');
    if (dot) dot.classList.remove('running');
    if (state) state.textContent = 'called';
    if (res) {
      res.hidden = false;
      res.classList.add('run-step');
      res.innerHTML = sc.result.map((kv) => '<span class="kv"><b>' + kv[0] + '</b> ' + kv[1] + '</span>').join('');
    }
  }, 4500);

  setTimeout(() => {
    body.insertAdjacentHTML('beforeend', '<div class="answer run-step" id="ansCard">' + sc.answer + '</div>');
    const a = document.getElementById('ansCard'); if (a) a.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }, 5500);
}

function wirePrompt() {
  const form = document.getElementById('promptForm');
  const input = document.getElementById('promptInput');
  if (input) input.addEventListener('input', () => autosize(input));
  if (form) {
    form.addEventListener('submit', (e) => {
      e.preventDefault();
      const brief = (input && input.value.trim()) || 'Did karst-positive area trend up across these 4 quarterly snapshots, or just wobble?';
      if (BRIDGE.live && projConnected()) runLive(brief);
      else runDemo(brief);
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
  const dot = document.getElementById('statusDot');
  const status = document.getElementById('userStatus');
  const maskKey = (k) => (k.length <= 6 ? k : k.slice(0, 6) + '••••' + k.slice(-2));
  const get = () => { try { return localStorage.getItem(LS) || ''; } catch (e) { return ''; } };
  const set = (v) => { try { v ? localStorage.setItem(LS, v) : localStorage.removeItem(LS); } catch (e) {} };
  function render() {
    const k = get(); const on = !!k;
    if (connect) connect.hidden = on;
    if (connected) connected.hidden = !on;
    if (dot) dot.classList.toggle('is-on', on);
    if (status) status.textContent = on ? (BRIDGE.live ? 'Connected · live' : 'Connected · demo') : 'Not connected';
    if (on) {
      const label = document.getElementById('keyStatusLabel');
      const body = document.getElementById('keyConnBody');
      if (label) label.textContent = BRIDGE.live ? 'Studio connected' : 'Connected · demo';
      if (body) body.innerHTML = BRIDGE.live
        ? 'Connected as <code>' + maskKey(k) + '</code> — briefs run the <strong>real agent</strong> against your Studio account through the local bridge.'
        : 'Key saved as <code>' + maskKey(k) + "</code>, but this preview <strong>doesn't call Studio yet</strong> — the data shown is sample, not your account.";
    }
    renderProjects();  // projects only load once a key is connected
  }
  renderKeyState = render;
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

/* ── Projects sidebar (New project + your projects, à la Vercel chatbot) ──
   Sample data for the mock; wired to olmoearth_search_projects when live. */
const PROJ_ICONS = {
  map:  '<path d="M9 3L3 6v15l6-3 6 3 6-3V3l-6 3-6-3z"/><path d="M9 3v15M15 6v15"/>',
  drop: '<path d="M12 3s6 6.5 6 11a6 6 0 01-12 0c0-4.5 6-11 6-11z"/>',
  trend:'<path d="M4 17l5-5 3 3 7-8"/><path d="M15 7h5v5"/>',
  leaf: '<path d="M5 19c0-7 5-12 14-12 0 9-5 14-12 12z"/><path d="M9 15c2-3 4-4 7-5"/>',
  sun:  '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/>',
};
const PROJECTS = [
  { id: 'karst',    name: 'PA Karst',                    meta: '12', icon: 'map'   },
  { id: 'ches',     name: 'Chesapeake — water quality',  meta: '5',  icon: 'drop'  },
  { id: 'potomac',  name: 'Potomac — change detection',  meta: '8',  icon: 'trend' },
  { id: 'mangrove', name: 'Mangrove extent — Indonesia', meta: '3',  icon: 'leaf'  },
  { id: 'solar',    name: 'Solar arrays — California',   meta: '2',  icon: 'sun'   },
];

function projConnected() {
  try { return !!localStorage.getItem('oe_studio_key'); } catch (e) { return false; }
}

function renderProjectList(list, projects) {
  list.innerHTML = projects.map((p, i) => `
    <button class="proj-item${i === 0 ? ' is-active' : ''}" data-id="${escapeHtml(p.id)}" title="${escapeHtml(p.name)}">
      <svg viewBox="0 0 24 24" class="ic">${PROJ_ICONS[p.icon] || PROJ_ICONS.map}</svg>
      <span class="proj-name">${escapeHtml(p.name)}</span>
      ${p.meta ? `<span class="proj-meta" title="${escapeHtml(p.meta)} predictions">${escapeHtml(p.meta)}</span>` : ''}
    </button>`).join('');
  list.querySelectorAll('.proj-item').forEach((el) => {
    el.addEventListener('click', () => {
      list.querySelectorAll('.proj-item').forEach((x) => x.classList.remove('is-active'));
      el.classList.add('is-active');
      const input = document.getElementById('promptInput');
      if (input) input.focus();
      const sb = document.getElementById('sidebar');
      if (sb) sb.classList.remove('open');
    });
  });
}

async function renderProjects() {
  const list = document.getElementById('projList');
  if (!list) return;
  if (!projConnected()) {
    list.innerHTML = '<div class="proj-empty">Connect your Studio key below to load your projects. <span class="proj-empty-sub">Nothing is fetched until you do.</span></div>';
    return;
  }
  if (!BRIDGE.live) {
    renderProjectList(list, PROJECTS);  // sample data (demo mode)
    return;
  }
  // Live: fetch the caller's real Studio projects through the bridge.
  list.innerHTML = '<div class="proj-empty">Loading your projects…</div>';
  try {
    const res = await fetch('/api/projects', { headers: { 'X-Olmoearth-Key': studioKey() } });
    if (!res.ok) throw new Error('HTTP ' + res.status);
    const data = await res.json();
    const projects = (data.projects || []).map((p) => ({
      id: p.id || '', name: p.name || '(unnamed)', icon: pickProjIcon(p.name), meta: '',
    }));
    if (!projects.length) {
      list.innerHTML = '<div class="proj-empty">No projects in your Studio account yet. <span class="proj-empty-sub">Create one in Studio, then refresh.</span></div>';
      return;
    }
    renderProjectList(list, projects);
  } catch (e) {
    list.innerHTML = '<div class="proj-empty">Couldn’t load projects — ' + escapeHtml(String((e && e.message) || e)) + '. <span class="proj-empty-sub">Check your key, or that the bridge can reach Studio.</span></div>';
  }
}

function wireNewProject() {
  const btn = document.getElementById('newProjectBtn');
  if (!btn) return;
  btn.addEventListener('click', () => {
    const rv = document.getElementById('runview');
    if (rv) { rv.hidden = true; rv.innerHTML = ''; }
    const ex = document.getElementById('examples');
    if (ex) ex.hidden = true;
    document.querySelectorAll('#projList .proj-item').forEach((x) => x.classList.remove('is-active'));
    const input = document.getElementById('promptInput');
    if (input) { input.value = ''; autosize(input); input.focus(); }
    const scroll = document.querySelector('.scroll');
    if (scroll) scroll.scrollTo({ top: 0, behavior: 'smooth' });
    const sb = document.getElementById('sidebar');
    if (sb) sb.classList.remove('open');
  });
}

/* ── Live bridge (optional) ───────────────────────────────────────────────
   When this page is served by olmoearth_agent.serve (FastAPI), /api/health
   answers and we switch from the canned demo to the real agent: real Studio
   projects, and briefs streamed through LeadAgent.run_stream over SSE. Opened
   as a static file (no bridge), detectBridge() leaves BRIDGE.live false and
   everything here stays dormant — the demo runs unchanged. */

async function detectBridge() {
  try {
    const res = await fetch('/api/health', { method: 'GET' });
    if (res.ok) {
      const data = await res.json();
      BRIDGE.live = !!(data && data.ok) && data.mode === 'live';
    }
  } catch (e) { BRIDGE.live = false; }
  BRIDGE.checked = true;
}

function studioKey() {
  try { return localStorage.getItem('oe_studio_key') || ''; } catch (e) { return ''; }
}

function pickProjIcon(name) {
  const n = (name || '').toLowerCase();
  if (/water|river|lake|chesa|quality|hydro|wetland/.test(n)) return 'drop';
  if (/change|trend|detect|delta|monitor/.test(n)) return 'trend';
  if (/forest|mangrove|veg|crop|tree|land|alfalfa/.test(n)) return 'leaf';
  if (/solar|energy|sun|panel/.test(n)) return 'sun';
  return 'map';
}

// Shared transcript scaffold for a run; returns the #runBody element.
function runShell(rv, brief) {
  rv.hidden = false;
  rv.innerHTML =
    '<div class="transcript live">' +
      '<div class="msg user run-step"><span class="role">Brief</span>' +
        '<div class="bubble">' + escapeHtml(brief) + '</div></div>' +
      '<div class="msg agent"><span class="role">OlmoEarth Agent</span><div id="runBody"></div></div>' +
    '</div>';
  return rv.querySelector('#runBody');
}

// Render one tool result envelope ({ok, result|error}) as kv chips + JSON.
function liveResultHtml(ev) {
  const r = ev.result || {};
  if (!ev.ok) {
    return '<span class="kv"><b>error</b> ' + escapeHtml(String(r.error || 'failed')) + '</span>';
  }
  const inner = r.result;
  const chips = ['<span class="kv"><b>ok</b> true</span>'];
  if (inner && typeof inner === 'object' && !Array.isArray(inner)) {
    for (const k of Object.keys(inner)) {
      const v = inner[k];
      if (v === null || typeof v !== 'object') {
        chips.push('<span class="kv"><b>' + escapeHtml(k) + '</b> ' + escapeHtml(String(v)) + '</span>');
      }
      if (chips.length >= 7) break;
    }
  }
  let json;
  try { json = JSON.stringify(inner === undefined ? r : inner, null, 2); } catch (e) { json = String(inner); }
  if (json && json.length > 700) json = json.slice(0, 700) + '\n…';
  return chips.join('') + '<pre class="tc-args tc-result-json">' + escapeHtml(json) + '</pre>';
}

// Apply one streamed event to the transcript body.
function handleRunEvent(body, ev) {
  if (ev.type === 'thinking') {
    const row = document.createElement('div');
    row.className = 'think run-step';
    const span = document.createElement('span');
    row.appendChild(span);
    body.appendChild(row);
    typeText(span, ev.text || '');
    row.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  } else if (ev.type === 'tool_call') {
    let args; try { args = JSON.stringify(ev.arguments || {}, null, 2); } catch (e) { args = '{}'; }
    body.insertAdjacentHTML('beforeend',
      '<div class="toolcall run-step">' +
        '<div class="tc-head"><span class="tc-dot running"></span><span class="tc-state">calling</span> <code>' + escapeHtml(ev.name || '') + '</code></div>' +
        '<pre class="tc-args">' + escapeHtml(args) + '</pre>' +
        '<div class="tc-result" hidden></div>' +
      '</div>');
    const cards = body.querySelectorAll('.toolcall');
    const card = cards[cards.length - 1];
    if (card) card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  } else if (ev.type === 'tool_result') {
    // run_stream emits call→result in order, so fill the last unfilled card.
    const cards = body.querySelectorAll('.toolcall');
    let card = null;
    for (let i = cards.length - 1; i >= 0; i--) {
      const r = cards[i].querySelector('.tc-result');
      if (r && r.hidden) { card = cards[i]; break; }
    }
    if (!card && cards.length) card = cards[cards.length - 1];
    if (card) {
      const dot = card.querySelector('.tc-dot'); if (dot) dot.classList.remove('running');
      const st = card.querySelector('.tc-state'); if (st) st.textContent = ev.ok ? 'called' : 'failed';
      const res = card.querySelector('.tc-result');
      if (res) { res.hidden = false; res.classList.add('run-step'); res.innerHTML = liveResultHtml(ev); }
    }
  } else if (ev.type === 'final') {
    body.insertAdjacentHTML('beforeend', '<div class="answer run-step">' + escapeHtml(ev.content || '(no answer)') + '</div>');
    const ans = body.querySelectorAll('.answer');
    const a = ans[ans.length - 1]; if (a) a.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  } else if (ev.type === 'max_turns') {
    body.insertAdjacentHTML('beforeend', '<div class="run-error run-step">Stopped at the turn cap (' + escapeHtml(String(ev.turns)) + ') without a final answer.</div>');
  } else if (ev.type === 'error') {
    body.insertAdjacentHTML('beforeend', '<div class="run-error run-step">⚠ ' + escapeHtml(ev.message || 'run failed') + '</div>');
  }
}

// Stream a real agent run over SSE and render it as it arrives.
async function runLive(brief) {
  const rv = document.getElementById('runview');
  if (!rv) return;
  const ex = document.getElementById('examples'); if (ex) ex.hidden = true;
  const body = runShell(rv, brief);
  rv.scrollIntoView({ behavior: 'smooth', block: 'start' });
  body.innerHTML = '<div class="think run-step" id="runStatus"><span class="typing"><span></span><span></span><span></span></span><span class="muted">Running the agent…</span></div>';
  try {
    const res = await fetch('/api/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-Olmoearth-Key': studioKey() },
      body: JSON.stringify({ brief }),
    });
    if (!res.ok || !res.body) throw new Error('bridge returned HTTP ' + res.status);
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = '';
    let cleared = false;
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let idx;
      while ((idx = buf.indexOf('\n\n')) >= 0) {
        const frame = buf.slice(0, idx); buf = buf.slice(idx + 2);
        const dataLine = frame.split('\n').find((l) => l.startsWith('data:'));
        if (!dataLine) continue;
        let ev; try { ev = JSON.parse(dataLine.slice(5).trim()); } catch (e) { continue; }
        if (ev.type === 'done') continue;
        if (!cleared) { const s = document.getElementById('runStatus'); if (s) s.remove(); cleared = true; }
        handleRunEvent(body, ev);
      }
    }
  } catch (e) {
    const s = document.getElementById('runStatus'); if (s) s.remove();
    handleRunEvent(body, { type: 'error', message: String((e && e.message) || e) });
  }
}

document.addEventListener('DOMContentLoaded', () => {
  renderCards();
  wireNewProject();
  wireTabs();
  wireExamples();
  wirePrompt();
  wireMenu();
  wireKey();  // initial render assumes demo mode
  // Upgrade to live mode if the bridge is serving this page, then re-render.
  detectBridge().then(renderKeyState);
});
