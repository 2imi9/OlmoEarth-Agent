/* OlmoEarth Agent web UI: chat + history, the 15-skill grid, and a Studio
   project tree. Static demo by default; upgrades to the live agent when served
   by olmoearth_agent.serve (see "Live bridge" below). No build step. */

/* Live-mode flag, flipped on once /api/health answers. While false the page
   behaves as the static demo. `renderKeyState` is wired by wireKey so
   detectBridge() can re-render the key card once the mode is known. */
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
  { n: 6,  slug: 'change-detect',     cat: 'Run',       icon: 'trend',       desc: 'A multi-date (≥3) trajectory diff; it refuses a naive 2-date diff.', pink: true },
  { n: 7,  slug: 'baseline-compare',  cat: 'Run',       icon: 'compare',     desc: 'OlmoEarth vs AlphaEarth, head-to-head on transfer regions.', pink: true },
  { n: 8,  slug: 'evaluate',          cat: 'Analyze',   icon: 'barcheck',    desc: 'A random-vs-spatial CV inflation check, plus per-class metrics.' },
  { n: 9,  slug: 'similarity',        cat: 'Analyze',   icon: 'search',      desc: 'Top-K embedding search with a geographic-prior warning.' },
  { n: 10, slug: 'uncertainty',       cat: 'Analyze',   icon: 'shield',      desc: 'A Meyer-Pebesma Area-of-Applicability out-of-distribution flag.' },
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
    });
  });
}

function autosize(el) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 200) + 'px';
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

/* ── Demo scenarios ──────────────────────────────────────────────────────
   In demo mode a brief plays a canned agent loop. The same event shape as the
   live SSE stream, so one renderer (handleRunEvent) drives both. */
const SCENARIOS = {
  change: {
    reasoning: "Four dates, that's ≥3, so a real trajectory diff is valid (a naive 2-date diff would hide the wobble). I'll call the change-detect skill.",
    tool: 'olmoearth_change_detect',
    args: '{\n  "series": [\n    {"date": "2024-03-01", "value": 0.12},\n    {"date": "2024-06-01", "value": 0.18},\n    {"date": "2024-09-01", "value": 0.15},\n    {"date": "2024-12-01", "value": 0.27}\n  ],\n  "metric": "karst_positive_fraction"\n}',
    result: [['trend', 'oscillating'], ['net_change', '+0.15'], ['reversals', '2'], ['max_step', 'Sep→Dec +0.12']],
    answer: "Net positive area rose +0.15 over the year, but it didn't trend cleanly upward. There were 2 reversals, and the Sep→Dec quarter (+0.12) drove most of the gain. I'd call this oscillating with an upward bias, not a steady increase. Want the per-quarter chart or a stakeholder brief?",
  },
  projects: {
    reasoning: "I'll load the account context first (per the run rules, never invent ids), then count the projects it returns.",
    tool: 'olmoearth_load_context',
    args: '{}',
    result: [['name', 'demo-user'], ['project_count', '5'], ['ok', 'true']],
    answer: "You have **5** OlmoEarth Studio projects. Here's how they relate to **water quality**:\n\n| Project | Relevance |\n|---|---|\n| **Chesapeake - water quality** | ✓ Strong: nutrient loading |\n| **Potomac - change detection** | ✓ Strong: sewage spill event |\n| **PA Karst** | ~ Moderate: karst aquifer vulnerability |\n| **Mangrove - Indonesia** | ✗ Low: coastal extent |\n| **Solar arrays - California** | ✗ None: energy infrastructure |\n\nWant the per-project predictions, or a stakeholder brief?",
  },
};

function pickScenario(brief) {
  return /project|how many|account|water quality/i.test(brief || '') ? SCENARIOS.projects : SCENARIOS.change;
}

function demoEvents(brief) {
  const sc = pickScenario(brief);
  let args; try { args = JSON.parse(sc.args || '{}'); } catch (e) { args = {}; }
  const resultObj = {};
  sc.result.forEach(([k, v]) => { resultObj[k] = v; });
  return [
    { type: 'thinking', text: sc.reasoning },
    { type: 'tool_call', name: sc.tool, arguments: args },
    { type: 'tool_result', name: sc.tool, ok: true, result: { ok: true, result: resultObj } },
    { type: 'final', content: sc.answer },
  ];
}

/* ── Run rendering (shared by demo + live) ──────────────────────────────── */

// Render one tool-result envelope ({ok, result|error}) as kv chips + JSON.
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

/* ── Minimal, safe Markdown → HTML for agent answers ─────────────────────
   The model replies in GitHub-flavored Markdown; render it (tables, bold,
   lists, code) instead of showing raw ** and | . Text is HTML-escaped FIRST,
   then tokenized, so model output can't inject markup. */
function mdInline(text) {
  let s = escapeHtml(text);
  s = s.replace(/`([^`]+)`/g, '<code>$1</code>');
  s = s.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
  s = s.replace(/(^|[^*\w])\*(\S(?:[^*\n]*\S)?)\*(?![*\w])/g, '$1<em>$2</em>');
  s = s.replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
  return s;
}
function mdSplitRow(line) {
  let s = line.trim();
  if (s.charAt(0) === '|') s = s.slice(1);
  if (s.charAt(s.length - 1) === '|') s = s.slice(0, -1);
  return s.split('|').map((c) => c.trim());
}
function mdIsTableStart(lines, i) {
  return i + 1 < lines.length && lines[i].indexOf('|') >= 0 &&
    lines[i + 1].indexOf('-') >= 0 && /^\s*\|?[\s:|-]+\|?\s*$/.test(lines[i + 1]);
}
function renderMarkdown(src) {
  const lines = String(src).replace(/\r\n?/g, '\n').split('\n');
  const out = [];
  let i = 0;
  const isBlock = (l, j) => /^```/.test(l) || /^#{1,4}\s+/.test(l) ||
    /^\s*[-*+]\s+/.test(l) || /^\s*\d+\.\s+/.test(l) || mdIsTableStart(lines, j);
  while (i < lines.length) {
    const line = lines[i];
    if (/^```/.test(line)) {
      const buf = []; i++;
      while (i < lines.length && !/^```/.test(lines[i])) { buf.push(lines[i]); i++; }
      i++;
      out.push('<pre class="md-pre"><code>' + escapeHtml(buf.join('\n')) + '</code></pre>');
      continue;
    }
    if (mdIsTableStart(lines, i)) {
      const header = mdSplitRow(line); i += 2;
      const rows = [];
      while (i < lines.length && lines[i].indexOf('|') >= 0 && lines[i].trim() !== '') { rows.push(mdSplitRow(lines[i])); i++; }
      const th = header.map((h) => '<th>' + mdInline(h) + '</th>').join('');
      const trs = rows.map((r) => '<tr>' + header.map((_, j) => '<td>' + mdInline(r[j] || '') + '</td>').join('') + '</tr>').join('');
      out.push('<div class="md-table-wrap"><table class="md-table"><thead><tr>' + th + '</tr></thead><tbody>' + trs + '</tbody></table></div>');
      continue;
    }
    const h = /^(#{1,4})\s+(.*)$/.exec(line);
    if (h) { out.push('<div class="md-h md-h' + h[1].length + '">' + mdInline(h[2]) + '</div>'); i++; continue; }
    if (/^\s*[-*+]\s+/.test(line)) {
      const items = [];
      while (i < lines.length && /^\s*[-*+]\s+/.test(lines[i])) { items.push('<li>' + mdInline(lines[i].replace(/^\s*[-*+]\s+/, '')) + '</li>'); i++; }
      out.push('<ul class="md-ul">' + items.join('') + '</ul>');
      continue;
    }
    if (/^\s*\d+\.\s+/.test(line)) {
      const items = [];
      while (i < lines.length && /^\s*\d+\.\s+/.test(lines[i])) { items.push('<li>' + mdInline(lines[i].replace(/^\s*\d+\.\s+/, '')) + '</li>'); i++; }
      out.push('<ol class="md-ol">' + items.join('') + '</ol>');
      continue;
    }
    if (line.trim() === '') { i++; continue; }
    const para = [];
    while (i < lines.length && lines[i].trim() !== '' && !isBlock(lines[i], i)) { para.push(lines[i]); i++; }
    out.push('<p class="md-p">' + mdInline(para.join('\n')).replace(/\n/g, '<br>') + '</p>');
  }
  return out.join('') || '<p class="md-p"></p>';
}

// Per-turn "Reasoning & tools" disclosure: thinking + tool calls live in a
// collapsible block, collapsed by default (a finished turn shows just the
// answer). Live runs auto-expand it while streaming, then collapse on `final`.
function ensureSteps(body) {
  let steps = body.querySelector(':scope > .steps');
  if (!steps) {
    steps = document.createElement('div');
    steps.className = 'steps collapsed run-step';
    steps.innerHTML =
      '<button class="steps-head" type="button">' +
        '<svg viewBox="0 0 24 24" class="caret"><path d="M9 6l6 6-6 6"/></svg>' +
        '<span class="steps-label">Reasoning &amp; tools</span>' +
        '<span class="steps-count"></span>' +
      '</button>' +
      '<div class="steps-body"></div>';
    body.insertBefore(steps, body.firstChild);
    steps.querySelector('.steps-head').addEventListener('click', () => steps.classList.toggle('collapsed'));
  }
  return steps;
}
function stepsBody(body) { return ensureSteps(body).querySelector('.steps-body'); }
function stepsExpand(body) { const s = body.querySelector(':scope > .steps'); if (s) s.classList.remove('collapsed'); }
function bumpStepsCount(body) {
  const steps = body.querySelector(':scope > .steps');
  if (!steps) return;
  const sb = steps.querySelector('.steps-body');
  const tools = sb.querySelectorAll('.toolcall').length;
  const thinks = sb.querySelectorAll('.think').length;
  const parts = [];
  if (thinks) parts.push('reasoning');
  if (tools) parts.push(tools + ' tool call' + (tools > 1 ? 's' : ''));
  steps.querySelector('.steps-count').textContent = parts.length ? '· ' + parts.join(' · ') : '';
}

// Apply one streamed event to an agent-body element. `staticRender` skips the
// typing animation + auto-scroll (used when replaying a saved chat).
function handleRunEvent(body, ev, staticRender) {
  if (ev.type === 'thinking') {
    const sb = stepsBody(body);
    const row = document.createElement('div');
    row.className = 'think run-step';
    const span = document.createElement('span');
    row.appendChild(span);
    sb.appendChild(row);
    if (staticRender) span.textContent = ev.text || '';
    else { stepsExpand(body); typeText(span, ev.text || ''); row.scrollIntoView({ behavior: 'smooth', block: 'nearest' }); }
    bumpStepsCount(body);
  } else if (ev.type === 'tool_call') {
    const sb = stepsBody(body);
    let args; try { args = JSON.stringify(ev.arguments || {}, null, 2); } catch (e) { args = '{}'; }
    sb.insertAdjacentHTML('beforeend',
      '<div class="toolcall run-step">' +
        '<div class="tc-head"><span class="tc-dot' + (staticRender ? '' : ' running') + '"></span><span class="tc-state">' + (staticRender ? 'called' : 'calling') + '</span> <code>' + escapeHtml(ev.name || '') + '</code></div>' +
        '<pre class="tc-args">' + escapeHtml(args) + '</pre>' +
        '<div class="tc-result" hidden></div>' +
      '</div>');
    bumpStepsCount(body);
    if (!staticRender) {
      stepsExpand(body);
      const cards = body.querySelectorAll('.toolcall');
      const card = cards[cards.length - 1];
      if (card) card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
  } else if (ev.type === 'tool_result') {
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
    if (!staticRender) { const steps = body.querySelector(':scope > .steps'); if (steps) steps.classList.add('collapsed'); }
    body.insertAdjacentHTML('beforeend', '<div class="answer run-step md">' + renderMarkdown(ev.content || '(no answer)') + '</div>');
    if (!staticRender) {
      const ans = body.querySelectorAll('.answer');
      const a = ans[ans.length - 1]; if (a) a.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
  } else if (ev.type === 'max_turns') {
    body.insertAdjacentHTML('beforeend', '<div class="run-error run-step">Stopped at the turn cap (' + escapeHtml(String(ev.turns)) + ') without a final answer.</div>');
  } else if (ev.type === 'error') {
    body.insertAdjacentHTML('beforeend', '<div class="run-error run-step">⚠ ' + escapeHtml(ev.message || 'run failed') + '</div>');
  }
}

function runStatusEl(label) {
  const status = document.createElement('div');
  status.className = 'think run-step';
  status.innerHTML = '<span class="typing"><span></span><span></span><span></span></span><span class="muted">' + label + '</span>';
  return status;
}

// Play the canned demo loop into `body`, emitting events to `onEvent`.
function runDemo(body, brief, onEvent) {
  const events = demoEvents(brief);
  body.innerHTML = '';
  const status = runStatusEl('Thinking…');
  body.appendChild(status);
  let cleared = false;
  const delay = { thinking: 700, tool_call: 1900, tool_result: 3300, final: 4500 };
  events.forEach((ev) => {
    setTimeout(() => {
      if (!cleared) { status.remove(); cleared = true; }
      if (onEvent) onEvent(ev);
      handleRunEvent(body, ev);
    }, delay[ev.type] || 800);
  });
}

// Stream a real agent run over SSE into `body`, emitting events to `onEvent`.
async function runLive(body, brief, history, onEvent) {
  body.innerHTML = '';
  const status = runStatusEl('Running the agent…');
  body.appendChild(status);
  let cleared = false;
  const clear = () => { if (!cleared) { status.remove(); cleared = true; } };
  try {
    const res = await fetch('/api/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-Olmoearth-Key': studioKey() },
      body: JSON.stringify({ brief, history: history || [], max_turns: agentMaxTurns() }),
    });
    if (!res.ok || !res.body) throw new Error('bridge returned HTTP ' + res.status);
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = '';
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
        clear();
        if (onEvent) onEvent(ev);
        handleRunEvent(body, ev);
      }
    }
  } catch (e) {
    clear();
    const ev = { type: 'error', message: String((e && e.message) || e) };
    if (onEvent) onEvent(ev);
    handleRunEvent(body, ev);
  }
}

/* ── Chat store + threads (localStorage, à la Claude history) ────────────── */
const CHATS_LS = 'oe_chats';
let activeChat = null;     // { id, title, createdAt, updatedAt, turns: [...] }
let activeChatId = null;

function loadChats() { try { return JSON.parse(localStorage.getItem(CHATS_LS) || '[]'); } catch (e) { return []; } }
function saveChats(chats) { try { localStorage.setItem(CHATS_LS, JSON.stringify(chats)); } catch (e) {} }
function genId() { return 'c' + Date.now().toString(36) + Math.random().toString(36).slice(2, 7); }

function persistChat(chat) {
  if (!chat) return;
  const chats = loadChats().filter((c) => c.id !== chat.id);
  chats.push(chat);
  saveChats(chats);
}

function ensureActiveChat() {
  if (!activeChat) {
    activeChat = { id: genId(), title: '', createdAt: Date.now(), updatedAt: Date.now(), turns: [] };
    activeChatId = activeChat.id;
  }
}

function finalOf(events) {
  const fin = (events || []).filter((e) => e.type === 'final').pop();
  return fin ? (fin.content || '') : '';
}

// Prior turns → [{role, content}] for the agent's multi-turn memory.
function historyForAgent(turns) {
  const out = [];
  for (const t of turns) {
    if (t.role === 'user') out.push({ role: 'user', content: t.text });
    else if (t.role === 'assistant') { const f = finalOf(t.events); if (f) out.push({ role: 'assistant', content: f }); }
  }
  return out;
}

function showEmpty() {
  const e = document.getElementById('chatEmpty'); const t = document.getElementById('chatThread');
  if (e) e.hidden = false;
  if (t) t.hidden = true;
}
function showThread() {
  const e = document.getElementById('chatEmpty'); const t = document.getElementById('chatThread');
  if (e) e.hidden = true;
  if (t) t.hidden = false;
}
function setTopTitle(title) { const el = document.getElementById('topbarTitle'); if (el) el.textContent = title || ''; }

// Create the DOM for one turn (user bubble + empty agent body); return the body.
function appendTurnDom(userText) {
  const thread = document.getElementById('chatThread');
  const card = document.createElement('div');
  card.className = 'transcript live';
  card.innerHTML =
    '<div class="msg user run-step"><span class="role">You</span>' +
      '<div class="bubble">' + escapeHtml(userText) + '</div></div>' +
    '<div class="msg agent"><span class="role">OlmoEarth Agent</span><div class="agent-body"></div></div>';
  thread.appendChild(card);
  card.querySelector('.msg.user').scrollIntoView({ behavior: 'smooth', block: 'start' });
  return card.querySelector('.agent-body');
}

function replayEvents(body, events) {
  body.innerHTML = '';
  (events || []).forEach((ev) => handleRunEvent(body, ev, true));
}

function renderThread() {
  const thread = document.getElementById('chatThread');
  if (!thread) return;
  thread.innerHTML = '';
  if (!activeChat || !activeChat.turns.length) { showEmpty(); return; }
  showThread();
  for (let i = 0; i < activeChat.turns.length; i++) {
    const t = activeChat.turns[i];
    if (t.role !== 'user') continue;
    const body = appendTurnDom(t.text);
    const a = activeChat.turns[i + 1];
    if (a && a.role === 'assistant') { replayEvents(body, a.events); i++; }
  }
  const sc = document.getElementById('scroll'); if (sc) sc.scrollTop = sc.scrollHeight;
}

function newChat() {
  activeChat = { id: genId(), title: '', createdAt: Date.now(), updatedAt: Date.now(), turns: [] };
  activeChatId = activeChat.id;
  const thread = document.getElementById('chatThread'); if (thread) thread.innerHTML = '';
  showEmpty();
  setTopTitle('');
  renderChatList();
  const input = document.getElementById('promptInput'); if (input) { input.value = ''; autosize(input); input.focus(); }
  const sb = document.getElementById('sidebar'); if (sb) sb.classList.remove('open');
  const sc = document.getElementById('scroll'); if (sc) sc.scrollTo({ top: 0, behavior: 'smooth' });
}

function openChat(id) {
  const c = loadChats().find((x) => x.id === id);
  if (!c) return;
  activeChat = c; activeChatId = c.id;
  renderThread();
  setTopTitle(c.title);
  renderChatList();
  const sb = document.getElementById('sidebar'); if (sb) sb.classList.remove('open');
}

function deleteChat(id) {
  saveChats(loadChats().filter((c) => c.id !== id));
  if (activeChatId === id) { activeChat = null; activeChatId = null; newChat(); }
  else renderChatList();
}

function renderChatList() {
  const list = document.getElementById('chatList');
  if (!list) return;
  const chats = loadChats().sort((a, b) => (b.updatedAt || 0) - (a.updatedAt || 0));
  if (!chats.length) {
    list.innerHTML = '<div class="side-note">No saved chats yet. Send a brief below to start one.</div>';
    return;
  }
  list.innerHTML = chats.map((c) => `
    <button class="chat-item${c.id === activeChatId ? ' is-active' : ''}" data-id="${escapeHtml(c.id)}" title="${escapeHtml(c.title || 'Untitled chat')}">
      <svg viewBox="0 0 24 24" class="ic"><path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z"/></svg>
      <span class="chat-title">${escapeHtml(c.title || 'Untitled chat')}</span>
      <span class="chat-del" data-del="${escapeHtml(c.id)}" title="Delete chat" role="button">×</span>
    </button>`).join('');
  list.querySelectorAll('.chat-item').forEach((el) => {
    el.addEventListener('click', (e) => {
      if (e.target.closest('.chat-del')) return;
      openChat(el.dataset.id);
    });
  });
  list.querySelectorAll('.chat-del').forEach((el) => {
    el.addEventListener('click', (e) => { e.stopPropagation(); deleteChat(el.dataset.del); });
  });
}

// The composer's send path: append the turn, run it, persist + update history.
function handleSend(brief) {
  ensureActiveChat();
  const chat = activeChat;
  const history = historyForAgent(chat.turns);  // prior turns only
  chat.turns.push({ role: 'user', text: brief });
  const aTurn = { role: 'assistant', events: [] };
  chat.turns.push(aTurn);
  if (!chat.title) chat.title = brief.slice(0, 60);
  chat.updatedAt = Date.now();

  showThread();
  const body = appendTurnDom(brief);
  persistChat(chat); renderChatList(); setTopTitle(chat.title);

  const onEvent = (ev) => aTurn.events.push(ev);
  const done = () => { chat.updatedAt = Date.now(); persistChat(chat); renderChatList(); };
  if (BRIDGE.live && projConnected()) runLive(body, brief, history, onEvent).then(done);
  else { runDemo(body, brief, onEvent); setTimeout(done, 5200); }
}

function wirePrompt() {
  const form = document.getElementById('promptForm');
  const input = document.getElementById('promptInput');
  if (input) {
    input.addEventListener('input', () => autosize(input));
    input.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        if (form.requestSubmit) form.requestSubmit();
        else form.dispatchEvent(new Event('submit', { cancelable: true }));
      }
    });
  }
  if (form) {
    form.addEventListener('submit', (e) => {
      e.preventDefault();
      const brief = (input && input.value.trim()) || '';
      if (!brief) return;
      if (input) { input.value = ''; autosize(input); }
      handleSend(brief);
    });
  }
}

function wireMenu() {
  const btn = document.getElementById('menuBtn');
  const sidebar = document.getElementById('sidebar');
  if (btn && sidebar) btn.addEventListener('click', () => sidebar.classList.toggle('open'));
}

function wireNewChat() {
  const btn = document.getElementById('newChatBtn');
  if (btn) btn.addEventListener('click', () => newChat());
}

/* General agent settings + legal links (popover above the user chip). */
function agentMaxTurns() {
  try { const v = parseInt(localStorage.getItem('oe_max_turns'), 10); return v >= 1 && v <= 12 ? v : 8; } catch (e) { return 8; }
}
function wireUserMenu() {
  const chip = document.getElementById('userChip');
  const menu = document.getElementById('userMenu');
  if (!chip || !menu) return;
  const close = () => { menu.hidden = true; chip.setAttribute('aria-expanded', 'false'); };
  chip.addEventListener('click', (e) => {
    e.stopPropagation();
    menu.hidden = !menu.hidden;
    chip.setAttribute('aria-expanded', String(!menu.hidden));
  });
  menu.addEventListener('click', (e) => e.stopPropagation());
  document.addEventListener('click', () => { if (!menu.hidden) close(); });
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape') close(); });
  const sel = document.getElementById('maxTurns');
  if (sel) {
    sel.value = String(agentMaxTurns());
    sel.addEventListener('change', () => { try { localStorage.setItem('oe_max_turns', sel.value); } catch (e) {} });
  }
  const clear = document.getElementById('clearChats');
  if (clear) clear.addEventListener('click', () => {
    try { localStorage.removeItem('oe_chats'); } catch (e) {}
    close(); newChat();
  });
}

// Collapsible sidebar sections (open/closed state persisted).
function wireCollapsibles() {
  const LS = 'oe_collapsed';
  let state = {};
  try { state = JSON.parse(localStorage.getItem(LS) || '{}'); } catch (e) {}
  document.querySelectorAll('.side-head').forEach((head) => {
    const key = head.dataset.collapse;
    const section = head.closest('.side-section');
    if (state[key]) section.classList.add('collapsed');
    head.setAttribute('aria-expanded', String(!state[key]));
    head.addEventListener('click', () => {
      const collapsed = section.classList.toggle('collapsed');
      head.setAttribute('aria-expanded', String(!collapsed));
      state[key] = collapsed;
      try { localStorage.setItem(LS, JSON.stringify(state)); } catch (e) {}
    });
  });
}

/* Bring-your-own-key: the user pastes their own Studio API key; stored
   client-side only (localStorage). Real email login is future work. */
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
    const topBtn = document.getElementById('topKeyBtn');
    if (topBtn) topBtn.textContent = on ? 'Studio key' : 'Add API key';
    if (on) {
      const label = document.getElementById('keyStatusLabel');
      const body = document.getElementById('keyConnBody');
      if (label) label.textContent = BRIDGE.live ? 'Studio connected' : 'Connected · demo';
      if (body) body.innerHTML = BRIDGE.live
        ? 'Connected as <code>' + maskKey(k) + '</code>. Briefs run the <strong>real agent</strong> against your Studio account through the local bridge.'
        : 'Key saved as <code>' + maskKey(k) + "</code>, but this preview <strong>doesn't call Studio yet</strong>. The data shown is sample, not your account.";
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
  const pop = document.getElementById('keyPop');
  if (top && pop) {
    const closePop = () => { pop.hidden = true; top.setAttribute('aria-expanded', 'false'); };
    top.addEventListener('click', (e) => {
      e.stopPropagation();
      pop.hidden = !pop.hidden;
      top.setAttribute('aria-expanded', String(!pop.hidden));
      if (!pop.hidden && input && connect && !connect.hidden) input.focus();
    });
    pop.addEventListener('click', (e) => e.stopPropagation());
    document.addEventListener('click', () => { if (!pop.hidden) closePop(); });
    document.addEventListener('keydown', (e) => { if (e.key === 'Escape') closePop(); });
  }
  render();
}

/* ── Projects drill-down tree ────────────────────────────────────────────
   Live: project → predictions (grouped by model_id into a synthetic Model
   level) → predictions → prediction-results, lazy-loaded on expand. Demo: a
   canned tree from the sample projects. */
const PROJ_ICONS = {
  map:  '<path d="M9 3L3 6v15l6-3 6 3 6-3V3l-6 3-6-3z"/><path d="M9 3v15M15 6v15"/>',
  drop: '<path d="M12 3s6 6.5 6 11a6 6 0 01-12 0c0-4.5 6-11 6-11z"/>',
  trend:'<path d="M4 17l5-5 3 3 7-8"/><path d="M15 7h5v5"/>',
  leaf: '<path d="M5 19c0-7 5-12 14-12 0 9-5 14-12 12z"/><path d="M9 15c2-3 4-4 7-5"/>',
  sun:  '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/>',
};
const NODE_ICONS = {
  model:      '<rect x="4" y="4" width="16" height="16" rx="2"/><path d="M9 9h6v6H9z"/>',
  prediction: '<path d="M4 17l5-5 3 3 7-8"/><path d="M15 7h5v5"/>',
  result:     '<rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/>',
};
const PROJECTS = [
  { id: 'karst',    name: 'PA Karst',                    meta: '12', icon: 'map'   },
  { id: 'ches',     name: 'Chesapeake - water quality',  meta: '5',  icon: 'drop'  },
  { id: 'potomac',  name: 'Potomac - change detection',  meta: '8',  icon: 'trend' },
  { id: 'mangrove', name: 'Mangrove extent - Indonesia', meta: '3',  icon: 'leaf'  },
  { id: 'solar',    name: 'Solar arrays - California',   meta: '2',  icon: 'sun'   },
];

function projConnected() {
  try { return !!localStorage.getItem('oe_studio_key'); } catch (e) { return false; }
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
function shortId(s) { s = String(s || ''); return s.length > 8 ? s.slice(0, 8) : s; }

function updateProjTag() {
  const tag = document.getElementById('projTag');
  if (!tag) return;
  if (!projConnected()) { tag.hidden = true; return; }
  tag.hidden = false;
  tag.textContent = BRIDGE.live ? 'live' : 'sample';
  tag.title = BRIDGE.live ? 'Your live Studio account' : 'Demo data: not your live Studio account';
}

function treeGlyph(node) {
  return (node.icon && PROJ_ICONS[node.icon]) || NODE_ICONS[node.kind] || PROJ_ICONS.map;
}

function makeTreeNode(node) {
  const el = document.createElement('div');
  el.className = 'tree-node' + (node.leaf ? ' leaf' : '');
  const status = node.status ? '<span class="pred-status ' + escapeHtml(node.status) + '">' + escapeHtml(node.status) + '</span>' : '';
  const meta = (node.meta != null && node.meta !== '') ? '<span class="tw-meta">' + escapeHtml(String(node.meta)) + '</span>' : '';
  el.innerHTML =
    '<button class="tree-row" type="button">' +
      '<svg viewBox="0 0 24 24" class="tw-caret"><path d="M9 6l6 6-6 6"/></svg>' +
      '<svg viewBox="0 0 24 24" class="ic tw-icon">' + treeGlyph(node) + '</svg>' +
      '<span class="tw-label" title="' + escapeHtml(node.name) + '">' + escapeHtml(node.name) + '</span>' +
      status + meta +
    '</button>' +
    '<div class="tree-children" hidden></div>';
  const row = el.querySelector('.tree-row');
  const childBox = el.querySelector('.tree-children');
  if (node.leaf) {
    row.addEventListener('click', () => {
      const input = document.getElementById('promptInput');
      if (input) input.focus();
    });
  } else {
    row.addEventListener('click', () => toggleNode(el, node, childBox));
  }
  return el;
}

async function toggleNode(el, node, childBox) {
  const open = el.classList.toggle('open');
  childBox.hidden = !open;
  if (!open || childBox.dataset.loaded) return;
  childBox.dataset.loaded = '1';
  childBox.innerHTML = '<div class="proj-empty">Loading…</div>';
  try {
    const children = await loadChildren(node);
    childBox.innerHTML = '';
    if (!children.length) { childBox.innerHTML = '<div class="proj-empty">' + emptyLabel(node) + '</div>'; return; }
    children.forEach((c) => childBox.appendChild(makeTreeNode(c)));
  } catch (e) {
    childBox.dataset.loaded = '';  // allow retry on next expand
    childBox.innerHTML = '<div class="proj-empty">Couldn’t load - ' + escapeHtml(String((e && e.message) || e)) + '</div>';
  }
}

function groupByModel(preds) {
  const groups = {};
  preds.forEach((p) => { const m = p.model_id || 'unknown'; (groups[m] = groups[m] || []).push(p); });
  return Object.keys(groups).map((mid) => ({
    kind: 'model', id: mid, name: 'Model ' + shortId(mid), meta: groups[mid].length, predictions: groups[mid],
  }));
}
function predNode(p) {
  return { kind: 'prediction', id: p.id, name: p.name || '(unnamed)', status: (p.status || '').toLowerCase() };
}
function resultNode(r) {
  const props = (r.property_names || []).join(', ');
  return { kind: 'result', id: r.id, name: props || ('result ' + shortId(r.id)), meta: r.file_format || '', leaf: true };
}

async function loadChildren(node) {
  if (!BRIDGE.live) return node.children || [];
  const hdr = { headers: { 'X-Olmoearth-Key': studioKey() } };
  if (node.kind === 'project') {
    const res = await fetch('/api/projects/' + encodeURIComponent(node.id) + '/predictions', hdr);
    if (!res.ok) throw new Error('HTTP ' + res.status);
    return groupByModel((await res.json()).predictions || []);
  }
  if (node.kind === 'model') return node.predictions.map(predNode);
  if (node.kind === 'prediction') {
    const res = await fetch('/api/predictions/' + encodeURIComponent(node.id) + '/results', hdr);
    if (!res.ok) throw new Error('HTTP ' + res.status);
    return ((await res.json()).results || []).map(resultNode);
  }
  return [];
}

function emptyLabel(node) {
  if (node.kind === 'project') return 'No predictions in this project yet.';
  if (node.kind === 'model') return 'No predictions for this model.';
  if (node.kind === 'prediction') return 'No results for this prediction yet.';
  return 'Empty.';
}

function demoProjects() {
  return PROJECTS.map((p) => ({
    kind: 'project', id: p.id, name: p.name, icon: p.icon, meta: p.meta,
    children: [
      { kind: 'model', id: 'm-' + p.id, name: 'Model ' + p.id.slice(0, 4), meta: 1, children: [
        { kind: 'prediction', id: 'pred-' + p.id, name: p.name + ' run', status: 'completed', children: [
          { kind: 'result', id: 'res-' + p.id, name: 'sample_score', meta: 'png', leaf: true },
        ] },
      ] },
    ],
  }));
}

function renderTree(container, nodes) {
  container.innerHTML = '';
  nodes.forEach((n) => container.appendChild(makeTreeNode(n)));
}

async function renderProjects() {
  const list = document.getElementById('projList');
  if (!list) return;
  updateProjTag();
  if (!projConnected()) {
    list.innerHTML = '<div class="proj-empty">Connect your Studio key below to load your projects. <span class="proj-empty-sub">Nothing is fetched until you do.</span></div>';
    return;
  }
  if (!BRIDGE.live) { renderTree(list, demoProjects()); return; }  // sample tree
  list.innerHTML = '<div class="proj-empty">Loading your projects…</div>';
  try {
    const res = await fetch('/api/projects', { headers: { 'X-Olmoearth-Key': studioKey() } });
    if (!res.ok) throw new Error('HTTP ' + res.status);
    const data = await res.json();
    const projects = (data.projects || []).map((p) => ({
      kind: 'project', id: p.id || '', name: p.name || '(unnamed)', icon: pickProjIcon(p.name),
    }));
    if (!projects.length) {
      list.innerHTML = '<div class="proj-empty">No projects in your Studio account yet. <span class="proj-empty-sub">Create one in Studio, then refresh.</span></div>';
      return;
    }
    renderTree(list, projects);
  } catch (e) {
    list.innerHTML = '<div class="proj-empty">Couldn’t load projects - ' + escapeHtml(String((e && e.message) || e)) + '. <span class="proj-empty-sub">Check your key, or that the bridge can reach Studio.</span></div>';
  }
}

/* ── Live bridge detection ───────────────────────────────────────────────
   When served by olmoearth_agent.serve, /api/health answers and we switch to
   the live agent. Opened as a static file, it 404s and the demo runs. */
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

document.addEventListener('DOMContentLoaded', () => {
  renderCards();
  wireNewChat();
  wireTabs();
  wireExamples();
  wirePrompt();
  wireMenu();
  wireUserMenu();
  wireCollapsibles();
  wireKey();        // initial render (demo assumptions) + renderProjects
  renderChatList();
  newChat();        // start on a fresh empty chat (landing visible)
  // Upgrade to live mode if the bridge is serving this page, then re-render.
  detectBridge().then(() => { renderKeyState(); });
});
