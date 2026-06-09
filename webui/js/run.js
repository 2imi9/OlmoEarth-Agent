/* The run-render engine, shared by the demo and the live agent. A brief plays
   either a canned demo loop or a real SSE stream; both emit the same event
   shape, so one renderer (handleRunEvent) drives both, and replayEvents
   re-renders a saved chat. */

import { escapeHtml, typeText } from './util.js';
import { renderMarkdown } from './markdown.js';
import { apiRunStream, apiSkills } from './api.js';
import { drawAndAttach } from './aoi.js';
import { renderResultViz } from './viz.js';

/* ── Demo scenarios ──────────────────────────────────────────────────────
   In demo mode a brief plays a canned agent loop with the same event shape as
   the live SSE stream. */
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
  // Honest demo: no fabricated stats - it explains the real flow and asks for a key.
  compare: {
    reasoning: "Two runs over the same area with no ground-truth labels, so I compare them to each other (agreement), not accuracy. That needs to sample both rasters - which requires your Studio key.",
    tool: 'olmoearth_compare_results',
    args: '{\n  "result_id_a": "<first result>",\n  "result_id_b": "<second result>"\n}',
    result: [['note', 'demo mode: connect a Studio key to sample the real rasters']],
    answer: "This is the demo, so I can't sample your real rasters here. Connect your **Studio key** (top bar), then either drag two prediction results into the chat or ask me to compare two runs. I'll report model-vs-model **agreement** (correlation, mean difference, agreement %) and scan a live **difference map** - blue where one run scores higher, pink where the other does. No ground-truth labels are needed; that would be *accuracy*, which is a different tool.",
  },
};

function pickScenario(brief) {
  const b = brief || '';
  if (/difference map|compare .*\bruns?\b|do they agree|two (prediction )?results/i.test(b)) return SCENARIOS.compare;
  if (/project|how many|account|water quality/i.test(b)) return SCENARIOS.projects;
  return SCENARIOS.change;
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

/* ── Workflow indicator ─────────────────────────────────────────────────
   A horizontal progress rail above "Reasoning & tools" mapping the agent's tool
   calls to the OlmoEarth pipeline. Only the stages RELEVANT to the run are shown:
   a real prediction run gets the full pipeline; an advisory task (e.g. the
   negative-sampler, which never submits) gets just Load context -> Report -- so
   the rail never shows submit/poll/fetch stuck "pending". Stage keys/predicates
   mirror src/olmoearth_agent/harness/workflow.py (and GET /api/skills). */
const WF_PIPELINE = [
  ['context', 'Load context', (n) => /load_context/.test(n)],
  ['aoi',     'Define AOI',    (n) => /request_aoi/.test(n)],
  ['model',   'Find model',    (n) => /load_skill|search_predictions/.test(n)],
  ['submit',  'Submit run',    (n) => /submit_prediction/.test(n)],
  ['poll',    'Poll status',   (n) => n === 'olmoearth_get_prediction' || /_poll/.test(n)],
  ['fetch',   'Fetch results', (n) => /fetch_results|get_prediction_result|get_results/.test(n)],
  ['report',  'Report',        () => false],  // completed by the final answer
];
const WF_KEYS = WF_PIPELINE.map((s) => s[0]);
const WF_LABEL = Object.fromEntries(WF_PIPELINE.map((s) => [s[0], s[1]]));
const WF_CORE = new Set(['aoi', 'model', 'submit', 'poll', 'fetch']);  // a real pipeline touches one
const WF_CHECK = '<svg viewBox="0 0 12 12" fill="none"><path d="M2.5 6.3l2.3 2.3 4.7-5" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>';
const WF_CROSS = '<svg viewBox="0 0 12 12" fill="none"><path d="M3.2 3.2l5.6 5.6M8.8 3.2l-5.6 5.6" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>';

function wfStageFor(name) {
  const n = name || '';
  for (const [key, , match] of WF_PIPELINE) if (match(n)) return key;
  return null;
}

/* The relevant stage keys for this run, in pipeline order. A forced skill (the
   "/" menu) declares its stages up front (via /api/skills, cached on the body);
   otherwise we infer from the stages reached so far (+ Report), expanding to the
   full pipeline once a prediction is submitted (you can't poll without submit). */
function wfRelevant(body) {
  const forced = body._wfForced && body._wfSkillStages && body._wfSkillStages[body._wfForced];
  if (forced) return WF_KEYS.filter((k) => forced.includes(k));
  const keys = new Set(['context', 'report']);
  (body._wfReached || new Set()).forEach((k) => keys.add(k));
  if (keys.has('submit')) WF_KEYS.forEach((k) => keys.add(k));
  return WF_KEYS.filter((k) => keys.has(k));
}

function wfRender(body) {
  let wf = body.querySelector(':scope > .workflow');
  if (!wf) { wf = document.createElement('div'); wf.className = 'workflow run-step'; body.insertBefore(wf, body.firstChild); }
  const keys = wfRelevant(body);
  const reached = body._wfReached || new Set();
  let activeIdx = -1;
  keys.forEach((k, i) => { if (reached.has(k)) activeIdx = i; });  // furthest reached = active
  const failIdx = body._wfFailedKey ? keys.indexOf(body._wfFailedKey) : -1;
  const done = !!body._wfDone;
  const status = keys.map((k, i) => {
    if (done) return 'is-done';
    if (failIdx === i) return 'is-failed';
    if (i < activeIdx) return 'is-done';
    if (i === activeIdx) return 'is-active';
    return 'is-pending';
  });
  wf.innerHTML = '<div class="wf-eyebrow">Workflow</div><div class="wf-rail">' +
    keys.map((k, i) => {
      const st = status[i];
      const prev = i > 0 ? status[i - 1] : null;
      const lead = prev === 'is-done' ? ' lead-done' : prev === 'is-failed' ? ' lead-fail' : '';
      const inner = st === 'is-done' ? WF_CHECK : st === 'is-failed' ? WF_CROSS : st === 'is-active' ? '<span class="wf-core"></span>' : '';
      return `<div class="wf-step ${st}${lead}"><span class="wf-dot">${inner}</span>` +
        `<span class="wf-label">${escapeHtml(WF_LABEL[k])}</span></div>`;
    }).join('') + '</div>';
}

function workflowOnTool(body, name) {
  const key = wfStageFor(name);
  if (!key) return;
  body._wfReached = body._wfReached || new Set();
  body._wfReached.add(key);
  body._wfFailedKey = null;  // a new tool call means the run advanced past any prior failure
  // Show the rail once a real pipeline stage is reached (not plain Q&A), or
  // immediately if a forced skill already declares its stages.
  if (!body._wfShown && (WF_CORE.has(key) || body._wfForced)) body._wfShown = true;
  if (body._wfShown) wfRender(body);
}
function workflowOnFail(body, name) {
  if (!body._wfShown) return;
  body._wfFailedKey = wfStageFor(name) || body._wfFailedKey || [...(body._wfReached || [])].pop() || 'context';
  wfRender(body);
}
function workflowOnFinal(body) {
  if (!body._wfShown) return;
  // Don't mark complete when the turn PAUSED for an AOI or ended on a failure.
  if (body._wfAwaiting || body._wfFailedKey) return;
  body._wfDone = true;
  wfRender(body);
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
    const wf = body.querySelector(':scope > .workflow');
    if (wf) wf.insertAdjacentElement('afterend', steps);  // keep the rail on top
    else body.insertBefore(steps, body.firstChild);
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
export function handleRunEvent(body, ev, staticRender) {
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
    workflowOnTool(body, ev.name);
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
    // The raw result (chips + JSON) stays in the collapsed "Reasoning & tools".
    // The visual RESULT (maps / charts / compare) belongs in the conversation,
    // so render it into the main body, not the collapsed steps.
    const viz = document.createElement('div');
    viz.className = 'result-viz run-step';
    if (renderResultViz(viz, ev)) body.appendChild(viz);
    if (ev.ok === false) workflowOnFail(body, ev.name);
    // request_aoi with needs_aoi pauses the turn for the user to draw — flag it
    // so the upcoming `final` doesn't mark the whole pipeline complete.
    const aoiInner = ev.result && ev.result.result;
    if (ev.name === 'olmoearth_request_aoi' && aoiInner && aoiInner.needs_aoi) body._wfAwaiting = true;
    maybeAoiPrompt(body, ev);
  } else if (ev.type === 'final') {
    workflowOnFinal(body);
    if (!staticRender) { const steps = body.querySelector(':scope > .steps'); if (steps) steps.classList.add('collapsed'); }
    body.insertAdjacentHTML('beforeend', '<div class="answer run-step md">' + renderMarkdown(ev.content || '(no answer)') + '</div>');
    if (!staticRender) {
      const ans = body.querySelectorAll('.answer');
      const a = ans[ans.length - 1]; if (a) a.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
  } else if (ev.type === 'max_turns') {
    workflowOnFail(body, null);
    body.insertAdjacentHTML('beforeend', '<div class="run-error run-step">Stopped at the turn cap (' + escapeHtml(String(ev.turns)) + ') without a final answer.</div>');
  } else if (ev.type === 'error') {
    workflowOnFail(body, null);
    body.insertAdjacentHTML('beforeend', '<div class="run-error run-step">⚠ ' + escapeHtml(ev.message || 'run failed') + '</div>');
  }
}

/* When the agent calls olmoearth_request_aoi, its result carries
   needs_aoi: true. Surface an inline "Draw the area" button: drawing stores
   the AOI and seeds a follow-up turn (so the agent continues with an
   area_id + bbox), instead of asking the user to type coordinates. */
function maybeAoiPrompt(body, ev) {
  if (ev.name !== 'olmoearth_request_aoi') return;
  const inner = ev.result && ev.result.result;
  if (!inner || !inner.needs_aoi) return;
  if (body.querySelector('.aoi-request')) return;  // one prompt per turn
  const purpose = inner.purpose ? ' for ' + escapeHtml(inner.purpose) : '';
  const wrap = document.createElement('div');
  wrap.className = 'aoi-request run-step';
  wrap.innerHTML =
    '<span class="aoi-request-text">Draw the area of interest' + purpose + ' on a map.</span>' +
    '<button class="aoi-request-btn" type="button">Draw the area</button>';
  body.appendChild(wrap);
  const btn = wrap.querySelector('.aoi-request-btn');
  btn.addEventListener('click', async () => {
    btn.disabled = true;
    const aoi = await drawAndAttach({ purpose: inner.purpose || '', suggestedName: inner.suggested_name || '' });
    btn.disabled = false;
    if (aoi) sendAoiFollowup();
  });
}

/* After a drawn AOI is attached, send a short follow-up brief through the
   normal composer path (which includes the AOI attachment + chat history),
   so the agent resumes with the area available. */
function sendAoiFollowup() {
  const input = document.getElementById('promptInput');
  const form = document.getElementById('promptForm');
  if (!input || !form) return;
  input.value = 'I have drawn the area of interest. Continue using it.';
  if (form.requestSubmit) form.requestSubmit();
  else form.dispatchEvent(new Event('submit', { cancelable: true }));
}

function runStatusEl(label) {
  const status = document.createElement('div');
  status.className = 'think run-step run-status';
  status.innerHTML =
    '<span class="typing"><span></span><span></span><span></span></span>' +
    '<div class="run-status-main">' +
      '<span class="run-status-label">' + label + '</span>' +
      '<div class="progress-bar"></div>' +
    '</div>';
  return status;
}

// Play the canned demo loop into `body`, emitting events to `onEvent`.
export function runDemo(body, brief, onEvent) {
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
// `forcedSkill` (optional) pins the run to one skill server-side (the "/" menu).
export async function runLive(body, brief, history, onEvent, forcedSkill) {
  body.innerHTML = '';
  // Workflow rail: track the stages reached; a forced skill declares its relevant
  // stages up front (so the rail shows the right subset immediately).
  body._wfForced = forcedSkill || null;
  body._wfReached = new Set();
  if (forcedSkill) {
    apiSkills().then((map) => {
      body._wfSkillStages = map;
      if (body._wfForced && map[body._wfForced]) { body._wfShown = true; wfRender(body); }
    }).catch(() => { /* fall back to observed-stage inference */ });
  }
  const status = runStatusEl('Running the agent…');
  body.appendChild(status);
  let cleared = false;
  const clear = () => { if (!cleared) { status.remove(); cleared = true; } };
  try {
    const res = await apiRunStream(brief, history, forcedSkill);
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

// Re-render a saved turn's events (no animation, no auto-scroll).
export function replayEvents(body, events) {
  body.innerHTML = '';
  (events || []).forEach((ev) => handleRunEvent(body, ev, true));
}
