/* The bridge API layer: every call to olmoearth_agent.serve lives here, with
   the request-credential readers (Studio key + LLM backend headers) those calls
   use. Centralizing them keeps the headers, error handling, and localStorage
   keys in one place (cf. open-webui's src/lib/apis). */

/* ── Request credentials / settings (localStorage-backed) ── */

export function studioKey() {
  try { return localStorage.getItem('oe_studio_key') || ''; } catch (e) { return ''; }
}

export function projConnected() {
  try { return !!localStorage.getItem('oe_studio_key'); } catch (e) { return false; }
}

export function agentMaxTurns() {
  try { const v = parseInt(localStorage.getItem('oe_max_turns'), 10); return v >= 1 && v <= 12 ? v : 8; } catch (e) { return 8; }
}

/* LLM backend selection (default local). For a hosted backend (claude | openai |
   gemini) the bridge builds a per-request client from the key, which rides in a
   header and is never stored server-side (like the Studio key). Key/model are
   stored per provider. Returns {} for local so the default path is untouched. */
export function llmHeaders() {
  try {
    const backend = localStorage.getItem('oe_llm_backend') || 'local';
    if (backend === 'local') return {};
    const key = (localStorage.getItem('oe_llm_key_' + backend) || '').trim();
    if (!key) return {};
    const h = { 'X-LLM-Backend': backend, 'X-LLM-Key': key };
    const model = (localStorage.getItem('oe_llm_model_' + backend) || '').trim();
    if (model) h['X-LLM-Model'] = model;
    return h;
  } catch (e) { return {}; }
}

/* ── Bridge endpoints ── */

function keyHeaders() { return { 'X-Olmoearth-Key': studioKey() }; }

// GET /api/health -> the health object, or null when absent (static-file demo).
export async function apiHealth() {
  const res = await fetch('/api/health', { method: 'GET' });
  return res.ok ? res.json() : null;
}

// POST /api/run -> the streaming Response; the caller reads the SSE body. Throws on !ok.
export async function apiRunStream(brief, history) {
  const res = await fetch('/api/run', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...keyHeaders(), ...llmHeaders() },
    body: JSON.stringify({ brief, history: history || [], max_turns: agentMaxTurns() }),
  });
  if (!res.ok || !res.body) throw new Error('bridge returned HTTP ' + res.status);
  return res;
}

// GET /api/llm/models for a hosted backend -> [modelId, ...]. Throws with the bridge detail on !ok.
export async function apiLlmModels(backend, key) {
  const res = await fetch('/api/llm/models', { headers: { 'X-LLM-Backend': backend, 'X-LLM-Key': key } });
  if (!res.ok) {
    let detail = 'HTTP ' + res.status;
    try { detail = (await res.json()).detail || detail; } catch (e) {}
    throw new Error(detail);
  }
  return (await res.json()).models || [];
}

// GET /api/projects -> [{id, name}, ...]. Throws on !ok.
export async function apiProjects() {
  const res = await fetch('/api/projects', { headers: keyHeaders() });
  if (!res.ok) throw new Error('HTTP ' + res.status);
  return (await res.json()).projects || [];
}

// POST /api/areas -> the stored area { id, name, project_id, bbox }. Throws on !ok.
export async function apiCreateArea(name, geom, projectId) {
  const res = await fetch('/api/areas', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...keyHeaders() },
    body: JSON.stringify({ name, geom, project_id: projectId }),
  });
  if (!res.ok) {
    let detail = 'HTTP ' + res.status;
    try { detail = (await res.json()).detail || detail; } catch (e) {}
    throw new Error(detail);
  }
  return (await res.json()).area || {};
}

// GET /api/projects/{id}/areas -> [{id, name}, ...]. Throws on !ok.
export async function apiAreas(projectId) {
  const res = await fetch('/api/projects/' + encodeURIComponent(projectId) + '/areas', { headers: keyHeaders() });
  if (!res.ok) throw new Error('HTTP ' + res.status);
  return (await res.json()).areas || [];
}

// GET /api/projects/{id}/predictions -> { predictions, models }. Throws on !ok.
export async function apiPredictions(projectId) {
  const res = await fetch('/api/projects/' + encodeURIComponent(projectId) + '/predictions', { headers: keyHeaders() });
  if (!res.ok) throw new Error('HTTP ' + res.status);
  return res.json();
}

// GET /api/predictions/{id}/results -> [result, ...]. Throws on !ok.
export async function apiResults(predId) {
  const res = await fetch('/api/predictions/' + encodeURIComponent(predId) + '/results', { headers: keyHeaders() });
  if (!res.ok) throw new Error('HTTP ' + res.status);
  return (await res.json()).results || [];
}
