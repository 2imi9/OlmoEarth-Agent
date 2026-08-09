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
   gemini | nim) the bridge builds a per-request client from the key, which rides in a
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
    /* Opt-in auto-routing: the bridge demotes simple read-only lookups to the
       free local model, saving hosted tokens (Shippy-style model routing). */
    if (localStorage.getItem('oe_llm_auto') === '1') h['X-LLM-Route'] = 'auto';
    return h;
  } catch (e) { return {}; }
}

/* ── Bridge endpoints ── */

function keyHeaders() { return { 'X-Olmoearth-Key': studioKey() }; }

/* Client-side GET cache: reopening the Projects tree or the AOI modal should
   not re-fetch the same read through the (slow) proxy every time. Caches the
   in-flight promise (concurrent callers share one request) + the resolved
   JSON for `ttl` ms; a failed request is evicted so it retries. Cleared on a
   key change; areas invalidated after a create. */
const _getCache = new Map();  // url -> { ts, promise }
const _TTL_LONG = 300000;     // projects / areas / extent: rarely change in a session
const _TTL_SHORT = 45000;     // predictions / results: change as runs progress

function _cachedGet(url, ttl) {
  const hit = _getCache.get(url);
  if (hit && Date.now() - hit.ts < ttl) return hit.promise;
  const promise = fetch(url, { headers: keyHeaders() })
    .then((res) => { if (!res.ok) throw new Error('HTTP ' + res.status); return res.json(); })
    .catch((e) => { _getCache.delete(url); throw e; });
  _getCache.set(url, { ts: Date.now(), promise });
  return promise;
}

/* Drop all cached reads (e.g. when the Studio key changes). */
export function clearApiCache() { _getCache.clear(); }

/* Drop cached reads whose URL contains `substr` (e.g. after creating an area). */
export function invalidateApi(substr) {
  for (const k of Array.from(_getCache.keys())) if (k.includes(substr)) _getCache.delete(k);
}

// GET /api/health -> the health object, or null when absent (static-file demo).
export async function apiHealth() {
  const res = await fetch('/api/health', { method: 'GET' });
  return res.ok ? res.json() : null;
}

// POST /api/run -> the streaming Response; the caller reads the SSE body. Throws on !ok.
// `forcedSkill` (optional) pins the run to one skill server-side (the "/" menu slug).
export async function apiRunStream(brief, history, forcedSkill) {
  const body = { brief, history: history || [], max_turns: agentMaxTurns() };
  if (forcedSkill) body.forced_skill = forcedSkill;
  const res = await fetch('/api/run', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...keyHeaders(), ...llmHeaders() },
    body: JSON.stringify(body),
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

// GET /api/projects -> [{id, name}, ...]. Cached. Throws on !ok.
export async function apiProjects() {
  return (await _cachedGet('/api/projects', _TTL_LONG)).projects || [];
}

// GET /api/skills -> { slug: [stageKey, ...] } (per-skill relevant workflow
// stages). Cached long (the catalog is static). Returns {} on error so the
// workflow rail falls back to inferring stages from observed tool calls.
export async function apiSkills() {
  try {
    const body = await _cachedGet('/api/skills', _TTL_LONG);
    return Object.fromEntries((body.skills || []).map((s) => [s.slug, s.stages || []]));
  } catch (e) { return {}; }
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
  // The project's area list just changed -> drop its cached read.
  invalidateApi('/api/projects/' + encodeURIComponent(projectId) + '/areas');
  return (await res.json()).area || {};
}

// GET /api/projects/{id}/areas -> [{id, name}, ...]. Cached. Throws on !ok.
export async function apiAreas(projectId) {
  const url = '/api/projects/' + encodeURIComponent(projectId) + '/areas';
  return (await _cachedGet(url, _TTL_LONG)).areas || [];
}

// GET /api/pixel-value -> { value, property, categorical } (value null = nodata). Throws on !ok.
export async function apiPixelValue(resultId, lon, lat, property) {
  const qs = new URLSearchParams({ result_id: resultId, lon: String(lon), lat: String(lat) });
  if (property) qs.set('property', property);
  const res = await fetch('/api/pixel-value?' + qs.toString(), { headers: keyHeaders() });
  if (!res.ok) throw new Error('HTTP ' + res.status);
  return res.json();
}

// GET /api/areas/{id} -> { id, name, project_id, geom, bbox }. Cached. Throws on !ok.
export async function apiArea(areaId) {
  const url = '/api/areas/' + encodeURIComponent(areaId);
  return (await _cachedGet(url, _TTL_LONG)).area || {};
}

// GET /api/results/{id}/extent -> { result_id, bbox, tile_url, property }. Cached. Throws on !ok.
export async function apiResultExtent(resultId) {
  const url = '/api/results/' + encodeURIComponent(resultId) + '/extent';
  return (await _cachedGet(url, _TTL_LONG)).extent || {};
}

// GET /api/projects/{id}/predictions -> { predictions, models }. Cached (short). Throws on !ok.
export async function apiPredictions(projectId) {
  const url = '/api/projects/' + encodeURIComponent(projectId) + '/predictions';
  return _cachedGet(url, _TTL_SHORT);
}

// GET /api/predictions/{id}/results -> [result, ...]. Cached (short). Throws on !ok.
export async function apiResults(predId) {
  const url = '/api/predictions/' + encodeURIComponent(predId) + '/results';
  return (await _cachedGet(url, _TTL_SHORT)).results || [];
}
