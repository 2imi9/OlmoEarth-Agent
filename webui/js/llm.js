/* LLM backend picker + the cloud-path nudge. Provider subtab: Local | Claude |
   Gemini | ChatGPT. Picking a hosted provider reveals a key field + an
   auto-detected model dropdown; key/model are remembered per provider, and the
   active provider is oe_llm_backend. "Detect" lists the provider's models. */

import { BRIDGE } from './store.js';
import { escapeHtml } from './util.js';
import { apiLlmModels } from './api.js';

export function wireLlmSubtab() {
  const tabs = document.getElementById('llmTabs');
  const panel = document.getElementById('llmPanel');
  const keyEl = document.getElementById('llmKey');
  const modelEl = document.getElementById('llmModel');
  const detectBtn = document.getElementById('llmDetect');
  const note = document.getElementById('llmNote');
  const localNote = document.getElementById('llmLocalNote');
  if (!tabs || !panel) return;
  const lsGet = (k, d) => { try { return localStorage.getItem(k) || d; } catch (e) { return d; } };
  const lsSet = (k, v) => { try { localStorage.setItem(k, v); } catch (e) {} };
  const DEFAULT_NOTE = 'Key is sent only to the provider, per run. Never stored on the server.';
  let backend = lsGet('oe_llm_backend', 'local');

  function fillModels(ids) {
    if (!modelEl) return;
    const saved = lsGet('oe_llm_model_' + backend, '');
    modelEl.innerHTML = '<option value="">' + (ids.length ? 'Select a model' : 'Auto-detect models...') + '</option>'
      + ids.map((id) => '<option value="' + escapeHtml(id) + '">' + escapeHtml(id) + '</option>').join('');
    if (saved) modelEl.value = saved;
  }

  function syncProvider() {
    tabs.querySelectorAll('.llm-tab').forEach((t) => t.classList.toggle('is-active', t.dataset.backend === backend));
    panel.hidden = backend === 'local';
    if (localNote) localNote.hidden = backend !== 'local';  // Local: no provider key, just the Studio key
    if (note) note.textContent = DEFAULT_NOTE;
    if (backend === 'local') return;
    const savedKey = lsGet('oe_llm_key_' + backend, '');
    if (keyEl) keyEl.value = savedKey;
    if (note && backend === 'claude' && BRIDGE.claudeAvailable === false) {
      note.textContent = 'Claude needs the anthropic extra on the server (uv sync --extra claude).';
    } else if (note && !savedKey) {
      note.textContent = 'No API key yet - runs use the local model until you add one.';
    }
    let cached = [];
    try { cached = JSON.parse(lsGet('oe_llm_models_' + backend, '[]')); } catch (e) {}
    fillModels(Array.isArray(cached) ? cached : []);
  }

  async function detectModels() {
    if (backend === 'local' || !keyEl) return;
    const b = backend;  // capture: the user may switch tabs before the fetch resolves
    const key = keyEl.value.trim();
    if (!key) { if (note) note.textContent = 'Enter an API key first.'; return; }
    if (note) note.textContent = 'Detecting models...';
    try {
      const ids = await apiLlmModels(b, key);
      try { lsSet('oe_llm_models_' + b, JSON.stringify(ids)); } catch (e2) {}
      if (b !== backend) return;  // user switched provider mid-flight; leave the active panel alone
      fillModels(ids);
      if (note) note.textContent = ids.length
        ? (ids.length + ' models detected. ' + DEFAULT_NOTE)
        : 'No models returned; the provider default will be used.';
    } catch (e) {
      if (b !== backend) return;
      if (note) note.textContent = 'Could not detect models (' + ((e && e.message) || e) + ').';
    }
  }

  tabs.querySelectorAll('.llm-tab').forEach((tab) => {
    tab.addEventListener('click', () => {
      backend = tab.dataset.backend || 'local';
      lsSet('oe_llm_backend', backend);
      syncProvider();
      updateLlmNudge();  // hide once a cloud provider is chosen; show again on 'local'
      if (backend !== 'local' && keyEl && keyEl.value.trim()) detectModels();
    });
  });
  if (keyEl) keyEl.addEventListener('change', () => {
    lsSet('oe_llm_key_' + backend, keyEl.value.trim());
    if (keyEl.value.trim()) detectModels();
  });
  if (modelEl) modelEl.addEventListener('change', () => lsSet('oe_llm_model_' + backend, modelEl.value));
  if (detectBtn) detectBtn.addEventListener('click', () => detectModels());
  syncProvider();
}

/* Cloud-path nudge. When the bridge is live, the default 'local' backend is
   selected, and /api/health reported the local model unreachable, show a
   one-line banner pointing to a cloud provider (or `make serve`). Dismissed for
   the session only, so it returns on reload while the condition holds. */
let llmNudgeDismissed = false;
function activeLlmBackend() {
  try { return localStorage.getItem('oe_llm_backend') || 'local'; } catch (e) { return 'local'; }
}
export function updateLlmNudge() {
  const el = document.getElementById('llmNudge');
  if (!el) return;
  const show = BRIDGE.live && !llmNudgeDismissed
    && activeLlmBackend() === 'local' && BRIDGE.llmLocalUp === false;
  el.hidden = !show;
}
export function wireLlmNudge() {
  const dismiss = document.getElementById('llmNudgeDismiss');
  const pick = document.getElementById('llmNudgePick');
  if (dismiss) dismiss.addEventListener('click', () => { llmNudgeDismissed = true; updateLlmNudge(); });
  if (pick) pick.addEventListener('click', (e) => {
    e.stopPropagation();  // don't let the document handler close the menu we open
    const chip = document.getElementById('userChip');
    const menu = document.getElementById('userMenu');
    if (menu && chip) { menu.hidden = false; chip.setAttribute('aria-expanded', 'true'); }
    const claudeTab = document.querySelector('.llm-tab[data-backend="claude"]');
    if (claudeTab) claudeTab.focus();
  });
}
