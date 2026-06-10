/* Saved results: a sidebar section (parallel to Chats and Projects) that stores
   the outputs a chat produces. A result is either a quantitative two-raster
   COMPARISON (real stats + the sampled diff grid, captured from a live scan -
   never fabricated) or a downloadable FILE artifact in some format (an SLD
   style, a case-narrative report, exported JSON/CSV, ...). Each record carries a
   `type` ('comparison' | 'file') and a `format` badge; persisted to
   localStorage. Comparisons re-open in a difference-map modal; file results open
   a text preview + re-download. */

import { escapeHtml, shortId } from './util.js';
import { renderStoredComparison } from './viz.js';
import { downloadText } from './download.js';

const LS = 'oe_results';
const LEGACY_LS = 'oe_compares';  // the old comparisons-only store (migrated once)

function persist(list) { try { localStorage.setItem(LS, JSON.stringify(list)); } catch (e) {} }

function load() {
  try {
    const raw = localStorage.getItem(LS);
    if (raw != null) { const v = JSON.parse(raw); return Array.isArray(v) ? v : []; }
  } catch (e) {}
  // One-time migration: fold the legacy comparisons into the new results store.
  try {
    const legacy = JSON.parse(localStorage.getItem(LEGACY_LS) || '[]');
    const migrated = (Array.isArray(legacy) ? legacy : []).map((c) => ({ ...c, type: 'comparison', format: 'raster-diff' }));
    if (migrated.length) persist(migrated);
    return migrated;
  } catch (e) { return []; }
}

function newId() { return 'res' + Math.random().toString(36).slice(2, 9); }

function cmpTitle(rec) {
  const a = shortId(rec.resultIdA || ''), b = shortId(rec.resultIdB || '');
  return (rec.property ? rec.property + ': ' : '') + a + ' vs ' + b;
}

/* Save any result. `rec.type` is 'file' (a downloadable artifact:
   {format, filename, mime, text, title}) or 'comparison' (a raster diff, the
   default). Returns the stored id. */
export function saveResult(rec) {
  if (!rec) return;
  const list = load();
  let stored;
  if (rec.type === 'file') {
    if (!rec.text) return;
    const format = (rec.format || 'file').toLowerCase();
    stored = {
      id: newId(), type: 'file', format,
      title: rec.title || rec.filename || (format + ' output'),
      filename: rec.filename || ('output.' + format), mime: rec.mime || 'text/plain', text: rec.text,
    };
  } else {
    if (!rec.stats) return;  // a comparison must carry real captured stats
    stored = {
      id: newId(), type: 'comparison', format: 'raster-diff',
      title: rec.title || cmpTitle(rec),
      resultIdA: rec.resultIdA, resultIdB: rec.resultIdB, property: rec.property || null,
      kind: rec.kind, bbox: rec.bbox, cellSize: rec.cellSize, tolerance: rec.tolerance,
      stats: rec.stats, cells: rec.cells || [],
    };
    // Idempotent re-save: the same A-vs-B-on-property comparison maps to one row.
    const dupe = list.find((r) => r.type === 'comparison'
      && r.resultIdA === stored.resultIdA && r.resultIdB === stored.resultIdB && r.property === stored.property);
    if (dupe) return dupe.id;
  }
  list.unshift(stored);
  persist(list.slice(0, 60));  // bound localStorage
  renderResultsList();
  return stored.id;
}

function deleteResult(id) { persist(load().filter((r) => r.id !== id)); renderResultsList(); }

/* A short uppercase badge for the format chip. */
const FORMAT_BADGE = {
  'raster-diff': 'DIFF', json: 'JSON', csv: 'CSV', geojson: 'GEOJSON',
  sld: 'SLD', md: 'MD', markdown: 'MD', qlr: 'QLR', xml: 'XML', file: 'FILE',
};
function badgeFor(r) { return FORMAT_BADGE[r.format] || (r.format ? r.format.toUpperCase().slice(0, 7) : 'FILE'); }

function openResult(id) {
  const rec = load().find((r) => r.id === id);
  if (!rec) return;
  const back = document.createElement('div');
  back.className = 'aoi-modal';  // reuse the modal backdrop/card styling
  const purpose = rec.type === 'file' ? ('saved ' + (rec.format || 'file')) : 'saved comparison';
  back.innerHTML =
    '<div class="aoi-card" role="dialog" aria-modal="true" aria-label="Saved result">' +
      '<div class="aoi-head"><div class="aoi-title">' + escapeHtml(rec.title) +
        '<span class="aoi-purpose">' + escapeHtml(purpose) + '</span></div>' +
        '<button class="aoi-x" type="button" aria-label="Close">&times;</button></div>' +
      '<div class="cmp-modal-body"></div>' +
    '</div>';
  document.body.appendChild(back);
  const body = back.querySelector('.cmp-modal-body');
  if (rec.type === 'file') {
    const pre = document.createElement('pre');
    pre.className = 'result-file-pre';
    pre.textContent = rec.text.length > 20000 ? rec.text.slice(0, 20000) + '\n…(truncated)' : rec.text;
    body.appendChild(pre);
    const dl = document.createElement('button');
    dl.type = 'button'; dl.className = 'dl-btn';
    dl.textContent = 'Download ' + (rec.filename || 'file');
    dl.addEventListener('click', () => downloadText(rec.filename || 'output', rec.text, rec.mime || 'text/plain'));
    body.appendChild(dl);
  } else {
    void renderStoredComparison(body, rec);
  }
  const close = () => { back.remove(); document.removeEventListener('keydown', onKey); };
  const onKey = (e) => { if (e.key === 'Escape') close(); };
  document.addEventListener('keydown', onKey);
  back.querySelector('.aoi-x').addEventListener('click', close);
  back.addEventListener('click', (e) => { if (e.target === back) close(); });
}

/* Render the saved-results list into the sidebar. */
export function renderResultsList() {
  const list = document.getElementById('resultsList');
  if (!list) return;
  const items = load();
  const countEl = document.getElementById('resultsCount');
  if (countEl) countEl.textContent = items.length ? String(items.length) : '';
  if (!items.length) {
    list.innerHTML = '<div class="side-note">No saved results yet. Save a comparison, or any downloadable output (SLD, report, JSON) a chat produces.</div>';
    return;
  }
  list.innerHTML = items.map((r) => {
    const corr = (r.type === 'comparison' && r.stats && r.stats.correlation != null) ? 'r ' + r.stats.correlation : '';
    return '<button class="chat-item result-item" data-id="' + escapeHtml(r.id) + '" title="' + escapeHtml(r.title) + '">' +
      '<span class="res-badge res-' + escapeHtml(r.format || 'file') + '">' + escapeHtml(badgeFor(r)) + '</span>' +
      '<span class="chat-title">' + escapeHtml(r.title) + '</span>' +
      (corr ? '<span class="tw-meta">' + escapeHtml(corr) + '</span>' : '') +
      '<span class="chat-del" data-del="' + escapeHtml(r.id) + '" title="Delete" role="button">×</span>' +
    '</button>';
  }).join('');
  list.querySelectorAll('.chat-item').forEach((el) => {
    el.addEventListener('click', (e) => { if (e.target.closest('.chat-del')) return; openResult(el.dataset.id); });
  });
  list.querySelectorAll('.chat-del').forEach((el) => {
    el.addEventListener('click', (e) => { e.stopPropagation(); deleteResult(el.dataset.del); });
  });
}

/* Wire the panel: a difference scan still emits oe:save-comparison; any other
   output emits oe:save-result with {type:'file', ...}. */
export function wireResults() {
  document.addEventListener('oe:save-comparison', (e) => saveResult({ ...e.detail, type: 'comparison' }));
  document.addEventListener('oe:save-result', (e) => saveResult(e.detail));
  renderResultsList();
}
