/* Saved comparisons: a sidebar section (parallel to Chats and Projects) that
   stores quantitative two-raster comparisons. A comparison is saved straight
   from a live difference scan (real stats + the sampled diff grid - never
   fabricated), persisted to localStorage, and re-openable in a modal that
   redraws the stored difference map and its stats. */

import { escapeHtml, shortId } from './util.js';
import { renderStoredComparison } from './viz.js';

const LS = 'oe_compares';

function load() { try { return JSON.parse(localStorage.getItem(LS) || '[]'); } catch (e) { return []; } }
function persist(list) { try { localStorage.setItem(LS, JSON.stringify(list)); } catch (e) {} }

function defaultTitle(rec) {
  const a = shortId(rec.resultIdA || ''), b = shortId(rec.resultIdB || '');
  return (rec.property ? rec.property + ': ' : '') + a + ' vs ' + b;
}

/* Save a comparison record (from a completed difference scan). */
export function saveComparison(rec) {
  if (!rec || !rec.stats) return;
  const list = load();
  const stored = {
    id: 'cmp' + Math.random().toString(36).slice(2, 9),
    title: rec.title || defaultTitle(rec),
    resultIdA: rec.resultIdA, resultIdB: rec.resultIdB, property: rec.property || null,
    bbox: rec.bbox, cellSize: rec.cellSize, tolerance: rec.tolerance,
    stats: rec.stats, cells: rec.cells || [],
  };
  list.unshift(stored);
  persist(list.slice(0, 50));  // bound localStorage
  renderCompareList();
  return stored.id;
}

function deleteComparison(id) {
  persist(load().filter((c) => c.id !== id));
  renderCompareList();
}

function openComparison(id) {
  const rec = load().find((c) => c.id === id);
  if (!rec) return;
  const back = document.createElement('div');
  back.className = 'aoi-modal';  // reuse the modal backdrop/card styling
  back.innerHTML =
    '<div class="aoi-card" role="dialog" aria-modal="true" aria-label="Saved comparison">' +
      '<div class="aoi-head"><div class="aoi-title">' + escapeHtml(rec.title) +
        '<span class="aoi-purpose">saved comparison</span></div>' +
        '<button class="aoi-x" type="button" aria-label="Close">&times;</button></div>' +
      '<div class="cmp-modal-body"></div>' +
    '</div>';
  document.body.appendChild(back);
  const body = back.querySelector('.cmp-modal-body');
  void renderStoredComparison(body, rec);
  const close = () => { back.remove(); document.removeEventListener('keydown', onKey); };
  const onKey = (e) => { if (e.key === 'Escape') close(); };
  document.addEventListener('keydown', onKey);
  back.querySelector('.aoi-x').addEventListener('click', close);
  back.addEventListener('click', (e) => { if (e.target === back) close(); });
}

/* Render the saved-comparisons list into the sidebar. */
export function renderCompareList() {
  const list = document.getElementById('cmpList');
  if (!list) return;
  const items = load();
  const countEl = document.getElementById('cmpCount');
  if (countEl) countEl.textContent = items.length ? String(items.length) : '';
  if (!items.length) {
    list.innerHTML = '<div class="side-note">No saved comparisons yet. Compare two result rasters, then "Save comparison".</div>';
    return;
  }
  list.innerHTML = items.map((c) => {
    const corr = c.stats && c.stats.correlation != null ? 'r ' + c.stats.correlation : '';
    return '<button class="chat-item" data-id="' + escapeHtml(c.id) + '" title="' + escapeHtml(c.title) + '">' +
      '<svg viewBox="0 0 24 24" class="ic"><path d="M3 3v18h18"/><path d="M7 14l4-4 3 3 5-6"/></svg>' +
      '<span class="chat-title">' + escapeHtml(c.title) + '</span>' +
      (corr ? '<span class="tw-meta">' + escapeHtml(corr) + '</span>' : '') +
      '<span class="chat-del" data-del="' + escapeHtml(c.id) + '" title="Delete" role="button">×</span>' +
    '</button>';
  }).join('');
  list.querySelectorAll('.chat-item').forEach((el) => {
    el.addEventListener('click', (e) => { if (e.target.closest('.chat-del')) return; openComparison(el.dataset.id); });
  });
  list.querySelectorAll('.chat-del').forEach((el) => {
    el.addEventListener('click', (e) => { e.stopPropagation(); deleteComparison(el.dataset.del); });
  });
}

/* Wire the panel: listen for save events from a difference scan, render once. */
export function wireCompares() {
  document.addEventListener('oe:save-comparison', (e) => saveComparison(e.detail));
  renderCompareList();
}
