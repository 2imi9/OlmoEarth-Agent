/* The sidebar's segmented switcher: Projects · Comparisons · Areas all feed one
   secondary pane, only one shown at a time (Claude-rail style). The active
   segment is persisted, and switching to a pane (re-)renders its list lazily —
   cheap, since the underlying fetches are cached. */

import { renderProjects } from './projects.js';
import { renderCompareList } from './compares.js';
import { renderAreas } from './areas.js';

const LS = 'oe_sb_segment';
const PANES = { projects: 'paneProjects', comparisons: 'paneComparisons', areas: 'paneAreas' };

function current() {
  try { const v = localStorage.getItem(LS); return PANES[v] ? v : 'projects'; }
  catch (e) { return 'projects'; }
}

function renderSegment(seg) {
  if (seg === 'projects') renderProjects();
  else if (seg === 'comparisons') renderCompareList();
  else if (seg === 'areas') renderAreas();
}

/* Show one segment: flip the tab state, swap the visible pane, render its list. */
export function showSegment(seg) {
  if (!PANES[seg]) seg = 'projects';
  try { localStorage.setItem(LS, seg); } catch (e) {}
  document.querySelectorAll('#sideSeg .seg-btn').forEach((b) => {
    b.setAttribute('aria-selected', String(b.dataset.seg === seg));
  });
  Object.entries(PANES).forEach(([key, id]) => {
    const pane = document.getElementById(id);
    if (pane) pane.hidden = key !== seg;
  });
  renderSegment(seg);
}

/* Re-render whichever pane is currently visible (e.g. after the Studio key
   changes and caches are cleared). */
export function refreshActiveSegment() { renderSegment(current()); }

export function wireSwitcher() {
  const seg = document.getElementById('sideSeg');
  if (seg) {
    seg.addEventListener('click', (e) => {
      const b = e.target.closest('.seg-btn');
      if (b && b.dataset.seg) showSegment(b.dataset.seg);
    });
  }
  showSegment(current());  // initial paint of the persisted segment
}
