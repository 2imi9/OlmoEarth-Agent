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

/* Show one segment: flip the tab state (+ roving tabindex), swap the visible
   pane, render its list. */
export function showSegment(seg) {
  if (!PANES[seg]) seg = 'projects';
  try { localStorage.setItem(LS, seg); } catch (e) {}
  document.querySelectorAll('#sideSeg .seg-btn').forEach((b) => {
    const on = b.dataset.seg === seg;
    b.setAttribute('aria-selected', String(on));
    b.tabIndex = on ? 0 : -1;  // roving tabindex: only the active tab is in the Tab order
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
    // WAI-ARIA tablist keyboard pattern: Left/Right (and Home/End) move between
    // tabs, auto-activating each (cheap panel swap) and following focus.
    seg.addEventListener('keydown', (e) => {
      const keys = ['ArrowLeft', 'ArrowRight', 'Home', 'End'];
      if (!keys.includes(e.key)) return;
      const tabs = [...seg.querySelectorAll('.seg-btn')];
      if (!tabs.length) return;
      const cur = tabs.findIndex((t) => t.getAttribute('aria-selected') === 'true');
      let next = cur < 0 ? 0 : cur;
      if (e.key === 'ArrowLeft') next = (cur - 1 + tabs.length) % tabs.length;
      else if (e.key === 'ArrowRight') next = (cur + 1) % tabs.length;
      else if (e.key === 'Home') next = 0;
      else if (e.key === 'End') next = tabs.length - 1;
      e.preventDefault();
      showSegment(tabs[next].dataset.seg);
      tabs[next].focus();
    });
  }
  showSegment(current());  // initial paint of the persisted segment
}
