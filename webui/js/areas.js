/* Areas pane: every AOI across all your projects, flattened into one labeled
   list (area name + its project). Studio has no single "all areas" endpoint, so
   live mode fans out one fetch per project and fills the list progressively as
   each resolves — never blocking on a full load. Each row drags into the chat to
   attach the AOI (same contract as the Projects tree). Chats + Comparisons are
   localStorage; Projects + Areas need a Studio key. */

import { BRIDGE } from './store.js';
import { escapeHtml } from './util.js';
import { projConnected, apiProjects, apiAreas } from './api.js';

const PIN = '<path d="M12 2C8 2 5 5 5 9c0 5 7 13 7 13s7-8 7-13c0-4-3-7-7-7z"/><circle cx="12" cy="9" r="2.5"/>';

// Guards against a stale fan-out painting after the key/segment changed.
let renderToken = 0;

function setCount(text) {
  const el = document.getElementById('areaCount');
  if (el) el.textContent = text || '';
}

function connectNudge() {
  return '<div class="sb-nudge">' +
    '<span class="nu-ic"><svg viewBox="0 0 24 24" class="ic"><circle cx="8" cy="15" r="4"/><path d="M10.8 12.2L20 3M17 4l2 2M14 7l2 2"/></svg></span>' +
    '<div class="nu-t">Connect your Studio key</div>' +
    '<div class="nu-d">Add your key below to list AOIs across all your projects.</div>' +
    '</div>';
}

/* One AOI row. `area` = {id, name, projectId, projectName, geom?, demo?}. Live
   rows carry only id/name/project_id; attach.js fetches the geometry on drop. */
function areaRow(area) {
  const el = document.createElement('div');
  el.className = 'area-row';
  el.innerHTML =
    '<svg viewBox="0 0 24 24" class="ic ar-pin">' + PIN + '</svg>' +
    '<span class="ar-meta">' +
      '<span class="ar-name" title="' + escapeHtml(area.name) + '">' + escapeHtml(area.name) + '</span>' +
      '<span class="ar-proj" title="' + escapeHtml(area.projectName) + '">' + escapeHtml(area.projectName) + '</span>' +
    '</span>';
  el.draggable = true;
  el.title = 'Drag into the chat to attach this area';
  el.addEventListener('dragstart', (e) => {
    const payload = area.demo
      ? { id: area.id, name: area.name, geom: area.geom, bbox: area.bbox, project_id: area.projectId, demo: true }
      : { id: area.id, name: area.name, project_id: area.projectId };
    e.dataTransfer.setData('application/x-oe-aoi', JSON.stringify(payload));
    e.dataTransfer.setData('text/plain', area.name || 'AOI');
    e.dataTransfer.effectAllowed = 'copy';
  });
  return el;
}

/* Demo areas: one AOI per sample project, so the pane is populated before a key
   is connected (mirrors the demo Projects tree). */
function demoAreas() {
  const demoGeom = { type: 'Polygon', coordinates: [[[-0.1, 0.0], [0.1, 0.0], [0.1, 0.15], [-0.1, 0.15], [-0.1, 0.0]]] };
  const names = [
    ['PA Karst', 'PA Karst'],
    ['Chesapeake - water quality', 'Chesapeake - water quality'],
    ['Potomac - change detection', 'Potomac - change detection'],
    ['Mangrove extent - Indonesia', 'Mangrove extent - Indonesia'],
  ];
  return names.map(([proj], i) => ({
    id: 'area-demo-' + i, name: proj + ' AOI', projectName: proj, projectId: 'demo-' + i,
    geom: demoGeom, bbox: [-0.1, 0.0, 0.1, 0.15], demo: true,
  }));
}

export async function renderAreas() {
  const list = document.getElementById('areaList');
  if (!list) return;
  const token = ++renderToken;

  if (!projConnected()) {
    list.innerHTML = connectNudge();
    setCount('');
    return;
  }

  if (!BRIDGE.live) {  // sample mode: a key is set but the bridge isn't live
    const demo = demoAreas();
    list.innerHTML = '';
    demo.forEach((a) => list.appendChild(areaRow(a)));
    setCount(String(demo.length));
    return;
  }

  // Live: fan out one fetch per project, fill in as each resolves.
  list.innerHTML = '<div class="side-note">Loading areas…</div>';
  let projects;
  try { projects = await apiProjects(); }
  catch (e) {
    if (token !== renderToken) return;
    list.innerHTML = '<div class="proj-empty">Couldn’t load projects - ' + escapeHtml(String((e && e.message) || e)) + '.</div>';
    return;
  }
  if (token !== renderToken) return;
  if (!projects.length) {
    list.innerHTML = '<div class="proj-empty">No projects in your Studio account yet.</div>';
    setCount('');
    return;
  }

  list.innerHTML = '';
  const footer = document.createElement('div');
  footer.className = 'area-prog';
  list.appendChild(footer);
  const total = projects.length;
  let done = 0, found = 0;
  const updateFooter = () => {
    const remaining = total - done;
    footer.hidden = remaining <= 0;
    footer.innerHTML = remaining > 0
      ? '<span class="ar-spin"></span>Loading ' + remaining + ' more project' + (remaining === 1 ? '' : 's') + '…'
      : '';
    setCount(remaining > 0 ? done + ' / ' + total : String(found));
  };
  updateFooter();

  await Promise.all(projects.map(async (p) => {
    let areas = [];
    try { areas = await apiAreas(p.id); } catch (e) { areas = []; }
    if (token !== renderToken) return;  // a newer render superseded this fan-out
    for (const a of areas) {
      const row = areaRow({ id: a.id, name: a.name || '(unnamed)', projectId: p.id, projectName: p.name || '(project)' });
      list.insertBefore(row, footer);
      found += 1;
    }
    done += 1;
    updateFooter();
  }));

  if (token !== renderToken) return;
  if (!found) {
    list.innerHTML = '<div class="proj-empty">No saved areas yet. <span class="proj-empty-sub">Draw one with the map button in the composer, or open a project and save an AOI.</span></div>';
    setCount('');
  }
}
