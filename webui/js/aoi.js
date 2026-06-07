/* Interactive AOI draw widget. Opens a map (Leaflet + Leaflet.draw, OSM
   basemap) in a modal where the user draws a rectangle or polygon; the drawn
   GeoJSON is optionally stored as an OlmoEarth Studio area (POST /api/areas)
   and attached to the next brief so the agent can feed it to AOI-needing
   skills. Leaflet ships UMD, so it is loaded lazily via <script> tags (like
   attach.js lazy-loads pdf.js) - no build step. */

import { BRIDGE } from './store.js';
import { escapeHtml } from './util.js';
import { apiArea, apiAreas, apiCreateArea, apiProjects, projConnected } from './api.js';
import { addAoiAttachment } from './attach.js';
import { loadLeaflet, geomBbox } from './leaflet.js';

function fmtBbox(b) {
  return b.map((n) => n.toFixed(4)).join(', ');
}

function defaultName() {
  // A drawn-area name; the user can edit it. Date is not used (kept generic).
  return 'Drawn AOI';
}

/* Build the modal DOM and return its key elements. */
function buildModalDom(opts) {
  const back = document.createElement('div');
  back.className = 'aoi-modal';
  back.innerHTML =
    '<div class="aoi-card" role="dialog" aria-modal="true" aria-label="Draw an area of interest">' +
      '<div class="aoi-head">' +
        '<div class="aoi-title">Draw an area of interest' +
          (opts.purpose ? '<span class="aoi-purpose">' + escapeHtml(opts.purpose) + '</span>' : '') +
        '</div>' +
        '<button class="aoi-x" type="button" aria-label="Close">&times;</button>' +
      '</div>' +
      '<p class="aoi-hint">Use the rectangle or polygon tool (top-left of the map) to draw. ' +
        'Click points for a polygon, then click the first point to close it.</p>' +
      '<div class="aoi-map" id="aoiMap"></div>' +
      '<div class="aoi-meta">' +
        '<label class="aoi-field"><span>Name</span>' +
          '<input class="aoi-input" id="aoiName" type="text" autocomplete="off" spellcheck="false" /></label>' +
        '<label class="aoi-field aoi-proj-field"><span>Store in project</span>' +
          '<select class="aoi-input" id="aoiProject" aria-label="Project to store the area in"></select></label>' +
        '<label class="aoi-field aoi-saved-field"><span>Or reuse a saved area</span>' +
          '<select class="aoi-input" id="aoiSaved" aria-label="Reuse a saved area"></select></label>' +
      '</div>' +
      '<div class="aoi-bbox" id="aoiBbox">No area drawn yet.</div>' +
      '<div class="aoi-actions">' +
        '<button class="aoi-btn aoi-cancel" type="button">Cancel</button>' +
        '<button class="aoi-btn aoi-use is-primary" type="button" disabled>Use area</button>' +
      '</div>' +
      '<div class="aoi-status" id="aoiStatus" hidden></div>' +
    '</div>';
  document.body.appendChild(back);
  return {
    back,
    map: back.querySelector('#aoiMap'),
    name: back.querySelector('#aoiName'),
    project: back.querySelector('#aoiProject'),
    projField: back.querySelector('.aoi-proj-field'),
    saved: back.querySelector('#aoiSaved'),
    savedField: back.querySelector('.aoi-saved-field'),
    bbox: back.querySelector('#aoiBbox'),
    use: back.querySelector('.aoi-use'),
    cancel: back.querySelector('.aoi-cancel'),
    close: back.querySelector('.aoi-x'),
    status: back.querySelector('#aoiStatus'),
  };
}

/* Populate the project picker from the live account; hide it when storing
   isn't possible (demo mode or no key) - the AOI is then used un-stored. */
async function fillProjects(els) {
  const canStore = BRIDGE.live && projConnected();
  if (!canStore) {
    els.projField.hidden = true;
    els.savedField.hidden = true;
    els.use.textContent = 'Use area';
    return [];
  }
  els.project.innerHTML = '<option value="">Loading projects…</option>';
  try {
    const projects = await apiProjects();
    if (!projects.length) {
      els.project.innerHTML = '<option value="">No projects - area used un-stored</option>';
      els.projField.classList.add('is-empty');
      els.savedField.hidden = true;
      return [];
    }
    els.project.innerHTML = projects
      .map((p) => `<option value="${escapeHtml(p.id)}">${escapeHtml(p.name || p.id)}</option>`)
      .join('');
    return projects;
  } catch (e) {
    els.project.innerHTML = '<option value="">Couldn’t load projects</option>';
    els.savedField.hidden = true;
    return [];
  }
}

const _NEW_AREA = '__new__';

/* Populate the "reuse a saved area" dropdown for the selected project. The
   first option always draws a new area; the rest are the project's stored
   areas (id + name). */
async function loadAreas(els, projectId) {
  if (!projectId) { els.savedField.hidden = true; return; }
  els.savedField.hidden = false;
  els.saved.innerHTML = `<option value="${_NEW_AREA}">Loading saved areas…</option>`;
  try {
    const areas = await apiAreas(projectId);
    const opts = [`<option value="${_NEW_AREA}">— Draw a new area —</option>`];
    for (const a of areas) {
      opts.push(`<option value="${escapeHtml(a.id)}">${escapeHtml(a.name || a.id)}</option>`);
    }
    els.saved.innerHTML = opts.join('');
  } catch (e) {
    els.saved.innerHTML = `<option value="${_NEW_AREA}">— Draw a new area —</option>`;
  }
}

function setStatus(els, msg, kind) {
  if (!msg) { els.status.hidden = true; return; }
  els.status.hidden = false;
  els.status.className = 'aoi-status' + (kind ? ' ' + kind : '');
  els.status.textContent = msg;
}

/* Open the draw modal. Resolves to an AOI object
   {name, geom, bbox, area_id?, project_id?, stored} or null if cancelled. */
export async function openDrawModal(opts = {}) {
  opts = opts || {};
  let L;
  try {
    L = await loadLeaflet({ draw: true });
  } catch (e) {
    alert('Could not load the map library (' + ((e && e.message) || e) + ').');
    return null;
  }
  const els = buildModalDom(opts);
  els.name.value = opts.suggestedName || defaultName();
  const projectsP = fillProjects(els);

  const map = L.map(els.map, { worldCopyJump: true }).setView([20, 0], 2);
  L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 19,
    attribution: '&copy; OpenStreetMap contributors',
  }).addTo(map);
  const drawn = L.featureGroup().addTo(map);
  const drawControl = new L.Control.Draw({
    position: 'topleft',
    draw: {
      polygon: { allowIntersection: false, showArea: true },
      rectangle: { showArea: true },
      polyline: false, circle: false, marker: false, circlemarker: false,
    },
    edit: { featureGroup: drawn, remove: true },
  });
  map.addControl(drawControl);
  // The map needs a size recalculation once it is visible in the modal.
  setTimeout(() => map.invalidateSize(), 60);

  let geom = null;
  let reuseAreaId = null;  // set when an existing saved area is selected
  let reuseName = null;
  const resetUse = () => { els.use.textContent = 'Use area'; els.use.classList.remove('is-fallback'); };
  const onDrawn = (layer) => {
    // Drawing a fresh shape supersedes any reused area.
    reuseAreaId = null;
    if (els.saved) els.saved.value = _NEW_AREA;
    resetUse();
    drawn.clearLayers();
    drawn.addLayer(layer);
    geom = layer.toGeoJSON().geometry;
    els.bbox.textContent = 'bbox: ' + fmtBbox(geomBbox(geom));
    els.use.disabled = false;
  };
  map.on(L.Draw.Event.CREATED, (e) => onDrawn(e.layer));
  map.on(L.Draw.Event.DELETED, () => {
    if (!drawn.getLayers().length) { geom = null; reuseAreaId = null; els.bbox.textContent = 'No area drawn yet.'; els.use.disabled = true; }
  });

  // Load the selected project's saved areas; reload when the project changes.
  projectsP.then((projects) => { if (projects && projects.length) loadAreas(els, els.project.value); });
  els.project.addEventListener('change', () => {
    reuseAreaId = null; geom = null; drawn.clearLayers(); resetUse();
    els.bbox.textContent = 'No area drawn yet.'; els.use.disabled = true;
    loadAreas(els, els.project.value);
  });

  // Pick a saved area -> fetch its geometry, show it on the map, reuse it.
  els.saved.addEventListener('change', async () => {
    const id = els.saved.value;
    if (!id || id === _NEW_AREA) {
      reuseAreaId = null; geom = null; drawn.clearLayers(); resetUse();
      els.bbox.textContent = 'No area drawn yet.'; els.use.disabled = true;
      return;
    }
    setStatus(els, 'Loading saved area…');
    els.use.disabled = true;
    try {
      const area = await apiArea(id);
      geom = area.geom;
      reuseAreaId = area.id;
      reuseName = area.name;
      drawn.clearLayers();
      const layer = L.geoJSON(geom);
      layer.eachLayer((l) => drawn.addLayer(l));
      try { map.fitBounds(drawn.getBounds(), { maxZoom: 13, padding: [20, 20] }); } catch (e) {}
      els.bbox.textContent = 'bbox: ' + fmtBbox(area.bbox || geomBbox(geom)) + '  (saved area)';
      els.use.textContent = 'Use saved area';
      els.use.disabled = false;
      setStatus(els, null);
    } catch (e) {
      reuseAreaId = null;
      setStatus(els, 'Couldn’t load that area: ' + ((e && e.message) || e), 'is-error');
      els.use.disabled = !geom;
    }
  });

  return new Promise((resolve) => {
    let settled = false;
    const cleanup = () => { try { map.remove(); } catch (e) {} els.back.remove(); document.removeEventListener('keydown', onKey); };
    const finish = (val) => { if (settled) return; settled = true; cleanup(); resolve(val); };
    const onKey = (e) => { if (e.key === 'Escape') finish(null); };
    document.addEventListener('keydown', onKey);
    els.cancel.addEventListener('click', () => finish(null));
    els.close.addEventListener('click', () => finish(null));
    els.back.addEventListener('click', (e) => { if (e.target === els.back) finish(null); });

    els.use.addEventListener('click', async () => {
      // Reuse a saved area: it already exists in Studio, so no new POST.
      if (reuseAreaId && geom) {
        finish({
          name: reuseName || defaultName(),
          geom,
          bbox: geomBbox(geom),
          area_id: reuseAreaId,
          project_id: (els.project.value || '').trim(),
          stored: true,
        });
        return;
      }
      if (!geom) return;
      const name = (els.name.value || '').trim() || defaultName();
      const bbox = geomBbox(geom);
      const projectId = (els.project.value || '').trim();
      const canStore = BRIDGE.live && projConnected() && projectId;
      if (!canStore) {
        finish({ name, geom, bbox, stored: false });
        return;
      }
      els.use.disabled = true;
      els.use.classList.add('is-busy');  // pink fill -> dark on-pink ring
      const useSpin = document.createElement('span');
      useSpin.className = 'spinner sm on-pink';
      useSpin.style.marginRight = '7px';
      els.use.insertBefore(useSpin, els.use.firstChild);
      const clearBusy = () => { useSpin.remove(); els.use.classList.remove('is-busy'); };
      setStatus(els, 'Storing the area in OlmoEarth Studio…');
      try {
        const area = await apiCreateArea(name, geom, projectId);
        clearBusy();
        finish({
          name: area.name || name,
          geom,
          bbox: area.bbox || bbox,
          area_id: area.id,
          project_id: area.project_id || projectId,
          stored: true,
        });
      } catch (e) {
        clearBusy();
        els.use.disabled = false;
        setStatus(els, 'Couldn’t store the area: ' + ((e && e.message) || e) + '. Use it un-stored?', 'is-error');
        els.use.textContent = 'Use without storing';
        els.use.classList.add('is-fallback');
        els.use.onclick = () => finish({ name, geom, bbox, stored: false });
      }
    });
  });
}

/* Open the draw modal and, on a drawn area, attach it to the composer so the
   next brief carries it. Shared by the composer button and the agent-loop
   request-AOI prompt. */
export async function drawAndAttach(opts = {}) {
  const aoi = await openDrawModal(opts);
  if (aoi) addAoiAttachment(aoi);
  return aoi;
}

/* Wire the composer "Draw AOI" button. */
export function wireAoi() {
  const btn = document.getElementById('aoiBtn');
  if (btn) btn.addEventListener('click', () => drawAndAttach());
}
