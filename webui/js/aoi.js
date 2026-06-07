/* Interactive AOI draw widget. Opens a map (Leaflet + Leaflet.draw, OSM
   basemap) in a modal where the user draws a rectangle or polygon; the drawn
   GeoJSON is optionally stored as an OlmoEarth Studio area (POST /api/areas)
   and attached to the next brief so the agent can feed it to AOI-needing
   skills. Leaflet ships UMD, so it is loaded lazily via <script> tags (like
   attach.js lazy-loads pdf.js) - no build step. */

import { BRIDGE } from './store.js';
import { escapeHtml } from './util.js';
import { apiCreateArea, apiProjects, projConnected } from './api.js';
import { addAoiAttachment } from './attach.js';

/* Leaflet 1.7.1 + Leaflet.draw 1.0.4: a known-good pair (Leaflet 1.8+ renamed
   L.Polyline._flat, which Leaflet.draw 1.0.4 still calls). */
const LEAFLET = 'https://cdn.jsdelivr.net/npm/leaflet@1.7.1/dist';
const LEAFLET_DRAW = 'https://cdn.jsdelivr.net/npm/leaflet-draw@1.0.4/dist';

let _leafletPromise = null;

function _loadCss(href) {
  if (document.querySelector(`link[href="${href}"]`)) return;
  const link = document.createElement('link');
  link.rel = 'stylesheet';
  link.href = href;
  document.head.appendChild(link);
}

function _loadScript(src) {
  return new Promise((resolve, reject) => {
    const existing = document.querySelector(`script[data-src="${src}"]`);
    if (existing) { existing.addEventListener('load', resolve); existing.addEventListener('error', reject); return; }
    const s = document.createElement('script');
    s.src = src; s.dataset.src = src; s.async = false;
    s.addEventListener('load', resolve);
    s.addEventListener('error', () => reject(new Error('failed to load ' + src)));
    document.head.appendChild(s);
  });
}

/* Load Leaflet + Leaflet.draw once; resolve to window.L. */
function loadLeaflet() {
  if (_leafletPromise) return _leafletPromise;
  _leafletPromise = (async () => {
    _loadCss(`${LEAFLET}/leaflet.css`);
    _loadCss(`${LEAFLET_DRAW}/leaflet.draw.css`);
    await _loadScript(`${LEAFLET}/leaflet.js`);
    await _loadScript(`${LEAFLET_DRAW}/leaflet.draw.js`);
    if (!window.L) throw new Error('Leaflet did not load');
    return window.L;
  })().catch((e) => { _leafletPromise = null; throw e; });
  return _leafletPromise;
}

/* Bounding box [minLon, minLat, maxLon, maxLat] from a GeoJSON geometry. */
function geomBbox(geom) {
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  const walk = (c) => {
    if (typeof c[0] === 'number' && typeof c[1] === 'number') {
      minX = Math.min(minX, c[0]); maxX = Math.max(maxX, c[0]);
      minY = Math.min(minY, c[1]); maxY = Math.max(maxY, c[1]);
    } else if (Array.isArray(c)) { c.forEach(walk); }
  };
  walk(geom.coordinates || []);
  return [minX, minY, maxX, maxY];
}

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
    els.use.textContent = 'Use area';
    return [];
  }
  els.project.innerHTML = '<option value="">Loading projects…</option>';
  try {
    const projects = await apiProjects();
    if (!projects.length) {
      els.project.innerHTML = '<option value="">No projects - area used un-stored</option>';
      els.projField.classList.add('is-empty');
      return [];
    }
    els.project.innerHTML = projects
      .map((p) => `<option value="${escapeHtml(p.id)}">${escapeHtml(p.name || p.id)}</option>`)
      .join('');
    return projects;
  } catch (e) {
    els.project.innerHTML = '<option value="">Couldn’t load projects</option>';
    return [];
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
    L = await loadLeaflet();
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
  const onDrawn = (layer) => {
    drawn.clearLayers();
    drawn.addLayer(layer);
    const gj = layer.toGeoJSON();
    geom = gj.geometry;
    const b = geomBbox(geom);
    els.bbox.textContent = 'bbox: ' + fmtBbox(b);
    els.use.disabled = false;
  };
  map.on(L.Draw.Event.CREATED, (e) => onDrawn(e.layer));
  map.on(L.Draw.Event.DELETED, () => {
    if (!drawn.getLayers().length) { geom = null; els.bbox.textContent = 'No area drawn yet.'; els.use.disabled = true; }
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
      setStatus(els, 'Storing the area in OlmoEarth Studio…');
      try {
        const area = await apiCreateArea(name, geom, projectId);
        finish({
          name: area.name || name,
          geom,
          bbox: area.bbox || bbox,
          area_id: area.id,
          project_id: area.project_id || projectId,
          stored: true,
        });
      } catch (e) {
        els.use.disabled = false;
        setStatus(els, 'Couldn’t store the area: ' + ((e && e.message) || e) + '. Use it un-stored?', 'is-error');
        els.use.textContent = 'Use without storing';
        els.use.classList.add('is-fallback');
        els.use.onclick = () => finish({ name, geom, bbox, stored: false });
      }
    });
    void projectsP;
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
