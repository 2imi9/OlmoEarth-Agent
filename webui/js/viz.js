/* In-chat result visuals. Given a tool result, render the most useful inline
   visual: a Leaflet map of the result's raster tiles (authenticated via the
   bridge /api/tile proxy), or an SVG trajectory chart for change detection.
   Pure SVG charts (no deps); Leaflet is lazy-loaded (shared with the AOI
   widget via js/leaflet.js). Renders nothing for results with no visual. */

import { escapeHtml } from './util.js';
import { studioKey } from './api.js';
import { loadLeaflet, osmLayer } from './leaflet.js';

/* Collect XYZ tile templates from a tool result, however it nests them. */
function tileTemplates(inner) {
  const out = [];
  const push = (t) => { if (t && typeof t === 'string' && /\{z\}/.test(t)) out.push(t); };
  if (!inner || typeof inner !== 'object') return out;
  push(inner.tile_url);
  (inner.tile_urls || []).forEach(push);
  push(inner.xyz_url);  // qgis-bridge
  (inner.results || []).forEach((r) => { push(r && r.tile_url); (r && r.tile_urls || []).forEach(push); });
  return out;
}

/* A Leaflet TileLayer that fetches each tile through the bridge proxy with the
   Studio key in a header (img requests can't carry one), so auth-gated Studio
   rasters render. Missing tiles (non-200) resolve to a blank image. */
function authTileLayer(L, template) {
  const Layer = L.TileLayer.extend({
    createTile(coords, done) {
      const img = document.createElement('img');
      img.alt = '';
      const url = '/api/tile/' + coords.z + '/' + coords.x + '/' + coords.y +
        '?src=' + encodeURIComponent(template);
      fetch(url, { headers: { 'X-Olmoearth-Key': studioKey() } })
        .then((r) => (r.ok ? r.blob() : Promise.reject(r.status)))
        .then((b) => { img.src = URL.createObjectURL(b); done(null, img); })
        .catch(() => done(null, img));  // missing/forbidden tile -> blank
      return img;
    },
  });
  return new Layer('', { opacity: 0.8, maxZoom: 19 });
}

function shortLabel(t, i) {
  const m = /property_name=([^&]+)/.exec(t);
  if (m) return decodeURIComponent(m[1]);
  const p = /prediction-results\/([0-9a-f]{6,8})/i.exec(t);
  return p ? 'result ' + p[1] : 'layer ' + (i + 1);
}

/* Render a map of one-or-more result raster layers. Multiple layers get a
   toggle (radio) so two rasters can be compared by switching. */
async function renderResultMap(container, templates, opts = {}) {
  let L;
  try { L = await loadLeaflet(); } catch (e) { container.textContent = 'map unavailable'; return; }
  const el = document.createElement('div');
  el.className = 'viz-map';
  container.appendChild(el);
  const map = L.map(el, { worldCopyJump: true, attributionControl: false }).setView(
    opts.center || [20, 0], opts.zoom || 2,
  );
  osmLayer(L).addTo(map);
  const overlays = {};
  templates.forEach((t, i) => {
    const layer = authTileLayer(L, t);
    overlays[shortLabel(t, i)] = layer;
    if (i === 0) layer.addTo(map);
  });
  if (templates.length > 1) {
    L.control.layers(overlays, null, { collapsed: false }).addTo(map);
  }
  setTimeout(() => map.invalidateSize(), 60);
  const hint = document.createElement('div');
  hint.className = 'viz-cap';
  hint.textContent = templates.length > 1
    ? `${templates.length} raster layers - use the toggle to compare; pan/zoom to your area of interest.`
    : 'Result raster on an OpenStreetMap basemap; pan/zoom to your area of interest.';
  container.appendChild(hint);
}

/* A compact SVG line chart for a change-detection trajectory. */
function renderTrajectory(container, dates, values, trend) {
  const W = 460, H = 150, pad = 28;
  const xs = values.map((_, i) => pad + (i * (W - 2 * pad)) / Math.max(1, values.length - 1));
  const lo = Math.min(...values), hi = Math.max(...values);
  const span = hi - lo || 1;
  const y = (v) => H - pad - ((v - lo) / span) * (H - 2 * pad);
  const pts = values.map((v, i) => `${xs[i].toFixed(1)},${y(v).toFixed(1)}`).join(' ');
  const dots = values.map((v, i) =>
    `<circle cx="${xs[i].toFixed(1)}" cy="${y(v).toFixed(1)}" r="3" class="viz-dot"/>`).join('');
  const xlab = dates.map((d, i) =>
    `<text x="${xs[i].toFixed(1)}" y="${H - 8}" class="viz-axt" text-anchor="middle">${escapeHtml(String(d).slice(0, 10))}</text>`).join('');
  const svg =
    `<svg viewBox="0 0 ${W} ${H}" class="viz-svg" role="img" aria-label="trajectory">` +
      `<text x="${pad}" y="14" class="viz-axt">${hi.toFixed(3)}</text>` +
      `<text x="${pad}" y="${H - pad}" class="viz-axt">${lo.toFixed(3)}</text>` +
      `<polyline points="${pts}" class="viz-line" fill="none"/>` + dots + xlab +
    `</svg>`;
  const wrap = document.createElement('div');
  wrap.className = 'viz-chart';
  wrap.innerHTML = svg +
    `<div class="viz-cap">Trajectory over ${values.length} dates${trend ? ` - trend: <strong>${escapeHtml(trend)}</strong>` : ''}.</div>`;
  container.appendChild(wrap);
}

/* Render the best inline visual for a tool-result event into `container`.
   Returns true if it rendered something. */
export function renderResultViz(container, ev) {
  try {
    if (!ev || !ev.ok) return false;
    const inner = ev.result && ev.result.result;
    if (!inner || typeof inner !== 'object') return false;

    const templates = tileTemplates(inner);
    if (templates.length) { void renderResultMap(container, templates); return true; }

    if (Array.isArray(inner.dates) && Array.isArray(inner.values) && inner.values.length >= 2) {
      renderTrajectory(container, inner.dates, inner.values, inner.trend);
      return true;
    }
    return false;
  } catch (e) {
    return false;
  }
}
