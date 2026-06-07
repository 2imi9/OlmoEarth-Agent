/* In-chat result visuals. Given a tool result, render the most useful inline
   visual: the result's raster picture(s) on a Leaflet map fit to the raster's
   extent (authenticated via the bridge /api/tile proxy), or an SVG trajectory
   chart for change detection. Two-or-more rasters render side by side to
   compare. Pure SVG charts (no deps); Leaflet is lazy-loaded (shared with the
   AOI widget via js/leaflet.js). Renders nothing for results with no visual. */

import { escapeHtml } from './util.js';
import { studioKey, apiResultExtent } from './api.js';
import { loadLeaflet, osmLayer } from './leaflet.js';

/* Collect {template, resultId} raster entries from a tool result, however it
   nests them. resultId (when present) lets us fetch the raster's extent so the
   map can fit to it. */
function tileEntries(inner) {
  const out = [];
  const seen = new Set();
  const add = (template, resultId) => {
    if (!template || typeof template !== 'string' || !/\{z\}/.test(template)) return;
    if (seen.has(template)) return;
    seen.add(template);
    out.push({ template, resultId: resultId || null });
  };
  const fromObj = (o) => {
    if (!o || typeof o !== 'object') return;
    const id = o.result_id || o.id || null;
    add(o.tile_url, id);
    (o.tile_urls || []).forEach((t) => add(t, id));
    add(o.xyz_url, id);  // qgis-bridge
  };
  if (!inner || typeof inner !== 'object') return out;
  fromObj(inner);
  (inner.results || []).forEach(fromObj);
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
  return new Layer('', { opacity: 0.85, maxZoom: 19 });
}

function labelFor(entry, i) {
  const m = /property_name=([^&]+)/.exec(entry.template);
  if (m) return decodeURIComponent(m[1]);
  return 'layer ' + (i + 1);
}

/* Fit a Leaflet map to a [minLon, minLat, maxLon, maxLat] bbox. */
function fitBbox(map, b) {
  if (b && b.length === 4 && b.every((n) => Number.isFinite(n))) {
    map.fitBounds([[b[1], b[0]], [b[3], b[2]]], { padding: [12, 12], maxZoom: 13 });
  }
}

/* Render the raster picture(s). Maps are created immediately (so they show
   without waiting on the slow extent lookup) and then snap to each raster's
   extent as it resolves; 2+ rasters render side by side to compare. The tile
   template already carries the colormap, so only the bbox is fetched. */
async function renderResultMap(container, entries) {
  let L;
  try { L = await loadLeaflet(); } catch (e) { container.textContent = 'map unavailable'; return; }

  const maps = document.createElement('div');
  maps.className = 'viz-maps' + (entries.length > 1 ? ' is-multi' : '');
  container.appendChild(maps);
  entries.forEach((entry, i) => {
    const cell = document.createElement('div');
    cell.className = 'viz-map-cell';
    if (entries.length > 1) {
      const cap = document.createElement('div');
      cap.className = 'viz-map-label';
      cap.textContent = labelFor(entry, i);
      cell.appendChild(cap);
    }
    const el = document.createElement('div');
    el.className = 'viz-map';
    cell.appendChild(el);
    maps.appendChild(cell);
    const map = L.map(el, { worldCopyJump: true, attributionControl: false }).setView([20, 0], 2);
    osmLayer(L).addTo(map);
    authTileLayer(L, entry.template).addTo(map);
    setTimeout(() => map.invalidateSize(), 60);
    // Snap to the raster's extent once it resolves (don't block the map on it).
    if (entry.resultId) {
      apiResultExtent(entry.resultId)
        .then((ext) => fitBbox(map, ext.bbox))
        .catch(() => { /* extent unknown -> stay at world view */ });
    }
  });

  const hint = document.createElement('div');
  hint.className = 'viz-cap';
  hint.textContent = entries.length > 1
    ? `${entries.length} result rasters, side by side - fit to each raster's extent over an OpenStreetMap basemap.`
    : 'Result raster, fit to its extent over an OpenStreetMap basemap.';
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

    const entries = tileEntries(inner);
    if (entries.length) { void renderResultMap(container, entries); return true; }

    if (Array.isArray(inner.dates) && Array.isArray(inner.values) && inner.values.length >= 2) {
      renderTrajectory(container, inner.dates, inner.values, inner.trend);
      return true;
    }
    return false;
  } catch (e) {
    return false;
  }
}
