/* In-chat result visuals. Given a tool result, render the most useful inline
   visual: the result's raster picture(s) on a Leaflet map fit to the raster's
   extent (authenticated via the bridge /api/tile proxy), or an SVG trajectory
   chart for change detection. Two-or-more rasters render side by side to
   compare. Pure SVG charts (no deps); Leaflet is lazy-loaded (shared with the
   AOI widget via js/leaflet.js). Renders nothing for results with no visual. */

import { escapeHtml } from './util.js';
import { studioKey, apiResultExtent, apiPixelValue } from './api.js';
import { loadLeaflet, osmLayer } from './leaflet.js';

/* Pearson correlation of paired samples (null if undefined). */
function pearson(xs, ys) {
  const n = xs.length;
  if (n < 2) return null;
  const mx = xs.reduce((a, b) => a + b, 0) / n;
  const my = ys.reduce((a, b) => a + b, 0) / n;
  let sxx = 0, syy = 0, sxy = 0;
  for (let i = 0; i < n; i++) { const dx = xs[i] - mx, dy = ys[i] - my; sxx += dx * dx; syy += dy * dy; sxy += dx * dy; }
  if (sxx <= 0 || syy <= 0) return null;
  return sxy / Math.sqrt(sxx * syy);
}

/* Run `worker` over `items` with at most `n` concurrent. */
async function pool(items, n, worker) {
  let i = 0;
  const runners = Array.from({ length: Math.min(n, items.length) }, async () => {
    while (i < items.length) { const idx = i++; await worker(items[idx], idx); }
  });
  await Promise.all(runners);
}

/* Intersection of two [minLon,minLat,maxLon,maxLat] boxes, or null. */
function intersectBbox(a, b) {
  if (!a || !b) return null;
  const x0 = Math.max(a[0], b[0]), y0 = Math.max(a[1], b[1]);
  const x1 = Math.min(a[2], b[2]), y1 = Math.min(a[3], b[3]);
  return x1 > x0 && y1 > y0 ? [x0, y0, x1, y1] : null;
}

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

  // Exactly two rasters with ids -> offer a quantitative difference scan.
  if (entries.length === 2 && entries[0].resultId && entries[1].resultId) {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'viz-diff-btn';
    btn.textContent = 'Scan difference map';
    container.appendChild(btn);
    const out = document.createElement('div');
    out.className = 'viz-diff';
    container.appendChild(out);
    btn.addEventListener('click', () => {
      btn.disabled = true;
      btn.textContent = 'Scanning difference...';
      renderDiffScan(out, entries[0], entries[1])
        .then(() => { btn.textContent = 'Difference map below'; })
        .catch(() => { btn.disabled = false; btn.textContent = 'Scan difference map'; });
    });
  }
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

/* Diff color: blue where A>B, red where B>A; opacity scales with magnitude. */
function diffStyle(d, scaleMax) {
  const t = Math.max(-1, Math.min(1, d / (scaleMax || 1)));
  return { color: t >= 0 ? '#f0529c' : '#37a0ff', fillColor: t >= 0 ? '#f0529c' : '#37a0ff', weight: 0, fillOpacity: 0.15 + 0.7 * Math.abs(t) };
}

/* Progressive difference scan of two result rasters: sample both on a grid
   over their shared extent and paint each cell by (B - A) as it resolves, so
   the user watches the difference map build up, then see the final stats. */
export async function renderDiffScan(container, a, b, opts = {}) {
  // Pointwise pixel-value through the proxy is slow, so default to a modest
  // grid; the scan is progressive, so the map fills in as it goes.
  const n = Math.max(4, Math.min(14, opts.grid || 7));
  let L;
  try { L = await loadLeaflet(); } catch (e) { container.textContent = 'map unavailable'; return; }
  const status = document.createElement('div');
  status.className = 'viz-cap';
  status.textContent = 'Locating the shared extent...';
  container.appendChild(status);

  let bbox = null;
  try {
    const [ea, eb] = await Promise.all([apiResultExtent(a.resultId), apiResultExtent(b.resultId)]);
    bbox = intersectBbox(ea.bbox, eb.bbox);
  } catch (e) { /* fall through */ }
  if (!bbox) { status.textContent = 'Could not determine the two rasters’ shared extent.'; return; }

  const el = document.createElement('div');
  el.className = 'viz-map';
  container.insertBefore(el, status);
  const map = L.map(el, { worldCopyJump: true, attributionControl: false });
  osmLayer(L).addTo(map);
  map.fitBounds([[bbox[1], bbox[0]], [bbox[3], bbox[2]]], { padding: [10, 10], maxZoom: 13 });
  setTimeout(() => map.invalidateSize(), 60);

  const [minx, miny, maxx, maxy] = bbox;
  const dx = (maxx - minx) / n, dy = (maxy - miny) / n;
  const cells = [];
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
    cells.push({ lon: minx + dx * (i + 0.5), lat: miny + dy * (j + 0.5),
      bounds: [[miny + dy * j, minx + dx * i], [miny + dy * (j + 1), minx + dx * (i + 1)]] });
  }
  const total = cells.length;
  let done = 0, scaleMax = 0.05;
  const xs = [], ys = [], absd = [];
  const recolor = () => cells.forEach((c) => { if (c.rect && c.diff != null) c.rect.setStyle(diffStyle(c.diff, scaleMax)); });

  await pool(cells, 8, async (c) => {
    let va = null, vb = null;
    try { const r = await apiPixelValue(a.resultId, c.lon, c.lat, opts.property); va = r.value; } catch (e) {}
    try { const r = await apiPixelValue(b.resultId, c.lon, c.lat, opts.property); vb = r.value; } catch (e) {}
    done++;
    if (typeof va === 'number' && typeof vb === 'number') {
      c.diff = vb - va; xs.push(va); ys.push(vb); absd.push(Math.abs(c.diff));
      if (Math.abs(c.diff) > scaleMax) { scaleMax = Math.abs(c.diff); recolor(); }
      c.rect = L.rectangle(c.bounds, diffStyle(c.diff, scaleMax)).addTo(map);
    }
    status.textContent = 'Scanning the difference... ' + done + '/' + total + ' cells';
  });

  const m = absd.length;
  if (!m) { status.textContent = 'No overlapping valid pixels to difference.'; return; }
  const meanAbs = absd.reduce((p, q) => p + q, 0) / m;
  const within = absd.filter((d) => d <= (opts.tolerance || 0.1)).length / m;
  const r = pearson(xs, ys);
  status.innerHTML =
    'Difference map (B - A) over ' + m + ' sampled cells · mean |diff| <strong>' + meanAbs.toFixed(3) +
    '</strong> · corr <strong>' + (r == null ? 'n/a' : r.toFixed(3)) + '</strong> · ' +
    (within * 100).toFixed(0) + '% agree (±' + (opts.tolerance || 0.1) + '). ' +
    '<span class="viz-legend"><span class="viz-sw" style="background:#37a0ff"></span>A higher' +
    '<span class="viz-sw" style="background:#f0529c"></span>B higher</span>';
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
