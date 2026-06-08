/* In-chat result visuals. Given a tool result, render the most useful inline
   visual: the result's raster picture(s) on a Leaflet map fit to the raster's
   extent (authenticated via the bridge /api/tile proxy), or an SVG trajectory
   chart for change detection. Two-or-more rasters render side by side to
   compare. Pure SVG charts (no deps); Leaflet is lazy-loaded (shared with the
   AOI widget via js/leaflet.js). Renders nothing for results with no visual. */

import { escapeHtml, shortId } from './util.js';
import { studioKey, apiResultExtent, apiPixelValue } from './api.js';
import { loadLeaflet, osmLayer } from './leaflet.js';
import { downloadBar, downloadJson, downloadText, comparisonCsv } from './download.js';

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
    const skel = document.createElement('div');  // shimmer until the result tiles paint
    skel.className = 'skeleton skel-tile';
    el.appendChild(skel);
    cell.appendChild(el);
    maps.appendChild(cell);
    const map = L.map(el, { worldCopyJump: true, attributionControl: false }).setView([20, 0], 2);
    osmLayer(L).addTo(map);
    const tiles = authTileLayer(L, entry.template);
    tiles.on('load', () => skel.remove());
    tiles.addTo(map);
    setTimeout(() => map.invalidateSize(), 60);
    setTimeout(() => skel.remove(), 2500);  // fallback if 'load' never fires (cached/sparse)
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
      btn.classList.add('is-busy');  // blocks re-click + dims; spinner reads on the mint fill
      btn.innerHTML = '<span class="spinner sm on-pink"></span>Scanning difference…';
      renderDiffScan(out, entries[0], entries[1])
        .then(() => { btn.classList.remove('is-busy'); btn.textContent = 'Difference map below'; })
        .catch(() => { btn.classList.remove('is-busy'); btn.textContent = 'Scan difference map'; });
    });
  }
}

/* A compact SVG line chart for a change-detection trajectory. Axis labels live
   in dedicated gutters (a left margin for the y-range, a bottom margin for the
   dates) so they never overlap the plotted line/points; first + last dates are
   edge-anchored so they don't clip, and dates are thinned when there are many. */
function renderTrajectory(container, dates, values, trend) {
  const W = 480, H = 170;
  const padL = 46, padR = 16, padT = 16, padB = 30;  // gutters: y-labels left, x-dates bottom
  const n = values.length;
  const plotW = W - padL - padR, plotH = H - padT - padB, y0 = H - padB;
  const xs = values.map((_, i) => padL + (i * plotW) / Math.max(1, n - 1));
  const lo = Math.min(...values), hi = Math.max(...values);
  const span = hi - lo || 1;
  const y = (v) => y0 - ((v - lo) / span) * plotH;
  const pts = values.map((v, i) => `${xs[i].toFixed(1)},${y(v).toFixed(1)}`).join(' ');
  const dots = values.map((v, i) =>
    `<circle cx="${xs[i].toFixed(1)}" cy="${y(v).toFixed(1)}" r="3" class="viz-dot"/>`).join('');
  // Date labels: always the first + last (edge-anchored), plus evenly-spaced
  // middles with enough gap that none overlap — including a middle that would
  // otherwise land right next to the forced last label.
  const step = Math.max(1, Math.ceil((90 * (n - 1)) / plotW));
  const showIdx = new Set([0, n - 1]);
  for (let i = step; i < n - 1; i += step) if (n - 1 - i >= step) showIdx.add(i);
  const xlab = dates.map((d, i) => {
    if (!showIdx.has(i)) return '';
    const anchor = i === 0 ? 'start' : (i === n - 1 ? 'end' : 'middle');
    return `<text x="${xs[i].toFixed(1)}" y="${H - 10}" class="viz-axt" text-anchor="${anchor}">${escapeHtml(String(d).slice(0, 10))}</text>`;
  }).join('');
  const svg =
    `<svg viewBox="0 0 ${W} ${H}" class="viz-svg" role="img" aria-label="trajectory">` +
      `<line x1="${padL}" y1="${padT}" x2="${padL}" y2="${y0}" class="viz-axis"/>` +
      `<line x1="${padL}" y1="${y0}" x2="${W - padR}" y2="${y0}" class="viz-axis"/>` +
      `<text x="${padL - 7}" y="${padT + 3}" class="viz-axt" text-anchor="end">${hi.toFixed(3)}</text>` +
      `<text x="${padL - 7}" y="${y0}" class="viz-axt" text-anchor="end">${lo.toFixed(3)}</text>` +
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

/* A small "which result is A / B" label so the B - A direction is unambiguous. */
function abLabel(idA, idB) {
  const el = document.createElement('div');
  el.className = 'viz-ab';
  el.innerHTML =
    '<span class="viz-ab-a">A</span> result ' + escapeHtml(shortId(idA || '?')) +
    ' &nbsp;<span class="viz-ab-b">B</span> result ' + escapeHtml(shortId(idB || '?')) +
    ' &nbsp;<span class="viz-ab-d">difference = B - A</span>';
  return el;
}

/* Overlay viewer: A, B, and the difference on ONE map, each with an opacity
   slider so you can fade between them. A/B raster tiles come from each
   result's extent (tile template + bbox); the difference is the saved grid. */
export async function renderOverlayViewer(record) {
  const back = document.createElement('div');
  back.className = 'aoi-modal';
  back.innerHTML =
    '<div class="aoi-card" role="dialog" aria-modal="true" aria-label="A / B / difference overlay">' +
      '<div class="aoi-head"><div class="aoi-title">Overlay: A, B and the difference' +
        '<span class="aoi-purpose">slide to fade each layer</span></div>' +
        '<button class="aoi-x" type="button" aria-label="Close">&times;</button></div>' +
      '<div class="viz-map ov-map"></div>' +
      '<div class="ov-controls"></div>' +
    '</div>';
  document.body.appendChild(back);
  let map = null;
  const close = () => { try { map && map.remove(); } catch (e) {} back.remove(); document.removeEventListener('keydown', onKey); };
  const onKey = (e) => { if (e.key === 'Escape') close(); };
  document.addEventListener('keydown', onKey);
  back.querySelector('.aoi-x').addEventListener('click', close);
  back.addEventListener('click', (e) => { if (e.target === back) close(); });

  let L;
  try { L = await loadLeaflet(); } catch (e) { back.querySelector('.ov-map').textContent = 'map unavailable'; return; }
  const el = back.querySelector('.ov-map');
  map = L.map(el, { worldCopyJump: true, attributionControl: false });
  osmLayer(L).addTo(map);
  map.createPane('ovdiff');
  map.getPane('ovdiff').style.zIndex = '450';

  let bbox = record.bbox || null;
  let tA = null, tB = null;
  try { const ea = await apiResultExtent(record.resultIdA); tA = ea.tile_url; bbox = bbox || ea.bbox; } catch (e) {}
  try { const eb = await apiResultExtent(record.resultIdB); tB = eb.tile_url; } catch (e) {}
  const layerA = tA ? authTileLayer(L, tA) : null;
  const layerB = tB ? authTileLayer(L, tB) : null;
  if (layerA) { layerA.setOpacity(1); layerA.addTo(map); }
  if (layerB) { layerB.setOpacity(0); layerB.addTo(map); }

  const cells = record.cells || [];
  const [dx, dy] = record.cellSize || [0.01, 0.01];
  const scaleMax = Math.max(0.05, ...cells.map((c) => Math.abs(c.diff)));
  cells.forEach((c) => {
    const b = [[c.lat - dy / 2, c.lon - dx / 2], [c.lat + dy / 2, c.lon + dx / 2]];
    L.rectangle(b, { ...diffStyle(c.diff, scaleMax), pane: 'ovdiff' }).addTo(map);
  });
  map.getPane('ovdiff').style.opacity = '0.6';
  if (bbox) map.fitBounds([[bbox[1], bbox[0]], [bbox[3], bbox[2]]], { padding: [10, 10], maxZoom: 13 });
  setTimeout(() => map.invalidateSize(), 60);

  const controls = back.querySelector('.ov-controls');
  const slider = (labelHtml, val, disabled, on) => {
    const row = document.createElement('label');
    row.className = 'ov-row';
    row.innerHTML = '<span class="ov-lbl">' + labelHtml + '</span>';
    const inp = document.createElement('input');
    inp.type = 'range'; inp.min = '0'; inp.max = '100'; inp.value = String(val);
    inp.disabled = disabled;
    inp.addEventListener('input', () => on(Number(inp.value) / 100));
    row.appendChild(inp);
    controls.appendChild(row);
  };
  slider('<span class="viz-ab-a">A</span> ' + escapeHtml(shortId(record.resultIdA || '?')), 100, !layerA, (v) => layerA && layerA.setOpacity(v));
  slider('<span class="viz-ab-b">B</span> ' + escapeHtml(shortId(record.resultIdB || '?')), 0, !layerB, (v) => layerB && layerB.setOpacity(v));
  slider('<span class="viz-ab-d">Difference (B - A)</span>', 60, cells.length === 0, (v) => { map.getPane('ovdiff').style.opacity = String(v); });
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
  container.insertBefore(abLabel(a.resultId, b.resultId), el);
  const map = L.map(el, { worldCopyJump: true, attributionControl: false });
  osmLayer(L).addTo(map);
  map.fitBounds([[bbox[1], bbox[0]], [bbox[3], bbox[2]]], { padding: [10, 10], maxZoom: 13 });
  setTimeout(() => map.invalidateSize(), 60);

  // Soft radar sweep + per-cell ignite while the diff cells paint in (CSS-driven;
  // decoupled from the async cell arrival). The beam overlay sits above Leaflet's
  // overlay pane; JS only toggles .is-scanning here and .viz-cell--in per cell.
  const scan = document.createElement('div');
  scan.className = 'viz-scan';
  el.appendChild(scan);
  el.classList.add('is-scanning');

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

  // Swap the "Locating…" caption for a live determinate progress pill ("Scanning 8/25").
  status.style.display = 'none';
  const prog = document.createElement('div');
  prog.className = 'progress-pill';
  prog.innerHTML =
    '<span class="spinner sm"></span>' +
    '<span class="pp-label">Scanning</span>' +
    '<span class="pp-count">0/' + total + '</span>' +
    '<span class="pp-track"><span class="pp-fill"></span></span>';
  const ppCount = prog.querySelector('.pp-count');
  container.insertBefore(prog, status);
  const endProgress = () => {
    prog.remove(); status.style.display = '';
    setTimeout(() => el.classList.remove('is-scanning'), 600);  // let the sweep finish a pass
  };

  await pool(cells, 8, async (c) => {
    let va = null, vb = null;
    try { const r = await apiPixelValue(a.resultId, c.lon, c.lat, opts.property); va = r.value; } catch (e) {}
    try { const r = await apiPixelValue(b.resultId, c.lon, c.lat, opts.property); vb = r.value; } catch (e) {}
    done++;
    if (typeof va === 'number' && typeof vb === 'number') {
      c.diff = vb - va; xs.push(va); ys.push(vb); absd.push(Math.abs(c.diff));
      if (Math.abs(c.diff) > scaleMax) { scaleMax = Math.abs(c.diff); recolor(); }
      c.rect = L.rectangle(c.bounds, { ...diffStyle(c.diff, scaleMax), className: 'viz-cell' }).addTo(map);
      if (c.rect._path) c.rect._path.classList.add('viz-cell--in');  // ignite the SVG path
    }
    ppCount.textContent = done + '/' + total;
    prog.style.setProperty('--pp', (done / total).toFixed(3));
  });
  endProgress();

  const m = absd.length;
  if (!m) { status.textContent = 'No overlapping valid pixels to difference.'; return null; }
  const tol = opts.tolerance || 0.1;
  const diffs = cells.filter((c) => c.diff != null).map((c) => c.diff);
  const meanAbs = absd.reduce((p, q) => p + q, 0) / m;
  const maxAbs = Math.max(...absd);
  const meanDiff = diffs.reduce((p, q) => p + q, 0) / m;
  const rmse = Math.sqrt(diffs.reduce((p, q) => p + q * q, 0) / m);
  const within = absd.filter((d) => d <= tol).length / m;
  const r = pearson(xs, ys);
  const stats = {
    n_samples: m,
    correlation: r == null ? null : Number(r.toFixed(4)),
    agreement_fraction: Number(within.toFixed(4)),
    mean_abs_diff: Number(meanAbs.toFixed(6)),
    max_abs_diff: Number(maxAbs.toFixed(6)),
    mean_diff_b_minus_a: Number(meanDiff.toFixed(6)),
    rmse_between_models: Number(rmse.toFixed(6)),
    tolerance: tol,
  };
  status.innerHTML =
    'Difference map (B - A) over ' + m + ' sampled cells · mean |diff| <strong>' + meanAbs.toFixed(3) +
    '</strong> · corr <strong>' + (r == null ? 'n/a' : r.toFixed(3)) + '</strong> · ' +
    (within * 100).toFixed(0) + '% agree (±' + tol + '). ' +
    '<span class="viz-legend"><span class="viz-sw" style="background:#37a0ff"></span>A higher' +
    '<span class="viz-sw" style="background:#f0529c"></span>B higher</span>';

  // Captured straight from the live scan (real values), so a saved comparison
  // is never fabricated. Offer to store it in the Comparisons panel.
  const record = {
    resultIdA: a.resultId, resultIdB: b.resultId, property: opts.property || null,
    bbox, cellSize: [dx, dy], tolerance: tol, stats,
    cells: cells.filter((c) => c.diff != null).map((c) => ({
      lon: Number(c.lon.toFixed(6)), lat: Number(c.lat.toFixed(6)), diff: Number(c.diff.toFixed(6)),
    })),
  };
  if (opts.saveable !== false) {
    const save = document.createElement('button');
    save.type = 'button';
    save.className = 'viz-diff-btn';
    save.textContent = 'Save comparison';
    save.addEventListener('click', () => {
      document.dispatchEvent(new CustomEvent('oe:save-comparison', { detail: record }));
      save.textContent = 'Saved to Comparisons';
      save.disabled = true;
    });
    container.appendChild(save);
  }
  container.appendChild(downloadBar([
    { label: 'Open overlay', onClick: () => renderOverlayViewer(record) },
    { label: 'Download JSON', onClick: () => downloadJson('comparison.json', record) },
    { label: 'Download CSV', onClick: () => downloadText('comparison.csv', comparisonCsv(record), 'text/csv') },
  ]));
  return record;
}

/* A compact stat card from a compare `stats` object (real numbers only). */
function statChips(s, caption) {
  s = s || {};
  const rows = [];
  const add = (k, v) => { if (v !== undefined && v !== null) rows.push([k, v]); };
  add('correlation', s.correlation);
  add('agreement', s.agreement_fraction != null ? (s.agreement_fraction * 100).toFixed(0) + '%' : null);
  add('mean |diff|', s.mean_abs_diff);
  add('RMSE', s.rmse_between_models);
  add('max |diff|', s.max_abs_diff);
  add('samples', s.n_samples);
  const card = document.createElement('div');
  card.className = 'viz-statcard';
  card.innerHTML = rows.map(([k, v]) =>
    `<span class="viz-stat">${escapeHtml(k)}<b>${escapeHtml(String(v))}</b></span>`).join('') +
    (caption ? `<div class="viz-cap">${escapeHtml(caption)}</div>` : '');
  return card;
}

/* A compact stat card for olmoearth_compare_results, then the difference map
   auto-scanned (so a quantitative compare appears with both the numbers and
   the visual, no button needed). */
function renderCompareCard(container, inner) {
  const s = inner.stats || {};
  container.appendChild(statChips(s, 'Model-vs-model agreement (no ground truth). Difference map below.'));
  container.appendChild(abLabel(inner.result_id_a, inner.result_id_b));
  if (inner.result_id_a && inner.result_id_b) {
    const out = document.createElement('div');
    out.className = 'viz-diff';
    container.appendChild(out);
    void renderDiffScan(out, { resultId: inner.result_id_a }, { resultId: inner.result_id_b },
      { property: inner.property_name, tolerance: s.tolerance });
  }
}

/* Re-render a SAVED comparison (stored real stats + grid cells) - no re-fetch,
   no fabrication: it draws exactly the data captured during the live scan. */
export async function renderStoredComparison(container, record) {
  container.appendChild(statChips(record.stats, 'Saved comparison - model-vs-model agreement (no ground truth).'));
  container.appendChild(abLabel(record.resultIdA, record.resultIdB));
  let L;
  try { L = await loadLeaflet(); } catch (e) { return; }
  const el = document.createElement('div');
  el.className = 'viz-map';
  container.appendChild(el);
  const map = L.map(el, { worldCopyJump: true, attributionControl: false });
  osmLayer(L).addTo(map);
  const b = record.bbox;
  if (b) map.fitBounds([[b[1], b[0]], [b[3], b[2]]], { padding: [10, 10], maxZoom: 13 });
  setTimeout(() => map.invalidateSize(), 60);
  const cells = record.cells || [];
  const [dx, dy] = record.cellSize || [0.01, 0.01];
  const scaleMax = Math.max(0.05, ...cells.map((c) => Math.abs(c.diff)));
  cells.forEach((c) => {
    const bounds = [[c.lat - dy / 2, c.lon - dx / 2], [c.lat + dy / 2, c.lon + dx / 2]];
    L.rectangle(bounds, diffStyle(c.diff, scaleMax)).addTo(map);
  });
  const cap = document.createElement('div');
  cap.className = 'viz-cap';
  cap.innerHTML = 'Difference map (B - A) over ' + cells.length + ' cells. ' +
    '<span class="viz-legend"><span class="viz-sw" style="background:#37a0ff"></span>A higher' +
    '<span class="viz-sw" style="background:#f0529c"></span>B higher</span>';
  container.appendChild(cap);
  container.appendChild(downloadBar([
    { label: 'Open overlay', onClick: () => renderOverlayViewer(record) },
    { label: 'Download JSON', onClick: () => downloadJson('comparison.json', record) },
    { label: 'Download CSV', onClick: () => downloadText('comparison.csv', comparisonCsv(record), 'text/csv') },
  ]));
}

/* Horizontal bars for 0..1 metric rows [{label, value, color}]. */
function bars(rows) {
  const wrap = document.createElement('div');
  wrap.className = 'viz-bars';
  wrap.innerHTML = rows.map((r) => {
    const v = Number(r.value);
    const pct = Number.isFinite(v) ? Math.max(0, Math.min(100, v * 100)) : 0;
    return '<div class="viz-bar-row">' +
      `<span class="viz-bar-label" title="${escapeHtml(r.label)}">${escapeHtml(r.label)}</span>` +
      `<span class="viz-bar-track"><span class="viz-bar-fill" style="width:${pct.toFixed(1)}%;background:${r.color || 'var(--mint)'}"></span></span>` +
      `<span class="viz-bar-val">${Number.isFinite(v) ? v.toFixed(3) : 'n/a'}</span>` +
    '</div>';
  }).join('');
  return wrap;
}

/* baseline-compare: grouped bars (model A vs model B) per headline metric. */
function renderComparisonBars(container, inner) {
  const la = (inner.model_a || {}).label || 'a';
  const lb = (inner.model_b || {}).label || 'b';
  const rows = [];
  (inner.comparison || []).forEach((c) => {
    rows.push({ label: `${c.metric} - ${la}`, value: c[la], color: 'var(--mint)' });
    rows.push({ label: `${c.metric} - ${lb}`, value: c[lb], color: 'var(--pink)' });
  });
  container.appendChild(bars(rows));
  const cap = document.createElement('div');
  cap.className = 'viz-cap';
  cap.innerHTML = `0-1 metric scores, ${escapeHtml(la)} vs ${escapeHtml(lb)}` +
    (inner.overall_winner ? ` - overall winner: <strong>${escapeHtml(inner.overall_winner)}</strong>` : '') + '.';
  container.appendChild(cap);
}

/* evaluate: overall metrics + per-class F1 bars. */
function renderMetricsBars(container, inner) {
  const rows = [
    { label: 'accuracy', value: inner.accuracy, color: 'var(--mint)' },
    { label: 'macro F1', value: inner.macro_f1, color: 'var(--mint)' },
    { label: 'mean IoU', value: inner.mean_iou, color: 'var(--mint)' },
  ];
  Object.entries(inner.per_class || {}).slice(0, 10).forEach(([cls, m]) =>
    rows.push({ label: `${cls} F1`, value: (m || {}).f1, color: 'var(--pink)' }));
  container.appendChild(bars(rows));
  const cap = document.createElement('div');
  cap.className = 'viz-cap';
  cap.innerHTML = `Accuracy <strong>${((inner.accuracy || 0) * 100).toFixed(0)}%</strong> over n=${escapeHtml(String(inner.n))}; per-class F1 in pink.`;
  container.appendChild(cap);
}

/* Render the best inline visual for a tool-result event into `container`.
   Returns true if it rendered something. */
export function renderResultViz(container, ev) {
  try {
    if (!ev || !ev.ok) return false;
    const inner = ev.result && ev.result.result;
    if (!inner || typeof inner !== 'object') return false;
    let any = false;

    // Downloadable text artifacts (don't early-return: a qgis result also maps).
    if (typeof inner.sld === 'string' && inner.sld) {
      const name = (inner.layer_name || 'style') + '.sld';
      container.appendChild(downloadBar([{ label: 'Download .sld style',
        onClick: () => downloadText(name, inner.sld, 'application/vnd.ogc.sld+xml') }]));
      any = true;
    }
    if (typeof inner.markdown === 'string' && inner.markdown) {
      container.appendChild(downloadBar([{ label: 'Download report (.md)',
        onClick: () => downloadText('case-narrative.md', inner.markdown, 'text/markdown') }]));
      any = true;
    }

    // Quantitative compare: show the numbers + auto-scan the difference map.
    if (inner.comparable === true && inner.result_id_a && inner.result_id_b) {
      renderCompareCard(container, inner);
      return true;
    }
    // baseline-compare: grouped metric bars.
    if (Array.isArray(inner.comparison) && inner.model_a && inner.model_b) {
      renderComparisonBars(container, inner);
      return true;
    }
    // evaluate: classification metrics bars.
    if (inner.per_class && typeof inner.per_class === 'object' && inner.accuracy != null) {
      renderMetricsBars(container, inner);
      return true;
    }

    const entries = tileEntries(inner);
    if (entries.length) { void renderResultMap(container, entries); return true; }

    if (Array.isArray(inner.dates) && Array.isArray(inner.values) && inner.values.length >= 2) {
      renderTrajectory(container, inner.dates, inner.values, inner.trend);
      return true;
    }
    return any;  // true if a downloadable artifact (sld / markdown) was shown
  } catch (e) {
    return false;
  }
}
