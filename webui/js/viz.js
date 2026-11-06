/* In-chat result visuals. Given a tool result, render the most useful inline
   visual: the result's raster picture(s) on a Leaflet map fit to the raster's
   extent (authenticated via the bridge /api/tile proxy). Two-or-more rasters
   render side by side to compare. Leaflet is lazy-loaded (shared with the
   AOI widget via js/leaflet.js). Renders nothing for results with no visual. */

import { escapeHtml, shortId } from './util.js';
import { studioKey, apiResultExtent, apiPixelValue } from './api.js';
import { loadLeaflet, osmLayer } from './leaflet.js';
import { downloadBar, downloadJson, downloadText, comparisonCsv } from './download.js';
import { getFigure } from './figstore.js';

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

/* One pixel value, RETRYING transient failures. The grid + sample points are
   deterministic, so the only thing that made a re-scan of the same pair give
   different stats was cells dropped on a flaky fetch (a transient HTTP/timeout
   error was swallowed, so that cell wasn't counted). Retrying gives consistent
   coverage -> reproducible stats. A valid response with a non-numeric value is
   genuine nodata, so it's returned as null without retrying (consistently). */
async function pixelAt(resultId, lon, lat, property, tries = 4) {
  for (let t = 0; t < tries; t++) {
    try {
      const r = await apiPixelValue(resultId, lon, lat, property);
      return r && typeof r.value === 'number' ? r.value : null;
    } catch (e) {
      if (t === tries - 1) return null;
      await new Promise((res) => setTimeout(res, 120 * (t + 1)));  // 120/240/360ms backoff
    }
  }
  return null;
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

/* The 5-stop YlOrRd ramp, so a result-raster legend reads low->high. */
const _RAMP = ['#ffffb2', '#fecc5c', '#fd8d3c', '#f03b20', '#bd0026'];

/* Drag/hover the gradient bar to read the mapped score at that position. */
function wireGrad(bar) {
  const vmin = +bar.dataset.vmin, vmax = +bar.dataset.vmax;
  const line = bar.querySelector('.rl-line'), tip = bar.querySelector('.rl-tip');
  const move = (e) => {
    const r = bar.getBoundingClientRect();
    const cx = e.touches ? e.touches[0].clientX : e.clientX;
    const x = Math.max(0, Math.min(r.width, cx - r.left));
    line.style.left = x + 'px';
    tip.style.left = Math.max(14, Math.min(r.width - 14, x)) + 'px';
    tip.textContent = (vmin + (x / r.width) * (vmax - vmin)).toFixed(2);
    bar.classList.add('is-hover');
  };
  bar.addEventListener('pointermove', move);
  bar.addEventListener('pointerdown', move);
  bar.addEventListener('pointerleave', () => bar.classList.remove('is-hover'));
}

/* A value->color legend under a result raster, so a score map is readable for
   analysis. Continuous (a score band) by default; pass legend data with
   kind:'categorical' (entries:[{color,label}]) for a class map. The continuous
   range defaults to 0..1 (OlmoEarth score outputs); a real range can be supplied. */
function renderRasterLegend(cell, property, legend) {
  legend = legend || { kind: 'continuous', vmin: 0, vmax: 1 };
  const el = document.createElement('div');
  el.className = 'viz-rlegend';
  if (legend.kind === 'categorical' && Array.isArray(legend.entries)) {
    el.innerHTML =
      `<div class="rl-prop"><span class="rl-name">${escapeHtml(property)}</span>` +
      '<span class="rl-kind">classes</span></div>' +
      '<div class="rl-cats">' + legend.entries.map((e) =>
        `<span class="rl-cat"><span class="rl-sw" style="background:${escapeHtml(e.color)}"></span>` +
        `${escapeHtml(String(e.label))}</span>`).join('') + '</div>';
  } else {
    const vmin = legend.vmin != null ? legend.vmin : 0;
    const vmax = legend.vmax != null ? legend.vmax : 1;
    const stops = legend.stops || _RAMP.map((c, i) => ({ color: c, value: vmin + (vmax - vmin) * i / (_RAMP.length - 1) }));
    const grad = stops.map((s) => `${s.color} ${((s.value - vmin) / (vmax - vmin) * 100).toFixed(1)}%`).join(', ');
    const fmt = (v) => (Number.isInteger(v) ? v : Number(v).toFixed(2));
    el.innerHTML =
      `<div class="rl-prop"><span class="rl-name">${escapeHtml(property)}</span>` +
      '<span class="rl-kind">score</span></div>' +
      `<div class="rl-grad"><div class="rl-bar" data-vmin="${vmin}" data-vmax="${vmax}" ` +
      `style="background:linear-gradient(90deg,${grad})"><div class="rl-line"></div><div class="rl-tip"></div></div>` +
      `<div class="rl-scale"><span>${fmt(vmin)}</span><span>${fmt((vmin + vmax) / 2)}</span><span>${fmt(vmax)}</span></div></div>`;
    cell.appendChild(el);
    wireGrad(el.querySelector('.rl-bar'));
    return;
  }
  cell.appendChild(el);
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
    // A value->color legend under each raster so the map is readable for analysis.
    if (/property_name=/.test(entry.template)) renderRasterLegend(cell, labelFor(entry, i), entry.legend);
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

/* Diff color: blue where A>B, red where B>A; opacity scales with magnitude. */
function diffStyle(d, scaleMax) {
  const t = Math.max(-1, Math.min(1, d / (scaleMax || 1)));
  return { color: t >= 0 ? '#f0529c' : '#37a0ff', fillColor: t >= 0 ? '#f0529c' : '#37a0ff', weight: 0, fillOpacity: 0.15 + 0.7 * Math.abs(t) };
}

/* Vocabulary for the two comparison kinds. cross_model = two different models
   over the same area (how much they agree); temporal = one model's output at an
   earlier (A) vs a later (B) date (net change over time). The numbers are
   identical; only how we describe them differs. The diff map colors stay fixed
   (blue = negative / A higher / decreased, pink = positive / B higher /
   increased) -- only the legend words change. */
function compareVocab(kind) {
  if (kind === 'temporal') {
    return {
      kind: 'temporal',
      aRole: 'earlier', bRole: 'later',
      diffExpr: 'later - earlier',
      mapTitle: 'Change map (later - earlier)',
      diffLabel: 'Change (later - earlier)',
      diffWord: 'change', agreeWord: 'stable',
      legendLow: 'decreased', legendHigh: 'increased',
      statCaption: 'Same model, two dates - net change over time (later - earlier). Map below.',
      savedCaption: 'Saved comparison - net change over time (later - earlier).',
    };
  }
  return {
    kind: 'cross_model',
    aRole: 'result', bRole: 'result',
    diffExpr: 'B - A',
    mapTitle: 'Difference map (B - A)',
    diffLabel: 'Difference (B - A)',
    diffWord: 'diff', agreeWord: 'agree',
    legendLow: 'A higher', legendHigh: 'B higher',
    statCaption: 'Model-vs-model agreement (no ground truth). Difference map below.',
    savedCaption: 'Saved comparison - model-vs-model agreement (no ground truth).',
  };
}

/* The fixed blue/pink swatch legend, with kind-aware words. */
function legendHtml(kind) {
  const v = compareVocab(kind);
  return '<span class="viz-legend"><span class="viz-sw" style="background:#37a0ff"></span>' + v.legendLow +
    '<span class="viz-sw" style="background:#f0529c"></span>' + v.legendHigh + '</span>';
}

/* A small "which result is A / B" label so the difference direction is
   unambiguous (B - A for models, later - earlier for a temporal compare). */
function abLabel(idA, idB, kind) {
  const v = compareVocab(kind);
  const el = document.createElement('div');
  el.className = 'viz-ab';
  el.innerHTML =
    '<span class="viz-ab-a">A</span> ' + v.aRole + ' ' + escapeHtml(shortId(idA || '?')) +
    ' &nbsp;<span class="viz-ab-b">B</span> ' + v.bRole + ' ' + escapeHtml(shortId(idB || '?')) +
    ' &nbsp;<span class="viz-ab-d">difference = ' + v.diffExpr + '</span>';
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
  slider('<span class="viz-ab-d">' + compareVocab(record.kind).diffLabel + '</span>', 60, cells.length === 0, (v) => { map.getPane('ovdiff').style.opacity = String(v); });
}

/* Progressive difference scan of two result rasters: sample both on a grid
   over their shared extent and paint each cell by (B - A) as it resolves, so
   the user watches the difference map build up, then see the final stats. */
export async function renderDiffScan(container, a, b, opts = {}) {
  // Pointwise pixel-value through the proxy is slow (~grid^2 x2 samples), so
  // default to a small grid; the scan is progressive, so the map fills in as it
  // goes. 5x5 = 25 cells keeps a fresh scan responsive; raise opts.grid for detail.
  const n = Math.max(4, Math.min(14, opts.grid || 5));
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
  // The compare card already prints the A/B legend above the scan, so let the
  // caller suppress this one to avoid a duplicate line.
  if (!opts.noAbLabel) container.insertBefore(abLabel(a.resultId, b.resultId, opts.kind), el);
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
    const va = await pixelAt(a.resultId, c.lon, c.lat, opts.property);
    const vb = await pixelAt(b.resultId, c.lon, c.lat, opts.property);
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
  // Cells where either map had no value: off-raster, or Studio's no-data
  // sentinel, which the bridge (serve.py api_pixel_value) returns as null.
  const skipped = total - m;
  if (!m) { status.textContent = 'No overlapping valid pixels to difference (' + skipped + ' no-data cells skipped).'; return null; }
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
  const v = compareVocab(opts.kind);
  status.innerHTML =
    v.mapTitle + ' over ' + m + ' sampled cells · mean |' + v.diffWord + '| <strong>' + meanAbs.toFixed(3) +
    '</strong>' + (v.kind === 'temporal' ? ' · net <strong>' + meanDiff.toFixed(3) + '</strong>' : '') +
    ' · corr <strong>' + (r == null ? 'n/a' : r.toFixed(3)) + '</strong> · ' +
    (within * 100).toFixed(0) + '% ' + v.agreeWord + ' (±' + tol + ')' +
    (skipped ? ' · ' + skipped + ' no-data cells skipped' : '') + '. ' + legendHtml(v.kind);

  // Captured straight from the live scan (real values), so a saved comparison
  // is never fabricated. Offer to store it in the Results panel.
  const record = {
    resultIdA: a.resultId, resultIdB: b.resultId, property: opts.property || null,
    kind: v.kind, bbox, cellSize: [dx, dy], tolerance: tol, stats,
    cells: cells.filter((c) => c.diff != null).map((c) => ({
      lon: Number(c.lon.toFixed(6)), lat: Number(c.lat.toFixed(6)), diff: Number(c.diff.toFixed(6)),
    })),
  };
  if (opts.saveable !== false) {
    const save = document.createElement('button');
    save.type = 'button';
    save.className = 'viz-diff-btn';
    save.textContent = 'Save to Results';
    save.addEventListener('click', () => {
      document.dispatchEvent(new CustomEvent('oe:save-comparison', { detail: record }));
      save.textContent = 'Saved to Results';
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

/* A compact stat card from [label, value] rows; null/undefined values are
   skipped (real numbers only, never a placeholder). */
function chipCard(rows, caption) {
  const card = document.createElement('div');
  card.className = 'viz-statcard';
  card.innerHTML = rows.filter(([, v]) => v !== undefined && v !== null).map(([k, v]) =>
    `<span class="viz-stat">${escapeHtml(k)}<b>${escapeHtml(String(v))}</b></span>`).join('') +
    (caption ? `<div class="viz-cap">${escapeHtml(caption)}</div>` : '');
  return card;
}

const pctOf = (x) => (x != null ? (x * 100).toFixed(0) + '%' : null);

/* A compact stat card from a compare `stats` object (real numbers only). */
function statChips(s, caption) {
  s = s || {};
  return chipCard([
    ['correlation', s.correlation],
    ['agreement', pctOf(s.agreement_fraction)],
    ['mean |diff|', s.mean_abs_diff],
    ['RMSE', s.rmse_between_models],
    ['max |diff|', s.max_abs_diff],
    ['samples', s.n_samples],
  ], caption);
}

/* olmoearth_compare_results in its group / series / ensemble modes: the
   headline numbers the tool returned and its own headline + framing. (A pair
   renders renderCompareCard: numbers plus the live difference scan.) */
function renderCompareSummary(container, inner) {
  const nar = inner.narration || {};
  const cls = inner.value_type === 'classification';
  let rows = [];
  if (inner.mode === 'group') {
    const e = inner.ensemble || {};
    rows = [
      ['results', (inner.result_ids || []).length],
      [cls ? 'unanimous' : 'consensus', pctOf(cls ? e.unanimous_fraction : e.consensus_fraction)],
      ['mean spread', e.mean_spread],
      ['cells', e.n_points_used],
    ];
  } else if (inner.mode === 'series') {
    const t = inner.trajectory || {};
    rows = [
      ['dates', (inner.dates || []).length],
      ['mean net change', t.mean_total_change],
      ['shifted', pctOf(cls ? t.net_changed_fraction : t.shifted_fraction)],
      ['cells', t.n_points_used],
    ];
  } else if (inner.mode === 'ensemble') {
    const sm = inner.summary || {};
    rows = [
      ['results', inner.n_results],
      ['mean confidence', sm.mean_confidence],
      [cls ? 'mean entropy' : 'mean CV', cls ? sm.mean_entropy : sm.mean_cv],
      ['cells', inner.n_points],
    ];
  }
  rows.push(['no-data dropped', inner.n_nodata_dropped]);
  const caption = [nar.headline, nar.framing].filter(Boolean).join('. ');
  container.appendChild(chipCard(rows, caption));
  if (inner.warning) {
    const warn = document.createElement('div');
    warn.className = 'viz-cap';
    warn.textContent = inner.warning;
    container.appendChild(warn);
  }
}

/* A compact stat card for olmoearth_compare_results' pair mode (which keeps
   result_id_a / result_id_b), then the difference map auto-scanned (so a
   quantitative compare appears with both the numbers and the visual, no
   button needed). */
function renderCompareCard(container, inner) {
  const s = inner.stats || {};
  const kind = inner.kind;
  const v = compareVocab(kind);
  container.appendChild(statChips(s, v.statCaption));
  container.appendChild(abLabel(inner.result_id_a, inner.result_id_b, kind));
  if (inner.warning) {
    // Two different properties (allow_different_properties): a B - A
    // difference map would subtract two different quantities, so no scan.
    const warn = document.createElement('div');
    warn.className = 'viz-cap';
    warn.textContent = inner.warning;
    container.appendChild(warn);
    return;
  }
  if (inner.result_id_a && inner.result_id_b) {
    const out = document.createElement('div');
    out.className = 'viz-diff';
    container.appendChild(out);
    void renderDiffScan(out, { resultId: inner.result_id_a }, { resultId: inner.result_id_b },
      { property: inner.property_name, tolerance: s.tolerance, kind, noAbLabel: true });
  }
}

/* Re-render a SAVED comparison (stored real stats + grid cells) - no re-fetch,
   no fabrication: it draws exactly the data captured during the live scan. */
export async function renderStoredComparison(container, record) {
  const v = compareVocab(record.kind);
  container.appendChild(statChips(record.stats, v.savedCaption));
  container.appendChild(abLabel(record.resultIdA, record.resultIdB, record.kind));
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
  cap.innerHTML = v.mapTitle + ' over ' + cells.length + ' cells. ' + legendHtml(record.kind);
  container.appendChild(cap);
  container.appendChild(downloadBar([
    { label: 'Open overlay', onClick: () => renderOverlayViewer(record) },
    { label: 'Download JSON', onClick: () => downloadJson('comparison.json', record) },
    { label: 'Download CSV', onClick: () => downloadText('comparison.csv', comparisonCsv(record), 'text/csv') },
  ]));
}

/* A demo figure shipped with the web UI (demo mode has no bridge to inline one). */
const _DEMO_FIGURE_RE = /^demo\/figures\/[a-z0-9_-]+\.svg$/;

/* An SVG's text as a data: URL, base64 of its UTF-8 bytes (btoa alone takes
   Latin-1 only, and a caption's "±" or "…" is not). */
function svgDataUrl(svg) {
  const bytes = new TextEncoder().encode(svg);
  let bin = '';
  for (let i = 0; i < bytes.length; i += 0x8000) {
    bin += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
  }
  return 'data:image/svg+xml;base64,' + btoa(bin);
}

/* A blob: URL of a page that shows the SVG full size, for a new tab. The page
   holds the SVG in an <img> (as a data: URL), so it stays inert there too: the
   SVG opened as a document from a blob: URL would run any script in it with
   this page's origin. The page's text is fixed; no result data is in it.
   Made on the figure's first click (figureFrame), not when it is drawn. */
function figurePageUrl(svg) {
  const html = '<!doctype html><meta charset="utf-8"><title>Figure</title>' +
    '<body style="margin:0;background:#fff">' +
    '<img alt="Figure" style="display:block" src="' + svgDataUrl(svg) + '">';
  return URL.createObjectURL(new Blob([html], { type: 'text/html' }));
}

/* The figure's short title for its <img> alt: "Figure: " and the caption's
   first sentence, up to its first period (one that ends a sentence, so the
   "." of "5.0%" does not cut it); the whole caption is the figcaption. */
function figureAlt(caption) {
  const title = caption.split(/\.(?:\s|$)/)[0].trim();
  return title ? 'Figure: ' + title : "Figure of the determination, drawn from the run's data.";
}

/* The figure's frame: its image (an <img>, so any script in the SVG stays
   inert), linked to the figure full size in a new tab. `page` is that page's
   URL (a demo file), or a function that makes it (a blob: URL from
   figurePageUrl): made on the figure's first click and kept for this figure,
   so figures drawn but never opened hold no blob: URL, and reopening chats
   does not pile them up until the page closes. */
function figureFrame(page, src, caption) {
  const frame = document.createElement('div');
  frame.className = 'det-figure-frame';
  const link = document.createElement('a');
  link.className = 'det-figure-link';
  link.target = '_blank';
  link.rel = 'noopener';
  link.title = 'Open the figure full size in a new tab';
  if (typeof page === 'function') {
    let url = '';  // this figure's page, once made
    link.href = '#';
    // Before the browser follows the link: on a press of any pointer button
    // (a middle click or the menu's "Open link in new tab" gets the page too)
    // and on the click itself (the keyboard's Enter).
    const open = (e) => {
      try {
        if (!url) { url = page(); link.href = url; }
      } catch (err) {
        e.preventDefault();  // no page to open: stay here
      }
    };
    link.addEventListener('pointerdown', open);
    link.addEventListener('click', open);
  } else {
    link.href = page;
  }
  const img = document.createElement('img');
  img.alt = figureAlt(caption);
  img.src = src;
  link.appendChild(img);
  frame.appendChild(link);
  return frame;
}

/* The figure's download button; `load` gives the SVG's text (or a promise of
   it). The file keeps the name the tool gave it, else figure.svg. */
function figureDownload(source, load) {
  const named = typeof source === 'string' ? source.split(/[\\/]/).pop() : '';
  const filename = /^[\w.-]+\.svg$/.test(named) ? named : 'figure.svg';
  return downloadBar([{
    label: 'Download figure (.svg)',
    onClick: () => {
      Promise.resolve().then(load).then((text) => downloadText(filename, text, 'image/svg+xml'))
        .catch(() => { /* the file is missing: nothing to save */ });
    },
  }]);
}

/* A replayed figure (a saved chat stores no SVG): this browser's copy, kept
   in figstore.js when the figure was shown live. Found, it goes above the
   caption with the link and download a live figure gets, and the "Figure
   saved at" line goes; not found (another browser, cleared or evicted), the
   caption and that line stay. They stay too when the copy was stored under
   another caption: a later determination wrote its own figure to the same
   path, and that figure must not show under this chat's caption. */
async function restoreFigure(fig, where, path, caption) {
  try {
    const stored = await getFigure(path);
    if (!stored || !fig.isConnected) return;  // none kept, or the chat was closed meanwhile
    if (caption && stored.caption && stored.caption !== caption) return;  // another determination's figure
    const svg = stored.svg;
    fig.insertBefore(figureFrame(() => figurePageUrl(svg), svgDataUrl(svg), caption || stored.caption), fig.firstChild);
    where.remove();
    fig.after(figureDownload(path, () => svg));
  } catch (e) {
    /* keep the caption and the saved path */
  }
}

/* The determination's figure, drawn from the run's own data by the tool: the
   SVG the bridge inlined (ev.figure_svg), or in demo mode a shipped file
   (ev.figure_src), with the tool's caption and a download; the image links to
   the figure full size in a new tab. Shown through <img>, so any script in the
   SVG stays inert. A saved chat stores no SVG: it replays the caption and
   where the tool saved the figure, then shows the figure if this browser kept
   it (restoreFigure). Returns true if it rendered. */
function renderFigure(container, ev, inner) {
  const svg = typeof ev.figure_svg === 'string' && ev.figure_svg ? ev.figure_svg : null;
  const demo = !svg && typeof ev.figure_src === 'string' && _DEMO_FIGURE_RE.test(ev.figure_src)
    ? ev.figure_src : null;
  const caption = typeof inner.figure_caption === 'string' ? inner.figure_caption : '';
  const path = typeof inner.figure_path === 'string' ? inner.figure_path : '';
  if (!svg && !demo && !caption) return false;

  const fig = document.createElement('figure');
  fig.className = 'det-figure';
  if (svg) fig.appendChild(figureFrame(() => figurePageUrl(svg), svgDataUrl(svg), caption));
  else if (demo) fig.appendChild(figureFrame(demo, demo, caption));
  let where = null;
  if (caption) {
    const cap = document.createElement('figcaption');
    cap.textContent = caption;
    if (!svg && !demo && path) {
      where = document.createElement('span');
      where.className = 'det-figure-path';
      where.textContent = 'Figure saved at ' + path;
      cap.appendChild(where);
    }
    fig.appendChild(cap);
  }
  container.appendChild(fig);

  if (svg) container.appendChild(figureDownload(path, () => svg));
  else if (demo) {
    container.appendChild(figureDownload(demo,
      () => fetch(demo).then((r) => (r.ok ? r.text() : Promise.reject(r.status)))));
  } else if (where) void restoreFigure(fig, where, path, caption);  // not awaited: the caption shows now
  return true;
}

/* Render the best inline visual for a tool-result event into `container`.
   Returns true if it rendered something. */
export function renderResultViz(container, ev) {
  try {
    if (!ev || !ev.ok) return false;
    const inner = ev.result && ev.result.result;
    if (!inner || typeof inner !== 'object') return false;
    // The determination's figure first, before any other visual.
    let any = renderFigure(container, ev, inner);

    // A downloadable report (don't early-return: the result may also map).
    if (typeof inner.markdown === 'string' && inner.markdown) {
      container.appendChild(downloadBar([
        { label: 'Download report (.md)', onClick: () => downloadText('case-narrative.md', inner.markdown, 'text/markdown') },
        { label: 'Save to Results', onClick: () => document.dispatchEvent(new CustomEvent('oe:save-result', { detail: {
          type: 'file', format: 'md', filename: 'case-narrative.md', mime: 'text/markdown',
          text: inner.markdown, title: 'Case narrative' } })) },
      ]));
      any = true;
    }

    // Quantitative compare: show the numbers + auto-scan the difference map.
    if (inner.comparable === true && inner.result_id_a && inner.result_id_b) {
      renderCompareCard(container, inner);
      return true;
    }
    // The same tool's group / series / ensemble modes: a stat card.
    if (ev.name === 'olmoearth_compare_results' && inner.comparable === true && inner.mode) {
      renderCompareSummary(container, inner);
      return true;
    }

    const entries = tileEntries(inner);
    if (entries.length) { void renderResultMap(container, entries); return true; }
    return any;  // true if a downloadable report was shown
  } catch (e) {
    return false;
  }
}
