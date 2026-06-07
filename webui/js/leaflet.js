/* Lazy UMD loader for Leaflet (+ optional Leaflet.draw), shared by the AOI
   draw widget (js/aoi.js) and the in-chat result visualizer (js/viz.js).
   Leaflet ships UMD (attaches to window.L), so it is loaded via injected
   <script> tags rather than ES import - no build step. Resolves to window.L. */

/* Leaflet 1.7.1 + Leaflet.draw 1.0.4: a known-good pair (Leaflet 1.8+ renamed
   L.Polyline._flat, which Leaflet.draw 1.0.4 still calls). */
const LEAFLET = 'https://cdn.jsdelivr.net/npm/leaflet@1.7.1/dist';
const LEAFLET_DRAW = 'https://cdn.jsdelivr.net/npm/leaflet-draw@1.0.4/dist';

let _corePromise = null;
let _drawPromise = null;

function loadCss(href) {
  if (document.querySelector(`link[href="${href}"]`)) return;
  const link = document.createElement('link');
  link.rel = 'stylesheet';
  link.href = href;
  document.head.appendChild(link);
}

function loadScript(src) {
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

/* Load Leaflet once; resolve to window.L. With { draw: true }, also loads
   Leaflet.draw on top. */
export function loadLeaflet({ draw = false } = {}) {
  if (!_corePromise) {
    _corePromise = (async () => {
      loadCss(`${LEAFLET}/leaflet.css`);
      await loadScript(`${LEAFLET}/leaflet.js`);
      if (!window.L) throw new Error('Leaflet did not load');
      return window.L;
    })().catch((e) => { _corePromise = null; throw e; });
  }
  if (!draw) return _corePromise;
  if (!_drawPromise) {
    _drawPromise = (async () => {
      const L = await _corePromise;
      loadCss(`${LEAFLET_DRAW}/leaflet.draw.css`);
      await loadScript(`${LEAFLET_DRAW}/leaflet.draw.js`);
      return L;
    })().catch((e) => { _drawPromise = null; throw e; });
  }
  return _drawPromise;
}

/* Bounding box [minLon, minLat, maxLon, maxLat] from a GeoJSON geometry. */
export function geomBbox(geom) {
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  const walk = (c) => {
    if (typeof c[0] === 'number' && typeof c[1] === 'number') {
      minX = Math.min(minX, c[0]); maxX = Math.max(maxX, c[0]);
      minY = Math.min(minY, c[1]); maxY = Math.max(maxY, c[1]);
    } else if (Array.isArray(c)) { c.forEach(walk); }
  };
  walk((geom && geom.coordinates) || []);
  return [minX, minY, maxX, maxY];
}

/* The OSM basemap tile layer (shared default basemap). */
export function osmLayer(L) {
  return L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 19,
    attribution: '&copy; OpenStreetMap contributors',
  });
}
