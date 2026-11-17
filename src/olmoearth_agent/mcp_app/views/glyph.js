/* The property glyph (inlined into the Studio panel by views.py): a small
   drawing that says what a thing is, never how good it is.

   Its fields come from the server (mcp_app/glyph.py), already reduced to
   small typed values; the page never sees a record. What it draws:
   - a month ring: twelve segments, January from twelve to one o'clock,
     clockwise, a tick at the top for the year's start, gaps at every size;
     - a prediction's: dark where its requested range covers the month
       (binary), with the year as text beside it;
     - a project's: a month darker the more of its predictions cover it, in
       fixed steps (none, 1, 2 to 4, 5 or more), with the count beside it;
       all years and areas counted together, as its caption says;
   - a model's output tile, a legend and not a picture of a map: 4x4 per
     pixel, 2x2 per window; classes as cells (hollow while their colours are
     not read), values as grey bands light at the bottom, detection as a box,
     embeddings as stripes; one thin frame;
   - "not recorded" as a dashed outline (a ring with no readable dates, a
     tile with no output kind, or classes or values with no grain), never as
     an empty ring or a classification;
   - sensors (S2, S1, L8/9) and the size as text badges beside it.
   Never mint, never animated: the status chip stays the only status sign.
   In a row (about 30 px) a prediction keeps the ring only and a model the
   tile only; a model's months, and what its record does not say, are in its
   caption (a model's months ring belongs to a card, which this panel does
   not draw). Each drawing has a text name (role=img), or is aria-hidden
   beside a caption that says the same. */

OE.glyph = {};

const SVG_NS = 'http://www.w3.org/2000/svg';
OE.svg = (tag, attrs = {}, ...children) => {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value !== undefined && value !== null && value !== false) node.setAttribute(key, String(value));
  }
  for (const child of children.flat()) if (child) node.append(child);
  return node;
};

OE.glyph.COLORS = {
  plate: '#ffffff',
  covered: '#0a3235',
  track: 'rgba(16,82,87,.38)',
  // Project steps 1, 2 to 4, 5 or more: step 1 is about 2.3:1 against the track.
  steps: [null, 'rgba(16,82,87,.75)', 'rgba(16,82,87,.88)', '#0a3235'],
  frame: 'rgba(10,50,53,.5)',
  hollow: 'rgba(10,50,53,.6)',  // about 4:1 on white: the cells that say "classes"
  greys: ['#616161', '#8e8e8e', '#bdbdbd', '#e0e0e0'],  // top (dark) to bottom (light)
};

OE.glyph.MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
OE.glyph.SENSOR_NAMES = { S2: 'Sentinel-2', S1: 'Sentinel-1', 'L8/9': 'Landsat 8/9' };

/* A point on the circle, the angle in degrees clockwise from twelve o'clock. */
const at = (c, r, deg) => {
  const a = (deg * Math.PI) / 180;
  return [c + r * Math.sin(a), c - r * Math.cos(a)];
};
const fmt = (n) => Math.round(n * 100) / 100;

/* The ring's parts in an SVG of size s: a white plate, twelve arcs (each a
   colour or null for the track), the year tick; or a dashed circle when the
   months are not recorded (colors null). */
OE.glyph.ring = (s, { colors = null } = {}) => {
  const c = s / 2;
  const outer = c - 3;
  const w = s >= 48 ? 6 : 3.5;
  const r = outer - w / 2;
  const gap = ((s >= 48 ? 1.6 : 1.1) / r) * (180 / Math.PI);
  const parts = [OE.svg('circle', { cx: c, cy: c, r: c - 0.5, fill: OE.glyph.COLORS.plate })];
  if (!colors) {
    parts.push(OE.svg('circle', { cx: c, cy: c, r, fill: 'none', stroke: OE.glyph.COLORS.frame,
      'stroke-width': 1.2, 'stroke-dasharray': '2.5 2.5' }));
    return parts;
  }
  for (let i = 0; i < 12; i += 1) {
    const [x0, y0] = at(c, r, i * 30 + gap / 2);
    const [x1, y1] = at(c, r, (i + 1) * 30 - gap / 2);
    parts.push(OE.svg('path', {
      d: `M ${fmt(x0)} ${fmt(y0)} A ${fmt(r)} ${fmt(r)} 0 0 1 ${fmt(x1)} ${fmt(y1)}`,
      fill: 'none', stroke: colors[i] || OE.glyph.COLORS.track, 'stroke-width': w, 'stroke-linecap': 'butt',
    }));
  }
  // The year's start: a short tick outside the ring at twelve o'clock.
  parts.push(OE.svg('line', { x1: c, y1: c - outer - 0.4, x2: c, y2: c - outer - 2.6,
    stroke: OE.glyph.COLORS.covered, 'stroke-width': 1.5 }));
  return parts;
};

/* An output tile of side t at (x, y). */
OE.glyph.tile = (x, y, t, m) => {
  const C = OE.glyph.COLORS;
  const n = m.grain === 'pixel' ? 4 : m.grain === 'window' ? 2 : 3;
  // Dashed only for what is not recorded: no kind, or classes or values with
  // no grain. Detection and embeddings draw no grain, so their frame is solid.
  const dashedFrame = !m.kind || (!m.grain && (m.kind === 'classes' || m.kind === 'values'));
  const cell = t / n;
  const parts = [OE.svg('rect', { x, y, width: t, height: t, fill: C.plate })];
  if (m.kind === 'classes') {
    for (let row = 0; row < n; row += 1) {
      for (let col = 0; col < n; col += 1) {
        // Class colours are not read yet: hollow cells, never invented colours.
        parts.push(OE.svg('rect', { x: fmt(x + col * cell + 1), y: fmt(y + row * cell + 1),
          width: fmt(cell - 2), height: fmt(cell - 2), fill: 'none', stroke: C.hollow, 'stroke-width': 1 }));
      }
    }
  } else if (m.kind === 'values') {
    const greys = n === 2 ? [C.greys[0], C.greys[2]] : n === 3 ? [C.greys[0], C.greys[1], C.greys[3]] : C.greys;
    for (let row = 0; row < n; row += 1) {
      parts.push(OE.svg('rect', { x, y: fmt(y + row * cell), width: t, height: fmt(cell), fill: greys[row] }));
    }
  } else if (m.kind === 'detection') {
    parts.push(OE.svg('rect', { x: fmt(x + t * 0.28), y: fmt(y + t * 0.28), width: fmt(t * 0.44), height: fmt(t * 0.44),
      fill: 'none', stroke: C.covered, 'stroke-width': 1.2 }));
  } else if (m.kind === 'embeddings') {
    for (let col = 0; col < 4; col += 1) {
      parts.push(OE.svg('rect', { x: fmt(x + col * (t / 4)), y, width: fmt(t / 8), height: t, fill: C.greys[1] }));
    }
  }
  parts.push(OE.svg('rect', { x: fmt(x + 0.5), y: fmt(y + 0.5), width: t - 1, height: t - 1, fill: 'none',
    stroke: C.frame, 'stroke-width': 1, 'stroke-dasharray': dashedFrame ? '2.5 2' : null }));
  return parts;
};

/* An SVG with a text name (role=img), or hidden beside a caption. */
const drawing = (s, parts, label) => OE.svg('svg', {
  width: s, height: s, viewBox: `0 0 ${s} ${s}`, class: 'oe-glyph', focusable: 'false',
  ...(label ? { role: 'img', 'aria-label': label } : { 'aria-hidden': 'true' }),
}, parts);

/* Words for a model's output beside its tile (the Models list's caption):
   "per-pixel classes (drawn here without their colours)", "per-window
   values", "embeddings (numbers for other programs; ...)" (B/P5, B/P6). */
// WORDING: r4p2
OE.glyph.outputText = (m) => {
  if (!m || !m.kind) return 'output not recorded';
  if (m.kind === 'detection') return m.kind;
  if (m.kind === 'embeddings') return 'embeddings (numbers for other programs; these jobs cannot check or rank it)';
  const grain = m.grain === 'pixel' ? 'per-pixel ' : m.grain === 'window' ? 'per-window ' : '';
  // Said in words when the dashed frame shows it: no grain recorded.
  const noGrain = m.grain ? '' : ', per pixel or per window not recorded';
  // Values, not a confidence: the grey bands are the same for every values model.
  return m.kind === 'classes' ? grain + 'classes (drawn here without their colours' + noGrain + ')' : grain + 'values' + (noGrain ? ' (' + noGrain.slice(2) + ')' : '');
};

/* A model's output in the Models list's own words, without what its tile
   shows (a prediction's row draws no tile): "per-pixel classes",
   "per-window values", "embeddings", "detection" (B/P1). */
OE.glyph.outputWords = (m) => {
  if (!m || !m.kind) return 'output not recorded';
  if (m.kind === 'detection' || m.kind === 'embeddings') return m.kind;
  const grain = m.grain === 'pixel' ? 'per-pixel ' : m.grain === 'window' ? 'per-window ' : '';
  return grain + (m.kind === 'classes' ? 'classes' : 'values');
};

/* Month names grouped by step, for a project's ring in words. */
const stepWords = (months) => {
  const names = ['none', '1', '2 to 4', '5 or more'];
  const groups = [3, 2, 1].map((step) => {
    const list = months.map((v, i) => (v === step ? OE.glyph.MONTHS[i] : null)).filter(Boolean);
    return list.length ? names[step] + ': ' + list.join(', ') : null;
  }).filter(Boolean);
  return groups.length ? groups.join('; ') : 'no month covered';
};

/* A prediction in a row: the ring (its months), the year beside it. */
OE.glyph.predictionRow = (p) => {
  const s = 30;
  const colors = Array.isArray(p.months) ? p.months.map((v) => (v ? OE.glyph.COLORS.covered : null)) : null;
  const label = p.span ? 'Covers ' + p.span : 'Dates not recorded';
  return OE.el('span', { class: 'glyph-row' },
    drawing(s, OE.glyph.ring(s, { colors }), label),
    OE.el('span', { class: 'glyph-year', 'aria-hidden': 'true' }, p.years || 'not recorded'));
};

/* A project on its card: the graded ring, the count beside it, the caption. */
OE.glyph.projectCard = (g) => {
  const s = 56;
  // No readable dates at all: the dashed "not recorded" ring, not "no month".
  const colors = Array.isArray(g.months) ? g.months.map((v) => OE.glyph.COLORS.steps[v] || null) : null;
  const count = g.count + (g.capped ? '+' : '') + ' prediction' + (g.count === 1 && !g.capped ? '' : 's');
  const undated = g.count - g.dated;
  // WORDING: r4p2 (B/P3)
  const note = !colors ? 'dates not recorded'
    : 'months of the year its predictions cover (all years and areas counted together)'
      + (undated > 0 ? '; ' + undated + ' without dates' : '');
  return OE.el('div', { class: 'glyph-card' },
    drawing(s, OE.glyph.ring(s, { colors })),
    OE.el('div', {},
      OE.el('div', { class: 'glyph-count' }, count),
      OE.el('div', { class: 'small faint' }, note),
      colors ? OE.el('span', { class: 'sr-only' }, 'Predictions per month: ' + stepWords(g.months) + '.') : null));
};

/* A badge: monospace text for a sensor (its full name for a screen reader),
   a word for a size. */
OE.glyph.badge = (text, { mono = false, full = null } = {}) => OE.el('span', { class: 'badge' + (mono ? ' mono' : '') },
  text, full ? OE.el('span', { class: 'sr-only' }, ' (' + full + ')') : null);

/* A model in a row: its tile only (aria-hidden: the caption says the same),
   then the badges, and the caption in words, which also says what the
   record does not: months, sources, size. */
OE.glyph.modelRow = (m) => {
  const s = 30;
  const t = 22;
  const sensors = m.sensors || [];
  const caption = [
    OE.glyph.outputText(m),
    m.context || 'months not recorded',
    sensors.length ? null : 'sources not recorded',
    m.size ? null : 'size not recorded',
  ].filter(Boolean).join(' · ');
  return {
    tile: drawing(s, OE.glyph.tile((s - t) / 2, (s - t) / 2, t, m)),
    badges: [
      ...sensors.map((b) => OE.glyph.badge(b, { mono: true, full: OE.glyph.SENSOR_NAMES[b] || b })),
      m.size ? OE.glyph.badge(m.size + (m.version ? ' ' + m.version : '')) : null,
    ].filter(Boolean),
    caption,
  };
};

OE.glyph.STYLE = `
.oe-glyph { flex: none; display: block; }
.glyph-row { display: inline-flex; align-items: center; gap: 6px; }
.glyph-year { font-size: 12px; line-height: 18px; color: var(--text); min-width: 7em; white-space: nowrap; }
.glyph-card { display: flex; align-items: center; gap: 12px; margin: 4px 0 8px; min-height: 56px; }
.glyph-count { font-weight: 600; font-size: 14px; line-height: 21px; color: var(--text); }
.badge { display: inline-flex; align-items: center; height: 20px; padding: 0 6px; border: 1px solid var(--divider);
  border-radius: 4px; font-size: 12px; line-height: 18px; color: var(--text); background: var(--paper); white-space: nowrap; }
.badge.mono { font-family: var(--mono); font-size: 11px; }
.sr-only { position: absolute; width: 1px; height: 1px; overflow: hidden; clip: rect(0 0 0 0); white-space: nowrap; }
`;
