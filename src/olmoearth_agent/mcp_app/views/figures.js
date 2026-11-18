/* Small figures for the Check card and the estimate's result card (inlined
   by views.py into the check and result views, after common.js).

   Each takes plain values the server already wrote into the view data and
   returns a node, or null when it has nothing to draw. None calls the
   server, none redraws itself (no timer, no listener): a figure never
   updates while labelling. Each puts its rules in the page once
   (OE.fig.style), with common.js's tokens. A figure that holds text is
   role=img with the same text in its aria-label; the steps' pictures are
   aria-hidden beside the sentence that says the same (the glyph's rule).
   Everything lives inside OE.fig, so no name here meets one of the scripts
   inlined beside it.

   - steps(sentences): up to four small schematic pictures in a row, each
     over the caller's sentence: a grid of squares; the same grid with a
     few picked; a satellite picture with an outlined square and two answer
     buttons; a 0% to 100% line with a dot and a band. A drawing, not the
     user's data: no map class, confidence or place.
   - rateLine({rate, low, high, rateLow, rateHigh, rateText, rangeText,
     lowText, highText}): a 0% to 100% line, the 95% range low to high as a
     pale band, and the rate as a dot labelled rateText; with no rate (Cannot
     tell answers), the rate's own two ends rateLow to rateHigh as a short
     dark bar over the band, labelled rateText (r4-h R4). Under the band, its
     two ends as numbers, lowText and highText as the server wrote them (r4-k
     K6: the owner's choice of 16 Nov 2026, which reverses r4-h R1's "no
     words under the line", so the figure alone, or a picture of it, never
     shows a rate without its range): each under its end, or, where the band
     is too narrow for two, one label centred under it (rangeText, the
     server's "a to b"); the 0% or the 100% label is left out where an end's
     label would touch it. With Cannot tell the bar keeps its label (the
     rate's own two ends) and the band gets its ends as without. Without
     lowText and highText, nothing under the band. The aria-label says the
     rate and the range. rate, low, high, rateLow and rateHigh are fractions
     in [0, 1]; a value outside it is not drawn. The figure computes no
     number: it places the server's texts.
   - parts(parts, {mode}): two or more parts of a map side by side, each as
     wide as its share (share_value or share; at least 12% wide, and never
     narrower than its label's longest word, so no word is broken), its label
     (else its name) inside its box, every box one height (r5-e R5; it stood
     under an empty bar); on a panel of 560 px or less the parts are stacked,
     each full width. mode 'plan' (the Check card, r4-h A/F08):
     state 'open' dashed (not picked yet), 'picked' a solid outline, neither
     filled. mode 'result': state 'this' filled, 'graded' lightly filled,
     'not' dashed.
   - squares({cells, classes, legend}): the squares checked, 14 px each,
     wrapping, in the order given, marked by the class the reviewer saw
     (the first class seen filled --text; the second hollow, a 2 px --text
     ring on --paper; the third a mid teal, --text-2 at 45%, with a 1 px
     --text ring: r5-b J04, so each is told from the card and from the
     others by more than a pale fill, and mint stays the ring of a completed
     status; classes no cell has take no mark), a thick red outline set off
     from the square where the map said otherwise (r4-h V2), a Cannot tell
     square (unjudged) empty and dashed, never filled or outlined (r4-i O2),
     and a legend of at most five entries, one per class seen, one for Cannot
     tell and one for the outline, whose sample is the red ring alone with
     nothing inside (r5-b J04: it was a white square, which read as a class).

   Round r5-b, J29: the steps keep 12 px above them (b: this style sheet is
   put in the page after common.js's, so its margin beat .stack's gap and
   the map's name touched the pictures); the 0% and 100% labels are --text-2
   and the range's band is 45% (d: both were too pale to read). */

OE.fig = (() => {
  const NS = 'http://www.w3.org/2000/svg';

  /* An SVG element: svg('rect', {x: 1}, ...children). */
  const svg = (tag, attrs = {}, ...children) => {
    const node = document.createElementNS(NS, tag);
    for (const [key, value] of Object.entries(attrs || {})) {
      if (value !== undefined && value !== null && value !== false) node.setAttribute(key, String(value));
    }
    for (const child of children.flat()) {
      if (child === undefined || child === null || child === false) continue;
      node.append(child instanceof Node ? child : document.createTextNode(String(child)));
    }
    return node;
  };

  const round2 = (n) => Math.round(n * 100) / 100;

  /* A fraction in [0, 1], else null (never a boolean or a string). */
  const fraction = (v) => (typeof v === 'number' && Number.isFinite(v) && v >= 0 && v <= 1 ? v : null);

  /* A string with words in it, trimmed, else ''. */
  const words = (v) => (typeof v === 'string' && v.trim() ? v.trim() : '');

  const STYLE = `
.oe-fig text { font-family: var(--font); fill: var(--text); }
.oe-fig .axis { stroke: var(--grey); stroke-width: 1; }
.oe-fig .band { fill: var(--text-2); fill-opacity: .45; }
.oe-fig .dot { fill: var(--text); stroke: var(--paper); stroke-width: 1.5; }
.oe-fig .tick-label { font-size: 11px; fill: var(--text-2); }
.oe-fig .rate-label { font-size: 12px; font-weight: 600; }
.oe-fig .end-label { font-size: 11px; font-weight: 600; }
.oe-fig .rate-bar { fill: var(--text); }
.oe-steps { list-style: none; margin: 12px 0 0; padding: 0; display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; }
.oe-steps > li { display: flex; flex-direction: column; align-items: flex-start; gap: 6px; min-width: 0; }
.oe-steps .oe-step-pic { display: block; flex: none; }
.oe-steps .cell { fill: var(--wash-2); stroke: var(--outline); stroke-width: 1; }
.oe-steps .cell.picked { fill: var(--text); stroke: var(--text); }
.oe-steps .scene { fill: var(--text-2); }
.oe-steps .scene-2 { fill: var(--text); fill-opacity: .45; }
.oe-steps .outline { fill: none; stroke: var(--paper); stroke-width: 1.5; }
.oe-steps .pill { fill: var(--paper); stroke: var(--outline); stroke-width: 1; }
.oe-steps .pill.on { fill: var(--selected); stroke: var(--text); }
@media (max-width: 560px) { .oe-steps { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
.oe-rateline svg { display: block; max-width: 100%; height: auto; }
.oe-parts { display: flex; gap: 4px; align-items: stretch; }
.oe-part { min-width: min-content; display: flex; flex-direction: column; gap: 4px; }
.oe-part-box { flex: 1; min-height: 28px; padding: 4px 8px; display: flex; align-items: center;
  border: 1.5px solid var(--text-2); border-radius: 2px; background: var(--paper); }
.oe-part.this .oe-part-box { background: var(--text); border-color: var(--text); }
.oe-part.this .oe-part-label { color: var(--canvas); }
.oe-part.graded .oe-part-box { background: rgba(16,82,87,.3); }
.oe-part.not .oe-part-box { border-style: dashed; background: transparent; }
.oe-part.open .oe-part-box { border-style: dashed; background: transparent; }
.oe-part.picked .oe-part-box { background: transparent; }
.oe-part-label { color: var(--text); overflow-wrap: break-word; }
@media (max-width: 560px) { .oe-parts { display: grid; grid-template-columns: minmax(0, 1fr); grid-auto-rows: 1fr; } }
.oe-sq-grid { display: flex; flex-wrap: wrap; gap: 8px; padding: 5px; }
.oe-sq, .oe-swatch { display: inline-block; flex: none; width: 14px; height: 14px; border-radius: 2px;
  box-shadow: inset 0 0 0 1px var(--divider); }
.oe-sq.f0, .oe-swatch.f0 { background: var(--text); }
.oe-sq.f1, .oe-swatch.f1 { background: var(--paper); box-shadow: inset 0 0 0 2px var(--text); }
.oe-sq.f2, .oe-swatch.f2 { background: rgba(16,82,87,.45); box-shadow: inset 0 0 0 1px var(--text); }
.oe-swatch.blank { background: transparent; box-shadow: none; }
.oe-sq.cannot, .oe-swatch.cannot { background: transparent; box-shadow: none; border: 1px dashed var(--text); box-sizing: border-box; }
.oe-sq.wrong, .oe-swatch.wrong { outline: 3px solid var(--error); outline-offset: 2px; }
.oe-sq-legend { list-style: none; margin: 8px 0 0; padding: 0 0 0 5px; display: flex; flex-wrap: wrap; gap: 8px 16px; }
.oe-sq-legend > li { display: inline-flex; align-items: center; gap: 12px; }
`;

  /* The figures' rules, put in the page's head once. */
  const STYLE_ID = 'oe-fig-style';
  const style = () => {
    if (document.getElementById(STYLE_ID)) return;
    const node = document.createElement('style');
    node.id = STYLE_ID;
    node.textContent = STYLE;
    document.head.append(node);
  };

  /* The steps: four pictures of 96 x 56, each aria-hidden (its sentence is
     beside it). Fixed drawings: the same on every card. */
  const PIC_W = 96;
  const PIC_H = 56;
  const picture = (...shapes) => svg('svg', {
    class: 'oe-step-pic', width: PIC_W, height: PIC_H, viewBox: `0 0 ${PIC_W} ${PIC_H}`,
    'aria-hidden': 'true', focusable: 'false',
  }, shapes);

  /* A grid of 7 x 4 squares; those at picked ([column, row]) filled. */
  const grid = (picked = []) => {
    const cells = [];
    for (let row = 0; row < 4; row += 1) {
      for (let col = 0; col < 7; col += 1) {
        const on = picked.some(([c, r]) => c === col && r === row);
        cells.push(svg('rect', {
          class: on ? 'cell picked' : 'cell', x: round2(3.5 + col * 13), y: 3 + row * 13, width: 11, height: 11, rx: 1,
        }));
      }
    }
    return picture(...cells);
  };
  const PICKED = [[1, 0], [5, 1], [2, 2], [4, 3]];

  /* One satellite picture with a square outlined in its middle, and two
     answer buttons under it, one pressed. No words on them. */
  const look = () => picture(
    svg('rect', { class: 'scene', x: 8, y: 2, width: 80, height: 36, rx: 2 }),
    svg('path', { class: 'scene-2', d: 'M8 26 C 26 16, 44 34, 62 22 S 84 18, 88 24 L 88 38 L 8 38 Z' }),
    svg('rect', { class: 'outline', x: 41, y: 13, width: 14, height: 14 }),
    svg('rect', { class: 'pill on', x: 18, y: 43, width: 26, height: 11, rx: 5.5 }),
    svg('rect', { class: 'pill', x: 52, y: 43, width: 26, height: 11, rx: 5.5 }));

  /* A 0% to 100% line with a band and a dot: a shape, no number of anyone's. */
  const line = () => picture(
    svg('line', { class: 'axis', x1: 8, y1: 24, x2: 88, y2: 24 }),
    svg('line', { class: 'axis', x1: 8, y1: 19, x2: 8, y2: 29 }),
    svg('line', { class: 'axis', x1: 88, y1: 19, x2: 88, y2: 29 }),
    svg('rect', { class: 'band', x: 24, y: 19, width: 22, height: 10, rx: 1 }),
    svg('circle', { class: 'dot', cx: 33, cy: 24, r: 4 }),
    svg('text', { class: 'tick-label', x: 4, y: 48, 'text-anchor': 'start' }, '0%'),
    svg('text', { class: 'tick-label', x: 92, y: 48, 'text-anchor': 'end' }, '100%'));

  const PICTURES = [() => grid(), () => grid(PICKED), look, line];

  /* The four steps as a list, each picture over its sentence (the caller's
     words, as given); a sentence left empty is left out with its picture. */
  const steps = (sentences) => {
    const items = [];
    (Array.isArray(sentences) ? sentences : []).slice(0, PICTURES.length).forEach((sentence, i) => {
      const said = words(sentence);
      if (said) items.push(OE.el('li', {}, PICTURES[i](), OE.el('span', { class: 'small' }, said)));
    });
    if (!items.length) return null;
    style();
    return OE.el('ol', { class: 'oe-fig oe-steps' }, items);
  };

  /* The rate on a 0% to 100% line: 320 x 50, the line from x 16 to 304 at
     y 24, the rate's label above it, the ends' labels under it. */
  const RL = { w: 320, h: 50, left: 16, right: 304, y: 24 };
  const xOf = (v) => RL.left + v * (RL.right - RL.left);
  /* A label above x, run inward near an end so it stays inside the figure. */
  const labelAt = (x, text) => {
    const anchor = x < 60 ? 'start' : x > RL.w - 60 ? 'end' : 'middle';
    const tx = anchor === 'start' ? Math.max(0, x - 6) : anchor === 'end' ? Math.min(RL.w, x + 6) : x;
    return svg('text', { class: 'rate-label', x: round2(tx), y: 12, 'text-anchor': anchor }, text);
  };
  /* A label under the line (at y 44, 11 px): its width estimated at UNDER_CH
     a character (a digit of the 600 label is about 6.8 px, so a little over),
     centred on x and run inward near an end so it stays inside the figure;
     with its left and right edges. */
  const UNDER_CH = 7;
  const UNDER_GAP = 4;
  const underAt = (x, text) => {
    const half = (text.length * UNDER_CH) / 2;
    const cx = Math.min(Math.max(x, half), RL.w - half);
    return { text, cx, left: cx - half, right: cx + half };
  };
  /* Whether two labels under the line keep at least UNDER_GAP apart. */
  const apart = (a, b) => a.right + UNDER_GAP <= b.left || b.right + UNDER_GAP <= a.left;
  const rateLine = (values) => {
    const o = values && typeof values === 'object' ? values : {};
    const rate = fraction(o.rate);
    const low = fraction(o.low);
    const high = fraction(o.high);
    const ranged = low !== null && high !== null && high >= low;
    if (rate === null && !ranged) return null;
    // No rate (Cannot tell answers): the rate's own two ends, drawn as a bar
    // over the 95% band, never without it.
    const rateLow = fraction(o.rateLow);
    const rateHigh = fraction(o.rateHigh);
    const barred = rate === null && ranged && rateLow !== null && rateHigh !== null && rateHigh >= rateLow;
    style();
    const rateText = rate !== null || barred ? words(o.rateText) : '';
    const rangeText = words(o.rangeText);
    // The band's two ends as numbers under it (r4-k K6), the server's own
    // texts, with or without a rate: each under its end while the two keep
    // apart, else one label centred under the band, the server's "a to b"
    // (rangeText; its two texts joined only if it gave none). Nothing is
    // computed here but where each label goes.
    const lowText = ranged ? words(o.lowText) : '';
    const highText = ranged ? words(o.highText) : '';
    let ends = [];
    if (lowText && highText) {
      const a = underAt(xOf(low), lowText);
      const b = underAt(xOf(high), highText);
      // WORDING: r4k K6
      ends = apart(a, b) ? [a, b] : [underAt((xOf(low) + xOf(high)) / 2, rangeText || lowText + ' to ' + highText)];
    }
    // The 0% and 100% labels, each left out where an end's label would touch it.
    const tick = (x, text) => (ends.every((e) => apart(e, underAt(x, text)))
      ? svg('text', { class: 'tick-label', x, y: 44, 'text-anchor': 'middle' }, text) : null);
    const marks = [
      svg('line', { class: 'axis', x1: RL.left, y1: RL.y, x2: RL.right, y2: RL.y }),
      svg('line', { class: 'axis', x1: RL.left, y1: RL.y - 5, x2: RL.left, y2: RL.y + 5 }),
      svg('line', { class: 'axis', x1: RL.right, y1: RL.y - 5, x2: RL.right, y2: RL.y + 5 }),
      tick(RL.left, '0%'),
      tick(RL.right, '100%'),
    ].filter(Boolean);
    if (ranged) {
      const x0 = xOf(low);
      marks.push(svg('rect', {
        class: 'band', x: round2(x0), y: RL.y - 5, width: round2(Math.max(2, xOf(high) - x0)), height: 10, rx: 1,
      }));
    }
    for (const e of ends) {
      marks.push(svg('text', { class: 'end-label', x: round2(e.cx), y: 44, 'text-anchor': 'middle' }, e.text));
    }
    if (rate !== null) {
      const x = xOf(rate);
      marks.push(svg('circle', { class: 'dot', cx: round2(x), cy: RL.y, r: 5 }));
      if (rateText) marks.push(labelAt(x, rateText));
    } else if (barred) {
      // At least 3 px wide, about the middle of its two ends.
      const x0 = xOf(rateLow);
      const mid = (x0 + xOf(rateHigh)) / 2;
      const across = Math.max(3, xOf(rateHigh) - x0);
      marks.push(svg('rect', {
        class: 'rate-bar', x: round2(mid - across / 2), y: RL.y - 3, width: round2(across), height: 6,
      }));
      if (rateText) marks.push(labelAt(mid, rateText));
    }
    // WORDING: r4h R1
    const said = rangeText ? '95% range ' + rangeText : '';
    const label = [rateText, said].filter(Boolean).join('; ');
    return OE.el('div', { class: 'oe-fig oe-rateline', ...(label ? { role: 'img', 'aria-label': label } : {}) },
      svg('svg', { width: RL.w, height: RL.h, viewBox: `0 0 ${RL.w} ${RL.h}`, focusable: 'false', 'aria-hidden': 'true' },
        marks));
  };

  /* A part's size: share_value, else share (a number, or a percent text
     such as "46.7%" read as 0.467); null when it gives none. */
  const shareOf = (p) => {
    for (const v of [p.share_value, p.share]) {
      if (typeof v === 'number' && Number.isFinite(v) && v >= 0) return v;
      if (typeof v === 'string') {
        const m = /^\s*(\d+(?:\.\d+)?)\s*%\s*$/.exec(v);
        if (m) return Number(m[1]) / 100;
      }
    }
    return null;
  };

  /* Each part's width, as a fraction of the row: by its share, at least
     FLOOR (a part under it is drawn at FLOOR and the others share the rest
     by their shares); equal widths when a share is missing, they add up to
     nothing, or the floor cannot hold for so many parts. The style sheet
     keeps a part at least as wide as its label's longest word, and stacks
     the parts full width on a panel of 560 px or less, where the widths
     here are not used. */
  const FLOOR = 0.12;
  const widths = (shares) => {
    const n = shares.length;
    const total = shares.reduce((sum, s) => sum + (s === null ? 0 : s), 0);
    if (shares.some((s) => s === null) || !(total > 0) || n * FLOOR >= 1) return shares.map(() => 1 / n);
    let w = shares.map((s) => s / total);
    const fixed = new Set();
    for (let pass = 0; pass < n; pass += 1) {
      const under = shares.map((_, i) => i).filter((i) => !fixed.has(i) && w[i] < FLOOR);
      if (!under.length) break;
      under.forEach((i) => fixed.add(i));
      const free = shares.reduce((sum, s, i) => sum + (fixed.has(i) ? 0 : s), 0);
      const room = 1 - FLOOR * fixed.size;
      const left = n - fixed.size;
      w = shares.map((s, i) => (fixed.has(i) ? FLOOR : free > 0 ? (s / free) * room : room / left));
    }
    return w;
  };

  /* The states each mode draws: the result card's, and the Check card's
     (r4-h A/F08: dashed while not picked, a solid outline once picked; no
     fill, which on the result card means a result). */
  const STATES = { result: ['this', 'graded', 'not'], plan: ['open', 'picked'] };

  /* A map's parts side by side (two or more; one part draws nothing). Each
     part's words sit inside its own box, and every box in the row is one
     height (r5-e R5, the owner's note of 17 Nov 2026: no loose words under
     an empty box); on a filled box (this result) the words are light. The
     widths follow each part's share from 561 px up, never narrower than a
     part's longest word (a part at the 12% floor broke its words letter by
     letter); at 560 px or less the parts are stacked, each full width, in
     rows of one height. */
  const parts = (list, options) => {
    const items = (Array.isArray(list) ? list : []).filter((p) => p && typeof p === 'object');
    if (items.length < 2) return null;
    style();
    const known = STATES[options && options.mode === 'result' ? 'result' : 'plan'];
    const w = widths(items.map(shareOf));
    const labels = items.map((p, i) => {
      // WORDING: squares
      const name = words(p.name) || words(p.part) || 'Part ' + (i + 1);
      return words(p.label) || name;
    });
    const pieces = items.map((p, i) => {
      // A state the card does not know is drawn outlined, saying nothing.
      const state = known.includes(p.state) ? p.state : 'plan';
      return OE.el('div', { class: 'oe-part ' + state, style: 'flex: ' + round2(w[i] * 100) + ' 1 0%' },
        OE.el('div', { class: 'oe-part-box' }, OE.el('span', { class: 'oe-part-label small' }, labels[i])));
    });
    return OE.el('div', { class: 'oe-fig oe-parts', role: 'img', 'aria-label': labels.join('; ') }, pieces);
  };

  /* The squares checked. cells [{seen, wrong, unjudged}]: seen is the class
     code the reviewer gave, wrong whether the map said otherwise (true or
     1), unjudged true for a square answered Cannot tell (r4-i O2: drawn
     empty and dashed, no fill and never outlined, and left out of the
     fills' count and order). classes [{code, name}] give the order of the
     fills among the classes that have cells; legend
     [{code, text}] names a class's fill, [{unjudged: true, text}] the dashed
     square, [{wrong: true, text}] the outline (a class without an entry is
     named by its name, the dashed square by CANNOT_KEY, the outline by
     WRONG_KEY). The legend's order: the classes, Cannot tell, the outline.
     The grid's aria-label is the legend with each count. */
  const FILLS = ['f0', 'f1', 'f2'];
  // WORDING: r4p2
  const WRONG_KEY = 'Outlined in red: the map said otherwise';
  // WORDING: r4i O2
  const CANNOT_KEY = 'Cannot tell';
  const isWrong = (c) => c.wrong === true || c.wrong === 1;
  const isCannot = (c) => c.unjudged === true;
  const squares = (data) => {
    const d = data && typeof data === 'object' ? data : {};
    const cells = (Array.isArray(d.cells) ? d.cells : []).filter((c) => c && typeof c === 'object');
    if (!cells.length) return null;
    style();
    const classes = (Array.isArray(d.classes) ? d.classes : []).filter((c) => c && typeof c === 'object');
    const legend = (Array.isArray(d.legend) ? d.legend : []).filter((e) => e && typeof e === 'object');
    const key = (v) => String(v ?? '');
    // The squares with a class seen: a Cannot tell square takes no fill.
    const judged = cells.filter((c) => !isCannot(c));
    const count = new Map();
    for (const c of judged) count.set(key(c.seen), (count.get(key(c.seen)) || 0) + 1);
    // The fills go in turn to the classes that have cells only, in the
    // classes' order (then any code seen that they do not list): on a map of
    // many classes, the at most three seen get f0, f1 and f2, never one fill
    // for two of them (round 4 pass 2, findings 4 and 21).
    const order = [];
    for (const v of [...classes.map((c) => c.code), ...judged.map((c) => c.seen)]) {
      const k = key(v);
      if (count.get(k) && !order.includes(k)) order.push(k);
    }
    const fill = (code) => FILLS[Math.min(order.indexOf(key(code)), FILLS.length - 1)];
    const nameOf = (code) => {
      const given = legend.find((e) => !e.wrong && !e.unjudged && key(e.code) === code);
      if (given && words(given.text)) return words(given.text);
      const c = classes.find((x) => key(x.code) === code);
      return (c && words(c.name)) || code;
    };
    const entries = order.filter((code) => count.get(code)).slice(0, FILLS.length)
      .map((code) => ({ cls: fill(code), text: nameOf(code), n: count.get(code) }));
    // Cannot tell: after the classes, before the outline.
    const nCannot = cells.length - judged.length;
    if (nCannot) {
      const given = legend.find((e) => e.unjudged === true && words(e.text));
      entries.push({ cls: 'cannot', text: given ? words(given.text) : CANNOT_KEY, n: nCannot });
    }
    const nWrong = judged.filter(isWrong).length;
    if (nWrong) {
      const given = legend.find((e) => e.wrong === true && words(e.text));
      entries.push({ cls: 'blank wrong', text: given ? words(given.text) : WRONG_KEY, n: nWrong });
    }
    const field = OE.el('div', {
      class: 'oe-sq-grid', role: 'img', 'aria-label': entries.map((e) => e.text + ' (' + e.n + ')').join('; '),
    }, cells.map((c) => OE.el('span', {
      class: isCannot(c) ? 'oe-sq cannot' : 'oe-sq ' + fill(c.seen) + (isWrong(c) ? ' wrong' : ''),
    })));
    // The legend is said by the grid's aria-label, with the counts.
    const keys = OE.el('ul', { class: 'oe-sq-legend small', 'aria-hidden': 'true' },
      entries.map((e) => OE.el('li', {}, OE.el('span', { class: 'oe-swatch ' + e.cls }), e.text)));
    return OE.el('div', { class: 'oe-fig oe-squares' }, field, keys);
  };

  return { STYLE, style, steps, rateLine, parts, squares };
})();
