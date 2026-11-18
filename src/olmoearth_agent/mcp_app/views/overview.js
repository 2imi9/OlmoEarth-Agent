/* The overview figure on the start card (inlined by views.py into the start
   view, after common.js): how OlmoEarth makes a map, and where this app
   comes in.

   On a panel at least 680 px wide it is the overview figure of the OlmoEarth
   docs itself (docs.olmoearth.allenai.org, used with Ai2's permission, owner
   17 Nov 2026), as a picture inlined by views.py (@OVERVIEW_DATA_URL@, from
   views/brand/olmoearth-overview.jpg: nothing is loaded from elsewhere). On
   a narrower panel that picture's words would be too small to read, so a
   strip of three boxes drawn in code says the same in plain words: Studio
   makes a map, you check random squares, you get how often it is wrong.

   One of the two shows at a time (the rules below). The figure's text name
   is its wrapper's (role=img, the title and the description as its
   aria-label; the picture and the drawing are aria-hidden). The caption and
   the docs link sit outside that wrapper (the start card puts them there):
   what a role=img element holds is hidden from a screen reader. The caption
   has two forms, CAPTION for the docs figure and CAPTION_NARROW for the
   strip (r5-d R04), shown at the same widths as what each describes.

   The drawing uses common.js's tokens, but for the one small tile, which
   takes the fixed colours TILE (a tile stands for a map, whose classes have
   colours of their own). No mint: the dot on the line is the canvas colour.
   The drawing says no number: the line with a band and a dot has no label,
   so it never reads as a result. No call, no timer, no listener; everything
   lives inside OE.overview, so no name here meets one of the scripts
   inlined beside it. */

OE.overview = (() => {
  const NS = 'http://www.w3.org/2000/svg';

  /* The docs figure, as a data URL (views.py fills it in). */
  const PICTURE = '@OVERVIEW_DATA_URL@';

  /* An SVG element: svg('rect', {x: 1}, ...children); lists of children,
     nested too, are taken apart. */
  const svg = (tag, attrs = {}, ...children) => {
    const node = document.createElementNS(NS, tag);
    for (const [key, value] of Object.entries(attrs || {})) {
      if (value !== undefined && value !== null && value !== false) node.setAttribute(key, String(value));
    }
    for (const child of children.flat(Infinity)) {
      if (child === undefined || child === null || child === false) continue;
      node.append(child instanceof Node ? child : document.createTextNode(String(child)));
    }
    return node;
  };

  /* The figure's text name: its title, then what it shows, in plain words. */
  // WORDING: r5c docs figure
  const TITLE = 'How OlmoEarth makes a map, and where this app comes in.';
  // WORDING: r5c docs figure
  const DESC = 'Your labelled examples and satellite pictures go into OlmoEarth Studio. There you teach a model and run it, again and again. Studio makes maps, such as crops, mangrove loss and fire risk. This app then checks a map: you look at squares picked at random and get how often the map is wrong, with a range.';

  /* The one small line under the figure (the start card draws it outside
     the role=img wrapper, the docs link after it). */
  // WORDING: r5c docs figure
  const CAPTION = 'How OlmoEarth makes maps (a figure from the OlmoEarth docs). This app comes after: it checks how often a map is wrong.';
  /* The caption on a narrow panel (r5-d R04), where the strip shows and not
     the docs figure that CAPTION describes. The start card draws each in a
     span of its own; the panel's width picks the one shown (STYLE), as it
     picks the drawing. */
  // WORDING: r5d R04
  const CAPTION_NARROW = 'How a check of a Studio map works.';

  /* The narrow strip's words: three boxes. A list is one label over several
     lines (SVG text does not wrap: each break is set here). No digit and no
     per cent sign in any of them. */
  // WORDING: r5a V1
  const WORDS = {
    narrowStudio: ['Studio', 'makes a map'],
    narrowCheck: ['You check', 'random squares'],
    narrowGet: ['You get', 'how often it is', 'wrong, with a', 'range'],
  };

  /* The tile's colours: its ground, then its patches'. The one place a
     colour is written out (see the head of this file). */
  const TILE = { crops: ['#d9e8c8', '#7fb069', '#e6c36a', '#b7d48f'] };

  /* The tile's patches over its ground, as [which of its TILE colours, the
     patch's outline] in the tile's own 38 x 38 units; a patch at a corner
     follows the tile's rounded corner. */
  const PATCHES = {
    crops: [
      [1, 'M4 0H19V13H0V4A4 4 0 0 1 4 0Z'],
      [2, 'M19 13H38V25H19Z'],
      [3, 'M6 25H21V38H6Z'],
      [3, 'M25 0H34A4 4 0 0 1 38 4V9H25Z'],
    ],
  };

  const STYLE = `
.oe-overview img.wide { display: none; width: 100%; max-width: 748px; height: auto;
  background: var(--paper); border: 1px solid var(--divider); border-radius: 12px; }
.oe-overview svg { display: block; width: 100%; height: auto; }
.oe-overview svg.narrow { max-width: 440px; }
.oe-overview-cap .cap-wide { display: none; }
@media (min-width: 680px) {
  .oe-overview img.wide { display: block; }
  .oe-overview svg.narrow { display: none; }
  .oe-overview-cap .cap-wide { display: inline; }
  .oe-overview-cap .cap-narrow { display: none; }
}
.oe-overview text { font-family: var(--font); fill: var(--text); }
.oe-overview .t { font-size: 13px; font-weight: 600; }
.oe-overview .s { font-size: 11.5px; font-weight: 400; fill: var(--text-2); }
.oe-overview .box { fill: var(--paper); stroke: var(--divider); stroke-width: 1; }
.oe-overview .ln { fill: none; stroke: var(--text-2); stroke-width: 1.25; stroke-linecap: round;
  stroke-linejoin: round; opacity: .55; }
.oe-overview .tile-edge { fill: none; stroke: var(--field); stroke-width: 1; }
.oe-overview .sq { fill: var(--text); opacity: .2; }
.oe-overview .sq.on { opacity: 1; }
.oe-overview .app { fill: var(--text); }
.oe-overview .dark text { fill: var(--canvas); }
.oe-overview .dark .t { font-weight: 700; }
.oe-overview .axis { fill: none; stroke: var(--canvas); stroke-width: 1.25; opacity: .7; }
.oe-overview .band { fill: none; stroke: var(--canvas); stroke-width: 5; opacity: .55; }
.oe-overview .dot { fill: var(--canvas); stroke: var(--text); stroke-width: 1.5; }
`;

  /* The figure's rules, put in the page's head once. */
  const STYLE_ID = 'oe-overview-style';
  const style = () => {
    if (document.getElementById(STYLE_ID)) return;
    const node = document.createElement('style');
    node.id = STYLE_ID;
    node.textContent = STYLE;
    document.head.append(node);
  };

  /* One line of text from (x, y); cls sets its size and weight (t a title,
     s a small line). */
  const text = (cls, x, y, words, anchor = null) => svg('text', { class: cls, x, y, 'text-anchor': anchor }, words);

  /* A box and what it holds, as one group (g.inbox, the box first): white
     (cls box), or the dark box of this app (cls app, the group .dark). The
     browser test measures that each text of a group stays inside its box. */
  const inbox = (cls, x, y, w, h, rx, ...inside) => svg('g', { class: cls === 'app' ? 'inbox dark' : 'inbox' },
    svg('rect', { class: cls, x, y, width: w, height: h, rx }), inside);

  /* An arrow: the line d and, at the line's end (x, y), its head pointing
     dir (r, l, u or d): two arms back from the tip, in the same path. */
  const DIRS = { r: [1, 0], l: [-1, 0], u: [0, -1], d: [0, 1] };
  const arrow = (d, x, y, dir) => {
    const [dx, dy] = DIRS[dir];
    const bx = x - 5.25 * dx;
    const by = y - 5.25 * dy;
    return svg('path', {
      class: 'ln', d: `${d}M${bx + 3 * dy} ${by + 3 * dx}L${x} ${y}L${bx - 3 * dy} ${by - 3 * dx}`,
    });
  };

  /* One example tile with its corner at (x, y), size units wide (38 as
     drawn): its ground, its patches, then a thin edge. */
  const tile = (kind, x, y, size = 38) => svg('g', { transform: `translate(${x} ${y}) scale(${size / 38})` },
    svg('rect', { width: 38, height: 38, rx: 4, fill: TILE[kind][0] }),
    PATCHES[kind].map(([colour, d]) => svg('path', { d, fill: TILE[kind][colour] })),
    svg('rect', { class: 'tile-edge', width: 38, height: 38, rx: 4 }));

  /* A grid of count squares, cols to a row, from (x, y): each size wide,
     pitch apart; those whose place in the grid is in picked stand out. */
  const squares = (x, y, size, pitch, cols, count, picked) => Array.from({ length: count }, (_, i) => svg('rect', {
    class: picked.includes(i) ? 'sq on' : 'sq',
    x: x + pitch * (i % cols), y: y + pitch * Math.floor(i / cols), width: size, height: size, rx: 2,
  }));

  /* The line with a band and a dot, in the dark box: an axis from x0 to x1
     at y with a tick at each end, a band from b0 to b1 and a dot at dot. No
     label: a picture of "with a range", never a result. */
  const rate = (x0, x1, y, b0, b1, dot) => [
    svg('path', { class: 'axis', d: `M${x0} ${y}H${x1}M${x0} ${y - 4}V${y + 4}M${x1} ${y - 4}V${y + 4}` }),
    svg('path', { class: 'band', d: `M${b0} ${y}H${b1}` }),
    svg('circle', { class: 'dot', cx: dot, cy: y, r: 3.5 }),
  ];

  /* The docs figure, for a wide panel: a picture, named by the wrapper. */
  const wide = () => OE.el('img', {
    class: 'wide', src: PICTURE, alt: '', 'aria-hidden': 'true', draggable: 'false', width: '1500', height: '843',
  });

  /* The narrow strip, 348 x 96: Studio makes a map, you check random
     squares, you get how often it is wrong, with a range. */
  const narrow = () => svg('svg', { class: 'narrow', viewBox: '0 0 348 96', 'aria-hidden': 'true', focusable: 'false' },
    inbox('box', 1, 1, 100, 94, 8,
      text('t', 11, 21, WORDS.narrowStudio[0]), text('s', 11, 37, WORDS.narrowStudio[1]),
      tile('crops', 11, 46, 40)),
    arrow('M103 48H119', 119, 48, 'r'),
    inbox('box', 123, 1, 100, 94, 8,
      text('t', 133, 21, WORDS.narrowCheck[0]), text('s', 133, 37, WORDS.narrowCheck[1]),
      squares(133, 48, 12, 16, 4, 8, [1, 6])),
    arrow('M225 48H241', 241, 48, 'r'),
    inbox('app', 245, 1, 102, 94, 8,
      text('t', 255, 21, WORDS.narrowGet[0]),
      text('s', 255, 37, WORDS.narrowGet[1]), text('s', 255, 52, WORDS.narrowGet[2]),
      text('s', 255, 67, WORDS.narrowGet[3]),
      rate(255, 337, 82, 263, 287, 273)));

  /* The figure: one role=img element named by the title and the
     description, holding the picture and the drawing; the panel's width
     picks the one shown (STYLE). */
  const figure = () => {
    style();
    return OE.el('div', { class: 'oe-overview', role: 'img', 'aria-label': TITLE + ' ' + DESC }, wide(), narrow());
  };

  return { STYLE, style, figure, CAPTION, CAPTION_NARROW };
})();
