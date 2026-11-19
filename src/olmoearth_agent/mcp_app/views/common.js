/* Shared by every OlmoEarth Agent view (inlined by views.py).

   The page imports App from the pinned MCP Apps client before this script;
   everything here runs inside the host's sandboxed iframe and reaches the
   server only through app.callServerTool. A view never calls Studio itself,
   and never sees the API key: the sign-in opens a page the server serves on
   127.0.0.1 in the user's browser (OE.signInPanel).

   The look is OlmoEarth Studio's working pages (a MUI theme), from values
   measured with getComputedStyle on 11 November 2026:
   - one slim line on top: the AGENT tag and the sentence that this panel
     is the OlmoEarth Agent, an app inside Claude (not Studio itself); at
     its right, on the start and projects cards, the account and Sign out
     (r5-a V2);
   - the cream canvas (#faf2e9), text #0a3235, secondary text #105257;
   - cards: white, 1px rgba(0,0,0,.12), 4px corners, no shadow at rest (object
     cards 12px, raised on hover);
   - buttons: 500 14px, no capitals; one contained dark-teal button per view,
     the rest outlined (border rgba(10,50,53,.5)) or text; a small size for
     rows; grey disabled states; joined toggle groups (uppercase 500 13px);
   - type: 600 headings (20px page and card titles, 16px sections), 400 14px
     body, 400 12px/18px captions;
   - tables: 500 14px sentence-case headers, rgb(224,224,224) rules;
   - chips for kinds and statuses: rgba(0,0,0,.08), 13px, 16px corners;
   - fields: border rgba(0,0,0,.23), focus 2px #0a3235; error red #d02f2e;
   - Manrope and Roboto Mono.
   It does not follow the host's theme: the panel looks the same in a light or
   a dark Claude. */

const OE = {};

/* The AGENT tag, as a data URL (views.py fills it in from views/brand/). */
OE.TAG = '@AGENT_TAG_DATA_URL@';
OE.FONTS = 'https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700&family=Roboto+Mono:wght@400;500&display=swap';

OE.STYLE = `
:root {
  color-scheme: light;
  --canvas: #faf2e9; --paper: #ffffff;
  --text: #0a3235; --text-2: #105257; --grey: rgba(0,0,0,.6); --disabled: rgba(0,0,0,.38);
  --divider: rgba(0,0,0,.12); --rule: rgb(224,224,224); --field: rgba(0,0,0,.23);
  --outline: rgba(10,50,53,.5); --wash: rgba(0,0,0,.04); --wash-2: rgba(0,0,0,.08);
  --selected: rgba(10,50,53,.08); --primary-dark: rgb(7,35,37);
  --mint: #0fcb8c; --error: #d02f2e;
  --shadow-btn: 0 3px 1px -2px rgba(0,0,0,.2), 0 2px 2px 0 rgba(0,0,0,.14), 0 1px 5px 0 rgba(0,0,0,.12);
  --shadow-2: 0 3px 1px -2px rgba(0,0,0,.2), 0 2px 2px 0 rgba(0,0,0,.14), 0 1px 5px 0 rgba(0,0,0,.12);
  --font: "Manrope", ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  --mono: "Roboto Mono", ui-monospace, "SF Mono", Menlo, Consolas, monospace;
}
* { box-sizing: border-box; }
html { background: var(--canvas); }
body { margin: 0; padding: 0 16px 16px; background: var(--canvas); color: var(--text);
  font-family: var(--font); font-size: 14px; line-height: 21px; font-weight: 400;
  -webkit-font-smoothing: antialiased; text-rendering: optimizeLegibility; }
* { scrollbar-width: thin; scrollbar-color: rgba(0,0,0,.26) transparent; }
/* One slim line on top, edge to edge: the AGENT tag and the sentence, a hairline under them. */
.oe-line { display: flex; align-items: center; gap: 8px; margin: 0 -16px 12px; padding: 6px 16px;
  border-bottom: 1px solid var(--divider); font-size: 12px; line-height: 18px; color: var(--text-2); }
.oe-line img { height: 14px; flex: none; display: block; }
/* The line's words (r5-a V2): the sentence, then the account item at its
   right (OE.setAccount: who, and a Sign out text button sized as the line);
   on a narrow panel the item drops under the sentence, from its left edge.
   Empty (signed out, and on the views that set none) the item is not drawn. */
.oe-line-text { flex: 1 1 0; min-width: 0; display: flex; flex-wrap: wrap; align-items: flex-start;
  justify-content: space-between; gap: 2px 12px; }
.oe-said { flex: 1 1 260px; min-width: 0; }
.oe-account { flex: 0 1 auto; min-width: 0; max-width: 100%; display: inline-flex; align-items: center; gap: 4px; }
.oe-account:empty { display: none; }
.oe-account .who { min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
  color: var(--text); font-weight: 600; }
.oe-account button.link { flex: none; font-size: 12px; line-height: 18px; padding: 0 4px; }
/* The address of a link the host did not open (OE.docsLink, OE.accessLink), whole and selectable. The slot is a
   status in place before the first press (r5-i B5): empty, it is drawn but takes no height (never display none),
   so it is in the accessibility tree before its first words, as the toast's region and OE.typeSlot's line are,
   and a screen reader hears them. */
.oe-link-slot code { user-select: all; overflow-wrap: anywhere; }
h1, h2, h3 { margin: 0; }
h1 { font-size: 20px; line-height: 28px; font-weight: 600; color: var(--text-2); }
h2 { font-size: 16px; line-height: 28px; font-weight: 600; color: var(--text); }
h3 { font-size: 14px; line-height: 21px; font-weight: 600; color: var(--text); }
.muted { color: var(--text-2); }
.faint { color: var(--grey); }
.small { font-size: 12px; line-height: 18px; }
.overline { font-size: 12px; line-height: 18px; font-weight: 400; text-transform: uppercase;
  letter-spacing: .06em; color: var(--text-2); }
.row { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
.spread { display: flex; gap: 8px; align-items: center; justify-content: space-between; flex-wrap: wrap; }
.stack > * + * { margin-top: 12px; }
.card { background: var(--paper); border: 1px solid var(--divider); border-radius: 4px; padding: 16px; }
/* r5-e: one rule for every group of same-kind controls: equal columns, every row as tall as the tallest. */
.oe-same, .face .answers { display: grid; grid-template-columns: repeat(auto-fill, minmax(var(--same-min, 136px), 1fr)); grid-auto-rows: 1fr; gap: 8px; align-items: stretch; }
.oe-same > *, .face .answers > * { min-width: 0; }
/* r5-e: a joined group or pair: every member as wide as the widest. */
.toggle, .face-top > .row { display: inline-grid; grid-auto-flow: column; grid-auto-columns: 1fr; }
.card.object { border-radius: 12px; cursor: pointer; transition: box-shadow .3s cubic-bezier(.4,0,.2,1), background .3s; }
.card.object:hover { box-shadow: var(--shadow-2); background: linear-gradient(var(--wash), var(--wash)), var(--paper); }
.card.summary { background: var(--canvas); }
/* A Studio dialog without its modal parts: the title 16px 24px, the content 0 24px 20px. */
.card.dialog { padding: 0 24px 20px; }
.card.dialog > h2:first-child { font-size: 20px; line-height: 28px; margin: 0 -24px; padding: 16px 24px; }
.card.dialog > h2:first-child + * { margin-top: 0; }
/* The sign-in's note (OE.signInPanel, r5-b J19): a few plain lines, each its own. */
.oe-sign-note > div + div { margin-top: 4px; }
.error { color: var(--error); }
.card.error { border-color: rgba(208,47,46,.5); }
.ok { color: var(--text); }
a { color: var(--text-2); font-weight: 700; text-decoration: none; }
/* Buttons: Studio's MUI buttons. Outlined is the default; .primary is the one
   contained button of a view; .small for rows; .link is a text button. */
button { font: inherit; font-size: 14px; line-height: 24.5px; font-weight: 500; cursor: pointer;
  padding: 5px 15px; border-radius: 4px; border: 1px solid var(--outline);
  background: transparent; color: var(--text); transition: background .25s, border-color .25s, box-shadow .25s; }
button:hover { background: rgba(10,50,53,.04); border-color: var(--text); }
button.small { font-size: 13px; line-height: 22.75px; padding: 3px 9px; }
button.primary { background: var(--text); color: var(--canvas); border: 0; padding: 6px 16px; box-shadow: var(--shadow-btn); }
button.primary.small { padding: 4px 10px; }
button.primary:hover { background: var(--primary-dark); }
button:disabled, button:disabled:hover { cursor: default; background: transparent; color: rgba(0,0,0,.26);
  border-color: var(--divider); box-shadow: none; }
button.primary:disabled, button.primary:disabled:hover { background: rgba(0,0,0,.12); color: rgba(0,0,0,.26); }
/* Busy (aria-disabled): looks disabled, keeps the keyboard focus. */
button[aria-disabled="true"], button[aria-disabled="true"]:hover { cursor: default; background: transparent;
  color: rgba(0,0,0,.26); border-color: var(--divider); box-shadow: none; }
button.primary[aria-disabled="true"], button.primary[aria-disabled="true"]:hover { background: rgba(0,0,0,.12); }
button.link { border: 0; background: none; padding: 4px 5px; border-radius: 4px; box-shadow: none;
  font-size: 13px; line-height: 22.75px; font-weight: 500; color: var(--text-2); }
button.link:hover { background: var(--wash); }
/* A text button disabled or busy greys too (the rules above lose to .link's). */
button.link:disabled, button.link:disabled:hover,
button.link[aria-disabled="true"], button.link[aria-disabled="true"]:hover {
  cursor: default; color: rgba(0,0,0,.26); background: none; }
/* A joined toggle group (Studio's LIST VIEW / CARD VIEW). The pressed one is
   told by more than its pale fill (r5-b J16): 700, and a dark line inside
   its edge. Its members are as wide as the widest (the r5-e rule after
   .card); the -1px joins still work in its grid. */
.toggle button { border-radius: 0; border: 1px solid var(--divider); padding: 7px; margin-left: -1px;
  font-size: 13px; line-height: 22.75px; font-weight: 500; text-transform: uppercase; color: var(--grey); }
.toggle button:first-child { margin-left: 0; border-radius: 4px 0 0 4px; }
.toggle button:last-child { border-radius: 0 4px 4px 0; }
.toggle button:hover { background: var(--wash); border-color: var(--divider); }
.toggle button[aria-pressed="true"] { background: var(--selected); color: var(--text); font-weight: 700;
  box-shadow: inset 0 0 0 2px var(--text); }
.toggle.small button { padding: 3px 9px; }
/* A label sheet row's group of answers takes the focus after an answer
   (tabindex -1, not a button): a ring when the keyboard brought it there. */
.toggle[tabindex]:focus { outline: none; }
.toggle[tabindex]:focus-visible { outline: 2px solid var(--text); outline-offset: 2px; border-radius: 4px; }
/* r5-e: on a panel under 480px a list row's answers (the one joined toggle in
   a table cell) stand one under the other, each as wide as the widest: side
   by side they did not fit their cell there, and each column fell back to
   its own words' width. */
@media (max-width: 480px) {
  td > .toggle { grid-auto-flow: row; grid-auto-rows: 1fr; }
  td > .toggle button { margin-left: 0; margin-top: -1px; border-radius: 0; }
  td > .toggle button:first-child { margin-top: 0; border-radius: 4px 4px 0 0; }
  td > .toggle button:last-child { border-radius: 0 0 4px 4px; }
}
/* Chips for kinds and statuses, never for an error rate. */
.chip { display: inline-flex; align-items: center; gap: 6px; height: 24px; padding: 0 10px; border-radius: 16px;
  background: var(--wash-2); font-size: 13px; line-height: 19.5px; font-weight: 400; color: var(--text); white-space: nowrap; }
.chip.ring::before { content: ""; width: 10px; height: 10px; border-radius: 50%; border: 2px solid var(--mint); }
/* Fields: Studio's dense outlined fields. */
input[type=text], input[type=password], input[type=search], select {
  font: inherit; font-size: 14px; line-height: 21px; padding: 5px 13px; border-radius: 4px;
  border: 1px solid var(--field); background: var(--paper); color: var(--text); }
select { padding-right: 32px; }
input:focus, select:focus { outline: none; border-color: var(--text); box-shadow: inset 0 0 0 1px var(--text); }
button:focus-visible { outline: 2px solid var(--text); outline-offset: 2px; }
/* A field's grey hint, dark enough to read (r5-b J29: .6, before .42; r5-f
   F03: .7, 5.3:1 on a white field: at .6 it was 4.0:1, under 4.5:1). */
input::placeholder { color: rgba(10,50,53,.7); }
.search { border-radius: 32px; padding: 8px 16px 8px 38px; width: 100%; background-repeat: no-repeat;
  background-position: 14px center; background-size: 16px;
  background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'><path fill='%230a3235' d='M15.5 14h-.8l-.3-.3A6.5 6.5 0 1 0 14 15.5l.3.3v.8l5 5 1.5-1.5-5-5zm-6 0a4.5 4.5 0 1 1 0-9 4.5 4.5 0 0 1 0 9z'/></svg>"); }
.field-label { display: block; font-size: 12px; line-height: 18px; color: var(--grey); margin-bottom: 2px; }
input[type=checkbox] { accent-color: var(--text-2); width: 16px; height: 16px; }
/* Tables: Studio's grid. */
.table-frame { border: 1px solid var(--rule); border-radius: 4px; overflow: auto; background: var(--paper); }
table { border-collapse: collapse; width: 100%; }
th, td { text-align: left; padding: 6px 12px; vertical-align: middle; border-bottom: 1px solid var(--rule); }
th { font-size: 14px; line-height: 21px; font-weight: 500; color: var(--text); background: var(--paper); position: sticky; top: 0; }
tr:last-child td { border-bottom: 0; }
code, .mono { font-family: var(--mono); font-size: 12px; line-height: 18px; color: var(--text); }
ul { margin: 4px 0; padding-left: 20px; }
/* Key and value rows (Studio's Project Info). */
.kv { display: grid; grid-template-columns: 1fr; gap: 8px 16px; }
.kv div > span:first-child { display: block; font-size: 12px; line-height: 18px; color: var(--grey); }
.kv div > span:last-child { font-size: 14px; line-height: 21px; overflow-wrap: anywhere; }
/* A collapsible section header with a chevron at the right. */
details > summary { list-style: none; cursor: pointer; display: flex; align-items: center; justify-content: space-between;
  padding: 4px 0; font-size: 12px; line-height: 24px; font-weight: 700; text-transform: uppercase; color: var(--text-2); }
details > summary::-webkit-details-marker { display: none; }
details > summary::after { content: ""; width: 7px; height: 7px; border-right: 2px solid var(--text-2);
  border-bottom: 2px solid var(--text-2); transform: rotate(45deg); margin-right: 4px; transition: transform .2s; }
details[open] > summary::after { transform: rotate(-135deg); }
.footnote { font-size: 12px; line-height: 18px; color: var(--text); }
/* A card's plain lines under its heading (OE.lead): at most two, in the body
   colour; the rest goes in the card's Details fold (r5-e). */
.oe-lead > p { margin: 0; }
.oe-lead > p + p { margin-top: 4px; }
/* The rest of a card in a closed fold (OE.fold), spaced as a stack once open. */
.oe-more > summary + * { margin-top: 8px; }
.oe-more > * + * { margin-top: 12px; }
/* What became of a job button's message (OE.sendJob, r5-b J20), on a line of
   its own under the row: that it was sent, or that it was not, over the
   message, whole and selectable, to type in the chat. Until a press its
   status line is empty, and the slot is out of the layout (no gap in a grid,
   a row or a stack) but drawn, so the line is in the accessibility tree
   before its first words and a screen reader hears them. */
.oe-type { flex-basis: 100%; margin-top: 4px; }
.oe-type:has(> [role=status]:empty) { position: absolute; width: 1px; height: 1px; margin: 0; overflow: hidden;
  clip-path: inset(50%); }
.oe-type code { display: block; margin-top: 2px; padding: 6px 8px; background: var(--paper); border: 1px solid var(--divider);
  border-radius: 4px; user-select: all; overflow-wrap: anywhere; white-space: pre-wrap; }
/* The label sheet's footer: sticky, a hairline above, the counter and the one button. */
/* No scroll anchoring on a sheet (r5-f F01, after the check): when the pictures pane under the answers grows,
   the browser kept a node below it in place and scrolled the face's heading and answers away, and undid the
   scroll that brings the question into view. */
section.sheet { overflow-anchor: none; }
.sheet-footer { position: sticky; bottom: 0; margin: 16px -16px -16px; padding: 12px 16px;
  border-top: 1px solid var(--divider); background: var(--paper); border-radius: 0 0 4px 4px; }
/* The footer's row with the finish button keeps to the right edge, also when
   the line beside the button makes it wrap under the counter (r5-b J29), and
   when that line and the button wrap inside it on a narrow panel. div.row:
   the counter is a span.row, and stays at the left. */
.sheet-footer > div.row { margin-left: auto; justify-content: flex-end; }
.sheet-footer > div.row > .row { justify-content: flex-end; }
/* The label sheet's table and pictures pane: the pane under the table, or
   beside it while it is open on a panel at least 760px wide, the table's
   cells and answer buttons then a little narrower so the answers stay in view. */
.sheet-work > * + * { margin-top: 12px; }
@media (min-width: 760px) {
  .sheet-work.pane { display: grid; grid-template-columns: minmax(0,1fr) minmax(300px,42%); gap: 16px; align-items: start; }
  .sheet-work.pane > * + * { margin-top: 0; }
  .sheet-work.pane > .stack { position: sticky; top: 0; }
  .sheet-work.pane th, .sheet-work.pane td { padding-left: 8px; padding-right: 8px; }
  .sheet-work.pane .toggle.small button { padding-left: 6px; padding-right: 6px; }
  /* r5-e: beside the open pane a row's answers keep their words' widths (not
     the equal columns of a joined toggle), so the row stays one line. */
  .sheet-work.pane td > .toggle { grid-auto-columns: auto; }
}
/* The label sheet's face: one window at a time. Inline it is a pager, the
   heading, Previous and Next and the answers on top and the dates in a strip
   that scrolls sideways; in the host's full screen (.full) the dates are a
   grid. An answer given shows pressed; none is the contained button. The
   face takes the focus after each answer: a ring when the keyboard brought
   it there, as the list's group of answers has. In the strip (r5-b J24) each
   date starts at the top, so a month without a picture is not stretched
   beside the real ones, and none is wider than the strip, so a date's row
   wraps Detail under Context on a narrow panel. A strip of several dates
   keeps each 40px narrower than itself, so the edge of the next one stays in
   view as a sign that the strip goes on; a lone date, and each cell of the
   full screen grid, keep the whole width. Everything about the one square
   sits in the face's own box (r5-e: a hairline edge, before only a focus
   ring, which touched the text), the ring 2px outside it; on a panel of
   420px or less its sides keep 6px inside, not 12px (the rule after it). */
.face { border: 1px solid var(--rule); border-radius: 4px; padding: 12px; }
.face:focus { outline: none; }
.face:focus-visible { outline: 2px solid var(--text); outline-offset: 2px; }
/* The rule's box right above the face (r5-e H2) keeps the face's inset, so
   the rule's words and the square's heading start at one left edge
   (.card.oe-rule beats .card). */
.card.oe-rule { padding: 12px; }
/* r5-e C4: on a narrow panel the face keeps 6px at its sides (and the rule's
   box with it, so the two still start at one edge), so a date of the strip
   still holds a 240px picture (imagery._SCREEN): 380 - 32 - 34 - 2 - 12 =
   300, less 40 and the date's 18 = 242. */
@media (max-width: 420px) { .face, .card.oe-rule { padding: 12px 6px; } }
.face-dates:focus { outline: none; }
.face-dates:focus-visible { outline: 2px solid var(--text); outline-offset: 2px; border-radius: 4px; }
.face-top { display: flex; flex-wrap: wrap; gap: 12px; align-items: center; justify-content: space-between; }
.face-dates { display: flex; gap: 12px; overflow-x: auto; align-items: flex-start; }
.face-dates > * { flex: none; max-width: 100%; }
.face-dates > :not(:only-child) { max-width: calc(100% - 40px); }
.face.full .face-dates { display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); overflow: visible; }
.face.full .face-dates > * { max-width: none; }
.face .answers > select { grid-column: span 2; }
.face .answers button { min-width: 0; padding: 10px 12px; }
.face .answers button.oe-other { border-style: dashed; color: var(--text-2); }
/* r5-e: a class name of one long word breaks inside its cell (the cell's
   width is the grid's, no longer the words'); it does not run over the
   button's edge into the next cell. */
.face .answers button { overflow-wrap: anywhere; }
/* r5-e: where only one column fits (the face's inside under 280px: on the
   label sheet a panel under 360px), the list takes one cell: a span of 2
   would add a second column of another width. */
@media (max-width: 359.98px) { .face .answers > select { grid-column: auto; } }
/* Only a chosen answer looks chosen (r4-h A/F22): no hover fill or border
   change on an answer not chosen. A chosen one (r5-b J16) has the pale
   selected fill, a dark line, 700 and a tick before its words, the look of
   a chosen number on the Check card; the dark fill is the view's one
   action's alone (before, r4-h filled a chosen answer dark too). The tick
   has empty alt text, so the button's name stays its words and a screen
   reader says "pressed" once (a browser without that form keeps the first
   content, the tick alone). */
.face .answers button:not([aria-pressed="true"]):not([aria-disabled="true"]):not(:disabled):hover {
  background: transparent; border-color: var(--outline); }
.face .answers button[aria-pressed="true"] { background: var(--selected); color: var(--text); border-color: var(--text);
  box-shadow: inset 0 0 0 1px var(--text); font-weight: 700; }
.face .answers button[aria-pressed="true"]::before { content: "\\2713\\00a0"; content: "\\2713\\00a0" / ""; }
`;

/* HTML-escape any value. */
OE.esc = (value) => String(value ?? '').replace(/[&<>"']/g, (c) => (
  { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

/* Make an element: OE.el('button', {class: 'primary', onclick}, 'Save'). */
OE.el = (tag, attrs = {}, ...children) => {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs || {})) {
    if (value === undefined || value === null || value === false) continue;
    if (key.startsWith('on') && typeof value === 'function') node.addEventListener(key.slice(2), value);
    else if (key === 'class') node.className = value;
    else node.setAttribute(key, value === true ? '' : String(value));
  }
  for (const child of children.flat()) {
    if (child === undefined || child === null || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
};

/* A name's words as children for OE.el, with a place to break after each
   underscore: a wbr element, which holds no character, so the text stays as
   it was written (r5-f F14: on a label sheet's answer a class name such as
   "evergreen_needleleaf_woodland" breaks after an underscore first, no
   longer in mid-word; the overflow-wrap of .face .answers button stays for
   a word with none). The split uses no lookbehind, which an older browser
   would refuse for the whole script. */
OE.softBreaks = (text) => {
  const parts = String(text).split('_');
  const out = [];
  parts.forEach((part, i) => {
    const piece = i < parts.length - 1 ? part + '_' : part;
    if (!piece) return;
    if (out.length) out.push(document.createElement('wbr'));
    out.push(piece);
  });
  return out;
};


/* What a tool result carries for its view: the server puts it in the result's
   _meta (not read by the model); an app tool's result has it as structuredContent. */
OE.viewData = (params) => (params && params._meta && params._meta['olmoearth/view'])
  || (params && params.structuredContent) || {};

/* Call one of the server's app tools; its structured result, or an Error with
   the server's own message, needsSignIn when the key is missing or refused,
   and stale when a save was refused because the sheet was saved elsewhere
   since the panel read it (sheet.StaleSheet: read it again, then save).
   fromServer: the server answered and refused; an Error without it is a call
   that got no answer (the host's timeout, a dropped bridge), so the server
   may have done the work. options go to app.callServerTool as they are
   ({ timeout } in ms, for a call that may run long). */
OE.call = async (app, name, args = {}, options = undefined) => {
  const result = await app.callServerTool({ name, arguments: args }, options);
  if (result.isError) {
    let body = result.structuredContent;
    if (!body) {
      const text = (result.content || []).map((c) => c.text || '').join(' ');
      try { body = JSON.parse(text); } catch (_) { body = { error: text }; }
    }
    const err = new Error((body && body.error) || 'The call failed.');
    err.needsSignIn = Boolean(body && body.needs_sign_in);
    err.stale = Boolean(body && body.stale);
    err.fromServer = true;
    throw err;
  }
  return result.structuredContent || {};
};

/* Send the user's words to Claude, as if typed (a button the user pressed).
   A host that declines answers isError rather than throwing. True when the
   host took the message, false when it did not (after a toast, unless
   quiet). A message taken clears the status line while it still shows the
   failure line this function wrote last (r5-b J20), so a card never says
   that a message was sent beside a line that says one was not; any other
   line stays. quiet (r5-d R07): a press whose own slot says what became of
   the message (OE.sendJob) writes no failure line, so a message not taken
   is said once, beside its button, and a screen reader hears it once; a
   press with no slot ("Tell Claude: this project", the sign-in's "Continue
   in the chat") keeps the toast. */
// WORDING: r5b J20
OE.NOT_TAKEN = 'Claude did not take the message; type it in the chat instead.';
let saidFailed = null;  // the failure line OE.say last wrote
OE.say = async (app, text, { quiet = false } = {}) => {
  try {
    const answer = await app.sendMessage({ role: 'user', content: [{ type: 'text', text }] });
    if (answer && answer.isError) {
      if (!quiet) {
        saidFailed = OE.NOT_TAKEN;
        OE.toast(saidFailed, true);
      }
      return false;
    }
    const bar = document.getElementById('oe-toast');
    if (bar && saidFailed && bar.textContent === saidFailed) OE.toast('');
    saidFailed = null;
    return true;
  } catch (err) {
    if (!quiet) {
      saidFailed = 'Could not send the message to Claude: ' + err.message;
      OE.toast(saidFailed, true);
    }
    return false;
  }
};

/* What a job button's slot says after a press (r5-b J20): that the message
   went to the chat, or that it did not, over the message. r5-f F08: "Copy
   it into the chat" (it said "Type it in the chat": nobody retypes ten lines
   of ids, and the box under the line selects whole on one click). */
// WORDING: r5b J20
OE.JOB_SENT = 'Sent to the chat. Claude answers there.';
// WORDING: r5f F08
OE.JOB_NOT_SENT = 'Claude did not get this message. Copy it into the chat:';

/* Where a job button's press says what became of its message (OE.sendJob):
   its first child is its status line (r5-b J20), the slot's one live part,
   so a screen reader hears that line and not the message to copy under it.
   Empty, and kept out of the layout but in the accessibility tree, until a
   press, so a screen reader hears its first line. */
OE.typeSlot = () => OE.el('div', { class: 'oe-type small' }, OE.el('div', { role: 'status' }));

/* Empty a job button's slot (OE.typeSlot), its status line kept in place for
   the next press to write in: a press that ends before it sends anything
   leaves no word of the one before (r5-b J20). Returns the line. */
OE.clearSlot = (slot) => {
  const line = slot.firstElementChild;
  while (line.nextSibling) line.nextSibling.remove();
  line.textContent = '';
  return line;
};

/* A job button's press: its request goes to Claude as the user's words, and
   slot says what became of it beside the button (r5-b J20), so a press never
   seems to have done nothing and the reason for a text to copy is where the
   text is. Sent: the line says OE.JOB_SENT, muted. Not sent (the host
   declined it): OE.JOB_NOT_SENT in the error colour, then, outside the line,
   the whole text, selectable, so its ids are not lost. Each press replaces
   what the one before left; a press that ends before sending clears it
   (OE.clearSlot). With a slot, OE.say writes no toast (quiet, r5-d R07):
   the slot says it; a toast an earlier press without a slot wrote goes once
   a later message is taken. */
OE.sendJob = async (app, text, slot) => {
  const sent = await OE.say(app, text, { quiet: Boolean(slot) });
  if (slot) {
    const line = OE.clearSlot(slot);
    if (!sent) slot.append(OE.el('code', {}, text));
    line.className = sent ? 'muted' : 'error';
    line.textContent = sent ? OE.JOB_SENT : OE.JOB_NOT_SENT;
  }
  return sent;
};

/* Open a link in the user's browser; false when the host declined. The
   host's own error is never shown (r4-k K7): it may repeat the link, and a
   street map link holds a place. A declined link writes LINK_NOT_OPENED in
   the status line; a later link opened clears that line while it still
   shows it (r5-g G2, as OE.say does, r5-b J20), so a card never says a link
   was not opened after one was; any other line stays. With quiet (r5-i B5:
   the docs and Request access links, whose own slot says it beside the
   link) a declined link writes no toast; a street map link, which has no
   slot, keeps it. */
// WORDING: r4k K7
OE.LINK_NOT_OPENED = 'The link was not opened.';
OE.open = async (app, url, { quiet = false } = {}) => {
  try {
    const answer = await app.openLink({ url });
    if (answer && answer.isError) { if (!quiet) OE.toast(OE.LINK_NOT_OPENED, true); return false; }
    const bar = document.getElementById('oe-toast');
    if (bar && bar.textContent === OE.LINK_NOT_OPENED) OE.toast('');
    return true;
  } catch (_) {
    if (!quiet) OE.toast(OE.LINK_NOT_OPENED, true);
    return false;
  }
};

/* The OlmoEarth docs' address (public and fixed): what the start card's
   docs link opens (r5-a V1). */
OE.DOCS_URL = 'https://docs.olmoearth.allenai.org/';

/* The docs link's words: its label, and what its slot says over the address
   when the host did not open it. */
// WORDING: r5a V1
OE.DOCS_LABEL = 'What is OlmoEarth? (docs)';
// WORDING: r5a V1
OE.DOCS_BY_HAND = 'Open this address in your browser:';

/* A text button that opens the OlmoEarth docs in the user's browser
   (OE.open), and its slot: [button, slot]. A button, as every link of a
   view is: the host opens the address, the page never does. aria-disabled
   while the host is asked, so it keeps the focus. When the host does not
   open it, the slot says so itself (r5-i B5: OE.LINK_NOT_OPENED, then
   OE.DOCS_BY_HAND, one line) and shows the address, whole and selectable:
   it is public and fixed, no id and no place in it. OE.open writes no toast
   for it (quiet), so the panel's foot does not say it a second time, far
   from the link; the slot is a status, so a screen reader hears it. A later
   press that opens it empties the slot. */
OE.docsLink = (app) => {
  const slot = OE.el('div', { class: 'oe-link-slot small', role: 'status' });
  let busy = false;
  const button = OE.el('button', { type: 'button', class: 'link' }, OE.DOCS_LABEL);
  button.addEventListener('click', async () => {
    if (busy) return;
    busy = true;
    button.setAttribute('aria-disabled', 'true');
    try {
      const opened = await OE.open(app, OE.DOCS_URL, { quiet: true });
      // WORDING: r5i B5
      slot.replaceChildren(...(opened ? [] : [OE.el('div', {}, OE.LINK_NOT_OPENED + ' ' + OE.DOCS_BY_HAND), OE.el('code', {}, OE.DOCS_URL)]));
    } finally {
      busy = false;
      button.setAttribute('aria-disabled', 'false');
    }
  });
  return [button, slot];
};

/* OlmoEarth's own site (public and fixed), whose "Request access" asks for
   a Studio account: what the start card's signed-out note opens for a
   person with none (r5-g G2, the owner's words of 18 Nov 2026). The label
   is the site's own words for its button. */
OE.ACCESS_URL = 'https://olmoearth.allenai.org/';
// WORDING: r5g
OE.ACCESS_LABEL = 'Request access';

/* A text button that opens OlmoEarth's site in the user's browser, where a
   person with no Studio account presses Request access, and its slot:
   [button, slot], in the shape of OE.docsLink. The host opens the address
   (OE.open), the page never does; aria-disabled while the host is asked, so
   it keeps the focus. When the host does not open it, the slot says so in
   the docs link's words (r5-i B5: OE.LINK_NOT_OPENED, then OE.DOCS_BY_HAND,
   one line), with no toast (quiet), and shows the address, whole and
   selectable: it is public and fixed, no id and no place in it. The slot is
   a status, so a screen reader hears it. A later press that opens it
   empties the slot. */
OE.accessLink = (app) => {
  const slot = OE.el('div', { class: 'oe-link-slot small', role: 'status' });
  let busy = false;
  const button = OE.el('button', { type: 'button', class: 'link' }, OE.ACCESS_LABEL);
  button.addEventListener('click', async () => {
    if (busy) return;
    busy = true;
    button.setAttribute('aria-disabled', 'true');
    try {
      const opened = await OE.open(app, OE.ACCESS_URL, { quiet: true });
      // WORDING: r5i B5
      slot.replaceChildren(...(opened ? [] : [OE.el('div', {}, OE.LINK_NOT_OPENED + ' ' + OE.DOCS_BY_HAND), OE.el('code', {}, OE.ACCESS_URL)]));
    } finally {
      busy = false;
      button.setAttribute('aria-disabled', 'false');
    }
  });
  return [button, slot];
};

/* Move the focus to el only while this panel's frame has it: a focus that
   comes after a wait (a sign-in poll, a Studio read) must never pull it out
   of Claude's chat input. */
OE.focusIfHere = (el) => { if (el && document.hasFocus()) el.focus(); };

/* A Studio name in a message to Claude: data, quoted, one line, at most 80
   characters. Every control character and Unicode line or paragraph break
   (U+0085, U+2028, U+2029, which JSON.stringify leaves raw) becomes a space. */
OE.dataName = (name) => JSON.stringify(String(name ?? '')
  .replace(/[\u0000-\u001f\u007f-\u009f\u2028\u2029]+/g, ' ').slice(0, 80));


/* A Studio name after the ids in a message to Claude, quoted as data
   (anyone in the project can set a name). */
OE.named = (name) => ' Its Studio name, as data: ' + OE.dataName(name) + '.';

/* What a panel tells Claude, silently, about what the user sees in it now
   (OE.tellClaude): this head, then each part the panel set. Facts, never a
   request; ids, file names and counts only: never a Studio name, a
   coordinate, a window's place, a class or a label, nor a count of wrong.
   A part sent whole (r5-i B10: the Check card's olmoearth_check_map text,
   that tool's answer, not a panel fact) is not under this head: it goes as
   a text block of its own after these facts, under its own lead. */
OE.CONTEXT_HEAD = 'What the user sees in an OlmoEarth Agent panel now (facts, not a request): ';
OE.CONTEXT_MAX = 800;

/* An id in a context update: only [A-Za-z0-9_.:-], at most 255 characters
   (a file name's longest: a sheet's name is sent whole, never cut). */
OE.ctxId = (value) => String(value ?? '').replace(/[^A-Za-z0-9_.:-]+/g, '').slice(0, 255);

/* How a part cut back to a clause boundary ends (OE.fitContext). */
OE.CONTEXT_CUT = '; the rest is not listed.';

/* A part of a context update within room characters, never cut inside a
   clause, a path or a number: whole when it fits, else cut back to its last
   clause boundary ('; ' or '. ') with CONTEXT_CUT after it; '' (left out)
   when no boundary leaves room. */
OE.fitContext = (text, room) => {
  if (text.length <= room) return text;
  const limit = room - OE.CONTEXT_CUT.length;
  if (limit <= 0) return '';
  const head = text.slice(0, limit + 2);
  const at = Math.max(head.lastIndexOf('; '), head.lastIndexOf('. '));
  return at > 0 ? text.slice(0, at) + OE.CONTEXT_CUT : '';
};

/* A part made of clauses (a zone, a sheet in progress, each whole), within
   room: head, the clauses joined by '; ', then tail. Over the room, whole
   clauses go from the end, and more(k), a clause of its own, says how many
   are not listed. With not even one clause left it is head, more(all) and
   tail, which OE.fitContext cuts back or leaves out if that is still too long. */
OE.fitClauses = (head, clauses, tail, room, more) => {
  const join = (list) => head + list.join('; ') + tail;
  for (let i = clauses.length; i > 0; i -= 1) {
    const text = join(i === clauses.length ? clauses : [...clauses.slice(0, i), more(clauses.length - i)]);
    if (text.length <= room) return text;
  }
  return join(clauses.length ? [more(clauses.length)] : []);
};

/* The parts this page told Claude last, by name: the panel's own line
   ('panel', the default) and the label editor's ('sheet'), which goes first.
   Each update replaces the view's last one, so every update carries them all. */
OE.contextParts = new Map();
OE.contextSent = null;
/* The parts sent whole (OE.tellClaude with { whole: true }), by name: each
   a text block of its own after the facts, never cut and outside
   CONTEXT_MAX (r5-i B10: the Check card's 'check'). One longer than
   CONTEXT_WHOLE_MAX is left out, never cut: its caller sends shorter words
   then (check.html READY_TOLD_LONG). */
OE.contextWhole = new Map();
OE.CONTEXT_WHOLE_MAX = 12000;

/* Tell Claude what this panel shows now (ui/update-model-context), as one
   part of its update: text replaces that part ('' removes it). text is a
   string, or a function of the room left for it (in characters) that gives
   the longest text of the part that fits (the Check card's zones, the
   projects panel's sheets: OE.fitClauses). Sent unless the host says it
   takes no context updates (r4-k K3): its capabilities carry
   updateModelContext true, or no such key at all, since the stable spec
   (2026-01-26) defines ui/update-model-context but no flag for it; skipped
   only where the key is there and false, and before the host connected (no
   capabilities yet). Control characters and Unicode line or paragraph
   breaks become spaces. The parts go in order ('sheet' first), each in the
   room the ones before it left within CONTEXT_MAX, cut only at a clause
   boundary (OE.fitContext), never inside a path or a number. With
   opts.whole (a string; r5-i B10) the part goes whole instead, as a text
   block of its own after those facts, without CONTEXT_HEAD, in every later
   update too (OE.contextWhole); an update of such parts alone has no facts
   block. The update is not sent again when it is the one sent last (all
   its blocks). Never a toast, never the
   focus: every failure is swallowed, and a host that never answers costs
   nothing. */
OE.tellClaude = (app, text, part = 'panel', opts = {}) => {
  try {
    if (!app || typeof app.updateModelContext !== 'function') return;
    const caps = typeof app.getHostCapabilities === 'function' ? app.getHostCapabilities() : null;
    if (!caps || typeof caps !== 'object') return;
    if ('updateModelContext' in caps && !caps.updateModelContext) return;
    const cleaned = (value) => String(value ?? '').replace(/[\u0000-\u001f\u007f-\u009f\u2028\u2029]+/g, ' ').trim();
    OE.contextWhole.delete(part);
    if (typeof text === 'function') OE.contextParts.set(part, text);
    else if (cleaned(text) && opts && opts.whole === true) {
      OE.contextParts.delete(part);
      OE.contextWhole.set(part, cleaned(text));
    } else if (cleaned(text)) OE.contextParts.set(part, cleaned(text));
    else OE.contextParts.delete(part);
    if (!OE.contextParts.size && !OE.contextWhole.size) return;
    const order = [...OE.contextParts.keys()].sort((a, b) => Number(b === 'sheet') - Number(a === 'sheet'));
    let room = OE.CONTEXT_MAX - OE.CONTEXT_HEAD.length;
    const said = [];
    for (const key of order) {
      const value = OE.contextParts.get(key);
      const space = said.length ? 1 : 0;
      const one = OE.fitContext(cleaned(typeof value === 'function' ? value(room - space) : value), room - space);
      if (!one) continue;
      said.push(one);
      room -= space + one.length;
    }
    const blocks = [...OE.contextWhole.values()].filter((value) => value.length <= OE.CONTEXT_WHOLE_MAX);
    if (!said.length && !blocks.length) return;
    const content = [...(said.length ? [OE.CONTEXT_HEAD + said.join(' ')] : []), ...blocks]
      .map((value) => ({ type: 'text', text: value }));
    const whole = JSON.stringify(content);
    if (whole === OE.contextSent) return;
    OE.contextSent = whole;
    Promise.resolve(app.updateModelContext({ content })).catch(() => {});
  } catch (_) { /* nothing on screen changes */ }
};

/* "1 area", "3 areas": a count and its noun. */
OE.counted = (n, noun) => n + ' ' + noun + (n === 1 ? '' : 's');

/* What each job button sends: one complete request with a verb, the ids
   Claude's tools take and the Studio names quoted as data. r is
   { result_id, prediction_id, name }; m is a model with its project_id. A
   check opens the map's Check card (olmoearth_check_map), where the user
   picks how many squares to check (the owner's ruling 6: "squares", not
   "windows", in what the user sees and sends); the plan of two maps needs a
   label budget, so its request asks for it; the map lets the user draw an
   area or choose a saved one. points is the Check card's button on a class
   map without scores (r4-h C6, E/check-1): a check on single points of a
   grid, the result's id (and the prediction's, when known) and no Studio
   name; Claude asks how many points and which way to pick them first. The
   one source for any later sender of that request. byClass is the result
   card's button under its uncovered lines (r4-i O3): r is { result_id,
   part_file, unit }, a new check of that result (and of the part in that
   file, on a map in several parts) picked class by class, its own check
   never pooled with this one; Claude asks how many squares (or points)
   first. It names a route that keeps to the part (r4-j J1): squares are
   planned on the scores_path that olmoearth_check_map gives for that part
   (olmoearth_plan_label_sample has no part argument, and by result_id it
   samples points over the whole result), points by result_id. */
OE.JOB_TEXT = {
  // WORDING: squares
  check: (r) => 'Check how wrong this map is: open its Check card (olmoearth_check_map) for the Studio prediction result '
    + r.result_id + ' (prediction_id ' + r.prediction_id + '); I will pick how many squares to check there.'
    + OE.named(r.name),
  // WORDING: r4h C6
  points: (r) => 'Check this map on points: Studio prediction result ' + r.result_id
    + (r.prediction_id ? ' (prediction_id ' + r.prediction_id + ')' : '')
    + '. First ask me how many points to check and which way to pick them.',
  // WORDING: r4j J1
  byClass: (r) => (r.unit === 'point'
    ? 'Check each class the map draws: a new check of Studio prediction result ' + r.result_id
      + ', picked class by class (olmoearth_plan_label_sample by result_id). Its own check and ranges, never pooled'
      + ' with this one. First ask me how many points to check.'
    : 'Check each class the map draws: a new check of Studio prediction result ' + r.result_id
      + (r.part_file ? ' (the part in file ' + r.part_file + ')' : '')
      + ', picked class by class (olmoearth_plan_label_sample on the scores_path that olmoearth_check_map gives for '
      + (r.part_file ? 'that part' : 'this map') + '). Its own check and ranges, never pooled with this one. First'
      + ' ask me how many squares to check.'),
  // WORDING: held for the reader test
  first: (r) => 'Find the windows of this map to look at first: rank the windows of the Studio prediction result '
    + r.result_id + ' (prediction_id ' + r.prediction_id + '), a class map with top-k scores, for review.'
    + OE.named(r.name),
  /* Two Studio results are planned by result_id: their sample is points of a
     grid (sheet.window_unit), so the request says points (the owner's ruling 6). */
  // WORDING: squares
  which: (a, b) => 'Say which of two maps is more accurate: ask me how many points I can label, then plan a'
    + ' labelled sample where the Studio prediction results ' + a.result_id + ' (prediction_id ' + a.prediction_id
    + ') and ' + b.result_id + ' (prediction_id ' + b.prediction_id + ') differ. Their Studio names, as data: '
    + OE.dataName(a.name) + ' and ' + OE.dataName(b.name) + '.',
  run: (m) => 'Run the Studio model with model_id ' + m.id + ' (project_id ' + m.project_id + ') on an area:'
    + ' open the map so I can draw or choose it, then ask me for the period and submit the prediction.'
    + OE.named(m.name),
};

/* Which job buttons a Studio prediction or result offers (the projects
   card). A model's kind (glyph.model_glyph: classes, values, detection,
   embeddings, or null when not recorded) and a result's property names.
   Check reads a class map with top-k scores or a [0, 1] score band, so it is
   not offered for embeddings or detection, where it can only fail; a
   regression keeps it, and the Check card says why when it refuses.
   "Where to look first" is offered only while the server says so
   (first_in_studio, server.FIRST_IN_STUDIO: the owner's decision D1), and
   then only for a class map with top-k scores: the model's kind is classes
   or not recorded, and a result's first property is the base of its top-k
   bands, as the tool reads it with no property_name (tools/sampling.py:
   names[0], or the base of names[0] when that is one of its ranks). */
OE.FIRST_KINDS = ['classes'];
OE.NO_CHECK_KINDS = ['embeddings', 'detection'];
OE.checkOffered = (kind) => !OE.NO_CHECK_KINDS.includes(kind);
OE.hasTopK = (names) => {
  const list = (Array.isArray(names) ? names : []).filter((n) => typeof n === 'string');
  if (!list.length) return false;
  const first = list[0];
  const part = /^(.+)_rank[1-9][0-9]*_(?:prob|class)$/.exec(first);
  const base = list.includes(first + '_rank1_prob') ? first : (part ? part[1] : null);
  return Boolean(base) && [base, base + '_rank1_prob', base + '_rank2_prob'].every((n) => list.includes(n));
};
/* names undefined: a prediction row, whose results are not read yet (the press decides). */
OE.firstOffered = (kind, names) => (!kind || OE.FIRST_KINDS.includes(kind))
  && (names === undefined || OE.hasTopK(names));
/* Said when a prediction offers no result with top-k class scores: neutral,
   since a kind not recorded may be a regression or embeddings map. */
// WORDING: held for the reader test
OE.NO_FIRST = 'This map was not saved with class scores, so there is no order of windows to look at first.';

/* A card's plain lines under its heading (the reader test's keys): what it
   is, what to do next, what it cannot tell, each a paragraph with its key as
   its class. The start and projects cards give their own; a result view and
   the sheet give theirs. null while every line is null (a lead the server
   has no words for), so nothing is drawn and the layout is unchanged;
   otherwise one block of at most two lines (r5-e, the owner: fewer; each
   card sends two or fewer, so the cap is only a guard), all in the body
   colour. */
OE.lead = (lead) => {
  const lines = ['what', 'next', 'cannot'].map((k) => [k, lead && typeof lead[k] === 'string' ? lead[k] : null])
    .filter(([, s]) => s).slice(0, 2);  // r5-e: at most two lines (owner: fewer)
  if (!lines.length) return null;
  return OE.el('div', { class: 'oe-lead' }, lines.map(([k, s]) => OE.el('p', { class: k }, s)));
};
/* A closed fold for the rest of a card once its lead has words: label is
   the fold's words, nodes what it holds. */
OE.fold = (label, ...nodes) => OE.el('details', { class: 'oe-more' },
  OE.el('summary', {}, label), ...nodes);

/* What the panel is: the OlmoEarth Agent inside Claude, not Studio (B/S6). */
// WORDING: r4p2
OE.SUB = 'This panel is the OlmoEarth Agent, an app inside Claude. It works with your OlmoEarth Studio data. It is not Studio itself.';

/* The one slim line on top of every view: the AGENT tag, OE.SUB and, right
   of the sentence, the account item (#oe-account, r5-a V2): empty, and not
   drawn, until a card that shows the account sets it (OE.setAccount: the
   start card and the projects card; the other views leave it empty). The
   tag's alt is empty, since the sentence names the agent. */
OE.line = () => OE.el('header', { class: 'oe-line' },
  OE.el('img', { src: OE.TAG, alt: '' }),
  OE.el('div', { class: 'oe-line-text' },
    OE.el('span', { class: 'oe-said' }, OE.SUB),
    OE.el('span', { id: 'oe-account', class: 'oe-account' })));

/* A short status line at the bottom of the page: a live region that OE.start
   puts in place before any message, so a screen reader announces the first.
   A plain message is announced politely (role status); an error at once
   (r5-b J29: role alert, aria-live assertive), and the next plain message
   politely again. */
OE.toastRegion = () => {
  let bar = document.getElementById('oe-toast');
  if (!bar) {
    bar = OE.el('div', { id: 'oe-toast', class: 'small', role: 'status', 'aria-live': 'polite', style: 'margin-top:12px' });
    document.body.append(bar);
  }
  return bar;
};
OE.toast = (text, isError = false) => {
  const bar = OE.toastRegion();
  bar.setAttribute('role', isError ? 'alert' : 'status');
  bar.setAttribute('aria-live', isError ? 'assertive' : 'polite');
  bar.className = 'small ' + (isError ? 'error' : 'muted');
  bar.textContent = text;
};

/* What a card says when it cannot go on (r4-k K2). CANCELLED: the host
   stopped the tool's call before its result came (ui/notifications/
   tool-cancelled); it replaces the card's waiting line only while no result
   has been drawn, never with the host's reason, and never says that nothing
   was saved (a stopped call may have done its work). NOT_STARTED: the card
   could not connect to the host; the same words as the watchdog each page
   runs before its module (a classic inline script), which shows them when
   OE.start never got that far (the MCP Apps client, its integrity hash or
   the map's Leaflet did not load). CARD_ERROR: the host bridge reported an
   error, one toast. None promises that Claude can list the squares: with
   panels on, the server hides them from Claude. */
// WORDING: r4k K2
OE.CANCELLED = 'This step was stopped before it finished. Ask Claude to run it again.';
// WORDING: r4k K2
OE.NOT_STARTED = 'This card could not start here. Carry on in the chat.';
// WORDING: r4k K2
OE.CARD_ERROR = 'Something went wrong in this card.';

/* What a card stops when Claude closes it (r4-k K1, OE.start's
   onteardown): functions, each run once then, every failure swallowed
   (the sign-in's poll timer: OE.signInPanel). */
OE.cleanups = new Set();

/* How long the teardown waits for the label editors' last saves (each
   editor's oeFlush, sheet.js) before it answers the host (r4-k K1). */
OE.TEARDOWN_WAIT_MS = 3000;

/* The server's name of each view (views.VIEWS), by the name its page starts
   with (OE.start): the host report's view. */
OE.VIEW_NAMES = {
  'olmoearth-label-sheet': 'sheet',
  'olmoearth-result': 'result',
  'olmoearth-check': 'check',
  'olmoearth-studio': 'studio',
  'olmoearth-start': 'start',
  'olmoearth-sign-in': 'sign-in',
  'olmoearth-area-map': 'map',
};

/* At most this many names in each list of the host report. */
OE.REPORT_MAX = 40;

/* A name in the host report: only [A-Za-z0-9_.:-], at most 40 characters. */
OE.reportName = (value) => String(value ?? '').replace(/[^A-Za-z0-9_.:-]+/g, '').slice(0, 40);

/* A list of names for the host report: each short, none empty, at most REPORT_MAX. */
OE.reportNames = (list) => (Array.isArray(list) ? list : [])
  .map((value) => OE.reportName(value)).filter(Boolean).slice(0, OE.REPORT_MAX);

/* Tell the server what the host offers this card (r4-k K3), through the
   app-only tool olmoearth_app_host_report: the view, the event ('start'
   once after connect, 'teardown' when Claude closes the card), the names of
   the host's capabilities, its display modes and its context's keys, which
   height the host gives the card ('height' fixed, 'maxHeight', or 'none')
   and the host's name. Names only, never a value: no id, path, class, label
   or place, and no tool argument (toolInfo is a key name here, nothing
   more). The server logs one line. A failure (an older server, a host that
   declines) is swallowed: no toast, no focus. Resolves true when the server
   took it. */
OE.hostReport = (app, view, event) => {
  try {
    const caps = (typeof app.getHostCapabilities === 'function' && app.getHostCapabilities()) || {};
    const ctx = (typeof app.getHostContext === 'function' && app.getHostContext()) || {};
    const info = (typeof app.getHostVersion === 'function' && app.getHostVersion()) || {};
    const box = ctx.containerDimensions && typeof ctx.containerDimensions === 'object' ? ctx.containerDimensions : {};
    const container = box.height !== undefined ? 'height' : box.maxHeight !== undefined ? 'maxHeight' : 'none';
    // The host's name as a name: at most 64 of these characters, with a letter, or ''.
    const named = String(info.name ?? '').replace(/[^A-Za-z0-9 _.:-]+/g, ' ').trim().slice(0, 64);
    return OE.call(app, 'olmoearth_app_host_report', {
      view: OE.reportName(view),
      event,
      caps: OE.reportNames(Object.keys(caps)),
      modes: OE.reportNames(ctx.availableDisplayModes),
      context_keys: OE.reportNames(Object.keys(ctx)),
      container,
      host: /[A-Za-z]/.test(named) ? named : '',
    }).then(() => true, () => false);
  } catch (_) {
    return Promise.resolve(false);
  }
};

/* The host's display mode as this page last heard it (r4-k K5): from the
   host's context at connect and at each change (OE.start, OE.takeContext),
   and from a Full screen button's answer; null before any. The label
   editors lay their face out for it (sheet.js OE.sheetMode), the area map
   its height (map.html fitHeight). */
OE.displayMode = null;

/* The display modes the host offers this card (its context's
   availableDisplayModes, as last heard), or null before any. */
OE.hostModes = null;

/* Every Full screen button built on this page: each is relabelled or
   hidden at every change (OE.syncFullscreen). */
OE.fullscreenButtons = new Set();

/* The Full screen button's two labels (r4-k K5): inline, and in full screen. */
// WORDING: r4k K5
OE.FULL_SCREEN = 'Full screen';
// WORDING: r4k K5
OE.BACK_TO_CHAT = 'Back to the chat';

/* What the Full screen button asks for now: back to the chat ('inline') in
   full screen, else 'fullscreen', with its label; offered only where the
   host lists that mode. */
OE.fullscreenTarget = (app) => {
  const ctx = (app && typeof app.getHostContext === 'function' && app.getHostContext()) || {};
  const modes = Array.isArray(OE.hostModes) ? OE.hostModes
    : Array.isArray(ctx.availableDisplayModes) ? ctx.availableDisplayModes : [];
  const full = OE.displayMode === 'fullscreen';
  const mode = full ? 'inline' : 'fullscreen';
  return { mode, label: full ? OE.BACK_TO_CHAT : OE.FULL_SCREEN, offered: modes.includes(mode) };
};

/* Relabel or hide every Full screen button for the mode the card is in; a
   button taken off the page after it was on it is dropped from the set. */
OE.syncFullscreen = (app) => {
  const target = OE.fullscreenTarget(app);
  for (const button of [...OE.fullscreenButtons]) {
    if (!button.isConnected && button.dataset.oeShown) { OE.fullscreenButtons.delete(button); continue; }
    if (button.isConnected) button.dataset.oeShown = '1';
    button.textContent = target.label;
    button.hidden = !target.offered;
  }
};

/* The host's context, at connect (changed null) or changed (what changed,
   which the client also merges into app.getHostContext()): the display
   mode and the display modes it names become OE.displayMode and
   OE.hostModes, and every Full screen button follows. */
OE.takeContext = (app, changed) => {
  const merged = (app && typeof app.getHostContext === 'function' && app.getHostContext()) || {};
  const pick = (key) => (changed && changed[key] !== undefined ? changed[key] : merged[key]);
  const mode = pick('displayMode');
  if (typeof mode === 'string' && mode) OE.displayMode = mode;
  const modes = pick('availableDisplayModes');
  if (Array.isArray(modes)) OE.hostModes = modes.filter((m) => typeof m === 'string');
  OE.syncFullscreen(app);
};

/* Whether the host saves a file the card hands it (r4-k K8): the client
   has downloadFile and the host's capabilities carry downloadFile. False
   before the host connected. */
OE.canDownload = (app) => {
  try {
    if (!app || typeof app.downloadFile !== 'function') return false;
    const caps = typeof app.getHostCapabilities === 'function' ? app.getHostCapabilities() : null;
    return Boolean(caps && typeof caps === 'object' && caps.downloadFile);
  } catch (_) {
    return false;
  }
};

/* Connect to the host. Handlers are set before connect(): the host sends the
   tool's result once, right after the handshake, and a one-shot event set
   later is missed. The views declare the display modes they can use.
   onContext(ctx) gets the host's context changes (its displayMode when the
   user leaves full screen), after OE.takeContext has taken them. Also:
   - the result: once it came, the cancelled line no longer replaces the card;
   - stopped (r4-k K2): the host's tool-cancelled replaces #root with
     OE.CANCELLED while no result came, never the host's reason;
   - an error the bridge reports (r4-k K2): one toast, OE.CARD_ERROR, once
     per page; a late answer to a call whose wait already ended (the
     client's "unknown message ID"; the caller said so, as the Check card's
     press does) is no error of this card;
   - closed (r4-k K1, ui/resource-teardown): every label editor's last
     answers are saved quietly (oeFlush), waiting at most TEARDOWN_WAIT_MS,
     then the editors drop their timers and listeners, every OE.cleanups
     entry runs, the host report says 'teardown' (sent, not awaited), and
     the host gets {}; no toast and no focus;
   - connect() in try/catch: a card that cannot connect says NOT_STARTED in
     #root; window.OE_STARTED is set right after it connects, which stops
     the page's watchdog (r4-k K2), and a card whose watchdog already gave
     up (a host that answered after 10 s) gets its own nodes back;
   - after connect: the host's first context (OE.takeContext) and one host
     report, 'start' (r4-k K3), not awaited. */
OE.start = async (name, { onResult, onInput, onContext } = {}) => {
  const fonts = document.createElement('link');
  fonts.rel = 'stylesheet';
  fonts.href = OE.FONTS;  // Studio's Manrope and Roboto Mono; the system font until they arrive
  document.head.append(fonts);
  const style = document.createElement('style');
  style.textContent = OE.STYLE;
  document.head.append(style);
  document.body.prepend(OE.line());
  OE.toastRegion();
  const app = new App({ name, version: '1.0.0' }, { availableDisplayModes: ['inline', 'fullscreen'] });
  const view = OE.VIEW_NAMES[name] || String(name).replace(/^olmoearth-/, '');
  // The card's own nodes before it connects (its waiting line, the area
  // map's map and controls): put back if the watchdog gave up on a host
  // that answered after its 10 s.
  const root = document.getElementById('root');
  const waiting = root ? [...root.childNodes] : [];
  // Each host report once per event (start, teardown).
  const reported = new Set();
  const report = (event) => {
    if (reported.has(event)) return Promise.resolve(false);
    reported.add(event);
    return OE.hostReport(app, view, event);
  };
  // Set when the tool's result came: the cancelled line no longer replaces the card.
  let resultCame = false;
  app.ontoolresult = (params) => {
    resultCame = true;
    return onResult ? onResult(params, app) : undefined;
  };
  if (onInput) app.ontoolinput = (params) => onInput(params, app);
  app.onhostcontextchanged = (ctx) => {
    OE.takeContext(app, ctx || {});
    if (onContext) onContext(ctx || {}, app);
  };
  app.ontoolcancelled = () => {
    if (resultCame) return;
    if (root) root.replaceChildren(OE.el('div', { class: 'card summary', role: 'status' }, OE.CANCELLED));
  };
  let errorShown = false;
  app.onerror = (err) => {
    if (errorShown || /unknown message id/i.test(String((err && err.message) || ''))) return;
    errorShown = true;
    OE.toast(OE.CARD_ERROR, true);
  };
  app.onteardown = async () => {
    try {
      const editors = [...document.querySelectorAll('section.sheet')].filter((e) => typeof e.oeFlush === 'function');
      let timer = null;
      await Promise.race([
        Promise.all(editors.map((e) => Promise.resolve().then(() => e.oeFlush()).catch(() => false))),
        new Promise((resolve) => { timer = setTimeout(resolve, OE.TEARDOWN_WAIT_MS); }),
      ]);
      clearTimeout(timer);
      if (typeof OE.disposeEditors === 'function') OE.disposeEditors(document);
    } catch (_) { /* the host still gets its answer */ }
    for (const stop of [...OE.cleanups]) {
      try { stop(); } catch (_) { /* the next one still runs */ }
    }
    OE.cleanups.clear();
    report('teardown');
    return {};
  };
  try {
    await app.connect();
  } catch (_) {
    if (root) root.textContent = OE.NOT_STARTED;
    return app;
  }
  window.OE_STARTED = true;
  if (root && root.textContent === OE.NOT_STARTED && waiting.length) root.replaceChildren(...waiting);
  OE.takeContext(app, null);
  report('start');
  return app;
};

/* A small outlined button that takes the card to the host's full screen
   and back (r4-k K5): 'Full screen' inline, 'Back to the chat' in full
   screen, each shown only where the host's availableDisplayModes lists the
   mode it asks for (else hidden), relabelled or hidden at every context
   change (OE.syncFullscreen). The host's answer is the mode the card is in
   now, which may differ from the one asked for: the label follows the
   answer, and a full screen not given says so in a toast. after(mode) runs
   with the answer's mode, as before. A press while one is being answered
   does nothing. */
OE.fullscreenButton = (app, after, cls = 'small') => {
  let asking = false;
  const button = OE.el('button', { class: cls, type: 'button' });
  button.addEventListener('click', async () => {
    const target = OE.fullscreenTarget(app);
    if (asking || !target.offered) return;
    asking = true;
    try {
      const answer = await app.requestDisplayMode({ mode: target.mode });
      const mode = answer && typeof answer.mode === 'string' && answer.mode ? answer.mode : null;
      if (mode) OE.displayMode = mode;
      OE.syncFullscreen(app);
      if (target.mode === 'fullscreen' && mode !== 'fullscreen') OE.toast('Full screen is not available here.', true);
      if (after) after(mode);
    } catch (err) {
      if (target.mode === 'fullscreen') OE.toast('Full screen is not available here.', true);
    } finally {
      asking = false;
    }
  });
  OE.fullscreenButtons.add(button);
  OE.syncFullscreen(app);
  return button;
};

/* A joined toggle group: options [[value, label], ...]; onChange(value). With
   clearable, a second press on the option that is pressed clears it: no option
   is pressed then, and onChange(null). */
OE.toggle = (options, value, onChange, { label, small = true, clearable = false } = {}) => {
  const group = OE.el('div', { class: 'toggle' + (small ? ' small' : ''), role: 'group', 'aria-label': label });
  const buttons = options.map(([v, text]) => {
    const b = OE.el('button', { type: 'button', 'aria-pressed': String(v === value) }, text);
    b.addEventListener('click', () => {
      const clear = clearable && b.getAttribute('aria-pressed') === 'true';
      buttons.forEach((other) => other.setAttribute('aria-pressed', String(!clear && other === b)));
      onChange(clear ? null : v);
    });
    return b;
  });
  group.append(...buttons);
  return group;
};

/* A chip for a kind or a status; ring for a finished one (Studio's mint ring). */
OE.chip = (text, { ring = false } = {}) => OE.el('span', { class: 'chip' + (ring ? ' ring' : '') }, text);

/* Key and value rows: [[label, value-or-node], ...]. */
OE.kv = (pairs) => OE.el('div', { class: 'kv' },
  pairs.filter(([, v]) => v !== undefined && v !== null && v !== '')
    .map(([k, v]) => OE.el('div', {}, OE.el('span', {}, k), OE.el('span', {}, v))));

/* A tool's field name as words: "n_population" -> "N population". */
OE.humanize = (key) => {
  const words = String(key).replace(/_/g, ' ').trim();
  return words.charAt(0).toUpperCase() + words.slice(1);
};

OE.accountLine = (account) => {
  if (!account) return 'Signed in to OlmoEarth Studio.';
  const who = account.user_name || 'your account';
  const org = account.organization ? ' (' + account.organization + ')' : '';
  return 'Signed in to OlmoEarth Studio as ' + who + org + '.';
};

/* The account in the top line, short (r5-a V2): "Ana (Lab)"; "Your account
   (Lab)" when Studio gave no name; "Signed in" with no account at all. With
   the key the app was installed with (source 'environment', r5-c S01), what
   is in use in place of whose: no Sign out can remove that key. */
// WORDING: r5a V2
OE.accountShort = (account, source = null) => {
  // WORDING: r5c S01
  if (source === 'environment') return 'Signed in with the app\'s own key';
  if (!account) return 'Signed in';
  const who = account.user_name || 'Your account';
  const org = account.organization ? ' (' + account.organization + ')' : '';
  return who + org;
};

/* The sign-in panel's words (r5-b J19). Its button says what its press
   does. Its note is a few plain lines: SIGN_IN_OPENS says what the button
   opens, where the key is typed and where Studio makes one; SIGN_IN_NOTE,
   the default, adds where the key goes, in the words every sign-in text
   uses (SIGN_IN_KEY_GOES; r5-e: no longer that Studio checks it, which a
   wrong key's own error shows). A card that needs Studio for its own step
   (the Check card, a result card) gives SIGN_IN_OPENS and SIGN_IN_KEY_GOES
   too (r5-e K6, R2: the first line of its own, which said its heading
   again, is gone). */
// WORDING: r5b J19
OE.SIGN_IN_BUTTON = 'Open the sign-in page';
// WORDING: r5b J19
OE.SIGN_IN_OPENS = 'The button opens a sign-in page in your browser, on this computer. Enter your Studio API key there (in Studio: your profile, then API Keys).';
// WORDING: r5b J19
OE.SIGN_IN_KEY_GOES = 'The key goes to the OlmoEarth Agent here, not through Claude.';
// WORDING: r5e
OE.SIGN_IN_NOTE = [OE.SIGN_IN_OPENS, OE.SIGN_IN_KEY_GOES];

/* The Studio sign-in, framed as a Studio dialog without the modal parts: a
   white card, as every other card (r5-b J19; before, cream on the cream
   canvas). note is its lines under the title, each its own div: a list of
   sentences, or one string; none gives SIGN_IN_NOTE. A line may be an
   element, a div placed as given (the start card's Request access line, its
   text button inside it: r5-g G2). The key is never typed here: the button
   opens a one-time page this server serves on 127.0.0.1 in the user's own
   browser, where the key goes straight
   to the server, in no MCP message. The panel then asks the server, 2 s after
   each answer, whether a sign-in was made on the page since the button was
   pressed (the server's sign_ins count went up), so a key already in effect
   never reads as this sign-in. It stops when the page closes, and when
   Claude closes the card (its stop is in OE.cleanups, r4-k K1).
   With acceptCurrent, a press first asks whether a key is in effect (a
   sign-in made in another panel) and takes it without opening the page.
   onDone(status) runs once signed in. While the button's press is being
   answered it is aria-disabled, not disabled, so it keeps the focus. */
OE.signInPanel = (app, { onDone, note, primary = true, acceptCurrent = false } = {}) => {
  // A live region for the waiting and closed messages; the outcome is
  // announced once, by the page's status line (OE.toast), which outlasts the panel.
  const status = OE.el('div', { class: 'small muted', role: 'status', 'aria-live': 'polite' });
  const link = OE.el('div', { class: 'small' });
  const button = OE.el('button', { class: primary ? 'primary' : '' }, OE.SIGN_IN_BUTTON);
  let timer = null;
  let run = 0;  // the current wait; a new click ends the one before
  let busy = false;
  const setBusy = (value) => { busy = value; button.setAttribute('aria-disabled', String(value)); };
  const stop = () => { if (timer) clearTimeout(timer); timer = null; run += 1; };
  // When Claude closes the card, the poll stops (r4-k K1: OE.start's teardown).
  OE.cleanups.add(stop);
  const signedIn = (now) => {
    stop();
    status.removeAttribute('role');
    status.removeAttribute('aria-live');
    status.className = 'small ok';
    status.textContent = OE.accountLine(now.account);
    link.replaceChildren();
    setBusy(false);
    OE.toast(OE.accountLine(now.account));
    if (onDone) onDone(now);
  };
  const poll = async (mine, since) => {
    let now = null;
    try { now = await OE.call(app, 'olmoearth_app_status'); } catch (err) { now = null; }
    if (mine !== run) return;
    if (now && now.sign_ins > since && now.signed_in && !now.key_problem) { signedIn(now); return; }
    if (now && !now.sign_in_page_open) {
      stop();
      setBusy(false);
      status.className = 'small muted';
      status.textContent = 'The sign-in page has closed. Open it again to sign in.';
      link.replaceChildren();
      return;
    }
    timer = setTimeout(() => poll(mine, since), 2000);
  };
  button.addEventListener('click', async () => {
    if (busy) return;
    setBusy(true);
    try {
      if (acceptCurrent) {
        const now = await OE.call(app, 'olmoearth_app_status');
        if (now && now.signed_in && !now.key_problem) { signedIn(now); return; }
      }
      const page = await OE.call(app, 'olmoearth_app_begin_sign_in');
      const opened = await OE.open(app, page.url);
      status.className = 'small muted';
      status.textContent = opened
        ? 'Waiting for you to sign in on the page that opened in your browser...'
        : 'Open this address in a browser on this computer to sign in:';
      link.replaceChildren(OE.el('code', { style: 'user-select:all;overflow-wrap:anywhere' }, page.url));
      stop();
      const mine = run;
      timer = setTimeout(() => poll(mine, Number(page.sign_ins) || 0), 2000);
      // The wait has started: a new press opens the page again (the server
      // gives the open page's address) and ends this wait (run).
      setBusy(false);
    } catch (err) {
      setBusy(false);
      status.className = 'small error';
      status.textContent = err.message;
    }
  });
  // The note's lines: a list as given, one string as one line, else the
  // default; a line that is an element (r5-g G2) is placed as it is.
  const given = (Array.isArray(note) ? note : [note]).filter((s) => (typeof s === 'string' && s) || s instanceof Node);
  const lines = given.length ? given : OE.SIGN_IN_NOTE;
  return OE.el('section', { class: 'card dialog stack' },
    OE.el('h2', {}, 'Sign in to OlmoEarth Studio'),
    OE.el('div', { class: 'oe-sign-note' }, lines.map((s) => (s instanceof Node ? s : OE.el('div', {}, s)))),
    status, link,
    OE.el('div', { class: 'row', style: 'justify-content:flex-end' }, button));
};

/* A Sign out button: a text button in the top line's account item ('link',
   OE.setAccount), small outlined on the account card ('small'). */
OE.signOutButton = (app, { onSignOut, cls = 'link' } = {}) => {
  const out = OE.el('button', {
    class: cls,
    onclick: async () => {
      out.disabled = true;
      try {
        const result = await OE.call(app, 'olmoearth_app_sign_out');
        // WORDING: r5c S01 (the first; the second: r4p2). By the key in effect,
        // not environment_key_remains: an OLMOEARTH_API_KEY that is no key
        // (the placeholder) leaves the user signed out.
        OE.toast(result.signed_in && result.source === 'environment'
          ? 'Still signed in: this app was installed with its own key. Only whoever installed it can remove it.'
          : 'Signed out; the saved key was removed.');
        if (onSignOut) onSignOut(result);
      } catch (err) {
        OE.toast(err.message, true);
      } finally { out.disabled = false; }
    },
  }, 'Sign out');
  return out;
};

/* The account item of the top line (#oe-account, OE.line), set by a card
   that shows the account at each of its drawings (r5-a V2: the start card
   and the projects card). Signed in: who (OE.accountShort; the whole
   sentence, OE.accountLine, is the item's name for a screen reader and the
   name's tooltip) and a Sign out text button, which calls the card's
   onSignOut(result). With the key the app was installed with (source
   'environment', r5-c S01) no Sign out: its press cannot remove that key, so
   the item says only that this key is in use. Signed out: empty, which the
   style does not draw, so a name never outlasts its sign-in. The pointer and
   the keyboard stay where they are. Returns the Sign out button, or null. */
OE.setAccount = (app, { signedIn = false, account = null, source = null, onSignOut } = {}) => {
  const item = document.getElementById('oe-account');
  if (!item) return null;
  if (!signedIn) {
    item.replaceChildren();
    item.removeAttribute('role');
    item.removeAttribute('aria-label');
    return null;
  }
  const said = OE.accountLine(account);
  const out = source === 'environment' ? null : OE.signOutButton(app, { onSignOut });
  item.setAttribute('role', 'group');
  item.setAttribute('aria-label', said);
  item.replaceChildren(...[OE.el('span', { class: 'who', title: said }, OE.accountShort(account, source)), out]
    .filter(Boolean));
  return out;
};
