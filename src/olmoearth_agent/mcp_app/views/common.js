/* Shared by every OlmoEarth Agent view (inlined by views.py).

   The page imports App from the pinned MCP Apps client before this script;
   everything here runs inside the host's sandboxed iframe and reaches the
   server only through app.callServerTool. A view never calls Studio itself,
   and never sees the API key: the sign-in opens a page the server serves on
   127.0.0.1 in the user's browser (OE.signInPanel).

   The look is OlmoEarth Studio's working pages (a MUI theme), from values
   measured with getComputedStyle on 11 November 2026:
   - one slim line on top: the AGENT tag and the sentence that this panel
     is the OlmoEarth Agent, an app inside Claude (not Studio itself);
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
.card.object { border-radius: 12px; cursor: pointer; transition: box-shadow .3s cubic-bezier(.4,0,.2,1), background .3s; }
.card.object:hover { box-shadow: var(--shadow-2); background: linear-gradient(var(--wash), var(--wash)), var(--paper); }
.card.summary { background: var(--canvas); }
/* A Studio dialog without its modal parts: the title 16px 24px, the content 0 24px 20px. */
.card.dialog { padding: 0 24px 20px; }
.card.dialog > h2:first-child { font-size: 20px; line-height: 28px; margin: 0 -24px; padding: 16px 24px; }
.card.dialog > h2:first-child + * { margin-top: 0; }
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
/* A joined toggle group (Studio's LIST VIEW / CARD VIEW). */
.toggle { display: inline-flex; }
.toggle button { border-radius: 0; border: 1px solid var(--divider); padding: 7px; margin-left: -1px;
  font-size: 13px; line-height: 22.75px; font-weight: 500; text-transform: uppercase; color: var(--grey); }
.toggle button:first-child { margin-left: 0; border-radius: 4px 0 0 4px; }
.toggle button:last-child { border-radius: 0 4px 4px 0; }
.toggle button:hover { background: var(--wash); border-color: var(--divider); }
.toggle button[aria-pressed="true"] { background: var(--selected); color: var(--text); }
.toggle.small button { padding: 3px 9px; }
/* A label sheet row's group of answers takes the focus after an answer
   (tabindex -1, not a button): a ring when the keyboard brought it there. */
.toggle[tabindex]:focus { outline: none; }
.toggle[tabindex]:focus-visible { outline: 2px solid var(--text); outline-offset: 2px; border-radius: 4px; }
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
input::placeholder { color: rgba(10,50,53,.42); }
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
/* A card's three plain lines (OE.lead): what it is, what to do next, what it
   cannot tell, the last in the secondary colour. */
.oe-lead > p { margin: 0; }
.oe-lead > p + p { margin-top: 4px; }
.oe-lead > p.cannot { color: var(--text-2); }
/* The rest of a card in a closed fold (OE.fold), spaced as a stack once open. */
.oe-more > summary + * { margin-top: 8px; }
.oe-more > * + * { margin-top: 12px; }
/* A job's message the host did not take, to type in the chat: on a line of its
   own under the row, whole and selectable. */
.oe-type { flex-basis: 100%; margin-top: 4px; }
.oe-type:empty { display: none; }
.oe-type code { display: block; margin-top: 2px; padding: 6px 8px; background: var(--paper); border: 1px solid var(--divider);
  border-radius: 4px; user-select: all; overflow-wrap: anywhere; white-space: pre-wrap; }
/* The label sheet's footer: sticky, a hairline above, the counter and the one button. */
.sheet-footer { position: sticky; bottom: 0; margin: 16px -16px -16px; padding: 12px 16px;
  border-top: 1px solid var(--divider); background: var(--paper); border-radius: 0 0 4px 4px; }
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
}
/* The label sheet's face: one window at a time. Inline it is a pager, the
   heading, Previous and Next and the answers on top and the dates in a strip
   that scrolls sideways; in the host's full screen (.full) the dates are a
   grid. An answer given shows pressed; none is the contained button. The
   face takes the focus after each answer: a ring when the keyboard brought
   it there, as the list's group of answers has. */
.face:focus { outline: none; }
.face:focus-visible { outline: 2px solid var(--text); outline-offset: 4px; border-radius: 4px; }
.face-dates:focus { outline: none; }
.face-dates:focus-visible { outline: 2px solid var(--text); outline-offset: 2px; border-radius: 4px; }
.face-top { display: flex; flex-wrap: wrap; gap: 12px; align-items: center; justify-content: space-between; }
.face-dates { display: flex; gap: 12px; overflow-x: auto; }
.face-dates > * { flex: none; }
.face.full .face-dates { display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); overflow: visible; }
.face .answers button { min-width: 96px; padding: 10px 16px; }
/* Only a chosen answer looks chosen (r4-h A/F22): no hover fill or border
   change on an answer not chosen, and a chosen one filled dark. */
.face .answers button:not([aria-pressed="true"]):not([aria-disabled="true"]):not(:disabled):hover {
  background: transparent; border-color: var(--outline); }
.face .answers button[aria-pressed="true"] { background: var(--text); color: var(--canvas); border-color: var(--text);
  font-weight: 600; }
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
   host took the message, false (after a toast) when it did not. */
OE.say = async (app, text) => {
  try {
    const answer = await app.sendMessage({ role: 'user', content: [{ type: 'text', text }] });
    if (answer && answer.isError) {
      OE.toast('Claude did not take the message; type it in the chat instead.', true);
      return false;
    }
    return true;
  } catch (err) {
    OE.toast('Could not send the message to Claude: ' + err.message, true);
    return false;
  }
};

/* Where a job button's message shows when the host does not take it: empty,
   and not drawn, until then. */
OE.typeSlot = () => OE.el('div', { class: 'oe-type small' });

/* A job button's press: its request goes to Claude as the user's words. When
   the host declines it, slot shows "Type this in the chat:" and the whole
   text, selectable, so its ids are not lost; a later send clears it. */
OE.sendJob = async (app, text, slot) => {
  const sent = await OE.say(app, text);
  if (slot) {
    slot.replaceChildren(...(sent ? [] : [OE.el('div', {}, 'Type this in the chat:'), OE.el('code', {}, text)]));
  }
  return sent;
};

/* Open a link in the user's browser; false when the host declined. The
   host's own error is never shown (r4-k K7): it may repeat the link, and a
   street map link holds a place. */
OE.open = async (app, url) => {
  try {
    const answer = await app.openLink({ url });
    if (answer && answer.isError) { OE.toast('The link was not opened.', true); return false; }
    return true;
  } catch (_) {
    // WORDING: r4k K7
    OE.toast('The link was not opened.', true);
    return false;
  }
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
   coordinate, a window's place, a class or a label, nor a count of wrong. */
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
   boundary (OE.fitContext), never inside a path or a number; the update is
   not sent again when it is the one sent last. Never a toast, never the
   focus: every failure is swallowed, and a host that never answers costs
   nothing. */
OE.tellClaude = (app, text, part = 'panel') => {
  try {
    if (!app || typeof app.updateModelContext !== 'function') return;
    const caps = typeof app.getHostCapabilities === 'function' ? app.getHostCapabilities() : null;
    if (!caps || typeof caps !== 'object') return;
    if ('updateModelContext' in caps && !caps.updateModelContext) return;
    const cleaned = (value) => String(value ?? '').replace(/[\u0000-\u001f\u007f-\u009f\u2028\u2029]+/g, ' ').trim();
    if (typeof text === 'function') OE.contextParts.set(part, text);
    else if (cleaned(text)) OE.contextParts.set(part, cleaned(text));
    else OE.contextParts.delete(part);
    if (!OE.contextParts.size) return;
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
    if (!said.length) return;
    const whole = OE.CONTEXT_HEAD + said.join(' ');
    if (whole === OE.contextSent) return;
    OE.contextSent = whole;
    Promise.resolve(app.updateModelContext({ content: [{ type: 'text', text: whole }] })).catch(() => {});
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

/* A card's three plain lines (the reader test's): what it is, what to do
   next, what it cannot tell, each a paragraph with its key as its class.
   The start and projects cards give their own; a result view and the Check
   card give the server's lead. null while every line is null (a lead the
   server has no words for), so nothing is drawn and the layout is
   unchanged; otherwise one block of at most three lines. */
OE.lead = (lead) => {
  const lines = ['what', 'next', 'cannot'].map((k) => [k, lead && typeof lead[k] === 'string' ? lead[k] : null])
    .filter(([, s]) => s);
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

/* The one slim line on top of every view: the AGENT tag and OE.SUB. The tag's
   alt is empty, since the sentence names the agent. */
OE.line = () => OE.el('header', { class: 'oe-line' },
  OE.el('img', { src: OE.TAG, alt: '' }),
  OE.el('span', {}, OE.SUB));

/* A short status line at the bottom of the page: a live region that OE.start
   puts in place before any message, so a screen reader announces the first. */
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

/* The Studio sign-in, framed as a Studio dialog without the modal parts. The
   key is never typed here: the button opens a one-time page this server
   serves on 127.0.0.1 in the user's own browser, where the key goes straight
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
  const button = OE.el('button', { class: primary ? 'primary' : '' }, 'Sign in in your browser');
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
  return OE.el('section', { class: 'card summary dialog stack' },
    OE.el('h2', {}, 'Sign in to OlmoEarth Studio'),
    // WORDING: r4p2 (where Studio makes a key, after "Your Studio API key")
    OE.el('div', {}, note
      || 'You sign in on a page this app opens in your browser, served on this computer. Your Studio API key (in Studio, open your profile, then API Keys) goes from that page to the OlmoEarth Agent here, not through Claude, and is checked with Studio before it is used.'),
    status, link,
    OE.el('div', { class: 'row', style: 'justify-content:flex-end' }, button));
};

/* A Sign out button: a text button in a page's account line ('link'), small
   outlined on the account card ('small'). */
OE.signOutButton = (app, { onSignOut, cls = 'link' } = {}) => {
  const out = OE.el('button', {
    class: cls,
    onclick: async () => {
      out.disabled = true;
      try {
        const result = await OE.call(app, 'olmoearth_app_sign_out');
        // WORDING: r4p2 (the first: B/S7)
        OE.toast(result.environment_key_remains
          ? 'Signed out here. A key put in this app\'s settings when it was installed (OLMOEARTH_API_KEY) is still in use; whoever installed the app can remove it.'
          : 'Signed out; the saved key was removed.');
        if (onSignOut) onSignOut(result);
      } catch (err) {
        OE.toast(err.message, true);
      } finally { out.disabled = false; }
    },
  }, 'Sign out');
  return out;
};

/* The signed-in line with a Sign out text button. */
OE.accountBar = (app, account, { onSignOut } = {}) =>
  OE.el('div', { class: 'row small' }, OE.accountLine(account), OE.signOutButton(app, { onSignOut }));
