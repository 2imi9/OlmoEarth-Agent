/* Figures for reopened chats. A chat saved in localStorage keeps no figure SVG
   (60 to 160 KB each; localStorage holds about 5 MB), so the SVG of each figure
   shown live is kept here, in IndexedDB (database 'oe_figures', store
   'figures'), keyed by the figure's path (the tool result's figure_path); a
   chat reopened from the history looks its figures up by that path. The store
   keeps at most 50 MB and 300 figures: past either, the least recently stored
   or shown go first. Where the browser's storage for the page fills up before
   that (QuotaExceededError), the least recently used figures make room for the
   new one, which is tried once more. Every function resolves and never
   rejects: where IndexedDB is missing, blocked (a private window) or fails,
   storing does nothing and a lookup finds nothing, and the chat shows the
   caption and the saved path as before. */

const DB_NAME = 'oe_figures';
const STORE = 'figures';
const BY_USE = 'by_use';             // index on [at, bytes]: oldest use first, sizes without reading the SVGs
const MAX_BYTES = 50 * 1024 * 1024;  // 50 MB of figures in all
const MAX_COUNT = 300;               // and at most 300 of them

let opening = null;  // the open database's promise, shared by every call
let clears = 0;      // clearFigures calls so far: a retry begun before one stores nothing after it

/* The database, or null where IndexedDB is missing, blocked or fails. A call
   blocked by another tab (it holds the database open at an older version)
   gets null, but the null is not kept: the next call tries again. */
function openDb() {
  if (opening) return opening;
  const attempt = new Promise((resolve) => {
    let settled = false;  // this promise has its answer: a connection opening after it is closed
    try {
      const req = indexedDB.open(DB_NAME, 1);
      req.onupgradeneeded = (event) => {
        // From no database (version 0) to version 1: the store and its index.
        // A later version adds its own step, for the version it comes from.
        if (event.oldVersion < 1) {
          const store = req.result.createObjectStore(STORE, { keyPath: 'key' });
          store.createIndex(BY_USE, ['at', 'bytes']);
        }
      };
      req.onsuccess = () => {
        const db = req.result;
        if (settled) { db.close(); return; }  // opened after onblocked gave up on it: nobody holds it
        settled = true;
        db.onversionchange = () => { db.close(); if (opening === attempt) opening = null; };
        db.onclose = () => { if (opening === attempt) opening = null; };  // closed by the browser (site data cleared)
        resolve(db);
      };
      req.onerror = () => { settled = true; resolve(null); };
      req.onblocked = () => {
        settled = true;
        if (opening === attempt) opening = null;  // not kept: the next call opens again
        resolve(null);
      };
    } catch (e) {
      resolve(null);  // no indexedDB here, or opening it throws (storage blocked)
    }
  });
  opening = attempt;
  return attempt;
}

/* Run `work(store, keep)` in one transaction on the figures. `keep(value)`
   resolves with `value` at once (a lookup answers on its request's success,
   not at the transaction's end). Otherwise the promise resolves when the
   transaction ends: with `fallback` once it commits; once it aborts, with
   `aborted(error)` where given (the error it aborted with, or null), else
   `fallback`. Where there is no database, or `work` throws, it resolves with
   `fallback`, after aborting the transaction so that no write `work` queued
   commits. Never rejects. */
function inStore(mode, work, fallback, aborted) {
  return openDb().then((db) => new Promise((resolve) => {
    if (!db) { resolve(fallback); return; }
    let tx = null;  // outside the try: a throw after a write was queued aborts that write
    try {
      tx = db.transaction(STORE, mode);
      tx.oncomplete = () => resolve(fallback);
      tx.onabort = () => resolve(aborted ? aborted(tx.error) : fallback);
      const keep = (value) => resolve(value);
      work(tx.objectStore(STORE), keep);
    } catch (e) {
      if (tx) {
        try { tx.abort(); } catch (err) { /* it had ended already */ }
      }
      resolve(fallback);
    }
  })).catch(() => fallback);
}

/* The name of the error a transaction aborted with ('AbortError' if none). */
function abortName(error) {
  return (error && error.name) || 'AbortError';
}

/* A string's size in UTF-8 bytes. */
function byteLength(s) {
  return new Blob([s]).size;
}

/* Walk the [at, bytes] index by key only, so no SVG is read, and hand `then`
   every figure as [key, bytes], least recently used first, and their total. */
function byAge(store, then) {
  const entries = [];
  let total = 0;
  const req = store.index(BY_USE).openKeyCursor();
  req.onsuccess = () => {
    const cursor = req.result;
    if (cursor) {
      const bytes = Number(cursor.key[1]) || 0;
      entries.push([cursor.primaryKey, bytes]);
      total += bytes;
      cursor.continue();
      return;
    }
    then(entries, total);
  };
}

/* Delete the least recently used figures while the store holds more than
   MAX_BYTES of them or more than MAX_COUNT, never `spare` (the figure just
   stored, whose time may sort first if the clock went back). */
function evict(store, spare) {
  byAge(store, (entries, total) => {
    let count = entries.length;
    for (const [key, bytes] of entries) {
      if (total <= MAX_BYTES && count <= MAX_COUNT) break;
      if (key === spare) continue;
      store.delete(key);
      total -= bytes;
      count -= 1;
    }
  });
}

/* Make room after the browser refused a figure for lack of space
   (QuotaExceededError): delete the least recently used figures, oldest first,
   until `need` bytes (the refused figure's) are freed or half of what the
   store holds, whichever comes first, so that a figure too large for what is
   left does not take every other figure with it; never `spare` (the refused
   figure's key, whose older copy stays shown if the retry fails too). */
function freeRoom(need, spare) {
  return inStore('readwrite', (store) => {
    byAge(store, (entries, total) => {
      const target = Math.min(need, total / 2);
      let freed = 0;
      for (const [key, bytes] of entries) {
        if (freed >= target) break;
        if (key === spare) continue;
        store.delete(key);
        freed += bytes;
      }
    });
  }, undefined);
}

/**
 * Keep a figure shown live, under its path, then evict the least recently
 * used figures past 50 MB or 300 figures. Where the browser's storage is full
 * (QuotaExceededError), free room by the least recently used figures and try
 * once more; any further failure leaves the figure unstored. Not awaited by
 * the chat.
 * @param {string} key the figure's path (the tool result's figure_path)
 * @param {string} svg the figure's SVG text (the event's figure_svg)
 * @param {string} caption the tool's figure_caption ('' if none)
 * @returns {Promise<void>} resolves once stored, or once it cannot be; never rejects
 */
export function putFigure(key, svg, caption) {
  if (typeof key !== 'string' || !key || typeof svg !== 'string' || !svg) return Promise.resolve();
  const text = typeof caption === 'string' ? caption : '';
  let bytes;
  try { bytes = byteLength(svg) + byteLength(text); } catch (e) { return Promise.resolve(); }
  if (bytes > MAX_BYTES) return Promise.resolve();  // one figure larger than the whole store
  const rec = { key, svg, caption: text, bytes, at: Date.now() };
  const generation = clears;
  // One transaction: the figure, then eviction past the caps (sparing it).
  // Resolves with '' once stored, else the name of the error it aborted with.
  const put = () => inStore('readwrite', (store) => {
    store.put(rec);
    evict(store, key);
  }, '', abortName);
  return put().then((failed) => {
    if (failed !== 'QuotaExceededError' || generation !== clears) return undefined;
    // The browser's storage for the page is full: make room, then one retry,
    // unless the figures were cleared meanwhile.
    return freeRoom(bytes, key).then(() => (generation === clears ? put() : '')).then(() => undefined);
  }).catch(() => undefined);
}

/* Count a figure as used now, so eviction keeps it longer: its own readwrite
   transaction, which a lookup does not wait for. It reads the record again
   there, so a figure stored under the same path since the lookup is kept as
   stored; if this fails, the figure was shown all the same. */
function touch(key) {
  return inStore('readwrite', (store) => {
    const req = store.get(key);
    req.onsuccess = () => {
      const rec = req.result;
      if (rec) store.put({ ...rec, at: Date.now() });
    };
  }, undefined);
}

/**
 * The figure kept under a path, or null. Read in a readonly transaction and
 * answered on the read's success; a figure found counts as used now (touch,
 * not awaited), so eviction keeps it longer.
 * @param {string} key the figure's path (the tool result's figure_path)
 * @returns {Promise<{svg: string, caption: string} | null>} never rejects
 */
export function getFigure(key) {
  if (typeof key !== 'string' || !key) return Promise.resolve(null);
  return inStore('readonly', (store, keep) => {
    const req = store.get(key);
    req.onsuccess = () => {
      const rec = req.result;
      if (!rec || typeof rec.svg !== 'string' || !rec.svg) { keep(null); return; }
      keep({ svg: rec.svg, caption: typeof rec.caption === 'string' ? rec.caption : '' });
      void touch(key);
    };
  }, null);
}

/**
 * Delete the figures kept under these paths (a deleted chat's figures that no
 * other saved chat shows).
 * @param {string[]} keys the figures' paths (the tool results' figure_path)
 * @returns {Promise<void>} never rejects
 */
export function deleteFigures(keys) {
  const list = Array.isArray(keys) ? keys.filter((k) => typeof k === 'string' && k) : [];
  if (!list.length) return Promise.resolve();
  return inStore('readwrite', (store) => {
    for (const key of list) store.delete(key);
  }, undefined);
}

/**
 * Delete every kept figure (the chat history is being cleared).
 * @returns {Promise<void>} never rejects
 */
export function clearFigures() {
  clears += 1;
  return inStore('readwrite', (store) => { store.clear(); }, undefined);
}
