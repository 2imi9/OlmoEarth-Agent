/* OlmoEarth Agent web UI entry point. Wires the app shell (menu, new chat, user
   menu, Studio key, collapsibles, bridge detection) and delegates the rest to
   feature modules. Static demo by default; upgrades to the live agent when
   served by olmoearth_agent.serve. No build step - native ES modules. */

import { BRIDGE } from './store.js';
import { apiHealth, agentMaxTurns, clearApiCache } from './api.js';
import { renderCards, wireCards } from './skills.js';
import { newChat, renderChatList, wirePrompt } from './chat.js';
import { wireSlash } from './slash.js';
import { wireTabs, wireExamples } from './landing.js';
import { wireAttach } from './attach.js';
import { wireAoi } from './aoi.js';
import { wireLlmSubtab, wireLlmNudge, updateLlmNudge } from './llm.js';
import { wireResults } from './results.js';
import { wireSwitcher, refreshActiveSegment } from './switcher.js';

// Re-rendered by wireKey so detectBridge() can refresh the key card once the
// live/demo mode is known.
let renderKeyState = () => {};

function wireMenu() {
  const btn = document.getElementById('menuBtn');
  const sidebar = document.getElementById('sidebar');
  if (btn && sidebar) btn.addEventListener('click', () => sidebar.classList.toggle('open'));
}

function wireNewChat() {
  const btn = document.getElementById('newChatBtn');
  if (btn) btn.addEventListener('click', () => newChat());
  const clear = document.getElementById('clearChats');  // now lives under New chat
  if (clear) clear.addEventListener('click', () => {
    try { localStorage.removeItem('oe_chats'); } catch (e) {}
    newChat();
  });
}

/* General agent settings + legal links (popover above the user chip). */
function wireUserMenu() {
  const chip = document.getElementById('userChip');
  const menu = document.getElementById('userMenu');
  if (!chip || !menu) return;
  const close = () => { menu.hidden = true; chip.setAttribute('aria-expanded', 'false'); };
  chip.addEventListener('click', (e) => {
    e.stopPropagation();
    menu.hidden = !menu.hidden;
    chip.setAttribute('aria-expanded', String(!menu.hidden));
  });
  menu.addEventListener('click', (e) => e.stopPropagation());
  document.addEventListener('click', () => { if (!menu.hidden) close(); });
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape') close(); });
  const sel = document.getElementById('maxTurns');
  if (sel) {
    sel.value = String(agentMaxTurns());
    sel.addEventListener('change', () => { try { localStorage.setItem('oe_max_turns', sel.value); } catch (e) {} });
  }
  wireLlmSubtab();
}

/* Bring-your-own-key: the user pastes their own Studio API key; stored
   client-side only (localStorage). Real email login is future work. */
function wireKey() {
  const LS = 'oe_studio_key';
  const connect = document.getElementById('keyConnect');
  const connected = document.getElementById('keyConnected');
  const form = document.getElementById('keyForm');
  const input = document.getElementById('keyInput');
  const dot = document.getElementById('statusDot');
  const status = document.getElementById('userStatus');
  const maskKey = (k) => (k.length <= 6 ? k : k.slice(0, 6) + '••••' + k.slice(-2));
  const get = () => { try { return localStorage.getItem(LS) || ''; } catch (e) { return ''; } };
  // Changing the key invalidates every cached read (different account / scope).
  const set = (v) => { try { v ? localStorage.setItem(LS, v) : localStorage.removeItem(LS); } catch (e) {} clearApiCache(); };
  function render() {
    const k = get(); const on = !!k;
    if (connect) connect.hidden = on;
    if (connected) connected.hidden = !on;
    if (dot) dot.classList.toggle('is-on', on);
    if (status) status.textContent = on ? (BRIDGE.live ? 'Connected · live' : 'Connected · demo') : 'Not connected';
    const topBtn = document.getElementById('topKeyBtn');
    if (topBtn) topBtn.textContent = on ? 'Studio key' : 'Add API key';
    if (on) {
      const label = document.getElementById('keyStatusLabel');
      const body = document.getElementById('keyConnBody');
      if (label) label.textContent = BRIDGE.live ? 'Studio connected' : 'Connected · demo';
      if (body) body.innerHTML = BRIDGE.live
        ? 'Connected as <code>' + maskKey(k) + '</code>. Briefs run the <strong>real agent</strong> against your Studio account through the local bridge.'
        : 'Key saved as <code>' + maskKey(k) + "</code>, but this preview <strong>doesn't call Studio yet</strong>. The data shown is sample, not your account.";
    }
    refreshActiveSegment();  // re-render the visible pane (Projects/Areas load once a key is connected)
  }
  renderKeyState = render;
  if (form) form.addEventListener('submit', (e) => {
    e.preventDefault();
    const v = (input.value || '').trim();
    if (!v) return;
    set(v); input.value = ''; render();
  });
  const dis = document.getElementById('keyDisconnect');
  if (dis) dis.addEventListener('click', () => { set(''); render(); });
  const top = document.getElementById('topKeyBtn');
  const pop = document.getElementById('keyPop');
  if (top && pop) {
    const closePop = () => { pop.hidden = true; top.setAttribute('aria-expanded', 'false'); };
    top.addEventListener('click', (e) => {
      e.stopPropagation();
      pop.hidden = !pop.hidden;
      top.setAttribute('aria-expanded', String(!pop.hidden));
      if (!pop.hidden && input && connect && !connect.hidden) input.focus();
    });
    pop.addEventListener('click', (e) => e.stopPropagation());
    document.addEventListener('click', () => { if (!pop.hidden) closePop(); });
    document.addEventListener('keydown', (e) => { if (e.key === 'Escape') closePop(); });
  }
  render();
}

/* ── Live bridge detection ───────────────────────────────────────────────
   When served by olmoearth_agent.serve, /api/health answers and we switch to
   the live agent. Opened as a static file, it 404s and the demo runs. */
async function detectBridge() {
  try {
    const data = await apiHealth();
    if (data) {
      BRIDGE.live = !!(data && data.ok) && data.mode === 'live';
      BRIDGE.claudeAvailable = !!(data && data.claude_available);
      // Only nudge when the bridge explicitly reports the local model down.
      BRIDGE.llmLocalUp = !(data && data.llm_local_up === false);
    }
  } catch (e) { BRIDGE.live = false; }
  BRIDGE.checked = true;
}

document.addEventListener('DOMContentLoaded', () => {
  renderCards();
  wireCards();
  wireNewChat();
  wireTabs();
  wireExamples();
  wireAttach();
  wireAoi();
  wireSlash();   // before wirePrompt: its Enter handler must run first
  wirePrompt();
  wireMenu();
  wireUserMenu();
  wireLlmNudge();
  wireSwitcher();   // segmented Projects · Areas · Results switcher
  wireResults();    // saved-results listener (comparisons + file outputs)
  wireKey();        // initial render (demo assumptions) + refreshActiveSegment
  renderChatList();
  newChat();        // start on a fresh empty chat (landing visible)
  // Upgrade to live mode if the bridge is serving this page, then re-render.
  detectBridge().then(() => { renderKeyState(); updateLlmNudge(); });
});
