/* Chat store + threads (localStorage, à la Claude history) and the composer's
   send path. A brief appends a turn, runs it (demo or live), and persists the
   events so reopening a chat replays them. */

import { BRIDGE } from './store.js';
import { escapeHtml, autosize } from './util.js';
import { runLive, runDemo, replayEvents } from './run.js';
import { buildAgentBrief, getPendingAttachments, clearPendingAttachments } from './attach.js';
import { parseSkillSlash } from './slash.js';
import { projConnected } from './api.js';
import { putFigure, clearFigures, deleteFigures } from './figstore.js';

const CHATS_LS = 'oe_chats';
let activeChat = null;     // { id, title, createdAt, updatedAt, turns: [...] }
let activeChatId = null;
let sending = false;       // a run is in flight; gates concurrent sends
let historyEpoch = 0;      // bumped by clearChatHistory: a run begun before a clear keeps no figure and saves no chat after it

function loadChats() { try { return JSON.parse(localStorage.getItem(CHATS_LS) || '[]'); } catch (e) { return []; } }
function genId() { return 'c' + Date.now().toString(36) + Math.random().toString(36).slice(2, 7); }

/* A chat as stored: its events without figure_svg (60 to 160 KB of SVG each,
   which would fill localStorage after about a dozen check chats). The event
   keeps the result's figure_path and figure_caption, so a replay shows the
   legend; only the live session's in-memory copy keeps the SVG, and the
   browser's figure store (figstore.js) a copy by figure_path, which a replay
   looks up. */
function storedChat(chat) {
  const strip = (ev) => {
    if (!ev || typeof ev !== 'object' || !('figure_svg' in ev)) return ev;
    const copy = { ...ev };
    delete copy.figure_svg;
    return copy;
  };
  const turns = (chat.turns || []).map((t) => (Array.isArray(t.events) ? { ...t, events: t.events.map(strip) } : t));
  return { ...chat, turns };
}

/* A short notice above the composer while the browser will not store the
   chats (its storage for the page is full or blocked), so a lost chat is not
   silent; hidden again once a save succeeds. */
function setSaveNotice(on) {
  let el = document.getElementById('chatSaveNotice');
  if (!on) { if (el) el.hidden = true; return; }
  if (!el) {
    const composer = document.getElementById('composer');
    if (!composer) return;
    el = document.createElement('div');
    el.id = 'chatSaveNotice';
    el.className = 'save-notice';
    el.setAttribute('role', 'status');
    el.textContent = "Chats are not being saved: this browser's storage for the page is full or blocked. Delete older chats to free space.";
    composer.insertBefore(el, composer.firstChild);
  }
  el.hidden = false;
}

/* Store the chats; on an error (the quota, mostly) drop figure_svg from every
   stored event, which a chat saved by an earlier version may still carry, and
   retry once; if that fails too, say so. True once stored. */
function saveChats(chats) {
  try {
    localStorage.setItem(CHATS_LS, JSON.stringify(chats));
    setSaveNotice(false);
    return true;
  } catch (e) { /* retried below without the figures */ }
  try {
    localStorage.setItem(CHATS_LS, JSON.stringify(chats.map(storedChat)));
    setSaveNotice(false);
    return true;
  } catch (e) {
    setSaveNotice(true);
    return false;
  }
}

/* The figure paths (figure_path) a chat's tool results name: the keys of its
   figures in the figure store. */
function figurePaths(chat) {
  const paths = new Set();
  const turns = chat && Array.isArray(chat.turns) ? chat.turns : [];
  for (const turn of turns) {
    if (!turn || !Array.isArray(turn.events)) continue;
    for (const ev of turn.events) {
      const inner = ev && ev.result && ev.result.result;
      if (inner && typeof inner.figure_path === 'string' && inner.figure_path) paths.add(inner.figure_path);
    }
  }
  return paths;
}

/* A live event's figure into the browser's figure store (figstore.js), under
   the result's figure_path, so the chat shows it again when reopened from the
   history. Not awaited: a store that fails or is missing changes nothing here. */
function keepFigure(ev) {
  if (!ev || typeof ev.figure_svg !== 'string' || !ev.figure_svg) return;
  const inner = ev.result && ev.result.result;
  const path = inner && typeof inner.figure_path === 'string' ? inner.figure_path : '';
  if (!path) return;
  const caption = typeof inner.figure_caption === 'string' ? inner.figure_caption : '';
  void putFigure(path, ev.figure_svg, caption);
}

function persistChat(chat) {
  if (!chat) return;
  const chats = loadChats().filter((c) => c.id !== chat.id);
  chats.push(storedChat(chat));
  saveChats(chats);
}

function ensureActiveChat() {
  if (!activeChat) {
    activeChat = { id: genId(), title: '', createdAt: Date.now(), updatedAt: Date.now(), turns: [] };
    activeChatId = activeChat.id;
  }
}

function finalOf(events) {
  const fin = (events || []).filter((e) => e.type === 'final').pop();
  return fin ? (fin.content || '') : '';
}

// Prior turns → [{role, content}] for the agent's multi-turn memory.
function historyForAgent(turns) {
  const out = [];
  for (const t of turns) {
    if (t.role === 'user') out.push({ role: 'user', content: t.text });
    else if (t.role === 'assistant') { const f = finalOf(t.events); if (f) out.push({ role: 'assistant', content: f }); }
  }
  return out;
}

function showEmpty() {
  const e = document.getElementById('chatEmpty'); const t = document.getElementById('chatThread');
  if (e) e.hidden = false;
  if (t) t.hidden = true;
}
function showThread() {
  const e = document.getElementById('chatEmpty'); const t = document.getElementById('chatThread');
  if (e) e.hidden = true;
  if (t) t.hidden = false;
}
function setTopTitle(title) { const el = document.getElementById('topbarTitle'); if (el) el.textContent = title || ''; }

// Create the DOM for one turn (user bubble + empty agent body); return the body.
function appendTurnDom(userText, attachments) {
  const thread = document.getElementById('chatThread');
  const card = document.createElement('div');
  card.className = 'transcript live';
  const files = (attachments && attachments.length)
    ? '<div class="msg-files">' + attachments.map((a) =>
        `<span class="att-chip${a.kind === 'image' ? ' is-image' : ''}"><span class="nm">${escapeHtml(a.name)}</span></span>`).join('') + '</div>'
    : '';
  card.innerHTML =
    '<div class="msg user run-step"><span class="role">You</span>' +
      '<div class="bubble">' + escapeHtml(userText) + files + '</div></div>' +
    '<div class="msg agent"><span class="role">OlmoEarth Agent</span><div class="agent-body"></div></div>';
  thread.appendChild(card);
  card.querySelector('.msg.user').scrollIntoView({ behavior: 'smooth', block: 'start' });
  return card.querySelector('.agent-body');
}

export function renderThread() {
  const thread = document.getElementById('chatThread');
  if (!thread) return;
  thread.innerHTML = '';
  if (!activeChat || !activeChat.turns.length) { showEmpty(); return; }
  showThread();
  for (let i = 0; i < activeChat.turns.length; i++) {
    const t = activeChat.turns[i];
    if (t.role !== 'user') continue;
    const body = appendTurnDom(t.text, t.files);
    const a = activeChat.turns[i + 1];
    if (a && a.role === 'assistant') { replayEvents(body, a.events); i++; }
  }
  const sc = document.getElementById('scroll'); if (sc) sc.scrollTop = sc.scrollHeight;
}

export function newChat() {
  activeChat = { id: genId(), title: '', createdAt: Date.now(), updatedAt: Date.now(), turns: [] };
  activeChatId = activeChat.id;
  const thread = document.getElementById('chatThread'); if (thread) thread.innerHTML = '';
  showEmpty();
  setTopTitle('');
  renderChatList();
  const input = document.getElementById('promptInput'); if (input) { input.value = ''; autosize(input); input.focus(); }
  const sb = document.getElementById('sidebar'); if (sb) sb.classList.remove('open');
  const sc = document.getElementById('scroll'); if (sc) sc.scrollTo({ top: 0, behavior: 'smooth' });
}

/* "Clear chat history": the stored chats and the figures kept for them. A run
   still streaming keeps no figure and saves no chat after this (historyEpoch). */
export function clearChatHistory() {
  historyEpoch += 1;
  try { localStorage.removeItem(CHATS_LS); } catch (e) { /* storage blocked: nothing stored */ }
  void clearFigures();
  newChat();
}

function openChat(id) {
  const c = loadChats().find((x) => x.id === id);
  if (!c) return;
  activeChat = c; activeChatId = c.id;
  renderThread();
  setTopTitle(c.title);
  renderChatList();
  const sb = document.getElementById('sidebar'); if (sb) sb.classList.remove('open');
}

/* Delete one saved chat, and the figures it showed that no other chat shows:
   not the remaining saved chats, nor the open one (it may hold a figure not
   saved yet). Its figures stay if the chats could not be saved without it. */
function deleteChat(id) {
  const chats = loadChats();
  const gone = chats.find((c) => c.id === id);
  const left = chats.filter((c) => c.id !== id);
  if (saveChats(left) && gone) {
    const shown = new Set();
    for (const c of left) figurePaths(c).forEach((p) => shown.add(p));
    if (activeChat && activeChat.id !== id) figurePaths(activeChat).forEach((p) => shown.add(p));
    const orphans = [...figurePaths(gone)].filter((p) => !shown.has(p));
    if (orphans.length) void deleteFigures(orphans);
  }
  if (activeChatId === id) { activeChat = null; activeChatId = null; newChat(); }
  else {
    // The space freed may let the open chat be stored, if saving it failed.
    if (activeChat && activeChat.turns.length) persistChat(activeChat);
    renderChatList();
  }
}

export function renderChatList() {
  const list = document.getElementById('chatList');
  if (!list) return;
  const chats = loadChats().sort((a, b) => (b.updatedAt || 0) - (a.updatedAt || 0));
  if (!chats.length) {
    list.innerHTML = '<div class="side-note">No saved chats yet. Send a brief below to start one.</div>';
    return;
  }
  list.innerHTML = chats.map((c) => `
    <button class="chat-item${c.id === activeChatId ? ' is-active' : ''}" data-id="${escapeHtml(c.id)}" title="${escapeHtml(c.title || 'Untitled chat')}">
      <svg viewBox="0 0 24 24" class="ic"><path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z"/></svg>
      <span class="chat-title">${escapeHtml(c.title || 'Untitled chat')}</span>
      <span class="chat-del" data-del="${escapeHtml(c.id)}" title="Delete chat" role="button">×</span>
    </button>`).join('');
  list.querySelectorAll('.chat-item').forEach((el) => {
    el.addEventListener('click', (e) => {
      if (e.target.closest('.chat-del')) return;
      openChat(el.dataset.id);
    });
  });
  list.querySelectorAll('.chat-del').forEach((el) => {
    el.addEventListener('click', (e) => { e.stopPropagation(); deleteChat(el.dataset.del); });
  });
}

// The composer's send path: append the turn, run it, persist + update history.
function setSendingUI(on) {
  const btn = document.querySelector('.send');
  if (!btn) return;
  btn.disabled = on;
  btn.classList.toggle('is-busy', on);  // hides the .ic arrow, dims, blocks input
  let sp = btn.querySelector('.spinner');
  if (on && !sp) { sp = document.createElement('span'); sp.className = 'spinner sm on-pink'; btn.appendChild(sp); }
  else if (!on && sp) sp.remove();
}

function handleSend(brief, attachments) {
  const atts = attachments || [];
  const epoch = historyEpoch;  // the history this run belongs to; a clear during the run ends it
  ensureActiveChat();
  const chat = activeChat;
  const history = historyForAgent(chat.turns);  // prior turns only
  chat.turns.push({ role: 'user', text: brief, files: atts.map((a) => ({ name: a.name, kind: a.kind })) });
  const aTurn = { role: 'assistant', events: [] };
  chat.turns.push(aTurn);
  if (!chat.title) chat.title = brief.slice(0, 60);
  chat.updatedAt = Date.now();

  showThread();
  const body = appendTurnDom(brief, atts);
  persistChat(chat); renderChatList(); setTopTitle(chat.title);

  // "/skill ..." routes to that skill server-side via forced_skill; the brief is
  // sent clean (no rewrite) and the displayed message stays as typed. When the
  // brief is only "/skill" (no text), fall back to a minimal directive so the
  // run still has a non-empty brief.
  const { skill: forcedSkill, brief: rest } = parseSkillSlash(brief);
  const cleanBrief = rest || (forcedSkill ? `Use the ${forcedSkill} skill for this request.` : brief);
  const agentBrief = buildAgentBrief(cleanBrief, atts);
  const onEvent = (ev) => { aTurn.events.push(ev); if (epoch === historyEpoch) keepFigure(ev); };
  const done = () => {
    sending = false; setSendingUI(false);
    if (epoch !== historyEpoch) return;  // the history was cleared during the run: nothing saved back
    chat.updatedAt = Date.now(); persistChat(chat); renderChatList();
  };
  sending = true; setSendingUI(true);
  if (BRIDGE.live && projConnected()) runLive(body, agentBrief, history, onEvent, forcedSkill).then(done, done);
  else { runDemo(body, agentBrief, onEvent); setTimeout(done, 5200); }
}

export function wirePrompt() {
  const form = document.getElementById('promptForm');
  const input = document.getElementById('promptInput');
  if (input) {
    input.addEventListener('input', () => autosize(input));
    input.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        if (form.requestSubmit) form.requestSubmit();
        else form.dispatchEvent(new Event('submit', { cancelable: true }));
      }
    });
  }
  if (form) {
    form.addEventListener('submit', (e) => {
      e.preventDefault();
      const brief = (input && input.value.trim()) || '';
      const atts = getPendingAttachments();
      if (!brief && !atts.length) return;
      if (sending) return;  // a run is already in flight; wait for it to finish
      if (input) { input.value = ''; autosize(input); }
      clearPendingAttachments();
      handleSend(brief || '(see attached files)', atts);
    });
  }
}
