/* Chat store + threads (localStorage, à la Claude history) and the composer's
   send path. A brief appends a turn, runs it (demo or live), and persists the
   events so reopening a chat replays them. */

import { BRIDGE } from './store.js';
import { escapeHtml, autosize } from './util.js';
import { runLive, runDemo, replayEvents } from './run.js';
import { buildAgentBrief, getPendingAttachments, clearPendingAttachments } from './attach.js';
import { projConnected } from './api.js';

const CHATS_LS = 'oe_chats';
let activeChat = null;     // { id, title, createdAt, updatedAt, turns: [...] }
let activeChatId = null;
let sending = false;       // a run is in flight; gates concurrent sends

function loadChats() { try { return JSON.parse(localStorage.getItem(CHATS_LS) || '[]'); } catch (e) { return []; } }
function saveChats(chats) { try { localStorage.setItem(CHATS_LS, JSON.stringify(chats)); } catch (e) {} }
function genId() { return 'c' + Date.now().toString(36) + Math.random().toString(36).slice(2, 7); }

function persistChat(chat) {
  if (!chat) return;
  const chats = loadChats().filter((c) => c.id !== chat.id);
  chats.push(chat);
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

function openChat(id) {
  const c = loadChats().find((x) => x.id === id);
  if (!c) return;
  activeChat = c; activeChatId = c.id;
  renderThread();
  setTopTitle(c.title);
  renderChatList();
  const sb = document.getElementById('sidebar'); if (sb) sb.classList.remove('open');
}

function deleteChat(id) {
  saveChats(loadChats().filter((c) => c.id !== id));
  if (activeChatId === id) { activeChat = null; activeChatId = null; newChat(); }
  else renderChatList();
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
  if (btn) btn.disabled = on;
}

function handleSend(brief, attachments) {
  const atts = attachments || [];
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

  const agentBrief = buildAgentBrief(brief, atts);  // file text appended for the agent
  const onEvent = (ev) => aTurn.events.push(ev);
  const done = () => { sending = false; setSendingUI(false); chat.updatedAt = Date.now(); persistChat(chat); renderChatList(); };
  sending = true; setSendingUI(true);
  if (BRIDGE.live && projConnected()) runLive(body, agentBrief, history, onEvent).then(done, done);
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
