/* In-chat skill slash-commands. Type "/" in the composer to pick one of the 16
   skills (like Claude Code's /commands); the picked skill is then routed to the
   agent. The displayed message stays as the user typed it; the brief is sent
   CLEAN and the skill is pinned SERVER-SIDE via a structured `forced_skill`
   field (see serve.py / LeadAgent), instead of rewriting the brief into a "use
   the X skill" directive. In demo mode (no server) the clean brief still drives
   the canned scenario. */

import { SKILL_LIST } from './skills.js';
import { escapeHtml, autosize } from './util.js';

const bySlug = new Map(SKILL_LIST.map((s) => [s.slug, s]));

/* Parse "/slug rest" into { skill, brief }: the recognized skill slug (routed to
   the agent as `forced_skill`) and the brief with the "/slug" stripped, so what
   the agent sees is clean. Returns { skill: null, brief } unchanged when it
   isn't a recognized skill slash-command (so a stray "/" is sent verbatim). */
export function parseSkillSlash(brief) {
  const m = /^\s*\/([a-zA-Z0-9-]+)\b[ \t]*([\s\S]*)$/.exec(brief || '');
  if (!m) return { skill: null, brief };
  const skill = bySlug.get(m[1].toLowerCase());
  if (!skill) return { skill: null, brief };
  return { skill: skill.slug, brief: (m[2] || '').trim() };
}

/* Wire the composer's "/" menu. MUST run before wirePrompt() so this keydown
   handler is registered first and can stopImmediatePropagation the Enter key
   (selecting a skill) before the submit handler sees it. */
export function wireSlash() {
  const input = document.getElementById('promptInput');
  const form = document.getElementById('promptForm');
  if (!input || !form) return;
  const anchor = form.querySelector('.prompt') || form;

  const menu = document.createElement('div');
  menu.className = 'slash-menu';
  menu.hidden = true;
  anchor.appendChild(menu);

  let items = [];
  let active = -1;
  const isOpen = () => !menu.hidden;
  const close = () => { menu.hidden = true; items = []; active = -1; };

  const render = (q) => {
    const ql = q.toLowerCase();
    items = SKILL_LIST.filter((s) => s.slug.includes(ql)).slice(0, 8);
    if (!items.length) { close(); return; }
    active = 0;
    menu.innerHTML =
      '<div class="slash-head">Skills · ↑↓ then ↵</div>' +
      items.map((s, i) =>
        `<button type="button" class="slash-item${i === 0 ? ' is-active' : ''}" data-i="${i}">` +
          `<span class="slash-slug">/${escapeHtml(s.slug)}</span>` +
          `<span class="slash-desc">${escapeHtml(s.desc)}</span>` +
        '</button>').join('');
    menu.hidden = false;
  };

  const setActive = (i) => {
    if (!items.length) return;
    active = (i + items.length) % items.length;
    [...menu.querySelectorAll('.slash-item')].forEach((el, idx) => el.classList.toggle('is-active', idx === active));
    const el = menu.querySelector('.slash-item.is-active');
    if (el) el.scrollIntoView({ block: 'nearest' });
  };

  const choose = (i) => {
    const s = items[i];
    if (!s) return;
    input.value = `/${s.slug} `;
    close();
    input.focus();
    autosize(input);
    const len = input.value.length;
    input.setSelectionRange(len, len);
  };

  input.addEventListener('input', () => {
    const m = /^\/([a-zA-Z0-9-]*)$/.exec(input.value);  // "/" + partial, no space, at the start
    if (m) render(m[1]); else close();
  });

  input.addEventListener('keydown', (e) => {
    if (!isOpen()) return;
    if (e.key === 'ArrowDown') { e.preventDefault(); e.stopImmediatePropagation(); setActive(active + 1); }
    else if (e.key === 'ArrowUp') { e.preventDefault(); e.stopImmediatePropagation(); setActive(active - 1); }
    else if (e.key === 'Enter' || e.key === 'Tab') { e.preventDefault(); e.stopImmediatePropagation(); choose(active); }
    else if (e.key === 'Escape') { e.preventDefault(); e.stopImmediatePropagation(); close(); }
  });

  // mousedown (not click) so focus stays in the input.
  menu.addEventListener('mousedown', (e) => {
    const btn = e.target.closest('.slash-item');
    if (!btn) return;
    e.preventDefault();
    choose(Number(btn.dataset.i));
  });

  document.addEventListener('click', (e) => { if (isOpen() && !anchor.contains(e.target)) close(); });
}
