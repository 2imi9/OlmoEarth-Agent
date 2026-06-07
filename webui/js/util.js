/* Pure DOM / format helpers, no app state. */

export function escapeHtml(s) {
  return String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
}

export function autosize(el) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 200) + 'px';
}

export function typeText(el, text, speed) {
  el.textContent = '';
  el.classList.add('caret');
  let i = 0;
  const tick = () => {
    if (i <= text.length) { el.textContent = text.slice(0, i); i += 1; setTimeout(tick, speed || 14); }
    else el.classList.remove('caret');
  };
  tick();
}

export function shortId(s) { s = String(s || ''); return s.length > 8 ? s.slice(0, 8) : s; }

/* Skeleton-shimmer placeholder rows (dot + line) for list/tree loading states. */
export function skelRows(n) {
  let s = '<div class="skel-list">';
  for (let i = 0; i < (n || 3); i++) {
    s += '<div class="skel-row"><span class="skeleton skel-dot"></span>' +
         '<span class="skeleton skel-line' + (i % 2 ? ' short' : '') + '"></span></div>';
  }
  return s + '</div>';
}
