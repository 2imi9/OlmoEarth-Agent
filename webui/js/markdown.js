/* Minimal, safe Markdown → HTML for agent answers. The model replies in
   GitHub-flavored Markdown; render it (tables, bold, lists, code) instead of
   showing raw ** and | . Text is HTML-escaped FIRST, then tokenized, so model
   output can't inject markup. */

import { escapeHtml } from './util.js';

function mdInline(text) {
  let s = escapeHtml(text);
  s = s.replace(/`([^`]+)`/g, '<code>$1</code>');
  s = s.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
  s = s.replace(/(^|[^*\w])\*(\S(?:[^*\n]*\S)?)\*(?![*\w])/g, '$1<em>$2</em>');
  s = s.replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
  return s;
}
function mdSplitRow(line) {
  let s = line.trim();
  if (s.charAt(0) === '|') s = s.slice(1);
  if (s.charAt(s.length - 1) === '|') s = s.slice(0, -1);
  return s.split('|').map((c) => c.trim());
}
function mdIsTableStart(lines, i) {
  return i + 1 < lines.length && lines[i].indexOf('|') >= 0 &&
    lines[i + 1].indexOf('-') >= 0 && /^\s*\|?[\s:|-]+\|?\s*$/.test(lines[i + 1]);
}
export function renderMarkdown(src) {
  const lines = String(src).replace(/\r\n?/g, '\n').split('\n');
  const out = [];
  let i = 0;
  const isBlock = (l, j) => /^```/.test(l) || /^#{1,4}\s+/.test(l) ||
    /^\s*[-*+]\s+/.test(l) || /^\s*\d+\.\s+/.test(l) || mdIsTableStart(lines, j);
  while (i < lines.length) {
    const line = lines[i];
    if (/^```/.test(line)) {
      const buf = []; i++;
      while (i < lines.length && !/^```/.test(lines[i])) { buf.push(lines[i]); i++; }
      i++;
      out.push('<pre class="md-pre"><code>' + escapeHtml(buf.join('\n')) + '</code></pre>');
      continue;
    }
    if (mdIsTableStart(lines, i)) {
      const header = mdSplitRow(line); i += 2;
      const rows = [];
      while (i < lines.length && lines[i].indexOf('|') >= 0 && lines[i].trim() !== '') { rows.push(mdSplitRow(lines[i])); i++; }
      const th = header.map((h) => '<th>' + mdInline(h) + '</th>').join('');
      const trs = rows.map((r) => '<tr>' + header.map((_, j) => '<td>' + mdInline(r[j] || '') + '</td>').join('') + '</tr>').join('');
      out.push('<div class="md-table-wrap"><table class="md-table"><thead><tr>' + th + '</tr></thead><tbody>' + trs + '</tbody></table></div>');
      continue;
    }
    const h = /^(#{1,4})\s+(.*)$/.exec(line);
    if (h) { out.push('<div class="md-h md-h' + h[1].length + '">' + mdInline(h[2]) + '</div>'); i++; continue; }
    if (/^\s*[-*+]\s+/.test(line)) {
      const items = [];
      while (i < lines.length && /^\s*[-*+]\s+/.test(lines[i])) { items.push('<li>' + mdInline(lines[i].replace(/^\s*[-*+]\s+/, '')) + '</li>'); i++; }
      out.push('<ul class="md-ul">' + items.join('') + '</ul>');
      continue;
    }
    if (/^\s*\d+\.\s+/.test(line)) {
      const items = [];
      while (i < lines.length && /^\s*\d+\.\s+/.test(lines[i])) { items.push('<li>' + mdInline(lines[i].replace(/^\s*\d+\.\s+/, '')) + '</li>'); i++; }
      out.push('<ol class="md-ol">' + items.join('') + '</ol>');
      continue;
    }
    if (line.trim() === '') { i++; continue; }
    const para = [];
    while (i < lines.length && lines[i].trim() !== '' && !isBlock(lines[i], i)) { para.push(lines[i]); i++; }
    out.push('<p class="md-p">' + mdInline(para.join('\n')).replace(/\n/g, '<br>') + '</p>');
  }
  return out.join('') || '<p class="md-p"></p>';
}
