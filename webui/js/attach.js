/* File attachments. Attach text files (GeoJSON, code, CSV, ...) and PDFs; their
   extracted text is appended to the brief so the agent reads them. The local
   model is text-only, so images are accepted but flagged as not readable. A
   Studio result dropped from the project tree attaches as context too. */

import { escapeHtml, shortId } from './util.js';
import { apiArea } from './api.js';

const _TEXT_EXT = /\.(geojson|json|csv|tsv|txt|md|markdown|ya?ml|toml|xml|html?|css|py|js|ts|tsx|jsx|sh|r|sql|ipynb|log|cfg|ini)$/i;
const _MAX_FILE_CHARS = 60000;
const _MAX_TOTAL_CHARS = 120000;
let pendingAttachments = [];  // {name, kind: 'text'|'pdf'|'image'|'result', text, error?}

export function getPendingAttachments() { return pendingAttachments.slice(); }
export function clearPendingAttachments() { pendingAttachments = []; renderAttachmentChips(); }

function _readText(file) {
  return new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => resolve(String(r.result || ''));
    r.onerror = () => reject(r.error || new Error('read failed'));
    r.readAsText(file);
  });
}

let _pdfjs = null;
async function _readPdf(file) {
  if (!_pdfjs) {  // lazy-load pdf.js from CDN only when a PDF is attached
    const base = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/4.4.168';
    _pdfjs = await import(`${base}/pdf.min.mjs`);
    _pdfjs.GlobalWorkerOptions.workerSrc = `${base}/pdf.worker.min.mjs`;
  }
  const doc = await _pdfjs.getDocument({ data: await file.arrayBuffer() }).promise;
  const pages = [];
  for (let p = 1; p <= doc.numPages; p++) {
    const tc = await (await doc.getPage(p)).getTextContent();
    pages.push(tc.items.map((i) => i.str).join(' '));
    if (pages.join('\n').length > _MAX_FILE_CHARS) break;
  }
  return pages.join('\n').slice(0, _MAX_FILE_CHARS);
}

function _classifyFile(file) {
  if ((file.type || '').startsWith('image/')) return 'image';
  if (/\.pdf$/i.test(file.name) || file.type === 'application/pdf') return 'pdf';
  return 'text';  // text files + unknown: read as text
}

async function addFiles(fileList) {
  for (const file of Array.from(fileList || [])) {
    const att = { name: file.name, kind: _classifyFile(file), text: '' };
    try {
      if (att.kind === 'pdf') att.text = await _readPdf(file);
      else if (att.kind === 'text') att.text = (await _readText(file)).slice(0, _MAX_FILE_CHARS);
    } catch (e) {
      att.error = att.kind === 'pdf' ? 'could not extract PDF text' : 'could not read file';
    }
    pendingAttachments.push(att);
  }
  renderAttachmentChips();
}

function renderAttachmentChips() {
  const box = document.getElementById('attachments');
  if (!box) return;
  box.innerHTML = pendingAttachments.map((a, i) => {
    const tag = a.kind === 'image' ? ' (image, not read)'
      : a.kind === 'aoi' ? (a.stored ? ' (stored)' : ' (not stored)')
      : a.error ? ` (${a.error})` : a.kind === 'pdf' ? ' (pdf)' : '';
    const cls = a.kind === 'image' ? ' is-image' : a.kind === 'aoi' ? ' is-aoi' : '';
    return `<span class="att-chip${cls}">` +
      `<span class="nm">${escapeHtml(a.name)}${tag}</span>` +
      `<span class="x" data-rm="${i}" title="Remove" role="button">x</span></span>`;
  }).join('');
  box.querySelectorAll('.x[data-rm]').forEach((x) => {
    x.addEventListener('click', () => { pendingAttachments.splice(Number(x.dataset.rm), 1); renderAttachmentChips(); });
  });
}

/* A drawn AOI attaches like a file: a chip in the composer + a structured
   block appended to the brief so the agent can feed it to AOI-needing skills
   (area_id -> submit_prediction, bbox -> bbox-based tools). */
export function addAoiAttachment(aoi) {
  if (!aoi || (!aoi.geom && !aoi.area_id)) return;
  const bbox = Array.isArray(aoi.bbox) ? aoi.bbox.map((n) => Number(n).toFixed(5)).join(', ') : '';
  const lines = ['The user drew an area of interest (AOI) on the map.'];
  if (aoi.stored && aoi.area_id) {
    lines.push('It is stored in OlmoEarth Studio.');
    lines.push('area_id: ' + aoi.area_id + '  (use this for olmoearth_submit_prediction)');
    if (aoi.project_id) lines.push('project_id: ' + aoi.project_id);
  } else {
    lines.push('It is NOT stored in Studio (no project/key); store it first if a prediction needs an area_id.');
  }
  if (bbox) lines.push('bbox (min_lon, min_lat, max_lon, max_lat): ' + bbox + '  (use for bbox-based tools)');
  lines.push('Reference the area_id / bbox in tool calls; do not print raw coordinates back to the user.');
  pendingAttachments.push({
    name: 'AOI: ' + (aoi.name || 'drawn area'),
    kind: 'aoi',
    stored: !!aoi.stored,
    text: lines.join('\n'),
  });
  renderAttachmentChips();
}

/* An AOI dragged in from the Projects tree carries only {id, name,
   project_id} (the tree list omits geometry). Fetch its geom + bbox, then
   attach it like a drawn area. A demo-tree area carries its geom inline. */
async function addAoiFromDrag(data) {
  if (!data || !data.id) return;
  if (data.geom) {
    addAoiAttachment({ name: data.name, geom: data.geom, bbox: data.bbox, area_id: data.id, project_id: data.project_id, stored: !data.demo });
    return;
  }
  try {
    const a = await apiArea(data.id);
    addAoiAttachment({ name: a.name || data.name, geom: a.geom, bbox: a.bbox, area_id: a.id, project_id: a.project_id || data.project_id, stored: true });
  } catch (e) {
    // Couldn't fetch the geometry; attach the id anyway (still usable for predictions).
    addAoiAttachment({ name: data.name, area_id: data.id, project_id: data.project_id, stored: true });
  }
}

function addResultAttachment(r) {
  const text = [
    'OlmoEarth Studio prediction result.',
    'result_id: ' + (r.id || ''),
    'properties: ' + (r.properties || '(none)'),
    'format: ' + (r.format || 'unknown'),
    'tile_url (XYZ template): ' + (r.tile_url || '(none)'),
  ].join('\n');
  pendingAttachments.push({
    name: 'result ' + shortId(r.id || '') + (r.format ? ' (' + r.format + ')' : ''),
    kind: 'result',
    text,
  });
  renderAttachmentChips();
}

export function buildAgentBrief(brief, atts) {
  if (!atts || !atts.length) return brief;
  let budget = _MAX_TOTAL_CHARS;
  const blocks = atts.map((a) => {
    if (a.kind === 'image') return `--- ${a.name} (image) ---\n[image attachment; the text-only model cannot read images]`;
    if (a.error || !a.text) return `--- ${a.name} ---\n[${a.error || 'no text extracted'}]`;
    const body = a.text.slice(0, Math.max(0, budget));
    budget -= body.length;
    return `--- ${a.name} (${a.kind}) ---\n${body}${body.length < a.text.length ? '\n[truncated]' : ''}`;
  });
  return `${brief}\n\nAttached files:\n${blocks.join('\n\n')}`;
}

export function wireAttach() {
  const btn = document.getElementById('attachBtn');
  const input = document.getElementById('fileInput');
  const composer = document.getElementById('composer');
  if (btn && input) {
    btn.addEventListener('click', () => input.click());
    input.addEventListener('change', () => { addFiles(input.files); input.value = ''; });
  }
  if (composer) {
    composer.addEventListener('dragover', (e) => { e.preventDefault(); composer.classList.add('drop'); });
    composer.addEventListener('dragleave', () => composer.classList.remove('drop'));
    composer.addEventListener('drop', (e) => {
      e.preventDefault();
      composer.classList.remove('drop');
      const dt = e.dataTransfer;
      if (!dt) return;
      const resultJson = dt.getData('application/x-oe-result');
      if (resultJson) { try { addResultAttachment(JSON.parse(resultJson)); } catch (err) {} return; }
      const aoiJson = dt.getData('application/x-oe-aoi');
      if (aoiJson) { try { addAoiFromDrag(JSON.parse(aoiJson)); } catch (err) {} return; }
      if (dt.files && dt.files.length) addFiles(dt.files);
    });
  }
}
