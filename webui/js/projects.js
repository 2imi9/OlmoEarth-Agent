/* Projects drill-down tree. Live: project → predictions (grouped by model_id
   into a synthetic Model level) → predictions → prediction-results, lazy-loaded
   on expand. Demo: a canned tree from the sample projects. */

import { BRIDGE } from './store.js';
import { escapeHtml, shortId } from './util.js';
import { projConnected, apiProjects, apiPredictions, apiResults } from './api.js';

const PROJ_ICONS = {
  map:  '<path d="M9 3L3 6v15l6-3 6 3 6-3V3l-6 3-6-3z"/><path d="M9 3v15M15 6v15"/>',
  drop: '<path d="M12 3s6 6.5 6 11a6 6 0 01-12 0c0-4.5 6-11 6-11z"/>',
  trend:'<path d="M4 17l5-5 3 3 7-8"/><path d="M15 7h5v5"/>',
  leaf: '<path d="M5 19c0-7 5-12 14-12 0 9-5 14-12 12z"/><path d="M9 15c2-3 4-4 7-5"/>',
  sun:  '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/>',
};
const NODE_ICONS = {
  model:      '<rect x="4" y="4" width="16" height="16" rx="2"/><path d="M9 9h6v6H9z"/>',
  embeddings: '<circle cx="6" cy="7" r="1.5"/><circle cx="12" cy="5" r="1.5"/><circle cx="18" cy="8" r="1.5"/><circle cx="8" cy="13" r="1.5"/><circle cx="15" cy="14" r="1.5"/><circle cx="19" cy="17" r="1.5"/><circle cx="6" cy="19" r="1.5"/><circle cx="12" cy="20" r="1.5"/>',
  prediction: '<path d="M4 17l5-5 3 3 7-8"/><path d="M15 7h5v5"/>',
  result:     '<rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/>',
};

/* Studio model_type -> a friendly badge label and a coarse kind for the
   icon/badge colour. The tree's headline distinction is a fine-tuned model vs
   an embeddings run; unknown types fall back to a title-cased label. */
const MODEL_TYPE_LABELS = {
  fine_tuned: 'Fine-tuned',
  embeddings: 'Embeddings',
  embedding: 'Embeddings',
  pretrained: 'Foundation',
  foundation: 'Foundation',
  foundation_model: 'Foundation',
};
function modelTypeLabel(t) {
  if (!t) return '';
  return MODEL_TYPE_LABELS[t] || String(t).replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());
}
function modelTypeKind(t) { return /embed/i.test(t || '') ? 'embeddings' : 'model'; }
const PROJECTS = [
  { id: 'karst',    name: 'PA Karst',                    meta: '12', icon: 'map'   },
  { id: 'ches',     name: 'Chesapeake - water quality',  meta: '5',  icon: 'drop'  },
  { id: 'potomac',  name: 'Potomac - change detection',  meta: '8',  icon: 'trend' },
  { id: 'mangrove', name: 'Mangrove extent - Indonesia', meta: '3',  icon: 'leaf'  },
  { id: 'solar',    name: 'Solar arrays - California',   meta: '2',  icon: 'sun', model: 'embeddings' },
];

function pickProjIcon(name) {
  const n = (name || '').toLowerCase();
  if (/water|river|lake|chesa|quality|hydro|wetland/.test(n)) return 'drop';
  if (/change|trend|detect|delta|monitor/.test(n)) return 'trend';
  if (/forest|mangrove|veg|crop|tree|land|alfalfa/.test(n)) return 'leaf';
  if (/solar|energy|sun|panel/.test(n)) return 'sun';
  return 'map';
}

function updateProjTag() {
  const tag = document.getElementById('projTag');
  if (!tag) return;
  if (!projConnected()) { tag.hidden = true; return; }
  tag.hidden = false;
  tag.textContent = BRIDGE.live ? 'live' : 'sample';
  tag.title = BRIDGE.live ? 'Your live Studio account' : 'Demo data: not your live Studio account';
}

function treeGlyph(node) {
  if (node.kind === 'model' && modelTypeKind(node.modelType) === 'embeddings') return NODE_ICONS.embeddings;
  return (node.icon && PROJ_ICONS[node.icon]) || NODE_ICONS[node.kind] || PROJ_ICONS.map;
}

function makeTreeNode(node) {
  const el = document.createElement('div');
  el.className = 'tree-node' + (node.leaf ? ' leaf' : '');
  const status = node.status ? '<span class="pred-status ' + escapeHtml(node.status) + '">' + escapeHtml(node.status) + '</span>' : '';
  const badge = node.badge
    ? '<span class="model-type ' + escapeHtml(modelTypeKind(node.modelType)) + '"'
      + (node.foundation ? ' title="' + escapeHtml(node.foundation) + '"' : '')
      + '>' + escapeHtml(node.badge) + '</span>'
    : '';
  const meta = (node.meta != null && node.meta !== '') ? '<span class="tw-meta">' + escapeHtml(String(node.meta)) + '</span>' : '';
  el.innerHTML =
    '<button class="tree-row" type="button">' +
      '<svg viewBox="0 0 24 24" class="tw-caret"><path d="M9 6l6 6-6 6"/></svg>' +
      '<svg viewBox="0 0 24 24" class="ic tw-icon">' + treeGlyph(node) + '</svg>' +
      '<span class="tw-label" title="' + escapeHtml(node.name) + '">' + escapeHtml(node.name) + '</span>' +
      badge + status + meta +
    '</button>' +
    '<div class="tree-children" hidden></div>';
  const row = el.querySelector('.tree-row');
  const childBox = el.querySelector('.tree-children');
  if (node.leaf) {
    row.addEventListener('click', () => {
      const input = document.getElementById('promptInput');
      if (input) input.focus();
    });
    if (node.kind === 'result' && node.result) {
      row.draggable = true;
      row.title = 'Drag into the chat to attach this result';
      row.addEventListener('dragstart', (e) => {
        e.dataTransfer.setData('application/x-oe-result', JSON.stringify(node.result));
        e.dataTransfer.setData('text/plain', node.result.tile_url || node.name);
        e.dataTransfer.effectAllowed = 'copy';
      });
    }
  } else {
    row.addEventListener('click', () => toggleNode(el, node, childBox));
  }
  return el;
}

async function toggleNode(el, node, childBox) {
  const open = el.classList.toggle('open');
  childBox.hidden = !open;
  if (!open || childBox.dataset.loaded) return;
  childBox.dataset.loaded = '1';
  childBox.innerHTML = '<div class="proj-empty">Loading…</div>';
  try {
    const children = await loadChildren(node);
    childBox.innerHTML = '';
    if (!children.length) { childBox.innerHTML = '<div class="proj-empty">' + emptyLabel(node) + '</div>'; return; }
    children.forEach((c) => childBox.appendChild(makeTreeNode(c)));
  } catch (e) {
    childBox.dataset.loaded = '';  // allow retry
    el.classList.remove('open');   // collapse the caret so the next click re-expands and retries in one click
    childBox.innerHTML = '<div class="proj-empty">Couldn’t load - ' + escapeHtml(String((e && e.message) || e)) + '</div>';
  }
}

function groupByModel(preds, models) {
  models = models || {};
  const groups = {};
  preds.forEach((p) => { const m = p.model_id || 'unknown'; (groups[m] = groups[m] || []).push(p); });
  return Object.keys(groups).map((mid) => {
    const info = models[mid] || {};
    const type = info.model_type || '';
    return {
      kind: 'model', id: mid,
      name: info.name || ('Model ' + shortId(mid)),
      modelType: type, badge: modelTypeLabel(type), foundation: info.foundation || '',
      meta: groups[mid].length, predictions: groups[mid],
    };
  });
}
function predNode(p) {
  return { kind: 'prediction', id: p.id, name: p.name || '(unnamed)', status: (p.status || '').toLowerCase() };
}
function resultNode(r) {
  const props = (r.property_names || []).join(', ');
  return {
    kind: 'result', id: r.id, name: props || ('result ' + shortId(r.id)),
    meta: r.file_format || '', leaf: true,
    result: { id: r.id, properties: props, format: r.file_format || '', tile_url: r.tile_url || '' },
  };
}

async function loadChildren(node) {
  if (!BRIDGE.live) return node.children || [];
  if (node.kind === 'project') {
    const data = await apiPredictions(node.id);
    return groupByModel(data.predictions || [], data.models || {});
  }
  if (node.kind === 'model') return node.predictions.map(predNode);
  if (node.kind === 'prediction') {
    return (await apiResults(node.id)).map(resultNode);
  }
  return [];
}

function emptyLabel(node) {
  if (node.kind === 'project') return 'No predictions in this project yet.';
  if (node.kind === 'model') return 'No predictions for this model.';
  if (node.kind === 'prediction') return 'No results for this prediction yet.';
  return 'Empty.';
}

function demoProjects() {
  return PROJECTS.map((p) => ({
    kind: 'project', id: p.id, name: p.name, icon: p.icon, meta: p.meta,
    children: [
      { kind: 'model', id: 'm-' + p.id, name: p.name + ' model',
        modelType: p.model || 'fine_tuned', badge: modelTypeLabel(p.model || 'fine_tuned'),
        foundation: 'OlmoEarth Nano v1', meta: 1, children: [
        { kind: 'prediction', id: 'pred-' + p.id, name: p.name + ' run', status: 'completed', children: [
          { kind: 'result', id: 'res-' + p.id, name: 'sample_score', meta: 'png', leaf: true },
        ] },
      ] },
    ],
  }));
}

function renderTree(container, nodes) {
  container.innerHTML = '';
  nodes.forEach((n) => container.appendChild(makeTreeNode(n)));
}

export async function renderProjects() {
  const list = document.getElementById('projList');
  if (!list) return;
  updateProjTag();
  if (!projConnected()) {
    list.innerHTML = '<div class="proj-empty">Connect your Studio key below to load your projects. <span class="proj-empty-sub">Nothing is fetched until you do.</span></div>';
    return;
  }
  if (!BRIDGE.live) { renderTree(list, demoProjects()); return; }  // sample tree
  list.innerHTML = '<div class="proj-empty">Loading your projects…</div>';
  try {
    const projects = (await apiProjects()).map((p) => ({
      kind: 'project', id: p.id || '', name: p.name || '(unnamed)', icon: pickProjIcon(p.name),
    }));
    if (!projects.length) {
      list.innerHTML = '<div class="proj-empty">No projects in your Studio account yet. <span class="proj-empty-sub">Create one in Studio, then refresh.</span></div>';
      return;
    }
    renderTree(list, projects);
  } catch (e) {
    list.innerHTML = '<div class="proj-empty">Couldn’t load projects - ' + escapeHtml(String((e && e.message) || e)) + '. <span class="proj-empty-sub">Check your key, or that the bridge can reach Studio.</span></div>';
  }
}
