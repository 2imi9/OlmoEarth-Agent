/* Client-side file downloads for chat artifacts (compare results, SLD styles,
   result JSON). Builds a Blob and triggers a save - no server round-trip. */

export function downloadText(filename, text, mime) {
  const blob = new Blob([text], { type: (mime || 'text/plain') + ';charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export function downloadJson(filename, obj) {
  downloadText(filename, JSON.stringify(obj, null, 2), 'application/json');
}

/* A comparison record's grid as CSV (lon, lat, diff = B - A). */
export function comparisonCsv(record) {
  const head = 'lon,lat,diff_b_minus_a\n';
  const rows = (record.cells || []).map((c) => `${c.lon},${c.lat},${c.diff}`).join('\n');
  return head + rows + '\n';
}

/* A small "Download" control group; `items` = [{label, onClick}]. */
export function downloadBar(items) {
  const bar = document.createElement('div');
  bar.className = 'dl-bar';
  items.forEach(({ label, onClick }) => {
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'dl-btn';
    b.textContent = label;
    b.addEventListener('click', onClick);
    bar.appendChild(b);
  });
  return bar;
}
