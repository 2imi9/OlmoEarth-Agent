# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Record the OlmoEarth Agent webui FEATURE SHOWCASE (GIF + MP4) in demo mode.

A longer "full function" walkthrough of the UI-polish pass, driven entirely
through the REAL UI code (no mocked CSS): the skill slash-command menu,
loading/busy states, the consecutive-workflow indicator, the in-conversation
result block, and the difference-map radar scan. The complex run + diff scan are
played through the app's real `handleRunEvent` / Leaflet renderers with
representative demo data.

Outputs:
  webui/demo/olmoearth-agent-showcase.mp4  - full walkthrough (~35-40s)
  webui/demo/olmoearth-agent-showcase.gif  - short highlight clip

Deps (install once):

    uv pip install playwright imageio-ffmpeg
    .venv/Scripts/playwright install chromium
    .venv/Scripts/python webui/demo/record_showcase.py
"""

from __future__ import annotations

import functools
import http.server
import os
import socketserver
import subprocess
import threading
import time
from pathlib import Path

WEBUI = Path(__file__).resolve().parents[1]
DEMO = WEBUI / "demo"
REC = DEMO / "_rec"
PORT = 8124
W, H = 1180, 820

# The GIF is a short highlight; the full walkthrough stays in the MP4. The window
# is set after recording from the logged segment offsets (see main()).
GIF_FPS = 8
GIF_W = 600

# ── The two showcase segments, played through the REAL UI code ──────────────
# A complex pipeline run: drives handleRunEvent with the canonical tool sequence,
# so the workflow rail advances + the result block + reasoning render for real.
JS_COMPLEX_RUN = r"""
async (pace) => {
  const { handleRunEvent } = await import('/js/run.js');
  const thread = document.getElementById('chatThread');
  document.getElementById('chatEmpty')?.setAttribute('hidden','');
  if (thread) { thread.hidden = false; thread.innerHTML = ''; }
  const card = document.createElement('div'); card.className = 'transcript live';
  card.innerHTML =
    '<div class="msg user run-step"><span class="role">You</span>' +
      '<div class="bubble">Run change detection over my Chesapeake AOI and compare the two model runs.</div></div>' +
    '<div class="msg agent"><span class="role">OlmoEarth Agent</span><div class="agent-body"></div></div>';
  thread.appendChild(card);
  const body = card.querySelector('.agent-body');
  card.scrollIntoView({ behavior:'smooth', block:'start' });
  const wait = (ms) => new Promise(r => setTimeout(r, ms));
  const tc = (name, args) => ({ type:'tool_call', name, arguments: args||{} });
  const tr = (name, result) => ({ type:'tool_result', name, ok:true, result:{ ok:true, result: result||{} } });
  // staticRender=true keeps the "Reasoning & tools" disclosure collapsed so the
  // workflow rail stays the visible focus + advances in place (no auto-scroll
  // to the streaming tool cards).
  handleRunEvent(body, { type:'thinking', text:'Four quarterly snapshots over the AOI, so a real trajectory diff is valid.' }, true);
  await wait(pace * 1.2);
  const seq = [
    [tc('olmoearth_load_context'), tr('olmoearth_load_context',{ name:'demo-user', project_count:5 })],
    [tc('olmoearth_request_aoi'), tr('olmoearth_request_aoi',{ area_id:'aoi_chesapeake', bbox:[-76.5,38.8,-76.0,39.2] })],
    [tc('olmoearth_load_skill',{ skill:'olmoearth_change_detect' }), tr('olmoearth_load_skill',{ loaded:'olmoearth_change_detect' })],
    [tc('olmoearth_submit_prediction'), tr('olmoearth_submit_prediction',{ prediction_id:'pred_8842' })],
    [tc('olmoearth_get_prediction'), tr('olmoearth_get_prediction',{ status:'completed' })],
    [tc('olmoearth_fetch_results'), tr('olmoearth_fetch_results',{ dates:['2024-03-01','2024-06-01','2024-09-01','2024-12-01'], values:[0.12,0.18,0.15,0.27], trend:'oscillating' })],
  ];
  for (const [call, res] of seq) {
    handleRunEvent(body, call, true); await wait(pace);
    handleRunEvent(body, res, true); await wait(pace * 0.55);
    card.scrollIntoView({ behavior:'smooth', block:'start' });  // keep the rail framed
  }
  handleRunEvent(body, { type:'final', content:'Net positive area rose **+0.15** over the year with **2 reversals** — oscillating with an upward bias, driven by the Sep→Dec quarter.' }, true);
  card.scrollIntoView({ behavior:'smooth', block:'start' });
  return true;
}
"""

# The difference-map radar scan: real Leaflet + the real .viz-scan / .viz-cell CSS,
# cells igniting out of order behind the sweep, with the determinate progress pill.
JS_DIFF_SCAN = r"""
async (cellMs) => {
  const m = await import('/js/leaflet.js');
  const L = await m.loadLeaflet();
  const thread = document.getElementById('chatThread');
  const card = document.createElement('div'); card.className = 'transcript live';
  card.innerHTML =
    '<div class="msg user run-step"><span class="role">You</span>' +
      '<div class="bubble">Now compare run A vs run B and scan the difference map.</div></div>' +
    '<div class="msg agent"><span class="role">OlmoEarth Agent</span><div class="agent-body"></div></div>';
  thread.appendChild(card);
  const body = card.querySelector('.agent-body');
  const viz = document.createElement('div'); viz.className = 'result-viz'; body.appendChild(viz);
  const ab = document.createElement('div'); ab.className = 'viz-ab';
  ab.innerHTML = '<span class="viz-ab-a">A</span> run 0a098ce7 <span class="viz-ab-d">vs</span> <span class="viz-ab-b">B</span> run e362ef10 <span class="viz-ab-d">· difference = B − A</span>';
  viz.appendChild(ab);
  const out = document.createElement('div'); out.className = 'viz-diff'; viz.appendChild(out);
  const el = document.createElement('div'); el.className = 'viz-map'; out.appendChild(el);
  const map = L.map(el, { attributionControl:false }).setView([39.0,-76.25], 10);
  el.style.background = '#0a3235';  // dark fallback (beats Leaflet's #ddd) so a failed/slow basemap reads on-theme, not gray
  m.osmLayer(L).addTo(map);
  const scan = document.createElement('div'); scan.className = 'viz-scan'; el.appendChild(scan);
  el.classList.add('is-scanning');
  setTimeout(() => map.invalidateSize(), 50);
  const prog = document.createElement('div'); prog.className = 'progress-pill';
  prog.innerHTML = '<span class="spinner sm"></span><span class="pp-label">Scanning</span><span class="pp-count">0/36</span><span class="pp-track"><span class="pp-fill"></span></span>';
  out.appendChild(prog);
  card.scrollIntoView({ behavior:'smooth', block:'center' });
  await new Promise(r => setTimeout(r, 1700));  // let the OSM basemap tiles paint before the cells
  const bb=[-76.5,38.8,-76.0,39.2], n=6, dx=(bb[2]-bb[0])/n, dy=(bb[3]-bb[1])/n;
  const order=[]; for (let i=0;i<n;i++) for (let j=0;j<n;j++) order.push([i,j]);
  order.sort(() => Math.random() - 0.5);  // out-of-order arrival, like the real async scan
  const total = order.length; let done = 0;
  const wait = (ms) => new Promise(r => setTimeout(r, ms));
  for (const [i,j] of order) {
    const aH = ((i*3 + j*7) % 2) === 0;
    const mag = 0.2 + ((i*n + j) % 8) / 8 * 0.6;
    const r = L.rectangle([[bb[1]+dy*j, bb[0]+dx*i],[bb[1]+dy*(j+1), bb[0]+dx*(i+1)]],
      { className:'viz-cell', color: aH?'#37a0ff':'#f0529c', fillColor: aH?'#37a0ff':'#f0529c', weight:0, fillOpacity: 0.15 + 0.7*mag }).addTo(map);
    if (r._path) r._path.classList.add('viz-cell--in');
    done++;
    prog.querySelector('.pp-count').textContent = done + '/' + total;
    prog.style.setProperty('--pp', (done/total).toFixed(2));
    await wait(cellMs);
  }
  prog.remove();
  const cap = document.createElement('div'); cap.className = 'viz-cap';
  cap.innerHTML = 'Difference map (B − A) over 36 sampled cells · mean |diff| <strong>0.12</strong> · corr <strong>0.94</strong> · 78% agree (±0.1). ' +
    '<span class="viz-legend"><span class="viz-sw" style="background:#37a0ff"></span>A higher<span class="viz-sw" style="background:#f0529c"></span>B higher</span>';
  out.appendChild(cap);
  const dl = document.createElement('div'); dl.className = 'dl-bar';
  dl.innerHTML = '<button class="dl-btn" type="button">↓ Download compare</button><button class="dl-btn" type="button">↓ Export SLD</button>';
  out.appendChild(dl);
  setTimeout(() => el.classList.remove('is-scanning'), 1100);  // let the sweep finish a pass
  return true;
}
"""


def _serve() -> socketserver.TCPServer:
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(WEBUI))
    socketserver.TCPServer.allow_reuse_address = True
    httpd = socketserver.TCPServer(("127.0.0.1", PORT), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


def _record() -> tuple[Path, dict]:
    """Drive the showcase in Chromium; return the .webm path + segment offsets."""
    from playwright.sync_api import sync_playwright

    REC.mkdir(exist_ok=True)
    marks: dict[str, float] = {}
    with sync_playwright() as p:
        launch_kwargs: dict = {}
        proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY")
        if proxy:
            launch_kwargs["proxy"] = {"server": proxy, "bypass": "127.0.0.1,localhost"}
        browser = p.chromium.launch(**launch_kwargs)
        ctx = browser.new_context(
            viewport={"width": W, "height": H},
            record_video_dir=str(REC),
            record_video_size={"width": W, "height": H},
        )
        page = ctx.new_page()
        t0 = time.monotonic()
        mark = lambda name: marks.setdefault(name, round(time.monotonic() - t0, 2))  # noqa: E731

        page.goto(f"http://127.0.0.1:{PORT}/index.html")
        page.wait_for_selector("#promptForm", timeout=15000)
        time.sleep(1.0)

        # 1) Connect a demo Studio key (the real first step).
        page.click("#topKeyBtn"); time.sleep(0.6)
        page.fill("#keyInput", "sk_pro_demo_key_2026"); time.sleep(0.4)
        page.click("#keyForm button[type=submit]"); time.sleep(0.8)
        page.keyboard.press("Escape"); time.sleep(0.5)

        # 2) Skill slash-command + loading + result block: type "/" to open the
        #    skill menu, filter it live, select a skill (Enter -> the composer
        #    fills "/<slug> "), then finish the brief and send. The busy send
        #    button + run-status bar show, then the routed scenario answers.
        mark("slash")
        page.click("#promptInput")
        page.keyboard.type("/", delay=0)  # "/" opens the skill menu
        page.wait_for_selector(".slash-menu:not([hidden]) .slash-item", timeout=4000)
        time.sleep(0.9)  # hold on the full skill menu
        page.keyboard.type("change", delay=90)  # filter live down to change-detection
        time.sleep(1.0)  # hold on the filtered match
        page.keyboard.press("Enter")  # select -> composer fills "/change-detection "
        time.sleep(0.6)
        page.keyboard.type(
            "Did forest cover decline across these four quarterly snapshots?", delay=18
        )
        time.sleep(0.5)
        mark("loading")
        page.click(".composer .send")
        # The routed change scenario answers in prose (no result-viz), so wait on
        # the answer card directly — avoids a dead 15s result-viz timeout.
        page.wait_for_selector("#chatThread .answer", timeout=15000)
        time.sleep(2.6)  # hold on the routed answer card
        mark("slash_end")

        # 3) Fresh thread, then the complex pipeline run (workflow rail + result).
        page.click(".btn-new"); time.sleep(0.8)
        mark("workflow")
        page.evaluate(JS_COMPLEX_RUN, 700)  # pace (ms) between events
        time.sleep(2.4)

        # 4) The difference-map radar scan (sweep + cell ignite + progress pill).
        mark("diffscan")
        page.evaluate(JS_DIFF_SCAN, 80)  # ms per cell
        time.sleep(3.0)
        mark("end")

        ctx.close()
        browser.close()

    vids = sorted(REC.glob("*.webm"), key=lambda f: f.stat().st_mtime)
    if not vids:
        raise RuntimeError("no video was recorded")
    return vids[-1], marks


def _convert(webm: Path, gif_start: float, gif_secs: float) -> tuple[Path, Path]:
    import imageio_ffmpeg

    ff = imageio_ffmpeg.get_ffmpeg_exe()
    mp4 = DEMO / "olmoearth-agent-showcase.mp4"
    gif = DEMO / "olmoearth-agent-showcase.gif"
    palette = DEMO / "_palette.png"
    # Full walkthrough MP4 (skip the brief blank page-load head).
    subprocess.run(  # noqa: S603
        [ff, "-y", "-ss", "0.6", "-i", str(webm), "-movflags", "+faststart",
         "-pix_fmt", "yuv420p", "-vf", f"scale={W}:-2", str(mp4)],
        check=True,
    )
    # Short highlight GIF over [gif_start, gif_start+gif_secs].
    common = f"fps={GIF_FPS},scale={GIF_W}:-1:flags=lanczos"
    subprocess.run(  # noqa: S603
        [ff, "-y", "-ss", str(gif_start), "-t", str(gif_secs), "-i", str(webm),
         "-vf", f"{common},palettegen=reserve_transparent=0", str(palette)],
        check=True,
    )
    subprocess.run(  # noqa: S603
        [ff, "-y", "-ss", str(gif_start), "-t", str(gif_secs), "-i", str(webm), "-i", str(palette),
         "-lavfi", f"{common}[x];[x][1:v]paletteuse", "-gifflags", "-transdiff", str(gif)],
        check=True,
    )
    palette.unlink(missing_ok=True)
    return mp4, gif


def main() -> None:
    httpd = _serve()
    try:
        webm, marks = _record()
        print("segment offsets (s):", marks)
        # GIF highlight = the skill slash-command flow ("/" -> filter -> select ->
        # type the brief -> send), the newest feature. End just after the routed
        # brief is sent; the streaming answer, workflow rail, and diff-scan radar
        # sweep stay in the full MP4 (and keep the GIF small — streaming text
        # frames are heavy).
        gif_start = max(0.6, marks.get("slash", 6.0) - 0.3)
        gif_end = marks.get("loading", gif_start + 6.0) + 2.2
        gif_secs = min(8.5, max(5.0, gif_end - gif_start))
        mp4, gif = _convert(webm, gif_start, gif_secs)
        print(
            f"wrote {gif.name} ({gif.stat().st_size // 1024} KB) "
            f"and {mp4.name} ({mp4.stat().st_size // 1024} KB); "
            f"gif window [{gif_start:.1f}s +{gif_secs:.1f}s]"
        )
        print(f"kept webm for re-conversion: {webm}")
    finally:
        httpd.shutdown()


if __name__ == "__main__":
    main()
