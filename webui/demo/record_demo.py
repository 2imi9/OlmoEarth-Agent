# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Record the OlmoEarth Agent webui demo (GIF + MP4) in DEMO mode.

Drives the static webui in a headless browser and records a short walkthrough:
connect a key, drill into the project tree, send a brief, watch the chat stream,
ask a follow-up, expand "Reasoning & tools", open settings. No LLM needed.

These deps are NOT part of the package — install once:

    uv pip install playwright imageio imageio-ffmpeg
    uv run --no-sync playwright install chromium
    uv run --no-sync python webui/demo/record_demo.py
"""

from __future__ import annotations

import functools
import http.server
import socketserver
import subprocess
import threading
import time
from pathlib import Path

WEBUI = Path(__file__).resolve().parents[1]
DEMO = WEBUI / "demo"
REC = DEMO / "_rec"
PORT = 8123
W, H = 1180, 760
GIF_SECONDS = 10  # the GIF is a short clip; the full walkthrough stays in the MP4
# Opaque (flicker-free) GIF frames are large, so the GIF runs at a lower fps/width
# than the MP4 to keep the file README-friendly.
GIF_FPS = 7
GIF_W = 620


def _serve() -> socketserver.TCPServer:
    """Serve webui/ statically on a background thread (demo mode)."""
    handler = functools.partial(
        http.server.SimpleHTTPRequestHandler, directory=str(WEBUI)
    )
    socketserver.TCPServer.allow_reuse_address = True
    httpd = socketserver.TCPServer(("127.0.0.1", PORT), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


def _record() -> Path:
    """Drive the walkthrough in Chromium and return the recorded .webm path."""
    from playwright.sync_api import sync_playwright

    REC.mkdir(exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        ctx = browser.new_context(
            viewport={"width": W, "height": H},
            record_video_dir=str(REC),
            record_video_size={"width": W, "height": H},
        )
        page = ctx.new_page()
        page.goto(f"http://127.0.0.1:{PORT}/index.html")
        page.wait_for_selector("#promptForm", timeout=15000)
        time.sleep(1.2)

        # Connect a (demo) key via the top-bar popover.
        page.click("#topKeyBtn")
        time.sleep(0.7)
        page.fill("#keyInput", "sk_pro_demo_key_2026")
        time.sleep(0.4)
        page.click("#keyForm button[type=submit]")
        time.sleep(0.9)
        page.keyboard.press("Escape")
        time.sleep(0.6)

        # Drill into the project tree.
        row = page.query_selector("#projList .tree-row")
        if row:
            row.click()
            time.sleep(1.0)
            sub = page.query_selector(
                "#projList .tree-node.open .tree-children .tree-row"
            )
            if sub:
                sub.click()
                time.sleep(1.0)

        # Send a brief and watch the chat stream.
        page.fill(
            "#promptInput",
            "How many OlmoEarth projects do I have, and which relate to water quality?",
        )
        time.sleep(0.4)
        page.click(".composer .send")
        page.wait_for_selector("#chatThread .answer", timeout=15000)
        time.sleep(1.4)

        # Follow-up (multi-turn).
        page.fill(
            "#promptInput",
            "Did karst-positive area trend up across the 4 quarterly snapshots?",
        )
        time.sleep(0.4)
        page.click(".composer .send")
        page.wait_for_function(
            "document.querySelectorAll('#chatThread .answer').length >= 2",
            timeout=15000,
        )
        time.sleep(1.2)

        # Expand the per-turn "Reasoning & tools" disclosure.
        heads = page.query_selector_all("#chatThread .steps-head")
        if heads:
            heads[-1].click()
            time.sleep(1.5)

        # Open the settings menu.
        page.click("#userChip")
        time.sleep(1.5)
        page.keyboard.press("Escape")
        time.sleep(0.6)

        ctx.close()  # finalizes the video file
        browser.close()

    vids = sorted(REC.glob("*.webm"), key=lambda f: f.stat().st_mtime)
    if not vids:
        raise RuntimeError("no video was recorded")
    return vids[-1]


def _convert(webm: Path) -> tuple[Path, Path]:
    """Transcode the .webm to a README-friendly MP4 + palette GIF via ffmpeg."""
    import imageio_ffmpeg

    ff = imageio_ffmpeg.get_ffmpeg_exe()
    mp4 = DEMO / "olmoearth-agent-demo.mp4"
    gif = DEMO / "olmoearth-agent-demo.gif"
    palette = DEMO / "_palette.png"
    subprocess.run(  # noqa: S603 - trusted ffmpeg binary + literal args
        [
            ff,
            "-y",
            "-i",
            str(webm),
            "-movflags",
            "+faststart",
            "-pix_fmt",
            "yuv420p",
            "-vf",
            f"scale={W}:-2",
            str(mp4),
        ],
        check=True,
    )
    # GIF — a SHORT clip (first GIF_SECONDS), smaller for the README.
    gif_t = str(GIF_SECONDS)
    subprocess.run(  # noqa: S603 - trusted ffmpeg binary + literal args
        [
            ff,
            "-y",
            "-t",
            gif_t,
            "-i",
            str(webm),
            "-vf",
            # reserve_transparent=0 → no transparent palette entry, so frames
            # are fully opaque (prevents the white flash between frames / on loop).
            f"fps={GIF_FPS},scale={GIF_W}:-1:flags=lanczos,palettegen=reserve_transparent=0",
            str(palette),
        ],
        check=True,
    )
    subprocess.run(  # noqa: S603 - trusted ffmpeg binary + literal args
        [
            ff,
            "-y",
            "-t",
            gif_t,
            "-i",
            str(webm),
            "-i",
            str(palette),
            "-lavfi",
            f"fps={GIF_FPS},scale={GIF_W}:-1:flags=lanczos[x];[x][1:v]paletteuse",
            # -transdiff off → write full frames (no transparent diff), so no
            # white show-through during the chat animation or at the loop point.
            "-gifflags",
            "-transdiff",
            str(gif),
        ],
        check=True,
    )
    palette.unlink(missing_ok=True)
    return mp4, gif


def main() -> None:
    """Serve, record, transcode, and tidy intermediate files."""
    httpd = _serve()
    try:
        webm = _record()
        mp4, gif = _convert(webm)
        print(
            f"wrote {gif.name} ({gif.stat().st_size // 1024} KB) "
            f"and {mp4.name} ({mp4.stat().st_size // 1024} KB)"
        )
    finally:
        httpd.shutdown()
    webm.unlink(missing_ok=True)
    try:
        REC.rmdir()
    except OSError:
        pass


if __name__ == "__main__":
    main()
