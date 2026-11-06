# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Record the web UI walkthrough (MP4, animated WebP and cover) in demo mode.

Drives the static web UI in headless Chromium: connect a demo Studio key, open
the project tree, ask which windows to check first, ask how wrong the map is,
open and close the last answer's reasoning and tool call. Demo mode
replays what ``make verify-demo`` recorded (docs/VERIFICATION_DEMO.md), so no
key, LLM or network call to Studio is made. Writes olmoearth-agent-demo.mp4, its
animated WebP olmoearth-agent-demo.webp (which the README shows, as GitHub plays
no video stored in the repository) and the cover olmoearth-agent-demo.png next
to this file.

These dependencies are not part of the package:

    uv run --no-project --with playwright --with imageio-ffmpeg \
        python -m playwright install chromium
    uv run --no-project --with playwright --with imageio-ffmpeg \
        python webui/demo/record_demo.py
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
PORT = 8123
W, H = 1180, 760
START = 0.8  # skip the page load, so the video never opens on a blank frame

REVIEW_BRIEF = "Which windows of this land-cover map should I check first? I can check 5%."
ERROR_BRIEF = "How wrong is the map? I labelled the 300 random windows you planned."


def _serve() -> socketserver.TCPServer:
    """Serve webui/ statically on a background thread (demo mode)."""
    handler = functools.partial(
        http.server.SimpleHTTPRequestHandler, directory=str(WEBUI)
    )
    socketserver.TCPServer.allow_reuse_address = True
    httpd = socketserver.TCPServer(("127.0.0.1", PORT), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


def _ask(page, brief: str, answers: int) -> None:  # noqa: ANN001 - playwright Page
    """Type a brief, send it and wait until the thread holds ``answers`` answers."""
    page.fill("#promptInput", brief)
    time.sleep(0.6)
    page.click(".composer .send")
    page.wait_for_function(
        f"document.querySelectorAll('#chatThread .answer').length >= {answers}",
        timeout=20000,
    )


def _show_figure(page, n: int) -> None:  # noqa: ANN001 - playwright Page
    """Wait for the ``n``-th figure of the thread and scroll it into view."""
    page.wait_for_function(
        f"document.querySelectorAll('#chatThread .det-figure img').length > {n}",
        timeout=20000,
    )
    page.evaluate(
        f"() => document.querySelectorAll('#chatThread .det-figure')[{n}]"
        ".scrollIntoView({ behavior: 'smooth', block: 'start' })"
    )
    time.sleep(1.2)


def _record() -> Path:
    """Drive the walkthrough in Chromium and return the recorded .webm path."""
    from playwright.sync_api import sync_playwright

    REC.mkdir(exist_ok=True)
    with sync_playwright() as p:
        # Route the browser through a proxy if one is set (so Google Fonts loads),
        # but not the localhost web UI server.
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
        page.goto(f"http://127.0.0.1:{PORT}/index.html")
        page.wait_for_selector("#promptForm", timeout=15000)
        time.sleep(0.8)

        # 1) Connect a demo Studio key: the first step, before any question.
        page.click("#topKeyBtn")
        time.sleep(0.7)
        page.fill("#keyInput", "sk_pro_demo_key_2026")
        time.sleep(0.5)
        page.click("#keyForm button[type=submit]")
        time.sleep(1.0)
        page.keyboard.press("Escape")
        time.sleep(0.6)

        # 2) Open the project tree, now that a key is connected.
        row = page.query_selector("#projList .tree-row")
        if row:
            row.click()
            time.sleep(0.9)

        # 3) Without labels: which windows to check first, and its figure.
        _ask(page, REVIEW_BRIEF, 1)
        _show_figure(page, 0)
        time.sleep(4.0)  # hold on the figure and its caption

        # 4) With labels: how wrong the map is; its figure, then its table.
        _ask(page, ERROR_BRIEF, 2)
        page.wait_for_selector("#chatThread .answer .md-table", timeout=20000)
        _show_figure(page, 1)
        time.sleep(1.5)
        page.screenshot(path=str(DEMO / "olmoearth-agent-demo.png"))
        time.sleep(3.0)

        # 5) Open the last answer's reasoning and tool call, then close it again,
        #    so the video ends on the answer's table.
        heads = page.query_selector_all("#chatThread .steps-head")
        if heads:
            heads[-1].click()
            time.sleep(3.0)
            heads[-1].click()
            time.sleep(0.8)
        page.evaluate(
            "() => { const t = document.querySelector('#chatThread');"
            " t.scrollIntoView(false); window.scrollTo(0, document.body.scrollHeight);"
            " (t.closest('[class*=scroll]') || t.parentElement).scrollTop = 1e9; }"
        )
        time.sleep(3.0)

        ctx.close()  # finalizes the video file
        browser.close()

    vids = sorted(REC.glob("*.webm"), key=lambda f: f.stat().st_mtime)
    if not vids:
        raise RuntimeError("no video was recorded")
    return vids[-1]


def _convert(webm: Path) -> Path:
    """Transcode the .webm to a README-friendly MP4 with ffmpeg."""
    import imageio_ffmpeg

    mp4 = DEMO / "olmoearth-agent-demo.mp4"
    subprocess.run(  # noqa: S603 - the bundled ffmpeg binary and literal arguments
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-y",
            "-ss",
            str(START),
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
    return mp4


def _animate(mp4: Path) -> Path:
    """The README's animated WebP of the MP4 (GitHub plays no repository video)."""
    import imageio_ffmpeg

    webp = DEMO / "olmoearth-agent-demo.webp"
    subprocess.run(  # noqa: S603 - the bundled ffmpeg binary and literal arguments
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-y",
            "-i",
            str(mp4),
            "-vf",
            "fps=10,scale=960:-1:flags=lanczos",
            "-c:v",
            "libwebp_anim",
            "-lossless",
            "0",
            "-q:v",
            "60",
            "-loop",
            "0",
            str(webp),
        ],
        check=True,
    )
    return webp


def main() -> None:
    """Serve, record, transcode and remove the intermediate files."""
    httpd = _serve()
    try:
        webm = _record()
        mp4 = _convert(webm)
        webp = _animate(mp4)
        cover = DEMO / "olmoearth-agent-demo.png"
        print(
            f"wrote {mp4.name} ({mp4.stat().st_size // 1024} KB), "
            f"{webp.name} ({webp.stat().st_size // 1024} KB) "
            f"and {cover.name} ({cover.stat().st_size // 1024} KB)"
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
