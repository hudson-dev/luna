#!/usr/bin/env python3
"""Record the Three.js parafoil viewer to MP4 (+ optional GIF).

Requires the HTML to already exist (run_sim.py). Uses Playwright Chromium.
"""
from __future__ import annotations

import argparse
import http.server
import pathlib
import shutil
import socketserver
import sys
import tempfile
import threading
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _serve(directory: pathlib.Path, port: int) -> socketserver.TCPServer:
    directory = directory.resolve()

    class Quiet(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(directory), **kwargs)

        def log_message(self, fmt, *args):  # noqa: ANN001
            return

    class ReuseTCPServer(socketserver.TCPServer):
        allow_reuse_address = True

    httpd = ReuseTCPServer(("127.0.0.1", port), Quiet)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    return httpd


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--html", default="output/steady_wind_3d.html",
                    help="viewer HTML relative to parafoil_sim/")
    ap.add_argument("--out", default="output/steady_wind_flight.mp4")
    ap.add_argument("--gif", default="output/steady_wind_flight.gif")
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--sim-dt", type=float, default=0.333,
                    help="simulation seconds advanced per video frame (0.333 @ 12fps ≈ 4×)")
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--camera", default="follow", choices=("follow", "rig", "free"))
    ap.add_argument("--port", type=int, default=18931)
    ap.add_argument("--max-frames", type=int, default=0, help="0 = full flight")
    ap.add_argument("--gif-max-frames", type=int, default=120,
                    help="max frames in GIF; durations stretch to match MP4 length")
    args = ap.parse_args()

    html_path = (ROOT / args.html).resolve()
    if not html_path.is_file():
        print(f"missing {html_path}; run run_sim.py first", file=sys.stderr)
        return 1

    out_mp4 = (ROOT / args.out).resolve()
    out_gif = (ROOT / args.gif).resolve() if args.gif else None
    out_mp4.parent.mkdir(parents=True, exist_ok=True)

    from playwright.sync_api import sync_playwright
    import imageio.v2 as imageio

    httpd = _serve(html_path.parent, args.port)
    url = f"http://127.0.0.1:{args.port}/{html_path.name}"
    print(f"serving {url}")

    frames_dir = pathlib.Path(tempfile.mkdtemp(prefix="parafoil_frames_"))
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=True,
                args=[
                    "--use-angle=swiftshader",
                    "--enable-webgl",
                    "--ignore-gpu-blocklist",
                    "--disable-web-security",
                ],
            )
            page = browser.new_page(
                viewport={"width": args.width, "height": args.height},
                device_scale_factor=1,
            )
            page.goto(url, wait_until="domcontentloaded", timeout=120_000)
            page.wait_for_function(
                "() => window.__PARAFOIL_VIEWER && window.__PARAFOIL_VIEWER.ready",
                timeout=120_000,
            )
            # settle first frame + camera
            page.evaluate(
                """([cam]) => {
                  const v = window.__PARAFOIL_VIEWER;
                  v.pause();
                  v.setCamera(cam);
                  v.setFrame(0);
                  // warm camera lerp
                  for (let i = 0; i < 45; i++) v.renderOnce();
                }""",
                [args.camera],
            )
            time.sleep(0.3)

            duration = page.evaluate("() => window.__PARAFOIL_VIEWER.duration")
            n_est = int(duration / args.sim_dt) + 2
            print(f"flight {duration:.1f}s → ~{n_est} frames @ {args.fps} fps "
                  f"(sim_dt={args.sim_dt}s, camera={args.camera})")

            paths: list[pathlib.Path] = []
            done = False
            i = 0
            while not done:
                if args.max_frames and i >= args.max_frames:
                    break
                # settle a few renders so camera lerp catches up
                page.evaluate(
                    """() => {
                      const v = window.__PARAFOIL_VIEWER;
                      for (let k = 0; k < 8; k++) v.renderOnce();
                    }"""
                )
                shot = frames_dir / f"f_{i:05d}.png"
                page.screenshot(path=str(shot), type="png")
                paths.append(shot)
                if i % 25 == 0:
                    tnow = page.evaluate("() => window.__PARAFOIL_VIEWER.getTime()")
                    print(f"  frame {i:4d}  t={tnow:6.1f}s")
                done = page.evaluate(
                    "([dt]) => window.__PARAFOIL_VIEWER.stepSimTime(dt)",
                    [args.sim_dt],
                )
                i += 1

            browser.close()

        print(f"encoding {len(paths)} frames → {out_mp4}")
        # imageio-ffmpeg / system ffmpeg
        writer = imageio.get_writer(
            str(out_mp4),
            fps=args.fps,
            codec="libx264",
            quality=7,
            pixelformat="yuv420p",
            macro_block_size=None,
        )
        try:
            for pth in paths:
                writer.append_data(imageio.imread(pth))
        finally:
            writer.close()

        if out_gif:
            # Match MP4 wall-clock duration; subsample frames for size.
            print(f"building GIF → {out_gif}")
            from PIL import Image
            max_gif = max(2, args.gif_max_frames)
            step = max(1, len(paths) // max_gif)
            gif_paths = paths[::step]
            duration_ms = int(round(1000 * len(paths) / args.fps / len(gif_paths)))
            gif_frames = []
            for pth in gif_paths:
                im = Image.open(pth).convert("RGB")
                im = im.resize((640, 360), Image.Resampling.BILINEAR)
                gif_frames.append(im)
            gif_frames[0].save(
                out_gif,
                save_all=True,
                append_images=gif_frames[1:],
                duration=duration_ms,
                loop=0,
                optimize=True,
            )
            print(
                f"GIF frames={len(gif_frames)} duration_ms={duration_ms} "
                f"total={len(gif_frames)*duration_ms/1000:.1f}s "
                f"size={out_gif.stat().st_size / 1e6:.1f} MB"
            )

        print(f"MP4 size {out_mp4.stat().st_size / 1e6:.1f} MB")
        print("DONE")
        return 0
    finally:
        httpd.shutdown()
        shutil.rmtree(frames_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
