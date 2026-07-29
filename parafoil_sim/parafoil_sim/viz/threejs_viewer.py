"""Generate a self-contained Three.js WebGL flight viewer HTML.

Embeds the flight log JSON and `viewer.js`. Three.js is loaded from a CDN
via ES modules (works when opening the HTML from disk in a modern browser).
"""
from __future__ import annotations

import json
import pathlib

from .flight_log import export_flight_log

_HERE = pathlib.Path(__file__).resolve().parent
_VIEWER_JS = _HERE / "viewer.js"

_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Parafoil 3D — __NAME__</title>
<style>
  :root {
    --bg: rgba(18, 24, 32, 0.82);
    --fg: #f2f4f7;
    --muted: #a8b0bc;
    --accent: #e8641e;
    --border: rgba(255,255,255,0.12);
  }
  * { box-sizing: border-box; }
  html, body { margin: 0; height: 100%; overflow: hidden;
    font-family: "IBM Plex Sans", "Segoe UI", system-ui, sans-serif;
    background: #0b1016; color: var(--fg); }
  #c { display: block; width: 100%; height: 100%; }
  #hud {
    position: absolute; inset: 0; pointer-events: none;
    display: flex; flex-direction: column; justify-content: space-between;
  }
  .panel {
    pointer-events: auto;
    margin: 14px;
    padding: 12px 16px;
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 12px;
    backdrop-filter: blur(10px);
    box-shadow: 0 8px 28px rgba(0,0,0,0.28);
  }
  #top { max-width: 520px; }
  #top h1 { margin: 0 0 4px; font-size: 1.15rem; font-weight: 650;
    letter-spacing: -0.01em; }
  #top p { margin: 0; color: var(--muted); font-size: 0.82rem; line-height: 1.35; }
  #meta { margin-top: 8px; font-size: 0.78rem; color: var(--muted); }
  #meta span { color: var(--fg); font-weight: 600; }
  #bottom {
    display: flex; flex-wrap: wrap; gap: 10px; align-items: center;
  }
  button, select {
    appearance: none; border: 1px solid var(--border); background: rgba(255,255,255,0.06);
    color: var(--fg); border-radius: 8px; padding: 7px 12px; font: inherit;
    font-size: 0.85rem; cursor: pointer;
  }
  button:hover, select:hover { background: rgba(255,255,255,0.12); }
  button.active { background: var(--accent); border-color: transparent; color: #fff; }
  #scrub { flex: 1 1 220px; min-width: 160px; accent-color: var(--accent); }
  .row { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; width: 100%; }
  .tog { display: flex; gap: 12px; align-items: center; font-size: 0.78rem; color: var(--muted); }
  .tog label { display: flex; gap: 5px; align-items: center; cursor: pointer; }
  #phase-label { font-weight: 700; letter-spacing: 0.04em; }
  #brake-label { font-size: 0.78rem; color: var(--muted); }
  .legend {
    position: absolute; top: 14px; right: 14px; pointer-events: none;
    font-size: 0.72rem; color: var(--muted);
  }
  .legend i { display: inline-block; width: 10px; height: 10px; border-radius: 2px;
    margin-right: 5px; vertical-align: -1px; }
  #boot {
    position: absolute; inset: 0; display: grid; place-items: center;
    background: #0b1016; color: var(--muted); font-size: 0.95rem; z-index: 5;
  }
</style>
</head>
<body>
<div id="boot">Loading Three.js viewer…</div>
<canvas id="c"></canvas>
<div id="hud">
  <div>
    <div class="panel" id="top">
      <h1 id="title">Parafoil 3D</h1>
      <p id="subtitle"></p>
      <div id="meta">
        <span id="time-label">t = 0.0 s</span> ·
        <span id="phase-label">—</span><br/>
        <span id="brake-label"></span><br/>
        <span id="miss-label"></span>
      </div>
    </div>
    <div class="legend panel">
      <div><i style="background:#3b82f6"></i>HOMING</div>
      <div><i style="background:#f59e0b"></i>LOITER</div>
      <div><i style="background:#a855f7"></i>EXTEND</div>
      <div><i style="background:#22c55e"></i>APPROACH</div>
      <div><i style="background:#ef4444"></i>FLARE</div>
      <div style="margin-top:6px"><i style="background:#d62728"></i>L brake ·
           <i style="background:#2ca02c"></i>R brake</div>
    </div>
  </div>
  <div class="panel" id="bottom">
    <div class="row">
      <button id="btn-play">Pause</button>
      <input id="scrub" type="range" min="0" max="100" value="0"/>
      <select id="speed" title="Playback speed">
        <option value="0.25">0.25×</option>
        <option value="0.5">0.5×</option>
        <option value="1" selected>1×</option>
        <option value="2">2×</option>
        <option value="4">4×</option>
        <option value="8">8×</option>
      </select>
      <button data-cam="free">Free orbit</button>
      <button data-cam="follow" class="active">Follow</button>
      <button data-cam="rig">Rig close-up</button>
      <div class="tog">
        <label><input type="checkbox" id="tog-traj" checked/> trajectory</label>
        <label><input type="checkbox" id="tog-wind" checked/> wind</label>
      </div>
    </div>
  </div>
</div>
<script type="importmap">
{
  "imports": {
    "three": "https://unpkg.com/three@0.160.0/build/three.module.js",
    "three/addons/": "https://unpkg.com/three@0.160.0/examples/jsm/"
  }
}
</script>
<script>
window.FLIGHT_LOG = __LOG_JSON__;
</script>
<script type="module">
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
window.THREE = THREE;
window.OrbitControls = OrbitControls;
const boot = document.getElementById("boot");
if (boot) boot.remove();
__VIEWER_JS__
</script>
</body>
</html>
"""


def save_threejs_html(res, path, *, dt_out: float = 0.1,
                      also_json: bool = True) -> str:
    """Write interactive Three.js viewer HTML (and optional sibling JSON)."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.stem.endswith("_3d"):
        json_path = path.with_name(path.stem[:-3] + "_flight.json")
    else:
        json_path = path.with_name(path.stem + "_flight.json")

    export_flight_log(res, json_path, dt_out=dt_out)
    log_obj = json.loads(json_path.read_text())
    if not also_json:
        json_path.unlink(missing_ok=True)

    return write_viewer_html(log_obj, path, title_name=res.scenario.name)


def write_viewer_html(log_obj: dict, path, *, title_name: str | None = None) -> str:
    """Embed an existing flight-log dict into the viewer HTML shell."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    name = title_name or log_obj.get("scenario", {}).get("name", "flight")
    viewer_js = _VIEWER_JS.read_text()
    html = (
        _HTML
        .replace("__NAME__", name)
        .replace("__LOG_JSON__", json.dumps(log_obj, separators=(",", ":")))
        .replace("__VIEWER_JS__", viewer_js)
    )
    path.write_text(html)
    return str(path)
