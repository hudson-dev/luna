"""Interactive Plotly visualization.

`save_flight_html(result, path)` writes a single self-contained HTML file with:

* an animated 3D flight scene (play button + time slider): trajectory colored
  by guidance phase, an attitude-true parafoil rig glyph (multi-cell canopy
  with deflecting trailing edge, suspension lines, brake lines, payload +
  servo spools; drawn oversized for visibility), the MPC's predicted horizon,
  the wind vector at the vehicle, a wind-profile mast over the pad, and
  pad/release/FAP markers;
* a true-scale close-up "rig view" animation: the same rig at 1:1 scale with
  attitude only (vehicle pinned at the origin), showing the trailing edge,
  brake-line pull, and servo spool arms working through the flight;
* a time-series dashboard: altitude/phase, speeds, brake deflections + line
  travel per side [cm], MPC cost, wind estimate vs truth, distance to pad,
  heading.

Plot axes are ENU (x East, y North, z Up); the simulator's NED states are
converted here. Port (left) brake line is drawn red, starboard (right) green.
"""
from __future__ import annotations

import pathlib

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..dynamics.quat import quat_from_euler, quat_to_dcm
from .rig_geometry import build_rig, transform_parts, _polyline_concat

PHASE_COLORS = {"HOMING": "#1f77b4", "LOITER": "#ff7f0e", "EXTEND": "#9467bd",
                "APPROACH": "#2ca02c", "FLARE": "#d62728"}
GLYPH_SCALE = 14.0     # rig drawn oversized in the flight view
CANOPY_COLOR = "#e8641e"
PAYLOAD_COLOR = "#37474f"
LINE_COLOR = "rgba(70,70,70,0.75)"
BRAKE_L_COLOR = "#d62728"   # port red
BRAKE_R_COLOR = "#2ca02c"   # starboard green


# ---------------------------------------------------------------- helpers
def _ned_to_enu(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p)
    return np.stack([p[..., 1], p[..., 0], -p[..., 2]], axis=-1)


def _rig_traces(res, i: int, scale: float, at_origin: bool,
                show_legend: bool):
    """The six rig traces for log index i (order is fixed: canopy, payload,
    rig lines, brake L, brake R). Coordinates in ENU."""
    veh = res.scenario.vehicle
    lt = veh.line_travel_max
    bl = float(res.line_l[i] / lt)
    br = float(res.line_r[i] / lt)
    sl = float(res.line_l[i] / veh.spool_radius)
    sr = float(res.line_r[i] / veh.spool_radius)

    parts = build_rig(veh, bl, br, sl, sr)
    R = quat_to_dcm(quat_from_euler(*res.euler[i]))
    pos = np.zeros(3) if at_origin else res.pos[i]
    parts = transform_parts(parts, R, pos, scale)

    def enu_xyz(segs):
        xs, ys, zs = _polyline_concat([_ned_to_enu(s) for s in segs])
        return xs, ys, zs

    cv, cf = parts["canopy"]
    cv = _ned_to_enu(cv)
    pv, pf = parts["payload"]
    pv = _ned_to_enu(pv)
    lx, ly, lz = enu_xyz(parts["ribs"] + parts["susp"])
    blx, bly, blz = enu_xyz(parts["brake_l"])
    brx, bry, brz = enu_xyz(parts["brake_r"])

    return [
        go.Mesh3d(x=cv[:, 0], y=cv[:, 1], z=cv[:, 2],
                  i=cf[:, 0], j=cf[:, 1], k=cf[:, 2], color=CANOPY_COLOR,
                  opacity=0.92, flatshading=True, name="canopy",
                  showlegend=False),
        go.Mesh3d(x=pv[:, 0], y=pv[:, 1], z=pv[:, 2],
                  i=pf[:, 0], j=pf[:, 1], k=pf[:, 2], color=PAYLOAD_COLOR,
                  opacity=1.0, flatshading=True, name="payload + servos",
                  showlegend=False),
        go.Scatter3d(x=lx, y=ly, z=lz, mode="lines",
                     line=dict(color=LINE_COLOR, width=2),
                     name="ribs + suspension", showlegend=show_legend),
        go.Scatter3d(x=blx, y=bly, z=blz, mode="lines",
                     line=dict(color=BRAKE_L_COLOR, width=5),
                     name="left brake line (servo)", showlegend=show_legend),
        go.Scatter3d(x=brx, y=bry, z=brz, mode="lines",
                     line=dict(color=BRAKE_R_COLOR, width=5),
                     name="right brake line (servo)", showlegend=show_legend),
    ]


def _play_controls(frames, y=0.02):
    steps = [dict(method="animate",
                  args=[[f.name], dict(mode="immediate",
                                       frame=dict(duration=0, redraw=True),
                                       transition=dict(duration=0))],
                  label=f"{float(f.name):.0f}s") for f in frames]
    menus = [dict(type="buttons", direction="left", x=0.05, y=y,
                  buttons=[
                      dict(label="&#9654; Play", method="animate",
                           args=[None, dict(frame=dict(duration=55, redraw=True),
                                            fromcurrent=True,
                                            transition=dict(duration=0))]),
                      dict(label="&#9646;&#9646; Pause", method="animate",
                           args=[[None], dict(mode="immediate",
                                              frame=dict(duration=0, redraw=True),
                                              transition=dict(duration=0))]),
                  ])]
    sliders = [dict(steps=steps, x=0.15, len=0.8, y=y,
                    currentvalue=dict(prefix="t = ", visible=True))]
    return menus, sliders


# ------------------------------------------------------------- 3D figure
def _flight_figure_3d(res) -> go.Figure:
    scn = res.scenario
    pos = res.pos
    enu = _ned_to_enu(pos)
    phase = np.array(res.phase)
    t = res.t

    fig = go.Figure()

    # trajectory per phase (static)
    for ph, col in PHASE_COLORS.items():
        m = phase == ph
        if not m.any():
            continue
        fig.add_trace(go.Scatter3d(
            x=enu[m, 0], y=enu[m, 1], z=enu[m, 2], mode="lines",
            line=dict(color=col, width=5), name=f"phase {ph}"))

    # ground shadow
    fig.add_trace(go.Scatter3d(
        x=enu[:, 0], y=enu[:, 1], z=np.zeros(len(enu)), mode="lines",
        line=dict(color="rgba(120,120,120,0.5)", width=2), name="ground track"))

    # pad, release, FAP
    tgt = np.array(scn.target)
    fig.add_trace(go.Scatter3d(x=[tgt[1]], y=[tgt[0]], z=[0], mode="markers+text",
                               marker=dict(size=9, color="red", symbol="diamond"),
                               text=["PAD"], textposition="top center", name="launch pad"))
    fig.add_trace(go.Scatter3d(x=[enu[0, 0]], y=[enu[0, 1]], z=[enu[0, 2]],
                               mode="markers+text", marker=dict(size=6, color="black"),
                               text=["release"], textposition="top center", name="release"))
    if res.fap:
        fap = np.array(res.fap[-1])
        fig.add_trace(go.Scatter3d(x=[fap[1]], y=[fap[0]], z=[0], mode="markers+text",
                                   marker=dict(size=5, color="#ff7f0e", symbol="x"),
                                   text=["FAP"], textposition="bottom center",
                                   name="final approach point"))

    # wind profile mast above the pad (true mean wind vs altitude)
    from ..environment.wind import WindField
    rng = np.random.default_rng(0)
    wf = WindField(scn.wind, rng)
    hs = np.linspace(20, max(-pos[:, 2].max(), 100), 8)
    wvec = np.array([wf.mean_wind(h) for h in hs])
    fig.add_trace(go.Scatter3d(x=tgt[1] * np.ones(len(hs)), y=tgt[0] * np.ones(len(hs)),
                               z=hs, mode="lines", line=dict(color="gray", dash="dash"),
                               name="wind mast", showlegend=False))
    if np.linalg.norm(wvec[:, :2]) > 0.1:
        fig.add_trace(go.Cone(
            x=tgt[1] + wvec[:, 1] * 0, y=tgt[0] + wvec[:, 0] * 0, z=hs,
            u=wvec[:, 1], v=wvec[:, 0], w=-wvec[:, 2],
            sizemode="absolute", sizeref=6, anchor="tail",
            colorscale=[[0, "#4a90d9"], [1, "#08306b"]], showscale=False,
            name="mean wind", hovertemplate="h=%{z:.0f} m<br>wind=(%{v:.1f} N, %{u:.1f} E) m/s"))

    # ---- animated traces: rig glyph + MPC plan + wind at vehicle ----------
    n = len(t)
    n_frames = min(130, n)
    idxs = np.linspace(0, n - 1, n_frames).astype(int)
    t_mpc = np.array(res.t_mpc)

    def frame_traces(i, show_legend=False):
        traces = _rig_traces(res, i, GLYPH_SCALE, at_origin=False,
                             show_legend=show_legend)
        j = int(np.searchsorted(t_mpc, t[i], side="right")) - 1
        j = max(0, min(j, len(res.mpc_pred) - 1))
        xp = res.mpc_pred[j]
        traces.append(go.Scatter3d(
            x=xp[:, 1], y=xp[:, 0], z=np.maximum(xp[:, 2], 0),
            mode="lines+markers", line=dict(color="magenta", width=4, dash="dot"),
            marker=dict(size=2), name="MPC plan", showlegend=show_legend))
        w = res.wind_true[i]
        traces.append(go.Scatter3d(
            x=[enu[i, 0], enu[i, 0] + w[1] * 12], y=[enu[i, 1], enu[i, 1] + w[0] * 12],
            z=[enu[i, 2], enu[i, 2] - w[2] * 12], mode="lines",
            line=dict(color="#08306b", width=6), name="wind @ vehicle",
            showlegend=show_legend))
        return traces

    base = frame_traces(idxs[0], show_legend=True)
    n_static = len(fig.data)
    for tr in base:
        fig.add_trace(tr)
    dyn_ids = list(range(n_static, n_static + len(base)))

    fig.frames = [go.Frame(data=frame_traces(i), traces=dyn_ids,
                           name=f"{t[i]:.1f}") for i in idxs]
    menus, sliders = _play_controls(fig.frames)

    fig.update_layout(
        updatemenus=menus, sliders=sliders,
        scene=dict(
            xaxis_title="East [m]", yaxis_title="North [m]", zaxis_title="Altitude [m]",
            aspectmode="data",
            camera=dict(eye=dict(x=1.4, y=-1.6, z=0.8)),
        ),
        legend=dict(x=0.82, y=0.95),
        margin=dict(l=0, r=0, t=45, b=0),
        title=(f"{scn.name}: 3D flight — miss {res.miss_distance:.1f} m, "
               f"touchdown {res.td_ground_speed:.1f} m/s ground / "
               f"{res.td_sink_rate:.1f} m/s sink (rig glyph x{GLYPH_SCALE:.0f})"),
        height=760,
    )
    return fig


# ------------------------------------------------------ close-up rig view
def _rig_view_figure(res) -> go.Figure:
    """True-scale animation of the rig alone (attitude only, CG at origin):
    canopy trailing edge, brake-line pull, and servo spool arms."""
    t = res.t
    n = len(t)
    n_frames = min(110, n)
    idxs = np.linspace(0, n - 1, n_frames).astype(int)

    fig = go.Figure()

    def frame_traces(i, show_legend=False):
        traces = _rig_traces(res, i, 1.0, at_origin=True,
                             show_legend=show_legend)
        w = res.wind_true[i]
        wn = np.linalg.norm(w) + 1e-6
        wdir = w / wn * min(wn, 8.0) * 0.28
        traces.append(go.Scatter3d(
            x=[-2.2, -2.2 + wdir[1]], y=[2.2, 2.2 + wdir[0]],
            z=[1.8, 1.8 - wdir[2]], mode="lines+text",
            line=dict(color="#08306b", width=7),
            text=[f"wind {wn:.1f} m/s", ""], textposition="top center",
            name="wind", showlegend=show_legend))
        return traces

    base = frame_traces(idxs[0], show_legend=True)
    for tr in base:
        fig.add_trace(tr)
    dyn_ids = list(range(len(base)))

    fig.frames = [go.Frame(data=frame_traces(i), traces=dyn_ids,
                           name=f"{t[i]:.1f}") for i in idxs]
    menus, sliders = _play_controls(fig.frames)

    r = 2.3
    fig.update_layout(
        updatemenus=menus, sliders=sliders,
        scene=dict(
            xaxis=dict(range=[-r, r], title="East [m]"),
            yaxis=dict(range=[-r, r], title="North [m]"),
            zaxis=dict(range=[-1.0, 2.2], title="Up [m]"),
            aspectmode="cube",
            camera=dict(eye=dict(x=0.95, y=1.15, z=0.30)),
        ),
        legend=dict(x=0.82, y=0.95),
        margin=dict(l=0, r=0, t=45, b=0),
        title=(f"{res.scenario.name}: rig close-up (true scale) — trailing edge, "
               "brake lines (port red / starboard green), servo spools"),
        height=640,
    )
    return fig


# ------------------------------------------------------- time-series figure
def _timeseries_figure(res) -> go.Figure:
    scn = res.scenario
    t = res.t
    pos = res.pos
    h = -pos[:, 2]
    gs = np.hypot(res.vel_ned[:, 0], res.vel_ned[:, 1])
    d_pad = np.hypot(pos[:, 0] - scn.target[0], pos[:, 1] - scn.target[1])
    phase = np.array(res.phase)

    fig = make_subplots(
        rows=4, cols=2, shared_xaxes=True, vertical_spacing=0.06,
        specs=[[{}, {}], [{"secondary_y": True}, {}], [{}, {}], [{}, {}]],
        subplot_titles=("Altitude & guidance phase", "Airspeed / ground speed",
                        "Brake deflection & servo line travel", "MPC cost per solve",
                        "Wind North: true vs estimate", "Wind East: true vs estimate",
                        "Distance to pad", "Heading (yaw)"))

    for ph, col in PHASE_COLORS.items():
        m = phase == ph
        if m.any():
            fig.add_trace(go.Scatter(x=t[m], y=h[m], mode="markers",
                                     marker=dict(size=2, color=col), name=ph), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=res.airspeed, name="airspeed",
                             line=dict(color="#1f77b4")), row=1, col=2)
    fig.add_trace(go.Scatter(x=t, y=gs, name="ground speed",
                             line=dict(color="#ff7f0e")), row=1, col=2)

    fig.add_trace(go.Scatter(x=t, y=res.da, name="da (asym, actual)",
                             line=dict(color="#7f7f7f")), row=2, col=1)
    fig.add_trace(go.Scatter(x=t, y=res.da_cmd, name="da command",
                             line=dict(color="#7f7f7f", dash="dot", width=1)), row=2, col=1)
    fig.add_trace(go.Scatter(x=t, y=res.line_l * 100, name="left line travel [cm]",
                             line=dict(color=BRAKE_L_COLOR)), row=2, col=1,
                  secondary_y=True)
    fig.add_trace(go.Scatter(x=t, y=res.line_r * 100, name="right line travel [cm]",
                             line=dict(color=BRAKE_R_COLOR)), row=2, col=1,
                  secondary_y=True)
    cost = np.array(res.mpc_cost, dtype=float)
    cost = np.where(np.isfinite(cost), cost, np.nan)
    fig.add_trace(go.Scatter(x=res.t_mpc, y=cost, name="MPC objective",
                             line=dict(color="purple")), row=2, col=2)

    for r, c, k, lab in ((3, 1, 0, "North"), (3, 2, 1, "East")):
        fig.add_trace(go.Scatter(x=t, y=res.wind_true[:, k], name=f"true w{lab}",
                                 line=dict(color="#1f77b4")), row=r, col=c)
        fig.add_trace(go.Scatter(x=t, y=res.wind_hat[:, k], name=f"est w{lab}",
                                 line=dict(color="#d62728", dash="dash")), row=r, col=c)

    fig.add_trace(go.Scatter(x=t, y=d_pad, name="dist to pad",
                             line=dict(color="black")), row=4, col=1)
    fig.add_trace(go.Scatter(x=t, y=np.rad2deg(res.euler[:, 2]), name="yaw",
                             line=dict(color="#8c564b")), row=4, col=2)

    fig.update_xaxes(title_text="t [s]", row=4, col=1)
    fig.update_xaxes(title_text="t [s]", row=4, col=2)
    fig.update_yaxes(title_text="m", row=1, col=1)
    fig.update_yaxes(title_text="m/s", row=1, col=2)
    fig.update_yaxes(title_text="norm.", row=2, col=1, secondary_y=False)
    fig.update_yaxes(title_text="cm", row=2, col=1, secondary_y=True)
    fig.update_yaxes(title_text="m/s", row=3, col=1)
    fig.update_yaxes(title_text="m/s", row=3, col=2)
    fig.update_yaxes(title_text="m", row=4, col=1)
    fig.update_yaxes(title_text="deg", row=4, col=2)
    fig.update_layout(height=980, title=f"{scn.name}: time series",
                      legend=dict(orientation="h", y=-0.05), margin=dict(t=60))
    return fig


# ----------------------------------------------------------------- entry
def save_flight_html(res, path) -> str:
    path = pathlib.Path(path)
    fig3d = _flight_figure_3d(res)
    figrig = _rig_view_figure(res)
    figts = _timeseries_figure(res)
    scn = res.scenario
    header = f"""
    <div style="font-family: -apple-system, Helvetica, sans-serif; margin: 14px 24px;">
      <h2 style="margin-bottom:2px;">Guided parafoil flight — scenario: {scn.name}</h2>
      <p style="color:#444; margin-top:2px;">{scn.description}</p>
      <p style="color:#111;"><b>Landing:</b> miss {res.miss_distance:.1f} m &nbsp;|&nbsp;
         ground speed {res.td_ground_speed:.2f} m/s &nbsp;|&nbsp;
         sink {res.td_sink_rate:.2f} m/s &nbsp;|&nbsp;
         flare {'triggered' if res.flare_triggered else 'NOT triggered'} &nbsp;|&nbsp;
         flight time {res.t_flight:.1f} s</p>
    </div>"""
    html = ("<html><head><meta charset='utf-8'><title>parafoil " + scn.name +
            "</title></head><body>" + header +
            fig3d.to_html(full_html=False, include_plotlyjs="inline") +
            figrig.to_html(full_html=False, include_plotlyjs=False) +
            figts.to_html(full_html=False, include_plotlyjs=False) +
            "</body></html>")
    path.write_text(html)
    return str(path)
