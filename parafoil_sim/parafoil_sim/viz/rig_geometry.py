"""Parametric 3D geometry of the parafoil-payload rig for visualization.

Everything is built in the body frame (FRD: x forward, y right, z down,
origin at the system CG) in true meters, from the same `VehicleParams` used
by the dynamics, then rotated/translated/scaled by the caller:

* multi-cell ram-air canopy on an anhedral arc, with chordwise camber,
  visible cell ribs, and a trailing edge whose left/right sections deflect
  downward proportionally to the actual brake-line pull on that side;
* simplified suspension line set: A and C line rows cascading from the
  canopy to per-side confluence points, then risers to the payload;
* left/right brake (steering) lines from the deflected trailing edge through
  a guide at the confluence down to the winch servo spools on the payload --
  the only lines the servos act on;
* payload box with two servo glyphs whose spool arms rotate with the actual
  spool angle.

Left/right line colors follow the nautical/aviation convention:
port (left) red, starboard (right) green.
"""
from __future__ import annotations

import numpy as np

from ..config import VehicleParams

TE_DEFLECT_MAX = np.deg2rad(40.0)   # trailing-edge angle at full brake
HINGE_FRACTION = 0.30               # last 30% of chord deflects

N_RIB = 9                           # spanwise stations (8 cells)
N_CHORD = 7                         # chordwise stations


# --------------------------------------------------------------------------
def _canopy_grid(veh: VehicleParams, brake_l: float, brake_r: float):
    """Vertex grid (N_RIB, N_CHORD, 3) of the canopy surface, body frame."""
    b, c = veh.span, veh.chord
    arc = veh.arc_ratio * b                      # arc drop of the tips
    R = ((b / 2) ** 2 + arc ** 2) / (2 * arc)    # arc radius through tips
    th_m = np.arcsin((b / 2) / R)
    z_c = veh.r_canopy[2]                        # canopy center height (< 0)

    thetas = np.linspace(-th_m, th_m, N_RIB)
    s = np.linspace(0.0, 1.0, N_CHORD)           # 0 = LE, 1 = TE
    x_chord = c / 2 - s * c                      # LE at +c/2, TE at -c/2
    camber = -0.08 * c * np.sin(np.pi * s)       # upward bulge (z down +)

    hinge_s = 1.0 - HINGE_FRACTION
    x_hinge = c / 2 - hinge_s * c

    grid = np.zeros((N_RIB, N_CHORD, 3))
    for i, th in enumerate(thetas):
        y = R * np.sin(th)
        z_rib = z_c + R * (1 - np.cos(th))       # tips hang lower (+z)
        y_hat = y / (b / 2)
        norm = brake_l * max(0.0, -y_hat) + brake_r * max(0.0, y_hat)
        delta = TE_DEFLECT_MAX * norm * (0.25 + 0.75 * abs(y_hat))
        cd, sd = np.cos(delta), np.sin(delta)
        for j in range(N_CHORD):
            x, dz = x_chord[j], camber[j]
            if x < x_hinge:                       # aft of hinge: rotate down
                dx = x - x_hinge
                dzh = dz - camber[int(hinge_s * (N_CHORD - 1))]
                x = x_hinge + dx * cd + dzh * sd
                dz = camber[int(hinge_s * (N_CHORD - 1))] - dx * sd + dzh * cd
            grid[i, j] = (x, y, z_rib + dz)
    return grid


def _grid_mesh(grid: np.ndarray):
    """Triangulate the (N_RIB, N_CHORD) grid into a Mesh3d vert/face set."""
    nr, nc, _ = grid.shape
    verts = grid.reshape(-1, 3)
    faces = []
    for i in range(nr - 1):
        for j in range(nc - 1):
            a = i * nc + j
            faces += [[a, a + nc, a + 1], [a + 1, a + nc, a + nc + 1]]
    return verts, np.array(faces)


def _box(center, dims):
    cx, cy, cz = center
    dx, dy, dz = np.asarray(dims) / 2
    v = np.array([[sx, sy, sz] for sx in (-dx, dx) for sy in (-dy, dy)
                  for sz in (-dz, dz)]) + np.array([cx, cy, cz])
    f = np.array([[0, 1, 3], [0, 3, 2], [4, 7, 5], [4, 6, 7],
                  [0, 5, 1], [0, 4, 5], [2, 3, 7], [2, 7, 6],
                  [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
    return v, f


def _polyline_concat(segments):
    """Concatenate polylines with None separators for a single Scatter3d."""
    xs, ys, zs = [], [], []
    for seg in segments:
        for p in seg:
            xs.append(p[0]); ys.append(p[1]); zs.append(p[2])
        xs.append(None); ys.append(None); zs.append(None)
    return xs, ys, zs


# --------------------------------------------------------------------------
def build_rig(veh: VehicleParams, brake_l: float, brake_r: float,
              spool_l: float, spool_r: float):
    """Full rig geometry in the body frame.

    brake_l/r: normalized line pull per side (0..1);
    spool_l/r: spool angles [rad] for the servo-arm glyphs.

    Returns a dict of parts:
      canopy   -> (verts, faces)          payload -> (verts, faces)
      ribs     -> polyline segments       susp    -> polyline segments
      brake_l / brake_r -> polyline segments (incl. rotating spool arm)
    """
    veh_grid = _canopy_grid(veh, brake_l, brake_r)
    canopy = _grid_mesh(veh_grid)

    # cell ribs: chordwise polylines at every station
    ribs = [veh_grid[i] for i in range(N_RIB)]
    # leading/trailing edge outlines for crispness
    ribs += [veh_grid[:, 0], veh_grid[:, -1]]

    # --- payload + servos --------------------------------------------------
    p0 = veh.r_payload                          # payload CP (body frame)
    pay_c = np.array([0.05, 0.0, p0[2] + 0.05])
    v_pay, f_pay = _box(pay_c, (0.50, 0.12, 0.14))
    servo_c = {"L": np.array([-0.08, -0.085, p0[2] - 0.02]),
               "R": np.array([-0.08, +0.085, p0[2] - 0.02])}
    v_sl, f_sl = _box(servo_c["L"], (0.07, 0.05, 0.07))
    v_sr, f_sr = _box(servo_c["R"], (0.07, 0.05, 0.07))
    verts = np.vstack([v_pay, v_sl, v_sr])
    faces = np.vstack([f_pay, f_sl + len(v_pay), f_sr + len(v_pay) + len(v_sl)])
    payload = (verts, faces)

    # --- suspension lines (A row near LE, C row mid-chord) -----------------
    riser_top = {"L": np.array([0.02, -0.10, p0[2] - 0.10]),
                 "R": np.array([0.02, +0.10, p0[2] - 0.10])}
    conf = {"L": riser_top["L"] + 0.62 * (veh.r_canopy - riser_top["L"]) + np.array([0, -0.12, 0]),
            "R": riser_top["R"] + 0.62 * (veh.r_canopy - riser_top["R"]) + np.array([0, +0.12, 0])}
    susp = []
    for j_frac, x_pull in ((0.12, None), (0.55, None)):   # A row, C row
        j = int(round(j_frac * (N_CHORD - 1)))
        for i in range(0, N_RIB, 2):
            side = "L" if veh_grid[i, j, 1] < 0 else "R"
            susp.append(np.vstack([veh_grid[i, j], conf[side]]))
    for side in ("L", "R"):                                # risers
        susp.append(np.vstack([conf[side], riser_top[side]]))

    # --- brake/steering lines + spool arms ---------------------------------
    def brake_side(side: str, spool_angle: float):
        sgn = -1.0 if side == "L" else 1.0
        i_att = 1 if side == "L" else N_RIB - 2          # outboard TE rib
        te_att = veh_grid[i_att, -1]                      # deflects with brake
        guide = conf[side] + np.array([-0.05, sgn * 0.03, 0.05])
        spool = servo_c[side]
        segs = [np.vstack([te_att, guide, spool])]
        # rotating spool arm (in the body x-z plane)
        arm = spool + 0.055 * np.array([np.cos(spool_angle), 0.0,
                                        -np.sin(spool_angle)])
        segs.append(np.vstack([spool, arm]))
        return segs

    return {
        "canopy": canopy,
        "payload": payload,
        "ribs": ribs,
        "susp": susp,
        "brake_l": brake_side("L", spool_l),
        "brake_r": brake_side("R", spool_r),
    }


# --------------------------------------------------------------------------
def transform_parts(parts: dict, R: np.ndarray, pos_ned: np.ndarray,
                    scale: float) -> dict:
    """Rotate body->NED, scale about the CG, translate to pos_ned."""
    def tf(pts):
        return (R @ (np.asarray(pts).T * scale)).T + np.asarray(pos_ned)

    out = {}
    for key, val in parts.items():
        if key in ("canopy", "payload"):
            out[key] = (tf(val[0]), val[1])
        else:
            out[key] = [tf(seg) for seg in val]
    return out
