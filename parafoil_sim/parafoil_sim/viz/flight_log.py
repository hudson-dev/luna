"""Export a compact flight log JSON for the Three.js viewer.

Keeps dynamics/MPC untouched: consumes a finished `SimResult` and writes
downsampled trajectory + vehicle params the viewer needs to rebuild the
parafoil mesh and animate brake/spool motion.
"""
from __future__ import annotations

import json
import pathlib
from typing import Any

import numpy as np

from ..dynamics.quat import quat_from_euler


def _round_arr(a: np.ndarray, nd: int) -> list:
    return np.round(np.asarray(a, dtype=float), nd).tolist()


def export_flight_log(res, path, *, dt_out: float = 0.1) -> str:
    """Write a downsampled flight log JSON.

    Parameters
    ----------
    res : SimResult
    path : path-like
    dt_out : float
        Output sample period [s]. Default 0.1 s (10 Hz) keeps files small
        while preserving smooth playback of attitude and brake motion.
    """
    path = pathlib.Path(path)
    t = np.asarray(res.t, dtype=float)
    if len(t) == 0:
        raise ValueError("empty SimResult")

    # stride to ~dt_out, always include the final sample
    dt_in = float(t[1] - t[0]) if len(t) > 1 else dt_out
    step = max(1, int(round(dt_out / max(dt_in, 1e-6))))
    idxs = np.arange(0, len(t), step)
    if idxs[-1] != len(t) - 1:
        idxs = np.append(idxs, len(t) - 1)

    veh = res.scenario.vehicle
    scn = res.scenario

    # body→NED quaternions (scalar-first) for stable attitude playback
    eul = np.asarray(res.euler)[idxs]
    quat = np.array([quat_from_euler(*e) for e in eul])

    pos = np.asarray(res.pos)[idxs]
    # ENU for the viewer: (East, North, Up)
    enu = np.stack([pos[:, 1], pos[:, 0], -pos[:, 2]], axis=-1)

    phase = [res.phase[i] for i in idxs]
    wind = np.asarray(res.wind_true)[idxs]
    # wind ENU: (wE, wN, wUp) with Up = -Down
    wind_enu = np.stack([wind[:, 1], wind[:, 0], -wind[:, 2]], axis=-1)

    payload: dict[str, Any] = {
        "version": 1,
        "scenario": {
            "name": scn.name,
            "description": scn.description,
            "target_enu": [float(scn.target[1]), float(scn.target[0]), 0.0],
            "seed": int(scn.sim.seed),
        },
        "summary": {
            "miss_distance_m": float(res.miss_distance),
            "t_flight_s": float(res.t_flight),
            "td_ground_speed_mps": float(res.td_ground_speed),
            "td_sink_rate_mps": float(res.td_sink_rate),
            "flare_triggered": bool(res.flare_triggered),
            "landed": bool(res.landed),
        },
        "vehicle": {
            "span": float(veh.span),
            "chord": float(veh.chord),
            "thickness": float(veh.thickness),
            "arc_ratio": float(veh.arc_ratio),
            "line_length": float(veh.line_length),
            "line_travel_max": float(veh.line_travel_max),
            "spool_radius": float(veh.spool_radius),
            "r_canopy": _round_arr(veh.r_canopy, 4),
            "r_payload": _round_arr(veh.r_payload, 4),
            "mass": float(veh.mass),
        },
        "frames": {
            "t": _round_arr(t[idxs], 3),
            "pos_enu": _round_arr(enu, 3),
            "quat_wxyz": _round_arr(quat, 5),   # body→NED
            "euler_rpy": _round_arr(eul, 4),
            "line_l": _round_arr(np.asarray(res.line_l)[idxs], 4),
            "line_r": _round_arr(np.asarray(res.line_r)[idxs], 4),
            "da": _round_arr(np.asarray(res.da)[idxs], 4),
            "ds": _round_arr(np.asarray(res.ds)[idxs], 4),
            "wind_enu": _round_arr(wind_enu, 3),
            "phase": phase,
            "airspeed": _round_arr(np.asarray(res.airspeed)[idxs], 3),
        },
    }

    # optional FAP (last guidance estimate) for pad markers
    if res.fap:
        fap = np.asarray(res.fap[-1], dtype=float)
        payload["scenario"]["fap_enu"] = [float(fap[1]), float(fap[0]), 0.0]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, separators=(",", ":")))
    return str(path)
