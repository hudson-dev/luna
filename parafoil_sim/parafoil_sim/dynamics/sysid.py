"""Calibrate the reduced prediction model against the 6-DOF plant.

Runs two short open-loop 6-DOF experiments in calm air:

1. Trim glide (da = ds = 0): recover steady (Vh, Vv), trim pitch and body
   velocity (also used to initialize the plant without a large transient).
2. Turn step (da = 0.5): recover the steady turn rate -> K_turn, and the
   63%-rise time of yaw rate -> tau_turn (lumping servo + roll/yaw dynamics).

This is a one-time startup cost (a few thousand RK4 steps) and means the MPC
prediction model tracks whatever the 6-DOF parameter set actually flies like,
instead of relying on hand-derived constants.
"""
from __future__ import annotations

import numpy as np

from ..config import VehicleParams, AtmosphereParams
from ..environment.actuator import BrakeActuators
from .quat import quat_to_euler
from .reduced import ReducedModelParams
from .sixdof import ParafoilPlant

_CALM = lambda h: np.zeros(3)


def _run(plant: ParafoilPlant, da_cmd: float, T: float, dt: float = 0.01):
    act = BrakeActuators(plant.veh)
    n = int(T / dt)
    t = np.linspace(dt, T, n)
    log = {"vel_ned": np.zeros((n, 3)), "r": np.zeros(n), "pitch": np.zeros(n),
           "v_body": np.zeros((n, 3))}
    for k in range(n):
        da, ds = act.step(dt, da_cmd, 0.0)
        plant.step(dt, da, ds, _CALM)
        log["vel_ned"][k] = plant.vel_ned
        log["r"][k] = plant.omega[2]
        _, log["pitch"][k], _ = quat_to_euler(plant.quat)
        log["v_body"][k] = plant.v_body
    return t, log


def identify_reduced_model(veh: VehicleParams, atmo: AtmosphereParams,
                           alt: float = 400.0, verbose: bool = False) -> ReducedModelParams:
    # --- experiment 1: trim glide ---------------------------------------
    plant = ParafoilPlant(veh, atmo)
    plant.set_state((0.0, 0.0), alt, 0.0, np.array([7.0, 0.0, 2.0]), pitch=-0.1)
    _run(plant, 0.0, 25.0)                      # let transients die
    t, log = _run(plant, 0.0, 10.0)             # measure
    vel = log["vel_ned"].mean(axis=0)
    Vh = float(np.hypot(vel[0], vel[1]))
    Vv = float(vel[2])
    trim_pitch = float(log["pitch"].mean())
    trim_vb = log["v_body"].mean(axis=0)

    # --- experiment 2: turn step -----------------------------------------
    plant2 = ParafoilPlant(veh, atmo)
    plant2.set_state((0.0, 0.0), alt, 0.0, trim_vb.copy(), pitch=trim_pitch)
    _run(plant2, 0.0, 5.0)
    da_step = 0.5
    t, log = _run(plant2, da_step, 30.0)
    r_ss = float(log["r"][-int(len(t) * 0.3):].mean())
    K_turn = r_ss / da_step
    # 63% rise time of yaw rate as the lumped first-order time constant
    idx = np.argmax(log["r"] >= 0.632 * r_ss) if r_ss > 1e-4 else 0
    tau_turn = float(np.clip(t[idx] if idx > 0 else 1.0, 0.3, 4.0))

    params = ReducedModelParams(Vh=Vh, Vv=Vv, K_turn=K_turn, tau_turn=tau_turn,
                                trim_pitch=trim_pitch, trim_v_body=trim_vb)
    if verbose:
        print(f"[sysid] Vh={Vh:.2f} m/s  Vv={Vv:.2f} m/s  (L/D={Vh/max(Vv,1e-6):.2f})  "
              f"K_turn={K_turn:.3f} rad/s per da  tau_turn={tau_turn:.2f} s  "
              f"trim_pitch={np.rad2deg(trim_pitch):.1f} deg")
    return params
