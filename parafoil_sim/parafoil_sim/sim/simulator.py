"""Closed-loop simulation: 6-DOF plant + sensors + wind estimator + phased
guidance + LTV-QP MPC + servo actuators.

Architecture per plant step (dt):

    true wind ──► 6-DOF plant ──► sensors (GPS/baro/AHRS, ZOH)
                     ▲                    │
                servo lag/rate      wind estimator (LPF)
                     ▲                    │
                     └── MPC (every t_control): guidance phase + reference
                         rollout -> linearize -> OSQP -> u0

The controller never sees the true wind or the true state when sensors are
enabled; its prediction model is the sysid-calibrated reduced model.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from ..config import Scenario
from ..control.ltv_mpc import LTVMPC
from ..dynamics.quat import quat_to_euler, wrap_angle
from ..dynamics.reduced import ReducedModel
from ..dynamics.sixdof import ParafoilPlant
from ..dynamics.sysid import identify_reduced_model
from ..environment.actuator import BrakeActuators
from ..environment.sensors import SensorSuite
from ..environment.wind import WindField
from ..estimation.glide_estimator import GlideEstimator
from ..estimation.wind_estimator import WindEstimator
from ..guidance.guidance import PhasedGuidance, FLARE


def apply_bank_limit(u_cmd: float, roll: float, bank_max: float,
                     da_lim: float) -> float:
    """Enforce a bank envelope without fighting normal coordinated turns.

    Inside ``±0.9·bank_max`` the command is only clipped to ``±da_lim``.
    Between the soft threshold and ``bank_max``, further brake in the
    direction that increases bank is ramped to zero. Past ``bank_max``,
    only unloading / opposite brake is allowed.

    Convention (body FRD): +roll = right wing down; +da = right brake → +roll.
    """
    bank_max = max(float(bank_max), 1e-3)
    da_lim = max(float(da_lim), 0.0)
    u = float(np.clip(u_cmd, -da_lim, da_lim))
    soft = 0.9 * bank_max
    if roll >= soft and u > 0.0:
        scale = float(np.clip((bank_max - roll) / (bank_max - soft + 1e-9), 0.0, 1.0))
        u = min(u, da_lim * scale)
    elif roll <= -soft and u < 0.0:
        scale = float(np.clip((bank_max + roll) / (bank_max - soft + 1e-9), 0.0, 1.0))
        u = max(u, -da_lim * scale)
    return float(u)


@dataclass
class SimResult:
    scenario: Scenario
    model_params: object
    # per-plant-step logs (arrays of length n)
    t: np.ndarray = None
    pos: np.ndarray = None          # (n,3) NED
    euler: np.ndarray = None        # (n,3) roll/pitch/yaw
    vel_ned: np.ndarray = None      # (n,3)
    airspeed: np.ndarray = None
    da: np.ndarray = None           # actual asymmetric deflection
    ds: np.ndarray = None           # actual symmetric deflection
    da_cmd: np.ndarray = None
    line_l: np.ndarray = None       # left brake-line travel [m]
    line_r: np.ndarray = None       # right brake-line travel [m]
    wind_true: np.ndarray = None    # (n,3) at vehicle altitude
    wind_hat: np.ndarray = None     # (n,2)
    phase: list = field(default_factory=list)
    # per-control-step logs
    t_mpc: list = field(default_factory=list)
    mpc_cost: list = field(default_factory=list)
    mpc_ok: list = field(default_factory=list)
    mpc_pred: list = field(default_factory=list)   # (N,5) predicted states
    fap: list = field(default_factory=list)
    # touchdown summary
    miss_distance: float = np.nan
    t_flight: float = np.nan
    td_ground_speed: float = np.nan
    td_sink_rate: float = np.nan
    td_airspeed: float = np.nan
    flare_triggered: bool = False
    landed: bool = False

    def summary(self) -> str:
        s = self.scenario
        lines = [
            f"scenario '{s.name}': {'LANDED' if self.landed else 'DID NOT LAND (aborted)'}",
            f"  flight time         : {self.t_flight:7.1f} s",
            f"  miss distance       : {self.miss_distance:7.1f} m",
            f"  touchdown ground spd: {self.td_ground_speed:7.2f} m/s",
            f"  touchdown sink rate : {self.td_sink_rate:7.2f} m/s",
            f"  touchdown airspeed  : {self.td_airspeed:7.2f} m/s",
            f"  flare triggered     : {self.flare_triggered}",
        ]
        return "\n".join(lines)


def run_scenario(scn: Scenario, verbose: bool = True) -> SimResult:
    rng = np.random.default_rng(scn.sim.seed)
    t_start = time.time()

    # --- calibrate the reduced prediction model against the 6-DOF plant ----
    mparams = identify_reduced_model(scn.vehicle, scn.atmosphere, verbose=verbose)
    model = ReducedModel(mparams)

    # --- plant, environment, GNC -------------------------------------------
    plant = ParafoilPlant(scn.vehicle, scn.atmosphere)
    plant.set_state(scn.release_pos, scn.release_alt,
                    np.deg2rad(scn.release_heading_deg),
                    mparams.trim_v_body.copy(), pitch=mparams.trim_pitch)
    wind = WindField(scn.wind, rng)
    sensors = SensorSuite(scn.sensors, rng)
    actuators = BrakeActuators(scn.vehicle)
    estimator = WindEstimator(scn.estimator.tau_wind, mparams.Vh)
    glide_est = GlideEstimator(mparams.Vh, mparams.Vv)
    target = np.asarray(scn.target, dtype=float)
    guidance = PhasedGuidance(scn.guidance, scn.mpc, model, target)
    mpc = LTVMPC(scn.mpc, model, servo_rate_max=scn.vehicle.servo_rate_max)

    dt = scn.sim.dt
    n_max = int(scn.sim.t_max / dt) + 1
    res = SimResult(scenario=scn, model_params=mparams)
    L = {k: np.zeros((n_max, d)) if d > 1 else np.zeros(n_max)
         for k, d in [("t", 1), ("pos", 3), ("euler", 3), ("vel_ned", 3),
                      ("airspeed", 1), ("da", 1), ("ds", 1), ("da_cmd", 1),
                      ("line_l", 1), ("line_r", 1),
                      ("wind_true", 3), ("wind_hat", 2)]}

    # controller internal state
    psi_unwrapped = np.deg2rad(scn.release_heading_deg)
    da_est = 0.0                    # controller's belief of the lateral state
    u_hold = 0.0                    # last MPC asymmetric command (pre bank-limit)
    u_cmd = 0.0
    ds_cmd = 0.0
    ctimer = 0.0

    def wind_pred_fn_factory(w_hat: np.ndarray, h_now: float):
        """Shear-aware prediction wind: scale the estimate with the power law."""
        if scn.mpc.shear_aware:
            h0 = max(h_now, 5.0)
            ex = scn.mpc.shear_exponent

            def fn(h: float) -> np.ndarray:
                s = (max(h, 2.0) / h0) ** ex
                return np.array([w_hat[0] * s, w_hat[1] * s, 0.0])
        else:
            def fn(h: float) -> np.ndarray:
                return np.array([w_hat[0], w_hat[1], 0.0])
        return fn

    t = 0.0
    k = 0
    while k < n_max:
        h = plant.altitude
        if h <= 0.0:
            break

        # --- environment ------------------------------------------------
        w_true = wind.wind_at(h)
        wind.step(dt, h, plant.airspeed(w_true))

        # --- sensing & estimation ----------------------------------------
        vel = plant.vel_ned
        roll, pitch, psi_true = quat_to_euler(plant.quat)
        meas = sensors.measure(t, plant.pos_ned[0:2].copy(), h,
                               vel[0:2].copy(), -vel[2], psi_true)
        w_hat = estimator.update(dt, meas.vel_ne, meas.psi)
        # keep the prediction model's glide polar honest (turn-induced sink,
        # turbulence, density altitude)
        mparams.Vh, mparams.Vv = glide_est.update(dt, meas.vel_ne, meas.vz, w_hat)
        psi_unwrapped += wrap_angle(meas.psi - psi_unwrapped)

        # --- GNC at the control rate --------------------------------------
        da_lim = scn.mpc.da_limit - scn.mpc.da_margin
        bank_max = np.deg2rad(scn.mpc.bank_max_deg)
        if ctimer <= 0.0:
            x0 = np.array([meas.pos_ne[0], meas.pos_ne[1], max(meas.h, 0.0),
                           psi_unwrapped, da_est])
            # conservative planning wind (constraint-margin philosophy: bias
            # the inevitable energy errors toward the upwind side of the pad)
            w_ctrl = scn.mpc.wind_margin * w_hat
            wind_pred = wind_pred_fn_factory(w_ctrl, meas.h)
            N = mpc.horizon_length(meas.h)
            guid = guidance.update(t, x0, w_ctrl, wind_pred, N)
            sol = mpc.solve(x0, guid, wind_pred, u_hold)
            u_hold = sol.u0
            ds_cmd = guid.ds_cmd
            res.t_mpc.append(t)
            res.mpc_cost.append(sol.cost)
            res.mpc_ok.append(sol.ok)
            res.mpc_pred.append(sol.x_pred.copy())
            res.fap.append(guid.fap.copy())
            ctimer = scn.sim.t_control
        ctimer -= dt
        if guidance.phase == FLARE:
            ds_cmd = guid.ds_cmd = guidance.gp.flare_ds * float(
                np.clip((t - guidance._t_flare) / guidance.gp.flare_ramp, 0.0, 1.0))
            res.flare_triggered = True

        # bank envelope: applied every plant step so we unload brakes as soon
        # as roll approaches the limit (not just at the MPC tick)
        u_cmd = apply_bank_limit(u_hold, roll, bank_max, da_lim)

        # controller's internal lateral-state propagation (matches model)
        da_est += dt * (u_cmd - da_est) / mparams.tau_turn
        da_est = float(np.clip(da_est, -da_lim, da_lim))

        # --- actuate & integrate plant --------------------------------------
        da_act, ds_act = actuators.step(dt, u_cmd, ds_cmd)
        plant.step(dt, da_act, ds_act, wind.wind_at)

        # --- log ---------------------------------------------------------
        L["t"][k] = t
        L["pos"][k] = plant.pos_ned
        L["euler"][k] = quat_to_euler(plant.quat)
        L["vel_ned"][k] = plant.vel_ned
        L["airspeed"][k] = plant.airspeed(w_true)
        L["da"][k], L["ds"][k], L["da_cmd"][k] = da_act, ds_act, u_cmd
        L["line_l"][k], L["line_r"][k] = actuators.line_left, actuators.line_right
        L["wind_true"][k] = w_true
        L["wind_hat"][k] = w_hat
        res.phase.append(guidance.phase)

        t += dt
        k += 1

    # --- trim logs & touchdown summary -------------------------------------
    for key, arr in L.items():
        setattr(res, key, arr[:k])
    res.landed = plant.altitude <= 0.0
    res.t_flight = t
    pos_ne = plant.pos_ned[0:2]
    res.miss_distance = float(np.linalg.norm(pos_ne - target))
    vel = plant.vel_ned
    res.td_ground_speed = float(np.hypot(vel[0], vel[1]))
    res.td_sink_rate = float(vel[2])
    res.td_airspeed = plant.airspeed(wind.wind_at(0.0))

    if verbose:
        n_fail = sum(1 for ok in res.mpc_ok if not ok)
        print(res.summary())
        print(f"  MPC solves          : {len(res.mpc_ok)} "
              f"({n_fail} failed)  |  wall time {time.time() - t_start:.1f} s")
    return res
