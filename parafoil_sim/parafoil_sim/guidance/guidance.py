"""Phased guidance: homing -> loiter (energy management) -> final approach ->
sensor-triggered flare.

The guidance layer does two jobs each control cycle:

1. Phase logic based on altitude and geometry, all wind-aware via the online
   wind estimate:
   * HOMING    : fly toward the loiter/entry point near the pad.
   * LOITER    : circle the final-approach entry point to bleed excess
                 altitude (energy management).
   * APPROACH  : descend along a straight line INTO the estimated wind,
                 arriving at the pad at ground level.
   * FLARE     : below h_flare (barometer/rangefinder trigger, deliberately
                 NOT model-predicted), ramp symmetric brake to kill speed.

2. Reference generation: roll the reduced kinematic model forward under a
   simple pursuit law appropriate to the phase. This produces a *feasible*
   state/input reference which the LTV-MPC linearizes along and then refines
   -- the standard "linearize once per step along a reference trajectory"
   embedded MPC pattern.

Geometry: the Final Approach Point (FAP) sits downwind of the target at the
distance the vehicle covers while descending from h_approach flying upwind:

    L_app = max(Vh - W, Vg_min) * h_approach / Vv
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import GuidanceParams, MPCParams
from ..dynamics.quat import wrap_angle
from ..dynamics.reduced import ReducedModel

HOMING, LOITER, EXTEND, APPROACH, FLARE = ("HOMING", "LOITER", "EXTEND",
                                            "APPROACH", "FLARE")


@dataclass
class StageWeights:
    q_pos: float
    q_psi: float
    pin_terminal_to_target: bool


@dataclass
class GuidanceOutput:
    phase: str
    x_ref: np.ndarray            # (N+1, 5) reference states (incl. current)
    u_ref: np.ndarray            # (N,) reference inputs
    weights: StageWeights
    ds_cmd: float                # symmetric brake command (flare)
    loiter_center: np.ndarray = field(default_factory=lambda: np.zeros(2))
    fap: np.ndarray = field(default_factory=lambda: np.zeros(2))


class PhasedGuidance:
    def __init__(self, gp: GuidanceParams, mp: MPCParams, model: ReducedModel,
                 target: np.ndarray):
        self.gp = gp
        self.mp = mp
        self.model = model
        self.target = np.asarray(target, dtype=float)
        self.phase = HOMING
        self._loiter_dir = 1.0        # +1 = CCW (as seen from above, NED)
        self._t_flare: float | None = None
        self._downwind = np.array([1.0, 0.0])  # fallback if wind ~ 0
        # Shear-aware energy bookkeeping: mean wind over a descent from h to
        # ground under the power-law profile is W(h) * 1/(1+exponent).
        self._shear_avg = 1.0 / (1.0 + mp.shear_exponent) if mp.shear_aware else 1.0

    # ------------------------------------------------------------- geometry
    def _wind_geometry(self, w_hat: np.ndarray):
        W = float(np.linalg.norm(w_hat))
        if W > 0.4:
            self._downwind = w_hat / W
        d_hat = self._downwind
        Vg_app = max(self.model.p.Vh - self._shear_avg * W, self.gp.approach_speed_min)
        # Nominal approach consumes ~65% of the approach altitude cap, and the
        # loiter circle radius is subtracted so even the far side of the
        # loiter stays comfortably inside the reachable cone.
        L_app = Vg_app * (0.65 * self.gp.h_approach) / self.model.p.Vv
        L_fap = max(L_app - self.gp.loiter_radius, 25.0)
        fap = self.target + d_hat * L_fap
        return W, d_hat, fap, L_app

    # ---------------------------------------------------------- phase logic
    def _update_phase(self, t: float, x: np.ndarray, fap: np.ndarray, W: float,
                      w_hat: np.ndarray) -> None:
        h = x[2]
        if self.phase == FLARE:
            return
        if h <= self.gp.h_flare:
            self.phase = FLARE
            self._t_flare = t
            return
        if self.phase == APPROACH:
            return                       # committed
        # Energy-matched approach commit: remaining descent time vs the time
        # needed to fly (and turn) to the pad against the wind, with margin.
        mdl = self.model.p
        t_go = h / mdl.Vv
        turn_rate = mdl.K_turn * (self.mp.da_limit - self.mp.da_margin)
        err_pad = abs(wrap_angle(self._bearing(x[0:2], self.target) - x[3]))
        t_turn = 0.75 * err_pad / max(turn_rate, 0.05)
        # Wind drifts the vehicle during the line-up turn; evaluate distance
        # and ground speed from the post-turn position along the actual
        # bearing to the pad (handles up-, down-, and cross-wind finals).
        pos_turn = x[0:2] + w_hat * t_turn
        rel = self.target - pos_turn
        d_pad = max(float(np.linalg.norm(rel)), 1e-3)
        u_hat = rel / d_pad
        Vg = max(mdl.Vh + self._shear_avg * float(w_hat @ u_hat),
                 self.gp.approach_speed_min)
        if self.phase == EXTEND:
            # The return from the extend leg is upwind by construction; use
            # the conservative magnitude-based ground speed so turbulence
            # rotating the wind estimate cannot postpone the base turn.
            Vg = max(mdl.Vh - self._shear_avg * W, self.gp.approach_speed_min)
        m, pad_s = self.gp.approach_margin, self.gp.approach_pad_s
        t_req = m * (d_pad / Vg + t_turn) + pad_s

        # Final commit: energy matches the (turn + upwind run) to the pad.
        if self.phase == EXTEND and t_go <= t_req:
            self.phase = APPROACH
            return
        # Energy deficit safety net from any phase: straight run barely makes
        # it -- commit immediately regardless of geometry.
        if h <= self.gp.h_approach and t_go <= m * (d_pad / Vg) + pad_s:
            self.phase = APPROACH
            return
        # Leave the loiter onto the downwind extend leg once the remaining
        # excess is small enough to be absorbed by the extension.
        if (self.phase == LOITER and h <= self.gp.h_approach
                and t_go <= t_req + self.gp.extend_reserve_s):
            self.phase = EXTEND
            return
        if self.phase == EXTEND:
            return
        dist = float(np.linalg.norm(x[0:2] - fap))
        if self.phase == HOMING and dist < self.gp.loiter_capture:
            # choose circulation direction requiring the least initial turning
            rel = x[0:2] - fap
            tangent_ccw = np.arctan2(rel[0], -rel[1])   # +90 deg from radial
            e_ccw = abs(wrap_angle(tangent_ccw - x[3]))
            self._loiter_dir = 1.0 if e_ccw < np.pi / 2 else -1.0
            self.phase = LOITER
        elif self.phase == LOITER and dist > 2.5 * self.gp.loiter_capture:
            self.phase = HOMING

    # ----------------------------------------------------- pursuit sub-laws
    def _psi_des(self, x: np.ndarray, w: np.ndarray, fap: np.ndarray,
                 L_app: float) -> float:
        """Desired heading (crab-corrected desired course) for current phase."""
        pos = x[0:2]
        if self.phase == HOMING:
            chi = self._bearing(pos, fap)
        elif self.phase == LOITER:
            rel = pos - fap
            d = max(np.linalg.norm(rel), 1e-6)
            R = self.gp.loiter_radius
            tangent = np.arctan2(self._loiter_dir * rel[0], -self._loiter_dir * rel[1])
            corr = np.clip(1.2 * (d - R) / R, -np.pi / 2.5, np.pi / 2.5)
            chi = tangent + self._loiter_dir * corr
        elif self.phase == EXTEND:
            # downwind leg along the approach corridor, away from the pad
            chi = self._bearing(np.zeros(2), self._downwind)
        else:  # APPROACH / FLARE: pursue a glide-slope point sliding to target
            W_loc = float(np.hypot(w[0], w[1]))
            Vg = max(self.model.p.Vh - self._shear_avg * W_loc,
                     self.gp.approach_speed_min)
            # point on the upwind final line that matches remaining energy
            # (0.9: aim slightly inside the reachable cone; small excess energy
            # is absorbed by the MPC weaving rather than landing short)
            L_rem = 0.9 * Vg * max(x[2], 0.0) / self.model.p.Vv
            d_pad = float(np.linalg.norm(pos - self.target))
            # never aim farther from the pad than the vehicle is: excess energy
            # is absorbed by the curved intercept, not by flying outbound
            line_pt = self.target + self._downwind * min(L_rem, L_app, d_pad)
            chi_up = self._bearing(self._downwind, np.zeros(2))   # into the wind
            if self.model.p.Vh - W_loc <= 2.4:   # local penetration margin
                # Near wind saturation (W ~ Vh) the pursuit geometry
                # degenerates: hold the into-wind heading and steer the small
                # controllable margin against the cross-corridor error.
                p_hat = np.array([-self._downwind[1], self._downwind[0]])
                e_lat = float((pos - self.target) @ p_hat)
                chi = chi_up + float(np.clip(0.6 * e_lat / max(self.model.p.Vh, 1.0),
                                             -0.5, 0.5))
            elif np.linalg.norm(line_pt - pos) > 30.0:
                # pursuit needs a minimum lookahead or the bearing degenerates
                # into chatter when the vehicle sits on the aim point
                chi = self._bearing(pos, line_pt)
            else:
                chi = chi_up
            if x[2] < self.gp.h_blend:   # align into the wind for touchdown
                lam = np.clip(x[2] / self.gp.h_blend, 0.0, 1.0)
                chi = chi_up + lam * wrap_angle(chi - chi_up)
        # crab correction so ground track follows chi
        c, p_hat = np.array([np.cos(chi), np.sin(chi)]), np.array([-np.sin(chi), np.cos(chi)])
        w_perp = float(w[0:2] @ p_hat)
        crab = np.arcsin(np.clip(w_perp / max(self.model.p.Vh, 1.0), -0.9, 0.9))
        return chi - crab

    @staticmethod
    def _bearing(frm: np.ndarray, to: np.ndarray) -> float:
        d = to - frm
        return float(np.arctan2(d[1], d[0]))

    # -------------------------------------------------------------- rollout
    def update(self, t: float, x0: np.ndarray, w_hat: np.ndarray, wind_fn,
               N: int) -> GuidanceOutput:
        """Generate phase + reference trajectory for the horizon.

        x0: current reduced state [pN, pE, h, psi(unwrapped), da]
        wind_fn: h -> (wN, wE, ...) prediction wind (shear-aware estimate)
        """
        W, d_hat, fap, L_app = self._wind_geometry(w_hat)
        self._update_phase(t, x0, fap, W, np.asarray(w_hat, dtype=float))

        Ts = self.mp.Ts
        mdl = self.model
        da_lim = self.mp.da_limit - self.mp.da_margin
        # keep the discrete pursuit contraction |1 - kp*Ts| < 1 (stability of
        # the reference rollout itself)
        kp = min(self.gp.kp_heading, 0.6 / Ts)
        x_ref = np.zeros((N + 1, 5))
        u_ref = np.zeros(N)
        x_ref[0] = x0
        xk = x0.copy()
        for k in range(N):
            w_k = wind_fn(max(xk[2], 0.0))
            psi_des = self._psi_des(xk, w_k, fap, L_app)
            err = wrap_angle(psi_des - xk[3])
            u = float(np.clip(kp * err / mdl.p.K_turn, -da_lim, da_lim))
            u_ref[k] = u
            # Clip integration at h=0 so the terminal reference is the
            # touchdown point, not an underground extrapolation.
            xk = mdl.step(xk, u, Ts, wind_fn, stop_at_ground=True)
            x_ref[k + 1] = xk
            if xk[2] <= 0.0:
                u_ref[k + 1:] = u
                x_ref[k + 2:] = xk
                break

        # phase-dependent weighting
        mp = self.mp
        if self.phase in (APPROACH, FLARE):
            wts = StageWeights(q_pos=3.0 * mp.q_pos_run, q_psi=mp.q_psi_approach,
                               pin_terminal_to_target=True)
        elif self.phase == LOITER:
            wts = StageWeights(q_pos=4.0 * mp.q_pos_run, q_psi=mp.q_psi_run,
                               pin_terminal_to_target=False)
        else:
            wts = StageWeights(q_pos=mp.q_pos_run, q_psi=mp.q_psi_run,
                               pin_terminal_to_target=False)
        if wts.pin_terminal_to_target:
            x_ref[N, 0:2] = self.target

        # flare symmetric brake ramp (sensor-triggered, time-based ramp)
        ds_cmd = 0.0
        if self.phase == FLARE and self._t_flare is not None:
            ds_cmd = self.gp.flare_ds * float(np.clip((t - self._t_flare) / self.gp.flare_ramp, 0.0, 1.0))

        return GuidanceOutput(phase=self.phase, x_ref=x_ref, u_ref=u_ref,
                              weights=wts, ds_cmd=ds_cmd,
                              loiter_center=fap.copy(), fap=fap.copy())
