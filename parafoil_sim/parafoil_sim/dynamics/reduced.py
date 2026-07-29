"""Reduced 4-DOF kinematic prediction model for the MPC.

This mirrors standard parafoil GNC practice (and the team's MATLAB model):
the high-fidelity 6-DOF plant is controlled through a low-order kinematic
model whose parameters are *identified from the 6-DOF plant* at startup
(see sysid.py), so the prediction model is a calibrated abstraction rather
than a hand-tuned guess.

State x = [pN, pE, h, psi, da], control u = da_cmd:

    pN'  = Vh cos(psi) + wN(h)
    pE'  = Vh sin(psi) + wE(h)
    h'   = -Vv
    psi' = K_turn * da
    da'  = (u - da) / tau_turn

* (Vh, Vv) are the identified trim airspeeds; wind w(h) comes from the online
  wind estimator (optionally rescaled with a shear power law).
* K_turn is the identified steady turn rate per unit asymmetric brake, and
  tau_turn the identified lateral response time constant, so actuator lag +
  the roll/yaw transient are lumped into one first-order channel.

`discrete_jacobians` linearizes the RK4-discretized map by finite differences;
these (A_k, B_k, c_k) triplets are the LTV matrices for the QP.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .rk4 import rk4_step

WindFn = Callable[[float], np.ndarray]   # altitude -> (wN, wE)

NX = 5
NU = 1


@dataclass
class ReducedModelParams:
    Vh: float = 7.5          # horizontal airspeed [m/s]
    Vv: float = 3.8          # sink rate [m/s]
    K_turn: float = 0.5      # steady yaw rate per unit da [rad/s]
    tau_turn: float = 1.0    # lateral response time constant [s]
    trim_pitch: float = 0.0  # 6-DOF trim pitch (for plant init) [rad]
    trim_v_body: np.ndarray | None = None  # 6-DOF trim body velocity


class ReducedModel:
    def __init__(self, params: ReducedModelParams):
        self.p = params

    def f(self, x: np.ndarray, u: float, wind_fn: WindFn) -> np.ndarray:
        p = self.p
        w = wind_fn(max(x[2], 0.0))
        return np.array([
            p.Vh * np.cos(x[3]) + w[0],
            p.Vh * np.sin(x[3]) + w[1],
            -p.Vv,
            p.K_turn * x[4],
            (u - x[4]) / p.tau_turn,
        ])

    def step(self, x: np.ndarray, u: float, Ts: float, wind_fn: WindFn,
             stop_at_ground: bool = False) -> np.ndarray:
        # Substep so the stiff lag state (tau_turn << Ts) stays inside the RK4
        # stability region (|dt/tau| < 2.78). Optionally clip the step at
        # touchdown so (pN, pE) is the impact point, not a past-ground state.
        remaining = float(Ts)
        if stop_at_ground and self.p.Vv > 0.0:
            remaining = min(remaining, max(float(x[2]), 0.0) / self.p.Vv)
        if remaining <= 0.0:
            out = x.copy()
            out[2] = max(out[2], 0.0)
            return out
        n_sub = max(1, int(np.ceil(remaining / (1.5 * self.p.tau_turn))))
        dt = remaining / n_sub
        for _ in range(n_sub):
            x = rk4_step(lambda xx: self.f(xx, u, wind_fn), x, dt)
        if stop_at_ground:
            x = x.copy()
            x[2] = max(float(x[2]), 0.0)
        return x

    def discrete_jacobians(self, x: np.ndarray, u: float, Ts: float, wind_fn: WindFn,
                           eps: float = 1e-5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Finite-difference (A, B, c) of x+ = f_d(x, u) about (x, u):
        x+ ≈ A x + B u + c."""
        f0 = self.step(x, u, Ts, wind_fn)
        A = np.zeros((NX, NX))
        for i in range(NX):
            dx = np.zeros(NX)
            dx[i] = eps
            A[:, i] = (self.step(x + dx, u, Ts, wind_fn) - self.step(x - dx, u, Ts, wind_fn)) / (2 * eps)
        B = ((self.step(x, u + eps, Ts, wind_fn) - self.step(x, u - eps, Ts, wind_fn)) / (2 * eps)).reshape(NX, NU)
        c = f0 - A @ x - B.flatten() * u
        return A, B, c
