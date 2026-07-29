"""Online glide-polar estimator.

Low-pass filters the measured air-relative horizontal speed and the measured
sink rate. This keeps the reduced prediction model's (Vh, Vv) honest when the
actual polar shifts (turn-induced sink, turbulence, density altitude), which
directly improves the guidance energy match at the terminal phase.
"""
from __future__ import annotations

import numpy as np


class GlideEstimator:
    """The outputs are clamped to a band around the identified trim polar:
    thermals/updrafts and turbulence would otherwise corrupt the estimate
    (e.g. a 10 s updraft halves the apparent sink rate) and make the
    guidance energy logic dangerously optimistic. Never let the planner
    believe the vehicle sinks much slower than trim."""

    def __init__(self, Vh0: float, Vv0: float, tau: float = 10.0):
        self.Vh0, self.Vv0 = Vh0, Vv0
        self.Vh = Vh0
        self.Vv = Vv0
        self.tau = tau

    def update(self, dt: float, vel_ne_meas: np.ndarray, vz_meas: float,
               w_hat: np.ndarray) -> tuple[float, float]:
        """vz_meas: climb rate (up positive)."""
        v_air = vel_ne_meas - w_hat
        Vh_raw = float(np.hypot(v_air[0], v_air[1]))
        Vv_raw = float(-vz_meas)
        a = dt / (dt + self.tau)
        self.Vh += a * (np.clip(Vh_raw, 2.0, 20.0) - self.Vh)
        self.Vv += a * (np.clip(Vv_raw, 0.5, 8.0) - self.Vv)
        Vh = float(np.clip(self.Vh, 0.85 * self.Vh0, 1.15 * self.Vh0))
        Vv = float(np.clip(self.Vv, 0.92 * self.Vv0, 1.60 * self.Vv0))
        return Vh, Vv
