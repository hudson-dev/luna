"""Online horizontal wind estimator.

Same structure as the team's flight approach: the raw wind sample is

    w_raw = v_ground(GPS) - v_air(predicted)

where the predicted air velocity is the identified trim airspeed along the
measured heading. A first-order low-pass filter rejects GPS noise and
turbulence while tracking the persistent (biasing) component of the wind --
which is exactly the failure mode the estimator exists to kill.
"""
from __future__ import annotations

import numpy as np


class WindEstimator:
    def __init__(self, tau: float, Vh: float):
        self.tau = tau
        self.Vh = Vh
        self.w_hat = np.zeros(2)   # (wN, wE)

    def update(self, dt: float, vel_ne_meas: np.ndarray, psi_meas: float) -> np.ndarray:
        v_air_pred = self.Vh * np.array([np.cos(psi_meas), np.sin(psi_meas)])
        w_raw = vel_ne_meas - v_air_pred
        a = dt / (dt + self.tau)
        self.w_hat = self.w_hat + a * (w_raw - self.w_hat)
        return self.w_hat.copy()
