"""Sensor models: GPS position/velocity, barometric altitude, AHRS heading.

When disabled the controller sees truth (useful for debugging / tuning).
Each sensor holds its last sample between updates (zero-order hold), matching
how a flight computer consumes asynchronous sensor data.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import SensorParams


@dataclass
class NavMeasurement:
    pos_ne: np.ndarray      # (2,) north/east position [m]
    h: float                # altitude AGL [m]
    vel_ne: np.ndarray      # (2,) north/east ground velocity [m/s]
    vz: float               # climb rate (up +) [m/s]
    psi: float              # heading (yaw) [rad]


class SensorSuite:
    def __init__(self, params: SensorParams, rng: np.random.Generator):
        self.p = params
        self.rng = rng
        self._t_gps = -np.inf
        self._t_baro = -np.inf
        self._gps_pos = np.zeros(2)
        self._gps_vel = np.zeros(2)
        self._gps_vz = 0.0
        self._baro_h = 0.0
        self._init = False

    def measure(self, t: float, pos_ne: np.ndarray, h: float,
                vel_ne: np.ndarray, vz: float, psi: float) -> NavMeasurement:
        p = self.p
        if not p.enabled:
            return NavMeasurement(pos_ne.copy(), h, vel_ne.copy(), vz, psi)

        rng = self.rng
        if not self._init or (t - self._t_gps) >= 1.0 / p.gps_rate:
            self._t_gps = t
            self._gps_pos = pos_ne + p.gps_pos_sigma * rng.standard_normal(2)
            self._gps_vel = vel_ne + p.gps_vel_sigma * rng.standard_normal(2)
            self._gps_vz = vz + p.gps_vel_sigma * 1.5 * rng.standard_normal()
        if not self._init or (t - self._t_baro) >= 1.0 / p.baro_rate:
            self._t_baro = t
            self._baro_h = h + p.baro_sigma * rng.standard_normal()
        self._init = True

        psi_m = psi + np.deg2rad(p.heading_sigma_deg) * rng.standard_normal()
        return NavMeasurement(self._gps_pos.copy(), self._baro_h,
                              self._gps_vel.copy(), self._gps_vz, psi_m)
