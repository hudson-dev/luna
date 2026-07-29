"""Brake-line servo rig: two winch servos reeling in the left/right brake
(steering) lines.

Physical chain (per the project safety principle, the servos act ONLY on the
brake lines, never the primary risers):

    GNC command (da, ds)  --mix-->  left/right line-travel commands [m]
        left  = clip(ds - da, 0, 1) * line_travel_max
        right = clip(ds + da, 0, 1) * line_travel_max
    servo dynamics (per side): first-order lag (tau) + line-speed limit
        (servo_rate_max * line_travel_max  [m/s])  + travel saturation
    spool angle = line_travel / spool_radius
    aero inputs recovered from the actual line states:
        da = (right - left) / line_travel_max
        ds = min(left, right) / line_travel_max

With both sides unsaturated this is dynamically identical to independent
first-order da/ds channels (the mixing is linear), so the closed-loop
behavior matches the original normalized model; the physical states exist for
realism, logging, and the rig visualization.
"""
from __future__ import annotations

import numpy as np

from ..config import VehicleParams


class BrakeActuators:
    def __init__(self, veh: VehicleParams):
        self.veh = veh
        self.line_left = 0.0    # actual left brake-line travel [m]
        self.line_right = 0.0   # actual right brake-line travel [m]

    # ------------------------------------------------------------ accessors
    @property
    def da(self) -> float:
        return (self.line_right - self.line_left) / self.veh.line_travel_max

    @property
    def ds(self) -> float:
        return min(self.line_left, self.line_right) / self.veh.line_travel_max

    @property
    def spool_angle_left(self) -> float:
        """Left spool angle [rad] (0 = brake released)."""
        return self.line_left / self.veh.spool_radius

    @property
    def spool_angle_right(self) -> float:
        return self.line_right / self.veh.spool_radius

    # -------------------------------------------------------------- dynamics
    def step(self, dt: float, da_cmd: float, ds_cmd: float) -> tuple[float, float]:
        v = self.veh
        da_cmd = float(np.clip(da_cmd, -v.da_max, v.da_max))
        ds_cmd = float(np.clip(ds_cmd, 0.0, v.ds_max))
        cmd_l = float(np.clip(ds_cmd - da_cmd, 0.0, 1.0)) * v.line_travel_max
        cmd_r = float(np.clip(ds_cmd + da_cmd, 0.0, 1.0)) * v.line_travel_max

        v_line_max = v.line_speed_max
        for name, cmd in (("line_left", cmd_l), ("line_right", cmd_r)):
            x = getattr(self, name)
            rate = (cmd - x) / v.servo_tau
            rate = float(np.clip(rate, -v_line_max, v_line_max))
            x = float(np.clip(x + dt * rate, 0.0, v.line_travel_max))
            setattr(self, name, x)
        return self.da, self.ds
