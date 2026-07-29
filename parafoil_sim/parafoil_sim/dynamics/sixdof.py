"""6-DOF rigid-body parafoil-payload dynamics.

State (13):
    p     (3)  NED position [m]           (altitude h = -p_z)
    q     (4)  quaternion body->NED, scalar first
    v_b   (3)  CG velocity in body axes [m/s]
    omega (3)  body angular rate [rad/s]

Equations of motion in momentum form with apparent mass (see
`apparent_mass.py` for the Lissaman & Brown terms and the stated
approximations):

    (m I + Ma) v_b' = F_aero + R^T (m g e3) - omega x ((m I + Ma) v_b)
    (J + Ja) omega' = M_aero - omega x ((J + Ja) omega)
    p' = R v_b
    q' = 1/2 q ⊗ [0, omega]

Wind enters through the air-relative velocity v_air = v_b - R^T w(h), with the
wind field evaluated at the instantaneous altitude inside the RK4 stages so
shear is felt within an integration step. Brake actuator states (servo lag +
rate limit) are integrated separately at the plant rate (see
environment/actuator.py) and enter here as the actual deflections.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from ..config import VehicleParams, AtmosphereParams
from ..environment.atmosphere import air_density
from .aero import aero_forces_moments
from .apparent_mass import apparent_mass_matrices
from .quat import quat_to_dcm, quat_derivative, quat_normalize, quat_from_euler
from .rk4 import rk4_step

G = 9.80665

WindFn = Callable[[float], np.ndarray]   # altitude AGL -> wind NED


class ParafoilPlant:
    def __init__(self, veh: VehicleParams, atmo: AtmosphereParams):
        self.veh = veh
        self.atmo = atmo
        self.J = veh.inertia()
        self.x = np.zeros(13)
        self.x[3] = 1.0  # identity quaternion

    # ------------------------------------------------------------ accessors
    @property
    def pos_ned(self) -> np.ndarray:
        return self.x[0:3]

    @property
    def altitude(self) -> float:
        return float(-self.x[2])

    @property
    def quat(self) -> np.ndarray:
        return self.x[3:7]

    @property
    def v_body(self) -> np.ndarray:
        return self.x[7:10]

    @property
    def omega(self) -> np.ndarray:
        return self.x[10:13]

    @property
    def vel_ned(self) -> np.ndarray:
        return quat_to_dcm(self.quat) @ self.v_body

    def airspeed(self, wind_ned: np.ndarray) -> float:
        return float(np.linalg.norm(self.vel_ned - wind_ned))

    # -------------------------------------------------------------- dynamics
    def derivative(self, x: np.ndarray, da: float, ds: float, wind_fn: WindFn) -> np.ndarray:
        veh = self.veh
        q = quat_normalize(x[3:7])
        v_b = x[7:10]
        om = x[10:13]
        R = quat_to_dcm(q)
        h = -x[2]
        rho = air_density(max(h, 0.0), self.atmo)
        Ma, Ja = apparent_mass_matrices(veh, rho)
        Mtot = veh.mass * np.eye(3) + Ma
        Jtot = self.J + Ja

        wind = wind_fn(max(h, 0.0))
        v_air = v_b - R.T @ wind
        F_aero, M_aero = aero_forces_moments(veh, v_air, om, rho, da, ds)
        F_grav = R.T @ np.array([0.0, 0.0, veh.mass * G])

        v_dot = np.linalg.solve(Mtot, F_aero + F_grav - np.cross(om, Mtot @ v_b))
        om_dot = np.linalg.solve(Jtot, M_aero - np.cross(om, Jtot @ om))

        xdot = np.empty(13)
        xdot[0:3] = R @ v_b
        xdot[3:7] = quat_derivative(q, om)
        xdot[7:10] = v_dot
        xdot[10:13] = om_dot
        return xdot

    def step(self, dt: float, da: float, ds: float, wind_fn: WindFn) -> None:
        self.x = rk4_step(lambda xx: self.derivative(xx, da, ds, wind_fn), self.x, dt)
        self.x[3:7] = quat_normalize(self.x[3:7])

    # ------------------------------------------------------- initialization
    def set_state(self, pos_ne: tuple[float, float], alt: float, heading: float,
                  v_body: np.ndarray, pitch: float = 0.0, roll: float = 0.0) -> None:
        self.x = np.zeros(13)
        self.x[0], self.x[1], self.x[2] = pos_ne[0], pos_ne[1], -alt
        self.x[3:7] = quat_from_euler(roll, pitch, heading)
        self.x[7:10] = v_body
