"""Aerodynamic force and moment model for the parafoil-payload system.

Canopy aerodynamics use a standard linear-coefficient buildup (Slegers &
Costello style) evaluated in the canopy frame (== body frame; the body x-axis
is aligned with the canopy chord):

    CL = CL0 + CLa*alpha + CLds*ds
    CD = CD0 + CDa2*alpha^2 + CDds*ds + CDda*|da|
    CY = CYb*beta
    Cl = Clb*beta + Clp*p_hat + Clda*da
    Cm = Cm0 + Cma*alpha + Cmq*q_hat
    Cn = Cnb*beta + Cnr*r_hat + Cnda*da

with p_hat = p*b/(2V) etc. Lift/drag act in wind axes at the canopy aero
center, which sits above the CG, so the aero force also produces the pendulum
restoring moment via r_canopy x F. The payload contributes a pure drag force
at its own station below the CG.
"""
from __future__ import annotations

import numpy as np

from ..config import VehicleParams


def aero_forces_moments(veh: VehicleParams, v_air_b: np.ndarray, omega_b: np.ndarray,
                        rho: float, da: float, ds: float) -> tuple[np.ndarray, np.ndarray]:
    """Total aerodynamic force and moment (body frame, about CG).

    v_air_b: air-relative velocity of the CG in body axes [m/s]
    omega_b: body angular rate [rad/s]
    da, ds:  asymmetric [-1,1] / symmetric [0,1] brake deflections (actual)
    """
    V = float(np.linalg.norm(v_air_b))
    if V < 0.3:
        return np.zeros(3), np.zeros(3)

    u, v, w = v_air_b
    alpha = float(np.arctan2(w, u))
    beta = float(np.arcsin(np.clip(v / V, -1.0, 1.0)))
    p, q, r = omega_b
    b, c, S = veh.span, veh.chord, veh.S
    qbar = 0.5 * rho * V * V
    p_hat = p * b / (2.0 * V)
    q_hat = q * c / (2.0 * V)
    r_hat = r * b / (2.0 * V)

    CL = veh.CL0 + veh.CLa * alpha + veh.CLds * ds
    CD = veh.CD0 + veh.CDa2 * alpha**2 + veh.CDds * ds + veh.CDda * abs(da)
    CY = veh.CYb * beta
    Cl = veh.Clb * beta + veh.Clp * p_hat + veh.Clda * da
    Cm = veh.Cm0 + veh.Cma * alpha + veh.Cmq * q_hat
    Cn = veh.Cnb * beta + veh.Cnr * r_hat + veh.Cnda * da

    # wind->body rotation (standard alpha/beta transformation)
    ca, sa = np.cos(alpha), np.sin(alpha)
    cb, sb = np.cos(beta), np.sin(beta)
    C_bw = np.array([
        [ca * cb, -ca * sb, -sa],
        [sb,       cb,       0.0],
        [sa * cb, -sa * sb,  ca],
    ])
    F_canopy = C_bw @ np.array([-qbar * S * CD, qbar * S * CY, -qbar * S * CL])

    # payload drag (opposes air-relative velocity)
    F_payload = -0.5 * rho * V * veh.CD_payload * veh.S_payload * v_air_b

    F = F_canopy + F_payload
    M = qbar * S * np.array([b * Cl, c * Cm, b * Cn])
    M += np.cross(veh.r_canopy, F_canopy)
    M += np.cross(veh.r_payload, F_payload)
    return F, M
