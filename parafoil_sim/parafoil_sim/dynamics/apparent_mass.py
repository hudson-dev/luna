"""Apparent (added) mass and inertia for a ram-air canopy.

A parafoil canopy is very light relative to the air it displaces and
accelerates, so the air co-moving with the canopy contributes significant
effective mass and inertia. We use the classic Lissaman & Brown (1993)
estimates for an arched wing (as used by Slegers & Costello and most parafoil
GNC literature):

    Translational (canopy frame, x fwd / y right / z down):
        m_A = 0.666 rho (1 + 8/3 a*^2) t^2 b            (surge)
        m_B = 0.267 rho (t^2 + 2 a^2 (1 - t*^2)) c      (sway)
        m_C = 0.785 rho sqrt(AR / (1 + AR)) c^2 b       (heave, dominant)

    Rotational:
        I_A = 0.0555 rho (1 + 8 a*^2) t^2 b^3            (roll)
        I_B = 0.0308 rho (AR / (1 + AR)) c^4 b           (pitch)
        I_C = 0.0555 rho t^2 b^3                          (yaw)

where t is inflated thickness, c chord, b span, a arc height, a* = a/b,
t* = t/c, AR = b/c.

Simplification (documented): the apparent-mass tensor is applied as a diagonal
matrix about the system CG in body axes, i.e. we neglect the off-diagonal
CG-offset coupling terms of the full Barrows formulation. This captures the
dominant effect (heave/sway added mass of order the vehicle mass) while
keeping the equations of motion in standard momentum form.
"""
from __future__ import annotations

import numpy as np

from ..config import VehicleParams


def apparent_mass_matrices(veh: VehicleParams, rho: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (Ma, Ja): diagonal apparent mass and inertia matrices [SI]."""
    if not veh.use_apparent_mass:
        return np.zeros((3, 3)), np.zeros((3, 3))
    b, c, t = veh.span, veh.chord, veh.thickness
    a = veh.arc_ratio * b
    a_s = a / b
    t_s = t / c
    AR = veh.AR

    m_A = 0.666 * rho * (1.0 + (8.0 / 3.0) * a_s**2) * t**2 * b
    m_B = 0.267 * rho * (t**2 + 2.0 * a**2 * (1.0 - t_s**2)) * c
    m_C = 0.785 * rho * np.sqrt(AR / (1.0 + AR)) * c**2 * b

    I_A = 0.0555 * rho * (1.0 + 8.0 * a_s**2) * t**2 * b**3
    I_B = 0.0308 * rho * (AR / (1.0 + AR)) * c**4 * b
    I_C = 0.0555 * rho * t**2 * b**3

    return np.diag([m_A, m_B, m_C]), np.diag([I_A, I_B, I_C])
