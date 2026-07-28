"""Fixed-step 4th-order Runge-Kutta integrator (from scratch)."""
from __future__ import annotations

from typing import Callable

import numpy as np


def rk4_step(f: Callable[[np.ndarray], np.ndarray], x: np.ndarray, dt: float) -> np.ndarray:
    """One RK4 step of x' = f(x)."""
    k1 = f(x)
    k2 = f(x + 0.5 * dt * k1)
    k3 = f(x + 0.5 * dt * k2)
    k4 = f(x + dt * k3)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
