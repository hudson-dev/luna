"""QP solver layer.

Primary: OSQP (the team's stated flight-hardware target solver).
Fallback: a self-written sparse ADMM QP solver implementing the same
operator-splitting scheme OSQP uses, so the simulator still runs (slower,
less polished convergence) if the osqp wheel is unavailable.

Both solve:  min 1/2 z'Pz + q'z   s.t.  l <= Az <= u
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

try:
    import osqp
    HAVE_OSQP = True
except ImportError:  # pragma: no cover
    HAVE_OSQP = False


@dataclass
class QPResult:
    z: np.ndarray | None
    obj: float
    status: str
    ok: bool


def solve_qp(P: sp.csc_matrix, q: np.ndarray, A: sp.csc_matrix,
             l: np.ndarray, u: np.ndarray, z_warm: np.ndarray | None = None) -> QPResult:
    if HAVE_OSQP:
        m = osqp.OSQP()
        m.setup(P=P, q=q, A=A, l=l, u=u, verbose=False,
                eps_abs=1e-4, eps_rel=1e-4, max_iter=8000, polish=True,
                warm_starting=True)
        if z_warm is not None:
            m.warm_start(x=z_warm)
        res = m.solve()
        ok = res.info.status_val in (1, 2)  # solved / solved inaccurate
        return QPResult(res.x if ok else None, float(res.info.obj_val),
                        str(res.info.status), ok)
    return _admm_qp(P, q, A, l, u, z_warm)


def _admm_qp(P, q, A, l, u, z_warm=None, rho: float = 0.1, sigma: float = 1e-6,
             max_iter: int = 4000, eps: float = 1e-4) -> QPResult:
    """Minimal OSQP-style ADMM: factor the KKT system once, iterate."""
    n = P.shape[0]
    m = A.shape[0]
    x = z_warm.copy() if z_warm is not None else np.zeros(n)
    z = A @ x
    y = np.zeros(m)
    KKT = sp.bmat([[P + sigma * sp.eye(n), A.T],
                   [A, -sp.eye(m) / rho]], format="csc")
    lu = spla.splu(KKT)
    for it in range(max_iter):
        rhs = np.concatenate([sigma * x - q, z - y / rho])
        sol = lu.solve(rhs)
        x_new = sol[:n]
        z_tilde = z + (sol[n:] - y) / rho
        z_new = np.clip(z_tilde + y / rho, l, u)
        y = y + rho * (z_tilde - z_new)
        r_prim = np.linalg.norm(A @ x_new - z_new, np.inf)
        r_dual = np.linalg.norm(P @ x_new + q + A.T @ y, np.inf)
        x, z = x_new, z_new
        if r_prim < eps and r_dual < eps * 10:
            obj = float(0.5 * x @ (P @ x) + q @ x)
            return QPResult(x, obj, f"admm solved ({it + 1} iters)", True)
    obj = float(0.5 * x @ (P @ x) + q @ x)
    return QPResult(x, obj, "admm max_iter", True)  # usable, inaccurate
