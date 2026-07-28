"""LTV-QP receding-horizon MPC (built from scratch, solved with OSQP).

Once per control step:

1. Take the feasible reference trajectory (x_ref, u_ref) from guidance
   (a rollout of the reduced model from the current state).
2. Linearize the RK4-discretized reduced model along it:
       x_{k+1} ≈ A_k x_k + B_k u_k + c_k
3. Assemble one sparse QP over z = [x_1..x_N, u_0..u_{N-1}]:

   min  Σ_{k=1..N} (x_k - x_ref_k)' Q_k (x_k - x_ref_k)
      + Σ_{k=0..N-1} r_u (u_k - u_ref_k)^2 + r_du (u_k - u_{k-1})^2

   s.t. dynamics equalities,
        |u_k| <= da_limit - margin          (deflection limit + margin)
        |u_k - u_{k-1}| <= du_max            (deflection rate limit)

   with Q_k = diag(q_pos, q_pos, 0, q_psi, q_da) and a terminal position
   weight that grows as predicted touchdown altitude shrinks
   (Q_N ~ q_terminal * (1 + c_terminal / h_end)), reproducing the shrinking
   -horizon "place the touchdown point" behavior of the team's MATLAB MPC.

4. Solve with OSQP (warm-started), apply u_0, repeat next cycle. Robustness
   comes from this receding-horizon feedback, not from model accuracy.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

from ..config import MPCParams
from ..dynamics.reduced import ReducedModel, NX
from ..guidance.guidance import GuidanceOutput
from .qp_solver import solve_qp


@dataclass
class MPCSolution:
    u0: float
    cost: float
    status: str
    ok: bool
    x_pred: np.ndarray = field(default_factory=lambda: np.zeros((0, NX)))
    n_horizon: int = 0


class LTVMPC:
    def __init__(self, params: MPCParams, model: ReducedModel,
                 servo_rate_max: float = 2.0):
        self.p = params
        self.model = model
        self.servo_rate_max = servo_rate_max
        self._z_prev: np.ndarray | None = None

    def horizon_length(self, h: float) -> int:
        """Shrinking horizon: never plan (much) past predicted touchdown."""
        t_go = max(h, 1.0) / self.model.p.Vv
        return int(np.clip(np.ceil(t_go / self.p.Ts), self.p.N_min, self.p.N_max))

    def solve(self, x0: np.ndarray, guid: GuidanceOutput, wind_fn,
              u_prev: float) -> MPCSolution:
        p = self.p
        mdl = self.model
        x_ref, u_ref = guid.x_ref, guid.u_ref
        N = len(u_ref)
        nx, nu = NX, 1
        nz = N * (nx + nu)
        iu = N * nx                       # offset of u block in z

        # ---- 1. LTV matrices along the reference -------------------------
        A_k, B_k, c_k = [], [], []
        for k in range(N):
            A, B, c = mdl.discrete_jacobians(x_ref[k], float(u_ref[k]), p.Ts, wind_fn)
            A_k.append(A); B_k.append(B); c_k.append(c)

        # ---- 2. cost ------------------------------------------------------
        w = guid.weights
        h_end = max(float(x_ref[N, 2]), 8.0)
        qT = p.q_terminal * (1.0 + p.c_terminal / h_end)
        Q_diag = np.zeros((N, nx))
        for k in range(1, N + 1):
            Q_diag[k - 1] = [w.q_pos, w.q_pos, 0.0, w.q_psi, p.q_da]
        Q_diag[N - 1, 0] += qT
        Q_diag[N - 1, 1] += qT

        P_x = sp.diags(2.0 * Q_diag.flatten())
        q_x = (-2.0 * Q_diag * x_ref[1:]).flatten()

        # u block: r_u (u-u_ref)^2 + r_du (u_k - u_{k-1})^2, u_{-1} = u_prev
        H_u = np.zeros((N, N))
        q_u = -2.0 * p.r_u * u_ref.copy()
        for k in range(N):
            H_u[k, k] += 2.0 * p.r_u
        for k in range(N):
            H_u[k, k] += 2.0 * p.r_du
            if k >= 1:
                H_u[k - 1, k - 1] += 2.0 * p.r_du
                H_u[k, k - 1] -= 2.0 * p.r_du
                H_u[k - 1, k] -= 2.0 * p.r_du
        q_u[0] += -2.0 * p.r_du * u_prev

        P_qp = sp.block_diag([P_x, sp.csc_matrix(H_u)], format="csc")
        q_qp = np.concatenate([q_x, q_u])

        # ---- 3. constraints ------------------------------------------------
        # dynamics equalities: x_{k+1} - A_k x_k - B_k u_k = c_k  (x_0 known)
        rows, cols, vals = [], [], []
        eq_l = np.zeros(N * nx)
        for k in range(N):
            r0 = k * nx
            for i in range(nx):                       # +I on x_{k+1}
                rows.append(r0 + i); cols.append(k * nx + i); vals.append(1.0)
            if k >= 1:                                # -A_k on x_k
                for i in range(nx):
                    for j in range(nx):
                        a = A_k[k][i, j]
                        if a != 0.0:
                            rows.append(r0 + i); cols.append((k - 1) * nx + j); vals.append(-a)
            for i in range(nx):                       # -B_k on u_k
                b = B_k[k][i, 0]
                if b != 0.0:
                    rows.append(r0 + i); cols.append(iu + k); vals.append(-b)
            eq_l[r0:r0 + nx] = c_k[k] + (A_k[k] @ x0 if k == 0 else 0.0)
        A_eq = sp.coo_matrix((vals, (rows, cols)), shape=(N * nx, nz))

        # input bounds
        da_lim = p.da_limit - p.da_margin
        A_u = sp.hstack([sp.csc_matrix((N, N * nx)), sp.eye(N)])
        l_u = -da_lim * np.ones(N)
        u_u = da_lim * np.ones(N)

        # rate bounds: u_0 - u_prev, u_k - u_{k-1}
        du_max = self.model_du_max()
        D = sp.eye(N, format="lil") - sp.eye(N, k=-1, format="lil")
        A_du = sp.hstack([sp.csc_matrix((N, N * nx)), D])
        l_du = -du_max * np.ones(N)
        u_du = du_max * np.ones(N)
        l_du[0] += u_prev
        u_du[0] += u_prev

        A_qp = sp.vstack([A_eq, A_u, A_du], format="csc")
        l_qp = np.concatenate([eq_l, l_u, l_du])
        u_qp = np.concatenate([eq_l, u_u, u_du])

        # ---- 4. solve --------------------------------------------------------
        z_warm = self._warm_start(nz, N, nx, x_ref, u_ref)
        res = solve_qp(P_qp, q_qp, A_qp, l_qp, u_qp, z_warm)
        if not res.ok or res.z is None:
            return MPCSolution(float(np.clip(u_ref[0], -da_lim, da_lim)),
                               np.inf, res.status, False,
                               x_pred=x_ref[1:].copy(), n_horizon=N)

        z = res.z
        self._z_prev = z
        u0 = float(np.clip(z[iu], -da_lim, da_lim))
        x_pred = z[:N * nx].reshape(N, nx)
        return MPCSolution(u0, res.obj, res.status, True, x_pred=x_pred, n_horizon=N)

    def model_du_max(self) -> float:
        """Servo rate limit mapped to a per-step command rate, with margin."""
        return self.p.du_rate_frac * self.p.Ts * self.servo_rate_max

    def _warm_start(self, nz, N, nx, x_ref, u_ref):
        # reference trajectory itself is an excellent warm start
        return np.concatenate([x_ref[1:].flatten(), u_ref])
