"""Wind field: mean profile with shear + Dryden turbulence + discrete gusts.

The total wind (NED, m/s) is

    w(t, h) = w_mean(h) + w_turb(t) + w_gust(t)

* Mean profile: uniform, power-law, or logarithmic shear vs altitude.
* Turbulence: MIL-F-8785C low-altitude Dryden model implemented as discrete
  shaping filters driven by white noise (from scratch, no toolbox).
  Longitudinal/lateral components use the first-order Dryden filter; the
  vertical component uses the proper second-order filter.
* Discrete gusts: '1-cosine' ramp-hold-ramp events.

The turbulence filters are advanced at the plant rate by `step(dt, ...)`;
`wind_at(h)` then evaluates the full field at any altitude within the step
(the frozen-turbulence value is uniform in space over one step, which is fine
for a single vehicle).
"""
from __future__ import annotations

import numpy as np

from ..config import WindParams

_FT = 0.3048


class WindField:
    def __init__(self, params: WindParams, rng: np.random.Generator):
        self.p = params
        self.rng = rng
        d = np.deg2rad(params.dir_deg)
        self._dir = np.array([np.cos(d), np.sin(d)])          # unit, blows toward
        # Dryden filter states
        self._u = 0.0                                          # along-wind
        self._v = 0.0                                          # cross-wind
        self._w = np.zeros(2)                                  # vertical, 2nd order
        self._turb = np.zeros(3)                               # NED turbulence output
        self._t = 0.0

    # ------------------------------------------------------------------ mean
    def mean_wind(self, h: float) -> np.ndarray:
        """Mean wind (NED) at altitude h [m AGL]."""
        p = self.p
        if p.W_ref <= 0.0:
            return np.zeros(3)
        h_eff = max(h, 2.0)
        if p.shear_model == "power":
            mag = p.W_ref * (h_eff / p.h_ref) ** p.shear_exponent
        elif p.shear_model == "log":
            mag = p.W_ref * np.log(h_eff / p.z0) / np.log(p.h_ref / p.z0)
        else:
            mag = p.W_ref
        return np.array([mag * self._dir[0], mag * self._dir[1], 0.0])

    # ----------------------------------------------------------- turbulence
    def _dryden_params(self, h: float):
        """Length scales [m] and intensities [m/s], MIL-F-8785C low altitude."""
        h_ft = np.clip(h / _FT, 10.0, 1000.0)
        Lw = h_ft * _FT
        Lu = h_ft / (0.177 + 0.000823 * h_ft) ** 1.2 * _FT
        W20 = self.p.turb_W20 if self.p.turb_W20 is not None else self.p.W_ref
        sig_w = 0.1 * max(W20, 0.5)
        sig_u = sig_w / (0.177 + 0.000823 * h_ft) ** 0.4
        return Lu, Lw, sig_u, sig_w

    def step(self, dt: float, h: float, V: float) -> None:
        """Advance turbulence filters by dt at altitude h and airspeed V."""
        self._t += dt
        if not self.p.turbulence:
            return
        V = max(V, 2.0)
        Lu, Lw, sig_u, sig_w = self._dryden_params(h)
        rng = self.rng

        # longitudinal & lateral: first-order Dryden (OU) filters
        for name, L, sig in (("_u", Lu, sig_u), ("_v", Lu, sig_u)):
            tau = L / V
            a = np.exp(-dt / tau)
            x = getattr(self, name)
            x = a * x + sig * np.sqrt(1.0 - a * a) * rng.standard_normal()
            setattr(self, name, x)

        # vertical: second-order Dryden filter
        #   H(s) ~ (1 + sqrt(3) tau s) / (1 + tau s)^2,  tau = Lw/V
        # States: x1'' form (companion). With unit-intensity white noise the
        # stationary covariance is Var(x1) = tau^3/4, Var(x2) = tau/4 (Lyapunov
        # closed form), so the output gain below yields Var(wz) = sig_w^2 exactly
        # while keeping the Dryden spectral shape.
        tau = Lw / V
        x1, x2 = self._w
        eta = rng.standard_normal() / np.sqrt(dt)
        dx1 = x2
        dx2 = -x1 / tau**2 - 2.0 * x2 / tau + eta
        self._w = np.array([x1 + dt * dx1, x2 + dt * dx2])
        c = sig_w / tau**1.5
        wz = float(c * (self._w[0] + np.sqrt(3.0) * tau * self._w[1]))
        wz = float(np.clip(wz, -3.5 * sig_w, 3.5 * sig_w))

        # rotate (u along mean wind, v across) into NED
        cu, su = self._dir
        self._turb = np.array([self._u * cu - self._v * su,
                               self._u * su + self._v * cu,
                               -wz])

    # ---------------------------------------------------------------- gusts
    def _gust_wind(self) -> np.ndarray:
        w = np.zeros(3)
        for g in self.p.gusts:
            s = self._t - g.t_start
            T = 2 * g.t_ramp + g.t_hold
            if s <= 0.0 or s >= T:
                continue
            if s < g.t_ramp:
                f = 0.5 * (1.0 - np.cos(np.pi * s / g.t_ramp))
            elif s < g.t_ramp + g.t_hold:
                f = 1.0
            else:
                f = 0.5 * (1.0 - np.cos(np.pi * (T - s) / g.t_ramp))
            d = np.deg2rad(g.direction_deg)
            w += f * np.array([g.magnitude * np.cos(d),
                               g.magnitude * np.sin(d),
                               g.vertical])
        return w

    # ---------------------------------------------------------------- total
    def wind_at(self, h: float) -> np.ndarray:
        """Total wind vector (NED, m/s) at altitude h for the current time."""
        return self.mean_wind(h) + self._turb + self._gust_wind()
