"""International Standard Atmosphere (troposphere) density model."""
from __future__ import annotations

from ..config import AtmosphereParams

_LAPSE = 2.25577e-5   # combined lapse constant [1/m]
_EXP = 4.25588        # g / (R * L) - 1


def air_density(h_agl: float, params: AtmosphereParams) -> float:
    """Density [kg/m^3] at altitude above ground; ISA troposphere if enabled."""
    if not params.use_isa:
        return params.rho0
    h_msl = max(0.0, h_agl + params.h_ground_msl)
    return params.rho0 * (1.0 - _LAPSE * h_msl) ** _EXP
