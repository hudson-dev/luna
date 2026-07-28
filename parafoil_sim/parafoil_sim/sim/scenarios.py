"""Predefined scenarios.

All release the parafoil ~1.1-1.2 km from the pad at 600 m AGL. Wind
direction convention: azimuth the wind blows TOWARD (deg from North).
The release point is placed roughly upwind so the return is energetically
feasible (the vehicle trims at ~6.5 m/s airspeed).
"""
from __future__ import annotations

from ..config import (DiscreteGust, Scenario, SensorParams, WindParams)


def calm() -> Scenario:
    return Scenario(
        name="calm",
        description="No wind, ISA atmosphere, sensor noise on.",
        release_pos=(900.0, -700.0), release_alt=600.0, release_heading_deg=90.0,
        wind=WindParams(W_ref=0.0, turbulence=False),
        sensors=SensorParams(enabled=True),
    )


def steady_wind() -> Scenario:
    return Scenario(
        name="steady_wind",
        description="3.5 m/s steady wind (power-law shear), no turbulence.",
        release_pos=(900.0, -700.0), release_alt=600.0, release_heading_deg=90.0,
        wind=WindParams(W_ref=3.5, dir_deg=150.0, shear_model="power",
                        shear_exponent=0.14, turbulence=False),
        sensors=SensorParams(enabled=True),
    )


def shear_turbulence() -> Scenario:
    return Scenario(
        name="shear_turbulence",
        description="3 m/s wind with power-law shear + Dryden turbulence.",
        release_pos=(850.0, -750.0), release_alt=600.0, release_heading_deg=45.0,
        wind=WindParams(W_ref=3.0, dir_deg=170.0, shear_model="power",
                        shear_exponent=0.2, turbulence=True, turb_W20=5.0),
        sensors=SensorParams(enabled=True),
    )


def strong_gusts() -> Scenario:
    return Scenario(
        name="strong_gusts",
        description="4 m/s sheared wind, heavy Dryden turbulence, and two "
                    "discrete 4 m/s gust events.",
        release_pos=(800.0, -800.0), release_alt=650.0, release_heading_deg=0.0,
        wind=WindParams(W_ref=4.0, dir_deg=160.0, shear_model="power",
                        shear_exponent=0.14, turbulence=True, turb_W20=8.0,
                        gusts=[
                            DiscreteGust(t_start=60.0, t_ramp=3.0, t_hold=6.0,
                                         magnitude=4.0, direction_deg=250.0),
                            DiscreteGust(t_start=170.0, t_ramp=2.0, t_hold=4.0,
                                         magnitude=4.0, direction_deg=100.0,
                                         vertical=1.0),
                        ]),
        sensors=SensorParams(enabled=True),
    )


SCENARIOS = {s().name: s for s in (calm, steady_wind, shear_turbulence, strong_gusts)}
