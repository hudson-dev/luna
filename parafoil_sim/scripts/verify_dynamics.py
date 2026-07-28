"""Open-loop 6-DOF verification: trim glide, turn response, apparent mass.

Run:  .venv/bin/python scripts/verify_dynamics.py
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np

from parafoil_sim.config import VehicleParams, AtmosphereParams
from parafoil_sim.dynamics.apparent_mass import apparent_mass_matrices
from parafoil_sim.dynamics.sysid import identify_reduced_model, _run
from parafoil_sim.dynamics.sixdof import ParafoilPlant

veh = VehicleParams()
atmo = AtmosphereParams()

Ma, Ja = apparent_mass_matrices(veh, 1.225)
print(f"mass = {veh.mass:.2f} kg,  S = {veh.S:.2f} m^2,  AR = {veh.AR:.2f}")
print(f"apparent mass diag = {np.diag(Ma).round(3)} kg  (heave/mass = {Ma[2,2]/veh.mass:.2f})")
print(f"apparent inertia diag = {np.diag(Ja).round(4)} kg m^2")
print(f"rigid inertia diag = {np.diag(veh.inertia()).round(3)} kg m^2")
print(f"r_canopy = {veh.r_canopy.round(3)},  r_payload = {veh.r_payload.round(3)}")
print()

params = identify_reduced_model(veh, atmo, verbose=True)

# full-brake turn
plant = ParafoilPlant(veh, atmo)
plant.set_state((0.0, 0.0), 400.0, 0.0, params.trim_v_body.copy(), pitch=params.trim_pitch)
_run(plant, 0.0, 5.0)
t, log = _run(plant, 1.0, 30.0)
r_full = log["r"][-500:].mean()
vel = log["vel_ned"][-500:].mean(axis=0)
print(f"full brake: turn rate = {np.rad2deg(r_full):.1f} deg/s "
      f"({r_full:.3f} rad/s), Vh = {np.hypot(vel[0], vel[1]):.2f}, sink = {vel[2]:.2f} m/s")

# symmetric brake effect (flare authority)
plant = ParafoilPlant(veh, atmo)
plant.set_state((0.0, 0.0), 400.0, 0.0, params.trim_v_body.copy(), pitch=params.trim_pitch)
from parafoil_sim.environment.actuator import BrakeActuators
act = BrakeActuators(veh)
dt = 0.01
for k in range(int(20 / dt)):
    da, ds = act.step(dt, 0.0, 0.9)
    plant.step(dt, da, ds, lambda h: np.zeros(3))
vel = plant.vel_ned
print(f"ds=0.9 steady: Vh = {np.hypot(vel[0], vel[1]):.2f}, sink = {vel[2]:.2f} m/s")

ok = (5.0 < params.Vh < 11.0 and 2.0 < params.Vv < 6.0 and
      0.1 < params.K_turn < 1.5 and abs(np.rad2deg(r_full)) < 90.0)
print("\nDYNAMICS CHECK:", "PASS" if ok else "FAIL")
