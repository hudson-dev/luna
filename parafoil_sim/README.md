# Guided Parafoil Recovery Simulator

High-fidelity 3D simulation of an autonomous guided parafoil returning a small
rocket to its launch pad, with a from-scratch **LTV-QP model-predictive
controller solved with OSQP** — the flight-hardware formulation from the
project roadmap (no MATLAB, no off-the-shelf MPC library).

```
.venv/bin/python run_sim.py                  # all four scenarios + Plotly + Three.js
.venv/bin/python run_sim.py --scenario steady_wind
.venv/bin/python run_sim.py --list
.venv/bin/python scripts/verify_dynamics.py  # open-loop 6-DOF checks
open output/steady_wind_3d.html              # interactive Three.js viewer
```

Outputs land in `output/`:

* **`output/<scenario>_3d.html`** — interactive **Three.js / WebGL** viewer
  (open in a browser; needs network once for the Three.js CDN). Features a
  high-fidelity ram-air canopy (arched multi-cell with thickness/camber,
  visible ribs, left/right trailing-edge brake deflection), A/B/C suspension
  cascade, risers, payload, animated servo spools on the brake lines,
  phase-colored trajectory ribbon, pad target, optional wind vectors, and
  playback controls (play/pause, scrubber, speed) plus camera presets
  (free orbit / follow vehicle / rig close-up). Sibling
  `output/<scenario>_flight.json` is the downsampled flight log the viewer
  consumes.
* **`output/<scenario>.html`** — Plotly dashboard (glyph flight view, 1:1
  rig close-up, time-series): altitude/phase, speeds, brake deflection with
  per-side servo line travel [cm], MPC cost, wind estimate vs truth,
  distance to pad, heading.

```
open output/steady_wind_3d.html    # Three.js viewer (preferred 3D)
open output/steady_wind.html       # Plotly dashboard
```

## Setup

```
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

If the `osqp` wheel is unavailable the MPC automatically falls back to a
self-written sparse ADMM QP solver (same operator-splitting scheme OSQP uses).

## Package layout

```
parafoil_sim/
  config.py           all dataclasses: vehicle, wind, sensors, guidance, MPC, scenario
  dynamics/
    rk4.py            fixed-step RK4 integrator (from scratch)
    quat.py           quaternion utilities (scalar-first, body->NED)
    aero.py           canopy aero coefficient buildup + payload drag
    apparent_mass.py  Lissaman & Brown apparent mass/inertia
    sixdof.py         6-DOF quaternion rigid-body plant
    reduced.py        4-DOF kinematic MPC prediction model + LTV Jacobians
    sysid.py          calibrates the reduced model against the 6-DOF plant
  environment/
    atmosphere.py     ISA density vs altitude
    wind.py           mean profile (power/log shear) + Dryden turbulence + 1-cos gusts
    sensors.py        GPS / baro / AHRS models with rates and noise
    actuator.py       brake-line servo rig: two winch servos, physical line travel
  estimation/
    wind_estimator.py LPF of (GPS ground velocity - predicted air velocity)
    glide_estimator.py online (Vh, Vv) polar estimate
  guidance/
    guidance.py       phase logic + reference trajectory generation
  control/
    ltv_mpc.py        LTV-QP construction (sparse, built by hand)
    qp_solver.py      OSQP wrapper + fallback ADMM solver
  sim/
    simulator.py      closed-loop simulation
    scenarios.py      calm / steady_wind / shear_turbulence / strong_gusts
  viz/
    rig_geometry.py   parametric canopy/lines/payload/servo geometry
    plotly_viz.py     animated 3D flight + rig close-up + dashboard -> HTML
    flight_log.py     downsampled JSON flight-log export for the WebGL viewer
    threejs_viewer.py Three.js / WebGL viewer HTML generator
    viewer.js         client-side parafoil mesh + playback (embedded in HTML)
run_sim.py            CLI entry point
scripts/verify_dynamics.py  open-loop trim/turn/flare verification
```

## Models

### 6-DOF plant (`dynamics/sixdof.py`)

State: NED position, attitude quaternion, body velocity, body rates.
Momentum-form equations with apparent mass (the canopy is light relative to
the air it entrains — heave apparent mass is ~16% of vehicle mass here):

```
(mI + Ma) v̇ = F_aero + Rᵀ m g e₃ − ω×((mI + Ma) v)
(J + Ja) ω̇ = M_aero − ω×((J + Ja) ω)
```

`Ma`, `Ja` use the Lissaman & Brown (1993) estimates for an arched wing,
applied as diagonal matrices about the CG (the Barrows off-diagonal
CG-coupling terms are neglected; documented in `apparent_mass.py`).

Aerodynamics: linear coefficient buildup (Slegers & Costello style) — lift,
drag, side force, and roll/pitch/yaw moments with rate damping; control
derivatives for asymmetric brake `da` (Cn_da yaw + Cl_da same-sense roll +
drag) and symmetric brake `ds` (CL/CD increments for the flare). Pendulum
stability comes from the canopy aero center riding ~1.3 m above the CG. The
payload contributes pure drag at its own station. Wind (with shear) is
evaluated at the instantaneous altitude inside each RK4 stage.

Default vehicle: 5.6 kg total, 1.5 m² canopy → trim glide 7.4 m/s horizontal,
2.5 m/s sink (L/D ≈ 3), full-brake turn rate ≈ 38°/s. All parameters live in
`VehicleParams` in `config.py`.

### Servo / brake-line rig (`environment/actuator.py`)

The actuation chain is modeled physically: two winch servos on the payload
reel in the **left and right brake (steering) lines only — never the primary
risers** (core safety principle). GNC commands mix into per-side line travel:

```
left  = clip(ds − da, 0, 1) · line_travel_max     (25 cm full travel)
right = clip(ds + da, 0, 1) · line_travel_max
spool angle = line_travel / spool_radius          (1.5 cm spool)
```

Each side has first-order servo lag (0.15 s) and a line-speed limit
(0.5 m/s); the aero inputs are recovered from the actual line states
(`da = (right − left)/max`, `ds = min(left, right)/max`). With both sides
unsaturated this is dynamically identical to the original normalized da/ds
channels (verified: identical sysid results), so the physical units are free
realism. Line travel per side is logged and plotted in cm.

### Reduced prediction model (`dynamics/reduced.py`)

The MPC predicts with a 4-DOF kinematic model (the standard practice, and the
same structure as the team's MATLAB model):

```
ṗN = Vh cos ψ + wN(h)      ψ̇  = K_turn · da
ṗE = Vh sin ψ + wE(h)      ḋa = (u − da)/τ_turn
ḣ  = −Vv
```

Its parameters are **identified from the 6-DOF plant at startup**
(`sysid.py`: trim run → Vh, Vv; brake-step run → K_turn, τ_turn), and
(Vh, Vv) are refined online by the glide estimator. Wind in prediction is the
estimated wind rescaled with the power-law shear profile (shear-aware), with
a 1.10 planning margin that biases energy errors toward the upwind side.

### Environment (`environment/`)

All toggleable per scenario: steady wind; power-law or logarithmic shear;
MIL-F-8785C low-altitude **Dryden turbulence** (first-order filters for the
horizontal components, the proper second-order filter for vertical, with the
output gain derived from the exact stationary Lyapunov solution so the
intensity is correct by construction); discrete 1-cosine gust events; ISA
density; GPS/baro/AHRS noise with realistic sample rates; servo lag + rate
limits.

## GNC stack

**Wind estimator** — low-pass filter on (GPS ground velocity − predicted air
velocity), exactly the flight approach; it kills the persistent-bias failure
mode. **Glide estimator** — LPFs of measured air-relative speed and sink rate
keep the prediction polar honest under turbulence and turn-induced sink.

**Phased guidance** (`guidance/guidance.py`):
`HOMING → LOITER → EXTEND → APPROACH → FLARE`

* HOMING: pursuit toward the loiter point.
* LOITER: circle the final-approach entry point (energy management).
* EXTEND: downwind leg along the approach corridor; the turn-back fires when
  remaining descent time matches the wind-drift-corrected time to the pad —
  this removes the energy quantization of committing straight off the circle.
* APPROACH: descend along a line **into the estimated wind**, pursuing an
  energy-matched glide-slope point; heading blends to upwind near the ground.
* FLARE: at 6 m (sensor-triggered by design, not model-predicted), symmetric
  brakes ramp to 0.9 to bleed speed.

Each cycle guidance also rolls the reduced model forward under these pursuit
laws to produce a *feasible* reference trajectory for the MPC.

**LTV-QP MPC** (`control/ltv_mpc.py`) — once per 0.5 s control step:

1. linearize the RK4-discretized reduced model along the guidance reference
   (finite-difference Jacobians): `x_{k+1} ≈ A_k x_k + B_k u_k + c_k`;
2. build one sparse QP over `[x_1..x_N, u_0..u_{N-1}]`: reference tracking on
   position/heading, control effort, control-rate smoothing, and a terminal
   position weight that grows as predicted touchdown altitude shrinks
   (`~ 1 + 250/h_end`) — the shrinking horizon (`N = t_go/Ts`, max 24 × 1.5 s)
   literally places the touchdown point late in the flight;
3. constraints: brake deflection limits **with 0.1 margin**, and
   deflection-rate limits (80% of servo rate);
4. solve with OSQP (warm-started with the reference), apply `u₀`.

Robustness comes from receding-horizon feedback, not model accuracy: the QP is
re-anchored to the measured state every cycle, and the three known failure
modes are each addressed (bias → wind estimator; constraint violation under
disturbance → deflection/rate margins; terminal model reliance → sensor
-triggered flare).

## Scenarios and verified results

Release: ~1.1–1.2 km from the pad at 600–650 m AGL. Miss distance at
touchdown, default seed and across seeds 1–5 (turbulence/noise realizations):

| scenario | wind | miss (seed 1) | median (5 seeds) | max |
|---|---|---|---|---|
| `calm` | none | 20.2 m | 23.0 m | 24.7 m |
| `steady_wind` | 3.5 m/s @ 10 m, power shear | 12.6 m | 14.3 m | 19.5 m |
| `shear_turbulence` | 3 m/s, strong shear + Dryden | 17.0 m | 26.2 m | 46.4 m |
| `strong_gusts` | 4 m/s shear + heavy Dryden + two 4 m/s discrete gusts | 22.9 m | 26.9 m | 44.3 m |

(The 5-seed statistics were measured just before the physical servo-rig
refactor; the refactor is dynamically equivalent and seed-1 results moved
only at noise level.) In the two stochastic scenarios the wind aloft reaches
5.5–7 m/s against a 7.4 m/s trim airspeed (~90% wind saturation); the outlier
seeds are wind-drift-limited on final rather than controller-limited. Each run prints a
landing summary (miss distance, touchdown ground speed/sink, flare status,
MPC solve statistics). Run `--seed N` to vary the noise realization.
