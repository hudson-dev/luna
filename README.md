# Luna — Guided Parafoil Recovery

Autonomous guided parafoil recovery for a rocketry team: steer a deployed
parafoil back to the launch pad under wind disturbances.

## Contents

| Path | Description |
|------|-------------|
| [`guided_parachutes.md`](guided_parachutes.md) | Project context, design principles, roadmap |
| [`parafoil_sim/`](parafoil_sim/) | High-fidelity 6-DOF Python simulator + LTV-QP MPC (no MATLAB) |
| [`algo/mpc_v1/`](algo/mpc_v1/) | Legacy MATLAB MPC reference |
| [`rocket/`](rocket/) | Rocket / OpenRocket design assets |
| [`presentation/`](presentation/) | Editable MPC technical-review deck (`.pptx`) + generator script |

## Quick start (Python simulator)

```bash
cd parafoil_sim
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python run_sim.py --scenario steady_wind
```

Open the interactive 3D viewer: `parafoil_sim/output/steady_wind_3d.html`

See [`parafoil_sim/README.md`](parafoil_sim/README.md) for architecture, scenarios, and MPC details.
