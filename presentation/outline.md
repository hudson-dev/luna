# Luna — presentations

## Math deck (equations + project results)

```bash
python presentation/gen_math_assets.py      # plots, diagrams from mpc_v1 + parafoil_sim
python presentation/build_math_deck.py      # → Luna_Parafoil_NMPC_Math.pptx
```

- Math source: `papers/parafoil_mpc_math.pdf`, `algo/mpc_v1/`
- Video frames: `parafoil_sim/output/steady_wind_flight.mp4`
- 6-DOF results: `parafoil_sim/output/*_flight.json`
- Equation panels are editable text boxes (Consolas)

## GNC overview deck

```bash
python presentation/build_deck.py           # → Luna_MPC_Parafoil_Recovery.pptx
```
