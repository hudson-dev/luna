# Parafoil NMPC math paper

Conference manuscript deriving the mathematics of the `algo/mpc_v1` guided-parafoil simulation.

## Files

| File | Role |
|------|------|
| `parafoil_mpc_math.tex` | IEEE conference paper (self-contained bibliography) |
| `references.bib` | BibTeX companion (same entries) |

## Build

```bash
pdflatex parafoil_mpc_math.tex
pdflatex parafoil_mpc_math.tex
```

## Companion code

`../../algo/mpc_v1/` — `parafoil_dynamics.m`, `wind_model.m`, `mpc_parafoil.m`, `simulate_flight.m`, `rk4.m`, `run_parafoil_mpc.m`
