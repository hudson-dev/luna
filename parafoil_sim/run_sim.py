#!/usr/bin/env python
"""Top-level runner for the guided parafoil simulator.

Usage:
    python run_sim.py                        # run all scenarios + viz
    python run_sim.py --scenario steady_wind # one scenario
    python run_sim.py --list                 # list scenarios
    python run_sim.py --no-viz               # skip HTML generation
    python run_sim.py --seed 7               # override RNG seed
"""
from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from parafoil_sim.sim.scenarios import SCENARIOS
from parafoil_sim.sim.simulator import run_scenario


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", default="all",
                    help=f"one of {list(SCENARIOS)} or 'all'")
    ap.add_argument("--seed", type=int, default=None, help="override RNG seed")
    ap.add_argument("--no-viz", action="store_true", help="skip visualization")
    ap.add_argument("--out", default="output", help="output directory for HTML")
    ap.add_argument("--list", action="store_true", help="list scenarios and exit")
    args = ap.parse_args()

    if args.list:
        for name, fac in SCENARIOS.items():
            print(f"{name:18s} {fac().description}")
        return 0

    names = list(SCENARIOS) if args.scenario == "all" else [args.scenario]
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for name in names:
        if name not in SCENARIOS:
            print(f"unknown scenario '{name}'; use --list")
            return 1
        scn = SCENARIOS[name]()
        if args.seed is not None:
            scn.sim.seed = args.seed
        print(f"\n=== {name}: {scn.description}")
        res = run_scenario(scn, verbose=True)
        results.append(res)
        if not args.no_viz:
            from parafoil_sim.viz.plotly_viz import save_flight_html
            from parafoil_sim.viz.threejs_viewer import save_threejs_html
            path = save_flight_html(res, out_dir / f"{name}.html")
            print(f"  visualization       : {path}")
            path3d = save_threejs_html(res, out_dir / f"{name}_3d.html")
            print(f"  3D viewer           : {path3d}")

    print("\n=== landing summary ===")
    for r in results:
        print(f"  {r.scenario.name:18s} miss = {r.miss_distance:6.1f} m | "
              f"ground spd = {r.td_ground_speed:5.2f} m/s | "
              f"sink = {r.td_sink_rate:5.2f} m/s | flare = {r.flare_triggered}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
