"""Guided parafoil recovery simulator: 6-DOF plant, LTV-QP MPC (OSQP),
phased guidance, wind estimation, and interactive 3D visualization."""

from .config import Scenario
from .sim.scenarios import SCENARIOS
from .sim.simulator import run_scenario

__all__ = ["Scenario", "SCENARIOS", "run_scenario"]
