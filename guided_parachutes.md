# Guided Parafoil Recovery System: Project Context

## Mission

Develop an autonomous guided parafoil recovery system for a rocketry team. The system must steer a deployed parafoil back to the launch pad under real-world wind disturbances. The project spans GNC algorithm development, mechanical and actuator design, and eventual deployment on flight hardware.

Success is defined on two axes:

1. **Theoretical rigor**: a simulation-validated MPC-based guidance and control stack.
2. **Practical trustworthiness**: a system that behaves predictably and fails gracefully on real flight hardware.

These are treated as distinct engineering goals requiring different design choices, and the project deliberately maintains two parallel tracks: a higher-fidelity primary approach and a simpler, more robust fallback.

## Technical Domains

- Model predictive control (MPC) and trajectory optimization
- Guidance law design (phased waypoint guidance)
- Wind estimation
- Kinematic and dynamic modeling of parafoil flight
- Servo and mechanical actuator design for the brake/control line system

## Current State

### Guidance and Control

- **MPC**: implemented in MATLAB using `fmincon` with SQP as the primary solver. Nonlinear receding-horizon formulation.
- **Fallback guidance**: a phased waypoint guidance law is being developed in parallel as the reliable flight-hardware option.
- **Wind estimation**: active in the system. A low-pass filter is applied to the difference between GPS ground velocity and predicted air velocity.

### Mechanical

- Designing servo shock-load protection for the control line actuator.
- Developing a locking spool mechanism for the brake/control line system.

### Theory Study

- Working through the MIT Underactuated Robotics course (underactuated.csail.mit.edu) selectively. The parafoil codebase serves as the applied lab work; coursework is mined for relevant theory rather than followed end to end.

## Roadmap

- **Solver migration**: move MPC from `fmincon`/SQP to an LTV-QP formulation solved with OSQP for flight hardware, driven by compute and reliability constraints.
- **Flare maneuver**: implement a rangefinder or barometric-triggered flare, chosen over model-based terminal prediction.
- **Successive convexification**: potentially explore as a middle ground between SQP and LTV-QP.
- **Model upgrades**: investigate shear-aware wind prediction and 6-DOF dynamics with apparent mass terms.
- **Case study**: review the perching-glider example in MIT Underactuated Chapter 10 as the closest analogue to the parafoil landing problem.

## Key Design Principles and Learnings

### Control

- MPC's practical robustness comes from receding-horizon feedback on measured state, not from model accuracy.
- Three identified MPC failure modes and their mitigations:
  1. **Persistent bias**: addressed by the wind estimator.
  2. **Constraint violations under disturbance**: requires constraint margin or tube MPC.
  3. **Model reliance at the terminal phase**: addressed by the sensor-triggered flare.
- **Solver tradeoff**: LTV-QP (linearize once per timestep along a reference trajectory, then solve a single QP) is the preferred embedded formulation. SQP (iterative linearize-solve-step inside `fmincon`) is more accurate but too compute-heavy for the flight computer.

### Mechanical Safety

- The servo lives only on the brake/control line branch, never in the primary riser load path. This is a core safety principle.
- Self-energizing pawl geometry is easy to get wrong (seating moment direction is subtle). The friction brake variant of the locking mechanism is the only one that fails gracefully under shock, by slipping rather than transmitting load into the servo.

## Tools and Resources

| Tool | Role |
|------|------|
| MATLAB | Primary development environment (`fmincon`, Optimization Toolbox, custom RK4 integrator) |
| OSQP | Target QP solver for flight hardware deployment |
| MIT Underactuated Robotics | Selective theoretical reference |

### Key MATLAB Files

- `run_parafoil_mpc.m`: top-level MPC run script
- `simulate_flight.m`: closed-loop flight simulation
- `mpc_parafoil.m`: MPC formulation and solver call
- `parafoil_dynamics.m`: vehicle dynamics model
- `wind_model.m`: wind disturbance model
- `rk4.m`: fixed-step RK4 integrator

## Working Style

- Concise, direct engineering communication focused on tradeoffs and judgment calls rather than exhaustive surveys.
- Structured comparisons (tables, prioritized lists) when evaluating design options.
- The active codebase is the primary learning lab; formal coursework supplements it selectively.
