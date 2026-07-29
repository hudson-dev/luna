"""Central configuration for the parafoil simulator.

All physical parameters, environment settings, GNC tuning knobs, and scenario
definitions live here as dataclasses so a scenario is a single serializable
object.

Frame conventions
-----------------
* Inertial frame: NED (x North, y East, z Down). Altitude h = -z.
* Body frame: FRD (x forward along canopy chord, y right, z down), origin at
  the system center of mass.
* Quaternion: scalar-first [w, x, y, z], rotates body vectors into NED.
* Wind direction convention: azimuth the wind blows TOWARD (rad, from North).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


# ----------------------------------------------------------------------------
# Vehicle
# ----------------------------------------------------------------------------
@dataclass
class VehicleParams:
    """Small ram-air parafoil recovering a ~5 kg rocket.

    Geometry / mass are sized so the trimmed glide is roughly 7-8 m/s
    horizontal and 3.5-4 m/s sink (glide ratio ~2), matching the team's
    MATLAB 3-DOF model.
    """

    # --- mass properties -----------------------------------------------------
    m_payload: float = 5.0        # rocket + avionics [kg]
    m_canopy: float = 0.6         # canopy + lines [kg]
    line_length: float = 1.5      # payload CG to canopy aero center [m]

    # --- canopy geometry (sized so trim airspeed ~8 m/s comfortably exceeds
    # the design wind envelope; wing loading ~3.7 kg/m^2) --------------------
    span: float = 2.0             # b [m]
    chord: float = 0.75           # c [m]
    thickness: float = 0.12       # inflated cell thickness t [m]
    arc_ratio: float = 0.10       # canopy arc height / span (anhedral arc)

    # --- payload drag body ---------------------------------------------------
    S_payload: float = 0.08       # payload reference area [m^2]
    CD_payload: float = 0.9

    # --- aerodynamic coefficients (canopy frame == body frame) ---------------
    CL0: float = 0.40
    CLa: float = 2.20             # per rad
    CLds: float = 0.30            # symmetric brake lift increment
    CD0: float = 0.22
    CDa2: float = 0.90            # induced-ish quadratic term, per rad^2
    CDds: float = 0.30            # symmetric brake drag increment
    CDda: float = 0.06            # asymmetric brake drag increment (|da|)
    CYb: float = -0.30            # side force per rad sideslip
    Clb: float = -0.06            # roll from sideslip
    Clp: float = -0.45            # roll damping
    Clda: float = 0.010           # roll from asymmetric brake (rolls toward the
                                  # braked side, same sense as the yaw moment)
    Cm0: float = 0.02
    Cma: float = -0.25            # canopy own pitch stiffness (pendulum dominates)
    Cmq: float = -1.80            # pitch damping
    Cnb: float = 0.05             # weathercock
    Cnr: float = -0.25            # yaw damping
    Cnda: float = 0.020           # yaw from asymmetric brake (drag differential)

    # --- actuator: brake-line servo rig ---------------------------------------
    # Two winch servos on the payload reel in the LEFT and RIGHT brake
    # (steering) lines only -- never the primary risers. Line travel maps
    # linearly to trailing-edge deflection:
    #     brake_norm (per side, 0..1) = line_travel / line_travel_max
    #     da = right_norm - left_norm      (asymmetric, aero yaw/roll input)
    #     ds = min(left_norm, right_norm)  (symmetric, aero lift/drag input)
    # spool angle = line_travel / spool_radius.
    servo_tau: float = 0.15         # servo+line first-order lag [s]
    servo_rate_max: float = 2.0     # line rate limit [normalized travel / s]
    line_travel_max: float = 0.25   # full brake line travel [m] (25 cm)
    spool_radius: float = 0.015     # winch spool radius [m] (1.5 cm)
    da_max: float = 1.0             # asymmetric brake, normalized [-1, 1]
    ds_max: float = 1.0             # symmetric brake, normalized [0, 1]

    @property
    def line_speed_max(self) -> float:
        """Line reel-in speed limit [m/s] implied by servo_rate_max."""
        return self.servo_rate_max * self.line_travel_max

    @property
    def spool_rate_max(self) -> float:
        """Spool angular rate limit [rad/s]."""
        return self.line_speed_max / self.spool_radius

    # --- apparent mass --------------------------------------------------------
    use_apparent_mass: bool = True

    # ------------------------------------------------------------------ derived
    @property
    def mass(self) -> float:
        return self.m_payload + self.m_canopy

    @property
    def S(self) -> float:
        return self.span * self.chord

    @property
    def AR(self) -> float:
        return self.span / self.chord

    @property
    def r_canopy(self) -> np.ndarray:
        """Canopy aero-center position relative to system CG, body FRD [m]."""
        z_cg = -self.m_canopy * self.line_length / self.mass  # CG below payload-line midpoint
        return np.array([0.0, 0.0, -(self.line_length + z_cg)])

    @property
    def r_payload(self) -> np.ndarray:
        """Payload CP position relative to system CG, body FRD [m]."""
        z_cg = -self.m_canopy * self.line_length / self.mass
        return np.array([0.0, 0.0, -z_cg])

    def inertia(self) -> np.ndarray:
        """Rigid-body inertia about system CG from point payload + flat-plate canopy."""
        J = np.zeros((3, 3))
        for m, r in ((self.m_payload, self.r_payload), (self.m_canopy, self.r_canopy)):
            J += m * (np.dot(r, r) * np.eye(3) - np.outer(r, r))
        # canopy own inertia (thin rectangular plate)
        b, c, mc = self.span, self.chord, self.m_canopy
        J += np.diag([mc * b**2 / 12.0, mc * c**2 / 12.0, mc * (b**2 + c**2) / 12.0])
        return J


# ----------------------------------------------------------------------------
# Environment
# ----------------------------------------------------------------------------
@dataclass
class DiscreteGust:
    """A single '1-cosine' discrete gust event (MIL-F-8785C style)."""
    t_start: float          # onset time [s]
    t_ramp: float           # ramp-up = ramp-down duration [s]
    t_hold: float           # hold duration at peak [s]
    magnitude: float        # peak gust speed [m/s]
    direction_deg: float    # azimuth gust blows TOWARD [deg from North]
    vertical: float = 0.0   # peak vertical (down +) component [m/s]


@dataclass
class WindParams:
    # steady / mean wind
    W_ref: float = 0.0            # mean speed at reference height [m/s]
    dir_deg: float = 0.0          # azimuth wind blows TOWARD [deg from North]
    h_ref: float = 10.0           # reference height for the profile [m]

    # shear profile: 'none' | 'power' | 'log'
    shear_model: str = "power"
    shear_exponent: float = 0.14  # power-law exponent
    z0: float = 0.05              # roughness length for log profile [m]

    # continuous turbulence (Dryden, MIL-F-8785C low-altitude)
    turbulence: bool = False
    turb_W20: float | None = None  # wind at 20 ft driving intensity; default = W_ref

    # discrete gusts
    gusts: list[DiscreteGust] = field(default_factory=list)


@dataclass
class AtmosphereParams:
    rho0: float = 1.225           # sea-level density [kg/m^3]
    use_isa: bool = True          # density varies with altitude (ISA troposphere)
    h_ground_msl: float = 0.0     # ground altitude above MSL [m]


@dataclass
class SensorParams:
    enabled: bool = False
    gps_rate: float = 5.0         # [Hz]
    gps_pos_sigma: float = 1.2    # horizontal position noise [m]
    gps_vel_sigma: float = 0.12   # velocity noise [m/s]
    baro_rate: float = 20.0       # [Hz]
    baro_sigma: float = 0.5       # altitude noise [m]
    heading_sigma_deg: float = 2.0  # AHRS yaw noise [deg]


# ----------------------------------------------------------------------------
# GNC
# ----------------------------------------------------------------------------
@dataclass
class EstimatorParams:
    tau_wind: float = 4.0         # wind estimator LPF time constant [s]


@dataclass
class GuidanceParams:
    h_approach: float = 110.0     # max altitude to consider final approach [m]
    h_blend: float = 30.0         # blend heading to upwind below this alt [m]
    approach_margin: float = 1.03 # energy margin factor on required time
    approach_pad_s: float = 4.0   # additive time margin [s]
    extend_reserve_s: float = 25.0  # max excess absorbable by the extend leg [s]
    h_flare: float = 6.0          # sensor-triggered flare altitude [m]
    flare_ds: float = 0.75        # symmetric brake at flare (was 0.9)
    flare_ramp: float = 0.7       # flare ramp time [s]
    loiter_radius: float = 45.0   # energy-management circle radius [m]
    loiter_capture: float = 140.0 # switch homing->loiter within this range [m]
    approach_speed_min: float = 2.5  # min assumed ground speed into wind [m/s]
    kp_heading: float = 0.8       # pursuit gain for reference generation [1/s]


@dataclass
class MPCParams:
    Ts: float = 1.5               # prediction step [s]
    N_max: int = 24               # max horizon length (-> 36 s lookahead)
    N_min: int = 4
    # weights (state = [pN, pE, h, psi, da])
    q_pos_run: float = 0.02       # running position tracking
    q_psi_run: float = 2.0        # running heading tracking (loiter/approach)
    q_psi_approach: float = 8.0   # heading weight on final approach
    q_da: float = 0.05            # penalize standing asymmetric deflection
    r_u: float = 12.0             # control effort
    r_du: float = 120.0           # control rate (smoothness / anti-chatter)
    q_terminal: float = 4.0       # terminal position weight (base)
    c_terminal: float = 250.0     # terminal growth ~ (1 + c/h_end)
    # constraints
    da_limit: float = 0.70        # asymmetric brake command ceiling (normalized);
                                  # 1.0 let guidance saturate near-full travel
    da_margin: float = 0.10       # constraint margin (robustness)
    bank_max_deg: float = 30.0    # bank envelope: unload turn-brake past ~0.9 of
                                  # this; hard stop at the limit
    du_rate_frac: float = 0.8     # fraction of servo rate allowed as command rate
    # prediction-model wind handling
    shear_aware: bool = True      # scale wind estimate with power-law vs altitude
    shear_exponent: float = 0.14
    wind_margin: float = 1.10     # inflate wind estimate for planning: biases
                                  # energy errors toward landing (slightly)
                                  # upwind of the pad instead of downwind


@dataclass
class SimParams:
    dt: float = 0.02              # plant integration step [s]
    t_control: float = 0.5        # MPC update period [s]
    t_max: float = 600.0          # hard stop [s]
    seed: int = 1


# ----------------------------------------------------------------------------
# Scenario
# ----------------------------------------------------------------------------
@dataclass
class Scenario:
    name: str = "scenario"
    description: str = ""
    # release state
    release_pos: tuple[float, float] = (900.0, -700.0)  # (N, E) offset from pad [m]
    release_alt: float = 600.0    # AGL [m]
    release_heading_deg: float = 90.0  # initial heading [deg from North]
    target: tuple[float, float] = (0.0, 0.0)            # launch pad (N, E) [m]

    vehicle: VehicleParams = field(default_factory=VehicleParams)
    wind: WindParams = field(default_factory=WindParams)
    atmosphere: AtmosphereParams = field(default_factory=AtmosphereParams)
    sensors: SensorParams = field(default_factory=SensorParams)
    estimator: EstimatorParams = field(default_factory=EstimatorParams)
    guidance: GuidanceParams = field(default_factory=GuidanceParams)
    mpc: MPCParams = field(default_factory=MPCParams)
    sim: SimParams = field(default_factory=SimParams)
