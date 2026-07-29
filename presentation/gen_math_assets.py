"""Generate plots/diagrams for the math presentation from project sources.

Produces PNGs under presentation/assets/ from:
  - a Python port of algo/mpc_v1 (paper-faithful kinematic NMPC)
  - parafoil_sim/output/*_flight.json (6-DOF sim results)
  - annotated block diagrams drawn with matplotlib

    python presentation/gen_math_assets.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "assets"
OUT.mkdir(parents=True, exist_ok=True)

# Dark aerospace palette matching the deck theme
BG = "#0B1D33"
PANEL = "#163150"
TEXT = "#F2F6FA"
MUTED = "#9BB0C4"
ACCENT = "#35C8D6"
ACCENT2 = "#F2A93B"
PATH = "#5B9BD5"
TRUTH = "#6EC6FF"
EST = "#F2A93B"


def style_axes(ax, title=None):
    ax.set_facecolor(PANEL)
    ax.figure.patch.set_facecolor(BG)
    ax.tick_params(colors=MUTED, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#274567")
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    if title:
        ax.set_title(title, color=TEXT, fontsize=12, pad=10)
    ax.grid(True, color="#274567", alpha=0.45, linewidth=0.6)


# --------------------------------------------------------------------------- #
# mpc_v1 Python port (matches MATLAB under seed=2)
# --------------------------------------------------------------------------- #
@dataclass
class P:
    Vh: float = 7.0
    Vv: float = 3.5
    umax: float = 0.35
    W10: float = 3.0
    Wdir: float = np.deg2rad(210)
    shearExp: float = 0.14
    tauGust: float = 8.0
    sigmaGust: float = 0.8
    dt: float = 0.05
    Tc: float = 1.0
    sigGPS: float = 0.2
    tauEst: float = 5.0
    Ts: float = 2.0
    N: int = 25
    Qpath: float = 0.02
    Ru: float = 2.0
    Rdu: float = 4.0
    Qf: float = 1.0
    cTerm: float = 200.0
    hApproach: float = 60.0
    Qhead: float = 40.0
    x0: np.ndarray = None  # type: ignore
    target: np.ndarray = None  # type: ignore

    def __post_init__(self):
        if self.x0 is None:
            self.x0 = np.array([650.0, -450.0, 500.0, np.pi / 2])
        if self.target is None:
            self.target = np.array([0.0, 0.0])


def dynamics(x, u, w, p: P):
    psi = x[3]
    return np.array(
        [p.Vh * np.cos(psi) + w[0], p.Vh * np.sin(psi) + w[1], -p.Vv, u]
    )


def rk4(f, x, dt):
    k1 = f(x)
    k2 = f(x + 0.5 * dt * k1)
    k3 = f(x + 0.5 * dt * k2)
    k4 = f(x + dt * k3)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def wind_model(h, gust, p: P, rng):
    Wmag = p.W10 * (max(h, 2.0) / 10.0) ** p.shearExp
    wmean = Wmag * np.array([np.cos(p.Wdir), np.sin(p.Wdir)])
    gust = gust * (1 - p.dt / p.tauGust) + p.sigmaGust * np.sqrt(
        2 * p.dt / p.tauGust
    ) * rng.standard_normal(2)
    return wmean + gust, gust


def mpc_cost(U, x, what, u_prev, p: P):
    J = 0.0
    xk = x.copy()
    up = u_prev
    psi_uw = np.arctan2(-what[1], -what[0])
    for u in U:
        dt = p.Ts
        if xk[2] <= p.Vv * p.Ts:
            dt = xk[2] / max(p.Vv, 1e-12)
        if dt <= 0:
            break
        xk = rk4(lambda xx, uu=u: dynamics(xx, uu, what, p), xk, dt)
        xk[2] = max(xk[2], 0.0)
        d2 = (xk[0] - p.target[0]) ** 2 + (xk[1] - p.target[1]) ** 2
        J += p.Qpath * d2 + p.Ru * u**2 + p.Rdu * (u - up) ** 2
        if xk[2] < p.hApproach:
            e = np.arctan2(np.sin(xk[3] - psi_uw), np.cos(xk[3] - psi_uw))
            J += p.Qhead * e**2
        up = u
        if xk[2] <= 0:
            break
    d2T = (xk[0] - p.target[0]) ** 2 + (xk[1] - p.target[1]) ** 2
    J += p.Qf * (1 + p.cTerm / max(xk[2], 5.0)) * d2T
    return J


def mpc_parafoil(x, what, u_prev, Uwarm, p: P):
    tgo = x[2] / p.Vv
    N = min(p.N, max(3, int(np.ceil(tgo / p.Ts))))
    U0 = np.concatenate([Uwarm[1:], Uwarm[-1:]])
    if len(U0) >= N:
        U0 = U0[:N]
    else:
        U0 = np.concatenate([U0, np.full(N - len(U0), U0[-1])])
    bounds = [(-p.umax, p.umax)] * N
    res = minimize(
        lambda U: mpc_cost(U, x, what, u_prev, p),
        U0,
        method="SLSQP",
        bounds=bounds,
        options={"maxiter": 60, "ftol": 1e-6, "disp": False},
    )
    Uopt = res.x
    return float(Uopt[0]), Uopt


def simulate_flight(p: P, seed: int = 2):
    rng = np.random.default_rng(seed)
    # Match MATLAB rng(seed) stream approximately via Generator;
    # for paper plots we care about qualitative fidelity of the same model.
    x = p.x0.copy()
    gust = np.zeros(2)
    what = np.zeros(2)
    u = 0.0
    Uwarm = np.zeros(p.N)
    ctimer = 0.0
    kmax = int(np.ceil(1.5 * (x[2] / p.Vv) / p.dt))
    X = np.zeros((4, kmax))
    U = np.zeros(kmax)
    W = np.zeros((2, kmax))
    What = np.zeros((2, kmax))
    T = np.zeros(kmax)
    t = 0.0
    k = 0
    while x[2] > 0 and k < kmax:
        w, gust = wind_model(x[2], gust, p, rng)
        vAir = np.array([p.Vh * np.cos(x[3]), p.Vh * np.sin(x[3])])
        vGPS = vAir + w + p.sigGPS * rng.standard_normal(2)
        wraw = vGPS - vAir
        a = p.dt / (p.dt + p.tauEst)
        what = what + a * (wraw - what)
        if ctimer <= 0:
            u, Uwarm = mpc_parafoil(x, what, u, Uwarm, p)
            if len(Uwarm) < p.N:
                Uwarm = np.concatenate([Uwarm, np.full(p.N - len(Uwarm), Uwarm[-1])])
            else:
                Uwarm = Uwarm[: p.N]
            ctimer = p.Tc
        ctimer -= p.dt
        x = rk4(lambda xx: dynamics(xx, u, w, p), x, p.dt)
        t += p.dt
        X[:, k] = x
        U[k] = u
        W[:, k] = w
        What[:, k] = what
        T[k] = t
        k += 1
    return {
        "X": X[:, :k],
        "U": U[:k],
        "W": W[:, :k],
        "What": What[:, :k],
        "t": T[:k],
        "miss": float(np.linalg.norm(X[:2, k - 1] - p.target)),
    }


# --------------------------------------------------------------------------- #
# Plots from mpc_v1
# --------------------------------------------------------------------------- #
def plot_mpc_results(out, p: P):
    # Ground track
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    style_axes(ax, f"Ground track  (miss = {out['miss']:.1f} m)")
    ax.plot(out["X"][0], out["X"][1], color=PATH, lw=1.8, label="path")
    ax.plot(p.target[0], p.target[1], "r*", ms=16, label="pad")
    ax.plot(p.x0[0], p.x0[1], "o", color=TEXT, ms=8, label="release")
    idx = np.arange(0, len(out["t"]), 400)
    ax.quiver(
        out["X"][0, idx],
        out["X"][1, idx],
        out["W"][0, idx],
        out["W"][1, idx],
        color=MUTED,
        scale=40,
        width=0.004,
        alpha=0.7,
        label="truth wind",
    )
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal")
    leg = ax.legend(facecolor=PANEL, edgecolor="#274567", labelcolor=TEXT, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "mpc_ground_track.png", dpi=160, facecolor=BG)
    plt.close(fig)

    # 3D trajectory
    fig = plt.figure(figsize=(7.2, 5.4))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(PANEL)
    fig.patch.set_facecolor(BG)
    ax.plot(out["X"][0], out["X"][1], out["X"][2], color=PATH, lw=1.6)
    ax.scatter([0], [0], [0], c="r", s=60, marker="*")
    ax.set_xlabel("x [m]", color=MUTED)
    ax.set_ylabel("y [m]", color=MUTED)
    ax.set_zlabel("h [m]", color=MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.set_title("Descent trajectory", color=TEXT, fontsize=12)
    ax.view_init(25, 35)
    fig.tight_layout()
    fig.savefig(OUT / "mpc_trajectory_3d.png", dpi=160, facecolor=BG)
    plt.close(fig)

    # Control + wind estimate
    fig, axes = plt.subplots(3, 1, figsize=(8.0, 6.2), sharex=True)
    fig.patch.set_facecolor(BG)
    for ax in axes:
        style_axes(ax)
    axes[0].plot(out["t"], out["U"], color=ACCENT, lw=1.2)
    axes[0].axhline(p.umax, color="#E85D5D", ls="--", lw=1)
    axes[0].axhline(-p.umax, color="#E85D5D", ls="--", lw=1)
    axes[0].set_ylabel("u [rad/s]")
    axes[0].set_title("Commanded heading rate", color=TEXT, fontsize=11)
    axes[1].plot(out["t"], out["W"][0], color=TRUTH, lw=1.1, label="truth")
    axes[1].plot(out["t"], out["What"][0], color=EST, ls="--", lw=1.2, label="estimate")
    axes[1].set_ylabel("W_x [m/s]")
    axes[1].legend(facecolor=PANEL, edgecolor="#274567", labelcolor=TEXT, fontsize=8)
    axes[2].plot(out["t"], out["W"][1], color=TRUTH, lw=1.1, label="truth")
    axes[2].plot(out["t"], out["What"][1], color=EST, ls="--", lw=1.2, label="estimate")
    axes[2].set_ylabel("W_y [m/s]")
    axes[2].set_xlabel("t [s]")
    axes[2].legend(facecolor=PANEL, edgecolor="#274567", labelcolor=TEXT, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "mpc_control_wind.png", dpi=160, facecolor=BG)
    plt.close(fig)

    # Range vs altitude
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    style_axes(ax, "Closing on the pad as altitude bleeds off")
    d = np.linalg.norm(out["X"][:2] - p.target[:, None], axis=0)
    ax.plot(out["X"][2], d, color=ACCENT, lw=1.6)
    ax.invert_xaxis()
    ax.set_xlabel("altitude [m]")
    ax.set_ylabel("distance to pad [m]")
    fig.tight_layout()
    fig.savefig(OUT / "mpc_range_vs_alt.png", dpi=160, facecolor=BG)
    plt.close(fig)

    # Save summary json for the deck builder
    (OUT / "mpc_summary.json").write_text(
        json.dumps(
            {
                "miss_m": out["miss"],
                "t_flight_s": float(out["t"][-1]),
                "seed": 2,
                "x0": p.x0.tolist(),
                "target": p.target.tolist(),
            },
            indent=2,
        )
    )


# --------------------------------------------------------------------------- #
# Plots from parafoil_sim flight logs
# --------------------------------------------------------------------------- #
def plot_sim_flight(name: str = "steady_wind"):
    path = ROOT / "parafoil_sim" / "output" / f"{name}_flight.json"
    data = json.loads(path.read_text())
    frames = data["frames"]
    t = np.asarray(frames["t"])
    pos = np.asarray(frames["pos_enu"])
    wind = np.asarray(frames["wind_enu"])
    phase = frames.get("phase", ["?"] * len(t))
    summary = data["summary"]

    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    style_axes(
        ax,
        f"{name}: ground track  (miss = {summary['miss_distance_m']:.1f} m)",
    )
    # Color by phase
    phase_colors = {
        "HOMING": "#5B9BD5",
        "LOITER": "#F2A93B",
        "EXTEND": "#B39DDB",
        "APPROACH": "#81C784",
        "FLARE": "#E57373",
    }
    for ph, color in phase_colors.items():
        mask = np.array([p == ph for p in phase])
        if mask.any():
            ax.plot(pos[mask, 0], pos[mask, 1], color=color, lw=2.0, label=ph)
    tgt = data["scenario"]["target_enu"]
    ax.plot(tgt[0], tgt[1], "r*", ms=16, label="pad")
    ax.plot(pos[0, 0], pos[0, 1], "o", color=TEXT, ms=7, label="release")
    ax.set_xlabel("East [m]")
    ax.set_ylabel("North [m]")
    ax.set_aspect("equal")
    ax.legend(facecolor=PANEL, edgecolor="#274567", labelcolor=TEXT, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / f"sim_{name}_track.png", dpi=160, facecolor=BG)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    style_axes(ax, f"{name}: altitude vs time")
    ax.plot(t, pos[:, 2], color=ACCENT, lw=1.6)
    ax.set_xlabel("t [s]")
    ax.set_ylabel("altitude [m]")
    fig.tight_layout()
    fig.savefig(OUT / f"sim_{name}_altitude.png", dpi=160, facecolor=BG)
    plt.close(fig)

    # Wind truth magnitude along trajectory
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    style_axes(ax, f"{name}: horizontal wind along path")
    wmag = np.linalg.norm(wind[:, :2], axis=1)
    ax.plot(t, wmag, color=TRUTH, lw=1.4)
    ax.set_xlabel("t [s]")
    ax.set_ylabel("|w| [m/s]")
    fig.tight_layout()
    fig.savefig(OUT / f"sim_{name}_wind.png", dpi=160, facecolor=BG)
    plt.close(fig)

    (OUT / f"sim_{name}_summary.json").write_text(json.dumps(summary, indent=2))


# --------------------------------------------------------------------------- #
# Annotated diagrams
# --------------------------------------------------------------------------- #
def _box(ax, xy, w, h, text, *, fc=PANEL, ec=ACCENT):
    patch = FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        facecolor=fc,
        edgecolor=ec,
        linewidth=1.6,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + w / 2,
        xy[1] + h / 2,
        text,
        ha="center",
        va="center",
        color=TEXT,
        fontsize=9,
        wrap=True,
    )
    return patch


def _arrow(ax, p1, p2, color=ACCENT):
    ax.add_patch(
        FancyArrowPatch(
            p1,
            p2,
            arrowstyle="-|>",
            mutation_scale=14,
            color=color,
            lw=1.6,
            connectionstyle="arc3,rad=0",
        )
    )


def diagram_closed_loop():
    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4.2)
    ax.axis("off")
    ax.set_title("Closed-loop information flow", color=TEXT, fontsize=13, pad=8)

    _box(ax, (0.3, 1.5), 1.8, 1.2, "Plant\n3-DOF +\ntruth wind")
    _box(ax, (2.7, 1.5), 1.8, 1.2, "GPS\nv_GPS =\nv_air + w + ν")
    _box(ax, (5.1, 1.5), 1.8, 1.2, "LPF estimator\nŵ (τe = 5 s)")
    _box(ax, (7.5, 1.5), 2.0, 1.2, "NMPC\nmin J(U)\napply u₀")
    _box(ax, (10.1, 1.5), 1.5, 1.2, "Actuator\nu = ψ̇")

    _arrow(ax, (2.1, 2.1), (2.7, 2.1))
    _arrow(ax, (4.5, 2.1), (5.1, 2.1))
    _arrow(ax, (6.9, 2.1), (7.5, 2.1))
    _arrow(ax, (9.5, 2.1), (10.1, 2.1))
    # feedback
    ax.annotate(
        "",
        xy=(1.2, 1.5),
        xytext=(10.8, 1.5),
        arrowprops=dict(arrowstyle="-|>", color=ACCENT2, lw=1.4,
                        connectionstyle="arc3,rad=0.45"),
    )
    ax.text(6.0, 0.55, "state x updates every Δt = 0.05 s; NMPC every Tc = 1 s",
            ha="center", color=MUTED, fontsize=9)
    ax.text(6.0, 0.2, "prediction freezes ŵ — truth wind shears & gusts → replan absorbs mismatch",
            ha="center", color=MUTED, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "diagram_closed_loop.png", dpi=170, facecolor=BG)
    plt.close(fig)


def diagram_kinematics():
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(PANEL)
    ax.set_xlim(-1.2, 5.5)
    ax.set_ylim(-1.5, 4.2)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Kinematic velocity composition", color=TEXT, fontsize=13, pad=10)

    origin = np.array([1.5, 1.2])
    v_air = np.array([2.4, 1.5])
    w = np.array([1.3, -0.4])
    v_g = v_air + w

    ax.annotate("", xy=origin + v_air, xytext=origin,
                arrowprops=dict(arrowstyle="-|>", color=ACCENT, lw=2.2))
    ax.annotate("", xy=origin + v_air + w, xytext=origin + v_air,
                arrowprops=dict(arrowstyle="-|>", color=ACCENT2, lw=2.2))
    ax.annotate("", xy=origin + v_g, xytext=origin,
                arrowprops=dict(arrowstyle="-|>", color="#81C784", lw=2.4))

    ax.plot(*origin, "o", color=TEXT, ms=7)
    ax.text(*(origin + v_air * 0.5 + np.array([-0.15, 0.35])),
            r"$v_{\mathrm{air}} = V_h[\cos\psi,\sin\psi]^T$", color=ACCENT, fontsize=10)
    ax.text(*(origin + v_air + w * 0.5 + np.array([0.1, -0.35])),
            r"$w = [w_x, w_y]^T$", color=ACCENT2, fontsize=10)
    ax.text(*(origin + v_g * 0.55 + np.array([0.15, 0.25])),
            r"$v_g = v_{\mathrm{air}} + w$", color="#81C784", fontsize=11)

    ax.text(0.1, 3.6, r"$\dot x = V_h\cos\psi + w_x$", color=TEXT, fontsize=12)
    ax.text(0.1, 3.15, r"$\dot y = V_h\sin\psi + w_y$", color=TEXT, fontsize=12)
    ax.text(0.1, 2.7, r"$\dot h = -V_v$", color=TEXT, fontsize=12)
    ax.text(0.1, 2.25, r"$\dot\psi = u$", color=TEXT, fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "diagram_kinematics.png", dpi=170, facecolor=BG)
    plt.close(fig)


def diagram_wind_profile():
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.6))
    fig.patch.set_facecolor(BG)
    h = np.linspace(2, 500, 200)
    W10, alpha = 3.0, 0.14
    wmag = W10 * (h / 10.0) ** alpha
    style_axes(axes[0], "Power-law shear profile")
    axes[0].plot(wmag, h, color=ACCENT, lw=2)
    axes[0].axhline(10, color=MUTED, ls=":", lw=1)
    axes[0].set_xlabel("|w_mean| [m/s]")
    axes[0].set_ylabel("altitude h [m]")
    axes[0].text(2.2, 30, r"$|w|=W_{10}(h/10)^\alpha$", color=TEXT, fontsize=11)

    # OU sample path
    style_axes(axes[1], "Ornstein–Uhlenbeck gust sample")
    rng = np.random.default_rng(0)
    dt, tau, sig = 0.05, 8.0, 0.8
    n = 2000
    g = np.zeros((2, n))
    for k in range(1, n):
        g[:, k] = g[:, k - 1] * (1 - dt / tau) + sig * np.sqrt(2 * dt / tau) * rng.standard_normal(2)
    t = np.arange(n) * dt
    axes[1].plot(t, g[0], color=TRUTH, lw=1.0, label=r"$\xi_x$")
    axes[1].plot(t, g[1], color=EST, lw=1.0, label=r"$\xi_y$")
    axes[1].set_xlabel("t [s]")
    axes[1].set_ylabel("gust [m/s]")
    axes[1].legend(facecolor=PANEL, edgecolor="#274567", labelcolor=TEXT, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "diagram_wind.png", dpi=170, facecolor=BG)
    plt.close(fig)


def diagram_horizon():
    fig, ax = plt.subplots(figsize=(9.5, 3.8))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")
    ax.set_title("Shrinking horizon: N tracks remaining flight time", color=TEXT, fontsize=13)

    # High altitude
    for i, (y, n, label) in enumerate(
        [(2.7, 8, "High altitude → long N (path shaping)"),
         (1.3, 4, "Near ground → short N (touchdown placement)")]
    ):
        ax.text(0.2, y + 0.55, label, color=MUTED, fontsize=10)
        for j in range(n):
            x0 = 0.4 + j * 1.05
            _box(ax, (x0, y - 0.15), 0.9, 0.55, f"u{j}", fc=PANEL, ec=ACCENT if j == 0 else "#274567")
        ax.text(0.4 + n * 1.05, y + 0.1, "→ predicted TD", color=ACCENT2, fontsize=9)
    ax.text(5, 0.35,
            r"$N = \min(N_{max},\ \max(3,\ \lceil t_{go}/T_s\rceil))$"
            "   with   $t_{go}=h/V_v$",
            ha="center", color=TEXT, fontsize=11)
    fig.savefig(OUT / "diagram_horizon.png", dpi=170, facecolor=BG, bbox_inches="tight")
    plt.close(fig)


def diagram_cost_stack():
    fig, ax = plt.subplots(figsize=(9.8, 4.0))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.2)
    ax.axis("off")
    ax.set_title("Cost J(U): what each term buys", color=TEXT, fontsize=13)

    terms = [
        (0.3, "Path\nQpath‖r−rt‖²", "Stay near pad\nalong horizon"),
        (2.5, "Effort\nRu u²", "Limit bank /\nturn rate"),
        (4.7, "Smooth\nRΔu (Δu)²", "Avoid bang-\nbang brakes"),
        (6.9, "Upwind\nQhead e²", "Into-wind final\nheading (h<60 m)"),
    ]
    for x, title, note in terms:
        _box(ax, (x, 1.8), 1.9, 1.4, title)
        ax.text(x + 0.95, 1.2, note, ha="center", va="top", color=MUTED, fontsize=8)
    _box(ax, (2.5, 0.15), 5.0, 0.7,
         "Terminal: Qf (1 + cterm/max(h,5)) ‖rN−rt‖²  → weight grows as h falls",
         ec=ACCENT2)
    fig.tight_layout()
    fig.savefig(OUT / "diagram_cost.png", dpi=170, facecolor=BG)
    plt.close(fig)


def main():
    print("Running mpc_v1 port (seed=2)…")
    p = P()
    out = simulate_flight(p, seed=2)
    print(f"  flight time = {out['t'][-1]:.1f} s, miss = {out['miss']:.1f} m")
    plot_mpc_results(out, p)

    print("Plotting parafoil_sim flight logs…")
    for name in ("steady_wind", "shear_turbulence", "strong_gusts", "calm"):
        path = ROOT / "parafoil_sim" / "output" / f"{name}_flight.json"
        if path.exists():
            plot_sim_flight(name)
            print(f"  {name}")

    print("Drawing diagrams…")
    diagram_closed_loop()
    diagram_kinematics()
    diagram_wind_profile()
    diagram_horizon()
    diagram_cost_stack()
    print(f"Wrote assets to {OUT}")


if __name__ == "__main__":
    main()
