"""
2D surrogate flow around a symmetric orifice-like obstruction with live simulation.

This keeps the Idelchik-style local loss estimate, but replaces the static PNG
plots with a 2D time-stepping particle simulation that resembles the sketch:
two vertical plates from top and bottom with a gap, flow left-to-right, and
recirculation near the plate edges. This is NOT CFD; it is a kinematic field.
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def areas(D: float, d0: float) -> tuple[float, float]:
    """Return pipe area A1 and orifice area A0 [m^2]."""
    A1 = np.pi * D**2 / 4.0
    A0 = np.pi * d0**2 / 4.0
    return A1, A0

def bulk_velocity(Q: float, A: float) -> float:
    """Mean (bulk) velocity U = Q/A [m/s]."""
    return Q / A

def reynolds(rho: float, U: float, D: float, mu: float) -> float:
    """Re_D = rho*U*D/mu [-]."""
    return rho * U * D / mu

def local_loss_dp(zeta: float, rho: float, w_ref: float) -> float:
    """Δp = ζ * (ρ w_ref^2 / 2) [Pa]."""
    return zeta * 0.5 * rho * w_ref**2

def velocity_field(
    x: np.ndarray,
    y: np.ndarray,
    U0: float,
    H: float,
    gap: float,
    plate_x: float,
    plate_w: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Kinematic 2D velocity field resembling flow through a narrow gap.
    - Base uniform flow left-to-right.
    - Smooth deflection around two vertical plates (top/bottom).
    - Recirculation via a pair of vortices downstream of the edges.
    """
    u = np.full_like(x, U0)
    v = np.zeros_like(y)

    # Plates: rectangles from top and bottom with a gap around y=0.
    y_top = (gap / 2.0, H / 2.0)
    y_bot = (-H / 2.0, -gap / 2.0)

    def plate_mask(y_range: tuple[float, float]) -> np.ndarray:
        in_x = (x >= plate_x) & (x <= plate_x + plate_w)
        in_y = (y >= y_range[0]) & (y <= y_range[1])
        return in_x & in_y

    mask_top = plate_mask(y_top)
    mask_bot = plate_mask(y_bot)

    # Smooth repulsion around plates to bend streamlines around the gap.
    dist_x = x - (plate_x + plate_w / 2.0)
    dist_y_top = y - y_top[0]
    dist_y_bot = y - y_bot[1]
    repulse = np.exp(-((dist_x / (0.25 * plate_w)) ** 2))
    v += 0.9 * U0 * repulse * np.tanh(-dist_y_top / (0.12 * gap))
    v += 0.9 * U0 * repulse * np.tanh(-dist_y_bot / (0.12 * gap))

    # Throttle velocity through the gap for jet-like core.
    gap_band = np.exp(-((y / (0.45 * gap)) ** 2))
    jet_boost = 1.5 * U0 * np.exp(-((x - (plate_x + plate_w)) / (0.8 * gap)) ** 2)
    u += jet_boost * gap_band

    # Vortices downstream of the plate edges to suggest recirculation.
    vortices = [
        (plate_x + 0.8 * plate_w, gap / 2.0 + 0.2 * gap, 0.06 * U0 * gap),
        (plate_x + 0.8 * plate_w, -gap / 2.0 - 0.2 * gap, -0.06 * U0 * gap),
    ]
    for xv, yv, gamma in vortices:
        dx = x - xv
        dy = y - yv
        r2 = dx * dx + dy * dy + (0.08 * gap) ** 2
        u += -gamma * dy / (2.0 * np.pi * r2)
        v += gamma * dx / (2.0 * np.pi * r2)

    # Zero velocity inside plates and slow flow in their immediate vicinity.
    u[mask_top | mask_bot] = 0.0
    v[mask_top | mask_bot] = 0.0
    near_plate = ((x >= plate_x - 0.05 * plate_w) & (x <= plate_x + 1.05 * plate_w))
    near_plate &= (y >= -H / 2.0) & (y <= H / 2.0)
    u[near_plate] *= 0.6
    v[near_plate] *= 0.6
    return u, v

def main() -> None:
    rho = 1440.0
    mu = 1.0
    D = 4.3
    Q = 0.0097
    d0 = 2.61
    zeta = 10.0

    A1, A0 = areas(D, d0)
    U1 = bulk_velocity(Q, A1)
    Us = bulk_velocity(Q, A0)
    Re = reynolds(rho, U1, D, mu)

    dp = local_loss_dp(zeta, rho, U1)

    print("=== Inputs ===")
    print(f"rho = {rho} kg/m^3")
    print(f"mu  = {mu} Pa*s")
    print(f"D   = {D} m")
    print(f"d0  = {d0} m")
    print(f"Q   = {Q} m^3/s")
    print(f"zeta= {zeta} [-]")
    print("\n=== Derived ===")
    print(f"A1  = {A1:.6f} m^2")
    print(f"A0  = {A0:.6f} m^2")
    print(f"U1  = {U1:.6e} m/s   (bulk in pipe)")
    print(f"Us  = {Us:.6e} m/s   (mean through opening)")
    print(f"ReD = {Re:.6e} [-]")
    print("\n=== Idelchik-style local loss ===")
    print(f"Δp  = {dp:.6e} Pa    using w_ref = U1")

    # 2D simulation domain (cartesian)
    Lx = 8.0 * d0
    H = D
    gap = d0
    plate_x = -0.2 * d0
    plate_w = 0.25 * d0

    # Particle initialization (left inlet)
    rng = np.random.default_rng(3)
    n_particles = 800
    px = rng.uniform(-Lx / 2.0, -0.3 * d0, n_particles)
    py = rng.uniform(-H / 2.0, H / 2.0, n_particles)

    # Precompute background grid for velocity evaluation
    nx, ny = 200, 140
    gx = np.linspace(-Lx / 2.0, Lx / 2.0, nx)
    gy = np.linspace(-H / 2.0, H / 2.0, ny)
    GX, GY = np.meshgrid(gx, gy, indexing="xy")

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.set_xlim(-Lx / 2.0, Lx / 2.0)
    ax.set_ylim(-H / 2.0, H / 2.0)
    ax.set_aspect("equal")
    ax.set_title("2D surrogate flow simulation (particles)")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

    # Draw plates
    ax.add_patch(plt.Rectangle((plate_x, gap / 2.0), plate_w, H / 2.0 - gap / 2.0,
                               facecolor="#d84a3a", edgecolor="#7a1f1f", alpha=0.9))
    ax.add_patch(plt.Rectangle((plate_x, -H / 2.0), plate_w, H / 2.0 - gap / 2.0,
                               facecolor="#d84a3a", edgecolor="#7a1f1f", alpha=0.9))

    # Background speed magnitude + streamlines + velocity vectors
    gu, gv = velocity_field(GX, GY, U0=U1, H=H, gap=gap, plate_x=plate_x, plate_w=plate_w)
    gspeed = np.sqrt(gu ** 2 + gv ** 2)
    ax.contourf(GX, GY, gspeed, levels=30, cmap="YlOrBr", alpha=0.6)
    ax.streamplot(gx, gy, gu, gv, color="#e57d1a", density=1.2, linewidth=1.2, arrowsize=1.1)

    # Sparse velocity vectors to show direction and magnitude
    step = 10
    ax.quiver(
        GX[::step, ::step],
        GY[::step, ::step],
        gu[::step, ::step],
        gv[::step, ::step],
        color="#a34b00",
        alpha=0.7,
        scale=12.0,
        width=0.0025,
    )

    scat = ax.scatter(px, py, s=8, c="#f2993a", alpha=0.9)

    dt = 0.02 * d0 / max(U1, 1e-6)

    def step_particles(xp: np.ndarray, yp: np.ndarray) -> None:
        u, v = velocity_field(xp, yp, U0=U1, H=H, gap=gap, plate_x=plate_x, plate_w=plate_w)
        xp[:] = xp + u * dt
        yp[:] = yp + v * dt

        # Recycle particles leaving the domain to the inlet
        out_right = xp > Lx / 2.0
        xp[out_right] = rng.uniform(-Lx / 2.0, -0.3 * d0, out_right.sum())
        yp[out_right] = rng.uniform(-H / 2.0, H / 2.0, out_right.sum())

        out_left = xp < -Lx / 2.0
        xp[out_left] = rng.uniform(-Lx / 2.0, -0.3 * d0, out_left.sum())
        yp[out_left] = rng.uniform(-H / 2.0, H / 2.0, out_left.sum())

        # Reflect at top/bottom walls
        yp[yp > H / 2.0] = H / 2.0
        yp[yp < -H / 2.0] = -H / 2.0

        # Reflect off the plates so nothing passes through the obstruction.
        in_plate_x = (xp >= plate_x) & (xp <= plate_x + plate_w)
        in_plate_top = in_plate_x & (yp >= gap / 2.0) & (yp <= H / 2.0)
        in_plate_bot = in_plate_x & (yp <= -gap / 2.0) & (yp >= -H / 2.0)
        hit_plate = in_plate_top | in_plate_bot
        if np.any(hit_plate):
            xp[hit_plate] = plate_x - 0.02 * d0
            yp[hit_plate] += rng.uniform(-0.02 * gap, 0.02 * gap, hit_plate.sum())

    def update(_frame: int):
        step_particles(px, py)
        scat.set_offsets(np.c_[px, py])
        return (scat,)

    FuncAnimation(fig, update, interval=30, blit=True)
    plt.show()

if __name__ == "__main__":
    main()
