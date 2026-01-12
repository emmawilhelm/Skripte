"""
Interaktive 2D-"Simulation" (Surrogat, kein CFD) für Strömung durch eine Lochscheibe.
Beim Ausführen öffnet sich ein Matplotlib-Fenster und zeigt eine Animation (quasi live Simulation).

Physikalische Skalierung:
- Bulk-Geschwindigkeit im Rohr: U1 = Q / (pi*D^2/4)
- mittlere Geschwindigkeit in der Öffnung: U0 = Q / (pi*d0^2/4)
- Druckverlust (Idelchik-Form): Δp = ζ * (ρ U1^2 / 2)   (w_ref = U1)

WICHTIG:
- Das gezeigte 2D-Feld ist ein kinematisches Surrogat (anschaulich, skizzenähnlich),
  keine Lösung der Navier–Stokes-Gleichungen.
- Die "Zeit" in der Animation ist ein Parameter, der die Stärke/Position der Wirbel leicht variiert,
  um eine dynamische Darstellung zu erhalten.

Benötigt:
  pip install numpy matplotlib
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation

# -------------------- Inputs --------------------
rho = 1440.0  # [kg/m^3]
mu  = 1.0     # [Pa*s]
D   = 4.30    # [m]
Q   = 0.0097  # [m^3/s]
d0  = 2.61    # [m]
t   = 0.8     # [m] (nur für Darstellung/Maskierung)
zeta= 10.0    # [-]
# ------------------------------------------------

# ---------- Derived scalars ----------
A1 = np.pi * D**2 / 4.0
A0 = np.pi * d0**2 / 4.0
U1 = Q / A1
U0 = Q / A0
ReD = rho * U1 * D / mu
dp  = zeta * 0.5 * rho * U1**2  # [Pa] using w_ref=U1

print("=== Scalars ===")
print(f"A1  = {A1:.6f} m^2")
print(f"A0  = {A0:.6f} m^2")
print(f"U1  = {U1:.6e} m/s (Rohr-Bulk)")
print(f"U0  = {U0:.6e} m/s (Öffnung-Mittelwert)")
print(f"ReD = {ReD:.6e} [-]")
print(f"Δp  = {dp:.6e} Pa  (Idelchik: ζ * ρ U1²/2)")

# ---------- 2D domain (x-y) ----------
H = D                 # "Kanalhöhe" im 2D-Bild
L_up = 3.0 * d0
L_dn = 10.0 * d0

nx, ny = 620, 250
x = np.linspace(-L_up, L_dn, nx)
y = np.linspace(-H/2, H/2, ny)
X, Y = np.meshgrid(x, y, indexing="xy")

opening_half = d0 / 2.0

# ---------- potential elements (regularized) ----------
def source_velocity(strength: float, x0: float, y0: float, X: np.ndarray, Y: np.ndarray, core: float) -> tuple[np.ndarray, np.ndarray]:
    """2D source/sink at (x0,y0). strength [m^2/s]. core [m] regularizes singularity."""
    dx = X - x0
    dy = Y - y0
    r2 = dx*dx + dy*dy + core*core
    u = strength/(2*np.pi) * dx / r2
    v = strength/(2*np.pi) * dy / r2
    return u, v

def vortex_velocity(gamma: float, x0: float, y0: float, X: np.ndarray, Y: np.ndarray, core: float) -> tuple[np.ndarray, np.ndarray]:
    """2D point vortex at (x0,y0). gamma [m^2/s]. core [m] regularizes singularity."""
    dx = X - x0
    dy = Y - y0
    r2 = dx*dx + dy*dy + core*core
    u = -gamma/(2*np.pi) * dy / r2
    v =  gamma/(2*np.pi) * dx / r2
    return u, v

# ---------- base "pipe-like" profile (visual) ----------
# Parabolic with mean U1 in a 2D channel of height H:
# mean(u) = 2/3 * u_max  -> u_max = 1.5*U1
u_base = 1.5 * U1 * (1 - (2*Y/H)**2)
u_base = np.clip(u_base, 0, None)
v_base = np.zeros_like(u_base)

# ---------- fixed geometry mask (solid plate except opening) ----------
plate_mask = (np.abs(X) < t/2) & (np.abs(Y) > opening_half)

# ---------- envelopes to localize perturbations ----------
env_up = np.exp(-((np.maximum(-X, 0)/(1.2*d0))**2))               # near plate upstream
env_dn = 1 - np.exp(-(np.maximum(X, 0)/(1.5*d0))**2)             # grows downstream
env = 0.7*env_up + 0.8*env_dn

env_wall = 1 - (np.abs(Y)/(H/2))**8
env_wall = np.clip(env_wall, 0, 1)

# ---------- strengths (scaled to your computed velocities) ----------
core_s = 0.10 * d0
core_v = 0.08 * d0

# source/sink pair around plate to create convergence + throughflow
m0 = U1 * d0 * 2.5  # [m^2/s] (scaled; you can tune for more/less "suck-in")
eps0 = 0.25 * t     # [m] separation around x=0

# vortex strength scaled with excess jet speed (U0-U1) and orifice size
Gamma0 = 0.8 * (U0 - U1) * d0  # [m^2/s]

# ---------- visualization setup ----------
fig, ax = plt.subplots(figsize=(12, 4.8))
ax.set_xlim(-L_up, L_dn)
ax.set_ylim(-H/2, H/2)
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
ax.set_title("Interaktive 2D-Strömungsdarstellung durch Lochscheibe (Surrogat)")

# plate drawing
ax.add_patch(patches.Rectangle((-t/2, opening_half), t, H/2-opening_half,
                               linewidth=1.5, edgecolor="black", facecolor="lightcoral", alpha=0.85))
ax.add_patch(patches.Rectangle((-t/2, -H/2), t, H/2-opening_half,
                               linewidth=1.5, edgecolor="black", facecolor="lightcoral", alpha=0.85))

# initial field (timestep 0)
def compute_field(phase: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build u,v and speed fields for a given phase.
    phase is a dimensionless time-like parameter [rad].
    """
    # Slightly modulate source-sink and vortex positions/strengths over time for animation
    m = m0 * (1.0 + 0.10*np.sin(phase))
    eps = eps0 * (1.0 + 0.05*np.cos(phase))

    # downstream vortex centers wobble a bit
    xv = 0.9*d0 + 0.20*d0*np.sin(0.7*phase)
    yv = 0.35*d0 + 0.08*d0*np.sin(phase)

    Gamma = Gamma0 * (1.0 + 0.15*np.cos(phase))

    # source/sink
    u_sink, v_sink = source_velocity(-m, -eps, 0.0, X, Y, core_s)
    u_src,  v_src  = source_velocity(+m, +eps, 0.0, X, Y, core_s)

    # counter-rotating vortices (wake recirculation)
    u_v1, v_v1 = vortex_velocity(+Gamma, xv, +yv, X, Y, core_v)
    u_v2, v_v2 = vortex_velocity(-Gamma, xv, -yv, X, Y, core_v)

    u = u_base + env*env_wall*(u_sink + u_src + u_v1 + u_v2)
    v = v_base + env*env_wall*(v_sink + v_src + v_v1 + v_v2)

    # mask solid regions
    u_m = np.ma.array(u, mask=plate_mask)
    v_m = np.ma.array(v, mask=plate_mask)
    speed = np.ma.sqrt(u_m**2 + v_m**2)
    return u_m, v_m, speed

def streamfunction_from_uv(x: np.ndarray, y: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Approximiert die Stromfunktion ψ(x,y) für ein 2D-Feld (u,v) auf einem kartesischen Gitter.

    Definition (2D inkompressibel):
      u = ∂ψ/∂y
      v = -∂ψ/∂x

    Wir integrieren ψ zeilenweise aus u (über y) und korrigieren danach grob über v (über x).
    Für dein Surrogatfeld (nicht exakt CFD/divergenzfrei) ist das eine robuste Visualisierung.
    """
    # Masken in normale Arrays überführen (maskierte Punkte -> 0, damit Integration stabil bleibt)
    u_f = np.ma.filled(u, 0.0)
    v_f = np.ma.filled(v, 0.0)

    nx = len(x)
    ny = len(y)
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    psi = np.zeros((ny, nx), dtype=float)

    # 1) Integration über y: psi(y) = ∫ u dy  (für jedes x)
    psi[1:, :] = np.cumsum(0.5 * (u_f[1:, :] + u_f[:-1, :]) * dy, axis=0)

    # 2) Grobe Konsistenzkorrektur über x mit v: dψ/dx ≈ -v  -> psi(x) = psi(x0) - ∫ v dx
    # Wir korrigieren jede y-Zeile so, dass die x-Integration von v ungefähr passt.
    corr = np.zeros_like(psi)
    corr[:, 1:] = -np.cumsum(0.5 * (v_f[:, 1:] + v_f[:, :-1]) * dx, axis=1)

    # Mischfaktor (0..1): 0 = nur u-Integration, 1 = nur v-Integration
    alpha = 0.35
    psi = (1 - alpha) * psi + alpha * corr

    return psi

u_m, v_m, speed = compute_field(0.0)

# background speed field
im = ax.contourf(X, Y, speed, levels=45)
cbar = fig.colorbar(im, ax=ax)
cbar.set_label("|u| [m/s]")

# streamlines (we will redraw each frame)
stream = None
psi_contours = None

# For speed: use a coarser grid for streamlines
sx = slice(0, nx, 2)
sy = slice(0, ny, 2)
x_s = x[sx]
y_s = y[sy]
X_s, Y_s = np.meshgrid(x_s, y_s, indexing="xy")

def update(frame: int):
    global stream, im, psi_contours

    phase = frame * 0.12  # "time" step size (tune)
    u_m, v_m, speed = compute_field(phase)


    # --- NEW: ψ-Konturen (Stromlinien) ---
    psi = streamfunction_from_uv(x, y, u_m, v_m)

    # Falls du die alten ψ-Konturen entfernen willst:
    if hasattr(update, "psi_contours") and update.psi_contours is not None:
        for coll in update.psi_contours.collections:
            coll.remove()

    # ψ-Isolinien zeichnen (Strömungslinien)
    update.psi_contours = ax.contour(X, Y, psi, levels=28, linewidths=0.9)
    # Remove previous contour collections
    for coll in im.collections:
        coll.remove()
    im = ax.contourf(X, Y, speed, levels=45)

    # Remove previous streamlines if present
    if stream is not None:
        # stream.lines is a LineCollection; stream.arrows is a PatchCollection
        stream.lines.remove()
        stream.arrows.remove()

    # Draw new streamlines on downsampled grid
    u_s = u_m[sy, sx]
    v_s = v_m[sy, sx]
    stream = ax.streamplot(x_s, y_s, u_s, v_s, density=2.0, linewidth=1.0, arrowsize=1.0)

    ax.set_title(f"Interaktive 2D-Strömungsdarstellung (Surrogat) | frame={frame}")

    return im.collections

ani = FuncAnimation(fig, update, frames=220, interval=40, blit=False)

# Show interactive window (this is what you want in VS Code)
plt.show()
