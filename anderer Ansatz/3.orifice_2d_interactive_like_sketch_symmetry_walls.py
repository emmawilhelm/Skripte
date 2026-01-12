"""
Interaktive 2D-"Simulation" (Surrogat, kein CFD) für Strömung durch eine Lochscheibe.
Beim Ausführen öffnet sich ein Matplotlib-Fenster und zeigt eine Animation (quasi live Simulation).

Fixes gegenüber der ersten Version:
1) Achsensymmetrie: u(x,+y)=u(x,-y), v(x,+y)=-v(x,-y) wird pro Frame erzwungen (numerische Symmetrisierung).
2) Rohrwände: Streamplot-Startpunkte werden explizit im Inneren gesetzt (nicht auf den Wänden),
   damit keine Linien "von der Wand entspringen". Zusätzlich gehen Perturbationen an den Wänden gegen 0.
3) Stromlinien-Konturen (ψ-Isolinien): ψ wird symmetrisch vom Zentrum (y=0) integriert (ψ(x,0)=0).

Physikalische Skalierung:
- Bulk-Geschwindigkeit im Rohr: U1 = Q / (pi*D^2/4)
- mittlere Geschwindigkeit in der Öffnung: U0 = Q / (pi*d0^2/4)
- Druckverlust (Idelchik-Form): Δp = ζ * (ρ U1^2 / 2)   (w_ref = U1)

WICHTIG:
- Das gezeigte 2D-Feld ist ein kinematisches Surrogat (anschaulich, skizzenähnlich),
  keine Lösung der Navier–Stokes-Gleichungen.
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
def source_velocity(strength: float, x0: float, y0: float, X: np.ndarray, Y: np.ndarray, core: float):
    dx = X - x0
    dy = Y - y0
    r2 = dx*dx + dy*dy + core*core
    u = strength/(2*np.pi) * dx / r2
    v = strength/(2*np.pi) * dy / r2
    return u, v

def vortex_velocity(gamma: float, x0: float, y0: float, X: np.ndarray, Y: np.ndarray, core: float):
    dx = X - x0
    dy = Y - y0
    r2 = dx*dx + dy*dy + core*core
    u = -gamma/(2*np.pi) * dy / r2
    v =  gamma/(2*np.pi) * dx / r2
    return u, v

# ---------- base "pipe-like" profile (visual) ----------
u_base = 1.5 * U1 * (1 - (2*Y/H)**2)
u_base = np.clip(u_base, 0, None)
v_base = np.zeros_like(u_base)

# ---------- fixed geometry mask (solid plate except opening) ----------
plate_mask = (np.abs(X) < t/2) & (np.abs(Y) > opening_half)

# ---------- envelopes to localize perturbations ----------
env_up = np.exp(-((np.maximum(-X, 0)/(1.2*d0))**2))
env_dn = 1 - np.exp(-(np.maximum(X, 0)/(1.5*d0))**2)
env = 0.7*env_up + 0.8*env_dn

# Perturbations vanish at the walls (helps "no-penetration" visual)
env_wall = 1 - (np.abs(Y)/(H/2))**10
env_wall = np.clip(env_wall, 0, 1)

# ---------- strengths (scaled to your computed velocities) ----------
core_s = 0.10 * d0
core_v = 0.08 * d0

m0 = U1 * d0 * 2.5
eps0 = 0.25 * t
Gamma0 = 0.8 * (U0 - U1) * d0

def enforce_axis_symmetry(u: np.ndarray, v: np.ndarray):
    """Enforce symmetry about y=0: u even, v odd."""
    u_flip = np.flipud(u)
    v_flip = np.flipud(v)
    u_sym = 0.5 * (u + u_flip)
    v_sym = 0.5 * (v - v_flip)
    return u_sym, v_sym

def streamfunction_centerline(x: np.ndarray, y: np.ndarray, u: np.ndarray):
    """Compute ψ by integrating u=∂ψ/∂y from the centerline y=0 (ψ(x,0)=0)."""
    u_f = np.ma.filled(u, 0.0)
    dy = y[1] - y[0]
    ny, nx = u_f.shape
    psi = np.zeros((ny, nx), dtype=float)
    i0 = int(np.argmin(np.abs(y)))  # index closest to y=0

    for i in range(i0 + 1, ny):
        psi[i, :] = psi[i-1, :] + 0.5 * (u_f[i, :] + u_f[i-1, :]) * dy
    for i in range(i0 - 1, -1, -1):
        psi[i, :] = psi[i+1, :] - 0.5 * (u_f[i, :] + u_f[i+1, :]) * dy
    return psi

# ---------- visualization setup ----------
fig, ax = plt.subplots(figsize=(12, 4.8))
ax.set_xlim(-L_up, L_dn)
ax.set_ylim(-H/2, H/2)
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
ax.set_title("Interaktive 2D-Strömungsdarstellung durch Lochscheibe (Surrogat)")

ax.add_patch(patches.Rectangle((-t/2, opening_half), t, H/2-opening_half,
                               linewidth=1.5, edgecolor="black", facecolor="lightcoral", alpha=0.85))
ax.add_patch(patches.Rectangle((-t/2, -H/2), t, H/2-opening_half,
                               linewidth=1.5, edgecolor="black", facecolor="lightcoral", alpha=0.85))

def compute_field(phase: float):
    m = m0 * (1.0 + 0.10*np.sin(phase))
    eps = eps0 * (1.0 + 0.05*np.cos(phase))

    xv = 0.9*d0 + 0.20*d0*np.sin(0.7*phase)
    yv = 0.35*d0 + 0.08*d0*np.sin(phase)
    Gamma = Gamma0 * (1.0 + 0.15*np.cos(phase))

    u_sink, v_sink = source_velocity(-m, -eps, 0.0, X, Y, core_s)
    u_src,  v_src  = source_velocity(+m, +eps, 0.0, X, Y, core_s)
    u_v1, v_v1 = vortex_velocity(+Gamma, xv, +yv, X, Y, core_v)
    u_v2, v_v2 = vortex_velocity(-Gamma, xv, -yv, X, Y, core_v)

    u = u_base + env*env_wall*(u_sink + u_src + u_v1 + u_v2)
    v = v_base + env*env_wall*(v_sink + v_src + v_v1 + v_v2)

    # enforce symmetry
    u, v = enforce_axis_symmetry(u, v)

    u_m = np.ma.array(u, mask=plate_mask)
    v_m = np.ma.array(v, mask=plate_mask)
    speed = np.ma.sqrt(u_m**2 + v_m**2)
    return u_m, v_m, speed

u_m, v_m, speed = compute_field(0.0)

im = ax.contourf(X, Y, speed, levels=45)
cbar = fig.colorbar(im, ax=ax)
cbar.set_label("|u| [m/s]")

stream = None
psi_contours = None

# Streamline seeds inside the pipe (avoid walls)
wall_margin = 0.06 * H
ys = np.linspace(-H/2 + wall_margin, H/2 - wall_margin, 26)
start_points = np.vstack([
    np.column_stack([np.full_like(ys, -2.2*d0), ys]),
    np.column_stack([np.full_like(ys, +2.2*d0), ys]),
])

def update(frame: int):
    global stream, im, psi_contours

    phase = frame * 0.12
    u_m, v_m, speed = compute_field(phase)

    for coll in im.collections:
        coll.remove()
    im = ax.contourf(X, Y, speed, levels=45)

    if stream is not None:
        stream.lines.remove()
        stream.arrows.remove()

    u_f = np.ma.filled(u_m, 0.0)
    v_f = np.ma.filled(v_m, 0.0)
    stream = ax.streamplot(x, y, u_f, v_f, start_points=start_points,
                           density=2.0, linewidth=1.0, arrowsize=1.0)

    psi = streamfunction_centerline(x, y, u_m)
    if psi_contours is not None:
        for coll in psi_contours.collections:
            coll.remove()
    psi_contours = ax.contour(X, Y, psi, levels=28, linewidths=0.9, alpha=0.85)

    ax.set_title(f"Interaktive 2D-Strömungsdarstellung (Surrogat) | frame={frame}")
    return im.collections

ani = FuncAnimation(fig, update, frames=220, interval=40, blit=False)
plt.show()
