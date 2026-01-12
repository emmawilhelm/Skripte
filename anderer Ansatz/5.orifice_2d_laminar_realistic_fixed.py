"""
Interaktive 2D-Darstellung (Surrogat, kein CFD) für Strömung durch eine Lochscheibe im Rohr
– low-Re (hier Re≈4) mit physikalisch konsistenteren Randbedingungen:

Wesentliche Fixes gegenüber der vorherigen laminar-Version:
1) Jet-Exzess nur stromab: u_excess(x<0)=0  (kein "Jet" upstream).
2) Querschnitts-Durchflusskonstanz: Für jedes x wird mean_y(u)=U1 erzwungen
   -> entspricht konstantem Volumenstrom pro Tiefe im 2D-Schnitt.
3) v(x,y) wird danach aus du/dx + dv/dy = 0 (mit v(y=0)=0) rekonstruiert,
   damit die Stromlinien konsistent bleiben.

Hinweis zur "CFD-ähnlichen" Jet-Form:
- Das Feld ist kein Navier–Stokes-Löser, aber der Jet wird so konstruiert,
  dass er (a) nur downstream existiert und (b) sich viskos (∼sqrt(x)) aufweitet,
  was zum low-Re Regime passt.

Benötigt: numpy, matplotlib
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
t   = 0.8     # [m] (nur Darstellung/Maskierung)
zeta= 10.0    # [-]
# ------------------------------------------------

# ---------- Derived scalars ----------
A1 = np.pi * D**2 / 4.0
A0 = np.pi * d0**2 / 4.0
U1 = Q / A1                 # [m/s] Bulk im Rohr
U0 = Q / A0                 # [m/s] mittlere Durchtrittsgeschwindigkeit (3D)
nu = mu / rho               # [m^2/s]
ReD = rho * U1 * D / mu
dp  = zeta * 0.5 * rho * U1**2

print("=== Scalars ===")
print(f"nu = {nu:.6e} m^2/s")
print(f"U1 = {U1:.6e} m/s (Rohr-Bulk)")
print(f"U0 = {U0:.6e} m/s (Öffnung-Mittelwert, 3D)")
print(f"ReD= {ReD:.6e} [-]")
print(f"Δp = {dp:.6e} Pa (Idelchik: ζ*ρ*U1²/2)")

# ---------- 2D domain (x-y) ----------
H = D
L_up = 3.0 * d0
L_dn = 10.0 * d0
nx, ny = 620, 250
x = np.linspace(-L_up, L_dn, nx)
y = np.linspace(-H/2, H/2, ny)
X, Y = np.meshgrid(x, y, indexing="xy")

opening_half = d0 / 2.0
plate_mask = (np.abs(X) < t/2) & (np.abs(Y) > opening_half)

# ---------- base "pipe-like" profile (visual; mean = U1) ----------
u_base = 1.5 * U1 * (1 - (2*Y/H)**2)
u_base = np.clip(u_base, 0, None)
v_base = np.zeros_like(u_base)

def enforce_axis_symmetry(u: np.ndarray, v: np.ndarray):
    """u even, v odd about y=0."""
    u_flip = np.flipud(u)
    v_flip = np.flipud(v)
    u_sym = 0.5 * (u + u_flip)
    v_sym = 0.5 * (v - v_flip)
    return u_sym, v_sym

# ---------- Laminar jet model: u_excess(x,y) ----------
# Width growth (viscous spreading):
#   b(x) = sqrt(b0^2 + 4*nu*x/U0), x>0
# Excess amplitude chosen so that ∫ u_excess dy is constant:
#   A(x) = A0 * b0 / b(x)
b0 = (d0/2.0) * 0.65
A0_excess = max(U0 - U1, 0.0)

def b_of_x(xpos: np.ndarray):
    return np.sqrt(b0*b0 + 4.0*nu*xpos/max(U0, 1e-12))

def jet_excess_u(X: np.ndarray, Y: np.ndarray):
    # Only downstream of the plate
    Hx = (X > 0.0).astype(float)
    xpos = np.where(X > 0.0, X, 0.0)
    b = b_of_x(xpos)
    A = A0_excess * (b0 / np.maximum(b, 1e-12))
    return Hx * A * np.exp(-(Y / np.maximum(b, 1e-12))**2)

def v_from_u(u: np.ndarray, x: np.ndarray, y: np.ndarray):
    """Approximate incompressibility in 2D: du/dx + dv/dy = 0 with v(y=0)=0."""
    du_dx = np.gradient(u, x, axis=1)
    v = np.zeros_like(u)
    i0 = int(np.argmin(np.abs(y)))
    dy = y[1] - y[0]
    # integrate upward
    for i in range(i0 + 1, len(y)):
        v[i, :] = v[i-1, :] + (-0.5*(du_dx[i, :] + du_dx[i-1, :])) * dy
    # integrate downward
    for i in range(i0 - 1, -1, -1):
        v[i, :] = v[i+1, :] - (-0.5*(du_dx[i, :] + du_dx[i+1, :])) * dy
    return v

def enforce_constant_flux(u: np.ndarray, U_target: float):
    """
    Enforce mean_y(u)=U_target for every x (column-wise).
    This keeps the 2D volumetric flow (per unit depth) constant along x.
    """
    u_mean = np.mean(u, axis=0)           # mean over y -> function of x
    return u - (u_mean - U_target)        # shift each x-column

# ---------- visualization ----------
fig, ax = plt.subplots(figsize=(12, 4.8))
ax.set_xlim(-L_up, L_dn)
ax.set_ylim(-H/2, H/2)
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
ax.set_title("Laminar-orientiertes Surrogat: Jet hinter Lochscheibe im Rohr (fixes)")

# draw plate segments
ax.add_patch(patches.Rectangle((-t/2, opening_half), t, H/2 - opening_half,
                               linewidth=1.5, edgecolor="black", facecolor="lightcoral", alpha=0.85))
ax.add_patch(patches.Rectangle((-t/2, -H/2), t, H/2 - opening_half,
                               linewidth=1.5, edgecolor="black", facecolor="lightcoral", alpha=0.85))

def compute_field(phase: float):
    # very small modulation for animation (visual only)
    mod = 1.0 + 0.02*np.sin(phase)

    # Build u from base + downstream jet excess
    u = u_base + mod * jet_excess_u(X, Y)

    # Enforce constant flux for every x
    u = enforce_constant_flux(u, U1)

    # Build v consistent with u
    v = v_base + mod * v_from_u(u, x, y)

    # symmetry
    u, v = enforce_axis_symmetry(u, v)

    # mask the solid plate
    u_m = np.ma.array(u, mask=plate_mask)
    v_m = np.ma.array(v, mask=plate_mask)
    speed = np.ma.sqrt(u_m**2 + v_m**2)
    return u_m, v_m, speed

u_m, v_m, speed = compute_field(0.0)

# Background speed field (transparent so streamlines are clear)
im = ax.contourf(X, Y, speed, levels=28, alpha=0.70)
cbar = fig.colorbar(im, ax=ax)
cbar.set_label("|u| [m/s]")

stream = None

# Seeds inside pipe upstream (avoid walls)
wall_margin = 0.06 * H
ys = np.linspace(-H/2 + wall_margin, H/2 - wall_margin, 38)
start_points = np.column_stack([np.full_like(ys, -2.5*d0), ys])

def update(frame: int):
    global stream, im
    phase = frame * 0.10
    u_m, v_m, speed = compute_field(phase)

    # redraw background
    for coll in im.collections:
        coll.remove()
    im = ax.contourf(X, Y, speed, levels=28, alpha=0.70)

    # remove old streamlines
    if stream is not None:
        stream.lines.remove()
        stream.arrows.remove()

    u_f = np.ma.filled(u_m, 0.0)
    v_f = np.ma.filled(v_m, 0.0)

    # high-contrast streamlines
    stream = ax.streamplot(
        x, y, u_f, v_f,
        start_points=start_points,
        density=2.2,
        color="white",
        linewidth=1.2,
        arrowsize=1.0
    )

    ax.axhline(0.0, color="white", linewidth=0.6, alpha=0.35)
    ax.set_title(f"Laminar-orientiertes Surrogat (low-Re, fixes) | frame={frame}")
    return im.collections

ani = FuncAnimation(fig, update, frames=240, interval=40, blit=False)
plt.show()
