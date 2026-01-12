"""
Interaktive 2D-Darstellung (Surrogat, kein CFD) für Strömung durch eine Lochscheibe
im Rohr – angepasst für sehr kleine Reynolds-Zahl (hier Re≈O(1..10)).

Ziel: Jet-Ausdehnung in x basierend auf einer literaturkonformen Skalierung für
laminar-viskos dominierte Jet-Aufweitung:
Transversale Diffusions-/Grenzschichtskala:  δ(x) ~ sqrt(ν x / U_c)
(diese Skalierung folgt aus Grenzschicht-/Ähnlichkeitsansätzen für laminare Jets). citeturn1search9

Konsequenz bei kleinen Re:
- Keine ausgeprägten turbulenten Mischschichten/Rezirkulationsblasen wie bei hohen Re.
- Der Jet glättet/spreizt durch Viskosität relativ schnell und füllt im Rohr nach einigen Metern wieder den Querschnitt.

Hinweise (Hintergrund):
- Low-Re Orifice-Flow ist laminar und zeigt die Relevanz des laminar-regimes. citeturn0search11
- Vena contracta: engster Strahlquerschnitt leicht stromab der Öffnung. citeturn0search8
- Vena-contracta-Effekt hängt von Re und Geometrie ab. citeturn0search0

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
D   = 4.30    # [m] Rohrdurchmesser
Q   = 0.0097  # [m^3/s] Volumenstrom
d0  = 2.61    # [m] Lochdurchmesser (in 2D als "Öffnungshöhe" interpretiert)
t   = 0.8     # [m] Plattendicke (nur Darstellung/Maskierung)
zeta= 10.0    # [-]
# ------------------------------------------------

# ---------- Derived scalars ----------
A1 = np.pi * D**2 / 4.0
A0 = np.pi * d0**2 / 4.0
U1 = Q / A1                 # [m/s] Bulk im Rohr
U0 = Q / A0                 # [m/s] mittlere Durchtrittsgeschwindigkeit (3D)
nu = mu / rho               # [m^2/s] kinem. Viskosität
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
    u_flip = np.flipud(u)
    v_flip = np.flipud(v)
    u_sym = 0.5 * (u + u_flip)
    v_sym = 0.5 * (v - v_flip)
    return u_sym, v_sym

# ---------- Laminar jet model (formula-based surrogate) ----------
# Excess velocity behind the orifice: Gaussian in y with width b(x).
#
# Viscous spreading (smooth pragmatic form):
#   b(x) = sqrt(b0^2 + 4*nu*x/U0) for x>0
# -> b ~ sqrt(nu*x/U0) for larger x  (laminar diffusion scaling).
#
# Amplitude chosen so the *excess* 2D flow (per unit depth) stays constant:
#   ∫(u_excess) dy = A(x)*b(x)*sqrt(pi) = const -> A(x)=A0*b0/b(x)
#
b0 = (d0/2.0) * 0.65           # [m] effective initial half-width (vena contracta-like)
A0_excess = max(U0 - U1, 0.0)  # [m/s] initial excess speed scale

def b_of_x(xpos: np.ndarray):
    return np.sqrt(b0*b0 + 4.0*nu*xpos/max(U0, 1e-12))

def jet_excess_u(X: np.ndarray, Y: np.ndarray):
    xpos = np.maximum(X, 0.0)
    b = b_of_x(xpos)
    A = A0_excess * (b0 / np.maximum(b, 1e-12))
    return A * np.exp(-(Y / np.maximum(b, 1e-12))**2)

def v_from_u(u: np.ndarray, x: np.ndarray, y: np.ndarray):
    """Approximate incompressibility in 2D: du/dx + dv/dy = 0, with v(y=0)=0."""
    du_dx = np.gradient(u, x, axis=1)
    v = np.zeros_like(u)
    i0 = int(np.argmin(np.abs(y)))
    dy = y[1]-y[0]
    for i in range(i0+1, len(y)):
        v[i,:] = v[i-1,:] + (-0.5*(du_dx[i,:]+du_dx[i-1,:]))*dy
    for i in range(i0-1, -1, -1):
        v[i,:] = v[i+1,:] - (-0.5*(du_dx[i,:]+du_dx[i+1,:]))*dy
    return v

# ---------- visualization setup ----------
fig, ax = plt.subplots(figsize=(12, 4.8))
ax.set_xlim(-L_up, L_dn)
ax.set_ylim(-H/2, H/2)
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
ax.set_title("Laminar-orientiertes Surrogat: Jet hinter Lochscheibe im Rohr")

ax.add_patch(patches.Rectangle((-t/2, opening_half), t, H/2-opening_half,
                               linewidth=1.5, edgecolor="black", facecolor="lightcoral", alpha=0.85))
ax.add_patch(patches.Rectangle((-t/2, -H/2), t, H/2-opening_half,
                               linewidth=1.5, edgecolor="black", facecolor="lightcoral", alpha=0.85))

def compute_field(phase: float):
    # very small modulation (visual only; keep laminar feel)
    mod = 1.0 + 0.03*np.sin(phase)

    u = u_base + mod * jet_excess_u(X, Y)
    v = v_base + mod * v_from_u(u, x, y)

    u, v = enforce_axis_symmetry(u, v)

    u_m = np.ma.array(u, mask=plate_mask)
    v_m = np.ma.array(v, mask=plate_mask)
    speed = np.ma.sqrt(u_m**2 + v_m**2)
    return u_m, v_m, speed

u_m, v_m, speed = compute_field(0.0)

# transparent background so streamlines are clearly visible
im = ax.contourf(X, Y, speed, levels=25, alpha=0.70)
cbar = fig.colorbar(im, ax=ax)
cbar.set_label("|u| [m/s]")

stream = None

# Seeds inside pipe upstream (no wall seeding)
wall_margin = 0.06 * H
ys = np.linspace(-H/2 + wall_margin, H/2 - wall_margin, 34)
start_points = np.column_stack([np.full_like(ys, -2.2*d0), ys])

def update(frame: int):
    global stream, im
    phase = frame * 0.10
    u_m, v_m, speed = compute_field(phase)

    for coll in im.collections:
        coll.remove()
    im = ax.contourf(X, Y, speed, levels=25, alpha=0.70)

    if stream is not None:
        stream.lines.remove()
        stream.arrows.remove()

    u_f = np.ma.filled(u_m, 0.0)
    v_f = np.ma.filled(v_m, 0.0)

    stream = ax.streamplot(
        x, y, u_f, v_f,
        start_points=start_points,
        density=2.2,
        color="white",
        linewidth=1.2,
        arrowsize=1.0
    )

    ax.axhline(0.0, color="white", linewidth=0.6, alpha=0.35)
    ax.set_title(f"Laminar-orientiertes Surrogat (low-Re) | frame={frame}")
    return im.collections

ani = FuncAnimation(fig, update, frames=240, interval=40, blit=False)
plt.show()
