import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from dataclasses import dataclass

# -----------------------------
# Geometrieparameter (in m)
# -----------------------------
D = 4.30     # Rohr-Innendurchmesser
L = 6.00     # Rohrlänge (Ausschnitt)
x_or = 3.00  # Position der Messblende (Mitte)
d = 2.61     # Öffnungsdurchmesser (Orifice)
t = 0.80     # Dicke der Messblende

# Optik: Rohrwandstärke (nur Darstellung)
s = 0.15     # Rohrwandstärke [m]

R = D / 2.0
r_or = d / 2.0
R_out = R + s

# Orifice-Plattenbereich in x
x0 = x_or - t / 2.0
x1 = x_or + t / 2.0

# -----------------------------
# Medium: newtonsches Fluid (Definition)
# -----------------------------
@dataclass(frozen=True)
class NewtonianFluid:
    name: str
    rho: float       # Dichte [kg/m^3]
    mu: float        # dyn. Viskosität [Pa*s]
    Q: float         # Volumenstrom [m^3/s]
    p_in_gauge: float  # Relativdruck bei x=0 [Pa]

fluid = NewtonianFluid(
    name="Newtonian fluid",
    rho=1440.0,
    mu=1.0,
    Q=0.0097,
    p_in_gauge=4e5  # 4 bar relativ = 4e5 Pa
)

# -----------------------------
# Strömung (stationär, inkompressibel, Q konstant)
# -----------------------------
def diameter_at_x(x: float) -> float:
    """Sprunghafte Änderung des Durchmessers im Messblendenbereich."""
    return d if (x0 <= x <= x1) else D

def area_from_diameter(Dloc: float) -> float:
    return np.pi * (Dloc**2) / 4.0

def u_of_x(x: float) -> float:
    """u(x) = Q / A(x) (hier: Plug-Flow-Ansatz zur Visualisierung)."""
    return fluid.Q / area_from_diameter(diameter_at_x(x))

def reynolds_number(v: float, D_h: float) -> float:
    """Re = (rho * v * D) / mu (dynamische Viskosität gegeben)."""
    return (fluid.rho * v * D_h) / fluid.mu

# Kontrollwerte (Ausgabe)
v_pipe = fluid.Q / area_from_diameter(D)
v_orif = fluid.Q / area_from_diameter(d)
Re_pipe = reynolds_number(v_pipe, D)

print(f"Kontrolle v (Rohr, D={D:.2f} m):    {v_pipe:.6f} m/s  (Soll ~ 0.0007)")
print(f"Kontrolle v (Orifice, d={d:.2f} m): {v_orif:.6f} m/s  (Soll ~ 0.0027)")
print(f"Kontrolle Re (Rohr, D={D:.2f} m):   {Re_pipe:.2f}     (Soll ~ 4.33)")

# -----------------------------
# Plot-Setup
# -----------------------------
fig, ax = plt.subplots(figsize=(13, 4))

# Platz rechts für Infoboxen schaffen
x_info = L + 0.6 * L
ax.set_xlim(-0.2, x_info)
ax.set_ylim(-R_out * 1.15, R_out * 1.15)
ax.set_aspect("equal", adjustable="box")
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m] (radiale Richtung, Schnitt durch Achse)")
ax.set_title("2D-Schnitt (achsensymmetrisch) – Rohr + Messblende + Fluid + Geschwindigkeitsvektoren")
ax.grid(True, linewidth=0.5, alpha=0.4)

# -----------------------------
# Rohrwand als Fläche (oben/unten)
# -----------------------------
pipe_color = "0.85"
ax.add_patch(Rectangle((0, R),     L, s, facecolor=pipe_color, edgecolor="k", linewidth=1, zorder=3))
ax.add_patch(Rectangle((0, -R_out), L, s, facecolor=pipe_color, edgecolor="k", linewidth=1, zorder=3))

# Innen- und Außenkontur
ax.plot([0, L], [ R,  R], color="k", linewidth=1, zorder=4)
ax.plot([0, L], [-R, -R], color="k", linewidth=1, zorder=4)
ax.plot([0, L], [ R_out,  R_out], color="k", linewidth=1, zorder=4)
ax.plot([0, L], [-R_out, -R_out], color="k", linewidth=1, zorder=4)

# Symmetrieachse
ax.plot([0, L], [0, 0], linestyle="--", linewidth=1, zorder=2)

# -----------------------------
# Fluid (statisch gefüllt)
# - Rohrbereiche: |y| <= R
# - Im Messblendenbereich ist nur |y| <= r_or "frei", der Rest ist Platte (kein Fluid)
# -----------------------------
fluid_color = "#cfe8ff"

# Fluid links von der Platte
ax.add_patch(Rectangle((0, -R), max(0.0, x0 - 0.0), 2 * R,
                       facecolor=fluid_color, edgecolor="none", zorder=0))

# Fluid in der Platte (nur Öffnung)
ax.add_patch(Rectangle((x0, -r_or), (x1 - x0), 2 * r_or,
                       facecolor=fluid_color, edgecolor="none", zorder=0))

# Fluid rechts von der Platte
ax.add_patch(Rectangle((x1, -R), max(0.0, L - x1), 2 * R,
                       facecolor=fluid_color, edgecolor="none", zorder=0))

# -----------------------------
# Messblende (Platte) als blockierter Bereich oben/unten
# -----------------------------
plate_color = "0.75"
upper_plate = Rectangle((x0, r_or), width=(x1 - x0), height=(R - r_or),
                        facecolor=plate_color, edgecolor="k", linewidth=1, zorder=5)
lower_plate = Rectangle((x0, -R), width=(x1 - x0), height=(R - r_or),
                        facecolor=plate_color, edgecolor="k", linewidth=1, zorder=5)
ax.add_patch(upper_plate)
ax.add_patch(lower_plate)

# Markierung der Messblendenmitte
ax.axvline(x_or, linestyle=":", linewidth=1, zorder=6)
ax.text(x_or, R_out * 1.03, "x = 3.0 m (Messblende)", ha="center", va="bottom", zorder=6)

# ============================================================
# Physikalisch begründetes quasi-2D Strömungsfeld (ohne CFD)
# ============================================================

# -----------------------------
# Modellparameter (Literatur)
# -----------------------------
C_c = 0.70
l_contr = 0.30 * D
l_exp   = 1.00 * D

# -----------------------------
# Effektiver Radius R_eff(x)
# -----------------------------
def smoothstep(x, a, b):
    xi = np.clip((x - a) / (b - a), 0.0, 1.0)
    return xi * xi * (3 - 2 * xi)

def R_eff(x):
    # Kontraktion beginnt vor der Platte
    x_c_start = x0 - l_contr
    x_c_end   = x0

    # Expansion nach der Platte
    x_e_start = x1
    x_e_end   = x1 + l_exp

    R_min = C_c * r_or

    if x < x_c_start:
        return R
    elif x_c_start <= x < x_c_end:
        s = smoothstep(x, x_c_start, x_c_end)
        return R - s * (R - R_min)
    elif x_c_end <= x <= x_e_start:
        return R_min
    elif x_e_start < x <= x_e_end:
        s = smoothstep(x, x_e_start, x_e_end)
        return R_min + s * (R - R_min)
    else:
        return R

# -----------------------------
# Gitter
# -----------------------------
nx, ny = 500, 200
xg = np.linspace(0.0, L, nx)
yg = np.linspace(-R, R, ny)
XX, YY = np.meshgrid(xg, yg)

# -----------------------------
# Geschwindigkeit u_x, u_r
# -----------------------------
U = np.zeros_like(XX)
V = np.zeros_like(XX)

for i in range(nx):
    Reff = R_eff(xg[i])
    Aeff = np.pi * Reff**2
    u_mean = fluid.Q / Aeff

    # Ableitung du_mean/dx (für u_r)
    if 0 < i < nx - 1:
        Reff_p = R_eff(xg[i+1])
        Reff_m = R_eff(xg[i-1])
        A_p = np.pi * Reff_p**2
        A_m = np.pi * Reff_m**2
        u_mean_p = fluid.Q / A_p
        u_mean_m = fluid.Q / A_m
        du_dx = (u_mean_p - u_mean_m) / (xg[1] - xg[0])
    else:
        du_dx = 0.0

    for j in range(ny):
        r = abs(yg[j])

        if r <= Reff:
            # Poiseuille-Profil
            U[j, i] = 2 * u_mean * (1 - (r / Reff)**2)

            # Radialgeschwindigkeit aus Kontinuität
            V[j, i] = -0.5 * yg[j] * du_dx
        else:
            U[j, i] = np.nan
            V[j, i] = np.nan

# -----------------------------
# Betrag der Geschwindigkeit
# -----------------------------
U_mag = np.sqrt(U**2 + V**2)

# -----------------------------
# Konvektive Beschleunigung
# -----------------------------
dx = xg[1] - xg[0]
dy = yg[1] - yg[0]

dU_dy, dU_dx = np.gradient(U, dy, dx)
dV_dy, dV_dx = np.gradient(V, dy, dx)

a_x = U * dU_dx + V * dU_dy
a_y = U * dV_dx + V * dV_dy
a_mag = np.sqrt(a_x**2 + a_y**2)

# -----------------------------
# Farbfelder plotten
# -----------------------------
vel_plot = ax.pcolormesh(
    XX, YY, U_mag,
    shading="gouraud",
    cmap="viridis",
    alpha=0.9,
    zorder=1
)

acc_plot = ax.pcolormesh(
    XX, YY, a_mag,
    shading="gouraud",
    cmap="inferno",
    alpha=0.6,
    zorder=2
)

# -----------------------------
# Stromlinien
# -----------------------------
ax.streamplot(
    xg, yg, U, V,
    color="white",
    linewidth=0.9,
    density=1.4,
    arrowsize=1.0,
    zorder=8
)

# -----------------------------
# Farbskalen
# -----------------------------
cbar1 = fig.colorbar(vel_plot, ax=ax, pad=0.02)
cbar1.set_label("|u| [m/s]")

cbar2 = fig.colorbar(acc_plot, ax=ax, pad=0.08)
cbar2.set_label("|a| [m/s²]")


# -----------------------------
# Infoboxen rechts neben dem Rohr
# -----------------------------
info_geom = (
    f"D = {D:.2f} m, L = {L:.2f} m\n"
    f"Orifice bei x = {x_or:.2f} m\n"
    f"d = {d:.2f} m, t = {t:.2f} m"
)
info_fluid = (
    f"Fluid: {fluid.name}\n"
    f"rho = {fluid.rho:.0f} kg/m³\n"
    f"mu  = {fluid.mu:.2f} Pa·s\n"
    f"Q   = {fluid.Q:.4f} m³/s\n"
    f"p(x=0) = {fluid.p_in_gauge/1e5:.2f} bar(rel)\n"
    f"Re (Rohr) ≈ {Re_pipe:.2f}"
)

ax.text(
    L + 0.1 * L, R_out,
    info_geom,
    ha="left", va="top",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
    zorder=20
)
ax.text(
    L + 0.1 * L, -R_out,
    info_fluid,
    ha="left", va="bottom",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
    zorder=20
)

plt.tight_layout()
plt.show()