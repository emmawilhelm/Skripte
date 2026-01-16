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
ax.set_title("2D-Schnitt (achsensymmetrisch) – Rohr + Messblende + newtonsches Fluid + Geschwindigkeitsfeld +Stromlinien")
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


# -----------------------------
# OPTION 2: Streamlines mit geglättetem u(x) (Δx = 0.05 m)
# - u(x) wird an x0 und x1 weich überblendet, damit v = -dpsi/dx definiert ist
# - psi(x,y) = u(x)*y
# - u,v werden aus psi abgeleitet: u = dpsi/dy, v = -dpsi/dx
# - Streamlines werden in Wandnähe maskiert (eps), damit keine Linie die Wand berührt.
# -----------------------------
nx, ny = 350, 180
x_grid = np.linspace(0.0, L, nx)
y_grid = np.linspace(-R, R, ny)
Xg, Yg = np.meshgrid(x_grid, y_grid)

# Sicherheitsabstand zu Wänden
eps = 0.02  # [m]

# Glättungsbreite
dx_smooth = 0.05  # [m] wie von dir vorgegeben

A_pipe = np.pi * (D**2) / 4.0
A_orif = np.pi * (d**2) / 4.0
u_pipe = fluid.Q / A_pipe
u_orif = fluid.Q / A_orif

def smooth_step(x, x_center, width):
    """
    Glatter Übergang 0->1 via tanh.
    width ~ Übergangshalbbreite; größer -> weicher.
    """
    return 0.5 * (1.0 + np.tanh((x - x_center) / width))

# u(x): von Rohr zu Orifice bei x0, zurück bei x1
# s0 steigt bei x0 von 0->1, s1 steigt bei x1 von 0->1
s0 = smooth_step(x_grid, x0, dx_smooth)
s1 = smooth_step(x_grid, x1, dx_smooth)

# In Orifice-Plateau: s_plateau ~ 1 zwischen x0 und x1, ~0 außerhalb
s_plateau = np.clip(s0 - s1, 0.0, 1.0)

# u(x) geglättet
u1d = u_pipe + (u_orif - u_pipe) * s_plateau  # 1D

# Stromfunktion psi(x,y) = u(x)*y
psi = np.outer(y_grid, u1d)  # shape (ny, nx)

# Ableitungen (numerisch) -> u,v
dy = y_grid[1] - y_grid[0]
dx = x_grid[1] - x_grid[0]

# np.gradient gibt Ableitungen entlang Achsen:
dpsi_dy, dpsi_dx = np.gradient(psi, dy, dx, edge_order=2)

Ug = dpsi_dy            # u = ∂psi/∂y
Vg = -dpsi_dx           # v = -∂psi/∂x

# Fluidmaske (inkl. "nicht an Wand berühren")
fluid_mask = (np.abs(Yg) < (R - eps))
orifice_region = (Xg >= x0) & (Xg <= x1)
fluid_mask_orifice = orifice_region & (np.abs(Yg) < (r_or - eps))
fluid_mask = (fluid_mask & ~orifice_region) | fluid_mask_orifice

Ug_plot = np.where(fluid_mask, Ug, np.nan)
Vg_plot = np.where(fluid_mask, Vg, np.nan)

# -----------------------------
# Farbfeld: laminares parabolisches Profil im Querschnitt
# (Streamlines bleiben aus Ug_plot/Vg_plot wie bisher!)
# -----------------------------
# mittlere Geschwindigkeit vm(x) = Q/A(x) -> hier u1d (geglättet)
# maximale Geschwindigkeit vmax(x) = 2*vm(x)
vmax_1d = 2.0 * u1d  # u1d ist bereits geglättet und entspricht vm(x)

# freier Radius ya(x): im Rohr R, im Orifice r_or, geglättet passend zu s_plateau
ya_1d = R + (r_or - R) * s_plateau  # geglättet wie dein u1d

Vmax = np.tile(vmax_1d, (ny, 1))
Ya   = np.tile(ya_1d,   (ny, 1))

# Parabolisches Profil: u_par(x,y) = vmax(x) * (1 - (y/ya(x))^2)
u_par = Vmax * (1.0 - (Yg / Ya)**2)
u_par = np.clip(u_par, 0.0, None)

# Maskieren (damit keine Farbe an Wänden / in der Platte erscheint)
u_par_plot = np.where(fluid_mask, u_par, np.nan)

# Feste Farbskala:
# ACHTUNG: Für Parabel ist der Maximalwert 2*vm -> also vmax im Orifice
vmax_orif = 2.0 * u_orif

c = ax.pcolormesh(
    x_grid, y_grid, u_par_plot,
    shading="auto",
    cmap="viridis",
    vmin=0.0,
    vmax=vmax_orif,
    alpha=0.75,
    zorder=1
)

cb = plt.colorbar(c, ax=ax, pad=0.02)
cb.set_label(r"$|\vec{u}|$ [m/s]")


ax.streamplot(
    x_grid, y_grid,
    Ug_plot, Vg_plot,
    density=1.2,
    linewidth=1.0,
    arrowsize=1.0,
    zorder=12
)

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