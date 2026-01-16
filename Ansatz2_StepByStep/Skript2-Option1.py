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


# -----------------------------
# OPTION 1: Streamlines mit hartem Sprung (v=0 gesetzt)
# - u(x,y) = Q/A(x), unabhängig von y (Plug im freien Querschnitt)
# - v(x,y) = 0
# - psi wird nur zur Konsistenz gesetzt: psi = u(x)*y
#   (an Sprungstellen ist dpsi/dx nicht definiert -> v wird explizit 0 gehalten)
# - Streamlines werden in Wandnähe maskiert (eps), damit keine Linie die Wand berührt.
# -----------------------------
nx, ny = 350, 180
x_grid = np.linspace(0.0, L, nx)
y_grid = np.linspace(-R, R, ny)
Xg, Yg = np.meshgrid(x_grid, y_grid)

# Sicherheitsabstand zu Wänden (damit Streamlines die Wand nicht berühren)
eps = 0.02  # [m] (kannst du bei Bedarf kleiner/größer setzen)

def u_piecewise(x):
    # harter Sprung
    A = np.pi * (d**2) / 4.0 if (x0 <= x <= x1) else np.pi * (D**2) / 4.0
    return fluid.Q / A

# u(x) als 2D-Feld (nur x-abhängig)
u1d = np.array([u_piecewise(x) for x in x_grid])
Ug = np.tile(u1d, (ny, 1))

# v = 0 überall im Fluidgebiet
Vg = np.zeros_like(Ug)

# psi optional (nicht zwingend für streamplot benötigt, aber wie von dir gewünscht)
# psi(x,y) = u(x) * y
psi = Ug * Yg

# Fluidmaske:
# - außerhalb Rohr: |y| >= R-eps -> maskieren
# - im Messblendenbereich: nur |y| <= r_or-eps ist Fluid; darüber Platte -> maskieren
fluid_mask = (np.abs(Yg) < (R - eps))  # innen im Rohr
orifice_region = (Xg >= x0) & (Xg <= x1)
fluid_mask_orifice = orifice_region & (np.abs(Yg) < (r_or - eps))

# Gesamte Fluidmaske: im Orifice-Bereich gilt engere Bedingung
fluid_mask = (fluid_mask & ~orifice_region) | fluid_mask_orifice

# Maskieren: außerhalb Fluidgebiet keine Werte -> keine Streamlines
Ug_plot = np.where(fluid_mask, Ug, np.nan)
Vg_plot = np.where(fluid_mask, Vg, np.nan)

# Streamlines zeichnen
ax.streamplot(
    x_grid, y_grid,
    Ug_plot, Vg_plot,
    density=1.2,      # Linienstreuung (nach Geschmack)
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