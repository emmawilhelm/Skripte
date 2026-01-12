import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# -----------------------------
# Geometrieparameter (in m)
# -----------------------------
D = 4.30     # Rohrdurchmesser
L = 6.00     # Rohrlänge (Ausschnitt)
x_or = 3.00  # Position der Messblende (Mitte)
d = 2.61     # Öffnungsdurchmesser (Orifice)
t = 0.80     # Wanddicke / Plattendicke der Störstelle

R = D / 2.0          # Rohrradius
r_or = d / 2.0       # Orifice-Radius
s = 0.15        # Rohrwandstärke in m
R_out = R + s

# Orifice-Platte als Blockierelement mit Loch:
x0 = x_or - t / 2.0
x1 = x_or + t / 2.0

from dataclasses import dataclass

# -----------------------------
# Medium: newtonsches Fluid (nur Definition, keine Strömungsberechnung)
# -----------------------------
@dataclass(frozen=True)
class NewtonianFluid:
    name: str
    rho: float       # Dichte [kg/m^3]
    mu: float        # dyn. Viskosität [Pa*s]
    Q: float         # Volumenstrom [m^3/s]
    p_in: float      # Druck bei x=0 [Pa]

# Gegebene Fluiddaten
fluid = NewtonianFluid(
    name="Newtonian fluid",
    rho=1440.0,
    mu=1.0,
    Q=0.0097,
    p_in=4e5  # 4 bar = 4e5 Pa
)

# Optional: kurze Konsistenzchecks (keine Physik, nur Eingaben plausibel)
if fluid.rho <= 0 or fluid.mu <= 0 or fluid.Q <= 0 or fluid.p_in <= 0:
    raise ValueError("Fluidparameter müssen > 0 sein.")

# -----------------------------
# Plot
# -----------------------------
fig, ax = plt.subplots(figsize=(12, 4))

# Rohrwände (2D-Schnitt, symmetrisch um y=0, beide Seiten anzeigen)
from matplotlib.patches import Rectangle

# Rohrwand (oben/unten) als gefüllte Rechtecke
pipe_color = "0.85"

# obere Wand: von y=R bis y=R_out
ax.add_patch(Rectangle((0, R), L, s, facecolor=pipe_color, edgecolor="k", linewidth=1))

# untere Wand: von y=-R_out bis y=-R
ax.add_patch(Rectangle((0, -R_out), L, s, facecolor=pipe_color, edgecolor="k", linewidth=1))

# Innenkontur optional noch als Linie (macht's klarer)
ax.plot([0, L], [ R,  R], color="k", linewidth=1)
ax.plot([0, L], [-R, -R], color="k", linewidth=1)

# Außenkontur optional
ax.plot([0, L], [ R_out,  R_out], color="k", linewidth=1)
ax.plot([0, L], [-R_out, -R_out], color="k", linewidth=1)

# Achse (Symmetrieachse) als gestrichelte Linie
ax.plot([0, L], [0, 0], linestyle="--", linewidth=1)

# -----------------------------
# Fluid im Rohr (statisch, vollständig gefüllt)
# -----------------------------
fluid_color = "#cfe8ff"  # helles Blau

ax.add_patch(
    Rectangle(
        (0, -R), L, 2 * R,
        facecolor=fluid_color,
        edgecolor="none",
        zorder=0
    )
)

# Messblende: zwei Rechtecke (oben/unten), die den Querschnitt blockieren,
# mit einer runden Öffnung (im 2D-Schnitt: "Schlitz" von -r_or bis +r_or)
plate_color = "0.75"

# Oberer blockierter Bereich
upper_plate = Rectangle((x0, r_or), width=(x1 - x0), height=(R - r_or),
                        facecolor=plate_color, edgecolor="k", linewidth=1)
# Unterer blockierter Bereich
lower_plate = Rectangle((x0, -R), width=(x1 - x0), height=(R - r_or),
                        facecolor=plate_color, edgecolor="k", linewidth=1)

ax.add_patch(upper_plate)
ax.add_patch(lower_plate)

# Hilfslinien/Annotationen (optional, zur Orientierung)
ax.axvline(x_or, linestyle=":", linewidth=1)
ax.text(x_or, R*1.03, "x = 3.0 m (Messblende)", ha="center", va="bottom")

# Grenzen & Darstellung
x_info = L + 0.6 * L   # Platz rechts neben dem Rohr
ax.set_xlim(-0.2, x_info)
ax.set_ylim(-R_out*1.15, R_out*1.15)
ax.set_aspect("equal", adjustable="box")
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m] (radiale Richtung, Schnitt durch Achse)")
ax.set_title("2D-Schnitt (achsensymmetrisch) – Rohr + Messblende (Orifice)")
ax.grid(True, linewidth=0.5, alpha=0.4)

# Kurzinfo im Plot
info = (
    f"D = {D:.2f} m, L = {L:.2f} m\n"
    f"Orifice bei x = {x_or:.2f} m\n"
    f"d = {d:.2f} m, t = {t:.2f} m"
)
ax.text(
    L + 0.1 * L, R_out,
    info,
    ha="left", va="top",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.9)
)

fluid_info = (
    f"Fluid: {fluid.name}\n"
    f"rho = {fluid.rho:.0f} kg/m³\n"
    f"mu = {fluid.mu:.2f} Pa·s\n"
    f"Q = {fluid.Q:.4f} m³/s\n"
    f"p(x=0) = {fluid.p_in/1e5:.2f} bar"
)
ax.text(
    L + 0.1 * L, -R_out,
    fluid_info,
    ha="left", va="bottom",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.9)
)

plt.tight_layout()
plt.show()