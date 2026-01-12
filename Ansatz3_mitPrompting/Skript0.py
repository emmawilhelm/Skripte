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

# Orifice-Platte als Blockierelement mit Loch:
x0 = x_or - t / 2.0
x1 = x_or + t / 2.0

# -----------------------------
# Plot
# -----------------------------
fig, ax = plt.subplots(figsize=(12, 4))

# Rohrwände (2D-Schnitt, symmetrisch um y=0, beide Seiten anzeigen)
ax.plot([0, L], [ R,  R], linewidth=2)
ax.plot([0, L], [-R, -R], linewidth=2)

# Achse (Symmetrieachse) als gestrichelte Linie
ax.plot([0, L], [0, 0], linestyle="--", linewidth=1)

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
ax.set_xlim(-0.2, L + 0.2)
ax.set_ylim(-R*1.15, R*1.15)
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
ax.text(0.02, 0.98, info, transform=ax.transAxes, ha="left", va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

plt.tight_layout()
plt.show()