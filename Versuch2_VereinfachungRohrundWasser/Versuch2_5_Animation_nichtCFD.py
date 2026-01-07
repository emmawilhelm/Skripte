import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# -----------------------------
# Parameter (Geometrie + Fluid)
# -----------------------------
g = 9.81
rho = 1000.0
nu = 1.0e-6  # Wasser ~20°C

D = 4.30
R = D / 2
d = 2.00
r0 = d / 2

L = 6.0       # 3m vor + 3m nach
x0 = 3.0      # Lochscheibe in der Mitte
plate_thickness = 0.10  # nur für Darstellung

A_pipe = math.pi * D**2 / 4
A_hole = math.pi * d**2 / 4

ks = 0.00005          # Rauheit [m] (z.B. 0.05 mm)
zeta_orifice = 0.6    # dein Schätzwert (als ζ verwendet)
zeta_ref = "pipe"     # "pipe" oder "hole" (Bezugsgeschwindigkeit für ζ)

# "Modellparameter" für Visualisierung des Jets:
mix_length = 1.5 * D  # Mischlänge: wie schnell Jet wieder "Rohrprofil" wird
spread_strength = 0.35 # Stärke der radialen Jet-Aufweitung (nur für Streamlines)
# Deflektion der Stromlinien um die Scheibe (nur Visualisierung)
deflect_len = 0.60      # [m] Einflusslänge vor/nach der Scheibe
deflect_strength = 0.35 # Stärke der seitlichen Ablenkung
edge_sigma = 0.18       # [m] "Weichheit" am Lochrand (Rundung)

# -----------------------------
# Reibungsbeiwert lambda (Colebrook-White + laminar fallback)
# -----------------------------
def reynolds(v, D, nu):
    return v * D / nu

def colebrook_white(Re, ks, D, tol=1e-10, maxit=80):
    if Re < 1e-12:
        return 0.0
    if Re < 2300:
        return 64.0 / Re  # laminar
    lam = 0.02
    for _ in range(maxit):
        rhs = -2.0 * math.log10(2.51/(Re*math.sqrt(lam)) + ks/(3.71*D))
        lam_new = 1.0 / (rhs*rhs)
        if abs(lam_new - lam) < tol:
            return lam_new
        lam = lam_new
    return lam

# -----------------------------
# Q(t) Vorgabe (DEIN "Antrieb")
# -----------------------------
def Q_of_t(t):
   return 10.0  # m^3/s (konstanter Volumenstrom)
# -----------------------------
# 1D-Formelmodell: p(x) aus Verlusten
# -----------------------------
def compute_1d(Q):
    v_pipe = Q / A_pipe
    v_hole = Q / A_hole

    Re = reynolds(v_pipe, D, nu)
    lam = colebrook_white(Re, ks, D)

    # Reibungsverlust pro Länge als Druckgradient:
    # dp/dx = rho * (lambda/D) * (v^2/2)
    dpdx = rho * (lam / D) * (v_pipe**2 / 2.0)  # Pa/m

    # Orifice Drucksprung:
    v_ref = v_hole if zeta_ref == "hole" else v_pipe
    dp_orif = rho * zeta_orifice * (v_ref**2 / 2.0)  # Pa

    return v_pipe, v_hole, lam, Re, dpdx, dp_orif

# -----------------------------
# Visualisierungsfeld u(x,r), v(x,r)
# -----------------------------
def poiseuille_profile(v_mean, r_abs):
    # Für laminar: u(r) = 2*v_mean*(1-(r/R)^2)
    return 2.0 * v_mean * (1.0 - (r_abs/R)**2)

def jet_profile(v_hole_mean, r_abs):
    # Einfacher "Top-hat" Jet im Loch + weiche Kante:
    # innen ~ v_hole_mean, außen ~ 0
    # glätte Kante mit tanh
    eps = 0.04 * R
    return 0.5 * v_hole_mean * (1.0 - np.tanh((r_abs - r0)/eps))

def build_field(x, y_full, v_pipe, v_hole):
    """
    y_full: [-R..+R], r_abs = |y|
    upstream: Poiseuille
    at/downstream: blend( Jet -> Poiseuille ) über Mischlänge
    radial velocity (für Streamlines): modellierte Aufweitung hinter dem Loch
    """
    r_abs = np.abs(y_full)          # (Ny,)
    r_abs2d = r_abs[:, None]        # (Ny,1)
    X = x[None, :]                  # (1,Nx)

    s = np.clip(X - x0, 0.0, None)
    alpha = 1.0 - np.exp(-s / mix_length)

    U_poi = poiseuille_profile(v_pipe, r_abs2d)
    U_jet = jet_profile(v_hole, r_abs2d)

    upstream = (X < x0).astype(float)
    downstream = 1.0 - upstream

    U = upstream * U_poi + downstream * ((1.0 - alpha) * U_jet + alpha * U_poi)

   # -------------------------
    # Radialkomponente V: "Ausweichen" an der Lochscheibe + Jet-Aufweitung
    # -------------------------
    y = y_full[:, None]          # (Ny,1)
    r = r_abs2d                  # (Ny,1)

    # 1) Jet-Aufweitung downstream (wie bisher)
    sign = np.sign(y_full)[:, None]
    V_jet = downstream * (spread_strength * (1.0 - alpha)) * sign * (r / R) * U_jet * 0.15

    # 2) Deflektion um die Scheibe herum (vor + nach der Scheibe)
    #    - wirkt nur außerhalb des Lochs (|y| > r0)
    #    - lenkt Strömung sanft weg vom blockierten Bereich
    X2 = X  # (1,Nx)
    gauss_x = np.exp(-((X2 - x0) / deflect_len)**2)   # lokal um x0

    # "Lochrand" weich machen: Übergang um r0 mit tanh
    # near_hole ~ 0 im Loch, ~1 außerhalb
    near_hole = 0.5 * (1.0 + np.tanh((r - r0) / edge_sigma))

    # Richtung: oben nach oben, unten nach unten
    V_deflect = deflect_strength * gauss_x * near_hole * sign * np.maximum(U_poi, 0.0) * 0.35

    V = V_jet + V_deflect


    # Maskierung (hier jetzt korrekt, weil r_abs 1D ist)
    U[r_abs > R, :] = 0.0
    V[r_abs > R, :] = 0.0

    return U, V


def pressure_profile(x, dpdx, dp_orif):
    """
    p(x) relativ: linearer Abfall + Sprung an x0.
    Wir setzen p(0)=0 und zeigen nur relative Werte (für Plot reicht das).
    """
    p = -dpdx * x
    p = p - (x >= x0) * dp_orif
    return p

# -----------------------------
# Grid für Darstellung (voller Durchmesser)
# -----------------------------
Nx = 260
Ny_half = 120
x = np.linspace(0.0, L, Nx)
y_half = np.linspace(0.0, R, Ny_half)
y_full = np.concatenate((-y_half[:0:-1], y_half))  # -R..R ohne doppelte 0

extent = [x.min(), x.max(), y_full.min(), y_full.max()]

# -----------------------------
# Plot Setup
# -----------------------------
fig = plt.figure(figsize=(11, 5))
gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.15)

ax = fig.add_subplot(gs[0])
axp = fig.add_subplot(gs[1], sharex=ax)

ax.set_title("Formelbasiertes Modell: Orifice im Rohr (voller Querschnitt + Streamlines)")
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")

axp.set_xlabel("x [m]")
axp.set_ylabel("p_rel [Pa]")

# Orifice Darstellung (grauer Balken außerhalb Loch)
# Wir zeichnen einfach ein Rechteckband, und lassen in der Mitte ein Loch frei
# (nur visuell, nicht als CFD-Gitter)
plate_x0 = x0 - plate_thickness/2
plate_x1 = x0 + plate_thickness/2

# Initial fields
t0 = 0.0
Q0 = Q_of_t(t0)
v_pipe0, v_hole0, lam0, Re0, dpdx0, dp_orif0 = compute_1d(Q0)
U0, V0 = build_field(x, y_full, v_pipe0, v_hole0)
speed0 = np.sqrt(U0**2 + V0**2)

im = ax.imshow(speed0, origin="lower", extent=extent, aspect="auto")
cbar = fig.colorbar(im, ax=ax, pad=0.01)
cbar.set_label("|v| [m/s]")

# Orifice overlay: blockierter Bereich (außer Loch)
# Wir bauen eine Maske in Plotkoordinaten
Y = y_full[:, None]
X = x[None, :]
solid_vis = (X >= plate_x0) & (X <= plate_x1) & (np.abs(Y) > r0)
mask_vis = np.where(solid_vis, 1.0, np.nan)
ax.imshow(mask_vis, origin="lower", extent=extent, aspect="auto",
          alpha=0.55, cmap="gray", vmin=0, vmax=1)

# Pressure line
p0 = pressure_profile(x, dpdx0, dp_orif0)
pline, = axp.plot(x, p0)

# Streamlines handle
stream = None
def init():
    global stream
    # einmalig initiale Streamlines erzeugen
    xs = x[::sx]
    ys = y_full[::sy]
    UU = U0[::sy, ::sx]
    VV = V0[::sy, ::sx]
    stream = ax.streamplot(xs, ys, UU, VV, density=1.2, linewidth=0.8, arrowsize=0.8)
    return (im, pline)

# decimation for streamplot
sx = 4
sy = 4

status_text = ax.text(0.01, 0.99, "", transform=ax.transAxes,
                      va="top", ha="left", color="white",
                      bbox=dict(facecolor="black", alpha=0.35, pad=4))

def update(frame):
    global stream

    t = frame * 0.12  # Zeitschritt für Animation (nur Darstellung)
    Q = Q_of_t(t)

    v_pipe, v_hole, lam, Re, dpdx, dp_orif = compute_1d(Q)

    U, V = build_field(x, y_full, v_pipe, v_hole)
    speed = np.sqrt(U**2 + V**2)

    im.set_data(speed)
    vmax = np.nanmax(speed)
    im.set_clim(0.0, max(1e-6, 0.95 * vmax))

    # Druckprofil aktualisieren
    p = pressure_profile(x, dpdx, dp_orif)
    pline.set_ydata(p)
    axp.relim()
    axp.autoscale_view()

    # alte Streamlines entfernen (Matplotlib: PatchCollection)
    if stream is not None:
     try:
        if stream.lines in ax.collections:
            stream.lines.remove()
     except Exception:
        pass
     try:
        if stream.arrows in ax.collections:
            stream.arrows.remove()
     except Exception:
        pass
    # -------- Festkörper-Maske: Stromlinien abbrechen an der Orifice-Scheibe --------
    Xg, Yg = np.meshgrid(x, y_full)

    solid_mask = (
    (Xg >= plate_x0) &
    (Xg <= plate_x1) &
    (np.abs(Yg) > r0)
    )

    # Kopien erzeugen, damit Originalfelder nicht verändert werden
    U = U.copy()
    V = V.copy()

    U[solid_mask] = np.nan
    V[solid_mask] = np.nan

    

   # Stromlinien nur am Einlass starten
    y_seeds = np.linspace(-R*0.98, R*0.98, 25)
    start_points = np.column_stack((np.full_like(y_seeds, x[0] + 1e-3), y_seeds))

    stream = ax.streamplot(
        x, y_full, U, V,
        start_points=start_points,
        linewidth=0.9,
        arrowsize=1.0,
        color="orange"
    )

    status_text.set_text(
        f"t={t:5.2f} s  |  Q={Q:6.2f} m³/s  |  v_pipe={v_pipe:5.2f} m/s  |  v_hole={v_hole:5.2f} m/s\n"
        f"Re={Re: .2e}  λ={lam: .4f}  Δp_orif≈{dp_orif: .0f} Pa  dp/dx≈{dpdx: .1f} Pa/m"
    )

    return (im, pline)

ani = FuncAnimation(fig, update, init_func=init, frames=300, interval=35, blit=False)
try:
    plt.show()
except KeyboardInterrupt:
    print("Animation vom Benutzer abgebrochen (Ctrl+C).")