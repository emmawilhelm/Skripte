import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ----------------------------
# Physikalische Maße
# ----------------------------
D_pipe = 4.30
R = D_pipe / 2.0

d_hole = 2.00
r0 = d_hole / 2.0

L = 20.0                 # Länge des simulierten Abschnitts [m] (anpassbar)
x_orifice = 6.0          # Position der Scheibe [m]
plate_thickness = 0.10   # Dicke der Scheibe [m] (numerisch als mehrere Zellen)

rho = 1000.0             # Wasser [kg/m³]
nu = 1e-4                # kinematische Viskosität [m²/s] (laminar "freundlich" gewählt)
Umax = 0.6               # maximale Eintrittsgeschwindigkeit [m/s]

# ----------------------------
# Numerisches Gitter
# ----------------------------
Nx = 360
Nr = 140

x = np.linspace(0, L, Nx)
r = np.linspace(0, R, Nr)
dx = x[1] - x[0]
dr = r[1] - r[0]

# Stabilität (sehr konservativ)
dt = 0.25 * min(dx, dr)**2 / nu

# ----------------------------
# Felder: u = axial, v = radial, p = Druck
# ----------------------------
u = np.zeros((Nr, Nx), dtype=np.float64)
v = np.zeros((Nr, Nx), dtype=np.float64)
p = np.zeros((Nr, Nx), dtype=np.float64)

# ----------------------------
# Hilfsfunktionen: Ableitungen (zentral) + Randbehandlung separat
# ----------------------------
def ddx(f):
    g = np.zeros_like(f)
    g[:, 1:-1] = (f[:, 2:] - f[:, :-2]) / (2*dx)
    g[:, 0] = (f[:, 1] - f[:, 0]) / dx
    g[:, -1] = (f[:, -1] - f[:, -2]) / dx
    return g

def ddr(f):
    g = np.zeros_like(f)
    g[1:-1, :] = (f[2:, :] - f[:-2, :]) / (2*dr)
    g[0, :] = (f[1, :] - f[0, :]) / dr  # r=0 wird später symmetrisch korrigiert
    g[-1, :] = (f[-1, :] - f[-2, :]) / dr
    return g

def laplacian_axisym(f):
    """
    ∇²f = f_xx + f_rr + (1/r) f_r   (achsensymmetrisch, ohne Theta-Abhängigkeit)
    Achtung bei r=0: (1/r)f_r -> 0 im Grenzwert, wird separat behandelt.
    """
    f_xx = np.zeros_like(f)
    f_rr = np.zeros_like(f)

    f_xx[:, 1:-1] = (f[:, 2:] - 2*f[:, 1:-1] + f[:, :-2]) / dx**2
    f_xx[:, 0] = (f[:, 1] - f[:, 0]) / dx**2
    f_xx[:, -1] = (f[:, -2] - f[:, -1]) / dx**2

    f_rr[1:-1, :] = (f[2:, :] - 2*f[1:-1, :] + f[:-2, :]) / dr**2
    f_rr[0, :] = (f[1, :] - f[0, :]) / dr**2
    f_rr[-1, :] = (f[-2, :] - f[-1, :]) / dr**2

    f_r = ddr(f)
    term = np.zeros_like(f)
    term[1:, :] = (1.0 / r[1:, None]) * f_r[1:, :]  # ab r>0
    term[0, :] = 0.0  # Grenzwert r->0

    return f_xx + f_rr + term

# ----------------------------
# Geometrie: Orifice Plate Maske (solid=True => no-slip)
# ----------------------------
solid = np.zeros((Nr, Nx), dtype=bool)

# Rohrwand r=R (oberster Index): no-slip Rand (nicht als "solid" nötig, wir setzen u,v dort auf 0)
# Scheibe:
ix0 = np.argmin(np.abs(x - x_orifice))
plate_cells = max(1, int(round(plate_thickness / dx)))
ix1 = min(Nx-1, ix0 + plate_cells)

# In der Scheibenregion: r > r0 ist blockiert
for j in range(ix0, ix1):
    solid[:, j] = (r > r0)

# ----------------------------
# Randbedingungen
# ----------------------------
def apply_bc(u, v, p):
    # 1) Achse r=0: Symmetrie -> v=0, du/dr=0
    v[0, :] = 0.0
    u[0, :] = u[1, :]  # du/dr=0
    p[0, :] = p[1, :]

    # 2) Wand r=R: no-slip
    u[-1, :] = 0.0
    v[-1, :] = 0.0

    # 3) Inlet x=0: parabolisches Profil für u, v=0
    # u(r) = Umax * (1 - (r/R)^2)
    u[:, 0] = Umax * (1.0 - (r / R)**2)
    v[:, 0] = 0.0

    # 4) Outlet x=L: Null-Gradient (du/dx=0, dv/dx=0), p=0 als Referenz
    u[:, -1] = u[:, -2]
    v[:, -1] = v[:, -2]
    p[:, -1] = 0.0

    # 5) Orifice plate (solid): no-slip in den blockierten Zellen
    u[solid] = 0.0
    v[solid] = 0.0

# ----------------------------
# Druck-Poisson-Löser (SOR)
# ∇²p = (rho/dt) * div(u*)
# ----------------------------
def solve_pressure_poisson(p, rhs, iters=200, omega=1.7):
    for _ in range(iters):
        p_old = p.copy()

        # Innenpunkte (ohne Ränder)
        for i in range(1, Nr-1):
            ri = r[i]
            ar = 1.0/dr**2
            ax = 1.0/dx**2
            br = 1.0/(2*dr) * (1.0/ri) if ri > 0 else 0.0

            # Diskret: p_rr + (1/r)p_r + p_xx = rhs
            # p_rr ~ (p[i+1]-2p[i]+p[i-1])/dr^2
            # (1/r)p_r ~ (1/r)*(p[i+1]-p[i-1])/(2dr)
            # p_xx ~ (p[:,j+1]-2p[:,j]+p[:,j-1])/dx^2

            # wir iterieren über j
            for j in range(1, Nx-1):
                if solid[i, j]:
                    continue  # in der Scheibe p egal; wir lassen es stehen
                coef_center = -2*(ar + ax)
                val = (
                    ar*(p[i+1, j] + p[i-1, j]) +
                    br*(p[i+1, j] - p[i-1, j]) +
                    ax*(p[i, j+1] + p[i, j-1]) -
                    rhs[i, j]
                ) / (-coef_center)

                p[i, j] = (1-omega)*p[i, j] + omega*val

        # Druck-Randbedingungen (Neumann überall außer outlet Referenz)
        p[0, :] = p[1, :]
        p[-1, :] = p[-2, :]
        p[:, 0] = p[:, 1]
        p[:, -1] = 0.0

        # optional: Abbruch, wenn kaum Änderung
        if np.max(np.abs(p - p_old)) < 1e-5:
            break
    return p

# ----------------------------
# Zeitintegration (Projection Method)
# ----------------------------
def step(u, v, p):
    apply_bc(u, v, p)

    # Konvektion
    u_x = ddx(u)
    u_r = ddr(u)
    v_x = ddx(v)
    v_r = ddr(v)

    # Divergenz in Zylinderkoordinaten (axi):
    # div = du/dx + (1/r) d(r v)/dr
    rv = r[:, None] * v
    drv_dr = ddr(rv)
    div = u_x + np.where(r[:, None] > 0, (1.0/r[:, None]) * drv_dr, 0.0)
    div[0, :] = u_x[0, :] + 2.0 * (v[1, :] - v[0, :]) / dr  # Grenzwert r->0 (heuristisch stabil)

    # Diffusion
    Lu = laplacian_axisym(u)
    Lv = laplacian_axisym(v)

    # Zwischenwerte u*, v*
    u_star = u + dt * (- (u*u_x + v*u_r) + nu*Lu)
    v_star = v + dt * (- (u*v_x + v*v_r) + nu*Lv)

    # Solid enforced
    u_star[solid] = 0.0
    v_star[solid] = 0.0

    # rhs für Druck
    u_star_x = ddx(u_star)
    rv_star = r[:, None] * v_star
    drvstar_dr = ddr(rv_star)
    div_star = u_star_x + np.where(r[:, None] > 0, (1.0/r[:, None]) * drvstar_dr, 0.0)
    div_star[0, :] = u_star_x[0, :] + 2.0 * (v_star[1, :] - v_star[0, :]) / dr

    rhs = (rho/dt) * div_star

    # Druck lösen
    p = solve_pressure_poisson(p, rhs, iters=120, omega=1.7)

    # Geschwindigkeiten korrigieren
    p_x = ddx(p)
    p_r = ddr(p)

    u_new = u_star - (dt/rho) * p_x
    v_new = v_star - (dt/rho) * p_r

    # Randbedingungen nochmal
    apply_bc(u_new, v_new, p)

    return u_new, v_new, p

# ----------------------------
# Animation / Plot
# ----------------------------
steps_per_frame = 8
frames = 250

fig, ax = plt.subplots(figsize=(10, 3.2))
ax.set_title("Achsensymmetrisch: Rohrströmung durch Lochscheibe (Orifice)")
ax.set_xlabel("x [m]")
ax.set_ylabel("r [m]")

# Darstellung nur in x-r; zur besseren Optik: extent in physikalischen Einheiten
extent = [x.min(), x.max(), r.min(), r.max()]
speed0 = np.zeros((Nr, Nx))
im = ax.imshow(speed0, origin="lower", extent=extent, aspect="auto")

# Orifice-Overlay: zeichne blockierten Bereich
# wir zeigen die Scheibenregion als schwarze Fläche
mask_show = np.where(solid, 1.0, np.nan)
ax.imshow(mask_show, origin="lower", extent=extent, aspect="auto", alpha=0.35)

plt.tight_layout()

def update(frame):
    global u, v, p
    for _ in range(steps_per_frame):
        u, v, p = step(u, v, p)

    speed = np.sqrt(u*u + v*v)
    speed[solid] = np.nan
    im.set_data(speed)
    vmax = np.nanmax(speed)
    im.set_clim(0.0, vmax*0.95 if np.isfinite(vmax) and vmax > 0 else 1.0)
    return (im,)

ani = FuncAnimation(fig, update, frames=frames, interval=30, blit=True)
plt.show()
