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

# Gewünschter Ausschnitt: 3 m vor und 3 m nach der Störstelle
L = 6.0
x_orifice = 3.0
plate_thickness = 0.10  # [m]

rho = 1000.0
nu  = 1e-4              # (laminar-freundlich)
Umax = 0.6              # [m/s]

# ----------------------------
# Numerisches Gitter
# ----------------------------
Nx = 360
Nr = 140

x = np.linspace(0, L, Nx)
r = np.linspace(0, R, Nr)
dx = x[1] - x[0]
dr = r[1] - r[0]

dt = 0.25 * min(dx, dr)**2 / nu  # konservativ stabil

# Felder: u=axial, v=radial, p
u = np.zeros((Nr, Nx), dtype=np.float64)
v = np.zeros((Nr, Nx), dtype=np.float64)
p = np.zeros((Nr, Nx), dtype=np.float64)

# ----------------------------
# Ableitungen
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
    g[0, :] = (f[1, :] - f[0, :]) / dr
    g[-1, :] = (f[-1, :] - f[-2, :]) / dr
    return g

def laplacian_axisym(f):
    f_xx = np.zeros_like(f)
    f_rr = np.zeros_like(f)

    f_xx[:, 1:-1] = (f[:, 2:] - 2*f[:, 1:-1] + f[:, :-2]) / dx**2
    f_xx[:, 0]  = (f[:, 1] - f[:, 0]) / dx**2
    f_xx[:, -1] = (f[:, -2] - f[:, -1]) / dx**2

    f_rr[1:-1, :] = (f[2:, :] - 2*f[1:-1, :] + f[:-2, :]) / dr**2
    f_rr[0, :]  = (f[1, :] - f[0, :]) / dr**2
    f_rr[-1, :] = (f[-2, :] - f[-1, :]) / dr**2

    f_r = ddr(f)
    term = np.zeros_like(f)
    term[1:, :] = (1.0 / r[1:, None]) * f_r[1:, :]
    term[0, :]  = 0.0

    return f_xx + f_rr + term

# ----------------------------
# Geometrie: Orifice Plate Maske
# ----------------------------
solid = np.zeros((Nr, Nx), dtype=bool)

ix0 = np.argmin(np.abs(x - x_orifice))
plate_cells = max(1, int(round(plate_thickness / dx)))
ix1 = min(Nx-1, ix0 + plate_cells)

# Plattenregion: für r > r0 blockiert
solid[:, ix0:ix1] = (r[:, None] > r0)

# ----------------------------
# Randbedingungen
# ----------------------------
def apply_bc(u, v, p):
    # Achse r=0: Symmetrie
    v[0, :] = 0.0
    u[0, :] = u[1, :]
    p[0, :] = p[1, :]

    # Wand r=R: no-slip
    u[-1, :] = 0.0
    v[-1, :] = 0.0

    # Inlet x=0: parabolisch
    u[:, 0] = Umax * (1.0 - (r / R)**2)
    v[:, 0] = 0.0

    # Outlet x=L: Nullgradient, p als Referenz 0
    u[:, -1] = u[:, -2]
    v[:, -1] = v[:, -2]
    p[:, -1] = 0.0

    # Platte: no-slip
    u[solid] = 0.0
    v[solid] = 0.0

# ----------------------------
# Druck-Poisson (SOR)
# ----------------------------
def solve_pressure_poisson(p, rhs, iters=120, omega=1.7):
    for _ in range(iters):
        p_old = p.copy()
        for i in range(1, Nr-1):
            ri = r[i]
            ar = 1.0/dr**2
            ax = 1.0/dx**2
            br = 1.0/(2*dr) * (1.0/ri) if ri > 0 else 0.0

            for j in range(1, Nx-1):
                if solid[i, j]:
                    continue
                coef_center = -2*(ar + ax)
                val = (
                    ar*(p[i+1, j] + p[i-1, j]) +
                    br*(p[i+1, j] - p[i-1, j]) +
                    ax*(p[i, j+1] + p[i, j-1]) -
                    rhs[i, j]
                ) / (-coef_center)

                p[i, j] = (1-omega)*p[i, j] + omega*val

        # Neumann-Ränder + Outlet Referenz
        p[0, :] = p[1, :]
        p[-1, :] = p[-2, :]
        p[:, 0] = p[:, 1]
        p[:, -1] = 0.0

        if np.max(np.abs(p - p_old)) < 1e-5:
            break
    return p

# ----------------------------
# Zeitschritt (Projection)
# ----------------------------
def step(u, v, p):
    apply_bc(u, v, p)

    u_x = ddx(u); u_r = ddr(u)
    v_x = ddx(v); v_r = ddr(v)

    Lu = laplacian_axisym(u)
    Lv = laplacian_axisym(v)

    u_star = u + dt * (-(u*u_x + v*u_r) + nu*Lu)
    v_star = v + dt * (-(u*v_x + v*v_r) + nu*Lv)

    u_star[solid] = 0.0
    v_star[solid] = 0.0

    u_star_x = ddx(u_star)
    rv_star = r[:, None] * v_star
    drvstar_dr = ddr(rv_star)

    div_star = u_star_x + np.where(r[:, None] > 0, (1.0/r[:, None]) * drvstar_dr, 0.0)
    div_star[0, :] = u_star_x[0, :] + 2.0 * (v_star[1, :] - v_star[0, :]) / dr

    rhs = (rho/dt) * div_star
    p = solve_pressure_poisson(p, rhs)

    p_x = ddx(p)
    p_r = ddr(p)

    u_new = u_star - (dt/rho) * p_x
    v_new = v_star - (dt/rho) * p_r

    apply_bc(u_new, v_new, p)
    return u_new, v_new, p

# ----------------------------
# "Vollrohr"-Darstellung durch Spiegelung
# ----------------------------
def mirror_full_pipe(u, v, solid):
    # y_full = [-r ... 0 ... +r], ohne doppelte 0
    y_full = np.concatenate((-r[:0:-1], r))
    u_full = np.vstack((u[:0:-1, :], u))
    # radial v: im unteren Halbkreis zeigt "nach unten" => Vorzeichen flip
    v_full = np.vstack((-v[:0:-1, :], v))
    solid_full = np.vstack((solid[:0:-1, :], solid))
    return y_full, u_full, v_full, solid_full

# ----------------------------
# Plot / Animation (mit Streamlines)
# ----------------------------
steps_per_frame = 30
frames = 260

fig, ax = plt.subplots(figsize=(10.5, 4.0))
ax.set_title("Rohr (voll) + Streamlines: Strömung durch Lochscheibe (Orifice)")
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")  # jetzt -R..+R

# initial mirrored
y_full, u_full, v_full, solid_full = mirror_full_pipe(u, v, solid)
extent = [x.min(), x.max(), y_full.min(), y_full.max()]

speed_full = np.sqrt(u_full*u_full + v_full*v_full)
speed_full[solid_full] = np.nan
im = ax.imshow(speed_full, origin="lower", extent=extent, aspect="auto")

# Orifice Overlay im Vollrohr (grau)
mask_show = np.where(solid_full, 1.0, np.nan)
ax.imshow(mask_show, origin="lower", extent=extent, aspect="auto",
          alpha=0.6, cmap="gray", vmin=0, vmax=1)

# Streamlines: wir halten das Objekt, um es pro Frame zu entfernen/neu zu zeichnen
stream = None

# Ausdünnung fürs Streamplot (Performance)
sx = 6   # x step
sy = 3   # y step

plt.tight_layout()

def update(frame):
    global u, v, p, stream

    for _ in range(steps_per_frame):
        u, v, p = step(u, v, p)

    y_full, u_full, v_full, solid_full = mirror_full_pipe(u, v, solid)

    # Maskieren in Solids
    U = u_full.copy()
    V = v_full.copy()
    U[solid_full] = np.nan
    V[solid_full] = np.nan

    speed = np.sqrt(U*U + V*V)
    im.set_data(speed)

    # Farbskala: entweder dynamisch oder fix
    vmax = np.nanmax(speed)
    im.set_clim(0.0, vmax*0.95 if np.isfinite(vmax) and vmax > 0 else 1.0)

    # alte Streamlines entfernen
    if stream is not None:
        # streamplot liefert ein StreamplotSet mit .lines und .arrows
        stream.lines.remove()
        for a in stream.arrows:
            a.remove()

    # neue Streamlines (ausgedünnt)
    xs = x[::sx]
    ys = y_full[::sy]
    UU = U[::sy, ::sx]
    VV = V[::sy, ::sx]

    stream = ax.streamplot(xs, ys, UU, VV, density=1.2, linewidth=0.8, arrowsize=0.9)

    return (im,)

ani = FuncAnimation(fig, update, frames=frames, interval=30, blit=False)
plt.show()