import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Parameters (geometry + fluid)
# -----------------------------
rho = 1000.0
nu = 1.0e-6  # water ~20C

D = 4.30
R = D / 2.0

d = 2.00
r0 = d / 2.0

L = 6.0       # 3m upstream + 3m downstream
x0 = 3.0      # orifice plate center
plate_thickness = 0.10

Q = 0.01      # m^3/s
p_out = 4.0e5 # Pa (reference outlet pressure)

A_pipe = np.pi * D**2 / 4.0
U_in = Q / A_pipe

# -----------------------------
# Grid
# -----------------------------
Nx = 220
Ny = 120

x = np.linspace(0.0, L, Nx)
y = np.linspace(-R, R, Ny)
X, Y = np.meshgrid(x, y)

dx = x[1] - x[0]
dy = y[1] - y[0]

plate_x0 = x0 - plate_thickness / 2.0
plate_x1 = x0 + plate_thickness / 2.0

# Solid mask: plate blocks flow outside the hole
solid = (X >= plate_x0) & (X <= plate_x1) & (np.abs(Y) > r0)

# Streamfunction boundary values
psi_bottom = 0.0
psi_top = U_in * (2.0 * R)

# -----------------------------
# Helper functions
# -----------------------------

def enforce_psi_bc(psi):
    # Walls
    psi[0, :] = psi_bottom
    psi[-1, :] = psi_top

    # Inlet: uniform inflow
    psi[:, 0] = U_in * (y + R)

    # Outlet: zero-gradient
    psi[:, -1] = psi[:, -2]

    # Solid plate: psi constant (no-penetration)
    psi_solid = np.where(Y >= 0.0, psi_top, psi_bottom)
    psi[solid] = psi_solid[solid]


def enforce_omega_bc(psi, omega):
    # Walls (no-slip)
    omega[0, 1:-1] = -2.0 * (psi[1, 1:-1] - psi[0, 1:-1]) / (dy**2)
    omega[-1, 1:-1] = -2.0 * (psi[-2, 1:-1] - psi[-1, 1:-1]) / (dy**2)

    # Inlet and outlet
    omega[:, 0] = 0.0
    omega[:, -1] = omega[:, -2]

    # Solid plate: enforce no-slip via boundary vorticity on fluid neighbors
    fluid = ~solid

    # Fluid cells to the left of a solid cell
    left_neighbors = solid[:, 1:] & fluid[:, :-1]
    omega[:, :-1][left_neighbors] = -2.0 * (psi[:, :-1][left_neighbors] - psi[:, 1:][left_neighbors]) / (dx**2)

    # Fluid cells to the right of a solid cell
    right_neighbors = solid[:, :-1] & fluid[:, 1:]
    omega[:, 1:][right_neighbors] = -2.0 * (psi[:, 1:][right_neighbors] - psi[:, :-1][right_neighbors]) / (dx**2)

    # Fluid cells below a solid cell
    bottom_neighbors = solid[1:, :] & fluid[:-1, :]
    omega[:-1, :][bottom_neighbors] = -2.0 * (psi[:-1, :][bottom_neighbors] - psi[1:, :][bottom_neighbors]) / (dy**2)

    # Fluid cells above a solid cell
    top_neighbors = solid[:-1, :] & fluid[1:, :]
    omega[1:, :][top_neighbors] = -2.0 * (psi[1:, :][top_neighbors] - psi[:-1, :][top_neighbors]) / (dy**2)

    # Inside solid
    omega[solid] = 0.0


# -----------------------------
# Solver (vorticity-streamfunction)
# -----------------------------
psi = np.zeros((Ny, Nx))
omega = np.zeros((Ny, Nx))

# Initial condition: uniform flow
psi[:, :] = U_in * (y[:, None] + R)
enforce_psi_bc(psi)

# Time step based on convection + diffusion stability
u_max = max(1e-6, abs(U_in))
dt_conv = 0.25 * min(dx, dy) / u_max
dt_diff = 0.25 * min(dx, dy)**2 / nu
dt = min(dt_conv, dt_diff)

max_iter = 3500
poisson_iter = 150
tol = 1e-6

for it in range(max_iter):
    enforce_psi_bc(psi)

    # Poisson solve for streamfunction
    for _ in range(poisson_iter):
        psi_new = psi.copy()
        psi_new[1:-1, 1:-1] = (
            (psi[1:-1, 2:] + psi[1:-1, :-2]) * dy**2 +
            (psi[2:, 1:-1] + psi[:-2, 1:-1]) * dx**2 -
            omega[1:-1, 1:-1] * dx**2 * dy**2
        ) / (2.0 * (dx**2 + dy**2))

        psi = psi_new
        enforce_psi_bc(psi)

    # Velocity from streamfunction
    u = np.zeros_like(psi)
    v = np.zeros_like(psi)
    u[1:-1, 1:-1] = (psi[2:, 1:-1] - psi[:-2, 1:-1]) / (2.0 * dy)
    v[1:-1, 1:-1] = -(psi[1:-1, 2:] - psi[1:-1, :-2]) / (2.0 * dx)

    u[solid] = 0.0
    v[solid] = 0.0

    # Vorticity transport (explicit pseudo-time)
    omega_x = (omega[1:-1, 2:] - omega[1:-1, :-2]) / (2.0 * dx)
    omega_y = (omega[2:, 1:-1] - omega[:-2, 1:-1]) / (2.0 * dy)
    omega_xx = (omega[1:-1, 2:] - 2.0 * omega[1:-1, 1:-1] + omega[1:-1, :-2]) / (dx**2)
    omega_yy = (omega[2:, 1:-1] - 2.0 * omega[1:-1, 1:-1] + omega[:-2, 1:-1]) / (dy**2)

    omega_new = omega.copy()
    omega_new[1:-1, 1:-1] = (
        omega[1:-1, 1:-1]
        + dt * (
            -u[1:-1, 1:-1] * omega_x
            -v[1:-1, 1:-1] * omega_y
            + nu * (omega_xx + omega_yy)
        )
    )

    omega_new[solid] = 0.0

    err = np.max(np.abs(omega_new - omega))
    omega = omega_new

    enforce_omega_bc(psi, omega)

    if it % 200 == 0:
        print(f"iter={it:4d}  max|domega|={err:.3e}")

    if err < tol:
        print(f"Converged at iter={it}")
        break

# -----------------------------
# Final field for visualization
# -----------------------------
u = np.zeros_like(psi)
v = np.zeros_like(psi)

u[1:-1, 1:-1] = (psi[2:, 1:-1] - psi[:-2, 1:-1]) / (2.0 * dy)
v[1:-1, 1:-1] = -(psi[1:-1, 2:] - psi[1:-1, :-2]) / (2.0 * dx)

u[solid] = np.nan
v[solid] = np.nan

speed = np.sqrt(u**2 + v**2)

# -----------------------------
# Plot
# -----------------------------
fig, ax = plt.subplots(figsize=(11, 5))

im = ax.imshow(
    speed,
    origin="lower",
    extent=[x.min(), x.max(), y.min(), y.max()],
    aspect="auto",
)

cbar = fig.colorbar(im, ax=ax, pad=0.01)
cbar.set_label("|v| [m/s]")

# Orifice plate overlay
mask_vis = np.where(solid, 1.0, np.nan)
ax.imshow(
    mask_vis,
    origin="lower",
    extent=[x.min(), x.max(), y.min(), y.max()],
    aspect="auto",
    alpha=0.5,
    cmap="gray",
    vmin=0,
    vmax=1,
)

# Streamlines
start_y = np.linspace(-R * 0.95, R * 0.95, 26)
start_points = np.column_stack((np.full_like(start_y, x[0] + 1e-6), start_y))
ax.streamplot(x, y, u, v, start_points=start_points, density=1.6, arrowsize=0.0, linewidth=1.0)

ax.set_title("2D steady incompressible flow with orifice plate")
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")

text = (
    f"Q={Q:.3f} m^3/s  U_in={U_in:.4f} m/s  "
    f"Re={U_in*D/nu:.0f}  p_out={p_out/1e5:.1f} bar"
)
ax.text(0.01, 0.99, text, transform=ax.transAxes, va="top", ha="left", color="white",
        bbox=dict(facecolor="black", alpha=0.35, pad=4))

plt.tight_layout()
plt.show()
