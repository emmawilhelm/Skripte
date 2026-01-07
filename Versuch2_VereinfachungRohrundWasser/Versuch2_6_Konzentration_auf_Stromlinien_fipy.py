import numpy as np
import matplotlib.pyplot as plt

try:
    from fipy import (
        Grid2D,
        CellVariable,
        FaceVariable,
        DiffusionTerm,
        ConvectionTerm,
        TransientTerm,
        ImplicitSourceTerm,
    )
except ImportError as exc:
    raise SystemExit(
        "FiPy is not installed. Install with: python3 -m pip install fipy"
    ) from exc



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
Nx = 200
Ny = 110

x = np.linspace(0.0, L, Nx)
y = np.linspace(-R, R, Ny)

dx = x[1] - x[0]
dy = y[1] - y[0]

mesh = Grid2D(dx=dx, dy=dy, nx=Nx, ny=Ny)

# cell centers
xc, yc = mesh.cellCenters

plate_x0 = x0 - plate_thickness / 2.0
plate_x1 = x0 + plate_thickness / 2.0

# Solid mask: plate blocks flow outside the hole
solid_mask = (xc >= plate_x0) & (xc <= plate_x1) & (np.abs(yc) > r0)
solid = CellVariable(mesh=mesh, value=0.0)
solid.value = solid_mask.astype(float)

# Streamfunction boundary values
psi_bottom = 0.0
psi_top = U_in * (2.0 * R)

# Streamfunction target inside solid (no-penetration)
psi_solid = np.where(yc >= 0.0, psi_top, psi_bottom)

# -----------------------------
# Variables
# -----------------------------
psi = CellVariable(mesh=mesh, name="psi", value=0.0)
omega = CellVariable(mesh=mesh, name="omega", value=0.0)

velocity = FaceVariable(mesh=mesh, rank=1)

# -----------------------------
# Boundary conditions
# -----------------------------
# Inlet: uniform flow -> psi(y) = U_in*(y + R)
face_y = mesh.faceCenters[1]
psi.constrain(U_in * (face_y + R), where=mesh.facesLeft)

# Walls: fixed psi
psi.constrain(psi_bottom, where=mesh.facesBottom)
psi.constrain(psi_top, where=mesh.facesTop)

# Outlet: zero-gradient for psi
psi.faceGrad.constrain(0.0, where=mesh.facesRight)

# Vorticity: simple slip-like conditions on boundaries
omega.constrain(0.0, where=mesh.facesLeft | mesh.facesTop | mesh.facesBottom)
omega.faceGrad.constrain(0.0, where=mesh.facesRight)

# -----------------------------
# Equations (vorticity-streamfunction)
# -----------------------------
big = 1.0e8

psi_eq = (
    DiffusionTerm(coeff=1.0)
    + ImplicitSourceTerm(coeff=big * solid)
    == -omega + big * solid * psi_solid
)

omega_eq = (
    TransientTerm()
    + ConvectionTerm(coeff=velocity)
    == DiffusionTerm(coeff=nu) + ImplicitSourceTerm(coeff=big * solid)
)

# -----------------------------
# Solve (pseudo-time march)
# -----------------------------
psi.value = U_in * (yc + R)

u_max = max(1e-6, abs(U_in))
dt_conv = 0.25 * min(dx, dy) / u_max
# diffusion-limited dt for stability
nu_max_diff = max(1e-12, nu)
dt_diff = 0.25 * min(dx, dy) ** 2 / nu_max_diff
dt = min(dt_conv, dt_diff)

max_iter = 2500
tol = 1e-6

for it in range(max_iter):
    # Update velocity from streamfunction
    velocity[0] = psi.faceGrad[1]
    velocity[1] = -psi.faceGrad[0]

    psi_eq.solve(var=psi)

    omega_old = omega.value.copy()
    omega_eq.solve(var=omega, dt=dt)

    # Force zero vorticity inside solid
    omega.value[solid_mask] = 0.0

    err = np.max(np.abs(omega.value - omega_old))
    if it % 200 == 0:
        print(f"iter={it:4d}  max|domega|={err:.3e}")

    if err < tol:
        print(f"Converged at iter={it}")
        break

# -----------------------------
# Post-processing for plotting
# -----------------------------
# Cell-centered velocity from streamfunction
u = psi.grad[1]
v = -psi.grad[0]

# Reshape to 2D arrays (FiPy cell values are stored with x varying fastest)
def to_grid(values):
    return values.reshape((Nx, Ny), order="F").T

U = to_grid(u.value)
V = to_grid(v.value)

# Mask solid
solid_2d = to_grid(solid_mask)
U = np.where(solid_2d, np.nan, U)
V = np.where(solid_2d, np.nan, V)

speed = np.sqrt(U**2 + V**2)

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
mask_vis = np.where(solid_2d, 1.0, np.nan)
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
ax.streamplot(x, y, U, V, start_points=start_points, density=1.6, arrowsize=0.0, linewidth=1.0)

ax.set_title("2D steady incompressible flow (FiPy, vorticity-streamfunction)")
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
