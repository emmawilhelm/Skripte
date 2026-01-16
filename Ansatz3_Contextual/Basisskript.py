import numpy as np
import matplotlib.pyplot as plt


def logistic(z: np.ndarray) -> np.ndarray:
    """Stable-ish logistic sigmoid."""
    # Clip to avoid overflow in exp for large |z|
    zc = np.clip(z, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-zc))


def smooth_top_hat_sigmoid(
    x: np.ndarray,
    xS: float,
    L_orifice: float,
    Lc: float,
    eps_level: float = 0.01,
) -> np.ndarray:
    """
    Smoothed top-hat g(x) in [0,1] using sigmoid transitions.

    Orifice region is centered at xS with length L_orifice.
    Each flank transition (from ~eps to ~(1-eps)) happens over length Lc.

    We choose logistic steepness k such that:
      logistic((+Lc/2)/k) - logistic((-Lc/2)/k) spans ~[eps, 1-eps]
    More precisely, logistic reaches eps at -Lc/2 and 1-eps at +Lc/2.
    For logistic, inverse is log(p/(1-p)).
    """
    # Orifice edges
    x1 = xS - 0.5 * L_orifice
    x2 = xS + 0.5 * L_orifice

    # Compute steepness parameter k from desired flank length Lc
    # logistic((x-x0)/k) = p  => (x-x0)/k = ln(p/(1-p))
    # Want p=eps at x=x0-Lc/2, and p=1-eps at x=x0+Lc/2
    # Then Lc/2k = ln((1-eps)/eps)
    eps = float(eps_level)
    if not (0.0 < eps < 0.5):
        raise ValueError("eps_level must be in (0, 0.5).")
    k = (Lc / 2.0) / np.log((1.0 - eps) / eps)

    # Sigmoid step-up at x1 and step-down at x2
    step_up = logistic((x - x1) / k)
    step_down = logistic((x2 - x) / k)

    # Smooth top-hat: ~1 between edges, ~0 outside, with sigmoid flanks
    return step_up * step_down


def build_velocity_field(
    D_pipe: float,
    L_pipe: float,
    d_orifice: float,
    xS: float,
    Q: float,
    rho: float,
    mu: float,
    L_orifice: float,
    Lc: float,
    nx: int = 600,
    ny: int = 240,
):
    """
    2D velocity field (u,v) in x-y plane (longitudinal cut).

    - u(x,y): Poiseuille parabola using local radius R(x)
    - v(x,y): from continuity dv/dy = -du/dx with v(x,0)=0
    - constant volumetric flow rate Q => u_mean(x)=Q/A(x)
    """
    R_pipe = D_pipe / 2.0
    R_orifice = d_orifice / 2.0

    # Grid
    x = np.linspace(0.0, L_pipe, nx)
    y = np.linspace(-R_pipe, R_pipe, ny)
    X, Y = np.meshgrid(x, y)

    # --- 1) Abrupte Geometrie (Wand) ---
    x1 = xS - 0.5 * L_orifice   # 2.60 m
    x2 = xS + 0.5 * L_orifice   # 3.40 m

    R_geom = np.full_like(x, R_pipe)
    R_geom[(x >= x1) & (x <= x2)] = R_orifice

    # --- 2) Effektiver Radius nur fürs Strömungsfeld (glatte Streamlines) ---
    # Glättung soll von x_start bis x_end laufen
    x_start = x1 - Lc            # 2.60 - 1.29 = 1.31 m
    x_end = x1                   # 2.60 m

    # Sigmoid-Gewichtung w(x) von 0 -> 1 über [x_start, x_end]
    xm = 0.5 * (x_start + x_end)
    L = (x_end - x_start)

    eps = 0.01
    k = (L / 2.0) / np.log((1.0 - eps) / eps)

    w = logistic((x - xm) / k)
    w[x <= x_start] = 0.0
    w[x >= x_end] = 1.0

    R_eff = R_pipe - w * (R_pipe - R_orifice)

    # Local mean velocity from constant Q (auf Basis von R_eff!)
    A_x = np.pi * (R_eff ** 2)
    u_mean_x = Q / A_x
    u_max_x = 2.0 * u_mean_x

    # Broadcast to mesh
    R_geom_mesh = np.tile(R_geom, (ny, 1))
    R_eff_mesh = np.tile(R_eff, (ny, 1))
    u_max_mesh = np.tile(u_max_x, (ny, 1))

    # Domäne (Maske) aus *realer* Geometrie
    inside = np.abs(Y) <= R_geom_mesh

    # Axial velocity:
    # - innerhalb realer Geometrie: definiert
    # - Poiseuille bezogen auf R_eff (glatte Feld-Änderung)
    u = np.full_like(X, np.nan, dtype=float)
    u[inside] = u_max_mesh[inside] * (1.0 - (Y[inside] / R_eff_mesh[inside]) ** 2)

    # Punkte innerhalb der realen Wand, aber außerhalb des effektiven Radius würden sonst negative u liefern:
    outer_ring = inside & (np.abs(Y) > R_eff_mesh)
    u[outer_ring] = 0.0


    # du/dx (row-wise), ignoring NaNs outside
    du_dx = np.full_like(u, np.nan, dtype=float)
    for j in range(ny):
        row = u[j, :]
        valid = np.isfinite(row)
        if np.count_nonzero(valid) < 3:
            continue
        idx = np.where(valid)[0]
        i0, i1 = idx[0], idx[-1]
        du_dx[j, i0:i1 + 1] = np.gradient(row[i0:i1 + 1], x[i0:i1 + 1])

    # v from continuity with symmetry v(x,0)=0
    v = np.full_like(u, np.nan, dtype=float)
    j0 = int(np.argmin(np.abs(y - 0.0)))
    y_pos = y[j0:]

    for i in range(nx):
        col = du_dx[j0:, i]
        if np.count_nonzero(np.isfinite(col)) < 2:
            continue
        v_col = np.zeros_like(col)
        for k in range(1, len(y_pos)):
            if np.isfinite(col[k]) and np.isfinite(col[k - 1]):
                dy = y_pos[k] - y_pos[k - 1]
                v_col[k] = v_col[k - 1] - 0.5 * (col[k] + col[k - 1]) * dy
            else:
                v_col[k] = np.nan
        v[j0:, i] = v_col

    # Mirror antisymmetrically to y<0: v(x,-y) = -v(x,y)
    for j in range(0, j0):
        j_mirror = j0 + (j0 - j)
        if j_mirror < ny:
            v[j, :] = -v[j_mirror, :]

    # Mask outside
    u_masked = np.ma.array(u, mask=~inside)
    v_masked = np.ma.array(v, mask=~inside)

    # Diagnostics
    v_pipe = Q / (np.pi * (R_pipe ** 2))
    v_orifice = Q / (np.pi * (R_orifice ** 2))
    Re_pipe = rho * v_pipe * D_pipe / mu
    Re_orifice = rho * v_orifice * d_orifice / mu

    diag = {
        "v_pipe_mean": v_pipe,
        "v_orifice_mean": v_orifice,
        "Re_pipe": Re_pipe,
        "Re_orifice": Re_orifice,
    }

    return X, Y, u_masked, v_masked, R_geom, R_eff, diag


def main():
    # --- Fixed parameters (from your setup) ---
    D_pipe = 4.30      # m
    L_pipe = 6.00      # m
    d_orifice = 2.61   # m
    xS = L_pipe / 2.0  # m (centered)

    rho = 1440.0  # kg/m^3
    mu = 1.0      # Pa*s
    Q = 0.0097    # m^3/s

    # --- Your clarifications ---
    L_orifice = 0.80       # m (Störstellenlänge t)
    Lc = 1.29              # m (0.3*D = 1.29 m), smoothing length per flank

    X, Y, u, v, R_geom, R_eff, diag = build_velocity_field(
        D_pipe=D_pipe,
        L_pipe=L_pipe,
        d_orifice=d_orifice,
        xS=xS,
        Q=Q,
        rho=rho,
        mu=mu,
        L_orifice=L_orifice,
        Lc=Lc,
        nx=700,
        ny=260,
    )

    print("Diagnosewerte:")
    print(f"Mittlere Geschwindigkeit im Rohr:     {diag['v_pipe_mean']:.6f} m/s")
    print(f"Mittlere Geschwindigkeit in Orifice:  {diag['v_orifice_mean']:.6f} m/s")
    print(f"Re im Rohr:                           {diag['Re_pipe']:.3f}")
    print(f"Re in Orifice:                        {diag['Re_orifice']:.3f}")

    # --- Plot: geometry outline and streamplot ---
    x = X[0, :]
    R_pipe = D_pipe / 2.0

    fig, ax = plt.subplots(figsize=(12, 4.8))

    raise RuntimeError("BREAKPOINT: Wenn du das siehst, läuft der neue Code.")
    # -------------------------------------------------
    # 1) Rohrwand (konstant, gerade)
    # -------------------------------------------------
    R_pipe = D_pipe / 2.0
    ax.plot([0, L_pipe], [ R_pipe,  R_pipe], "k-", linewidth=1.8)
    ax.plot([0, L_pipe], [-R_pipe, -R_pipe], "k-", linewidth=1.8)

    # -------------------------------------------------
    # 2) Orifice-Platte (schwarze Fläche mit Öffnung)
    #    -> kommt DIREKT NACH der Rohrwand
    # -------------------------------------------------
    x1 = xS - 0.5 * L_orifice   # 2.60 m
    x2 = xS + 0.5 * L_orifice   # 3.40 m
    R_orifice = d_orifice / 2.0

    # oberer Plattenteil
    ax.fill_between(
        [x1, x2],
        [R_pipe, R_pipe],
        [R_orifice, R_orifice],
        color="black",
        zorder=5
    )

    # unterer Plattenteil
    ax.fill_between(
        [x1, x2],
        [-R_orifice, -R_orifice],
        [-R_pipe, -R_pipe],
        color="black",
        zorder=5
    )

    # Streamplot (fill masked with 0 just for plotting)
    u_plot = np.where(u.mask, 0.0, u.data)
    v_plot = np.where(v.mask, 0.0, v.data)

    ax.streamplot(
        X, Y, u_plot, v_plot,
        density=1.8,
        linewidth=0.8,
        arrowsize=1.1
    )

    ax.set_title("Rohrströmung mit Störstelle (Sigmoid-Glättung, Poiseuille + Kontinuität)")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y (radial) [m]")
    ax.set_xlim(0, L_pipe)
    ax.set_ylim(-R_pipe, R_pipe)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()