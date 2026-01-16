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

    # Kanten der Orifice-Platte
    # x1 = xS - 0.5 * L_orifice   (existiert schon weiter oben)
    # x2 = xS + 0.5 * L_orifice

    # Kontraktion: 1.31 -> 2.60
    x_c_start = x1 - Lc          # 1.31 m
    x_c_end   = x1               # 2.60 m

    # Expansion: 3.40 -> 4.69
    x_e_start = x2               # 3.40 m
    x_e_end   = x2 + Lc          # 4.69 m

    def sigmoid_ramp(xarr: np.ndarray, a: float, b: float, eps_level: float = 0.01) -> np.ndarray:
        xm = 0.5 * (a + b)
        L = (b - a)
        eps = float(eps_level)
        k = (L / 2.0) / np.log((1.0 - eps) / eps)
        w = logistic((xarr - xm) / k)
        w[xarr <= a] = 0.0
        w[xarr >= b] = 1.0
        return w

    # weights
    w_c = sigmoid_ramp(x, x_c_start, x_c_end, eps_level=0.01)  # 0->1 (pipe->orifice)
    w_e = sigmoid_ramp(x, x_e_start, x_e_end, eps_level=0.01)  # 0->1 (orifice->pipe)

    # Build R_eff piecewise
    R_eff = np.full_like(x, R_pipe)

    # Kontraktion: R_pipe -> R_orifice
    mask_contr = (x >= x_c_start) & (x <= x_c_end)
    R_eff[mask_contr] = R_pipe - w_c[mask_contr] * (R_pipe - R_orifice)

    # In der Orifice-Platte: konstant R_orifice
    mask_orifice = (x >= x1) & (x <= x2)
    R_eff[mask_orifice] = R_orifice

    # Expansion: R_orifice -> R_pipe
    mask_exp = (x >= x_e_start) & (x <= x_e_end)
    R_eff[mask_exp] = R_orifice + w_e[mask_exp] * (R_pipe - R_orifice)

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
    # kleines u, damit Streamlines nicht in 0-Zonen enden
    u_mean_mesh = np.tile(u_mean_x, (ny, 1))
    u[outer_ring] = 0.01 * u_mean_mesh[outer_ring]

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

    # ------------------------------------------------------------
    # v aus Kontinuität: dv/dy = -du/dx
    # Randbedingung: keine Durchströmung an den Wänden -> v=0 an top/bottom
    # Vorgehen:
    #  - integriere von oben nach unten mit v(top)=0  -> v_top
    #  - integriere von unten nach oben mit v(bottom)=0 -> v_bot
    #  - mitteln: v = 0.5*(v_top + v_bot)
    # ------------------------------------------------------------
    v = np.full_like(u, np.nan, dtype=float)

    for i in range(nx):
        # gültige Punkte in dieser x-Spalte (innerhalb Geometrie)
        valid = np.isfinite(du_dx[:, i]) & inside[:, i]
        idx = np.where(valid)[0]
        if idx.size < 2:
            continue

        j_top = idx[-1]     # größtes y (oben)
        j_bot = idx[0]      # kleinstes y (unten)

        # 1) Integration von oben nach unten: v(top)=0
        v_top = np.full(ny, np.nan, dtype=float)
        v_top[j_top] = 0.0
        for j in range(j_top - 1, j_bot - 1, -1):
            if valid[j] and valid[j + 1]:
                dy = y[j + 1] - y[j]  # >0
                # v(j) = v(j+1) + ∫_{y_j}^{y_{j+1}} du/dx dy  (weil dv/dy=-du/dx)
                v_top[j] = v_top[j + 1] + 0.5 * (du_dx[j, i] + du_dx[j + 1, i]) * dy

        # 2) Integration von unten nach oben: v(bottom)=0
        v_bot = np.full(ny, np.nan, dtype=float)
        v_bot[j_bot] = 0.0
        for j in range(j_bot + 1, j_top + 1):
            if valid[j] and valid[j - 1]:
                dy = y[j] - y[j - 1]  # >0
                # v(j) = v(j-1) - ∫_{y_{j-1}}^{y_j} du/dx dy
                v_bot[j] = v_bot[j - 1] - 0.5 * (du_dx[j, i] + du_dx[j - 1, i]) * dy

         # 3) Mittelwert, um beide Wandbedingungen "gleich" zu behandeln
        v[:, i] = 0.5 * (v_top + v_bot)

    # Optional: numerisches Rauschen sehr nah an Wänden abschwächen (kann helfen)
    # v[np.abs(v) < 1e-12] = 0.0


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

    # Betrag der Geschwindigkeit (masked bleibt erhalten)
    speed = np.ma.sqrt(u**2 + v**2)

    print("Diagnosewerte:")
    print(f"Mittlere Geschwindigkeit im Rohr:     {diag['v_pipe_mean']:.6f} m/s")
    print(f"Mittlere Geschwindigkeit in Orifice:  {diag['v_orifice_mean']:.6f} m/s")
    print(f"Re im Rohr:                           {diag['Re_pipe']:.3f}")
    print(f"Re in Orifice:                        {diag['Re_orifice']:.3f}")

    # --- Plot: geometry outline and streamplot ---
    x = X[0, :]
    R_pipe = D_pipe / 2.0

    fig, ax = plt.subplots(figsize=(12, 4.8))

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

    # -------------------------------------------------
    # 2.5) Farbverlauf: Betrag der Geschwindigkeit
    #      -> unter Streamlines und unter der Platte
    # -------------------------------------------------
    # pcolormesh ist robust für regelmäßige Gitter
    pcm = ax.pcolormesh(
        X, Y, speed,
        shading="auto",
        alpha=0.9,
        zorder=0
    )

    cbar = plt.colorbar(pcm, ax=ax, pad=0.02)
    cbar.set_label("|u| [m/s]")

    # Streamplot (fill masked with 0 just for plotting)
    u_plot = u
    v_plot = v

    x0 = 0.05
    ys = np.linspace(-R_pipe * 0.95, R_pipe * 0.95, 28)
    start_points = np.column_stack([np.full_like(ys, x0), ys])

    ax.streamplot(
        X, Y, 
        u_plot, v_plot,
        start_points=start_points,
        density=2.0,
        linewidth=0.8,
        arrowsize=1.1,
        minlength=0.2,    # verhindert sehr kurze Abbrüche
        maxlength=10.0,   # erlaubt lange Linien
        zorder=3
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