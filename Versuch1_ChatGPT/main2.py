from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# 1) Parameter & Konfiguration
# -----------------------------
@dataclass
class ModelParams:
    # Zeitschritt & Laufzeit
    dt: float = 0.02
    t_end: float = 10.0

    # Schneidrad / Rotation (vereinfachtes 2D-Modell)
    omega: float = 1.0  # rad/s

    # Arbeitsraum (2D)
    x_min: float = -1.5
    x_max: float = 1.5
    y_min: float = -1.5
    y_max: float = 1.5

    # Diskretisierung
    n_particles: int = 1200
    seed: int = 42

    # Schneidrad-Geometrie
    R: float = 1.0                   # Radius der Scheibe
    n_openings: int = 6              # Anzahl Öffnungen
    opening_half_angle: float = 0.18 # halbe Öffnungsbreite [rad] (0.18 ~ 10°)
    opening_phase: float = 0.0       # Drehwinkel-Offset [rad]

    # Startverteilung (Material vorne)
    start_x_range: tuple = (-1.4, -1.05)
    start_y_range: tuple = (-0.9, 0.9)


# -----------------------------
# 2) Hilfsfunktionen
# -----------------------------
def set_seed(seed: int) -> None:
    np.random.seed(seed)


def time_grid(dt: float, t_end: float) -> np.ndarray:
    n = int(np.floor(t_end / dt)) + 1
    return np.linspace(0.0, dt * (n - 1), n)


def wrap_to_pi(angle: np.ndarray) -> np.ndarray:
    """Winkel in (-pi, pi]"""
    return (angle + np.pi) % (2 * np.pi) - np.pi


# -----------------------------
# 3) Geometrie: Öffnungen
# -----------------------------
def opening_centers(params: ModelParams) -> np.ndarray:
    """Zentren der Öffnungen als Winkel (rad)"""
    return params.opening_phase + np.linspace(0, 2*np.pi, params.n_openings, endpoint=False)


def is_in_opening(theta: np.ndarray, params: ModelParams) -> np.ndarray:
    """
    True, wenn Winkel theta innerhalb einer Öffnung liegt.
    Öffnungen sind Intervalle um die Öffnungszentren.
    """
    centers = opening_centers(params)  # (n_openings,)
    # Broadcast: theta (N,) gegen centers (M,)
    d = wrap_to_pi(theta[:, None] - centers[None, :])  # (N, M)
    inside_any = np.any(np.abs(d) <= params.opening_half_angle, axis=1)  # (N,)
    return inside_any


# -----------------------------
# 4) Initialisierung
# -----------------------------
def initialize_particles(params: ModelParams) -> np.ndarray:
    x = np.random.uniform(params.start_x_range[0], params.start_x_range[1], size=params.n_particles)
    y = np.random.uniform(params.start_y_range[0], params.start_y_range[1], size=params.n_particles)
    return np.column_stack([x, y])


# -----------------------------
# 5) Geschwindigkeitsfeld (Toy)
# -----------------------------
def velocity_field(xy: np.ndarray, t: float, params: ModelParams) -> np.ndarray:
    """
    Toy-Feld: Rotation um Ursprung + kleiner Drift nach rechts (Richtung Kammer).
    """
    x = xy[:, 0]
    y = xy[:, 1]

    # Rotation
    vx = -params.omega * y
    vy = params.omega * x

    # leichter "Förderdrift" Richtung Kammer (x positiv)
    vx += 0.25

    return np.column_stack([vx, vy])


def step_euler(xy: np.ndarray, v: np.ndarray, dt: float) -> np.ndarray:
    return xy + dt * v


# -----------------------------
# 6) Randbedingung Schneidrad: reflektieren außer Öffnung
# -----------------------------
def apply_cutterhead_boundary(xy_prev: np.ndarray, xy_new: np.ndarray, passed: np.ndarray, params: ModelParams):
    """
    Schneidrad als Kreis r=R.
    - Partikel, die noch nicht "passed" sind, werden an der Kreisgrenze reflektiert,
      außer sie treffen eine Öffnung -> dann werden sie als passed markiert.
    - Partikel, die passed=True sind, werden nicht mehr reflektiert (sie sind "hinten").
    """
    # Nur Partikel vorne (nicht passed) behandeln
    mask = ~passed
    if not np.any(mask):
        return xy_new, passed

    x = xy_new[mask, 0]
    y = xy_new[mask, 1]
    r = np.sqrt(x**2 + y**2)

    outside = r > params.R
    if not np.any(outside):
        return xy_new, passed

    # Winkel der neuen Position
    theta = np.arctan2(y[outside], x[outside])
    hit_opening = is_in_opening(theta, params)

    # Für die, die eine Öffnung treffen: passed = True (durchgelassen)
    idx_mask = np.where(mask)[0]              # globale Indizes der maskierten
    idx_outside_global = idx_mask[outside]    # globale Indizes "außerhalb"
    idx_pass = idx_outside_global[hit_opening]
    passed[idx_pass] = True

    # Für die, die NICHT durch Öffnung dürfen: reflektieren -> zurück auf alte Position
    idx_reflect = idx_outside_global[~hit_opening]
    xy_new[idx_reflect] = xy_prev[idx_reflect]

    return xy_new, passed


# -----------------------------
# 7) Simulation
# -----------------------------
def simulate(params: ModelParams):
    set_seed(params.seed)
    ts = time_grid(params.dt, params.t_end)

    xy = initialize_particles(params)
    passed = np.zeros(params.n_particles, dtype=bool)

    traj_sample_idx = np.linspace(0, params.n_particles - 1, 80, dtype=int)
    traj = np.zeros((ts.size, traj_sample_idx.size, 2), dtype=float)
    passed_log = np.zeros(ts.size, dtype=int)

    for k, t in enumerate(ts):
        xy_prev = xy.copy()
        v = velocity_field(xy, t, params)
        xy_new = step_euler(xy, v, params.dt)

        xy_new, passed = apply_cutterhead_boundary(xy_prev, xy_new, passed, params)
        xy = xy_new

        traj[k, :, :] = xy[traj_sample_idx, :]
        passed_log[k] = int(np.sum(passed))

    return ts, traj, passed_log, params


# -----------------------------
# 8) Plot
# -----------------------------
def plot_scene(traj: np.ndarray, passed_log: np.ndarray, params: ModelParams) -> None:
    plt.figure()

    # Trajektorien
    for i in range(traj.shape[1]):
        plt.plot(traj[:, i, 0], traj[:, i, 1], linewidth=0.8)

    # Schneidrad-Kreis
    ang = np.linspace(0, 2*np.pi, 400)
    plt.plot(params.R*np.cos(ang), params.R*np.sin(ang), linewidth=2.0)

    # Öffnungen markieren (Bögen)
    centers = opening_centers(params)
    for c in centers:
        a1 = c - params.opening_half_angle
        a2 = c + params.opening_half_angle
        aa = np.linspace(a1, a2, 50)
        plt.plot(params.R*np.cos(aa), params.R*np.sin(aa), linewidth=4.0)

    plt.gca().set_aspect("equal", "box")
    plt.xlim(params.x_min, params.x_max)
    plt.ylim(params.y_min, params.y_max)
    plt.title("Trajektorien mit Schneidrad (Kreis) + Öffnungen (Bögen)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

    plt.figure()
    plt.plot(passed_log)
    plt.title("Anzahl Partikel, die durch Öffnungen 'hinten' angekommen sind")
    plt.xlabel("Zeitschritt")
    plt.ylabel("passed count")
    plt.show()


def main():
    params = ModelParams()
    ts, traj, passed_log, params = simulate(params)
    plot_scene(traj, passed_log, params)


if __name__ == "__main__":
    main()

