"""
EPB-TBM – Vereinfachtes Materialfluss-Modell (Bachelorarbeit)
Schrittweiser Aufbau: Geometrie -> Materialfluss -> Auswertung -> Visualisierung

Autorin: Emma Wilhelm
"""

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
    x_min: float = -1.2
    x_max: float = 1.2
    y_min: float = -1.2
    y_max: float = 1.2

    # Diskretisierung
    n_particles: int = 800

    # Reproduzierbarkeit
    seed: int = 42


# -----------------------------
# 2) Hilfsfunktionen
# -----------------------------
def set_seed(seed: int) -> None:
    np.random.seed(seed)


def time_grid(dt: float, t_end: float) -> np.ndarray:
    n = int(np.floor(t_end / dt)) + 1
    return np.linspace(0.0, dt * (n - 1), n)


# -----------------------------
# 3) Platzhalter: Geometrie/Modell
#    (füllen wir im nächsten Schritt)
# -----------------------------
def initialize_particles(params: ModelParams) -> np.ndarray:
    """
    Initialisiert Partikelpositionen im Arbeitsraum.
    Rückgabe: positions shape (N, 2)
    """
    x = np.random.uniform(params.x_min, params.x_max, size=params.n_particles)
    y = np.random.uniform(params.y_min, params.y_max, size=params.n_particles)
    return np.column_stack([x, y])


def velocity_field(xy: np.ndarray, t: float, params: ModelParams) -> np.ndarray:
    """
    Platzhalter für ein Geschwindigkeitsfeld v(x,y,t).
    Wird später ersetzt (Rotation + Öffnungen + Stagnationszonen etc.)
    """
    # Simple starres Rotationsfeld um den Ursprung: v = omega * (-y, x)
    x = xy[:, 0]
    y = xy[:, 1]
    vx = -params.omega * y
    vy = params.omega * x
    return np.column_stack([vx, vy])


def step_euler(xy: np.ndarray, v: np.ndarray, dt: float) -> np.ndarray:
    return xy + dt * v


# -----------------------------
# 4) Simulation
# -----------------------------
def simulate(params: ModelParams):
    set_seed(params.seed)
    ts = time_grid(params.dt, params.t_end)

    xy = initialize_particles(params)

    # Logging (optional)
    traj_sample_idx = np.linspace(0, params.n_particles - 1, 50, dtype=int)
    traj = np.zeros((ts.size, traj_sample_idx.size, 2), dtype=float)

    for k, t in enumerate(ts):
        v = velocity_field(xy, t, params)
        xy = step_euler(xy, v, params.dt)

        # speichern (nur Sample)
        traj[k, :, :] = xy[traj_sample_idx, :]

    return ts, traj


# -----------------------------
# 5) Visualisierung
# -----------------------------
def plot_trajectories(ts: np.ndarray, traj: np.ndarray) -> None:
    plt.figure()
    for i in range(traj.shape[1]):
        plt.plot(traj[:, i, 0], traj[:, i, 1], linewidth=0.8)
    plt.gca().set_aspect("equal", "box")
    plt.title("Trajektorien (Sample) – Platzhaltermodell")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


def main():
    params = ModelParams()
    ts, traj = simulate(params)
    plot_trajectories(ts, traj)


if __name__ == "__main__":
    main()
