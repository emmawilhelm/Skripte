from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt


@dataclass
class ModelParams:
    dt: float = 0.01
    t_end: float = 8.0
    seed: int = 42
    n_particles: int = 1200

    # Domain (Seitenansicht)
    x_min: float = -2.0   # Boden links
    x_max: float =  2.0   # Abbaukammer rechts
    y_min: float = -1.0
    y_max: float =  1.0

    # Schneidrad als Platte bei x=0
    plate_x: float = 0.0
    eps: float = 1e-3  # kleine Distanz, damit wir nicht exakt auf der Platte landen

    # Öffnungen als Schlitze (y-Intervalle)
    # Beispiel: 2 Schlitze
    slot_centers: tuple = (-0.35, 0.35)
    slot_half_height: float = 0.10  # Schlitzhöhe = 2*slot_half_height

    # Strömungs-/Kinematik-Parameter (Toy, aber Seitenansicht-plausibel)
    v_push: float = 0.45            # Haupttransport nach +x (Druck/Schub)
    vy_noise: float = 0.02          # kleine Streuung (Heterogenität)
    slide_speed: float = 0.25       # Gleiten entlang der Platte, wenn geschlossen
    stickiness: float = 0.5         # 0..1 bremst Gleiten (1 = sehr klebrig)
    slot_pull: float = 0.9          # zieht Partikel in Richtung nächster Öffnung (nur nahe Platte)
    pull_band: float = 0.20         # wirkt nur, wenn |x-plate_x| < pull_band

    # Hinter der Platte (Abbaukammer)
    v_chamber: float = 0.25         # zusätzlicher Transport nach +x

     # Abbaukammer: Verteilung/“Auffüllen”
    chamber_diff: float = 0.12      # Stärke der seitlichen Dispersion (größer = schnelleres Auffüllen)
    chamber_mix: float = 0.25       # Rückführung Richtung Kammermitte (0 = aus)
    reflect_y: bool = True          # an y-Wänden reflektieren
    reflect_xmax: bool = False      # optional: an x_max reflektieren (für "voller Kasten")


def set_seed(seed: int) -> None:
    np.random.seed(seed)


def time_grid(dt: float, t_end: float) -> np.ndarray:
    n = int(np.floor(t_end / dt)) + 1
    return np.linspace(0.0, dt * (n - 1), n)


def is_in_any_slot(y: np.ndarray, params: ModelParams) -> np.ndarray:
    """True, wenn y in irgendeinem Schlitzintervall liegt."""
    inside = np.zeros_like(y, dtype=bool)
    for c in params.slot_centers:
        inside |= (np.abs(y - c) <= params.slot_half_height)
    return inside


def nearest_slot_center(y: np.ndarray, params: ModelParams) -> np.ndarray:
    """Gibt für jedes y den nächstgelegenen Schlitzmittelpunkt zurück."""
    centers = np.array(params.slot_centers, dtype=float)
    idx = np.argmin(np.abs(y[:, None] - centers[None, :]), axis=1)
    return centers[idx]


def initialize_particles(params: ModelParams) -> np.ndarray:
    # Start links (Boden)
    x = np.random.uniform(params.x_min, params.plate_x - 0.4, size=params.n_particles)
    y = np.random.uniform(params.y_min * 0.9, params.y_max * 0.9, size=params.n_particles)
    return np.column_stack([x, y])


def velocity_field(xy: np.ndarray, t: float, params: ModelParams) -> np.ndarray:
    x = xy[:, 0]
    y = xy[:, 1]

    vx = np.full_like(x, params.v_push)
    vy = np.random.normal(0.0, params.vy_noise, size=y.shape)

    # Zug zu Öffnungen nur in der Nähe der Platte
    band = np.abs(x - params.plate_x) < params.pull_band
    if np.any(band):
        target = nearest_slot_center(y[band], params)
        vy[band] += params.slot_pull * (target - y[band])

    # Hinter der Platte: Kammertransport + Mischen + Dispersion
    behind = x > params.plate_x
    if np.any(behind):
        vx[behind] += params.v_chamber
        vy[behind] += -params.chamber_mix * y[behind]
        vy[behind] += np.random.normal(
            0.0, params.chamber_diff, size=np.sum(behind)
        ) / np.sqrt(max(params.dt, 1e-9))

    # ✅ Return MUSS immer am Ende der Funktion stehen (nicht im if!)
    return np.column_stack([vx, vy])
    


def step_euler(xy: np.ndarray, v: np.ndarray, dt: float) -> np.ndarray:
    return xy + dt * v


def apply_plate_boundary(xy_prev: np.ndarray, xy_new: np.ndarray, passed: np.ndarray, params: ModelParams):
    """
    Platte bei x=plate_x ist undurchlässig außer in Schlitzen.
    - Wenn Partikel von links nach rechts über die Platte geht:
        * wenn y im Schlitz: passieren lassen (passed=True)
        * sonst: an Platte "abfangen" und entlang Platte gleiten (Stau/Umleitung)
    """
    x0 = xy_prev[:, 0]
    x1 = xy_new[:, 0]
    y1 = xy_new[:, 1]

    crossing = (~passed) & (x0 < params.plate_x) & (x1 >= params.plate_x)
    if not np.any(crossing):
        return xy_new, passed

    idx = np.where(crossing)[0]
    y_hit = y1[idx]

    in_slot = is_in_any_slot(y_hit, params)

    # Fall A: Öffnung -> passieren
    idx_pass = idx[in_slot]
    passed[idx_pass] = True
    xy_new[idx_pass, 0] = params.plate_x + params.eps  # direkt "hinter" der Platte starten

    # Fall B: geschlossen -> Stau + Gleiten
    idx_block = idx[~in_slot]
    if idx_block.size > 0:
        # auf Vorderseite der Platte setzen
        xy_new[idx_block, 0] = params.plate_x - params.eps

        # entlang Platte gleiten Richtung nächster Öffnung
        yb = xy_new[idx_block, 1]
        target = nearest_slot_center(yb, params)
        sign = np.sign(target - yb)
        slide = params.slide_speed * (1.0 - params.stickiness)
        xy_new[idx_block, 1] = yb + sign * slide * params.dt

    return xy_new, passed


def simulate(params: ModelParams):
    set_seed(params.seed)
    ts = time_grid(params.dt, params.t_end)

    xy = initialize_particles(params)
    passed = np.zeros(params.n_particles, dtype=bool)

    traj_idx = np.linspace(0, params.n_particles - 1, 80, dtype=int)
    traj = np.zeros((ts.size, traj_idx.size, 2), dtype=float)
    passed_log = np.zeros(ts.size, dtype=int)

    for k, t in enumerate(ts):
        xy_prev = xy.copy()
        v = velocity_field(xy, t, params)
        xy_new = step_euler(xy, v, params.dt)

        xy_new, passed = apply_plate_boundary(xy_prev, xy_new, passed, params)

        # --- y-Wände: reflektieren statt clip ---
        if params.reflect_y:
            y = xy_new[:, 1]
            low = y < params.y_min
            high = y > params.y_max
            y[low] = params.y_min + (params.y_min - y[low])
            y[high] = params.y_max - (y[high] - params.y_max)
            xy_new[:, 1] = y
        else:
            xy_new[:, 1] = np.clip(xy_new[:, 1], params.y_min, params.y_max)

        # --- x-Grenzen ---
        xy_new[:, 0] = np.maximum(xy_new[:, 0], params.x_min)

        if params.reflect_xmax:
            x = xy_new[:, 0]
            highx = x > params.x_max
            x[highx] = params.x_max - (x[highx] - params.x_max)
            xy_new[:, 0] = x
        else:
            xy_new[:, 0] = np.minimum(xy_new[:, 0], params.x_max)

        # Update
        xy = xy_new
        traj[k, :, :] = xy[traj_idx, :]
        passed_log[k] = int(np.sum(passed))

    return ts, traj, passed_log

def plot_sideview(traj: np.ndarray, passed_log: np.ndarray, params: ModelParams):
    plt.figure()

    # Trajektorien
    for i in range(traj.shape[1]):
        plt.plot(traj[:, i, 0], traj[:, i, 1], linewidth=0.9)

    # Platte (Schneidrad) als Linie
    plt.plot([params.plate_x, params.plate_x], [params.y_min, params.y_max], linewidth=4)

    # Schlitze markieren
    for c in params.slot_centers:
        plt.plot([params.plate_x, params.plate_x], [c - params.slot_half_height, c + params.slot_half_height], linewidth=10)

    plt.xlim(params.x_min, params.x_max)
    plt.ylim(params.y_min, params.y_max)
    plt.xlabel("x (Boden -> Schneidrad -> Abbaukammer)")
    plt.ylabel("y")
    plt.title("Seitenansicht: Materialfluss durch Schlitze im Schneidrad (Toy-Modell)")
    plt.show()

    plt.figure()
    plt.plot(passed_log)
    plt.title("Anzahl Partikel, die durch Öffnungen in die Abbaukammer gelangt sind")
    plt.xlabel("Zeitschritt")
    plt.ylabel("passed count")
    plt.show()


def main():
    params = ModelParams()
    ts, traj, passed_log = simulate(params)
    plot_sideview(traj, passed_log, params)


if __name__ == "__main__":
    main()
    