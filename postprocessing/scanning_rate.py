from __future__ import annotations
import argparse
import math
from pathlib import Path
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parent.parent
SIM_BASE = REPO_ROOT / "data" / "simulations"

def read_static(sim_dir: Path) -> Tuple[float, List[str], float, float, int]:
    static_path = sim_dir / "static.txt"
    if not static_path.exists():
        raise FileNotFoundError(f"No se encontró static.txt para la simulación '{sim_dir.name}'")

    lines = [line.strip() for line in static_path.read_text().splitlines() if line.strip()]
    if len(lines) < 4:
        raise ValueError(f"static.txt en {static_path} debería tener al menos 4 líneas; se obtuvieron {len(lines)}")

    L = float(lines[0])
    declared_N = int(lines[1])
    r_min = float(lines[2])
    r_max = float(lines[3])
    states = [line.upper() for line in lines[4:]]
    if len(states) == 0:
        raise ValueError(f"No se listaron estados de partículas en {static_path}")
    if len(states) < declared_N:
        raise ValueError(
            f"static declara N={declared_N} pero solo se proporcionaron {len(states)} estados de partícula en {static_path}"
        )
    return L, states, r_min, r_max, declared_N


def packing_fraction_excluding_central(L: float, r_min: float, states_count: int) -> float:
    mobile_n = max(states_count - 1, 0)
    area = mobile_n * math.pi * (r_min * r_min)
    return area / (L * L)


def read_hits(sim_dir: Path, particle_count: int) -> Tuple[np.ndarray, np.ndarray]:
    dynamic_path = sim_dir / "dynamic.txt"
    if not dynamic_path.exists():
        raise FileNotFoundError(f"No se encontró dynamic.txt para la simulación '{sim_dir.name}'")

    times: List[float] = []
    hits: List[int] = []

    with dynamic_path.open("r") as f:
        while True:
            t_line = f.readline()
            if not t_line:
                break
            t_line = t_line.strip()
            if not t_line:
                continue

            parts = t_line.split()
            if not parts:
                continue
            time_value = float(parts[0])
            hit_value = int(parts[1]) if len(parts) > 1 else 0
            times.append(time_value)
            hits.append(hit_value)

            for _ in range(particle_count):
                data_line = f.readline()
                if not data_line:
                    raise ValueError(
                        f"Fin de archivo inesperado en {dynamic_path} al leer datos de partículas (tiempo={time_value})"
                    )
                if len(data_line.strip().split()) < 5:
                    raise ValueError(
                        f"Se esperaban datos de partículas con 5 columnas en el tiempo {time_value}; se obtuvo: '{data_line.strip()}'"
                    )

    if not times:
        raise ValueError(f"No se encontraron cuadros en {dynamic_path}")
    return np.asarray(times, float), np.asarray(hits, float)


def load_run(sim_dir: Path) -> Tuple[np.ndarray, np.ndarray, float]:
    L, states, r_min, r_max, declared_N = read_static(sim_dir)

    states_N = len(states)
    if states_N < declared_N:
        raise ValueError(
            f"static.txt declara N={declared_N} pero solo hay {states_N} estados en {sim_dir/'static.txt'}."
        )
    phi = packing_fraction_excluding_central(L, r_min, states_N)

    t, h = read_hits(sim_dir, states_N)

    return t, h, phi


def sse_given_b(t: np.ndarray, y: np.ndarray, b: float) -> Tuple[float, float]:
    a_star = float(np.mean(y - b * t))
    resid = y - (a_star + b * t)
    return float(resid @ resid), a_star


def scan_b_and_minimize(t: np.ndarray, y: np.ndarray, ngrid: int = 801) -> Tuple[float, float, float]:
    T = t - t.mean()
    Y = y - y.mean()
    denom = float(T @ T) if float(T @ T) != 0.0 else 1.0
    b_ols = float((T @ Y) / denom)

    duration = float(np.max(t) - np.min(t))
    y_range = float(np.max(y) - np.min(y))
    span = max(10 * abs(b_ols), 0.1 * (y_range / duration if duration > 0 else 1.0), 1e-6)

    b_grid = np.linspace(b_ols - span, b_ols + span, ngrid)
    sse_grid = np.empty_like(b_grid)
    a_grid = np.empty_like(b_grid)
    for i, b in enumerate(b_grid):
        sse_grid[i], a_grid[i] = sse_given_b(t, y, b)

    i_min = int(np.argmin(sse_grid))
    return float(b_grid[i_min]), float(a_grid[i_min]), float(sse_grid[i_min])


def q_phi_for_base(base_name: str, tmark: float) -> Tuple[float, float, float, float]:
    replicas = [f"{base_name}_{i}" for i in range(1, 4)]
    phis: List[float] = []
    qs: List[float] = []

    ref_t: np.ndarray | None = None
    for rep in replicas:
        sim_dir = SIM_BASE / rep
        if not sim_dir.exists():
            raise FileNotFoundError(f"No existe la carpeta de simulación: {sim_dir}")
        t, h, phi = load_run(sim_dir)
        if ref_t is None:
            ref_t = t
        else:
            if len(t) != len(ref_t) or not np.allclose(t, ref_t, rtol=1e-9, atol=1e-9):
                raise ValueError(f"Las réplicas de '{base_name}' no comparten los mismos timestamps (ver {rep}).")

        m = t >= tmark
        if not np.any(m):
            raise ValueError(f"No hay datos con t >= {tmark} en {rep}")
        t_win = t[m]
        y_win = h[m]
        b_star, _, _ = scan_b_and_minimize(t_win, y_win)
        phis.append(phi)
        qs.append(b_star)

    phi_mean = float(np.mean(phis))
    q_mean = float(np.mean(qs))
    phi_std = float(np.std(phis, ddof=1)) if len(phis) > 1 else 0.0
    q_std = float(np.std(qs, ddof=1)) if len(qs) > 1 else 0.0
    return phi_mean, phi_std, q_mean, q_std


def main():
    ap = argparse.ArgumentParser(
        description="Gráfico principal Q vs φ con barras de error (mismo color para todos los puntos)."
    )
    ap.add_argument(
        "simulations",
        nargs="+",
        help="Nombres base; se esperan réplicas '<name>_1', '_2', '_3' en data/simulations/",
    )
    ap.add_argument(
        "--tmark",
        type=float,
        default=20.0,
        help="Tiempo a partir del cual se considera el estacionario para el ajuste lineal (default 20 s).",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "data" / "graphics" / "hits",
        help="Carpeta de salida.",
    )
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    x_phi_mean, x_phi_err, y_q_mean, y_q_err = [], [], [], []
    for base in args.simulations:
        phi_mean, phi_std, q_mean, q_std = q_phi_for_base(base, args.tmark)
        x_phi_mean.append(phi_mean)
        x_phi_err.append(phi_std if phi_std > 0 else 0.0)
        y_q_mean.append(q_mean)
        y_q_err.append(q_std if q_std > 0 else 0.0)

    x_phi_mean = np.asarray(x_phi_mean)
    x_phi_err  = np.asarray(x_phi_err)
    y_q_mean   = np.asarray(y_q_mean)
    y_q_err    = np.asarray(y_q_err)

    plt.figure(figsize=(7.6, 5.2))
    plt.errorbar(
        x_phi_mean, y_q_mean,
        xerr=x_phi_err if np.any(x_phi_err > 0) else None,
        yerr=y_q_err if np.any(y_q_err > 0) else None,
        fmt="o", ms=6, elinewidth=1.2, capsize=3, capthick=1.2,
        color="C0", ecolor="C0", mec="C0", mfc="white",
        zorder=3,
    )
    order = np.argsort(x_phi_mean)
    x_line = x_phi_mean[order]
    y_line = y_q_mean[order]
    plt.plot(x_line, y_line, '-', lw=1.8, color="C0", alpha=0.9, zorder=2)

    plt.xlabel(r"$\phi$", fontsize=14)
    plt.ylabel(r"$Q$ (1/s)", fontsize=14)
    plt.tick_params(axis="both", labelsize=14)
    plt.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
    plt.tight_layout()

    out_path = args.out / "Q_vs_phi.png"
    plt.savefig(out_path, dpi=180)
    plt.show()
    plt.close()
    print(f"[OK] Gráfico guardado: {out_path}")


if __name__ == "__main__":
    main()
