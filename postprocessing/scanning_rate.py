#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Q vs φ a partir de CURVAS PROMEDIO DE HITS por base: todas las marcas con el mismo color y barras de error.

Uso:
    python3 postprocessing/q_vs_phi.py test10_600 test20_600 test30_600 ... [--tmark 20] [--out data/graphics/hits]
Cada nombre base espera tres réplicas: <base>_1, <base>_2, <base>_3 bajo data/simulations/.
"""

from __future__ import annotations
import argparse
import math
from pathlib import Path
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt

# --- Rutas base ---
THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parent.parent
SIM_BASE = REPO_ROOT / "data" / "simulations"

# ---------- Lectura de estático ----------
def read_static(sim_dir: Path) -> Tuple[float, List[str], float, float, int]:
    static_path = sim_dir / "static.txt"
    if not static_path.exists():
        raise FileNotFoundError(f"static.txt not found for simulation '{sim_dir.name}'")

    lines = [line.strip() for line in static_path.read_text().splitlines() if line.strip()]
    if len(lines) < 4:
        raise ValueError(f"static.txt at {static_path} should have at least 4 lines, got {len(lines)}")

    L = float(lines[0])
    declared_N = int(lines[1])
    r_min = float(lines[2])
    r_max = float(lines[3])
    states = [line.upper() for line in lines[4:]]
    if len(states) == 0:
        raise ValueError(f"No particle states listed in {static_path}")
    if len(states) < declared_N:
        raise ValueError(
            f"static declares N={declared_N} but only {len(states)} particle states were provided in {static_path}"
        )
    # Si hay más estados que N, ignoraremos los extra solo para chequeo; para lectura usamos len(states).
    return L, states, r_min, r_max, declared_N


# ---------- ϕ física: solo partículas móviles (excluye la fija) ----------
def packing_fraction_excluding_central(L: float, r_min: float, states_count: int) -> float:
    """
    ϕ = ((states_count - 1) * π r_min^2) / L^2
    Excluye una partícula (la central/fija) y usa radio impenetrable r_min para las móviles.
    """
    mobile_n = max(states_count - 1, 0)
    area = mobile_n * math.pi * (r_min * r_min)
    return area / (L * L)


# ---------- Lectura de dinámico (exactamente len(states) por frame) ----------
def read_hits(sim_dir: Path, particle_count: int) -> Tuple[np.ndarray, np.ndarray]:
    dynamic_path = sim_dir / "dynamic.txt"
    if not dynamic_path.exists():
        raise FileNotFoundError(f"dynamic.txt not found for simulation '{sim_dir.name}'")

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

            # Leer EXACTAMENTE 'particle_count' líneas de partículas
            for _ in range(particle_count):
                data_line = f.readline()
                if not data_line:
                    raise ValueError(
                        f"Unexpected end of file in {dynamic_path} while reading particle data (time={time_value})"
                    )
                if len(data_line.strip().split()) < 5:
                    raise ValueError(
                        f"Expected particle data with 5 columns at time {time_value}, got: '{data_line.strip()}'"
                    )

    if not times:
        raise ValueError(f"No frames found in {dynamic_path}")
    return np.asarray(times, float), np.asarray(hits, float)


def load_run(sim_dir: Path) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    t, h, φ para una sola ejecución (una réplica).
    - Lectura como plot_hits: consume EXACTAMENTE len(states) por frame (incluye la fija).
    - φ excluye la central y usa r_min para móviles.
    """
    L, states, r_min, r_max, declared_N = read_static(sim_dir)

    states_N = len(states)               # cantidad REAL de líneas F/M
    if states_N < declared_N:
        raise ValueError(
            f"static.txt declara N={declared_N} pero solo hay {states_N} estados en {sim_dir/'static.txt'}."
        )
    # φ sin la central:
    phi = packing_fraction_excluding_central(L, r_min, states_N)

    # Lectura dinámica: EXACTAMENTE len(states) líneas por frame (para no desfasar)
    t, h = read_hits(sim_dir, states_N)

    return t, h, phi


# ---------- Ajuste lineal y barrido en b ----------
def sse_given_b(t: np.ndarray, y: np.ndarray, b: float) -> Tuple[float, float]:
    """Devuelve SSE(b) y a*(b)=mean(y-bt)."""
    a_star = float(np.mean(y - b * t))
    resid = y - (a_star + b * t)
    return float(resid @ resid), a_star


def scan_b_and_minimize(t: np.ndarray, y: np.ndarray, ngrid: int = 801) -> Tuple[float, float, float]:
    """Devuelve b*, a*, SSE* escaneando alrededor de OLS."""
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


# ---------- Q (pendiente) por base con replicación ----------
def q_phi_for_base(base_name: str, tmark: float) -> Tuple[float, float, float, float]:
    """Para una base: computa Q=b* por réplica (t>=tmark), devuelve
       (phi_mean, phi_std, Q_mean, Q_std). Espera réplicas base_1..base_3."""
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

    # Recolectar puntos (phi_mean, phi_std, Q_mean, Q_std) por base
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

    # --- Plot Q vs φ (mismo color para todos) ---
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