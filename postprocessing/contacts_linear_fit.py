#!/usr/bin/env python3
"""Ajuste lineal sobre curvas PROMEDIO de hits: (A) Error vs b y (B) rectas de ajuste (t>=t0)."""

from __future__ import annotations
import argparse
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

# --- Rutas base (idéntico a plot_hits.py) ---
THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parent.parent
SIM_BASE = REPO_ROOT / "data" / "simulations"

# ---------- Lectura idéntica a plot_hits.py ----------
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
    # Nota: si hay más estados que N, luego usaremos exactamente declared_N.
    return L, states, r_min, r_max, declared_N


# ---------- ϕ física SOLO con radios impenetrables ----------
def packing_fraction(L: float, r_min: float, declared_N: int) -> float:
    """ϕ = (N * π r_min^2) / L^2 usando exactamente N discos impenetrables."""
    area = declared_N * math.pi * (r_min * r_min)
    return area / (L * L)


def read_hits(sim_dir: Path, particle_count: int) -> Tuple[List[float], List[int]]:
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

            # Leer EXACTAMENTE N líneas de partículas y validar columnas
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
    return times, hits


def load_run(sim_dir: Path) -> Tuple[np.ndarray, np.ndarray, float, int]:
    L, states, r_min, r_max, declared_N = read_static(sim_dir)

    # Usar la cantidad REAL de líneas F/M para leer el dynamic.txt
    states_N = len(states)
    if states_N < declared_N:
        raise ValueError(
            f"static.txt declara N={declared_N} pero solo hay {states_N} estados en {sim_dir/'static.txt'}."
        )

    # φ: mantené tu definición actual (solo r_min y N declarado)
    phi = packing_fraction(L, r_min, declared_N)

    # Lectura de dinámico: EXACTAMENTE len(states) líneas por frame (incluida la central)
    t, h = read_hits(sim_dir, states_N)

    return np.asarray(t, dtype=float), np.asarray(h, dtype=float), phi, declared_N

# ---------- Helpers de ajuste (inspirado en galaxy_linear_fit.py) ----------
def sse_given_b(t: np.ndarray, y: np.ndarray, b: float) -> Tuple[float, float]:
    """Devuelve SSE(b) y a*(b)=mean(y-bt)."""
    a_star = float(np.mean(y - b * t))
    resid = y - (a_star + b * t)
    return float(resid @ resid), a_star


def scan_b_and_minimize(t: np.ndarray, y: np.ndarray, ngrid: int = 801) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
    """Escanea b alrededor del OLS y devuelve b*, a*, SSE*, y las grillas (b_grid, sse_grid)."""
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
    return float(b_grid[i_min]), float(a_grid[i_min]), float(sse_grid[i_min]), b_grid, sse_grid

# ---------- Lógica principal ----------
def average_hits_of_base(base_name: str) -> Tuple[np.ndarray, np.ndarray, float, int]:
    """Promedia las 3 réplicas <base>_1.._3 y retorna (t_ref, hits_prom, phi_prom, N)."""
    replicas = [f"{base_name}_{i}" for i in range(1, 4)]
    times_list: List[np.ndarray] = []
    hits_list: List[np.ndarray] = []
    phis: List[float] = []
    Ns: List[int] = []

    for rep in replicas:
        sim_dir = SIM_BASE / rep
        if not sim_dir.exists():
            raise FileNotFoundError(f"No existe la carpeta de simulación: {sim_dir}")
        t, h, phi, N = load_run(sim_dir)
        times_list.append(t)
        hits_list.append(h)
        phis.append(phi)
        Ns.append(N)

    # Validar timestamps idénticos
    tref = times_list[0]
    for tr, rep in zip(times_list[1:], replicas[1:]):
        if len(tr) != len(tref) or not np.allclose(tr, tref, rtol=1e-9, atol=1e-9):
            raise ValueError(f"Las réplicas de '{base_name}' no comparten los mismos timestamps (ver {rep}).")

    if len(set(Ns)) != 1:
        raise ValueError(f"Se esperaban Ns idénticos en las réplicas de '{base_name}', obtuve {Ns}.")

    H = np.vstack(hits_list)  # (R, T)
    h_mean = H.mean(axis=0)
    return tref, h_mean, float(np.mean(phis)), int(Ns[0])


def main():
    ap = argparse.ArgumentParser(description="Error vs b y líneas de ajuste sobre CURVAS PROMEDIO de hits (t>=t0).")
    ap.add_argument("simulations", nargs="+", help="Nombres base; se esperan réplicas '<name>_1', '_2', '_3' en data/simulations/")
    ap.add_argument("--t0", type=float, default=20.0, help="Inicio de ventana temporal para la línea vertical y etiqueta (gráfico).")
    ap.add_argument("--tmark", type=float, default=20.0, help="Inicio del estacionario para el ajuste (cálculo).")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "graphics" / "hits", help="Carpeta de salida de gráficos.")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    # Construir una lista de curvas promedio (una por base)
    curves: List[Tuple[str, np.ndarray, np.ndarray, float, int]] = []  # (label, t, y, phi, N)
    for base in args.simulations:
        t, y, phi, N = average_hits_of_base(base)
        label = f"N={N} - φ={phi:.4f}"
        curves.append((label, t, y, phi, N))

    # Paleta consistente por N (si hay varios base con mismo N, comparten color)
    Ns_unique = sorted(set(N for _, _, _, _, N in curves))
    prop_colors = plt.rcParams['axes.prop_cycle'].by_key().get('color', [f"C{i}" for i in range(10)])
    color_of: Dict[int, str] = {N: prop_colors[i % len(prop_colors)] for i, N in enumerate(Ns_unique)}

    # ---------- (A) Error vs b por curva promedio ----------
    plt.figure(figsize=(9.2, 5.2))
    grids = []
    for label, t, y, _, N in curves:
        # *** ÚNICO CAMBIO: usar tmark para el estacionario ***
        mask = t >= args.tmark
        t_win = t[mask]
        y_win = y[mask]
        b_star, a_star, sse_star, b_grid, sse_grid = scan_b_and_minimize(t_win, y_win)
        color = color_of[N]
        plt.plot(b_grid, sse_grid, lw=2, color=color, label=label)
        i_min = int(np.argmin(sse_grid))
        plt.scatter([b_grid[i_min]], [sse_grid[i_min]], s=26, color=color, zorder=3)
        grids.append((label, b_grid, sse_grid, b_star, a_star))

    plt.xlabel(r"$Q$ (1/s)", fontsize=14)
    plt.ylabel("Error", fontsize=14)
    plt.tick_params(axis='both', labelsize=14)
    plt.legend(ncol=1, frameon=True, fontsize=14)
    ax = plt.gca()

    fmt = ScalarFormatter(useMathText=True)
    fmt.set_scientific(True)
    fmt.set_powerlimits((0, 0))   # obliga 10^{k} siempre
    ax.yaxis.set_major_formatter(fmt)
    ax.yaxis.get_offset_text().set_fontsize(14)

    # Pequeño auto-zoom alrededor del mínimo global para ver bien las curvas
    xs = np.concatenate([g[1] for g in grids])
    ys = np.concatenate([g[2] for g in grids])
    imin = int(np.nanargmin(ys))
    b_star_global = float(xs[imin])
    xr = ax.get_xlim()
    dx = max(0.1 * (xr[1] - xr[0]), 0.1)
    ax.set_xlim(b_star_global - dx, b_star_global + dx)
    # Y-lims enfocados
    xlo, xhi = ax.get_xlim()
    ys_win = []
    for _, b_grid, sse_grid, _, _ in grids:
        m = (b_grid >= xlo) & (b_grid <= xhi)
        if np.any(m):
            ys_win.append(sse_grid[m])
    if ys_win:
        ys_win = np.concatenate(ys_win)
        y_min = float(np.nanmin(ys_win))
        y_p95 = float(np.nanpercentile(ys_win, 5))
        span = max(y_p95 - y_min, 1e-9)
        ax.set_ylim(y_min - 0.15 * span, y_p95 + 0.1 * span)

    plt.tight_layout()
    f1 = args.out / "error_vs_q.png"
    plt.savefig(f1, dpi=160)
    plt.show()
    plt.close()
    print(f"[OK] Gráfico guardado: {f1}")

    # ---------- (B) Rectas de ajuste sobre cada curva promedio ----------
    plt.figure(figsize=(9.2, 5.2))
    ax = plt.gca()
    plt.axvline(args.tmark, linestyle="--", color="0.4", lw=1)

    for label, t, y, _, N in curves:
        # *** ÚNICO CAMBIO: usar tmark para el estacionario ***
        mask = t >= args.tmark
        t_win = t[mask]
        y_win = y[mask]
        b_star, a_star, sse_star, _, _ = scan_b_and_minimize(t_win, y_win)
        color = color_of[N]
        # curva original (suave)
        plt.plot(t, y, lw=1.2, alpha=0.30, color=color)
        # recta ajustada en la ventana (definida por tmark)
        y_fit = a_star + b_star * t_win
        plt.plot(t_win, y_fit, lw=2.5, color=color, label=f"{label}")

    plt.xlabel("Tiempo (s)", fontsize=14)
    plt.ylabel("Contactos Únicos", fontsize=14)
    plt.tick_params(axis='both', labelsize=14)
    plt.legend(ncol=1, frameon=True, fontsize=14)

    y_top = ax.get_ylim()[1]
    ax.annotate(
        f"t = {args.tmark:g} s",
        xy=(args.tmark, y_top), xycoords="data",
        xytext=(0, -20), textcoords="offset points",
        ha="center", va="top",
        fontsize=14,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.9),
        clip_on=True,
    )

    plt.tight_layout()
    f2 = args.out / "fit_lines.png"
    plt.savefig(f2, dpi=160)
    plt.show()
    plt.close()
    print(f"[OK] Gráfico guardado: {f2}")


if __name__ == "__main__":
    main()
