#!/usr/bin/env python3
"""Plot cumulative central-particle hits vs time averaging over triplicate runs, con inset [0, 40] s."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parent.parent
SIM_BASE = REPO_ROOT / "data" / "simulations"
OUT_DIR = REPO_ROOT / "data" / "graphics"

# ---- Marca vertical fija (sin pasar por parámetro) ----
T_MARK = 20.0  # segundos


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
    return L, states, r_min, r_max, declared_N


def packing_fraction(L: float, states: Iterable[str], r_min: float, declared_N: int,
                     central_is_obstacle: bool = False, central_radius: float | None = None) -> float:
    """ϕ física: suma de discos impenetrables sobre el área del dominio."""
    area = 0.0
    # Usa exactamente N estados (por si el archivo trae líneas extra).
    for i, _ in enumerate(states[:declared_N]):
        if central_is_obstacle and i == 0:
            r = central_radius if central_radius is not None else r_min
        else:
            r = r_min
        area += math.pi * r * r
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


def load_simulation(sim_dir: Path) -> Tuple[List[float], List[int], float, int]:
    """Return time series plus metadata for a single simulation run."""
    L, states, r_min, r_max, declared_N = read_static(sim_dir)
    phi = packing_fraction(L, states, r_min, declared_N, central_is_obstacle=False)
    times, hits = read_hits(sim_dir, len(states))
    return times, hits, phi, declared_N


def plot_hits(sim_names: List[str]) -> None:
    # almacenamos las series para re-usarlas en el inset
    series = []  # (times: np.ndarray, hits: np.ndarray, color: str, label: str)

    fig, ax = plt.subplots(figsize=(12, 6))

    for sim_name in sim_names:
        replicate_names = [f"{sim_name}_{i}" for i in range(1, 4)]
        replicate_times: List[List[float]] = []
        replicate_hits: List[List[int]] = []
        phis: List[float] = []
        population_sizes: List[int] = []

        for replicate in replicate_names:
            sim_dir = SIM_BASE / replicate
            if not sim_dir.exists():
                raise FileNotFoundError(f"Simulation directory not found: {sim_dir}")

            times, hits, phi, declared_N = load_simulation(sim_dir)
            replicate_times.append(times)
            replicate_hits.append(hits)
            phis.append(phi)
            population_sizes.append(declared_N)

        reference_times = replicate_times[0]
        for other_times, replicate in zip(replicate_times[1:], replicate_names[1:]):
            if len(other_times) != len(reference_times) or not all(
                math.isclose(t_ref, t_other, rel_tol=1e-9, abs_tol=1e-9)
                for t_ref, t_other in zip(reference_times, other_times)
            ):
                raise ValueError(
                    "All replicates must share identical timestamps. "
                    f"Mismatch found between '{replicate_names[0]}' and '{replicate}'."
                )

        averaged_hits = [sum(values) / len(values) for values in zip(*replicate_hits)]
        if len(set(population_sizes)) != 1:
            raise ValueError(
                f"Expected identical particle counts across replicates for '{sim_name}', got {population_sizes}."
            )
        averaged_phi = sum(phis) / len(phis)
        label = f"N={population_sizes[0]} - φ={averaged_phi:.4f}"

        # plot principal y guardamos color/serie para el inset
        line, = ax.plot(reference_times, averaged_hits, label=label)
        series.append(
            (
                np.asarray(reference_times, dtype=float),
                np.asarray(averaged_hits, dtype=float),
                line.get_color(),
                label,
            )
        )

    # ----- Inset zoom del transitorio (t in [0, 50]) -----
    axins = inset_axes(ax, width="30%", height="30%", loc="upper center", borderpad=0.8)

    x0, x1 = 0.0, 50.0
    ymins, ymaxs = [], []
    for t, y, color, _ in series:
        axins.plot(t, y, color=color, lw=1.0)
        m = (t >= x0) & (t <= x1)
        if np.any(m):
            ymins.append(float(y[m].min()))
            ymaxs.append(float(y[m].max()))

    axins.set_xlim(x0, x1)
    if ymins and ymaxs:
        pad = 0.05 * (max(ymaxs) - min(ymins) if max(ymaxs) > min(ymins) else 1.0)
        axins.set_ylim(min(ymins) - pad, max(ymaxs) + pad)

    # ticks visibles (sin títulos) con tamaño 14 (dejé tu valor actual)
    axins.set_xlabel("")
    axins.set_ylabel("")
    axins.tick_params(axis="both", which="both", labelsize=14, length=3)
    axins.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

    # líneas/rectángulo que indican la región del zoom (dos esquinas opuestas)
    try:
        ax.indicate_inset_zoom(axins, edgecolor="0.4")
        mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.4", lw=1.0)
    except Exception:
        pass

    # ----- Línea vertical punteada y etiqueta en t = T_MARK -----
    # (hacemos esto tras ajustar límites del inset para tener el ylim real del eje principal)
    ax.axvline(T_MARK, linestyle="--", color="0.25", lw=1.2, dashes=(4, 3))
    y_top = ax.get_ylim()[1]
    ax.annotate(
        f"t = {T_MARK:g} s",
        xy=(T_MARK, y_top),
        xytext=(0, -20),           # 6 px por encima del borde superior
        textcoords="offset points",
        ha="center", va="bottom",
        fontsize=14,
        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85),
        clip_on=False,
    )

    # Si T_MARK cae dentro de la ventana del inset, dibujamos la línea también allí
    x0_in, x1_in = axins.get_xlim()
    if x0_in <= T_MARK <= x1_in:
        axins.axvline(T_MARK, linestyle="--", color="0.25", lw=1.0, dashes=(4, 3))

    # ----- Etiquetas y guardado (en el eje principal) -----
    ax.set_xlabel("Tiempo (s)", fontsize=14)
    ax.set_ylabel("Contactos Únicos", fontsize=14)
    ax.tick_params(axis='both', labelsize=14)
    #ax.legend(fontsize=14)  # leyenda como antes
    leg = ax.legend(
        loc="upper right",            # arriba a la derecha
        bbox_to_anchor=(0.98, 0.98),  # un pelín hacia adentro del borde
        bbox_transform=ax.transAxes,  # fijo al eje (no a los datos)
        fontsize=14,
        frameon=True,
    )
    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_alpha(0.9)

    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "central_hits_vs_time.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Figura guardada en: {out_path}")
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot cumulative central-particle hits versus time, averaging three replicates per simulation."
    )
    parser.add_argument(
        "simulations",
        nargs="+",
        help="Base simulation names; each expects sub-runs '<name>_1', '<name>_2', '<name>_3' under data/simulations",
    )
    args = parser.parse_args()
    plot_hits(args.simulations)


if __name__ == "__main__":
    main()
