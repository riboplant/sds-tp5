from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parent.parent
SIM_BASE = REPO_ROOT / "data" / "simulations"
OUT_DIR = REPO_ROOT / "data" / "graphics"


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


def packing_fraction(L: float, states: Iterable[str], r_min: float, declared_N: int,
                     central_is_obstacle: bool = False, central_radius: float | None = None) -> float:
    """ϕ física: suma de discos impenetrables sobre el área del dominio."""
    area = 0.0
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
    return times, hits


def load_simulation(sim_dir: Path) -> Tuple[List[float], List[int], float, int]:
    """Return time series plus metadata for a single simulation run."""
    L, states, r_min, r_max, declared_N = read_static(sim_dir)
    phi = packing_fraction(L, states, r_min, declared_N, central_is_obstacle=False)
    times, hits = read_hits(sim_dir, len(states))
    return times, hits, phi, declared_N


def plot_hits(sim_names: List[str], t_mark: float = 20.0) -> None:
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
                raise FileNotFoundError(f"No se encontró el directorio de la simulación: {sim_dir}")

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
                    "Todas las réplicas deben compartir timestamps idénticos. "
                    f"Se detectó una discrepancia entre '{replicate_names[0]}' y '{replicate}'."
                )

        averaged_hits = [sum(values) / len(values) for values in zip(*replicate_hits)]
        if len(set(population_sizes)) != 1:
            raise ValueError(
                f"Se esperaban cantidades de partículas idénticas entre réplicas para '{sim_name}', pero se obtuvieron {population_sizes}."
            )
        averaged_phi = sum(phis) / len(phis)
        label = f"N={population_sizes[0]} - φ={averaged_phi:.4f}"

        ax.plot(reference_times, averaged_hits, label=label)

    ax.axvline(t_mark, linestyle="--", color="0.25", lw=1.2, dashes=(4, 3))
    y_top = ax.get_ylim()[1]
    ax.annotate(
        f"t = {t_mark:g} s",
        xy=(t_mark, y_top),
        xytext=(0, -20),
        textcoords="offset points",
        ha="center", va="bottom",
        fontsize=14,
        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85),
        clip_on=False,
    )

    ax.set_xlabel("Tiempo (s)", fontsize=14)
    ax.set_ylabel("$N_c(t)$", fontsize=14)
    ax.tick_params(axis='both', labelsize=14)
    leg = ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=1,
        fontsize=14,
        frameon=True,
    )
    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_alpha(0.9)


    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "central_hits_vs_time_noinset.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Figura guardada en: {out_path}")
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot cumulative central-particle hits versus time, averaging three replicates per simulation (sin inset)."
    )
    parser.add_argument(
        "simulations",
        nargs="+",
        help="Base simulation names; each expects sub-runs '<name>_1', '<name>_2', '<name>_3' under data/simulations",
    )
    parser.add_argument(
        "--tmark",
        type=float,
        default=20.0,
        help="Tiempo (s) donde dibujar la línea vertical punteada (default: 20).",
    )
    args = parser.parse_args()
    plot_hits(args.simulations, t_mark=args.tmark)


if __name__ == "__main__":
    main()
