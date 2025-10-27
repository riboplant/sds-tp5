#!/usr/bin/env python3
"""Plot cumulative central-particle hits vs time for one or more simulations."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable, List, Tuple

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parent.parent
SIM_BASE = REPO_ROOT / "data" / "simulations"


def read_static(sim_dir: Path) -> Tuple[float, List[str], float, float]:
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


def packing_fraction(L: float, states: Iterable[str], r_min: float, r_max: float) -> float:
    """Compute φ = Σ π r_i^2 / L², assuming fixed particles use r_max and moving ones r_min."""
    area = 0.0
    moving_radius = 0.5 * (r_min + r_max)
    for s in states:
        radius = r_max if s == "F" else moving_radius
        area += math.pi * radius * radius
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
                # we tolerate missing columns but enforce at least 5 to detect inconsistent files
                if len(data_line.strip().split()) < 5:
                    raise ValueError(
                        f"Expected particle data with 5 columns at time {time_value}, got: '{data_line.strip()}'"
                    )

    if not times:
        raise ValueError(f"No frames found in {dynamic_path}")
    return times, hits


def plot_hits(sim_names: List[str]) -> None:
    import matplotlib.pyplot as plt

    for sim_name in sim_names:
        sim_dir = SIM_BASE / sim_name
        if not sim_dir.exists():
            raise FileNotFoundError(f"Simulation directory not found: {sim_dir}")

        L, states, r_min, r_max, declared_N = read_static(sim_dir)
        phi = packing_fraction(L, states, r_min, r_max)
        times, hits = read_hits(sim_dir, len(states))

        label = f"N={declared_N} - φ={phi:.4f}"
        plt.plot(times, hits, label=label)

    plt.xlabel("Time [s]")
    plt.ylabel("Central hits (cumulative)")
    plt.title("Central particle hits vs time")
    plt.legend()
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    plt.tight_layout()
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot cumulative central-particle hits versus time for one or more simulations."
    )
    parser.add_argument(
        "simulations",
        nargs="+",
        help="Simulation folder names under data/simulations (e.g. test1 test2 ...)",
    )
    args = parser.parse_args()
    plot_hits(args.simulations)


if __name__ == "__main__":
    main()
