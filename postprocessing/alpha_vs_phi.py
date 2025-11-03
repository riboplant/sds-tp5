#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
alpha_vs_phi.py — v6 (formato estricto + acumulado, τ>0 y ajuste powerlaw)

- Recibe 1+ prefijos de grupo: para cada PREFIJO asume carpetas PREFIJO_1, _2, _3
  dentro de data/simulations/.
- static.txt:
    línea 1: L
    línea 2: N (partículas en movimiento)
    línea 3: r_min (AACPM)
    línea 4: r_max (AACPM)
    líneas 5..(N+5-1): etiquetas 'F'/'M' (N+1 en total; la central es extra)
- dynamic.txt (por bloques):
    cabecera: "t  k_acum"  (k_acum = contactos acumulados hasta t)
    luego N+1 líneas: "x y vx vy r touched"  (se saltean)
  Contacto único: si k_acum(t) > k_acum(prev), registrar **una** vez ese t.
- t0 = 20
- τ = diferencias entre tiempos ordenados; se filtran τ ≤ 0 (para ajuste continuo).
- Intercalado round-robin de los τ de las 3 simulaciones del grupo.
- φ = (N * π * r_min^2) / L^2   (por defecto **NO** cuenta a la central).
- Ajuste power-law usando powerlaw.Fit (continuous): estima xmin y α por KS/MLE.
- Incertidumbre σ_α y p-valor de KS provistos por powerlaw.
- Salida: data/simulations/alpha_vs_phi_<TIMESTAMP>.png
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt
import powerlaw

# --- Fuente global 14pt ---
plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 14,
    "axes.labelsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "figure.titlesize": 14,
})

# ---- Config ----
REPO_ROOT = Path(__file__).resolve().parent.parent
SIM_BASE = REPO_ROOT / "data" / "simulations"
GRAPH_BASE = REPO_ROOT / "data" / "graphs"
T0 = 20.0

# φ: incluir a la partícula central? (el enunciado usualmente usa N en movimiento)
INCLUDE_CENTRAL_IN_PHI = False


# ========= Lectura (formato estricto) =========

@dataclass
class StaticInfo:
    L: float
    N_moving: int
    r_min: float
    r_max: float
    tags: List[str]  # 'F'/'M', largo N+1


def read_static_strict(sim_dir: Path) -> StaticInfo:
    path = sim_dir / "static.txt"
    if not path.exists():
        raise FileNotFoundError(f"Falta {path}")

    lines = [ln.strip() for ln in path.read_text(encoding="utf-8", errors="ignore").splitlines() if ln.strip() != ""]
    if len(lines) < 4:
        raise ValueError(f"{path}: se requieren al menos 4 líneas (L, N, r_min, r_max).")

    try:
        L = float(lines[0])
        N = int(float(lines[1]))     # por si viene '600.0'
        r_min = float(lines[2])
        r_max = float(lines[3])
    except Exception as e:
        raise ValueError(f"{path}: no pude parsear L/N/r_min/r_max: {e}")

    expected = N + 1
    if len(lines) < 4 + expected:
        raise ValueError(f"{path}: se esperaban {expected} etiquetas F/M y hay {len(lines)-4}.")

    tags = [lines[4 + i].split()[0].upper() for i in range(expected)]
    return StaticInfo(L=L, N_moving=N, r_min=r_min, r_max=r_max, tags=tags)


def phi_from_static(si: StaticInfo) -> float:
    N_total = si.N_moving + (1 if INCLUDE_CENTRAL_IN_PHI else 0)
    return float(N_total * np.pi * (si.r_min ** 2) / (si.L ** 2))


def read_contact_times_from_dynamic(sim_dir: Path, tags: List[str], t0: float = T0) -> np.ndarray:
    """
    dynamic.txt por bloques:
      cabecera: t  k_acum
      luego len(tags) líneas: x y vx vy r touched (se saltean)
    Contacto único: si k_acum sube respecto del bloque anterior, registrar UNA vez t.
    """
    path = sim_dir / "dynamic.txt"
    if not path.exists():
        raise FileNotFoundError(f"Falta {path}")

    n_lines_block = len(tags)
    times: List[float] = []

    it = iter(path.read_text(encoding="utf-8", errors="ignore").splitlines())
    prev_k = 0

    while True:
        try:
            header = next(it)
        except StopIteration:
            break

        header = header.strip()
        if not header:
            continue

        parts = header.replace(",", " ").split()
        if len(parts) < 2:
            raise ValueError(f"{path}: cabecera inválida (esperaba 't k_acum'): '{header}'")
        try:
            t = float(parts[0])
            k_acum = int(float(parts[1]))
        except Exception:
            raise ValueError(f"{path}: no pude parsear 't k_acum' en '{header}'")

        # saltar N+1 líneas de estado
        for _ in range(n_lines_block):
            try:
                next(it)
            except StopIteration:
                raise ValueError(f"{path}: EOF dentro de un bloque (faltan {n_lines_block} líneas).")

        if k_acum > prev_k and t >= t0:
            times.append(t)

        prev_k = k_acum

    return np.asarray(sorted(times), dtype=float)


def taus_from_times(times: np.ndarray) -> np.ndarray:
    """τ_i = t_{i+1} - t_i, filtrando τ ≤ 0."""
    if times.size < 2:
        return np.array([], dtype=float)
    taus = np.diff(times)
    return taus[taus > 0]


def interleave_round_robin(arrs: Sequence[np.ndarray]) -> np.ndarray:
    """[a0,a1,...],[b0,...],[c0,...] -> [a0,b0,c1,a1,b1,c1,...] (ignora vacíos)."""
    max_len = max((len(a) for a in arrs), default=0)
    out: List[float] = []
    for i in range(max_len):
        for a in arrs:
            if i < len(a):
                out.append(float(a[i]))
    return np.array(out, dtype=float)


def fit_powerlaw_distribution(data: np.ndarray) -> tuple[float, float, float, float, float, int]:
    x = np.asarray(data, dtype=float)
    x = x[np.isfinite(x)]
    x = x[x > 0]
    if x.size < 50:
        raise ValueError("Se necesitan al menos 50 τ positivos para un ajuste power-law confiable.")

    try:
        fit = powerlaw.Fit(x, xmin=None, discrete=False, verbose=False)
    except Exception as exc:
        raise RuntimeError(f"powerlaw.Fit falló: {exc}")

    alpha = float(fit.power_law.alpha)
    sigma_alpha = float(getattr(fit.power_law, "sigma", float("nan")))
    xmin = float(fit.power_law.xmin)

    ks = float(getattr(fit.power_law, "D", float("nan")))
    p_value = float(getattr(fit.power_law, "KS_probability", float("nan")))

    try:
        ks_result = fit.power_law.KS(return_probability=True)
        if isinstance(ks_result, tuple):
            ks = float(ks_result[0])
            if len(ks_result) > 1:
                p_value = float(ks_result[1])
        else:
            ks = float(ks_result)
    except TypeError:
        ks = float(fit.power_law.KS())
    except Exception as exc:
        print(f"[powerlaw] Error obteniendo KS: {exc}")

    if not np.isfinite(p_value):
        try:
            p_value = float(fit.power_law.KS_significance)
        except Exception:
            pass

    n_tail = int(np.sum(x >= xmin)) if np.isfinite(xmin) else 0
    return alpha, sigma_alpha, xmin, ks, p_value, n_tail


# ========= Pipeline por grupo =========

@dataclass
class GroupResult:
    prefix: str
    phi: float
    alpha: float
    sigma_alpha: float
    xmin: float
    ks: float
    p_value: float
    n_tail: int
    n_tau: int
    taus: np.ndarray


def process_group(prefix: str, t0: float) -> GroupResult:
    sims = [SIM_BASE / f"{prefix}_{i}" for i in (1, 2, 3)]

    phis: List[float] = []
    taus_by_sim: List[np.ndarray] = []

    for sdir in sims:
        si = read_static_strict(sdir)
        phis.append(phi_from_static(si))

        times = read_contact_times_from_dynamic(sdir, tags=si.tags, t0=t0)
        taus_by_sim.append(taus_from_times(times))

    tau_interleaved = interleave_round_robin(taus_by_sim)
    if tau_interleaved.size == 0:
        raise ValueError(f"Grupo {prefix}: conjunto de τ vacío.")

    alpha, sigma_alpha, xmin, ks, p_value, n_tail = fit_powerlaw_distribution(tau_interleaved)

    return GroupResult(
        prefix=prefix,
        phi=float(np.mean(phis)),
        alpha=alpha,
        sigma_alpha=sigma_alpha,
        xmin=xmin,
        ks=ks,
        p_value=p_value,
        n_tail=n_tail,
        n_tau=int(tau_interleaved.size),
        taus=tau_interleaved,
    )


# ========= Main / Plot =========

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=f"α vs ϕ (t0 configurable, default {T0}) con acumulado y ajuste powerlaw.")
    ap.add_argument("--t0", type=float, default=T0, help="Tiempo mínimo para registrar contactos.")
    ap.add_argument("prefix", nargs="+", help="Prefijos de grupos (cada uno con *_1, *_2, *_3).")
    args = ap.parse_args(argv)

    results: List[GroupResult] = []
    for pref in args.prefix:
        try:
            res = process_group(pref, t0=args.t0)
            p_str = f"{res.p_value:.10f}" if np.isfinite(res.p_value) else "nan"
            print(
                f"[{pref}] phi={res.phi:.6f}  alpha={res.alpha:.4f} ± {res.sigma_alpha:.4f}  "
                f"xmin={res.xmin:.4g}  KS={res.ks:.4f}  p={p_str}  n_tail={res.n_tail}  n_tau={res.n_tau}"
            )
            results.append(res)
        except Exception as e:
            print(f"ERROR en grupo '{pref}': {e}")

    if not results:
        print("No hay resultados válidos para graficar.")
        return 2

    results.sort(key=lambda r: r.phi)

    # ---- Figura α vs φ ----
    fig, ax = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    x = [r.phi for r in results]
    y = [r.alpha for r in results]
    yerr = [r.sigma_alpha for r in results]
    ax.errorbar(x, y, yerr=yerr, fmt="o-", capsize=4, lw=1.5, ms=5)
    ax.set_xlabel(r"$\phi$", fontsize=14)
    ax.set_ylabel(r"$\alpha$", fontsize=14)
    ax.tick_params(axis="both", labelsize=14)
    ax.grid(True, alpha=0.3)

    # ---- Figura CCDF log–log ----
    fig_log, ax_log = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    plotted_any = False
    for r in results:
        tau = np.asarray(r.taus, dtype=float)
        tau = tau[np.isfinite(tau)]
        tau = tau[tau > 0]
        if tau.size == 0:
            continue
        tau_sorted = np.sort(tau)
        ccdf = np.arange(tau_sorted.size, 0, -1) / tau_sorted.size
        label_main = f"φ={r.phi:.3f}"
        line = ax_log.plot(tau_sorted, ccdf, "--", linewidth=1.2, label=label_main)[0]
        color = line.get_color()
        if np.isfinite(r.xmin) and np.isfinite(r.alpha) and r.alpha > 1.0:
            xmin_fit = float(r.xmin)
            tau_tail = tau_sorted[tau_sorted >= xmin_fit]
            if tau_tail.size >= 2:
                xmax_fit = float(tau_tail[-1])
                x_fit = np.geomspace(xmin_fit, xmax_fit, 200)
                tail_fraction = tau_tail.size / tau_sorted.size
                y_fit = tail_fraction * np.power(x_fit / xmin_fit, 1.0 - r.alpha)
                ax_log.plot(x_fit, y_fit, color=color, linewidth=1.0, linestyle="-", alpha=0.8)
        plotted_any = True
    if plotted_any:
        ax_log.set_xscale("log")
        ax_log.set_yscale("log")
        ax_log.set_xlabel(r"$\tau$ (s)", fontsize=14)
        ax_log.set_ylabel(r"$P(\tau \geq \tau_{\mathrm{min}})$", fontsize=14)
        ax_log.tick_params(axis="both", labelsize=14)
        ax_log.grid(True, alpha=0.3)
        ax_log.legend(fontsize=14)
    else:
        plt.close(fig_log)

    # ---- Figura p-valor vs φ ----
    finite_results = [r for r in results if np.isfinite(r.p_value)]
    if finite_results:
        fig_p, ax_p = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
        x_p = [r.phi for r in finite_results]
        y_p = [r.p_value for r in finite_results]
        ax_p.plot(x_p, y_p, "o-", lw=1.5, ms=5)
        ax_p.axhline(0.1, linestyle="--", color="gray", linewidth=1)
        ax_p.set_xlabel(r"$\phi$", fontsize=14)
        ax_p.set_ylabel("p-valor", fontsize=14)
        ax_p.tick_params(axis="both", labelsize=14)
        ax_p.grid(True, alpha=0.3)

    # ---- Guardado ----
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = GRAPH_BASE / f"alpha_vs_phi_{ts}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    print(f"Figura guardada en: {out_path}")

    if plotted_any:
        out_path_log = GRAPH_BASE / f"log-log_{ts}.png"
        fig_log.savefig(out_path_log, dpi=200)
        print(f"Figura guardada en: {out_path_log}")

    if finite_results:
        out_path_p = GRAPH_BASE / f"p-value_vs_phi_{ts}.png"
        fig_p.savefig(out_path_p, dpi=200)
        print(f"Figura guardada en: {out_path_p}")

    plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
