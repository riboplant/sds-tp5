# -*- coding: utf-8 -*-
# Animación simple (sin flechas) leyendo la salida del Engine.
# Estructura esperada (relativa a este archivo):
#   postprocessing/animate.py
#   data/simulations/<NOMBRE>/static.txt
#   data/simulations/<NOMBRE>/dynamic.txt
#
# static.txt: 3 líneas numéricas. Usamos la 1ª como L.
# dynamic.txt: bloques por tiempo:
#   - línea de tiempo: "t=...", "time: ...", "time ...", o UNA sola columna numérica
#   - luego N filas "id x y [ ... ]" (ignoramos columnas extra)
#
# Uso:
#   python3 postprocessing/animate.py --sim 1761433142746

import argparse
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def read_static_L(static_path: str) -> float:
    """Lee la 1ª línea numérica de static.txt como L."""
    nums = []
    with open(static_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                nums.append(float(s))
            except ValueError:
                # por si hubiera comentarios/labels, los ignoro
                pass
    if not nums:
        raise ValueError("static.txt no contiene números. Se esperaba 3 líneas numéricas (L, N, ...).")
    return float(nums[0])

_num = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"

def is_time_line(s: str) -> float | None:
    """
    Devuelve el tiempo como float si la línea 's' es una línea de tiempo,
    o None si no lo es. Acepta:
      - t=..., t: ..., time=..., time: ...
      - una única columna numérica
    """
    s = s.strip()
    if not s:
        return None
    # t=..., time: ...
    m = re.match(rf"^(?:t|time)\s*[:=]\s*({_num})\s*$", s, flags=re.IGNORECASE)
    if m:
        return float(m.group(1))
    # "time 0.10" o "t 0.10"
    m = re.match(rf"^(?:t|time)\s+({_num})\s*$", s, flags=re.IGNORECASE)
    if m:
        return float(m.group(1))
    # una sola columna numérica
    toks = s.split()
    if len(toks) == 1:
        try:
            return float(toks[0])
        except ValueError:
            return None
    return None

def smart_split_nums(s: str):
    """Divide por espacios o coma y devuelve solo tokens numéricos convertibles a float."""
    toks = re.split(r"[,\s]+", s.strip())
    out = []
    for t in toks:
        if not t:
            continue
        try:
            out.append(float(t))
        except ValueError:
            return None  # línea no puramente numérica
    return out

def read_dynamic_blocks(dynamic_path: str):
    """
    Parsea dynamic.txt en bloques de tiempo.
    Devuelve:
      times: np.ndarray de tiempos ordenados
      frames: dict[time] -> np.ndarray shape (M, 3) con columnas [id, x, y] ordenadas por id
      id_order: np.ndarray de ids globales ordenados
    """
    times = []
    frames_raw = {}  # time -> list of rows (id,x,y)
    current_t = None

    with open(dynamic_path, "r", encoding="utf-8") as f:
        for raw in f:
            s = raw.strip()
            if not s:
                continue

            t_val = is_time_line(s)
            if t_val is not None:
                current_t = t_val
                times.append(current_t)
                frames_raw[current_t] = []
                continue

            if current_t is None:
                # saltar basura previa al primer tiempo
                continue

            nums = smart_split_nums(s)
            if not nums:
                continue
            # esperamos al menos: id, x, y
            if len(nums) < 3:
                continue

            pid = int(round(nums[0]))
            x, y = float(nums[1]), float(nums[2])
            frames_raw[current_t].append((pid, x, y))

    if not times:
        raise ValueError("dynamic.txt: no se detectaron bloques de tiempo (líneas 't=...' o una sola columna numérica).")

    # ordenar tiempos y ordenar filas por id en cada frame
    times = np.array(sorted(times), dtype=float)
    frames = {}
    ids_set = set()
    for t in times:
        arr = np.array(frames_raw[t], dtype=float) if frames_raw[t] else np.zeros((0,3))
        if arr.size > 0:
            arr = arr[np.argsort(arr[:,0])]  # ordenar por id
            frames[t] = arr
            ids_set.update(arr[:,0].astype(int).tolist())
        else:
            frames[t] = arr

    id_order = np.array(sorted(list(ids_set)), dtype=int)
    return times, frames, id_order

def reindex_xy(arr: np.ndarray, id_order: np.ndarray):
    """arr: shape (M,3) [id,x,y] -> devuelve x,y ordenados según id_order (NaN si falta)."""
    N = len(id_order)
    x = np.full(N, np.nan)
    y = np.full(N, np.nan)
    if arr.size == 0:
        return x, y
    # mapa id -> (x,y)
    for row in arr:
        pid, rx, ry = int(row[0]), row[1], row[2]
        # búsqueda binaria del índice
        # (como N suele ser moderado, un dict también va bien; dejamos simple)
    pos = {int(row[0]): (row[1], row[2]) for row in arr}
    for k, pid in enumerate(id_order):
        if pid in pos:
            x[k], y[k] = pos[pid]
    return x, y

def main():
    ap = argparse.ArgumentParser(description="Animación simple de la salida del Engine (sin flechas)")
    ap.add_argument("--sim", required=True, help="Nombre de la simulación (carpeta dentro de data/simulations)")
    ap.add_argument("--fps", type=int, default=30, help="Frames por segundo (animación)")
    ap.add_argument("--point-size", type=float, default=20, help="Tamaño de los puntos")
    args = ap.parse_args()

    # Rutas relativas: postprocessing/ -> ../data/simulations/<sim>/
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "simulations", args.sim))
    static_path  = os.path.join(base_dir, "static.txt")
    dynamic_path = os.path.join(base_dir, "dynamic.txt")

    if not os.path.isfile(static_path):
        raise FileNotFoundError(f"No existe: {static_path}")
    if not os.path.isfile(dynamic_path):
        raise FileNotFoundError(f"No existe: {dynamic_path}")

    L = read_static_L(static_path)
    times, frames, id_order = read_dynamic_blocks(dynamic_path)

    # Primer frame
    first = frames[times[0]]
    x0, y0 = reindex_xy(first, id_order)

    # Figura simple
    fig, ax = plt.subplots()
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Simulación: {args.sim}")

    # Marco del dominio
    ax.plot([0, L, L, 0, 0], [0, 0, L, L, 0], color="black", lw=1.0)

    # Puntos
    scat = ax.scatter(x0, y0, s=args.point_size)

    # Texto de tiempo
    time_text = ax.text(0.02, 0.98, f"t = {times[0]:.3f}", transform=ax.transAxes, va="top")

    def update(k):
        t = times[k]
        arr = frames[t]
        x, y = reindex_xy(arr, id_order)
        scat.set_offsets(np.column_stack([x, y]))
        time_text.set_text(f"t = {t:.3f}")
        return scat, time_text

    anim = FuncAnimation(fig, update, frames=len(times), interval=1000/max(1, args.fps), blit=False, repeat=True)
    plt.show()

if __name__ == "__main__":
    main()
