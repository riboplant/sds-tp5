import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib import animation
import matplotlib.colors as mcolors

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parent.parent
BASE_PATH = REPO_ROOT / 'data' / 'simulations'


def read_static(static_path: Path):
    with static_path.open('r') as f:
        lines = [line.strip() for line in f if line.strip() != '']
    if len(lines) < 3:
        raise ValueError(f"static.txt at {static_path} has {len(lines)} lines, expected 3.")
    L = float(lines[0])
    N = int(lines[1])
    R = float(lines[2])
    return L, N, R


def read_dynamic(dynamic_path: Path, N: int):
    frames_t = []
    frames = []

    with dynamic_path.open('r') as f:
        while True:
            t_line = f.readline()
            if not t_line:
                break  # EOF
            t_line = t_line.strip()
            if t_line == '':
                continue

            try:
                t = float(t_line)
            except ValueError as e:
                raise ValueError(f"Expected a time value, got '{t_line}'") from e

            xs, ys, vxs, vys = [], [], [], []
            for i in range(N):
                data_line = f.readline()
                if not data_line:
                    raise ValueError(f"Unexpected EOF while reading particle {i+1}/{N} at time {t}")
                parts = data_line.strip().split()
                if len(parts) < 4:
                    raise ValueError(f"Expected x y vx vy on line: '{data_line.strip()}'")
                x, y, vx, vy = map(float, parts[:4])
                xs.append(x); ys.append(y); vxs.append(vx); vys.append(vy)

            frames_t.append(t)
            frames.append((xs, ys, vxs, vys))

    return frames_t, frames


def make_animation(L, N, R, times, frames, slow_factor: float):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    ax.set_aspect('equal', adjustable='box')
    ax.set_axis_off()
    border = Rectangle((0, 0), L, L, fill=False, linewidth=1.5)
    ax.add_patch(border)
    edge_color = 'C0'
    face_rgba = mcolors.to_rgba('C0', 0.25)

    patches = [Circle((0, 0), R, facecolor=face_rgba, edgecolor=edge_color, linewidth=2.0) for _ in range(N)]
    for c in patches:
        ax.add_patch(c)

    # Velocity arrow scale
    vel_scale = 0.30
    zeros = [0.0] * N
    quiv = ax.quiver(
        zeros, zeros, zeros, zeros,
        angles='xy', scale_units='xy', scale=1.0,
        width=0.003, color=edge_color
    )

    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, va='top')

    def init():
        time_text.set_text('')
        for c in patches:
            c.center = (L + 1, L + 1)
        quiv.set_offsets(list(zip(zeros, zeros)))
        quiv.set_UVC(zeros, zeros)
        return (*patches, quiv, time_text)

    def update(frame_idx):
        xs, ys, vxs, vys = frames[frame_idx]

        for i, c in enumerate(patches):
            c.center = (xs[i], ys[i])

        vxs_s = [vx * vel_scale for vx in vxs]
        vys_s = [vy * vel_scale for vy in vys]

        quiv.set_offsets(list(zip(xs, ys)))
        quiv.set_UVC(vxs_s, vys_s)

        time_text.set_text(f't = {times[frame_idx]:.3f} s')
        return (*patches, quiv, time_text)

    base_interval_ms = 20
    interval_ms = max(1, int(base_interval_ms * slow_factor))

    ani = animation.FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=len(frames),
        interval=interval_ms,
        blit=True
    )

    return fig, ani


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sim', required=True, help='Simulation name (folder under data/simulations)')
    parser.add_argument('--slow', type=float, default=1.0, help='Factor para ralentizar la animación (2.0 = 2x más lenta)')
    args = parser.parse_args()

    sim_dir = BASE_PATH / args.sim
    static_path = sim_dir / 'static.txt'
    dynamic_path = sim_dir / 'dynamic.txt'

    if not static_path.exists():
        print(f"ERROR: static file not found: {static_path}", file=sys.stderr)
        sys.exit(1)
    if not dynamic_path.exists():
        print(f"ERROR: dynamic file not found: {dynamic_path}", file=sys.stderr)
        sys.exit(1)

    L, N, R = read_static(static_path)
    times, frames = read_dynamic(dynamic_path, N)

    if len(frames) == 0:
        print("No frames found in dynamic.txt. Nothing to animate.", file=sys.stderr)
        sys.exit(1)

    fig, ani = make_animation(L, N, R, times, frames, args.slow)
    plt.tight_layout()

    out_dir = REPO_ROOT / 'data' / 'animations'
    out_dir.mkdir(parents=True, exist_ok=True)

    _base_interval_ms = 20
    _interval_ms = max(1, int(_base_interval_ms * args.slow))
    _fps = max(1, int(1000 / _interval_ms))

    outfile = out_dir / f'{args.sim}.mp4'
    try:
        from matplotlib.animation import FFMpegWriter
        ani.save(outfile, writer=FFMpegWriter(fps=_fps))
    except Exception:
        from matplotlib.animation import PillowWriter
        outfile = out_dir / f'{args.sim}.gif'
        ani.save(outfile, writer=PillowWriter(fps=_fps))
    print(f'Animación guardada en {outfile}')

    plt.show()


if __name__ == '__main__':
    main()
