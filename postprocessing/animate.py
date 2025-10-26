import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib import animation

# -------- Configuration --------
# Resolve BASE_PATH relative to *this file*, not the current working directory.
THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parent.parent
BASE_PATH = REPO_ROOT / 'data' / 'simulations'


def read_static(static_path: Path):
    """Read L, N, R from static.txt"""
    with static_path.open('r') as f:
        lines = [line.strip() for line in f if line.strip() != '']
    if len(lines) < 3:
        raise ValueError(f"static.txt at {static_path} has {len(lines)} lines, expected 3.")
    L = float(lines[0])
    N = int(lines[1])
    R = float(lines[2])  # 'fixedRadius' in the Java code; use it as particle radius for the simple animation
    return L, N, R


def read_dynamic(dynamic_path: Path, N: int):
    """
    Parse dynamic.txt into a list of (t, positions) where positions is an Nx2 list.
    Each block is:
      t
      x y vx vy  (N lines; we only use x y)
    """
    frames_t = []
    frames_xy = []

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

            xs = []
            ys = []
            for i in range(N):
                data_line = f.readline()
                if not data_line:
                    raise ValueError(f"Unexpected EOF while reading particle {i+1}/{N} at time {t}")
                parts = data_line.strip().split()
                if len(parts) < 2:
                    raise ValueError(f"Expected at least x y on line: '{data_line.strip()}'")
                x, y = float(parts[0]), float(parts[1])
                xs.append(x)
                ys.append(y)

            frames_t.append(t)
            frames_xy.append((xs, ys))

    return frames_t, frames_xy


def make_animation(L, N, R, times, frames_xy, slow_factor: float):
    """
    Build a Matplotlib animation with:
      - a square box of size L x L
      - N particles drawn as circles with radius R
      - speed scaled by `slow_factor` (e.g., 2.0 => twice slower)
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title('TP5 Simulation')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    # Draw domain border explicitly so it looks like a box
    border = Rectangle((0, 0), L, L, fill=False, linewidth=1.5)
    ax.add_patch(border)

    # Particle patches
    patches = [Circle((0, 0), R, fill=True, alpha=0.7) for _ in range(N)]
    for c in patches:
        ax.add_patch(c)

    # Time text
    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, va='top')

    def init():
        time_text.set_text('')
        for c in patches:
            c.center = (L+1, L+1)  # start off-canvas
        return (*patches, time_text)

    def update(frame_idx):
        xs, ys = frames_xy[frame_idx]
        for i, c in enumerate(patches):
            c.center = (xs[i], ys[i])
        time_text.set_text(f't = {times[frame_idx]:.3f} s')
        return (*patches, time_text)

    # Base interval (ms). Multiply by slow_factor to slow down visually.
    base_interval_ms = 20
    interval_ms = max(1, int(base_interval_ms * slow_factor))

    ani = animation.FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=len(frames_xy),
        interval=interval_ms,
        blit=True
    )

    return fig, ani


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sim', required=True, help='Simulation name (folder under data/simulations)')
    parser.add_argument('--slow', type=float, default=1.0, help='Factor to slow down the animation (e.g., 2.0 = 2x slower)')
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
    times, frames_xy = read_dynamic(dynamic_path, N)

    if len(frames_xy) == 0:
        print("No frames found in dynamic.txt. Nothing to animate.", file=sys.stderr)
        sys.exit(1)

    fig, ani = make_animation(L, N, R, times, frames_xy, args.slow)
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
        # Fallback a GIF si no hay ffmpeg
        from matplotlib.animation import PillowWriter
        outfile = out_dir / f'{args.sim}.gif'
        ani.save(outfile, writer=PillowWriter(fps=_fps))

    print(f'Animación guardada en {outfile}')
    plt.show()


if __name__ == '__main__':
    main()
