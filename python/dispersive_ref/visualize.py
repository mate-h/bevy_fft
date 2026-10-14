"""Render free-surface height over time from a reference simulation."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .params import Params
from .step import Sim, step


def free_surface(sim: Sim) -> np.ndarray:
    """η = h + bed (meters above the bed datum used by init_basin)."""
    return sim.h + sim.bed


def run_height_history(
    p: Params,
    frames: int,
    *,
    progress: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Advance `frames` steps and return (eta[T,N,N], times[T]).

    Frame 0 is the initial condition before any hybrid step.
    """
    sim = Sim.create(p)
    eta0 = free_surface(sim)
    history = np.empty((frames + 1, p.n, p.n), dtype=np.float32)
    times = np.empty(frames + 1, dtype=np.float64)
    history[0] = eta0
    times[0] = 0.0
    for i in range(frames):
        step(sim)
        history[i + 1] = free_surface(sim)
        times[i + 1] = (i + 1) * p.dt
        if progress and (i + 1) % max(frames // 10, 1) == 0:
            print(
                f"frame {i+1}/{frames} η_max={history[i+1].max():.4f}",
                flush=True,
            )
    return history, times


def write_height_video(
    out_path: Path,
    history: np.ndarray,
    times: np.ndarray,
    *,
    fps: float = 30.0,
    stride: int = 1,
    cmap: str = "viridis",
    title: str = "dispersive_ref free surface",
) -> Path:
    """Write an MP4 (or GIF) of η(x,y,t) with a centerline cut."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    frames = history[:: max(stride, 1)]
    t = times[:: max(stride, 1)]
    n = frames.shape[1]
    mid = n // 2
    vmin = float(frames.min())
    vmax = float(frames.max())
    # Keep a small span so a flat field still maps.
    if vmax - vmin < 1e-6:
        vmax = vmin + 1e-3

    x = np.arange(n)
    fig, (ax_map, ax_cut) = plt.subplots(
        1, 2, figsize=(10.5, 4.6), gridspec_kw={"width_ratios": [1.05, 1.0]}
    )
    im = ax_map.imshow(
        frames[0],
        origin="lower",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
        animated=True,
    )
    ax_map.set_title("η = h + bed")
    ax_map.set_xlabel("x (cells)")
    ax_map.set_ylabel("y (cells)")
    ax_map.set_aspect("equal")
    cbar = fig.colorbar(im, ax=ax_map, fraction=0.046, pad=0.04)
    cbar.set_label("η (m)")

    (line,) = ax_cut.plot(x, frames[0, mid, :], color="#1f77b4", lw=1.5)
    ax_cut.set_xlim(0, n - 1)
    ax_cut.set_ylim(vmin, vmax)
    ax_cut.set_xlabel("x (cells)")
    ax_cut.set_ylabel("η (m)")
    ax_cut.set_title(f"centerline y = {mid}")
    ax_cut.grid(True, alpha=0.3)
    time_text = fig.suptitle(f"{title}\nt = {t[0]:.3f} s", fontsize=12)
    fig.tight_layout()

    def update(i: int):
        im.set_data(frames[i])
        line.set_ydata(frames[i, mid, :])
        time_text.set_text(f"{title}\nt = {t[i]:.3f} s  (frame {i * stride})")
        return im, line, time_text

    anim = animation.FuncAnimation(
        fig, update, frames=len(frames), interval=1000.0 / max(fps, 1e-6), blit=False
    )

    suffix = out_path.suffix.lower()
    if suffix == ".gif":
        writer = animation.PillowWriter(fps=fps)
    else:
        if suffix != ".mp4":
            out_path = out_path.with_suffix(".mp4")
        writer = animation.FFMpegWriter(
            fps=fps,
            bitrate=4000,
            metadata={"title": title, "artist": "dispersive_ref"},
        )
    anim.save(str(out_path), writer=writer)
    plt.close(fig)
    return out_path
