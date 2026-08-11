"""Save NumPy reference checkpoints (.f32le RGBA)."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def save_f32le(path: Path, data: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(np.asarray(data, dtype="<f4").tobytes())


def pack_rgba(h: np.ndarray, qx: np.ndarray, qy: np.ndarray) -> np.ndarray:
    n = h.shape[0]
    out = np.zeros((n, n, 4), dtype=np.float32)
    out[..., 0] = h
    out[..., 1] = qx
    out[..., 2] = qy
    return out


def write_sim_dump(dir_path: Path, sim) -> None:
    """Write state, bar, tilde, and bed as little-endian f32 textures."""
    d = Path(dir_path)
    d.mkdir(parents=True, exist_ok=True)
    save_f32le(d / "state.f32le", pack_rgba(sim.h, sim.qx, sim.qy))
    save_f32le(d / "bar.f32le", pack_rgba(sim.h_bar, sim.qx_bar, sim.qy_bar))
    save_f32le(
        d / "tilde.f32le", pack_rgba(sim.h_tilde, sim.qx_tilde, sim.qy_tilde)
    )
    bed = np.zeros((sim.n, sim.n, 4), dtype=np.float32)
    bed[..., 0] = sim.bed
    save_f32le(d / "bed.f32le", bed)
    (d / "meta.txt").write_text(
        f"n={sim.n}\ndx={sim.p.dx}\ndt={sim.p.dt}\nsource=dispersive_ref\n"
    )
