"""Flat basin plus optional raised-cosine center mound for scenario ICs."""

from __future__ import annotations

import numpy as np

from .params import Params


def init_basin(p: Params) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = p.n
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float64)
    c = 0.5 * n
    d = np.hypot(xx - c, yy - c)
    r = p.mound_radius_frac * n
    h = np.full((n, n), p.h_rest, dtype=np.float64)
    inside = d < r
    t = np.where(inside, d / r, 1.0)
    w = 0.5 * (1.0 + np.cos(np.pi * t))
    h = np.where(inside, h + p.mound_amp * w * w, h)
    bed = np.full((n, n), p.bed, dtype=np.float64)
    qx = np.zeros((n, n), dtype=np.float64)
    qy = np.zeros((n, n), dtype=np.float64)
    return h, qx, qy, bed
