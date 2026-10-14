"""Wave decomposition via depth-dependent diffusion (paper §4.1, Eq. 11–13)."""

from __future__ import annotations

import numpy as np

from .params import Params


def _roll(a: np.ndarray, dy: int, dx: int) -> np.ndarray:
    return np.roll(np.roll(a, dy, axis=0), dx, axis=1)


def _diffuse_field(
    field: np.ndarray,
    alpha: np.ndarray,
    dx: float,
    iters: int,
) -> np.ndarray:
    """FTCS-diffuse a scalar field with the same per-cell dT as η."""
    f = field.copy()
    dx2 = dx * dx
    for _ in range(iters):
        dT_max = 0.24 * dx2 / np.maximum(4.0 * alpha, 1e-8)
        dT = np.minimum(0.25, dT_max)
        lap = (
            _roll(f, 0, 1)
            + _roll(f, 0, -1)
            + _roll(f, 1, 0)
            + _roll(f, -1, 0)
            - 4.0 * f
        ) / dx2
        f = f + alpha * lap * dT
        f = np.where(np.isfinite(f), f, field)
    return f


def decompose(
    h: np.ndarray,
    qx: np.ndarray,
    qy: np.ndarray,
    bed: np.ndarray,
    p: Params,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return h_bar, qx_bar, qy_bar, h_tilde, qx_tilde, qy_tilde.

    Diffuse η for the height split. All momentum stays in the bar; surface q̃ is
    produced by the Airy/eWave step from h̃ (carrying a raw q residual pumps
    energy on fine grids).
    """
    eta = bed + np.maximum(h, 0.0)
    h_safe = np.maximum(h, 1e-3)
    gx = (_roll(h_safe, 0, -1) - _roll(h_safe, 0, 1)) / (2.0 * p.dx)
    gy = (_roll(h_safe, -1, 0) - _roll(h_safe, 1, 0)) / (2.0 * p.dx)
    grad2 = gx * gx + gy * gy
    alpha = (h_safe * h_safe / 64.0) * np.exp(-p.d_grad_penalty * grad2)

    H = _diffuse_field(eta, alpha, p.dx, p.diffusion_iters)
    h_bar = np.maximum(H - bed, 0.0)
    h_w = np.maximum(h, 0.0)
    h_tilde = h_w - h_bar
    qx_bar = qx.copy()
    qy_bar = qy.copy()
    qx_tilde = np.zeros_like(qx)
    qy_tilde = np.zeros_like(qy)
    return h_bar, qx_bar, qy_bar, h_tilde, qx_tilde, qy_tilde
