"""Transport surface waves through bulk velocity (paper §4.4, Alg. 3–4)."""

from __future__ import annotations

import numpy as np

from .params import Params


def _catmull_rom(p0, p1, p2, p3, t):
    t2 = t * t
    t3 = t2 * t
    return 0.5 * (
        (2.0 * p1)
        + (-p0 + p2) * t
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
    )


def cubic_sl_sample(field: np.ndarray, y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Periodic cubic SL sample with clamp to the four nearest cells (§4.4)."""
    n = field.shape[0]
    x = np.mod(x, n)
    y = np.mod(y, n)
    i0 = np.floor(x).astype(np.int64)
    j0 = np.floor(y).astype(np.int64)
    tx = x - i0
    ty = y - j0

    def at(jj, ii):
        return field[np.mod(jj, n), np.mod(ii, n)]

    rows = []
    for dj in (-1, 0, 1, 2):
        a = at(j0 + dj, i0 - 1)
        b = at(j0 + dj, i0)
        c = at(j0 + dj, i0 + 1)
        d = at(j0 + dj, i0 + 2)
        rows.append(_catmull_rom(a, b, c, d, tx))
    v = _catmull_rom(rows[0], rows[1], rows[2], rows[3], ty)
    v00 = at(j0, i0)
    v10 = at(j0, i0 + 1)
    v01 = at(j0 + 1, i0)
    v11 = at(j0 + 1, i0 + 1)
    vmin = np.minimum(np.minimum(v00, v10), np.minimum(v01, v11))
    vmax = np.maximum(np.maximum(v00, v10), np.maximum(v01, v11))
    return np.clip(v, vmin, vmax)


def clamp_q(h: np.ndarray, qx: np.ndarray, qy: np.ndarray, p: Params) -> tuple[np.ndarray, np.ndarray]:
    qm = np.maximum(h, 0.0) * p.dx / (4.0 * max(p.dt, 1e-12))
    nq = np.hypot(qx, qy)
    scale = np.ones_like(nq)
    mask = (nq > qm) & (nq > 1e-12)
    scale[mask] = qm[mask] / nq[mask]
    return qx * scale, qy * scale


def _cell_velocity(u_face: np.ndarray, v_face: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ux_c = 0.5 * (u_face + np.roll(u_face, 1, axis=1))
    uy_c = 0.5 * (v_face + np.roll(v_face, 1, axis=0))
    return ux_c, uy_c


def _growth_factor(u_face: np.ndarray, v_face: np.ndarray, p: Params) -> np.ndarray:
    dx = p.dx
    div = (u_face - np.roll(u_face, 1, axis=1)) / dx + (
        v_face - np.roll(v_face, 1, axis=0)
    ) / dx
    g = np.minimum(-div, -p.gamma_surf * div)
    g = np.clip(g, -8.0, 8.0)
    return np.exp(g * p.dt)


def transport(
    h_tilde: np.ndarray,
    qx_tilde: np.ndarray,
    qy_tilde: np.ndarray,
    h_bar: np.ndarray,
    u_end: np.ndarray,
    v_end: np.ndarray,
    p: Params,
    *,
    u_mid: np.ndarray | None = None,
    v_mid: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Alg. 3 for q̃ (midpoint ū) then Alg. 4 for h̃ (end ū)."""
    dx = p.dx
    u_q = u_end if u_mid is None else u_mid
    v_q = v_end if v_mid is None else v_mid

    fac_q = _growth_factor(u_q, v_q, p)
    fac_h = _growth_factor(u_end, v_end, p)

    ht = h_tilde * fac_h
    qx = qx_tilde * fac_q
    qy = qy_tilde * fac_q
    qx, qy = clamp_q(np.maximum(ht + h_bar, 0.05), qx, qy, p)

    n = p.n
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float64)
    ux_q, uy_q = _cell_velocity(u_q, v_q)
    ux_h, uy_h = _cell_velocity(u_end, v_end)

    # q̃: midpoint velocity (Alg. 3).
    px_q = xx - ux_q * (p.dt / max(dx, 1e-12))
    py_q = yy - uy_q * (p.dt / max(dx, 1e-12))
    qx = cubic_sl_sample(qx, py_q, px_q)
    qy = cubic_sl_sample(qy, py_q, px_q)

    # h̃: end-of-step velocity (Alg. 4).
    px_h = xx - ux_h * (p.dt / max(dx, 1e-12))
    py_h = yy - uy_h * (p.dt / max(dx, 1e-12))
    ht = cubic_sl_sample(ht, py_h, px_h)

    qx, qy = clamp_q(np.maximum(h_bar + ht, 1e-3), qx, qy, p)
    return ht, qx, qy


def face_check_flux(
    h_tilde: np.ndarray,
    u_face: np.ndarray,
    v_face: np.ndarray,
    p: Params,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Paper §4.5: half-step SL of transported h̃ onto faces, then q̌ = h̃ ū.

    Returns east/west/north/south face fluxes for the cell-centered divergence
    stencil (qe - qw)/dx + (qn - qs)/dx.
    """
    n = p.n
    dx = p.dx
    dt_half = 0.5 * p.dt
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float64)

    # +x face at (i+0.5, j); cell q_x storage matches this face.
    u_e = u_face
    v_e = 0.5 * (v_face + np.roll(v_face, -1, axis=1))
    px_e = (xx + 0.5) - u_e * (dt_half / max(dx, 1e-12))
    py_e = yy - v_e * (dt_half / max(dx, 1e-12))
    ht_e = cubic_sl_sample(h_tilde, py_e, px_e)
    qe = ht_e * u_e

    # −x face = +x face of left neighbor.
    qw = np.roll(qe, 1, axis=1)

    # +y face at (i, j+0.5).
    v_n = v_face
    u_n = 0.5 * (u_face + np.roll(u_face, -1, axis=0))
    px_n = xx - u_n * (dt_half / max(dx, 1e-12))
    py_n = (yy + 0.5) - v_n * (dt_half / max(dx, 1e-12))
    ht_n = cubic_sl_sample(h_tilde, py_n, px_n)
    qn = ht_n * v_n

    qs = np.roll(qn, 1, axis=0)
    return qe, qw, qn, qs
