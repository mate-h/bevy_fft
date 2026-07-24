"""Bulk SWE: Chentanez–Müller 2010 MacCormack path (GPU parity).

Mirrors `bar_*` / `bar_cmf10_*` in assets/dispersive/dispersive.wgsl:
sync faces → MacCormack SL on u/w → upwind height → η pressure → gather q.
"""

from __future__ import annotations

import numpy as np

from .params import Params


def _wrap(i: np.ndarray | int, n: int) -> np.ndarray | int:
    return np.mod(i, n)


def _clamp_vel(v: np.ndarray, p: Params) -> np.ndarray:
    vm = p.vel_clamp_alpha * p.dx / max(p.dt, 1e-12)
    return np.clip(v, -vm, vm)


def _sample_u(u: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Bilinear sample u faces. u shape (n, n+1) as [jc, fi]. Matches GPU clamp."""
    n = u.shape[0]
    nxf = n + 1
    jf = y - 0.5
    j0 = np.floor(jf).astype(np.int64)
    ty = jf - j0
    j0 = np.clip(j0, 0, n - 1)
    j1 = np.minimum(j0 + 1, n - 1)
    i0 = np.floor(x).astype(np.int64)
    tx = x - i0
    i0 = np.clip(i0, 0, nxf - 2)
    i1 = np.minimum(i0 + 1, nxf - 1)
    u00 = u[j0, i0]
    u10 = u[j0, i1]
    u01 = u[j1, i0]
    u11 = u[j1, i1]
    return (u00 * (1 - tx) + u10 * tx) * (1 - ty) + (u01 * (1 - tx) + u11 * tx) * ty


def _sample_w(w: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Bilinear sample w faces. w shape (n+1, n) as [fj, ic]."""
    n = w.shape[1]
    nyf = n + 1
    ifloat = x - 0.5
    i0 = np.floor(ifloat).astype(np.int64)
    tx = ifloat - i0
    i0 = np.clip(i0, 0, n - 1)
    i1 = np.minimum(i0 + 1, n - 1)
    j0 = np.floor(y).astype(np.int64)
    ty = y - j0
    j0 = np.clip(j0, 0, nyf - 2)
    j1 = np.minimum(j0 + 1, nyf - 1)
    w00 = w[j0, i0]
    w10 = w[j0, i1]
    w01 = w[j1, i0]
    w11 = w[j1, i1]
    return (w00 * (1 - tx) + w10 * tx) * (1 - ty) + (w01 * (1 - tx) + w11 * tx) * ty


def _trace(pos_x: np.ndarray, pos_y: np.ndarray, vx: np.ndarray, vy: np.ndarray, dt: float, dx: float, n: int) -> tuple[np.ndarray, np.ndarray]:
    # GPU uses clamp (not wrap) in bar_trace_semi_lag.
    px = pos_x - vx * (dt / dx)
    py = pos_y - vy * (dt / dx)
    nf = float(n)
    return np.clip(px, 0.0, nf), np.clip(py, 0.0, nf)


def sync_faces(
    h_bar: np.ndarray, qx_bar: np.ndarray, qy_bar: np.ndarray, p: Params
) -> tuple[np.ndarray, np.ndarray]:
    """Build (n, n+1) u-faces and (n+1, n) w-faces from cell q/h."""
    n = p.n
    h = np.maximum(h_bar, 0.0)
    ul = qx_bar / np.maximum(h, 1e-6)
    vl = qy_bar / np.maximum(h, 1e-6)

    u = np.zeros((n, n + 1), dtype=np.float64)
    # Face fi between cell fi-1 and fi (periodic); fi=0 and fi=n are the same seam.
    for fi in range(n + 1):
        i_right = 0 if fi == n else fi
        i_left = (i_right - 1) % n
        u[:, fi] = 0.5 * (ul[:, i_left] + ul[:, i_right])
    u = _clamp_vel(u, p)

    w = np.zeros((n + 1, n), dtype=np.float64)
    for fj in range(n + 1):
        j_up = 0 if fj == n else fj
        j_dn = (j_up - 1) % n
        w[fj, :] = 0.5 * (vl[j_dn, :] + vl[j_up, :])
    w = _clamp_vel(w, p)
    return u, w


def _maccormack_u(u: np.ndarray, w: np.ndarray, p: Params) -> np.ndarray:
    n = p.n
    jc = np.arange(n)[:, None]
    fi = np.arange(n + 1)[None, :]
    pos_x = fi.astype(np.float64)
    pos_y = jc.astype(np.float64) + 0.5
    u_n = u.copy()

    # Forward SL: sample mac0 velocity, backtrace, sample u^n.
    vx = _sample_u(u_n, pos_x, pos_y)
    vy = _sample_w(w, pos_x, pos_y)
    bx, by = _trace(pos_x, pos_y, vx, vy, p.dt, p.dx, n)
    u_hat = _clamp_vel(_sample_u(u_n, bx, by), p)

    # Reverse SL from u_hat field (GPU reverse samples live face u = u_hat).
    u_live = u_hat
    vx = _sample_u(u_n, pos_x, pos_y)  # advect vel still from mac0 + live w
    vy = _sample_w(w, pos_x, pos_y)
    bx, by = _trace(pos_x, pos_y, vx, vy, -p.dt, p.dx, n)
    u_tilde = _sample_u(u_live, bx, by)

    u_new = u_hat + 0.5 * (u_n - u_tilde)
    min_u = np.minimum(u_n, u_hat)
    max_u = np.maximum(u_n, u_hat)
    bad = (u_new < min_u - 1e-5) | (u_new > max_u + 1e-5)
    u_new = np.where(bad, u_hat, u_new)
    return _clamp_vel(u_new, p)


def _maccormack_w(u: np.ndarray, w: np.ndarray, p: Params) -> np.ndarray:
    n = p.n
    fj = np.arange(n + 1)[:, None]
    ic = np.arange(n)[None, :]
    pos_x = ic.astype(np.float64) + 0.5
    pos_y = fj.astype(np.float64)
    w_n = w.copy()

    vx = _sample_u(u, pos_x, pos_y)
    vy = _sample_w(w_n, pos_x, pos_y)
    bx, by = _trace(pos_x, pos_y, vx, vy, p.dt, p.dx, n)
    w_hat = _clamp_vel(_sample_w(w_n, bx, by), p)

    w_live = w_hat
    vx = _sample_u(u, pos_x, pos_y)
    vy = _sample_w(w_n, pos_x, pos_y)
    bx, by = _trace(pos_x, pos_y, vx, vy, -p.dt, p.dx, n)
    w_tilde = _sample_w(w_live, bx, by)

    w_new = w_hat + 0.5 * (w_n - w_tilde)
    min_w = np.minimum(w_n, w_hat)
    max_w = np.maximum(w_n, w_hat)
    bad = (w_new < min_w - 1e-5) | (w_new > max_w + 1e-5)
    w_new = np.where(bad, w_hat, w_new)
    return _clamp_vel(w_new, p)


def _h_adj(h: np.ndarray, p: Params) -> np.ndarray:
    beta = p.h_avgmax_beta
    h_avgmax = beta * p.dx / (p.g * max(p.dt, 1e-12))
    avg = 0.25 * (
        np.roll(h, -1, 1) + np.roll(h, 1, 1) + np.roll(h, -1, 0) + np.roll(h, 1, 0)
    )
    return np.maximum(0.0, avg - h_avgmax)


def _upwind_flux(h_up: np.ndarray, h_dn: np.ndarray, u: np.ndarray, adj_up: np.ndarray, adj_dn: np.ndarray) -> np.ndarray:
    hu = np.where(u > 0.0, h_up - adj_up, h_dn - adj_dn)
    return u * np.maximum(0.0, hu)


def integrate_height(h_bar: np.ndarray, u: np.ndarray, w: np.ndarray, p: Params) -> np.ndarray:
    n = p.n
    h = np.maximum(h_bar, 0.0)
    adj = _h_adj(h, p)
    # u[:, i] = left face of cell i, u[:, i+1] = right face.
    uL = u[:, :n]
    uR = u[:, 1 : n + 1]
    wB = w[:n, :]
    wT = w[1 : n + 1, :]

    h_l = np.roll(h, 1, 1)
    h_r = np.roll(h, -1, 1)
    h_d = np.roll(h, 1, 0)
    h_u = np.roll(h, -1, 0)
    adj_l = np.roll(adj, 1, 1)
    adj_r = np.roll(adj, -1, 1)
    adj_d = np.roll(adj, 1, 0)
    adj_t = np.roll(adj, -1, 0)

    f_e = _upwind_flux(h, h_r, uR, adj, adj_r)
    f_w = _upwind_flux(h_l, h, uL, adj_l, adj)
    f_n = _upwind_flux(h, h_u, wT, adj, adj_t)
    f_s = _upwind_flux(h_d, h, wB, adj_d, adj)
    div = f_e - f_w + f_n - f_s
    return np.maximum(h - (p.dt / p.dx) * div, 0.0)


def pressure_velocity(
    h_bar: np.ndarray, bed: np.ndarray, u: np.ndarray, w: np.ndarray, p: Params
) -> tuple[np.ndarray, np.ndarray]:
    n = p.n
    eta = bed + np.maximum(h_bar, 0.0)
    # u faces fi=1..n get pressure (fi=0 skipped), matching GPU.
    u_new = u.copy()
    for fi in range(1, n + 1):
        i_r = 0 if fi == n else fi
        i_l = (i_r - 1) % n
        u_new[:, fi] = u[:, fi] + (-p.g / p.dx * (eta[:, i_r] - eta[:, i_l])) * p.dt
    u_new = _clamp_vel(u_new, p)

    w_new = w.copy()
    for fj in range(1, n + 1):
        j_u = 0 if fj == n else fj
        j_d = (j_u - 1) % n
        w_new[fj, :] = w[fj, :] + (-p.g / p.dx * (eta[j_u, :] - eta[j_d, :])) * p.dt
    w_new = _clamp_vel(w_new, p)
    return u_new, w_new


def gather_q(h_bar: np.ndarray, u: np.ndarray, w: np.ndarray, p: Params) -> tuple[np.ndarray, np.ndarray]:
    n = p.n
    h = np.maximum(h_bar, 0.0)
    uL = u[:, :n]
    uR = u[:, 1 : n + 1]
    wB = w[:n, :]
    wT = w[1 : n + 1, :]
    u_avg = 0.5 * (uL + uR)
    v_avg = 0.5 * (wB + wT)
    qx = h * u_avg
    qy = h * v_avg
    h_safe = np.maximum(h, 1e-3)
    u_m = p.dx / (4.0 * max(p.dt, 1e-12))
    speed = np.hypot(qx / h_safe, qy / h_safe)
    sc = np.where(speed > u_m, u_m / np.maximum(speed, 1e-12), 1.0)
    return qx * sc, qy * sc


def cell_uv_from_faces(u: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cell-centered (+x/+y face) velocities for transport."""
    n = u.shape[0]
    # Match previous transport convention: cell stores +x / +y face vel.
    return u[:, 1 : n + 1].copy(), w[1 : n + 1, :].copy()


def step_cmf10(
    h_bar: np.ndarray,
    qx_bar: np.ndarray,
    qy_bar: np.ndarray,
    bed: np.ndarray,
    p: Params,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One CMF10 bar step. Returns h_bar, qx_bar, qy_bar, u_cell, v_cell."""
    u, w = sync_faces(h_bar, qx_bar, qy_bar, p)
    u = _maccormack_u(u, w, p)
    w = _maccormack_w(u, w, p)
    h_bar = integrate_height(h_bar, u, w, p)
    u, w = pressure_velocity(h_bar, bed, u, w, p)
    qx_bar, qy_bar = gather_q(h_bar, u, w, p)
    u_cell, v_cell = cell_uv_from_faces(u, w)
    return h_bar, qx_bar, qy_bar, u_cell, v_cell
