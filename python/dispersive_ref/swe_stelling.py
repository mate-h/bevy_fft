"""Bulk SWE: staggered momentum-conserving update (paper Appendix A / Stelling).

This is a compact NumPy port of Eqs. 18–20 for a periodic wet domain. It is the
paper bulk solver, not the GPU CMF10 MacCormack path.
"""

from __future__ import annotations

import numpy as np

from .params import Params


def _clamp_u(u: np.ndarray, u_max: float) -> np.ndarray:
    return np.clip(u, -u_max, u_max)


def cell_q_from_faces(
    h_bar: np.ndarray, u: np.ndarray, v: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Gather face velocities to cell-centered q ≈ h * u_face(+x/+y)."""
    # u is (n, n): store +x face of each cell (periodic: face i+1/2 at index i).
    # Convention: u[i,j] = velocity on right face of cell (j,i) wait — arrays are [y,x].
    # We use u[y, x] = velocity on the +x face of cell (y,x).
    # v[y, x] = velocity on the +y face of cell (y,x).
    h = np.maximum(h_bar, 1e-3)
    qx = h * u
    qy = h * v
    return qx, qy


def faces_from_cell_q(
    h_bar: np.ndarray, qx_bar: np.ndarray, qy_bar: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    h = np.maximum(h_bar, 1e-3)
    return qx_bar / h, qy_bar / h


def step_stelling(
    h_bar: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    bed: np.ndarray,
    p: Params,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One explicit Stelling-style update. Returns new h_bar, u, v."""
    dx = p.dx
    dt = p.dt
    g = p.g
    u_max = p.u_max
    h = np.maximum(h_bar, 0.0)

    # Upwind face heights for continuity (Eq. 18).
    # +x face of cell (y,x): if u>0 use h[y,x], else h[y,x+1].
    u_e = u
    u_w = np.roll(u, 1, axis=1)
    v_n = v
    v_s = np.roll(v, 1, axis=0)

    h_e = np.where(u_e >= 0.0, h, np.roll(h, -1, axis=1))
    h_w = np.where(u_w >= 0.0, np.roll(h, 1, axis=1), h)
    h_n = np.where(v_n >= 0.0, h, np.roll(h, -1, axis=0))
    h_s = np.where(v_s >= 0.0, np.roll(h, 1, axis=0), h)

    dh = -(
        (h_e * u_e - h_w * u_w) / dx + (h_n * v_n - h_s * v_s) / dx
    )
    h_new = np.maximum(h + dt * dh, 0.0)

    # Momentum (Eqs. 19–20), positive-flow branch with abs-based upwinding.
    eta = bed + h
    # Pressure gradient on +x faces between cells x and x+1.
    d_eta_x = (np.roll(eta, -1, axis=1) - eta) / dx
    d_eta_y = (np.roll(eta, -1, axis=0) - eta) / dx

    # Advective u·∇u with donor-cell upwinding (Appendix A positive-flow case).
    u_c = u
    v_c = v
    u_l = np.roll(u, 1, axis=1)
    u_r = np.roll(u, -1, axis=1)
    u_d = np.roll(u, 1, axis=0)
    u_u = np.roll(u, -1, axis=0)
    v_face_for_u = 0.5 * (v + np.roll(v, -1, axis=1))
    adv_u_x = np.where(u_c >= 0.0, u_c * (u_c - u_l) / dx, u_r * (u_r - u_c) / dx)
    adv_u_y = np.where(
        v_face_for_u >= 0.0,
        v_face_for_u * (u_c - u_d) / dx,
        v_face_for_u * (u_u - u_c) / dx,
    )

    v_l = np.roll(v, 1, axis=1)
    v_r = np.roll(v, -1, axis=1)
    v_d = np.roll(v, 1, axis=0)
    v_u = np.roll(v, -1, axis=0)
    u_face_for_v = 0.5 * (u + np.roll(u, -1, axis=0))
    adv_v_x = np.where(
        u_face_for_v >= 0.0,
        u_face_for_v * (v_c - v_l) / dx,
        u_face_for_v * (v_r - v_c) / dx,
    )
    adv_v_y = np.where(v_c >= 0.0, v_c * (v_c - v_d) / dx, v_u * (v_u - v_c) / dx)

    u_new = u_c - dt * (adv_u_x + adv_u_y + g * d_eta_x)
    v_new = v_c - dt * (adv_v_x + adv_v_y + g * d_eta_y)

    # Light Laplacian viscosity keeps long mound runs from cascading into Airy.
    # Paper Stelling has none; this is a practical stabilizer for the NumPy port.
    nu = 0.15 * dx * dx / max(dt, 1e-12)
    lap_u = (
        np.roll(u_c, 1, 1) + np.roll(u_c, -1, 1) + np.roll(u_c, 1, 0) + np.roll(u_c, -1, 0) - 4.0 * u_c
    )
    lap_v = (
        np.roll(v_c, 1, 1) + np.roll(v_c, -1, 1) + np.roll(v_c, 1, 0) + np.roll(v_c, -1, 0) - 4.0 * v_c
    )
    u_new = u_new + dt * nu * lap_u / (dx * dx)
    v_new = v_new + dt * nu * lap_v / (dx * dx)

    u_new = _clamp_u(u_new, u_max)
    v_new = _clamp_u(v_new, u_max)

    wet_x = (h > 1e-6) | (np.roll(h, -1, axis=1) > 1e-6)
    wet_y = (h > 1e-6) | (np.roll(h, -1, axis=0) > 1e-6)
    u_new = np.where(wet_x, u_new, 0.0)
    v_new = np.where(wet_y, v_new, 0.0)
    return h_new, u_new, v_new
