"""Merge bulk + surface into conserved height (paper §4.5 / Eq. 17)."""

from __future__ import annotations

import numpy as np

from .params import Params
from .transport import clamp_q


def merge_to_state(
    h: np.ndarray,
    h_bar: np.ndarray,
    qx_bar: np.ndarray,
    qy_bar: np.ndarray,
    h_tilde: np.ndarray,
    qx_tilde: np.ndarray,
    qy_tilde: np.ndarray,
    p: Params,
    *,
    include_div_check: bool = True,
    u_face: np.ndarray | None = None,
    v_face: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Finite-volume merge: h ← h − Δt ∇·(q + q̌) (paper §4.5 / Eq. 8)."""
    dx = p.dx
    dt = p.dt

    qe = qx_bar + qx_tilde
    qw = np.roll(qx_bar, 1, axis=1) + np.roll(qx_tilde, 1, axis=1)
    qn = qy_bar + qy_tilde
    qs = np.roll(qy_bar, 1, axis=0) + np.roll(qy_tilde, 1, axis=0)
    divq = (qe - qw) / dx + (qn - qs) / dx

    if include_div_check:
        if u_face is not None and v_face is not None:
            # Paper §4.5: q̌ on faces from half-step SL of transported h̃.
            from .transport import face_check_flux

            qxe, qxw, qyn, qys = face_check_flux(
                h_tilde, u_face, v_face, p
            )
            div_check = (qxe - qxw) / dx + (qyn - qys) / dx
        else:
            # Legacy collocated shortcut (pre-paper face resample).
            u_e = qx_bar / np.maximum(h_bar, 1e-3)
            u_w = np.roll(qx_bar, 1, axis=1) / np.maximum(
                np.roll(h_bar, 1, axis=1), 1e-3
            )
            v_n = qy_bar / np.maximum(h_bar, 1e-3)
            v_s = np.roll(qy_bar, 1, axis=0) / np.maximum(
                np.roll(h_bar, 1, axis=0), 1e-3
            )
            ht = h_tilde
            ht_l = np.roll(ht, 1, axis=1)
            ht_d = np.roll(ht, 1, axis=0)
            div_check = (ht * u_e - ht_l * u_w) / dx + (ht * v_n - ht_d * v_s) / dx
    else:
        div_check = 0.0

    h1 = np.maximum(h - dt * (divq + div_check), 0.0)
    h1 = np.where(np.isfinite(h1), h1, np.maximum(h, 0.0))
    qx1, qy1 = clamp_q(h1, qe, qn, p)
    return h1, qx1, qy1
