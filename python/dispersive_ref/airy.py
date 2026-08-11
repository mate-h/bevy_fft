"""Airy surface waves with multi-depth blend and β correction (paper §4.3)."""

from __future__ import annotations

import numpy as np

from .params import Params
from .transport import clamp_q


def wavenumbers(n: int, dx: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    kx_1d = 2.0 * np.pi * np.fft.fftfreq(n, d=dx)
    ky_1d = 2.0 * np.pi * np.fft.fftfreq(n, d=dx)
    kx, ky = np.meshgrid(kx_1d, ky_1d)
    klen = np.hypot(kx, ky)
    return kx, ky, klen


def spectral_damp_mask(klen: np.ndarray, p: Params) -> np.ndarray:
    if not p.spectral_damp:
        return np.ones_like(klen)
    k_nyq = np.pi / max(p.dx, 1e-12)
    kn = klen / max(k_nyq, 1e-12)
    out = np.ones_like(klen)
    out = np.where(kn > p.spectral_damp_kn_cut, 0.0, out)
    t = np.clip(
        (kn - p.spectral_damp_kn_soft)
        / max(p.spectral_damp_kn_cut - p.spectral_damp_kn_soft, 1e-12),
        0.0,
        1.0,
    )
    soft = np.exp(-8.0 * t * t)
    return np.where(kn > p.spectral_damp_kn_soft, soft * out, out)


def beta_num(k: np.ndarray, dx: float) -> np.ndarray:
    s = np.sin(k * dx * 0.5)
    return np.sqrt(np.maximum((2.0 * k / max(dx, 1e-12)) * s, 0.0))


def omega_disp(g: float, k: np.ndarray, hbar: float, dx: float) -> np.ndarray:
    """Corrected ω = ω_phys * k / β (paper Eq. 2 + Eq. 27, dimensionless β/k)."""
    raw = np.sqrt(np.maximum(g * k * np.tanh(k * hbar), 0.0))
    b = np.maximum(beta_num(k, dx), 1e-6)
    out = raw * (k / b)
    out = np.where(k < 1e-10, 0.0, out)
    return out


def _zero_nyquist_deriv(kx: np.ndarray, ky: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    kxd = kx.copy()
    kyd = ky.copy()
    if n >= 2 and n % 2 == 0:
        half = n // 2
        kxd[:, half] = 0.0
        kyd[half, :] = 0.0
    return kxd, kyd


def evolve_q_fixed_depth(
    h_tilde: np.ndarray,
    qx_tilde: np.ndarray,
    qy_tilde: np.ndarray,
    hbar: float,
    p: Params,
) -> tuple[np.ndarray, np.ndarray]:
    """Algorithm 2 at one constant depth. Returns updated qx, qy."""
    n = p.n
    kx, ky, klen = wavenumbers(n, p.dx)
    kxd, kyd = _zero_nyquist_deriv(kx, ky, n)
    damp = spectral_damp_mask(klen, p)

    h_hat = np.fft.fft2(h_tilde) * damp
    qx_hat = np.fft.fft2(qx_tilde)
    qy_hat = np.fft.fft2(qy_tilde)

    w = omega_disp(p.g, klen, hbar, p.dx)
    c = np.cos(w * p.dt)
    s = np.sin(w * p.dt)
    k2 = np.maximum(klen * klen, 1e-12)

    # Half-cell shift on ĥ for staggered q (paper §4.3).
    h_shift_x = h_hat * np.exp(-1j * kxd * p.dx * 0.5)
    h_shift_y = h_hat * np.exp(-1j * kyd * p.dx * 0.5)

    dhd_x = 1j * kxd * h_shift_x
    dhd_y = 1j * kyd * h_shift_y
    sc = s * (w / k2)
    qx_new_hat = (c * qx_hat - sc * dhd_x) * damp
    qy_new_hat = (c * qy_hat - sc * dhd_y) * damp
    qx_new_hat = np.where(klen < 1e-8, qx_hat * damp, qx_new_hat)
    qy_new_hat = np.where(klen < 1e-8, qy_hat * damp, qy_new_hat)

    qx_out = np.fft.ifft2(qx_new_hat).real
    qy_out = np.fft.ifft2(qy_new_hat).real
    qx_out = np.where(np.isfinite(qx_out), qx_out, 0.0)
    qy_out = np.where(np.isfinite(qy_out), qy_out, 0.0)
    return qx_out, qy_out


def blend_depths(
    stacks_qx: list[np.ndarray],
    stacks_qy: list[np.ndarray],
    h_bar: np.ndarray,
    depths: tuple[float, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Piecewise-linear blend of fixed-depth Airy results by local h̄."""
    depths_a = np.asarray(depths, dtype=np.float64)
    order = np.argsort(depths_a)
    depths_a = depths_a[order]
    qx_s = [stacks_qx[i] for i in order]
    qy_s = [stacks_qy[i] for i in order]

    h = np.clip(h_bar, depths_a[0], depths_a[-1])
    qx = np.zeros_like(h_bar)
    qy = np.zeros_like(h_bar)
    for i in range(len(depths_a) - 1):
        d0, d1 = depths_a[i], depths_a[i + 1]
        mask = (h >= d0) & (h <= d1) if i == 0 else (h > d0) & (h <= d1)
        if i == len(depths_a) - 2:
            mask = h >= d0
        t = (h - d0) / max(d1 - d0, 1e-12)
        t = np.clip(t, 0.0, 1.0)
        qx = np.where(mask, (1.0 - t) * qx_s[i] + t * qx_s[i + 1], qx)
        qy = np.where(mask, (1.0 - t) * qy_s[i] + t * qy_s[i + 1], qy)
    qx = np.where(h_bar <= depths_a[0], qx_s[0], qx)
    qy = np.where(h_bar <= depths_a[0], qy_s[0], qy)
    qx = np.where(h_bar >= depths_a[-1], qx_s[-1], qx)
    qy = np.where(h_bar >= depths_a[-1], qy_s[-1], qy)
    return qx, qy


def step_airy(
    h_tilde: np.ndarray,
    qx_tilde: np.ndarray,
    qy_tilde: np.ndarray,
    h_bar: np.ndarray,
    p: Params,
    *,
    h_tilde_prev: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Multi-depth Airy update for q̃ (paper Alg. 2). Returns (h_tilde, qx, qy).

    h̃ is only band-limited here. Transport (Alg. 4) moves it in space. The
    optional previous h̃ is averaged with the current field to un-stagger time
    before the FFT, matching Alg. 2's (h̃^{t−Δt/2} + h̃^{t+Δt/2})/2.
    """
    if h_tilde_prev is None:
        h_avg = h_tilde
    else:
        h_avg = 0.5 * (h_tilde + h_tilde_prev)

    depths = tuple(sorted(p.airy_depths))
    stack_qx = []
    stack_qy = []
    for d in depths:
        qx_i, qy_i = evolve_q_fixed_depth(h_avg, qx_tilde, qy_tilde, d, p)
        stack_qx.append(qx_i)
        stack_qy.append(qy_i)
    qx_b, qy_b = blend_depths(
        stack_qx, stack_qy, np.maximum(h_bar, p.h_bar_omega), depths
    )
    qx_b, qy_b = clamp_q(np.maximum(h_bar + h_tilde, 1e-3), qx_b, qy_b, p)

    # Match Bevy: band-limit h̃ with the same spectral damp used on Airy.
    _, _, klen = wavenumbers(p.n, p.dx)
    damp = spectral_damp_mask(klen, p)
    h_out = np.fft.ifft2(np.fft.fft2(h_tilde) * damp).real
    h_out = np.where(np.isfinite(h_out), h_out, h_tilde)
    return h_out, qx_b, qy_b


def damp_state_fields(
    h: np.ndarray,
    qx: np.ndarray,
    qy: np.ndarray,
    bed: np.ndarray,
    p: Params,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Band-limit the merged free surface and momentum.

    Airy already damps h̃ and q̃, but merge writes into total h from fluxes that
    never remove a stuck high-k spike at the IC center. Applying the same soft
    kn roll-off to η and q after merge clears that static leftover. Physical
    dispersive bands for the ripple IC sit well below the soft cutoff.
    """
    if not p.spectral_damp:
        return h, qx, qy
    _, _, klen = wavenumbers(p.n, p.dx)
    damp = spectral_damp_mask(klen, p)
    eta = bed + np.maximum(h, 0.0)
    eta_d = np.fft.ifft2(np.fft.fft2(eta) * damp).real
    h_d = np.maximum(eta_d - bed, 0.0)
    qx_d = np.fft.ifft2(np.fft.fft2(qx) * damp).real
    qy_d = np.fft.ifft2(np.fft.fft2(qy) * damp).real
    h_d = np.where(np.isfinite(h_d), h_d, np.maximum(h, 0.0))
    qx_d = np.where(np.isfinite(qx_d), qx_d, 0.0)
    qy_d = np.where(np.isfinite(qy_d), qy_d, 0.0)
    qx_d, qy_d = clamp_q(h_d, qx_d, qy_d, p)
    return h_d, qx_d, qy_d
