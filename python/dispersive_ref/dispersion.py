"""Paper Fig. 6 / §4.3.1: β-corrected Airy dispersion (Eq. 2 + Eq. 27)."""

from __future__ import annotations

import numpy as np

from .airy import beta_num, omega_disp
from .params import Params


def theoretical_omega(g: float, k: float, h: float) -> float:
    return float(np.sqrt(max(g * k * np.tanh(k * h), 0.0)))


def check_wavelength(wavelength_cells: float, h: float, p: Params) -> dict:
    lam = wavelength_cells * p.dx
    k = 2.0 * np.pi / lam
    w_phys = theoretical_omega(p.g, k, h)
    w_corr = float(omega_disp(p.g, np.array([k]), h, p.dx)[0])
    b = float(beta_num(np.array([k]), p.dx)[0])
    recovered = w_corr * (b / max(k, 1e-12))
    return {
        "wavelength_m": lam,
        "wavelength_cells": wavelength_cells,
        "k": k,
        "h": h,
        "omega_phys": w_phys,
        "omega_corrected": w_corr,
        "beta": b,
        "c_phys": w_phys / k,
        "c_corrected": w_corr / k,
        "rel_err_recovered": float(abs(recovered - w_phys) / max(w_phys, 1e-12)),
    }


def dispersion_suite(p: Params | None = None) -> dict:
    """Sample wavelengths toward Nyquist (paper Fig. 6 pool depth 4 m)."""
    p = p or Params()
    rows = []
    for cells in (32.0, 16.0, 8.0, 4.0, 3.0, 2.5, 2.0):
        if cells > p.n:
            continue
        rows.append(check_wavelength(cells, h=4.0, p=p))
    max_err = max(r["rel_err_recovered"] for r in rows) if rows else 1.0
    return {
        "rows": rows,
        "max_rel_err_recovered": max_err,
        "passed": max_err < 1e-9,
        "note": "With ω_corr = ω_phys · k/β, recovered ω_phys must match Eq. 2.",
    }
