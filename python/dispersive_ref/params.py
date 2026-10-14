"""Simulation parameters aligned with the paper and the Bevy example defaults."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Params:
    """Jeschke & Wojtan 2023 defaults (§5) with Bevy example scale (dx≈1 m)."""

    n: int = 256
    tile_world: float = 256.0
    dt: float = 1.0 / 60.0
    g: float = 9.81
    gamma_surf: float = 0.25
    d_grad_penalty: float = 0.01
    diffusion_iters: int = 128
    airy_depths: tuple[float, ...] = (1.0, 4.0, 16.0, 64.0)
    # Floor used when bar depth is tiny in ω(k,h̄).
    h_bar_omega: float = 2.0
    # Resting basin used by init_basin (matches assets/dispersive/dispersive.wgsl).
    bed: float = -4.0
    h_rest: float = 4.0
    mound_amp: float = 4.0
    mound_radius_frac: float = 0.45
    # Soft high-k roll-off for Airy (matches Bevy). Paper Fig. 6 claims β to
    # Nyquist; without this, lattices still appear when q̃ is poorly conditioned.
    spectral_damp: bool = True
    spectral_damp_kn_cut: float = 0.5
    spectral_damp_kn_soft: float = 0.28
    # Bulk solver: "stelling" (paper Appendix A) or "cmf10" (GPU MacCormack).
    bulk_solver: str = "stelling"
    # CMF10 / Bevy DispersiveController defaults.
    vel_clamp_alpha: float = 0.5
    h_avgmax_beta: float = 2.0

    @property
    def dx(self) -> float:
        return self.tile_world / float(self.n)

    @property
    def u_max(self) -> float:
        return self.dx / (4.0 * max(self.dt, 1e-12))
