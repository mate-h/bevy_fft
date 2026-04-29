//! Analytic factors from Jeschke & Wojtan 2023 (Appendix B) for regression tests.

/// Numerical correction β(k) for finite-volume + spectral derivative mismatch.
/// `k` wavenumber magnitude (1/m), `dx` grid spacing (m). σ = k Δx / 2.
pub fn beta_dispersion(k: f32, dx: f32) -> f32 {
    if dx <= 0.0 {
        return 1.0;
    }
    let sigma = k * dx * 0.5;
    let s = sigma.sin();
    // β = sqrt((2k/Δx) * sin(Δx k/2)) — Eq. 27, Appendix B; DC uses β → 1.
    let v = (2.0 * k / dx) * s;
    v.max(0.0).sqrt()
}

/// Angular frequency (rad/s) for gravity–finite-depth dispersion with correction.
/// `omega_raw = sqrt(g * k * tanh(k h))`, then `omega = omega_raw / max(beta, ε)`.
pub fn omega_airy_dispersive(g: f32, k: f32, h_bar: f32, dx: f32) -> f32 {
    if k <= 0.0 {
        return 0.0;
    }
    let raw = (g * k * (k * h_bar).tanh()).max(0.0).sqrt();
    let b = beta_dispersion(k, dx).max(1e-6);
    raw / b
}

#[cfg(test)]
mod tests {
    use super::{beta_dispersion, omega_airy_dispersive};

    #[test]
    fn beta_matches_small_k_limit() {
        let dx = 1.0;
        let k = 0.1;
        let b = beta_dispersion(k, dx);
        // sin(x)~x, β ~ sqrt(2k/Δx * k Δx/2) = k
        assert!((b - k).abs() < 0.02, "beta ~ k for small k, got {b}");
    }

    #[test]
    fn omega_airy_decreases_in_shallow() {
        let g = 9.81;
        let k = 0.2;
        let dx = 1.0;
        let w_deep = omega_airy_dispersive(g, k, 50.0, dx);
        let w_shallow = omega_airy_dispersive(g, k, 1.0, dx);
        assert!(
            w_shallow < w_deep,
            "shallow h should give lower omega than deep for same k: {w_shallow} >= {w_deep}"
        );
    }
}
