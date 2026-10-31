// Finite-depth gravity wave dispersion shared by `init_h0` and the per-frame spectrum pass.
//
// ω² = g k tanh(k h). For k h above about 3, tanh is 1 to within f32 precision and this reduces to
// the deep-water relation ω = √(g k).

/// Depth beyond which `tanh(k h)` is treated as 1. Avoids `exp` overflow inside `tanh`.
const DEEP_KH: f32 = 10.0;

fn tanh_kh(k: f32, depth: f32) -> f32 {
    let kh = k * max(depth, 1e-3);
    if kh > DEEP_KH {
        return 1.0;
    }
    return tanh(kh);
}

/// Angular frequency ω(k) in rad/s for wavenumber `k` (rad/m) over water of `depth` meters.
fn omega_finite_depth(k: f32, depth: f32, g: f32) -> f32 {
    return sqrt(max(g * k * tanh_kh(k, depth), 0.0));
}

/// Group-velocity Jacobian dω/dk for the finite-depth relation.
///
/// dω/dk = g (tanh(kh) + kh sech²(kh)) / (2ω). Uses sech² = 1 - tanh².
fn domega_dk_finite_depth(k: f32, depth: f32, g: f32) -> f32 {
    let w = omega_finite_depth(k, depth, g);
    if w < 1e-9 {
        // Long-wave limit: ω ≈ k √(g h), so dω/dk ≈ √(g h).
        return sqrt(g * max(depth, 1e-3));
    }
    let th = tanh_kh(k, depth);
    let kh = k * max(depth, 1e-3);
    let sech2 = select(1.0 - th * th, 0.0, kh > DEEP_KH);
    return g * (th + kh * sech2) / (2.0 * w);
}

/// TMA depth attenuation of a deep-water spectrum (Kitaigorodskii factor, Thompson and Vincent fit).
///
/// Takes ω in rad/s. Returns 1 in deep water and falls toward 0 as waves feel the bottom.
fn tma_factor(omega: f32, depth: f32, g: f32) -> f32 {
    let wh = omega * sqrt(max(depth, 1e-3) / g);
    if wh <= 1.0 {
        return 0.5 * wh * wh;
    }
    if wh < 2.0 {
        let a = 2.0 - wh;
        return 1.0 - 0.5 * a * a;
    }
    return 1.0;
}
