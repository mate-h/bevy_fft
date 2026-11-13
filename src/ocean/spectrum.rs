//! CPU copy of the ocean spectrum in `init_h0.wesl`, used for quantities that integrate over it.
//!
//! The surface Stokes drift is the slow mean velocity of water particles under the waves. A linear
//! wave of amplitude `a` moves a particle on a closed orbit, which the displacement lanes already
//! hold, and the second-order correction adds a net drift of ω k a² along the wave's direction,
//! scaled by cosh(2kh) / (2 sinh²(kh)) in finite depth. Summed over the spectrum this gives
//! u_s = ∫ 2 ω k F(kh) Φ(k) k̂ d²k, since a mode of amplitude `a` carries variance a² / 2. Foam and
//! other floating matter in the undisplaced coordinates of the LOD data move with it.

use std::f32::consts::PI;

use bevy::prelude::*;

use super::{OCEAN_PM_PEAK_COEFF, OceanSimSettings};

/// Matches `DEEP_KH` in `dispersion.wesl`.
const DEEP_KH: f32 = 10.0;
/// Matches `SWELL_NORM_SAMPLES` in `init_h0.wesl`.
const SWELL_NORM_SAMPLES: u32 = 64;
/// Log-spaced radial samples of the Stokes drift integral.
const RADIAL_SAMPLES: u32 = 192;
/// Angular samples of the Stokes drift integral.
const ANGULAR_SAMPLES: u32 = 96;

/// Surface Stokes drift of one cascade in meters per second along X and Z, before any amplitude
/// shaping applied after the FFT. Kept up to date by [`update_ocean_stokes_drift`].
#[derive(Component, Clone, Copy, Debug, Default, PartialEq, Reflect)]
pub struct OceanStokesDrift(pub Vec2);

fn tanh_kh(k: f32, depth: f32) -> f32 {
    let kh = k * depth.max(1e-3);
    if kh > DEEP_KH { 1.0 } else { kh.tanh() }
}

fn omega_finite_depth(k: f32, depth: f32, g: f32) -> f32 {
    (g * k * tanh_kh(k, depth)).max(0.0).sqrt()
}

fn domega_dk_finite_depth(k: f32, depth: f32, g: f32) -> f32 {
    let w = omega_finite_depth(k, depth, g);
    if w < 1e-9 {
        return (g * depth.max(1e-3)).sqrt();
    }
    let th = tanh_kh(k, depth);
    let kh = k * depth.max(1e-3);
    let sech2 = if kh > DEEP_KH { 0.0 } else { 1.0 - th * th };
    g * (th + kh * sech2) / (2.0 * w)
}

fn tma_factor(omega: f32, depth: f32, g: f32) -> f32 {
    let wh = omega * (depth.max(1e-3) / g).sqrt();
    if wh <= 1.0 {
        0.5 * wh * wh
    } else if wh < 2.0 {
        let a = 2.0 - wh;
        1.0 - 0.5 * a * a
    } else {
        1.0
    }
}

fn pierson_moskowitz_s(omega: f32, omega_p: f32, g: f32) -> f32 {
    0.0081 * g * g / omega.powi(5) * (-1.25 * (omega_p / omega).powi(4)).exp()
}

fn jonswap_factor(omega: f32, omega_p: f32, gamma: f32) -> f32 {
    let sigma = if omega <= omega_p { 0.07 } else { 0.09 };
    let r = (-(omega - omega_p).powi(2) / (2.0 * sigma * sigma * omega_p * omega_p + 1e-10)).exp();
    gamma.max(1.0).powf(r)
}

fn directional_spread_d(diff: f32, spread: f32) -> f32 {
    let s = spread.max(1.0);
    diff.cos().max(0.0).powf(s) * (s + 1.0) / (2.0 * PI)
}

fn swell_spread(diff: f32, omega: f32, omega_p: f32, swell: f32) -> f32 {
    let s_xi = 16.0 * (omega_p / omega.max(1e-6)).tanh() * swell * swell;
    (0.5 * diff).cos().abs().powf(2.0 * s_xi)
}

/// Factor `init_h0` applies to the swell-shaped spread so its integral over θ matches the
/// wind-sea spread. It depends on ω only.
fn swell_norm(omega: f32, omega_p: f32, settings: &OceanSimSettings) -> f32 {
    let mut num = 0.0;
    let mut den = 0.0;
    for i in 0..SWELL_NORM_SAMPLES {
        let th = -PI + (i as f32 + 0.5) * (2.0 * PI / SWELL_NORM_SAMPLES as f32);
        let d = directional_spread_d(th, settings.directional_spread);
        num += d;
        den += d * swell_spread(th, omega, omega_p, settings.swell);
    }
    num / den.max(1e-12)
}

/// Finite-depth Stokes factor cosh(2kh) / (2 sinh²(kh)), which is 1 in deep water.
fn stokes_depth_factor(k: f32, depth: f32) -> f32 {
    let kh = k * depth.max(1e-3);
    if kh > DEEP_KH {
        return 1.0;
    }
    let sh = kh.sinh();
    (2.0 * kh).cosh() / (2.0 * sh * sh).max(1e-12)
}

/// Surface Stokes drift of the waves one cascade simulates, in meters per second along X and Z.
///
/// Integrates the same spectrum as `init_h0`, including `amplitude_scale`, over the wavenumbers the
/// cascade's grid holds: from its fundamental 2π / tile up to its Nyquist limit, clipped to its band.
pub fn stokes_drift(settings: &OceanSimSettings) -> Vec2 {
    let g = settings.gravity;
    let depth = settings.depth;
    let delta_k = 2.0 * PI / settings.tile_size.max(1e-3);
    let k_lo = settings.band_k_min.max(delta_k);
    let k_hi = settings
        .band_k_max
        .min(0.5 * delta_k * settings.texture_size as f32);
    if k_hi.partial_cmp(&k_lo) != Some(std::cmp::Ordering::Greater) {
        return Vec2::ZERO;
    }
    let omega_p = (OCEAN_PM_PEAK_COEFF * g / settings.wind_speed.max(1e-6)).clamp(1e-4, 1e4);
    let cutoff = settings.small_wave_cutoff.max(1e-4);
    let log_lo = k_lo.ln();
    let d_log = (k_hi.ln() - log_lo) / RADIAL_SAMPLES as f32;
    let d_theta = 2.0 * PI / ANGULAR_SAMPLES as f32;
    let mut drift = Vec2::ZERO;
    for i in 0..RADIAL_SAMPLES {
        let k = (log_lo + (i as f32 + 0.5) * d_log).exp();
        let omega = omega_finite_depth(k, depth, g);
        let s = pierson_moskowitz_s(omega, omega_p, g)
            * jonswap_factor(omega, omega_p, settings.peak_enhancement)
            * tma_factor(omega, depth, g);
        // Φ(k, θ) = S |dω/dk| D(θ) / k, then d²k = k dk dθ with dk = k d(ln k).
        let radial =
            s * domega_dk_finite_depth(k, depth, g).abs() * (-k * k * cutoff * cutoff).exp();
        let norm = if settings.swell > 0.0 {
            swell_norm(omega, omega_p, settings)
        } else {
            1.0
        };
        let weight = 2.0 * omega * k * stokes_depth_factor(k, depth) * radial * k * d_log;
        let mut lobe = Vec2::ZERO;
        for j in 0..ANGULAR_SAMPLES {
            let theta = -PI + (j as f32 + 0.5) * d_theta;
            let diff = theta - settings.wind_direction;
            let diff = diff.sin().atan2(diff.cos());
            let mut d = directional_spread_d(diff, settings.directional_spread);
            if settings.swell > 0.0 {
                d *= swell_spread(diff, omega, omega_p, settings.swell) * norm;
            }
            lobe += Vec2::new(theta.cos(), theta.sin()) * d;
        }
        drift += lobe * d_theta * weight;
    }
    drift * settings.amplitude_scale * settings.amplitude_scale
}

/// Recomputes [`OceanStokesDrift`] for cascades whose spectrum settings changed.
#[allow(clippy::type_complexity)]
pub(super) fn update_ocean_stokes_drift(
    mut commands: Commands,
    cascades: Query<
        (Entity, &OceanSimSettings, Option<&OceanStokesDrift>),
        Or<(Changed<OceanSimSettings>, Without<OceanStokesDrift>)>,
    >,
) {
    for (entity, settings, current) in &cascades {
        let drift = OceanStokesDrift(stokes_drift(settings));
        if current != Some(&drift) {
            commands.entity(entity).insert(drift);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One cascade wide enough to hold the spectral peak of a 10 m/s wind sea.
    fn wide_cascade() -> OceanSimSettings {
        OceanSimSettings {
            tile_size: 1000.0,
            texture_size: 512,
            ..default()
        }
    }

    #[test]
    fn drift_follows_the_wind() {
        let mut s = wide_cascade();
        s.wind_direction = 0.7;
        let d = stokes_drift(&s);
        assert!(d.length() > 0.0);
        assert!((d.to_angle() - 0.7).abs() < 1e-3, "{d:?}");
    }

    #[test]
    fn drift_is_a_small_fraction_of_wind_speed() {
        // Field and model estimates put the surface Stokes drift of a wind sea at roughly 1 to 2
        // percent of the wind speed. The cascade grid cuts off the shortest waves, so allow a wide
        // margin below.
        let s = wide_cascade();
        let ratio = stokes_drift(&s).length() / s.wind_speed;
        assert!((0.002..0.03).contains(&ratio), "drift / wind = {ratio}");
    }

    #[test]
    fn drift_scales_with_amplitude_squared() {
        let mut s = wide_cascade();
        let base = stokes_drift(&s).length();
        s.amplitude_scale = 2.0;
        let doubled = stokes_drift(&s).length();
        assert!((doubled / base - 4.0).abs() < 1e-3);
    }

    #[test]
    fn empty_band_has_no_drift() {
        let mut s = wide_cascade();
        s.band_k_min = 100.0;
        s.band_k_max = 50.0;
        assert_eq!(stokes_drift(&s), Vec2::ZERO);
    }
}
