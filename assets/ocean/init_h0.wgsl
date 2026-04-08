const PI: f32 = 3.14159265359;
/// Fully developed Pierson–Moskowitz peak in rad/s: `ω_pm ≈ PM_PEAK_COEFF * g / U` with fetch-limited U in m/s.
const PM_PEAK_COEFF: f32 = 0.87;
/// After using PM peak `ω_pm`, clamp `ω_p` (rad/s) for numerical stability only. Not a physical prior.
const OMEGA_P_MIN: f32 = 1e-4;
const OMEGA_P_MAX: f32 = 1e4;

@group(0) @binding(0) var H0_out: texture_storage_2d<rgba32float, write>;

struct OceanH0Params {
    texture_size: u32,
    _pad0: u32,
    tile_size: f32,
    wind_direction: f32,
    wind_speed: f32,
    peak_enhancement: f32,
    directional_spread: f32,
    small_wave_cutoff: f32,
    gravity: f32,
    // Linear multiplier on RMS after calibration; 1 is nominal. Not a clamp.
    amplitude_scale: f32,
    h0_serial: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(1) var<uniform> params: OceanH0Params;

fn pcg32(v: u32) -> u32 {
    var x = v * 747796405u + 2891336453u;
    let word = ((x >> ((x >> 28u) + 4u)) ^ x) * 2778037370u;
    return (word >> 22u) ^ word;
}

fn rand01(h: u32) -> f32 {
    return f32(pcg32(h)) * (1.0 / 4294967296.0);
}

fn gaussian2(seed: u32) -> vec2<f32> {
    let u1 = max(rand01(seed), 1e-7);
    let u2 = rand01(seed + 797u);
    let r = sqrt(-2.0 * log(u1));
    let th = 2.0 * PI * u2;
    return vec2(r * cos(th), r * sin(th));
}

/// One-sided Pierson–Moskowitz variance density S(ω) in m²·s (ω in rad/s), deep water.
fn pierson_moskowitz_S(omega: f32, omega_p: f32, g: f32) -> f32 {
    let alpha = 0.0081;
    let beta = 1.25;
    return alpha * g * g / pow(omega, 5.0) * exp(-beta * pow(omega_p / omega, 4.0));
}

/// JONSWAP multiplier γ^r(ω) with Hasselmann-style σ and peak ωp.
fn jonswap_factor(omega: f32, omega_p: f32, gamma: f32) -> f32 {
    // empirical values from Hasselmann
    let sigma = select(0.09, 0.07, omega <= omega_p);
    let r = exp(-pow(omega - omega_p, 2.0) / (2.0 * sigma * sigma * omega_p * omega_p + 1e-10));
    return pow(max(gamma, 1.0), r);
}

/// Directional weight cos^s_+(Δθ), scaled so total directional variance is in the right order of magnitude.
/// Exact ∫ normalization would need Γ; this matches common game-ocean usage (see Tessendorf-style lobes).
fn directional_spread_D(diff_from_wind: f32, spread: f32) -> f32 {
    let s = max(spread, 1.0);
    let c = max(cos(diff_from_wind), 0.0);
    return pow(c, s) * (s + 1.0) / (2.0 * PI);
}

/// Omnidirectional variance density in k (rad/m): Φ(k) = S(ω) |dω/dk| D(θ) / k, with ω = √(g|k|).
/// Pure physics only; RMS of each `h̃(k)` mode is fixed in `init_h0` from Φ, Δk, N, and the IFFT (1/N per axis).
///
/// Uses `ω_pm = PM_PEAK_COEFF * g / U` as the PM peak before JONSWAP.
fn jonswap_phi_k(k: vec2<f32>) -> f32 {
    let g = params.gravity;
    let k2 = dot(k, k);
    let k_len = sqrt(k2);
    if k_len < 1e-9 {
        return 0.0;
    }
    let omega = sqrt(g * k_len);
    let u = params.wind_speed;
    let omega_p_raw = PM_PEAK_COEFF * g / max(u, 1e-6);
    let omega_p = clamp(omega_p_raw, OMEGA_P_MIN, OMEGA_P_MAX);
    let S_pm = pierson_moskowitz_S(omega, omega_p, g);
    let jf = jonswap_factor(omega, omega_p, params.peak_enhancement);
    let S = S_pm * jf;
    let dwdk = g / (2.0 * omega);
    let theta = atan2(k.y, k.x);
    let diff = theta - params.wind_direction;
    let D = directional_spread_D(diff, params.directional_spread);
    var phi = S * abs(dwdk) * D / max(k_len, 1e-9);
    let l = max(params.small_wave_cutoff, 1e-4);
    phi *= exp(-k2 * l * l);
    return max(phi, 0.0);
}

@compute @workgroup_size(8, 8, 1)
fn init_h0(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = params.texture_size;
    if gid.x >= n || gid.y >= n {
        return;
    }
    let id_u = vec2<u32>(gid.xy);
    let mirror = vec2<u32>((n - gid.x) % n, (n - gid.y) % n);

    let delta_k = 2.0 * PI / params.tile_size;
    let nx = f32(i32(gid.x) - i32(n / 2u));
    let nz = f32(i32(gid.y) - i32(n / 2u));
    let k = vec2<f32>(nx, nz) * delta_k;

    let dk_sq = delta_k * delta_k;
    let seed = gid.x + gid.y * n + params.h0_serial * 7919u;
    let xi = gaussian2(seed);
    let phi_k = jonswap_phi_k(k);
    // Two separable IFFTs each apply 1/N. For independent Fourier coefficients on an N×N Hermitian grid,
    // target E[|h̃|²] = (N⁴/2) Φ Δk² so that E[η²] ≈ Σ Φ Δk² in meters². With h̃ = amp (ξ_r + i ξ_i),
    // E[|h̃|²] = 2 amp², hence amp = (N²/2) √(Φ Δk²).
    let n_f = f32(n);
    let amp =
        sqrt(max(phi_k * dk_sq, 0.0)) * n_f * n_f * 0.5 * params.amplitude_scale;
    let h0_k = vec2<f32>(xi.x * amp, xi.y * amp);

    let nx_m = f32(i32(mirror.x) - i32(n / 2u));
    let nz_m = f32(i32(mirror.y) - i32(n / 2u));
    let k_m = vec2<f32>(nx_m, nz_m) * delta_k;
    let xi_m = gaussian2(seed ^ 0xdeadbeefu);
    let phi_m = jonswap_phi_k(k_m);
    let amp_m =
        sqrt(max(phi_m * dk_sq, 0.0)) * n_f * n_f * 0.5 * params.amplitude_scale;
    let h0_mk = vec2<f32>(xi_m.x * amp_m, xi_m.y * amp_m);
    let conj_mk = vec2<f32>(h0_mk.x, -h0_mk.y);

    textureStore(H0_out, id_u, vec4<f32>(h0_k, conj_mk));
}
