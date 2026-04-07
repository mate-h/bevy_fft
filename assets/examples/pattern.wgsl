#define_import_path bevy_fft::pattern

#import bevy_fft::complex::c32;
#import bevy_render::globals::Globals;

// Must match `FftSettings` / `FftRoots` and binding indices in `bevy_fft::bindings`
// (pattern shares bind group 0 with FFT pipelines).
struct FftSettings {
    size: vec2<u32>,
    orders: u32,
    padding: vec2<u32>,
    schedule: u32,
    pattern_target: u32,
    window_type: u32,
    window_strength: f32,
    radial_falloff: f32,
    normalization: f32,
}

struct FftRoots {
    roots: array<c32, 8192>,
}

@group(0) @binding(0) var<uniform> globals: Globals;
@group(0) @binding(1) var<uniform> settings: FftSettings;
@group(0) @binding(2) var<storage, read_write> roots_buffer: FftRoots;
@group(0) @binding(3) var buffer_a_re: texture_storage_2d<rgba32float, read_write>;
@group(0) @binding(4) var buffer_a_im: texture_storage_2d<rgba32float, read_write>;
@group(0) @binding(5) var buffer_b_re: texture_storage_2d<rgba32float, read_write>;
@group(0) @binding(6) var buffer_b_im: texture_storage_2d<rgba32float, read_write>;
@group(0) @binding(7) var buffer_c_re: texture_storage_2d<rgba32float, read_write>;
@group(0) @binding(8) var buffer_c_im: texture_storage_2d<rgba32float, read_write>;
@group(0) @binding(9) var buffer_d_re: texture_storage_2d<rgba32float, read_write>;
@group(0) @binding(10) var buffer_d_im: texture_storage_2d<rgba32float, read_write>;

const PATTERN_TARGET_SPECTRUM_C: u32 = 1u;

fn store_pattern_rgba(pos: vec2<u32>, re: vec4<f32>, im: vec4<f32>) {
    if (settings.pattern_target == PATTERN_TARGET_SPECTRUM_C) {
        textureStore(buffer_c_re, pos, re);
        textureStore(buffer_c_im, pos, im);
    } else {
        textureStore(buffer_a_re, pos, re);
        textureStore(buffer_a_im, pos, im);
    }
}

@compute
@workgroup_size(16, 16, 1)
fn generate_concentric_circles(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    let pos = vec2<u32>(global_id.xy);
    let size = settings.size;
    
    // Skip if out of bounds
    if (any(pos >= size)) {
        return;
    }
    
    // Calculate normalized coordinates in [-1, 1] range
    let center = vec2<f32>(size) / 2.0;
    let uv = (vec2<f32>(pos) - center) / center;
    
    // Generate concentric circles pattern in frequency domain
    let distance = length(uv);
    
    
    let phase_shift = globals.time;
    // Create 3 phase-shifted patterns (60 degrees = π/3 radians apart)
    let phase_r = phase_shift;
    let phase_g = phase_shift + 3.14159 / 3.0;  // 60 degrees phase shift
    let phase_b = phase_shift + 2.0 * 3.14159 / 3.0;  // 120 degrees phase shift

    var value_r = 0.0;
    var value_g = 0.0;
    var value_b = 0.0;
    var value_r_im = 0.0;
    var value_g_im = 0.0;
    var value_b_im = 0.0;
    
    // Radial bands + Cartesian / diagonal beats for dense high-frequency content.
    let k = 6.28318;
    let nx = uv.x * 67.0;
    let ny = uv.y * 61.0;
    let ndiag = (uv.x * 2.4 + uv.y * 3.8) * 44.0;
    let nmix = (uv.x * 5.1 - uv.y * 4.7) * 38.0;
    let hf_r =
        0.10 * sin(nx * k + phase_r * 1.12) * cos(ny * k - phase_r * 0.88)
            + 0.085 * sin(ndiag * k + globals.time * 2.9)
            + 0.065 * sin(nmix * k - globals.time * 3.2);
    let hf_g =
        0.10 * sin(nx * k + phase_g * 1.12) * cos(ny * k - phase_g * 0.88)
            + 0.085 * sin(ndiag * k + globals.time * 2.9 + 0.7)
            + 0.065 * sin(nmix * k - globals.time * 3.2 + 0.9);
    let hf_b =
        0.10 * sin(nx * k + phase_b * 1.12) * cos(ny * k - phase_b * 0.88)
            + 0.085 * sin(ndiag * k + globals.time * 2.9 + 1.4)
            + 0.065 * sin(nmix * k - globals.time * 3.2 + 1.8);

    // Calculate values for each RGB channel
    value_r = get_value(distance, phase_r) + hf_r;
    value_g = get_value(distance, phase_g) + hf_g;
    value_b = get_value(distance, phase_b) + hf_b;
    
    value_r_im = 0.0;
    value_g_im = 0.0;
    value_b_im = 0.0;
    
    store_pattern_rgba(
        pos,
        vec4<f32>(value_r, value_g, value_b, 1.0),
        vec4<f32>(value_r_im, value_g_im, value_b_im, 1.0),
    );
}

fn get_value(distance: f32, phase_shift: f32) -> f32 {
    // Radial harmonics (integer-ish ring counts); weights chosen so sum |a| < 1 for headroom vs HF layer.
    let k = 6.28318;
    let a1 = 0.30;
    let a2 = 0.24;
    let a3 = 0.18;
    let a4 = 0.13;
    let a5 = 0.095;
    let wobble = sin(globals.time * 0.85);
    return a1 * cos(distance * 5.0 * k + phase_shift)
        + a2 * cos(distance * 11.0 * k + phase_shift * 1.13 + wobble)
        + a3 * cos(distance * 19.0 * k - phase_shift * 0.91 + globals.time * 2.1)
        + a4 * cos(distance * 31.0 * k + phase_shift * 1.27 + globals.time * 3.4)
        + a5 * cos(distance * 47.0 * k - phase_shift * 0.73 + globals.time * 1.55);
}

/// Diagonal (u + v) phase; R/G/B sines with 120° separation. Output buffer is [`FftPatternTarget`].
@compute
@workgroup_size(16, 16, 1)
fn generate_diagonal_rgb_sine(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    let pos = vec2<u32>(global_id.xy);
    let size = settings.size;
    if (any(pos >= size)) {
        return;
    }

    let u = f32(pos.x) / max(f32(max(size.x, 1u) - 1u), 1.0);
    let v = f32(pos.y) / max(f32(max(size.y, 1u) - 1u), 1.0);
    let k = 6.28318;
    // Multiple directions / scales so the whole square is busy (no radial fade).
    let t1 = (u + v) * k * 10.0;
    let t2 = (u * 3.7 - v * 2.9) * k * 6.0;
    let t3 = (u * v) * k * 24.0;
    let time = globals.time * 2.2;

    let value_r =
        0.45 * sin(t1 + time)
            + 0.33 * sin(t2 + time * 1.07)
            + 0.22 * sin(t3 + time * 0.83);
    let value_g =
        0.45 * sin(t1 + time + 2.0944)
            + 0.33 * sin(t2 + time * 1.07 + 2.0944)
            + 0.22 * sin(t3 + time * 0.83 + 2.0944);
    let value_b =
        0.45 * sin(t1 + time + 4.18879)
            + 0.33 * sin(t2 + time * 1.07 + 4.18879)
            + 0.22 * sin(t3 + time * 0.83 + 4.18879);

    store_pattern_rgba(
        pos,
        vec4<f32>(value_r, value_g, value_b, 1.0),
        vec4<f32>(0.0, 0.0, 0.0, 1.0),
    );
}

/// Horizontal phase (depends on `x` only); same RGB separation as diagonal.
@compute
@workgroup_size(16, 16, 1)
fn generate_horizontal_rgb_sine(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    let pos = vec2<u32>(global_id.xy);
    let size = settings.size;
    if (any(pos >= size)) {
        return;
    }

    let u = f32(pos.x) / max(f32(max(size.x, 1u) - 1u), 1.0);
    let v = f32(pos.y) / max(f32(max(size.y, 1u) - 1u), 1.0);
    let k = 6.28318;
    // Strong stripes + medium bands + dense high-frequency grid / checker beats (full frame).
    let tx0 = u * k * 14.0;
    let tx1 = u * k * 29.0;
    let ty = v * k * 9.0;
    let tmix = (u + v * 0.5) * k * 17.0;
    let txh = u * k * 53.0;
    let tyh = v * k * 47.0;
    let txy = (u * 2.7 - v * 2.3) * k * 41.0;
    let twarp = sin(u * k * 71.0) * sin(v * k * 67.0);
    let time = globals.time * 2.0;

    let value_r =
        0.34 * sin(tx0 + time)
            + 0.23 * sin(tx1 - time * 1.11)
            + 0.14 * sin(ty + time * 0.95)
            + 0.095 * sin(tmix + time * 1.2)
            + 0.085 * sin(txh + time * 1.35)
            + 0.075 * sin(tyh - time * 1.28)
            + 0.065 * sin(txy + time * 1.5)
            + 0.055 * twarp;
    let value_g =
        0.34 * sin(tx0 + time + 2.0944)
            + 0.23 * sin(tx1 - time * 1.11 + 2.0944)
            + 0.14 * sin(ty + time * 0.95 + 2.0944)
            + 0.095 * sin(tmix + time * 1.2 + 2.0944)
            + 0.085 * sin(txh + time * 1.35 + 2.0944)
            + 0.075 * sin(tyh - time * 1.28 + 2.0944)
            + 0.065 * sin(txy + time * 1.5 + 2.0944)
            + 0.055 * twarp;
    let value_b =
        0.34 * sin(tx0 + time + 4.18879)
            + 0.23 * sin(tx1 - time * 1.11 + 4.18879)
            + 0.14 * sin(ty + time * 0.95 + 4.18879)
            + 0.095 * sin(tmix + time * 1.2 + 4.18879)
            + 0.085 * sin(txh + time * 1.35 + 4.18879)
            + 0.075 * sin(tyh - time * 1.28 + 4.18879)
            + 0.065 * sin(txy + time * 1.5 + 4.18879)
            + 0.055 * twarp;

    store_pattern_rgba(
        pos,
        vec4<f32>(value_r, value_g, value_b, 1.0),
        vec4<f32>(0.0, 0.0, 0.0, 1.0),
    );
}
