#define_import_path bevy_fft::pattern

#import bevy_fft::complex::c32;
#import bevy_render::globals::Globals;

// Must match `FftSettings` / `FftRoots` and binding indices in `bevy_fft::bindings`
// (pattern shares bind group 0 with FFT pipelines).
struct FftSettings {
    size: vec2<u32>,
    orders: u32,
    padding: vec2<u32>,
    inverse: u32,
    roundtrip: u32,
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
    
    // Calculate values for each RGB channel
    value_r = get_value(distance, phase_r);
    value_g = get_value(distance, phase_g);
    value_b = get_value(distance, phase_b);
    
    // Calculate imaginary components (90 degrees phase shift from real components)
    // value_r_im = get_value(distance, phase_r);
    // value_g_im = get_value(distance, phase_g);
    // value_b_im = get_value(distance, phase_b);
    value_r_im = 0.0;
    value_g_im = 0.0;
    value_b_im = 0.0;
    
    // Store as complex value with RGB channels. When inverse mode is enabled
    // we populate buffer C directly so the IFFT can consume it without
    // running the forward FFT stage first.
    if (settings.roundtrip != 0u) {
        textureStore(buffer_a_re, pos, vec4<f32>(value_r, value_g, value_b, 1.0));
        textureStore(buffer_a_im, pos, vec4<f32>(value_r_im, value_g_im, value_b_im, 1.0));
    } else if (settings.inverse != 0u) {
        textureStore(buffer_c_re, pos, vec4<f32>(value_r, value_g, value_b, 1.0));
        textureStore(buffer_c_im, pos, vec4<f32>(value_r_im, value_g_im, value_b_im, 1.0));
    } else {
        textureStore(buffer_a_re, pos, vec4<f32>(value_r, value_g, value_b, 1.0));
        textureStore(buffer_a_im, pos, vec4<f32>(value_r_im, value_g_im, value_b_im, 1.0));
    }
}

fn get_value(distance: f32, phase_shift: f32) -> f32 {
    // Make frequency oscillate over time using a sine wave
    let frequency_amplitude = 1.0; // How much the frequency changes
    let frequency_oscillation = 4.0; 
    let amplitude = 1.0;
    let falloff = 10.0;
    return amplitude * cos(distance * frequency_oscillation * 6.28 + phase_shift) * exp(-falloff * pow(distance, 2.0));
}

/// Diagonal (u + v) phase; R/G/B sines with 120° separation.
/// - `roundtrip`: spatial data in buffer A (then forward FFT → C and IFFT → B).
/// - `inverse` only: synthetic spectrum in C (IFFT maps it to spatial math, not “same” diagonal).
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

    // Phase along diagonal (1,1); lower factor = longer waves (smaller frequency).
    let t = f32(pos.x + pos.y) * (6.28318 * 0.025);
    let time = globals.time * 1.5;

    let value_r = sin(t + time);
    let value_g = sin(t + time + 6.28318 / 3.0);
    let value_b = sin(t + time + 2.0 * 6.28318 / 3.0);

    if (settings.roundtrip != 0u) {
        textureStore(buffer_a_re, pos, vec4<f32>(value_r, value_g, value_b, 1.0));
        textureStore(buffer_a_im, pos, vec4<f32>(0.0, 0.0, 0.0, 1.0));
    } else if (settings.inverse != 0u) {
        textureStore(buffer_c_re, pos, vec4<f32>(value_r, value_g, value_b, 1.0));
        textureStore(buffer_c_im, pos, vec4<f32>(0.0, 0.0, 0.0, 1.0));
    } else {
        textureStore(buffer_a_re, pos, vec4<f32>(value_r, value_g, value_b, 1.0));
        textureStore(buffer_a_im, pos, vec4<f32>(0.0, 0.0, 0.0, 1.0));
    }
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

    let t = f32(pos.x) * (6.28318 * 0.025);
    let time = globals.time * 1.5;

    let value_r = sin(t + time);
    let value_g = sin(t + time + 6.28318 / 3.0);
    let value_b = sin(t + time + 2.0 * 6.28318 / 3.0);

    if (settings.roundtrip != 0u) {
        textureStore(buffer_a_re, pos, vec4<f32>(value_r, value_g, value_b, 1.0));
        textureStore(buffer_a_im, pos, vec4<f32>(0.0, 0.0, 0.0, 1.0));
    } else if (settings.inverse != 0u) {
        textureStore(buffer_c_re, pos, vec4<f32>(value_r, value_g, value_b, 1.0));
        textureStore(buffer_c_im, pos, vec4<f32>(0.0, 0.0, 0.0, 1.0));
    } else {
        textureStore(buffer_a_re, pos, vec4<f32>(value_r, value_g, value_b, 1.0));
        textureStore(buffer_a_im, pos, vec4<f32>(0.0, 0.0, 0.0, 1.0));
    }
}
