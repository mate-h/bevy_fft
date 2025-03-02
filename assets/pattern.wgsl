#define_import_path bevy_fft::pattern

#import bevy_fft::{
    complex::{
        splat_c32_n,
        c32,
        c32_2,
        c32_3,
        c32_4,
    },
    bindings::{
        globals,
        settings,
        buffer_a_re,
        buffer_a_im,
        buffer_b_re,
        buffer_b_im,
        buffer_c_re,
        buffer_c_im,
        buffer_d_re,
        buffer_d_im,
    }
};

#ifdef CHANNELS
#if CHANNELS == 1 
    alias c32_n = c32;
#else if CHANNELS == 2;
    alias c32_n = c32_2;
#else if CHANNELS == 3;
    alias c32_n = c32_3;
#else if CHANNELS == 4;
    alias c32_n = c32_4;
#endif
#endif

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
    value_r_im = get_value(distance, phase_r);
    value_g_im = get_value(distance, phase_g);
    value_b_im = get_value(distance, phase_b);
    
    // Store as complex value with RGB channels
    textureStore(buffer_a_re, pos, vec4<f32>(value_r, value_g, value_b, 1.0));
    textureStore(buffer_a_im, pos, vec4<f32>(value_r_im, value_g_im, value_b_im, 1.0));
}

fn get_value(distance: f32, phase_shift: f32) -> f32 {
    // Make frequency oscillate over time using a sine wave
    let frequency_amplitude = 1.0; // How much the frequency changes
    let frequency_oscillation = 4.0; 
    let amplitude = 1.0;
    let falloff = 10.0;
    return amplitude * cos(distance * frequency_oscillation * 6.28 + phase_shift) * exp(-falloff * pow(distance, 2.0));
}