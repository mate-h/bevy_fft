#define_import_path bevy_fft::bindings

#import bevy_fft::complex::c32;
#import bevy_render::globals::Globals;
// Keep this struct byte-for-byte identical to the Rust `FftSettings` uniform.
struct FftSettings {
    size: vec2<u32>,
    orders: u32,
    // Keeps `orders` on a 16-byte boundary before the next fields.
    padding: vec2<u32>,
    // Same numeric encoding as `FftSchedule` on the Rust side.
    schedule: u32,
    // Same numeric encoding as `FftPatternTarget` on the Rust side.
    pattern_target: u32,
    window_type: u32, // 0 none, 1 Tukey, 2 Blackman, 3 Kaiser
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

// Complex workspace buffers **A**–**D**.
@group(0) @binding(3) var buffer_a_re: texture_storage_2d<rgba32float, read_write>;
@group(0) @binding(4) var buffer_a_im: texture_storage_2d<rgba32float, read_write>;
@group(0) @binding(5) var buffer_b_re: texture_storage_2d<rgba32float, read_write>;
@group(0) @binding(6) var buffer_b_im: texture_storage_2d<rgba32float, read_write>;
@group(0) @binding(7) var buffer_c_re: texture_storage_2d<rgba32float, read_write>;
@group(0) @binding(8) var buffer_c_im: texture_storage_2d<rgba32float, read_write>;
@group(0) @binding(9) var buffer_d_re: texture_storage_2d<rgba32float, read_write>;
@group(0) @binding(10) var buffer_d_im: texture_storage_2d<rgba32float, read_write>;
