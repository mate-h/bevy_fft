#define_import_path bevy_fft::bindings

#import bevy_fft::complex::c32;
#import bevy_render::globals::Globals;
struct FftSettings {
    size: vec2<u32>,
    padding: vec2<u32>,
    orders: u32,
    inverse: u32,
    window_type: u32,      // 0=None, 1=Tukey, 2=Blackman, 3=Kaiser
    window_strength: f32,  // 0.0-1.0 how strongly to apply the window
    radial_falloff: f32,   // For visualization only
    normalization: f32,    // For visualization only
}

struct FftRoots {
    roots: array<c32, 8192>,
}

@group(0) @binding(0) var<uniform> globals: Globals;
@group(0) @binding(1) var<uniform> settings: FftSettings;
@group(0) @binding(2) var<storage, read_write> roots_buffer: FftRoots;

// two ping-pong buffers
@group(0) @binding(3) var buffer_a_re: texture_storage_2d<rgba32float, read_write>;
@group(0) @binding(4) var buffer_a_im: texture_storage_2d<rgba32float, read_write>;
@group(0) @binding(5) var buffer_b_re: texture_storage_2d<rgba32float, read_write>;
@group(0) @binding(6) var buffer_b_im: texture_storage_2d<rgba32float, read_write>;
@group(0) @binding(7) var buffer_c_re: texture_storage_2d<rgba32float, read_write>;
@group(0) @binding(8) var buffer_c_im: texture_storage_2d<rgba32float, read_write>;
@group(0) @binding(9) var buffer_d_re: texture_storage_2d<rgba32float, read_write>;
@group(0) @binding(10) var buffer_d_im: texture_storage_2d<rgba32float, read_write>;
