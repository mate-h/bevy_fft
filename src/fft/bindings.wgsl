#define_import_path bevy_fft::bindings

#import bevy_fft::complex::c32;
#import bevy_render::globals::Globals;
// Must match `FftSettings` in `src/fft/mod.rs` (`ShaderType` uniform layout).
struct FftSettings {
    size: vec2<u32>,
    orders: u32,
    // `vec2<u32>` aligns to 8; `orders` ends at byte 12 → implicit pad 12..16
    padding: vec2<u32>,
    inverse: u32,
    // If nonzero, run forward FFT then IFFT (OpenGL-style roundtrip). Pattern then fills spatial buffer A.
    roundtrip: u32,
    window_type: u32, // 0=None, 1=Tukey, 2=Blackman, 3=Kaiser
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

// two ping-pong buffers
@group(0) @binding(3) var buffer_a_re: texture_storage_2d<rgba32float, read_write>;
@group(0) @binding(4) var buffer_a_im: texture_storage_2d<rgba32float, read_write>;
@group(0) @binding(5) var buffer_b_re: texture_storage_2d<rgba32float, read_write>;
@group(0) @binding(6) var buffer_b_im: texture_storage_2d<rgba32float, read_write>;
@group(0) @binding(7) var buffer_c_re: texture_storage_2d<rgba32float, read_write>;
@group(0) @binding(8) var buffer_c_im: texture_storage_2d<rgba32float, read_write>;
@group(0) @binding(9) var buffer_d_re: texture_storage_2d<rgba32float, read_write>;
@group(0) @binding(10) var buffer_d_im: texture_storage_2d<rgba32float, read_write>;
