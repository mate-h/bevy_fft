#define_import_path bevy_fft::bindings

#import bevy_fft::complex::c32;

struct FftSettings {
    src_size: vec2<u32>,
    src_padding: vec2<u32>,
    orders: u32,
}

struct FftRoots {
    roots: array<c32, 8192>,
}

@group(0) @binding(0) var<uniform> uniforms: FftSettings;
@group(0) @binding(1) var<storage, read_write> roots_buffer: FftRoots;
@group(0) @binding(2) var src_tex: texture_2d<f32>;
@group(0) @binding(3) var dst_tex: texture_storage_2d<rgba32uint, read_write>;
@group(0) @binding(4) var re_tex: texture_storage_2d<rgba32float, read_write>;
@group(0) @binding(5) var im_tex: texture_storage_2d<rgba32float, read_write>;