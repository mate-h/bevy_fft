#import bevy_fft::{c32, c32::c32}

@group(0) @binding(0) var impulse_tex: texture_2d<u32>;
@group(0) @binding(1) var freq_tex: texture_storage_2d<r32uint>;

@compute
@workgroup_size(16, 16, 1)
fn fft_vertical(@builtin(global_invocation_id) id: vec3<u32>) {
}

@compute
@workgroup_size(16, 16, 1)
fn fft_horizontal(@builtin(global_invocation_id) id: vec3<u32>) {
}
