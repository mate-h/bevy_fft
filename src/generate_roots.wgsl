#import bevy_fft::c32::{c32, exp};

@group(0) @binding(0) var roots: array<c32>;

@compute
@workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    let n = firstLeadingBit(idx + 1) - 1;
    let k = idx - (1 << n);
    let root = }
