#define_import_path bevy_fft::fft_common

#import bevy_fft::{
    complex::{
        c32,
    },
    bindings::{
        roots_buffer,
    },
};

fn fft_reverse_lower_bits(x: u32, order: u32) -> u32 {
    return reverseBits(x) >> (32u - order);
}

fn get_fft_root(stage_plus_one: u32, index: u32) -> c32 {
    let base = 1u << stage_plus_one;
    let count = max(1u, base >> 1u);
    let i = base + index % count;
    return roots_buffer.roots[i];
}
