import bevy_fft::complex::c32::{
    c32,
    c32_2,
    c32_3,
    c32_4,
    packed_c32,
    packed_c32_2,
    packed_c32_3,
    packed_c32_4,
    pack_c32,
    pack_c32_2,
    pack_c32_3,
    pack_c32_4,
    unpack_c32,
    unpack_c32_2,
    unpack_c32_3,
    unpack_c32_4,
};
import bevy_fft::fft::bindings::{
    src_tex,
    dst_tex,
    re_tex,
    im_tex,
};

@if(CHANNELS)
alias c32_n = c32_4;
@if(CHANNELS)
alias packed_c32_n = packed_c32_4;

// --- load packed texels as channel structs ---


fn load_re_c32(tex: texture_2d<f32>, pos: vec2<u32>) -> c32 {
    let real = textureLoad(tex, pos, 0).x;
    return c32(real, 0.0);
}

fn load_re_c32_2(tex: texture_2d<f32>, pos: vec2<u32>) -> c32_2 {
    let real = textureLoad(tex, pos, 0).xy;
    return c32_2(real, vec2(0.0));
}

fn load_re_c32_3(tex: texture_2d<f32>, pos: vec2<u32>) -> c32_3 {
    let real = textureLoad(tex, pos, 0).xyz;
    return c32_3(real, vec3(0.0));
}

fn load_re_c32_4(tex: texture_2d<f32>, pos: vec2<u32>) -> c32_4 {
    let real = textureLoad(tex, pos, 0).xyzw;
    return c32_4(real, vec4(0.0));
}


// --- pack channel structs into storage texels ---




