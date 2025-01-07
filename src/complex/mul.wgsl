#import bevy_fft::c32::c32;

#import bevy_fft::c32::{
    c32,
    c32_n,
    load_c32_n,
    mul_c32_n,
    muls_c32_n,
    fma_c32_n,
}

#ifdef SRC_SCALAR
alias src_texel_format = texel_c32;
#else
alias src_texel_format = texel_c32_n;
#endif

@group(0) @binding(0) var src_tex: texture_storage_2d<src_texel_format>;
//@group(0) @binding(1) var src_tex_meta: 

@group(0) @binding(2) var dst_tex: texture_storage_2d<texel_c32_n>;
//@group(0) @binding(3) var dst_tex_meta: 

@compute
@workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let dst = load_c32_n(dst_tex, id.xy);

    #ifdef SRC_SCALAR
    let src = load_c32(src_tex, id.xy);
    let new_dst = muls_c32_n(dst, src);
    #else
    let src = load_c32_n(src_tex, id.xy);
    let new_dst = mul_c32_n(dst, src);
    #endif

    store_c32_n(dst_tex, id.xy, new_dst);
}
