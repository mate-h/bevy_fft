#define_import_path bevy_fft::texel

#import bevy_fft::{
    complex::{
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
    },
    // These functions are broken when using storage textures as arguments to functions
    // therefore we need to import the bindings directly in order to use them
    bindings::{
        src_tex,
        dst_tex,
    },
};

#ifdef CHANNELS
#if CHANNELS == 1 
    alias c32_n = c32;
    alias packed_c32_n = packed_c32;
#else if CHANNELS == 2;
    alias c32_n = c32_2;
    alias packed_c32_n = packed_c32_2;
#else if CHANNELS == 3;
    alias c32_n = c32_3;
    alias packed_c32_n = packed_c32_3;
#else if CHANNELS == 4;
    alias c32_n = c32_4;
    alias packed_c32_n = packed_c32_4;
#endif
#endif

// LOADING

#ifdef CHANNELS
#if CHANNELS == 1 
fn load_c32_n(pos: vec2<u32>) -> c32 {
    let packed = textureLoad(dst_tex, pos).x;
    // create u32_data from packed
    var data = packed_c32(packed);
    return unpack_c32(data);
}
#else if CHANNELS == 2
fn load_c32_n(pos: vec2<u32>) -> c32_2 {
    let packed = textureLoad(dst_tex, pos).xy;
    var data = packed_c32_2(packed);
    return unpack_c32_2(data);
}
#else if CHANNELS == 3
fn load_c32_n(pos: vec2<u32>) -> c32_3 {
    let packed = textureLoad(dst_tex, pos).xyz;
    var data = packed_c32_3(packed);
    return unpack_c32_3(data);
}
#else if CHANNELS == 4
fn load_c32_n(pos: vec2<u32>) -> c32_4 {
    let packed = textureLoad(dst_tex, pos).xyzw;
    var data = packed_c32_4(packed);
    return unpack_c32_4(data);
}
#endif
#endif

fn load_real_c32(tex: texture_2d<f32>, pos: vec2<u32>) -> c32 {
    let real = textureLoad(tex, pos, 0).x;
    return c32(real, 0.0);
}

fn load_real_c32_2(tex: texture_2d<f32>, pos: vec2<u32>) -> c32_2 {
    let real = textureLoad(tex, pos, 0).xy;
    return c32_2(real, vec2(0.0));
}

fn load_real_c32_3(tex: texture_2d<f32>, pos: vec2<u32>) -> c32_3 {
    let real = textureLoad(tex, pos, 0).xyz;
    return c32_3(real, vec3(0.0));
}

fn load_real_c32_4(tex: texture_2d<f32>, pos: vec2<u32>) -> c32_4 {
    let real = textureLoad(tex, pos, 0).xyzw;
    return c32_4(real, vec4(0.0));
}

#ifdef CHANNELS
fn load_real_c32_n(tex: texture_2d<f32>, pos: vec2<u32>) -> c32_n {
#if CHANNELS == 1 
    return load_real_c32(tex, pos);
#else if CHANNELS == 2
    return load_real_c32_2(tex, pos);
#else if CHANNELS == 3
    return load_real_c32_3(tex, pos);
#else if CHANNELS == 4
    return load_real_c32_4(tex, pos);
#endif
}
#endif

// STORING
#ifdef CHANNELS
#if CHANNELS == 1 
fn store_c32_n(pos: vec2<u32>, c: c32) {
    let packed = pack_c32(c);
    textureStore(dst_tex, pos, vec4(packed.value, 0u, 0u, 0u));
}
#else if CHANNELS == 2
fn store_c32_n(pos: vec2<u32>, c: c32_2) {
    let packed = pack_c32_2(c);
    textureStore(dst_tex, pos, vec4(packed.value, 0u, 0u));
}
#else if CHANNELS == 3
fn store_c32_n(pos: vec2<u32>, c: c32_3) {
    let packed = pack_c32_3(c);
    textureStore(dst_tex, pos, vec4(packed.value, 0u));
}
#else if CHANNELS == 4
fn store_c32_n(pos: vec2<u32>, c: c32_4) {
    let packed = pack_c32_4(c);
    textureStore(dst_tex, pos, vec4(packed.value));
}
#endif
#endif

#ifdef CHANNELS
#if CHANNELS == 1 
fn store_real_c32_n(pos: vec2<u32>, c: c32) {
    textureStore(dst_tex, pos, vec4(c.re, 0.0, 0.0, 0.0));
}
#else if CHANNELS == 2
fn store_real_c32_n(tex: texture_storage_2d<rgb32uint, read_write>, pos: vec2<u32>, c: c32_3) {
    textureStore(dst_tex, pos, vec4(c.re, 0.0, 0.0));
}
#else if CHANNELS == 3
fn store_real_c32_n(tex: texture_storage_2d<rgba32uint, read_write>, pos: vec2<u32>, c: c32_3) {
    textureStore(dst_tex, pos, vec4(c.re, 0.0));
}
#else if CHANNELS == 4
fn store_real_c32_n(tex: texture_storage_2d<rgba32uint, read_write>, pos: vec2<u32>, c: c32_4) {
    textureStore(dst_tex, pos, vec4(c.re));
}
#endif
#endif