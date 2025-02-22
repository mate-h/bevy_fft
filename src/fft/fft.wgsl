#import bevy_fft::c32::{
    c32,
    c32_2,
    c32_3,
    c32_4,
    load_c32_n,
    load_real_c32_n,
    pack_c32,
    pack_c32_2,
    pack_c32_3,
    pack_c32_4,
    fma_c32_n,
    splat_c32_n,
}


struct FftSettings {
    src_size: vec2<u32>,
    src_padding: vec2<u32>,
    orders: u32,
}

@group(0) @binding(0) var<uniform> uniforms: FftSettings;
@group(0) @binding(1) var<storage, read> roots: array<c32, 8192>;
// #ifdef CHANNELS
// #if CHANNELS == 1
// @group(0) @binding(2) var src_tex: texture_2d<f32>;
// @group(0) @binding(3) var dst_tex: texture_storage_2d<r32uint, read_write>;
// #else if CHANNELS == 2
// @group(0) @binding(2) var src_tex: texture_2d<f32>;
// @group(0) @binding(3) var dst_tex: texture_storage_2d<rg32uint, read_write>;
// #else if CHANNELS == 3
// @group(0) @binding(2) var src_tex: texture_2d<f32>;
// @group(0) @binding(3) var dst_tex: texture_storage_2d<rgba32uint, read_write>;
// #else if CHANNELS == 4
// @group(0) @binding(2) var src_tex: texture_2d<f32>;
// @group(0) @binding(3) var dst_tex: texture_storage_2d<rgba32uint, read_write>;
// #endif
// #endif

// #ifdef CHANNELS
// #if CHANNELS == 1
// var<workgroup> temp: array<c32, 256>;
// #else if CHANNELS == 2
// var<workgroup> temp: array<c32_2, 256>;
// #else if CHANNELS == 3
// var<workgroup> temp: array<c32_3, 256>;
// #else if CHANNELS == 4
// var<workgroup> temp: array<c32_4, 256>;
// #endif
// #endif

// without conditional compilation
@group(0) @binding(2) var src_tex: texture_2d<f32>;
@group(0) @binding(3) var dst_tex: texture_storage_2d<rgba32uint, read_write>;

var<workgroup> temp: array<c32_4, 256>;

fn get_root(order: u32, index: u32) -> c32 {
    let base = 1u << order;
    let count = base >> 1u;
    let i = base + index % count;
    let root = roots[i];
    if (index >= count) {
        return c32(-root.re, -root.im);
    }
    return root;
}

// fn load_in_tex(pos: vec2<u32>) -> c32 {
//     let real = textureLoad(src_tex, pos).x;
//     let c = c32(real, 0.0);
// }

fn swizzle(order: u32, index: u32) -> u32 {
    return reverseBits(index) >> (32u - order);
}

@compute
@workgroup_size(256, 1, 1)
fn fft(
    @builtin(global_invocation_id) global_index: vec3<u32>,
    @builtin(local_invocation_index) wg_index: u32,
) {
    let i = global_index.x;
    let j = global_index.y;

    let pos = vec2(i, j);
    let in_bounds = all(pos < uniforms.src_size + uniforms.src_padding) && all(pos >= uniforms.src_padding);
    let swizzled_index = swizzle(8u, i);
    if (in_bounds) {
        temp[swizzled_index] = load_real_c32_n(src_tex, pos - uniforms.src_padding);
    } else {
        temp[swizzled_index] = splat_c32_n(c32(0.0, 0.0));
    }
    workgroupBarrier();

    //handle first 8 iterations in fast workgroup memory
    for (var order = 0u; order < 8u; order++) {
        let subsection_count = 1u << order;
        let i_subsection = i % subsection_count;
        var offset: i32;
        if (i_subsection >= (subsection_count >> 1u)) {
            offset = i32(subsection_count);
        } else {
            offset = -i32(subsection_count);
        }

        let root = get_root(uniforms.orders, i_subsection * (uniforms.orders - order));
        temp[i] = fma_c32_n(temp[i], root, temp[u32(i32(i) + offset)]);
        workgroupBarrier();
    }

    // #ifdef CHANNELS
    // #if CHANNELS == 1
    // let packed = pack_c32(temp[i]);
    // textureStore(dst_tex, vec2(i, j), vec4(packed.value, 0u, 0u, 0u));
    // #else if CHANNELS == 2
    // let packed = pack_c32_2(temp[i]);
    // textureStore(dst_tex, vec2(i, j), vec4(packed.value, 0u, 0u));
    // #else if CHANNELS == 3
    // let packed = pack_c32_3(temp[i]);
    // textureStore(dst_tex, vec2(i, j), vec4(packed.value, 0u));
    // #else if CHANNELS == 4
    // let packed = pack_c32_4(temp[i]);
    // textureStore(dst_tex, vec2(i, j), vec4(packed.value));
    // #endif
    // #endif

    let packed = pack_c32_4(temp[i]);
    textureStore(dst_tex, vec2(i, j), vec4(packed.value));

    storageBarrier();

    for (var order = 8u; order < uniforms.orders; order++) {
        let subsection_count = 1u << order;
        let i_subsection = i % subsection_count;
        var offset: i32;
        if (i_subsection >= (subsection_count >> 1u)) {
            offset = i32(subsection_count);
        } else {
            offset = -i32(subsection_count);
        }

        let root = get_root(uniforms.orders, i_subsection * (uniforms.orders - order));
        let c_1 = load_c32_n(dst_tex, vec2(i, j));
        let c_2 = load_c32_n(dst_tex, vec2(u32(i32(i) + offset), j));
        let c_o = fma_c32_n(c_1, root, c_2);

        // #ifdef CHANNELS
        // #if CHANNELS == 1
        // let packed = pack_c32(c_o);
        // textureStore(dst_tex, vec2(i, j), vec4(packed.value, 0u, 0u, 0u));
        // #else if CHANNELS == 2
        // let packed = pack_c32_2(c_o);
        // textureStore(dst_tex, vec2(i, j), vec4(packed.value, 0u, 0u));
        // #else if CHANNELS == 3
        // let packed = pack_c32_3(c_o);
        // textureStore(dst_tex, vec2(i, j), vec4(packed.value, 0u));
        // #else if CHANNELS == 4
        // let packed = pack_c32_4(c_o);
        // textureStore(dst_tex, vec2(i, j), vec4(packed.value));
        // #endif
        // #endif

        let packed = pack_c32_4(c_o);
        textureStore(dst_tex, vec2(i, j), vec4(packed.value));

        storageBarrier();
    }
}
