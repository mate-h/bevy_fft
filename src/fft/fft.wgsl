
#import bevy_fft::c32::c32;

#if CHANNELS == 1 

#import bevy_fft::c32::{
    c32 as c32_n,
    load_c32 as load_c32_n,
    load_real_c32 as load_real_c32,
    store_c32 as store_c32_n,
    fma_c32 as fma_c32_n,
}

@group(0) @binding(3) var freq_tex: texture_storage_2d<r32uint>;

#else if CHANNELS == 2

#import bevy_fft::c32::{
    c32_2 as c32_n,
    load_c32_2 as load_c32_n,
    load_real_c32_2 as load_real_c32_n,
    store_c32_2 as store_c32_n,
    fma_c32_2 as fma_c32_n
}

@group(0) @binding(3) var freq_tex: texture_storage_2d<rg32uint>;

#else if CHANNELS == 3

#import bevy_fft::c32::{
    c32_3 as c32_n,
    load_c32_3 as load_c32_n,
    load_real_c32_3 as load_real_c32_n,
    store_c32_3 as store_c32_n,
    fma_c32_3 as fma_c32_n,
}

@group(0) @binding(3) var freq_tex: texture_storage_2d<rgba32uint>;

#else if CHANNELS == 4

#import bevy_fft::c32::{
    c32_4 as c32_n,
    load_c32_4 as load_c32_n,
    load_real_c32_4 as load_real_c32_n,
    store_c32_4 as store_c32_n,
    fma_c32_4 as fma_c32_n,
}

@group(0) @binding(3) var freq_tex: texture_storage_2d<rgba32uint>;

#endif

struct FftUniform {
    src_size: vec2<u32>,
    src_padding: vec2<u32>,
    orders: u32,
}

@group(0) @binding(0) var<uniform> uniform: FftUniform;
@group(0) @binding(1) var<uniform> roots: array<c32, 8192>;
@group(0) @binding(2) var in_tex: texture_2d<u32>;

var<workgroup> temp: array<c32_n, 256>;

fn get_root(order: u32, index: u32) -> c32 {
    let base = 1u << n;
    let count = base >> 1u;
    let i = base + index % count;
    let root = roots[i];
    return select(root, c32(-root.real, -root.imag), index >= count);
}

fn load_in_tex(pos: vec2<u32>) -> c32 {
    let real = textureLoad(in_tex, pos, 0u).x;
    let c = c32(real, 0.0);
}

fn swizzle(order: u32, index: u32) -> u32 {
    return reverseBits(global_index.x) >> (32u - order);
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
    temp[swizzle(8, i)] = select(c32(0.0, 0.0), load_real_c32_n(in_tex, pos - uniforms.src_padding), in_bounds);
    workgroupBarrier()

    //handle first 8 iterations in fast workgroup memory
    for (var order = 0u; order < 8u; i++) {
        let subsection_count = 1u << order;
        let i_subsection = i % subsection_count;
        let offset = select(-subsection_count, subsection_count, i_subsection >= (subsection_count >> 1u));

        let root = get_root(max_order, i_subsection * (uniforms.orders - order));
        temp[i] = fma_c32_n(temp[i], root, temp[i + offset]);
        workgroupBarrier();
    }

    store_c32_n(freq_tex, vec2(i, j), temp[i]);
    storageBarrier();

    for (var order = 8u; order < uniforms.orders; i++) {
        let subsection_count = 1u << order;
        let i_subsection = i % subsection_count;
        let offset = select(-subsection_count, subsection_count, i_subsection >= (subsection_count >> 1u));

        let root = get_root(max_order, i_subsection * (uniforms.orders - order));
        let c_1 = load_c32_n(freq_tex, vec2(i, j));
        let c_2 = load_c32_n(freq_tex, vec2(i + offset, j));
        let c_o = fma_c32_n(c1, root, c_2);
        store_c32_n(freq_tex, vec2(i, j), c_o);

        storageBarrier();
    }
}
