#define_import_path bevy_fft::fft

#import bevy_fft::{
    complex::{
        splat_c32_n,
        fma_c32_n,
        // channel specific imports (aliasing doesn't work cross-modules)
        c32,
        c32_2,
        c32_3,
        c32_4,
    }, 
    bindings::{
        uniforms,
        roots_buffer,
        src_tex,
        dst_tex,
        re_tex,
        im_tex,
    },
    texel::{
        load_c32_n,
        store_c32_n,
        load_re_c32_n,
        store_re_c32_n,
        store_im_c32_n,
    }
};

#ifdef CHANNELS
#if CHANNELS == 1 
    alias c32_n = c32;
#else if CHANNELS == 2;
    alias c32_n = c32_2;
#else if CHANNELS == 3;
    alias c32_n = c32_3;
#else if CHANNELS == 4;
    alias c32_n = c32_4;
#endif
#endif

var<workgroup> temp: array<c32_n, 256>;

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
        temp[swizzled_index] = load_re_c32_n(src_tex, pos - uniforms.src_padding);
    } else {
        temp[swizzled_index] = splat_c32_n(c32(0.0, 0.0));
    }
    workgroupBarrier();

    // handle first 8 iterations in fast workgroup memory
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

    store_c32_n(vec2(i, j), temp[i]);
    store_re_c32_n(vec2(i, j), temp[i]);
    store_im_c32_n(vec2(i, j), temp[i]);
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
        let c_1 = load_c32_n(vec2(i, j));
        let c_2 = load_c32_n(vec2(u32(i32(i) + offset), j));
        let c_o = fma_c32_n(c_1, root, c_2);

        store_c32_n(vec2(i, j), c_o);
        store_re_c32_n(vec2(i, j), c_o);
        store_im_c32_n(vec2(i, j), c_o);
        storageBarrier();
    }
}

fn swizzle(order: u32, index: u32) -> u32 {
    return reverseBits(index) >> (32u - order);
}

fn get_root(order: u32, index: u32) -> c32 {
    let base = 1u << order;
    let count = base >> 1u;
    let i = base + index % count;
    let root = roots_buffer.roots[i];
    if (index >= count) {
        return c32(-root.re, -root.im);
    }
    return root;
}
