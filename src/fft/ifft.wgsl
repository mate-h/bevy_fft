#define_import_path bevy_fft::ifft

#import bevy_fft::{
    complex::{
        splat_c32_n,
        fma_c32_n,
        mul_c32_n,
        add_c32_n,
        c32,
        c32_2,
        c32_3,
        c32_4,
    }, 
    bindings::{
        settings,
        roots_buffer,
        buffer_a_re,
        buffer_a_im,
        buffer_b_re,
        buffer_b_im,
    },
    buffer::{
        read_buffer_a,
        read_buffer_b,
        write_buffer_a,
        write_buffer_b,
        // write_shifted_d,
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
fn ifft(
    @builtin(global_invocation_id) global_index: vec3<u32>,
    @builtin(local_invocation_index) wg_index: u32,
) {
    #ifdef VERTICAL
        let i = global_index.y;
        let j = global_index.x;
        let access_index = wg_index;
        let global_index_for_calc = j;
    #else
        let i = global_index.x;
        let j = global_index.y;
        let workgroup_base = (i / 256u) * 256u;
        let local_index = i % 256u;
        let access_index = swizzle(8u, local_index);
        let global_index_for_calc = workgroup_base + local_index;
    #endif

    let pos = vec2(i, j);
    let in_bounds = all(pos < settings.size + settings.padding) && all(pos >= settings.padding);

    // Load initial data
    if (in_bounds) {
        // Read from buffer A initially and conjugate for IFFT
        let n = read_buffer_a(pos);
        temp[access_index] = c32_n(n.re, -n.im);
    } else {
        temp[access_index] = splat_c32_n(c32(0.0, 0.0));
    }
    workgroupBarrier();

    // First 8 iterations in workgroup memory
    for (var order = 0u; order < 8u; order++) {
        let subsection_count = 1u << order;
        let half_subsection = subsection_count >> 1u;
        
        #ifdef VERTICAL
            let i_subsection = j;
        #else
            let i_subsection = global_index_for_calc;
        #endif

        let pair_index = i_subsection >> order;
        let offset_in_pair = i_subsection & (half_subsection - 1u);
        let is_second_half = (i_subsection & half_subsection) != 0u;
        
        var offset: i32;
        if (is_second_half) {
            offset = -i32(half_subsection);
        } else {
            offset = i32(half_subsection);
        }

        // Use conjugate of root for IFFT
        let root = conj_c32(get_root(order, offset_in_pair));
        
        let value_1 = temp[access_index];
        let value_2 = temp[u32(i32(access_index) + offset)];
        
        var new_value: c32_n;
        if (is_second_half) {
            new_value = add_c32_n(value_1, fma_c32_n(value_2, c32(-root.re, -root.im), splat_c32_n(c32(0.0, 0.0))));
        } else {
            new_value = add_c32_n(value_1, fma_c32_n(value_2, root, splat_c32_n(c32(0.0, 0.0))));
        }

        workgroupBarrier();
        temp[access_index] = new_value;
        workgroupBarrier();
    }

    // Write results to buffer B
    let result = temp[access_index];
    if (in_bounds) {
        write_buffer_b(pos, result);
    }
    
    storageBarrier();

    // Global memory iterations
    var using_buffer_a = true;
    for (var order = 8u; order < settings.orders; order++) {
        let subsection_count = 1u << order;
        #ifdef VERTICAL
            let i_subsection = global_index_for_calc % subsection_count;
        #else
            let i_subsection = global_index_for_calc % subsection_count;
        #endif

        let half_subsection = subsection_count >> 1u;
        let pair_offset = i_subsection % half_subsection;
        let is_second_half = i_subsection >= half_subsection;
        
        // Use conjugate of root for IFFT
        let root = conj_c32(get_root(order, pair_offset));
        let o_1 = vec2(i, j);
        
        var o_2: vec2<u32>;
        #ifdef VERTICAL
            if (is_second_half) {
                o_2 = vec2(i, j + half_subsection);
            } else {
                o_2 = vec2(i, j - half_subsection);
            }
        #else
            if (is_second_half) {
                o_2 = vec2(i + half_subsection, j);
            } else {
                o_2 = vec2(i - half_subsection, j);
            }
        #endif
        
        var c_1: c32_n;
        var c_2: c32_n;
        if (using_buffer_a) {
            c_1 = read_buffer_a(o_1);
            c_2 = read_buffer_a(o_2);
        } else {
            c_1 = read_buffer_b(o_1);
            c_2 = read_buffer_b(o_2);
        }
        
        var c_o: c32_n;
        if (is_second_half) {
            c_o = add_c32_n(c_1, fma_c32_n(c_2, c32(-root.re, -root.im), splat_c32_n(c32(0.0, 0.0))));
        } else {
            c_o = add_c32_n(c_1, fma_c32_n(c_2, root, splat_c32_n(c32(0.0, 0.0))));
        }
        
        storageBarrier();
        
        if (in_bounds) {
            if (using_buffer_a) {
                write_buffer_b(o_1, c_o);
            } else {
                write_buffer_a(o_1, c_o);
            }
        }
        
        storageBarrier();
        using_buffer_a = !using_buffer_a;
        
        // Final output with scaling
        if (order == settings.orders - 1u && in_bounds) {
            // Scale by 1/N for IFFT
            let scale_factor = 1.0 / f32(1u << settings.orders);
            let scaled_c_o = mul_c32_n(c_o, splat_c32_n(c32(scale_factor, 0.0)));
            
            // write_shifted_d(pos - settings.padding, scaled_c_o);
        }
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

fn conj_c32(z: c32) -> c32 {
    return c32(z.re, -z.im);
}
