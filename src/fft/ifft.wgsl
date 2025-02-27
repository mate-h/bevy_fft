#define_import_path bevy_fft::ifft

#import bevy_fft::{
    complex::{
        splat_c32_n,
        fma_c32_n,
        mul_c32_n,
        add_c32_n,
        // channel specific imports (aliasing doesn't work cross-modules)
        c32,
        c32_2,
        c32_3,
        c32_4,
    }, 
    bindings::{
        settings,
        roots_buffer,
    },
    buffer::{
        read_buffer,
        write_buffer,
        write_output,
    },
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
var<push_constant> iteration_override: u32;

@compute
@workgroup_size(256, 1, 1)
fn ifft(
    @builtin(global_invocation_id) global_index: vec3<u32>,
    @builtin(local_invocation_index) wg_index: u32,
) {
    #ifdef VERTICAL
        // swap x and y
        let i = global_index.y;
        let j = global_index.x;
    #else
        let i = global_index.x;
        let j = global_index.y; 
    #endif

    let pos = vec2(i, j);
    let in_bounds = all(pos < settings.src_size + settings.src_padding) && all(pos >= settings.src_padding);
    let swizzled_index = swizzle(8u, i);

    // Initial load - special case for first iteration
    if (iteration_override == 0u) {
        // First pass: load from input texture
        if (in_bounds) {
            let n = read_buffer(pos, 0u);
            // Conjugate for IFFT
            temp[swizzled_index] = c32_n(n.re, -n.im);
        } else {
            temp[swizzled_index] = splat_c32_n(c32(0.0, 0.0));
        }
    } else {
        // Subsequent passes: load from ping-pong buffer
        temp[swizzled_index] = read_buffer(pos, iteration_override);
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

        // Use conjugate of root for IFFT
        let root = conj_c32(get_root(settings.orders, i_subsection * (settings.orders - order)));
        temp[i] = fma_c32_n(temp[i], root, temp[u32(i32(i) + offset)]);
        workgroupBarrier();
    }

    // Store intermediate results to the appropriate buffer
    let result = temp[i];
    write_buffer(vec2(i, j), result, iteration_override);
    storageBarrier();

    for (var order = 8u; order < settings.orders; order++) {
        let subsection_count = 1u << order;
        let i_subsection = i % subsection_count;
        var offset: i32;
        if (i_subsection >= (subsection_count >> 1u)) {
            offset = i32(subsection_count);
        } else {
            offset = -i32(subsection_count);
        }

        // Use conjugate of root for IFFT
        let root = conj_c32(get_root(settings.orders, i_subsection * (settings.orders - order)));
        let o_1 = vec2(i, j);
        let o_2 = vec2(u32(i32(i) + offset), j);
        
        // Calculate the current iteration based on the order
        let current_iteration = iteration_override + (order - 8u);
        
        // Read from current buffer
        let c_1 = read_buffer(o_1, current_iteration);
        let c_2 = read_buffer(o_2, current_iteration);
        
        let c_o = fma_c32_n(c_1, root, c_2);
        
        // For the final iteration, apply scaling and write to output
        if (order == settings.orders - 1u) {
            // For IFFT, we need to scale by 1/N
            let scale_factor = 1.0 / f32(1u << settings.orders);
            let scaled_c_o = mul_c32_n(c_o, splat_c32_n(c32(scale_factor, 0.0)));
            
            // Write to final output textures
            if (in_bounds) {
                write_output(pos - settings.src_padding, scaled_c_o);
            }
        } else {
            // Write to ping-pong buffer for intermediate results
            write_buffer(vec2(i, j), c_o, current_iteration);
        }
        
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

fn conj_c32(z: c32) -> c32 {
    return c32(z.re, -z.im);
}
