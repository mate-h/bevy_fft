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
        roots_buffer
    },
    buffer::{
        read_buffer_a,
        read_buffer_b,
        read_buffer_c,
        write_buffer_a,
        write_buffer_b,
        write_shifted_d_re,
        write_shifted_d_im,
    },
    plot::{
        viridis_quintic,
        apply_window
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
        let pos = vec2(global_index.y, global_index.x);
    #else
        let pos = vec2(global_index.x, global_index.y);
    #endif
    
    let sequential = global_index.x;
    let bit_reversed = swizzle(8u, sequential);
    
    let in_bounds = all(pos < vec2(256u)) && all(pos >= vec2(0u));
    let c_zero = splat_c32_n(c32(0.0, 0.0));
    var c_o: c32_n;
    var input_value: c32_n;

    // Load initial data
    if (in_bounds) {
        #ifdef VERTICAL
            input_value = read_buffer_a(pos);
        #else
            input_value = read_buffer_c(pos);
        #endif
        temp[bit_reversed] = c32_n(input_value.re, -input_value.im);
    } else {
        temp[bit_reversed] = c_zero;
    }
    workgroupBarrier();

    // Second phase: Butterfly operations in sequential order
    for (var order = 0u; order < 8u; order++) {
        let subsection_count = 1u << order;
        let half_subsection = subsection_count >> 1u;
        
        // Calculate butterfly pattern using sequential indices
        let subsection_index = sequential & (subsection_count - 1u);
        let offset_in_pair = subsection_index & (half_subsection - 1u);
        let is_second_half = (subsection_index & half_subsection) != 0u;
        
        let pair_index = sequential ^ half_subsection;
        
        let root = conj_c32(get_root(order, offset_in_pair));
        let root_inverted = c32(-root.re, -root.im);
        
        let value_1 = temp[sequential];
        let value_2 = temp[pair_index];
        
        if (is_second_half) {
            c_o = fma_c32_n(value_1, root_inverted, value_2);
        } else {
            c_o = fma_c32_n(value_2, root, value_1);
        };

        workgroupBarrier();
        temp[sequential] = c_o;
        workgroupBarrier();
    }

    // Write results
    c_o = temp[sequential];
    c_o.re.w = 1.0;
    c_o.im.w = 1.0;
    let center = (pos - 128u) % 256u;
    if (in_bounds) {
        #ifdef VERTICAL
            // For vertical pass, write to buffer B
            write_buffer_b(center, c_o);
        #else
            // For horizontal pass, just write to buffer A for the vertical pass's input
            write_buffer_a(center, c_o);
        #endif
    }

    if (in_bounds) {
        // Scale by 1/N² for 2D IFFT (N×N points)
        let scale_factor = 1.0 / f32(1u << 16u);  // 1/(256*256)
        let scaled_c_o = mul_c32_n(c_o, splat_c32_n(c32(scale_factor, 0.0)));
        
        // Apply window function
        let window_type = 0u;
        let window_strength = 1.0;
        var window = apply_window(pos, vec2(256u), window_type, window_strength);
        let windowed_result = mul_c32_n(scaled_c_o, splat_c32_n(c32(window, 0.0)));
        
        // Visualize the result
        // let value = windowed_result.re;
        let normalized_value = log(1.0 + abs(windowed_result.re.x)) * 10.0;
        let color = viridis_quintic(normalized_value);

        // Write visualization output
        #ifdef VERTICAL
            write_shifted_d_im(pos, vec4(color.xyz, 1.0));
        #else
            write_shifted_d_re(pos, vec4(color.xyz, 1.0));
        #endif
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
    return root;
}

fn conj_c32(z: c32) -> c32 {
    return c32(z.re, -z.im);
}
