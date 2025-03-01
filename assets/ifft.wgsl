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

    // Load initial data
    if (in_bounds) {
        // Read from buffer A initially and conjugate for IFFT
        let n = read_buffer_c(pos);
        temp[bit_reversed] = c32_n(n.re, -n.im);
    } else {
        temp[bit_reversed] = c_zero;
    }
    workgroupBarrier();

    // First phase: Butterfly operations in sequential order (similar to FFT)
    for (var order = 0u; order < 8u; order++) {
        let subsection_count = 1u << order;
        let half_subsection = subsection_count >> 1u;
        
        // Calculate butterfly pattern using sequential indices
        let subsection_index = sequential & (subsection_count - 1u);
        let offset_in_pair = subsection_index & (half_subsection - 1u);
        let is_second_half = (subsection_index & half_subsection) != 0u;
        
        let pair_index = sequential ^ half_subsection;
        
        // Use conjugate of root for IFFT
        let root = conj_c32(get_root(order, offset_in_pair));
        let root_conj = c32(-root.re, -root.im);
        
        let value_1 = temp[sequential];
        let value_2 = temp[pair_index];
        
        if (is_second_half) {
            c_o = fma_c32_n(value_1, root_conj, value_2);
        } else {
            c_o = fma_c32_n(value_2, root, value_1);
        };

        workgroupBarrier();
        temp[sequential] = c_o;
        workgroupBarrier();
    }

    // Write results to buffer B
    c_o = temp[sequential];
    if (in_bounds) {
        write_buffer_b(pos, c_o);
    }
    
    storageBarrier();

    // Global memory iterations
    var using_buffer_a = true;
    for (var order = 8u; order < settings.orders; order++) {
        let subsection_count = 1u << order;
        let half_subsection = subsection_count >> 1u;
        
        #ifdef VERTICAL
            let i_subsection = sequential % subsection_count;
        #else
            let i_subsection = sequential % subsection_count;
        #endif
        
        let offset_in_pair = i_subsection & (half_subsection - 1u);
        let is_second_half = (i_subsection & half_subsection) != 0u;
        
        // Use conjugate of root for IFFT
        let root = conj_c32(get_root(order, offset_in_pair));
        let root_conj = c32(-root.re, -root.im);
        
        let o_1 = pos;
        
        var o_2: vec2<u32>;
        #ifdef VERTICAL
            if (is_second_half) {
                o_2 = vec2(pos.x, pos.y - half_subsection);
            } else {
                o_2 = vec2(pos.x, pos.y + half_subsection);
            }
        #else
            if (is_second_half) {
                o_2 = vec2(pos.x - half_subsection, pos.y);
            } else {
                o_2 = vec2(pos.x + half_subsection, pos.y);
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
        
        if (is_second_half) {
            c_o = fma_c32_n(c_1, root_conj, c_2);
        } else {
            c_o = fma_c32_n(c_2, root, c_1);
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
    }
    
    // Final output with scaling and visualization
    if (in_bounds) {
        // Scale by 1/N for IFFT
        let scale_factor = 1.0 / f32(1u << settings.orders);
        let scaled_c_o = mul_c32_n(c_o, splat_c32_n(c32(scale_factor, 0.0)));
        
        // Apply window function
        let window_type = 0u;
        let window_strength = 1.0;
        var window = apply_window(pos, settings.size + settings.padding, window_type, window_strength);
        let windowed_result = mul_c32_n(scaled_c_o, splat_c32_n(c32(window, 0.0)));
        
        // Visualize the result
        let mag = windowed_result.re * windowed_result.re + windowed_result.im * windowed_result.im;
        let mag_normalized = log(1.0 + abs(mag)) * 0.1;
        let color = viridis_quintic(mag_normalized.x);
        
        // Calculate centered position for visualization
        let center = settings.size / 2u;
        let centered_pos = (pos - center) % settings.size;
        
        // Write visualization output
        #ifdef VERTICAL
            write_shifted_d_im(centered_pos, vec4(color.xyz, 1.0));
        #else
            write_shifted_d_re(centered_pos, vec4(color.xyz, 1.0));
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
    if (index >= count) {
        return c32(-root.re, -root.im);
    }
    return root;
}

fn conj_c32(z: c32) -> c32 {
    return c32(z.re, -z.im);
}
