#define_import_path bevy_fft::fft

#import bevy_fft::{
    complex::{
        splat_c32_n,
        fma_c32_n,
        add_c32_n,
        mul_c32_n,
        // channel specific imports (aliasing doesn't work cross-modules)
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
        write_buffer_c,
        write_shifted_d_re,
        write_shifted_d_im
    },
    plot::{
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
// var<push_constant> iteration_override: u32;

@compute
@workgroup_size(256, 1, 1)
fn fft(
    @builtin(global_invocation_id) global_index: vec3<u32>
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

    // First phase: Bit reversal permutation
    if (in_bounds) {
        #ifdef VERTICAL
            input_value = read_buffer_b(pos);
        #else
            input_value = read_buffer_a(pos);
        #endif
        
        // Apply window function
        let window_type = 0u;
        let window_strength = 1.0;
        var window = apply_window(pos, vec2(256u), window_type, window_strength);
        input_value = mul_c32_n(input_value, splat_c32_n(c32(window, 0.0)));
        
        temp[bit_reversed] = input_value;
    } else {
        temp[bit_reversed] = c_zero;
    }
    workgroupBarrier();

    // Radix-2 DIT: stride half = 1,2,...,N/2 (eight stages for N=256). Twiddle table uses
    // base 2^(s+1) per stage s → pass (s + 1) into get_root.
    for (var order = 0u; order < 8u; order++) {
        let half_subsection = 1u << order;
        let subsection_count = half_subsection << 1u;

        let subsection_index = sequential & (subsection_count - 1u);
        let offset_in_pair = subsection_index & (half_subsection - 1u);
        let is_second_half = (subsection_index & half_subsection) != 0u;

        let pair_index = sequential ^ half_subsection;

        let root = get_root(order + 1u, offset_in_pair);
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

    // Same index layout for both 1D passes (no fftshift in-between); see ifft.wgsl.
    c_o = temp[sequential];
    let fft_out_coord = pos;
    if (in_bounds) {
        // Opaque alpha on real/imag textures for sprite blending; per-channel butterflies unchanged
        // for RGB. Same for im.w so *_im buffers are not fully transparent.
        c_o.re.w = 1.0;
        c_o.im.w = 1.0;
        #ifdef VERTICAL
            write_buffer_c(fft_out_coord, c_o);
        #else
            write_buffer_b(fft_out_coord, c_o);
        #endif
    }

    // Raw real/imag RGBA to D (no display scaling).
    if (in_bounds) {
        write_shifted_d_re(fft_out_coord, c_o.re);
        write_shifted_d_im(fft_out_coord, c_o.im);
    }
}

fn swizzle(order: u32, index: u32) -> u32 {
    return reverseBits(index) >> (32u - order);
}

fn get_root(order: u32, index: u32) -> c32 {
    let base = 1u << order;
    let count = max(1u, base >> 1u);
    let i = base + index % count;
    return roots_buffer.roots[i];
}

