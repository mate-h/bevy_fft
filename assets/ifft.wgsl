#define_import_path bevy_fft::ifft

#import bevy_fft::{
    complex::{
        splat_c32_n,
        fma_c32_n,
        mul_c32_n,
        add_c32_n,
        conj_c32_n,
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

    // x = (1/N) conj(FFT(conj(X))); butterfly output is FFT(conj(X)), so conj before scale.
    c_o = conj_c32_n(temp[sequential]);
    let inv_dim = 1.0 / 256.0;
    c_o = mul_c32_n(c_o, splat_c32_n(c32(inv_dim, 0.0)));
    // Keep lane 3 (`.w`) so a fourth IFFT band (e.g. ocean chop) survives into `spatial_output.a`.
    // No fftshift between 1D passes: 2D IFFT must be ifft_y(ifft_x(C)) on the same index
    // layout as common FFT stacks (e.g. NumPy). Per-axis shift between passes breaks separability and
    // collapses diagonal spectra into line artifacts.
    let fft_out_coord = pos;
    if (in_bounds) {
        #ifdef VERTICAL
            write_buffer_b(fft_out_coord, c_o);
        #else
            write_buffer_a(fft_out_coord, c_o);
        #endif
    }

    // D textures: always write both each pass so the last IFFT vertical leaves d_re = final
    // spatial real (what buffer_b_re holds) and d_im = residual imaginary for debug.
    // Previously horizontal wrote only d_re and vertical only d_im, so d_re stayed stale
    // (e.g. per-row IFFT imag) and never showed the full 2D output real part.
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

