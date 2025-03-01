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
        settings,
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
            input_value = read_buffer_c(pos);
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

    // Second phase: Butterfly operations in sequential order
    for (var order = 0u; order < 8u; order++) {
        let subsection_count = 1u << order;
        let half_subsection = subsection_count >> 1u;
        
        // Calculate butterfly pattern using sequential indices
        let subsection_index = sequential & (subsection_count - 1u);
        let offset_in_pair = subsection_index & (half_subsection - 1u);
        let is_second_half = (subsection_index & half_subsection) != 0u;
        
        let pair_index = sequential ^ half_subsection;
        
        let root = get_root(order, offset_in_pair);
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

    // Write results maintaining the bit-reversed ordering from the start
    c_o = temp[sequential];
    if (in_bounds) {
        #ifdef VERTICAL
            write_buffer_a(pos, c_o);
        #else
            write_buffer_b(pos, c_o);
        #endif
    }

    if (in_bounds) {
        let mag = c_o.re * c_o.re + c_o.im * c_o.im;
        let mag_normalized = log(1.0 + abs(mag)) * 0.1;
        let color = viridis_quintic(mag_normalized.x);
        
        // Ensure proper FFT shift by using modulo arithmetic
        #ifdef VERTICAL
            write_shifted_d_im(pos, vec4(color.xyz, 1.0));
        #else
            write_shifted_d_re(pos, vec4(color.xyz, 1.0));
        #endif
    }
}

fn viridis_quintic(x: f32) -> vec3<f32> {
    let x_clamped = clamp(x, 0.0, 1.0);
    let x1 = vec4(1.0, x_clamped, x_clamped * x_clamped, x_clamped * x_clamped * x_clamped);
    let x2 = x1 * x1.w * x_clamped;
    return vec3<f32>(
        dot(x1.xyzw, vec4<f32>(0.280268003, -0.143510503, 2.225793877, -14.815088879)) + 
            dot(x2.xy, vec2<f32>(25.212752309, -11.772589584)),
            
        dot(x1.xyzw, vec4<f32>(-0.002117546, 1.617109353, -1.909305070, 2.701152864)) + 
            dot(x2.xy, vec2<f32>(-1.685288385, 0.178738871)),
            
        dot(x1.xyzw, vec4<f32>(0.300805501, 2.614650302, -12.019139090, 28.933559110)) + 
            dot(x2.xy, vec2<f32>(-33.491294770, 13.762053843))
    );
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

// Unified window function with multiple options
fn apply_window(pos: vec2<u32>, size: vec2<u32>, window_type: u32, strength: f32) -> f32 {
    // Normalize coordinates
    let x_norm = f32(pos.x) / f32(size.x - 1u);
    let y_norm = f32(pos.y) / f32(size.y - 1u);
    
    var window_value = 1.0;
    
    // No window
    if (window_type == 0u) {
        window_value = 1.0;
    }
    // Tukey window (good for ocean waves - minimal edge effects with flat center)
    else if (window_type == 1u) {
        let alpha = 0.1; // Small alpha preserves most of the pattern
        let x_window = tukey_1d(x_norm, alpha);
        let y_window = tukey_1d(y_norm, alpha);
        window_value = x_window * y_window;
    }
    // Blackman window (good for bloom - smooth falloff)
    else if (window_type == 2u) {
        let a0 = 0.42;
        let a1 = 0.5;
        let a2 = 0.08;
        
        let x_window = a0 - a1 * cos(2.0 * 3.14159265359 * x_norm) + a2 * cos(4.0 * 3.14159265359 * x_norm);
        let y_window = a0 - a1 * cos(2.0 * 3.14159265359 * y_norm) + a2 * cos(4.0 * 3.14159265359 * y_norm);
        
        window_value = x_window * y_window;
    }
    // Kaiser window (good for precise frequency control)
    else if (window_type == 3u) {
        let beta = 2.0;
        let x_centered = 2.0 * x_norm - 1.0;
        let y_centered = 2.0 * y_norm - 1.0;
        let r_squared = x_centered * x_centered + y_centered * y_centered;
        
        if (r_squared < 1.0) {
            let term = 1.0 - r_squared;
            window_value = exp(beta * (sqrt(term) - 1.0));
        } else {
            window_value = 0.0;
        }
    }
    
    // Apply window strength (blend between no window and full window)
    return (1.0 - strength) + strength * window_value;
}

fn tukey_1d(x: f32, alpha: f32) -> f32 {
    let safe_alpha = max(0.0001, min(1.0, alpha));
    
    if (x < 0.0 || x > 1.0) {
        return 0.0;
    } else if (x < safe_alpha/2.0) {
        return 0.5 * (1.0 + cos(2.0 * 3.14159265359 * (x / safe_alpha - 0.5)));
    } else if (x > (1.0 - safe_alpha/2.0)) {
        return 0.5 * (1.0 + cos(2.0 * 3.14159265359 * (x / safe_alpha - 1.0/safe_alpha + 0.5)));
    } else {
        return 1.0;
    }
}
