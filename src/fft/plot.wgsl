#define_import_path bevy_fft::plot

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
