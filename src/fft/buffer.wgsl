#define_import_path bevy_fft::buffer

#import bevy_fft::{
    complex::{
        splat_c32_n,
        c32,
        c32_2,
        c32_3,
        c32_4,
    },
    bindings::{
        settings,
        buffer_a_re,
        buffer_a_im,
        buffer_b_re,
        buffer_b_im,
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

// Input/output bindings in group 1
#ifdef VERTICAL
@group(1) @binding(0) var src_re_tex: texture_storage_2d<rgba32float, read_write>;
@group(1) @binding(1) var src_im_tex: texture_storage_2d<rgba32float, read_write>;
#else
@group(1) @binding(0) var src_re_tex: texture_2d<f32>;
@group(1) @binding(1) var src_im_tex: texture_2d<f32>;
#endif
@group(1) @binding(2) var dst_re_tex: texture_storage_2d<rgba32float, write>;
@group(1) @binding(3) var dst_im_tex: texture_storage_2d<rgba32float, write>;

// Helper functions that take an explicit iteration parameter
fn read_buffer(pos: vec2<u32>, iter: u32) -> c32_n {
    #ifdef HORIZONTAL
    // For horizontal pass
    if (iter == 0u) {
        // Initial load from input textures (group 1)
        let tex_pos = vec2<i32>(i32(pos.x), i32(pos.y));
        
        // Check if the position is within bounds
        if (pos.x < settings.src_size.x && pos.y < settings.src_size.y) {
            // Use textureLoad for sampled textures - direct texel access
            let re = textureLoad(src_re_tex, tex_pos, 0);
            let im = textureLoad(src_im_tex, tex_pos, 0);
            
            return c32_n(re, im);
        } else {
            // Out of bounds, return zero
            return splat_c32_n(c32(0.0, 0.0));
        }
    } else if (iter % 2u == 0u) {
        // Even iterations: read from buffer A
        return c32_n(
            textureLoad(buffer_a_re, pos),
            textureLoad(buffer_a_im, pos)
        );
    } else {
        // Odd iterations: read from buffer B
        return c32_n(
            textureLoad(buffer_b_re, pos),
            textureLoad(buffer_b_im, pos)
        );
    }
    #else
    // For vertical pass
    if (iter == 0u) {
        // Initial load from input textures (group 1)
        return c32_n(
            textureLoad(src_re_tex, pos),
            textureLoad(src_im_tex, pos)
        );
    } else if (iter % 2u == 0u) {
        // Even iterations: read from buffer A
        return c32_n(
            textureLoad(buffer_a_re, pos),
            textureLoad(buffer_a_im, pos)
        );
    } else {
        // Odd iterations: read from buffer B
        return c32_n(
            textureLoad(buffer_b_re, pos),
            textureLoad(buffer_b_im, pos)
        );
    }
    #endif
}

fn write_buffer(pos: vec2<u32>, value: c32_n, iter: u32) {
    if (iter % 2u == 0u) {
        // Even iterations: write to buffer B
        textureStore(buffer_b_re, pos, vec4<f32>(value.re.xyz, 1.0));
        textureStore(buffer_b_im, pos, vec4<f32>(value.im.xyz, 1.0));
    } else {
        // Odd iterations: write to buffer A
        textureStore(buffer_a_re, pos, vec4<f32>(value.re.xyz, 1.0));
        textureStore(buffer_a_im, pos, vec4<f32>(value.im.xyz, 1.0));
    }
}

// Add a new function to write the final output
fn write_output(pos: vec2<u32>, value: c32_n) {
    // Write to the output textures (group 1) with alpha = 1.0
    textureStore(dst_re_tex, pos, vec4<f32>(value.re.xyz, 1.0));
    textureStore(dst_im_tex, pos, vec4<f32>(value.im.xyz, 1.0));
}