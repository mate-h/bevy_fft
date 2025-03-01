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
        buffer_c_re,
        buffer_c_im,
        buffer_d_re,
        buffer_d_im,
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

// Helper functions that take an explicit iteration parameter
fn read_buffer_a(pos: vec2<u32>) -> c32_n {
    return c32_n(
        textureLoad(buffer_a_re, pos),
        textureLoad(buffer_a_im, pos)
    );
}

fn read_buffer_b(pos: vec2<u32>) -> c32_n {
    return c32_n(
        textureLoad(buffer_b_re, pos),
        textureLoad(buffer_b_im, pos)
    );
}

fn read_buffer_c(pos: vec2<u32>) -> c32_n {
    return c32_n(
        textureLoad(buffer_c_re, pos),
        textureLoad(buffer_c_im, pos)
    );
}

fn read_buffer_d(pos: vec2<u32>) -> c32_n {
    return c32_n(
        textureLoad(buffer_d_re, pos),
        textureLoad(buffer_d_im, pos)
    );
}

fn write_buffer_a(pos: vec2<u32>, value: c32_n) {
    textureStore(buffer_a_re, pos, value.re);
    textureStore(buffer_a_im, pos, value.im);
}

fn write_buffer_b(pos: vec2<u32>, value: c32_n) {
    textureStore(buffer_b_re, pos, value.re);
    textureStore(buffer_b_im, pos, value.im);
}

fn write_buffer_c(pos: vec2<u32>, value: c32_n) {
    textureStore(buffer_c_re, pos, value.re);
    textureStore(buffer_c_im, pos, value.im);
}

fn write_buffer_d(pos: vec2<u32>, value: c32_n) {
    textureStore(buffer_d_re, pos, value.re);
    textureStore(buffer_d_im, pos, value.im);
}

// Add this helper function at the top level
fn shift_coords(pos: vec2<u32>, size: vec2<u32>) -> vec2<u32> {
    let half_x = size.x >> 1u;  // divide by 2
    let half_y = size.y >> 1u;
    
    // Shift each coordinate by half the size
    let shifted_x = (pos.x + half_x) % size.x;
    let shifted_y = (pos.y + half_y) % size.y;
    
    return vec2(shifted_x, shifted_y);
}

fn write_shifted_d_re(pos: vec2<u32>, value: vec4<f32>) {
    let shifted_pos = shift_coords(pos + 0u, vec2(256u, 256u));
    textureStore(buffer_d_re, pos, value);
}

fn write_shifted_d_im(pos: vec2<u32>, value: vec4<f32>) {
    let shifted_pos = shift_coords(pos + 0u, vec2(256u, 256u));
    textureStore(buffer_d_im, pos, value);
}
