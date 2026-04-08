#define_import_path bevy_fft::ifft

#import bevy_fft::{
    complex::{
        conj_c32_n,
        c32,
        c32_2,
        c32_3,
        c32_4,
    },
    bindings::{
        settings,
    },
    buffer::{
        read_buffer_a,
        read_buffer_b,
        read_buffer_c,
        write_buffer_a,
        write_buffer_b,
        write_buffer_c,
    },
    fft_common::{
        fft_reverse_lower_bits,
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

@compute @workgroup_size(8, 8, 1)
fn ifft_br_horizontal(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = gid.xy;
    let dims = settings.size;
    if (p.x >= dims.x || p.y >= dims.y) {
        return;
    }
    let order = settings.orders;
    let rx = fft_reverse_lower_bits(p.x, order);
    let v = read_buffer_c(p);
    let cv = conj_c32_n(v);
    let dst = vec2<u32>(rx, p.y);
    write_buffer_a(dst, cv);
}

@compute @workgroup_size(8, 8, 1)
fn ifft_br_vertical(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = gid.xy;
    let dims = settings.size;
    if (p.x >= dims.x || p.y >= dims.y) {
        return;
    }
    let order = settings.orders;
    let ry = fft_reverse_lower_bits(p.y, order);
    let v = read_buffer_a(p);
    let cv = conj_c32_n(v);
    let dst = vec2<u32>(p.x, ry);
    write_buffer_b(dst, cv);
}
