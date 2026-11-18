import bevy_fft::complex::c32::{
    splat_c32_n,
    add_c32_n,
    mul_c32_n,
    conj_c32_n,
    neg_c32_n,
    c32,
    c32_2,
    c32_3,
    c32_4,
};
import bevy_fft::fft::bindings::settings;
import bevy_fft::fft::buffer::{
    read_buffer_a,
    read_buffer_b,
    read_buffer_c,
    write_buffer_a,
    write_buffer_b,
    write_buffer_c,
    write_shifted_d_re,
    write_shifted_d_im,
};
import bevy_fft::fft::plot::apply_window;
import bevy_fft::fft::fft_common::{
    fft_reverse_lower_bits,
    get_fft_root,
};

alias c32_n = c32_4;

struct FftPushConstants {
    stage: u32,
    axis: u32,
    src_buffer: u32,
    dst_buffer: u32,
    flags: u32,
}

var<immediate> pc: FftPushConstants;

const FLAG_INVERSE_FINALIZE: u32 = 1u;
const FLAG_FORWARD_ALPHA: u32 = 2u;
const WG: u32 = 256u;

fn read_fft_buf(buf_id: u32, pos: vec2<u32>) -> c32_n {
    switch buf_id {
        case 0u: { return read_buffer_a(pos); }
        case 1u: { return read_buffer_b(pos); }
        case 2u: { return read_buffer_c(pos); }
        default: { return splat_c32_n(c32(0.0, 0.0)); }
    }
}

fn write_fft_buf(buf_id: u32, pos: vec2<u32>, value: c32_n) {
    switch buf_id {
        case 0u: { write_buffer_a(pos, value); }
        case 1u: { write_buffer_b(pos, value); }
        case 2u: { write_buffer_c(pos, value); }
        default: { }
    }
}

fn mark_opaque_alpha(vs: c32_n) -> c32_n {
    var out = vs;
    out.re.w = 1.0;
    out.im.w = 1.0;
    return out;
}

fn dit_butterfly_writes(pos_u: vec2<u32>, pos_v: vec2<u32>, j: u32) {
    let N = settings.size.x;
    let inv_scale = 1.0 / f32(N);
    let root = get_fft_root(pc.stage + 1u, j);

    let a = read_fft_buf(pc.src_buffer, pos_u);
    let b = mul_c32_n(read_fft_buf(pc.src_buffer, pos_v), splat_c32_n(root));
    var out_u = add_c32_n(a, b);
    var out_v = add_c32_n(a, neg_c32_n(b));

    if ((pc.flags & FLAG_INVERSE_FINALIZE) != 0u) {
        out_u = mul_c32_n(conj_c32_n(out_u), splat_c32_n(c32(inv_scale, 0.0)));
        out_v = mul_c32_n(conj_c32_n(out_v), splat_c32_n(c32(inv_scale, 0.0)));
    }

    if ((pc.flags & FLAG_FORWARD_ALPHA) != 0u) {
        out_u = mark_opaque_alpha(out_u);
        out_v = mark_opaque_alpha(out_v);
    }

    write_fft_buf(pc.dst_buffer, pos_u, out_u);
    write_fft_buf(pc.dst_buffer, pos_v, out_v);

    if ((pc.flags & FLAG_FORWARD_ALPHA) != 0u) {
        write_shifted_d_re(pos_u, out_u.re);
        write_shifted_d_im(pos_u, out_u.im);
        write_shifted_d_re(pos_v, out_v.re);
        write_shifted_d_im(pos_v, out_v.im);
    }
}

/// Bit-reverse permute along rows, optional Tukey window on spatial data, natural order in X.
@compute @workgroup_size(8, 8, 1)
fn fft_forward_br_horizontal(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = gid.xy;
    let dims = settings.size;
    if (p.x >= dims.x || p.y >= dims.y) {
        return;
    }
    let order = settings.orders;
    let rx = fft_reverse_lower_bits(p.x, order);
    var v = read_buffer_a(p);
    let window_type = settings.window_type;
    let window_strength = settings.window_strength;
    let w = apply_window(p, dims, window_type, window_strength);
    v = mul_c32_n(v, splat_c32_n(c32(w, 0.0)));
    v = mark_opaque_alpha(v);
    let dst = vec2<u32>(rx, p.y);
    write_buffer_b(dst, v);
    write_shifted_d_re(p, v.re);
    write_shifted_d_im(p, v.im);
}

/// Bit-reverse along columns after row FFT; reads B, writes C.
@compute @workgroup_size(8, 8, 1)
fn fft_forward_br_vertical(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = gid.xy;
    let dims = settings.size;
    if (p.x >= dims.x || p.y >= dims.y) {
        return;
    }
    let order = settings.orders;
    let ry = fft_reverse_lower_bits(p.y, order);
    var v = read_buffer_b(p);
    v = mark_opaque_alpha(v);
    let dst = vec2<u32>(p.x, ry);
    write_buffer_c(dst, v);
    write_shifted_d_re(p, v.re);
    write_shifted_d_im(p, v.im);
}

/// One radix-2 DIT stage; horizontal axis if pc.axis == 0, else vertical.
@compute @workgroup_size(256, 1, 1)
fn fft_radix2_dit(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = settings.size;
    let N = dims.x;
    let half_n = N >> 1u;
    let butterfly = gid.x;
    let line = gid.y;
    if (butterfly >= half_n || line >= N) {
        return;
    }

    let m = 1u << (pc.stage + 1u);
    let m2 = m >> 1u;
    let group = butterfly / m2;
    let j = butterfly % m2;
    let k = group * m;
    let u = k + j;
    let v = u + m2;

    if (pc.axis == 0u) {
        let pos_u = vec2<u32>(u, line);
        let pos_v = vec2<u32>(v, line);
        dit_butterfly_writes(pos_u, pos_v, j);
    } else {
        let pos_u = vec2<u32>(line, u);
        let pos_v = vec2<u32>(line, v);
        dit_butterfly_writes(pos_u, pos_v, j);
    }
}

/// Longest line `fft_dit_line` transforms: its real and imaginary lanes fill 16 KiB of workgroup
/// memory. Longer lines take one `fft_radix2_dit` dispatch per stage.
const LINE_MAX: u32 = 512u;
var<workgroup> line_re: array<vec4<f32>, LINE_MAX>;
var<workgroup> line_im: array<vec4<f32>, LINE_MAX>;

fn line_pos(i: u32, line: u32) -> vec2<u32> {
    return select(vec2<u32>(line, i), vec2<u32>(i, line), pc.axis == 0u);
}

/// Every radix-2 DIT stage of one bit-reversed line, in workgroup memory, in place in
/// `pc.src_buffer`: one dispatch and one read and write per element, where `fft_radix2_dit` makes
/// a dispatch and a round trip through the buffers per stage. The butterflies and their order
/// match it, so the results are the same.
@compute @workgroup_size(256, 1, 1)
fn fft_dit_line(
    @builtin(local_invocation_index) lid: u32,
    @builtin(workgroup_id) wg: vec3<u32>,
) {
    let N = settings.size.x;
    let line = wg.y;
    for (var i = lid; i < N; i += WG) {
        let v = read_fft_buf(pc.src_buffer, line_pos(i, line));
        line_re[i] = v.re;
        line_im[i] = v.im;
    }
    workgroupBarrier();
    let half_n = N >> 1u;
    for (var stage = 0u; stage < settings.orders; stage++) {
        let m2 = 1u << stage;
        for (var butterfly = lid; butterfly < half_n; butterfly += WG) {
            let j = butterfly & (m2 - 1u);
            let u = ((butterfly >> stage) << (stage + 1u)) + j;
            let v = u + m2;
            let root = get_fft_root(stage + 1u, j);
            let a = c32_n(line_re[u], line_im[u]);
            let b = mul_c32_n(c32_n(line_re[v], line_im[v]), splat_c32_n(root));
            let out_u = add_c32_n(a, b);
            let out_v = add_c32_n(a, neg_c32_n(b));
            line_re[u] = out_u.re;
            line_im[u] = out_u.im;
            line_re[v] = out_v.re;
            line_im[v] = out_v.im;
        }
        workgroupBarrier();
    }
    let inv_scale = 1.0 / f32(N);
    for (var i = lid; i < N; i += WG) {
        let pos = line_pos(i, line);
        var out = c32_n(line_re[i], line_im[i]);
        if ((pc.flags & FLAG_INVERSE_FINALIZE) != 0u) {
            out = mul_c32_n(conj_c32_n(out), splat_c32_n(c32(inv_scale, 0.0)));
        }
        if ((pc.flags & FLAG_FORWARD_ALPHA) != 0u) {
            out = mark_opaque_alpha(out);
            write_shifted_d_re(pos, out.re);
            write_shifted_d_im(pos, out.im);
        }
        write_fft_buf(pc.dst_buffer, pos, out);
    }
}

@compute @workgroup_size(16, 16, 1)
fn fft_copy_buffer(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p64 = vec2<u32>(gid.xy);
    let dims = settings.size;
    if (p64.x >= dims.x || p64.y >= dims.y) {
        return;
    }
    let p32 = vec2<u32>(p64.x, p64.y);
    let v = read_fft_buf(pc.src_buffer, p32);
    write_fft_buf(pc.dst_buffer, p32, v);
}
