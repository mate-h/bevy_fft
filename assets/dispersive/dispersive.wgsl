// Hybrid dispersive: decomposition, bar SWE, pack/unpack, spectrum ops, transport, merge.
#import bevy_fft::bindings::{
    buffer_a_re,
    buffer_a_im,
    buffer_b_re,
    buffer_c_re,
    buffer_c_im,
}

struct DispersiveSim {
    n: u32,
    diffusion_iters: u32,
    g: f32,
    dt: f32,
    dx: f32,
    tile_world: f32,
    gamma_surf: f32,
    d_grad_penalty: f32,
    timestamp: u32,
    spectral_flags: u32,
    h_bar_omega: f32,
    spectral_fixed_depth: f32,
    airy_depth_0: f32,
    airy_depth_1: f32,
    airy_depth_2: f32,
    airy_depth_3: f32,
    airy_stack_channel: u32,
    h_avgmax_beta: f32,
    vel_clamp_alpha: f32,
    _cmf_pad: u32,
}

const SPECTRAL_FIXED_DEPTH: u32 = 1u;

@group(1) @binding(0) var<uniform> sim: DispersiveSim;
@group(1) @binding(1) var d_state: texture_storage_2d<rgba32float, read_write>;
@group(1) @binding(2) var d_bar: texture_storage_2d<rgba32float, read_write>;
@group(1) @binding(3) var d_tilde: texture_storage_2d<rgba32float, read_write>;
@group(1) @binding(4) var d_bed: texture_storage_2d<rgba32float, read_write>;
@group(1) @binding(5) var d_scratch: texture_storage_2d<rgba32float, read_write>;
@group(1) @binding(6) var d_hspec_re: texture_storage_2d<rgba32float, read_write>;
@group(1) @binding(7) var d_hspec_im: texture_storage_2d<rgba32float, read_write>;
@group(1) @binding(8) var d_qspec_backup: texture_storage_2d<rgba32float, read_write>;
@group(1) @binding(9) var d_airy_stack_qx: texture_storage_2d<rgba32float, read_write>;
@group(1) @binding(10) var d_airy_stack_qy: texture_storage_2d<rgba32float, read_write>;
@group(1) @binding(11) var bar_face_u: texture_storage_2d<r32float, read_write>;
@group(1) @binding(12) var bar_face_w: texture_storage_2d<r32float, read_write>;
@group(1) @binding(13) var bar_mac_u: texture_storage_2d<rgba32float, read_write>;
@group(1) @binding(14) var bar_mac_w: texture_storage_2d<rgba32float, read_write>;

const PI: f32 = 3.1415926535;

fn clampi(i: i32, n: i32) -> i32 { return clamp(i, 0, n - 1i); }

fn b_at(ix: i32, iy: i32) -> f32 {
    let n1 = i32(sim.n);
    return textureLoad(d_bed, vec2u(u32(clampi(ix, n1)), u32(clampi(iy, n1)))).r;
}

fn s_at(ix: i32, iy: i32) -> vec4f {
    let n1 = i32(sim.n);
    return textureLoad(d_state, vec2u(u32(clampi(ix, n1)), u32(clampi(iy, n1))));
}

@compute @workgroup_size(16, 16, 1)
fn decompose_init(@builtin(global_invocation_id) id: vec3u) {
    let n = sim.n;
    if (id.x >= n || id.y >= n) { return; }
    let ix = i32(id.x);
    let iy = i32(id.y);
    let b = b_at(ix, iy);
    let s = textureLoad(d_state, id.xy);
    let hw = max(s.r, 0.0);
    let eta = b + hw;
    textureStore(d_scratch, id.xy, vec4(eta, 0.0, 0.0, 0.0));
}

fn grad_h2(ix: i32, iy: i32) -> f32 {
    let hl = max(s_at(ix - 1, iy).r, 0.0);
    let hr = max(s_at(ix + 1, iy).r, 0.0);
    let hd = max(s_at(ix, iy - 1).r, 0.0);
    let hu = max(s_at(ix, iy + 1).r, 0.0);
    let gx = (hr - hl) / (2.0 * sim.dx);
    let gy = (hu - hd) / (2.0 * sim.dx);
    return gx * gx + gy * gy;
}

@compute @workgroup_size(16, 16, 1)
fn diffuse_eta_step(@builtin(global_invocation_id) id: vec3u) {
    let n = sim.n;
    if (id.x >= n || id.y >= n) { return; }
    let ix = i32(id.x);
    let iy = i32(id.y);
    let n1 = i32(n);
    let h_c = max(s_at(ix, iy).r, 1e-3);
    let d_pen = sim.d_grad_penalty;
    let ax = (h_c * h_c / 64.0) * exp(-d_pen * grad_h2(ix, iy));
    let dx = sim.dx;
    let dT = 0.25;
    let Hc = textureLoad(d_scratch, id.xy).r;
    let Hl = textureLoad(d_scratch, vec2u(u32(clampi(ix - 1, n1)), u32(iy))).r;
    let Hr = textureLoad(d_scratch, vec2u(u32(clampi(ix + 1, n1)), u32(iy))).r;
    let Hd = textureLoad(d_scratch, vec2u(u32(ix), u32(clampi(iy - 1, n1)))).r;
    let Hu = textureLoad(d_scratch, vec2u(u32(ix), u32(clampi(iy + 1, n1)))).r;
    let lap = (Hl + Hr + Hd + Hu - 4.0 * Hc) / (dx * dx);
    let H_new = Hc + ax * lap * dT;
    textureStore(d_scratch, id.xy, vec4(H_new, 0.0, 0.0, 0.0));
}

@compute @workgroup_size(16, 16, 1)
fn decompose_split(@builtin(global_invocation_id) id: vec3u) {
    let n = sim.n;
    if (id.x >= n || id.y >= n) { return; }
    let ix = i32(id.x);
    let iy = i32(id.y);
    let b = b_at(ix, iy);
    let eta_f = textureLoad(d_scratch, id.xy).r;
    let h_bar = max(eta_f - b, 0.0);
    let s = s_at(ix, iy);
    let hw = max(s.r, 0.0);
    let h_t = hw - h_bar;
    let qx = s.g;
    let qy = s.b;
    let den = max(hw, 1e-5);
    let r = h_bar / den;
    let qxb = qx * r;
    let qyb = qy * r;
    textureStore(d_bar, id.xy, vec4(h_bar, qxb, qyb, 0.0));
    textureStore(d_tilde, id.xy, vec4(h_t, qx - qxb, qy - qyb, 0.0));
}

// --- Bulk bar: Chentanez–Müller 2010 (CMF10), same order as `shallow_water/simulator.wgsl` MacCormack + upwind h + η pressure.

fn bar_nxi() -> i32 { return i32(sim.n); }

fn bar_h_at(ic: i32, jc: i32) -> f32 {
    let c = vec2u(u32(clampi(ic, bar_nxi())), u32(clampi(jc, bar_nxi())));
    return max(textureLoad(d_bar, c).r, 0.0);
}

fn bar_eta_at(ic: i32, jc: i32) -> f32 {
    return b_at(ic, jc) + bar_h_at(ic, jc);
}

fn bar_u_face_ij(fi: i32, jc: i32) -> vec2u {
    let n1 = bar_nxi();
    return vec2u(u32(clamp(fi, 0, n1)), u32(clamp(jc, 0, n1 - 1)));
}

fn bar_w_face_ij(ic: i32, fj: i32) -> vec2u {
    let n1 = bar_nxi();
    return vec2u(u32(clamp(ic, 0, n1 - 1)), u32(clamp(fj, 0, n1)));
}

fn bar_u_face_read(fi: i32, jc: i32) -> f32 {
    return textureLoad(bar_face_u, bar_u_face_ij(fi, jc)).x;
}

fn bar_w_face_read(ic: i32, fj: i32) -> f32 {
    return textureLoad(bar_face_w, bar_w_face_ij(ic, fj)).x;
}

fn bar_u_mac0_at(fi: i32, jc: i32) -> f32 {
    return textureLoad(bar_mac_u, bar_u_face_ij(fi, jc)).x;
}

fn bar_w_mac0_at(ic: i32, fj: i32) -> f32 {
    return textureLoad(bar_mac_w, bar_w_face_ij(ic, fj)).x;
}

fn bar_sample_u_bilinear(x: f32, y: f32) -> f32 {
    let nyi = bar_nxi();
    let nxf = bar_nxi() + 1;
    let jf = y - 0.5;
    var j0 = i32(floor(jf));
    let ty = jf - f32(j0);
    j0 = clamp(j0, 0, nyi - 1);
    let j1 = min(j0 + 1, nyi - 1);
    var i0 = i32(floor(x));
    let tx = x - f32(i0);
    i0 = clamp(i0, 0, nxf - 2);
    let i1 = min(i0 + 1, nxf - 1);
    let u00 = bar_u_face_read(i0, j0);
    let u10 = bar_u_face_read(i1, j0);
    let u01 = bar_u_face_read(i0, j1);
    let u11 = bar_u_face_read(i1, j1);
    return mix(mix(u00, u10, tx), mix(u01, u11, tx), ty);
}

fn bar_sample_u_bilinear_mac0(x: f32, y: f32) -> f32 {
    let nyi = bar_nxi();
    let nxf = bar_nxi() + 1;
    let jf = y - 0.5;
    var j0 = i32(floor(jf));
    let ty = jf - f32(j0);
    j0 = clamp(j0, 0, nyi - 1);
    let j1 = min(j0 + 1, nyi - 1);
    var i0 = i32(floor(x));
    let tx = x - f32(i0);
    i0 = clamp(i0, 0, nxf - 2);
    let i1 = min(i0 + 1, nxf - 1);
    let u00 = bar_u_mac0_at(i0, j0);
    let u10 = bar_u_mac0_at(i1, j0);
    let u01 = bar_u_mac0_at(i0, j1);
    let u11 = bar_u_mac0_at(i1, j1);
    return mix(mix(u00, u10, tx), mix(u01, u11, tx), ty);
}

fn bar_sample_w_bilinear(x: f32, y: f32) -> f32 {
    let nxi = bar_nxi();
    let nyf = bar_nxi() + 1;
    let ifloat = x - 0.5;
    var i0 = i32(floor(ifloat));
    let tx = ifloat - f32(i0);
    i0 = clamp(i0, 0, nxi - 1);
    let i1 = min(i0 + 1, nxi - 1);
    var fj = floor(y);
    let ty = y - fj;
    var j0 = i32(fj);
    j0 = clamp(j0, 0, nyf - 2);
    let j1 = min(j0 + 1, nyf - 1);
    let w00 = bar_w_face_read(i0, j0);
    let w10 = bar_w_face_read(i1, j0);
    let w01 = bar_w_face_read(i0, j1);
    let w11 = bar_w_face_read(i1, j1);
    return mix(mix(w00, w10, tx), mix(w01, w11, tx), ty);
}

fn bar_sample_w_bilinear_mac0(x: f32, y: f32) -> f32 {
    let nxi = bar_nxi();
    let nyf = bar_nxi() + 1;
    let ifloat = x - 0.5;
    var i0 = i32(floor(ifloat));
    let tx = ifloat - f32(i0);
    i0 = clamp(i0, 0, nxi - 1);
    let i1 = min(i0 + 1, nxi - 1);
    var fj = floor(y);
    let ty = y - fj;
    var j0 = i32(fj);
    j0 = clamp(j0, 0, nyf - 2);
    let j1 = min(j0 + 1, nyf - 1);
    let w00 = bar_w_mac0_at(i0, j0);
    let w10 = bar_w_mac0_at(i1, j0);
    let w01 = bar_w_mac0_at(i0, j1);
    let w11 = bar_w_mac0_at(i1, j1);
    return mix(mix(w00, w10, tx), mix(w01, w11, tx), ty);
}

fn bar_sample_mac_velocity_u_advect(x: f32, y: f32) -> vec2f {
    return vec2f(bar_sample_u_bilinear_mac0(x, y), bar_sample_w_bilinear(x, y));
}

fn bar_sample_mac_velocity_w_advect(x: f32, y: f32) -> vec2f {
    return vec2f(bar_sample_u_bilinear(x, y), bar_sample_w_bilinear_mac0(x, y));
}

fn bar_trace_semi_lag(pos: vec2f, v: vec2f, dt: f32) -> vec2f {
    let dx = sim.dx;
    let nx_f = f32(bar_nxi());
    let p = pos - v * (dt / dx);
    return vec2f(clamp(p.x, 0.0, nx_f), clamp(p.y, 0.0, nx_f));
}

fn bar_semi_lag_u_from(pos: vec2f, dt_sign: f32) -> f32 {
    let v = bar_sample_mac_velocity_u_advect(pos.x, pos.y);
    let pback = bar_trace_semi_lag(pos, v, sim.dt * dt_sign);
    return bar_sample_u_bilinear_mac0(pback.x, pback.y);
}

fn bar_semi_lag_w_from(pos: vec2f, dt_sign: f32) -> f32 {
    let v = bar_sample_mac_velocity_w_advect(pos.x, pos.y);
    let pback = bar_trace_semi_lag(pos, v, sim.dt * dt_sign);
    return bar_sample_w_bilinear_mac0(pback.x, pback.y);
}

fn bar_semi_lag_u_reverse_hat(pos: vec2f) -> f32 {
    let v = bar_sample_mac_velocity_u_advect(pos.x, pos.y);
    let pback = bar_trace_semi_lag(pos, v, -sim.dt);
    return bar_sample_u_bilinear(pback.x, pback.y);
}

fn bar_semi_lag_w_reverse_hat(pos: vec2f) -> f32 {
    let v = bar_sample_mac_velocity_w_advect(pos.x, pos.y);
    let pback = bar_trace_semi_lag(pos, v, -sim.dt);
    return bar_sample_w_bilinear(pback.x, pback.y);
}

fn bar_clamp_face_vel(v: f32) -> f32 {
    let vm = sim.vel_clamp_alpha * sim.dx / max(sim.dt, 1e-6);
    return clamp(v, -vm, vm);
}

@compute @workgroup_size(16, 16, 1)
fn bar_sync_u_faces(@builtin(global_invocation_id) id: vec3u) {
    let nx = bar_nxi();
    if (id.x > u32(nx) || id.y >= u32(nx)) { return; }
    let fi = i32(id.x);
    let jc = i32(id.y);
    if (fi == 0 || fi == nx) {
        textureStore(bar_face_u, id.xy, vec4f(0.0));
        return;
    }
    let hl = bar_h_at(fi - 1, jc);
    let hr = bar_h_at(fi, jc);
    let ql = textureLoad(d_bar, vec2u(u32(clampi(fi - 1, nx)), u32(clampi(jc, nx)))).g;
    let qr = textureLoad(d_bar, vec2u(u32(clampi(fi, nx)), u32(clampi(jc, nx)))).g;
    let ul = ql / max(hl, 1e-6);
    let ur = qr / max(hr, 1e-6);
    let u_face = 0.5 * (ul + ur);
    textureStore(bar_face_u, id.xy, vec4f(bar_clamp_face_vel(u_face)));
}

@compute @workgroup_size(16, 16, 1)
fn bar_sync_w_faces(@builtin(global_invocation_id) id: vec3u) {
    let nx = bar_nxi();
    if (id.x >= u32(nx) || id.y > u32(nx)) { return; }
    let ic = i32(id.x);
    let fj = i32(id.y);
    if (fj == 0 || fj == nx) {
        textureStore(bar_face_w, id.xy, vec4f(0.0));
        return;
    }
    let hd = bar_h_at(ic, fj - 1);
    let hu = bar_h_at(ic, fj);
    let qd = textureLoad(d_bar, vec2u(u32(clampi(ic, nx)), u32(clampi(fj - 1, nx)))).b;
    let qu = textureLoad(d_bar, vec2u(u32(clampi(ic, nx)), u32(clampi(fj, nx)))).b;
    let vd = qd / max(hd, 1e-6);
    let vu = qu / max(hu, 1e-6);
    let w_face = 0.5 * (vd + vu);
    textureStore(bar_face_w, id.xy, vec4f(bar_clamp_face_vel(w_face)));
}

@compute @workgroup_size(16, 16, 1)
fn bar_mac_u_copy(@builtin(global_invocation_id) id: vec3u) {
    let nx = bar_nxi();
    let sx = u32(nx + 1);
    let sy = u32(nx);
    if (id.x >= sx || id.y >= sy) { return; }
    let u = textureLoad(bar_face_u, id.xy).x;
    textureStore(bar_mac_u, id.xy, vec4f(u, 0.0, 0.0, 0.0));
}

@compute @workgroup_size(16, 16, 1)
fn bar_mac_u_sl_forward(@builtin(global_invocation_id) id: vec3u) {
    let nx = bar_nxi();
    let sx = u32(nx + 1);
    let sy = u32(nx);
    if (id.x >= sx || id.y >= sy) { return; }
    let fi = i32(id.x);
    let jc = i32(id.y);
    let pos = vec2f(f32(fi), f32(jc) + 0.5);
    let u_hat = bar_semi_lag_u_from(pos, 1.0);
    textureStore(bar_face_u, id.xy, vec4f(bar_clamp_face_vel(u_hat)));
}

@compute @workgroup_size(16, 16, 1)
fn bar_mac_u_sl_reverse(@builtin(global_invocation_id) id: vec3u) {
    let nx = bar_nxi();
    let sx = u32(nx + 1);
    let sy = u32(nx);
    if (id.x >= sx || id.y >= sy) { return; }
    let fi = i32(id.x);
    let jc = i32(id.y);
    let pos = vec2f(f32(fi), f32(jc) + 0.5);
    let u_tilde = bar_semi_lag_u_reverse_hat(pos);
    var mu = textureLoad(bar_mac_u, id.xy);
    mu.y = u_tilde;
    textureStore(bar_mac_u, id.xy, mu);
}

@compute @workgroup_size(16, 16, 1)
fn bar_mac_u_combine(@builtin(global_invocation_id) id: vec3u) {
    let nx = bar_nxi();
    let sx = u32(nx + 1);
    let sy = u32(nx);
    if (id.x >= sx || id.y >= sy) { return; }
    let mu = textureLoad(bar_mac_u, id.xy);
    let u_n = mu.x;
    let u_hat = textureLoad(bar_face_u, id.xy).x;
    let u_tilde = mu.y;
    var u_new = u_hat + 0.5 * (u_n - u_tilde);
    let min_u = min(u_n, u_hat);
    let max_u = max(u_n, u_hat);
    if (u_new < min_u - 1e-5 || u_new > max_u + 1e-5) { u_new = u_hat; }
    textureStore(bar_face_u, id.xy, vec4f(bar_clamp_face_vel(u_new)));
}

@compute @workgroup_size(16, 16, 1)
fn bar_mac_w_copy(@builtin(global_invocation_id) id: vec3u) {
    let nx = bar_nxi();
    let sx = u32(nx);
    let sy = u32(nx + 1);
    if (id.x >= sx || id.y >= sy) { return; }
    let w = textureLoad(bar_face_w, id.xy).x;
    textureStore(bar_mac_w, id.xy, vec4f(w, 0.0, 0.0, 0.0));
}

@compute @workgroup_size(16, 16, 1)
fn bar_mac_w_sl_forward(@builtin(global_invocation_id) id: vec3u) {
    let nx = bar_nxi();
    let sx = u32(nx);
    let sy = u32(nx + 1);
    if (id.x >= sx || id.y >= sy) { return; }
    let ic = i32(id.x);
    let fj = i32(id.y);
    let pos = vec2f(f32(ic) + 0.5, f32(fj));
    let w_hat = bar_semi_lag_w_from(pos, 1.0);
    textureStore(bar_face_w, id.xy, vec4f(bar_clamp_face_vel(w_hat)));
}

@compute @workgroup_size(16, 16, 1)
fn bar_mac_w_sl_reverse(@builtin(global_invocation_id) id: vec3u) {
    let nx = bar_nxi();
    let sx = u32(nx);
    let sy = u32(nx + 1);
    if (id.x >= sx || id.y >= sy) { return; }
    let ic = i32(id.x);
    let fj = i32(id.y);
    let pos = vec2f(f32(ic) + 0.5, f32(fj));
    let w_tilde = bar_semi_lag_w_reverse_hat(pos);
    var mw = textureLoad(bar_mac_w, id.xy);
    mw.y = w_tilde;
    textureStore(bar_mac_w, id.xy, mw);
}

@compute @workgroup_size(16, 16, 1)
fn bar_mac_w_combine(@builtin(global_invocation_id) id: vec3u) {
    let nx = bar_nxi();
    let sx = u32(nx);
    let sy = u32(nx + 1);
    if (id.x >= sx || id.y >= sy) { return; }
    let mw = textureLoad(bar_mac_w, id.xy);
    let w_n = mw.x;
    let w_hat = textureLoad(bar_face_w, id.xy).x;
    let w_tilde = mw.y;
    var w_new = w_hat + 0.5 * (w_n - w_tilde);
    let min_w = min(w_n, w_hat);
    let max_w = max(w_n, w_hat);
    if (w_new < min_w - 1e-5 || w_new > max_w + 1e-5) { w_new = w_hat; }
    textureStore(bar_face_w, id.xy, vec4f(bar_clamp_face_vel(w_new)));
}

fn bar_h_adj_cell(ic: i32, jc: i32) -> f32 {
    let g = sim.g;
    let dt = sim.dt;
    let dx = sim.dx;
    let beta = sim.h_avgmax_beta;
    let h_avgmax = beta * dx / (g * dt);
    let h_e = bar_h_at(ic + 1, jc);
    let h_w = bar_h_at(ic - 1, jc);
    let h_n = bar_h_at(ic, jc + 1);
    let h_s = bar_h_at(ic, jc - 1);
    let avg = 0.25 * (h_e + h_w + h_n + h_s);
    return max(0.0, avg - h_avgmax);
}

fn bar_upwind_h_flux(h_up: f32, h_dn: f32, u_face: f32, h_adj_up: f32, h_adj_dn: f32) -> f32 {
    let hu = select(h_dn - h_adj_dn, h_up - h_adj_up, u_face > 0.0);
    return u_face * max(0.0, hu);
}

@compute @workgroup_size(16, 16, 1)
fn bar_cmf10_integrate_height(@builtin(global_invocation_id) id: vec3u) {
    let n = sim.n;
    if (id.x >= n || id.y >= n) { return; }
    let i = i32(id.x);
    let j = i32(id.y);
    let dx = sim.dx;
    let dt = sim.dt;
    let uL = bar_u_face_read(i, j);
    let uR = bar_u_face_read(i + 1, j);
    let wB = bar_w_face_read(i, j);
    let wT = bar_w_face_read(i, j + 1);
    let aj = bar_h_adj_cell(i, j);
    let aj_l = bar_h_adj_cell(i - 1, j);
    let aj_r = bar_h_adj_cell(i + 1, j);
    let aj_b = bar_h_adj_cell(i, j - 1);
    let aj_t = bar_h_adj_cell(i, j + 1);
    let h_c = bar_h_at(i, j);
    let h_l = bar_h_at(i - 1, j);
    let h_r = bar_h_at(i + 1, j);
    let h_d = bar_h_at(i, j - 1);
    let h_u = bar_h_at(i, j + 1);
    let f_e = bar_upwind_h_flux(h_c, h_r, uR, aj, aj_r);
    let f_w = bar_upwind_h_flux(h_l, h_c, uL, aj_l, aj);
    let f_n = bar_upwind_h_flux(h_c, h_u, wT, aj, aj_t);
    let f_s = bar_upwind_h_flux(h_d, h_c, wB, aj_b, aj);
    let div = f_e - f_w + f_n - f_s;
    var db = textureLoad(d_bar, id.xy);
    db.x += -dt / dx * div;
    db.x = max(0.0, db.x);
    textureStore(d_bar, id.xy, db);
}

@compute @workgroup_size(16, 16, 1)
fn bar_cmf10_vel_u(@builtin(global_invocation_id) id: vec3u) {
    let nx = bar_nxi();
    let sx = u32(nx + 1);
    let sy = u32(nx);
    if (id.x >= sx || id.y >= sy) { return; }
    let fi = i32(id.x);
    let jc = i32(id.y);
    if (fi < 1 || fi > nx) { return; }
    let g = sim.g;
    let dx = sim.dx;
    let dt = sim.dt;
    let eta_l = bar_eta_at(fi - 1, jc);
    let eta_r = bar_eta_at(fi, jc);
    var u = textureLoad(bar_face_u, id.xy).x;
    u += (-g / dx * (eta_r - eta_l)) * dt;
    textureStore(bar_face_u, id.xy, vec4f(bar_clamp_face_vel(u)));
}

@compute @workgroup_size(16, 16, 1)
fn bar_cmf10_vel_w(@builtin(global_invocation_id) id: vec3u) {
    let nx = bar_nxi();
    let sx = u32(nx);
    let sy = u32(nx + 1);
    if (id.x >= sx || id.y >= sy) { return; }
    let ic = i32(id.x);
    let fj = i32(id.y);
    if (fj < 1 || fj > nx) { return; }
    let g = sim.g;
    let dx = sim.dx;
    let dt = sim.dt;
    let eta_d = bar_eta_at(ic, fj - 1);
    let eta_u = bar_eta_at(ic, fj);
    var wv = textureLoad(bar_face_w, id.xy).x;
    wv += (-g / dx * (eta_u - eta_d)) * dt;
    textureStore(bar_face_w, id.xy, vec4f(bar_clamp_face_vel(wv)));
}

@compute @workgroup_size(16, 16, 1)
fn bar_gather_cell_q(@builtin(global_invocation_id) id: vec3u) {
    let n = sim.n;
    if (id.x >= n || id.y >= n) { return; }
    let i = i32(id.x);
    let j = i32(id.y);
    let h = max(textureLoad(d_bar, id.xy).r, 0.0);
    let uL = bar_u_face_read(i, j);
    let uR = bar_u_face_read(i + 1, j);
    let wB = bar_w_face_read(i, j);
    let wT = bar_w_face_read(i, j + 1);
    let u_avg = 0.5 * (uL + uR);
    let v_avg = 0.5 * (wB + wT);
    var qx = h * u_avg;
    var qy = h * v_avg;
    let dx = sim.dx;
    let dt = sim.dt;
    let h_safe = max(h, 1e-3);
    let u_m = dx / (4.0 * max(dt, 1e-6));
    let ul = length(vec2f(qx / h_safe, qy / h_safe));
    if (ul > u_m) {
        let sc = u_m / max(ul, 1e-6);
        qx = (qx / h_safe) * sc * h_safe;
        qy = (qy / h_safe) * sc * h_safe;
    }
    textureStore(d_bar, id.xy, vec4(h, qx, qy, 0.0));
}

@compute @workgroup_size(16, 16, 1)
fn pack_h_t_to_a(@builtin(global_invocation_id) id: vec3u) {
    let n = sim.n;
    if (id.x >= n || id.y >= n) { return; }
    let t = textureLoad(d_tilde, id.xy);
    let h = t.r;
    let z = vec4f(0.0, 0.0, 0.0, 0.0);
    textureStore(buffer_a_re, id.xy, vec4(h, 0.0, 0.0, 0.0));
    textureStore(buffer_a_im, id.xy, z);
}

@compute @workgroup_size(16, 16, 1)
fn copy_c_to_hspec(@builtin(global_invocation_id) id: vec3u) {
    let n = sim.n;
    if (id.x >= n || id.y >= n) { return; }
    let cr = textureLoad(buffer_c_re, id.xy);
    let ci = textureLoad(buffer_c_im, id.xy);
    textureStore(d_hspec_re, id.xy, cr);
    textureStore(d_hspec_im, id.xy, ci);
}

@compute @workgroup_size(16, 16, 1)
fn pack_qx_t_to_a(@builtin(global_invocation_id) id: vec3u) {
    let n = sim.n;
    if (id.x >= n || id.y >= n) { return; }
    let t = textureLoad(d_tilde, id.xy);
    let z = vec4f(0.0, 0.0, 0.0, 0.0);
    textureStore(buffer_a_re, id.xy, vec4(t.g, 0.0, 0.0, 0.0));
    textureStore(buffer_a_im, id.xy, z);
}

@compute @workgroup_size(16, 16, 1)
fn pack_qy_t_to_a(@builtin(global_invocation_id) id: vec3u) {
    let n = sim.n;
    if (id.x >= n || id.y >= n) { return; }
    let t = textureLoad(d_tilde, id.xy);
    let z = vec4f(0.0, 0.0, 0.0, 0.0);
    textureStore(buffer_a_re, id.xy, vec4(t.b, 0.0, 0.0, 0.0));
    textureStore(buffer_a_im, id.xy, z);
}

@compute @workgroup_size(16, 16, 1)
fn qspec_save_c_to_backup(@builtin(global_invocation_id) id: vec3u) {
    let n = sim.n;
    if (id.x >= n || id.y >= n) { return; }
    let cr = textureLoad(buffer_c_re, id.xy);
    let ci = textureLoad(buffer_c_im, id.xy);
    textureStore(d_qspec_backup, id.xy, vec4(cr.x, ci.x, 0.0, 0.0));
}

@compute @workgroup_size(16, 16, 1)
fn qspec_restore_backup_to_c(@builtin(global_invocation_id) id: vec3u) {
    let n = sim.n;
    if (id.x >= n || id.y >= n) { return; }
    let b = textureLoad(d_qspec_backup, id.xy);
    var cr = textureLoad(buffer_c_re, id.xy);
    var ci = textureLoad(buffer_c_im, id.xy);
    cr.x = b.r;
    ci.x = b.g;
    textureStore(buffer_c_re, id.xy, cr);
    textureStore(buffer_c_im, id.xy, ci);
}

@compute @workgroup_size(16, 16, 1)
fn airy_stack_clear_qx(@builtin(global_invocation_id) id: vec3u) {
    let n = sim.n;
    if (id.x >= n || id.y >= n) { return; }
    textureStore(d_airy_stack_qx, id.xy, vec4(0.0));
}

@compute @workgroup_size(16, 16, 1)
fn airy_stack_clear_qy(@builtin(global_invocation_id) id: vec3u) {
    let n = sim.n;
    if (id.x >= n || id.y >= n) { return; }
    textureStore(d_airy_stack_qy, id.xy, vec4(0.0));
}

@compute @workgroup_size(16, 16, 1)
fn airy_stack_write_qx_from_b(@builtin(global_invocation_id) id: vec3u) {
    let n = sim.n;
    if (id.x >= n || id.y >= n) { return; }
    let br = textureLoad(buffer_b_re, id.xy);
    let s = textureLoad(d_airy_stack_qx, id.xy);
    let ch = sim.airy_stack_channel;
    var o = s;
    switch ch {
        case 0u: { o.x = br.x; }
        case 1u: { o.y = br.x; }
        case 2u: { o.z = br.x; }
        default: { o.w = br.x; }
    }
    textureStore(d_airy_stack_qx, id.xy, o);
}

@compute @workgroup_size(16, 16, 1)
fn airy_stack_write_qy_from_b(@builtin(global_invocation_id) id: vec3u) {
    let n = sim.n;
    if (id.x >= n || id.y >= n) { return; }
    let br = textureLoad(buffer_b_re, id.xy);
    let s = textureLoad(d_airy_stack_qy, id.xy);
    let ch = sim.airy_stack_channel;
    var o = s;
    switch ch {
        case 0u: { o.x = br.x; }
        case 1u: { o.y = br.x; }
        case 2u: { o.z = br.x; }
        default: { o.w = br.x; }
    }
    textureStore(d_airy_stack_qy, id.xy, o);
}

fn lerp_depth_sample(h_bar: f32, d0: f32, d1: f32, d2: f32, d3: f32, v: vec4f) -> f32 {
    let h = max(h_bar, 1e-6);
    if (h <= d0) {
        return v.x;
    }
    if (h >= d3) {
        return v.w;
    }
    if (h <= d1) {
        let t = (h - d0) / max(d1 - d0, 1e-6);
        return mix(v.x, v.y, clamp(t, 0.0, 1.0));
    }
    if (h <= d2) {
        let t = (h - d1) / max(d2 - d1, 1e-6);
        return mix(v.y, v.z, clamp(t, 0.0, 1.0));
    }
    let t = (h - d2) / max(d3 - d2, 1e-6);
    return mix(v.z, v.w, clamp(t, 0.0, 1.0));
}

@compute @workgroup_size(16, 16, 1)
fn blend_airy_qx_to_tilde(@builtin(global_invocation_id) id: vec3u) {
    let n = sim.n;
    if (id.x >= n || id.y >= n) { return; }
    let hb = max(textureLoad(d_bar, id.xy).r, 1e-6);
    let st = textureLoad(d_airy_stack_qx, id.xy);
    let t = textureLoad(d_tilde, id.xy);
    let qx_b = lerp_depth_sample(hb, sim.airy_depth_0, sim.airy_depth_1, sim.airy_depth_2, sim.airy_depth_3, st);
    textureStore(d_tilde, id.xy, vec4(t.r, qx_b, t.b, 0.0));
}

@compute @workgroup_size(16, 16, 1)
fn blend_airy_qy_to_tilde(@builtin(global_invocation_id) id: vec3u) {
    let n = sim.n;
    if (id.x >= n || id.y >= n) { return; }
    let hb = max(textureLoad(d_bar, id.xy).r, 1e-6);
    let st = textureLoad(d_airy_stack_qy, id.xy);
    let t = textureLoad(d_tilde, id.xy);
    let qy_b = lerp_depth_sample(hb, sim.airy_depth_0, sim.airy_depth_1, sim.airy_depth_2, sim.airy_depth_3, st);
    textureStore(d_tilde, id.xy, vec4(t.r, t.g, qy_b, 0.0));
}

fn k_xy(id: vec2u) -> vec2f {
    let d = 2.0 * PI / sim.tile_world;
    let n_f = f32(sim.n);
    let hf = 0.5 * n_f;
    let kx = select(
        (f32(id.x) - n_f) * d,
        f32(id.x) * d,
        f32(id.x) <= hf
    );
    let ky = select(
        (f32(id.y) - n_f) * d,
        f32(id.y) * d,
        f32(id.y) <= hf
    );
    return vec2f(kx, ky);
}

fn beta_num(k: f32, dx: f32) -> f32 {
    if (k < 1e-10) { return 1.0; }
    let s = sin(k * dx * 0.5);
    return sqrt(max((2.0 * k / max(dx, 1e-6)) * s, 0.0));
}

fn omega_disp(g: f32, k: f32, hbar: f32, dx: f32) -> f32 {
    if (k < 1e-10) { return 0.0; }
    let b = max(beta_num(k, dx), 1e-6);
    let raw = sqrt(max(g * k * tanh(k * hbar), 0.0));
    return raw / b;
}

// Half-cell shift e^{-i k·Δ/2} on ĥ for x (qx) or y (qy) derivative (paper Sec. 4.3).
fn h_phase_shift_x(hre: f32, him: f32, kx: f32) -> vec2f {
    let th = kx * sim.dx * 0.5;
    let ct = cos(th);
    let st = sin(th);
    return vec2f(hre * ct + him * st, him * ct - hre * st);
}

fn h_phase_shift_y(hre: f32, him: f32, ky: f32) -> vec2f {
    let th = ky * sim.dx * 0.5;
    let ct = cos(th);
    let st = sin(th);
    return vec2f(hre * ct + him * st, him * ct - hre * st);
}

@compute @workgroup_size(16, 16, 1)
fn hybrid_k_qx(@builtin(global_invocation_id) id: vec3u) {
    let n = sim.n;
    if (id.x >= n || id.y >= n) { return; }
    let k2 = k_xy(id.xy);
    let klen = length(k2);
    var hbar = max(textureLoad(d_bar, id.xy).r, sim.h_bar_omega);
    if ((sim.spectral_flags & SPECTRAL_FIXED_DEPTH) != 0u) {
        hbar = max(sim.spectral_fixed_depth, sim.h_bar_omega);
    }
    let w = omega_disp(sim.g, klen, hbar, sim.dx);
    let dt = sim.dt;
    let c = cos(w * dt);
    let s = sin(w * dt);
    var qre = textureLoad(buffer_c_re, id.xy);
    var qim = textureLoad(buffer_c_im, id.xy);
    let hre0 = textureLoad(d_hspec_re, id.xy).x;
    let him0 = textureLoad(d_hspec_im, id.xy).x;
    let k2_safe = max(klen * klen, 1e-12);
    if (klen < 1e-8) {
        textureStore(buffer_c_re, id.xy, qre);
        textureStore(buffer_c_im, id.xy, qim);
        return;
    }
    let kx = k2.x;
    let hp = h_phase_shift_x(hre0, him0, kx);
    let hre = hp.x;
    let him = hp.y;
    let sc = s * (w / k2_safe);
    let oqx_re = c * qre.x + sc * kx * him;
    let oqx_im = c * qim.x - sc * kx * hre;
    var out_re = qre;
    out_re.x = oqx_re;
    var out_im = qim;
    out_im.x = oqx_im;
    textureStore(buffer_c_re, id.xy, out_re);
    textureStore(buffer_c_im, id.xy, out_im);
}

@compute @workgroup_size(16, 16, 1)
fn hybrid_k_qy(@builtin(global_invocation_id) id: vec3u) {
    let n = sim.n;
    if (id.x >= n || id.y >= n) { return; }
    let k2 = k_xy(id.xy);
    let klen = length(k2);
    var hbar = max(textureLoad(d_bar, id.xy).r, sim.h_bar_omega);
    if ((sim.spectral_flags & SPECTRAL_FIXED_DEPTH) != 0u) {
        hbar = max(sim.spectral_fixed_depth, sim.h_bar_omega);
    }
    let w = omega_disp(sim.g, klen, hbar, sim.dx);
    let dt = sim.dt;
    let c = cos(w * dt);
    let s = sin(w * dt);
    var qre = textureLoad(buffer_c_re, id.xy);
    var qim = textureLoad(buffer_c_im, id.xy);
    let hre0 = textureLoad(d_hspec_re, id.xy).x;
    let him0 = textureLoad(d_hspec_im, id.xy).x;
    let k2_safe = max(klen * klen, 1e-12);
    if (klen < 1e-8) {
        textureStore(buffer_c_re, id.xy, qre);
        textureStore(buffer_c_im, id.xy, qim);
        return;
    }
    let ky = k2.y;
    let hp = h_phase_shift_y(hre0, him0, ky);
    let hre = hp.x;
    let him = hp.y;
    let sc = s * (w / k2_safe);
    let oqy_re = c * qre.x + sc * ky * him;
    let oqy_im = c * qim.x - sc * ky * hre;
    var out_re = qre;
    out_re.x = oqy_re;
    var out_im = qim;
    out_im.x = oqy_im;
    textureStore(buffer_c_re, id.xy, out_re);
    textureStore(buffer_c_im, id.xy, out_im);
}

fn bilerp_v4_from_scratch(p: vec2f, n: u32) -> vec4f {
    let nf = f32(n) - 1.0;
    let x = clamp(p.x, 0.0, nf);
    let y = clamp(p.y, 0.0, nf);
    let i0 = u32(floor(x));
    let j0 = u32(floor(y));
    let i1 = min(i0 + 1u, n - 1u);
    let j1 = min(j0 + 1u, n - 1u);
    let tx = x - f32(i0);
    let ty = y - f32(j0);
    let v00 = textureLoad(d_scratch, vec2u(i0, j0));
    let v10 = textureLoad(d_scratch, vec2u(i1, j0));
    let v01 = textureLoad(d_scratch, vec2u(i0, j1));
    let v11 = textureLoad(d_scratch, vec2u(i1, j1));
    return mix(mix(v00, v10, tx), mix(v01, v11, tx), ty);
}

// Algorithm 3 order: growth exp(G Δt) at the cell, then semi-Lagrangian fetch (paper).
@compute @workgroup_size(16, 16, 1)
fn transport_copy_tilde_to_scratch(@builtin(global_invocation_id) id: vec3u) {
    let n = sim.n;
    if (id.x >= n || id.y >= n) { return; }
    let ix = i32(id.x);
    let iy = i32(id.y);
    let n1 = i32(n);
    let b = textureLoad(d_bar, id.xy);
    let h_b = max(b.r, 1e-3);
    let ux = b.g / h_b;
    let uy = b.b / h_b;
    let bl = textureLoad(d_bar, vec2u(u32(clampi(ix - 1, n1)), u32(iy)));
    let br = textureLoad(d_bar, vec2u(u32(clampi(ix + 1, n1)), u32(iy)));
    let bd = textureLoad(d_bar, vec2u(u32(ix), u32(clampi(iy - 1, n1))));
    let bu = textureLoad(d_bar, vec2u(u32(ix), u32(clampi(iy + 1, n1))));
    let ux_r = br.g / max(br.r, 1e-3);
    let ux_l = bl.g / max(bl.r, 1e-3);
    let uy_u = bu.b / max(bu.r, 1e-3);
    let uy_d = bd.b / max(bd.r, 1e-3);
    let dx = sim.dx;
    let div = (ux_r - ux_l) / (2.0 * dx) + (uy_u - uy_d) / (2.0 * dx);
    var G = -div;
    let gm = -sim.gamma_surf * div;
    G = min(G, gm);
    let fac = exp(G * sim.dt);
    let t = textureLoad(d_tilde, id.xy);
    textureStore(d_scratch, id.xy, vec4(t.r * fac, t.g * fac, t.b * fac, 0.0));
}

@compute @workgroup_size(16, 16, 1)
fn transport_advect_tilde(@builtin(global_invocation_id) id: vec3u) {
    let n = sim.n;
    if (id.x >= n || id.y >= n) { return; }
    let ix = i32(id.x);
    let iy = i32(id.y);
    let b = textureLoad(d_bar, id.xy);
    let h_b = max(b.r, 1e-3);
    let ux = b.g / h_b;
    let uy = b.b / h_b;
    let dx = sim.dx;
    let p = vec2f(f32(id.x), f32(id.y)) - vec2f(ux, uy) * (sim.dt / max(dx, 1e-6));
    let t_s = bilerp_v4_from_scratch(p, n);
    textureStore(d_tilde, id.xy, vec4(t_s.r, t_s.g, t_s.b, 0.0));
}

// Eq. 17: h += Δt ∇·(q̄ + q̃ + q̌), q̌ ≈ h̃ ū at cell centers (collocated).
@compute @workgroup_size(16, 16, 1)
fn merge_to_state(@builtin(global_invocation_id) id: vec3u) {
    let n = sim.n;
    if (id.x >= n || id.y >= n) { return; }
    let ix = i32(id.x);
    let iy = i32(id.y);
    let n1 = i32(n);
    let s0 = s_at(ix, iy);
    let b = textureLoad(d_bar, id.xy);
    let t = textureLoad(d_tilde, id.xy);
    let bl = textureLoad(d_bar, vec2u(u32(clampi(ix - 1, n1)), u32(iy)));
    let br = textureLoad(d_bar, vec2u(u32(clampi(ix + 1, n1)), u32(iy)));
    let bd = textureLoad(d_bar, vec2u(u32(ix), u32(clampi(iy - 1, n1))));
    let bu = textureLoad(d_bar, vec2u(u32(ix), u32(clampi(iy + 1, n1))));
    let tl = textureLoad(d_tilde, vec2u(u32(clampi(ix - 1, n1)), u32(iy)));
    let tr = textureLoad(d_tilde, vec2u(u32(clampi(ix + 1, n1)), u32(iy)));
    let td = textureLoad(d_tilde, vec2u(u32(ix), u32(clampi(iy - 1, n1))));
    let tu = textureLoad(d_tilde, vec2u(u32(ix), u32(clampi(iy + 1, n1))));
    let qx = b.g + t.g;
    let qy = b.b + t.b;
    let dx = sim.dx;
    let divq = (br.g + tr.g - bl.g - tl.g) / (2.0 * dx) + (bu.b + tu.b - bd.b - td.b) / (2.0 * dx);
    let u_c = b.g / max(b.r, 1e-3);
    let v_c = b.b / max(b.r, 1e-3);
    let u_r = br.g / max(br.r, 1e-3);
    let u_l = bl.g / max(bl.r, 1e-3);
    let v_u = bu.b / max(bu.r, 1e-3);
    let v_d = bd.b / max(bd.r, 1e-3);
    let qcx_here = t.r * u_c;
    let qcx_r = tr.r * u_r;
    let qcx_l = tl.r * u_l;
    let qcy_here = t.r * v_c;
    let qcy_u = tu.r * v_u;
    let qcy_d = td.r * v_d;
    let div_check = (qcx_r - qcx_l) / (2.0 * dx) + (qcy_u - qcy_d) / (2.0 * dx);
    let div_total = divq + div_check;
    var h1 = s0.r - sim.dt * div_total;
    h1 = max(h1, 0.0);
    var qx1 = qx;
    var qy1 = qy;
    let qm = h1 * dx / (4.0 * max(sim.dt, 1e-6));
    let nq = length(vec2f(qx1, qy1));
    if (nq > qm) {
        let sc = qm / max(nq, 1e-6);
        qx1 *= sc;
        qy1 *= sc;
    }
    textureStore(d_state, id.xy, vec4(h1, qx1, qy1, 0.0));
}

@compute @workgroup_size(16, 16, 1)
fn init_sloped_beach(@builtin(global_invocation_id) id: vec3u) {
    let n = sim.n;
    if (id.x >= n || id.y >= n) { return; }
    let t = f32(n - 1u);
    let sx = f32(id.x) / t;
    let b = -2.0 + sx * 3.5;
    textureStore(d_bed, id.xy, vec4(b, 0.0, 0.0, 0.0));
    if (b < -0.2) {
        let h0 = 2.5;
        let q0 = 0.0;
        textureStore(d_state, id.xy, vec4(h0, q0, q0, 0.0));
    } else {
        let h0 = 0.2;
        textureStore(d_state, id.xy, vec4(h0, 0.0, 0.0, 0.0));
    }
}
