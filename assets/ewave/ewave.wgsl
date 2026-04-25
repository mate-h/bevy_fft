#import bevy_fft::bindings::{
    buffer_a_re,
    buffer_a_im,
    buffer_b_re,
    buffer_c_re,
    buffer_c_im,
}

struct EwaveSim {
    n: u32,
    _pad0: u32,
    g: f32,
    dt: f32,
    tile_world: f32,
    timestamp: u32,
    brush_on: u32,
    brush_radius: f32,
    brush_strength: f32,
    pointer_x: f32,
    pointer_y: f32,
    pointer_ox: f32,
    pointer_oy: f32,
}

@group(1) @binding(0) var<uniform> sim: EwaveSim;
@group(1) @binding(1) var h_phi: texture_storage_2d<rgba32float, read_write>;
@group(1) @binding(2) var h_hat_re: texture_storage_2d<rgba32float, read_write>;
@group(1) @binding(3) var h_hat_im: texture_storage_2d<rgba32float, read_write>;
@group(1) @binding(4) var p_hat_re: texture_storage_2d<rgba32float, read_write>;
@group(1) @binding(5) var p_hat_im: texture_storage_2d<rgba32float, read_write>;

const PI: f32 = 3.14159265;

@compute @workgroup_size(16, 16, 1)
fn clear_spatial_to_flat(@builtin(global_invocation_id) id: vec3u) {
    let n = sim.n;
    if (id.x >= n || id.y >= n) { return; }
    textureStore(h_phi, id.xy, vec4(0.0, 0.0, 0.0, 0.0));
}

@compute @workgroup_size(16, 16, 1)
fn pack_h_to_a(@builtin(global_invocation_id) id: vec3u) {
    let n = sim.n;
    if (id.x >= n || id.y >= n) { return; }
    let hp = textureLoad(h_phi, id.xy);
    let z = vec4(0.0, 0.0, 0.0, 0.0);
    textureStore(buffer_a_re, id.xy, vec4(hp.x, 0.0, 0.0, 0.0));
    textureStore(buffer_a_im, id.xy, z);
}

@compute @workgroup_size(16, 16, 1)
fn pack_phi_to_a(@builtin(global_invocation_id) id: vec3u) {
    let n = sim.n;
    if (id.x >= n || id.y >= n) { return; }
    let hp = textureLoad(h_phi, id.xy);
    let z = vec4(0.0, 0.0, 0.0, 0.0);
    textureStore(buffer_a_re, id.xy, vec4(hp.y, 0.0, 0.0, 0.0));
    textureStore(buffer_a_im, id.xy, z);
}

@compute @workgroup_size(16, 16, 1)
fn copy_c_to_spectrum_h(@builtin(global_invocation_id) id: vec3u) {
    let n = sim.n;
    if (id.x >= n || id.y >= n) { return; }
    let cr = textureLoad(buffer_c_re, id.xy);
    let ci = textureLoad(buffer_c_im, id.xy);
    textureStore(h_hat_re, id.xy, cr);
    textureStore(h_hat_im, id.xy, ci);
}

@compute @workgroup_size(16, 16, 1)
fn copy_c_to_spectrum_p(@builtin(global_invocation_id) id: vec3u) {
    let n = sim.n;
    if (id.x >= n || id.y >= n) { return; }
    let cr = textureLoad(buffer_c_re, id.xy);
    let ci = textureLoad(buffer_c_im, id.xy);
    textureStore(p_hat_re, id.xy, cr);
    textureStore(p_hat_im, id.xy, ci);
}

@compute @workgroup_size(16, 16, 1)
fn ewave_k_step(@builtin(global_invocation_id) id: vec3u) {
    let n = sim.n;
    if (id.x >= n || id.y >= n) { return; }
    // FFT alignment: (k_x, k_y) at texel (id.x, id.y) must follow the same bin order as the
    // forward FFT output (DC at (0,0); positive then negative bins on each axis). A centered
    // index `id - N/2` would mis-assign ω(k) and break the integrator. Matches Tessendorf / FFTW style
    // as in reference eWave CPU code (e.g. `k` loop in `calc_eWave`).
    let dkx = 2.0 * PI / sim.tile_world;
    let dky = 2.0 * PI / sim.tile_world;
    let n_f = f32(n);
    let hf = 0.5 * n_f;
    let kx = select(
        (f32(id.x) - n_f) * dkx,
        f32(id.x) * dkx,
        f32(id.x) <= hf
    );
    let ky = select(
        (f32(id.y) - n_f) * dky,
        f32(id.y) * dky,
        f32(id.y) <= hf
    );
    let klen = max(length(vec2f(kx, ky)), 0.0);
    let w = sim.g * klen;
    let omega = sqrt(max(w, 0.0));
    let c = cos(omega * sim.dt);
    let s = sin(omega * sim.dt);
    var rk: f32 = 0.0;
    var rg: f32 = 0.0;
    if (klen >= 1e-8) {
        rk = (klen / max(omega, 1e-8)) * s;
        rg = (sim.g / max(omega, 1e-8)) * s;
    }
    let hre = textureLoad(h_hat_re, id.xy);
    let him = textureLoad(h_hat_im, id.xy);
    let pre = textureLoad(p_hat_re, id.xy);
    let pim = textureLoad(p_hat_im, id.xy);
    var hr0 = c * hre.x + rk * pre.x;
    var hi0 = c * him.x + rk * pim.x;
    var pr0 = c * pre.x - rg * hre.x;
    var pi0 = c * pim.x - rg * him.x;
    if (klen < 1e-8) {
        hr0 = hre.x;
        hi0 = him.x;
        pr0 = pre.x;
        pi0 = pim.x;
    }
    var hre_out = hre;
    hre_out.x = hr0;
    var him_out = him;
    him_out.x = hi0;
    var pre_out = pre;
    pre_out.x = pr0;
    var pim_out = pim;
    pim_out.x = pi0;
    textureStore(h_hat_re, id.xy, hre_out);
    textureStore(h_hat_im, id.xy, him_out);
    textureStore(p_hat_re, id.xy, pre_out);
    textureStore(p_hat_im, id.xy, pim_out);
}

@compute @workgroup_size(16, 16, 1)
fn copy_spectrum_h_to_c(@builtin(global_invocation_id) id: vec3u) {
    let n = sim.n;
    if (id.x >= n || id.y >= n) { return; }
    let cr = textureLoad(h_hat_re, id.xy);
    let ci = textureLoad(h_hat_im, id.xy);
    textureStore(buffer_c_re, id.xy, cr);
    textureStore(buffer_c_im, id.xy, ci);
}

@compute @workgroup_size(16, 16, 1)
fn copy_spectrum_p_to_c(@builtin(global_invocation_id) id: vec3u) {
    let n = sim.n;
    if (id.x >= n || id.y >= n) { return; }
    let cr = textureLoad(p_hat_re, id.xy);
    let ci = textureLoad(p_hat_im, id.xy);
    textureStore(buffer_c_re, id.xy, cr);
    textureStore(buffer_c_im, id.xy, ci);
}

@compute @workgroup_size(16, 16, 1)
fn extract_b_to_h(@builtin(global_invocation_id) id: vec3u) {
    let n = sim.n;
    if (id.x >= n || id.y >= n) { return; }
    let br = textureLoad(buffer_b_re, id.xy);
    let hp = textureLoad(h_phi, id.xy);
    textureStore(h_phi, id.xy, vec4(br.x, hp.y, 0.0, 0.0));
}

@compute @workgroup_size(16, 16, 1)
fn extract_b_to_phi(@builtin(global_invocation_id) id: vec3u) {
    let n = sim.n;
    if (id.x >= n || id.y >= n) { return; }
    let br = textureLoad(buffer_b_re, id.xy);
    let hp = textureLoad(h_phi, id.xy);
    textureStore(h_phi, id.xy, vec4(hp.x, br.x, 0.0, 0.0));
}

@compute @workgroup_size(16, 16, 1)
fn brush_stroke(@builtin(global_invocation_id) id: vec3u) {
    let n = sim.n;
    if (id.x >= n || id.y >= n) { return; }
    if (sim.brush_on == 0u) { return; }
    let p = vec2f(f32(id.x), f32(id.y));
    let d = length(p - vec2f(sim.pointer_x, sim.pointer_y));
    if (d < sim.brush_radius) {
        let hp = textureLoad(h_phi, id.xy);
        let t = 1.0 - d / max(sim.brush_radius, 1e-3);
        let bump = t * t * sim.brush_strength;
        textureStore(h_phi, id.xy, vec4(hp.x + bump, hp.y, 0.0, 0.0));
    }
}
