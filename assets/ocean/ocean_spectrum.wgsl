#import bevy_fft::bindings::{
    settings,
    buffer_c_re,
    buffer_c_im,
};
#import bevy_fft::complex::c32_4;

const PI: f32 = 3.14159265359;

struct OceanDynamicParams {
    texture_size: u32,
    _pad0: u32,
    tile_size: f32,
    elapsed_seconds: f32,
    gravity: f32,
    wind_direction: f32,
    _pad2: f32,
    _pad3: f32,
}

@group(1) @binding(0) var<uniform> ocean_dyn: OceanDynamicParams;
@group(2) @binding(0) var H0_src: texture_storage_2d<rgba32float, read>;

fn omega_deep_water(k_len: f32, g: f32) -> f32 {
    return sqrt(max(k_len, 0.0) * g);
}

fn complex_mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

/// Complex height `h(k, t)` at `ip`.
fn ocean_htilde(ip: vec2<i32>, k_vec: vec2<f32>) -> vec2<f32> {
    let h0_tex = textureLoad(H0_src, ip);
    let h0_k = h0_tex.xy;
    let h0_mk_conj = h0_tex.zw;

    let w = omega_deep_water(length(k_vec), ocean_dyn.gravity);
    let th = ocean_dyn.elapsed_seconds * w;
    let ep = vec2(cos(th), sin(th));

    let t1 = complex_mul(h0_k, ep);
    let t2 = complex_mul(h0_mk_conj, vec2(ep.x, -ep.y));
    return t1 + t2;
}

/// Packs four bands into buffer **C** for one inverse FFT. Lanes 0 and 1 are height slopes, lane 2 is `h`, lane 3 is
/// wind-aligned chop from `(k dot w_hat)/|k|` and `i*h` with `w_hat` on XZ. Chop shares the `i*h` factor used for slopes so it
/// stays in phase with height. Full 2D horizontal displacement would need another band or pass.
@compute @workgroup_size(8, 8, 1)
fn ocean_fill_spectrum_c(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = ocean_dyn.texture_size;
    if gid.x >= n || gid.y >= n {
        return;
    }

    let dims = settings.size;
    if gid.x >= dims.x || gid.y >= dims.y {
        return;
    }

    let pos = gid.xy;
    let ip = vec2<i32>(i32(pos.x), i32(pos.y));

    let delta_k = 2.0 * PI / ocean_dyn.tile_size;
    let nx = f32(i32(pos.x) - i32(n / 2u));
    let nz = f32(i32(pos.y) - i32(n / 2u));
    let k_vec = vec2<f32>(nx, nz) * delta_k;
    let k_len = length(k_vec) + 0.001;

    let h = ocean_htilde(ip, k_vec);

    let ih = vec2(-h.y, h.x);
    let d_eta_dx_k = complex_mul(vec2<f32>(k_vec.x, 0.0), ih);
    let d_eta_dz_k = complex_mul(vec2<f32>(k_vec.y, 0.0), ih);
    let w_hat = vec2(cos(ocean_dyn.wind_direction), sin(ocean_dyn.wind_direction));
    let k_proj = dot(k_vec, w_hat);
    // `ih` is `i*h` (see slope lanes). Use it for chop too so horizontal displacement matches η in phase.
    let chop_wind_k = complex_mul(vec2(k_proj / k_len, 0.0), ih);

    let pack = c32_4(
        vec4<f32>(d_eta_dx_k.x, d_eta_dz_k.x, h.x, chop_wind_k.x),
        vec4<f32>(d_eta_dx_k.y, d_eta_dz_k.y, h.y, chop_wind_k.y),
    );

    let to_ifft = vec2<u32>((pos.x + n / 2u) % n, (pos.y + n / 2u) % n);

    textureStore(buffer_c_re, to_ifft, pack.re);
    textureStore(buffer_c_im, to_ifft, pack.im);
}
