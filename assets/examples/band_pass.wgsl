#import bevy_fft::bindings::{
    settings,
    buffer_c_re,
    buffer_c_im,
};

struct BandPassParams {
    band_center: f32,
    band_width: f32,
}

@group(1) @binding(0) var<uniform> band_pass: BandPassParams;

/// Radial band-pass on spectrum buffer **C**. `r_norm` is folded distance from **DC** at (0,0)
/// in unshifted FFT layout (matches the fftshifted power-spectrum preview).
@compute
@workgroup_size(16, 16, 1)
fn radial_band_pass(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pos = gid.xy;
    let dims = settings.size;
    if (pos.x >= dims.x || pos.y >= dims.y) {
        return;
    }

    let ip = vec2<i32>(i32(pos.x), i32(pos.y));
    var cre = textureLoad(buffer_c_re, ip);
    var cim = textureLoad(buffer_c_im, ip);

    let ku = min(pos.x, dims.x - pos.x);
    let kv = min(pos.y, dims.y - pos.y);
    let r = length(vec2<f32>(f32(ku), f32(kv)));
    let scale = 0.5 * length(vec2<f32>(f32(dims.x), f32(dims.y)));
    let r_norm = r / max(scale, 1e-6);

    let half = max(band_pass.band_width, 0.0) * 0.5;
    let c = clamp(band_pass.band_center, 0.0, 1.0);
    let low = clamp(c - half, 0.0, 1.0);
    let high = clamp(c + half, 0.0, 1.0);
    let edge = 0.05;
    let w = smoothstep(low - edge, low + edge, r_norm) * (1.0 - smoothstep(high - edge, high + edge, r_norm));

    // Avoid `0.0 * inf -> NaN` if upstream spectrum has non-finite bins; zero passband must hard-clear.
    // (`pass` is reserved in WGSL.)
    let in_band = w > 0.0;
    cre = select(vec4<f32>(0.0), cre, in_band) * select(0.0, w, in_band);
    cim = select(vec4<f32>(0.0), cim, in_band) * select(0.0, w, in_band);
    textureStore(buffer_c_re, pos, cre);
    textureStore(buffer_c_im, pos, cim);
}
