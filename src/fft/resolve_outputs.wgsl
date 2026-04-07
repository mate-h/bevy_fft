#define_import_path bevy_fft::resolve_outputs

// Mirror `FftSettings` from `bindings.wgsl` whenever the uniform changes.
struct FftSettings {
    size: vec2<u32>,
    orders: u32,
    padding: vec2<u32>,
    schedule: u32,
    pattern_target: u32,
    window_type: u32,
    window_strength: f32,
    radial_falloff: f32,
    normalization: f32,
}

@group(0) @binding(0) var<uniform> settings: FftSettings;
@group(0) @binding(1) var spectrum_c_re: texture_storage_2d<rgba32float, read>;
@group(0) @binding(2) var spectrum_c_im: texture_storage_2d<rgba32float, read>;
@group(0) @binding(3) var spatial_b_re: texture_storage_2d<rgba32float, read>;
@group(0) @binding(4) var power_spectrum_out: texture_storage_2d<rgba32float, write>;
@group(0) @binding(5) var spatial_output_out: texture_storage_2d<rgba32float, write>;

// fftshifted log magnitude for RGB, plus a copy of the spatial preview.
@compute
@workgroup_size(16, 16, 1)
fn resolve_fft_outputs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = settings.size;
    let pos = gid.xy;
    if (pos.x >= dims.x || pos.y >= dims.y) {
        return;
    }

    let ip = vec2<i32>(i32(pos.x), i32(pos.y));

    let spatial = textureLoad(spatial_b_re, ip);
    let s = vec4<f32>(spatial.xyz * settings.normalization, spatial.w);
    textureStore(spatial_output_out, pos, s);

    let hx = dims.x >> 1u;
    let hy = dims.y >> 1u;
    let sp = vec2<u32>((pos.x + hx) % dims.x, (pos.y + hy) % dims.y);
    let isp = vec2<i32>(i32(sp.x), i32(sp.y));

    let cre = textureLoad(spectrum_c_re, isp);
    let cim = textureLoad(spectrum_c_im, isp);

    var mag = vec3<f32>(0.0);
    for (var ch = 0u; ch < 3u; ch++) {
        let r = cre[ch];
        let i = cim[ch];
        mag[ch] = sqrt(r * r + i * i);
    }

    let gain = 0.12;
    var c = vec3<f32>(
        log(1.0 + gain * mag.x),
        log(1.0 + gain * mag.y),
        log(1.0 + gain * mag.z),
    );
    let mx = max(c.x, max(c.y, c.z));
    if (mx > 1e-8) {
        c = c / mx;
    } else {
        c = vec3<f32>(0.0);
    }
    textureStore(power_spectrum_out, pos, vec4<f32>(c, 1.0));
}
