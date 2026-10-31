// Periodic texture fetches shared by the ocean surface material and the height query pass.

fn wrap_texel(p: vec2<i32>, dims: vec2<i32>) -> vec2<i32> {
    return ((p % dims) + dims) % dims;
}

/// Wrapped corner texels and weights of one bilinear fetch, shareable between same-sized textures.
struct WrapTaps {
    i0: vec2<i32>,
    i1: vec2<i32>,
    f: vec2<f32>,
}

/// Taps for a texture of size `dims` at `uv` in tiles, so `uv = 1` is one full period.
fn wrap_taps(dims: vec2<i32>, uv: vec2<f32>) -> WrapTaps {
    let x = uv * vec2<f32>(dims) - 0.5;
    let base = floor(x);
    let i0 = wrap_texel(vec2<i32>(base), dims);
    let next = i0 + vec2<i32>(1);
    return WrapTaps(i0, select(next, vec2<i32>(0), next >= dims), x - base);
}

/// Bilinear fetch at precomputed taps. Uses `textureLoad` so float32 textures need no filterable sampler.
fn bilinear_taps(tex: texture_2d<f32>, t: WrapTaps) -> vec4<f32> {
    let a = textureLoad(tex, t.i0, 0);
    let b = textureLoad(tex, vec2<i32>(t.i1.x, t.i0.y), 0);
    let c = textureLoad(tex, vec2<i32>(t.i0.x, t.i1.y), 0);
    let d = textureLoad(tex, t.i1, 0);
    return mix(mix(a, b, t.f.x), mix(c, d, t.f.x), t.f.y);
}

/// Bilinear fetch with periodic wrap. `uv` is in tiles, so `uv = 1` is one full period.
fn bilinear_wrap(tex: texture_2d<f32>, uv: vec2<f32>) -> vec4<f32> {
    return bilinear_taps(tex, wrap_taps(vec2<i32>(textureDimensions(tex, 0)), uv));
}
