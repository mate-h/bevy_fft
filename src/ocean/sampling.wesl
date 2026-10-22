// Periodic texture fetches shared by the ocean surface material and the height query pass.

fn wrap_texel(p: vec2<i32>, dims: vec2<i32>) -> vec2<i32> {
    return ((p % dims) + dims) % dims;
}

/// Bilinear fetch with periodic wrap. `uv` is in tiles, so `uv = 1` is one full period.
/// Uses `textureLoad` so float32 textures need no filterable sampler.
fn bilinear_wrap(tex: texture_2d<f32>, uv: vec2<f32>) -> vec4<f32> {
    let dims = vec2<i32>(textureDimensions(tex, 0));
    let x = uv * vec2<f32>(dims) - 0.5;
    let i0 = vec2<i32>(floor(x));
    let f = x - floor(x);
    let a = textureLoad(tex, wrap_texel(i0, dims), 0);
    let b = textureLoad(tex, wrap_texel(i0 + vec2(1, 0), dims), 0);
    let c = textureLoad(tex, wrap_texel(i0 + vec2(0, 1), dims), 0);
    let d = textureLoad(tex, wrap_texel(i0 + vec2(1, 1), dims), 0);
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}
