// Water LOD data: camera-centred texture array slices shared by every water solver and reader.
//
// Slice `i` is a square of `resolution` texels with texel width `slices[i].z`, centred at
// `slices[i].xy` (world X and Z). Each slice covers twice the width of the previous one. Readers
// pick two neighbouring slices by taxicab distance from the slice-0 centre and blend them, so the
// data never pops between resolutions.

const WATER_LOD_MAX_SLICES: u32 = 12u;

struct WaterLodSlices {
    /// Per slice: snapped centre X, snapped centre Z, texel width in meters, slice width in meters.
    slices: array<vec4<f32>, 12>,
    count: u32,
    resolution: u32,
    _pad0: u32,
    _pad1: u32,
}

/// UV of world `xz` inside slice `i`, where 0 to 1 spans the slice.
fn lod_uv(lod: WaterLodSlices, xz: vec2<f32>, i: u32) -> vec2<f32> {
    let s = lod.slices[i];
    return (xz - s.xy) / s.w + 0.5;
}

/// World `xz` of texel `texel` in slice `i`.
fn lod_texel_world(lod: WaterLodSlices, texel: vec2<u32>, i: u32) -> vec2<f32> {
    let s = lod.slices[i];
    return s.xy + (vec2<f32>(texel) + 0.5 - f32(lod.resolution) * 0.5) * s.z;
}

/// Continuous slice coordinate for `xz`: the integer part picks a slice, the fraction blends to
/// the next one. `bias` adds whole slices, for example to keep vertices on coarser data than pixels.
fn lod_level(lod: WaterLodSlices, xz: vec2<f32>, bias: f32) -> f32 {
    let c = lod.slices[0];
    let d = max(abs(xz.x - c.x), abs(xz.y - c.y));
    // A point at taxicab distance d sits well inside slice s when d <= slice_width(s) / 4.
    let r0 = c.w * 0.25;
    let level = max(log2(max(d, 1e-6) / r0), 0.0) + bias;
    return clamp(level, 0.0, f32(max(lod.count, 1u) - 1u));
}

/// Bilinear fetch from one array layer with clamp to edge. Uses `textureLoad`, so float32 arrays
/// need no filterable sampler.
fn bilinear_clamp_layer(tex: texture_2d_array<f32>, uv: vec2<f32>, layer: u32) -> vec4<f32> {
    let dims = vec2<i32>(textureDimensions(tex, 0));
    let x = uv * vec2<f32>(dims) - 0.5;
    let i0 = vec2<i32>(floor(x));
    let f = x - floor(x);
    let hi = dims - 1;
    let a = textureLoad(tex, clamp(i0, vec2(0), hi), layer, 0);
    let b = textureLoad(tex, clamp(i0 + vec2(1, 0), vec2(0), hi), layer, 0);
    let c = textureLoad(tex, clamp(i0 + vec2(0, 1), vec2(0), hi), layer, 0);
    let d = textureLoad(tex, clamp(i0 + vec2(1, 1), vec2(0), hi), layer, 0);
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

/// Samples `tex` at world `xz`, blending the two slices chosen by `lod_level`.
fn sample_lod(tex: texture_2d_array<f32>, lod: WaterLodSlices, xz: vec2<f32>, bias: f32) -> vec4<f32> {
    let level = lod_level(lod, xz, bias);
    let s0 = u32(floor(level));
    let s1 = min(s0 + 1u, max(lod.count, 1u) - 1u);
    let alpha = level - floor(level);
    let a = bilinear_clamp_layer(tex, lod_uv(lod, xz, s0), s0);
    if alpha <= 0.0 || s1 == s0 {
        return a;
    }
    let b = bilinear_clamp_layer(tex, lod_uv(lod, xz, s1), s1);
    return mix(a, b, alpha);
}

/// Samples the finest slice that contains `xz`, for physics queries that want full detail.
fn sample_lod_finest(tex: texture_2d_array<f32>, lod: WaterLodSlices, xz: vec2<f32>) -> vec4<f32> {
    let count = max(lod.count, 1u);
    for (var i = 0u; i < count; i++) {
        let uv = lod_uv(lod, xz, i);
        // Keep one texel of margin so the bilinear footprint stays inside the slice.
        let margin = 1.0 / f32(max(lod.resolution, 2u));
        if all(uv > vec2(margin)) && all(uv < vec2(1.0 - margin)) {
            return bilinear_clamp_layer(tex, uv, i);
        }
    }
    return bilinear_clamp_layer(tex, lod_uv(lod, xz, count - 1u), count - 1u);
}
