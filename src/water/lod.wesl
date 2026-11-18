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
    /// Wind direction in radians on XZ, set by the base writer.
    wind_direction: f32,
    _pad1: u32,
}

/// Placement of slice `i`, with a constant index in each branch. The helpers here take the slices
/// by value, and indexing that copy at run time makes the GPU compiler keep the whole array in
/// registers, which spilled (1.2 KiB per thread in the foam sim on an M5). Constant indices read
/// the caller's uniform directly.
fn lod_slice(lod: WaterLodSlices, i: u32) -> vec4<f32> {
    switch i {
        case 0u: { return lod.slices[0]; }
        case 1u: { return lod.slices[1]; }
        case 2u: { return lod.slices[2]; }
        case 3u: { return lod.slices[3]; }
        case 4u: { return lod.slices[4]; }
        case 5u: { return lod.slices[5]; }
        case 6u: { return lod.slices[6]; }
        case 7u: { return lod.slices[7]; }
        case 8u: { return lod.slices[8]; }
        case 9u: { return lod.slices[9]; }
        case 10u: { return lod.slices[10]; }
        default: { return lod.slices[11]; }
    }
}

/// UV of world `xz` inside slice `i`, where 0 to 1 spans the slice.
fn lod_uv(lod: WaterLodSlices, xz: vec2<f32>, i: u32) -> vec2<f32> {
    let s = lod_slice(lod, i);
    return (xz - s.xy) / s.w + 0.5;
}

/// World `xz` of texel `texel` in slice `i`.
fn lod_texel_world(lod: WaterLodSlices, texel: vec2<u32>, i: u32) -> vec2<f32> {
    let s = lod_slice(lod, i);
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

/// One where `xz` lies well inside the coarsest slice, falling to zero at its edge. Past that edge
/// the slices hold no data and a clamped fetch would repeat the border texels outward, so readers
/// fade toward a neutral value with this weight.
fn lod_coverage(lod: WaterLodSlices, xz: vec2<f32>) -> f32 {
    let s = lod_slice(lod, max(lod.count, 1u) - 1u);
    let d = max(abs(xz.x - s.x), abs(xz.y - s.y)) / (0.5 * s.w);
    return 1.0 - smoothstep(0.75, 1.0, d);
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

/// Share of each level, measured from its far end, over which `sample_lod` blends into the next
/// slice. Closer points read one slice only.
const LOD_BLEND_BAND: f32 = 0.3;

/// The slices a read at one point blends, found once and shared by every texture read there.
struct LodPick {
    /// The finer slice and its UV.
    s0: u32,
    uv0: vec2<f32>,
    /// The coarser slice and its UV, and how much of it to take.
    s1: u32,
    uv1: vec2<f32>,
    alpha: f32,
    /// `lod_coverage` at the point.
    coverage: f32,
}

/// Picks the two slices `lod_level` chooses for `xz`, blending into the next one near the far end
/// of each level.
fn lod_pick(lod: WaterLodSlices, xz: vec2<f32>, bias: f32) -> LodPick {
    var pick: LodPick;
    let level = lod_level(lod, xz, bias);
    pick.s0 = u32(floor(level));
    pick.s1 = min(pick.s0 + 1u, max(lod.count, 1u) - 1u);
    pick.alpha = select(saturate((level - floor(level) - (1.0 - LOD_BLEND_BAND)) / LOD_BLEND_BAND), 0.0, pick.s1 == pick.s0);
    pick.uv0 = lod_uv(lod, xz, pick.s0);
    pick.uv1 = lod_uv(lod, xz, pick.s1);
    pick.coverage = lod_coverage(lod, xz);
    return pick;
}

/// Reads `tex` at a pick with a bilinear fetch per slice. Slice `s` lives in layer
/// `s * layers_per_slice + layer`.
fn sample_pick(tex: texture_2d_array<f32>, pick: LodPick, layers_per_slice: u32, layer: u32) -> vec4<f32> {
    let a = bilinear_clamp_layer(tex, pick.uv0, pick.s0 * layers_per_slice + layer);
    if pick.alpha <= 0.0 {
        return a;
    }
    let b = bilinear_clamp_layer(tex, pick.uv1, pick.s1 * layers_per_slice + layer);
    return mix(a, b, pick.alpha);
}

/// Reads `tex` at a pick with one hardware bilinear fetch per slice through `samp`, which must be
/// linear and clamp to edge. Needs a filterable texture format.
fn sample_pick_filtered(tex: texture_2d_array<f32>, samp: sampler, pick: LodPick) -> vec4<f32> {
    let a = textureSampleLevel(tex, samp, pick.uv0, pick.s0, 0.0);
    if pick.alpha <= 0.0 {
        return a;
    }
    let b = textureSampleLevel(tex, samp, pick.uv1, pick.s1, 0.0);
    return mix(a, b, pick.alpha);
}

/// Samples `tex` at world `xz`, blending the two slices chosen by `lod_level` near the far end of
/// each level.
fn sample_lod(tex: texture_2d_array<f32>, lod: WaterLodSlices, xz: vec2<f32>, bias: f32) -> vec4<f32> {
    return sample_pick(tex, lod_pick(lod, xz, bias), 1u, 0u);
}

/// Same as `sample_lod`, with one hardware bilinear fetch per slice through `samp`, which must be
/// linear and clamp to edge. Needs a filterable texture format.
fn sample_lod_filtered(
    tex: texture_2d_array<f32>,
    samp: sampler,
    lod: WaterLodSlices,
    xz: vec2<f32>,
    bias: f32,
) -> vec4<f32> {
    return sample_pick_filtered(tex, samp, lod_pick(lod, xz, bias));
}

/// Samples the finest slice that contains `xz`, for physics queries that want full detail.
fn sample_lod_finest(tex: texture_2d_array<f32>, lod: WaterLodSlices, xz: vec2<f32>) -> vec4<f32> {
    return fetch_finest(tex, finest_texel(lod, xz));
}

/// The bilinear footprint of one point in the finest slice holding it, found once and shared by
/// every texture read there.
struct FinestTexel {
    layer: u32,
    /// Lower texel of the footprint and the weights toward the upper one.
    i0: vec2<i32>,
    f: vec2<f32>,
    /// Highest texel index.
    hi: vec2<i32>,
}

/// The footprint `sample_lod_finest` reads at `xz`: the finest slice that holds it with a texel
/// of margin, so the footprint stays inside the slice, or else the coarsest.
fn finest_texel(lod: WaterLodSlices, xz: vec2<f32>) -> FinestTexel {
    let count = max(lod.count, 1u);
    let margin = 1.0 / f32(max(lod.resolution, 2u));
    var layer = count - 1u;
    var uv = vec2<f32>(0.0);
    for (var i = 0u; i < count; i++) {
        uv = lod_uv(lod, xz, i);
        if all(uv > vec2(margin)) && all(uv < vec2(1.0 - margin)) {
            layer = i;
            break;
        }
    }
    let dims = vec2<i32>(i32(lod.resolution));
    let x = uv * vec2<f32>(dims) - 0.5;
    return FinestTexel(layer, vec2<i32>(floor(x)), x - floor(x), dims - 1);
}

/// Bilinear read of `tex` at a footprint from `finest_texel`.
fn fetch_finest(tex: texture_2d_array<f32>, t: FinestTexel) -> vec4<f32> {
    let a = textureLoad(tex, clamp(t.i0, vec2(0), t.hi), t.layer, 0);
    let b = textureLoad(tex, clamp(t.i0 + vec2(1, 0), vec2(0), t.hi), t.layer, 0);
    let c = textureLoad(tex, clamp(t.i0 + vec2(0, 1), vec2(0), t.hi), t.layer, 0);
    let d = textureLoad(tex, clamp(t.i0 + vec2(1, 1), vec2(0), t.hi), t.layer, 0);
    return mix(mix(a, b, t.f.x), mix(c, d, t.f.x), t.f.y);
}

/// Fixed-point steps that undo horizontal displacement in `displaced_height`.
const DISPLACED_HEIGHT_STEPS: u32 = 4u;

/// Height of the displaced surface above world `xz`, relative to the mesh plane, as the water
/// surface material's pixels draw it. `foam` holds horizontal displacement Z in its blue lane, and
/// `samp` must be linear and clamp to edge.
fn displaced_height(
    displacement: texture_2d_array<f32>,
    foam: texture_2d_array<f32>,
    samp: sampler,
    lod: WaterLodSlices,
    target_xz: vec2<f32>,
) -> f32 {
    var xz = target_xz;
    for (var step = 0u; step < DISPLACED_HEIGHT_STEPS; step++) {
        let dx = sample_lod_filtered(displacement, samp, lod, xz, 0.0).a;
        let dz = sample_lod_filtered(foam, samp, lod, xz, 0.0).b;
        xz = target_xz - vec2<f32>(dx, dz) * lod_coverage(lod, xz);
    }
    return sample_lod_filtered(displacement, samp, lod, xz, 0.0).b * lod_coverage(lod, xz);
}
