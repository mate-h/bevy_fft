// Mip chain of each cascade's Jacobian statistics for far whitecaps.
//
// One dispatch writes five levels below `mip_in`: each 16x16 workgroup averages a 32x32 tile into
// its 16x16 texels of the first level, then halves that in workgroup memory four more times. A
// 256 texel cascade then needs two dispatches instead of eight, and the dispatches, not the
// arithmetic, are what cost. Outputs past the last level are bound to a scratch texture.

/// One level of the Jacobian statistics.
@group(0) @binding(0) var mip_in: texture_2d<f32>;
/// The five levels below it.
@group(0) @binding(1) var out_1: texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var out_2: texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var out_3: texture_storage_2d<rgba16float, write>;
@group(0) @binding(4) var out_4: texture_storage_2d<rgba16float, write>;
@group(0) @binding(5) var out_5: texture_storage_2d<rgba16float, write>;

var<workgroup> tile: array<vec4<f32>, 256>;

/// Mean of the 2x2 tile texels under texel `lid` of the next level.
fn tile_mean(lid: vec2<u32>) -> vec4<f32> {
    let i = lid.y * 32u + lid.x * 2u;
    return 0.25 * (tile[i] + tile[i + 1u] + tile[i + 16u] + tile[i + 17u]);
}

/// Halves the tile's level of width `2 w` into width `w`, leaving it in the tile's corner, and
/// returns this thread's texel of it, or zero for threads outside it.
fn halve(lid: vec2<u32>, w: u32) -> vec4<f32> {
    let inside = all(lid < vec2<u32>(w));
    var v = vec4<f32>(0.0);
    if inside {
        v = tile_mean(lid);
    }
    workgroupBarrier();
    if inside {
        tile[lid.y * 16u + lid.x] = v;
    }
    workgroupBarrier();
    return v;
}

/// Averages 2x2 texels of each level into the next. Means of the term and of its square both
/// filter linearly.
@compute @workgroup_size(16, 16, 1)
fn ocean_jacobian_downsample(
    @builtin(local_invocation_id) local: vec3<u32>,
    @builtin(workgroup_id) group: vec3<u32>,
) {
    let lid = local.xy;
    let p = group.xy * 16u + lid;
    let last = vec2<i32>(textureDimensions(mip_in)) - 1;
    let s = vec2<i32>(p * 2u);
    let v = 0.25 * (textureLoad(mip_in, min(s, last), 0)
        + textureLoad(mip_in, min(s + vec2(1, 0), last), 0)
        + textureLoad(mip_in, min(s + vec2(0, 1), last), 0)
        + textureLoad(mip_in, min(s + vec2(1, 1), last), 0));
    if all(p < textureDimensions(out_1)) {
        textureStore(out_1, p, v);
    }
    tile[lid.y * 16u + lid.x] = v;
    workgroupBarrier();

    let v2 = halve(lid, 8u);
    let p2 = group.xy * 8u + lid;
    if all(lid < vec2(8u)) && all(p2 < textureDimensions(out_2)) {
        textureStore(out_2, p2, v2);
    }
    let v3 = halve(lid, 4u);
    let p3 = group.xy * 4u + lid;
    if all(lid < vec2(4u)) && all(p3 < textureDimensions(out_3)) {
        textureStore(out_3, p3, v3);
    }
    let v4 = halve(lid, 2u);
    let p4 = group.xy * 2u + lid;
    if all(lid < vec2(2u)) && all(p4 < textureDimensions(out_4)) {
        textureStore(out_4, p4, v4);
    }
    let v5 = halve(lid, 1u);
    if all(lid == vec2(0u)) && all(group.xy < textureDimensions(out_5)) {
        textureStore(out_5, group.xy, v5);
    }
}
