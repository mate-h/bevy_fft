struct OceanFoamParams {
    texture_size: u32,
    _pad0: u32,
    tile_size: f32,
    /// Scale the FFT resolve applies to the spatial output, applied here to the imaginary lane.
    normalization: f32,
}

@group(0) @binding(0) var<uniform> params: OceanFoamParams;
/// Inverse FFT result. Bound for the shared layout and not read here.
@group(0) @binding(1) var spatial_in: texture_storage_2d<rgba32float, read>;
/// Imaginary half of the inverse FFT. `.w` holds horizontal displacement Z.
@group(0) @binding(2) var spatial_im: texture_storage_2d<rgba32float, read>;
@group(0) @binding(3) var foam_prev: texture_storage_2d<rgba32float, read>;
/// Horizontal displacement Z in `.g` and last frame's displacement Z in `.b`.
@group(0) @binding(4) var foam_out: texture_storage_2d<rgba32float, write>;

/// Stores horizontal displacement Z for the LOD writer, since the FFT spatial output has room for
/// X only, and keeps last frame's Z beside it. The LOD writer finds breaking from these lanes.
@compute @workgroup_size(8, 8, 1)
fn ocean_jacobian_foam(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = params.texture_size;
    if gid.x >= n || gid.y >= n {
        return;
    }
    let p = vec2<i32>(gid.xy);
    let dz = textureLoad(spatial_im, p).w * params.normalization;
    let prev = textureLoad(foam_prev, p);
    textureStore(foam_out, p, vec4<f32>(0.0, dz, prev.g, 1.0));
}
