struct OceanFoamParams {
    texture_size: u32,
    /// Amplitude times choppiness.
    chop: f32,
    tile_size: f32,
    /// Scale the FFT resolve applies to the spatial output, applied here to the imaginary lane.
    normalization: f32,
}

@group(0) @binding(0) var<uniform> params: OceanFoamParams;
/// Inverse FFT result: slopes, height, and horizontal displacement X in `.w`.
@group(0) @binding(1) var spatial_in: texture_storage_2d<rgba32float, read>;
/// Imaginary half of the inverse FFT. `.w` holds horizontal displacement Z.
@group(0) @binding(2) var spatial_im: texture_storage_2d<rgba32float, read>;
@group(0) @binding(3) var foam_prev: texture_storage_2d<rgba32float, read>;
/// Horizontal displacement Z in `.g` and last frame's displacement Z in `.b`.
@group(0) @binding(4) var foam_out: texture_storage_2d<rgba32float, write>;
/// Level 0 of this cascade's Jacobian statistics: Dupuy and Bruneton's term in `.r`, its square
/// in `.g`.
@group(0) @binding(5) var jacobian_out: texture_storage_2d<rgba16float, write>;

fn wrap_ix(i: i32, n: u32) -> i32 {
    let nn = i32(n);
    return (i % nn + nn) % nn;
}

/// Unscaled horizontal displacement X and Z at a wrapped texel.
fn load_displacement(ix: i32, iz: i32) -> vec2<f32> {
    let p = vec2<i32>(wrap_ix(ix, params.texture_size), wrap_ix(iz, params.texture_size));
    let dx = textureLoad(spatial_in, p).w;
    let dz = textureLoad(spatial_im, p).w * params.normalization;
    return vec2<f32>(dx, dz);
}

/// Stores horizontal displacement Z for the LOD writer, since the FFT spatial output has room for
/// X only, and keeps last frame's Z beside it. The LOD writer finds breaking from these lanes.
/// Also writes this cascade's term of the Jacobian determinant, `k_i - 1 / n` in Dupuy and
/// Bruneton (2012, Eq. 7), from central differences, and its square.
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

    let scale = params.chop * 0.5 * f32(n) / params.tile_size;
    let d_dx = (load_displacement(p.x + 1, p.y) - load_displacement(p.x - 1, p.y)) * scale;
    let d_dz = (load_displacement(p.x, p.y + 1) - load_displacement(p.x, p.y - 1)) * scale;
    let m = d_dx.x + d_dz.y + d_dx.x * d_dz.y - d_dz.x * d_dx.y;
    textureStore(jacobian_out, p, vec4<f32>(m, m * m, 0.0, 0.0));
}
