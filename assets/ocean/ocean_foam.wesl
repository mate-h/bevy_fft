struct OceanFoamParams {
    texture_size: u32,
    _pad0: u32,
    tile_size: f32,
    /// Scale the FFT resolve applies to the spatial output, applied here to the imaginary lane.
    normalization: f32,
    amplitude: f32,
    choppiness: f32,
    foam_cutoff: f32,
    foam_falloff: f32,
    /// Fraction of last frame's foam kept this frame, 0.5^(dt / half-life).
    foam_retention: f32,
    _pad2: f32,
    _pad3: f32,
    _pad4: f32,
}

@group(0) @binding(0) var<uniform> params: OceanFoamParams;
/// Inverse FFT result: slopes, height, and horizontal displacement X in `.w`.
@group(0) @binding(1) var spatial_in: texture_storage_2d<rgba32float, read>;
/// Imaginary half of the inverse FFT. `.w` holds horizontal displacement Z.
@group(0) @binding(2) var spatial_im: texture_storage_2d<rgba32float, read>;
@group(0) @binding(3) var foam_prev: texture_storage_2d<rgba32float, read>;
/// Foam in `.r`, horizontal displacement Z in `.g`, and last frame's displacement Z in `.b`.
@group(0) @binding(4) var foam_out: texture_storage_2d<rgba32float, write>;

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

/// Whitecaps from the horizontal displacement Jacobian. Foam starts where its smaller eigenvalue
/// drops below `foam_cutoff`, so compression along any direction counts, then fades with the
/// half-life folded into `foam_retention`. The pass also stores displacement Z for the LOD writer,
/// since the FFT spatial output has room for X only.
@compute @workgroup_size(8, 8, 1)
fn ocean_jacobian_foam(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = params.texture_size;
    if gid.x >= n || gid.y >= n {
        return;
    }

    let ix = i32(gid.x);
    let iz = i32(gid.y);
    let nf = max(f32(n) - 1.0, 1.0);
    let scale = params.amplitude * params.choppiness * 0.5 * nf / params.tile_size;

    let d_dx = (load_displacement(ix + 1, iz) - load_displacement(ix - 1, iz)) * scale;
    let d_dz = (load_displacement(ix, iz + 1) - load_displacement(ix, iz - 1)) * scale;

    let jxx = 1.0 + d_dx.x;
    let jzz = 1.0 + d_dz.y;
    let half_trace = 0.5 * (jxx + jzz);
    let det = jxx * jzz - d_dz.x * d_dx.y;
    let lambda_min = half_trace - sqrt(max(half_trace * half_trace - det, 0.0));

    let c = params.foam_cutoff;
    let fall = max(params.foam_falloff, 1e-5);
    let instant = clamp((c - lambda_min) / fall, 0.0, 1.0);
    let prev = textureLoad(foam_prev, vec2<i32>(ix, iz));
    let retention = clamp(params.foam_retention, 0.0, 1.0);
    let trails = max(instant, prev.r * retention);

    let dz = load_displacement(ix, iz).y;
    textureStore(foam_out, vec2<i32>(ix, iz), vec4<f32>(trails, dz, prev.g, 1.0));
}
