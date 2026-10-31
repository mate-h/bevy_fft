struct OceanFoamParams {
    texture_size: u32,
    _pad0: u32,
    tile_size: f32,
    _pad1: f32,
    amplitude: f32,
    choppiness: f32,
    wind_direction: f32,
    foam_cutoff: f32,
    foam_falloff: f32,
    /// Fraction of last frame's foam kept this frame, 0.5^(dt / half-life).
    foam_retention: f32,
    _pad2: f32,
    _pad3: f32,
    _pad4: f32,
}

@group(0) @binding(0) var<uniform> params: OceanFoamParams;
@group(0) @binding(1) var spatial_in: texture_storage_2d<rgba32float, read>;
@group(0) @binding(2) var foam_prev: texture_storage_2d<rgba32float, read>;
@group(0) @binding(3) var foam_out: texture_storage_2d<rgba32float, write>;

fn wrap_ix(i: i32, n: u32) -> i32 {
    let nn = i32(n);
    return (i % nn + nn) % nn;
}

fn load_chop(ix: i32, iz: i32) -> f32 {
    let wx = wrap_ix(ix, params.texture_size);
    let wz = wrap_ix(iz, params.texture_size);
    let s = textureLoad(spatial_in, vec2<i32>(wx, wz));
    return s.w;
}

/// Whitecaps from the horizontal displacement Jacobian. Foam starts where its smaller eigenvalue
/// drops below `foam_cutoff`, so compression along any direction counts, then fades with the
/// half-life folded into `foam_retention`.
@compute @workgroup_size(8, 8, 1)
fn ocean_jacobian_foam(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = params.texture_size;
    if gid.x >= n || gid.y >= n {
        return;
    }

    let ix = i32(gid.x);
    let iz = i32(gid.y);
    let tile = params.tile_size;
    let nf = max(f32(n) - 1.0, 1.0);
    let dscale = nf / tile;

    let dchop_dx = (load_chop(ix + 1, iz) - load_chop(ix - 1, iz)) * 0.5 * dscale;
    let dchop_dz = (load_chop(ix, iz + 1) - load_chop(ix, iz - 1)) * 0.5 * dscale;

    let wx = cos(params.wind_direction);
    let wz = sin(params.wind_direction);
    let amp_chop = params.amplitude * params.choppiness;

    let du_dx = amp_chop * wx * dchop_dx;
    let du_dz = amp_chop * wx * dchop_dz;
    let dv_dx = amp_chop * wz * dchop_dx;
    let dv_dz = amp_chop * wz * dchop_dz;

    let jxx = 1.0 + du_dx;
    let jzz = 1.0 + dv_dz;
    let half_trace = 0.5 * (jxx + jzz);
    let det = jxx * jzz - du_dz * dv_dx;
    let lambda_min = half_trace - sqrt(max(half_trace * half_trace - det, 0.0));

    let c = params.foam_cutoff;
    let fall = max(params.foam_falloff, 1e-5);
    let instant = clamp((c - lambda_min) / fall, 0.0, 1.0);
    let prev = textureLoad(foam_prev, vec2<i32>(ix, iz)).r;
    let retention = clamp(params.foam_retention, 0.0, 1.0);
    let trails = max(instant, prev * retention);

    textureStore(foam_out, vec2<i32>(ix, iz), vec4<f32>(trails, 0.0, 0.0, 1.0));
}
