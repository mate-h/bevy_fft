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
    /// Multiply last frame's foam each tick; lower values shorten trails (typical 0.9 to 0.99).
    foam_trail_decay: f32,
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

/// Jacobian of horizontal displacement vs. plane coordinates; foam when folding drives J below `foam_cutoff`.
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

    let J = (1.0 + du_dx) * (1.0 + dv_dz) - du_dz * dv_dx;

    let c = params.foam_cutoff;
    let fall = max(params.foam_falloff, 1e-5);
    let instant = clamp((c - J) / fall, 0.0, 1.0);
    let prev = textureLoad(foam_prev, vec2<i32>(ix, iz)).r;
    let decay = clamp(params.foam_trail_decay, 0.0, 1.0);
    let trails = max(instant, prev * decay);

    textureStore(foam_out, vec2<i32>(ix, iz), vec4<f32>(trails, 0.0, 0.0, 1.0));
}
