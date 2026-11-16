// Writes the ocean's long-wave surface height over a placed shallow water grid.
//
// The shallow water ocean band and PML blend toward this height, so waves from the FFT ocean enter
// the grid. Only cascades the grid's cells can resolve contribute, using the same weight the LOD
// writer applies per slice. The horizontal displacement is undone with a few fixed-point steps, so each cell gets the
// surface height above its centre rather than at its undisplaced point. Each frame writes the edge
// strip the band reads. On reset `full_grid` writes every cell so the solver can seed the interior.

import bevy_fft::ocean::sampling::bilinear_wrap;

const INVERT_STEPS: u32 = 3u;

struct ShallowWaterForcingParams {
    /// World tile edge in meters for cascades 0 to 3.
    cascade_tile_sizes: vec4<f32>,
    /// World X and Z of the grid centre in meters.
    grid_center: vec2<f32>,
    cells: vec2<u32>,
    cell_size: f32,
    /// Height of the solver's zero level above the surface mesh plane, in meters.
    datum: f32,
    amplitude: f32,
    choppiness: f32,
    cascade_count: u32,
    /// Width in cells of the edge strip that gets a target. Ignored when `full_grid` is set.
    edge_cells: u32,
    /// Nonzero writes every cell. Zero writes the edge strip only.
    full_grid: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> params: ShallowWaterForcingParams;
@group(0) @binding(1) var eta_out: texture_storage_2d<r32float, write>;
@group(0) @binding(2) var displacement_0: texture_2d<f32>;
@group(0) @binding(3) var displacement_1: texture_2d<f32>;
@group(0) @binding(4) var displacement_2: texture_2d<f32>;
@group(0) @binding(5) var displacement_3: texture_2d<f32>;
/// Ocean foam textures, whose `.g` holds horizontal displacement Z.
@group(0) @binding(6) var foam_0: texture_2d<f32>;
@group(0) @binding(7) var foam_1: texture_2d<f32>;
@group(0) @binding(8) var foam_2: texture_2d<f32>;
@group(0) @binding(9) var foam_3: texture_2d<f32>;

fn displacement_cascade(i: u32, uv: vec2<f32>) -> vec4<f32> {
    if i == 0u {
        return bilinear_wrap(displacement_0, uv);
    } else if i == 1u {
        return bilinear_wrap(displacement_1, uv);
    } else if i == 2u {
        return bilinear_wrap(displacement_2, uv);
    }
    return bilinear_wrap(displacement_3, uv);
}

fn displacement_z_cascade(i: u32, uv: vec2<f32>) -> f32 {
    if i == 0u {
        return bilinear_wrap(foam_0, uv).g;
    } else if i == 1u {
        return bilinear_wrap(foam_1, uv).g;
    } else if i == 2u {
        return bilinear_wrap(foam_2, uv).g;
    }
    return bilinear_wrap(foam_3, uv).g;
}

/// Matches `cascade_weight` in the ocean LOD writer, with the grid cell as the texel.
fn cascade_weight(i: u32) -> f32 {
    let tile = params.cascade_tile_sizes[i];
    return 1.0 - smoothstep(tile / 24.0, tile / 8.0, params.cell_size);
}

/// Height in `.x` and horizontal displacement X and Z in `.yz`, in meters, at undisplaced point `xz`.
fn height_and_displacement(xz: vec2<f32>) -> vec3<f32> {
    var sum = vec3<f32>(0.0);
    let count = min(params.cascade_count, 4u);
    for (var i = 0u; i < count; i++) {
        let w = cascade_weight(i);
        if w > 0.0 {
            let uv = xz / params.cascade_tile_sizes[i];
            let d = displacement_cascade(i, uv);
            sum += vec3<f32>(d.b, d.a, displacement_z_cascade(i, uv)) * w;
        }
    }
    let chop = params.amplitude * params.choppiness;
    return sum * vec3<f32>(params.amplitude, chop, chop);
}

/// With `full_grid` the dispatch covers the grid. Otherwise `gid.z` picks one of the four edge
/// bands, `gid.x` runs along it and `gid.y` inward. The corners belong to two bands and get the
/// same value twice.
@compute @workgroup_size(16, 16, 1)
fn shallow_water_forcing(@builtin(global_invocation_id) gid: vec3<u32>) {
    var id = gid.xy;
    if params.full_grid == 0u {
        let along = gid.x;
        let inward = gid.y;
        if inward >= params.edge_cells || inward >= min(params.cells.x, params.cells.y) {
            return;
        }
        switch gid.z {
            case 0u: { id = vec2<u32>(along, inward); }
            case 1u: { id = vec2<u32>(along, params.cells.y - 1u - inward); }
            case 2u: { id = vec2<u32>(inward, along); }
            default: { id = vec2<u32>(params.cells.x - 1u - inward, along); }
        }
    }
    if id.x >= params.cells.x || id.y >= params.cells.y {
        return;
    }
    let cell = vec2<f32>(id) + 0.5 - 0.5 * vec2<f32>(params.cells);
    let target_xz = params.grid_center + cell * params.cell_size;
    var xz = target_xz;
    for (var step = 0u; step < INVERT_STEPS; step++) {
        xz = target_xz - height_and_displacement(xz).yz;
    }
    let eta = height_and_displacement(xz).x - params.datum;
    textureStore(eta_out, id, vec4<f32>(eta, 0.0, 0.0, 0.0));
}
