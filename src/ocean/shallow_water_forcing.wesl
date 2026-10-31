// Writes the ocean's long-wave surface height over a placed shallow water grid.
//
// The shallow water ocean band and PML blend toward this height, so waves from the FFT ocean enter
// the grid. Only cascades the grid's cells can resolve contribute, using the same weight the LOD
// writer applies per slice. The chop is undone with a few fixed-point steps, so each cell gets the
// surface height above its centre rather than at its undisplaced point. Each frame writes the edge
// strip the band reads. On reset `full_grid` writes every cell so the solver can seed the interior.

import bevy_fft::ocean::sampling::bilinear_wrap;

const INVERT_STEPS: u32 = 3u;

struct ShallowWaterForcingParams {
    /// World tile edge in meters for cascades 0 to 3.
    cascade_tile_sizes: vec4<f32>,
    /// World X and Z of the grid centre in meters.
    grid_center: vec2<f32>,
    /// Unit horizontal direction of the chop lane.
    chop_dir: vec2<f32>,
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

/// Matches `cascade_weight` in the ocean LOD writer, with the grid cell as the texel.
fn cascade_weight(i: u32) -> f32 {
    let tile = params.cascade_tile_sizes[i];
    return 1.0 - smoothstep(tile / 24.0, tile / 8.0, params.cell_size);
}

/// Height and chop in meters at undisplaced point `xz`.
fn height_and_chop(xz: vec2<f32>) -> vec2<f32> {
    var sum = vec2<f32>(0.0);
    let count = min(params.cascade_count, 4u);
    for (var i = 0u; i < count; i++) {
        let w = cascade_weight(i);
        if w > 0.0 {
            let d = displacement_cascade(i, xz / params.cascade_tile_sizes[i]);
            sum += vec2<f32>(d.b, d.a) * w;
        }
    }
    return sum * vec2<f32>(params.amplitude, params.amplitude * params.choppiness);
}

@compute @workgroup_size(16, 16, 1)
fn shallow_water_forcing(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= params.cells.x || gid.y >= params.cells.y {
        return;
    }
    if params.full_grid == 0u {
        let edge = min(min(gid.x, params.cells.x - 1u - gid.x), min(gid.y, params.cells.y - 1u - gid.y));
        if edge >= params.edge_cells {
            return;
        }
    }
    let cell = vec2<f32>(gid.xy) + 0.5 - 0.5 * vec2<f32>(params.cells);
    let target_xz = params.grid_center + cell * params.cell_size;
    var xz = target_xz;
    for (var step = 0u; step < INVERT_STEPS; step++) {
        xz = target_xz - height_and_chop(xz).y * params.chop_dir;
    }
    let eta = height_and_chop(xz).x - params.datum;
    textureStore(eta_out, gid.xy, vec4<f32>(eta, 0.0, 0.0, 0.0));
}
