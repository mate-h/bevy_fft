// Blends the shallow water surface into every water LOD slice that overlaps the grid.
//
// Inside the grid the solver replaces the data below it. Along the grid edge it fades into that
// data over `blend_cells`, so the ocean or any other base writer shows through without a seam.
// Flow takes the solver's velocity the same way, with the vertical lane from mass conservation.
// The base data comes from a scratch copy made just before this pass.

import bevy_fft::water::lod::{WaterLodSlices, lod_texel_world};

struct ShallowWaterLodParams {
    /// World X and Z of the grid centre in meters.
    center: vec2<f32>,
    cell_size: f32,
    /// Height above the surface mesh plane of the solver's zero level, in meters.
    datum: f32,
    cells: vec2<u32>,
    blend_cells: f32,
    _pad: f32,
}

@group(0) @binding(0) var<uniform> params: ShallowWaterLodParams;
@group(0) @binding(1) var<uniform> lod: WaterLodSlices;
@group(0) @binding(2) var displacement_out: texture_storage_2d_array<rgba16float, write>;
@group(0) @binding(3) var foam_out: texture_storage_2d_array<rgba16float, write>;
@group(0) @binding(4) var flow_out: texture_storage_2d_array<rgba32float, write>;
@group(0) @binding(5) var base_displacement: texture_2d_array<f32>;
@group(0) @binding(6) var base_foam: texture_2d_array<f32>;
@group(0) @binding(7) var base_flow: texture_2d_array<f32>;
/// Bed height in `.x` and water depth in `.y`, both in meters.
@group(0) @binding(8) var bed_water: texture_2d<f32>;
/// Depth-averaged velocity at cell centres in `.xy`, in meters per second.
@group(0) @binding(9) var velocity: texture_2d<f32>;

/// Bilinear bed and depth at grid coordinate `g`, where cell `i` spans `i..i + 1`.
fn bed_water_at(g: vec2<f32>) -> vec2<f32> {
    let hi = vec2<i32>(params.cells) - 1;
    let x = g - 0.5;
    let i0 = vec2<i32>(floor(x));
    let f = x - floor(x);
    let a = textureLoad(bed_water, clamp(i0, vec2(0), hi), 0).xy;
    let b = textureLoad(bed_water, clamp(i0 + vec2(1, 0), vec2(0), hi), 0).xy;
    let c = textureLoad(bed_water, clamp(i0 + vec2(0, 1), vec2(0), hi), 0).xy;
    let d = textureLoad(bed_water, clamp(i0 + vec2(1, 1), vec2(0), hi), 0).xy;
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

/// Bilinear velocity at grid coordinate `g`.
fn velocity_at(g: vec2<f32>) -> vec2<f32> {
    let hi = vec2<i32>(params.cells) - 1;
    let x = g - 0.5;
    let i0 = vec2<i32>(floor(x));
    let f = x - floor(x);
    let a = textureLoad(velocity, clamp(i0, vec2(0), hi), 0).xy;
    let b = textureLoad(velocity, clamp(i0 + vec2(1, 0), vec2(0), hi), 0).xy;
    let c = textureLoad(velocity, clamp(i0 + vec2(0, 1), vec2(0), hi), 0).xy;
    let d = textureLoad(velocity, clamp(i0 + vec2(1, 1), vec2(0), hi), 0).xy;
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

/// Volume flux per unit width, depth times velocity, at grid coordinate `g`.
fn flux_at(g: vec2<f32>) -> vec2<f32> {
    return bed_water_at(g).y * velocity_at(g);
}

fn eta_at(g: vec2<f32>) -> f32 {
    let bw = bed_water_at(g);
    return bw.x + bw.y;
}

@compute @workgroup_size(8, 8, 1)
fn shallow_water_lod_write(@builtin(global_invocation_id) gid: vec3<u32>) {
    let res = lod.resolution;
    let slice = gid.z;
    if gid.x >= res || gid.y >= res || slice >= lod.count {
        return;
    }
    let xz = lod_texel_world(lod, gid.xy, slice);
    let n = vec2<f32>(params.cells);
    let g = (xz - params.center) / params.cell_size + 0.5 * n;
    if any(g < vec2(0.0)) || any(g > n) {
        return;
    }

    let edge = min(min(g.x, n.x - g.x), min(g.y, n.y - g.y));
    let w = smoothstep(0.0, max(params.blend_cells, 1e-3), edge);

    // Difference over at least one texel so coarse slices see the slope they can resolve.
    let span = max(lod.slices[slice].z / params.cell_size, 1.0);
    let dx = vec2<f32>(span, 0.0);
    let dz = vec2<f32>(0.0, span);
    let inv = 1.0 / (2.0 * span * params.cell_size);
    let slope = vec2<f32>(eta_at(g + dx) - eta_at(g - dx), eta_at(g + dz) - eta_at(g - dz)) * inv;
    // Mass conservation: the surface rises where flux converges.
    let rise = -((flux_at(g + dx).x - flux_at(g - dx).x) + (flux_at(g + dz).y - flux_at(g - dz).y)) * inv;
    let bw = bed_water_at(g);
    let surface = vec4<f32>(slope, params.datum + bw.x + bw.y, 0.0);

    let base = textureLoad(base_displacement, gid.xy, slice, 0);
    let base_f = textureLoad(base_foam, gid.xy, slice, 0);
    textureStore(displacement_out, gid.xy, slice, mix(base, surface, w));
    let foam = vec4<f32>(base_f.r * (1.0 - w), mix(base_f.g, bw.y, w), base_f.ba);
    textureStore(foam_out, gid.xy, slice, foam);
    let base_v = textureLoad(base_flow, gid.xy, slice, 0);
    let flow = vec4<f32>(velocity_at(g), rise, 0.0);
    textureStore(flow_out, gid.xy, slice, mix(base_v, flow, w));
}
