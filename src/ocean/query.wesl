// Samples the water LOD data at CPU-provided world positions for gameplay and physics.
//
// The input buffer starts with a header the main world writes each frame, then one `vec2` per query
// point. The output starts with a header that echoes the frame id, then one `QuerySample` per point.

import bevy_fft::water::lod::{WaterLodSlices, sample_lod_finest};

/// Fixed-point steps that undo horizontal displacement, so a query returns the surface above the point.
const INVERT_STEPS: u32 = 4u;

struct QueryHeader {
    frame: u32,
    count: u32,
    _pad0: u32,
    _pad1: u32,
}

struct QueryInput {
    header: QueryHeader,
    points: array<vec2<f32>>,
}

struct QuerySample {
    /// Height in meters, height slope along X, slope along Z, horizontal displacement X in meters.
    surface: vec4<f32>,
    /// Velocity in m/s along X, along Z, and vertical, then horizontal displacement Z in meters.
    flow: vec4<f32>,
}

struct QueryOutput {
    /// `x` = frame id echoed from the input, `y` = number of valid samples.
    header: vec4<u32>,
    samples: array<QuerySample>,
}

@group(0) @binding(0) var<storage, read> query_in: QueryInput;
@group(0) @binding(1) var<storage, read_write> query_out: QueryOutput;
@group(0) @binding(2) var lod_displacement: texture_2d_array<f32>;
@group(0) @binding(3) var<uniform> lod: WaterLodSlices;
@group(0) @binding(4) var lod_flow: texture_2d_array<f32>;
@group(0) @binding(5) var lod_foam: texture_2d_array<f32>;

/// Finest available LOD data at `xz`, in the lane order the surface uses
/// (slope x, slope z, height, displacement x).
fn ocean_sample(xz: vec2<f32>) -> vec4<f32> {
    return sample_lod_finest(lod_displacement, lod, xz);
}

/// Horizontal displacement at `xz`. Z lives in the foam texture's blue lane.
fn ocean_horizontal(xz: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(ocean_sample(xz).a, sample_lod_finest(lod_foam, lod, xz).b);
}

@compute @workgroup_size(64, 1, 1)
fn ocean_query(@builtin(global_invocation_id) gid: vec3<u32>) {
    let count = query_in.header.count;
    if gid.x == 0u {
        query_out.header = vec4<u32>(query_in.header.frame, count, 0u, 0u);
    }
    if gid.x >= count || gid.x >= arrayLength(&query_out.samples) {
        return;
    }
    let target_xz = query_in.points[gid.x];
    // Find the undisplaced point whose displacement lands on the target (Bowles, fixed-point iteration).
    var xz = target_xz;
    for (var step = 0u; step < INVERT_STEPS; step++) {
        xz = target_xz - ocean_horizontal(xz);
    }
    // `ocean_sample` keeps the FFT lane order (slope x, slope z, height, displacement x). The
    // output puts height first, matching `OceanSample` on the CPU.
    let s = ocean_sample(xz);
    let dz = sample_lod_finest(lod_foam, lod, xz).b;
    let flow = sample_lod_finest(lod_flow, lod, xz);
    query_out.samples[gid.x] = QuerySample(vec4<f32>(s.b, s.r, s.g, s.a), vec4<f32>(flow.xyz, dz));
}
