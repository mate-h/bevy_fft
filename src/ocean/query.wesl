// Samples the water LOD data at CPU-provided world positions for gameplay and physics.
//
// The input buffer starts with a header the main world writes each frame, then one `vec2` per query
// point. The output starts with a header that echoes the frame id, then one `QuerySample` per point.

import bevy_fft::water::lod::{WaterLodSlices, lod_chop_dir, sample_lod_finest};

/// Fixed-point steps that undo horizontal chop, so a query returns the surface above the point.
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
    /// Height in meters, height slope along X, slope along Z, chop offset in meters.
    surface: vec4<f32>,
    /// Velocity in m/s along X, along Z, and vertical.
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

/// Finest available LOD data at `xz`, in the lane order the surface uses
/// (slope x, slope z, height, chop).
fn ocean_sample(xz: vec2<f32>) -> vec4<f32> {
    return sample_lod_finest(lod_displacement, lod, xz);
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
    let chop_dir = lod_chop_dir(lod);
    // Find the undisplaced point whose chop lands on the target (Bowles, fixed-point iteration).
    var xz = target_xz;
    for (var step = 0u; step < INVERT_STEPS; step++) {
        let s = ocean_sample(xz);
        xz = target_xz - s.a * chop_dir;
    }
    // `ocean_sample` keeps the FFT lane order (slope x, slope z, height, chop). The output puts
    // height first, matching `OceanSample` on the CPU.
    let s = ocean_sample(xz);
    let flow = sample_lod_finest(lod_flow, lod, xz);
    query_out.samples[gid.x] = QuerySample(vec4<f32>(s.b, s.r, s.g, s.a), vec4<f32>(flow.xyz, 0.0));
}
