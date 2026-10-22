// Samples the summed ocean cascades at CPU-provided world positions for gameplay and physics.
//
// The input buffer starts with a header the main world writes each frame, then one `vec2` per query
// point. The output starts with a header that echoes the frame id, then one `vec4` per point.

import bevy_fft::ocean::sampling::bilinear_wrap;

/// Fixed-point steps that undo horizontal chop, so a query returns the surface above the point.
const INVERT_STEPS: u32 = 4u;

struct QueryHeader {
    frame: u32,
    count: u32,
    cascade_count: u32,
    _pad: u32,
    cascade_tile_sizes: vec4<f32>,
    amplitude: f32,
    choppiness: f32,
    wind_direction: f32,
    _pad2: f32,
}

struct QueryInput {
    header: QueryHeader,
    points: array<vec2<f32>>,
}

struct QueryOutput {
    /// `x` = frame id echoed from the input, `y` = number of valid samples.
    header: vec4<u32>,
    /// Per point: height in meters, height slope along X, slope along Z, chop offset in meters.
    samples: array<vec4<f32>>,
}

@group(0) @binding(0) var<storage, read> query_in: QueryInput;
@group(0) @binding(1) var<storage, read_write> query_out: QueryOutput;
@group(0) @binding(2) var displacement_0: texture_2d<f32>;
@group(0) @binding(3) var displacement_1: texture_2d<f32>;
@group(0) @binding(4) var displacement_2: texture_2d<f32>;
@group(0) @binding(5) var displacement_3: texture_2d<f32>;

fn cascade(i: u32, xz: vec2<f32>) -> vec4<f32> {
    let uv = xz / query_in.header.cascade_tile_sizes[i];
    if i == 0u {
        return bilinear_wrap(displacement_0, uv);
    } else if i == 1u {
        return bilinear_wrap(displacement_1, uv);
    } else if i == 2u {
        return bilinear_wrap(displacement_2, uv);
    }
    return bilinear_wrap(displacement_3, uv);
}

/// Same sum and scaling as `ocean_sample` in the surface material.
fn ocean_sample(xz: vec2<f32>) -> vec4<f32> {
    var sum = vec4<f32>(0.0);
    let count = min(query_in.header.cascade_count, 4u);
    for (var i = 0u; i < count; i++) {
        sum += cascade(i, xz);
    }
    let amp = query_in.header.amplitude;
    return vec4<f32>(sum.rgb * amp, sum.a * amp * query_in.header.choppiness);
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
    let wind = vec2<f32>(
        cos(query_in.header.wind_direction),
        sin(query_in.header.wind_direction),
    );
    // Find the undisplaced point whose chop lands on the target (Bowles, fixed-point iteration).
    var xz = target_xz;
    for (var step = 0u; step < INVERT_STEPS; step++) {
        let s = ocean_sample(xz);
        xz = target_xz - s.a * wind;
    }
    let s = ocean_sample(xz);
    query_out.samples[gid.x] = s;
}
