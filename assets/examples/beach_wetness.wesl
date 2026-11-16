// Persistent ground moisture. Each invocation owns one solver cell.
struct Params { grid: vec4<u32>, timing: vec4<f32> }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var bed_water: texture_2d<f32>;
@group(0) @binding(2) var capillary: texture_2d<f32>;
@group(0) @binding(3) var<storage, read_write> wetness: array<f32>;

@compute @workgroup_size(8, 8)
fn update_wetness(@builtin(global_invocation_id) gid: vec3<u32>) {
    if any(gid.xy >= params.grid.xy) { return; }
    let cell = textureLoad(bed_water, vec2<i32>(gid.xy), 0).xy;
    let index = gid.y * params.grid.x + gid.x;
    let baseline = textureLoad(capillary, vec2<i32>(gid.xy), 0).r;
    let remaining = wetness[index] * exp2(-params.timing.x / params.timing.y);
    // Require a visible column so residual solver films do not keep an entire bank saturated.
    let flooded = smoothstep(0.005, 0.025, cell.y);
    wetness[index] = clamp(max(baseline, max(flooded, remaining)), 0.0, 1.0);
}
