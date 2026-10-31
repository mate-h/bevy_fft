// GPU CDLOD selection and draw counts. The tile header matches `surface.wesl`.
struct Params {
    eye: vec4<f32>,
    shape: vec4<f32>,
    limits: vec4<u32>,
    heights: vec4<f32>,
    planes: array<vec4<f32>, 6>,
}
struct Tile { bounds: vec4<f32>, morph: vec4<f32> }
struct Tiles {
    eye: vec4<f32>,
    count: atomic<u32>,
    cells: u32,
    enabled: u32,
    dropped: atomic<u32>,
    tiles: array<Tile>,
}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> output: Tiles;

@compute @workgroup_size(1)
fn reset() {
    output.eye = params.eye;
    atomicStore(&output.count, 0u);
    output.cells = params.limits.z;
    output.enabled = 1u;
    atomicStore(&output.dropped, 0u);
}

fn visible(minimum: vec2<f32>, size: f32) -> bool {
    if params.limits.w == 0u { return true; }
    let center = vec3<f32>(minimum.x + size * 0.5, (params.heights.x + params.heights.y) * 0.5, minimum.y + size * 0.5);
    let half_size = vec3<f32>(size * 0.5 + params.shape.w, (params.heights.y - params.heights.x) * 0.5, size * 0.5 + params.shape.w);
    for (var i = 0u; i < 6u; i++) {
        let plane = params.planes[i];
        if dot(plane.xyz, center) + plane.w + dot(abs(plane.xyz), half_size) <= 0.0 { return false; }
    }
    return true;
}

// Each root has a depth-first stack. At most three siblings per level remain pending.
@compute @workgroup_size(1)
fn refine(@builtin(global_invocation_id) id: vec3<u32>) {
    let top = params.limits.x - 1u;
    let root_size = params.shape.x * exp2(f32(top));
    let base = floor(params.eye.xz / root_size);
    let offset = vec2<f32>(f32(id.x % 3u) - 1.0, f32(id.x / 3u) - 1.0);
    var stack: array<vec4<f32>, 76>;
    stack[0] = vec4<f32>((base + offset) * root_size, root_size, f32(top));
    var pending = 1u;
    loop {
        if pending == 0u { break; }
        pending -= 1u;
        let node = stack[pending];
        let minimum = node.xy;
        let size = node.z;
        let level = u32(node.w);
        if !visible(minimum, size) { continue; }
        let nearest = clamp(params.eye.xz, minimum, minimum + size);
        let distance = length(vec3<f32>(nearest.x, 0.0, nearest.y) - params.eye.xyz);
        let end = size * params.shape.y;
        if level > 0u && distance < end * 0.5 {
            let half_size = size * 0.5;
            for (var child = 0u; child < 4u; child++) {
                let child_offset = vec2<f32>(f32(child % 2u), f32(child / 2u)) * half_size;
                stack[pending] = vec4<f32>(minimum + child_offset, half_size, f32(level - 1u));
                pending += 1u;
            }
            continue;
        }
        let slot = atomicAdd(&output.count, 1u);
        if slot >= params.limits.y {
            atomicAdd(&output.dropped, 1u);
            continue;
        }
        output.tiles[slot].bounds = vec4<f32>(minimum, size, f32(level));
        output.tiles[slot].morph = vec4<f32>(end * 0.5 * (1.0 + params.shape.z), end, 0.0, 0.0);
    }
}

@compute @workgroup_size(1)
fn finish() { atomicMin(&output.count, params.limits.y); }

struct DrawIndexed {
    index_count: u32,
    instance_count: u32,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
}
@group(1) @binding(0) var<storage, read> source_args: array<DrawIndexed>;
@group(1) @binding(1) var<storage, read_write> draw_args: array<DrawIndexed>;

// Preserve Bevy's mesh offsets and per-view instance indices. Only water uses this copy.
@compute @workgroup_size(64)
fn prepare_draws(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= arrayLength(&source_args) { return; }
    var args = source_args[id.x];
    args.index_count = atomicLoad(&output.count) * output.cells * output.cells * 6u;
    draw_args[id.x] = args;
}
