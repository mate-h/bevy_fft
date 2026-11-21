// GPU CDLOD selection and draw counts. The tile header matches `surface.wesl`.
//
// With a dry depth set, selection also drops tiles with no water under them. `wet_mask` first
// reduces each LOD slice's water depth, the green lane of the LOD foam array, to its maximum over
// blocks of `WET_BLOCK` texels and builds a max pyramid over the blocks. `refine` then skips a node
// when every cell under it, widened by the horizontal margin, is shallower than the dry depth, so
// the surface would discard all of its pixels anyway. Nodes outside every slice count as wet, as
// the surface draws open water there.
import bevy_fft::water::lod::WaterLodSlices;

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
@group(0) @binding(2) var<uniform> lod: WaterLodSlices;
/// LOD foam array, whose green lane holds the water depth in meters.
@group(0) @binding(3) var lod_foam: texture_2d_array<f32>;
/// Deepest water per block of `WET_BLOCK` texels, then per 2x2, 4x4 and so on up to one cell per
/// slice: slice by slice, level by level, row by row. See `wet_level_offset`.
@group(0) @binding(4) var<storage, read_write> wet: array<f32>;

/// LOD texels per side of one `wet` block.
const WET_BLOCK: u32 = 8u;
/// Threads per side of the `wet_mask` workgroup, one workgroup per slice.
const WET_GROUP: u32 = 16u;

/// Blocks per side of a slice's finest `wet` level, a power of two since LOD resolutions are.
fn wet_blocks() -> u32 {
    return max(lod.resolution / WET_BLOCK, 1u);
}

/// Levels of a slice's `wet` pyramid, down to one cell.
fn wet_levels() -> u32 {
    return firstLeadingBit(wet_blocks()) + 1u;
}

/// Index in `wet` of level `level`, cell (0, 0), of slice `slice`. With `b` blocks per side, a
/// power of two, the levels before `level` hold 4 (b^2 - (b >> level)^2) / 3 cells and a whole
/// slice (4 b^2 - 1) / 3.
fn wet_level_offset(slice: u32, level: u32) -> u32 {
    let b = wet_blocks();
    let n = b >> level;
    return slice * ((4u * b * b - 1u) / 3u) + 4u * (b * b - n * n) / 3u;
}

/// One workgroup per slice: the deepest water per block, then the max pyramid over it.
@compute @workgroup_size(16, 16, 1)
fn wet_mask(
    @builtin(local_invocation_id) local: vec3<u32>,
    @builtin(workgroup_id) group: vec3<u32>,
) {
    let slice = group.z;
    if slice >= lod.count { return; }
    let blocks = wet_blocks();
    let base = wet_level_offset(slice, 0u);
    for (var by = local.y; by < blocks; by += WET_GROUP) {
        for (var bx = local.x; bx < blocks; bx += WET_GROUP) {
            var deepest = 0.0;
            for (var y = 0u; y < WET_BLOCK; y++) {
                for (var x = 0u; x < WET_BLOCK; x++) {
                    let texel = vec2<i32>(vec2(bx, by) * WET_BLOCK + vec2(x, y));
                    deepest = max(deepest, textureLoad(lod_foam, texel, slice, 0).g);
                }
            }
            wet[base + by * blocks + bx] = deepest;
        }
    }
    var below = base;
    for (var level = 1u; level < wet_levels(); level++) {
        storageBarrier();
        workgroupBarrier();
        let n = blocks >> level;
        let width_below = blocks >> (level - 1u);
        let here = below + width_below * width_below;
        for (var y = local.y; y < n; y += WET_GROUP) {
            for (var x = local.x; x < n; x += WET_GROUP) {
                let i = below + 2u * y * width_below + 2u * x;
                wet[here + y * n + x] = max(max(wet[i], wet[i + 1u]), max(wet[i + width_below], wet[i + width_below + 1u]));
            }
        }
        below = here;
    }
}

/// Whether any water at least `params.heights.z` deep lies under the node, widened by the
/// horizontal margin. Tested on the finest slice that holds the widened node, at the pyramid
/// level whose cells are at least half its width, so at most 3x3 cells.
fn has_water(minimum: vec2<f32>, size: f32) -> bool {
    let dry = params.heights.z;
    if dry <= 0.0 { return true; }
    let lo = minimum - params.shape.w;
    let hi = minimum + size + params.shape.w;
    for (var i = 0u; i < lod.count; i++) {
        let s = lod.slices[i];
        let corner = s.xy - 0.5 * s.w;
        if any(lo < corner) || any(hi > corner + s.w) { continue; }
        var level = 0u;
        var cell = s.z * f32(WET_BLOCK);
        while level + 1u < wet_levels() && 2.0 * cell < hi.x - lo.x {
            level += 1u;
            cell *= 2.0;
        }
        let n = i32(wet_blocks() >> level);
        let offset = wet_level_offset(i, level);
        let c0 = clamp(vec2<i32>(floor((lo - corner) / cell)), vec2(0), vec2(n - 1));
        let c1 = clamp(vec2<i32>(floor((hi - corner) / cell)), vec2(0), vec2(n - 1));
        for (var y = c0.y; y <= c1.y; y++) {
            for (var x = c0.x; x <= c1.x; x++) {
                if wet[offset + u32(y * n + x)] >= dry { return true; }
            }
        }
        return false;
    }
    return true;
}

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
        if !visible(minimum, size) || !has_water(minimum, size) { continue; }
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
