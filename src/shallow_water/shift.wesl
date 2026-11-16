// Moves a shallow water grid over the world by whole cells. The grid's fields were copied into the
// source textures first, so no thread reads a cell another thread overwrites. Cells that come in
// take their bed from a world bed texture and their water from the parent grid's surface, or from
// a still level.

struct ShiftSettings {
    cells : vec2u,
    /// The new cell `i` takes the old cell `i + shift`.
    shift : vec2i,
    /// World XZ of the grid's corner after the shift.
    origin : vec2f,
    cell_size : f32,
    /// Bed samples per cell along each axis. A parent takes 3 so its bed is the mean of its
    /// child's beds over the same ground.
    subsamples : u32,
    /// World XZ of the bed texture's corner and its texel width.
    bed_origin : vec2f,
    bed_texel : f32,
    /// Still water level for cells that come in without a parent.
    fill_level : f32,
    parent_cells : vec2u,
    /// Parent cell under this grid's cell (0, 0) after the shift.
    parent_offset : vec2i,
    has_parent : u32,
    eps_wet : f32,
}

@group(0) @binding(0) var bedWater : texture_storage_2d<rgba32float, read_write>;
@group(0) @binding(1) var faceU : texture_storage_2d<r32float, read_write>;
@group(0) @binding(2) var faceW : texture_storage_2d<r32float, read_write>;

@group(1) @binding(0) var<uniform> shift : ShiftSettings;

@group(2) @binding(0) var oldBedWater : texture_2d<f32>;
@group(2) @binding(1) var oldFaceU : texture_2d<f32>;
@group(2) @binding(2) var oldFaceW : texture_2d<f32>;
/// World bed height, one texel per `bed_texel` meters.
@group(2) @binding(3) var worldBed : texture_2d<f32>;
@group(2) @binding(4) var parentBedWater : texture_2d<f32>;

fn world_bed_texel(t : vec2i) -> f32 {
    let size = vec2i(textureDimensions(worldBed));
    return textureLoad(worldBed, clamp(t, vec2i(0), size - 1), 0).r;
}

/// Bilinear world bed at world XZ `p`, clamped at the texture's edge.
fn world_bed_at(p : vec2f) -> f32 {
    let f = (p - shift.bed_origin) / shift.bed_texel - 0.5;
    let i = vec2i(floor(f));
    let t = f - vec2f(i);
    let a = mix(world_bed_texel(i), world_bed_texel(i + vec2i(1, 0)), t.x);
    let b = mix(world_bed_texel(i + vec2i(0, 1)), world_bed_texel(i + vec2i(1, 1)), t.x);
    return mix(a, b, t.y);
}

/// Mean world bed over cell `c`, from `subsamples`² points.
fn cell_bed(c : vec2i) -> f32 {
    let s = max(shift.subsamples, 1u);
    var sum = 0.0;
    for (var k = 0u; k < s * s; k++) {
        let sub = (vec2f(f32(k % s), f32(k / s)) + 0.5) / f32(s);
        sum += world_bed_at(shift.origin + (vec2f(c) + sub) * shift.cell_size);
    }
    return sum / f32(s * s);
}

/// Parent surface at `x` in parent cell units, bilinear over wet cells. `.y` is the wet weight.
fn parent_eta_at(x : vec2f) -> vec2f {
    let f = x - 0.5;
    let i = vec2i(floor(f));
    let t = f - vec2f(i);
    var eta = 0.0;
    var weight = 0.0;
    let n = vec2i(shift.parent_cells);
    for (var k = 0; k < 4; k++) {
        let d = vec2i(k & 1, k >> 1);
        let w = mix(1.0 - t.x, t.x, f32(d.x)) * mix(1.0 - t.y, t.y, f32(d.y));
        let hw = textureLoad(parentBedWater, vec2u(clamp(i + d, vec2i(0), n - 1)), 0).xy;
        let wet = select(0.0, 1.0, hw.y > shift.eps_wet);
        eta += w * wet * (hw.x + hw.y);
        weight += w * wet;
    }
    return vec2f(eta / max(weight, 1e-6), weight);
}

/// Water depth for a cell that comes in over bed `bed`.
fn fill_depth(c : vec2i, bed : f32) -> f32 {
    if (shift.has_parent != 0u) {
        let eta = parent_eta_at(vec2f(shift.parent_offset) + (vec2f(c) + 0.5) / 3.0);
        return select(0.0, max(0.0, eta.x - bed), eta.y > 0.0);
    }
    return max(0.0, shift.fill_level - bed);
}

@compute @workgroup_size(16, 16)
fn shiftGrid(@builtin(global_invocation_id) id : vec3u) {
    let n = vec2i(shift.cells);
    let c = vec2i(id.xy);
    let s = c + shift.shift;
    if (all(c < n)) {
        if (all(s >= vec2i(0)) && all(s < n)) {
            textureStore(bedWater, id.xy, textureLoad(oldBedWater, vec2u(s), 0));
        } else {
            let bed = cell_bed(c);
            textureStore(bedWater, id.xy, vec4f(bed, fill_depth(c, bed), 0.0, 0.0));
        }
    }
    if (c.x <= n.x && c.y < n.y) {
        var u = 0.0;
        if (s.x >= 0 && s.x <= n.x && s.y >= 0 && s.y < n.y) {
            u = textureLoad(oldFaceU, vec2u(s), 0).r;
        }
        textureStore(faceU, id.xy, vec4f(u, 0.0, 0.0, 0.0));
    }
    if (c.x < n.x && c.y <= n.y) {
        var w = 0.0;
        if (s.x >= 0 && s.x < n.x && s.y >= 0 && s.y <= n.y) {
            w = textureLoad(oldFaceW, vec2u(s), 0).r;
        }
        textureStore(faceW, id.xy, vec4f(w, 0.0, 0.0, 0.0));
    }
}
