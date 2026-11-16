// Two-way nesting of a fine shallow water grid inside a coarse one at refinement 3 (Debreu and
// Blayo 2008; Debreu, Marchesiello, Penven and Cambon 2012). Each kernel writes the grid bound in
// group 0, the target, and reads the other grid, the source, from group 2. Prolongation targets
// the child; the snapshot and the feedback target the parent.

struct NestSettings {
    /// Child grid size in cells, a multiple of 3 on each axis.
    child_cells : vec2u,
    parent_cells : vec2u,
    /// Parent cell under the child's cell (0, 0).
    offset : vec2i,
    /// Width of the child's dynamic band in parent cells. The band is three times as wide in
    /// child cells.
    band : u32,
    /// Where in the parent's step the prolonged values sit, 0 at its start and 1 at its end.
    time_blend : f32,
    /// Parent cell width.
    parent_dx : f32,
    /// Depth under which a parent cell counts as dry for prolongation.
    eps_wet : f32,
    _pad : vec2f,
}

const RHO = 3;

@group(0) @binding(0) var bedWater : texture_storage_2d<rgba32float, read_write>;
@group(0) @binding(1) var faceU : texture_storage_2d<r32float, read_write>;
@group(0) @binding(2) var faceW : texture_storage_2d<r32float, read_write>;
@group(0) @binding(3) var prevBedWater : texture_storage_2d<rgba32float, read_write>;
@group(0) @binding(4) var prevFaceU : texture_storage_2d<r32float, read_write>;
@group(0) @binding(5) var prevFaceW : texture_storage_2d<r32float, read_write>;

@group(1) @binding(0) var<uniform> nest : NestSettings;

@group(2) @binding(0) var srcBedWater : texture_2d<f32>;
@group(2) @binding(1) var srcFaceU : texture_2d<f32>;
@group(2) @binding(2) var srcFaceW : texture_2d<f32>;
@group(2) @binding(3) var srcPrevBedWater : texture_2d<f32>;
@group(2) @binding(4) var srcPrevFaceU : texture_2d<f32>;
@group(2) @binding(5) var srcPrevFaceW : texture_2d<f32>;
/// The source's accumulated face volumes in `.z`.
@group(2) @binding(6) var srcMacU : texture_2d<f32>;
@group(2) @binding(7) var srcMacW : texture_2d<f32>;
/// The target's own accumulated face volumes in `.z`.
@group(2) @binding(8) var ownMacU : texture_2d<f32>;
@group(2) @binding(9) var ownMacW : texture_2d<f32>;

fn child_n() -> vec2i { return vec2i(nest.child_cells); }
fn parent_n() -> vec2i { return vec2i(nest.parent_cells); }
/// The child's extent in parent cells.
fn covered() -> vec2i { return child_n() / RHO; }

// --- Snapshot (target: parent) ------------------------------------------------------------------

/// Copies the parent's depth and face velocities before its step, so prolongation can blend
/// between the start and the end of that step.
@compute @workgroup_size(16, 16)
fn snapshotParent(@builtin(global_invocation_id) id : vec3u) {
    let n = nest.parent_cells;
    if (id.x < n.x && id.y < n.y) {
        textureStore(prevBedWater, id.xy, textureLoad(bedWater, id.xy));
    }
    if (id.x <= n.x && id.y < n.y) {
        textureStore(prevFaceU, id.xy, textureLoad(faceU, id.xy));
    }
    if (id.x < n.x && id.y <= n.y) {
        textureStore(prevFaceW, id.xy, textureLoad(faceW, id.xy));
    }
}

// --- Prolongation (target: child, source: parent) -----------------------------------------------

fn parent_cell(p : vec2i) -> vec2u {
    return vec2u(clamp(p, vec2i(0), parent_n() - 1));
}

/// Parent surface at time `nest.time_blend` and whether the parent cell is wet.
fn parent_eta_wet(p : vec2i) -> vec2f {
    let c = parent_cell(p);
    let now = textureLoad(srcBedWater, c, 0).xy;
    let before = textureLoad(srcPrevBedWater, c, 0).xy;
    let hw = mix(before, now, nest.time_blend);
    return vec2f(hw.x + hw.y, select(0.0, 1.0, hw.y > nest.eps_wet));
}

/// Bilinear parent surface at `x` in parent cell units, over wet cells only. `.y` is the weight of
/// the wet cells, zero where all four are dry.
fn parent_eta_at(x : vec2f) -> vec2f {
    let f = x - 0.5;
    let i = vec2i(floor(f));
    let t = f - vec2f(i);
    var eta = 0.0;
    var weight = 0.0;
    for (var k = 0; k < 4; k++) {
        let d = vec2i(k & 1, k >> 1);
        let w = mix(1.0 - t.x, t.x, f32(d.x)) * mix(1.0 - t.y, t.y, f32(d.y));
        let s = parent_eta_wet(i + d);
        eta += w * s.y * s.x;
        weight += w * s.y;
    }
    return vec2f(eta / max(weight, 1e-6), weight);
}

fn parent_u(f : vec2i) -> f32 {
    let c = vec2u(clamp(f, vec2i(0), parent_n() + vec2i(0, -1)));
    return mix(textureLoad(srcPrevFaceU, c, 0).r, textureLoad(srcFaceU, c, 0).r, nest.time_blend);
}

fn parent_w(f : vec2i) -> f32 {
    let c = vec2u(clamp(f, vec2i(0), parent_n() + vec2i(-1, 0)));
    return mix(textureLoad(srcPrevFaceW, c, 0).r, textureLoad(srcFaceW, c, 0).r, nest.time_blend);
}

/// Bilinear parent u at `x` in parent cell units. u faces sit at integer x and half-integer y.
fn parent_u_at(x : vec2f) -> f32 {
    let f = x - vec2f(0.0, 0.5);
    let i = vec2i(floor(f));
    let t = f - vec2f(i);
    let a = mix(parent_u(i), parent_u(i + vec2i(1, 0)), t.x);
    let b = mix(parent_u(i + vec2i(0, 1)), parent_u(i + vec2i(1, 1)), t.x);
    return mix(a, b, t.y);
}

/// Bilinear parent w at `x`. w faces sit at half-integer x and integer y.
fn parent_w_at(x : vec2f) -> f32 {
    let f = x - vec2f(0.5, 0.0);
    let i = vec2i(floor(f));
    let t = f - vec2f(i);
    let a = mix(parent_w(i), parent_w(i + vec2i(1, 0)), t.x);
    let b = mix(parent_w(i + vec2i(0, 1)), parent_w(i + vec2i(1, 1)), t.x);
    return mix(a, b, t.y);
}

/// A point in child cell units, in parent cell units.
fn to_parent(x : vec2f) -> vec2f {
    return vec2f(nest.offset) + x / f32(RHO);
}

/// Whether child cell `c` is in the dynamic band along the child's edge.
fn in_band(c : vec2i) -> bool {
    let b = i32(nest.band) * RHO;
    return any(c < vec2i(b)) || any(c >= child_n() - b);
}

/// Sets the child's dynamic band to the parent's state at `time_blend`: the surface, over the
/// child's own bed, and the face velocities. Faces on the band's inner edge keep the child's
/// value, so the fluxes through the interface are the child's own.
@compute @workgroup_size(16, 16)
fn prolongBand(@builtin(global_invocation_id) id : vec3u) {
    let n = child_n();
    let c = vec2i(id.xy);
    let b = i32(nest.band) * RHO;
    if (c.x < n.x && c.y < n.y && in_band(c)) {
        let eta = parent_eta_at(to_parent(vec2f(c) + 0.5));
        var hw = textureLoad(bedWater, id.xy);
        hw.y = select(0.0, max(0.0, eta.x - hw.x), eta.y > 0.0);
        textureStore(bedWater, id.xy, hw);
    }
    // u face (fi, j) lies between cells fi - 1 and fi. It is in the band when both are, or when
    // it is on the child's edge.
    if (c.x <= n.x && c.y < n.y) {
        let both = in_band(c - vec2i(1, 0)) && in_band(c);
        let edge = c.x == 0 || c.x == n.x;
        let row_in = c.y >= b && c.y < n.y - b;
        let inner = (c.x == b || c.x == n.x - b) && row_in;
        if ((both || edge) && !inner) {
            let u = parent_u_at(to_parent(vec2f(f32(c.x), f32(c.y) + 0.5)));
            textureStore(faceU, id.xy, vec4f(u, 0.0, 0.0, 0.0));
        }
    }
    if (c.x < n.x && c.y <= n.y) {
        let both = in_band(c - vec2i(0, 1)) && in_band(c);
        let edge = c.y == 0 || c.y == n.y;
        let col_in = c.x >= b && c.x < n.x - b;
        let inner = (c.y == b || c.y == n.y - b) && col_in;
        if ((both || edge) && !inner) {
            let w = parent_w_at(to_parent(vec2f(f32(c.x) + 0.5, f32(c.y))));
            textureStore(faceW, id.xy, vec4f(w, 0.0, 0.0, 0.0));
        }
    }
}

// --- Feedback (target: parent, source: child) ---------------------------------------------------

/// The parent cell in child-covered coordinates, 0 at the child's first parent cell.
fn local(p : vec2i) -> vec2i { return p - nest.offset; }

/// Whether parent cell `p` lies over the child's interior, inside the dynamic band.
fn is_covered(p : vec2i) -> bool {
    let l = local(p);
    let b = i32(nest.band);
    return all(l >= vec2i(b)) && all(l < covered() - b);
}

fn child_hw(c : vec2i) -> vec2f {
    return textureLoad(srcBedWater, vec2u(c), 0).xy;
}

/// Volume per unit width through child u face `(fi, j)` this frame.
fn child_volume_u(fi : i32, j : i32) -> f32 {
    return textureLoad(srcMacU, vec2u(vec2i(fi, j)), 0).z;
}

fn child_volume_w(i : i32, fj : i32) -> f32 {
    return textureLoad(srcMacW, vec2u(vec2i(i, fj)), 0).z;
}

/// Parent depth change that swaps the parent's volume through its u face `fi` in row `j` for the
/// child's three faces on that line. `side` is +1 for the cell west of the face and -1 east.
fn reflux_u(fi : i32, j : i32, side : f32) -> f32 {
    let lc = vec2i(fi, j) - nest.offset;
    var fine = 0.0;
    for (var k = 0; k < RHO; k++) {
        fine += child_volume_u(lc.x * RHO, lc.y * RHO + k);
    }
    let coarse = textureLoad(ownMacU, vec2u(vec2i(fi, j)), 0).z;
    // Fine volumes are per child face width, a third of the parent's.
    return side * (coarse - fine / f32(RHO)) / nest.parent_dx;
}

fn reflux_w(i : i32, fj : i32, side : f32) -> f32 {
    let lc = vec2i(i, fj) - nest.offset;
    var fine = 0.0;
    for (var k = 0; k < RHO; k++) {
        fine += child_volume_w(lc.x * RHO + k, lc.y * RHO);
    }
    let coarse = textureLoad(ownMacW, vec2u(vec2i(i, fj)), 0).z;
    return side * (coarse - fine / f32(RHO)) / nest.parent_dx;
}

/// Feeds the child back into the parent. Parent cells over the child's interior take the mean
/// bed and depth of their nine child cells, and the faces between them the mean of their three child
/// faces. Parent cells beside that region swap their flux through the shared face for the
/// child's (Berger and Colella 1989), so the parent conserves the water the child moves.
@compute @workgroup_size(16, 16)
fn feedback(@builtin(global_invocation_id) id : vec3u) {
    let n = parent_n();
    let p = vec2i(id.xy);
    let b = i32(nest.band);
    let m = covered();
    if (p.x < n.x && p.y < n.y) {
        var hw = textureLoad(bedWater, id.xy);
        let l = local(p);
        if (is_covered(p)) {
            var hw_child = vec2f(0.0);
            for (var k = 0; k < RHO * RHO; k++) {
                hw_child += child_hw(l * RHO + vec2i(k % RHO, k / RHO));
            }
            // The bed too, so the parent's surface is the child's mean surface even where the
            // child moved over ground the parent sampled differently.
            hw = vec4f(hw_child / f32(RHO * RHO), hw.zw);
        } else {
            let rows = l.y >= b && l.y < m.y - b;
            let cols = l.x >= b && l.x < m.x - b;
            var dh = 0.0;
            if (rows && l.x == b - 1) { dh += reflux_u(p.x + 1, p.y, 1.0); }
            if (rows && l.x == m.x - b) { dh += reflux_u(p.x, p.y, -1.0); }
            if (cols && l.y == b - 1) { dh += reflux_w(p.x, p.y + 1, 1.0); }
            if (cols && l.y == m.y - b) { dh += reflux_w(p.x, p.y, -1.0); }
            hw.y = max(0.0, hw.y + dh);
        }
        textureStore(bedWater, id.xy, hw);
    }
    // Faces strictly inside the covered region.
    if (p.x <= n.x && p.y < n.y) {
        let l = local(p);
        if (l.x > b && l.x < m.x - b && l.y >= b && l.y < m.y - b) {
            var u = 0.0;
            for (var k = 0; k < RHO; k++) {
                u += textureLoad(srcFaceU, vec2u(vec2i(l.x * RHO, l.y * RHO + k)), 0).r;
            }
            textureStore(faceU, id.xy, vec4f(u / f32(RHO), 0.0, 0.0, 0.0));
        }
    }
    if (p.x < n.x && p.y <= n.y) {
        let l = local(p);
        if (l.y > b && l.y < m.y - b && l.x >= b && l.x < m.x - b) {
            var w = 0.0;
            for (var k = 0; k < RHO; k++) {
                w += textureLoad(srcFaceW, vec2u(vec2i(l.x * RHO + k, l.y * RHO)), 0).r;
            }
            textureStore(faceW, id.xy, vec4f(w / f32(RHO), 0.0, 0.0, 0.0));
        }
    }
}
