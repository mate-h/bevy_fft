// Writes the dispersive patch into every water LOD slice that overlaps it.
//
// In add mode the patch's deviation from rest goes on top of the data below, so ocean waves run
// through it and a wake rides on them. In replace mode the patch takes over the height and the
// water depth lane. Flow follows the same mode, with velocity from the solver's flux and the
// vertical lane from mass conservation. Drift takes the solver's velocity in replace mode. In add
// mode the patch carries only waves on top of the water below, so drift stays as it was. The
// foam lane gets the breaking intensity of CMF10 fronts in the patch and of the water a moving hull
// pushes aside. Both modes fade out along the patch edge over `blend_cells`. The data below comes
// from a scratch copy made just before this pass.

import bevy_fft::water::breaking::{breaking_front, max_abs};
import bevy_fft::water::lod::{WaterLodSlices, lod_texel_world};

const BLEND_REPLACE: u32 = 1u;

struct DispersiveLodParams {
    /// World X and Z of the patch centre in meters.
    center: vec2<f32>,
    cell_size: f32,
    /// Height above the surface mesh plane of the solver's rest surface, in meters.
    datum: f32,
    cells: u32,
    blend_cells: f32,
    blend: u32,
    _pad: u32,
    /// Hull position this frame and last frame, in cells.
    hull: vec2<f32>,
    hull_prev: vec2<f32>,
    /// Hull radius in cells.
    hull_radius: f32,
    /// Hull speed in meters per second, zero while no hull is in the water.
    hull_speed: f32,
    _pad1: f32,
    _pad2: f32,
}

@group(0) @binding(0) var<uniform> params: DispersiveLodParams;
@group(0) @binding(1) var<uniform> lod: WaterLodSlices;
@group(0) @binding(2) var displacement_out: texture_storage_2d_array<rgba16float, write>;
@group(0) @binding(3) var foam_out: texture_storage_2d_array<rgba16float, write>;
@group(0) @binding(4) var flow_out: texture_storage_2d_array<rgba32float, write>;
@group(0) @binding(5) var base_displacement: texture_2d_array<f32>;
@group(0) @binding(6) var base_foam: texture_2d_array<f32>;
@group(0) @binding(7) var base_flow: texture_2d_array<f32>;
/// Water depth in `.r` in meters, and volume flux per unit width along X and Z in `.gb` in m²/s.
@group(0) @binding(8) var state: texture_2d<f32>;
/// Bed height in `.r`, in meters relative to the rest surface.
@group(0) @binding(9) var bed: texture_2d<f32>;
@group(0) @binding(10) var drift_out: texture_storage_2d_array<rg32float, write>;
@group(0) @binding(11) var base_drift: texture_2d_array<f32>;

/// Depth below which velocity is taken from a floor depth, so flux over nearly dry cells does not
/// turn into runaway speeds.
const MIN_FLOW_DEPTH: f32 = 0.05;
/// Hull speed in m/s across which the water it pushes aside turns white.
const HULL_FOAM_SPEED_START: f32 = 0.5;
const HULL_FOAM_SPEED_END: f32 = 3.0;

/// Breaking intensity in the ring of water around the hull's outline, swept from last frame's
/// position to this one's.
fn hull_foam(g: vec2<f32>) -> f32 {
    if params.hull_speed <= 0.0 {
        return 0.0;
    }
    let seg = params.hull - params.hull_prev;
    let t = saturate(dot(g - params.hull_prev, seg) / max(dot(seg, seg), 1e-6));
    let d = length(g - (params.hull_prev + seg * t));
    let r = max(params.hull_radius, 1.0);
    let ring = smoothstep(0.3 * r, 0.8 * r, d) * (1.0 - smoothstep(0.8 * r, 1.4 * r, d));
    return ring * smoothstep(HULL_FOAM_SPEED_START, HULL_FOAM_SPEED_END, params.hull_speed);
}

/// Bilinear bed and depth at grid coordinate `g`, where cell `i` spans `i..i + 1`.
fn bed_depth_at(g: vec2<f32>) -> vec2<f32> {
    let hi = vec2<i32>(i32(params.cells) - 1);
    let x = g - 0.5;
    let i0 = vec2<i32>(floor(x));
    let f = x - floor(x);
    let c00 = clamp(i0, vec2(0), hi);
    let c10 = clamp(i0 + vec2(1, 0), vec2(0), hi);
    let c01 = clamp(i0 + vec2(0, 1), vec2(0), hi);
    let c11 = clamp(i0 + vec2(1, 1), vec2(0), hi);
    let a = vec2<f32>(textureLoad(bed, c00, 0).r, textureLoad(state, c00, 0).r);
    let b = vec2<f32>(textureLoad(bed, c10, 0).r, textureLoad(state, c10, 0).r);
    let c = vec2<f32>(textureLoad(bed, c01, 0).r, textureLoad(state, c01, 0).r);
    let d = vec2<f32>(textureLoad(bed, c11, 0).r, textureLoad(state, c11, 0).r);
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

/// Bilinear depth and flux at grid coordinate `g`.
fn state_at(g: vec2<f32>) -> vec3<f32> {
    let hi = vec2<i32>(i32(params.cells) - 1);
    let x = g - 0.5;
    let i0 = vec2<i32>(floor(x));
    let f = x - floor(x);
    let a = textureLoad(state, clamp(i0, vec2(0), hi), 0).rgb;
    let b = textureLoad(state, clamp(i0 + vec2(1, 0), vec2(0), hi), 0).rgb;
    let c = textureLoad(state, clamp(i0 + vec2(0, 1), vec2(0), hi), 0).rgb;
    let d = textureLoad(state, clamp(i0 + vec2(1, 1), vec2(0), hi), 0).rgb;
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

fn eta_at(g: vec2<f32>) -> f32 {
    let bd = bed_depth_at(g);
    return bd.x + bd.y;
}

@compute @workgroup_size(8, 8, 1)
fn dispersive_lod_write(@builtin(global_invocation_id) gid: vec3<u32>) {
    let res = lod.resolution;
    let slice = gid.z;
    if gid.x >= res || gid.y >= res || slice >= lod.count {
        return;
    }
    let xz = lod_texel_world(lod, gid.xy, slice);
    let n = f32(params.cells);
    let g = (xz - params.center) / params.cell_size + 0.5 * n;
    if any(g < vec2(0.0)) || any(g > vec2(n)) {
        return;
    }

    let edge = min(min(g.x, n - g.x), min(g.y, n - g.y));
    let w = smoothstep(0.0, max(params.blend_cells, 1e-3), edge);

    // Difference over at least one texel so coarse slices see the slope they can resolve.
    let span = max(lod.slices[slice].z / params.cell_size, 1.0);
    let dx = vec2<f32>(span, 0.0);
    let dz = vec2<f32>(0.0, span);
    let inv = 1.0 / (2.0 * span * params.cell_size);
    let slope = vec2<f32>(eta_at(g + dx) - eta_at(g - dx), eta_at(g + dz) - eta_at(g - dz)) * inv;
    let bd = bed_depth_at(g);
    let eta = bd.x + bd.y;
    let st = state_at(g);
    let rise = -((state_at(g + dx).y - state_at(g - dx).y) + (state_at(g + dz).z - state_at(g - dz).z)) * inv;
    let flow = vec4<f32>(st.yz / max(st.x, MIN_FLOW_DEPTH), rise, 0.0);
    let base_v = textureLoad(base_flow, gid.xy, slice, 0);

    let base = textureLoad(base_displacement, gid.xy, slice, 0);
    let base_f = textureLoad(base_foam, gid.xy, slice, 0);
    let h = span * params.cell_size;
    let ex = vec2<f32>(eta_at(g + dx), eta_at(g - dx));
    let ez = vec2<f32>(eta_at(g + dz), eta_at(g - dz));
    let steepest = length(vec2<f32>(max_abs(ex.x - eta, eta - ex.y), max_abs(ez.x - eta, eta - ez.y))) / h;
    let curvature = (ex.x + ex.y + ez.x + ez.y - 4.0 * eta) / (h * h);
    let breaking = max(breaking_front(steepest, rise, curvature), hull_foam(g)) * w;
    if params.blend == BLEND_REPLACE {
        let surface = vec4<f32>(slope, params.datum + eta, 0.0);
        textureStore(displacement_out, gid.xy, slice, mix(base, surface, w));
        let foam = vec4<f32>(
            base_f.r * (1.0 - w) + breaking,
            mix(base_f.g, bd.y, w),
            base_f.b * (1.0 - w),
            base_f.a,
        );
        textureStore(foam_out, gid.xy, slice, foam);
        textureStore(flow_out, gid.xy, slice, mix(base_v, flow, w));
        let base_d = textureLoad(base_drift, gid.xy, slice, 0).xy;
        textureStore(drift_out, gid.xy, slice, vec4<f32>(mix(base_d, flow.xy, w), 0.0, 0.0));
    } else {
        let deviation = vec4<f32>(slope, params.datum + eta, 0.0) * w;
        textureStore(displacement_out, gid.xy, slice, base + deviation);
        textureStore(foam_out, gid.xy, slice, vec4<f32>(max(base_f.r, breaking), base_f.gba));
        textureStore(flow_out, gid.xy, slice, base_v + flow * w);
    }
}
